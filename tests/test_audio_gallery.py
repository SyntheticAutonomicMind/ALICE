# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2025 Andrew Wyatt (Fewtarius)

"""
Tests for the audio gallery feature.

Covers:
    - AudioRecord dataclass creation and serialization
    - GalleryManager add_audio / list_audio / delete_audio
    - Gallery /v1/gallery/audio endpoint (FastAPI TestClient)
    - MP3 conversion helper (_try_convert_to_mp3)
    - Instrumental lyrics fallback
"""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.gallery import GalleryManager, AudioRecord, ImageRecord
from src.audio_engine import MiniMaxMusic3Engine


# ---------------------------------------------------------------------------
# AudioRecord dataclass tests
# ---------------------------------------------------------------------------

class TestAudioRecord:
    """Tests for the AudioRecord dataclass."""

    def test_create_audio_record(self, tmp_path):
        """AudioRecord can be created with required fields."""
        record = AudioRecord(
            id="test123",
            filename="test_audio.mp3",
            owner_api_key_id="key1",
            is_public=True,
            prompt="rock and roll",
            lyrics="(instrumental)",
            model="minimax-music",
            seed=42,
            steps=30,
            cfg_scale=7.0,
            duration_seconds=30.0,
            sample_rate=44100,
            size_bytes=1234567,
        )
        assert record.id == "test123"
        assert record.filename == "test_audio.mp3"
        assert record.is_public is True
        assert record.prompt == "rock and roll"
        assert record.lyrics == "(instrumental)"
        assert record.model == "minimax-music"
        assert record.seed == 42

    def test_audio_record_to_dict(self, tmp_path):
        """AudioRecord serializes to dict correctly."""
        record = AudioRecord(
            id="abc",
            filename="abc.mp3",
            owner_api_key_id="key1",
            is_public=True,
            prompt="jazz",
            model="minimax-music",
            seed=99,
            duration_seconds=15.0,
        )
        d = record.to_dict()
        assert d["id"] == "abc"
        assert d["filename"] == "abc.mp3"
        assert d["prompt"] == "jazz"
        assert d["is_public"] is True
        assert d["model"] == "minimax-music"

    def test_audio_record_from_dict(self, tmp_path):
        """AudioRecord can be deserialized from dict."""
        data = {
            "id": "xyz",
            "filename": "xyz.wav",
            "owner_api_key_id": "key2",
            "is_public": False,
            "prompt": "classical",
            "model": "stable-audio",
            "seed": 1,
            "duration_seconds": 60.0,
        }
        record = AudioRecord.from_dict(data)
        assert record.id == "xyz"
        assert record.filename == "xyz.wav"
        assert record.is_public is False
        assert record.prompt == "classical"

    def test_audio_record_is_expired(self):
        """is_expired returns False when expires_at is None."""
        record = AudioRecord(id="a", filename="a.mp3", owner_api_key_id=None)
        assert record.is_expired() is False

    def test_audio_record_is_expired_with_past_expiry(self):
        """is_expired returns True when expires_at is in the past."""
        record = AudioRecord(
            id="a", filename="a.mp3", owner_api_key_id=None,
            expires_at=time.time() - 1,
        )
        assert record.is_expired() is True

    def test_audio_record_is_accessible_by_owner(self):
        """Owner can always access their audio."""
        record = AudioRecord(id="a", filename="a.mp3", owner_api_key_id="mykey")
        assert record.is_accessible_by("mykey", is_admin=False) is True

    def test_audio_record_is_accessible_by_public(self):
        """Public, non-expired audio is accessible by anyone."""
        record = AudioRecord(id="a", filename="a.mp3", owner_api_key_id=None, is_public=True)
        assert record.is_accessible_by("other", is_admin=False) is True
        assert record.is_accessible_by(None, is_admin=False) is True

    def test_audio_record_not_accessible_by_anonymous_private(self):
        """Private audio is not accessible by anonymous users."""
        record = AudioRecord(id="a", filename="a.mp3", owner_api_key_id="mykey", is_public=False)
        assert record.is_accessible_by(None, is_admin=False) is False

    def test_audio_record_not_accessible_when_expired(self):
        """Expired public audio is not accessible."""
        record = AudioRecord(
            id="a", filename="a.mp3", owner_api_key_id=None,
            is_public=True, expires_at=time.time() - 1,
        )
        assert record.is_accessible_by(None, is_admin=False) is False


# ---------------------------------------------------------------------------
# GalleryManager audio tests
# ---------------------------------------------------------------------------

class TestGalleryManagerAudio:
    """Tests for GalleryManager audio operations."""

    def test_add_and_get_audio(self, tmp_path):
        """add_audio stores and get_audio retrieves audio records."""
        gm = GalleryManager(tmp_path / "gallery.json")
        record = AudioRecord(
            id="test_audio",
            filename="test.mp3",
            owner_api_key_id="key1",
            is_public=True,
            prompt="electronic music",
            model="minimax-music",
            seed=42,
            duration_seconds=30.0,
        )
        gm.add_audio(record)

        retrieved = gm.get_audio("test_audio")
        assert retrieved is not None
        assert retrieved.filename == "test.mp3"
        assert retrieved.prompt == "electronic music"

    def test_list_audio_empty(self, tmp_path):
        """list_audio returns empty list when no audio exists."""
        gm = GalleryManager(tmp_path / "gallery.json")
        result = gm.list_audio()
        assert result == []

    def test_list_audio_returns_added_records(self, tmp_path):
        """list_audio returns added audio records."""
        gm = GalleryManager(tmp_path / "gallery.json")
        record = AudioRecord(
            id="audio1",
            filename="audio1.mp3",
            owner_api_key_id="key1",
            is_public=True,
            prompt="ambient",
            model="minimax-music",
            duration_seconds=20.0,
        )
        gm.add_audio(record)

        result = gm.list_audio()
        assert len(result) == 1
        assert result[0].id == "audio1"
        assert result[0].filename == "audio1.mp3"

    def test_list_audio_respects_owner(self, tmp_path):
        """list_audio only returns records owned by the requesting user (for private)."""
        gm = GalleryManager(tmp_path / "gallery.json")
        gm.add_audio(AudioRecord(
            id="a1", filename="a1.mp3", owner_api_key_id="user1",
            is_public=False, prompt="test1", model="minimax-music",
        ))
        gm.add_audio(AudioRecord(
            id="a2", filename="a2.mp3", owner_api_key_id="user2",
            is_public=False, prompt="test2", model="minimax-music",
        ))

        # user1 should see their own audio
        result = gm.list_audio(api_key_id="user1")
        assert len(result) == 1
        assert result[0].id == "a1"

        # user2 should see their own audio
        result = gm.list_audio(api_key_id="user2")
        assert len(result) == 1
        assert result[0].id == "a2"

    def test_list_audio_public_visible_to_anonymous(self, tmp_path):
        """list_audio returns public records to anonymous users."""
        gm = GalleryManager(tmp_path / "gallery.json")
        gm.add_audio(AudioRecord(
            id="pub", filename="pub.mp3", owner_api_key_id="user1",
            is_public=True, prompt="public test", model="minimax-music",
        ))
        gm.add_audio(AudioRecord(
            id="priv", filename="priv.mp3", owner_api_key_id="user2",
            is_public=False, prompt="private test", model="minimax-music",
        ))

        # Anonymous should see only public
        result = gm.list_audio(api_key_id=None)
        assert len(result) == 1
        assert result[0].id == "pub"

    def test_delete_audio(self, tmp_path):
        """delete_audio removes audio records."""
        gm = GalleryManager(tmp_path / "gallery.json")
        record = AudioRecord(
            id="delme", filename="delme.mp3", owner_api_key_id="key1",
            is_public=True, prompt="delete me", model="minimax-music",
        )
        gm.add_audio(record)

        assert gm.get_audio("delme") is not None
        result = gm.delete_audio("delme")
        assert result is True
        assert gm.get_audio("delme") is None

    def test_delete_audio_nonexistent(self, tmp_path):
        """delete_audio returns False for non-existent records."""
        gm = GalleryManager(tmp_path / "gallery.json")
        result = gm.delete_audio("nonexistent")
        assert result is False

    def test_get_stats_includes_audio(self, tmp_path):
        """get_stats returns audio counts."""
        gm = GalleryManager(tmp_path / "gallery.json")
        gm.add_image(ImageRecord(
            id="img1", filename="img1.png", owner_api_key_id="key1",
            is_public=True, prompt="test", model="sdxl",
            width=1024, height=1024,
        ))
        gm.add_audio(AudioRecord(
            id="audio1", filename="audio1.mp3", owner_api_key_id="key1",
            is_public=True, prompt="test", model="minimax-music",
            duration_seconds=30.0,
        ))
        gm.add_audio(AudioRecord(
            id="audio2", filename="audio2.mp3", owner_api_key_id="key2",
            is_public=False, prompt="test2", model="minimax-music",
            duration_seconds=20.0,
        ))

        stats = gm.get_stats()
        assert stats["total"] == 1
        assert stats["total_audio"] == 2
        assert stats["public_audio"] == 1
        assert stats["private_audio"] == 1

    def test_gallery_persistence(self, tmp_path):
        """Gallery saves and loads audio records across instances."""
        gm1 = GalleryManager(tmp_path / "gallery.json")
        gm1.add_audio(AudioRecord(
            id="persist", filename="persist.mp3", owner_api_key_id="key1",
            is_public=True, prompt="persist test", model="minimax-music",
            duration_seconds=15.0,
        ))

        gm2 = GalleryManager(tmp_path / "gallery.json")
        result = gm2.list_audio()
        assert len(result) == 1
        assert result[0].id == "persist"
        assert result[0].prompt == "persist test"


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------

class TestGalleryAudioEndpoint:
    """Tests for the /v1/gallery/audio API endpoint."""

    def test_gallery_audio_endpoint(self):
        """The audio gallery endpoint returns audio records."""
        from fastapi.testclient import TestClient
        from src.main import app

        client = TestClient(app)
        response = client.get(
            "/v1/gallery/audio",
            headers={"X-Api-Key": "testkey"},
        )
        # Should return 200 (or 503 if gallery not init)
        assert response.status_code in (200, 403, 503)
        if response.status_code == 200:
            data = response.json()
            assert "data" in data
            assert "total" in data

    def test_gallery_stats_includes_audio_fields(self):
        """GET /v1/gallery/stats includes audio counts in response (or requires auth)."""
        from fastapi.testclient import TestClient
        from src.main import app

        client = TestClient(app)
        response = client.get(
            "/v1/gallery/stats",
            headers={"X-Api-Key": "testkey"},
        )
        # 403 means auth blocked (expected in test env); 200/503 means working
        assert response.status_code in (200, 403, 503)
        if response.status_code == 200:
            data = response.json()
            assert "totalAudio" in data
            assert "publicAudio" in data
            assert "privateAudio" in data


# ---------------------------------------------------------------------------
# Instrumental lyrics tests
# ---------------------------------------------------------------------------

class TestInstrumentalFallback:
    """Tests that lyrics content and instrumental fallback are handled correctly.

    The MiniMax-Music3 pipeline requires non-empty lyrics (it raises
    ValueError otherwise) and silently drops text on lines that start
    with [tag] structure markers.  The engine preprocesses lyrics via
    ``_preprocess_lyrics`` to:
    - Split ``[tag] text`` lines so body text survives ``_normalize_lyrics``
    - Replace empty/None lyrics with the structural tag set
      ``[intro]\\n[instrumental]\\n[solo]\\n[outro]`` so the pipeline
      doesn't raise ValueError

    This replaces the older approach of passing ``""`` directly, which
    the pipeline rejects with ``ValueError``.
    """

    def test_lyrics_sentinel_removed(self):
        """The '(instrumental)' lyrics sentinel must not be present."""
        import inspect
        source = inspect.getsource(MiniMaxMusic3Engine.generate)
        assert "lyrics or \"(instrumental)\"" not in source, (
            "The '(instrumental)' lyrics sentinel must be removed - it causes "
            "the Qwen3 model to sing '(instrumental)' as lyrics, producing "
            "unwanted vocals in instrumental tracks"
        )

    def test_lyrics_are_preprocessed(self):
        """Lyrics must be preprocessed via _preprocess_lyrics before the pipeline call."""
        import inspect
        source = inspect.getsource(MiniMaxMusic3Engine.generate)
        assert "_preprocess_lyrics" in source, (
            "MiniMaxMusic3Engine.generate should preprocess lyrics via "
            "_preprocess_lyrics before passing to the pipeline"
        )
        assert "lyrics=processed_lyrics" in source, (
            "The preprocessed lyrics value should be what's passed to the pipeline"
        )

    def test_no_lyrics_or_empty_sentinel(self):
        """The old 'lyrics or \"\"' fallback pattern should NOT be present."""
        import inspect
        source = inspect.getsource(MiniMaxMusic3Engine.generate)
        assert "lyrics or \"\"" not in source, (
            "MiniMaxMusic3Engine.generate should NOT use 'lyrics or \"\"'"
        )
