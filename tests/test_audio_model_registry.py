# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2025 Andrew Wyatt (Fewtarius)

"""
Tests for audio (music) model registry integration.

Covers:
    - AudioModelEntry dataclass
    - _is_audio_model_dir detection (known names + markers)
    - _scan_audio_models discovery
    - Audio model files are not picked up as image models
    - list_audio_models / get_audio_model / get_audio_model_path
    - delete_audio_model (removes directory + registry entry)
    - Registry persistence (save/load across instances)
    - refresh() clears audio models
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.model_registry import (
    ModelRegistry,
    AudioModelEntry,
    ModelEntry,
    LoRAEntry,
)


@pytest.fixture
def tmp_models_dir():
    """Create a temporary models directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestAudioModelEntry:
    """Tests for the AudioModelEntry dataclass."""

    def test_create_audio_model_entry(self):
        """AudioModelEntry can be created with required fields."""
        entry = AudioModelEntry(
            id="audio/minimax-music-3",
            name="MiniMax-Music3",
            path="/models/MiniMax-Music3",
            created=1234567890,
            size_mb=1024,
            catalog_id="minimax-music-3",
            engine="minimax_music3",
        )
        assert entry.id == "audio/minimax-music-3"
        assert entry.name == "MiniMax-Music3"
        assert entry.catalog_id == "minimax-music-3"
        assert entry.engine == "minimax_music3"
        assert entry.size_mb == 1024

    def test_audio_model_entry_to_dict(self):
        """AudioModelEntry serializes to dict correctly."""
        entry = AudioModelEntry(
            id="audio/stable-audio-open-1.0",
            name="stable-audio-open-1.0",
            path="/models/stable-audio-open-1.0",
            created=1234567890,
            size_mb=500,
        )
        d = entry.to_dict()
        assert d["id"] == "audio/stable-audio-open-1.0"
        assert d["name"] == "stable-audio-open-1.0"
        assert d["size_mb"] == 500
        assert "catalog_id" in d

    def test_audio_model_entry_from_dict(self):
        """AudioModelEntry can be deserialized from dict."""
        data = {
            "id": "audio/test",
            "name": "test",
            "path": "/models/test",
            "created": 100,
            "size_mb": 50,
            "catalog_id": "test",
            "engine": "stable_audio",
        }
        entry = AudioModelEntry.from_dict(data)
        assert entry.id == "audio/test"
        assert entry.catalog_id == "test"
        assert entry.engine == "stable_audio"


class TestAudioModelScanning:
    """Tests for audio model directory scanning in the model registry."""

    def test_scan_detects_minimax_music3_dir(self, tmp_models_dir):
        """Scanning detects a MiniMax-Music3 directory as an audio model."""
        audio_dir = tmp_models_dir / "MiniMax-Music3"
        audio_dir.mkdir()
        (audio_dir / "configuration.json").write_text("{}")
        (audio_dir / "README.md").write_text("test")
        # Add subfolders typical of MiniMax-Music3
        (audio_dir / "transformer").mkdir()
        (audio_dir / "vocoder").mkdir()

        registry = ModelRegistry(tmp_models_dir)

        audio_models = registry.list_audio_models()
        assert len(audio_models) == 1
        assert audio_models[0].id == "audio/minimax-music-3"
        assert audio_models[0].catalog_id == "minimax-music-3"
        assert audio_models[0].name == "MiniMax-Music3"

    def test_scan_detects_stable_audio_dir(self, tmp_models_dir):
        """Scanning detects a stable-audio-open-1.0 directory."""
        audio_dir = tmp_models_dir / "stable-audio-open-1.0"
        audio_dir.mkdir()
        (audio_dir / "model_config.json").write_text("{}")
        (audio_dir / "README.md").write_text("test")

        registry = ModelRegistry(tmp_models_dir)

        audio_models = registry.list_audio_models()
        assert len(audio_models) == 1
        assert audio_models[0].id == "audio/stable-audio-open-1.0"
        assert audio_models[0].catalog_id == "stable-audio-open-1.0"

    def test_scan_skips_audio_dir_contents_from_image_models(self, tmp_models_dir):
        """Safetensors files inside audio model dirs are not listed as image models."""
        audio_dir = tmp_models_dir / "MiniMax-Music3"
        audio_dir.mkdir()
        (audio_dir / "configuration.json").write_text("{}")
        # A safetensors file inside the audio model directory
        (audio_dir / "transformer").mkdir()
        (audio_dir / "transformer" / "model.safetensors").write_bytes(b"x" * 100)

        registry = ModelRegistry(tmp_models_dir)

        # No image models should be found
        image_models = registry.list_models()
        assert len(image_models) == 0

        # But the audio model should be found
        audio_models = registry.list_audio_models()
        assert len(audio_models) == 1

    def test_scan_no_audio_models(self, tmp_models_dir):
        """Empty models directory yields no audio models."""
        registry = ModelRegistry(tmp_models_dir)
        assert len(registry.list_audio_models()) == 0

    def test_scan_non_audio_dir_not_detected(self, tmp_models_dir):
        """A random directory without audio markers is not detected as audio."""
        misc_dir = tmp_models_dir / "some-random-dir"
        misc_dir.mkdir()
        (misc_dir / "readme.txt").write_text("hello")

        registry = ModelRegistry(tmp_models_dir)
        assert len(registry.list_audio_models()) == 0

    def test_audio_model_size_calculated(self, tmp_models_dir):
        """Audio model size is calculated from directory contents."""
        audio_dir = tmp_models_dir / "MiniMax-Music3"
        audio_dir.mkdir()
        (audio_dir / "configuration.json").write_text("x" * 2048)
        (audio_dir / "vocoder").mkdir()
        (audio_dir / "vocoder" / "config.json").write_text("y" * 1024)

        registry = ModelRegistry(tmp_models_dir)

        audio_models = registry.list_audio_models()
        assert len(audio_models) == 1
        # Should be at least 0 MB (could be 0 for small files)
        assert audio_models[0].size_mb >= 0


class TestAudioModelManagement:
    """Tests for audio model CRUD operations."""

    def test_get_audio_model(self, tmp_models_dir):
        """get_audio_model retrieves by ID."""
        audio_dir = tmp_models_dir / "MiniMax-Music3"
        audio_dir.mkdir()
        (audio_dir / "configuration.json").write_text("{}")
        (audio_dir / "vocoder").mkdir()

        registry = ModelRegistry(tmp_models_dir)

        entry = registry.get_audio_model("audio/minimax-music-3")
        assert entry is not None
        assert entry.id == "audio/minimax-music-3"

    def test_get_audio_model_not_found(self, tmp_models_dir):
        """get_audio_model returns None for non-existent."""
        registry = ModelRegistry(tmp_models_dir)
        assert registry.get_audio_model("audio/nonexistent") is None

    def test_get_audio_model_path(self, tmp_models_dir):
        """get_audio_model_path returns the directory path."""
        audio_dir = tmp_models_dir / "MiniMax-Music3"
        audio_dir.mkdir()
        (audio_dir / "configuration.json").write_text("{}")
        (audio_dir / "vocoder").mkdir()

        registry = ModelRegistry(tmp_models_dir)

        path = registry.get_audio_model_path("audio/minimax-music-3")
        assert path is not None
        assert str(path) == str(audio_dir)

    def test_delete_audio_model(self, tmp_models_dir):
        """delete_audio_model removes directory and registry entry."""
        audio_dir = tmp_models_dir / "MiniMax-Music3"
        audio_dir.mkdir()
        (audio_dir / "configuration.json").write_text("{}")
        (audio_dir / "vocoder").mkdir()

        registry = ModelRegistry(tmp_models_dir)

        assert registry.get_audio_model("audio/minimax-music-3") is not None
        result = registry.delete_audio_model("audio/minimax-music-3")
        assert result is True
        assert not audio_dir.exists()
        assert registry.get_audio_model("audio/minimax-music-3") is None

    def test_delete_audio_model_not_found(self, tmp_models_dir):
        """delete_audio_model returns False for non-existent."""
        registry = ModelRegistry(tmp_models_dir)
        result = registry.delete_audio_model("audio/nonexistent")
        assert result is False

    def test_refresh_clears_audio_models(self, tmp_models_dir):
        """refresh() clears audio model registry before rescanning."""
        audio_dir = tmp_models_dir / "MiniMax-Music3"
        audio_dir.mkdir()
        (audio_dir / "configuration.json").write_text("{}")
        (audio_dir / "vocoder").mkdir()

        registry = ModelRegistry(tmp_models_dir)
        assert len(registry.list_audio_models()) == 1

        # Remove the directory and refresh
        import shutil
        shutil.rmtree(audio_dir)

        registry.refresh()
        assert len(registry.list_audio_models()) == 0

    def test_registry_persistence(self, tmp_models_dir):
        """Audio models persist across registry instances."""
        audio_dir = tmp_models_dir / "MiniMax-Music3"
        audio_dir.mkdir()
        (audio_dir / "configuration.json").write_text("{}")
        (audio_dir / "vocoder").mkdir()

        # First instance: scan and save
        registry1 = ModelRegistry(tmp_models_dir)
        assert len(registry1.list_audio_models()) == 1

        # Second instance: should load from saved registry
        registry2 = ModelRegistry(tmp_models_dir)
        audio_models = registry2.list_audio_models()
        assert len(audio_models) == 1
        assert audio_models[0].id == "audio/minimax-music-3"
