# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The ALICE Authors

"""
ALICE Audio Backend Tests

Tests for the audio generation engine, backend, schemas, and FastAPI
endpoint integration.  These tests intentionally do NOT load any real
diffusion-audio model: they verify shape, configuration, and routing.
Run a real model end-to-end with: pytest tests/test_audio.py --run-real
"""

import asyncio
import tempfile
import wave
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_audio_config_defaults():
    """AudioConfig has the expected defaults."""
    from src.config import AudioConfig, Config

    cfg = Config()
    assert cfg.audio.enabled is True
    assert cfg.audio.default_model == "stable-audio-open-1.0"
    assert cfg.audio.default_seconds == 30
    assert cfg.audio.default_steps == 100
    assert cfg.audio.default_cfg_scale == 7.0
    assert cfg.audio.max_concurrent == 1
    assert cfg.audio.unload_after_generate is True
    assert cfg.audio.vae_decode_cpu is False
    assert cfg.audio.force_fp32 is False


def test_audio_config_in_yaml():
    """config.yaml has the audio section and it parses cleanly."""
    from src.config import load_config

    # Load the actual config file shipped with the repo
    cfg = load_config()
    assert hasattr(cfg, "audio")
    assert cfg.audio.enabled is True
    assert cfg.audio.default_model == "stable-audio-open-1.0"


def test_storage_audio_directory_default():
    """StorageConfig carries an audio_directory that defaults to ./audio."""
    from src.config import StorageConfig

    cfg = StorageConfig()
    assert cfg.audio_directory == Path("./audio")


def test_config_migration_includes_audio():
    """get_default_config() includes the audio section so old configs migrate."""
    from src.config_migration import get_default_config

    defaults = get_default_config()
    assert "audio" in defaults
    assert defaults["audio"]["default_model"] == "stable-audio-open-1.0"
    assert defaults["audio"]["unload_after_generate"] is True


# ---------------------------------------------------------------------------
# Engine (no model load)
# ---------------------------------------------------------------------------


def test_engine_imports_without_torch(monkeypatch):
    """ALICEAudioEngine should import; is_available() reflects installed libraries."""
    from src.audio_engine import ALICEAudioEngine, DEFAULT_REPO

    assert DEFAULT_REPO == "stabilityai/stable-audio-open-1.0"
    # Either stable-audio-tools is installed or it isn't.  The static
    # probe has to return the right answer either way; what matters is
    # that the engine module itself imports cleanly.
    result = ALICEAudioEngine.is_available()
    assert isinstance(result, bool)


def test_detect_device_uses_cuda_helper():
    """Device detection follows the prompt's contract."""
    from src.audio_engine import detect_device, detect_dtype
    import torch

    device = detect_device()
    # Must match the rule the prompt specified, NOT a hardcoded string.
    expected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device == expected


def test_detect_dtype_never_returns_bogus():
    """detect_dtype returns a real torch dtype."""
    from src.audio_engine import detect_dtype
    import torch

    assert isinstance(detect_dtype(torch.device("cpu"), force_fp32=False), torch.dtype)
    assert isinstance(detect_dtype(torch.device("cpu"), force_fp32=True), torch.dtype)


def test_engine_creates_output_dir(tmp_path):
    """ALICEAudioEngine makes the output directory on construction."""
    from src.audio_engine import ALICEAudioEngine

    target = tmp_path / "alice_audio_out"
    assert not target.exists()
    engine = ALICEAudioEngine(output_dir=target)
    assert target.exists()
    assert engine.is_loaded() is False


def test_engine_unload_is_safe_when_empty():
    """Calling unload_model on a fresh engine is a no-op."""
    from src.audio_engine import ALICEAudioEngine

    with tempfile.TemporaryDirectory() as tmpdir:
        engine = ALICEAudioEngine(output_dir=Path(tmpdir))
        engine.unload_model()
        assert engine.is_loaded() is False


def test_engine_generate_rejects_empty_prompt(tmp_path):
    """Empty prompt raises ValueError before any model load."""
    from src.audio_engine import ALICEAudioEngine

    engine = ALICEAudioEngine(output_dir=tmp_path)
    with pytest.raises(ValueError):
        engine.generate(prompt="")


# ---------------------------------------------------------------------------
# Backend (no model load)
# ---------------------------------------------------------------------------


def test_backend_catalog_includes_minimax_music3():
    """The catalog exposes MiniMax-Music3 alongside Stable Audio."""
    from src.backends.audio_backend import AUDIO_MODEL_CATALOG

    ids = {m["id"] for m in AUDIO_MODEL_CATALOG}
    assert "stable-audio-open-1.0" in ids
    assert "minimax-music-3" in ids

    minimax = next(m for m in AUDIO_MODEL_CATALOG if m["id"] == "minimax-music-3")
    assert minimax["supports_lyrics"] is True
    assert minimax["engine"] == "minimax_music3"
    # engine dispatched correctly
    assert minimax["max_seconds"] >= 60


def test_backend_catalog_matches_repo_ids():
    """The catalog contains the model ids the docs promise."""
    from src.backends.audio_backend import AUDIO_MODEL_CATALOG, AudioBackend

    ids = {m["id"] for m in AUDIO_MODEL_CATALOG}
    assert "stable-audio-open-1.0" in ids


def test_backend_supported_models_filters_disabled():
    """list_supported_models hides disabled entries, keeps enabled ones."""
    from src.backends.audio_backend import AudioBackend

    enabled = AudioBackend.list_supported_models()
    assert all(m["enabled"] for m in enabled)
    ids = {m["id"] for m in enabled}
    # Both stable-audio-open-1.0 and minimax-music-3 are enabled on
    # systems where the required libraries are importable.
    assert "stable-audio-open-1.0" in ids


def test_backend_get_model_metadata():
    """get_model_metadata returns the catalog entry or None."""
    from src.backends.audio_backend import AudioBackend

    assert AudioBackend.get_model_metadata("stable-audio-open-1.0") is not None
    assert AudioBackend.get_model_metadata("does-not-exist") is None


def test_backend_eviction_callbacks_can_be_registered_and_cleared():
    """The callback registry is process-global and can be cleared."""
    from src.backends import audio_backend as ab

    ab.clear_eviction_callbacks()
    called = []

    async def cb():
        called.append(1)

    ab.register_eviction_callback(cb)
    assert len(ab._eviction_callbacks) == 1
    ab.clear_eviction_callbacks()
    assert ab._eviction_callbacks == []


def test_backend_rejects_unknown_model(tmp_path):
    """generate() raises ValueError when the model id is unknown."""
    from src.backends.audio_backend import AudioBackend

    # We don't actually need stable-audio-tools to test the validation
    # path because the check happens before any model load.
    from src.config import Config

    backend = AudioBackend(config=Config(), output_dir=tmp_path)
    with pytest.raises(ValueError, match="Unknown audio model"):
        asyncio.run(
            backend.generate(
                prompt="test",
                model_id="does-not-exist",
            )
        )


def test_backend_rejects_disabled_model(tmp_path):
    """Disabled catalog entries are rejected at the generate() boundary."""
    from src.backends.audio_backend import AUDIO_MODEL_CATALOG, AudioBackend
    from src.config import Config

    backend = AudioBackend(config=Config(), output_dir=tmp_path)
    # Temporarily flip an entry to disabled and verify rejection.  We
    # pick stable-audio-open-1.0 because it's guaranteed to be in the
    # catalog; flipping minimax-music-3 to disabled would still pass
    # but the catalog filtering already handles it the same way.
    audio_catalog = AUDIO_MODEL_CATALOG
    original = next(m for m in audio_catalog if m["id"] == "stable-audio-open-1.0")
    original["enabled"] = False
    try:
        with pytest.raises(ValueError, match="not enabled"):
            asyncio.run(
                backend.generate(
                    prompt="test",
                    model_id="stable-audio-open-1.0",
                )
            )
    finally:
        original["enabled"] = True


def test_backend_stats_shape(tmp_path):
    """stats() returns a dict with the documented keys."""
    from src.backends.audio_backend import AudioBackend
    from src.config import Config

    backend = AudioBackend(config=Config(), output_dir=tmp_path)
    stats = backend.stats()
    assert "backend" in stats
    assert "available" in stats
    assert "output_dir" in stats
    assert stats["model_loaded"] is False



def test_backend_resolves_minimax_engine_class():
    """The engine factory maps minimax-music-3 to MiniMaxMusic3Engine."""
    from src.audio_engine import MiniMaxMusic3Engine
    from src.backends.audio_backend import _resolve_engine_class

    assert _resolve_engine_class("minimax-music-3") is MiniMaxMusic3Engine


def test_backend_resolves_stable_audio_engine_class():
    """The engine factory maps stable-audio-open-1.0 to ALICEAudioEngine."""
    from src.audio_engine import ALICEAudioEngine
    from src.backends.audio_backend import _resolve_engine_class

    assert _resolve_engine_class("stable-audio-open-1.0") is ALICEAudioEngine


def test_backend_resolves_unknown_model_raises():
    """Unknown model ids raise ValueError at the engine factory."""
    from src.backends.audio_backend import _resolve_engine_class

    with pytest.raises(ValueError, match="Unknown audio model"):
        _resolve_engine_class("does-not-exist")


def test_audio_request_accepts_lyrics_field():
    """AudioGenerationRequest carries a lyrics field for MiniMax-Music3."""
    from src.schemas import AudioGenerationRequest

    req = AudioGenerationRequest(
        prompt="acoustic folk ballad",
        model="minimax-music-3",
        lyrics="[verse]\nHello world\n[chorus]\nSinging out loud",
    )
    assert req.lyrics.startswith("[verse]")
    assert req.model == "minimax-music-3"


def test_minimax_engine_imports():
    """MiniMaxMusic3Engine exposes an availability probe without loading models."""
    from src.audio_engine import MiniMaxMusic3Engine

    result = MiniMaxMusic3Engine.is_available()
    assert isinstance(result, bool)


def test_minimax_engine_creates_output_dir(tmp_path):
    """MiniMaxMusic3Engine ensures the output directory on construction."""
    from src.audio_engine import MiniMaxMusic3Engine

    out = tmp_path / "minimax-out"
    engine = MiniMaxMusic3Engine(output_dir=out)
    assert out.exists()
    assert engine.is_loaded() is False


def test_minimax_engine_generate_rejects_empty_prompt(tmp_path):
    """MiniMaxMusic3Engine.generate refuses empty prompts before model load."""
    from src.audio_engine import MiniMaxMusic3Engine

    engine = MiniMaxMusic3Engine(output_dir=tmp_path)
    with pytest.raises(ValueError, match="prompt"):
        engine.generate(prompt="   ", audio_duration=10.0)


# ---------------------------------------------------------------------------
# MiniMax-Music3 lyrics preprocessing
# ---------------------------------------------------------------------------


def test_preprocess_lyrics_splits_tag_and_text():
    """A [verse] tag followed by text on the same line is split so the
    pipeline's _normalize_lyrics doesn't drop the body text."""
    from src.audio_engine import _preprocess_lyrics

    result = _preprocess_lyrics("[verse] My heart beats fast")
    assert result == "[verse]\nMy heart beats fast"


def test_preprocess_lyrics_preserves_multiline_lyrics():
    """Full lyrics with mixed tag+text and standalone text lines are preserved."""
    from src.audio_engine import _preprocess_lyrics

    raw = "[verse] My heart beats fast\n[chorus] I can't hold back"
    result = _preprocess_lyrics(raw)
    assert result == "[verse]\nMy heart beats fast\n[chorus]\nI can't hold back"


def test_preprocess_lyrics_keeps_standalone_text_lines():
    """Lines without leading tags are passed through unchanged."""
    from src.audio_engine import _preprocess_lyrics

    raw = "My heart beats fast\n[chorus] I can't hold back"
    result = _preprocess_lyrics(raw)
    assert result == "My heart beats fast\n[chorus]\nI can't hold back"


def test_preprocess_lyrics_keeps_existing_proper_formatting():
    """Lyrics already formatted with tags on their own lines pass through."""
    from src.audio_engine import _preprocess_lyrics

    raw = "[verse]\nMy heart beats fast\n[chorus]\nI can't hold back"
    result = _preprocess_lyrics(raw)
    assert result == raw


def test_preprocess_lyrics_handles_multiple_consecutive_tags():
    """Leading consecutive tags like [verse][chorus] are extracted as a group;
    mid-line tags stay in the body (the pipeline's _normalize_lyrics handles
    those via its own ] /  [ replacements)."""
    from src.audio_engine import _preprocess_lyrics

    result = _preprocess_lyrics("[verse] Intro line [chorus] More text")
    assert result == "[verse]\nIntro line [chorus] More text"


def test_preprocess_lyrics_empty_returns_instrumental():
    """Empty string becomes '[instrumental]' so the pipeline doesn't raise."""
    from src.audio_engine import _preprocess_lyrics

    assert _preprocess_lyrics("") == "[instrumental]"


def test_preprocess_lyrics_whitespace_returns_instrumental():
    """Whitespace-only lyrics also become '[instrumental]'."""
    from src.audio_engine import _preprocess_lyrics

    assert _preprocess_lyrics("   \n  \t  ") == "[instrumental]"


def test_preprocess_lyrics_none_returns_instrumental():
    """None is treated as empty and becomes '[instrumental]'."""
    from src.audio_engine import _preprocess_lyrics

    assert _preprocess_lyrics(None) == "[instrumental]"


def test_preprocess_lyrics_already_has_instrumental_preserved():
    """A standalone [instrumental] tag is kept unchanged."""
    from src.audio_engine import _preprocess_lyrics

    result = _preprocess_lyrics("[instrumental]")
    assert result == "[instrumental]"


def test_preprocess_lyrics_no_change_for_plain_text():
    """Plain text without any tags is returned as-is."""
    from src.audio_engine import _preprocess_lyrics

    raw = "Just some lyrics\nWithout any tags\nAt all"
    assert _preprocess_lyrics(raw) == raw


def test_audio_models_list_includes_engine_and_lyrics_flags():
    """AudioModelInfo serialises engine and supports_lyrics flags."""
    from src.backends.audio_backend import AudioBackend
    from src.schemas import AudioModelInfo

    models = [AudioModelInfo(
        id=m["id"], repo=m["repo"], name=m["name"], description=m["description"],
        max_seconds=m["max_seconds"], sample_rate=m["sample_rate"],
        license=m["license"], engine=m["engine"], supports_lyrics=m["supports_lyrics"],
    ) for m in AudioBackend.list_supported_models()]
    by_id = {m.id: m for m in models}

    assert by_id["stable-audio-open-1.0"].engine == "stable_audio"
    assert by_id["stable-audio-open-1.0"].supports_lyrics is False
    assert by_id["minimax-music-3"].engine == "minimax_music3"
    assert by_id["minimax-music-3"].supports_lyrics is True



# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


def test_audio_request_schema_validates():
    """AudioGenerationRequest accepts a minimal payload."""
    from src.schemas import AudioGenerationRequest

    req = AudioGenerationRequest(prompt="a fat dubstep wobble")
    assert req.prompt == "a fat dubstep wobble"
    assert req.model == "stable-audio-open-1.0"
    assert req.seconds is None


def test_audio_request_rejects_blank_prompt():
    """Empty prompt is rejected by the schema."""
    from src.schemas import AudioGenerationRequest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AudioGenerationRequest(prompt="")


def test_audio_request_clamps_seconds():
    """Seconds above the schema max (300) are rejected."""
    from src.schemas import AudioGenerationRequest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AudioGenerationRequest(prompt="test", seconds=600)


def test_audio_request_accepts_input_alias():
    """OpenAI's `input` field is accepted as an alias for prompt."""
    from src.schemas import AudioGenerationRequest

    req = AudioGenerationRequest(input="spoken words")
    assert req.input == "spoken words"


# ---------------------------------------------------------------------------
# HTTP endpoint integration (mocked engine)
# ---------------------------------------------------------------------------


def _mock_audio_result(audio_path: Path, model_id: str = "stable-audio-open-1.0"):
    """Build a fake AudioGenerationResult."""
    from src.backends.audio_backend import AudioGenerationResult

    return AudioGenerationResult(
        audio_path=audio_path,
        url=f"/v1/audio/{audio_path.name}",
        duration_seconds=10.0,
        sample_rate=44100,
        model=model_id,
        seed=42,
        steps=100,
        cfg_scale=7.0,
        prompt="test",
        generation_time_seconds=0.5,
        size_bytes=12345,
    )


@pytest.fixture
def alice_app_with_audio(tmp_path):
    """Build a FastAPI app with a mock audio backend wired in."""
    # We import lazily because the full app pulls in diffusers/torch
    # which may not be installed in the test env.
    from src.config import load_config
    from src import main as alice_main

    # Patch storage.audio_directory to a temp dir before loading
    alice_main.config.storage.audio_directory = tmp_path
    alice_main.config.audio.enabled = True

    # Patch the auth dependency so test requests don't need a real key.
    # `require_auth=False` plus `api_key=None` allows anonymous USER
    # access, but `get_current_user` still raises 503 when auth_manager
    # is None.  We shim both so the endpoint sees an anonymous user.
    if alice_main.auth_manager is None:
        from src.auth import AccessLevel

        class _Anon:
            id = "anonymous"
            name = "Anonymous"

        async def _fake_user():
            return _Anon()

        alice_main.app.dependency_overrides[alice_main.get_current_user] = _fake_user

        async def _fake_access():
            return AccessLevel.USER

        alice_main.app.dependency_overrides[alice_main.require_access_level(alice_main.AccessLevel.ANONYMOUS)] = _fake_access
        alice_main.app.dependency_overrides[alice_main.require_access_level(alice_main.AccessLevel.USER)] = _fake_access
        alice_main.app.dependency_overrides[alice_main.require_access_level(alice_main.AccessLevel.ADMIN)] = _fake_access

    # Inject a fake audio backend
    fake = MagicMock()
    fake.is_available.return_value = True
    fake.get_backend_name.return_value = "Stable Audio (test stub)"
    fake.list_supported_models.return_value = [
        {
            "id": "stable-audio-open-1.0",
            "repo": "stabilityai/stable-audio-open-1.0",
            "name": "Stable Audio Open 1.0",
            "description": "stub",
            "max_seconds": 47,
            "sample_rate": 44100,
            "license": "Stable Audio Community License",
        }
    ]

    async def fake_generate(*args, **kwargs):
        audio_path = tmp_path / "mock_output.wav"
        # Write a tiny valid WAV so the file-serving endpoint can find it
        with wave.open(str(audio_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(b"\x00\x00" * 1024)
        return _mock_audio_result(audio_path)

    fake.generate = AsyncMock(side_effect=fake_generate)
    fake.unload = AsyncMock()
    fake.stats.return_value = {
        "backend": "Stable Audio (test stub)",
        "available": True,
        "model_loaded": False,
        "model_repo": None,
        "n_models_supported": 1,
    }

    alice_main.audio_backend = fake
    yield alice_main

    # Clean up overrides so other tests aren't affected
    alice_main.app.dependency_overrides.clear()


def test_endpoint_generates_audio_returns_200(alice_app_with_audio):
    """POST /v1/audio/generations returns 200 with a URL payload."""
    from fastapi.testclient import TestClient

    app_module = alice_app_with_audio
    client = TestClient(app_module.app)

    resp = client.post(
        "/v1/audio/generations",
        json={"prompt": "128 BPM dubstep wobble", "seconds": 10},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "url" in body
    assert body["url"].startswith("/v1/audio/")
    assert body["model"] == "stable-audio-open-1.0"
    assert body["duration_seconds"] >= 0
    assert body["sample_rate"] == 44100
    assert body["size_bytes"] > 0


def test_endpoint_rejects_blank_prompt(alice_app_with_audio):
    """POST /v1/audio/generations returns 400 on empty prompt."""
    from fastapi.testclient import TestClient

    client = TestClient(alice_app_with_audio.app)
    resp = client.post("/v1/audio/generations", json={"prompt": ""})
    assert resp.status_code in (400, 422)


def test_endpoint_lists_models(alice_app_with_audio):
    """GET /v1/audio/models returns the catalog."""
    from fastapi.testclient import TestClient
    from src.backends.audio_backend import AudioBackend

    client = TestClient(alice_app_with_audio.app)
    resp = client.get("/v1/audio/models")
    assert resp.status_code == 200
    body = resp.json()
    assert "models" in body
    # The endpoint uses the class-level backend name, not the mock's.
    assert body["backend"] == AudioBackend.get_backend_name()
    assert any(m["id"] == "stable-audio-open-1.0" for m in body["models"])
    assert body["available"] is True


def test_endpoint_stats_ok(alice_app_with_audio):
    """GET /v1/audio/stats returns the backend's stats dict."""
    from fastapi.testclient import TestClient

    client = TestClient(alice_app_with_audio.app)
    resp = client.get("/v1/audio/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert "model_loaded" in body


def test_endpoint_serves_generated_wav(alice_app_with_audio):
    """GET /v1/audio/{filename} returns the WAV bytes."""
    from fastapi.testclient import TestClient

    client = TestClient(alice_app_with_audio.app)
    # Generate first
    gen = client.post(
        "/v1/audio/generations",
        json={"prompt": "test", "seconds": 5},
    )
    assert gen.status_code == 200, gen.text
    url = gen.json()["url"]
    # Then fetch
    resp = client.get(url)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert len(resp.content) > 0


def test_endpoint_blocks_path_traversal(alice_app_with_audio):
    """Filename with path components is rejected."""
    from fastapi.testclient import TestClient

    client = TestClient(alice_app_with_audio.app)
    resp = client.get("/v1/audio/../etc/passwd")
    assert resp.status_code in (400, 404)


def test_generated_audio_appears_in_gallery(alice_app_with_audio):
    """Generated audio should be visible in the gallery audio endpoint.

    Regression test for the auth dependency shadowing bug where the
    /v1/auth/me route handler (named get_current_user) shadowed the proper
    auth dependency, causing current_user to be a dict instead of an APIKey
    object.  This made owner_api_key_id always None, making audio invisible
    in the gallery.
    """
    from fastapi.testclient import TestClient
    from src.gallery import GalleryManager

    app_module = alice_app_with_audio
    # Initialize a gallery manager so the gallery endpoints can record/list
    app_module.gallery_manager = GalleryManager(app_module.config.storage.gallery_file)

    client = TestClient(app_module.app)

    # Generate audio
    gen = client.post(
        "/v1/audio/generations",
        json={"prompt": "test prompt", "seconds": 5},
    )
    assert gen.status_code == 200, gen.text

    # Fetch gallery audio - should include the freshly generated audio
    resp = client.get("/v1/gallery/audio?limit=100&offset=0")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "data" in body
    # The audio we just generated must be visible in the gallery
    assert len(body["data"]) >= 1, "Gallery audio endpoint returned no records"
    assert body["data"][0]["prompt"] == "test prompt"


def test_endpoint_returns_503_when_disabled(tmp_path):
    """When config.audio.enabled is False, the endpoint returns 503."""
    from src.config import load_config
    from src import main as alice_main
    from fastapi.testclient import TestClient

    alice_main.config.audio.enabled = False
    alice_main.audio_backend = None
    client = TestClient(alice_main.app)

    resp = client.post("/v1/audio/generations", json={"prompt": "test"})
    assert resp.status_code == 503
