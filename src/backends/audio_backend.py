# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2026 Andrew Wyatt (Fewtarius)

"""
ALICE Audio Backend

Service-layer wrapper around `ALICEAudioEngine` that knows how to talk to
the rest of ALICE: configuration, storage, cancellation, and the GPU
coordination required so audio generation does not collide with image
generation.

Design notes:
    - We deliberately do NOT subclass `BaseBackend`.  That abstract class
      is hard-wired to image generation (PIL.Image, LoRA, schedulers)
      and would force awkward adapter code.  Audio has a different
      shape, so we mirror the API surface without inheriting the
      contract.
    - The GPU lock is a server-wide asyncio lock (`_gpu_lock`).  Any
      image generation that holds the GPU while loaded is evicted
      via `request_eviction()` before audio runs.  Conversely, audio
      unloads itself after each generation by default so the image
      pipeline can resume on the next request without manual
      intervention.
    - Cancellation hooks go through the same `cancellation` registry
      that image generation uses.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import torch

from ..audio_engine import (
    ALICEAudioEngine,
    DEFAULT_REPO,
    DEFAULT_SECONDS,
    DEFAULT_STEPS,
    MiniMaxMusic3Engine,
    MINIMAX_MUSIC3_DEFAULT_REPO,
    MINIMAX_MUSIC3_DEFAULT_SECONDS,
    MINIMAX_MUSIC3_DEFAULT_STEPS,
)
from ..config import Config

logger = logging.getLogger(__name__)


# Catalog of supported audio models.  Each entry is metadata the API
# surfaces via GET /v1/audio/models.  Add new repos here when broader
# model support lands (e.g. MiniMax-Music3 once its diffusers pipeline
# loses the "Inference requires CUDA" caveat).
AUDIO_MODEL_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "stable-audio-open-1.0",
        "repo": DEFAULT_REPO,
        "name": "Stable Audio Open 1.0",
        "description": (
            "Open-weights diffusion-audio model from Stability AI. "
            "Up to 47s of stereo audio at 44.1kHz from text prompts. "
            "Best for sound effects, field recordings, and short music loops."
        ),
        "max_seconds": 47,
        "sample_rate": 44100,
        "license": "Stable Audio Community License",
        "default_steps": 100,
        "default_cfg_scale": 7.0,
        "engine": "stable_audio",
        "supports_lyrics": False,
        "param_shape": "stable_audio",
        "enabled": True,
    },
    {
        "id": "minimax-music-3",
        "repo": MINIMAX_MUSIC3_DEFAULT_REPO,
        "name": "MiniMax Music 3",
        "description": (
            "Text-to-music with lyrics + structured music description. "
            "Up to 6-minute songs at 44.1kHz stereo. Powered by a Qwen3-8B "
            "language model + flow-matching diffusion + DAC vocoder. "
            "Slow on consumer hardware (~25x realtime) but produces full songs."
        ),
        "max_seconds": 360,
        "sample_rate": 44100,
        "license": "MiniMax Music 3 Model License",
        "default_steps": 30,
        "default_cfg_scale": 0.0,  # not used; flow-matching has no CFG
        "engine": "minimax_music3",
        "supports_lyrics": True,
        "param_shape": "minimax_music3",
        "enabled": True,
    },
]


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------


# Each engine class implements the same minimal interface that AudioBackend
# needs: is_available(), is_loaded(), load_model(), unload_model(), generate(),
# gpu_info().  Stable Audio and MiniMax-Music3 have completely different
# underlying APIs (stable-audio-tools vs diffusers modular pipeline) so we
# keep the engines independent rather than forcing a shared interface.
_ENGINE_CLASSES = {
    "stable_audio": ALICEAudioEngine,
    "minimax_music3": MiniMaxMusic3Engine,
}


def _resolve_engine_class(model_id: str) -> type:
    """Look up the engine class for a given model id."""
    metadata = _find_metadata(model_id)
    if metadata is None:
        raise ValueError(f"Unknown audio model: {model_id}")
    engine_kind = metadata.get("engine", "stable_audio")
    cls = _ENGINE_CLASSES.get(engine_kind)
    if cls is None:
        raise ValueError(f"Unknown audio engine '{engine_kind}' for model {model_id}")
    return cls


def _find_metadata(model_id: str) -> Optional[Dict[str, Any]]:
    for entry in AUDIO_MODEL_CATALOG:
        if entry["id"] == model_id:
            return entry
    return None


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class AudioGenerationResult:
    """Path + metadata returned by AudioBackend.generate()."""
    audio_path: Path
    url: str
    duration_seconds: float
    sample_rate: int
    model: str
    seed: int
    steps: int
    cfg_scale: float
    prompt: str
    generation_time_seconds: float
    size_bytes: int


# ---------------------------------------------------------------------------
# Coordination hook
# ---------------------------------------------------------------------------


# Image-side helpers can register a callback here so the audio backend
# can request that loaded image models be evicted before audio runs.
# The mirror direction (audio eviction before image) is handled by
# `unload_after_generate=True` which is the default below.
EvictionCallback = Callable[[], Awaitable[None]]
_eviction_callbacks: List[EvictionCallback] = []


def register_eviction_callback(cb: EvictionCallback) -> None:
    """Image side calls this at startup to register a coroutine that
    unloads any cached image models.  Audio uses it before loading."""
    _eviction_callbacks.append(cb)


def clear_eviction_callbacks() -> None:
    """Used by tests."""
    _eviction_callbacks.clear()


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class AudioBackend:
    """
    High-level audio service.

    Single instance per ALICE process.  Owns one ALICEAudioEngine.  All
    public methods are async to match the rest of the FastAPI stack.
    """

    def __init__(
        self,
        config: Config,
        output_dir: Path,
        max_concurrent: int = 1,
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Directory where model weights live (from config.models.directory).
        # Audio backends store model files here alongside image models.
        self.models_dir = config.models.directory

        self._gpu_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max(1, max_concurrent))
        self._engine: Optional[Any] = None
        self._engine_model_id: Optional[str] = None
        self._engine_lock = asyncio.Lock()
        self._inflight: int = 0

        logger.info(
            "AudioBackend ready: output=%s max_concurrent=%d",
            self.output_dir, max_concurrent,
        )

    # -- availability --------------------------------------------------------

    @staticmethod
    def is_available() -> bool:
        """
        True iff the underlying libraries can be imported.  Does not
        check that a usable GPU is present; the engine will fall back
        to CPU.
        """
        # True if any audio engine imports cleanly.  Per-model availability
        # is reported via `is_model_available()` and surfaced in the
        # catalog (`available` field on each entry).
        return (
            ALICEAudioEngine.is_available()
            or MiniMaxMusic3Engine.is_available()
        )

    @staticmethod
    def get_backend_name() -> str:
        # Reflects the union of supported engines; surfaced in
        # /v1/audio/models and /health so clients see the full surface.
        names = []
        if ALICEAudioEngine.is_available():
            names.append("Stable Audio (stable-audio-tools)")
        if MiniMaxMusic3Engine.is_available():
            names.append("MiniMax-Music3 (diffusers modular)")
        return ", ".join(names) if names else "audio (no engines available)"

    @staticmethod
    def list_supported_models() -> List[Dict[str, Any]]:
        return [m for m in AUDIO_MODEL_CATALOG if m.get("enabled", True)]

    @staticmethod
    def get_model_metadata(model_id: str) -> Optional[Dict[str, Any]]:
        return _find_metadata(model_id)

    @staticmethod
    def is_model_available(model_id: str) -> bool:
        """Engine import check for a specific model id."""
        metadata = _find_metadata(model_id)
        if metadata is None or not metadata.get("enabled", True):
            return False
        engine_kind = metadata.get("engine", "stable_audio")
        if engine_kind == "stable_audio":
            return ALICEAudioEngine.is_available()
        if engine_kind == "minimax_music3":
            return MiniMaxMusic3Engine.is_available()
        return False

    def _resolve_model_path(self, model_id: str) -> Optional[Path]:
        """
        Resolve the local model path for a model id.

        Checks three locations:
          1. The exact path given by the catalog 'repo' field (if it
             exists on disk - a pre-downloaded local checkout).
          2. <models.directory>/<repo_basename>  (e.g. .../MiniMax-Music3)
          3. <models.directory>/<model_id.replace('-','')>  (fallback)

        Returns None if no local checkout exists, in which case the
        engine falls back to downloading from the HF repo id.
        """
        metadata = _find_metadata(model_id)
        if metadata is None:
            return None
        repo = metadata["repo"]

        # 1. If the catalog already gives a local path, use it.
        candidate = Path(repo)
        if candidate.exists():
            return candidate

        # 2. Look under the configured models directory using the repo's
        # basename (e.g. "MiniMaxAI/MiniMax-Music3" -> "MiniMax-Music3").
        basename = repo.rsplit("/", 1)[-1]
        candidate = self.models_dir / basename
        if candidate.exists():
            return candidate

        # 3. Fallback: model_id as a directory name.
        candidate = self.models_dir / model_id
        if candidate.exists():
            return candidate

        return None

    # -- engine lifecycle ----------------------------------------------------

    async def _get_engine(self, model_id: str) -> Any:
        """Return the engine instance for `model_id`, instantiating it on first use."""
        if self._engine is not None and self._engine_model_id == model_id:
            return self._engine
        # Model switch: unload the previous engine and stand up the new one.
        async with self._engine_lock:
            if self._engine is not None and self._engine_model_id != model_id:
                logger.info(
                    "AudioBackend switching engine %s -> %s",
                    self._engine_model_id, model_id,
                )
                try:
                    self._engine.unload_model()
                except Exception:
                    pass
                self._engine = None
                self._engine_model_id = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if self._engine is None:
                engine_cls = _resolve_engine_class(model_id)
                self._engine = engine_cls(
                    output_dir=self.output_dir,
                    force_fp32=self.config.generation.force_float32,
                    vae_decode_cpu=self.config.generation.vae_decode_cpu,
                )
                self._engine_model_id = model_id
        return self._engine

    async def unload(self) -> None:
        """Drop the loaded model and free VRAM."""
        async with self._engine_lock:
            if self._engine is not None:
                logger.info("AudioBackend unloading engine")
                self._engine.unload_model()
                self._engine = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- generation ----------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        model_id: str = "stable-audio-open-1.0",
        seconds: Optional[int] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        seed: Optional[int] = None,
        lyrics: str = "",
        unload_after_generate: bool = True,
        request_id: Optional[str] = None,
        cancellation_check: Optional[Callable[[], bool]] = None,
    ) -> AudioGenerationResult:
        """
        Generate one audio clip.

        Args:
            prompt: Text prompt (music description for MiniMax-Music3).
            model_id: API model id (e.g. "stable-audio-open-1.0",
                "minimax-music-3").
            seconds: Clip length in seconds (clamped to model max).
            steps: Diffusion/flow-matching steps (model-specific).
            cfg_scale: Classifier-free guidance scale (ignored for
                MiniMax-Music3 which uses flow-matching without CFG).
            seed: Reproducibility seed.
            lyrics: Lyrics with optional [verse]/[chorus] structure
                tags.  Empty string for instrumental.  Only used by
                MiniMax-Music3.
            unload_after_generate: Drop the model after generation.
                Default True so image generation resumes on the next
                request without manual eviction.
            request_id: Optional request id for logging.
            cancellation_check: Optional callable returning True when
                the caller wants to abort.

        Returns:
            AudioGenerationResult with path, URL, and metadata.
        """
        request_id = request_id or f"audio-{uuid.uuid4().hex[:12]}"
        metadata = self.get_model_metadata(model_id)
        if metadata is None:
            raise ValueError(f"Unknown audio model: {model_id}")
        if not metadata.get("enabled", True):
            raise ValueError(
                f"Audio model '{model_id}' is not enabled on this server"
            )
        if not self.is_model_available(model_id):
            raise ValueError(
                f"Audio model '{model_id}' requires libraries that are not installed"
            )

        seconds = seconds if seconds is not None else DEFAULT_SECONDS
        seconds = max(1, min(int(seconds), int(metadata["max_seconds"])))
        steps = steps if steps is not None else int(metadata["default_steps"])
        cfg_scale = cfg_scale if cfg_scale is not None else float(metadata["default_cfg_scale"])

        # Per-model path resolution.  Stable Audio and MiniMax-Music3
        # have different output file locations and may produce different
        # durations (Stable Audio rounds to its latent grid; MiniMax-Music3
        # uses floating-point seconds).
        engine_kind = metadata.get("engine", "stable_audio")
        is_minimax = engine_kind == "minimax_music3"

        async with self._semaphore:
            async with self._gpu_lock:
                self._inflight += 1
                try:
                    # Ask any registered image backend to drop its
                    # cached models so we have room for the audio VAE.
                    for cb in _eviction_callbacks:
                        try:
                            await cb()
                        except Exception as exc:  # pragma: no cover
                            logger.warning(
                                "Eviction callback raised (ignored): %s", exc
                            )

                    if cancellation_check is not None and cancellation_check():
                        raise RuntimeError("Audio generation cancelled before start")

                    engine = await self._get_engine(model_id)
                    logger.info(
                        "[%s] audio generate: model=%s seconds=%d steps=%d cfg=%.2f",
                        request_id, model_id, seconds, steps, cfg_scale,
                    )

                    # Stable Audio Open 1.0's text encoder and DiT
                    # can take 30-60s; MiniMax-Music3 takes much longer
                    # (Qwen3 AR is the bottleneck).  In both cases we
                    # push the work to a thread so the event loop stays
                    # responsive to cancellation pings.
                    # Resolve the local model path before calling the engine.
                    # Falls back to the HF repo id if nothing is on disk.
                    resolved_path = self._resolve_model_path(model_id)
                    model_location = str(resolved_path) if resolved_path is not None else metadata["repo"]

                    if is_minimax:
                        audio_path = await asyncio.to_thread(
                            engine.generate,
                            prompt=prompt,
                            lyrics=lyrics or "",
                            audio_duration=float(seconds),
                            num_inference_steps=steps,
                            seed=seed,
                            model_repo_or_path=model_location,
                        )
                    else:
                        audio_path = await asyncio.to_thread(
                            engine.generate,
                            prompt=prompt,
                            seconds=seconds,
                            steps=steps,
                            cfg_scale=cfg_scale,
                            seed=seed,
                            model_repo=model_location,
                        )

                    if cancellation_check is not None and cancellation_check():
                        # Best-effort cleanup if we were cancelled during gen.
                        try:
                            audio_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                        raise RuntimeError("Audio generation cancelled mid-flight")

                    # Resolve generation start time for elapsed tracking
                    gen_start = time.time()
                    stat = audio_path.stat()
                    rel = audio_path.name
                    elapsed = time.time() - gen_start
                    return AudioGenerationResult(
                        audio_path=audio_path,
                        url=f"/v1/audio/{rel}",
                        duration_seconds=float(seconds),
                        sample_rate=int(metadata["sample_rate"]),
                        model=model_id,
                        seed=int(seed) if seed is not None else 0,
                        steps=int(steps),
                        cfg_scale=float(cfg_scale),
                        prompt=prompt,
                        generation_time_seconds=round(elapsed, 3),
                        size_bytes=stat.st_size,
                    )
                finally:
                    self._inflight = max(0, self._inflight - 1)
                    if unload_after_generate:
                        await self.unload()

    # -- introspection -------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Snapshot for /v1/audio/models and /health endpoints."""
        engine = self._engine
        return {
            "backend": self.get_backend_name(),
            "available": self.is_available(),
            "output_dir": str(self.output_dir),
            "max_concurrent": self._semaphore._value,  # type: ignore[attr-defined]
            "inflight": self._inflight,
            "model_loaded": engine.is_loaded() if engine else False,
            "model_id": self._engine_model_id,
            "model_repo": engine.model_repo if engine else None,
            "n_models_supported": len(self.list_supported_models()),
        }


__all__ = [
    "AudioBackend",
    "AudioGenerationResult",
    "AUDIO_MODEL_CATALOG",
    "register_eviction_callback",
    "clear_eviction_callbacks",
]
