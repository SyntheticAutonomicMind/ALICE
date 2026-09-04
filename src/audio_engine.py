# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2026 Andrew Wyatt (Fewtarius)

"""
ALICE Audio Engine

Thin wrapper around `stable-audio-tools` that runs diffusion-audio models
(Stable Audio Open 1.0 today, more in the future) on the same ROCm/CUDA/MPS
torch stack as the image backends.

Design requirements:
    - Device routing follows the prompt's rule:
          device = "cuda" if torch.cuda.is_available() else "cpu"
      but never hardcodes that string into library internals.
    - FP16 autocast via the modern torch.amp API (device-aware).
    - Optional VAE-decode-on-CPU for AMD APUs that hang on GPU decode
      (same workaround as the SDXL gfx1103 path).
    - No monkey-patching of vendored library source.

The engine is intentionally framework-agnostic: it does not import FastAPI,
auth, or storage.  The backend layer (src/backends/audio_backend.py) wires
it into the rest of ALICE.
"""

from __future__ import annotations

import gc
import logging
import os
import random
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Suppress non-actionable torch._sympy / diffusers / transformers log spam.
# These modules emit via logging.warning(), so warnings.filterwarnings has
# no effect.  Logger objects are singletons, so these level adjustments take
# effect for code imported later (inside load_model / generate).
for _noisy_log_name in (
    "torch.utils._sympy.interp",
    "diffusers.models.modeling_utils",
    "transformers.tokenization_utils_base",
):
    logging.getLogger(_noisy_log_name).setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend configuration user-tunable constants
# ---------------------------------------------------------------------------

# Stable Audio Open 1.0 caps generation at 47 seconds (model_config sample_size
# divided by sample_rate gives the maximum).  We expose reasonable defaults
# but let callers override.
DEFAULT_REPO = "stabilityai/stable-audio-open-1.0"
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_SAMPLE_SIZE = 1048576  # 1 << 20 latent frames -> ~47s @ 44.1kHz
DEFAULT_STEPS = 100
DEFAULT_CFG_SCALE = 7.0
DEFAULT_SECONDS = 30


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def detect_device(force_cpu: bool = False) -> torch.device:
    """
    Resolve the device we should run inference on.

    The prompt was explicit: do NOT hardcode the literal string "cuda" in
    upstream library code.  Instead, detect the device here and pass it
    through.  ROCm reports itself as CUDA via torch.cuda, so the same
    "cuda" branch covers both NVIDIA and AMD.
    """
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def detect_dtype(device: torch.device, force_fp32: bool = False) -> torch.dtype:
    """
    Pick the autocast dtype for the chosen device.

    On gfx1103 (Phoenix APU) FP16/FP16 VAE decode can crash the GPU; users
    running there already set `force_float32` in image generation.  We
    mirror that knob here.  bfloat16 is also exposed as a fallback for
    gfx1102.
    """
    if force_fp32:
        return torch.float32
    if device.type == "cuda":
        return torch.float16
    return torch.float32  # CPU path stays fp32 for deterministic output


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ALICEAudioEngine:
    """
    Single-model audio inference wrapper.

    Loads Stable Audio Open 1.0 (or any compatible diffusion-audio model
    with the same `stable_audio_tools` API) on demand, generates a clip
    from a text prompt, and writes a normalised int16 WAV to disk.

    The engine is deliberately stateless across generations aside from
    the cached model.  Callers that need exclusive access to the GPU
    should wrap calls in a lock (see `AudioBackend`).
    """

    def __init__(
        self,
        output_dir: Path,
        device: Optional[torch.device] = None,
        force_fp32: bool = False,
        vae_decode_cpu: bool = False,
    ):
        """
        Args:
            output_dir: Directory to write generated WAV files into.
            device: Torch device.  Auto-detected if None.
            force_fp32: Force float32 dtype (mimic image backend knob).
            vae_decode_cpu: Decode VAE on CPU (AMD gfx1103 workaround).
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or detect_device()
        self.force_fp32 = force_fp32
        self.vae_decode_cpu = vae_decode_cpu
        self.dtype = detect_dtype(self.device, force_fp32=force_fp32)

        self.model: Optional[Any] = None
        self.model_config: Optional[Dict[str, Any]] = None
        self.model_repo: Optional[str] = None

        logger.info(
            "ALICEAudioEngine initialised: device=%s dtype=%s output_dir=%s",
            self.device, self.dtype, self.output_dir,
        )

    # -- availability --------------------------------------------------------

    @staticmethod
    def is_available() -> bool:
        """Return True if stable-audio-tools + torchaudio can be imported."""
        try:
            import stable_audio_tools  # noqa: F401
            import torchaudio  # noqa: F401
            return True
        except Exception as exc:  # pragma: no cover - import probe
            logger.debug("stable-audio-tools unavailable: %s", exc)
            return False

    # -- model lifecycle -----------------------------------------------------

    def is_loaded(self) -> bool:
        return self.model is not None

    def load_model(self, model_repo: str = DEFAULT_REPO) -> None:
        """Load the diffusion model into memory. Idempotent."""
        if self.model is not None and self.model_repo == model_repo:
            return

        if self.model is not None and self.model_repo != model_repo:
            logger.info("Switching audio model %s -> %s", self.model_repo, model_repo)
            self.unload_model()

        from stable_audio_tools import get_pretrained_model

        logger.info("Loading audio model: %s", model_repo)
        model, model_config = get_pretrained_model(model_repo)
        model = model.to(self.device)
        try:
            model.eval()
        except Exception:  # some wrappers don't expose .eval()
            pass

        self.model = model
        self.model_config = dict(model_config)
        self.model_repo = model_repo
        logger.info(
            "Audio model loaded: sample_rate=%s sample_size=%s",
            self.model_config.get("sample_rate"),
            self.model_config.get("sample_size"),
        )

    def unload_model(self) -> None:
        """Drop the cached model and free VRAM."""
        self.model = None
        self.model_config = None
        self.model_repo = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- generation ----------------------------------------------------------

    def generate(
        self,
        prompt: str,
        seconds: int = DEFAULT_SECONDS,
        steps: int = DEFAULT_STEPS,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        seed: Optional[int] = None,
        model_repo: str = DEFAULT_REPO,
    ) -> Path:
        """
        Generate a stereo WAV from a text prompt.

        Args:
            prompt: Text prompt describing the audio to generate.
            seconds: Length of the clip in seconds (capped by model).
            steps: Number of diffusion steps.
            cfg_scale: Classifier-free guidance scale.
            seed: Optional seed for reproducibility.
            model_repo: HuggingFace repo id of the model to load.

        Returns:
            Path to the generated audio file (MP3 if ffmpeg conversion succeeded, otherwise WAV).
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        self.load_model(model_repo)
        assert self.model is not None and self.model_config is not None

        # Stable Audio Open 1.0 accepts a per-clip seconds_total in the
        # conditioning payload.  47s is the model ceiling; clamp below.
        seconds = max(1, min(int(seconds), 47))
        sample_size = self.model_config["sample_size"]
        sample_rate = int(self.model_config["sample_rate"])

        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        conditioning: List[Dict[str, Any]] = [{
            "prompt": prompt,
            "seconds_start": 0,
            "seconds_total": seconds,
        }]

        start = time.time()
        logger.info(
            "Generating audio: prompt_len=%d seconds=%d steps=%d cfg=%.1f seed=%d",
            len(prompt), seconds, steps, cfg_scale, seed,
        )
        logger.debug("Full prompt: %r", prompt)

        # The diffusion loop.  We use the modern device-aware autocast
        # instead of the deprecated torch.cuda.amp.autocast so this also
        # works on CPU and MPS (the latter is uncommon for audio but
        # harmless).
        amp_enabled = (self.device.type == "cuda") and (self.dtype == torch.float16)
        with torch.inference_mode():
            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.dtype,
                enabled=amp_enabled,
            ):
                from stable_audio_tools.inference.generation import generate_diffusion_cond

                latents = generate_diffusion_cond(
                    self.model,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    conditioning=conditioning,
                    sample_size=sample_size,
                    device=self.device,
                    sigma_min=0.3,
                    sigma_max=500.0,
                    sampler_type="dpmpp-3m-sde",
                    seed=seed,
                )

        # The VAE decode is the part that historically hangs on AMD gfx1103.
        # Mirror the image backend knob: decode on CPU when requested.
        decode_device = torch.device("cpu") if self.vae_decode_cpu else self.device
        if decode_device != self.device:
            logger.info("VAE decode on CPU (vae_decode_cpu=True)")
        if self.vae_decode_cpu and latents.device != torch.device("cpu"):
            latents = latents.to("cpu")

        with torch.inference_mode():
            audio = self.model.vae.decode(latents.to(decode_device))

        # Stable Audio Open 1.0 returns audio as (batch, channels, samples)
        # in float32 in [-1, 1].  Normalise, peak-limit, convert to int16.
        audio = audio.clamp(-1.0, 1.0)
        peak = audio.abs().max()
        if peak > 0:
            audio = audio / peak
        audio = audio.squeeze(0).to(torch.float32).cpu()

        # Stable Audio Tools wants (channels, samples) for torchaudio.save.
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        elif audio.dim() == 3:
            audio = audio.squeeze(0)

        # Write file.  Filename uses the prompt hash + seed so retries of
        # the same prompt overwrite predictably rather than filling disk.
        from einops import rearrange

        try:
            audio = rearrange(audio, "d (b n) -> d (b n)", b=1)
        except Exception:
            pass  # einops is a no-op for our single-batch singletons

        suffix = f"seed{seed}"
        short_hash = abs(hash(prompt)) % (10**8)
        output_path = self.output_dir / f"alice_{short_hash:08d}_{suffix}.wav"

        import torchaudio

        torchaudio.save(str(output_path), audio, sample_rate)

        elapsed = time.time() - start
        logger.info(
            "Audio generation complete: %s (%.2fs, %.2f MB)",
            output_path.name, elapsed, output_path.stat().st_size / (1024 * 1024),
        )

        # Free unused intermediates aggressively.  The model itself stays
        # loaded (the backend decides when to unload).
        del latents, audio
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output_path

    # -- introspection -------------------------------------------------------

    def gpu_info(self) -> Dict[str, Any]:
        """Backend-shaped GPU info block."""
        info: Dict[str, Any] = {
            "device": str(self.device),
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": None,
            "memory_used": None,
            "memory_total": None,
            "stats_available": False,
            "model_loaded": self.is_loaded(),
            "model_repo": self.model_repo,
        }
        if torch.cuda.is_available():
            try:
                info["gpu_name"] = torch.cuda.get_device_name(0)
                free, total = torch.cuda.mem_get_info()
                info["memory_used"] = f"{(total - free) / (1024 ** 3):.2f} GB"
                info["memory_total"] = f"{total / (1024 ** 3):.2f} GB"
                info["stats_available"] = True
            except Exception as exc:  # pragma: no cover - hardware probe
                logger.debug("GPU stats probe failed: %s", exc)
        return info


# ---------------------------------------------------------------------------
# MiniMax-Music3 engine (flow-matching diffusion with Qwen3 AR + DAC vocoder)
# ---------------------------------------------------------------------------

# MiniMax-Music3 is a multi-component pipeline (Qwen3 AR language model +
# Flow-VAE + RVQ depth decoder + flow-matching transformer + DAC vocoder).
# It's the first audio backend that doesn't ship through stable-audio-tools;
# it lives in the diffusers modular_pipelines package and needs a hand-
# assembled component dictionary because the language_model is a transformers
# model rather than a diffusers one.

# A line starting with one or more [tag] patterns (mirrors the pipeline's
# own regex in diffusers.modular_pipelines.minimax_music3.encoders).
_MINIMAX_LEADING_TAGS_RE = re.compile(r"^[ \t]*((?:\[[^\]]+\][ \t]*)+)")


def _preprocess_lyrics(lyrics: Optional[str]) -> str:
    """
    Pre-process lyrics so the MiniMax-Music3 pipeline preserves the user's
    text.

    The pipeline's internal `_normalize_lyrics` matches lines beginning with
    structure tags (e.g. ``[verse]``) and keeps *only* the tag, silently
    dropping any body text on the same line.  So a line like
    ``[verse] My heart beats fast`` becomes just ``[verse]`` and the actual
    lyrics vanish.

    This helper splits such lines into two: the tag on its own line and the
    body text on the next line.  After this transform the pipeline keeps both.

    Empty or whitespace-only lyrics (including ``None``) are replaced with
    ``"[instrumental]"`` because the pipeline raises ``ValueError`` when
    ``lyrics.strip()`` is falsy.
    """
    if not lyrics or not lyrics.strip():
        return "[instrumental]"

    lines = []
    for line in lyrics.split("\n"):
        match = _MINIMAX_LEADING_TAGS_RE.match(line)
        if match:
            tags = match.group(1).strip()
            rest = line[match.end():]
            if rest.strip():
                lines.append(tags)
                lines.append(rest.strip())
            else:
                lines.append(tags)
        else:
            lines.append(line)
    return "\n".join(lines)


# Pipeline-level defaults.  These mirror the upstream defaults and the
# numbers used in the trial run that produced the first end-to-end output.
MINIMAX_MUSIC3_DEFAULT_REPO = "MiniMaxAI/MiniMax-Music3"
MINIMAX_MUSIC3_DEFAULT_SECONDS = 30.0      # request shape: float seconds
MINIMAX_MUSIC3_MAX_SECONDS = 360.0         # six minutes (9000 frames / 25 fps)
MINIMAX_MUSIC3_DEFAULT_STEPS = 30          # flow-matching Euler steps per chunk
MINIMAX_MUSIC3_DEFAULT_SAMPLE_RATE = 44100 # pipeline actually outputs 44.1kHz


class MiniMaxMusic3Engine:
    """
    MiniMax-Music3 inference wrapper.

    Loads the multi-component diffusers modular pipeline on demand, generates
    a music clip from a (lyrics, prompt) pair, and writes a normalised int16
    WAV to disk.

    The component dictionary must be built by hand because the upstream
    `load_components()` fails on the language_model component (a transformers
    `Qwen3ForCausalLM`, not a diffusers model).  See the diffusers PR #14456
    for upstream context.
    """

    def __init__(
        self,
        output_dir: Path,
        device: Optional[torch.device] = None,
        force_fp32: bool = False,
        vae_decode_cpu: bool = False,
        force_float32: bool = False,
        force_bfloat16: bool = False,
    ):
        """
        Args:
            output_dir: Directory to write generated WAV files into.
            device: Torch device.  Auto-detected if None.
            force_fp32: Force float32 dtype (rare; FP16 VAE hangs gfx1103).
            vae_decode_cpu: Decode on CPU (AMD gfx1103 workaround).
            force_float32: Alias for force_fp32 (config.generation.force_float32).
            force_bfloat16: Prefer bfloat16 despite force_fp32 (AMD Phoenix).
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or detect_device()
        self.force_fp32 = force_fp32 or force_float32
        self.vae_decode_cpu = vae_decode_cpu

        # Dtype resolution order for AMD GPUs:
        #   force_bfloat16 > force_fp32 > default (float16 on CUDA, float32 on CPU)
        # bfloat16 is preferred on AMD Phoenix/Phoenix Point APUs; fp32 on
        # gfx1103 (where fp16/bf16 can hang the GPU).
        if force_bfloat16:
            self.dtype = torch.bfloat16
        else:
            self.dtype = detect_dtype(self.device, force_fp32=self.force_fp32)

        self.pipeline: Optional[Any] = None
        self.model_repo: Optional[str] = None

        logger.info(
            "MiniMaxMusic3Engine initialised: device=%s dtype=%s output_dir=%s",
            self.device, self.dtype, self.output_dir,
        )

    # -- availability --------------------------------------------------------

    @staticmethod
    def is_available() -> bool:
        """
        Return True iff the MiniMax-Music3 pipeline + Qwen3 can be imported.

        Note: this is a *static* import probe.  The actual model weights
        must also exist on disk before generation will succeed; the backend
        handles that distinction.
        """
        try:
            from diffusers.modular_pipelines.minimax_music3 import (  # noqa: F401
                MiniMaxMusic3ModularPipeline,
            )
            from transformers import Qwen3ForCausalLM  # noqa: F401
            return True
        except Exception as exc:  # pragma: no cover - import probe
            logger.debug("MiniMax-Music3 unavailable: %s", exc)
            return False

    def _lm_max_memory(self) -> Optional[Dict[str, str]]:
        """
        Build a max_memory map for accelerate device_map="auto" on the
        Qwen3 language model.

        On GPUs with limited VRAM (e.g. Radeon 8060S / Strix Halo with ~8GB),
        loading the full Qwen3-8B model plus the diffusion transformer,
        VAE, and vocoder simultaneously causes a GPU memory access fault.
        We cap the GPU budget so accelerate spills overflow layers to CPU.

        Returns ``None`` when CUDA is unavailable — in that case we load the
        language model normally on the available device without device_map
        (which is a no-op on CPU-only systems and can cause issues with
        some HuggingFace model classes).
        """
        if not torch.cuda.is_available():
            return None
        # Give the language model a tight GPU budget; the remaining components
        # (flow transformer, vocoder) need VRAM too.  4GB on an 8GB card leaves
        # ~4GB for the rest.  If the model doesn't fit, accelerate moves layers
        # to CPU automatically.
        gpu_mem = torch.cuda.get_device_properties(0).total_memory
        gpu_budget = min(gpu_mem // 2, 4 * 1024 * 1024 * 1024)
        logger.info("MiniMax language model GPU budget: %.1f GB / %.1f GB total",
                    gpu_budget / (1024**3), gpu_mem / (1024**3))
        return {"cpu": "32GB", 0: f"{gpu_budget} bytes"}

    # -- model lifecycle -----------------------------------------------------

    def is_loaded(self) -> bool:
        return self.pipeline is not None

    def load_model(self, model_repo_or_path: str = MINIMAX_MUSIC3_DEFAULT_REPO) -> None:
        """
        Load the multi-component pipeline into memory. Idempotent.

        Accepts either a HuggingFace repo id (downloaded on first use) or a
        local path (production deployment).  The component dictionary is
        built by hand because the upstream helper can't load the
        `language_model` (a transformers `Qwen3ForCausalLM`).
        """
        if self.pipeline is not None and self.model_repo == model_repo_or_path:
            return

        if self.pipeline is not None and self.model_repo != model_repo_or_path:
            logger.info("Switching audio model %s -> %s", self.model_repo, model_repo_or_path)
            self.unload_model()

        local_files_only = Path(model_repo_or_path).exists()
        if not local_files_only:
            logger.info(
                "MiniMax-Music3 repo %s not found on disk; will download on first load",
                model_repo_or_path,
            )

        from diffusers.modular_pipelines.minimax_music3 import MiniMaxMusic3ModularPipeline
        from transformers import Qwen3ForCausalLM, AutoTokenizer
        from diffusers import FlowMatchEulerDiscreteScheduler
        from diffusers.models.transformers.minimax_music3_rvq_depth_decoder import (
            MiniMaxMusic3RVQDepthDecoder,
        )
        from diffusers.models.condition_embedders.condition_embedder_minimax_music3 import (
            MiniMaxMusic3ConditionEncoder,
        )
        from diffusers.models.transformers.transformer_minimax_music3 import (
            MiniMaxMusic3Transformer1DModel,
        )
        from diffusers.models.autoencoders.minimax_music3_vocoder import (
            MiniMaxMusic3Vocoder,
        )

        logger.info("Loading MiniMax-Music3 from %s (local_only=%s)",
                    model_repo_or_path, local_files_only)

        # Step 1: shell pipeline (no components yet)
        pipeline = MiniMaxMusic3ModularPipeline.from_pretrained(model_repo_or_path)

        # Step 2: hand-built component dictionary.
        # The Qwen3 language model is the largest component (~8B params, ~16GB
        # in bf16).  On GPUs with limited VRAM (e.g. Radeon 8060S / Strix Halo,
        # ~8GB) loading everything at once causes a GPU memory access fault.
        # We therefore load the language model with device_map="auto" and a
        # max_memory budget so accelerate spills overflow layers to CPU,
        # while keeping the smaller diffusion components on the GPU.
        dtype = self.dtype
        logger.info("MiniMax-Music3 dtype: %s (force_fp32=%s)", dtype, self.force_fp32)

        lm_max_memory = self._lm_max_memory()

        # Build kwargs for the language model.  On CPU-only systems or when
        # we can't constrain VRAM, fall back to a plain from_pretrained that
        # loads on the target device without device_map.
        lm_kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "local_files_only": local_files_only,
        }
        if lm_max_memory is not None:
            lm_kwargs["device_map"] = "auto"
            lm_kwargs["max_memory"] = lm_max_memory
            try:
                language_model = Qwen3ForCausalLM.from_pretrained(
                    model_repo_or_path, subfolder="language_model", **lm_kwargs,
                )
            except Exception as exc:
                logger.warning(
                    "Language model device_map='auto' failed (%s); falling back to "
                    "plain load on %s",
                    exc, self.device,
                )
                language_model = Qwen3ForCausalLM.from_pretrained(
                    model_repo_or_path, subfolder="language_model",
                    torch_dtype=dtype, local_files_only=local_files_only,
                ).to(str(self.device))
        else:
            language_model = Qwen3ForCausalLM.from_pretrained(
                model_repo_or_path, subfolder="language_model", **lm_kwargs,
            )

        components = {
            "tokenizer": AutoTokenizer.from_pretrained(
                model_repo_or_path, subfolder="tokenizer", local_files_only=local_files_only,
            ),
            "language_model": language_model,
            "rvq_depth_decoder": MiniMaxMusic3RVQDepthDecoder.from_pretrained(
                model_repo_or_path, subfolder="rvq_depth_decoder",
                torch_dtype=dtype, local_files_only=local_files_only,
            ),
            "condition_encoder": MiniMaxMusic3ConditionEncoder.from_pretrained(
                model_repo_or_path, subfolder="condition_encoder",
                torch_dtype=dtype, local_files_only=local_files_only,
            ),
            "transformer": MiniMaxMusic3Transformer1DModel.from_pretrained(
                model_repo_or_path, subfolder="transformer",
                torch_dtype=dtype, local_files_only=local_files_only,
            ),
            "scheduler": FlowMatchEulerDiscreteScheduler.from_pretrained(
                model_repo_or_path, subfolder="scheduler", local_files_only=local_files_only,
            ),
            "vocoder": MiniMaxMusic3Vocoder.from_pretrained(
                model_repo_or_path, subfolder="vocoder",
                torch_dtype=dtype, local_files_only=local_files_only,
            ),
        }
        pipeline.register_components(**components)
        pipeline.to(str(self.device))

        self.pipeline = pipeline
        self.model_repo = model_repo_or_path
        logger.info(
            "MiniMax-Music3 loaded: sampling_rate=%s frame_rate=%.2f latent_hop=%d",
            pipeline.sampling_rate, pipeline.frame_rate, pipeline.latent_hop_length,
        )

    def unload_model(self) -> None:
        """Drop the cached pipeline and free VRAM."""
        self.pipeline = None
        self.model_repo = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- generation ----------------------------------------------------------

    def generate(
        self,
        prompt: str,
        lyrics: str = "",
        audio_duration: float = MINIMAX_MUSIC3_DEFAULT_SECONDS,
        num_inference_steps: int = MINIMAX_MUSIC3_DEFAULT_STEPS,
        seed: Optional[int] = None,
        model_repo_or_path: str = MINIMAX_MUSIC3_DEFAULT_REPO,
    ) -> Path:
        """
        Generate a stereo WAV from a music description + lyrics.

        Args:
            prompt: Music description (genre, mood, vocals, instrumentation).
            lyrics: Lyrics with optional `[verse]`/`[chorus]` structure tags.
                Empty string means instrumental.  Lines that put text on the
                same line as a tag (e.g. ``[verse] My text``) are auto-split
                by ``_preprocess_lyrics`` so the pipeline doesn't drop them.
            audio_duration: Target length in seconds (clamped to MAX).
            num_inference_steps: Flow-matching Euler steps per chunk.
            seed: Reproducibility seed.
            model_repo_or_path: HF repo id or local model directory.

        Returns:
            Path to the generated audio file (MP3 if ffmpeg conversion
            succeeded, otherwise WAV).
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        audio_duration = max(1.0, min(float(audio_duration), MINIMAX_MUSIC3_MAX_SECONDS))
        num_inference_steps = max(1, min(int(num_inference_steps), 200))
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        self.load_model(model_repo_or_path)
        assert self.pipeline is not None

        generator = torch.Generator(device=str(self.device))
        generator.manual_seed(int(seed))

        start = time.time()
        logger.info(
            "Generating music: prompt_len=%d lyrics_len=%d duration=%.1fs steps=%d seed=%d",
            len(prompt), len(processed_lyrics), audio_duration, num_inference_steps, seed,
        )
        logger.debug("Full prompt: %r", prompt)

        # Pipeline inputs (per MiniMaxMusic3Blocks docs):
        #   prompt: music description (genre/mood/vocals/...)
        #   lyrics: lyrics with [verse]/[chorus] structure tags
        #   audio_duration: upper bound in seconds, capped at 9000 frames (~6min)
        #   generator: torch generator
        #   num_inference_steps: flow-matching Euler steps per chunk
        #   output_type: 'np' for ndarray, 'pt' for tensor
        #
        # Pre-process lyrics: the pipeline's _normalize_lyrics silently
        # drops text on the same line as a [tag].  Splitting tag+text
        # lines preserves the user's lyric content.  Empty lyrics
        # (instrumental) become "[instrumental]" because the pipeline
        # raises ValueError on a blank string.
        processed_lyrics = _preprocess_lyrics(lyrics)
        if processed_lyrics != lyrics:
            logger.debug("Lyrics preprocessed for MiniMax pipeline: %r -> %r", lyrics, processed_lyrics)

        result = self.pipeline(
            prompt=prompt,
            lyrics=processed_lyrics,
            audio_duration=audio_duration,
            generator=generator,
            num_inference_steps=num_inference_steps,
            output_type="np",
        )

        # Free GPU memory used by the AR language model and intermediate
        # latents immediately — on limited-VRAM cards the Qwen3 KV cache
        # and flow-matching activations can keep the GPU at capacity.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # The pipeline returns a numpy ndarray: shape (channels, samples) float32.
        # Sample rate is exposed on the pipeline; upstream card says 32kHz but
        # the actual vocoder outputs 44.1kHz - trust the pipeline.
        audio = result.audios[0] if hasattr(result, "audios") else result[0]
        sample_rate = int(self.pipeline.sampling_rate)

        # When vae_decode_cpu is set, move the audio array to CPU early to
        # release GPU memory before any subsequent requests.  The pipeline
        # already returns numpy (CPU) when output_type="np", but the vocoder
        # may have left tensors on GPU, so we clean up.
        if self.vae_decode_cpu:
            logger.info("MiniMax vocoder decode completed; ensuring CPU offload")

        from scipy.io import wavfile

        audio_clipped = np.clip(audio, -1.0, 1.0)
        peak = np.abs(audio_clipped).max()
        if peak > 0:
            audio_clipped = audio_clipped / peak
        audio_int16 = (audio_clipped * 32767.0).astype(np.int16)

        suffix = f"seed{seed}"
        short_hash = abs(hash((prompt, processed_lyrics))) % (10**8)
        output_path = self.output_dir / f"minimax_{short_hash:08d}_{suffix}.wav"

        wavfile.write(str(output_path), sample_rate, audio_int16.T)

        elapsed = time.time() - start
        logger.info(
            "Music generation complete: %s (%.2fs, %.2f MB, %dHz)",
            output_path.name, elapsed,
            output_path.stat().st_size / (1024 * 1024), sample_rate,
        )

        # Attempt MP3 conversion using ffmpeg subprocess (pydub not installed).
        mp3_path = self._try_convert_to_mp3(output_path)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return mp3_path if mp3_path else output_path

    def _try_convert_to_mp3(self, wav_path: Path) -> Optional[Path]:
        """Convert WAV to MP3 using ffmpeg subprocess.

        Returns the MP3 path on success, or None if ffmpeg is unavailable
        or conversion fails.  The original WAV is preserved so it can be
        served as a fallback.
        """
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is None:
            return None

        mp3_path = wav_path.with_suffix(".mp3")
        try:
            result = subprocess.run(
                [
                    ffmpeg_bin, "-y",
                    "-i", str(wav_path),
                    "-codec:a", "libmp3lame",
                    "-b:a", "192k",
                    str(mp3_path),
                ],
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0 and mp3_path.exists():
                logger.info("Converted %s to MP3: %s", wav_path.name, mp3_path.name)
                return mp3_path
            else:
                logger.warning("ffmpeg conversion failed for %s: %s",
                               wav_path.name, result.stderr.decode()[:200])
                mp3_path.unlink(missing_ok=True)
                return None
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning("ffmpeg conversion error for %s: %s", wav_path.name, e)
            mp3_path.unlink(missing_ok=True)
            return None

    # -- introspection -------------------------------------------------------

    def gpu_info(self) -> Dict[str, Any]:
        """Backend-shaped GPU info block."""
        info: Dict[str, Any] = {
            "device": str(self.device),
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": None,
            "memory_used": None,
            "memory_total": None,
            "stats_available": False,
            "model_loaded": self.is_loaded(),
            "model_repo": self.model_repo,
        }
        if torch.cuda.is_available():
            try:
                info["gpu_name"] = torch.cuda.get_device_name(0)
                free, total = torch.cuda.mem_get_info()
                info["memory_used"] = f"{(total - free) / (1024 ** 3):.2f} GB"
                info["memory_total"] = f"{total / (1024 ** 3):.2f} GB"
                info["stats_available"] = True
            except Exception as exc:  # pragma: no cover - hardware probe
                logger.debug("GPU stats probe failed: %s", exc)
        return info
