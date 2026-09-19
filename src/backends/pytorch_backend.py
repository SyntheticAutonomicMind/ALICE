# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2026 Andrew Wyatt (Fewtarius)

"""
PyTorch Backend for ALICE

Image generation using PyTorch and diffusers library.
Supports CUDA (NVIDIA), ROCm (AMD), and CPU inference.
"""

import asyncio
import json
import logging
import os
import gc
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Type, List, TYPE_CHECKING

# Set TRITON_CACHE_DIR before torch import — Triton compiler caches kernels in
# this directory.  Under systemd's ProtectHome=yes, the default ~/.triton is
# /opt/alice/.triton (alice's HOME), which is owned by deck and not writable.
if "TRITON_CACHE_DIR" not in os.environ:
    for _triton_cache in (Path("/var/lib/alice/.triton"), Path("/tmp/.triton")):
        try:
            _triton_cache.mkdir(parents=True, exist_ok=True)
            if os.access(_triton_cache, os.W_OK):
                os.environ["TRITON_CACHE_DIR"] = str(_triton_cache)
                break
        except (PermissionError, OSError):
            continue

# Set torch CPU thread counts BEFORE importing torch — these are read at
# import time and cannot be changed afterwards (inter-op in particular).
_cpu_count = os.cpu_count() or 1
_torch_threads = max(1, _cpu_count // 2)
os.environ.setdefault("OMP_NUM_THREADS", str(_torch_threads))
os.environ.setdefault("MKL_NUM_THREADS", str(_torch_threads))
os.environ.setdefault("TORCH_NUM_THREADS", str(_torch_threads))

import torch
from PIL import Image

from .base import BaseBackend
from ..cancellation import CancellationToken, CancellationError

# Lazy import CompelForSDXL only when needed to avoid torchvision import at module load
if TYPE_CHECKING:
    from compel import CompelForSDXL

logger = logging.getLogger(__name__)


@dataclass
class _CachedModel:
   """Holds a cached model's pipeline and associated per-model state.

   When a model is loaded into the cache, all its state (pipeline,
   CompelForSDXL instance, compiled UNet, etc.) is stored here.
   The "active" instance variables on PyTorchBackend are kept in sync
   with whichever _CachedModel is currently most-recently-used.
   """
   pipeline: Any
   model_path: Path
   model_type: str
   device_map_active: bool
   is_single_file_sdxl: bool
   compel: Optional[Any] = None
   unet_compiled: bool = False
   original_unet: Optional[Any] = None
   loaded_at: float = field(default_factory=time.time)
   # Estimated VRAM footprint (bytes) after loading.  Used to decide
   # how many models can coexist in GPU memory without guessing.
   vram_footprint_bytes: int = 0

# AMD ROCm gfx1103 (Phoenix APU) compatibility settings
# CRITICAL: MIOPEN_DEBUG_FIND_ALL=0 prevents GPU hangs during MIOpen solver search
# This should be set in environment before MIOpen is initialized
if "MIOPEN_DEBUG_FIND_ALL" not in os.environ:
    os.environ["MIOPEN_DEBUG_FIND_ALL"] = "0"
    logger.info("Set MIOPEN_DEBUG_FIND_ALL=0 for AMD GPU compatibility")

# Disable cuDNN/MIOpen which can cause GPU hangs on gfx1103
# This forces fallback to non-accelerated convolution paths
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False
    logger.info("Disabled cuDNN/MIOpen for AMD GPU compatibility")

# Disable problematic SDPA backends for AMD GPU compatibility
# Flash and memory-efficient attention can cause GPU hangs on ROCm
if torch.cuda.is_available():
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        logger.info("Configured SDPA: disabled flash/mem_efficient, enabled math-only")
    except Exception as e:
        logger.debug("Could not configure SDPA backends: %s", e)

# Suppress noisy warnings from transformers, huggingface_hub, and torch
# These are informational warnings that clutter logs without actionable info:
# - HF cache permission denied (cache dir owned by root, writes are non-critical)
# - Token indices too long (CompelForSDXL handles truncation internally)
# - torch._sympy interp failures (Inductor internal, non-blocking)
# - Float32 module warnings (diffusers internal, no action needed)
# - Siglip2ImageProcessorFast deprecation (transformers internal)
# - PYTORCH_HIP_ALLOC_CONF deprecation (handled by alias below)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_LOG_LEVEL", "error")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Set HF_HOME before huggingface_hub is imported — it caches HF_HOME at import
# time, so we must set it here (not in the lifespan) to take effect.
# Default to the config models directory's parent cache path if available,
# otherwise fall back to a user-writable location.
if "HF_HOME" not in os.environ:
    try:
        from ..config import config as _alice_config
        _hf_cache = _alice_config.models.directory.parent / ".cache" / "huggingface"
        _hf_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(_hf_cache)
    except Exception:
        pass  # huggingface_hub will use its default (~/.cache/huggingface)

# Fix deprecated PYTORCH_HIP_ALLOC_CONF -> PYTORCH_ALLOC_CONF
# PyTorch now reads PYTORCH_ALLOC_CONF for both CUDA and HIP backends
if "PYTORCH_HIP_ALLOC_CONF" in os.environ and "PYTORCH_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_ALLOC_CONF"] = os.environ["PYTORCH_HIP_ALLOC_CONF"]
    logger.info("Migrated PYTORCH_HIP_ALLOC_CONF to PYTORCH_ALLOC_CONF")

# Suppress specific non-actionable warning messages at the Python warnings level
import warnings
warnings.filterwarnings("ignore", message=".*failed while executing.*")
warnings.filterwarnings("ignore", message=".*should be kept in float32.*")
warnings.filterwarnings("ignore", message=".*Siglip2ImageProcessorFast.*")
warnings.filterwarnings("ignore", message=".*Token indices sequence length.*")
warnings.filterwarnings("ignore", message=".*Dynamo detected.*")
warnings.filterwarnings("ignore", message=".*PYTORCH_HIP_ALLOC_CONF is deprecated.*")

# --- Suppress the same messages at the Python logging level ---
# Several of the messages above are emitted via logging.warning() (not
# warnings.warn()), so warnings.filterwarnings has no effect on them.
# Verified sources:
#   - "failed while executing"        -> torch.utils._sympy.interp (log.warning)
#   - "should be kept in float32"     -> diffusers.models.modeling_utils (logger.warning)
#   - "Token indices sequence length" -> transformers.tokenization_utils_base (logger.warning)
# Logger objects are singletons, so setting the level here takes effect even
# though the loggers are created later inside torch/diffusers/transformers.
# Using ERROR level is intentional — these modules do not emit useful
# WARNING-level messages that we would lose.
for _noisy_log_name in (
    "torch.utils._sympy.interp",
    "torch.fx.experimental.symbolic_shapes",
    "diffusers.models.modeling_utils",
    "transformers.tokenization_utils_base",
):
    logging.getLogger(_noisy_log_name).setLevel(logging.ERROR)

# Lazy imports for diffusers to speed up startup
_diffusers_imported = False
_pipeline_classes: Dict[str, Type] = {}
_scheduler_classes: Dict[str, Tuple[Type, Dict[str, Any]]] = {}


def _import_diffusers() -> None:
    """Lazy import diffusers modules."""
    global _diffusers_imported, _pipeline_classes, _scheduler_classes
    
    if _diffusers_imported:
        return
    
    logger.info("Importing diffusers library...")
    
    from diffusers import (
        DiffusionPipeline,
        StableDiffusionPipeline,
        StableDiffusionImg2ImgPipeline,
        AutoPipelineForText2Image,
        DPMSolverMultistepScheduler,
        EulerDiscreteScheduler,
        EulerAncestralDiscreteScheduler,
        DDIMScheduler,
        PNDMScheduler,
        LMSDiscreteScheduler,
    )
    
    # Try to import Qwen Image Edit pipeline (may not be available in all versions)
    try:
        from diffusers import QwenImageEditPlusPipeline
        _pipeline_classes["QwenImageEditPlusPipeline"] = QwenImageEditPlusPipeline
    except ImportError:
        logger.debug("QwenImageEditPlusPipeline not available - install latest diffusers for Qwen img2img support")
    
    # Try to import flow matching schedulers (may not be available in all versions)
    try:
        from diffusers import FlowMatchEulerDiscreteScheduler
        _scheduler_classes["flow_match_euler"] = (FlowMatchEulerDiscreteScheduler, {})
    except ImportError:
        logger.debug("FlowMatchEulerDiscreteScheduler not available")
    
    # Pipeline classes
    _pipeline_classes.update({
        "DiffusionPipeline": DiffusionPipeline,
        "StableDiffusionPipeline": StableDiffusionPipeline,
        "StableDiffusionImg2ImgPipeline": StableDiffusionImg2ImgPipeline,
        "AutoPipelineForText2Image": AutoPipelineForText2Image,
    })
    
    # Scheduler mapping
    # NOTE: DPM++ schedulers use solver_order=2 for proper second-order accuracy
    # Per-request scheduler instances (implemented below) prevent state corruption and IndexError
    _scheduler_classes.update({
        "dpm++": (DPMSolverMultistepScheduler, {
            "use_karras_sigmas": False,
            "algorithm_type": "dpmsolver++",
            "solver_order": 2,
        }),
        "dpm++_karras": (DPMSolverMultistepScheduler, {
            "use_karras_sigmas": True,
            "algorithm_type": "dpmsolver++",
            "solver_order": 2,
        }),
        "dpm++_sde": (DPMSolverMultistepScheduler, {
            "use_karras_sigmas": False,
            "algorithm_type": "sde-dpmsolver++",
            "solver_order": 2,
        }),
        "dpm++_sde_karras": (DPMSolverMultistepScheduler, {
            "use_karras_sigmas": True,
            "algorithm_type": "sde-dpmsolver++",
            "solver_order": 2,
        }),
        "euler": (EulerDiscreteScheduler, {}),
        "euler_a": (EulerAncestralDiscreteScheduler, {}),
        "euler_ancestral": (EulerAncestralDiscreteScheduler, {}),
        "ddim": (DDIMScheduler, {}),
        "pndm": (PNDMScheduler, {}),
        "lms": (LMSDiscreteScheduler, {}),
    })
    
    _diffusers_imported = True
    logger.info("Diffusers library imported successfully")


def _round_to_multiple(value: int, multiple: int = 8) -> int:
    """
    Round a value to the nearest multiple.
    
    Stable Diffusion models require dimensions divisible by 8 due to VAE downsampling.
    
    Args:
        value: Input dimension
        multiple: Multiple to round to (default 8)
        
    Returns:
        Rounded dimension
    """
    return ((value + multiple - 1) // multiple) * multiple


def _detect_pipeline_class(model_path: Path) -> Tuple[str, str]:
    """
    Detect appropriate pipeline class for the model.
    
    Args:
        model_path: Path to model file or directory
        
    Returns:
        Tuple of (pipeline_class_name, model_type)
    """
    default_pipeline = "StableDiffusionPipeline"
    default_type = "sd15"
    
    # Single file models use default pipeline
    if model_path.is_file():
        logger.debug("Single file model - using default pipeline")
        return default_pipeline, default_type
    
    # Check model_index.json for diffusers directories
    model_index_path = model_path / "model_index.json"
    if not model_index_path.exists():
        logger.debug("No model_index.json found - using default pipeline")
        return default_pipeline, default_type
    
    try:
        with open(model_index_path) as f:
            model_index = json.load(f)
        
        pipeline_class = model_index.get("_class_name", default_pipeline)
        logger.debug("Detected pipeline class: %s", pipeline_class)
        
        # Detect model type from pipeline name
        model_type = default_type
        class_lower = pipeline_class.lower()
        if "qwen" in class_lower:
            model_type = "qwen"
        elif "xl" in class_lower:
            model_type = "sdxl"
        elif "flux" in class_lower:
            model_type = "flux"
        elif "sd3" in class_lower:
            model_type = "sd3"
        
        return pipeline_class, model_type
        
    except Exception as e:
        logger.warning("Error reading model_index.json: %s", e)
        return default_pipeline, default_type


def _get_scheduler(scheduler_name: str, pipeline_config: Dict) -> Any:
    """
    Get scheduler instance from name.
    
    Args:
        scheduler_name: Name of scheduler
        pipeline_config: Pipeline scheduler config dict
        
    Returns:
        Configured scheduler instance
    """
    if scheduler_name not in _scheduler_classes:
        available = list(_scheduler_classes.keys())
        raise ValueError(f"Unknown scheduler: {scheduler_name}. Available: {available}")
    
    scheduler_class, scheduler_config = _scheduler_classes[scheduler_name]
    
    # For DPM++ schedulers, create completely fresh instance to avoid state corruption
    # DPMSolverMultistepScheduler maintains internal state (model_outputs, timesteps) that can corrupt
    if scheduler_name.startswith("dpm"):
        # Get only essential config keys, ignore any state
        config_dict = {
            "num_train_timesteps": pipeline_config.get("num_train_timesteps", 1000),
            "beta_start": pipeline_config.get("beta_start", 0.00085),
            "beta_end": pipeline_config.get("beta_end", 0.012),
            "beta_schedule": pipeline_config.get("beta_schedule", "scaled_linear"),
        }
        # Apply our workaround settings
        config_dict.update(scheduler_config)
        
        try:
            return scheduler_class.from_config(config_dict)
        except Exception as e:
            logger.warning("Failed to create DPM++ scheduler with minimal config: %s. Using defaults.", e)
            return scheduler_class(**scheduler_config)
    
    # For Euler Ancestral schedulers, apply boundary check patch
    # FIX: EulerAncestralDiscreteScheduler has IndexError bug when accessing sigmas[step_index+1] on final step
    # The scheduler tries to access self.sigmas[self.step_index + 1] without checking if it's the last step
    # This causes: IndexError: index N is out of bounds for dimension 0 with size N
    # Monkey-patch to handle the final step correctly
    if scheduler_name in ["euler_a", "euler_ancestral"]:
        # Create scheduler instance first
        config_dict = dict(pipeline_config)
        incompatible_keys = [
            "mu", "timestep_type", "rescale_betas_zero_snr", "variance_type",
            "clip_sample", "clip_sample_range", "thresholding", "dynamic_thresholding_ratio",
            "sample_max_value", "prediction_type", "steps_offset"
        ]
        for key in incompatible_keys:
            config_dict.pop(key, None)
        
        config_dict.update(scheduler_config)
        
        try:
            scheduler = scheduler_class.from_config(config_dict)
        except Exception as e:
            logger.warning("Failed to create Euler Ancestral scheduler with pipeline config: %s. Using defaults.", e)
            scheduler = scheduler_class(**scheduler_config)
        
        # Import needed for the patched step method
        from diffusers.utils.torch_utils import randn_tensor
        from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteSchedulerOutput
        
        # Store original step method
        original_step = scheduler.step
        
        def patched_step(model_output, timestep, sample, generator=None, return_dict=True):
            """
            Patched step() with boundary checking to prevent IndexError on final timestep.
            
            This fixes a bug in diffusers.EulerAncestralDiscreteScheduler where it tries to access
            self.sigmas[self.step_index + 1] without checking if step_index is at the last valid index.
            """
            # Validate inputs (copied from original)
            if isinstance(timestep, (int, torch.IntTensor, torch.LongTensor)):
                raise ValueError(
                    (
                        "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                        " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                        " one of the `scheduler.timesteps` as a timestep."
                    ),
                )

            if not scheduler.is_scale_input_called:
                logger.warning(
                    "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                    "See `StableDiffusionPipeline` for a usage example."
                )

            if scheduler.step_index is None:
                scheduler._init_step_index(timestep)

            sigma = scheduler.sigmas[scheduler.step_index]

            # Upcast to avoid precision issues when computing prev_sample
            sample = sample.to(torch.float32)

            # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            if scheduler.config.prediction_type == "epsilon":
                pred_original_sample = sample - sigma * model_output
            elif scheduler.config.prediction_type == "v_prediction":
                # * c_out + input * c_skip
                pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
            elif scheduler.config.prediction_type == "sample":
                raise NotImplementedError("prediction_type not implemented yet: sample")
            else:
                raise ValueError(
                    f"prediction_type given as {scheduler.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
                )

            sigma_from = scheduler.sigmas[scheduler.step_index]
            
            # FIX: Check boundaries before accessing step_index + 1
            # Original bug: sigma_to = scheduler.sigmas[scheduler.step_index + 1]
            # This fails when step_index == len(sigmas) - 2 (last valid step before final 0)
            if scheduler.step_index + 1 < len(scheduler.sigmas):
                sigma_to = scheduler.sigmas[scheduler.step_index + 1]
            else:
                # On final step, use the last sigma value (which should be 0)
                sigma_to = scheduler.sigmas[-1]
                logger.debug(
                    "Euler Ancestral final step: using sigma_to=%.6f (last element), step_index=%d/%d",
                    sigma_to, scheduler.step_index, len(scheduler.sigmas) - 1
                )

            sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
            sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5

            # 2. Convert to an ODE derivative
            derivative = (sample - pred_original_sample) / sigma

            dt = sigma_down - sigma

            prev_sample = sample + derivative * dt

            device = model_output.device
            noise = randn_tensor(model_output.shape, dtype=model_output.dtype, device=device, generator=generator)
            prev_sample = prev_sample + noise * sigma_up

            # Cast sample back to model compatible dtype
            prev_sample = prev_sample.to(model_output.dtype)

            # upon completion increase step index by one
            scheduler._step_index += 1

            if not return_dict:
                return (
                    prev_sample,
                    pred_original_sample,
                )

            return EulerAncestralDiscreteSchedulerOutput(
                prev_sample=prev_sample, pred_original_sample=pred_original_sample
            )
        
        # Replace step method with patched version
        scheduler.step = patched_step
        logger.info("Applied boundary check patch to EulerAncestralDiscreteScheduler (fixes IndexError on final step)")
        return scheduler
    
    # For other schedulers, use existing logic
    config_dict = dict(pipeline_config)
    incompatible_keys = [
        "mu", "timestep_type", "rescale_betas_zero_snr", "variance_type",
        "clip_sample", "clip_sample_range", "thresholding", "dynamic_thresholding_ratio",
        "sample_max_value", "prediction_type", "steps_offset"
    ]
    for key in incompatible_keys:
        config_dict.pop(key, None)
    
    config_dict.update(scheduler_config)
    
    try:
        return scheduler_class.from_config(config_dict)
    except Exception as e:
        logger.warning("Failed to create scheduler with pipeline config: %s. Using defaults.", e)
        return scheduler_class(**scheduler_config)


class PyTorchBackend(BaseBackend):
    """
    PyTorch backend for image generation.
    
    Supports CUDA (NVIDIA), ROCm (AMD), and CPU inference using
    PyTorch and the diffusers library.
    """
    
    def __init__(
        self,
        images_dir: Path,
        default_steps: int = 25,
        default_guidance_scale: float = 7.5,
        default_scheduler: str = "dpm++_sde_karras",
        default_width: int = 512,
        default_height: int = 512,
        force_cpu: bool = False,
        device_map: Optional[str] = None,
        force_float32: bool = False,
        force_bfloat16: bool = False,
        enable_vae_slicing: bool = True,
        enable_vae_tiling: bool = False,
        enable_model_cpu_offload: bool = False,
        enable_sequential_cpu_offload: bool = False,
        attention_slice_size: Optional[str] = "auto",
        vae_decode_cpu: bool = False,
        enable_torch_compile: bool = False,
        torch_compile_mode: str = "reduce-overhead",
        max_concurrent_generations: int = 1,
        max_cached_models: int = 2,
        vram_evict_threshold_gb: float = 2.0,
        max_cpu_cached_models: int = 8,
    ):
        """
        Initialize generator service.
        
        Args:
            images_dir: Directory to save generated images
            default_steps: Default inference steps
            default_guidance_scale: Default guidance scale
            default_scheduler: Default scheduler name
            default_width: Default image width
            default_height: Default image height
            force_cpu: Force CPU mode even if GPU is available
            device_map: Device map for model loading (e.g., 'balanced')
            force_float32: Force float32 dtype (required for some AMD GPUs)
            force_bfloat16: Force bfloat16 dtype (better for AMD Phoenix APU)
            enable_vae_slicing: Enable VAE slicing for lower memory usage
            enable_vae_tiling: Enable VAE tiling for very large images
            enable_model_cpu_offload: Enable model CPU offload (slower but uses less VRAM)
            enable_sequential_cpu_offload: Enable sequential CPU offload (slowest, minimum VRAM)
            attention_slice_size: Attention slice size: 'auto', 'max', or a number
            vae_decode_cpu: Decode VAE on CPU (fixes GPU hang on AMD gfx1103)
            enable_torch_compile: Enable torch.compile for UNet (PyTorch 2.0+)
            torch_compile_mode: Torch compile mode ('default', 'reduce-overhead', 'max-autotune')
            max_concurrent_generations: Maximum concurrent generation requests
            max_cached_models: Maximum models to keep in GPU memory simultaneously (LRU eviction)
            vram_evict_threshold_gb: Free VRAM (GB) below which cached models are evicted (0=disabled)
            max_cpu_cached_models: Max models to keep in CPU RAM after VRAM eviction
        """
        self.images_dir = Path(images_dir)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        self.default_steps = default_steps
        self.default_guidance_scale = default_guidance_scale
        self.default_scheduler = default_scheduler
        self.default_width = default_width
        self.default_height = default_height
        self.force_cpu = force_cpu
        self.device_map = device_map
        self.force_float32 = force_float32
        self.force_bfloat16 = force_bfloat16
        
        # Memory optimization settings
        self.enable_vae_slicing = enable_vae_slicing
        self.enable_vae_tiling = enable_vae_tiling
        self.enable_model_cpu_offload = enable_model_cpu_offload
        self.enable_sequential_cpu_offload = enable_sequential_cpu_offload
        self.attention_slice_size = attention_slice_size
        self.vae_decode_cpu = vae_decode_cpu
        
        # Performance optimization settings
        self.enable_torch_compile = enable_torch_compile
        self.torch_compile_mode = torch_compile_mode
        
        # Model caching - multi-model LRU cache
        # self._pipeline and friends are the "active" model's state, kept in sync
        # with the most-recently-used entry in self._model_cache
        self._pipeline: Optional[Any] = None  # Active pipeline (MRU cached model)
        self._current_model: Optional[str] = None  # Active model path
        self._current_model_path: Optional[Path] = None
        self._device_map_active: bool = False  # Track if device_map was applied to current model
        self._is_single_file_sdxl: bool = False  # Track if loaded model is single-file SDXL (needs extra memory opts)
        self._compel: Optional[CompelForSDXL] = None  # CompelForSDXL instance for long prompt support
        self._unet_compiled: bool = False  # Track if UNet has been compiled
        self._original_unet: Optional[Any] = None  # Eager UNet kept for runtime Inductor fallback
        self._current_model_type: Optional[str] = None  # For Qwen support later
        
        # Multi-model LRU cache: str(model_path) -> _CachedModel
        # OrderedDict maintains insertion order; move_to_end() marks as MRU
        self._model_cache: "OrderedDict[str, _CachedModel]" = OrderedDict()
        self._max_cached_models: int = max_cached_models
        self._vram_evict_threshold_gb: float = vram_evict_threshold_gb
        
        # CPU-offload cache: models evicted from VRAM are stored in CPU RAM
        # for fast reload without hitting disk.  On systems with 112GB of RAM,
        # keeping several full pipelines in CPU memory is trivial.
        self._cpu_cache: "OrderedDict[str, _CachedModel]" = OrderedDict()
        self._max_cpu_cached_models: int = max_cpu_cached_models
        
        # Periodic gc counter — full gc.collect() runs every N generations,
        # not every generation, to avoid blocking the event loop.
        self._gc_counter: int = 0
        self._gc_interval: int = 5
        
        # Concurrency control
        self._model_lock: asyncio.Lock = asyncio.Lock()  # Protects model loading operations
        self._generation_semaphore: asyncio.Semaphore = asyncio.Semaphore(max_concurrent_generations)
        self._max_concurrent = max_concurrent_generations
        
        # Request queue tracking
        self._pending_requests: int = 0  # Track number of queued requests
        self._queue_lock: asyncio.Lock = asyncio.Lock()  # Protects _pending_requests counter
        
        # Device detection
        self._device = self._detect_device()

        # CPU thread pool sizing — PyTorch defaults to 1 intra-op thread which
        # leaves 31 cores idle on a 32-thread CPU.  We use a fraction of the
        # available cores to speed up CPU-bound portions of generation (VAE
        # decode on CPU, latent init, post-processing) without starving the
        # OS or other process.
        self._configure_torch_threads()

        # Statistics
        self.total_generations = 0
        self.total_generation_time = 0.0
        
        logger.info(
            "Generator initialized: device=%s, images_dir=%s, max_concurrent=%d, vae_slicing=%s, vae_tiling=%s",
            self._device, self.images_dir, max_concurrent_generations, self.enable_vae_slicing, self.enable_vae_tiling
        )
    
    def _detect_device(self) -> str:
        """Detect best available compute device."""
        if self.force_cpu:
            logger.info("CPU mode forced by configuration")
            return "cpu"
        
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            logger.info("Using CUDA device: %s", gpu_name)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using Apple Silicon MPS device")
        else:
            device = "cpu"
            logger.info("Using CPU device")
        
        return device

    def _configure_torch_threads(self) -> None:
        """Configure PyTorch's CPU thread pools for multi-core systems.

        PyTorch defaults to 1 intra-op thread — on a 32-core CPU this
        leaves 31 cores idle during CPU-bound generation steps (VAE decode
        on CPU, latent sampling, image post-processing).  We use half the
        available cores for intra-op and reserve some for inter-op.

        On GPU systems the GPU does most of the heavy lifting, but CPU
        thread parallelism still helps for VAE decode on CPU and for
        torch.compile/Inductor codegen.
        """
        try:
            cpu_count = os.cpu_count() or 1
            # Use half the cores for intra-op (model parallelism within
            # a single operation, e.g. matmul), leave the rest for system
            # overhead and inter-op parallelism.
            intra_threads = max(1, cpu_count // 2)
            inter_threads = max(1, min(4, cpu_count))

            current_intra = torch.get_num_threads()
            current_inter = torch.get_num_interop_threads()

            if intra_threads != current_intra or inter_threads != current_inter:
                torch.set_num_threads(intra_threads)
                # set_num_interop_threads must be called before any parallel
                # work; it's a no-op if the pool is already initialized
                try:
                    torch.set_num_interop_threads(inter_threads)
                except RuntimeError:
                    pass  # Already initialized — accept the default

                logger.info(
                    "Configured torch threads: intra=%d (was %d), inter=%d (was %d) "
                    "[cpu_count=%d]",
                    intra_threads, current_intra, inter_threads, current_inter, cpu_count
                )
        except Exception as e:
            logger.debug("Thread configuration skipped: %s", e)

    @property
    def current_model(self) -> Optional[str]:
        """Get currently active model path (most recently used)."""
        return self._current_model
    
    @property
    def is_model_loaded(self) -> bool:
        """Check if any model is currently loaded (cached)."""
        return len(self._model_cache) > 0
    
    @property
    def loaded_models(self) -> List[str]:
        """Get list of all cached model paths (most recently used first)."""
        return list(reversed(self._model_cache.keys()))

    @property
    def cpu_cached_models(self) -> List[str]:
        """Get list of models warm in CPU RAM (evicted from VRAM)."""
        return list(reversed(self._cpu_cache.keys()))
    
    def get_queue_depth(self) -> int:
        """Get current number of pending generation requests."""
        return self._pending_requests
    
    def get_active_generations(self) -> int:
        """Get number of currently executing generations."""
        # Number of currently executing = total pending - waiting in queue
        # Since we have a semaphore limiting concurrent generations, active = min(pending, max_concurrent)
        if self._pending_requests == 0:
            return 0
        return min(self._pending_requests, self._max_concurrent)
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information with ROCm/AMD support."""
        info = {
            "device": self._device,
            "gpu_available": self._device in ("cuda", "mps"),
            "memory_used": "0 GB",
            "memory_total": "0 GB",
            "utilization": 0.0,
            "stats_available": False,  # Whether detailed stats are available
        }
        
        if self._device == "cuda":
            # Check if this is ROCm (AMD) pretending to be CUDA
            try:
                props = torch.cuda.get_device_properties(0)
                device_name = props.name.lower()
                is_amd = "radeon" in device_name or "amd" in device_name or "gfx" in device_name
                
                if is_amd:
                    # AMD GPU with ROCm - use rocm-smi for accurate stats
                    amd_stats = self._get_amd_gpu_stats()
                    if amd_stats:
                        info.update(amd_stats)
                        info["gpu_name"] = props.name
                        info["stats_available"] = True
                    else:
                        # Fallback to torch memory (less accurate for utilization)
                        allocated = torch.cuda.memory_allocated(0)
                        total = props.total_memory
                        info["memory_used"] = f"{allocated / (1024**3):.1f} GB"
                        info["memory_total"] = f"{total / (1024**3):.1f} GB"
                        info["utilization"] = allocated / total if total > 0 else 0.0
                        info["gpu_name"] = props.name
                        info["stats_available"] = True
                else:
                    # NVIDIA GPU - use torch stats
                    allocated = torch.cuda.memory_allocated(0)
                    total = props.total_memory
                    info["memory_used"] = f"{allocated / (1024**3):.1f} GB"
                    info["memory_total"] = f"{total / (1024**3):.1f} GB"
                    info["utilization"] = allocated / total if total > 0 else 0.0
                    info["gpu_name"] = props.name
                    info["stats_available"] = True
            except Exception as e:
                logger.warning("Failed to get CUDA/ROCm info: %s", e)
        elif self._device == "mps":
            # MPS (Apple Silicon) - PyTorch doesn't expose detailed memory stats yet
            info["gpu_name"] = "Apple Silicon (MPS)"
            info["stats_available"] = False
            logger.debug("MPS device detected - detailed stats not available")
        
        return info
    
    def _get_amd_gpu_stats(self) -> Optional[Dict[str, Any]]:
        """Get AMD GPU statistics using rocm-smi or sysfs fallback."""
        import subprocess
        import re
        from pathlib import Path
        
        # Try rocm-smi first (most accurate)
        try:
            result = subprocess.run(
                ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--json"],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                
                # Parse rocm-smi JSON output
                # Format varies by version, try common structures
                stats = {}
                
                # Try to find GPU 0 stats
                if isinstance(data, dict):
                    gpu_data = data.get("card0") or data.get("0") or next(iter(data.values()), {})
                    
                    # GPU utilization
                    gpu_use = gpu_data.get("GPU use (%)", gpu_data.get("use", 0))
                    if isinstance(gpu_use, str):
                        gpu_use = float(gpu_use.strip("%"))
                    stats["utilization"] = float(gpu_use) / 100.0
                    
                    # Memory info
                    vram = gpu_data.get("VRAM Total Memory (B)", gpu_data.get("vram_total"))
                    vram_used = gpu_data.get("VRAM Total Used Memory (B)", gpu_data.get("vram_used"))
                    
                    if vram:
                        stats["memory_total"] = f"{int(vram) / (1024**3):.1f} GB"
                    if vram_used:
                        stats["memory_used"] = f"{int(vram_used) / (1024**3):.1f} GB"
                    
                    if stats:
                        return stats
                        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.debug("rocm-smi failed, trying sysfs fallback: %s", e)
        except Exception as e:
            # Catch JSONDecodeError and other parsing errors (json may not be imported if rocm-smi not found)
            logger.debug("rocm-smi parsing failed, trying sysfs fallback: %s", e)
        
        # Fallback to sysfs (/sys/class/drm)
        try:
            drm_path = Path("/sys/class/drm")
            
            # Find AMD GPU card
            for card_dir in sorted(drm_path.glob("card*")):
                if not card_dir.is_dir():
                    continue
                
                device_dir = card_dir / "device"
                if not device_dir.exists():
                    continue
                
                # Check if it's an AMD GPU
                vendor_file = device_dir / "vendor"
                if vendor_file.exists():
                    vendor = vendor_file.read_text().strip()
                    if vendor != "0x1002":  # AMD vendor ID
                        continue
                
                stats = {}
                
                # GPU utilization
                gpu_busy_file = device_dir / "gpu_busy_percent"
                if gpu_busy_file.exists():
                    try:
                        busy_percent = int(gpu_busy_file.read_text().strip())
                        stats["utilization"] = busy_percent / 100.0
                    except (ValueError, IOError):
                        pass
                
                # Memory usage — on AMD APUs the "VRAM" is shared system RAM.
                # sysfs reports the stolen VDRAM separately from GTT; the
                # real total available to the GPU is vram_total + gtt_total.
                mem_vram_used = device_dir / "mem_info_vram_used"
                mem_vram_total = device_dir / "mem_info_vram_total"
                mem_gtt_used = device_dir / "mem_info_gtt_used"
                mem_gtt_total = device_dir / "mem_info_gtt_total"

                vram_used = vram_total = gtt_used = gtt_total = 0
                has_vram = mem_vram_used.exists() and mem_vram_total.exists()
                has_gtt = mem_gtt_used.exists() and mem_gtt_total.exists()

                if has_vram:
                    try:
                        vram_used = int(mem_vram_used.read_text().strip())
                        vram_total = int(mem_vram_total.read_text().strip())
                    except (ValueError, IOError):
                        pass
                if has_gtt:
                    try:
                        gtt_used = int(mem_gtt_used.read_text().strip())
                        gtt_total = int(mem_gtt_total.read_text().strip())
                    except (ValueError, IOError):
                        pass

                # Combine VRAM + GTT for the total addressable GPU memory.
                # On APs, this is the sum of dedicated VRAM and shared
                # system RAM (GTT).  If only one is available, use that.
                total = vram_total + gtt_total
                used = vram_used + gtt_used
                if total > 0:
                    stats["memory_used"] = f"{used / (1024**3):.1f} GB"
                    stats["memory_total"] = f"{total / (1024**3):.1f} GB"
                    if "utilization" not in stats and total > 0:
                        stats["utilization"] = min(1.0, used / total) if total > 0 else 0.0
                
                if stats:
                    return stats
                    
        except Exception as e:
            logger.debug("sysfs GPU stats failed: %s", e)
        
        return None
    
    def _apply_memory_optimizations(self) -> None:
        """Apply memory optimization settings to the loaded pipeline."""
        if self._pipeline is None:
            return
        
        optimizations_applied = []
        
        # Sequential CPU offload (most aggressive, takes precedence)
        if self.enable_sequential_cpu_offload:
            try:
                self._pipeline.enable_sequential_cpu_offload()
                optimizations_applied.append("sequential_cpu_offload")
            except Exception as e:
                logger.warning("Could not enable sequential CPU offload: %s", e)
        # Model CPU offload (less aggressive than sequential)
        elif self.enable_model_cpu_offload:
            try:
                self._pipeline.enable_model_cpu_offload()
                optimizations_applied.append("model_cpu_offload")
            except Exception as e:
                logger.warning("Could not enable model CPU offload: %s", e)
        
        # Attention slicing (reduces memory during attention layers)
        # Only enable if explicitly configured by user
        if self.attention_slice_size:
            try:
                if self.attention_slice_size == "auto":
                    self._pipeline.enable_attention_slicing()
                    optimizations_applied.append(f"attention_slicing(auto)")
                elif self.attention_slice_size == "max":
                    self._pipeline.enable_attention_slicing(slice_size="max")
                    optimizations_applied.append(f"attention_slicing(max)")
                else:
                    # Try to parse as integer
                    try:
                        slice_size = int(self.attention_slice_size)
                        self._pipeline.enable_attention_slicing(slice_size=slice_size)
                        optimizations_applied.append(f"attention_slicing({slice_size})")
                    except ValueError:
                        self._pipeline.enable_attention_slicing()
                        optimizations_applied.append(f"attention_slicing(auto)")
            except Exception as e:
                logger.warning("Could not enable attention slicing: %s", e)
        
        # VAE slicing (reduces memory during decode)
        # Note: Some pipeline types (e.g. SDXL single-file) don't have
        # enable_vae_slicing() on the pipeline itself - check VAE subcomponent.
        if self.enable_vae_slicing:
            if hasattr(self._pipeline, 'enable_vae_slicing'):
                self._pipeline.enable_vae_slicing()
                optimizations_applied.append("vae_slicing")
            elif hasattr(self._pipeline, 'vae') and hasattr(self._pipeline.vae, 'enable_slicing'):
                self._pipeline.vae.enable_slicing()
                optimizations_applied.append("vae_slicing")
            else:
                logger.debug("VAE slicing not supported on this pipeline type")
        
        # VAE tiling (for very large images)
        if self.enable_vae_tiling:
            if hasattr(self._pipeline, 'enable_vae_tiling'):
                self._pipeline.enable_vae_tiling()
                optimizations_applied.append("vae_tiling")
            elif hasattr(self._pipeline, 'vae') and hasattr(self._pipeline.vae, 'enable_tiling'):
                self._pipeline.vae.enable_tiling()
                optimizations_applied.append("vae_tiling")
            else:
                logger.debug("VAE tiling not supported on this pipeline type")
        
        # Torch compile (PyTorch 2.0+ performance optimization)
        if self.enable_torch_compile and not self._unet_compiled:
            try:
                import torch
                import shutil
                if not shutil.which("g++") and not shutil.which("gcc") and not shutil.which("cc"):
                    logger.warning("torch.compile skipped: no C/C++ compiler found (install gcc/g++ or disable torch.compile)")
                elif hasattr(torch, 'compile'):
                    # SDXL's UNet produces kernels that torch._inductor cannot split for CUDA graph
                    # capture under 'reduce-overhead' mode (raises CantSplit on the 2560-dim bottleneck).
                    # Auto-fallback to 'default' mode to avoid the failure.
                    effective_mode = self.torch_compile_mode
                    if self._current_model_type == "sdxl" and effective_mode == "reduce-overhead":
                        logger.warning(
                            "torch_compile_mode='reduce-overhead' is incompatible with SDXL models "
                            "(Inductor raises 'CantSplit' on the 2560-dim bottleneck). "
                            "Auto-fallback to mode='default' for this model. "
                            "Set torch_compile_mode='default' in config.yaml to silence this warning."
                        )
                        effective_mode = "default"

                    logger.info("Compiling UNet with torch.compile (mode=%s). First run will be slower...", effective_mode)
                    # Preserve eager UNet for runtime Inductor fallback. SDXL's 2560-dim bottleneck
                    # can trigger CantSplit during Inductor's codegen on first execution, not at
                    # compile-wrap time. If that happens, we restore the eager UNet and retry.
                    self._original_unet = self._pipeline.unet
                    try:
                        # Reduce Inductor symbolic-shape recursion depth on complex UNets.
                        # SDXL's unet has many dynamic shape dimensions; the default
                        # max_rotation_size triggers RecursionError in _fast_expand which
                        # is slow but non-fatal.  Disabling rotation helps on AMD ROCm.
                        _inductor_cfg = getattr(torch, "_inductor", None)
                        if _inductor_cfg is not None:
                            _cfg = _inductor_cfg.config
                            # These reduce symbolic-shape analysis overhead on
                            # complex UNets (e.g. SDXL's 2560-dim bottleneck).
                            # set_float32_fallback and others may not exist
                            # across all torch versions — guard each.
                            for _key, _val in [
                                ("max_rotation_size", 128),
                                ("coordinate_descent_tuning", False),
                            ]:
                                if hasattr(_cfg, _key):
                                    try:
                                        setattr(_cfg, _key, _val)
                                    except Exception:
                                        pass
                            _triton = getattr(_cfg, "triton", None)
                            if _triton is not None and hasattr(_triton, "cudasim"):
                                try:
                                    _triton.cudasim = False
                                except Exception:
                                    pass
                        self._pipeline.unet = torch.compile(
                            self._pipeline.unet,
                            mode=effective_mode,
                            fullgraph=False,  # Allow graph breaks for compatibility
                        )
                    except Exception as compile_error:
                        # Map known Inductor failures to a friendly message and proceed
                        # without torch.compile rather than failing the whole model load.
                        error_msg = str(compile_error)
                        if "CantSplit" in error_msg or "Inductor" in error_msg or "torch._dynamo" in error_msg:
                            logger.error(
                                "torch.compile failed: %s. "
                                "Common cause: SDXL + reduce-overhead mode. "
                                "Fix: set torch_compile_mode='default' or enable_torch_compile=false in config.yaml. "
                                "Continuing without torch.compile for this model.",
                                error_msg.split("\n", 1)[0]  # First line only
                            )
                        else:
                            logger.warning("Could not compile UNet: %s", error_msg)
                    else:
                        self._unet_compiled = True
                        optimizations_applied.append(f"torch_compile({effective_mode})")
                        logger.info("UNet compilation complete. Subsequent generations will be faster.")
                else:
                    logger.warning("torch.compile not available (requires PyTorch 2.0+)")
            except Exception as e:
                logger.warning("Could not compile UNet: %s", e)
        
        if optimizations_applied:
            logger.info("Memory optimizations enabled: %s", ", ".join(optimizations_applied))
    
    def _get_free_vram_bytes(self) -> Optional[int]:
        """Get free VRAM in bytes.
        
        Returns None if not on GPU or unable to determine.
        """
        if self._device != "cuda":
            return None
        try:
            free, _total = torch.cuda.mem_get_info()
            return free
        except Exception:
            pass
        # Fallback: try rocm-smi for ROCm
        try:
            import subprocess
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--JSON"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                gpu_data = data.get("card0") or next(iter(data.values()), {})
                vram_total = gpu_data.get("VRAM Total Memory (B)", 0)
                vram_used = gpu_data.get("VRAM Total Used Memory (B)", 0)
                return int(vram_total) - int(vram_used)
        except Exception:
            pass
        # Fallback: sysfs — combine VRAM + GTT for APUs
        try:
            drm_path = Path("/sys/class/drm")
            for card_dir in sorted(drm_path.glob("card*")):
                device_dir = card_dir / "device"
                vendor_file = device_dir / "vendor"
                if vendor_file.exists() and vendor_file.read_text().strip() == "0x1002":
                    vram_used = gtt_used = 0
                    vram_total = gtt_total = 0
                    for label, attr in [("vram_used", "mem_info_vram_used"),
                                        ("vram_total", "mem_info_vram_total"),
                                        ("gtt_used", "mem_info_gtt_used"),
                                        ("gtt_total", "mem_info_gtt_total")]:
                        f = device_dir / attr
                        if f.exists():
                            try:
                                val = int(f.read_text().strip())
                                if label == "vram_used": vram_used = val
                                elif label == "vram_total": vram_total = val
                                elif label == "gtt_used": gtt_used = val
                                elif label == "gtt_total": gtt_total = val
                            except (ValueError, IOError):
                                pass
                    # Total available = vram_total + gtt_total; used = vram_used + gtt_used
                    total = vram_total + gtt_total
                    used = vram_used + gtt_used
                    if total > 0:
                        return total - used
        except Exception:
            pass
        return None
    
    def _should_evict_vram(self) -> bool:
        """Check if free VRAM is below the eviction threshold."""
        if self._vram_evict_threshold_gb <= 0:
            return False
        free = self._get_free_vram_bytes()
        if free is None:
            return False
        threshold_bytes = int(self._vram_evict_threshold_gb * (1024 ** 3))
        return free < threshold_bytes
    
    def _evict_lru(self, count: int = 1, move_to_cpu: bool = True) -> int:
        """Evict least-recently-used models from the GPU VRAM cache.

        When ``move_to_cpu`` is True (the default) evicted models are
        offloaded to CPU RAM instead of being deleted, so they can be
        quickly reloaded to VRAM without re-reading from disk.  On systems
        with ample system RAM (e.g. 112 GB APU) this keeps many models
        warm in CPU memory while only the active model occupies VRAM.

        Args:
            count: Number of LRU models to evict (may evict fewer).
            move_to_cpu: If True, store evicted models in the CPU cache
                instead of deleting them outright.

        Returns:
            Number of models actually evicted from VRAM.
        """
        evicted = 0
        for _ in range(min(count, len(self._model_cache))):
            # OrderedDict: first item is LRU
            lru_key, lru_model = self._model_cache.popitem(last=False)
            logger.info("Evicting LRU cached model: %s", lru_model.model_path.name)

            if move_to_cpu:
                self._move_model_to_cpu(lru_key, lru_model)
            else:
                del lru_model.pipeline
                if lru_model.compel is not None:
                    del lru_model.compel
                    lru_model.compel = None

            evicted += 1

        if evicted > 0 and self._device == "cuda":
            # Sync + empty_cache in a thread to avoid blocking the event
            # loop on AMD ROCm (HIP allocator can deadlock).
            def _evict_cleanup():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning("empty_cache failed during eviction: %s", e)
            try:
                asyncio.create_task(asyncio.to_thread(_evict_cleanup))
            except RuntimeError:
                pass

        # If we evicted the active model, set a new active one
        if self._current_model not in self._model_cache and len(self._model_cache) > 0:
            self._set_active(next(reversed(self._model_cache)))
        elif self._current_model not in self._model_cache:
            self._pipeline = None
            self._current_model = None
            self._current_model_path = None
            self._current_model_type = None
            self._device_map_active = False
            self._is_single_file_sdxl = False
            self._compel = None
            self._unet_compiled = False
            self._original_unet = None

        return evicted

    def _move_model_to_cpu(self, model_key: str, cached: _CachedModel) -> None:
        """Offload a cached model's pipeline to CPU RAM.

        The pipeline is moved to CPU, freeing GPU VRAM while keeping the
        weights in system RAM for fast reload.  On systems with ample RAM
        (e.g. 112 GB APU) this lets dozens of models stay warm.
        """
        logger.info("Offloading model to CPU RAM: %s", cached.model_path.name)
        try:
            # Move pipeline to CPU — free GPU memory while keeping weights
            # in system RAM for fast reload.
            cached.pipeline = cached.pipeline.to("cpu")
            # Free compel and compiled unet references (GPU-specific)
            if cached.compel is not None:
                del cached.compel
                cached.compel = None
            if cached.original_unet is not None:
                del cached.original_unet
                cached.original_unet = None
            cached.unet_compiled = False
            cached.vram_footprint_bytes = 0  # No longer in VRAM
        except Exception as e:
            logger.warning("Failed to offload %s to CPU: %s. Deleting instead.",
                           cached.model_path.name, e)
            del cached.pipeline
            if cached.compel is not None:
                del cached.compel
                cached.compel = None
            cached.pipeline = None
            return

        # Enforce CPU cache size limit
        self._cpu_cache[model_key] = cached
        self._cpu_cache.move_to_end(model_key)
        while len(self._cpu_cache) > self._max_cpu_cached_models:
            cpu_key, cpu_model = self._cpu_cache.popitem(last=False)
            logger.info("Evicting CPU-cached model (CPU cache full): %s",
                        cpu_model.model_path.name)
            if cpu_model.pipeline is not None:
                del cpu_model.pipeline

    def _move_model_to_gpu(self, model_key: str) -> bool:
        """Attempt to reload a CPU-cached model into GPU VRAM.

        Returns True if the model was found in the CPU cache and moved to GPU,
        False if it was not in the CPU cache (caller should load from disk).
        """
        if model_key not in self._cpu_cache:
            return False

        cached = self._cpu_cache.pop(model_key)
        logger.info("Reloading CPU-cached model to GPU: %s", cached.model_path.name)
        try:
            cached.pipeline = cached.pipeline.to(self._device)
            cached.vram_footprint_bytes = self._estimate_vram_footprint(cached.pipeline)
        except Exception as e:
            logger.warning("Failed to reload %s to GPU: %s. Will load from disk.",
                           cached.model_path.name, e)
            del cached.pipeline
            return False

        return True
    
    def _set_active(self, model_key: str) -> None:
        """Set the active model state from the cache.
        
        Copies the cached model's fields to the active instance variables
        so that generate_image() and _apply_memory_optimizations() work
        with the most-recently-used model.
        """
        cached = self._model_cache[model_key]
        self._pipeline = cached.pipeline
        self._current_model = model_key
        self._current_model_path = cached.model_path
        self._current_model_type = cached.model_type
        self._device_map_active = cached.device_map_active
        self._is_single_file_sdxl = cached.is_single_file_sdxl
        self._compel = cached.compel
        self._unet_compiled = cached.unet_compiled
        self._original_unet = cached.original_unet
    
    def _sync_active_to_cache(self, model_key: str) -> None:
        """Sync active instance variables back to the cached model entry.
        
        Called after _apply_memory_optimizations() and CompelForSDXL initialization,
        which may have modified _unet_compiled, _original_unet, and _compel.
        """
        cached = self._model_cache[model_key]
        cached.compel = self._compel
        cached.unet_compiled = self._unet_compiled
        cached.original_unet = self._original_unet
        cached.loaded_at = time.time()

    def _estimate_vram_footprint(self, pipeline: Any) -> int:
        """Estimate the VRAM footprint of a loaded pipeline in bytes."""
        if self._device == "cuda":
            try:
                return torch.cuda.memory_allocated()
            except Exception:
                pass
        return 0

    def _estimate_model_vram(self, model_path: Path, dtype: Any) -> int:
        """Estimate VRAM needed for a model based on file size and dtype.

        Falls back to a heuristic when the model isn't loaded yet.
        File size ≈ weight size; VRAM footprint ≈ 1.3x (VAE, scheduler,
        activation buffers).
        """
        if model_path.is_file():
            file_bytes = model_path.stat().st_size
            return int(file_bytes * 1.3)
        elif model_path.is_dir():
            total = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
            return int(total * 1.3)
        return 0

    async def _cleanup_after_generation(self) -> None:
        """Non-blocking GPU memory cleanup after a generation.

        On AMD ROCm, both gc.collect() and torch.cuda.empty_cache() can
        block the event loop for seconds or deadlock against the HIP
        command processor when called synchronously.  Runs cleanup in a
        thread pool with a synchronization barrier and timeout.

        gc.collect() runs only every _gc_interval generations;
        torch.cuda.empty_cache() runs every time.
        """
        self._gc_counter += 1
        do_full_gc = (self._gc_counter % self._gc_interval == 0)

        if self._device != "cuda":
            if do_full_gc:
                await asyncio.to_thread(gc.collect)
            return

        def _cleanup_sync():
            try:
                torch.cuda.synchronize()
            except Exception as e:
                logger.debug("GPU sync failed during cleanup: %s", e)
            if do_full_gc:
                gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning("torch.cuda.empty_cache() failed: %s", e)

        try:
            await asyncio.wait_for(
                asyncio.to_thread(_cleanup_sync), timeout=30.0
            )
        except asyncio.TimeoutError:
            logger.warning("GPU cleanup timed out after 30s, skipping")

    async def load_model(self, model_path: Path) -> None:
        """
        Load a model into memory.

        If the model is already cached, this is a fast no-op (just marks it
        as most-recently-used). If the model cache is full or VRAM is low,
        least-recently-used models are evicted before loading.

        Uses model_lock to ensure only one model load happens at a time.

        Args:
            model_path: Path to model file or directory
        """
        model_path = Path(model_path)
        model_key = str(model_path)

        # Fast path (no lock needed for read): check if already cached
        if model_key in self._model_cache:
            # Move to MRU position
            self._model_cache.move_to_end(model_key)
            self._set_active(model_key)
            logger.debug("Model already cached: %s", model_key)
            return

        # Slow path: need to load from disk or CPU cache
        # Use model_lock to serialize model loading across all requests
        async with self._model_lock:
            # Check again inside lock (another request might have loaded it)
            if model_key in self._model_cache:
                self._model_cache.move_to_end(model_key)
                self._set_active(model_key)
                logger.debug("Model already cached by concurrent request: %s", model_key)
                return

            # Check CPU cache — a previously evicted model may be warm in RAM
            if model_key in self._cpu_cache:
                logger.info("Model found in CPU cache, reloading to GPU: %s", model_key)
                cached_cpu = self._cpu_cache.pop(model_key)
                try:
                    cached_cpu.pipeline = cached_cpu.pipeline.to(self._device)
                    cached_cpu.vram_footprint_bytes = self._estimate_vram_footprint(cached_cpu.pipeline)
                    self._model_cache[model_key] = cached_cpu
                    self._set_active(model_key)
                    self._apply_memory_optimizations()
                    if self._current_model_type == "sdxl":
                        try:
                            from compel import CompelForSDXL
                            self._compel = CompelForSDXL(self._pipeline, device=self._device)
                        except Exception as e:
                            logger.warning("Failed to re-init CompelForSDXL: %s", e)
                    self._sync_active_to_cache(model_key)
                    logger.info("Model reloaded from CPU cache: %s", model_key)
                    return
                except Exception as e:
                    logger.warning("CPU->GPU reload failed for %s: %s. Loading from disk.",
                                   model_path.name, e)
                    self._cpu_cache.pop(model_key, None)
                    if cached_cpu.pipeline is not None:
                        del cached_cpu.pipeline

            logger.info("Loading model: cache_size=%d, max=%d, requested=%s",
                        len(self._model_cache), self._max_cached_models, model_key)

            # Estimate VRAM needed for the new model
            dtype = torch.float16
            if self.force_bfloat16:
                dtype = torch.bfloat16
            elif self.force_float32 or self._device == "cpu":
                dtype = torch.float32
            elif self._device == "mps":
                dtype = torch.float32
            estimated_vram = self._estimate_model_vram(model_path, dtype)
            logger.info("Estimated VRAM for %s: %.1f GB", model_path.name,
                        estimated_vram / (1024**3))

            # Adaptive eviction: evict LRU models until the new model fits
            # or we hit the max_cached_models limit.
            while len(self._model_cache) >= self._max_cached_models:
                logger.info("Cache full (%d/%d), evicting LRU model",
                            len(self._model_cache), self._max_cached_models)
                self._evict_lru(1, move_to_cpu=True)

            # VRAM-based eviction: evict more if free VRAM is below threshold
            free_vram = self._get_free_vram_bytes()
            if free_vram is not None and self._vram_evict_threshold_gb > 0:
                threshold_bytes = int(self._vram_evict_threshold_gb * (1024**3))
                # Evict if even after freeing everything we wouldn't have
                # enough room for the new model + threshold margin.
                while free_vram is not None and (
                    free_vram < threshold_bytes
                    or (free_vram < estimated_vram + threshold_bytes
                        and len(self._model_cache) > 0)
                ) and len(self._model_cache) > 0:
                    logger.info("VRAM insufficient (free=%.1f GB, need=%.1f GB), "
                                "evicting LRU model",
                                free_vram / (1024**3),
                                (estimated_vram + threshold_bytes) / (1024**3))
                    evicted = self._evict_lru(1, move_to_cpu=True)
                    if evicted == 0:
                        break
                    free_vram = self._get_free_vram_bytes()

            # Load model in thread pool
            logger.info("Loading model in thread pool: %s", model_path)
            loop = asyncio.get_running_loop()
            pipeline, model_type, device_map_applied, is_single_file_sdxl = await loop.run_in_executor(
                None,
                self._load_model_blocking,
                model_path
            )

            # Estimate actual VRAM footprint after loading
            vram_footprint = self._estimate_vram_footprint(pipeline)

            # Create cached model entry
            cached = _CachedModel(
                pipeline=pipeline,
                model_path=model_path,
                model_type=model_type,
                device_map_active=device_map_applied,
                is_single_file_sdxl=is_single_file_sdxl,
                vram_footprint_bytes=vram_footprint,
            )
            self._model_cache[model_key] = cached

            # Set as active model
            self._set_active(model_key)

            # Apply optimizations (may modify _unet_compiled, _original_unet)
            self._apply_memory_optimizations()

            # Initialize CompelForSDXL for SDXL if needed (may set _compel)
            if self._current_model_type == "sdxl":
                try:
                    from compel import CompelForSDXL
                    self._compel = CompelForSDXL(self._pipeline, device=self._device)
                    logger.info("Initialized CompelForSDXL for long prompt support (device=%s)", self._device)
                except Exception as e:
                    logger.warning("Failed to initialize CompelForSDXL: %s", e)

            # Sync post-optimization state back to cache
            self._sync_active_to_cache(model_key)

            logger.info("Model loaded and cached: %s (cache_size=%d/%d)",
                        model_key, len(self._model_cache), self._max_cached_models)
    
    def _load_model_blocking(self, model_path: Path) -> Tuple[Any, str, bool, bool]:
        """
        Blocking model load operation - runs in thread pool.
        
        Args:
            model_path: Path to model file or directory
            
        Returns:
            Tuple of (pipeline, model_type, device_map_applied, is_single_file_sdxl)
        """
        logger.info("Loading model in thread pool: %s", model_path)

        # GGUF files are sd.cpp/Vulkan-only - cannot be loaded by PyTorch/diffusers
        if model_path.is_file() and model_path.suffix.lower() == ".gguf":
            raise RuntimeError(
                f"Model '{model_path.name}' is a GGUF file which requires the Vulkan (sd.cpp) backend. "
                f"The PyTorch backend cannot load GGUF models. "
                f"Either change 'backend' to 'vulkan' or 'auto' in config.yaml, "
                f"or use a .safetensors model instead."
            )

        # Import diffusers if needed
        _import_diffusers()
        
        # Detect pipeline class
        pipeline_class_name, model_type = _detect_pipeline_class(model_path)
    
        # Determine dtype based on device
        if self._device == "mps":
            # MPS requires float32 to avoid black images
            dtype = torch.float32
        elif self._device == "cuda":
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        # Force bfloat16 if configured (best for AMD Phoenix APU gfx1102)
        if self.force_bfloat16:
            dtype = torch.bfloat16
            logger.info("Using bfloat16 dtype for AMD GPU compatibility")
        # Force float32 if configured (required for some AMD GPUs like gfx1103)
        elif self.force_float32:
            dtype = torch.float32
            logger.info("Forcing float32 dtype as configured for AMD GPU compatibility")
        
        logger.info("Using dtype: %s for device: %s", dtype, self._device)
        
        # Load pipeline
        try:
            # Base load kwargs
            load_kwargs = {
                "torch_dtype": dtype,
                "safety_checker": None,
                "feature_extractor": None,
            }
            
            if model_path.is_file() and model_path.suffix in [".safetensors", ".ckpt"]:
                # Load from single file - device_map: sequential works here
                # Detect if this is an SDXL model (typically >6GB) vs SD 1.5 (~4GB)
                file_size_gb = model_path.stat().st_size / (1024**3)
                is_sdxl = file_size_gb > 5.5  # SDXL models are ~6.5-7GB, SD 1.5 is ~4GB
                
                if is_sdxl:
                    logger.info("Detected SDXL single-file model (%.1f GB)", file_size_gb)
                    from diffusers import StableDiffusionXLPipeline
                    pipeline_class = StableDiffusionXLPipeline
                    is_sdxl_result = True  # Enable extra memory optimizations
                    model_type = "sdxl"  # Override detection for single-file SDXL
                else:
                    logger.info("Detected SD 1.5 single-file model (%.1f GB)", file_size_gb)
                    from diffusers import StableDiffusionPipeline
                    pipeline_class = StableDiffusionPipeline
                    is_sdxl_result = False
                
                load_kwargs["use_safetensors"] = True
                if self.device_map:
                    load_kwargs["device_map"] = self.device_map
                    logger.info("Using device_map: %s (single-file mode)", self.device_map)
                pipeline = pipeline_class.from_single_file(
                    str(model_path),
                    **load_kwargs,
                )
                is_single_file_sdxl_result = is_sdxl_result
            else:
                # Load from directory
                is_single_file_sdxl_result = False  # Directory models are more memory efficient
                
                # Check if this is a Qwen model that needs a specific pipeline
                if model_type == "qwen":
                    # Qwen models use DiffusionPipeline.from_pretrained which will auto-resolve
                    # to QwenImageEditPlusPipeline based on model_index.json
                    logger.info("Loading Qwen img2img model with DiffusionPipeline (auto-resolve)")
                    from diffusers import DiffusionPipeline
                    if self.device_map and self.device_map in ('balanced', 'cuda'):
                        load_kwargs["device_map"] = self.device_map
                        logger.info("Using device_map: %s (Qwen model)", self.device_map)
                    elif self.device_map:
                        logger.warning(
                            "device_map '%s' not supported for Qwen models. "
                            "Valid options: 'balanced', 'cuda'. Loading without device_map.",
                            self.device_map
                        )
                    pipeline = DiffusionPipeline.from_pretrained(
                        str(model_path),
                        **load_kwargs,
                    )
                else:
                    # Load from directory using AutoPipeline for text2image models
                    # NOTE: device_map: sequential does NOT work with from_pretrained
                    # Only 'balanced' and 'cuda' are valid for from_pretrained
                    from diffusers import AutoPipelineForText2Image
                    if self.device_map and self.device_map in ('balanced', 'cuda'):
                        load_kwargs["device_map"] = self.device_map
                        logger.info("Using device_map: %s (directory mode)", self.device_map)
                    elif self.device_map:
                        logger.warning(
                            "device_map '%s' not supported for directory models. "
                            "Valid options: 'balanced', 'cuda'. Loading without device_map.",
                            self.device_map
                        )
                    pipeline = AutoPipelineForText2Image.from_pretrained(
                        str(model_path),
                        **load_kwargs,
                    )
            
            # Determine if we need to move to device
            # - If device_map was in load_kwargs, the model is already on the right device(s)
            # - If using CPU offload, the pipeline handles device placement
            # - Otherwise, explicitly move to the target device
            device_map_applied = "device_map" in load_kwargs
            if not device_map_applied and not self.enable_model_cpu_offload and not self.enable_sequential_cpu_offload:
                logger.info("Moving pipeline to device: %s", self._device)
                pipeline = pipeline.to(self._device)
            
            logger.info("Model file loaded successfully: %s (type=%s)", model_path.name, model_type)
            return pipeline, model_type, device_map_applied, is_single_file_sdxl_result
            
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            raise
    async def _unload_model_internal(self) -> None:
        """Unload all cached models without acquiring lock (internal use when lock already held)."""
        if len(self._model_cache) == 0 and len(self._cpu_cache) == 0:
            return

        logger.info("Unloading %d VRAM model(s), %d CPU model(s)",
                    len(self._model_cache), len(self._cpu_cache))

        # Clear all VRAM cached models
        for model_key, cached in self._model_cache.items():
            if cached.compel is not None:
                del cached.compel
                cached.compel = None
            del cached.pipeline

        self._model_cache.clear()

        # Clear CPU cache
        for model_key, cached in self._cpu_cache.items():
            if cached.compel is not None:
                del cached.compel
                cached.compel = None
            if cached.pipeline is not None:
                del cached.pipeline
        self._cpu_cache.clear()

        # Reset active model state
        self._pipeline = None
        self._current_model = None
        self._current_model_path = None
        self._current_model_type = None
        self._device_map_active = False
        self._is_single_file_sdxl = False
        self._compel = None
        self._unet_compiled = False
        self._original_unet = None

        # Clear GPU cache — run in thread to avoid blocking event loop
        if self._device == "cuda":
            def _unload_cleanup():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning("torch.cuda.empty_cache() failed during unload: %s", e)
            try:
                asyncio.create_task(asyncio.to_thread(_unload_cleanup))
            except RuntimeError:
                pass

        logger.info("All models unloaded")
    
    async def unload_model(self, model_path: Optional[Path] = None) -> None:
        """
        Unload model(s) from memory (public API, acquires lock).

        Args:
            model_path: Specific cached model to evict. If None, unloads ALL cached models.
        """
        async with self._model_lock:
            if model_path is None:
                await self._unload_model_internal()
                return

            # Per-model eviction: remove only the requested cached entry
            target_key = str(Path(model_path))
            evicted_from_vram = target_key in self._model_cache
            evicted_from_cpu = target_key in self._cpu_cache

            if evicted_from_vram:
                cached = self._model_cache.pop(target_key)
                if cached.compel is not None:
                    del cached.compel
                    cached.compel = None
                del cached.pipeline
            if evicted_from_cpu:
                cached = self._cpu_cache.pop(target_key)
                if cached.compel is not None:
                    del cached.compel
                    cached.compel = None
                if cached.pipeline is not None:
                    del cached.pipeline

            if not evicted_from_vram and not evicted_from_cpu:
                logger.debug("Model not in cache, nothing to evict: %s", target_key)
                return

            if self._device == "cuda":
                def _per_model_cleanup():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    try:
                        torch.cuda.empty_cache()
                    except Exception as e:
                        logger.warning("torch.cuda.empty_cache() failed: %s", e)
                try:
                    asyncio.create_task(asyncio.to_thread(_per_model_cleanup))
                except RuntimeError:
                    pass

            # If we evicted the active model, point active state at another cached model
            # or reset it to None if the cache is empty.
            if self._current_model == target_key:
                if len(self._model_cache) > 0:
                    self._set_active(next(reversed(self._model_cache)))
                else:
                    self._pipeline = None
                    self._current_model = None
                    self._current_model_path = None
                    self._current_model_type = None
                    self._device_map_active = False
                    self._is_single_file_sdxl = False
                    self._compel = None
                    self._unet_compiled = False
                    self._original_unet = None
    
    async def generate_image(
        self,
        model_path: Path,
        prompt: str,
        negative_prompt: Optional[str] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
        scheduler: Optional[str] = None,
        num_images: int = 1,
        lora_paths: Optional[List[Path]] = None,
        lora_scales: Optional[List[float]] = None,
        cancellation_token: Optional[CancellationToken] = None,
        input_images: Optional[List[Any]] = None,
        strength: Optional[float] = None,
    ) -> Tuple[List[Path], Dict[str, Any]]:
        """
        Generate image(s) from prompt, optionally with input images (img2img).
        
        Args:
            model_path: Path to model
            prompt: Image generation prompt
            negative_prompt: Negative prompt
            steps: Inference steps (uses default if None)
            guidance_scale: Guidance scale (uses default if None)
            width: Image width (uses default if None)
            height: Image height (uses default if None)
            seed: Random seed for reproducibility
            scheduler: Scheduler name (uses default if None)
            num_images: Number of images to generate
            lora_paths: List of paths to LoRA files to apply
            lora_scales: List of LoRA weights (0.0-1.0)
            cancellation_token: Optional token for cancellation support
            input_images: Optional list of PIL images for img2img
            strength: Denoising strength for img2img (0.0-1.0)
            
        Returns:
            Tuple of (list_of_image_paths, metadata_dict)
            
        Raises:
            CancellationError: If generation is cancelled via token
            num_images: Number of images to generate
            lora_paths: List of paths to LoRA files to apply
            lora_scales: List of LoRA weights (0.0-1.0)
            
        Returns:
            Tuple of (list_of_image_paths, metadata_dict)
        """
        import time
        start_time = time.time()
        
        # Track this request in the queue
        async with self._queue_lock:
            self._pending_requests += 1
            current_queue = self._pending_requests
        
        logger.debug("Request added to queue (queue_depth=%d)", current_queue)
        
        try:
            # Apply defaults
            steps = steps or self.default_steps
            guidance_scale = guidance_scale if guidance_scale is not None else self.default_guidance_scale
            width = width or self.default_width
            height = height or self.default_height
            scheduler = scheduler or self.default_scheduler
            
            # Validate and round dimensions to multiple of 8 (required for SD VAE)
            # The VAE downsampling requires dimensions divisible by 8
            original_width = width
            original_height = height
            width = _round_to_multiple(width, 8)
            height = _round_to_multiple(height, 8)
            
            if width != original_width or height != original_height:
                logger.info(
                    "Rounded dimensions from %dx%d to %dx%d (SD requires dimensions divisible by 8)",
                    original_width, original_height, width, height
                )
            
            # ACQUIRE SEMAPHORE FIRST - serializes all concurrent generation requests
            # This ensures only one model load + generation happens at a time
            async with self._generation_semaphore:
                logger.debug("Acquired generation semaphore, proceeding with generation")
                
                # Load model if needed (INSIDE semaphore to prevent multiple loads)
                await self.load_model(model_path)
                
                if self._pipeline is None:
                    raise RuntimeError("No model loaded")
                
                # Load LoRAs if specified
                lora_names = []
                if lora_paths and len(lora_paths) > 0:
                    # Set default scales if not provided
                    if lora_scales is None:
                        lora_scales = [1.0] * len(lora_paths)
                    elif len(lora_scales) < len(lora_paths):
                        # Pad with 1.0 for missing scales
                        lora_scales = list(lora_scales) + [1.0] * (len(lora_paths) - len(lora_scales))
                    
                    try:
                        for i, lora_path in enumerate(lora_paths):
                            lora_name = Path(lora_path).stem
                            lora_names.append(lora_name)
                            adapter_name = f"lora_{i}"
                            
                            logger.info("Loading LoRA: %s (scale=%.2f)", lora_name, lora_scales[i])
                            self._pipeline.load_lora_weights(
                                str(lora_path),
                                adapter_name=adapter_name
                            )
                        
                        # Set LoRA scales
                        if len(lora_paths) > 0:
                            adapter_names = [f"lora_{i}" for i in range(len(lora_paths))]
                            self._pipeline.set_adapters(adapter_names, adapter_weights=lora_scales[:len(lora_paths)])
                            logger.debug("Set LoRA adapters: %s with scales %s", adapter_names, lora_scales[:len(lora_paths)])
                            
                    except Exception as e:
                        logger.warning("Failed to load LoRAs: %s. Continuing without LoRAs.", e)
                        lora_names = []
                
                logger.info(
                    "Generating image: prompt='%s...', steps=%d, guidance=%.1f, size=%dx%d",
                    prompt[:50], steps, guidance_scale, width, height
                )
                
                # Set seed
                # When device_map is active, latents are generated on CPU
                generator_device = "cpu" if self._device_map_active else self._device
                generator = None
                actual_seed = seed
                if seed is not None:
                    generator = torch.Generator(device=generator_device).manual_seed(seed)
                else:
                    # Generate random seed for reproducibility tracking
                    actual_seed = torch.randint(0, 2**32, (1,)).item()
                    generator = torch.Generator(device=generator_device).manual_seed(actual_seed)
                
                # Generate in thread pool to avoid blocking
                def _generate():
                    # CRITICAL FIX: Create scheduler instance PER REQUEST to avoid race conditions
                    # When multiple concurrent requests share the same scheduler, they corrupt
                    # each other's state (step_index, sigmas, timesteps), causing index errors.
                    # Each request must have its own scheduler instance with independent state.
                    request_scheduler = None
                    original_scheduler = None
                    try:
                        request_scheduler = _get_scheduler(scheduler, self._pipeline.scheduler.config)
                        request_scheduler.set_timesteps(steps, device=self._device)
                        
                        # Diagnostic logging for Euler schedulers
                        if scheduler in ["euler_a", "euler", "euler_ancestral"]:
                            sigmas_len = len(request_scheduler.sigmas)
                            logger.debug(
                                "Created fresh scheduler for request: %s, steps=%d, sigmas_len=%d",
                                scheduler, steps, sigmas_len
                            )
                        
                        # Temporarily replace pipeline's scheduler with our request-specific instance
                        original_scheduler = self._pipeline.scheduler
                        self._pipeline.scheduler = request_scheduler
                        
                    except Exception as e:
                        logger.warning("Failed to create request scheduler %s: %s. Using shared scheduler (may have race conditions).", scheduler, e)
                    
                    try:
                        with torch.inference_mode():
                            # If VAE CPU decode is enabled, get latents instead of images
                            output_type = "latent" if self.vae_decode_cpu else "pil"
                            
                            # Determine generation mode: img2img vs txt2img
                            is_img2img = input_images is not None and len(input_images) > 0
                            
                            # Build pipeline kwargs based on generation mode
                            if is_img2img and self._current_model_type == "qwen":
                                # Qwen img2img: uses image parameter, no width/height
                                # Qwen models accept a list of images or a single image
                                logger.info("Using Qwen img2img mode with %d input image(s)", len(input_images))
                                pipeline_kwargs = {
                                    "prompt": prompt,
                                    "negative_prompt": negative_prompt if negative_prompt else " ",
                                    "image": input_images if len(input_images) > 1 else input_images[0],
                                    "num_inference_steps": steps,
                                    "num_images_per_prompt": num_images,
                                    "generator": generator,
                                    "output_type": output_type,
                                    "true_cfg_scale": guidance_scale,
                                    "guidance_scale": 1.0,  # Qwen uses true_cfg_scale instead
                                }
                            elif is_img2img:
                                # Generic img2img: SD/SDXL img2img pipeline
                                logger.info("Using generic img2img mode with %d input image(s)", len(input_images))
                                pipeline_kwargs = {
                                    "prompt": prompt,
                                    "negative_prompt": negative_prompt if negative_prompt else None,
                                    "image": input_images[0],  # SD img2img takes a single image
                                    "num_inference_steps": steps,
                                    "strength": strength if strength is not None else 0.75,
                                    "num_images_per_prompt": num_images,
                                    "generator": generator,
                                    "output_type": output_type,
                                    "guidance_scale": guidance_scale,
                                }
                            else:
                                # Standard text-to-image generation
                                pipeline_kwargs = {
                                    "prompt": prompt,
                                    "negative_prompt": negative_prompt if negative_prompt else None,
                                    "num_inference_steps": steps,
                                    "width": width,
                                    "height": height,
                                    "num_images_per_prompt": num_images,
                                    "generator": generator,
                                    "output_type": output_type,
                                }
                            
                                # Qwen text2img models use true_cfg_scale instead of guidance_scale
                                if self._current_model_type == "qwen":
                                    pipeline_kwargs["true_cfg_scale"] = guidance_scale
                                    logger.debug("Using true_cfg_scale=%s for Qwen model", guidance_scale)
                                else:
                                    pipeline_kwargs["guidance_scale"] = guidance_scale
                            
                            # Use CompelForSDXL for SDXL models to support long prompts
                            if self._compel is not None and not is_img2img:
                                logger.info("Using CompelForSDXL for long prompt encoding")
                                try:
                                    # Encode with CompelForSDXL (proper SDXL support, no length limits)
                                    # CompelForSDXL returns a LabelledConditioning object with:
                                    #   .embeds, .pooled_embeds, .negative_embeds, .negative_pooled_embeds
                                    logger.debug("Encoding prompts with CompelForSDXL")
                                    
                                    # Ensure text encoders are on the correct device before calling Compel.
                                    # With CPU offload or after model swaps, text encoders may be on CPU
                                    # while Compel sends token IDs to the target device.
                                    target_device = torch.device(self._device)
                                    if hasattr(self._pipeline, 'text_encoder') and self._pipeline.text_encoder is not None:
                                        te_device = next(self._pipeline.text_encoder.parameters()).device
                                        if te_device != target_device:
                                            logger.debug("Moving text_encoder from %s to %s", te_device, target_device)
                                            self._pipeline.text_encoder = self._pipeline.text_encoder.to(target_device)
                                    if hasattr(self._pipeline, 'text_encoder_2') and self._pipeline.text_encoder_2 is not None:
                                        te2_device = next(self._pipeline.text_encoder_2.parameters()).device
                                        if te2_device != target_device:
                                            logger.debug("Moving text_encoder_2 from %s to %s", te2_device, target_device)
                                            self._pipeline.text_encoder_2 = self._pipeline.text_encoder_2.to(target_device)
                                    
                                    # Use empty string for negative prompt if not provided.
                                    # CompelForSDXL returns None for negative_embeds when negative_prompt
                                    # is None, which causes AttributeError on .to(device). Using "" forces
                                    # proper negative embedding generation.
                                    compel_negative = negative_prompt if negative_prompt else ""
                                    
                                    result = self._compel(
                                        main_prompt=prompt,
                                        negative_prompt=compel_negative
                                    )
                                    
                                    logger.debug("Embedding shapes: embeds=%s, pooled=%s", 
                                                result.embeds.shape, result.pooled_embeds.shape)
                                    
                                    # Ensure all embeddings are on the correct device
                                    device = torch.device(self._device)
                                    
                                    # Replace text prompts with pre-computed embeddings
                                    pipeline_kwargs["prompt_embeds"] = result.embeds.to(device)
                                    pipeline_kwargs["negative_prompt_embeds"] = result.negative_embeds.to(device)
                                    pipeline_kwargs["pooled_prompt_embeds"] = result.pooled_embeds.to(device)
                                    pipeline_kwargs["negative_pooled_prompt_embeds"] = result.negative_pooled_embeds.to(device)
                                    
                                    # Remove text prompts (using embeddings instead)
                                    del pipeline_kwargs["prompt"]
                                    del pipeline_kwargs["negative_prompt"]
                                    
                                    logger.info("CompelForSDXL encoding complete - long prompts fully supported")
                                except Exception as e:
                                    logger.error("CompelForSDXL encoding failed: %s. Falling back to standard prompts.", e, exc_info=True)
                                    # Clean up any partially-set embeddings to prevent
                                    # "Cannot forward both prompt and prompt_embeds" errors
                                    for key in ["prompt_embeds", "negative_prompt_embeds",
                                                "pooled_prompt_embeds", "negative_pooled_prompt_embeds"]:
                                        pipeline_kwargs.pop(key, None)
                            
                            # Add cancellation callback if token provided
                            if cancellation_token is not None:
                                def cancellation_callback(pipe, step_index, timestep, callback_kwargs):
                                    """Check for cancellation between diffusion steps."""
                                    # Check if cancelled
                                    if cancellation_token.is_cancelled():
                                        logger.info("Cancellation detected at step %d/%d", step_index, steps)
                                        # Raise exception to abort generation
                                        raise CancellationError(f"Generation cancelled at step {step_index}")
                                    return callback_kwargs
                                
                                pipeline_kwargs["callback_on_step_end"] = cancellation_callback
                                logger.debug("Added cancellation callback to pipeline")
                            
                            try:
                                result = self._pipeline(**pipeline_kwargs)
                            except Exception as unet_error:
                                # Runtime Inductor fallback: if the compiled UNet fails during
                                # codegen (e.g. CantSplit on SDXL's 2560-dim bottleneck), swap
                                # back to the eager UNet and retry. Compile speedup is preserved
                                # for all subsequent generations on shapes Inductor handles.
                                error_msg = str(unet_error)
                                inductor_failure = (
                                    "CantSplit" in error_msg
                                    or "Inductor" in error_msg
                                    or "torch._inductor" in error_msg
                                )
                                if inductor_failure and self._original_unet is not None:
                                    logger.error(
                                        "torch.compile failed at runtime (%s). "
                                        "Falling back to eager UNet for this generation. "
                                        "Subsequent runs will retry compilation; this is a "
                                        "known Inductor limitation on SDXL bottleneck shapes.",
                                        error_msg.split("\n", 1)[0]
                                    )
                                    self._pipeline.unet = self._original_unet
                                    self._unet_compiled = False
                                    result = self._pipeline(**pipeline_kwargs)
                                else:
                                    raise
                        
                        # Decode VAE on CPU if enabled (fixes GPU hang on AMD gfx1103)
                        if self.vae_decode_cpu:
                            logger.info("Decoding VAE on CPU...")
                            latents = result.images
                            
                            # Get VAE's current dtype before moving
                            vae_dtype = next(self._pipeline.vae.parameters()).dtype
                            original_vae_device = next(self._pipeline.vae.parameters()).device
                            
                            # Move entire VAE to CPU and convert to float32 (CPU doesn't support bfloat16 well)
                            self._pipeline.vae.cpu().float()
                            latents_cpu = latents.cpu().float()
                            
                            with torch.no_grad():
                                images = self._pipeline.vae.decode(
                                    latents_cpu / self._pipeline.vae.config.scaling_factor,
                                    return_dict=False
                                )[0]
                                pil_images = self._pipeline.image_processor.postprocess(images, output_type="pil")
                            
                            # Move VAE back to original device and restore dtype
                            self._pipeline.vae.to(original_vae_device, dtype=vae_dtype)
                            
                            # Create a result-like object with PIL images
                            class VAEResult:
                                def __init__(self, images):
                                    self.images = images
                            result = VAEResult(pil_images)
                            logger.info("VAE decode on CPU completed")
                        
                        return result
                    finally:
                        # Restore original scheduler after generation completes
                        if original_scheduler is not None:
                            self._pipeline.scheduler = original_scheduler
                
                # Run generation
                loop = asyncio.get_running_loop()
                try:
                    result = await loop.run_in_executor(None, _generate)
                except CancellationError:
                    # Generation was cancelled - clean up and re-raise
                    logger.info("Generation cancelled, cleaning up...")
                    # Unload LoRAs if they were loaded
                    if lora_names:
                        try:
                            self._pipeline.unload_lora_weights()
                            logger.debug("Unloaded LoRA weights after cancellation")
                        except Exception as e:
                            logger.warning("Failed to unload LoRAs during cancellation cleanup: %s", e)
                    # Re-raise to propagate to endpoint
                    raise
                
                # Save image
                image_id = uuid.uuid4().hex[:12]
                image_path = self.images_dir / f"{image_id}.png"
                
                # Save first image (or all if num_images > 1)
                if num_images == 1:
                    result.images[0].save(image_path)
                    saved_paths = [image_path]
                else:
                    saved_paths = []
                    for i, img in enumerate(result.images):
                        path = self.images_dir / f"{image_id}_{i}.png"
                        img.save(path)
                        saved_paths.append(path)
                    image_path = saved_paths[0]  # Return first image path
                
                # Calculate statistics
                generation_time = time.time() - start_time
                self.total_generations += 1
                self.total_generation_time += generation_time
                
                logger.info(
                    "Image generated: %s (%.2fs)",
                    image_path.name, generation_time
                )
                
                # Determine generation mode
                is_img2img = input_images is not None and len(input_images) > 0
                
                # Build metadata
                metadata = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "steps": steps,
                    "guidance_scale": guidance_scale,
                    "width": width,
                    "height": height,
                    "seed": actual_seed,
                    "scheduler": scheduler,
                    "model": str(model_path.name),
                    "generation_time": generation_time,
                    "backend": self.get_backend_name(),
                    "num_images": num_images,
                    "loras": lora_names if lora_names else None,
                    "lora_scales": list(lora_scales[:len(lora_paths)]) if lora_paths and lora_scales else None,
                    "mode": "img2img" if is_img2img else "txt2img",
                    "input_image_count": len(input_images) if is_img2img else None,
                }
                
                # Unload LoRAs after generation to prevent memory buildup
                if lora_names:
                    try:
                        self._pipeline.unload_lora_weights()
                        logger.debug("Unloaded LoRA weights")
                    except Exception as e:
                        logger.warning("Failed to unload LoRAs: %s", e)
                
                logger.debug("Released generation semaphore")
                
                # Return all saved paths (for multi-image generation)
                return saved_paths, metadata
        
        finally:
            # Free GPU memory after each generation to prevent accumulation
            # that leads to crashes after ~20+ images or during model switching.
            # Runs in a thread pool to avoid blocking the event loop; on AMD
            # ROCm, gc.collect() and torch.cuda.empty_cache() can deadlock
            # when called synchronously.
            await self._cleanup_after_generation()
            
            # Always decrement queue counter, even on error
            async with self._queue_lock:
                self._pending_requests -= 1
                remaining_queue = self._pending_requests
            logger.debug("Request completed (remaining_queue_depth=%d)", remaining_queue)
    
    def get_average_generation_time(self) -> float:
        """Get average generation time in seconds."""
        if self.total_generations == 0:
            return 0.0
        return self.total_generation_time / self.total_generations
    
    async def shutdown(self) -> None:
        """Shutdown the generator service and cleanup resources."""
        logger.info("Shutting down generator service...")
        
        # Unload model
        await self.unload_model()
        
        logger.info("Generator service shutdown complete")
    
    @staticmethod
    def is_available() -> bool:
        """
        Check if PyTorch backend is available.
        
        Returns:
            True if PyTorch and GPU/CPU are available
        """
        try:
            import torch
            return True
        except ImportError:
            return False
    
    @staticmethod
    def get_backend_name() -> str:
        """
        Get human-readable backend name.
        
        Returns:
            Backend name with device info
        """
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                if "NVIDIA" in device_name or "Tesla" in device_name or "GeForce" in device_name:
                    return f"PyTorch (CUDA - {device_name})"
                else:
                    return f"PyTorch (ROCm - {device_name})"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "PyTorch (MPS - Apple Silicon)"
            else:
                return "PyTorch (CPU)"
        except:
            return "PyTorch"
