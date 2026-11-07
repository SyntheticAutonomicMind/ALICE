# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2026 Andrew Wyatt (Fewtarius)

"""
ALICE Image Generator

Backward-compatible wrapper around the pluggable backend system.
Uses backend factory to create PyTorch, Vulkan, or other backends.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

from .backends import get_backend, detect_backend, BaseBackend
from .cancellation import CancellationToken

logger = logging.getLogger(__name__)

# VRAM must be at least this far below the configured evict threshold
# before a backend switch forces LRU eviction of the other backend's cache.
_BYTES_PER_GB = 1024 * 1024 * 1024


class GeneratorService:
    """Media generation service (backward-compatible wrapper).
    
    This class maintains the same public API as the original GeneratorService
    but delegates all work to a pluggable backend (PyTorch, Vulkan, etc.).
    
    The backend is selected at initialization time based on config or auto-detection.
    """
    
    def __init__(
        self,
        images_dir: str | Path = "./images",
        backend_name: str = "auto",
        gguf_backend: str = "auto",
        sdcpp_binary: Optional[Path] = None,
        sdcpp_threads: int = 8,
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
        enable_mmap: bool = False,
        keep_clip_on_cpu: bool = False,
        attention_slice_size: Optional[str] = None,
        vae_decode_cpu: bool = False,
        enable_torch_compile: bool = False,
        torch_compile_mode: str = "reduce-overhead",
        diffusion_conv_direct: bool = False,
        vae_conv_direct: bool = False,
        circular: bool = False,
        enable_flash_attention: bool = True,  # Default True - faster AND lower memory with COOPMAT1
        max_concurrent_generations: int = 1,
        max_cached_models: int = 2,
        vram_evict_threshold_gb: float = 2.0,
        max_cpu_cached_models: int = 8,
    ):
        """
        Initialize generator service with backend.
        
        Args:
            images_dir: Directory to save generated images
            backend_name: Backend to use ("auto", "pytorch", "vulkan")
            gguf_backend: Backend for GGUF models ("auto", "pytorch", "sdcpp")
            sdcpp_binary: Path to sd-cli binary (for Vulkan backend)
            sdcpp_threads: CPU threads for Vulkan backend
            default_steps: Default inference steps
            default_guidance_scale: Default guidance scale
            default_scheduler: Default scheduler name
            default_width: Default image width
            default_height: Default image height
            
            All other arguments are backend-specific and passed through.
        """
        self.images_dir = Path(images_dir)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._gguf_backend = (gguf_backend or "auto").lower()
        
        # Track statistics
        self.total_generations = 0
        
        logger.info("Initializing GeneratorService with backend: %s", backend_name)
        
        # Resolve the primary backend key ("auto" and "vulkan" are aliases).
        primary_key = (backend_name or "auto").lower()
        if primary_key == "vulkan":
            primary_key = "sdcpp"
        elif primary_key == "auto":
            primary_key = detect_backend()

        # Create backend using factory
        self._backend: BaseBackend = get_backend(
            backend_name=primary_key,
            images_dir=self.images_dir,
            sdcpp_binary=sdcpp_binary,
            sdcpp_threads=sdcpp_threads,
            default_steps=default_steps,
            default_guidance_scale=default_guidance_scale,
            default_scheduler=default_scheduler,
            default_width=default_width,
            default_height=default_height,
            force_cpu=force_cpu,
            device_map=device_map,
            force_float32=force_float32,
            force_bfloat16=force_bfloat16,
            enable_vae_slicing=enable_vae_slicing,
            enable_vae_tiling=enable_vae_tiling,
            enable_model_cpu_offload=enable_model_cpu_offload,
            enable_sequential_cpu_offload=enable_sequential_cpu_offload,
            enable_mmap=enable_mmap,
            keep_clip_on_cpu=keep_clip_on_cpu,
            attention_slice_size=attention_slice_size,
            vae_decode_cpu=vae_decode_cpu,
            enable_torch_compile=enable_torch_compile,
            torch_compile_mode=torch_compile_mode,
            diffusion_conv_direct=diffusion_conv_direct,
            vae_conv_direct=vae_conv_direct,
            circular=circular,
            enable_flash_attention=enable_flash_attention,
            max_concurrent_generations=max_concurrent_generations,
            max_cached_models=max_cached_models,
            vram_evict_threshold_gb=vram_evict_threshold_gb,
            max_cpu_cached_models=max_cpu_cached_models,
        )
        
        # Remember constructor kwargs so lazily-created routing backends
        # are configured identically to the primary backend.
        self._backend_kwargs = dict(
            sdcpp_binary=sdcpp_binary,
            sdcpp_threads=sdcpp_threads,
            default_steps=default_steps,
            default_guidance_scale=default_guidance_scale,
            default_scheduler=default_scheduler,
            default_width=default_width,
            default_height=default_height,
            force_cpu=force_cpu,
            device_map=device_map,
            force_float32=force_float32,
            force_bfloat16=force_bfloat16,
            enable_vae_slicing=enable_vae_slicing,
            enable_vae_tiling=enable_vae_tiling,
            enable_model_cpu_offload=enable_model_cpu_offload,
            enable_sequential_cpu_offload=enable_sequential_cpu_offload,
            enable_mmap=enable_mmap,
            keep_clip_on_cpu=keep_clip_on_cpu,
            attention_slice_size=attention_slice_size,
            vae_decode_cpu=vae_decode_cpu,
            enable_torch_compile=enable_torch_compile,
            torch_compile_mode=torch_compile_mode,
            diffusion_conv_direct=diffusion_conv_direct,
            vae_conv_direct=vae_conv_direct,
            circular=circular,
            enable_flash_attention=enable_flash_attention,
            max_concurrent_generations=max_concurrent_generations,
            max_cached_models=max_cached_models,
            vram_evict_threshold_gb=vram_evict_threshold_gb,
            max_cpu_cached_models=max_cpu_cached_models,
        )
        self._backend_key = primary_key
        self._routing_backends: Dict[str, BaseBackend] = {}
        self._model_backend: Dict[str, str] = {}
        self._last_backend_key: Optional[str] = None
        
        logger.info("Generator initialized with backend: %s", self._backend.get_backend_name())
    
    def _get_routing_backend(self, key: str) -> BaseBackend:
        """Get or lazily create the secondary backend for model-based routing.
        
        Args:
            key: Backend key ("pytorch" or "sdcpp")
        
        Returns:
            Backend instance (cached across calls)
        
        Raises:
            RuntimeError: If the backend is not available on this system
        """
        if key == self._backend_key:
            return self._backend
        if key not in self._routing_backends:
            self._routing_backends[key] = get_backend(
                backend_name=key,
                images_dir=self.images_dir,
                **self._backend_kwargs,
            )
        return self._routing_backends[key]
    
    def _select_backend_for_model(self, model_path: Path) -> BaseBackend:
        """Select the backend that can load the given model.

        GGUF models route to diffusers (PyTorch) when diffusers supports the
        model family - generally much faster than sd.cpp on ROCm/CUDA - and
        fall back to sd.cpp otherwise. The generation.gguf_backend config
        can force a specific backend ('auto', 'pytorch', 'sdcpp').

        Args:
            model_path: Path to the model file or directory

        Returns:
            Backend instance for this model

        Raises:
            RuntimeError: If a GGUF model is requested but sd.cpp is unavailable
        """
        if model_path.suffix.lower() == ".gguf":
            gguf_pref = (self._gguf_backend or "auto").lower()

            if gguf_pref in ("pytorch", "diffusers", "auto"):
                try:
                    from .backends.pytorch_backend import pytorch_backend_can_load_gguf
                    if pytorch_backend_can_load_gguf(model_path):
                        return self._get_routing_backend("pytorch") if self._backend_key != "pytorch" else self._backend
                except Exception as e:
                    logger.debug("GGUF diffusers routing check failed: %s", e)
                if gguf_pref in ("pytorch", "diffusers"):
                    raise RuntimeError(
                        f"Model '{model_path.name}' is a GGUF file but diffusers cannot load "
                        f"its model family. Install the latest diffusers or set "
                        f"generation.gguf_backend to 'sdcpp'."
                    )

            if self._backend_key == "sdcpp":
                return self._backend

            try:
                from .backends.sdcpp_backend import SDCppBackend
                available = SDCppBackend.is_available()
            except Exception:
                available = False
            configured_binary = self._backend_kwargs.get("sdcpp_binary")
            if configured_binary and not Path(configured_binary).is_file():
                available = False

            if not available:
                raise RuntimeError(
                    f"Model '{model_path.name}' is a GGUF file which requires the "
                    f"sd.cpp (Vulkan) backend, but the sd-cli binary was not found. "
                    f"Build stable-diffusion.cpp or use a .safetensors model instead."
                )
            return self._get_routing_backend("sdcpp")
        return self._backend
    
    def _last_backend_key_used(self):
        """Return the backend instance last used, if any."""
        if self._last_backend_key is None:
            return None
        if self._last_backend_key == self._backend_key:
            return self._backend
        return self._routing_backends.get(self._last_backend_key)
    
    def _evict_for_switch(self, source: BaseBackend) -> None:
        """Evict LRU models from the source backend until VRAM is available.
        
        Args:
            source: Backend whose VRAM cache should be trimmed
        """
        get_free = getattr(source, "_get_free_vram_bytes", None)
        evict = getattr(source, "_evict_lru", None)
        if get_free is None or evict is None:
            return
        threshold_gb = getattr(source, "_vram_evict_threshold_gb", 0.0)
        if threshold_gb <= 0:
            return
        while source.loaded_models:
            free = get_free()
            if free is None or free >= threshold_gb * _BYTES_PER_GB:
                break
            evicted = evict(count=1, move_to_cpu=True)
            if not evicted:
                break
            logger.info(
                "Backend switch: evicted LRU model from %s to free VRAM",
                source.get_backend_name(),
            )
    
    def _route(self, model_path: Path) -> BaseBackend:
        """Resolve the backend for a model and handle memory on switch.
        
        Args:
            model_path: Path to the model file or directory
        
        Returns:
            Backend instance to use for this model
        """
        backend = self._select_backend_for_model(model_path)
        self._model_backend[str(model_path.resolve())] = backend.get_backend_name()
        all_backends = [self._backend, *self._routing_backends.values()]
        for other in all_backends:
            if other is not backend and other.is_model_loaded:
                self._evict_for_switch(other)
        if backend is self._backend:
            self._last_backend_key = self._backend_key
        else:
            for key, routing_backend in self._routing_backends.items():
                if routing_backend is backend:
                    self._last_backend_key = key
                    break
        return backend
    
    async def load_model(self, model_path: Path) -> None:
        """Load a model into memory."""
        backend = self._route(model_path)
        await backend.load_model(model_path)
    
    async def unload_model(self, model_path: Optional[Path] = None) -> None:
        """Unload model(s) from memory. None unloads all cached models."""
        if model_path is not None:
            backend = self._select_backend_for_model(model_path)
            await backend.unload_model(model_path)
            self._model_backend.pop(str(model_path.resolve()), None)
        else:
            await self._backend.unload_model(None)
            for backend in self._routing_backends.values():
                await backend.unload_model(None)
            self._model_backend.clear()
    
    @property
    def loaded_models(self) -> List[str]:
        """Get list of all cached model paths (most recently used first)."""
        models: List[str] = []
        for backend in [self._backend, *self._routing_backends.values()]:
            for model in backend.loaded_models:
                if model not in models:
                    models.append(model)
        return models
    
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
        input_images: Optional[list] = None,
        strength: Optional[float] = None,
    ) -> Tuple[List[Path], Dict[str, Any]]:
        """
        Generate image(s) from prompt, optionally with input images (img2img).
        
        Returns:
            Tuple of (list_of_image_paths, metadata_dict)
        """
        backend = self._route(model_path)
        result = await backend.generate_image(
            model_path=model_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            seed=seed,
            scheduler=scheduler,
            num_images=num_images,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            cancellation_token=cancellation_token,
            input_images=input_images,
            strength=strength,
        )
        
        # Update statistics
        self.total_generations += 1
        
        return result
    
    # Alias for backward compatibility with main.py
    async def generate(self, *args, **kwargs) -> Tuple[List[Path], Dict[str, Any]]:
        """Alias for generate_image (backward compatibility)."""
        return await self.generate_image(*args, **kwargs)
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU/device information."""
        # Prefer the PyTorch backend for detailed stats when present.
        for backend in [self._backend, *self._routing_backends.values()]:
            if "pytorch" in backend.get_backend_name().lower():
                return backend.get_gpu_info()
        return self._backend.get_gpu_info()
    
    @property
    def current_model(self) -> Optional[str]:
        """Get currently loaded model path."""
        for backend in [self._backend, *self._routing_backends.values()]:
            if backend.current_model is not None:
                return backend.current_model
        return None
    
    @property
    def is_model_loaded(self) -> bool:
        """Check if a model is loaded."""
        return any(
            backend.is_model_loaded
            for backend in [self._backend, *self._routing_backends.values()]
        )
    
    def get_queue_depth(self) -> int:
        """
        Get current queue depth.
        
        Note: Not all backends support queue tracking.
        Returns 0 if backend doesn't implement this.
        """
        depth = 0
        for backend in [self._backend, *self._routing_backends.values()]:
            if hasattr(backend, 'get_queue_depth'):
                depth += backend.get_queue_depth()
        return depth
    
    def get_active_generations(self) -> int:
        """
        Get number of currently executing generations.
        
        Note: Not all backends support this metric.
        Returns 0 if backend doesn't implement this.
        """
        active = 0
        for backend in [self._backend, *self._routing_backends.values()]:
            if hasattr(backend, 'get_active_generations'):
                active += backend.get_active_generations()
        return active
    
    def get_average_generation_time(self) -> float:
        """
        Get average generation time.
        
        Note: Not all backends support this metric.
        Returns 0.0 if backend doesn't implement this.
        """
        if hasattr(self._backend, 'get_average_generation_time'):
            return self._backend.get_average_generation_time()
        return 0.0
    
    async def shutdown(self) -> None:
        """Shutdown the generator service."""
        for backend in [self._backend, *self._routing_backends.values()]:
            if hasattr(backend, 'shutdown'):
                await backend.shutdown()
        await self.unload_model()
