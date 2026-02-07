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

from .backends import get_backend, BaseBackend
from .cancellation import CancellationToken

logger = logging.getLogger(__name__)


class GeneratorService:
    """
    Image generation service (backward-compatible wrapper).
    
    This class maintains the same public API as the original GeneratorService
    but delegates all work to a pluggable backend (PyTorch, Vulkan, etc.).
    
    The backend is selected at initialization time based on config or auto-detection.
    """
    
    def __init__(
        self,
        images_dir: str | Path = "./images",
        backend_name: str = "auto",
        sdcpp_binary: Optional[Path] = None,
        sdcpp_threads: int = 8,
        default_steps: int = 20,
        default_guidance_scale: float = 7.5,
        default_scheduler: str = "dpm++_sde_karras",
        default_width: int = 512,
        default_height: int = 512,
        force_cpu: bool = False,
        device_map: Optional[str] = None,
        force_float32: bool = False,
        force_bfloat16: bool = False,
        enable_vae_slicing: bool = False,
        enable_vae_tiling: bool = False,
        enable_model_cpu_offload: bool = False,
        enable_sequential_cpu_offload: bool = False,
        enable_mmap: bool = False,
        keep_clip_on_cpu: bool = False,
        attention_slice_size: Optional[str] = None,
        vae_decode_cpu: bool = False,
        enable_torch_compile: bool = False,
        torch_compile_mode: str = "default",
        diffusion_conv_direct: bool = False,
        vae_conv_direct: bool = False,
        circular: bool = False,
        enable_flash_attention: bool = True,  # Default True - faster AND lower memory with COOPMAT1
        max_concurrent_generations: int = 1,
    ):
        """
        Initialize generator service with backend.
        
        Args:
            images_dir: Directory to save generated images
            backend_name: Backend to use ("auto", "pytorch", "vulkan")
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
        
        # Track statistics
        self.total_generations = 0
        
        logger.info("Initializing GeneratorService with backend: %s", backend_name)
        
        # Create backend using factory
        self._backend: BaseBackend = get_backend(
            backend_name=backend_name,
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
        )
        
        logger.info("Generator initialized with backend: %s", self._backend.get_backend_name())
    
    async def load_model(self, model_path: Path) -> None:
        """Load a model into memory."""
        await self._backend.load_model(model_path)
    
    async def unload_model(self) -> None:
        """Unload current model from memory."""
        await self._backend.unload_model()
    
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
    ) -> Tuple[List[Path], Dict[str, Any]]:
        """
        Generate image(s) from prompt.
        
        Returns:
            Tuple of (list_of_image_paths, metadata_dict)
        """
        result = await self._backend.generate_image(
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
        return self._backend.get_gpu_info()
    
    @property
    def current_model(self) -> Optional[str]:
        """Get currently loaded model path."""
        return self._backend.current_model
    
    @property
    def is_model_loaded(self) -> bool:
        """Check if a model is loaded."""
        return self._backend.is_model_loaded
    
    def get_queue_depth(self) -> int:
        """
        Get current queue depth.
        
        Note: Not all backends support queue tracking.
        Returns 0 if backend doesn't implement this.
        """
        if hasattr(self._backend, 'get_queue_depth'):
            return self._backend.get_queue_depth()
        return 0
    
    def get_active_generations(self) -> int:
        """
        Get number of currently executing generations.
        
        Note: Not all backends support this metric.
        Returns 0 if backend doesn't implement this.
        """
        if hasattr(self._backend, 'get_active_generations'):
            return self._backend.get_active_generations()
        return 0
    
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
        if hasattr(self._backend, 'shutdown'):
            await self._backend.shutdown()
        else:
            await self.unload_model()
