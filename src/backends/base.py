# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2026 Andrew Wyatt (Fewtarius)

"""
Backend Base Interface

Abstract base class defining the interface for image generation backends.
All backends (PyTorch, stable-diffusion.cpp, etc.) must implement this interface.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import logging

from PIL import Image

logger = logging.getLogger(__name__)


class BaseBackend(ABC):
    """
    Abstract base class for image generation backends.
    
    All backends must implement these methods to be compatible with ALICE.
    The GeneratorService uses this interface to remain backend-agnostic.
    """
    
    @abstractmethod
    def __init__(
        self,
        images_dir: Path,
        default_steps: int,
        default_guidance_scale: float,
        default_scheduler: str,
        default_width: int,
        default_height: int,
        **kwargs  # Backend-specific options
    ):
        """
        Initialize backend with configuration.
        
        Args:
            images_dir: Directory to save generated images
            default_steps: Default number of inference steps
            default_guidance_scale: Default guidance scale
            default_scheduler: Default scheduler/sampler name
            default_width: Default image width
            default_height: Default image height
            **kwargs: Backend-specific configuration options
        """
        pass
    
    @abstractmethod
    async def load_model(self, model_path: Path) -> None:
        """
        Load a model into memory.
        
        Args:
            model_path: Path to model file or directory
            
        Raises:
            FileNotFoundError: If model doesn't exist
            RuntimeError: If model loading fails
        """
        pass
    
    @abstractmethod
    async def unload_model(self) -> None:
        """
        Unload current model from memory.
        
        Called when switching models or shutting down.
        """
        pass
    
    @abstractmethod
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
        cancellation_token: Optional[Any] = None,
        input_images: Optional[List[Image.Image]] = None,
        strength: Optional[float] = None,
    ) -> Tuple[List[Path], Dict[str, Any]]:
        """
        Generate image(s) from prompt, optionally with input images (img2img).
        
        Args:
            model_path: Path to model
            prompt: Text prompt for image generation
            negative_prompt: Negative prompt (what to avoid)
            steps: Number of inference steps (uses default if None)
            guidance_scale: Guidance scale (uses default if None)
            width: Image width (uses default if None)
            height: Image height (uses default if None)
            seed: Random seed for reproducibility (random if None)
            scheduler: Scheduler/sampler name (uses default if None)
            num_images: Number of images to generate
            lora_paths: List of LoRA model paths to apply
            lora_scales: List of LoRA weights (0.0-1.0)
            cancellation_token: Optional cancellation token
            input_images: Optional list of PIL images for img2img generation
            strength: Denoising strength for img2img (0.0-1.0, default varies by model)
            
        Returns:
            Tuple of (list_of_image_paths, metadata_dict)
            
            image_paths: List of paths to saved PNG files
            metadata: Dict containing:
                - prompt: The prompt used
                - negative_prompt: The negative prompt used
                - steps: Actual steps used
                - guidance_scale: Actual guidance scale used
                - width: Image width
                - height: Image height
                - seed: Seed used
                - scheduler: Scheduler used
                - model: Model name/path
                - generation_time: Time taken in seconds
                - backend: Backend name (e.g., "pytorch", "sdcpp")
                - mode: "txt2img" or "img2img"
                - input_image_count: Number of input images (img2img only)
        
        Raises:
            RuntimeError: If generation fails
            CancellationError: If cancelled via token
        """
        pass
    
    @abstractmethod
    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Get GPU/device information.
        
        Returns:
            Dict containing:
                - device: Device name (e.g., "cuda", "vulkan", "cpu")
                - gpu_available: bool
                - gpu_name: GPU model name (if available)
                - memory_used: Memory used string (e.g., "4.2 GB")
                - memory_total: Total memory string (e.g., "8.0 GB")
                - utilization: GPU utilization 0.0-1.0 (if available)
                - stats_available: Whether detailed stats are available
        """
        pass
    
    @property
    @abstractmethod
    def current_model(self) -> Optional[str]:
        """
        Get currently loaded model path.
        
        Returns:
            Path string of loaded model, or None if no model loaded
        """
        pass
    
    @property
    @abstractmethod
    def is_model_loaded(self) -> bool:
        """
        Check if a model is currently loaded.
        
        Returns:
            True if model loaded, False otherwise
        """
        pass
    
    @staticmethod
    @abstractmethod
    def is_available() -> bool:
        """
        Check if this backend is available on the current system.
        
        Should check for required dependencies (PyTorch, Vulkan, etc.)
        without raising exceptions.
        
        Returns:
            True if backend can be used, False otherwise
        """
        pass
    
    @staticmethod
    @abstractmethod
    def get_backend_name() -> str:
        """
        Get human-readable backend name.
        
        Returns:
            Backend name (e.g., "PyTorch (ROCm)", "Vulkan (stable-diffusion.cpp)")
        """
        pass
