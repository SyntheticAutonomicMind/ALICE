# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2026 Andrew Wyatt (Fewtarius)

"""
ALICE Backend System

Provides pluggable backend architecture for image generation.
Supports multiple backends (PyTorch, Vulkan, etc.) with automatic detection.
"""

import logging
from pathlib import Path
from typing import Optional

from .base import BaseBackend

logger = logging.getLogger(__name__)


def detect_backend() -> str:
    """
    Auto-detect best backend for current system.
    
    Detection strategy:
    1. Check for NVIDIA GPU -> PyTorch (CUDA is well-supported)
    2. Check for AMD GPU:
       - If sdcpp available -> Vulkan (safer, universal support)
       - Otherwise -> PyTorch (may not work on all AMD GPUs)
    3. Fallback -> sdcpp if available, else PyTorch
    
    Returns:
        Backend name: "pytorch" or "sdcpp"
    """
    logger.debug("Auto-detecting best backend...")
    
    # First check if sdcpp is available (won't trigger PyTorch import)
    sdcpp_available = False
    try:
        from .sdcpp_backend import SDCppBackend
        sdcpp_available = SDCppBackend.is_available()
    except (ImportError, Exception) as e:
        logger.debug("SDCpp backend check failed: %s", e)
    
    # Try to detect GPU type
    try:
        import subprocess
        
        # Check for NVIDIA GPU
        result = subprocess.run(
            ["lspci"], 
            capture_output=True, 
            text=True, 
            timeout=2
        )
        
        if result.returncode == 0:
            output_lower = result.stdout.lower()
            
            # NVIDIA - use PyTorch (CUDA well-supported)
            if "nvidia" in output_lower and "vga" in output_lower:
                logger.info("Detected NVIDIA GPU - selecting PyTorch backend")
                return "pytorch"
            
            # AMD - prefer Vulkan for stability
            if ("amd" in output_lower or "ati" in output_lower) and "vga" in output_lower:
                if sdcpp_available:
                    logger.info("Detected AMD GPU - selecting Vulkan (sdcpp) backend for stability")
                    return "sdcpp"
                else:
                    logger.warning(
                        "Detected AMD GPU but Vulkan backend not available. "
                        "Using PyTorch (may not work on all AMD GPUs). "
                        "Build stable-diffusion.cpp for better compatibility."
                    )
                    return "pytorch"
    
    except Exception as e:
        logger.warning("GPU detection failed: %s. Using default backend.", e)
    
    # Fallback: Prefer sdcpp if available (works everywhere), else PyTorch
    if sdcpp_available:
        logger.info("Using Vulkan (sdcpp) backend (default)")
        return "sdcpp"
    else:
        logger.info("Using PyTorch backend (default)")
        return "pytorch"


def get_backend(
    backend_name: str,
    images_dir: Path,
    **kwargs
) -> BaseBackend:
    """
    Factory function to create backend instance.
    
    Args:
        backend_name: "pytorch", "sdcpp", "vulkan", or "auto" (auto-detect)
        images_dir: Directory to save generated images
        **kwargs: Backend-specific configuration options
        
    Returns:
        BaseBackend instance
        
    Raises:
        ValueError: If backend_name is unknown
        RuntimeError: If backend is not available on this system
        
    Example:
        >>> backend = get_backend("auto", Path("./images"), default_steps=20)
        >>> await backend.generate_image(...)
    """
    # Normalize backend names
    backend_name = backend_name.lower()
    if backend_name == "vulkan":
        backend_name = "sdcpp"  # Vulkan is implemented via sdcpp
    
    # Auto-detect if requested
    if backend_name == "auto":
        backend_name = detect_backend()
        logger.info("Auto-detected backend: %s", backend_name)
    
    # Import ONLY the backend we need (avoid loading all backends)
    backend_class = None
    
    if backend_name == "pytorch":
        try:
            from .pytorch_backend import PyTorchBackend
            backend_class = PyTorchBackend
            # Filter kwargs - remove sdcpp/vulkan-specific params
            vulkan_only_params = {
                'sdcpp_binary', 'sdcpp_threads',
                'enable_mmap', 'keep_clip_on_cpu',
                'diffusion_conv_direct', 'vae_conv_direct',
                'circular', 'enable_flash_attention',
                'max_concurrent_generations'  # Generator-level, not backend
            }
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in vulkan_only_params}
        except ImportError as e:
            raise RuntimeError(f"PyTorch backend not available: {e}")
    
    elif backend_name == "sdcpp":
        try:
            from .sdcpp_backend import SDCppBackend
            backend_class = SDCppBackend
            # Filter kwargs - remove pytorch-only params (keep shared params like enable_vae_tiling)
            pytorch_only_params = {
                'force_cpu', 'device_map', 'force_float32', 'force_bfloat16',
                'enable_vae_slicing',  # PyTorch-specific
                'enable_sequential_cpu_offload',  # PyTorch-specific
                'attention_slice_size',  # PyTorch-specific
                'enable_torch_compile', 'torch_compile_mode',  # PyTorch-specific
                'max_concurrent_generations'  # Generator-level, not backend
            }
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in pytorch_only_params}
        except ImportError as e:
            raise RuntimeError(f"SDCpp backend not available: {e}")
    
    else:
        raise ValueError(
            f"Unknown backend: '{backend_name}'. "
            f"Valid backends: 'pytorch', 'sdcpp', 'vulkan' (alias for sdcpp), 'auto'"
        )
    
    # Check if backend is available on this system
    if not backend_class.is_available():
        raise RuntimeError(
            f"Backend '{backend_name}' ({backend_class.get_backend_name()}) "
            f"is not available on this system. Install required dependencies."
        )
    
    # Create and return backend instance
    logger.info("Creating backend: %s", backend_class.get_backend_name())
    return backend_class(images_dir=images_dir, **filtered_kwargs)



__all__ = [
    "BaseBackend",
    "get_backend",
    "detect_backend",
]
