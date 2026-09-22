# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2026 Andrew Wyatt (Fewtarius)

"""
ALICE Backend System

Provides pluggable backend architecture for image generation.
Supports multiple backends (PyTorch, Vulkan, etc.) with automatic detection.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from .base import BaseBackend

logger = logging.getLogger(__name__)


def detect_amd_gpu() -> Optional[Dict[str, Any]]:
    """
    Detect AMD GPU model and generation using lspci and sysfs.

    Returns a dict with:
      - name: human-readable GPU name (e.g. "AMD Radeon 780M")
      - generation: short code (e.g. "gfx1103", "gfx1151", "gfx1102")
      - is_apu: True if this is an integrated/aperture GPU (APU)
    Returns None if no AMD GPU is found.

    Uses lspci + sysfs so it works without importing torch.
    """
    # Try lspci first for a friendly name
    gpu_name = None
    try:
        import subprocess
        result = subprocess.run(["lspci"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                lower = line.lower()
                # AMD GPUs may show as "VGA compatible controller" or
                # "Display controller" depending on the PCIe device class.
                if ("amd" in lower or "ati" in lower or "radeon" in lower) and (
                    "vga" in lower or "display controller" in lower
                ):
                    # PCI lines look like: "67:00.0 Display controller: Advanced Micro..."
                    # or "01:00.0 VGA compatible controller: Advanced Micro..."
                    # Split on the device class prefix (after the bus address)
                    gpu_name = line
                    for prefix in ("Display controller: ", "VGA compatible controller: "):
                        idx = gpu_name.find(prefix)
                        if idx >= 0:
                            gpu_name = gpu_name[idx + len(prefix):].strip()
                            break
                    # Strip "Advanced Micro Devices, Inc. [AMD/ATI] " prefix
                    gpu_name = gpu_name.replace("Advanced Micro Devices, Inc. [AMD/ATI] ", "").strip()
                    break
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Use sysfs to find the device ID and map to generation
    drm_path = Path("/sys/class/drm")
    generation = None
    is_apu = False

    try:
        for card_dir in sorted(drm_path.glob("card*")):
            device_dir = card_dir / "device"
            if not device_dir.exists():
                continue
            vendor_file = device_dir / "vendor"
            if not vendor_file.exists():
                continue
            vendor = vendor_file.read_text().strip()
            if vendor != "0x1002":  # AMD vendor ID
                continue

            # Device ID tells us the generation
            dev_id_file = device_dir / "device"
            if dev_id_file.exists():
                dev_id = dev_id_file.read_text().strip()
                # Map device IDs to generations
                # gfx1103 = Phoenix APU (780M, 760M, 740M, 660M, 665M)
                # gfx1102 = Phoenix Point APU (8060S, etc.)
                # gfx1151 = Strix Halo (8050S, etc.)
                # gfx1100 = Navi 31, gfx1101 = Navi 32, etc.
                _AMD_DEVICE_MAP = {
                    "0x7480": "gfx900",   # Vega 10
                    "0x7481": "gfx900",
                    "0x74a1": "gfx906",   # Vega 20
                    "0x74a2": "gfx906",
                    "0x738c": "gfx908",   # MI100
                    "0x740c": "gfx90a",   # MI200
                    "0x7410": "gfx942",   # MI300
                    "0x7440": "gfx940",   # MI300X
                    "0x743f": "gfx942",
                    "0x7488": "gfx1100",  # Navi 31
                    "0x7489": "gfx1100",
                    "0x748a": "gfx1101",  # Navi 32
                    "0x748b": "gfx1102",  # Navi 33 (Phoenix Point)
                    "0x1586": "gfx1151",  # Strix Halo (8050S, 8060S)
                    "0x15b8": "gfx1151",  # Strix Halo variant
                    "0x15bc": "gfx1200",  # RDNA 4 (Navi 44)
                    "0x15bd": "gfx1201",  # RDNA 4 (Navi 48)
                }
                generation = _AMD_DEVICE_MAP.get(dev_id.lower(), "gfx_unknown")

                # Determine if APU (integrated graphics)
                # 0x1586 = Strix Halo (7840U, 8050S, 8060S) — it's an APU
                _APU_DEV_IDS = {"0x1586", "0x15b8"}
                is_apu = dev_id.lower() in _APU_DEV_IDS
                break
    except Exception:
        pass

    # Fall back to lspci-derived name if we have one but no generation
    if generation is None and gpu_name:
        name_lower = gpu_name.lower()
        if "780m" in name_lower or "phoenix" in name_lower or "760m" in name_lower or "740m" in name_lower:
            generation = "gfx1103"
            is_apu = True
        elif "8060" in name_lower or "phoenix point" in name_lower or "760m" in name_lower:
            generation = "gfx1102"
            is_apu = True
        elif "8050" in name_lower or "halo" in name_lower:
            generation = "gfx1151"
            is_apu = True
        elif "navi 31" in name_lower or "7900" in name_lower or "8070" in name_lower:
            generation = "gfx1100"
        elif "navi 32" in name_lower or "7800" in name_lower or "7700" in name_lower or "8060" in name_lower:
            generation = "gfx1101"
        elif "navi 33" in name_lower or "7600" in name_lower or "7500" in name_lower:
            generation = "gfx1102"

    if generation is None:
        return None

    # If we have a generation from sysfs but no friendly name from lspci,
    # set a readable name based on the generation code.
    if gpu_name is None:
        _GEN_NAMES = {
            "gfx1103": "AMD Radeon 780M (Phoenix APU)",
            "gfx1102": "AMD Radeon 8050S (Phoenix Point)",
            "gfx1151": "AMD Radeon 8060S (Strix Halo)",
            "gfx1100": "AMD Radeon RX 7900 (Navi 31)",
            "gfx1101": "AMD Radeon RX 7800 (Navi 32)",
            "gfx1200": "AMD Radeon RX 8800 (Navi 44)",
            "gfx1201": "AMD Radeon RX 8700 (Navi 48)",
        }
        gpu_name = _GEN_NAMES.get(generation, generation)

    return {
        "name": gpu_name,
        "generation": generation,
        "is_apu": is_apu,
    }


def log_gpu_recommendations() -> None:
    """
    Detect the GPU vendor and, if it's AMD, log configuration
    recommendations for optimal performance and stability.

    Called from the lifespan after config is loaded but before
    services are initialized, so users see guidance in their logs.
    """
    amd_info = detect_amd_gpu()
    if amd_info is None:
        logger.info("No AMD GPU detected; using default configuration.")
        return

    gen = amd_info.get("generation", "")
    name = amd_info.get("name", "")

    logger.info("AMD GPU detected: %s (%s)", name, gen)

    if gen == "gfx1103":
        # Phoenix APU — FP16 VAE decode hangs the GPU
        logger.info(
            "AMD Phoenix APU (%s) detected. Recommended config.yaml settings:\n"
            "  generation:\n"
            "    force_float32: true    # Required — FP16 causes GPU hangs\n"
            "    vae_decode_cpu: true   # Prevents VAE decode GPU hang\n"
            "    backend: vulkan        # Prefer Vulkan (sd.cpp) for stability",
            name,
        )
    elif gen in ("gfx1102", "gfx1151"):
        # Phoenix Point / Strix Halo — best performance at bfloat16
        logger.info(
            "AMD Phoenix Point / Strix Halo (%s) detected. Recommended config.yaml settings:\n"
            "  generation:\n"
            "    force_bfloat16: true  # Best APU performance at BF16\n"
            "    vae_decode_cpu: true   # Prevents VAE decode GPU hang\n"
            "    backend: auto         # PyTorch (ROCm) works well here\n"
            "  Environment:\n"
            "    PYTORCH_ROCM_ARCH=gfx1151  (or gfx1102 for Phoenix Point)\n"
            "    MIOPEN_DEBUG_FIND_ALL=0\n"
            "    PYTORCH_ALLOC_CONF=expandable_segments:True",
            name,
        )
    else:
        # Other AMD GPUs — Vulkan is generally safest
        logger.info(
            "AMD GPU (%s) detected. Consider using the Vulkan (sd.cpp) backend "
            "for the broadest stability. If using PyTorch (ROCm), set "
            "force_bfloat16: true or force_float32: true if you see GPU hangs.",
            name,
        )


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
