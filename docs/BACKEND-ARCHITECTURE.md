# ALICE Backend Architecture Design

**Date:** 2026-02-05  
**Purpose:** Add Vulkan-based stable-diffusion.cpp backend for AMD APU compatibility

## Problem Statement

AMD Cezanne/Renoir APUs (gfx90c) cause kernel panics with ROCm-based PyTorch. Vulkan provides universal AMD GPU support without ROCm dependencies.

## Solution: Pluggable Backend Architecture

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Endpoints                        │
│              (OpenAI-compatible API)                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 GeneratorService                            │
│              (Orchestrator/Factory)                         │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
┌──────────────────┐           ┌──────────────────────┐
│ PyTorchBackend   │           │ SDCppBackend         │
│ (ROCm/CUDA/CPU)  │           │ (Vulkan universal)   │
└──────────────────┘           └──────────────────────┘
```

### Backend Selection Logic

1. **Auto-detect on startup:**
   - Detect GPU vendor/model
   - Check for ROCm support (gfx architecture)
   - Default to Vulkan for unsupported chips (gfx90c, old GPUs)
   - Default to PyTorch for supported chips (gfx1103, gfx1100, CUDA)

2. **Admin override:**
   - Config file: `generation.backend: "pytorch" | "sdcpp" | "auto"`
   - Web UI: Settings page toggle (requires service restart)
   - Runtime switching NOT supported (models loaded at startup)

3. **No automatic fallback:**
   - Kernel panic = system reboot (can't catch/fallback)
   - User must manually switch backend in config

## File Structure

```
src/
├── backends/
│   ├── __init__.py                # Export BaseBackend, get_backend()
│   ├── base.py                    # Abstract base class
│   ├── pytorch_backend.py         # Current PyTorch implementation
│   └── sdcpp_backend.py           # New stable-diffusion.cpp wrapper
├── generator.py                   # Refactored to use backends
└── main.py                        # No changes (uses GeneratorService)
```

## Backend Interface (base.py)

```python
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

class BaseBackend(ABC):
    """Abstract base class for image generation backends."""
    
    @abstractmethod
    def __init__(
        self,
        images_dir: Path,
        default_steps: int,
        default_guidance_scale: float,
        default_scheduler: str,
        default_width: int,
        default_height: int,
        **kwargs
    ):
        """Initialize backend with config."""
        pass
    
    @abstractmethod
    async def load_model(self, model_path: Path) -> None:
        """Load a model into memory."""
        pass
    
    @abstractmethod
    async def unload_model(self) -> None:
        """Unload current model from memory."""
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
    ) -> Tuple[List[Path], Dict[str, Any]]:
        """
        Generate image(s).
        
        Returns:
            Tuple of (list_of_image_paths, metadata_dict)
        """
        pass
    
    @abstractmethod
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information."""
        pass
    
    @property
    @abstractmethod
    def current_model(self) -> Optional[str]:
        """Get currently loaded model path."""
        pass
    
    @property
    @abstractmethod
    def is_model_loaded(self) -> bool:
        """Check if a model is loaded."""
        pass
    
    @staticmethod
    @abstractmethod
    def is_available() -> bool:
        """Check if this backend is available on this system."""
        pass
    
    @staticmethod
    @abstractmethod
    def get_backend_name() -> str:
        """Get human-readable backend name."""
        pass
```

## Backend Factory (backends/__init__.py)

```python
from pathlib import Path
from typing import Optional
from .base import BaseBackend
from .pytorch_backend import PyTorchBackend
from .sdcpp_backend import SDCppBackend

def detect_backend() -> str:
    """
    Auto-detect best backend for current system.
    
    Returns:
        "pytorch" or "sdcpp"
    """
    # DEFAULT TO VULKAN for AMD systems
    # Vulkan is more stable and has universal AMD support
    # PyTorch/ROCm only for explicitly configured systems
    
    # Check if CUDA available (NVIDIA - use PyTorch)
    try:
        import torch
        if torch.cuda.is_available() and "NVIDIA" in torch.cuda.get_device_name(0):
            return "pytorch"
    except:
        pass
    
    # Default to Vulkan for all AMD systems (APUs and GPUs)
    # More stable, no kernel panic risk
    return "sdcpp"

def get_backend(
    backend_name: str,
    images_dir: Path,
    **kwargs
) -> BaseBackend:
    """
    Factory function to create backend instance.
    
    Args:
        backend_name: "pytorch", "sdcpp", or "auto"
        images_dir: Directory to save images
        **kwargs: Backend-specific config
        
    Returns:
        BaseBackend instance
    """
    if backend_name == "auto":
        backend_name = detect_backend()
    
    backends = {
        "pytorch": PyTorchBackend,
        "sdcpp": SDCppBackend,
    }
    
    if backend_name not in backends:
        raise ValueError(f"Unknown backend: {backend_name}. Available: {list(backends.keys())}")
    
    backend_class = backends[backend_name]
    
    if not backend_class.is_available():
        raise RuntimeError(
            f"Backend '{backend_name}' is not available on this system. "
            f"Install required dependencies or choose a different backend."
        )
    
    return backend_class(images_dir=images_dir, **kwargs)
```

## Config Changes (config.yaml)

```yaml
generation:
  # Backend selection
  backend: "auto"  # "pytorch", "sdcpp", or "auto" (detect best)
  
  # PyTorch-specific settings (ignored if backend=sdcpp)
  device_map: null
  force_float32: false
  enable_torch_compile: false
  
  # SDCpp-specific settings (ignored if backend=pytorch)
  sdcpp_binary: null  # Path to sd binary (auto-detect if null)
  sdcpp_threads: 8    # CPU threads for Vulkan
```

## SDCppBackend Implementation Strategy

### Installation

```bash
# Build stable-diffusion.cpp with Vulkan
git clone --recursive https://github.com/leejet/stable-diffusion.cpp
cd stable-diffusion.cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSD_VULKAN=ON
cmake --build . --config Release
```

### Model Compatibility

stable-diffusion.cpp uses GGUF format, NOT safetensors. We need conversion:

1. **Option A: Convert at load time**
   - Convert safetensors → GGUF on first load
   - Cache GGUF files in `models/.gguf/`
   - Use `scripts/convert.py` from sd.cpp repo

2. **Option B: Require pre-converted models**
   - User must convert manually
   - Faster startup, no conversion overhead

**Recommendation:** Option A (auto-convert) for seamless UX

### Subprocess Communication

```python
import subprocess
import json

async def generate_image(self, ...):
    # Build command
    cmd = [
        str(self.sdcpp_binary),
        "-m", str(model_path),
        "-p", prompt,
        "--steps", str(steps),
        "--cfg-scale", str(guidance_scale),
        "-W", str(width),
        "-H", str(height),
        "--seed", str(seed),
        "-o", str(output_path),
        "--verbose"
    ]
    
    # Run in thread pool (blocking I/O)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        subprocess.run,
        cmd,
        subprocess.PIPE,
        subprocess.PIPE,
        str(self.models_dir),
    )
    
    # Parse output, return paths
    return [output_path], metadata
```

## Migration Path

### Phase 1: Architecture (This PR)
- [ ] Create `src/backends/` directory
- [ ] Implement `BaseBackend` abstract class
- [ ] Refactor current code into `PyTorchBackend`
- [ ] Update `GeneratorService` to use factory pattern
- [ ] Test PyTorch backend still works

### Phase 2: Vulkan Support (Next PR)
- [ ] Build stable-diffusion.cpp on target system
- [ ] Implement `SDCppBackend` class
- [ ] Add model conversion logic (safetensors → GGUF)
- [ ] Add backend detection logic
- [ ] Update config schema
- [ ] Add admin UI toggle

### Phase 3: Testing & Docs
- [ ] Test on gfx90c APU (Cezanne)
- [ ] Test on gfx1103 APU (Phoenix) - both backends
- [ ] Test on NVIDIA GPU - PyTorch only
- [ ] Document backend selection in README
- [ ] Update deployment guides

## Testing Strategy

```python
# Test backend interface compliance
def test_backend_interface():
    """Ensure all backends implement BaseBackend."""
    for backend_cls in [PyTorchBackend, SDCppBackend]:
        assert issubclass(backend_cls, BaseBackend)
        
        # Check all abstract methods implemented
        abstract_methods = [
            "load_model", "unload_model", "generate_image",
            "get_gpu_info", "is_available", "get_backend_name"
        ]
        for method in abstract_methods:
            assert hasattr(backend_cls, method)

# Integration test
async def test_backend_generation():
    """Test actual image generation with both backends."""
    for backend_name in ["pytorch", "sdcpp"]:
        backend = get_backend(backend_name, images_dir=Path("./test_images"))
        
        paths, metadata = await backend.generate_image(
            model_path=Path("models/sd-v1-5.safetensors"),
            prompt="a red apple",
            steps=20
        )
        
        assert len(paths) == 1
        assert paths[0].exists()
        assert metadata["prompt"] == "a red apple"
```

## Performance Expectations

| Backend | GPU | Speed (512x512, 20 steps) | VRAM |
|---------|-----|---------------------------|------|
| PyTorch | gfx1103 (Phoenix) | ~8-12s | 4-6GB |
| PyTorch | CUDA (RTX 3060) | ~3-5s | 3-4GB |
| SDCpp Vulkan | gfx90c (Cezanne) | ~15-25s | 2-3GB |
| SDCpp Vulkan | gfx1103 (Phoenix) | ~10-15s | 2-3GB |

Vulkan is slightly slower but **stable** on unsupported APUs.

## Open Questions

1. **LoRA support in sd.cpp?**
   - Investigate if sd.cpp supports LoRA adapters
   - If not, document limitation

2. **Scheduler mapping?**
   - Map ALICE scheduler names to sd.cpp equivalents
   - `dpm++_sde_karras` → `euler_a` or `dpmpp_2m`?

3. **SDXL support?**
   - sd.cpp supports SDXL
   - Test memory requirements on 3GB APU

## Security Considerations

- **Subprocess injection:** Sanitize all inputs to sd.cpp binary
- **Path traversal:** Validate model paths before passing to binary
- **Resource limits:** Set timeout on sd.cpp subprocess (kill if hung)

## Success Criteria

- ✅ PyTorch backend still works on gfx1103/CUDA
- ✅ Vulkan backend works on gfx90c without kernel panic
- ✅ Config switch between backends works
- ✅ Auto-detection selects correct backend
- ✅ API compatibility unchanged (OpenAI format)
- ✅ Image quality comparable between backends
