<!-- SPDX-License-Identifier: CC-BY-NC-4.0 -->
<!-- SPDX-FileCopyrightText: Copyright (c) 2025 Andrew Wyatt (Fewtarius) -->

# TheRock ROCm for gfx1103 (Phoenix APU) Installation Guide

This guide documents how to install TheRock's nightly ROCm and PyTorch builds with proper gfx1103 support for AMD Phoenix APUs (like the Radeon 780M in Steam Deck OLED and Lenovo Legion Go S).

## Background

The official PyTorch ROCm packages (from pytorch.org) do NOT include kernels compiled for gfx1103. This causes segfaults during GPU inference. TheRock project provides nightly builds that include gfx1103 support.

## Quick Install (Recommended)

### Prerequisites

1. Python 3.11, 3.12, or 3.13
2. Virtual environment
3. GPU access configured (user in `video` and `render` groups)

### Install ROCm + PyTorch with gfx1103 Support

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with gfx1103 ROCm support (includes rocm dependencies)
pip install \
  --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ \
  --pre torch torchaudio torchvision

# Install diffusers and other dependencies
pip install diffusers transformers accelerate safetensors pillow
```

### Verify Installation

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Test simple tensor operation on GPU
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.matmul(x, x)
    print(f"GPU tensor test passed: {y.shape}")
```

## Supported GPUs in gfx110X-all Family

| Device | GPU Target |
|--------|------------|
| AMD RX 7900 XTX | gfx1100 |
| AMD RX 7800 XT | gfx1101 |
| AMD RX 7700S / Framework Laptop 16 | gfx1102 |
| AMD Radeon 780M Laptop iGPU (Phoenix) | gfx1103 |

## Environment Variables

The following environment variables are recommended for gfx1103:

```bash
# Set correct architecture
export PYTORCH_ROCM_ARCH=gfx1103

# Version override for compatibility
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Enable experimental features
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

For systemd services, add these to the service file:

```ini
[Service]
Environment="PYTORCH_ROCM_ARCH=gfx1103"
Environment="HSA_OVERRIDE_GFX_VERSION=11.0.0"
Environment="TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1"
```

## Alternative: Install ROCm Packages Only

If you need just the ROCm SDK without PyTorch:

```bash
pip install \
  --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ \
  "rocm[libraries,devel]"
```

## Package Versions

TheRock releases follow a versioning pattern like `6.5.0rc20250610`. To check compatible package versions:

| torch | torchaudio | torchvision |
|-------|------------|-------------|
| 2.10  | 2.10       | 0.25        |
| 2.9   | 2.9        | 0.24        |
| 2.8   | 2.8        | 0.23        |

To install specific versions:

```bash
pip install \
  --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ \
  --pre torch==2.10 torchaudio==2.10 torchvision==0.25
```

## Troubleshooting

### SEGFAULT during inference

If you still get segfaults after installing TheRock packages:

1. **Check GPU is detected:**
   ```bash
   rocminfo | grep gfx
   ```
   Should show `gfx1103`

2. **Verify TheRock PyTorch is installed:**
   ```bash
   pip show torch | grep Version
   ```
   Should show a version like `2.10.0.dev...+rocm6.5`

3. **Check environment variables:**
   ```bash
   echo $PYTORCH_ROCM_ARCH
   echo $HSA_OVERRIDE_GFX_VERSION
   ```

4. **Test GPU directly:**
   ```python
   import torch
   x = torch.tensor([1.0, 2.0, 3.0]).cuda()
   print(x + x)  # Should print tensor([2., 4., 6.], device='cuda:0')
   ```

### Memory Issues

Phoenix APU shares system memory. For 8GB systems:

- Enable VAE slicing and tiling
- Use SD1.5 models instead of SDXL
- Close other applications during generation

### Older Python Version

TheRock supports Python 3.11, 3.12, 3.13. For older Python versions, you may need to build from source.

## Building from Source (Advanced)

If you need to build TheRock from source (e.g., for custom kernels or debugging):

```bash
# Install dependencies (Ubuntu 24.04)
sudo apt update
sudo apt install gfortran git ninja-build cmake g++ pkg-config \
  xxd patchelf automake libtool python3-venv python3-dev \
  libegl1-mesa-dev texinfo bison flex

# Clone TheRock
git clone https://github.com/ROCm/TheRock.git
cd TheRock

# Setup Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Fetch sources
python3 ./build_tools/fetch_sources.py

# Configure for gfx1103
cmake -B build -GNinja . -DTHEROCK_AMDGPU_TARGETS=gfx1103

# Build (this takes a LONG time - 4-8+ hours)
cmake --build build
```

## References

- [TheRock GitHub Repository](https://github.com/ROCm/TheRock)
- [TheRock Releases](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)
- [Supported GPUs](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md)
- [ROCm Documentation](https://rocm.docs.amd.com/)
