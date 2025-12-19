<!-- SPDX-License-Identifier: CC-BY-NC-4.0 -->
<!-- SPDX-FileCopyrightText: Copyright (c) 2025 Andrew Wyatt (Fewtarius) -->

# ALICE Environment Documentation

**CRITICAL: DO NOT MODIFY THIS ENVIRONMENT WITHOUT TESTING**

This document describes the TESTED, WORKING environment for ALICE on AMD Phoenix APUs (Ryzen 7 8840U, Ryzen 9 8945HS, etc).

---

## Working Configuration Summary

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.13 | System Python on SteamOS 3.6 |
| PyTorch | 2.6.0+rocm6.1 | **CRITICAL: Must be ROCm 6.1** |
| torchvision | 0.21.0+rocm6.1 | Must match PyTorch ROCm version |
| diffusers | 0.35.2 | Tested working |
| transformers | 4.57.3 | Required by diffusers |
| accelerate | 1.12.0 | Required by diffusers |
| safetensors | 0.7.0 | For model loading |
| Pillow | 12.0.0 | Image processing |
| compel | 2.3.1 | Prompt weighting |
| huggingface-hub | 0.36.0 | Model downloads |
| fastapi | 0.104.1 | Web framework |
| uvicorn | 0.24.0 | ASGI server |
| pydantic | 2.12.5 | Data validation |
| pyyaml | 6.0.3 | Config parsing |

---

## AMD Phoenix APU ROCm Configuration

**CRITICAL: Phoenix APUs (gfx1103) require HSA override to gfx1102**

### Required Environment Variables

```bash
PYTORCH_ROCM_ARCH=gfx1102
HSA_OVERRIDE_GFX_VERSION=11.0.2
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

### Why gfx1102 Override?

- Phoenix APUs are gfx1103 architecture
- PyTorch ROCm wheels do NOT include gfx1103 kernels
- gfx1102 kernels are compatible with gfx1103 via HSA override
- The 11.0.2 version maps to gfx1102 kernel dispatch

---

## Systemd Service File

**File:** `~/.config/systemd/user/ALICE.service`

```ini
[Unit]
Description=ALICE - Remote Stable Diffusion Service
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/deck/ALICE
Environment="SD_API_CONFIG=/home/deck/.config/ALICE/config.yaml"
Environment="PYTORCH_ROCM_ARCH=gfx1102"
Environment="HSA_OVERRIDE_GFX_VERSION=11.0.2"
Environment="TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1"
ExecStart=/home/deck/ALICE/venv/bin/python -m src.main
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```

---

## Generator SDPA Configuration

**CRITICAL: Flash attention and memory-efficient attention MUST be disabled for AMD GPUs**

In `src/generator.py`, at module level (before any function definitions):

```python
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
```

**Why?** Flash attention and memory-efficient attention use optimized kernels that are not compatible with gfx1102/1103 on ROCm. Only math-based SDPA works reliably.

---

## Config File

**File:** `~/.config/ALICE/config.yaml`

```yaml
server:
  host: 0.0.0.0
  port: 8090
  api_key: null

models:
  directory: /home/deck/.local/share/ALICE/models
  auto_unload_timeout: 300
  default_model: null

generation:
  default_steps: 20
  default_guidance_scale: 7.5
  default_scheduler: ddim
  max_concurrent: 1
  request_timeout: 600
  default_width: 512
  default_height: 512
  force_cpu: false        # Set to true to force CPU mode
  device_map: balanced
  force_float32: false

storage:
  images_directory: /home/deck/.local/share/ALICE/images
  max_storage_gb: 100
  retention_days: 7

logging:
  level: INFO
  file: /home/deck/.local/share/ALICE/logs/ALICE.log
```

---

## Installation Commands

### Create Virtual Environment

```bash
cd ~/ALICE
python -m venv venv
source venv/bin/activate
```

### Install PyTorch ROCm 6.1 (CRITICAL: Must be 6.1)

```bash
pip install torch==2.6.0+rocm6.1 torchvision==0.21.0+rocm6.1 \
    --index-url https://download.pytorch.org/whl/rocm6.1
```

### Install Diffusers and Dependencies

```bash
pip install diffusers==0.35.2 transformers accelerate safetensors pillow compel
pip install fastapi uvicorn pyyaml pydantic aiohttp
```

### Verify Installation

```bash
# Check PyTorch ROCm version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
# Should output: PyTorch: 2.6.0+rocm6.1

# Check CUDA/ROCm availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# Should output: CUDA available: True

# Check device name
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
# Should output: AMD Radeon 780M Graphics (or similar Phoenix GPU name)
```

---

## Tested Performance

| Image Size | Steps | Scheduler | Time | Mode |
|------------|-------|-----------|------|------|
| 512x512 | 10 | DDIM | ~59s | GPU |
| 512x512 | 20 | DDIM | ~105s | GPU |
| 512x512 | 10 | DDIM | ~50s | CPU |
| 768x768 | 25 | DPM++ | ~180s+ | GPU |

**Note:** Phoenix APU shares system RAM for VRAM. Performance varies based on available memory and system load.

---

## Troubleshooting

### "HIP error: invalid device function"

**Cause:** Wrong PyTorch ROCm version or missing HSA_OVERRIDE_GFX_VERSION

**Fix:**
1. Verify PyTorch is ROCm 6.1: `pip show torch | grep Version`
2. Verify environment variables in systemd service
3. Restart service: `systemctl --user restart ALICE`

### "GPU Hang" or segfault during generation

**Cause:** Flash attention or memory-efficient attention enabled

**Fix:**
1. Ensure SDPA backends are disabled in generator.py (see above)
2. Verify the code at module level runs before any torch operations

### "Using CPU (forced)" when GPU expected

**Cause:** `force_cpu: true` in config.yaml

**Fix:**
1. Edit `~/.config/ALICE/config.yaml`
2. Set `force_cpu: false`
3. Restart service

### Generation times are very slow (>100s for 10 steps)

**Cause:** Running in CPU mode despite GPU available

**Fix:**
1. Check logs: `journalctl --user -u ALICE | grep -E "CUDA|CPU"`
2. Should see "Using CUDA device: AMD Radeon 780M"
3. If see "CPU mode forced", check config.yaml

---

## DO NOT DO

**The following configurations have been TESTED AND FAILED:**

| Configuration | Result |
|--------------|--------|
| PyTorch ROCm 6.2.4 + gfx1103 | GPU Hang |
| PyTorch ROCm 6.2.4 + gfx1102 override | HSA_STATUS_ERROR_INVALID_ISA |
| PyTorch ROCm 6.3 | Same as 6.2.4 |
| PyTorch ROCm 6.4 | Same as 6.2.4 |
| Python 3.12 + ROCm 6.2+ | Various errors |
| Flash attention enabled | GPU Hang |
| Memory-efficient attention enabled | GPU Hang |
| Subprocess-based generation | Segfault |

**Stick with the documented configuration!**

---

## Version History

- **2025-12-02:** Documented working configuration after extensive testing
  - PyTorch 2.6.0+rocm6.1 with gfx1102 override WORKS
  - SDPA must be disabled for AMD GPUs
  - Inline diffusers (not subprocess) is the working approach

---

**Last Tested:** December 2, 2025
**Device:** AMD Ryzen 7 8840U (Phoenix APU) on SteamOS 3.6
**Generation Test:** 512x512, 10 steps, DDIM scheduler = 58.95 seconds
