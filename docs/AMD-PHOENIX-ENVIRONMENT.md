<!-- SPDX-License-Identifier: CC-BY-NC-4.0 -->
<!-- SPDX-FileCopyrightText: Copyright (c) 2025 Andrew Wyatt (Fewtarius) -->

# ALICE Environment Documentation - AMD Phoenix APU

**Target Device:** AMD Ryzen 7 7840U with Radeon 780M (gfx1103)  
**OS:** SteamOS 3.6.x

## CRITICAL: GPU Support Status

**⚠️ AMD gfx1103 (Phoenix APU) is NOT supported by PyTorch ROCm wheels.**

### Supported GPU Architectures (PyTorch ROCm 6.4)
- gfx900 (Vega 10)
- gfx906 (Vega 20)
- gfx908 (MI100)
- gfx90a (MI200)
- gfx942 (MI300)
- gfx1030 (RDNA 2)
- gfx1100 (RDNA 3 - Navi 31)
- gfx1101 (RDNA 3 - Navi 32)
- gfx1102 (RDNA 3 - Navi 33)
- gfx1200 (RDNA 4 - Navi 44)
- gfx1201 (RDNA 4 - Navi 48)

**MISSING: gfx1103 (Phoenix/Hawk Point APUs)**

### Current Workaround: CPU Mode

Until PyTorch adds gfx1103 support, ALICE runs in CPU mode on Phoenix APUs.

**Performance:**
- ~6 seconds per inference step
- 10 steps ≈ 60 seconds
- 20 steps ≈ 120 seconds

## Working Environment

### Python Version
```
Python 3.12.12 (via pyenv)
Location: /home/deck/.pyenv/versions/3.12.12/bin/python
```

### Virtual Environment
```
Location: /home/deck/alice/venv312
Python: 3.12.12
```

### Installed Packages (Core)
| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.9.1+rocm6.4 | PyTorch with ROCm (CPU fallback) |
| torchvision | 0.24.1+rocm6.4 | |
| diffusers | 0.35.2 | Stable Diffusion pipeline |
| transformers | 4.57.3 | Text encoding |
| accelerate | 1.12.0 | Device management |
| safetensors | 0.7.0 | Model loading |
| compel | 2.3.1 | Prompt weighting |
| pillow | 11.3.0 | Image processing |
| pydantic | 2.12.5 | Data validation |
| fastapi | 0.123.4 | Web API |
| uvicorn | 0.38.0 | ASGI server |
| pyyaml | 6.0.3 | Configuration |

### Configuration File
**Location:** `/home/deck/.config/alice/config.yaml`

```yaml
server:
  host: 0.0.0.0
  port: 8090
  api_key: null

models:
  directory: /home/deck/.local/share/alice/models
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
  force_cpu: true          # REQUIRED for Phoenix APU
  device_map: balanced     # For future GPU support
  force_float32: true      # Stability

storage:
  images_directory: /home/deck/.local/share/alice/images
  max_storage_gb: 50
  retention_days: 7

logging:
  level: INFO
  file: /home/deck/.local/share/alice/logs/alice.log
```

### Systemd Service
**Location:** `/home/deck/.config/systemd/user/alice.service`

```ini
[Unit]
Description=ALICE - Remote Stable Diffusion Service
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/deck/alice
Environment="SD_API_CONFIG=/home/deck/.config/alice/config.yaml"
Environment="TMPDIR=/home/deck/tmp"
ExecStart=/home/deck/alice/venv312/bin/python -m src.main
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```

## Installation Steps

### 1. Install pyenv
```bash
curl https://pyenv.run | bash
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

### 2. Install Python 3.12
```bash
export TMPDIR=~/tmp
mkdir -p ~/tmp
pyenv install 3.12.12
```

### 3. Create Virtual Environment
```bash
~/.pyenv/versions/3.12.12/bin/python -m venv ~/alice/venv312
```

### 4. Install PyTorch
```bash
export TMPDIR=~/tmp
~/alice/venv312/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
```

### 5. Install Dependencies
```bash
~/alice/venv312/bin/pip install diffusers transformers accelerate safetensors compel psutil aiohttp fastapi uvicorn pyyaml
```

### 6. Configure Service
```bash
mkdir -p ~/.config/systemd/user
cp /path/to/alice.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable alice
systemctl --user start alice
```

## Future GPU Support

When PyTorch adds gfx1103 support:

1. Update config.yaml:
   ```yaml
   generation:
     force_cpu: false
   ```

2. Update service file to add environment variables:
   ```ini
   Environment="PYTORCH_ROCM_ARCH=gfx1103"
   Environment="HSA_OVERRIDE_GFX_VERSION=11.0.3"
   Environment="TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1"
   ```

3. Add SDPA fix in generator.py (already present):
   ```python
   if torch.cuda.is_available():
       torch.backends.cuda.enable_flash_sdp(False)
       torch.backends.cuda.enable_mem_efficient_sdp(False)
       torch.backends.cuda.enable_math_sdp(True)
   ```

4. Test:
   ```bash
   ~/alice/venv312/bin/python -c "import torch; x = torch.randn(2,2,device='cuda'); print(x)"
   ```

## Troubleshooting

### Check Service Status
```bash
systemctl --user status alice
journalctl --user -u alice -f
```

### Test Generation
```bash
curl -X POST http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "sd/MODEL_NAME", "messages": [{"role": "user", "content": "a cat"}]}'
```

### Check PyTorch GPU Support
```bash
~/alice/venv312/bin/python -c "import torch; print(torch.cuda.get_arch_list())"
```

### Common Errors

**"HIP error: invalid device function"**
- Cause: GPU architecture not supported
- Solution: Use `force_cpu: true` in config

**"HSA_STATUS_ERROR_INVALID_ISA"**
- Cause: gfx1103 not in PyTorch build
- Solution: Use CPU mode until PyTorch support added

**"No module named torch"**
- Cause: Wrong Python environment
- Solution: Ensure service uses venv312
