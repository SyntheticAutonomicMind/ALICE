<!-- SPDX-License-Identifier: CC-BY-NC-4.0 -->
<!-- SPDX-FileCopyrightText: Copyright (c) 2025 Andrew Wyatt (Fewtarius) -->

# ALICE AMD Deployment Guide

**Target Systems:** AMD Ryzen 7 7840U (Phoenix/gfx1103), Ryzen Z2 (Phoenix Point/gfx1102),
Strix Halo (gfx1151), and RDNA 2/3/4 discrete GPUs.

---

## System Requirements

- AMD GPU with ROCm support (gfx900–gfx1201 — see `docs/AMD-PHOENIX-ENVIRONMENT.md`)
- Linux with ROCm kernel drivers (`amdgpu` + `/dev/kfd`)
- Python 3.13+
- 16 GB+ system RAM (recommended)

---

## Phase 1: Environment Setup

### 1. Verify GPU and ROCm Access

```bash
# Check GPU
lspci | grep -i vga
# Should show: Advanced Micro Devices [AMD/ATI] Device <ID>

# Check ROCm device access
ls -la /dev/kfd
# Should show read/write for render/video groups

# Add user to required groups
sudo usermod -aG video,render $USER
```

### 2. Create Virtual Environment

```bash
cd ~/alice
python3 -m venv venv
source venv/bin/activate
```

### 3. Install PyTorch with ROCm

Install with the device-specific extra for your GPU:

```bash
# Phoenix APU (gfx1103)
pip install "torch[device-gfx1103]" --index-url https://stable.repo.amd.com/rocm/whl-next/

# Phoenix Point (gfx1102)
pip install "torch[device-gfx1102]" --index-url https://stable.repo.amd.com/rocm/whl-next/

# Strix Halo (gfx1151)
pip install "torch[device-gfx1151]" --index-url https://stable.repo.amd.com/rocm/whl-next/
```

Verify:
```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.is_available())"
```

### 4. Install ALICE Dependencies

```bash
pip install -r requirements.txt
```

---

## Phase 2: Model Setup

### Download a Model

```bash
# Create models directory
mkdir -p ~/.local/share/alice/models

# SD 1.5 (4.7 GB)
huggingface-cli download runwayml/stable-diffusion-v1-5 --local-dir ~/.local/share/alice/models/stable-diffusion-v1-5

# SDXL (6.5 GB)
huggingface-cli download stabilityai/sdxl --local-dir ~/.local/share/alice/models/sdxl
```

### Configure Model Directory

In `config.yaml`:
```yaml
models:
  directory: ~/.local/share/alice/models
```

---

## Phase 3: Configuration

ALICE auto-detects AMD GPUs at startup and logs recommendations. You can
also manually configure AMD-specific settings:

### Config File Format

```yaml
server:
  host: 0.0.0.0
  port: 8090
  # api_key: your-secret-key  # Uncomment to require auth

models:
  directory: ~/.local/share/alice/models
  default_model: sd/stable-diffusion-v1-5

generation:
  backend: auto          # Auto-selects Vulkan for AMD, PyTorch for NVIDIA
  default_steps: 20      # Lower steps = faster on AMD APUs
  default_guidance_scale: 7.5
  default_scheduler: dpm++_sde_karras
  default_width: 512
  default_height: 512

  # --- AMD-specific settings (uncomment for your GPU) ---

  # Phoenix APU (gfx1103):
  # force_float32: true    # Required — FP16 causes GPU hangs
  # vae_decode_cpu: true   # Prevents VAE decode GPU hang

  # Phoenix Point / Strix Halo (gfx1102, gfx1151):
  # force_bfloat16: true   # Best performance at BF16
  # vae_decode_cpu: true   # Prevents VAE decode GPU hang

storage:
  images_directory: ~/.local/share/alice/images
  max_storage_gb: 100
  retention_days: 7

logging:
  level: INFO
  file: ~/.local/share/alice/logs/alice.log

audio:
  enabled: true
  # request_timeout_seconds: 1800  # Default — allows full MiniMax-Music3 songs (up to 30 min)
```

### Backend Selection

| Backend | When to use | Notes |
|---|---|---|
| `auto` | Default | Selects Vulkan for AMD GPUs, PyTorch for NVIDIA |
| `vulkan` | All AMD GPUs | Uses stable-diffusion.cpp via Vulkan. Universal AMD support. |
| `pytorch` | ROCm-capable AMD + NVIDIA | Uses PyTorch/ROCm. Faster but may not work on all AMD GPUs. |

---

## Phase 4: Service Installation

### Systemd User Service

Create `~/.config/systemd/user/alice.service`:

```ini
[Unit]
Description=ALICE - Remote Stable Diffusion Service
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/deck/alice
ExecStart=/home/deck/alice/venv/bin/python -m src.main
Restart=on-failure
RestartSec=10

# AMD environment variables
Environment="MIOPEN_DEBUG_FIND_ALL=0"
Environment="PYTORCH_ROCM_ARCH=gfx1103"
Environment="PYTORCH_ALLOC_CONF=expandable_segments:True"
Environment="ALICE_CONFIG=/home/deck/.config/alice/config.yaml"
Environment="TMPDIR=/home/deck/tmp"

# Allow binding to port 8090
AmbientCapabilities=CAP_NET_BIND_SERVICE

[Install]
WantedBy=default.target
```

Enable and start:
```bash
systemctl --user daemon-reload
systemctl --user enable alice
systemctl --user start alice
```

### Docker Deployment

```bash
docker compose --profile rocm up -d
```

---

## Phase 5: Performance Tuning

### For AMD APUs (gfx1103 / gfx1102 / gfx1151)

```yaml
generation:
  # Use lower steps for faster generation on APUs
  default_steps: 20

  # Model caching — tune for limited VRAM
  max_cached_models: 1        # Only keep 1 model in VRAM
  vram_evict_threshold_gb: 2  # Evict when <2 GB free
  max_cpu_cached_models: 4    # Keep 4 models in CPU RAM for fast reload

  # Memory optimizations
  enable_vae_slicing: true    # Reduces VAE memory
  vae_decode_cpu: true        # Required for gfx1103 stability

  # torch.compile — optional, adds warmup time
  # enable_torch_compile: false  # SDXL + reduce-overhead can trigger Inductor CantSplit
```

### For AMD Discrete GPUs (RDNA 2/3/4)

```yaml
generation:
  max_concurrent: 1            # Increase for multi-GPU
  enable_torch_compile: true   # Safe on discrete GPUs with ample VRAM
  torch_compile_mode: default  # Avoid reduce-overhead on SDXL
```

### CPU Thread Tuning

For the Vulkan (sd.cpp) backend, `sdcpp_threads` controls CPU thread usage.
The default (8) is conservative. Set to your full thread count:

```yaml
generation:
  sdcpp_threads: 16
```

---

## Memory Optimization for Low-VRAM Systems

### Model Memory Requirements

| Model | Disk | BF16 VRAM | FP32 VRAM | With Offload |
|-------|------|-----------|-----------|--------------|
| SD 1.5 | 4.7 GB | ~5 GB | ~10 GB | ~6 GB peak |
| SDXL | 6.5 GB | ~7 GB | ~14 GB | ~8 GB peak |
| FLUX-schnell | 5.5 GB | ~6 GB | ~12 GB | ~8 GB peak |

### Recommended Settings for 8 GB VRAM

```yaml
generation:
  force_bfloat16: true         # Halves memory vs FP32
  enable_sequential_cpu_offload: true  # Slowest but minimum VRAM
  enable_vae_slicing: true     # Reduces VAE decode memory
  vae_decode_cpu: true         # VAE decode on CPU (frees GPU memory)
```

---

## Testing

### Start Service
```bash
systemctl --user start alice
journalctl --user -u alice -f    # View logs
```

### Test Generation
```bash
curl -X POST http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sd/stable-diffusion-v1-5",
    "messages": [{"role": "user", "content": "a red apple on a white table"}]
  }'
```

### Verify GPU Detection
```bash
# Check logs for AMD detection
journalctl --user -u alice -n 20
# Should see: "AMD GPU detected: <name> (<gen>)"
# Should see: "Recommended config.yaml settings: ..."
```

### Check Health
```bash
curl http://localhost:8090/health
```

---

## Troubleshooting

### ROCm Not Detected
```bash
# Check driver
lsmod | grep amdgpu

# Check device
ls -la /dev/kfd /dev/dri/renderD128

# Check groups
id | grep -E "video|render"

# Try environment variable
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

### Out of Memory
```bash
# Check memory usage
free -h

# Monitor during generation
rocm-smi
```

### Service Won't Start
```bash
# Check logs
journalctl --user -u alice -n 50

# Test manually
~/alice/venv/bin/python -m src.main
```

### GPU Hang During VAE Decode
- Ensure `vae_decode_cpu: true` is set in config.yaml for gfx1103
- This is the #1 cause of hangs on Phoenix APUs

### Slow Generation
- Check that `MIOPEN_DEBUG_FIND_ALL=0` is set (prevents MIOpen solver search hangs)
- Use Vulkan backend for more stable AMD performance
- Reduce `default_steps` (20 is good for APUs)

---

## Quick Reference

### Start Service
```bash
systemctl --user start alice
```

### Stop Service
```bash
systemctl --user stop alice
```

### View Logs
```bash
journalctl --user -u alice -f
```

### Check GPU Memory
```bash
rocm-smi --showuse --showmeminfo vram --json
```
