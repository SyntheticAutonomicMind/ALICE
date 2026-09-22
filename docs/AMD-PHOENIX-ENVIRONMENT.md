<!-- SPDX-License-Identifier: CC-BY-NC-4.0 -->
<!-- SPDX-FileCopyrightText: Copyright (c) 2025 Andrew Wyatt (Fewtarius) -->

# ALICE Environment Documentation - AMD Phoenix APU

**Target Devices:** AMD Ryzen 7 7840U (gfx1103 / Radeon 780M), AMD Ryzen Z2/Phoenix Point (gfx1102 / Radeon 8050S), Strix Halo (gfx1151 / Radeon 8060S)

---

## CRITICAL: GPU Architecture Support

### ROCm PyTorch Support (as of ROCm 10.0.0)

The ROCm stable index at `https://stable.repo.amd.com/rocm/whl-next/` supports device-specific
extras. Install PyTorch with the appropriate device extra for your GPU:

```bash
# Phoenix APU (gfx1103) — 7840U, 7640U, 7440U, 760M, 780M
pip install torch[device-gfx1103] --index-url https://stable.repo.amd.com/rocm/whl-next/

# Phoenix Point / Strix Halo (gfx1102) — 8050S, 8060S, 8035U
pip install torch[device-gfx1102] --index-url https://stable.repo.amd.com/rocm/whl-next/

# Strix Halo (gfx1151) — 8050S, 8060S (multi-GPU APU)
pip install torch[device-gfx1151] --index-url https://stable.repo.amd.com/rocm/whl-next/
```

This replaces the legacy TheRock nightly indices and all workarounds
(`--pre`, `--no-deps`, manual rocm-sdk-libraries, torchvision nms patch).

**Verified working on real gfx1151 hardware (Strix Halo / Radeon 8060S).**

### Fallback: CPU Mode

If ROCm is unavailable or unsupported, ALICE runs in CPU mode. Set
`force_cpu: true` in `config.yaml` under `generation:`.

**Performance (CPU mode, 512x512):**
- ~6 seconds per inference step
- 20 steps ≈ 120 seconds

---

## AMD-Specific Configuration

ALICE ships with automatic GPU detection (`backends/__init__.py:detect_amd_gpu()`).
At startup, it logs the detected AMD GPU generation and recommended settings.

### gfx1103 (Phoenix APU — 7840U)

```yaml
generation:
  force_float32: true      # Required — FP16/BF16 causes GPU hangs
  vae_decode_cpu: true     # Prevents VAE decode GPU hang
  backend: vulkan          # Prefer Vulkan (sd.cpp) for stability
```

Environment variables (set in systemd service or shell):
```bash
export MIOPEN_DEBUG_FIND_ALL=0
export PYTORCH_ROCM_ARCH=gfx1103
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

### gfx1102 / gfx1151 (Phoenix Point / Strix Halo)

```yaml
generation:
  force_bfloat16: true    # Best APU performance at BF16
  vae_decode_cpu: true    # Prevents VAE decode GPU hang
  backend: auto           # PyTorch (ROCm) works well here
```

Environment variables:
```bash
export MIOPEN_DEBUG_FIND_ALL=0
export PYTORCH_ROCM_ARCH=gfx1102   # or gfx1151
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

### Other AMD GPUs (RDNA 2/3/4 discrete)

```yaml
generation:
  backend: auto           # Auto-selects Vulkan for AMD by default
```

If using PyTorch (ROCm), add:
```yaml
generation:
  force_bfloat16: true    # Or force_float32: true if you see hangs
```

---

## Key Workarounds (Built-in)

These are applied automatically in `src/backends/pytorch_backend.py`:

| Workaround | Trigger | Purpose |
|---|---|---|
| `MIOPEN_DEBUG_FIND_ALL=0` | Module load (AMD only) | Prevents MIOpen solver search GPU hangs |
| `cudnn.enabled = False` | ROCm device detected | Prevents MIOpen convolution hangs (NVIDIA cuDNN is NOT disabled) |
| SDPA math-only mode | ROCm device detected | Disables flash/mem-efficient SDPA that hangs on ROCm |
| `vae_decode_cpu: true` | Config setting | CPU VAE decode — fixes gfx1103 GPU hang |
| `float32` CPU offload | `cpu_offload=True` | SDCppBackend CPU-only path |

---

## Working Environment

### Python Version
Python 3.13.x via pyenv or system Python.

### Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### Install PyTorch (ROCm)
```bash
# AMD Phoenix APU (gfx1103)
pip install torch torchvision torchaudio --index-url https://stable.repo.amd.com/rocm/whl-next/

# Or with device-specific extras (recommended for gfx1102/gfx1151)
pip install "torch[device-gfx1103]" --index-url https://stable.repo.amd.com/rocm/whl-next/
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Configuration File
**Location:** `~/.config/alice/config.yaml` (or set `ALICE_CONFIG` env var)

```yaml
server:
  host: 0.0.0.0
  port: 8090

models:
  directory: /home/deck/.local/share/alice/models

generation:
  # AMD Phoenix APU (gfx1103) — uncomment the block below:
  # force_float32: true
  # vae_decode_cpu: true
  # backend: vulkan

  # AMD Strix Halo (gfx1151) — uncomment the block below:
  # force_bfloat16: true
  # vae_decode_cpu: true
  # backend: auto

  default_steps: 20
  default_scheduler: dpm++_sde_karras

storage:
  images_directory: /home/deck/.local/share/alice/images

logging:
  level: INFO
  file: /home/deck/.local/share/alice/logs/alice.log
```

### Systemd Service
**Location:** `~/.config/systemd/user/alice.service`

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

# AMD GPU environment variables
Environment="MIOPEN_DEBUG_FIND_ALL=0"
Environment="PYTORCH_ROCM_ARCH=gfx1103"
Environment="PYTORCH_ALLOC_CONF=expandable_segments:True"
Environment="ALICE_CONFIG=/home/deck/.config/alice/config.yaml"
Environment="TMPDIR=/home/deck/tmp"

[Install]
WantedBy=default.target
```

---

## AMD CPU Thread Optimization

ALICE auto-configures PyTorch CPU threads to use half the available cores
for intra-op parallelism. On a Ryzen 7 7840U (8-core/16-thread), this uses
8 threads for model operations and reserves the rest for system overhead.

For Vulkan (sd.cpp) backend, CPU threads are controlled by
`generation.sdcpp_threads` (default 8). Set to match your core count:

```yaml
generation:
  sdcpp_threads: 16   # Full core count on 16-thread APU
```

---

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
  -d '{"model": "sd/stable-diffusion-v1-5", "messages": [{"role": "user", "content": "a cat"}]}'
```

### Check PyTorch GPU Support
```bash
python3 -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.is_available())"
```

### Common Errors

**"HIP error: invalid device function"**
- Cause: GPU architecture not in the PyTorch ROCm build
- Solution: Install with the correct device extra (`torch[device-gfx1103]`)

**"HSA_STATUS_ERROR_INVALID_ISA"**
- Cause: PyTorch was built without your GPU's architecture
- Solution: Use the device-specific extras from `stable.repo.amd.com/rocm/whl-next/`

**GPU hang during VAE decode**
- Cause: FP16/BF16 VAE decode on gfx1103
- Solution: Set `vae_decode_cpu: true` in config.yaml

**GPU hang during MIOpen solver search**
- Cause: MIOpen trying all solver variants
- Solution: Set `MIOPEN_DEBUG_FIND_ALL=0` (applied automatically in pytorch_backend.py)

**"No module named torch"**
- Cause: Wrong Python environment
- Solution: Ensure systemd service uses the correct venv Python

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
