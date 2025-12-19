<!-- SPDX-License-Identifier: CC-BY-NC-4.0 -->
<!-- SPDX-FileCopyrightText: Copyright (c) 2025 Andrew Wyatt (Fewtarius) -->

# ALICE AMD Deployment Guide

**Target System:** SteamFork (SteamOS-based) on AMD Ryzen 7 7840U with Radeon 780M  
**Created:** December 1, 2025  
**Purpose:** Deploy alice for remote Stable Diffusion generation from SAM

---

## System Overview

### Hardware Specifications
```
CPU:     AMD Ryzen 7 7840U (Zen 4, 8-core/16-thread, 5.1GHz boost)
         Currently: 4 cores online (power saving mode)
iGPU:    AMD Radeon 780M (RDNA 3, 12 CUs, ~4 TFLOPS FP16)
         VRAM: 3GB allocated (shared from system RAM)
RAM:     12GB visible (16GB physical, shared with GPU)
Storage: 1.9TB NVMe, 555GB available on /home
```

### Software Stack
```
OS:      SteamFork (SteamOS-based Arch Linux)
Kernel:  6.17.7-1
Python:  3.13.1
ROCm:    Installed at /opt/rocm
         /dev/kfd present (ROCm device)
Package: pacman (Arch-based)
```

### Current Status
- [x] ROCm drivers installed (amdgpu module loaded)
- [x] /dev/kfd accessible (world rw permissions)
- [x] User in video group
- [ ] PyTorch ROCm not installed
- [ ] alice not deployed

---

## Deployment Plan

### Phase 1: Environment Setup (30 minutes)

#### 1.1 Enable All CPU Cores (Optional, for max performance)
```bash
# SSH to system
ssh deck@2s

# Check current CPU state
lscpu | grep "On-line"

# Enable all cores (requires root)
sudo bash -c 'echo 1 > /sys/devices/system/cpu/cpu4/online'
sudo bash -c 'echo 1 > /sys/devices/system/cpu/cpu5/online'
sudo bash -c 'echo 1 > /sys/devices/system/cpu/cpu6/online'
sudo bash -c 'echo 1 > /sys/devices/system/cpu/cpu7/online'
# ... repeat for cpu8-cpu15

# Or use a loop
for i in {4..15}; do
  sudo bash -c "echo 1 > /sys/devices/system/cpu/cpu$i/online"
done

# Verify
lscpu | grep "On-line"
# Should show: 0-15
```

#### 1.2 Create Python Virtual Environment
```bash
# Create dedicated venv for alice
cd ~
python3 -m venv alice-venv
source ~/alice-venv/bin/activate

# Verify
python3 --version
# Should show: Python 3.13.1
```

#### 1.3 Install PyTorch with ROCm Support
```bash
# Activate venv
source ~/alice-venv/bin/activate

# Install PyTorch with ROCm 6.2 support
# Note: Check https://pytorch.org/get-started/locally/ for latest
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Verify ROCm detection
python3 -c "
import torch
print('PyTorch version:', torch.__version__)
print('ROCm available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('Device name:', torch.cuda.get_device_name(0))
"
```

**Expected output:**
```
PyTorch version: 2.x.x+rocm6.2
ROCm available: True
Device count: 1
Device name: AMD Radeon Graphics
```

**Troubleshooting:**
- If ROCm not detected, ensure user is in `video` and `render` groups
- Check `/dev/kfd` permissions: `ls -la /dev/kfd`
- Try: `HSA_OVERRIDE_GFX_VERSION=11.0.0` for 780M compatibility

#### 1.4 Install Diffusers and Dependencies
```bash
source ~/alice-venv/bin/activate

# Core dependencies
pip install diffusers transformers accelerate safetensors

# API server dependencies
pip install fastapi uvicorn python-multipart pyyaml pillow

# Optional: Better async performance
pip install httpx aiofiles
```

---

### Phase 2: Test ROCm with Z-Image (1-2 hours)

#### 2.1 Download Test Model
```bash
# Create models directory
mkdir -p ~/models/stable-diffusion

# Download stable-diffusion-v1-5 (good baseline for testing)
cd ~/models/stable-diffusion
huggingface-cli download runwayml/stable-diffusion-v1-5 --local-dir stable-diffusion-v1-5

# Or use git LFS
git lfs install
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
```

**Space required:** ~4GB for SD 1.5, ~7GB for SDXL

#### 2.2 Test Generation Script
Create `~/test_zimage_rocm.py`:
```python
#!/usr/bin/env python3
"""Test Z-Image generation on AMD ROCm"""

import os
import time
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline

# Force ROCm device
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'  # May be needed for 780M

model_path = Path.home() / "models/stable-diffusion/stable-diffusion-v1-5"

print(f"PyTorch: {torch.__version__}")
print(f"ROCm available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print()

# Determine device
if torch.cuda.is_available():
    device = "cuda"  # ROCm uses CUDA API
    dtype = torch.float16  # ROCm supports FP16
else:
    device = "cpu"
    dtype = torch.float32
    
print(f"Using device: {device}, dtype: {dtype}")

# Load pipeline
print(f"\nLoading model from: {model_path}")
load_start = time.time()

pipe = ZImagePipeline.from_pretrained(
    str(model_path),
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
)

# For 12GB RAM system, enable sequential offload
if device == "cuda":
    pipe.enable_sequential_cpu_offload()
    print("Sequential CPU offloading enabled (memory optimization)")
else:
    pipe = pipe.to(device)

load_time = time.time() - load_start
print(f"Load time: {load_time:.1f}s")

# Generate test image
print("\nGenerating test image (256x256, 4 steps)...")
gen_start = time.time()

try:
    result = pipe(
        prompt="a red apple on a white table, professional photo",
        num_inference_steps=4,
        guidance_scale=0.0,  # Z-Image uses no CFG
        height=256,
        width=256,
    )
    
    gen_time = time.time() - gen_start
    image = result.images[0]
    
    # Save result
    output_path = Path("/tmp/zimage_rocm_test.png")
    image.save(output_path)
    
    print(f"\n✓ Generation time: {gen_time:.1f}s ({gen_time/4:.1f}s/step)")
    print(f"✓ Image saved: {output_path}")
    print(f"✓ Image size: {image.size}")
    
except Exception as e:
    gen_time = time.time() - gen_start
    print(f"\n✗ Generation failed after {gen_time:.1f}s: {e}")
    
    # Fallback test with CPU
    if device == "cuda":
        print("\nRetrying with CPU fallback...")
        pipe = ZImagePipeline.from_pretrained(
            str(model_path),
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        pipe = pipe.to("cpu")
        
        cpu_start = time.time()
        result = pipe(
            prompt="a red apple on a white table",
            num_inference_steps=2,
            guidance_scale=0.0,
            height=256,
            width=256,
        )
        cpu_time = time.time() - cpu_start
        result.images[0].save("/tmp/zimage_cpu_test.png")
        print(f"✓ CPU fallback worked: {cpu_time:.1f}s")
```

Run the test:
```bash
source ~/alice-venv/bin/activate
python3 ~/test_zimage_rocm.py
```

**Expected Results:**

| Scenario | Expected Time (256×256, 4 steps) |
|----------|----------------------------------|
| ROCm works | ~30-60s total (~8-15s/step) |
| ROCm fails, CPU fallback | ~150s total (~38s/step) |
| Both fail | Debug needed |

#### 2.3 Benchmark Comparison
Create `~/benchmark_amd.py`:
```python
#!/usr/bin/env python3
"""Benchmark ROCm vs CPU performance"""

import os
import time
import torch
from pathlib import Path

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

model_path = Path.home() / "models/stable-diffusion/stable-diffusion-v1-5"

def test_device(device_mode):
    from diffusers import StableDiffusionPipeline
    
    print(f"\n{'='*60}")
    print(f"Testing: {device_mode}")
    print('='*60)
    
    if device_mode == "rocm":
        if not torch.cuda.is_available():
            print("ROCm not available, skipping")
            return None
        dtype = torch.float16
        device = "cuda"
    else:
        dtype = torch.float16  # FP16 also works on CPU
        device = "cpu"
    
    # Load
    load_start = time.time()
    pipe = ZImagePipeline.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    
    if device_mode == "rocm":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe = pipe.to(device)
    
    load_time = time.time() - load_start
    print(f"Load time: {load_time:.1f}s")
    
    # Generate
    steps = 4
    gen_start = time.time()
    
    try:
        result = pipe(
            prompt="a red apple on a white table",
            num_inference_steps=steps,
            guidance_scale=0.0,
            height=256,
            width=256,
        )
        
        gen_time = time.time() - gen_start
        per_step = gen_time / steps
        
        result.images[0].save(f"/tmp/benchmark_{device_mode}.png")
        
        print(f"✓ Generation: {gen_time:.1f}s ({per_step:.1f}s/step)")
        return {'mode': device_mode, 'load': load_time, 'gen': gen_time, 'per_step': per_step}
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return None

# Run benchmarks
results = []

# Test ROCm first
r = test_device("rocm")
if r: results.append(r)

# Clear GPU memory
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Test CPU
r = test_device("cpu")
if r: results.append(r)

# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print('='*60)
for r in results:
    print(f"{r['mode']:10s}: {r['per_step']:.1f}s/step (total: {r['gen']:.1f}s)")

if len(results) == 2:
    speedup = results[1]['per_step'] / results[0]['per_step']
    winner = results[0]['mode'] if results[0]['per_step'] < results[1]['per_step'] else results[1]['mode']
    print(f"\nWinner: {winner} ({speedup:.1f}x faster)")
```

---

### Phase 3: Deploy alice Service (2-3 hours)

#### 3.1 Clone alice Project
```bash
cd ~
git clone <your-alice-repo> alice
# Or copy from local
# scp -r /path/to/alice deck@2s:~/alice
```

#### 3.2 Configure alice
Create `~/alice/config.yaml`:
```yaml
server:
  host: 0.0.0.0
  port: 8080
  # api_key: your-secret-key  # Optional

models:
  directory: /home/deck/models/stable-diffusion
  
storage:
  images_directory: /home/deck/alice-images
  
generation:
  # Auto-detect ROCm, fallback to CPU
  device: auto
  # Use FP16 for memory efficiency
  dtype: float16
  # Enable memory optimization for 12GB system
  low_memory: true
  sequential_offload: true
  
defaults:
  steps: 8
  guidance_scale: 0.0  # For Z-Image
  width: 512
  height: 512
```

#### 3.3 Create Service Wrapper
Create `~/alice/run.sh`:
```bash
#!/bin/bash
# ALICE service wrapper for AMD ROCm

# ROCm compatibility
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Use all CPU cores
for i in {4..15}; do
  sudo bash -c "echo 1 > /sys/devices/system/cpu/cpu$i/online" 2>/dev/null
done

# Activate virtual environment
source ~/alice-venv/bin/activate

# Start server
cd ~/alice
python3 -m src.main
```

Make executable:
```bash
chmod +x ~/alice/run.sh
```

#### 3.4 Create systemd Service
Create `~/.config/systemd/user/alice.service`:
```ini
[Unit]
Description=ALICE Remote Stable Diffusion Service
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/deck/alice
ExecStart=/home/deck/alice/run.sh
Restart=on-failure
RestartSec=5

# Environment
Environment="HOME=/home/deck"
Environment="HSA_OVERRIDE_GFX_VERSION=11.0.0"

[Install]
WantedBy=default.target
```

Enable and start:
```bash
# Enable user service
systemctl --user daemon-reload
systemctl --user enable alice
systemctl --user start alice

# Check status
systemctl --user status alice

# View logs
journalctl --user -u alice -f
```

---

### Phase 4: SAM Integration (1 hour)

#### 4.1 Configure SAM to Use Remote SD
In SAM Preferences → Endpoints:
1. Add new endpoint: "Remote SD (AMD)"
2. Base URL: `http://2s:8080` (or IP address)
3. Provider Type: Remote Stable Diffusion
4. Enable provider

#### 4.2 Test from SAM
```bash
# From SAM system
curl -X POST http://2s:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sd/stable-diffusion-v1-5",
    "messages": [{"role": "user", "content": "a red apple on white table"}],
    "sam_config": {
      "steps": 8,
      "width": 512,
      "height": 512
    }
  }'
```

---

## Memory Optimization for 12GB System

### Z-Image Model Memory Requirements

| Model | Disk | FP32 RAM | FP16 RAM | With Offload |
|-------|------|----------|----------|--------------|
| z-image-8b | 19GB | ~38GB | ~19GB | ~8GB peak |
| z-image-16b | 30GB | ~60GB | ~30GB | ~10GB peak |

### Recommended Settings
```yaml
generation:
  # MUST enable for 12GB system
  sequential_offload: true
  low_memory: true
  
  # FP16 halves memory usage
  dtype: float16
  
  # Smaller default size
  default_width: 512
  default_height: 512
```

### If OOM Occurs
1. Reduce image size (512×512 → 256×256)
2. Use 8B model instead of 16B
3. Close other applications
4. Increase swap (already 7.2GB configured)

---

## Performance Expectations

### Based on M1 Mac Benchmarks
| Config | M1 CPU (76.5s/step) | AMD 7840U (estimated) |
|--------|---------------------|----------------------|
| CPU FP32 | 76.5s/step | ~60-80s/step |
| CPU FP16 | ~53s/step | ~40-60s/step |
| ROCm (if works) | N/A | ~20-40s/step |

### Scaling with Resolution
| Resolution | Relative Time |
|------------|---------------|
| 256×256 | 1× (baseline) |
| 512×512 | ~4× |
| 1024×1024 | ~16× |

### Step Scaling
| Steps | Relative Time |
|-------|---------------|
| 4 | 1× (baseline) |
| 8 | 2× |
| 20 | 5× |

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

# Check swap
swapon --show

# Monitor during generation
watch -n 1 'free -h; nvidia-smi 2>/dev/null || rocm-smi 2>/dev/null'
```

### Service Won't Start
```bash
# Check logs
journalctl --user -u alice -n 50

# Test manually
~/alice/run.sh
```

### Network Not Accessible
```bash
# Check firewall
sudo iptables -L

# Check service is listening
ss -tlnp | grep 8080

# Test locally first
curl http://localhost:8080/health
```

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

### Test Generation
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"sd/stable-diffusion-v1-5","messages":[{"role":"user","content":"test"}]}'
```

### Check GPU Memory
```bash
# ROCm equivalent of nvidia-smi
/opt/rocm/bin/rocm-smi 2>/dev/null || echo "rocm-smi not available"
```

---

## Next Steps

1. [ ] Install PyTorch ROCm
2. [ ] Download Z-Image model
3. [ ] Test ROCm vs CPU performance
4. [ ] Deploy alice service
5. [ ] Configure SAM integration
6. [ ] Benchmark end-to-end latency

---

## References


- [PyTorch ROCm Installation](https://pytorch.org/get-started/locally/)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers/)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [Stable Diffusion Models](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [ALICE Project](../alice/)

