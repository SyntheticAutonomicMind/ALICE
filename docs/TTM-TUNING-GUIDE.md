# TTM Memory Tuning Guide for AMD APUs

**Created:** December 25, 2025  
**Updated:** Based on [Jeff Geerling's AMD APU VRAM allocation guide](https://www.jeffgeerling.com/blog/2025/increasing-vram-allocation-on-amd-ai-apus-under-linux)  
**Target:** AMD64 APU devices with integrated graphics (Unified Memory Architecture)  
**Scope:** Increasing GPU memory allocation for AI/ML workloads on AMD APUs

---

## Table of Contents

1. [Overview](#overview)
2. [Understanding TTM Parameters](#understanding-ttm-parameters)
3. [Memory Configuration Examples](#memory-configuration-examples)
4. [Configuration Methods](#configuration-methods)
5. [Verification and Monitoring](#verification-and-monitoring)
6. [Troubleshooting](#troubleshooting)
7. [References](#references)

---

## Overview

### What is TTM?

TTM (Translation Table Manager) is the kernel subsystem that manages GPU memory allocation in Linux. For AMD APUs with integrated graphics, TTM controls how much system RAM can be allocated for GPU use.

**Key concepts:**

- **VRAM (Video RAM)** - On APUs, typically a small reserved portion of system RAM (512MB-2GB)
- **GTT (Graphics Translation Table)** - Dynamic GPU-accessible system memory pool
- **TTM parameters** - Kernel settings that control maximum GPU memory allocation
- **Unified Memory Architecture (UMA)** - APU shares physical RAM between CPU and GPU

### Why Configure TTM on APUs?

**Default limitations:**
- Most Linux distributions allocate only 512MB-2GB for GPU use by default
- BIOS/UEFI settings often cap at 2GB-4GB maximum
- This severely limits AI/ML workloads (Stable Diffusion, LLMs)
- Large models require 8GB-32GB+ of GPU-accessible memory

**Benefits of increasing TTM allocation:**
- Run large AI models (SDXL, FLUX, LLMs) entirely on GPU
- Load entire models into VRAM instead of CPU/GPU swapping
- Faster inference times (eliminates CPU fallback overhead)
- Better stability for long-running AI workloads
- Improved multi-tasking with GPU-accelerated applications

**Trade-offs:**
- Less RAM available for system/CPU tasks
- Requires careful balancing for system stability
- May need to close memory-intensive applications during AI workloads

---

## Understanding TTM Parameters

### CRITICAL: Both Parameters Must Match

**IMPORTANT:** On AMD APUs, set **both parameters to the same value**:

```bash
ttm.pages_limit=XXXXXX ttm.page_pool_size=XXXXXX
```

This differs from discrete GPU configuration. APUs dynamically allocate GPU memory from system RAM, and both parameters must match to ensure the full allocation is usable.

### Calculation Formula

**Primary formula (Jeff Geerling):**
```
pages_value = ([target GB] * 1024 * 1024) / 4.096
```

**Simplified formula (this guide):**
```
pages_value = ([target GB] * 1024 * 1024) / 4
```

The simplified formula is easier to calculate and the difference is negligible (<3%).

**Quick calculation examples:**
- 2GB: `(2 * 1024 * 1024) / 4 = 524,288`
- 4GB: `(4 * 1024 * 1024) / 4 = 1,048,576`
- 8GB: `(8 * 1024 * 1024) / 4 = 2,097,152`
- 12GB: `(12 * 1024 * 1024) / 4 = 3,145,728`
- 16GB: `(16 * 1024 * 1024) / 4 = 4,194,304`
- 24GB: `(24 * 1024 * 1024) / 4 = 6,291,456`
- 32GB: `(32 * 1024 * 1024) / 4 = 8,388,608`
- 48GB: `(48 * 1024 * 1024) / 4 = 12,582,912`
- 64GB: `(64 * 1024 * 1024) / 4 = 16,777,216`
- 108GB: `(108 * 1024 * 1024) / 4 = 28,311,552` (tested maximum on 128GB system)

### How Much Should You Allocate?

**General guidelines:**

| Total RAM | Conservative | Balanced | Aggressive | Use Case |
|-----------|--------------|----------|------------|----------|
| 8GB | 2GB | 3GB | 4GB | Light SD1.5, basic tasks |
| 12GB | 4GB | 6GB | 8GB | SD1.5/SDXL, gaming |
| 16GB | 6GB | 8GB | 10GB | SDXL, moderate AI workloads |
| 24GB | 10GB | 12GB | 16GB | FLUX, large models |
| 32GB | 12GB | 16GB | 20GB | Professional AI/ML |
| 48GB | 16GB | 24GB | 32GB | Large LLMs, research |
| 64GB | 24GB | 32GB | 40GB | Enterprise AI workloads |
| 128GB | 64GB | 96GB | 108GB | Multi-node LLM clustering |

**Recommendations by workload:**

- **SD 1.5 models:** 4GB minimum
- **SDXL models:** 8GB minimum, 12GB recommended
- **FLUX models:** 12GB minimum, 16GB+ recommended
- **Small LLMs (7B-13B):** 8GB-12GB
- **Medium LLMs (30B-70B):** 24GB-48GB
- **Large LLMs (405B):** 96GB-108GB (multi-node)

**Safety margin:** Always leave at least 4GB-8GB for the operating system and background processes.

---

## Deprecated: amdgpu.gttsize

**DO NOT USE:** The old `amdgpu.gttsize` parameter is deprecated since kernel 6.x and shows this warning:

```
[drm] Configuring gttsize via module parameter is deprecated, 
please use ttm.pages_limit
```

**Migration from old parameter:**

| Old Parameter | New Parameters |
|---------------|----------------|
| `amdgpu.gttsize=16384` | `ttm.pages_limit=4194304 ttm.page_pool_size=4194304` |
| `amdgpu.gttsize=24576` | `ttm.pages_limit=6291456 ttm.page_pool_size=6291456` |

---

## Memory Configuration Examples

### 8GB System Configuration

**Target:** Budget APU (Ryzen 5000/6000 series), light Stable Diffusion use

**Conservative (2GB GPU):**
```bash
ttm.pages_limit=524288 ttm.page_pool_size=524288
# Leaves 6GB for system - safest option
```

**Balanced (3GB GPU) - RECOMMENDED:**
```bash
ttm.pages_limit=786432 ttm.page_pool_size=786432
# Leaves 5GB for system - good for SD1.5
```

**Aggressive (4GB GPU):**
```bash
ttm.pages_limit=1048576 ttm.page_pool_size=1048576
# Leaves 4GB for system - ONLY if dedicated AI workstation
# High OOM risk - enable 8GB+ swap
```

---

### 12GB System Configuration

**Target:** Mid-range APU (Ryzen 7840U/780M, Steam Deck-class), SDXL capable

**Conservative (4GB GPU):**
```bash
ttm.pages_limit=1048576 ttm.page_pool_size=1048576
# Leaves 8GB for system - very safe
```

**Balanced (6GB GPU) - RECOMMENDED:**
```bash
ttm.pages_limit=1572864 ttm.page_pool_size=1572864
# Leaves 6GB for system - ideal for SDXL
# Good for Steam Deck / portable gaming PCs
```

**Aggressive (8GB GPU):**
```bash
ttm.pages_limit=2097152 ttm.page_pool_size=2097152
# Leaves 4GB for system - dedicated AI service only
# Example: ALICE server on Steam Deck
```

---

### 16GB System Configuration

**Target:** Gaming APU (Ryzen 7000 series), FLUX capable, content creation

**Conservative (6GB GPU):**
```bash
ttm.pages_limit=1572864 ttm.page_pool_size=1572864
# Leaves 10GB for system - very safe for multi-tasking
```

**Balanced (8GB GPU) - RECOMMENDED:**
```bash
ttm.pages_limit=2097152 ttm.page_pool_size=2097152
# Leaves 8GB for system - ideal for most users
# Perfect for SDXL and moderate FLUX use
```

**Aggressive (10GB GPU):**
```bash
ttm.pages_limit=2621440 ttm.page_pool_size=2621440
# Leaves 6GB for system - GPU-heavy workflows
# Good for FLUX development
```

**Maximum (12GB GPU):**
```bash
ttm.pages_limit=3145728 ttm.page_pool_size=3145728
# Leaves 4GB for system - dedicated GPU server ONLY
```

---

### 24GB System Configuration  

**Target:** High-end APU (Strix Point/Ryzen AI 9 HX 370), professional AI workstation

**Conservative (10GB GPU):**
```bash
ttm.pages_limit=2621440 ttm.page_pool_size=2621440
# Leaves 14GB for system - very conservative
```

**Balanced (12GB GPU) - RECOMMENDED:**
```bash
ttm.pages_limit=3145728 ttm.page_pool_size=3145728
# Leaves 12GB for system - excellent balance
# Perfect for FLUX and LLM experimentation
```

**Aggressive (16GB GPU):**
```bash
ttm.pages_limit=4194304 ttm.page_pool_size=4194304
# Leaves 8GB for system - still safe
# Good for larger LLMs (30B-70B models)
```

**Maximum (18GB GPU):**
```bash
ttm.pages_limit=4718592 ttm.page_pool_size=4718592
# Leaves 6GB for system - extreme workloads only
```

---

### 32GB System Configuration

**Target:** Enthusiast build, AI development, professional use

**Conservative (12GB GPU):**
```bash
ttm.pages_limit=3145728 ttm.page_pool_size=3145728
# Leaves 20GB for system - very safe
```

**Balanced (16GB GPU) - RECOMMENDED:**
```bash
ttm.pages_limit=4194304 ttm.page_pool_size=4194304
# Leaves 16GB for system - perfect 50/50 split
# Ideal for AI development and multi-tasking
```

**Aggressive (20GB GPU):**
```bash
ttm.pages_limit=5242880 ttm.page_pool_size=5242880
# Leaves 12GB for system
# Good for LLM fine-tuning
```

**Maximum (24GB GPU):**
```bash
ttm.pages_limit=6291456 ttm.page_pool_size=6291456
# Leaves 8GB for system - dedicated AI server
```

---

### 48GB System Configuration

**Target:** Future Strix Halo APUs, professional AI/research workstation

**Conservative (16GB GPU):**
```bash
ttm.pages_limit=4194304 ttm.page_pool_size=4194304
# Leaves 32GB for system
```

**Balanced (24GB GPU) - RECOMMENDED:**
```bash
ttm.pages_limit=6291456 ttm.page_pool_size=6291456
# Leaves 24GB for system - excellent balance
# Great for large-scale AI workflows
```

**Aggressive (32GB GPU):**
```bash
ttm.pages_limit=8388608 ttm.page_pool_size=8388608
# Leaves 16GB for system - still safe for most workloads
```

**Maximum (36GB GPU):**
```bash
ttm.pages_limit=9437184 ttm.page_pool_size=9437184
# Leaves 12GB for system - extreme AI workloads
```

---

### 64GB System Configuration

**Target:** Server/workstation-class APU systems, enterprise AI

**Conservative (20GB GPU):**
```bash
ttm.pages_limit=5242880 ttm.page_pool_size=5242880
# Leaves 44GB for system - very conservative
```

**Balanced (32GB GPU) - RECOMMENDED:**
```bash
ttm.pages_limit=8388608 ttm.page_pool_size=8388608
# Leaves 32GB for system - perfect 50/50 split
# Ideal for enterprise AI inference servers
```

**Aggressive (40GB GPU):**
```bash
ttm.pages_limit=10485760 ttm.page_pool_size=10485760
# Leaves 24GB for system
```

**Maximum (48GB GPU):**
```bash
ttm.pages_limit=12582912 ttm.page_pool_size=12582912
# Leaves 16GB for system - dedicated GPU compute
```

---

### 128GB System Configuration

**Target:** Strix Halo APU (Framework Mainboard AI Cluster), large-scale LLM inference

**Based on Jeff Geerling's testing with AI Max+ 395 APUs**

**Conservative (64GB GPU):**
```bash
ttm.pages_limit=16777216 ttm.page_pool_size=16777216
# Leaves 64GB for system - very safe
```

**Balanced (96GB GPU):**
```bash
ttm.pages_limit=25165824 ttm.page_pool_size=25165824
# Leaves 32GB for system
# Maximum allocation possible via BIOS on most boards
```

**Maximum Stable (108GB GPU) - Jeff Geerling's tested maximum:**
```bash
ttm.pages_limit=28311552 ttm.page_pool_size=28311552
# Leaves 20GB for system
# Highest stable allocation for Llama 3.1 405B multi-node clustering
# Going above 108GB causes segfaults during model loading
```

**Experimental (110GB GPU - UNSTABLE):**
```bash
ttm.pages_limit=28835840 ttm.page_pool_size=28835840
# Jeff Geerling reports segfaults at this level
# Do NOT use in production
```

---

## Configuration Methods

### Method 1: GRUB Bootloader (Most Common)

**Supported distros:** Debian, Ubuntu, Arch Linux, SteamOS, Fedora (with GRUB)

1. Edit GRUB configuration:
   ```bash
   sudo nano /etc/default/grub
   ```

2. Find the line starting with `GRUB_CMDLINE_LINUX_DEFAULT` and add TTM parameters:
   ```bash
   GRUB_CMDLINE_LINUX_DEFAULT="quiet splash ttm.pages_limit=4194304 ttm.page_pool_size=4194304"
   ```

3. Update GRUB configuration:
   ```bash
   # Debian/Ubuntu/SteamOS
   sudo update-grub
   
   # Arch Linux
   sudo grub-mkconfig -o /boot/grub/grub.cfg
   ```

4. Reboot to apply changes:
   ```bash
   sudo reboot
   ```

---

### Method 2: Fedora with grubby (Jeff Geerling's Method)

**Supported distros:** Fedora, RHEL, CentOS Stream

1. Add TTM parameters using grubby:
   ```bash
   # For 16GB GPU allocation
   sudo grubby --update-kernel=ALL --args='ttm.pages_limit=4194304'
   sudo grubby --update-kernel=ALL --args='ttm.page_pool_size=4194304'
   ```

2. Verify the parameters were added:
   ```bash
   sudo grubby --info=ALL | grep ttm
   ```

3. Reboot:
   ```bash
   sudo reboot
   ```

**To remove parameters:**
```bash
sudo grubby --update-kernel=ALL --remove-args='ttm.pages_limit'
sudo grubby --update-kernel=ALL --remove-args='ttm.page_pool_size'
```

---

### Method 4: Module Configuration File (Alternative)

**Pros:** No bootloader editing, easier to revert  
**Cons:** Applied after kernel module load, still requires reboot

1. Create TTM module configuration:
   ```bash
   sudo nano /etc/modprobe.d/ttm.conf
   ```

2. Add parameters:
   ```bash
   options ttm pages_limit=4194304 page_pool_size=4194304
   ```

3. Rebuild initramfs to include changes:
   ```bash
   # Debian/Ubuntu
   sudo update-initramfs -u
   
   # Arch/SteamOS
   sudo mkinitcpio -P
   
   # Fedora
   sudo dracut --force
   ```

4. Reboot:
   ```bash
   sudo reboot
   ```

---

## Verification and Monitoring

### Check Current TTM Settings

After rebooting, verify your settings were applied:

```bash
# View kernel command line (should show ttm parameters)
cat /proc/cmdline | grep ttm

# Check current TTM parameter values
cat /sys/module/ttm/parameters/pages_limit
cat /sys/module/ttm/parameters/page_pool_size

# Calculate GB values
echo "pages_limit GB: $(( $(cat /sys/module/ttm/parameters/pages_limit) * 4 / 1024 / 1024 ))"
echo "page_pool_size GB: $(( $(cat /sys/module/ttm/parameters/page_pool_size) * 4 / 1024 / 1024 ))"
```

### Verify GPU Memory Allocation

Check that the AMD driver sees the increased allocation:

```bash
# Check kernel messages for amdgpu memory detection
sudo dmesg | grep "amdgpu.*memory"

# Expected output (example for 16GB allocation):
# [drm] amdgpu: 512M of VRAM memory ready
# [drm] amdgpu: 16000M of GTT memory ready.
```

### Monitor GPU Memory Usage

**Install monitoring tools:**

```bash
# Arch/SteamOS
sudo pacman -S radeontop

# Debian/Ubuntu  
sudo apt install radeontop

# Fedora
sudo dnf install radeontop
```

**Monitor memory in real-time:**

```bash
# GPU-specific monitoring
radeontop

# General system memory
watch -n1 free -h

# AMD ROCm system management (if ROCm installed)
rocm-smi --showmeminfo
```

### Test GPU Memory Allocation

Create a test script to validate your settings:

```python
#!/usr/bin/env python3
"""Test GPU memory allocation with PyTorch"""
import torch
import sys

def test_gpu_memory(sizes_gb=[1, 2, 4, 8, 12, 16]):
    if not torch.cuda.is_available():
        print("ERROR: No CUDA/ROCm device available")
        print(f"PyTorch version: {torch.__version__}")
        sys.exit(1)
    
    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print()
    
    for size_gb in sizes_gb:
        try:
            # Allocate tensor (size_gb * 1024^3 / 4 bytes per float32)
            elements = (size_gb * 1024 * 1024 * 1024) // 4
            tensor = torch.randn(elements, device=device, dtype=torch.float32)
            print(f"✓ Successfully allocated {size_gb}GB")
            del tensor
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"✗ Failed to allocate {size_gb}GB")
            print(f"  Error: {str(e)[:100]}")
            break
    
    print(f"\nPeak memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

if __name__ == "__main__":
    test_gpu_memory()
```

Save as `test_ttm_allocation.py` and run:
```bash
python3 test_ttm_allocation.py
```

---

## Troubleshooting

### Issue: Parameters Not Applied After Reboot

**Symptoms:**
- `cat /proc/cmdline` doesn't show ttm parameters
- `cat /sys/module/ttm/parameters/pages_limit` shows old/default value

**Solutions:**

1. **Verify bootloader syntax:**
   ```bash
   # Correct:
   ttm.pages_limit=4194304 ttm.page_pool_size=4194304
   
   # WRONG (do not use):
   ttm.pages_limit: 4194304
   ttm.pages_limit 4194304
   amdttm.pages_limit=4194304  # "amdttm" is incorrect!
   ```

2. **Check if GRUB was actually updated:**
   ```bash
   # Debian/Ubuntu
   cat /boot/grub/grub.cfg | grep ttm
   
   # Arch Linux
   cat /boot/grub/grub.cfg | grep ttm
   ```

3. **Ensure you edited the right boot entry (systemd-boot):**
   ```bash
   # Check which entry is default
   cat /boot/loader/loader.conf
   
   # Verify your entry has the parameters
   cat /boot/loader/entries/YOUR_ENTRY.conf | grep ttm
   ```

---

### Issue: Out of Memory (OOM) Errors

**Symptoms:**
- System freezes or becomes unresponsive
- Applications crash with "Cannot allocate memory"
- Kernel messages: `Out of memory: Killed process XXXX`
- `journalctl` shows OOM killer messages

**Solutions:**

1. **Reduce GPU allocation:**
   ```bash
   # If using 16GB on a 24GB system, try 12GB instead
   ttm.pages_limit=3145728 ttm.page_pool_size=3145728
   ```

2. **Increase swap space:**
   ```bash
   # Check current swap
   free -h
   
   # Create 16GB swap file
   sudo fallocate -l 16G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   
   # Make permanent
   echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
   ```

3. **Close memory-intensive applications:**
   - Web browsers (especially Chromium/Chrome)
   - IDEs (VSCode, JetBrains products)
   - VMs or containers
   - Background applications

4. **Check for memory leaks:**
   ```bash
   # Monitor memory usage over time
   watch -n1 free -h
   
   # Check which processes are using the most memory
   ps aux --sort=-%mem | head -20
   ```

---

### Issue: GPU Not Detecting Full Allocation

**Symptoms:**
- `dmesg | grep amdgpu.*memory` shows less than expected
- `rocm-smi` shows smaller memory pool
- AI models fail to load despite sufficient TTM allocation

**Solutions:**

1. **Check BIOS/UEFI UMA settings:**
   - Reboot into BIOS/UEFI (usually Del, F2, or F12 during boot)
   - Find "UMA Frame Buffer Size" or "iGPU Memory" setting
   - Set to "Auto" or maximum available (2GB-4GB typical)
   - Some systems require this to be set for TTM to work properly

2. **Verify amdgpu driver loaded correctly:**
   ```bash
   lsmod | grep amdgpu
   dmesg | grep amdgpu | grep error
   ```

3. **Check for conflicting parameters:**
   ```bash
   # Remove any old amdgpu.gttsize parameters
   cat /proc/cmdline | grep gttsize
   
   # Should only see ttm.pages_limit, not gttsize
   ```

---

### Issue: System Instability / GPU Hangs

**Symptoms:**
- Random freezes when using GPU
- GPU reset messages in dmesg: `[drm:amdgpu_job_timedout]`
- Display driver crashes (especially Wayland)

**Solutions:**

1. **Start with conservative allocation and increase gradually:**
   ```bash
   # Begin with 50% of RAM
   # 16GB system → start with 8GB GPU allocation
   ttm.pages_limit=2097152 ttm.page_pool_size=2097152
   ```

2. **Check for conflicting kernel parameters:**
   ```bash
   # Some parameters can conflict with high TTM allocations
   cat /proc/cmdline
   
   # If you have amd_iommu=on, try without it (Jeff Geerling noted this)
   # Remove or disable amd_iommu if present
   ```

3. **Verify ROCm/driver compatibility:**
   ```bash
   # Check if ROCm is properly installed
   rocminfo | grep "Name:" | head -5
   
   # Verify PyTorch ROCm support
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

4. **Check kernel messages for hints:**
   ```bash
   sudo journalctl -k -b | grep -i "ttm\|amdgpu\|drm" | tail -50
   ```

---

### Issue: ALICE / Stable Diffusion Still Using CPU

**Symptoms:**
- ALICE reports GPU available but generation is slow
- `radeontop` shows 0% GPU usage during generation
- Generation times match CPU performance (~6s/step)

**Solutions:**

1. **Verify ALICE configuration uses GPU:**
   
   Edit `/etc/alice/config.yaml` or `~/.config/alice/config.yaml`:
   ```yaml
   generation:
     force_cpu: false  # CRITICAL: must be false for GPU use
     force_float32: true
     device_map: sequential  # For single-file models
     # OR
     device_map: balanced  # For diffusers directories
   ```

2. **Check PyTorch ROCm installation:**
   ```bash
   # For Phoenix APU (gfx1103), must use TheRock builds
   source ~/alice/venv/bin/activate
   python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
   
   # Should show:
   # 2.10.0+rocmXXX (or similar)
   # True
   ```

3. **Verify environment variables in service:**
   ```bash
   systemctl --user cat alice | grep Environment
   
   # Should include:
   # Environment="PYTORCH_ROCM_ARCH=gfx1103"
   # Environment="HSA_OVERRIDE_GFX_VERSION=11.0.0"
   ```

4. **Test GPU directly:**
   ```bash
   ~/alice/venv/bin/python3 -c "
   import torch
   print(f'CUDA available: {torch.cuda.is_available()}')
   if torch.cuda.is_available():
       x = torch.randn(1000, 1000, device='cuda')
       print(f'GPU tensor created successfully')
       print(f'Device name: {torch.cuda.get_device_name(0)}')
   "
   ```

---

## Best Practices

### General Guidelines

1. **Start Conservative, Increase Gradually**
   - Begin with 50% RAM allocation to GPU
   - Monitor system stability for 24-48 hours
   - Increase by 2GB-4GB increments if needed
   - Test with actual workloads (SD generation, LLM inference)

2. **Leave Adequate System Headroom**
   - **Minimum 4GB** for system processes on any configuration
   - **8GB+** recommended for desktop/multi-tasking use
   - **16GB+** for systems running Docker, VMs, or development tools

3. **Enable Swap as Safety Buffer**
   - Even with high RAM, swap prevents hard OOM crashes
   - Recommended: swap = 50-100% of RAM size
   - Use swapfile instead of swap partition for flexibility

4. **Match Configuration to Use Case**
   - **Casual use:** Conservative allocation
   - **Dedicated AI server:** Aggressive allocation  
   - **Workstation/multi-tasking:** Balanced allocation
   - **Development/testing:** Conservative (need RAM for IDEs, builds)

5. **Monitor Before Committing**
   ```bash
   # Watch memory usage during typical workload
   watch -n1 free -h
   
   # Check for OOM events
   sudo journalctl -k -b | grep -i "out of memory"
   
   # Monitor GPU memory specifically
   radeontop  # Real-time GPU monitoring
   ```

---

### APU-Specific Recommendations

#### Phoenix APU (gfx1103) - Ryzen 7840U, 780M iGPU

**Example: 12GB RAM system (Steam Deck OLED-class)**

```bash
# Recommended for ALICE server use
ttm.pages_limit=1572864 ttm.page_pool_size=1572864  # 6GB GPU, 6GB system
```

**BIOS/UEFI settings:**
- UMA Frame Buffer Size: Auto or 2GB
- IOMMU: Enabled (helps memory management)

**Kernel parameters (combine with TTM):**
```bash
amdgpu.dpm=1 amdgpu.gpu_recovery=1 iomem=relaxed
```

**ALICE config.yaml settings:**
```yaml
generation:
  force_cpu: false
  force_float32: true  # FP32 more stable than FP16 on Phoenix
  device_map: sequential  # For single-file .safetensors models
  vae_slicing: true
  attention_slicing: auto
```

**ROCm installation:**
- Use TheRock nightly builds: https://rocm.nightlies.amd.com/v2/gfx110X-all/
- Official PyTorch ROCm DOES NOT support gfx1103 (causes segfaults)
- See [THEROCK-GFX1103-INSTALL.md](THEROCK-GFX1103-INSTALL.md)

---

#### Strix Point (gfx1103) - Ryzen AI 9 HX 370, RDNA 3.5

**Example: 24GB RAM system**

```bash
# Recommended for professional AI work
ttm.pages_limit=3145728 ttm.page_pool_size=3145728  # 12GB GPU, 12GB system
```

**Same ROCm requirements as Phoenix** (TheRock builds required).

---

#### Strix Halo (AI Max+ 395) - Future High-End APU

**Example: 128GB RAM system (Jeff Geerling's cluster)**

```bash
# Maximum tested stable allocation
ttm.pages_limit=28311552 ttm.page_pool_size=28311552  # 108GB GPU, 20GB system
```

**Notes:**
- This is the maximum Jeff Geerling found stable for Llama 3.1 405B multi-node inference
- Going above 108GB (e.g., 110GB) causes segfaults during model loading
- Requires proper cooling (APU can pull significant power under load)

---

### Coordination with Other Settings

TTM allocation should be coordinated with other system settings:

1. **BIOS UMA allocation:**
   - Set to Auto or maximum (usually 2GB-4GB)
   - TTM provides additional dynamic allocation beyond BIOS limit

2. **Kernel swap:**
   - Always enable swap even with high RAM
   - Prevents hard crashes on OOM conditions
   - Recommended size: 8GB-16GB or 50% of RAM

3. **ALICE configuration:**
   - Disable `force_cpu` to use GPU
   - Enable memory optimizations (VAE slicing, attention slicing)
   - Use appropriate `device_map` for model format

4. **Docker/Podman containers:**
   - Set memory limits if running ALICE in containers
   - Leave headroom for host system and GPU allocation

---

## Calculator Script

Save this as `ttm-calculator.sh` for easy conversions:

```bash
#!/bin/bash
# TTM Calculator - Convert between GB and TTM pages_limit values

if [ $# -eq 0 ]; then
    echo "Usage: $0 <GB|pages> <value>"
    echo ""
    echo "Examples:"
    echo "  $0 GB 16        # Convert 16GB to pages"
    echo "  $0 pages 4194304 # Convert pages to GB"
    exit 1
fi

if [ "$1" == "GB" ] || [ "$1" == "gb" ]; then
    gb=$2
    pages=$(( gb * 1024 * 1024 / 4 ))
    echo "${gb}GB = ${pages} pages"
    echo ""
    echo "Add to kernel command line:"
    echo "ttm.pages_limit=${pages} ttm.page_pool_size=${pages}"
    
elif [ "$1" == "pages" ]; then
    pages=$2
    gb=$(echo "scale=2; $pages * 4 / 1024 / 1024" | bc)
    echo "${pages} pages = ${gb}GB"
    
else
    echo "Error: First argument must be 'GB' or 'pages'"
    exit 1
fi
```

**Make executable and use:**
```bash
chmod +x ttm-calculator.sh

./ttm-calculator.sh GB 16
# Output:
# 16GB = 4194304 pages
# Add to kernel command line:
# ttm.pages_limit=4194304 ttm.page_pool_size=4194304

./ttm-calculator.sh pages 4194304
# Output:
# 4194304 pages = 16.00GB
```

---

## Quick Reference Table

| Total RAM | Balanced GPU Allocation | pages_limit & page_pool_size |
|-----------|-------------------------|------------------------------|
| 8GB | 3GB | `786432` |
| 12GB | 6GB | `1572864` |
| 16GB | 8GB | `2097152` |
| 24GB | 12GB | `3145728` |
| 32GB | 16GB | `4194304` |
| 48GB | 24GB | `6291456` |
| 64GB | 32GB | `8388608` |
| 128GB | 108GB (max stable) | `28311552` |

**Remember:** Both parameters must be set to the same value on APUs.

---

## References

### Primary Sources

- [Jeff Geerling: Increasing VRAM allocation on AMD AI APUs under Linux](https://www.jeffgeerling.com/blog/2025/increasing-vram-allocation-on-amd-ai-apus-under-linux)
- [AMD Instinct MI300A System Optimization](https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/system-optimization/mi300a.html)
- [Linux Kernel DRM Memory Management](https://www.kernel.org/doc/html/latest/gpu/drm-mm.html)
- [AMD GPU Module Parameters](https://www.kernel.org/doc/html/latest/gpu/amdgpu/module-parameters.html)

### Community Resources

- [Framework Community Forum: iGPU VRAM Allocation Discussion](https://community.frame.work/t/igpu-vram-how-much-can-be-assigned/73081/)
- [linux-ng.de: Preparing AMD APUs for LLM Usage](https://blog.linux-ng.de/2025/07/13/getting-information-about-amd-apus/)
- [Arch Wiki: AMDGPU](https://wiki.archlinux.org/title/AMDGPU)

### ALICE Documentation

- [AMD Phoenix Environment](AMD-PHOENIX-ENVIRONMENT.md) - Phoenix APU specifics
- [AMD Deployment Guide](AMD-DEPLOYMENT-GUIDE.md) - Full deployment instructions
- [TheRock gfx1103 Installation](THEROCK-GFX1103-INSTALL.md) - ROCm for Phoenix APUs

---

## License

SPDX-License-Identifier: GPL-3.0-only  
SPDX-FileCopyrightText: Copyright (c) 2025 Andrew Wyatt (Fewtarius)

This document is part of the ALICE project.
