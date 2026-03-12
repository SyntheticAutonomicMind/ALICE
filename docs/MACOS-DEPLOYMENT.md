# ALICE on macOS - Deployment Guide

**Platform:** macOS 13 Ventura or later  
**Architecture:** Apple Silicon (M1/M2/M3/M4) recommended; Intel supported (CPU only)  
**Use case:** Running ALICE as a local image generation service for SAM

---

## Overview

ALICE runs as a user-level process on macOS - no system privileges required. On Apple Silicon, PyTorch uses the Metal Performance Shaders (MPS) backend for GPU-accelerated image generation. Intel Macs fall back to CPU-only inference, which is significantly slower.

### Paths

| Purpose | Path |
|---------|------|
| Application files | `~/Library/Application Support/alice/` |
| Models | `~/Library/Application Support/alice/data/models/` |
| Generated images | `~/Library/Application Support/alice/data/images/` |
| Configuration | `~/.config/alice/config.yaml` |
| Logs | `~/Library/Logs/alice/alice.log` |
| LaunchAgent plist | `~/Library/LaunchAgents/com.alice.plist` |

---

## Installation

### Prerequisites

**Homebrew** (if not already installed):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Python 3.10+:**

```bash
brew install python
python3 --version   # should show 3.10 or higher
```

**Git:**

```bash
brew install git
```

### Install ALICE

```bash
git clone https://github.com/SyntheticAutonomicMind/ALICE.git
cd ALICE
```

Choose how you want ALICE to run:

**Option 1 - Background service (recommended for SAM)**

Installs and immediately starts ALICE as a user LaunchAgent. ALICE will start automatically each time you log in.

```bash
./scripts/install_macos.sh
# or equivalently:
./scripts/install_macos.sh --service
```

**Option 2 - Manual start**

Installs ALICE but does not register a LaunchAgent. You start ALICE yourself when you need it.

```bash
./scripts/install_macos.sh --manual
```

---

## Service Management

ALICE registers as a user LaunchAgent (`com.alice`), not a system daemon. It runs as your user account and does not require any admin privileges after installation.

### Start / Stop

```bash
launchctl start com.alice
launchctl stop com.alice
```

### Restart

```bash
launchctl stop com.alice && launchctl start com.alice
```

### Disable auto-start (keep installed)

```bash
launchctl unload ~/Library/LaunchAgents/com.alice.plist
```

### Re-enable auto-start

```bash
launchctl load ~/Library/LaunchAgents/com.alice.plist
```

### Check status

```bash
launchctl list | grep alice
```

The PID column shows the process ID if running, or `-` if stopped. The exit status column shows the last exit code (0 = clean stop, non-zero = crash).

### Logs

```bash
tail -f ~/Library/Logs/alice/alice.log
```

---

## Manual Start (without LaunchAgent)

If you installed with `--manual`, or want to run ALICE in the foreground for debugging:

```bash
cd ~/Library/"Application Support"/alice
ALICE_CONFIG=~/.config/alice/config.yaml venv/bin/python -m src.main
```

Stop with `Ctrl+C`.

To promote a manual install to a service later:

```bash
# From the original ALICE source directory:
./scripts/install_macos.sh --service
```

---

## Connecting to SAM

1. Make sure ALICE is running (check `http://localhost:8080/health` in a browser)
2. Open SAM
3. Go to **Settings > Image Generation**
4. Set the server URL to: `http://localhost:8080`
5. SAM will auto-discover available models

---

## Adding Models

Place `.safetensors` or `.gguf` model files in:

```
~/Library/Application Support/alice/data/models/
```

ALICE detects models automatically when it starts (or when you call the `/v1/models` endpoint). No restart required for model discovery in most cases.

**Subdirectory layout:**

```
data/models/
├── mymodel.safetensors          # SD 1.5 or SDXL
├── flux-dev.safetensors         # FLUX
├── my_sdxl_diffusers/           # Diffusers directory format
│   ├── model_index.json
│   └── ...
└── loras/                       # LoRA weights
    └── my_lora.safetensors
```

You can also download models directly through the ALICE web UI at `http://localhost:8080/web/` (CivitAI and HuggingFace supported).

---

## Configuration

The configuration file lives at `~/.config/alice/config.yaml`. The installer pre-populates it with sensible macOS defaults. Key settings:

```yaml
server:
  host: 0.0.0.0
  port: 8080

generation:
  backend: pytorch          # pytorch (MPS/CPU) - sd.cpp not available on macOS
  max_concurrent: 1         # keep at 1 for MPS stability

models:
  directory: ~/Library/Application Support/alice/data/models

logging:
  file: ~/Library/Logs/alice/alice.log
  level: INFO
```

After editing, restart ALICE for changes to take effect:

```bash
launchctl stop com.alice && launchctl start com.alice
```

---

## Apple Silicon Notes

### MPS Backend

PyTorch uses Metal Performance Shaders (MPS) on Apple Silicon. MPS acceleration is automatic - no special configuration needed.

**Known MPS limitations:**

- Some model architectures load slower on first run (MPS compilation cache warms up over time)
- MPS does not support all PyTorch operations; ALICE falls back to CPU for unsupported ops automatically
- FLUX models require more memory - 16GB+ unified memory recommended
- SDXL works well on 8GB+ unified memory

### Memory

macOS unified memory is shared between CPU and GPU. If you run out during generation:

- Reduce image resolution
- Use SD 1.5 models instead of SDXL/FLUX
- Close other memory-heavy apps during generation

---

## Intel Mac Notes

Intel Macs do not have GPU acceleration. Expect generation times of several minutes per image at standard resolutions. Smaller models (SD 1.5) and lower resolutions (512x512) help.

There is no Vulkan support on macOS, so the `sd.cpp` backend is not available. ALICE uses the PyTorch CPU backend only.

---

## Uninstall

```bash
# From the original ALICE source directory:
./scripts/install_macos.sh --uninstall
```

This stops and removes the LaunchAgent, then prompts before removing each data directory. Models and generated images are preserved by default.

---

## Troubleshooting

### ALICE won't start

Check logs first:

```bash
tail -50 ~/Library/Logs/alice/alice.log
```

Common causes:

- **Port 8080 in use** - Change `server.port` in `~/.config/alice/config.yaml`
- **Python version mismatch** - Verify with `~/Library/"Application Support"/alice/venv/bin/python --version`
- **Missing dependencies** - Re-run pip: `~/Library/"Application Support"/alice/venv/bin/pip install -r ~/Library/"Application Support"/alice/requirements.txt`

### launchctl shows non-zero exit code

```bash
launchctl list | grep alice
# e.g.: -   78    com.alice  <- exit code 78 means config file error
```

Open `~/.config/alice/config.yaml` and check for YAML syntax errors.

### Generation is very slow

Normal on Intel Mac (CPU only). On Apple Silicon:

- Verify MPS is being used: check logs for "MPS" or "metal"
- First generation after startup is always slower (MPS warmup)
- Subsequent generations on the same model are faster

### "MPS backend out of memory"

Reduce resolution or use a smaller model. You can also add a swap via macOS's built-in memory compression - macOS handles this automatically.

### Reset everything

```bash
./scripts/install_macos.sh --uninstall
# answer Y to all prompts, then reinstall:
./scripts/install_macos.sh
```
