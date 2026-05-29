<!-- SPDX-License-Identifier: CC-BY-NC-4.0 -->
<!-- SPDX-FileCopyrightText: Copyright (c) 2025 Andrew Wyatt (Fewtarius) -->

# ALICE - Artificial Latent Image Composition Engine

**Local AI image generation that runs on your hardware. No subscription, no cloud uploads, no per-image cost.**

I built ALICE for fun. I wanted to generate images on my own hardware without paying per image or uploading prompts to someone else's server. ALICE is a standalone Stable Diffusion service with a web interface, an OpenAI-compatible API, and native integration with SAM. Your prompts and images never leave your hardware.

Use ALICE on its own through the web interface, connect it to SAM for image generation, or integrate it with any client that supports the OpenAI image API.

ALICE is part of [Synthetic Autonomic Mind](https://github.com/SyntheticAutonomicMind).

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.104%2B-green.svg)](https://fastapi.tiangolo.com/)
[![Part of Synthetic Autonomic Mind](https://img.shields.io/badge/Part%20of-Synthetic%20Autonomic%20Mind-blueviolet)](https://github.com/SyntheticAutonomicMind)

[Website](https://www.syntheticautonomicmind.org) | [GitHub](https://github.com/SyntheticAutonomicMind/ALICE) | [Issues](https://github.com/SyntheticAutonomicMind/ALICE/issues)

---

## What You Can Do With ALICE

- **Generate images locally** - Stable Diffusion runs on your hardware. No subscription, no cloud uploads, no per-image cost.
- **Standalone web interface** - Browse, generate, and manage images at `http://localhost:8080/web/`. No other software needed.
- **Use with SAM** - SAM connects to ALICE for image generation. Ask SAM to create an image and ALICE handles it.
- **OpenAI-compatible API** - Any client that speaks the OpenAI image API can use ALICE. Generate images, list models, manage galleries - all via REST.
- **Browse and download models** - Search CivitAI and HuggingFace from ALICE's web interface. One-click download.
- **Private by default** - Generated images are private. Share selectively with time-based expiration.
- **Works on your hardware** - NVIDIA (CUDA), AMD (ROCm, including Steam Deck), Apple Silicon (Metal), or CPU fallback.

---

## Quick Install

ALICE needs two things: the software (installed below) and at least one Stable Diffusion model. See [Getting Models](#getting-models) after installing.

### macOS (SAM integration)

Apple Silicon Macs (M1/M2/M3/M4) get GPU-accelerated generation via Metal. Intel Macs run CPU-only.

**Prerequisites:** [Homebrew](https://brew.sh) and Python 3.10+

```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.10+
brew install python
```

**Install ALICE:**

```bash
git clone https://github.com/SyntheticAutonomicMind/ALICE.git && cd ALICE

# Background service (recommended - starts automatically at login):
./scripts/install_macos.sh

# Or install without the background service (start manually):
./scripts/install_macos.sh --manual
```

The installer detects your Mac's architecture, installs PyTorch with the right backend (MPS for Apple Silicon, CPU for Intel), and sets up the service. It runs entirely as your user account.

**Service management:**

```bash
launchctl start com.alice          # Start
launchctl stop com.alice           # Stop
tail -f ~/Library/Logs/alice/alice.log  # Logs
```

**Connect to SAM:**

1. Make sure ALICE is running - open `http://localhost:8080/health` to check
2. In SAM go to **Settings > Image Generation**
3. Set the server URL to `http://localhost:8080`
4. Download a model (see [Getting Models](#getting-models))
5. Ask SAM to generate an image

For a full macOS setup guide, see [docs/MACOS-DEPLOYMENT.md](docs/MACOS-DEPLOYMENT.md).

### Linux (one-liner)

```bash
TAG="$(curl -s https://api.github.com/repos/SyntheticAutonomicMind/ALICE/releases/latest | python3 -c "import sys,json;print(json.load(sys.stdin)['tag_name'])")"
VERSION="${TAG#v}"
curl -sL "https://github.com/SyntheticAutonomicMind/ALICE/releases/download/${TAG}/alice-${VERSION}.tar.gz" | tar xz && cd alice-* && sudo ./scripts/install.sh
```

Installs as a systemd service to `/opt/alice`. Detects AMD (ROCm), NVIDIA (CUDA), and CPU-only automatically.

### Docker

```bash
git clone https://github.com/SyntheticAutonomicMind/ALICE.git && cd ALICE
make docker-up-cuda   # NVIDIA
make docker-up-rocm   # AMD
make docker-up        # CPU only
```

---

## Getting Models

ALICE needs at least one Stable Diffusion model to generate images. Models are large files (2-10GB) that contain the AI's learned knowledge. ALICE doesn't ship with models - you download them separately.

### Recommended: SDXL

SDXL produces high-quality images at 1024x1024. Requires about 6-8GB of memory (unified memory on Apple Silicon).

- **[Juggernaut XL](https://civitai.com/models/133005)** - Photorealistic, great all-rounder
- **[DreamShaper XL](https://civitai.com/models/112902)** - Versatile, handles artistic styles well
- **[RealVisXL](https://civitai.com/models/139562)** - Realistic portraits and scenes

### For Macs with 8GB or less: SD 1.5

SD 1.5 models are smaller (2-4GB) and work well on lower-memory systems. Native resolution is 512x512.

- **[Realistic Vision](https://civitai.com/models/4201)** - Photorealistic
- **[DreamShaper](https://civitai.com/models/4384)** - General purpose
- **[Deliberate](https://civitai.com/models/4823)** - Detailed and flexible

### How to download

**Option 1 - ALICE web interface (easiest)**

Open `http://localhost:8080/web/` and go to the **Download** tab. Search for a model on CivitAI or HuggingFace and click Download.

**Option 2 - Download manually**

Download a `.safetensors` file from [CivitAI](https://civitai.com) or [HuggingFace](https://huggingface.co) and place it in:

- **macOS:** `~/Library/Application Support/alice/data/models/`
- **Linux:** `/var/lib/alice/models/` (system install) or `~/.local/share/alice/models/` (SteamOS)

ALICE auto-discovers models on startup. In the web interface, go to **Models > Refresh** to pick up new models.

---

## Screenshots

<table>
  <tr>
    <td width="50%">
      <h3>Dashboard & Control Center</h3>
      <img src=".images/ALICE3.png"/>
      <em>Real-time system status, GPU monitoring, and quick generation access</em>
    </td>
    <td width="50%">
      <h3>Advanced Generation Interface</h3>
      <img src=".images/ALICE2.png"/>
      <em>Full parameter controls: model selection, schedulers, guidance scale, dimensions</em>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td width="50%">
      <h3>Private Image Gallery</h3>
      <img src=".images/ALICE1.png"/>
      <em>Organize and manage generated images with privacy controls and metadata</em>
    </td>
  </tr>
</table>

---

## Hardware Support

| Platform | Backend | Notes |
|----------|---------|-------|
| **NVIDIA** | CUDA 12.4 | Recommended. 8GB+ VRAM |
| **AMD** | ROCm 6.2 | Full support including Steam Deck. See [AMD Deployment Guide](docs/AMD-DEPLOYMENT-GUIDE.md) |
| **Apple Silicon** | MPS | M1/M2/M3/M4. Automatic acceleration |
| **CPU** | Fallback | Works on any system. Significantly slower |

---

## Supported Models

- **Stable Diffusion 1.5** - 512x512 native resolution
- **Stable Diffusion 2.x** - 768x768 native resolution
- **SDXL** - 1024x1024 native resolution
- **FLUX** - High-quality model with extended capabilities
- **Custom models** - Any Stable Diffusion variant
- **Formats** - Diffusers directory or single `.safetensors` file

---

## API

ALICE provides an OpenAI-compatible API. Generate images, list models, manage the gallery, and download models - all via REST endpoints.

For complete API documentation, see [docs/API.md](docs/API.md).

---

## Configuration

Edit `config.yaml` to customize ALICE behavior: server settings, model defaults, generation parameters, storage limits, and NSFW filtering.

For full configuration reference, see [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

---

## Requirements

- **Python** 3.10 or newer
- **RAM** 16GB minimum (32GB recommended for SDXL/FLUX)
- **Disk** 50GB+ for model storage
- **GPU** 8GB+ VRAM recommended (4GB for smaller models)

Key dependencies: PyTorch 2.6.0, diffusers 0.35.2, FastAPI 0.104.1. See [requirements.txt](requirements.txt) for the complete list.

---

## Documentation

| Document | What You'll Find |
|----------|-----------------|
| [macOS Deployment](docs/MACOS-DEPLOYMENT.md) | Full macOS setup and troubleshooting |
| [AMD Deployment](docs/AMD-DEPLOYMENT-GUIDE.md) | AMD/ROCm setup including Steam Deck |
| [Architecture](docs/ARCHITECTURE.md) | System design and internals |
| [Implementation Guide](docs/IMPLEMENTATION_GUIDE.md) | Development guide |
| [API Reference](docs/API.md) | Complete API documentation |
| [Configuration](docs/CONFIGURATION.md) | Full config reference |
| [Website](https://www.syntheticautonomicmind.org) | Online guides and updates |

---

## Part of the Ecosystem

ALICE is part of [Synthetic Autonomic Mind](https://github.com/SyntheticAutonomicMind) - a family of open source AI tools:

- **[SAM](https://github.com/SyntheticAutonomicMind/SAM)** - Native macOS AI assistant. SAM connects to ALICE for image generation.
- **[CLIO](https://github.com/SyntheticAutonomicMind/CLIO)** - Terminal AI coding assistant (macOS, Linux, Windows)
- **[SAM-Web](https://github.com/SyntheticAutonomicMind/SAM-web)** - Access SAM from iPad, iPhone, or any browser

---

## License

**GPL-3.0** - See [LICENSE](LICENSE) for details.

Created by Andrew Wyatt (Fewtarius) · [syntheticautonomicmind.org](https://www.syntheticautonomicmind.org) · [github.com/SyntheticAutonomicMind/ALICE](https://github.com/SyntheticAutonomicMind/ALICE)

Built with open source: [Stable Diffusion](https://github.com/CompVis/stable-diffusion) · [diffusers](https://github.com/huggingface/diffusers) · [FastAPI](https://github.com/tiangolo/fastapi) · [PyTorch](https://pytorch.org/)