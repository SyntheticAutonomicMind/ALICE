<!-- SPDX-License-Identifier: CC-BY-NC-4.0 -->
<!-- SPDX-FileCopyrightText: Copyright (c) 2025 Andrew Wyatt (Fewtarius) -->

# Artificial Latent Image Composition Engine (ALICE)
- A Remote Stable Diffusion Service

**Purpose:** OpenAI-compatible remote Stable Diffusion service

---

## Overview

ALICE is a standalone Python service that provides OpenAI-compatible REST API endpoints for Stable Diffusion image generation. It enables SAM clients and other applications to offload GPU-intensive image generation to remote servers.

**Key Features:**
- 🤖 OpenAI-compatible API (`/v1/chat/completions`)
- 🖼️ Stable Diffusion image generation via `diffusers`
- 🔒 Image privacy controls with ownership tracking
- 🖼️ Gallery view for managing generated images
- 🌐 Web-based management interface
- 📥 Model download manager (CivitAI & HuggingFace)
- 🐧 Linux daemon deployment (systemd)
- 🍎 macOS support (launchd)
- 🚀 Model caching for fast subsequent generations
- 📊 Real-time metrics and monitoring
- 🔐 API key authentication with session management
- 🌐 Works completely offline (local fonts, no CDN dependencies)

**Supported Hardware:**
- NVIDIA GPUs (CUDA) - Recommended
- AMD GPUs (ROCm) - See `docs/AMD-DEPLOYMENT-GUIDE.md`
- Apple Silicon (MPS) - Full support
- CPU fallback (slower but works everywhere)

---

## Quick Start

### Development Setup

```bash
# Clone the repository
git clone https://github.com/fewtarius/alice.git
cd alice

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch (platform-specific - see PYTORCH_INSTALL.md)
# For AMD/ROCm:
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/rocm6.2
# For NVIDIA/CUDA:
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
# For CPU only:
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
# For macOS (Apple Silicon):
pip install torch==2.6.0 torchvision==0.21.0

# Install remaining dependencies
pip install -r requirements.txt

# Create required directories
mkdir -p models images logs

# Add a Stable Diffusion model to ./models directory
# (Download from Hugging Face, CivitAI, etc.)

# Start the server
python -m src.main

# Open in browser
open http://localhost:8080/web/
```

**See [PYTORCH_INSTALL.md](PYTORCH_INSTALL.md) for detailed PyTorch installation instructions.**

### Production Deployment (SteamOS/Linux)

For SteamOS (Steam Deck) or other Linux systems with user services:

```bash
# Run the automated installer
./scripts/install_steamos.sh

# The installer will:
# - Detect AMD GPU and configure ROCm (if available)
# - Install PyTorch 2.6.0 with correct backend
# - Install all dependencies matching SAM's versions
# - Create systemd user service
# - Start ALICE automatically

# Add models to ~/.local/share/alice/models

# Check status
systemctl --user status alice

# View logs
journalctl --user -u alice -f
```

### Production Deployment (System-wide Linux/macOS)

```bash
# Run the installation script
sudo ./scripts/install.sh

# Add models to /var/lib/alice/models
# Edit configuration if needed: /etc/alice/config.yaml

# Start the service
sudo systemctl start alice    # Linux
sudo launchctl load /Library/LaunchDaemons/com.alice.plist  # macOS

# Check status
sudo systemctl status alice   # Linux
tail -f /var/log/alice/alice.log  # Both
```

---

## Architecture

```
┌─────────────────┐
│   Client App    │
│   (SAM, etc.)   │
└────────┬────────┘
         │ HTTP POST /v1/chat/completions
         │ {"model": "sd/...", "messages": [...]}
         ▼
┌─────────────────┐
│  ALICE Server   │
│  (FastAPI)      │
├─────────────────┤
│ • Model Registry│  ← Scans ./models for SD models
│ • Generator     │  ← PyTorch + diffusers
│ • Web UI        │  ← Management interface
│ • Image Storage │  ← Serves generated images
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    GPU/CPU      │
│ CUDA/ROCm/MPS   │
└─────────────────┘
```

---

## API Endpoints

### Chat Completions (Image Generation)

```bash
POST /v1/chat/completions
Content-Type: application/json
Authorization: Bearer <api-key>  # Optional

{
  "model": "sd/stable-diffusion-v1-5",
  "messages": [
    {"role": "user", "content": "a serene mountain landscape at sunset"}
  ],
  "sam_config": {
    "negative_prompt": "blurry, low quality",
    "steps": 25,
    "guidance_scale": 7.5,
    "width": 512,
    "height": 512,
    "seed": 42,
    "scheduler": "dpm++_sde_karras"
  }
}
```

**Response (OpenAI-compatible):**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "sd/stable-diffusion-v1-5",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Image generated successfully. URL: http://server:8080/images/abc.png",
      "image_urls": ["http://server:8080/images/abc.png"]
    },
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
}
```

### List Models

```bash
GET /v1/models
```

```json
{
  "object": "list",
  "data": [
    {"id": "sd/stable-diffusion-v1-5", "object": "model", "created": 1234567890, "owned_by": "alice"},
    {"id": "sd/stable-diffusion-xl-base-1.0", "object": "model", "created": 1234567891, "owned_by": "alice"}
  ]
}
```

### Health Check

```bash
GET /health
```

```json
{
  "status": "ok",
  "gpuAvailable": true,
  "modelsLoaded": 1,
  "version": "1.0.0"
}
```

### Metrics

```bash
GET /metrics
```

```json
{
  "queueDepth": 0,
  "gpuUtilization": 0.45,
  "gpuMemoryUsed": "4.2 GB",
  "gpuMemoryTotal": "12.0 GB",
  "modelsLoaded": 1,
  "totalGenerations": 42,
  "avgGenerationTime": 3.5
}
```

### Refresh Models

```bash
POST /v1/models/refresh
```

### Model Downloads (Admin only)

**Search CivitAI:**
```bash
POST /v1/models/search/civitai
Content-Type: application/json

{
  "query": "",           # Empty = browse popular, non-empty = search
  "types": ["Checkpoint"], # Checkpoint, LORA, TextualInversion
  "limit": 100,
  "page": 1
}
```

**Search HuggingFace:**
```bash
POST /v1/models/search/huggingface
Content-Type: application/json

{
  "query": "stable-diffusion",
  "limit": 100
}
```

**Download from CivitAI:**
```bash
POST /v1/models/download/civitai
Content-Type: application/json

{
  "modelId": 4384,
  "versionId": 128713  # Optional
}
```

**Download from HuggingFace:**
```bash
POST /v1/models/download/huggingface
Content-Type: application/json

{
  "repoId": "stabilityai/stable-diffusion-xl-base-1.0"
}
```

### Image Gallery (v1.1.0+)

**List gallery images:**
```bash
GET /v1/gallery?include_public=true&include_private=true&limit=100
```

Returns all images accessible to the current user (own images + public images).

**Update image privacy:**
```bash
PATCH /v1/gallery/{image_id}/privacy
Content-Type: application/json

{
  "isPublic": true,
  "expiresInHours": 168  # Optional, 1-168 hours
}
```

**Delete image:**
```bash
DELETE /v1/gallery/{image_id}
```

**Gallery statistics (admin only):**
```bash
GET /v1/gallery/stats
```

**Cleanup expired images (admin only):**
```bash
POST /v1/gallery/cleanup
```

**Privacy Features:**
- All generated images are **private by default**
- Only the owner can view their private images
- Images can be made public with optional expiration (1-168 hours, default 7 days)
- Public images are accessible to all users until expiration
- Admins can view all images
- Expired public images are automatically deleted hourly

---

## Configuration

**config.yaml:**

```yaml
server:
  host: 0.0.0.0
  port: 8080
  api_key: null  # Set to enable authentication

models:
  directory: ./models
  auto_unload_timeout: 300
  default_model: stable-diffusion-v1-5

generation:
  default_steps: 25
  default_guidance_scale: 7.5
  default_scheduler: dpm++_sde_karras
  max_concurrent: 1
  request_timeout: 300
  default_width: 512
  default_height: 512

storage:
  images_directory: ./images
  max_storage_gb: 100
  retention_days: 7

logging:
  level: INFO
  file: ./logs/alice.log
```

---

## Supported Models

ALICE supports any Stable Diffusion model in:
- **Diffusers format** - Directory with `model_index.json`
- **Single file** - `.safetensors` checkpoint files

**Supported architectures:**
- SD 1.4/1.5 (512x512 native)
- SD 2.0/2.1 (768x768 native)
- SDXL (1024x1024 native)
- SD 3.x
- FLUX

**Model naming:**
- All models use `sd/` prefix
- Model ID matches the directory/file name
- Example: `models/stable-diffusion-v1-5/` → `sd/stable-diffusion-v1-5`

---

## Schedulers

Available schedulers:
- `dpm++_sde_karras` (default, recommended)
- `dpm++_karras`
- `euler_a`
- `euler`
- `ddim`
- `pndm`
- `lms`

---

## Project Structure

```
alice/
├── src/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   ├── config.py         # Configuration management
│   ├── model_registry.py # Model scanning/management
│   ├── generator.py      # Image generation engine
│   ├── gallery.py        # Image gallery & privacy
│   ├── auth.py           # Authentication system
│   ├── downloader.py     # Model download manager
│   └── schemas.py        # Pydantic models
├── web/
│   ├── index.html        # Dashboard
│   ├── models.html       # Model management
│   ├── generate.html     # Generation interface
│   ├── gallery.html      # Image gallery
│   ├── download.html     # Model downloads
│   ├── admin.html        # Admin panel
│   ├── style.css         # Styles
│   └── app.js            # JavaScript client
├── scripts/
│   ├── install.sh        # Installation script
│   └── user_collaboration.sh
├── tests/
│   └── test_api.py       # API tests
├── docs/
│   ├── ARCHITECTURE.md
│   ├── IMPLEMENTATION_GUIDE.md
│   └── AMD-DEPLOYMENT-GUIDE.md
├── data/
│   ├── auth/             # API keys & sessions
│   └── gallery.json      # Image metadata
├── config.yaml           # Configuration
├── requirements.txt      # Python dependencies
├── alice.service        # systemd service
├── Makefile              # Build commands
└── README.md             # This file
```

---

## Requirements

### System Requirements

- Python 3.10+
- 16GB+ RAM (32GB recommended for SDXL)
- 50GB+ disk space for models
- GPU with 8GB+ VRAM (recommended)

### Python Dependencies

**Core versions match SAM's bundled Python environment for compatibility.**

See [requirements.txt](requirements.txt) for full list and [PYTORCH_INSTALL.md](PYTORCH_INSTALL.md) for PyTorch installation.

**Key packages:**
- PyTorch 2.6.0 (with ROCm 6.2, CUDA 12.4, or CPU backend)
- diffusers 0.35.2
- transformers 4.57.3
- accelerate 1.12.0
- compel 2.3.1 (prompt weighting)
- FastAPI 0.104.1
- Pydantic 2.12.5

---

## Web Interface

Access at `http://localhost:8080/web/`

**Dashboard:**
- Real-time service status
- GPU memory visualization
- Quick generation form
- Recent generations gallery

**Models:**
- List available models
- Model type detection (SD1.5, SDXL, etc.)
- Refresh model registry

**Download (Admin only):**
- Browse popular CivitAI Checkpoints on page load
- Search filters loaded models locally (instant)
- Download models from CivitAI or HuggingFace
- Real-time download progress
- Automatic model type detection

**Generate:**
- Full parameter controls
- Size presets
- Scheduler selection
- Generation log
- Image preview and download

---

## Development

```bash
# Install dependencies
make install

# Run in development mode (auto-reload)
make dev

# Run tests
make test

# Lint code
make lint

# Clean build artifacts
make clean
```

---

## Uninstallation

```bash
sudo ./scripts/install.sh --uninstall
```

---

## Troubleshooting

### Server won't start
- Check logs: `tail -f /var/log/alice/alice.log`
- Verify Python path: `which python3`
- Check port availability: `lsof -i:8080`

### Model not loading
- Verify model path exists
- Check model format (diffusers or safetensors)
- Ensure enough VRAM/RAM

### Generation fails
- Check GPU memory usage
- Try smaller resolution
- Reduce batch size

### AMD GPU issues
- See `docs/AMD-DEPLOYMENT-GUIDE.md`
- Verify ROCm installation
- Check `HSA_OVERRIDE_GFX_VERSION`

---

## License

GNU General Public License v3.0 (GPL-3.0)

---

