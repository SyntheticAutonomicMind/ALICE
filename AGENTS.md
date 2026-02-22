# AGENTS.md

**Version:** 2.0  
**Date:** 2026-02-22  
**Purpose:** Technical reference for ALICE development

---

## Project Overview

**ALICE** (Artificial Latent Image Composition Engine) is a remote Stable Diffusion service built for privacy, performance, and simplicity.

- **Language:** Python 3.10+
- **Architecture:** FastAPI web service with PyTorch/Diffusers backend
- **Philosophy:** Privacy-first, local-first AI image generation
- **Part of:** Synthetic Autonomic Mind ecosystem

**Key Technologies:**
- FastAPI 0.104+ for web server
- PyTorch 2.6.0 for model inference
- Diffusers (latest) for Stable Diffusion pipelines
- stable-diffusion.cpp (sd.cpp) for Vulkan-based inference
- Uvicorn for ASGI server
- Pydantic for data validation

**Supported Model Types:**
- **SD 1.5** - Stable Diffusion 1.5 (.safetensors)
- **SDXL** - Stable Diffusion XL (.safetensors or diffusers)
- **FLUX** - Black Forest Labs FLUX (.safetensors or diffusers)
- **Qwen Image Edit** - Qwen2.5-based image editing (.gguf, multi-component)

**Supported Generation Modes:**
- **Text-to-Image** - Generate images from text prompts
- **Image-to-Image** - Edit/transform existing images with text instructions

---

## Quick Setup

```bash
# Clone repository
git clone https://github.com/SyntheticAutonomicMind/ALICE.git
cd ALICE

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch (CRITICAL: platform-specific)
# For NVIDIA CUDA:
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# For AMD ROCm:
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/rocm6.2

# For macOS (Apple Silicon):
pip install torch==2.6.0 torchvision==0.21.0

# Install dependencies
pip install -r requirements.txt

# Create required directories
mkdir -p models images logs

# Run development server
python -m src.main

# Or use Makefile
make install
make run
```

---

## Architecture

```
FastAPI Application (src/main.py)
    |
    +-- Authentication Layer (src/auth.py)
    |   - API key management
    |   - Session handling
    |   - User roles (admin/user)
    |
    +-- Model Management
    |   +-- ModelRegistry (src/model_registry.py)
    |   |   - Model detection (SD 1.5, SDXL, FLUX)
    |   |   - Model listing and metadata
    |   |
    |   +-- ModelCache (src/model_cache.py)
    |   |   - Memory caching of loaded models
    |   |   - Auto-unload on timeout
    |   |
    |   +-- DownloadManager (src/downloader.py)
    |       - CivitAI integration
    |       - HuggingFace integration
    |       - Async download tasks
    |
    +-- Generation Pipeline
    |   +-- GeneratorService (src/generator.py)
    |   |   - Request coordination
    |   |   - Job queue management
    |   |   - Text-to-image and image-to-image routing
    |   |
    |   +-- JobQueue (src/job_queue.py)
    |   |   - Concurrent request management
    |   |   - Request timeout handling
    |   |
    |   +-- Backend System (src/backends/)
    |       +-- PyTorchBackend (pytorch_backend.py)
    |       |   - GPU-accelerated generation (ROCm/CUDA/MPS)
    |       |   - SD 1.5, SDXL, FLUX, Qwen support
    |       |   - Text-to-image and image-to-image
    |       |   - Memory optimization
    |       |
    |       +-- SDCppBackend (sdcpp_backend.py)
    |           - Vulkan-based generation
    |           - Universal GPU support
    |           - GGUF model format
    |           - Multi-component model configs (YAML)
    |           - Text-to-image and image-to-image
    |
    +-- Gallery Management (src/gallery.py)
    |   - Image metadata storage
    |   - Privacy controls (public/private)
    |   - Batch operations
    |
    +-- Web Interface (web/)
        - Dashboard (index.html)
        - Generation UI (generate.html)
        - Gallery (gallery.html)
        - Model management (download.html)
        - Admin panel (admin.html)
```

---

## Directory Structure

| Path | Purpose |
|------|---------|
| `src/` | Main Python application code |
| `src/backends/` | Generation backend implementations |
| `web/` | Static web UI files |
| `scripts/` | Installation and utility scripts |
| `docker/` | Docker configuration files |
| `docs/` | Technical documentation |
| `tests/` | Unit and integration tests |
| `models/` | Stable Diffusion models (gitignored) |
| `images/` | Generated images (gitignored) |
| `data/` | Database and metadata (gitignored) |
| `logs/` | Application logs (gitignored) |
| `venv/` | Python virtual environment (gitignored) |
| `.clio/` | CLIO agent configuration |
| `.github/workflows/` | GitHub Actions (CI/CD, release, triage) |
| `ai-assisted/` | Session handoff documents (gitignored) |
| `scratch/` | Working documents (gitignored) |

**Key Files:**

- `src/main.py` - FastAPI application entry point
- `src/__init__.py` - **Version single source of truth** (`__version__`)
- `src/updater.py` - Self-update system (GitHub Releases integration)
- `src/config_migration.py` - Config migration for cross-version upgrades
- `src/backends/pytorch_backend.py` - PyTorch generation backend (text2img + img2img)
- `src/backends/sdcpp_backend.py` - sd.cpp Vulkan backend (text2img + img2img, GGUF models)
- `src/model_registry.py` - Model scanning (.safetensors, .gguf, diffusers directories)
- `src/downloader.py` - Model download manager
- `src/model_cache.py` - Model caching system
- `src/auth.py` - Authentication and authorization
- `src/schemas.py` - Pydantic schemas (includes img2img fields)
- `config.yaml` - Configuration file
- `Makefile` - Build, run, update, Docker, and dist commands
- `release.sh` - Release automation script
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container build (CPU/CUDA)
- `Dockerfile.rocm` - Container build (AMD ROCm)
- `docker-compose.yml` - Multi-profile Docker Compose

**Model Configuration Files (YAML):**

Multi-component GGUF models use sidecar YAML configs:
- `models/mymodel.gguf` + `models/mymodel.yaml` - model + config
- Config specifies auxiliary files (VAE, LLM encoder) and sd-cli flags
- See `docs/BACKEND-ARCHITECTURE.md` for full schema

---

## Code Style

**Python Conventions:**

- Python 3.10+ with type hints
- **PEP 8** formatting with 120-character line limit
- **4 spaces** indentation (never tabs)
- **SPDX license headers** on all files
- **Docstrings** for all classes and public methods

**Module Template:**

```python
# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2025 Andrew Wyatt (Fewtarius)

"""
Module description.

Detailed explanation of module purpose and behavior.
"""

import asyncio
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class MyClass:
    """Brief class description.
    
    Longer explanation of what this class does and why.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize MyClass.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    async def process(self) -> None:
        """Process the thing.
        
        Raises:
            ValueError: If processing fails
        """
        pass
```

**Logging:**

```python
import logging
logger = logging.getLogger(__name__)

# Use appropriate levels
logger.debug("Detailed debugging information")
logger.info("General informational messages")
logger.warning("Warning messages")
logger.error("Error messages")
logger.exception("Errors with stack traces")
```

**Async Patterns:**

```python
# Prefer async/await
async def fetch_data() -> Dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

# Use asyncio.gather for parallel operations
results = await asyncio.gather(
    fetch_model_info(),
    fetch_download_status(),
    return_exceptions=True
)
```

---

## Module Naming Conventions

| Module | Purpose |
|--------|---------|
| `src/main.py` | FastAPI application and endpoints |
| `src/__init__.py` | Package init, `__version__` (single source of truth) |
| `src/config.py` | Configuration loading and validation |
| `src/config_migration.py` | Config migration for cross-version upgrades |
| `src/schemas.py` | Pydantic schemas for API requests/responses |
| `src/auth.py` | Authentication and authorization |
| `src/updater.py` | Self-update system (GitHub Releases integration) |
| `src/model_registry.py` | Model detection and listing |
| `src/model_cache.py` | In-memory model caching |
| `src/generator.py` | Generation service coordination |
| `src/job_queue.py` | Request queue management |
| `src/downloader.py` | Model download management |
| `src/gallery.py` | Image metadata and privacy |
| `src/cancellation.py` | Generation cancellation registry |
| `src/backends/__init__.py` | Backend factory and auto-detection |
| `src/backends/base.py` | Abstract backend interface |
| `src/backends/pytorch_backend.py` | PyTorch/Diffusers backend |
| `src/backends/sdcpp_backend.py` | sd.cpp Vulkan backend |

---

## Testing

**Before Committing:**

```bash
# 1. Run pytest
pytest tests/ -v

# 2. Run specific test file
pytest tests/test_config.py -v

# 3. Run linter (if configured)
flake8 src/ --max-line-length=120

# 4. Type checking (if using mypy)
mypy src/

# 5. Manual integration test
python -m src.main
# Visit http://localhost:8080/web/
# Test generation workflow
```

**Test Locations:**

- `tests/test_config.py` - Configuration loading tests
- `tests/test_api.py` - API endpoint tests
- `tests/test_generator.py` - Generation service tests

**Test Pattern:**

```python
# tests/test_myfeature.py
import pytest
from src.mymodule import MyClass


def test_myclass_init():
    """Test MyClass initialization."""
    obj = MyClass()
    assert obj is not None


@pytest.mark.asyncio
async def test_async_operation():
    """Test async operations."""
    result = await some_async_function()
    assert result == expected
```

---

## Commit Format

```
type(scope): brief description

Problem: What was broken/missing/needed
Solution: How you fixed/added it
Testing: How you verified it works
```

**Types:** `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`

**Scopes:** `backend`, `api`, `auth`, `gallery`, `models`, `web`, `config`, `docs`

**Example:**

```bash
git add -A
git commit -m "fix(backend): prevent GPU hang on AMD gfx1103

Problem: VAE decode on GPU causes system hang on AMD Phoenix APU
Solution: Added vae_decode_cpu option and MIOPEN_DEBUG_FIND_ALL=0 env var
Testing: Generated 1296x800 SDXL images successfully on gfx1103"
```

**Pre-Commit Checklist:**

- ✅ Tests pass: `pytest tests/ -v`
- ✅ Code formatted (PEP 8)
- ✅ SPDX headers on new files
- ✅ Docstrings for new functions/classes
- ✅ No secrets or API keys in code
- ✅ Config changes documented
- ✅ No `ai-assisted/` or `scratch/` files staged

---

## Development Tools

**Common Commands:**

```bash
# Development server with auto-reload
make dev
# Or:
uvicorn src.main:app --host 0.0.0.0 --port 8080 --reload

# Run with specific config
ALICE_CONFIG=/path/to/config.yaml python -m src.main

# Install dependencies
make install
# Or:
pip install -r requirements.txt

# Clean build artifacts
make clean

# View logs (if running as service)
journalctl --user -u alice -f          # SteamOS/user service
sudo journalctl -u alice -f            # System service
tail -f logs/alice.log                 # Development
```

**Useful Scripts:**

```bash
# Install on SteamOS/Steam Deck (user service)
./scripts/install_steamos.sh

# Install system-wide (requires sudo)
sudo ./scripts/install.sh

# Detect AMD GPU capabilities
./scripts/detect_amd_gpu.sh

# Test aspect ratios
./scripts/test_aspect_ratios.sh

# Quick API test
./scripts/quick_test.sh
```

**Investigation Commands:**

```bash
# Find all generation endpoints
grep -r "def.*generate" src/

# Find model type detection logic
grep -r "detect.*model" src/

# Find authentication decorators
grep -r "@.*auth" src/

# List all FastAPI endpoints
python -c "from src.main import app; print([r.path for r in app.routes])"

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Check ROCm devices (AMD)
rocm-smi
```

---

## Common Patterns

**Error Handling:**

```python
# FastAPI endpoint error handling
from fastapi import HTTPException
from src.schemas import ErrorResponse

@app.get("/api/models")
async def list_models():
    try:
        models = await model_registry.list_models()
        return {"models": models}
    except Exception as e:
        logger.exception("Failed to list models")
        raise HTTPException(status_code=500, detail=str(e))
```

**Authentication:**

```python
from fastapi import Depends, HTTPException
from src.auth import get_current_user, require_admin

# Require authenticated user
@app.get("/api/protected")
async def protected_endpoint(user = Depends(get_current_user)):
    return {"user": user["username"]}

# Require admin role
@app.post("/api/admin/users")
async def admin_endpoint(user = Depends(require_admin)):
    return {"status": "ok"}
```

**Async File I/O:**

```python
import aiofiles
from pathlib import Path

async def save_image(path: Path, data: bytes) -> None:
    """Save image data asynchronously."""
    async with aiofiles.open(path, 'wb') as f:
        await f.write(data)

async def read_config(path: Path) -> str:
    """Read configuration file asynchronously."""
    async with aiofiles.open(path, 'r') as f:
        return await f.read()
```

**Model Loading:**

```python
# Backend pattern for model loading
async def load_model(self, model_path: Path) -> None:
    """Load model into memory.
    
    Args:
        model_path: Path to model file or directory
    
    Raises:
        RuntimeError: If model loading fails
    """
    try:
        logger.info(f"Loading model: {model_path}")
        self.pipeline = await asyncio.to_thread(
            self._load_pipeline_sync, model_path
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.exception("Model loading failed")
        raise RuntimeError(f"Failed to load model: {e}")
```

---

## Documentation

### What Needs Documentation

| Change Type | Required Documentation |
|-------------|------------------------|
| New API endpoint | Docstring + update docs/API-USAGE.md |
| Configuration option | Update config.yaml comments + docs/ |
| Backend change | Update docs/BACKEND-ARCHITECTURE.md |
| Platform-specific fix | Update docs/AMD-*.md or create new |
| Architecture change | Update docs/ARCHITECTURE.md |
| Installation change | Update README.md + scripts/ |

### Documentation Files

| File | Purpose | Audience |
|------|---------|----------|
| `README.md` | Project overview, quick start | Everyone |
| `docs/ARCHITECTURE.md` | System architecture | Developers |
| `docs/BACKEND-ARCHITECTURE.md` | Backend design | Developers |
| `docs/API-USAGE.md` | API reference | Integrators |
| `docs/IMPLEMENTATION_GUIDE.md` | Implementation details | Developers |
| `docs/AMD-DEPLOYMENT-GUIDE.md` | AMD GPU setup | Users/Admins |
| `docs/AMD-PHOENIX-ENVIRONMENT.md` | AMD Phoenix APU | Users/Admins |
| `docs/TTM-TUNING-GUIDE.md` | Performance tuning | Admins |
| `.clio/instructions.md` | Project methodology | AI agents |
| `AGENTS.md` | Technical reference | AI agents |

### Working Documents (scratch/)

**Purpose:** Gitignored workspace for investigation and planning.

**Use scratch/ for:**
- Code analysis documents
- Refactoring plans
- Investigation summaries
- Performance benchmarks
- Testing notes

**NEVER create these in project root** - they clutter the repository.

---

## Anti-Patterns (What NOT To Do)

| Anti-Pattern | Why It's Wrong | What To Do |
|--------------|----------------|------------|
| Synchronous I/O in async functions | Blocks event loop | Use `aiofiles` or `asyncio.to_thread()` |
| Bare `except:` clauses | Hides errors | Catch specific exceptions |
| Hardcoded paths | Breaks deployments | Use config values |
| Loading models synchronously | Blocks server | Use async with `to_thread()` |
| No type hints | Reduces code clarity | Add type hints to all functions |
| Missing SPDX headers | Licensing ambiguity | Add to all new files |
| Committing `.pyc` files | Git pollution | Already in `.gitignore` |
| Committing secrets | Security risk | Use config or environment variables |
| Skipping pytest before commit | Breaks main | Run `pytest tests/ -v` |

**Platform-Specific Anti-Patterns:**

| Anti-Pattern | Platform | Why Wrong | Solution |
|--------------|----------|-----------|----------|
| VAE decode on GPU | AMD gfx1103 | GPU hang | Set `vae_decode_cpu: true` |
| MIOpen solver search | AMD Phoenix | GPU hang | Set `MIOPEN_DEBUG_FIND_ALL=0` |
| float16 everywhere | AMD gfx1103 | Crashes | Use `force_float32: true` or `force_bfloat16: true` |
| Missing `device_map` | SDXL on AMD | OOM | Set `device_map: null` for directory models |
| Too many concurrent | Limited VRAM | OOM | Set `max_concurrent: 1` |

---

## Quick Reference

**Start Development:**
```bash
source venv/bin/activate
python -m src.main
# Visit http://localhost:8080/web/
```

**Run Tests:**
```bash
pytest tests/ -v
```

**Deploy to SteamOS:**
```bash
./scripts/install_steamos.sh
systemctl --user status alice
```

**Check Logs:**
```bash
# Development
tail -f logs/alice.log

# Production (SteamOS)
journalctl --user -u alice -f

# Production (system)
sudo journalctl -u alice -f
```

**Update Dependencies:**
```bash
pip install -r requirements.txt --upgrade
```

**Generate Test Image:**
```bash
# Using API
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"A beautiful sunset"}]}'
```

**Key Configuration Locations:**

- Development: `config.yaml`
- SteamOS: `~/.config/alice/config.yaml`
- System: `/etc/alice/config.yaml`

**Key Data Locations:**

- Development: `./models`, `./images`, `./data`, `./logs`
- SteamOS: `~/.local/share/alice/`
- System: `/var/lib/alice/`, `/var/log/alice/`

---

## CRITICAL: Platform-Specific Requirements

**AMD Phoenix APU (gfx1103, gfx1102):**

MUST set in config.yaml:
```yaml
generation:
  force_bfloat16: true  # For gfx1102 (best)
  # OR
  force_float32: true   # For gfx1103 (fallback)
  vae_decode_cpu: true  # REQUIRED to prevent GPU hang
```

MUST set in environment (or systemd service):
```bash
export MIOPEN_DEBUG_FIND_ALL=0
export PYTORCH_ROCM_ARCH=gfx1103  # or gfx1102
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
```

**NVIDIA GPUs:**

Usually works with defaults. For optimal performance:
```yaml
generation:
  enable_torch_compile: true
  torch_compile_mode: reduce-overhead
```

**Apple Silicon (M1/M2/M3):**

```yaml
generation:
  backend: pytorch
  # MPS backend automatically detected
```

---

## Versioning

**All Synthetic Autonomic Mind software uses date-based versioning:**

```
Format: YYYYMMDD.RELEASE
Example: 20260222.1  (first release on Feb 22, 2026)
         20260222.2  (second release on same day)
         20260223.1  (first release next day)
```

**Single source of truth:** `src/__init__.py` (`__version__`)

All other files import from `__init__.py`:
- `src/main.py` - FastAPI app version
- `src/schemas.py` - HealthResponse default
- `src/updater.py` - Version comparison

**NEVER hardcode version strings.** Always use `from . import __version__`.

---

## Release Process

**Use the release script:**

```bash
# Syntax: ./release.sh YYYYMMDD.RELEASE
./release.sh 20260223.1
```

**What the script does:**
1. Validates version format
2. Updates `src/__init__.py`
3. Commits: `chore(release): prepare version YYYYMMDD.R`
4. Creates annotated tag: `vYYYYMMDD.R`
5. Pushes commit and tag to origin

**What GitHub Actions does (triggered by tag push):**
1. Verifies version in code matches tag
2. Builds distribution tarball (`alice-VERSION.tar.gz`)
3. Creates GitHub Release with assets
4. Builds 3 Docker images and pushes to GHCR:
   - `ghcr.io/syntheticautonomicmind/alice:VERSION` (CPU)
   - `ghcr.io/syntheticautonomicmind/alice:VERSION-cuda` (NVIDIA)
   - `ghcr.io/syntheticautonomicmind/alice:VERSION-rocm` (AMD)

**Manual release (without script):**

```bash
# 1. Update version
sed -i 's/__version__ = ".*"/__version__ = "20260223.1"/' src/__init__.py

# 2. Commit
git add src/__init__.py
git commit -m "chore(release): prepare version 20260223.1"

# 3. Tag and push
git tag -a v20260223.1 -m "ALICE 20260223.1"
git push origin main
git push origin v20260223.1
```

---

## Deployment Methods

### 1. Self-Update (Recommended for Existing Installations)

ALICE includes a built-in update system that checks GitHub Releases:

**API Endpoints (admin-only):**

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/v1/update/status` | Current update status and version info |
| `POST` | `/v1/update/check` | Check GitHub for available updates |
| `POST` | `/v1/update/apply?restart=true` | Download and apply update |
| `POST` | `/v1/update/rollback` | Rollback to previous backup |
| `GET` | `/v1/update/backups` | List available backups |

**How it works:**
1. Checks `https://api.github.com/repos/SyntheticAutonomicMind/ALICE/releases/latest`
2. Compares version tuples (date-based versions sort naturally)
3. Downloads named asset (`alice-VERSION.tar.gz`) or source tarball
4. Creates backup of current installation in `.alice-backups/backups/`
5. Extracts to staging directory, then copies files (skips user data)
6. Runs `pip install -r requirements.txt`
7. Migrates config (adds new fields with defaults)
8. Restarts service (systemd or SIGTERM)

**Protected paths (never overwritten):** `config.yaml`, `models/`, `images/`, `data/`, `logs/`, `venv/`

**Automatic checking:** Every 60 minutes (configurable, starts 60s after boot)

**Web UI:** Admin panel shows update banner with progress bar and one-click update

### 2. Docker (Recommended for New Deployments)

```bash
# CPU
docker compose --profile default up -d

# NVIDIA GPU
docker compose --profile cuda up -d

# AMD GPU
docker compose --profile rocm up -d
```

**Volume mounts:**
- `./models:/data/models` - Model files
- `alice-images:/data/images` - Generated images
- `alice-data:/data/data` - Database and metadata
- `./config.yaml:/config/config.yaml:ro` - Config override

**Update Docker:**
```bash
docker compose pull && docker compose up -d
```

### 3. Git-Based (Development)

```bash
make update           # Pull + reinstall deps
make update-restart   # Pull + reinstall + restart service
```

### 4. Script-Based (Traditional)

```bash
sudo ./scripts/install.sh          # System-wide install
./scripts/install_steamos.sh       # SteamOS/Steam Deck
```

---

## Health & Monitoring Endpoints

| Endpoint | Purpose | Auth Required |
|----------|---------|---------------|
| `GET /health` | Full health (version, GPU, uptime, backend) | No |
| `GET /api/health` | Alias for `/health` | No |
| `GET /livez` | Liveness probe (containers) | No |
| `GET /readyz` | Readiness probe (503 if starting) | No |
| `GET /metrics` | Service metrics | No |

**Health response example:**
```json
{
    "status": "ok",
    "gpuAvailable": true,
    "gpuStatsAvailable": true,
    "modelsLoaded": 1,
    "version": "20260222.1",
    "uptimeSeconds": 3600.5,
    "backend": "vulkan"
}
```

---

## Config Migration

When new config options are added between versions, `src/config_migration.py`:
1. Compares user's `config.yaml` against current defaults
2. Identifies missing keys
3. Creates a timestamped backup (`.bak`)
4. Adds missing keys with default values

**IMPORTANT:** When adding new config fields:
1. Add the field to the Pydantic model in `src/config.py`
2. Add the same default to `get_default_config()` in `src/config_migration.py`
3. These MUST stay in sync

**Runs automatically:** On startup and after self-updates.

---

**For project methodology and workflow, see .clio/instructions.md**
