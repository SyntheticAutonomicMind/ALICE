# AGENTS.md

**Version:** 1.0  
**Date:** 2026-02-07  
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
- Uvicorn for ASGI server
- Pydantic for data validation

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
    |   |
    |   +-- JobQueue (src/job_queue.py)
    |   |   - Concurrent request management
    |   |   - Request timeout handling
    |   |
    |   +-- Backend System (src/backends/)
    |       +-- PyTorchBackend (pytorch_backend.py)
    |       |   - GPU-accelerated generation
    |       |   - SD 1.5, SDXL, FLUX support
    |       |   - Memory optimization
    |       |
    |       +-- SDCppBackend (sdcpp_backend.py)
    |           - Vulkan-based generation
    |           - Universal GPU support
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
| `docs/` | Technical documentation |
| `tests/` | Unit and integration tests |
| `models/` | Stable Diffusion models (gitignored) |
| `images/` | Generated images (gitignored) |
| `data/` | Database and metadata (gitignored) |
| `logs/` | Application logs (gitignored) |
| `venv/` | Python virtual environment (gitignored) |
| `.clio/` | CLIO agent configuration |
| `ai-assisted/` | Session handoff documents (gitignored) |
| `scratch/` | Working documents (gitignored) |

**Key Files:**

- `src/main.py` - FastAPI application entry point (104,458 bytes)
- `src/backends/pytorch_backend.py` - PyTorch generation backend (61,215 bytes)
- `src/downloader.py` - Model download manager (38,232 bytes)
- `src/model_cache.py` - Model caching system (35,098 bytes)
- `src/auth.py` - Authentication and authorization (25,348 bytes)
- `config.yaml` - Configuration file
- `Makefile` - Build and run commands
- `requirements.txt` - Python dependencies

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
| `src/config.py` | Configuration loading and validation |
| `src/schemas.py` | Pydantic schemas for API requests/responses |
| `src/auth.py` | Authentication and authorization |
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

**For project methodology and workflow, see .clio/instructions.md**
