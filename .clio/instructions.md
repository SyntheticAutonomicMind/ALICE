# CLIO Project Instructions - ALICE

**Project:** ALICE (Artificial Latent Image Composition Engine)  
**Language:** Python 3.10+  
**Framework:** FastAPI  
**Purpose:** OpenAI-compatible remote Stable Diffusion image generation service

---

## CRITICAL: READ FIRST BEFORE ANY WORK

### The Unbroken Method (Core Principles)

This project follows **The Unbroken Method** for human-AI collaboration. This isn't just project style—it's the core operational framework.

**The Seven Pillars:**

1. **Continuous Context** - Never break the conversation. Maintain momentum through collaboration checkpoints.
2. **Complete Ownership** - If you find a bug, fix it. No "out of scope."
3. **Investigation First** - Read code before changing it. Never assume.
4. **Root Cause Focus** - Fix problems, not symptoms.
5. **Complete Deliverables** - No partial solutions. Finish what you start.
6. **Structured Handoffs** - Document everything for the next session.
7. **Learning from Failure** - Document mistakes to prevent repeats.

**If you skip this, you will violate the project's core methodology.**

### Collaboration Checkpoint Discipline

**Use collaboration tool at EVERY key decision point:**

| Checkpoint | When | Purpose |
|-----------|------|---------|
| Session Start | Always | Confirm context & plan |
| After Investigation | Before implementation | Share findings, get approval |
| After Implementation | Before commit | Show results, get OK |
| Session End | When work complete | Summary & handoff |

**[FAIL]** Create code/implementations alone  
**[OK]** Investigate freely, but checkpoint before committing changes

---

## Quick Start for NEW DEVELOPERS

### Before Touching Code

1. **Understand the system:**
   ```bash
   cat README.md                              # Service overview
   cat docs/ARCHITECTURE.md                   # System design (if exists)
   cat .github/copilot-instructions.md       # Additional guidelines
   ```

2. **Know the standards:**
   - All Python files: Use `# SPDX-License-Identifier: GPL-3.0-only`
   - Add `# SPDX-FileCopyrightText: Copyright (c) 2025 Andrew Wyatt (Fewtarius)`
   - Use type hints on all function signatures
   - Use logging module, never bare `print()`
   - Follow PEP 8 style guide (4-space indentation)
   - All modules have docstrings explaining purpose
   - Config values always read from `config.yaml`, never hardcoded

3. **Use the toolchain:**
   ```bash
   source venv/bin/activate                  # Activate virtual environment
   python -m src.main                        # Start dev server
   ./quick_test.sh                           # Quick functionality test
   git status                                # Always check before work
   ```

### Core Workflow

```
1. Read code first (investigation)
2. Use collaboration tool (get approval)
3. Make changes (implementation)
4. Test thoroughly (verify locally)
5. Commit with clear message (handoff)
```

---

## Key Directories & Files

### Core Application Files
| File | Purpose | Status |
|------|---------|--------|
| `src/main.py` | FastAPI app setup & endpoints | **LARGE (98 KB)** |
| `src/config.py` | Configuration management | Complete |
| `src/generator.py` | Stable Diffusion generation | **LARGE (50 KB)** |
| `src/model_registry.py` | Model discovery & management | Complete |
| `src/model_cache.py` | Model loading & caching | **LARGE (35 KB)** |
| `src/downloader.py` | CivitAI & HuggingFace downloads | **LARGE (38 KB)** |
| `src/auth.py` | Authentication & sessions | **LARGE (25 KB)** |
| `src/gallery.py` | Image storage & privacy | Complete |
| `src/schemas.py` | Pydantic request/response models | Complete |

### Directories
| Path | Purpose | Status |
|------|---------|--------|
| `src/` | Main application code | [ACTIVE] |
| `web/` | Frontend UI (web interface) | [ACTIVE] |
| `scripts/` | Deployment & testing utilities | [ACTIVE] |
| `docs/` | Documentation | [ACTIVE] |
| `tests/` | Unit & integration tests | [NEEDS EXPANSION] |
| `models/` | Local model storage (runtime) | [RUNTIME ONLY] |
| `images/` | Generated image storage (runtime) | [RUNTIME ONLY] |
| `logs/` | Application logs (runtime) | [RUNTIME ONLY] |

---

## Architecture Overview

```
User/Client
    v
HTTP POST /v1/chat/completions
    v
FastAPI Endpoint (main.py)
    v
Authentication (auth.py)
    v
Request Validation (schemas.py)
    v
GeneratorService (generator.py)
    ├─ ModelRegistry (model_registry.py) - Scan for models
    ├─ ModelCache (model_cache.py) - Load/cache models
    └─ diffusers pipeline - Execute generation
    v
Result Processing
    v
Image Storage (gallery.py)
    v
OpenAI-compatible JSON response
    v
Client App (SAM, etc.)
```

### Key Dependencies
- **FastAPI** - Web framework
- **PyTorch** - Deep learning (GPU acceleration)
- **diffusers** - Stable Diffusion pipelines
- **Pydantic** - Data validation
- **YAML** - Configuration files

### Hardware Support
- **NVIDIA (CUDA)** - Recommended, default setup
- **AMD (ROCm)** - See `docs/AMD-DEPLOYMENT-GUIDE.md` (gfx1103 Phoenix APU needs special config)
- **Apple Silicon (MPS)** - Full support
- **CPU** - Fallback (slow but works)

---

## Code Standards: MANDATORY

### Every Python File Must Have
```python
# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2025 Andrew Wyatt (Fewtarius)

"""
Module Name

Brief description of module purpose and key classes/functions.
"""

import logging
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

# Implementation...
```

### Type Hints: REQUIRED
```python
# [CORRECT] Always use type hints:
def generate_image(
    prompt: str, 
    steps: int = 25,
    guidance_scale: float = 7.5
) -> Path:
    """Generate image and return path to saved file."""
    pass

# [FAIL] Never:
def generate_image(prompt, steps=25):  # Missing type hints
    pass
```

### Logging: CONSISTENT PATTERN
```python
# [CORRECT] Always use logger module:
import logging
logger = logging.getLogger(__name__)

logger.info("Processing request: %s", request_id)
logger.error("Generation failed: %s", str(error))
logger.debug("GPU memory: %d MB", memory_used)

# [FAIL] Never:
print("debug message")                    # Goes to stdout
print(f"GPU memory: {memory_used} MB")    # Not logged
```

### Configuration: NO HARDCODING
```python
# [CORRECT] Read from config:
from .config import config

default_steps = config.generation.default_steps
model_dir = config.models.directory

# [FAIL] Never:
DEFAULT_STEPS = 25                        # Hardcoded
MODEL_DIR = "./models"                    # Hardcoded
```

### API Responses: USE SCHEMAS
```python
# [CORRECT] Use Pydantic schemas:
from .schemas import ChatCompletionResponse

response = ChatCompletionResponse(
    id=f"chatcmpl-{uuid.uuid4()}",
    model="sd/stable-diffusion-v1-5",
    choices=[...]
)
return response.model_dump()

# [FAIL] Never:
return {                                  # Raw dict without validation
    "id": "...",
    "model": "..."
}
```

---

## Testing Requirements

### Before Committing Changes
```bash
# 1. Activate venv
source venv/bin/activate

# 2. Run linting/type checking (if set up)
# python -m pylint src/your_module.py
# python -m mypy src/your_module.py

# 3. Run unit tests (if applicable)
python -m pytest tests/unit/ -v

# 4. Manual integration test
./quick_test.sh

# 5. Start server and test endpoint
python -m src.main &
sleep 2
curl http://localhost:8080/health
# Stop server: pkill -f "python -m src.main"
```

### Test File Location
- `tests/unit/` - Single module/function tests
- `tests/integration/` - Cross-module tests
- `tests/fixtures/` - Test data & mocks

### New Feature = New Test
If you add an endpoint or change business logic:
1. Create test file: `tests/unit/test_your_feature.py`
2. Run it: `python -m pytest tests/unit/test_your_feature.py -v`
3. Include it in commit message

---

## Commit Workflow

### Commit Message Format
```bash
type(scope): brief description

Problem: What was broken/incomplete
Solution: How you fixed it
Testing: How you verified the fix
```

**Types:** feat, fix, refactor, docs, test, chore  
**Scope:** main, generator, config, auth, gallery, downloader, etc.

**Example:**
```bash
git add src/generator.py tests/unit/test_generator.py
git commit -m "fix(generator): handle OOM on large models

Problem: Generating SDXL caused out-of-memory crash without graceful error
Solution: Added memory check before loading, clear cache on OOM, return error response
Testing: Tested with XL model on 4GB GPU - returns proper error instead of crashing"
```

### Before Committing: Checklist
- [ ] Code follows PEP 8 (if using linter)
- [ ] All functions have type hints
- [ ] All new config values added to `config.py` schema
- [ ] No hardcoded values (use `config.yaml`)
- [ ] Logging used instead of `print()`
- [ ] No commented-out code
- [ ] POD/docstring updated if API changed
- [ ] Commit message explains WHAT and WHY
- [ ] Test coverage for new code
- [ ] Manual testing completed locally

---

## Anti-Patterns: NEVER DO THESE

| Anti-Pattern | Why | What To Do Instead |
|--------------|-----|-------------------|
| Hardcoded config values | Breaks deployment | Use `config.yaml` |
| `print()` for debug | Breaks log collection | Use `logger.debug()` |
| Label bugs as "out of scope" | Harms quality | Own the problem, fix it |
| Missing type hints | Breaks static analysis | Add type hints to all functions |
| TODO comments in final code | Technical debt | Finish the implementation |
| Assume code behavior without reading | Causes mistakes | Read the code, understand it |
| Commit without testing | Breaks builds | Test locally before committing |
| Bare exceptions | Loses error context | Catch specific exceptions, log details |
| No docstrings | Confuses future devs | Document all modules & classes |
| Giant functions (>100 lines) | Hard to maintain | Split into focused functions |

---

## Development Tools & Commands

### Starting the Server
```bash
# Development (auto-reload, verbose logging)
source venv/bin/activate
export PYTHONUNBUFFERED=1
python -m src.main

# Or use Makefile:
make dev
```

### Testing
```bash
# Quick functionality test (health check, list models, etc.)
./quick_test.sh

# Full test suite
python -m pytest tests/ -v

# Test specific file
python -m pytest tests/unit/test_auth.py -v

# With coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Common Tasks
```bash
# See what changed
git diff src/main.py

# Review before commit
git diff --cached

# Stage specific files
git add src/generator.py tests/unit/test_generator.py

# See git history
git log --oneline -10

# Search for symbol usage
git grep "generate_image" src/

# Check git status
git status
```

### Debugging GPU Issues
```bash
# Check PyTorch GPU availability
python -c "import torch; print('GPU:', torch.cuda.is_available())"

# Test AMD/ROCm (if applicable)
python -c "import torch; print('Device:', torch.cuda.get_device_name())"

# Monitor generation (test with actual model)
python -c "from src.generator import GeneratorService; g = GeneratorService(); print(g.generate_image(...))"
```

---

## Key ALICE-Specific Concepts

### API Endpoints Overview

**OpenAI-Compatible:**
- `POST /v1/chat/completions` - Generate image from text
- `GET /v1/models` - List available models

**Management:**
- `GET /health` - Health check
- `GET /metrics` - Performance metrics
- `POST /v1/models/refresh` - Rescan models directory

**Gallery (Privacy-Aware):**
- `GET /v1/gallery` - List user's images
- `PATCH /v1/gallery/{id}/privacy` - Update privacy
- `DELETE /v1/gallery/{id}` - Delete image

**Downloads:**
- `POST /v1/models/search/civitai` - Search CivitAI
- `POST /v1/models/download/civitai` - Download from CivitAI
- `POST /v1/models/search/huggingface` - Search HuggingFace

### Configuration: `config.yaml`

Key sections:
```yaml
server:
  host: 0.0.0.0
  port: 8080
  block_nsfw: true

models:
  directory: ./models
  auto_unload_timeout: 300

generation:
  default_steps: 25
  default_scheduler: dpm++_sde_karras
  max_concurrent: 1

storage:
  images_directory: ./images
```

**CRITICAL:** All configuration is loaded into Pydantic models in `src/config.py`. Never hardcode values.

### Model Management

Models are discovered by scanning `./models` directory:
- Single files: `stable-diffusion-v1-5.safetensors`
- Directories: `stable-diffusion-v1-5/` (must contain diffusers structure)
- Model IDs use `sd/` prefix: `sd/stable-diffusion-v1-5`

### Image Privacy System

- All images **private by default** (only owner sees them)
- Can be made public with optional expiration (1-168 hours)
- Expired public images auto-deleted hourly
- Admins can view all images

---

## Critical: AMD GPU Configuration (gfx1103 Phoenix APU)

**If working on AMD/ROCm support:**

### TheRock ROCm Required
Official PyTorch ROCm packages **DO NOT** include gfx1103 kernels and cause segfaults.

```bash
# Use TheRock nightly builds (has gfx1103 support):
pip install \
  --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ \
  --pre torch torchaudio torchvision
```

### Working Configuration for Phoenix
```yaml
generation:
  force_float32: true          # CRITICAL
  device_map: sequential       # CRITICAL for single-file models
  vae_slicing: true
  attention_slicing: auto
```

### Service Environment Variables (systemd)
```
PYTORCH_ROCM_ARCH=gfx1103
HSA_OVERRIDE_GFX_VERSION=11.0.0
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

**DO NOT** assume official PyTorch packages work on gfx1103 - they don't.

---

## Session Handoff Procedures (MANDATORY)

### CRITICAL: Session Handoff Directory Structure

When ending a session, **ALWAYS** create a properly structured handoff directory:

```
ai-assisted/YYYYMMDD/HHMM/
├── CONTINUATION_PROMPT.md  [MANDATORY] - Next session's complete context
├── AGENT_PLAN.md           [MANDATORY] - Remaining priorities & blockers
├── CHANGELOG.md            [OPTIONAL]  - User-facing changes (if applicable)
└── NOTES.md                [OPTIONAL]  - Additional technical notes
```

**Format Details:**
- `YYYYMMDD` = Date (e.g., `20250109`)
- `HHMM` = Time in UTC (e.g., `1430` for 2:30 PM)
- Example: `ai-assisted/20250109/1430/`

### NEVER COMMIT Handoff Files

**[CRITICAL] BEFORE EVERY COMMIT:**

```bash
# ALWAYS verify no handoff files are staged:
git status

# If any `ai-assisted/` files appear:
git reset HEAD ai-assisted/

# Then commit only actual code/docs:
git add -A
git status  # Double-check no ai-assisted/ in staged
git commit -m "type(scope): description"
```

**Why:** Handoff documentation contains internal session context, work notes, and continuation details that should NEVER be in the public repository. This is a HARD REQUIREMENT.

### CONTINUATION_PROMPT.md (MANDATORY)

**Purpose:** Provides complete standalone context for the next session.

**Structure:**
```markdown
# Session Continuation Context

## What Was Done
- [List completed work items]
- [What changed and why]

## Current Status
- [What works/what's broken]
- [Test results]

## Next Steps (Prioritized)
1. [Priority item]
2. [Priority item]

## Blockers/Concerns
- [If any issues found]
- [If any uncertainties remain]

## Testing Status
- [What was tested]
- [What still needs testing]
```

### AGENT_PLAN.md (MANDATORY)

**Purpose:** Machine-readable priority list for AI agent to execute.

```markdown
# Work Plan

## High Priority
- [ ] Task 1 - Description
- [ ] Task 2 - Description

## Medium Priority
- [ ] Task 3 - Description

## Known Issues
- Issue: Description
  Impact: What breaks if not fixed
  Approach: How to fix

## File Changes Summary
- `src/file1.py` - What changed
- `src/file2.py` - What changed
```

### Before Committing: Final Verification

```bash
# Check for uncommitted handoff files
git status | grep "ai-assisted"

# If found, unstage them
git reset HEAD ai-assisted/

# Verify they're untracked (not in commit)
git diff --cached | grep "ai-assisted"

# Commit only actual code
git add -A
git commit -m "type(scope): description"
```

---

## Helpful Resources

### Documentation
- `README.md` - Service overview & quick start
- `.github/copilot-instructions.md` - Extended guidelines
- `docs/AMD-DEPLOYMENT-GUIDE.md` - AMD/ROCm setup
- `PYTORCH_INSTALL.md` - PyTorch installation (if present)

### Scripts
- `scripts/install.sh` - System-wide installation
- `scripts/install_steamos.sh` - SteamOS/Steam Deck setup
- `scripts/detect_amd_gpu.sh` - Detect AMD GPU & ROCm
- `scripts/quick_test.sh` - Quick functionality test

### Related Projects
- CLIO (this project's foundation) - `../CLIO-dist/`
- ALICE GitHub - https://github.com/fewtarius/alice

---

## Common Issues & Solutions

### Issue: Model Not Found
**Symptom:** `/v1/models` returns empty list  
**Fix:**
1. Check `models/` directory exists
2. Verify model files are readable
3. Check logs for parse errors
4. Restart server to rescan

### Issue: Out of Memory on Generation
**Symptom:** CUDA OOM or segfault  
**Fix:**
1. Check `vae_slicing: true` in config
2. Reduce `max_concurrent` to 1
3. Try smaller model or resolution
4. For AMD: ensure using TheRock PyTorch

### Issue: Image Generation Hangs
**Symptom:** Request times out  
**Fix:**
1. Check GPU utilization (is it actually running?)
2. Verify model loaded successfully
3. Check logs for errors
4. Increase `request_timeout` in config

### Issue: Authentication/Session Issues
**Symptom:** 401 Unauthorized, session expired  
**Fix:**
1. Check `require_auth` setting in config
2. Verify API key is correct
3. Check `session_timeout_seconds` setting
4. Review auth.py logs for details

---

## Remember

Your value is in:
1. **TAKING ACTION** - Not describing possible actions
2. **INVESTIGATING THOROUGHLY** - Understanding the codebase before changes
3. **COMPLETING WORK** - Finishing what you start
4. **COLLABORATING** - Using checkpoints to keep human in the loop

**The user expects an agent that DOES things within The Unbroken Method, not a chatbot that talks about doing things.**
