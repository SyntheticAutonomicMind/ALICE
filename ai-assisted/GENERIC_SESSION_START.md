# Generic Session Start - ALICE Development

**Purpose:** This document provides complete context for starting a brand new ALICE development session when no prior session context is available.

**Date:** January 7, 2026  
**Status:** Template for new contributors or fresh starts

---

## 🎯 YOUR FIRST ACTION

**MANDATORY:** Your very first tool call MUST be:

```bash
scripts/user_collaboration.sh "Session started.

✅ Read THE_UNBROKEN_METHOD.md: yes
✅ Read copilot-instructions: yes
📋 Continuation context: Generic session start (no prior context)
🎯 User request: [Waiting for user to describe tasks]

I am ready to collaborate on session requirements and tasks.

What would you like to work on today? Press Enter:"
```

**WAIT for user response.** They will describe what they need you to work on.

---

## 📋 SESSION INITIALIZATION CHECKLIST

Before starting work, complete these steps:

### 1. Read The Unbroken Method
```bash
cat ai-assisted/THE_UNBROKEN_METHOD.md
```

This is the foundational methodology. The Seven Pillars govern all work:
1. **Continuous Context** - Never break the conversation
2. **Complete Ownership** - Fix every bug you find
3. **Investigation First** - Understand before changing
4. **Root Cause Focus** - Fix problems, not symptoms
5. **Complete Deliverables** - No partial solutions
6. **Structured Handoffs** - Perfect context transfer
7. **Learning from Failure** - Document anti-patterns

### 2. Read Copilot Instructions
```bash
cat .github/copilot-instructions.md
```

This contains ALICE-specific development practices, critical rules, and workflow.

### 3. Check Recent Context
```bash
# Recent commits
git log --oneline -10

# Current status
git status

# Look for recent session handoffs
ls -lt ai-assisted/ | head -20

# Check for uncommitted work
git diff
```

### 4. Use Collaboration Tool (see YOUR FIRST ACTION above)

Wait for user to provide tasks and priorities.

---

## 📚 PROJECT CONTEXT

### What is ALICE?

ALICE (Artificial Latent Image Composition Engine) is a FastAPI-based Stable Diffusion image generation server with:

- **OpenAI-Compatible API:** `/v1/chat/completions`, `/v1/models`, `/v1/images/generations`
- **Multi-Model Support:** SD 1.5, SDXL, Flux, SD3, Qwen-Image
- **Authentication:** Cookie-based with admin/user roles
- **Model Management:** CivitAI/HuggingFace downloads, catalog caching
- **Image Gallery:** Privacy controls, expiration handling
- **AMD GPU Optimized:** Specifically for gfx1103 Phoenix APU
- **Web Interface:** Native HTML/JS (no build system)

### Technology Stack

- **Backend:** FastAPI 0.128.0, Python 3.9+
- **ML:** diffusers (git), transformers 4.57.3, PyTorch (TheRock ROCm)
- **Data:** Pydantic 2.12.5, SQLite (model cache)
- **Deployment:** systemd service, Linux (SteamOS, Ubuntu)
- **GPU:** AMD ROCm 6.2, gfx1103 native support via TheRock

### Architecture

```
┌─────────────────────────────────────────────────┐
│             ALICE FastAPI Server                │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌──────────────┐         ┌──────────────┐     │
│  │  REST API    │◄───────►│   Web UI     │     │
│  │  (FastAPI)   │         │  (HTML/JS)   │     │
│  └──────┬───────┘         └──────────────┘     │
│         │                                       │
│  ┌──────▼───────┐         ┌──────────────┐     │
│  │  Generator   │◄───────►│   Model      │     │
│  │  (diffusers) │         │   Registry   │     │
│  └──────┬───────┘         └──────────────┘     │
│         │                                       │
│  ┌──────▼───────┐         ┌──────────────┐     │
│  │   Gallery    │         │ Model Cache  │     │
│  │  (Privacy)   │         │ (CivitAI)    │     │
│  └──────────────┘         └──────────────┘     │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Key Source Directories

```
src/
├── main.py              # FastAPI app, routes, endpoints
├── generator.py         # Image generation, model loading
├── model_registry.py    # Model detection and metadata
├── model_cache.py       # CivitAI/HF catalog caching
├── gallery.py           # Image gallery and privacy
├── auth.py              # Authentication and sessions
├── downloader.py        # Model downloading
├── config.py            # Configuration management
└── schemas.py           # Pydantic models

web/
├── generate.html        # Generation interface
├── models.html          # Model management
├── gallery.html         # Image gallery
├── download.html        # Model downloads
├── admin.html           # Admin panel
├── app.js               # JavaScript client
└── style.css            # Styles
```

---

## 🔧 ESSENTIAL COMMANDS

### Build & Test
```bash
# Start development server (local)
source venv/bin/activate
python -m src.main

# Test syntax (ALWAYS do this after editing)
python3 -m py_compile src/main.py
python3 -m py_compile src/generator.py

# Test imports
python3 -c "from src import main; print('Import OK')"
python3 -c "from src import generator; print('Import OK')"

# Test API
curl http://localhost:8080/health
curl http://localhost:8080/v1/models
```

### Code Search
```bash
# Search for code patterns
git grep "pattern" src/

# Search specific files
git grep "pattern" -- "*.py"

# Search with context
git grep -n -C 3 "pattern" src/
```

### Collaboration (MANDATORY)
```bash
# Use at key checkpoints
scripts/user_collaboration.sh "message"
```

### Git Workflow
```bash
# Commit with detailed message
git add -A && git commit -m "type(scope): description

PROBLEM:
[what was broken or missing]

SOLUTION:
[how you fixed or built it]

TESTING:
✅ Syntax: python3 -m py_compile
✅ Import: successful
✅ Manual: [what you tested]"
```

### Deployment to 2s (Test System)
```bash
# MANDATORY: Backup config first (Rule 0)
ssh deck@2s 'sudo cp /etc/alice/config.yaml /etc/alice/config.yaml.backup-$(date +%Y%m%d-%H%M%S)'

# Stop service
ssh deck@2s 'sudo systemctl stop alice.service'

# Copy files and restart
# (See full deployment procedure in CONTINUATION_PROMPT.md)
```

---

## 📂 FILE LOCATIONS

| Purpose | Path |
|---------|------|
| Source code | `src/` |
| Web interface | `web/` |
| Configuration (dev) | `config.yaml` or `~/.config/alice/config.yaml` |
| Configuration (prod) | `/etc/alice/config.yaml` |
| Models (dev) | `models/` or `~/.cache/alice/models/` |
| Models (prod) | `/var/lib/alice/models/` |
| Generated images (dev) | `images/` or `data/images/` |
| Generated images (prod) | `/var/lib/alice/images/` |
| Logs (prod) | `/var/log/ALICE/ALICE.log` |
| Documentation | `docs/` |
| Session handoffs | `ai-assisted/YYYY-MM-DD/` |

---

## 📖 KEY DOCUMENTATION

Read these as needed for your work:

### Methodology & Process (MUST READ)
- `ai-assisted/THE_UNBROKEN_METHOD.md` - **Core methodology**
- `.github/copilot-instructions.md` - **ALICE development practices**

### Architecture & API
- `docs/ARCHITECTURE.md` - System architecture
- `docs/API-USAGE.md` - Complete API reference
- `docs/IMPLEMENTATION_GUIDE.md` - Implementation details

### AMD GPU Deployment
- `docs/AMD-DEPLOYMENT-GUIDE.md` - Deployment procedures
- `docs/AMD-PHOENIX-ENVIRONMENT.md` - gfx1103 specifics
- `docs/THEROCK-GFX1103-INSTALL.md` - PyTorch installation

### Session Handoffs
- Look in `ai-assisted/YYYY-MM-DD/` for recent work context

---

## 🚨 CRITICAL RULES

### Process
1. **Always use collaboration tool** at session start, before implementation, after testing, at session end
2. **Read code before changing it** - Investigation first (Pillar 3)
3. **Fix all bugs you find** - Complete ownership (Pillar 2)
4. **No partial solutions** - Complete deliverables (Pillar 5)
5. **NEVER skip backups** - Rule 0: Always backup config files before changes

### Code
1. **Use logger, NEVER print()** - `import logging; logger = logging.getLogger(__name__)`
2. **Test syntax after EVERY edit** - `python3 -m py_compile <file>`
3. **No hardcoding** - Query metadata when available, use config.yaml
4. **Type hints required** - All function signatures need types

### Documentation
1. **Update docs when behavior changes** - Keep in sync with code
2. **Create handoffs in dated folders** - `ai-assisted/YYYY-MM-DD/`
3. **Include complete context** - Next session should continue seamlessly

### Deployment (Rule 0)
```bash
# MANDATORY before ANY config file changes
ssh deck@2s 'sudo cp /etc/alice/config.yaml /etc/alice/config.yaml.backup-$(date +%Y%m%d-%H%M%S)'
```

---

## ⚠️ ANTI-PATTERNS (DO NOT DO THESE)

❌ Skip session start collaboration checkpoint  
❌ Modify /etc/alice/config.yaml without backup (Rule 0 violation)  
❌ Use `rm -rf` without explicit user approval (Rule 1 violation)  
❌ Label bugs as "out of scope" (you own them)  
❌ Create partial implementations ("TODO for later")  
❌ Assume how code works (investigate first)  
❌ Use `print()` instead of logger  
❌ Commit without testing  
❌ End session without user approval  
❌ Push untested code to production  

---

## 🎯 WORKFLOW PATTERN

For each task you work on:

1. **INVESTIGATE**
   - Read existing code
   - Search for patterns: `git grep "pattern" src/`
   - Understand WHY it works this way

2. **CHECKPOINT** (collaboration tool)
   - Share findings
   - Propose approach
   - WAIT for approval

3. **IMPLEMENT**
   - Make exact changes from approved plan
   - Follow code standards
   - Test after EVERY edit

4. **TEST**
   - Syntax: `python3 -m py_compile <file>`
   - Import: `python3 -c "from src import ..."`
   - Functionality: Manual testing

5. **CHECKPOINT** (collaboration tool)
   - Show test results
   - WAIT for approval

6. **COMMIT**
   - Full commit message with testing details

7. **CONTINUE**
   - Move to next task
   - Keep working until ALL issues resolved

---

## 🤝 COLLABORATION IS MANDATORY

You are working **WITH** a human partner, not **FOR** a human.

- Use `scripts/user_collaboration.sh` at all key points
- WAIT for user response at each checkpoint
- User may approve, request changes, or reject
- This is a conversation, not a command stream

**The methodology works. Follow it exactly.**

---

## 💡 COMMON DEVELOPMENT TASKS

### Adding a New API Endpoint

1. Define Pydantic schema in `src/schemas.py`
2. Add route handler in `src/main.py`
3. Add authentication check if needed
4. Test with curl
5. Update `docs/API-USAGE.md`

### Modifying Image Generation

1. Changes typically go in `src/generator.py`
2. Test syntax after EVERY change
3. Test with actual generation
4. Update config.yaml comments if new settings
5. Update UI if user-facing

### Adding Model Type Support

1. Detection logic in `src/model_registry.py` (`_detect_model_type`)
2. Pipeline handling in `src/generator.py`
3. Resolution presets in `src/main.py` (endpoint `get_model_info`)
4. Test with actual model of that type
5. Document in `docs/`

### Fixing a Bug

1. Search for similar patterns: `git grep "pattern" src/`
2. Read the code to understand root cause (not just symptoms)
3. Fix the root cause
4. Check for similar bugs elsewhere (you own them all)
5. Test thoroughly
6. Update docs if behavior changed

### Content Filtering

1. Filter logic in `src/main.py` (`check_nsfw_content` function)
2. Configurable via `server.block_nsfw` in config.yaml
3. Test with known bypass attempts
4. Balance blocking vs false positives

---

## 🔍 DEBUGGING COMMON ISSUES

### Service Won't Start
```bash
# Check service status
ssh deck@2s 'sudo systemctl status alice.service'

# View recent logs
ssh deck@2s 'sudo journalctl -u alice.service -n 100'

# Check for Python errors
ssh deck@2s 'sudo journalctl -u alice.service | grep -i error'
```

### Import Errors
```bash
# Test import directly
python3 -c "from src import main"

# Check syntax
python3 -m py_compile src/main.py

# Check for circular imports
grep -r "from src" src/
```

### Generation Failures
```bash
# Monitor logs during generation
ssh deck@2s 'sudo journalctl -u alice.service -f'

# Check GPU memory
ssh deck@2s 'rocm-smi'

# Check disk space
ssh deck@2s 'df -h /var/lib/alice'
```

### Model Not Loading
```bash
# Check model detection
grep "Model type:" /var/log/ALICE/ALICE.log

# Check model path
ls -la /var/lib/alice/models/

# Check permissions
ssh deck@2s 'ls -la /var/lib/alice/models/'
```

---

## 🎯 DEPLOYMENT TARGETS

### Development (Mac)
- **Location**: `~/ALICE/` or local directory
- **Config**: Local config.yaml
- **Python**: venv with CPU PyTorch
- **Purpose**: Local testing

### 2s (Test System)
- **Hostname**: 2s
- **User**: deck
- **Install**: `/opt/alice`
- **Data**: `/var/lib/alice`
- **Config**: `/etc/alice/config.yaml`
- **GPU**: AMD Phoenix gfx1103
- **Purpose**: Integration testing

### Flip (Production)
- **Hostname**: flip
- **User**: deck
- **Install**: `/opt/alice`
- **Data**: `/var/lib/alice`
- **Config**: `/etc/alice/config.yaml`
- **GPU**: AMD Phoenix gfx1103
- **Purpose**: Production

**Flow**: Mac → 2s (test) → Flip (production)

---

## 📞 GETTING STARTED

**Your next step:**

1. Use the collaboration tool (see YOUR FIRST ACTION at top)
2. WAIT for user to describe tasks
3. Read relevant code for context
4. Discuss approach with user via collaboration tool
5. Begin work using the WORKFLOW PATTERN above

**Remember:**
- Investigation before implementation
- Checkpoint before committing
- Test before deploying
- Collaborate throughout

**The methodology works. Follow it exactly.**

Good luck! 🚀
