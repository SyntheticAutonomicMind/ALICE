<instructions>
# ALICE Development Instructions

You are working on ALICE - a FastAPI-based Stable Diffusion image generation server.

## SESSION INITIALIZATION CHECKLIST

DO NOT proceed with analysis until you:
1. [ ] Run: `scripts/user_collaboration.sh "Session started. I see the Unbroken Method instructions. Ready to begin work. Press Enter:"`
2. [ ] Wait for user confirmation
3. [ ] Check `git log --oneline -5` for recent context
4. [ ] Review any todo file for priorities
5. [ ] Use collaboration tool BEFORE investigating or implementing

## COLLABORATION CHECKPOINT DISCIPLINE

For EVERY todo item or task:
- BEFORE reading code: Use collaboration tool to propose approach
- AFTER investigation: Use collaboration tool to report findings
- BEFORE changes: Use collaboration tool to validate approach
- AFTER completion: Use collaboration tool to confirm before moving to next item

This is NOT optional. The tool is a core part of the workflow, not a nice-to-have.

---

## CRITICAL RULES

### Rule 0: ALWAYS BACKUP CONFIG FILES BEFORE CHANGES
**MANDATORY backup procedure for ANY config file modification:**

BEFORE modifying /etc/alice/config.yaml or any production config:
1. STOP immediately
2. Create timestamped backup:
   ```bash
   ssh deck@flip 'sudo cp /etc/alice/config.yaml /etc/alice/config.yaml.backup-$(date +%Y%m%d-%H%M%S)'
   ```
3. Verify backup exists
4. ONLY THEN proceed with changes

**This applies to:**
- Direct config file edits
- Copying config files from repo
- ANY operation that touches /etc/alice/config.yaml
- Deployment scripts that might overwrite configs

**If you forget to backup, you WILL be called out for violating the Unbroken Method.**

### Rule 1: NEVER DELETE WITHOUT EXPLICIT USER APPROVAL
**ABSOLUTE PROHIBITION on destructive operations without user confirmation:**

NEVER execute ANY of these commands without explicit user approval via collaboration tool:
- `rm -rf`
- `rm -r`  
- `rm` on directories
- `sudo rm` (any variant)
- `git reset --hard` 
- `git clean -fd`
- `truncate`
- `> file` (overwriting files)
- `dd` 
- `mkfs` / `mkswap`
- `fdisk` / `parted` / `gparted`
- DROP TABLE / DELETE FROM (SQL)
- Anything that modifies/deletes user data

**Before ANY deletion:**
1. STOP immediately
2. Use collaboration tool to ask: "I need to delete [specific files]. Confirm? [list exact paths]"
3. Wait for explicit "yes" / "proceed" / "confirmed"
4. ONLY THEN proceed with deletion

**Verification Required:**
- Check mount points: `mount | grep <path>`
- List contents FIRST: `ls -la <path>`
- Verify bind mounts won't cause unintended deletion
- Double-check paths are not user data

**If in doubt about ANY operation, ASK FIRST.**

### Rule 2: MANDATORY Collaboration Checkpoints
Use `scripts/user_collaboration.sh` for:
- Session start (confirm instructions understood)
- Major findings (after investigation)
- Before implementation (get approval)
- After making changes (validate the fix actually works)
- Session end (wait for user validation)
- BEFORE ANY DESTRUCTIVE OPERATION (see Rule 0)
- BEFORE AND AFTER config file changes

**Never declare a fix complete without user validation.**

NO EXCEPTIONS. If you cannot use the tool, escalate to user immediately.

### Rule 3: Own All Discovered Issues
If you find a bug/issue during work, YOU fix it. Never say:
- "This is a separate issue"
- "Out of scope"
- "Would you like me to fix this?"

Just fix it, then continue.

### Rule 4: Complete Solutions Only
Deliver full implementations. Never:
- "Basic version first"
- "TODO: handle edge cases"
- "Good enough for now"

### Rule 5: Investigation Before Implementation
Before changing code:
1. Read the existing implementation
2. Search for related patterns: `git grep "pattern" src/`
3. Understand WHY it works the way it does
4. Then propose and implement changes

### Rule 6: MANDATORY Collaboration Checkpoints

Use `scripts/user_collaboration.sh` at these points:
- **Session start**: Confirm instructions understood (already in SESSION INITIALIZATION)
- **After investigation**: Report findings on tasks before proceeding
- **Before implementation**: Get explicit user validation
- **Session end**: Always wait for user confirmation

This is NOT optional. Collaboration is a core part of the development workflow, not a convenience feature.

If you cannot use the tool, immediately escalate to user instead of proceeding solo.

---

## BUILD & TEST

Start server:
```bash
source venv/bin/activate
python -m src.main
```

Test endpoints:
```bash
curl http://localhost:8080/health
curl http://localhost:8080/v1/models
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "sd/test", "messages": [{"role": "user", "content": "a cat"}]}'
```

Check logs:
```bash
tail -f /var/log/ALICE/ALICE.log
```

---

## CODE STANDARDS

### Logging
```python
# Use logger, not print()
import logging
logger = logging.getLogger(__name__)
logger.info("Message")     # No emojis
logger.error("Error: %s", str(error))
```

### No Hardcoding
- Use config.yaml for settings
- Use environment variables for deployment values
- Query model metadata instead of hardcoding sizes

### Type Hints Required
```python
def generate_image(prompt: str, steps: int = 25) -> Path:
    pass
```

### Commits
```bash
git add -A && git commit -m "type(scope): description"
```
Types: feat, fix, refactor, docs, test, chore

---

## CRITICAL: AMD GPU CONFIGURATION (gfx1103 Phoenix APU)

**DO NOT CHANGE THESE SETTINGS WITHOUT TESTING ON ACTUAL HARDWARE**

### TheRock ROCm Installation (RECOMMENDED)

The official PyTorch ROCm packages do NOT include gfx1103 kernels and cause segfaults.
Use TheRock nightly builds which have proper gfx1103 support:

```bash
# Install PyTorch with gfx1103 support from TheRock
pip install \
  --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ \
  --pre torch torchaudio torchvision
```

See: https://github.com/ROCm/TheRock/blob/main/RELEASES.md

### Working Configuration (December 2025)
```yaml
generation:
  force_cpu: false
  force_float32: true
  device_map: sequential    # CRITICAL for single-file models
  vae_slicing: true
  attention_slicing: auto
```

### Service Environment Variables (systemd)
```
PYTORCH_ROCM_ARCH=gfx1103
HSA_OVERRIDE_GFX_VERSION=11.0.0
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

### Model Format Compatibility
| Format | device_map: sequential | device_map: null |
|--------|----------------------|------------------|
| Single file (.safetensors) | WORKS | SEGFAULT (with old PyTorch) |
| Diffusers directory | ERROR: "sequential not supported" | Requires TheRock PyTorch |

**CRITICAL FINDINGS:**
1. `device_map: sequential` is ONLY valid with `from_single_file()` loading
2. Official PyTorch ROCm packages cause SEGFAULT on gfx1103 - use TheRock
3. TheRock nightly builds have native gfx1103 kernel support
4. Single-file models work: ~4.7s/step on Steam Deck (512x512)

### What NOT to Do
- DO NOT use official pytorch.org ROCm packages for gfx1103
- DO NOT set `device_map: balanced` - not appropriate for single GPU  
- DO NOT assume CPU fallback works - test actual generation
- DO NOT make commits claiming "CPU mode required" without testing

---

## ALICE SPECIFICS

### Architecture
- Uses diffusers library directly (not subprocess)
- OpenAI-compatible API responses
- Model IDs use `sd/` prefix

### Model Directory
```
models/
├── stable-diffusion-v1-5/
├── stable-diffusion-xl-base/
└── loras/
```

### Response Format
```json
{
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "model": "sd/stable-diffusion-v1-5",
    "choices": [{
        "message": {
            "content": "Image generated successfully",
            "image_urls": ["http://server:8080/images/abc.png"]
        }
    }]
}
```

---

## SESSION WORKFLOW

### Starting Work
1. Execute session initialization (above)
2. Check what needs work
3. **REQUIRED**: Use collaboration tool to propose tasks
4. Read relevant code before changing it

### During Work - Mandatory Collaboration Loop
For each task:
1. Use collaboration tool to propose approach
2. Read code, understand pattern
3. Get validation via collaboration tool
4. Make changes
5. Test with curl commands
6. Fix errors before proceeding
7. Commit after completion
8. Use collaboration tool to confirm before next task

### Ending Session
1. Fix ALL discovered issues
2. Test all endpoints
3. **Update ALL affected documentation:**
   - `docs/` - If features or APIs changed
4. Commit all changes
5. Use collaboration tool for final validation:
```bash
scripts/user_collaboration.sh "Work complete. [list fixes]. Press Enter to confirm: "
```
6. Wait for user confirmation before stopping

---

## HANDOFF PROTOCOL

When session must end (user request or high token usage), create in `ai-assisted/YYYY-MM-DD/HHMM/`:

1. **CONTINUATION_PROMPT.md** - Complete context for next session
2. **AGENT_PLAN.md** - Remaining priorities and approach
3. **FEATURES.md** - User-facing feature descriptions
4. **CHANGELOG.md** - Changes in standard format

The continuation prompt must be standalone - no external references.

### MANDATORY Documentation Updates

Before handoff, update ALL affected documentation:

**Documentation (`docs/`):**
- `docs/ARCHITECTURE.md` - Architecture changes
- `docs/IMPLEMENTATION_GUIDE.md` - Implementation details
- `docs/AMD-DEPLOYMENT-GUIDE.md` - Deployment changes
- `README.md` - User-facing feature changes

**Documentation Checklist:**
- [ ] New features documented in README.md
- [ ] API changes documented in docs/
- [ ] Architecture changes documented in ARCHITECTURE.md
- [ ] Configuration changes documented in config.yaml comments
- [ ] New endpoints documented in API docs

---

## FILE LOCATIONS

| Purpose | Location |
|---------|----------|
| Source code | src/ |
| Web interface | web/ |
| Configuration | config.yaml, ~/.config/alice/config.yaml |
| Models | models/ (configurable) |
| LoRAs | models/loras/ |
| Generated images | data/images/ (configurable) |
| Auth data | data/auth/ |
| Logs | /var/log/ALICE/ALICE.log |
| Scripts | scripts/ |
| Tests | tests/ |
| Documentation | docs/, ai-assisted/ |

---

## QUICK REFERENCE

| Task | Command |
|------|---------|
| Start server | `python -m src.main` |
| Test health | `curl http://localhost:8080/health` |
| Collaborate | `scripts/user_collaboration.sh "message"` |
| Commit | `git add -A && git commit -m "type(scope): desc"` |

---

## ANTI-PATTERNS

Do not:
- Use `print()` - use logger
- Use emojis in logs or code
- End response without collaboration tool
- Label issues as "out of scope"
- Create partial implementations
- Assume how code works without reading it
- Hardcode when configuration available
- Skip collaboration checkpoints - the tool is mandatory
- Ask user for permission - instructions already require collaboration tool
- Proceed without waiting for user response to collaboration messages
- Declare "fixed" without testing

---

## METHODOLOGY REFERENCE

For detailed methodology principles, see: `ai-assisted/THE_UNBROKEN_METHOD.md`

The seven pillars:
1. Continuous Context - maintain conversation continuity
2. Complete Ownership - fix what you find
3. Investigation First - understand before changing
4. Root Cause Focus - fix problems, not symptoms
5. Complete Deliverables - finish what you start
6. Structured Handoffs - perfect context transfer
7. Learning from Failure - document anti-patterns
</instructions>
