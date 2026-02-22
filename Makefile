# ALICE Makefile
# Build, run, test, and deployment commands for ALICE service

.PHONY: install run test clean lint dev setup-dirs update check-update docker docker-cuda docker-rocm

VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
UVICORN = $(VENV)/bin/uvicorn
VERSION = $(shell $(PYTHON) -c "from src import __version__; print(__version__)" 2>/dev/null || echo "unknown")

# Default target
all: install

# Create virtual environment
$(VENV)/bin/activate:
	python3 -m venv $(VENV)

# Install dependencies
install: $(VENV)/bin/activate
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Create required directories
setup-dirs:
	mkdir -p models images logs data

# Run development server
run: setup-dirs
	$(PYTHON) -m src.main

# Run with uvicorn (alternative)
run-uvicorn: setup-dirs
	$(UVICORN) src.main:app --host 0.0.0.0 --port 8080 --reload

# Run development server with auto-reload
dev: setup-dirs
	$(UVICORN) src.main:app --host 0.0.0.0 --port 8080 --reload

# Run tests
test:
	$(PYTHON) -m pytest tests/ -v

# Run linter
lint:
	$(PYTHON) -m flake8 src/ --max-line-length=120
	$(PYTHON) -m mypy src/

# ==========================================
# Update Management
# ==========================================

# Check for available updates
check-update:
	@echo "Checking for ALICE updates..."
	@$(PYTHON) -c "\
	import asyncio; \
	from src.updater import UpdateManager; \
	async def check(): \
	    m = UpdateManager(auto_check=False); \
	    s = await m.check_for_update(); \
	    if s.update_available: \
	        print(f'Update available: {s.current_version} -> {s.latest_version}'); \
	        print(f'Release: {s.release_info.html_url}'); \
	    else: \
	        print(f'Already up to date ({s.current_version})'); \
	asyncio.run(check())"

# Update ALICE from git (for git-based installations)
update:
	@echo "Updating ALICE..."
	@echo "Current version: $(VERSION)"
	git fetch origin
	git pull origin main
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@NEW_VERSION=$$($(PYTHON) -c "from src import __version__; print(__version__)" 2>/dev/null); \
	echo "Updated to: $$NEW_VERSION"
	@echo ""
	@echo "Restart the service to apply changes:"
	@echo "  sudo systemctl restart alice          # System service"
	@echo "  systemctl --user restart alice         # SteamOS/user service"
	@echo "  make run                               # Development"

# Update with service restart (system service)
update-restart: update
	@if systemctl is-active --quiet alice 2>/dev/null; then \
		echo "Restarting system service..."; \
		sudo systemctl restart alice; \
	elif systemctl --user is-active --quiet alice 2>/dev/null; then \
		echo "Restarting user service..."; \
		systemctl --user restart alice; \
	else \
		echo "No running service found. Start manually with: make run"; \
	fi

# ==========================================
# Docker
# ==========================================

# Build CPU Docker image
docker:
	docker build -t alice:latest -t alice:$(VERSION) .

# Build CUDA Docker image
docker-cuda:
	docker build --build-arg GPU=cuda -t alice:cuda -t alice:$(VERSION)-cuda .

# Build ROCm Docker image
docker-rocm:
	docker build -f Dockerfile.rocm -t alice:rocm -t alice:$(VERSION)-rocm .

# Run with Docker Compose (CPU)
docker-up:
	docker compose --profile default up -d

# Run with Docker Compose (CUDA)
docker-up-cuda:
	docker compose --profile cuda up -d

# Run with Docker Compose (ROCm)
docker-up-rocm:
	docker compose --profile rocm up -d

# Stop Docker Compose
docker-down:
	docker compose down

# Show Docker logs
docker-logs:
	docker compose logs -f

# ==========================================
# Distribution
# ==========================================

# Create a distribution tarball
dist: clean-dist
	@echo "Building ALICE $(VERSION) distribution..."
	@mkdir -p dist/alice-$(VERSION)
	@rsync -a --exclude-from=.distignore --exclude='dist/' --exclude='.alice-backups/' . dist/alice-$(VERSION)/
	@cd dist && tar -czf alice-$(VERSION).tar.gz alice-$(VERSION)
	@cd dist && sha256sum alice-$(VERSION).tar.gz > alice-$(VERSION).tar.gz.sha256
	@echo "Created: dist/alice-$(VERSION).tar.gz"
	@echo "SHA256:  $$(cat dist/alice-$(VERSION).tar.gz.sha256)"

clean-dist:
	rm -rf dist/

# ==========================================
# Version
# ==========================================

# Show current version
version:
	@echo "ALICE $(VERSION)"

# ==========================================
# Cleanup
# ==========================================

# Clean build artifacts
clean:
	rm -rf $(VENV)
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Remove generated images (be careful!)
clean-images:
	rm -rf images/*

# Remove logs
clean-logs:
	rm -rf logs/*

# Full clean (including images and logs)
clean-all: clean clean-images clean-logs clean-dist

# Show help
help:
	@echo "ALICE Makefile Commands:"
	@echo ""
	@echo "  Development:"
	@echo "    make install        - Create venv and install dependencies"
	@echo "    make run            - Run the server"
	@echo "    make dev            - Run server with auto-reload"
	@echo "    make test           - Run tests"
	@echo "    make lint           - Run linter"
	@echo "    make version        - Show current version"
	@echo ""
	@echo "  Updates:"
	@echo "    make check-update   - Check for available updates"
	@echo "    make update         - Update from git and reinstall deps"
	@echo "    make update-restart - Update and restart the service"
	@echo ""
	@echo "  Docker:"
	@echo "    make docker         - Build CPU Docker image"
	@echo "    make docker-cuda    - Build CUDA Docker image"
	@echo "    make docker-rocm    - Build ROCm Docker image"
	@echo "    make docker-up      - Start with Docker Compose (CPU)"
	@echo "    make docker-up-cuda - Start with Docker Compose (CUDA)"
	@echo "    make docker-up-rocm - Start with Docker Compose (ROCm)"
	@echo "    make docker-down    - Stop Docker Compose"
	@echo "    make docker-logs    - Show Docker logs"
	@echo ""
	@echo "  Distribution:"
	@echo "    make dist           - Create distribution tarball"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean          - Clean build artifacts"
	@echo "    make clean-images   - Remove generated images"
	@echo "    make clean-logs     - Remove log files"
	@echo "    make clean-all      - Full clean"
	@echo "    make help           - Show this help"
