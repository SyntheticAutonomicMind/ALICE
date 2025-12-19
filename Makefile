# ALICE Makefile
# Build and run commands for ALICE service

.PHONY: install run test clean lint dev setup-dirs

VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
UVICORN = $(VENV)/bin/uvicorn

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
	mkdir -p models images logs

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
clean-all: clean clean-images clean-logs

# Show help
help:
	@echo "ALICE Makefile Commands:"
	@echo "  make install      - Create venv and install dependencies"
	@echo "  make run          - Run the server"
	@echo "  make dev          - Run server with auto-reload"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linter"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make clean-images - Remove generated images"
	@echo "  make clean-logs   - Remove log files"
	@echo "  make clean-all    - Full clean"
	@echo "  make help         - Show this help"
