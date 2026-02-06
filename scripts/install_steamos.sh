#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2025 Andrew Wyatt (Fewtarius)
#
# ALICE Installation Script for SteamOS
# Automatically detects AMD GPU and configures the service
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALICE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_DIR="${HOME}/.config/alice"
DATA_DIR="${HOME}/.local/share/alice"
SERVICE_FILE="${HOME}/.config/systemd/user/alice.service"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Source the AMD detection script
source "${SCRIPT_DIR}/detect_amd_gpu.sh"

install_alice() {
    log_info "Installing ALICE for SteamOS..."
    
    # Check for AMD GPU
    log_info "Detecting GPU..."
    local gpu_env=$(detect_amd_gpu)
    
    if echo "$gpu_env" | grep -q "PYTORCH_ROCM_ARCH"; then
        log_info "AMD GPU detected with ROCm support"
        echo "$gpu_env" | grep -v "^#"
        USE_GPU=true
    else
        log_warn "No supported AMD GPU detected, will use CPU mode"
        USE_GPU=false
    fi
    
    # Create directories
    log_info "Creating directories..."
    mkdir -p "${CONFIG_DIR}"
    mkdir -p "${DATA_DIR}/models/loras"
    mkdir -p "${DATA_DIR}/images"
    mkdir -p "${DATA_DIR}/logs"
    mkdir -p "${DATA_DIR}/auth"
    mkdir -p "${DATA_DIR}/data"
    mkdir -p "${HOME}/.config/systemd/user"
    mkdir -p "${HOME}/tmp"  # For temporary files during generation
    
    # Create Python virtual environment if it doesn't exist
    if [[ ! -d "${ALICE_DIR}/venv" ]]; then
        log_info "Creating Python virtual environment..."
        python3 -m venv "${ALICE_DIR}/venv"
    fi
    
    # Install/upgrade pip and dependencies
    log_info "Installing Python dependencies..."
    "${ALICE_DIR}/venv/bin/pip" install --upgrade pip
    
    if [[ "$USE_GPU" == "true" ]]; then
        # Determine PyTorch version based on GPU architecture
        local gfx_arch=$(echo "$gpu_env" | grep PYTORCH_ROCM_ARCH | sed 's/export PYTORCH_ROCM_ARCH=//' | tr -d '"')
        
        if [[ "$gfx_arch" == "gfx1103" ]]; then
            # Install PyTorch with TheRock gfx110X-all ROCm support
            # TheRock builds include proper gfx1103 (Phoenix APU) kernels
            # The official PyTorch ROCm packages do NOT include gfx1103 and cause segfaults
            # See: https://github.com/ROCm/TheRock/blob/main/RELEASES.md
            log_info "Installing PyTorch with TheRock ROCm support (gfx110X family)..."
            log_info "This includes native gfx1103 (Phoenix/780M) support!"
            "${ALICE_DIR}/venv/bin/pip" install \
                --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ \
                --pre torch torchaudio torchvision
        elif [[ "$gfx_arch" == "gfx90c" ]]; then
            # Cezanne/Renoir APUs - limited ROCm support
            # Python 3.13 doesn't have ROCm wheels yet, check Python version
            local python_version=$("${ALICE_DIR}/venv/bin/python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
            if [[ "$python_version" == "3.13" ]]; then
                log_warn "Python 3.13 detected - PyTorch ROCm not available yet"
                log_warn "Installing CPU-only PyTorch. For GPU support:"
                log_warn "  1. Create venv with Python 3.11: python3.11 -m venv venv"
                log_warn "  2. Reinstall with ROCm 6.1: pip install torch --index-url https://download.pytorch.org/whl/rocm6.1"
                "${ALICE_DIR}/venv/bin/pip" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
                USE_GPU=false  # Update flag to reflect CPU mode
            else
                log_info "Installing PyTorch with ROCm 6.1 support (gfx90c Cezanne/Renoir)..."
                "${ALICE_DIR}/venv/bin/pip" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
            fi
        else
            # Other AMD GPUs - use official ROCm 6.2
            log_info "Installing PyTorch with ROCm 6.2 support ($gfx_arch)..."
            "${ALICE_DIR}/venv/bin/pip" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
        fi
    else
        # Install CPU-only PyTorch
        log_info "Installing PyTorch 2.6.0 (CPU only)..."
        "${ALICE_DIR}/venv/bin/pip" install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install other dependencies with exact tested versions
    log_info "Installing remaining dependencies..."
    "${ALICE_DIR}/venv/bin/pip" install \
        diffusers==0.35.2 \
        transformers==4.57.3 \
        accelerate==1.12.0 \
        safetensors==0.7.0 \
        compel==2.3.1 \
        pillow==12.0.0 \
        pyyaml==6.0.3 \
        psutil==7.1.3 \
        pydantic==2.12.5 \
        pydantic-settings==2.1.0 \
        fastapi==0.104.1 \
        uvicorn==0.24.0 \
        aiofiles==23.2.1 \
        aiohttp==3.9.3 \
        python-multipart==0.0.6 \
        huggingface-hub==0.36.0
    
    # Create config file if it doesn't exist
    if [[ ! -f "${CONFIG_DIR}/config.yaml" ]]; then
        log_info "Creating configuration file..."
        
        # Determine port (8090 preferred, or 8091 if busy)
        local port=8090
        if ss -tulpn 2>/dev/null | grep -q ":8090 "; then
            port=8091
        fi
        
        # Generate a secure random API key
        local admin_key=$(openssl rand -hex 16)
        
        cat > "${CONFIG_DIR}/config.yaml" << EOF
# ALICE Configuration - Auto-generated by install_steamos.sh
# Date: $(date -Iseconds)

server:
  host: "0.0.0.0"
  port: ${port}
  require_auth: true
  session_timeout_seconds: 3600
  registration_mode: disabled
  block_nsfw: false

storage:
  images_directory: ${DATA_DIR}/images
  gallery_file: ${DATA_DIR}/data/gallery.json
  auth_directory: ${DATA_DIR}/auth
  max_storage_gb: 100
  retention_days: 7
  public_image_expiration_hours: 168
  gallery_page_size: 100

models:
  directory: ${DATA_DIR}/models
  auto_unload_timeout: 300
  default_model: ""

generation:
  default_steps: 20
  default_guidance_scale: 7.5
  default_scheduler: "dpm++_sde_karras"
  default_width: 512
  default_height: 512
  request_timeout: 600
  max_concurrent: 1
  force_cpu: $([ "$USE_GPU" == "true" ] && echo "false" || echo "true")
  force_float32: $(if [[ "$gfx_arch" == "gfx1103" ]]; then echo "true"; else echo "false"; fi)
  device_map: $(if [[ "$gfx_arch" == "gfx1103" ]]; then echo '"sequential"'; else echo 'null'; fi)
  force_bfloat16: false
  
  # Memory optimizations for APU/GPU (3GB-8GB VRAM range)
  enable_vae_slicing: true
  enable_vae_tiling: false
  enable_model_cpu_offload: false
  enable_sequential_cpu_offload: false
  attention_slice_size: "auto"
  vae_decode_cpu: $(if [[ "$gfx_arch" == "gfx1103" ]]; then echo "true"; else echo "false"; fi)
  
  # Performance optimization settings (PyTorch 2.0+)
  enable_torch_compile: false
  torch_compile_mode: "reduce-overhead"

logging:
  level: "INFO"
  file: ${DATA_DIR}/logs/alice.log
  max_size_mb: 100
  backup_count: 5

model_cache:
  enabled: true
  database_path: ${DATA_DIR}/data/model_cache.db
  sync_on_startup: false
  sync_interval_hours: 24
  civitai_page_limit: null
  huggingface_limit: 10000
EOF

        log_info "Configuration saved to: ${CONFIG_DIR}/config.yaml"
        log_info ""
        log_info "IMPORTANT: On first run, access http://localhost:${port}/web/login.html"
        log_info "A temporary admin API key will be generated in the console log."
        log_info ""
    fi
    
    # Create systemd service
    log_info "Creating systemd service..."
    
    # Build environment variables section
    local env_lines="Environment=\"ALICE_CONFIG=${CONFIG_DIR}/config.yaml\"
Environment=\"TMPDIR=${HOME}/tmp\""
    
    if [[ "$USE_GPU" == "true" ]]; then
        # Extract env vars from detection
        local rocm_arch=$(echo "$gpu_env" | grep PYTORCH_ROCM_ARCH | sed 's/export //' | cut -d= -f2 | tr -d '"')
        local hsa_ver=$(echo "$gpu_env" | grep HSA_OVERRIDE_GFX_VERSION | sed 's/export //' | cut -d= -f2 | tr -d '"')
        
        env_lines="${env_lines}
Environment=\"PYTORCH_ROCM_ARCH=${rocm_arch}\"
Environment=\"HSA_OVERRIDE_GFX_VERSION=${hsa_ver}\"
Environment=\"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1\""
    fi
    
    cat > "${SERVICE_FILE}" << EOF
[Unit]
Description=ALICE - Remote Stable Diffusion Service
After=network.target

[Service]
Type=simple
WorkingDirectory=${ALICE_DIR}
${env_lines}
ExecStart=${ALICE_DIR}/venv/bin/python -m src.main
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
    
    # Reload systemd
    log_info "Reloading systemd..."
    systemctl --user daemon-reload
    
    # Enable and start service
    log_info "Enabling and starting ALICE service..."
    systemctl --user enable alice
    systemctl --user start alice
    
    # Wait for service to start
    sleep 3
    
    # Check status
    if systemctl --user is-active --quiet alice; then
        local config_port=$(grep "port:" "${CONFIG_DIR}/config.yaml" | head -1 | awk '{print $2}')
        log_info "ALICE installed and running!"
        log_info ""
        log_info "Access the web interface at: http://localhost:${config_port}/web/"
        log_info "API documentation at: http://localhost:${config_port}/docs"
        log_info ""
        log_info "On first run, check the logs for your temporary admin API key:"
        log_info "  journalctl --user -u alice | grep 'admin'"
        log_info ""
        log_info "To add models, copy .safetensors files to:"
        log_info "  ${DATA_DIR}/models/"
        log_info ""
        log_info "Or use the Download page to get models from CivitAI/HuggingFace"
    else
        log_error "Service failed to start. Check logs with:"
        log_error "  journalctl --user -u alice -n 50"
    fi
}

uninstall_alice() {
    log_info "Uninstalling ALICE..."
    
    # Stop and disable service
    systemctl --user stop alice 2>/dev/null || true
    systemctl --user disable alice 2>/dev/null || true
    
    # Remove service file
    rm -f "${SERVICE_FILE}"
    systemctl --user daemon-reload
    
    log_info "ALICE service removed."
    log_info ""
    log_info "Data directories preserved at:"
    log_info "  Config: ${CONFIG_DIR}"
    log_info "  Data: ${DATA_DIR}"
    log_info ""
    log_info "To completely remove, also delete:"
    log_info "  rm -rf ${CONFIG_DIR} ${DATA_DIR} ${ALICE_DIR}"
}

# Main
case "${1:-install}" in
    install)
        install_alice
        ;;
    uninstall)
        uninstall_alice
        ;;
    reinstall)
        uninstall_alice
        install_alice
        ;;
    *)
        echo "Usage: $0 [install|uninstall|reinstall]"
        exit 1
        ;;
esac
