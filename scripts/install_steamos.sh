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
        
        if [[ "$gfx_arch" == "gfx1103" || "$gfx_arch" == "gfx1151" ]]; then
            # Phoenix (gfx1103) and Strix Halo (gfx1151) are supported by
            # the ROCm 10.0.0 multi-arch stable index via device extras.
            # This replaces the legacy TheRock nightly indices and all
            # their workarounds (--pre, --no-deps, manual rocm-sdk-libraries,
            # and the torchvision::nms _meta_registrations patch).
            log_info "Detected AMD GPU ($gfx_arch) - using ROCm 10.0.0 multi-arch packages"
            "${ALICE_DIR}/venv/bin/pip" install \
                --index-url https://stable.repo.amd.com/rocm/whl-next/ \
                "torch[device-${gfx_arch}]==2.13.0+rocm10.0.0" \
                "torchvision[device-${gfx_arch}]==0.28.0+rocm10.0.0" \
                "torchaudio==2.11.0.2+rocm10.0.0"
        elif [[ "$gfx_arch" == "gfx90c" ]]; then
            # Cezanne/Renoir APUs (Ryzen 5000/4000 series) are not
            # supported by ROCm 10.0.0.  Fall back to CPU-only.
            log_warn "Detected AMD APU ($gfx_arch) - not supported by ROCm 10.0.0, using CPU-only PyTorch"
            "${ALICE_DIR}/venv/bin/pip" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            USE_GPU=false
        else
            # Other AMD GPUs (RDNA2/RDNA3/CDNA) - use ROCm 10.0.0 multi-arch
            # with device-all for broad architecture coverage.
            log_info "Detected AMD GPU ($gfx_arch) - using ROCm 10.0.0 multi-arch packages"
            "${ALICE_DIR}/venv/bin/pip" install \
                --index-url https://stable.repo.amd.com/rocm/whl-next/ \
                "torch[device-all]==2.13.0+rocm10.0.0" \
                "torchvision[device-all]==0.28.0+rocm10.0.0" \
                "torchaudio==2.11.0.2+rocm10.0.0"
        fi
    else
        # Install CPU-only PyTorch
        log_info "Installing latest PyTorch (CPU only)..."
        "${ALICE_DIR}/venv/bin/pip" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install remaining dependencies (latest compatible versions)
    log_info "Installing remaining dependencies..."
    "${ALICE_DIR}/venv/bin/pip" install \
        # diffusers: use the latest git main so MiniMax-Music3 and other
        # latest pipelines are importable.  Pinned releases may be too old.
        "diffusers @ git+https://github.com/huggingface/diffusers" \
        transformers>=5.0 \
        accelerate>=1.12.0 \
        safetensors>=0.7.0 \
        compel>=2.3.1 \
        pillow>=12.0.0 \
        pyyaml>=6.0.3 \
        psutil>=7.1.3 \
        pydantic>=2.12.5 \
        pydantic-settings>=2.1.0 \
        fastapi>=0.104.1 \
        uvicorn>=0.24.0 \
        aiofiles>=23.2.1 \
        aiohttp>=3.9.3 \
        python-multipart>=0.0.6 \
        huggingface-hub>=1.0.0 \
        einops \
        stable-audio-tools

    # Build stable-diffusion.cpp (Vulkan backend) — provides universal GPU
    # support via Vulkan, working alongside ROCm as a fallback.
    log_info "Building stable-diffusion.cpp (Vulkan backend)..."
    if [[ -f "${SCRIPT_DIR}/build_sdcpp.sh" ]]; then
        INSTALL_PREFIX="${ALICE_DIR}" bash "${SCRIPT_DIR}/build_sdcpp.sh" || {
            log_warn "sd.cpp build failed — Vulkan backend will be unavailable."
            log_warn "ALICE will fall back to ROCm/PyTorch for image generation."
        }
        # Make sd-cli available system-wide if it was built
        if [[ -f "${ALICE_DIR}/sd.cpp/build/bin/sd-cli" ]]; then
            ln -sf "${ALICE_DIR}/sd.cpp/build/bin/sd-cli" "${HOME}/.local/bin/sd-cli"
            mkdir -p "${HOME}/.local/bin"
            log_info "sd-cli symlinked to ~/.local/bin/sd-cli"
        fi
    else
        log_warn "build_sdcpp.sh not found at ${SCRIPT_DIR}/build_sdcpp.sh — skipping Vulkan backend"
    fi

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
  force_float32: $(if [[ "$gfx_arch" == "gfx1103" ]] || [[ "$gfx_arch" == "gfx90c" ]]; then echo "true"; else echo "false"; fi)
  device_map: $(if [[ "$gfx_arch" == "gfx1103" ]] || [[ "$gfx_arch" == "gfx90c" ]]; then echo '"sequential"'; else echo 'null'; fi)
  force_bfloat16: false
  
  # Memory optimizations for APU/GPU (3GB-8GB VRAM range)
  enable_vae_slicing: true
  enable_vae_tiling: false
  enable_model_cpu_offload: false
  enable_sequential_cpu_offload: false
  attention_slice_size: "auto"
  # VAE decode on CPU prevents GPU hangs on AMD gfx1103 and gfx1151
  vae_decode_cpu: $(if [[ "$gfx_arch" == "gfx1103" ]] || [[ "$gfx_arch" == "gfx1151" ]] || [[ "$gfx_arch" == "gfx90c" ]]; then echo "true"; else echo "false"; fi)
  
  # Performance optimization settings (PyTorch 2.0+)
  # Disabled on SteamOS to avoid stability issues with ROCm + systemd
  enable_torch_compile: false
  torch_compile_mode: "reduce-overhead"
  
  # stable-diffusion.cpp (Vulkan) backend configuration
  # sd-cli provides universal GPU support via Vulkan, working alongside ROCm.
  # Set backend to "auto" so ALICE picks Vulkan when ROCm isn't available.
  backend: "auto"
  sdcpp_binary: null
  sdcpp_threads: 4

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
