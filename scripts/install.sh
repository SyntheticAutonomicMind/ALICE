#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2025 Andrew Wyatt (Fewtarius)
#
# ALICE Installation Script
# Installs ALICE as a systemd service on Linux or a user LaunchAgent on macOS
#
# Linux usage:  sudo ./install.sh [--uninstall]
# macOS usage:  ./install.sh [--uninstall]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

APP_NAME="alice"

# Detect OS first so we can set platform-appropriate paths
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    else
        echo -e "${RED}Unsupported operating system: $OSTYPE${NC}"
        exit 1
    fi
}

detect_os

# Platform-specific paths
if [[ "$OS" == "macos" ]]; then
    # macOS: user-local installation (no root required)
    INSTALL_DIR="${HOME}/Library/Application Support/${APP_NAME}"
    CONFIG_DIR="${HOME}/.config/${APP_NAME}"
    DATA_DIR="${HOME}/Library/Application Support/${APP_NAME}/data"
    LOG_DIR="${HOME}/Library/Logs/${APP_NAME}"
    SERVICE_USER="$(whoami)"
    SERVICE_GROUP="staff"
else
    # Linux: system-wide installation (requires root)
    INSTALL_DIR="/opt/${APP_NAME}"
    CONFIG_DIR="/etc/${APP_NAME}"
    DATA_DIR="/var/lib/${APP_NAME}"
    LOG_DIR="/var/log/${APP_NAME}"
    SERVICE_USER="${APP_NAME}"
    SERVICE_GROUP="${APP_NAME}"
fi

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        echo -e "${RED}This script must be run as root (use sudo)${NC}"
        exit 1
    fi
}

# Print status message
print_status() {
    echo -e "${GREEN}[*]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[X]${NC} $1"
}

# Create system user
create_user() {
    print_status "Creating system user: ${SERVICE_USER}"
    
    if [[ "$OS" == "linux" ]]; then
        if ! id -u "${SERVICE_USER}" &>/dev/null; then
            useradd -r -s /bin/false -d "${INSTALL_DIR}" "${SERVICE_USER}"
        else
            print_warning "User ${SERVICE_USER} already exists"
        fi
    elif [[ "$OS" == "macos" ]]; then
        # On macOS, service runs as the current user via LaunchAgent
        print_status "macOS: Service will run as current user ($(whoami))"
    fi
}

# Create directories
create_directories() {
    print_status "Creating directories"
    
    mkdir -p "${INSTALL_DIR}"
    mkdir -p "${CONFIG_DIR}"
    mkdir -p "${LOG_DIR}"
    
    # On SteamOS, create data directory in offload partition
    if [[ -d /home/.steamos/offload ]]; then
        print_status "SteamOS detected - using offload partition for data"
        mkdir -p "/home/.steamos/offload/var/lib/${APP_NAME}/models/loras"
        mkdir -p "/home/.steamos/offload/var/lib/${APP_NAME}/images"
        mkdir -p "/home/.steamos/offload/var/lib/${APP_NAME}/data/auth"
    else
        mkdir -p "${DATA_DIR}/models/loras"
        mkdir -p "${DATA_DIR}/images"
        mkdir -p "${DATA_DIR}/data/auth"
    fi
    
    # Set permissions
    if [[ "$OS" == "linux" ]]; then
        chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${INSTALL_DIR}"
        chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${CONFIG_DIR}"
        chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${LOG_DIR}"
        
        if [[ -d /home/.steamos/offload ]]; then
            chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "/home/.steamos/offload/var/lib/${APP_NAME}"
        else
            chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${DATA_DIR}"
        fi
    fi
    
    chmod 755 "${INSTALL_DIR}"
    chmod 755 "${CONFIG_DIR}"
    chmod 755 "${DATA_DIR}" 2>/dev/null || true
    chmod 755 "${LOG_DIR}"
}

# Install application files
install_files() {
    print_status "Installing application files to ${INSTALL_DIR}"
    
    # Get script directory
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SOURCE_DIR="$(dirname "${SCRIPT_DIR}")"
    
    # Copy source files
    cp -r "${SOURCE_DIR}/src" "${INSTALL_DIR}/"
    cp -r "${SOURCE_DIR}/web" "${INSTALL_DIR}/"
    cp "${SOURCE_DIR}/requirements.txt" "${INSTALL_DIR}/"
    
    # Set ownership on copied files
    if [[ "$OS" == "linux" ]]; then
        chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${INSTALL_DIR}"
    fi
    
    # Copy config if not exists
    if [[ ! -f "${CONFIG_DIR}/config.yaml" ]]; then
        print_status "Installing default configuration"
        cp "${SOURCE_DIR}/config.yaml" "${CONFIG_DIR}/config.yaml"
        
        # Update paths in config for system installation
        if [[ "$OS" == "linux" ]]; then
            sed -i "s|directory: .*models|directory: ${DATA_DIR}/models|g" "${CONFIG_DIR}/config.yaml"
            sed -i "s|images_directory: .*images|images_directory: ${DATA_DIR}/images|g" "${CONFIG_DIR}/config.yaml"
            sed -i "s|gallery_file: .*gallery.json|gallery_file: ${DATA_DIR}/data/gallery.json|g" "${CONFIG_DIR}/config.yaml"
            sed -i "s|auth_directory: .*auth|auth_directory: ${DATA_DIR}/data/auth|g" "${CONFIG_DIR}/config.yaml"
            sed -i "s|database_path: .*model_cache.db|database_path: ${DATA_DIR}/data/model_cache.db|g" "${CONFIG_DIR}/config.yaml"
            sed -i "s|file: .*alice.log|file: ${LOG_DIR}/alice.log|g" "${CONFIG_DIR}/config.yaml"
            chown "${SERVICE_USER}:${SERVICE_GROUP}" "${CONFIG_DIR}/config.yaml"
        elif [[ "$OS" == "macos" ]]; then
            sed -i '' "s|directory: .*models|directory: ${DATA_DIR}/models|g" "${CONFIG_DIR}/config.yaml"
            sed -i '' "s|images_directory: .*images|images_directory: ${DATA_DIR}/images|g" "${CONFIG_DIR}/config.yaml"
            sed -i '' "s|gallery_file: .*gallery.json|gallery_file: ${DATA_DIR}/data/gallery.json|g" "${CONFIG_DIR}/config.yaml"
            sed -i '' "s|auth_directory: .*auth|auth_directory: ${DATA_DIR}/data/auth|g" "${CONFIG_DIR}/config.yaml"
            sed -i '' "s|database_path: .*model_cache.db|database_path: ${DATA_DIR}/data/model_cache.db|g" "${CONFIG_DIR}/config.yaml"
            sed -i '' "s|file: .*alice.log|file: ${LOG_DIR}/alice.log|g" "${CONFIG_DIR}/config.yaml"
        fi
    else
        print_warning "Configuration already exists, skipping"
    fi
}

# Create Python virtual environment
create_venv() {
    print_status "Creating Python virtual environment"
    
    python3 -m venv "${INSTALL_DIR}/venv"
    
    print_status "Installing Python dependencies"
    "${INSTALL_DIR}/venv/bin/pip" install --upgrade pip
    
    # Detect platform and install appropriate PyTorch
    if [[ "$OS" == "macos" ]]; then
        # macOS: Apple Silicon uses MPS acceleration, Intel uses CPU
        ARCH="$(uname -m)"
        if [[ "$ARCH" == "arm64" ]]; then
            print_status "Apple Silicon (arm64) detected - installing PyTorch with MPS support"
            GPU_TYPE="mps"
            "${INSTALL_DIR}/venv/bin/pip" install torch torchvision torchaudio
        else
            print_status "Intel Mac detected - installing CPU PyTorch (no GPU acceleration)"
            GPU_TYPE="cpu"
            "${INSTALL_DIR}/venv/bin/pip" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        fi
    else
        # Linux: detect GPU via lspci
        GPU_TYPE="cpu"
        if lspci 2>/dev/null | grep -iq "vga.*amd\|vga.*ati"; then
            print_status "AMD GPU detected - installing PyTorch with ROCm support"
            GPU_TYPE="amd"
            
            # Check for gfx1103 (Phoenix APU)
            if lspci 2>/dev/null | grep -i "phoenix"; then
                print_status "Phoenix APU (gfx1103) detected - using TheRock nightlies"
                "${INSTALL_DIR}/venv/bin/pip" install --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ --pre torch torchaudio torchvision
            else
                # Standard ROCm installation
                "${INSTALL_DIR}/venv/bin/pip" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
            fi
        elif lspci 2>/dev/null | grep -iq "vga.*nvidia"; then
            print_status "NVIDIA GPU detected - installing PyTorch with CUDA support"
            GPU_TYPE="nvidia"
            "${INSTALL_DIR}/venv/bin/pip" install torch torchvision torchaudio
        else
            print_warning "No supported GPU detected - installing CPU-only PyTorch"
            "${INSTALL_DIR}/venv/bin/pip" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        fi
    fi
    
    # Install remaining dependencies (excluding torch packages from requirements.txt)
    "${INSTALL_DIR}/venv/bin/pip" install -r "${INSTALL_DIR}/requirements.txt"
    
    # Set ownership
    if [[ "$OS" == "linux" ]]; then
        chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${INSTALL_DIR}/venv"
    fi
    
    print_status "PyTorch installation complete (type: ${GPU_TYPE})"
}

# Build and install stable-diffusion.cpp (Vulkan backend)
build_sdcpp() {
    print_status "Building stable-diffusion.cpp with Vulkan support"
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    # Run build script
    if [[ -f "${SCRIPT_DIR}/build_sdcpp.sh" ]]; then
        INSTALL_PREFIX="${INSTALL_DIR}" bash "${SCRIPT_DIR}/build_sdcpp.sh"
        
        # Make sd-cli available system-wide
        if [[ -f "${INSTALL_DIR}/bin/sd-cli" ]]; then
            ln -sf "${INSTALL_DIR}/bin/sd-cli" /usr/local/bin/sd-cli
            print_status "sd-cli installed to /usr/local/bin/sd-cli"
        fi
    else
        print_warning "build_sdcpp.sh not found, skipping Vulkan backend build"
    fi
}

# Install systemd service (Linux)
install_systemd_service() {
    print_status "Installing systemd service"
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SOURCE_DIR="$(dirname "${SCRIPT_DIR}")"
    
    # Check if running on SteamOS/Deck with offload partition
    if [[ -d /home/.steamos/offload ]]; then
        print_status "SteamOS detected - setting up offload mount for /var/lib/${APP_NAME}"
        
        # Create offload directory structure
        mkdir -p "/home/.steamos/offload/var/lib/${APP_NAME}"
        chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "/home/.steamos/offload/var/lib/${APP_NAME}"
        
        # Install mount unit if it exists
        if [[ -f "${SOURCE_DIR}/var-lib-${APP_NAME}.mount" ]]; then
            cp "${SOURCE_DIR}/var-lib-${APP_NAME}.mount" /etc/systemd/system/
            systemctl daemon-reload
            systemctl enable "var-lib-${APP_NAME}.mount"
            systemctl start "var-lib-${APP_NAME}.mount"
            print_status "Offload mount installed and started"
        fi
    fi
    
    # Install main service
    cp "${SOURCE_DIR}/${APP_NAME}.service" /etc/systemd/system/
    
    systemctl daemon-reload
    systemctl enable "${APP_NAME}"
    
    print_status "Service installed. Start with: systemctl start ${APP_NAME}"
}

# Install launchd user service (macOS)
install_launchd_service() {
    print_status "Installing launchd user service (LaunchAgent)"
    
    LAUNCH_AGENT_DIR="${HOME}/Library/LaunchAgents"
    PLIST_PATH="${LAUNCH_AGENT_DIR}/com.alice.plist"
    
    mkdir -p "${LAUNCH_AGENT_DIR}"
    
    cat > "${PLIST_PATH}" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.alice</string>
    <key>ProgramArguments</key>
    <array>
        <string>${INSTALL_DIR}/venv/bin/python</string>
        <string>-m</string>
        <string>uvicorn</string>
        <string>src.main:app</string>
        <string>--host</string>
        <string>0.0.0.0</string>
        <string>--port</string>
        <string>8080</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${INSTALL_DIR}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>ALICE_CONFIG</key>
        <string>${CONFIG_DIR}/config.yaml</string>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${LOG_DIR}/alice.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/alice.log</string>
</dict>
</plist>
EOF
    
    chmod 644 "${PLIST_PATH}"
    
    # Load the LaunchAgent for the current user session (no sudo required)
    launchctl load "${PLIST_PATH}" 2>/dev/null || true
    
    print_status "LaunchAgent installed and loaded"
    print_status "  Start:   launchctl start com.alice"
    print_status "  Stop:    launchctl stop com.alice"
    print_status "  Disable: launchctl unload ${PLIST_PATH}"
}

# Uninstall function
uninstall() {
    print_status "Uninstalling ${APP_NAME}"
    
    if [[ "$OS" == "linux" ]]; then
        # Stop and disable service
        systemctl stop "${APP_NAME}" 2>/dev/null || true
        systemctl disable "${APP_NAME}" 2>/dev/null || true
        rm -f "/etc/systemd/system/${APP_NAME}.service"
        systemctl daemon-reload
        
        # Remove user
        userdel "${SERVICE_USER}" 2>/dev/null || true
    elif [[ "$OS" == "macos" ]]; then
        PLIST_PATH="${HOME}/Library/LaunchAgents/com.alice.plist"
        launchctl unload "${PLIST_PATH}" 2>/dev/null || true
        rm -f "${PLIST_PATH}"
    fi
    
    # Remove directories (prompt for confirmation)
    echo ""
    read -p "Remove installation directory ${INSTALL_DIR}? [y/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "${INSTALL_DIR}"
        print_status "Removed ${INSTALL_DIR}"
    fi
    
    read -p "Remove data directory ${DATA_DIR}? (WARNING: This deletes models and images) [y/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "${DATA_DIR}"
        print_status "Removed ${DATA_DIR}"
    fi
    
    read -p "Remove configuration ${CONFIG_DIR}? [y/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "${CONFIG_DIR}"
        print_status "Removed ${CONFIG_DIR}"
    fi
    
    read -p "Remove logs ${LOG_DIR}? [y/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "${LOG_DIR}"
        print_status "Removed ${LOG_DIR}"
    fi
    
    print_status "Uninstallation complete"
}

# Main installation
install() {
    print_status "Installing ${APP_NAME}"
    echo ""
    
    create_user
    create_directories
    install_files
    create_venv
    if [[ "$OS" == "linux" ]]; then
        build_sdcpp  # Vulkan backend (Linux only)
    fi
    
    if [[ "$OS" == "linux" ]]; then
        install_systemd_service
    elif [[ "$OS" == "macos" ]]; then
        install_launchd_service
    fi
    
    echo ""
    print_status "Installation complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Add Stable Diffusion models to: ${DATA_DIR}/models"
    echo "  2. Edit configuration if needed: ${CONFIG_DIR}/config.yaml"
    if [[ "$OS" == "linux" ]]; then
        echo "  3. Start the service: sudo systemctl start ${APP_NAME}"
        echo "  4. Check status: sudo systemctl status ${APP_NAME}"
        echo "  5. View logs: sudo journalctl -u ${APP_NAME} -f"
    elif [[ "$OS" == "macos" ]]; then
        echo "  3. Service is already running via LaunchAgent"
        echo "     Stop:  launchctl stop com.alice"
        echo "     Start: launchctl start com.alice"
        echo "  4. Check logs: tail -f ${LOG_DIR}/alice.log"
    fi
    echo ""
    echo "Web interface will be available at: http://localhost:8080/web/"
    echo "API documentation at: http://localhost:8080/docs"
    echo ""
    if [[ "$OS" == "linux" ]]; then
        echo "Backends installed:"
        echo "  - PyTorch (ROCm/CUDA/CPU)"
        echo "  - stable-diffusion.cpp (Vulkan - universal AMD support)"
    else
        echo "Backend installed:"
        echo "  - PyTorch (Apple MPS / CPU)"
    fi
}

# Entry point - detect_os already called at top of script
if [[ "$1" == "--uninstall" ]]; then
    if [[ "$OS" == "linux" ]]; then
        check_root
    fi
    uninstall
else
    if [[ "$OS" == "linux" ]]; then
        check_root
    fi
    install
fi
