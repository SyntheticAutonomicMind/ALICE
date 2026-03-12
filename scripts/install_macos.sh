#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2025 Andrew Wyatt (Fewtarius)
#
# ALICE macOS Installer
# Sets up ALICE for use with SAM on macOS (Apple Silicon or Intel)
#
# Usage: ./scripts/install_macos.sh [--service | --manual | --uninstall]
#
#   (default)   Install as user LaunchAgent - starts at login automatically
#   --service   Same as default
#   --manual    Install files only, no background service
#   --uninstall Remove ALICE and its LaunchAgent

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

APP_NAME=alice
PORT=8080
INSTALL_DIR="${HOME}/Library/Application Support/${APP_NAME}"
CONFIG_DIR="${HOME}/.config/${APP_NAME}"
DATA_DIR="${HOME}/Library/Application Support/${APP_NAME}/data"
LOG_DIR="${HOME}/Library/Logs/${APP_NAME}"
MODELS_DIR="${DATA_DIR}/models"
IMAGES_DIR="${DATA_DIR}/images"
LAUNCH_AGENT_PLIST="${HOME}/Library/LaunchAgents/com.alice.plist"
PLIST_LABEL=com.alice

print_status()  { echo -e "${GREEN}[*]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
print_error()   { echo -e "${RED}[X]${NC} $1"; }
print_info()    { echo -e "${BLUE}[i]${NC} $1"; }

check_macos() {
    if [[ "$OSTYPE" != "darwin"* ]]; then
        print_error "This script is for macOS only. On Linux, use scripts/install.sh"
        exit 1
    fi
}

check_python() {
    if ! command -v python3 &>/dev/null; then
        print_error "python3 not found. Install it with: brew install python"
        print_info  "Get Homebrew from: https://brew.sh"
        exit 1
    fi

    PY_MAJ="$(python3 -c 'import sys; print(sys.version_info.major)')"
    PY_MIN="$(python3 -c 'import sys; print(sys.version_info.minor)')"
    PY_VER="${PY_MAJ}.${PY_MIN}"

    if [[ "$PY_MAJ" -lt 3 ]] || { [[ "$PY_MAJ" -eq 3 ]] && [[ "$PY_MIN" -lt 10 ]]; }; then
        print_error "Python 3.10+ required (found ${PY_VER})"
        print_info  "Upgrade with: brew install python"
        exit 1
    fi
    print_status "Python ${PY_VER} at $(which python3)"
}

detect_arch() {
    ARCH="$(uname -m)"
    if [[ "$ARCH" == arm64 ]]; then
        print_status "Apple Silicon (arm64) - MPS GPU acceleration enabled"
        PYTORCH_BACKEND=mps
    else
        print_warning "Intel Mac detected - CPU-only inference (significantly slower)"
        PYTORCH_BACKEND=cpu
    fi
}

create_directories() {
    print_status "Creating directories"
    mkdir -p "${INSTALL_DIR}" "${CONFIG_DIR}" "${DATA_DIR}" "${LOG_DIR}"
    mkdir -p "${MODELS_DIR}/loras" "${IMAGES_DIR}" "${DATA_DIR}/auth"
}

install_files() {
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SOURCE_DIR="$(dirname "${SCRIPT_DIR}")"

    print_status "Copying application files to ${INSTALL_DIR}"
    cp -r "${SOURCE_DIR}/src" "${INSTALL_DIR}/"
    cp -r "${SOURCE_DIR}/web" "${INSTALL_DIR}/"
    cp "${SOURCE_DIR}/requirements.txt" "${INSTALL_DIR}/"

    if [[ ! -f "${CONFIG_DIR}/config.yaml" ]]; then
        print_status "Installing default configuration"
        cp "${SOURCE_DIR}/config.yaml" "${CONFIG_DIR}/config.yaml"
        sed -i '' "s|directory: .*models|directory: ${MODELS_DIR}|g"          "${CONFIG_DIR}/config.yaml"
        sed -i '' "s|images_directory: .*|images_directory: ${IMAGES_DIR}|g"  "${CONFIG_DIR}/config.yaml"
        sed -i '' "s|gallery_file: .*|gallery_file: ${DATA_DIR}/gallery.json|g" "${CONFIG_DIR}/config.yaml"
        sed -i '' "s|auth_directory: .*|auth_directory: ${DATA_DIR}/auth|g"   "${CONFIG_DIR}/config.yaml"
        sed -i '' "s|database_path: .*|database_path: ${DATA_DIR}/model_cache.db|g" "${CONFIG_DIR}/config.yaml"
        sed -i '' "s|file: .*alice.log|file: ${LOG_DIR}/alice.log|g"          "${CONFIG_DIR}/config.yaml"
        sed -i '' "s|^  port: .*|  port: ${PORT}|g"                           "${CONFIG_DIR}/config.yaml"
    else
        print_warning "Config already exists at ${CONFIG_DIR}/config.yaml - skipping"
    fi
}

create_venv() {
    print_status "Creating Python virtual environment"
    python3 -m venv "${INSTALL_DIR}/venv"
    "${INSTALL_DIR}/venv/bin/pip" install --upgrade pip -q

    if [[ "$PYTORCH_BACKEND" == mps ]]; then
        print_status "Installing PyTorch for Apple Silicon (MPS)"
        "${INSTALL_DIR}/venv/bin/pip" install torch torchvision torchaudio -q
    else
        print_status "Installing PyTorch (CPU only, Intel Mac)"
        "${INSTALL_DIR}/venv/bin/pip" install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cpu -q
    fi

    print_status "Installing ALICE dependencies"
    "${INSTALL_DIR}/venv/bin/pip" install -r "${INSTALL_DIR}/requirements.txt" -q
}

write_plist() {
    # Use Python to write the plist so we avoid heredoc quoting issues
    python3 << PYEOF
import sys
plist_path = "${LAUNCH_AGENT_PLIST}"
install_dir = "${INSTALL_DIR}"
config_dir = "${CONFIG_DIR}"
log_dir = "${LOG_DIR}"
port = "${PORT}"
label = "${PLIST_LABEL}"
content = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>''' + label + '''</string>
    <key>ProgramArguments</key>
    <array>
        <string>''' + install_dir + '''/venv/bin/python</string>
        <string>-m</string>
        <string>uvicorn</string>
        <string>src.main:app</string>
        <string>--host</string>
        <string>0.0.0.0</string>
        <string>--port</string>
        <string>''' + port + '''</string>
    </array>
    <key>WorkingDirectory</key>
    <string>''' + install_dir + '''</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>ALICE_CONFIG</key>
        <string>''' + config_dir + '''/config.yaml</string>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>StandardOutPath</key>
    <string>''' + log_dir + '''/alice.log</string>
    <key>StandardErrorPath</key>
    <string>''' + log_dir + '''/alice.log</string>
    <key>ThrottleInterval</key>
    <integer>10</integer>
</dict>
</plist>'''
import os
os.makedirs(os.path.dirname(plist_path), exist_ok=True)
with open(plist_path, 'w') as f:
    f.write(content)
PYEOF
}

install_launchagent() {
    print_status "Installing user LaunchAgent (auto-starts at login)"
    write_plist
    chmod 644 "${LAUNCH_AGENT_PLIST}"
    launchctl unload "${LAUNCH_AGENT_PLIST}" 2>/dev/null || true
    launchctl load "${LAUNCH_AGENT_PLIST}"
    print_status "LaunchAgent loaded successfully"
}

uninstall() {
    print_status "Uninstalling ALICE"
    if [[ -f "${LAUNCH_AGENT_PLIST}" ]]; then
        launchctl unload "${LAUNCH_AGENT_PLIST}" 2>/dev/null || true
        rm -f "${LAUNCH_AGENT_PLIST}"
        print_status "LaunchAgent removed"
    fi
    echo ""
    read -p "Remove installation at ${INSTALL_DIR}? [y/N] " -n 1 -r; echo ""
    [[ $REPLY =~ ^[Yy]$ ]] && rm -rf "${INSTALL_DIR}" && print_status "Removed ${INSTALL_DIR}"
    read -p "Remove configuration at ${CONFIG_DIR}? [y/N] " -n 1 -r; echo ""
    [[ $REPLY =~ ^[Yy]$ ]] && rm -rf "${CONFIG_DIR}" && print_status "Removed ${CONFIG_DIR}"
    read -p "Remove logs at ${LOG_DIR}? [y/N] " -n 1 -r; echo ""
    [[ $REPLY =~ ^[Yy]$ ]] && rm -rf "${LOG_DIR}" && print_status "Removed ${LOG_DIR}"
    echo ""
    print_warning "Models at ${MODELS_DIR} were NOT removed."
    print_info    "To remove: rm -rf '${DATA_DIR}'"
    print_status "Uninstall complete"
}

print_summary() {
    echo ""
    echo "============================================"
    print_status "ALICE installation complete!"
    echo "============================================"
    echo ""
    echo "  Install: ${INSTALL_DIR}"
    echo "  Models:  ${MODELS_DIR}"
    echo "  Config:  ${CONFIG_DIR}/config.yaml"
    echo "  Logs:    ${LOG_DIR}/alice.log"
    echo "  Web UI:  http://localhost:${PORT}/web/"
    echo ""
    if [[ "$1" == service ]]; then
        echo "  Service management:"
        echo "    Start:   launchctl start ${PLIST_LABEL}"
        echo "    Stop:    launchctl stop ${PLIST_LABEL}"
        echo "    Disable: launchctl unload '${LAUNCH_AGENT_PLIST}'"
        echo "    Logs:    tail -f '${LOG_DIR}/alice.log'"
    else
        echo "  To start ALICE manually:"
        echo "    cd '${INSTALL_DIR}'"
        echo "    ALICE_CONFIG='${CONFIG_DIR}/config.yaml' venv/bin/python -m src.main"
        echo ""
        echo "  To install as a background service later:"
        echo "    ./scripts/install_macos.sh --service"
    fi
    echo ""
    echo "  Next steps:"
    echo "    1. Add models to:"
    echo "       ${MODELS_DIR}"
    echo "    2. In SAM: Settings > Image Generation"
    echo "       Server URL: http://localhost:${PORT}"
    echo "    3. ALICE auto-discovers your models"
    echo ""
}

# --- Main ---
check_macos
detect_arch

MODE=service
case "$1" in
    --manual)    MODE=manual ;;
    --uninstall) uninstall; exit 0 ;;
    --service|"") ;;
    *) echo "Usage: $0 [--service | --manual | --uninstall]"; exit 1 ;;
esac

check_python
create_directories
install_files
create_venv
[[ "$MODE" == service ]] && install_launchagent
print_summary "$MODE"
