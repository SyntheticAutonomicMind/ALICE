#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2026 Andrew Wyatt (Fewtarius)

#
# Build stable-diffusion.cpp with Vulkan support
# Provides Vulkan-based SD backend for AMD GPUs (universal support)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../build/sd.cpp"
INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local}"

echo "=== Building stable-diffusion.cpp with Vulkan ==="

# Check for required dependencies
check_deps() {
    local missing=()
    
    command -v cmake >/dev/null 2>&1 || missing+=("cmake")
    command -v git >/dev/null 2>&1 || missing+=("git")
    
    if [ ${#missing[@]} -gt 0 ]; then
        echo "ERROR: Missing dependencies: ${missing[*]}"
        echo "Install with: sudo pacman -S ${missing[*]}"
        exit 1
    fi
    
    # Check for Vulkan development files
    if ! pkg-config --exists vulkan; then
        echo "ERROR: Vulkan development files not found"
        echo "Install with: sudo pacman -S vulkan-headers vulkan-icd-loader shaderc"
        exit 1
    fi
    
    echo "[OK] All dependencies present"
}

# Clone repository
clone_repo() {
    if [ -d "$BUILD_DIR" ]; then
        echo "Build directory exists, pulling latest..."
        cd "$BUILD_DIR"
        git pull
        git submodule update --init --recursive
    else
        echo "Cloning stable-diffusion.cpp..."
        mkdir -p "$(dirname "$BUILD_DIR")"
        git clone --recursive https://github.com/SyntheticAutonomicMind/stable-diffusion.cpp "$BUILD_DIR"
    fi
}

# Build with Vulkan backend
build() {
    echo "Building with Vulkan backend..."
    cd "$BUILD_DIR"
    
    mkdir -p build
    cd build
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DSD_VULKAN=ON \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"
    
    cmake --build . --config Release -j$(nproc)
    
    echo "[OK] Build complete"
}

# Install binary
install_binary() {
    local src_bin="$BUILD_DIR/build/bin/sd-cli"
    local dest_bin="$INSTALL_PREFIX/bin/sd-cli"
    
    if [ ! -f "$src_bin" ]; then
        echo "ERROR: sd-cli binary not found at $src_bin"
        exit 1
    fi
    
    echo "Installing sd-cli to $dest_bin..."
    
    if [ "$INSTALL_PREFIX" = "/usr/local" ] || [ "$INSTALL_PREFIX" = "/usr" ]; then
        sudo install -Dm755 "$src_bin" "$dest_bin"
    else
        install -Dm755 "$src_bin" "$dest_bin"
    fi
    
    echo "[OK] Installed to $dest_bin"
}

# Verify installation
verify() {
    local bin="$INSTALL_PREFIX/bin/sd-cli"
    
    if [ ! -x "$bin" ]; then
        echo "ERROR: sd-cli not found or not executable"
        exit 1
    fi
    
    echo "Verifying installation..."
    "$bin" --version 2>&1 | head -1 || echo "  (version output not available)"
    
    echo ""
    echo "=== Installation complete ==="
    echo "Binary: $bin"
    echo "Size: $(du -h "$bin" | cut -f1)"
    echo ""
    echo "To test:"
    echo "  $bin -m model.safetensors -p 'a red apple' -o test.png"
}

# Main
main() {
    check_deps
    clone_repo
    build
    install_binary
    verify
}

main "$@"
