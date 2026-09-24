#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2026 Andrew Wyatt (Fewtarius)

#
# Build stable-diffusion.cpp with Vulkan support.
# Provides Vulkan-based SD backend for AMD GPUs (universal support).
#
# Usage:
#   INSTALL_PREFIX=/opt/alice bash scripts/build_sdcpp.sh
#
# Environment:
#   INSTALL_PREFIX  - Where to install the build (default: /usr/local)
#   SDCPP_BUILD_DIR - Where to build (default: ${INSTALL_PREFIX}/sd.cpp)
#   NO_CLEAN        - If set, skip cleaning build dir on reconfigure

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local}"
SDCPP_BUILD_DIR="${SDCPP_BUILD_DIR:-${SCRIPT_DIR}/../build/sd.cpp}"
SDCPP_REPO="https://github.com/SyntheticAutonomicMind/stable-diffusion.cpp"

echo "=== Building stable-diffusion.cpp with Vulkan ==="
echo "  Install prefix: ${INSTALL_PREFIX}"
echo "  Build dir:      ${SDCPP_BUILD_DIR}"

# ---------------------------------------------------------------------------
# Package manager detection and dependency installation
# ---------------------------------------------------------------------------

detect_package_manager() {
    if command -v pacman >/dev/null 2>&1; then
        echo "pacman"
    elif command -v apt-get >/dev/null 2>&1; then
        echo "apt"
    elif command -v dnf >/dev/null 2>&1; then
        echo "dnf"
    elif command -v yum >/dev/null 2>&1; then
        echo "yum"
    elif command -v zypper >/dev/null 2>&1; then
        echo "zypper"
    else
        echo "none"
    fi
}

PKG_MGR=$(detect_package_manager)

install_deps() {
    local missing=("$@")
    if [ ${#missing[@]} -eq 0 ]; then
        return 0
    fi

    echo "Attempting to install missing dependencies: ${missing[*]}"

    case "$PKG_MGR" in
        pacman)
            sudo pacman -Sy --noconfirm --needed "${missing[@]}"
            ;;
        apt)
            sudo apt-get update -qq
            sudo apt-get install -y --no-install-recommends "${missing[@]}"
            ;;
        dnf)
            sudo dnf install -y "${missing[@]}"
            ;;
        yum)
            sudo yum install -y "${missing[@]}"
            ;;
        zypper)
            sudo zypper install -y "${missing[@]}"
            ;;
        none)
            echo "ERROR: Cannot auto-install dependencies."
            echo "Missing: ${missing[*]}"
            echo "Please install them manually for your system."
            exit 1
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

check_deps() {
    local missing=()

    command -v cmake >/dev/null 2>&1 || missing+=("cmake")
    command -v git >/dev/null 2>&1 || missing+=("git")
    command -v pkg-config >/dev/null 2>&1 || missing+=("pkgconfig")

    if [ ${#missing[@]} -gt 0 ]; then
        # Map to package manager package names
        case "$PKG_MGR" in
            pacman) install_deps "${missing[@]}" ;;
            apt)    # cmake, git, pkgconf
                    install_deps "cmake" "git" "pkg-config" ;;
            dnf|yum|zypper)
                    install_deps "${missing[@]}" ;;
        esac
        # Re-check after install
        missing=()
        command -v cmake >/dev/null 2>&1 || missing+=("cmake")
        command -v git >/dev/null 2>&1 || missing+=("git")
        command -v pkg-config >/dev/null 2>&1 || missing+=("pkg-config")
        if [ ${#missing[@]} -gt 0 ]; then
            echo "ERROR: Still missing: ${missing[*]}"
            exit 1
        fi
    fi

    # Check for Vulkan development files
    if ! pkg-config --exists vulkan 2>/dev/null; then
        echo "Vulkan development files not found, attempting to install..."
        case "$PKG_MGR" in
            pacman) install_deps "vulkan-headers" "vulkan-icd-loader" "shaderc" ;;
            apt)    install_deps "libvulkan-dev" "vulkan-headers" "shaderc" ;;
            dnf)    install_deps "vulkan-loader" "vulkan-headers" "shaderc" ;;
            yum)    install_deps "vulkan-loader" "vulkan-headers" "shaderc" ;;
            zypper) install_deps "vulkan-loader" "vulkan-headers" "shaderc" ;;
        esac
        if ! pkg-config --exists vulkan 2>/dev/null; then
            echo "ERROR: Vulkan development files still not found after install attempt."
            echo "Please install vulkan-headers, vulkan-icd-loader, and shaderc for your platform."
            exit 1
        fi
    fi

    echo "[OK] All dependencies present"
}

# ---------------------------------------------------------------------------
# Clone / update repository
# ---------------------------------------------------------------------------

clone_repo() {
    if [ -d "$SDCPP_BUILD_DIR/.git" ]; then
        echo "Build directory exists, pulling latest..."
        cd "$SDCPP_BUILD_DIR"
        git pull --ff-only
        git submodule update --init --recursive
    else
        echo "Cloning stable-diffusion.cpp from ${SDCPP_REPO}..."
        mkdir -p "$(dirname "$SDCPP_BUILD_DIR")"
        git clone --recursive "$SDCPP_REPO" "$SDCPP_BUILD_DIR"
    fi
}

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

build() {
    echo "Building with Vulkan backend..."
    cd "$SDCPP_BUILD_DIR"

    if [ -z "$NO_CLEAN" ] && [ -d build ]; then
        rm -rf build
    fi
    mkdir -p build
    cd build

    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DSD_VULKAN=ON \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"

    cmake --build . --config Release -j$(nproc)

    echo "[OK] Build complete"
}

# ---------------------------------------------------------------------------
# Install binary to INSTALL_PREFIX/bin/sd-cli
# ---------------------------------------------------------------------------

install_binary() {
    local src_bin="$SDCPP_BUILD_DIR/build/bin/sd-cli"
    local dest_bin="${INSTALL_PREFIX}/bin/sd-cli"

    if [ ! -f "$src_bin" ]; then
        echo "ERROR: sd-cli binary not found at $src_bin"
        exit 1
    fi

    echo "Installing sd-cli to ${dest_bin}..."

    if [ "$EUID" -eq 0 ]; then
        install -Dm755 "$src_bin" "$dest_bin"
    elif [ "$INSTALL_PREFIX" = "/usr/local" ] || [ "$INSTALL_PREFIX" = "/usr" ]; then
        sudo install -Dm755 "$src_bin" "$dest_bin"
    else
        install -Dm755 "$src_bin" "$dest_bin"
    fi

    echo "[OK] Installed to ${dest_bin}"
}

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

verify() {
    local bin="${INSTALL_PREFIX}/bin/sd-cli"

    if [ ! -x "$bin" ]; then
        echo "ERROR: sd-cli not found or not executable at ${bin}"
        exit 1
    fi

    echo "Verifying installation..."
    "$bin" --version 2>&1 | head -1 || echo "  (version output not available)"

    echo ""
    echo "=== Installation complete ==="
    echo "Binary: ${bin}"
    echo "Size:   $(du -h "${bin}" | cut -f1)"
    echo ""
    echo "To test:"
    echo "  ${bin} -m model.safetensors -p 'a red apple' -o test.png"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    check_deps
    clone_repo
    build
    install_binary
    verify
}

main "$@"
