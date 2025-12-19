#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2025 Andrew Wyatt (Fewtarius)
#
# AMD GPU Detection Script for ALICE
# Detects AMD GPUs and outputs the required ROCm environment variables
#
# Usage: source detect_amd_gpu.sh
#        Or: eval $(./detect_amd_gpu.sh)
#

# AMD vendor ID
AMD_VENDOR="1002"

# Known AMD GPU mappings: PCI_ID -> GFX_VERSION, HSA_OVERRIDE
# Format: "PCI_ID:gfx_arch:hsa_override"
declare -A AMD_GPU_MAP=(
    # RDNA3 - Phoenix (Ryzen 7000/8000 series APUs)
    # Note: Phoenix is gfx1103 but works with gfx1100 kernels (11.0.0)
    # The 11.0.0 override provides better kernel compatibility than 11.0.2
    ["15bf"]="gfx1103:11.0.0"    # Phoenix1 (Ryzen 7 8840U, Ryzen 9 8945HS, etc.) - tested working
    ["15c8"]="gfx1103:11.0.0"    # Phoenix2 - use same override as Phoenix1
    ["1900"]="gfx1103:11.0.0"    # Phoenix (variant)
    
    # RDNA3 - Navi 31/32/33 (RX 7000 series)
    ["744c"]="gfx1100:11.0.0"    # Navi 31 (RX 7900 XTX/XT)
    ["7480"]="gfx1100:11.0.0"    # Navi 31 (variant)
    ["745e"]="gfx1101:11.0.1"    # Navi 32 (RX 7800/7700)
    ["7470"]="gfx1102:11.0.0"    # Navi 33 (RX 7600) - use 11.0.0 for compatibility
    
    # RDNA3 - Strix Point / Strix Halo (Ryzen AI 300 series)
    ["150e"]="gfx1103:11.0.0"    # Strix Point (Ryzen AI 9 HX 370, etc.)
    ["1502"]="gfx1103:11.0.0"    # Strix (variant)
    
    # RDNA2 - Steam Deck / Van Gogh
    ["163f"]="gfx1033:10.3.3"    # Van Gogh (Steam Deck original)
    
    # RDNA2 - Navi 21/22/23/24 (RX 6000 series)  
    ["73bf"]="gfx1030:10.3.0"    # Navi 21 (RX 6800/6900)
    ["73df"]="gfx1031:10.3.1"    # Navi 22 (RX 6700)
    ["73ff"]="gfx1032:10.3.2"    # Navi 23 (RX 6600)
    ["743f"]="gfx1034:10.3.4"    # Navi 24 (RX 6500/6400)
    
    # RDNA2 - Rembrandt (Ryzen 6000 series APUs)
    ["1681"]="gfx1035:10.3.5"    # Rembrandt (Ryzen 6000)
    
    # RDNA1 - Navi 10/12/14 (RX 5000 series)
    ["7310"]="gfx1010:10.1.0"    # Navi 10 (RX 5700)
    ["7312"]="gfx1010:10.1.0"    # Navi 10 (variant)
    ["7340"]="gfx1011:10.1.1"    # Navi 14 (RX 5500)
    
    # Vega - Cezanne/Renoir (Ryzen 5000/4000 APUs)
    ["1638"]="gfx90c:9.0.12"     # Cezanne (Ryzen 5000 APU) - ROCm limited support
    ["15d8"]="gfx90c:9.0.12"     # Renoir (Ryzen 4000 APU) - ROCm limited support
)

# Function to detect AMD GPU and output environment variables
detect_amd_gpu() {
    local gpu_found=false
    local pci_id=""
    local gfx_arch=""
    local hsa_version=""
    local gpu_name=""
    
    # Check for AMD GPU via sysfs
    for card in /sys/class/drm/card*; do
        if [[ -f "$card/device/vendor" ]]; then
            local vendor=$(cat "$card/device/vendor" 2>/dev/null | tr '[:upper:]' '[:lower:]' | sed 's/0x//')
            if [[ "$vendor" == "$AMD_VENDOR" ]]; then
                pci_id=$(cat "$card/device/device" 2>/dev/null | tr '[:upper:]' '[:lower:]' | sed 's/0x//')
                gpu_found=true
                break
            fi
        fi
    done
    
    if [[ "$gpu_found" == "false" ]]; then
        # Fallback to lspci
        local lspci_output=$(lspci -nn 2>/dev/null | grep -i "vga.*amd\|vga.*ati" | head -1)
        if [[ -n "$lspci_output" ]]; then
            # Extract PCI ID from format [1002:XXXX]
            pci_id=$(echo "$lspci_output" | grep -oP '\[1002:\K[0-9a-f]+' | tr '[:upper:]' '[:lower:]')
            gpu_name=$(echo "$lspci_output" | sed 's/.*: //' | sed 's/ \[.*//')
            gpu_found=true
        fi
    fi
    
    if [[ "$gpu_found" == "false" ]]; then
        echo "# No AMD GPU detected"
        return 1
    fi
    
    # Look up GPU in our map
    local mapping="${AMD_GPU_MAP[$pci_id]}"
    
    if [[ -n "$mapping" ]]; then
        gfx_arch=$(echo "$mapping" | cut -d: -f1)
        hsa_version=$(echo "$mapping" | cut -d: -f2)
        
        echo "# AMD GPU detected: PCI ID 0x$pci_id"
        [[ -n "$gpu_name" ]] && echo "# GPU Name: $gpu_name"
        echo "# GFX Architecture: $gfx_arch"
        echo "export PYTORCH_ROCM_ARCH=\"$gfx_arch\""
        echo "export HSA_OVERRIDE_GFX_VERSION=\"$hsa_version\""
        
        # Enable experimental features for newer GPUs
        case "$gfx_arch" in
            gfx110*|gfx103*)
                echo "export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1"
                ;;
        esac
        
        return 0
    else
        echo "# AMD GPU detected but not in known list: PCI ID 0x$pci_id"
        echo "# Please add this GPU to the detection script"
        echo "# You may need to manually set:"
        echo "# export PYTORCH_ROCM_ARCH=\"gfxXXXX\""
        echo "# export HSA_OVERRIDE_GFX_VERSION=\"XX.X.X\""
        return 2
    fi
}

# Function to generate systemd environment file
generate_env_file() {
    local output_file="${1:-/tmp/alice-rocm.env}"
    
    echo "# Auto-generated ROCm environment for ALICE" > "$output_file"
    echo "# Generated: $(date)" >> "$output_file"
    
    detect_amd_gpu | grep "^export" | sed 's/export //' >> "$output_file"
    
    if [[ -s "$output_file" ]]; then
        echo "Environment file written to: $output_file"
        return 0
    else
        echo "Failed to generate environment file"
        return 1
    fi
}

# Function to check if ROCm is properly configured
check_rocm_setup() {
    echo "=== ROCm Configuration Check ==="
    
    # Check for AMD GPU
    if lspci 2>/dev/null | grep -qi "vga.*amd\|vga.*ati"; then
        echo "[OK] AMD GPU detected"
    else
        echo "[WARN] No AMD GPU detected via lspci"
    fi
    
    # Check environment variables
    if [[ -n "$PYTORCH_ROCM_ARCH" ]]; then
        echo "[OK] PYTORCH_ROCM_ARCH=$PYTORCH_ROCM_ARCH"
    else
        echo "[MISSING] PYTORCH_ROCM_ARCH not set"
    fi
    
    if [[ -n "$HSA_OVERRIDE_GFX_VERSION" ]]; then
        echo "[OK] HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"
    else
        echo "[MISSING] HSA_OVERRIDE_GFX_VERSION not set"
    fi
    
    # Check for ROCm libraries
    if [[ -d "/opt/rocm" ]]; then
        echo "[OK] ROCm installation found at /opt/rocm"
    else
        echo "[INFO] No system ROCm found (may be using PyTorch bundled ROCm)"
    fi
    
    # Check amdgpu driver
    if lsmod 2>/dev/null | grep -q "amdgpu"; then
        echo "[OK] amdgpu kernel module loaded"
    else
        echo "[WARN] amdgpu kernel module not loaded"
    fi
}

# Main execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Script is being executed directly
    case "${1:-detect}" in
        detect)
            detect_amd_gpu
            ;;
        env)
            generate_env_file "${2:-/tmp/alice-rocm.env}"
            ;;
        check)
            check_rocm_setup
            ;;
        *)
            echo "Usage: $0 [detect|env [file]|check]"
            echo ""
            echo "Commands:"
            echo "  detect  - Detect AMD GPU and output environment variables"
            echo "  env     - Generate systemd environment file"
            echo "  check   - Check ROCm configuration"
            exit 1
            ;;
    esac
fi
