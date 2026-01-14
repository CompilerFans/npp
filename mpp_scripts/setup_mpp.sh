#!/bin/bash
#
# MPP One-Click Setup Script
#
# This script automates the complete MPP setup process:
# 1. Create wrapper directory structure
# 2. Generate function mappings (from original NPP API)
# 3. Generate wrapper files (from original NPP API)
# 4. Convert NPP source code to MPP naming
# 5. Convert CUDA Runtime API to MACA Runtime API
# 6. Copy MPP headers to wrapper/include
# 7. Build MPP library
# 8. Deploy to MACA SDK and cu-bridge (optional)
#
# IMPORTANT: Steps 1-3 must run BEFORE step 4, because wrapper generation
# needs to scan the original NPP API headers to create npp->mpp mappings.
#
# Usage:
#   ./setup_mpp.sh [options]
#
# Options:
#   --maca            Path to MACA SDK directory
#   --cu-bridge       Path to cu-bridge directory
#   --skip-convert    Skip NPP to MPP conversion
#   --skip-build      Skip library build
#   --skip-deploy     Skip deployment
#   --dry-run         Preview changes without applying
#   -h, --help        Show this help message
#
# Environment Variables (used by deploy step):
#   MACA_PATH     If set, will prompt to use as default MACA SDK path
#   CUCC_PATH     If set, will prompt to use as default cu-bridge path
#
# ============================================================================
# Individual Scripts (can be used standalone)
# ============================================================================
#
# 1. convert_npp2mpp.py - Convert NPP naming to MPP naming (Generic Rules)
#    Uses generic prefix replacement rules (npp*->mpp*, Npp*->Mpp*, NPP_*->MPP_*)
#    Scope: src/, test/, cmake/, examples/, API/ directories and root files
#    Usage:
#      python3 mpp_scripts/convert_npp2mpp.py --preview  # Preview changes
#      python3 mpp_scripts/convert_npp2mpp.py --backup   # Create backup in .npp_backup/
#      python3 mpp_scripts/convert_npp2mpp.py --apply    # Apply conversion
#      python3 mpp_scripts/convert_npp2mpp.py --restore  # Restore from backup
#
# 2. generate_function_mappings.py - Generate function name mappings
#    Scans API headers and generates npp2mpp_functions.h with #define mappings.
#    IMPORTANT: Must run BEFORE convert_npp2mpp.py (needs original NPP headers)
#    Usage:
#      python3 mpp_scripts/generate_function_mappings.py
#    Output: wrapper/npp2mpp_functions.h
#
# 3. generate_npp_wrapper.py - Generate NPP compatibility wrapper files
#    Creates type/enum/constant mappings and user-facing NPP headers.
#    Usage:
#      python3 mpp_scripts/generate_npp_wrapper.py
#    Output:
#      wrapper/*_wrapper.h        - Bridge headers (NPP types -> MPP types)
#      wrapper/user_include/*.h   - User-facing NPP headers
#
# 4. deploy_mpp.sh - Deploy MPP to MACA SDK and cu-bridge
#    Copies built libraries (libmppc + stubs) and headers to target directories.
#    Usage:
#      ./mpp_scripts/deploy_mpp.sh --maca <maca_sdk_path> --cu-bridge <cu_bridge_path>
#      ./mpp_scripts/deploy_mpp.sh --maca /opt/maca-3.3.0 --cu-bridge ~/cu-bridge --dry-run
#
# 5. convert_cuda2maca.py - Convert CUDA Runtime API to MACA Runtime API (Generic Rules)
#    Uses generic prefix replacement (cuda*->mc*) with exceptions for special cases.
#    Scope: src/, test/, examples/, API/ directories (wrapper/ excluded for compatibility)
#    Usage:
#      python3 mpp_scripts/convert_cuda2maca.py --check    # Preview changes
#      python3 mpp_scripts/convert_cuda2maca.py --backup   # Create backup in .cuda_backup/
#      python3 mpp_scripts/convert_cuda2maca.py --apply    # Apply conversion
#      python3 mpp_scripts/convert_cuda2maca.py --restore  # Restore from backup
#
# ============================================================================
#

set -e

# Script location
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
MACA_SDK=""
CU_BRIDGE=""
SKIP_CONVERT=false
SKIP_BUILD=false
SKIP_DEPLOY=false
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --maca)
            MACA_SDK="$2"
            shift 2
            ;;
        --cu-bridge)
            CU_BRIDGE="$2"
            shift 2
            ;;
        --skip-convert)
            SKIP_CONVERT=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-deploy)
            SKIP_DEPLOY=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "MPP One-Click Setup Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --maca            Path to MACA SDK directory"
            echo "  --cu-bridge       Path to cu-bridge directory"
            echo "  --skip-convert    Skip NPP to MPP conversion"
            echo "  --skip-build      Skip library build"
            echo "  --skip-deploy     Skip deployment"
            echo "  --dry-run         Preview changes without applying"
            echo "  -h, --help        Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  MACA_PATH    If set, will prompt to use as default MACA SDK path"
            echo "  CUCC_PATH    If set, will prompt to use as default cu-bridge path"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "  MPP One-Click Setup"
echo "========================================"
echo ""
echo "Project:     $PROJECT_DIR"
if [ "$DRY_RUN" = true ]; then
    echo "Mode:        DRY RUN"
fi
echo ""

# ============================================
# Step 1: Create wrapper directory
# ============================================
echo "Step 1: Create wrapper directory"
if [ "$DRY_RUN" = false ]; then
    mkdir -p "$PROJECT_DIR/wrapper/include"
    mkdir -p "$PROJECT_DIR/wrapper/user_include"
    echo "  Created: $PROJECT_DIR/wrapper/"
else
    echo "  [DRY-RUN] Would create: $PROJECT_DIR/wrapper/"
fi
echo ""

# ============================================
# Step 2: Generate function mappings
# (Must run BEFORE NPP->MPP conversion)
# ============================================
echo "Step 2: Generate function mappings (from original NPP API)"
if [ "$SKIP_CONVERT" = true ]; then
    echo "  Skipped (--skip-convert)"
elif [ -d "$PROJECT_DIR/src/mppi" ]; then
    echo "  Source already converted to MPP, skipping (wrapper should already exist)"
else
    if [ "$DRY_RUN" = false ]; then
        python3 "$SCRIPT_DIR/generate_function_mappings.py"
    else
        echo "  [DRY-RUN] Would generate npp2mpp_functions.h"
    fi
fi
echo ""

# ============================================
# Step 3: Generate wrapper files
# (Must run BEFORE NPP->MPP conversion)
# ============================================
echo "Step 3: Generate wrapper files (from original NPP API)"
if [ "$SKIP_CONVERT" = true ]; then
    echo "  Skipped (--skip-convert)"
elif [ -d "$PROJECT_DIR/src/mppi" ]; then
    echo "  Source already converted to MPP, skipping (wrapper should already exist)"
else
    if [ "$DRY_RUN" = false ]; then
        python3 "$SCRIPT_DIR/generate_npp_wrapper.py"
    else
        echo "  [DRY-RUN] Would generate wrapper files"
    fi
fi
echo ""

# ============================================
# Step 4: Convert NPP to MPP
# ============================================
echo "Step 4: Convert NPP to MPP naming"
if [ "$SKIP_CONVERT" = true ]; then
    echo "  Skipped (--skip-convert)"
elif [ -d "$PROJECT_DIR/src/mppi" ]; then
    echo "  Source already in MPP naming, skip conversion"
elif [ -d "$PROJECT_DIR/src/nppi" ]; then
    if [ "$DRY_RUN" = false ]; then
        echo "  Converting NPP to MPP..."
        cd "$PROJECT_DIR"
        python3 "$SCRIPT_DIR/convert_npp2mpp.py" --backup
        python3 "$SCRIPT_DIR/convert_npp2mpp.py" --apply
        echo "  Conversion complete"
    else
        echo "  [DRY-RUN] Would convert NPP to MPP"
    fi
else
    echo "  Error: Cannot determine source naming"
    exit 1
fi
echo ""

# ============================================
# Step 5: Convert CUDA to MACA Runtime API
# ============================================
echo "Step 5: Convert CUDA to MACA Runtime API"
if [ "$SKIP_CONVERT" = true ]; then
    echo "  Skipped (--skip-convert)"
else
    # Check if already converted (look for mc_runtime.h includes)
    if grep -rq "#include.*mc_runtime.h" "$PROJECT_DIR/src/" 2>/dev/null; then
        echo "  Source already using MACA Runtime API, skip conversion"
    elif grep -rq "#include.*cuda_runtime.h" "$PROJECT_DIR/src/" 2>/dev/null; then
        if [ "$DRY_RUN" = false ]; then
            echo "  Converting CUDA to MACA..."
            cd "$PROJECT_DIR"
            python3 "$SCRIPT_DIR/convert_cuda2maca.py" --backup
            python3 "$SCRIPT_DIR/convert_cuda2maca.py" --apply
            echo "  Conversion complete"
        else
            echo "  [DRY-RUN] Would convert CUDA to MACA"
        fi
    else
        echo "  No CUDA Runtime API usage found, skip conversion"
    fi
fi
echo ""

# ============================================
# Step 6: Copy MPP headers to wrapper/include
# ============================================
echo "Step 6: Copy MPP headers to wrapper/include"
if [ "$DRY_RUN" = false ]; then
    # Copy from src/include if it exists
    if [ -d "$PROJECT_DIR/src/include" ]; then
        cp "$PROJECT_DIR/src/include/"*.h "$PROJECT_DIR/wrapper/include/" 2>/dev/null || true
        echo "  Copied MPP headers from src/include/"
    fi

    # Generate main header files if they don't exist
    for module in mpp mppdefs mppcore mppi mpps; do
        if [ ! -f "$PROJECT_DIR/wrapper/include/${module}.h" ]; then
            # Create placeholder or copy from API
            if [ -f "$PROJECT_DIR/API/${module/mpp/npp}.h" ]; then
                # Use sed to convert npp to mpp in header
                sed 's/npp/mpp/g; s/Npp/Mpp/g; s/NPP/MPP/g' \
                    "$PROJECT_DIR/API/${module/mpp/npp}.h" > "$PROJECT_DIR/wrapper/include/${module}.h"
            fi
        fi
    done
    echo "  MPP headers ready in wrapper/include/"
else
    echo "  [DRY-RUN] Would copy MPP headers"
fi
echo ""

# ============================================
# Step 7: Build MPP library
# ============================================
echo "Step 7: Build MPP library"
if [ "$SKIP_BUILD" = true ]; then
    echo "  Skipped (--skip-build)"
elif [ "$DRY_RUN" = false ]; then
    cd "$PROJECT_DIR"
    if [ -f "./build.sh" ]; then
        ./build.sh --shared --maca --without-cu-bridge
        echo "  Build complete"
    else
        echo "  Warning: build.sh not found, skipping build"
    fi
else
    echo "  [DRY-RUN] Would build MPP library"
fi
echo ""

# ============================================
# Step 8: Deploy to MACA SDK and cu-bridge
# ============================================
echo "Step 8: Deploy to MACA SDK and cu-bridge"
if [ "$SKIP_DEPLOY" = true ]; then
    echo "  Skipped (--skip-deploy)"
else
    # Build deploy command arguments
    DEPLOY_ARGS=""
    if [ -n "$MACA_SDK" ]; then
        DEPLOY_ARGS="$DEPLOY_ARGS --maca $MACA_SDK"
    fi
    if [ -n "$CU_BRIDGE" ]; then
        DEPLOY_ARGS="$DEPLOY_ARGS --cu-bridge $CU_BRIDGE"
    fi
    if [ "$DRY_RUN" = true ]; then
        DEPLOY_ARGS="$DEPLOY_ARGS --dry-run"
    fi

    # Call deploy script (it handles env vars and prompts)
    bash "$SCRIPT_DIR/deploy_mpp.sh" $DEPLOY_ARGS
fi
echo ""

# ============================================
# Summary
# ============================================
echo "========================================"
echo "  Setup Complete"
echo "========================================"
echo ""
echo "Project structure:"
echo "  $PROJECT_DIR/"
echo "  ├── src/mppi/                    # MPP source code"
echo "  ├── src/mpps/"
echo "  ├── wrapper/"
echo "  │   ├── include/                 # MPP headers"
echo "  │   ├── user_include/            # NPP compatibility headers"
echo "  │   ├── *_wrapper.h              # NPP->MPP bridge"
echo "  │   └── npp2mpp_functions.h      # Function mappings"
echo "  └── build/src/"
echo "      ├── libmppc.{so,a}           # Main library (all implementations)"
echo "      └── libmpp*.{so,a}           # Stub libraries (link to libmppc)"
echo ""

if [ "$SKIP_DEPLOY" = false ] && [ -n "$MACA_SDK" ] && [ -n "$CU_BRIDGE" ]; then
    echo "Deployed to:"
    echo "  MACA SDK:  $MACA_SDK/include/mcmpp/"
    echo "  Libraries: $MACA_SDK/lib/libmppc.{so,a} + stubs"
    echo "  cu-bridge: $CU_BRIDGE/include/"
    echo ""
    echo "User code example:"
    echo "  #include <npp.h>"
    echo "  nppiMalloc_8u_C1(width, height, &step);"
    echo ""
    echo "Compile command:"
    echo "  cucc -I\$CU_BRIDGE/include -I\$MACA_SDK/include -L\$MACA_SDK/lib -lmppc user_code.cu"
    echo ""
else
    echo "To deploy later, run:"
    echo "  ./mpp_scripts/deploy_mpp.sh --maca <maca_sdk_dir> --cu-bridge <cu_bridge_dir>"
    echo ""
fi
