#!/bin/bash
#
# Deploy MPP to MACA SDK and cu-bridge
#
# Usage:
#   ./deploy_mpp.sh [options]
#
# Environment Variables:
#   MACA_PATH    Default MACA SDK path if --maca not specified
#   CUCC_PATH    Default cu-bridge path if --cu-bridge not specified
#
# Examples:
#   ./deploy_mpp.sh                                    # Use $MACA_PATH and $CUCC_PATH
#   ./deploy_mpp.sh --maca /opt/maca-3.3.0 --cu-bridge ~/cu-bridge
#

set -e

# Script and project directories
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default values (use environment variables if set)
MACA_SDK="${MACA_PATH:-}"
CU_BRIDGE="${CUCC_PATH:-}"
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
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --maca       Path to MACA SDK directory (default: \$MACA_PATH)"
            echo "  --cu-bridge  Path to cu-bridge directory (default: \$CUCC_PATH)"
            echo "  --dry-run    Preview changes without applying"
            echo "  -h, --help   Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  MACA_PATH    Default MACA SDK path if --maca not specified"
            echo "  CUCC_PATH    Default cu-bridge path if --cu-bridge not specified"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Prompt for paths if not set (via args or env vars)
if [ -z "$MACA_SDK" ]; then
    echo -n "Enter MACA SDK path (e.g., /path/to/maca-sdk/opt/maca-3.3.0): "
    read MACA_SDK
    if [ -z "$MACA_SDK" ]; then
        echo "Error: MACA SDK path is required"
        exit 1
    fi
fi

if [ -z "$CU_BRIDGE" ]; then
    echo -n "Enter cu-bridge path: "
    read CU_BRIDGE
    if [ -z "$CU_BRIDGE" ]; then
        echo "Error: cu-bridge path is required"
        exit 1
    fi
fi

# Convert to absolute paths
if [ -d "$MACA_SDK" ]; then
    MACA_SDK="$(cd "$MACA_SDK" && pwd)"
else
    echo "Error: MACA SDK path does not exist: $MACA_SDK"
    exit 1
fi

if [ -d "$CU_BRIDGE" ]; then
    CU_BRIDGE="$(cd "$CU_BRIDGE" && pwd)"
else
    echo "Error: cu-bridge path does not exist: $CU_BRIDGE"
    exit 1
fi

if [ "$DRY_RUN" = true ]; then
    echo "=== DRY RUN MODE ==="
fi

echo "========================================"
echo "  MPP Deployment Script"
echo "========================================"
echo ""
echo "Project:     $PROJECT_DIR"
echo "MACA SDK:    $MACA_SDK"
echo "cu-bridge:   $CU_BRIDGE"
echo ""

# Check directories exist
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Error: Project directory not found: $PROJECT_DIR"
    exit 1
fi

if [ ! -d "$MACA_SDK" ]; then
    echo "Error: MACA SDK not found: $MACA_SDK"
    exit 1
fi

if [ ! -d "$CU_BRIDGE" ]; then
    echo "Error: cu-bridge not found: $CU_BRIDGE"
    exit 1
fi

# ============================================
# Step 1: Generate wrapper files (if needed)
# Must run BEFORE NPP->MPP conversion
# ============================================
echo "Step 1: Generate wrapper files (if needed)"
if [ -d "$PROJECT_DIR/src/nppi" ]; then
    # Source is still in NPP naming, can generate wrapper
    if [ ! -f "$PROJECT_DIR/wrapper/npp2mpp_functions.h" ]; then
        echo "  Generating function mappings..."
        if [ "$DRY_RUN" = false ]; then
            mkdir -p "$PROJECT_DIR/wrapper/include"
            mkdir -p "$PROJECT_DIR/wrapper/user_include"
            python3 "$SCRIPT_DIR/generate_function_mappings.py"
        else
            echo "  [DRY-RUN] Would generate npp2mpp_functions.h"
        fi
    else
        echo "  Function mappings already exist"
    fi

    if [ ! -f "$PROJECT_DIR/wrapper/nppdefs_wrapper.h" ]; then
        echo "  Generating wrapper files..."
        if [ "$DRY_RUN" = false ]; then
            python3 "$SCRIPT_DIR/generate_npp_wrapper.py"
        else
            echo "  [DRY-RUN] Would generate wrapper files"
        fi
    else
        echo "  Wrapper files already exist"
    fi
elif [ -d "$PROJECT_DIR/src/mppi" ]; then
    # Source already converted, wrapper should exist
    if [ ! -f "$PROJECT_DIR/wrapper/npp2mpp_functions.h" ]; then
        echo "  Warning: Source already converted but wrapper not found!"
        echo "  Cannot generate wrapper from converted source."
        echo "  Please restore NPP source first or ensure wrapper exists."
        exit 1
    else
        echo "  Wrapper files already exist"
    fi
else
    echo "  Error: Cannot determine source naming (no src/nppi or src/mppi found)"
    exit 1
fi
echo ""

# ============================================
# Step 2: Convert NPP to MPP (if needed)
# ============================================
echo "Step 2: Check NPP to MPP conversion status"
if [ -d "$PROJECT_DIR/src/nppi" ]; then
    echo "  Source is in NPP naming, converting to MPP..."
    if [ "$DRY_RUN" = false ]; then
        cd "$PROJECT_DIR"
        python3 "$SCRIPT_DIR/convert_npp2mpp.py" --backup
        python3 "$SCRIPT_DIR/convert_npp2mpp.py" --apply
    else
        echo "  [DRY-RUN] Would convert NPP to MPP"
    fi
elif [ -d "$PROJECT_DIR/src/mppi" ]; then
    echo "  Source already in MPP naming, skip conversion"
else
    echo "  Error: Cannot determine source naming (no src/nppi or src/mppi found)"
    exit 1
fi
echo ""

# ============================================
# Step 3: Build MPP library (if not already built)
# ============================================
echo "Step 3: Check MPP library build status"
NEED_BUILD=false

# Library structure: libmpp contains all implementations, others are stubs
# Libraries: libmppc, libmppial, libmppicc, libmppidei, libmppif, libmppig, libmppim, libmppist, libmppisu, libmppitc, libmpps
MAIN_LIB="libmpp"

# Check if main library exists
if [ -f "$PROJECT_DIR/build/src/${MAIN_LIB}.a" ] || [ -f "$PROJECT_DIR/build/src/${MAIN_LIB}.so" ]; then
    # Determine which library file to check (prefer .a over .so for symbol checking)
    if [ -f "$PROJECT_DIR/build/src/${MAIN_LIB}.a" ]; then
        LIB_FILE="$PROJECT_DIR/build/src/${MAIN_LIB}.a"
    else
        LIB_FILE="$PROJECT_DIR/build/src/${MAIN_LIB}.so"
    fi

	# Check if library exports mpp symbols (not npp)
    if nm "$LIB_FILE" 2>/dev/null | grep -q " T mppGetLibVersion"; then
        echo "  MPP library already built with mpp*** symbols, skip build"
    elif nm "$LIB_FILE" 2>/dev/null | grep -q " T nppGetLibVersion"; then
        echo "  Library exists but has npp*** symbols, need rebuild"
        NEED_BUILD=true
    else
        echo "  Library exists but cannot verify symbols, need rebuild"
        NEED_BUILD=true
    fi
else
    echo "  Library not found, need build"
    NEED_BUILD=true
fi

if [ "$NEED_BUILD" = true ]; then
    if [ "$DRY_RUN" = false ]; then
        echo "  Building MPP library..."
        cd "$PROJECT_DIR"
        ./build.sh
    else
        echo "  [DRY-RUN] Would build MPP library"
    fi
else
    echo "  Using existing MPP library"
fi
echo ""

# ============================================
# Step 4: Deploy to MACA SDK (mcmpp directory)
# ============================================
echo "Step 4: Deploy MPP to MACA SDK"

# Create directories
if [ "$DRY_RUN" = false ]; then
    mkdir -p "$MACA_SDK/include/mcmpp"
else
    echo "  [DRY-RUN] Would create: $MACA_SDK/include/mcmpp"
fi

# Copy MPP headers (from wrapper/include)
echo "  Copying MPP implementation headers..."
if [ "$DRY_RUN" = false ]; then
    if [ -d "$PROJECT_DIR/wrapper/include" ]; then
        cp "$PROJECT_DIR/wrapper/include/"*.h "$MACA_SDK/include/mcmpp/"
        echo "    Copied to $MACA_SDK/include/mcmpp/:"
        ls "$PROJECT_DIR/wrapper/include/"*.h | xargs -n1 basename | sed 's/^/      /'
    else
        echo "    Error: $PROJECT_DIR/wrapper/include not found"
        exit 1
    fi
else
    echo "  [DRY-RUN] Would copy headers to $MACA_SDK/include/mcmpp/"
fi

# Copy libraries (main + stubs)
# Main library: libmpp (contains all implementations)
# Stub libraries: libmppial, libmppicc, libmppidei, libmppif, libmppig, libmppim, libmppist, libmppisu, libmppitc, libmpps
MPP_LIBS="mppc mppial mppicc mppidei mppif mppig mppim mppist mppisu mppitc mpps"

echo "  Copying MPP libraries..."
if [ "$DRY_RUN" = false ]; then
    for lib in $MPP_LIBS; do
        if [ -f "$PROJECT_DIR/build/src/lib${lib}.so" ]; then
            cp "$PROJECT_DIR/build/src/lib${lib}.so"* "$MACA_SDK/lib/"
            echo "    Copied: lib${lib}.so"
        fi
        if [ -f "$PROJECT_DIR/build/src/lib${lib}.a" ]; then
            cp "$PROJECT_DIR/build/src/lib${lib}.a" "$MACA_SDK/lib/"
            echo "    Copied: lib${lib}.a"
        fi
    done
else
    echo "  [DRY-RUN] Would copy libraries to $MACA_SDK/lib/:"
    echo "    Main: libmpp.{so,a}"
    echo "    Stubs: libmppial, libmppicc, libmppidei, libmppif, libmppig, libmppim, libmppist, libmppisu, libmppitc, libmpps"
fi
echo ""

# ============================================
# Step 5: Deploy to cu-bridge (NPP compatibility headers)
# ============================================
echo "Step 5: Deploy NPP->MPP bridge to cu-bridge"

# Create bridge/mpp directory
if [ "$DRY_RUN" = false ]; then
    mkdir -p "$CU_BRIDGE/include/bridge/mpp"
else
    echo "  [DRY-RUN] Would create: $CU_BRIDGE/include/bridge/mpp"
fi

# Copy bridge wrapper files (from wrapper/)
echo "  Copying bridge wrapper files..."
if [ "$DRY_RUN" = false ]; then
    if ls "$PROJECT_DIR/wrapper/"*_wrapper.h 1>/dev/null 2>&1; then
        cp "$PROJECT_DIR/wrapper/"*_wrapper.h "$CU_BRIDGE/include/bridge/mpp/"
        cp "$PROJECT_DIR/wrapper/npp2mpp_functions.h" "$CU_BRIDGE/include/bridge/mpp/"
        echo "    Copied to $CU_BRIDGE/include/bridge/mpp/:"
        ls "$PROJECT_DIR/wrapper/"*_wrapper.h "$PROJECT_DIR/wrapper/npp2mpp_functions.h" | xargs -n1 basename | sed 's/^/      /'
    else
        echo "    Error: Wrapper files not found in $PROJECT_DIR/wrapper/"
        exit 1
    fi
else
    echo "  [DRY-RUN] Would copy bridge files to $CU_BRIDGE/include/bridge/mpp/"
fi

# Copy top-level NPP headers (from wrapper/user_include)
echo "  Copying top-level NPP headers..."
if [ "$DRY_RUN" = false ]; then
    if [ -d "$PROJECT_DIR/wrapper/user_include" ]; then
        cp "$PROJECT_DIR/wrapper/user_include/"*.h "$CU_BRIDGE/include/"
        echo "    Copied to $CU_BRIDGE/include/:"
        ls "$PROJECT_DIR/wrapper/user_include/"*.h | xargs -n1 basename | sed 's/^/      /'
    else
        echo "    Error: $PROJECT_DIR/wrapper/user_include not found"
        exit 1
    fi
else
    echo "  [DRY-RUN] Would copy NPP headers to $CU_BRIDGE/include/"
fi

echo ""

# ============================================
# Summary
# ============================================
echo "========================================"
echo "  Deployment Complete"
echo "========================================"
echo ""
echo "MACA SDK:"
echo "  Headers: $MACA_SDK/include/mcmpp/"
echo "  Libraries: $MACA_SDK/lib/"
echo "    Main:  libmpp.{so,a} (contains all implementations)"
echo "    Stubs: libmppial, libmppicc, libmppidei, libmppif, libmppig, libmppim, libmppist, libmppisu, libmppitc, libmpps"
echo ""
echo "cu-bridge:"
echo "  Top-level: $CU_BRIDGE/include/npp*.h"
echo "  Bridge:    $CU_BRIDGE/include/bridge/mpp/*_wrapper.h"
echo ""
echo "User code example:"
echo "  #include <npp.h>"
echo "  nppiMalloc_8u_C1(width, height, &step);"
echo "  nppiSet_8u_C1R(value, dst, step, roi);"
echo ""
echo "Compile command (link to appropriate stub library):"
echo "  cucc -I\$CU_BRIDGE/include -I\$MACA_SDK/include -L\$MACA_SDK/lib -lmppial user_code.cu"
echo ""
echo "Or link to main library directly:"
echo "  cucc -I\$CU_BRIDGE/include -I\$MACA_SDK/include -L\$MACA_SDK/lib -lmppc user_code.cu"
echo ""
