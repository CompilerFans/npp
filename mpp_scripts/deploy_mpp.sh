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
echo "MPP Deployment Script"
echo "========================================"
echo "Project:   $PROJECT_DIR"
echo "MACA SDK:  $MACA_SDK"
echo "cu-bridge: $CU_BRIDGE"
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
# Pre-checks (conversion/build already done)
# ============================================
echo "Pre-checks"
if [ ! -d "$PROJECT_DIR/src/mppi" ]; then
    echo "  Error: Source not converted to MPP. Run convert_npp2mpp.py --apply first."
    exit 1
fi

if grep -rq "#include.*cuda_runtime.h" "$PROJECT_DIR/src/" 2>/dev/null; then
    echo "  Error: CUDA runtime includes still present. Run convert_cuda2maca.py --apply (or convert_npp2mpp.py --apply) first."
    exit 1
fi

if [ ! -d "$PROJECT_DIR/wrapper/mpp" ] || \
   [ ! -d "$PROJECT_DIR/wrapper/user_include" ]; then
    echo "  Error: Wrapper files missing. Run generate_function_mappings.py and generate_npp_wrapper.py first."
    exit 1
fi

INSTALL_DIR="$PROJECT_DIR/build/install"
if [ ! -d "$INSTALL_DIR" ]; then
    echo "  Error: Install directory not found: $INSTALL_DIR"
    echo "  Run ./build.sh first."
    exit 1
fi

INCLUDE_SRC="$INSTALL_DIR/include/mcmpp"
if [ ! -d "$INCLUDE_SRC" ]; then
    echo "  Error: Include directory not found: $INCLUDE_SRC"
    exit 1
fi

if [ ! -f "$INSTALL_DIR/lib/libmpp.so" ] && \
   [ ! -f "$INSTALL_DIR/lib/libmpp_static.a" ] && \
   [ ! -f "$INSTALL_DIR/lib/libmppc.a" ] && \
   [ ! -f "$INSTALL_DIR/lib/libmppc.so" ]; then
    echo "  Error: MPP libraries not found in $INSTALL_DIR/lib."
    echo "  Run ./build.sh first."
    exit 1
fi
echo ""

# ============================================
# Step 1: Deploy to MACA SDK
# ============================================
echo "Step 1: Deploy MPP to MACA SDK"

if [ "$DRY_RUN" = false ]; then
    if [ ! -d "$MACA_SDK/include" ]; then
        echo "  Error: MACA SDK include directory not found: $MACA_SDK/include"
        exit 1
    fi
    cp -r "$INCLUDE_SRC" "$MACA_SDK/include/"
    echo "  Copied headers to $MACA_SDK/include/$(basename "$INCLUDE_SRC"):"
    ls "$INCLUDE_SRC" | sed 's/^/    /'

    if [ ! -d "$MACA_SDK/lib" ]; then
        echo "  Error: MACA SDK lib directory not found: $MACA_SDK/lib"
        exit 1
    fi
    cp "$INSTALL_DIR/lib/"libmpp*.so* "$MACA_SDK/lib/" 2>/dev/null || true
    cp "$INSTALL_DIR/lib/"libmpp*.a "$MACA_SDK/lib/" 2>/dev/null || true
    echo "  Copied libraries to $MACA_SDK/lib/:"
    ls "$INSTALL_DIR/lib"/libmpp*.so* "$INSTALL_DIR/lib"/libmpp*.a 2>/dev/null | xargs -n1 basename | sed 's/^/    /'
else
    echo "  [DRY-RUN] Would copy headers from $INCLUDE_SRC to $MACA_SDK/include/"
    echo "  [DRY-RUN] Would copy libs from $INSTALL_DIR/lib to $MACA_SDK/lib/"
fi
echo ""

# ============================================
# Step 2: Deploy to cu-bridge (NPP compatibility headers)
# ============================================
echo "Step 2: Deploy NPP->MPP bridge to cu-bridge"

# Copy bridge wrapper files (from wrapper/mpp)
echo "  Copying bridge wrapper files..."
if [ "$DRY_RUN" = false ]; then
    if [ ! -d "$CU_BRIDGE/include/bridge" ]; then
        echo "    Error: $CU_BRIDGE/include/bridge not found"
        exit 1
    fi
    if [ -d "$PROJECT_DIR/wrapper/mpp" ]; then
        cp -r "$PROJECT_DIR/wrapper/mpp" "$CU_BRIDGE/include/bridge/"
        echo "    Copied to $CU_BRIDGE/include/bridge/mpp/:"
        ls "$PROJECT_DIR/wrapper/mpp" | sed 's/^/      /'
    else
        echo "    Error: $PROJECT_DIR/wrapper/mpp not found"
        exit 1
    fi
else
    echo "  [DRY-RUN] Would copy bridge files to $CU_BRIDGE/include/bridge/mpp/"
fi

# Copy top-level NPP headers (from wrapper/user_include)
echo "  Copying top-level NPP headers..."
if [ "$DRY_RUN" = false ]; then
    if [ ! -d "$CU_BRIDGE/include" ]; then
        echo "    Error: $CU_BRIDGE/include not found"
        exit 1
    fi
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
echo "Deployment Complete"
echo "========================================"
echo "MACA SDK include: $MACA_SDK/include/$(basename "$INCLUDE_SRC")"
echo "MACA SDK libs:"
ls "$MACA_SDK/lib"/libmpp*.so* "$MACA_SDK/lib"/libmpp*.a 2>/dev/null | xargs -n1 basename | sed 's/^/  /'
echo "cu-bridge headers:"
ls "$CU_BRIDGE/include"/npp*.h 2>/dev/null | xargs -n1 basename | sed 's/^/  /'
echo "cu-bridge bridge headers:"
ls "$CU_BRIDGE/include/bridge/mpp"/* 2>/dev/null | xargs -n1 basename | sed 's/^/  /'
