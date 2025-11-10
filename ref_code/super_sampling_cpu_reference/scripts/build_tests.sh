#!/bin/bash

# Build script for super sampling CPU reference tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
TEST_DIR="$ROOT_DIR/tests"

CUDA_PATH=${CUDA_PATH:-/usr/local/cuda}

if [ ! -d "$CUDA_PATH" ]; then
    echo "Error: CUDA toolkit not found at $CUDA_PATH"
    exit 1
fi

CXXFLAGS="-std=c++11 -O2 -I$CUDA_PATH/include -I$ROOT_DIR/src"
LDFLAGS="-L$CUDA_PATH/lib64 -lcudart -lnppc -lnppial -lnppig"

echo "=== Building Super Sampling CPU Reference Tests ==="
echo "CUDA Path: $CUDA_PATH"
echo "Root Dir: $ROOT_DIR"
echo ""

# Build validation test
echo "[1/2] Building validation test..."
g++ $CXXFLAGS -o "$ROOT_DIR/validate_reference" \
    "$TEST_DIR/reference_super_sampling.cpp" \
    $LDFLAGS
echo "  ✓ validate_reference"

# Build extensive shapes test
echo "[2/2] Building extensive shapes test..."
g++ $CXXFLAGS -o "$ROOT_DIR/test_shapes" \
    "$TEST_DIR/test_extensive_shapes.cpp" \
    $LDFLAGS
echo "  ✓ test_shapes"

echo ""
echo "=== Build Complete ==="
echo "Run tests:"
echo "  ./validate_reference  - Validate against NPP (11 tests)"
echo "  ./test_shapes         - Extensive shape testing (52 tests)"
