#!/bin/bash
#
# Build and run NPP user test on MACA platform
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  NPP User Test - MACA Platform"
echo "========================================"
echo ""

# Check environment
echo "[1] Checking environment..."
echo "  CUCC_PATH: ${CUCC_PATH:-not set}"
echo "  MACA_PATH: ${MACA_PATH:-not set}"
echo ""

# Compile
echo "[2] Compiling with cucc..."
echo "  cucc -lnppial -lnppc -lnppidei -lnppisu maca_test.cu -o maca_test "
echo ""

cucc -lnppial -lnppc -lnppidei -lnppisu maca_test.cu -o maca_test

echo "  Compilation successful!"
echo ""

# Run
echo "[3] Running test..."
echo ""

./maca_test

echo ""
echo "Done."
