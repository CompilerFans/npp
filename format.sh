#!/bin/bash

# Simple code formatting script using clang-format
# Formats C/C++/CUDA files in src/ and test/ directories with LLVM style

set -e

echo "=== NPP Code Formatting Script ==="

# Create .clang-format file with LLVM style
cat > .clang-format << EOF
---
BasedOnStyle: LLVM
IndentWidth: 2
ColumnLimit: 120
EOF

echo "Formatting C/C++/CUDA files in src/ and test/ directories..."

# Format all C/C++/CUDA files in src directory
if [ -d "src" ]; then
    echo "Formatting src/ directory..."
    find src -name "*.cpp" -o -name "*.c" -o -name "*.h" -o -name "*.hpp" -o -name "*.cu" -o -name "*.cuh" | \
    while read file; do
        echo "  Formatting: $file"
        clang-format -i --style=file "$file"
    done
fi

# Format all C/C++/CUDA files in test directory  
if [ -d "test" ]; then
    echo "Formatting test/ directory..."
    find test -name "*.cpp" -o -name "*.c" -o -name "*.h" -o -name "*.hpp" -o -name "*.cu" -o -name "*.cuh" | \
    while read file; do
        echo "  Formatting: $file"
        clang-format -i --style=file "$file"
    done
fi

echo "Code formatting completed!"
echo "Review changes with: git diff"
