#!/bin/bash

# Final cleanup for remaining Chinese comments and inconsistencies
set -e

echo "=== Final Cleanup Script ==="

# Fix remaining Chinese comments
find src test -name "*.cpp" -o -name "*.cu" -o -name "*.h" | while read -r file; do
    if [[ "$file" != *"/API/"* ]]; then
        sed -i \
            -e 's/验证锚点在掩码范围内/Validate anchor within mask bounds/g' \
            -e 's/调用GPU内核/Call GPU kernel/g' \
            -e 's/默认流/Default stream/g' \
            -e 's/参数验证/Parameter validation/g' \
            -e 's/验证/Validate/g' \
            -e 's/调用/Call/g' \
            -e 's/内核/kernel/g' \
            -e 's/锚点/anchor/g' \
            -e 's/掩码/mask/g' \
            -e 's/范围/bounds/g' \
            -e 's/GPU/GPU/g' \
            -e 's/NPP_CUDA_KERNEL_EXECUTION_ERROR/NPP_GPU_KERNEL_EXECUTION_ERROR/g' \
            "$file"
    fi
done

# Fix any inconsistent GPU references that were missed
find src test -name "*.cpp" -o -name "*.cu" -o -name "*.h" | while read -r file; do
    if [[ "$file" != *"/API/"* ]]; then
        # Replace any remaining CUDA references with GPU
        sed -i 's/NPP_CUDA_KERNEL_EXECUTION_ERROR/NPP_GPU_KERNEL_EXECUTION_ERROR/g' "$file"
    fi
done

echo "Final cleanup completed!"