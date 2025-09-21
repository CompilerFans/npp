#!/bin/bash

# Fix CUDA API naming that was incorrectly replaced

set -e

echo "Fixing CUDA API naming..."

# Restore CUDA API calls that should not be changed
find src/ test/ -type f \( -name "*.cpp" -o -name "*.cu" -o -name "*.h" -o -name "*.hpp" \) | while read -r file; do
    # Skip API directory
    if [[ "$file" == api/* ]]; then
        continue
    fi
    
    echo "Fixing: $file"
    
    # Fix specific CUDA API calls that were wrongly replaced
    sed -i '
        # Fix CUDA runtime API calls
        s/gpu_runtime\.h/cuda_runtime.h/g
        s/gpuError_t/cudaError_t/g
        s/gpuSuccess/cudaSuccess/g
        s/gpuGetDevice/cudaGetDevice/g
        s/gpuSetDevice/cudaSetDevice/g
        s/gpuGetDeviceCount/cudaGetDeviceCount/g
        s/gpuGetDeviceProperties/cudaGetDeviceProperties/g
        s/gpuDeviceProp/cudaDeviceProp/g
        s/gpuDeviceSynchronize/cudaDeviceSynchronize/g
        s/gpuGetErrorString/cudaGetErrorString/g
        s/gpuMalloc/cudaMalloc/g
        s/gpuFree/cudaFree/g
        s/gpuMemcpy/cudaMemcpy/g
        s/gpuMemcpy2D/cudaMemcpy2D/g
        s/gpuMemset/cudaMemset/g
        s/gpuMemcpyHostToDevice/cudaMemcpyHostToDevice/g
        s/gpuMemcpyDeviceToHost/cudaMemcpyDeviceToHost/g
        s/gpuStream_t/cudaStream_t/g
        s/gpuStreamLegacy/cudaStreamLegacy/g
        s/gpuStreamPerThread/cudaStreamPerThread/g
        s/gpuStreamDefault/cudaStreamDefault/g
        s/gpuStreamGetFlags/cudaStreamGetFlags/g
        
        # Fix struct field names that should remain CUDA-specific
        s/nGpuDeviceId/nCudaDeviceId/g
        s/nGpuDevAttr/nCudaDevAttr/g
        
        # Fix error status variables that were renamed
        s/gpuStatus/cudaStatus/g
        
        # But keep NPP error constants as they should be
        s/NPP_GPU_KERNEL_EXECUTION_ERROR/NPP_CUDA_KERNEL_EXECUTION_ERROR/g
        
        # Keep generic terms in comments but fix API calls
        /\/\//!{
            # This applies to non-comment lines only
        }
    ' "$file"
done

echo "CUDA API naming fixes completed!"
echo ""
echo "Fixed the following patterns:"
echo "  - Restored cuda_runtime.h include"
echo "  - Restored cudaError_t and related CUDA types"
echo "  - Restored CUDA runtime API function names"
echo "  - Restored CUDA-specific struct field names"
echo "  - Restored NPP error constants"
echo ""
echo "Generic terms in comments were preserved."