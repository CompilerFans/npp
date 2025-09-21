#!/bin/bash

# Fix remaining CUDA API calls that were missed

set -e

echo "Fixing remaining CUDA API calls..."

# Fix remaining CUDA API calls in .cu files specifically
find src/ test/ -name "*.cu" | while read -r file; do
    echo "Fixing .cu file: $file"
    
    sed -i '
        # Fix GPU function calls that should be CUDA
        s/gpuGetLastError/cudaGetLastError/g
        s/gpuStreamSynchronize/cudaStreamSynchronize/g
        s/gpuErr/cudaErr/g
        
        # Additional CUDA runtime calls that might have been missed
        s/gpuMemcpyAsync/cudaMemcpyAsync/g
        s/gpuEventCreate/cudaEventCreate/g
        s/gpuEventDestroy/cudaEventDestroy/g
        s/gpuEventRecord/cudaEventRecord/g
        s/gpuEventSynchronize/cudaEventSynchronize/g
        s/gpuStreamCreate/cudaStreamCreate/g
        s/gpuStreamDestroy/cudaStreamDestroy/g
        s/gpuOccupancyMaxPotentialBlockSize/cudaOccupancyMaxPotentialBlockSize/g
        
        # Fix variable names that might have been changed
        s/\bgpuErr\b/cudaErr/g
        s/\bgpuStatus\b/cudaStatus/g
        s/\bgpuResult\b/cudaResult/g
    ' "$file"
done

echo "Fixing remaining CUDA API calls completed!"