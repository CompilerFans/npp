#ifndef NPPI_MEMORY_ALLOCATOR_H
#define NPPI_MEMORY_ALLOCATOR_H

#include <cuda_runtime.h>
#include <cstddef>

/**
 * Internal memory allocator utilities for NPPI
 */
namespace nppi_internal {

/**
 * Generic pitched memory allocation function
 * This function is used internally by all nppiMalloc_* functions
 */
inline void* allocatePitchedMemory(size_t widthInBytes, int height, int* pStepBytes)
{
    if (widthInBytes == 0 || height <= 0 || !pStepBytes) {
        return nullptr;
    }
    
    void* devPtr = nullptr;
    size_t pitch = 0;
    
    cudaError_t result = cudaMallocPitch(&devPtr, &pitch, widthInBytes, height);
    
    if (result != cudaSuccess) {
        return nullptr;
    }
    
    *pStepBytes = static_cast<int>(pitch);
    return devPtr;
}

/**
 * Template wrapper for type-safe memory allocation
 */
template<typename T>
inline T* allocateTypedMemory(int width, int height, int channels, int* pStepBytes)
{
    size_t widthInBytes = static_cast<size_t>(width) * channels * sizeof(T);
    return static_cast<T*>(allocatePitchedMemory(widthInBytes, height, pStepBytes));
}

} // namespace nppi_internal

#endif // NPPI_MEMORY_ALLOCATOR_H