#include "npp.h"
#include <cuda_runtime.h>

/**
 * NPP Image Memory Management Functions - Simplified Implementation
 * Using plain C approach to avoid C++ template issues
 */

/**
 * Internal helper function for memory allocation
 */
static void* nppi_malloc_internal(size_t widthInBytes, int height, int* pStepBytes)
{
    if (widthInBytes == 0 || height <= 0 || !pStepBytes) {
        return NULL;
    }
    
    void* devPtr = NULL;
    size_t pitch = 0;
    
    cudaError_t result = cudaMallocPitch(&devPtr, &pitch, widthInBytes, height);
    
    if (result != cudaSuccess) {
        return NULL;
    }
    
    *pStepBytes = (int)pitch;
    return devPtr;
}

// 8-bit unsigned image memory allocators
Npp8u * nppiMalloc_8u_C1(int nWidthPixels, int nHeightPixels, int * pStepBytes)
{
    return (Npp8u*)nppi_malloc_internal(nWidthPixels * sizeof(Npp8u), nHeightPixels, pStepBytes);
}

Npp8u * nppiMalloc_8u_C2(int nWidthPixels, int nHeightPixels, int * pStepBytes)
{
    return (Npp8u*)nppi_malloc_internal(nWidthPixels * 2 * sizeof(Npp8u), nHeightPixels, pStepBytes);
}

Npp8u * nppiMalloc_8u_C3(int nWidthPixels, int nHeightPixels, int * pStepBytes)
{
    return (Npp8u*)nppi_malloc_internal(nWidthPixels * 3 * sizeof(Npp8u), nHeightPixels, pStepBytes);
}

Npp8u * nppiMalloc_8u_C4(int nWidthPixels, int nHeightPixels, int * pStepBytes)
{
    return (Npp8u*)nppi_malloc_internal(nWidthPixels * 4 * sizeof(Npp8u), nHeightPixels, pStepBytes);
}

// 16-bit unsigned image memory allocators
Npp16u * nppiMalloc_16u_C1(int nWidthPixels, int nHeightPixels, int * pStepBytes)
{
    return (Npp16u*)nppi_malloc_internal(nWidthPixels * sizeof(Npp16u), nHeightPixels, pStepBytes);
}

Npp16u * nppiMalloc_16u_C2(int nWidthPixels, int nHeightPixels, int * pStepBytes)
{
    return (Npp16u*)nppi_malloc_internal(nWidthPixels * 2 * sizeof(Npp16u), nHeightPixels, pStepBytes);
}

Npp16u * nppiMalloc_16u_C3(int nWidthPixels, int nHeightPixels, int * pStepBytes)
{
    return (Npp16u*)nppi_malloc_internal(nWidthPixels * 3 * sizeof(Npp16u), nHeightPixels, pStepBytes);
}

Npp16u * nppiMalloc_16u_C4(int nWidthPixels, int nHeightPixels, int * pStepBytes)
{
    return (Npp16u*)nppi_malloc_internal(nWidthPixels * 4 * sizeof(Npp16u), nHeightPixels, pStepBytes);
}

// 16-bit signed image memory allocators
Npp16s * nppiMalloc_16s_C1(int nWidthPixels, int nHeightPixels, int * pStepBytes)
{
    return (Npp16s*)nppi_malloc_internal(nWidthPixels * sizeof(Npp16s), nHeightPixels, pStepBytes);
}

Npp16s * nppiMalloc_16s_C2(int nWidthPixels, int nHeightPixels, int * pStepBytes)
{
    return (Npp16s*)nppi_malloc_internal(nWidthPixels * 2 * sizeof(Npp16s), nHeightPixels, pStepBytes);
}

Npp16s * nppiMalloc_16s_C4(int nWidthPixels, int nHeightPixels, int * pStepBytes)
{
    return (Npp16s*)nppi_malloc_internal(nWidthPixels * 4 * sizeof(Npp16s), nHeightPixels, pStepBytes);
}

// 32-bit signed image memory allocators
Npp32s * nppiMalloc_32s_C1(int nWidthPixels, int nHeightPixels, int * pStepBytes)
{
    return (Npp32s*)nppi_malloc_internal(nWidthPixels * sizeof(Npp32s), nHeightPixels, pStepBytes);
}

Npp32s * nppiMalloc_32s_C3(int nWidthPixels, int nHeightPixels, int * pStepBytes)
{
    return (Npp32s*)nppi_malloc_internal(nWidthPixels * 3 * sizeof(Npp32s), nHeightPixels, pStepBytes);
}

Npp32s * nppiMalloc_32s_C4(int nWidthPixels, int nHeightPixels, int * pStepBytes)
{
    return (Npp32s*)nppi_malloc_internal(nWidthPixels * 4 * sizeof(Npp32s), nHeightPixels, pStepBytes);
}

// 32-bit float image memory allocators
Npp32f * nppiMalloc_32f_C1(int nWidthPixels, int nHeightPixels, int * pStepBytes)
{
    return (Npp32f*)nppi_malloc_internal(nWidthPixels * sizeof(Npp32f), nHeightPixels, pStepBytes);
}

Npp32f * nppiMalloc_32f_C2(int nWidthPixels, int nHeightPixels, int * pStepBytes)
{
    return (Npp32f*)nppi_malloc_internal(nWidthPixels * 2 * sizeof(Npp32f), nHeightPixels, pStepBytes);
}

Npp32f * nppiMalloc_32f_C3(int nWidthPixels, int nHeightPixels, int * pStepBytes)
{
    return (Npp32f*)nppi_malloc_internal(nWidthPixels * 3 * sizeof(Npp32f), nHeightPixels, pStepBytes);
}

Npp32f * nppiMalloc_32f_C4(int nWidthPixels, int nHeightPixels, int * pStepBytes)
{
    return (Npp32f*)nppi_malloc_internal(nWidthPixels * 4 * sizeof(Npp32f), nHeightPixels, pStepBytes);
}

// 16-bit signed complex image memory allocators
Npp16sc * nppiMalloc_16sc_C1(int nWidthPixels, int nHeightPixels, int * pStepBytes)
{
    return (Npp16sc*)nppi_malloc_internal(nWidthPixels * sizeof(Npp16sc), nHeightPixels, pStepBytes);
}

Npp16sc * nppiMalloc_16sc_C2(int nWidthPixels, int nHeightPixels, int * pStepBytes)
{
    return (Npp16sc*)nppi_malloc_internal(nWidthPixels * 2 * sizeof(Npp16sc), nHeightPixels, pStepBytes);
}

Npp16sc * nppiMalloc_16sc_C3(int nWidthPixels, int nHeightPixels, int * pStepBytes)
{
    return (Npp16sc*)nppi_malloc_internal(nWidthPixels * 3 * sizeof(Npp16sc), nHeightPixels, pStepBytes);
}

Npp16sc * nppiMalloc_16sc_C4(int nWidthPixels, int nHeightPixels, int * pStepBytes)
{
    return (Npp16sc*)nppi_malloc_internal(nWidthPixels * 4 * sizeof(Npp16sc), nHeightPixels, pStepBytes);
}

// 32-bit signed complex image memory allocators
Npp32sc * nppiMalloc_32sc_C1(int nWidthPixels, int nHeightPixels, int * pStepBytes)
{
    return (Npp32sc*)nppi_malloc_internal(nWidthPixels * sizeof(Npp32sc), nHeightPixels, pStepBytes);
}

Npp32sc * nppiMalloc_32sc_C2(int nWidthPixels, int nHeightPixels, int * pStepBytes)
{
    return (Npp32sc*)nppi_malloc_internal(nWidthPixels * 2 * sizeof(Npp32sc), nHeightPixels, pStepBytes);
}

Npp32sc * nppiMalloc_32sc_C3(int nWidthPixels, int nHeightPixels, int * pStepBytes)
{
    return (Npp32sc*)nppi_malloc_internal(nWidthPixels * 3 * sizeof(Npp32sc), nHeightPixels, pStepBytes);
}

Npp32sc * nppiMalloc_32sc_C4(int nWidthPixels, int nHeightPixels, int * pStepBytes)
{
    return (Npp32sc*)nppi_malloc_internal(nWidthPixels * 4 * sizeof(Npp32sc), nHeightPixels, pStepBytes);
}

// 32-bit float complex image memory allocators
Npp32fc * nppiMalloc_32fc_C1(int nWidthPixels, int nHeightPixels, int * pStepBytes)
{
    return (Npp32fc*)nppi_malloc_internal(nWidthPixels * sizeof(Npp32fc), nHeightPixels, pStepBytes);
}

Npp32fc * nppiMalloc_32fc_C2(int nWidthPixels, int nHeightPixels, int * pStepBytes)
{
    return (Npp32fc*)nppi_malloc_internal(nWidthPixels * 2 * sizeof(Npp32fc), nHeightPixels, pStepBytes);
}

Npp32fc * nppiMalloc_32fc_C3(int nWidthPixels, int nHeightPixels, int * pStepBytes)
{
    return (Npp32fc*)nppi_malloc_internal(nWidthPixels * 3 * sizeof(Npp32fc), nHeightPixels, pStepBytes);
}

Npp32fc * nppiMalloc_32fc_C4(int nWidthPixels, int nHeightPixels, int * pStepBytes)
{
    return (Npp32fc*)nppi_malloc_internal(nWidthPixels * 4 * sizeof(Npp32fc), nHeightPixels, pStepBytes);
}

/**
 * Free method for any 2D allocated memory.
 */
void nppiFree(void * pData)
{
    if (pData) {
        cudaFree(pData);
    }
}