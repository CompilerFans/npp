#include "npp.h"
#include <cuda_runtime.h>
#include <cstring>
#include "../include/npps_kernels.h"

/**
 * NPPS Initialization Functions Implementation - Set Functions
 * Implements nppsSet functions for 1D signal initialization operations.
 */

// ==============================================================================
// Set Operations - Initialize signal elements with constant values
// ==============================================================================

/**
 * 8-bit unsigned char signal set
 */
NppStatus nppsSet_8u_Ctx(Npp8u nValue, Npp8u * pDst, size_t nLength, NppStreamContext nppStreamCtx)
{
    // Parameter validation
    if (!pDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    if (nLength == 0) {
        return NPP_SIZE_ERROR;
    }
    
    // Launch CUDA kernel for setting values
    cudaError_t cudaStatus = nppsSet_8u_kernel(nValue, pDst, nLength, nppStreamCtx.hStream);
    
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    return NPP_NO_ERROR;
}

NppStatus nppsSet_8u(Npp8u nValue, Npp8u * pDst, size_t nLength)
{
    NppStreamContext defaultContext;
    NppStatus status = nppGetStreamContext(&defaultContext);
    if (status != NPP_NO_ERROR) {
        return status;
    }
    
    return nppsSet_8u_Ctx(nValue, pDst, nLength, defaultContext);
}

/**
 * 32-bit float signal set
 */
NppStatus nppsSet_32f_Ctx(Npp32f nValue, Npp32f * pDst, size_t nLength, NppStreamContext nppStreamCtx)
{
    // Parameter validation
    if (!pDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    if (nLength == 0) {
        return NPP_SIZE_ERROR;
    }
    
    // Launch CUDA kernel for setting values
    cudaError_t cudaStatus = nppsSet_32f_kernel(nValue, pDst, nLength, nppStreamCtx.hStream);
    
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    return NPP_NO_ERROR;
}

NppStatus nppsSet_32f(Npp32f nValue, Npp32f * pDst, size_t nLength)
{
    NppStreamContext defaultContext;
    NppStatus status = nppGetStreamContext(&defaultContext);
    if (status != NPP_NO_ERROR) {
        return status;
    }
    
    return nppsSet_32f_Ctx(nValue, pDst, nLength, defaultContext);
}

/**
 * 32-bit float complex signal set
 */
NppStatus nppsSet_32fc_Ctx(Npp32fc nValue, Npp32fc * pDst, size_t nLength, NppStreamContext nppStreamCtx)
{
    // Parameter validation
    if (!pDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    if (nLength == 0) {
        return NPP_SIZE_ERROR;
    }
    
    // Launch CUDA kernel for setting complex values
    cudaError_t cudaStatus = nppsSet_32fc_kernel(nValue, pDst, nLength, nppStreamCtx.hStream);
    
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    return NPP_NO_ERROR;
}

NppStatus nppsSet_32fc(Npp32fc nValue, Npp32fc * pDst, size_t nLength)
{
    NppStreamContext defaultContext;
    NppStatus status = nppGetStreamContext(&defaultContext);
    if (status != NPP_NO_ERROR) {
        return status;
    }
    
    return nppsSet_32fc_Ctx(nValue, pDst, nLength, defaultContext);
}

/**
 * Zero signal initialization - convenient wrapper for set with zero
 */
NppStatus nppsZero_32f_Ctx(Npp32f * pDst, size_t nLength, NppStreamContext nppStreamCtx)
{
    return nppsSet_32f_Ctx(0.0f, pDst, nLength, nppStreamCtx);
}

NppStatus nppsZero_32f(Npp32f * pDst, size_t nLength)
{
    return nppsSet_32f(0.0f, pDst, nLength);
}

// Forward declarations for CUDA kernels (implemented in .cu file)
extern "C" {
    cudaError_t nppsSet_8u_kernel(Npp8u nValue, Npp8u* pDst, size_t nLength, cudaStream_t stream);
    cudaError_t nppsSet_32f_kernel(Npp32f nValue, Npp32f* pDst, size_t nLength, cudaStream_t stream);
    cudaError_t nppsSet_32fc_kernel(Npp32fc nValue, Npp32fc* pDst, size_t nLength, cudaStream_t stream);
}