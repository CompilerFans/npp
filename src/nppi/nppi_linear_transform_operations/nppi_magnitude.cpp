#include "npp.h"
#include <cuda_runtime.h>
#include <cstring>

/**
 * NPP Image Magnitude Functions Implementation
 * Implements nppiMagnitude and nppiMagnitudeSqr functions for complex to real transformations
 */

// Forward declarations for CUDA implementations
extern "C" {
NppStatus nppiMagnitude_32fc32f_C1R_Ctx_cuda(const Npp32fc* pSrc, int nSrcStep, Npp32f* pDst, int nDstStep, 
                                             NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiMagnitudeSqr_32fc32f_C1R_Ctx_cuda(const Npp32fc* pSrc, int nSrcStep, Npp32f* pDst, int nDstStep, 
                                                NppiSize oSizeROI, NppStreamContext nppStreamCtx);
}

/**
 * Helper function for parameter validation
 */
static NppStatus validateMagnitudeParameters(const void* pSrc, int nSrcStep, const void* pDst, int nDstStep, 
                                            NppiSize oSizeROI, int srcElementSize, int dstElementSize)
{
    if (!pSrc || !pDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }
    
    int minSrcStep = oSizeROI.width * srcElementSize;
    int minDstStep = oSizeROI.width * dstElementSize;
    
    if (nSrcStep < minSrcStep || nDstStep < minDstStep) {
        return NPP_STRIDE_ERROR;
    }
    
    return NPP_NO_ERROR;
}

// ============================================================================
// Magnitude functions
// ============================================================================

/**
 * 32-bit floating point complex to 32-bit floating point magnitude
 */
NppStatus nppiMagnitude_32fc32f_C1R_Ctx(const Npp32fc * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, 
                                        NppiSize oSizeROI, NppStreamContext nppStreamCtx)
{
    NppStatus status = validateMagnitudeParameters(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, sizeof(Npp32fc), sizeof(Npp32f));
    if (status != NPP_NO_ERROR) {
        return status;
    }
    
    return nppiMagnitude_32fc32f_C1R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMagnitude_32fc32f_C1R(const Npp32fc * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, 
                                    NppiSize oSizeROI)
{
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiMagnitude_32fc32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// ============================================================================
// Magnitude squared functions
// ============================================================================

/**
 * 32-bit floating point complex to 32-bit floating point squared magnitude
 */
NppStatus nppiMagnitudeSqr_32fc32f_C1R_Ctx(const Npp32fc * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, 
                                           NppiSize oSizeROI, NppStreamContext nppStreamCtx)
{
    NppStatus status = validateMagnitudeParameters(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, sizeof(Npp32fc), sizeof(Npp32f));
    if (status != NPP_NO_ERROR) {
        return status;
    }
    
    return nppiMagnitudeSqr_32fc32f_C1R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMagnitudeSqr_32fc32f_C1R(const Npp32fc * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, 
                                       NppiSize oSizeROI)
{
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiMagnitudeSqr_32fc32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}