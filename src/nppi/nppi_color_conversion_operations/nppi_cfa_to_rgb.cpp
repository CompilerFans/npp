#ifndef USE_NVIDIA_NPP

#include "npp.h"
#include <cuda_runtime.h>
#include <cstring>
#include <cstdio>

/**
 * NPP CFA to RGB Conversion Functions Implementation
 * Implements nppiCFAToRGB functions for Color Filter Array (Bayer pattern) to RGB conversion
 */

// Forward declarations for CUDA implementations
extern "C" {
NppStatus nppiCFAToRGB_8u_C1C3R_Ctx_cuda(const Npp8u* pSrc, int nSrcStep, NppiSize oSrcSize,
                                         NppiRect oSrcROI, Npp8u* pDst, int nDstStep,
                                         NppiBayerGridPosition eGrid, NppiInterpolationMode eInterpolation,
                                         NppStreamContext nppStreamCtx);
NppStatus nppiCFAToRGB_16u_C1C3R_Ctx_cuda(const Npp16u* pSrc, int nSrcStep, NppiSize oSrcSize,
                                          NppiRect oSrcROI, Npp16u* pDst, int nDstStep,
                                          NppiBayerGridPosition eGrid, NppiInterpolationMode eInterpolation,
                                          NppStreamContext nppStreamCtx);
}

// Input validation helper
static inline NppStatus validateCFAToRGBInputs(const void* pSrc, int nSrcStep, NppiSize oSrcSize,
                                               NppiRect oSrcROI, void* pDst, int nDstStep,
                                               NppiBayerGridPosition eGrid, NppiInterpolationMode /*eInterpolation*/) {
    if (oSrcSize.width <= 0 || oSrcSize.height <= 0) {
        return NPP_SIZE_ERROR;
    }
    
    if (nSrcStep <= 0 || nDstStep <= 0) {
        return NPP_STEP_ERROR;
    }
    
    if (!pSrc || !pDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    
    // Basic ROI validation
    if (oSrcROI.x < 0 || oSrcROI.y < 0 || 
        oSrcROI.x + oSrcROI.width > oSrcSize.width ||
        oSrcROI.y + oSrcROI.height > oSrcSize.height) {
        return NPP_SIZE_ERROR;
    }
    
    // Validate Bayer grid position
    if (eGrid < 0 || eGrid > 3) {
        return NPP_BAD_ARGUMENT_ERROR;
    }
    
    return NPP_SUCCESS;
}

/**
 * Convert single channel CFA (Color Filter Array) image to RGB
 * 8-bit unsigned implementation
 */
NppStatus nppiCFAToRGB_8u_C1C3R_Ctx(const Npp8u* pSrc, int nSrcStep, NppiSize oSrcSize,
                                   NppiRect oSrcROI, Npp8u* pDst, int nDstStep,
                                   NppiBayerGridPosition eGrid, NppiInterpolationMode eInterpolation,
                                   NppStreamContext nppStreamCtx) {
    NppStatus status = validateCFAToRGBInputs(pSrc, nSrcStep, oSrcSize, oSrcROI,
                                             pDst, nDstStep, eGrid, eInterpolation);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    return nppiCFAToRGB_8u_C1C3R_Ctx_cuda(pSrc, nSrcStep, oSrcSize, oSrcROI,
                                         pDst, nDstStep, eGrid, eInterpolation,
                                         nppStreamCtx);
}

NppStatus nppiCFAToRGB_8u_C1C3R(const Npp8u* pSrc, int nSrcStep, NppiSize oSrcSize,
                               NppiRect oSrcROI, Npp8u* pDst, int nDstStep,
                               NppiBayerGridPosition eGrid, NppiInterpolationMode eInterpolation) {
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiCFAToRGB_8u_C1C3R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcROI,
                                    pDst, nDstStep, eGrid, eInterpolation,
                                    nppStreamCtx);
}

/**
 * Convert single channel CFA (Color Filter Array) image to RGB
 * 16-bit unsigned implementation
 */
NppStatus nppiCFAToRGB_16u_C1C3R_Ctx(const Npp16u* pSrc, int nSrcStep, NppiSize oSrcSize,
                                    NppiRect oSrcROI, Npp16u* pDst, int nDstStep,
                                    NppiBayerGridPosition eGrid, NppiInterpolationMode eInterpolation,
                                    NppStreamContext nppStreamCtx) {
    NppStatus status = validateCFAToRGBInputs(pSrc, nSrcStep, oSrcSize, oSrcROI,
                                             pDst, nDstStep, eGrid, eInterpolation);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    return nppiCFAToRGB_16u_C1C3R_Ctx_cuda(pSrc, nSrcStep, oSrcSize, oSrcROI,
                                          pDst, nDstStep, eGrid, eInterpolation,
                                          nppStreamCtx);
}

NppStatus nppiCFAToRGB_16u_C1C3R(const Npp16u* pSrc, int nSrcStep, NppiSize oSrcSize,
                                NppiRect oSrcROI, Npp16u* pDst, int nDstStep,
                                NppiBayerGridPosition eGrid, NppiInterpolationMode eInterpolation) {
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiCFAToRGB_16u_C1C3R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcROI,
                                     pDst, nDstStep, eGrid, eInterpolation,
                                     nppStreamCtx);
}

#endif // USE_NVIDIA_NPP