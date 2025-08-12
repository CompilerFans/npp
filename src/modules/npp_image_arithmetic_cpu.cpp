#include "npp_image_arithmetic.h"
#include <cstring>
#include <cassert>
#include <limits>

/**
 * Reference CPU implementation of nppiAddC_8u_C1RSfs_Ctx
 * 
 * This function adds a constant value to each pixel in the source image,
 * scales the result by right-shifting, and clamps to the 8-bit range [0, 255].
 */
NppStatus 
nppiAddC_8u_C1RSfs_Ctx_reference(const Npp8u* pSrc1, int nSrc1Step, const Npp8u nConstant,
                                   Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    // Parameter validation
    if (!pSrc1 || !pDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }
    
    if (nSrc1Step < oSizeROI.width || nDstStep < oSizeROI.width) {
        return NPP_STRIDE_ERROR;
    }
    
    if (nScaleFactor < 0 || nScaleFactor > 16) {
        return NPP_BAD_ARGUMENT_ERROR;
    }

    // Process each pixel
    for (int y = 0; y < oSizeROI.height; y++) {
        const Npp8u* srcRow = pSrc1 + y * nSrc1Step;
        Npp8u* dstRow = pDst + y * nDstStep;
        
        for (int x = 0; x < oSizeROI.width; x++) {
            // Load source pixel
            Npp8u srcPixel = srcRow[x];
            
            // Add constant (promoted to int to avoid overflow)
            int result = static_cast<int>(srcPixel) + static_cast<int>(nConstant);
            
            // Scale by right-shifting
            result = result >> nScaleFactor;
            
            // Clamp to 8-bit range
            if (result < 0) {
                result = 0;
            } else if (result > 255) {
                result = 255;
            }
            
            // Store result
            dstRow[x] = static_cast<Npp8u>(result);
        }
    }
    
    return NPP_NO_ERROR;
}

/**
 * Host implementation (CPU-based) for nppiAddC_8u_C1RSfs_Ctx
 * Uses the same algorithm as the reference but with CUDA stream context
 */
NppStatus 
nppiAddC_8u_C1RSfs_Ctx_host(const Npp8u* pSrc1, int nSrc1Step, const Npp8u nConstant,
                              Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                              NppStreamContext nppStreamCtx)
{
    // This is a CPU implementation, so stream context is ignored
    return nppiAddC_8u_C1RSfs_Ctx_reference(pSrc1, nSrc1Step, nConstant, 
                                            pDst, nDstStep, oSizeROI, nScaleFactor);
}