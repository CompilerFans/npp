#include "npp.h"
#include <algorithm>
#include <cmath>

/**
 * CPU Reference Implementations for NPP AddC operations
 * Used for validation and comparison testing
 */

namespace nppi_reference {

/**
 * CPU reference implementation for 8u AddC
 */
NppStatus nppiAddC_8u_C1RSfs_reference(const Npp8u* pSrc, int nSrcStep, const Npp8u nConstant,
                                       Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    // Parameter validation
    if (!pSrc || !pDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }
    
    if (nSrcStep < oSizeROI.width || nDstStep < oSizeROI.width) {
        return NPP_STRIDE_ERROR;
    }
    
    if (nScaleFactor < 0 || nScaleFactor > 31) {
        return NPP_BAD_ARGUMENT_ERROR;
    }

    // Process each pixel
    for (int y = 0; y < oSizeROI.height; y++) {
        const Npp8u* srcRow = reinterpret_cast<const Npp8u*>(
            reinterpret_cast<const char*>(pSrc) + y * nSrcStep);
        Npp8u* dstRow = reinterpret_cast<Npp8u*>(
            reinterpret_cast<char*>(pDst) + y * nDstStep);
        
        for (int x = 0; x < oSizeROI.width; x++) {
            // Load source pixel
            int srcValue = static_cast<int>(srcRow[x]);
            
            // Add constant
            int result = srcValue + static_cast<int>(nConstant);
            
            // Scale by right-shifting
            result = result >> nScaleFactor;
            
            // Clamp to 8-bit range
            result = std::max(0, std::min(255, result));
            
            // Store result
            dstRow[x] = static_cast<Npp8u>(result);
        }
    }
    
    return NPP_NO_ERROR;
}

/**
 * CPU reference implementation for 16u AddC
 */
NppStatus nppiAddC_16u_C1RSfs_reference(const Npp16u* pSrc, int nSrcStep, const Npp16u nConstant,
                                        Npp16u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    // Parameter validation
    if (!pSrc || !pDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }
    
    if (nSrcStep < oSizeROI.width * sizeof(Npp16u) || nDstStep < oSizeROI.width * sizeof(Npp16u)) {
        return NPP_STRIDE_ERROR;
    }
    
    if (nScaleFactor < 0 || nScaleFactor > 31) {
        return NPP_BAD_ARGUMENT_ERROR;
    }

    // Process each pixel
    for (int y = 0; y < oSizeROI.height; y++) {
        const Npp16u* srcRow = reinterpret_cast<const Npp16u*>(
            reinterpret_cast<const char*>(pSrc) + y * nSrcStep);
        Npp16u* dstRow = reinterpret_cast<Npp16u*>(
            reinterpret_cast<char*>(pDst) + y * nDstStep);
        
        for (int x = 0; x < oSizeROI.width; x++) {
            // Load source pixel
            int srcValue = static_cast<int>(srcRow[x]);
            
            // Add constant
            int result = srcValue + static_cast<int>(nConstant);
            
            // Scale by right-shifting
            result = result >> nScaleFactor;
            
            // Clamp to 16-bit unsigned range
            result = std::max(0, std::min(65535, result));
            
            // Store result
            dstRow[x] = static_cast<Npp16u>(result);
        }
    }
    
    return NPP_NO_ERROR;
}

/**
 * CPU reference implementation for 16s AddC
 */
NppStatus nppiAddC_16s_C1RSfs_reference(const Npp16s* pSrc, int nSrcStep, const Npp16s nConstant,
                                        Npp16s* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    // Parameter validation
    if (!pSrc || !pDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }
    
    if (nSrcStep < oSizeROI.width * sizeof(Npp16s) || nDstStep < oSizeROI.width * sizeof(Npp16s)) {
        return NPP_STRIDE_ERROR;
    }
    
    if (nScaleFactor < 0 || nScaleFactor > 31) {
        return NPP_BAD_ARGUMENT_ERROR;
    }

    // Process each pixel
    for (int y = 0; y < oSizeROI.height; y++) {
        const Npp16s* srcRow = reinterpret_cast<const Npp16s*>(
            reinterpret_cast<const char*>(pSrc) + y * nSrcStep);
        Npp16s* dstRow = reinterpret_cast<Npp16s*>(
            reinterpret_cast<char*>(pDst) + y * nDstStep);
        
        for (int x = 0; x < oSizeROI.width; x++) {
            // Load source pixel
            int srcValue = static_cast<int>(srcRow[x]);
            
            // Add constant
            int result = srcValue + static_cast<int>(nConstant);
            
            // Scale by right-shifting
            result = result >> nScaleFactor;
            
            // Clamp to 16-bit signed range
            result = std::max(-32768, std::min(32767, result));
            
            // Store result
            dstRow[x] = static_cast<Npp16s>(result);
        }
    }
    
    return NPP_NO_ERROR;
}

/**
 * CPU reference implementation for 32f AddC
 */
NppStatus nppiAddC_32f_C1R_reference(const Npp32f* pSrc, int nSrcStep, const Npp32f nConstant,
                                     Npp32f* pDst, int nDstStep, NppiSize oSizeROI)
{
    // Parameter validation
    if (!pSrc || !pDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }
    
    if (nSrcStep < oSizeROI.width * sizeof(Npp32f) || nDstStep < oSizeROI.width * sizeof(Npp32f)) {
        return NPP_STRIDE_ERROR;
    }

    // Process each pixel
    for (int y = 0; y < oSizeROI.height; y++) {
        const Npp32f* srcRow = reinterpret_cast<const Npp32f*>(
            reinterpret_cast<const char*>(pSrc) + y * nSrcStep);
        Npp32f* dstRow = reinterpret_cast<Npp32f*>(
            reinterpret_cast<char*>(pDst) + y * nDstStep);
        
        for (int x = 0; x < oSizeROI.width; x++) {
            // Load source pixel and add constant (no scaling for float)
            dstRow[x] = srcRow[x] + nConstant;
        }
    }
    
    return NPP_NO_ERROR;
}

} // namespace nppi_reference