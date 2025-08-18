#include "npp.h"
#include <cuda_runtime.h>
#include <cstring>

/**
 * NPP Image Subtract Constant Functions Implementation
 * Implements nppiSubC functions for various data types
 */

// Forward declarations for CUDA implementations
extern "C" {
// Single channel
NppStatus nppiSubC_8u_C1RSfs_Ctx_cuda(const Npp8u* pSrc1, int nSrc1Step, const Npp8u nConstant,
                                      Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                      NppStreamContext nppStreamCtx);

NppStatus nppiSubC_16u_C1RSfs_Ctx_cuda(const Npp16u* pSrc1, int nSrc1Step, const Npp16u nConstant,
                                       Npp16u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx);

NppStatus nppiSubC_16s_C1RSfs_Ctx_cuda(const Npp16s* pSrc1, int nSrc1Step, const Npp16s nConstant,
                                       Npp16s* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx);

NppStatus nppiSubC_32s_C1RSfs_Ctx_cuda(const Npp32s* pSrc1, int nSrc1Step, const Npp32s nConstant,
                                       Npp32s* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx);

NppStatus nppiSubC_32f_C1R_Ctx_cuda(const Npp32f* pSrc1, int nSrc1Step, const Npp32f nConstant,
                                    Npp32f* pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);

// Three channel
NppStatus nppiSubC_8u_C3RSfs_Ctx_cuda(const Npp8u* pSrc1, int nSrc1Step, const Npp8u aConstants[3],
                                      Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                      NppStreamContext nppStreamCtx);

NppStatus nppiSubC_16u_C3RSfs_Ctx_cuda(const Npp16u* pSrc1, int nSrc1Step, const Npp16u aConstants[3],
                                       Npp16u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx);

NppStatus nppiSubC_32f_C3R_Ctx_cuda(const Npp32f* pSrc1, int nSrc1Step, const Npp32f aConstants[3],
                                    Npp32f* pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);

// Four channel
NppStatus nppiSubC_8u_C4RSfs_Ctx_cuda(const Npp8u* pSrc1, int nSrc1Step, const Npp8u aConstants[4],
                                      Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                      NppStreamContext nppStreamCtx);

// In-place operations
NppStatus nppiSubC_8u_C1IRSfs_Ctx_cuda(const Npp8u nConstant, Npp8u* pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx);
}

/**
 * Helper function for parameter validation
 */
static NppStatus validateParameters(const void* pSrc, int nSrcStep, const void* pDst, int nDstStep, 
                                   NppiSize oSizeROI, int nScaleFactor)
{
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
    
    return NPP_NO_ERROR;
}

/**
 * 8-bit unsigned, 1 channel image subtract constant with scale
 */
NppStatus nppiSubC_8u_C1RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u nConstant, 
                                 Npp8u * pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor, 
                                 NppStreamContext nppStreamCtx)
{
    // Validate parameters
    NppStatus status = validateParameters(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor);
    if (status != NPP_NO_ERROR) {
        return status;
    }
    
    // Call CUDA implementation
    return nppiSubC_8u_C1RSfs_Ctx_cuda(pSrc1, nSrc1Step, nConstant, pDst, nDstStep, 
                                       oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 8-bit unsigned, 1 channel image subtract constant with scale (no stream context)
 */
NppStatus nppiSubC_8u_C1RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u nConstant, 
                             Npp8u * pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    // Get default stream context
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    return nppiSubC_8u_C1RSfs_Ctx(pSrc1, nSrc1Step, nConstant, pDst, nDstStep, 
                                  oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 8-bit unsigned, 1 channel in-place image subtract constant with scale
 */
NppStatus nppiSubC_8u_C1IRSfs_Ctx(const Npp8u nConstant, Npp8u * pSrcDst, int nSrcDstStep, 
                                  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx)
{
    // Validate parameters for in-place operation
    if (!pSrcDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }
    
    if (nSrcDstStep < oSizeROI.width) {
        return NPP_STRIDE_ERROR;
    }
    
    if (nScaleFactor < 0 || nScaleFactor > 31) {
        return NPP_BAD_ARGUMENT_ERROR;
    }
    
    // Call CUDA implementation for in-place operation
    return nppiSubC_8u_C1IRSfs_Ctx_cuda(nConstant, pSrcDst, nSrcDstStep, 
                                        oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 8-bit unsigned, 1 channel in-place image subtract constant with scale (no stream context)
 */
NppStatus nppiSubC_8u_C1IRSfs(const Npp8u nConstant, Npp8u * pSrcDst, int nSrcDstStep, 
                              NppiSize oSizeROI, int nScaleFactor)
{
    // Get default stream context
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    return nppiSubC_8u_C1IRSfs_Ctx(nConstant, pSrcDst, nSrcDstStep, 
                                   oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 16-bit unsigned, 1 channel image subtract constant with scale
 */
NppStatus nppiSubC_16u_C1RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u nConstant, 
                                  Npp16u * pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor, 
                                  NppStreamContext nppStreamCtx)
{
    // Validate parameters
    NppStatus status = validateParameters(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor);
    if (status != NPP_NO_ERROR) {
        return status;
    }
    
    // Call CUDA implementation
    return nppiSubC_16u_C1RSfs_Ctx_cuda(pSrc1, nSrc1Step, nConstant, pDst, nDstStep, 
                                        oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 16-bit unsigned, 1 channel image subtract constant with scale (no stream context)
 */
NppStatus nppiSubC_16u_C1RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u nConstant, 
                              Npp16u * pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    // Get default stream context
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    return nppiSubC_16u_C1RSfs_Ctx(pSrc1, nSrc1Step, nConstant, pDst, nDstStep, 
                                   oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 16-bit unsigned, 1 channel in-place image subtract constant with scale
 */
NppStatus nppiSubC_16u_C1IRSfs_Ctx(const Npp16u nConstant, Npp16u * pSrcDst, int nSrcDstStep, 
                                   NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx)
{
    return nppiSubC_16u_C1RSfs_Ctx(pSrcDst, nSrcDstStep, nConstant, pSrcDst, nSrcDstStep, 
                                   oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 16-bit unsigned, 1 channel in-place image subtract constant with scale (no stream context)
 */
NppStatus nppiSubC_16u_C1IRSfs(const Npp16u nConstant, Npp16u * pSrcDst, int nSrcDstStep, 
                               NppiSize oSizeROI, int nScaleFactor)
{
    // Get default stream context
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    return nppiSubC_16u_C1IRSfs_Ctx(nConstant, pSrcDst, nSrcDstStep, 
                                    oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 16-bit signed, 1 channel image subtract constant with scale
 */
NppStatus nppiSubC_16s_C1RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s nConstant, 
                                  Npp16s * pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor, 
                                  NppStreamContext nppStreamCtx)
{
    // Validate parameters
    NppStatus status = validateParameters(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor);
    if (status != NPP_NO_ERROR) {
        return status;
    }
    
    // Call CUDA implementation
    return nppiSubC_16s_C1RSfs_Ctx_cuda(pSrc1, nSrc1Step, nConstant, pDst, nDstStep, 
                                        oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 16-bit signed, 1 channel image subtract constant with scale (no stream context)
 */
NppStatus nppiSubC_16s_C1RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s nConstant, 
                              Npp16s * pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    // Get default stream context
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    return nppiSubC_16s_C1RSfs_Ctx(pSrc1, nSrc1Step, nConstant, pDst, nDstStep, 
                                   oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 16-bit signed, 1 channel in-place image subtract constant with scale
 */
NppStatus nppiSubC_16s_C1IRSfs_Ctx(const Npp16s nConstant, Npp16s * pSrcDst, int nSrcDstStep, 
                                   NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx)
{
    return nppiSubC_16s_C1RSfs_Ctx(pSrcDst, nSrcDstStep, nConstant, pSrcDst, nSrcDstStep, 
                                   oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 16-bit signed, 1 channel in-place image subtract constant with scale (no stream context)
 */
NppStatus nppiSubC_16s_C1IRSfs(const Npp16s nConstant, Npp16s * pSrcDst, int nSrcDstStep, 
                               NppiSize oSizeROI, int nScaleFactor)
{
    // Get default stream context
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    return nppiSubC_16s_C1IRSfs_Ctx(nConstant, pSrcDst, nSrcDstStep, 
                                    oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 32-bit float, 1 channel image subtract constant (no scaling)
 */
NppStatus nppiSubC_32f_C1R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f nConstant, 
                               Npp32f * pDst, int nDstStep, NppiSize oSizeROI, 
                               NppStreamContext nppStreamCtx)
{
    // Validate parameters (no scale factor for float)
    if (!pSrc1 || !pDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }
    
    if (nSrc1Step < oSizeROI.width * sizeof(Npp32f) || nDstStep < oSizeROI.width * sizeof(Npp32f)) {
        return NPP_STRIDE_ERROR;
    }
    
    // Call CUDA implementation
    return nppiSubC_32f_C1R_Ctx_cuda(pSrc1, nSrc1Step, nConstant, pDst, nDstStep, 
                                     oSizeROI, nppStreamCtx);
}

/**
 * 32-bit float, 1 channel image subtract constant (no stream context)
 */
NppStatus nppiSubC_32f_C1R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f nConstant, 
                           Npp32f * pDst, int nDstStep, NppiSize oSizeROI)
{
    // Get default stream context
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    return nppiSubC_32f_C1R_Ctx(pSrc1, nSrc1Step, nConstant, pDst, nDstStep, 
                                oSizeROI, nppStreamCtx);
}

/**
 * 32-bit float, 1 channel in-place image subtract constant
 */
NppStatus nppiSubC_32f_C1IR_Ctx(const Npp32f nConstant, Npp32f * pSrcDst, int nSrcDstStep, 
                                NppiSize oSizeROI, NppStreamContext nppStreamCtx)
{
    return nppiSubC_32f_C1R_Ctx(pSrcDst, nSrcDstStep, nConstant, pSrcDst, nSrcDstStep, 
                                oSizeROI, nppStreamCtx);
}

/**
 * 32-bit float, 1 channel in-place image subtract constant (no stream context)
 */
NppStatus nppiSubC_32f_C1IR(const Npp32f nConstant, Npp32f * pSrcDst, int nSrcDstStep, 
                            NppiSize oSizeROI)
{
    // Get default stream context
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    return nppiSubC_32f_C1IR_Ctx(nConstant, pSrcDst, nSrcDstStep, 
                                 oSizeROI, nppStreamCtx);
}

/**
 * 32-bit signed integer, 1 channel image subtract constant with scale
 */
NppStatus nppiSubC_32s_C1RSfs_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s nConstant, 
                                  Npp32s * pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor, 
                                  NppStreamContext nppStreamCtx)
{
    // Validate parameters
    NppStatus status = validateParameters(pSrc1, nSrc1Step * sizeof(Npp32s), pDst, 
                                         nDstStep * sizeof(Npp32s), oSizeROI, nScaleFactor);
    if (status != NPP_NO_ERROR) {
        return status;
    }
    
    // Call CUDA implementation
    return nppiSubC_32s_C1RSfs_Ctx_cuda(pSrc1, nSrc1Step, nConstant, pDst, nDstStep, 
                                        oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSubC_32s_C1RSfs(const Npp32s * pSrc1, int nSrc1Step, const Npp32s nConstant, 
                              Npp32s * pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiSubC_32s_C1RSfs_Ctx(pSrc1, nSrc1Step, nConstant, pDst, nDstStep, 
                                   oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 8-bit unsigned, 3 channel image subtract constants with scale
 */
NppStatus nppiSubC_8u_C3RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u aConstants[3], 
                                 Npp8u * pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor, 
                                 NppStreamContext nppStreamCtx)
{
    // Validate parameters - for 3-channel, each pixel is 3 bytes
    if (!pSrc1 || !pDst || !aConstants) {
        return NPP_NULL_POINTER_ERROR;
    }
    
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }
    
    if (nSrc1Step < oSizeROI.width * 3 || nDstStep < oSizeROI.width * 3) {
        return NPP_STRIDE_ERROR;
    }
    
    if (nScaleFactor < 0 || nScaleFactor > 31) {
        return NPP_BAD_ARGUMENT_ERROR;
    }
    
    // Call CUDA implementation
    return nppiSubC_8u_C3RSfs_Ctx_cuda(pSrc1, nSrc1Step, aConstants, pDst, nDstStep, 
                                       oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSubC_8u_C3RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u aConstants[3], 
                             Npp8u * pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiSubC_8u_C3RSfs_Ctx(pSrc1, nSrc1Step, aConstants, pDst, nDstStep, 
                                  oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 16-bit unsigned, 3 channel image subtract constants with scale
 */
NppStatus nppiSubC_16u_C3RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u aConstants[3], 
                                  Npp16u * pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor, 
                                  NppStreamContext nppStreamCtx)
{
    if (!pSrc1 || !pDst || !aConstants) {
        return NPP_NULL_POINTER_ERROR;
    }
    
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }
    
    if (nSrc1Step < oSizeROI.width * 3 * sizeof(Npp16u) || 
        nDstStep < oSizeROI.width * 3 * sizeof(Npp16u)) {
        return NPP_STRIDE_ERROR;
    }
    
    if (nScaleFactor < 0 || nScaleFactor > 31) {
        return NPP_BAD_ARGUMENT_ERROR;
    }
    
    return nppiSubC_16u_C3RSfs_Ctx_cuda(pSrc1, nSrc1Step, aConstants, pDst, nDstStep, 
                                        oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSubC_16u_C3RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u aConstants[3], 
                              Npp16u * pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiSubC_16u_C3RSfs_Ctx(pSrc1, nSrc1Step, aConstants, pDst, nDstStep, 
                                   oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 32-bit float, 3 channel image subtract constants
 */
NppStatus nppiSubC_32f_C3R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f aConstants[3], 
                               Npp32f * pDst, int nDstStep, NppiSize oSizeROI, 
                               NppStreamContext nppStreamCtx)
{
    if (!pSrc1 || !pDst || !aConstants) {
        return NPP_NULL_POINTER_ERROR;
    }
    
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }
    
    if (nSrc1Step < oSizeROI.width * 3 * sizeof(Npp32f) || 
        nDstStep < oSizeROI.width * 3 * sizeof(Npp32f)) {
        return NPP_STRIDE_ERROR;
    }
    
    return nppiSubC_32f_C3R_Ctx_cuda(pSrc1, nSrc1Step, aConstants, pDst, nDstStep, 
                                     oSizeROI, nppStreamCtx);
}

NppStatus nppiSubC_32f_C3R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f aConstants[3], 
                           Npp32f * pDst, int nDstStep, NppiSize oSizeROI)
{
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiSubC_32f_C3R_Ctx(pSrc1, nSrc1Step, aConstants, pDst, nDstStep, 
                                oSizeROI, nppStreamCtx);
}

/**
 * 8-bit unsigned, 4 channel image subtract constants with scale
 */
NppStatus nppiSubC_8u_C4RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u aConstants[4], 
                                 Npp8u * pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor, 
                                 NppStreamContext nppStreamCtx)
{
    if (!pSrc1 || !pDst || !aConstants) {
        return NPP_NULL_POINTER_ERROR;
    }
    
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }
    
    if (nSrc1Step < oSizeROI.width * 4 || nDstStep < oSizeROI.width * 4) {
        return NPP_STRIDE_ERROR;
    }
    
    if (nScaleFactor < 0 || nScaleFactor > 31) {
        return NPP_BAD_ARGUMENT_ERROR;
    }
    
    return nppiSubC_8u_C4RSfs_Ctx_cuda(pSrc1, nSrc1Step, aConstants, pDst, nDstStep, 
                                       oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSubC_8u_C4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u aConstants[4], 
                             Npp8u * pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiSubC_8u_C4RSfs_Ctx(pSrc1, nSrc1Step, aConstants, pDst, nDstStep, 
                                  oSizeROI, nScaleFactor, nppStreamCtx);
}

