#include "npp.h"
#include <cuda_runtime.h>
#include <cstring>

/**
 * NPP Image Divide Constant Functions Implementation
 * Implements nppiDivC functions for various data types
 */

// Forward declarations for CUDA implementations
extern "C" {
NppStatus nppiDivC_8u_C1RSfs_Ctx_cuda(const Npp8u* pSrc1, int nSrc1Step, const Npp8u nConstant,
                                      Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                      NppStreamContext nppStreamCtx);

NppStatus nppiDivC_16u_C1RSfs_Ctx_cuda(const Npp16u* pSrc1, int nSrc1Step, const Npp16u nConstant,
                                       Npp16u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx);

NppStatus nppiDivC_16s_C1RSfs_Ctx_cuda(const Npp16s* pSrc1, int nSrc1Step, const Npp16s nConstant,
                                       Npp16s* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx);

NppStatus nppiDivC_32f_C1R_Ctx_cuda(const Npp32f* pSrc1, int nSrc1Step, const Npp32f nConstant,
                                    Npp32f* pDst, int nDstStep, NppiSize oSizeROI,
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
 * 8-bit unsigned, 1 channel image divide constant with scale
 */
NppStatus nppiDivC_8u_C1RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u nConstant, 
                                 Npp8u * pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor, 
                                 NppStreamContext nppStreamCtx)
{
    // Validate parameters
    NppStatus status = validateParameters(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor);
    if (status != NPP_NO_ERROR) {
        return status;
    }
    
    // Check for division by zero
    if (nConstant == 0) {
        return NPP_DIVIDE_BY_ZERO_ERROR;
    }
    
    // Call CUDA implementation
    return nppiDivC_8u_C1RSfs_Ctx_cuda(pSrc1, nSrc1Step, nConstant, pDst, nDstStep, 
                                       oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 8-bit unsigned, 1 channel image divide constant with scale (no stream context)
 */
NppStatus nppiDivC_8u_C1RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u nConstant, 
                             Npp8u * pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    // Get default stream context
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    return nppiDivC_8u_C1RSfs_Ctx(pSrc1, nSrc1Step, nConstant, pDst, nDstStep, 
                                  oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 8-bit unsigned, 1 channel in-place image divide constant with scale
 */
NppStatus nppiDivC_8u_C1IRSfs_Ctx(const Npp8u nConstant, Npp8u * pSrcDst, int nSrcDstStep, 
                                  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx)
{
    // For in-place operation, source and destination are the same
    return nppiDivC_8u_C1RSfs_Ctx(pSrcDst, nSrcDstStep, nConstant, pSrcDst, nSrcDstStep, 
                                  oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 8-bit unsigned, 1 channel in-place image divide constant with scale (no stream context)
 */
NppStatus nppiDivC_8u_C1IRSfs(const Npp8u nConstant, Npp8u * pSrcDst, int nSrcDstStep, 
                              NppiSize oSizeROI, int nScaleFactor)
{
    // Get default stream context
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    return nppiDivC_8u_C1IRSfs_Ctx(nConstant, pSrcDst, nSrcDstStep, 
                                   oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 16-bit unsigned, 1 channel image divide constant with scale
 */
NppStatus nppiDivC_16u_C1RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u nConstant, 
                                  Npp16u * pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor, 
                                  NppStreamContext nppStreamCtx)
{
    // Validate parameters
    NppStatus status = validateParameters(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor);
    if (status != NPP_NO_ERROR) {
        return status;
    }
    
    // Check for division by zero
    if (nConstant == 0) {
        return NPP_DIVIDE_BY_ZERO_ERROR;
    }
    
    // Call CUDA implementation
    return nppiDivC_16u_C1RSfs_Ctx_cuda(pSrc1, nSrc1Step, nConstant, pDst, nDstStep, 
                                        oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 16-bit unsigned, 1 channel image divide constant with scale (no stream context)
 */
NppStatus nppiDivC_16u_C1RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u nConstant, 
                              Npp16u * pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    // Get default stream context
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    return nppiDivC_16u_C1RSfs_Ctx(pSrc1, nSrc1Step, nConstant, pDst, nDstStep, 
                                   oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 16-bit unsigned, 1 channel in-place image divide constant with scale
 */
NppStatus nppiDivC_16u_C1IRSfs_Ctx(const Npp16u nConstant, Npp16u * pSrcDst, int nSrcDstStep, 
                                   NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx)
{
    return nppiDivC_16u_C1RSfs_Ctx(pSrcDst, nSrcDstStep, nConstant, pSrcDst, nSrcDstStep, 
                                   oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 16-bit unsigned, 1 channel in-place image divide constant with scale (no stream context)
 */
NppStatus nppiDivC_16u_C1IRSfs(const Npp16u nConstant, Npp16u * pSrcDst, int nSrcDstStep, 
                               NppiSize oSizeROI, int nScaleFactor)
{
    // Get default stream context
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    return nppiDivC_16u_C1IRSfs_Ctx(nConstant, pSrcDst, nSrcDstStep, 
                                    oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 16-bit signed, 1 channel image divide constant with scale
 */
NppStatus nppiDivC_16s_C1RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s nConstant, 
                                  Npp16s * pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor, 
                                  NppStreamContext nppStreamCtx)
{
    // Validate parameters
    NppStatus status = validateParameters(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor);
    if (status != NPP_NO_ERROR) {
        return status;
    }
    
    // Check for division by zero
    if (nConstant == 0) {
        return NPP_DIVIDE_BY_ZERO_ERROR;
    }
    
    // Call CUDA implementation
    return nppiDivC_16s_C1RSfs_Ctx_cuda(pSrc1, nSrc1Step, nConstant, pDst, nDstStep, 
                                        oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 16-bit signed, 1 channel image divide constant with scale (no stream context)
 */
NppStatus nppiDivC_16s_C1RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s nConstant, 
                              Npp16s * pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    // Get default stream context
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    return nppiDivC_16s_C1RSfs_Ctx(pSrc1, nSrc1Step, nConstant, pDst, nDstStep, 
                                   oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 16-bit signed, 1 channel in-place image divide constant with scale
 */
NppStatus nppiDivC_16s_C1IRSfs_Ctx(const Npp16s nConstant, Npp16s * pSrcDst, int nSrcDstStep, 
                                   NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx)
{
    return nppiDivC_16s_C1RSfs_Ctx(pSrcDst, nSrcDstStep, nConstant, pSrcDst, nSrcDstStep, 
                                   oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 16-bit signed, 1 channel in-place image divide constant with scale (no stream context)
 */
NppStatus nppiDivC_16s_C1IRSfs(const Npp16s nConstant, Npp16s * pSrcDst, int nSrcDstStep, 
                               NppiSize oSizeROI, int nScaleFactor)
{
    // Get default stream context
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    return nppiDivC_16s_C1IRSfs_Ctx(nConstant, pSrcDst, nSrcDstStep, 
                                    oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 32-bit float, 1 channel image divide constant (no scaling)
 */
NppStatus nppiDivC_32f_C1R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f nConstant, 
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
    
    // For float division, allow division by zero (produces inf/nan as appropriate)
    
    // Call CUDA implementation
    return nppiDivC_32f_C1R_Ctx_cuda(pSrc1, nSrc1Step, nConstant, pDst, nDstStep, 
                                     oSizeROI, nppStreamCtx);
}

/**
 * 32-bit float, 1 channel image divide constant (no stream context)
 */
NppStatus nppiDivC_32f_C1R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f nConstant, 
                           Npp32f * pDst, int nDstStep, NppiSize oSizeROI)
{
    // Get default stream context
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    return nppiDivC_32f_C1R_Ctx(pSrc1, nSrc1Step, nConstant, pDst, nDstStep, 
                                oSizeROI, nppStreamCtx);
}

/**
 * 32-bit float, 1 channel in-place image divide constant
 */
NppStatus nppiDivC_32f_C1IR_Ctx(const Npp32f nConstant, Npp32f * pSrcDst, int nSrcDstStep, 
                                NppiSize oSizeROI, NppStreamContext nppStreamCtx)
{
    return nppiDivC_32f_C1R_Ctx(pSrcDst, nSrcDstStep, nConstant, pSrcDst, nSrcDstStep, 
                                oSizeROI, nppStreamCtx);
}

/**
 * 32-bit float, 1 channel in-place image divide constant (no stream context)
 */
NppStatus nppiDivC_32f_C1IR(const Npp32f nConstant, Npp32f * pSrcDst, int nSrcDstStep, 
                            NppiSize oSizeROI)
{
    // Get default stream context
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    return nppiDivC_32f_C1IR_Ctx(nConstant, pSrcDst, nSrcDstStep, 
                                 oSizeROI, nppStreamCtx);
}