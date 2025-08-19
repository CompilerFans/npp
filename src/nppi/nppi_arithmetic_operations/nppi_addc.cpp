#include "npp.h"
#include <cuda_runtime.h>
#include <cstring>

/**
 * NPP Image Add Constant Functions Implementation
 * Implements nppiAddC functions for various data types
 */

// Forward declarations for CUDA implementations
extern "C" {
NppStatus nppiAddC_8u_C1RSfs_Ctx_cuda(const Npp8u* pSrc1, int nSrc1Step, const Npp8u nConstant,
                                      Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                      NppStreamContext nppStreamCtx);

NppStatus nppiAddC_16u_C1RSfs_Ctx_cuda(const Npp16u* pSrc1, int nSrc1Step, const Npp16u nConstant,
                                       Npp16u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx);

NppStatus nppiAddC_16s_C1RSfs_Ctx_cuda(const Npp16s* pSrc1, int nSrc1Step, const Npp16s nConstant,
                                       Npp16s* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx);

NppStatus nppiAddC_32f_C1R_Ctx_cuda(const Npp32f* pSrc1, int nSrc1Step, const Npp32f nConstant,
                                    Npp32f* pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);

NppStatus nppiAddC_8u_C3RSfs_Ctx_cuda(const Npp8u* pSrc1, int nSrc1Step, const Npp8u aConstants[3],
                                      Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
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
    
    // 与NVIDIA NPP兼容：zero width/height返回成功，负数返回错误
    if (oSizeROI.width < 0 || oSizeROI.height < 0) {
        return NPP_SIZE_ERROR;
    }
    
    // 对于zero尺寸，step验证不适用
    if (oSizeROI.width > 0 && oSizeROI.height > 0) {
        if (nSrcStep < oSizeROI.width || nDstStep < oSizeROI.width) {
            return NPP_STRIDE_ERROR;
        }
    }
    
    if (nScaleFactor < 0 || nScaleFactor > 31) {
        return NPP_BAD_ARGUMENT_ERROR;
    }
    
    return NPP_NO_ERROR;
}

/**
 * 8-bit unsigned, 1 channel image add constant with scale
 */
NppStatus nppiAddC_8u_C1RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u nConstant, 
                                 Npp8u * pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor, 
                                 NppStreamContext nppStreamCtx)
{
    // Validate parameters
    NppStatus status = validateParameters(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor);
    if (status != NPP_NO_ERROR) {
        return status;
    }
    
    // 如果ROI尺寸为0，直接返回成功（与NVIDIA NPP兼容）
    if (oSizeROI.width == 0 || oSizeROI.height == 0) {
        return NPP_NO_ERROR;
    }
    
    // Call CUDA implementation
    return nppiAddC_8u_C1RSfs_Ctx_cuda(pSrc1, nSrc1Step, nConstant, pDst, nDstStep, 
                                       oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 8-bit unsigned, 1 channel image add constant with scale (no stream context)
 */
NppStatus nppiAddC_8u_C1RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u nConstant, 
                             Npp8u * pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    // Get default stream context
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    return nppiAddC_8u_C1RSfs_Ctx(pSrc1, nSrc1Step, nConstant, pDst, nDstStep, 
                                  oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 8-bit unsigned, 1 channel in-place image add constant with scale
 */
NppStatus nppiAddC_8u_C1IRSfs_Ctx(const Npp8u nConstant, Npp8u * pSrcDst, int nSrcDstStep, 
                                  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx)
{
    // For in-place operation, source and destination are the same
    return nppiAddC_8u_C1RSfs_Ctx(pSrcDst, nSrcDstStep, nConstant, pSrcDst, nSrcDstStep, 
                                  oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 8-bit unsigned, 1 channel in-place image add constant with scale (no stream context)
 */
NppStatus nppiAddC_8u_C1IRSfs(const Npp8u nConstant, Npp8u * pSrcDst, int nSrcDstStep, 
                              NppiSize oSizeROI, int nScaleFactor)
{
    // Get default stream context
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    return nppiAddC_8u_C1IRSfs_Ctx(nConstant, pSrcDst, nSrcDstStep, 
                                   oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 8-bit unsigned, 3 channel image add constant with scale
 */
NppStatus nppiAddC_8u_C3RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u aConstants[3], 
                                 Npp8u * pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor, 
                                 NppStreamContext nppStreamCtx)
{
    // Validate parameters - for 3-channel, each pixel is 3 bytes
    if (!pSrc1 || !pDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    
    // 与NVIDIA NPP兼容：zero width/height返回成功，负数返回错误
    if (oSizeROI.width < 0 || oSizeROI.height < 0) {
        return NPP_SIZE_ERROR;
    }
    
    // 如果ROI尺寸为0，直接返回成功（与NVIDIA NPP兼容）
    if (oSizeROI.width == 0 || oSizeROI.height == 0) {
        return NPP_NO_ERROR;
    }
    
    // 对于非零尺寸，验证step
    if (nSrc1Step < oSizeROI.width * 3 || nDstStep < oSizeROI.width * 3) {
        return NPP_STRIDE_ERROR;
    }
    
    if (nScaleFactor < 0 || nScaleFactor > 31) {
        return NPP_BAD_ARGUMENT_ERROR;
    }
    
    if (!aConstants) {
        return NPP_NULL_POINTER_ERROR;
    }
    
    // Call CUDA implementation
    return nppiAddC_8u_C3RSfs_Ctx_cuda(pSrc1, nSrc1Step, aConstants, pDst, nDstStep, 
                                       oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 8-bit unsigned, 3 channel image add constant with scale (no stream context)
 */
NppStatus nppiAddC_8u_C3RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u aConstants[3], 
                             Npp8u * pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    // Get default stream context
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    return nppiAddC_8u_C3RSfs_Ctx(pSrc1, nSrc1Step, aConstants, pDst, nDstStep, 
                                  oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 8-bit unsigned, 3 channel in-place image add constant with scale
 */
NppStatus nppiAddC_8u_C3IRSfs_Ctx(const Npp8u aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, 
                                  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx)
{
    return nppiAddC_8u_C3RSfs_Ctx(pSrcDst, nSrcDstStep, aConstants, pSrcDst, nSrcDstStep, 
                                  oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 8-bit unsigned, 3 channel in-place image add constant with scale (no stream context)
 */
NppStatus nppiAddC_8u_C3IRSfs(const Npp8u aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, 
                              NppiSize oSizeROI, int nScaleFactor)
{
    // Get default stream context
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    return nppiAddC_8u_C3IRSfs_Ctx(aConstants, pSrcDst, nSrcDstStep, 
                                   oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 16-bit unsigned, 1 channel image add constant with scale
 */
NppStatus nppiAddC_16u_C1RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u nConstant, 
                                  Npp16u * pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor, 
                                  NppStreamContext nppStreamCtx)
{
    // Validate parameters
    NppStatus status = validateParameters(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor);
    if (status != NPP_NO_ERROR) {
        return status;
    }
    
    // Call CUDA implementation
    return nppiAddC_16u_C1RSfs_Ctx_cuda(pSrc1, nSrc1Step, nConstant, pDst, nDstStep, 
                                        oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 16-bit unsigned, 1 channel image add constant with scale (no stream context)
 */
NppStatus nppiAddC_16u_C1RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u nConstant, 
                              Npp16u * pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    // Get default stream context
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    return nppiAddC_16u_C1RSfs_Ctx(pSrc1, nSrc1Step, nConstant, pDst, nDstStep, 
                                   oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 16-bit unsigned, 1 channel in-place image add constant with scale
 */
NppStatus nppiAddC_16u_C1IRSfs_Ctx(const Npp16u nConstant, Npp16u * pSrcDst, int nSrcDstStep, 
                                   NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx)
{
    return nppiAddC_16u_C1RSfs_Ctx(pSrcDst, nSrcDstStep, nConstant, pSrcDst, nSrcDstStep, 
                                   oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 16-bit unsigned, 1 channel in-place image add constant with scale (no stream context)
 */
NppStatus nppiAddC_16u_C1IRSfs(const Npp16u nConstant, Npp16u * pSrcDst, int nSrcDstStep, 
                               NppiSize oSizeROI, int nScaleFactor)
{
    // Get default stream context
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    return nppiAddC_16u_C1IRSfs_Ctx(nConstant, pSrcDst, nSrcDstStep, 
                                    oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 16-bit signed, 1 channel image add constant with scale
 */
NppStatus nppiAddC_16s_C1RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s nConstant, 
                                  Npp16s * pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor, 
                                  NppStreamContext nppStreamCtx)
{
    // Validate parameters
    NppStatus status = validateParameters(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor);
    if (status != NPP_NO_ERROR) {
        return status;
    }
    
    // Call CUDA implementation
    return nppiAddC_16s_C1RSfs_Ctx_cuda(pSrc1, nSrc1Step, nConstant, pDst, nDstStep, 
                                        oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 16-bit signed, 1 channel image add constant with scale (no stream context)
 */
NppStatus nppiAddC_16s_C1RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s nConstant, 
                              Npp16s * pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    // Get default stream context
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    return nppiAddC_16s_C1RSfs_Ctx(pSrc1, nSrc1Step, nConstant, pDst, nDstStep, 
                                   oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 16-bit signed, 1 channel in-place image add constant with scale
 */
NppStatus nppiAddC_16s_C1IRSfs_Ctx(const Npp16s nConstant, Npp16s * pSrcDst, int nSrcDstStep, 
                                   NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx)
{
    return nppiAddC_16s_C1RSfs_Ctx(pSrcDst, nSrcDstStep, nConstant, pSrcDst, nSrcDstStep, 
                                   oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 16-bit signed, 1 channel in-place image add constant with scale (no stream context)
 */
NppStatus nppiAddC_16s_C1IRSfs(const Npp16s nConstant, Npp16s * pSrcDst, int nSrcDstStep, 
                               NppiSize oSizeROI, int nScaleFactor)
{
    // Get default stream context
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    return nppiAddC_16s_C1IRSfs_Ctx(nConstant, pSrcDst, nSrcDstStep, 
                                    oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 32-bit float, 1 channel image add constant (no scaling)
 */
NppStatus nppiAddC_32f_C1R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f nConstant, 
                               Npp32f * pDst, int nDstStep, NppiSize oSizeROI, 
                               NppStreamContext nppStreamCtx)
{
    // Validate parameters (no scale factor for float)
    if (!pSrc1 || !pDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    
    // 与NVIDIA NPP兼容：zero width/height返回成功，负数返回错误
    if (oSizeROI.width < 0 || oSizeROI.height < 0) {
        return NPP_SIZE_ERROR;
    }
    
    // 如果ROI尺寸为0，直接返回成功（与NVIDIA NPP兼容）
    if (oSizeROI.width == 0 || oSizeROI.height == 0) {
        return NPP_NO_ERROR;
    }
    
    // 对于非零尺寸，验证step
    if (nSrc1Step < static_cast<int>(oSizeROI.width * sizeof(Npp32f)) || nDstStep < static_cast<int>(oSizeROI.width * sizeof(Npp32f))) {
        return NPP_STRIDE_ERROR;
    }
    
    // Call CUDA implementation
    return nppiAddC_32f_C1R_Ctx_cuda(pSrc1, nSrc1Step, nConstant, pDst, nDstStep, 
                                     oSizeROI, nppStreamCtx);
}

/**
 * 32-bit float, 1 channel image add constant (no stream context)
 */
NppStatus nppiAddC_32f_C1R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f nConstant, 
                           Npp32f * pDst, int nDstStep, NppiSize oSizeROI)
{
    // Get default stream context
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    return nppiAddC_32f_C1R_Ctx(pSrc1, nSrc1Step, nConstant, pDst, nDstStep, 
                                oSizeROI, nppStreamCtx);
}

/**
 * 32-bit float, 1 channel in-place image add constant
 */
NppStatus nppiAddC_32f_C1IR_Ctx(const Npp32f nConstant, Npp32f * pSrcDst, int nSrcDstStep, 
                                NppiSize oSizeROI, NppStreamContext nppStreamCtx)
{
    return nppiAddC_32f_C1R_Ctx(pSrcDst, nSrcDstStep, nConstant, pSrcDst, nSrcDstStep, 
                                oSizeROI, nppStreamCtx);
}

/**
 * 32-bit float, 1 channel in-place image add constant (no stream context)
 */
NppStatus nppiAddC_32f_C1IR(const Npp32f nConstant, Npp32f * pSrcDst, int nSrcDstStep, 
                            NppiSize oSizeROI)
{
    // Get default stream context
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    return nppiAddC_32f_C1IR_Ctx(nConstant, pSrcDst, nSrcDstStep, 
                                 oSizeROI, nppStreamCtx);
}

// TODO: Implement 4-channel variants (C4, AC4)
// TODO: Implement device constant versions
// TODO: Implement 3-channel variants for other data types