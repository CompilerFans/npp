#include "npp.h"
#include <cuda_runtime.h>
#include <cstring>

/**
 * NPP Common Logarithm Operations Implementation
 * Computes base-10 logarithm of input image values: dst = log10(src)
 * 支持32位浮点数的常用对数运算
 */

// Forward declarations for CUDA implementations
extern "C" {
NppStatus nppiLog_32f_C1R_Ctx_cuda(const Npp32f* pSrc, int nSrcStep, Npp32f* pDst, int nDstStep,
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx);
}

/**
 * Validate common input parameters for log operations
 */
static inline NppStatus validateLogInputs(const void* pSrc, int nSrcStep, void* pDst, int nDstStep, NppiSize oSizeROI) {
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }
    
    if (nSrcStep <= 0 || nDstStep <= 0) {
        return NPP_STEP_ERROR;
    }
    
    if (!pSrc || !pDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    
    return NPP_SUCCESS;
}

/**
 * 32-bit float common logarithm (base 10)
 */
NppStatus nppiLog_32f_C1R_Ctx(const Npp32f* pSrc, int nSrcStep, Npp32f* pDst, int nDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    NppStatus status = validateLogInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    return nppiLog_32f_C1R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLog_32f_C1R(const Npp32f* pSrc, int nSrcStep, Npp32f* pDst, int nDstStep, NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiLog_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

/**
 * 32-bit float common logarithm - in place
 */
NppStatus nppiLog_32f_C1IR_Ctx(Npp32f* pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    NppStatus status = validateLogInputs(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    return nppiLog_32f_C1R_Ctx_cuda(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLog_32f_C1IR(Npp32f* pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiLog_32f_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}