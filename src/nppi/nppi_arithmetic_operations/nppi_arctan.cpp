#include "npp.h"
#include <cuda_runtime.h>
#include <cstring>

/**
 * NPP Arctangent Operations Implementation
 * Computes arctangent of input image values: dst = arctan(src)
 * 支持32位浮点数的反正切运算
 */

// Forward declarations for CUDA implementations
extern "C" {
NppStatus nppiArcTan_32f_C1R_Ctx_cuda(const Npp32f* pSrc, int nSrcStep, Npp32f* pDst, int nDstStep,
                                       NppiSize oSizeROI, NppStreamContext nppStreamCtx);
}

/**
 * Validate common input parameters for arctan operations
 */
static inline NppStatus validateArcTanInputs(const void* pSrc, int nSrcStep, void* pDst, int nDstStep, NppiSize oSizeROI) {
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
 * 32-bit float arctangent
 */
NppStatus nppiArcTan_32f_C1R_Ctx(const Npp32f* pSrc, int nSrcStep, Npp32f* pDst, int nDstStep,
                                  NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    NppStatus status = validateArcTanInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    return nppiArcTan_32f_C1R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiArcTan_32f_C1R(const Npp32f* pSrc, int nSrcStep, Npp32f* pDst, int nDstStep, NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiArcTan_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

/**
 * 32-bit float arctangent - in place
 */
NppStatus nppiArcTan_32f_C1IR_Ctx(Npp32f* pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    NppStatus status = validateArcTanInputs(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    return nppiArcTan_32f_C1R_Ctx_cuda(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiArcTan_32f_C1IR(Npp32f* pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiArcTan_32f_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}