#include "npp.h"
#include <cuda_runtime.h>
#include <cstring>

/**
 * NPP Power Operations Implementation
 * Computes power of input image values: dst = pow(src, power)
 * 支持整数类型的幂运算和浮点类型的任意幂
 */

// Forward declarations for CUDA implementations
extern "C" {
NppStatus nppiPow_8u_C1RSfs_Ctx_cuda(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst, int nDstStep,
                                      NppiSize oSizeROI, int nPower, int nScaleFactor, NppStreamContext nppStreamCtx);
NppStatus nppiPow_16u_C1RSfs_Ctx_cuda(const Npp16u* pSrc, int nSrcStep, Npp16u* pDst, int nDstStep,
                                       NppiSize oSizeROI, int nPower, int nScaleFactor, NppStreamContext nppStreamCtx);
NppStatus nppiPow_16s_C1RSfs_Ctx_cuda(const Npp16s* pSrc, int nSrcStep, Npp16s* pDst, int nDstStep,
                                       NppiSize oSizeROI, int nPower, int nScaleFactor, NppStreamContext nppStreamCtx);
NppStatus nppiPow_32f_C1R_Ctx_cuda(const Npp32f* pSrc, int nSrcStep, Npp32f* pDst, int nDstStep,
                                    NppiSize oSizeROI, Npp32f nPower, NppStreamContext nppStreamCtx);
}

/**
 * Validate common input parameters for pow operations
 */
static inline NppStatus validatePowInputs(const void* pSrc, int nSrcStep, void* pDst, int nDstStep, NppiSize oSizeROI) {
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
 * 8-bit unsigned power with scaling
 */
NppStatus nppiPow_8u_C1RSfs_Ctx(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst, int nDstStep,
                                 NppiSize oSizeROI, int nPower, int nScaleFactor, NppStreamContext nppStreamCtx) {
    NppStatus status = validatePowInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    if (nScaleFactor < 0) {
        return NPP_BAD_ARGUMENT_ERROR;
    }
    
    return nppiPow_8u_C1RSfs_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nPower, nScaleFactor, nppStreamCtx);
}

NppStatus nppiPow_8u_C1RSfs(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst, int nDstStep,
                             NppiSize oSizeROI, int nPower, int nScaleFactor) {
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiPow_8u_C1RSfs_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nPower, nScaleFactor, nppStreamCtx);
}

/**
 * 8-bit unsigned power with scaling - in place
 */
NppStatus nppiPow_8u_C1IRSfs_Ctx(Npp8u* pSrcDst, int nSrcDstStep, NppiSize oSizeROI, 
                                  int nPower, int nScaleFactor, NppStreamContext nppStreamCtx) {
    NppStatus status = validatePowInputs(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    if (nScaleFactor < 0) {
        return NPP_BAD_ARGUMENT_ERROR;
    }
    
    return nppiPow_8u_C1RSfs_Ctx_cuda(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, nPower, nScaleFactor, nppStreamCtx);
}

NppStatus nppiPow_8u_C1IRSfs(Npp8u* pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nPower, int nScaleFactor) {
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiPow_8u_C1IRSfs_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nPower, nScaleFactor, nppStreamCtx);
}

/**
 * 16-bit unsigned power with scaling
 */
NppStatus nppiPow_16u_C1RSfs_Ctx(const Npp16u* pSrc, int nSrcStep, Npp16u* pDst, int nDstStep,
                                  NppiSize oSizeROI, int nPower, int nScaleFactor, NppStreamContext nppStreamCtx) {
    NppStatus status = validatePowInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    if (nScaleFactor < 0) {
        return NPP_BAD_ARGUMENT_ERROR;
    }
    
    return nppiPow_16u_C1RSfs_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nPower, nScaleFactor, nppStreamCtx);
}

NppStatus nppiPow_16u_C1RSfs(const Npp16u* pSrc, int nSrcStep, Npp16u* pDst, int nDstStep,
                              NppiSize oSizeROI, int nPower, int nScaleFactor) {
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiPow_16u_C1RSfs_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nPower, nScaleFactor, nppStreamCtx);
}

/**
 * 16-bit unsigned power with scaling - in place
 */
NppStatus nppiPow_16u_C1IRSfs_Ctx(Npp16u* pSrcDst, int nSrcDstStep, NppiSize oSizeROI, 
                                   int nPower, int nScaleFactor, NppStreamContext nppStreamCtx) {
    NppStatus status = validatePowInputs(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    if (nScaleFactor < 0) {
        return NPP_BAD_ARGUMENT_ERROR;
    }
    
    return nppiPow_16u_C1RSfs_Ctx_cuda(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, nPower, nScaleFactor, nppStreamCtx);
}

NppStatus nppiPow_16u_C1IRSfs(Npp16u* pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nPower, int nScaleFactor) {
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiPow_16u_C1IRSfs_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nPower, nScaleFactor, nppStreamCtx);
}

/**
 * 16-bit signed power with scaling
 */
NppStatus nppiPow_16s_C1RSfs_Ctx(const Npp16s* pSrc, int nSrcStep, Npp16s* pDst, int nDstStep,
                                  NppiSize oSizeROI, int nPower, int nScaleFactor, NppStreamContext nppStreamCtx) {
    NppStatus status = validatePowInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    if (nScaleFactor < 0) {
        return NPP_BAD_ARGUMENT_ERROR;
    }
    
    return nppiPow_16s_C1RSfs_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nPower, nScaleFactor, nppStreamCtx);
}

NppStatus nppiPow_16s_C1RSfs(const Npp16s* pSrc, int nSrcStep, Npp16s* pDst, int nDstStep,
                              NppiSize oSizeROI, int nPower, int nScaleFactor) {
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiPow_16s_C1RSfs_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nPower, nScaleFactor, nppStreamCtx);
}

/**
 * 16-bit signed power with scaling - in place
 */
NppStatus nppiPow_16s_C1IRSfs_Ctx(Npp16s* pSrcDst, int nSrcDstStep, NppiSize oSizeROI, 
                                   int nPower, int nScaleFactor, NppStreamContext nppStreamCtx) {
    NppStatus status = validatePowInputs(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    if (nScaleFactor < 0) {
        return NPP_BAD_ARGUMENT_ERROR;
    }
    
    return nppiPow_16s_C1RSfs_Ctx_cuda(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, nPower, nScaleFactor, nppStreamCtx);
}

NppStatus nppiPow_16s_C1IRSfs(Npp16s* pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nPower, int nScaleFactor) {
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiPow_16s_C1IRSfs_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nPower, nScaleFactor, nppStreamCtx);
}

/**
 * 32-bit float power (no scaling needed)
 */
NppStatus nppiPow_32f_C1R_Ctx(const Npp32f* pSrc, int nSrcStep, Npp32f* pDst, int nDstStep,
                               NppiSize oSizeROI, Npp32f nPower, NppStreamContext nppStreamCtx) {
    NppStatus status = validatePowInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    return nppiPow_32f_C1R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nPower, nppStreamCtx);
}

NppStatus nppiPow_32f_C1R(const Npp32f* pSrc, int nSrcStep, Npp32f* pDst, int nDstStep, NppiSize oSizeROI, Npp32f nPower) {
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiPow_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nPower, nppStreamCtx);
}

/**
 * 32-bit float power - in place
 */
NppStatus nppiPow_32f_C1IR_Ctx(Npp32f* pSrcDst, int nSrcDstStep, NppiSize oSizeROI, Npp32f nPower, NppStreamContext nppStreamCtx) {
    NppStatus status = validatePowInputs(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    return nppiPow_32f_C1R_Ctx_cuda(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, nPower, nppStreamCtx);
}

NppStatus nppiPow_32f_C1IR(Npp32f* pSrcDst, int nSrcDstStep, NppiSize oSizeROI, Npp32f nPower) {
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiPow_32f_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nPower, nppStreamCtx);
}