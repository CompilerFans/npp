#include "npp.h"
#include <cuda_runtime.h>
#include <cstring>
#include <cstdio>

/**
 * NPP Image Absolute Difference Functions Implementation
 * Implements nppiAbsDiff functions for computing absolute difference between two images
 */

// Forward declarations for CUDA implementations
extern "C" {
NppStatus nppiAbsDiff_8u_C1R_Ctx_cuda(const Npp8u* pSrc1, int nSrc1Step, const Npp8u* pSrc2, int nSrc2Step,
                                      Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiAbsDiff_8u_C3R_Ctx_cuda(const Npp8u* pSrc1, int nSrc1Step, const Npp8u* pSrc2, int nSrc2Step,
                                      Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiAbsDiff_32f_C1R_Ctx_cuda(const Npp32f* pSrc1, int nSrc1Step, const Npp32f* pSrc2, int nSrc2Step,
                                       Npp32f* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
}

// Input validation helper
static inline NppStatus validateAbsDiffInputs(const void* pSrc1, int nSrc1Step, 
                                              const void* pSrc2, int nSrc2Step,
                                              void* pDst, int nDstStep, 
                                              NppiSize oSizeROI) {
    // 按NPP标准的错误检查优先级顺序
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }
    
    if (nSrc1Step <= 0 || nSrc2Step <= 0 || nDstStep <= 0) {
        return NPP_STEP_ERROR;
    }
    
    if (!pSrc1 || !pSrc2 || !pDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    
    return NPP_SUCCESS;
}

// 8-bit unsigned single channel absolute difference
NppStatus nppiAbsDiff_8u_C1R_Ctx(const Npp8u* pSrc1, int nSrc1Step,
                                 const Npp8u* pSrc2, int nSrc2Step,
                                 Npp8u* pDst, int nDstStep,
                                 NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    NppStatus status = validateAbsDiffInputs(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    return nppiAbsDiff_8u_C1R_Ctx_cuda(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiff_8u_C1R(const Npp8u* pSrc1, int nSrc1Step,
                            const Npp8u* pSrc2, int nSrc2Step,
                            Npp8u* pDst, int nDstStep,
                            NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiAbsDiff_8u_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// 8-bit unsigned three channel absolute difference
NppStatus nppiAbsDiff_8u_C3R_Ctx(const Npp8u* pSrc1, int nSrc1Step,
                                 const Npp8u* pSrc2, int nSrc2Step,
                                 Npp8u* pDst, int nDstStep,
                                 NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    NppStatus status = validateAbsDiffInputs(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    return nppiAbsDiff_8u_C3R_Ctx_cuda(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiff_8u_C3R(const Npp8u* pSrc1, int nSrc1Step,
                            const Npp8u* pSrc2, int nSrc2Step,
                            Npp8u* pDst, int nDstStep,
                            NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiAbsDiff_8u_C3R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}


// 32-bit float single channel absolute difference
NppStatus nppiAbsDiff_32f_C1R_Ctx(const Npp32f* pSrc1, int nSrc1Step,
                                  const Npp32f* pSrc2, int nSrc2Step,
                                  Npp32f* pDst, int nDstStep,
                                  NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    NppStatus status = validateAbsDiffInputs(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    return nppiAbsDiff_32f_C1R_Ctx_cuda(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiff_32f_C1R(const Npp32f* pSrc1, int nSrc1Step,
                             const Npp32f* pSrc2, int nSrc2Step,
                             Npp32f* pDst, int nDstStep,
                             NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiAbsDiff_32f_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}