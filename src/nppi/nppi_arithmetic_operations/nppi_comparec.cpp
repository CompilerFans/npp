#include "npp.h"
#include <cuda_runtime.h>
#include <cstring>
#include <cstdio>

/**
 * NPP Image Comparison with Constant Functions Implementation
 * Implements nppiCompareC functions for comparing images with constant values
 */

// Forward declarations for CUDA implementations
extern "C" {
NppStatus nppiCompareC_8u_C1R_Ctx_cuda(const Npp8u* pSrc, int nSrcStep, const Npp8u nConstant,
                                       Npp8u* pDst, int nDstStep, NppiSize oSizeROI, 
                                       NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);
NppStatus nppiCompareC_16s_C1R_Ctx_cuda(const Npp16s* pSrc, int nSrcStep, const Npp16s nConstant,
                                        Npp8u* pDst, int nDstStep, NppiSize oSizeROI, 
                                        NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);
NppStatus nppiCompareC_32f_C1R_Ctx_cuda(const Npp32f* pSrc, int nSrcStep, const Npp32f nConstant,
                                        Npp8u* pDst, int nDstStep, NppiSize oSizeROI, 
                                        NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);
}

// Input validation helper
static inline NppStatus validateCompareCInputs(const void* pSrc, int nSrcStep, 
                                               void* pDst, int nDstStep, 
                                               NppiSize oSizeROI, NppCmpOp eComparisonOperation) {
    if (!pSrc || !pDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }
    
    if (nSrcStep <= 0 || nDstStep <= 0) {
        return NPP_STEP_ERROR;
    }
    
    // Validate comparison operation
    if (eComparisonOperation < NPP_CMP_LESS || eComparisonOperation > NPP_CMP_GREATER) {
        return NPP_BAD_ARGUMENT_ERROR;
    }
    
    return NPP_SUCCESS;
}

// 8-bit unsigned single channel compare with constant
NppStatus nppiCompareC_8u_C1R_Ctx(const Npp8u* pSrc, int nSrcStep, const Npp8u nConstant,
                                  Npp8u* pDst, int nDstStep, NppiSize oSizeROI, 
                                  NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx) {
    NppStatus status = validateCompareCInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eComparisonOperation);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    return nppiCompareC_8u_C1R_Ctx_cuda(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, eComparisonOperation, nppStreamCtx);
}

NppStatus nppiCompareC_8u_C1R(const Npp8u* pSrc, int nSrcStep, const Npp8u nConstant,
                             Npp8u* pDst, int nDstStep, NppiSize oSizeROI, 
                             NppCmpOp eComparisonOperation) {
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiCompareC_8u_C1R_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, eComparisonOperation, nppStreamCtx);
}

// 16-bit signed single channel compare with constant
NppStatus nppiCompareC_16s_C1R_Ctx(const Npp16s* pSrc, int nSrcStep, const Npp16s nConstant,
                                   Npp8u* pDst, int nDstStep, NppiSize oSizeROI, 
                                   NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx) {
    NppStatus status = validateCompareCInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eComparisonOperation);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    return nppiCompareC_16s_C1R_Ctx_cuda(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, eComparisonOperation, nppStreamCtx);
}

NppStatus nppiCompareC_16s_C1R(const Npp16s* pSrc, int nSrcStep, const Npp16s nConstant,
                              Npp8u* pDst, int nDstStep, NppiSize oSizeROI, 
                              NppCmpOp eComparisonOperation) {
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiCompareC_16s_C1R_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, eComparisonOperation, nppStreamCtx);
}

// 32-bit float single channel compare with constant
NppStatus nppiCompareC_32f_C1R_Ctx(const Npp32f* pSrc, int nSrcStep, const Npp32f nConstant,
                                   Npp8u* pDst, int nDstStep, NppiSize oSizeROI, 
                                   NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx) {
    NppStatus status = validateCompareCInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eComparisonOperation);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    return nppiCompareC_32f_C1R_Ctx_cuda(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, eComparisonOperation, nppStreamCtx);
}

NppStatus nppiCompareC_32f_C1R(const Npp32f* pSrc, int nSrcStep, const Npp32f nConstant,
                              Npp8u* pDst, int nDstStep, NppiSize oSizeROI, 
                              NppCmpOp eComparisonOperation) {
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiCompareC_32f_C1R_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, eComparisonOperation, nppStreamCtx);
}