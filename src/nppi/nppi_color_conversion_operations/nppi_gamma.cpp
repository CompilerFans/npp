#include "npp.h"
#include <cuda_runtime.h>

// ============================================================================
// Forward Declarations for CUDA implementations (_Ctx versions)
// 声明在.cu文件中实现的函数
// ============================================================================

extern "C" {
NppStatus nppiGammaFwd_8u_C3R_Ctx(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiGammaFwd_8u_C3IR_Ctx(Npp8u* pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiGammaInv_8u_C3R_Ctx(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiGammaInv_8u_C3IR_Ctx(Npp8u* pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiGammaFwd_8u_AC4R_Ctx(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiGammaFwd_8u_AC4IR_Ctx(Npp8u* pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiGammaInv_8u_AC4R_Ctx(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiGammaInv_8u_AC4IR_Ctx(Npp8u* pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
}

// ============================================================================
// Input Validation Helper
// 所有Gamma函数共用的参数验证逻辑
// ============================================================================

static inline NppStatus validateGammaInputs(const void* pSrc, int nSrcStep, void* pDst, int nDstStep, 
                                            NppiSize oSizeROI, int channels) {
    // 检查指针有效性
    if (!pSrc || !pDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    
    // 检查ROI大小
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }
    
    // 检查Step（行字节跨度）
    int min_step = oSizeROI.width * channels * sizeof(Npp8u);
    if (nSrcStep < min_step || nDstStep < min_step) {
        return NPP_STEP_ERROR;
    }
    
    return NPP_SUCCESS;
}

static inline NppStatus validateGammaInputsInplace(void* pSrcDst, int nSrcDstStep, 
                                                    NppiSize oSizeROI, int channels) {
    if (!pSrcDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }
    
    int min_step = oSizeROI.width * channels * sizeof(Npp8u);
    if (nSrcDstStep < min_step) {
        return NPP_STEP_ERROR;
    }
    
    return NPP_SUCCESS;
}

// ============================================================================
// Non-Ctx Versions (wrappers that call Ctx versions with default stream)
// 非Ctx版本：创建默认的NppStreamContext（stream=0），然后调用Ctx版本
// ============================================================================

NppStatus nppiGammaFwd_8u_C3R(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst, int nDstStep, NppiSize oSizeROI) {
    NppStatus status = validateGammaInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 3);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;  // 使用默认CUDA stream
    return nppiGammaFwd_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiGammaFwd_8u_C3IR(Npp8u* pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
    NppStatus status = validateGammaInputsInplace(pSrcDst, nSrcDstStep, oSizeROI, 3);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiGammaFwd_8u_C3IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiGammaInv_8u_C3R(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst, int nDstStep, NppiSize oSizeROI) {
    NppStatus status = validateGammaInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 3);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiGammaInv_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiGammaInv_8u_C3IR(Npp8u* pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
    NppStatus status = validateGammaInputsInplace(pSrcDst, nSrcDstStep, oSizeROI, 3);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiGammaInv_8u_C3IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiGammaFwd_8u_AC4R(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst, int nDstStep, NppiSize oSizeROI) {
    NppStatus status = validateGammaInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 4);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiGammaFwd_8u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiGammaFwd_8u_AC4IR(Npp8u* pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
    NppStatus status = validateGammaInputsInplace(pSrcDst, nSrcDstStep, oSizeROI, 4);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiGammaFwd_8u_AC4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiGammaInv_8u_AC4R(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst, int nDstStep, NppiSize oSizeROI) {
    NppStatus status = validateGammaInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 4);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiGammaInv_8u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiGammaInv_8u_AC4IR(Npp8u* pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
    NppStatus status = validateGammaInputsInplace(pSrcDst, nSrcDstStep, oSizeROI, 4);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiGammaInv_8u_AC4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}
