#include "npp.h"
#include <cuda_runtime.h>

// 声明CUDA函数
extern "C" {
    NppStatus nppiFilterCannyBorderGetBufferSize_8u_C1R_Ctx_cuda(NppiSize oSizeROI, int* hpBufferSize);
    NppStatus nppiFilterCannyBorder_8u_C1R_Ctx_cuda(const Npp8u* pSrc, int nSrcStep, 
                                                     NppiSize oSrcSizeROI, NppiPoint oSrcOffset,
                                                     Npp8u* pDst, int nDstStep, NppiSize oDstSizeROI,
                                                     NppiMaskSize eMaskSize, Npp32f nLowThreshold, 
                                                     Npp32f nHighThreshold, NppiBorderType eBorderType,
                                                     Npp8u* pDeviceBuffer, NppStreamContext nppStreamCtx);
}

// 获取Canny边缘检测所需缓冲区大小
NppStatus nppiFilterCannyBorderGetBufferSize_8u_C1R(NppiSize oSizeROI, int* hpBufferSize) {
    if (hpBufferSize == nullptr) {
        return NPP_NULL_POINTER_ERROR;
    }

    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }

    return nppiFilterCannyBorderGetBufferSize_8u_C1R_Ctx_cuda(oSizeROI, hpBufferSize);
}

NppStatus nppiFilterCannyBorderGetBufferSize_8u_C1R_Ctx(NppiSize oSizeROI, int* hpBufferSize, 
                                                        NppStreamContext nppStreamCtx) {
    return nppiFilterCannyBorderGetBufferSize_8u_C1R(oSizeROI, hpBufferSize);
}

// Canny边缘检测实现
NppStatus nppiFilterCannyBorder_8u_C1R(const Npp8u* pSrc, int nSrcStep, NppiSize oSrcSizeROI, 
                                       NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, 
                                       NppiSize oDstSizeROI, NppiMaskSize eMaskSize, 
                                       Npp32f nLowThreshold, Npp32f nHighThreshold, 
                                       NppiBorderType eBorderType, Npp8u* pDeviceBuffer) {
    // 参数验证
    if (pSrc == nullptr || pDst == nullptr || pDeviceBuffer == nullptr) {
        return NPP_NULL_POINTER_ERROR;
    }

    if (oSrcSizeROI.width <= 0 || oSrcSizeROI.height <= 0 ||
        oDstSizeROI.width <= 0 || oDstSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }

    if (nSrcStep <= 0 || nDstStep <= 0) {
        return NPP_STEP_ERROR;
    }

    if (eMaskSize != NPP_MASK_SIZE_3_X_3 && eMaskSize != NPP_MASK_SIZE_5_X_5) {
        return NPP_MASK_SIZE_ERROR;
    }

    if (nLowThreshold < 0.0f || nHighThreshold < 0.0f || nLowThreshold >= nHighThreshold) {
        return NPP_BAD_ARGUMENT_ERROR;
    }

    if (eBorderType != NPP_BORDER_REPLICATE && eBorderType != NPP_BORDER_CONSTANT) {
        return NPP_BAD_ARGUMENT_ERROR;
    }

    NppStreamContext nppStreamCtx = {0};
    nppStreamCtx.hStream = 0; // 默认流

    return nppiFilterCannyBorder_8u_C1R_Ctx_cuda(pSrc, nSrcStep, oSrcSizeROI, oSrcOffset,
                                                 pDst, nDstStep, oDstSizeROI, eMaskSize,
                                                 nLowThreshold, nHighThreshold, eBorderType,
                                                 pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiFilterCannyBorder_8u_C1R_Ctx(const Npp8u* pSrc, int nSrcStep, 
                                           NppiSize oSrcSizeROI, NppiPoint oSrcOffset,
                                           Npp8u* pDst, int nDstStep, NppiSize oDstSizeROI,
                                           NppiMaskSize eMaskSize, Npp32f nLowThreshold, 
                                           Npp32f nHighThreshold, NppiBorderType eBorderType,
                                           Npp8u* pDeviceBuffer, NppStreamContext nppStreamCtx) {
    // 参数验证
    if (pSrc == nullptr || pDst == nullptr || pDeviceBuffer == nullptr) {
        return NPP_NULL_POINTER_ERROR;
    }

    if (oSrcSizeROI.width <= 0 || oSrcSizeROI.height <= 0 ||
        oDstSizeROI.width <= 0 || oDstSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }

    if (nSrcStep <= 0 || nDstStep <= 0) {
        return NPP_STEP_ERROR;
    }

    if (eMaskSize != NPP_MASK_SIZE_3_X_3 && eMaskSize != NPP_MASK_SIZE_5_X_5) {
        return NPP_MASK_SIZE_ERROR;
    }

    if (nLowThreshold < 0.0f || nHighThreshold < 0.0f || nLowThreshold >= nHighThreshold) {
        return NPP_BAD_ARGUMENT_ERROR;
    }

    if (eBorderType != NPP_BORDER_REPLICATE && eBorderType != NPP_BORDER_CONSTANT) {
        return NPP_BAD_ARGUMENT_ERROR;
    }

    return nppiFilterCannyBorder_8u_C1R_Ctx_cuda(pSrc, nSrcStep, oSrcSizeROI, oSrcOffset,
                                                 pDst, nDstStep, oDstSizeROI, eMaskSize,
                                                 nLowThreshold, nHighThreshold, eBorderType,
                                                 pDeviceBuffer, nppStreamCtx);
}