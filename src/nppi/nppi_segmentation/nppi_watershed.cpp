#include "npp.h"
#include <cuda_runtime.h>

// 声明CUDA函数
extern "C" {
    NppStatus nppiSegmentWatershedGetBufferSize_8u_C1R_Ctx_cuda(NppiSize oSizeROI, int* hpBufferSize);
    NppStatus nppiSegmentWatershed_8u_C1IR_Ctx_cuda(const Npp8u* pSrc, int nSrcStep,
                                                    Npp32s* pMarkers, int nMarkersStep,
                                                    NppiSize oSizeROI, Npp8u eNorm,
                                                    Npp8u* pDeviceBuffer, NppStreamContext nppStreamCtx);
}

// 获取Watershed分割所需缓冲区大小
NppStatus nppiSegmentWatershedGetBufferSize_8u_C1R(NppiSize oSizeROI, int* hpBufferSize) {
    if (hpBufferSize == nullptr) {
        return NPP_NULL_POINTER_ERROR;
    }

    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }

    return nppiSegmentWatershedGetBufferSize_8u_C1R_Ctx_cuda(oSizeROI, hpBufferSize);
}

NppStatus nppiSegmentWatershedGetBufferSize_8u_C1R_Ctx(NppiSize oSizeROI, int* hpBufferSize,
                                                      NppStreamContext nppStreamCtx) {
    return nppiSegmentWatershedGetBufferSize_8u_C1R(oSizeROI, hpBufferSize);
}

// Watershed图像分割
NppStatus nppiSegmentWatershed_8u_C1IR(const Npp8u* pSrc, int nSrcStep,
                                      Npp32s* pMarkers, int nMarkersStep,
                                      NppiSize oSizeROI, Npp8u eNorm,
                                      Npp8u* pDeviceBuffer) {
    // 参数验证
    if (pSrc == nullptr || pMarkers == nullptr || pDeviceBuffer == nullptr) {
        return NPP_NULL_POINTER_ERROR;
    }

    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }

    if (nSrcStep <= 0 || nMarkersStep <= 0) {
        return NPP_STEP_ERROR;
    }

    if (eNorm != 1 && eNorm != 2) {  // L1或L2范数
        return NPP_BAD_ARGUMENT_ERROR;
    }

    NppStreamContext nppStreamCtx = {0};
    nppStreamCtx.hStream = 0; // 默认流

    return nppiSegmentWatershed_8u_C1IR_Ctx_cuda(pSrc, nSrcStep, pMarkers, nMarkersStep,
                                                oSizeROI, eNorm, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiSegmentWatershed_8u_C1IR_Ctx(const Npp8u* pSrc, int nSrcStep,
                                          Npp32s* pMarkers, int nMarkersStep,
                                          NppiSize oSizeROI, Npp8u eNorm,
                                          Npp8u* pDeviceBuffer, NppStreamContext nppStreamCtx) {
    // 参数验证
    if (pSrc == nullptr || pMarkers == nullptr || pDeviceBuffer == nullptr) {
        return NPP_NULL_POINTER_ERROR;
    }

    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }

    if (nSrcStep <= 0 || nMarkersStep <= 0) {
        return NPP_STEP_ERROR;
    }

    if (eNorm != 1 && eNorm != 2) {  // L1或L2范数
        return NPP_BAD_ARGUMENT_ERROR;
    }

    return nppiSegmentWatershed_8u_C1IR_Ctx_cuda(pSrc, nSrcStep, pMarkers, nMarkersStep,
                                                oSizeROI, eNorm, pDeviceBuffer, nppStreamCtx);
}