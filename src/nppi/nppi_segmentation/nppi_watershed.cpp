#include "npp.h"
#include <cuda_runtime.h>

// 声明CUDA函数
extern "C" {
    NppStatus nppiSegmentWatershedGetBufferSize_8u_C1R_Ctx_cuda(NppiSize oSizeROI, size_t* hpBufferSize);
    NppStatus nppiSegmentWatershed_8u_C1IR_Ctx_cuda(Npp8u* pSrcDst, Npp32s nSrcDstStep,
                                                    Npp32u* pMarkerLabels, Npp32s nMarkerLabelsStep,
                                                    NppiNorm eNorm, NppiSize oSizeROI,
                                                    Npp8u* pDeviceBuffer, NppStreamContext nppStreamCtx);
}

// 获取Watershed分割所需缓冲区大小
NppStatus nppiSegmentWatershedGetBufferSize_8u_C1R(NppiSize oSizeROI, size_t* hpDeviceMemoryBufferSize) {
    if (hpDeviceMemoryBufferSize == nullptr) {
        return NPP_NULL_POINTER_ERROR;
    }

    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }

    return nppiSegmentWatershedGetBufferSize_8u_C1R_Ctx_cuda(oSizeROI, hpDeviceMemoryBufferSize);
}

// Watershed图像分割
NppStatus nppiSegmentWatershed_8u_C1IR(Npp8u* pSrcDst, Npp32s nSrcDstStep, Npp32u* pMarkerLabels, Npp32s nMarkerLabelsStep, NppiNorm eNorm, 
                                       NppiWatershedSegmentBoundaryType eSegmentBoundaryType, NppiSize oSizeROI, Npp8u* pDeviceMemoryBuffer) {
    // 参数验证
    if (pSrcDst == nullptr || pMarkerLabels == nullptr || pDeviceMemoryBuffer == nullptr) {
        return NPP_NULL_POINTER_ERROR;
    }

    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }

    if (nSrcDstStep <= 0 || nMarkerLabelsStep <= 0) {
        return NPP_STEP_ERROR;
    }

    NppStreamContext nppStreamCtx = {0};
    nppStreamCtx.hStream = 0; // 默认流

    return nppiSegmentWatershed_8u_C1IR_Ctx_cuda(pSrcDst, nSrcDstStep, pMarkerLabels, nMarkerLabelsStep,
                                                eNorm, oSizeROI, pDeviceMemoryBuffer, nppStreamCtx);
}

NppStatus nppiSegmentWatershed_8u_C1IR_Ctx(Npp8u* pSrcDst, Npp32s nSrcDstStep, Npp32u* pMarkerLabels, Npp32s nMarkerLabelsStep, NppiNorm eNorm, 
                                          NppiWatershedSegmentBoundaryType eSegmentBoundaryType, NppiSize oSizeROI, Npp8u* pDeviceMemoryBuffer, NppStreamContext nppStreamCtx) {
    // 参数验证
    if (pSrcDst == nullptr || pMarkerLabels == nullptr || pDeviceMemoryBuffer == nullptr) {
        return NPP_NULL_POINTER_ERROR;
    }

    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }

    if (nSrcDstStep <= 0 || nMarkerLabelsStep <= 0) {
        return NPP_STEP_ERROR;
    }

    return nppiSegmentWatershed_8u_C1IR_Ctx_cuda(pSrcDst, nSrcDstStep, pMarkerLabels, nMarkerLabelsStep,
                                                eNorm, oSizeROI, pDeviceMemoryBuffer, nppStreamCtx);
}