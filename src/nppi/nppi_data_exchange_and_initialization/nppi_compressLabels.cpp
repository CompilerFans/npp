#include "../../npp_internal.h"
#include "npp.h"
#include <cuda_runtime.h>

// 声明CUDA函数
extern "C" {
NppStatus nppiCompressMarkerLabelsGetBufferSize_32u_C1R_Ctx_cuda(int nMarkerLabels, int *hpBufferSize);
NppStatus nppiCompressMarkerLabelsUF_32u_C1IR_Ctx_cuda(Npp32u *pMarkerLabels, int nMarkerLabelsStep,
                                                       NppiSize oMarkerLabelsROI, int nStartingNumber,
                                                       int *pNewMarkerLabelsNumber, Npp8u *pDeviceBuffer,
                                                       NppStreamContext nppStreamCtx);
}

// 获取标签压缩所需缓冲区大小
NppStatus nppiCompressMarkerLabelsGetBufferSize_32u_C1R(int nMarkerLabels, int *hpBufferSize) {
  if (hpBufferSize == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (nMarkerLabels <= 0) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  return nppiCompressMarkerLabelsGetBufferSize_32u_C1R_Ctx_cuda(nMarkerLabels, hpBufferSize);
}

NppStatus nppiCompressMarkerLabelsGetBufferSize_32u_C1R_Ctx(int nMarkerLabels, int *hpBufferSize,
                                                            NppStreamContext nppStreamCtx) {
  // 使用流上下文参数以避免未使用警告
  if (nppStreamCtx.nCudaDeviceId < -1) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  return nppiCompressMarkerLabelsGetBufferSize_32u_C1R(nMarkerLabels, hpBufferSize);
}

// 使用Union-Find算法压缩标记标签
NppStatus nppiCompressMarkerLabelsUF_32u_C1IR(Npp32u *pMarkerLabels, int nMarkerLabelsStep, NppiSize oMarkerLabelsROI,
                                              int nStartingNumber, int *pNewMarkerLabelsNumber, Npp8u *pDeviceBuffer) {
  // 参数验证
  if (pMarkerLabels == nullptr || pNewMarkerLabelsNumber == nullptr || pDeviceBuffer == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oMarkerLabelsROI.width <= 0 || oMarkerLabelsROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  if (nMarkerLabelsStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (nStartingNumber < 0) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  NppStreamContext nppStreamCtx = nppCreateDefaultStreamContext();

  return nppiCompressMarkerLabelsUF_32u_C1IR_Ctx_cuda(pMarkerLabels, nMarkerLabelsStep, oMarkerLabelsROI,
                                                      nStartingNumber, pNewMarkerLabelsNumber, pDeviceBuffer,
                                                      nppStreamCtx);
}

NppStatus nppiCompressMarkerLabelsUF_32u_C1IR_Ctx(Npp32u *pMarkerLabels, int nMarkerLabelsStep,
                                                  NppiSize oMarkerLabelsROI, int nStartingNumber,
                                                  int *pNewMarkerLabelsNumber, Npp8u *pDeviceBuffer,
                                                  NppStreamContext nppStreamCtx) {
  // 参数验证
  if (pMarkerLabels == nullptr || pNewMarkerLabelsNumber == nullptr || pDeviceBuffer == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oMarkerLabelsROI.width <= 0 || oMarkerLabelsROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  if (nMarkerLabelsStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (nStartingNumber < 0) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  return nppiCompressMarkerLabelsUF_32u_C1IR_Ctx_cuda(pMarkerLabels, nMarkerLabelsStep, oMarkerLabelsROI,
                                                      nStartingNumber, pNewMarkerLabelsNumber, pDeviceBuffer,
                                                      nppStreamCtx);
}