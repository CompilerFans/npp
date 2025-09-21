#include "../../npp_internal.h"
#include "npp.h"
#include <cuda_runtime.h>

// Declare GPU functions
extern "C" {
NppStatus nppiCompressMarkerLabelsGetBufferSize_32u_C1R_Ctx_impl(int nMarkerLabels, int *hpBufferSize);
NppStatus nppiCompressMarkerLabelsUF_32u_C1IR_Ctx_impl(Npp32u *pMarkerLabels, int nMarkerLabelsStep,
                                                       NppiSize oMarkerLabelsROI, int nStartingNumber,
                                                       int *pNewMarkerLabelsNumber, Npp8u *pDeviceBuffer,
                                                       NppStreamContext nppStreamCtx);
}

// Get required buffer size
NppStatus nppiCompressMarkerLabelsGetBufferSize_32u_C1R(int nMarkerLabels, int *hpBufferSize) {
  if (hpBufferSize == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (nMarkerLabels <= 0) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  return nppiCompressMarkerLabelsGetBufferSize_32u_C1R_Ctx_impl(nMarkerLabels, hpBufferSize);
}

NppStatus nppiCompressMarkerLabelsGetBufferSize_32u_C1R_Ctx(int nMarkerLabels, int *hpBufferSize,
                                                            NppStreamContext nppStreamCtx) {
  // 使用流上下文参数以Avoid unused warning
  if (nppStreamCtx.nCudaDeviceId < -1) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  return nppiCompressMarkerLabelsGetBufferSize_32u_C1R(nMarkerLabels, hpBufferSize);
}

// Use Union-Find algorithm to compress marker labels
NppStatus nppiCompressMarkerLabelsUF_32u_C1IR(Npp32u *pMarkerLabels, int nMarkerLabelsStep, NppiSize oMarkerLabelsROI,
                                              int nStartingNumber, int *pNewMarkerLabelsNumber, Npp8u *pDeviceBuffer) {
  // Parameter validation
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

  return nppiCompressMarkerLabelsUF_32u_C1IR_Ctx_impl(pMarkerLabels, nMarkerLabelsStep, oMarkerLabelsROI,
                                                      nStartingNumber, pNewMarkerLabelsNumber, pDeviceBuffer,
                                                      nppStreamCtx);
}

NppStatus nppiCompressMarkerLabelsUF_32u_C1IR_Ctx(Npp32u *pMarkerLabels, int nMarkerLabelsStep,
                                                  NppiSize oMarkerLabelsROI, int nStartingNumber,
                                                  int *pNewMarkerLabelsNumber, Npp8u *pDeviceBuffer,
                                                  NppStreamContext nppStreamCtx) {
  // Parameter validation
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

  return nppiCompressMarkerLabelsUF_32u_C1IR_Ctx_impl(pMarkerLabels, nMarkerLabelsStep, oMarkerLabelsROI,
                                                      nStartingNumber, pNewMarkerLabelsNumber, pDeviceBuffer,
                                                      nppStreamCtx);
}
