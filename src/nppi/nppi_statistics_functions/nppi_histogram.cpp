#include "npp.h"
#include "npp_version_compat.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

// Forward declarations for mpp host func implementations
extern "C" {
NppStatus nppiHistogramEvenGetBufferSize_8u_C1R_Ctx_impl(NppiSize oSizeROI, int nLevels, size_t *hpBufferSize);
NppStatus nppiHistogramEven_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist,
                                            int nLevels, Npp32s nLowerLevel, Npp32s nUpperLevel, Npp8u *pDeviceBuffer,
                                            NppStreamContext nppStreamCtx);
NppStatus nppiHistogramEven_16u_C1R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist,
                                             int nLevels, Npp32s nLowerLevel, Npp32s nUpperLevel, Npp8u *pDeviceBuffer,
                                             NppStreamContext nppStreamCtx);
NppStatus nppiHistogramEven_16s_C1R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist,
                                             int nLevels, Npp32s nLowerLevel, Npp32s nUpperLevel, Npp8u *pDeviceBuffer,
                                             NppStreamContext nppStreamCtx);
NppStatus nppiHistogramEven_8u_C4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[4],
                                            int nLevels[4], Npp32s nLowerLevel[4], Npp32s nUpperLevel[4],
                                            Npp8u *pDeviceBuffer, NppStreamContext nppStreamCtx);
NppStatus nppiHistogramRange_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist,
                                             const Npp32s *pLevels, int nLevels, Npp8u *pDeviceBuffer,
                                             NppStreamContext nppStreamCtx);
NppStatus nppiHistogramRange_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist,
                                              const Npp32f *pLevels, int nLevels, Npp8u *pDeviceBuffer,
                                              NppStreamContext nppStreamCtx);
}

// Internal helper function for buffer size calculation (uses size_t internally)
static NppStatus nppiHistogramEvenGetBufferSize_internal(NppiSize oSizeROI, int nLevels, int *hpBufferSize) {
  // Validate input parameters
  if (!hpBufferSize) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  // NVIDIA NPP is permissive with nLevels, but we keep some basic validation
  if (nLevels < 0) {
    return NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR;
  }

  // Use size_t internally for accurate calculation
  size_t temp_size;
  NppStatus status = nppiHistogramEvenGetBufferSize_8u_C1R_Ctx_impl(oSizeROI, nLevels, &temp_size);

  if (status == NPP_SUCCESS) {
    // Convert size_t to int (with overflow check)
    if (temp_size > static_cast<size_t>(INT_MAX)) {
      return NPP_MEMORY_ALLOCATION_ERR;
    }
    *hpBufferSize = static_cast<int>(temp_size);
  }

  return status;
}

//=============================================================================
// 8-bit unsigned single channel Buffer Size APIs
//=============================================================================

NppStatus nppiHistogramEvenGetBufferSize_8u_C1R(NppiSize oSizeROI, int nLevels, int *hpBufferSize) {
  return nppiHistogramEvenGetBufferSize_internal(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramEvenGetBufferSize_8u_C1R_Ctx(NppiSize oSizeROI, int nLevels, int *hpBufferSize,
                                                    NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx; // Stream context not used for buffer size calculation
  return nppiHistogramEvenGetBufferSize_internal(oSizeROI, nLevels, hpBufferSize);
}

//=============================================================================
// 8-bit unsigned 4-channel Buffer Size APIs
//=============================================================================

NppStatus nppiHistogramEvenGetBufferSize_8u_C4R(NppiSize oSizeROI, int nLevels[4], int *hpBufferSize) {
  // Validate input parameters
  if (!hpBufferSize || !nLevels) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  // For 4-channel, calculate buffer size based on maximum levels across channels
  int maxLevels = nLevels[0];
  for (int i = 1; i < 4; i++) {
    if (nLevels[i] > maxLevels) {
      maxLevels = nLevels[i];
    }
  }

  // Use maximum levels for buffer calculation and multiply by 4 for channels
  size_t singleChannelSize;
  NppStatus status = nppiHistogramEvenGetBufferSize_8u_C1R_Ctx_impl(oSizeROI, maxLevels, &singleChannelSize);
  if (status == NPP_SUCCESS) {
    size_t totalSize = singleChannelSize * 4; // 4 channels
    if (totalSize > static_cast<size_t>(INT_MAX)) {
      return NPP_MEMORY_ALLOCATION_ERR;
    }
    *hpBufferSize = static_cast<int>(totalSize);
  }
  return status;
}

NppStatus nppiHistogramEvenGetBufferSize_8u_C4R_Ctx(NppiSize oSizeROI, int nLevels[4], int *hpBufferSize,
                                                    NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx; // Stream context not used for buffer size calculation
  return nppiHistogramEvenGetBufferSize_8u_C4R(oSizeROI, nLevels, hpBufferSize);
}

//=============================================================================
// 16-bit unsigned single channel Buffer Size APIs
//=============================================================================

NppStatus nppiHistogramEvenGetBufferSize_16u_C1R(NppiSize oSizeROI, int nLevels, int *hpBufferSize) {
  // For 16u, use same calculation as 8u since the buffer requirements are similar
  return nppiHistogramEvenGetBufferSize_internal(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramEvenGetBufferSize_16u_C1R_Ctx(NppiSize oSizeROI, int nLevels, int *hpBufferSize,
                                                     NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppiHistogramEvenGetBufferSize_16u_C1R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramEvenGetBufferSize_16u_C4R(NppiSize oSizeROI, int nLevels[4], int *hpBufferSize) {
  // For 16u C4R, use same logic as 8u C4R
  return nppiHistogramEvenGetBufferSize_8u_C4R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramEvenGetBufferSize_16u_C4R_Ctx(NppiSize oSizeROI, int nLevels[4], int *hpBufferSize,
                                                     NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppiHistogramEvenGetBufferSize_16u_C4R(oSizeROI, nLevels, hpBufferSize);
}

//=============================================================================
// 16-bit signed single channel Buffer Size APIs
//=============================================================================

NppStatus nppiHistogramEvenGetBufferSize_16s_C1R(NppiSize oSizeROI, int nLevels, int *hpBufferSize) {
  return nppiHistogramEvenGetBufferSize_internal(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramEvenGetBufferSize_16s_C1R_Ctx(NppiSize oSizeROI, int nLevels, int *hpBufferSize,
                                                     NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppiHistogramEvenGetBufferSize_16s_C1R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramEvenGetBufferSize_16s_C4R(NppiSize oSizeROI, int nLevels[4], int *hpBufferSize) {
  return nppiHistogramEvenGetBufferSize_8u_C4R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramEvenGetBufferSize_16s_C4R_Ctx(NppiSize oSizeROI, int nLevels[4], int *hpBufferSize,
                                                     NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppiHistogramEvenGetBufferSize_16s_C4R(oSizeROI, nLevels, hpBufferSize);
}

//=============================================================================
// Histogram Range Buffer Size APIs
//=============================================================================

NppStatus nppiHistogramRangeGetBufferSize_8u_C1R(NppiSize oSizeROI, int nLevels, int *hpBufferSize) {
  // Range histogram has similar buffer requirements as even histogram
  return nppiHistogramEvenGetBufferSize_internal(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramRangeGetBufferSize_8u_C1R_Ctx(NppiSize oSizeROI, int nLevels, int *hpBufferSize,
                                                     NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppiHistogramRangeGetBufferSize_8u_C1R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramRangeGetBufferSize_32f_C1R(NppiSize oSizeROI, int nLevels, int *hpBufferSize) {
  return nppiHistogramEvenGetBufferSize_internal(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramRangeGetBufferSize_32f_C1R_Ctx(NppiSize oSizeROI, int nLevels, int *hpBufferSize,
                                                      NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppiHistogramRangeGetBufferSize_32f_C1R(oSizeROI, nLevels, hpBufferSize);
}

//=============================================================================
// HistogramRange Multi-channel Buffer Size APIs - C3R variants
//=============================================================================

NppStatus nppiHistogramRangeGetBufferSize_8u_C3R(NppiSize oSizeROI, int nLevels[3], int *hpBufferSize) {
  return nppiHistogramEvenGetBufferSize_8u_C3R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramRangeGetBufferSize_8u_C3R_Ctx(NppiSize oSizeROI, int nLevels[3], int *hpBufferSize,
                                                     NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppiHistogramRangeGetBufferSize_8u_C3R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramRangeGetBufferSize_16u_C3R(NppiSize oSizeROI, int nLevels[3], int *hpBufferSize) {
  return nppiHistogramEvenGetBufferSize_8u_C3R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramRangeGetBufferSize_16u_C3R_Ctx(NppiSize oSizeROI, int nLevels[3], int *hpBufferSize,
                                                      NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppiHistogramRangeGetBufferSize_16u_C3R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramRangeGetBufferSize_16s_C3R(NppiSize oSizeROI, int nLevels[3], int *hpBufferSize) {
  return nppiHistogramEvenGetBufferSize_8u_C3R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramRangeGetBufferSize_16s_C3R_Ctx(NppiSize oSizeROI, int nLevels[3], int *hpBufferSize,
                                                      NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppiHistogramRangeGetBufferSize_16s_C3R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramRangeGetBufferSize_32f_C3R(NppiSize oSizeROI, int nLevels[3], int *hpBufferSize) {
  return nppiHistogramEvenGetBufferSize_8u_C3R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramRangeGetBufferSize_32f_C3R_Ctx(NppiSize oSizeROI, int nLevels[3], int *hpBufferSize,
                                                      NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppiHistogramRangeGetBufferSize_32f_C3R(oSizeROI, nLevels, hpBufferSize);
}

//=============================================================================
// HistogramRange Multi-channel Buffer Size APIs - C4R variants
//=============================================================================

NppStatus nppiHistogramRangeGetBufferSize_8u_C4R(NppiSize oSizeROI, int nLevels[4], int *hpBufferSize) {
  return nppiHistogramEvenGetBufferSize_8u_C4R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramRangeGetBufferSize_8u_C4R_Ctx(NppiSize oSizeROI, int nLevels[4], int *hpBufferSize,
                                                     NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppiHistogramRangeGetBufferSize_8u_C4R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramRangeGetBufferSize_16u_C4R(NppiSize oSizeROI, int nLevels[4], int *hpBufferSize) {
  return nppiHistogramEvenGetBufferSize_8u_C4R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramRangeGetBufferSize_16u_C4R_Ctx(NppiSize oSizeROI, int nLevels[4], int *hpBufferSize,
                                                      NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppiHistogramRangeGetBufferSize_16u_C4R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramRangeGetBufferSize_16s_C4R(NppiSize oSizeROI, int nLevels[4], int *hpBufferSize) {
  return nppiHistogramEvenGetBufferSize_8u_C4R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramRangeGetBufferSize_16s_C4R_Ctx(NppiSize oSizeROI, int nLevels[4], int *hpBufferSize,
                                                      NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppiHistogramRangeGetBufferSize_16s_C4R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramRangeGetBufferSize_32f_C4R(NppiSize oSizeROI, int nLevels[4], int *hpBufferSize) {
  return nppiHistogramEvenGetBufferSize_8u_C4R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramRangeGetBufferSize_32f_C4R_Ctx(NppiSize oSizeROI, int nLevels[4], int *hpBufferSize,
                                                      NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppiHistogramRangeGetBufferSize_32f_C4R(oSizeROI, nLevels, hpBufferSize);
}

//=============================================================================
// HistogramRange Multi-channel Buffer Size APIs - AC4R variants
//=============================================================================

NppStatus nppiHistogramRangeGetBufferSize_8u_AC4R(NppiSize oSizeROI, int nLevels[3], int *hpBufferSize) {
  return nppiHistogramEvenGetBufferSize_8u_AC4R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramRangeGetBufferSize_8u_AC4R_Ctx(NppiSize oSizeROI, int nLevels[3], int *hpBufferSize,
                                                      NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppiHistogramRangeGetBufferSize_8u_AC4R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramRangeGetBufferSize_16u_AC4R(NppiSize oSizeROI, int nLevels[3], int *hpBufferSize) {
  return nppiHistogramEvenGetBufferSize_8u_AC4R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramRangeGetBufferSize_16u_AC4R_Ctx(NppiSize oSizeROI, int nLevels[3], int *hpBufferSize,
                                                       NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppiHistogramRangeGetBufferSize_16u_AC4R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramRangeGetBufferSize_16s_AC4R(NppiSize oSizeROI, int nLevels[3], int *hpBufferSize) {
  return nppiHistogramEvenGetBufferSize_8u_AC4R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramRangeGetBufferSize_16s_AC4R_Ctx(NppiSize oSizeROI, int nLevels[3], int *hpBufferSize,
                                                       NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppiHistogramRangeGetBufferSize_16s_AC4R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramRangeGetBufferSize_32f_AC4R(NppiSize oSizeROI, int nLevels[3], int *hpBufferSize) {
  return nppiHistogramEvenGetBufferSize_8u_AC4R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramRangeGetBufferSize_32f_AC4R_Ctx(NppiSize oSizeROI, int nLevels[3], int *hpBufferSize,
                                                       NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppiHistogramRangeGetBufferSize_32f_AC4R(oSizeROI, nLevels, hpBufferSize);
}

//=============================================================================
// Multi-channel Buffer Size APIs - C3R variants
//=============================================================================

NppStatus nppiHistogramEvenGetBufferSize_8u_C3R(NppiSize oSizeROI, int nLevels[3], int *hpBufferSize) {
  // Validate input parameters
  if (!hpBufferSize || !nLevels) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  // For 3-channel, calculate buffer size based on maximum levels across channels
  int maxLevels = nLevels[0];
  for (int i = 1; i < 3; i++) {
    if (nLevels[i] > maxLevels) {
      maxLevels = nLevels[i];
    }
  }

  // Use maximum levels for buffer calculation and multiply by 3 for channels
  size_t singleChannelSize;
  NppStatus status = nppiHistogramEvenGetBufferSize_8u_C1R_Ctx_impl(oSizeROI, maxLevels, &singleChannelSize);
  if (status == NPP_SUCCESS) {
    size_t totalSize = singleChannelSize * 3; // 3 channels
    if (totalSize > static_cast<size_t>(INT_MAX)) {
      return NPP_MEMORY_ALLOCATION_ERR;
    }
    *hpBufferSize = static_cast<int>(totalSize);
  }
  return status;
}

NppStatus nppiHistogramEvenGetBufferSize_8u_C3R_Ctx(NppiSize oSizeROI, int nLevels[3], int *hpBufferSize,
                                                    NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx; // Stream context not used for buffer size calculation
  return nppiHistogramEvenGetBufferSize_8u_C3R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramEvenGetBufferSize_16u_C3R(NppiSize oSizeROI, int nLevels[3], int *hpBufferSize) {
  return nppiHistogramEvenGetBufferSize_8u_C3R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramEvenGetBufferSize_16u_C3R_Ctx(NppiSize oSizeROI, int nLevels[3], int *hpBufferSize,
                                                     NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppiHistogramEvenGetBufferSize_16u_C3R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramEvenGetBufferSize_16s_C3R(NppiSize oSizeROI, int nLevels[3], int *hpBufferSize) {
  return nppiHistogramEvenGetBufferSize_8u_C3R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramEvenGetBufferSize_16s_C3R_Ctx(NppiSize oSizeROI, int nLevels[3], int *hpBufferSize,
                                                     NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppiHistogramEvenGetBufferSize_16s_C3R(oSizeROI, nLevels, hpBufferSize);
}

//=============================================================================
// Multi-channel Buffer Size APIs - AC4R variants
//=============================================================================

NppStatus nppiHistogramEvenGetBufferSize_8u_AC4R(NppiSize oSizeROI, int nLevels[3], int *hpBufferSize) {
  // AC4R uses only 3 channels (ignores alpha), so same as C3R
  return nppiHistogramEvenGetBufferSize_8u_C3R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramEvenGetBufferSize_8u_AC4R_Ctx(NppiSize oSizeROI, int nLevels[3], int *hpBufferSize,
                                                     NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppiHistogramEvenGetBufferSize_8u_AC4R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramEvenGetBufferSize_16u_AC4R(NppiSize oSizeROI, int nLevels[3], int *hpBufferSize) {
  return nppiHistogramEvenGetBufferSize_8u_AC4R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramEvenGetBufferSize_16u_AC4R_Ctx(NppiSize oSizeROI, int nLevels[3], int *hpBufferSize,
                                                      NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppiHistogramEvenGetBufferSize_16u_AC4R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramEvenGetBufferSize_16s_AC4R(NppiSize oSizeROI, int nLevels[3], int *hpBufferSize) {
  return nppiHistogramEvenGetBufferSize_8u_AC4R(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramEvenGetBufferSize_16s_AC4R_Ctx(NppiSize oSizeROI, int nLevels[3], int *hpBufferSize,
                                                      NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppiHistogramEvenGetBufferSize_16s_AC4R(oSizeROI, nLevels, hpBufferSize);
}

//=============================================================================
// Histogram Computation APIs
//=============================================================================

NppStatus nppiHistogramEven_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist, int nLevels,
                                       Npp32s nLowerLevel, Npp32s nUpperLevel, Npp8u *pDeviceBuffer,
                                       NppStreamContext nppStreamCtx) {
  // Validate input parameters
  if (!pSrc || !pHist || !pDeviceBuffer) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep < oSizeROI.width) {
    return NPP_STEP_ERROR;
  }

  if (nLowerLevel >= nUpperLevel) {
    return NPP_RANGE_ERROR;
  }

  // Call GPU kernel implementation
  return nppiHistogramEven_8u_C1R_Ctx_impl(pSrc, nSrcStep, oSizeROI, pHist, nLevels, nLowerLevel, nUpperLevel,
                                           pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiHistogramEven_8u_C1R(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist, int nLevels,
                                   Npp32s nLowerLevel, Npp32s nUpperLevel, Npp8u *pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiHistogramEven_8u_C1R_Ctx(pSrc, nSrcStep, oSizeROI, pHist, nLevels, nLowerLevel, nUpperLevel, pDeviceBuffer,
                                      nppStreamCtx);
}

//=============================================================================
// 16-bit unsigned HistogramEven Computation APIs
//=============================================================================

NppStatus nppiHistogramEven_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist, int nLevels,
                                        Npp32s nLowerLevel, Npp32s nUpperLevel, Npp8u *pDeviceBuffer,
                                        NppStreamContext nppStreamCtx) {
  // Validate input parameters
  if (!pSrc || !pHist || !pDeviceBuffer) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep < static_cast<int>(oSizeROI.width * sizeof(Npp16u))) {
    return NPP_STEP_ERROR;
  }

  if (nLowerLevel >= nUpperLevel) {
    return NPP_RANGE_ERROR;
  }

  return nppiHistogramEven_16u_C1R_Ctx_impl(pSrc, nSrcStep, oSizeROI, pHist, nLevels, nLowerLevel, nUpperLevel,
                                            pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiHistogramEven_16u_C1R(const Npp16u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist, int nLevels,
                                    Npp32s nLowerLevel, Npp32s nUpperLevel, Npp8u *pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiHistogramEven_16u_C1R_Ctx(pSrc, nSrcStep, oSizeROI, pHist, nLevels, nLowerLevel, nUpperLevel,
                                       pDeviceBuffer, nppStreamCtx);
}

//=============================================================================
// 16-bit signed HistogramEven Computation APIs
//=============================================================================

NppStatus nppiHistogramEven_16s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist, int nLevels,
                                        Npp32s nLowerLevel, Npp32s nUpperLevel, Npp8u *pDeviceBuffer,
                                        NppStreamContext nppStreamCtx) {
  if (!pSrc || !pHist || !pDeviceBuffer) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep < static_cast<int>(oSizeROI.width * sizeof(Npp16s))) {
    return NPP_STEP_ERROR;
  }

  if (nLowerLevel >= nUpperLevel) {
    return NPP_RANGE_ERROR;
  }

  return nppiHistogramEven_16s_C1R_Ctx_impl(pSrc, nSrcStep, oSizeROI, pHist, nLevels, nLowerLevel, nUpperLevel,
                                            pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiHistogramEven_16s_C1R(const Npp16s *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist, int nLevels,
                                    Npp32s nLowerLevel, Npp32s nUpperLevel, Npp8u *pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiHistogramEven_16s_C1R_Ctx(pSrc, nSrcStep, oSizeROI, pHist, nLevels, nLowerLevel, nUpperLevel,
                                       pDeviceBuffer, nppStreamCtx);
}

//=============================================================================
// 8-bit unsigned 4-channel HistogramEven Computation APIs
//=============================================================================

NppStatus nppiHistogramEven_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[4],
                                       int nLevels[4], Npp32s nLowerLevel[4], Npp32s nUpperLevel[4],
                                       Npp8u *pDeviceBuffer, NppStreamContext nppStreamCtx) {
  // Validate input parameters
  if (!pSrc || !pHist || !pDeviceBuffer || !nLevels || !nLowerLevel || !nUpperLevel) {
    return NPP_NULL_POINTER_ERROR;
  }

  // Validate individual channel histogram pointers
  for (int c = 0; c < 4; c++) {
    if (!pHist[c]) {
      return NPP_NULL_POINTER_ERROR;
    }
  }

  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep < static_cast<int>(oSizeROI.width * 4 * sizeof(Npp8u))) {
    return NPP_STEP_ERROR;
  }

  // Validate range for each channel
  for (int c = 0; c < 4; c++) {
    if (nLowerLevel[c] >= nUpperLevel[c]) {
      return NPP_RANGE_ERROR;
    }
  }

  return nppiHistogramEven_8u_C4R_Ctx_impl(pSrc, nSrcStep, oSizeROI, pHist, nLevels, nLowerLevel, nUpperLevel,
                                           pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiHistogramEven_8u_C4R(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[4], int nLevels[4],
                                   Npp32s nLowerLevel[4], Npp32s nUpperLevel[4], Npp8u *pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiHistogramEven_8u_C4R_Ctx(pSrc, nSrcStep, oSizeROI, pHist, nLevels, nLowerLevel, nUpperLevel, pDeviceBuffer,
                                      nppStreamCtx);
}

//=============================================================================
// Utility Functions
//=============================================================================

// Host function to generate evenly spaced histogram levels
NppStatus nppiEvenLevelsHost_32s(Npp32s *pLevels, int nLevels, Npp32s nLowerBound, Npp32s nUpperBound) {
  // Validate input parameters
  if (pLevels == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (nLevels < 2) {
    return NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR;
  }

  if (nLowerBound >= nUpperBound) {
    return NPP_RANGE_ERROR;
  }

  // Generate evenly spaced levels
  for (int i = 0; i < nLevels; i++) {
    double ratio = static_cast<double>(i) / static_cast<double>(nLevels - 1);
    pLevels[i] = static_cast<Npp32s>(nLowerBound + ratio * (nUpperBound - nLowerBound));
  }

  return NPP_SUCCESS;
}

//=============================================================================
// HistogramRange Computation APIs
//=============================================================================

NppStatus nppiHistogramRange_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist,
                                        const Npp32s *pLevels, int nLevels, Npp8u *pDeviceBuffer,
                                        NppStreamContext nppStreamCtx) {
  // Validate input parameters
  if (!pSrc || !pHist || !pLevels || !pDeviceBuffer) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep < oSizeROI.width) {
    return NPP_STEP_ERROR;
  }

  if (nLevels < 2) {
    return NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR;
  }

  // Note: Cannot validate level ordering here as pLevels is a device pointer

  return nppiHistogramRange_8u_C1R_Ctx_impl(pSrc, nSrcStep, oSizeROI, pHist, pLevels, nLevels, pDeviceBuffer,
                                            nppStreamCtx);
}

NppStatus nppiHistogramRange_8u_C1R(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist,
                                    const Npp32s *pLevels, int nLevels, Npp8u *pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiHistogramRange_8u_C1R_Ctx(pSrc, nSrcStep, oSizeROI, pHist, pLevels, nLevels, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiHistogramRange_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist,
                                         const Npp32f *pLevels, int nLevels, Npp8u *pDeviceBuffer,
                                         NppStreamContext nppStreamCtx) {
  if (!pSrc || !pHist || !pLevels || !pDeviceBuffer) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep < static_cast<int>(oSizeROI.width * sizeof(Npp32f))) {
    return NPP_STEP_ERROR;
  }

  if (nLevels < 2) {
    return NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR;
  }

  // Note: Cannot validate level ordering here as pLevels is a device pointer

  return nppiHistogramRange_32f_C1R_Ctx_impl(pSrc, nSrcStep, oSizeROI, pHist, pLevels, nLevels, pDeviceBuffer,
                                             nppStreamCtx);
}

NppStatus nppiHistogramRange_32f_C1R(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist,
                                     const Npp32f *pLevels, int nLevels, Npp8u *pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiHistogramRange_32f_C1R_Ctx(pSrc, nSrcStep, oSizeROI, pHist, pLevels, nLevels, pDeviceBuffer, nppStreamCtx);
}