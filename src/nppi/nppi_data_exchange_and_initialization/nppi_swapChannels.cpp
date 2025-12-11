#include "npp.h"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

// Forward declarations for implementation functions
extern "C" {
NppStatus nppiSwapChannels_8u_C4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep,
                                            Npp8u *pDst, int nDstStep,
                                            NppiSize oSizeROI,
                                            const int aDstOrder[4],
                                            NppStreamContext nppStreamCtx);

NppStatus nppiSwapChannels_8u_C4IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep,
                                             NppiSize oSizeROI,
                                             const int aDstOrder[4],
                                             NppStreamContext nppStreamCtx);
}

// Input validation helper
static inline NppStatus validateSwapChannelsInputs(const void *pPtr, int nStep,
                                                    NppiSize oSizeROI,
                                                    const int aDstOrder[4]) {
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  if (nStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (!pPtr || !aDstOrder) {
    return NPP_NULL_POINTER_ERROR;
  }

  // Validate aDstOrder values are in range [0, 3]
  for (int i = 0; i < 4; i++) {
    if (aDstOrder[i] < 0 || aDstOrder[i] > 3) {
      return NPP_CHANNEL_ORDER_ERROR;
    }
  }

  return NPP_SUCCESS;
}

// ============================================================================
// nppiSwapChannels_8u_C4R - 4 channel swap with separate src/dst
// ============================================================================

NppStatus nppiSwapChannels_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                       Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       const int aDstOrder[4],
                                       NppStreamContext nppStreamCtx) {
  NppStatus status = validateSwapChannelsInputs(pSrc, nSrcStep, oSizeROI, aDstOrder);
  if (status != NPP_SUCCESS) {
    return status;
  }

  if (!pDst || nDstStep <= 0) {
    return NPP_NULL_POINTER_ERROR;
  }

  return nppiSwapChannels_8u_C4R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep,
                                           oSizeROI, aDstOrder, nppStreamCtx);
}

NppStatus nppiSwapChannels_8u_C4R(const Npp8u *pSrc, int nSrcStep,
                                   Npp8u *pDst, int nDstStep,
                                   NppiSize oSizeROI,
                                   const int aDstOrder[4]) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiSwapChannels_8u_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep,
                                      oSizeROI, aDstOrder, nppStreamCtx);
}

// ============================================================================
// nppiSwapChannels_8u_C4IR - 4 channel in-place swap
// ============================================================================

NppStatus nppiSwapChannels_8u_C4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        const int aDstOrder[4],
                                        NppStreamContext nppStreamCtx) {
  NppStatus status = validateSwapChannelsInputs(pSrcDst, nSrcDstStep, oSizeROI, aDstOrder);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiSwapChannels_8u_C4IR_Ctx_impl(pSrcDst, nSrcDstStep, oSizeROI,
                                            aDstOrder, nppStreamCtx);
}

NppStatus nppiSwapChannels_8u_C4IR(Npp8u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI,
                                    const int aDstOrder[4]) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiSwapChannels_8u_C4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI,
                                       aDstOrder, nppStreamCtx);
}
