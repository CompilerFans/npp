#include "npp.h"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

/**
 * NPP Image Absolute Value Functions Implementation
 * Implements nppiAbs functions for computing absolute values
 */

// Forward declarations for CUDA implementations
extern "C" {
// 16s implementations
NppStatus nppiAbs_16s_C1R_Ctx_cuda(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
NppStatus nppiAbs_16s_C1IR_Ctx_cuda(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiAbs_16s_C3R_Ctx_cuda(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
NppStatus nppiAbs_16s_C3IR_Ctx_cuda(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiAbs_16s_AC4R_Ctx_cuda(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
NppStatus nppiAbs_16s_AC4IR_Ctx_cuda(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
NppStatus nppiAbs_16s_C4R_Ctx_cuda(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
NppStatus nppiAbs_16s_C4IR_Ctx_cuda(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

// 16f implementations
NppStatus nppiAbs_16f_C1R_Ctx_cuda(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
NppStatus nppiAbs_16f_C1IR_Ctx_cuda(Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiAbs_16f_C3R_Ctx_cuda(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
NppStatus nppiAbs_16f_C3IR_Ctx_cuda(Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiAbs_16f_C4R_Ctx_cuda(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
NppStatus nppiAbs_16f_C4IR_Ctx_cuda(Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

// 32f implementations
NppStatus nppiAbs_32f_C1R_Ctx_cuda(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
NppStatus nppiAbs_32f_C1IR_Ctx_cuda(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiAbs_32f_C3R_Ctx_cuda(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
NppStatus nppiAbs_32f_C3IR_Ctx_cuda(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiAbs_32f_AC4R_Ctx_cuda(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
NppStatus nppiAbs_32f_AC4IR_Ctx_cuda(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
NppStatus nppiAbs_32f_C4R_Ctx_cuda(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
NppStatus nppiAbs_32f_C4IR_Ctx_cuda(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
}

/**
 * Helper function for parameter validation
 */
static NppStatus validateAbsParameters(const void *pSrc, int nSrcStep, const void *pDst, int nDstStep,
                                       NppiSize oSizeROI, int nChannels, int nElementSize) {
  // For in-place operations, pDst will be nullptr
  if (!pSrc || (pDst == nullptr && nDstStep != 0)) {
    return NPP_NULL_POINTER_ERROR;
  }

  // NVIDIA NPP behavior: zero-size ROI returns success (no processing needed)
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  // Early return for zero-size ROI (NVIDIA NPP compatible behavior)
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }

  int minSrcStep = oSizeROI.width * nChannels * nElementSize;
  if (nSrcStep < minSrcStep) {
    return NPP_STRIDE_ERROR;
  }

  // For not-in-place operations
  if (pDst) {
    int minDstStep = oSizeROI.width * nChannels * nElementSize;
    if (nDstStep < minDstStep) {
      return NPP_STRIDE_ERROR;
    }
  }

  return NPP_NO_ERROR;
}

// ============================================================================
// 16-bit signed integer implementations
// ============================================================================

NppStatus nppiAbs_16s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx) {
  NppStatus status = validateAbsParameters(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 1, sizeof(Npp16s));
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiAbs_16s_C1R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_C1R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiAbs_16s_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_C1IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateAbsParameters(pSrcDst, nSrcDstStep, nullptr, 0, oSizeROI, 1, sizeof(Npp16s));
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiAbs_16s_C1IR_Ctx_cuda(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_C1IR(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiAbs_16s_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_C3R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx) {
  NppStatus status = validateAbsParameters(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 3, sizeof(Npp16s));
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiAbs_16s_C3R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_C3R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiAbs_16s_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_C3IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateAbsParameters(pSrcDst, nSrcDstStep, nullptr, 0, oSizeROI, 3, sizeof(Npp16s));
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiAbs_16s_C3IR_Ctx_cuda(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_C3IR(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiAbs_16s_C3IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_AC4R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx) {
  NppStatus status = validateAbsParameters(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 4, sizeof(Npp16s));
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiAbs_16s_AC4R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_AC4R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiAbs_16s_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_AC4IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateAbsParameters(pSrcDst, nSrcDstStep, nullptr, 0, oSizeROI, 4, sizeof(Npp16s));
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiAbs_16s_AC4IR_Ctx_cuda(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_AC4IR(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiAbs_16s_AC4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_C4R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx) {
  NppStatus status = validateAbsParameters(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 4, sizeof(Npp16s));
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiAbs_16s_C4R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_C4R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiAbs_16s_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_C4IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateAbsParameters(pSrcDst, nSrcDstStep, nullptr, 0, oSizeROI, 4, sizeof(Npp16s));
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiAbs_16s_C4IR_Ctx_cuda(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_C4IR(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiAbs_16s_C4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

// ============================================================================
// 16-bit float implementations
// ============================================================================

NppStatus nppiAbs_16f_C1R_Ctx(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst, int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx) {
  NppStatus status = validateAbsParameters(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 1, sizeof(Npp16f));
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiAbs_16f_C1R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16f_C1R(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiAbs_16f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16f_C1IR_Ctx(Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateAbsParameters(pSrcDst, nSrcDstStep, nullptr, 0, oSizeROI, 1, sizeof(Npp16f));
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiAbs_16f_C1IR_Ctx_cuda(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16f_C1IR(Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiAbs_16f_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16f_C3R_Ctx(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst, int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx) {
  NppStatus status = validateAbsParameters(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 3, sizeof(Npp16f));
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiAbs_16f_C3R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16f_C3R(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiAbs_16f_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16f_C3IR_Ctx(Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateAbsParameters(pSrcDst, nSrcDstStep, nullptr, 0, oSizeROI, 3, sizeof(Npp16f));
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiAbs_16f_C3IR_Ctx_cuda(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16f_C3IR(Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiAbs_16f_C3IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16f_C4R_Ctx(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst, int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx) {
  NppStatus status = validateAbsParameters(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 4, sizeof(Npp16f));
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiAbs_16f_C4R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16f_C4R(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiAbs_16f_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16f_C4IR_Ctx(Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateAbsParameters(pSrcDst, nSrcDstStep, nullptr, 0, oSizeROI, 4, sizeof(Npp16f));
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiAbs_16f_C4IR_Ctx_cuda(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16f_C4IR(Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiAbs_16f_C4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

// ============================================================================
// 32-bit float implementations
// ============================================================================

NppStatus nppiAbs_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx) {
  NppStatus status = validateAbsParameters(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 1, sizeof(Npp32f));
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiAbs_32f_C1R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiAbs_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_C1IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateAbsParameters(pSrcDst, nSrcDstStep, nullptr, 0, oSizeROI, 1, sizeof(Npp32f));
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiAbs_32f_C1IR_Ctx_cuda(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_C1IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiAbs_32f_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx) {
  NppStatus status = validateAbsParameters(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 3, sizeof(Npp32f));
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiAbs_32f_C3R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_C3R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiAbs_32f_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_C3IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateAbsParameters(pSrcDst, nSrcDstStep, nullptr, 0, oSizeROI, 3, sizeof(Npp32f));
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiAbs_32f_C3IR_Ctx_cuda(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_C3IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiAbs_32f_C3IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx) {
  NppStatus status = validateAbsParameters(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 4, sizeof(Npp32f));
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiAbs_32f_AC4R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_AC4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiAbs_32f_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_AC4IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateAbsParameters(pSrcDst, nSrcDstStep, nullptr, 0, oSizeROI, 4, sizeof(Npp32f));
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiAbs_32f_AC4IR_Ctx_cuda(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_AC4IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiAbs_32f_AC4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx) {
  NppStatus status = validateAbsParameters(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 4, sizeof(Npp32f));
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiAbs_32f_C4R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_C4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiAbs_32f_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_C4IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateAbsParameters(pSrcDst, nSrcDstStep, nullptr, 0, oSizeROI, 4, sizeof(Npp32f));
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiAbs_32f_C4IR_Ctx_cuda(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_C4IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiAbs_32f_C4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

// ==================== 8位有符号绝对值 (未实现) ====================

NppStatus nppiAbs_8s_C1R(const Npp8s *pSrc, int nSrcStep, Npp8s *pDst, int nDstStep, NppiSize oSizeROI) {
  // 参数验证
  if (pSrc == nullptr || pDst == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (nSrcStep < static_cast<int>(oSizeROI.width * sizeof(Npp8s)) ||
      nDstStep < static_cast<int>(oSizeROI.width * sizeof(Npp8s))) {
    return NPP_STEP_ERROR;
  }

  // 打印未实现警告
  fprintf(stderr, "WARNING: nppiAbs_8s_C1R is not implemented in this NPP library build\n");

  return NPP_NOT_IMPLEMENTED_ERROR;
}