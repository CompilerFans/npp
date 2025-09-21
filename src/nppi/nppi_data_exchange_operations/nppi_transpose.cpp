#include "npp.h"
#include <cstring>
#include <cuda_runtime.h>

// Forward declarations for mpp host func implementations

extern "C" {
NppStatus nppiTranspose_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSrcROI,
                                        NppStreamContext nppStreamCtx);
NppStatus nppiTranspose_8u_C3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSrcROI,
                                        NppStreamContext nppStreamCtx);
NppStatus nppiTranspose_8u_C4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSrcROI,
                                        NppStreamContext nppStreamCtx);

NppStatus nppiTranspose_16u_C1R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSrcROI,
                                         NppStreamContext nppStreamCtx);
NppStatus nppiTranspose_16u_C3R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSrcROI,
                                         NppStreamContext nppStreamCtx);
NppStatus nppiTranspose_16u_C4R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSrcROI,
                                         NppStreamContext nppStreamCtx);

NppStatus nppiTranspose_16s_C1R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSrcROI,
                                         NppStreamContext nppStreamCtx);
NppStatus nppiTranspose_16s_C3R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSrcROI,
                                         NppStreamContext nppStreamCtx);
NppStatus nppiTranspose_16s_C4R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSrcROI,
                                         NppStreamContext nppStreamCtx);

NppStatus nppiTranspose_32s_C1R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oSrcROI,
                                         NppStreamContext nppStreamCtx);
NppStatus nppiTranspose_32s_C3R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oSrcROI,
                                         NppStreamContext nppStreamCtx);
NppStatus nppiTranspose_32s_C4R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oSrcROI,
                                         NppStreamContext nppStreamCtx);

NppStatus nppiTranspose_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSrcROI,
                                         NppStreamContext nppStreamCtx);
NppStatus nppiTranspose_32f_C3R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSrcROI,
                                         NppStreamContext nppStreamCtx);
NppStatus nppiTranspose_32f_C4R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSrcROI,
                                         NppStreamContext nppStreamCtx);
}

static NppStatus validateTransposeParameters(const void *pSrc, int nSrcStep, const void *pDst, int nDstStep,
                                             NppiSize oSrcROI, int elementSize, int channels) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSrcROI.width <= 0 || oSrcROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  int minSrcStep = oSrcROI.width * elementSize * channels;
  int minDstStep = oSrcROI.height * elementSize * channels; // Note: transposed dimensions

  if (nSrcStep < minSrcStep || nDstStep < minDstStep) {
    return NPP_STRIDE_ERROR;
  }

  return NPP_NO_ERROR;
}

// 8-bit unsigned implementations

NppStatus nppiTranspose_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSrcROI,
                                   NppStreamContext nppStreamCtx) {
  NppStatus status = validateTransposeParameters(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, sizeof(Npp8u), 1);
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiTranspose_8u_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, nppStreamCtx);
}

NppStatus nppiTranspose_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSrcROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiTranspose_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, nppStreamCtx);
}

NppStatus nppiTranspose_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSrcROI,
                                   NppStreamContext nppStreamCtx) {
  NppStatus status = validateTransposeParameters(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, sizeof(Npp8u), 3);
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiTranspose_8u_C3R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, nppStreamCtx);
}

NppStatus nppiTranspose_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSrcROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiTranspose_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, nppStreamCtx);
}

NppStatus nppiTranspose_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSrcROI,
                                   NppStreamContext nppStreamCtx) {
  NppStatus status = validateTransposeParameters(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, sizeof(Npp8u), 4);
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiTranspose_8u_C4R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, nppStreamCtx);
}

NppStatus nppiTranspose_8u_C4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSrcROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiTranspose_8u_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, nppStreamCtx);
}

// 16-bit unsigned implementations
NppStatus nppiTranspose_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSrcROI,
                                    NppStreamContext nppStreamCtx) {
  NppStatus status = validateTransposeParameters(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, sizeof(Npp16u), 1);
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiTranspose_16u_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, nppStreamCtx);
}

NppStatus nppiTranspose_16u_C1R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSrcROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiTranspose_16u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, nppStreamCtx);
}

NppStatus nppiTranspose_16u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSrcROI,
                                    NppStreamContext nppStreamCtx) {
  NppStatus status = validateTransposeParameters(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, sizeof(Npp16u), 3);
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiTranspose_16u_C3R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, nppStreamCtx);
}

NppStatus nppiTranspose_16u_C3R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSrcROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiTranspose_16u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, nppStreamCtx);
}

NppStatus nppiTranspose_16u_C4R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSrcROI,
                                    NppStreamContext nppStreamCtx) {
  NppStatus status = validateTransposeParameters(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, sizeof(Npp16u), 4);
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiTranspose_16u_C4R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, nppStreamCtx);
}

NppStatus nppiTranspose_16u_C4R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSrcROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiTranspose_16u_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, nppStreamCtx);
}

// 16-bit signed implementations
NppStatus nppiTranspose_16s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSrcROI,
                                    NppStreamContext nppStreamCtx) {
  NppStatus status = validateTransposeParameters(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, sizeof(Npp16s), 1);
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiTranspose_16s_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, nppStreamCtx);
}

NppStatus nppiTranspose_16s_C1R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSrcROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiTranspose_16s_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, nppStreamCtx);
}

NppStatus nppiTranspose_16s_C3R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSrcROI,
                                    NppStreamContext nppStreamCtx) {
  NppStatus status = validateTransposeParameters(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, sizeof(Npp16s), 3);
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiTranspose_16s_C3R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, nppStreamCtx);
}

NppStatus nppiTranspose_16s_C3R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSrcROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiTranspose_16s_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, nppStreamCtx);
}

NppStatus nppiTranspose_16s_C4R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSrcROI,
                                    NppStreamContext nppStreamCtx) {
  NppStatus status = validateTransposeParameters(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, sizeof(Npp16s), 4);
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiTranspose_16s_C4R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, nppStreamCtx);
}

NppStatus nppiTranspose_16s_C4R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSrcROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiTranspose_16s_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, nppStreamCtx);
}

// 32-bit signed implementations
NppStatus nppiTranspose_32s_C1R_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oSrcROI,
                                    NppStreamContext nppStreamCtx) {
  NppStatus status = validateTransposeParameters(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, sizeof(Npp32s), 1);
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiTranspose_32s_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, nppStreamCtx);
}

NppStatus nppiTranspose_32s_C1R(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oSrcROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiTranspose_32s_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, nppStreamCtx);
}

NppStatus nppiTranspose_32s_C3R_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oSrcROI,
                                    NppStreamContext nppStreamCtx) {
  NppStatus status = validateTransposeParameters(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, sizeof(Npp32s), 3);
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiTranspose_32s_C3R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, nppStreamCtx);
}

NppStatus nppiTranspose_32s_C3R(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oSrcROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiTranspose_32s_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, nppStreamCtx);
}

NppStatus nppiTranspose_32s_C4R_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oSrcROI,
                                    NppStreamContext nppStreamCtx) {
  NppStatus status = validateTransposeParameters(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, sizeof(Npp32s), 4);
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiTranspose_32s_C4R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, nppStreamCtx);
}

NppStatus nppiTranspose_32s_C4R(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oSrcROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiTranspose_32s_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, nppStreamCtx);
}

NppStatus nppiTranspose_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSrcROI,
                                    NppStreamContext nppStreamCtx) {
  NppStatus status = validateTransposeParameters(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, sizeof(Npp32f), 1);
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiTranspose_32f_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, nppStreamCtx);
}

NppStatus nppiTranspose_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSrcROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiTranspose_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, nppStreamCtx);
}

NppStatus nppiTranspose_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSrcROI,
                                    NppStreamContext nppStreamCtx) {
  NppStatus status = validateTransposeParameters(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, sizeof(Npp32f), 3);
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiTranspose_32f_C3R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, nppStreamCtx);
}

NppStatus nppiTranspose_32f_C3R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSrcROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiTranspose_32f_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, nppStreamCtx);
}

NppStatus nppiTranspose_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSrcROI,
                                    NppStreamContext nppStreamCtx) {
  NppStatus status = validateTransposeParameters(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, sizeof(Npp32f), 4);
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiTranspose_32f_C4R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, nppStreamCtx);
}

NppStatus nppiTranspose_32f_C4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSrcROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiTranspose_32f_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSrcROI, nppStreamCtx);
}
