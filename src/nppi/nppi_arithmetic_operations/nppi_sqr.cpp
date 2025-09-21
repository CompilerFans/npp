#include "npp.h"
#include <cstring>
#include <cuda_runtime.h>

// Forward declarations for mpp host func implementations
extern "C" {
NppStatus nppiSqr_8u_C1RSfs_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                     int nScaleFactor, NppStreamContext nppStreamCtx);
NppStatus nppiSqr_16u_C1RSfs_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                      int nScaleFactor, NppStreamContext nppStreamCtx);
NppStatus nppiSqr_16s_C1RSfs_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                      int nScaleFactor, NppStreamContext nppStreamCtx);
NppStatus nppiSqr_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
}
static inline NppStatus validateSqrInputs(const void *pSrc, int nSrcStep, void *pDst, int nDstStep, NppiSize oSizeROI) {
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  return NPP_SUCCESS;
}
NppStatus nppiSqr_8u_C1RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor, NppStreamContext nppStreamCtx) {
  NppStatus status = validateSqrInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  if (nScaleFactor < 0) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  return nppiSqr_8u_C1RSfs_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSqr_8u_C1RSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                            int nScaleFactor) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiSqr_8u_C1RSfs_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}
NppStatus nppiSqr_16u_C1RSfs_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor, NppStreamContext nppStreamCtx) {
  NppStatus status = validateSqrInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  if (nScaleFactor < 0) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  return nppiSqr_16u_C1RSfs_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSqr_16u_C1RSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                             int nScaleFactor) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiSqr_16u_C1RSfs_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}
NppStatus nppiSqr_16s_C1RSfs_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor, NppStreamContext nppStreamCtx) {
  NppStatus status = validateSqrInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  if (nScaleFactor < 0) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  return nppiSqr_16s_C1RSfs_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSqr_16s_C1RSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                             int nScaleFactor) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiSqr_16s_C1RSfs_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}
NppStatus nppiSqr_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx) {
  NppStatus status = validateSqrInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiSqr_32f_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSqr_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiSqr_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}
