#include "npp.h"
#include <cstring>
#include <cuda_runtime.h>

/**
 * NPP Exponential Operations Implementation
 * Computes exponential of input image values: dst = exp(src)
 */

// Forward declarations for mpp host func implementations
extern "C" {
NppStatus nppiExp_8u_C1RSfs_Ctx_cuda(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                     int nScaleFactor, NppStreamContext nppStreamCtx);
NppStatus nppiExp_16u_C1RSfs_Ctx_cuda(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                      int nScaleFactor, NppStreamContext nppStreamCtx);
NppStatus nppiExp_16s_C1RSfs_Ctx_cuda(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                      int nScaleFactor, NppStreamContext nppStreamCtx);
NppStatus nppiExp_32f_C1R_Ctx_cuda(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
}

/**
 * Validate common input parameters for exp operations
 */
static inline NppStatus validateExpInputs(const void *pSrc, int nSrcStep, void *pDst, int nDstStep, NppiSize oSizeROI) {
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

/**
 * 8-bit unsigned exponential with scaling
 */
NppStatus nppiExp_8u_C1RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor, NppStreamContext nppStreamCtx) {
  NppStatus status = validateExpInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  if (nScaleFactor < 0) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  return nppiExp_8u_C1RSfs_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiExp_8u_C1RSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                            int nScaleFactor) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiExp_8u_C1RSfs_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 8-bit unsigned exponential with scaling - in place
 */
NppStatus nppiExp_8u_C1IRSfs_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx) {
  NppStatus status = validateExpInputs(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  if (nScaleFactor < 0) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  return nppiExp_8u_C1RSfs_Ctx_cuda(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiExp_8u_C1IRSfs(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiExp_8u_C1IRSfs_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 16-bit unsigned exponential with scaling
 */
NppStatus nppiExp_16u_C1RSfs_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor, NppStreamContext nppStreamCtx) {
  NppStatus status = validateExpInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  if (nScaleFactor < 0) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  return nppiExp_16u_C1RSfs_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiExp_16u_C1RSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                             int nScaleFactor) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiExp_16u_C1RSfs_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 16-bit unsigned exponential with scaling - in place
 */
NppStatus nppiExp_16u_C1IRSfs_Ctx(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx) {
  NppStatus status = validateExpInputs(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  if (nScaleFactor < 0) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  return nppiExp_16u_C1RSfs_Ctx_cuda(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiExp_16u_C1IRSfs(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiExp_16u_C1IRSfs_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 16-bit signed exponential with scaling
 */
NppStatus nppiExp_16s_C1RSfs_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor, NppStreamContext nppStreamCtx) {
  NppStatus status = validateExpInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  if (nScaleFactor < 0) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  return nppiExp_16s_C1RSfs_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiExp_16s_C1RSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                             int nScaleFactor) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiExp_16s_C1RSfs_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 16-bit signed exponential with scaling - in place
 */
NppStatus nppiExp_16s_C1IRSfs_Ctx(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx) {
  NppStatus status = validateExpInputs(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  if (nScaleFactor < 0) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  return nppiExp_16s_C1RSfs_Ctx_cuda(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiExp_16s_C1IRSfs(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiExp_16s_C1IRSfs_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

/**
 * 32-bit float exponential (no scaling needed)
 */
NppStatus nppiExp_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx) {
  NppStatus status = validateExpInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiExp_32f_C1R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiExp_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiExp_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

/**
 * 32-bit float exponential - in place
 */
NppStatus nppiExp_32f_C1IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateExpInputs(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiExp_32f_C1R_Ctx_cuda(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiExp_32f_C1IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiExp_32f_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}