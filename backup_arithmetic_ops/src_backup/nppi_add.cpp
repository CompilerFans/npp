#include "npp.h"
#include <cuda_runtime.h>

// Forward declarations for mpp host func implementations

extern "C" {
NppStatus nppiAdd_8u_C1RSfs_Ctx_impl(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                     int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus nppiAdd_8u_C3RSfs_Ctx_impl(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                     int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus nppiAdd_8u_C4RSfs_Ctx_impl(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                     int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus nppiAdd_16u_C1RSfs_Ctx_impl(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step,
                                      Npp16u *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                      NppStreamContext nppStreamCtx);

NppStatus nppiAdd_16s_C1RSfs_Ctx_impl(const Npp16s *pSrc1, int nSrc1Step, const Npp16s *pSrc2, int nSrc2Step,
                                      Npp16s *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                      NppStreamContext nppStreamCtx);

NppStatus nppiAdd_32f_C1R_Ctx_impl(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                                   int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus nppiAdd_32f_C3R_Ctx_impl(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                                   int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
}

static NppStatus validateDualSourceParameters(const void *pSrc1, int nSrc1Step, const void *pSrc2, int nSrc2Step,
                                              const void *pDst, int nDstStep, NppiSize oSizeROI, int elementSize) {
  if (!pSrc1 || !pSrc2 || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  // vendor NPP behavior: zero-size ROI returns success (no processing needed)
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  // Early return for zero-size ROI (vendor NPP compatible behavior)
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }

  int minStep = oSizeROI.width * elementSize;
  if (nSrc1Step < minStep || nSrc2Step < minStep || nDstStep < minStep) {
    return NPP_STRIDE_ERROR;
  }

  return NPP_NO_ERROR;
}

// 8-bit unsigned implementations
NppStatus nppiAdd_8u_C1RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  // Validate parameters
  NppStatus status =
      validateDualSourceParameters(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, sizeof(Npp8u));
  if (status != NPP_NO_ERROR) {
    return status;
  }

  if (nScaleFactor < 0 || nScaleFactor > 31) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  // Call GPU implementation
  return nppiAdd_8u_C1RSfs_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                    nppStreamCtx);
}

NppStatus nppiAdd_8u_C1RSfs(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiAdd_8u_C1RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                               nppStreamCtx);
}

// In-place version
NppStatus nppiAdd_8u_C1IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                 int nScaleFactor, NppStreamContext nppStreamCtx) {
  // For in-place, pSrcDst is both source and destination
  return nppiAdd_8u_C1RSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor,
                               nppStreamCtx);
}

NppStatus nppiAdd_8u_C1IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                             int nScaleFactor) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiAdd_8u_C1IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

// 16-bit unsigned implementations
NppStatus nppiAdd_16u_C1RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                                 int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  // Validate parameters
  NppStatus status =
      validateDualSourceParameters(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, sizeof(Npp16u));
  if (status != NPP_NO_ERROR) {
    return status;
  }

  if (nScaleFactor < 0 || nScaleFactor > 31) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  // Call GPU implementation
  return nppiAdd_16u_C1RSfs_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                     nppStreamCtx);
}

NppStatus nppiAdd_16u_C1RSfs(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiAdd_16u_C1RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                nppStreamCtx);
}

// 32-bit float implementations (no scaling)
NppStatus nppiAdd_32f_C1R_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  // Validate parameters (no scale factor for float)
  NppStatus status =
      validateDualSourceParameters(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, sizeof(Npp32f));
  if (status != NPP_NO_ERROR) {
    return status;
  }

  // Call GPU implementation
  return nppiAdd_32f_C1R_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAdd_32f_C1R(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                          int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiAdd_32f_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// In-place version
NppStatus nppiAdd_32f_C1IR_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx) {
  return nppiAdd_32f_C1R_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAdd_32f_C1IR(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiAdd_32f_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

// Multi-channel implementations
// 8-bit unsigned, 3 channels
NppStatus nppiAdd_8u_C3RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  // Validate parameters (3 channels)
  NppStatus status =
      validateDualSourceParameters(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, 3 * sizeof(Npp8u));
  if (status != NPP_NO_ERROR) {
    return status;
  }

  if (nScaleFactor < 0 || nScaleFactor > 31) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  // Call GPU implementation
  return nppiAdd_8u_C3RSfs_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                    nppStreamCtx);
}

NppStatus nppiAdd_8u_C3RSfs(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiAdd_8u_C3RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                               nppStreamCtx);
}

// 32-bit float, 3 channels
NppStatus nppiAdd_32f_C3R_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  // Validate parameters (3 channels)
  NppStatus status =
      validateDualSourceParameters(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, 3 * sizeof(Npp32f));
  if (status != NPP_NO_ERROR) {
    return status;
  }

  // Call GPU implementation
  return nppiAdd_32f_C3R_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAdd_32f_C3R(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                          int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiAdd_32f_C3R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}
