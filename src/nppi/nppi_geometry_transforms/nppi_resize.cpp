#include "npp.h"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

// Forward declarations for mpp host func implementations

extern "C" {
NppStatus nppiResize_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                     Npp8u *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                     int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiResize_8u_C3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                     Npp8u *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                     int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiResize_8u_AC4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                      Npp8u *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                      int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiResize_16u_C1R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                      Npp16u *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                      int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiResize_16u_AC4R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                       Npp16u *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                       int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiResize_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                      Npp32f *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                      int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiResize_32f_C3R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                      Npp32f *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                      int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiResize_32f_AC4R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                       Npp32f *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                       int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiResize_8u_P3R_Ctx_impl(const Npp8u *const pSrc[3], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                     Npp8u *const pDst[3], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                     int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiResize_8u_P4R_Ctx_impl(const Npp8u *const pSrc[4], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                     Npp8u *const pDst[4], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                     int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiResize_16u_P3R_Ctx_impl(const Npp16u *const pSrc[3], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                      Npp16u *const pDst[3], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                      int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiResize_16u_P4R_Ctx_impl(const Npp16u *const pSrc[4], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                      Npp16u *const pDst[4], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                      int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiResize_32f_P3R_Ctx_impl(const Npp32f *const pSrc[3], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                      Npp32f *const pDst[3], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                      int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiResize_32f_P4R_Ctx_impl(const Npp32f *const pSrc[4], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                      Npp32f *const pDst[4], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                      int eInterpolation, NppStreamContext nppStreamCtx);
}

// Input validation helper
static inline NppStatus validateResizeInputs(const void *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                             void *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                             int eInterpolation) {
  if (oSrcSize.width <= 0 || oSrcSize.height <= 0 || oDstSize.width <= 0 || oDstSize.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  // Validate interpolation parameters to avoid unused warning
  if (eInterpolation < 0) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  // Basic ROI validation
  if (oSrcRectROI.x < 0 || oSrcRectROI.y < 0 || oSrcRectROI.x + oSrcRectROI.width > oSrcSize.width ||
      oSrcRectROI.y + oSrcRectROI.height > oSrcSize.height) {
    return NPP_SIZE_ERROR;
  }

  if (oDstRectROI.x < 0 || oDstRectROI.y < 0 || oDstRectROI.x + oDstRectROI.width > oDstSize.width ||
      oDstRectROI.y + oDstRectROI.height > oDstSize.height) {
    return NPP_SIZE_ERROR;
  }

  return NPP_SUCCESS;
}

static inline NppStatus validatePlanarInputs(const void *const *pSrcPlanes, int nSrcStep, NppiSize oSrcSize,
                                             NppiRect oSrcRectROI, void *const *pDstPlanes, int nDstStep,
                                             NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation,
                                             int channelCount) {
  if (!pSrcPlanes || !pDstPlanes) {
    return NPP_NULL_POINTER_ERROR;
  }

  for (int c = 0; c < channelCount; ++c) {
    if (!pSrcPlanes[c] || !pDstPlanes[c]) {
      return NPP_NULL_POINTER_ERROR;
    }
  }

  return validateResizeInputs(pSrcPlanes[0], nSrcStep, oSrcSize, oSrcRectROI, pDstPlanes[0], nDstStep, oDstSize,
                              oDstRectROI, eInterpolation);
}

// 8-bit unsigned single channel resize
NppStatus nppiResize_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp8u *pDst,
                                int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation,
                                NppStreamContext nppStreamCtx) {
  NppStatus status = validateResizeInputs(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                          eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiResize_8u_C1R_Ctx_impl(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                    eInterpolation, nppStreamCtx);
}

NppStatus nppiResize_8u_C1R(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp8u *pDst,
                            int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiResize_8u_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                               eInterpolation, nppStreamCtx);
}

// 8-bit unsigned three channel resize
NppStatus nppiResize_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp8u *pDst,
                                int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation,
                                NppStreamContext nppStreamCtx) {
  NppStatus status = validateResizeInputs(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                          eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiResize_8u_C3R_Ctx_impl(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                    eInterpolation, nppStreamCtx);
}

NppStatus nppiResize_8u_C3R(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp8u *pDst,
                            int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiResize_8u_C3R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                               eInterpolation, nppStreamCtx);
}

// 8-bit unsigned four channel resize (alpha preserved)
NppStatus nppiResize_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                 Npp8u *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                 int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateResizeInputs(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                          eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiResize_8u_AC4R_Ctx_impl(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                     eInterpolation, nppStreamCtx);
}

NppStatus nppiResize_8u_AC4R(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp8u *pDst,
                             int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiResize_8u_AC4R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                eInterpolation, nppStreamCtx);
}

// 8-bit unsigned planar three channel resize
NppStatus nppiResize_8u_P3R_Ctx(const Npp8u *pSrc[3], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                Npp8u *pDst[3], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                int eInterpolation, NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  const void *srcPlanes[3];
  void *dstPlanes[3];
  for (int c = 0; c < 3; ++c) {
    if (!pSrc[c] || !pDst[c]) {
      return NPP_NULL_POINTER_ERROR;
    }
    srcPlanes[c] = pSrc[c];
    dstPlanes[c] = pDst[c];
  }

  NppStatus status =
      validatePlanarInputs(srcPlanes, nSrcStep, oSrcSize, oSrcRectROI, dstPlanes, nDstStep, oDstSize, oDstRectROI,
                           eInterpolation, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiResize_8u_P3R_Ctx_impl(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                    eInterpolation, nppStreamCtx);
}

NppStatus nppiResize_8u_P3R(const Npp8u *pSrc[3], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp8u *pDst[3],
                            int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiResize_8u_P3R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                               eInterpolation, nppStreamCtx);
}

// 8-bit unsigned planar four channel resize
NppStatus nppiResize_8u_P4R_Ctx(const Npp8u *pSrc[4], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                Npp8u *pDst[4], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                int eInterpolation, NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  const void *srcPlanes[4];
  void *dstPlanes[4];
  for (int c = 0; c < 4; ++c) {
    if (!pSrc[c] || !pDst[c]) {
      return NPP_NULL_POINTER_ERROR;
    }
      srcPlanes[c] = pSrc[c];
      dstPlanes[c] = pDst[c];
  }

  NppStatus status =
      validatePlanarInputs(srcPlanes, nSrcStep, oSrcSize, oSrcRectROI, dstPlanes, nDstStep, oDstSize, oDstRectROI,
                           eInterpolation, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiResize_8u_P4R_Ctx_impl(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                    eInterpolation, nppStreamCtx);
}

NppStatus nppiResize_8u_P4R(const Npp8u *pSrc[4], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp8u *pDst[4],
                            int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiResize_8u_P4R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                               eInterpolation, nppStreamCtx);
}

// 16-bit unsigned single channel resize
NppStatus nppiResize_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                 Npp16u *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                 int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateResizeInputs(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                          eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiResize_16u_C1R_Ctx_impl(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                     eInterpolation, nppStreamCtx);
}

NppStatus nppiResize_16u_C1R(const Npp16u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp16u *pDst,
                             int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiResize_16u_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                eInterpolation, nppStreamCtx);
}

// 16-bit unsigned four channel resize (alpha preserved)
NppStatus nppiResize_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                  Npp16u *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                  int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateResizeInputs(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                          eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiResize_16u_AC4R_Ctx_impl(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                      eInterpolation, nppStreamCtx);
}

NppStatus nppiResize_16u_AC4R(const Npp16u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                              Npp16u *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                              int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiResize_16u_AC4R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                 eInterpolation, nppStreamCtx);
}

// 16-bit unsigned planar three channel resize
NppStatus nppiResize_16u_P3R_Ctx(const Npp16u *pSrc[3], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                 Npp16u *pDst[3], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                 int eInterpolation, NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  const void *srcPlanes[3];
  void *dstPlanes[3];
  for (int c = 0; c < 3; ++c) {
    if (!pSrc[c] || !pDst[c]) {
      return NPP_NULL_POINTER_ERROR;
    }
    srcPlanes[c] = pSrc[c];
    dstPlanes[c] = pDst[c];
  }

  NppStatus status =
      validatePlanarInputs(srcPlanes, nSrcStep, oSrcSize, oSrcRectROI, dstPlanes, nDstStep, oDstSize, oDstRectROI,
                           eInterpolation, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiResize_16u_P3R_Ctx_impl(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                     eInterpolation, nppStreamCtx);
}

NppStatus nppiResize_16u_P3R(const Npp16u *pSrc[3], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                             Npp16u *pDst[3], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                             int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiResize_16u_P3R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                eInterpolation, nppStreamCtx);
}

// 16-bit unsigned planar four channel resize
NppStatus nppiResize_16u_P4R_Ctx(const Npp16u *pSrc[4], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                 Npp16u *pDst[4], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                 int eInterpolation, NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  const void *srcPlanes[4];
  void *dstPlanes[4];
  for (int c = 0; c < 4; ++c) {
    if (!pSrc[c] || !pDst[c]) {
      return NPP_NULL_POINTER_ERROR;
    }
    srcPlanes[c] = pSrc[c];
    dstPlanes[c] = pDst[c];
  }

  NppStatus status =
      validatePlanarInputs(srcPlanes, nSrcStep, oSrcSize, oSrcRectROI, dstPlanes, nDstStep, oDstSize, oDstRectROI,
                           eInterpolation, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiResize_16u_P4R_Ctx_impl(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                     eInterpolation, nppStreamCtx);
}

NppStatus nppiResize_16u_P4R(const Npp16u *pSrc[4], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                             Npp16u *pDst[4], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                             int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiResize_16u_P4R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                eInterpolation, nppStreamCtx);
}

// 32-bit float single channel resize
NppStatus nppiResize_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                 Npp32f *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                 int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateResizeInputs(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                          eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiResize_32f_C1R_Ctx_impl(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                     eInterpolation, nppStreamCtx);
}

NppStatus nppiResize_32f_C1R(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp32f *pDst,
                             int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiResize_32f_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                eInterpolation, nppStreamCtx);
}

// 32-bit float four channel resize (alpha preserved)
NppStatus nppiResize_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                  Npp32f *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                  int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateResizeInputs(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                          eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiResize_32f_AC4R_Ctx_impl(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                      eInterpolation, nppStreamCtx);
}

NppStatus nppiResize_32f_AC4R(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp32f *pDst,
                              int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiResize_32f_AC4R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                 eInterpolation, nppStreamCtx);
}

// 32-bit float three channel resize
NppStatus nppiResize_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                 Npp32f *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                 int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateResizeInputs(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                          eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiResize_32f_C3R_Ctx_impl(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                     eInterpolation, nppStreamCtx);
}

NppStatus nppiResize_32f_C3R(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp32f *pDst,
                             int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiResize_32f_C3R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                eInterpolation, nppStreamCtx);
}

// 32-bit float planar three channel resize
NppStatus nppiResize_32f_P3R_Ctx(const Npp32f *pSrc[3], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                 Npp32f *pDst[3], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                 int eInterpolation, NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  const void *srcPlanes[3];
  void *dstPlanes[3];
  for (int c = 0; c < 3; ++c) {
    if (!pSrc[c] || !pDst[c]) {
      return NPP_NULL_POINTER_ERROR;
    }
    srcPlanes[c] = pSrc[c];
    dstPlanes[c] = pDst[c];
  }

  NppStatus status =
      validatePlanarInputs(srcPlanes, nSrcStep, oSrcSize, oSrcRectROI, dstPlanes, nDstStep, oDstSize, oDstRectROI,
                           eInterpolation, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiResize_32f_P3R_Ctx_impl(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                     eInterpolation, nppStreamCtx);
}

NppStatus nppiResize_32f_P3R(const Npp32f *pSrc[3], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                             Npp32f *pDst[3], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                             int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiResize_32f_P3R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                eInterpolation, nppStreamCtx);
}

// 32-bit float planar four channel resize
NppStatus nppiResize_32f_P4R_Ctx(const Npp32f *pSrc[4], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                 Npp32f *pDst[4], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                 int eInterpolation, NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  const void *srcPlanes[4];
  void *dstPlanes[4];
  for (int c = 0; c < 4; ++c) {
    if (!pSrc[c] || !pDst[c]) {
      return NPP_NULL_POINTER_ERROR;
    }
    srcPlanes[c] = pSrc[c];
    dstPlanes[c] = pDst[c];
  }

  NppStatus status =
      validatePlanarInputs(srcPlanes, nSrcStep, oSrcSize, oSrcRectROI, dstPlanes, nDstStep, oDstSize, oDstRectROI,
                           eInterpolation, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiResize_32f_P4R_Ctx_impl(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                     eInterpolation, nppStreamCtx);
}

NppStatus nppiResize_32f_P4R(const Npp32f *pSrc[4], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                             Npp32f *pDst[4], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                             int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiResize_32f_P4R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                eInterpolation, nppStreamCtx);
}
