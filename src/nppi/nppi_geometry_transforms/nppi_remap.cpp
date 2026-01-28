#include "npp.h"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

// Forward declarations for mpp host func implementations

extern "C" {
NppStatus nppiRemap_8u_C1R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                    const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp8u *pDst,
                                    int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                    NppStreamContext nppStreamCtx);
NppStatus nppiRemap_8u_C3R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                    const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp8u *pDst,
                                    int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                    NppStreamContext nppStreamCtx);
NppStatus nppiRemap_8u_C4R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                    const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp8u *pDst,
                                    int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                    NppStreamContext nppStreamCtx);
NppStatus nppiRemap_8u_AC4R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                     Npp8u *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                     NppStreamContext nppStreamCtx);
NppStatus nppiRemap_16u_C1R_Ctx_impl(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                     Npp16u *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                     NppStreamContext nppStreamCtx);
NppStatus nppiRemap_16u_C3R_Ctx_impl(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                     Npp16u *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                     NppStreamContext nppStreamCtx);
NppStatus nppiRemap_16u_C4R_Ctx_impl(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                     Npp16u *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                     NppStreamContext nppStreamCtx);
NppStatus nppiRemap_16u_AC4R_Ctx_impl(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                      Npp16u *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                      NppStreamContext nppStreamCtx);
NppStatus nppiRemap_16s_C1R_Ctx_impl(const Npp16s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                     Npp16s *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                     NppStreamContext nppStreamCtx);
NppStatus nppiRemap_16s_C3R_Ctx_impl(const Npp16s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                     Npp16s *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                     NppStreamContext nppStreamCtx);
NppStatus nppiRemap_16s_C4R_Ctx_impl(const Npp16s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                     Npp16s *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                     NppStreamContext nppStreamCtx);
NppStatus nppiRemap_16s_AC4R_Ctx_impl(const Npp16s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                      Npp16s *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                      NppStreamContext nppStreamCtx);
NppStatus nppiRemap_32f_C1R_Ctx_impl(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                     Npp32f *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                     NppStreamContext nppStreamCtx);
NppStatus nppiRemap_32f_C3R_Ctx_impl(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                     Npp32f *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                     NppStreamContext nppStreamCtx);
NppStatus nppiRemap_32f_C4R_Ctx_impl(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                     Npp32f *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                     NppStreamContext nppStreamCtx);
NppStatus nppiRemap_32f_AC4R_Ctx_impl(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                      Npp32f *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                      NppStreamContext nppStreamCtx);
NppStatus nppiRemap_64f_C1R_Ctx_impl(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     const Npp64f *pXMap, int nXMapStep, const Npp64f *pYMap, int nYMapStep,
                                     Npp64f *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                     NppStreamContext nppStreamCtx);
NppStatus nppiRemap_64f_C3R_Ctx_impl(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     const Npp64f *pXMap, int nXMapStep, const Npp64f *pYMap, int nYMapStep,
                                     Npp64f *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                     NppStreamContext nppStreamCtx);
NppStatus nppiRemap_64f_C4R_Ctx_impl(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     const Npp64f *pXMap, int nXMapStep, const Npp64f *pYMap, int nYMapStep,
                                     Npp64f *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                     NppStreamContext nppStreamCtx);
NppStatus nppiRemap_64f_AC4R_Ctx_impl(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      const Npp64f *pXMap, int nXMapStep, const Npp64f *pYMap, int nYMapStep,
                                      Npp64f *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                      NppStreamContext nppStreamCtx);
}

// Input validation helper
static inline NppStatus validateRemapInputs(const void *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                            const void *pXMap, int nXMapStep, const void *pYMap, int nYMapStep,
                                            void *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation) {
  if (oSrcSize.width <= 0 || oSrcSize.height <= 0 || oDstSizeROI.width <= 0 || oDstSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstStep <= 0 || nXMapStep <= 0 || nYMapStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (!pSrc || !pDst || !pXMap || !pYMap) {
    return NPP_NULL_POINTER_ERROR;
  }

  // Basic ROI validation
  if (oSrcROI.x < 0 || oSrcROI.y < 0 || oSrcROI.x + oSrcROI.width > oSrcSize.width ||
      oSrcROI.y + oSrcROI.height > oSrcSize.height) {
    return NPP_SIZE_ERROR;
  }

  // Validate interpolation parameter
  if (eInterpolation < 0) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  return NPP_SUCCESS;
}

static inline NppStatus validateRemapPlanarInputs(const void *const *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                  NppiRect oSrcROI, const void *pXMap, int nXMapStep,
                                                  const void *pYMap, int nYMapStep, void **pDst, int nDstStep,
                                                  NppiSize oDstSizeROI, int eInterpolation, int planes) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  for (int i = 0; i < planes; i++) {
    if (!pSrc[i] || !pDst[i]) {
      return NPP_NULL_POINTER_ERROR;
    }
  }

  return validateRemapInputs(pSrc[0], oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst[0],
                             nDstStep, oDstSizeROI, eInterpolation);
}

// 8-bit unsigned single channel remap
NppStatus nppiRemap_8u_C1R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                               const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp8u *pDst,
                               int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateRemapInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                         nDstStep, oDstSizeROI, eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiRemap_8u_C1R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                   nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx);
}

NppStatus nppiRemap_8u_C1R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const Npp32f *pXMap,
                           int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp8u *pDst, int nDstStep,
                           NppiSize oDstSizeROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiRemap_8u_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep,
                              oDstSizeROI, eInterpolation, nppStreamCtx);
}

// 8-bit unsigned three channel remap
NppStatus nppiRemap_8u_C3R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                               const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp8u *pDst,
                               int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateRemapInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                         nDstStep, oDstSizeROI, eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiRemap_8u_C3R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                   nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx);
}

NppStatus nppiRemap_8u_C3R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const Npp32f *pXMap,
                           int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp8u *pDst, int nDstStep,
                           NppiSize oDstSizeROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiRemap_8u_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep,
                              oDstSizeROI, eInterpolation, nppStreamCtx);
}

// 8-bit unsigned four channel remap
NppStatus nppiRemap_8u_C4R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                               const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp8u *pDst,
                               int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateRemapInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                         nDstStep, oDstSizeROI, eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiRemap_8u_C4R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                   nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx);
}

NppStatus nppiRemap_8u_C4R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const Npp32f *pXMap,
                           int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp8u *pDst, int nDstStep,
                           NppiSize oDstSizeROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiRemap_8u_C4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep,
                              oDstSizeROI, eInterpolation, nppStreamCtx);
}

// 8-bit unsigned AC4 remap
NppStatus nppiRemap_8u_AC4R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp8u *pDst,
                                int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateRemapInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                         nDstStep, oDstSizeROI, eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiRemap_8u_AC4R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                    nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx);
}

NppStatus nppiRemap_8u_AC4R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                            const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp8u *pDst,
                            int nDstStep, NppiSize oDstSizeROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiRemap_8u_AC4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep,
                               oDstSizeROI, eInterpolation, nppStreamCtx);
}

// 8-bit unsigned planar remap
NppStatus nppiRemap_8u_P3R_Ctx(const Npp8u *const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                               const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                               Npp8u *pDst[3], int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                               NppStreamContext nppStreamCtx) {
  NppStatus status = validateRemapPlanarInputs(reinterpret_cast<const void *const *>(pSrc), oSrcSize, nSrcStep, oSrcROI,
                                               pXMap, nXMapStep, pYMap, nYMapStep,
                                               reinterpret_cast<void **>(pDst), nDstStep, oDstSizeROI, eInterpolation, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  for (int i = 0; i < 3; i++) {
    status = nppiRemap_8u_C1R_Ctx_impl(pSrc[i], oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep,
                                      pDst[i], nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx);
    if (status != NPP_SUCCESS) {
      return status;
    }
  }

  return NPP_SUCCESS;
}

NppStatus nppiRemap_8u_P3R(const Npp8u *const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                           const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp8u *pDst[3],
                           int nDstStep, NppiSize oDstSizeROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiRemap_8u_P3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep,
                              oDstSizeROI, eInterpolation, nppStreamCtx);
}

NppStatus nppiRemap_8u_P4R_Ctx(const Npp8u *const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                               const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                               Npp8u *pDst[4], int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                               NppStreamContext nppStreamCtx) {
  NppStatus status = validateRemapPlanarInputs(reinterpret_cast<const void *const *>(pSrc), oSrcSize, nSrcStep, oSrcROI,
                                               pXMap, nXMapStep, pYMap, nYMapStep,
                                               reinterpret_cast<void **>(pDst), nDstStep, oDstSizeROI, eInterpolation, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  for (int i = 0; i < 4; i++) {
    status = nppiRemap_8u_C1R_Ctx_impl(pSrc[i], oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep,
                                      pDst[i], nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx);
    if (status != NPP_SUCCESS) {
      return status;
    }
  }

  return NPP_SUCCESS;
}

NppStatus nppiRemap_8u_P4R(const Npp8u *const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                           const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp8u *pDst[4],
                           int nDstStep, NppiSize oDstSizeROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiRemap_8u_P4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep,
                              oDstSizeROI, eInterpolation, nppStreamCtx);
}

// 16-bit unsigned single channel remap
NppStatus nppiRemap_16u_C1R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp16u *pDst,
                                int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateRemapInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                         nDstStep, oDstSizeROI, eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiRemap_16u_C1R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                    nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx);
}

// 16-bit unsigned three channel remap
NppStatus nppiRemap_16u_C3R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp16u *pDst,
                                int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateRemapInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                         nDstStep, oDstSizeROI, eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiRemap_16u_C3R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                    nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx);
}

NppStatus nppiRemap_16u_C1R(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const Npp32f *pXMap,
                            int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp16u *pDst, int nDstStep,
                            NppiSize oDstSizeROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiRemap_16u_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep,
                               oDstSizeROI, eInterpolation, nppStreamCtx);
}

NppStatus nppiRemap_16u_C3R(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const Npp32f *pXMap,
                            int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp16u *pDst, int nDstStep,
                            NppiSize oDstSizeROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiRemap_16u_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep,
                               oDstSizeROI, eInterpolation, nppStreamCtx);
}

// 16-bit unsigned four channel remap
NppStatus nppiRemap_16u_C4R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp16u *pDst,
                                int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateRemapInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                         nDstStep, oDstSizeROI, eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiRemap_16u_C4R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                    nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx);
}

NppStatus nppiRemap_16u_C4R(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                            const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp16u *pDst,
                            int nDstStep, NppiSize oDstSizeROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiRemap_16u_C4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep,
                               oDstSizeROI, eInterpolation, nppStreamCtx);
}

// 16-bit unsigned AC4 remap
NppStatus nppiRemap_16u_AC4R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                 const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp16u *pDst,
                                 int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateRemapInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                         nDstStep, oDstSizeROI, eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiRemap_16u_AC4R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                     nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx);
}

NppStatus nppiRemap_16u_AC4R(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                             const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp16u *pDst,
                             int nDstStep, NppiSize oDstSizeROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiRemap_16u_AC4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep,
                                oDstSizeROI, eInterpolation, nppStreamCtx);
}

// 16-bit unsigned planar remap
NppStatus nppiRemap_16u_P3R_Ctx(const Npp16u *const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                Npp16u *pDst[3], int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                NppStreamContext nppStreamCtx) {
  NppStatus status = validateRemapPlanarInputs(reinterpret_cast<const void *const *>(pSrc), oSrcSize, nSrcStep, oSrcROI,
                                               pXMap, nXMapStep, pYMap, nYMapStep,
                                               reinterpret_cast<void **>(pDst), nDstStep, oDstSizeROI, eInterpolation, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  for (int i = 0; i < 3; i++) {
    status = nppiRemap_16u_C1R_Ctx_impl(pSrc[i], oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep,
                                       pDst[i], nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx);
    if (status != NPP_SUCCESS) {
      return status;
    }
  }

  return NPP_SUCCESS;
}

NppStatus nppiRemap_16u_P3R(const Npp16u *const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                            const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp16u *pDst[3],
                            int nDstStep, NppiSize oDstSizeROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiRemap_16u_P3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep,
                               oDstSizeROI, eInterpolation, nppStreamCtx);
}

NppStatus nppiRemap_16u_P4R_Ctx(const Npp16u *const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                Npp16u *pDst[4], int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                NppStreamContext nppStreamCtx) {
  NppStatus status = validateRemapPlanarInputs(reinterpret_cast<const void *const *>(pSrc), oSrcSize, nSrcStep, oSrcROI,
                                               pXMap, nXMapStep, pYMap, nYMapStep,
                                               reinterpret_cast<void **>(pDst), nDstStep, oDstSizeROI, eInterpolation, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  for (int i = 0; i < 4; i++) {
    status = nppiRemap_16u_C1R_Ctx_impl(pSrc[i], oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep,
                                       pDst[i], nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx);
    if (status != NPP_SUCCESS) {
      return status;
    }
  }

  return NPP_SUCCESS;
}

NppStatus nppiRemap_16u_P4R(const Npp16u *const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                            const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp16u *pDst[4],
                            int nDstStep, NppiSize oDstSizeROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiRemap_16u_P4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep,
                               oDstSizeROI, eInterpolation, nppStreamCtx);
}

// 16-bit signed single channel remap
NppStatus nppiRemap_16s_C1R_Ctx(const Npp16s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp16s *pDst,
                                int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateRemapInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                         nDstStep, oDstSizeROI, eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiRemap_16s_C1R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                    nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx);
}

// 16-bit signed three channel remap
NppStatus nppiRemap_16s_C3R_Ctx(const Npp16s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp16s *pDst,
                                int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateRemapInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                         nDstStep, oDstSizeROI, eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiRemap_16s_C3R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                    nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx);
}

NppStatus nppiRemap_16s_C1R(const Npp16s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const Npp32f *pXMap,
                            int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp16s *pDst, int nDstStep,
                            NppiSize oDstSizeROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiRemap_16s_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep,
                               oDstSizeROI, eInterpolation, nppStreamCtx);
}

NppStatus nppiRemap_16s_C3R(const Npp16s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const Npp32f *pXMap,
                            int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp16s *pDst, int nDstStep,
                            NppiSize oDstSizeROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiRemap_16s_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep,
                               oDstSizeROI, eInterpolation, nppStreamCtx);
}

// 16-bit signed four channel remap
NppStatus nppiRemap_16s_C4R_Ctx(const Npp16s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp16s *pDst,
                                int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateRemapInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                         nDstStep, oDstSizeROI, eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiRemap_16s_C4R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                    nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx);
}

NppStatus nppiRemap_16s_C4R(const Npp16s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                            const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp16s *pDst,
                            int nDstStep, NppiSize oDstSizeROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiRemap_16s_C4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep,
                               oDstSizeROI, eInterpolation, nppStreamCtx);
}

// 16-bit signed AC4 remap
NppStatus nppiRemap_16s_AC4R_Ctx(const Npp16s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                 const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp16s *pDst,
                                 int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateRemapInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                         nDstStep, oDstSizeROI, eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiRemap_16s_AC4R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                     nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx);
}

NppStatus nppiRemap_16s_AC4R(const Npp16s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                             const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp16s *pDst,
                             int nDstStep, NppiSize oDstSizeROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiRemap_16s_AC4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep,
                                oDstSizeROI, eInterpolation, nppStreamCtx);
}

// 16-bit signed planar remap
NppStatus nppiRemap_16s_P3R_Ctx(const Npp16s *const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                Npp16s *pDst[3], int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                NppStreamContext nppStreamCtx) {
  NppStatus status = validateRemapPlanarInputs(reinterpret_cast<const void *const *>(pSrc), oSrcSize, nSrcStep, oSrcROI,
                                               pXMap, nXMapStep, pYMap, nYMapStep,
                                               reinterpret_cast<void **>(pDst), nDstStep, oDstSizeROI, eInterpolation, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  for (int i = 0; i < 3; i++) {
    status = nppiRemap_16s_C1R_Ctx_impl(pSrc[i], oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep,
                                       pDst[i], nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx);
    if (status != NPP_SUCCESS) {
      return status;
    }
  }

  return NPP_SUCCESS;
}

NppStatus nppiRemap_16s_P3R(const Npp16s *const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                            const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp16s *pDst[3],
                            int nDstStep, NppiSize oDstSizeROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiRemap_16s_P3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep,
                               oDstSizeROI, eInterpolation, nppStreamCtx);
}

NppStatus nppiRemap_16s_P4R_Ctx(const Npp16s *const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                Npp16s *pDst[4], int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                NppStreamContext nppStreamCtx) {
  NppStatus status = validateRemapPlanarInputs(reinterpret_cast<const void *const *>(pSrc), oSrcSize, nSrcStep, oSrcROI,
                                               pXMap, nXMapStep, pYMap, nYMapStep,
                                               reinterpret_cast<void **>(pDst), nDstStep, oDstSizeROI, eInterpolation, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  for (int i = 0; i < 4; i++) {
    status = nppiRemap_16s_C1R_Ctx_impl(pSrc[i], oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep,
                                       pDst[i], nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx);
    if (status != NPP_SUCCESS) {
      return status;
    }
  }

  return NPP_SUCCESS;
}

NppStatus nppiRemap_16s_P4R(const Npp16s *const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                            const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp16s *pDst[4],
                            int nDstStep, NppiSize oDstSizeROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiRemap_16s_P4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep,
                               oDstSizeROI, eInterpolation, nppStreamCtx);
}

// 32-bit float single channel remap
NppStatus nppiRemap_32f_C1R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp32f *pDst,
                                int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateRemapInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                         nDstStep, oDstSizeROI, eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiRemap_32f_C1R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                    nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx);
}

// 32-bit float three channel remap
NppStatus nppiRemap_32f_C3R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp32f *pDst,
                                int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateRemapInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                         nDstStep, oDstSizeROI, eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiRemap_32f_C3R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                    nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx);
}

NppStatus nppiRemap_32f_C1R(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const Npp32f *pXMap,
                            int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp32f *pDst, int nDstStep,
                            NppiSize oDstSizeROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiRemap_32f_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep,
                               oDstSizeROI, eInterpolation, nppStreamCtx);
}

NppStatus nppiRemap_32f_C3R(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const Npp32f *pXMap,
                            int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp32f *pDst, int nDstStep,
                            NppiSize oDstSizeROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiRemap_32f_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep,
                               oDstSizeROI, eInterpolation, nppStreamCtx);
}

// 32-bit float four channel remap
NppStatus nppiRemap_32f_C4R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp32f *pDst,
                                int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateRemapInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                         nDstStep, oDstSizeROI, eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiRemap_32f_C4R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                    nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx);
}

NppStatus nppiRemap_32f_C4R(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                            const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp32f *pDst,
                            int nDstStep, NppiSize oDstSizeROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiRemap_32f_C4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep,
                               oDstSizeROI, eInterpolation, nppStreamCtx);
}

// 32-bit float AC4 remap
NppStatus nppiRemap_32f_AC4R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                 const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp32f *pDst,
                                 int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateRemapInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                         nDstStep, oDstSizeROI, eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiRemap_32f_AC4R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                     nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx);
}

NppStatus nppiRemap_32f_AC4R(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                             const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp32f *pDst,
                             int nDstStep, NppiSize oDstSizeROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiRemap_32f_AC4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep,
                                oDstSizeROI, eInterpolation, nppStreamCtx);
}

// 32-bit float planar remap
NppStatus nppiRemap_32f_P3R_Ctx(const Npp32f *const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                Npp32f *pDst[3], int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                NppStreamContext nppStreamCtx) {
  NppStatus status = validateRemapPlanarInputs(reinterpret_cast<const void *const *>(pSrc), oSrcSize, nSrcStep, oSrcROI,
                                               pXMap, nXMapStep, pYMap, nYMapStep,
                                               reinterpret_cast<void **>(pDst), nDstStep, oDstSizeROI, eInterpolation, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  for (int i = 0; i < 3; i++) {
    status = nppiRemap_32f_C1R_Ctx_impl(pSrc[i], oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep,
                                       pDst[i], nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx);
    if (status != NPP_SUCCESS) {
      return status;
    }
  }

  return NPP_SUCCESS;
}

NppStatus nppiRemap_32f_P3R(const Npp32f *const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                            const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp32f *pDst[3],
                            int nDstStep, NppiSize oDstSizeROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiRemap_32f_P3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep,
                               oDstSizeROI, eInterpolation, nppStreamCtx);
}

NppStatus nppiRemap_32f_P4R_Ctx(const Npp32f *const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                Npp32f *pDst[4], int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                NppStreamContext nppStreamCtx) {
  NppStatus status = validateRemapPlanarInputs(reinterpret_cast<const void *const *>(pSrc), oSrcSize, nSrcStep, oSrcROI,
                                               pXMap, nXMapStep, pYMap, nYMapStep,
                                               reinterpret_cast<void **>(pDst), nDstStep, oDstSizeROI, eInterpolation, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  for (int i = 0; i < 4; i++) {
    status = nppiRemap_32f_C1R_Ctx_impl(pSrc[i], oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep,
                                       pDst[i], nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx);
    if (status != NPP_SUCCESS) {
      return status;
    }
  }

  return NPP_SUCCESS;
}

NppStatus nppiRemap_32f_P4R(const Npp32f *const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                            const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp32f *pDst[4],
                            int nDstStep, NppiSize oDstSizeROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiRemap_32f_P4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep,
                               oDstSizeROI, eInterpolation, nppStreamCtx);
}

// 64-bit double single channel remap
NppStatus nppiRemap_64f_C1R_Ctx(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                const Npp64f *pXMap, int nXMapStep, const Npp64f *pYMap, int nYMapStep, Npp64f *pDst,
                                int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateRemapInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                         nDstStep, oDstSizeROI, eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiRemap_64f_C1R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                    nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx);
}

// 64-bit double three channel remap
NppStatus nppiRemap_64f_C3R_Ctx(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                const Npp64f *pXMap, int nXMapStep, const Npp64f *pYMap, int nYMapStep, Npp64f *pDst,
                                int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateRemapInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                         nDstStep, oDstSizeROI, eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiRemap_64f_C3R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                    nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx);
}

NppStatus nppiRemap_64f_C1R(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const Npp64f *pXMap,
                            int nXMapStep, const Npp64f *pYMap, int nYMapStep, Npp64f *pDst, int nDstStep,
                            NppiSize oDstSizeROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiRemap_64f_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep,
                               oDstSizeROI, eInterpolation, nppStreamCtx);
}

NppStatus nppiRemap_64f_C3R(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const Npp64f *pXMap,
                            int nXMapStep, const Npp64f *pYMap, int nYMapStep, Npp64f *pDst, int nDstStep,
                            NppiSize oDstSizeROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiRemap_64f_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep,
                               oDstSizeROI, eInterpolation, nppStreamCtx);
}

// 64-bit double four channel remap
NppStatus nppiRemap_64f_C4R_Ctx(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                const Npp64f *pXMap, int nXMapStep, const Npp64f *pYMap, int nYMapStep, Npp64f *pDst,
                                int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateRemapInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                         nDstStep, oDstSizeROI, eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiRemap_64f_C4R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                    nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx);
}

NppStatus nppiRemap_64f_C4R(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                            const Npp64f *pXMap, int nXMapStep, const Npp64f *pYMap, int nYMapStep, Npp64f *pDst,
                            int nDstStep, NppiSize oDstSizeROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiRemap_64f_C4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep,
                               oDstSizeROI, eInterpolation, nppStreamCtx);
}

// 64-bit double AC4 remap
NppStatus nppiRemap_64f_AC4R_Ctx(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                 const Npp64f *pXMap, int nXMapStep, const Npp64f *pYMap, int nYMapStep, Npp64f *pDst,
                                 int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateRemapInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                         nDstStep, oDstSizeROI, eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiRemap_64f_AC4R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                     nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx);
}

NppStatus nppiRemap_64f_AC4R(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                             const Npp64f *pXMap, int nXMapStep, const Npp64f *pYMap, int nYMapStep, Npp64f *pDst,
                             int nDstStep, NppiSize oDstSizeROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiRemap_64f_AC4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep,
                                oDstSizeROI, eInterpolation, nppStreamCtx);
}

// 64-bit double planar remap
NppStatus nppiRemap_64f_P3R_Ctx(const Npp64f *const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                const Npp64f *pXMap, int nXMapStep, const Npp64f *pYMap, int nYMapStep,
                                Npp64f *pDst[3], int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                NppStreamContext nppStreamCtx) {
  NppStatus status = validateRemapPlanarInputs(reinterpret_cast<const void *const *>(pSrc), oSrcSize, nSrcStep, oSrcROI,
                                               pXMap, nXMapStep, pYMap, nYMapStep,
                                               reinterpret_cast<void **>(pDst), nDstStep, oDstSizeROI, eInterpolation, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  for (int i = 0; i < 3; i++) {
    status = nppiRemap_64f_C1R_Ctx_impl(pSrc[i], oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep,
                                       pDst[i], nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx);
    if (status != NPP_SUCCESS) {
      return status;
    }
  }

  return NPP_SUCCESS;
}

NppStatus nppiRemap_64f_P3R(const Npp64f *const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                            const Npp64f *pXMap, int nXMapStep, const Npp64f *pYMap, int nYMapStep, Npp64f *pDst[3],
                            int nDstStep, NppiSize oDstSizeROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiRemap_64f_P3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep,
                               oDstSizeROI, eInterpolation, nppStreamCtx);
}

NppStatus nppiRemap_64f_P4R_Ctx(const Npp64f *const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                const Npp64f *pXMap, int nXMapStep, const Npp64f *pYMap, int nYMapStep,
                                Npp64f *pDst[4], int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                NppStreamContext nppStreamCtx) {
  NppStatus status = validateRemapPlanarInputs(reinterpret_cast<const void *const *>(pSrc), oSrcSize, nSrcStep, oSrcROI,
                                               pXMap, nXMapStep, pYMap, nYMapStep,
                                               reinterpret_cast<void **>(pDst), nDstStep, oDstSizeROI, eInterpolation, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  for (int i = 0; i < 4; i++) {
    status = nppiRemap_64f_C1R_Ctx_impl(pSrc[i], oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep,
                                       pDst[i], nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx);
    if (status != NPP_SUCCESS) {
      return status;
    }
  }

  return NPP_SUCCESS;
}

NppStatus nppiRemap_64f_P4R(const Npp64f *const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                            const Npp64f *pXMap, int nXMapStep, const Npp64f *pYMap, int nYMapStep, Npp64f *pDst[4],
                            int nDstStep, NppiSize oDstSizeROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiRemap_64f_P4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep,
                               oDstSizeROI, eInterpolation, nppStreamCtx);
}
