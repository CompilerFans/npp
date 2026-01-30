#include "npp.h"

extern "C" {
NppStatus nppiGradientColorToGray_8u_C3C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                                    NppiSize oSizeROI, NppiNorm eNorm,
                                                    NppStreamContext nppStreamCtx);
NppStatus nppiGradientColorToGray_16u_C3C1R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
                                                     NppiSize oSizeROI, NppiNorm eNorm,
                                                     NppStreamContext nppStreamCtx);
NppStatus nppiGradientColorToGray_16s_C3C1R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep,
                                                     NppiSize oSizeROI, NppiNorm eNorm,
                                                     NppStreamContext nppStreamCtx);
NppStatus nppiGradientColorToGray_32f_C3C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                                     NppiSize oSizeROI, NppiNorm eNorm,
                                                     NppStreamContext nppStreamCtx);
}

static inline NppStatus validateGradientColorToGrayInputs(const void *pSrc, int nSrcStep, void *pDst, int nDstStep,
                                                          NppiSize oSizeROI, NppiNorm eNorm) {
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (eNorm != nppiNormL1 && eNorm != nppiNormL2 && eNorm != nppiNormInf) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  return NPP_SUCCESS;
}

NppStatus nppiGradientColorToGray_8u_C3C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                               NppiSize oSizeROI, NppiNorm eNorm,
                                               NppStreamContext nppStreamCtx) {
  NppStatus status = validateGradientColorToGrayInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eNorm);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiGradientColorToGray_8u_C3C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eNorm, nppStreamCtx);
}

NppStatus nppiGradientColorToGray_8u_C3C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                           NppiSize oSizeROI, NppiNorm eNorm) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiGradientColorToGray_8u_C3C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eNorm, nppStreamCtx);
}

NppStatus nppiGradientColorToGray_16u_C3C1R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
                                                NppiSize oSizeROI, NppiNorm eNorm,
                                                NppStreamContext nppStreamCtx) {
  NppStatus status = validateGradientColorToGrayInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eNorm);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiGradientColorToGray_16u_C3C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eNorm, nppStreamCtx);
}

NppStatus nppiGradientColorToGray_16u_C3C1R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
                                            NppiSize oSizeROI, NppiNorm eNorm) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiGradientColorToGray_16u_C3C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eNorm, nppStreamCtx);
}

NppStatus nppiGradientColorToGray_16s_C3C1R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep,
                                                NppiSize oSizeROI, NppiNorm eNorm,
                                                NppStreamContext nppStreamCtx) {
  NppStatus status = validateGradientColorToGrayInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eNorm);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiGradientColorToGray_16s_C3C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eNorm, nppStreamCtx);
}

NppStatus nppiGradientColorToGray_16s_C3C1R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep,
                                            NppiSize oSizeROI, NppiNorm eNorm) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiGradientColorToGray_16s_C3C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eNorm, nppStreamCtx);
}

NppStatus nppiGradientColorToGray_32f_C3C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                                NppiSize oSizeROI, NppiNorm eNorm,
                                                NppStreamContext nppStreamCtx) {
  NppStatus status = validateGradientColorToGrayInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eNorm);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiGradientColorToGray_32f_C3C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eNorm, nppStreamCtx);
}

NppStatus nppiGradientColorToGray_32f_C3C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                            NppiSize oSizeROI, NppiNorm eNorm) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiGradientColorToGray_32f_C3C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eNorm, nppStreamCtx);
}
