#include "npp.h"

extern "C" {
NppStatus nppiColorToGray_8u_C3C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                            NppiSize oSizeROI, const Npp32f aCoeffs[3],
                                            NppStreamContext nppStreamCtx);
NppStatus nppiColorToGray_8u_AC4C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                             NppiSize oSizeROI, const Npp32f aCoeffs[3],
                                             NppStreamContext nppStreamCtx);
NppStatus nppiColorToGray_8u_C4C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                            NppiSize oSizeROI, const Npp32f aCoeffs[4],
                                            NppStreamContext nppStreamCtx);
NppStatus nppiColorToGray_16u_C3C1R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
                                             NppiSize oSizeROI, const Npp32f aCoeffs[3],
                                             NppStreamContext nppStreamCtx);
NppStatus nppiColorToGray_16u_AC4C1R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
                                              NppiSize oSizeROI, const Npp32f aCoeffs[3],
                                              NppStreamContext nppStreamCtx);
NppStatus nppiColorToGray_16u_C4C1R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
                                             NppiSize oSizeROI, const Npp32f aCoeffs[4],
                                             NppStreamContext nppStreamCtx);
NppStatus nppiColorToGray_16s_C3C1R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep,
                                             NppiSize oSizeROI, const Npp32f aCoeffs[3],
                                             NppStreamContext nppStreamCtx);
NppStatus nppiColorToGray_16s_AC4C1R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep,
                                              NppiSize oSizeROI, const Npp32f aCoeffs[3],
                                              NppStreamContext nppStreamCtx);
NppStatus nppiColorToGray_16s_C4C1R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep,
                                             NppiSize oSizeROI, const Npp32f aCoeffs[4],
                                             NppStreamContext nppStreamCtx);
NppStatus nppiColorToGray_32f_C3C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                             NppiSize oSizeROI, const Npp32f aCoeffs[3],
                                             NppStreamContext nppStreamCtx);
NppStatus nppiColorToGray_32f_AC4C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                              NppiSize oSizeROI, const Npp32f aCoeffs[3],
                                              NppStreamContext nppStreamCtx);
NppStatus nppiColorToGray_32f_C4C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                             NppiSize oSizeROI, const Npp32f aCoeffs[4],
                                             NppStreamContext nppStreamCtx);
}

static inline NppStatus validateColorToGrayInputs(const void *pSrc, int nSrcStep, void *pDst, int nDstStep,
                                                  NppiSize oSizeROI, const Npp32f *aCoeffs) {
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }
  if (!pSrc || !pDst || !aCoeffs) {
    return NPP_NULL_POINTER_ERROR;
  }
  return NPP_SUCCESS;
}

NppStatus nppiColorToGray_8u_C3C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI, const Npp32f aCoeffs[3],
                                       NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorToGrayInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiColorToGray_8u_C3C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs, nppStreamCtx);
}

NppStatus nppiColorToGray_8u_C3C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   const Npp32f aCoeffs[3]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorToGray_8u_C3C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs, nppStreamCtx);
}

NppStatus nppiColorToGray_8u_AC4C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                        NppiSize oSizeROI, const Npp32f aCoeffs[3],
                                        NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorToGrayInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiColorToGray_8u_AC4C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs, nppStreamCtx);
}

NppStatus nppiColorToGray_8u_AC4C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                    const Npp32f aCoeffs[3]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorToGray_8u_AC4C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs, nppStreamCtx);
}

NppStatus nppiColorToGray_8u_C4C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI, const Npp32f aCoeffs[4],
                                       NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorToGrayInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiColorToGray_8u_C4C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs, nppStreamCtx);
}

NppStatus nppiColorToGray_8u_C4C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   const Npp32f aCoeffs[4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorToGray_8u_C4C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs, nppStreamCtx);
}

NppStatus nppiColorToGray_16u_C3C1R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
                                        NppiSize oSizeROI, const Npp32f aCoeffs[3],
                                        NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorToGrayInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiColorToGray_16u_C3C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs, nppStreamCtx);
}

NppStatus nppiColorToGray_16u_C3C1R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp32f aCoeffs[3]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorToGray_16u_C3C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs, nppStreamCtx);
}

NppStatus nppiColorToGray_16u_AC4C1R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
                                         NppiSize oSizeROI, const Npp32f aCoeffs[3],
                                         NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorToGrayInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiColorToGray_16u_AC4C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs, nppStreamCtx);
}

NppStatus nppiColorToGray_16u_AC4C1R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
                                     NppiSize oSizeROI, const Npp32f aCoeffs[3]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorToGray_16u_AC4C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs, nppStreamCtx);
}

NppStatus nppiColorToGray_16u_C4C1R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
                                        NppiSize oSizeROI, const Npp32f aCoeffs[4],
                                        NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorToGrayInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiColorToGray_16u_C4C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs, nppStreamCtx);
}

NppStatus nppiColorToGray_16u_C4C1R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                    const Npp32f aCoeffs[4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorToGray_16u_C4C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs, nppStreamCtx);
}

NppStatus nppiColorToGray_16s_C3C1R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep,
                                        NppiSize oSizeROI, const Npp32f aCoeffs[3],
                                        NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorToGrayInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiColorToGray_16s_C3C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs, nppStreamCtx);
}

NppStatus nppiColorToGray_16s_C3C1R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp32f aCoeffs[3]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorToGray_16s_C3C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs, nppStreamCtx);
}

NppStatus nppiColorToGray_16s_AC4C1R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep,
                                         NppiSize oSizeROI, const Npp32f aCoeffs[3],
                                         NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorToGrayInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiColorToGray_16s_AC4C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs, nppStreamCtx);
}

NppStatus nppiColorToGray_16s_AC4C1R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep,
                                     NppiSize oSizeROI, const Npp32f aCoeffs[3]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorToGray_16s_AC4C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs, nppStreamCtx);
}

NppStatus nppiColorToGray_16s_C4C1R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep,
                                        NppiSize oSizeROI, const Npp32f aCoeffs[4],
                                        NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorToGrayInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiColorToGray_16s_C4C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs, nppStreamCtx);
}

NppStatus nppiColorToGray_16s_C4C1R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                    const Npp32f aCoeffs[4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorToGray_16s_C4C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs, nppStreamCtx);
}

NppStatus nppiColorToGray_32f_C3C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                        NppiSize oSizeROI, const Npp32f aCoeffs[3],
                                        NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorToGrayInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiColorToGray_32f_C3C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs, nppStreamCtx);
}

NppStatus nppiColorToGray_32f_C3C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp32f aCoeffs[3]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorToGray_32f_C3C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs, nppStreamCtx);
}

NppStatus nppiColorToGray_32f_AC4C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                         NppiSize oSizeROI, const Npp32f aCoeffs[3],
                                         NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorToGrayInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiColorToGray_32f_AC4C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs, nppStreamCtx);
}

NppStatus nppiColorToGray_32f_AC4C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                     NppiSize oSizeROI, const Npp32f aCoeffs[3]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorToGray_32f_AC4C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs, nppStreamCtx);
}

NppStatus nppiColorToGray_32f_C4C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                        NppiSize oSizeROI, const Npp32f aCoeffs[4],
                                        NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorToGrayInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiColorToGray_32f_C4C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs, nppStreamCtx);
}

NppStatus nppiColorToGray_32f_C4C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                    const Npp32f aCoeffs[4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorToGray_32f_C4C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aCoeffs, nppStreamCtx);
}
