// Absolute difference operation using template-based API
#include "nppi_arithmetic_api.h"

using namespace nppi::arithmetic;

template <typename T, int C> using AbsDiff = BinaryOpAPI<T, C, AbsDiffOp>;

// ============================================================================
// Npp8u - Unsigned 8-bit
// ============================================================================

// C1R
NppStatus nppiAbsDiff_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return AbsDiff<Npp8u, 1>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiff_8u_C1R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI) {
  return nppiAbsDiff_8u_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C1IR (in-place)
NppStatus nppiAbsDiff_8u_C1IR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
  return AbsDiff<Npp8u, 1>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiff_8u_C1IR(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAbsDiff_8u_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// C3R
NppStatus nppiAbsDiff_8u_C3R_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return AbsDiff<Npp8u, 3>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiff_8u_C3R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI) {
  return nppiAbsDiff_8u_C3R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C3IR (in-place)
NppStatus nppiAbsDiff_8u_C3IR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
  return AbsDiff<Npp8u, 3>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiff_8u_C3IR(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAbsDiff_8u_C3IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// ============================================================================
// Npp16u - Unsigned 16-bit
// ============================================================================

// C1R
NppStatus nppiAbsDiff_16u_C1R_Ctx(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                                  int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return AbsDiff<Npp16u, 1>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiff_16u_C1R(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI) {
  return nppiAbsDiff_16u_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// ============================================================================
// Npp32f - 32-bit Float
// ============================================================================

// C1R
NppStatus nppiAbsDiff_32f_C1R_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                                  int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return AbsDiff<Npp32f, 1>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiff_32f_C1R(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI) {
  return nppiAbsDiff_32f_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C1IR (in-place)
NppStatus nppiAbsDiff_32f_C1IR_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return AbsDiff<Npp32f, 1>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiff_32f_C1IR(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAbsDiff_32f_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}
