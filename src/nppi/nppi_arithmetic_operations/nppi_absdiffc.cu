// Absolute difference with constant operation using template-based API
#include "nppi_arithmetic_api.h"

using namespace nppi::arithmetic;

template <typename T, int C> using AbsDiffC = ConstOpAPI<T, C, AbsDiffConstOp>;

// ============================================================================
// Npp8u - Unsigned 8-bit
// ============================================================================

// C1R
NppStatus nppiAbsDiffC_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  Npp8u nConstant, NppStreamContext nppStreamCtx) {
  return AbsDiffC<Npp8u, 1>::execute(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiffC_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                              Npp8u nConstant) {
  return nppiAbsDiffC_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nConstant, getDefaultStreamContext());
}

// ============================================================================
// Npp16u - Unsigned 16-bit
// ============================================================================

// C1R
NppStatus nppiAbsDiffC_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                   Npp16u nConstant, NppStreamContext nppStreamCtx) {
  return AbsDiffC<Npp16u, 1>::execute(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiffC_16u_C1R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                               Npp16u nConstant) {
  return nppiAbsDiffC_16u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nConstant, getDefaultStreamContext());
}

// ============================================================================
// Npp32f - 32-bit Float
// ============================================================================

// C1R
NppStatus nppiAbsDiffC_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                   Npp32f nConstant, NppStreamContext nppStreamCtx) {
  return AbsDiffC<Npp32f, 1>::execute(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiffC_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                               Npp32f nConstant) {
  return nppiAbsDiffC_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nConstant, getDefaultStreamContext());
}
