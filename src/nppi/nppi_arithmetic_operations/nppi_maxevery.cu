// MaxEvery operation using template-based API
#include "nppi_arithmetic_api.h"

using namespace nppi::arithmetic;

template <typename T, int C> using MaxEvery = BinaryOpAPI<T, C, MaxEveryOp>;

// ============================================================================
// Npp8u - Unsigned 8-bit
// ============================================================================

// C1IR (in-place only)
NppStatus nppiMaxEvery_8u_C1IR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return MaxEvery<Npp8u, 1>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMaxEvery_8u_C1IR(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiMaxEvery_8u_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// ============================================================================
// Npp16u - Unsigned 16-bit
// ============================================================================

// C1IR (in-place only)
NppStatus nppiMaxEvery_16u_C1IR_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MaxEvery<Npp16u, 1>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMaxEvery_16u_C1IR(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiMaxEvery_16u_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// ============================================================================
// Npp16s - Signed 16-bit
// ============================================================================

// C1IR (in-place only)
NppStatus nppiMaxEvery_16s_C1IR_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MaxEvery<Npp16s, 1>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMaxEvery_16s_C1IR(const Npp16s *pSrc, int nSrcStep, Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiMaxEvery_16s_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// ============================================================================
// Npp32s - Signed 32-bit
// ============================================================================

// C1IR (in-place only)
NppStatus nppiMaxEvery_32s_C1IR_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MaxEvery<Npp32s, 1>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMaxEvery_32s_C1IR(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiMaxEvery_32s_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// ============================================================================
// Npp32f - 32-bit Float
// ============================================================================

// C1IR (in-place only)
NppStatus nppiMaxEvery_32f_C1IR_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MaxEvery<Npp32f, 1>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMaxEvery_32f_C1IR(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiMaxEvery_32f_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}
