// Absolute value operation using template-based API
#include "nppi_arithmetic_api.h"

using namespace nppi::arithmetic;

template <typename T, int C> using Abs = UnaryOpAPI<T, C, AbsOp>;

// ============================================================================
// Npp8s - Signed 8-bit
// ============================================================================

// C1R
NppStatus nppiAbs_8s_C1R_Ctx(const Npp8s *pSrc, int nSrcStep, Npp8s *pDst, int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx) {
  return Abs<Npp8s, 1>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_8s_C1R(const Npp8s *pSrc, int nSrcStep, Npp8s *pDst, int nDstStep, NppiSize oSizeROI) {
  return nppiAbs_8s_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C1IR (in-place)
NppStatus nppiAbs_8s_C1IR_Ctx(Npp8s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Abs<Npp8s, 1>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_8s_C1IR(Npp8s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAbs_8s_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// ============================================================================
// Npp16s - Signed 16-bit
// ============================================================================

// C1R
NppStatus nppiAbs_16s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx) {
  return Abs<Npp16s, 1>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_C1R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI) {
  return nppiAbs_16s_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C1IR (in-place)
NppStatus nppiAbs_16s_C1IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Abs<Npp16s, 1>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_C1IR(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAbs_16s_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// C3R
NppStatus nppiAbs_16s_C3R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx) {
  return Abs<Npp16s, 3>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_C3R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI) {
  return nppiAbs_16s_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C3IR (in-place)
NppStatus nppiAbs_16s_C3IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Abs<Npp16s, 3>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_C3IR(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAbs_16s_C3IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// AC4R
NppStatus nppiAbs_16s_AC4R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx) {
  return Abs<Npp16s, 4>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_AC4R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI) {
  return nppiAbs_16s_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// AC4IR (in-place)
NppStatus nppiAbs_16s_AC4IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Abs<Npp16s, 4>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_AC4IR(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAbs_16s_AC4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// C4R
NppStatus nppiAbs_16s_C4R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx) {
  return Abs<Npp16s, 4>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_C4R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI) {
  return nppiAbs_16s_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C4IR (in-place)
NppStatus nppiAbs_16s_C4IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Abs<Npp16s, 4>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_C4IR(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAbs_16s_C4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// ============================================================================
// Npp32s - Signed 32-bit
// ============================================================================

// C1R
NppStatus nppiAbs_32s_C1R_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx) {
  return Abs<Npp32s, 1>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32s_C1R(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oSizeROI) {
  return nppiAbs_32s_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C1IR (in-place)
NppStatus nppiAbs_32s_C1IR_Ctx(Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Abs<Npp32s, 1>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32s_C1IR(Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAbs_32s_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// ============================================================================
// Npp32f - 32-bit Float
// ============================================================================

// C1R
NppStatus nppiAbs_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx) {
  return Abs<Npp32f, 1>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI) {
  return nppiAbs_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C1IR (in-place)
NppStatus nppiAbs_32f_C1IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Abs<Npp32f, 1>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_C1IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAbs_32f_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// C3R
NppStatus nppiAbs_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx) {
  return Abs<Npp32f, 3>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_C3R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI) {
  return nppiAbs_32f_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C3IR (in-place)
NppStatus nppiAbs_32f_C3IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Abs<Npp32f, 3>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_C3IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAbs_32f_C3IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// AC4R
NppStatus nppiAbs_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx) {
  return Abs<Npp32f, 4>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_AC4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI) {
  return nppiAbs_32f_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// AC4IR (in-place)
NppStatus nppiAbs_32f_AC4IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Abs<Npp32f, 4>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_AC4IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAbs_32f_AC4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// C4R
NppStatus nppiAbs_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx) {
  return Abs<Npp32f, 4>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_C4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI) {
  return nppiAbs_32f_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C4IR (in-place)
NppStatus nppiAbs_32f_C4IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Abs<Npp32f, 4>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_C4IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAbs_32f_C4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}
