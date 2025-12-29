// NOT operation using template-based API
#include "nppi_arithmetic_api.h"

using namespace nppi::arithmetic;

template <typename T, int C> using Not = UnaryOpAPI<T, C, NotOp>;
template <typename T> using NotAC4 = UnaryOpAC4API<T, NotOp>;

// ============================================================================
// Npp8u - Unsigned 8-bit
// ============================================================================

// C1R
NppStatus nppiNot_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx) {
  return Not<Npp8u, 1>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiNot_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  return nppiNot_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C1IR (in-place)
NppStatus nppiNot_8u_C1IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Not<Npp8u, 1>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiNot_8u_C1IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiNot_8u_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// C3R
NppStatus nppiNot_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx) {
  return Not<Npp8u, 3>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiNot_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  return nppiNot_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C3IR (in-place)
NppStatus nppiNot_8u_C3IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Not<Npp8u, 3>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiNot_8u_C3IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiNot_8u_C3IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// AC4R
NppStatus nppiNot_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx) {
  return NotAC4<Npp8u>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiNot_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  return nppiNot_8u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// AC4IR (in-place)
NppStatus nppiNot_8u_AC4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return NotAC4<Npp8u>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiNot_8u_AC4IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiNot_8u_AC4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// C4R
NppStatus nppiNot_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx) {
  return Not<Npp8u, 4>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiNot_8u_C4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  return nppiNot_8u_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C4IR (in-place)
NppStatus nppiNot_8u_C4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Not<Npp8u, 4>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiNot_8u_C4IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiNot_8u_C4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}
