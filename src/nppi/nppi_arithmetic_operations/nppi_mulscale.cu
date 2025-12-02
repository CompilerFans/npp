// MulScale operation using template-based API
#include "nppi_arithmetic_api.h"

using namespace nppi::arithmetic;

template <typename T, int C> using MulScale = BinaryOpAPI<T, C, MulScaleOp>;

// ============================================================================
// Npp8u - Unsigned 8-bit
// ============================================================================

// C1R
NppStatus nppiMulScale_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MulScale<Npp8u, 1>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulScale_8u_C1R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI) {
  return nppiMulScale_8u_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C1IR (in-place)
NppStatus nppiMulScale_8u_C1IR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return MulScale<Npp8u, 1>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulScale_8u_C1IR(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiMulScale_8u_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// C3R
NppStatus nppiMulScale_8u_C3R_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MulScale<Npp8u, 3>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulScale_8u_C3R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI) {
  return nppiMulScale_8u_C3R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C3IR (in-place)
NppStatus nppiMulScale_8u_C3IR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return MulScale<Npp8u, 3>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulScale_8u_C3IR(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiMulScale_8u_C3IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// AC4R
NppStatus nppiMulScale_8u_AC4R_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MulScale<Npp8u, 4>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulScale_8u_AC4R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI) {
  return nppiMulScale_8u_AC4R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// AC4IR (in-place)
NppStatus nppiMulScale_8u_AC4IR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return MulScale<Npp8u, 4>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulScale_8u_AC4IR(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiMulScale_8u_AC4IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// C4R
NppStatus nppiMulScale_8u_C4R_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MulScale<Npp8u, 4>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulScale_8u_C4R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI) {
  return nppiMulScale_8u_C4R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C4IR (in-place)
NppStatus nppiMulScale_8u_C4IR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return MulScale<Npp8u, 4>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulScale_8u_C4IR(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiMulScale_8u_C4IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// ============================================================================
// Npp16u - Unsigned 16-bit
// ============================================================================

// C1R
NppStatus nppiMulScale_16u_C1R_Ctx(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                                   int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MulScale<Npp16u, 1>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulScale_16u_C1R(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                               int nDstStep, NppiSize oSizeROI) {
  return nppiMulScale_16u_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C1IR (in-place)
NppStatus nppiMulScale_16u_C1IR_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MulScale<Npp16u, 1>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulScale_16u_C1IR(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiMulScale_16u_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// C3R
NppStatus nppiMulScale_16u_C3R_Ctx(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                                   int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MulScale<Npp16u, 3>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulScale_16u_C3R(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                               int nDstStep, NppiSize oSizeROI) {
  return nppiMulScale_16u_C3R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C3IR (in-place)
NppStatus nppiMulScale_16u_C3IR_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MulScale<Npp16u, 3>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulScale_16u_C3IR(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiMulScale_16u_C3IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// AC4R
NppStatus nppiMulScale_16u_AC4R_Ctx(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                                    int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MulScale<Npp16u, 4>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulScale_16u_AC4R(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                                int nDstStep, NppiSize oSizeROI) {
  return nppiMulScale_16u_AC4R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// AC4IR (in-place)
NppStatus nppiMulScale_16u_AC4IR_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MulScale<Npp16u, 4>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulScale_16u_AC4IR(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiMulScale_16u_AC4IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// C4R
NppStatus nppiMulScale_16u_C4R_Ctx(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                                   int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MulScale<Npp16u, 4>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulScale_16u_C4R(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                               int nDstStep, NppiSize oSizeROI) {
  return nppiMulScale_16u_C4R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C4IR (in-place)
NppStatus nppiMulScale_16u_C4IR_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MulScale<Npp16u, 4>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulScale_16u_C4IR(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiMulScale_16u_C4IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}
