// MulCScale operation using template-based API
// Multiplies each pixel by a constant and scales the result
#include "nppi_arithmetic_api.h"

using namespace nppi::arithmetic;

template <typename T, int C> using MulCScale = ConstOpAPI<T, C, MulCScaleOp>;
template <typename T, int C> using MulCScaleMulti = MultiConstOpAPI<T, C, MulCScaleOp>;

// ============================================================================
// Npp8u - Unsigned 8-bit
// ============================================================================

// C1R
NppStatus nppiMulCScale_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u nConstant, Npp8u *pDst, int nDstStep,
                                   NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MulCScale<Npp8u, 1>::execute(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulCScale_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u nConstant, Npp8u *pDst, int nDstStep,
                               NppiSize oSizeROI) {
  return nppiMulCScale_8u_C1R_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C1IR (in-place)
NppStatus nppiMulCScale_8u_C1IR_Ctx(Npp8u nConstant, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return MulCScale<Npp8u, 1>::executeInplace(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulCScale_8u_C1IR(Npp8u nConstant, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiMulCScale_8u_C1IR_Ctx(nConstant, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// C3R
NppStatus nppiMulCScale_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[3], Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MulCScaleMulti<Npp8u, 3>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulCScale_8u_C3R(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                               NppiSize oSizeROI) {
  return nppiMulCScale_8u_C3R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C3IR (in-place)
NppStatus nppiMulCScale_8u_C3IR_Ctx(const Npp8u aConstants[3], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return MulCScaleMulti<Npp8u, 3>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulCScale_8u_C3IR(const Npp8u aConstants[3], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiMulCScale_8u_C3IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// AC4R
NppStatus nppiMulCScale_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[3], Npp8u *pDst,
                                    int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MulCScaleMulti<Npp8u, 4>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulCScale_8u_AC4R(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                                NppiSize oSizeROI) {
  return nppiMulCScale_8u_AC4R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// AC4IR (in-place)
NppStatus nppiMulCScale_8u_AC4IR_Ctx(const Npp8u aConstants[3], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx) {
  return MulCScaleMulti<Npp8u, 4>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulCScale_8u_AC4IR(const Npp8u aConstants[3], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiMulCScale_8u_AC4IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// C4R
NppStatus nppiMulCScale_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[4], Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MulCScaleMulti<Npp8u, 4>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulCScale_8u_C4R(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[4], Npp8u *pDst, int nDstStep,
                               NppiSize oSizeROI) {
  return nppiMulCScale_8u_C4R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C4IR (in-place)
NppStatus nppiMulCScale_8u_C4IR_Ctx(const Npp8u aConstants[4], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return MulCScaleMulti<Npp8u, 4>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulCScale_8u_C4IR(const Npp8u aConstants[4], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiMulCScale_8u_C4IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// ============================================================================
// Npp16u - Unsigned 16-bit
// ============================================================================

// C1R
NppStatus nppiMulCScale_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u nConstant, Npp16u *pDst, int nDstStep,
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MulCScale<Npp16u, 1>::execute(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulCScale_16u_C1R(const Npp16u *pSrc, int nSrcStep, Npp16u nConstant, Npp16u *pDst, int nDstStep,
                                NppiSize oSizeROI) {
  return nppiMulCScale_16u_C1R_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C1IR (in-place)
NppStatus nppiMulCScale_16u_C1IR_Ctx(Npp16u nConstant, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx) {
  return MulCScale<Npp16u, 1>::executeInplace(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulCScale_16u_C1IR(Npp16u nConstant, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiMulCScale_16u_C1IR_Ctx(nConstant, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// C3R
NppStatus nppiMulCScale_16u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[3], Npp16u *pDst,
                                    int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MulCScaleMulti<Npp16u, 3>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulCScale_16u_C3R(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[3], Npp16u *pDst,
                                int nDstStep, NppiSize oSizeROI) {
  return nppiMulCScale_16u_C3R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C3IR (in-place)
NppStatus nppiMulCScale_16u_C3IR_Ctx(const Npp16u aConstants[3], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx) {
  return MulCScaleMulti<Npp16u, 3>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulCScale_16u_C3IR(const Npp16u aConstants[3], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiMulCScale_16u_C3IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// AC4R
NppStatus nppiMulCScale_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[3], Npp16u *pDst,
                                     int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MulCScaleMulti<Npp16u, 4>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulCScale_16u_AC4R(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[3], Npp16u *pDst,
                                 int nDstStep, NppiSize oSizeROI) {
  return nppiMulCScale_16u_AC4R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// AC4IR (in-place)
NppStatus nppiMulCScale_16u_AC4IR_Ctx(const Npp16u aConstants[3], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx) {
  return MulCScaleMulti<Npp16u, 4>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulCScale_16u_AC4IR(const Npp16u aConstants[3], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiMulCScale_16u_AC4IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// C4R
NppStatus nppiMulCScale_16u_C4R_Ctx(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[4], Npp16u *pDst,
                                    int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MulCScaleMulti<Npp16u, 4>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulCScale_16u_C4R(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[4], Npp16u *pDst,
                                int nDstStep, NppiSize oSizeROI) {
  return nppiMulCScale_16u_C4R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C4IR (in-place)
NppStatus nppiMulCScale_16u_C4IR_Ctx(const Npp16u aConstants[4], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx) {
  return MulCScaleMulti<Npp16u, 4>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulCScale_16u_C4IR(const Npp16u aConstants[4], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiMulCScale_16u_C4IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}
