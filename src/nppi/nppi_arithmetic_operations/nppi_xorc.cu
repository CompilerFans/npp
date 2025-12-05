// Logical XOR with constant operation using template-based API
#include "nppi_arithmetic_api.h"

using namespace nppi::arithmetic;

template <typename T, int C> using XorC = LogicalConstOpAPI<T, C, XorConstOp>;
template <typename T> using XorCAC4 = LogicalConstOpAC4API<T, XorConstOp>;

// ============================================================================
// 8u variants
// ============================================================================

NppStatus nppiXorC_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u nConstant, Npp8u *pDst, int nDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return XorC<Npp8u, 1>::execute(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u nConstant, Npp8u *pDst, int nDstStep,
                          NppiSize oSizeROI) {
  return nppiXorC_8u_C1R_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiXorC_8u_C1IR_Ctx(Npp8u nConstant, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx) {
  return XorC<Npp8u, 1>::executeInplace(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_8u_C1IR(Npp8u nConstant, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiXorC_8u_C1IR_Ctx(nConstant, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiXorC_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return XorC<Npp8u, 3>::executeMulti(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_8u_C3R(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                          NppiSize oSizeROI) {
  return nppiXorC_8u_C3R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiXorC_8u_C3IR_Ctx(const Npp8u aConstants[3], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx) {
  return XorC<Npp8u, 3>::executeMultiInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_8u_C3IR(const Npp8u aConstants[3], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiXorC_8u_C3IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiXorC_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return XorCAC4<Npp8u>::executeMulti(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_8u_AC4R(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                           NppiSize oSizeROI) {
  return nppiXorC_8u_AC4R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiXorC_8u_AC4IR_Ctx(const Npp8u aConstants[3], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx) {
  return XorCAC4<Npp8u>::executeMultiInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_8u_AC4IR(const Npp8u aConstants[3], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiXorC_8u_AC4IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiXorC_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[4], Npp8u *pDst, int nDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return XorC<Npp8u, 4>::executeMulti(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_8u_C4R(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[4], Npp8u *pDst, int nDstStep,
                          NppiSize oSizeROI) {
  return nppiXorC_8u_C4R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiXorC_8u_C4IR_Ctx(const Npp8u aConstants[4], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx) {
  return XorC<Npp8u, 4>::executeMultiInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_8u_C4IR(const Npp8u aConstants[4], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiXorC_8u_C4IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// ============================================================================
// 16u variants
// ============================================================================

NppStatus nppiXorC_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u nConstant, Npp16u *pDst, int nDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return XorC<Npp16u, 1>::execute(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_16u_C1R(const Npp16u *pSrc, int nSrcStep, Npp16u nConstant, Npp16u *pDst, int nDstStep,
                           NppiSize oSizeROI) {
  return nppiXorC_16u_C1R_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiXorC_16u_C1IR_Ctx(Npp16u nConstant, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx) {
  return XorC<Npp16u, 1>::executeInplace(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_16u_C1IR(Npp16u nConstant, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiXorC_16u_C1IR_Ctx(nConstant, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiXorC_16u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[3], Npp16u *pDst, int nDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return XorC<Npp16u, 3>::executeMulti(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_16u_C3R(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[3], Npp16u *pDst, int nDstStep,
                           NppiSize oSizeROI) {
  return nppiXorC_16u_C3R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiXorC_16u_C3IR_Ctx(const Npp16u aConstants[3], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx) {
  return XorC<Npp16u, 3>::executeMultiInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_16u_C3IR(const Npp16u aConstants[3], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiXorC_16u_C3IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiXorC_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[3], Npp16u *pDst,
                                int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return XorCAC4<Npp16u>::executeMulti(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_16u_AC4R(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[3], Npp16u *pDst, int nDstStep,
                            NppiSize oSizeROI) {
  return nppiXorC_16u_AC4R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiXorC_16u_AC4IR_Ctx(const Npp16u aConstants[3], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx) {
  return XorCAC4<Npp16u>::executeMultiInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_16u_AC4IR(const Npp16u aConstants[3], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiXorC_16u_AC4IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiXorC_16u_C4R_Ctx(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[4], Npp16u *pDst, int nDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return XorC<Npp16u, 4>::executeMulti(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_16u_C4R(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[4], Npp16u *pDst, int nDstStep,
                           NppiSize oSizeROI) {
  return nppiXorC_16u_C4R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiXorC_16u_C4IR_Ctx(const Npp16u aConstants[4], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx) {
  return XorC<Npp16u, 4>::executeMultiInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_16u_C4IR(const Npp16u aConstants[4], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiXorC_16u_C4IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// ============================================================================
// 32s variants
// ============================================================================

NppStatus nppiXorC_32s_C1R_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s nConstant, Npp32s *pDst, int nDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return XorC<Npp32s, 1>::execute(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_32s_C1R(const Npp32s *pSrc, int nSrcStep, Npp32s nConstant, Npp32s *pDst, int nDstStep,
                           NppiSize oSizeROI) {
  return nppiXorC_32s_C1R_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiXorC_32s_C1IR_Ctx(Npp32s nConstant, Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx) {
  return XorC<Npp32s, 1>::executeInplace(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_32s_C1IR(Npp32s nConstant, Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiXorC_32s_C1IR_Ctx(nConstant, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiXorC_32s_C3R_Ctx(const Npp32s *pSrc, int nSrcStep, const Npp32s aConstants[3], Npp32s *pDst, int nDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return XorC<Npp32s, 3>::executeMulti(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_32s_C3R(const Npp32s *pSrc, int nSrcStep, const Npp32s aConstants[3], Npp32s *pDst, int nDstStep,
                           NppiSize oSizeROI) {
  return nppiXorC_32s_C3R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiXorC_32s_C3IR_Ctx(const Npp32s aConstants[3], Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx) {
  return XorC<Npp32s, 3>::executeMultiInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_32s_C3IR(const Npp32s aConstants[3], Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiXorC_32s_C3IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiXorC_32s_AC4R_Ctx(const Npp32s *pSrc, int nSrcStep, const Npp32s aConstants[3], Npp32s *pDst,
                                int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return XorCAC4<Npp32s>::executeMulti(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_32s_AC4R(const Npp32s *pSrc, int nSrcStep, const Npp32s aConstants[3], Npp32s *pDst, int nDstStep,
                            NppiSize oSizeROI) {
  return nppiXorC_32s_AC4R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiXorC_32s_AC4IR_Ctx(const Npp32s aConstants[3], Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx) {
  return XorCAC4<Npp32s>::executeMultiInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_32s_AC4IR(const Npp32s aConstants[3], Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiXorC_32s_AC4IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiXorC_32s_C4R_Ctx(const Npp32s *pSrc, int nSrcStep, const Npp32s aConstants[4], Npp32s *pDst, int nDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return XorC<Npp32s, 4>::executeMulti(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_32s_C4R(const Npp32s *pSrc, int nSrcStep, const Npp32s aConstants[4], Npp32s *pDst, int nDstStep,
                           NppiSize oSizeROI) {
  return nppiXorC_32s_C4R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiXorC_32s_C4IR_Ctx(const Npp32s aConstants[4], Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx) {
  return XorC<Npp32s, 4>::executeMultiInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_32s_C4IR(const Npp32s aConstants[4], Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiXorC_32s_C4IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}
