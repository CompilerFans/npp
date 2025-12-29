// Left shift by constant operation using template-based API
#include "nppi_arithmetic_api.h"

using namespace nppi::arithmetic;

// ============================================================================
// Npp8u - Unsigned 8-bit
// ============================================================================

// C1R
NppStatus nppiLShiftC_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp32u nConstant, Npp8u *pDst, int nDstStep,
                                 NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return LShift<Npp8u, 1>::execute(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLShiftC_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp32u nConstant, Npp8u *pDst, int nDstStep,
                             NppiSize oSizeROI) {
  return nppiLShiftC_8u_C1R_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C1IR (in-place)
NppStatus nppiLShiftC_8u_C1IR_Ctx(Npp32u nConstant, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
  return LShift<Npp8u, 1>::executeInplace(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLShiftC_8u_C1IR(Npp32u nConstant, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiLShiftC_8u_C1IR_Ctx(nConstant, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// C3R
NppStatus nppiLShiftC_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp32u aConstants[3], Npp8u *pDst, int nDstStep,
                                 NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return LShiftMulti<Npp8u, 3>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLShiftC_8u_C3R(const Npp8u *pSrc, int nSrcStep, const Npp32u aConstants[3], Npp8u *pDst, int nDstStep,
                             NppiSize oSizeROI) {
  return nppiLShiftC_8u_C3R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C3IR (in-place)
NppStatus nppiLShiftC_8u_C3IR_Ctx(const Npp32u aConstants[3], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
  return LShiftMulti<Npp8u, 3>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLShiftC_8u_C3IR(const Npp32u aConstants[3], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiLShiftC_8u_C3IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// AC4R
NppStatus nppiLShiftC_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp32u aConstants[3], Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return LShiftMultiAC4<Npp8u>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLShiftC_8u_AC4R(const Npp8u *pSrc, int nSrcStep, const Npp32u aConstants[3], Npp8u *pDst, int nDstStep,
                              NppiSize oSizeROI) {
  return nppiLShiftC_8u_AC4R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// AC4IR (in-place)
NppStatus nppiLShiftC_8u_AC4IR_Ctx(const Npp32u aConstants[3], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return LShiftMultiAC4<Npp8u>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLShiftC_8u_AC4IR(const Npp32u aConstants[3], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiLShiftC_8u_AC4IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// C4R
NppStatus nppiLShiftC_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp32u aConstants[4], Npp8u *pDst, int nDstStep,
                                 NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return LShiftMulti<Npp8u, 4>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLShiftC_8u_C4R(const Npp8u *pSrc, int nSrcStep, const Npp32u aConstants[4], Npp8u *pDst, int nDstStep,
                             NppiSize oSizeROI) {
  return nppiLShiftC_8u_C4R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C4IR (in-place)
NppStatus nppiLShiftC_8u_C4IR_Ctx(const Npp32u aConstants[4], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
  return LShiftMulti<Npp8u, 4>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLShiftC_8u_C4IR(const Npp32u aConstants[4], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiLShiftC_8u_C4IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// ============================================================================
// Npp16u - Unsigned 16-bit
// ============================================================================

// C1R
NppStatus nppiLShiftC_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp32u nConstant, Npp16u *pDst, int nDstStep,
                                  NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return LShift<Npp16u, 1>::execute(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLShiftC_16u_C1R(const Npp16u *pSrc, int nSrcStep, Npp32u nConstant, Npp16u *pDst, int nDstStep,
                              NppiSize oSizeROI) {
  return nppiLShiftC_16u_C1R_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C1IR (in-place)
NppStatus nppiLShiftC_16u_C1IR_Ctx(Npp32u nConstant, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return LShift<Npp16u, 1>::executeInplace(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLShiftC_16u_C1IR(Npp32u nConstant, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiLShiftC_16u_C1IR_Ctx(nConstant, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// C3R
NppStatus nppiLShiftC_16u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep, const Npp32u aConstants[3], Npp16u *pDst,
                                  int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return LShiftMulti<Npp16u, 3>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLShiftC_16u_C3R(const Npp16u *pSrc, int nSrcStep, const Npp32u aConstants[3], Npp16u *pDst, int nDstStep,
                              NppiSize oSizeROI) {
  return nppiLShiftC_16u_C3R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C3IR (in-place)
NppStatus nppiLShiftC_16u_C3IR_Ctx(const Npp32u aConstants[3], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return LShiftMulti<Npp16u, 3>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLShiftC_16u_C3IR(const Npp32u aConstants[3], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiLShiftC_16u_C3IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// AC4R
NppStatus nppiLShiftC_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep, const Npp32u aConstants[3], Npp16u *pDst,
                                   int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return LShiftMultiAC4<Npp16u>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLShiftC_16u_AC4R(const Npp16u *pSrc, int nSrcStep, const Npp32u aConstants[3], Npp16u *pDst, int nDstStep,
                               NppiSize oSizeROI) {
  return nppiLShiftC_16u_AC4R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// AC4IR (in-place)
NppStatus nppiLShiftC_16u_AC4IR_Ctx(const Npp32u aConstants[3], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return LShiftMultiAC4<Npp16u>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLShiftC_16u_AC4IR(const Npp32u aConstants[3], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiLShiftC_16u_AC4IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// C4R
NppStatus nppiLShiftC_16u_C4R_Ctx(const Npp16u *pSrc, int nSrcStep, const Npp32u aConstants[4], Npp16u *pDst,
                                  int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return LShiftMulti<Npp16u, 4>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLShiftC_16u_C4R(const Npp16u *pSrc, int nSrcStep, const Npp32u aConstants[4], Npp16u *pDst, int nDstStep,
                              NppiSize oSizeROI) {
  return nppiLShiftC_16u_C4R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C4IR (in-place)
NppStatus nppiLShiftC_16u_C4IR_Ctx(const Npp32u aConstants[4], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return LShiftMulti<Npp16u, 4>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLShiftC_16u_C4IR(const Npp32u aConstants[4], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiLShiftC_16u_C4IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// ============================================================================
// Npp32s - Signed 32-bit
// ============================================================================

// C1R
NppStatus nppiLShiftC_32s_C1R_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32u nConstant, Npp32s *pDst, int nDstStep,
                                  NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return LShift<Npp32s, 1>::execute(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLShiftC_32s_C1R(const Npp32s *pSrc, int nSrcStep, Npp32u nConstant, Npp32s *pDst, int nDstStep,
                              NppiSize oSizeROI) {
  return nppiLShiftC_32s_C1R_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C1IR (in-place)
NppStatus nppiLShiftC_32s_C1IR_Ctx(Npp32u nConstant, Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return LShift<Npp32s, 1>::executeInplace(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLShiftC_32s_C1IR(Npp32u nConstant, Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiLShiftC_32s_C1IR_Ctx(nConstant, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// C3R
NppStatus nppiLShiftC_32s_C3R_Ctx(const Npp32s *pSrc, int nSrcStep, const Npp32u aConstants[3], Npp32s *pDst,
                                  int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return LShiftMulti<Npp32s, 3>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLShiftC_32s_C3R(const Npp32s *pSrc, int nSrcStep, const Npp32u aConstants[3], Npp32s *pDst, int nDstStep,
                              NppiSize oSizeROI) {
  return nppiLShiftC_32s_C3R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C3IR (in-place)
NppStatus nppiLShiftC_32s_C3IR_Ctx(const Npp32u aConstants[3], Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return LShiftMulti<Npp32s, 3>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLShiftC_32s_C3IR(const Npp32u aConstants[3], Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiLShiftC_32s_C3IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// AC4R
NppStatus nppiLShiftC_32s_AC4R_Ctx(const Npp32s *pSrc, int nSrcStep, const Npp32u aConstants[3], Npp32s *pDst,
                                   int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return LShiftMultiAC4<Npp32s>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLShiftC_32s_AC4R(const Npp32s *pSrc, int nSrcStep, const Npp32u aConstants[3], Npp32s *pDst, int nDstStep,
                               NppiSize oSizeROI) {
  return nppiLShiftC_32s_AC4R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// AC4IR (in-place)
NppStatus nppiLShiftC_32s_AC4IR_Ctx(const Npp32u aConstants[3], Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return LShiftMultiAC4<Npp32s>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLShiftC_32s_AC4IR(const Npp32u aConstants[3], Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiLShiftC_32s_AC4IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// C4R
NppStatus nppiLShiftC_32s_C4R_Ctx(const Npp32s *pSrc, int nSrcStep, const Npp32u aConstants[4], Npp32s *pDst,
                                  int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return LShiftMulti<Npp32s, 4>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLShiftC_32s_C4R(const Npp32s *pSrc, int nSrcStep, const Npp32u aConstants[4], Npp32s *pDst, int nDstStep,
                              NppiSize oSizeROI) {
  return nppiLShiftC_32s_C4R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C4IR (in-place)
NppStatus nppiLShiftC_32s_C4IR_Ctx(const Npp32u aConstants[4], Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return LShiftMulti<Npp32s, 4>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLShiftC_32s_C4IR(const Npp32u aConstants[4], Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiLShiftC_32s_C4IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}
