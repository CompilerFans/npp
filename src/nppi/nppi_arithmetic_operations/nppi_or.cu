// ============================================================================
// NPP Or Operations - Template-based Implementation
// ============================================================================

#include "nppi_arithmetic_api.h"

using namespace nppi::arithmetic;

// Type alias for cleaner code
template <typename T, int C> using Or = LogicalOpAPI<T, C, OrOp>;
template <typename T> using OrAC4 = LogicalOpAC4API<T, OrOp>;

// ============================================================================
// 8-bit unsigned (8u)
// ============================================================================

// C1
NppStatus nppiOr_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Or<Npp8u, 1>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiOr_8u_C1R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst, int nDstStep,
                        NppiSize oSizeROI) {
  return nppiOr_8u_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiOr_8u_C1IR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx) {
  return Or<Npp8u, 1>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiOr_8u_C1IR(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiOr_8u_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// C3
NppStatus nppiOr_8u_C3R_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Or<Npp8u, 3>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiOr_8u_C3R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst, int nDstStep,
                        NppiSize oSizeROI) {
  return nppiOr_8u_C3R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiOr_8u_C3IR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx) {
  return Or<Npp8u, 3>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiOr_8u_C3IR(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiOr_8u_C3IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// C4
NppStatus nppiOr_8u_C4R_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Or<Npp8u, 4>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiOr_8u_C4R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst, int nDstStep,
                        NppiSize oSizeROI) {
  return nppiOr_8u_C4R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiOr_8u_C4IR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx) {
  return Or<Npp8u, 4>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiOr_8u_C4IR(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiOr_8u_C4IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// AC4
NppStatus nppiOr_8u_AC4R_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return OrAC4<Npp8u>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiOr_8u_AC4R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                         int nDstStep, NppiSize oSizeROI) {
  return nppiOr_8u_AC4R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiOr_8u_AC4IR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx) {
  return OrAC4<Npp8u>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiOr_8u_AC4IR(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiOr_8u_AC4IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// ============================================================================
// 16-bit unsigned (16u)
// ============================================================================

// C1
NppStatus nppiOr_16u_C1R_Ctx(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                             int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Or<Npp16u, 1>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiOr_16u_C1R(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                         int nDstStep, NppiSize oSizeROI) {
  return nppiOr_16u_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiOr_16u_C1IR_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx) {
  return Or<Npp16u, 1>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiOr_16u_C1IR(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiOr_16u_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// C3
NppStatus nppiOr_16u_C3R_Ctx(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                             int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Or<Npp16u, 3>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiOr_16u_C3R(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                         int nDstStep, NppiSize oSizeROI) {
  return nppiOr_16u_C3R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiOr_16u_C3IR_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx) {
  return Or<Npp16u, 3>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiOr_16u_C3IR(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiOr_16u_C3IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// C4
NppStatus nppiOr_16u_C4R_Ctx(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                             int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Or<Npp16u, 4>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiOr_16u_C4R(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                         int nDstStep, NppiSize oSizeROI) {
  return nppiOr_16u_C4R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiOr_16u_C4IR_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx) {
  return Or<Npp16u, 4>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiOr_16u_C4IR(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiOr_16u_C4IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// AC4
NppStatus nppiOr_16u_AC4R_Ctx(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return OrAC4<Npp16u>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiOr_16u_AC4R(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                          int nDstStep, NppiSize oSizeROI) {
  return nppiOr_16u_AC4R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiOr_16u_AC4IR_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx) {
  return OrAC4<Npp16u>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiOr_16u_AC4IR(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiOr_16u_AC4IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// ============================================================================
// 32-bit signed (32s)
// ============================================================================

// C1
NppStatus nppiOr_32s_C1R_Ctx(const Npp32s *pSrc1, int nSrc1Step, const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                             int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Or<Npp32s, 1>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiOr_32s_C1R(const Npp32s *pSrc1, int nSrc1Step, const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                         int nDstStep, NppiSize oSizeROI) {
  return nppiOr_32s_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiOr_32s_C1IR_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx) {
  return Or<Npp32s, 1>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiOr_32s_C1IR(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiOr_32s_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// C3
NppStatus nppiOr_32s_C3R_Ctx(const Npp32s *pSrc1, int nSrc1Step, const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                             int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Or<Npp32s, 3>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiOr_32s_C3R(const Npp32s *pSrc1, int nSrc1Step, const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                         int nDstStep, NppiSize oSizeROI) {
  return nppiOr_32s_C3R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiOr_32s_C3IR_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx) {
  return Or<Npp32s, 3>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiOr_32s_C3IR(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiOr_32s_C3IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// C4
NppStatus nppiOr_32s_C4R_Ctx(const Npp32s *pSrc1, int nSrc1Step, const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                             int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Or<Npp32s, 4>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiOr_32s_C4R(const Npp32s *pSrc1, int nSrc1Step, const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                         int nDstStep, NppiSize oSizeROI) {
  return nppiOr_32s_C4R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiOr_32s_C4IR_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx) {
  return Or<Npp32s, 4>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiOr_32s_C4IR(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiOr_32s_C4IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// AC4
NppStatus nppiOr_32s_AC4R_Ctx(const Npp32s *pSrc1, int nSrc1Step, const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                              int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return OrAC4<Npp32s>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiOr_32s_AC4R(const Npp32s *pSrc1, int nSrc1Step, const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                          int nDstStep, NppiSize oSizeROI) {
  return nppiOr_32s_AC4R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiOr_32s_AC4IR_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx) {
  return OrAC4<Npp32s>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiOr_32s_AC4IR(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiOr_32s_AC4IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}
