// Multiply by constant operation using template-based API
#include "nppi_arithmetic_api.h"

using namespace nppi::arithmetic;

template <typename T, int C> using MulC = ConstOpAPI<T, C, MulConstOp>;
template <typename T, int C> using MulCMulti = MultiConstOpAPI<T, C, MulConstOp>;

// ============================================================================
// 8u variants
// ============================================================================

NppStatus nppiMulC_8u_C1RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u nConstant, Npp8u *pDst, int nDstStep,
                                 NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return MulC<Npp8u, 1>::execute(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiMulC_8u_C1RSfs(const Npp8u *pSrc, int nSrcStep, Npp8u nConstant, Npp8u *pDst, int nDstStep,
                             NppiSize oSizeROI, int nScaleFactor) {
  return nppiMulC_8u_C1RSfs_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor,
                                getDefaultStreamContext());
}

NppStatus nppiMulC_8u_C1IRSfs_Ctx(Npp8u nConstant, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx) {
  return MulC<Npp8u, 1>::executeInplace(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiMulC_8u_C1IRSfs(Npp8u nConstant, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiMulC_8u_C1IRSfs_Ctx(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiMulC_8u_C3RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                                 NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return MulCMulti<Npp8u, 3>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiMulC_8u_C3RSfs(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                             NppiSize oSizeROI, int nScaleFactor) {
  return nppiMulC_8u_C3RSfs_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                getDefaultStreamContext());
}

NppStatus nppiMulC_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MulCMulti<Npp8u, 3>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulC_8u_C3R(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                          NppiSize oSizeROI) {
  return nppiMulC_8u_C3R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiMulC_8u_C3IRSfs_Ctx(const Npp8u aConstants[3], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                  int nScaleFactor, NppStreamContext nppStreamCtx) {
  return MulCMulti<Npp8u, 3>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiMulC_8u_C3IRSfs(const Npp8u aConstants[3], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor) {
  return nppiMulC_8u_C3IRSfs_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiMulC_8u_AC4RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return MulCMulti<Npp8u, 4>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiMulC_8u_AC4RSfs(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor) {
  return nppiMulC_8u_AC4RSfs_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                 getDefaultStreamContext());
}

NppStatus nppiMulC_8u_AC4IRSfs_Ctx(const Npp8u aConstants[3], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor, NppStreamContext nppStreamCtx) {
  return MulCMulti<Npp8u, 4>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiMulC_8u_AC4IRSfs(const Npp8u aConstants[3], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor) {
  return nppiMulC_8u_AC4IRSfs_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiMulC_8u_C4RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[4], Npp8u *pDst, int nDstStep,
                                 NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return MulCMulti<Npp8u, 4>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiMulC_8u_C4RSfs(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[4], Npp8u *pDst, int nDstStep,
                             NppiSize oSizeROI, int nScaleFactor) {
  return nppiMulC_8u_C4RSfs_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                getDefaultStreamContext());
}

NppStatus nppiMulC_8u_C4IRSfs_Ctx(const Npp8u aConstants[4], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                  int nScaleFactor, NppStreamContext nppStreamCtx) {
  return MulCMulti<Npp8u, 4>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiMulC_8u_C4IRSfs(const Npp8u aConstants[4], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor) {
  return nppiMulC_8u_C4IRSfs_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

// ============================================================================
// 16u variants
// ============================================================================

NppStatus nppiMulC_16u_C1RSfs_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u nConstant, Npp16u *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return MulC<Npp16u, 1>::execute(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiMulC_16u_C1RSfs(const Npp16u *pSrc, int nSrcStep, Npp16u nConstant, Npp16u *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor) {
  return nppiMulC_16u_C1RSfs_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor,
                                 getDefaultStreamContext());
}

NppStatus nppiMulC_16u_C1IRSfs_Ctx(Npp16u nConstant, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor, NppStreamContext nppStreamCtx) {
  return MulC<Npp16u, 1>::executeInplace(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiMulC_16u_C1IRSfs(Npp16u nConstant, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor) {
  return nppiMulC_16u_C1IRSfs_Ctx(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiMulC_16u_C3RSfs_Ctx(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[3], Npp16u *pDst,
                                  int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return MulCMulti<Npp16u, 3>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                       nppStreamCtx);
}

NppStatus nppiMulC_16u_C3RSfs(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[3], Npp16u *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor) {
  return nppiMulC_16u_C3RSfs_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                 getDefaultStreamContext());
}

NppStatus nppiMulC_16u_C3IRSfs_Ctx(const Npp16u aConstants[3], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor, NppStreamContext nppStreamCtx) {
  return MulCMulti<Npp16u, 3>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiMulC_16u_C3IRSfs(const Npp16u aConstants[3], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor) {
  return nppiMulC_16u_C3IRSfs_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiMulC_16u_AC4RSfs_Ctx(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[3], Npp16u *pDst,
                                   int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return MulCMulti<Npp16u, 4>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                       nppStreamCtx);
}

NppStatus nppiMulC_16u_AC4RSfs(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[3], Npp16u *pDst, int nDstStep,
                               NppiSize oSizeROI, int nScaleFactor) {
  return nppiMulC_16u_AC4RSfs_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                  getDefaultStreamContext());
}

NppStatus nppiMulC_16u_AC4IRSfs_Ctx(const Npp16u aConstants[3], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                    int nScaleFactor, NppStreamContext nppStreamCtx) {
  return MulCMulti<Npp16u, 4>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiMulC_16u_AC4IRSfs(const Npp16u aConstants[3], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                int nScaleFactor) {
  return nppiMulC_16u_AC4IRSfs_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiMulC_16u_C4RSfs_Ctx(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[4], Npp16u *pDst,
                                  int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return MulCMulti<Npp16u, 4>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                       nppStreamCtx);
}

NppStatus nppiMulC_16u_C4RSfs(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[4], Npp16u *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor) {
  return nppiMulC_16u_C4RSfs_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                 getDefaultStreamContext());
}

NppStatus nppiMulC_16u_C4IRSfs_Ctx(const Npp16u aConstants[4], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor, NppStreamContext nppStreamCtx) {
  return MulCMulti<Npp16u, 4>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiMulC_16u_C4IRSfs(const Npp16u aConstants[4], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor) {
  return nppiMulC_16u_C4IRSfs_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

// ============================================================================
// 16s variants
// ============================================================================

NppStatus nppiMulC_16s_C1RSfs_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s nConstant, Npp16s *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return MulC<Npp16s, 1>::execute(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiMulC_16s_C1RSfs(const Npp16s *pSrc, int nSrcStep, Npp16s nConstant, Npp16s *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor) {
  return nppiMulC_16s_C1RSfs_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor,
                                 getDefaultStreamContext());
}

NppStatus nppiMulC_16s_C1IRSfs_Ctx(Npp16s nConstant, Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor, NppStreamContext nppStreamCtx) {
  return MulC<Npp16s, 1>::executeInplace(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiMulC_16s_C1IRSfs(Npp16s nConstant, Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor) {
  return nppiMulC_16s_C1IRSfs_Ctx(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiMulC_16s_C3RSfs_Ctx(const Npp16s *pSrc, int nSrcStep, const Npp16s aConstants[3], Npp16s *pDst,
                                  int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return MulCMulti<Npp16s, 3>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                       nppStreamCtx);
}

NppStatus nppiMulC_16s_C3RSfs(const Npp16s *pSrc, int nSrcStep, const Npp16s aConstants[3], Npp16s *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor) {
  return nppiMulC_16s_C3RSfs_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                 getDefaultStreamContext());
}

NppStatus nppiMulC_16s_C3IRSfs_Ctx(const Npp16s aConstants[3], Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor, NppStreamContext nppStreamCtx) {
  return MulCMulti<Npp16s, 3>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiMulC_16s_C3IRSfs(const Npp16s aConstants[3], Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor) {
  return nppiMulC_16s_C3IRSfs_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiMulC_16s_AC4RSfs_Ctx(const Npp16s *pSrc, int nSrcStep, const Npp16s aConstants[3], Npp16s *pDst,
                                   int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return MulCMulti<Npp16s, 4>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                       nppStreamCtx);
}

NppStatus nppiMulC_16s_AC4RSfs(const Npp16s *pSrc, int nSrcStep, const Npp16s aConstants[3], Npp16s *pDst, int nDstStep,
                               NppiSize oSizeROI, int nScaleFactor) {
  return nppiMulC_16s_AC4RSfs_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                  getDefaultStreamContext());
}

NppStatus nppiMulC_16s_AC4IRSfs_Ctx(const Npp16s aConstants[3], Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                    int nScaleFactor, NppStreamContext nppStreamCtx) {
  return MulCMulti<Npp16s, 4>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiMulC_16s_AC4IRSfs(const Npp16s aConstants[3], Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                int nScaleFactor) {
  return nppiMulC_16s_AC4IRSfs_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiMulC_16s_C4RSfs_Ctx(const Npp16s *pSrc, int nSrcStep, const Npp16s aConstants[4], Npp16s *pDst,
                                  int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return MulCMulti<Npp16s, 4>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                       nppStreamCtx);
}

NppStatus nppiMulC_16s_C4RSfs(const Npp16s *pSrc, int nSrcStep, const Npp16s aConstants[4], Npp16s *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor) {
  return nppiMulC_16s_C4RSfs_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                 getDefaultStreamContext());
}

NppStatus nppiMulC_16s_C4IRSfs_Ctx(const Npp16s aConstants[4], Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor, NppStreamContext nppStreamCtx) {
  return MulCMulti<Npp16s, 4>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiMulC_16s_C4IRSfs(const Npp16s aConstants[4], Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor) {
  return nppiMulC_16s_C4IRSfs_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

// ============================================================================
// 32f variants (no scale factor)
// ============================================================================

NppStatus nppiMulC_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f nConstant, Npp32f *pDst, int nDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MulC<Npp32f, 1>::execute(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulC_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f nConstant, Npp32f *pDst, int nDstStep,
                           NppiSize oSizeROI) {
  return nppiMulC_32f_C1R_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiMulC_32f_C1IR_Ctx(Npp32f nConstant, Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx) {
  return MulC<Npp32f, 1>::executeInplace(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulC_32f_C1IR(Npp32f nConstant, Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiMulC_32f_C1IR_Ctx(nConstant, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiMulC_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep, const Npp32f aConstants[3], Npp32f *pDst, int nDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MulCMulti<Npp32f, 3>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulC_32f_C3R(const Npp32f *pSrc, int nSrcStep, const Npp32f aConstants[3], Npp32f *pDst, int nDstStep,
                           NppiSize oSizeROI) {
  return nppiMulC_32f_C3R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiMulC_32f_C3IR_Ctx(const Npp32f aConstants[3], Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx) {
  return MulCMulti<Npp32f, 3>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulC_32f_C3IR(const Npp32f aConstants[3], Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiMulC_32f_C3IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiMulC_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep, const Npp32f aConstants[3], Npp32f *pDst,
                                int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MulCMulti<Npp32f, 4>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulC_32f_AC4R(const Npp32f *pSrc, int nSrcStep, const Npp32f aConstants[3], Npp32f *pDst, int nDstStep,
                            NppiSize oSizeROI) {
  return nppiMulC_32f_AC4R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiMulC_32f_AC4IR_Ctx(const Npp32f aConstants[3], Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx) {
  return MulCMulti<Npp32f, 4>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulC_32f_AC4IR(const Npp32f aConstants[3], Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiMulC_32f_AC4IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiMulC_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep, const Npp32f aConstants[4], Npp32f *pDst, int nDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MulCMulti<Npp32f, 4>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulC_32f_C4R(const Npp32f *pSrc, int nSrcStep, const Npp32f aConstants[4], Npp32f *pDst, int nDstStep,
                           NppiSize oSizeROI) {
  return nppiMulC_32f_C4R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiMulC_32f_C4IR_Ctx(const Npp32f aConstants[4], Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx) {
  return MulCMulti<Npp32f, 4>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulC_32f_C4IR(const Npp32f aConstants[4], Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiMulC_32f_C4IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// ============================================================================
// 32s variants (with scale factor)
// ============================================================================

NppStatus nppiMulC_32s_C1RSfs_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s nConstant, Npp32s *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return MulC<Npp32s, 1>::execute(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiMulC_32s_C1RSfs(const Npp32s *pSrc, int nSrcStep, Npp32s nConstant, Npp32s *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor) {
  return nppiMulC_32s_C1RSfs_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor,
                                 getDefaultStreamContext());
}

NppStatus nppiMulC_32s_C1IRSfs_Ctx(Npp32s nConstant, Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor, NppStreamContext nppStreamCtx) {
  return MulC<Npp32s, 1>::executeInplace(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiMulC_32s_C1IRSfs(Npp32s nConstant, Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor) {
  return nppiMulC_32s_C1IRSfs_Ctx(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiMulC_32s_C3RSfs_Ctx(const Npp32s *pSrc, int nSrcStep, const Npp32s aConstants[3], Npp32s *pDst,
                                  int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return MulCMulti<Npp32s, 3>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                       nppStreamCtx);
}

NppStatus nppiMulC_32s_C3RSfs(const Npp32s *pSrc, int nSrcStep, const Npp32s aConstants[3], Npp32s *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor) {
  return nppiMulC_32s_C3RSfs_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                 getDefaultStreamContext());
}

NppStatus nppiMulC_32s_C3IRSfs_Ctx(const Npp32s aConstants[3], Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor, NppStreamContext nppStreamCtx) {
  return MulCMulti<Npp32s, 3>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiMulC_32s_C3IRSfs(const Npp32s aConstants[3], Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor) {
  return nppiMulC_32s_C3IRSfs_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}
