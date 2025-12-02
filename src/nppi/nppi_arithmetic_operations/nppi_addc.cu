// Add constant operation using template-based API
#include "nppi_arithmetic_api.h"

using namespace nppi::arithmetic;

template <typename T, int C> using AddC = ConstOpAPI<T, C, AddConstOp>;
template <typename T, int C> using AddCMulti = MultiConstOpAPI<T, C, AddConstOp>;

// ============================================================================
// 8u variants
// ============================================================================

NppStatus nppiAddC_8u_C1RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u nConstant, Npp8u *pDst, int nDstStep,
                                 NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddC<Npp8u, 1>::execute(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_8u_C1RSfs(const Npp8u *pSrc, int nSrcStep, Npp8u nConstant, Npp8u *pDst, int nDstStep,
                             NppiSize oSizeROI, int nScaleFactor) {
  return nppiAddC_8u_C1RSfs_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor,
                                getDefaultStreamContext());
}

NppStatus nppiAddC_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u nConstant, Npp8u *pDst, int nDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return AddC<Npp8u, 1>::execute(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u nConstant, Npp8u *pDst, int nDstStep,
                          NppiSize oSizeROI) {
  return nppiAddC_8u_C1R_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAddC_8u_C1IRSfs_Ctx(Npp8u nConstant, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx) {
  return AddC<Npp8u, 1>::executeInplace(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_8u_C1IRSfs(Npp8u nConstant, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiAddC_8u_C1IRSfs_Ctx(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiAddC_8u_C1IR_Ctx(Npp8u nConstant, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx) {
  return AddC<Npp8u, 1>::executeInplace(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_8u_C1IR(Npp8u nConstant, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAddC_8u_C1IR_Ctx(nConstant, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAddC_8u_C3RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                                 NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp8u, 3>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_8u_C3RSfs(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                             NppiSize oSizeROI, int nScaleFactor) {
  return nppiAddC_8u_C3RSfs_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                getDefaultStreamContext());
}

NppStatus nppiAddC_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp8u, 3>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_8u_C3R(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                          NppiSize oSizeROI) {
  return nppiAddC_8u_C3R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAddC_8u_C3IRSfs_Ctx(const Npp8u aConstants[3], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                  int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp8u, 3>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_8u_C3IRSfs(const Npp8u aConstants[3], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor) {
  return nppiAddC_8u_C3IRSfs_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiAddC_8u_C3IR_Ctx(const Npp8u aConstants[3], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp8u, 3>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_8u_C3IR(const Npp8u aConstants[3], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAddC_8u_C3IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAddC_8u_AC4RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp8u, 4>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_8u_AC4RSfs(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor) {
  return nppiAddC_8u_AC4RSfs_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                 getDefaultStreamContext());
}

NppStatus nppiAddC_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp8u, 4>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_8u_AC4R(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                           NppiSize oSizeROI) {
  return nppiAddC_8u_AC4R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAddC_8u_AC4IRSfs_Ctx(const Npp8u aConstants[3], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp8u, 4>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_8u_AC4IRSfs(const Npp8u aConstants[3], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor) {
  return nppiAddC_8u_AC4IRSfs_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiAddC_8u_AC4IR_Ctx(const Npp8u aConstants[3], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp8u, 4>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_8u_AC4IR(const Npp8u aConstants[3], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAddC_8u_AC4IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAddC_8u_C4RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[4], Npp8u *pDst, int nDstStep,
                                 NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp8u, 4>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_8u_C4RSfs(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[4], Npp8u *pDst, int nDstStep,
                             NppiSize oSizeROI, int nScaleFactor) {
  return nppiAddC_8u_C4RSfs_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                getDefaultStreamContext());
}

NppStatus nppiAddC_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[4], Npp8u *pDst, int nDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp8u, 4>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_8u_C4R(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[4], Npp8u *pDst, int nDstStep,
                          NppiSize oSizeROI) {
  return nppiAddC_8u_C4R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAddC_8u_C4IRSfs_Ctx(const Npp8u aConstants[4], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                  int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp8u, 4>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_8u_C4IRSfs(const Npp8u aConstants[4], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor) {
  return nppiAddC_8u_C4IRSfs_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiAddC_8u_C4IR_Ctx(const Npp8u aConstants[4], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp8u, 4>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_8u_C4IR(const Npp8u aConstants[4], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAddC_8u_C4IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// ============================================================================
// 16u variants
// ============================================================================

NppStatus nppiAddC_16u_C1RSfs_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u nConstant, Npp16u *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddC<Npp16u, 1>::execute(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_16u_C1RSfs(const Npp16u *pSrc, int nSrcStep, Npp16u nConstant, Npp16u *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor) {
  return nppiAddC_16u_C1RSfs_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor,
                                 getDefaultStreamContext());
}

NppStatus nppiAddC_16u_C1IRSfs_Ctx(Npp16u nConstant, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddC<Npp16u, 1>::executeInplace(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_16u_C1IRSfs(Npp16u nConstant, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor) {
  return nppiAddC_16u_C1IRSfs_Ctx(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiAddC_16u_C3RSfs_Ctx(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[3], Npp16u *pDst,
                                  int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp16u, 3>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                       nppStreamCtx);
}

NppStatus nppiAddC_16u_C3RSfs(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[3], Npp16u *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor) {
  return nppiAddC_16u_C3RSfs_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                 getDefaultStreamContext());
}

NppStatus nppiAddC_16u_C3IRSfs_Ctx(const Npp16u aConstants[3], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp16u, 3>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_16u_C3IRSfs(const Npp16u aConstants[3], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor) {
  return nppiAddC_16u_C3IRSfs_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiAddC_16u_AC4RSfs_Ctx(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[3], Npp16u *pDst,
                                   int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp16u, 4>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                       nppStreamCtx);
}

NppStatus nppiAddC_16u_AC4RSfs(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[3], Npp16u *pDst, int nDstStep,
                               NppiSize oSizeROI, int nScaleFactor) {
  return nppiAddC_16u_AC4RSfs_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                  getDefaultStreamContext());
}

NppStatus nppiAddC_16u_AC4IRSfs_Ctx(const Npp16u aConstants[3], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                    int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp16u, 4>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_16u_AC4IRSfs(const Npp16u aConstants[3], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                int nScaleFactor) {
  return nppiAddC_16u_AC4IRSfs_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiAddC_16u_C4RSfs_Ctx(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[4], Npp16u *pDst,
                                  int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp16u, 4>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                       nppStreamCtx);
}

NppStatus nppiAddC_16u_C4RSfs(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[4], Npp16u *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor) {
  return nppiAddC_16u_C4RSfs_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                 getDefaultStreamContext());
}

NppStatus nppiAddC_16u_C4IRSfs_Ctx(const Npp16u aConstants[4], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp16u, 4>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_16u_C4IRSfs(const Npp16u aConstants[4], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor) {
  return nppiAddC_16u_C4IRSfs_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

// ============================================================================
// 16s variants
// ============================================================================

NppStatus nppiAddC_16s_C1RSfs_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s nConstant, Npp16s *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddC<Npp16s, 1>::execute(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_16s_C1RSfs(const Npp16s *pSrc, int nSrcStep, Npp16s nConstant, Npp16s *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor) {
  return nppiAddC_16s_C1RSfs_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor,
                                 getDefaultStreamContext());
}

NppStatus nppiAddC_16s_C1IRSfs_Ctx(Npp16s nConstant, Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddC<Npp16s, 1>::executeInplace(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_16s_C1IRSfs(Npp16s nConstant, Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor) {
  return nppiAddC_16s_C1IRSfs_Ctx(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiAddC_16s_C3RSfs_Ctx(const Npp16s *pSrc, int nSrcStep, const Npp16s aConstants[3], Npp16s *pDst,
                                  int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp16s, 3>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                       nppStreamCtx);
}

NppStatus nppiAddC_16s_C3RSfs(const Npp16s *pSrc, int nSrcStep, const Npp16s aConstants[3], Npp16s *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor) {
  return nppiAddC_16s_C3RSfs_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                 getDefaultStreamContext());
}

NppStatus nppiAddC_16s_C3IRSfs_Ctx(const Npp16s aConstants[3], Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp16s, 3>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_16s_C3IRSfs(const Npp16s aConstants[3], Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor) {
  return nppiAddC_16s_C3IRSfs_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiAddC_16s_AC4RSfs_Ctx(const Npp16s *pSrc, int nSrcStep, const Npp16s aConstants[3], Npp16s *pDst,
                                   int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp16s, 4>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                       nppStreamCtx);
}

NppStatus nppiAddC_16s_AC4RSfs(const Npp16s *pSrc, int nSrcStep, const Npp16s aConstants[3], Npp16s *pDst, int nDstStep,
                               NppiSize oSizeROI, int nScaleFactor) {
  return nppiAddC_16s_AC4RSfs_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                  getDefaultStreamContext());
}

NppStatus nppiAddC_16s_AC4IRSfs_Ctx(const Npp16s aConstants[3], Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                    int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp16s, 4>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_16s_AC4IRSfs(const Npp16s aConstants[3], Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                int nScaleFactor) {
  return nppiAddC_16s_AC4IRSfs_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiAddC_16s_C4RSfs_Ctx(const Npp16s *pSrc, int nSrcStep, const Npp16s aConstants[4], Npp16s *pDst,
                                  int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp16s, 4>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                       nppStreamCtx);
}

NppStatus nppiAddC_16s_C4RSfs(const Npp16s *pSrc, int nSrcStep, const Npp16s aConstants[4], Npp16s *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor) {
  return nppiAddC_16s_C4RSfs_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                 getDefaultStreamContext());
}

NppStatus nppiAddC_16s_C4IRSfs_Ctx(const Npp16s aConstants[4], Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp16s, 4>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_16s_C4IRSfs(const Npp16s aConstants[4], Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor) {
  return nppiAddC_16s_C4IRSfs_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

// ============================================================================
// 32f variants (no scale factor)
// ============================================================================

NppStatus nppiAddC_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f nConstant, Npp32f *pDst, int nDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return AddC<Npp32f, 1>::execute(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f nConstant, Npp32f *pDst, int nDstStep,
                           NppiSize oSizeROI) {
  return nppiAddC_32f_C1R_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAddC_32f_C1IR_Ctx(Npp32f nConstant, Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx) {
  return AddC<Npp32f, 1>::executeInplace(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_32f_C1IR(Npp32f nConstant, Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAddC_32f_C1IR_Ctx(nConstant, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAddC_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep, const Npp32f aConstants[3], Npp32f *pDst, int nDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp32f, 3>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_32f_C3R(const Npp32f *pSrc, int nSrcStep, const Npp32f aConstants[3], Npp32f *pDst, int nDstStep,
                           NppiSize oSizeROI) {
  return nppiAddC_32f_C3R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAddC_32f_C3IR_Ctx(const Npp32f aConstants[3], Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp32f, 3>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_32f_C3IR(const Npp32f aConstants[3], Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAddC_32f_C3IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAddC_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep, const Npp32f aConstants[3], Npp32f *pDst,
                                int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp32f, 4>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_32f_AC4R(const Npp32f *pSrc, int nSrcStep, const Npp32f aConstants[3], Npp32f *pDst, int nDstStep,
                            NppiSize oSizeROI) {
  return nppiAddC_32f_AC4R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAddC_32f_AC4IR_Ctx(const Npp32f aConstants[3], Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp32f, 4>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_32f_AC4IR(const Npp32f aConstants[3], Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAddC_32f_AC4IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAddC_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep, const Npp32f aConstants[4], Npp32f *pDst, int nDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp32f, 4>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_32f_C4R(const Npp32f *pSrc, int nSrcStep, const Npp32f aConstants[4], Npp32f *pDst, int nDstStep,
                           NppiSize oSizeROI) {
  return nppiAddC_32f_C4R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAddC_32f_C4IR_Ctx(const Npp32f aConstants[4], Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp32f, 4>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_32f_C4IR(const Npp32f aConstants[4], Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAddC_32f_C4IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// ============================================================================
// 32s variants (with scale factor)
// ============================================================================

NppStatus nppiAddC_32s_C1RSfs_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s nConstant, Npp32s *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddC<Npp32s, 1>::execute(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_32s_C1RSfs(const Npp32s *pSrc, int nSrcStep, Npp32s nConstant, Npp32s *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor) {
  return nppiAddC_32s_C1RSfs_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor,
                                 getDefaultStreamContext());
}

NppStatus nppiAddC_32s_C1IRSfs_Ctx(Npp32s nConstant, Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddC<Npp32s, 1>::executeInplace(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_32s_C1IRSfs(Npp32s nConstant, Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiAddC_32s_C1IRSfs_Ctx(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiAddC_32s_C3RSfs_Ctx(const Npp32s *pSrc, int nSrcStep, const Npp32s aConstants[3], Npp32s *pDst,
                                  int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp32s, 3>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                       nppStreamCtx);
}

NppStatus nppiAddC_32s_C3RSfs(const Npp32s *pSrc, int nSrcStep, const Npp32s aConstants[3], Npp32s *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor) {
  return nppiAddC_32s_C3RSfs_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                 getDefaultStreamContext());
}

NppStatus nppiAddC_32s_C3IRSfs_Ctx(const Npp32s aConstants[3], Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp32s, 3>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_32s_C3IRSfs(const Npp32s aConstants[3], Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor) {
  return nppiAddC_32s_C3IRSfs_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

// ============================================================================
// 16sc variants (16-bit signed complex)
// ============================================================================

NppStatus nppiAddC_16sc_C1RSfs_Ctx(const Npp16sc *pSrc, int nSrcStep, Npp16sc nConstant, Npp16sc *pDst, int nDstStep,
                                   NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddC<Npp16sc, 1>::execute(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_16sc_C1RSfs(const Npp16sc *pSrc, int nSrcStep, Npp16sc nConstant, Npp16sc *pDst, int nDstStep,
                               NppiSize oSizeROI, int nScaleFactor) {
  return nppiAddC_16sc_C1RSfs_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor,
                                  getDefaultStreamContext());
}

NppStatus nppiAddC_16sc_C1IRSfs_Ctx(Npp16sc nConstant, Npp16sc *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                    int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddC<Npp16sc, 1>::executeInplace(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_16sc_C1IRSfs(Npp16sc nConstant, Npp16sc *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                int nScaleFactor) {
  return nppiAddC_16sc_C1IRSfs_Ctx(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiAddC_16sc_C3RSfs_Ctx(const Npp16sc *pSrc, int nSrcStep, const Npp16sc aConstants[3], Npp16sc *pDst,
                                   int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp16sc, 3>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                        nppStreamCtx);
}

NppStatus nppiAddC_16sc_C3RSfs(const Npp16sc *pSrc, int nSrcStep, const Npp16sc aConstants[3], Npp16sc *pDst,
                               int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiAddC_16sc_C3RSfs_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                  getDefaultStreamContext());
}

NppStatus nppiAddC_16sc_C3IRSfs_Ctx(const Npp16sc aConstants[3], Npp16sc *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                    int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp16sc, 3>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_16sc_C3IRSfs(const Npp16sc aConstants[3], Npp16sc *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                int nScaleFactor) {
  return nppiAddC_16sc_C3IRSfs_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiAddC_16sc_AC4RSfs_Ctx(const Npp16sc *pSrc, int nSrcStep, const Npp16sc aConstants[3], Npp16sc *pDst,
                                    int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp16sc, 4>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                        nppStreamCtx);
}

NppStatus nppiAddC_16sc_AC4RSfs(const Npp16sc *pSrc, int nSrcStep, const Npp16sc aConstants[3], Npp16sc *pDst,
                                int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiAddC_16sc_AC4RSfs_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                   getDefaultStreamContext());
}

NppStatus nppiAddC_16sc_AC4IRSfs_Ctx(const Npp16sc aConstants[3], Npp16sc *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                     int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp16sc, 4>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_16sc_AC4IRSfs(const Npp16sc aConstants[3], Npp16sc *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                 int nScaleFactor) {
  return nppiAddC_16sc_AC4IRSfs_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor,
                                    getDefaultStreamContext());
}

// ============================================================================
// 32sc variants (32-bit signed complex)
// ============================================================================

NppStatus nppiAddC_32sc_C1RSfs_Ctx(const Npp32sc *pSrc, int nSrcStep, Npp32sc nConstant, Npp32sc *pDst, int nDstStep,
                                   NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddC<Npp32sc, 1>::execute(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_32sc_C1RSfs(const Npp32sc *pSrc, int nSrcStep, Npp32sc nConstant, Npp32sc *pDst, int nDstStep,
                               NppiSize oSizeROI, int nScaleFactor) {
  return nppiAddC_32sc_C1RSfs_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor,
                                  getDefaultStreamContext());
}

NppStatus nppiAddC_32sc_C1IRSfs_Ctx(Npp32sc nConstant, Npp32sc *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                    int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddC<Npp32sc, 1>::executeInplace(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_32sc_C1IRSfs(Npp32sc nConstant, Npp32sc *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                int nScaleFactor) {
  return nppiAddC_32sc_C1IRSfs_Ctx(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiAddC_32sc_C3RSfs_Ctx(const Npp32sc *pSrc, int nSrcStep, const Npp32sc aConstants[3], Npp32sc *pDst,
                                   int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp32sc, 3>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                        nppStreamCtx);
}

NppStatus nppiAddC_32sc_C3RSfs(const Npp32sc *pSrc, int nSrcStep, const Npp32sc aConstants[3], Npp32sc *pDst,
                               int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiAddC_32sc_C3RSfs_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                  getDefaultStreamContext());
}

NppStatus nppiAddC_32sc_C3IRSfs_Ctx(const Npp32sc aConstants[3], Npp32sc *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                    int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp32sc, 3>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_32sc_C3IRSfs(const Npp32sc aConstants[3], Npp32sc *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                int nScaleFactor) {
  return nppiAddC_32sc_C3IRSfs_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiAddC_32sc_AC4RSfs_Ctx(const Npp32sc *pSrc, int nSrcStep, const Npp32sc aConstants[3], Npp32sc *pDst,
                                    int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp32sc, 4>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                        nppStreamCtx);
}

NppStatus nppiAddC_32sc_AC4RSfs(const Npp32sc *pSrc, int nSrcStep, const Npp32sc aConstants[3], Npp32sc *pDst,
                                int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiAddC_32sc_AC4RSfs_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                   getDefaultStreamContext());
}

NppStatus nppiAddC_32sc_AC4IRSfs_Ctx(const Npp32sc aConstants[3], Npp32sc *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                     int nScaleFactor, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp32sc, 4>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_32sc_AC4IRSfs(const Npp32sc aConstants[3], Npp32sc *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                 int nScaleFactor) {
  return nppiAddC_32sc_AC4IRSfs_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor,
                                    getDefaultStreamContext());
}

// ============================================================================
// 32fc variants (32-bit float complex, no scale factor)
// ============================================================================

NppStatus nppiAddC_32fc_C1R_Ctx(const Npp32fc *pSrc, int nSrcStep, Npp32fc nConstant, Npp32fc *pDst, int nDstStep,
                                NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return AddC<Npp32fc, 1>::execute(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_32fc_C1R(const Npp32fc *pSrc, int nSrcStep, Npp32fc nConstant, Npp32fc *pDst, int nDstStep,
                            NppiSize oSizeROI) {
  return nppiAddC_32fc_C1R_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAddC_32fc_C1IR_Ctx(Npp32fc nConstant, Npp32fc *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx) {
  return AddC<Npp32fc, 1>::executeInplace(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_32fc_C1IR(Npp32fc nConstant, Npp32fc *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAddC_32fc_C1IR_Ctx(nConstant, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAddC_32fc_C3R_Ctx(const Npp32fc *pSrc, int nSrcStep, const Npp32fc aConstants[3], Npp32fc *pDst,
                                int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp32fc, 3>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_32fc_C3R(const Npp32fc *pSrc, int nSrcStep, const Npp32fc aConstants[3], Npp32fc *pDst, int nDstStep,
                            NppiSize oSizeROI) {
  return nppiAddC_32fc_C3R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAddC_32fc_C3IR_Ctx(const Npp32fc aConstants[3], Npp32fc *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp32fc, 3>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_32fc_C3IR(const Npp32fc aConstants[3], Npp32fc *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAddC_32fc_C3IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAddC_32fc_AC4R_Ctx(const Npp32fc *pSrc, int nSrcStep, const Npp32fc aConstants[3], Npp32fc *pDst,
                                 int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp32fc, 4>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_32fc_AC4R(const Npp32fc *pSrc, int nSrcStep, const Npp32fc aConstants[3], Npp32fc *pDst, int nDstStep,
                             NppiSize oSizeROI) {
  return nppiAddC_32fc_AC4R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAddC_32fc_AC4IR_Ctx(const Npp32fc aConstants[3], Npp32fc *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp32fc, 4>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_32fc_AC4IR(const Npp32fc aConstants[3], Npp32fc *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAddC_32fc_AC4IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAddC_32fc_C4R_Ctx(const Npp32fc *pSrc, int nSrcStep, const Npp32fc aConstants[4], Npp32fc *pDst,
                                int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp32fc, 4>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_32fc_C4R(const Npp32fc *pSrc, int nSrcStep, const Npp32fc aConstants[4], Npp32fc *pDst, int nDstStep,
                            NppiSize oSizeROI) {
  return nppiAddC_32fc_C4R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAddC_32fc_C4IR_Ctx(const Npp32fc aConstants[4], Npp32fc *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx) {
  return AddCMulti<Npp32fc, 4>::executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_32fc_C4IR(const Npp32fc aConstants[4], Npp32fc *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAddC_32fc_C4IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// ============================================================================
// 16f variants
// Note: 16f constant operations use Npp32f as constant type
// ============================================================================

NppStatus nppiAddC_16f_C1R_Ctx(const Npp16f *pSrc, int nSrcStep, Npp32f nConstant, Npp16f *pDst, int nDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  AddConst16fOp op(nConstant);
  return ConstOperationExecutor<Npp16f, 1, AddConst16fOp>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                                    nppStreamCtx.hStream, op);
}

NppStatus nppiAddC_16f_C1R(const Npp16f *pSrc, int nSrcStep, Npp32f nConstant, Npp16f *pDst, int nDstStep,
                           NppiSize oSizeROI) {
  return nppiAddC_16f_C1R_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAddC_16f_C1IR_Ctx(Npp32f nConstant, Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx) {
  return nppiAddC_16f_C1R_Ctx(pSrcDst, nSrcDstStep, nConstant, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_16f_C1IR(Npp32f nConstant, Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAddC_16f_C1IR_Ctx(nConstant, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAddC_16f_C3R_Ctx(const Npp16f *pSrc, int nSrcStep, const Npp32f aConstants[3], Npp16f *pDst, int nDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MultiConst16fOperationExecutor<3, AddConst16fOp>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                                    nppStreamCtx.hStream, aConstants);
}

NppStatus nppiAddC_16f_C3R(const Npp16f *pSrc, int nSrcStep, const Npp32f aConstants[3], Npp16f *pDst, int nDstStep,
                           NppiSize oSizeROI) {
  return nppiAddC_16f_C3R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAddC_16f_C3IR_Ctx(const Npp32f aConstants[3], Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx) {
  return nppiAddC_16f_C3R_Ctx(pSrcDst, nSrcDstStep, aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_16f_C3IR(const Npp32f aConstants[3], Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAddC_16f_C3IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAddC_16f_C4R_Ctx(const Npp16f *pSrc, int nSrcStep, const Npp32f aConstants[4], Npp16f *pDst, int nDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MultiConst16fOperationExecutor<4, AddConst16fOp>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                                    nppStreamCtx.hStream, aConstants);
}

NppStatus nppiAddC_16f_C4R(const Npp16f *pSrc, int nSrcStep, const Npp32f aConstants[4], Npp16f *pDst, int nDstStep,
                           NppiSize oSizeROI) {
  return nppiAddC_16f_C4R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAddC_16f_C4IR_Ctx(const Npp32f aConstants[4], Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx) {
  return nppiAddC_16f_C4R_Ctx(pSrcDst, nSrcDstStep, aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_16f_C4IR(const Npp32f aConstants[4], Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAddC_16f_C4IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}
