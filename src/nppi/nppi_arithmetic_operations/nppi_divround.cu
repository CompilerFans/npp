// ============================================================================
// NPP Div_Round Operations - Template-based Implementation
// ============================================================================
// Division with selectable rounding mode (NPP_RND_NEAR, NPP_RND_FINANCIAL, NPP_RND_ZERO)
// ============================================================================

#include "nppi_arithmetic_api.h"

using namespace nppi::arithmetic;

// ============================================================================
// Type aliases for cleaner code
// ============================================================================
template <typename T, int C> using DivRound = DivRoundOpAPI<T, C>;

// ============================================================================
// 8-bit unsigned (8u) - with scale factor and rounding mode
// ============================================================================

// C1R
NppStatus nppiDiv_Round_8u_C1RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                      int nDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor,
                                      NppStreamContext nppStreamCtx) {
  return DivRound<Npp8u, 1>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, rndMode, nScaleFactor,
                                     nppStreamCtx);
}

NppStatus nppiDiv_Round_8u_C1RSfs(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor) {
  return nppiDiv_Round_8u_C1RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, rndMode, nScaleFactor,
                                     getDefaultStreamContext());
}

// C1IR (in-place)
NppStatus nppiDiv_Round_8u_C1IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor,
                                       NppStreamContext nppStreamCtx) {
  return DivRound<Npp8u, 1>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, rndMode, nScaleFactor,
                                            nppStreamCtx);
}

NppStatus nppiDiv_Round_8u_C1IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   NppRoundMode rndMode, int nScaleFactor) {
  return nppiDiv_Round_8u_C1IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, rndMode, nScaleFactor,
                                      getDefaultStreamContext());
}

// C3R
NppStatus nppiDiv_Round_8u_C3RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                      int nDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor,
                                      NppStreamContext nppStreamCtx) {
  return DivRound<Npp8u, 3>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, rndMode, nScaleFactor,
                                     nppStreamCtx);
}

NppStatus nppiDiv_Round_8u_C3RSfs(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor) {
  return nppiDiv_Round_8u_C3RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, rndMode, nScaleFactor,
                                     getDefaultStreamContext());
}

// C3IR (in-place)
NppStatus nppiDiv_Round_8u_C3IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor,
                                       NppStreamContext nppStreamCtx) {
  return DivRound<Npp8u, 3>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, rndMode, nScaleFactor,
                                            nppStreamCtx);
}

NppStatus nppiDiv_Round_8u_C3IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   NppRoundMode rndMode, int nScaleFactor) {
  return nppiDiv_Round_8u_C3IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, rndMode, nScaleFactor,
                                      getDefaultStreamContext());
}

// AC4R
NppStatus nppiDiv_Round_8u_AC4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor,
                                       NppStreamContext nppStreamCtx) {
  return DivRound<Npp8u, 4>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, rndMode, nScaleFactor,
                                     nppStreamCtx);
}

NppStatus nppiDiv_Round_8u_AC4RSfs(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor) {
  return nppiDiv_Round_8u_AC4RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, rndMode,
                                      nScaleFactor, getDefaultStreamContext());
}

// AC4IR (in-place)
NppStatus nppiDiv_Round_8u_AC4IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor,
                                        NppStreamContext nppStreamCtx) {
  return DivRound<Npp8u, 4>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, rndMode, nScaleFactor,
                                            nppStreamCtx);
}

NppStatus nppiDiv_Round_8u_AC4IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                    NppRoundMode rndMode, int nScaleFactor) {
  return nppiDiv_Round_8u_AC4IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, rndMode, nScaleFactor,
                                       getDefaultStreamContext());
}

// C4R
NppStatus nppiDiv_Round_8u_C4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                      int nDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor,
                                      NppStreamContext nppStreamCtx) {
  return DivRound<Npp8u, 4>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, rndMode, nScaleFactor,
                                     nppStreamCtx);
}

NppStatus nppiDiv_Round_8u_C4RSfs(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor) {
  return nppiDiv_Round_8u_C4RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, rndMode, nScaleFactor,
                                     getDefaultStreamContext());
}

// C4IR (in-place)
NppStatus nppiDiv_Round_8u_C4IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor,
                                       NppStreamContext nppStreamCtx) {
  return DivRound<Npp8u, 4>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, rndMode, nScaleFactor,
                                            nppStreamCtx);
}

NppStatus nppiDiv_Round_8u_C4IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   NppRoundMode rndMode, int nScaleFactor) {
  return nppiDiv_Round_8u_C4IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, rndMode, nScaleFactor,
                                      getDefaultStreamContext());
}

// ============================================================================
// 16-bit unsigned (16u) - with scale factor and rounding mode
// ============================================================================

// C1R
NppStatus nppiDiv_Round_16u_C1RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step,
                                       Npp16u *pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode rndMode,
                                       int nScaleFactor, NppStreamContext nppStreamCtx) {
  return DivRound<Npp16u, 1>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, rndMode,
                                      nScaleFactor, nppStreamCtx);
}

NppStatus nppiDiv_Round_16u_C1RSfs(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                                   int nDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor) {
  return nppiDiv_Round_16u_C1RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, rndMode,
                                      nScaleFactor, getDefaultStreamContext());
}

// C1IR (in-place)
NppStatus nppiDiv_Round_16u_C1IRSfs_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor,
                                        NppStreamContext nppStreamCtx) {
  return DivRound<Npp16u, 1>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, rndMode, nScaleFactor,
                                             nppStreamCtx);
}

NppStatus nppiDiv_Round_16u_C1IRSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor) {
  return nppiDiv_Round_16u_C1IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, rndMode, nScaleFactor,
                                       getDefaultStreamContext());
}

// C3R
NppStatus nppiDiv_Round_16u_C3RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step,
                                       Npp16u *pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode rndMode,
                                       int nScaleFactor, NppStreamContext nppStreamCtx) {
  return DivRound<Npp16u, 3>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, rndMode,
                                      nScaleFactor, nppStreamCtx);
}

NppStatus nppiDiv_Round_16u_C3RSfs(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                                   int nDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor) {
  return nppiDiv_Round_16u_C3RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, rndMode,
                                      nScaleFactor, getDefaultStreamContext());
}

// C3IR (in-place)
NppStatus nppiDiv_Round_16u_C3IRSfs_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor,
                                        NppStreamContext nppStreamCtx) {
  return DivRound<Npp16u, 3>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, rndMode, nScaleFactor,
                                             nppStreamCtx);
}

NppStatus nppiDiv_Round_16u_C3IRSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor) {
  return nppiDiv_Round_16u_C3IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, rndMode, nScaleFactor,
                                       getDefaultStreamContext());
}

// AC4R
NppStatus nppiDiv_Round_16u_AC4RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step,
                                        Npp16u *pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode rndMode,
                                        int nScaleFactor, NppStreamContext nppStreamCtx) {
  return DivRound<Npp16u, 4>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, rndMode,
                                      nScaleFactor, nppStreamCtx);
}

NppStatus nppiDiv_Round_16u_AC4RSfs(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step,
                                    Npp16u *pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode rndMode,
                                    int nScaleFactor) {
  return nppiDiv_Round_16u_AC4RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, rndMode,
                                       nScaleFactor, getDefaultStreamContext());
}

// AC4IR (in-place)
NppStatus nppiDiv_Round_16u_AC4IRSfs_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor,
                                         NppStreamContext nppStreamCtx) {
  return DivRound<Npp16u, 4>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, rndMode, nScaleFactor,
                                             nppStreamCtx);
}

NppStatus nppiDiv_Round_16u_AC4IRSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor) {
  return nppiDiv_Round_16u_AC4IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, rndMode, nScaleFactor,
                                        getDefaultStreamContext());
}

// C4R
NppStatus nppiDiv_Round_16u_C4RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step,
                                       Npp16u *pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode rndMode,
                                       int nScaleFactor, NppStreamContext nppStreamCtx) {
  return DivRound<Npp16u, 4>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, rndMode,
                                      nScaleFactor, nppStreamCtx);
}

NppStatus nppiDiv_Round_16u_C4RSfs(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                                   int nDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor) {
  return nppiDiv_Round_16u_C4RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, rndMode,
                                      nScaleFactor, getDefaultStreamContext());
}

// C4IR (in-place)
NppStatus nppiDiv_Round_16u_C4IRSfs_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor,
                                        NppStreamContext nppStreamCtx) {
  return DivRound<Npp16u, 4>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, rndMode, nScaleFactor,
                                             nppStreamCtx);
}

NppStatus nppiDiv_Round_16u_C4IRSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor) {
  return nppiDiv_Round_16u_C4IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, rndMode, nScaleFactor,
                                       getDefaultStreamContext());
}

// ============================================================================
// 16-bit signed (16s) - with scale factor and rounding mode
// ============================================================================

// C1R
NppStatus nppiDiv_Round_16s_C1RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step, const Npp16s *pSrc2, int nSrc2Step,
                                       Npp16s *pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode rndMode,
                                       int nScaleFactor, NppStreamContext nppStreamCtx) {
  return DivRound<Npp16s, 1>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, rndMode,
                                      nScaleFactor, nppStreamCtx);
}

NppStatus nppiDiv_Round_16s_C1RSfs(const Npp16s *pSrc1, int nSrc1Step, const Npp16s *pSrc2, int nSrc2Step, Npp16s *pDst,
                                   int nDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor) {
  return nppiDiv_Round_16s_C1RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, rndMode,
                                      nScaleFactor, getDefaultStreamContext());
}

// C1IR (in-place)
NppStatus nppiDiv_Round_16s_C1IRSfs_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor,
                                        NppStreamContext nppStreamCtx) {
  return DivRound<Npp16s, 1>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, rndMode, nScaleFactor,
                                             nppStreamCtx);
}

NppStatus nppiDiv_Round_16s_C1IRSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor) {
  return nppiDiv_Round_16s_C1IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, rndMode, nScaleFactor,
                                       getDefaultStreamContext());
}

// C3R
NppStatus nppiDiv_Round_16s_C3RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step, const Npp16s *pSrc2, int nSrc2Step,
                                       Npp16s *pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode rndMode,
                                       int nScaleFactor, NppStreamContext nppStreamCtx) {
  return DivRound<Npp16s, 3>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, rndMode,
                                      nScaleFactor, nppStreamCtx);
}

NppStatus nppiDiv_Round_16s_C3RSfs(const Npp16s *pSrc1, int nSrc1Step, const Npp16s *pSrc2, int nSrc2Step, Npp16s *pDst,
                                   int nDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor) {
  return nppiDiv_Round_16s_C3RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, rndMode,
                                      nScaleFactor, getDefaultStreamContext());
}

// C3IR (in-place)
NppStatus nppiDiv_Round_16s_C3IRSfs_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor,
                                        NppStreamContext nppStreamCtx) {
  return DivRound<Npp16s, 3>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, rndMode, nScaleFactor,
                                             nppStreamCtx);
}

NppStatus nppiDiv_Round_16s_C3IRSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor) {
  return nppiDiv_Round_16s_C3IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, rndMode, nScaleFactor,
                                       getDefaultStreamContext());
}

// AC4R
NppStatus nppiDiv_Round_16s_AC4RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step, const Npp16s *pSrc2, int nSrc2Step,
                                        Npp16s *pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode rndMode,
                                        int nScaleFactor, NppStreamContext nppStreamCtx) {
  return DivRound<Npp16s, 4>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, rndMode,
                                      nScaleFactor, nppStreamCtx);
}

NppStatus nppiDiv_Round_16s_AC4RSfs(const Npp16s *pSrc1, int nSrc1Step, const Npp16s *pSrc2, int nSrc2Step,
                                    Npp16s *pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode rndMode,
                                    int nScaleFactor) {
  return nppiDiv_Round_16s_AC4RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, rndMode,
                                       nScaleFactor, getDefaultStreamContext());
}

// AC4IR (in-place)
NppStatus nppiDiv_Round_16s_AC4IRSfs_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor,
                                         NppStreamContext nppStreamCtx) {
  return DivRound<Npp16s, 4>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, rndMode, nScaleFactor,
                                             nppStreamCtx);
}

NppStatus nppiDiv_Round_16s_AC4IRSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor) {
  return nppiDiv_Round_16s_AC4IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, rndMode, nScaleFactor,
                                        getDefaultStreamContext());
}

// C4R
NppStatus nppiDiv_Round_16s_C4RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step, const Npp16s *pSrc2, int nSrc2Step,
                                       Npp16s *pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode rndMode,
                                       int nScaleFactor, NppStreamContext nppStreamCtx) {
  return DivRound<Npp16s, 4>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, rndMode,
                                      nScaleFactor, nppStreamCtx);
}

NppStatus nppiDiv_Round_16s_C4RSfs(const Npp16s *pSrc1, int nSrc1Step, const Npp16s *pSrc2, int nSrc2Step, Npp16s *pDst,
                                   int nDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor) {
  return nppiDiv_Round_16s_C4RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, rndMode,
                                      nScaleFactor, getDefaultStreamContext());
}

// C4IR (in-place)
NppStatus nppiDiv_Round_16s_C4IRSfs_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor,
                                        NppStreamContext nppStreamCtx) {
  return DivRound<Npp16s, 4>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, rndMode, nScaleFactor,
                                             nppStreamCtx);
}

NppStatus nppiDiv_Round_16s_C4IRSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor) {
  return nppiDiv_Round_16s_C4IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, rndMode, nScaleFactor,
                                       getDefaultStreamContext());
}
