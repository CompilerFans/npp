// Square operation using template-based API
#include "nppi_arithmetic_api.h"

using namespace nppi::arithmetic;

template <typename T, int C> using Sqr = UnaryOpAPI<T, C, SqrOp>;

// ============================================================================
// 8u variants
// ============================================================================

NppStatus nppiSqr_8u_C1RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sqr<Npp8u, 1>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSqr_8u_C1RSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                            int nScaleFactor) {
  return nppiSqr_8u_C1RSfs_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiSqr_8u_C1IRSfs_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx) {
  return Sqr<Npp8u, 1>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSqr_8u_C1IRSfs(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiSqr_8u_C1IRSfs_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiSqr_8u_C3RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sqr<Npp8u, 3>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSqr_8u_C3RSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                            int nScaleFactor) {
  return nppiSqr_8u_C3RSfs_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiSqr_8u_C3IRSfs_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx) {
  return Sqr<Npp8u, 3>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSqr_8u_C3IRSfs(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiSqr_8u_C3IRSfs_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiSqr_8u_AC4RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sqr<Npp8u, 4>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSqr_8u_AC4RSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                             int nScaleFactor) {
  return nppiSqr_8u_AC4RSfs_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiSqr_8u_AC4IRSfs_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx) {
  return Sqr<Npp8u, 4>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSqr_8u_AC4IRSfs(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiSqr_8u_AC4IRSfs_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiSqr_8u_C4RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sqr<Npp8u, 4>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSqr_8u_C4RSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                            int nScaleFactor) {
  return nppiSqr_8u_C4RSfs_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiSqr_8u_C4IRSfs_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx) {
  return Sqr<Npp8u, 4>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSqr_8u_C4IRSfs(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiSqr_8u_C4IRSfs_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

// ============================================================================
// 16u variants
// ============================================================================

NppStatus nppiSqr_16u_C1RSfs_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sqr<Npp16u, 1>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSqr_16u_C1RSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                             int nScaleFactor) {
  return nppiSqr_16u_C1RSfs_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiSqr_16u_C1IRSfs_Ctx(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx) {
  return Sqr<Npp16u, 1>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSqr_16u_C1IRSfs(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiSqr_16u_C1IRSfs_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiSqr_16u_C3RSfs_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sqr<Npp16u, 3>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSqr_16u_C3RSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                             int nScaleFactor) {
  return nppiSqr_16u_C3RSfs_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiSqr_16u_C3IRSfs_Ctx(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx) {
  return Sqr<Npp16u, 3>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSqr_16u_C3IRSfs(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiSqr_16u_C3IRSfs_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiSqr_16u_AC4RSfs_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sqr<Npp16u, 4>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSqr_16u_AC4RSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor) {
  return nppiSqr_16u_AC4RSfs_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiSqr_16u_AC4IRSfs_Ctx(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx) {
  return Sqr<Npp16u, 4>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSqr_16u_AC4IRSfs(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiSqr_16u_AC4IRSfs_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiSqr_16u_C4RSfs_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sqr<Npp16u, 4>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSqr_16u_C4RSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                             int nScaleFactor) {
  return nppiSqr_16u_C4RSfs_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiSqr_16u_C4IRSfs_Ctx(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx) {
  return Sqr<Npp16u, 4>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSqr_16u_C4IRSfs(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiSqr_16u_C4IRSfs_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

// ============================================================================
// 16s variants
// ============================================================================

NppStatus nppiSqr_16s_C1RSfs_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sqr<Npp16s, 1>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSqr_16s_C1RSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                             int nScaleFactor) {
  return nppiSqr_16s_C1RSfs_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiSqr_16s_C1IRSfs_Ctx(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx) {
  return Sqr<Npp16s, 1>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSqr_16s_C1IRSfs(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiSqr_16s_C1IRSfs_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiSqr_16s_C3RSfs_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sqr<Npp16s, 3>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSqr_16s_C3RSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                             int nScaleFactor) {
  return nppiSqr_16s_C3RSfs_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiSqr_16s_C3IRSfs_Ctx(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx) {
  return Sqr<Npp16s, 3>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSqr_16s_C3IRSfs(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiSqr_16s_C3IRSfs_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiSqr_16s_AC4RSfs_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sqr<Npp16s, 4>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSqr_16s_AC4RSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor) {
  return nppiSqr_16s_AC4RSfs_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiSqr_16s_AC4IRSfs_Ctx(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx) {
  return Sqr<Npp16s, 4>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSqr_16s_AC4IRSfs(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiSqr_16s_AC4IRSfs_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiSqr_16s_C4RSfs_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sqr<Npp16s, 4>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSqr_16s_C4RSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                             int nScaleFactor) {
  return nppiSqr_16s_C4RSfs_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiSqr_16s_C4IRSfs_Ctx(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx) {
  return Sqr<Npp16s, 4>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSqr_16s_C4IRSfs(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiSqr_16s_C4IRSfs_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

// ============================================================================
// 32f variants (no scale factor)
// ============================================================================

NppStatus nppiSqr_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx) {
  return Sqr<Npp32f, 1>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSqr_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI) {
  return nppiSqr_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiSqr_32f_C1IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Sqr<Npp32f, 1>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSqr_32f_C1IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiSqr_32f_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiSqr_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx) {
  return Sqr<Npp32f, 3>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSqr_32f_C3R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI) {
  return nppiSqr_32f_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiSqr_32f_C3IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Sqr<Npp32f, 3>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSqr_32f_C3IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiSqr_32f_C3IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiSqr_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx) {
  return Sqr<Npp32f, 4>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSqr_32f_AC4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI) {
  return nppiSqr_32f_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiSqr_32f_AC4IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Sqr<Npp32f, 4>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSqr_32f_AC4IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiSqr_32f_AC4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiSqr_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx) {
  return Sqr<Npp32f, 4>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSqr_32f_C4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI) {
  return nppiSqr_32f_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiSqr_32f_C4IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Sqr<Npp32f, 4>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSqr_32f_C4IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiSqr_32f_C4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}
