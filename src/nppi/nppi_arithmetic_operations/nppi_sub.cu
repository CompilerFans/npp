// ============================================================================
// NPP Sub Operations - Template-based Implementation
// ============================================================================
// This file implements all nppiSub* functions using template classes.
// No macros are used - only C++ templates for code generation.
// ============================================================================

#include "nppi_arithmetic_api.h"

using namespace nppi::arithmetic;

// ============================================================================
// Type aliases for cleaner code
// ============================================================================
template <typename T, int C> using Sub = BinaryOpAPI<T, C, SubOp>;

// ============================================================================
// 8-bit unsigned (8u) - with scale factor
// ============================================================================

// C1
NppStatus nppiSub_8u_C1RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sub<Npp8u, 1>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                nppStreamCtx);
}

NppStatus nppiSub_8u_C1RSfs(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiSub_8u_C1RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                               getDefaultStreamContext());
}

NppStatus nppiSub_8u_C1IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                 int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sub<Npp8u, 1>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSub_8u_C1IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                             int nScaleFactor) {
  return nppiSub_8u_C1IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor,
                                getDefaultStreamContext());
}

// C3
NppStatus nppiSub_8u_C3RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sub<Npp8u, 3>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                nppStreamCtx);
}

NppStatus nppiSub_8u_C3RSfs(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiSub_8u_C3RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                               getDefaultStreamContext());
}

NppStatus nppiSub_8u_C3IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                 int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sub<Npp8u, 3>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSub_8u_C3IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                             int nScaleFactor) {
  return nppiSub_8u_C3IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor,
                                getDefaultStreamContext());
}

// C4
NppStatus nppiSub_8u_C4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sub<Npp8u, 4>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                nppStreamCtx);
}

NppStatus nppiSub_8u_C4RSfs(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiSub_8u_C4RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                               getDefaultStreamContext());
}

NppStatus nppiSub_8u_C4IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                 int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sub<Npp8u, 4>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSub_8u_C4IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                             int nScaleFactor) {
  return nppiSub_8u_C4IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor,
                                getDefaultStreamContext());
}

// AC4
NppStatus nppiSub_8u_AC4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sub<Npp8u, 4>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                nppStreamCtx);
}

NppStatus nppiSub_8u_AC4RSfs(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiSub_8u_AC4RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                getDefaultStreamContext());
}

NppStatus nppiSub_8u_AC4IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                  int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sub<Npp8u, 4>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSub_8u_AC4IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor) {
  return nppiSub_8u_AC4IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor,
                                 getDefaultStreamContext());
}

// ============================================================================
// 16-bit unsigned (16u) - with scale factor
// ============================================================================

// C1
NppStatus nppiSub_16u_C1RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                                 int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sub<Npp16u, 1>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                 nppStreamCtx);
}

NppStatus nppiSub_16u_C1RSfs(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiSub_16u_C1RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                getDefaultStreamContext());
}

NppStatus nppiSub_16u_C1IRSfs_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                  int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sub<Npp16u, 1>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSub_16u_C1IRSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor) {
  return nppiSub_16u_C1IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor,
                                 getDefaultStreamContext());
}

// C3
NppStatus nppiSub_16u_C3RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                                 int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sub<Npp16u, 3>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                 nppStreamCtx);
}

NppStatus nppiSub_16u_C3RSfs(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiSub_16u_C3RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                getDefaultStreamContext());
}

NppStatus nppiSub_16u_C3IRSfs_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                  int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sub<Npp16u, 3>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSub_16u_C3IRSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor) {
  return nppiSub_16u_C3IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor,
                                 getDefaultStreamContext());
}

// C4
NppStatus nppiSub_16u_C4RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                                 int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sub<Npp16u, 4>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                 nppStreamCtx);
}

NppStatus nppiSub_16u_C4RSfs(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiSub_16u_C4RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                getDefaultStreamContext());
}

NppStatus nppiSub_16u_C4IRSfs_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                  int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sub<Npp16u, 4>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSub_16u_C4IRSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor) {
  return nppiSub_16u_C4IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor,
                                 getDefaultStreamContext());
}

// AC4
NppStatus nppiSub_16u_AC4RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                                  int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sub<Npp16u, 4>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                 nppStreamCtx);
}

NppStatus nppiSub_16u_AC4RSfs(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiSub_16u_AC4RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                 getDefaultStreamContext());
}

NppStatus nppiSub_16u_AC4IRSfs_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sub<Npp16u, 4>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSub_16u_AC4IRSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor) {
  return nppiSub_16u_AC4IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor,
                                  getDefaultStreamContext());
}

// ============================================================================
// 16-bit signed (16s) - with scale factor
// ============================================================================

// C1
NppStatus nppiSub_16s_C1RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step, const Npp16s *pSrc2, int nSrc2Step, Npp16s *pDst,
                                 int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sub<Npp16s, 1>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                 nppStreamCtx);
}

NppStatus nppiSub_16s_C1RSfs(const Npp16s *pSrc1, int nSrc1Step, const Npp16s *pSrc2, int nSrc2Step, Npp16s *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiSub_16s_C1RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                getDefaultStreamContext());
}

NppStatus nppiSub_16s_C1IRSfs_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                  int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sub<Npp16s, 1>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSub_16s_C1IRSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor) {
  return nppiSub_16s_C1IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor,
                                 getDefaultStreamContext());
}

// C3
NppStatus nppiSub_16s_C3RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step, const Npp16s *pSrc2, int nSrc2Step, Npp16s *pDst,
                                 int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sub<Npp16s, 3>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                 nppStreamCtx);
}

NppStatus nppiSub_16s_C3RSfs(const Npp16s *pSrc1, int nSrc1Step, const Npp16s *pSrc2, int nSrc2Step, Npp16s *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiSub_16s_C3RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                getDefaultStreamContext());
}

NppStatus nppiSub_16s_C3IRSfs_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                  int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sub<Npp16s, 3>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSub_16s_C3IRSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor) {
  return nppiSub_16s_C3IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor,
                                 getDefaultStreamContext());
}

// C4
NppStatus nppiSub_16s_C4RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step, const Npp16s *pSrc2, int nSrc2Step, Npp16s *pDst,
                                 int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sub<Npp16s, 4>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                 nppStreamCtx);
}

NppStatus nppiSub_16s_C4RSfs(const Npp16s *pSrc1, int nSrc1Step, const Npp16s *pSrc2, int nSrc2Step, Npp16s *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiSub_16s_C4RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                getDefaultStreamContext());
}

NppStatus nppiSub_16s_C4IRSfs_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                  int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sub<Npp16s, 4>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSub_16s_C4IRSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor) {
  return nppiSub_16s_C4IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor,
                                 getDefaultStreamContext());
}

// AC4
NppStatus nppiSub_16s_AC4RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step, const Npp16s *pSrc2, int nSrc2Step, Npp16s *pDst,
                                  int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sub<Npp16s, 4>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                 nppStreamCtx);
}

NppStatus nppiSub_16s_AC4RSfs(const Npp16s *pSrc1, int nSrc1Step, const Npp16s *pSrc2, int nSrc2Step, Npp16s *pDst,
                              int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiSub_16s_AC4RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                 getDefaultStreamContext());
}

NppStatus nppiSub_16s_AC4IRSfs_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sub<Npp16s, 4>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSub_16s_AC4IRSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor) {
  return nppiSub_16s_AC4IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor,
                                  getDefaultStreamContext());
}

// ============================================================================
// 32-bit float (32f) - no scale factor
// ============================================================================

// C1
NppStatus nppiSub_32f_C1R_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Sub<Npp32f, 1>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSub_32f_C1R(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                          int nDstStep, NppiSize oSizeROI) {
  return nppiSub_32f_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiSub_32f_C1IR_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx) {
  return Sub<Npp32f, 1>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSub_32f_C1IR(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiSub_32f_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// C3
NppStatus nppiSub_32f_C3R_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Sub<Npp32f, 3>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSub_32f_C3R(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                          int nDstStep, NppiSize oSizeROI) {
  return nppiSub_32f_C3R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiSub_32f_C3IR_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx) {
  return Sub<Npp32f, 3>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSub_32f_C3IR(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiSub_32f_C3IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// C4
NppStatus nppiSub_32f_C4R_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Sub<Npp32f, 4>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSub_32f_C4R(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                          int nDstStep, NppiSize oSizeROI) {
  return nppiSub_32f_C4R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiSub_32f_C4IR_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx) {
  return Sub<Npp32f, 4>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSub_32f_C4IR(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiSub_32f_C4IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// AC4
NppStatus nppiSub_32f_AC4R_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                               int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Sub<Npp32f, 4>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSub_32f_AC4R(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                           int nDstStep, NppiSize oSizeROI) {
  return nppiSub_32f_AC4R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiSub_32f_AC4IR_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx) {
  return Sub<Npp32f, 4>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSub_32f_AC4IR(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiSub_32f_AC4IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// ============================================================================
// 16-bit signed complex (16sc) - with scale factor
// ============================================================================

// C1
NppStatus nppiSub_16sc_C1RSfs_Ctx(const Npp16sc *pSrc1, int nSrc1Step, const Npp16sc *pSrc2, int nSrc2Step,
                                  Npp16sc *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx) {
  return Sub<Npp16sc, 1>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                  nppStreamCtx);
}

NppStatus nppiSub_16sc_C1RSfs(const Npp16sc *pSrc1, int nSrc1Step, const Npp16sc *pSrc2, int nSrc2Step, Npp16sc *pDst,
                              int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiSub_16sc_C1RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                 getDefaultStreamContext());
}

NppStatus nppiSub_16sc_C1IRSfs_Ctx(const Npp16sc *pSrc, int nSrcStep, Npp16sc *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sub<Npp16sc, 1>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSub_16sc_C1IRSfs(const Npp16sc *pSrc, int nSrcStep, Npp16sc *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor) {
  return nppiSub_16sc_C1IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor,
                                  getDefaultStreamContext());
}

// C3
NppStatus nppiSub_16sc_C3RSfs_Ctx(const Npp16sc *pSrc1, int nSrc1Step, const Npp16sc *pSrc2, int nSrc2Step,
                                  Npp16sc *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx) {
  return Sub<Npp16sc, 3>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                  nppStreamCtx);
}

NppStatus nppiSub_16sc_C3RSfs(const Npp16sc *pSrc1, int nSrc1Step, const Npp16sc *pSrc2, int nSrc2Step, Npp16sc *pDst,
                              int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiSub_16sc_C3RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                 getDefaultStreamContext());
}

NppStatus nppiSub_16sc_C3IRSfs_Ctx(const Npp16sc *pSrc, int nSrcStep, Npp16sc *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sub<Npp16sc, 3>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSub_16sc_C3IRSfs(const Npp16sc *pSrc, int nSrcStep, Npp16sc *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor) {
  return nppiSub_16sc_C3IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor,
                                  getDefaultStreamContext());
}

// AC4
NppStatus nppiSub_16sc_AC4RSfs_Ctx(const Npp16sc *pSrc1, int nSrc1Step, const Npp16sc *pSrc2, int nSrc2Step,
                                   Npp16sc *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx) {
  return Sub<Npp16sc, 4>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                  nppStreamCtx);
}

NppStatus nppiSub_16sc_AC4RSfs(const Npp16sc *pSrc1, int nSrc1Step, const Npp16sc *pSrc2, int nSrc2Step, Npp16sc *pDst,
                               int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiSub_16sc_AC4RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                  getDefaultStreamContext());
}

NppStatus nppiSub_16sc_AC4IRSfs_Ctx(const Npp16sc *pSrc, int nSrcStep, Npp16sc *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sub<Npp16sc, 4>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSub_16sc_AC4IRSfs(const Npp16sc *pSrc, int nSrcStep, Npp16sc *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                int nScaleFactor) {
  return nppiSub_16sc_AC4IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor,
                                   getDefaultStreamContext());
}

// ============================================================================
// 32-bit signed complex (32sc) - with scale factor
// ============================================================================

// C1
NppStatus nppiSub_32sc_C1RSfs_Ctx(const Npp32sc *pSrc1, int nSrc1Step, const Npp32sc *pSrc2, int nSrc2Step,
                                  Npp32sc *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx) {
  return Sub<Npp32sc, 1>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                  nppStreamCtx);
}

NppStatus nppiSub_32sc_C1RSfs(const Npp32sc *pSrc1, int nSrc1Step, const Npp32sc *pSrc2, int nSrc2Step, Npp32sc *pDst,
                              int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiSub_32sc_C1RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                 getDefaultStreamContext());
}

NppStatus nppiSub_32sc_C1IRSfs_Ctx(const Npp32sc *pSrc, int nSrcStep, Npp32sc *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sub<Npp32sc, 1>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSub_32sc_C1IRSfs(const Npp32sc *pSrc, int nSrcStep, Npp32sc *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor) {
  return nppiSub_32sc_C1IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor,
                                  getDefaultStreamContext());
}

// C3
NppStatus nppiSub_32sc_C3RSfs_Ctx(const Npp32sc *pSrc1, int nSrc1Step, const Npp32sc *pSrc2, int nSrc2Step,
                                  Npp32sc *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx) {
  return Sub<Npp32sc, 3>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                  nppStreamCtx);
}

NppStatus nppiSub_32sc_C3RSfs(const Npp32sc *pSrc1, int nSrc1Step, const Npp32sc *pSrc2, int nSrc2Step, Npp32sc *pDst,
                              int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiSub_32sc_C3RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                 getDefaultStreamContext());
}

NppStatus nppiSub_32sc_C3IRSfs_Ctx(const Npp32sc *pSrc, int nSrcStep, Npp32sc *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sub<Npp32sc, 3>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSub_32sc_C3IRSfs(const Npp32sc *pSrc, int nSrcStep, Npp32sc *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor) {
  return nppiSub_32sc_C3IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor,
                                  getDefaultStreamContext());
}

// AC4
NppStatus nppiSub_32sc_AC4RSfs_Ctx(const Npp32sc *pSrc1, int nSrc1Step, const Npp32sc *pSrc2, int nSrc2Step,
                                   Npp32sc *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx) {
  return Sub<Npp32sc, 4>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                  nppStreamCtx);
}

NppStatus nppiSub_32sc_AC4RSfs(const Npp32sc *pSrc1, int nSrc1Step, const Npp32sc *pSrc2, int nSrc2Step, Npp32sc *pDst,
                               int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiSub_32sc_AC4RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                  getDefaultStreamContext());
}

NppStatus nppiSub_32sc_AC4IRSfs_Ctx(const Npp32sc *pSrc, int nSrcStep, Npp32sc *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Sub<Npp32sc, 4>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSub_32sc_AC4IRSfs(const Npp32sc *pSrc, int nSrcStep, Npp32sc *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                int nScaleFactor) {
  return nppiSub_32sc_AC4IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor,
                                   getDefaultStreamContext());
}

// ============================================================================
// 32-bit float complex (32fc) - no scale factor
// ============================================================================

// C1
NppStatus nppiSub_32fc_C1R_Ctx(const Npp32fc *pSrc1, int nSrc1Step, const Npp32fc *pSrc2, int nSrc2Step, Npp32fc *pDst,
                               int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Sub<Npp32fc, 1>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSub_32fc_C1R(const Npp32fc *pSrc1, int nSrc1Step, const Npp32fc *pSrc2, int nSrc2Step, Npp32fc *pDst,
                           int nDstStep, NppiSize oSizeROI) {
  return nppiSub_32fc_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiSub_32fc_C1IR_Ctx(const Npp32fc *pSrc, int nSrcStep, Npp32fc *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx) {
  return Sub<Npp32fc, 1>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSub_32fc_C1IR(const Npp32fc *pSrc, int nSrcStep, Npp32fc *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiSub_32fc_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// C3
NppStatus nppiSub_32fc_C3R_Ctx(const Npp32fc *pSrc1, int nSrc1Step, const Npp32fc *pSrc2, int nSrc2Step, Npp32fc *pDst,
                               int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Sub<Npp32fc, 3>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSub_32fc_C3R(const Npp32fc *pSrc1, int nSrc1Step, const Npp32fc *pSrc2, int nSrc2Step, Npp32fc *pDst,
                           int nDstStep, NppiSize oSizeROI) {
  return nppiSub_32fc_C3R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiSub_32fc_C3IR_Ctx(const Npp32fc *pSrc, int nSrcStep, Npp32fc *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx) {
  return Sub<Npp32fc, 3>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSub_32fc_C3IR(const Npp32fc *pSrc, int nSrcStep, Npp32fc *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiSub_32fc_C3IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// AC4
NppStatus nppiSub_32fc_AC4R_Ctx(const Npp32fc *pSrc1, int nSrc1Step, const Npp32fc *pSrc2, int nSrc2Step, Npp32fc *pDst,
                                int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Sub<Npp32fc, 4>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSub_32fc_AC4R(const Npp32fc *pSrc1, int nSrc1Step, const Npp32fc *pSrc2, int nSrc2Step, Npp32fc *pDst,
                            int nDstStep, NppiSize oSizeROI) {
  return nppiSub_32fc_AC4R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiSub_32fc_AC4IR_Ctx(const Npp32fc *pSrc, int nSrcStep, Npp32fc *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Sub<Npp32fc, 4>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSub_32fc_AC4IR(const Npp32fc *pSrc, int nSrcStep, Npp32fc *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiSub_32fc_AC4IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// C4
NppStatus nppiSub_32fc_C4R_Ctx(const Npp32fc *pSrc1, int nSrc1Step, const Npp32fc *pSrc2, int nSrc2Step, Npp32fc *pDst,
                               int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Sub<Npp32fc, 4>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSub_32fc_C4R(const Npp32fc *pSrc1, int nSrc1Step, const Npp32fc *pSrc2, int nSrc2Step, Npp32fc *pDst,
                           int nDstStep, NppiSize oSizeROI) {
  return nppiSub_32fc_C4R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiSub_32fc_C4IR_Ctx(const Npp32fc *pSrc, int nSrcStep, Npp32fc *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx) {
  return Sub<Npp32fc, 4>::executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSub_32fc_C4IR(const Npp32fc *pSrc, int nSrcStep, Npp32fc *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiSub_32fc_C4IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}
