// Natural logarithm operation using template-based API
#include "nppi_arithmetic_api.h"

using namespace nppi::arithmetic;

template <typename T, int C> using Ln = UnaryOpAPI<T, C, LnOp>;

// ============================================================================
// Npp8u - Unsigned 8-bit (with scale factor)
// ============================================================================

// C1RSfs
NppStatus nppiLn_8u_C1RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                               int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Ln<Npp8u, 1>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiLn_8u_C1RSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                           int nScaleFactor) {
  return nppiLn_8u_C1RSfs_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

// C1IRSfs (in-place)
NppStatus nppiLn_8u_C1IRSfs_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor,
                                NppStreamContext nppStreamCtx) {
  return Ln<Npp8u, 1>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiLn_8u_C1IRSfs(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiLn_8u_C1IRSfs_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

// ============================================================================
// Npp16u - Unsigned 16-bit (with scale factor)
// ============================================================================

// C1RSfs
NppStatus nppiLn_16u_C1RSfs_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Ln<Npp16u, 1>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiLn_16u_C1RSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                            int nScaleFactor) {
  return nppiLn_16u_C1RSfs_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

// C1IRSfs (in-place)
NppStatus nppiLn_16u_C1IRSfs_Ctx(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx) {
  return Ln<Npp16u, 1>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiLn_16u_C1IRSfs(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiLn_16u_C1IRSfs_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

// ============================================================================
// Npp16s - Signed 16-bit (with scale factor)
// ============================================================================

// C1RSfs
NppStatus nppiLn_16s_C1RSfs_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Ln<Npp16s, 1>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiLn_16s_C1RSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                            int nScaleFactor) {
  return nppiLn_16s_C1RSfs_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

// C1IRSfs (in-place)
NppStatus nppiLn_16s_C1IRSfs_Ctx(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx) {
  return Ln<Npp16s, 1>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiLn_16s_C1IRSfs(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiLn_16s_C1IRSfs_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

// ============================================================================
// Npp32f - 32-bit Float (no scale factor)
// ============================================================================

// C1R
NppStatus nppiLn_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx) {
  return Ln<Npp32f, 1>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLn_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI) {
  return nppiLn_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

// C1IR (in-place)
NppStatus nppiLn_32f_C1IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Ln<Npp32f, 1>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLn_32f_C1IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiLn_32f_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// ============================================================================
// C3 variants
// ============================================================================

// 8u C3
NppStatus nppiLn_8u_C3RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                               int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Ln<Npp8u, 3>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiLn_8u_C3RSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                           int nScaleFactor) {
  return nppiLn_8u_C3RSfs_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiLn_8u_C3IRSfs_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor,
                                NppStreamContext nppStreamCtx) {
  return Ln<Npp8u, 3>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiLn_8u_C3IRSfs(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiLn_8u_C3IRSfs_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

// 16u C3
NppStatus nppiLn_16u_C3RSfs_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Ln<Npp16u, 3>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiLn_16u_C3RSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                            int nScaleFactor) {
  return nppiLn_16u_C3RSfs_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiLn_16u_C3IRSfs_Ctx(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx) {
  return Ln<Npp16u, 3>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiLn_16u_C3IRSfs(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiLn_16u_C3IRSfs_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

// 16s C3
NppStatus nppiLn_16s_C3RSfs_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor, NppStreamContext nppStreamCtx) {
  return Ln<Npp16s, 3>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiLn_16s_C3RSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                            int nScaleFactor) {
  return nppiLn_16s_C3RSfs_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

NppStatus nppiLn_16s_C3IRSfs_Ctx(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx) {
  return Ln<Npp16s, 3>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiLn_16s_C3IRSfs(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor) {
  return nppiLn_16s_C3IRSfs_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, getDefaultStreamContext());
}

// 32f C3
NppStatus nppiLn_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx) {
  return Ln<Npp32f, 3>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLn_32f_C3R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI) {
  return nppiLn_32f_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiLn_32f_C3IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Ln<Npp32f, 3>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLn_32f_C3IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiLn_32f_C3IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// ============================================================================
// 16-bit float (16f)
// ============================================================================

// C1
NppStatus nppiLn_16f_C1R_Ctx(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst, int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx) {
  return Ln<Npp16f, 1>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLn_16f_C1R(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst, int nDstStep, NppiSize oSizeROI) {
  return nppiLn_16f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiLn_16f_C1IR_Ctx(Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Ln<Npp16f, 1>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLn_16f_C1IR(Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiLn_16f_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// C3
NppStatus nppiLn_16f_C3R_Ctx(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst, int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx) {
  return Ln<Npp16f, 3>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLn_16f_C3R(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst, int nDstStep, NppiSize oSizeROI) {
  return nppiLn_16f_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiLn_16f_C3IR_Ctx(Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Ln<Npp16f, 3>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLn_16f_C3IR(Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiLn_16f_C3IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// C4
NppStatus nppiLn_16f_C4R_Ctx(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst, int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx) {
  return Ln<Npp16f, 4>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLn_16f_C4R(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst, int nDstStep, NppiSize oSizeROI) {
  return nppiLn_16f_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiLn_16f_C4IR_Ctx(Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return Ln<Npp16f, 4>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLn_16f_C4IR(Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiLn_16f_C4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}
