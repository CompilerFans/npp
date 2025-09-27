#include "nppi_arithmetic_executor.h"
#include "nppi_arithmetic_ops.h"

using namespace nppi::arithmetic;

// Implementation functions using the unified executor
extern "C" {

NppStatus nppiLn_8u_C1RSfs_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                    int nScaleFactor, NppStreamContext nppStreamCtx) {
  return UnaryOperationExecutor<Npp8u, 1, LnOp<Npp8u>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                                nppStreamCtx.hStream);
}

NppStatus nppiLn_16u_C1RSfs_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                     int nScaleFactor, NppStreamContext nppStreamCtx) {
  return UnaryOperationExecutor<Npp16u, 1, LnOp<Npp16u>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                                  nScaleFactor, nppStreamCtx.hStream);
}

NppStatus nppiLn_16s_C1RSfs_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                     int nScaleFactor, NppStreamContext nppStreamCtx) {
  return UnaryOperationExecutor<Npp16s, 1, LnOp<Npp16s>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                                  nScaleFactor, nppStreamCtx.hStream);
}

NppStatus nppiLn_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
  return UnaryOperationExecutor<Npp32f, 1, LnOp<Npp32f>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                                  nppStreamCtx.hStream);
}

} // extern "C"

// Public API functions
NppStatus nppiLn_8u_C1RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                               int nScaleFactor, NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return nppiLn_8u_C1RSfs_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiLn_8u_C1RSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                           int nScaleFactor) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiLn_8u_C1RSfs_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiLn_16u_C1RSfs_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor, NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return nppiLn_16u_C1RSfs_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiLn_16u_C1RSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                            int nScaleFactor) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiLn_16u_C1RSfs_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiLn_16s_C1RSfs_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor, NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return nppiLn_16s_C1RSfs_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiLn_16s_C1RSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                            int nScaleFactor) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiLn_16s_C1RSfs_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiLn_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx) {
  return nppiLn_32f_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLn_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiLn_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLn_32f_C1IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiLn_32f_C1R_Ctx_impl(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLn_32f_C1IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiLn_32f_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}
