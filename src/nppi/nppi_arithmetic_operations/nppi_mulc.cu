#include "nppi_arithmetic_executor.h"
#include "nppi_arithmetic_ops.h"

using namespace nppi::arithmetic;

// Implementation functions using the unified executor
extern "C" {

// 8u C1 operations with scale factor
NppStatus nppiMulC_8u_C1RSfs_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u nConstant, Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  MulConstOp<Npp8u> op(nConstant);
  return ConstOperationExecutor<Npp8u, 1, MulConstOp<Npp8u>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                                      nScaleFactor, nppStreamCtx.hStream, op);
}

// 16u C1 operations with scale factor
NppStatus nppiMulC_16u_C1RSfs_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u nConstant, Npp16u *pDst, int nDstStep,
                                       NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  MulConstOp<Npp16u> op(nConstant);
  return ConstOperationExecutor<Npp16u, 1, MulConstOp<Npp16u>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                                        nScaleFactor, nppStreamCtx.hStream, op);
}

// 16s C1 operations with scale factor
NppStatus nppiMulC_16s_C1RSfs_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s nConstant, Npp16s *pDst, int nDstStep,
                                       NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  MulConstOp<Npp16s> op(nConstant);
  return ConstOperationExecutor<Npp16s, 1, MulConstOp<Npp16s>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                                        nScaleFactor, nppStreamCtx.hStream, op);
}

// 32f C1 operations (no scale factor)
NppStatus nppiMulC_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f nConstant, Npp32f *pDst, int nDstStep,
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  MulConstOp<Npp32f> op(nConstant);
  return ConstOperationExecutor<Npp32f, 1, MulConstOp<Npp32f>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                                        nppStreamCtx.hStream, op);
}

} // extern "C"

// Public API functions
NppStatus nppiMulC_8u_C1RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u nConstant, Npp8u *pDst, int nDstStep,
                                 NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return nppiMulC_8u_C1RSfs_Ctx_impl(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiMulC_8u_C1RSfs(const Npp8u *pSrc, int nSrcStep, Npp8u nConstant, Npp8u *pDst, int nDstStep,
                             NppiSize oSizeROI, int nScaleFactor) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiMulC_8u_C1RSfs_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiMulC_16u_C1RSfs_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u nConstant, Npp16u *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return nppiMulC_16u_C1RSfs_Ctx_impl(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiMulC_16u_C1RSfs(const Npp16u *pSrc, int nSrcStep, Npp16u nConstant, Npp16u *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiMulC_16u_C1RSfs_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiMulC_16s_C1RSfs_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s nConstant, Npp16s *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return nppiMulC_16s_C1RSfs_Ctx_impl(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiMulC_16s_C1RSfs(const Npp16s *pSrc, int nSrcStep, Npp16s nConstant, Npp16s *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiMulC_16s_C1RSfs_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiMulC_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f nConstant, Npp32f *pDst, int nDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiMulC_32f_C1R_Ctx_impl(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulC_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f nConstant, Npp32f *pDst, int nDstStep,
                           NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiMulC_32f_C1R_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// In-place versions
NppStatus nppiMulC_8u_C1IRSfs_Ctx(const Npp8u nConstant, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                  int nScaleFactor, NppStreamContext nppStreamCtx) {
  return nppiMulC_8u_C1RSfs_Ctx(pSrcDst, nSrcDstStep, nConstant, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor,
                                nppStreamCtx);
}

NppStatus nppiMulC_8u_C1IRSfs(const Npp8u nConstant, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiMulC_8u_C1IRSfs_Ctx(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiMulC_16u_C1IRSfs_Ctx(const Npp16u nConstant, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor, NppStreamContext nppStreamCtx) {
  return nppiMulC_16u_C1RSfs_Ctx(pSrcDst, nSrcDstStep, nConstant, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor,
                                 nppStreamCtx);
}

NppStatus nppiMulC_16u_C1IRSfs(const Npp16u nConstant, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiMulC_16u_C1IRSfs_Ctx(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiMulC_16s_C1IRSfs_Ctx(const Npp16s nConstant, Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor, NppStreamContext nppStreamCtx) {
  return nppiMulC_16s_C1RSfs_Ctx(pSrcDst, nSrcDstStep, nConstant, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor,
                                 nppStreamCtx);
}

NppStatus nppiMulC_16s_C1IRSfs(const Npp16s nConstant, Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiMulC_16s_C1IRSfs_Ctx(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiMulC_32f_C1IR_Ctx(const Npp32f nConstant, Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx) {
  return nppiMulC_32f_C1R_Ctx(pSrcDst, nSrcDstStep, nConstant, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulC_32f_C1IR(const Npp32f nConstant, Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiMulC_32f_C1IR_Ctx(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}