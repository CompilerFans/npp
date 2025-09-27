#include "nppi_arithmetic_executor.h"
#include "nppi_arithmetic_ops.h"

using namespace nppi::arithmetic;

// Implementation functions using the unified executor
extern "C" {

NppStatus nppiAbsDiffC_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI, Npp8u nConstant, NppStreamContext nppStreamCtx) {
  AbsDiffConstOp<Npp8u> op(nConstant);
  return ConstOperationExecutor<Npp8u, 1, AbsDiffConstOp<Npp8u>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                                           nppStreamCtx.hStream, op);
}

NppStatus nppiAbsDiffC_16u_C1R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
                                        NppiSize oSizeROI, Npp16u nConstant, NppStreamContext nppStreamCtx) {
  AbsDiffConstOp<Npp16u> op(nConstant);
  return ConstOperationExecutor<Npp16u, 1, AbsDiffConstOp<Npp16u>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                                             nppStreamCtx.hStream, op);
}

NppStatus nppiAbsDiffC_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                        NppiSize oSizeROI, Npp32f nConstant, NppStreamContext nppStreamCtx) {
  AbsDiffConstOp<Npp32f> op(nConstant);
  return ConstOperationExecutor<Npp32f, 1, AbsDiffConstOp<Npp32f>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                                             nppStreamCtx.hStream, op);
}

} // extern "C"

// Public API functions
NppStatus nppiAbsDiffC_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                  NppiSize oSizeROI, Npp8u nConstant, NppStreamContext nppStreamCtx) {
  return nppiAbsDiffC_8u_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nConstant, nppStreamCtx);
}

NppStatus nppiAbsDiffC_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                              NppiSize oSizeROI, Npp8u nConstant) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiAbsDiffC_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nConstant, nppStreamCtx);
}

NppStatus nppiAbsDiffC_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
                                   NppiSize oSizeROI, Npp16u nConstant, NppStreamContext nppStreamCtx) {
  return nppiAbsDiffC_16u_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nConstant, nppStreamCtx);
}

NppStatus nppiAbsDiffC_16u_C1R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
                               NppiSize oSizeROI, Npp16u nConstant) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiAbsDiffC_16u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nConstant, nppStreamCtx);
}

NppStatus nppiAbsDiffC_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                   NppiSize oSizeROI, Npp32f nConstant, NppStreamContext nppStreamCtx) {
  return nppiAbsDiffC_32f_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nConstant, nppStreamCtx);
}

NppStatus nppiAbsDiffC_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                               NppiSize oSizeROI, Npp32f nConstant) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiAbsDiffC_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nConstant, nppStreamCtx);
}