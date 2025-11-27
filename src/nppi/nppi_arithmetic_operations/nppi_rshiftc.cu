#include "nppi_arithmetic_executor.h"
#include "nppi_arithmetic_ops.h"

using namespace nppi::arithmetic;

// Implementation functions using the unified executor
extern "C" {

NppStatus nppiRShiftC_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp32u nConstant, Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  RShiftConstOp<Npp8u> op(nConstant);
  return ConstOperationExecutor<Npp8u, 1, RShiftConstOp<Npp8u>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                                         nppStreamCtx.hStream, op);
}

NppStatus nppiRShiftC_8u_C3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, const Npp32u aConstants[3], Npp8u *pDst,
                                      int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  RShiftConstOp<Npp8u> op(aConstants[0]);
  return ConstOperationExecutor<Npp8u, 3, RShiftConstOp<Npp8u>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                                         nppStreamCtx.hStream, op);
}

NppStatus nppiRShiftC_8u_AC4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, const Npp32u aConstants[3], Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  RShiftConstOp<Npp8u> op(aConstants[0]);
  return ConstOperationExecutor<Npp8u, 4, RShiftConstOp<Npp8u>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                                         nppStreamCtx.hStream, op);
}

NppStatus nppiRShiftC_8u_C4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, const Npp32u aConstants[4], Npp8u *pDst,
                                      int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  RShiftConstOp<Npp8u> op(aConstants[0]);
  return ConstOperationExecutor<Npp8u, 4, RShiftConstOp<Npp8u>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                                         nppStreamCtx.hStream, op);
}

NppStatus nppiRShiftC_16u_C1R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp32u nConstant, Npp16u *pDst, int nDstStep,
                                       NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  RShiftConstOp<Npp16u> op(nConstant);
  return ConstOperationExecutor<Npp16u, 1, RShiftConstOp<Npp16u>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                                           nppStreamCtx.hStream, op);
}

NppStatus nppiRShiftC_16u_C3R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, const Npp32u aConstants[3], Npp16u *pDst,
                                       int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  RShiftConstOp<Npp16u> op(aConstants[0]);
  return ConstOperationExecutor<Npp16u, 3, RShiftConstOp<Npp16u>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                                           nppStreamCtx.hStream, op);
}

NppStatus nppiRShiftC_16u_AC4R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, const Npp32u aConstants[3], Npp16u *pDst,
                                        int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  RShiftConstOp<Npp16u> op(aConstants[0]);
  return ConstOperationExecutor<Npp16u, 4, RShiftConstOp<Npp16u>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                                           nppStreamCtx.hStream, op);
}

NppStatus nppiRShiftC_16u_C4R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, const Npp32u aConstants[4], Npp16u *pDst,
                                       int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  RShiftConstOp<Npp16u> op(aConstants[0]);
  return ConstOperationExecutor<Npp16u, 4, RShiftConstOp<Npp16u>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                                           nppStreamCtx.hStream, op);
}

NppStatus nppiRShiftC_16s_C1R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp32u nConstant, Npp16s *pDst, int nDstStep,
                                       NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  RShiftConstOp<Npp16s> op(nConstant);
  return ConstOperationExecutor<Npp16s, 1, RShiftConstOp<Npp16s>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                                           nppStreamCtx.hStream, op);
}

NppStatus nppiRShiftC_16s_C3R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, const Npp32u aConstants[3], Npp16s *pDst,
                                       int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  RShiftConstOp<Npp16s> op(aConstants[0]);
  return ConstOperationExecutor<Npp16s, 3, RShiftConstOp<Npp16s>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                                           nppStreamCtx.hStream, op);
}

NppStatus nppiRShiftC_16s_AC4R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, const Npp32u aConstants[3], Npp16s *pDst,
                                        int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  RShiftConstOp<Npp16s> op(aConstants[0]);
  return ConstOperationExecutor<Npp16s, 4, RShiftConstOp<Npp16s>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                                           nppStreamCtx.hStream, op);
}

NppStatus nppiRShiftC_16s_C4R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, const Npp32u aConstants[4], Npp16s *pDst,
                                       int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  RShiftConstOp<Npp16s> op(aConstants[0]);
  return ConstOperationExecutor<Npp16s, 4, RShiftConstOp<Npp16s>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                                           nppStreamCtx.hStream, op);
}

NppStatus nppiRShiftC_32s_C1R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, Npp32u nConstant, Npp32s *pDst, int nDstStep,
                                       NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  RShiftConstOp<Npp32s> op(nConstant);
  return ConstOperationExecutor<Npp32s, 1, RShiftConstOp<Npp32s>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                                           nppStreamCtx.hStream, op);
}

NppStatus nppiRShiftC_32s_C3R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, const Npp32u aConstants[3], Npp32s *pDst,
                                       int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  RShiftConstOp<Npp32s> op(aConstants[0]);
  return ConstOperationExecutor<Npp32s, 3, RShiftConstOp<Npp32s>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                                           nppStreamCtx.hStream, op);
}

NppStatus nppiRShiftC_32s_AC4R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, const Npp32u aConstants[3], Npp32s *pDst,
                                        int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  RShiftConstOp<Npp32s> op(aConstants[0]);
  return ConstOperationExecutor<Npp32s, 4, RShiftConstOp<Npp32s>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                                           nppStreamCtx.hStream, op);
}

NppStatus nppiRShiftC_32s_C4R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, const Npp32u aConstants[4], Npp32s *pDst,
                                       int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  RShiftConstOp<Npp32s> op(aConstants[0]);
  return ConstOperationExecutor<Npp32s, 4, RShiftConstOp<Npp32s>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                                           nppStreamCtx.hStream, op);
}

} // extern "C"

// Public API functions - 8u variants
NppStatus nppiRShiftC_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp32u nConstant, Npp8u *pDst, int nDstStep,
                                 NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  if (nConstant > 31) // Validate shift count
    return NPP_BAD_ARGUMENT_ERROR;
  return nppiRShiftC_8u_C1R_Ctx_impl(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiRShiftC_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp32u nConstant, Npp8u *pDst, int nDstStep,
                             NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiRShiftC_8u_C1R_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiRShiftC_8u_C1IR_Ctx(Npp32u nConstant, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
  return nppiRShiftC_8u_C1R_Ctx(pSrcDst, nSrcDstStep, nConstant, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiRShiftC_8u_C1IR(Npp32u nConstant, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiRShiftC_8u_C1IR_Ctx(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

// 16u variants
NppStatus nppiRShiftC_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp32u nConstant, Npp16u *pDst, int nDstStep,
                                  NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  if (nConstant > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return nppiRShiftC_16u_C1R_Ctx_impl(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiRShiftC_16u_C1R(const Npp16u *pSrc, int nSrcStep, Npp32u nConstant, Npp16u *pDst, int nDstStep,
                              NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiRShiftC_16u_C1R_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// 16s variants
NppStatus nppiRShiftC_16s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp32u nConstant, Npp16s *pDst, int nDstStep,
                                  NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  if (nConstant > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return nppiRShiftC_16s_C1R_Ctx_impl(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiRShiftC_16s_C1R(const Npp16s *pSrc, int nSrcStep, Npp32u nConstant, Npp16s *pDst, int nDstStep,
                              NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiRShiftC_16s_C1R_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// 32s variants
NppStatus nppiRShiftC_32s_C1R_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32u nConstant, Npp32s *pDst, int nDstStep,
                                  NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  if (nConstant > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return nppiRShiftC_32s_C1R_Ctx_impl(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiRShiftC_32s_C1R(const Npp32s *pSrc, int nSrcStep, Npp32u nConstant, Npp32s *pDst, int nDstStep,
                              NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiRShiftC_32s_C1R_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nppStreamCtx);
}