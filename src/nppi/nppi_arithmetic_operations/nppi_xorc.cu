#include "nppi_arithmetic_executor.h"
#include "nppi_arithmetic_ops.h"

using namespace nppi::arithmetic;

// Implementation functions using the unified executor
extern "C" {

NppStatus nppiXorC_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u nConstant, Npp8u *pDst, int nDstStep,
                                   NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  XorConstOp<Npp8u> op(nConstant);
  return ConstOperationExecutor<Npp8u, 1, XorConstOp<Npp8u>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                                      nppStreamCtx.hStream, op);
}

NppStatus nppiXorC_8u_C3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[3], Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MultiConstOperationExecutor<Npp8u, 3, XorConstOp<Npp8u>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                                           nppStreamCtx.hStream, aConstants);
}

NppStatus nppiXorC_8u_AC4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[3], Npp8u *pDst,
                                    int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MultiConstOperationExecutor<Npp8u, 4, XorConstOp<Npp8u>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                                           nppStreamCtx.hStream, aConstants);
}

NppStatus nppiXorC_8u_C4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[4], Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MultiConstOperationExecutor<Npp8u, 4, XorConstOp<Npp8u>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                                           nppStreamCtx.hStream, aConstants);
}

NppStatus nppiXorC_16u_C1R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u nConstant, Npp16u *pDst, int nDstStep,
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  XorConstOp<Npp16u> op(nConstant);
  return ConstOperationExecutor<Npp16u, 1, XorConstOp<Npp16u>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                                        nppStreamCtx.hStream, op);
}

NppStatus nppiXorC_16u_C3R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[3], Npp16u *pDst,
                                    int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MultiConstOperationExecutor<Npp16u, 3, XorConstOp<Npp16u>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                                             0, nppStreamCtx.hStream, aConstants);
}

NppStatus nppiXorC_16u_AC4R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[3], Npp16u *pDst,
                                     int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MultiConstOperationExecutor<Npp16u, 4, XorConstOp<Npp16u>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                                             0, nppStreamCtx.hStream, aConstants);
}

NppStatus nppiXorC_16u_C4R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[4], Npp16u *pDst,
                                    int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MultiConstOperationExecutor<Npp16u, 4, XorConstOp<Npp16u>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                                             0, nppStreamCtx.hStream, aConstants);
}

NppStatus nppiXorC_32s_C1R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, Npp32s nConstant, Npp32s *pDst, int nDstStep,
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  XorConstOp<Npp32s> op(nConstant);
  return ConstOperationExecutor<Npp32s, 1, XorConstOp<Npp32s>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                                        nppStreamCtx.hStream, op);
}

NppStatus nppiXorC_32s_C3R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, const Npp32s aConstants[3], Npp32s *pDst,
                                    int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MultiConstOperationExecutor<Npp32s, 3, XorConstOp<Npp32s>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                                             0, nppStreamCtx.hStream, aConstants);
}

NppStatus nppiXorC_32s_AC4R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, const Npp32s aConstants[3], Npp32s *pDst,
                                     int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MultiConstOperationExecutor<Npp32s, 4, XorConstOp<Npp32s>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                                             0, nppStreamCtx.hStream, aConstants);
}

NppStatus nppiXorC_32s_C4R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, const Npp32s aConstants[4], Npp32s *pDst,
                                    int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MultiConstOperationExecutor<Npp32s, 4, XorConstOp<Npp32s>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                                             0, nppStreamCtx.hStream, aConstants);
}

} // extern "C"

// Public API functions - 8u variants
NppStatus nppiXorC_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u nConstant, Npp8u *pDst, int nDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiXorC_8u_C1R_Ctx_impl(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u nConstant, Npp8u *pDst, int nDstStep,
                          NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiXorC_8u_C1R_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_8u_C1IR_Ctx(Npp8u nConstant, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx) {
  return nppiXorC_8u_C1R_Ctx(pSrcDst, nSrcDstStep, nConstant, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_8u_C1IR(Npp8u nConstant, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiXorC_8u_C1IR_Ctx(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiXorC_8u_C3R_Ctx_impl(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_8u_C3R(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                          NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiXorC_8u_C3R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_8u_C3IR_Ctx(const Npp8u aConstants[3], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx) {
  return nppiXorC_8u_C3R_Ctx(pSrcDst, nSrcDstStep, aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_8u_C3IR(const Npp8u aConstants[3], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiXorC_8u_C3IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiXorC_8u_AC4R_Ctx_impl(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_8u_AC4R(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                           NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiXorC_8u_AC4R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_8u_AC4IR_Ctx(const Npp8u aConstants[3], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx) {
  return nppiXorC_8u_AC4R_Ctx(pSrcDst, nSrcDstStep, aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_8u_AC4IR(const Npp8u aConstants[3], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiXorC_8u_AC4IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[4], Npp8u *pDst, int nDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiXorC_8u_C4R_Ctx_impl(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_8u_C4R(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[4], Npp8u *pDst, int nDstStep,
                          NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiXorC_8u_C4R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_8u_C4IR_Ctx(const Npp8u aConstants[4], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx) {
  return nppiXorC_8u_C4R_Ctx(pSrcDst, nSrcDstStep, aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_8u_C4IR(const Npp8u aConstants[4], Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiXorC_8u_C4IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

// 16u variants
NppStatus nppiXorC_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u nConstant, Npp16u *pDst, int nDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiXorC_16u_C1R_Ctx_impl(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_16u_C1R(const Npp16u *pSrc, int nSrcStep, Npp16u nConstant, Npp16u *pDst, int nDstStep,
                           NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiXorC_16u_C1R_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_16u_C1IR_Ctx(Npp16u nConstant, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx) {
  return nppiXorC_16u_C1R_Ctx(pSrcDst, nSrcDstStep, nConstant, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_16u_C1IR(Npp16u nConstant, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiXorC_16u_C1IR_Ctx(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_16u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[3], Npp16u *pDst, int nDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiXorC_16u_C3R_Ctx_impl(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_16u_C3R(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[3], Npp16u *pDst, int nDstStep,
                           NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiXorC_16u_C3R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_16u_C3IR_Ctx(const Npp16u aConstants[3], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx) {
  return nppiXorC_16u_C3R_Ctx(pSrcDst, nSrcDstStep, aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_16u_C3IR(const Npp16u aConstants[3], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiXorC_16u_C3IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[3], Npp16u *pDst,
                                int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiXorC_16u_AC4R_Ctx_impl(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_16u_AC4R(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[3], Npp16u *pDst, int nDstStep,
                            NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiXorC_16u_AC4R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_16u_AC4IR_Ctx(const Npp16u aConstants[3], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx) {
  return nppiXorC_16u_AC4R_Ctx(pSrcDst, nSrcDstStep, aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_16u_AC4IR(const Npp16u aConstants[3], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiXorC_16u_AC4IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_16u_C4R_Ctx(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[4], Npp16u *pDst, int nDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiXorC_16u_C4R_Ctx_impl(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_16u_C4R(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[4], Npp16u *pDst, int nDstStep,
                           NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiXorC_16u_C4R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_16u_C4IR_Ctx(const Npp16u aConstants[4], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx) {
  return nppiXorC_16u_C4R_Ctx(pSrcDst, nSrcDstStep, aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_16u_C4IR(const Npp16u aConstants[4], Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiXorC_16u_C4IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

// 32s variants
NppStatus nppiXorC_32s_C1R_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s nConstant, Npp32s *pDst, int nDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiXorC_32s_C1R_Ctx_impl(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_32s_C1R(const Npp32s *pSrc, int nSrcStep, Npp32s nConstant, Npp32s *pDst, int nDstStep,
                           NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiXorC_32s_C1R_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_32s_C1IR_Ctx(Npp32s nConstant, Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx) {
  return nppiXorC_32s_C1R_Ctx(pSrcDst, nSrcDstStep, nConstant, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_32s_C1IR(Npp32s nConstant, Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiXorC_32s_C1IR_Ctx(nConstant, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_32s_C3R_Ctx(const Npp32s *pSrc, int nSrcStep, const Npp32s aConstants[3], Npp32s *pDst, int nDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiXorC_32s_C3R_Ctx_impl(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_32s_C3R(const Npp32s *pSrc, int nSrcStep, const Npp32s aConstants[3], Npp32s *pDst, int nDstStep,
                           NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiXorC_32s_C3R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_32s_C3IR_Ctx(const Npp32s aConstants[3], Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx) {
  return nppiXorC_32s_C3R_Ctx(pSrcDst, nSrcDstStep, aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_32s_C3IR(const Npp32s aConstants[3], Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiXorC_32s_C3IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_32s_AC4R_Ctx(const Npp32s *pSrc, int nSrcStep, const Npp32s aConstants[3], Npp32s *pDst,
                                int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiXorC_32s_AC4R_Ctx_impl(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_32s_AC4R(const Npp32s *pSrc, int nSrcStep, const Npp32s aConstants[3], Npp32s *pDst, int nDstStep,
                            NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiXorC_32s_AC4R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_32s_AC4IR_Ctx(const Npp32s aConstants[3], Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx) {
  return nppiXorC_32s_AC4R_Ctx(pSrcDst, nSrcDstStep, aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_32s_AC4IR(const Npp32s aConstants[3], Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiXorC_32s_AC4IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_32s_C4R_Ctx(const Npp32s *pSrc, int nSrcStep, const Npp32s aConstants[4], Npp32s *pDst, int nDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiXorC_32s_C4R_Ctx_impl(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_32s_C4R(const Npp32s *pSrc, int nSrcStep, const Npp32s aConstants[4], Npp32s *pDst, int nDstStep,
                           NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiXorC_32s_C4R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_32s_C4IR_Ctx(const Npp32s aConstants[4], Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx) {
  return nppiXorC_32s_C4R_Ctx(pSrcDst, nSrcDstStep, aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiXorC_32s_C4IR(const Npp32s aConstants[4], Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiXorC_32s_C4IR_Ctx(aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}