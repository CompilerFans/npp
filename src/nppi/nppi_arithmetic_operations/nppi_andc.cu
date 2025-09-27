#include "nppi_arithmetic_unified.h"

using namespace nppi::arithmetic;

// Implementation functions using the unified executor
extern "C" {

NppStatus nppiAndC_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, 
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI, Npp8u nConstant, 
                                    NppStreamContext nppStreamCtx) {
    AndConstOp<Npp8u> op(nConstant);
    return ConstOperationExecutor<Npp8u, 1, AndConstOp<Npp8u>>::execute(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0, nppStreamCtx.hStream, op);
}

NppStatus nppiAndC_8u_C3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, 
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp8u aConstants[3], 
                                    NppStreamContext nppStreamCtx) {
    AndConstOp<Npp8u> op(aConstants[0]);
    return ConstOperationExecutor<Npp8u, 3, AndConstOp<Npp8u>>::execute(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0, nppStreamCtx.hStream, op);
}

NppStatus nppiAndC_16u_C3R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, 
                                     Npp16u *pDst, int nDstStep,
                                     NppiSize oSizeROI, const Npp16u aConstants[3], 
                                     NppStreamContext nppStreamCtx) {
    AndConstOp<Npp16u> op(aConstants[0]);
    return ConstOperationExecutor<Npp16u, 3, AndConstOp<Npp16u>>::execute(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0, nppStreamCtx.hStream, op);
}

NppStatus nppiAndC_16u_C4R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, 
                                     Npp16u *pDst, int nDstStep,
                                     NppiSize oSizeROI, const Npp16u aConstants[4], 
                                     NppStreamContext nppStreamCtx) {
    AndConstOp<Npp16u> op(aConstants[0]);
    return ConstOperationExecutor<Npp16u, 4, AndConstOp<Npp16u>>::execute(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0, nppStreamCtx.hStream, op);
}

NppStatus nppiAndC_32s_C3R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, 
                                     Npp32s *pDst, int nDstStep,
                                     NppiSize oSizeROI, const Npp32s aConstants[3], 
                                     NppStreamContext nppStreamCtx) {
    AndConstOp<Npp32s> op(aConstants[0]);
    return ConstOperationExecutor<Npp32s, 3, AndConstOp<Npp32s>>::execute(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0, nppStreamCtx.hStream, op);
}

NppStatus nppiAndC_32s_C4R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, 
                                     Npp32s *pDst, int nDstStep,
                                     NppiSize oSizeROI, const Npp32s aConstants[4], 
                                     NppStreamContext nppStreamCtx) {
    AndConstOp<Npp32s> op(aConstants[0]);
    return ConstOperationExecutor<Npp32s, 4, AndConstOp<Npp32s>>::execute(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0, nppStreamCtx.hStream, op);
}

} // extern "C"

// Public API functions
NppStatus nppiAndC_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, 
                               Npp8u *pDst, int nDstStep,
                               NppiSize oSizeROI, Npp8u nConstant, 
                               NppStreamContext nppStreamCtx) {
    return nppiAndC_8u_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, 
                                     oSizeROI, nConstant, nppStreamCtx);
}

NppStatus nppiAndC_8u_C1R(const Npp8u *pSrc, int nSrcStep, 
                          Npp8u *pDst, int nDstStep,
                          NppiSize oSizeROI, Npp8u nConstant) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAndC_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, 
                               oSizeROI, nConstant, nppStreamCtx);
}

NppStatus nppiAndC_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, 
                               Npp8u *pDst, int nDstStep,
                               NppiSize oSizeROI, const Npp8u aConstants[3], 
                               NppStreamContext nppStreamCtx) {
    return nppiAndC_8u_C3R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, 
                                     oSizeROI, aConstants, nppStreamCtx);
}

NppStatus nppiAndC_8u_C3R(const Npp8u *pSrc, int nSrcStep, 
                          Npp8u *pDst, int nDstStep,
                          NppiSize oSizeROI, const Npp8u aConstants[3]) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAndC_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, 
                               oSizeROI, aConstants, nppStreamCtx);
}

NppStatus nppiAndC_16u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep, 
                                Npp16u *pDst, int nDstStep,
                                NppiSize oSizeROI, const Npp16u aConstants[3], 
                                NppStreamContext nppStreamCtx) {
    return nppiAndC_16u_C3R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, 
                                      oSizeROI, aConstants, nppStreamCtx);
}

NppStatus nppiAndC_16u_C3R(const Npp16u *pSrc, int nSrcStep, 
                           Npp16u *pDst, int nDstStep,
                           NppiSize oSizeROI, const Npp16u aConstants[3]) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAndC_16u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, 
                                oSizeROI, aConstants, nppStreamCtx);
}

NppStatus nppiAndC_16u_C4R_Ctx(const Npp16u *pSrc, int nSrcStep, 
                                Npp16u *pDst, int nDstStep,
                                NppiSize oSizeROI, const Npp16u aConstants[4], 
                                NppStreamContext nppStreamCtx) {
    return nppiAndC_16u_C4R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, 
                                      oSizeROI, aConstants, nppStreamCtx);
}

NppStatus nppiAndC_16u_C4R(const Npp16u *pSrc, int nSrcStep, 
                           Npp16u *pDst, int nDstStep,
                           NppiSize oSizeROI, const Npp16u aConstants[4]) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAndC_16u_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, 
                                oSizeROI, aConstants, nppStreamCtx);
}

NppStatus nppiAndC_32s_C3R_Ctx(const Npp32s *pSrc, int nSrcStep, 
                                Npp32s *pDst, int nDstStep,
                                NppiSize oSizeROI, const Npp32s aConstants[3], 
                                NppStreamContext nppStreamCtx) {
    return nppiAndC_32s_C3R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, 
                                      oSizeROI, aConstants, nppStreamCtx);
}

NppStatus nppiAndC_32s_C3R(const Npp32s *pSrc, int nSrcStep, 
                           Npp32s *pDst, int nDstStep,
                           NppiSize oSizeROI, const Npp32s aConstants[3]) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAndC_32s_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, 
                                oSizeROI, aConstants, nppStreamCtx);
}

NppStatus nppiAndC_32s_C4R_Ctx(const Npp32s *pSrc, int nSrcStep, 
                                Npp32s *pDst, int nDstStep,
                                NppiSize oSizeROI, const Npp32s aConstants[4], 
                                NppStreamContext nppStreamCtx) {
    return nppiAndC_32s_C4R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, 
                                      oSizeROI, aConstants, nppStreamCtx);
}

NppStatus nppiAndC_32s_C4R(const Npp32s *pSrc, int nSrcStep, 
                           Npp32s *pDst, int nDstStep,
                           NppiSize oSizeROI, const Npp32s aConstants[4]) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAndC_32s_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, 
                                oSizeROI, aConstants, nppStreamCtx);
}

