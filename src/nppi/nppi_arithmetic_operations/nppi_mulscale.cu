#include "nppi_arithmetic_ops.h"
#include "nppi_arithmetic_executor.h"

using namespace nppi::arithmetic;

// Implementation functions using the unified executor
extern "C" {

NppStatus nppiMulScale_8u_C1R_Ctx_impl(const Npp8u *pSrc1, int nSrc1Step, 
                                        const Npp8u *pSrc2, int nSrc2Step,
                                        Npp8u *pDst, int nDstStep, 
                                        NppiSize oSizeROI, 
                                        NppStreamContext nppStreamCtx) {
    return BinaryOperationExecutor<Npp8u, 1, MulScaleOp<Npp8u>>::execute(
        pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
        oSizeROI, 0, nppStreamCtx.hStream);
}

NppStatus nppiMulScale_16u_C1R_Ctx_impl(const Npp16u *pSrc1, int nSrc1Step, 
                                         const Npp16u *pSrc2, int nSrc2Step,
                                         Npp16u *pDst, int nDstStep, 
                                         NppiSize oSizeROI, 
                                         NppStreamContext nppStreamCtx) {
    return BinaryOperationExecutor<Npp16u, 1, MulScaleOp<Npp16u>>::execute(
        pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
        oSizeROI, 0, nppStreamCtx.hStream);
}

} // extern "C"

// Public API functions
NppStatus nppiMulScale_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step, 
                                   const Npp8u *pSrc2, int nSrc2Step,
                                   Npp8u *pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   NppStreamContext nppStreamCtx) {
    return nppiMulScale_8u_C1R_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                                         pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulScale_8u_C1R(const Npp8u *pSrc1, int nSrc1Step, 
                              const Npp8u *pSrc2, int nSrc2Step,
                              Npp8u *pDst, int nDstStep, 
                              NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiMulScale_8u_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                                    pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulScale_8u_C1IR_Ctx(const Npp8u *pSrc, int nSrcStep, 
                                    Npp8u *pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    NppStreamContext nppStreamCtx) {
    return nppiMulScale_8u_C1R_Ctx_impl(pSrc, nSrcStep, pSrcDst, nSrcDstStep, 
                                         pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulScale_8u_C1IR(const Npp8u *pSrc, int nSrcStep, 
                               Npp8u *pSrcDst, int nSrcDstStep, 
                               NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiMulScale_8u_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, 
                                     oSizeROI, nppStreamCtx);
}

NppStatus nppiMulScale_16u_C1R_Ctx(const Npp16u *pSrc1, int nSrc1Step, 
                                    const Npp16u *pSrc2, int nSrc2Step,
                                    Npp16u *pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    NppStreamContext nppStreamCtx) {
    return nppiMulScale_16u_C1R_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                                          pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulScale_16u_C1R(const Npp16u *pSrc1, int nSrc1Step, 
                               const Npp16u *pSrc2, int nSrc2Step,
                               Npp16u *pDst, int nDstStep, 
                               NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiMulScale_16u_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                                     pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulScale_16u_C1IR_Ctx(const Npp16u *pSrc, int nSrcStep, 
                                     Npp16u *pSrcDst, int nSrcDstStep, 
                                     NppiSize oSizeROI, 
                                     NppStreamContext nppStreamCtx) {
    return nppiMulScale_16u_C1R_Ctx_impl(pSrc, nSrcStep, pSrcDst, nSrcDstStep, 
                                          pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulScale_16u_C1IR(const Npp16u *pSrc, int nSrcStep, 
                                Npp16u *pSrcDst, int nSrcDstStep, 
                                NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiMulScale_16u_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, 
                                      oSizeROI, nppStreamCtx);
}
