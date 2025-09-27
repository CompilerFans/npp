#include "nppi_arithmetic_ops.h"
#include "nppi_arithmetic_executor.h"

using namespace nppi::arithmetic;

// Implementation functions using the unified executor
extern "C" {

// 8u C1 operations
NppStatus nppiAbsDiff_8u_C1R_Ctx_impl(const Npp8u *pSrc1, int nSrc1Step, 
                                       const Npp8u *pSrc2, int nSrc2Step,
                                       Npp8u *pDst, int nDstStep, 
                                       NppiSize oSizeROI, 
                                       NppStreamContext nppStreamCtx) {
    return BinaryOperationExecutor<Npp8u, 1, AbsDiffOp<Npp8u>>::execute(
        pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
        oSizeROI, 0, nppStreamCtx.hStream);
}

// 8u C3 operations
NppStatus nppiAbsDiff_8u_C3R_Ctx_impl(const Npp8u *pSrc1, int nSrc1Step, 
                                       const Npp8u *pSrc2, int nSrc2Step,
                                       Npp8u *pDst, int nDstStep, 
                                       NppiSize oSizeROI, 
                                       NppStreamContext nppStreamCtx) {
    return BinaryOperationExecutor<Npp8u, 3, AbsDiffOp<Npp8u>>::execute(
        pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
        oSizeROI, 0, nppStreamCtx.hStream);
}

// 16u C1 operations
NppStatus nppiAbsDiff_16u_C1R_Ctx_impl(const Npp16u *pSrc1, int nSrc1Step, 
                                        const Npp16u *pSrc2, int nSrc2Step,
                                        Npp16u *pDst, int nDstStep, 
                                        NppiSize oSizeROI, 
                                        NppStreamContext nppStreamCtx) {
    return BinaryOperationExecutor<Npp16u, 1, AbsDiffOp<Npp16u>>::execute(
        pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
        oSizeROI, 0, nppStreamCtx.hStream);
}

// 32f C1 operations
NppStatus nppiAbsDiff_32f_C1R_Ctx_impl(const Npp32f *pSrc1, int nSrc1Step, 
                                        const Npp32f *pSrc2, int nSrc2Step,
                                        Npp32f *pDst, int nDstStep, 
                                        NppiSize oSizeROI, 
                                        NppStreamContext nppStreamCtx) {
    return BinaryOperationExecutor<Npp32f, 1, AbsDiffOp<Npp32f>>::execute(
        pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
        oSizeROI, 0, nppStreamCtx.hStream);
}

} // extern "C"

// Public API functions
NppStatus nppiAbsDiff_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step, 
                                  const Npp8u *pSrc2, int nSrc2Step,
                                  Npp8u *pDst, int nDstStep, 
                                  NppiSize oSizeROI, 
                                  NppStreamContext nppStreamCtx) {
    return nppiAbsDiff_8u_C1R_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                                        pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiff_8u_C1R(const Npp8u *pSrc1, int nSrc1Step, 
                             const Npp8u *pSrc2, int nSrc2Step,
                             Npp8u *pDst, int nDstStep, 
                             NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAbsDiff_8u_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                                  pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiff_8u_C3R_Ctx(const Npp8u *pSrc1, int nSrc1Step, 
                                  const Npp8u *pSrc2, int nSrc2Step,
                                  Npp8u *pDst, int nDstStep, 
                                  NppiSize oSizeROI, 
                                  NppStreamContext nppStreamCtx) {
    return nppiAbsDiff_8u_C3R_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                                        pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiff_8u_C3R(const Npp8u *pSrc1, int nSrc1Step, 
                             const Npp8u *pSrc2, int nSrc2Step,
                             Npp8u *pDst, int nDstStep, 
                             NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAbsDiff_8u_C3R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                                  pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiff_16u_C1R_Ctx(const Npp16u *pSrc1, int nSrc1Step, 
                                   const Npp16u *pSrc2, int nSrc2Step,
                                   Npp16u *pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   NppStreamContext nppStreamCtx) {
    return nppiAbsDiff_16u_C1R_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                                         pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiff_16u_C1R(const Npp16u *pSrc1, int nSrc1Step, 
                              const Npp16u *pSrc2, int nSrc2Step,
                              Npp16u *pDst, int nDstStep, 
                              NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAbsDiff_16u_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                                   pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiff_32f_C1R_Ctx(const Npp32f *pSrc1, int nSrc1Step, 
                                   const Npp32f *pSrc2, int nSrc2Step,
                                   Npp32f *pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   NppStreamContext nppStreamCtx) {
    return nppiAbsDiff_32f_C1R_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                                         pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiff_32f_C1R(const Npp32f *pSrc1, int nSrc1Step, 
                              const Npp32f *pSrc2, int nSrc2Step,
                              Npp32f *pDst, int nDstStep, 
                              NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAbsDiff_32f_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                                   pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// In-place versions
NppStatus nppiAbsDiff_8u_C1IR_Ctx(const Npp8u *pSrc, int nSrcStep, 
                                  Npp8u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return nppiAbsDiff_8u_C1R_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, 
                                  pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiff_8u_C1IR(const Npp8u *pSrc, int nSrcStep, 
                              Npp8u *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAbsDiff_8u_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiff_8u_C3IR_Ctx(const Npp8u *pSrc, int nSrcStep, 
                                  Npp8u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return nppiAbsDiff_8u_C3R_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, 
                                  pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiff_8u_C3IR(const Npp8u *pSrc, int nSrcStep, 
                              Npp8u *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAbsDiff_8u_C3IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiff_32f_C1IR_Ctx(const Npp32f *pSrc, int nSrcStep, 
                                   Npp32f *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return nppiAbsDiff_32f_C1R_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, 
                                   pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiff_32f_C1IR(const Npp32f *pSrc, int nSrcStep, 
                               Npp32f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAbsDiff_32f_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}