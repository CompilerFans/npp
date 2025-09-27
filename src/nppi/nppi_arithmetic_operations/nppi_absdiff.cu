#include "nppi_arithmetic_common.h"

using namespace nppi::arithmetic;

// Template instantiations for AbsDiff operations
IMPLEMENT_BINARY_OP_IMPL_NO_SCALE(AbsDiff, AbsDiffOp, 8u, 1)
IMPLEMENT_BINARY_OP_IMPL_NO_SCALE(AbsDiff, AbsDiffOp, 8u, 3) 
IMPLEMENT_BINARY_OP_IMPL_NO_SCALE(AbsDiff, AbsDiffOp, 32f, 1)

// Public API functions using unified framework
IMPLEMENT_BINARY_OP_API_NO_SCALE(AbsDiff, 8u, 1)
IMPLEMENT_BINARY_OP_API_NO_SCALE(AbsDiff, 8u, 3)
IMPLEMENT_BINARY_OP_API_NO_SCALE(AbsDiff, 32f, 1)

// Non-context versions
NppStatus nppiAbsDiff_8u_C1R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                             Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAbsDiff_8u_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
                                  oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiff_8u_C3R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                             Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAbsDiff_8u_C3R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
                                  oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiff_32f_C1R(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step,
                              Npp32f *pDst, int nDstStep, NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAbsDiff_32f_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
                                   oSizeROI, nppStreamCtx);
}

// In-place versions
NppStatus nppiAbsDiff_8u_C1IR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return nppiAbsDiff_8u_C1R_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep,
                                  oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiff_8u_C1IR(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAbsDiff_8u_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiff_8u_C3IR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return nppiAbsDiff_8u_C3R_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep,
                                  oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiff_8u_C3IR(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAbsDiff_8u_C3IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiff_32f_C1IR_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return nppiAbsDiff_32f_C1R_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep,
                                   oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiff_32f_C1IR(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAbsDiff_32f_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}