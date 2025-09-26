#include "nppi_arithmetic_common.h"

using namespace nppi::arithmetic;

// Template instantiations for Div operation
IMPLEMENT_BINARY_OP_IMPL(Div, DivOp, 8u, 1, Sfs)
IMPLEMENT_BINARY_OP_IMPL(Div, DivOp, 8u, 3, Sfs)
IMPLEMENT_BINARY_OP_IMPL_NO_SCALE(Div, DivOp, 32f, 1)
IMPLEMENT_BINARY_OP_IMPL_NO_SCALE(Div, DivOp, 32f, 3)

// Public API functions using unified framework
IMPLEMENT_BINARY_OP_API(Div, 8u, 1, Sfs, true)
IMPLEMENT_BINARY_OP_API(Div, 8u, 3, Sfs, true)
IMPLEMENT_BINARY_OP_API_NO_SCALE(Div, 32f, 1)
IMPLEMENT_BINARY_OP_API_NO_SCALE(Div, 32f, 3)

// Non-context versions
NppStatus nppiDiv_8u_C1RSfs(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                             Npp8u *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiDiv_8u_C1RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
                                  oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiDiv_8u_C3RSfs(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                             Npp8u *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiDiv_8u_C3RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
                                  oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiDiv_32f_C1R(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step,
                           Npp32f *pDst, int nDstStep, NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiDiv_32f_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
                                oSizeROI, nppStreamCtx);
}

NppStatus nppiDiv_32f_C3R(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step,
                           Npp32f *pDst, int nDstStep, NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiDiv_32f_C3R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
                                oSizeROI, nppStreamCtx);
}

// In-place versions
NppStatus nppiDiv_8u_C1IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
    return nppiDiv_8u_C1RSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep,
                                  oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiDiv_8u_C1IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, int nScaleFactor) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiDiv_8u_C1IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiDiv_32f_C1IR_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return nppiDiv_32f_C1R_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep,
                                oSizeROI, nppStreamCtx);
}

NppStatus nppiDiv_32f_C1IR(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep,
                            NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiDiv_32f_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}