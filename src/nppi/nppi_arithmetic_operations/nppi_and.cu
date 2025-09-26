#include "nppi_arithmetic_common.h"

using namespace nppi::arithmetic;

// Template instantiations for And operation
IMPLEMENT_BINARY_OP_IMPL(And, AndOp, 8u, 1, Sfs)
IMPLEMENT_BINARY_OP_IMPL(And, AndOp, 8u, 3, Sfs)

// Public API functions using unified framework
IMPLEMENT_BINARY_OP_API(And, 8u, 1, Sfs, false)
IMPLEMENT_BINARY_OP_API(And, 8u, 3, Sfs, false)

// Non-context versions
NppStatus nppiAnd_8u_C1R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                         Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAnd_8u_C1RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
                                 oSizeROI, 0, nppStreamCtx);
}

NppStatus nppiAnd_8u_C3R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                         Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAnd_8u_C3RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
                                 oSizeROI, 0, nppStreamCtx);
}

NppStatus nppiAnd_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                             Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return nppiAnd_8u_C1RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
                                 oSizeROI, 0, nppStreamCtx);
}

NppStatus nppiAnd_8u_C3R_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                             Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return nppiAnd_8u_C3RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
                                 oSizeROI, 0, nppStreamCtx);
}

// In-place versions  
NppStatus nppiAnd_8u_C1IR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return nppiAnd_8u_C1R_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep,
                              oSizeROI, nppStreamCtx);
}

NppStatus nppiAnd_8u_C1IR(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep,
                          NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAnd_8u_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAnd_8u_C3IR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return nppiAnd_8u_C3R_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep,
                              oSizeROI, nppStreamCtx);
}

NppStatus nppiAnd_8u_C3IR(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep,
                          NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAnd_8u_C3IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}