#include "nppi_arithmetic_ops.h"
#include "nppi_arithmetic_executor.h"

using namespace nppi::arithmetic;

// Implementation functions using the unified executor
extern "C" {

// 8u C1 operations with scale factor
NppStatus nppiDivC_8u_C1RSfs_Ctx_impl(const Npp8u *pSrc, int nSrcStep, 
                                       Npp8u nConstant, Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI, int nScaleFactor, 
                                       NppStreamContext nppStreamCtx) {
    DivConstOp<Npp8u> op(nConstant);
    return ConstOperationExecutor<Npp8u, 1, DivConstOp<Npp8u>>::execute(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx.hStream, op);
}

// 8u C3 operations with scale factor
NppStatus nppiDivC_8u_C3RSfs_Ctx_impl(const Npp8u *pSrc, int nSrcStep, 
                                       const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI, int nScaleFactor, 
                                       NppStreamContext nppStreamCtx) {
    DivConstOp<Npp8u> op(aConstants[0]);
    return ConstOperationExecutor<Npp8u, 3, DivConstOp<Npp8u>>::execute(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx.hStream, op);
}

// 16u C1 operations with scale factor
NppStatus nppiDivC_16u_C1RSfs_Ctx_impl(const Npp16u *pSrc, int nSrcStep, 
                                        Npp16u nConstant, Npp16u *pDst, int nDstStep,
                                        NppiSize oSizeROI, int nScaleFactor, 
                                        NppStreamContext nppStreamCtx) {
    DivConstOp<Npp16u> op(nConstant);
    return ConstOperationExecutor<Npp16u, 1, DivConstOp<Npp16u>>::execute(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx.hStream, op);
}

// 16s C1 operations with scale factor
NppStatus nppiDivC_16s_C1RSfs_Ctx_impl(const Npp16s *pSrc, int nSrcStep, 
                                        Npp16s nConstant, Npp16s *pDst, int nDstStep,
                                        NppiSize oSizeROI, int nScaleFactor, 
                                        NppStreamContext nppStreamCtx) {
    DivConstOp<Npp16s> op(nConstant);
    return ConstOperationExecutor<Npp16s, 1, DivConstOp<Npp16s>>::execute(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx.hStream, op);
}

// 32f C1 operations (no scale factor)
NppStatus nppiDivC_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, 
                                     Npp32f nConstant, Npp32f *pDst, int nDstStep,
                                     NppiSize oSizeROI, 
                                     NppStreamContext nppStreamCtx) {
    DivConstOp<Npp32f> op(nConstant);
    return ConstOperationExecutor<Npp32f, 1, DivConstOp<Npp32f>>::execute(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0, nppStreamCtx.hStream, op);
}

} // extern "C"

// Public API functions
NppStatus nppiDivC_8u_C1RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, 
                                  Npp8u nConstant, Npp8u *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor, 
                                  NppStreamContext nppStreamCtx) {
    if (nScaleFactor < 0 || nScaleFactor > 31) return NPP_BAD_ARGUMENT_ERROR;
    if (nConstant == 0) return NPP_DIVIDE_BY_ZERO_ERROR;
    return nppiDivC_8u_C1RSfs_Ctx_impl(pSrc, nSrcStep, nConstant, pDst, nDstStep, 
                                        oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiDivC_8u_C1RSfs(const Npp8u *pSrc, int nSrcStep, 
                             Npp8u nConstant, Npp8u *pDst, int nDstStep,
                             NppiSize oSizeROI, int nScaleFactor) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiDivC_8u_C1RSfs_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, 
                                  oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiDivC_8u_C3RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, 
                                  const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor, 
                                  NppStreamContext nppStreamCtx) {
    if (nScaleFactor < 0 || nScaleFactor > 31) return NPP_BAD_ARGUMENT_ERROR;
    if (!aConstants || aConstants[0] == 0 || aConstants[1] == 0 || aConstants[2] == 0) {
        return NPP_DIVISOR_ERROR;
    }
    return nppiDivC_8u_C3RSfs_Ctx_impl(pSrc, nSrcStep, aConstants, pDst, nDstStep, 
                                        oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiDivC_8u_C3RSfs(const Npp8u *pSrc, int nSrcStep, 
                             const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                             NppiSize oSizeROI, int nScaleFactor) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiDivC_8u_C3RSfs_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, 
                                  oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiDivC_16u_C1RSfs_Ctx(const Npp16u *pSrc, int nSrcStep, 
                                   Npp16u nConstant, Npp16u *pDst, int nDstStep,
                                   NppiSize oSizeROI, int nScaleFactor, 
                                   NppStreamContext nppStreamCtx) {
    if (nScaleFactor < 0 || nScaleFactor > 31) return NPP_BAD_ARGUMENT_ERROR;
    if (nConstant == 0) return NPP_DIVIDE_BY_ZERO_ERROR;
    return nppiDivC_16u_C1RSfs_Ctx_impl(pSrc, nSrcStep, nConstant, pDst, nDstStep, 
                                         oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiDivC_16u_C1RSfs(const Npp16u *pSrc, int nSrcStep, 
                              Npp16u nConstant, Npp16u *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiDivC_16u_C1RSfs_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, 
                                   oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiDivC_16s_C1RSfs_Ctx(const Npp16s *pSrc, int nSrcStep, 
                                   Npp16s nConstant, Npp16s *pDst, int nDstStep,
                                   NppiSize oSizeROI, int nScaleFactor, 
                                   NppStreamContext nppStreamCtx) {
    if (nScaleFactor < 0 || nScaleFactor > 31) return NPP_BAD_ARGUMENT_ERROR;
    if (nConstant == 0) return NPP_DIVIDE_BY_ZERO_ERROR;
    return nppiDivC_16s_C1RSfs_Ctx_impl(pSrc, nSrcStep, nConstant, pDst, nDstStep, 
                                         oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiDivC_16s_C1RSfs(const Npp16s *pSrc, int nSrcStep, 
                              Npp16s nConstant, Npp16s *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiDivC_16s_C1RSfs_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, 
                                   oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiDivC_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, 
                               Npp32f nConstant, Npp32f *pDst, int nDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return nppiDivC_32f_C1R_Ctx_impl(pSrc, nSrcStep, nConstant, pDst, nDstStep, 
                                      oSizeROI, nppStreamCtx);
}

NppStatus nppiDivC_32f_C1R(const Npp32f *pSrc, int nSrcStep, 
                           Npp32f nConstant, Npp32f *pDst, int nDstStep,
                           NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiDivC_32f_C1R_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, 
                                oSizeROI, nppStreamCtx);
}

// In-place versions
NppStatus nppiDivC_8u_C1IRSfs_Ctx(const Npp8u nConstant, Npp8u *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor, 
                                   NppStreamContext nppStreamCtx) {
    return nppiDivC_8u_C1RSfs_Ctx(pSrcDst, nSrcDstStep, nConstant, pSrcDst, nSrcDstStep, 
                                  oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiDivC_8u_C1IRSfs(const Npp8u nConstant, Npp8u *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, int nScaleFactor) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiDivC_8u_C1IRSfs_Ctx(nConstant, pSrcDst, nSrcDstStep, 
                                   oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiDivC_16u_C1IRSfs_Ctx(const Npp16u nConstant, Npp16u *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor, 
                                   NppStreamContext nppStreamCtx) {
    return nppiDivC_16u_C1RSfs_Ctx(pSrcDst, nSrcDstStep, nConstant, pSrcDst, nSrcDstStep, 
                                   oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiDivC_16u_C1IRSfs(const Npp16u nConstant, Npp16u *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, int nScaleFactor) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiDivC_16u_C1IRSfs_Ctx(nConstant, pSrcDst, nSrcDstStep, 
                                    oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiDivC_16s_C1IRSfs_Ctx(const Npp16s nConstant, Npp16s *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor, 
                                   NppStreamContext nppStreamCtx) {
    return nppiDivC_16s_C1RSfs_Ctx(pSrcDst, nSrcDstStep, nConstant, pSrcDst, nSrcDstStep, 
                                   oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiDivC_16s_C1IRSfs(const Npp16s nConstant, Npp16s *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, int nScaleFactor) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiDivC_16s_C1IRSfs_Ctx(nConstant, pSrcDst, nSrcDstStep, 
                                    oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiDivC_32f_C1IR_Ctx(const Npp32f nConstant, Npp32f *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return nppiDivC_32f_C1R_Ctx(pSrcDst, nSrcDstStep, nConstant, pSrcDst, nSrcDstStep, 
                                oSizeROI, nppStreamCtx);
}

NppStatus nppiDivC_32f_C1IR(const Npp32f nConstant, Npp32f *pSrcDst, int nSrcDstStep,
                            NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiDivC_32f_C1IR_Ctx(nConstant, pSrcDst, nSrcDstStep, 
                                 oSizeROI, nppStreamCtx);
}