#include "nppi_arithmetic_ops.h"
#include "nppi_arithmetic_executor.h"

using namespace nppi::arithmetic;

// Implementation functions using the unified executor
extern "C" {

// 8u C1 operations with scale factor
NppStatus nppiSub_8u_C1RSfs_Ctx_impl(const Npp8u *pSrc1, int nSrc1Step, 
                                      const Npp8u *pSrc2, int nSrc2Step,
                                      Npp8u *pDst, int nDstStep, 
                                      NppiSize oSizeROI, int nScaleFactor, 
                                      NppStreamContext nppStreamCtx) {
    return BinaryOperationExecutor<Npp8u, 1, SubOp<Npp8u>>::execute(
        pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
        oSizeROI, nScaleFactor, nppStreamCtx.hStream);
}

// 8u C3 operations with scale factor
NppStatus nppiSub_8u_C3RSfs_Ctx_impl(const Npp8u *pSrc1, int nSrc1Step, 
                                      const Npp8u *pSrc2, int nSrc2Step,
                                      Npp8u *pDst, int nDstStep, 
                                      NppiSize oSizeROI, int nScaleFactor, 
                                      NppStreamContext nppStreamCtx) {
    return BinaryOperationExecutor<Npp8u, 3, SubOp<Npp8u>>::execute(
        pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
        oSizeROI, nScaleFactor, nppStreamCtx.hStream);
}

// 16u C1 operations with scale factor
NppStatus nppiSub_16u_C1RSfs_Ctx_impl(const Npp16u *pSrc1, int nSrc1Step, 
                                       const Npp16u *pSrc2, int nSrc2Step,
                                       Npp16u *pDst, int nDstStep, 
                                       NppiSize oSizeROI, int nScaleFactor, 
                                       NppStreamContext nppStreamCtx) {
    return BinaryOperationExecutor<Npp16u, 1, SubOp<Npp16u>>::execute(
        pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
        oSizeROI, nScaleFactor, nppStreamCtx.hStream);
}

// 32f C1 operations (no scale factor)
NppStatus nppiSub_32f_C1R_Ctx_impl(const Npp32f *pSrc1, int nSrc1Step, 
                                    const Npp32f *pSrc2, int nSrc2Step,
                                    Npp32f *pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    NppStreamContext nppStreamCtx) {
    return BinaryOperationExecutor<Npp32f, 1, SubOp<Npp32f>>::execute(
        pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
        oSizeROI, 0, nppStreamCtx.hStream);
}

} // extern "C"

// Public API functions
NppStatus nppiSub_8u_C1RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, 
                                 const Npp8u *pSrc2, int nSrc2Step,
                                 Npp8u *pDst, int nDstStep, 
                                 NppiSize oSizeROI, int nScaleFactor, 
                                 NppStreamContext nppStreamCtx) {
    if (nScaleFactor < 0 || nScaleFactor > 31) return NPP_BAD_ARGUMENT_ERROR;
    return nppiSub_8u_C1RSfs_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                                       pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSub_8u_C1RSfs(const Npp8u *pSrc1, int nSrc1Step, 
                            const Npp8u *pSrc2, int nSrc2Step,
                            Npp8u *pDst, int nDstStep, 
                            NppiSize oSizeROI, int nScaleFactor) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiSub_8u_C1RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                                 pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSub_8u_C3RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, 
                                 const Npp8u *pSrc2, int nSrc2Step,
                                 Npp8u *pDst, int nDstStep, 
                                 NppiSize oSizeROI, int nScaleFactor, 
                                 NppStreamContext nppStreamCtx) {
    if (nScaleFactor < 0 || nScaleFactor > 31) return NPP_BAD_ARGUMENT_ERROR;
    return nppiSub_8u_C3RSfs_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                                       pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSub_8u_C3RSfs(const Npp8u *pSrc1, int nSrc1Step, 
                            const Npp8u *pSrc2, int nSrc2Step,
                            Npp8u *pDst, int nDstStep, 
                            NppiSize oSizeROI, int nScaleFactor) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiSub_8u_C3RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                                 pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSub_16u_C1RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step, 
                                  const Npp16u *pSrc2, int nSrc2Step,
                                  Npp16u *pDst, int nDstStep, 
                                  NppiSize oSizeROI, int nScaleFactor, 
                                  NppStreamContext nppStreamCtx) {
    if (nScaleFactor < 0 || nScaleFactor > 31) return NPP_BAD_ARGUMENT_ERROR;
    return nppiSub_16u_C1RSfs_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                                        pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSub_16u_C1RSfs(const Npp16u *pSrc1, int nSrc1Step, 
                             const Npp16u *pSrc2, int nSrc2Step,
                             Npp16u *pDst, int nDstStep, 
                             NppiSize oSizeROI, int nScaleFactor) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiSub_16u_C1RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                                  pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSub_32f_C1R_Ctx(const Npp32f *pSrc1, int nSrc1Step, 
                               const Npp32f *pSrc2, int nSrc2Step,
                               Npp32f *pDst, int nDstStep, 
                               NppiSize oSizeROI, 
                               NppStreamContext nppStreamCtx) {
    return nppiSub_32f_C1R_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                                     pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSub_32f_C1R(const Npp32f *pSrc1, int nSrc1Step, 
                          const Npp32f *pSrc2, int nSrc2Step,
                          Npp32f *pDst, int nDstStep, 
                          NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiSub_32f_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                               pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// In-place versions
NppStatus nppiSub_8u_C1IR_Ctx(const Npp8u *pSrc, int nSrcStep, 
                              Npp8u *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return nppiSub_8u_C1RSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, 
                                 pSrcDst, nSrcDstStep, oSizeROI, 0, nppStreamCtx);
}

NppStatus nppiSub_8u_C1IR(const Npp8u *pSrc, int nSrcStep, 
                          Npp8u *pSrcDst, int nSrcDstStep,
                          NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiSub_8u_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSub_8u_C3IR_Ctx(const Npp8u *pSrc, int nSrcStep, 
                              Npp8u *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return nppiSub_8u_C3RSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, 
                                 pSrcDst, nSrcDstStep, oSizeROI, 0, nppStreamCtx);
}

NppStatus nppiSub_8u_C3IR(const Npp8u *pSrc, int nSrcStep, 
                          Npp8u *pSrcDst, int nSrcDstStep,
                          NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiSub_8u_C3IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSub_32f_C1IR_Ctx(const Npp32f *pSrc, int nSrcStep, 
                               Npp32f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return nppiSub_32f_C1R_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, 
                               pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSub_32f_C1IR(const Npp32f *pSrc, int nSrcStep, 
                           Npp32f *pSrcDst, int nSrcDstStep,
                           NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiSub_32f_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}