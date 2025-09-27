#include "nppi_arithmetic_unified.h"

using namespace nppi::arithmetic;

// Implementation functions using the unified executor
extern "C" {

// 8u C1 operations (no scale factor for logical operations)
NppStatus nppiOr_8u_C1RSfs_Ctx_impl(const Npp8u *pSrc1, int nSrc1Step, 
                                     const Npp8u *pSrc2, int nSrc2Step,
                                     Npp8u *pDst, int nDstStep, 
                                     NppiSize oSizeROI, int nScaleFactor, 
                                     NppStreamContext nppStreamCtx) {
    return BinaryOperationExecutor<Npp8u, 1, OrOp<Npp8u>>::execute(
        pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
        oSizeROI, 0, nppStreamCtx.hStream); // Scale factor ignored for logical ops
}

// 8u C3 operations (no scale factor for logical operations)
NppStatus nppiOr_8u_C3RSfs_Ctx_impl(const Npp8u *pSrc1, int nSrc1Step, 
                                     const Npp8u *pSrc2, int nSrc2Step,
                                     Npp8u *pDst, int nDstStep, 
                                     NppiSize oSizeROI, int nScaleFactor, 
                                     NppStreamContext nppStreamCtx) {
    return BinaryOperationExecutor<Npp8u, 3, OrOp<Npp8u>>::execute(
        pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
        oSizeROI, 0, nppStreamCtx.hStream); // Scale factor ignored for logical ops
}

// 16u C1 operations
NppStatus nppiOr_16u_C1RSfs_Ctx_impl(const Npp16u *pSrc1, int nSrc1Step, 
                                      const Npp16u *pSrc2, int nSrc2Step,
                                      Npp16u *pDst, int nDstStep, 
                                      NppiSize oSizeROI, int nScaleFactor, 
                                      NppStreamContext nppStreamCtx) {
    return BinaryOperationExecutor<Npp16u, 1, OrOp<Npp16u>>::execute(
        pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
        oSizeROI, 0, nppStreamCtx.hStream);
}

// 32s C1 operations
NppStatus nppiOr_32s_C1RSfs_Ctx_impl(const Npp32s *pSrc1, int nSrc1Step, 
                                      const Npp32s *pSrc2, int nSrc2Step,
                                      Npp32s *pDst, int nDstStep, 
                                      NppiSize oSizeROI, int nScaleFactor, 
                                      NppStreamContext nppStreamCtx) {
    return BinaryOperationExecutor<Npp32s, 1, OrOp<Npp32s>>::execute(
        pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
        oSizeROI, 0, nppStreamCtx.hStream);
}

} // extern "C"

// Public API functions
NppStatus nppiOr_8u_C1RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, 
                                const Npp8u *pSrc2, int nSrc2Step,
                                Npp8u *pDst, int nDstStep, 
                                NppiSize oSizeROI, int nScaleFactor, 
                                NppStreamContext nppStreamCtx) {
    return nppiOr_8u_C1RSfs_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                                      pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiOr_8u_C1RSfs(const Npp8u *pSrc1, int nSrc1Step, 
                           const Npp8u *pSrc2, int nSrc2Step,
                           Npp8u *pDst, int nDstStep, 
                           NppiSize oSizeROI, int nScaleFactor) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiOr_8u_C1RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                                pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiOr_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step, 
                            const Npp8u *pSrc2, int nSrc2Step,
                            Npp8u *pDst, int nDstStep, 
                            NppiSize oSizeROI, 
                            NppStreamContext nppStreamCtx) {
    return nppiOr_8u_C1RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                                pDst, nDstStep, oSizeROI, 0, nppStreamCtx);
}

NppStatus nppiOr_8u_C1R(const Npp8u *pSrc1, int nSrc1Step, 
                        const Npp8u *pSrc2, int nSrc2Step,
                        Npp8u *pDst, int nDstStep, 
                        NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiOr_8u_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                             pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiOr_8u_C3RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, 
                                const Npp8u *pSrc2, int nSrc2Step,
                                Npp8u *pDst, int nDstStep, 
                                NppiSize oSizeROI, int nScaleFactor, 
                                NppStreamContext nppStreamCtx) {
    return nppiOr_8u_C3RSfs_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                                      pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiOr_8u_C3RSfs(const Npp8u *pSrc1, int nSrc1Step, 
                           const Npp8u *pSrc2, int nSrc2Step,
                           Npp8u *pDst, int nDstStep, 
                           NppiSize oSizeROI, int nScaleFactor) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiOr_8u_C3RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                                pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiOr_8u_C3R_Ctx(const Npp8u *pSrc1, int nSrc1Step, 
                            const Npp8u *pSrc2, int nSrc2Step,
                            Npp8u *pDst, int nDstStep, 
                            NppiSize oSizeROI, 
                            NppStreamContext nppStreamCtx) {
    return nppiOr_8u_C3RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                                pDst, nDstStep, oSizeROI, 0, nppStreamCtx);
}

NppStatus nppiOr_8u_C3R(const Npp8u *pSrc1, int nSrc1Step, 
                        const Npp8u *pSrc2, int nSrc2Step,
                        Npp8u *pDst, int nDstStep, 
                        NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiOr_8u_C3R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                             pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiOr_16u_C1RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step, 
                                 const Npp16u *pSrc2, int nSrc2Step,
                                 Npp16u *pDst, int nDstStep, 
                                 NppiSize oSizeROI, int nScaleFactor, 
                                 NppStreamContext nppStreamCtx) {
    return nppiOr_16u_C1RSfs_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                                       pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiOr_16u_C1RSfs(const Npp16u *pSrc1, int nSrc1Step, 
                            const Npp16u *pSrc2, int nSrc2Step,
                            Npp16u *pDst, int nDstStep, 
                            NppiSize oSizeROI, int nScaleFactor) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiOr_16u_C1RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                                 pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiOr_32s_C1RSfs_Ctx(const Npp32s *pSrc1, int nSrc1Step, 
                                 const Npp32s *pSrc2, int nSrc2Step,
                                 Npp32s *pDst, int nDstStep, 
                                 NppiSize oSizeROI, int nScaleFactor, 
                                 NppStreamContext nppStreamCtx) {
    return nppiOr_32s_C1RSfs_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                                       pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiOr_32s_C1RSfs(const Npp32s *pSrc1, int nSrc1Step, 
                            const Npp32s *pSrc2, int nSrc2Step,
                            Npp32s *pDst, int nDstStep, 
                            NppiSize oSizeROI, int nScaleFactor) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiOr_32s_C1RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                                 pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

// In-place versions
NppStatus nppiOr_8u_C1IR_Ctx(const Npp8u *pSrc, int nSrcStep, 
                             Npp8u *pSrcDst, int nSrcDstStep,
                             NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return nppiOr_8u_C1R_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, 
                             pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiOr_8u_C1IR(const Npp8u *pSrc, int nSrcStep, 
                         Npp8u *pSrcDst, int nSrcDstStep,
                         NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiOr_8u_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiOr_8u_C3IR_Ctx(const Npp8u *pSrc, int nSrcStep, 
                             Npp8u *pSrcDst, int nSrcDstStep,
                             NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return nppiOr_8u_C3R_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, 
                             pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiOr_8u_C3IR(const Npp8u *pSrc, int nSrcStep, 
                         Npp8u *pSrcDst, int nSrcDstStep,
                         NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiOr_8u_C3IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}