#include "nppi_arithmetic_unified.h"

using namespace nppi::arithmetic;

// Implementation functions using the unified executor
extern "C" {

// 8u C1 operations with scale factor
NppStatus nppiAddC_8u_C1RSfs_Ctx_impl(const Npp8u *pSrc, int nSrcStep, 
                                       Npp8u nConstant, Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI, int nScaleFactor, 
                                       NppStreamContext nppStreamCtx) {
    AddConstOp<Npp8u> op(nConstant);
    return ConstOperationExecutor<Npp8u, 1, AddConstOp<Npp8u>>::execute(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx.hStream, op);
}

// 8u C3 operations with scale factor
NppStatus nppiAddC_8u_C3RSfs_Ctx_impl(const Npp8u *pSrc, int nSrcStep, 
                                       const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI, int nScaleFactor, 
                                       NppStreamContext nppStreamCtx) {
    AddConstOp<Npp8u> op(aConstants[0]);
    return ConstOperationExecutor<Npp8u, 3, AddConstOp<Npp8u>>::execute(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx.hStream, op);
}

// 8u C4 operations with scale factor
NppStatus nppiAddC_8u_C4RSfs_Ctx_impl(const Npp8u *pSrc, int nSrcStep, 
                                       const Npp8u aConstants[4], Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI, int nScaleFactor, 
                                       NppStreamContext nppStreamCtx) {
    AddConstOp<Npp8u> op(aConstants[0]);
    return ConstOperationExecutor<Npp8u, 4, AddConstOp<Npp8u>>::execute(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx.hStream, op);
}

// 8u AC4 operations with scale factor (Alpha Channel 4 - only RGB channels processed)
NppStatus nppiAddC_8u_AC4RSfs_Ctx_impl(const Npp8u *pSrc, int nSrcStep, 
                                        const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                                        NppiSize oSizeROI, int nScaleFactor, 
                                        NppStreamContext nppStreamCtx) {
    AddConstOp<Npp8u> op(aConstants[0]);
    return ConstOperationExecutor<Npp8u, 3, AddConstOp<Npp8u>>::execute(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx.hStream, op);
}

// 16u C1 operations with scale factor
NppStatus nppiAddC_16u_C1RSfs_Ctx_impl(const Npp16u *pSrc, int nSrcStep, 
                                        Npp16u nConstant, Npp16u *pDst, int nDstStep,
                                        NppiSize oSizeROI, int nScaleFactor, 
                                        NppStreamContext nppStreamCtx) {
    AddConstOp<Npp16u> op(nConstant);
    return ConstOperationExecutor<Npp16u, 1, AddConstOp<Npp16u>>::execute(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx.hStream, op);
}

// 16s C1 operations with scale factor
NppStatus nppiAddC_16s_C1RSfs_Ctx_impl(const Npp16s *pSrc, int nSrcStep, 
                                        Npp16s nConstant, Npp16s *pDst, int nDstStep,
                                        NppiSize oSizeROI, int nScaleFactor, 
                                        NppStreamContext nppStreamCtx) {
    AddConstOp<Npp16s> op(nConstant);
    return ConstOperationExecutor<Npp16s, 1, AddConstOp<Npp16s>>::execute(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx.hStream, op);
}

// 32f C1 operations (no scale factor)
NppStatus nppiAddC_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, 
                                     Npp32f nConstant, Npp32f *pDst, int nDstStep,
                                     NppiSize oSizeROI, 
                                     NppStreamContext nppStreamCtx) {
    AddConstOp<Npp32f> op(nConstant);
    return ConstOperationExecutor<Npp32f, 1, AddConstOp<Npp32f>>::execute(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0, nppStreamCtx.hStream, op);
}

} // extern "C"

// Public API functions
NppStatus nppiAddC_8u_C1RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, 
                                  Npp8u nConstant, Npp8u *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor, 
                                  NppStreamContext nppStreamCtx) {
    if (nScaleFactor < 0 || nScaleFactor > 31) return NPP_BAD_ARGUMENT_ERROR;
    return nppiAddC_8u_C1RSfs_Ctx_impl(pSrc, nSrcStep, nConstant, pDst, nDstStep, 
                                        oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_8u_C1RSfs(const Npp8u *pSrc, int nSrcStep, 
                             Npp8u nConstant, Npp8u *pDst, int nDstStep,
                             NppiSize oSizeROI, int nScaleFactor) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAddC_8u_C1RSfs_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, 
                                  oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, 
                              Npp8u nConstant, Npp8u *pDst, int nDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return nppiAddC_8u_C1RSfs_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, 
                                  oSizeROI, 0, nppStreamCtx);
}

NppStatus nppiAddC_8u_C1R(const Npp8u *pSrc, int nSrcStep, 
                          Npp8u nConstant, Npp8u *pDst, int nDstStep,
                          NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAddC_8u_C1R_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, 
                               oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_8u_C3RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, 
                                  const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor, 
                                  NppStreamContext nppStreamCtx) {
    if (nScaleFactor < 0 || nScaleFactor > 31) return NPP_BAD_ARGUMENT_ERROR;
    return nppiAddC_8u_C3RSfs_Ctx_impl(pSrc, nSrcStep, aConstants, pDst, nDstStep, 
                                        oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_8u_C3RSfs(const Npp8u *pSrc, int nSrcStep, 
                             const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                             NppiSize oSizeROI, int nScaleFactor) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAddC_8u_C3RSfs_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, 
                                  oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, 
                              const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return nppiAddC_8u_C3RSfs_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, 
                                  oSizeROI, 0, nppStreamCtx);
}

NppStatus nppiAddC_8u_C3R(const Npp8u *pSrc, int nSrcStep, 
                          const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                          NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAddC_8u_C3R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, 
                               oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_8u_C4RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, 
                                  const Npp8u aConstants[4], Npp8u *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor, 
                                  NppStreamContext nppStreamCtx) {
    if (nScaleFactor < 0 || nScaleFactor > 31) return NPP_BAD_ARGUMENT_ERROR;
    return nppiAddC_8u_C4RSfs_Ctx_impl(pSrc, nSrcStep, aConstants, pDst, nDstStep, 
                                        oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_8u_C4RSfs(const Npp8u *pSrc, int nSrcStep, 
                             const Npp8u aConstants[4], Npp8u *pDst, int nDstStep,
                             NppiSize oSizeROI, int nScaleFactor) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAddC_8u_C4RSfs_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, 
                                  oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, 
                              const Npp8u aConstants[4], Npp8u *pDst, int nDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return nppiAddC_8u_C4RSfs_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, 
                                  oSizeROI, 0, nppStreamCtx);
}

NppStatus nppiAddC_8u_C4R(const Npp8u *pSrc, int nSrcStep, 
                          const Npp8u aConstants[4], Npp8u *pDst, int nDstStep,
                          NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAddC_8u_C4R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, 
                               oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_8u_AC4RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, 
                                   const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                                   NppiSize oSizeROI, int nScaleFactor, 
                                   NppStreamContext nppStreamCtx) {
    if (nScaleFactor < 0 || nScaleFactor > 31) return NPP_BAD_ARGUMENT_ERROR;
    return nppiAddC_8u_AC4RSfs_Ctx_impl(pSrc, nSrcStep, aConstants, pDst, nDstStep, 
                                         oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_8u_AC4RSfs(const Npp8u *pSrc, int nSrcStep, 
                              const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAddC_8u_AC4RSfs_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, 
                                   oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, 
                               const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return nppiAddC_8u_AC4RSfs_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, 
                                   oSizeROI, 0, nppStreamCtx);
}

NppStatus nppiAddC_8u_AC4R(const Npp8u *pSrc, int nSrcStep, 
                           const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                           NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAddC_8u_AC4R_Ctx(pSrc, nSrcStep, aConstants, pDst, nDstStep, 
                                oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_16u_C1RSfs_Ctx(const Npp16u *pSrc, int nSrcStep, 
                                   Npp16u nConstant, Npp16u *pDst, int nDstStep,
                                   NppiSize oSizeROI, int nScaleFactor, 
                                   NppStreamContext nppStreamCtx) {
    if (nScaleFactor < 0 || nScaleFactor > 31) return NPP_BAD_ARGUMENT_ERROR;
    return nppiAddC_16u_C1RSfs_Ctx_impl(pSrc, nSrcStep, nConstant, pDst, nDstStep, 
                                         oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_16u_C1RSfs(const Npp16u *pSrc, int nSrcStep, 
                              Npp16u nConstant, Npp16u *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAddC_16u_C1RSfs_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, 
                                   oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_16s_C1RSfs_Ctx(const Npp16s *pSrc, int nSrcStep, 
                                   Npp16s nConstant, Npp16s *pDst, int nDstStep,
                                   NppiSize oSizeROI, int nScaleFactor, 
                                   NppStreamContext nppStreamCtx) {
    if (nScaleFactor < 0 || nScaleFactor > 31) return NPP_BAD_ARGUMENT_ERROR;
    return nppiAddC_16s_C1RSfs_Ctx_impl(pSrc, nSrcStep, nConstant, pDst, nDstStep, 
                                         oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_16s_C1RSfs(const Npp16s *pSrc, int nSrcStep, 
                              Npp16s nConstant, Npp16s *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAddC_16s_C1RSfs_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, 
                                   oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, 
                               Npp32f nConstant, Npp32f *pDst, int nDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return nppiAddC_32f_C1R_Ctx_impl(pSrc, nSrcStep, nConstant, pDst, nDstStep, 
                                      oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_32f_C1R(const Npp32f *pSrc, int nSrcStep, 
                           Npp32f nConstant, Npp32f *pDst, int nDstStep,
                           NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAddC_32f_C1R_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, 
                                oSizeROI, nppStreamCtx);
}

// In-place versions
NppStatus nppiAddC_8u_C1IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep, 
                                   Npp8u *pSrcDst, int nSrcDstStep,
                                   Npp8u nConstant, NppiSize oSizeROI, 
                                   int nScaleFactor, NppStreamContext nppStreamCtx) {
    return nppiAddC_8u_C1RSfs_Ctx(pSrc, nSrcStep, nConstant, pSrcDst, nSrcDstStep, 
                                  oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_8u_C1IRSfs(const Npp8u *pSrc, int nSrcStep, 
                              Npp8u *pSrcDst, int nSrcDstStep,
                              Npp8u nConstant, NppiSize oSizeROI, int nScaleFactor) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAddC_8u_C1IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, 
                                   nConstant, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAddC_8u_C1IR_Ctx(const Npp8u *pSrc, int nSrcStep, 
                               Npp8u *pSrcDst, int nSrcDstStep,
                               Npp8u nConstant, NppiSize oSizeROI, 
                               NppStreamContext nppStreamCtx) {
    return nppiAddC_8u_C1R_Ctx(pSrc, nSrcStep, nConstant, pSrcDst, nSrcDstStep, 
                               oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_8u_C1IR(const Npp8u *pSrc, int nSrcStep, 
                           Npp8u *pSrcDst, int nSrcDstStep,
                           Npp8u nConstant, NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAddC_8u_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, 
                                nConstant, oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_32f_C1IR_Ctx(const Npp32f *pSrc, int nSrcStep, 
                                Npp32f *pSrcDst, int nSrcDstStep,
                                Npp32f nConstant, NppiSize oSizeROI, 
                                NppStreamContext nppStreamCtx) {
    return nppiAddC_32f_C1R_Ctx(pSrc, nSrcStep, nConstant, pSrcDst, nSrcDstStep, 
                                oSizeROI, nppStreamCtx);
}

NppStatus nppiAddC_32f_C1IR(const Npp32f *pSrc, int nSrcStep, 
                            Npp32f *pSrcDst, int nSrcDstStep,
                            Npp32f nConstant, NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAddC_32f_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, 
                                 nConstant, oSizeROI, nppStreamCtx);
}