#include "nppi_arithmetic_unified.h"

using namespace nppi::arithmetic;

// Implementation functions using the unified executor
extern "C" {

NppStatus nppiAbs_8s_C1R_Ctx_impl(const Npp8s *pSrc, int nSrcStep, 
                                   Npp8s *pDst, int nDstStep,
                                   NppiSize oSizeROI, 
                                   NppStreamContext nppStreamCtx) {
    return UnaryOperationExecutor<Npp8s, 1, AbsOp<Npp8s>>::execute(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0, nppStreamCtx.hStream);
}

NppStatus nppiAbs_16s_C1R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, 
                                    Npp16s *pDst, int nDstStep,
                                    NppiSize oSizeROI, 
                                    NppStreamContext nppStreamCtx) {
    return UnaryOperationExecutor<Npp16s, 1, AbsOp<Npp16s>>::execute(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0, nppStreamCtx.hStream);
}

NppStatus nppiAbs_32s_C1R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, 
                                    Npp32s *pDst, int nDstStep,
                                    NppiSize oSizeROI, 
                                    NppStreamContext nppStreamCtx) {
    return UnaryOperationExecutor<Npp32s, 1, AbsOp<Npp32s>>::execute(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0, nppStreamCtx.hStream);
}

NppStatus nppiAbs_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, 
                                    Npp32f *pDst, int nDstStep,
                                    NppiSize oSizeROI, 
                                    NppStreamContext nppStreamCtx) {
    return UnaryOperationExecutor<Npp32f, 1, AbsOp<Npp32f>>::execute(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0, nppStreamCtx.hStream);
}

} // extern "C"

// Public API functions
NppStatus nppiAbs_8s_C1R_Ctx(const Npp8s *pSrc, int nSrcStep, 
                              Npp8s *pDst, int nDstStep,
                              NppiSize oSizeROI, 
                              NppStreamContext nppStreamCtx) {
    return nppiAbs_8s_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_8s_C1R(const Npp8s *pSrc, int nSrcStep, 
                         Npp8s *pDst, int nDstStep, 
                         NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAbs_8s_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep, 
                               Npp16s *pDst, int nDstStep,
                               NppiSize oSizeROI, 
                               NppStreamContext nppStreamCtx) {
    return nppiAbs_16s_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_C1R(const Npp16s *pSrc, int nSrcStep, 
                          Npp16s *pDst, int nDstStep, 
                          NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAbs_16s_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32s_C1R_Ctx(const Npp32s *pSrc, int nSrcStep, 
                               Npp32s *pDst, int nDstStep,
                               NppiSize oSizeROI, 
                               NppStreamContext nppStreamCtx) {
    return nppiAbs_32s_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32s_C1R(const Npp32s *pSrc, int nSrcStep, 
                          Npp32s *pDst, int nDstStep, 
                          NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAbs_32s_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, 
                               Npp32f *pDst, int nDstStep,
                               NppiSize oSizeROI, 
                               NppStreamContext nppStreamCtx) {
    return nppiAbs_32f_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_C1R(const Npp32f *pSrc, int nSrcStep, 
                          Npp32f *pDst, int nDstStep, 
                          NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAbs_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// In-place versions
NppStatus nppiAbs_8s_C1IR_Ctx(Npp8s *pSrcDst, int nSrcDstStep, 
                               NppiSize oSizeROI, 
                               NppStreamContext nppStreamCtx) {
    return nppiAbs_8s_C1R_Ctx(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, 
                              oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_8s_C1IR(Npp8s *pSrcDst, int nSrcDstStep, 
                          NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAbs_8s_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_C1IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep, 
                                NppiSize oSizeROI, 
                                NppStreamContext nppStreamCtx) {
    return nppiAbs_16s_C1R_Ctx(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, 
                               oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_C1IR(Npp16s *pSrcDst, int nSrcDstStep, 
                           NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAbs_16s_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32s_C1IR_Ctx(Npp32s *pSrcDst, int nSrcDstStep, 
                                NppiSize oSizeROI, 
                                NppStreamContext nppStreamCtx) {
    return nppiAbs_32s_C1R_Ctx(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, 
                               oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32s_C1IR(Npp32s *pSrcDst, int nSrcDstStep, 
                           NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAbs_32s_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_C1IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep, 
                                NppiSize oSizeROI, 
                                NppStreamContext nppStreamCtx) {
    return nppiAbs_32f_C1R_Ctx(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, 
                               oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_C1IR(Npp32f *pSrcDst, int nSrcDstStep, 
                           NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAbs_32f_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}