#include "nppi_arithmetic_api.h"
#include "nppi_arithmetic_executor.h"

using namespace nppi::arithmetic;

// ============================================================================
// AddWeighted Implementation
// nppiAddWeighted: result = src * alpha + dst * (1 - alpha) (in-place)
// Uses unified template executors for consistent implementation
// ============================================================================

extern "C" {

// ============================================================================
// 8u32f versions (8-bit source, 32-bit float destination)
// ============================================================================

NppStatus nppiAddWeighted_8u32f_C1IR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, Npp32f nAlpha, NppStreamContext nppStreamCtx) {
  return WeightedOperationExecutor<Npp8u, Npp32f, 1, AddWeightedInplaceOp<Npp8u, Npp32f>>::execute(
      pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nAlpha, nppStreamCtx.hStream);
}

NppStatus nppiAddWeighted_8u32f_C1IR(const Npp8u *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI, Npp32f nAlpha) {
  return nppiAddWeighted_8u32f_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nAlpha,
                                        getDefaultStreamContext());
}

// ============================================================================
// 16u32f versions (16-bit unsigned source, 32-bit float destination)
// ============================================================================

NppStatus nppiAddWeighted_16u32f_C1IR_Ctx(const Npp16u *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep,
                                          NppiSize oSizeROI, Npp32f nAlpha, NppStreamContext nppStreamCtx) {
  return WeightedOperationExecutor<Npp16u, Npp32f, 1, AddWeightedInplaceOp<Npp16u, Npp32f>>::execute(
      pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nAlpha, nppStreamCtx.hStream);
}

NppStatus nppiAddWeighted_16u32f_C1IR(const Npp16u *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI, Npp32f nAlpha) {
  return nppiAddWeighted_16u32f_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nAlpha,
                                         getDefaultStreamContext());
}

// ============================================================================
// 32f versions (32-bit float source and destination)
// ============================================================================

NppStatus nppiAddWeighted_32f_C1IR_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI, Npp32f nAlpha, NppStreamContext nppStreamCtx) {
  return WeightedOperationExecutor<Npp32f, Npp32f, 1, AddWeightedInplaceOp<Npp32f, Npp32f>>::execute(
      pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nAlpha, nppStreamCtx.hStream);
}

NppStatus nppiAddWeighted_32f_C1IR(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, Npp32f nAlpha) {
  return nppiAddWeighted_32f_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nAlpha,
                                      getDefaultStreamContext());
}

// ============================================================================
// Masked versions
// Uses MaskedWeightedOperationExecutor for consistent implementation
// ============================================================================

NppStatus nppiAddWeighted_8u32f_C1IMR_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep,
                                          Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, Npp32f nAlpha,
                                          NppStreamContext nppStreamCtx) {
  return MaskedWeightedOperationExecutor<Npp8u, Npp32f, 1, AddWeightedInplaceOp<Npp8u, Npp32f>>::execute(
      pSrc, nSrcStep, pMask, nMaskStep, pSrcDst, nSrcDstStep, oSizeROI, nAlpha, nppStreamCtx.hStream);
}

NppStatus nppiAddWeighted_8u32f_C1IMR(const Npp8u *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep,
                                      Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, Npp32f nAlpha) {
  return nppiAddWeighted_8u32f_C1IMR_Ctx(pSrc, nSrcStep, pMask, nMaskStep, pSrcDst, nSrcDstStep, oSizeROI, nAlpha,
                                         getDefaultStreamContext());
}

NppStatus nppiAddWeighted_16u32f_C1IMR_Ctx(const Npp16u *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep,
                                           Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, Npp32f nAlpha,
                                           NppStreamContext nppStreamCtx) {
  return MaskedWeightedOperationExecutor<Npp16u, Npp32f, 1, AddWeightedInplaceOp<Npp16u, Npp32f>>::execute(
      pSrc, nSrcStep, pMask, nMaskStep, pSrcDst, nSrcDstStep, oSizeROI, nAlpha, nppStreamCtx.hStream);
}

NppStatus nppiAddWeighted_16u32f_C1IMR(const Npp16u *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep,
                                       Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, Npp32f nAlpha) {
  return nppiAddWeighted_16u32f_C1IMR_Ctx(pSrc, nSrcStep, pMask, nMaskStep, pSrcDst, nSrcDstStep, oSizeROI, nAlpha,
                                          getDefaultStreamContext());
}

NppStatus nppiAddWeighted_32f_C1IMR_Ctx(const Npp32f *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep,
                                        Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, Npp32f nAlpha,
                                        NppStreamContext nppStreamCtx) {
  return MaskedWeightedOperationExecutor<Npp32f, Npp32f, 1, AddWeightedInplaceOp<Npp32f, Npp32f>>::execute(
      pSrc, nSrcStep, pMask, nMaskStep, pSrcDst, nSrcDstStep, oSizeROI, nAlpha, nppStreamCtx.hStream);
}

NppStatus nppiAddWeighted_32f_C1IMR(const Npp32f *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep,
                                    Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, Npp32f nAlpha) {
  return nppiAddWeighted_32f_C1IMR_Ctx(pSrc, nSrcStep, pMask, nMaskStep, pSrcDst, nSrcDstStep, oSizeROI, nAlpha,
                                       getDefaultStreamContext());
}

} // extern "C"
