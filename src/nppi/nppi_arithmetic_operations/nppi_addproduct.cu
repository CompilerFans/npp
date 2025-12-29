#include "nppi_arithmetic_api.h"
#include "nppi_arithmetic_executor.h"

using namespace nppi::arithmetic;

// ============================================================================
// AddProduct Implementation
// nppiAddProduct: result = dst + (src1 * src2)
// Uses unified template executors for consistent implementation
// ============================================================================

extern "C" {

// ============================================================================
// 8u32f versions (mixed type: 8u input, 32f output)
// ============================================================================

NppStatus nppiAddProduct_8u32f_C1IR_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                                        Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                        NppStreamContext nppStreamCtx) {
  return MixedTernaryOperationExecutor<Npp8u, Npp8u, Npp32f, 1, AddProductOp<Npp32f>>::execute(
      pSrcDst, nSrcDstStep, pSrc1, nSrc1Step, pSrc2, nSrc2Step, pSrcDst, nSrcDstStep, oSizeROI, 0,
      nppStreamCtx.hStream);
}

NppStatus nppiAddProduct_8u32f_C1IR(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                                    Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAddProduct_8u32f_C1IR_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pSrcDst, nSrcDstStep, oSizeROI,
                                       getDefaultStreamContext());
}

// ============================================================================
// 16u32f versions (mixed type: 16u input, 32f output)
// ============================================================================

NppStatus nppiAddProduct_16u32f_C1IR_Ctx(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step,
                                         Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx) {
  return MixedTernaryOperationExecutor<Npp16u, Npp16u, Npp32f, 1, AddProductOp<Npp32f>>::execute(
      pSrcDst, nSrcDstStep, pSrc1, nSrc1Step, pSrc2, nSrc2Step, pSrcDst, nSrcDstStep, oSizeROI, 0,
      nppStreamCtx.hStream);
}

NppStatus nppiAddProduct_16u32f_C1IR(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step,
                                     Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAddProduct_16u32f_C1IR_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pSrcDst, nSrcDstStep, oSizeROI,
                                        getDefaultStreamContext());
}

// ============================================================================
// 32f versions (same type: 32f input and output)
// ============================================================================

NppStatus nppiAddProduct_32f_C1IR_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step,
                                      Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx) {
  return TernaryOperationExecutor<Npp32f, 1, AddProductOp<Npp32f>>::execute(pSrcDst, nSrcDstStep, pSrc1, nSrc1Step,
                                                                            pSrc2, nSrc2Step, pSrcDst, nSrcDstStep,
                                                                            oSizeROI, 0, nppStreamCtx.hStream);
}

NppStatus nppiAddProduct_32f_C1IR(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step,
                                  Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAddProduct_32f_C1IR_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pSrcDst, nSrcDstStep, oSizeROI,
                                     getDefaultStreamContext());
}

// ============================================================================
// Masked AddProduct Functions (_C1IMR)
// Uses MaskedBinaryOperationExecutor for consistent implementation
// ============================================================================

NppStatus nppiAddProduct_8u32f_C1IMR_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                                         const Npp8u *pMask, int nMaskStep, Npp32f *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MaskedBinaryOperationExecutor<Npp8u, Npp32f, 1, AddProductOp<Npp32f>>::execute(
      pSrc1, nSrc1Step, pSrc2, nSrc2Step, pMask, nMaskStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx.hStream);
}

NppStatus nppiAddProduct_8u32f_C1IMR(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                                     const Npp8u *pMask, int nMaskStep, Npp32f *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI) {
  return nppiAddProduct_8u32f_C1IMR_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pMask, nMaskStep, pSrcDst, nSrcDstStep,
                                        oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAddProduct_16u32f_C1IMR_Ctx(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step,
                                          const Npp8u *pMask, int nMaskStep, Npp32f *pSrcDst, int nSrcDstStep,
                                          NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MaskedBinaryOperationExecutor<Npp16u, Npp32f, 1, AddProductOp<Npp32f>>::execute(
      pSrc1, nSrc1Step, pSrc2, nSrc2Step, pMask, nMaskStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx.hStream);
}

NppStatus nppiAddProduct_16u32f_C1IMR(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step,
                                      const Npp8u *pMask, int nMaskStep, Npp32f *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI) {
  return nppiAddProduct_16u32f_C1IMR_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pMask, nMaskStep, pSrcDst, nSrcDstStep,
                                         oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAddProduct_32f_C1IMR_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step,
                                       const Npp8u *pMask, int nMaskStep, Npp32f *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MaskedBinaryOperationExecutor<Npp32f, Npp32f, 1, AddProductOp<Npp32f>>::execute(
      pSrc1, nSrc1Step, pSrc2, nSrc2Step, pMask, nMaskStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx.hStream);
}

NppStatus nppiAddProduct_32f_C1IMR(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step,
                                   const Npp8u *pMask, int nMaskStep, Npp32f *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI) {
  return nppiAddProduct_32f_C1IMR_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pMask, nMaskStep, pSrcDst, nSrcDstStep,
                                      oSizeROI, getDefaultStreamContext());
}

// ============================================================================
// 16f AddProduct Implementation
// Uses TernaryOperationExecutor with AddProduct16fOp
// ============================================================================

NppStatus nppiAddProduct_16f_C1IR_Ctx(const Npp16f *pSrc1, int nSrc1Step, const Npp16f *pSrc2, int nSrc2Step,
                                      Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx) {
  return TernaryOperationExecutor<Npp16f, 1, AddProduct16fOp>::execute(pSrcDst, nSrcDstStep, pSrc1, nSrc1Step, pSrc2,
                                                                       nSrc2Step, pSrcDst, nSrcDstStep, oSizeROI, 0,
                                                                       nppStreamCtx.hStream);
}

NppStatus nppiAddProduct_16f_C1IR(const Npp16f *pSrc1, int nSrc1Step, const Npp16f *pSrc2, int nSrc2Step,
                                  Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAddProduct_16f_C1IR_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pSrcDst, nSrcDstStep, oSizeROI,
                                     getDefaultStreamContext());
}

} // extern "C"
