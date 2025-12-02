#include "nppi_arithmetic_api.h"
#include "nppi_arithmetic_executor.h"

using namespace nppi::arithmetic;

// ============================================================================
// AddSquare Implementation
// nppiAddSquare: result = dst + (src * src)
// Uses unified template executors for consistent implementation
// ============================================================================

extern "C" {

// ============================================================================
// 8u32f versions (mixed type: 8u input, 32f output)
// ============================================================================

NppStatus nppiAddSquare_8u32f_C1IR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MixedUnaryOperationExecutor<Npp8u, Npp32f, 1, AddSquareOp<Npp32f>>::execute(
      pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx.hStream);
}

NppStatus nppiAddSquare_8u32f_C1IR(const Npp8u *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI) {
  return nppiAddSquare_8u32f_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// ============================================================================
// 16u32f versions (mixed type: 16u input, 32f output)
// ============================================================================

NppStatus nppiAddSquare_16u32f_C1IR_Ctx(const Npp16u *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return MixedUnaryOperationExecutor<Npp16u, Npp32f, 1, AddSquareOp<Npp32f>>::execute(
      pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx.hStream);
}

NppStatus nppiAddSquare_16u32f_C1IR(const Npp16u *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI) {
  return nppiAddSquare_16u32f_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// ============================================================================
// 32f versions (same type: 32f input and output)
// ============================================================================

NppStatus nppiAddSquare_32f_C1IR_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return BinaryOperationExecutor<Npp32f, 1, AddSquareOp<Npp32f>>::execute(
      pSrcDst, nSrcDstStep, pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, 0, nppStreamCtx.hStream);
}

NppStatus nppiAddSquare_32f_C1IR(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI) {
  return nppiAddSquare_32f_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// ============================================================================
// 8u versions (integer with scale factor)
// ============================================================================

NppStatus nppiAddSquare_8u_C1IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return BinaryOperationExecutor<Npp8u, 1, AddSquareOp<Npp8u>>::execute(
      pSrcDst, nSrcDstStep, pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx.hStream);
}

NppStatus nppiAddSquare_8u_C1IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor) {
  return nppiAddSquare_8u_C1IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor,
                                      getDefaultStreamContext());
}

// ============================================================================
// 16u versions (integer with scale factor)
// ============================================================================

NppStatus nppiAddSquare_16u_C1IRSfs_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return BinaryOperationExecutor<Npp16u, 1, AddSquareOp<Npp16u>>::execute(
      pSrcDst, nSrcDstStep, pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx.hStream);
}

NppStatus nppiAddSquare_16u_C1IRSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, int nScaleFactor) {
  return nppiAddSquare_16u_C1IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor,
                                       getDefaultStreamContext());
}

// ============================================================================
// Masked AddSquare Functions (_C1IMR)
// Uses MaskedUnaryOperationExecutor for consistent implementation
// ============================================================================

NppStatus nppiAddSquare_8u32f_C1IMR_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep,
                                        Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                        NppStreamContext nppStreamCtx) {
  return MaskedUnaryOperationExecutor<Npp8u, Npp32f, 1, AddSquareOp<Npp32f>>::execute(
      pSrc, nSrcStep, pMask, nMaskStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx.hStream);
}

NppStatus nppiAddSquare_8u32f_C1IMR(const Npp8u *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep, Npp32f *pSrcDst,
                                    int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAddSquare_8u32f_C1IMR_Ctx(pSrc, nSrcStep, pMask, nMaskStep, pSrcDst, nSrcDstStep, oSizeROI,
                                       getDefaultStreamContext());
}

NppStatus nppiAddSquare_16u32f_C1IMR_Ctx(const Npp16u *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep,
                                         Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx) {
  return MaskedUnaryOperationExecutor<Npp16u, Npp32f, 1, AddSquareOp<Npp32f>>::execute(
      pSrc, nSrcStep, pMask, nMaskStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx.hStream);
}

NppStatus nppiAddSquare_16u32f_C1IMR(const Npp16u *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep,
                                     Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAddSquare_16u32f_C1IMR_Ctx(pSrc, nSrcStep, pMask, nMaskStep, pSrcDst, nSrcDstStep, oSizeROI,
                                        getDefaultStreamContext());
}

NppStatus nppiAddSquare_32f_C1IMR_Ctx(const Npp32f *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep,
                                      Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx) {
  return MaskedUnaryOperationExecutor<Npp32f, Npp32f, 1, AddSquareOp<Npp32f>>::execute(
      pSrc, nSrcStep, pMask, nMaskStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx.hStream);
}

NppStatus nppiAddSquare_32f_C1IMR(const Npp32f *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep, Npp32f *pSrcDst,
                                  int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAddSquare_32f_C1IMR_Ctx(pSrc, nSrcStep, pMask, nMaskStep, pSrcDst, nSrcDstStep, oSizeROI,
                                     getDefaultStreamContext());
}

} // extern "C"
