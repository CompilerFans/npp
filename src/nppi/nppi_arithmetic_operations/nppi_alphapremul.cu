#include "nppi_arithmetic_api.h"
#include "nppi_arithmetic_executor.h"

using namespace nppi::arithmetic;

// ============================================================================
// AlphaPremul Implementation
// nppiAlphaPremulC: Alpha premultiplication with constant alpha value
// nppiAlphaPremul: Alpha premultiplication with pixel alpha value
// Uses unified template executors for consistent implementation
// ============================================================================

extern "C" {

// ============================================================================
// AlphaPremulC - Constant Alpha Premultiplication
// Uses AlphaPremulConstExecutor with AlphaPremulConstSimpleOp
// ============================================================================

// 8u C1 versions
NppStatus nppiAlphaPremulC_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step, Npp8u nAlpha1, Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return AlphaPremulConstExecutor<Npp8u, 1, AlphaPremulConstSimpleOp<Npp8u>>::execute(
      pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nAlpha1, nppStreamCtx.hStream);
}

NppStatus nppiAlphaPremulC_8u_C1R(const Npp8u *pSrc1, int nSrc1Step, Npp8u nAlpha1, Npp8u *pDst, int nDstStep,
                                  NppiSize oSizeROI) {
  return nppiAlphaPremulC_8u_C1R_Ctx(pSrc1, nSrc1Step, nAlpha1, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAlphaPremulC_8u_C1IR_Ctx(Npp8u nAlpha1, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx) {
  return AlphaPremulConstExecutor<Npp8u, 1, AlphaPremulConstSimpleOp<Npp8u>>::executeInplace(
      pSrcDst, nSrcDstStep, oSizeROI, nAlpha1, nppStreamCtx.hStream);
}

NppStatus nppiAlphaPremulC_8u_C1IR(Npp8u nAlpha1, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAlphaPremulC_8u_C1IR_Ctx(nAlpha1, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// 8u C3 versions
NppStatus nppiAlphaPremulC_8u_C3R_Ctx(const Npp8u *pSrc1, int nSrc1Step, Npp8u nAlpha1, Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return AlphaPremulConstExecutor<Npp8u, 3, AlphaPremulConstSimpleOp<Npp8u>>::execute(
      pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nAlpha1, nppStreamCtx.hStream);
}

NppStatus nppiAlphaPremulC_8u_C3R(const Npp8u *pSrc1, int nSrc1Step, Npp8u nAlpha1, Npp8u *pDst, int nDstStep,
                                  NppiSize oSizeROI) {
  return nppiAlphaPremulC_8u_C3R_Ctx(pSrc1, nSrc1Step, nAlpha1, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAlphaPremulC_8u_C3IR_Ctx(Npp8u nAlpha1, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx) {
  return AlphaPremulConstExecutor<Npp8u, 3, AlphaPremulConstSimpleOp<Npp8u>>::executeInplace(
      pSrcDst, nSrcDstStep, oSizeROI, nAlpha1, nppStreamCtx.hStream);
}

NppStatus nppiAlphaPremulC_8u_C3IR(Npp8u nAlpha1, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAlphaPremulC_8u_C3IR_Ctx(nAlpha1, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// 8u C4 versions
NppStatus nppiAlphaPremulC_8u_C4R_Ctx(const Npp8u *pSrc1, int nSrc1Step, Npp8u nAlpha1, Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return AlphaPremulConstExecutor<Npp8u, 4, AlphaPremulConstSimpleOp<Npp8u>>::execute(
      pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nAlpha1, nppStreamCtx.hStream);
}

NppStatus nppiAlphaPremulC_8u_C4R(const Npp8u *pSrc1, int nSrc1Step, Npp8u nAlpha1, Npp8u *pDst, int nDstStep,
                                  NppiSize oSizeROI) {
  return nppiAlphaPremulC_8u_C4R_Ctx(pSrc1, nSrc1Step, nAlpha1, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAlphaPremulC_8u_C4IR_Ctx(Npp8u nAlpha1, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx) {
  return AlphaPremulConstExecutor<Npp8u, 4, AlphaPremulConstSimpleOp<Npp8u>>::executeInplace(
      pSrcDst, nSrcDstStep, oSizeROI, nAlpha1, nppStreamCtx.hStream);
}

NppStatus nppiAlphaPremulC_8u_C4IR(Npp8u nAlpha1, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAlphaPremulC_8u_C4IR_Ctx(nAlpha1, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// 8u AC4 versions (same as C4 for constant alpha - alpha channel not involved)
NppStatus nppiAlphaPremulC_8u_AC4R_Ctx(const Npp8u *pSrc1, int nSrc1Step, Npp8u nAlpha1, Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return AlphaPremulConstExecutor<Npp8u, 4, AlphaPremulConstSimpleOp<Npp8u>>::execute(
      pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nAlpha1, nppStreamCtx.hStream);
}

NppStatus nppiAlphaPremulC_8u_AC4R(const Npp8u *pSrc1, int nSrc1Step, Npp8u nAlpha1, Npp8u *pDst, int nDstStep,
                                   NppiSize oSizeROI) {
  return nppiAlphaPremulC_8u_AC4R_Ctx(pSrc1, nSrc1Step, nAlpha1, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAlphaPremulC_8u_AC4IR_Ctx(Npp8u nAlpha1, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                        NppStreamContext nppStreamCtx) {
  return AlphaPremulConstExecutor<Npp8u, 4, AlphaPremulConstSimpleOp<Npp8u>>::executeInplace(
      pSrcDst, nSrcDstStep, oSizeROI, nAlpha1, nppStreamCtx.hStream);
}

NppStatus nppiAlphaPremulC_8u_AC4IR(Npp8u nAlpha1, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAlphaPremulC_8u_AC4IR_Ctx(nAlpha1, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// ============================================================================
// 16u versions
// ============================================================================

NppStatus nppiAlphaPremulC_16u_C1R_Ctx(const Npp16u *pSrc1, int nSrc1Step, Npp16u nAlpha1, Npp16u *pDst, int nDstStep,
                                       NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return AlphaPremulConstExecutor<Npp16u, 1, AlphaPremulConstSimpleOp<Npp16u>>::execute(
      pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nAlpha1, nppStreamCtx.hStream);
}

NppStatus nppiAlphaPremulC_16u_C1R(const Npp16u *pSrc1, int nSrc1Step, Npp16u nAlpha1, Npp16u *pDst, int nDstStep,
                                   NppiSize oSizeROI) {
  return nppiAlphaPremulC_16u_C1R_Ctx(pSrc1, nSrc1Step, nAlpha1, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAlphaPremulC_16u_C1IR_Ctx(Npp16u nAlpha1, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                        NppStreamContext nppStreamCtx) {
  return AlphaPremulConstExecutor<Npp16u, 1, AlphaPremulConstSimpleOp<Npp16u>>::executeInplace(
      pSrcDst, nSrcDstStep, oSizeROI, nAlpha1, nppStreamCtx.hStream);
}

NppStatus nppiAlphaPremulC_16u_C1IR(Npp16u nAlpha1, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAlphaPremulC_16u_C1IR_Ctx(nAlpha1, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAlphaPremulC_16u_C3R_Ctx(const Npp16u *pSrc1, int nSrc1Step, Npp16u nAlpha1, Npp16u *pDst, int nDstStep,
                                       NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return AlphaPremulConstExecutor<Npp16u, 3, AlphaPremulConstSimpleOp<Npp16u>>::execute(
      pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nAlpha1, nppStreamCtx.hStream);
}

NppStatus nppiAlphaPremulC_16u_C3R(const Npp16u *pSrc1, int nSrc1Step, Npp16u nAlpha1, Npp16u *pDst, int nDstStep,
                                   NppiSize oSizeROI) {
  return nppiAlphaPremulC_16u_C3R_Ctx(pSrc1, nSrc1Step, nAlpha1, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAlphaPremulC_16u_C3IR_Ctx(Npp16u nAlpha1, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                        NppStreamContext nppStreamCtx) {
  return AlphaPremulConstExecutor<Npp16u, 3, AlphaPremulConstSimpleOp<Npp16u>>::executeInplace(
      pSrcDst, nSrcDstStep, oSizeROI, nAlpha1, nppStreamCtx.hStream);
}

NppStatus nppiAlphaPremulC_16u_C3IR(Npp16u nAlpha1, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAlphaPremulC_16u_C3IR_Ctx(nAlpha1, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAlphaPremulC_16u_C4R_Ctx(const Npp16u *pSrc1, int nSrc1Step, Npp16u nAlpha1, Npp16u *pDst, int nDstStep,
                                       NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return AlphaPremulConstExecutor<Npp16u, 4, AlphaPremulConstSimpleOp<Npp16u>>::execute(
      pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nAlpha1, nppStreamCtx.hStream);
}

NppStatus nppiAlphaPremulC_16u_C4R(const Npp16u *pSrc1, int nSrc1Step, Npp16u nAlpha1, Npp16u *pDst, int nDstStep,
                                   NppiSize oSizeROI) {
  return nppiAlphaPremulC_16u_C4R_Ctx(pSrc1, nSrc1Step, nAlpha1, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAlphaPremulC_16u_C4IR_Ctx(Npp16u nAlpha1, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                        NppStreamContext nppStreamCtx) {
  return AlphaPremulConstExecutor<Npp16u, 4, AlphaPremulConstSimpleOp<Npp16u>>::executeInplace(
      pSrcDst, nSrcDstStep, oSizeROI, nAlpha1, nppStreamCtx.hStream);
}

NppStatus nppiAlphaPremulC_16u_C4IR(Npp16u nAlpha1, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAlphaPremulC_16u_C4IR_Ctx(nAlpha1, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAlphaPremulC_16u_AC4R_Ctx(const Npp16u *pSrc1, int nSrc1Step, Npp16u nAlpha1, Npp16u *pDst, int nDstStep,
                                        NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return AlphaPremulConstExecutor<Npp16u, 4, AlphaPremulConstSimpleOp<Npp16u>>::execute(
      pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nAlpha1, nppStreamCtx.hStream);
}

NppStatus nppiAlphaPremulC_16u_AC4R(const Npp16u *pSrc1, int nSrc1Step, Npp16u nAlpha1, Npp16u *pDst, int nDstStep,
                                    NppiSize oSizeROI) {
  return nppiAlphaPremulC_16u_AC4R_Ctx(pSrc1, nSrc1Step, nAlpha1, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAlphaPremulC_16u_AC4IR_Ctx(Npp16u nAlpha1, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx) {
  return AlphaPremulConstExecutor<Npp16u, 4, AlphaPremulConstSimpleOp<Npp16u>>::executeInplace(
      pSrcDst, nSrcDstStep, oSizeROI, nAlpha1, nppStreamCtx.hStream);
}

NppStatus nppiAlphaPremulC_16u_AC4IR(Npp16u nAlpha1, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAlphaPremulC_16u_AC4IR_Ctx(nAlpha1, pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

// ============================================================================
// AlphaPremul - Pixel Alpha Premultiplication (AC4 format only)
// Uses AlphaPremulAC4Executor with AlphaPremulPixelOp
// ============================================================================

NppStatus nppiAlphaPremul_8u_AC4R_Ctx(const Npp8u *pSrc1, int nSrc1Step, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx) {
  return AlphaPremulAC4Executor<Npp8u, AlphaPremulPixelOp<Npp8u>>::execute(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI,
                                                                           nppStreamCtx.hStream);
}

NppStatus nppiAlphaPremul_8u_AC4R(const Npp8u *pSrc1, int nSrc1Step, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  return nppiAlphaPremul_8u_AC4R_Ctx(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAlphaPremul_8u_AC4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx) {
  return AlphaPremulAC4Executor<Npp8u, AlphaPremulPixelOp<Npp8u>>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI,
                                                                                  nppStreamCtx.hStream);
}

NppStatus nppiAlphaPremul_8u_AC4IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAlphaPremul_8u_AC4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAlphaPremul_16u_AC4R_Ctx(const Npp16u *pSrc1, int nSrc1Step, Npp16u *pDst, int nDstStep,
                                       NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return AlphaPremulAC4Executor<Npp16u, AlphaPremulPixelOp<Npp16u>>::execute(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI,
                                                                             nppStreamCtx.hStream);
}

NppStatus nppiAlphaPremul_16u_AC4R(const Npp16u *pSrc1, int nSrc1Step, Npp16u *pDst, int nDstStep, NppiSize oSizeROI) {
  return nppiAlphaPremul_16u_AC4R_Ctx(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, getDefaultStreamContext());
}

NppStatus nppiAlphaPremul_16u_AC4IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                        NppStreamContext nppStreamCtx) {
  return AlphaPremulAC4Executor<Npp16u, AlphaPremulPixelOp<Npp16u>>::executeInplace(pSrcDst, nSrcDstStep, oSizeROI,
                                                                                    nppStreamCtx.hStream);
}

NppStatus nppiAlphaPremul_16u_AC4IR(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  return nppiAlphaPremul_16u_AC4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, getDefaultStreamContext());
}

} // extern "C"
