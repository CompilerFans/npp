#include "nppi_arithmetic_executor.h"
#include "nppi_arithmetic_ops.h"

using namespace nppi::arithmetic;

// ============================================================================
// MaxEvery Implementation  
// nppiMaxEvery: result = max(src1, src2) for each pixel
// ============================================================================

extern "C" {

// 8u versions
NppStatus nppiMaxEvery_8u_C1IR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return BinaryOperationExecutor<Npp8u, 1, MaxEveryOp<Npp8u>>::execute(
      pSrc, nSrcStep, pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, 0, nppStreamCtx.hStream);
}

NppStatus nppiMaxEvery_8u_C1IR(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = 0; // Default CUDA stream
  return nppiMaxEvery_8u_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, defaultCtx);
}

// 16u versions
NppStatus nppiMaxEvery_16u_C1IR_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return BinaryOperationExecutor<Npp16u, 1, MaxEveryOp<Npp16u>>::execute(
      pSrc, nSrcStep, pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, 0, nppStreamCtx.hStream);
}

NppStatus nppiMaxEvery_16u_C1IR(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = 0; // Default CUDA stream
  return nppiMaxEvery_16u_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, defaultCtx);
}

// 16s versions
NppStatus nppiMaxEvery_16s_C1IR_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return BinaryOperationExecutor<Npp16s, 1, MaxEveryOp<Npp16s>>::execute(
      pSrc, nSrcStep, pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, 0, nppStreamCtx.hStream);
}

NppStatus nppiMaxEvery_16s_C1IR(const Npp16s *pSrc, int nSrcStep, Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = 0; // Default CUDA stream
  return nppiMaxEvery_16s_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, defaultCtx);
}

// 32s versions
NppStatus nppiMaxEvery_32s_C1IR_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return BinaryOperationExecutor<Npp32s, 1, MaxEveryOp<Npp32s>>::execute(
      pSrc, nSrcStep, pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, 0, nppStreamCtx.hStream);
}

NppStatus nppiMaxEvery_32s_C1IR(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = 0; // Default CUDA stream
  return nppiMaxEvery_32s_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, defaultCtx);
}

// 32f versions
NppStatus nppiMaxEvery_32f_C1IR_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return BinaryOperationExecutor<Npp32f, 1, MaxEveryOp<Npp32f>>::execute(
      pSrc, nSrcStep, pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, 0, nppStreamCtx.hStream);
}

NppStatus nppiMaxEvery_32f_C1IR(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = 0; // Default CUDA stream
  return nppiMaxEvery_32f_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, defaultCtx);
}

} // extern "C"