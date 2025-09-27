#include "nppi_arithmetic_executor.h"
#include "nppi_arithmetic_ops.h"

using namespace nppi::arithmetic;

// ============================================================================
// AddProduct Implementation
// nppiAddProduct: result = dst + (src1 * src2)
// ============================================================================

// 8u32f versions
extern "C" {

NppStatus nppiAddProduct_8u32f_C1IR_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                                         Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx) {
  // Use mixed type executor: dst(32f) + src1(8u) * src2(8u)
  return MixedTernaryOperationExecutor<Npp8u, Npp8u, Npp32f, 1, AddProductOp<Npp32f>>::execute(
      pSrcDst, nSrcDstStep, pSrc1, nSrc1Step, pSrc2, nSrc2Step,
      pSrcDst, nSrcDstStep, oSizeROI, 0, nppStreamCtx.hStream);
}

NppStatus nppiAddProduct_8u32f_C1IR(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                                    Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = 0; // Default CUDA stream
  return nppiAddProduct_8u32f_C1IR_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pSrcDst, nSrcDstStep, oSizeROI, defaultCtx);
}

// 16u32f versions
NppStatus nppiAddProduct_16u32f_C1IR_Ctx(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step,
                                         Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx) {
  // Use mixed type executor: dst(32f) + src1(16u) * src2(16u)
  return MixedTernaryOperationExecutor<Npp16u, Npp16u, Npp32f, 1, AddProductOp<Npp32f>>::execute(
      pSrcDst, nSrcDstStep, pSrc1, nSrc1Step, pSrc2, nSrc2Step,
      pSrcDst, nSrcDstStep, oSizeROI, 0, nppStreamCtx.hStream);
}

NppStatus nppiAddProduct_16u32f_C1IR(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step,
                                     Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = 0; // Default CUDA stream
  return nppiAddProduct_16u32f_C1IR_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pSrcDst, nSrcDstStep, oSizeROI, defaultCtx);
}

// 32f versions
NppStatus nppiAddProduct_32f_C1IR_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step,
                                      Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx) {
  return TernaryOperationExecutor<Npp32f, 1, AddProductOp<Npp32f>>::execute(
      pSrcDst, nSrcDstStep, pSrc1, nSrc1Step, pSrc2, nSrc2Step,
      pSrcDst, nSrcDstStep, oSizeROI, 0, nppStreamCtx.hStream);
}

NppStatus nppiAddProduct_32f_C1IR(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step,
                                  Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = 0; // Default CUDA stream
  return nppiAddProduct_32f_C1IR_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pSrcDst, nSrcDstStep, oSizeROI, defaultCtx);
}

} // extern "C"