#include "nppi_arithmetic_executor.h"
#include "nppi_arithmetic_ops.h"

using namespace nppi::arithmetic;

// ============================================================================
// Masked AddProduct Kernel
// Updates destination only when mask is non-zero
// ============================================================================

template <typename SrcType, typename DstType>
__global__ void addProductMaskedKernel(const SrcType *pSrc1, int nSrc1Step, const SrcType *pSrc2, int nSrc2Step,
                                       const Npp8u *pMask, int nMaskStep, DstType *pSrcDst, int nSrcDstStep, int width,
                                       int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *maskRow = (const Npp8u *)((const char *)pMask + y * nMaskStep);

    // Only update if mask is non-zero
    if (maskRow[x] != 0) {
      const SrcType *src1Row = (const SrcType *)((const char *)pSrc1 + y * nSrc1Step);
      const SrcType *src2Row = (const SrcType *)((const char *)pSrc2 + y * nSrc2Step);
      DstType *dstRow = (DstType *)((char *)pSrcDst + y * nSrcDstStep);

      // AddProduct: dst = dst + (src1 * src2)
      DstType val1 = static_cast<DstType>(src1Row[x]);
      DstType val2 = static_cast<DstType>(src2Row[x]);
      dstRow[x] = dstRow[x] + (val1 * val2);
    }
  }
}

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
      pSrcDst, nSrcDstStep, pSrc1, nSrc1Step, pSrc2, nSrc2Step, pSrcDst, nSrcDstStep, oSizeROI, 0,
      nppStreamCtx.hStream);
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
      pSrcDst, nSrcDstStep, pSrc1, nSrc1Step, pSrc2, nSrc2Step, pSrcDst, nSrcDstStep, oSizeROI, 0,
      nppStreamCtx.hStream);
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
  return TernaryOperationExecutor<Npp32f, 1, AddProductOp<Npp32f>>::execute(pSrcDst, nSrcDstStep, pSrc1, nSrc1Step,
                                                                            pSrc2, nSrc2Step, pSrcDst, nSrcDstStep,
                                                                            oSizeROI, 0, nppStreamCtx.hStream);
}

NppStatus nppiAddProduct_32f_C1IR(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step,
                                  Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = 0; // Default CUDA stream
  return nppiAddProduct_32f_C1IR_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pSrcDst, nSrcDstStep, oSizeROI, defaultCtx);
}

// ============================================================================
// Masked AddProduct Functions (_C1IMR)
// Updates destination only when mask is non-zero
// ============================================================================

// 8u32f masked version
NppStatus nppiAddProduct_8u32f_C1IMR_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                                         const Npp8u *pMask, int nMaskStep, Npp32f *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  // Validate parameters
  if (!pSrc1 || !pSrc2 || !pMask || !pSrcDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  // Launch kernel
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  addProductMaskedKernel<Npp8u, Npp32f><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, pSrc2, nSrc2Step, pMask, nMaskStep, pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height);

  return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAddProduct_8u32f_C1IMR(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                                     const Npp8u *pMask, int nMaskStep, Npp32f *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = 0;
  return nppiAddProduct_8u32f_C1IMR_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pMask, nMaskStep, pSrcDst, nSrcDstStep,
                                        oSizeROI, defaultCtx);
}

// 16u32f masked version
NppStatus nppiAddProduct_16u32f_C1IMR_Ctx(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step,
                                          const Npp8u *pMask, int nMaskStep, Npp32f *pSrcDst, int nSrcDstStep,
                                          NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  // Validate parameters
  if (!pSrc1 || !pSrc2 || !pMask || !pSrcDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  // Launch kernel
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  addProductMaskedKernel<Npp16u, Npp32f><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, pSrc2, nSrc2Step, pMask, nMaskStep, pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height);

  return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAddProduct_16u32f_C1IMR(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step,
                                      const Npp8u *pMask, int nMaskStep, Npp32f *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = 0;
  return nppiAddProduct_16u32f_C1IMR_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pMask, nMaskStep, pSrcDst, nSrcDstStep,
                                         oSizeROI, defaultCtx);
}

// 32f masked version
NppStatus nppiAddProduct_32f_C1IMR_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step,
                                       const Npp8u *pMask, int nMaskStep, Npp32f *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  // Validate parameters
  if (!pSrc1 || !pSrc2 || !pMask || !pSrcDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  // Launch kernel
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  addProductMaskedKernel<Npp32f, Npp32f><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, pSrc2, nSrc2Step, pMask, nMaskStep, pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height);

  return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAddProduct_32f_C1IMR(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step,
                                   const Npp8u *pMask, int nMaskStep, Npp32f *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = 0;
  return nppiAddProduct_32f_C1IMR_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pMask, nMaskStep, pSrcDst, nSrcDstStep,
                                      oSizeROI, defaultCtx);
}

// ============================================================================
// 16f AddProduct Implementation
// ============================================================================

__global__ void addProduct16fKernel(const Npp16f *pSrc1, int nSrc1Step, const Npp16f *pSrc2, int nSrc2Step,
                                    Npp16f *pSrcDst, int nSrcDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp16f *src1Row = (const Npp16f *)((const char *)pSrc1 + y * nSrc1Step);
    const Npp16f *src2Row = (const Npp16f *)((const char *)pSrc2 + y * nSrc2Step);
    Npp16f *dstRow = (Npp16f *)((char *)pSrcDst + y * nSrcDstStep);

    __half a = npp16f_to_half(src1Row[x]);
    __half b = npp16f_to_half(src2Row[x]);
    __half d = npp16f_to_half(dstRow[x]);
    dstRow[x] = half_to_npp16f(__hadd(d, __hmul(a, b)));
  }
}

NppStatus nppiAddProduct_16f_C1IR_Ctx(const Npp16f *pSrc1, int nSrc1Step, const Npp16f *pSrc2, int nSrc2Step,
                                      Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx) {
  NppStatus status = validateUnaryParameters(pSrc1, nSrc1Step, pSrcDst, nSrcDstStep, oSizeROI, 1);
  if (status != NPP_NO_ERROR)
    return status;
  if (!pSrc2)
    return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0)
    return NPP_NO_ERROR;

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  addProduct16fKernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pSrcDst,
                                                                        nSrcDstStep, oSizeROI.width, oSizeROI.height);

  return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAddProduct_16f_C1IR(const Npp16f *pSrc1, int nSrc1Step, const Npp16f *pSrc2, int nSrc2Step,
                                  Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  return nppiAddProduct_16f_C1IR_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pSrcDst, nSrcDstStep, oSizeROI, ctx);
}

} // extern "C"