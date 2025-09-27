#include "nppi_arithmetic_executor.h"
#include "nppi_arithmetic_ops.h"

using namespace nppi::arithmetic;

// ============================================================================
// AddSquare Implementation
// nppiAddSquare: result = src1 + (src2 * src2)
// ============================================================================

// Mixed type kernel for AddSquare operation
template <typename SrcType, typename DstType>
__global__ void addSquareMixedKernel(const SrcType *pSrc, int nSrcStep, DstType *pSrcDst, int nSrcDstStep,
                                     int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const SrcType *srcRow = (const SrcType *)((const char *)pSrc + y * nSrcStep);
    DstType *dstRow = (DstType *)((char *)pSrcDst + y * nSrcDstStep);

    // AddSquare: dst = dst + (src * src)
    DstType srcVal = static_cast<DstType>(srcRow[x]);
    dstRow[x] = dstRow[x] + (srcVal * srcVal);
  }
}

extern "C" {

// 8u32f versions (mixed type: 8u input, 32f output)
NppStatus nppiAddSquare_8u32f_C1IR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  // Validate parameters
  if (!pSrc || !pSrcDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  // Launch kernel
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, 
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  addSquareMixedKernel<Npp8u, Npp32f><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height);

  return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAddSquare_8u32f_C1IR(const Npp8u *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = 0; // Default CUDA stream
  return nppiAddSquare_8u32f_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, defaultCtx);
}

// 16u32f versions (mixed type: 16u input, 32f output)
NppStatus nppiAddSquare_16u32f_C1IR_Ctx(const Npp16u *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  // Validate parameters
  if (!pSrc || !pSrcDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  // Launch kernel
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, 
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  addSquareMixedKernel<Npp16u, Npp32f><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height);

  return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAddSquare_16u32f_C1IR(const Npp16u *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = 0; // Default CUDA stream
  return nppiAddSquare_16u32f_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, defaultCtx);
}

// 32f versions (same type: 32f input and output)
NppStatus nppiAddSquare_32f_C1IR_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return BinaryOperationExecutor<Npp32f, 1, AddSquareOp<Npp32f>>::execute(
      pSrcDst, nSrcDstStep, pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, 0, nppStreamCtx.hStream);
}

NppStatus nppiAddSquare_32f_C1IR(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = 0; // Default CUDA stream
  return nppiAddSquare_32f_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, defaultCtx);
}

// 8u versions (integer with scale factor)
NppStatus nppiAddSquare_8u_C1IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return BinaryOperationExecutor<Npp8u, 1, AddSquareOp<Npp8u>>::execute(
      pSrcDst, nSrcDstStep, pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx.hStream);
}

NppStatus nppiAddSquare_8u_C1IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, 
                                   NppiSize oSizeROI, int nScaleFactor) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = 0; // Default CUDA stream
  return nppiAddSquare_8u_C1IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, defaultCtx);
}

// 16u versions (integer with scale factor)
NppStatus nppiAddSquare_16u_C1IRSfs_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return BinaryOperationExecutor<Npp16u, 1, AddSquareOp<Npp16u>>::execute(
      pSrcDst, nSrcDstStep, pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx.hStream);
}

NppStatus nppiAddSquare_16u_C1IRSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, int nScaleFactor) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = 0; // Default CUDA stream
  return nppiAddSquare_16u_C1IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, defaultCtx);
}

} // extern "C"