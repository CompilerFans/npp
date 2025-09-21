#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>



// Device function for comparison operations
__device__ inline bool performComparison(float src, float constant, NppCmpOp op) {
  switch (op) {
  case NPP_CMP_LESS:
    return src < constant;
  case NPP_CMP_LESS_EQ:
    return src <= constant;
  case NPP_CMP_EQ:
    return src == constant;
  case NPP_CMP_GREATER_EQ:
    return src >= constant;
  case NPP_CMP_GREATER:
    return src > constant;
  default:
    return false;
  }
}

// Kernel for 8-bit unsigned single channel compare with constant
__global__ void nppiCompareC_8u_C1R_kernel(const Npp8u *pSrc, int nSrcStep, const Npp8u nConstant, Npp8u *pDst,
                                           int nDstStep, int width, int height, NppCmpOp eComparisonOperation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

    Npp8u src_val = src_row[x];
    bool result = performComparison((float)src_val, (float)nConstant, eComparisonOperation);
    dst_row[x] = result ? 255 : 0; // NPP uses 255 for true, 0 for false
  }
}

// Kernel for 16-bit signed single channel compare with constant
__global__ void nppiCompareC_16s_C1R_kernel(const Npp16s *pSrc, int nSrcStep, const Npp16s nConstant, Npp8u *pDst,
                                            int nDstStep, int width, int height, NppCmpOp eComparisonOperation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp16s *src_row = (const Npp16s *)((const char *)pSrc + y * nSrcStep);
    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

    Npp16s src_val = src_row[x];
    bool result = performComparison((float)src_val, (float)nConstant, eComparisonOperation);
    dst_row[x] = result ? 255 : 0;
  }
}

// Kernel for 32-bit float single channel compare with constant
__global__ void nppiCompareC_32f_C1R_kernel(const Npp32f *pSrc, int nSrcStep, const Npp32f nConstant, Npp8u *pDst,
                                            int nDstStep, int width, int height, NppCmpOp eComparisonOperation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp32f *src_row = (const Npp32f *)((const char *)pSrc + y * nSrcStep);
    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

    Npp32f src_val = src_row[x];
    bool result = performComparison(src_val, nConstant, eComparisonOperation);
    dst_row[x] = result ? 255 : 0;
  }
}

extern "C" {

// 8-bit unsigned single channel implementation
NppStatus nppiCompareC_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, const Npp8u nConstant, Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI, NppCmpOp eComparisonOperation,
                                       NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiCompareC_8u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI.width, oSizeROI.height, eComparisonOperation);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 16-bit signed single channel implementation
NppStatus nppiCompareC_16s_C1R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, const Npp16s nConstant, Npp8u *pDst,
                                        int nDstStep, NppiSize oSizeROI, NppCmpOp eComparisonOperation,
                                        NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiCompareC_16s_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI.width, oSizeROI.height, eComparisonOperation);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 32-bit float single channel implementation
NppStatus nppiCompareC_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, const Npp32f nConstant, Npp8u *pDst,
                                        int nDstStep, NppiSize oSizeROI, NppCmpOp eComparisonOperation,
                                        NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiCompareC_32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI.width, oSizeROI.height, eComparisonOperation);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

} // extern "C"
