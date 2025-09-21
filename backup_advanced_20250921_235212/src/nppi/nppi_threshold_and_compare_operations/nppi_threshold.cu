#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Implementation file

// Device function for threshold operation
template <typename T> __device__ inline T performThreshold(T src, T threshold, NppCmpOp op) {
  switch (op) {
  case NPP_CMP_LESS:
    return (src < threshold) ? threshold : src;
  case NPP_CMP_GREATER:
    return (src > threshold) ? threshold : src;
  default:
    return src;
  }
}

// Kernel for 8-bit unsigned single channel threshold (non-inplace)
__global__ void nppiThreshold_8u_C1R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                            int height, Npp8u nThreshold, NppCmpOp eComparisonOperation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

    dst_row[x] = performThreshold(src_row[x], nThreshold, eComparisonOperation);
  }
}

// Kernel for 8-bit unsigned single channel threshold (inplace)
__global__ void nppiThreshold_8u_C1IR_kernel(Npp8u *pSrcDst, int nSrcDstStep, int width, int height, Npp8u nThreshold,
                                             NppCmpOp eComparisonOperation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    Npp8u *row = (Npp8u *)((char *)pSrcDst + y * nSrcDstStep);
    row[x] = performThreshold(row[x], nThreshold, eComparisonOperation);
  }
}

// Kernel for 32-bit float single channel threshold (non-inplace)
__global__ void nppiThreshold_32f_C1R_kernel(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, int width,
                                             int height, Npp32f nThreshold, NppCmpOp eComparisonOperation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp32f *src_row = (const Npp32f *)((const char *)pSrc + y * nSrcStep);
    Npp32f *dst_row = (Npp32f *)((char *)pDst + y * nDstStep);

    dst_row[x] = performThreshold(src_row[x], nThreshold, eComparisonOperation);
  }
}

extern "C" {

// 8-bit unsigned single channel threshold implementation (non-inplace)
NppStatus nppiThreshold_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                        const Npp8u nThreshold, NppCmpOp eComparisonOperation,
                                        NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiThreshold_8u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nThreshold, eComparisonOperation);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 8-bit unsigned single channel threshold implementation (inplace)
NppStatus nppiThreshold_8u_C1IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp8u nThreshold,
                                         NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiThreshold_8u_C1IR_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height, nThreshold, eComparisonOperation);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 32-bit float single channel threshold implementation (non-inplace)
NppStatus nppiThreshold_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                         NppiSize oSizeROI, const Npp32f nThreshold, NppCmpOp eComparisonOperation,
                                         NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiThreshold_32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nThreshold, eComparisonOperation);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}
}