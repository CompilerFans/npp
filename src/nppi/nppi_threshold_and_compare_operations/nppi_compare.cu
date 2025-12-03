#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template <typename T>
__device__ inline Npp8u performCompare(T src1, T src2, NppCmpOp op) {
  bool result = false;
  switch (op) {
  case NPP_CMP_LESS:
    result = (src1 < src2);
    break;
  case NPP_CMP_LESS_EQ:
    result = (src1 <= src2);
    break;
  case NPP_CMP_EQ:
    result = (src1 == src2);
    break;
  case NPP_CMP_GREATER_EQ:
    result = (src1 >= src2);
    break;
  case NPP_CMP_GREATER:
    result = (src1 > src2);
    break;
  default:
    result = false;
  }
  return result ? 0xFF : 0x00;
}

__global__ void nppiCompare_8u_C1R_kernel(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                                          Npp8u *pDst, int nDstStep, int width, int height, NppCmpOp eCompOp) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *src1_row = (const Npp8u *)((const char *)pSrc1 + y * nSrc1Step);
  const Npp8u *src2_row = (const Npp8u *)((const char *)pSrc2 + y * nSrc2Step);
  Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

  dst_row[x] = performCompare(src1_row[x], src2_row[x], eCompOp);
}

__global__ void nppiCompare_16u_C1R_kernel(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step,
                                           Npp8u *pDst, int nDstStep, int width, int height, NppCmpOp eCompOp) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp16u *src1_row = (const Npp16u *)((const char *)pSrc1 + y * nSrc1Step);
  const Npp16u *src2_row = (const Npp16u *)((const char *)pSrc2 + y * nSrc2Step);
  Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

  dst_row[x] = performCompare(src1_row[x], src2_row[x], eCompOp);
}

__global__ void nppiCompare_32f_C1R_kernel(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step,
                                           Npp8u *pDst, int nDstStep, int width, int height, NppCmpOp eCompOp) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp32f *src1_row = (const Npp32f *)((const char *)pSrc1 + y * nSrc1Step);
  const Npp32f *src2_row = (const Npp32f *)((const char *)pSrc2 + y * nSrc2Step);
  Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

  dst_row[x] = performCompare(src1_row[x], src2_row[x], eCompOp);
}

extern "C" {

NppStatus nppiCompare_8u_C1R_Ctx_impl(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                                      Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppCmpOp eCompOp,
                                      NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiCompare_8u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI.width, oSizeROI.height, eCompOp);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

NppStatus nppiCompare_16u_C1R_Ctx_impl(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step,
                                       Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppCmpOp eCompOp,
                                       NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiCompare_16u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI.width, oSizeROI.height, eCompOp);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

NppStatus nppiCompare_32f_C1R_Ctx_impl(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step,
                                       Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppCmpOp eCompOp,
                                       NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiCompare_32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI.width, oSizeROI.height, eCompOp);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}
}
