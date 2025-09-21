#include "npp.h"
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>



// Kernel for 8-bit unsigned single channel absolute difference
__global__ void nppiAbsDiff_8u_C1R_kernel(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                                          Npp8u *pDst, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *src1_row = (const Npp8u *)((const char *)pSrc1 + y * nSrc1Step);
    const Npp8u *src2_row = (const Npp8u *)((const char *)pSrc2 + y * nSrc2Step);
    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

    int val1 = src1_row[x];
    int val2 = src2_row[x];
    dst_row[x] = abs(val1 - val2);
  }
}

// Kernel for 8-bit unsigned three channel absolute difference
__global__ void nppiAbsDiff_8u_C3R_kernel(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                                          Npp8u *pDst, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *src1_row = (const Npp8u *)((const char *)pSrc1 + y * nSrc1Step);
    const Npp8u *src2_row = (const Npp8u *)((const char *)pSrc2 + y * nSrc2Step);
    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

    int idx = x * 3;
    for (int c = 0; c < 3; c++) {
      int val1 = src1_row[idx + c];
      int val2 = src2_row[idx + c];
      dst_row[idx + c] = abs(val1 - val2);
    }
  }
}

// Kernel for 32-bit float single channel absolute difference
__global__ void nppiAbsDiff_32f_C1R_kernel(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step,
                                           Npp32f *pDst, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp32f *src1_row = (const Npp32f *)((const char *)pSrc1 + y * nSrc1Step);
    const Npp32f *src2_row = (const Npp32f *)((const char *)pSrc2 + y * nSrc2Step);
    Npp32f *dst_row = (Npp32f *)((char *)pDst + y * nDstStep);

    float val1 = src1_row[x];
    float val2 = src2_row[x];
    dst_row[x] = fabsf(val1 - val2);
  }
}

extern "C" {

// 8-bit unsigned single channel implementation
NppStatus nppiAbsDiff_8u_C1R_Ctx_impl(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                      int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiAbsDiff_8u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI.width, oSizeROI.height);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 8-bit unsigned three channel implementation
NppStatus nppiAbsDiff_8u_C3R_Ctx_impl(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                      int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiAbsDiff_8u_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI.width, oSizeROI.height);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 32-bit float single channel implementation
NppStatus nppiAbsDiff_32f_C1R_Ctx_impl(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step,
                                       Npp32f *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiAbsDiff_32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI.width, oSizeROI.height);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

} // extern "C"
