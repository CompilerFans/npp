#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void nppiSwapChannels_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                                int width, int height, int order0, int order1, int order2) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
  Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

  int src_idx = x * 3;
  int dst_idx = x * 3;

  Npp8u src0 = src_row[src_idx + 0];
  Npp8u src1 = src_row[src_idx + 1];
  Npp8u src2 = src_row[src_idx + 2];

  Npp8u values[3] = {src0, src1, src2};
  dst_row[dst_idx + 0] = values[order0];
  dst_row[dst_idx + 1] = values[order1];
  dst_row[dst_idx + 2] = values[order2];
}

__global__ void nppiSwapChannels_8u_C4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                                int width, int height, int order0, int order1, int order2, int order3) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
  Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

  int src_idx = x * 4;
  int dst_idx = x * 4;

  Npp8u src0 = src_row[src_idx + 0];
  Npp8u src1 = src_row[src_idx + 1];
  Npp8u src2 = src_row[src_idx + 2];
  Npp8u src3 = src_row[src_idx + 3];

  Npp8u values[4] = {src0, src1, src2, src3};
  dst_row[dst_idx + 0] = values[order0];
  dst_row[dst_idx + 1] = values[order1];
  dst_row[dst_idx + 2] = values[order2];
  dst_row[dst_idx + 3] = values[order3];
}

__global__ void nppiSwapChannels_32f_C3R_kernel(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                                 int width, int height, int order0, int order1, int order2) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp32f *src_row = (const Npp32f *)((const char *)pSrc + y * nSrcStep);
  Npp32f *dst_row = (Npp32f *)((char *)pDst + y * nDstStep);

  int src_idx = x * 3;
  int dst_idx = x * 3;

  Npp32f src0 = src_row[src_idx + 0];
  Npp32f src1 = src_row[src_idx + 1];
  Npp32f src2 = src_row[src_idx + 2];

  Npp32f values[3] = {src0, src1, src2};
  dst_row[dst_idx + 0] = values[order0];
  dst_row[dst_idx + 1] = values[order1];
  dst_row[dst_idx + 2] = values[order2];
}

extern "C" {

NppStatus nppiSwapChannels_8u_C3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                           NppiSize oSizeROI, const int aDstOrder[3],
                                           NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiSwapChannels_8u_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, aDstOrder[0], aDstOrder[1], aDstOrder[2]);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

NppStatus nppiSwapChannels_8u_C4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                           NppiSize oSizeROI, const int aDstOrder[4],
                                           NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiSwapChannels_8u_C4R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, aDstOrder[0], aDstOrder[1], aDstOrder[2], aDstOrder[3]);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

NppStatus nppiSwapChannels_32f_C3R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                            NppiSize oSizeROI, const int aDstOrder[3],
                                            NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiSwapChannels_32f_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, aDstOrder[0], aDstOrder[1], aDstOrder[2]);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}
}
