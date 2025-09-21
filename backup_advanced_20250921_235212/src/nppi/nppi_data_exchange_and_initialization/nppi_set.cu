#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Implementation file

// Set kernel for 8-bit single channel
__global__ void nppiSet_8u_C1R_kernel(Npp8u nValue, Npp8u *pDst, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);
  dst_row[x] = nValue;
}

// Set kernel for 8-bit three channel
__global__ void nppiSet_8u_C3R_kernel(const Npp8u *aValue, Npp8u *pDst, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);
  int idx = x * 3;

  dst_row[idx + 0] = aValue[0];
  dst_row[idx + 1] = aValue[1];
  dst_row[idx + 2] = aValue[2];
}

// Set kernel for 32-bit float single channel
__global__ void nppiSet_32f_C1R_kernel(Npp32f nValue, Npp32f *pDst, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  Npp32f *dst_row = (Npp32f *)((char *)pDst + y * nDstStep);
  dst_row[x] = nValue;
}

extern "C" {

// 8-bit unsigned single channel implementation
NppStatus nppiSet_8u_C1R_Ctx_impl(Npp8u nValue, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiSet_8u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(nValue, pDst, nDstStep, oSizeROI.width,
                                                                          oSizeROI.height);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 8-bit unsigned three channel implementation
NppStatus nppiSet_8u_C3R_Ctx_impl(const Npp8u aValue[3], Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
  // Copy value array to device memory
  Npp8u *d_aValue;
  cudaError_t cudaStatus = cudaMalloc(&d_aValue, 3 * sizeof(Npp8u));
  if (cudaStatus != cudaSuccess) {
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  cudaStatus = cudaMemcpyAsync(d_aValue, aValue, 3 * sizeof(Npp8u), cudaMemcpyHostToDevice, nppStreamCtx.hStream);
  if (cudaStatus != cudaSuccess) {
    cudaFree(d_aValue);
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiSet_8u_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(d_aValue, pDst, nDstStep, oSizeROI.width,
                                                                          oSizeROI.height);

  cudaStatus = cudaGetLastError();

  // Sync stream and free memory
  cudaStreamSynchronize(nppStreamCtx.hStream);
  cudaFree(d_aValue);

  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 32-bit float single channel implementation
NppStatus nppiSet_32f_C1R_Ctx_impl(Npp32f nValue, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiSet_32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(nValue, pDst, nDstStep, oSizeROI.width,
                                                                           oSizeROI.height);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}
}