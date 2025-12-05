#include "npp.h"
#include <cuda_runtime.h>

// ==============================================================================
// GPU Kernels for Sum Operations
// ==============================================================================

__global__ void nppsSum_32f_kernel_impl(const Npp32f *pSrc, size_t nLength, Npp32f *pPartialSums) {
  __shared__ Npp32f sdata[256];

  size_t tid = threadIdx.x;
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (i < nLength) ? pSrc[i] : 0.0f;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    pPartialSums[blockIdx.x] = sdata[0];
  }
}

__global__ void nppsSum_32f_final_kernel(const Npp32f *pPartialSums, size_t numBlocks, Npp32f *pSum) {
  __shared__ Npp32f sdata[256];

  size_t tid = threadIdx.x;
  size_t i = threadIdx.x;

  sdata[tid] = 0.0f;

  while (i < numBlocks) {
    sdata[tid] += pPartialSums[i];
    i += blockDim.x;
  }
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    *pSum = sdata[0];
  }
}

__global__ void nppsSum_32fc_kernel_impl(const Npp32fc *pSrc, size_t nLength, Npp32fc *pPartialSums) {
  __shared__ Npp32fc sdata[256];

  size_t tid = threadIdx.x;
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < nLength) {
    sdata[tid] = pSrc[i];
  } else {
    sdata[tid].re = 0.0f;
    sdata[tid].im = 0.0f;
  }
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid].re += sdata[tid + s].re;
      sdata[tid].im += sdata[tid + s].im;
    }
    __syncthreads();
  }

  if (tid == 0) {
    pPartialSums[blockIdx.x] = sdata[0];
  }
}

__global__ void nppsSum_32fc_final_kernel(const Npp32fc *pPartialSums, size_t numBlocks, Npp32fc *pSum) {
  __shared__ Npp32fc sdata[256];

  size_t tid = threadIdx.x;
  size_t i = threadIdx.x;

  sdata[tid].re = 0.0f;
  sdata[tid].im = 0.0f;

  while (i < numBlocks) {
    sdata[tid].re += pPartialSums[i].re;
    sdata[tid].im += pPartialSums[i].im;
    i += blockDim.x;
  }
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid].re += sdata[tid + s].re;
      sdata[tid].im += sdata[tid + s].im;
    }
    __syncthreads();
  }

  if (tid == 0) {
    *pSum = sdata[0];
  }
}

// ==============================================================================
// Kernel Launch Functions
// ==============================================================================

extern "C" {

cudaError_t nppsSum_32f_kernel(const Npp32f *pSrc, size_t nLength, Npp32f *pSum, Npp8u *pDeviceBuffer,
                               cudaStream_t stream) {
  const int blockSize = 256;
  const size_t gridSize = (nLength + blockSize - 1) / blockSize;

  Npp32f *partialSums = reinterpret_cast<Npp32f *>(pDeviceBuffer);

  nppsSum_32f_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc, nLength, partialSums);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return error;
  }

  nppsSum_32f_final_kernel<<<1, 256, 0, stream>>>(partialSums, gridSize, pSum);

  return cudaGetLastError();
}

cudaError_t nppsSum_32fc_kernel(const Npp32fc *pSrc, size_t nLength, Npp32fc *pSum, Npp8u *pDeviceBuffer,
                                cudaStream_t stream) {
  const int blockSize = 256;
  const size_t gridSize = (nLength + blockSize - 1) / blockSize;

  Npp32fc *partialSums = reinterpret_cast<Npp32fc *>(pDeviceBuffer);

  nppsSum_32fc_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc, nLength, partialSums);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return error;
  }

  nppsSum_32fc_final_kernel<<<1, 256, 0, stream>>>(partialSums, gridSize, pSum);

  return cudaGetLastError();
}
}
