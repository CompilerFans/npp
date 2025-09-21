#include "npp.h"
#include <cuda_runtime.h>

/**
 * NPPS Statistics Functions CUDA Kernels - Sum Functions
 * GPU kernels for 1D signal summation operations using reduction.
 */

// ==============================================================================
// CUDA Kernels for Sum Operations
// ==============================================================================

/**
 * kernel for 32-bit float signal sum using block-level reduction
 */
__global__ void nppsSum_32f_kernel_impl(const Npp32f *pSrc, size_t nLength, Npp32f *pPartialSums) {
  __shared__ Npp32f sdata[256];

  size_t tid = threadIdx.x;
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  // Load data into shared memory
  sdata[tid] = (i < nLength) ? pSrc[i] : 0.0f;
  __syncthreads();

  // Perform reduction in shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // Write result of this block to global memory
  if (tid == 0) {
    pPartialSums[blockIdx.x] = sdata[0];
  }
}

/**
 * kernel for final reduction of partial sums
 */
__global__ void nppsSum_32f_final_kernel(const Npp32f *pPartialSums, size_t numBlocks, Npp32f *pSum) {
  __shared__ Npp32f sdata[256];

  size_t tid = threadIdx.x;
  size_t i = threadIdx.x;

  // Initialize shared memory
  sdata[tid] = 0.0f;

  // Accumulate all partial sums with stride
  while (i < numBlocks) {
    sdata[tid] += pPartialSums[i];
    i += blockDim.x;
  }
  __syncthreads();

  // Perform final reduction within the block
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // Write final result
  if (tid == 0) {
    *pSum = sdata[0];
  }
}

/**
 * kernel for 32-bit float complex signal sum using block-level reduction
 */
__global__ void nppsSum_32fc_kernel_impl(const Npp32fc *pSrc, size_t nLength, Npp32fc *pPartialSums) {
  __shared__ Npp32fc sdata[256];

  size_t tid = threadIdx.x;
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  // Load data into shared memory
  if (i < nLength) {
    sdata[tid] = pSrc[i];
  } else {
    sdata[tid].re = 0.0f;
    sdata[tid].im = 0.0f;
  }
  __syncthreads();

  // Perform reduction in shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid].re += sdata[tid + s].re;
      sdata[tid].im += sdata[tid + s].im;
    }
    __syncthreads();
  }

  // Write result of this block to global memory
  if (tid == 0) {
    pPartialSums[blockIdx.x] = sdata[0];
  }
}

/**
 * kernel for final reduction of complex partial sums
 */
__global__ void nppsSum_32fc_final_kernel(const Npp32fc *pPartialSums, size_t numBlocks, Npp32fc *pSum) {
  __shared__ Npp32fc sdata[256];

  size_t tid = threadIdx.x;
  size_t i = threadIdx.x;

  // Initialize shared memory
  sdata[tid].re = 0.0f;
  sdata[tid].im = 0.0f;

  // Accumulate all partial sums with stride
  while (i < numBlocks) {
    sdata[tid].re += pPartialSums[i].re;
    sdata[tid].im += pPartialSums[i].im;
    i += blockDim.x;
  }
  __syncthreads();

  // Perform final reduction within the block
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid].re += sdata[tid + s].re;
      sdata[tid].im += sdata[tid + s].im;
    }
    __syncthreads();
  }

  // Write final result
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
  const int gridSize = (nLength + blockSize - 1) / blockSize;

  // Cast buffer to proper type
  Npp32f *partialSums = reinterpret_cast<Npp32f *>(pDeviceBuffer);

  // First kernel: reduce each block to partial sums
  nppsSum_32f_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc, nLength, partialSums);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return error;
  }

  // Second kernel: reduce partial sums to final result
  nppsSum_32f_final_kernel<<<1, 256, 0, stream>>>(partialSums, gridSize, pSum);

  return cudaGetLastError();
}

cudaError_t nppsSum_32fc_kernel(const Npp32fc *pSrc, size_t nLength, Npp32fc *pSum, Npp8u *pDeviceBuffer,
                                cudaStream_t stream) {
  const int blockSize = 256;
  const int gridSize = (nLength + blockSize - 1) / blockSize;

  // Cast buffer to proper type
  Npp32fc *partialSums = reinterpret_cast<Npp32fc *>(pDeviceBuffer);

  // First kernel: reduce each block to partial sums
  nppsSum_32fc_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc, nLength, partialSums);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return error;
  }

  // Second kernel: reduce partial sums to final result
  nppsSum_32fc_final_kernel<<<1, 256, 0, stream>>>(partialSums, gridSize, pSum);

  return cudaGetLastError();
}

} // extern "C"