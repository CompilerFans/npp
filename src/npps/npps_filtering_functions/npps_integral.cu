#include "npp.h"
#include <cuda_runtime.h>

// ==============================================================================
// GPU Kernels for Integral (Exclusive Prefix Sum) Operations
// NVIDIA NPP uses exclusive scan: result[i] = sum of all elements before index i
// ==============================================================================

#define BLOCK_SIZE 256

// Exclusive scan within a block using shared memory
__device__ void blockExclusiveScan(Npp32s *sdata, int tid) {
  // Up-sweep (reduce) phase
  for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
    int index = (tid + 1) * stride * 2 - 1;
    if (index < BLOCK_SIZE) {
      sdata[index] += sdata[index - stride];
    }
    __syncthreads();
  }

  // Clear the last element for exclusive scan
  if (tid == 0) {
    sdata[BLOCK_SIZE - 1] = 0;
  }
  __syncthreads();

  // Down-sweep phase
  for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
    int index = (tid + 1) * stride * 2 - 1;
    if (index < BLOCK_SIZE) {
      Npp32s temp = sdata[index];
      sdata[index] += sdata[index - stride];
      sdata[index - stride] = temp;
    }
    __syncthreads();
  }
}

// First pass: compute block-local exclusive prefix sums and store block totals
__global__ void nppsIntegral_32s_pass1(const Npp32s *pSrc, Npp32s *pDst, size_t nLength, Npp32s *pBlockSums) {
  __shared__ Npp32s sdata[BLOCK_SIZE];
  __shared__ Npp32s blockTotal;

  int tid = threadIdx.x;
  size_t globalIdx = blockIdx.x * BLOCK_SIZE + tid;

  // Load data into shared memory
  Npp32s val = (globalIdx < nLength) ? pSrc[globalIdx] : 0;
  sdata[tid] = val;
  __syncthreads();

  // Compute block total before scan modifies data
  // Use reduction to compute sum
  for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    blockTotal = sdata[0];
  }
  __syncthreads();

  // Reload data for scan
  sdata[tid] = val;
  __syncthreads();

  // Perform exclusive scan within block
  blockExclusiveScan(sdata, tid);

  // Write results
  if (globalIdx < nLength) {
    pDst[globalIdx] = sdata[tid];
  }

  // Store block sum (total of all elements in block)
  if (tid == 0) {
    pBlockSums[blockIdx.x] = blockTotal;
  }
}

// Second pass: exclusive scan of block sums
__global__ void nppsIntegral_32s_scanBlockSums(Npp32s *pBlockSums, size_t numBlocks) {
  __shared__ Npp32s sdata[BLOCK_SIZE];

  int tid = threadIdx.x;

  // Load block sums
  sdata[tid] = (static_cast<size_t>(tid) < numBlocks) ? pBlockSums[tid] : 0;
  __syncthreads();

  // Perform exclusive scan
  blockExclusiveScan(sdata, tid);

  // Write back scanned block sums
  if (static_cast<size_t>(tid) < numBlocks) {
    pBlockSums[tid] = sdata[tid];
  }
}

// Third pass: add block prefix to each element
__global__ void nppsIntegral_32s_pass3(Npp32s *pDst, size_t nLength, const Npp32s *pBlockSums) {
  int tid = threadIdx.x;
  size_t globalIdx = blockIdx.x * BLOCK_SIZE + tid;

  if (globalIdx < nLength) {
    pDst[globalIdx] += pBlockSums[blockIdx.x];
  }
}

// Simple sequential kernel for small arrays (exclusive scan)
__global__ void nppsIntegral_32s_simple(const Npp32s *pSrc, Npp32s *pDst, size_t nLength) {
  // Single thread sequential exclusive prefix sum for small arrays
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    Npp32s sum = 0;
    for (size_t i = 0; i < nLength; i++) {
      Npp32s val = pSrc[i];
      pDst[i] = sum;
      sum += val;
    }
  }
}

// Simple sequential kernel for block sums (exclusive scan)
__global__ void nppsIntegral_32s_scanBlockSums_simple(Npp32s *pBlockSums, size_t numBlocks) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    Npp32s sum = 0;
    for (size_t i = 0; i < numBlocks; i++) {
      Npp32s val = pBlockSums[i];
      pBlockSums[i] = sum;
      sum += val;
    }
  }
}

// ==============================================================================
// Kernel Launch Function
// ==============================================================================

extern "C" {

cudaError_t nppsIntegral_32s_kernel(const Npp32s *pSrc, Npp32s *pDst, size_t nLength, Npp8u *pDeviceBuffer,
                                    cudaStream_t stream) {
  if (nLength == 0) {
    return cudaErrorInvalidValue;
  }

  // For small arrays, use simple sequential approach
  if (nLength <= BLOCK_SIZE) {
    nppsIntegral_32s_simple<<<1, 1, 0, stream>>>(pSrc, pDst, nLength);
    return cudaGetLastError();
  }

  // For larger arrays, use parallel scan algorithm
  size_t numBlocks = (nLength + BLOCK_SIZE - 1) / BLOCK_SIZE;
  Npp32s *pBlockSums = reinterpret_cast<Npp32s *>(pDeviceBuffer);

  // Pass 1: Block-local exclusive prefix sums
  nppsIntegral_32s_pass1<<<static_cast<unsigned int>(numBlocks), BLOCK_SIZE, 0, stream>>>(pSrc, pDst, nLength,
                                                                                          pBlockSums);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return error;
  }

  // Pass 2: Exclusive scan of block sums
  if (numBlocks > 1) {
    if (numBlocks <= BLOCK_SIZE) {
      nppsIntegral_32s_scanBlockSums<<<1, BLOCK_SIZE, 0, stream>>>(pBlockSums, numBlocks);
    } else {
      // For very large arrays with many blocks, use sequential approach
      nppsIntegral_32s_scanBlockSums_simple<<<1, 1, 0, stream>>>(pBlockSums, numBlocks);
    }

    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return error;
    }

    // Pass 3: Add block prefixes
    nppsIntegral_32s_pass3<<<static_cast<unsigned int>(numBlocks), BLOCK_SIZE, 0, stream>>>(pDst, nLength, pBlockSums);
  }

  return cudaGetLastError();
}
}
