#include "nppdefs.h"
#include <cuda_runtime.h>

#if 1
#define WARP_SIZE 32
#define SHFL_MASK 0xFFFFFFFF
#else
#define WARP_SIZE 64
#define SHFL_MASK 0xFFFFFFFFFFFFFFFFULL
#endif

__device__ __forceinline__ double warpReduceSum(double val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(SHFL_MASK, val, offset);
  }
  return val;
}

// Block-level reduction for double precision

__device__ __forceinline__ double blockReduceSum(double val) {
  static __shared__ double shared[32]; // 32 warps max
  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;

  val = warpReduceSum(val);

  if (lane == 0)
    shared[wid] = val;
  __syncthreads();

  // Read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;

  if (wid == 0)
    val = warpReduceSum(val);

  return val;
}

// Mean and standard deviation kernel for 8-bit unsigned single channel
__global__ void nppiMean_StdDev_8u_C1R_kernel_impl(const Npp8u *pSrc, int nSrcStep, int width, int height,
                                                   double *pBlockSums, double *pBlockSumSquares) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int totalPixels = width * height;

  double localSum = 0.0;
  double localSumSquares = 0.0;

  // Grid-stride loop to process multiple pixels per thread
  for (int i = tid; i < totalPixels; i += blockDim.x * gridDim.x) {
    int y = i / width;
    int x = i % width;

    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    double pixelValue = (double)pSrcRow[x];

    localSum += pixelValue;
    localSumSquares += pixelValue * pixelValue;
  }

  // Reduce within block
  localSum = blockReduceSum(localSum);
  localSumSquares = blockReduceSum(localSumSquares);

  // Store block results
  if (threadIdx.x == 0) {
    pBlockSums[blockIdx.x] = localSum;
    pBlockSumSquares[blockIdx.x] = localSumSquares;
  }
}

// Final reduction kernel
__global__ void finalReduction_kernel(double *pBlockSums, double *pBlockSumSquares, int numBlocks, int totalPixels,
                                      double *pMean, double *pStdDev) {
  double totalSum = 0.0;
  double totalSumSquares = 0.0;

  for (int i = threadIdx.x; i < numBlocks; i += blockDim.x) {
    totalSum += pBlockSums[i];
    totalSumSquares += pBlockSumSquares[i];
  }

  totalSum = blockReduceSum(totalSum);
  totalSumSquares = blockReduceSum(totalSumSquares);

  if (threadIdx.x == 0) {
    double mean = totalSum / totalPixels;
    double variance = (totalSumSquares / totalPixels) - (mean * mean);
    double stddev = sqrt(fmax(variance, 0.0)); // Ensure non-negative

    *pMean = mean;
    *pStdDev = stddev;
  }
}

// Mean and standard deviation kernel for 32-bit float single channel
__global__ void nppiMean_StdDev_32f_C1R_kernel_impl(const Npp32f *pSrc, int nSrcStep, int width, int height,
                                                    double *pBlockSums, double *pBlockSumSquares) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int totalPixels = width * height;

  double localSum = 0.0;
  double localSumSquares = 0.0;

  // Grid-stride loop to process multiple pixels per thread
  for (int i = tid; i < totalPixels; i += blockDim.x * gridDim.x) {
    int y = i / width;
    int x = i % width;

    // Calculate byte offset for float data
    const Npp32f *pSrcRow = (const Npp32f *)((const char *)pSrc + y * nSrcStep);
    double pixelValue = (double)pSrcRow[x];

    localSum += pixelValue;
    localSumSquares += pixelValue * pixelValue;
  }

  // Reduce within block
  localSum = blockReduceSum(localSum);
  localSumSquares = blockReduceSum(localSumSquares);

  // Store block results
  if (threadIdx.x == 0) {
    pBlockSums[blockIdx.x] = localSum;
    pBlockSumSquares[blockIdx.x] = localSumSquares;
  }
}

extern "C" {
cudaError_t nppiMean_StdDev_8u_C1R_kernel(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,
                                          Npp64f *pMean, Npp64f *pStdDev, cudaStream_t stream) {
  int totalPixels = oSizeROI.width * oSizeROI.height;
  int blockSize = 256;
  int numBlocks = (totalPixels + blockSize - 1) / blockSize;

  // Use device buffer for intermediate results
  double *pBlockSums = (double *)pDeviceBuffer;
  double *pBlockSumSquares = pBlockSums + numBlocks;

  // Launch first kernel for block-wise reduction
  nppiMean_StdDev_8u_C1R_kernel_impl<<<numBlocks, blockSize, 0, stream>>>(
      pSrc, nSrcStep, oSizeROI.width, oSizeROI.height, pBlockSums, pBlockSumSquares);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return err;

  // Launch final reduction kernel
  finalReduction_kernel<<<1, blockSize, 0, stream>>>(pBlockSums, pBlockSumSquares, numBlocks, totalPixels, pMean,
                                                     pStdDev);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    if (stream == 0) {
      cudaDeviceSynchronize();
    }
  }
  return err;
}

cudaError_t nppiMean_StdDev_32f_C1R_kernel(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,
                                           Npp64f *pMean, Npp64f *pStdDev, cudaStream_t stream) {
  int totalPixels = oSizeROI.width * oSizeROI.height;
  int blockSize = 256;
  int numBlocks = (totalPixels + blockSize - 1) / blockSize;

  // Use device buffer for intermediate results
  double *pBlockSums = (double *)pDeviceBuffer;
  double *pBlockSumSquares = pBlockSums + numBlocks;

  // Launch first kernel for block-wise reduction
  nppiMean_StdDev_32f_C1R_kernel_impl<<<numBlocks, blockSize, 0, stream>>>(
      pSrc, nSrcStep, oSizeROI.width, oSizeROI.height, pBlockSums, pBlockSumSquares);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return err;

  // Launch final reduction kernel
  finalReduction_kernel<<<1, blockSize, 0, stream>>>(pBlockSums, pBlockSumSquares, numBlocks, totalPixels, pMean,
                                                     pStdDev);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    if (stream == 0) {
      cudaDeviceSynchronize();
    }
  }
  return err;
}
}
