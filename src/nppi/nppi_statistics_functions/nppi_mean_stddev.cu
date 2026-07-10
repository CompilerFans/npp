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

__device__ __forceinline__ double blockReduceSum(double val, double *shared) {
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

  __syncthreads();
  return val;
}

__global__ void nppiMean_8u_C1R_kernel_impl(const Npp8u *pSrc, int nSrcStep, int width, int height,
                                            double *pBlockSums) {
  __shared__ double shared[WARP_SIZE];
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int totalPixels = width * height;
  double localSum = 0.0;
  for (int i = tid; i < totalPixels; i += blockDim.x * gridDim.x) {
    const int y = i / width;
    const int x = i % width;
    const Npp8u *row = reinterpret_cast<const Npp8u *>(reinterpret_cast<const char *>(pSrc) + y * nSrcStep);
    localSum += static_cast<double>(row[x]);
  }
  localSum = blockReduceSum(localSum, shared);
  if (threadIdx.x == 0) {
    pBlockSums[blockIdx.x] = localSum;
  }
}

template <typename T>
__global__ void nppiMean_CxR_kernel_impl(const T *pSrc, int nSrcStep, int width, int height, int sourceChannels,
                                         int outputChannels, double *pBlockSums) {
  __shared__ double shared[WARP_SIZE];
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int totalPixels = width * height;
  double localSums[4] = {0.0, 0.0, 0.0, 0.0};
  for (int index = tid; index < totalPixels; index += blockDim.x * gridDim.x) {
    const int y = index / width;
    const int x = index % width;
    const T *row = reinterpret_cast<const T *>(reinterpret_cast<const char *>(pSrc) + y * nSrcStep);
    for (int channel = 0; channel < outputChannels; ++channel) {
      localSums[channel] += static_cast<double>(row[x * sourceChannels + channel]);
    }
  }
  for (int channel = 0; channel < outputChannels; ++channel) {
    const double channelSum = blockReduceSum(localSums[channel], shared);
    if (threadIdx.x == 0) {
      pBlockSums[channel * gridDim.x + blockIdx.x] = channelSum;
    }
  }
}

__global__ void finalMeanChannels_kernel(const double *pBlockSums, int numBlocks, int totalPixels,
                                         double *pMean) {
  __shared__ double shared[WARP_SIZE];
  const int channel = blockIdx.x;
  double totalSum = 0.0;
  for (int index = threadIdx.x; index < numBlocks; index += blockDim.x) {
    totalSum += pBlockSums[channel * numBlocks + index];
  }
  totalSum = blockReduceSum(totalSum, shared);
  if (threadIdx.x == 0) {
    pMean[channel] = totalSum / static_cast<double>(totalPixels);
  }
}

__global__ void nppiAverageError_8u_C1R_kernel_impl(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2,
                                                    int nSrc2Step, int width, int height, double *pBlockSums) {
  __shared__ double shared[WARP_SIZE];
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int totalPixels = width * height;
  double localSum = 0.0;
  for (int i = tid; i < totalPixels; i += blockDim.x * gridDim.x) {
    const int y = i / width;
    const int x = i % width;
    const Npp8u *row1 = reinterpret_cast<const Npp8u *>(reinterpret_cast<const char *>(pSrc1) + y * nSrc1Step);
    const Npp8u *row2 = reinterpret_cast<const Npp8u *>(reinterpret_cast<const char *>(pSrc2) + y * nSrc2Step);
    localSum += static_cast<double>(abs(static_cast<int>(row1[x]) - static_cast<int>(row2[x])));
  }
  localSum = blockReduceSum(localSum, shared);
  if (threadIdx.x == 0) {
    pBlockSums[blockIdx.x] = localSum;
  }
}

__global__ void finalAverage_kernel(const double *pBlockSums, int numBlocks, int totalPixels, double *pAverage) {
  __shared__ double shared[WARP_SIZE];
  double totalSum = 0.0;
  for (int i = threadIdx.x; i < numBlocks; i += blockDim.x) {
    totalSum += pBlockSums[i];
  }
  totalSum = blockReduceSum(totalSum, shared);
  if (threadIdx.x == 0) {
    *pAverage = totalSum / static_cast<double>(totalPixels);
  }
}

// Mean and standard deviation kernel for 8-bit unsigned single channel
__global__ void nppiMean_StdDev_8u_C1R_kernel_impl(const Npp8u *pSrc, int nSrcStep, int width, int height,
                                                   double *pBlockSums, double *pBlockSumSquares) {
  __shared__ double shared[WARP_SIZE];
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
  localSum = blockReduceSum(localSum, shared);
  localSumSquares = blockReduceSum(localSumSquares, shared);

  // Store block results
  if (threadIdx.x == 0) {
    pBlockSums[blockIdx.x] = localSum;
    pBlockSumSquares[blockIdx.x] = localSumSquares;
  }
}

__global__ void nppiMean_StdDev_8u_C3CR_kernel_impl(const Npp8u *pSrc, int nSrcStep, int width, int height,
                                                    int channel, double *pBlockSums, double *pBlockSumSquares) {
  __shared__ double shared[WARP_SIZE];
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int totalPixels = width * height;
  double localSum = 0.0;
  double localSumSquares = 0.0;

  for (int i = tid; i < totalPixels; i += blockDim.x * gridDim.x) {
    const int y = i / width;
    const int x = i % width;
    const Npp8u *row = reinterpret_cast<const Npp8u *>(reinterpret_cast<const char *>(pSrc) + y * nSrcStep);
    const double value = static_cast<double>(row[x * 3 + channel]);
    localSum += value;
    localSumSquares += value * value;
  }

  localSum = blockReduceSum(localSum, shared);
  localSumSquares = blockReduceSum(localSumSquares, shared);
  if (threadIdx.x == 0) {
    pBlockSums[blockIdx.x] = localSum;
    pBlockSumSquares[blockIdx.x] = localSumSquares;
  }
}

// Final reduction kernel
__global__ void finalReduction_kernel(double *pBlockSums, double *pBlockSumSquares, int numBlocks, int totalPixels,
                                      double *pMean, double *pStdDev) {
  __shared__ double shared[WARP_SIZE];
  double totalSum = 0.0;
  double totalSumSquares = 0.0;

  for (int i = threadIdx.x; i < numBlocks; i += blockDim.x) {
    totalSum += pBlockSums[i];
    totalSumSquares += pBlockSumSquares[i];
  }

  totalSum = blockReduceSum(totalSum, shared);
  totalSumSquares = blockReduceSum(totalSumSquares, shared);

  if (threadIdx.x == 0) {
    double mean = totalSum / totalPixels;
    double variance = (totalSumSquares / totalPixels) - (mean * mean);
    double stddev = sqrt(fmax(variance, 0.0)); // Ensure non-negative

    *pMean = mean;
    *pStdDev = stddev;
  }
}

// Final reduction kernel for masked versions
__global__ void finalReductionMasked_kernel(double *pBlockSums, double *pBlockSumSquares, double *pValidCounts,
                                            int numBlocks, double *pMean, double *pStdDev) {
  __shared__ double shared[WARP_SIZE];
  double totalSum = 0.0;
  double totalSumSquares = 0.0;
  double totalValidCount = 0.0;

  for (int i = threadIdx.x; i < numBlocks; i += blockDim.x) {
    totalSum += pBlockSums[i];
    totalSumSquares += pBlockSumSquares[i];
    totalValidCount += pValidCounts[i];
  }

  totalSum = blockReduceSum(totalSum, shared);
  totalSumSquares = blockReduceSum(totalSumSquares, shared);
  totalValidCount = blockReduceSum(totalValidCount, shared);

  if (threadIdx.x == 0) {
    if (totalValidCount > 0) {
      double mean = totalSum / totalValidCount;
      double variance = (totalSumSquares / totalValidCount) - (mean * mean);
      double stddev = sqrt(fmax(variance, 0.0));

      *pMean = mean;
      *pStdDev = stddev;
    } else {
      // No valid pixels in mask
      *pMean = 0.0;
      *pStdDev = 0.0;
    }
  }
}

// Mean and standard deviation kernel for 32-bit float single channel
__global__ void nppiMean_StdDev_32f_C1R_kernel_impl(const Npp32f *pSrc, int nSrcStep, int width, int height,
                                                    double *pBlockSums, double *pBlockSumSquares) {
  __shared__ double shared[WARP_SIZE];
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
  localSum = blockReduceSum(localSum, shared);
  localSumSquares = blockReduceSum(localSumSquares, shared);

  // Store block results
  if (threadIdx.x == 0) {
    pBlockSums[blockIdx.x] = localSum;
    pBlockSumSquares[blockIdx.x] = localSumSquares;
  }
}

extern "C" {
cudaError_t nppiMean_8u_C1R_kernel(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,
                                   Npp64f *pMean, cudaStream_t stream) {
  const int totalPixels = oSizeROI.width * oSizeROI.height;
  const int blockSize = 256;
  const int numBlocks = (totalPixels + blockSize - 1) / blockSize;
  double *pBlockSums = reinterpret_cast<double *>(pDeviceBuffer);
  nppiMean_8u_C1R_kernel_impl<<<numBlocks, blockSize, 0, stream>>>(pSrc, nSrcStep, oSizeROI.width, oSizeROI.height,
                                                                  pBlockSums);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return error;
  }
  finalAverage_kernel<<<1, blockSize, 0, stream>>>(pBlockSums, numBlocks, totalPixels, pMean);
  return cudaGetLastError();
}

cudaError_t nppiMean_8u_CxR_kernel(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, int nSourceChannels,
                                   int nOutputChannels, Npp8u *pDeviceBuffer, Npp64f *pMean,
                                   cudaStream_t stream) {
  const int totalPixels = oSizeROI.width * oSizeROI.height;
  const int blockSize = 256;
  const int numBlocks = (totalPixels + blockSize - 1) / blockSize;
  double *pBlockSums = reinterpret_cast<double *>(pDeviceBuffer);
  nppiMean_CxR_kernel_impl<Npp8u><<<numBlocks, blockSize, 0, stream>>>(
      pSrc, nSrcStep, oSizeROI.width, oSizeROI.height, nSourceChannels, nOutputChannels, pBlockSums);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return error;
  }
  finalMeanChannels_kernel<<<nOutputChannels, blockSize, 0, stream>>>(pBlockSums, numBlocks, totalPixels, pMean);
  return cudaGetLastError();
}

cudaError_t nppiMean_16u_CxR_kernel(const Npp16u *pSrc, int nSrcStep, NppiSize oSizeROI, int nSourceChannels,
                                    int nOutputChannels, Npp8u *pDeviceBuffer, Npp64f *pMean,
                                    cudaStream_t stream) {
  const int totalPixels = oSizeROI.width * oSizeROI.height;
  const int blockSize = 256;
  const int numBlocks = (totalPixels + blockSize - 1) / blockSize;
  double *pBlockSums = reinterpret_cast<double *>(pDeviceBuffer);
  nppiMean_CxR_kernel_impl<Npp16u><<<numBlocks, blockSize, 0, stream>>>(
      pSrc, nSrcStep, oSizeROI.width, oSizeROI.height, nSourceChannels, nOutputChannels, pBlockSums);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return error;
  }
  finalMeanChannels_kernel<<<nOutputChannels, blockSize, 0, stream>>>(pBlockSums, numBlocks, totalPixels, pMean);
  return cudaGetLastError();
}

cudaError_t nppiAverageError_8u_C1R_kernel(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                                           NppiSize oSizeROI, Npp64f *pError, Npp8u *pDeviceBuffer,
                                           cudaStream_t stream) {
  const int totalPixels = oSizeROI.width * oSizeROI.height;
  const int blockSize = 256;
  const int numBlocks = (totalPixels + blockSize - 1) / blockSize;
  double *pBlockSums = reinterpret_cast<double *>(pDeviceBuffer);
  nppiAverageError_8u_C1R_kernel_impl<<<numBlocks, blockSize, 0, stream>>>(
      pSrc1, nSrc1Step, pSrc2, nSrc2Step, oSizeROI.width, oSizeROI.height, pBlockSums);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return error;
  }
  finalAverage_kernel<<<1, blockSize, 0, stream>>>(pBlockSums, numBlocks, totalPixels, pError);
  return cudaGetLastError();
}

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

cudaError_t nppiMean_StdDev_8u_C3CR_kernel(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, int nCOI,
                                           Npp8u *pDeviceBuffer, Npp64f *pMean, Npp64f *pStdDev, cudaStream_t stream) {
  const int totalPixels = oSizeROI.width * oSizeROI.height;
  const int blockSize = 256;
  const int numBlocks = (totalPixels + blockSize - 1) / blockSize;
  double *pBlockSums = reinterpret_cast<double *>(pDeviceBuffer);
  double *pBlockSumSquares = pBlockSums + numBlocks;

  nppiMean_StdDev_8u_C3CR_kernel_impl<<<numBlocks, blockSize, 0, stream>>>(
      pSrc, nSrcStep, oSizeROI.width, oSizeROI.height, nCOI - 1, pBlockSums, pBlockSumSquares);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return error;
  }
  finalReduction_kernel<<<1, blockSize, 0, stream>>>(pBlockSums, pBlockSumSquares, numBlocks, totalPixels, pMean,
                                                     pStdDev);
  return cudaGetLastError();
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

// ============ MASKED VERSIONS C1MR ===================

// Mean and standard deviation kernel for 8u single channel with mask
__global__ void nppiMean_StdDev_8u_C1MR_kernel_impl(const Npp8u *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep,
                                                    int width, int height, double *pBlockSums, double *pBlockSumSquares,
                                                    double *pValidCounts) {
  __shared__ double shared[WARP_SIZE];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int totalPixels = width * height;

  double localSum = 0.0;
  double localSumSquares = 0.0;
  double localValidCount = 0.0;

  // Grid-stride loop to process multiple pixels per thread
  for (int i = tid; i < totalPixels; i += blockDim.x * gridDim.x) {
    int y = i / width;
    int x = i % width;

    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    const Npp8u *pMaskRow = (const Npp8u *)((const char *)pMask + y * nMaskStep);

    // Only process pixel if mask is non-zero
    if (pMaskRow[x] > 0) {
      double pixelValue = (double)pSrcRow[x];
      localSum += pixelValue;
      localSumSquares += pixelValue * pixelValue;
      localValidCount += 1.0;
    }
  }

  // Reduce within block
  localSum = blockReduceSum(localSum, shared);
  localSumSquares = blockReduceSum(localSumSquares, shared);
  localValidCount = blockReduceSum(localValidCount, shared);

  // Store block results
  if (threadIdx.x == 0) {
    pBlockSums[blockIdx.x] = localSum;
    pBlockSumSquares[blockIdx.x] = localSumSquares;
    pValidCounts[blockIdx.x] = localValidCount;
  }
}

__global__ void nppiMean_StdDev_8u_C3CMR_kernel_impl(const Npp8u *pSrc, int nSrcStep, const Npp8u *pMask,
                                                     int nMaskStep, int width, int height, int channel,
                                                     double *pBlockSums, double *pBlockSumSquares,
                                                     double *pValidCounts) {
  __shared__ double shared[WARP_SIZE];
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int totalPixels = width * height;
  double localSum = 0.0;
  double localSumSquares = 0.0;
  double localValidCount = 0.0;

  for (int i = tid; i < totalPixels; i += blockDim.x * gridDim.x) {
    const int y = i / width;
    const int x = i % width;
    const Npp8u *maskRow =
        reinterpret_cast<const Npp8u *>(reinterpret_cast<const char *>(pMask) + y * nMaskStep);
    if (maskRow[x] != 0) {
      const Npp8u *sourceRow =
          reinterpret_cast<const Npp8u *>(reinterpret_cast<const char *>(pSrc) + y * nSrcStep);
      const double value = static_cast<double>(sourceRow[x * 3 + channel]);
      localSum += value;
      localSumSquares += value * value;
      localValidCount += 1.0;
    }
  }

  localSum = blockReduceSum(localSum, shared);
  localSumSquares = blockReduceSum(localSumSquares, shared);
  localValidCount = blockReduceSum(localValidCount, shared);
  if (threadIdx.x == 0) {
    pBlockSums[blockIdx.x] = localSum;
    pBlockSumSquares[blockIdx.x] = localSumSquares;
    pValidCounts[blockIdx.x] = localValidCount;
  }
}

// Mean and standard deviation kernel for 32f single channel with mask
__global__ void nppiMean_StdDev_32f_C1MR_kernel_impl(const Npp32f *pSrc, int nSrcStep, const Npp8u *pMask,
                                                     int nMaskStep, int width, int height, double *pBlockSums,
                                                     double *pBlockSumSquares, double *pValidCounts) {
  __shared__ double shared[WARP_SIZE];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int totalPixels = width * height;

  double localSum = 0.0;
  double localSumSquares = 0.0;
  double localValidCount = 0.0;

  // Grid-stride loop to process multiple pixels per thread
  for (int i = tid; i < totalPixels; i += blockDim.x * gridDim.x) {
    int y = i / width;
    int x = i % width;

    const Npp32f *pSrcRow = (const Npp32f *)((const char *)pSrc + y * nSrcStep);
    const Npp8u *pMaskRow = (const Npp8u *)((const char *)pMask + y * nMaskStep);

    // Only process pixel if mask is non-zero
    if (pMaskRow[x] > 0) {
      double pixelValue = (double)pSrcRow[x];
      localSum += pixelValue;
      localSumSquares += pixelValue * pixelValue;
      localValidCount += 1.0;
    }
  }

  // Reduce within block
  localSum = blockReduceSum(localSum, shared);
  localSumSquares = blockReduceSum(localSumSquares, shared);
  localValidCount = blockReduceSum(localValidCount, shared);

  // Store block results
  if (threadIdx.x == 0) {
    pBlockSums[blockIdx.x] = localSum;
    pBlockSumSquares[blockIdx.x] = localSumSquares;
    pValidCounts[blockIdx.x] = localValidCount;
  }
}

cudaError_t nppiMean_StdDev_8u_C1MR_kernel(const Npp8u *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep,
                                           NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp64f *pMean, Npp64f *pStdDev,
                                           cudaStream_t stream) {
  int totalPixels = oSizeROI.width * oSizeROI.height;
  int blockSize = 256;
  int numBlocks = (totalPixels + blockSize - 1) / blockSize;

  // Use device buffer for intermediate results (3 arrays: sums, sumSquares, validCounts)
  double *pBlockSums = (double *)pDeviceBuffer;
  double *pBlockSumSquares = pBlockSums + numBlocks;
  double *pValidCounts = pBlockSumSquares + numBlocks;

  // Launch first kernel for block-wise reduction
  nppiMean_StdDev_8u_C1MR_kernel_impl<<<numBlocks, blockSize, 0, stream>>>(
      pSrc, nSrcStep, pMask, nMaskStep, oSizeROI.width, oSizeROI.height, pBlockSums, pBlockSumSquares, pValidCounts);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return err;

  // Launch final reduction kernel
  finalReductionMasked_kernel<<<1, blockSize, 0, stream>>>(pBlockSums, pBlockSumSquares, pValidCounts, numBlocks, pMean,
                                                           pStdDev);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    if (stream == 0) {
      cudaDeviceSynchronize();
    }
  }
  return err;
}

cudaError_t nppiMean_StdDev_8u_C3CMR_kernel(const Npp8u *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep,
                                            NppiSize oSizeROI, int nCOI, Npp8u *pDeviceBuffer, Npp64f *pMean,
                                            Npp64f *pStdDev, cudaStream_t stream) {
  const int totalPixels = oSizeROI.width * oSizeROI.height;
  const int blockSize = 256;
  const int numBlocks = (totalPixels + blockSize - 1) / blockSize;
  double *pBlockSums = reinterpret_cast<double *>(pDeviceBuffer);
  double *pBlockSumSquares = pBlockSums + numBlocks;
  double *pValidCounts = pBlockSumSquares + numBlocks;

  nppiMean_StdDev_8u_C3CMR_kernel_impl<<<numBlocks, blockSize, 0, stream>>>(
      pSrc, nSrcStep, pMask, nMaskStep, oSizeROI.width, oSizeROI.height, nCOI - 1, pBlockSums, pBlockSumSquares,
      pValidCounts);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return error;
  }
  finalReductionMasked_kernel<<<1, blockSize, 0, stream>>>(pBlockSums, pBlockSumSquares, pValidCounts, numBlocks, pMean,
                                                           pStdDev);
  return cudaGetLastError();
}

cudaError_t nppiMean_StdDev_32f_C1MR_kernel(const Npp32f *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep,
                                            NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp64f *pMean, Npp64f *pStdDev,
                                            cudaStream_t stream) {
  int totalPixels = oSizeROI.width * oSizeROI.height;
  int blockSize = 256;
  int numBlocks = (totalPixels + blockSize - 1) / blockSize;

  // Use device buffer for intermediate results (3 arrays: sums, sumSquares, validCounts)
  double *pBlockSums = (double *)pDeviceBuffer;
  double *pBlockSumSquares = pBlockSums + numBlocks;
  double *pValidCounts = pBlockSumSquares + numBlocks;

  // Launch first kernel for block-wise reduction
  nppiMean_StdDev_32f_C1MR_kernel_impl<<<numBlocks, blockSize, 0, stream>>>(
      pSrc, nSrcStep, pMask, nMaskStep, oSizeROI.width, oSizeROI.height, pBlockSums, pBlockSumSquares, pValidCounts);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return err;

  // Launch final reduction kernel
  finalReductionMasked_kernel<<<1, blockSize, 0, stream>>>(pBlockSums, pBlockSumSquares, pValidCounts, numBlocks, pMean,
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
