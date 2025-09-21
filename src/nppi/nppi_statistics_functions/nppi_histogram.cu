#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * kernels for MPP Histogram Functions
 */

// Shared memory histogram kernel (more efficient for small histograms)
__global__ void nppiHistogramEven_8u_C1R_kernel_shared(const Npp8u *pSrc, int nSrcStep, int width, int height,
                                                       Npp32s *pHist, int nLevels, Npp32s nLowerLevel,
                                                       Npp32s nUpperLevel) {
  extern __shared__ int shared_hist[];

  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int threads_per_block = blockDim.x * blockDim.y;

  // Initialize shared memory histogram
  for (int i = tid; i < nLevels - 1; i += threads_per_block) {
    shared_hist[i] = 0;
  }
  __syncthreads();

  // Calculate global thread position
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Process pixels and update shared histogram
  if (x < width && y < height) {
    const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    int pixel_value = src_row[x];

    // Map pixel value to histogram bin
    if (pixel_value >= nLowerLevel && pixel_value < nUpperLevel) {
      int range = nUpperLevel - nLowerLevel;
      int bin = ((pixel_value - nLowerLevel) * (nLevels - 1)) / range;
      bin = min(bin, nLevels - 2); // Clamp to valid range

      atomicAdd(&shared_hist[bin], 1);
    }
  }

  __syncthreads();

  // Copy shared histogram to global memory
  for (int i = tid; i < nLevels - 1; i += threads_per_block) {
    atomicAdd(&pHist[i], shared_hist[i]);
  }
}

// Global memory histogram kernel (for large images)
__global__ void nppiHistogramEven_8u_C1R_kernel_global(const Npp8u *pSrc, int nSrcStep, int width, int height,
                                                       Npp32s *pHist, int nLevels, Npp32s nLowerLevel,
                                                       Npp32s nUpperLevel) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    int pixel_value = src_row[x];

    // Map pixel value to histogram bin
    if (pixel_value >= nLowerLevel && pixel_value < nUpperLevel) {
      int range = nUpperLevel - nLowerLevel;
      int bin = ((pixel_value - nLowerLevel) * (nLevels - 1)) / range;
      bin = min(bin, nLevels - 2); // Clamp to valid range

      atomicAdd(&pHist[bin], 1);
    }
  }
}

extern "C" {

// Get buffer size for histogram computation
NppStatus nppiHistogramEvenGetBufferSize_8u_C1R_Ctx_cuda(NppiSize oSizeROI, int nLevels, size_t *hpBufferSize) {
  // Calculate required buffer size
  // For histogram computation, we need space for temporary data
  size_t histogramSize = (size_t)(nLevels - 1) * sizeof(Npp32s);

  // Buffer size includes space for histogram and some temporary workspace
  *hpBufferSize = histogramSize + 1024; // Add some padding for alignment

  return NPP_SUCCESS;
}

// Histogram computation with even levels
NppStatus nppiHistogramEven_8u_C1R_Ctx_cuda(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist,
                                            int nLevels, Npp32s nLowerLevel, Npp32s nUpperLevel, Npp8u *pDeviceBuffer,
                                            NppStreamContext nppStreamCtx) {
  // Initialize histogram to zero
  cudaMemset(pHist, 0, (nLevels - 1) * sizeof(Npp32s));

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  // Choose kernel based on histogram size and image size
  size_t imageSize = (size_t)oSizeROI.width * oSizeROI.height;
  int histBins = nLevels - 1;

  if (histBins <= 256 && imageSize < 1024 * 1024) {
    // Use shared memory kernel for small histograms and images
    size_t sharedMemSize = histBins * sizeof(int);

    nppiHistogramEven_8u_C1R_kernel_shared<<<gridSize, blockSize, sharedMemSize, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSizeROI.width, oSizeROI.height, pHist, nLevels, nLowerLevel, nUpperLevel);
  } else {
    // Use global memory kernel for large histograms or images
    nppiHistogramEven_8u_C1R_kernel_global<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSizeROI.width, oSizeROI.height, pHist, nLevels, nLowerLevel, nUpperLevel);
  }

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

} // extern "C"