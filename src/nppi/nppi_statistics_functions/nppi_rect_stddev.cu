#include "npp.h"
#include <cuda_runtime.h>
#include <cmath>

// CUDA kernel for computing rectangular standard deviation
// This computes the standard deviation within a rectangular window for each pixel
// For each output pixel (x, y), the window in source image starts at (x + rect.x, y + rect.y)
// and has dimensions rect.width × rect.height
__global__ void rectStdDevKernel_32s32f_C1R(const Npp32s *pSrc, int srcStep, const Npp64f *pSqr, int sqrStep,
                                             Npp32f *pDst, int dstStep, int width, int height, NppiRect rect) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  // Calculate the window boundaries for this pixel in the source image
  // The window starts at (x + rect.x, y + rect.y) and spans rect.width × rect.height
  int x1 = x + rect.x;
  int y1 = y + rect.y;
  int x2 = x1 + rect.width - 1;
  int y2 = y1 + rect.height - 1;

  // Compute sum and sum of squares within the window
  double sum = 0.0;
  double sumSq = 0.0;
  int count = 0;

  for (int wy = y1; wy <= y2; wy++) {
    const Npp32s *srcRow = (const Npp32s *)((const Npp8u *)pSrc + wy * srcStep);
    const Npp64f *sqrRow = (const Npp64f *)((const Npp8u *)pSqr + wy * sqrStep);

    for (int wx = x1; wx <= x2; wx++) {
      double val = static_cast<double>(srcRow[wx]);
      sum += val;
      sumSq += sqrRow[wx];
      count++;
    }
  }

  // Compute standard deviation: sqrt(E[X^2] - E[X]^2)
  if (count > 0) {
    double mean = sum / count;
    double meanSq = sumSq / count;
    double variance = meanSq - mean * mean;

    // Ensure variance is non-negative (handle numerical errors)
    variance = max(0.0, variance);

    Npp32f *dstRow = (Npp32f *)((Npp8u *)pDst + y * dstStep);
    dstRow[x] = static_cast<Npp32f>(sqrt(variance));
  }
}

extern "C" NppStatus nppiRectStdDev_32s32f_C1R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, const Npp64f *pSqr,
                                                         int nSqrStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                                         NppiRect oRect, NppStreamContext nppStreamCtx) {
  // Configure kernel launch parameters
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  // Launch kernel
  rectStdDevKernel_32s32f_C1R<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pSqr, nSqrStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, oRect);

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}
