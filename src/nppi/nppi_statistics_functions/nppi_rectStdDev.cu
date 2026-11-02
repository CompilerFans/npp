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

  // The inputs are integral images: the window sum is the usual integral
  // difference, exactly as in nppiRectStdDev_32f_C1R.
  const int x1 = x + rect.x, y1 = y + rect.y;
  const int x2 = x1 + rect.width, y2 = y1 + rect.height;
  const Npp32s *srcRow0 = (const Npp32s *)((const Npp8u *)pSrc + (size_t)y1 * srcStep);
  const Npp32s *srcRow1 = (const Npp32s *)((const Npp8u *)pSrc + (size_t)y2 * srcStep);
  const Npp64f *sqrRow0 = (const Npp64f *)((const Npp8u *)pSqr + (size_t)y1 * sqrStep);
  const Npp64f *sqrRow1 = (const Npp64f *)((const Npp8u *)pSqr + (size_t)y2 * sqrStep);
  const double sum = (double)srcRow1[x2] - (double)srcRow0[x2] - (double)srcRow1[x1] + (double)srcRow0[x1];
  const double sumSq = sqrRow1[x2] - sqrRow0[x2] - sqrRow1[x1] + sqrRow0[x1];
  const double count = (double)rect.width * rect.height;
  double variance = (count * sumSq - sum * sum) / (count * count);
  variance = variance > 0.0 ? variance : 0.0;
  Npp32f *dstRow = (Npp32f *)((Npp8u *)pDst + y * dstStep);
  dstRow[x] = static_cast<Npp32f>(sqrt(variance));
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
