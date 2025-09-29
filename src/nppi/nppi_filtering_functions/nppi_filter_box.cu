#include "nppdefs.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// IMPLEMENTATION NOTE:
// This filter box implementation assumes valid data exists outside the ROI boundaries.
// Unlike zero-padding approaches, this implementation directly accesses memory outside
// the specified ROI without bounds checking. This design choice:
//
// 1. Improves performance by eliminating boundary condition checks
// 2. Requires the caller to ensure pSrc points to a location within a sufficiently
//    large buffer to accommodate all filter accesses
// 3. Is suitable when processing sub-regions of larger images where valid data
//    exists in the surrounding areas
//
// Memory Access Pattern:
// For pixel (x,y) with mask size (mw,mh) and anchor (ax,ay):
// - Accesses from (x-ax, y-ay) to (x-ax+mw-1, y-ay+mh-1)
// - No bounds checking is performed on these accesses

// Box filter kernel implementation - assumes data exists outside boundaries
__global__ void nppiFilterBox_8u_C1R_kernel_impl(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                                 int width, int height, int maskWidth, int maskHeight, int anchorX,
                                                 int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    // Compute filter coverage area
    int startX = x - anchorX;
    int startY = y - anchorY;
    int endX = startX + maskWidth;
    int endY = startY + maskHeight;

    // No boundary checking - assume valid data exists outside ROI
    int sum = 0;
    int totalMaskPixels = maskWidth * maskHeight;

    for (int j = startY; j < endY; j++) {
      for (int i = startX; i < endX; i++) {
        // Access data directly without bounds checking
        const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + j * nSrcStep);
        sum += pSrcRow[i];
      }
    }

    // Compute average using total mask size with truncation (toward zero)
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);
    pDstRow[x] = sum / totalMaskPixels; // Integer division (truncation)
  }
}

// Box filter kernel for 4-channel 8-bit unsigned - assumes data exists outside boundaries
__global__ void nppiFilterBox_8u_C4R_kernel_impl(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                                 int width, int height, int maskWidth, int maskHeight, int anchorX,
                                                 int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    // Calculate filter coverage area
    int startX = x - anchorX;
    int startY = y - anchorY;
    int endX = startX + maskWidth;
    int endY = startY + maskHeight;

    // No boundary checking - assume valid data exists outside ROI
    int sum[4] = {0, 0, 0, 0};
    int totalMaskPixels = maskWidth * maskHeight;

    for (int j = startY; j < endY; j++) {
      for (int i = startX; i < endX; i++) {
        // Access data directly without bounds checking
        const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + j * nSrcStep);
        for (int c = 0; c < 4; c++) {
          sum[c] += pSrcRow[i * 4 + c];
        }
      }
    }

    // Calculate average using total mask size with truncation (toward zero)
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);
    for (int c = 0; c < 4; c++) {
      pDstRow[x * 4 + c] = sum[c] / totalMaskPixels; // Integer division (truncation)
    }
  }
}

// Box filter kernel for single-channel 32-bit float - assumes data exists outside boundaries
__global__ void nppiFilterBox_32f_C1R_kernel_impl(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                                  int width, int height, int maskWidth, int maskHeight, int anchorX,
                                                  int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    // Calculate filter coverage area
    int startX = x - anchorX;
    int startY = y - anchorY;
    int endX = startX + maskWidth;
    int endY = startY + maskHeight;

    // No boundary checking - assume valid data exists outside ROI
    float sum = 0.0f;
    int totalMaskPixels = maskWidth * maskHeight;

    for (int j = startY; j < endY; j++) {
      for (int i = startX; i < endX; i++) {
        // Access data directly without bounds checking
        const Npp32f *pSrcRow = (const Npp32f *)((const char *)pSrc + j * nSrcStep);
        sum += pSrcRow[i];
      }
    }

    // Calculate average using total mask size
    Npp32f *pDstRow = (Npp32f *)((char *)pDst + y * nDstStep);
    pDstRow[x] = sum / totalMaskPixels;
  }
}

extern "C" {
cudaError_t nppiFilterBox_8u_C1R_kernel(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                        NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiFilterBox_8u_C1R_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                       oSizeROI.height, oMaskSize.width,
                                                                       oMaskSize.height, oAnchor.x, oAnchor.y);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    if (stream == 0) {
      cudaDeviceSynchronize();
    }
  }
  return err;
}

cudaError_t nppiFilterBox_8u_C4R_kernel(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                        NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiFilterBox_8u_C4R_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                       oSizeROI.height, oMaskSize.width,
                                                                       oMaskSize.height, oAnchor.x, oAnchor.y);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    if (stream == 0) {
      cudaDeviceSynchronize();
    }
  }
  return err;
}

cudaError_t nppiFilterBox_32f_C1R_kernel(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                         NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor,
                                         cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiFilterBox_32f_C1R_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                        oSizeROI.height, oMaskSize.width,
                                                                        oMaskSize.height, oAnchor.x, oAnchor.y);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    if (stream == 0) {
      cudaDeviceSynchronize();
    }
  }
  return err;
}
}
