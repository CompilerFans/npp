#include "nppdefs.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifndef MPP_FILTERBOX_ZERO_PADDING
#define MPP_FILTERBOX_ZERO_PADDING
#endif
// Define boundary handling mode
// MPP_FILTERBOX_ZERO_PADDING: Use zero padding for out-of-bounds pixels
// If not defined, assumes valid data exists outside boundaries
// #define MPP_FILTERBOX_ZERO_PADDING

// IMPLEMENTATION NOTE:
// This filter box implementation supports two boundary handling modes:
//
// 1. Zero Padding Mode (MPP_FILTERBOX_ZERO_PADDING defined):
//    - Out-of-bounds pixels are treated as zeros
//    - Safe for processing images without surrounding context
//    - Slightly slower due to boundary checks
//
// 2. Direct Access Mode (default):
//    - Assumes valid data exists outside the ROI boundaries
//    - No bounds checking is performed
//    - Faster but requires sufficient buffer around ROI
//    - Suitable for processing sub-regions of larger images

// Box filter kernel implementation with configurable boundary handling
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

    int sum = 0;
    int totalMaskPixels = maskWidth * maskHeight;

    for (int j = startY; j < endY; j++) {
      for (int i = startX; i < endX; i++) {
#ifdef MPP_FILTERBOX_ZERO_PADDING
        // Check bounds and use zero for out-of-bounds pixels
        if (i >= 0 && i < width && j >= 0 && j < height) {
          const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + j * nSrcStep);
          sum += pSrcRow[i];
        }
        // else: pixel is out of bounds, implicitly add 0
#else
        // Direct access without bounds checking
        const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + j * nSrcStep);
        sum += pSrcRow[i];
#endif
      }
    }

    // Compute average using total mask size with truncation (toward zero)
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);
    pDstRow[x] = sum / totalMaskPixels; // Integer division (truncation)
  }
}

// Box filter kernel for 4-channel 8-bit unsigned with configurable boundary handling
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

    int sum[4] = {0, 0, 0, 0};
    int totalMaskPixels = maskWidth * maskHeight;

    for (int j = startY; j < endY; j++) {
      for (int i = startX; i < endX; i++) {
#ifdef MPP_FILTERBOX_ZERO_PADDING
        // Check bounds and use zero for out-of-bounds pixels
        if (i >= 0 && i < width && j >= 0 && j < height) {
          const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + j * nSrcStep);
          for (int c = 0; c < 4; c++) {
            sum[c] += pSrcRow[i * 4 + c];
          }
        }
        // else: pixel is out of bounds, implicitly add 0
#else
        // Direct access without bounds checking
        const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + j * nSrcStep);
        for (int c = 0; c < 4; c++) {
          sum[c] += pSrcRow[i * 4 + c];
        }
#endif
      }
    }

    // Calculate average using total mask size with truncation (toward zero)
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);
    for (int c = 0; c < 4; c++) {
      pDstRow[x * 4 + c] = sum[c] / totalMaskPixels; // Integer division (truncation)
    }
  }
}

// Box filter kernel for single-channel 32-bit float with configurable boundary handling
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

    float sum = 0.0f;
    int totalMaskPixels = maskWidth * maskHeight;

    for (int j = startY; j < endY; j++) {
      for (int i = startX; i < endX; i++) {
#ifdef MPP_FILTERBOX_ZERO_PADDING
        // Check bounds and use zero for out-of-bounds pixels
        if (i >= 0 && i < width && j >= 0 && j < height) {
          const Npp32f *pSrcRow = (const Npp32f *)((const char *)pSrc + j * nSrcStep);
          sum += pSrcRow[i];
        }
        // else: pixel is out of bounds, implicitly add 0.0f
#else
        // Direct access without bounds checking
        const Npp32f *pSrcRow = (const Npp32f *)((const char *)pSrc + j * nSrcStep);
        sum += pSrcRow[i];
#endif
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