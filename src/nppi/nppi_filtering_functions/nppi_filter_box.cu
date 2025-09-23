#include "nppdefs.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Box filter kernel implementation with zero padding - compute neighborhood average
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

    // Zero padding algorithm: sum all mask positions, treat out-of-bounds as 0
    int sum = 0;
    int totalMaskPixels = maskWidth * maskHeight;

    for (int j = startY; j < endY; j++) {
      for (int i = startX; i < endX; i++) {
        // Check if coordinates are within image bounds
        if (i >= 0 && i < width && j >= 0 && j < height) {
          const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + j * nSrcStep);
          sum += pSrcRow[i];
        }
        // Out-of-bounds pixels contribute 0 to sum (zero padding)
      }
    }

    // Compute average using total mask size (including zero-padded pixels)
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);
    pDstRow[x] = (sum + totalMaskPixels / 2) / totalMaskPixels; // Round to nearest
  }
}

// Box filter kernel for 4-channel 8-bit unsigned with zero padding
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

    // Zero padding algorithm: sum all mask positions, treat out-of-bounds as 0
    int sum[4] = {0, 0, 0, 0};
    int totalMaskPixels = maskWidth * maskHeight;

    for (int j = startY; j < endY; j++) {
      for (int i = startX; i < endX; i++) {
        // Check if coordinates are within image bounds
        if (i >= 0 && i < width && j >= 0 && j < height) {
          const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + j * nSrcStep);
          for (int c = 0; c < 4; c++) {
            sum[c] += pSrcRow[i * 4 + c];
          }
        }
        // Out-of-bounds pixels contribute 0 to sum for all channels (zero padding)
      }
    }

    // Calculate average using total mask size (including zero-padded pixels)
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);
    for (int c = 0; c < 4; c++) {
      pDstRow[x * 4 + c] = (sum[c] + totalMaskPixels / 2) / totalMaskPixels; // Round to nearest
    }
  }
}

// Box filter kernel for single-channel 32-bit float with zero padding
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

    // Zero padding algorithm: sum all mask positions, treat out-of-bounds as 0.0
    float sum = 0.0f;
    int totalMaskPixels = maskWidth * maskHeight;

    for (int j = startY; j < endY; j++) {
      for (int i = startX; i < endX; i++) {
        // Check if coordinates are within image bounds
        if (i >= 0 && i < width && j >= 0 && j < height) {
          const Npp32f *pSrcRow = (const Npp32f *)((const char *)pSrc + j * nSrcStep);
          sum += pSrcRow[i];
        }
        // Out-of-bounds pixels contribute 0.0 to sum (zero padding)
      }
    }

    // Calculate average using total mask size (including zero-padded pixels)
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
