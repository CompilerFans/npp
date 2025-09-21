#include "npp.h"
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Sobel X and Y direction kernels
__constant__ float c_sobelX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__constant__ float c_sobelY[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

// Gaussian filter kernel (3x3)
__constant__ float c_gaussianKernel3x3[9] = {1.0f / 16, 2.0f / 16, 1.0f / 16, 2.0f / 16, 4.0f / 16,
                                             2.0f / 16, 1.0f / 16, 2.0f / 16, 1.0f / 16};

// Gaussian filter kernel (5x5)
__constant__ float c_gaussianKernel5x5[25] = {
    1.0f / 256,  4.0f / 256, 6.0f / 256,  4.0f / 256,  1.0f / 256,  4.0f / 256, 16.0f / 256, 24.0f / 256, 16.0f / 256,
    4.0f / 256,  6.0f / 256, 24.0f / 256, 36.0f / 256, 24.0f / 256, 6.0f / 256, 4.0f / 256,  16.0f / 256, 24.0f / 256,
    16.0f / 256, 4.0f / 256, 1.0f / 256,  4.0f / 256,  6.0f / 256,  4.0f / 256, 1.0f / 256};

template <typename T>
__device__ T getBorderPixelCanny(const T *pSrc, int nSrcStep, int width, int height, int x, int y,
                                 NppiBorderType eBorderType) {
  if (x >= 0 && x < width && y >= 0 && y < height) {
    const T *src_row = (const T *)((const char *)pSrc + y * nSrcStep);
    return src_row[x];
  }

  switch (eBorderType) {
  case NPP_BORDER_REPLICATE:
    x = max(0, min(x, width - 1));
    y = max(0, min(y, height - 1));
    break;
  case NPP_BORDER_CONSTANT:
  default:
    return T(0);
  }

  const T *src_row = (const T *)((const char *)pSrc + y * nSrcStep);
  return src_row[x];
}

// Step 1: Gaussian filtering for noise reduction
__global__ void cannyGaussianBlur_kernel(const Npp8u *pSrc, int nSrcStep, float *pBlurred, int nBlurredStep,
                                         int srcWidth, int srcHeight, NppiMaskSize eMaskSize,
                                         NppiBorderType eBorderType) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= srcWidth || y >= srcHeight)
    return;

  float sum = 0.0f;
  int maskSize = (eMaskSize == NPP_MASK_SIZE_3_X_3) ? 3 : 5;
  int half = maskSize / 2;

  for (int dy = -half; dy <= half; dy++) {
    for (int dx = -half; dx <= half; dx++) {
      float pixel = (float)getBorderPixelCanny<Npp8u>(pSrc, nSrcStep, srcWidth, srcHeight, x + dx, y + dy, eBorderType);

      if (eMaskSize == NPP_MASK_SIZE_3_X_3) {
        sum += pixel * c_gaussianKernel3x3[(dy + half) * 3 + (dx + half)];
      } else {
        sum += pixel * c_gaussianKernel5x5[(dy + half) * 5 + (dx + half)];
      }
    }
  }

  float *blurred_row = (float *)((char *)pBlurred + y * nBlurredStep);
  blurred_row[x] = sum;
}

// 第二步：Compute gradient magnitude和方向
__global__ void cannySobelGradient_kernel(const float *pBlurred, int nBlurredStep, float *pGradMag, int nMagStep,
                                          float *pGradDir, int nDirStep, int width, int height,
                                          NppiBorderType eBorderType) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  float gx = 0.0f, gy = 0.0f;

  // Apply Sobel operator
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      int px = x + dx;
      int py = y + dy;

      float pixel;
      if (px >= 0 && px < width && py >= 0 && py < height) {
        const float *blurred_row = (const float *)((const char *)pBlurred + py * nBlurredStep);
        pixel = blurred_row[px];
      } else {
        // Boundary handling
        if (eBorderType == NPP_BORDER_REPLICATE) {
          px = max(0, min(px, width - 1));
          py = max(0, min(py, height - 1));
          const float *blurred_row = (const float *)((const char *)pBlurred + py * nBlurredStep);
          pixel = blurred_row[px];
        } else {
          pixel = 0.0f;
        }
      }

      int kernelIdx = (dy + 1) * 3 + (dx + 1);
      gx += pixel * c_sobelX[kernelIdx];
      gy += pixel * c_sobelY[kernelIdx];
    }
  }

  // Compute gradient magnitude和方向
  float magnitude = sqrtf(gx * gx + gy * gy);
  float direction = atan2f(gy, gx) * 180.0f / M_PI; // Convert to degrees

  // Normalize direction to 0-180 degrees
  if (direction < 0)
    direction += 180.0f;

  float *mag_row = (float *)((char *)pGradMag + y * nMagStep);
  float *dir_row = (float *)((char *)pGradDir + y * nDirStep);

  mag_row[x] = magnitude;
  dir_row[x] = direction;
}

// Step 3: Non-maximum suppression
__global__ void cannyNonMaxSuppression_kernel(const float *pGradMag, int nMagStep, const float *pGradDir, int nDirStep,
                                              float *pSuppressed, int nSuppressedStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const float *mag_row = (const float *)((const char *)pGradMag + y * nMagStep);
  const float *dir_row = (const float *)((const char *)pGradDir + y * nDirStep);
  float *supp_row = (float *)((char *)pSuppressed + y * nSuppressedStep);

  float magnitude = mag_row[x];
  float direction = dir_row[x];

  // Quantize direction to 4 main directions
  float angle = direction;
  if (angle > 180)
    angle -= 180;

  int dx1, dy1, dx2, dy2;
  if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
    // Horizontal direction
    dx1 = 1;
    dy1 = 0;
    dx2 = -1;
    dy2 = 0;
  } else if (angle >= 22.5 && angle < 67.5) {
    // Diagonal direction /
    dx1 = 1;
    dy1 = -1;
    dx2 = -1;
    dy2 = 1;
  } else if (angle >= 67.5 && angle < 112.5) {
    // Vertical direction
    dx1 = 0;
    dy1 = 1;
    dx2 = 0;
    dy2 = -1;
  } else {
    // Diagonal direction \
        dx1 = -1; dy1 = -1; dx2 = 1; dy2 = 1;
  }

  // Check adjacent pixels
  float mag1 = 0.0f, mag2 = 0.0f;

  int x1 = x + dx1, y1 = y + dy1;
  int x2 = x + dx2, y2 = y + dy2;

  if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
    const float *mag1_row = (const float *)((const char *)pGradMag + y1 * nMagStep);
    mag1 = mag1_row[x1];
  }

  if (x2 >= 0 && x2 < width && y2 >= 0 && y2 < height) {
    const float *mag2_row = (const float *)((const char *)pGradMag + y2 * nMagStep);
    mag2 = mag2_row[x2];
  }

  // Non-maximum suppression
  if (magnitude >= mag1 && magnitude >= mag2) {
    supp_row[x] = magnitude;
  } else {
    supp_row[x] = 0.0f;
  }
}

// Step 4: Double threshold detection and edge linking
__global__ void cannyDoubleThreshold_kernel(const float *pSuppressed, int nSuppressedStep, Npp8u *pDst, int nDstStep,
                                            int width, int height, float nLowThreshold, float nHighThreshold) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const float *supp_row = (const float *)((const char *)pSuppressed + y * nSuppressedStep);
  Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

  float magnitude = supp_row[x];

  if (magnitude >= nHighThreshold) {
    dst_row[x] = 255; // Strong edge
  } else if (magnitude >= nLowThreshold) {
    dst_row[x] = 128; // Weak edge
  } else {
    dst_row[x] = 0; // Non-edge
  }
}

// Edge linking (simplified version)
__global__ void cannyEdgeTracing_kernel(Npp8u *pDst, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

  if (dst_row[x] == 128) { // Weak edge
    // 检查8邻域是否有Strong edge
    bool hasStrongNeighbor = false;

    for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
        if (dx == 0 && dy == 0)
          continue;

        int nx = x + dx;
        int ny = y + dy;

        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
          const Npp8u *neighbor_row = (const Npp8u *)((const char *)pDst + ny * nDstStep);
          if (neighbor_row[nx] == 255) {
            hasStrongNeighbor = true;
            break;
          }
        }
      }
      if (hasStrongNeighbor)
        break;
    }

    dst_row[x] = hasStrongNeighbor ? 255 : 0;
  }
}

extern "C" {

// Get required buffer size
NppStatus nppiFilterCannyBorderGetBufferSize_8u_C1R_Ctx_impl(NppiSize oSizeROI, int *hpBufferSize) {
  size_t imageSize = (size_t)oSizeROI.width * oSizeROI.height;

  // 需要多个临时缓冲区：
  // 1. Gaussian filtering结果 (float)
  // 2. 梯度强度 (float)
  // 3. 梯度方向 (float)
  // 4. Non-maximum suppression结果 (float)

  size_t tempBufferSize = imageSize * sizeof(float) * 4; // 4个float缓冲区
  size_t alignedSize = (tempBufferSize + 511) & ~511;    // 512byte alignment

  *hpBufferSize = (int)alignedSize;
  return NPP_SUCCESS;
}

// CannyEdge detection主函数
NppStatus nppiFilterCannyBorder_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize,
                                                NppiPoint oSrcOffset, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                                NppiDifferentialKernel eFilterType, NppiMaskSize eMaskSize,
                                                Npp16s nLowThreshold, Npp16s nHighThreshold, NppiNorm eNorm,
                                                NppiBorderType eBorderType, Npp8u *pDeviceBuffer,
                                                NppStreamContext nppStreamCtx) {

  int srcWidth = oSrcSize.width;
  int srcHeight = oSrcSize.height;
  int dstWidth = oSizeROI.width;
  int dstHeight = oSizeROI.height;

  // 设置临时缓冲区
  size_t imageSize = srcWidth * srcHeight;
  float *pBlurred = (float *)pDeviceBuffer;
  float *pGradMag = pBlurred + imageSize;
  float *pGradDir = pGradMag + imageSize;
  float *pSuppressed = pGradDir + imageSize;

  int floatStep = srcWidth * sizeof(float);

  dim3 blockSize(16, 16);
  dim3 gridSize((srcWidth + blockSize.x - 1) / blockSize.x, (srcHeight + blockSize.y - 1) / blockSize.y);

  // 第一步：Gaussian filtering
  cannyGaussianBlur_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pBlurred, floatStep, srcWidth, srcHeight, eMaskSize, eBorderType);

  // 第二步：计算梯度
  cannySobelGradient_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pBlurred, floatStep, pGradMag, floatStep, pGradDir, floatStep, srcWidth, srcHeight, eBorderType);

  // Step 3: Non-maximum suppression
  cannyNonMaxSuppression_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pGradMag, floatStep, pGradDir, floatStep, pSuppressed, floatStep, srcWidth, srcHeight);

  // 调整网格大小用于输出
  dim3 dstGridSize((dstWidth + blockSize.x - 1) / blockSize.x, (dstHeight + blockSize.y - 1) / blockSize.y);

  // 第四步：双threshold检测
  cannyDoubleThreshold_kernel<<<dstGridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSuppressed, floatStep, pDst, nDstStep, dstWidth, dstHeight, nLowThreshold, nHighThreshold);

  // 第五步：边缘connection（多次迭代）
  for (int i = 0; i < 3; i++) {
    cannyEdgeTracing_kernel<<<dstGridSize, blockSize, 0, nppStreamCtx.hStream>>>(pDst, nDstStep, dstWidth, dstHeight);
  }

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}
}
