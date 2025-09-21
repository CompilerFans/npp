#include "nppdefs.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Box滤波器kernel实现 - 计算邻域平均值
__global__ void nppiFilterBox_8u_C1R_kernel_impl(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                                 int width, int height, int maskWidth, int maskHeight, int anchorX,
                                                 int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    // 计算滤波器覆盖的区域
    int startX = x - anchorX;
    int startY = y - anchorY;
    int endX = startX + maskWidth;
    int endY = startY + maskHeight;

    // 边界处理 - 限制在图像bounds内
    startX = max(startX, 0);
    startY = max(startY, 0);
    endX = min(endX, width);
    endY = min(endY, height);

    // 计算区域内所有像素的和
    int sum = 0;
    int count = 0;

    for (int j = startY; j < endY; j++) {
      const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + j * nSrcStep);
      for (int i = startX; i < endX; i++) {
        sum += pSrcRow[i];
        count++;
      }
    }

    // 计算平均值并写入输出
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);
    pDstRow[x] = (count > 0) ? ((sum + count / 2) / count) : 0; // 四舍五入
  }
}

// Box filter kernel for 4-channel 8-bit unsigned
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

    // Boundary handling - clamp to image bounds
    startX = max(startX, 0);
    startY = max(startY, 0);
    endX = min(endX, width);
    endY = min(endY, height);

    // Calculate sum for each channel
    int sum[4] = {0, 0, 0, 0};
    int count = 0;

    for (int j = startY; j < endY; j++) {
      const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + j * nSrcStep);
      for (int i = startX; i < endX; i++) {
        for (int c = 0; c < 4; c++) {
          sum[c] += pSrcRow[i * 4 + c];
        }
        count++;
      }
    }

    // Calculate average and write output
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);
    if (count > 0) {
      for (int c = 0; c < 4; c++) {
        pDstRow[x * 4 + c] = (sum[c] + count / 2) / count; // Round to nearest
      }
    } else {
      for (int c = 0; c < 4; c++) {
        pDstRow[x * 4 + c] = 0;
      }
    }
  }
}

// Box filter kernel for single-channel 32-bit float
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

    // Boundary handling - clamp to image bounds
    startX = max(startX, 0);
    startY = max(startY, 0);
    endX = min(endX, width);
    endY = min(endY, height);

    // Calculate sum of all pixels in the region
    float sum = 0.0f;
    int count = 0;

    for (int j = startY; j < endY; j++) {
      const Npp32f *pSrcRow = (const Npp32f *)((const char *)pSrc + j * nSrcStep);
      for (int i = startX; i < endX; i++) {
        sum += pSrcRow[i];
        count++;
      }
    }

    // Calculate average and write output
    Npp32f *pDstRow = (Npp32f *)((char *)pDst + y * nDstStep);
    pDstRow[x] = (count > 0) ? (sum / count) : 0.0f;
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
                                         NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, cudaStream_t stream) {
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