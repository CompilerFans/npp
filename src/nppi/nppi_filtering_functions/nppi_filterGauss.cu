#include "npp.h"
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * 简化的CUDA kernels for NPP Image Gaussian Filter Functions
 * 修复内存访问问题
 */

// 简单的3x3高斯内核
__constant__ float c_gauss3x3[9] = {0.0625f, 0.125f, 0.0625f, 0.125f, 0.25f, 0.125f, 0.0625f, 0.125f, 0.0625f};

// 简单的5x5高斯内核
__constant__ float c_gauss5x5[25] = {0.0039f, 0.0156f, 0.0234f, 0.0156f, 0.0039f, 0.0156f, 0.0625f, 0.0938f, 0.0625f,
                                     0.0156f, 0.0234f, 0.0938f, 0.1406f, 0.0938f, 0.0234f, 0.0156f, 0.0625f, 0.0938f,
                                     0.0625f, 0.0156f, 0.0039f, 0.0156f, 0.0234f, 0.0156f, 0.0039f};

// 7x7高斯内核
__constant__ float c_gauss7x7[49] = {
    0.0009f, 0.0039f, 0.0078f, 0.0098f, 0.0078f, 0.0039f, 0.0009f, 0.0039f, 0.0156f, 0.0313f, 0.0391f, 0.0313f, 0.0156f,
    0.0039f, 0.0078f, 0.0313f, 0.0625f, 0.0781f, 0.0625f, 0.0313f, 0.0078f, 0.0098f, 0.0391f, 0.0781f, 0.0977f, 0.0781f,
    0.0391f, 0.0098f, 0.0078f, 0.0313f, 0.0625f, 0.0781f, 0.0625f, 0.0313f, 0.0078f, 0.0039f, 0.0156f, 0.0313f, 0.0391f,
    0.0313f, 0.0156f, 0.0039f, 0.0009f, 0.0039f, 0.0078f, 0.0098f, 0.0078f, 0.0039f, 0.0009f};

// 修复的8位单通道高斯滤波内核
__global__ void nppiFilterGauss_8u_C1R_kernel_fixed(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                                    int width, int height, int kernelSize) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  float result = 0.0f;
  int center = kernelSize / 2;

  for (int ky = 0; ky < kernelSize; ky++) {
    for (int kx = 0; kx < kernelSize; kx++) {
      int srcX = x + kx - center;
      int srcY = y + ky - center;

      // 边界处理：镜像
      srcX = max(0, min(srcX, width - 1));
      srcY = max(0, min(srcY, height - 1));

      // 正确的内存访问
      const Npp8u *src_pixel = (const Npp8u *)((const char *)pSrc + srcY * nSrcStep) + srcX;
      float pixelValue = (float)(*src_pixel);

      float kernelValue;
      if (kernelSize == 3) {
        kernelValue = c_gauss3x3[ky * kernelSize + kx];
      } else if (kernelSize == 5) {
        kernelValue = c_gauss5x5[ky * kernelSize + kx];
      } else { // kernelSize == 7
        kernelValue = c_gauss7x7[ky * kernelSize + kx];
      }

      result += pixelValue * kernelValue;
    }
  }

  // 输出到正确位置
  Npp8u *dst_pixel = (Npp8u *)((char *)pDst + y * nDstStep) + x;
  *dst_pixel = (Npp8u)fmaxf(0.0f, fminf(255.0f, result));
}

// 修复的8位三通道高斯滤波内核
__global__ void nppiFilterGauss_8u_C3R_kernel_fixed(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                                    int width, int height, int kernelSize) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int center = kernelSize / 2;

  // 处理每个通道
  for (int ch = 0; ch < 3; ch++) {
    float result = 0.0f;

    for (int ky = 0; ky < kernelSize; ky++) {
      for (int kx = 0; kx < kernelSize; kx++) {
        int srcX = x + kx - center;
        int srcY = y + ky - center;

        // 边界处理：镜像
        srcX = max(0, min(srcX, width - 1));
        srcY = max(0, min(srcY, height - 1));

        // 正确的内存访问（3通道）
        const Npp8u *src_pixel = (const Npp8u *)((const char *)pSrc + srcY * nSrcStep) + srcX * 3 + ch;
        float pixelValue = (float)(*src_pixel);

        float kernelValue;
        if (kernelSize == 3) {
          kernelValue = c_gauss3x3[ky * kernelSize + kx];
        } else if (kernelSize == 5) {
          kernelValue = c_gauss5x5[ky * kernelSize + kx];
        } else { // kernelSize == 7
          kernelValue = c_gauss7x7[ky * kernelSize + kx];
        }

        result += pixelValue * kernelValue;
      }
    }

    // 输出到正确位置（3通道）
    Npp8u *dst_pixel = (Npp8u *)((char *)pDst + y * nDstStep) + x * 3 + ch;
    *dst_pixel = (Npp8u)fmaxf(0.0f, fminf(255.0f, result));
  }
}

// 修复的32位浮点单通道高斯滤波内核
__global__ void nppiFilterGauss_32f_C1R_kernel_fixed(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                                     int width, int height, int kernelSize) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  float result = 0.0f;
  int center = kernelSize / 2;

  for (int ky = 0; ky < kernelSize; ky++) {
    for (int kx = 0; kx < kernelSize; kx++) {
      int srcX = x + kx - center;
      int srcY = y + ky - center;

      // 边界处理：镜像
      srcX = max(0, min(srcX, width - 1));
      srcY = max(0, min(srcY, height - 1));

      // 正确的内存访问（32位浮点）
      const Npp32f *src_pixel = (const Npp32f *)((const char *)pSrc + srcY * nSrcStep) + srcX;
      float pixelValue = *src_pixel;

      float kernelValue;
      if (kernelSize == 3) {
        kernelValue = c_gauss3x3[ky * kernelSize + kx];
      } else if (kernelSize == 5) {
        kernelValue = c_gauss5x5[ky * kernelSize + kx];
      } else { // kernelSize == 7
        kernelValue = c_gauss7x7[ky * kernelSize + kx];
      }

      result += pixelValue * kernelValue;
    }
  }

  // 输出到正确位置
  Npp32f *dst_pixel = (Npp32f *)((char *)pDst + y * nDstStep) + x;
  *dst_pixel = result;
}

extern "C" {

// 8位单通道高斯滤波实现
NppStatus nppiFilterGauss_8u_C1R_Ctx_cuda_fixed(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                                NppiSize oSizeROI, NppiMaskSize eMaskSize,
                                                NppStreamContext nppStreamCtx) {
  // Early return for zero-size ROI (NVIDIA NPP compatible behavior)
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }

  int kernelSize;

  switch (eMaskSize) {
  case NPP_MASK_SIZE_3_X_3:
    kernelSize = 3;
    break;
  case NPP_MASK_SIZE_5_X_5:
    kernelSize = 5;
    break;
  case NPP_MASK_SIZE_7_X_7:
    kernelSize = 7;
    break;
  default:
    return NPP_NOT_SUPPORTED_MODE_ERROR;
  }

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiFilterGauss_8u_C1R_kernel_fixed<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, kernelSize);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  // 同步等待内核完成
  if (nppStreamCtx.hStream == 0) {
    cudaStatus = cudaDeviceSynchronize();
  } else {
    cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
  }
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 8位三通道高斯滤波实现
NppStatus nppiFilterGauss_8u_C3R_Ctx_cuda_fixed(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                                NppiSize oSizeROI, NppiMaskSize eMaskSize,
                                                NppStreamContext nppStreamCtx) {
  // Early return for zero-size ROI (NVIDIA NPP compatible behavior)
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }

  int kernelSize;

  switch (eMaskSize) {
  case NPP_MASK_SIZE_3_X_3:
    kernelSize = 3;
    break;
  case NPP_MASK_SIZE_5_X_5:
    kernelSize = 5;
    break;
  case NPP_MASK_SIZE_7_X_7:
    kernelSize = 7;
    break;
  default:
    return NPP_NOT_SUPPORTED_MODE_ERROR;
  }

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiFilterGauss_8u_C3R_kernel_fixed<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, kernelSize);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  // 同步等待内核完成
  if (nppStreamCtx.hStream == 0) {
    cudaStatus = cudaDeviceSynchronize();
  } else {
    cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
  }
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 32位浮点单通道高斯滤波实现
NppStatus nppiFilterGauss_32f_C1R_Ctx_cuda_fixed(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                                 NppiSize oSizeROI, NppiMaskSize eMaskSize,
                                                 NppStreamContext nppStreamCtx) {
  // Early return for zero-size ROI (NVIDIA NPP compatible behavior)
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }

  int kernelSize;

  switch (eMaskSize) {
  case NPP_MASK_SIZE_3_X_3:
    kernelSize = 3;
    break;
  case NPP_MASK_SIZE_5_X_5:
    kernelSize = 5;
    break;
  case NPP_MASK_SIZE_7_X_7:
    kernelSize = 7;
    break;
  default:
    return NPP_NOT_SUPPORTED_MODE_ERROR;
  }

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiFilterGauss_32f_C1R_kernel_fixed<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, kernelSize);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  // 同步等待内核完成
  if (nppStreamCtx.hStream == 0) {
    cudaStatus = cudaDeviceSynchronize();
  } else {
    cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
  }
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

} // extern "C"