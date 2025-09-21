#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * kernels for MPP Image Filter Functions
 * Implements general 2D convolution operations
 */

// Kernel for 8-bit unsigned single channel 2D convolution
__global__ void nppiFilter_8u_C1R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                         int height, const Npp32s *pKernel, int kernelWidth, int kernelHeight,
                                         int anchorX, int anchorY, Npp32s nDivisor) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

  // Perform convolution
  Npp32s sum = 0;

  for (int ky = 0; ky < kernelHeight; ky++) {
    for (int kx = 0; kx < kernelWidth; kx++) {
      int srcX = x + kx - anchorX;
      int srcY = y + ky - anchorY;

      // Clamp coordinates (zero padding)
      if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
        const Npp8u *src_kernel_row = (const Npp8u *)((const char *)pSrc + srcY * nSrcStep);
        sum += src_kernel_row[srcX] * pKernel[ky * kernelWidth + kx];
      }
    }
  }

  // Apply divisor and clamp to [0, 255]
  sum = sum / nDivisor;
  dst_row[x] = (Npp8u)max(0, min(255, sum));
}

// Kernel for 8-bit unsigned three channel 2D convolution
__global__ void nppiFilter_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                         int height, const Npp32s *pKernel, int kernelWidth, int kernelHeight,
                                         int anchorX, int anchorY, Npp32s nDivisor) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

  // Perform convolution for each channel
  for (int c = 0; c < 3; c++) {
    Npp32s sum = 0;

    for (int ky = 0; ky < kernelHeight; ky++) {
      for (int kx = 0; kx < kernelWidth; kx++) {
        int srcX = x + kx - anchorX;
        int srcY = y + ky - anchorY;

        // Clamp coordinates (zero padding)
        if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
          const Npp8u *src_kernel_row = (const Npp8u *)((const char *)pSrc + srcY * nSrcStep);
          sum += src_kernel_row[srcX * 3 + c] * pKernel[ky * kernelWidth + kx];
        }
      }
    }

    // Apply divisor and clamp to [0, 255]
    sum = sum / nDivisor;
    dst_row[x * 3 + c] = (Npp8u)max(0, min(255, sum));
  }
}

// Kernel for 32-bit float single channel 2D convolution
__global__ void nppiFilter_32f_C1R_kernel(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, int width,
                                          int height, const Npp32f *pKernel, int kernelWidth, int kernelHeight,
                                          int anchorX, int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  Npp32f *dst_row = (Npp32f *)((char *)pDst + y * nDstStep);

  // Perform convolution
  Npp32f sum = 0.0f;

  for (int ky = 0; ky < kernelHeight; ky++) {
    for (int kx = 0; kx < kernelWidth; kx++) {
      int srcX = x + kx - anchorX;
      int srcY = y + ky - anchorY;

      // Clamp coordinates (zero padding)
      if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
        const Npp32f *src_kernel_row = (const Npp32f *)((const char *)pSrc + srcY * nSrcStep);
        sum += src_kernel_row[srcX] * pKernel[ky * kernelWidth + kx];
      }
    }
  }

  dst_row[x] = sum;
}

extern "C" {

// 8-bit unsigned single channel 2D convolution implementation
NppStatus nppiFilter_8u_C1R_Ctx_cuda(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                     const Npp32s *pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor,
                                     NppStreamContext nppStreamCtx) {
  // Copy kernel to device memory
  Npp32s *d_kernel;
  int kernelSizeBytes = oKernelSize.width * oKernelSize.height * sizeof(Npp32s);
  cudaError_t cudaStatus = cudaMalloc(&d_kernel, kernelSizeBytes);
  if (cudaStatus != cudaSuccess) {
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  cudaStatus = cudaMemcpy(d_kernel, pKernel, kernelSizeBytes, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    cudaFree(d_kernel);
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiFilter_8u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, d_kernel, oKernelSize.width, oKernelSize.height,
      oAnchor.x, oAnchor.y, nDivisor);

  cudaStatus = cudaGetLastError();
  cudaFree(d_kernel);

  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 8-bit unsigned three channel 2D convolution implementation
NppStatus nppiFilter_8u_C3R_Ctx_cuda(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                     const Npp32s *pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor,
                                     NppStreamContext nppStreamCtx) {
  // Copy kernel to device memory
  Npp32s *d_kernel;
  int kernelSizeBytes = oKernelSize.width * oKernelSize.height * sizeof(Npp32s);
  cudaError_t cudaStatus = cudaMalloc(&d_kernel, kernelSizeBytes);
  if (cudaStatus != cudaSuccess) {
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  cudaStatus = cudaMemcpy(d_kernel, pKernel, kernelSizeBytes, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    cudaFree(d_kernel);
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiFilter_8u_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, d_kernel, oKernelSize.width, oKernelSize.height,
      oAnchor.x, oAnchor.y, nDivisor);

  cudaStatus = cudaGetLastError();
  cudaFree(d_kernel);

  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 32-bit float single channel 2D convolution implementation
NppStatus nppiFilter_32f_C1R_Ctx_cuda(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                      const Npp32f *pKernel, NppiSize oKernelSize, NppiPoint oAnchor,
                                      NppStreamContext nppStreamCtx) {
  // Copy kernel to device memory
  Npp32f *d_kernel;
  int kernelSizeBytes = oKernelSize.width * oKernelSize.height * sizeof(Npp32f);
  cudaError_t cudaStatus = cudaMalloc(&d_kernel, kernelSizeBytes);
  if (cudaStatus != cudaSuccess) {
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  cudaStatus = cudaMemcpy(d_kernel, pKernel, kernelSizeBytes, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    cudaFree(d_kernel);
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiFilter_32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, d_kernel, oKernelSize.width, oKernelSize.height,
      oAnchor.x, oAnchor.y);

  cudaStatus = cudaGetLastError();
  cudaFree(d_kernel);

  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}
}