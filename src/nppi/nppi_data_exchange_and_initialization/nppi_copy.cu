#include "npp.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * CUDA kernels for NPP Image Copy Functions
 * Implements efficient GPU-based image copying operations
 */

// Kernel for 8-bit unsigned single channel copy
__global__ void nppiCopy_8u_C1R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                       int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
  Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

  dst_row[x] = src_row[x];
}

// Kernel for 8-bit unsigned three channel copy
__global__ void nppiCopy_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                       int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
  Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

  // Copy all three channels
  dst_row[x * 3 + 0] = src_row[x * 3 + 0]; // R or B
  dst_row[x * 3 + 1] = src_row[x * 3 + 1]; // G
  dst_row[x * 3 + 2] = src_row[x * 3 + 2]; // B or R
}

// Kernel for 8-bit unsigned four channel copy
__global__ void nppiCopy_8u_C4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                       int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
  Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

  // Copy all four channels (RGBA/BGRA)
  dst_row[x * 4 + 0] = src_row[x * 4 + 0];
  dst_row[x * 4 + 1] = src_row[x * 4 + 1];
  dst_row[x * 4 + 2] = src_row[x * 4 + 2];
  dst_row[x * 4 + 3] = src_row[x * 4 + 3];
}

// Kernel for 32-bit float single channel copy
__global__ void nppiCopy_32f_C1R_kernel(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, int width,
                                        int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const Npp32f *src_row = (const Npp32f *)((const char *)pSrc + y * nSrcStep);
  Npp32f *dst_row = (Npp32f *)((char *)pDst + y * nDstStep);

  dst_row[x] = src_row[x];
}

// Kernel for 32-bit float three channel copy
__global__ void nppiCopy_32f_C3R_kernel(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, int width,
                                        int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const Npp32f *src_row = (const Npp32f *)((const char *)pSrc + y * nSrcStep);
  Npp32f *dst_row = (Npp32f *)((char *)pDst + y * nDstStep);

  // Copy all three channels
  dst_row[x * 3 + 0] = src_row[x * 3 + 0]; // Channel 0 (R)
  dst_row[x * 3 + 1] = src_row[x * 3 + 1]; // Channel 1 (G)
  dst_row[x * 3 + 2] = src_row[x * 3 + 2]; // Channel 2 (B)
}

// Kernel for 32-bit float packed to planar copy (C3P3R = packed to planar)
__global__ void nppiCopy_32f_C3P3R_kernel(const Npp32f *pSrc, int nSrcStep, Npp32f *const *pDst, int nDstStep,
                                          int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  // Read from packed source (RGB interleaved)
  const Npp32f *src_row = (const Npp32f *)((const char *)pSrc + y * nSrcStep);

  // Write to planar destinations (separate channels)
  for (int c = 0; c < 3; c++) {
    Npp32f *dst_row = (Npp32f *)((char *)pDst[c] + y * nDstStep);
    dst_row[x] = src_row[x * 3 + c]; // Extract channel c from packed format
  }
}

// Optimized kernel using vectorized loads for better memory throughput
__global__ void nppiCopy_8u_C4R_vectorized_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                                  int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const uint32_t *src_row = (const uint32_t *)((const char *)pSrc + y * nSrcStep);
  uint32_t *dst_row = (uint32_t *)((char *)pDst + y * nDstStep);

  // Copy 4 bytes (1 pixel) at once
  dst_row[x] = src_row[x];
}

extern "C" {

// 8-bit unsigned single channel copy implementation
NppStatus nppiCopy_8u_C1R_Ctx_cuda(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiCopy_8u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                           oSizeROI.width, oSizeROI.height);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 8-bit unsigned three channel copy implementation
NppStatus nppiCopy_8u_C3R_Ctx_cuda(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiCopy_8u_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                           oSizeROI.width, oSizeROI.height);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 8-bit unsigned four channel copy implementation
NppStatus nppiCopy_8u_C4R_Ctx_cuda(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  // Use vectorized version if width is aligned to 4-byte boundaries
  if (oSizeROI.width % 4 == 0 && nSrcStep % 4 == 0 && nDstStep % 4 == 0) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    nppiCopy_8u_C4R_vectorized_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height);
  } else {
    // Fall back to regular implementation
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    nppiCopy_8u_C4R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                             oSizeROI.width, oSizeROI.height);
  }

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 32-bit float single channel copy implementation
NppStatus nppiCopy_32f_C1R_Ctx_cuda(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiCopy_32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                            oSizeROI.width, oSizeROI.height);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 32-bit float three channel copy implementation
NppStatus nppiCopy_32f_C3R_Ctx_cuda(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiCopy_32f_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                            oSizeROI.width, oSizeROI.height);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 32-bit float packed to planar copy implementation
NppStatus nppiCopy_32f_C3P3R_Ctx_cuda(const Npp32f *pSrc, int nSrcStep, Npp32f *const pDst[3], int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  // Copy destination pointers to device memory for kernel access
  Npp32f **d_pDst;

  cudaError_t cudaStatus = cudaMalloc(&d_pDst, 3 * sizeof(Npp32f *));
  if (cudaStatus != cudaSuccess) {
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  cudaStatus = cudaMemcpy(d_pDst, pDst, 3 * sizeof(Npp32f *), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    cudaFree(d_pDst);
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiCopy_32f_C3P3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, d_pDst, nDstStep,
                                                                              oSizeROI.width, oSizeROI.height);

  cudaStatus = cudaGetLastError();

  // Clean up device memory
  cudaFree(d_pDst);

  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}
}