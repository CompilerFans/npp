#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Color twist kernel for 3-channel 8-bit images
__global__ void nppiColorTwist32f_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                                int height, const float *twist) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
  Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

  int src_idx = x * 3;
  int dst_idx = x * 3;

  // Get source RGB values
  float r = (float)src_row[src_idx + 0];
  float g = (float)src_row[src_idx + 1];
  float b = (float)src_row[src_idx + 2];

  // Apply color twist matrix: [R' G' B']^T = [twist] * [R G B 1]^T
  float r_new = twist[0] * r + twist[1] * g + twist[2] * b + twist[3];
  float g_new = twist[4] * r + twist[5] * g + twist[6] * b + twist[7];
  float b_new = twist[8] * r + twist[9] * g + twist[10] * b + twist[11];

  // Clamp and store results
  dst_row[dst_idx + 0] = (Npp8u)fmaxf(0.0f, fminf(255.0f, r_new + 0.5f));
  dst_row[dst_idx + 1] = (Npp8u)fmaxf(0.0f, fminf(255.0f, g_new + 0.5f));
  dst_row[dst_idx + 2] = (Npp8u)fmaxf(0.0f, fminf(255.0f, b_new + 0.5f));
}

// Color twist kernel for 4-channel 8-bit images (alpha copy)
__global__ void nppiColorTwist32f_8u_C4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                                int height, const float *twist) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
  Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

  int src_idx = x * 4;
  int dst_idx = x * 4;

  float r = (float)src_row[src_idx + 0];
  float g = (float)src_row[src_idx + 1];
  float b = (float)src_row[src_idx + 2];
  Npp8u a = src_row[src_idx + 3];

  float r_new = twist[0] * r + twist[1] * g + twist[2] * b + twist[3];
  float g_new = twist[4] * r + twist[5] * g + twist[6] * b + twist[7];
  float b_new = twist[8] * r + twist[9] * g + twist[10] * b + twist[11];

  dst_row[dst_idx + 0] = (Npp8u)fmaxf(0.0f, fminf(255.0f, r_new + 0.5f));
  dst_row[dst_idx + 1] = (Npp8u)fmaxf(0.0f, fminf(255.0f, g_new + 0.5f));
  dst_row[dst_idx + 2] = (Npp8u)fmaxf(0.0f, fminf(255.0f, b_new + 0.5f));
  dst_row[dst_idx + 3] = a;
}

// Color twist kernel for AC4 (alpha output zero)
__global__ void nppiColorTwist32f_8u_AC4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                                 int width, int height, const float *twist) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
  Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

  int src_idx = x * 4;
  int dst_idx = x * 4;

  float r = (float)src_row[src_idx + 0];
  float g = (float)src_row[src_idx + 1];
  float b = (float)src_row[src_idx + 2];

  float r_new = twist[0] * r + twist[1] * g + twist[2] * b + twist[3];
  float g_new = twist[4] * r + twist[5] * g + twist[6] * b + twist[7];
  float b_new = twist[8] * r + twist[9] * g + twist[10] * b + twist[11];

  dst_row[dst_idx + 0] = (Npp8u)fmaxf(0.0f, fminf(255.0f, r_new + 0.5f));
  dst_row[dst_idx + 1] = (Npp8u)fmaxf(0.0f, fminf(255.0f, g_new + 0.5f));
  dst_row[dst_idx + 2] = (Npp8u)fmaxf(0.0f, fminf(255.0f, b_new + 0.5f));
  dst_row[dst_idx + 3] = 0;
}

// Color twist kernel for single-channel 8-bit images (grayscale)
__global__ void nppiColorTwist32f_8u_C1R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                                int height, const float *twist) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
  Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

  // Get source grayscale value
  float gray = (float)src_row[x];

  // Apply first row of color twist matrix (treat as RGB->R transformation)
  float result = twist[0] * gray + twist[1] * gray + twist[2] * gray + twist[3];

  // Clamp and store result
  dst_row[x] = (Npp8u)fmaxf(0.0f, fminf(255.0f, result + 0.5f));
}

extern "C" {

// 3-channel 8-bit implementation
NppStatus nppiColorTwist32f_8u_C3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                            NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                            NppStreamContext nppStreamCtx) {
  // Copy twist matrix to device memory
  float *d_twist;
  cudaError_t cudaStatus = cudaMalloc(&d_twist, 12 * sizeof(float));
  if (cudaStatus != cudaSuccess) {
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  // Flatten 3x4 matrix to 1D array
  float h_twist[12];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      h_twist[i * 4 + j] = aTwist[i][j];
    }
  }

  cudaStatus = cudaMemcpyAsync(d_twist, h_twist, 12 * sizeof(float), cudaMemcpyHostToDevice, nppStreamCtx.hStream);
  if (cudaStatus != cudaSuccess) {
    cudaFree(d_twist);
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiColorTwist32f_8u_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, d_twist);

  cudaStatus = cudaGetLastError();

  // Sync stream and free memory
  cudaStreamSynchronize(nppStreamCtx.hStream);
  cudaFree(d_twist);

  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 4-channel 8-bit implementation (alpha copy)
NppStatus nppiColorTwist32f_8u_C4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                            NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                            NppStreamContext nppStreamCtx) {
  float *d_twist;
  cudaError_t cudaStatus = cudaMalloc(&d_twist, 12 * sizeof(float));
  if (cudaStatus != cudaSuccess) {
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  float h_twist[12];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      h_twist[i * 4 + j] = aTwist[i][j];
    }
  }

  cudaStatus = cudaMemcpyAsync(d_twist, h_twist, 12 * sizeof(float), cudaMemcpyHostToDevice, nppStreamCtx.hStream);
  if (cudaStatus != cudaSuccess) {
    cudaFree(d_twist);
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiColorTwist32f_8u_C4R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, d_twist);

  cudaStatus = cudaGetLastError();
  cudaStreamSynchronize(nppStreamCtx.hStream);
  cudaFree(d_twist);

  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// AC4 8-bit implementation (alpha zero)
NppStatus nppiColorTwist32f_8u_AC4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                             NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                             NppStreamContext nppStreamCtx) {
  float *d_twist;
  cudaError_t cudaStatus = cudaMalloc(&d_twist, 12 * sizeof(float));
  if (cudaStatus != cudaSuccess) {
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  float h_twist[12];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      h_twist[i * 4 + j] = aTwist[i][j];
    }
  }

  cudaStatus = cudaMemcpyAsync(d_twist, h_twist, 12 * sizeof(float), cudaMemcpyHostToDevice, nppStreamCtx.hStream);
  if (cudaStatus != cudaSuccess) {
    cudaFree(d_twist);
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiColorTwist32f_8u_AC4R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, d_twist);

  cudaStatus = cudaGetLastError();
  cudaStreamSynchronize(nppStreamCtx.hStream);
  cudaFree(d_twist);

  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// Single-channel 8-bit implementation
NppStatus nppiColorTwist32f_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                            NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                            NppStreamContext nppStreamCtx) {
  // Copy twist matrix to device memory
  float *d_twist;
  cudaError_t cudaStatus = cudaMalloc(&d_twist, 12 * sizeof(float));
  if (cudaStatus != cudaSuccess) {
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  // Flatten 3x4 matrix to 1D array
  float h_twist[12];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      h_twist[i * 4 + j] = aTwist[i][j];
    }
  }

  cudaStatus = cudaMemcpyAsync(d_twist, h_twist, 12 * sizeof(float), cudaMemcpyHostToDevice, nppStreamCtx.hStream);
  if (cudaStatus != cudaSuccess) {
    cudaFree(d_twist);
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiColorTwist32f_8u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, d_twist);

  cudaStatus = cudaGetLastError();

  // Sync stream and free memory
  cudaStreamSynchronize(nppStreamCtx.hStream);
  cudaFree(d_twist);

  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// In-place variants
NppStatus nppiColorTwist32f_8u_C1IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                             const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppiColorTwist32f_8u_C1R_Ctx_impl(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_C3IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                             const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppiColorTwist32f_8u_C3R_Ctx_impl(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_C4IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                             const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppiColorTwist32f_8u_C4R_Ctx_impl(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_AC4IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                              const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppiColorTwist32f_8u_C4R_Ctx_impl(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}
}
