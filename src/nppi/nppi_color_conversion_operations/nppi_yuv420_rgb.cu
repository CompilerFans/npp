#include "nppdefs.h"
#include <cuda_runtime.h>

__device__ inline Npp8u clamp_u8(float v) { return (Npp8u)fmaxf(0.0f, fminf(255.0f, v)); }

__device__ inline void yuv_to_rgb(Npp8u y, Npp8u u, Npp8u v, Npp8u &r, Npp8u &g, Npp8u &b) {
  float fy = (float)y;
  float fu = (float)u - 128.0f;
  float fv = (float)v - 128.0f;

  float fr = fy + 1.140f * fv;
  float fg = fy - 0.394f * fu - 0.581f * fv;
  float fb = fy + 2.032f * fu;

  r = clamp_u8(fr + 0.5f);
  g = clamp_u8(fg + 0.5f);
  b = clamp_u8(fb + 0.5f);
}

__global__ void yuv420_to_rgb_p3c3_kernel(const Npp8u *pSrcY, const Npp8u *pSrcU, const Npp8u *pSrcV, int srcStepY,
                                          int srcStepU, int srcStepV, Npp8u *pDst, int dstStep, int width,
                                          int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *rowY = (const Npp8u *)((const char *)pSrcY + y * srcStepY);
  const Npp8u *rowU = (const Npp8u *)((const char *)pSrcU + (y >> 1) * srcStepU);
  const Npp8u *rowV = (const Npp8u *)((const char *)pSrcV + (y >> 1) * srcStepV);

  Npp8u Y = rowY[x];
  Npp8u U = rowU[x >> 1];
  Npp8u V = rowV[x >> 1];

  Npp8u r, g, b;
  yuv_to_rgb(Y, U, V, r, g, b);

  Npp8u *dstRow = (Npp8u *)((char *)pDst + y * dstStep);
  int idx = x * 3;
  dstRow[idx + 0] = r;
  dstRow[idx + 1] = g;
  dstRow[idx + 2] = b;
}

__global__ void yuv420_to_rgb_p3c4_kernel(const Npp8u *pSrcY, const Npp8u *pSrcU, const Npp8u *pSrcV, int srcStepY,
                                          int srcStepU, int srcStepV, Npp8u *pDst, int dstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *rowY = (const Npp8u *)((const char *)pSrcY + y * srcStepY);
  const Npp8u *rowU = (const Npp8u *)((const char *)pSrcU + (y >> 1) * srcStepU);
  const Npp8u *rowV = (const Npp8u *)((const char *)pSrcV + (y >> 1) * srcStepV);

  Npp8u Y = rowY[x];
  Npp8u U = rowU[x >> 1];
  Npp8u V = rowV[x >> 1];

  Npp8u r, g, b;
  yuv_to_rgb(Y, U, V, r, g, b);

  Npp8u *dstRow = (Npp8u *)((char *)pDst + y * dstStep);
  int idx = x * 4;
  dstRow[idx + 0] = r;
  dstRow[idx + 1] = g;
  dstRow[idx + 2] = b;
  dstRow[idx + 3] = 0xFF;
}

__global__ void yuv420_to_rgb_p3ac4_kernel(const Npp8u *pSrcY, const Npp8u *pSrcU, const Npp8u *pSrcV, int srcStepY,
                                           int srcStepU, int srcStepV, Npp8u *pDst, int dstStep, int width,
                                           int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *rowY = (const Npp8u *)((const char *)pSrcY + y * srcStepY);
  const Npp8u *rowU = (const Npp8u *)((const char *)pSrcU + (y >> 1) * srcStepU);
  const Npp8u *rowV = (const Npp8u *)((const char *)pSrcV + (y >> 1) * srcStepV);

  Npp8u Y = rowY[x];
  Npp8u U = rowU[x >> 1];
  Npp8u V = rowV[x >> 1];

  Npp8u r, g, b;
  yuv_to_rgb(Y, U, V, r, g, b);

  Npp8u *dstRow = (Npp8u *)((char *)pDst + y * dstStep);
  int idx = x * 4;
  Npp8u a = dstRow[idx + 3];
  dstRow[idx + 0] = r;
  dstRow[idx + 1] = g;
  dstRow[idx + 2] = b;
  dstRow[idx + 3] = a;
}

__global__ void yuv420_to_rgb_p3r_kernel(const Npp8u *pSrcY, const Npp8u *pSrcU, const Npp8u *pSrcV, int srcStepY,
                                         int srcStepU, int srcStepV, Npp8u *pDstR, Npp8u *pDstG, Npp8u *pDstB,
                                         int dstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *rowY = (const Npp8u *)((const char *)pSrcY + y * srcStepY);
  const Npp8u *rowU = (const Npp8u *)((const char *)pSrcU + (y >> 1) * srcStepU);
  const Npp8u *rowV = (const Npp8u *)((const char *)pSrcV + (y >> 1) * srcStepV);

  Npp8u Y = rowY[x];
  Npp8u U = rowU[x >> 1];
  Npp8u V = rowV[x >> 1];

  Npp8u r, g, b;
  yuv_to_rgb(Y, U, V, r, g, b);

  Npp8u *rowR = (Npp8u *)((char *)pDstR + y * dstStep);
  Npp8u *rowG = (Npp8u *)((char *)pDstG + y * dstStep);
  Npp8u *rowB = (Npp8u *)((char *)pDstB + y * dstStep);

  rowR[x] = r;
  rowG[x] = g;
  rowB[x] = b;
}

extern "C" {
cudaError_t nppiYUV420ToRGB_8u_P3R_kernel(const Npp8u *pSrcY, const Npp8u *pSrcU, const Npp8u *pSrcV, int srcStepY,
                                          int srcStepU, int srcStepV, Npp8u *pDstR, Npp8u *pDstG, Npp8u *pDstB,
                                          int dstStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  yuv420_to_rgb_p3r_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcY, pSrcU, pSrcV, srcStepY, srcStepU, srcStepV, pDstR,
                                                               pDstG, pDstB, dstStep, oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiYUV420ToRGB_8u_P3C3R_kernel(const Npp8u *pSrcY, const Npp8u *pSrcU, const Npp8u *pSrcV, int srcStepY,
                                            int srcStepU, int srcStepV, Npp8u *pDst, int dstStep, NppiSize oSizeROI,
                                            cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  yuv420_to_rgb_p3c3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcY, pSrcU, pSrcV, srcStepY, srcStepU, srcStepV, pDst,
                                                                dstStep, oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiYUV420ToRGB_8u_P3C4R_kernel(const Npp8u *pSrcY, const Npp8u *pSrcU, const Npp8u *pSrcV, int srcStepY,
                                            int srcStepU, int srcStepV, Npp8u *pDst, int dstStep, NppiSize oSizeROI,
                                            cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  yuv420_to_rgb_p3c4_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcY, pSrcU, pSrcV, srcStepY, srcStepU, srcStepV, pDst,
                                                                dstStep, oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiYUV420ToRGB_8u_P3AC4R_kernel(const Npp8u *pSrcY, const Npp8u *pSrcU, const Npp8u *pSrcV, int srcStepY,
                                             int srcStepU, int srcStepV, Npp8u *pDst, int dstStep, NppiSize oSizeROI,
                                             cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  yuv420_to_rgb_p3ac4_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcY, pSrcU, pSrcV, srcStepY, srcStepU, srcStepV, pDst,
                                                                 dstStep, oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}
}
