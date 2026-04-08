#include "npp.h"
#include <cuda_runtime.h>

__device__ inline void yuv422_to_rgb_bt601(Npp8u y, Npp8u u, Npp8u v, Npp8u &r, Npp8u &g, Npp8u &b) {
  float Y = static_cast<float>(y);
  float U = static_cast<float>(u) - 128.0f;
  float V = static_cast<float>(v) - 128.0f;

  float R = Y + 1.140f * V;
  float G = Y - 0.395f * U - 0.581f * V;
  float B = Y + 2.032f * U;

  r = static_cast<Npp8u>(fminf(fmaxf(R, 0.0f), 255.0f));
  g = static_cast<Npp8u>(fminf(fmaxf(G, 0.0f), 255.0f));
  b = static_cast<Npp8u>(fminf(fmaxf(B, 0.0f), 255.0f));
}

__global__ void yuv422_to_rgb_p3c3_kernel(const Npp8u *__restrict__ srcY, int srcYStep, const Npp8u *__restrict__ srcU,
                                          int srcUStep, const Npp8u *__restrict__ srcV, int srcVStep,
                                          Npp8u *__restrict__ dst, int dstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  Npp8u Y = srcY[y * srcYStep + x];
  int uv_x = x >> 1;
  Npp8u U = srcU[y * srcUStep + uv_x];
  Npp8u V = srcV[y * srcVStep + uv_x];

  Npp8u R, G, B;
  yuv422_to_rgb_bt601(Y, U, V, R, G, B);

  int dst_offset = y * dstStep + x * 3;
  dst[dst_offset] = R;
  dst[dst_offset + 1] = G;
  dst[dst_offset + 2] = B;
}

__global__ void yuv422_to_rgb_p3p3_kernel(const Npp8u *__restrict__ srcY, int srcYStep, const Npp8u *__restrict__ srcU,
                                          int srcUStep, const Npp8u *__restrict__ srcV, int srcVStep,
                                          Npp8u *__restrict__ dstR, Npp8u *__restrict__ dstG, Npp8u *__restrict__ dstB,
                                          int dstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  Npp8u Y = srcY[y * srcYStep + x];
  int uv_x = x >> 1;
  Npp8u U = srcU[y * srcUStep + uv_x];
  Npp8u V = srcV[y * srcVStep + uv_x];

  Npp8u R, G, B;
  yuv422_to_rgb_bt601(Y, U, V, R, G, B);

  dstR[y * dstStep + x] = R;
  dstG[y * dstStep + x] = G;
  dstB[y * dstStep + x] = B;
}

__global__ void yuv422_to_rgb_p3ac4_kernel(const Npp8u *__restrict__ srcY, int srcYStep, const Npp8u *__restrict__ srcU,
                                           int srcUStep, const Npp8u *__restrict__ srcV, int srcVStep,
                                           Npp8u *__restrict__ dst, int dstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  Npp8u Y = srcY[y * srcYStep + x];
  int uv_x = x >> 1;
  Npp8u U = srcU[y * srcUStep + uv_x];
  Npp8u V = srcV[y * srcVStep + uv_x];

  Npp8u R, G, B;
  yuv422_to_rgb_bt601(Y, U, V, R, G, B);

  int dst_offset = y * dstStep + x * 4;
  dst[dst_offset] = R;
  dst[dst_offset + 1] = G;
  dst[dst_offset + 2] = B;
  dst[dst_offset + 3] = 0; // alpha cleared
}

__global__ void yuv422_to_rgb_c2c3_kernel(const Npp8u *__restrict__ src, int srcStep, Npp8u *__restrict__ dst,
                                          int dstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *row = src + y * srcStep;
  int pair = (x >> 1) * 4;
  Npp8u y0 = row[pair + 0];
  Npp8u u0 = row[pair + 1];
  Npp8u y1 = row[pair + 2];
  Npp8u v0 = row[pair + 3];
  Npp8u Y = (x & 1) ? y1 : y0;

  Npp8u R, G, B;
  yuv422_to_rgb_bt601(Y, u0, v0, R, G, B);

  int dst_offset = y * dstStep + x * 3;
  dst[dst_offset] = R;
  dst[dst_offset + 1] = G;
  dst[dst_offset + 2] = B;
}

extern "C" cudaError_t nppiYUV422ToRGB_8u_P3C3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcU,
                                                        int nSrcUStep, const Npp8u *pSrcV, int nSrcVStep, Npp8u *pDst,
                                                        int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  yuv422_to_rgb_p3c3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcY, nSrcYStep, pSrcU, nSrcUStep, pSrcV, nSrcVStep,
                                                                pDst, nDstStep, oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

extern "C" cudaError_t nppiYUV422ToRGB_8u_P3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcU,
                                                      int nSrcUStep, const Npp8u *pSrcV, int nSrcVStep, Npp8u *pDstR,
                                                      Npp8u *pDstG, Npp8u *pDstB, int nDstStep, NppiSize oSizeROI,
                                                      cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  yuv422_to_rgb_p3p3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcY, nSrcYStep, pSrcU, nSrcUStep, pSrcV, nSrcVStep,
                                                                pDstR, pDstG, pDstB, nDstStep, oSizeROI.width,
                                                                oSizeROI.height);
  return cudaGetLastError();
}

extern "C" cudaError_t nppiYUV422ToRGB_8u_P3AC4R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcU,
                                                         int nSrcUStep, const Npp8u *pSrcV, int nSrcVStep, Npp8u *pDst,
                                                         int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  yuv422_to_rgb_p3ac4_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcY, nSrcYStep, pSrcU, nSrcUStep, pSrcV, nSrcVStep,
                                                                 pDst, nDstStep, oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

extern "C" cudaError_t nppiYUV422ToRGB_8u_C2C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                                        NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  yuv422_to_rgb_c2c3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                oSizeROI.height);
  return cudaGetLastError();
}
