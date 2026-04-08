#include "nppdefs.h"
#include <cuda_runtime.h>

__device__ inline Npp8u clamp_u8_int(int v) { return static_cast<Npp8u>(max(0, min(255, v))); }

__device__ inline void rgb_to_yuv_pixel_floor(Npp8u r, Npp8u g, Npp8u b, Npp8u &y, Npp8u &u, Npp8u &v) {
  float R = static_cast<float>(r);
  float G = static_cast<float>(g);
  float B = static_cast<float>(b);

  float Y = 0.299f * R + 0.587f * G + 0.114f * B;
  float U = -0.147f * R - 0.289f * G + 0.436f * B + 128.0f;
  float V = 0.615f * R - 0.515f * G - 0.100f * B + 128.0f;

  int yi = static_cast<int>(Y);
  int ui = static_cast<int>(U);
  int vi = static_cast<int>(V);

  y = clamp_u8_int(yi);
  u = clamp_u8_int(ui);
  v = clamp_u8_int(vi);
}

__global__ void rgb_to_yuv422_y_c3_kernel(const Npp8u *__restrict__ src, int srcStep, Npp8u *__restrict__ dstY,
                                          int dstYStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *srcRow = (const Npp8u *)((const char *)src + y * srcStep);
  int idx = x * 3;
  Npp8u r = srcRow[idx + 0];
  Npp8u g = srcRow[idx + 1];
  Npp8u b = srcRow[idx + 2];

  Npp8u yv, uv, vv;
  rgb_to_yuv_pixel_floor(r, g, b, yv, uv, vv);

  Npp8u *dstRow = (Npp8u *)((char *)dstY + y * dstYStep);
  dstRow[x] = yv;
}

__global__ void rgb_to_yuv422_uv_c3_kernel(const Npp8u *__restrict__ src, int srcStep, Npp8u *__restrict__ dstU,
                                           int dstUStep, Npp8u *__restrict__ dstV, int dstVStep, int width,
                                           int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int uvWidth = width >> 1;
  if (x >= uvWidth || y >= height) {
    return;
  }

  int baseX = x << 1;
  const Npp8u *srcRow = (const Npp8u *)((const char *)src + y * srcStep);

  Npp8u y0, u0, v0;
  Npp8u y1, u1, v1;

  int idx0 = baseX * 3;
  int idx1 = (baseX + 1) * 3;

  rgb_to_yuv_pixel_floor(srcRow[idx0 + 0], srcRow[idx0 + 1], srcRow[idx0 + 2], y0, u0, v0);
  rgb_to_yuv_pixel_floor(srcRow[idx1 + 0], srcRow[idx1 + 1], srcRow[idx1 + 2], y1, u1, v1);

  Npp8u *dstRowU = (Npp8u *)((char *)dstU + y * dstUStep);
  Npp8u *dstRowV = (Npp8u *)((char *)dstV + y * dstVStep);
  dstRowU[x] = static_cast<Npp8u>((u0 + u1) / 2);
  dstRowV[x] = static_cast<Npp8u>((v0 + v1) / 2);
}

__global__ void rgb_to_yuv422_y_p3_kernel(const Npp8u *__restrict__ srcR, const Npp8u *__restrict__ srcG,
                                          const Npp8u *__restrict__ srcB, int srcStep, Npp8u *__restrict__ dstY,
                                          int dstYStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *rowR = (const Npp8u *)((const char *)srcR + y * srcStep);
  const Npp8u *rowG = (const Npp8u *)((const char *)srcG + y * srcStep);
  const Npp8u *rowB = (const Npp8u *)((const char *)srcB + y * srcStep);

  Npp8u yv, uv, vv;
  rgb_to_yuv_pixel_floor(rowR[x], rowG[x], rowB[x], yv, uv, vv);

  Npp8u *dstRow = (Npp8u *)((char *)dstY + y * dstYStep);
  dstRow[x] = yv;
}

__global__ void rgb_to_yuv422_uv_p3_kernel(const Npp8u *__restrict__ srcR, const Npp8u *__restrict__ srcG,
                                           const Npp8u *__restrict__ srcB, int srcStep, Npp8u *__restrict__ dstU,
                                           int dstUStep, Npp8u *__restrict__ dstV, int dstVStep, int width,
                                           int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int uvWidth = width >> 1;
  if (x >= uvWidth || y >= height) {
    return;
  }

  int baseX = x << 1;
  const Npp8u *rowR = (const Npp8u *)((const char *)srcR + y * srcStep);
  const Npp8u *rowG = (const Npp8u *)((const char *)srcG + y * srcStep);
  const Npp8u *rowB = (const Npp8u *)((const char *)srcB + y * srcStep);

  Npp8u y0, u0, v0;
  Npp8u y1, u1, v1;
  rgb_to_yuv_pixel_floor(rowR[baseX], rowG[baseX], rowB[baseX], y0, u0, v0);
  rgb_to_yuv_pixel_floor(rowR[baseX + 1], rowG[baseX + 1], rowB[baseX + 1], y1, u1, v1);

  Npp8u *dstRowU = (Npp8u *)((char *)dstU + y * dstUStep);
  Npp8u *dstRowV = (Npp8u *)((char *)dstV + y * dstVStep);
  dstRowU[x] = static_cast<Npp8u>((u0 + u1) / 2);
  dstRowV[x] = static_cast<Npp8u>((v0 + v1) / 2);
}

__global__ void rgb_to_yuv422_c2_kernel(const Npp8u *__restrict__ src, int srcStep, Npp8u *__restrict__ dst,
                                        int dstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int pairWidth = width >> 1;
  if (x >= pairWidth || y >= height) {
    return;
  }

  int baseX = x << 1;
  const Npp8u *srcRow = (const Npp8u *)((const char *)src + y * srcStep);
  Npp8u *dstRow = (Npp8u *)((char *)dst + y * dstStep);

  Npp8u y0, u0, v0;
  Npp8u y1, u1, v1;
  int idx0 = baseX * 3;
  int idx1 = (baseX + 1) * 3;
  rgb_to_yuv_pixel_floor(srcRow[idx0 + 0], srcRow[idx0 + 1], srcRow[idx0 + 2], y0, u0, v0);
  rgb_to_yuv_pixel_floor(srcRow[idx1 + 0], srcRow[idx1 + 1], srcRow[idx1 + 2], y1, u1, v1);

  Npp8u u = static_cast<Npp8u>((u0 + u1) / 2);
  Npp8u v = static_cast<Npp8u>((v0 + v1) / 2);

  int outIdx = x * 4;
  dstRow[outIdx + 0] = y0;
  dstRow[outIdx + 1] = u;
  dstRow[outIdx + 2] = y1;
  dstRow[outIdx + 3] = v;
}

extern "C" cudaError_t nppiRGBToYUV422_8u_C3P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, int nDstYStep,
                                                       Npp8u *pDstU, int nDstUStep, Npp8u *pDstV, int nDstVStep,
                                                       NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  rgb_to_yuv422_y_c3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDstY, nDstYStep, oSizeROI.width,
                                                                oSizeROI.height);

  dim3 uvGridSize(((oSizeROI.width / 2) + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);
  rgb_to_yuv422_uv_c3_kernel<<<uvGridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDstU, nDstUStep, pDstV, nDstVStep,
                                                                   oSizeROI.width, oSizeROI.height);

  return cudaGetLastError();
}

extern "C" cudaError_t nppiRGBToYUV422_8u_P3R_kernel(const Npp8u *pSrcR, const Npp8u *pSrcG, const Npp8u *pSrcB,
                                                     int nSrcStep, Npp8u *pDstY, int nDstYStep, Npp8u *pDstU,
                                                     int nDstUStep, Npp8u *pDstV, int nDstVStep, NppiSize oSizeROI,
                                                     cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  rgb_to_yuv422_y_p3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcR, pSrcG, pSrcB, nSrcStep, pDstY, nDstYStep,
                                                                oSizeROI.width, oSizeROI.height);

  dim3 uvGridSize(((oSizeROI.width / 2) + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);
  rgb_to_yuv422_uv_p3_kernel<<<uvGridSize, blockSize, 0, stream>>>(pSrcR, pSrcG, pSrcB, nSrcStep, pDstU, nDstUStep,
                                                                   pDstV, nDstVStep, oSizeROI.width,
                                                                   oSizeROI.height);

  return cudaGetLastError();
}

extern "C" cudaError_t nppiRGBToYUV422_8u_C3C2R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                                       NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize(((oSizeROI.width / 2) + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  rgb_to_yuv422_c2_kernel<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                              oSizeROI.height);

  return cudaGetLastError();
}
