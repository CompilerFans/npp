#include "npp.h"
#include <cstdint>
#include <cuda_runtime.h>

__device__ inline void yuv420_to_rgb_pixel(uint8_t y, uint8_t u, uint8_t v, uint8_t &r, uint8_t &g, uint8_t &b) {
  // Fixed-point coefficients (scale = 128) tuned to match NVIDIA NPP YUV420 behavior.
  int yv = static_cast<int>(y);
  int ud = static_cast<int>(u) - 128;
  int vd = static_cast<int>(v) - 128;

  int r_val = yv + ((145 * vd) >> 7);
  int g_val = yv + (((-22 * ud) + (-45 * vd)) >> 7);
  int b_val = yv + ((255 * ud) >> 7);

  r_val = max(0, min(255, r_val));
  g_val = max(0, min(255, g_val));
  b_val = max(0, min(255, b_val));

  r = static_cast<uint8_t>(r_val);
  g = static_cast<uint8_t>(g_val);
  b = static_cast<uint8_t>(b_val);
}

__global__ void yuv420_to_rgb_p3c3_kernel(const uint8_t *__restrict__ srcY, int srcYStep,
                                          const uint8_t *__restrict__ srcU, int srcUStep,
                                          const uint8_t *__restrict__ srcV, int srcVStep,
                                          uint8_t *__restrict__ dst, int dstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  uint8_t Y = srcY[y * srcYStep + x];
  int uv_x = x >> 1;
  int uv_y = y >> 1;
  uint8_t U = srcU[uv_y * srcUStep + uv_x];
  uint8_t V = srcV[uv_y * srcVStep + uv_x];

  uint8_t R, G, B;
  yuv420_to_rgb_pixel(Y, U, V, R, G, B);

  int dst_offset = y * dstStep + x * 3;
  dst[dst_offset] = R;
  dst[dst_offset + 1] = G;
  dst[dst_offset + 2] = B;
}

// Alpha behavior matches NVIDIA NPP:
// - P3C4R: alpha = luma (Y) per pixel
// - P3AC4R: alpha = 0
__global__ void yuv420_to_rgb_p3c4_kernel(const uint8_t *__restrict__ srcY, int srcYStep,
                                          const uint8_t *__restrict__ srcU, int srcUStep,
                                          const uint8_t *__restrict__ srcV, int srcVStep,
                                          uint8_t *__restrict__ dst, int dstStep, int width, int height,
                                          int alpha_mode) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  uint8_t Y = srcY[y * srcYStep + x];
  int uv_x = x >> 1;
  int uv_y = y >> 1;
  uint8_t U = srcU[uv_y * srcUStep + uv_x];
  uint8_t V = srcV[uv_y * srcVStep + uv_x];

  uint8_t R, G, B;
  yuv420_to_rgb_pixel(Y, U, V, R, G, B);

  int dst_offset = y * dstStep + x * 4;
  dst[dst_offset] = R;
  dst[dst_offset + 1] = G;
  dst[dst_offset + 2] = B;
  uint8_t alpha = 0;
  if (alpha_mode == 0) {
    alpha = Y;       // match NVIDIA behavior for C4: alpha from luma
  } else if (alpha_mode == 1) {
    alpha = 0xFF;    // constant 0xFF
  } else {
    alpha = 0;       // clear alpha
  }
  dst[dst_offset + 3] = alpha;
}

__global__ void yuv420_to_rgb_p3p3_kernel(const uint8_t *__restrict__ srcY, int srcYStep,
                                          const uint8_t *__restrict__ srcU, int srcUStep,
                                          const uint8_t *__restrict__ srcV, int srcVStep,
                                          uint8_t *__restrict__ dstR, uint8_t *__restrict__ dstG,
                                          uint8_t *__restrict__ dstB, int dstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  uint8_t Y = srcY[y * srcYStep + x];
  int uv_x = x >> 1;
  int uv_y = y >> 1;
  uint8_t U = srcU[uv_y * srcUStep + uv_x];
  uint8_t V = srcV[uv_y * srcVStep + uv_x];

  uint8_t R, G, B;
  yuv420_to_rgb_pixel(Y, U, V, R, G, B);

  dstR[y * dstStep + x] = R;
  dstG[y * dstStep + x] = G;
  dstB[y * dstStep + x] = B;
}

extern "C" cudaError_t nppiYUV420ToRGB_8u_P3C3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcU,
                                                        int nSrcUStep, const Npp8u *pSrcV, int nSrcVStep, Npp8u *pDst,
                                                        int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  yuv420_to_rgb_p3c3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcY, nSrcYStep, pSrcU, nSrcUStep, pSrcV, nSrcVStep,
                                                                pDst, nDstStep, oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

extern "C" cudaError_t nppiYUV420ToRGB_8u_P3C4R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcU,
                                                        int nSrcUStep, const Npp8u *pSrcV, int nSrcVStep, Npp8u *pDst,
                                                        int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  yuv420_to_rgb_p3c4_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcY, nSrcYStep, pSrcU, nSrcUStep, pSrcV, nSrcVStep,
                                                                pDst, nDstStep, oSizeROI.width, oSizeROI.height, 0);
  return cudaGetLastError();
}

extern "C" cudaError_t nppiYUV420ToRGB_8u_P3AC4R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcU,
                                                         int nSrcUStep, const Npp8u *pSrcV, int nSrcVStep, Npp8u *pDst,
                                                         int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  yuv420_to_rgb_p3c4_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcY, nSrcYStep, pSrcU, nSrcUStep, pSrcV, nSrcVStep,
                                                                pDst, nDstStep, oSizeROI.width, oSizeROI.height, 2);
  return cudaGetLastError();
}

extern "C" cudaError_t nppiYUV420ToRGB_8u_P3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcU,
                                                      int nSrcUStep, const Npp8u *pSrcV, int nSrcVStep, Npp8u *pDstR,
                                                      Npp8u *pDstG, Npp8u *pDstB, int nDstStep, NppiSize oSizeROI,
                                                      cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  yuv420_to_rgb_p3p3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcY, nSrcYStep, pSrcU, nSrcUStep, pSrcV, nSrcVStep,
                                                                pDstR, pDstG, pDstB, nDstStep, oSizeROI.width,
                                                                oSizeROI.height);
  return cudaGetLastError();
}
