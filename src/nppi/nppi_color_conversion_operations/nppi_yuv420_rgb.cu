#include "npp.h"
#include <cstdint>
#include <cuda_runtime.h>

__device__ inline void yuv420_to_rgb_pixel(uint8_t y, uint8_t u, uint8_t v, uint8_t &r, uint8_t &g, uint8_t &b) {
  // Measured NPP 13.4 JPEG-style full-range YUV matrix (see the num tests).
  // Explicit round-to-nearest intrinsics keep the evaluation stepwise.
  const float Y = static_cast<float>(y);
  const float U = __fsub_rn(static_cast<float>(u), 128.0f);
  const float V = __fsub_rn(static_cast<float>(v), 128.0f);
  const float R = __fadd_rn(Y, __fmul_rn(1.13983f, V));
  const float G = __fsub_rn(__fsub_rn(Y, __fmul_rn(0.39465f, U)), __fmul_rn(0.58060f, V));
  const float B = __fadd_rn(Y, __fmul_rn(2.03211f, U));
  int ri = static_cast<int>(R), gi = static_cast<int>(G), bi = static_cast<int>(B);
  r = static_cast<uint8_t>(ri < 0 ? 0: (ri > 255 ? 255: ri));
  g = static_cast<uint8_t>(gi < 0 ? 0: (gi > 255 ? 255: gi));
  b = static_cast<uint8_t>(bi < 0 ? 0: (bi > 255 ? 255: bi));
}

__device__ inline void yuv420_to_rgb_bt601_pixel(uint8_t y, uint8_t u, uint8_t v, uint8_t &r, uint8_t &g,
                                                 uint8_t &b) {
  yuv420_to_rgb_pixel(y, u, v, r, g, b);
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
  if (alpha_mode == 3) {
    // Measured reference layout of YUV420ToRGB_8u_P3C4R: the constant alpha
    // byte precedes the colour channels (ARGB).  The reference kernel's RGB
    // addressing is broken (luma read at 4x the pixel index); we reproduce
    // the observable layout with the correct RGB values.
    dst[dst_offset] = 0xFF;
    dst[dst_offset + 1] = R;
    dst[dst_offset + 2] = G;
    dst[dst_offset + 3] = B;
    return;
  }
  dst[dst_offset] = R;
  dst[dst_offset + 1] = G;
  dst[dst_offset + 2] = B;
  uint8_t alpha = 0;
  if (alpha_mode == 0) {
    alpha = Y;       // alpha from luma
  } else if (alpha_mode == 1) {
    alpha = 0xFF;    // constant 0xFF
  } else {
    alpha = 0;       // clear alpha
  }
  dst[dst_offset + 3] = alpha;
}

__global__ void yuv420_to_bgr_p3c3_kernel(const uint8_t *__restrict__ srcY, int srcYStep,
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
  yuv420_to_rgb_bt601_pixel(Y, U, V, R, G, B);

  int dst_offset = y * dstStep + x * 3;
  dst[dst_offset] = B;
  dst[dst_offset + 1] = G;
  dst[dst_offset + 2] = R;
}

__global__ void yuv420_to_bgr_p3c4_kernel(const uint8_t *__restrict__ srcY, int srcYStep,
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
  yuv420_to_rgb_bt601_pixel(Y, U, V, R, G, B);

  int dst_offset = y * dstStep + x * 4;
  dst[dst_offset] = B;
  dst[dst_offset + 1] = G;
  dst[dst_offset + 2] = R;
  dst[dst_offset + 3] = 0xFF;
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
                                                                pDst, nDstStep, oSizeROI.width, oSizeROI.height, 3);
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

extern "C" cudaError_t nppiYUV420ToBGR_8u_P3C3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcU,
                                                        int nSrcUStep, const Npp8u *pSrcV, int nSrcVStep, Npp8u *pDst,
                                                        int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  yuv420_to_bgr_p3c3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcY, nSrcYStep, pSrcU, nSrcUStep, pSrcV, nSrcVStep,
                                                                pDst, nDstStep, oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

extern "C" cudaError_t nppiYUV420ToBGR_8u_P3C4R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcU,
                                                        int nSrcUStep, const Npp8u *pSrcV, int nSrcVStep, Npp8u *pDst,
                                                        int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  yuv420_to_bgr_p3c4_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcY, nSrcYStep, pSrcU, nSrcUStep, pSrcV, nSrcVStep,
                                                                pDst, nDstStep, oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}
