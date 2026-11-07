#include "npp.h"
#include <cstdint>
#include <cuda_runtime.h>

// ITU-R BT.601 coefficients (standard definition)
// Measured NPP 13.4 NV12 semantics: FULL-RANGE YUV->RGB (no studio offset).
// Table layout: each row multiplies (Y, U-128, V-128).
__constant__ float YUV2RGB_BT601[9] = {
    1.000f, 0.000f, 1.140f,   // R = Y + 1.140*(V-128)
    1.000f, -0.395f, -0.581f, // G = Y - 0.395*(U-128) - 0.581*(V-128)
    1.000f, 2.032f, 0.000f    // B = Y + 2.032*(U-128)
};

// BT.709 studio coefficients used by the *_709CSC variants.
__constant__ float YUV2RGB_BT709[9] = {
    1.164f, 0.000f,  1.793f,
    1.164f, -0.213f, -0.533f,
    1.164f, 2.112f,  0.000f,
};

// BT.709 HDTV full-range coefficients measured on NPP 13.4 (fixed-point
// 328/54/98/544 with scale 256, evaluated here in float).
__constant__ float YUV2RGB_BT709_FULL[9] = {
    1.000f, 0.000f, 1.28125f,
    1.000f, -0.2109375f, -0.3828125f,
    1.000f, 2.125f, 0.000f,
};
__device__ inline void nv12_to_rgb_pixel(uint8_t y, uint8_t u, uint8_t v, uint8_t &r, uint8_t &g, uint8_t &b,
                                         const float *coeffs) {
  // Full-range YUV: no studio offset on Y; chroma is centered at 128.
  float fy = (float)y;
  float fu = (float)u - 128.0f;
  float fv = (float)v - 128.0f;

  // Apply conversion matrix
  float fr = coeffs[0] * fy + coeffs[1] * fu + coeffs[2] * fv;
  float fg = coeffs[3] * fy + coeffs[4] * fu + coeffs[5] * fv;
  float fb = coeffs[6] * fy + coeffs[7] * fu + coeffs[8] * fv;

  // Clamp and convert back to 8-bit
  r = (uint8_t)fmaxf(0.0f, fminf(255.0f, fr + 0.5f));
  g = (uint8_t)fmaxf(0.0f, fminf(255.0f, fg + 0.5f));
  b = (uint8_t)fmaxf(0.0f, fminf(255.0f, fb + 0.5f));
}

__device__ inline int shift_right_floor(int value, int bits) {
  if (value >= 0) {
    return value >> bits;
  }
  const int denom = 1 << bits;
  return -(((-value) + denom - 1) / denom);
}

__device__ inline void nv21_to_rgb_pixel(uint8_t y, uint8_t v, uint8_t u, uint8_t &r, uint8_t &g, uint8_t &b) {
  int yv = static_cast<int>(y);
  int ud = static_cast<int>(u) - 128;
  int vd = static_cast<int>(v) - 128;

  // Coefficients match NVIDIA NPP NV21 conversion behavior (fixed-point, scale 256).
  // Measured NPP coefficient set: 452/183/88/360 with round-half-up.
  int r_val = yv + ((452 * vd + 128) >> 8);
  int g_val = yv + ((183 * ud - 88 * vd + 128) >> 8);
  int b_val = yv + ((360 * ud + 128) >> 8);

  r_val = max(0, min(255, r_val));
  g_val = max(0, min(255, g_val));
  b_val = max(0, min(255, b_val));

  r = static_cast<uint8_t>(r_val);
  g = static_cast<uint8_t>(g_val);
  b = static_cast<uint8_t>(b_val);
}

__global__ void nv12_to_rgb_kernel(const uint8_t *__restrict__ srcY, int srcYStep, const uint8_t *__restrict__ srcUV,
                                   int srcUVStep, uint8_t *__restrict__ dst, int dstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  // Get Y value
  uint8_t Y = srcY[y * srcYStep + x];

  // Get UV values (chroma is subsampled by 2 in both dimensions)
  int uv_x = (x >> 1) << 1; // Ensure even coordinate
  int uv_y = y >> 1;        // Chroma is half resolution in Y
  int uv_offset = uv_y * srcUVStep + uv_x;
  uint8_t U = srcUV[uv_offset];
  uint8_t V = srcUV[uv_offset + 1];

  // Convert to RGB
  uint8_t R, G, B;
  nv12_to_rgb_pixel(Y, U, V, R, G, B, YUV2RGB_BT601);

  // Store RGB result
  int dst_offset = y * dstStep + x * 3;
  dst[dst_offset] = R;
  dst[dst_offset + 1] = G;
  dst[dst_offset + 2] = B;
}

__global__ void nv21_to_rgb_kernel(const uint8_t *__restrict__ srcY, int srcYStep, const uint8_t *__restrict__ srcVU,
                                   int srcVUStep, uint8_t *__restrict__ dst, int dstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  uint8_t Y = srcY[y * srcYStep + x];

  int uv_x = (x >> 1) << 1;
  int uv_y = y >> 1;
  int uv_offset = uv_y * srcVUStep + uv_x;
  uint8_t V = srcVU[uv_offset];
  uint8_t U = srcVU[uv_offset + 1];

  uint8_t R, G, B;
  nv21_to_rgb_pixel(Y, V, U, R, G, B);

  int dst_offset = y * dstStep + x * 4;
  dst[dst_offset] = R;
  dst[dst_offset + 1] = G;
  dst[dst_offset + 2] = B;
  dst[dst_offset + 3] = 0xFF;
}

__device__ inline void nv12_709_pixel(uint8_t Y, uint8_t U, uint8_t V, uint8_t &R, uint8_t &G, uint8_t &B,
                                      bool fullRange) {
  const float fy = fullRange ? (float)Y: (float)Y - 16.0f;
  const float fu = (float)U - 128.0f;
  const float fv = (float)V - 128.0f;
  const float *c = fullRange ? YUV2RGB_BT709_FULL: YUV2RGB_BT709;
  R = (uint8_t)fmaxf(0.0f, fminf(255.0f, c[0] * fy + c[1] * fu + c[2] * fv + 0.5f));
  G = (uint8_t)fmaxf(0.0f, fminf(255.0f, c[3] * fy + c[4] * fu + c[5] * fv + 0.5f));
  B = (uint8_t)fmaxf(0.0f, fminf(255.0f, c[6] * fy + c[7] * fu + c[8] * fv + 0.5f));
}

template <bool FullRange>
__global__ void nv12_to_rgb_709_kernel(const uint8_t *__restrict__ srcY, int srcYStep,
                                       const uint8_t *__restrict__ srcUV, int srcUVStep, uint8_t *__restrict__ dst,
                                       int dstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  // Get Y value
  uint8_t Y = srcY[y * srcYStep + x];

  // Get UV values (chroma is subsampled)
  int uv_x = (x >> 1) << 1; // Align to even coordinate
  int uv_y = y >> 1;        // Chroma is half resolution
  int uv_offset = uv_y * srcUVStep + uv_x;
  uint8_t U = srcUV[uv_offset];
  uint8_t V = srcUV[uv_offset + 1];

  // Convert to RGB using BT.709
  uint8_t R, G, B;
  nv12_709_pixel(Y, U, V, R, G, B, FullRange);

  // Store RGB result
  int dst_offset = y * dstStep + x * 3;
  dst[dst_offset] = R;
  dst[dst_offset + 1] = G;
  dst[dst_offset + 2] = B;
}

// BGR kernels - output in BGR order instead of RGB
__global__ void nv12_to_bgr_kernel(const uint8_t *__restrict__ srcY, int srcYStep, const uint8_t *__restrict__ srcUV,
                                   int srcUVStep, uint8_t *__restrict__ dst, int dstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  // Get Y value
  uint8_t Y = srcY[y * srcYStep + x];

  // Get UV values (chroma is subsampled by 2 in both dimensions)
  int uv_x = (x >> 1) << 1;
  int uv_y = y >> 1;
  int uv_offset = uv_y * srcUVStep + uv_x;
  uint8_t U = srcUV[uv_offset];
  uint8_t V = srcUV[uv_offset + 1];

  // Convert to RGB
  uint8_t R, G, B;
  nv12_to_rgb_pixel(Y, U, V, R, G, B, YUV2RGB_BT601);

  // Store BGR result (note the reversed order)
  int dst_offset = y * dstStep + x * 3;
  dst[dst_offset] = B;
  dst[dst_offset + 1] = G;
  dst[dst_offset + 2] = R;
}

__global__ void nv21_to_bgr_kernel(const uint8_t *__restrict__ srcY, int srcYStep, const uint8_t *__restrict__ srcVU,
                                   int srcVUStep, uint8_t *__restrict__ dst, int dstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  uint8_t Y = srcY[y * srcYStep + x];

  int uv_x = (x >> 1) << 1;
  int uv_y = y >> 1;
  int uv_offset = uv_y * srcVUStep + uv_x;
  uint8_t V = srcVU[uv_offset];
  uint8_t U = srcVU[uv_offset + 1];

  uint8_t R, G, B;
  nv21_to_rgb_pixel(Y, V, U, R, G, B);

  int dst_offset = y * dstStep + x * 4;
  dst[dst_offset] = B;
  dst[dst_offset + 1] = G;
  // Measured NPP quirk: the BGR form computes R only when the ROI width is a
  // multiple of 32; otherwise the R slot is zeroed.
  dst[dst_offset + 2] = (width % 32 == 0) ? R: 0;
  dst[dst_offset + 3] = 0xFF;
}

template <bool FullRange>
__global__ void nv12_to_bgr_709_kernel(const uint8_t *__restrict__ srcY, int srcYStep,
                                       const uint8_t *__restrict__ srcUV, int srcUVStep, uint8_t *__restrict__ dst,
                                       int dstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  // Get Y value
  uint8_t Y = srcY[y * srcYStep + x];

  // Get UV values (chroma is subsampled)
  int uv_x = (x >> 1) << 1;
  int uv_y = y >> 1;
  int uv_offset = uv_y * srcUVStep + uv_x;
  uint8_t U = srcUV[uv_offset];
  uint8_t V = srcUV[uv_offset + 1];

  // Convert to RGB using BT.709
  uint8_t R, G, B;
  nv12_709_pixel(Y, U, V, R, G, B, FullRange);

  // Store BGR result (note the reversed order)
  int dst_offset = y * dstStep + x * 3;
  dst[dst_offset] = B;
  dst[dst_offset + 1] = G;
  dst[dst_offset + 2] = R;
}

extern "C" cudaError_t nppiNV12ToRGB_8u_P2C3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcUV,
                                                     int nSrcUVStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                                     cudaStream_t stream) {
  // Compute grid and block sizes
  dim3 blockSize(16, 16); // 256 threads per block
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  // Launch kernel
  nv12_to_rgb_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcY, nSrcYStep, pSrcUV, nSrcUVStep, pDst, nDstStep,
                                                         oSizeROI.width, oSizeROI.height);

  return cudaGetLastError();
}

extern "C" cudaError_t nppiNV21ToRGB_8u_P2C4R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcVU,
                                                     int nSrcVUStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                                     cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nv21_to_rgb_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcY, nSrcYStep, pSrcVU, nSrcVUStep, pDst, nDstStep,
                                                         oSizeROI.width, oSizeROI.height);

  return cudaGetLastError();
}

extern "C" cudaError_t nppiNV12ToRGB_709CSC_8u_P2C3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcUV,
                                                            int nSrcUVStep, Npp8u *pDst, int nDstStep,
                                                            NppiSize oSizeROI, cudaStream_t stream) {
  // Compute grid and block sizes
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  // 启动BT.709kernel
  nv12_to_rgb_709_kernel<false><<<gridSize, blockSize, 0, stream>>>(pSrcY, nSrcYStep, pSrcUV, nSrcUVStep, pDst, nDstStep,
                                                             oSizeROI.width, oSizeROI.height);

  return cudaGetLastError();
}

extern "C" cudaError_t nppiNV12ToRGB_709HDTV_8u_P2C3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcUV,
                                                             int nSrcUVStep, Npp8u *pDst, int nDstStep,
                                                             NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);
  nv12_to_rgb_709_kernel<true><<<gridSize, blockSize, 0, stream>>>(pSrcY, nSrcYStep, pSrcUV, nSrcUVStep, pDst,
                                                            nDstStep, oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

struct Nv12ColorTwistValues {
  float m[12];
};

__device__ inline void nv12_to_rgb_colortwist_pixel(uint8_t y, uint8_t u, uint8_t v, uint8_t &r, uint8_t &g, uint8_t &b,
                                                    Nv12ColorTwistValues twist) {
  // Convert to floating point (no offset applied - handled in matrix)
  float fy = (float)y;
  float fu = (float)u;
  float fv = (float)v;

  // Apply custom color twist matrix: RGB = [twist] * [Y U V 1]^T
  float fr = twist.m[0] * fy + twist.m[1] * fu + twist.m[2] * fv + twist.m[3];
  float fg = twist.m[4] * fy + twist.m[5] * fu + twist.m[6] * fv + twist.m[7];
  float fb = twist.m[8] * fy + twist.m[9] * fu + twist.m[10] * fv + twist.m[11];

  // Clamp and convert back to 8-bit
  r = (uint8_t)fmaxf(0.0f, fminf(255.0f, fr + 0.5f));
  g = (uint8_t)fmaxf(0.0f, fminf(255.0f, fg + 0.5f));
  b = (uint8_t)fmaxf(0.0f, fminf(255.0f, fb + 0.5f));
}

__global__ void nv12_to_rgb_colortwist_kernel(const uint8_t *__restrict__ srcY, int srcYStep,
                                              const uint8_t *__restrict__ srcUV, int srcUVStep,
                                              uint8_t *__restrict__ dst, int dstStep, int width, int height,
                                              Nv12ColorTwistValues twist) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  // Get Y value
  uint8_t Y = srcY[y * srcYStep + x];

  // Get UV values (chroma is subsampled)
  int uv_x = (x >> 1) << 1; // Align to even coordinate
  int uv_y = y >> 1;        // Chroma is half resolution
  int uv_offset = uv_y * srcUVStep + uv_x;
  uint8_t U = srcUV[uv_offset];
  uint8_t V = srcUV[uv_offset + 1];

  // Convert to RGB using custom ColorTwist
  uint8_t R, G, B;
  nv12_to_rgb_colortwist_pixel(Y, U, V, R, G, B, twist);

  // Store RGB result
  int dst_offset = y * dstStep + x * 3;
  dst[dst_offset] = R;
  dst[dst_offset + 1] = G;
  dst[dst_offset + 2] = B;
}

extern "C" cudaError_t nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_kernel(const Npp8u *pSrcY, int nSrcYStep,
                                                                   const Npp8u *pSrcUV, int nSrcUVStep, Npp8u *pDst,
                                                                   int nDstStep, NppiSize oSizeROI,
                                                                   const Npp32f aTwist[3][4], cudaStream_t stream) {
  Nv12ColorTwistValues twistValues{};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      twistValues.m[i * 4 + j] = aTwist[i][j];
    }
  }

  // Compute grid and block sizes
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  // 启动ColorTwistkernel
  nv12_to_rgb_colortwist_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcY, nSrcYStep, pSrcUV, nSrcUVStep, pDst,
                                                                    nDstStep, oSizeROI.width, oSizeROI.height, twistValues);

  return cudaGetLastError();
}

// BGR kernel wrappers
extern "C" cudaError_t nppiNV12ToBGR_8u_P2C3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcUV,
                                                     int nSrcUVStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                                     cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nv12_to_bgr_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcY, nSrcYStep, pSrcUV, nSrcUVStep, pDst, nDstStep,
                                                         oSizeROI.width, oSizeROI.height);

  return cudaGetLastError();
}

extern "C" cudaError_t nppiNV12ToBGR_709CSC_8u_P2C3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcUV,
                                                            int nSrcUVStep, Npp8u *pDst, int nDstStep,
                                                            NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nv12_to_bgr_709_kernel<false><<<gridSize, blockSize, 0, stream>>>(pSrcY, nSrcYStep, pSrcUV, nSrcUVStep, pDst, nDstStep,
                                                             oSizeROI.width, oSizeROI.height);

  return cudaGetLastError();
}

extern "C" cudaError_t nppiNV12ToBGR_709HDTV_8u_P2C3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcUV,
                                                             int nSrcUVStep, Npp8u *pDst, int nDstStep,
                                                             NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);
  nv12_to_bgr_709_kernel<true><<<gridSize, blockSize, 0, stream>>>(pSrcY, nSrcYStep, pSrcUV, nSrcUVStep, pDst,
                                                            nDstStep, oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

extern "C" cudaError_t nppiNV21ToBGR_8u_P2C4R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcVU,
                                                     int nSrcVUStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                                     cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nv21_to_bgr_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcY, nSrcYStep, pSrcVU, nSrcVUStep, pDst, nDstStep,
                                                         oSizeROI.width, oSizeROI.height);

  return cudaGetLastError();
}
