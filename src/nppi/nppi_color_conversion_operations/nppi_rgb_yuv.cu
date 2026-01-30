#include "nppdefs.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// RGB到YUV转换的kernelimplementation
// 使用ITU-R BT.601标准转换公式
__device__ inline Npp8u clamp_u8(float v) { return static_cast<Npp8u>(fminf(fmaxf(v, 0.0f), 255.0f)); }

__device__ inline Npp8u clamp_u8_double(double v) {
  if (v < 0.0) {
    v = 0.0;
  } else if (v > 255.0) {
    v = 255.0;
  }
  return static_cast<Npp8u>(v);
}

__device__ inline void rgb_to_yuv_pixel(Npp8u r, Npp8u g, Npp8u b, Npp8u &y, Npp8u &u, Npp8u &v) {
  float R = r;
  float G = g;
  float B = b;
  float Y = 0.299f * R + 0.587f * G + 0.114f * B;
  float U = -0.147f * R - 0.289f * G + 0.436f * B + 128.0f;
  float V = 0.615f * R - 0.515f * G - 0.100f * B + 128.0f;
  y = clamp_u8(Y);
  u = clamp_u8(U);
  v = clamp_u8(V);
}

__device__ inline void rgb_to_ycbcr_pixel(Npp8u r, Npp8u g, Npp8u b, Npp8u &y, Npp8u &cb, Npp8u &cr) {
  double R = r;
  double G = g;
  double B = b;
  double Y = 0.257 * R + 0.504 * G + 0.098 * B + 16.0;
  double Cb = -0.148 * R - 0.291 * G + 0.439 * B + 128.0;
  double Cr = 0.439 * R - 0.368 * G - 0.071 * B + 128.0;

  y = clamp_u8_double(Y);
  cb = clamp_u8_double(Cb);
  cr = clamp_u8_double(Cr);
}

template <bool kBgr>
__global__ void rgb_to_yuv_c3_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                     int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    int idx = x * 3;
    Npp8u r = kBgr ? pSrcRow[idx + 2] : pSrcRow[idx + 0];
    Npp8u g = pSrcRow[idx + 1];
    Npp8u b = kBgr ? pSrcRow[idx + 0] : pSrcRow[idx + 2];

    Npp8u yv, uv, vv;
    rgb_to_yuv_pixel(r, g, b, yv, uv, vv);

    pDstRow[idx + 0] = yv;
    pDstRow[idx + 1] = uv;
    pDstRow[idx + 2] = vv;
  }
}

template <bool kBgr>
__global__ void rgb_to_ycbcr_c3_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                       int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    int idx = x * 3;
    Npp8u r = kBgr ? pSrcRow[idx + 2] : pSrcRow[idx + 0];
    Npp8u g = pSrcRow[idx + 1];
    Npp8u b = kBgr ? pSrcRow[idx + 0] : pSrcRow[idx + 2];

    Npp8u yv, cbv, crv;
    rgb_to_ycbcr_pixel(r, g, b, yv, cbv, crv);

    pDstRow[idx + 0] = yv;
    pDstRow[idx + 1] = cbv;
    pDstRow[idx + 2] = crv;
  }
}

template <bool kBgr>
__global__ void rgb_to_yuv_ac4_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                      int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    int idx = x * 4;
    Npp8u r = kBgr ? pSrcRow[idx + 2] : pSrcRow[idx + 0];
    Npp8u g = pSrcRow[idx + 1];
    Npp8u b = kBgr ? pSrcRow[idx + 0] : pSrcRow[idx + 2];

    Npp8u yv, uv, vv;
    rgb_to_yuv_pixel(r, g, b, yv, uv, vv);

    pDstRow[idx + 0] = yv;
    pDstRow[idx + 1] = uv;
    pDstRow[idx + 2] = vv;
    // Match NVIDIA NPP: packed AC4 output clears alpha.
    pDstRow[idx + 3] = 0;
  }
}

template <bool kBgr>
__global__ void rgb_to_ycbcr_ac4_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                        int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    int idx = x * 4;
    Npp8u r = kBgr ? pSrcRow[idx + 2] : pSrcRow[idx + 0];
    Npp8u g = pSrcRow[idx + 1];
    Npp8u b = kBgr ? pSrcRow[idx + 0] : pSrcRow[idx + 2];

    Npp8u yv, cbv, crv;
    rgb_to_ycbcr_pixel(r, g, b, yv, cbv, crv);

    pDstRow[idx + 0] = yv;
    pDstRow[idx + 1] = cbv;
    pDstRow[idx + 2] = crv;
    // Match NVIDIA NPP: packed AC4 output clears alpha.
    pDstRow[idx + 3] = 0;
  }
}

template <bool kBgr>
__global__ void rgb_to_yuv_c3p3_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, Npp8u *pDstU, Npp8u *pDstV,
                                      int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *pDstRowY = (Npp8u *)((char *)pDstY + y * nDstStep);
    Npp8u *pDstRowU = (Npp8u *)((char *)pDstU + y * nDstStep);
    Npp8u *pDstRowV = (Npp8u *)((char *)pDstV + y * nDstStep);

    int idx = x * 3;
    Npp8u r = kBgr ? pSrcRow[idx + 2] : pSrcRow[idx + 0];
    Npp8u g = pSrcRow[idx + 1];
    Npp8u b = kBgr ? pSrcRow[idx + 0] : pSrcRow[idx + 2];

    Npp8u yv, uv, vv;
    rgb_to_yuv_pixel(r, g, b, yv, uv, vv);

    pDstRowY[x] = yv;
    pDstRowU[x] = uv;
    pDstRowV[x] = vv;
  }
}

template <bool kBgr>
__global__ void rgb_to_ycbcr_c3p3_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, Npp8u *pDstCb,
                                        Npp8u *pDstCr, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *pDstRowY = (Npp8u *)((char *)pDstY + y * nDstStep);
    Npp8u *pDstRowCb = (Npp8u *)((char *)pDstCb + y * nDstStep);
    Npp8u *pDstRowCr = (Npp8u *)((char *)pDstCr + y * nDstStep);

    int idx = x * 3;
    Npp8u r = kBgr ? pSrcRow[idx + 2] : pSrcRow[idx + 0];
    Npp8u g = pSrcRow[idx + 1];
    Npp8u b = kBgr ? pSrcRow[idx + 0] : pSrcRow[idx + 2];

    Npp8u yv, cbv, crv;
    rgb_to_ycbcr_pixel(r, g, b, yv, cbv, crv);

    pDstRowY[x] = yv;
    pDstRowCb[x] = cbv;
    pDstRowCr[x] = crv;
  }
}

__global__ void rgb_to_yuv_p3_kernel(const Npp8u *pSrcR, const Npp8u *pSrcG, const Npp8u *pSrcB, int nSrcStep,
                                     Npp8u *pDstY, Npp8u *pDstU, Npp8u *pDstV, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRowR = (const Npp8u *)((const char *)pSrcR + y * nSrcStep);
    const Npp8u *pSrcRowG = (const Npp8u *)((const char *)pSrcG + y * nSrcStep);
    const Npp8u *pSrcRowB = (const Npp8u *)((const char *)pSrcB + y * nSrcStep);
    Npp8u *pDstRowY = (Npp8u *)((char *)pDstY + y * nDstStep);
    Npp8u *pDstRowU = (Npp8u *)((char *)pDstU + y * nDstStep);
    Npp8u *pDstRowV = (Npp8u *)((char *)pDstV + y * nDstStep);

    Npp8u yv, uv, vv;
    rgb_to_yuv_pixel(pSrcRowR[x], pSrcRowG[x], pSrcRowB[x], yv, uv, vv);

    pDstRowY[x] = yv;
    pDstRowU[x] = uv;
    pDstRowV[x] = vv;
  }
}

__global__ void rgb_to_ycbcr_p3_kernel(const Npp8u *pSrcR, const Npp8u *pSrcG, const Npp8u *pSrcB, int nSrcStep,
                                       Npp8u *pDstY, Npp8u *pDstCb, Npp8u *pDstCr, int nDstStep, int width,
                                       int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRowR = (const Npp8u *)((const char *)pSrcR + y * nSrcStep);
    const Npp8u *pSrcRowG = (const Npp8u *)((const char *)pSrcG + y * nSrcStep);
    const Npp8u *pSrcRowB = (const Npp8u *)((const char *)pSrcB + y * nSrcStep);
    Npp8u *pDstRowY = (Npp8u *)((char *)pDstY + y * nDstStep);
    Npp8u *pDstRowCb = (Npp8u *)((char *)pDstCb + y * nDstStep);
    Npp8u *pDstRowCr = (Npp8u *)((char *)pDstCr + y * nDstStep);

    Npp8u yv, cbv, crv;
    rgb_to_ycbcr_pixel(pSrcRowR[x], pSrcRowG[x], pSrcRowB[x], yv, cbv, crv);

    pDstRowY[x] = yv;
    pDstRowCb[x] = cbv;
    pDstRowCr[x] = crv;
  }
}

template <bool kBgr>
__global__ void rgb_to_yuv_ac4p4_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, Npp8u *pDstU, Npp8u *pDstV,
                                       Npp8u *pDstA, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *pDstRowY = (Npp8u *)((char *)pDstY + y * nDstStep);
    Npp8u *pDstRowU = (Npp8u *)((char *)pDstU + y * nDstStep);
    Npp8u *pDstRowV = (Npp8u *)((char *)pDstV + y * nDstStep);
    Npp8u *pDstRowA = (Npp8u *)((char *)pDstA + y * nDstStep);

    int idx = x * 4;
    Npp8u r = kBgr ? pSrcRow[idx + 2] : pSrcRow[idx + 0];
    Npp8u g = pSrcRow[idx + 1];
    Npp8u b = kBgr ? pSrcRow[idx + 0] : pSrcRow[idx + 2];
    Npp8u a = pSrcRow[idx + 3];

    Npp8u yv, uv, vv;
    rgb_to_yuv_pixel(r, g, b, yv, uv, vv);

    pDstRowY[x] = yv;
    pDstRowU[x] = uv;
    pDstRowV[x] = vv;
    pDstRowA[x] = a;
  }
}

template <bool kBgr>
__global__ void rgb_to_ycbcr_ac4p3_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, Npp8u *pDstCb,
                                         Npp8u *pDstCr, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *pDstRowY = (Npp8u *)((char *)pDstY + y * nDstStep);
    Npp8u *pDstRowCb = (Npp8u *)((char *)pDstCb + y * nDstStep);
    Npp8u *pDstRowCr = (Npp8u *)((char *)pDstCr + y * nDstStep);

    int idx = x * 4;
    Npp8u r = kBgr ? pSrcRow[idx + 2] : pSrcRow[idx + 0];
    Npp8u g = pSrcRow[idx + 1];
    Npp8u b = kBgr ? pSrcRow[idx + 0] : pSrcRow[idx + 2];

    Npp8u yv, cbv, crv;
    rgb_to_ycbcr_pixel(r, g, b, yv, cbv, crv);

    pDstRowY[x] = yv;
    pDstRowCb[x] = cbv;
    pDstRowCr[x] = crv;
  }
}

template <bool kBgr>
__global__ void rgb_to_ycbcr_ac4p4_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, Npp8u *pDstCb,
                                         Npp8u *pDstCr, Npp8u *pDstA, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *pDstRowY = (Npp8u *)((char *)pDstY + y * nDstStep);
    Npp8u *pDstRowCb = (Npp8u *)((char *)pDstCb + y * nDstStep);
    Npp8u *pDstRowCr = (Npp8u *)((char *)pDstCr + y * nDstStep);
    Npp8u *pDstRowA = (Npp8u *)((char *)pDstA + y * nDstStep);

    int idx = x * 4;
    Npp8u r = kBgr ? pSrcRow[idx + 2] : pSrcRow[idx + 0];
    Npp8u g = pSrcRow[idx + 1];
    Npp8u b = kBgr ? pSrcRow[idx + 0] : pSrcRow[idx + 2];
    Npp8u a = pSrcRow[idx + 3];

    Npp8u yv, cbv, crv;
    rgb_to_ycbcr_pixel(r, g, b, yv, cbv, crv);

    pDstRowY[x] = yv;
    pDstRowCb[x] = cbv;
    pDstRowCr[x] = crv;
    pDstRowA[x] = a;
  }
}

// YUV到RGB转换的kernelimplementation
__device__ inline void yuv_to_rgb_pixel(Npp8u y, Npp8u u, Npp8u v, Npp8u &r, Npp8u &g, Npp8u &b) {
  float Y = y;
  float U = static_cast<float>(u) - 128.0f;
  float V = static_cast<float>(v) - 128.0f;

  float R = Y + 1.140f * V;
  float G = Y - 0.395f * U - 0.581f * V;
  float B = Y + 2.032f * U;

  r = clamp_u8(R);
  g = clamp_u8(G);
  b = clamp_u8(B);
}

__device__ inline void ycbcr_to_rgb_pixel(Npp8u y, Npp8u cb, Npp8u cr, Npp8u &r, Npp8u &g, Npp8u &b) {
  double Y = 1.164 * (static_cast<double>(y) - 16.0);
  double Cr = static_cast<double>(cr) - 128.0;
  double Cb = static_cast<double>(cb) - 128.0;

  double R = Y + 1.596 * Cr;
  double G = Y - 0.813 * Cr - 0.392 * Cb;
  double B = Y + 2.017 * Cb;

  r = clamp_u8_double(R);
  g = clamp_u8_double(G);
  b = clamp_u8_double(B);
}

template <bool kBgr>
__global__ void yuv_to_rgb_c3_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                     int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    int idx = x * 3;

    Npp8u r, g, b;
    yuv_to_rgb_pixel(pSrcRow[idx + 0], pSrcRow[idx + 1], pSrcRow[idx + 2], r, g, b);

    pDstRow[idx + 0] = kBgr ? b : r;
    pDstRow[idx + 1] = g;
    pDstRow[idx + 2] = kBgr ? r : b;
  }
}

template <bool kBgr>
__global__ void yuv_to_rgb_ac4_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                      int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    int idx = x * 4;

    Npp8u r, g, b;
    yuv_to_rgb_pixel(pSrcRow[idx + 0], pSrcRow[idx + 1], pSrcRow[idx + 2], r, g, b);

    pDstRow[idx + 0] = kBgr ? b : r;
    pDstRow[idx + 1] = g;
    pDstRow[idx + 2] = kBgr ? r : b;
    // Match NVIDIA NPP: alpha is cleared for AC4 output.
    pDstRow[idx + 3] = 0;
  }
}

template <bool kBgr>
__global__ void yuv_to_rgb_p3c3_kernel(const Npp8u *pSrcY, int nSrcStep, const Npp8u *pSrcU, const Npp8u *pSrcV,
                                       Npp8u *pDst, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRowY = (const Npp8u *)((const char *)pSrcY + y * nSrcStep);
    const Npp8u *pSrcRowU = (const Npp8u *)((const char *)pSrcU + y * nSrcStep);
    const Npp8u *pSrcRowV = (const Npp8u *)((const char *)pSrcV + y * nSrcStep);
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    Npp8u r, g, b;
    yuv_to_rgb_pixel(pSrcRowY[x], pSrcRowU[x], pSrcRowV[x], r, g, b);

    int idx = x * 3;
    pDstRow[idx + 0] = kBgr ? b : r;
    pDstRow[idx + 1] = g;
    pDstRow[idx + 2] = kBgr ? r : b;
  }
}

template <bool kBgr>
__global__ void yuv_to_rgb_p3_kernel(const Npp8u *pSrcY, const Npp8u *pSrcU, const Npp8u *pSrcV, int nSrcStep,
                                     Npp8u *pDst0, Npp8u *pDst1, Npp8u *pDst2, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRowY = (const Npp8u *)((const char *)pSrcY + y * nSrcStep);
    const Npp8u *pSrcRowU = (const Npp8u *)((const char *)pSrcU + y * nSrcStep);
    const Npp8u *pSrcRowV = (const Npp8u *)((const char *)pSrcV + y * nSrcStep);
    Npp8u *pDstRow0 = (Npp8u *)((char *)pDst0 + y * nDstStep);
    Npp8u *pDstRow1 = (Npp8u *)((char *)pDst1 + y * nDstStep);
    Npp8u *pDstRow2 = (Npp8u *)((char *)pDst2 + y * nDstStep);

    Npp8u r, g, b;
    yuv_to_rgb_pixel(pSrcRowY[x], pSrcRowU[x], pSrcRowV[x], r, g, b);

    if (kBgr) {
      pDstRow0[x] = b;
      pDstRow1[x] = g;
      pDstRow2[x] = r;
    } else {
      pDstRow0[x] = r;
      pDstRow1[x] = g;
      pDstRow2[x] = b;
    }
  }
}

template <bool kBgr>
__global__ void ycbcr_to_rgb_c3_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                       int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    int idx = x * 3;
    Npp8u r, g, b;
    ycbcr_to_rgb_pixel(pSrcRow[idx + 0], pSrcRow[idx + 1], pSrcRow[idx + 2], r, g, b);

    pDstRow[idx + 0] = kBgr ? b : r;
    pDstRow[idx + 1] = g;
    pDstRow[idx + 2] = kBgr ? r : b;
  }
}

template <bool kBgr>
__global__ void ycbcr_to_rgb_ac4_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                        int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    int idx = x * 4;
    Npp8u r, g, b;
    ycbcr_to_rgb_pixel(pSrcRow[idx + 0], pSrcRow[idx + 1], pSrcRow[idx + 2], r, g, b);

    pDstRow[idx + 0] = kBgr ? b : r;
    pDstRow[idx + 1] = g;
    pDstRow[idx + 2] = kBgr ? r : b;
    // Match NVIDIA NPP: packed AC4 output clears alpha.
    pDstRow[idx + 3] = 0;
  }
}

template <bool kBgr>
__global__ void ycbcr_to_rgb_p3c3_kernel(const Npp8u *pSrcY, int nSrcStep, const Npp8u *pSrcCb,
                                         const Npp8u *pSrcCr, Npp8u *pDst, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRowY = (const Npp8u *)((const char *)pSrcY + y * nSrcStep);
    const Npp8u *pSrcRowCb = (const Npp8u *)((const char *)pSrcCb + y * nSrcStep);
    const Npp8u *pSrcRowCr = (const Npp8u *)((const char *)pSrcCr + y * nSrcStep);
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    Npp8u r, g, b;
    ycbcr_to_rgb_pixel(pSrcRowY[x], pSrcRowCb[x], pSrcRowCr[x], r, g, b);

    int idx = x * 3;
    pDstRow[idx + 0] = kBgr ? b : r;
    pDstRow[idx + 1] = g;
    pDstRow[idx + 2] = kBgr ? r : b;
  }
}

template <bool kBgr>
__global__ void ycbcr_to_rgb_p3_kernel(const Npp8u *pSrcY, const Npp8u *pSrcCb, const Npp8u *pSrcCr, int nSrcStep,
                                       Npp8u *pDst0, Npp8u *pDst1, Npp8u *pDst2, int nDstStep, int width,
                                       int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRowY = (const Npp8u *)((const char *)pSrcY + y * nSrcStep);
    const Npp8u *pSrcRowCb = (const Npp8u *)((const char *)pSrcCb + y * nSrcStep);
    const Npp8u *pSrcRowCr = (const Npp8u *)((const char *)pSrcCr + y * nSrcStep);
    Npp8u *pDstRow0 = (Npp8u *)((char *)pDst0 + y * nDstStep);
    Npp8u *pDstRow1 = (Npp8u *)((char *)pDst1 + y * nDstStep);
    Npp8u *pDstRow2 = (Npp8u *)((char *)pDst2 + y * nDstStep);

    Npp8u r, g, b;
    ycbcr_to_rgb_pixel(pSrcRowY[x], pSrcRowCb[x], pSrcRowCr[x], r, g, b);

    if (kBgr) {
      pDstRow0[x] = b;
      pDstRow1[x] = g;
      pDstRow2[x] = r;
    } else {
      pDstRow0[x] = r;
      pDstRow1[x] = g;
      pDstRow2[x] = b;
    }
  }
}

template <bool kBgr>
__global__ void ycbcr_to_rgb_p3c4_kernel(const Npp8u *pSrcY, int nSrcStep, const Npp8u *pSrcCb,
                                         const Npp8u *pSrcCr, Npp8u *pDst, int nDstStep, int width, int height,
                                         Npp8u alpha) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRowY = (const Npp8u *)((const char *)pSrcY + y * nSrcStep);
    const Npp8u *pSrcRowCb = (const Npp8u *)((const char *)pSrcCb + y * nSrcStep);
    const Npp8u *pSrcRowCr = (const Npp8u *)((const char *)pSrcCr + y * nSrcStep);
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    Npp8u r, g, b;
    ycbcr_to_rgb_pixel(pSrcRowY[x], pSrcRowCb[x], pSrcRowCr[x], r, g, b);

    int idx = x * 4;
    pDstRow[idx + 0] = kBgr ? b : r;
    pDstRow[idx + 1] = g;
    pDstRow[idx + 2] = kBgr ? r : b;
    pDstRow[idx + 3] = alpha;
  }
}

extern "C" {
// RGB到YUV转换
cudaError_t nppiRGBToYUV_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  rgb_to_yuv_c3_kernel<false><<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                  oSizeROI.height);

  return cudaGetLastError();
}

cudaError_t nppiRGBToYUV_8u_AC4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                        cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  rgb_to_yuv_ac4_kernel<false><<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                   oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiRGBToYUV_8u_C3P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, Npp8u *pDstU, Npp8u *pDstV,
                                         int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  rgb_to_yuv_c3p3_kernel<false><<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDstY, pDstU, pDstV, nDstStep,
                                                                    oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiRGBToYUV_8u_P3R_kernel(const Npp8u *pSrcR, int nSrcStep, const Npp8u *pSrcG, const Npp8u *pSrcB,
                                       Npp8u *pDstY, Npp8u *pDstU, Npp8u *pDstV, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  rgb_to_yuv_p3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcR, pSrcG, pSrcB, nSrcStep, pDstY, pDstU, pDstV,
                                                           nDstStep, oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiRGBToYUV_8u_AC4P4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, Npp8u *pDstU,
                                          Npp8u *pDstV, Npp8u *pDstA, int nDstStep, NppiSize oSizeROI,
                                          cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  rgb_to_yuv_ac4p4_kernel<false><<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDstY, pDstU, pDstV, pDstA,
                                                                     nDstStep, oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiBGRToYUV_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  rgb_to_yuv_c3_kernel<true><<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                 oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiBGRToYUV_8u_AC4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                        cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  rgb_to_yuv_ac4_kernel<true><<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                  oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiBGRToYUV_8u_C3P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, Npp8u *pDstU, Npp8u *pDstV,
                                         int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  rgb_to_yuv_c3p3_kernel<true><<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDstY, pDstU, pDstV, nDstStep,
                                                                   oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiBGRToYUV_8u_P3R_kernel(const Npp8u *pSrcR, int nSrcStep, const Npp8u *pSrcG, const Npp8u *pSrcB,
                                       Npp8u *pDstY, Npp8u *pDstU, Npp8u *pDstV, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  rgb_to_yuv_p3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcR, pSrcG, pSrcB, nSrcStep, pDstY, pDstU, pDstV,
                                                           nDstStep, oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiBGRToYUV_8u_AC4P4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, Npp8u *pDstU,
                                          Npp8u *pDstV, Npp8u *pDstA, int nDstStep, NppiSize oSizeROI,
                                          cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  rgb_to_yuv_ac4p4_kernel<true><<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDstY, pDstU, pDstV, pDstA,
                                                                    nDstStep, oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiRGBToYCbCr_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                         cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  rgb_to_ycbcr_c3_kernel<false><<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                    oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiRGBToYCbCr_8u_AC4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                          NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  rgb_to_ycbcr_ac4_kernel<false><<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                     oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiRGBToYCbCr_8u_C3P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, Npp8u *pDstCb,
                                           Npp8u *pDstCr, int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  rgb_to_ycbcr_c3p3_kernel<false><<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDstY, pDstCb, pDstCr, nDstStep,
                                                                      oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiRGBToYCbCr_8u_P3R_kernel(const Npp8u *pSrcR, int nSrcStep, const Npp8u *pSrcG, const Npp8u *pSrcB,
                                         Npp8u *pDstY, Npp8u *pDstCb, Npp8u *pDstCr, int nDstStep, NppiSize oSizeROI,
                                         cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  rgb_to_ycbcr_p3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcR, pSrcG, pSrcB, nSrcStep, pDstY, pDstCb, pDstCr,
                                                             nDstStep, oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiRGBToYCbCr_8u_AC4P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, Npp8u *pDstCb,
                                            Npp8u *pDstCr, int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  rgb_to_ycbcr_ac4p3_kernel<false><<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDstY, pDstCb, pDstCr, nDstStep,
                                                                       oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiBGRToYCbCr_8u_C3P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, Npp8u *pDstCb,
                                           Npp8u *pDstCr, int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  rgb_to_ycbcr_c3p3_kernel<true><<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDstY, pDstCb, pDstCr, nDstStep,
                                                                     oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiBGRToYCbCr_8u_AC4P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, Npp8u *pDstCb,
                                            Npp8u *pDstCr, int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  rgb_to_ycbcr_ac4p3_kernel<true><<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDstY, pDstCb, pDstCr, nDstStep,
                                                                      oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiBGRToYCbCr_8u_AC4P4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, Npp8u *pDstCb,
                                            Npp8u *pDstCr, Npp8u *pDstA, int nDstStep, NppiSize oSizeROI,
                                            cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  rgb_to_ycbcr_ac4p4_kernel<true><<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDstY, pDstCb, pDstCr, pDstA,
                                                                      nDstStep, oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

// YUV到RGB转换
cudaError_t nppiYUVToRGB_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  yuv_to_rgb_c3_kernel<false><<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                  oSizeROI.height);

  return cudaGetLastError();
}

cudaError_t nppiYUVToRGB_8u_AC4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                        cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  yuv_to_rgb_ac4_kernel<false><<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                   oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiYUVToRGB_8u_P3C3R_kernel(const Npp8u *pSrcY, int nSrcStep, const Npp8u *pSrcU, const Npp8u *pSrcV,
                                         Npp8u *pDst, int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  yuv_to_rgb_p3c3_kernel<false><<<gridSize, blockSize, 0, stream>>>(pSrcY, nSrcStep, pSrcU, pSrcV, pDst, nDstStep,
                                                                    oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiYUVToRGB_8u_P3R_kernel(const Npp8u *pSrcY, int nSrcStep, const Npp8u *pSrcU, const Npp8u *pSrcV,
                                       Npp8u *pDstR, Npp8u *pDstG, Npp8u *pDstB, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  yuv_to_rgb_p3_kernel<false><<<gridSize, blockSize, 0, stream>>>(pSrcY, pSrcU, pSrcV, nSrcStep, pDstR, pDstG, pDstB,
                                                                  nDstStep, oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiYUVToBGR_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  yuv_to_rgb_c3_kernel<true><<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                 oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiYUVToBGR_8u_AC4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                        cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  yuv_to_rgb_ac4_kernel<true><<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                  oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiYUVToBGR_8u_P3C3R_kernel(const Npp8u *pSrcY, int nSrcStep, const Npp8u *pSrcU, const Npp8u *pSrcV,
                                         Npp8u *pDst, int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  yuv_to_rgb_p3c3_kernel<true><<<gridSize, blockSize, 0, stream>>>(pSrcY, nSrcStep, pSrcU, pSrcV, pDst, nDstStep,
                                                                   oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiYUVToBGR_8u_P3R_kernel(const Npp8u *pSrcY, int nSrcStep, const Npp8u *pSrcU, const Npp8u *pSrcV,
                                       Npp8u *pDstB, Npp8u *pDstG, Npp8u *pDstR, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  yuv_to_rgb_p3_kernel<true><<<gridSize, blockSize, 0, stream>>>(pSrcY, pSrcU, pSrcV, nSrcStep, pDstB, pDstG, pDstR,
                                                                 nDstStep, oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiYCbCrToRGB_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  ycbcr_to_rgb_c3_kernel<false><<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                    oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiYCbCrToRGB_8u_AC4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                          NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  ycbcr_to_rgb_ac4_kernel<false><<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                     oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiYCbCrToRGB_8u_P3C3R_kernel(const Npp8u *pSrcY, int nSrcStep, const Npp8u *pSrcCb,
                                           const Npp8u *pSrcCr, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                           cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  ycbcr_to_rgb_p3c3_kernel<false><<<gridSize, blockSize, 0, stream>>>(pSrcY, nSrcStep, pSrcCb, pSrcCr, pDst, nDstStep,
                                                                      oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiYCbCrToRGB_8u_P3R_kernel(const Npp8u *pSrcY, int nSrcStep, const Npp8u *pSrcCb,
                                         const Npp8u *pSrcCr, Npp8u *pDstR, Npp8u *pDstG, Npp8u *pDstB,
                                         int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  ycbcr_to_rgb_p3_kernel<false><<<gridSize, blockSize, 0, stream>>>(pSrcY, pSrcCb, pSrcCr, nSrcStep, pDstR, pDstG,
                                                                    pDstB, nDstStep, oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiYCbCrToRGB_8u_P3C4R_kernel(const Npp8u *pSrcY, int nSrcStep, const Npp8u *pSrcCb,
                                           const Npp8u *pSrcCr, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                           Npp8u alpha, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  ycbcr_to_rgb_p3c4_kernel<false><<<gridSize, blockSize, 0, stream>>>(pSrcY, nSrcStep, pSrcCb, pSrcCr, pDst, nDstStep,
                                                                      oSizeROI.width, oSizeROI.height, alpha);
  return cudaGetLastError();
}

cudaError_t nppiYCbCrToBGR_8u_P3C3R_kernel(const Npp8u *pSrcY, int nSrcStep, const Npp8u *pSrcCb,
                                           const Npp8u *pSrcCr, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                           cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  ycbcr_to_rgb_p3c3_kernel<true><<<gridSize, blockSize, 0, stream>>>(pSrcY, nSrcStep, pSrcCb, pSrcCr, pDst, nDstStep,
                                                                     oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiYCbCrToBGR_8u_P3C4R_kernel(const Npp8u *pSrcY, int nSrcStep, const Npp8u *pSrcCb,
                                           const Npp8u *pSrcCr, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                           Npp8u alpha, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  ycbcr_to_rgb_p3c4_kernel<true><<<gridSize, blockSize, 0, stream>>>(pSrcY, nSrcStep, pSrcCb, pSrcCr, pDst, nDstStep,
                                                                     oSizeROI.width, oSizeROI.height, alpha);
  return cudaGetLastError();
}
}
