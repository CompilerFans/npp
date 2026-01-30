#include "nppdefs.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ inline Npp8u clamp_u8_double(double v) {
  if (v < 0.0) {
    v = 0.0;
  } else if (v > 255.0) {
    v = 255.0;
  }
  return static_cast<Npp8u>(v);
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

__global__ void rgb_to_ycbcr411_y_c3_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, int nDstYStep, int width,
                                            int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *srcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
  int idx = x * 3;
  Npp8u r = srcRow[idx + 0];
  Npp8u g = srcRow[idx + 1];
  Npp8u b = srcRow[idx + 2];

  Npp8u yv, cbv, crv;
  rgb_to_ycbcr_pixel(r, g, b, yv, cbv, crv);

  Npp8u *dstRowY = (Npp8u *)((char *)pDstY + y * nDstYStep);
  dstRowY[x] = yv;
}

__global__ void bgr_to_ycbcr411_y_c3_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, int nDstYStep, int width,
                                            int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *srcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
  int idx = x * 3;
  Npp8u b = srcRow[idx + 0];
  Npp8u g = srcRow[idx + 1];
  Npp8u r = srcRow[idx + 2];

  Npp8u yv, cbv, crv;
  rgb_to_ycbcr_pixel(r, g, b, yv, cbv, crv);

  Npp8u *dstRowY = (Npp8u *)((char *)pDstY + y * nDstYStep);
  dstRowY[x] = yv;
}

__global__ void rgb_to_ycbcr411_cbcr_c3_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstCb, int nDstCbStep,
                                               Npp8u *pDstCr, int nDstCrStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int cbWidth = width >> 2;
  if (x >= cbWidth || y >= height) {
    return;
  }

  int baseX = x << 2;
  int sumCb = 0;
  int sumCr = 0;

  const Npp8u *srcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
  for (int dx = 0; dx < 4; ++dx) {
    int idx = (baseX + dx) * 3;
    Npp8u r = srcRow[idx + 0];
    Npp8u g = srcRow[idx + 1];
    Npp8u b = srcRow[idx + 2];

    Npp8u yv, cbv, crv;
    rgb_to_ycbcr_pixel(r, g, b, yv, cbv, crv);
    sumCb += cbv;
    sumCr += crv;
  }

  Npp8u *dstRowCb = (Npp8u *)((char *)pDstCb + y * nDstCbStep);
  Npp8u *dstRowCr = (Npp8u *)((char *)pDstCr + y * nDstCrStep);
  dstRowCb[x] = static_cast<Npp8u>(sumCb / 4);
  dstRowCr[x] = static_cast<Npp8u>(sumCr / 4);
}

__global__ void bgr_to_ycbcr411_cbcr_c3_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstCb, int nDstCbStep,
                                               Npp8u *pDstCr, int nDstCrStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int cbWidth = width >> 2;
  if (x >= cbWidth || y >= height) {
    return;
  }

  int baseX = x << 2;
  int sumCb = 0;
  int sumCr = 0;

  const Npp8u *srcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
  for (int dx = 0; dx < 4; ++dx) {
    int idx = (baseX + dx) * 3;
    Npp8u b = srcRow[idx + 0];
    Npp8u g = srcRow[idx + 1];
    Npp8u r = srcRow[idx + 2];

    Npp8u yv, cbv, crv;
    rgb_to_ycbcr_pixel(r, g, b, yv, cbv, crv);
    sumCb += cbv;
    sumCr += crv;
  }

  Npp8u *dstRowCb = (Npp8u *)((char *)pDstCb + y * nDstCbStep);
  Npp8u *dstRowCr = (Npp8u *)((char *)pDstCr + y * nDstCrStep);
  dstRowCb[x] = static_cast<Npp8u>(sumCb / 4);
  dstRowCr[x] = static_cast<Npp8u>(sumCr / 4);
}

__global__ void rgb_to_ycbcr411_y_ac4_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, int nDstYStep, int width,
                                             int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *srcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
  int idx = x * 4;
  Npp8u r = srcRow[idx + 0];
  Npp8u g = srcRow[idx + 1];
  Npp8u b = srcRow[idx + 2];

  Npp8u yv, cbv, crv;
  rgb_to_ycbcr_pixel(r, g, b, yv, cbv, crv);

  Npp8u *dstRowY = (Npp8u *)((char *)pDstY + y * nDstYStep);
  dstRowY[x] = yv;
}

__global__ void bgr_to_ycbcr411_y_ac4_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, int nDstYStep, int width,
                                             int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *srcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
  int idx = x * 4;
  Npp8u b = srcRow[idx + 0];
  Npp8u g = srcRow[idx + 1];
  Npp8u r = srcRow[idx + 2];

  Npp8u yv, cbv, crv;
  rgb_to_ycbcr_pixel(r, g, b, yv, cbv, crv);

  Npp8u *dstRowY = (Npp8u *)((char *)pDstY + y * nDstYStep);
  dstRowY[x] = yv;
}

__global__ void rgb_to_ycbcr411_cbcr_ac4_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstCb, int nDstCbStep,
                                                Npp8u *pDstCr, int nDstCrStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int cbWidth = width >> 2;
  if (x >= cbWidth || y >= height) {
    return;
  }

  int baseX = x << 2;
  int sumCb = 0;
  int sumCr = 0;

  const Npp8u *srcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
  for (int dx = 0; dx < 4; ++dx) {
    int idx = (baseX + dx) * 4;
    Npp8u r = srcRow[idx + 0];
    Npp8u g = srcRow[idx + 1];
    Npp8u b = srcRow[idx + 2];

    Npp8u yv, cbv, crv;
    rgb_to_ycbcr_pixel(r, g, b, yv, cbv, crv);
    sumCb += cbv;
    sumCr += crv;
  }

  Npp8u *dstRowCb = (Npp8u *)((char *)pDstCb + y * nDstCbStep);
  Npp8u *dstRowCr = (Npp8u *)((char *)pDstCr + y * nDstCrStep);
  dstRowCb[x] = static_cast<Npp8u>(sumCb / 4);
  dstRowCr[x] = static_cast<Npp8u>(sumCr / 4);
}

__global__ void bgr_to_ycbcr411_cbcr_ac4_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstCb, int nDstCbStep,
                                                Npp8u *pDstCr, int nDstCrStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int cbWidth = width >> 2;
  if (x >= cbWidth || y >= height) {
    return;
  }

  int baseX = x << 2;
  int sumCb = 0;
  int sumCr = 0;

  const Npp8u *srcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
  for (int dx = 0; dx < 4; ++dx) {
    int idx = (baseX + dx) * 4;
    Npp8u b = srcRow[idx + 0];
    Npp8u g = srcRow[idx + 1];
    Npp8u r = srcRow[idx + 2];

    Npp8u yv, cbv, crv;
    rgb_to_ycbcr_pixel(r, g, b, yv, cbv, crv);
    sumCb += cbv;
    sumCr += crv;
  }

  Npp8u *dstRowCb = (Npp8u *)((char *)pDstCb + y * nDstCbStep);
  Npp8u *dstRowCr = (Npp8u *)((char *)pDstCr + y * nDstCrStep);
  dstRowCb[x] = static_cast<Npp8u>(sumCb / 4);
  dstRowCr[x] = static_cast<Npp8u>(sumCr / 4);
}

__global__ void rgb_to_ycbcr411_y_p3_kernel(const Npp8u *pSrcR, const Npp8u *pSrcG, const Npp8u *pSrcB, int nSrcStep,
                                            Npp8u *pDstY, int nDstYStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *rowR = (const Npp8u *)((const char *)pSrcR + y * nSrcStep);
  const Npp8u *rowG = (const Npp8u *)((const char *)pSrcG + y * nSrcStep);
  const Npp8u *rowB = (const Npp8u *)((const char *)pSrcB + y * nSrcStep);

  Npp8u yv, cbv, crv;
  rgb_to_ycbcr_pixel(rowR[x], rowG[x], rowB[x], yv, cbv, crv);

  Npp8u *dstRowY = (Npp8u *)((char *)pDstY + y * nDstYStep);
  dstRowY[x] = yv;
}

__global__ void bgr_to_ycbcr411_y_p3_kernel(const Npp8u *pSrcB, const Npp8u *pSrcG, const Npp8u *pSrcR, int nSrcStep,
                                            Npp8u *pDstY, int nDstYStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *rowB = (const Npp8u *)((const char *)pSrcB + y * nSrcStep);
  const Npp8u *rowG = (const Npp8u *)((const char *)pSrcG + y * nSrcStep);
  const Npp8u *rowR = (const Npp8u *)((const char *)pSrcR + y * nSrcStep);

  Npp8u yv, cbv, crv;
  rgb_to_ycbcr_pixel(rowR[x], rowG[x], rowB[x], yv, cbv, crv);

  Npp8u *dstRowY = (Npp8u *)((char *)pDstY + y * nDstYStep);
  dstRowY[x] = yv;
}

__global__ void rgb_to_ycbcr411_cbcr_p3_kernel(const Npp8u *pSrcR, const Npp8u *pSrcG, const Npp8u *pSrcB,
                                               int nSrcStep, Npp8u *pDstCb, int nDstCbStep, Npp8u *pDstCr,
                                               int nDstCrStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int cbWidth = width >> 2;
  if (x >= cbWidth || y >= height) {
    return;
  }

  int baseX = x << 2;
  int sumCb = 0;
  int sumCr = 0;

  const Npp8u *rowR = (const Npp8u *)((const char *)pSrcR + y * nSrcStep);
  const Npp8u *rowG = (const Npp8u *)((const char *)pSrcG + y * nSrcStep);
  const Npp8u *rowB = (const Npp8u *)((const char *)pSrcB + y * nSrcStep);
  for (int dx = 0; dx < 4; ++dx) {
    Npp8u yv, cbv, crv;
    rgb_to_ycbcr_pixel(rowR[baseX + dx], rowG[baseX + dx], rowB[baseX + dx], yv, cbv, crv);
    sumCb += cbv;
    sumCr += crv;
  }

  Npp8u *dstRowCb = (Npp8u *)((char *)pDstCb + y * nDstCbStep);
  Npp8u *dstRowCr = (Npp8u *)((char *)pDstCr + y * nDstCrStep);
  dstRowCb[x] = static_cast<Npp8u>(sumCb / 4);
  dstRowCr[x] = static_cast<Npp8u>(sumCr / 4);
}

__global__ void bgr_to_ycbcr411_cbcr_p3_kernel(const Npp8u *pSrcB, const Npp8u *pSrcG, const Npp8u *pSrcR,
                                               int nSrcStep, Npp8u *pDstCb, int nDstCbStep, Npp8u *pDstCr,
                                               int nDstCrStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int cbWidth = width >> 2;
  if (x >= cbWidth || y >= height) {
    return;
  }

  int baseX = x << 2;
  int sumCb = 0;
  int sumCr = 0;

  const Npp8u *rowB = (const Npp8u *)((const char *)pSrcB + y * nSrcStep);
  const Npp8u *rowG = (const Npp8u *)((const char *)pSrcG + y * nSrcStep);
  const Npp8u *rowR = (const Npp8u *)((const char *)pSrcR + y * nSrcStep);
  for (int dx = 0; dx < 4; ++dx) {
    Npp8u yv, cbv, crv;
    rgb_to_ycbcr_pixel(rowR[baseX + dx], rowG[baseX + dx], rowB[baseX + dx], yv, cbv, crv);
    sumCb += cbv;
    sumCr += crv;
  }

  Npp8u *dstRowCb = (Npp8u *)((char *)pDstCb + y * nDstCbStep);
  Npp8u *dstRowCr = (Npp8u *)((char *)pDstCr + y * nDstCrStep);
  dstRowCb[x] = static_cast<Npp8u>(sumCb / 4);
  dstRowCr[x] = static_cast<Npp8u>(sumCr / 4);
}

__global__ void ycbcr411_to_rgb_p3c3_kernel(const Npp8u *pSrcY, const Npp8u *pSrcCb, const Npp8u *pSrcCr, int nSrcYStep,
                                            int nSrcCbStep, int nSrcCrStep, Npp8u *pDst, int nDstStep, int width,
                                            int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *rowY = (const Npp8u *)((const char *)pSrcY + y * nSrcYStep);
  const Npp8u *rowCb = (const Npp8u *)((const char *)pSrcCb + y * nSrcCbStep);
  const Npp8u *rowCr = (const Npp8u *)((const char *)pSrcCr + y * nSrcCrStep);

  int cbx = x >> 2;
  Npp8u r, g, b;
  ycbcr_to_rgb_pixel(rowY[x], rowCb[cbx], rowCr[cbx], r, g, b);

  Npp8u *dstRow = (Npp8u *)((char *)pDst + y * nDstStep);
  int idx = x * 3;
  dstRow[idx + 0] = r;
  dstRow[idx + 1] = g;
  dstRow[idx + 2] = b;
}

__global__ void ycbcr411_to_bgr_p3c3_kernel(const Npp8u *pSrcY, const Npp8u *pSrcCb, const Npp8u *pSrcCr, int nSrcYStep,
                                            int nSrcCbStep, int nSrcCrStep, Npp8u *pDst, int nDstStep, int width,
                                            int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *rowY = (const Npp8u *)((const char *)pSrcY + y * nSrcYStep);
  const Npp8u *rowCb = (const Npp8u *)((const char *)pSrcCb + y * nSrcCbStep);
  const Npp8u *rowCr = (const Npp8u *)((const char *)pSrcCr + y * nSrcCrStep);

  int cbx = x >> 2;
  Npp8u r, g, b;
  ycbcr_to_rgb_pixel(rowY[x], rowCb[cbx], rowCr[cbx], r, g, b);

  Npp8u *dstRow = (Npp8u *)((char *)pDst + y * nDstStep);
  int idx = x * 3;
  dstRow[idx + 0] = b;
  dstRow[idx + 1] = g;
  dstRow[idx + 2] = r;
}

__global__ void ycbcr411_to_rgb_p3c4_kernel(const Npp8u *pSrcY, const Npp8u *pSrcCb, const Npp8u *pSrcCr, int nSrcYStep,
                                            int nSrcCbStep, int nSrcCrStep, Npp8u *pDst, int nDstStep, int width,
                                            int height, Npp8u alpha) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *rowY = (const Npp8u *)((const char *)pSrcY + y * nSrcYStep);
  const Npp8u *rowCb = (const Npp8u *)((const char *)pSrcCb + y * nSrcCbStep);
  const Npp8u *rowCr = (const Npp8u *)((const char *)pSrcCr + y * nSrcCrStep);

  int cbx = x >> 2;
  Npp8u r, g, b;
  ycbcr_to_rgb_pixel(rowY[x], rowCb[cbx], rowCr[cbx], r, g, b);

  Npp8u *dstRow = (Npp8u *)((char *)pDst + y * nDstStep);
  int idx = x * 4;
  dstRow[idx + 0] = r;
  dstRow[idx + 1] = g;
  dstRow[idx + 2] = b;
  dstRow[idx + 3] = alpha;
}

__global__ void ycbcr411_to_bgr_p3c4_kernel(const Npp8u *pSrcY, const Npp8u *pSrcCb, const Npp8u *pSrcCr, int nSrcYStep,
                                            int nSrcCbStep, int nSrcCrStep, Npp8u *pDst, int nDstStep, int width,
                                            int height, Npp8u alpha) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *rowY = (const Npp8u *)((const char *)pSrcY + y * nSrcYStep);
  const Npp8u *rowCb = (const Npp8u *)((const char *)pSrcCb + y * nSrcCbStep);
  const Npp8u *rowCr = (const Npp8u *)((const char *)pSrcCr + y * nSrcCrStep);

  int cbx = x >> 2;
  Npp8u r, g, b;
  ycbcr_to_rgb_pixel(rowY[x], rowCb[cbx], rowCr[cbx], r, g, b);

  Npp8u *dstRow = (Npp8u *)((char *)pDst + y * nDstStep);
  int idx = x * 4;
  dstRow[idx + 0] = b;
  dstRow[idx + 1] = g;
  dstRow[idx + 2] = r;
  dstRow[idx + 3] = alpha;
}

__global__ void ycbcr411_to_rgb_p3_kernel(const Npp8u *pSrcY, const Npp8u *pSrcCb, const Npp8u *pSrcCr, int nSrcYStep,
                                          int nSrcCbStep, int nSrcCrStep, Npp8u *pDstR, Npp8u *pDstG, Npp8u *pDstB,
                                          int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *rowY = (const Npp8u *)((const char *)pSrcY + y * nSrcYStep);
  const Npp8u *rowCb = (const Npp8u *)((const char *)pSrcCb + y * nSrcCbStep);
  const Npp8u *rowCr = (const Npp8u *)((const char *)pSrcCr + y * nSrcCrStep);

  int cbx = x >> 2;
  Npp8u r, g, b;
  ycbcr_to_rgb_pixel(rowY[x], rowCb[cbx], rowCr[cbx], r, g, b);

  Npp8u *dstRowR = (Npp8u *)((char *)pDstR + y * nDstStep);
  Npp8u *dstRowG = (Npp8u *)((char *)pDstG + y * nDstStep);
  Npp8u *dstRowB = (Npp8u *)((char *)pDstB + y * nDstStep);
  dstRowR[x] = r;
  dstRowG[x] = g;
  dstRowB[x] = b;
}

__global__ void ycbcr411_to_bgr_p3_kernel(const Npp8u *pSrcY, const Npp8u *pSrcCb, const Npp8u *pSrcCr, int nSrcYStep,
                                          int nSrcCbStep, int nSrcCrStep, Npp8u *pDstB, Npp8u *pDstG, Npp8u *pDstR,
                                          int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *rowY = (const Npp8u *)((const char *)pSrcY + y * nSrcYStep);
  const Npp8u *rowCb = (const Npp8u *)((const char *)pSrcCb + y * nSrcCbStep);
  const Npp8u *rowCr = (const Npp8u *)((const char *)pSrcCr + y * nSrcCrStep);

  int cbx = x >> 2;
  Npp8u r, g, b;
  ycbcr_to_rgb_pixel(rowY[x], rowCb[cbx], rowCr[cbx], r, g, b);

  Npp8u *dstRowB = (Npp8u *)((char *)pDstB + y * nDstStep);
  Npp8u *dstRowG = (Npp8u *)((char *)pDstG + y * nDstStep);
  Npp8u *dstRowR = (Npp8u *)((char *)pDstR + y * nDstStep);
  dstRowB[x] = b;
  dstRowG[x] = g;
  dstRowR[x] = r;
}

extern "C" {

cudaError_t nppiRGBToYCbCr411_8u_C3P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, int nDstYStep,
                                              Npp8u *pDstCb, int nDstCbStep, Npp8u *pDstCr, int nDstCrStep,
                                              NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  rgb_to_ycbcr411_y_c3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDstY, nDstYStep, oSizeROI.width,
                                                                  oSizeROI.height);

  dim3 gridCb((oSizeROI.width / 4 + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);
  rgb_to_ycbcr411_cbcr_c3_kernel<<<gridCb, blockSize, 0, stream>>>(pSrc, nSrcStep, pDstCb, nDstCbStep, pDstCr,
                                                                   nDstCrStep, oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiBGRToYCbCr411_8u_C3P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, int nDstYStep,
                                              Npp8u *pDstCb, int nDstCbStep, Npp8u *pDstCr, int nDstCrStep,
                                              NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  bgr_to_ycbcr411_y_c3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDstY, nDstYStep, oSizeROI.width,
                                                                  oSizeROI.height);

  dim3 gridCb((oSizeROI.width / 4 + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);
  bgr_to_ycbcr411_cbcr_c3_kernel<<<gridCb, blockSize, 0, stream>>>(pSrc, nSrcStep, pDstCb, nDstCbStep, pDstCr,
                                                                   nDstCrStep, oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiRGBToYCbCr411_8u_AC4P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, int nDstYStep,
                                               Npp8u *pDstCb, int nDstCbStep, Npp8u *pDstCr, int nDstCrStep,
                                               NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  rgb_to_ycbcr411_y_ac4_kernel<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDstY, nDstYStep, oSizeROI.width,
                                                                   oSizeROI.height);

  dim3 gridCb((oSizeROI.width / 4 + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);
  rgb_to_ycbcr411_cbcr_ac4_kernel<<<gridCb, blockSize, 0, stream>>>(pSrc, nSrcStep, pDstCb, nDstCbStep, pDstCr,
                                                                    nDstCrStep, oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiBGRToYCbCr411_8u_AC4P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, int nDstYStep,
                                               Npp8u *pDstCb, int nDstCbStep, Npp8u *pDstCr, int nDstCrStep,
                                               NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  bgr_to_ycbcr411_y_ac4_kernel<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDstY, nDstYStep, oSizeROI.width,
                                                                   oSizeROI.height);

  dim3 gridCb((oSizeROI.width / 4 + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);
  bgr_to_ycbcr411_cbcr_ac4_kernel<<<gridCb, blockSize, 0, stream>>>(pSrc, nSrcStep, pDstCb, nDstCbStep, pDstCr,
                                                                    nDstCrStep, oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiRGBToYCbCr411_8u_P3R_kernel(const Npp8u *pSrcR, int nSrcStep, const Npp8u *pSrcG, const Npp8u *pSrcB,
                                            Npp8u *pDstY, int nDstYStep, Npp8u *pDstCb, int nDstCbStep, Npp8u *pDstCr,
                                            int nDstCrStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  rgb_to_ycbcr411_y_p3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcR, pSrcG, pSrcB, nSrcStep, pDstY, nDstYStep,
                                                                  oSizeROI.width, oSizeROI.height);

  dim3 gridCb((oSizeROI.width / 4 + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);
  rgb_to_ycbcr411_cbcr_p3_kernel<<<gridCb, blockSize, 0, stream>>>(pSrcR, pSrcG, pSrcB, nSrcStep, pDstCb, nDstCbStep,
                                                                   pDstCr, nDstCrStep, oSizeROI.width,
                                                                   oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiBGRToYCbCr411_8u_P3R_kernel(const Npp8u *pSrcB, int nSrcStep, const Npp8u *pSrcG, const Npp8u *pSrcR,
                                            Npp8u *pDstY, int nDstYStep, Npp8u *pDstCb, int nDstCbStep, Npp8u *pDstCr,
                                            int nDstCrStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  bgr_to_ycbcr411_y_p3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcB, pSrcG, pSrcR, nSrcStep, pDstY, nDstYStep,
                                                                  oSizeROI.width, oSizeROI.height);

  dim3 gridCb((oSizeROI.width / 4 + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);
  bgr_to_ycbcr411_cbcr_p3_kernel<<<gridCb, blockSize, 0, stream>>>(pSrcB, pSrcG, pSrcR, nSrcStep, pDstCb, nDstCbStep,
                                                                   pDstCr, nDstCrStep, oSizeROI.width,
                                                                   oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiYCbCr411ToRGB_8u_P3C3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCb, int nSrcCbStep,
                                              const Npp8u *pSrcCr, int nSrcCrStep, Npp8u *pDst, int nDstStep,
                                              NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  ycbcr411_to_rgb_p3c3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcY, pSrcCb, pSrcCr, nSrcYStep, nSrcCbStep,
                                                                  nSrcCrStep, pDst, nDstStep, oSizeROI.width,
                                                                  oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiYCbCr411ToBGR_8u_P3C3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCb, int nSrcCbStep,
                                              const Npp8u *pSrcCr, int nSrcCrStep, Npp8u *pDst, int nDstStep,
                                              NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  ycbcr411_to_bgr_p3c3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcY, pSrcCb, pSrcCr, nSrcYStep, nSrcCbStep,
                                                                  nSrcCrStep, pDst, nDstStep, oSizeROI.width,
                                                                  oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiYCbCr411ToRGB_8u_P3C4R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCb, int nSrcCbStep,
                                              const Npp8u *pSrcCr, int nSrcCrStep, Npp8u *pDst, int nDstStep,
                                              NppiSize oSizeROI, Npp8u alpha, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  ycbcr411_to_rgb_p3c4_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcY, pSrcCb, pSrcCr, nSrcYStep, nSrcCbStep,
                                                                  nSrcCrStep, pDst, nDstStep, oSizeROI.width,
                                                                  oSizeROI.height, alpha);
  return cudaGetLastError();
}

cudaError_t nppiYCbCr411ToBGR_8u_P3C4R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCb, int nSrcCbStep,
                                              const Npp8u *pSrcCr, int nSrcCrStep, Npp8u *pDst, int nDstStep,
                                              NppiSize oSizeROI, Npp8u alpha, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  ycbcr411_to_bgr_p3c4_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcY, pSrcCb, pSrcCr, nSrcYStep, nSrcCbStep,
                                                                  nSrcCrStep, pDst, nDstStep, oSizeROI.width,
                                                                  oSizeROI.height, alpha);
  return cudaGetLastError();
}

cudaError_t nppiYCbCr411ToRGB_8u_P3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCb, int nSrcCbStep,
                                            const Npp8u *pSrcCr, int nSrcCrStep, Npp8u *pDstR, Npp8u *pDstG,
                                            Npp8u *pDstB, int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  ycbcr411_to_rgb_p3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcY, pSrcCb, pSrcCr, nSrcYStep, nSrcCbStep,
                                                                nSrcCrStep, pDstR, pDstG, pDstB, nDstStep,
                                                                oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiYCbCr411ToBGR_8u_P3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCb, int nSrcCbStep,
                                            const Npp8u *pSrcCr, int nSrcCrStep, Npp8u *pDstB, Npp8u *pDstG,
                                            Npp8u *pDstR, int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  ycbcr411_to_bgr_p3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcY, pSrcCb, pSrcCr, nSrcYStep, nSrcCbStep,
                                                                nSrcCrStep, pDstB, pDstG, pDstR, nDstStep,
                                                                oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}
}
