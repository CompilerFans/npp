#include "nppdefs.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

__device__ inline Npp8u clamp_u8_round(float v) {
  if (v < 0.0f) {
    return 0;
  }
  if (v > 255.0f) {
    return 255;
  }
  return static_cast<Npp8u>(lrintf(v));
}

__device__ inline void rgb_to_hls_pixel(Npp8u r, Npp8u g, Npp8u b, Npp8u &h, Npp8u &l, Npp8u &s) {
  float rf = r / 255.0f;
  float gf = g / 255.0f;
  float bf = b / 255.0f;

  float maxv = fmaxf(rf, fmaxf(gf, bf));
  float minv = fminf(rf, fminf(gf, bf));
  float lval = 0.5f * (maxv + minv);

  float hval = 0.0f;
  float sval = 0.0f;
  float delta = maxv - minv;

  if (delta > 0.0f) {
    if (lval < 0.5f) {
      sval = delta / (maxv + minv);
    } else {
      sval = delta / (2.0f - maxv - minv);
    }

    if (maxv == rf) {
      hval = (gf - bf) / delta;
      if (gf < bf) {
        hval += 6.0f;
      }
    } else if (maxv == gf) {
      hval = 2.0f + (bf - rf) / delta;
    } else {
      hval = 4.0f + (rf - gf) / delta;
    }
    hval /= 6.0f;
  }

  h = clamp_u8_round(hval * 255.0f);
  l = clamp_u8_round(lval * 255.0f);
  s = clamp_u8_round(sval * 255.0f);
}

__device__ inline float hue_to_rgb(float p, float q, float t) {
  if (t < 0.0f) {
    t += 1.0f;
  }
  if (t > 1.0f) {
    t -= 1.0f;
  }
  if (t < 1.0f / 6.0f) {
    return p + (q - p) * 6.0f * t;
  }
  if (t < 0.5f) {
    return q;
  }
  if (t < 2.0f / 3.0f) {
    return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
  }
  return p;
}

__device__ inline void hls_to_rgb_pixel(Npp8u h, Npp8u l, Npp8u s, Npp8u &r, Npp8u &g, Npp8u &b) {
  float hf = h / 255.0f;
  float lf = l / 255.0f;
  float sf = s / 255.0f;

  float rf, gf, bf;
  if (sf <= 0.0f) {
    rf = gf = bf = lf;
  } else {
    float q = (lf < 0.5f) ? (lf * (1.0f + sf)) : (lf + sf - lf * sf);
    float p = 2.0f * lf - q;
    rf = hue_to_rgb(p, q, hf + 1.0f / 3.0f);
    gf = hue_to_rgb(p, q, hf);
    bf = hue_to_rgb(p, q, hf - 1.0f / 3.0f);
  }

  r = clamp_u8_round(rf * 255.0f);
  g = clamp_u8_round(gf * 255.0f);
  b = clamp_u8_round(bf * 255.0f);
}

__global__ void rgb_to_hls_c3_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                     int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    int idx = x * 3;
    Npp8u r = pSrcRow[idx + 0];
    Npp8u g = pSrcRow[idx + 1];
    Npp8u b = pSrcRow[idx + 2];

    Npp8u h, l, s;
    rgb_to_hls_pixel(r, g, b, h, l, s);

    pDstRow[idx + 0] = h;
    pDstRow[idx + 1] = l;
    pDstRow[idx + 2] = s;
  }
}

__global__ void rgb_to_hls_ac4_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                      int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    int idx = x * 4;
    Npp8u r = pSrcRow[idx + 0];
    Npp8u g = pSrcRow[idx + 1];
    Npp8u b = pSrcRow[idx + 2];

    Npp8u h, l, s;
    rgb_to_hls_pixel(r, g, b, h, l, s);

    pDstRow[idx + 0] = h;
    pDstRow[idx + 1] = l;
    pDstRow[idx + 2] = s;
    // Match NVIDIA NPP: clear alpha.
    pDstRow[idx + 3] = 0;
  }
}

__global__ void hls_to_rgb_c3_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                     int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    int idx = x * 3;
    Npp8u h = pSrcRow[idx + 0];
    Npp8u l = pSrcRow[idx + 1];
    Npp8u s = pSrcRow[idx + 2];

    Npp8u r, g, b;
    hls_to_rgb_pixel(h, l, s, r, g, b);

    pDstRow[idx + 0] = r;
    pDstRow[idx + 1] = g;
    pDstRow[idx + 2] = b;
  }
}

__global__ void hls_to_rgb_ac4_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                      int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    int idx = x * 4;
    Npp8u h = pSrcRow[idx + 0];
    Npp8u l = pSrcRow[idx + 1];
    Npp8u s = pSrcRow[idx + 2];

    Npp8u r, g, b;
    hls_to_rgb_pixel(h, l, s, r, g, b);

    pDstRow[idx + 0] = r;
    pDstRow[idx + 1] = g;
    pDstRow[idx + 2] = b;
    // Match NVIDIA NPP: clear alpha.
    pDstRow[idx + 3] = 0;
  }
}



__global__ void bgr_to_hls_c3p3_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstH, Npp8u *pDstL, Npp8u *pDstS,
                                      int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *pDstRowH = (Npp8u *)((char *)pDstH + y * nDstStep);
    Npp8u *pDstRowL = (Npp8u *)((char *)pDstL + y * nDstStep);
    Npp8u *pDstRowS = (Npp8u *)((char *)pDstS + y * nDstStep);

    int idx = x * 3;
    Npp8u b = pSrcRow[idx + 0];
    Npp8u g = pSrcRow[idx + 1];
    Npp8u r = pSrcRow[idx + 2];

    Npp8u h, l, s;
    rgb_to_hls_pixel(r, g, b, h, l, s);

    pDstRowH[x] = h;
    pDstRowL[x] = l;
    pDstRowS[x] = s;
  }
}

__global__ void bgr_to_hls_p3c3_kernel(const Npp8u *pSrcB, const Npp8u *pSrcG, const Npp8u *pSrcR, int nSrcStep,
                                      Npp8u *pDst, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pRowB = (const Npp8u *)((const char *)pSrcB + y * nSrcStep);
    const Npp8u *pRowG = (const Npp8u *)((const char *)pSrcG + y * nSrcStep);
    const Npp8u *pRowR = (const Npp8u *)((const char *)pSrcR + y * nSrcStep);
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    Npp8u h, l, s;
    rgb_to_hls_pixel(pRowR[x], pRowG[x], pRowB[x], h, l, s);

    int idx = x * 3;
    pDstRow[idx + 0] = h;
    pDstRow[idx + 1] = l;
    pDstRow[idx + 2] = s;
  }
}

__global__ void bgr_to_hls_p3_kernel(const Npp8u *pSrcB, const Npp8u *pSrcG, const Npp8u *pSrcR, int nSrcStep,
                                    Npp8u *pDstH, Npp8u *pDstL, Npp8u *pDstS, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pRowB = (const Npp8u *)((const char *)pSrcB + y * nSrcStep);
    const Npp8u *pRowG = (const Npp8u *)((const char *)pSrcG + y * nSrcStep);
    const Npp8u *pRowR = (const Npp8u *)((const char *)pSrcR + y * nSrcStep);
    Npp8u *pDstRowH = (Npp8u *)((char *)pDstH + y * nDstStep);
    Npp8u *pDstRowL = (Npp8u *)((char *)pDstL + y * nDstStep);
    Npp8u *pDstRowS = (Npp8u *)((char *)pDstS + y * nDstStep);

    Npp8u h, l, s;
    rgb_to_hls_pixel(pRowR[x], pRowG[x], pRowB[x], h, l, s);

    pDstRowH[x] = h;
    pDstRowL[x] = l;
    pDstRowS[x] = s;
  }
}

__global__ void bgr_to_hls_ac4_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                      int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    int idx = x * 4;
    Npp8u b = pSrcRow[idx + 0];
    Npp8u g = pSrcRow[idx + 1];
    Npp8u r = pSrcRow[idx + 2];

    Npp8u h, l, s;
    rgb_to_hls_pixel(r, g, b, h, l, s);

    pDstRow[idx + 0] = h;
    pDstRow[idx + 1] = l;
    pDstRow[idx + 2] = s;
    pDstRow[idx + 3] = 0;
  }
}

__global__ void bgr_to_hls_ac4p4_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstH, Npp8u *pDstL,
                                       Npp8u *pDstS, Npp8u *pDstA, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *pDstRowH = (Npp8u *)((char *)pDstH + y * nDstStep);
    Npp8u *pDstRowL = (Npp8u *)((char *)pDstL + y * nDstStep);
    Npp8u *pDstRowS = (Npp8u *)((char *)pDstS + y * nDstStep);
    Npp8u *pDstRowA = (Npp8u *)((char *)pDstA + y * nDstStep);

    int idx = x * 4;
    Npp8u b = pSrcRow[idx + 0];
    Npp8u g = pSrcRow[idx + 1];
    Npp8u r = pSrcRow[idx + 2];

    Npp8u h, l, s;
    rgb_to_hls_pixel(r, g, b, h, l, s);

    pDstRowH[x] = h;
    pDstRowL[x] = l;
    pDstRowS[x] = s;
    pDstRowA[x] = 0;
  }
}

__global__ void bgr_to_hls_ap4c4_kernel(const Npp8u *pSrcB, const Npp8u *pSrcG, const Npp8u *pSrcR,
                                       const Npp8u *pSrcA, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                       int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pRowB = (const Npp8u *)((const char *)pSrcB + y * nSrcStep);
    const Npp8u *pRowG = (const Npp8u *)((const char *)pSrcG + y * nSrcStep);
    const Npp8u *pRowR = (const Npp8u *)((const char *)pSrcR + y * nSrcStep);
    (void)pSrcA;
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    Npp8u h, l, s;
    rgb_to_hls_pixel(pRowR[x], pRowG[x], pRowB[x], h, l, s);

    int idx = x * 4;
    pDstRow[idx + 0] = h;
    pDstRow[idx + 1] = l;
    pDstRow[idx + 2] = s;
    pDstRow[idx + 3] = 0;
  }
}

__global__ void bgr_to_hls_ap4_kernel(const Npp8u *pSrcB, const Npp8u *pSrcG, const Npp8u *pSrcR,
                                     const Npp8u *pSrcA, int nSrcStep, Npp8u *pDstH, Npp8u *pDstL, Npp8u *pDstS,
                                     Npp8u *pDstA, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pRowB = (const Npp8u *)((const char *)pSrcB + y * nSrcStep);
    const Npp8u *pRowG = (const Npp8u *)((const char *)pSrcG + y * nSrcStep);
    const Npp8u *pRowR = (const Npp8u *)((const char *)pSrcR + y * nSrcStep);
    (void)pSrcA;
    Npp8u *pDstRowH = (Npp8u *)((char *)pDstH + y * nDstStep);
    Npp8u *pDstRowL = (Npp8u *)((char *)pDstL + y * nDstStep);
    Npp8u *pDstRowS = (Npp8u *)((char *)pDstS + y * nDstStep);
    Npp8u *pDstRowA = (Npp8u *)((char *)pDstA + y * nDstStep);

    Npp8u h, l, s;
    rgb_to_hls_pixel(pRowR[x], pRowG[x], pRowB[x], h, l, s);

    pDstRowH[x] = h;
    pDstRowL[x] = l;
    pDstRowS[x] = s;
    pDstRowA[x] = 0;
  }
}

__global__ void hls_to_bgr_c3p3_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstB, Npp8u *pDstG,
                                      Npp8u *pDstR, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *pDstRowB = (Npp8u *)((char *)pDstB + y * nDstStep);
    Npp8u *pDstRowG = (Npp8u *)((char *)pDstG + y * nDstStep);
    Npp8u *pDstRowR = (Npp8u *)((char *)pDstR + y * nDstStep);

    int idx = x * 3;
    Npp8u h = pSrcRow[idx + 0];
    Npp8u l = pSrcRow[idx + 1];
    Npp8u s = pSrcRow[idx + 2];

    Npp8u r, g, b;
    hls_to_rgb_pixel(h, l, s, r, g, b);

    pDstRowB[x] = b;
    pDstRowG[x] = g;
    pDstRowR[x] = r;
  }
}

__global__ void hls_to_bgr_p3c3_kernel(const Npp8u *pSrcH, const Npp8u *pSrcL, const Npp8u *pSrcS, int nSrcStep,
                                      Npp8u *pDst, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pRowH = (const Npp8u *)((const char *)pSrcH + y * nSrcStep);
    const Npp8u *pRowL = (const Npp8u *)((const char *)pSrcL + y * nSrcStep);
    const Npp8u *pRowS = (const Npp8u *)((const char *)pSrcS + y * nSrcStep);
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    Npp8u r, g, b;
    hls_to_rgb_pixel(pRowH[x], pRowL[x], pRowS[x], r, g, b);

    int idx = x * 3;
    pDstRow[idx + 0] = b;
    pDstRow[idx + 1] = g;
    pDstRow[idx + 2] = r;
  }
}

__global__ void hls_to_bgr_p3_kernel(const Npp8u *pSrcH, const Npp8u *pSrcL, const Npp8u *pSrcS, int nSrcStep,
                                    Npp8u *pDstB, Npp8u *pDstG, Npp8u *pDstR, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pRowH = (const Npp8u *)((const char *)pSrcH + y * nSrcStep);
    const Npp8u *pRowL = (const Npp8u *)((const char *)pSrcL + y * nSrcStep);
    const Npp8u *pRowS = (const Npp8u *)((const char *)pSrcS + y * nSrcStep);
    Npp8u *pDstRowB = (Npp8u *)((char *)pDstB + y * nDstStep);
    Npp8u *pDstRowG = (Npp8u *)((char *)pDstG + y * nDstStep);
    Npp8u *pDstRowR = (Npp8u *)((char *)pDstR + y * nDstStep);

    Npp8u r, g, b;
    hls_to_rgb_pixel(pRowH[x], pRowL[x], pRowS[x], r, g, b);

    pDstRowB[x] = b;
    pDstRowG[x] = g;
    pDstRowR[x] = r;
  }
}

__global__ void hls_to_bgr_ac4p4_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstB, Npp8u *pDstG,
                                       Npp8u *pDstR, Npp8u *pDstA, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *pDstRowB = (Npp8u *)((char *)pDstB + y * nDstStep);
    Npp8u *pDstRowG = (Npp8u *)((char *)pDstG + y * nDstStep);
    Npp8u *pDstRowR = (Npp8u *)((char *)pDstR + y * nDstStep);
    Npp8u *pDstRowA = (Npp8u *)((char *)pDstA + y * nDstStep);

    int idx = x * 4;
    Npp8u h = pSrcRow[idx + 0];
    Npp8u l = pSrcRow[idx + 1];
    Npp8u s = pSrcRow[idx + 2];

    Npp8u r, g, b;
    hls_to_rgb_pixel(h, l, s, r, g, b);

    pDstRowB[x] = b;
    pDstRowG[x] = g;
    pDstRowR[x] = r;
    pDstRowA[x] = 0;
  }
}

__global__ void hls_to_bgr_ap4c4_kernel(const Npp8u *pSrcH, const Npp8u *pSrcL, const Npp8u *pSrcS,
                                       const Npp8u *pSrcA, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                       int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pRowH = (const Npp8u *)((const char *)pSrcH + y * nSrcStep);
    const Npp8u *pRowL = (const Npp8u *)((const char *)pSrcL + y * nSrcStep);
    const Npp8u *pRowS = (const Npp8u *)((const char *)pSrcS + y * nSrcStep);
    (void)pSrcA;
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    Npp8u r, g, b;
    hls_to_rgb_pixel(pRowH[x], pRowL[x], pRowS[x], r, g, b);

    int idx = x * 4;
    pDstRow[idx + 0] = b;
    pDstRow[idx + 1] = g;
    pDstRow[idx + 2] = r;
    pDstRow[idx + 3] = 0;
  }
}

__global__ void hls_to_bgr_ap4_kernel(const Npp8u *pSrcH, const Npp8u *pSrcL, const Npp8u *pSrcS,
                                     const Npp8u *pSrcA, int nSrcStep, Npp8u *pDstB, Npp8u *pDstG, Npp8u *pDstR,
                                     Npp8u *pDstA, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pRowH = (const Npp8u *)((const char *)pSrcH + y * nSrcStep);
    const Npp8u *pRowL = (const Npp8u *)((const char *)pSrcL + y * nSrcStep);
    const Npp8u *pRowS = (const Npp8u *)((const char *)pSrcS + y * nSrcStep);
    (void)pSrcA;
    Npp8u *pDstRowB = (Npp8u *)((char *)pDstB + y * nDstStep);
    Npp8u *pDstRowG = (Npp8u *)((char *)pDstG + y * nDstStep);
    Npp8u *pDstRowR = (Npp8u *)((char *)pDstR + y * nDstStep);
    Npp8u *pDstRowA = (Npp8u *)((char *)pDstA + y * nDstStep);

    Npp8u r, g, b;
    hls_to_rgb_pixel(pRowH[x], pRowL[x], pRowS[x], r, g, b);

    pDstRowB[x] = b;
    pDstRowG[x] = g;
    pDstRowR[x] = r;
    pDstRowA[x] = 0;
  }
}

extern "C" {

cudaError_t nppiRGBToHLS_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  rgb_to_hls_c3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                           oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiRGBToHLS_8u_AC4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                        cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  rgb_to_hls_ac4_kernel<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                           oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiHLSToRGB_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  hls_to_rgb_c3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                           oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiHLSToRGB_8u_AC4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                        cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  hls_to_rgb_ac4_kernel<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                            oSizeROI.height);
  return cudaGetLastError();
}


cudaError_t nppiBGRToHLS_8u_AC4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                        cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  bgr_to_hls_ac4_kernel<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                           oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiBGRToHLS_8u_C3P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstH, Npp8u *pDstL,
                                         Npp8u *pDstS, int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  bgr_to_hls_c3p3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDstH, pDstL, pDstS, nDstStep,
                                                            oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiBGRToHLS_8u_P3C3R_kernel(const Npp8u *pSrcB, int nSrcStep, const Npp8u *pSrcG, const Npp8u *pSrcR,
                                         Npp8u *pDst, int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  bgr_to_hls_p3c3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcB, pSrcG, pSrcR, nSrcStep, pDst, nDstStep,
                                                            oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiBGRToHLS_8u_P3R_kernel(const Npp8u *pSrcB, int nSrcStep, const Npp8u *pSrcG, const Npp8u *pSrcR,
                                       Npp8u *pDstH, Npp8u *pDstL, Npp8u *pDstS, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  bgr_to_hls_p3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcB, pSrcG, pSrcR, nSrcStep, pDstH, pDstL, pDstS,
                                                          nDstStep, oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiBGRToHLS_8u_AC4P4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstH, Npp8u *pDstL,
                                          Npp8u *pDstS, Npp8u *pDstA, int nDstStep, NppiSize oSizeROI,
                                          cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  bgr_to_hls_ac4p4_kernel<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDstH, pDstL, pDstS, pDstA, nDstStep,
                                                             oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiBGRToHLS_8u_AP4C4R_kernel(const Npp8u *pSrcB, int nSrcStep, const Npp8u *pSrcG, const Npp8u *pSrcR,
                                          const Npp8u *pSrcA, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                          cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  bgr_to_hls_ap4c4_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcB, pSrcG, pSrcR, pSrcA, nSrcStep, pDst, nDstStep,
                                                             oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiBGRToHLS_8u_AP4R_kernel(const Npp8u *pSrcB, int nSrcStep, const Npp8u *pSrcG, const Npp8u *pSrcR,
                                        const Npp8u *pSrcA, Npp8u *pDstH, Npp8u *pDstL, Npp8u *pDstS, Npp8u *pDstA,
                                        int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  bgr_to_hls_ap4_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcB, pSrcG, pSrcR, pSrcA, nSrcStep, pDstH, pDstL, pDstS,
                                                           pDstA, nDstStep, oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiHLSToBGR_8u_C3P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstB, Npp8u *pDstG,
                                         Npp8u *pDstR, int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  hls_to_bgr_c3p3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDstB, pDstG, pDstR, nDstStep,
                                                            oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiHLSToBGR_8u_P3C3R_kernel(const Npp8u *pSrcH, int nSrcStep, const Npp8u *pSrcL, const Npp8u *pSrcS,
                                         Npp8u *pDst, int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  hls_to_bgr_p3c3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcH, pSrcL, pSrcS, nSrcStep, pDst, nDstStep,
                                                            oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiHLSToBGR_8u_P3R_kernel(const Npp8u *pSrcH, int nSrcStep, const Npp8u *pSrcL, const Npp8u *pSrcS,
                                       Npp8u *pDstB, Npp8u *pDstG, Npp8u *pDstR, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  hls_to_bgr_p3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcH, pSrcL, pSrcS, nSrcStep, pDstB, pDstG, pDstR,
                                                          nDstStep, oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiHLSToBGR_8u_AC4P4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstB, Npp8u *pDstG,
                                          Npp8u *pDstR, Npp8u *pDstA, int nDstStep, NppiSize oSizeROI,
                                          cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  hls_to_bgr_ac4p4_kernel<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDstB, pDstG, pDstR, pDstA, nDstStep,
                                                             oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiHLSToBGR_8u_AP4C4R_kernel(const Npp8u *pSrcH, int nSrcStep, const Npp8u *pSrcL, const Npp8u *pSrcS,
                                          const Npp8u *pSrcA, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                          cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  hls_to_bgr_ap4c4_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcH, pSrcL, pSrcS, pSrcA, nSrcStep, pDst, nDstStep,
                                                             oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiHLSToBGR_8u_AP4R_kernel(const Npp8u *pSrcH, int nSrcStep, const Npp8u *pSrcL, const Npp8u *pSrcS,
                                        const Npp8u *pSrcA, Npp8u *pDstB, Npp8u *pDstG, Npp8u *pDstR, Npp8u *pDstA,
                                        int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  hls_to_bgr_ap4_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcH, pSrcL, pSrcS, pSrcA, nSrcStep, pDstB, pDstG, pDstR,
                                                           pDstA, nDstStep, oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

}