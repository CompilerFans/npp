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

__device__ inline void rgb_to_hsv_pixel(Npp8u r, Npp8u g, Npp8u b, Npp8u &h, Npp8u &s, Npp8u &v) {
  float rf = r / 255.0f;
  float gf = g / 255.0f;
  float bf = b / 255.0f;

  float maxv = fmaxf(rf, fmaxf(gf, bf));
  float minv = fminf(rf, fminf(gf, bf));
  float delta = maxv - minv;

  float hval = 0.0f;
  float sval = (maxv <= 0.0f) ? 0.0f : (delta / maxv);

  if (delta > 0.0f) {
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
  s = clamp_u8_round(sval * 255.0f);
  v = clamp_u8_round(maxv * 255.0f);
}

__device__ inline void hsv_to_rgb_pixel(Npp8u h, Npp8u s, Npp8u v, Npp8u &r, Npp8u &g, Npp8u &b) {
  float hf = h / 255.0f;
  float sf = s / 255.0f;
  float vf = v / 255.0f;

  if (sf <= 0.0f) {
    r = g = b = clamp_u8_round(vf * 255.0f);
    return;
  }

  float h6 = hf * 6.0f;
  int i = static_cast<int>(floorf(h6));
  float f = h6 - static_cast<float>(i);
  float p = vf * (1.0f - sf);
  float q = vf * (1.0f - sf * f);
  float t = vf * (1.0f - sf * (1.0f - f));

  float rf = 0.0f, gf = 0.0f, bf = 0.0f;
  switch (i % 6) {
  case 0:
    rf = vf;
    gf = t;
    bf = p;
    break;
  case 1:
    rf = q;
    gf = vf;
    bf = p;
    break;
  case 2:
    rf = p;
    gf = vf;
    bf = t;
    break;
  case 3:
    rf = p;
    gf = q;
    bf = vf;
    break;
  case 4:
    rf = t;
    gf = p;
    bf = vf;
    break;
  default:
    rf = vf;
    gf = p;
    bf = q;
    break;
  }

  r = clamp_u8_round(rf * 255.0f);
  g = clamp_u8_round(gf * 255.0f);
  b = clamp_u8_round(bf * 255.0f);
}

__global__ void rgb_to_hsv_c3_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
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

    Npp8u h, s, v;
    rgb_to_hsv_pixel(r, g, b, h, s, v);

    pDstRow[idx + 0] = h;
    pDstRow[idx + 1] = s;
    pDstRow[idx + 2] = v;
  }
}

__global__ void rgb_to_hsv_ac4_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
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

    Npp8u h, s, v;
    rgb_to_hsv_pixel(r, g, b, h, s, v);

    pDstRow[idx + 0] = h;
    pDstRow[idx + 1] = s;
    pDstRow[idx + 2] = v;
    pDstRow[idx + 3] = 0;
  }
}

__global__ void hsv_to_rgb_c3_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                     int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    int idx = x * 3;
    Npp8u h = pSrcRow[idx + 0];
    Npp8u s = pSrcRow[idx + 1];
    Npp8u v = pSrcRow[idx + 2];

    Npp8u r, g, b;
    hsv_to_rgb_pixel(h, s, v, r, g, b);

    pDstRow[idx + 0] = r;
    pDstRow[idx + 1] = g;
    pDstRow[idx + 2] = b;
  }
}

__global__ void hsv_to_rgb_ac4_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                      int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    int idx = x * 4;
    Npp8u h = pSrcRow[idx + 0];
    Npp8u s = pSrcRow[idx + 1];
    Npp8u v = pSrcRow[idx + 2];

    Npp8u r, g, b;
    hsv_to_rgb_pixel(h, s, v, r, g, b);

    pDstRow[idx + 0] = r;
    pDstRow[idx + 1] = g;
    pDstRow[idx + 2] = b;
    pDstRow[idx + 3] = 0;
  }
}

extern "C" {

cudaError_t nppiRGBToHSV_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  rgb_to_hsv_c3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                           oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiRGBToHSV_8u_AC4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                        cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  rgb_to_hsv_ac4_kernel<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                           oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiHSVToRGB_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  hsv_to_rgb_c3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                           oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiHSVToRGB_8u_AC4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                        cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  hsv_to_rgb_ac4_kernel<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                            oSizeROI.height);
  return cudaGetLastError();
}
}
