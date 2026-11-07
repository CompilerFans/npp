// c3 color-space conversion kernels (see nppi_c3_color_executor.h).
//
// Coefficients come from channel-ramp fits against the NVIDIA NPP 13.4
// reference and are applied in single precision in the recorded order.
// Rounding is C-style truncation towards zero followed by saturation.

#include "nppi_c3_color_executor.h"
#include "nppi_c3_cmyk_mul.h"

#include <cuda_runtime.h>

namespace nppi {
namespace c3color {
namespace {

constexpr int kBlockX = 128;
constexpr int kBlockY = 8;

inline dim3 gridFor(int w, int h) {
  return dim3((w + kBlockX - 1) / kBlockX, (h + kBlockY - 1) / kBlockY);
}
inline bool step_ok(int nStep, int nMinRowBytes) {
  return static_cast<unsigned>(nStep) >= static_cast<unsigned>(nMinRowBytes);
}
inline bool roi_valid(NppiSize roi) { return roi.width >= 0 && roi.height >= 0; }
inline NppStatus launch_ok() {
  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

__device__ __forceinline__ int sat_u8(float v) {
  int t = static_cast<int>(v);
  return t < 0 ? 0: (t > 255 ? 255: t);
}

struct Mat {
  float yr, yg, yb, yo;
  float cbr, cbg, cbb, cbo;
  float crr, crg, crb, cro;
};

__host__ __device__ __forceinline__ Mat matrix(int id) {
  Mat m;
  if (id == 0) { // BT.601 video range
    m = {0.257f, 0.504f, 0.098f, 16.0f, -0.148f, -0.291f, 0.439f, 128.0f, 0.439f, -0.368f, -0.071f, 128.0f};
  } else if (id == 1) { // BT.709 video range
    m = {0.183f, 0.614f, 0.062f, 16.0f, -0.101f, -0.338f, 0.439f, 128.0f, 0.439f, -0.399f, -0.040f, 128.0f};
  } else { // BT.709 full range (HDTV)
    m = {0.213f, 0.715f, 0.072f, 0.0f, -0.117f, -0.394f, 0.511f, 128.0f, 0.511f, -0.464f, -0.047f, 128.0f};
  }
  return m;
}

__device__ __forceinline__ void rgb_of(const Npp8u *s, int pp, int bgr, float &r, float &g, float &b) {
  r = static_cast<float>(bgr ? s[2]: s[0]);
  g = static_cast<float>(s[1]);
  b = static_cast<float>(bgr ? s[0]: s[2]);
  (void)pp;
}

__global__ void k_fwd420_y(const Npp8u *__restrict__ pSrc, int nSrcStep, int pp, int bgr, Npp8u *__restrict__ pDst,
                           int nDstStep, int w, int h, Mat m) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= (w & ~1) || y >= (h & ~1)) {
    return;
  }
  float r, g, b;
  rgb_of(pSrc + (size_t)y * nSrcStep + (size_t)x * pp, pp, bgr, r, g, b);
  pDst[(size_t)y * nDstStep + x] = (Npp8u)sat_u8(m.yr * r + m.yg * g + m.yb * b + m.yo);
}

__global__ void k_fwd420_c(const Npp8u *__restrict__ pSrc, int nSrcStep, int pp, int bgr, Npp8u *__restrict__ pC0,
                           int nC0Step, Npp8u *__restrict__ pC1, int nC1Step, int w, int h, Mat m) {
  const int cx = blockIdx.x * blockDim.x + threadIdx.x;
  const int cy = blockIdx.y * blockDim.y + threadIdx.y;
  if (cx >= (w >> 1) || cy >= (h >> 1)) {
    return;
  }
  int sc0 = 0, sc1 = 0;
  for (int dy = 0; dy < 2; ++dy) {
    for (int dx = 0; dx < 2; ++dx) {
      float r, g, b;
      rgb_of(pSrc + (size_t)(2 * cy + dy) * nSrcStep + (size_t)(2 * cx + dx) * pp, pp, bgr, r, g, b);
      sc0 += sat_u8(m.cbr * r + m.cbg * g + m.cbb * b + m.cbo);
      sc1 += sat_u8(m.crr * r + m.crg * g + m.crb * b + m.cro);
    }
  }
  pC0[(size_t)cy * nC0Step + cx] = (Npp8u)(sc0 >> 2);
  pC1[(size_t)cy * nC1Step + cx] = (Npp8u)(sc1 >> 2);
}

// Packed 4:2:2 output.  cbycr=0 -> [Y0 Cr Y1 Cb] (YCrCb422), 1 -> [Cb Y0 Cr Y1].
__global__ void k_fwdC2(const Npp8u *__restrict__ pSrc, int nSrcStep, int pp, int bgr, Npp8u *__restrict__ pDst,
                        int nDstStep, int w, int h, Mat m, int cbycr) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (p >= (w >> 1) || y >= h) {
    return;
  }
  float r0, g0, b0, r1, g1, b1;
  const Npp8u *row = pSrc + (size_t)y * nSrcStep;
  rgb_of(row + (size_t)(2 * p) * pp, pp, bgr, r0, g0, b0);
  rgb_of(row + (size_t)(2 * p + 1) * pp, pp, bgr, r1, g1, b1);
  const int y0 = sat_u8(m.yr * r0 + m.yg * g0 + m.yb * b0 + m.yo);
  const int y1 = sat_u8(m.yr * r1 + m.yg * g1 + m.yb * b1 + m.yo);
  const int cb = (sat_u8(m.cbr * r0 + m.cbg * g0 + m.cbb * b0 + m.cbo) +
                  sat_u8(m.cbr * r1 + m.cbg * g1 + m.cbb * b1 + m.cbo)) >> 1;
  const int cr = (sat_u8(m.crr * r0 + m.crg * g0 + m.crb * b0 + m.cro) +
                  sat_u8(m.crr * r1 + m.crg * g1 + m.crb * b1 + m.cro)) >> 1;
  Npp8u *d = pDst + (size_t)y * nDstStep + (size_t)p * 4;
  if (cbycr) {
    d[0] = (Npp8u)cb;
    d[1] = (Npp8u)y0;
    d[2] = (Npp8u)cr;
    d[3] = (Npp8u)y1;
  } else {
    d[0] = (Npp8u)y0;
    d[1] = (Npp8u)cr;
    d[2] = (Npp8u)y1;
    d[3] = (Npp8u)cb;
  }
}

// Planar RGB -> packed 4:2:2 YCrCb422 [Y0 Cr Y1 Cb].
__global__ void k_fwdP3C2(const Npp8u *__restrict__ pR, const Npp8u *__restrict__ pG, const Npp8u *__restrict__ pB,
                          int nSrcStep, Npp8u *__restrict__ pDst, int nDstStep, int w, int h) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (p >= (w >> 1) || y >= h) {
    return;
  }
  const size_t i0 = (size_t)y * nSrcStep + 2 * p;
  const size_t i1 = i0 + 1;
  const float r0 = pR[i0], g0 = pG[i0], b0 = pB[i0];
  const float r1 = pR[i1], g1 = pG[i1], b1 = pB[i1];
  const Mat m = matrix(0);
  const int y0 = sat_u8(m.yr * r0 + m.yg * g0 + m.yb * b0 + m.yo);
  const int y1 = sat_u8(m.yr * r1 + m.yg * g1 + m.yb * b1 + m.yo);
  const int cb = (sat_u8(m.cbr * r0 + m.cbg * g0 + m.cbb * b0 + m.cbo) +
                  sat_u8(m.cbr * r1 + m.cbg * g1 + m.cbb * b1 + m.cbo)) >> 1;
  const int cr = (sat_u8(m.crr * r0 + m.crg * g0 + m.crb * b0 + m.cro) +
                  sat_u8(m.crr * r1 + m.crg * g1 + m.crb * b1 + m.cro)) >> 1;
  Npp8u *d = pDst + (size_t)y * nDstStep + (size_t)p * 4;
  d[0] = (Npp8u)y0;
  d[1] = (Npp8u)cr;
  d[2] = (Npp8u)y1;
  d[3] = (Npp8u)cb;
}

// Inverse: YUV -> RGB/BGR for packed C3/C4 destinations.
__device__ __forceinline__ void yuv_to_rgb(int yv, int cb, int cr, int matrixId, int &r, int &g, int &b) {
  const float fy = static_cast<float>(yv);
  const float fcb = static_cast<float>(cb) - 128.0f;
  const float fcr = static_cast<float>(cr) - 128.0f;
  if (matrixId == 0) {
    const float ny = 1.164f * (fy - 16.0f);
    r = sat_u8(ny + 1.596f * fcr);
    g = sat_u8(ny - 0.392f * fcb - 0.813f * fcr);
    b = sat_u8(ny + 2.017f * fcb);
  } else if (matrixId == 1) {
    const float ny = 1.164f * (fy - 16.0f);
    r = sat_u8(ny + 1.793f * fcr);
    g = sat_u8(ny - 0.213f * fcb - 0.534f * fcr);
    b = sat_u8(ny + 2.115f * fcb);
  } else {
    r = sat_u8(fy + 1.54f * fcr);
    g = sat_u8(fy - 0.183f * fcb - 0.459f * fcr);
    b = sat_u8(fy + 1.816f * fcb);
  }
}

template <int BGR>
__global__ void k_inv420(const Npp8u *__restrict__ pY, int nYStep, const Npp8u *__restrict__ pC0, int nC0Step,
                         const Npp8u *__restrict__ pC1, int nC1Step, Npp8u *__restrict__ pDst, int nDstStep, int pp,
                         int w, int h, int matrixId, int srcSwap, Npp8u aval) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) {
    return;
  }
  const int cb = srcSwap ? pC1[(size_t)(y >> 1) * nC1Step + (x >> 1)]: pC0[(size_t)(y >> 1) * nC0Step + (x >> 1)];
  const int cr = srcSwap ? pC0[(size_t)(y >> 1) * nC0Step + (x >> 1)]: pC1[(size_t)(y >> 1) * nC1Step + (x >> 1)];
  int r, g, b;
  yuv_to_rgb(pY[(size_t)y * nYStep + x], cb, cr, matrixId, r, g, b);
  Npp8u *d = pDst + (size_t)y * nDstStep + (size_t)x * pp;
  if (BGR) {
    d[0] = (Npp8u)b;
    d[1] = (Npp8u)g;
    d[2] = (Npp8u)r;
  } else {
    d[0] = (Npp8u)r;
    d[1] = (Npp8u)g;
    d[2] = (Npp8u)b;
  }
  if (pp == 4) {
    d[3] = aval;
  }
}

// 4:2:2 planar source (Y, Cb, Cr planes with a single common step) -> BGR.
__global__ void k_inv422(const Npp8u *__restrict__ pY, const Npp8u *__restrict__ pCb, const Npp8u *__restrict__ pCr,
                         int nSrcStep, Npp8u *__restrict__ pDst, int nDstStep, int pp, int w, int h,
                         Npp8u aval) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) {
    return;
  }
  // The 709CSC planar family is full resolution (4:4:4): one chroma sample
  // per luma pixel.
  const size_t i = (size_t)y * nSrcStep + x;
  int r, g, b;
  yuv_to_rgb(pY[(size_t)y * nSrcStep + x], pCb[i], pCr[i], 1, r, g, b);
  Npp8u *d = pDst + (size_t)y * nDstStep + (size_t)x * pp;
  d[0] = (Npp8u)b;
  d[1] = (Npp8u)g;
  d[2] = (Npp8u)r;
  if (pp == 4) {
    d[3] = aval;
  }
}

// Packed CbYCr422 [Cb Y0 Cr Y1] -> BGR (BT.709 full range).
__global__ void k_invC2(const Npp8u *__restrict__ pSrc, int nSrcStep, Npp8u *__restrict__ pDst, int nDstStep, int pp,
                        int w, int h, Npp8u aval) {
  const int px = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= w || y >= h) {
    return;
  }
  int p = px >> 1;
  const int lastPair = (w >> 1) - 1;
  if (p > lastPair) {
    p = lastPair; // trailing odd pixel reuses the last pair's chroma
  }
  const Npp8u *s = pSrc + (size_t)y * nSrcStep + (size_t)p * 4;
  int r, g, b;
  yuv_to_rgb(s[(px & 1) ? 3: 1], s[0], s[2], 2, r, g, b);
  Npp8u *d = pDst + (size_t)y * nDstStep + (size_t)px * pp;
  d[0] = (Npp8u)b;
  d[1] = (Npp8u)g;
  d[2] = (Npp8u)r;
  if (pp == 4) {
    d[3] = aval;
  }
}

// Packed YCrCb422 [Y0 Cr Y1 Cb] -> RGB.
__device__ __forceinline__ void ycrcb422_pixel(const Npp8u *__restrict__ pSrc, int nSrcStep, int x, int y, int w,
                                               int &r, int &g, int &b) {
  int p = x >> 1;
  const int lastPair = (w >> 1) - 1;
  if (p > lastPair) {
    p = lastPair;
  }
  const Npp8u *s = pSrc + (size_t)y * nSrcStep + (size_t)p * 4;
  yuv_to_rgb(s[(x & 1) ? 2: 0], s[3], s[1], 0, r, g, b);
}

__global__ void k_ycrcb422_to_rgb_c2(const Npp8u *__restrict__ pSrc, int nSrcStep, Npp8u *__restrict__ pDst,
                                     int nDstStep, int pp, int w, int h) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) {
    return;
  }
  int r, g, b;
  ycrcb422_pixel(pSrc, nSrcStep, x, y, w, r, g, b);
  Npp8u *d = pDst + (size_t)y * nDstStep + (size_t)x * pp;
  d[0] = (Npp8u)r;
  d[1] = (Npp8u)g;
  d[2] = (Npp8u)b;
}

__global__ void k_ycrcb422_to_rgb_p3(const Npp8u *__restrict__ pSrc, int nSrcStep, Npp8u *__restrict__ pR,
                                     Npp8u *__restrict__ pG, Npp8u *__restrict__ pB, int nDstStep, int w, int h) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) {
    return;
  }
  int r, g, b;
  ycrcb422_pixel(pSrc, nSrcStep, x, y, w, r, g, b);
  pR[(size_t)y * nDstStep + x] = (Npp8u)r;
  pG[(size_t)y * nDstStep + x] = (Npp8u)g;
  pB[(size_t)y * nDstStep + x] = (Npp8u)b;
}

// PhotoYCC forward/inverse (packed C3/AC4).
template <int PP, int BGR>
__global__ void k_rgb_to_ycc(const Npp8u *__restrict__ pSrc, int nSrcStep, Npp8u *__restrict__ pDst, int nDstStep,
                             int w, int h) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) {
    return;
  }
  const Npp8u *s = pSrc + (size_t)y * nSrcStep + (size_t)x * PP;
  const float nr = static_cast<float>(BGR ? s[2]: s[0]) * 0.003921569f;
  const float ng = static_cast<float>(s[1]) * 0.003921569f;
  const float nb = static_cast<float>(BGR ? s[0]: s[2]) * 0.003921569f;
  const float ny = 0.299f * nr + 0.587f * ng + 0.114f * nb;
  float c1 = nb - ny;
  c1 = 111.4f * 0.003921569f * c1 + 156.0f * 0.003921569f;
  float c2 = nr - ny;
  c2 = 135.64f * 0.003921569f * c2 + 137.0f * 0.003921569f;
  const float yv = 0.713267f * ny;
  Npp8u *d = pDst + (size_t)y * nDstStep + (size_t)x * PP;
  d[0] = (Npp8u)sat_u8(yv * 255.0f);
  d[1] = (Npp8u)sat_u8(c1 * 255.0f);
  d[2] = (Npp8u)sat_u8(c2 * 255.0f);
  // PP == 4: the destination alpha slot is left untouched (measured).
}

template <int PP>
__global__ void k_ycc_to_rgb(const Npp8u *__restrict__ pSrc, int nSrcStep, Npp8u *__restrict__ pDst, int nDstStep,
                             int w, int h) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) {
    return;
  }
  const Npp8u *s = pSrc + (size_t)y * nSrcStep + (size_t)x * PP;
  const float ny = (static_cast<float>(s[0]) * 0.003921569f) * 1.3584f;
  const float c1 = ((static_cast<float>(s[1]) * 0.003921569f) - 156.0f * 0.003921569f) * 2.2179f;
  const float c2 = ((static_cast<float>(s[2]) * 0.003921569f) - 137.0f * 0.003921569f) * 1.8215f;
  const int r = sat_u8((ny + c2) * 255.0f);
  const int g = sat_u8((ny - 0.194f * c1 - 0.509f * c2) * 255.0f);
  const int b = sat_u8((ny + c1) * 255.0f);
  Npp8u *d = pDst + (size_t)y * nDstStep + (size_t)x * PP;
  d[0] = (Npp8u)r;
  d[1] = (Npp8u)g;
  d[2] = (Npp8u)b;
  // PP == 4: NVIDIA leaves the destination alpha slot untouched for
  // YCCToRGB_AC4R (measured; unlike RGBToYCC_AC4R which copies it).
}

// One channel of the JPEG CMYK -> RGB multiply, matching the reference's
// integer path exactly (see nppi_c3_cmyk_mul.h).
__device__ __forceinline__ int cmyk_mul(int c, int k) {
  const int x = 255 - c;
  const int y = 255 - k;
  const int p = x * y;
  const int q = p / 255;
  if (q > 0 && p == q * 255) {
    const unsigned key = (static_cast<unsigned>(y) << 8) | static_cast<unsigned>(x);
    int lo = 0, hi = 387;
    while (lo <= hi) {
      const int mid = (lo + hi) >> 1;
      const unsigned v = kCmykMulExceptions[mid];
      if (v == key) {
        return q;
      }
      if (v < key) {
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    return q - 1;
  }
  return q;
}

// JPEG CMYK -> RGB/BGR.
template <int BGR>
__device__ __forceinline__ void cmyk_pixel(int c, int m, int yv, int k, int &r, int &g, int &b) {
  r = cmyk_mul(c, k);
  g = cmyk_mul(m, k);
  b = cmyk_mul(yv, k);
  (void)BGR;
}

template <int BGR, int PP>
__global__ void k_cmyk_to_packed(const Npp8u *__restrict__ p0, const Npp8u *__restrict__ p1,
                                 const Npp8u *__restrict__ p2, const Npp8u *__restrict__ p3, int nSrcStep,
                                 Npp8u *__restrict__ pDst, int nDstStep, int w, int h) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) {
    return;
  }
  const size_t i = (size_t)y * nSrcStep + x;
  int r, g, b;
  cmyk_pixel<BGR>(p0[i], p1[i], p2[i], p3[i], r, g, b);
  Npp8u *d = pDst + (size_t)y * nDstStep + (size_t)x * PP;
  if (BGR) {
    d[0] = (Npp8u)b;
    d[1] = (Npp8u)g;
    d[2] = (Npp8u)r;
  } else {
    d[0] = (Npp8u)r;
    d[1] = (Npp8u)g;
    d[2] = (Npp8u)b;
  }
}

__global__ void k_cmyk_to_planar(const Npp8u *__restrict__ p0, const Npp8u *__restrict__ p1,
                                 const Npp8u *__restrict__ p2, const Npp8u *__restrict__ p3, int nSrcStep,
                                 Npp8u *__restrict__ pR, Npp8u *__restrict__ pG, Npp8u *__restrict__ pB, int nDstStep,
                                 int w, int h) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) {
    return;
  }
  const size_t i = (size_t)y * nSrcStep + x;
  int r, g, b;
  cmyk_pixel<0>(p0[i], p1[i], p2[i], p3[i], r, g, b);
  pR[(size_t)y * nDstStep + x] = (Npp8u)r;
  pG[(size_t)y * nDstStep + x] = (Npp8u)g;
  pB[(size_t)y * nDstStep + x] = (Npp8u)b;
}

__global__ void k_ycck_to_cmyk(const Npp8u *__restrict__ p0, const Npp8u *__restrict__ p1,
                               const Npp8u *__restrict__ p2, const Npp8u *__restrict__ p3, int nSrcStep,
                               Npp8u *__restrict__ q0, Npp8u *__restrict__ q1, Npp8u *__restrict__ q2,
                               Npp8u *__restrict__ q3, int nDstStep, int w, int h) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) {
    return;
  }
  const size_t i = (size_t)y * nSrcStep + x;
  const float fy = static_cast<float>(p0[i]);
  const float fcb = static_cast<float>(p1[i]);
  const float fcr = static_cast<float>(p2[i]);
  const float nR = fy + 1.402f * fcr - 179.456f;
  const float nG = fy - 0.34414f * fcb - 0.71414f * fcr + 135.45984f;
  const float nB = fy + 1.772f * fcb - 226.816f;
  const size_t o = (size_t)y * nDstStep + x;
  q0[o] = (Npp8u)sat_u8(255.0f - nR);
  q1[o] = (Npp8u)sat_u8(255.0f - nG);
  q2[o] = (Npp8u)sat_u8(255.0f - nB);
  q3[o] = p3[i];
}

} // namespace

// reference: a packed 4-channel source step must be a multiple of the pixel size
// (4 bytes for 8u); a misaligned step is NOT_EVEN_STEP_ERROR, checked after
// the too-short-step test.
inline bool ac4_src_step_bad(int nSrcStep, int srcPp) {
  return srcPp == 4 && (static_cast<unsigned>(nSrcStep) & 3u) != 0u;
}

NppStatus fwd420(const Npp8u *pSrc, int nSrcStep, int srcPp, int bgr, Npp8u *pDst[3], const int aDstStep[3],
                 NppiSize oSizeROI, cudaStream_t stream, int matrixId, int swapPlanes) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nSrcStep, srcPp * oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  if (ac4_src_step_bad(nSrcStep, srcPp)) {
    return NPP_NOT_EVEN_STEP_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  if (w == 0 || h == 0) {
    return NPP_NO_ERROR;
  }
  const Mat m = matrix(matrixId);
  k_fwd420_y<<<gridFor(w, h), dim3(kBlockX, kBlockY), 0, stream>>>(pSrc, nSrcStep, srcPp, bgr, pDst[0], aDstStep[0], w,
                                                                  h, m);
  Npp8u *c0 = swapPlanes ? pDst[2]: pDst[1];
  Npp8u *c1 = swapPlanes ? pDst[1]: pDst[2];
  const int s0 = swapPlanes ? aDstStep[2]: aDstStep[1];
  const int s1 = swapPlanes ? aDstStep[1]: aDstStep[2];
  if ((w >> 1) > 0 && (h >> 1) > 0) {
    k_fwd420_c<<<gridFor(w >> 1, h >> 1), dim3(kBlockX, kBlockY), 0, stream>>>(pSrc, nSrcStep, srcPp, bgr, c0, s0, c1,
                                                                               s1, w, h, m);
  }
  NppStatus st = launch_ok();
  if (st != NPP_NO_ERROR) {
    return st;
  }
  if ((w & 1) || (h & 1)) {
    return NPP_DOUBLE_SIZE_WARNING;
  }
  return NPP_NO_ERROR;
}

NppStatus fwdC2(const Npp8u *pSrc, int nSrcStep, int srcPp, int bgr, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                cudaStream_t stream, int matrixId, int cbycr) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nSrcStep, srcPp * oSizeROI.width) || !step_ok(nDstStep, 2 * oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  if (ac4_src_step_bad(nSrcStep, srcPp)) {
    return NPP_NOT_EVEN_STEP_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  if (w == 0 || h == 0) {
    return NPP_NO_ERROR;
  }
  k_fwdC2<<<gridFor(w >> 1, h), dim3(kBlockX, kBlockY), 0, stream>>>(pSrc, nSrcStep, srcPp, bgr, pDst, nDstStep, w, h,
                                                                     matrix(matrixId), cbycr);
  return launch_ok();
}

NppStatus fwdP3C2(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                  cudaStream_t stream) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pSrc[2] || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nSrcStep, oSizeROI.width) || !step_ok(nDstStep, 2 * oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  if (w == 0 || h == 0) {
    return NPP_NO_ERROR;
  }
  k_fwdP3C2<<<gridFor(w >> 1, h), dim3(kBlockX, kBlockY), 0, stream>>>(pSrc[0], pSrc[1], pSrc[2], nSrcStep, pDst,
                                                                     nDstStep, w, h);
  NppStatus st = launch_ok();
  if (st != NPP_NO_ERROR) {
    return st;
  }
  if (w & 1) {
    return NPP_DOUBLE_SIZE_WARNING;
  }
  return NPP_NO_ERROR;
}

NppStatus inv420(const Npp8u *const pSrc[3], const int aSrcStep[3], Npp8u *pDst, int nDstStep, int dstPp, int bgr,
                 Npp8u nAval, NppiSize oSizeROI, cudaStream_t stream, int matrix, int srcSwap) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pSrc[2] || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(aSrcStep[0], oSizeROI.width) || !step_ok(nDstStep, dstPp * oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  if (w == 0 || h == 0) {
    return NPP_NO_ERROR;
  }
  // The BT.709 video-range (CSC) inverse only converts whole 2-row blocks and
  // warns for odd heights; the other 4:2:0 inverses convert every pixel.
  const int ew = w;
  const int eh = (matrix == 1) ? (h & ~1): h;
  if (bgr) {
    k_inv420<1><<<gridFor(ew, eh), dim3(kBlockX, kBlockY), 0, stream>>>(pSrc[0], aSrcStep[0], pSrc[1], aSrcStep[1],
                                                                      pSrc[2], aSrcStep[2], pDst, nDstStep, dstPp, ew,
                                                                      eh, matrix, srcSwap, nAval);
  } else {
    k_inv420<0><<<gridFor(ew, eh), dim3(kBlockX, kBlockY), 0, stream>>>(pSrc[0], aSrcStep[0], pSrc[1], aSrcStep[1],
                                                                      pSrc[2], aSrcStep[2], pDst, nDstStep, dstPp, ew,
                                                                      eh, matrix, srcSwap, nAval);
  }
  NppStatus st = launch_ok();
  if (st != NPP_NO_ERROR) {
    return st;
  }
  if (matrix == 1 && (h & 1)) {
    return NPP_DOUBLE_SIZE_WARNING;
  }
  return NPP_NO_ERROR;
}

NppStatus inv422Planar(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst, int nDstStep, int dstPp, Npp8u nAval,
                       NppiSize oSizeROI, cudaStream_t stream) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pSrc[2] || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nSrcStep, oSizeROI.width) || !step_ok(nDstStep, dstPp * oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  if (w == 0 || h == 0) {
    return NPP_NO_ERROR;
  }
  k_inv422<<<gridFor(w, h), dim3(kBlockX, kBlockY), 0, stream>>>(pSrc[0], pSrc[1], pSrc[2], nSrcStep, pDst, nDstStep,
                                                                dstPp, w, h, nAval);
  return launch_ok();
}

NppStatus invC2(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int dstPp, Npp8u nAval, NppiSize oSizeROI,
                cudaStream_t stream) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nSrcStep, 2 * oSizeROI.width) || !step_ok(nDstStep, dstPp * oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  if (w == 0 || h == 0) {
    return NPP_NO_ERROR;
  }
  k_invC2<<<gridFor(w, h), dim3(kBlockX, kBlockY), 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, dstPp, w, h, nAval);
  return launch_ok();
}

NppStatus yCrCb422ToRgbC2(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int dstPp, NppiSize oSizeROI,
                          cudaStream_t stream) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nSrcStep, 2 * oSizeROI.width) || !step_ok(nDstStep, dstPp * oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  if (w == 0 || h == 0) {
    return NPP_NO_ERROR;
  }
  k_ycrcb422_to_rgb_c2<<<gridFor(w, h), dim3(kBlockX, kBlockY), 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, dstPp, w,
                                                                            h);
  return launch_ok();
}

NppStatus yCrCb422ToRgbP3(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int nDstStep, NppiSize oSizeROI,
                          cudaStream_t stream) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nSrcStep, 2 * oSizeROI.width) || !step_ok(nDstStep, oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  if (w == 0 || h == 0) {
    return NPP_NO_ERROR;
  }
  k_ycrcb422_to_rgb_p3<<<gridFor(w, h), dim3(kBlockX, kBlockY), 0, stream>>>(pSrc, nSrcStep, pDst[0], pDst[1], pDst[2],
                                                                          nDstStep, w, h);
  return launch_ok();
}

NppStatus rgbToYcc(const Npp8u *pSrc, int nSrcStep, int pp, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                   cudaStream_t stream, int bgr) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nSrcStep, pp * oSizeROI.width) || !step_ok(nDstStep, pp * oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  if (ac4_src_step_bad(nSrcStep, pp)) {
    return NPP_NOT_EVEN_STEP_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  if (w == 0 || h == 0) {
    return NPP_NO_ERROR;
  }
  if (pp == 4) {
    if (bgr) {
      k_rgb_to_ycc<4, 1><<<gridFor(w, h), dim3(kBlockX, kBlockY), 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, w, h);
    } else {
      k_rgb_to_ycc<4, 0><<<gridFor(w, h), dim3(kBlockX, kBlockY), 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, w, h);
    }
  } else {
    if (bgr) {
      k_rgb_to_ycc<3, 1><<<gridFor(w, h), dim3(kBlockX, kBlockY), 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, w, h);
    } else {
      k_rgb_to_ycc<3, 0><<<gridFor(w, h), dim3(kBlockX, kBlockY), 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, w, h);
    }
  }
  return launch_ok();
}

NppStatus yccToRgb(const Npp8u *pSrc, int nSrcStep, int pp, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                   cudaStream_t stream) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nSrcStep, pp * oSizeROI.width) || !step_ok(nDstStep, pp * oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  if (ac4_src_step_bad(nSrcStep, pp)) {
    return NPP_NOT_EVEN_STEP_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  if (w == 0 || h == 0) {
    return NPP_NO_ERROR;
  }
  if (pp == 4) {
    k_ycc_to_rgb<4><<<gridFor(w, h), dim3(kBlockX, kBlockY), 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, w, h);
  } else {
    k_ycc_to_rgb<3><<<gridFor(w, h), dim3(kBlockX, kBlockY), 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, w, h);
  }
  return launch_ok();
}

NppStatus cmykToRgb(const Npp8u *const pSrc[4], int nSrcStep, Npp8u *pDst, int nDstStep, int dstPp, int bgr,
                    NppiSize oSizeROI, cudaStream_t stream) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pSrc[2] || !pSrc[3] || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nSrcStep, oSizeROI.width) || !step_ok(nDstStep, dstPp * oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  if (w == 0 || h == 0) {
    return NPP_NO_ERROR;
  }
  if (bgr) {
    k_cmyk_to_packed<1, 3><<<gridFor(w, h), dim3(kBlockX, kBlockY), 0, stream>>>(
        pSrc[0], pSrc[1], pSrc[2], pSrc[3], nSrcStep, pDst, nDstStep, w, h);
  } else {
    k_cmyk_to_packed<0, 3><<<gridFor(w, h), dim3(kBlockX, kBlockY), 0, stream>>>(
        pSrc[0], pSrc[1], pSrc[2], pSrc[3], nSrcStep, pDst, nDstStep, w, h);
  }
  return launch_ok();
}

NppStatus cmykToRgbPlanar(const Npp8u *const pSrc[4], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                          NppiSize oSizeROI, cudaStream_t stream, int bgr) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pSrc[2] || !pSrc[3] || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nSrcStep, oSizeROI.width) || !step_ok(nDstStep, oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  if (w == 0 || h == 0) {
    return NPP_NO_ERROR;
  }
  Npp8u *pR = bgr ? pDst[2]: pDst[0];
  Npp8u *pG = pDst[1];
  Npp8u *pB = bgr ? pDst[0]: pDst[2];
  k_cmyk_to_planar<<<gridFor(w, h), dim3(kBlockX, kBlockY), 0, stream>>>(pSrc[0], pSrc[1], pSrc[2], pSrc[3], nSrcStep,
                                                                        pR, pG, pB, nDstStep, w, h);
  return launch_ok();
}

NppStatus ycckToCmyk(const Npp8u *const pSrc[4], int nSrcStep, Npp8u *pDst[4], int nDstStep, NppiSize oSizeROI,
                     cudaStream_t stream) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nSrcStep, oSizeROI.width) || !step_ok(nDstStep, oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  if (w == 0 || h == 0) {
    return NPP_NO_ERROR;
  }
  k_ycck_to_cmyk<<<gridFor(w, h), dim3(kBlockX, kBlockY), 0, stream>>>(pSrc[0], pSrc[1], pSrc[2], pSrc[3], nSrcStep,
                                                                      pDst[0], pDst[1], pDst[2], pDst[3], nDstStep, w,
                                                                      h);
  return launch_ok();
}

} // namespace c3color
} // namespace nppi
