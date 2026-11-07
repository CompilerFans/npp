// c3 ColorTwist kernels (see nppi_c3_twist_executor.h).

#include "nppi_c3_twist_executor.h"

#include <cuda_runtime.h>

namespace nppi {
namespace c3twist {
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

struct Twist {
  float m[3][4];
};

__host__ __device__ __forceinline__ Twist loadTwist(const Npp32f a[3][4]) {
  Twist t;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      t.m[i][j] = a[i][j];
    }
  }
  return t;
}

__device__ __forceinline__ int round_clamp(float v, int maxv) {
  int t = static_cast<int>(v + 0.5f);
  return t < 0 ? 0: (t > maxv ? maxv: t);
}

__device__ __forceinline__ float twistRow(const Twist &t, int row, float a, float b, float c) {
  return t.m[row][0] * a + t.m[row][1] * b + t.m[row][2] * c + t.m[row][3];
}

// Packed C3 source helper.
template <typename T>
__device__ __forceinline__ void read3(const T *s, float &a, float &b, float &c) {
  a = static_cast<float>(s[0]);
  b = static_cast<float>(s[1]);
  c = static_cast<float>(s[2]);
}

// --- forward 4:2:0, packed source ---
template <typename T, int MAXV>
__global__ void k_fwd420_c3(const T *__restrict__ pSrc, int nSrcStep, T *__restrict__ pDst0, int s0,
                            T *__restrict__ pDst1, int s1, T *__restrict__ pDst2, int s2, int w, int h, Twist t) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= ((w & ~1) >> 1) || j >= ((h & ~1) >> 1)) {
    return;
  }
  float v[3][4];
  for (int dy = 0; dy < 2; ++dy) {
    for (int dx = 0; dx < 2; ++dx) {
      const T *s = pSrc + (size_t)(2 * j + dy) * nSrcStep + (size_t)(2 * x + dx) * 3;
      float a, b, c;
      read3(s, a, b, c);
      for (int r = 0; r < 3; ++r) {
        v[r][dy * 2 + dx] = twistRow(t, r, a, b, c);
      }
    }
  }
  // Luma is per pixel; chroma is the rounded 2x2 average.
  for (int dy = 0; dy < 2; ++dy) {
    for (int dx = 0; dx < 2; ++dx) {
      pDst0[(size_t)(2 * j + dy) * s0 + (2 * x + dx)] = static_cast<T>(round_clamp(v[0][dy * 2 + dx], MAXV));
    }
  }
  for (int r = 1; r < 3; ++r) {
    const float avg = (v[r][0] + v[r][1] + v[r][2] + v[r][3]) * 0.25f;
    T *dst = r == 1 ? pDst1: pDst2;
    const int st = r == 1 ? s1: s2;
    dst[(size_t)j * st + x] = static_cast<T>(round_clamp(avg, MAXV));
  }
}

// --- forward 4:2:0, planar source ---
template <typename T, int MAXV>
__global__ void k_fwd420_p3(const T *__restrict__ p0, const T *__restrict__ p1, const T *__restrict__ p2,
                            int nSrcStep0, int nSrcStep1, int nSrcStep2, T *__restrict__ pDst0, int s0,
                            T *__restrict__ pDst1, int s1, T *__restrict__ pDst2, int s2, int w, int h, Twist t) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= ((w & ~1) >> 1) || j >= ((h & ~1) >> 1)) {
    return;
  }
  float v[3][4];
  for (int dy = 0; dy < 2; ++dy) {
    for (int dx = 0; dx < 2; ++dx) {
      const int px = 2 * x + dx;
      const int py = 2 * j + dy;
      const float a = static_cast<float>(p0[(size_t)py * nSrcStep0 + px]);
      const float b = static_cast<float>(p1[(size_t)py * nSrcStep1 + px]);
      const float c = static_cast<float>(p2[(size_t)py * nSrcStep2 + px]);
      for (int r = 0; r < 3; ++r) {
        v[r][dy * 2 + dx] = twistRow(t, r, a, b, c);
      }
    }
  }
  for (int dy = 0; dy < 2; ++dy) {
    for (int dx = 0; dx < 2; ++dx) {
      pDst0[(size_t)(2 * j + dy) * s0 + (2 * x + dx)] = static_cast<T>(round_clamp(v[0][dy * 2 + dx], MAXV));
    }
  }
  T *dst[3] = {pDst1, pDst2};
  const int st[2] = {s1, s2};
  for (int r = 1; r < 3; ++r) {
    const float avg = (v[r][0] + v[r][1] + v[r][2] + v[r][3]) * 0.25f;
    dst[r - 1][(size_t)j * st[r - 1] + x] = static_cast<T>(round_clamp(avg, MAXV));
  }
}

// --- forward 4:2:2, packed C3 -> packed C2 [U Y0 V Y1] ---
template <typename T, int MAXV>
__global__ void k_fwd422_c2(const T *__restrict__ pSrc, int nSrcStep, T *__restrict__ pDst, int nDstStep, int w, int h,
                            Twist t) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (p >= (w >> 1) || y >= h) {
    return;
  }
  float v[3][2];
  for (int dx = 0; dx < 2; ++dx) {
    const T *s = pSrc + (size_t)y * nSrcStep + (size_t)(2 * p + dx) * 3;
    float a, b, c;
    read3(s, a, b, c);
    for (int r = 0; r < 3; ++r) {
      v[r][dx] = twistRow(t, r, a, b, c);
    }
  }
  T *d = pDst + (size_t)y * nDstStep + (size_t)p * 4;
  d[0] = static_cast<T>(round_clamp((v[1][0] + v[1][1]) * 0.5f, MAXV)); // U
  d[1] = static_cast<T>(round_clamp(v[0][0], MAXV));                    // Y0
  d[2] = static_cast<T>(round_clamp((v[2][0] + v[2][1]) * 0.5f, MAXV)); // V
  d[3] = static_cast<T>(round_clamp(v[0][1], MAXV));                    // Y1
}

// --- forward 4:2:2 to planar destination ---
template <typename T, int MAXV, int PACKED>
__global__ void k_fwd422_p3(const T *__restrict__ pSrc0, const T *__restrict__ pSrc1, const T *__restrict__ pSrc2,
                            int nSrcStep0, int nSrcStep1, int nSrcStep2, T *__restrict__ pDst0, int s0,
                            T *__restrict__ pDst1, int s1, T *__restrict__ pDst2, int s2, int w, int h, Twist t) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= (w & ~1)) {
    return;
  }
  if (y >= h) {
    return;
  }
  float a, b, c;
  if (PACKED) {
    read3(pSrc0 + (size_t)y * nSrcStep0 + (size_t)x * 3, a, b, c);
  } else {
    a = static_cast<float>(pSrc0[(size_t)y * nSrcStep0 + x]);
    b = static_cast<float>(pSrc1[(size_t)y * nSrcStep1 + x]);
    c = static_cast<float>(pSrc2[(size_t)y * nSrcStep2 + x]);
  }
  pDst0[(size_t)y * s0 + x] = static_cast<T>(round_clamp(twistRow(t, 0, a, b, c), MAXV));
  if (!(x & 1)) {
    const int other = x + 1;
    float a2, b2, c2;
    if (PACKED) {
      read3(pSrc0 + (size_t)y * nSrcStep0 + (size_t)other * 3, a2, b2, c2);
    } else {
      a2 = static_cast<float>(pSrc0[(size_t)y * nSrcStep0 + other]);
      b2 = static_cast<float>(pSrc1[(size_t)y * nSrcStep1 + other]);
      c2 = static_cast<float>(pSrc2[(size_t)y * nSrcStep2 + other]);
    }
    const float fu = 0.5f * (twistRow(t, 1, a, b, c) + twistRow(t, 1, a2, b2, c2));
    const float fv = 0.5f * (twistRow(t, 2, a, b, c) + twistRow(t, 2, a2, b2, c2));
    pDst1[(size_t)y * s1 + (x >> 1)] = static_cast<T>(round_clamp(fu, MAXV));
    pDst2[(size_t)y * s2 + (x >> 1)] = static_cast<T>(round_clamp(fv, MAXV));
  }
}

// --- inverse 4:2:0 ---
template <typename T, int MAXV>
__device__ __forceinline__ void inv420_pixel(const T *__restrict__ pY, int ys, const T *__restrict__ pU, int us,
                                             const T *__restrict__ pV, int vs, int x, int y, Twist t, int &o0,
                                             int &o1, int &o2) {
  const float fy = static_cast<float>(pY[(size_t)y * ys + x]);
  const float fu = static_cast<float>(pU[(size_t)(y >> 1) * us + (x >> 1)]);
  const float fv = static_cast<float>(pV[(size_t)(y >> 1) * vs + (x >> 1)]);
  o0 = round_clamp(twistRow(t, 0, fy, fu, fv), MAXV);
  o1 = round_clamp(twistRow(t, 1, fy, fu, fv), MAXV);
  o2 = round_clamp(twistRow(t, 2, fy, fu, fv), MAXV);
}

template <typename T, int MAXV, int PP>
__global__ void k_inv420_c(const T *__restrict__ pY, int ys, const T *__restrict__ pU, int us,
                           const T *__restrict__ pV, int vs, T *__restrict__ pDst, int nDstStep, T alpha, int w,
                           int h, Twist t) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) {
    return;
  }
  int o0, o1, o2;
  inv420_pixel<T, MAXV>(pY, ys, pU, us, pV, vs, x, y, t, o0, o1, o2);
  T *d = pDst + (size_t)y * nDstStep + (size_t)x * PP;
  d[0] = static_cast<T>(o0);
  d[1] = static_cast<T>(o1);
  d[2] = static_cast<T>(o2);
  if (PP == 4) {
    d[3] = alpha;
  }
}

template <typename T, int MAXV>
__global__ void k_inv420_p3(const T *__restrict__ pY, int ys, const T *__restrict__ pU, int us,
                            const T *__restrict__ pV, int vs, T *__restrict__ pDst0, int s0, T *__restrict__ pDst1,
                            int s1, T *__restrict__ pDst2, int s2, int w, int h, Twist t) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) {
    return;
  }
  int o0, o1, o2;
  inv420_pixel<T, MAXV>(pY, ys, pU, us, pV, vs, x, y, t, o0, o1, o2);
  pDst0[(size_t)y * s0 + x] = static_cast<T>(o0);
  pDst1[(size_t)y * s1 + x] = static_cast<T>(o1);
  pDst2[(size_t)y * s2 + x] = static_cast<T>(o2);
}

// --- inverse 4:2:2 ---
template <typename T, int MAXV>
__device__ __forceinline__ void inv422_pixel(const T *__restrict__ pY, int ys, const T *__restrict__ pU, int us,
                                             const T *__restrict__ pV, int vs, int x, int y, Twist t, int &o0,
                                             int &o1, int &o2) {
  const float fy = static_cast<float>(pY[(size_t)y * ys + x]);
  const float fu = static_cast<float>(pU[(size_t)y * us + (x >> 1)]);
  const float fv = static_cast<float>(pV[(size_t)y * vs + (x >> 1)]);
  o0 = round_clamp(twistRow(t, 0, fy, fu, fv), MAXV);
  o1 = round_clamp(twistRow(t, 1, fy, fu, fv), MAXV);
  o2 = round_clamp(twistRow(t, 2, fy, fu, fv), MAXV);
}

template <typename T, int MAXV, int PP>
__global__ void k_inv422_p3c(const T *__restrict__ pY, int ys, const T *__restrict__ pU, int us,
                             const T *__restrict__ pV, int vs, T *__restrict__ pDst, int nDstStep, T alpha, int w,
                             int h, Twist t) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) {
    return;
  }
  int o0, o1, o2;
  inv422_pixel<T, MAXV>(pY, ys, pU, us, pV, vs, x, y, t, o0, o1, o2);
  T *d = pDst + (size_t)y * nDstStep + (size_t)x * PP;
  d[0] = static_cast<T>(o0);
  d[1] = static_cast<T>(o1);
  d[2] = static_cast<T>(o2);
  if (PP == 4) {
    d[3] = alpha;
  }
}

template <typename T, int MAXV>
__global__ void k_inv422_p3(const T *__restrict__ pY, int ys, const T *__restrict__ pU, int us,
                            const T *__restrict__ pV, int vs, T *__restrict__ pDst0, int s0, T *__restrict__ pDst1,
                            int s1, T *__restrict__ pDst2, int s2, int w, int h, Twist t) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) {
    return;
  }
  int o0, o1, o2;
  inv422_pixel<T, MAXV>(pY, ys, pU, us, pV, vs, x, y, t, o0, o1, o2);
  pDst0[(size_t)y * s0 + x] = static_cast<T>(o0);
  pDst1[(size_t)y * s1 + x] = static_cast<T>(o1);
  pDst2[(size_t)y * s2 + x] = static_cast<T>(o2);
}

// Packed YUV422 [U Y0 V Y1] -> packed RGB.
template <typename T, int MAXV>
__global__ void k_inv422_c2(const T *__restrict__ pSrc, int nSrcStep, T *__restrict__ pDst, int nDstStep, int w, int h,
                            Twist t) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) {
    return;
  }
  int p = x >> 1;
  const int lastPair = (w >> 1) - 1;
  if (p > lastPair) {
    p = lastPair;
  }
  const T *s = pSrc + (size_t)y * nSrcStep + (size_t)p * 4;
  const float fy = static_cast<float>(s[(x & 1) ? 3: 1]);
  const float fu = static_cast<float>(s[0]);
  const float fv = static_cast<float>(s[2]);
  T *d = pDst + (size_t)y * nDstStep + (size_t)x * 3;
  d[0] = static_cast<T>(round_clamp(twistRow(t, 0, fy, fu, fv), MAXV));
  d[1] = static_cast<T>(round_clamp(twistRow(t, 1, fy, fu, fv), MAXV));
  d[2] = static_cast<T>(round_clamp(twistRow(t, 2, fy, fu, fv), MAXV));
}

} // namespace

NppStatus fwd420(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], const int aDstStep[3], NppiSize oSizeROI,
                 cudaStream_t stream, const Npp32f aTwist[3][4]) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nSrcStep, 3 * oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  if (w == 0 || h == 0) {
    return NPP_NO_ERROR;
  }
  const Twist t = loadTwist(aTwist);
  if ((w >> 1) > 0 && (h >> 1) > 0) {
    k_fwd420_c3<Npp8u, 255><<<gridFor(w >> 1, h >> 1), dim3(kBlockX, kBlockY), 0, stream>>>(
        pSrc, nSrcStep, pDst[0], aDstStep[0], pDst[1], aDstStep[1], pDst[2], aDstStep[2], w, h, t);
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

NppStatus fwd420P3(const Npp8u *const pSrc[3], const int aSrcStep[3], Npp8u *pDst[3], const int aDstStep[3],
                   NppiSize oSizeROI, cudaStream_t stream, const Npp32f aTwist[3][4]) {
  if (!pSrc || !pSrc[0] || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(aSrcStep[0], oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  if (w == 0 || h == 0) {
    return NPP_NO_ERROR;
  }
  const Twist t = loadTwist(aTwist);
  if ((w >> 1) > 0 && (h >> 1) > 0) {
    k_fwd420_p3<Npp8u, 255><<<gridFor(w >> 1, h >> 1), dim3(kBlockX, kBlockY), 0, stream>>>(
        pSrc[0], pSrc[1], pSrc[2], aSrcStep[0], aSrcStep[1], aSrcStep[2], pDst[0], aDstStep[0], pDst[1], aDstStep[1],
        pDst[2], aDstStep[2], w, h, t);
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

NppStatus fwd420_16u(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst[3], const int aDstStep[3], NppiSize oSizeROI,
                     cudaStream_t stream, const Npp32f aTwist[3][4]) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nSrcStep, 3 * oSizeROI.width * 2)) {
    return NPP_STEP_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  if (w == 0 || h == 0) {
    return NPP_NO_ERROR;
  }
  const Twist t = loadTwist(aTwist);
  if ((w >> 1) > 0 && (h >> 1) > 0) {
    k_fwd420_c3<Npp16u, 65535><<<gridFor(w >> 1, h >> 1), dim3(kBlockX, kBlockY), 0, stream>>>(
        pSrc, nSrcStep / 2, pDst[0], aDstStep[0] / 2, pDst[1], aDstStep[1] / 2, pDst[2], aDstStep[2] / 2, w, h, t);
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

NppStatus fwd420P3_16u(const Npp16u *const pSrc[3], const int aSrcStep[3], Npp16u *pDst[3], const int aDstStep[3],
                       NppiSize oSizeROI, cudaStream_t stream, const Npp32f aTwist[3][4]) {
  if (!pSrc || !pSrc[0] || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(aSrcStep[0], oSizeROI.width * 2)) {
    return NPP_STEP_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  if (w == 0 || h == 0) {
    return NPP_NO_ERROR;
  }
  const Twist t = loadTwist(aTwist);
  if ((w >> 1) > 0 && (h >> 1) > 0) {
    k_fwd420_p3<Npp16u, 65535><<<gridFor(w >> 1, h >> 1), dim3(kBlockX, kBlockY), 0, stream>>>(
        pSrc[0], pSrc[1], pSrc[2], aSrcStep[0] / 2, aSrcStep[1] / 2, aSrcStep[2] / 2, pDst[0], aDstStep[0] / 2, pDst[1],
        aDstStep[1] / 2, pDst[2], aDstStep[2] / 2, w, h, t);
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

NppStatus fwd422C2(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                   cudaStream_t stream, const Npp32f aTwist[3][4]) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nSrcStep, 3 * oSizeROI.width) || !step_ok(nDstStep, 2 * oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  if (w == 0 || h == 0) {
    return NPP_NO_ERROR;
  }
  const Twist t = loadTwist(aTwist);
  if ((w >> 1) > 0) {
    k_fwd422_c2<Npp8u, 255><<<gridFor(w >> 1, h), dim3(kBlockX, kBlockY), 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                                      w, h, t);
  }
  NppStatus st = launch_ok();
  if (st != NPP_NO_ERROR) {
    return st;
  }
  if (w & 1) {
    return NPP_DOUBLE_SIZE_WARNING;
  }
  return NPP_NO_ERROR;
}

NppStatus fwd422C2_16u(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                       cudaStream_t stream, const Npp32f aTwist[3][4]) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nSrcStep, 3 * oSizeROI.width * 2) || !step_ok(nDstStep, 2 * oSizeROI.width * 2)) {
    return NPP_STEP_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  if (w == 0 || h == 0) {
    return NPP_NO_ERROR;
  }
  const Twist t = loadTwist(aTwist);
  if ((w >> 1) > 0) {
    k_fwd422_c2<Npp16u, 65535><<<gridFor(w >> 1, h), dim3(kBlockX, kBlockY), 0, stream>>>(
        pSrc, nSrcStep / 2, pDst, nDstStep / 2, w, h, t);
  }
  NppStatus st = launch_ok();
  if (st != NPP_NO_ERROR) {
    return st;
  }
  if (w & 1) {
    return NPP_DOUBLE_SIZE_WARNING;
  }
  return NPP_NO_ERROR;
}

NppStatus fwd422C3P3(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], const int aDstStep[3], NppiSize oSizeROI,
                     cudaStream_t stream, const Npp32f aTwist[3][4]) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nSrcStep, 3 * oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  if (w == 0 || h == 0) {
    return NPP_NO_ERROR;
  }
  const Twist t = loadTwist(aTwist);
  k_fwd422_p3<Npp8u, 255, 1><<<gridFor(w, h), dim3(kBlockX, kBlockY), 0, stream>>>(
      pSrc, nullptr, nullptr, nSrcStep, 0, 0, pDst[0], aDstStep[0], pDst[1], aDstStep[1], pDst[2], aDstStep[2], w, h,
      t);
  NppStatus st = launch_ok();
  if (st != NPP_NO_ERROR) {
    return st;
  }
  if (w & 1) {
    return NPP_DOUBLE_SIZE_WARNING;
  }
  return NPP_NO_ERROR;
}

NppStatus fwd422C3P3_16u(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst[3], const int aDstStep[3], NppiSize oSizeROI,
                         cudaStream_t stream, const Npp32f aTwist[3][4]) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nSrcStep, 3 * oSizeROI.width * 2)) {
    return NPP_STEP_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  if (w == 0 || h == 0) {
    return NPP_NO_ERROR;
  }
  const Twist t = loadTwist(aTwist);
  k_fwd422_p3<Npp16u, 65535, 1><<<gridFor(w, h), dim3(kBlockX, kBlockY), 0, stream>>>(
      pSrc, nullptr, nullptr, nSrcStep / 2, 0, 0, pDst[0], aDstStep[0] / 2, pDst[1], aDstStep[1] / 2, pDst[2],
      aDstStep[2] / 2, w, h, t);
  NppStatus st = launch_ok();
  if (st != NPP_NO_ERROR) {
    return st;
  }
  if (w & 1) {
    return NPP_DOUBLE_SIZE_WARNING;
  }
  return NPP_NO_ERROR;
}

NppStatus fwd422P3(const Npp8u *const pSrc[3], const int aSrcStep[3], Npp8u *pDst[3], const int aDstStep[3],
                   NppiSize oSizeROI, cudaStream_t stream, const Npp32f aTwist[3][4]) {
  if (!pSrc || !pSrc[0] || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(aSrcStep[0], oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  if (w == 0 || h == 0) {
    return NPP_NO_ERROR;
  }
  const Twist t = loadTwist(aTwist);
  k_fwd422_p3<Npp8u, 255, 0><<<gridFor(w, h), dim3(kBlockX, kBlockY), 0, stream>>>(
      pSrc[0], pSrc[1], pSrc[2], aSrcStep[0], aSrcStep[1], aSrcStep[2], pDst[0], aDstStep[0], pDst[1], aDstStep[1],
      pDst[2], aDstStep[2], w, h, t);
  NppStatus st = launch_ok();
  if (st != NPP_NO_ERROR) {
    return st;
  }
  if (w & 1) {
    return NPP_DOUBLE_SIZE_WARNING;
  }
  return NPP_NO_ERROR;
}

NppStatus fwd422P3_16u(const Npp16u *const pSrc[3], const int aSrcStep[3], Npp16u *pDst[3], const int aDstStep[3],
                       NppiSize oSizeROI, cudaStream_t stream, const Npp32f aTwist[3][4]) {
  if (!pSrc || !pSrc[0] || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(aSrcStep[0], oSizeROI.width * 2)) {
    return NPP_STEP_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  if (w == 0 || h == 0) {
    return NPP_NO_ERROR;
  }
  const Twist t = loadTwist(aTwist);
  k_fwd422_p3<Npp16u, 65535, 0><<<gridFor(w, h), dim3(kBlockX, kBlockY), 0, stream>>>(
      pSrc[0], pSrc[1], pSrc[2], aSrcStep[0] / 2, aSrcStep[1] / 2, aSrcStep[2] / 2, pDst[0], aDstStep[0] / 2, pDst[1],
      aDstStep[1] / 2, pDst[2], aDstStep[2] / 2, w, h, t);
  NppStatus st = launch_ok();
  if (st != NPP_NO_ERROR) {
    return st;
  }
  if (w & 1) {
    return NPP_DOUBLE_SIZE_WARNING;
  }
  return NPP_NO_ERROR;
}

NppStatus inv420(const Npp8u *const pSrc[3], const int aSrcStep[3], Npp8u *pDst, int nDstStep, int dstPp, Npp8u alpha,
                 NppiSize oSizeROI, cudaStream_t stream, const Npp32f aTwist[3][4]) {
  if (!pSrc || !pSrc[0] || !pDst) {
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
  const Twist t = loadTwist(aTwist);
  if (dstPp == 4) {
    k_inv420_c<Npp8u, 255, 4><<<gridFor(w, h), dim3(kBlockX, kBlockY), 0, stream>>>(
        pSrc[0], aSrcStep[0], pSrc[1], aSrcStep[1], pSrc[2], aSrcStep[2], pDst, nDstStep, alpha, w, h, t);
  } else {
    k_inv420_c<Npp8u, 255, 3><<<gridFor(w, h), dim3(kBlockX, kBlockY), 0, stream>>>(
        pSrc[0], aSrcStep[0], pSrc[1], aSrcStep[1], pSrc[2], aSrcStep[2], pDst, nDstStep, alpha, w, h, t);
  }
  return launch_ok();
}

NppStatus inv420P3(const Npp8u *const pSrc[3], const int aSrcStep[3], Npp8u *pDst[3], const int aDstStep[3],
                   NppiSize oSizeROI, cudaStream_t stream, const Npp32f aTwist[3][4]) {
  if (!pSrc || !pSrc[0] || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(aSrcStep[0], oSizeROI.width) || !step_ok(aDstStep[0], oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  if (w == 0 || h == 0) {
    return NPP_NO_ERROR;
  }
  const Twist t = loadTwist(aTwist);
  k_inv420_p3<Npp8u, 255><<<gridFor(w, h), dim3(kBlockX, kBlockY), 0, stream>>>(
      pSrc[0], aSrcStep[0], pSrc[1], aSrcStep[1], pSrc[2], aSrcStep[2], pDst[0], aDstStep[0], pDst[1], aDstStep[1],
      pDst[2], aDstStep[2], w, h, t);
  return launch_ok();
}

NppStatus inv420_16u(const Npp16u *const pSrc[3], const int aSrcStep[3], Npp16u *pDst, int nDstStep, int dstPp,
                     Npp16u alpha, NppiSize oSizeROI, cudaStream_t stream, const Npp32f aTwist[3][4]) {
  if (!pSrc || !pSrc[0] || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(aSrcStep[0], oSizeROI.width * 2) || !step_ok(nDstStep, dstPp * oSizeROI.width * 2)) {
    return NPP_STEP_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  if (w == 0 || h == 0) {
    return NPP_NO_ERROR;
  }
  const Twist t = loadTwist(aTwist);
  if (dstPp == 4) {
    k_inv420_c<Npp16u, 65535, 4><<<gridFor(w, h), dim3(kBlockX, kBlockY), 0, stream>>>(
        pSrc[0], aSrcStep[0] / 2, pSrc[1], aSrcStep[1] / 2, pSrc[2], aSrcStep[2] / 2, pDst, nDstStep / 2, alpha, w, h,
        t);
  } else {
    k_inv420_c<Npp16u, 65535, 3><<<gridFor(w, h), dim3(kBlockX, kBlockY), 0, stream>>>(
        pSrc[0], aSrcStep[0] / 2, pSrc[1], aSrcStep[1] / 2, pSrc[2], aSrcStep[2] / 2, pDst, nDstStep / 2, alpha, w, h,
        t);
  }
  return launch_ok();
}

NppStatus inv420P3_16u(const Npp16u *const pSrc[3], const int aSrcStep[3], Npp16u *pDst[3], const int aDstStep[3],
                       NppiSize oSizeROI, cudaStream_t stream, const Npp32f aTwist[3][4]) {
  if (!pSrc || !pSrc[0] || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(aSrcStep[0], oSizeROI.width * 2) || !step_ok(aDstStep[0], oSizeROI.width * 2)) {
    return NPP_STEP_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  if (w == 0 || h == 0) {
    return NPP_NO_ERROR;
  }
  const Twist t = loadTwist(aTwist);
  k_inv420_p3<Npp16u, 65535><<<gridFor(w, h), dim3(kBlockX, kBlockY), 0, stream>>>(
      pSrc[0], aSrcStep[0] / 2, pSrc[1], aSrcStep[1] / 2, pSrc[2], aSrcStep[2] / 2, pDst[0], aDstStep[0] / 2, pDst[1],
      aDstStep[1] / 2, pDst[2], aDstStep[2] / 2, w, h, t);
  return launch_ok();
}

NppStatus inv422P3(const Npp8u *const pSrc[3], const int aSrcStep[3], Npp8u *pDst[3], const int aDstStep[3],
                   NppiSize oSizeROI, cudaStream_t stream, const Npp32f aTwist[3][4]) {
  if (!pSrc || !pSrc[0] || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(aSrcStep[0], oSizeROI.width) || !step_ok(aDstStep[0], oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  if (w == 0 || h == 0) {
    return NPP_NO_ERROR;
  }
  const Twist t = loadTwist(aTwist);
  k_inv422_p3<Npp8u, 255><<<gridFor(w, h), dim3(kBlockX, kBlockY), 0, stream>>>(
      pSrc[0], aSrcStep[0], pSrc[1], aSrcStep[1], pSrc[2], aSrcStep[2], pDst[0], aDstStep[0], pDst[1], aDstStep[1],
      pDst[2], aDstStep[2], w, h, t);
  return launch_ok();
}

NppStatus inv422P3C(const Npp8u *const pSrc[3], const int aSrcStep[3], Npp8u *pDst, int nDstStep, int dstPp,
                    Npp8u alpha, NppiSize oSizeROI, cudaStream_t stream, const Npp32f aTwist[3][4]) {
  if (!pSrc || !pSrc[0] || !pDst) {
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
  const Twist t = loadTwist(aTwist);
  if (dstPp == 4) {
    k_inv422_p3c<Npp8u, 255, 4><<<gridFor(w, h), dim3(kBlockX, kBlockY), 0, stream>>>(
        pSrc[0], aSrcStep[0], pSrc[1], aSrcStep[1], pSrc[2], aSrcStep[2], pDst, nDstStep, alpha, w, h, t);
  } else {
    k_inv422_p3c<Npp8u, 255, 3><<<gridFor(w, h), dim3(kBlockX, kBlockY), 0, stream>>>(
        pSrc[0], aSrcStep[0], pSrc[1], aSrcStep[1], pSrc[2], aSrcStep[2], pDst, nDstStep, alpha, w, h, t);
  }
  return launch_ok();
}

NppStatus inv422(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, cudaStream_t stream,
                 const Npp32f aTwist[3][4]) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nSrcStep, 2 * oSizeROI.width) || !step_ok(nDstStep, 3 * oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  if (w == 0 || h == 0) {
    return NPP_NO_ERROR;
  }
  const Twist t = loadTwist(aTwist);
  k_inv422_c2<Npp8u, 255><<<gridFor(w, h), dim3(kBlockX, kBlockY), 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, w, h, t);
  return launch_ok();
}

NppStatus inv422_16u(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                     cudaStream_t stream, const Npp32f aTwist[3][4]) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nSrcStep, 2 * oSizeROI.width * 2) || !step_ok(nDstStep, 3 * oSizeROI.width * 2)) {
    return NPP_STEP_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  if (w == 0 || h == 0) {
    return NPP_NO_ERROR;
  }
  const Twist t = loadTwist(aTwist);
  k_inv422_c2<Npp16u, 65535><<<gridFor(w, h), dim3(kBlockX, kBlockY), 0, stream>>>(
      pSrc, nSrcStep / 2, pDst, nDstStep / 2, w, h, t);
  return launch_ok();
}

NppStatus inv422P3_16u(const Npp16u *const pSrc[3], const int aSrcStep[3], Npp16u *pDst[3], const int aDstStep[3],
                       NppiSize oSizeROI, cudaStream_t stream, const Npp32f aTwist[3][4]) {
  if (!pSrc || !pSrc[0] || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(aSrcStep[0], oSizeROI.width * 2) || !step_ok(aDstStep[0], oSizeROI.width * 2)) {
    return NPP_STEP_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  if (w == 0 || h == 0) {
    return NPP_NO_ERROR;
  }
  const Twist t = loadTwist(aTwist);
  k_inv422_p3<Npp16u, 65535><<<gridFor(w, h), dim3(kBlockX, kBlockY), 0, stream>>>(
      pSrc[0], aSrcStep[0] / 2, pSrc[1], aSrcStep[1] / 2, pSrc[2], aSrcStep[2] / 2, pDst[0], aDstStep[0] / 2, pDst[1],
      aDstStep[1] / 2, pDst[2], aDstStep[2] / 2, w, h, t);
  return launch_ok();
}

NppStatus inv422P3C_16u(const Npp16u *const pSrc[3], const int aSrcStep[3], Npp16u *pDst, int nDstStep, int dstPp,
                        Npp16u alpha, NppiSize oSizeROI, cudaStream_t stream, const Npp32f aTwist[3][4]) {
  if (!pSrc || !pSrc[0] || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(aSrcStep[0], oSizeROI.width * 2) || !step_ok(nDstStep, dstPp * oSizeROI.width * 2)) {
    return NPP_STEP_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  if (w == 0 || h == 0) {
    return NPP_NO_ERROR;
  }
  const Twist t = loadTwist(aTwist);
  if (dstPp == 4) {
    k_inv422_p3c<Npp16u, 65535, 4><<<gridFor(w, h), dim3(kBlockX, kBlockY), 0, stream>>>(
        pSrc[0], aSrcStep[0] / 2, pSrc[1], aSrcStep[1] / 2, pSrc[2], aSrcStep[2] / 2, pDst, nDstStep / 2, alpha, w, h,
        t);
  } else {
    k_inv422_p3c<Npp16u, 65535, 3><<<gridFor(w, h), dim3(kBlockX, kBlockY), 0, stream>>>(
        pSrc[0], aSrcStep[0] / 2, pSrc[1], aSrcStep[1] / 2, pSrc[2], aSrcStep[2] / 2, pDst, nDstStep / 2, alpha, w, h,
        t);
  }
  return launch_ok();
}

} // namespace c3twist
} // namespace nppi
