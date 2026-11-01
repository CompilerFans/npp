// WarpAffine family executor: kernels + explicit instantiations (package g1).
// See nppi_warp_affine_executor.h for the semantic contract.

#include "nppi_warp_affine_executor.h"
#include "npp_kernel_utils.h"

#include <cmath>
#include <cuda_fp16.h>
#include <type_traits>

namespace nppi {
namespace geometry {
namespace {

// ============================================================================
// Device-side scalar helpers
// ============================================================================

// Working type of the interpolation arithmetic: single precision for every
// pixel type except 64f, which keeps double precision end to end.
template <typename T> struct WorkType { using type = float; };
template <> struct WorkType<Npp64f> { using type = double; };
template <typename T> using work_t = typename WorkType<T>::type;

__device__ __forceinline__ float halfAsFloat(Npp16f v) {
  return __half2float(__ushort_as_half(static_cast<unsigned short>(v.fp16)));
}

__device__ __forceinline__ Npp16f floatAsHalf(float v) {
  Npp16f out;
  out.fp16 = static_cast<short>(__half_as_ushort(__float2half(v)));
  return out;
}

// Device -> pixel conversion. Integer destinations saturate to the type
// range first and then round half away from zero; float destinations cast
// directly.  The 5090 SASS implements the rounding as FADD.RZ of a
// sign-copied 0.5 followed by F2I.RZ (__float2int_rz(__fadd_rz(v, +/-0.5))).
// That differs from floorf(v + 0.5f) twice: the RZ addition never crosses an
// integer boundary upwards (RN can, on values just below x.0 - 0.5), and
// negative halves round away from zero (floor(0.5 + -0.5) = 0 but
// trunc(-0.5 - 0.5) = -1, which the 32s goldens require).
template <typename T> __device__ __forceinline__ T castSample(work_t<T> v);

__device__ __forceinline__ int roundHalfAwayFromZero(float v) {
  return __float2int_rz(__fadd_rz(v, copysignf(0.5f, v)));
}

template <> __device__ __forceinline__ Npp8u castSample<Npp8u>(float v) {
  v = fminf(fmaxf(v, 0.0f), 255.0f);
  return static_cast<Npp8u>(roundHalfAwayFromZero(v));
}
template <> __device__ __forceinline__ Npp16u castSample<Npp16u>(float v) {
  v = fminf(fmaxf(v, 0.0f), 65535.0f);
  return static_cast<Npp16u>(roundHalfAwayFromZero(v));
}
template <> __device__ __forceinline__ Npp32s castSample<Npp32s>(float v) {
  v = fminf(fmaxf(v, -2147483648.0f), 2147483520.0f);
  return static_cast<Npp32s>(roundHalfAwayFromZero(v));
}
template <> __device__ __forceinline__ Npp16f castSample<Npp16f>(float v) { return floatAsHalf(v); }
template <> __device__ __forceinline__ Npp32f castSample<Npp32f>(float v) { return v; }
template <> __device__ __forceinline__ Npp64f castSample<Npp64f>(double v) { return v; }

__device__ __forceinline__ float pixelAsFloat(Npp8u v) { return static_cast<float>(v); }
__device__ __forceinline__ float pixelAsFloat(Npp16u v) { return static_cast<float>(v); }
__device__ __forceinline__ float pixelAsFloat(Npp32s v) { return static_cast<float>(v); }
__device__ __forceinline__ float pixelAsFloat(Npp32f v) { return v; }
__device__ __forceinline__ float pixelAsFloat(Npp16f v) { return halfAsFloat(v); }
__device__ __forceinline__ double pixelAsFloat(Npp64f v) { return v; }

// Catmull-Rom cubic interpolation weight (a = -0.5); used for the double
// working type (no 64f cubic goldens exist, the polynomial form is kept).
template <typename F> __device__ __forceinline__ F cubicWeight(F x) {
  x = fabs(x);
  if (x <= F(1)) {
    return ((F(1.5) * x - F(2.5)) * x) * x + F(1);
  }
  if (x < F(2)) {
    return (((-F(0.5) * x + F(2.5)) * x) - F(4)) * x + F(2);
  }
  return F(0);
}

// ---- Exact single-precision cubic weights (5090 NPP replay) ----
// The float path evaluates the four Catmull-Rom/Lagrange weights as products
// of first-degree factors rounded at every step, with x/3 computed by a
// corrective division.  This reproduces the 8u/32f cubic goldens bit for
// bit; the plain polynomial form is off by 1 ulp on many taps.

// x/3 in f32: two-rounding corrective division as emitted by the compiler
// for the reciprocal-multiply sequence in the SASS.
__device__ __forceinline__ float div3(float x) {
  const float kThird = 0x1.555556p-2f;  // RN(1/3)
  const float a = __fmaf_rn(x, kThird, 0.0f);
  const float r = __fmaf_rn(a, -3.0f, x);
  return __fmaf_rn(r, kThird, a);
}

// Weights for taps (tx-1, tx, tx+1, tx+2) given t = fx - (tx-1) in [1, 2).
__device__ __forceinline__ void cubicWeights(float t, float w[4]) {
  const float t1 = __fsub_rn(t, 1.0f);
  const float t2 = __fsub_rn(t, 2.0f);
  const float t3 = __fsub_rn(t, 3.0f);
  w[0] = __fmul_rn(__fmul_rn(t1, __fmul_rn(t2, -0.5f)), div3(t3));
  w[1] = __fmul_rn(__fmul_rn(t, t2), __fmul_rn(t3, 0.5f));
  w[2] = __fmul_rn(t3, __fmul_rn(t1, __fmul_rn(t, -0.5f)));
  w[3] = __fmul_rn(t2, __fmul_rn(__fmul_rn(t1, 0.5f), div3(t)));
}

// ============================================================================
// Plane / pixel accessors
// ============================================================================

// Up to four plane pointers, passed to the kernels by value. Packed layouts
// only use src[0]/dst[0]; planar layouts carry one pointer per channel.
template <typename T> struct WarpPlanes {
  const T *src[4];
  T *dst[4];
};

// Source sampling: coordinates outside the source image are edge-replicated
// (clamped to the image bounds).  The 5090 SASS implements the linear and
// cubic tap clamps as FMNMX min(tap, N-1) / max(tap, 0), which is exactly
// this rule, so the accessor performs it once for all interpolators.
template <typename T, int Channels, bool Planar> class SrcAccessor {
public:
  __device__ __forceinline__ SrcAccessor(const WarpPlanes<T> &planes, int nSrcStep, NppiSize oSrcSize)
      : planes_(planes), nSrcStep_(nSrcStep), oSrcSize_(oSrcSize) {}

  __device__ __forceinline__ work_t<T> at(int c, int x, int y) const {
    if (x < 0) {
      x = 0;
    } else if (x >= oSrcSize_.width) {
      x = oSrcSize_.width - 1;
    }
    if (y < 0) {
      y = 0;
    } else if (y >= oSrcSize_.height) {
      y = oSrcSize_.height - 1;
    }
    const T *base = Planar ? planes_.src[c] : planes_.src[0];
    const size_t idx = Planar ? static_cast<size_t>(x) : (static_cast<size_t>(x) * Channels + c);
    const T *row = reinterpret_cast<const T *>(reinterpret_cast<const char *>(base) +
                                              static_cast<size_t>(y) * nSrcStep_);
    return pixelAsFloat(row[idx]);
  }

private:
  WarpPlanes<T> planes_;
  int nSrcStep_;
  NppiSize oSrcSize_;
};

// Destination write, addressed in absolute image coordinates.
template <typename T, int Channels, bool Planar> class DstAccessor {
public:
  __device__ __forceinline__ DstAccessor(const WarpPlanes<T> &planes, int nDstStep)
      : planes_(planes), nDstStep_(nDstStep) {}

  __device__ __forceinline__ void write(int x, int y, int c, T v) const {
    T *base = Planar ? planes_.dst[c] : planes_.dst[0];
    const size_t idx = Planar ? static_cast<size_t>(x) : (static_cast<size_t>(x) * Channels + c);
    T *row = reinterpret_cast<T *>(reinterpret_cast<char *>(base) + static_cast<size_t>(y) * nDstStep_);
    row[idx] = v;
  }

private:
  WarpPlanes<T> planes_;
  int nDstStep_;
};

// ============================================================================
// Interpolators
// ============================================================================

template <typename Acc>
__device__ __forceinline__ auto sampleNN(const Acc &acc, int c, float fx, float fy) -> decltype(acc.at(c, 0, 0)) {
  const int x = static_cast<int>(floorf(fx + 0.5f));
  const int y = static_cast<int>(floorf(fy + 0.5f));
  return acc.at(c, x, y);
}

// Non-fused round-to-nearest arithmetic.  nvcc would otherwise contract
// a*b+c into an FMA and change the last bits, which flips the exact .5
// rounding ties in the 8u/16u goldens.
__device__ __forceinline__ float mul_rn(float a, float b) { return a * b; }
__device__ __forceinline__ float add_rn(float a, float b) { return a + b; }
__device__ __forceinline__ float sub_rn(float a, float b) { return a - b; }
__device__ __forceinline__ double mul_rn(double a, double b) { return a * b; }
__device__ __forceinline__ double add_rn(double a, double b) { return a + b; }
__device__ __forceinline__ double sub_rn(double a, double b) { return a - b; }

// fxX selects the x tap base and fraction; it differs from fx only for the
// batch kernel's row-pair tap-block reuse (see warpAffineBatchKernel).
template <typename Acc>
__device__ __forceinline__ auto sampleLinear(const Acc &acc, int c, float fx, float fy,
                                             float fxX) -> decltype(acc.at(c, 0, 0)) {
  // 5090 SASS: negative coordinates are clamped to 0 (FSEL) before the
  // floor.  Dead under the in-ROI guard but kept for faithfulness.
  fx = fx < 0.0f ? 0.0f : fx;
  fy = fy < 0.0f ? 0.0f : fy;
  fxX = fxX < 0.0f ? 0.0f : fxX;
  const int x0 = static_cast<int>(floorf(fxX));
  const int y0 = static_cast<int>(floorf(fy));
  const float dx = sub_rn(fxX, static_cast<float>(x0));
  const float dy = sub_rn(fy, static_cast<float>(y0));
  const auto v00 = acc.at(c, x0, y0);
  const auto v10 = acc.at(c, x0 + 1, y0);
  const auto v01 = acc.at(c, x0, y0 + 1);
  const auto v11 = acc.at(c, x0 + 1, y0 + 1);
  using VT = decltype(v00);
  const VT dxv = static_cast<VT>(dx);
  const VT dyv = static_cast<VT>(dy);
  // Weight form (matches NPP bit for bit; the lerp form differs by 1 ulp on
  // exact .5 ties): v0 = (1-dx)v00 + dx v10, v1 likewise, v = (1-dy)v0 + dy v1.
  const VT one = static_cast<VT>(1);
  const VT ux = sub_rn(one, dxv);
  const VT uy = sub_rn(one, dyv);
  const auto v0 = add_rn(mul_rn(ux, v00), mul_rn(dxv, v10));
  const auto v1 = add_rn(mul_rn(ux, v01), mul_rn(dxv, v11));
  return add_rn(mul_rn(uy, v0), mul_rn(dyv, v1));
}

template <typename Acc>
__device__ __forceinline__ auto sampleCubic(const Acc &acc, int c, float fx, float fy,
                                            float fxX) -> decltype(acc.at(c, 0, 0)) {
  const int tx = static_cast<int>(floorf(fxX));
  const int ty = static_cast<int>(floorf(fy));
  const float fxC = fxX;
  using VT = decltype(acc.at(c, 0, 0));
  if constexpr (std::is_same_v<VT, float>) {
    // Exact 5090 sequence: t = fx - (tx-1) in [1,2), Lagrange-product
    // weights, row sums accumulated in tap order 1,0,2,3 starting from a
    // plain multiply, then the column pass in tap order 0,1,2,3.  Taps are
    // edge-replicated by the accessor.
    const float t = __fsub_rn(fxC, static_cast<float>(tx - 1));
    const float u = __fsub_rn(fy, static_cast<float>(ty - 1));
    float wx[4], wy[4];
    cubicWeights(t, wx);
    cubicWeights(u, wy);
    float rows[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int y = ty - 1 + j;
      float r = __fmul_rn(wx[1], acc.at(c, tx, y));
      r = __fmaf_rn(wx[0], acc.at(c, tx - 1, y), r);
      r = __fmaf_rn(wx[2], acc.at(c, tx + 1, y), r);
      r = __fmaf_rn(wx[3], acc.at(c, tx + 2, y), r);
      rows[j] = r;
    }
    float out = __fmul_rn(wy[0], rows[0]);
    out = __fmaf_rn(wy[1], rows[1], out);
    out = __fmaf_rn(wy[2], rows[2], out);
    out = __fmaf_rn(wy[3], rows[3], out);
    return out;
  } else {
    const float ax = fxC - static_cast<float>(tx);
    const float ay = fy - static_cast<float>(ty);
    float wx[4], wy[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      wx[i] = cubicWeight(ax - static_cast<float>(i - 1));
      wy[i] = cubicWeight(ay - static_cast<float>(i - 1));
    }
    auto sum = acc.at(c, tx, ty) * 0;  // zero of the working type
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        sum += acc.at(c, tx + i - 1, ty + j - 1) * (wx[i] * wy[j]);
      }
    }
    return sum;
  }
}

// ============================================================================
// Host-side validation and matrix helpers
// ============================================================================

inline int clipMin(int a, int b) { return a < b ? a : b; }
inline int clipMax(int a, int b) { return a > b ? a : b; }

// Intersection of the source ROI with the source image must be non-empty and
// at least 2x2 pixels (NPP affine transform error codes).
inline NppStatus validateSrcRoi(NppiSize oSrcSize, NppiRect oSrcROI) {
  // 5090: a negative source ROI origin is a SIZE_ERROR (an ROI that is merely
  // outside the image but non-negative is WRONG_INTERSECTION_ROI_ERROR).
  if (oSrcROI.x < 0 || oSrcROI.y < 0 || oSrcROI.width < 0 || oSrcROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  const int x0 = clipMax(oSrcROI.x, 0);
  const int y0 = clipMax(oSrcROI.y, 0);
  const int x1 = clipMin(oSrcROI.x + oSrcROI.width, oSrcSize.width);
  const int y1 = clipMin(oSrcROI.y + oSrcROI.height, oSrcSize.height);
  if (x1 <= x0 || y1 <= y0) {
    return NPP_WRONG_INTERSECTION_ROI_ERROR;
  }
  if (x1 - x0 <= 1 || y1 - y0 <= 1) {
    return NPP_RECTANGLE_ERROR;
  }
  return NPP_NO_ERROR;
}

inline NppStatus validateDstRoi(NppiRect oDstROI) {
  // 5090: a negative destination extent is a SIZE_ERROR; a zero extent is the
  // empty-output no-op (NO_ERROR), even at a non-zero ROI offset.
  if (oDstROI.width < 0 || oDstROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  return NPP_NO_ERROR;
}

inline NppStatus validateInterpolation(int eInterpolation) {
  if (eInterpolation == NPPI_INTER_NN || eInterpolation == NPPI_INTER_LINEAR || eInterpolation == NPPI_INTER_CUBIC) {
    return NPP_NO_ERROR;
  }
  return NPP_INTERPOLATION_ERROR;
}

// Invert a 2x3 affine matrix; false when (numerically) singular.
inline bool invertAffine(const double in[2][3], double out[2][3]) {
  const double det = in[0][0] * in[1][1] - in[0][1] * in[1][0];
  if (std::fabs(det) < 1e-12) {
    return false;
  }
  const double invDet = 1.0 / det;
  out[0][0] = in[1][1] * invDet;
  out[0][1] = -in[0][1] * invDet;
  out[1][0] = -in[1][0] * invDet;
  out[1][1] = in[0][0] * invDet;
  out[0][2] = -(out[0][0] * in[0][2] + out[0][1] * in[1][2]);
  out[1][2] = -(out[1][0] * in[0][2] + out[1][1] * in[1][2]);
  return true;
}

// Affine matrix mapping the first three srcQuad vertices onto the first three
// dstQuad vertices (nppiWarpAffineQuad contract). Returns false when the three
// source vertices are collinear.
inline bool affineFromQuads(const double aSrcQuad[4][2], const double aDstQuad[4][2], double out[2][3],
                            bool *quadWarning) {
  const double u1x = aSrcQuad[1][0] - aSrcQuad[0][0];
  const double u1y = aSrcQuad[1][1] - aSrcQuad[0][1];
  const double u2x = aSrcQuad[2][0] - aSrcQuad[0][0];
  const double u2y = aSrcQuad[2][1] - aSrcQuad[0][1];
  const double det = u1x * u2y - u2x * u1y;
  if (std::fabs(det) < 1e-12) {
    return false;
  }
  const double v1x = aDstQuad[1][0] - aDstQuad[0][0];
  const double v1y = aDstQuad[1][1] - aDstQuad[0][1];
  const double v2x = aDstQuad[2][0] - aDstQuad[0][0];
  const double v2y = aDstQuad[2][1] - aDstQuad[0][1];
  out[0][0] = (v1x * u2y - v2x * u1y) / det;
  out[0][1] = (v2x * u1x - v1x * u2x) / det;
  out[1][0] = (v1y * u2y - v2y * u1y) / det;
  out[1][1] = (v2y * u1x - v1y * u2x) / det;
  out[0][2] = aDstQuad[0][0] - out[0][0] * aSrcQuad[0][0] - out[0][1] * aSrcQuad[0][1];
  out[1][2] = aDstQuad[0][1] - out[1][0] * aSrcQuad[0][0] - out[1][1] * aSrcQuad[0][1];

  *quadWarning = false;
  const double m3x = out[0][0] * aSrcQuad[3][0] + out[0][1] * aSrcQuad[3][1] + out[0][2];
  const double m3y = out[1][0] * aSrcQuad[3][0] + out[1][1] * aSrcQuad[3][1] + out[1][2];
  const double scale = 1.0 + std::fabs(aDstQuad[3][0]) + std::fabs(aDstQuad[3][1]);
  if (std::fabs(m3x - aDstQuad[3][0]) > 1e-6 * scale || std::fabs(m3y - aDstQuad[3][1]) > 1e-6 * scale) {
    *quadWarning = true;
  }
  return true;
}

// 5090 sampling gate shared by the single-image and batch kernels: the mapped
// point must lie inside the clipped source ROI with an inclusive upper bound of
// the last pixel coordinate (x1-1, y1-1).  Probed on both entry points: a
// coordinate of 6.4 on a 7-wide ROI is rejected for NN, linear and cubic even
// though it would round to pixel 6, while exactly 6.0 is accepted.
__device__ __forceinline__ bool inSourceRoi(float fx, float fy, NppiRect roi) {
  return fx >= static_cast<float>(roi.x) && fx <= static_cast<float>(roi.x + roi.width - 1) &&
         fy >= static_cast<float>(roi.y) && fy <= static_cast<float>(roi.y + roi.height - 1);
}

// ============================================================================
// Resampling kernel
// ============================================================================

template <typename T, int Channels, bool AlphaPreserve, bool Planar>
__global__ void warpAffineKernel(WarpPlanes<T> planes, NppiSize oSrcSize, NppiRect oSrcROI, int nSrcStep,
                                 int nDstStep, NppiRect oDstROI, double c00, double c01, double c02, double c10,
                                 double c11, double c12, int eInterpolation) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= oDstROI.width || y >= oDstROI.height) {
    return;
  }
  // NPP applies the affine map in single precision: the double coefficients
  // are rounded to float and the linear part is fused into one FMA
  // (fx = fma(dx, c00, dy * c01) + c02, 5090 SASS FFMA + FADD).  The
  // non-fused form differs in the last bit whenever c01 != 0 (shear/rotate)
  // and those bits flip exact .5 rounding ties downstream.
  const float dx = static_cast<float>(oDstROI.x + x);
  const float dy = static_cast<float>(oDstROI.y + y);
  const float fx = __fadd_rn(__fmaf_rn(dx, static_cast<float>(c00), __fmul_rn(static_cast<float>(c01), dy)),
                             static_cast<float>(c02));
  const float fy = __fadd_rn(__fmaf_rn(dx, static_cast<float>(c10), __fmul_rn(static_cast<float>(c11), dy)),
                             static_cast<float>(c12));
  // NPP leaves the destination pixel untouched when the mapped source point
  // falls outside the (clipped) source ROI (5090: the dst buffer keeps its
  // previous value for those pixels, it is not zero-filled).  Samples inside
  // the ROI whose bilinear/cubic neighbourhood crosses the image edge
  // edge-replicate the missing neighbours (see SrcAccessor).
  // The gate compares the double-precision map: a quad coordinate of
  // 2.99999994 is rejected while the same pixel reached through an exactly
  // representable map (3.0) is written.
  if (!inSourceRoi(fx, fy, oSrcROI)) {
    return;
  }

  const SrcAccessor<T, Channels, Planar> src(planes, nSrcStep, oSrcSize);
  const DstAccessor<T, Channels, Planar> dst(planes, nDstStep);
  constexpr int kActive = AlphaPreserve ? Channels - 1 : Channels;
#pragma unroll
  for (int c = 0; c < kActive; ++c) {
    work_t<T> v(0);
    switch (eInterpolation) {
    case NPPI_INTER_NN:
      v = sampleNN(src, c, fx, fy);
      break;
    case NPPI_INTER_LINEAR:
      v = sampleLinear(src, c, fx, fy, fx);
      break;
    case NPPI_INTER_CUBIC:
      v = sampleCubic(src, c, fx, fy, fx);
      break;
    default:
      v = work_t<T>(0);
      break;
    }
    // 5090: writes land at absolute image coordinates (dstROI.x + x,
    // dstROI.y + y), not ROI-relative ones -- a non-zero dstROI offset
    // reproduces the golden only with absolute addressing.
    dst.write(oDstROI.x + x, oDstROI.y + y, c, castSample<T>(v));
  }
}

template <typename T, int Channels, bool AlphaPreserve, bool Planar>
NppStatus launchWarp(const WarpPlanes<T> &planes, NppiSize oSrcSize, NppiRect oSrcROI, int nSrcStep, int nDstStep,
                     NppiRect oDstROI, const double aMap[2][3], int eInterpolation, cudaStream_t stream) {
  if (oDstROI.width <= 0 || oDstROI.height <= 0) {
    return NPP_NO_ERROR;
  }
  const int rx = clipMax(oSrcROI.x, 0);
  const int ry = clipMax(oSrcROI.y, 0);
  const int rx1 = clipMin(oSrcROI.x + oSrcROI.width, oSrcSize.width);
  const int ry1 = clipMin(oSrcROI.y + oSrcROI.height, oSrcSize.height);
  const NppiRect roi{rx, ry, rx1 - rx, ry1 - ry};
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x,
                (oDstROI.height + blockSize.y - 1) / blockSize.y);
  warpAffineKernel<T, Channels, AlphaPreserve, Planar>
      <<<gridSize, blockSize, 0, stream>>>(planes, oSrcSize, roi, nSrcStep, nDstStep, oDstROI, aMap[0][0],
                                           aMap[0][1], aMap[0][2], aMap[1][0], aMap[1][1], aMap[1][2],
                                           eInterpolation);
  return npp_kernel::checkCudaKernelError();
}

// Common validation for the single-image families. Returns the mapped (dst ->
// src) matrix in pMap when the caller passes a non-null pointer.
inline NppStatus validateWarp(const void *pSrc, const void *pDst, NppiSize oSrcSize, NppiRect oSrcROI,
                              NppiRect oDstROI, int eInterpolation, int minSrcSize) {
  if (pSrc == nullptr || pDst == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSrcSize.width < minSrcSize || oSrcSize.height < minSrcSize) {
    return NPP_SIZE_ERROR;
  }
  const NppStatus roiStatus = validateSrcRoi(oSrcSize, oSrcROI);
  if (roiStatus != NPP_NO_ERROR) {
    return roiStatus;
  }
  const NppStatus dstStatus = validateDstRoi(oDstROI);
  if (dstStatus != NPP_NO_ERROR) {
    return dstStatus;
  }
  return validateInterpolation(eInterpolation);
}

}  // namespace

// ============================================================================
// Executor definitions
// ============================================================================

// ---- nppiWarpAffine / nppiWarpAffineBack (shared shape, differing map) ----

template <typename T, int Channels, bool AlphaPreserve, bool InvertMap>
NppStatus warpAffineImpl(const T *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, T *pDst, int nDstStep,
                         NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation, cudaStream_t stream) {
  const NppStatus status = validateWarp(pSrc, pDst, oSrcSize, oSrcROI, oDstROI, eInterpolation, 2);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  if (oDstROI.width == 0 || oDstROI.height == 0) {
    return NPP_NO_ERROR;  // empty destination: nothing written, steps unchecked
  }
  // Step validation (5090): the size check comes first, then the channel
  // alignment modulus sizeof(element) * 2^v2(Channels); source before dest.
  constexpr int chAlign = Channels & -Channels;
  constexpr int align = static_cast<int>(sizeof(T)) * chAlign;
  if (static_cast<unsigned>(nSrcStep) < static_cast<unsigned>(oSrcROI.width * Channels * static_cast<int>(sizeof(T)))) {
    return NPP_STEP_ERROR;
  }
  if (nSrcStep % align != 0) {
    return NPP_NOT_EVEN_STEP_ERROR;
  }
  if (static_cast<unsigned>(nDstStep) < static_cast<unsigned>(oDstROI.width * Channels * static_cast<int>(sizeof(T)))) {
    return NPP_STEP_ERROR;
  }
  if (nDstStep % align != 0) {
    return NPP_NOT_EVEN_STEP_ERROR;
  }
  double map[2][3];
  if (InvertMap) {
    // nppiWarpAffine receives the forward src -> dst map and inverts it.
    if (!invertAffine(aCoeffs, map)) {
      return NPP_COEFFICIENT_ERROR;
    }
  } else {
    // nppiWarpAffineBack receives the dst -> src map directly.
    map[0][0] = aCoeffs[0][0];
    map[0][1] = aCoeffs[0][1];
    map[0][2] = aCoeffs[0][2];
    map[1][0] = aCoeffs[1][0];
    map[1][1] = aCoeffs[1][1];
    map[1][2] = aCoeffs[1][2];
  }
  WarpPlanes<T> planes;
  planes.src[0] = pSrc;
  planes.dst[0] = pDst;
  return launchWarp<T, Channels, AlphaPreserve, false>(planes, oSrcSize, oSrcROI, nSrcStep, nDstStep, oDstROI, map,
                                                       eInterpolation, stream);
}

template <typename T, int Channels, bool AlphaPreserve>
NppStatus warpAffine(const T *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, T *pDst, int nDstStep,
                     NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation, cudaStream_t stream) {
  return warpAffineImpl<T, Channels, AlphaPreserve, true>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI,
                                                          aCoeffs, eInterpolation, stream);
}

template <typename T, int Channels>
NppStatus warpAffinePlanar(const T *const *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, T *const *pDst,
                           int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation,
                           cudaStream_t stream) {
  if (pSrc == nullptr || pDst == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  const NppStatus status = validateWarp(pSrc[0], pDst[0], oSrcSize, oSrcROI, oDstROI, eInterpolation, 1);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  double map[2][3];
  if (!invertAffine(aCoeffs, map)) {
    return NPP_COEFFICIENT_ERROR;
  }
  WarpPlanes<T> planes;
  for (int c = 0; c < Channels; ++c) {
    planes.src[c] = pSrc[c];
    planes.dst[c] = pDst[c];
  }
  return launchWarp<T, Channels, false, true>(planes, oSrcSize, oSrcROI, nSrcStep, nDstStep, oDstROI, map, eInterpolation,
                                              stream);
}

template <typename T, int Channels, bool AlphaPreserve>
NppStatus warpAffineBack(const T *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, T *pDst, int nDstStep,
                         NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation, cudaStream_t stream) {
  return warpAffineImpl<T, Channels, AlphaPreserve, false>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI,
                                                           aCoeffs, eInterpolation, stream);
}

template <typename T, int Channels>
NppStatus warpAffineBackPlanar(const T *const *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                               T *const *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                               int eInterpolation, cudaStream_t stream) {
  if (pSrc == nullptr || pDst == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  const NppStatus status = validateWarp(pSrc[0], pDst[0], oSrcSize, oSrcROI, oDstROI, eInterpolation, 1);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  double map[2][3] = {{aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2]}, {aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2]}};
  WarpPlanes<T> planes;
  for (int c = 0; c < Channels; ++c) {
    planes.src[c] = pSrc[c];
    planes.dst[c] = pDst[c];
  }
  return launchWarp<T, Channels, false, true>(planes, oSrcSize, oSrcROI, nSrcStep, nDstStep, oDstROI, map, eInterpolation,
                                              stream);
}

// ---- nppiWarpAffineQuad ----

template <typename T, int Channels, bool AlphaPreserve>
NppStatus warpAffineQuad(const T *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2],
                         T *pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation,
                         cudaStream_t stream) {
  const NppStatus status = validateWarp(pSrc, pDst, oSrcSize, oSrcROI, oDstROI, eInterpolation, 1);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  double fwd[2][3];
  bool quadWarning = false;
  if (!affineFromQuads(aSrcQuad, aDstQuad, fwd, &quadWarning)) {
    return NPP_COEFFICIENT_ERROR;
  }
  // 5090: the quad kernel is fed float-rounded map coefficients.
  for (int r = 0; r < 2; ++r) {
    for (int k = 0; k < 3; ++k) {
      fwd[r][k] = static_cast<double>(static_cast<float>(fwd[r][k]));
    }
  }
  double map[2][3];
  if (!invertAffine(fwd, map)) {
    return NPP_COEFFICIENT_ERROR;
  }
  WarpPlanes<T> planes;
  planes.src[0] = pSrc;
  planes.dst[0] = pDst;
  const NppStatus warpStatus =
      launchWarp<T, Channels, AlphaPreserve, false>(planes, oSrcSize, oSrcROI, nSrcStep, nDstStep, oDstROI, map,
                                                    eInterpolation, stream);
  if (warpStatus != NPP_NO_ERROR) {
    return warpStatus;
  }
  return quadWarning ? NPP_AFFINE_QUAD_INCORRECT_WARNING : NPP_NO_ERROR;
}

template <typename T, int Channels>
NppStatus warpAffineQuadPlanar(const T *const *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                               const double aSrcQuad[4][2], T *const *pDst, int nDstStep, NppiRect oDstROI,
                               const double aDstQuad[4][2], int eInterpolation, cudaStream_t stream) {
  if (pSrc == nullptr || pDst == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  const NppStatus status = validateWarp(pSrc[0], pDst[0], oSrcSize, oSrcROI, oDstROI, eInterpolation, 1);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  double fwd[2][3];
  bool quadWarning = false;
  if (!affineFromQuads(aSrcQuad, aDstQuad, fwd, &quadWarning)) {
    return NPP_COEFFICIENT_ERROR;
  }
  double map[2][3];
  if (!invertAffine(fwd, map)) {
    return NPP_COEFFICIENT_ERROR;
  }
  WarpPlanes<T> planes;
  for (int c = 0; c < Channels; ++c) {
    planes.src[c] = pSrc[c];
    planes.dst[c] = pDst[c];
  }
  const NppStatus warpStatus = launchWarp<T, Channels, false, true>(planes, oSrcSize, oSrcROI, nSrcStep, nDstStep, oDstROI,
                                                                    map, eInterpolation, stream);
  if (warpStatus != NPP_NO_ERROR) {
    return warpStatus;
  }
  return quadWarning ? NPP_AFFINE_QUAD_INCORRECT_WARNING : NPP_NO_ERROR;
}

// ---- nppiWarpAffineBatch / nppiWarpAffineBatchInit ----

namespace {

// nppiWarpAffineBatchInit, replayed from the 5090 kernel SASS
// (warpAffineBatchInit):
//   det = fma(a00, a11, -(a01*a10))
//   d   = (det != 0) ? det : a00          <- singular maps use a00 as the
//                                            divisor and 1.0 for the [0][0]
//                                            cofactor (recorded singular
//                                            goldens: {1,2,0,2,4,0} maps to
//                                            {1,-2,0, 0,1,0})
//   the four cofactors are divided in double and rounded to float, then the
//   translation is recomputed from the *float* cofactors and also rounded to
//   float (the stored doubles are float values).
__device__ __forceinline__ void invertAffineBatch(const double in[6], double out[6]) {
  const double det = __fma_rn(in[0], in[4], -__dmul_rn(in[1], in[3]));
  const bool invertible = det != 0.0;
  const double inv = 1.0 / (invertible ? det : in[0]);
  const float m00 = static_cast<float>((invertible ? in[4] : 1.0) * inv);
  const float m01 = static_cast<float>((-in[1]) * inv);
  const float m10 = static_cast<float>((invertible ? -in[3] : 0.0) * inv);
  const float m11 = static_cast<float>(in[0] * inv);
  out[0] = m00;
  out[1] = m01;
  out[3] = m10;
  out[4] = m11;
  out[2] = -static_cast<float>(__fma_rn(in[2], static_cast<double>(m00), __dmul_rn(in[5], static_cast<double>(m01))));
  out[5] = -static_cast<float>(__fma_rn(in[2], static_cast<double>(m10), __dmul_rn(in[5], static_cast<double>(m11))));
}

__global__ void warpAffineBatchInitKernel(NppiWarpAffineBatchCXR *pBatchList, unsigned int nBatchSize) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nBatchSize) {
    return;
  }
  NppiWarpAffineBatchCXR &item = pBatchList[i];
  if (item.pCoeffs == nullptr) {
    return;
  }
  double in[6], out[6];
#pragma unroll
  for (int k = 0; k < 6; ++k) {
    in[k] = static_cast<double>(item.pCoeffs[k]);
  }
  invertAffineBatch(in, out);
#pragma unroll
  for (int k = 0; k < 6; ++k) {
    item.aTransformedCoeffs[k / 3][k % 3] = out[k];
  }
}

__device__ __forceinline__ void batchMap(const double c[2][3], int x, int y, float &fx, float &fy) {
  const double dx = static_cast<double>(static_cast<float>(x));
  const double dy = static_cast<double>(static_cast<float>(y));
  fx = static_cast<float>(__dadd_rn(__fma_rn(c[0][1], dy, __dmul_rn(c[0][0], dx)), c[0][2]));
  fy = static_cast<float>(__dadd_rn(__fma_rn(c[1][1], dy, __dmul_rn(c[1][0], dx)), c[1][2]));
}

// The batch kernel mirrors the single-image one: same absolute destination
// addressing, same source-ROI gate and same interpolators.  The 5090 SASS
// (warpAffineNaiveBatchBy2Rows<float, ...>) evaluates the map in double and
// rounds the result to float for the gates and the sampling -- DMUL + DFMA +
// DADD for a00*x + a01*y + a02, then F2F.F32.F64.
template <typename T, int Channels, bool AlphaPreserve>
__global__ void warpAffineBatchKernel(const NppiWarpAffineBatchCXR *pBatchList, unsigned int nBatchSize,
                                      NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI,
                                      int eInterpolation) {
  const unsigned int img = blockIdx.z;
  if (img >= nBatchSize) {
    return;
  }
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= oDstRectROI.width || y >= oDstRectROI.height) {
    return;
  }
  const NppiWarpAffineBatchCXR &item = pBatchList[img];
  if (item.pSrc == nullptr || item.pDst == nullptr) {
    // NVIDIA NPP 13.4 faults here; leaving the image untouched is strictly
    // safer and matches the documented "no work for an empty entry".
    return;
  }
  // 5090: the batch kernel maps the *ROI-relative* destination index (not the
  // absolute image position; probed with non-zero oDstRectROI offsets) while
  // the write still lands at the absolute pixel.  The map is evaluated with
  // the (float-rounded) transformed coefficients in double and rounded to
  // float: DMUL a00*x, DFMA a01*y + that, DADD + a02, F2F.F32.F64.
  const int ax = oDstRectROI.x + x;
  const int ay = oDstRectROI.y + y;
  float fx, fy;
  batchMap(item.aTransformedCoeffs, x, y, fx, fy);
  // 5090 batch gate: the mapped point must be inside the *clipped source ROI*
  // with an inclusive upper bound of the last pixel (x1-1, y1-1).  Unlike the
  // single-image kernel this rejects coordinates such as 6.4 on a 7-wide ROI
  // even when they would round to pixel 6.
  if (!inSourceRoi(fx, fy, oSrcRectROI)) {
    return;
  }
  // The NPP batch kernel resamples two rows per iteration and only advances
  // the row coordinate when the even row's mapped point is inside the gate;
  // an odd destination row is therefore dropped whenever the even row of the
  // same pair is outside the source ROI (observed on the 5090 with a
  // half-pixel vertical translation: rows 0 and 1 stay untouched).
  float fxX = fx;
  if ((y & 1) != 0) {
    float ex, ey;
    batchMap(item.aTransformedCoeffs, x, y - 1, ex, ey);
    if (!inSourceRoi(ex, ey, oSrcRectROI)) {
      return;
    }
    // The float kernel loads one row-pair tap block per iteration and only
    // reloads it when the y tap base changes.  When the even and odd rows share
    // the same base row the odd row therefore reuses the even row's x tap base
    // and x fraction (its own y weights are still used).  The 16f instantiation
    // (SampleLinearImageBatchBy2Rows_16f) does not show this reuse in the
    // recorded goldens.
    if constexpr (!std::is_same_v<T, Npp16f>) {
      if (static_cast<int>(floorf(ey)) == static_cast<int>(floorf(fy))) {
        fxX = ex;
      }
    }
  }

  WarpPlanes<T> planes;
  planes.src[0] = static_cast<const T *>(item.pSrc);
  planes.dst[0] = static_cast<T *>(item.pDst);
  const SrcAccessor<T, Channels, false> src(planes, item.nSrcStep, oSmallestSrcSize);
  const DstAccessor<T, Channels, false> dst(planes, item.nDstStep);
  constexpr int kActive = AlphaPreserve ? Channels - 1 : Channels;
  for (int c = 0; c < kActive; ++c) {
    work_t<T> v(0);
    switch (eInterpolation) {
    case NPPI_INTER_NN:
      v = sampleNN(src, c, fx, fy);
      break;
    case NPPI_INTER_LINEAR:
      v = sampleLinear(src, c, fx, fy, fxX);
      break;
    case NPPI_INTER_CUBIC:
      v = sampleCubic(src, c, fx, fy, fxX);
      break;
    default:
      v = work_t<T>(0);
      break;
    }
    // Absolute image coordinates, like the single-image kernel.
    dst.write(oDstRectROI.x + x, oDstRectROI.y + y, c, castSample<T>(v));
  }
}

}  // namespace

NppStatus warpAffineBatchInit(NppiWarpAffineBatchCXR *pBatchList, unsigned int nBatchSize, cudaStream_t stream) {
  if (pBatchList == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nBatchSize == 0) {
    // 5090: the reference implementation launches a zero-block kernel and
    // reports NPP_CUDA_KERNEL_EXECUTION_ERROR (recorded
    // nppiWarpAffineBatchInit/err/batchsize0).  Reproduced for status
    // compatibility; the list is left untouched.
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
  const int threads = 128;
  const int blocks = static_cast<int>((nBatchSize + threads - 1) / threads);
  warpAffineBatchInitKernel<<<blocks, threads, 0, stream>>>(pBatchList, nBatchSize);
  return npp_kernel::checkCudaKernelError();
}

template <typename T, int Channels, bool AlphaPreserve>
NppStatus warpAffineBatch(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, int eInterpolation,
                          NppiWarpAffineBatchCXR *pBatchList, unsigned int nBatchSize, cudaStream_t stream) {
  if (pBatchList == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  // 5090: the documented "must be > 1" is enforced -- 0 and 1 both return
  // NPP_BAD_ARGUMENT_ERROR (recorded nppiWarpAffineBatch/err/batchsize0).
  if (nBatchSize < 2) {
    return NPP_BAD_ARGUMENT_ERROR;
  }
  // 5090 batch validation differs from the single-image one.  A negative ROI
  // origin is still NPP_SIZE_ERROR, but every degenerate ROI -- including a
  // source ROI that does not intersect the image at all, a 1-pixel wide clipped
  // ROI and a destination ROI with a zero extent -- is NPP_RECTANGLE_ERROR
  // (recorded srcsize_1 / srcroi_outside / dstroi_zero / dstroi_neg).  The
  // per-image steps are not validated: NVIDIA 13.4 accepts short/misaligned
  // steps here (probed), and reading the device-side list on the host is not
  // possible.
  if (oSrcRectROI.x < 0 || oSrcRectROI.y < 0 || oSrcRectROI.width < 0 || oSrcRectROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  const int rx = clipMax(oSrcRectROI.x, 0);
  const int ry = clipMax(oSrcRectROI.y, 0);
  const int rx1 = clipMin(oSrcRectROI.x + oSrcRectROI.width, oSmallestSrcSize.width);
  const int ry1 = clipMin(oSrcRectROI.y + oSrcRectROI.height, oSmallestSrcSize.height);
  if (rx1 - rx <= 1 || ry1 - ry <= 1) {
    return NPP_RECTANGLE_ERROR;
  }
  if (oDstRectROI.width < 1 || oDstRectROI.height < 1) {
    return NPP_RECTANGLE_ERROR;
  }
  const NppStatus interpStatus = validateInterpolation(eInterpolation);
  if (interpStatus != NPP_NO_ERROR) {
    return interpStatus;
  }
  const NppiRect roi{rx, ry, rx1 - rx, ry1 - ry};
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstRectROI.width + blockSize.x - 1) / blockSize.x,
                (oDstRectROI.height + blockSize.y - 1) / blockSize.y, nBatchSize);
  warpAffineBatchKernel<T, Channels, AlphaPreserve>
      <<<gridSize, blockSize, 0, stream>>>(pBatchList, nBatchSize, oSmallestSrcSize, roi, oDstRectROI,
                                           eInterpolation);
  return npp_kernel::checkCudaKernelError();
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template NppStatus warpAffine<Npp8u, 1, false>(const Npp8u *, NppiSize, int, NppiRect, Npp8u *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffine<Npp8u, 3, false>(const Npp8u *, NppiSize, int, NppiRect, Npp8u *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffine<Npp8u, 4, false>(const Npp8u *, NppiSize, int, NppiRect, Npp8u *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffine<Npp8u, 4, true>(const Npp8u *, NppiSize, int, NppiRect, Npp8u *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffine<Npp16u, 1, false>(const Npp16u *, NppiSize, int, NppiRect, Npp16u *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffine<Npp16u, 3, false>(const Npp16u *, NppiSize, int, NppiRect, Npp16u *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffine<Npp16u, 4, false>(const Npp16u *, NppiSize, int, NppiRect, Npp16u *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffine<Npp16u, 4, true>(const Npp16u *, NppiSize, int, NppiRect, Npp16u *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffine<Npp32s, 1, false>(const Npp32s *, NppiSize, int, NppiRect, Npp32s *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffine<Npp32s, 3, false>(const Npp32s *, NppiSize, int, NppiRect, Npp32s *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffine<Npp32s, 4, false>(const Npp32s *, NppiSize, int, NppiRect, Npp32s *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffine<Npp32s, 4, true>(const Npp32s *, NppiSize, int, NppiRect, Npp32s *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffine<Npp16f, 1, false>(const Npp16f *, NppiSize, int, NppiRect, Npp16f *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffine<Npp16f, 3, false>(const Npp16f *, NppiSize, int, NppiRect, Npp16f *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffine<Npp16f, 4, false>(const Npp16f *, NppiSize, int, NppiRect, Npp16f *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffine<Npp32f, 1, false>(const Npp32f *, NppiSize, int, NppiRect, Npp32f *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffine<Npp32f, 3, false>(const Npp32f *, NppiSize, int, NppiRect, Npp32f *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffine<Npp32f, 4, false>(const Npp32f *, NppiSize, int, NppiRect, Npp32f *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffine<Npp32f, 4, true>(const Npp32f *, NppiSize, int, NppiRect, Npp32f *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffine<Npp64f, 1, false>(const Npp64f *, NppiSize, int, NppiRect, Npp64f *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffine<Npp64f, 3, false>(const Npp64f *, NppiSize, int, NppiRect, Npp64f *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffine<Npp64f, 4, false>(const Npp64f *, NppiSize, int, NppiRect, Npp64f *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffine<Npp64f, 4, true>(const Npp64f *, NppiSize, int, NppiRect, Npp64f *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffinePlanar<Npp8u, 3>(const Npp8u *const *, NppiSize, int, NppiRect, Npp8u *const *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffinePlanar<Npp8u, 4>(const Npp8u *const *, NppiSize, int, NppiRect, Npp8u *const *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffinePlanar<Npp16u, 3>(const Npp16u *const *, NppiSize, int, NppiRect, Npp16u *const *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffinePlanar<Npp16u, 4>(const Npp16u *const *, NppiSize, int, NppiRect, Npp16u *const *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffinePlanar<Npp32s, 3>(const Npp32s *const *, NppiSize, int, NppiRect, Npp32s *const *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffinePlanar<Npp32s, 4>(const Npp32s *const *, NppiSize, int, NppiRect, Npp32s *const *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffinePlanar<Npp32f, 3>(const Npp32f *const *, NppiSize, int, NppiRect, Npp32f *const *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffinePlanar<Npp32f, 4>(const Npp32f *const *, NppiSize, int, NppiRect, Npp32f *const *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffinePlanar<Npp64f, 3>(const Npp64f *const *, NppiSize, int, NppiRect, Npp64f *const *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffinePlanar<Npp64f, 4>(const Npp64f *const *, NppiSize, int, NppiRect, Npp64f *const *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffineBack<Npp8u, 1, false>(const Npp8u *, NppiSize, int, NppiRect, Npp8u *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffineBack<Npp8u, 3, false>(const Npp8u *, NppiSize, int, NppiRect, Npp8u *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffineBack<Npp8u, 4, false>(const Npp8u *, NppiSize, int, NppiRect, Npp8u *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffineBack<Npp8u, 4, true>(const Npp8u *, NppiSize, int, NppiRect, Npp8u *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffineBack<Npp16u, 1, false>(const Npp16u *, NppiSize, int, NppiRect, Npp16u *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffineBack<Npp16u, 3, false>(const Npp16u *, NppiSize, int, NppiRect, Npp16u *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffineBack<Npp16u, 4, false>(const Npp16u *, NppiSize, int, NppiRect, Npp16u *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffineBack<Npp16u, 4, true>(const Npp16u *, NppiSize, int, NppiRect, Npp16u *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffineBack<Npp32s, 1, false>(const Npp32s *, NppiSize, int, NppiRect, Npp32s *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffineBack<Npp32s, 3, false>(const Npp32s *, NppiSize, int, NppiRect, Npp32s *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffineBack<Npp32s, 4, false>(const Npp32s *, NppiSize, int, NppiRect, Npp32s *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffineBack<Npp32s, 4, true>(const Npp32s *, NppiSize, int, NppiRect, Npp32s *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffineBack<Npp32f, 1, false>(const Npp32f *, NppiSize, int, NppiRect, Npp32f *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffineBack<Npp32f, 3, false>(const Npp32f *, NppiSize, int, NppiRect, Npp32f *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffineBack<Npp32f, 4, false>(const Npp32f *, NppiSize, int, NppiRect, Npp32f *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffineBack<Npp32f, 4, true>(const Npp32f *, NppiSize, int, NppiRect, Npp32f *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffineBackPlanar<Npp8u, 3>(const Npp8u *const *, NppiSize, int, NppiRect, Npp8u *const *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffineBackPlanar<Npp8u, 4>(const Npp8u *const *, NppiSize, int, NppiRect, Npp8u *const *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffineBackPlanar<Npp16u, 3>(const Npp16u *const *, NppiSize, int, NppiRect, Npp16u *const *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffineBackPlanar<Npp16u, 4>(const Npp16u *const *, NppiSize, int, NppiRect, Npp16u *const *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffineBackPlanar<Npp32s, 3>(const Npp32s *const *, NppiSize, int, NppiRect, Npp32s *const *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffineBackPlanar<Npp32s, 4>(const Npp32s *const *, NppiSize, int, NppiRect, Npp32s *const *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffineBackPlanar<Npp32f, 3>(const Npp32f *const *, NppiSize, int, NppiRect, Npp32f *const *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffineBackPlanar<Npp32f, 4>(const Npp32f *const *, NppiSize, int, NppiRect, Npp32f *const *, int, NppiRect, const double[2][3], int, cudaStream_t);
template NppStatus warpAffineQuad<Npp8u, 1, false>(const Npp8u *, NppiSize, int, NppiRect, const double[4][2], Npp8u *, int, NppiRect, const double[4][2], int, cudaStream_t);
template NppStatus warpAffineQuad<Npp8u, 3, false>(const Npp8u *, NppiSize, int, NppiRect, const double[4][2], Npp8u *, int, NppiRect, const double[4][2], int, cudaStream_t);
template NppStatus warpAffineQuad<Npp8u, 4, false>(const Npp8u *, NppiSize, int, NppiRect, const double[4][2], Npp8u *, int, NppiRect, const double[4][2], int, cudaStream_t);
template NppStatus warpAffineQuad<Npp8u, 4, true>(const Npp8u *, NppiSize, int, NppiRect, const double[4][2], Npp8u *, int, NppiRect, const double[4][2], int, cudaStream_t);
template NppStatus warpAffineQuad<Npp16u, 1, false>(const Npp16u *, NppiSize, int, NppiRect, const double[4][2], Npp16u *, int, NppiRect, const double[4][2], int, cudaStream_t);
template NppStatus warpAffineQuad<Npp16u, 3, false>(const Npp16u *, NppiSize, int, NppiRect, const double[4][2], Npp16u *, int, NppiRect, const double[4][2], int, cudaStream_t);
template NppStatus warpAffineQuad<Npp16u, 4, false>(const Npp16u *, NppiSize, int, NppiRect, const double[4][2], Npp16u *, int, NppiRect, const double[4][2], int, cudaStream_t);
template NppStatus warpAffineQuad<Npp16u, 4, true>(const Npp16u *, NppiSize, int, NppiRect, const double[4][2], Npp16u *, int, NppiRect, const double[4][2], int, cudaStream_t);
template NppStatus warpAffineQuad<Npp32s, 1, false>(const Npp32s *, NppiSize, int, NppiRect, const double[4][2], Npp32s *, int, NppiRect, const double[4][2], int, cudaStream_t);
template NppStatus warpAffineQuad<Npp32s, 3, false>(const Npp32s *, NppiSize, int, NppiRect, const double[4][2], Npp32s *, int, NppiRect, const double[4][2], int, cudaStream_t);
template NppStatus warpAffineQuad<Npp32s, 4, false>(const Npp32s *, NppiSize, int, NppiRect, const double[4][2], Npp32s *, int, NppiRect, const double[4][2], int, cudaStream_t);
template NppStatus warpAffineQuad<Npp32s, 4, true>(const Npp32s *, NppiSize, int, NppiRect, const double[4][2], Npp32s *, int, NppiRect, const double[4][2], int, cudaStream_t);
template NppStatus warpAffineQuad<Npp32f, 1, false>(const Npp32f *, NppiSize, int, NppiRect, const double[4][2], Npp32f *, int, NppiRect, const double[4][2], int, cudaStream_t);
template NppStatus warpAffineQuad<Npp32f, 3, false>(const Npp32f *, NppiSize, int, NppiRect, const double[4][2], Npp32f *, int, NppiRect, const double[4][2], int, cudaStream_t);
template NppStatus warpAffineQuad<Npp32f, 4, false>(const Npp32f *, NppiSize, int, NppiRect, const double[4][2], Npp32f *, int, NppiRect, const double[4][2], int, cudaStream_t);
template NppStatus warpAffineQuad<Npp32f, 4, true>(const Npp32f *, NppiSize, int, NppiRect, const double[4][2], Npp32f *, int, NppiRect, const double[4][2], int, cudaStream_t);
template NppStatus warpAffineQuadPlanar<Npp8u, 3>(const Npp8u *const *, NppiSize, int, NppiRect, const double[4][2], Npp8u *const *, int, NppiRect, const double[4][2], int, cudaStream_t);
template NppStatus warpAffineQuadPlanar<Npp8u, 4>(const Npp8u *const *, NppiSize, int, NppiRect, const double[4][2], Npp8u *const *, int, NppiRect, const double[4][2], int, cudaStream_t);
template NppStatus warpAffineQuadPlanar<Npp16u, 3>(const Npp16u *const *, NppiSize, int, NppiRect, const double[4][2], Npp16u *const *, int, NppiRect, const double[4][2], int, cudaStream_t);
template NppStatus warpAffineQuadPlanar<Npp16u, 4>(const Npp16u *const *, NppiSize, int, NppiRect, const double[4][2], Npp16u *const *, int, NppiRect, const double[4][2], int, cudaStream_t);
template NppStatus warpAffineQuadPlanar<Npp32s, 3>(const Npp32s *const *, NppiSize, int, NppiRect, const double[4][2], Npp32s *const *, int, NppiRect, const double[4][2], int, cudaStream_t);
template NppStatus warpAffineQuadPlanar<Npp32s, 4>(const Npp32s *const *, NppiSize, int, NppiRect, const double[4][2], Npp32s *const *, int, NppiRect, const double[4][2], int, cudaStream_t);
template NppStatus warpAffineQuadPlanar<Npp32f, 3>(const Npp32f *const *, NppiSize, int, NppiRect, const double[4][2], Npp32f *const *, int, NppiRect, const double[4][2], int, cudaStream_t);
template NppStatus warpAffineQuadPlanar<Npp32f, 4>(const Npp32f *const *, NppiSize, int, NppiRect, const double[4][2], Npp32f *const *, int, NppiRect, const double[4][2], int, cudaStream_t);
template NppStatus warpAffineBatch<Npp8u, 1, false>(NppiSize, NppiRect, NppiRect, int, NppiWarpAffineBatchCXR *, unsigned int, cudaStream_t);
template NppStatus warpAffineBatch<Npp8u, 3, false>(NppiSize, NppiRect, NppiRect, int, NppiWarpAffineBatchCXR *, unsigned int, cudaStream_t);
template NppStatus warpAffineBatch<Npp8u, 4, false>(NppiSize, NppiRect, NppiRect, int, NppiWarpAffineBatchCXR *, unsigned int, cudaStream_t);
template NppStatus warpAffineBatch<Npp8u, 4, true>(NppiSize, NppiRect, NppiRect, int, NppiWarpAffineBatchCXR *, unsigned int, cudaStream_t);
template NppStatus warpAffineBatch<Npp16f, 1, false>(NppiSize, NppiRect, NppiRect, int, NppiWarpAffineBatchCXR *, unsigned int, cudaStream_t);
template NppStatus warpAffineBatch<Npp16f, 3, false>(NppiSize, NppiRect, NppiRect, int, NppiWarpAffineBatchCXR *, unsigned int, cudaStream_t);
template NppStatus warpAffineBatch<Npp16f, 4, false>(NppiSize, NppiRect, NppiRect, int, NppiWarpAffineBatchCXR *, unsigned int, cudaStream_t);
template NppStatus warpAffineBatch<Npp32f, 1, false>(NppiSize, NppiRect, NppiRect, int, NppiWarpAffineBatchCXR *, unsigned int, cudaStream_t);
template NppStatus warpAffineBatch<Npp32f, 3, false>(NppiSize, NppiRect, NppiRect, int, NppiWarpAffineBatchCXR *, unsigned int, cudaStream_t);
template NppStatus warpAffineBatch<Npp32f, 4, false>(NppiSize, NppiRect, NppiRect, int, NppiWarpAffineBatchCXR *, unsigned int, cudaStream_t);
template NppStatus warpAffineBatch<Npp32f, 4, true>(NppiSize, NppiRect, NppiRect, int, NppiWarpAffineBatchCXR *, unsigned int, cudaStream_t);

}  // namespace geometry
}  // namespace nppi
