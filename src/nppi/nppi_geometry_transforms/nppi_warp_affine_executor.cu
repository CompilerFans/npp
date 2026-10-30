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

// Device -> pixel conversion. Integer destinations round half away from zero
// and saturate to the type range; float destinations cast directly.
template <typename T> __device__ __forceinline__ T castSample(work_t<T> v);

template <> __device__ __forceinline__ Npp8u castSample<Npp8u>(float v) {
  v = floorf(v + 0.5f);
  v = fminf(fmaxf(v, 0.0f), 255.0f);
  return static_cast<Npp8u>(v);
}
template <> __device__ __forceinline__ Npp16u castSample<Npp16u>(float v) {
  v = floorf(v + 0.5f);
  v = fminf(fmaxf(v, 0.0f), 65535.0f);
  return static_cast<Npp16u>(v);
}
template <> __device__ __forceinline__ Npp32s castSample<Npp32s>(float v) {
  v = floorf(v + 0.5f);
  v = fminf(fmaxf(v, -2147483648.0f), 2147483520.0f);
  return static_cast<Npp32s>(v);
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

// Catmull-Rom cubic interpolation weight (a = -0.5).
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

// ============================================================================
// Plane / pixel accessors
// ============================================================================

// Up to four plane pointers, passed to the kernels by value. Packed layouts
// only use src[0]/dst[0]; planar layouts carry one pointer per channel.
template <typename T> struct WarpPlanes {
  const T *src[4];
  T *dst[4];
};

// Source sampling: reads outside the source image contribute 0.
template <typename T, int Channels, bool Planar> class SrcAccessor {
public:
  __device__ __forceinline__ SrcAccessor(const WarpPlanes<T> &planes, int nSrcStep, NppiSize oSrcSize)
      : planes_(planes), nSrcStep_(nSrcStep), oSrcSize_(oSrcSize) {}

  __device__ __forceinline__ work_t<T> at(int c, int x, int y) const {
    if (x < 0 || x >= oSrcSize_.width || y < 0 || y >= oSrcSize_.height) {
      return work_t<T>(0);
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

// Destination write, addressed in ROI-relative coordinates.
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

template <typename Acc>
__device__ __forceinline__ auto sampleLinear(const Acc &acc, int c, float fx, float fy) -> decltype(acc.at(c, 0, 0)) {
  const int x0 = static_cast<int>(floorf(fx));
  const int y0 = static_cast<int>(floorf(fy));
  const float dx = fx - static_cast<float>(x0);
  const float dy = fy - static_cast<float>(y0);
  const auto v00 = acc.at(c, x0, y0);
  const auto v10 = acc.at(c, x0 + 1, y0);
  const auto v01 = acc.at(c, x0, y0 + 1);
  const auto v11 = acc.at(c, x0 + 1, y0 + 1);
  const auto r0 = v00 + (v10 - v00) * dx;
  const auto r1 = v01 + (v11 - v01) * dx;
  return r0 + (r1 - r0) * dy;
}

template <typename Acc>
__device__ __forceinline__ auto sampleCubic(const Acc &acc, int c, float fx, float fy) -> decltype(acc.at(c, 0, 0)) {
  const int tx = static_cast<int>(floorf(fx));
  const int ty = static_cast<int>(floorf(fy));
  const float ax = fx - static_cast<float>(tx);
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

// ============================================================================
// Host-side validation and matrix helpers
// ============================================================================

inline int clipMin(int a, int b) { return a < b ? a : b; }
inline int clipMax(int a, int b) { return a > b ? a : b; }

// Intersection of the source ROI with the source image must be non-empty and
// at least 2x2 pixels (NPP affine transform error codes).
inline NppStatus validateSrcRoi(NppiSize oSrcSize, NppiRect oSrcROI) {
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
  if (oDstROI.width < 1 || oDstROI.height < 1) {
    return NPP_RECTANGLE_ERROR;
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

// ============================================================================
// Resampling kernel
// ============================================================================

template <typename T, int Channels, bool AlphaPreserve, bool Planar>
__global__ void warpAffineKernel(WarpPlanes<T> planes, NppiSize oSrcSize, int nSrcStep, int nDstStep,
                                 NppiRect oDstROI, double c00, double c01, double c02, double c10, double c11,
                                 double c12, int eInterpolation) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= oDstROI.width || y >= oDstROI.height) {
    return;
  }
  const double dx = static_cast<double>(oDstROI.x + x);
  const double dy = static_cast<double>(oDstROI.y + y);
  const double sx = c00 * dx + c01 * dy + c02;
  const double sy = c10 * dx + c11 * dy + c12;
  const float fx = static_cast<float>(sx);
  const float fy = static_cast<float>(sy);

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
      v = sampleLinear(src, c, fx, fy);
      break;
    case NPPI_INTER_CUBIC:
      v = sampleCubic(src, c, fx, fy);
      break;
    default:
      v = work_t<T>(0);
      break;
    }
    dst.write(x, y, c, castSample<T>(v));
  }
}

template <typename T, int Channels, bool AlphaPreserve, bool Planar>
NppStatus launchWarp(const WarpPlanes<T> &planes, NppiSize oSrcSize, int nSrcStep, int nDstStep, NppiRect oDstROI,
                     const double aMap[2][3], int eInterpolation, cudaStream_t stream) {
  if (oDstROI.width <= 0 || oDstROI.height <= 0) {
    return NPP_NO_ERROR;
  }
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x,
                (oDstROI.height + blockSize.y - 1) / blockSize.y);
  warpAffineKernel<T, Channels, AlphaPreserve, Planar>
      <<<gridSize, blockSize, 0, stream>>>(planes, oSrcSize, nSrcStep, nDstStep, oDstROI, aMap[0][0], aMap[0][1],
                                           aMap[0][2], aMap[1][0], aMap[1][1], aMap[1][2], eInterpolation);
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
  const NppStatus status = validateWarp(pSrc, pDst, oSrcSize, oSrcROI, oDstROI, eInterpolation, 1);
  if (status != NPP_NO_ERROR) {
    return status;
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
  return launchWarp<T, Channels, AlphaPreserve, false>(planes, oSrcSize, nSrcStep, nDstStep, oDstROI, map,
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
  return launchWarp<T, Channels, false, true>(planes, oSrcSize, nSrcStep, nDstStep, oDstROI, map, eInterpolation,
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
  return launchWarp<T, Channels, false, true>(planes, oSrcSize, nSrcStep, nDstStep, oDstROI, map, eInterpolation,
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
  double map[2][3];
  if (!invertAffine(fwd, map)) {
    return NPP_COEFFICIENT_ERROR;
  }
  WarpPlanes<T> planes;
  planes.src[0] = pSrc;
  planes.dst[0] = pDst;
  const NppStatus warpStatus =
      launchWarp<T, Channels, AlphaPreserve, false>(planes, oSrcSize, nSrcStep, nDstStep, oDstROI, map,
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
  const NppStatus warpStatus = launchWarp<T, Channels, false, true>(planes, oSrcSize, nSrcStep, nDstStep, oDstROI,
                                                                    map, eInterpolation, stream);
  if (warpStatus != NPP_NO_ERROR) {
    return warpStatus;
  }
  return quadWarning ? NPP_AFFINE_QUAD_INCORRECT_WARNING : NPP_NO_ERROR;
}

// ---- nppiWarpAffineBatch / nppiWarpAffineBatchInit ----

namespace {

__device__ __forceinline__ bool invertAffineDevice(const double in[6], double out[6]) {
  const double det = in[0] * in[4] - in[1] * in[3];
  if (fabs(det) < 1e-12) {
    return false;
  }
  const double invDet = 1.0 / det;
  out[0] = in[4] * invDet;
  out[1] = -in[1] * invDet;
  out[3] = -in[3] * invDet;
  out[4] = in[0] * invDet;
  out[2] = -(out[0] * in[2] + out[1] * in[5]);
  out[5] = -(out[3] * in[2] + out[4] * in[5]);
  return true;
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
  if (invertAffineDevice(in, out)) {
#pragma unroll
    for (int k = 0; k < 6; ++k) {
      item.aTransformedCoeffs[k / 3][k % 3] = out[k];
    }
  } else {
#pragma unroll
    for (int k = 0; k < 6; ++k) {
      item.aTransformedCoeffs[k / 3][k % 3] = 0.0;
    }
  }
}

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
    return;
  }
  const double dx = static_cast<double>(oDstRectROI.x + x);
  const double dy = static_cast<double>(oDstRectROI.y + y);
  const double sx = item.aTransformedCoeffs[0][0] * dx + item.aTransformedCoeffs[0][1] * dy +
                    item.aTransformedCoeffs[0][2];
  const double sy = item.aTransformedCoeffs[1][0] * dx + item.aTransformedCoeffs[1][1] * dy +
                    item.aTransformedCoeffs[1][2];
  const float fx = static_cast<float>(sx);
  const float fy = static_cast<float>(sy);

  WarpPlanes<T> planes;
  planes.src[0] = static_cast<const T *>(item.pSrc);
  planes.dst[0] = static_cast<T *>(item.pDst);
  const SrcAccessor<T, Channels, false> src(planes, item.nSrcStep, oSmallestSrcSize);
  constexpr int kActive = AlphaPreserve ? Channels - 1 : Channels;
  for (int c = 0; c < kActive; ++c) {
    work_t<T> v(0);
    switch (eInterpolation) {
    case NPPI_INTER_NN:
      v = sampleNN(src, c, fx, fy);
      break;
    case NPPI_INTER_LINEAR:
      v = sampleLinear(src, c, fx, fy);
      break;
    case NPPI_INTER_CUBIC:
      v = sampleCubic(src, c, fx, fy);
      break;
    default:
      v = work_t<T>(0);
      break;
    }
    const size_t idx = static_cast<size_t>(x) * Channels + c;
    T *row = reinterpret_cast<T *>(reinterpret_cast<char *>(item.pDst) +
                                   static_cast<size_t>(y) * item.nDstStep);
    row[idx] = castSample<T>(v);
  }
}

}  // namespace

NppStatus warpAffineBatchInit(NppiWarpAffineBatchCXR *pBatchList, unsigned int nBatchSize, cudaStream_t stream) {
  if (pBatchList == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nBatchSize == 0) {
    return NPP_NO_ERROR;
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
  if (nBatchSize == 0) {
    return NPP_NO_ERROR;
  }
  if (oSmallestSrcSize.width < 2 || oSmallestSrcSize.height < 2) {
    return NPP_SIZE_ERROR;
  }
  const NppStatus roiStatus = validateSrcRoi(oSmallestSrcSize, oSrcRectROI);
  if (roiStatus != NPP_NO_ERROR) {
    return roiStatus;
  }
  const NppStatus dstStatus = validateDstRoi(oDstRectROI);
  if (dstStatus != NPP_NO_ERROR) {
    return dstStatus;
  }
  const NppStatus interpStatus = validateInterpolation(eInterpolation);
  if (interpStatus != NPP_NO_ERROR) {
    return interpStatus;
  }
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstRectROI.width + blockSize.x - 1) / blockSize.x,
                (oDstRectROI.height + blockSize.y - 1) / blockSize.y, nBatchSize);
  warpAffineBatchKernel<T, Channels, AlphaPreserve>
      <<<gridSize, blockSize, 0, stream>>>(pBatchList, nBatchSize, oSmallestSrcSize, oSrcRectROI, oDstRectROI,
                                           eInterpolation);
  return npp_kernel::checkCudaKernelError();
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template NppStatus nppi::geometry::warpAffine<Npp8u, 1, false>;
template NppStatus nppi::geometry::warpAffine<Npp8u, 3, false>;
template NppStatus nppi::geometry::warpAffine<Npp8u, 4, false>;
template NppStatus nppi::geometry::warpAffine<Npp8u, 4, true>;
template NppStatus nppi::geometry::warpAffine<Npp16u, 1, false>;
template NppStatus nppi::geometry::warpAffine<Npp16u, 3, false>;
template NppStatus nppi::geometry::warpAffine<Npp16u, 4, false>;
template NppStatus nppi::geometry::warpAffine<Npp16u, 4, true>;
template NppStatus nppi::geometry::warpAffine<Npp32s, 1, false>;
template NppStatus nppi::geometry::warpAffine<Npp32s, 3, false>;
template NppStatus nppi::geometry::warpAffine<Npp32s, 4, false>;
template NppStatus nppi::geometry::warpAffine<Npp32s, 4, true>;
template NppStatus nppi::geometry::warpAffine<Npp16f, 1, false>;
template NppStatus nppi::geometry::warpAffine<Npp16f, 3, false>;
template NppStatus nppi::geometry::warpAffine<Npp16f, 4, false>;
template NppStatus nppi::geometry::warpAffine<Npp32f, 1, false>;
template NppStatus nppi::geometry::warpAffine<Npp32f, 3, false>;
template NppStatus nppi::geometry::warpAffine<Npp32f, 4, false>;
template NppStatus nppi::geometry::warpAffine<Npp32f, 4, true>;
template NppStatus nppi::geometry::warpAffine<Npp64f, 1, false>;
template NppStatus nppi::geometry::warpAffine<Npp64f, 3, false>;
template NppStatus nppi::geometry::warpAffine<Npp64f, 4, false>;
template NppStatus nppi::geometry::warpAffine<Npp64f, 4, true>;
template NppStatus nppi::geometry::warpAffinePlanar<Npp8u, 3>;
template NppStatus nppi::geometry::warpAffinePlanar<Npp8u, 4>;
template NppStatus nppi::geometry::warpAffinePlanar<Npp16u, 3>;
template NppStatus nppi::geometry::warpAffinePlanar<Npp16u, 4>;
template NppStatus nppi::geometry::warpAffinePlanar<Npp32s, 3>;
template NppStatus nppi::geometry::warpAffinePlanar<Npp32s, 4>;
template NppStatus nppi::geometry::warpAffinePlanar<Npp32f, 3>;
template NppStatus nppi::geometry::warpAffinePlanar<Npp32f, 4>;
template NppStatus nppi::geometry::warpAffinePlanar<Npp64f, 3>;
template NppStatus nppi::geometry::warpAffinePlanar<Npp64f, 4>;
template NppStatus nppi::geometry::warpAffineBack<Npp8u, 1, false>;
template NppStatus nppi::geometry::warpAffineBack<Npp8u, 3, false>;
template NppStatus nppi::geometry::warpAffineBack<Npp8u, 4, false>;
template NppStatus nppi::geometry::warpAffineBack<Npp8u, 4, true>;
template NppStatus nppi::geometry::warpAffineBack<Npp16u, 1, false>;
template NppStatus nppi::geometry::warpAffineBack<Npp16u, 3, false>;
template NppStatus nppi::geometry::warpAffineBack<Npp16u, 4, false>;
template NppStatus nppi::geometry::warpAffineBack<Npp16u, 4, true>;
template NppStatus nppi::geometry::warpAffineBack<Npp32s, 1, false>;
template NppStatus nppi::geometry::warpAffineBack<Npp32s, 3, false>;
template NppStatus nppi::geometry::warpAffineBack<Npp32s, 4, false>;
template NppStatus nppi::geometry::warpAffineBack<Npp32s, 4, true>;
template NppStatus nppi::geometry::warpAffineBack<Npp32f, 1, false>;
template NppStatus nppi::geometry::warpAffineBack<Npp32f, 3, false>;
template NppStatus nppi::geometry::warpAffineBack<Npp32f, 4, false>;
template NppStatus nppi::geometry::warpAffineBack<Npp32f, 4, true>;
template NppStatus nppi::geometry::warpAffineBackPlanar<Npp8u, 3>;
template NppStatus nppi::geometry::warpAffineBackPlanar<Npp8u, 4>;
template NppStatus nppi::geometry::warpAffineBackPlanar<Npp16u, 3>;
template NppStatus nppi::geometry::warpAffineBackPlanar<Npp16u, 4>;
template NppStatus nppi::geometry::warpAffineBackPlanar<Npp32s, 3>;
template NppStatus nppi::geometry::warpAffineBackPlanar<Npp32s, 4>;
template NppStatus nppi::geometry::warpAffineBackPlanar<Npp32f, 3>;
template NppStatus nppi::geometry::warpAffineBackPlanar<Npp32f, 4>;
template NppStatus nppi::geometry::warpAffineQuad<Npp8u, 1, false>;
template NppStatus nppi::geometry::warpAffineQuad<Npp8u, 3, false>;
template NppStatus nppi::geometry::warpAffineQuad<Npp8u, 4, false>;
template NppStatus nppi::geometry::warpAffineQuad<Npp8u, 4, true>;
template NppStatus nppi::geometry::warpAffineQuad<Npp16u, 1, false>;
template NppStatus nppi::geometry::warpAffineQuad<Npp16u, 3, false>;
template NppStatus nppi::geometry::warpAffineQuad<Npp16u, 4, false>;
template NppStatus nppi::geometry::warpAffineQuad<Npp16u, 4, true>;
template NppStatus nppi::geometry::warpAffineQuad<Npp32s, 1, false>;
template NppStatus nppi::geometry::warpAffineQuad<Npp32s, 3, false>;
template NppStatus nppi::geometry::warpAffineQuad<Npp32s, 4, false>;
template NppStatus nppi::geometry::warpAffineQuad<Npp32s, 4, true>;
template NppStatus nppi::geometry::warpAffineQuad<Npp32f, 1, false>;
template NppStatus nppi::geometry::warpAffineQuad<Npp32f, 3, false>;
template NppStatus nppi::geometry::warpAffineQuad<Npp32f, 4, false>;
template NppStatus nppi::geometry::warpAffineQuad<Npp32f, 4, true>;
template NppStatus nppi::geometry::warpAffineQuadPlanar<Npp8u, 3>;
template NppStatus nppi::geometry::warpAffineQuadPlanar<Npp8u, 4>;
template NppStatus nppi::geometry::warpAffineQuadPlanar<Npp16u, 3>;
template NppStatus nppi::geometry::warpAffineQuadPlanar<Npp16u, 4>;
template NppStatus nppi::geometry::warpAffineQuadPlanar<Npp32s, 3>;
template NppStatus nppi::geometry::warpAffineQuadPlanar<Npp32s, 4>;
template NppStatus nppi::geometry::warpAffineQuadPlanar<Npp32f, 3>;
template NppStatus nppi::geometry::warpAffineQuadPlanar<Npp32f, 4>;
template NppStatus nppi::geometry::warpAffineBatch<Npp8u, 1, false>;
template NppStatus nppi::geometry::warpAffineBatch<Npp8u, 3, false>;
template NppStatus nppi::geometry::warpAffineBatch<Npp8u, 4, false>;
template NppStatus nppi::geometry::warpAffineBatch<Npp8u, 4, true>;
template NppStatus nppi::geometry::warpAffineBatch<Npp16f, 1, false>;
template NppStatus nppi::geometry::warpAffineBatch<Npp16f, 3, false>;
template NppStatus nppi::geometry::warpAffineBatch<Npp16f, 4, false>;
template NppStatus nppi::geometry::warpAffineBatch<Npp32f, 1, false>;
template NppStatus nppi::geometry::warpAffineBatch<Npp32f, 3, false>;
template NppStatus nppi::geometry::warpAffineBatch<Npp32f, 4, false>;
template NppStatus nppi::geometry::warpAffineBatch<Npp32f, 4, true>;

}  // namespace geometry
}  // namespace nppi
