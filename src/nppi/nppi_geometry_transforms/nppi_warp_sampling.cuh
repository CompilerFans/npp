#pragma once

// Device-side sampling helpers shared by the g5 geometry executors
// (nppi_warp_perspective_executor.cu and nppi_rotate_executor.cu).
//
// The routines below reproduce the sampling model measured on the RTX 5090
// with NVIDIA NPP 13.4 (see docs/agent_packages/g5_notes.md and
// test/unit/diff/test_nppi_warp_perspective_{planar,quad,batch}_generated.cpp
// plus test/unit/diff/test_nppi_rotate_generated.cpp):
//
//   * NN     : src[floor(v + 0.5)][floor(u + 0.5)] (round half up).
//   * LINEAR : bilinear taps floor(u)/floor(v) with the usual (1-fx)(1-fy)
//              weights, accumulated in double.
//   * CUBIC  : four-point cubic *Lagrange* kernels on taps floor(u)-1..floor(u)+2
//              (and likewise in v).  The 1-D weights are
//                L(-1) = -d(d-1)(d-2)/6, L(0) = (d+1)(d-1)(d-2)/2,
//                L(+1) = -(d+1)d(d-2)/2, L(+2) = (d+1)d(d-1)/6,
//              recovered from the cubic-impulse goldens
//              (fwd/32f_P3R/cubw_x*, cubw_y*, cubw_xy*).
//   * Interpolation taps are clamped to the *image* bounds (o size - 1), not
//     to the ROI: the persp_subroi_cubic probes read from the image rows and
//     columns just outside a sub-rect ROI.
//   * Integer destinations round with floor(v + 0.5) and saturate; Npp16f
//     destinations round through __float2half.

#include "npp.h"
#include "nppi_warp_perspective_executor.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

namespace nppi {
namespace warp {

// Element conversion (loads go through float, like the NPP interpolators).
template <typename T> __device__ __forceinline__ float elemToFloat(T v) { return static_cast<float>(v); }

template <> __device__ __forceinline__ float elemToFloat<Npp16f>(Npp16f v) {
  return __half2float(__ushort_as_half(static_cast<unsigned short>(v.fp16)));
}

template <typename T> __device__ __forceinline__ T floatToElem(double v);

template <> __device__ __forceinline__ Npp8u floatToElem<Npp8u>(double v) {
  const double r = floor(v + 0.5);
  return static_cast<Npp8u>(r < 0.0 ? 0 : (r > 255.0 ? 255 : r));
}
template <> __device__ __forceinline__ Npp16u floatToElem<Npp16u>(double v) {
  const double r = floor(v + 0.5);
  return static_cast<Npp16u>(r < 0.0 ? 0 : (r > 65535.0 ? 65535 : r));
}
template <> __device__ __forceinline__ Npp16s floatToElem<Npp16s>(double v) {
  const double r = floor(v + 0.5);
  return static_cast<Npp16s>(r < -32768.0 ? -32768 : (r > 32767.0 ? 32767 : r));
}
template <> __device__ __forceinline__ Npp32s floatToElem<Npp32s>(double v) {
  const double r = floor(v + 0.5);
  return static_cast<Npp32s>(r < -2147483648.0 ? -2147483648.0 : (r > 2147483647.0 ? 2147483647.0 : r));
}
template <> __device__ __forceinline__ Npp32f floatToElem<Npp32f>(double v) { return static_cast<Npp32f>(v); }
template <> __device__ __forceinline__ Npp16f floatToElem<Npp16f>(double v) {
  Npp16f r;
  r.fp16 = static_cast<short>(__half_as_ushort(__float2half(static_cast<float>(v))));
  return r;
}

__device__ __forceinline__ int clampIndex(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// 1-D cubic Lagrange weights for taps floor-1 .. floor+2.
__device__ __forceinline__ void cubicWeights(double d, double w[4]) {
  w[0] = -d * (d - 1.0) * (d - 2.0) / 6.0;
  w[1] = (d + 1.0) * (d - 1.0) * (d - 2.0) / 2.0;
  w[2] = -(d + 1.0) * d * (d - 2.0) / 2.0;
  w[3] = (d + 1.0) * d * (d - 1.0) / 6.0;
}

// Interpolated value of one channel.  `base` points at element 0 of the
// channel inside the image, `step` is the row stride in elements, `chStride`
// is the distance between two pixels (1 for planar planes).
template <typename T>
__device__ __forceinline__ T sampleChannel(const T *base, int step, int chStride, int width, int height, double u,
                                           double v, int interp) {
  if (interp == NPPI_INTER_NN) {
    const int ix = clampIndex(static_cast<int>(floor(u + 0.5)), 0, width - 1);
    const int iy = clampIndex(static_cast<int>(floor(v + 0.5)), 0, height - 1);
    return base[(size_t)iy * step + (size_t)ix * chStride];
  }
  const int iu = static_cast<int>(floor(u));
  const int iv = static_cast<int>(floor(v));
  const double fx = u - iu;
  const double fy = v - iv;
  if (interp == NPPI_INTER_LINEAR) {
    const int x0 = clampIndex(iu, 0, width - 1);
    const int x1 = clampIndex(iu + 1, 0, width - 1);
    const int y0 = clampIndex(iv, 0, height - 1);
    const int y1 = clampIndex(iv + 1, 0, height - 1);
    const double v00 = elemToFloat(base[(size_t)y0 * step + (size_t)x0 * chStride]);
    const double v01 = elemToFloat(base[(size_t)y0 * step + (size_t)x1 * chStride]);
    const double v10 = elemToFloat(base[(size_t)y1 * step + (size_t)x0 * chStride]);
    const double v11 = elemToFloat(base[(size_t)y1 * step + (size_t)x1 * chStride]);
    const double acc = (1.0 - fx) * (1.0 - fy) * v00 + fx * (1.0 - fy) * v01 + (1.0 - fx) * fy * v10 +
                       fx * fy * v11;
    return floatToElem<T>(acc);
  }
  double wx[4], wy[4];
  cubicWeights(fx, wx);
  cubicWeights(fy, wy);
  double acc = 0.0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int yy = clampIndex(iv - 1 + j, 0, height - 1);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int xx = clampIndex(iu - 1 + i, 0, width - 1);
      acc += wx[i] * wy[j] * elemToFloat(base[(size_t)yy * step + (size_t)xx * chStride]);
    }
  }
  return floatToElem<T>(acc);
}

} // namespace warp
} // namespace nppi
