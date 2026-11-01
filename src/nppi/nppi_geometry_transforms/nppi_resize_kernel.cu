// Resize kernels shared by nppiResize, nppiResizeSqrPixel and
// nppiResizeBatch.  See nppi_resize_executor.h for the measured semantics.

#include "nppi_resize_executor.h"

#include <cuda_fp16.h>
#include <math_constants.h>

#include <cmath>
#include <limits>
#include <vector>

namespace nppi {
namespace resize {
namespace {

// ---------------------------------------------------------------------------
// Element access
// ---------------------------------------------------------------------------
template <typename T> struct Traits {
  using VT = float;
};
template <> struct Traits<Npp32f> {
  using VT = float;
};
template <> struct Traits<Npp64f> {
  using VT = double;
};
template <> struct Traits<Npp16f> {
  using VT = float;
};

template <typename T, typename VT> __device__ __forceinline__ VT loadVal(const T *p) {
  if constexpr (std::is_same_v<T, Npp16f>) {
    return __half2float(*reinterpret_cast<const __half *>(p));
  } else {
    return static_cast<VT>(*p);
  }
}

template <typename T, typename VT> __device__ __forceinline__ void storeVal(T *p, VT v) {
  if constexpr (std::is_same_v<T, Npp16f>) {
    *reinterpret_cast<__half *>(p) = __float2half_rn(static_cast<float>(v));
  } else if constexpr (std::is_same_v<T, Npp32f>) {
    *p = static_cast<float>(v);
  } else if constexpr (std::is_same_v<T, Npp64f>) {
    *p = static_cast<double>(v);
  } else if constexpr (std::is_same_v<T, Npp8u>) {
    VT r = v >= (VT)0 ? floor(v + (VT)0.5) : ceil(v - (VT)0.5);
    r = r < (VT)0 ? (VT)0 : r;
    r = r > (VT)255 ? (VT)255 : r;
    *p = static_cast<Npp8u>(r);
  } else if constexpr (std::is_same_v<T, Npp16u>) {
    VT r = v >= (VT)0 ? floor(v + (VT)0.5) : ceil(v - (VT)0.5);
    r = r < (VT)0 ? (VT)0 : r;
    r = r > (VT)65535 ? (VT)65535 : r;
    *p = static_cast<Npp16u>(r);
  } else if constexpr (std::is_same_v<T, Npp16s>) {
    VT r = v >= (VT)0 ? floor(v + (VT)0.5) : ceil(v - (VT)0.5);
    r = r < (VT)-32768 ? (VT)-32768 : r;
    r = r > (VT)32767 ? (VT)32767 : r;
    *p = static_cast<Npp16s>(r);
  } else {
    *p = static_cast<T>(v);
  }
}

// ---------------------------------------------------------------------------
// Interpolation kernels
// ---------------------------------------------------------------------------
enum Mode {
  MODE_NN = 0,
  MODE_LINEAR = 1,
  MODE_CUBIC = 2,
  MODE_LANCZOS = 3,
  MODE_SUPER = 4,
};



// 4-point Lagrange cubic stencil {x0-1, x0, x0+1, x0+2} (measured baseline).
__device__ __forceinline__ void lagrange4(double t, double w[4]) {
  w[0] = -t * (t - 1.0) * (t - 2.0) / 6.0;
  w[1] = (t + 1.0) * (t - 1.0) * (t - 2.0) / 2.0;
  w[2] = -t * (t + 1.0) * (t - 2.0) / 2.0;
  w[3] = t * (t + 1.0) * (t - 1.0) / 6.0;
}

__device__ __forceinline__ double lanczos3(double d) {
  double a = fabs(d);
  if (a >= 3.0) return 0.0;
  if (a < 1e-12) return 1.0;
  const double pi = 3.14159265358979323846;
  double x = pi * a;
  return (sin(x) / x) * (sin(x / 3.0) / (x / 3.0));
}

template <typename T, int NC, typename VT>
__device__ __forceinline__ bool samplePixel(const T *src, int sStep, int srcW, int srcH, VT sx, VT sy, int mode,
                                            VT adjF, VT adjFy, int tapLoX, int tapHiX, int tapLoY, int tapHiY,
                                            VT out[NC]) {
  const int stride = sStep / (int)sizeof(T);
  if (mode == MODE_NN) {
    int ix = (int)floor((double)sx + 0.5);
    int iy = (int)floor((double)sy + 0.5);
    ix = ix < tapLoX ? tapLoX : (ix > tapHiX ? tapHiX : ix);
    iy = iy < tapLoY ? tapLoY : (iy > tapHiY ? tapHiY : iy);
    const T *p = src + (size_t)iy * stride + (size_t)ix * NC;
    for (int c = 0; c < NC; ++c) out[c] = loadVal<T, VT>(p + c);
    return true;
  }
  if (mode == MODE_LINEAR || mode == MODE_CUBIC || mode == MODE_LANCZOS) {
    const int taps = (mode == MODE_LINEAR) ? 2 : (mode == MODE_CUBIC ? 4 : 6);
    const int base = (int)floor((double)sx);
    const int baseY = (int)floor((double)sy);
    double wx[6], wy[6];
    int ox[6], oy[6];
    if (mode == MODE_LINEAR) {
      double tx = (double)sx - base;
      double ty = (double)sy - baseY;
      wx[0] = 1.0 - tx;
      wx[1] = tx;
      wy[0] = 1.0 - ty;
      wy[1] = ty;
      for (int i = 0; i < 2; ++i) {
        ox[i] = base + i;
        oy[i] = baseY + i;
      }
    } else if (mode == MODE_CUBIC) {
      lagrange4((double)sx - base, wx);
      lagrange4((double)sy - baseY, wy);
      for (int i = 0; i < 4; ++i) {
        ox[i] = base - 1 + i;
        oy[i] = baseY - 1 + i;
      }
    } else {
      for (int i = 0; i < 6; ++i) {
        ox[i] = base - 2 + i;
        oy[i] = baseY - 2 + i;
        wx[i] = lanczos3((double)ox[i] - (double)sx);
        wy[i] = lanczos3((double)oy[i] - (double)sy);
      }
    }
    VT acc[NC];
    for (int c = 0; c < NC; ++c) acc[c] = (VT)0;
    double wsum = 0.0;
    for (int j = 0; j < taps; ++j) {
      int yy = oy[j];
      yy = yy < tapLoY ? tapLoY : (yy > tapHiY ? tapHiY : yy);
      const T *row = src + (size_t)yy * stride;
      for (int i = 0; i < taps; ++i) {
        int xx = ox[i];
        xx = xx < tapLoX ? tapLoX : (xx > tapHiX ? tapHiX : xx);
        const T *p = row + (size_t)xx * NC;
        const double w = wx[i] * wy[j];
        wsum += w;
        for (int c = 0; c < NC; ++c) acc[c] += (VT)(w * (double)loadVal<T, VT>(p + c));
      }
    }
    // The truncated Lanczos-3 weights do not sum to one; NVIDIA normalises by
    // the (clamped) weight sum (verified against the 32f goldens).
    if (mode == MODE_LANCZOS && wsum != 0.0) {
      for (int c = 0; c < NC; ++c) acc[c] = acc[c] / (VT)wsum;
    }
    for (int c = 0; c < NC; ++c) out[c] = acc[c];
    return true;
  }
  // SUPER: area average over a source-space box of width adjF starting at sx-0.5.
  double x0 = (double)sx - 0.5, y0 = (double)sy - 0.5;
  double x1 = x0 + (double)adjF, y1 = y0 + (double)adjFy;
  int ix0 = (int)floor(x0), ix1 = (int)ceil(x1) - 1;
  int iy0 = (int)floor(y0), iy1 = (int)ceil(y1) - 1;
  double sumW = 0.0;
  VT acc[NC];
  for (int c = 0; c < NC; ++c) acc[c] = (VT)0;
  for (int yy = iy0; yy <= iy1; ++yy) {
    double oy = fmin(y1, (double)yy + 1.0) - fmax(y0, (double)yy);
    if (oy <= 0.0) continue;
    int yc = yy < tapLoY ? tapLoY : (yy > tapHiY ? tapHiY : yy);
    const T *row = src + (size_t)yc * stride;
    for (int xx = ix0; xx <= ix1; ++xx) {
      double ox = fmin(x1, (double)xx + 1.0) - fmax(x0, (double)xx);
      if (ox <= 0.0) continue;
      double w = ox * oy;
      sumW += w;
      int xc = xx < tapLoX ? tapLoX : (xx > tapHiX ? tapHiX : xx);
      const T *p = row + (size_t)xc * NC;
      for (int c = 0; c < NC; ++c) acc[c] += (VT)(w * (double)loadVal<T, VT>(p + c));
    }
  }
  if (sumW <= 0.0) sumW = 1.0;
  for (int c = 0; c < NC; ++c) out[c] = acc[c] / (VT)sumW;
  return true;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------
template <typename T, int NC, int PROC, typename VT>
__global__ void resizeKernel(const T *__restrict__ src, int sStep, int srcW, int srcH, int clipHiX, int clipHiY, T *__restrict__ dst, int dStep, NppiRect dROI, VT afx,
                             VT asx, VT afy, VT asy, VT clipLoX, VT clipLoY, VT coordLoX, VT coordHiX, VT coordLoY,
                             VT coordHiY, int tapLoX, int tapHiX, int tapLoY, int tapHiY, int mode) {
  int dx = dROI.x + blockIdx.x * blockDim.x + threadIdx.x;
  int dy = dROI.y + blockIdx.y * blockDim.y + threadIdx.y;
  if (dx >= dROI.x + dROI.width || dy >= dROI.y + dROI.height) return;
  VT sx, sy;
  if constexpr (std::is_same_v<VT, double>) {
    sx = afx * (VT)dx - asx;
    sy = afy * (VT)dy - asy;
  } else {
    sx = fmaf(afx, (VT)dx, -asx);
    sy = fmaf(afy, (VT)dy, -asy);
  }
  if (!(sx > clipLoX && sx < (VT)clipHiX && sy > clipLoY && sy < (VT)clipHiY)) return;
  sx = sx < coordLoX ? coordLoX : (sx > coordHiX ? coordHiX : sx);
  sy = sy < coordLoY ? coordLoY : (sy > coordHiY ? coordHiY : sy);
  VT v[NC];
  if (!samplePixel<T, NC, VT>(src, sStep, srcW, srcH, sx, sy, mode, afx, afy, tapLoX, tapHiX, tapLoY, tapHiY, v))
    return;
  const int dstride = dStep / (int)sizeof(T);
  T *p = dst + (size_t)dy * dstride + (size_t)dx * NC;
  // AC4 leaves the alpha channel untouched (measured).
  for (int c = 0; c < PROC; ++c) storeVal<T, VT>(p + c, v[c]);
}

// Planar: one lane handles one plane independently.
template <typename T, int NC, typename VT>
__global__ void resizePlanarKernel(const T *__restrict__ src, int sStep, int srcW, int srcH, int clipHiX, int clipHiY, T *__restrict__ dst, int dStep, NppiRect dROI, VT afx,
                                   VT asx, VT afy, VT asy, VT clipLoX, VT clipLoY, VT coordLoX, VT coordHiX, VT coordLoY,
                             VT coordHiY, int tapLoX, int tapHiX, int tapLoY, int tapHiY, int mode) {
  int dx = dROI.x + blockIdx.x * blockDim.x + threadIdx.x;
  int dy = dROI.y + blockIdx.y * blockDim.y + threadIdx.y;
  if (dx >= dROI.x + dROI.width || dy >= dROI.y + dROI.height) return;
  VT sx, sy;
  if constexpr (std::is_same_v<VT, double>) {
    sx = afx * (VT)dx - asx;
    sy = afy * (VT)dy - asy;
  } else {
    sx = fmaf(afx, (VT)dx, -asx);
    sy = fmaf(afy, (VT)dy, -asy);
  }
  if (!(sx > clipLoX && sx < (VT)clipHiX && sy > clipLoY && sy < (VT)clipHiY)) return;
  sx = sx < coordLoX ? coordLoX : (sx > coordHiX ? coordHiX : sx);
  sy = sy < coordLoY ? coordLoY : (sy > coordHiY ? coordHiY : sy);
  VT v[1];
  if (!samplePixel<T, 1, VT>(src, sStep, srcW, srcH, sx, sy, mode, afx, afy, tapLoX, tapHiX, tapLoY, tapHiY, v))
    return;
  const int dstride = dStep / (int)sizeof(T);
  storeVal<T, VT>(dst + (size_t)dy * dstride + (size_t)dx, v[0]);
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------
template <typename T, typename VT> VT coordFactor(double f) { return (VT)(1.0 / f); }

inline bool interpSupported(int eInterpolation) {
  switch (eInterpolation) {
    case NPPI_INTER_NN:
    case NPPI_INTER_LINEAR:
    case NPPI_INTER_CUBIC:
    case NPPI_INTER_LANCZOS:
    case NPPI_INTER_SUPER:
      return true;
    default:
      return false;
  }
}

inline int toMode(int eInterpolation) {
  switch (eInterpolation) {
    case NPPI_INTER_NN: return MODE_NN;
    case NPPI_INTER_LINEAR: return MODE_LINEAR;
    case NPPI_INTER_CUBIC: return MODE_CUBIC;
    case NPPI_INTER_LANCZOS: return MODE_LANCZOS;
    default: return MODE_SUPER;
  }
}

struct ClipRect {
  int x, y, w, h;
};

inline ClipRect intersectROI(NppiRect roi, NppiSize size) {
  int x0 = roi.x > 0 ? roi.x : 0;
  int y0 = roi.y > 0 ? roi.y : 0;
  int x1 = roi.x + roi.width;
  int y1 = roi.y + roi.height;
  if (x1 > size.width) x1 = size.width;
  if (y1 > size.height) y1 = size.height;
  ClipRect c;
  c.x = x0;
  c.y = y0;
  c.w = x1 - x0;
  c.h = y1 - y0;
  return c;
}

template <typename T, typename VT>
static void launchKernel(const T *src, int sStep, NppiSize srcSize, ClipRect clip, T *dst, int dStep, NppiRect dROI,
                         VT afx, VT asx, VT afy, VT asy, VT clipLoX, VT clipLoY, int tapLoX, int tapHiX, int tapLoY,
                         int tapHiY, int mode, int nc, bool planar, bool clampCoord, int proc,
                         cudaStream_t stream) {
  // nppiResize clamps the sample coordinate into the ROI intersection before
  // sampling (measured); SqrPixel does not clamp the coordinate at all.
  const VT big = (VT)1e30;
  // nppiResize clamps the interpolation taps to [ROI start, ROI end] where the
  // end is inclusive of the pixel just past the ROI rectangle (measured).
  const VT coordLo = clampCoord ? (VT)tapLoX : -big;
  const VT coordHi = clampCoord ? (VT)tapHiX : big;
  const VT coordLoY = clampCoord ? (VT)tapLoY : -big;
  const VT coordHiY = clampCoord ? (VT)tapHiY : big;
  if (dROI.width <= 0 || dROI.height <= 0) return;
  dim3 block(16, 16);
  dim3 grid((dROI.width + block.x - 1) / block.x, (dROI.height + block.y - 1) / block.y);
  if (planar) {
    if (nc == 3) {
      resizePlanarKernel<T, 3, VT><<<grid, block, 0, stream>>>(src, sStep, srcSize.width, srcSize.height, clip.x + clip.w,
                                                              clip.y + clip.h, dst, dStep, dROI, afx, asx, afy,
                                                              asy, clipLoX, clipLoY, coordLo, coordHi, coordLoY, coordHiY, tapLoX, tapHiX, tapLoY, tapHiY, mode);
    } else {
      resizePlanarKernel<T, 4, VT><<<grid, block, 0, stream>>>(src, sStep, srcSize.width, srcSize.height, clip.x + clip.w,
                                                              clip.y + clip.h, dst, dStep, dROI, afx, asx, afy,
                                                              asy, clipLoX, clipLoY, coordLo, coordHi, coordLoY, coordHiY, tapLoX, tapHiX, tapLoY, tapHiY, mode);
    }
    return;
  }
  switch (nc) {
    case 1:
      resizeKernel<T, 1, 1, VT><<<grid, block, 0, stream>>>(src, sStep, srcSize.width, srcSize.height,
                                                         clip.x + clip.w, clip.y + clip.h, dst, dStep, dROI, afx,
                                                         asx, afy, asy, clipLoX, clipLoY, coordLo, coordHi, coordLoY, coordHiY, tapLoX, tapHiX, tapLoY, tapHiY, mode);
      break;
    case 3:
      resizeKernel<T, 3, 3, VT><<<grid, block, 0, stream>>>(src, sStep, srcSize.width, srcSize.height,
                                                         clip.x + clip.w, clip.y + clip.h, dst, dStep, dROI, afx,
                                                         asx, afy, asy, clipLoX, clipLoY, coordLo, coordHi, coordLoY, coordHiY, tapLoX, tapHiX, tapLoY, tapHiY, mode);
      break;
    default:
      if (proc == 3) {
        resizeKernel<T, 4, 3, VT><<<grid, block, 0, stream>>>(src, sStep, srcSize.width, srcSize.height,
                                                              clip.x + clip.w, clip.y + clip.h, dst, dStep, dROI,
                                                              afx, asx, afy, asy, clipLoX, clipLoY, coordLo, coordHi, coordLoY, coordHiY, tapLoX, tapHiX, tapLoY, tapHiY, mode);
      } else {
        resizeKernel<T, 4, 4, VT><<<grid, block, 0, stream>>>(src, sStep, srcSize.width, srcSize.height,
                                                              clip.x + clip.w, clip.y + clip.h, dst, dStep, dROI,
                                                              afx, asx, afy, asy, clipLoX, clipLoY, coordLo, coordHi, coordLoY, coordHiY, tapLoX, tapHiX, tapLoY, tapHiY, mode);
      }
      break;
  }
}

template <typename T, typename VT>
static NppStatus runSqrPixel(const T *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, T *pDst, int nDstStep,
                             NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift,
                             int eInterpolation, int nc, int proc, cudaStream_t stream) {
  if (nXFactor <= 0.0 || nYFactor <= 0.0) return NPP_RESIZE_FACTOR_ERROR;
  if (pSrc == nullptr || pDst == nullptr) return NPP_NULL_POINTER_ERROR;
  if (oSrcSize.width < 0 || oSrcSize.height < 0) return NPP_SIZE_ERROR;
  if (oSrcSize.width == 0 || oSrcSize.height == 0) return NPP_NO_ERROR;
  if (oSrcSize.width < 2 || oSrcSize.height < 2) return NPP_RECTANGLE_ERROR;
  if (nSrcStep < oSrcSize.width * nc * (int)sizeof(T)) return NPP_STEP_ERROR;
  if (oDstROI.width < 0 || oDstROI.height < 0) return NPP_SIZE_ERROR;
  if (oDstROI.width == 0 || oDstROI.height == 0) return NPP_NO_ERROR;
  if (oSrcROI.width <= 0 || oSrcROI.height <= 0) return NPP_RECTANGLE_ERROR;
  ClipRect clip = intersectROI(oSrcROI, oSrcSize);
  if (clip.w <= 0 || clip.h <= 0) return NPP_WRONG_INTERSECTION_ROI_WARNING;
  if (!interpSupported(eInterpolation)) return NPP_INTERPOLATION_ERROR;
  if (eInterpolation == NPPI_INTER_SUPER && (nXFactor > 1.0 || nYFactor > 1.0)) return NPP_RESIZE_FACTOR_ERROR;
  if (nDstStep < (oDstROI.x + oDstROI.width) * nc * (int)sizeof(T)) return NPP_STEP_ERROR;
  const VT afx = (VT)(1.0 / nXFactor);
  const VT asx = (VT)(nXShift * (1.0 / nXFactor) + (1.0 - 1.0 / nXFactor) * 0.5);
  const VT afy = (VT)(1.0 / nYFactor);
  const VT asy = (VT)(nYShift * (1.0 / nYFactor) + (1.0 - 1.0 / nYFactor) * 0.5);
  launchKernel<T, VT>(pSrc, nSrcStep, oSrcSize, clip, pDst, nDstStep, oDstROI, afx, asx, afy, asy,
                      (VT)(oSrcROI.x - 0.5 * nXFactor), (VT)(oSrcROI.y - 0.5 * nYFactor), 0, oSrcSize.width - 1, 0,
                      oSrcSize.height - 1, toMode(eInterpolation), nc, false, false, proc, stream);
  return NPP_NO_ERROR;
}

template <typename T, typename VT>
static NppStatus runResize(const T *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcROI, T *pDst, int nDstStep,
                           NppiSize oDstSize, NppiRect oDstROI, int eInterpolation, int nc, int proc,
                           cudaStream_t stream) {
  (void)oDstSize;
  if (pSrc == nullptr || pDst == nullptr) return NPP_NULL_POINTER_ERROR;
  if (nSrcStep < oSrcSize.width * nc * (int)sizeof(T)) return NPP_STEP_ERROR;
  if (nDstStep < oDstSize.width * nc * (int)sizeof(T)) return NPP_STEP_ERROR;
  if (oSrcSize.width < 2 || oSrcSize.height < 2) return NPP_RESIZE_NO_OPERATION_ERROR;
  if (oSrcROI.width < 1 || oSrcROI.height < 1) return NPP_RESIZE_NO_OPERATION_ERROR;
  if (oDstSize.width < 1 || oDstSize.height < 1) return NPP_RESIZE_NO_OPERATION_ERROR;
  if (oDstROI.width < 1 || oDstROI.height < 1) return NPP_RESIZE_NO_OPERATION_ERROR;
  ClipRect clip = intersectROI(oSrcROI, oSrcSize);
  if (clip.w <= 0 || clip.h <= 0) return NPP_RESIZE_NO_OPERATION_ERROR;
  if (!interpSupported(eInterpolation)) return NPP_INTERPOLATION_ERROR;
  const double fx = (double)oDstROI.width / (double)oSrcROI.width;
  const double fy = (double)oDstROI.height / (double)oSrcROI.height;
  if (eInterpolation == NPPI_INTER_SUPER && (fx > 1.0 || fy > 1.0)) return NPP_RESIZE_FACTOR_ERROR;
  const VT afx = (VT)((double)oSrcROI.width / (double)oDstROI.width);
  const VT asx = (VT)(-(double)oSrcROI.x + 0.5 + ((double)oDstROI.x - 0.5) * (double)afx);
  const VT afy = (VT)((double)oSrcROI.height / (double)oDstROI.height);
  const VT asy = (VT)(-(double)oSrcROI.y + 0.5 + ((double)oDstROI.y - 0.5) * (double)afy);
  launchKernel<T, VT>(pSrc, nSrcStep, oSrcSize, clip, pDst, nDstStep, oDstROI, afx, asx, afy, asy,
                      (VT)(clip.x - 0.5), (VT)(clip.y - 0.5), clip.x,
                      (clip.x + clip.w < oSrcSize.width - 1 ? clip.x + clip.w : oSrcSize.width - 1), clip.y,
                      (clip.y + clip.h < oSrcSize.height - 1 ? clip.y + clip.h : oSrcSize.height - 1),
                      toMode(eInterpolation), nc, false, false, proc, stream);
  return NPP_NO_ERROR;
}

template <typename T, typename VT>
static NppStatus runSqrPixelPlanar(const T *const pSrc[], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcROI,
                                   T *const pDst[], int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor,
                                   double nXShift, double nYShift, int eInterpolation, int nc, cudaStream_t stream) {
  if (nXFactor <= 0.0 || nYFactor <= 0.0) return NPP_RESIZE_FACTOR_ERROR;
  if (pSrc == nullptr || pDst == nullptr) return NPP_NULL_POINTER_ERROR;
  for (int p = 0; p < nc; ++p) {
    if (pSrc[p] == nullptr || pDst[p] == nullptr) return NPP_NULL_POINTER_ERROR;
  }
  if (oSrcSize.width < 0 || oSrcSize.height < 0) return NPP_SIZE_ERROR;
  if (oSrcSize.width == 0 || oSrcSize.height == 0) return NPP_NO_ERROR;
  if (oSrcSize.width < 2 || oSrcSize.height < 2) return NPP_RECTANGLE_ERROR;
  if (nSrcStep < oSrcSize.width * (int)sizeof(T)) return NPP_STEP_ERROR;
  if (oDstROI.width < 0 || oDstROI.height < 0) return NPP_SIZE_ERROR;
  if (oDstROI.width == 0 || oDstROI.height == 0) return NPP_NO_ERROR;
  if (oSrcROI.width <= 0 || oSrcROI.height <= 0) return NPP_RECTANGLE_ERROR;
  ClipRect clip = intersectROI(oSrcROI, oSrcSize);
  if (clip.w <= 0 || clip.h <= 0) return NPP_WRONG_INTERSECTION_ROI_WARNING;
  if (!interpSupported(eInterpolation)) return NPP_INTERPOLATION_ERROR;
  if (eInterpolation == NPPI_INTER_SUPER && (nXFactor > 1.0 || nYFactor > 1.0)) return NPP_RESIZE_FACTOR_ERROR;
  if (nDstStep < (oDstROI.x + oDstROI.width) * (int)sizeof(T)) return NPP_STEP_ERROR;
  const VT afx = (VT)(1.0 / nXFactor);
  const VT asx = (VT)(nXShift * (1.0 / nXFactor) + (1.0 - 1.0 / nXFactor) * 0.5);
  const VT afy = (VT)(1.0 / nYFactor);
  const VT asy = (VT)(nYShift * (1.0 / nYFactor) + (1.0 - 1.0 / nYFactor) * 0.5);
  for (int p = 0; p < nc; ++p) {
    launchKernel<T, VT>(pSrc[p], nSrcStep, oSrcSize, clip, pDst[p], nDstStep, oDstROI, afx, asx, afy, asy,
                        (VT)(oSrcROI.x - 0.5 * nXFactor), (VT)(oSrcROI.y - 0.5 * nYFactor), 0, oSrcSize.width - 1, 0,
                        oSrcSize.height - 1, toMode(eInterpolation), 1, true, false, 1, stream);
  }
  return NPP_NO_ERROR;
}

template <typename T, typename VT>
static NppStatus runResizePlanar(const T *const pSrc[], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcROI,
                                 T *const pDst[], int nDstStep, NppiSize oDstSize, NppiRect oDstROI,
                                 int eInterpolation, int nc, cudaStream_t stream) {
  if (pSrc == nullptr || pDst == nullptr) return NPP_NULL_POINTER_ERROR;
  for (int p = 0; p < nc; ++p) {
    if (pSrc[p] == nullptr || pDst[p] == nullptr) return NPP_NULL_POINTER_ERROR;
  }
  if (nSrcStep < oSrcSize.width * (int)sizeof(T)) return NPP_STEP_ERROR;
  if (nDstStep < oDstSize.width * (int)sizeof(T)) return NPP_STEP_ERROR;
  if (oSrcSize.width < 2 || oSrcSize.height < 2) return NPP_RESIZE_NO_OPERATION_ERROR;
  if (oSrcROI.width < 1 || oSrcROI.height < 1) return NPP_RESIZE_NO_OPERATION_ERROR;
  if (oDstSize.width < 1 || oDstSize.height < 1) return NPP_RESIZE_NO_OPERATION_ERROR;
  if (oDstROI.width < 1 || oDstROI.height < 1) return NPP_RESIZE_NO_OPERATION_ERROR;
  ClipRect clip = intersectROI(oSrcROI, oSrcSize);
  if (clip.w <= 0 || clip.h <= 0) return NPP_RESIZE_NO_OPERATION_ERROR;
  if (!interpSupported(eInterpolation)) return NPP_INTERPOLATION_ERROR;
  const double fx = (double)oDstROI.width / (double)oSrcROI.width;
  const double fy = (double)oDstROI.height / (double)oSrcROI.height;
  if (eInterpolation == NPPI_INTER_SUPER && (fx > 1.0 || fy > 1.0)) return NPP_RESIZE_FACTOR_ERROR;
  const VT afx = (VT)((double)oSrcROI.width / (double)oDstROI.width);
  const VT asx = (VT)(-(double)oSrcROI.x + 0.5 + ((double)oDstROI.x - 0.5) * (double)afx);
  const VT afy = (VT)((double)oSrcROI.height / (double)oDstROI.height);
  const VT asy = (VT)(-(double)oSrcROI.y + 0.5 + ((double)oDstROI.y - 0.5) * (double)afy);
  for (int p = 0; p < nc; ++p) {
    launchKernel<T, VT>(pSrc[p], nSrcStep, oSrcSize, clip, pDst[p], nDstStep, oDstROI, afx, asx, afy, asy,
                        (VT)(clip.x - 0.5), (VT)(clip.y - 0.5), clip.x,
                        (clip.x + clip.w < oSrcSize.width - 1 ? clip.x + clip.w : oSrcSize.width - 1), clip.y,
                        (clip.y + clip.h < oSrcSize.height - 1 ? clip.y + clip.h : oSrcSize.height - 1),
                        toMode(eInterpolation), 1, true, false, 1, stream);
  }
  return NPP_NO_ERROR;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public template definitions
// ---------------------------------------------------------------------------
template <typename T, int C, bool AC4>
NppStatus sqrPixel(const T *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, T *pDst, int nDstStep,
                   NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift,
                   int eInterpolation, cudaStream_t stream) {
  using VT = typename Traits<T>::VT;
  return runSqrPixel<T, VT>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift,
                            nYShift, eInterpolation, C, AC4 ? 3 : C, stream);
}

template <typename T, int C>
NppStatus sqrPixelPlanar(const T *const pSrc[], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcROI, T *const pDst[],
                         int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift,
                         double nYShift, int eInterpolation, cudaStream_t stream) {
  using VT = typename Traits<T>::VT;
  return runSqrPixelPlanar<T, VT>(pSrc, nSrcStep, oSrcSize, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor,
                                  nXShift, nYShift, eInterpolation, C, stream);
}

template <typename T, int C, bool AC4>
NppStatus sqrPixelAdvanced(const T *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, T *pDst, int nDstStep,
                           NppiRect oDstROI, double nXFactor, double nYFactor, T *pBuffer, int eInterpolationMode,
                           cudaStream_t stream) {
  if (eInterpolationMode != NPPI_INTER_LANCZOS3_ADVANCED) return NPP_NOT_IMPLEMENTED_ERROR;
  if (pBuffer == nullptr) return NPP_NULL_POINTER_ERROR;
  using VT = typename Traits<T>::VT;
  return runSqrPixel<T, VT>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, 0.0, 0.0,
                            NPPI_INTER_LANCZOS, C, AC4 ? 3 : C, stream);
}

template <typename T, int C, bool AC4>
NppStatus resize(const T *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcROI, T *pDst, int nDstStep,
                 NppiSize oDstSize, NppiRect oDstROI, int eInterpolation, cudaStream_t stream) {
  using VT = typename Traits<T>::VT;
  return runResize<T, VT>(pSrc, nSrcStep, oSrcSize, oSrcROI, pDst, nDstStep, oDstSize, oDstROI, eInterpolation, C,
                          AC4 ? 3 : C, stream);
}

template <typename T, int C>
NppStatus resizePlanar(const T *const pSrc[], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcROI, T *const pDst[],
                       int nDstStep, NppiSize oDstSize, NppiRect oDstROI, int eInterpolation, cudaStream_t stream) {
  using VT = typename Traits<T>::VT;
  return runResizePlanar<T, VT>(pSrc, nSrcStep, oSrcSize, oSrcROI, pDst, nDstStep, oDstSize, oDstROI, eInterpolation,
                                C, stream);
}

template <typename T, int C, bool AC4>
NppStatus resizeBatch(NppiSize oSmallestSrcSize, NppiRect oSrcROI, NppiSize oSmallestDstSize, NppiRect oDstROI,
                      int eInterpolation, NppiResizeBatchCXR *pBatchList, unsigned int nBatchSize,
                      cudaStream_t stream) {
  if (pBatchList == nullptr) return NPP_NULL_POINTER_ERROR;
  if (nBatchSize == 0) return NPP_RESIZE_NO_OPERATION_ERROR;
  std::vector<NppiResizeBatchCXR> host(nBatchSize);
  if (cudaMemcpyAsync(host.data(), pBatchList, nBatchSize * sizeof(NppiResizeBatchCXR), cudaMemcpyDeviceToHost,
                      stream) != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  for (unsigned int i = 0; i < nBatchSize; ++i) {
    NppStatus st = resize<T, C, AC4>(static_cast<const T *>(host[i].pSrc), host[i].nSrcStep, oSmallestSrcSize, oSrcROI,
                                     static_cast<T *>(host[i].pDst), host[i].nDstStep, oSmallestDstSize, oDstROI,
                                     eInterpolation, stream);
    if (st != NPP_NO_ERROR) return st;
  }
  return NPP_NO_ERROR;
}

template <typename T, int C, bool AC4>
NppStatus resizeBatchAdvanced(int nMaxWidth, int nMaxHeight, NppiImageDescriptor *pBatchSrc,
                              NppiImageDescriptor *pBatchDst, NppiResizeBatchROI_Advanced *pBatchROI,
                              unsigned int nBatchSize, int eInterpolation, cudaStream_t stream) {
  (void)nMaxWidth;
  (void)nMaxHeight;
  if (pBatchSrc == nullptr || pBatchDst == nullptr || pBatchROI == nullptr) return NPP_NULL_POINTER_ERROR;
  if (nBatchSize == 0) return NPP_RESIZE_NO_OPERATION_ERROR;
  std::vector<NppiImageDescriptor> src(nBatchSize), dst(nBatchSize);
  std::vector<NppiResizeBatchROI_Advanced> roi(nBatchSize);
  if (cudaMemcpyAsync(src.data(), pBatchSrc, nBatchSize * sizeof(NppiImageDescriptor), cudaMemcpyDeviceToHost,
                      stream) != cudaSuccess ||
      cudaMemcpyAsync(dst.data(), pBatchDst, nBatchSize * sizeof(NppiImageDescriptor), cudaMemcpyDeviceToHost,
                      stream) != cudaSuccess ||
      cudaMemcpyAsync(roi.data(), pBatchROI, nBatchSize * sizeof(NppiResizeBatchROI_Advanced),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  for (unsigned int i = 0; i < nBatchSize; ++i) {
    NppStatus st = resize<T, C, AC4>(static_cast<const T *>(src[i].pData), src[i].nStep, src[i].oSize,
                                     roi[i].oSrcRectROI, static_cast<T *>(dst[i].pData), dst[i].nStep, dst[i].oSize,
                                     roi[i].oDstRectROI, eInterpolation, stream);
    if (st != NPP_NO_ERROR) return st;
  }
  return NPP_NO_ERROR;
}

// ---------------------------------------------------------------------------
// Explicit instantiations (keep in sync with the generated wrappers)
// ---------------------------------------------------------------------------
#define NPPI_RESIZE_INSTANTIATE_T(T)                                                        \
  template NppStatus sqrPixel<T, 1, false>(const T *, NppiSize, int, NppiRect, T *, int, NppiRect, double, double, \
                                           double, double, int, cudaStream_t);          \
  template NppStatus sqrPixel<T, 3, false>(const T *, NppiSize, int, NppiRect, T *, int, NppiRect, double, double, \
                                           double, double, int, cudaStream_t);          \
  template NppStatus sqrPixel<T, 4, false>(const T *, NppiSize, int, NppiRect, T *, int, NppiRect, double, double, \
                                           double, double, int, cudaStream_t);          \
  template NppStatus sqrPixel<T, 4, true>(const T *, NppiSize, int, NppiRect, T *, int, NppiRect, double, double,  \
                                          double, double, int, cudaStream_t);           \
  template NppStatus resize<T, 1, false>(const T *, int, NppiSize, NppiRect, T *, int, NppiSize, NppiRect, int,    \
                                         cudaStream_t);                                 \
  template NppStatus resize<T, 3, false>(const T *, int, NppiSize, NppiRect, T *, int, NppiSize, NppiRect, int,    \
                                         cudaStream_t);                                 \
  template NppStatus resize<T, 4, false>(const T *, int, NppiSize, NppiRect, T *, int, NppiSize, NppiRect, int,    \
                                         cudaStream_t);                                 \
  template NppStatus resize<T, 4, true>(const T *, int, NppiSize, NppiRect, T *, int, NppiSize, NppiRect, int,     \
                                        cudaStream_t);

NPPI_RESIZE_INSTANTIATE_T(Npp8u)
NPPI_RESIZE_INSTANTIATE_T(Npp16u)
NPPI_RESIZE_INSTANTIATE_T(Npp16s)
NPPI_RESIZE_INSTANTIATE_T(Npp16f)
NPPI_RESIZE_INSTANTIATE_T(Npp32f)
NPPI_RESIZE_INSTANTIATE_T(Npp64f)

#define NPPI_RESIZE_INSTANTIATE_PLANAR(T)                                                         \
  template NppStatus sqrPixelPlanar<T, 3>(const T *const[], int, NppiSize, NppiRect, T *const[], int, NppiRect, \
                                          double, double, double, double, int, cudaStream_t);   \
  template NppStatus sqrPixelPlanar<T, 4>(const T *const[], int, NppiSize, NppiRect, T *const[], int, NppiRect, \
                                          double, double, double, double, int, cudaStream_t);   \
  template NppStatus resizePlanar<T, 3>(const T *const[], int, NppiSize, NppiRect, T *const[], int, NppiSize,   \
                                        NppiRect, int, cudaStream_t);                            \
  template NppStatus resizePlanar<T, 4>(const T *const[], int, NppiSize, NppiRect, T *const[], int, NppiSize,   \
                                        NppiRect, int, cudaStream_t);

NPPI_RESIZE_INSTANTIATE_PLANAR(Npp8u)
NPPI_RESIZE_INSTANTIATE_PLANAR(Npp16u)
NPPI_RESIZE_INSTANTIATE_PLANAR(Npp16s)
NPPI_RESIZE_INSTANTIATE_PLANAR(Npp32f)
NPPI_RESIZE_INSTANTIATE_PLANAR(Npp64f)

#define NPPI_RESIZE_INSTANTIATE_BATCH(T)                                                            \
  template NppStatus resizeBatch<T, 1, false>(NppiSize, NppiRect, NppiSize, NppiRect, int,          \
                                              NppiResizeBatchCXR *, unsigned int, cudaStream_t);    \
  template NppStatus resizeBatch<T, 3, false>(NppiSize, NppiRect, NppiSize, NppiRect, int,          \
                                              NppiResizeBatchCXR *, unsigned int, cudaStream_t);    \
  template NppStatus resizeBatch<T, 4, false>(NppiSize, NppiRect, NppiSize, NppiRect, int,          \
                                              NppiResizeBatchCXR *, unsigned int, cudaStream_t);    \
  template NppStatus resizeBatch<T, 4, true>(NppiSize, NppiRect, NppiSize, NppiRect, int,           \
                                             NppiResizeBatchCXR *, unsigned int, cudaStream_t);     \
  template NppStatus resizeBatchAdvanced<T, 1, false>(int, int, NppiImageDescriptor *,              \
                                                      NppiImageDescriptor *,                        \
                                                      NppiResizeBatchROI_Advanced *, unsigned int,  \
                                                      int, cudaStream_t);                           \
  template NppStatus resizeBatchAdvanced<T, 3, false>(int, int, NppiImageDescriptor *,              \
                                                      NppiImageDescriptor *,                        \
                                                      NppiResizeBatchROI_Advanced *, unsigned int,  \
                                                      int, cudaStream_t);                           \
  template NppStatus resizeBatchAdvanced<T, 4, false>(int, int, NppiImageDescriptor *,              \
                                                      NppiImageDescriptor *,                        \
                                                      NppiResizeBatchROI_Advanced *, unsigned int,  \
                                                      int, cudaStream_t);                           \
  template NppStatus resizeBatchAdvanced<T, 4, true>(int, int, NppiImageDescriptor *,               \
                                                     NppiImageDescriptor *,                         \
                                                     NppiResizeBatchROI_Advanced *, unsigned int,   \
                                                     int, cudaStream_t);

NPPI_RESIZE_INSTANTIATE_BATCH(Npp8u)
NPPI_RESIZE_INSTANTIATE_BATCH(Npp32f)
NPPI_RESIZE_INSTANTIATE_BATCH(Npp16f)

// Advanced SqrPixel (8u C1 only in the public API).
template NppStatus sqrPixelAdvanced<Npp8u, 1, false>(const Npp8u *, NppiSize, int, NppiRect, Npp8u *, int, NppiRect,
                                                     double, double, Npp8u *, int, cudaStream_t);

}  // namespace resize
}  // namespace nppi
