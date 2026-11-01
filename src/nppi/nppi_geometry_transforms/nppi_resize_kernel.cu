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

// Lanczos-3 coefficients as used by NVIDIA NPP 13.4: 301 float32 samples of
// the 3-lobe Lanczos kernel at d = i/100 (the table also holds a trailing zero
// so that the last interval can be interpolated).  The values equal the exact
// double sinc(d)*sinc(d/3) rounded to float32 (NPP's table differs from that
// rounding by at most 1 ulp); the 64f impulse goldens pin down the table.
__device__ __constant__ float kLanczos3Table[302] = {
    1.000000000e+00f, 9.998171930e-01f, 9.992690680e-01f, 9.983559250e-01f, 9.970785380e-01f, 9.954377410e-01f, 9.934347870e-01f,
    9.910711650e-01f, 9.883486030e-01f, 9.852690100e-01f, 9.818348880e-01f, 9.780485030e-01f, 9.739127760e-01f, 9.694306250e-01f,
    9.646055100e-01f, 9.594408270e-01f, 9.539404510e-01f, 9.481082560e-01f, 9.419486520e-01f, 9.354659910e-01f, 9.286651020e-01f,
    9.215508100e-01f, 9.141284820e-01f, 9.064031840e-01f, 8.983808760e-01f, 8.900670410e-01f, 8.814678190e-01f, 8.725894090e-01f,
    8.634380700e-01f, 8.540204170e-01f, 8.443431850e-01f, 8.344133500e-01f, 8.242378230e-01f, 8.138239380e-01f, 8.031790850e-01f,
    7.923105960e-01f, 7.812263370e-01f, 7.699339990e-01f, 7.584414480e-01f, 7.467567920e-01f, 7.348879580e-01f, 7.228434090e-01f,
    7.106313710e-01f, 6.982601880e-01f, 6.857384440e-01f, 6.730745430e-01f, 6.602772470e-01f, 6.473550800e-01f, 6.343168620e-01f,
    6.211712360e-01f, 6.079270840e-01f, 5.945931670e-01f, 5.811783080e-01f, 5.676913260e-01f, 5.541409850e-01f, 5.405362250e-01f,
    5.268858080e-01f, 5.131984350e-01f, 4.994829600e-01f, 4.857479630e-01f, 4.720021780e-01f, 4.582542480e-01f, 4.445127250e-01f,
    4.307860430e-01f, 4.170826670e-01f, 4.034108220e-01f, 3.897787030e-01f, 3.761946860e-01f, 3.626667260e-01f, 3.492026330e-01f,
    3.358102740e-01f, 3.224974270e-01f, 3.092714550e-01f, 2.961401640e-01f, 2.831105290e-01f, 2.701897920e-01f, 2.573850150e-01f,
    2.447030250e-01f, 2.321505250e-01f, 2.197338490e-01f, 2.074596430e-01f, 1.953339430e-01f, 1.833627820e-01f, 1.715520620e-01f,
    1.599072660e-01f, 1.484338190e-01f, 1.371371750e-01f, 1.260223690e-01f, 1.150941180e-01f, 1.043571530e-01f, 9.381592270e-02f,
    8.347468820e-02f, 7.333750280e-02f, 6.340817360e-02f, 5.369032170e-02f, 4.418735950e-02f, 3.490247950e-02f, 2.583865080e-02f,
    1.699863930e-02f, 8.384970020e-03f, -0.000000000e+00f, -8.154179900e-03f, -1.607571910e-02f, -2.376293020e-02f, -3.121435080e-02f,
    -3.842873130e-02f, -4.540504890e-02f, -5.214256050e-02f, -5.864053960e-02f, -6.489866970e-02f, -7.091671970e-02f, -7.669480890e-02f,
    -8.223310860e-02f, -8.753207330e-02f, -9.259230640e-02f, -9.741467240e-02f, -1.020001320e-01f, -1.063499670e-01f, -1.104655790e-01f,
    -1.143485310e-01f, -1.180005600e-01f, -1.214236170e-01f, -1.246198420e-01f, -1.275914910e-01f, -1.303409930e-01f, -1.328710170e-01f,
    -1.351842730e-01f, -1.372836980e-01f, -1.391722860e-01f, -1.408533300e-01f, -1.423301100e-01f, -1.436061110e-01f, -1.446849260e-01f,
    -1.455702630e-01f, -1.462659540e-01f, -1.467759160e-01f, -1.471042190e-01f, -1.472550030e-01f, -1.472325330e-01f, -1.470410970e-01f,
    -1.466851380e-01f, -1.461691260e-01f, -1.454976500e-01f, -1.446752850e-01f, -1.437067540e-01f, -1.425968110e-01f, -1.413502400e-01f,
    -1.399718670e-01f, -1.384665820e-01f, -1.368393000e-01f, -1.350949110e-01f, -1.332384050e-01f, -1.312747600e-01f, -1.292089520e-01f,
    -1.270459590e-01f, -1.247907730e-01f, -1.224483770e-01f, -1.200237130e-01f, -1.175217930e-01f, -1.149475130e-01f, -1.123058130e-01f,
    -1.096015130e-01f, -1.068395000e-01f, -1.040245670e-01f, -1.011614870e-01f, -9.825497120e-02f, -9.530962260e-02f, -9.233005340e-02f,
    -8.932082360e-02f, -8.628634360e-02f, -8.323109890e-02f, -8.015937360e-02f, -7.707537710e-02f, -7.398339360e-02f, -7.088737190e-02f,
    -6.779135020e-02f, -6.469924000e-02f, -6.161485980e-02f, -5.854190890e-02f, -5.548401920e-02f, -5.244470020e-02f, -4.942736030e-02f,
    -4.643530030e-02f, -4.347173870e-02f, -4.053976010e-02f, -3.764234110e-02f, -3.478233890e-02f, -3.196251020e-02f, -2.918549070e-02f,
    -2.645380980e-02f, -2.376987040e-02f, -2.113595050e-02f, -1.855422000e-02f, -1.602675950e-02f, -1.355540010e-02f, -1.114204990e-02f,
    -8.788379840e-03f, -6.495940030e-03f, -4.266189880e-03f, -2.100470010e-03f, 0.000000000e+00f, 2.034150060e-03f, 4.000939890e-03f,
    5.899569950e-03f, 7.729259790e-03f, 9.489360270e-03f, 1.117935030e-02f, 1.279881970e-02f, 1.434744990e-02f, 1.582504060e-02f,
    1.723149050e-02f, 1.856680030e-02f, 1.983108930e-02f, 2.102456990e-02f, 2.214751950e-02f, 2.320032940e-02f, 2.418350990e-02f,
    2.509763090e-02f, 2.594335940e-02f, 2.672144030e-02f, 2.743268010e-02f, 2.807803080e-02f, 2.865846080e-02f, 2.917500960e-02f,
    2.962883000e-02f, 3.002109010e-02f, 3.035306000e-02f, 3.062608090e-02f, 3.084147910e-02f, 3.100074080e-02f, 3.110530040e-02f,
    3.115670940e-02f, 3.115656040e-02f, 3.110644970e-02f, 3.100804980e-02f, 3.086302990e-02f, 3.067312020e-02f, 3.044009950e-02f,
    3.016570960e-02f, 2.985179050e-02f, 2.950013060e-02f, 2.911259980e-02f, 2.869101050e-02f, 2.823724970e-02f, 2.775318920e-02f,
    2.724068050e-02f, 2.670161050e-02f, 2.613786980e-02f, 2.555128000e-02f, 2.494373920e-02f, 2.431708950e-02f, 2.367315070e-02f,
    2.301377060e-02f, 2.234072980e-02f, 2.165582030e-02f, 2.096080970e-02f, 2.025740970e-02f, 1.954736000e-02f, 1.883232030e-02f,
    1.811393910e-02f, 1.739381070e-02f, 1.667352950e-02f, 1.595463980e-02f, 1.523859990e-02f, 1.452692040e-02f, 1.382100020e-02f,
    1.312220000e-02f, 1.243184040e-02f, 1.175121030e-02f, 1.108152980e-02f, 1.042397970e-02f, 9.779700080e-03f, 9.149749760e-03f,
    8.535170000e-03f, 7.936920040e-03f, 7.355920040e-03f, 6.793050100e-03f, 6.249099970e-03f, 5.724839870e-03f, 5.220950120e-03f,
    4.738090090e-03f, 4.276840020e-03f, 3.837730040e-03f, 3.421230010e-03f, 3.027779980e-03f, 2.657729900e-03f, 2.311399910e-03f,
    1.989020030e-03f, 1.690819980e-03f, 1.416930000e-03f, 1.167460000e-03f, 9.424400050e-04f, 7.418600140e-04f, 5.656700230e-04f,
    4.137599900e-04f, 2.859600060e-04f, 1.820799950e-04f, 1.018600010e-04f, 4.501000150e-05f, 1.118000000e-05f, 0.000000000e+00f,
    0.000000000e+00f,
};

__device__ __forceinline__ double lanczos3(double d) {
  double a = fabs(d);
  if (a >= 3.0) return 0.0;
  double x = a * 100.0;
  int i = (int)x;
  double f = x - (double)i;
  float lo = kLanczos3Table[i];
  float hi = kLanczos3Table[i + 1];
  return (double)lo + f * ((double)hi - (double)lo);
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
  // Separable evaluation: NPP resizes with a horizontal pass followed by a
  // vertical pass, so the x taps are accumulated first and the y taps combine
  // the per-row results.  The rounding of that order is observable in the
  // integer goldens (e.g. a value of exactly 128.5 stays 128).
  constexpr int kMaxTaps = 8;
  double wx[kMaxTaps], wy[kMaxTaps];
  int ox[kMaxTaps], oy[kMaxTaps];
  int taps = 0;
  double normSum = 1.0;
  bool normalize = false;
  if (mode == MODE_LINEAR || mode == MODE_CUBIC || mode == MODE_LANCZOS) {
    taps = (mode == MODE_LINEAR) ? 2 : (mode == MODE_CUBIC ? 4 : 6);
    const int base = (int)floor((double)sx);
    const int baseY = (int)floor((double)sy);
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
      // The truncated Lanczos-3 weights do not sum to one; NVIDIA normalises
      // each axis by its own weight sum (the recorded 64f impulse responses
      // are reproduced exactly by per-axis normalised weights).
      normalize = false;
      double sumX = 0.0, sumY = 0.0;
      for (int i = 0; i < 6; ++i) {
        sumX += wx[i];
        sumY += wy[i];
      }
      if (sumX != 0.0 && sumY != 0.0) {
        for (int i = 0; i < 6; ++i) {
          wx[i] /= sumX;
          wy[i] /= sumY;
        }
      }
    }
  } else {
    // SUPER: area average over the source interval [af*(x-shift), af*(x-shift+1))
    // belonging to each destination pixel, i.e. a box of width 1/f whose start
    // is sx + (1 - 1/f)/2.  Evaluated as the outer product of the axis
    // overlaps (measured with the super/down* impulse probes).
    double x0 = (double)sx + 0.5 * (1.0 - (double)adjF), y0 = (double)sy + 0.5 * (1.0 - (double)adjFy);
    double x1 = x0 + (double)adjF, y1 = y0 + (double)adjFy;
    int ix0 = (int)floor(x0), ix1 = (int)ceil(x1) - 1;
    int iy0 = (int)floor(y0), iy1 = (int)ceil(y1) - 1;
    int nx = 0, ny = 0;
    double sumX = 0.0, sumY = 0.0;
    for (int xx = ix0; xx <= ix1 && nx < kMaxTaps; ++xx) {
      double ov = fmin(x1, (double)xx + 1.0) - fmax(x0, (double)xx);
      if (ov <= 0.0) continue;
      wx[nx] = ov;
      ox[nx] = xx;
      sumX += ov;
      ++nx;
    }
    for (int yy = iy0; yy <= iy1 && ny < kMaxTaps; ++yy) {
      double ov = fmin(y1, (double)yy + 1.0) - fmax(y0, (double)yy);
      if (ov <= 0.0) continue;
      wy[ny] = ov;
      oy[ny] = yy;
      sumY += ov;
      ++ny;
    }
    if (nx == 0 || ny == 0) {
      for (int c = 0; c < NC; ++c) out[c] = (VT)0;
      return true;
    }
    taps = nx;  // x and y tap counts differ; handled below
    normSum = sumX * sumY;
    if (normSum <= 0.0) normSum = 1.0;
    normalize = true;
    // Store the y tap count in the unused tail of oy via nx/ny encoded below.
    // (Kept simple: SUPER is evaluated in its own loop.)
    VT acc[NC];
    for (int c = 0; c < NC; ++c) acc[c] = (VT)0;
    for (int j = 0; j < ny; ++j) {
      int yc = oy[j] < tapLoY ? tapLoY : (oy[j] > tapHiY ? tapHiY : oy[j]);
      const T *row = src + (size_t)yc * stride;
      VT hs[NC];
      for (int c = 0; c < NC; ++c) hs[c] = (VT)0;
      for (int i = 0; i < nx; ++i) {
        int xc = ox[i] < tapLoX ? tapLoX : (ox[i] > tapHiX ? tapHiX : ox[i]);
        const T *p = row + (size_t)xc * NC;
        for (int c = 0; c < NC; ++c) hs[c] += (VT)(wx[i] * (double)loadVal<T, VT>(p + c));
      }
      for (int c = 0; c < NC; ++c) acc[c] += (VT)(wy[j] * (double)hs[c]);
    }
    for (int c = 0; c < NC; ++c) out[c] = acc[c] / (VT)normSum;
    return true;
  }
  VT acc[NC];
  for (int c = 0; c < NC; ++c) acc[c] = (VT)0;
  for (int j = 0; j < taps; ++j) {
    int yc = oy[j] < tapLoY ? tapLoY : (oy[j] > tapHiY ? tapHiY : oy[j]);
    const T *row = src + (size_t)yc * stride;
    VT hs[NC];
    for (int c = 0; c < NC; ++c) hs[c] = (VT)0;
    for (int i = 0; i < taps; ++i) {
      int xc = ox[i] < tapLoX ? tapLoX : (ox[i] > tapHiX ? tapHiX : ox[i]);
      const T *p = row + (size_t)xc * NC;
      for (int c = 0; c < NC; ++c) hs[c] += (VT)(wx[i] * (double)loadVal<T, VT>(p + c));
    }
    for (int c = 0; c < NC; ++c) acc[c] += (VT)(wy[j] * (double)hs[c]);
  }
  if (normalize && normSum != 0.0) {
    for (int c = 0; c < NC; ++c) acc[c] = acc[c] / (VT)normSum;
  }
  for (int c = 0; c < NC; ++c) out[c] = acc[c];
  return true;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------
template <typename T, int NC, int PROC, typename VT>
__global__ void resizeKernel(const T *__restrict__ src, int sStep, int srcW, int srcH, VT clipHiX, VT clipHiY, T *__restrict__ dst, int dStep, NppiRect dROI, VT afx,
                             VT offx, VT basex, VT afy, VT offy, VT basey, VT clipLoX, VT clipLoY, VT coordLoX, VT coordHiX, VT coordLoY,
                             VT coordHiY, int tapLoX, int tapHiX, int tapLoY, int tapHiY, int mode) {
  int dx = dROI.x + blockIdx.x * blockDim.x + threadIdx.x;
  int dy = dROI.y + blockIdx.y * blockDim.y + threadIdx.y;
  if (dx >= dROI.x + dROI.width || dy >= dROI.y + dROI.height) return;
  VT sx, sy;
  if constexpr (std::is_same_v<VT, double>) {
    sx = afx * ((VT)dx - offx) + basex;
    sy = afy * ((VT)dy - offy) + basey;
  } else {
    sx = fmaf(afx, (VT)dx - offx, basex);
    sy = fmaf(afy, (VT)dy - offy, basey);
  }
  if (!(sx >= clipLoX && sx < (VT)clipHiX && sy >= clipLoY && sy < (VT)clipHiY)) return;
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
__global__ void resizePlanarKernel(const T *__restrict__ src, int sStep, int srcW, int srcH, VT clipHiX, VT clipHiY, T *__restrict__ dst, int dStep, NppiRect dROI, VT afx,
                                   VT offx, VT basex, VT afy, VT offy, VT basey, VT clipLoX, VT clipLoY, VT coordLoX, VT coordHiX, VT coordLoY,
                             VT coordHiY, int tapLoX, int tapHiX, int tapLoY, int tapHiY, int mode) {
  int dx = dROI.x + blockIdx.x * blockDim.x + threadIdx.x;
  int dy = dROI.y + blockIdx.y * blockDim.y + threadIdx.y;
  if (dx >= dROI.x + dROI.width || dy >= dROI.y + dROI.height) return;
  VT sx, sy;
  if constexpr (std::is_same_v<VT, double>) {
    sx = afx * ((VT)dx - offx) + basex;
    sy = afy * ((VT)dy - offy) + basey;
  } else {
    sx = fmaf(afx, (VT)dx - offx, basex);
    sy = fmaf(afy, (VT)dy - offy, basey);
  }
  if (!(sx >= clipLoX && sx < (VT)clipHiX && sy >= clipLoY && sy < (VT)clipHiY)) return;
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

// The CUBIC2P modes are accepted by the interpolation-mode validation but then
// rejected with NPP_INTERPOLATION_ERROR once the step checks have passed
// (measured: sqr_code/*_roi_beyond returns NPP_STEP_ERROR for those modes).
inline bool interpCubic2P(int eInterpolation) {
  return eInterpolation == NPPI_INTER_CUBIC2P_BSPLINE || eInterpolation == NPPI_INTER_CUBIC2P_CATMULLROM ||
         eInterpolation == NPPI_INTER_CUBIC2P_B05C03;
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

// nppiResize extends the source-ROI tap window on the right by the filter
// radius (measured with coordinate-coded probes): linear allows one pixel past
// the ROI, cubic two and Lanczos three.  The left edge is the ROI start.
inline int filterRadius(int mode) {
  switch (mode) {
    case MODE_LINEAR: return 1;
    case MODE_CUBIC: return 2;
    case MODE_LANCZOS: return 3;
    default: return 0;
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
                         VT afx, VT offx, VT basex, VT afy, VT offy, VT basey, VT clipLoX, VT clipLoY, VT clipHiX,
                         VT clipHiY, int tapLoX,
                         int tapHiX, int tapLoY, int tapHiY, int mode, int nc, bool planar, bool clampCoord, int proc,
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
      resizePlanarKernel<T, 3, VT><<<grid, block, 0, stream>>>(src, sStep, srcSize.width, srcSize.height, clipHiX, clipHiY, dst, dStep, dROI, afx, offx, basex, afy, offy, basey,
                                                              clipLoX, clipLoY, coordLo, coordHi, coordLoY, coordHiY, tapLoX, tapHiX, tapLoY, tapHiY, mode);
    } else {
      resizePlanarKernel<T, 4, VT><<<grid, block, 0, stream>>>(src, sStep, srcSize.width, srcSize.height, clipHiX, clipHiY, dst, dStep, dROI, afx, offx, basex, afy, offy, basey,
                                                              clipLoX, clipLoY, coordLo, coordHi, coordLoY, coordHiY, tapLoX, tapHiX, tapLoY, tapHiY, mode);
    }
    return;
  }
  switch (nc) {
    case 1:
      resizeKernel<T, 1, 1, VT><<<grid, block, 0, stream>>>(src, sStep, srcSize.width, srcSize.height,
                                                         clipHiX, clipHiY, dst, dStep, dROI, afx,
                                                         offx, basex, afy, offy, basey, clipLoX, clipLoY, coordLo, coordHi, coordLoY, coordHiY, tapLoX, tapHiX, tapLoY, tapHiY, mode);
      break;
    case 3:
      resizeKernel<T, 3, 3, VT><<<grid, block, 0, stream>>>(src, sStep, srcSize.width, srcSize.height,
                                                         clipHiX, clipHiY, dst, dStep, dROI, afx,
                                                         offx, basex, afy, offy, basey, clipLoX, clipLoY, coordLo, coordHi, coordLoY, coordHiY, tapLoX, tapHiX, tapLoY, tapHiY, mode);
      break;
    default:
      if (proc == 3) {
        resizeKernel<T, 4, 3, VT><<<grid, block, 0, stream>>>(src, sStep, srcSize.width, srcSize.height,
                                                              clipHiX, clipHiY, dst, dStep, dROI,
                                                              afx, offx, basex, afy, offy, basey, clipLoX, clipLoY, coordLo, coordHi, coordLoY, coordHiY, tapLoX, tapHiX, tapLoY, tapHiY, mode);
      } else {
        resizeKernel<T, 4, 4, VT><<<grid, block, 0, stream>>>(src, sStep, srcSize.width, srcSize.height,
                                                              clipHiX, clipHiY, dst, dStep, dROI,
                                                              afx, offx, basex, afy, offy, basey, clipLoX, clipLoY, coordLo, coordHi, coordLoY, coordHiY, tapLoX, tapHiX, tapLoY, tapHiY, mode);
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
  if (!interpSupported(eInterpolation) && !interpCubic2P(eInterpolation)) return NPP_INTERPOLATION_ERROR;
  if (nDstStep < (oDstROI.x + oDstROI.width) * nc * (int)sizeof(T)) return NPP_STEP_ERROR;
  if (interpCubic2P(eInterpolation)) return NPP_INTERPOLATION_ERROR;
  if (eInterpolation == NPPI_INTER_SUPER && (nXFactor >= 1.0 || nYFactor >= 1.0)) return NPP_RESIZE_FACTOR_ERROR;
  const VT afx = (VT)(1.0 / nXFactor);
  const VT asx = (VT)(nXShift * (1.0 / nXFactor) + (1.0 - 1.0 / nXFactor) * 0.5);
  const VT afy = (VT)(1.0 / nYFactor);
  const VT asy = (VT)(nYShift * (1.0 / nYFactor) + (1.0 - 1.0 / nYFactor) * 0.5);
  launchKernel<T, VT>(pSrc, nSrcStep, oSrcSize, clip, pDst, nDstStep, oDstROI, afx, (VT)0, (VT)(-asx), afy, (VT)0,
                      (VT)(-asy), (VT)(oSrcROI.x - 0.5 * nXFactor), (VT)(oSrcROI.y - 0.5 * nYFactor), (VT)(clip.x + clip.w),
                      (VT)(clip.y + clip.h), 0, oSrcSize.width - 1, 0, oSrcSize.height - 1, toMode(eInterpolation), nc,
                      false, false, proc, stream);
  return NPP_NO_ERROR;
}

template <typename T, typename VT>
static NppStatus runResize(const T *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcROI, T *pDst, int nDstStep,
                           NppiSize oDstSize, NppiRect oDstROI, int eInterpolation, int nc, int proc, bool batch,
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
  if (eInterpolation == NPPI_INTER_SUPER && (fx >= 1.0 || fy >= 1.0)) return NPP_RESIZE_FACTOR_ERROR;
  const VT afx = (VT)((double)oSrcROI.width / (double)oDstROI.width);
  const VT offx = (VT)((double)oDstROI.x - 0.5);
  const VT basex = (VT)((double)oSrcROI.x - 0.5);
  const VT afy = (VT)((double)oSrcROI.height / (double)oDstROI.height);
  const VT offy = (VT)((double)oDstROI.y - 0.5);
  const VT basey = (VT)((double)oSrcROI.y - 0.5);
  const int mode = toMode(eInterpolation);
  const VT big = (VT)1e30;
  VT clipLoX = -big, clipLoY = -big, clipHiX = big, clipHiY = big;
  int tapLoX = 0, tapHiX = oSrcSize.width - 1, tapLoY = 0, tapHiY = oSrcSize.height - 1;
  if (!batch) {
    const int radius = filterRadius(mode);
    tapLoX = clip.x;
    tapLoY = clip.y;
    tapHiX = clip.x + clip.w - 1 + radius;
    tapHiY = clip.y + clip.h - 1 + radius;
    if (tapHiX > oSrcSize.width - 1) tapHiX = oSrcSize.width - 1;
    if (tapHiY > oSrcSize.height - 1) tapHiY = oSrcSize.height - 1;
    clipLoX = (VT)(clip.x - 0.5);
    clipLoY = (VT)(clip.y - 0.5);
    clipHiX = (VT)(clip.x + clip.w);
    clipHiY = (VT)(clip.y + clip.h);
  }
  launchKernel<T, VT>(pSrc, nSrcStep, oSrcSize, clip, pDst, nDstStep, oDstROI, afx, offx, basex, afy, offy, basey,
                      clipLoX, clipLoY, clipHiX, clipHiY, tapLoX, tapHiX, tapLoY, tapHiY, mode, nc, false, false, proc,
                      stream);
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
  if (!interpSupported(eInterpolation) && !interpCubic2P(eInterpolation)) return NPP_INTERPOLATION_ERROR;
  if (nDstStep < (oDstROI.x + oDstROI.width) * (int)sizeof(T)) return NPP_STEP_ERROR;
  if (interpCubic2P(eInterpolation)) return NPP_INTERPOLATION_ERROR;
  if (eInterpolation == NPPI_INTER_SUPER && (nXFactor >= 1.0 || nYFactor >= 1.0)) return NPP_RESIZE_FACTOR_ERROR;
  const VT afx = (VT)(1.0 / nXFactor);
  const VT asx = (VT)(nXShift * (1.0 / nXFactor) + (1.0 - 1.0 / nXFactor) * 0.5);
  const VT afy = (VT)(1.0 / nYFactor);
  const VT asy = (VT)(nYShift * (1.0 / nYFactor) + (1.0 - 1.0 / nYFactor) * 0.5);
  for (int p = 0; p < nc; ++p) {
    launchKernel<T, VT>(pSrc[p], nSrcStep, oSrcSize, clip, pDst[p], nDstStep, oDstROI, afx, (VT)0, (VT)(-asx), afy,
                        (VT)0, (VT)(-asy), (VT)(oSrcROI.x - 0.5 * nXFactor), (VT)(oSrcROI.y - 0.5 * nYFactor), (VT)(clip.x + clip.w),
                        (VT)(clip.y + clip.h), 0, oSrcSize.width - 1, 0, oSrcSize.height - 1, toMode(eInterpolation), 1,
                        true, false, 1, stream);
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
  if (eInterpolation == NPPI_INTER_SUPER && (fx >= 1.0 || fy >= 1.0)) return NPP_RESIZE_FACTOR_ERROR;
  const VT afx = (VT)((double)oSrcROI.width / (double)oDstROI.width);
  const VT offx = (VT)((double)oDstROI.x - 0.5);
  const VT basex = (VT)((double)oSrcROI.x - 0.5);
  const VT afy = (VT)((double)oSrcROI.height / (double)oDstROI.height);
  const VT offy = (VT)((double)oDstROI.y - 0.5);
  const VT basey = (VT)((double)oSrcROI.y - 0.5);
  const int mode = toMode(eInterpolation);
  const int radius = filterRadius(mode);
  int tapHiX = clip.x + clip.w - 1 + radius;
  int tapHiY = clip.y + clip.h - 1 + radius;
  if (tapHiX > oSrcSize.width - 1) tapHiX = oSrcSize.width - 1;
  if (tapHiY > oSrcSize.height - 1) tapHiY = oSrcSize.height - 1;
  for (int p = 0; p < nc; ++p) {
    launchKernel<T, VT>(pSrc[p], nSrcStep, oSrcSize, clip, pDst[p], nDstStep, oDstROI, afx, offx, basex, afy, offy,
                        basey, (VT)(clip.x - 0.5), (VT)(clip.y - 0.5), (VT)(clip.x + clip.w), (VT)(clip.y + clip.h),
                        clip.x, tapHiX, clip.y, tapHiY, mode, 1, true, false, 1, stream);
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
                          AC4 ? 3 : C, false, stream);
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
    using VT = typename Traits<T>::VT;
    NppStatus st = runResize<T, VT>(static_cast<const T *>(host[i].pSrc), host[i].nSrcStep, oSmallestSrcSize, oSrcROI,
                                    static_cast<T *>(host[i].pDst), host[i].nDstStep, oSmallestDstSize, oDstROI,
                                    eInterpolation, C, AC4 ? 3 : C, true, stream);
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
    using VT = typename Traits<T>::VT;
    NppStatus st = runResize<T, VT>(static_cast<const T *>(src[i].pData), src[i].nStep, src[i].oSize,
                                    roi[i].oSrcRectROI, static_cast<T *>(dst[i].pData), dst[i].nStep, dst[i].oSize,
                                    roi[i].oDstRectROI, eInterpolation, C, AC4 ? 3 : C, true, stream);
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
