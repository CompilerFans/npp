// ColorTwist kernels and executor entry points.
// See nppi_color_twist_executor.h for the family contract.
#include "npp.h"
#include "nppi_color_twist_executor.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

namespace nppi {
namespace colortwist {
namespace {

constexpr int kBlockX = 16;
constexpr int kBlockY = 16;

// Row-major twist coefficients: 3x4 (m[0..11]), or 4x4 (m[0..15]) followed by
// the 4 additive constants (m[16..19]) for the "C" variants.  nMin/nMax are
// only used by the batch forms.
struct TwistParams {
  float m[20];
  float nMin;
  float nMax;
};

inline NppStatus checkLaunch() {
  return (cudaGetLastError() == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

// ---- dtype conversion helpers ----------------------------------------------

template <typename T> __device__ __forceinline__ float toFloat(const T &v) { return static_cast<float>(v); }

__device__ __forceinline__ float toFloat(const Npp16f &v) {
  const unsigned short bits = static_cast<unsigned short>(v.fp16);
  const __half h = *reinterpret_cast<const __half *>(&bits);
  return __half2float(h);
}

// Integer conversion, pinned by the recorded goldens (RTX 5090, NPP 13.4):
// truncate toward zero, then saturate to the dtype range.  The +0.5f rounding
// bias of the packed CH>=3 path is added by the caller (BIAS template flag)
// BEFORE this conversion, so those layouts round half-up.
template <typename T> __device__ __forceinline__ T fromFloatInt(float v, int lo, int hi) {
  const int i = __float2int_rz(v);
  if (i < lo) return static_cast<T>(lo);
  if (i > hi) return static_cast<T>(hi);
  return static_cast<T>(i);
}

template <typename T> __device__ __forceinline__ T fromFloat(float v);

template <> __device__ __forceinline__ Npp8u fromFloat<Npp8u>(float v) { return fromFloatInt<Npp8u>(v, 0, 255); }

template <> __device__ __forceinline__ Npp8s fromFloat<Npp8s>(float v) { return fromFloatInt<Npp8s>(v, -128, 127); }

template <> __device__ __forceinline__ Npp16u fromFloat<Npp16u>(float v) { return fromFloatInt<Npp16u>(v, 0, 65535); }

template <> __forceinline__ __device__ Npp16s fromFloat<Npp16s>(float v) {
  return fromFloatInt<Npp16s>(v, -32768, 32767);
}

// NPP saturates float results to [-FLT_MAX, FLT_MAX] (a +inf result is stored
// as FLT_MAX); measured on the 5090 for the overflowing C3/planar cases.
template <> __device__ __forceinline__ Npp32f fromFloat<Npp32f>(float v) {
  if (v > 3.402823466e+38F) return 3.402823466e+38F;
  if (v < -3.402823466e+38F) return -3.402823466e+38F;
  return v;
}

// Npp16f: plain round-to-nearest, NO rounding bias on any layout (measured).
template <> __device__ __forceinline__ Npp16f fromFloat<Npp16f>(float v) {
  const __half h = __float2half_rn(v);
  Npp16f r;
  r.fp16 = static_cast<short>(__half_raw(h).x);
  return r;
}

// The +0.5f bias is applied by the packed CH>=3 kernels of NPP 13.4 for the
// float and integer dtypes; the Npp16f kernels have no bias on any layout.
template <typename T> struct has_rounding_bias : std::is_same<T, Npp32f> {};
template <> struct has_rounding_bias<Npp8u> : std::true_type {};
template <> struct has_rounding_bias<Npp8s> : std::true_type {};
template <> struct has_rounding_bias<Npp16u> : std::true_type {};
template <> struct has_rounding_bias<Npp16s> : std::true_type {};

template <typename T, int CH> __host__ __device__ constexpr bool packedBias() {
  return (CH >= 3) && has_rounding_bias<T>::value;
}

// The BATCH kernels bias differently (measured on the RTX 5090): Npp32f adds
// the +0.5f bias on EVERY layout (even 32f C1R keeps src + 0.5f), the integer
// dtypes follow the packed CH >= 3 rule and Npp16f never has a bias.
template <typename T, int CH> __host__ __device__ constexpr bool batchBias() {
  return std::is_same<T, Npp32f>::value || packedBias<T, CH>();
}

// Packed destination steps must be a multiple of the vector width NPP 13.4
// loads per layout (measured on the RTX 5090): a full 2-element vector for C2,
// a full 4-element vector for C4/AC4, a single element for C1/C3 (and for the
// planar variants).  A violating step returns NPP_NOT_EVEN_STEP_ERROR; the
// check is a signed modulo, so a negative step with a non-zero residue is
// rejected while an even negative step keeps the "accepted, no launch"
// behavior.  The check runs after the null/size/step-range/empty-ROI checks
// and only against the destination steps (the packed src step is not
// validated at all; the planar variants check both).
template <int CH> __host__ __device__ constexpr int stepMultiple() {
  return (CH == 2 || CH == 4) ? CH : 1;
}

// Batch clamp (measured on the RTX 5090): comparison-based, so a NaN produced
// by the dot product (0 * +-inf terms) survives instead of being replaced by
// fminf/fmaxf NaN-suppression semantics.
__device__ __forceinline__ float clampRange(float v, const TwistParams &p) {
  return (v < p.nMin) ? p.nMin : ((v > p.nMax) ? p.nMax : v);
}

// Batch dtype conversion (measured): the 8u batch kernels store the LOW BYTE of
// the u32 conversion - (Npp8u)__float2uint_rz(x), i.e. negative -> 0 and NO
// [0,255] saturation (64 * v + 8192 wraps around) - unlike the packed
// single-image kernels, which saturate (fromFloat).  The one exception is the
// AC4 channel-2 path below, which does saturate.  All other dtypes convert
// exactly like the single-image kernels.
template <typename T> __device__ __forceinline__ T batchFromFloat(float v) { return fromFloat<T>(v); }

template <> __device__ __forceinline__ Npp8u batchFromFloat<Npp8u>(float v) {
  return static_cast<Npp8u>(__float2uint_rz(v));
}

// Apply the twist to one packed pixel (CH channels at s / d).  Semantics of the
// missing input channels and of the alpha channel are documented in the header.
// BIAS: the packed 3-channel-and-up path of NPP 13.4 adds a +0.5f rounding
// bias to the accumulated value and keeps it in the float output (measured on
// the RTX 5090); C1/C2 packed variants and the planar variants do not.
template <typename T, int CH, bool AC4, bool HAS_CONST, bool BATCH, bool BIAS>
__device__ __forceinline__ void twistPixel(const T *s, T *d, const TwistParams &p) {
  const float v0 = toFloat(s[0]);
  const float v1 = (CH >= 2) ? toFloat(s[1]) : 0.0f;
  const float v2 = (CH >= 3) ? toFloat(s[2]) : 0.0f;
  const float v3 = (CH == 4 && !AC4) ? toFloat(s[3]) : 0.0f;

  float o0, o1, o2, o3;
  // BATCH order (measured): the +0.5f bias is added FIRST, then the clamp
  // min(max(acc + 0.5f, nMin), nMax) is applied, then the value is converted.
  // The AC4 batch kernels exempt output channel 2 from the clamp and push it
  // through the saturating conversion instead (measured on the 5090).
  if (HAS_CONST) {
    if constexpr (BATCH) {
      // Batch "C" forms take a 4x5 matrix: the constant is the fifth element
      // of each row (m[4], m[9], m[14], m[19]).
      o0 = p.m[0] * v0 + p.m[1] * v1 + p.m[2] * v2 + p.m[3] * v3 + p.m[4];
      o1 = p.m[5] * v0 + p.m[6] * v1 + p.m[7] * v2 + p.m[8] * v3 + p.m[9];
      o2 = p.m[10] * v0 + p.m[11] * v1 + p.m[12] * v2 + p.m[13] * v3 + p.m[14];
      o3 = p.m[15] * v0 + p.m[16] * v1 + p.m[17] * v2 + p.m[18] * v3 + p.m[19];
    } else {
      // Single-image "C" forms take a 4x4 twist plus a trailing constant block
      // (fillTwistC packs it at m[16..19]).
      o0 = p.m[0] * v0 + p.m[1] * v1 + p.m[2] * v2 + p.m[3] * v3 + p.m[16];
      o1 = p.m[4] * v0 + p.m[5] * v1 + p.m[6] * v2 + p.m[7] * v3 + p.m[17];
      o2 = p.m[8] * v0 + p.m[9] * v1 + p.m[10] * v2 + p.m[11] * v3 + p.m[18];
      o3 = p.m[12] * v0 + p.m[13] * v1 + p.m[14] * v2 + p.m[15] * v3 + p.m[19];
    }
    if (BIAS) {
      o0 = o0 + 0.5f;
      o1 = o1 + 0.5f;
      o2 = o2 + 0.5f;
      o3 = o3 + 0.5f;
    }
    if (BATCH) {
      o0 = clampRange(o0, p);
      o1 = clampRange(o1, p);
      o2 = clampRange(o2, p);
      o3 = clampRange(o3, p);
    }
  } else {
    o0 = p.m[0] * v0 + p.m[1] * v1 + p.m[2] * v2 + p.m[3];
    o1 = p.m[4] * v0 + p.m[5] * v1 + p.m[6] * v2 + p.m[7];
    o2 = p.m[8] * v0 + p.m[9] * v1 + p.m[10] * v2 + p.m[11];
    if (BIAS) {
      o0 = o0 + 0.5f;
      o1 = o1 + 0.5f;
      o2 = o2 + 0.5f;
    }
    if (BATCH) {
      o0 = clampRange(o0, p);
      o1 = clampRange(o1, p);
      if (!AC4) {
        o2 = clampRange(o2, p);
      }
    }
  }

  if (BATCH) {
    d[0] = batchFromFloat<T>(o0);
    if (CH >= 2)
      d[1] = batchFromFloat<T>(o1);
    if (CH >= 3)
      d[2] = (AC4 && !HAS_CONST) ? fromFloat<T>(o2) : batchFromFloat<T>(o2);
    if (CH == 4) {
      if (HAS_CONST) {
        d[3] = batchFromFloat<T>(o3);
      } else if (std::is_same<T, Npp16f>::value) {
        // The 16f batch kernels copy the source alpha (measured); the 8u/32f
        // batch kernels leave the destination alpha untouched.
        d[3] = s[3];
      }
    }
  } else {
    d[0] = fromFloat<T>(o0);
    if (CH >= 2)
      d[1] = fromFloat<T>(o1);
    if (CH >= 3)
      d[2] = fromFloat<T>(o2);
    if (CH == 4) {
      if (HAS_CONST) {
        d[3] = fromFloat<T>(o3);
      } else if (!AC4) {
        d[3] = s[3];
      }
    }
  }
}

// ---- kernels -----------------------------------------------------------------

template <typename T, int CH, bool AC4, bool HAS_CONST, bool BATCH, bool BIAS>
__global__ void colorTwistPackedKernel(const T *src, int srcStep, T *dst, int dstStep, int width, int height,
                                       TwistParams p) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const T *s = reinterpret_cast<const T *>(reinterpret_cast<const char *>(src) + (size_t)y * srcStep) +
               (size_t)x * CH;
  T *d = reinterpret_cast<T *>(reinterpret_cast<char *>(dst) + (size_t)y * dstStep) + (size_t)x * CH;
  twistPixel<T, CH, AC4, HAS_CONST, BATCH, BIAS>(s, d, p);
}

template <typename T>
__global__ void colorTwistPlanarKernel(const T *s0, const T *s1, const T *s2, int srcStep, T *d0, T *d1, T *d2,
                                       int dstStep, int width, int height, TwistParams p) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const size_t off = (size_t)y * srcStep;
  const float v0 = toFloat(reinterpret_cast<const T *>(reinterpret_cast<const char *>(s0) + off)[x]);
  const float v1 = toFloat(reinterpret_cast<const T *>(reinterpret_cast<const char *>(s1) + off)[x]);
  const float v2 = toFloat(reinterpret_cast<const T *>(reinterpret_cast<const char *>(s2) + off)[x]);
  float o0 = p.m[0] * v0 + p.m[1] * v1 + p.m[2] * v2 + p.m[3];
  float o1 = p.m[4] * v0 + p.m[5] * v1 + p.m[6] * v2 + p.m[7];
  float o2 = p.m[8] * v0 + p.m[9] * v1 + p.m[10] * v2 + p.m[11];
  const size_t doff = (size_t)y * dstStep;
  reinterpret_cast<T *>(reinterpret_cast<char *>(d0) + doff)[x] = fromFloat<T>(o0);
  reinterpret_cast<T *>(reinterpret_cast<char *>(d1) + doff)[x] = fromFloat<T>(o1);
  reinterpret_cast<T *>(reinterpret_cast<char *>(d2) + doff)[x] = fromFloat<T>(o2);
}

// One image per blockIdx.z; the batch entry (and its twist matrix, read from
// device memory) is selected on the device.
template <typename T, int CH, bool AC4, bool HAS_CONST, bool INPLACE>
__global__ void colorTwistBatchKernel(NppiColorTwistBatchCXR *list, int width, int height, float nMin, float nMax) {
  const NppiColorTwistBatchCXR entry = list[blockIdx.z];
  // Negative per-entry steps are unspecified in NPP (probing them would fault
  // the reference recorder); mcMPP defensively skips such entries instead of
  // reading/writing at negative offsets.
  if (entry.nSrcStep < 0 || entry.nDstStep < 0) {
    return;
  }
  TwistParams p;
  p.nMin = nMin;
  p.nMax = nMax;
  const int ncoef = HAS_CONST ? 20 : 12;
  for (int i = 0; i < ncoef; ++i) {
    p.m[i] = entry.pTwist[i];
  }
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const T *src = reinterpret_cast<const T *>(entry.pSrc);
  void *dstRaw = INPLACE ? const_cast<void *>(entry.pSrc) : entry.pDst;
  T *dst = reinterpret_cast<T *>(dstRaw);
  const T *s =
      reinterpret_cast<const T *>(reinterpret_cast<const char *>(src) + (size_t)y * entry.nSrcStep) + (size_t)x * CH;
  T *d = reinterpret_cast<T *>(reinterpret_cast<char *>(dst) + (size_t)y * entry.nDstStep) + (size_t)x * CH;
  twistPixel<T, CH, AC4, HAS_CONST, true, batchBias<T, CH>()>(s, d, p);
}

// ---- shared host-side validation ----------------------------------------------

// Packed 3x4 twist validation, pinned against NVIDIA NPP 13.4 (RTX 5090):
//  - negative ROI                 -> NPP_SIZE_ERROR
//  - dst step 0 .. rowBytes-1     -> NPP_STEP_ERROR (an unsigned-range check:
//                                    negative dst steps are ACCEPTED by NPP)
//  - the source step is NOT validated at all (step 0 aliases row 0, NPP still
//    launches and returns NPP_NO_ERROR)
//  - null src/dst                 -> NPP_NULL_POINTER_ERROR
//  - null aTwist crashes NPP      -> mcMPP answers NPP_NULL_POINTER_ERROR
//  - empty ROI (0 width/height)   -> NPP_NO_ERROR, nothing launched
inline NppStatus validatePackedR(const void *pSrc, const void *pDst, int nDstStep, int width, int channels,
                                 size_t elemSize, const void *aTwist) {
  if (width < 0) {
    return NPP_SIZE_ERROR;  // caller pre-checks height as well
  }
  const size_t rowBytes = (size_t)width * channels * elemSize;
  if (nDstStep >= 0 && (size_t)nDstStep < rowBytes) {
    return NPP_STEP_ERROR;
  }
  if (!pSrc || !pDst || !aTwist) {
    return NPP_NULL_POINTER_ERROR;
  }
  return NPP_NO_ERROR;
}

inline void fillTwist(TwistParams &p, const Npp32f aTwist[3][4]) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      p.m[i * 4 + j] = aTwist[i][j];
    }
  }
  p.nMin = 0.0f;
  p.nMax = 0.0f;
}

inline void fillTwistC(TwistParams &p, const Npp32f aTwist[4][4], const Npp32f aConstants[4]) {
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      p.m[i * 4 + j] = aTwist[i][j];
    }
  }
  for (int i = 0; i < 4; ++i) {
    p.m[16 + i] = aConstants[i];
  }
  p.nMin = 0.0f;
  p.nMax = 0.0f;
}

inline dim3 gridSize(int width, int height) {
  return dim3((width + kBlockX - 1) / kBlockX, (height + kBlockY - 1) / kBlockY);
}

}  // namespace

// ---- packed 3x4 twist --------------------------------------------------------

template <typename T, int CH, bool AC4>
NppStatus twistR(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4],
                 cudaStream_t stream) {
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  NppStatus status = validatePackedR(pSrc, pDst, nDstStep, oSizeROI.width, CH, sizeof(T), aTwist);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  if (nDstStep % (stepMultiple<CH>() * (int)sizeof(T)) != 0) {
    return NPP_NOT_EVEN_STEP_ERROR;
  }
  if (nSrcStep < 0 || nDstStep < 0) {
    // NPP launches and touches memory outside the buffers here; mcMPP keeps
    // the status but declines to launch (unobservable via the API).
    return NPP_NO_ERROR;
  }
  TwistParams p;
  fillTwist(p, aTwist);
  colorTwistPackedKernel<T, CH, AC4, false, false, packedBias<T, CH>()>
      <<<gridSize(oSizeROI.width, oSizeROI.height), dim3(kBlockX, kBlockY), 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                                          oSizeROI.width, oSizeROI.height,
                                                                                          p);
  return checkLaunch();
}

template <typename T, int CH, bool AC4>
NppStatus twistIR(T *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], cudaStream_t stream) {
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  // The single in-place step plays the destination role: 0 .. rowBytes-1 is
  // NPP_STEP_ERROR, negative steps are accepted (unsigned-range check).
  const size_t rowBytes = (size_t)oSizeROI.width * CH * sizeof(T);
  if (nSrcDstStep >= 0 && (size_t)nSrcDstStep < rowBytes) {
    return NPP_STEP_ERROR;
  }
  if (!pSrcDst || !aTwist) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  if (nSrcDstStep % (stepMultiple<CH>() * (int)sizeof(T)) != 0) {
    return NPP_NOT_EVEN_STEP_ERROR;
  }
  if (nSrcDstStep < 0) {
    return NPP_NO_ERROR;  // as above: no OOB launch
  }
  TwistParams p;
  fillTwist(p, aTwist);
  colorTwistPackedKernel<T, CH, AC4, false, false, packedBias<T, CH>()>
      <<<gridSize(oSizeROI.width, oSizeROI.height), dim3(kBlockX, kBlockY), 0, stream>>>(
          pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height, p);
  return checkLaunch();
}

// ---- planar 3x4 twist --------------------------------------------------------

// Planar 3x4 twist validation, measured on NPP 13.4 (RTX 5090).  Unlike the
// packed variants, BOTH steps are validated and the comparison is signed:
// step <= 0 or step < rowBytes -> NPP_STEP_ERROR (a negative src step is
// rejected here even though the packed path accepts it).
template <typename T>
NppStatus twistP3(const T *const pSrc[3], int nSrcStep, T *const pDst[3], int nDstStep, NppiSize oSizeROI,
                  const Npp32f aTwist[3][4], cudaStream_t stream) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pSrc[2] || !pDst || !pDst[0] || !pDst[1] || !pDst[2] || !aTwist) {
    // A null plane array crashes the reference; a null plane is rejected.
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  const size_t rowBytes = (size_t)oSizeROI.width * sizeof(T);
  if (nSrcStep < (int)rowBytes || nDstStep < (int)rowBytes) {
    return NPP_STEP_ERROR;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  if (nSrcStep % (int)sizeof(T) != 0 || nDstStep % (int)sizeof(T) != 0) {
    return NPP_NOT_EVEN_STEP_ERROR;
  }
  TwistParams p;
  fillTwist(p, aTwist);
  colorTwistPlanarKernel<T>
      <<<gridSize(oSizeROI.width, oSizeROI.height), dim3(kBlockX, kBlockY), 0, stream>>>(pSrc[0], pSrc[1], pSrc[2],
                                                                                          nSrcStep, pDst[0], pDst[1],
                                                                                          pDst[2], nDstStep,
                                                                                          oSizeROI.width,
                                                                                          oSizeROI.height, p);
  return checkLaunch();
}

template <typename T>
NppStatus twistIP3(T *const pSrcDst[3], int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4],
                   cudaStream_t stream) {
  if (!pSrcDst || !pSrcDst[0] || !pSrcDst[1] || !pSrcDst[2] || !aTwist) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  const size_t rowBytes = (size_t)oSizeROI.width * sizeof(T);
  if (nSrcDstStep < (int)rowBytes) {
    return NPP_STEP_ERROR;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  if (nSrcDstStep % (int)sizeof(T) != 0) {
    return NPP_NOT_EVEN_STEP_ERROR;
  }
  TwistParams p;
  fillTwist(p, aTwist);
  colorTwistPlanarKernel<T>
      <<<gridSize(oSizeROI.width, oSizeROI.height), dim3(kBlockX, kBlockY), 0, stream>>>(
          pSrcDst[0], pSrcDst[1], pSrcDst[2], nSrcDstStep, pSrcDst[0], pSrcDst[1], pSrcDst[2], nSrcDstStep,
          oSizeROI.width, oSizeROI.height, p);
  return checkLaunch();
}

// ---- 4x4 twist + constants ("C" variants) -------------------------------------

template <typename T>
NppStatus twistCR(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[4][4],
                  const Npp32f aConstants[4], cudaStream_t stream) {
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  NppStatus status = validatePackedR(pSrc, pDst, nDstStep, oSizeROI.width, 4, sizeof(T), aTwist);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  if (!aConstants) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  if (nDstStep % (4 * (int)sizeof(T)) != 0) {
    return NPP_NOT_EVEN_STEP_ERROR;
  }
  if (nSrcStep < 0 || nDstStep < 0) {
    return NPP_NO_ERROR;  // no OOB launch (see twistR)
  }
  TwistParams p;
  fillTwistC(p, aTwist, aConstants);
  colorTwistPackedKernel<T, 4, false, true, false, packedBias<T, 4>()>
      <<<gridSize(oSizeROI.width, oSizeROI.height), dim3(kBlockX, kBlockY), 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                                          oSizeROI.width, oSizeROI.height,
                                                                                          p);
  return checkLaunch();
}

template <typename T>
NppStatus twistCIR(T *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[4][4],
                   const Npp32f aConstants[4], cudaStream_t stream) {
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  const size_t rowBytes = (size_t)oSizeROI.width * 4 * sizeof(T);
  if (nSrcDstStep >= 0 && (size_t)nSrcDstStep < rowBytes) {
    return NPP_STEP_ERROR;
  }
  if (!pSrcDst || !aTwist || !aConstants) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  if (nSrcDstStep % (4 * (int)sizeof(T)) != 0) {
    return NPP_NOT_EVEN_STEP_ERROR;
  }
  if (nSrcDstStep < 0) {
    return NPP_NO_ERROR;  // no OOB launch
  }
  TwistParams p;
  fillTwistC(p, aTwist, aConstants);
  colorTwistPackedKernel<T, 4, false, true, false, packedBias<T, 4>()>
      <<<gridSize(oSizeROI.width, oSizeROI.height), dim3(kBlockX, kBlockY), 0, stream>>>(
          pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height, p);
  return checkLaunch();
}

// ---- batch ---------------------------------------------------------------------

template <typename T, int CH, bool AC4, bool INPLACE, bool HAS_CONST>
NppStatus batchTwist(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI, NppiColorTwistBatchCXR *pBatchList, int nBatchSize,
                     cudaStream_t stream) {
  // Validation order measured against NPP 13.4 (RTX 5090): a null list, then
  // nBatchSize < 2 (NPP_BAD_ARGUMENT_ERROR - "must be > 1" is enforced), then
  // a negative ROI.  Steps and per-entry pointers are NOT validated: NPP
  // launches anyway (a null entry pointer returns NPP_NO_ERROR and faults on
  // the device).  An empty ROI also launches - with a zero grid dimension -
  // and yields NPP_CUDA_KERNEL_EXECUTION_ERROR, which checkLaunch() below
  // reproduces through the same degenerate launch.
  if (!pBatchList) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nBatchSize < 2) {
    return NPP_BAD_ARGUMENT_ERROR;
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  dim3 grid = gridSize(oSizeROI.width, oSizeROI.height);
  grid.z = static_cast<unsigned>(nBatchSize);
  colorTwistBatchKernel<T, CH, AC4, HAS_CONST, INPLACE>
      <<<grid, dim3(kBlockX, kBlockY), 0, stream>>>(pBatchList, oSizeROI.width, oSizeROI.height, nMin, nMax);
  return checkLaunch();
}

// ---- explicit instantiations ----------------------------------------------------

#define INST_TWIST_R(T, CH, AC4)                                                                                       \
  template NppStatus twistR<T, CH, AC4>(const T *, int, T *, int, NppiSize, const Npp32f[3][4], cudaStream_t);          \
  template NppStatus twistIR<T, CH, AC4>(T *, int, NppiSize, const Npp32f[3][4], cudaStream_t);

// family nppiColorTwist (32f images)
INST_TWIST_R(Npp32f, 1, false)
INST_TWIST_R(Npp32f, 2, false)
INST_TWIST_R(Npp32f, 3, false)
INST_TWIST_R(Npp32f, 4, false)
INST_TWIST_R(Npp32f, 4, true)
// nppiColorTwist32f 8u/16u/16s image variants (the nppi_color_twist.cpp
// wrappers for the 8u/16u/16s families call these directly).
INST_TWIST_R(Npp8u, 1, false)
INST_TWIST_R(Npp8u, 2, false)
INST_TWIST_R(Npp8u, 3, false)
INST_TWIST_R(Npp8u, 4, false)
INST_TWIST_R(Npp8u, 4, true)
INST_TWIST_R(Npp16u, 1, false)
INST_TWIST_R(Npp16u, 2, false)
INST_TWIST_R(Npp16u, 3, false)
INST_TWIST_R(Npp16u, 4, true)
INST_TWIST_R(Npp16s, 1, false)
INST_TWIST_R(Npp16s, 2, false)
INST_TWIST_R(Npp16s, 3, false)
INST_TWIST_R(Npp16s, 4, true)
// family nppiColorTwist32f, missing dtypes 8s / 16f
INST_TWIST_R(Npp8s, 1, false)
INST_TWIST_R(Npp8s, 2, false)
INST_TWIST_R(Npp8s, 3, false)
INST_TWIST_R(Npp8s, 4, false)
INST_TWIST_R(Npp8s, 4, true)
INST_TWIST_R(Npp16f, 1, false)
INST_TWIST_R(Npp16f, 2, false)
INST_TWIST_R(Npp16f, 3, false)
INST_TWIST_R(Npp16f, 4, false)

#define INST_TWIST_P3(T)                                                                                               \
  template NppStatus twistP3<T>(const T *const *, int, T *const *, int, NppiSize, const Npp32f[3][4], cudaStream_t);    \
  template NppStatus twistIP3<T>(T *const *, int, NppiSize, const Npp32f[3][4], cudaStream_t);

INST_TWIST_P3(Npp32f)
INST_TWIST_P3(Npp8s)
INST_TWIST_P3(Npp8u)
INST_TWIST_P3(Npp16u)
INST_TWIST_P3(Npp16s)

#define INST_TWIST_C(T)                                                                                                \
  template NppStatus twistCR<T>(const T *, int, T *, int, NppiSize, const Npp32f[4][4], const Npp32f[4], cudaStream_t); \
  template NppStatus twistCIR<T>(T *, int, NppiSize, const Npp32f[4][4], const Npp32f[4], cudaStream_t);

INST_TWIST_C(Npp8u)
INST_TWIST_C(Npp16f)
INST_TWIST_C(Npp32f)

#define INST_BATCH(T, CH, AC4, INPLACE, HAS_CONST)                                                                      \
  template NppStatus batchTwist<T, CH, AC4, INPLACE, HAS_CONST>(Npp32f, Npp32f, NppiSize, NppiColorTwistBatchCXR *, int, \
                                                               cudaStream_t);

// family nppiColorTwistBatch (32f images, 3x4 twist)
INST_BATCH(Npp32f, 1, false, false, false)
INST_BATCH(Npp32f, 1, false, true, false)
INST_BATCH(Npp32f, 3, false, false, false)
INST_BATCH(Npp32f, 3, false, true, false)
INST_BATCH(Npp32f, 4, false, false, false)
INST_BATCH(Npp32f, 4, false, true, false)
INST_BATCH(Npp32f, 4, true, false, false)
INST_BATCH(Npp32f, 4, true, true, false)
// family nppiColorTwistBatch32f (8u / 16f images, 3x4 twist)
INST_BATCH(Npp8u, 1, false, false, false)
INST_BATCH(Npp8u, 1, false, true, false)
INST_BATCH(Npp8u, 3, false, false, false)
INST_BATCH(Npp8u, 3, false, true, false)
INST_BATCH(Npp8u, 4, false, false, false)
INST_BATCH(Npp8u, 4, false, true, false)
INST_BATCH(Npp8u, 4, true, false, false)
INST_BATCH(Npp8u, 4, true, true, false)
INST_BATCH(Npp16f, 1, false, false, false)
INST_BATCH(Npp16f, 1, false, true, false)
INST_BATCH(Npp16f, 3, false, false, false)
INST_BATCH(Npp16f, 3, false, true, false)
INST_BATCH(Npp16f, 4, false, false, false)
INST_BATCH(Npp16f, 4, false, true, false)
// family nppiColorTwistBatch32fC / nppiColorTwistBatch_32fC (4x4 + constants)
INST_BATCH(Npp8u, 4, false, false, true)
INST_BATCH(Npp8u, 4, false, true, true)
INST_BATCH(Npp16f, 4, false, false, true)
INST_BATCH(Npp16f, 4, false, true, true)
INST_BATCH(Npp32f, 4, false, false, true)
INST_BATCH(Npp32f, 4, false, true, true)

}  // namespace colortwist
}  // namespace nppi
