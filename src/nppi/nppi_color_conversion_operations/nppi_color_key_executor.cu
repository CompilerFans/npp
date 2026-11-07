#include "nppi_color_key_executor.h"

#include <cuda_runtime.h>

#include "npp_kernel_utils.h"

namespace nppi {
namespace color_key {

namespace {

// Color key passed to the kernel by value (plain arrays would decay to
// pointers, and the host-side key lives on the stack).
template <typename T, int Channels> struct ColorKeyVec {
  T v[Channels];
};

// Round-half-down integer division: n/d rounds to the nearest integer with
// ties towards zero (the rounding NVIDIA's fixed-point blend uses at every
// stage).
__device__ __forceinline__ int rhdDiv(int n, int d) {
  return (2 * n + d - 1) / (2 * d);
}

__device__ __forceinline__ int clamp255(int v) {
  return v < 0 ? 0: (v > 255 ? 255: v);
}

// dst = (src1 == key on every channel) ? src2: src1
template <typename T, int Channels>
__global__ void compColorKeyKernel(const T *pSrc1, int nSrc1Step, const T *pSrc2, int nSrc2Step, T *pDst,
                                   int nDstStep, NppiSize oSizeROI, ColorKeyVec<T, Channels> key) {
  int x = 0, y = 0;
  if (!npp_kernel::getThreadCoords(x, y, oSizeROI.width, oSizeROI.height)) {
    return;
  }
  const T *src1 = npp_kernel::getSrcRow(pSrc1, y, nSrc1Step) + x * Channels;
  const T *src2 = npp_kernel::getSrcRow(pSrc2, y, nSrc2Step) + x * Channels;
  T *dst = npp_kernel::getDstRow(pDst, y, nDstStep) + x * Channels;

  bool match = true;
#pragma unroll
  for (int c = 0; c < Channels; ++c) {
    match = match && (src1[c] == key.v[c]);
  }
  const T *from = match ? src2: src1;
#pragma unroll
  for (int c = 0; c < Channels; ++c) {
    dst[c] = from[c];
  }
}


// Blend coefficients, measured in the reference implementation for
// nppiAlphaCompColorKey_8u_AC4R_Ctx: the per-image opacities are fixed-point
// values in [0, 256/256] where a raw alpha of 255 is promoted to 256 (i.e.
// exactly 1.0), and the products are rounded half-down to a coefficient in
// [0, 256/256] as well:
//   c2 = (1 - a1) * a2          src2 weight of OVER / ATOP
//   c1 = a1 * a2                IN
//   c0 = a1 * (1 - a2)          OUT
// The XOR kernel clamps its coefficients at 255/256: with (alpha1, alpha2) =
// (255, 0) or (0, 255) the product is exactly 65536 and NVIDIA's XOR path
// behaves as if the coefficient were 255, while OUT/IN keep using 256.
struct AlphaCoeffs {
  int a1, a2;   // promoted opacities in [0, 256]
  int c0, c1;   // OUT / IN coefficients in [0, 256]
  int c2;       // OVER/ATOP src2 coefficient in [0, 256]
  int c0x, c2x; // XOR coefficients clamped at 255
};

__device__ __forceinline__ AlphaCoeffs alphaCoeffs(int nAlpha1, int nAlpha2) {
  AlphaCoeffs k;
  k.a1 = nAlpha1 == 255 ? 256: nAlpha1;
  k.a2 = nAlpha2 == 255 ? 256: nAlpha2;
  k.c2 = rhdDiv((256 - k.a1) * k.a2, 256);
  k.c1 = rhdDiv(k.a1 * k.a2, 256);
  k.c0 = rhdDiv(k.a1 * (256 - k.a2), 256);
  k.c0x = k.c0 > 255 ? 255: k.c0;
  k.c2x = k.c2 > 255 ? 255: k.c2;
  return k;
}

// RGB blend of one channel for operator op; measured fixed-point model, every
// division rounds half-down and the final result saturates to [0, 255].
__device__ __forceinline__ int alphaBlendChannel(int op, const AlphaCoeffs &k, int s1, int s2) {
  int r;
  switch (op) {
  case NPPI_OP_ALPHA_OVER:
    r = rhdDiv(s1 * k.a1, 256) + rhdDiv(s2 * k.c2, 256);
    break;
  case NPPI_OP_ALPHA_IN:
    r = rhdDiv(s1 * k.c1, 256);
    break;
  case NPPI_OP_ALPHA_OUT:
    r = rhdDiv(s1 * k.c0, 256);
    break;
  case NPPI_OP_ALPHA_ATOP:
    r = rhdDiv(rhdDiv(s1 * k.a1, 256) * k.a2 + s2 * k.c2, 256);
    break;
  case NPPI_OP_ALPHA_XOR:
    r = rhdDiv(s1 * k.c0x + s2 * k.c2x, 256);
    break;
  default: // NPPI_OP_ALPHA_PLUS
    r = rhdDiv(s1 * k.a1, 256) + rhdDiv(s2 * k.a2, 256);
    break;
  }
  return clamp255(r);
}

// dst.rgb = (src1 == key) ? src2.rgb: blend(src1, src2); dst.alpha is only
// written for key matches (copied from src2), otherwise left untouched.
template <typename T, int Channels>
__global__ void alphaCompColorKeyKernel(const T *pSrc1, int nSrc1Step, int nAlpha1, const T *pSrc2, int nSrc2Step,
                                        int nAlpha2, T *pDst, int nDstStep, NppiSize oSizeROI,
                                        ColorKeyVec<T, Channels> key, int nAlphaOp) {
  int x = 0, y = 0;
  if (!npp_kernel::getThreadCoords(x, y, oSizeROI.width, oSizeROI.height)) {
    return;
  }
  const T *src1 = npp_kernel::getSrcRow(pSrc1, y, nSrc1Step) + x * Channels;
  const T *src2 = npp_kernel::getSrcRow(pSrc2, y, nSrc2Step) + x * Channels;
  T *dst = npp_kernel::getDstRow(pDst, y, nDstStep) + x * Channels;

  bool match = true;
#pragma unroll
  for (int c = 0; c < Channels; ++c) {
    match = match && (src1[c] == key.v[c]);
  }
  if (match) {
#pragma unroll
    for (int c = 0; c < Channels; ++c) {
      dst[c] = src2[c];
    }
    return;
  }

  const AlphaCoeffs k = alphaCoeffs(nAlpha1, nAlpha2);
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    dst[c] = static_cast<T>(alphaBlendChannel(nAlphaOp, k, src1[c], src2[c]));
  }
}

// Shared step validation, measured on the reference for both families:
//   * step == 0 -> NPP_STEP_ERROR (an empty ROI returns before this);
//   * the comparison against the ROI row length is UNSIGNED, so a negative
//     step passes; 4-byte pixel layouts then require step % 4 == 0
//     (NPP_NOT_EVEN_STEP_ERROR) while 8u C3 has no alignment requirement;
//   * a negative step that survives is a successful no-op (we never launch
//     with it; NVIDIA would walk out-of-bounds rows).
// Returns the status the caller should hand back; *launch tells it whether
// the kernel still has to run.
NppStatus validateSteps(int nSrc1Step, int nSrc2Step, int nDstStep, int minStep, int align, bool *launch) {
  *launch = false;
  if (nSrc1Step == 0 || nSrc2Step == 0 || nDstStep == 0) {
    return NPP_STEP_ERROR;
  }
  if (static_cast<unsigned>(nSrc1Step) < static_cast<unsigned>(minStep) ||
      static_cast<unsigned>(nSrc2Step) < static_cast<unsigned>(minStep) ||
      static_cast<unsigned>(nDstStep) < static_cast<unsigned>(minStep)) {
    return NPP_STEP_ERROR;
  }
  if (align > 1 && ((nSrc1Step % align) != 0 || (nSrc2Step % align) != 0 || (nDstStep % align) != 0)) {
    return NPP_NOT_EVEN_STEP_ERROR;
  }
  if (nSrc1Step < 0 || nSrc2Step < 0 || nDstStep < 0) {
    return NPP_NO_ERROR;
  }
  *launch = true;
  return NPP_NO_ERROR;
}

// Step alignment requirement of a pixel layout: 4-byte pixels must land on a
// 4-byte row boundary, 3-byte (8u C3) pixels are unrestricted.
constexpr int stepAlignment(int channels, int typeSize) {
  return (channels * typeSize) % 4 == 0 ? 4: 1;
}

}  // namespace

template <typename T, int Channels>
NppStatus compColorKey(const T *pSrc1, int nSrc1Step, const T *pSrc2, int nSrc2Step, T *pDst, int nDstStep,
                       NppiSize oSizeROI, const T *pColorKey, cudaStream_t stream) {
  if (pSrc1 == nullptr || pSrc2 == nullptr || pDst == nullptr || pColorKey == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  const int minStep = oSizeROI.width * Channels * static_cast<int>(sizeof(T));
  bool launch = false;
  const NppStatus stepStatus = validateSteps(nSrc1Step, nSrc2Step, nDstStep, minStep,
                                             stepAlignment(Channels, static_cast<int>(sizeof(T))), &launch);
  if (!launch) {
    return stepStatus;
  }

  ColorKeyVec<T, Channels> key;
  for (int c = 0; c < Channels; ++c) {
    key.v[c] = pColorKey[c];
  }

  const dim3 blockSize = npp_kernel::BlockConfig<16, 16>::blockSize();
  const dim3 gridSize = npp_kernel::BlockConfig<16, 16>::gridSize(oSizeROI.width, oSizeROI.height);
  compColorKeyKernel<T, Channels>
      <<<gridSize, blockSize, 0, stream>>>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, key);
  return npp_kernel::checkCudaKernelError();
}

template <typename T, int Channels>
NppStatus alphaCompColorKey(const T *pSrc1, int nSrc1Step, int nAlpha1, const T *pSrc2, int nSrc2Step, int nAlpha2,
                            T *pDst, int nDstStep, NppiSize oSizeROI, const T *pColorKey, NppiAlphaOp nppAlphaOp,
                            cudaStream_t stream) {
  static_assert(Channels == 4, "alphaCompColorKey exists for AC4 layouts only");
  // Premultiplied and out-of-range operators are measured no-ops that skip
  // every other check (NVIDIA returns NPP_NO_ERROR even for null pointers).
  const int op = static_cast<int>(nppAlphaOp);
  if (op < NPPI_OP_ALPHA_OVER || op > NPPI_OP_ALPHA_PLUS) {
    return NPP_NO_ERROR;
  }
  if (pSrc1 == nullptr || pSrc2 == nullptr || pDst == nullptr || pColorKey == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  const int minStep = oSizeROI.width * Channels * static_cast<int>(sizeof(T));
  bool launch = false;
  const NppStatus stepStatus = validateSteps(nSrc1Step, nSrc2Step, nDstStep, minStep,
                                             stepAlignment(Channels, static_cast<int>(sizeof(T))), &launch);
  if (!launch) {
    return stepStatus;
  }

  ColorKeyVec<T, Channels> key;
  for (int c = 0; c < Channels; ++c) {
    key.v[c] = pColorKey[c];
  }

  const dim3 blockSize = npp_kernel::BlockConfig<16, 16>::blockSize();
  const dim3 gridSize = npp_kernel::BlockConfig<16, 16>::gridSize(oSizeROI.width, oSizeROI.height);
  alphaCompColorKeyKernel<T, Channels><<<gridSize, blockSize, 0, stream>>>(
      pSrc1, nSrc1Step, nAlpha1, pSrc2, nSrc2Step, nAlpha2, pDst, nDstStep, oSizeROI, key, op);
  return npp_kernel::checkCudaKernelError();
}

// Explicit instantiations for the generated wrappers.
#define NPPI_INSTANTIATE_COMPCOLORKEY(T)                                                          \
  template NppStatus compColorKey<T, 1>(const T *, int, const T *, int, T *, int, NppiSize, const T *,  \
                                        cudaStream_t);                                            \
  template NppStatus compColorKey<T, 3>(const T *, int, const T *, int, T *, int, NppiSize, const T *,  \
                                        cudaStream_t);                                            \
  template NppStatus compColorKey<T, 4>(const T *, int, const T *, int, T *, int, NppiSize, const T *,  \
                                        cudaStream_t)

NPPI_INSTANTIATE_COMPCOLORKEY(Npp8u);

#define NPPI_INSTANTIATE_ALPHACOMPCOLORKEY(T)                                                     \
  template NppStatus alphaCompColorKey<T, 4>(const T *, int, int, const T *, int, int, T *, int,   \
                                             NppiSize, const T *, NppiAlphaOp, cudaStream_t)

NPPI_INSTANTIATE_ALPHACOMPCOLORKEY(Npp8u);

}  // namespace color_key
}  // namespace nppi
