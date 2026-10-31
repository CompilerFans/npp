// Fixed-kernel filter kernels and executor entry points.
// See nppi_filter_fixed_executor.h for the family contract.
//
// Kernel tables and the fixed-point emission are the 5090-measured NVIDIA NPP
// 13.4 behaviour (probe-verified on random and delta images for every dtype):
//   * HighPass/Laplace accumulate exact integer kernels (denominator 1).
//   * LowPass 3x3/5x5 emit trunc((sum + bias) / n) with bias n/2 (4 resp. 12)
//     -- C truncation toward zero, decisive on 16s inputs.
//   * Sharpen emits trunc((16*center - sum8 + 3) / 8).
//   * 32f runs float arithmetic without the integer rounding bias.
#include "npp.h"
#include "nppi_filter_fixed_executor.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <type_traits>

namespace nppi {
namespace filter {
namespace {

constexpr int kBlockX = 16;
constexpr int kBlockY = 16;

// Convolution coefficients k[a][b] (a = row offset, top row first) plus the
// fixed-point emission (numerator + bias, then C-truncating division by denom).
struct FixedCoef {
  int size;      // 3 or 5
  int k[5][5];
  int bias;      // integer emission only
  int denom;     // 1 for exact kernels
};

inline FixedCoef coef3(const int *k9, int bias, int denom) {
  FixedCoef c{};
  c.size = 3;
  c.bias = bias;
  c.denom = denom;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) c.k[a][b] = k9[a * 3 + b];
  return c;
}

inline FixedCoef coef5(const int *k25, int bias, int denom) {
  FixedCoef c{};
  c.size = 5;
  c.bias = bias;
  c.denom = denom;
  for (int a = 0; a < 5; ++a)
    for (int b = 0; b < 5; ++b) c.k[a][b] = k25[a * 5 + b];
  return c;
}

inline FixedCoef fixed_coef(FixedOp op, int maskSize) {
  // HighPass: n*n*center - sum(window) with n = 3 / 5, i.e. a zero-sum kernel
  // with center tap n*n-1 (8 resp. 24) and -1 elsewhere (probe-verified).
  static const int HighPass3[9] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
  static const int HighPass5[25] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 24, -1, -1,
                                    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1};
  // LowPass: box sum / n.
  static const int Box3[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  static const int Box5[25] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  // Sharpen: (16*center - sum(8 neighbours)) / 8.
  static const int Sharpen3[9] = {-1, -1, -1, -1, 16, -1, -1, -1, -1};
  // Laplace (zero-sum second derivative).
  static const int Laplace3[9] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
  static const int Laplace5[25] = {-1, -3, -4, -3, -1, -3, 0, 6,  0,  -3, -4, 6,  20, 6,  -4,
                                   -3, 0,  6,  0,  -3, -1, -3, -4, -3, -1};
  const int size = (op == kSharpen) ? 3 : maskSize;
  switch (op) {
    case kHighPass: return size == 5 ? coef5(HighPass5, 0, 1) : coef3(HighPass3, 0, 1);
    case kLowPass: return size == 5 ? coef5(Box5, 12, 25) : coef3(Box3, 4, 9);
    case kSharpen: return coef3(Sharpen3, 3, 8);
    case kLaplace: return size == 5 ? coef5(Laplace5, 0, 1) : coef3(Laplace3, 0, 1);
  }
  return coef3(Laplace3, 0, 1);
}

// Saturating fixed-point emission for integer destinations:
// trunc((num + bias) / denom) clamped to the destination range.
template <typename Dst>
__host__ __device__ __forceinline__ Dst fixed_emit(int num, int bias, int denom) {
  const int v = (num + bias) / denom;
  if constexpr (std::is_same_v<Dst, Npp8u>) {
    return v <= 0 ? (Npp8u)0 : (v >= 255 ? (Npp8u)255 : (Npp8u)v);
  } else if constexpr (std::is_same_v<Dst, Npp16u>) {
    return v <= 0 ? (Npp16u)0 : (v >= 65535 ? (Npp16u)65535 : (Npp16u)v);
  } else if constexpr (std::is_same_v<Dst, Npp16s>) {
    return v <= -32768 ? (Npp16s)-32768 : (v >= 32767 ? (Npp16s)32767 : (Npp16s)v);
  } else {
    static_assert(sizeof(Dst) == 0, "unexpected integer destination type");
  }
}

// 2D ROI convolution kernel.
//   * Border=false: raw reads pSrc[y+a-r][x+b-r]; out-of-buffer reads are
//     caller-visible garbage, matching NVIDIA NPP.
//   * Border=true (NPP_BORDER_REPLICATE): reads clamp to the source image
//     bounds expressed in pSrc-relative coordinates,
//     [-offY, srcH-1-offY] x [-offX, srcW-1-offX], where pSrc points at
//     oSrcOffset inside the oSrcSize image.
template <typename Src, typename Dst, int Channels, bool AlphaPr, bool Border>
__global__ void fixed_kernel(const Src *__restrict__ pSrc, int nSrcStep, Dst *__restrict__ pDst, int nDstStep,
                             int width, int height, FixedCoef coef, int srcW, int srcH, int offX, int offY) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const int r = coef.size >> 1;
  const int nch = AlphaPr ? 3 : Channels;
  const int loX = -offX, hiX = srcW - 1 - offX;
  const int loY = -offY, hiY = srcH - 1 - offY;
  using Acc = std::conditional_t<std::is_floating_point_v<Src>, float, int>;
  Acc acc[4];
  for (int c = 0; c < nch; ++c) acc[c] = Acc(0);
  const char *const srcBase = reinterpret_cast<const char *>(pSrc);
  for (int a = 0; a < coef.size; ++a) {
    int ry = y + a - r;
    if constexpr (Border) ry = ry < loY ? loY : (ry > hiY ? hiY : ry);
    const Src *row = reinterpret_cast<const Src *>(srcBase + static_cast<std::ptrdiff_t>(ry) * nSrcStep);
    for (int b = 0; b < coef.size; ++b) {
      const int kv = coef.k[a][b];
      if (kv == 0) continue;
      int rx = x + b - r;
      if constexpr (Border) rx = rx < loX ? loX : (rx > hiX ? hiX : rx);
      const Src *pix = row + static_cast<std::ptrdiff_t>(rx) * Channels;
      for (int c = 0; c < nch; ++c) acc[c] += static_cast<Acc>(kv) * static_cast<Acc>(pix[c]);
    }
  }
  Dst *drow = reinterpret_cast<Dst *>(reinterpret_cast<char *>(pDst) + static_cast<std::ptrdiff_t>(y) * nDstStep) +
              static_cast<std::ptrdiff_t>(x) * Channels;
  if constexpr (std::is_floating_point_v<Src>) {
    // 32f: plain float arithmetic, no integer rounding bias.
    for (int c = 0; c < nch; ++c)
      drow[c] = static_cast<Dst>(coef.denom == 1 ? acc[c] : acc[c] / static_cast<float>(coef.denom));
  } else {
    for (int c = 0; c < nch; ++c) drow[c] = fixed_emit<Dst>(static_cast<int>(acc[c]), coef.bias, coef.denom);
  }
}

inline NppStatus finish_launch(cudaStream_t stream) {
  cudaStreamSynchronize(stream);
  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

inline bool fixed_mask_size(NppiMaskSize eMaskSize, int *size) {
  if (eMaskSize == NPP_MASK_SIZE_3_X_3) {
    *size = 3;
    return true;
  }
  if (eMaskSize == NPP_MASK_SIZE_5_X_5) {
    *size = 5;
    return true;
  }
  return false;
}

// Shared step validation (unsigned comparison: a negative step passes, exactly
// like NVIDIA NPP).  dst step must cover the ROI row and be a multiple of the
// alignment (element size for C1/C3, pixel size for C4/AC4).
template <typename Dst, int Channels>
NppStatus check_dst_step(int nDstStep, int roiWidth) {
  const int pix = static_cast<int>(sizeof(Dst)) * Channels;
  const int align = Channels == 4 ? pix : static_cast<int>(sizeof(Dst));
  const unsigned uStep = static_cast<unsigned>(nDstStep);
  if (uStep < static_cast<unsigned>(roiWidth * pix)) return NPP_STEP_ERROR;
  if (uStep % static_cast<unsigned>(align) != 0) return NPP_NOT_EVEN_STEP_ERROR;
  return NPP_NO_ERROR;
}

template <typename Src, typename Dst, int Channels, bool AlphaPr, bool Border>
NppStatus launch(const Src *pSrc, int nSrcStep, Dst *pDst, int nDstStep, NppiSize oSizeROI, const FixedCoef &coef,
                 NppiSize oSrcSize, NppiPoint oSrcOffset, cudaStream_t stream) {
  NppStatus st = check_dst_step<Dst, Channels>(nDstStep, oSizeROI.width);
  if (st != NPP_NO_ERROR) return st;
  (void)cudaGetLastError();
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + kBlockX - 1) / kBlockX, (oSizeROI.height + kBlockY - 1) / kBlockY);
  fixed_kernel<Src, Dst, Channels, AlphaPr, Border>
      <<<grid, block, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, coef,
                                   oSrcSize.width, oSrcSize.height, oSrcOffset.x, oSrcOffset.y);
  return finish_launch(stream);
}

}  // namespace

template <typename Src, typename Dst, int Channels, bool AlphaPreserve, FixedOp Op>
NppStatus filterFixedMask(const Src *pSrc, int nSrcStep, Dst *pDst, int nDstStep, NppiSize oSizeROI,
                          NppiMaskSize eMaskSize, cudaStream_t stream) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  int maskSize = 0;
  if (!fixed_mask_size(eMaskSize, &maskSize)) return NPP_MASK_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  return launch<Src, Dst, Channels, AlphaPreserve, false>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                          fixed_coef(Op, maskSize), NppiSize{0, 0}, NppiPoint{0, 0},
                                                          stream);
}

template <typename Src, typename Dst, int Channels, bool AlphaPreserve, FixedOp Op>
NppStatus filterFixed(const Src *pSrc, int nSrcStep, Dst *pDst, int nDstStep, NppiSize oSizeROI,
                      cudaStream_t stream) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  return launch<Src, Dst, Channels, AlphaPreserve, false>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                          fixed_coef(Op, 3), NppiSize{0, 0}, NppiPoint{0, 0},
                                                          stream);
}

template <typename Src, typename Dst, int Channels, bool AlphaPreserve, FixedOp Op>
NppStatus filterFixedMaskBorder(const Src *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Dst *pDst,
                                int nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiBorderType eBorderType,
                                cudaStream_t stream) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSrcSize.width <= 0 || oSrcSize.height <= 0) return NPP_SIZE_ERROR;
  if (oSrcOffset.x < 0 || oSrcOffset.y < 0 || oSrcOffset.x >= oSrcSize.width || oSrcOffset.y >= oSrcSize.height)
    return NPP_OUT_OFF_RANGE_ERROR;
  if (eBorderType != NPP_BORDER_REPLICATE) return NPP_NOT_SUPPORTED_MODE_ERROR;
  int maskSize = 0;
  if (!fixed_mask_size(eMaskSize, &maskSize)) return NPP_MASK_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  return launch<Src, Dst, Channels, AlphaPreserve, true>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                         fixed_coef(Op, maskSize), oSrcSize, oSrcOffset, stream);
}

template <typename Src, typename Dst, int Channels, bool AlphaPreserve, FixedOp Op>
NppStatus filterFixedBorder(const Src *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Dst *pDst,
                            int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, cudaStream_t stream) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSrcSize.width <= 0 || oSrcSize.height <= 0) return NPP_SIZE_ERROR;
  if (oSrcOffset.x < 0 || oSrcOffset.y < 0 || oSrcOffset.x >= oSrcSize.width || oSrcOffset.y >= oSrcSize.height)
    return NPP_OUT_OFF_RANGE_ERROR;
  if (eBorderType != NPP_BORDER_REPLICATE) return NPP_NOT_SUPPORTED_MODE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  return launch<Src, Dst, Channels, AlphaPreserve, true>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                         fixed_coef(Op, 3), oSrcSize, oSrcOffset, stream);
}

}  // namespace filter
}  // namespace nppi

// ---------------------------------------------------------------------------
// Explicit instantiations (f11 package: HighPass/LowPass/Sharpen/Laplace,
// plain + Border, 128 functions).
// ---------------------------------------------------------------------------
// kHighPass
template NppStatus nppi::filter::filterFixedMask<Npp8u, Npp8u, 1, 0, nppi::filter::kHighPass>(const Npp8u*, int, Npp8u*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp8u, Npp8u, 3, 0, nppi::filter::kHighPass>(const Npp8u*, int, Npp8u*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp8u, Npp8u, 4, 0, nppi::filter::kHighPass>(const Npp8u*, int, Npp8u*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp8u, Npp8u, 4, 1, nppi::filter::kHighPass>(const Npp8u*, int, Npp8u*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp16u, Npp16u, 1, 0, nppi::filter::kHighPass>(const Npp16u*, int, Npp16u*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp16u, Npp16u, 3, 0, nppi::filter::kHighPass>(const Npp16u*, int, Npp16u*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp16u, Npp16u, 4, 0, nppi::filter::kHighPass>(const Npp16u*, int, Npp16u*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp16u, Npp16u, 4, 1, nppi::filter::kHighPass>(const Npp16u*, int, Npp16u*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp16s, Npp16s, 1, 0, nppi::filter::kHighPass>(const Npp16s*, int, Npp16s*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp16s, Npp16s, 3, 0, nppi::filter::kHighPass>(const Npp16s*, int, Npp16s*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp16s, Npp16s, 4, 0, nppi::filter::kHighPass>(const Npp16s*, int, Npp16s*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp16s, Npp16s, 4, 1, nppi::filter::kHighPass>(const Npp16s*, int, Npp16s*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp32f, Npp32f, 1, 0, nppi::filter::kHighPass>(const Npp32f*, int, Npp32f*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp32f, Npp32f, 3, 0, nppi::filter::kHighPass>(const Npp32f*, int, Npp32f*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp32f, Npp32f, 4, 0, nppi::filter::kHighPass>(const Npp32f*, int, Npp32f*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp32f, Npp32f, 4, 1, nppi::filter::kHighPass>(const Npp32f*, int, Npp32f*, int, NppiSize, NppiMaskSize, cudaStream_t);
// kLowPass
template NppStatus nppi::filter::filterFixedMask<Npp8u, Npp8u, 1, 0, nppi::filter::kLowPass>(const Npp8u*, int, Npp8u*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp8u, Npp8u, 3, 0, nppi::filter::kLowPass>(const Npp8u*, int, Npp8u*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp8u, Npp8u, 4, 0, nppi::filter::kLowPass>(const Npp8u*, int, Npp8u*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp8u, Npp8u, 4, 1, nppi::filter::kLowPass>(const Npp8u*, int, Npp8u*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp16u, Npp16u, 1, 0, nppi::filter::kLowPass>(const Npp16u*, int, Npp16u*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp16u, Npp16u, 3, 0, nppi::filter::kLowPass>(const Npp16u*, int, Npp16u*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp16u, Npp16u, 4, 0, nppi::filter::kLowPass>(const Npp16u*, int, Npp16u*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp16u, Npp16u, 4, 1, nppi::filter::kLowPass>(const Npp16u*, int, Npp16u*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp16s, Npp16s, 1, 0, nppi::filter::kLowPass>(const Npp16s*, int, Npp16s*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp16s, Npp16s, 3, 0, nppi::filter::kLowPass>(const Npp16s*, int, Npp16s*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp16s, Npp16s, 4, 0, nppi::filter::kLowPass>(const Npp16s*, int, Npp16s*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp16s, Npp16s, 4, 1, nppi::filter::kLowPass>(const Npp16s*, int, Npp16s*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp32f, Npp32f, 1, 0, nppi::filter::kLowPass>(const Npp32f*, int, Npp32f*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp32f, Npp32f, 3, 0, nppi::filter::kLowPass>(const Npp32f*, int, Npp32f*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp32f, Npp32f, 4, 0, nppi::filter::kLowPass>(const Npp32f*, int, Npp32f*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp32f, Npp32f, 4, 1, nppi::filter::kLowPass>(const Npp32f*, int, Npp32f*, int, NppiSize, NppiMaskSize, cudaStream_t);
// kHighPass (border)
template NppStatus nppi::filter::filterFixedMaskBorder<Npp8u, Npp8u, 1, 0, nppi::filter::kHighPass>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp8u, Npp8u, 3, 0, nppi::filter::kHighPass>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp8u, Npp8u, 4, 0, nppi::filter::kHighPass>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp8u, Npp8u, 4, 1, nppi::filter::kHighPass>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp16u, Npp16u, 1, 0, nppi::filter::kHighPass>(const Npp16u*, int, NppiSize, NppiPoint, Npp16u*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp16u, Npp16u, 3, 0, nppi::filter::kHighPass>(const Npp16u*, int, NppiSize, NppiPoint, Npp16u*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp16u, Npp16u, 4, 0, nppi::filter::kHighPass>(const Npp16u*, int, NppiSize, NppiPoint, Npp16u*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp16u, Npp16u, 4, 1, nppi::filter::kHighPass>(const Npp16u*, int, NppiSize, NppiPoint, Npp16u*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp16s, Npp16s, 1, 0, nppi::filter::kHighPass>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp16s, Npp16s, 3, 0, nppi::filter::kHighPass>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp16s, Npp16s, 4, 0, nppi::filter::kHighPass>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp16s, Npp16s, 4, 1, nppi::filter::kHighPass>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp32f, Npp32f, 1, 0, nppi::filter::kHighPass>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp32f, Npp32f, 3, 0, nppi::filter::kHighPass>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp32f, Npp32f, 4, 0, nppi::filter::kHighPass>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp32f, Npp32f, 4, 1, nppi::filter::kHighPass>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
// kLowPass (border)
template NppStatus nppi::filter::filterFixedMaskBorder<Npp8u, Npp8u, 1, 0, nppi::filter::kLowPass>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp8u, Npp8u, 3, 0, nppi::filter::kLowPass>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp8u, Npp8u, 4, 0, nppi::filter::kLowPass>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp8u, Npp8u, 4, 1, nppi::filter::kLowPass>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp16u, Npp16u, 1, 0, nppi::filter::kLowPass>(const Npp16u*, int, NppiSize, NppiPoint, Npp16u*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp16u, Npp16u, 3, 0, nppi::filter::kLowPass>(const Npp16u*, int, NppiSize, NppiPoint, Npp16u*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp16u, Npp16u, 4, 0, nppi::filter::kLowPass>(const Npp16u*, int, NppiSize, NppiPoint, Npp16u*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp16u, Npp16u, 4, 1, nppi::filter::kLowPass>(const Npp16u*, int, NppiSize, NppiPoint, Npp16u*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp16s, Npp16s, 1, 0, nppi::filter::kLowPass>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp16s, Npp16s, 3, 0, nppi::filter::kLowPass>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp16s, Npp16s, 4, 0, nppi::filter::kLowPass>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp16s, Npp16s, 4, 1, nppi::filter::kLowPass>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp32f, Npp32f, 1, 0, nppi::filter::kLowPass>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp32f, Npp32f, 3, 0, nppi::filter::kLowPass>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp32f, Npp32f, 4, 0, nppi::filter::kLowPass>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp32f, Npp32f, 4, 1, nppi::filter::kLowPass>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
// kSharpen
template NppStatus nppi::filter::filterFixed<Npp8u, Npp8u, 1, 0, nppi::filter::kSharpen>(const Npp8u*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterFixed<Npp8u, Npp8u, 3, 0, nppi::filter::kSharpen>(const Npp8u*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterFixed<Npp8u, Npp8u, 4, 0, nppi::filter::kSharpen>(const Npp8u*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterFixed<Npp8u, Npp8u, 4, 1, nppi::filter::kSharpen>(const Npp8u*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterFixed<Npp16u, Npp16u, 1, 0, nppi::filter::kSharpen>(const Npp16u*, int, Npp16u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterFixed<Npp16u, Npp16u, 3, 0, nppi::filter::kSharpen>(const Npp16u*, int, Npp16u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterFixed<Npp16u, Npp16u, 4, 0, nppi::filter::kSharpen>(const Npp16u*, int, Npp16u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterFixed<Npp16u, Npp16u, 4, 1, nppi::filter::kSharpen>(const Npp16u*, int, Npp16u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterFixed<Npp16s, Npp16s, 1, 0, nppi::filter::kSharpen>(const Npp16s*, int, Npp16s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterFixed<Npp16s, Npp16s, 3, 0, nppi::filter::kSharpen>(const Npp16s*, int, Npp16s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterFixed<Npp16s, Npp16s, 4, 0, nppi::filter::kSharpen>(const Npp16s*, int, Npp16s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterFixed<Npp16s, Npp16s, 4, 1, nppi::filter::kSharpen>(const Npp16s*, int, Npp16s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterFixed<Npp32f, Npp32f, 1, 0, nppi::filter::kSharpen>(const Npp32f*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterFixed<Npp32f, Npp32f, 3, 0, nppi::filter::kSharpen>(const Npp32f*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterFixed<Npp32f, Npp32f, 4, 0, nppi::filter::kSharpen>(const Npp32f*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterFixed<Npp32f, Npp32f, 4, 1, nppi::filter::kSharpen>(const Npp32f*, int, Npp32f*, int, NppiSize, cudaStream_t);
// kSharpen (border)
template NppStatus nppi::filter::filterFixedBorder<Npp8u, Npp8u, 1, 0, nppi::filter::kSharpen>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedBorder<Npp8u, Npp8u, 3, 0, nppi::filter::kSharpen>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedBorder<Npp8u, Npp8u, 4, 0, nppi::filter::kSharpen>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedBorder<Npp8u, Npp8u, 4, 1, nppi::filter::kSharpen>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedBorder<Npp16u, Npp16u, 1, 0, nppi::filter::kSharpen>(const Npp16u*, int, NppiSize, NppiPoint, Npp16u*, int, NppiSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedBorder<Npp16u, Npp16u, 3, 0, nppi::filter::kSharpen>(const Npp16u*, int, NppiSize, NppiPoint, Npp16u*, int, NppiSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedBorder<Npp16u, Npp16u, 4, 0, nppi::filter::kSharpen>(const Npp16u*, int, NppiSize, NppiPoint, Npp16u*, int, NppiSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedBorder<Npp16u, Npp16u, 4, 1, nppi::filter::kSharpen>(const Npp16u*, int, NppiSize, NppiPoint, Npp16u*, int, NppiSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedBorder<Npp16s, Npp16s, 1, 0, nppi::filter::kSharpen>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedBorder<Npp16s, Npp16s, 3, 0, nppi::filter::kSharpen>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedBorder<Npp16s, Npp16s, 4, 0, nppi::filter::kSharpen>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedBorder<Npp16s, Npp16s, 4, 1, nppi::filter::kSharpen>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedBorder<Npp32f, Npp32f, 1, 0, nppi::filter::kSharpen>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedBorder<Npp32f, Npp32f, 3, 0, nppi::filter::kSharpen>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedBorder<Npp32f, Npp32f, 4, 0, nppi::filter::kSharpen>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedBorder<Npp32f, Npp32f, 4, 1, nppi::filter::kSharpen>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppiBorderType, cudaStream_t);
// kLaplace
template NppStatus nppi::filter::filterFixedMask<Npp8u, Npp8u, 1, 0, nppi::filter::kLaplace>(const Npp8u*, int, Npp8u*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp8u, Npp8u, 3, 0, nppi::filter::kLaplace>(const Npp8u*, int, Npp8u*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp8u, Npp8u, 4, 0, nppi::filter::kLaplace>(const Npp8u*, int, Npp8u*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp8u, Npp8u, 4, 1, nppi::filter::kLaplace>(const Npp8u*, int, Npp8u*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp16s, Npp16s, 1, 0, nppi::filter::kLaplace>(const Npp16s*, int, Npp16s*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp16s, Npp16s, 3, 0, nppi::filter::kLaplace>(const Npp16s*, int, Npp16s*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp16s, Npp16s, 4, 0, nppi::filter::kLaplace>(const Npp16s*, int, Npp16s*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp16s, Npp16s, 4, 1, nppi::filter::kLaplace>(const Npp16s*, int, Npp16s*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp32f, Npp32f, 1, 0, nppi::filter::kLaplace>(const Npp32f*, int, Npp32f*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp32f, Npp32f, 3, 0, nppi::filter::kLaplace>(const Npp32f*, int, Npp32f*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp32f, Npp32f, 4, 0, nppi::filter::kLaplace>(const Npp32f*, int, Npp32f*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp32f, Npp32f, 4, 1, nppi::filter::kLaplace>(const Npp32f*, int, Npp32f*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp8u, Npp16s, 1, 0, nppi::filter::kLaplace>(const Npp8u*, int, Npp16s*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterFixedMask<Npp8s, Npp16s, 1, 0, nppi::filter::kLaplace>(const Npp8s*, int, Npp16s*, int, NppiSize, NppiMaskSize, cudaStream_t);
// kLaplace (border)
template NppStatus nppi::filter::filterFixedMaskBorder<Npp8u, Npp8u, 1, 0, nppi::filter::kLaplace>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp8u, Npp8u, 3, 0, nppi::filter::kLaplace>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp8u, Npp8u, 4, 0, nppi::filter::kLaplace>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp8u, Npp8u, 4, 1, nppi::filter::kLaplace>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp16s, Npp16s, 1, 0, nppi::filter::kLaplace>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp16s, Npp16s, 3, 0, nppi::filter::kLaplace>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp16s, Npp16s, 4, 0, nppi::filter::kLaplace>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp16s, Npp16s, 4, 1, nppi::filter::kLaplace>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp32f, Npp32f, 1, 0, nppi::filter::kLaplace>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp32f, Npp32f, 3, 0, nppi::filter::kLaplace>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp32f, Npp32f, 4, 0, nppi::filter::kLaplace>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp32f, Npp32f, 4, 1, nppi::filter::kLaplace>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp8u, Npp16s, 1, 0, nppi::filter::kLaplace>(const Npp8u*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
template NppStatus nppi::filter::filterFixedMaskBorder<Npp8s, Npp16s, 1, 0, nppi::filter::kLaplace>(const Npp8s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t);
