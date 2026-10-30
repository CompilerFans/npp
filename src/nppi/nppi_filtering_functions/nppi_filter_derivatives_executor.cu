// Derivative-filter kernels and executor entry points.
// See nppi_filter_derivatives_executor.h for the family contract.
//
// Kernel coefficient tables are the 5090-measured NVIDIA NPP 13.4 behaviour
// (least-squares fits over probe outputs, exact for integer inputs):
//   * PrewittHoriz and SobelHoriz measure NEGATED w.r.t. the NPP docs
//     (top row -1, bottom row +1).
//   * RobertsDown = center - top-left, RobertsUp = center - top-right.
//   * The 5x5 kernels of the conversion / Mask families coincide with the
//     documented 5x5 Sobel kernels (Horiz sign-flipped).
//   * Second-derivative and Cross kernels use 3x3 taps for NPP_MASK_SIZE_3_X_3
//     (NOT the center of the 5x5 kernel).
#include "npp.h"
#include "nppi_filter_derivatives_executor.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <type_traits>

namespace nppi {
namespace filter {
namespace {

constexpr int kBlockX = 16;
constexpr int kBlockY = 16;

// Convolution coefficients, k[a][b] with a the row offset (top row first).
struct DerivCoef {
  int size;      // 3 or 5
  int k[5][5];
};

inline DerivCoef coef3(const int *k9) {
  DerivCoef c{};
  c.size = 3;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) c.k[a][b] = k9[a * 3 + b];
  return c;
}

inline DerivCoef coef5(const int *k25) {
  DerivCoef c{};
  c.size = 5;
  for (int a = 0; a < 5; ++a)
    for (int b = 0; b < 5; ++b) c.k[a][b] = k25[a * 5 + b];
  return c;
}

inline DerivCoef deriv_coef(DerivOp op, int maskSize) {
  static const int PrewittHoriz3[9] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
  static const int PrewittVert3[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
  static const int RobertsDown3[9] = {-1, 0, 0, 0, 1, 0, 0, 0, 0};
  static const int RobertsUp3[9] = {0, 0, -1, 0, 1, 0, 0, 0, 0};
  static const int SobelHoriz3[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
  static const int SobelVert3[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  static const int SobelHoriz5[25] = {-1, -4, -6, -4, -1, -2, -8, -12, -8, -2, 0, 0, 0, 0, 0,
                                      2, 8,  12, 8,  2,  1,  4,  6,   4,  1};
  static const int SobelVert5[25] = {-1, -2, 0,  2,  1,  -4, -8, 0,  8,  4,  -6, -12, 0,  12, 6,
                                     -4, -8, 0,  8,  4,  -1, -2, 0,  2,  1};
  static const int Cross3[9] = {-1, 0, 1, 0, 0, 0, 1, 0, -1};
  static const int Cross5[25] = {-1, -2, 0,  2,  1,  -2, -4, 0,  4,  2, 0, 0,  0,  0,  0,
                                 2,  4,  0,  -4, -2, 1,   2,  0,  -2, -1};
  static const int HorizSecond3[9] = {1, 2, 1, -2, -4, -2, 1, 2, 1};
  static const int HorizSecond5[25] = {1, 4, 6,  4,  1, 0, 0, 0,  0,  0, -2, -8, -12, -8, -2,
                                       0, 0, 0,  0,  0, 1, 4, 6,  4,  1};
  static const int VertSecond3[9] = {1, -2, 1, 2, -4, 2, 1, -2, 1};
  static const int VertSecond5[25] = {1, 0, -2,  0, 1, 4, 0, -8,  0, 4, 6, 0, -12, 0, 6,
                                      4, 0, -8,  0, 4, 1, 0, -2,  0, 1};
  switch (op) {
    case kPrewittHoriz: return coef3(PrewittHoriz3);
    case kPrewittVert: return coef3(PrewittVert3);
    case kRobertsDown: return coef3(RobertsDown3);
    case kRobertsUp: return coef3(RobertsUp3);
    case kSobelHoriz: return maskSize == 5 ? coef5(SobelHoriz5) : coef3(SobelHoriz3);
    case kSobelVert: return maskSize == 5 ? coef5(SobelVert5) : coef3(SobelVert3);
    case kSobelCross: return maskSize == 5 ? coef5(Cross5) : coef3(Cross3);
    case kSobelHorizSecond: return maskSize == 5 ? coef5(HorizSecond5) : coef3(HorizSecond3);
    case kSobelVertSecond: return maskSize == 5 ? coef5(VertSecond5) : coef3(VertSecond3);
  }
  return coef3(SobelHoriz3);
}

// Saturating emission: 8u -> [0,255], 16s -> [-32768,32767], float passthrough.
template <typename Dst>
__host__ __device__ __forceinline__ Dst deriv_emit(float v) {
  if constexpr (std::is_same_v<Dst, Npp8u>) {
    return v <= 0.0f ? (Npp8u)0 : (v >= 255.0f ? (Npp8u)255 : (Npp8u)v);
  } else if constexpr (std::is_same_v<Dst, Npp16s>) {
    return v <= -32768.0f ? (Npp16s)-32768 : (v >= 32767.0f ? (Npp16s)32767 : (Npp16s)v);
  } else {
    return static_cast<Dst>(v);
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
__global__ void deriv_kernel(const Src *__restrict__ pSrc, int nSrcStep, Dst *__restrict__ pDst, int nDstStep,
                             int width, int height, DerivCoef coef, int srcW, int srcH, int offX, int offY) {
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
  for (int c = 0; c < nch; ++c) drow[c] = deriv_emit<Dst>(static_cast<float>(acc[c]));
}

inline NppStatus finish_launch(cudaStream_t stream) {
  cudaStreamSynchronize(stream);
  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

inline bool deriv_mask_size(NppiMaskSize eMaskSize, int *size) {
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

// Shared step validation: dst step must cover the ROI row and be a multiple of
// the alignment (element size for C1/C3, pixel size for C4/AC4).
template <typename Dst, int Channels>
NppStatus check_dst_step(int nDstStep, int roiWidth) {
  const int pix = static_cast<int>(sizeof(Dst)) * Channels;
  const int align = Channels == 4 ? pix : static_cast<int>(sizeof(Dst));
  if (nDstStep < roiWidth * pix) return NPP_STEP_ERROR;
  if (nDstStep % align != 0) return NPP_NOT_EVEN_STEP_ERROR;
  return NPP_NO_ERROR;
}

template <typename Src, typename Dst, int Channels, bool AlphaPr, bool Border>
NppStatus launch(const Src *pSrc, int nSrcStep, Dst *pDst, int nDstStep, NppiSize oSizeROI, const DerivCoef &coef,
                 NppiSize oSrcSize, NppiPoint oSrcOffset, cudaStream_t stream) {
  NppStatus st = check_dst_step<Dst, Channels>(nDstStep, oSizeROI.width);
  if (st != NPP_NO_ERROR) return st;
  (void)cudaGetLastError();
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + kBlockX - 1) / kBlockX, (oSizeROI.height + kBlockY - 1) / kBlockY);
  deriv_kernel<Src, Dst, Channels, AlphaPr, Border>
      <<<grid, block, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, coef,
                                   oSrcSize.width, oSrcSize.height, oSrcOffset.x, oSrcOffset.y);
  return finish_launch(stream);
}

}  // namespace

template <typename Src, typename Dst, int Channels, bool AlphaPreserve, DerivOp Op>
NppStatus filterDeriv(const Src *pSrc, int nSrcStep, Dst *pDst, int nDstStep, NppiSize oSizeROI,
                      cudaStream_t stream) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  return launch<Src, Dst, Channels, AlphaPreserve, false>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, deriv_coef(Op, 3),
                                                          NppiSize{0, 0}, NppiPoint{0, 0}, stream);
}

template <typename Src, typename Dst, int Channels, bool AlphaPreserve, DerivOp Op>
NppStatus filterDerivBorder(const Src *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Dst *pDst,
                            int nDstStep, NppiSize oSizeROI, NppBorderType eBorderType, cudaStream_t stream) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  if (oSrcSize.width <= 0 || oSrcSize.height <= 0) return NPP_OUT_OFF_RANGE_ERROR;
  if (oSrcOffset.x < 0 || oSrcOffset.y < 0) return NPP_OUT_OFF_RANGE_ERROR;
  if (eBorderType != NPP_BORDER_REPLICATE) return NPP_NOT_SUPPORTED_MODE_ERROR;
  return launch<Src, Dst, Channels, AlphaPreserve, true>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, deriv_coef(Op, 3),
                                                         oSrcSize, oSrcOffset, stream);
}

template <typename Src, typename Dst, DerivOp Op>
NppStatus filterDerivMask(const Src *pSrc, int nSrcStep, Dst *pDst, int nDstStep, NppiSize oSizeROI,
                          NppiMaskSize eMaskSize, cudaStream_t stream) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  int maskSize = 0;
  if (!deriv_mask_size(eMaskSize, &maskSize)) return NPP_MASK_SIZE_ERROR;
  return launch<Src, Dst, 1, false, false>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, deriv_coef(Op, maskSize),
                                           NppiSize{0, 0}, NppiPoint{0, 0}, stream);
}

template <typename Src, typename Dst, DerivOp Op>
NppStatus filterDerivMaskBorder(const Src *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Dst *pDst,
                                int nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppBorderType eBorderType,
                                cudaStream_t stream) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  if (oSrcSize.width <= 0 || oSrcSize.height <= 0) return NPP_OUT_OFF_RANGE_ERROR;
  if (oSrcOffset.x < 0 || oSrcOffset.y < 0) return NPP_OUT_OFF_RANGE_ERROR;
  if (eBorderType != NPP_BORDER_REPLICATE) return NPP_NOT_SUPPORTED_MODE_ERROR;
  int maskSize = 0;
  if (!deriv_mask_size(eMaskSize, &maskSize)) return NPP_MASK_SIZE_ERROR;
  return launch<Src, Dst, 1, false, true>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, deriv_coef(Op, maskSize),
                                          oSrcSize, oSrcOffset, stream);
}

}  // namespace filter
}  // namespace nppi

// ---------------------------------------------------------------------------
// Explicit instantiations (f4 package: 22 families, 348 functions).
// ---------------------------------------------------------------------------
// kPrewittHoriz
template NppStatus nppi::filter::filterDeriv<Npp8u, Npp8u, 1, false, kPrewittHoriz, nppi::filter::nppi::filter::kPrewittHoriz>(const Npp8u*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp8u, Npp8u, 1, false, kPrewittHoriz, nppi::filter::nppi::filter::kPrewittHoriz>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp8u, Npp8u, 3, false, kPrewittHoriz, nppi::filter::nppi::filter::kPrewittHoriz>(const Npp8u*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp8u, Npp8u, 3, false, kPrewittHoriz, nppi::filter::nppi::filter::kPrewittHoriz>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp8u, Npp8u, 4, false, kPrewittHoriz, nppi::filter::nppi::filter::kPrewittHoriz>(const Npp8u*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp8u, Npp8u, 4, false, kPrewittHoriz, nppi::filter::nppi::filter::kPrewittHoriz>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp8u, Npp8u, 4, true, kPrewittHoriz, nppi::filter::nppi::filter::kPrewittHoriz>(const Npp8u*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp8u, Npp8u, 4, true, kPrewittHoriz, nppi::filter::nppi::filter::kPrewittHoriz>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp16s, Npp16s, 1, false, kPrewittHoriz, nppi::filter::nppi::filter::kPrewittHoriz>(const Npp16s*, int, Npp16s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp16s, Npp16s, 1, false, kPrewittHoriz, nppi::filter::nppi::filter::kPrewittHoriz>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp16s, Npp16s, 3, false, kPrewittHoriz, nppi::filter::nppi::filter::kPrewittHoriz>(const Npp16s*, int, Npp16s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp16s, Npp16s, 3, false, kPrewittHoriz, nppi::filter::nppi::filter::kPrewittHoriz>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp16s, Npp16s, 4, false, kPrewittHoriz, nppi::filter::nppi::filter::kPrewittHoriz>(const Npp16s*, int, Npp16s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp16s, Npp16s, 4, false, kPrewittHoriz, nppi::filter::nppi::filter::kPrewittHoriz>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp16s, Npp16s, 4, true, kPrewittHoriz, nppi::filter::nppi::filter::kPrewittHoriz>(const Npp16s*, int, Npp16s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp16s, Npp16s, 4, true, kPrewittHoriz, nppi::filter::nppi::filter::kPrewittHoriz>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp32f, Npp32f, 1, false, kPrewittHoriz, nppi::filter::nppi::filter::kPrewittHoriz>(const Npp32f*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp32f, Npp32f, 1, false, kPrewittHoriz, nppi::filter::nppi::filter::kPrewittHoriz>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp32f, Npp32f, 3, false, kPrewittHoriz, nppi::filter::nppi::filter::kPrewittHoriz>(const Npp32f*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp32f, Npp32f, 3, false, kPrewittHoriz, nppi::filter::nppi::filter::kPrewittHoriz>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp32f, Npp32f, 4, false, kPrewittHoriz, nppi::filter::nppi::filter::kPrewittHoriz>(const Npp32f*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp32f, Npp32f, 4, false, kPrewittHoriz, nppi::filter::nppi::filter::kPrewittHoriz>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp32f, Npp32f, 4, true, kPrewittHoriz, nppi::filter::nppi::filter::kPrewittHoriz>(const Npp32f*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp32f, Npp32f, 4, true, kPrewittHoriz, nppi::filter::nppi::filter::kPrewittHoriz>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppBorderType, cudaStream_t);
// kPrewittVert
template NppStatus nppi::filter::filterDeriv<Npp8u, Npp8u, 1, false, kPrewittVert, nppi::filter::nppi::filter::kPrewittVert>(const Npp8u*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp8u, Npp8u, 1, false, kPrewittVert, nppi::filter::nppi::filter::kPrewittVert>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp8u, Npp8u, 3, false, kPrewittVert, nppi::filter::nppi::filter::kPrewittVert>(const Npp8u*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp8u, Npp8u, 3, false, kPrewittVert, nppi::filter::nppi::filter::kPrewittVert>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp8u, Npp8u, 4, false, kPrewittVert, nppi::filter::nppi::filter::kPrewittVert>(const Npp8u*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp8u, Npp8u, 4, false, kPrewittVert, nppi::filter::nppi::filter::kPrewittVert>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp8u, Npp8u, 4, true, kPrewittVert, nppi::filter::nppi::filter::kPrewittVert>(const Npp8u*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp8u, Npp8u, 4, true, kPrewittVert, nppi::filter::nppi::filter::kPrewittVert>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp16s, Npp16s, 1, false, kPrewittVert, nppi::filter::nppi::filter::kPrewittVert>(const Npp16s*, int, Npp16s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp16s, Npp16s, 1, false, kPrewittVert, nppi::filter::nppi::filter::kPrewittVert>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp16s, Npp16s, 3, false, kPrewittVert, nppi::filter::nppi::filter::kPrewittVert>(const Npp16s*, int, Npp16s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp16s, Npp16s, 3, false, kPrewittVert, nppi::filter::nppi::filter::kPrewittVert>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp16s, Npp16s, 4, false, kPrewittVert, nppi::filter::nppi::filter::kPrewittVert>(const Npp16s*, int, Npp16s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp16s, Npp16s, 4, false, kPrewittVert, nppi::filter::nppi::filter::kPrewittVert>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp16s, Npp16s, 4, true, kPrewittVert, nppi::filter::nppi::filter::kPrewittVert>(const Npp16s*, int, Npp16s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp16s, Npp16s, 4, true, kPrewittVert, nppi::filter::nppi::filter::kPrewittVert>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp32f, Npp32f, 1, false, kPrewittVert, nppi::filter::nppi::filter::kPrewittVert>(const Npp32f*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp32f, Npp32f, 1, false, kPrewittVert, nppi::filter::nppi::filter::kPrewittVert>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp32f, Npp32f, 3, false, kPrewittVert, nppi::filter::nppi::filter::kPrewittVert>(const Npp32f*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp32f, Npp32f, 3, false, kPrewittVert, nppi::filter::nppi::filter::kPrewittVert>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp32f, Npp32f, 4, false, kPrewittVert, nppi::filter::nppi::filter::kPrewittVert>(const Npp32f*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp32f, Npp32f, 4, false, kPrewittVert, nppi::filter::nppi::filter::kPrewittVert>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp32f, Npp32f, 4, true, kPrewittVert, nppi::filter::nppi::filter::kPrewittVert>(const Npp32f*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp32f, Npp32f, 4, true, kPrewittVert, nppi::filter::nppi::filter::kPrewittVert>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppBorderType, cudaStream_t);
// kRobertsDown
template NppStatus nppi::filter::filterDeriv<Npp8u, Npp8u, 1, false, kRobertsDown, nppi::filter::nppi::filter::kRobertsDown>(const Npp8u*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp8u, Npp8u, 1, false, kRobertsDown, nppi::filter::nppi::filter::kRobertsDown>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp8u, Npp8u, 3, false, kRobertsDown, nppi::filter::nppi::filter::kRobertsDown>(const Npp8u*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp8u, Npp8u, 3, false, kRobertsDown, nppi::filter::nppi::filter::kRobertsDown>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp8u, Npp8u, 4, false, kRobertsDown, nppi::filter::nppi::filter::kRobertsDown>(const Npp8u*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp8u, Npp8u, 4, false, kRobertsDown, nppi::filter::nppi::filter::kRobertsDown>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp8u, Npp8u, 4, true, kRobertsDown, nppi::filter::nppi::filter::kRobertsDown>(const Npp8u*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp8u, Npp8u, 4, true, kRobertsDown, nppi::filter::nppi::filter::kRobertsDown>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp16s, Npp16s, 1, false, kRobertsDown, nppi::filter::nppi::filter::kRobertsDown>(const Npp16s*, int, Npp16s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp16s, Npp16s, 1, false, kRobertsDown, nppi::filter::nppi::filter::kRobertsDown>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp16s, Npp16s, 3, false, kRobertsDown, nppi::filter::nppi::filter::kRobertsDown>(const Npp16s*, int, Npp16s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp16s, Npp16s, 3, false, kRobertsDown, nppi::filter::nppi::filter::kRobertsDown>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp16s, Npp16s, 4, false, kRobertsDown, nppi::filter::nppi::filter::kRobertsDown>(const Npp16s*, int, Npp16s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp16s, Npp16s, 4, false, kRobertsDown, nppi::filter::nppi::filter::kRobertsDown>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp16s, Npp16s, 4, true, kRobertsDown, nppi::filter::nppi::filter::kRobertsDown>(const Npp16s*, int, Npp16s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp16s, Npp16s, 4, true, kRobertsDown, nppi::filter::nppi::filter::kRobertsDown>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp32f, Npp32f, 1, false, kRobertsDown, nppi::filter::nppi::filter::kRobertsDown>(const Npp32f*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp32f, Npp32f, 1, false, kRobertsDown, nppi::filter::nppi::filter::kRobertsDown>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp32f, Npp32f, 3, false, kRobertsDown, nppi::filter::nppi::filter::kRobertsDown>(const Npp32f*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp32f, Npp32f, 3, false, kRobertsDown, nppi::filter::nppi::filter::kRobertsDown>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp32f, Npp32f, 4, false, kRobertsDown, nppi::filter::nppi::filter::kRobertsDown>(const Npp32f*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp32f, Npp32f, 4, false, kRobertsDown, nppi::filter::nppi::filter::kRobertsDown>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp32f, Npp32f, 4, true, kRobertsDown, nppi::filter::nppi::filter::kRobertsDown>(const Npp32f*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp32f, Npp32f, 4, true, kRobertsDown, nppi::filter::nppi::filter::kRobertsDown>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppBorderType, cudaStream_t);
// kRobertsUp
template NppStatus nppi::filter::filterDeriv<Npp8u, Npp8u, 1, false, kRobertsUp, nppi::filter::nppi::filter::kRobertsUp>(const Npp8u*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp8u, Npp8u, 1, false, kRobertsUp, nppi::filter::nppi::filter::kRobertsUp>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp8u, Npp8u, 3, false, kRobertsUp, nppi::filter::nppi::filter::kRobertsUp>(const Npp8u*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp8u, Npp8u, 3, false, kRobertsUp, nppi::filter::nppi::filter::kRobertsUp>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp8u, Npp8u, 4, false, kRobertsUp, nppi::filter::nppi::filter::kRobertsUp>(const Npp8u*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp8u, Npp8u, 4, false, kRobertsUp, nppi::filter::nppi::filter::kRobertsUp>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp8u, Npp8u, 4, true, kRobertsUp, nppi::filter::nppi::filter::kRobertsUp>(const Npp8u*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp8u, Npp8u, 4, true, kRobertsUp, nppi::filter::nppi::filter::kRobertsUp>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp16s, Npp16s, 1, false, kRobertsUp, nppi::filter::nppi::filter::kRobertsUp>(const Npp16s*, int, Npp16s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp16s, Npp16s, 1, false, kRobertsUp, nppi::filter::nppi::filter::kRobertsUp>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp16s, Npp16s, 3, false, kRobertsUp, nppi::filter::nppi::filter::kRobertsUp>(const Npp16s*, int, Npp16s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp16s, Npp16s, 3, false, kRobertsUp, nppi::filter::nppi::filter::kRobertsUp>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp16s, Npp16s, 4, false, kRobertsUp, nppi::filter::nppi::filter::kRobertsUp>(const Npp16s*, int, Npp16s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp16s, Npp16s, 4, false, kRobertsUp, nppi::filter::nppi::filter::kRobertsUp>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp16s, Npp16s, 4, true, kRobertsUp, nppi::filter::nppi::filter::kRobertsUp>(const Npp16s*, int, Npp16s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp16s, Npp16s, 4, true, kRobertsUp, nppi::filter::nppi::filter::kRobertsUp>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp32f, Npp32f, 1, false, kRobertsUp, nppi::filter::nppi::filter::kRobertsUp>(const Npp32f*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp32f, Npp32f, 1, false, kRobertsUp, nppi::filter::nppi::filter::kRobertsUp>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp32f, Npp32f, 3, false, kRobertsUp, nppi::filter::nppi::filter::kRobertsUp>(const Npp32f*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp32f, Npp32f, 3, false, kRobertsUp, nppi::filter::nppi::filter::kRobertsUp>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp32f, Npp32f, 4, false, kRobertsUp, nppi::filter::nppi::filter::kRobertsUp>(const Npp32f*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp32f, Npp32f, 4, false, kRobertsUp, nppi::filter::nppi::filter::kRobertsUp>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp32f, Npp32f, 4, true, kRobertsUp, nppi::filter::nppi::filter::kRobertsUp>(const Npp32f*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp32f, Npp32f, 4, true, kRobertsUp, nppi::filter::nppi::filter::kRobertsUp>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppBorderType, cudaStream_t);
// kSobelHoriz
template NppStatus nppi::filter::filterDeriv<Npp8u, Npp8u, 1, false, kSobelHoriz, nppi::filter::nppi::filter::kSobelHoriz>(const Npp8u*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp8u, Npp8u, 1, false, kSobelHoriz, nppi::filter::nppi::filter::kSobelHoriz>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp8u, Npp8u, 3, false, kSobelHoriz, nppi::filter::nppi::filter::kSobelHoriz>(const Npp8u*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp8u, Npp8u, 3, false, kSobelHoriz, nppi::filter::nppi::filter::kSobelHoriz>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp8u, Npp8u, 4, false, kSobelHoriz, nppi::filter::nppi::filter::kSobelHoriz>(const Npp8u*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp8u, Npp8u, 4, false, kSobelHoriz, nppi::filter::nppi::filter::kSobelHoriz>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp8u, Npp8u, 4, true, kSobelHoriz, nppi::filter::nppi::filter::kSobelHoriz>(const Npp8u*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp8u, Npp8u, 4, true, kSobelHoriz, nppi::filter::nppi::filter::kSobelHoriz>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp16s, Npp16s, 1, false, kSobelHoriz, nppi::filter::nppi::filter::kSobelHoriz>(const Npp16s*, int, Npp16s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp16s, Npp16s, 1, false, kSobelHoriz, nppi::filter::nppi::filter::kSobelHoriz>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp16s, Npp16s, 3, false, kSobelHoriz, nppi::filter::nppi::filter::kSobelHoriz>(const Npp16s*, int, Npp16s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp16s, Npp16s, 3, false, kSobelHoriz, nppi::filter::nppi::filter::kSobelHoriz>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp16s, Npp16s, 4, false, kSobelHoriz, nppi::filter::nppi::filter::kSobelHoriz>(const Npp16s*, int, Npp16s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp16s, Npp16s, 4, false, kSobelHoriz, nppi::filter::nppi::filter::kSobelHoriz>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp16s, Npp16s, 4, true, kSobelHoriz, nppi::filter::nppi::filter::kSobelHoriz>(const Npp16s*, int, Npp16s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp16s, Npp16s, 4, true, kSobelHoriz, nppi::filter::nppi::filter::kSobelHoriz>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp32f, Npp32f, 1, false, kSobelHoriz, nppi::filter::nppi::filter::kSobelHoriz>(const Npp32f*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp32f, Npp32f, 1, false, kSobelHoriz, nppi::filter::nppi::filter::kSobelHoriz>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp32f, Npp32f, 3, false, kSobelHoriz, nppi::filter::nppi::filter::kSobelHoriz>(const Npp32f*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp32f, Npp32f, 3, false, kSobelHoriz, nppi::filter::nppi::filter::kSobelHoriz>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp32f, Npp32f, 4, false, kSobelHoriz, nppi::filter::nppi::filter::kSobelHoriz>(const Npp32f*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp32f, Npp32f, 4, false, kSobelHoriz, nppi::filter::nppi::filter::kSobelHoriz>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp32f, Npp32f, 4, true, kSobelHoriz, nppi::filter::nppi::filter::kSobelHoriz>(const Npp32f*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp32f, Npp32f, 4, true, kSobelHoriz, nppi::filter::nppi::filter::kSobelHoriz>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppBorderType, cudaStream_t);
// kSobelVert
template NppStatus nppi::filter::filterDeriv<Npp8u, Npp8u, 1, false, kSobelVert, nppi::filter::nppi::filter::kSobelVert>(const Npp8u*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp8u, Npp8u, 1, false, kSobelVert, nppi::filter::nppi::filter::kSobelVert>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp8u, Npp8u, 3, false, kSobelVert, nppi::filter::nppi::filter::kSobelVert>(const Npp8u*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp8u, Npp8u, 3, false, kSobelVert, nppi::filter::nppi::filter::kSobelVert>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp8u, Npp8u, 4, false, kSobelVert, nppi::filter::nppi::filter::kSobelVert>(const Npp8u*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp8u, Npp8u, 4, false, kSobelVert, nppi::filter::nppi::filter::kSobelVert>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp8u, Npp8u, 4, true, kSobelVert, nppi::filter::nppi::filter::kSobelVert>(const Npp8u*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp8u, Npp8u, 4, true, kSobelVert, nppi::filter::nppi::filter::kSobelVert>(const Npp8u*, int, NppiSize, NppiPoint, Npp8u*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp16s, Npp16s, 1, false, kSobelVert, nppi::filter::nppi::filter::kSobelVert>(const Npp16s*, int, Npp16s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp16s, Npp16s, 1, false, kSobelVert, nppi::filter::nppi::filter::kSobelVert>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp16s, Npp16s, 3, false, kSobelVert, nppi::filter::nppi::filter::kSobelVert>(const Npp16s*, int, Npp16s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp16s, Npp16s, 3, false, kSobelVert, nppi::filter::nppi::filter::kSobelVert>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp16s, Npp16s, 4, false, kSobelVert, nppi::filter::nppi::filter::kSobelVert>(const Npp16s*, int, Npp16s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp16s, Npp16s, 4, false, kSobelVert, nppi::filter::nppi::filter::kSobelVert>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp16s, Npp16s, 4, true, kSobelVert, nppi::filter::nppi::filter::kSobelVert>(const Npp16s*, int, Npp16s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp16s, Npp16s, 4, true, kSobelVert, nppi::filter::nppi::filter::kSobelVert>(const Npp16s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp32f, Npp32f, 1, false, kSobelVert, nppi::filter::nppi::filter::kSobelVert>(const Npp32f*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp32f, Npp32f, 1, false, kSobelVert, nppi::filter::nppi::filter::kSobelVert>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp32f, Npp32f, 3, false, kSobelVert, nppi::filter::nppi::filter::kSobelVert>(const Npp32f*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp32f, Npp32f, 3, false, kSobelVert, nppi::filter::nppi::filter::kSobelVert>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp32f, Npp32f, 4, false, kSobelVert, nppi::filter::nppi::filter::kSobelVert>(const Npp32f*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp32f, Npp32f, 4, false, kSobelVert, nppi::filter::nppi::filter::kSobelVert>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDeriv<Npp32f, Npp32f, 4, true, kSobelVert, nppi::filter::nppi::filter::kSobelVert>(const Npp32f*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivBorder<Npp32f, Npp32f, 4, true, kSobelVert, nppi::filter::nppi::filter::kSobelVert>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppBorderType, cudaStream_t);
// kSobelHoriz (masked / conversion variants)
template NppStatus nppi::filter::filterDerivMask<Npp8u, Npp16s, nppi::filter::kSobelHoriz>(const Npp8u*, int, Npp16s*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivMaskBorder<Npp8u, Npp16s, nppi::filter::kSobelHoriz>(const Npp8u*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppiMaskSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDerivMask<Npp8s, Npp16s, nppi::filter::kSobelHoriz>(const Npp8s*, int, Npp16s*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivMaskBorder<Npp8s, Npp16s, nppi::filter::kSobelHoriz>(const Npp8s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppiMaskSize, NppBorderType, cudaStream_t);
// kSobelVert (masked / conversion variants)
template NppStatus nppi::filter::filterDerivMask<Npp8u, Npp16s, nppi::filter::kSobelVert>(const Npp8u*, int, Npp16s*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivMaskBorder<Npp8u, Npp16s, nppi::filter::kSobelVert>(const Npp8u*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppiMaskSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDerivMask<Npp8s, Npp16s, nppi::filter::kSobelVert>(const Npp8s*, int, Npp16s*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivMaskBorder<Npp8s, Npp16s, nppi::filter::kSobelVert>(const Npp8s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppiMaskSize, NppBorderType, cudaStream_t);
// kSobelCross (masked / conversion variants)
template NppStatus nppi::filter::filterDerivMask<Npp8u, Npp16s, nppi::filter::kSobelCross>(const Npp8u*, int, Npp16s*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivMaskBorder<Npp8u, Npp16s, nppi::filter::kSobelCross>(const Npp8u*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppiMaskSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDerivMask<Npp8s, Npp16s, nppi::filter::kSobelCross>(const Npp8s*, int, Npp16s*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivMaskBorder<Npp8s, Npp16s, nppi::filter::kSobelCross>(const Npp8s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppiMaskSize, NppBorderType, cudaStream_t);
// kSobelHorizSecond (masked / conversion variants)
template NppStatus nppi::filter::filterDerivMask<Npp8u, Npp16s, nppi::filter::kSobelHorizSecond>(const Npp8u*, int, Npp16s*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivMaskBorder<Npp8u, Npp16s, nppi::filter::kSobelHorizSecond>(const Npp8u*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppiMaskSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDerivMask<Npp8s, Npp16s, nppi::filter::kSobelHorizSecond>(const Npp8s*, int, Npp16s*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivMaskBorder<Npp8s, Npp16s, nppi::filter::kSobelHorizSecond>(const Npp8s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppiMaskSize, NppBorderType, cudaStream_t);
// kSobelVertSecond (masked / conversion variants)
template NppStatus nppi::filter::filterDerivMask<Npp8u, Npp16s, nppi::filter::kSobelVertSecond>(const Npp8u*, int, Npp16s*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivMaskBorder<Npp8u, Npp16s, nppi::filter::kSobelVertSecond>(const Npp8u*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppiMaskSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDerivMask<Npp8s, Npp16s, nppi::filter::kSobelVertSecond>(const Npp8s*, int, Npp16s*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivMaskBorder<Npp8s, Npp16s, nppi::filter::kSobelVertSecond>(const Npp8s*, int, NppiSize, NppiPoint, Npp16s*, int, NppiSize, NppiMaskSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDerivMask<Npp32f, Npp32f, nppi::filter::kSobelCross>(const Npp32f*, int, Npp32f*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivMaskBorder<Npp32f, Npp32f, nppi::filter::kSobelCross>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppiMaskSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDerivMask<Npp32f, Npp32f, nppi::filter::kSobelHorizSecond>(const Npp32f*, int, Npp32f*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivMaskBorder<Npp32f, Npp32f, nppi::filter::kSobelHorizSecond>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppiMaskSize, NppBorderType, cudaStream_t);
template NppStatus nppi::filter::filterDerivMask<Npp32f, Npp32f, nppi::filter::kSobelVertSecond>(const Npp32f*, int, Npp32f*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivMaskBorder<Npp32f, Npp32f, nppi::filter::kSobelVertSecond>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppiMaskSize, NppBorderType, cudaStream_t);
// kSobelHoriz Mask families (32f C1)
template NppStatus nppi::filter::filterDerivMask<Npp32f, Npp32f, nppi::filter::kSobelHoriz>(const Npp32f*, int, Npp32f*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivMaskBorder<Npp32f, Npp32f, nppi::filter::kSobelHoriz>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppiMaskSize, NppBorderType, cudaStream_t);
// kSobelVert Mask families (32f C1)
template NppStatus nppi::filter::filterDerivMask<Npp32f, Npp32f, nppi::filter::kSobelVert>(const Npp32f*, int, Npp32f*, int, NppiSize, NppiMaskSize, cudaStream_t);
template NppStatus nppi::filter::filterDerivMaskBorder<Npp32f, Npp32f, nppi::filter::kSobelVert>(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, NppiSize, NppiMaskSize, NppBorderType, cudaStream_t);
