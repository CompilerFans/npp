// GradientVector kernels and executor entry points.
// See nppi_gradient_vector_executor.h for the family contract.
#include "npp.h"
#include "nppi_gradient_vector_executor.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <type_traits>

namespace nppi {
namespace gv {
namespace {

constexpr int kBlockX = 16;
constexpr int kBlockY = 16;

struct GvCoef {
  int size;      // 3 or 5
  int k[5][5];   // k[a][b], a = row offset (top row first)
};

inline GvCoef gv_coef3(const int *k9) {
  GvCoef c{};
  c.size = 3;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) c.k[a][b] = k9[a * 3 + b];
  return c;
}

inline GvCoef gv_coef5(const int *k25) {
  GvCoef c{};
  c.size = 5;
  for (int a = 0; a < 5; ++a)
    for (int b = 0; b < 5; ++b) c.k[a][b] = k25[a * 5 + b];
  return c;
}

// X/Y tables.  Measured convention: X is the NEGATED vertical response for
// Sobel/Prewitt but the POSITIVE vertical response for Scharr; Y is the
// positive horizontal response for all three families (same tap tables as the
// derivative-filter executor).
struct GvTaps {
  GvCoef x, y;
};

inline GvTaps gv_taps(GvOp op, int maskSize) {
  static const int SobelHoriz3[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
  static const int SobelVert3[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  static const int SobelHoriz5[25] = {-1, -4, -6, -4, -1, -2, -8, -12, -8, -2, 0, 0, 0, 0, 0,
                                      2, 8,  12, 8,  2,  1,  4,  6,   4,  1};
  static const int SobelVert5[25] = {-1, -2, 0,  2,  1,  -4, -8, 0,  8,  4,  -6, -12, 0,  12, 6,
                                     -4, -8, 0,  8,  4,  -1, -2, 0,  2,  1};
  static const int PrewittHoriz3[9] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
  static const int PrewittVert3[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
  static const int ScharrHoriz3[9] = {-3, -10, -3, 0, 0, 0, 3, 10, 3};
  static const int ScharrVert3[9] = {-3, 0, 3, -10, 0, 10, -3, 0, 3};
  GvCoef h, v;
  if (op == kSobel && maskSize == 5) {
    h = gv_coef5(SobelHoriz5);
    v = gv_coef5(SobelVert5);
  } else {
    const int *h3 = op == kSobel ? SobelHoriz3 : (op == kPrewitt ? PrewittHoriz3 : ScharrHoriz3);
    const int *v3 = op == kSobel ? SobelVert3 : (op == kPrewitt ? PrewittVert3 : ScharrVert3);
    h = gv_coef3(h3);
    v = gv_coef3(v3);
  }
  // X = +v for Scharr, -v for Sobel/Prewitt.
  GvCoef x = v;
  if (op != kScharr)
    for (int a = 0; a < 5; ++a)
      for (int b = 0; b < 5; ++b) x.k[a][b] = -x.k[a][b];
  return GvTaps{x, h};
}

// Emission: X/Y are exact integers for integer sources (|response| <= 16320
// for 8u Sobel 5x5, well inside 16s), so a truncating cast is exact; the
// magnitude of an 8u Sobel 5x5 L1 response reaches 32640 and also stays in
// range.  32f destinations keep the float value.
template <typename Dst>
__host__ __device__ __forceinline__ Dst gv_emit(float v) {
  if constexpr (std::is_same_v<Dst, Npp16s>) {
    return v <= -32768.0f ? (Npp16s)-32768 : (v >= 32767.0f ? (Npp16s)32767 : (Npp16s)v);
  } else {
    return static_cast<Dst>(v);
  }
}

inline NppStatus finish_launch(cudaStream_t stream) {
  cudaStreamSynchronize(stream);
  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

// 2D convolution with NPP_BORDER_REPLICATE clamping to the source bounds
// expressed in pSrc-relative coordinates, [-offY, srcH-1-offY] x
// [-offX, srcW-1-offX].
template <typename Src, typename Dst, int Channels, GvOp Op>
__global__ void gv_kernel(const Src *__restrict__ pSrc, int nSrcStep, Dst *__restrict__ pDstX, int nDstXStep,
                          Dst *__restrict__ pDstY, int nDstYStep, Dst *__restrict__ pDstMag, int nDstMagStep,
                          Npp32f *__restrict__ pDstAngle, int nDstAngleStep, int width, int height, GvTaps taps,
                          int srcW, int srcH, int offX, int offY, int normL1) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const int r = taps.x.size >> 1;
  const int loX = -offX, hiX = srcW - 1 - offX;
  const int loY = -offY, hiY = srcH - 1 - offY;
  using Acc = std::conditional_t<std::is_floating_point_v<Src>, float, int>;
  const char *const srcBase = reinterpret_cast<const char *>(pSrc);

  Acc ax[3], ay[3];
  for (int c = 0; c < Channels; ++c) ax[c] = Acc(0);
  for (int c = 0; c < Channels; ++c) ay[c] = Acc(0);
  for (int a = 0; a < taps.x.size; ++a) {
    int ry = y + a - r;
    ry = ry < loY ? loY : (ry > hiY ? hiY : ry);
    const Src *row = reinterpret_cast<const Src *>(srcBase + static_cast<std::ptrdiff_t>(ry) * nSrcStep);
    for (int b = 0; b < taps.x.size; ++b) {
      int rx = x + b - r;
      rx = rx < loX ? loX : (rx > hiX ? hiX : rx);
      const Src *pix = row + static_cast<std::ptrdiff_t>(rx) * Channels;
      for (int c = 0; c < Channels; ++c) {
        const Acc v = static_cast<Acc>(pix[c]);
        ax[c] += static_cast<Acc>(taps.x.k[a][b]) * v;
        ay[c] += static_cast<Acc>(taps.y.k[a][b]) * v;
      }
    }
  }

  // C3C1R: all four outputs come from the channel with the largest
  // X^2+Y^2 (strict >, channel 0 wins ties).
  int sel = 0;
  if (Channels == 3) {
    double best = -1.0;
    for (int c = 0; c < 3; ++c) {
      const double gx = static_cast<double>(ax[c]);
      const double gy = static_cast<double>(ay[c]);
      const double l2 = gx * gx + gy * gy;
      if (l2 > best) {
        best = l2;
        sel = c;
      }
    }
  }

  const float gx = static_cast<float>(ax[sel]);
  const float gy = static_cast<float>(ay[sel]);
  if (pDstX) {
    Dst *drow = reinterpret_cast<Dst *>(reinterpret_cast<char *>(pDstX) +
                                        static_cast<std::ptrdiff_t>(y) * nDstXStep) +
                static_cast<std::ptrdiff_t>(x);
    *drow = gv_emit<Dst>(gx);
  }
  if (pDstY) {
    Dst *drow = reinterpret_cast<Dst *>(reinterpret_cast<char *>(pDstY) +
                                        static_cast<std::ptrdiff_t>(y) * nDstYStep) +
                static_cast<std::ptrdiff_t>(x);
    *drow = gv_emit<Dst>(gy);
  }
  if (pDstMag) {
    const double mag = normL1 ? (std::fabs((double)gx) + std::fabs((double)gy))
                              : std::sqrt((double)gx * gx + (double)gy * gy);
    Dst *drow = reinterpret_cast<Dst *>(reinterpret_cast<char *>(pDstMag) +
                                        static_cast<std::ptrdiff_t>(y) * nDstMagStep) +
                static_cast<std::ptrdiff_t>(x);
    *drow = gv_emit<Dst>(static_cast<float>(mag));
  }
  if (pDstAngle) {
    Npp32f *drow = reinterpret_cast<Npp32f *>(reinterpret_cast<char *>(pDstAngle) +
                                              static_cast<std::ptrdiff_t>(y) * nDstAngleStep) +
                   static_cast<std::ptrdiff_t>(x);
    *drow = static_cast<Npp32f>(std::atan2((double)gy, (double)gx));
  }
}

}  // namespace

template <typename Src, typename Dst, int SrcChannels, GvOp Op>
NppStatus gradientVectorBorder(const Src *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Dst *pDstX,
                               int nDstXStep, Dst *pDstY, int nDstYStep, Dst *pDstMag, int nDstMagStep,
                               Npp32f *pDstAngle, int nDstAngleStep, NppiSize oSizeROI, NppiMaskSize eMaskSize,
                               NppiNorm eNorm, NppiBorderType eBorderType, cudaStream_t stream) {
  // 1) Required pointers.  The four outputs are individually optional; only
  //    all-null is an error.
  if (!pSrc) return NPP_NULL_POINTER_ERROR;
  if (!pDstX && !pDstY && !pDstMag && !pDstAngle) return NPP_NULL_POINTER_ERROR;
  // 2) Negative ROI.
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  // 3) Zero-size ROI: no launch, no further validation.
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  // 4) Step of the FIRST non-null output: must cover a ROI row, then be a
  //    multiple of the element size.  Other outputs' steps are not checked.
  const int esz = static_cast<int>(sizeof(Dst));
  const int firstEsz = pDstX ? esz : (pDstY ? esz : (pDstMag ? esz : 4));
  const int firstStep = pDstX ? nDstXStep : (pDstY ? nDstYStep : (pDstMag ? nDstMagStep : nDstAngleStep));
  if (firstStep < oSizeROI.width * firstEsz) return NPP_STEP_ERROR;
  if (firstStep % firstEsz != 0) return NPP_NOT_EVEN_STEP_ERROR;
  // 5) Source size, offset must be inside the source.
  if (oSrcSize.width <= 0 || oSrcSize.height <= 0) return NPP_SIZE_ERROR;
  if (oSrcOffset.x < 0 || oSrcOffset.y < 0 || oSrcOffset.x >= oSrcSize.width || oSrcOffset.y >= oSrcSize.height)
    return NPP_OUT_OFF_RANGE_ERROR;
  // 6) Border type.
  if (eBorderType != NPP_BORDER_REPLICATE) return NPP_NOT_SUPPORTED_MODE_ERROR;
  // 7) Norm (validated even when pDstMag is null).
  const int normL1 = eNorm == nppiNormL1;
  if (!normL1 && eNorm != nppiNormL2) return NPP_BAD_ARGUMENT_ERROR;
  // 8) Mask size: 3x3 always, 5x5 for Sobel only.
  int maskSize = 0;
  if (eMaskSize == NPP_MASK_SIZE_3_X_3)
    maskSize = 3;
  else if (eMaskSize == NPP_MASK_SIZE_5_X_5 && Op == kSobel)
    maskSize = 5;
  else
    return NPP_MASK_SIZE_ERROR;

  (void)cudaGetLastError();
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + kBlockX - 1) / kBlockX, (oSizeROI.height + kBlockY - 1) / kBlockY);
  gv_kernel<Src, Dst, SrcChannels, Op>
      <<<grid, block, 0, stream>>>(pSrc, nSrcStep, pDstX, nDstXStep, pDstY, nDstYStep, pDstMag, nDstMagStep,
                                   pDstAngle, nDstAngleStep, oSizeROI.width, oSizeROI.height, gv_taps(Op, maskSize),
                                   oSrcSize.width, oSrcSize.height, oSrcOffset.x, oSrcOffset.y, normL1);
  return finish_launch(stream);
}

}  // namespace gv
}  // namespace nppi

// ---------------------------------------------------------------------------
// Explicit instantiations (f13 GradientVector families: 3 ops x 4 source/dest
// type pairs x {C1R, C3C1R}).
// ---------------------------------------------------------------------------
#define GV_INST(Src, Dst, Ch, Op) \
  template NppStatus nppi::gv::gradientVectorBorder<Src, Dst, Ch, nppi::gv::Op>( \
      const Src*, int, NppiSize, NppiPoint, Dst*, int, Dst*, int, Dst*, int, Npp32f*, int, NppiSize, NppiMaskSize, \
      NppiNorm, NppiBorderType, cudaStream_t);

// Sobel
GV_INST(Npp8u, Npp16s, 1, kSobel)
GV_INST(Npp8u, Npp16s, 3, kSobel)
GV_INST(Npp16s, Npp32f, 1, kSobel)
GV_INST(Npp16s, Npp32f, 3, kSobel)
GV_INST(Npp16u, Npp32f, 1, kSobel)
GV_INST(Npp16u, Npp32f, 3, kSobel)
GV_INST(Npp32f, Npp32f, 1, kSobel)
GV_INST(Npp32f, Npp32f, 3, kSobel)
// Prewitt
GV_INST(Npp8u, Npp16s, 1, kPrewitt)
GV_INST(Npp8u, Npp16s, 3, kPrewitt)
GV_INST(Npp16s, Npp32f, 1, kPrewitt)
GV_INST(Npp16s, Npp32f, 3, kPrewitt)
GV_INST(Npp16u, Npp32f, 1, kPrewitt)
GV_INST(Npp16u, Npp32f, 3, kPrewitt)
GV_INST(Npp32f, Npp32f, 1, kPrewitt)
GV_INST(Npp32f, Npp32f, 3, kPrewitt)
// Scharr
GV_INST(Npp8u, Npp16s, 1, kScharr)
GV_INST(Npp8u, Npp16s, 3, kScharr)
GV_INST(Npp16s, Npp32f, 1, kScharr)
GV_INST(Npp16s, Npp32f, 3, kScharr)
GV_INST(Npp16u, Npp32f, 1, kScharr)
GV_INST(Npp16u, Npp32f, 3, kScharr)
GV_INST(Npp32f, Npp32f, 1, kScharr)
GV_INST(Npp32f, Npp32f, 3, kScharr)
