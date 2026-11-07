// Gauss filter family kernels and executor entry points.
// See nppi_filter_gauss_executor.h for the measured family contract.
#include "npp.h"
#include "nppi_filter_gauss_executor.h"

#include <cmath>
#include <cuda_runtime.h>
#include <type_traits>

namespace nppi {
namespace filter {
namespace {

constexpr int kBlockX = 16;
constexpr int kBlockY = 16;
// Largest 1D kernel kept in dynamic shared memory; bigger user kernels are
// read straight from global memory (Advanced only).
constexpr int kMaxSharedTaps = 1024;

template <typename Dst>
__host__ __device__ __forceinline__ Dst gauss_emit(float acc) {
  if constexpr (std::is_floating_point_v<Dst>) {
    return static_cast<Dst>(acc);
  } else {
    // NPP truncates toward zero, then saturates to the full destination range
    // (including the negative side; a user kernel in GaussAdvanced can drive
    // the accumulator below zero).
    constexpr float kLo = std::is_same_v<Dst, Npp16s> ? -32768.0f: 0.0f;
    constexpr float kHi = std::is_same_v<Dst, Npp8u> ? 255.0f: std::is_same_v<Dst, Npp16u> ? 65535.0f: 32767.0f;
    const float lo = acc < kLo ? kLo: acc;
    const float v = lo > kHi ? kHi: lo;
    const int truncated = static_cast<int>(v);
    return static_cast<Dst>(truncated);
  }
}

// 2D Gauss convolution.
//   * Advanced=false: sK holds the normalised 1D Gaussian kernel (generated per
//     block); the 2D coefficient is sK[j]*sK[i].
//   * Advanced=true: sK holds the user 1D kernel (taps values) in shared memory
//     when it fits, otherwise the global pointer is read directly.
//   * Border=false: raw reads around pSrc (caller provides a margin).
//   * Border=true: read coordinates clamp to the source image bounds relative to
//     pSrc, i.e. x in [-offX, srcW-1-offX] and y in [-offY, srcH-1-offY].
template <typename Src, typename Dst, int Channels, bool AlphaPr, bool Border, bool Advanced>
__global__ void gauss_kernel(const Src *__restrict__ pSrc, int nSrcStep, Dst *__restrict__ pDst, int nDstStep,
                             int width, int height, const Npp32f *__restrict__ pUserKernel, int taps, int srcW,
                             int srcH, int offX, int offY) {
  extern __shared__ float sK[];
  const int r = taps >> 1;
  const bool useShared = !Advanced || taps <= kMaxSharedTaps;
  const int tid = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
  const int nthreads = static_cast<int>(blockDim.x * blockDim.y);
  if (useShared) {
    if constexpr (Advanced) {
      for (int i = tid; i < taps; i += nthreads) sK[i] = pUserKernel[i];
    } else {
      const float sigma = 0.6f + 0.4f * static_cast<float>(r);
      const float denom = 2.0f * sigma * sigma;
      for (int i = tid; i < taps; i += nthreads) {
        const float d = static_cast<float>(i - r);
        sK[i] = expf(-(d * d) / denom);
      }
      __syncthreads();
      // One thread per block, not one per block row: every thread with
      // threadIdx.x == 0 would otherwise re-normalise sK concurrently.
      if (tid == 0) {
        float s = 0.0f;
        for (int i = 0; i < taps; ++i) s += sK[i];
        for (int i = 0; i < taps; ++i) sK[i] = sK[i] / s;
      }
    }
    __syncthreads();
  }

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  const int nch = AlphaPr ? 3: Channels;
  const int loX = -offX, hiX = srcW - 1 - offX;
  const int loY = -offY, hiY = srcH - 1 - offY;
  const char *const srcBase = reinterpret_cast<const char *>(pSrc);
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int j = 0; j < taps; ++j) {
    int ry = y + j - r;
    if constexpr (Border) ry = ry < loY ? loY: (ry > hiY ? hiY: ry);
    const Src *row = reinterpret_cast<const Src *>(srcBase + static_cast<std::ptrdiff_t>(ry) * nSrcStep);
    const float kj = useShared ? sK[j]: pUserKernel[j];
    for (int i = 0; i < taps; ++i) {
      int rx = x + i - r;
      if constexpr (Border) rx = rx < loX ? loX: (rx > hiX ? hiX: rx);
      const float ki = useShared ? sK[i]: pUserKernel[i];
      const float kv = kj * ki;
      const Src *pix = row + static_cast<std::ptrdiff_t>(rx) * Channels;
      for (int c = 0; c < nch; ++c) acc[c] = fmaf(kv, static_cast<float>(pix[c]), acc[c]);
    }
  }
  Dst *drow = reinterpret_cast<Dst *>(reinterpret_cast<char *>(pDst) + static_cast<std::ptrdiff_t>(y) * nDstStep) +
              static_cast<std::ptrdiff_t>(x) * Channels;
  for (int c = 0; c < nch; ++c) drow[c] = gauss_emit<Dst>(acc[c]);
}

inline NppStatus finish_launch(cudaStream_t stream) {
  // NPP's Ctx entry points are asynchronous and stream-capture safe: no
  // cudaStreamSynchronize here (the value is only used to keep the signature
  // uniform with the other executors).
  (void)stream;
  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

inline bool gauss_fixed_taps(NppiMaskSize eMaskSize, int *taps) {
  switch (static_cast<int>(eMaskSize)) {
    case NPP_MASK_SIZE_3_X_3: *taps = 3; return true;
    case NPP_MASK_SIZE_5_X_5: *taps = 5; return true;
    case NPP_MASK_SIZE_7_X_7: *taps = 7; return true;
    case NPP_MASK_SIZE_9_X_9: *taps = 9; return true;
    case NPP_MASK_SIZE_11_X_11: *taps = 11; return true;
    case NPP_MASK_SIZE_13_X_13: *taps = 13; return true;
    case NPP_MASK_SIZE_15_X_15: *taps = 15; return true;
    default: return false;
  }
}

template <typename Dst, int Channels>
NppStatus check_dst_step(int nDstStep, int roiWidth) {
  const int pix = static_cast<int>(sizeof(Dst)) * Channels;
  const int align = Channels == 4 ? pix: static_cast<int>(sizeof(Dst));
  const unsigned uStep = static_cast<unsigned>(nDstStep);
  if (uStep < static_cast<unsigned>(roiWidth * pix)) return NPP_STEP_ERROR;
  if (align > 1 && uStep % static_cast<unsigned>(align) != 0) return NPP_NOT_EVEN_STEP_ERROR;
  return NPP_NO_ERROR;
}

template <typename Src, typename Dst, int Channels, bool AlphaPr, bool Border, bool Advanced>
NppStatus launch(const Src *pSrc, int nSrcStep, Dst *pDst, int nDstStep, NppiSize oSizeROI,
                 const Npp32f *pUserKernel, int taps, NppiSize oSrcSize, NppiPoint oSrcOffset,
                 cudaStream_t stream) {
  NppStatus st = check_dst_step<Dst, Channels>(nDstStep, oSizeROI.width);
  if (st != NPP_NO_ERROR) return st;
  (void)cudaGetLastError();
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + kBlockX - 1) / kBlockX, (oSizeROI.height + kBlockY - 1) / kBlockY);
  const int sharedBytes = taps <= kMaxSharedTaps ? taps * static_cast<int>(sizeof(float)): 0;
  gauss_kernel<Src, Dst, Channels, AlphaPr, Border, Advanced><<<grid, block, sharedBytes, stream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pUserKernel, taps, oSrcSize.width,
      oSrcSize.height, oSrcOffset.x, oSrcOffset.y);
  return finish_launch(stream);
}

}  // namespace

template <typename Src, typename Dst, int Channels, bool AlphaPreserve>
NppStatus filterGauss(const Src *pSrc, int nSrcStep, Dst *pDst, int nDstStep, NppiSize oSizeROI,
                      NppiMaskSize eMaskSize, cudaStream_t stream) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  int taps = 0;
  if (!gauss_fixed_taps(eMaskSize, &taps)) return NPP_MASK_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  return launch<Src, Dst, Channels, AlphaPreserve, false, false>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                                 nullptr, taps, NppiSize{0, 0}, NppiPoint{0, 0},
                                                                 stream);
}

template <typename Src, typename Dst, int Channels, bool AlphaPreserve>
NppStatus filterGaussBorder(const Src *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Dst *pDst,
                            int nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiBorderType eBorderType,
                            cudaStream_t stream) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSrcSize.width <= 0 || oSrcSize.height <= 0) return NPP_SIZE_ERROR;
  if (oSrcOffset.x < 0 || oSrcOffset.y < 0 || oSrcOffset.x >= oSrcSize.width || oSrcOffset.y >= oSrcSize.height)
    return NPP_OUT_OFF_RANGE_ERROR;
  if (eBorderType != NPP_BORDER_REPLICATE) return NPP_NOT_SUPPORTED_MODE_ERROR;
  int taps = 0;
  if (!gauss_fixed_taps(eMaskSize, &taps)) return NPP_MASK_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  return launch<Src, Dst, Channels, AlphaPreserve, true, false>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nullptr,
                                                                taps, oSrcSize, oSrcOffset, stream);
}

template <typename Src, typename Dst, int Channels, bool AlphaPreserve>
NppStatus filterGaussAdvanced(const Src *pSrc, int nSrcStep, Dst *pDst, int nDstStep, NppiSize oSizeROI,
                              int nFilterTaps, const Npp32f *pKernel, cudaStream_t stream) {
  if (!pSrc || !pDst || !pKernel) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    // reference quirk: the empty-ROI path validates the tap count differently from
    // the regular path (<=0 -> SIZE_ERROR, 1/2 -> MASK_SIZE_ERROR, >=3 passes
    // regardless of parity).
    if (nFilterTaps <= 0) return NPP_SIZE_ERROR;
    if (nFilterTaps < 3) return NPP_MASK_SIZE_ERROR;
    return NPP_NO_ERROR;
  }
  if (nFilterTaps < 0) return NPP_OUT_OFF_RANGE_ERROR;
  // reference order: tap counts below 3 are rejected before the dst-step check, but
  // an even tap count >= 3 is rejected only after it (taps4 + small step ->
  // STEP_ERROR; taps0 + small step -> MASK_SIZE_ERROR).
  if (nFilterTaps < 3) return NPP_MASK_SIZE_ERROR;
  {
    const NppStatus stepSt = check_dst_step<Dst, Channels>(nDstStep, oSizeROI.width);
    if (stepSt != NPP_NO_ERROR) return stepSt;
  }
  if ((nFilterTaps & 1) == 0) return NPP_MASK_SIZE_ERROR;
  return launch<Src, Dst, Channels, AlphaPreserve, false, true>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel,
                                                                nFilterTaps, NppiSize{0, 0}, NppiPoint{0, 0},
                                                                stream);
}

template <typename Src, typename Dst, int Channels, bool AlphaPreserve>
NppStatus filterGaussAdvancedBorder(const Src *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
                                    Dst *pDst, int nDstStep, NppiSize oSizeROI, int nFilterTaps,
                                    const Npp32f *pKernel, NppiBorderType eBorderType, cudaStream_t stream) {
  if (!pSrc || !pDst || !pKernel) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSrcSize.width <= 0 || oSrcSize.height <= 0) return NPP_SIZE_ERROR;
  if (oSrcOffset.x < 0 || oSrcOffset.y < 0 || oSrcOffset.x >= oSrcSize.width || oSrcOffset.y >= oSrcSize.height)
    return NPP_OUT_OFF_RANGE_ERROR;
  if (eBorderType != NPP_BORDER_REPLICATE) return NPP_NOT_SUPPORTED_MODE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    // reference quirk: the empty-ROI path rejects only taps < 3 (MASK_SIZE_ERROR)
    // and accepts any odd or even count >= 3.
    if (nFilterTaps < 3) return NPP_MASK_SIZE_ERROR;
    return NPP_NO_ERROR;
  }
  // Border family: negative tap counts are MASK_SIZE_ERROR (unlike the plain
  // family's OUT_OFF_RANGE), checked before the dst step; even counts >= 3 are
  // checked after it.
  if (nFilterTaps < 3) return NPP_MASK_SIZE_ERROR;
  {
    const NppStatus stepSt = check_dst_step<Dst, Channels>(nDstStep, oSizeROI.width);
    if (stepSt != NPP_NO_ERROR) return stepSt;
  }
  if ((nFilterTaps & 1) == 0) return NPP_MASK_SIZE_ERROR;
  return launch<Src, Dst, Channels, AlphaPreserve, true, true>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel,
                                                               nFilterTaps, oSrcSize, oSrcOffset, stream);
}

}  // namespace filter
}  // namespace nppi

// ---------------------------------------------------------------------------
// Explicit instantiations.
// ---------------------------------------------------------------------------
#define NPPI_GAUSS_INSTANCE(T)                                                                     \
  template NppStatus nppi::filter::filterGauss<T, T, 1, false>(const T *, int, T *, int, NppiSize,  \
                                                               NppiMaskSize, cudaStream_t);        \
  template NppStatus nppi::filter::filterGauss<T, T, 3, false>(const T *, int, T *, int, NppiSize,  \
                                                               NppiMaskSize, cudaStream_t);        \
  template NppStatus nppi::filter::filterGauss<T, T, 4, false>(const T *, int, T *, int, NppiSize,  \
                                                               NppiMaskSize, cudaStream_t);        \
  template NppStatus nppi::filter::filterGauss<T, T, 4, true>(const T *, int, T *, int, NppiSize,   \
                                                              NppiMaskSize, cudaStream_t);         \
  template NppStatus nppi::filter::filterGaussBorder<T, T, 1, false>(                              \
      const T *, int, NppiSize, NppiPoint, T *, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t); \
  template NppStatus nppi::filter::filterGaussBorder<T, T, 3, false>(                              \
      const T *, int, NppiSize, NppiPoint, T *, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t); \
  template NppStatus nppi::filter::filterGaussBorder<T, T, 4, false>(                              \
      const T *, int, NppiSize, NppiPoint, T *, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t); \
  template NppStatus nppi::filter::filterGaussBorder<T, T, 4, true>(                               \
      const T *, int, NppiSize, NppiPoint, T *, int, NppiSize, NppiMaskSize, NppiBorderType, cudaStream_t); \
  template NppStatus nppi::filter::filterGaussAdvanced<T, T, 1, false>(const T *, int, T *, int, NppiSize, int,  \
                                                                      const Npp32f *, cudaStream_t);            \
  template NppStatus nppi::filter::filterGaussAdvanced<T, T, 3, false>(const T *, int, T *, int, NppiSize, int,  \
                                                                      const Npp32f *, cudaStream_t);            \
  template NppStatus nppi::filter::filterGaussAdvanced<T, T, 4, false>(const T *, int, T *, int, NppiSize, int,  \
                                                                      const Npp32f *, cudaStream_t);            \
  template NppStatus nppi::filter::filterGaussAdvanced<T, T, 4, true>(const T *, int, T *, int, NppiSize, int,   \
                                                                     const Npp32f *, cudaStream_t);             \
  template NppStatus nppi::filter::filterGaussAdvancedBorder<T, T, 1, false>(                     \
      const T *, int, NppiSize, NppiPoint, T *, int, NppiSize, int, const Npp32f *, NppiBorderType, cudaStream_t); \
  template NppStatus nppi::filter::filterGaussAdvancedBorder<T, T, 3, false>(                     \
      const T *, int, NppiSize, NppiPoint, T *, int, NppiSize, int, const Npp32f *, NppiBorderType, cudaStream_t); \
  template NppStatus nppi::filter::filterGaussAdvancedBorder<T, T, 4, false>(                     \
      const T *, int, NppiSize, NppiPoint, T *, int, NppiSize, int, const Npp32f *, NppiBorderType, cudaStream_t); \
  template NppStatus nppi::filter::filterGaussAdvancedBorder<T, T, 4, true>(                      \
      const T *, int, NppiSize, NppiPoint, T *, int, NppiSize, int, const Npp32f *, NppiBorderType, cudaStream_t);

NPPI_GAUSS_INSTANCE(Npp8u)
NPPI_GAUSS_INSTANCE(Npp16u)
NPPI_GAUSS_INSTANCE(Npp16s)
NPPI_GAUSS_INSTANCE(Npp32f)
