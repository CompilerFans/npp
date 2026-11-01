// Wiener filter kernel and executor entry point.
// See nppi_filter_wiener_executor.h for the measured family contract.
#include "npp.h"
#include "nppi_filter_wiener_executor.h"

#include <cmath>
#include <cuda_runtime.h>
#include <type_traits>

namespace nppi {
namespace filter {
namespace {

constexpr int kBlockX = 16;
constexpr int kBlockY = 16;

struct WienerNoise {
  float n[4];
};

template <typename T>
__host__ __device__ __forceinline__ T wiener_emit(float v) {
  if constexpr (std::is_floating_point_v<T>) {
    return static_cast<T>(v);
  } else {
    const float hi = std::is_same_v<T, Npp8u> ? 255.0f : 32767.0f;
    const float lo = std::is_same_v<T, Npp8u> ? 0.0f : -32768.0f;
    if (v > hi) v = hi;
    if (v < lo) v = lo;
    return static_cast<T>(v);  // C truncation toward zero
  }
}

// NPP's mean-set sum for the quirky statistics window (see header).
template <typename T, int Channels>
__device__ __forceinline__ float wiener_stats(const T *__restrict__ pSrc, int nSrcStep, int x, int y, int mw,
                                              int mh, int ax, int ay, int loX, int hiX, int loY, int hiY,
                                              int c, bool rectangle) {
  const char *const base = reinterpret_cast<const char *>(pSrc);
  auto clampx = [&](int v) { return v < loX ? loX : (v > hiX ? hiX : v); };
  auto clampy = [&](int v) { return v < loY ? loY : (v > hiY ? hiY : v); };
  auto load = [&](int yy, int xx) {
    const T *row = reinterpret_cast<const T *>(base + static_cast<std::ptrdiff_t>(yy) * nSrcStep);
    return static_cast<float>(row[static_cast<std::ptrdiff_t>(xx) * Channels + c]);
  };

  float sum = 0.0f;
  if (rectangle) {
    for (int dy = ay - (mh - 1); dy <= ay; ++dy) {
      const int ry = clampy(y + dy);
      for (int dx = ax - (mw - 1); dx <= ax; ++dx) sum += load(ry, clampx(x + dx));
    }
    return sum;
  }
  if (mh == 1) {
    const int ry = clampy(y + ay);
    for (int dx = ax - (mw - 1); dx <= ax; ++dx) sum += load(ry, clampx(x + dx));
    return sum;
  }
  const int rows = mh - 1;
  for (int r = 0; r < rows; ++r) {
    const int ry = clampy(y + ay - r);
    const int x0 = ax - (mw - 1) + (r == 0 ? 0 : 1);
    for (int dx = x0; dx <= ax; ++dx) sum += load(ry, clampx(x + dx));
  }
  return sum;
}

// NPP's variance-set accumulation: sum of fma(v, v, -mean*mean), exactly the
// SASS sequence (FFMA then FADD) over the variance window.
template <typename T, int Channels>
__device__ __forceinline__ void wiener_vsum(const T *__restrict__ pSrc, int nSrcStep, int x, int y, int mw,
                                            int mh, int ax, int ay, int loX, int hiX, int loY, int hiY,
                                            int c, float mean, float *acc, bool rectangle) {
  const char *const base = reinterpret_cast<const char *>(pSrc);
  auto clampx = [&](int v) { return v < loX ? loX : (v > hiX ? hiX : v); };
  auto clampy = [&](int v) { return v < loY ? loY : (v > hiY ? hiY : v); };
  auto add = [&](int yy, int xx) {
    const T *row = reinterpret_cast<const T *>(base + static_cast<std::ptrdiff_t>(yy) * nSrcStep);
    const float v = static_cast<float>(row[static_cast<std::ptrdiff_t>(xx) * Channels + c]);
    *acc += fmaf(v, v, -mean * mean);
  };
  if (rectangle) {
    for (int dy = ay - (mh - 1); dy <= ay; ++dy) {
      const int ry = clampy(y + dy);
      for (int dx = ax - (mw - 1); dx <= ax; ++dx) add(ry, clampx(x + dx));
    }
    return;
  }
  if (mh == 1) {
    const int ry = clampy(y + ay - 1);
    for (int dx = ax - (mw - 1); dx <= ax; ++dx) add(ry, clampx(x + dx));
    return;
  }
  const int rows = mh - 1;
  for (int r = 0; r < rows; ++r) {
    const int ry = clampy(y + ay - r);
    const int x0 = ax - (mw - 1) + (r == 0 ? 0 : 1);
    for (int dx = x0; dx <= ax; ++dx) add(ry, clampx(x + dx));
  }
}

template <typename T, int Channels, bool AlphaPr>
__global__ void wiener_kernel(const T *__restrict__ pSrc, int nSrcStep, T *__restrict__ pDst, int nDstStep,
                              int width, int height, int mw, int mh, int ax, int ay, WienerNoise noise,
                              float noiseScale, int srcW, int srcH, int offX, int offY) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  const int nch = AlphaPr ? 3 : Channels;
  const int loX = -offX, hiX = srcW - 1 - offX;
  const int loY = -offY, hiY = srcH - 1 - offY;
  const float invN = 1.0f / static_cast<float>(mw * mh);
  const char *const srcBase = reinterpret_cast<const char *>(pSrc);
  const T *srow = reinterpret_cast<const T *>(srcBase + static_cast<std::ptrdiff_t>(y) * nSrcStep);
  const float srcv[4] = {nch > 0 ? static_cast<float>(srow[static_cast<std::ptrdiff_t>(x) * Channels + 0]) : 0.f,
                         nch > 1 ? static_cast<float>(srow[static_cast<std::ptrdiff_t>(x) * Channels + 1]) : 0.f,
                         nch > 2 ? static_cast<float>(srow[static_cast<std::ptrdiff_t>(x) * Channels + 2]) : 0.f,
                         nch > 3 ? static_cast<float>(srow[static_cast<std::ptrdiff_t>(x) * Channels + 3]) : 0.f};
  T out[4] = {T(0), T(0), T(0), T(0)};
  for (int c = 0; c < nch; ++c) {
    const float sum = wiener_stats<T, Channels>(pSrc, nSrcStep, x, y, mw, mh, ax, ay, loX, hiX, loY, hiY, c,
                                                AlphaPr);
    const float mean = sum * invN;
    float vsum = 0.0f;
    wiener_vsum<T, Channels>(pSrc, nSrcStep, x, y, mw, mh, ax, ay, loX, hiX, loY, hiY, c, mean, &vsum,
                             AlphaPr);
    const float var = vsum * invN;
    const float ne = noise.n[c] * noiseScale;
    float o;
    if (noise.n[c] == 0.0f) {
      o = var;
    } else if (var > ne) {
      o = mean + (var - ne) / var * (srcv[c] - mean);
    } else {
      o = mean;
    }
    out[c] = wiener_emit<T>(o);
  }
  T *drow = reinterpret_cast<T *>(reinterpret_cast<char *>(pDst) + static_cast<std::ptrdiff_t>(y) * nDstStep) +
            static_cast<std::ptrdiff_t>(x) * Channels;
  for (int c = 0; c < nch; ++c) drow[c] = out[c];
}

template <typename T, int Channels>
NppStatus check_dst_step(int nDstStep, int roiWidth) {
  const int pix = static_cast<int>(sizeof(T)) * Channels;
  const int align = Channels == 4 ? pix : static_cast<int>(sizeof(T));
  const unsigned uStep = static_cast<unsigned>(nDstStep);
  if (uStep < static_cast<unsigned>(roiWidth * pix)) return NPP_STEP_ERROR;
  if (align > 1 && uStep % static_cast<unsigned>(align) != 0) return NPP_NOT_EVEN_STEP_ERROR;
  return NPP_NO_ERROR;
}

}  // namespace

template <typename T, int Channels, bool AlphaPreserve>
NppStatus filterWienerBorder(const T *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, T *pDst,
                             int nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor,
                             const Npp32f *aNoise, NppiBorderType eBorderType, cudaStream_t stream) {
  const int nch = AlphaPreserve ? 3 : Channels;
  // NPP 13.4 validates (and dereferences) aNoise before anything else.
  if (!aNoise) return NPP_NULL_POINTER_ERROR;
  for (int c = 0; c < nch; ++c) {
    const float n = aNoise[c];
    if (!(n >= 0.0f && n <= 1.0f)) return NPP_OUT_OFF_RANGE_ERROR;
  }
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSrcSize.width <= 0 || oSrcSize.height <= 0) return NPP_SIZE_ERROR;
  if (oSrcOffset.x < 0 || oSrcOffset.y < 0 || oSrcOffset.x >= oSrcSize.width || oSrcOffset.y >= oSrcSize.height)
    return NPP_OUT_OFF_RANGE_ERROR;
  if (eBorderType != NPP_BORDER_REPLICATE) return NPP_NOT_SUPPORTED_MODE_ERROR;
  if (oMaskSize.width <= 0 || oMaskSize.height <= 0) return NPP_SIZE_ERROR;
  if (oAnchor.x >= oMaskSize.width || oAnchor.y >= oMaskSize.height) return NPP_OUT_OFF_RANGE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;

  NppStatus st = check_dst_step<T, Channels>(nDstStep, oSizeROI.width);
  if (st != NPP_NO_ERROR) return st;
  (void)cudaGetLastError();
  WienerNoise noise{};
  for (int c = 0; c < nch; ++c) noise.n[c] = aNoise[c];
  float noiseScale = 1.0f;
  if constexpr (std::is_same_v<T, Npp8u>) noiseScale = 255.0f * 255.0f;
  if constexpr (std::is_same_v<T, Npp16s>) noiseScale = 32767.0f * 32767.0f;

  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + kBlockX - 1) / kBlockX, (oSizeROI.height + kBlockY - 1) / kBlockY);
  wiener_kernel<T, Channels, AlphaPreserve><<<grid, block, 0, stream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, oMaskSize.width, oMaskSize.height,
      oAnchor.x, oAnchor.y, noise, noiseScale, oSrcSize.width, oSrcSize.height, oSrcOffset.x, oSrcOffset.y);
  cudaStreamSynchronize(stream);
  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

}  // namespace filter
}  // namespace nppi

// ---------------------------------------------------------------------------
// Explicit instantiations (f10 package: nppiFilterWienerBorder, 24 functions).
// ---------------------------------------------------------------------------
#define NPPI_WIENER_INSTANCE(T)                                                                   \
  template NppStatus nppi::filter::filterWienerBorder<T, 1, false>(                               \
      const T *, int, NppiSize, NppiPoint, T *, int, NppiSize, NppiSize, NppiPoint, const Npp32f *, \
      NppiBorderType, cudaStream_t);                                                              \
  template NppStatus nppi::filter::filterWienerBorder<T, 3, false>(                               \
      const T *, int, NppiSize, NppiPoint, T *, int, NppiSize, NppiSize, NppiPoint, const Npp32f *, \
      NppiBorderType, cudaStream_t);                                                              \
  template NppStatus nppi::filter::filterWienerBorder<T, 4, false>(                               \
      const T *, int, NppiSize, NppiPoint, T *, int, NppiSize, NppiSize, NppiPoint, const Npp32f *, \
      NppiBorderType, cudaStream_t);                                                              \
  template NppStatus nppi::filter::filterWienerBorder<T, 4, true>(                                \
      const T *, int, NppiSize, NppiPoint, T *, int, NppiSize, NppiSize, NppiPoint, const Npp32f *, \
      NppiBorderType, cudaStream_t);

NPPI_WIENER_INSTANCE(Npp8u)
NPPI_WIENER_INSTANCE(Npp16s)
NPPI_WIENER_INSTANCE(Npp32f)
