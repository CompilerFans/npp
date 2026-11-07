// Flood fill executor - see nppi_flood_fill_executor.h for the measured
// NPP 13.4 semantics.  The region is computed as a boolean mask over the ROI:
// the seed is seeded first and the mask is grown by horizontal run extension
// plus 4/8-neighbour propagation until no new pixel is marked.  A second pass
// writes nNewValue (and nBoundaryValue for the *Boundary forms).
#include "nppi_flood_fill_executor.h"

#include <climits>
#include <cuda_runtime.h>

namespace nppi {
namespace floodfill {
namespace {

constexpr int kStatsWords = 8;

// Value parameters travel as kernel arguments (constant memory) instead of
// device pointers: the API passes them by value/host pointer and the kernels
// must not dereference host memory.
template <typename T, int C>
struct ValuePack {
  T v[C];
};

__device__ __forceinline__ bool isBoundaryVariant(int v) {
  return v == kBoundary || v == kRangeBoundary || v == kGradientBoundary;
}

template <typename T, int C, Variant V>
__device__ __forceinline__ const T *pixelAt(const T *src, int srcStep, int x, int y) {
  return (const T *)((const char *)src + (long long)y * srcStep) + (long long)x * C;
}

// Predicate of the region, evaluated on the *original* image values.
template <typename T, int C, Variant V>
__device__ __forceinline__ bool matches(const T *src, int srcStep, int x, int y, const T *seedVal,
                                        const T *minv, const T *maxv) {
  const T *p = pixelAt<T, C, V>(src, srcStep, x, y);
  if (V == kPlain || V == kBoundary) {
    for (int c = 0; c < C; ++c)
      if (p[c] != seedVal[c]) return false;
    return true;
  }
  if (V == kRange || V == kRangeBoundary) {
    for (int c = 0; c < C; ++c)
      if (p[c] < minv[c] || p[c] > maxv[c]) return false;
    return true;
  }
  const T hiLimit = (T)~(T)0;
  for (int c = 0; c < C; ++c) {
    const T lo = (minv[c] >= seedVal[c]) ? (T)0: (T)(seedVal[c] - minv[c]);
    const T hi = (maxv[c] >= (T)(hiLimit - seedVal[c])) ? hiLimit: (T)(seedVal[c] + maxv[c]);
    if (p[c] < lo || p[c] > hi) return false;
  }
  return true;
}

// NPP reads the seed value at `y * nSrcDstStep + x * C * C * sizeof(T)` (the
// column term is scaled by the channel count twice, measured with
//  for C1 that is the correct pixel, for C3 it is
// another pixel or even past the end of the image.  Reproduced here.
template <typename T, int C>
__global__ void captureSeedKernel(const T *src, int srcStep, NppiPoint seed, T *out) {
  const T *p = (const T *)((const char *)src + (long long)seed.y * srcStep +
                           (long long)seed.x * C * C * (int)sizeof(T));
  for (int c = 0; c < C; ++c) out[c] = p[c];
}

__global__ void initMaskKernel(unsigned char *mask, int w, int h, NppiPoint seed) {
  const int total = w * h;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x)
    mask[idx] = (idx == seed.y * w + seed.x) ? 1: 0;
}

__global__ void resetStatsKernel(int *stats) {
  stats[1] = 0;
  stats[2] = INT_MAX;
  stats[3] = INT_MAX;
  stats[4] = -1;
  stats[5] = -1;
}

template <typename T, int C, Variant V>
__device__ __forceinline__ void extendRun(const T *src, int srcStep, int w, int y,
                                          unsigned char *mask, const T *seedVal, const T *minv,
                                          const T *maxv, int x, int *changed) {
  const int row = y * w;
  for (int xn = x - 1; xn >= 0; --xn) {
    if (mask[row + xn]) continue;
    if (!matches<T, C, V>(src, srcStep, xn, y, seedVal, minv, maxv)) break;
    mask[row + xn] = 1;
    atomicExch(changed, 1);
  }
  for (int xn = x + 1; xn < w; ++xn) {
    if (mask[row + xn]) continue;
    if (!matches<T, C, V>(src, srcStep, xn, y, seedVal, minv, maxv)) break;
    mask[row + xn] = 1;
    atomicExch(changed, 1);
  }
}

template <typename T, int C, Variant V, bool EightWay>
__global__ void spreadKernel(const T *src, int srcStep, NppiSize roi, unsigned char *mask,
                             const T *seedVal, ValuePack<T, C> minv, ValuePack<T, C> maxv,
                             int *stats) {
  const int w = roi.width, h = roi.height, total = w * h;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
    if (mask[idx]) continue;
    const int x = idx % w, y = idx / w;
    if (!matches<T, C, V>(src, srcStep, x, y, seedVal, minv.v, maxv.v)) continue;
    bool hit = false;
    for (int dy = -1; dy <= 1 && !hit; ++dy) {
      const int ny = y + dy;
      if (ny < 0 || ny >= h) continue;
      for (int dx = -1; dx <= 1 && !hit; ++dx) {
        if (dx == 0 && dy == 0) continue;
        if (!EightWay && dx != 0 && dy != 0) continue;  // 4-way: orthogonal only
        const int nx = x + dx;
        if (nx < 0 || nx >= w) continue;
        if (mask[ny * w + nx]) hit = true;
      }
    }
    if (!hit) continue;
    mask[idx] = 1;
    atomicExch(stats, 1);
    extendRun<T, C, V>(src, srcStep, w, y, mask, seedVal, minv.v, maxv.v, x, stats);
  }
}

template <typename T, int C, Variant V>
__global__ void writeKernel(T *dst, int dstStep, NppiSize roi, const unsigned char *mask,
                            ValuePack<T, C> newv, ValuePack<T, C> bndv, int *stats) {
  const int w = roi.width, h = roi.height, total = w * h;
  const bool boundaryVariant = isBoundaryVariant((int)V);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
    if (!mask[idx]) continue;
    const int x = idx % w, y = idx / w;
    bool onBorder = false;
    if (boundaryVariant) {
      for (int dy = -1; dy <= 1 && !onBorder; ++dy) {
        const int ny = y + dy;
        if (ny < 0 || ny >= h) continue;  // outside the ROI is not "outside the region"
        for (int dx = -1; dx <= 1 && !onBorder; ++dx) {
          if (dx == 0 && dy == 0) continue;
          const int nx = x + dx;
          if (nx < 0 || nx >= w) continue;
          if (!mask[ny * w + nx]) onBorder = true;
        }
      }
    }
    const T *vals = onBorder ? bndv.v: newv.v;
    T *p = (T *)((char *)dst + (long long)y * dstStep) + (long long)x * C;
    for (int c = 0; c < C; ++c) p[c] = vals[c];
    atomicAdd(&stats[1], 1);
    atomicMin(&stats[2], x);
    atomicMin(&stats[3], y);
    atomicMax(&stats[4], x);
    atomicMax(&stats[5], y);
  }
}

template <typename T, int C, Variant V>
NppStatus run(T *pSrcDst, int nSrcDstStep, NppiPoint oSeed, const T *pNewValue,
              const T *pBoundaryValue, const T *pMin, const T *pMax, NppiNorm eNorm,
              NppiSize oSizeROI, NppiConnectedRegion *pConnectedRegion, Npp8u *pBuffer,
              cudaStream_t stream) {
  if (pSrcDst == nullptr) return NPP_NULL_POINTER_ERROR;
  if (pBuffer == nullptr) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
  if (oSeed.x < 0 || oSeed.y < 0 || oSeed.x >= oSizeROI.width || oSeed.y >= oSizeROI.height)
    return NPP_OUT_OFF_RANGE_ERROR;
  if (eNorm != nppiNormL1 && eNorm != nppiNormInf) return NPP_BAD_ARGUMENT_ERROR;
  if (V == kRange || V == kRangeBoundary) {
    for (int c = 0; c < C; ++c)
      if (pMin[c] > pMax[c]) return NPP_RANGE_ERROR;
  }

  const int w = oSizeROI.width, h = oSizeROI.height;
  const size_t maskBytes = (size_t)w * h;
  unsigned char *mask = pBuffer;
  int *stats = (int *)(pBuffer + ((maskBytes + 15) & ~(size_t)15));
  T *seedVal = (T *)(pBuffer + ((maskBytes + 15) & ~(size_t)15) + sizeof(int) * kStatsWords);
  const int blocks = (w * h + 255) / 256;

  ValuePack<T, C> newPack{}, bndPack{}, minPack{}, maxPack{};
  for (int c = 0; c < C; ++c) {
    newPack.v[c] = pNewValue[c];
    bndPack.v[c] = (pBoundaryValue != nullptr) ? pBoundaryValue[c]: (T)0;
    minPack.v[c] = (pMin != nullptr) ? pMin[c]: (T)0;
    maxPack.v[c] = (pMax != nullptr) ? pMax[c]: (T)0;
  }
  const bool eightWay = (eNorm == nppiNormInf);
  (void)cudaGetLastError();
  initMaskKernel<<<blocks, 256, 0, stream>>>(mask, w, h, oSeed);
  captureSeedKernel<T, C><<<1, 1, 0, stream>>>(pSrcDst, nSrcDstStep, oSeed, seedVal);
  resetStatsKernel<<<1, 1, 0, stream>>>(stats);

  // Grow the mask: every pass marks at least one new pixel until convergence,
  // so the loop is bounded by the number of ROI pixels.
  const long long limit = (long long)w * h + 1;
  for (long long it = 0; it < limit; ++it) {
    const int zero = 0;
    cudaMemcpyAsync(stats, &zero, sizeof(int), cudaMemcpyHostToDevice, stream);
    if (eightWay)
      spreadKernel<T, C, V, true><<<blocks, 256, 0, stream>>>(pSrcDst, nSrcDstStep, oSizeROI, mask,
                                                              seedVal, minPack, maxPack, stats);
    else
      spreadKernel<T, C, V, false><<<blocks, 256, 0, stream>>>(pSrcDst, nSrcDstStep, oSizeROI, mask,
                                                               seedVal, minPack, maxPack, stats);
    int changed = 0;
    cudaMemcpyAsync(&changed, stats, sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    if (!changed) break;
  }

  resetStatsKernel<<<1, 1, 0, stream>>>(stats);
  writeKernel<T, C, V><<<blocks, 256, 0, stream>>>(pSrcDst, nSrcDstStep, oSizeROI, mask, newPack,
                                                   bndPack, stats);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  if (pConnectedRegion != nullptr) {
    int hostStats[kStatsWords] = {0};
    T hostSeed[C] = {};
    if (cudaMemcpyAsync(hostStats, stats, sizeof(int) * kStatsWords, cudaMemcpyDeviceToHost,
                        stream) != cudaSuccess)
      return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    if (cudaMemcpyAsync(hostSeed, seedVal, sizeof(T) * C, cudaMemcpyDeviceToHost, stream) !=
        cudaSuccess)
      return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    if (cudaStreamSynchronize(stream) != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    pConnectedRegion->oBoundingBox.x = hostStats[2];
    pConnectedRegion->oBoundingBox.y = hostStats[3];
    pConnectedRegion->oBoundingBox.width = hostStats[4];
    pConnectedRegion->oBoundingBox.height = hostStats[5];
    pConnectedRegion->nConnectedPixelCount = (Npp32u)hostStats[1];
    pConnectedRegion->aSeedPixelValue[0] = (Npp32u)hostSeed[0];
    pConnectedRegion->aSeedPixelValue[1] = (Npp32u)(C > 1 ? hostSeed[1]: 0);
    pConnectedRegion->aSeedPixelValue[2] = (Npp32u)(C > 2 ? hostSeed[2]: 0);
  }
  return NPP_NO_ERROR;
}

}  // namespace

NppStatus getBufferSize(NppiSize oSizeROI, int *hpBufferSize) {
  if (hpBufferSize == nullptr) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
  *hpBufferSize = 4 * oSizeROI.width * oSizeROI.height + 4 * oSizeROI.width + 64;
  return NPP_NO_ERROR;
}

template <typename T, int C>
NppStatus floodFill(T *pSrcDst, int nSrcDstStep, NppiPoint oSeed, const T *pNewValue,
                    NppiNorm eNorm, NppiSize oSizeROI, NppiConnectedRegion *pConnectedRegion,
                    Npp8u *pBuffer, cudaStream_t stream) {
  return run<T, C, kPlain>(pSrcDst, nSrcDstStep, oSeed, pNewValue, nullptr, nullptr, nullptr, eNorm,
                           oSizeROI, pConnectedRegion, pBuffer, stream);
}

template <typename T, int C>
NppStatus floodFillBoundary(T *pSrcDst, int nSrcDstStep, NppiPoint oSeed, const T *pNewValue,
                            const T *pBoundaryValue, NppiNorm eNorm, NppiSize oSizeROI,
                            NppiConnectedRegion *pConnectedRegion, Npp8u *pBuffer,
                            cudaStream_t stream) {
  return run<T, C, kBoundary>(pSrcDst, nSrcDstStep, oSeed, pNewValue, pBoundaryValue, nullptr,
                              nullptr, eNorm, oSizeROI, pConnectedRegion, pBuffer, stream);
}

template <typename T, int C>
NppStatus floodFillRange(T *pSrcDst, int nSrcDstStep, NppiPoint oSeed, const T *pMin, const T *pMax,
                         const T *pNewValue, NppiNorm eNorm, NppiSize oSizeROI,
                         NppiConnectedRegion *pConnectedRegion, Npp8u *pBuffer,
                         cudaStream_t stream) {
  return run<T, C, kRange>(pSrcDst, nSrcDstStep, oSeed, pNewValue, nullptr, pMin, pMax, eNorm,
                           oSizeROI, pConnectedRegion, pBuffer, stream);
}

template <typename T, int C>
NppStatus floodFillGradient(T *pSrcDst, int nSrcDstStep, NppiPoint oSeed, const T *pMin,
                            const T *pMax, const T *pNewValue, NppiNorm eNorm, NppiSize oSizeROI,
                            NppiConnectedRegion *pConnectedRegion, Npp8u *pBuffer,
                            cudaStream_t stream) {
  return run<T, C, kGradient>(pSrcDst, nSrcDstStep, oSeed, pNewValue, nullptr, pMin, pMax, eNorm,
                              oSizeROI, pConnectedRegion, pBuffer, stream);
}

template <typename T, int C>
NppStatus floodFillRangeBoundary(T *pSrcDst, int nSrcDstStep, NppiPoint oSeed, const T *pMin,
                                 const T *pMax, const T *pNewValue, const T *pBoundaryValue,
                                 NppiNorm eNorm, NppiSize oSizeROI,
                                 NppiConnectedRegion *pConnectedRegion, Npp8u *pBuffer,
                                 cudaStream_t stream) {
  return run<T, C, kRangeBoundary>(pSrcDst, nSrcDstStep, oSeed, pNewValue, pBoundaryValue, pMin,
                                   pMax, eNorm, oSizeROI, pConnectedRegion, pBuffer, stream);
}

template <typename T, int C>
NppStatus floodFillGradientBoundary(T *pSrcDst, int nSrcDstStep, NppiPoint oSeed, const T *pMin,
                                    const T *pMax, const T *pNewValue, const T *pBoundaryValue,
                                    NppiNorm eNorm, NppiSize oSizeROI,
                                    NppiConnectedRegion *pConnectedRegion, Npp8u *pBuffer,
                                    cudaStream_t stream) {
  return run<T, C, kGradientBoundary>(pSrcDst, nSrcDstStep, oSeed, pNewValue, pBoundaryValue, pMin,
                                      pMax, eNorm, oSizeROI, pConnectedRegion, pBuffer, stream);
}

#define NPPI_FLOOD_INSTANTIATE(T, C)                                                              \
  template NppStatus floodFill<T, C>(T *, int, NppiPoint, const T *, NppiNorm, NppiSize,          \
                                     NppiConnectedRegion *, Npp8u *, cudaStream_t);               \
  template NppStatus floodFillBoundary<T, C>(T *, int, NppiPoint, const T *, const T *, NppiNorm, \
                                             NppiSize, NppiConnectedRegion *, Npp8u *,            \
                                             cudaStream_t);                                       \
  template NppStatus floodFillRange<T, C>(T *, int, NppiPoint, const T *, const T *, const T *,   \
                                          NppiNorm, NppiSize, NppiConnectedRegion *, Npp8u *,     \
                                          cudaStream_t);                                          \
  template NppStatus floodFillGradient<T, C>(T *, int, NppiPoint, const T *, const T *,           \
                                             const T *, NppiNorm, NppiSize,                       \
                                             NppiConnectedRegion *, Npp8u *, cudaStream_t);       \
  template NppStatus floodFillRangeBoundary<T, C>(T *, int, NppiPoint, const T *, const T *,      \
                                                  const T *, const T *, NppiNorm, NppiSize,       \
                                                  NppiConnectedRegion *, Npp8u *, cudaStream_t);  \
  template NppStatus floodFillGradientBoundary<T, C>(T *, int, NppiPoint, const T *, const T *,   \
                                                     const T *, const T *, NppiNorm, NppiSize,    \
                                                     NppiConnectedRegion *, Npp8u *,              \
                                                     cudaStream_t);

NPPI_FLOOD_INSTANTIATE(Npp8u, 1)
NPPI_FLOOD_INSTANTIATE(Npp8u, 3)
NPPI_FLOOD_INSTANTIATE(Npp16u, 1)
NPPI_FLOOD_INSTANTIATE(Npp16u, 3)
NPPI_FLOOD_INSTANTIATE(Npp32u, 1)
NPPI_FLOOD_INSTANTIATE(Npp32u, 3)
#undef NPPI_FLOOD_INSTANTIATE

}  // namespace floodfill
}  // namespace nppi

// The buffer-size helper has no _Ctx twin, so it is not generated.
extern "C" NppStatus nppiFloodFillGetBufferSize(NppiSize oSizeROI, int *hpBufferSize) {
  return nppi::floodfill::getBufferSize(oSizeROI, hpBufferSize);
}
