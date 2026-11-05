// NPPI LUT executors (package c1) -- device kernels + host entry points.
//
// The numerical recipes below were reverse engineered from NVIDIA NPP 13.4
// goldens recorded on the RTX 5090 and, for the cubic sampler, from the SASS
// of libnppicc's SampleLUTCubic<int> instantiation.  See the header for the
// contract summary.

#include "nppi_lut_executor.h"
#include "npp_kernel_utils.h"

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace nppi {
namespace lut {
namespace {

using npp_kernel::BlockConfig;
using npp_kernel::checkCudaKernelError;
using npp_kernel::getDstRow;
using npp_kernel::getSrcRow;
using npp_kernel::getThreadCoords;

constexpr int kMaxTables = 4;

template <typename ValT> struct LutTables {
  const ValT *values[kMaxTables];
  const ValT *levels[kMaxTables];
  int nLevels[kMaxTables];
};

template <typename DstT, typename ValT> __device__ __forceinline__ DstT lutCast(ValT value) {
  if constexpr (std::is_same<DstT, Npp32f>::value) {
    return static_cast<Npp32f>(value);
  } else {
    return static_cast<DstT>(static_cast<Npp32s>(value));
  }
}

// NPP's public contract takes device pointers, but the mcMPP legacy wrappers
// (and test/unit/nppi/test_npp_stream_sync.cpp) additionally pass pageable
// host arrays and expect them to be staged before the call returns.  Detect
// the pointer kind and stage host tables so both call styles work.
inline bool isDevicePointer(const void *ptr) {
  cudaPointerAttributes attributes{};
  cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);
  if (err != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  return attributes.type == cudaMemoryTypeDevice;
}

// Forward interval scan.  Returns -1 when the sample is outside the mapping
// range (the caller copies the source pixel through) and otherwise the index
// of the last level <= sample, clamped to nLevels-2.
template <typename SrcT, typename ValT>
__device__ __forceinline__ int lutFindIndex(SrcT src, const ValT *levels, int nLevels) {
  const ValT s = static_cast<ValT>(src);
  if (s < levels[0] || s >= levels[nLevels - 1]) {
    return -1;
  }
  int idx = 0;
  for (int i = 1; i < nLevels; ++i) {
    if (s >= levels[i]) {
      idx = i;
    } else {
      break;
    }
  }
  return idx < nLevels - 1 ? idx : nLevels - 2;
}

// Point sample uses the source value unchanged outside the level range.
template <typename SrcT, typename ValT>
__device__ __forceinline__ ValT lutSamplePoint(SrcT src, const ValT *values, const ValT *levels, int nLevels) {
  const int idx = lutFindIndex(src, levels, nLevels);
  if (idx < 0) {
    return static_cast<ValT>(src);
  }
  return values[idx];
}

// Linear sample: integer dtypes use C truncating integer division (verified
// against the 8u/16u full-ramp goldens); 32f stays in float.
template <typename SrcT, typename ValT>
__device__ __forceinline__ ValT lutSampleLinear(SrcT src, const ValT *values, const ValT *levels, int nLevels) {
  const int idx = lutFindIndex(src, levels, nLevels);
  if (idx < 0) {
    return static_cast<ValT>(src);
  }
  if constexpr (std::is_same<SrcT, Npp32f>::value) {
    const float s = static_cast<float>(src);
    const float y0 = static_cast<float>(values[idx]);
    const float y1 = static_cast<float>(values[idx + 1]);
    const float x0 = static_cast<float>(levels[idx]);
    const float x1 = static_cast<float>(levels[idx + 1]);
    return static_cast<ValT>(y0 + (s - x0) * (y1 - y0) / (x1 - x0));
  } else {
    const long long s = static_cast<long long>(src);
    const long long x0 = static_cast<long long>(levels[idx]);
    const long long x1 = static_cast<long long>(levels[idx + 1]);
    const long long y0 = static_cast<long long>(values[idx]);
    const long long y1 = static_cast<long long>(values[idx + 1]);
    const long long delta = x1 - x0;
    return static_cast<ValT>(y0 + (s - x0) * (y1 - y0) / delta);
  }
}

// Cubic sample: 4-point cubic through (levels[i-1..i+2], values[i-1..i+2]) with
// linear extrapolation of the missing end point.  The formula is the one the
// NPP SampleLUTCubic device function evaluates (SASS: libnppicc
// mapLUT_8u_C1R<uchar,int,SampleLUTCubic<int>>); the explicit fmaf ordering
// matters at integer rounding boundaries.
template <typename SrcT, typename ValT>
__device__ __forceinline__ ValT lutSampleCubic(SrcT src, const ValT *values, const ValT *levels, int nLevels) {
  const int idx = lutFindIndex(src, levels, nLevels);
  if (idx < 0) {
    return static_cast<ValT>(src);
  }
  const float x0 = static_cast<float>(levels[idx]);
  const float x1 = static_cast<float>(levels[idx + 1]);
  const float v0 = static_cast<float>(values[idx]);
  const float v1 = static_cast<float>(values[idx + 1]);
  const float xm = idx >= 1 ? static_cast<float>(levels[idx - 1]) : 2.0f * x0 - x1;
  const float vm = idx >= 1 ? static_cast<float>(values[idx - 1]) : 2.0f * v0 - v1;
  const float x2 = idx + 2 < nLevels ? static_cast<float>(levels[idx + 2]) : 2.0f * x1 - x0;
  const float v2 = idx + 2 < nLevels ? static_cast<float>(values[idx + 2]) : 2.0f * v1 - v0;

  const float s2 = (v2 - v1) / (x2 - x1);
  const float s1 = (v1 - v0) / (x1 - x0);
  const float s0 = (v0 - vm) / (x0 - xm);

  const float dx = x1 - x0;
  const float d0 = 2.0f * (x1 - xm);
  const float q = dx / d0;
  const float a0 = 6.0f * (s1 - s0);
  const float b0 = 6.0f * (s2 - s1);
  const float e1 = fmaf(-q, dx, 2.0f * (x2 - x0));
  const float b = fmaf(-q, a0, b0) / e1;
  const float a = (a0 - dx * b) / d0;
  const float c3 = (b - a) / (6.0f * dx);
  const float c1 = fmaf(fmaf(2.0f * dx, a, dx * b), -1.0f / 6.0f, s1);

  const float h = static_cast<float>(src) - x0;
  const float h2 = h * h;
  const float result = fmaf(c1, h, fmaf(a * 0.5f, h2, h2 * h * c3));
  if constexpr (std::is_same<ValT, Npp32f>::value) {
    return v0 + result;
  } else {
    return static_cast<ValT>(static_cast<Npp32s>(v0 + result));
  }
}

template <typename SrcT, typename ValT, LutMode Mode>
__device__ __forceinline__ ValT lutSample(SrcT src, const ValT *values, const ValT *levels, int nLevels) {
  if constexpr (Mode == LutMode::Point) {
    return lutSamplePoint(src, values, levels, nLevels);
  } else if constexpr (Mode == LutMode::Linear) {
    return lutSampleLinear(src, values, levels, nLevels);
  } else {
    return lutSampleCubic(src, values, levels, nLevels);
  }
}

template <typename SrcT, int Channels, bool AlphaOnly, LutMode Mode>
__global__ void lutKernel(const SrcT *pSrc, int nSrcStep, SrcT *pDst, int nDstStep, int width, int height,
                          LutTables<typename LutValue<SrcT>::type> tables, bool fastPath) {
  int x, y;
  if (!getThreadCoords(x, y, width, height)) {
    return;
  }
  using ValT = typename LutValue<SrcT>::type;
  const SrcT *srcRow = getSrcRow(pSrc, y, nSrcStep);
  SrcT *dstRow = getDstRow(pDst, y, nDstStep);
  const int tables_used = AlphaOnly ? Channels - 1 : Channels;
  for (int c = 0; c < tables_used; ++c) {
    const int index = x * Channels + c;
    const SrcT src = srcRow[index];
    if (fastPath) {
      dstRow[index] = lutCast<SrcT, ValT>(tables.values[c][static_cast<int>(src)]);
    } else {
      dstRow[index] = lutCast<SrcT, ValT>(lutSample<SrcT, ValT, Mode>(src, tables.values[c], tables.levels[c],
                                                                     tables.nLevels[c]));
    }
  }
}

template <typename SrcT, int Channels, bool AlphaOnly, LutMode Mode>
NppStatus launchLut(const SrcT *pSrc, int nSrcStep, SrcT *pDst, int nDstStep, NppiSize oSizeROI,
                    const typename LutValue<SrcT>::type *const *pValues,
                    const typename LutValue<SrcT>::type *const *pLevels, const int *nLevels, cudaStream_t stream) {
  using ValT = typename LutValue<SrcT>::type;
  constexpr int kTablesUsed = AlphaOnly ? Channels - 1 : Channels;

  if (!pSrc || !pDst || !pValues || !pLevels || !nLevels) {
    return NPP_NULL_POINTER_ERROR;
  }
  for (int c = 0; c < kTablesUsed; ++c) {
    if (!pValues[c] || !pLevels[c]) {
      return NPP_NULL_POINTER_ERROR;
    }
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  const int maxLevels = std::is_same<SrcT, Npp8u>::value ? 256 : 1024;
  bool fastPath = std::is_same<SrcT, Npp8u>::value;
  for (int c = 0; c < kTablesUsed; ++c) {
    if (nLevels[c] < 2 || nLevels[c] > maxLevels) {
      return NPP_LUT_NUMBER_OF_LEVELS_ERROR;
    }
    if (nLevels[c] != 256) {
      fastPath = false;
    }
  }

  // NPP short-circuits a width-0 / height>0 ROI; height 0 still launches an
  // invalid (zero-sized) grid and reports a kernel execution error.
  if (oSizeROI.width == 0 && oSizeROI.height > 0) {
    return NPP_SUCCESS;
  }

  bool stage = false;
  for (int c = 0; c < kTablesUsed; ++c) {
    if (!isDevicePointer(pValues[c]) || !isDevicePointer(pLevels[c])) {
      stage = true;
    }
  }

  LutTables<ValT> tables{};
  ValT *stagedValues[kMaxTables] = {nullptr, nullptr, nullptr, nullptr};
  ValT *stagedLevels[kMaxTables] = {nullptr, nullptr, nullptr, nullptr};
  for (int c = 0; c < kTablesUsed; ++c) {
    tables.values[c] = pValues[c];
    tables.levels[c] = pLevels[c];
    tables.nLevels[c] = nLevels[c];
  }
  if (stage) {
    for (int c = 0; c < kTablesUsed; ++c) {
      const size_t tableBytes = static_cast<size_t>(nLevels[c]) * sizeof(ValT);
      if (cudaMallocAsync(reinterpret_cast<void **>(&stagedValues[c]), tableBytes, stream) != cudaSuccess ||
          cudaMallocAsync(reinterpret_cast<void **>(&stagedLevels[c]), tableBytes, stream) != cudaSuccess) {
        (void)cudaGetLastError();
        for (int j = 0; j <= c; ++j) {
          if (stagedValues[j]) (void)cudaFreeAsync(stagedValues[j], stream);
          if (stagedLevels[j]) (void)cudaFreeAsync(stagedLevels[j], stream);
        }
        return NPP_MEMORY_ALLOCATION_ERR;
      }
      cudaError_t copyValues = cudaMemcpyAsync(stagedValues[c], pValues[c], tableBytes, cudaMemcpyHostToDevice, stream);
      cudaError_t copyLevels = cudaMemcpyAsync(stagedLevels[c], pLevels[c], tableBytes, cudaMemcpyHostToDevice, stream);
      if (copyValues != cudaSuccess || copyLevels != cudaSuccess) {
        (void)cudaGetLastError();
        for (int j = 0; j <= c; ++j) {
          if (stagedValues[j]) (void)cudaFreeAsync(stagedValues[j], stream);
          if (stagedLevels[j]) (void)cudaFreeAsync(stagedLevels[j], stream);
        }
        return NPP_MEMCPY_ERROR;
      }
      tables.values[c] = stagedValues[c];
      tables.levels[c] = stagedLevels[c];
    }
  }

  dim3 block = BlockConfig<>::blockSize();
  dim3 grid = BlockConfig<>::gridSize(oSizeROI.width, oSizeROI.height);
  lutKernel<SrcT, Channels, AlphaOnly, Mode><<<grid, block, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                         oSizeROI.width, oSizeROI.height, tables,
                                                                         fastPath);
  NppStatus status = checkCudaKernelError();
  for (int c = 0; c < kTablesUsed; ++c) {
    if (stagedValues[c]) (void)cudaFreeAsync(stagedValues[c], stream);
    if (stagedLevels[c]) (void)cudaFreeAsync(stagedLevels[c], stream);
  }
  return status;
}

template <typename SrcT, LutMode Mode>
NppStatus launchLutC1(const SrcT *pSrc, int nSrcStep, SrcT *pDst, int nDstStep, NppiSize oSizeROI,
                      const typename LutValue<SrcT>::type *pValues,
                      const typename LutValue<SrcT>::type *pLevels, int nLevels, cudaStream_t stream) {
  const typename LutValue<SrcT>::type *values[1] = {pValues};
  const typename LutValue<SrcT>::type *levels[1] = {pLevels};
  const int counts[1] = {nLevels};
  return launchLut<SrcT, 1, false, Mode>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, values, levels, counts, stream);
}

// ---------------------------------------------------------------------------
// Palette kernels
// ---------------------------------------------------------------------------

__device__ __forceinline__ unsigned int paletteMask(int bitSize) {
  return bitSize >= 32 ? 0xFFFFFFFFu : ((1u << bitSize) - 1u);
}

template <typename SrcT, typename TabT, typename DstT>
__global__ void paletteC1Kernel(const SrcT *pSrc, int nSrcStep, DstT *pDst, int nDstStep, int width, int height,
                                const TabT *pTable, int bitSize) {
  int x, y;
  if (!getThreadCoords(x, y, width, height)) {
    return;
  }
  const unsigned int mask = paletteMask(bitSize);
  const SrcT *srcRow = getSrcRow(pSrc, y, nSrcStep);
  DstT *dstRow = getDstRow(pDst, y, nDstStep);
  dstRow[x] = static_cast<DstT>(pTable[static_cast<unsigned int>(srcRow[x]) & mask]);
}

template <typename SrcT>
__global__ void palette24Kernel(const SrcT *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width, int height,
                                const Npp8u *pTable, int bitSize) {
  int x, y;
  if (!getThreadCoords(x, y, width, height)) {
    return;
  }
  const unsigned int mask = paletteMask(bitSize);
  const SrcT *srcRow = getSrcRow(pSrc, y, nSrcStep);
  Npp8u *dstRow = getDstRow(pDst, y, nDstStep) + x * 3;
  const Npp8u *entry = pTable + (static_cast<unsigned int>(srcRow[x]) & mask) * 3u;
  dstRow[0] = entry[0];
  dstRow[1] = entry[1];
  dstRow[2] = entry[2];
}

template <typename SrcT, int Channels, bool AlphaOnly>
__global__ void paletteMultiKernel(const SrcT *pSrc, int nSrcStep, SrcT *pDst, int nDstStep, int width, int height,
                                   const SrcT *const *pTables, int bitSize) {
  int x, y;
  if (!getThreadCoords(x, y, width, height)) {
    return;
  }
  const unsigned int mask = paletteMask(bitSize);
  const SrcT *srcRow = getSrcRow(pSrc, y, nSrcStep);
  SrcT *dstRow = getDstRow(pDst, y, nDstStep);
  const int tables_used = AlphaOnly ? Channels - 1 : Channels;
  for (int c = 0; c < tables_used; ++c) {
    const int index = x * Channels + c;
    // 5090 quirk: the 16u C4 kernel maps the fourth channel through the
    // third table (the 8u C4 kernel uses pTables[3]).
    const int table = (sizeof(SrcT) == 2 && Channels == 4 && c == 3) ? 2 : c;
    dstRow[index] = pTables[table][static_cast<unsigned int>(srcRow[index]) & mask];
  }
}

template <typename T>
__global__ void paletteSwapKernel(const T *pSrc, int nSrcStep, int nAlphaValue, T *pDst, int nDstStep, int width,
                                  int height, const T *const *pTables, int bitSize) {
  int x, y;
  if (!getThreadCoords(x, y, width, height)) {
    return;
  }
  const unsigned int mask = paletteMask(bitSize);
  const T *srcRow = getSrcRow(pSrc, y, nSrcStep);
  T *dstRow = getDstRow(pDst, y, nDstStep);
  const int src_index = x * 3;
  const int dst_index = x * 4;
  if constexpr (sizeof(T) == 1) {
    // 8u: destination slots 0 and 2 come from the last and first tables.
    dstRow[dst_index + 0] = pTables[2][static_cast<unsigned int>(srcRow[src_index + 2]) & mask];
    dstRow[dst_index + 1] = pTables[1][static_cast<unsigned int>(srcRow[src_index + 1]) & mask];
    dstRow[dst_index + 2] = pTables[0][static_cast<unsigned int>(srcRow[src_index + 0]) & mask];
  } else {
    // 16u: the *source channels* are reversed while the tables stay in order
    // (dst0 = table0[src2], dst2 = table2[src0]; measured on the 5090).
    dstRow[dst_index + 0] = pTables[0][static_cast<unsigned int>(srcRow[src_index + 2]) & mask];
    dstRow[dst_index + 1] = pTables[1][static_cast<unsigned int>(srcRow[src_index + 1]) & mask];
    dstRow[dst_index + 2] = pTables[2][static_cast<unsigned int>(srcRow[src_index + 0]) & mask];
  }
  // Alpha: 8u accepts only 0..255 (larger values leave the destination
  // untouched); 16u accepts any non-negative value and truncates it (65536 -> 0).
  const long long alpha = static_cast<long long>(nAlphaValue);
  const long long max_alpha = static_cast<long long>((1ULL << (8 * sizeof(T))) - 1ULL);
  if (alpha >= 0 && (sizeof(T) != 1 || alpha <= max_alpha)) {
    dstRow[dst_index + 3] = static_cast<T>(nAlphaValue);
  }
}

// ---------------------------------------------------------------------------
// Trilinear kernel
// ---------------------------------------------------------------------------

__device__ __forceinline__ void trilinearAxis(int value, const Npp8u *levels, int nLevels, int &index, float &frac) {
  // Below levels[0] NPP extrapolates with a negative fraction (measured:
  // src 3 under levels {10,100,200} and src 0 under {5,130,250}); only the
  // top level clamps.
  if (value >= levels[nLevels - 1]) {
    index = nLevels - 2;
    frac = 1.0f;
    return;
  }
  int idx = 0;
  for (int i = 1; i < nLevels; ++i) {
    if (value >= levels[i]) {
      idx = i;
    } else {
      break;
    }
  }
  if (idx > nLevels - 2) {
    idx = nLevels - 2;
  }
  index = idx;
  frac = static_cast<float>(value - levels[idx]) / static_cast<float>(levels[idx + 1] - levels[idx]);
}

__device__ __forceinline__ int trilinearByte(unsigned int packed, int channel) {
  return static_cast<int>((packed >> (8 * channel)) & 0xFFu);
}

__global__ void trilinearKernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width, int height,
                                const Npp32u *pValues, const Npp8u *levelsX, const Npp8u *levelsY,
                                const Npp8u *levelsZ, int nx, int ny, int nz, bool alphaSrc) {
  int x, y;
  if (!getThreadCoords(x, y, width, height)) {
    return;
  }
  const Npp8u *srcRow = getSrcRow(pSrc, y, nSrcStep) + x * 4;
  Npp8u *dstRow = getDstRow(pDst, y, nDstStep) + x * 4;

  int ix, iy, iz;
  float fx, fy, fz;
  trilinearAxis(srcRow[0], levelsX, nx, ix, fx);
  trilinearAxis(srcRow[1], levelsY, ny, iy, fy);
  trilinearAxis(srcRow[2], levelsZ, nz, iz, fz);

  float out[3] = {0.0f, 0.0f, 0.0f};
  for (int dz = 0; dz < 2; ++dz) {
    const float wz = dz ? fz : 1.0f - fz;
    for (int dy = 0; dy < 2; ++dy) {
      const float wy = dy ? fy : 1.0f - fy;
      for (int dx = 0; dx < 2; ++dx) {
        const float wx = dx ? fx : 1.0f - fx;
        const unsigned int corner =
            pValues[((iz + dz) * ny + (iy + dy)) * nx + (ix + dx)];
        const float weight = wx * wy * wz;
        out[0] += weight * static_cast<float>(trilinearByte(corner, 0));
        out[1] += weight * static_cast<float>(trilinearByte(corner, 1));
        out[2] += weight * static_cast<float>(trilinearByte(corner, 2));
      }
    }
  }
  for (int c = 0; c < 3; ++c) {
    const float clamped = fminf(fmaxf(out[c], 0.0f), 255.0f);
    dstRow[c] = static_cast<Npp8u>(static_cast<int>(clamped + 0.5f));
  }
  if (alphaSrc) {
    dstRow[3] = srcRow[3];
  }
}

} // namespace

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

template <typename SrcT, LutMode Mode>
NppStatus runC1(const SrcT *pSrc, int nSrcStep, SrcT *pDst, int nDstStep, NppiSize oSizeROI,
                const typename LutValue<SrcT>::type *pValues, const typename LutValue<SrcT>::type *pLevels,
                int nLevels, cudaStream_t stream) {
  return launchLutC1<SrcT, Mode>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, stream);
}

template <typename SrcT, int Channels, bool AlphaOnly, LutMode Mode>
NppStatus run(const SrcT *pSrc, int nSrcStep, SrcT *pDst, int nDstStep, NppiSize oSizeROI,
              const typename LutValue<SrcT>::type *const *pValues,
              const typename LutValue<SrcT>::type *const *pLevels, const int *nLevels, cudaStream_t stream) {
  return launchLut<SrcT, Channels, AlphaOnly, Mode>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels,
                                                    nLevels, stream);
}

template <typename SrcT, typename TabT, typename DstT>
NppStatus paletteC1(const SrcT *pSrc, int nSrcStep, DstT *pDst, int nDstStep, NppiSize oSizeROI, const TabT *pTable,
                    int nBitSize, cudaStream_t stream) {
  if (!pSrc || !pDst || !pTable) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  if (nBitSize <= 0 || nBitSize > static_cast<int>(8 * sizeof(SrcT))) {
    return NPP_LUT_PALETTE_BITSIZE_ERROR;
  }
  dim3 block = BlockConfig<>::blockSize();
  dim3 grid = BlockConfig<>::gridSize(oSizeROI.width, oSizeROI.height);
  paletteC1Kernel<SrcT, TabT, DstT><<<grid, block, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                               oSizeROI.height, pTable, nBitSize);
  return checkCudaKernelError();
}

template <typename SrcT>
NppStatus palette24(const SrcT *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                    const Npp8u *pTable, int nBitSize, cudaStream_t stream) {
  if (!pSrc || !pDst || !pTable) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  if (nBitSize <= 0 || nBitSize > static_cast<int>(8 * sizeof(SrcT))) {
    return NPP_LUT_PALETTE_BITSIZE_ERROR;
  }
  dim3 block = BlockConfig<>::blockSize();
  dim3 grid = BlockConfig<>::gridSize(oSizeROI.width, oSizeROI.height);
  palette24Kernel<SrcT><<<grid, block, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height,
                                                    pTable, nBitSize);
  return checkCudaKernelError();
}

template <typename SrcT, int Channels, bool AlphaOnly>
NppStatus paletteMulti(const SrcT *pSrc, int nSrcStep, SrcT *pDst, int nDstStep, NppiSize oSizeROI,
                       const SrcT *const *pTables, int nBitSize, cudaStream_t stream) {
  constexpr int kTablesUsed = AlphaOnly ? Channels - 1 : Channels;
  if (!pSrc || !pDst || !pTables) {
    return NPP_NULL_POINTER_ERROR;
  }
  for (int c = 0; c < kTablesUsed; ++c) {
    if (!pTables[c]) {
      return NPP_NULL_POINTER_ERROR;
    }
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  if (nBitSize <= 0 || nBitSize > static_cast<int>(8 * sizeof(SrcT))) {
    return NPP_LUT_PALETTE_BITSIZE_ERROR;
  }
  dim3 block = BlockConfig<>::blockSize();
  dim3 grid = BlockConfig<>::gridSize(oSizeROI.width, oSizeROI.height);
  // pTables is a host array of per-channel device pointers (NPP contract);
  // stage the pointer array so the device never dereferences host memory.
  const SrcT **d_tables = nullptr;
  cudaError_t err = cudaMallocAsync(reinterpret_cast<void **>(&d_tables),
                                    (size_t)kTablesUsed * sizeof(const SrcT *), stream);
  if (err != cudaSuccess) {
    return NPP_MEMORY_ALLOCATION_ERR;
  }
  err = cudaMemcpyAsync(d_tables, pTables, (size_t)kTablesUsed * sizeof(const SrcT *),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    cudaFreeAsync(d_tables, stream);
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
  paletteMultiKernel<SrcT, Channels, AlphaOnly><<<grid, block, 0, stream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, d_tables, nBitSize);
  cudaFreeAsync(d_tables, stream);
  return checkCudaKernelError();
}

template <typename T>
NppStatus paletteSwap(const T *pSrc, int nSrcStep, int nAlphaValue, T *pDst, int nDstStep, NppiSize oSizeROI,
                      const T *const *pTables, int nBitSize, cudaStream_t stream) {
  if (!pSrc || !pDst || !pTables || !pTables[0] || !pTables[1] || !pTables[2]) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  if (nBitSize <= 0 || nBitSize > static_cast<int>(8 * sizeof(T))) {
    return NPP_LUT_PALETTE_BITSIZE_ERROR;
  }
  dim3 block = BlockConfig<>::blockSize();
  dim3 grid = BlockConfig<>::gridSize(oSizeROI.width, oSizeROI.height);
  // pTables is a host array of three device pointers (NPP contract).
  const T **d_tables = nullptr;
  cudaError_t err = cudaMallocAsync(reinterpret_cast<void **>(&d_tables), 3 * sizeof(const T *), stream);
  if (err != cudaSuccess) {
    return NPP_MEMORY_ALLOCATION_ERR;
  }
  err = cudaMemcpyAsync(d_tables, pTables, 3 * sizeof(const T *), cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    cudaFreeAsync(d_tables, stream);
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
  paletteSwapKernel<T><<<grid, block, 0, stream>>>(pSrc, nSrcStep, nAlphaValue, pDst, nDstStep, oSizeROI.width,
                                                   oSizeROI.height, d_tables, nBitSize);
  cudaFreeAsync(d_tables, stream);
  return checkCudaKernelError();
}

NppStatus trilinear(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                    const Npp32u *pValues, const Npp8u *const pLevels[3], const int aLevels[3], bool alphaSrc,
                    cudaStream_t stream) {
  if (!pSrc || !pDst || !pValues || !pLevels || !pLevels[0] || !pLevels[1] || !pLevels[2]) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  for (int i = 0; i < 3; ++i) {
    if (aLevels[i] < 2) {
      return NPP_LUT_NUMBER_OF_LEVELS_ERROR;
    }
  }

  Npp8u *device_levels[3] = {nullptr, nullptr, nullptr};
  for (int i = 0; i < 3; ++i) {
    cudaError_t err = cudaMallocAsync(reinterpret_cast<void **>(&device_levels[i]),
                                      static_cast<size_t>(aLevels[i]), stream);
    if (err != cudaSuccess) {
      for (int j = 0; j < i; ++j) {
        (void)cudaFreeAsync(device_levels[j], stream);
      }
      (void)cudaGetLastError();
      return NPP_MEMORY_ALLOCATION_ERR;
    }
    err = cudaMemcpyAsync(device_levels[i], pLevels[i], static_cast<size_t>(aLevels[i]), cudaMemcpyHostToDevice,
                          stream);
    if (err != cudaSuccess) {
      for (int j = 0; j <= i; ++j) {
        (void)cudaFreeAsync(device_levels[j], stream);
      }
      (void)cudaGetLastError();
      return NPP_MEMCPY_ERROR;
    }
  }

  dim3 block = BlockConfig<>::blockSize();
  dim3 grid = BlockConfig<>::gridSize(oSizeROI.width, oSizeROI.height);
  trilinearKernel<<<grid, block, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height,
                                              pValues, device_levels[0], device_levels[1], device_levels[2],
                                              aLevels[0], aLevels[1], aLevels[2], alphaSrc);
  NppStatus status = checkCudaKernelError();
  for (int i = 0; i < 3; ++i) {
    (void)cudaFreeAsync(device_levels[i], stream);
  }
  return status;
}

// ---------------------------------------------------------------------------
// Explicit instantiations used by the generated wrappers
// ---------------------------------------------------------------------------

#define MPP_LUT_INSTANTIATE_C1(SRC)                                                                                 \
  template NppStatus runC1<SRC, LutMode::Point>(const SRC *, int, SRC *, int, NppiSize,                             \
                                                const LutValue<SRC>::type *, const LutValue<SRC>::type *, int,      \
                                                cudaStream_t);                                                      \
  template NppStatus runC1<SRC, LutMode::Linear>(const SRC *, int, SRC *, int, NppiSize,                            \
                                                 const LutValue<SRC>::type *, const LutValue<SRC>::type *, int,     \
                                                 cudaStream_t);                                                     \
  template NppStatus runC1<SRC, LutMode::Cubic>(const SRC *, int, SRC *, int, NppiSize,                             \
                                                const LutValue<SRC>::type *, const LutValue<SRC>::type *, int,      \
                                                cudaStream_t);

#define MPP_LUT_INSTANTIATE_RUN_CHANNELS(SRC, MODE)                                                                 \
  template NppStatus run<SRC, 1, false, MODE>(const SRC *, int, SRC *, int, NppiSize,                              \
                                              const LutValue<SRC>::type *const *, const LutValue<SRC>::type *const *, \
                                              const int *, cudaStream_t);                                           \
  template NppStatus run<SRC, 3, false, MODE>(const SRC *, int, SRC *, int, NppiSize,                              \
                                              const LutValue<SRC>::type *const *, const LutValue<SRC>::type *const *, \
                                              const int *, cudaStream_t);                                           \
  template NppStatus run<SRC, 4, false, MODE>(const SRC *, int, SRC *, int, NppiSize,                              \
                                              const LutValue<SRC>::type *const *, const LutValue<SRC>::type *const *, \
                                              const int *, cudaStream_t);                                           \
  template NppStatus run<SRC, 4, true, MODE>(const SRC *, int, SRC *, int, NppiSize,                               \
                                             const LutValue<SRC>::type *const *, const LutValue<SRC>::type *const *,  \
                                             const int *, cudaStream_t);

#define MPP_LUT_INSTANTIATE_RUN(SRC)                                                                                \
  MPP_LUT_INSTANTIATE_RUN_CHANNELS(SRC, LutMode::Point)                                                             \
  MPP_LUT_INSTANTIATE_RUN_CHANNELS(SRC, LutMode::Linear)                                                            \
  MPP_LUT_INSTANTIATE_RUN_CHANNELS(SRC, LutMode::Cubic)

MPP_LUT_INSTANTIATE_C1(Npp8u)
MPP_LUT_INSTANTIATE_C1(Npp16u)
MPP_LUT_INSTANTIATE_C1(Npp16s)
MPP_LUT_INSTANTIATE_C1(Npp32f)
MPP_LUT_INSTANTIATE_RUN(Npp8u)
MPP_LUT_INSTANTIATE_RUN(Npp16u)
MPP_LUT_INSTANTIATE_RUN(Npp16s)
MPP_LUT_INSTANTIATE_RUN(Npp32f)

template NppStatus paletteC1<Npp8u, Npp8u, Npp8u>(const Npp8u *, int, Npp8u *, int, NppiSize, const Npp8u *, int,
                                                  cudaStream_t);
template NppStatus paletteC1<Npp8u, Npp32u, Npp32u>(const Npp8u *, int, Npp32u *, int, NppiSize, const Npp32u *, int,
                                                    cudaStream_t);
template NppStatus paletteC1<Npp16u, Npp8u, Npp8u>(const Npp16u *, int, Npp8u *, int, NppiSize, const Npp8u *, int,
                                                    cudaStream_t);
template NppStatus paletteC1<Npp16u, Npp16u, Npp16u>(const Npp16u *, int, Npp16u *, int, NppiSize, const Npp16u *,
                                                      int, cudaStream_t);
template NppStatus paletteC1<Npp16u, Npp32u, Npp32u>(const Npp16u *, int, Npp32u *, int, NppiSize, const Npp32u *,
                                                      int, cudaStream_t);
template NppStatus palette24<Npp8u>(const Npp8u *, int, Npp8u *, int, NppiSize, const Npp8u *, int, cudaStream_t);
template NppStatus palette24<Npp16u>(const Npp16u *, int, Npp8u *, int, NppiSize, const Npp8u *, int, cudaStream_t);
template NppStatus paletteMulti<Npp8u, 3, false>(const Npp8u *, int, Npp8u *, int, NppiSize, const Npp8u *const *,
                                                  int, cudaStream_t);
template NppStatus paletteMulti<Npp8u, 4, false>(const Npp8u *, int, Npp8u *, int, NppiSize, const Npp8u *const *,
                                                  int, cudaStream_t);
template NppStatus paletteMulti<Npp8u, 4, true>(const Npp8u *, int, Npp8u *, int, NppiSize, const Npp8u *const *,
                                                 int, cudaStream_t);
template NppStatus paletteMulti<Npp16u, 3, false>(const Npp16u *, int, Npp16u *, int, NppiSize, const Npp16u *const *,
                                                   int, cudaStream_t);
template NppStatus paletteMulti<Npp16u, 4, false>(const Npp16u *, int, Npp16u *, int, NppiSize, const Npp16u *const *,
                                                   int, cudaStream_t);
template NppStatus paletteMulti<Npp16u, 4, true>(const Npp16u *, int, Npp16u *, int, NppiSize, const Npp16u *const *,
                                                  int, cudaStream_t);
template NppStatus paletteSwap<Npp8u>(const Npp8u *, int, int, Npp8u *, int, NppiSize, const Npp8u *const *, int,
                                       cudaStream_t);
template NppStatus paletteSwap<Npp16u>(const Npp16u *, int, int, Npp16u *, int, NppiSize, const Npp16u *const *, int,
                                        cudaStream_t);

} // namespace lut
} // namespace nppi
