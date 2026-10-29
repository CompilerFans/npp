#pragma once

// Two-stage reduction cores shared by the nppi statistics executors
// (introduced by package p0.1 as a pure refactoring; see
// docs/agent_packages/p0.1.md).
//
// Device-only header (.cuh): every translation unit that includes it is
// compiled with nvcc, so __global__ kernels are declared unguarded. Never
// include this header from plain .cpp code; use npp_kernel_utils.h for the
// host-safe utilities.
//
// Semantics are pinned by the differential tests against NVIDIA NPP 13.4;
// every definition here is moved verbatim from
// nppi/nppi_statistics_functions/nppi_statistics_executor.cu.
//
// The norm kernels are templated on the neutral npp_reduce::NormKind instead
// of nppi::statistics::NormType so this header does not depend on a family
// header; the statistics executor converts between the two.

#include "npp.h"

#include <cmath>
#include <cuda_runtime.h>

namespace npp_reduce {

constexpr int kBlock = 256;

// Two-stage reduction over the flattened ROI.  Each thread first accumulates
// several pixels, then the block reduces to thread 0.
template <typename Acc> __device__ Acc blockReduce(Acc val) {
  static __shared__ Acc shared[kBlock];
  shared[threadIdx.x] = val;
  __syncthreads();
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      shared[threadIdx.x] += shared[threadIdx.x + s];
    }
    __syncthreads();
  }
  return shared[0];
}

// ---------------------------------------------------------------------------
// 16-byte vectorized walkers (package opt_reduce).  When rows16Aligned() holds
// for every source the partial kernels walk the ROI in VecPack units: one
// 16-byte load per pack instead of one load per element.  Non-aligned cases
// (padded/odd steps, negative steps) keep the original scalar loops, so both
// paths visit exactly the same elements and only the grouping differs.
// ---------------------------------------------------------------------------

// 16-byte load unit: kVecElems<T>() elements per pack.
template <typename T> struct alignas(16) VecPack {
  T v[16 / sizeof(T)];
};
template <typename T> __host__ __device__ constexpr int kVecElems() { return int(16 / sizeof(T)); }

// True when p is 16-byte aligned and step is a whole multiple of 16 bytes, so
// every row start (p + y*step) is 16-byte aligned.  The step is compared as an
// unsigned value: -16 stays vectorizable (row starts stay aligned and the
// walker reads the same addresses the scalar loop would); any other negative
// step falls back to the scalar loop.
__host__ __device__ __forceinline__ bool rows16Aligned(const void *p, int step) {
  return (reinterpret_cast<size_t>(p) & size_t(15)) == size_t(0) && (static_cast<unsigned>(step) & 15u) == 0u;
}

// Number of 16-byte work units covered by the ROI (host side; used to size the
// partial-stage grid).  A "unit" is always 16 bytes of source data, so the
// per-thread work volume is comparable across dtypes, channel counts and
// walker kinds (pack / 3-pack group / COI sample groups).
__host__ __device__ __forceinline__ long long vec16Units(int width, int height, int channels, int esize) {
  return (((long long)width * channels * esize * height) + 15) >> 4;
}

// Target block count for a full occupancy wave on the current device, so the
// work-per-thread heuristic does not under-fill the GPU on small ROIs (e.g.
// 640x480 16u C1 is 38 blocks at kItemsPerThread=4, i.e. ~17% of a 28-SM
// 600U).  Queried once per process; falls back to a fixed wave when the
// runtime attribute is unavailable.
__host__ __device__ __forceinline__ int reduceMinBlocks() {
#if !defined(__CUDA_ARCH__)
  static const int kMin = [] {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 256;
    int sms = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || sms <= 0) return 256;
    int maxThreadsPerSm = 0;
    if (cudaDeviceGetAttribute(&maxThreadsPerSm, cudaDevAttrMaxThreadsPerMultiProcessor, dev) != cudaSuccess ||
        maxThreadsPerSm <= 0)
      maxThreadsPerSm = 2048;
    int blocksPerSm = maxThreadsPerSm / kBlock;
    if (blocksPerSm < 1) blocksPerSm = 1;
    return sms * blocksPerSm;
  }();
  return kMin;
#else
  return 256;
#endif
}

// Launch geometry for the partial stage of the two-stage reduction: aim for
// kItemsPerThread 16-byte units per thread, floored at one full occupancy
// wave (see reduceMinBlocks) and capped so that the block count never exceeds
// the classic one-item-per-thread count that every *GetBufferHostSize helper
// reserves (numBlocks(area), or a safe multiple), and never exceeds
// kMaxReduceBlocks.  One unit is 16 bytes of source for the vectorized path
// or one pixel for the scalar fallback; the cap keeps every existing
// buffer-size helper sufficient because a 16-byte unit always covers at least
// one pixel's worth of data units (worst case 32fc/64f C4: 32 bytes per
// pixel = 2 units, i.e. blocks <= area/512 <= numBlocks(area)).  The
// occupancy floor can never exceed that cap either, so buffer sufficiency is
// unaffected; idle threads in over-provisioned blocks simply skip the
// grid-stride loop.
constexpr int kItemsPerThread = 4;
constexpr int kMaxReduceBlocks = 8192;
__host__ __device__ __forceinline__ int reduceGridBlocks(long long workUnits, int classicBlocks) {
  const long long perWave = (long long)kBlock * kItemsPerThread;
  long long want = (workUnits + perWave - 1) / perWave;
  const long long floor_ = reduceMinBlocks();
  if (want < floor_) want = floor_;
  if (want < 1) want = 1;
  if (want > kMaxReduceBlocks) want = kMaxReduceBlocks;
  if (want > classicBlocks) want = classicBlocks;
  return (int)want;
}

// Walk whole-pixel 16-byte packs of a Channels-packed ROI (Channels 1/2/4;
// kVecElems is a multiple of Channels for every statistics dtype).  onPack is
// called once per pack with the pixel index of the pack's first element and
// the row; element k of the pack is channel k % Channels of pixel x0 + k /
// Channels (compile-time constants inside an unrolled callback).  Elements
// past the last full pack of each row are delivered to onRem one by one with a
// runtime channel index.
template <int Channels, typename D, typename PackFn, typename RemFn>
__device__ __forceinline__ void walkVecPacks(const D *pSrc, int nSrcStep, int width, int height, PackFn onPack,
                                             RemFn onRem) {
  constexpr int V = kVecElems<D>();
  static_assert(V % Channels == 0, "pack must hold whole pixels");
  const int rowElems = width * Channels;
  const int rowPacks = rowElems / V;
  const int remElems = rowElems - rowPacks * V;
  const char *base = reinterpret_cast<const char *>(pSrc);
  const unsigned stride = gridDim.x * blockDim.x;
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned totalPacks = (unsigned)rowPacks * (unsigned)height;
  for (unsigned s = tid; s < totalPacks; s += stride) {
    const int y = int(s / (unsigned)rowPacks);
    const int p = int(s - (unsigned)y * (unsigned)rowPacks);
    const D *row = reinterpret_cast<const D *>(base + (long long)y * nSrcStep);
    onPack(*reinterpret_cast<const VecPack<D> *>(row + (size_t)p * V), p * (V / Channels), y);
  }
  if (remElems > 0) {
    const unsigned totalRem = (unsigned)remElems * (unsigned)height;
    for (unsigned s = tid; s < totalRem; s += stride) {
      const int y = int(s / (unsigned)remElems);
      const int e = int(s - (unsigned)y * (unsigned)remElems) + rowPacks * V;
      const D *row = reinterpret_cast<const D *>(base + (long long)y * nSrcStep);
      onRem(e % Channels, row[e], e / Channels, y);
    }
  }
}

// Two-source variant of walkVecPacks (both sources must satisfy
// rows16Aligned).  onPack2 receives both packs; onRem2 both remainder values.
template <int Channels, typename D, typename PackFn, typename RemFn>
__device__ __forceinline__ void walkVecPacks2(const D *pSrc1, int nSrc1Step, const D *pSrc2, int nSrc2Step, int width,
                                              int height, PackFn onPack2, RemFn onRem2) {
  constexpr int V = kVecElems<D>();
  static_assert(V % Channels == 0, "pack must hold whole pixels");
  const int rowElems = width * Channels;
  const int rowPacks = rowElems / V;
  const int remElems = rowElems - rowPacks * V;
  const char *base1 = reinterpret_cast<const char *>(pSrc1);
  const char *base2 = reinterpret_cast<const char *>(pSrc2);
  const unsigned stride = gridDim.x * blockDim.x;
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned totalPacks = (unsigned)rowPacks * (unsigned)height;
  for (unsigned s = tid; s < totalPacks; s += stride) {
    const int y = int(s / (unsigned)rowPacks);
    const int p = int(s - (unsigned)y * (unsigned)rowPacks);
    const D *row1 = reinterpret_cast<const D *>(base1 + (long long)y * nSrc1Step);
    const D *row2 = reinterpret_cast<const D *>(base2 + (long long)y * nSrc2Step);
    onPack2(*reinterpret_cast<const VecPack<D> *>(row1 + (size_t)p * V),
            *reinterpret_cast<const VecPack<D> *>(row2 + (size_t)p * V), p * (V / Channels), y);
  }
  if (remElems > 0) {
    const unsigned totalRem = (unsigned)remElems * (unsigned)height;
    for (unsigned s = tid; s < totalRem; s += stride) {
      const int y = int(s / (unsigned)remElems);
      const int e = int(s - (unsigned)y * (unsigned)remElems) + rowPacks * V;
      const D *row1 = reinterpret_cast<const D *>(base1 + (long long)y * nSrc1Step);
      const D *row2 = reinterpret_cast<const D *>(base2 + (long long)y * nSrc2Step);
      onRem2(e % Channels, row1[e], row2[e], e / Channels, y);
    }
  }
}

// Walk a three-channel ROI in 48-byte groups (three consecutive packs = V
// pixels, so element j of the group is channel j % 3 of pixel x0 + j / 3 and
// stays a compile-time constant inside an unrolled callback).  Leftover
// elements (partial groups and row tails) go to onRem one by one.
template <typename D, typename GroupFn, typename RemFn>
__device__ __forceinline__ void walkVecGroups3(const D *pSrc, int nSrcStep, int width, int height, GroupFn onGroup,
                                               RemFn onRem) {
  constexpr int V = kVecElems<D>();
  const int rowElems = width * 3;
  const int rowGroups = rowElems / (3 * V);
  const int remElems = rowElems - rowGroups * 3 * V;
  const char *base = reinterpret_cast<const char *>(pSrc);
  const unsigned stride = gridDim.x * blockDim.x;
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned totalGroups = (unsigned)rowGroups * (unsigned)height;
  for (unsigned s = tid; s < totalGroups; s += stride) {
    const int y = int(s / (unsigned)rowGroups);
    const int g = int(s - (unsigned)y * (unsigned)rowGroups);
    const D *row = reinterpret_cast<const D *>(base + (long long)y * nSrcStep) + (size_t)g * 3 * V;
    const VecPack<D> *pv = reinterpret_cast<const VecPack<D> *>(row);
    onGroup(pv[0], pv[1], pv[2], g * V, y);
  }
  if (remElems > 0) {
    const unsigned totalRem = (unsigned)remElems * (unsigned)height;
    for (unsigned s = tid; s < totalRem; s += stride) {
      const int y = int(s / (unsigned)remElems);
      const int e = int(s - (unsigned)y * (unsigned)remElems) + rowGroups * 3 * V;
      const D *row = reinterpret_cast<const D *>(base + (long long)y * nSrcStep);
      onRem(e % 3, row[e], e / 3, y);
    }
  }
}

// Two-source variant of walkVecGroups3 (both sources must satisfy
// rows16Aligned).
template <typename D, typename GroupFn, typename RemFn>
__device__ __forceinline__ void walkVecGroups3_2(const D *pSrc1, int nSrc1Step, const D *pSrc2, int nSrc2Step, int width,
                                                 int height, GroupFn onGroup2, RemFn onRem2) {
  constexpr int V = kVecElems<D>();
  const int rowElems = width * 3;
  const int rowGroups = rowElems / (3 * V);
  const int remElems = rowElems - rowGroups * 3 * V;
  const char *base1 = reinterpret_cast<const char *>(pSrc1);
  const char *base2 = reinterpret_cast<const char *>(pSrc2);
  const unsigned stride = gridDim.x * blockDim.x;
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned totalGroups = (unsigned)rowGroups * (unsigned)height;
  for (unsigned s = tid; s < totalGroups; s += stride) {
    const int y = int(s / (unsigned)rowGroups);
    const int g = int(s - (unsigned)y * (unsigned)rowGroups);
    const D *row1 = reinterpret_cast<const D *>(base1 + (long long)y * nSrc1Step) + (size_t)g * 3 * V;
    const D *row2 = reinterpret_cast<const D *>(base2 + (long long)y * nSrc2Step) + (size_t)g * 3 * V;
    const VecPack<D> *pv1 = reinterpret_cast<const VecPack<D> *>(row1);
    const VecPack<D> *pv2 = reinterpret_cast<const VecPack<D> *>(row2);
    onGroup2(pv1[0], pv1[1], pv1[2], pv2[0], pv2[1], pv2[2], g * V, y);
  }
  if (remElems > 0) {
    const unsigned totalRem = (unsigned)remElems * (unsigned)height;
    for (unsigned s = tid; s < totalRem; s += stride) {
      const int y = int(s / (unsigned)remElems);
      const int e = int(s - (unsigned)y * (unsigned)remElems) + rowGroups * 3 * V;
      const D *row1 = reinterpret_cast<const D *>(base1 + (long long)y * nSrc1Step);
      const D *row2 = reinterpret_cast<const D *>(base2 + (long long)y * nSrc2Step);
      onRem2(e % 3, row1[e], row2[e], e / 3, y);
    }
  }
}

// Walk one channel (Chan, 0-based; a template parameter so the sample
// extraction stays register-static) of a Channels-packed ROI in 16-byte
// units.  onSample is called for every sample of channel Chan with its pixel
// index and row; leftovers go through a per-element loop with a runtime
// channel test.  Channels==3 walks three-pack groups (3*V elements = V
// pixels); Channels 1/2/4 walk whole-pixel packs.
template <int Channels, int Chan, typename D, typename SampleFn>
__device__ __forceinline__ void walkVecCoi(const D *pSrc, int nSrcStep, int width, int height, SampleFn onSample) {
  constexpr int V = kVecElems<D>();
  static_assert(Chan >= 0 && Chan < Channels, "channel out of range");
  const int rowElems = width * Channels;
  const int rowFull = (Channels == 3) ? (rowElems / (3 * V)) * 3 * V : (rowElems / V) * V;
  const int fullUnits = (Channels == 3) ? (rowFull / (3 * V)) : (rowFull / V);  // groups or packs per row
  const int remElems = rowElems - rowFull;
  const char *base = reinterpret_cast<const char *>(pSrc);
  const unsigned stride = gridDim.x * blockDim.x;
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned totalUnits = (unsigned)fullUnits * (unsigned)height;
  for (unsigned s = tid; s < totalUnits; s += stride) {
    const int y = int(s / (unsigned)fullUnits);
    const int u = int(s - (unsigned)y * (unsigned)fullUnits);
    const D *row = reinterpret_cast<const D *>(base + (long long)y * nSrcStep);
    if constexpr (Channels == 3) {
      const VecPack<D> *pv = reinterpret_cast<const VecPack<D> *>(row + (size_t)u * 3 * V);
      // Pixel m of the group is element 3*m + Chan (pack 3*m/V, lane 3*m%V).
#pragma unroll
      for (int m = 0; m < V; ++m) {
        const int e = 3 * m + Chan;
        const int l = e & (V - 1);
        if (e < V) {
          onSample(pv[0].v[l], u * V + m, y);
        } else if (e < 2 * V) {
          onSample(pv[1].v[l], u * V + m, y);
        } else {
          onSample(pv[2].v[l], u * V + m, y);
        }
      }
    } else {
      const VecPack<D> p = *reinterpret_cast<const VecPack<D> *>(row + (size_t)u * V);
#pragma unroll
      for (int m = 0; m < V / Channels; ++m) onSample(p.v[m * Channels + Chan], u * (V / Channels) + m, y);
    }
  }
  if (remElems > 0) {
    const unsigned totalRem = (unsigned)remElems * (unsigned)height;
    for (unsigned s = tid; s < totalRem; s += stride) {
      const int y = int(s / (unsigned)remElems);
      const int e = int(s - (unsigned)y * (unsigned)remElems) + rowFull;
      const D *row = reinterpret_cast<const D *>(base + (long long)y * nSrcStep);
      if (e % Channels == Chan) onSample(row[e], e / Channels, y);
    }
  }
}

// Walk the ROI as a flat sequence of 16-byte element packs with no channel
// semantics (every element folds identically, e.g. AverageError).  Leftover
// row-tail elements go to onRemElem one by one.
template <typename D, typename PackFn, typename RemFn>
__device__ __forceinline__ void walkVecFlat(const D *pSrc, int nSrcStep, int rowElems, int height, PackFn onPack,
                                            RemFn onRemElem) {
  constexpr int V = kVecElems<D>();
  const int rowPacks = rowElems / V;
  const int remElems = rowElems - rowPacks * V;
  const char *base = reinterpret_cast<const char *>(pSrc);
  const unsigned stride = gridDim.x * blockDim.x;
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned totalPacks = (unsigned)rowPacks * (unsigned)height;
  for (unsigned s = tid; s < totalPacks; s += stride) {
    const int y = int(s / (unsigned)rowPacks);
    const int p = int(s - (unsigned)y * (unsigned)rowPacks);
    const D *row = reinterpret_cast<const D *>(base + (long long)y * nSrcStep);
    onPack(*reinterpret_cast<const VecPack<D> *>(row + (size_t)p * V));
  }
  if (remElems > 0) {
    const unsigned totalRem = (unsigned)remElems * (unsigned)height;
    for (unsigned s = tid; s < totalRem; s += stride) {
      const int y = int(s / (unsigned)remElems);
      const int e = int(s - (unsigned)y * (unsigned)remElems) + rowPacks * V;
      const D *row = reinterpret_cast<const D *>(base + (long long)y * nSrcStep);
      onRemElem(row[e]);
    }
  }
}

// Two-source variant of walkVecFlat (both sources must satisfy rows16Aligned).
template <typename D, typename PackFn, typename RemFn>
__device__ __forceinline__ void walkVecFlat2(const D *pSrc1, int nSrc1Step, const D *pSrc2, int nSrc2Step, int rowElems,
                                             int height, PackFn onPack2, RemFn onRemElem2) {
  constexpr int V = kVecElems<D>();
  const int rowPacks = rowElems / V;
  const int remElems = rowElems - rowPacks * V;
  const char *base1 = reinterpret_cast<const char *>(pSrc1);
  const char *base2 = reinterpret_cast<const char *>(pSrc2);
  const unsigned stride = gridDim.x * blockDim.x;
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned totalPacks = (unsigned)rowPacks * (unsigned)height;
  for (unsigned s = tid; s < totalPacks; s += stride) {
    const int y = int(s / (unsigned)rowPacks);
    const int p = int(s - (unsigned)y * (unsigned)rowPacks);
    const D *row1 = reinterpret_cast<const D *>(base1 + (long long)y * nSrc1Step);
    const D *row2 = reinterpret_cast<const D *>(base2 + (long long)y * nSrc2Step);
    onPack2(*reinterpret_cast<const VecPack<D> *>(row1 + (size_t)p * V),
            *reinterpret_cast<const VecPack<D> *>(row2 + (size_t)p * V));
  }
  if (remElems > 0) {
    const unsigned totalRem = (unsigned)remElems * (unsigned)height;
    for (unsigned s = tid; s < totalRem; s += stride) {
      const int y = int(s / (unsigned)remElems);
      const int e = int(s - (unsigned)y * (unsigned)remElems) + rowPacks * V;
      const D *row1 = reinterpret_cast<const D *>(base1 + (long long)y * nSrc1Step);
      const D *row2 = reinterpret_cast<const D *>(base2 + (long long)y * nSrc2Step);
      onRemElem2(row1[e], row2[e]);
    }
  }
}

// ---------------------------------------------------------------------------
// Sum / Mean (plain multi-channel).  Writes nch partials per block.
// ---------------------------------------------------------------------------

// vecPath selects the 16-byte pack walk (host-checked rows16Aligned) over the
// original scalar grid-stride loop; both visit the same elements.
template <typename D, int Channels, bool AlphaPreserve, typename Acc>
__global__ void reducePartialKernel(const D *pSrc, int nSrcStep, int width, int height, Acc *pPartial, int nch,
                                    int vecPath) {
  Acc local[4] = {0, 0, 0, 0};
  const int totalPixels = width * height;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr int readCh = AlphaPreserve ? Channels - 1 : Channels;

  if (vecPath) {
    if constexpr (Channels == 3) {
      auto groupFn = [&](const VecPack<D> &p0, const VecPack<D> &p1, const VecPack<D> &p2, int x0, int y) {
#pragma unroll
        for (int c = 0; c < 3; ++c) {
          Acc t = 0;
#pragma unroll
          for (int m = 0; m < kVecElems<D>(); ++m) {
            const int e = 3 * m + c;  // element index within the 3-pack group
            const int l = e & (kVecElems<D>() - 1);
            t += static_cast<Acc>((e < kVecElems<D>()) ? p0.v[l] : (e < 2 * kVecElems<D>()) ? p1.v[l] : p2.v[l]);
          }
          local[c] += t;
        }
      };
      auto remFn = [&](int c, D v, int x, int y) {
        if (c == 0) local[0] += static_cast<Acc>(v);
        else if (c == 1) local[1] += static_cast<Acc>(v);
        else local[2] += static_cast<Acc>(v);
      };
      walkVecGroups3(pSrc, nSrcStep, width, height, groupFn, remFn);
    } else {
      auto packFn = [&](const VecPack<D> &p, int x0, int y) {
#pragma unroll
        for (int c = 0; c < readCh; ++c) {
          Acc t = 0;
#pragma unroll
          for (int m = 0; m < kVecElems<D>() / Channels; ++m) t += static_cast<Acc>(p.v[m * Channels + c]);
          local[c] += t;
        }
      };
      auto remFn = [&](int c, D v, int x, int y) {
        if (c == 0) local[0] += static_cast<Acc>(v);
        else if (c == 1) local[1] += static_cast<Acc>(v);
        else if (c == 2) local[2] += static_cast<Acc>(v);
        else if (!AlphaPreserve) local[3] += static_cast<Acc>(v);
      };
      walkVecPacks<Channels>(pSrc, nSrcStep, width, height, packFn, remFn);
    }
  } else {
    for (int i = tid; i < totalPixels; i += gridDim.x * blockDim.x) {
      const int y = i / width;
      const int x = i % width;
      const D *row = reinterpret_cast<const D *>(reinterpret_cast<const char *>(pSrc) + y * nSrcStep);
      for (int c = 0; c < readCh; ++c) {
        local[c] += static_cast<Acc>(row[x * Channels + c]);
      }
    }
  }

  for (int c = 0; c < readCh; ++c) {
    const Acc r = blockReduce(local[c]);
    if (threadIdx.x == 0) {
      pPartial[(size_t)c * gridDim.x + blockIdx.x] = r;
    }
  }
}

// Re-add the partial buckets (column `c` has its entries strided by gridDim.x).
template <typename Acc, bool Divide>
__global__ void finalizeOutKernel(const Acc *pPartial, int nblocks, int nch, double total, Acc *pOut) {
  __shared__ Acc shared[kBlock];
  for (int c = 0; c < nch; ++c) {
    Acc acc = 0;
    for (int i = threadIdx.x; i < nblocks; i += blockDim.x) {
      acc += pPartial[(size_t)c * nblocks + i];
    }
    shared[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      Acc v = shared[0];
      pOut[c] = Divide ? static_cast<Acc>(static_cast<double>(v) / total) : v;
    }
  }
}

// ---------------------------------------------------------------------------
// Masked Mean (single output channel)
// ---------------------------------------------------------------------------

template <typename D, int Channels>
__global__ void maskedSumKernel(const D *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep, int width, int height,
                                int nCOI, double *pSum, double *pCount) {
  const int totalPixels = width * height;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  double lsum = 0.0, lcnt = 0.0;
  for (int i = tid; i < totalPixels; i += gridDim.x * blockDim.x) {
    const int y = i / width;
    const int x = i % width;
    const Npp8u *maskRow = reinterpret_cast<const Npp8u *>(reinterpret_cast<const char *>(pMask) + y * nMaskStep);
    if (maskRow[x] != 0) {
      const D *row = reinterpret_cast<const D *>(reinterpret_cast<const char *>(pSrc) + y * nSrcStep);
      lsum += static_cast<double>(row[x * Channels + nCOI]);
      lcnt += 1.0;
    }
  }
  double rs = blockReduce(lsum);
  double rc = blockReduce(lcnt);
  if (threadIdx.x == 0) {
    pSum[blockIdx.x] = rs;
    pCount[blockIdx.x] = rc;
  }
}

__global__ void finalizeMaskedKernel(const double *pSum, const double *pCount, int nblocks, double *pOut) {
  __shared__ double sums[kBlock];
  __shared__ double cnts[kBlock];
  double lsum = 0.0, lcnt = 0.0;
  for (int i = threadIdx.x; i < nblocks; i += blockDim.x) {
    lsum += pSum[i];
    lcnt += pCount[i];
  }
  sums[threadIdx.x] = lsum;
  cnts[threadIdx.x] = lcnt;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sums[threadIdx.x] += sums[threadIdx.x + s];
      cnts[threadIdx.x] += cnts[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    double d = cnts[0];
    pOut[0] = (d > 0.0) ? sums[0] / d : 0.0;
  }
}

// ---------------------------------------------------------------------------
// Mean + StdDev (single output channel)
// ---------------------------------------------------------------------------

// Chan is the 0-based source channel as a template parameter (package
// opt_reduce) so the vectorized path extracts samples with compile-time
// constant indices; runMeanStdDev launches Chan=0, runMeanStdDevCoi
// dispatches Chan = nCOI-1 in {0,1,2}.  vecPath selects the 16-byte walk.
template <typename D, int Channels, int Chan>
__global__ void msSumKernel(const D *pSrc, int nSrcStep, int width, int height, double *pSum, double *pSumSq,
                            int vecPath) {
  const int totalPixels = width * height;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  double lsum = 0.0, lsq = 0.0;
  if (vecPath) {
    auto sampleFn = [&](D s, int x, int y) {
      const double v = static_cast<double>(s);
      lsum += v;
      lsq += v * v;
    };
    walkVecCoi<Channels, Chan>(pSrc, nSrcStep, width, height, sampleFn);
  } else {
    for (int i = tid; i < totalPixels; i += gridDim.x * blockDim.x) {
      const int y = i / width;
      const int x = i % width;
      const D *row = reinterpret_cast<const D *>(reinterpret_cast<const char *>(pSrc) + y * nSrcStep);
      const double v = static_cast<double>(row[x * Channels + Chan]);
      lsum += v;
      lsq += v * v;
    }
  }
  double rs = blockReduce(lsum);
  double rsq = blockReduce(lsq);
  if (threadIdx.x == 0) {
    pSum[blockIdx.x] = rs;
    pSumSq[blockIdx.x] = rsq;
  }
}

template <typename D, int Channels>
__global__ void msMaskedSumKernel(const D *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep, int width,
                                  int height, int nCOI, double *pSum, double *pSumSq, double *pCount) {
  const int totalPixels = width * height;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  double lsum = 0.0, lsq = 0.0, lcnt = 0.0;
  for (int i = tid; i < totalPixels; i += gridDim.x * blockDim.x) {
    const int y = i / width;
    const int x = i % width;
    const Npp8u *maskRow = reinterpret_cast<const Npp8u *>(reinterpret_cast<const char *>(pMask) + y * nMaskStep);
    if (maskRow[x] != 0) {
      const D *row = reinterpret_cast<const D *>(reinterpret_cast<const char *>(pSrc) + y * nSrcStep);
      const double v = static_cast<double>(row[x * Channels + nCOI]);
      lsum += v;
      lsq += v * v;
      lcnt += 1.0;
    }
  }
  double rs = blockReduce(lsum);
  double rsq = blockReduce(lsq);
  double rc = blockReduce(lcnt);
  if (threadIdx.x == 0) {
    pSum[blockIdx.x] = rs;
    pSumSq[blockIdx.x] = rsq;
    pCount[blockIdx.x] = rc;
  }
}

__device__ __forceinline__ double stdFromSumSq(double totalSum, double totalSq, double N) {
  if (N <= 0.0) return 0.0;
  const double mean = totalSum / N;
  const double var = (totalSq / N) - mean * mean;
  return sqrt(fmax(var, 0.0));
}

__global__ void finalizeMSKernel(const double *pSum, const double *pSumSq, int nblocks, double totalPixels,
                                 double *pMean, double *pStdDev) {
  __shared__ double sums[kBlock];
  __shared__ double sqs[kBlock];
  double lsum = 0.0, lsq = 0.0;
  for (int i = threadIdx.x; i < nblocks; i += blockDim.x) {
    lsum += pSum[i];
    lsq += pSumSq[i];
  }
  sums[threadIdx.x] = lsum;
  sqs[threadIdx.x] = lsq;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sums[threadIdx.x] += sums[threadIdx.x + s];
      sqs[threadIdx.x] += sqs[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const double N = totalPixels;
    pMean[0] = (N > 0.0) ? sums[0] / N : 0.0;
    pStdDev[0] = (N > 0.0) ? stdFromSumSq(sums[0], sqs[0], N) : 0.0;
  }
}

__global__ void finalizeMSMaskedKernel(const double *pSum, const double *pSumSq, const double *pCount, int nblocks,
                                       double *pMean, double *pStdDev) {
  __shared__ double sums[kBlock];
  __shared__ double sqs[kBlock];
  __shared__ double cnts[kBlock];
  double lsum = 0.0, lsq = 0.0, lcnt = 0.0;
  for (int i = threadIdx.x; i < nblocks; i += blockDim.x) {
    lsum += pSum[i];
    lsq += pSumSq[i];
    lcnt += pCount[i];
  }
  sums[threadIdx.x] = lsum;
  sqs[threadIdx.x] = lsq;
  cnts[threadIdx.x] = lcnt;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sums[threadIdx.x] += sums[threadIdx.x + s];
      sqs[threadIdx.x] += sqs[threadIdx.x + s];
      cnts[threadIdx.x] += cnts[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const double N = cnts[0];
    pMean[0] = (N > 0.0) ? sums[0] / N : 0.0;
    pStdDev[0] = (N > 0.0) ? stdFromSumSq(sums[0], sqs[0], N) : 0.0;
  }
}

inline int numBlocks(int area) { return (area + kBlock - 1) / kBlock; }

// ---------------------------------------------------------------------------
// Norm / NormDiff (L1 / L2 / Inf)
// ---------------------------------------------------------------------------

// Neutral norm-kind tag (see the header comment).
enum class NormKind { L1, L2, Inf };

// Contribution of one raw (possibly diff) sample.  L2 keeps the square (sqrt
// is applied at finalize); L1 / Inf use |raw|.
template <NormKind NT>
__device__ __forceinline__ double normContribution(double raw) {
  if constexpr (NT == NormKind::L2) return raw * raw;
  return fabs(raw);
}

// Fold raw into acc: L1/L2 accumulate, Inf keeps the running maximum.
template <NormKind NT>
__device__ __forceinline__ void normAcc(double &acc, double raw) {
  double c = normContribution<NT>(raw);
  if constexpr (NT == NormKind::Inf) acc = fmax(acc, c);
  else acc += c;
}

// Block-reduce a per-thread norm partial (sum for L1/L2, max for Inf).
template <bool IsInf> __device__ double normBlockReduce(double val) {
  static __shared__ double shared[kBlock];
  shared[threadIdx.x] = val;
  __syncthreads();
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      if constexpr (IsInf) shared[threadIdx.x] = fmax(shared[threadIdx.x], shared[threadIdx.x + s]);
      else shared[threadIdx.x] += shared[threadIdx.x + s];
    }
    __syncthreads();
  }
  return shared[0];
}

// Multi-channel plain norm (Diff=false) or norm-diff (Diff=true).  vecPath
// selects the 16-byte walk (requires both sources aligned for Diff).
template <NormKind NT, typename D, int Channels, bool AlphaPreserve, bool Diff>
__global__ void normPartialKernel(const D *pSrc, int nSrcStep, const D *pSrc2, int nSrc2Step, int width, int height,
                                  double *pPartial, int nch, int vecPath) {
  constexpr int readCh = AlphaPreserve ? Channels - 1 : Channels;
  constexpr bool IsInf = (NT == NormKind::Inf);
  double local[4] = {0.0, 0.0, 0.0, 0.0};
  const int totalPixels = width * height;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (vecPath) {
    if constexpr (Diff) {
      if constexpr (Channels == 3) {
        auto groupFn = [&](const VecPack<D> &a0, const VecPack<D> &a1, const VecPack<D> &a2, const VecPack<D> &b0,
                           const VecPack<D> &b1, const VecPack<D> &b2, int x0, int y) {
#pragma unroll
          for (int c = 0; c < 3; ++c) {
#pragma unroll
            for (int m = 0; m < kVecElems<D>(); ++m) {
              const int e = 3 * m + c;
              const int l = e & (kVecElems<D>() - 1);
              const D va = (e < kVecElems<D>()) ? a0.v[l] : (e < 2 * kVecElems<D>()) ? a1.v[l] : a2.v[l];
              const D vb = (e < kVecElems<D>()) ? b0.v[l] : (e < 2 * kVecElems<D>()) ? b1.v[l] : b2.v[l];
              normAcc<NT>(local[c], static_cast<double>(va) - static_cast<double>(vb));
            }
          }
        };
        auto remFn = [&](int c, D v1, D v2, int x, int y) {
          if (c == 0) normAcc<NT>(local[0], static_cast<double>(v1) - static_cast<double>(v2));
          else if (c == 1) normAcc<NT>(local[1], static_cast<double>(v1) - static_cast<double>(v2));
          else normAcc<NT>(local[2], static_cast<double>(v1) - static_cast<double>(v2));
        };
        walkVecGroups3_2(pSrc, nSrcStep, pSrc2, nSrc2Step, width, height, groupFn, remFn);
      } else {
        auto packFn = [&](const VecPack<D> &pa, const VecPack<D> &pb, int x0, int y) {
#pragma unroll
          for (int c = 0; c < readCh; ++c) {
#pragma unroll
            for (int m = 0; m < kVecElems<D>() / Channels; ++m) {
              const double v = static_cast<double>(pa.v[m * Channels + c]) - static_cast<double>(pb.v[m * Channels + c]);
              normAcc<NT>(local[c], v);
            }
          }
        };
        auto remFn = [&](int c, D v1, D v2, int x, int y) {
          const double v = static_cast<double>(v1) - static_cast<double>(v2);
          if (c == 0) normAcc<NT>(local[0], v);
          else if (c == 1) normAcc<NT>(local[1], v);
          else if (c == 2) normAcc<NT>(local[2], v);
          else if (!AlphaPreserve) normAcc<NT>(local[3], v);
        };
        walkVecPacks2<Channels>(pSrc, nSrcStep, pSrc2, nSrc2Step, width, height, packFn, remFn);
      }
    } else {
      if constexpr (Channels == 3) {
        auto groupFn = [&](const VecPack<D> &p0, const VecPack<D> &p1, const VecPack<D> &p2, int x0, int y) {
#pragma unroll
          for (int c = 0; c < 3; ++c) {
#pragma unroll
            for (int m = 0; m < kVecElems<D>(); ++m) {
              const int e = 3 * m + c;
              const int l = e & (kVecElems<D>() - 1);
              const D v = (e < kVecElems<D>()) ? p0.v[l] : (e < 2 * kVecElems<D>()) ? p1.v[l] : p2.v[l];
              normAcc<NT>(local[c], static_cast<double>(v));
            }
          }
        };
        auto remFn = [&](int c, D v, int x, int y) {
          if (c == 0) normAcc<NT>(local[0], static_cast<double>(v));
          else if (c == 1) normAcc<NT>(local[1], static_cast<double>(v));
          else normAcc<NT>(local[2], static_cast<double>(v));
        };
        walkVecGroups3(pSrc, nSrcStep, width, height, groupFn, remFn);
      } else {
        auto packFn = [&](const VecPack<D> &p, int x0, int y) {
#pragma unroll
          for (int c = 0; c < readCh; ++c) {
#pragma unroll
            for (int m = 0; m < kVecElems<D>() / Channels; ++m)
              normAcc<NT>(local[c], static_cast<double>(p.v[m * Channels + c]));
          }
        };
        auto remFn = [&](int c, D v, int x, int y) {
          if (c == 0) normAcc<NT>(local[0], static_cast<double>(v));
          else if (c == 1) normAcc<NT>(local[1], static_cast<double>(v));
          else if (c == 2) normAcc<NT>(local[2], static_cast<double>(v));
          else if (!AlphaPreserve) normAcc<NT>(local[3], static_cast<double>(v));
        };
        walkVecPacks<Channels>(pSrc, nSrcStep, width, height, packFn, remFn);
      }
    }
  } else {
    for (int i = tid; i < totalPixels; i += gridDim.x * blockDim.x) {
      const int y = i / width;
      const int x = i % width;
      const D *row = reinterpret_cast<const D *>(reinterpret_cast<const char *>(pSrc) + y * nSrcStep);
      const D *row2 = Diff ? reinterpret_cast<const D *>(reinterpret_cast<const char *>(pSrc2) + y * nSrc2Step) : nullptr;
      for (int c = 0; c < readCh; ++c) {
        double v = static_cast<double>(row[x * Channels + c]);
        if constexpr (Diff) v -= static_cast<double>(row2[x * Channels + c]);
        normAcc<NT>(local[c], v);
      }
    }
  }
  for (int c = 0; c < readCh; ++c) {
    const double r = normBlockReduce<IsInf>(local[c]);
    if (threadIdx.x == 0) pPartial[(size_t)c * gridDim.x + blockIdx.x] = r;
  }
}

// Finalize per-channel block partials into the output norm (L2 applies sqrt).
template <NormKind NT>
__global__ void normFinalizeKernel(const double *pPartial, int nblocks, int nch, Npp64f *pOut) {
  __shared__ double shared[kBlock];
  constexpr bool IsInf = (NT == NormKind::Inf);
  for (int c = 0; c < nch; ++c) {
    double acc = 0.0;
    for (int i = threadIdx.x; i < nblocks; i += blockDim.x) {
      const double v = pPartial[(size_t)c * nblocks + i];
      if constexpr (IsInf) acc = fmax(acc, v);
      else acc += v;
    }
    shared[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) {
        if constexpr (IsInf) shared[threadIdx.x] = fmax(shared[threadIdx.x], shared[threadIdx.x + s]);
        else shared[threadIdx.x] += shared[threadIdx.x + s];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      const double v = shared[0];
      pOut[c] = (NT == NormKind::L2) ? sqrt(v) : v;
    }
    __syncthreads();
  }
}

// Masked norm / norm-diff over a single channel (nCOI is the 0-based channel).
template <NormKind NT, typename D, int Channels, bool Diff>
__global__ void normMaskedKernel(const D *pSrc, int nSrcStep, const D *pSrc2, int nSrc2Step, const Npp8u *pMask,
                                  int nMaskStep, int width, int height, int nCOI, double *pPartial) {
  constexpr bool IsInf = (NT == NormKind::Inf);
  double local = 0.0;
  const int totalPixels = width * height;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = tid; i < totalPixels; i += gridDim.x * blockDim.x) {
    const int y = i / width;
    const int x = i % width;
    const Npp8u *maskRow = reinterpret_cast<const Npp8u *>(reinterpret_cast<const char *>(pMask) + y * nMaskStep);
    if (maskRow[x] == 0) continue;
    const D *row = reinterpret_cast<const D *>(reinterpret_cast<const char *>(pSrc) + y * nSrcStep);
    double v = static_cast<double>(row[x * Channels + nCOI]);
    if constexpr (Diff) {
      const D *row2 = reinterpret_cast<const D *>(reinterpret_cast<const char *>(pSrc2) + y * nSrc2Step);
      v -= static_cast<double>(row2[x * Channels + nCOI]);
    }
    normAcc<NT>(local, v);
  }
  const double r = normBlockReduce<IsInf>(local);
  if (threadIdx.x == 0) pPartial[blockIdx.x] = r;
}

template <NormKind NT>
__global__ void normMaskedFinalizeKernel(const double *pPartial, int nblocks, double *pOut) {
  __shared__ double shared[kBlock];
  constexpr bool IsInf = (NT == NormKind::Inf);
  double acc = 0.0;
  for (int i = threadIdx.x; i < nblocks; i += blockDim.x) {
    const double v = pPartial[i];
    if constexpr (IsInf) acc = fmax(acc, v);
    else acc += v;
  }
  shared[threadIdx.x] = acc;
  __syncthreads();
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      if constexpr (IsInf) shared[threadIdx.x] = fmax(shared[threadIdx.x], shared[threadIdx.x + s]);
      else shared[threadIdx.x] += shared[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const double v = shared[0];
    pOut[0] = (NT == NormKind::L2) ? sqrt(v) : v;
  }
}

// ---------------------------------------------------------------------------
// NormRel (L1 / L2 / Inf): relative error norm between two images.
//   NormRel = norm(pSrc1 - pSrc2) / norm(pSrc2)   (per channel / per valid
//   pixel set when masked).
// ---------------------------------------------------------------------------

// Multi-channel plain NormRel.  Writes two partial slabs (numerator = diff,
// denominator = src2 norm) of nch columns each, block-strided by gridDim.x.
// vecPath selects the 16-byte two-source walk (both sources aligned).
template <NormKind NT, typename D, int Channels, bool AlphaPreserve>
__global__ void normRelPartialKernel(const D *pSrc1, int nSrc1Step, const D *pSrc2, int nSrc2Step, int width, int height,
                                     double *pDiff, double *pNorm, int nch, int vecPath) {
  constexpr int readCh = AlphaPreserve ? Channels - 1 : Channels;
  constexpr bool IsInf = (NT == NormKind::Inf);
  double dlocal[4] = {0.0, 0.0, 0.0, 0.0};
  double nlocal[4] = {0.0, 0.0, 0.0, 0.0};
  const int totalPixels = width * height;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (vecPath) {
    if constexpr (Channels == 3) {
      auto groupFn = [&](const VecPack<D> &a0, const VecPack<D> &a1, const VecPack<D> &a2, const VecPack<D> &b0,
                         const VecPack<D> &b1, const VecPack<D> &b2, int x0, int y) {
#pragma unroll
        for (int c = 0; c < 3; ++c) {
#pragma unroll
          for (int m = 0; m < kVecElems<D>(); ++m) {
            const int e = 3 * m + c;
            const int l = e & (kVecElems<D>() - 1);
            const double v1 = static_cast<double>((e < kVecElems<D>()) ? a0.v[l] : (e < 2 * kVecElems<D>()) ? a1.v[l] : a2.v[l]);
            const double v2 = static_cast<double>((e < kVecElems<D>()) ? b0.v[l] : (e < 2 * kVecElems<D>()) ? b1.v[l] : b2.v[l]);
            normAcc<NT>(dlocal[c], v1 - v2);
            normAcc<NT>(nlocal[c], v2);
          }
        }
      };
      auto remFn = [&](int c, D v1, D v2, int x, int y) {
        const double d = static_cast<double>(v1) - static_cast<double>(v2);
        const double n = static_cast<double>(v2);
        if (c == 0) {
          normAcc<NT>(dlocal[0], d);
          normAcc<NT>(nlocal[0], n);
        } else if (c == 1) {
          normAcc<NT>(dlocal[1], d);
          normAcc<NT>(nlocal[1], n);
        } else {
          normAcc<NT>(dlocal[2], d);
          normAcc<NT>(nlocal[2], n);
        }
      };
      walkVecGroups3_2(pSrc1, nSrc1Step, pSrc2, nSrc2Step, width, height, groupFn, remFn);
    } else {
      auto packFn = [&](const VecPack<D> &pa, const VecPack<D> &pb, int x0, int y) {
#pragma unroll
        for (int c = 0; c < readCh; ++c) {
#pragma unroll
          for (int m = 0; m < kVecElems<D>() / Channels; ++m) {
            const double v1 = static_cast<double>(pa.v[m * Channels + c]);
            const double v2 = static_cast<double>(pb.v[m * Channels + c]);
            normAcc<NT>(dlocal[c], v1 - v2);
            normAcc<NT>(nlocal[c], v2);
          }
        }
      };
      auto remFn = [&](int c, D v1, D v2, int x, int y) {
        const double d = static_cast<double>(v1) - static_cast<double>(v2);
        const double n = static_cast<double>(v2);
        if (c == 0) {
          normAcc<NT>(dlocal[0], d);
          normAcc<NT>(nlocal[0], n);
        } else if (c == 1) {
          normAcc<NT>(dlocal[1], d);
          normAcc<NT>(nlocal[1], n);
        } else if (c == 2) {
          normAcc<NT>(dlocal[2], d);
          normAcc<NT>(nlocal[2], n);
        } else if (!AlphaPreserve) {
          normAcc<NT>(dlocal[3], d);
          normAcc<NT>(nlocal[3], n);
        }
      };
      walkVecPacks2<Channels>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, width, height, packFn, remFn);
    }
  } else {
    for (int i = tid; i < totalPixels; i += gridDim.x * blockDim.x) {
      const int y = i / width;
      const int x = i % width;
      const D *r1 = reinterpret_cast<const D *>(reinterpret_cast<const char *>(pSrc1) + y * nSrc1Step);
      const D *r2 = reinterpret_cast<const D *>(reinterpret_cast<const char *>(pSrc2) + y * nSrc2Step);
      for (int c = 0; c < readCh; ++c) {
        const double v1 = static_cast<double>(r1[x * Channels + c]);
        const double v2 = static_cast<double>(r2[x * Channels + c]);
        normAcc<NT>(dlocal[c], v1 - v2);
        normAcc<NT>(nlocal[c], v2);
      }
    }
  }
  for (int c = 0; c < readCh; ++c) {
    const double dr = normBlockReduce<IsInf>(dlocal[c]);
    const double nr = normBlockReduce<IsInf>(nlocal[c]);
    if (threadIdx.x == 0) {
      const size_t base = (size_t)c * gridDim.x + blockIdx.x;
      pDiff[base] = dr;
      pNorm[base] = nr;
    }
  }
}

// Combine the block partials into the per-channel relative norm.
// zeroDenomValue carries the source-dtype positive maximum: NVIDIA NPP 13.4
// (RTX 5090) returns 0 when numerator and denominator norms are both zero and
// the dtype maximum when only the denominator is zero; finite ratios are never
// clamped (verified by the normrel zero-denominator / big-ratio goldens).
template <NormKind NT>
__global__ void normRelFinalizeKernel(const double *pDiff, const double *pNorm, int nblocks, int nch,
                                      double zeroDenomValue, Npp64f *pOut) {
  __shared__ double ds[kBlock];
  __shared__ double ns[kBlock];
  constexpr bool IsInf = (NT == NormKind::Inf);
  for (int c = 0; c < nch; ++c) {
    double dacc = 0.0, nacc = 0.0;
    for (int i = threadIdx.x; i < nblocks; i += blockDim.x) {
      const double d = pDiff[(size_t)c * nblocks + i];
      const double n = pNorm[(size_t)c * nblocks + i];
      if constexpr (IsInf) {
        dacc = fmax(dacc, d);
        nacc = fmax(nacc, n);
      } else {
        dacc += d;
        nacc += n;
      }
    }
    ds[threadIdx.x] = dacc;
    ns[threadIdx.x] = nacc;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) {
        if constexpr (IsInf) {
          ds[threadIdx.x] = fmax(ds[threadIdx.x], ds[threadIdx.x + s]);
          ns[threadIdx.x] = fmax(ns[threadIdx.x], ns[threadIdx.x + s]);
        } else {
          ds[threadIdx.x] += ds[threadIdx.x + s];
          ns[threadIdx.x] += ns[threadIdx.x + s];
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      double d = ds[0];
      double n = ns[0];
      if (NT == NormKind::L2) {
        d = sqrt(d);
        n = sqrt(n);
      }
      pOut[c] = (n > 0.0) ? d / n : ((d == 0.0) ? 0.0 : zeroDenomValue);
    }
    __syncthreads();
  }
}

// Masked NormRel over a single channel (nCOI is the 0-based channel index).
template <NormKind NT, typename D, int Channels>
__global__ void normRelMaskedKernel(const D *pSrc1, int nSrc1Step, const D *pSrc2, int nSrc2Step, const Npp8u *pMask,
                                      int nMaskStep, int width, int height, int nCOI, double *pDiff, double *pNorm) {
  constexpr bool IsInf = (NT == NormKind::Inf);
  double dlocal = 0.0, nlocal = 0.0;
  const int totalPixels = width * height;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = tid; i < totalPixels; i += gridDim.x * blockDim.x) {
    const int y = i / width;
    const int x = i % width;
    const Npp8u *maskRow = reinterpret_cast<const Npp8u *>(reinterpret_cast<const char *>(pMask) + y * nMaskStep);
    if (maskRow[x] == 0) continue;
    const D *r1 = reinterpret_cast<const D *>(reinterpret_cast<const char *>(pSrc1) + y * nSrc1Step);
    const D *r2 = reinterpret_cast<const D *>(reinterpret_cast<const char *>(pSrc2) + y * nSrc2Step);
    const double v1 = static_cast<double>(r1[x * Channels + nCOI]);
    const double v2 = static_cast<double>(r2[x * Channels + nCOI]);
    normAcc<NT>(dlocal, v1 - v2);
    normAcc<NT>(nlocal, v2);
  }
  const double dr = normBlockReduce<IsInf>(dlocal);
  const double nr = normBlockReduce<IsInf>(nlocal);
  if (threadIdx.x == 0) {
    pDiff[blockIdx.x] = dr;
    pNorm[blockIdx.x] = nr;
  }
}

// See normRelFinalizeKernel for the zero-denominator convention.
template <NormKind NT>
__global__ void normRelMaskedFinalizeKernel(const double *pDiff, const double *pNorm, int nblocks,
                                            double zeroDenomValue, double *pOut) {
  __shared__ double ds[kBlock];
  __shared__ double ns[kBlock];
  constexpr bool IsInf = (NT == NormKind::Inf);
  double dacc = 0.0, nacc = 0.0;
  for (int i = threadIdx.x; i < nblocks; i += blockDim.x) {
    const double d = pDiff[i];
    const double n = pNorm[i];
    if constexpr (IsInf) {
      dacc = fmax(dacc, d);
      nacc = fmax(nacc, n);
    } else {
      dacc += d;
      nacc += n;
    }
  }
  ds[threadIdx.x] = dacc;
  ns[threadIdx.x] = nacc;
  __syncthreads();
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      if constexpr (IsInf) {
        ds[threadIdx.x] = fmax(ds[threadIdx.x], ds[threadIdx.x + s]);
        ns[threadIdx.x] = fmax(ns[threadIdx.x], ns[threadIdx.x + s]);
      } else {
        ds[threadIdx.x] += ds[threadIdx.x + s];
        ns[threadIdx.x] += ns[threadIdx.x + s];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    double d = ds[0];
    double n = ns[0];
    if (NT == NormKind::L2) {
      d = sqrt(d);
      n = sqrt(n);
    }
    pOut[0] = (n > 0.0) ? d / n : ((d == 0.0) ? 0.0 : zeroDenomValue);
  }
}

// ---------------------------------------------------------------------------
// DotProd: per-channel sum of pSrc1 * pSrc2.
// ---------------------------------------------------------------------------

// vecPath selects the 16-byte two-source walk (both sources aligned).
template <typename D, int Channels, bool AlphaPreserve>
__global__ void dotProdPartialKernel(const D *pSrc1, int nSrc1Step, const D *pSrc2, int nSrc2Step, int width, int height,
                                       double *pPartial, int nch, int vecPath) {
  constexpr int readCh = AlphaPreserve ? Channels - 1 : Channels;
  double local[4] = {0.0, 0.0, 0.0, 0.0};
  const int totalPixels = width * height;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (vecPath) {
    if constexpr (Channels == 3) {
      auto groupFn = [&](const VecPack<D> &a0, const VecPack<D> &a1, const VecPack<D> &a2, const VecPack<D> &b0,
                         const VecPack<D> &b1, const VecPack<D> &b2, int x0, int y) {
#pragma unroll
        for (int c = 0; c < 3; ++c) {
          double t = 0.0;
#pragma unroll
          for (int m = 0; m < kVecElems<D>(); ++m) {
            const int e = 3 * m + c;
            const int l = e & (kVecElems<D>() - 1);
            const double v1 = static_cast<double>((e < kVecElems<D>()) ? a0.v[l] : (e < 2 * kVecElems<D>()) ? a1.v[l] : a2.v[l]);
            const double v2 = static_cast<double>((e < kVecElems<D>()) ? b0.v[l] : (e < 2 * kVecElems<D>()) ? b1.v[l] : b2.v[l]);
            t += v1 * v2;
          }
          local[c] += t;
        }
      };
      auto remFn = [&](int c, D v1, D v2, int x, int y) {
        const double t = static_cast<double>(v1) * static_cast<double>(v2);
        if (c == 0) local[0] += t;
        else if (c == 1) local[1] += t;
        else local[2] += t;
      };
      walkVecGroups3_2(pSrc1, nSrc1Step, pSrc2, nSrc2Step, width, height, groupFn, remFn);
    } else {
      auto packFn = [&](const VecPack<D> &pa, const VecPack<D> &pb, int x0, int y) {
#pragma unroll
        for (int c = 0; c < readCh; ++c) {
          double t = 0.0;
#pragma unroll
          for (int m = 0; m < kVecElems<D>() / Channels; ++m)
            t += static_cast<double>(pa.v[m * Channels + c]) * static_cast<double>(pb.v[m * Channels + c]);
          local[c] += t;
        }
      };
      auto remFn = [&](int c, D v1, D v2, int x, int y) {
        const double t = static_cast<double>(v1) * static_cast<double>(v2);
        if (c == 0) local[0] += t;
        else if (c == 1) local[1] += t;
        else if (c == 2) local[2] += t;
        else if (!AlphaPreserve) local[3] += t;
      };
      walkVecPacks2<Channels>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, width, height, packFn, remFn);
    }
  } else {
    for (int i = tid; i < totalPixels; i += gridDim.x * blockDim.x) {
      const int y = i / width;
      const int x = i % width;
      const D *r1 = reinterpret_cast<const D *>(reinterpret_cast<const char *>(pSrc1) + y * nSrc1Step);
      const D *r2 = reinterpret_cast<const D *>(reinterpret_cast<const char *>(pSrc2) + y * nSrc2Step);
      for (int c = 0; c < readCh; ++c) {
        local[c] += static_cast<double>(r1[x * Channels + c]) * static_cast<double>(r2[x * Channels + c]);
      }
    }
  }
  for (int c = 0; c < readCh; ++c) {
    const double r = blockReduce(local[c]);
    if (threadIdx.x == 0) pPartial[(size_t)c * gridDim.x + blockIdx.x] = r;
  }
}

__global__ void dotProdFinalizeKernel(const double *pPartial, int nblocks, int nch, Npp64f *pOut) {
  __shared__ double shared[kBlock];
  for (int c = 0; c < nch; ++c) {
    double acc = 0.0;
    for (int i = threadIdx.x; i < nblocks; i += blockDim.x) acc += pPartial[(size_t)c * nblocks + i];
    shared[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
      __syncthreads();
    }
    if (threadIdx.x == 0) pOut[c] = shared[0];
    __syncthreads();
  }
}

} // namespace npp_reduce
