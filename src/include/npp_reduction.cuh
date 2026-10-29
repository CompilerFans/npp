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
// Sum / Mean (plain multi-channel).  Writes nch partials per block.
// ---------------------------------------------------------------------------

template <typename D, int Channels, bool AlphaPreserve, typename Acc>
__global__ void reducePartialKernel(const D *pSrc, int nSrcStep, int width, int height, Acc *pPartial, int nch) {
  Acc local[4] = {0, 0, 0, 0};
  const int totalPixels = width * height;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr int readCh = AlphaPreserve ? Channels - 1 : Channels;

  for (int i = tid; i < totalPixels; i += gridDim.x * blockDim.x) {
    const int y = i / width;
    const int x = i % width;
    const D *row = reinterpret_cast<const D *>(reinterpret_cast<const char *>(pSrc) + y * nSrcStep);
    for (int c = 0; c < readCh; ++c) {
      local[c] += static_cast<Acc>(row[x * Channels + c]);
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

template <typename D, int Channels>
__global__ void msSumKernel(const D *pSrc, int nSrcStep, int width, int height, int nCOI, double *pSum,
                            double *pSumSq) {
  const int totalPixels = width * height;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  double lsum = 0.0, lsq = 0.0;
  for (int i = tid; i < totalPixels; i += gridDim.x * blockDim.x) {
    const int y = i / width;
    const int x = i % width;
    const D *row = reinterpret_cast<const D *>(reinterpret_cast<const char *>(pSrc) + y * nSrcStep);
    const double v = static_cast<double>(row[x * Channels + nCOI]);
    lsum += v;
    lsq += v * v;
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

// Multi-channel plain norm (Diff=false) or norm-diff (Diff=true).
template <NormKind NT, typename D, int Channels, bool AlphaPreserve, bool Diff>
__global__ void normPartialKernel(const D *pSrc, int nSrcStep, const D *pSrc2, int nSrc2Step, int width, int height,
                                  double *pPartial, int nch) {
  constexpr int readCh = AlphaPreserve ? Channels - 1 : Channels;
  constexpr bool IsInf = (NT == NormKind::Inf);
  double local[4] = {0.0, 0.0, 0.0, 0.0};
  const int totalPixels = width * height;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
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
template <NormKind NT, typename D, int Channels, bool AlphaPreserve>
__global__ void normRelPartialKernel(const D *pSrc1, int nSrc1Step, const D *pSrc2, int nSrc2Step, int width, int height,
                                     double *pDiff, double *pNorm, int nch) {
  constexpr int readCh = AlphaPreserve ? Channels - 1 : Channels;
  constexpr bool IsInf = (NT == NormKind::Inf);
  double dlocal[4] = {0.0, 0.0, 0.0, 0.0};
  double nlocal[4] = {0.0, 0.0, 0.0, 0.0};
  const int totalPixels = width * height;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
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

template <typename D, int Channels, bool AlphaPreserve>
__global__ void dotProdPartialKernel(const D *pSrc1, int nSrc1Step, const D *pSrc2, int nSrc2Step, int width, int height,
                                       double *pPartial, int nch) {
  constexpr int readCh = AlphaPreserve ? Channels - 1 : Channels;
  double local[4] = {0.0, 0.0, 0.0, 0.0};
  const int totalPixels = width * height;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = tid; i < totalPixels; i += gridDim.x * blockDim.x) {
    const int y = i / width;
    const int x = i % width;
    const D *r1 = reinterpret_cast<const D *>(reinterpret_cast<const char *>(pSrc1) + y * nSrc1Step);
    const D *r2 = reinterpret_cast<const D *>(reinterpret_cast<const char *>(pSrc2) + y * nSrc2Step);
    for (int c = 0; c < readCh; ++c) {
      local[c] += static_cast<double>(r1[x * Channels + c]) * static_cast<double>(r2[x * Channels + c]);
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
