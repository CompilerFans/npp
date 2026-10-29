// nppiSum / nppiMean / nppiMean_StdDev executor kernels + explicit
// instantiation (see nppi_statistics_executor.h for the host surface).
#include "npp.h"
#include "nppi_statistics_executor.h"
#include <cmath>
#include <cuda_runtime.h>
#include <limits>

namespace {


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

// Zero one device double (matches NVIDIA's nCOI<=0 convention for COI kernels).
inline NppStatus zeroDevice(Npp64f *p, cudaStream_t s) {
  cudaError_t e = cudaMemsetAsync(p, 0, sizeof(Npp64f), s);
  return (e == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
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
// Min / Max / MinMax
// ---------------------------------------------------------------------------

// Neutral initial value: largest for a min-reduce, smallest for a max-reduce.
template <typename D, bool IsMin> __device__ __forceinline__ D extremeInit() {
  return IsMin ? std::numeric_limits<D>::max() : std::numeric_limits<D>::lowest();
}

// Block-wide min/max reduction (IsMin selects the combine op).
template <typename D, bool IsMin> __device__ D blockReduceExtreme(D val) {
  static __shared__ D shared[kBlock];
  shared[threadIdx.x] = val;
  __syncthreads();
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      D a = shared[threadIdx.x];
      D b = shared[threadIdx.x + s];
      shared[threadIdx.x] = IsMin ? (a < b ? a : b) : (a > b ? a : b);
    }
    __syncthreads();
  }
  return shared[0];
}

template <typename D, int Channels, bool AlphaPreserve, bool IsMin>
__global__ void reduceExtremeKernel(const D *pSrc, int nSrcStep, int width, int height, D *pPartial, int nch) {
  D local[4];
  constexpr int readCh = AlphaPreserve ? Channels - 1 : Channels;
  for (int c = 0; c < readCh; ++c) local[c] = extremeInit<D, IsMin>();
  const int totalPixels = width * height;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = tid; i < totalPixels; i += gridDim.x * blockDim.x) {
    const int y = i / width;
    const int x = i % width;
    const D *row = reinterpret_cast<const D *>(reinterpret_cast<const char *>(pSrc) + y * nSrcStep);
    for (int c = 0; c < readCh; ++c) {
      const D v = row[x * Channels + c];
      if (IsMin) {
        local[c] = (v < local[c]) ? v : local[c];
      } else {
        local[c] = (v > local[c]) ? v : local[c];
      }
    }
  }
  for (int c = 0; c < readCh; ++c) {
    const D r = blockReduceExtreme<D, IsMin>(local[c]);
    if (threadIdx.x == 0) pPartial[(size_t)c * gridDim.x + blockIdx.x] = r;
  }
}

template <typename D, bool IsMin>
__global__ void finalizeExtremeKernel(const D *pPartial, int nblocks, int nch, D *pOut) {
  __shared__ D shared[kBlock];
  for (int c = 0; c < nch; ++c) {
    D acc = extremeInit<D, IsMin>();
    for (int i = threadIdx.x; i < nblocks; i += blockDim.x) {
      const D v = pPartial[(size_t)c * nblocks + i];
      acc = IsMin ? (v < acc ? v : acc) : (v > acc ? v : acc);
    }
    shared[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) {
        D a = shared[threadIdx.x];
        D b = shared[threadIdx.x + s];
        shared[threadIdx.x] = IsMin ? (a < b ? a : b) : (a > b ? a : b);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) pOut[c] = shared[0];
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// MinIndx / MaxIndx: per-channel extreme value + first-occurrence coordinate.
// ---------------------------------------------------------------------------

// Candidate a=(av,ax,ay) is "better" than b when a is the preferred extreme and,
// on a tie, a holds the earlier pixel in row-major scan order (smaller y then x).
template <typename D, bool IsMin> __device__ __forceinline__ bool extrBeats(D av, int ax, int ay, D bv, int bx, int by) {
  if (IsMin) {
    if (av < bv) return true;
    if (bv < av) return false;
  } else {
    if (av > bv) return true;
    if (bv > av) return false;
  }
  if (ay != by) return ay < by;
  return ax < bx;
}

template <typename D, int Channels, bool AlphaPreserve, bool IsMin>
__global__ void extrIndxKernel(const D *pSrc, int nSrcStep, int width, int height, D *pVal, int *pX, int *pY, int nch) {
  constexpr int readCh = AlphaPreserve ? Channels - 1 : Channels;
  D lv[4];
  int lx[4], ly[4];
  for (int c = 0; c < readCh; ++c) {
    lv[c] = extremeInit<D, IsMin>();
    lx[c] = 0;
    ly[c] = 0;
  }
  const int total = width * height;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = tid; i < total; i += gridDim.x * blockDim.x) {
    const int y = i / width;
    const int x = i % width;
    const D *row = reinterpret_cast<const D *>(reinterpret_cast<const char *>(pSrc) + y * nSrcStep);
    for (int c = 0; c < readCh; ++c) {
      const D v = row[x * Channels + c];
      if (extrBeats<D, IsMin>(v, x, y, lv[c], lx[c], ly[c])) {
        lv[c] = v;
        lx[c] = x;
        ly[c] = y;
      }
    }
  }

  __shared__ D sv[kBlock][4];
  __shared__ int sx[kBlock][4], sy[kBlock][4];
  for (int c = 0; c < readCh; ++c) {
    sv[threadIdx.x][c] = lv[c];
    sx[threadIdx.x][c] = lx[c];
    sy[threadIdx.x][c] = ly[c];
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) {
        D a = sv[threadIdx.x][c];
        int ax = sx[threadIdx.x][c], ay = sy[threadIdx.x][c];
        D b = sv[threadIdx.x + s][c];
        int bx = sx[threadIdx.x + s][c], by = sy[threadIdx.x + s][c];
        if (extrBeats<D, IsMin>(b, bx, by, a, ax, ay)) {
          sv[threadIdx.x][c] = b;
          sx[threadIdx.x][c] = bx;
          sy[threadIdx.x][c] = by;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      const size_t base = (size_t)c * gridDim.x;
      pVal[base + blockIdx.x] = sv[0][c];
      pX[base + blockIdx.x] = sx[0][c];
      pY[base + blockIdx.x] = sy[0][c];
    }
    __syncthreads();
  }
}

template <typename D, bool IsMin>
__global__ void extrIndxFinalKernel(const D *pVal, const int *pX, const int *pY, int nblocks, int nch, D *outVal,
                                    int *outX, int *outY) {
  __shared__ D sv[kBlock];
  __shared__ int sx[kBlock], sy[kBlock];
  for (int c = 0; c < nch; ++c) {
    D accv = extremeInit<D, IsMin>();
    int accx = 0, accy = 0;
    for (int i = threadIdx.x; i < nblocks; i += blockDim.x) {
      const size_t base = (size_t)c * nblocks;
      D v = pVal[base + i];
      int x = pX[base + i], y = pY[base + i];
      if (extrBeats<D, IsMin>(v, x, y, accv, accx, accy)) {
        accv = v;
        accx = x;
        accy = y;
      }
    }
    sv[threadIdx.x] = accv;
    sx[threadIdx.x] = accx;
    sy[threadIdx.x] = accy;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) {
        D a = sv[threadIdx.x];
        int ax = sx[threadIdx.x], ay = sy[threadIdx.x];
        D b = sv[threadIdx.x + s];
        int bx = sx[threadIdx.x + s], by = sy[threadIdx.x + s];
        if (extrBeats<D, IsMin>(b, bx, by, a, ax, ay)) {
          sv[threadIdx.x] = b;
          sx[threadIdx.x] = bx;
          sy[threadIdx.x] = by;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      outVal[c] = sv[0];
      outX[c] = sx[0];
      outY[c] = sy[0];
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// MinMaxIndx: scalar min + scalar max for one selected channel, each with its
// coordinate.  pMask may be null (unmasked); when present a non-zero byte marks
// a valid pixel.  channel is the 0-based source channel index.
// ---------------------------------------------------------------------------

template <typename D, int Channels>
__global__ void minmaxIndxKernel(const D *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep, int width, int height,
                                 int channel, D *minV, int *minX, int *minY, D *maxV, int *maxX, int *maxY) {
  D lmin = extremeInit<D, true>();
  D lmax = extremeInit<D, false>();
  int lminx = 0, lminy = 0, lmaxx = 0, lmaxy = 0;
  const int total = width * height;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = tid; i < total; i += gridDim.x * blockDim.x) {
    const int y = i / width;
    const int x = i % width;
    const Npp8u *maskRow =
        pMask ? reinterpret_cast<const Npp8u *>(reinterpret_cast<const char *>(pMask) + y * nMaskStep) : nullptr;
    if (maskRow && maskRow[x] == 0) continue;
    const D *row = reinterpret_cast<const D *>(reinterpret_cast<const char *>(pSrc) + y * nSrcStep);
    const D v = row[x * Channels + channel];
    if (extrBeats<D, true>(v, x, y, lmin, lminx, lminy)) {
      lmin = v;
      lminx = x;
      lminy = y;
    }
    if (extrBeats<D, false>(v, x, y, lmax, lmaxx, lmaxy)) {
      lmax = v;
      lmaxx = x;
      lmaxy = y;
    }
  }
  __shared__ D smin[kBlock], smax[kBlock];
  __shared__ int sminx[kBlock], sminy[kBlock], smaxx[kBlock], smaxy[kBlock];
  smin[threadIdx.x] = lmin;
  smax[threadIdx.x] = lmax;
  sminx[threadIdx.x] = lminx;
  sminy[threadIdx.x] = lminy;
  smaxx[threadIdx.x] = lmaxx;
  smaxy[threadIdx.x] = lmaxy;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      const int o = threadIdx.x + s;
      if (extrBeats<D, true>(smin[o], sminx[o], sminy[o], smin[threadIdx.x], sminx[threadIdx.x],
                             sminy[threadIdx.x])) {
        smin[threadIdx.x] = smin[o];
        sminx[threadIdx.x] = sminx[o];
        sminy[threadIdx.x] = sminy[o];
      }
      if (extrBeats<D, false>(smax[o], smaxx[o], smaxy[o], smax[threadIdx.x], smaxx[threadIdx.x],
                              smaxy[threadIdx.x])) {
        smax[threadIdx.x] = smax[o];
        smaxx[threadIdx.x] = smaxx[o];
        smaxy[threadIdx.x] = smaxy[o];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    minV[blockIdx.x] = smin[0];
    minX[blockIdx.x] = sminx[0];
    minY[blockIdx.x] = sminy[0];
    maxV[blockIdx.x] = smax[0];
    maxX[blockIdx.x] = smaxx[0];
    maxY[blockIdx.x] = smaxy[0];
  }
}

template <typename D, bool IsMin>
__device__ void mmFinalizeReduce(const D *pV, const int *pX, const int *pY, int nblocks, D &outV, int &outX, int &outY) {
  D accv = extremeInit<D, IsMin>();
  int accx = 0, accy = 0;
  for (int i = threadIdx.x; i < nblocks; i += blockDim.x) {
    if (extrBeats<D, IsMin>(pV[i], pX[i], pY[i], accv, accx, accy)) {
      accv = pV[i];
      accx = pX[i];
      accy = pY[i];
    }
  }
  __shared__ D sv[kBlock];
  __shared__ int sx[kBlock], sy[kBlock];
  sv[threadIdx.x] = accv;
  sx[threadIdx.x] = accx;
  sy[threadIdx.x] = accy;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      if (extrBeats<D, IsMin>(sv[threadIdx.x + s], sx[threadIdx.x + s], sy[threadIdx.x + s], sv[threadIdx.x],
                              sx[threadIdx.x], sy[threadIdx.x])) {
        sv[threadIdx.x] = sv[threadIdx.x + s];
        sx[threadIdx.x] = sx[threadIdx.x + s];
        sy[threadIdx.x] = sy[threadIdx.x + s];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    outV = sv[0];
    outX = sx[0];
    outY = sy[0];
  }
  __syncthreads();
}

template <typename D>
__global__ void minmaxIndxFinalKernel(const D *minV, const int *minX, const int *minY, const D *maxV, const int *maxX,
                                      const int *maxY, int nblocks, D *pMinValue, NppiPoint *pMinIndex, D *pMaxValue,
                                      NppiPoint *pMaxIndex) {
  D ov, MV;
  int ox, oy, Mx, My;
  mmFinalizeReduce<D, true>(minV, minX, minY, nblocks, ov, ox, oy);
  mmFinalizeReduce<D, false>(maxV, maxX, maxY, nblocks, MV, Mx, My);
  if (threadIdx.x == 0) {
    *pMinValue = ov;
    pMinIndex->x = ox;
    pMinIndex->y = oy;
    *pMaxValue = MV;
    pMaxIndex->x = Mx;
    pMaxIndex->y = My;
  }
}

// ---------------------------------------------------------------------------
// Norm / NormDiff (L1 / L2 / Inf)
// ---------------------------------------------------------------------------

// Contribution of one raw (possibly diff) sample.  L2 keeps the square (sqrt
// is applied at finalize); L1 / Inf use |raw|.
template <nppi::statistics::NormType NT>
__device__ __forceinline__ double normContribution(double raw) {
  if constexpr (NT == nppi::statistics::NormType::L2) return raw * raw;
  return fabs(raw);
}

// Fold raw into acc: L1/L2 accumulate, Inf keeps the running maximum.
template <nppi::statistics::NormType NT>
__device__ __forceinline__ void normAcc(double &acc, double raw) {
  double c = normContribution<NT>(raw);
  if constexpr (NT == nppi::statistics::NormType::Inf) acc = fmax(acc, c);
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
template <nppi::statistics::NormType NT, typename D, int Channels, bool AlphaPreserve, bool Diff>
__global__ void normPartialKernel(const D *pSrc, int nSrcStep, const D *pSrc2, int nSrc2Step, int width, int height,
                                  double *pPartial, int nch) {
  constexpr int readCh = AlphaPreserve ? Channels - 1 : Channels;
  constexpr bool IsInf = (NT == nppi::statistics::NormType::Inf);
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
template <nppi::statistics::NormType NT>
__global__ void normFinalizeKernel(const double *pPartial, int nblocks, int nch, Npp64f *pOut) {
  __shared__ double shared[kBlock];
  constexpr bool IsInf = (NT == nppi::statistics::NormType::Inf);
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
      pOut[c] = (NT == nppi::statistics::NormType::L2) ? sqrt(v) : v;
    }
    __syncthreads();
  }
}

// Masked norm / norm-diff over a single channel (nCOI is the 0-based channel).
template <nppi::statistics::NormType NT, typename D, int Channels, bool Diff>
__global__ void normMaskedKernel(const D *pSrc, int nSrcStep, const D *pSrc2, int nSrc2Step, const Npp8u *pMask,
                                  int nMaskStep, int width, int height, int nCOI, double *pPartial) {
  constexpr bool IsInf = (NT == nppi::statistics::NormType::Inf);
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

template <nppi::statistics::NormType NT>
__global__ void normMaskedFinalizeKernel(const double *pPartial, int nblocks, double *pOut) {
  __shared__ double shared[kBlock];
  constexpr bool IsInf = (NT == nppi::statistics::NormType::Inf);
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
    pOut[0] = (NT == nppi::statistics::NormType::L2) ? sqrt(v) : v;
  }
}

// ---------------------------------------------------------------------------
// NormRel (L1 / L2 / Inf): relative error norm between two images.
//   NormRel = norm(pSrc1 - pSrc2) / norm(pSrc2)   (per channel / per valid
//   pixel set when masked).
// ---------------------------------------------------------------------------

// Multi-channel plain NormRel.  Writes two partial slabs (numerator = diff,
// denominator = src2 norm) of nch columns each, block-strided by gridDim.x.
template <nppi::statistics::NormType NT, typename D, int Channels, bool AlphaPreserve>
__global__ void normRelPartialKernel(const D *pSrc1, int nSrc1Step, const D *pSrc2, int nSrc2Step, int width, int height,
                                     double *pDiff, double *pNorm, int nch) {
  constexpr int readCh = AlphaPreserve ? Channels - 1 : Channels;
  constexpr bool IsInf = (NT == nppi::statistics::NormType::Inf);
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
template <nppi::statistics::NormType NT>
__global__ void normRelFinalizeKernel(const double *pDiff, const double *pNorm, int nblocks, int nch,
                                      double zeroDenomValue, Npp64f *pOut) {
  __shared__ double ds[kBlock];
  __shared__ double ns[kBlock];
  constexpr bool IsInf = (NT == nppi::statistics::NormType::Inf);
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
      if (NT == nppi::statistics::NormType::L2) {
        d = sqrt(d);
        n = sqrt(n);
      }
      pOut[c] = (n > 0.0) ? d / n : ((d == 0.0) ? 0.0 : zeroDenomValue);
    }
    __syncthreads();
  }
}

// Masked NormRel over a single channel (nCOI is the 0-based channel index).
template <nppi::statistics::NormType NT, typename D, int Channels>
__global__ void normRelMaskedKernel(const D *pSrc1, int nSrc1Step, const D *pSrc2, int nSrc2Step, const Npp8u *pMask,
                                      int nMaskStep, int width, int height, int nCOI, double *pDiff, double *pNorm) {
  constexpr bool IsInf = (NT == nppi::statistics::NormType::Inf);
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
template <nppi::statistics::NormType NT>
__global__ void normRelMaskedFinalizeKernel(const double *pDiff, const double *pNorm, int nblocks,
                                            double zeroDenomValue, double *pOut) {
  __shared__ double ds[kBlock];
  __shared__ double ns[kBlock];
  constexpr bool IsInf = (NT == nppi::statistics::NormType::Inf);
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
    if (NT == nppi::statistics::NormType::L2) {
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

// ---------------------------------------------------------------------------
// AverageError / AverageRelativeError: scalar mean of |a - b| (relative:
// |a - b| / max(|a|, |b|), 0 when both are zero) over all channels.
// Complex elements contribute their complex modulus.
// ---------------------------------------------------------------------------

__device__ __forceinline__ double absDiff(Npp16sc a, Npp16sc b) {
  return hypot((double)a.re - (double)b.re, (double)a.im - (double)b.im);
}
__device__ __forceinline__ double absDiff(Npp32sc a, Npp32sc b) {
  return hypot((double)a.re - (double)b.re, (double)a.im - (double)b.im);
}
__device__ __forceinline__ double absDiff(Npp32fc a, Npp32fc b) {
  return hypot((double)a.re - (double)b.re, (double)a.im - (double)b.im);
}
template <typename D>
__device__ __forceinline__ double absDiff(D a, D b) {
  return fabs((double)a - (double)b);
}

__device__ __forceinline__ double absVal(Npp16sc a) { return hypot((double)a.re, (double)a.im); }
__device__ __forceinline__ double absVal(Npp32sc a) { return hypot((double)a.re, (double)a.im); }
__device__ __forceinline__ double absVal(Npp32fc a) { return hypot((double)a.re, (double)a.im); }
template <typename D>
__device__ __forceinline__ double absVal(D a) {
  return fabs((double)a);
}

// One scalar partial per block (unlike dotProd's per-channel slab).
template <typename D, int Channels, bool Relative>
__global__ void avgErrPartialKernel(const D *pSrc1, int nSrc1Step, const D *pSrc2, int nSrc2Step, int width, int height,
                                    double *pPartial) {
  double local = 0.0;
  const int totalPixels = width * height;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = tid; i < totalPixels; i += gridDim.x * blockDim.x) {
    const int y = i / width;
    const int x = i % width;
    const D *r1 = reinterpret_cast<const D *>(reinterpret_cast<const char *>(pSrc1) + y * nSrc1Step);
    const D *r2 = reinterpret_cast<const D *>(reinterpret_cast<const char *>(pSrc2) + y * nSrc2Step);
    for (int c = 0; c < Channels; ++c) {
      const D a = r1[x * Channels + c];
      const D b = r2[x * Channels + c];
      if (Relative) {
        const double den = fmax(absVal(a), absVal(b));
        local += (den > 0.0) ? absDiff(a, b) / den : 0.0;
      } else {
        local += absDiff(a, b);
      }
    }
  }
  const double r = blockReduce(local);
  if (threadIdx.x == 0) pPartial[blockIdx.x] = r;
}

__global__ void avgErrFinalizeKernel(const double *pPartial, int nblocks, double count, Npp64f *pOut) {
  __shared__ double shared[kBlock];
  double acc = 0.0;
  for (int i = threadIdx.x; i < nblocks; i += blockDim.x) acc += pPartial[i];
  shared[threadIdx.x] = acc;
  __syncthreads();
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) pOut[0] = shared[0] / count;
}

}  // namespace

namespace nppi {
namespace statistics {

// ============================== Sum =========================================

template <typename D, int Channels, typename Acc, bool AlphaPreserve>
NppStatus runSum(const D *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Acc *pSum,
                 cudaStream_t stream) {
  if (!pSrc) return NPP_NULL_POINTER_ERROR;
  if (!pDeviceBuffer || !pSum) return NPP_NO_ERROR;      // NVIDIA: result/scratch buffers not validated
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;  // empty ROI is a no-op
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  const int esize = (int)sizeof(D);
  if (nSrcStep < oSizeROI.width * Channels * esize) return NPP_STEP_ERROR;

  constexpr int nCh = Channels - (AlphaPreserve ? 1 : 0);
  const int area = oSizeROI.width * oSizeROI.height;
  const int blocks = numBlocks(area);
  Acc *partial = reinterpret_cast<Acc *>(pDeviceBuffer);

  (void)cudaGetLastError();
  reducePartialKernel<D, Channels, AlphaPreserve, Acc>
      <<<blocks, kBlock, 0, stream>>>(pSrc, nSrcStep, oSizeROI.width, oSizeROI.height, partial, nCh);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;

  finalizeOutKernel<Acc, false><<<1, kBlock, 0, stream>>>(partial, blocks, nCh, 0.0, pSum);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

// ============================== Mean ========================================

template <typename DType, int Channels, bool AlphaPreserve>
NppStatus runMean(const DType *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp64f *pMean,
                  cudaStream_t stream) {
  if (!pSrc) return NPP_NULL_POINTER_ERROR;
  if (!pDeviceBuffer || !pMean) return NPP_NO_ERROR;      // NVIDIA: result/scratch buffers not validated
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;  // empty ROI is a no-op
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  const int esize = (int)sizeof(DType);
  if (nSrcStep < oSizeROI.width * Channels * esize) return NPP_STEP_ERROR;

  constexpr int nCh = Channels - (AlphaPreserve ? 1 : 0);
  const int area = oSizeROI.width * oSizeROI.height;
  const int blocks = numBlocks(area);
  Npp64f *partial = reinterpret_cast<Npp64f *>(pDeviceBuffer);

  (void)cudaGetLastError();
  reducePartialKernel<DType, Channels, AlphaPreserve, Npp64f>
      <<<blocks, kBlock, 0, stream>>>(pSrc, nSrcStep, oSizeROI.width, oSizeROI.height, partial, nCh);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;

  finalizeOutKernel<Npp64f, true>
      <<<1, kBlock, 0, stream>>>(partial, blocks, nCh, static_cast<double>(area), pMean);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

template <typename DType>
NppStatus runMeanMasked(const DType *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep, NppiSize oi,
                        Npp8u *pDeviceBuffer, Npp64f *pMean, cudaStream_t stream) {
  if (!pSrc || !pMask || !pDeviceBuffer || !pMean) return NPP_NULL_POINTER_ERROR;
  if (oi.width <= 0 || oi.height <= 0) return NPP_SIZE_ERROR;
  if (nSrcStep < oi.width * (int)sizeof(DType)) return NPP_STEP_ERROR;
  if (nMaskStep < oi.width) return NPP_STEP_ERROR;

  const int blocks = numBlocks(oi.width * oi.height);
  double *sums = reinterpret_cast<double *>(pDeviceBuffer);
  double *counts = sums + blocks;

  (void)cudaGetLastError();
  maskedSumKernel<DType, 1><<<blocks, kBlock, 0, stream>>>(pSrc, nSrcStep, pMask, nMaskStep, oi.width, oi.height, 0,
                                                           sums, counts);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;

  finalizeMaskedKernel<<<1, kBlock, 0, stream>>>(sums, counts, blocks, pMean);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

template <typename DType>
NppStatus runMeanMaskedCoi(const DType *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep, NppiSize oi, int nCOI,
                           Npp8u *pDeviceBuffer, Npp64f *pMean, cudaStream_t stream) {
  if (!pSrc || !pMask || !pDeviceBuffer || !pMean) return NPP_NULL_POINTER_ERROR;
  if (oi.width <= 0 || oi.height <= 0) return NPP_SIZE_ERROR;
  if (nCOI < 0 || nCOI > 3) return NPP_COI_ERROR;
  if (nSrcStep < oi.width * 3 * (int)sizeof(DType)) return NPP_STEP_ERROR;
  if (nMaskStep < oi.width) return NPP_STEP_ERROR;

  // NVIDIA NPP treats the masked-COI channel as 1-based (channel = nCOI-1);
  // nCOI<=0 yields no channel and the mean is 0.
  const int srcCh = nCOI - 1;
  if (nCOI <= 0) return zeroDevice(pMean, stream);       // nCOI <= 0 -> mean 0

  const int blocks = numBlocks(oi.width * oi.height);
  double *sums = reinterpret_cast<double *>(pDeviceBuffer);
  double *counts = sums + blocks;

  (void)cudaGetLastError();
  maskedSumKernel<DType, 3><<<blocks, kBlock, 0, stream>>>(pSrc, nSrcStep, pMask, nMaskStep, oi.width, oi.height, srcCh,
                                                           sums, counts);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;

  finalizeMaskedKernel<<<1, kBlock, 0, stream>>>(sums, counts, blocks, pMean);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

// ============================== Mean + StdDev ===============================

template <typename DType>
NppStatus runMeanStdDev(const DType *pSrc, int nSrcStep, NppiSize oi, Npp8u *pDeviceBuffer, Npp64f *pMean,
                        Npp64f *pStdDev, cudaStream_t stream) {
  if (!pSrc || !pDeviceBuffer || !pMean || !pStdDev) return NPP_NULL_POINTER_ERROR;
  if (oi.width <= 0 || oi.height <= 0) return NPP_SIZE_ERROR;
  if (nSrcStep < oi.width * (int)sizeof(DType)) return NPP_STEP_ERROR;

  const int blocks = numBlocks(oi.width * oi.height);
  double *sums = reinterpret_cast<double *>(pDeviceBuffer);
  double *sq = sums + blocks;

  (void)cudaGetLastError();
  msSumKernel<DType, 1><<<blocks, kBlock, 0, stream>>>(pSrc, nSrcStep, oi.width, oi.height, 0, sums, sq);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;

  finalizeMSKernel<<<1, kBlock, 0, stream>>>(sums, sq, blocks, static_cast<double>(oi.width * oi.height), pMean,
                                             pStdDev);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

template <typename DType>
NppStatus runMeanStdDevCoi(const DType *pSrc, int nSrcStep, NppiSize oi, int nCOI, Npp8u *pDeviceBuffer, Npp64f *pMean,
                           Npp64f *pStdDev, cudaStream_t stream) {
  if (!pSrc || !pDeviceBuffer || !pMean || !pStdDev) return NPP_NULL_POINTER_ERROR;
  if (oi.width <= 0 || oi.height <= 0) return NPP_SIZE_ERROR;
  if (nCOI < 0 || nCOI > 3) return NPP_COI_ERROR;
  if (nSrcStep < oi.width * 3 * (int)sizeof(DType)) return NPP_STEP_ERROR;

  // NVIDIA NPP: COI channel is 1-based (channel = nCOI-1); nCOI<=0 -> COI error.
  const int srcCh = nCOI - 1;
  if (nCOI <= 0) return NPP_COI_ERROR;

  const int blocks = numBlocks(oi.width * oi.height);
  double *sums = reinterpret_cast<double *>(pDeviceBuffer);
  double *sq = sums + blocks;

  (void)cudaGetLastError();
  msSumKernel<DType, 3><<<blocks, kBlock, 0, stream>>>(pSrc, nSrcStep, oi.width, oi.height, srcCh, sums, sq);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;

  finalizeMSKernel<<<1, kBlock, 0, stream>>>(sums, sq, blocks, static_cast<double>(oi.width * oi.height), pMean,
                                             pStdDev);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

template <typename DType>
NppStatus runMeanStdDevMasked(const DType *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep, NppiSize oi,
                              Npp8u *pDeviceBuffer, Npp64f *pMean, Npp64f *pStdDev, cudaStream_t stream) {
  if (!pSrc || !pMask || !pDeviceBuffer || !pMean || !pStdDev) return NPP_NULL_POINTER_ERROR;
  if (oi.width <= 0 || oi.height <= 0) return NPP_SIZE_ERROR;
  if (nSrcStep < oi.width * (int)sizeof(DType)) return NPP_STEP_ERROR;
  if (nMaskStep < oi.width) return NPP_STEP_ERROR;

  const int blocks = numBlocks(oi.width * oi.height);
  double *sums = reinterpret_cast<double *>(pDeviceBuffer);
  double *sq = sums + blocks;
  double *cnt = sq + blocks;

  (void)cudaGetLastError();
  msMaskedSumKernel<DType, 1><<<blocks, kBlock, 0, stream>>>(pSrc, nSrcStep, pMask, nMaskStep, oi.width, oi.height, 0,
                                                             sums, sq, cnt);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;

  finalizeMSMaskedKernel<<<1, kBlock, 0, stream>>>(sums, sq, cnt, blocks, pMean, pStdDev);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

template <typename DType>
NppStatus runMeanStdDevMaskedCoi(const DType *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep, NppiSize oi, int nCOI,
                                 Npp8u *pDeviceBuffer, Npp64f *pMean, Npp64f *pStdDev, cudaStream_t stream) {
  if (!pSrc || !pMask || !pDeviceBuffer || !pMean || !pStdDev) return NPP_NULL_POINTER_ERROR;
  if (oi.width <= 0 || oi.height <= 0) return NPP_SIZE_ERROR;
  if (nCOI < 0 || nCOI > 3) return NPP_COI_ERROR;
  if (nSrcStep < oi.width * 3 * (int)sizeof(DType)) return NPP_STEP_ERROR;
  if (nMaskStep < oi.width) return NPP_STEP_ERROR;

  // NVIDIA NPP: masked-COI channel is 1-based (channel = nCOI-1); nCOI<=0 -> COI error.
  const int srcCh = nCOI - 1;
  if (nCOI <= 0) return NPP_COI_ERROR;

  const int blocks = numBlocks(oi.width * oi.height);
  double *sums = reinterpret_cast<double *>(pDeviceBuffer);
  double *sq = sums + blocks;
  double *cnt = sq + blocks;

  (void)cudaGetLastError();
  msMaskedSumKernel<DType, 3>
      <<<blocks, kBlock, 0, stream>>>(pSrc, nSrcStep, pMask, nMaskStep, oi.width, oi.height, srcCh, sums, sq, cnt);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;

  finalizeMSMaskedKernel<<<1, kBlock, 0, stream>>>(sums, sq, cnt, blocks, pMean, pStdDev);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

// ============================== Min / Max / MinMax ==========================
// Return the per-channel block-reduced min (IsMin true) or max from pPartial and
// store into pOut.  Writes are staged so the single buffer can hold both the
// min partials and the max partials when run together.

template <typename D, int Channels, bool AlphaPreserve>
NppStatus runMin(const D *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, D *pMin,
                 cudaStream_t stream) {
  if (!pSrc) return NPP_NULL_POINTER_ERROR;
  if (!pDeviceBuffer || !pMin) return NPP_NO_ERROR;      // NVIDIA: result/scratch buffers not validated
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;  // empty ROI is a no-op
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  const int esize = (int)sizeof(D);
  if (nSrcStep < oSizeROI.width * Channels * esize) return NPP_STEP_ERROR;

  constexpr int nCh = Channels - (AlphaPreserve ? 1 : 0);
  const int area = oSizeROI.width * oSizeROI.height;
  const int blocks = numBlocks(area);
  D *partial = reinterpret_cast<D *>(pDeviceBuffer);

  (void)cudaGetLastError();
  reduceExtremeKernel<D, Channels, AlphaPreserve, true>
      <<<blocks, kBlock, 0, stream>>>(pSrc, nSrcStep, oSizeROI.width, oSizeROI.height, partial, nCh);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;

  finalizeExtremeKernel<D, true><<<1, kBlock, 0, stream>>>(partial, blocks, nCh, pMin);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

template <typename D, int Channels, bool AlphaPreserve>
NppStatus runMax(const D *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, D *pMax,
                 cudaStream_t stream) {
  if (!pSrc) return NPP_NULL_POINTER_ERROR;
  if (!pDeviceBuffer || !pMax) return NPP_NO_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  const int esize = (int)sizeof(D);
  if (nSrcStep < oSizeROI.width * Channels * esize) return NPP_STEP_ERROR;

  constexpr int nCh = Channels - (AlphaPreserve ? 1 : 0);
  const int area = oSizeROI.width * oSizeROI.height;
  const int blocks = numBlocks(area);
  D *partial = reinterpret_cast<D *>(pDeviceBuffer);

  (void)cudaGetLastError();
  reduceExtremeKernel<D, Channels, AlphaPreserve, false>
      <<<blocks, kBlock, 0, stream>>>(pSrc, nSrcStep, oSizeROI.width, oSizeROI.height, partial, nCh);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;

  finalizeExtremeKernel<D, false><<<1, kBlock, 0, stream>>>(partial, blocks, nCh, pMax);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

template <typename D, int Channels, bool AlphaPreserve>
NppStatus runMinMax(const D *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, D *pMin, D *pMax,
                    cudaStream_t stream) {
  if (!pSrc) return NPP_NULL_POINTER_ERROR;
  if (!pDeviceBuffer || !pMin || !pMax) return NPP_NO_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  const int esize = (int)sizeof(D);
  if (nSrcStep < oSizeROI.width * Channels * esize) return NPP_STEP_ERROR;

  constexpr int nCh = Channels - (AlphaPreserve ? 1 : 0);
  const int area = oSizeROI.width * oSizeROI.height;
  const int blocks = numBlocks(area);
  D *minPart = reinterpret_cast<D *>(pDeviceBuffer);
  D *maxPart = minPart + (size_t)blocks * nCh;

  (void)cudaGetLastError();
  reduceExtremeKernel<D, Channels, AlphaPreserve, true>
      <<<blocks, kBlock, 0, stream>>>(pSrc, nSrcStep, oSizeROI.width, oSizeROI.height, minPart, nCh);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  reduceExtremeKernel<D, Channels, AlphaPreserve, false>
      <<<blocks, kBlock, 0, stream>>>(pSrc, nSrcStep, oSizeROI.width, oSizeROI.height, maxPart, nCh);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;

  finalizeExtremeKernel<D, true><<<1, kBlock, 0, stream>>>(minPart, blocks, nCh, pMin);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  finalizeExtremeKernel<D, false><<<1, kBlock, 0, stream>>>(maxPart, blocks, nCh, pMax);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

// ============================== MinIndx / MaxIndx ==========================

template <typename D, int Channels, bool AlphaPreserve, bool IsMin>
NppStatus runExtrIndx(const D *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, D *pValue, int *pIndexX,
                      int *pIndexY, cudaStream_t stream) {
  if (!pSrc) return NPP_NULL_POINTER_ERROR;
  if (!pDeviceBuffer || !pValue || !pIndexX || !pIndexY) return NPP_NO_ERROR;  // NVIDIA: buffers not validated
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;        // empty ROI is a no-op
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  const int esize = (int)sizeof(D);
  if (nSrcStep < oSizeROI.width * Channels * esize) return NPP_STEP_ERROR;

  constexpr int nCh = Channels - (AlphaPreserve ? 1 : 0);
  const int area = oSizeROI.width * oSizeROI.height;
  const int blocks = numBlocks(area);
  // Align each partial section so 1/2-byte dtypes do not create misaligned
  // int position arrays (MACA traps on misaligned stores).
  char *base = reinterpret_cast<char *>(pDeviceBuffer);
  D *vPart = reinterpret_cast<D *>(base);
  size_t off = (size_t)blocks * nCh * sizeof(D);
  off = (off + 7u) & ~(size_t)7u;
  int *xPart = reinterpret_cast<int *>(base + off);
  off += (size_t)blocks * nCh * sizeof(int);
  off = (off + 7u) & ~(size_t)7u;
  int *yPart = reinterpret_cast<int *>(base + off);

  (void)cudaGetLastError();
  extrIndxKernel<D, Channels, AlphaPreserve, IsMin>
      <<<blocks, kBlock, 0, stream>>>(pSrc, nSrcStep, oSizeROI.width, oSizeROI.height, vPart, xPart, yPart, nCh);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  extrIndxFinalKernel<D, IsMin><<<1, kBlock, 0, stream>>>(vPart, xPart, yPart, blocks, nCh, pValue, pIndexX, pIndexY);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

// ============================== MinMaxIndx ===================================

template <typename D, int Channels>
NppStatus runMinMaxIndx(const D *p, int nSrcStep, const Npp8u *pMask, int nMaskStep, NppiSize oi, int nCOI,
                        D *pMinValue, D *pMaxValue, NppiPoint *pMinIndex, NppiPoint *pMaxIndex, Npp8u *pDeviceBuffer,
                        cudaStream_t stream) {
  if (!p || !pMinValue || !pMaxValue || !pMinIndex || !pMaxIndex || !pDeviceBuffer) return NPP_NULL_POINTER_ERROR;
  if (oi.width <= 0 || oi.height <= 0) return NPP_SIZE_ERROR;
  if (pMask && nMaskStep < oi.width) return NPP_STEP_ERROR;

  int channel = 0;
  if (Channels == 1) {
    if (nCOI != 0) return NPP_COI_ERROR;
  } else {
    if (nCOI < 1 || nCOI > 3) return NPP_COI_ERROR;
    channel = nCOI - 1;
  }
  if (nSrcStep < oi.width * Channels * (int)sizeof(D)) return NPP_STEP_ERROR;

  const int blocks = numBlocks(oi.width * oi.height);
  char *base = reinterpret_cast<char *>(pDeviceBuffer);
  D *minV = reinterpret_cast<D *>(base);
  size_t off = (size_t)blocks * sizeof(D);
  off = (off + 7u) & ~(size_t)7u;
  int *minX = reinterpret_cast<int *>(base + off);
  off += (size_t)blocks * sizeof(int);
  off = (off + 7u) & ~(size_t)7u;
  int *minY = reinterpret_cast<int *>(base + off);
  off += (size_t)blocks * sizeof(int);
  off = (off + 7u) & ~(size_t)7u;
  D *maxV = reinterpret_cast<D *>(base + off);
  off += (size_t)blocks * sizeof(D);
  off = (off + 7u) & ~(size_t)7u;
  int *maxX = reinterpret_cast<int *>(base + off);
  off += (size_t)blocks * sizeof(int);
  off = (off + 7u) & ~(size_t)7u;
  int *maxY = reinterpret_cast<int *>(base + off);

  (void)cudaGetLastError();
  minmaxIndxKernel<D, Channels><<<blocks, kBlock, 0, stream>>>(p, nSrcStep, pMask, nMaskStep, oi.width, oi.height,
                                                               channel, minV, minX, minY, maxV, maxX, maxY);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  minmaxIndxFinalKernel<D>
      <<<1, kBlock, 0, stream>>>(minV, minX, minY, maxV, maxX, maxY, blocks, pMinValue, pMinIndex, pMaxValue, pMaxIndex);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

// ============================== Norm / NormDiff =============================

template <NormType NT, typename D, int Channels, bool AlphaPreserve>
NppStatus runNorm(const D *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp64f *pNorm,
                  cudaStream_t stream) {
  if (!pSrc) return NPP_NULL_POINTER_ERROR;
  if (!pNorm) return NPP_NULL_POINTER_ERROR;               // NVIDIA 5090 validates the result pointer
  if (!pDeviceBuffer) return NPP_NO_ERROR;                 // scratch buffer not validated
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;  // empty ROI is a no-op
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  const int esize = (int)sizeof(D);
  if (nSrcStep < oSizeROI.width * Channels * esize) return NPP_STEP_ERROR;

  constexpr int nCh = Channels - (AlphaPreserve ? 1 : 0);
  const int area = oSizeROI.width * oSizeROI.height;
  const int blocks = numBlocks(area);
  double *partial = reinterpret_cast<double *>(pDeviceBuffer);

  (void)cudaGetLastError();
  normPartialKernel<NT, D, Channels, AlphaPreserve, false>
      <<<blocks, kBlock, 0, stream>>>(pSrc, nSrcStep, nullptr, 0, oSizeROI.width, oSizeROI.height, partial, nCh);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;

  normFinalizeKernel<NT><<<1, kBlock, 0, stream>>>(partial, blocks, nCh, pNorm);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

template <NormType NT, typename D, int Channels, bool AlphaPreserve>
NppStatus runNormDiff(const D *pSrc1, int nSrc1Step, const D *pSrc2, int nSrc2Step, NppiSize oSizeROI,
                      Npp8u *pDeviceBuffer, Npp64f *pNorm, cudaStream_t stream) {
  if (!pSrc1 || !pSrc2) return NPP_NULL_POINTER_ERROR;
  if (!pNorm) return NPP_NULL_POINTER_ERROR;               // NVIDIA 5090 validates the result pointer
  if (!pDeviceBuffer) return NPP_NO_ERROR;                 // scratch buffer not validated
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  const int esize = (int)sizeof(D);
  if (nSrc1Step < oSizeROI.width * Channels * esize) return NPP_STEP_ERROR;
  if (nSrc2Step < oSizeROI.width * Channels * esize) return NPP_STEP_ERROR;

  constexpr int nCh = Channels - (AlphaPreserve ? 1 : 0);
  const int area = oSizeROI.width * oSizeROI.height;
  const int blocks = numBlocks(area);
  double *partial = reinterpret_cast<double *>(pDeviceBuffer);

  (void)cudaGetLastError();
  normPartialKernel<NT, D, Channels, AlphaPreserve, true>
      <<<blocks, kBlock, 0, stream>>>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, oSizeROI.width, oSizeROI.height, partial, nCh);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;

  normFinalizeKernel<NT><<<1, kBlock, 0, stream>>>(partial, blocks, nCh, pNorm);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

template <NormType NT, typename D>
NppStatus runNormMasked(const D *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep, NppiSize oi, Npp8u *pDeviceBuffer,
                        Npp64f *pNorm, cudaStream_t stream) {
  if (!pSrc || !pMask || !pDeviceBuffer || !pNorm) return NPP_NULL_POINTER_ERROR;
  if (oi.width <= 0 || oi.height <= 0) return NPP_SIZE_ERROR;
  if (nSrcStep < oi.width * (int)sizeof(D)) return NPP_STEP_ERROR;
  if (nMaskStep < oi.width) return NPP_STEP_ERROR;

  const int blocks = numBlocks(oi.width * oi.height);
  double *partial = reinterpret_cast<double *>(pDeviceBuffer);

  (void)cudaGetLastError();
  normMaskedKernel<NT, D, 1, false>
      <<<blocks, kBlock, 0, stream>>>(pSrc, nSrcStep, nullptr, 0, pMask, nMaskStep, oi.width, oi.height, 0, partial);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;

  normMaskedFinalizeKernel<NT><<<1, kBlock, 0, stream>>>(partial, blocks, pNorm);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

template <NormType NT, typename D>
NppStatus runNormMaskedCoi(const D *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep, NppiSize oi, int nCOI,
                           Npp8u *pDeviceBuffer, Npp64f *pNorm, cudaStream_t stream) {
  if (!pSrc || !pMask || !pDeviceBuffer || !pNorm) return NPP_NULL_POINTER_ERROR;
  if (oi.width <= 0 || oi.height <= 0) return NPP_SIZE_ERROR;
  if (nCOI < 0 || nCOI > 3) return NPP_COI_ERROR;
  if (nSrcStep < oi.width * 3 * (int)sizeof(D)) return NPP_STEP_ERROR;
  if (nMaskStep < oi.width) return NPP_STEP_ERROR;

  // NVIDIA NPP treats the masked-COI channel as 1-based (channel = nCOI-1);
  // nCOI <= 0 yields no channel and the norm is 0.
  const int srcCh = nCOI - 1;
  if (nCOI <= 0) return zeroDevice(pNorm, stream);

  const int blocks = numBlocks(oi.width * oi.height);
  double *partial = reinterpret_cast<double *>(pDeviceBuffer);

  (void)cudaGetLastError();
  normMaskedKernel<NT, D, 3, false>
      <<<blocks, kBlock, 0, stream>>>(pSrc, nSrcStep, nullptr, 0, pMask, nMaskStep, oi.width, oi.height, srcCh, partial);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;

  normMaskedFinalizeKernel<NT><<<1, kBlock, 0, stream>>>(partial, blocks, pNorm);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

template <NormType NT, typename D>
NppStatus runNormDiffMasked(const D *pSrc1, int nSrc1Step, const D *pSrc2, int nSrc2Step, const Npp8u *pMask,
                            int nMaskStep, NppiSize oi, Npp8u *pDeviceBuffer, Npp64f *pNorm, cudaStream_t stream) {
  if (!pSrc1 || !pSrc2 || !pMask || !pDeviceBuffer || !pNorm) return NPP_NULL_POINTER_ERROR;
  if (oi.width <= 0 || oi.height <= 0) return NPP_SIZE_ERROR;
  if (nSrc1Step < oi.width * (int)sizeof(D)) return NPP_STEP_ERROR;
  if (nSrc2Step < oi.width * (int)sizeof(D)) return NPP_STEP_ERROR;
  if (nMaskStep < oi.width) return NPP_STEP_ERROR;

  const int blocks = numBlocks(oi.width * oi.height);
  double *partial = reinterpret_cast<double *>(pDeviceBuffer);

  (void)cudaGetLastError();
  normMaskedKernel<NT, D, 1, true>
      <<<blocks, kBlock, 0, stream>>>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pMask, nMaskStep, oi.width, oi.height, 0, partial);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;

  normMaskedFinalizeKernel<NT><<<1, kBlock, 0, stream>>>(partial, blocks, pNorm);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

template <NormType NT, typename D>
NppStatus runNormDiffMaskedCoi(const D *pSrc1, int nSrc1Step, const D *pSrc2, int nSrc2Step, const Npp8u *pMask,
                               int nMaskStep, NppiSize oi, int nCOI, Npp8u *pDeviceBuffer, Npp64f *pNorm,
                               cudaStream_t stream) {
  if (!pSrc1 || !pSrc2 || !pMask || !pDeviceBuffer || !pNorm) return NPP_NULL_POINTER_ERROR;
  if (oi.width <= 0 || oi.height <= 0) return NPP_SIZE_ERROR;
  if (nCOI < 0 || nCOI > 3) return NPP_COI_ERROR;
  if (nSrc1Step < oi.width * 3 * (int)sizeof(D)) return NPP_STEP_ERROR;
  if (nSrc2Step < oi.width * 3 * (int)sizeof(D)) return NPP_STEP_ERROR;
  if (nMaskStep < oi.width) return NPP_STEP_ERROR;

  const int srcCh = nCOI - 1;
  if (nCOI <= 0) return zeroDevice(pNorm, stream);

  const int blocks = numBlocks(oi.width * oi.height);
  double *partial = reinterpret_cast<double *>(pDeviceBuffer);

  (void)cudaGetLastError();
  normMaskedKernel<NT, D, 3, true>
      <<<blocks, kBlock, 0, stream>>>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pMask, nMaskStep, oi.width, oi.height, srcCh, partial);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;

  normMaskedFinalizeKernel<NT><<<1, kBlock, 0, stream>>>(partial, blocks, pNorm);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

// ============================== NormRel =====================================

// Positive maximum of the source dtype (NPP_8U_MAX / NPP_16S_MAX / FLT_MAX /
// ...): the value NVIDIA NPP 13.4 returns for a NormRel channel whose
// denominator norm is zero while the numerator is not.
template <typename D> constexpr double normRelZeroDenomValue() {
  return static_cast<double>(std::numeric_limits<D>::max());
}

template <NormType NT, typename D, int Channels, bool AlphaPreserve>
NppStatus runNormRel(const D *pSrc1, int nSrc1Step, const D *pSrc2, int nSrc2Step, NppiSize oSizeROI,
                     Npp8u *pDeviceBuffer, Npp64f *pNorm, cudaStream_t stream) {
  if (!pSrc1 || !pSrc2) return NPP_NULL_POINTER_ERROR;
  if (!pNorm) return NPP_NULL_POINTER_ERROR;               // NVIDIA 5090 validates the result pointer
  if (!pDeviceBuffer) return NPP_NO_ERROR;                 // scratch buffer not validated
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;  // empty ROI is a no-op
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  const int esize = (int)sizeof(D);
  if (nSrc1Step < oSizeROI.width * Channels * esize) return NPP_STEP_ERROR;
  if (nSrc2Step < oSizeROI.width * Channels * esize) return NPP_STEP_ERROR;

  constexpr int nCh = Channels - (AlphaPreserve ? 1 : 0);
  const int area = oSizeROI.width * oSizeROI.height;
  const int blocks = numBlocks(area);
  // Two partial slabs (numerator and denominator), nch columns each.
  double *pDiff = reinterpret_cast<double *>(pDeviceBuffer);
  double *pNormP = pDiff + (size_t)nCh * blocks;

  (void)cudaGetLastError();
  normRelPartialKernel<NT, D, Channels, AlphaPreserve>
      <<<blocks, kBlock, 0, stream>>>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, oSizeROI.width, oSizeROI.height, pDiff, pNormP, nCh);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;

  normRelFinalizeKernel<NT><<<1, kBlock, 0, stream>>>(pDiff, pNormP, blocks, nCh, normRelZeroDenomValue<D>(), pNorm);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

template <NormType NT, typename D>
NppStatus runNormRelMasked(const D *pSrc1, int nSrc1Step, const D *pSrc2, int nSrc2Step, const Npp8u *pMask,
                          int nMaskStep, NppiSize oi, Npp8u *pDeviceBuffer, Npp64f *pNorm, cudaStream_t stream) {
  if (!pSrc1 || !pSrc2 || !pMask || !pDeviceBuffer || !pNorm) return NPP_NULL_POINTER_ERROR;
  if (oi.width <= 0 || oi.height <= 0) return NPP_SIZE_ERROR;
  if (nSrc1Step < oi.width * (int)sizeof(D)) return NPP_STEP_ERROR;
  if (nSrc2Step < oi.width * (int)sizeof(D)) return NPP_STEP_ERROR;
  if (nMaskStep < oi.width) return NPP_STEP_ERROR;

  const int blocks = numBlocks(oi.width * oi.height);
  double *pDiff = reinterpret_cast<double *>(pDeviceBuffer);
  double *pNormP = pDiff + blocks;

  (void)cudaGetLastError();
  normRelMaskedKernel<NT, D, 1>
      <<<blocks, kBlock, 0, stream>>>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pMask, nMaskStep, oi.width, oi.height, 0, pDiff, pNormP);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;

  normRelMaskedFinalizeKernel<NT><<<1, kBlock, 0, stream>>>(pDiff, pNormP, blocks, normRelZeroDenomValue<D>(), pNorm);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

template <NormType NT, typename D>
NppStatus runNormRelMaskedCoi(const D *pSrc1, int nSrc1Step, const D *pSrc2, int nSrc2Step, const Npp8u *pMask,
                             int nMaskStep, NppiSize oi, int nCOI, Npp8u *pDeviceBuffer, Npp64f *pNorm,
                             cudaStream_t stream) {
  if (!pSrc1 || !pSrc2 || !pMask || !pDeviceBuffer || !pNorm) return NPP_NULL_POINTER_ERROR;
  if (oi.width <= 0 || oi.height <= 0) return NPP_SIZE_ERROR;
  if (nCOI < 0 || nCOI > 3) return NPP_COI_ERROR;
  if (nSrc1Step < oi.width * 3 * (int)sizeof(D)) return NPP_STEP_ERROR;
  if (nSrc2Step < oi.width * 3 * (int)sizeof(D)) return NPP_STEP_ERROR;
  if (nMaskStep < oi.width) return NPP_STEP_ERROR;

  const int srcCh = nCOI - 1;
  if (nCOI <= 0) return zeroDevice(pNorm, stream);

  const int blocks = numBlocks(oi.width * oi.height);
  double *pDiff = reinterpret_cast<double *>(pDeviceBuffer);
  double *pNormP = pDiff + blocks;

  (void)cudaGetLastError();
  normRelMaskedKernel<NT, D, 3>
      <<<blocks, kBlock, 0, stream>>>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pMask, nMaskStep, oi.width, oi.height, srcCh, pDiff, pNormP);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;

  normRelMaskedFinalizeKernel<NT><<<1, kBlock, 0, stream>>>(pDiff, pNormP, blocks, normRelZeroDenomValue<D>(), pNorm);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

// ============================== DotProd =====================================

template <typename D, int Channels, bool AlphaPreserve>
NppStatus runDotProd(const D *pSrc1, int nSrc1Step, const D *pSrc2, int nSrc2Step, NppiSize oSizeROI,
                     Npp8u *pDeviceBuffer, Npp64f *pDp, cudaStream_t stream) {
  if (!pSrc1 || !pSrc2) return NPP_NULL_POINTER_ERROR;
  if (!pDp) return NPP_NULL_POINTER_ERROR;                 // NVIDIA 5090 validates the result pointer
  if (!pDeviceBuffer) return NPP_NO_ERROR;                 // scratch buffer not validated
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;  // empty ROI is a no-op
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  const int esize = (int)sizeof(D);
  if (nSrc1Step < oSizeROI.width * Channels * esize) return NPP_STEP_ERROR;
  if (nSrc2Step < oSizeROI.width * Channels * esize) return NPP_STEP_ERROR;

  constexpr int nCh = Channels - (AlphaPreserve ? 1 : 0);
  const int area = oSizeROI.width * oSizeROI.height;
  const int blocks = numBlocks(area);
  double *partial = reinterpret_cast<double *>(pDeviceBuffer);

  (void)cudaGetLastError();
  dotProdPartialKernel<D, Channels, AlphaPreserve>
      <<<blocks, kBlock, 0, stream>>>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, oSizeROI.width, oSizeROI.height, partial, nCh);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;

  dotProdFinalizeKernel<<<1, kBlock, 0, stream>>>(partial, blocks, nCh, pDp);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

// ============================== AverageError ================================
// Validation order and status codes pinned against NVIDIA NPP 13.4 on the
// 5090 (differential tests + step-matrix probes); see nppi_statistics_executor.h.

template <typename D, int Channels, bool Relative>
NppStatus runAverageErrorImpl(const D *pSrc1, int nSrc1Step, const D *pSrc2, int nSrc2Step, NppiSize oSizeROI,
                              Npp8u *pDeviceBuffer, Npp64f *pError, cudaStream_t stream) {
  if (!pSrc1 || !pSrc2) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;  // empty ROI is a no-op
  // NVIDIA 13.4 compares the steps as unsigned 32-bit values against the row
  // size and requires multiples of sizeof(D) * 2^v2(Channels) (C1/C3: esize,
  // C2: 2*esize, C4: 4*esize); each source is checked fully (STEP, then
  // NOT_EVEN) before the next one.  step == 0 -> STEP_ERROR, step == -1
  // passes (huge unsigned), negative steps take the unsigned modulo.
  constexpr int kStepFactor = (Channels % 4 == 0) ? 4 : ((Channels % 2 == 0) ? 2 : 1);
  const Npp32u rowBytes = (Npp32u)oSizeROI.width * (Npp32u)Channels * (Npp32u)sizeof(D);
  const Npp32u align = (Npp32u)sizeof(D) * (Npp32u)kStepFactor;
  if ((Npp32u)nSrc1Step < rowBytes) return NPP_STEP_ERROR;
  if ((Npp32u)nSrc1Step % align != 0) return NPP_NOT_EVEN_STEP_ERROR;
  if ((Npp32u)nSrc2Step < rowBytes) return NPP_STEP_ERROR;
  if ((Npp32u)nSrc2Step % align != 0) return NPP_NOT_EVEN_STEP_ERROR;
  if (!pError) return NPP_NULL_POINTER_ERROR;
  if (!pDeviceBuffer) return NPP_NULL_POINTER_ERROR;  // scratch IS validated (unlike Sum/Mean)

  const int area = oSizeROI.width * oSizeROI.height;
  const int blocks = numBlocks(area);
  double *partial = reinterpret_cast<double *>(pDeviceBuffer);

  (void)cudaGetLastError();
  avgErrPartialKernel<D, Channels, Relative>
      <<<blocks, kBlock, 0, stream>>>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, oSizeROI.width, oSizeROI.height, partial);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;

  avgErrFinalizeKernel<<<1, kBlock, 0, stream>>>(
      partial, blocks, (double)oSizeROI.width * (double)oSizeROI.height * (double)Channels, pError);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

template <typename D, int Channels>
NppStatus runAverageError(const D *pSrc1, int nSrc1Step, const D *pSrc2, int nSrc2Step, NppiSize oSizeROI,
                          Npp8u *pDeviceBuffer, Npp64f *pError, cudaStream_t stream) {
  return runAverageErrorImpl<D, Channels, false>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, oSizeROI, pDeviceBuffer, pError,
                                                 stream);
}

template <typename D, int Channels>
NppStatus runAverageRelativeError(const D *pSrc1, int nSrc1Step, const D *pSrc2, int nSrc2Step, NppiSize oSizeROI,
                                  Npp8u *pDeviceBuffer, Npp64f *pError, cudaStream_t stream) {
  return runAverageErrorImpl<D, Channels, true>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, oSizeROI, pDeviceBuffer, pError,
                                                stream);
}

// ============================== Buffer size ==================================

NppStatus sumGetBufferHostSize(NppiSize oi, int *hpBufferSize) {
  if (!hpBufferSize) return NPP_NULL_POINTER_ERROR;
  if (oi.width <= 0 || oi.height <= 0) return NPP_SIZE_ERROR;
  *hpBufferSize = static_cast<int>(numBlocks(oi.width * oi.height) * 4 * (int)sizeof(Npp64f));
  return NPP_NO_ERROR;
}

NppStatus meanGetBufferHostSize(NppiSize oi, int *hpBufferSize) {
  if (!hpBufferSize) return NPP_NULL_POINTER_ERROR;
  if (oi.width <= 0 || oi.height <= 0) return NPP_SIZE_ERROR;
  *hpBufferSize = static_cast<int>(numBlocks(oi.width * oi.height) * 4 * (int)sizeof(Npp64f));
  return NPP_NO_ERROR;
}

NppStatus meanStdDevGetBufferHostSize(NppiSize oi, int *hpBufferSize) {
  if (!hpBufferSize) return NPP_NULL_POINTER_ERROR;
  if (oi.width <= 0 || oi.height <= 0) return NPP_SIZE_ERROR;
  *hpBufferSize = static_cast<int>(numBlocks(oi.width * oi.height) * 3 * (int)sizeof(double));
  return NPP_NO_ERROR;
}

// Per-channel min or max fits in blocks * 4 elements of the (single) output
// dtype; MinMax additionally needs a second equally sized slab.  Exact bytes
// are not a contract, only that the returned size is sufficient.
NppStatus minGetBufferHostSize(NppiSize oi, int *hpBufferSize) {
  if (!hpBufferSize) return NPP_NULL_POINTER_ERROR;
  if (oi.width <= 0 || oi.height <= 0) return NPP_SIZE_ERROR;
  *hpBufferSize = static_cast<int>(numBlocks(oi.width * oi.height) * 4 * (int)sizeof(Npp64f));
  return NPP_NO_ERROR;
}

NppStatus maxGetBufferHostSize(NppiSize oi, int *hpBufferSize) {
  return minGetBufferHostSize(oi, hpBufferSize);
}

NppStatus minMaxGetBufferHostSize(NppiSize oi, int *hpBufferSize) {
  if (!hpBufferSize) return NPP_NULL_POINTER_ERROR;
  if (oi.width <= 0 || oi.height <= 0) return NPP_SIZE_ERROR;
  *hpBufferSize = static_cast<int>(numBlocks(oi.width * oi.height) * 8 * (int)sizeof(Npp64f));
  return NPP_NO_ERROR;
}

NppStatus minIndxGetBufferHostSize(NppiSize oi, int *hpBufferSize) {
  if (!hpBufferSize) return NPP_NULL_POINTER_ERROR;
  if (oi.width <= 0 || oi.height <= 0) return NPP_SIZE_ERROR;
  *hpBufferSize = static_cast<int>(numBlocks(oi.width * oi.height) * 8 * (int)sizeof(Npp64f));
  return NPP_NO_ERROR;
}

NppStatus maxIndxGetBufferHostSize(NppiSize oi, int *hpBufferSize) {
  return minIndxGetBufferHostSize(oi, hpBufferSize);
}

// min and max block partials each need (value D + x int + y int); a safe upper
// bound is 2 slabs of block-entries of 4 int-sized fields.
NppStatus minMaxIndxGetBufferHostSize(NppiSize oi, int *hpBufferSize) {
  if (!hpBufferSize) return NPP_NULL_POINTER_ERROR;
  if (oi.width <= 0 || oi.height <= 0) return NPP_SIZE_ERROR;
  *hpBufferSize = static_cast<int>(numBlocks(oi.width * oi.height) * 8 * (int)sizeof(Npp64f));
  return NPP_NO_ERROR;
}

// Norm partials hold at most 3 channel doubles per block; a 4-double slab is a
// safe upper bound for both plain and masked reductions.  Exact bytes are not a
// contract, only that the returned size is sufficient.
NppStatus normL1GetBufferHostSize(NppiSize oi, int *hpBufferSize) {
  if (!hpBufferSize) return NPP_NULL_POINTER_ERROR;
  if (oi.width <= 0 || oi.height <= 0) return NPP_SIZE_ERROR;
  *hpBufferSize = static_cast<int>(numBlocks(oi.width * oi.height) * 4 * (int)sizeof(Npp64f));
  return NPP_NO_ERROR;
}

NppStatus normL2GetBufferHostSize(NppiSize oi, int *hpBufferSize) { return normL1GetBufferHostSize(oi, hpBufferSize); }
NppStatus normInfGetBufferHostSize(NppiSize oi, int *hpBufferSize) { return normL1GetBufferHostSize(oi, hpBufferSize); }
NppStatus normDiffL1GetBufferHostSize(NppiSize oi, int *hpBufferSize) { return normL1GetBufferHostSize(oi, hpBufferSize); }
NppStatus normDiffL2GetBufferHostSize(NppiSize oi, int *hpBufferSize) { return normL1GetBufferHostSize(oi, hpBufferSize); }
NppStatus normDiffInfGetBufferHostSize(NppiSize oi, int *hpBufferSize) { return normL1GetBufferHostSize(oi, hpBufferSize); }

// NormRel needs two partial slabs (numerator and denominator), each holding at
// most 3 channel doubles per block; an 8-double slab per block is a safe upper
// bound.  Exact bytes are not a contract, only that the size is sufficient.
NppStatus normRelL1GetBufferHostSize(NppiSize oi, int *hpBufferSize) {
  if (!hpBufferSize) return NPP_NULL_POINTER_ERROR;
  if (oi.width <= 0 || oi.height <= 0) return NPP_SIZE_ERROR;
  *hpBufferSize = static_cast<int>(numBlocks(oi.width * oi.height) * 8 * (int)sizeof(Npp64f));
  return NPP_NO_ERROR;
}
NppStatus normRelL2GetBufferHostSize(NppiSize oi, int *hpBufferSize) { return normRelL1GetBufferHostSize(oi, hpBufferSize); }
NppStatus normRelInfGetBufferHostSize(NppiSize oi, int *hpBufferSize) { return normRelL1GetBufferHostSize(oi, hpBufferSize); }

// DotProd partials hold at most 4 channel doubles per block.
NppStatus dotProdGetBufferHostSize(NppiSize oi, int *hpBufferSize) {
  if (!hpBufferSize) return NPP_NULL_POINTER_ERROR;
  if (oi.width <= 0 || oi.height <= 0) return NPP_SIZE_ERROR;
  *hpBufferSize = static_cast<int>(numBlocks(oi.width * oi.height) * 4 * (int)sizeof(Npp64f));
  return NPP_NO_ERROR;
}

// CrossCorr NormLevel scratch-size helpers.  These reserve a conservative scratch
// large enough for a same-size NormLevel reduction on the ROI; exact bytes are
// not a contract (the only guarantee is the buffer can be allocated and the
// matching kernel then succeeds).  Only status and size positivity are checked.
NppStatus fullNormLevelGetBufferHostSize(NppiSize oi, int *hpBufferSize) {
  if (!hpBufferSize) return NPP_NULL_POINTER_ERROR;
  if (oi.width <= 0 || oi.height <= 0) return NPP_SIZE_ERROR;
  *hpBufferSize = static_cast<int>(numBlocks(oi.width * oi.height) * 8 * (int)sizeof(Npp64f));
  return NPP_NO_ERROR;
}
NppStatus sameNormLevelGetBufferHostSize(NppiSize oi, int *hpBufferSize) {
  return fullNormLevelGetBufferHostSize(oi, hpBufferSize);
}
NppStatus validNormLevelGetBufferHostSize(NppiSize oi, int *hpBufferSize) {
  return fullNormLevelGetBufferHostSize(oi, hpBufferSize);
}

// Unlike every other helper above, NVIDIA 13.4 accepts a zero-sized ROI here
// and returns NPP_NO_ERROR without writing anything (5090-measured); negative
// sizes are still NPP_SIZE_ERROR.  One scalar double partial per block.
NppStatus averageErrorGetBufferHostSize(NppiSize oi, int *hpBufferSize) {
  if (!hpBufferSize) return NPP_NULL_POINTER_ERROR;
  if (oi.width < 0 || oi.height < 0) return NPP_SIZE_ERROR;
  if (oi.width == 0 || oi.height == 0) return NPP_NO_ERROR;
  *hpBufferSize = static_cast<int>(numBlocks(oi.width * oi.height) * (int)sizeof(Npp64f));
  return NPP_NO_ERROR;
}
NppStatus averageRelativeErrorGetBufferHostSize(NppiSize oi, int *hpBufferSize) {
  return averageErrorGetBufferHostSize(oi, hpBufferSize);
}

}  // namespace statistics
}  // namespace nppi


// Explicit instantiations for all variants listed in api_analysis/packages/p3.1a1.csv.
namespace nppi {
namespace statistics {
template NppStatus runSum<Npp8u,1,Npp64f,false>(const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runSum<Npp8u,3,Npp64f,false>(const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runSum<Npp8u,4,Npp64f,false>(const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runSum<Npp8u,4,Npp64f,true>(const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runSum<Npp16u,1,Npp64f,false>(const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runSum<Npp16u,3,Npp64f,false>(const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runSum<Npp16u,4,Npp64f,false>(const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runSum<Npp16u,4,Npp64f,true>(const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runSum<Npp16s,1,Npp64f,false>(const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runSum<Npp16s,3,Npp64f,false>(const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runSum<Npp16s,4,Npp64f,false>(const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runSum<Npp16s,4,Npp64f,true>(const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runSum<Npp32f,1,Npp64f,false>(const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runSum<Npp32f,3,Npp64f,false>(const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runSum<Npp32f,4,Npp64f,false>(const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runSum<Npp32f,4,Npp64f,true>(const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runSum<Npp8u,1,Npp64s,false>(const Npp8u*,int,NppiSize,Npp8u*,Npp64s*,cudaStream_t);
template NppStatus runSum<Npp8u,4,Npp64s,false>(const Npp8u*,int,NppiSize,Npp8u*,Npp64s*,cudaStream_t);
template NppStatus runMean<Npp8u,1,false>(const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runMean<Npp8u,3,false>(const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runMean<Npp8u,4,false>(const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runMean<Npp8u,4,true>(const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runMean<Npp16u,1,false>(const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runMean<Npp16u,3,false>(const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runMean<Npp16u,4,false>(const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runMean<Npp16u,4,true>(const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runMean<Npp16s,1,false>(const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runMean<Npp16s,3,false>(const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runMean<Npp16s,4,false>(const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runMean<Npp16s,4,true>(const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runMean<Npp32f,1,false>(const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runMean<Npp32f,3,false>(const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runMean<Npp32f,4,false>(const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runMean<Npp32f,4,true>(const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runMeanMasked<Npp8u>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runMeanMaskedCoi<Npp8u>(const Npp8u*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runMeanStdDev<Npp8u>(const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,Npp64f*,cudaStream_t);
template NppStatus runMeanStdDevCoi<Npp8u>(const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,Npp64f*,cudaStream_t);
template NppStatus runMeanStdDevMasked<Npp8u>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,Npp64f*,cudaStream_t);
template NppStatus runMeanStdDevMaskedCoi<Npp8u>(const Npp8u*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,Npp64f*,cudaStream_t);
template NppStatus runMeanMasked<Npp16u>(const Npp16u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runMeanMaskedCoi<Npp16u>(const Npp16u*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runMeanStdDev<Npp16u>(const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,Npp64f*,cudaStream_t);
template NppStatus runMeanStdDevCoi<Npp16u>(const Npp16u*,int,NppiSize,int,Npp8u*,Npp64f*,Npp64f*,cudaStream_t);
template NppStatus runMeanStdDevMasked<Npp16u>(const Npp16u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,Npp64f*,cudaStream_t);
template NppStatus runMeanStdDevMaskedCoi<Npp16u>(const Npp16u*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,Npp64f*,cudaStream_t);
template NppStatus runMeanMasked<Npp32f>(const Npp32f*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runMeanMaskedCoi<Npp32f>(const Npp32f*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runMeanStdDev<Npp32f>(const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,Npp64f*,cudaStream_t);
template NppStatus runMeanStdDevCoi<Npp32f>(const Npp32f*,int,NppiSize,int,Npp8u*,Npp64f*,Npp64f*,cudaStream_t);
template NppStatus runMeanStdDevMasked<Npp32f>(const Npp32f*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,Npp64f*,cudaStream_t);
template NppStatus runMeanStdDevMaskedCoi<Npp32f>(const Npp32f*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,Npp64f*,cudaStream_t);
template NppStatus runMeanMasked<Npp8s>(const Npp8s*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runMeanMaskedCoi<Npp8s>(const Npp8s*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runMeanStdDev<Npp8s>(const Npp8s*,int,NppiSize,Npp8u*,Npp64f*,Npp64f*,cudaStream_t);
template NppStatus runMeanStdDevCoi<Npp8s>(const Npp8s*,int,NppiSize,int,Npp8u*,Npp64f*,Npp64f*,cudaStream_t);
template NppStatus runMeanStdDevMasked<Npp8s>(const Npp8s*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,Npp64f*,cudaStream_t);
template NppStatus runMeanStdDevMaskedCoi<Npp8s>(const Npp8s*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,Npp64f*,cudaStream_t);

// --- Min / Max / MinMax: 8u, 16u, 16s, 32f x {C1, C3, C4, AC4} ---
template NppStatus runMin<Npp8u,1,false>(const Npp8u*,int,NppiSize,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runMin<Npp8u,3,false>(const Npp8u*,int,NppiSize,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runMin<Npp8u,4,false>(const Npp8u*,int,NppiSize,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runMin<Npp8u,4,true>(const Npp8u*,int,NppiSize,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runMin<Npp16u,1,false>(const Npp16u*,int,NppiSize,Npp8u*,Npp16u*,cudaStream_t);
template NppStatus runMin<Npp16u,3,false>(const Npp16u*,int,NppiSize,Npp8u*,Npp16u*,cudaStream_t);
template NppStatus runMin<Npp16u,4,false>(const Npp16u*,int,NppiSize,Npp8u*,Npp16u*,cudaStream_t);
template NppStatus runMin<Npp16u,4,true>(const Npp16u*,int,NppiSize,Npp8u*,Npp16u*,cudaStream_t);
template NppStatus runMin<Npp16s,1,false>(const Npp16s*,int,NppiSize,Npp8u*,Npp16s*,cudaStream_t);
template NppStatus runMin<Npp16s,3,false>(const Npp16s*,int,NppiSize,Npp8u*,Npp16s*,cudaStream_t);
template NppStatus runMin<Npp16s,4,false>(const Npp16s*,int,NppiSize,Npp8u*,Npp16s*,cudaStream_t);
template NppStatus runMin<Npp16s,4,true>(const Npp16s*,int,NppiSize,Npp8u*,Npp16s*,cudaStream_t);
template NppStatus runMin<Npp32f,1,false>(const Npp32f*,int,NppiSize,Npp8u*,Npp32f*,cudaStream_t);
template NppStatus runMin<Npp32f,3,false>(const Npp32f*,int,NppiSize,Npp8u*,Npp32f*,cudaStream_t);
template NppStatus runMin<Npp32f,4,false>(const Npp32f*,int,NppiSize,Npp8u*,Npp32f*,cudaStream_t);
template NppStatus runMin<Npp32f,4,true>(const Npp32f*,int,NppiSize,Npp8u*,Npp32f*,cudaStream_t);
template NppStatus runMax<Npp8u,1,false>(const Npp8u*,int,NppiSize,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runMax<Npp8u,3,false>(const Npp8u*,int,NppiSize,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runMax<Npp8u,4,false>(const Npp8u*,int,NppiSize,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runMax<Npp8u,4,true>(const Npp8u*,int,NppiSize,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runMax<Npp16u,1,false>(const Npp16u*,int,NppiSize,Npp8u*,Npp16u*,cudaStream_t);
template NppStatus runMax<Npp16u,3,false>(const Npp16u*,int,NppiSize,Npp8u*,Npp16u*,cudaStream_t);
template NppStatus runMax<Npp16u,4,false>(const Npp16u*,int,NppiSize,Npp8u*,Npp16u*,cudaStream_t);
template NppStatus runMax<Npp16u,4,true>(const Npp16u*,int,NppiSize,Npp8u*,Npp16u*,cudaStream_t);
template NppStatus runMax<Npp16s,1,false>(const Npp16s*,int,NppiSize,Npp8u*,Npp16s*,cudaStream_t);
template NppStatus runMax<Npp16s,3,false>(const Npp16s*,int,NppiSize,Npp8u*,Npp16s*,cudaStream_t);
template NppStatus runMax<Npp16s,4,false>(const Npp16s*,int,NppiSize,Npp8u*,Npp16s*,cudaStream_t);
template NppStatus runMax<Npp16s,4,true>(const Npp16s*,int,NppiSize,Npp8u*,Npp16s*,cudaStream_t);
template NppStatus runMax<Npp32f,1,false>(const Npp32f*,int,NppiSize,Npp8u*,Npp32f*,cudaStream_t);
template NppStatus runMax<Npp32f,3,false>(const Npp32f*,int,NppiSize,Npp8u*,Npp32f*,cudaStream_t);
template NppStatus runMax<Npp32f,4,false>(const Npp32f*,int,NppiSize,Npp8u*,Npp32f*,cudaStream_t);
template NppStatus runMax<Npp32f,4,true>(const Npp32f*,int,NppiSize,Npp8u*,Npp32f*,cudaStream_t);
template NppStatus runMinMax<Npp8u,1,false>(const Npp8u*,int,NppiSize,Npp8u*,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runMinMax<Npp8u,3,false>(const Npp8u*,int,NppiSize,Npp8u*,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runMinMax<Npp8u,4,false>(const Npp8u*,int,NppiSize,Npp8u*,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runMinMax<Npp8u,4,true>(const Npp8u*,int,NppiSize,Npp8u*,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runMinMax<Npp16u,1,false>(const Npp16u*,int,NppiSize,Npp8u*,Npp16u*,Npp16u*,cudaStream_t);
template NppStatus runMinMax<Npp16u,3,false>(const Npp16u*,int,NppiSize,Npp8u*,Npp16u*,Npp16u*,cudaStream_t);
template NppStatus runMinMax<Npp16u,4,false>(const Npp16u*,int,NppiSize,Npp8u*,Npp16u*,Npp16u*,cudaStream_t);
template NppStatus runMinMax<Npp16u,4,true>(const Npp16u*,int,NppiSize,Npp8u*,Npp16u*,Npp16u*,cudaStream_t);
template NppStatus runMinMax<Npp16s,1,false>(const Npp16s*,int,NppiSize,Npp8u*,Npp16s*,Npp16s*,cudaStream_t);
template NppStatus runMinMax<Npp16s,3,false>(const Npp16s*,int,NppiSize,Npp8u*,Npp16s*,Npp16s*,cudaStream_t);
template NppStatus runMinMax<Npp16s,4,false>(const Npp16s*,int,NppiSize,Npp8u*,Npp16s*,Npp16s*,cudaStream_t);
template NppStatus runMinMax<Npp16s,4,true>(const Npp16s*,int,NppiSize,Npp8u*,Npp16s*,Npp16s*,cudaStream_t);
template NppStatus runMinMax<Npp32f,1,false>(const Npp32f*,int,NppiSize,Npp8u*,Npp32f*,Npp32f*,cudaStream_t);
template NppStatus runMinMax<Npp32f,3,false>(const Npp32f*,int,NppiSize,Npp8u*,Npp32f*,Npp32f*,cudaStream_t);
template NppStatus runMinMax<Npp32f,4,false>(const Npp32f*,int,NppiSize,Npp8u*,Npp32f*,Npp32f*,cudaStream_t);
template NppStatus runMinMax<Npp32f,4,true>(const Npp32f*,int,NppiSize,Npp8u*,Npp32f*,Npp32f*,cudaStream_t);

// --- MinIndx / MaxIndx: 8u, 16u, 16s, 32f x {C1, C3, C4, AC4} x {min, max} ---
template NppStatus runExtrIndx<Npp8u,1,false,true>(const Npp8u*,int,NppiSize,Npp8u*,Npp8u*,int*,int*,cudaStream_t);
template NppStatus runExtrIndx<Npp8u,3,false,true>(const Npp8u*,int,NppiSize,Npp8u*,Npp8u*,int*,int*,cudaStream_t);
template NppStatus runExtrIndx<Npp8u,4,false,true>(const Npp8u*,int,NppiSize,Npp8u*,Npp8u*,int*,int*,cudaStream_t);
template NppStatus runExtrIndx<Npp8u,4,true,true>(const Npp8u*,int,NppiSize,Npp8u*,Npp8u*,int*,int*,cudaStream_t);
template NppStatus runExtrIndx<Npp16u,1,false,true>(const Npp16u*,int,NppiSize,Npp8u*,Npp16u*,int*,int*,cudaStream_t);
template NppStatus runExtrIndx<Npp16u,3,false,true>(const Npp16u*,int,NppiSize,Npp8u*,Npp16u*,int*,int*,cudaStream_t);
template NppStatus runExtrIndx<Npp16u,4,false,true>(const Npp16u*,int,NppiSize,Npp8u*,Npp16u*,int*,int*,cudaStream_t);
template NppStatus runExtrIndx<Npp16u,4,true,true>(const Npp16u*,int,NppiSize,Npp8u*,Npp16u*,int*,int*,cudaStream_t);
template NppStatus runExtrIndx<Npp16s,1,false,true>(const Npp16s*,int,NppiSize,Npp8u*,Npp16s*,int*,int*,cudaStream_t);
template NppStatus runExtrIndx<Npp16s,3,false,true>(const Npp16s*,int,NppiSize,Npp8u*,Npp16s*,int*,int*,cudaStream_t);
template NppStatus runExtrIndx<Npp16s,4,false,true>(const Npp16s*,int,NppiSize,Npp8u*,Npp16s*,int*,int*,cudaStream_t);
template NppStatus runExtrIndx<Npp16s,4,true,true>(const Npp16s*,int,NppiSize,Npp8u*,Npp16s*,int*,int*,cudaStream_t);
template NppStatus runExtrIndx<Npp32f,1,false,true>(const Npp32f*,int,NppiSize,Npp8u*,Npp32f*,int*,int*,cudaStream_t);
template NppStatus runExtrIndx<Npp32f,3,false,true>(const Npp32f*,int,NppiSize,Npp8u*,Npp32f*,int*,int*,cudaStream_t);
template NppStatus runExtrIndx<Npp32f,4,false,true>(const Npp32f*,int,NppiSize,Npp8u*,Npp32f*,int*,int*,cudaStream_t);
template NppStatus runExtrIndx<Npp32f,4,true,true>(const Npp32f*,int,NppiSize,Npp8u*,Npp32f*,int*,int*,cudaStream_t);
template NppStatus runExtrIndx<Npp8u,1,false,false>(const Npp8u*,int,NppiSize,Npp8u*,Npp8u*,int*,int*,cudaStream_t);
template NppStatus runExtrIndx<Npp8u,3,false,false>(const Npp8u*,int,NppiSize,Npp8u*,Npp8u*,int*,int*,cudaStream_t);
template NppStatus runExtrIndx<Npp8u,4,false,false>(const Npp8u*,int,NppiSize,Npp8u*,Npp8u*,int*,int*,cudaStream_t);
template NppStatus runExtrIndx<Npp8u,4,true,false>(const Npp8u*,int,NppiSize,Npp8u*,Npp8u*,int*,int*,cudaStream_t);
template NppStatus runExtrIndx<Npp16u,1,false,false>(const Npp16u*,int,NppiSize,Npp8u*,Npp16u*,int*,int*,cudaStream_t);
template NppStatus runExtrIndx<Npp16u,3,false,false>(const Npp16u*,int,NppiSize,Npp8u*,Npp16u*,int*,int*,cudaStream_t);
template NppStatus runExtrIndx<Npp16u,4,false,false>(const Npp16u*,int,NppiSize,Npp8u*,Npp16u*,int*,int*,cudaStream_t);
template NppStatus runExtrIndx<Npp16u,4,true,false>(const Npp16u*,int,NppiSize,Npp8u*,Npp16u*,int*,int*,cudaStream_t);
template NppStatus runExtrIndx<Npp16s,1,false,false>(const Npp16s*,int,NppiSize,Npp8u*,Npp16s*,int*,int*,cudaStream_t);
template NppStatus runExtrIndx<Npp16s,3,false,false>(const Npp16s*,int,NppiSize,Npp8u*,Npp16s*,int*,int*,cudaStream_t);
template NppStatus runExtrIndx<Npp16s,4,false,false>(const Npp16s*,int,NppiSize,Npp8u*,Npp16s*,int*,int*,cudaStream_t);
template NppStatus runExtrIndx<Npp16s,4,true,false>(const Npp16s*,int,NppiSize,Npp8u*,Npp16s*,int*,int*,cudaStream_t);
template NppStatus runExtrIndx<Npp32f,1,false,false>(const Npp32f*,int,NppiSize,Npp8u*,Npp32f*,int*,int*,cudaStream_t);
template NppStatus runExtrIndx<Npp32f,3,false,false>(const Npp32f*,int,NppiSize,Npp8u*,Npp32f*,int*,int*,cudaStream_t);
template NppStatus runExtrIndx<Npp32f,4,false,false>(const Npp32f*,int,NppiSize,Npp8u*,Npp32f*,int*,int*,cudaStream_t);
template NppStatus runExtrIndx<Npp32f,4,true,false>(const Npp32f*,int,NppiSize,Npp8u*,Npp32f*,int*,int*,cudaStream_t);

// --- MinMaxIndx: 16u, 32f, 8s, 8u x {C1 (C1R/C1MR), C3 (C3CR/C3CMR)} ---
template NppStatus runMinMaxIndx<Npp8u,1>(const Npp8u*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp8u*,NppiPoint*,NppiPoint*,Npp8u*,cudaStream_t);
template NppStatus runMinMaxIndx<Npp8u,3>(const Npp8u*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp8u*,NppiPoint*,NppiPoint*,Npp8u*,cudaStream_t);
template NppStatus runMinMaxIndx<Npp8s,1>(const Npp8s*,int,const Npp8u*,int,NppiSize,int,Npp8s*,Npp8s*,NppiPoint*,NppiPoint*,Npp8u*,cudaStream_t);
template NppStatus runMinMaxIndx<Npp8s,3>(const Npp8s*,int,const Npp8u*,int,NppiSize,int,Npp8s*,Npp8s*,NppiPoint*,NppiPoint*,Npp8u*,cudaStream_t);
template NppStatus runMinMaxIndx<Npp16u,1>(const Npp16u*,int,const Npp8u*,int,NppiSize,int,Npp16u*,Npp16u*,NppiPoint*,NppiPoint*,Npp8u*,cudaStream_t);
template NppStatus runMinMaxIndx<Npp16u,3>(const Npp16u*,int,const Npp8u*,int,NppiSize,int,Npp16u*,Npp16u*,NppiPoint*,NppiPoint*,Npp8u*,cudaStream_t);
template NppStatus runMinMaxIndx<Npp32f,1>(const Npp32f*,int,const Npp8u*,int,NppiSize,int,Npp32f*,Npp32f*,NppiPoint*,NppiPoint*,Npp8u*,cudaStream_t);
template NppStatus runMinMaxIndx<Npp32f,3>(const Npp32f*,int,const Npp8u*,int,NppiSize,int,Npp32f*,Npp32f*,NppiPoint*,NppiPoint*,Npp8u*,cudaStream_t);
// --- Norm / NormDiff explicit instantiations (package p3.1b1) ---
template NppStatus runNorm<nppi::statistics::NormType::L1,Npp8u,1,0>(const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::L1,Npp8u,3,0>(const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::L1,Npp8u,4,0>(const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::L1,Npp8u,4,1>(const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::L1,Npp16u,1,0>(const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::L1,Npp16u,3,0>(const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::L1,Npp16u,4,0>(const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::L1,Npp16u,4,1>(const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::L1,Npp16s,1,0>(const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::L1,Npp16s,3,0>(const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::L1,Npp16s,4,0>(const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::L1,Npp16s,4,1>(const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::L1,Npp32f,1,0>(const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::L1,Npp32f,3,0>(const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::L1,Npp32f,4,0>(const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::L1,Npp32f,4,1>(const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::L1,Npp32s,1,0>(const Npp32s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::L2,Npp8u,1,0>(const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::L2,Npp8u,3,0>(const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::L2,Npp8u,4,0>(const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::L2,Npp8u,4,1>(const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::L2,Npp16u,1,0>(const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::L2,Npp16u,3,0>(const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::L2,Npp16u,4,0>(const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::L2,Npp16u,4,1>(const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::L2,Npp16s,1,0>(const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::L2,Npp16s,3,0>(const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::L2,Npp16s,4,0>(const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::L2,Npp16s,4,1>(const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::L2,Npp32f,1,0>(const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::L2,Npp32f,3,0>(const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::L2,Npp32f,4,0>(const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::L2,Npp32f,4,1>(const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::L2,Npp32s,1,0>(const Npp32s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::Inf,Npp8u,1,0>(const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::Inf,Npp8u,3,0>(const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::Inf,Npp8u,4,0>(const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::Inf,Npp8u,4,1>(const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::Inf,Npp16u,1,0>(const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::Inf,Npp16u,3,0>(const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::Inf,Npp16u,4,0>(const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::Inf,Npp16u,4,1>(const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::Inf,Npp16s,1,0>(const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::Inf,Npp16s,3,0>(const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::Inf,Npp16s,4,0>(const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::Inf,Npp16s,4,1>(const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::Inf,Npp32f,1,0>(const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::Inf,Npp32f,3,0>(const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::Inf,Npp32f,4,0>(const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::Inf,Npp32f,4,1>(const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNorm<nppi::statistics::NormType::Inf,Npp32s,1,0>(const Npp32s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::L1,Npp8u,1,0>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::L1,Npp8u,3,0>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::L1,Npp8u,4,0>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::L1,Npp8u,4,1>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::L1,Npp16u,1,0>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::L1,Npp16u,3,0>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::L1,Npp16u,4,0>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::L1,Npp16u,4,1>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::L1,Npp16s,1,0>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::L1,Npp16s,3,0>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::L1,Npp16s,4,0>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::L1,Npp16s,4,1>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::L1,Npp32f,1,0>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::L1,Npp32f,3,0>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::L1,Npp32f,4,0>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::L1,Npp32f,4,1>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::L1,Npp32s,1,0>(const Npp32s*,int,const Npp32s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::L2,Npp8u,1,0>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::L2,Npp8u,3,0>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::L2,Npp8u,4,0>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::L2,Npp8u,4,1>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::L2,Npp16u,1,0>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::L2,Npp16u,3,0>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::L2,Npp16u,4,0>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::L2,Npp16u,4,1>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::L2,Npp16s,1,0>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::L2,Npp16s,3,0>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::L2,Npp16s,4,0>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::L2,Npp16s,4,1>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::L2,Npp32f,1,0>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::L2,Npp32f,3,0>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::L2,Npp32f,4,0>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::L2,Npp32f,4,1>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::L2,Npp32s,1,0>(const Npp32s*,int,const Npp32s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::Inf,Npp8u,1,0>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::Inf,Npp8u,3,0>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::Inf,Npp8u,4,0>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::Inf,Npp8u,4,1>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::Inf,Npp16u,1,0>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::Inf,Npp16u,3,0>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::Inf,Npp16u,4,0>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::Inf,Npp16u,4,1>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::Inf,Npp16s,1,0>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::Inf,Npp16s,3,0>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::Inf,Npp16s,4,0>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::Inf,Npp16s,4,1>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::Inf,Npp32f,1,0>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::Inf,Npp32f,3,0>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::Inf,Npp32f,4,0>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::Inf,Npp32f,4,1>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiff<nppi::statistics::NormType::Inf,Npp32s,1,0>(const Npp32s*,int,const Npp32s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormMasked<nppi::statistics::NormType::L1,Npp8u>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormMaskedCoi<nppi::statistics::NormType::L1,Npp8u>(const Npp8u*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiffMasked<nppi::statistics::NormType::L1,Npp8u>(const Npp8u*,int,const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiffMaskedCoi<nppi::statistics::NormType::L1,Npp8u>(const Npp8u*,int,const Npp8u*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormMasked<nppi::statistics::NormType::L1,Npp16u>(const Npp16u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormMaskedCoi<nppi::statistics::NormType::L1,Npp16u>(const Npp16u*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiffMasked<nppi::statistics::NormType::L1,Npp16u>(const Npp16u*,int,const Npp16u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiffMaskedCoi<nppi::statistics::NormType::L1,Npp16u>(const Npp16u*,int,const Npp16u*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormMasked<nppi::statistics::NormType::L1,Npp32f>(const Npp32f*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormMaskedCoi<nppi::statistics::NormType::L1,Npp32f>(const Npp32f*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiffMasked<nppi::statistics::NormType::L1,Npp32f>(const Npp32f*,int,const Npp32f*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiffMaskedCoi<nppi::statistics::NormType::L1,Npp32f>(const Npp32f*,int,const Npp32f*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormMasked<nppi::statistics::NormType::L1,Npp8s>(const Npp8s*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormMaskedCoi<nppi::statistics::NormType::L1,Npp8s>(const Npp8s*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiffMasked<nppi::statistics::NormType::L1,Npp8s>(const Npp8s*,int,const Npp8s*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiffMaskedCoi<nppi::statistics::NormType::L1,Npp8s>(const Npp8s*,int,const Npp8s*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormMasked<nppi::statistics::NormType::L2,Npp8u>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormMaskedCoi<nppi::statistics::NormType::L2,Npp8u>(const Npp8u*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiffMasked<nppi::statistics::NormType::L2,Npp8u>(const Npp8u*,int,const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiffMaskedCoi<nppi::statistics::NormType::L2,Npp8u>(const Npp8u*,int,const Npp8u*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormMasked<nppi::statistics::NormType::L2,Npp16u>(const Npp16u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormMaskedCoi<nppi::statistics::NormType::L2,Npp16u>(const Npp16u*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiffMasked<nppi::statistics::NormType::L2,Npp16u>(const Npp16u*,int,const Npp16u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiffMaskedCoi<nppi::statistics::NormType::L2,Npp16u>(const Npp16u*,int,const Npp16u*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormMasked<nppi::statistics::NormType::L2,Npp32f>(const Npp32f*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormMaskedCoi<nppi::statistics::NormType::L2,Npp32f>(const Npp32f*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiffMasked<nppi::statistics::NormType::L2,Npp32f>(const Npp32f*,int,const Npp32f*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiffMaskedCoi<nppi::statistics::NormType::L2,Npp32f>(const Npp32f*,int,const Npp32f*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormMasked<nppi::statistics::NormType::L2,Npp8s>(const Npp8s*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormMaskedCoi<nppi::statistics::NormType::L2,Npp8s>(const Npp8s*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiffMasked<nppi::statistics::NormType::L2,Npp8s>(const Npp8s*,int,const Npp8s*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiffMaskedCoi<nppi::statistics::NormType::L2,Npp8s>(const Npp8s*,int,const Npp8s*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormMasked<nppi::statistics::NormType::Inf,Npp8u>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormMaskedCoi<nppi::statistics::NormType::Inf,Npp8u>(const Npp8u*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiffMasked<nppi::statistics::NormType::Inf,Npp8u>(const Npp8u*,int,const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiffMaskedCoi<nppi::statistics::NormType::Inf,Npp8u>(const Npp8u*,int,const Npp8u*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormMasked<nppi::statistics::NormType::Inf,Npp16u>(const Npp16u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormMaskedCoi<nppi::statistics::NormType::Inf,Npp16u>(const Npp16u*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiffMasked<nppi::statistics::NormType::Inf,Npp16u>(const Npp16u*,int,const Npp16u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiffMaskedCoi<nppi::statistics::NormType::Inf,Npp16u>(const Npp16u*,int,const Npp16u*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormMasked<nppi::statistics::NormType::Inf,Npp32f>(const Npp32f*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormMaskedCoi<nppi::statistics::NormType::Inf,Npp32f>(const Npp32f*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiffMasked<nppi::statistics::NormType::Inf,Npp32f>(const Npp32f*,int,const Npp32f*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiffMaskedCoi<nppi::statistics::NormType::Inf,Npp32f>(const Npp32f*,int,const Npp32f*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormMasked<nppi::statistics::NormType::Inf,Npp8s>(const Npp8s*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormMaskedCoi<nppi::statistics::NormType::Inf,Npp8s>(const Npp8s*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiffMasked<nppi::statistics::NormType::Inf,Npp8s>(const Npp8s*,int,const Npp8s*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormDiffMaskedCoi<nppi::statistics::NormType::Inf,Npp8s>(const Npp8s*,int,const Npp8s*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
// --- NormRel explicit instantiations (package p3.1b2) ---
template NppStatus runNormRel<nppi::statistics::NormType::L1,Npp16s,1,0>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::L1,Npp16s,3,0>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::L1,Npp16s,4,0>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::L1,Npp16s,4,1>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::L1,Npp16u,1,0>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::L1,Npp16u,3,0>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::L1,Npp16u,4,0>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::L1,Npp16u,4,1>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::L1,Npp32f,1,0>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::L1,Npp32f,3,0>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::L1,Npp32f,4,0>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::L1,Npp32f,4,1>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::L1,Npp8u,1,0>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::L1,Npp8u,3,0>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::L1,Npp8u,4,0>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::L1,Npp8u,4,1>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRelMasked<nppi::statistics::NormType::L1,Npp16u>(const Npp16u*,int,const Npp16u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRelMaskedCoi<nppi::statistics::NormType::L1,Npp16u>(const Npp16u*,int,const Npp16u*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRelMasked<nppi::statistics::NormType::L1,Npp32f>(const Npp32f*,int,const Npp32f*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRelMaskedCoi<nppi::statistics::NormType::L1,Npp32f>(const Npp32f*,int,const Npp32f*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRelMasked<nppi::statistics::NormType::L1,Npp8s>(const Npp8s*,int,const Npp8s*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRelMaskedCoi<nppi::statistics::NormType::L1,Npp8s>(const Npp8s*,int,const Npp8s*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRelMasked<nppi::statistics::NormType::L1,Npp8u>(const Npp8u*,int,const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRelMaskedCoi<nppi::statistics::NormType::L1,Npp8u>(const Npp8u*,int,const Npp8u*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::L2,Npp16s,1,0>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::L2,Npp16s,3,0>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::L2,Npp16s,4,0>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::L2,Npp16s,4,1>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::L2,Npp16u,1,0>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::L2,Npp16u,3,0>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::L2,Npp16u,4,0>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::L2,Npp16u,4,1>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::L2,Npp32f,1,0>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::L2,Npp32f,3,0>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::L2,Npp32f,4,0>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::L2,Npp32f,4,1>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::L2,Npp8u,1,0>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::L2,Npp8u,3,0>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::L2,Npp8u,4,0>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::L2,Npp8u,4,1>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRelMasked<nppi::statistics::NormType::L2,Npp16u>(const Npp16u*,int,const Npp16u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRelMaskedCoi<nppi::statistics::NormType::L2,Npp16u>(const Npp16u*,int,const Npp16u*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRelMasked<nppi::statistics::NormType::L2,Npp32f>(const Npp32f*,int,const Npp32f*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRelMaskedCoi<nppi::statistics::NormType::L2,Npp32f>(const Npp32f*,int,const Npp32f*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRelMasked<nppi::statistics::NormType::L2,Npp8s>(const Npp8s*,int,const Npp8s*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRelMaskedCoi<nppi::statistics::NormType::L2,Npp8s>(const Npp8s*,int,const Npp8s*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRelMasked<nppi::statistics::NormType::L2,Npp8u>(const Npp8u*,int,const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRelMaskedCoi<nppi::statistics::NormType::L2,Npp8u>(const Npp8u*,int,const Npp8u*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::Inf,Npp16s,1,0>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::Inf,Npp16s,3,0>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::Inf,Npp16s,4,0>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::Inf,Npp16s,4,1>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::Inf,Npp16u,1,0>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::Inf,Npp16u,3,0>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::Inf,Npp16u,4,0>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::Inf,Npp16u,4,1>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::Inf,Npp32f,1,0>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::Inf,Npp32f,3,0>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::Inf,Npp32f,4,0>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::Inf,Npp32f,4,1>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::Inf,Npp8u,1,0>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::Inf,Npp8u,3,0>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::Inf,Npp8u,4,0>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRel<nppi::statistics::NormType::Inf,Npp8u,4,1>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRelMasked<nppi::statistics::NormType::Inf,Npp16u>(const Npp16u*,int,const Npp16u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRelMaskedCoi<nppi::statistics::NormType::Inf,Npp16u>(const Npp16u*,int,const Npp16u*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRelMasked<nppi::statistics::NormType::Inf,Npp32f>(const Npp32f*,int,const Npp32f*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRelMaskedCoi<nppi::statistics::NormType::Inf,Npp32f>(const Npp32f*,int,const Npp32f*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRelMasked<nppi::statistics::NormType::Inf,Npp8s>(const Npp8s*,int,const Npp8s*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRelMaskedCoi<nppi::statistics::NormType::Inf,Npp8s>(const Npp8s*,int,const Npp8s*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRelMasked<nppi::statistics::NormType::Inf,Npp8u>(const Npp8u*,int,const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runNormRelMaskedCoi<nppi::statistics::NormType::Inf,Npp8u>(const Npp8u*,int,const Npp8u*,int,const Npp8u*,int,NppiSize,int,Npp8u*,Npp64f*,cudaStream_t);
// --- DotProd explicit instantiations (package p3.1b2) ---
template NppStatus runDotProd<Npp8u,1,0>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runDotProd<Npp8u,3,0>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runDotProd<Npp8u,4,0>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runDotProd<Npp8u,4,1>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runDotProd<Npp8s,1,0>(const Npp8s*,int,const Npp8s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runDotProd<Npp8s,3,0>(const Npp8s*,int,const Npp8s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runDotProd<Npp8s,4,0>(const Npp8s*,int,const Npp8s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runDotProd<Npp8s,4,1>(const Npp8s*,int,const Npp8s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runDotProd<Npp16u,1,0>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runDotProd<Npp16u,3,0>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runDotProd<Npp16u,4,0>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runDotProd<Npp16u,4,1>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runDotProd<Npp16s,1,0>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runDotProd<Npp16s,3,0>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runDotProd<Npp16s,4,0>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runDotProd<Npp16s,4,1>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runDotProd<Npp32u,1,0>(const Npp32u*,int,const Npp32u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runDotProd<Npp32u,3,0>(const Npp32u*,int,const Npp32u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runDotProd<Npp32u,4,0>(const Npp32u*,int,const Npp32u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runDotProd<Npp32u,4,1>(const Npp32u*,int,const Npp32u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runDotProd<Npp32s,1,0>(const Npp32s*,int,const Npp32s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runDotProd<Npp32s,3,0>(const Npp32s*,int,const Npp32s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runDotProd<Npp32s,4,0>(const Npp32s*,int,const Npp32s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runDotProd<Npp32s,4,1>(const Npp32s*,int,const Npp32s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runDotProd<Npp32f,1,0>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runDotProd<Npp32f,3,0>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runDotProd<Npp32f,4,0>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runDotProd<Npp32f,4,1>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
}  // namespace statistics
}  // namespace nppi

// Explicit instantiations for all variants listed in api_analysis/packages/p1.3c.csv
// (AverageError / AverageRelativeError, 11 dtypes x C1..C4).
namespace nppi {
namespace statistics {
template NppStatus runAverageError<Npp8u,1>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp8u,2>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp8u,3>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp8u,4>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp8s,1>(const Npp8s*,int,const Npp8s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp8s,2>(const Npp8s*,int,const Npp8s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp8s,3>(const Npp8s*,int,const Npp8s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp8s,4>(const Npp8s*,int,const Npp8s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp16u,1>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp16u,2>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp16u,3>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp16u,4>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp16s,1>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp16s,2>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp16s,3>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp16s,4>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp16sc,1>(const Npp16sc*,int,const Npp16sc*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp16sc,2>(const Npp16sc*,int,const Npp16sc*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp16sc,3>(const Npp16sc*,int,const Npp16sc*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp16sc,4>(const Npp16sc*,int,const Npp16sc*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp32u,1>(const Npp32u*,int,const Npp32u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp32u,2>(const Npp32u*,int,const Npp32u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp32u,3>(const Npp32u*,int,const Npp32u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp32u,4>(const Npp32u*,int,const Npp32u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp32s,1>(const Npp32s*,int,const Npp32s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp32s,2>(const Npp32s*,int,const Npp32s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp32s,3>(const Npp32s*,int,const Npp32s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp32s,4>(const Npp32s*,int,const Npp32s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp32sc,1>(const Npp32sc*,int,const Npp32sc*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp32sc,2>(const Npp32sc*,int,const Npp32sc*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp32sc,3>(const Npp32sc*,int,const Npp32sc*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp32sc,4>(const Npp32sc*,int,const Npp32sc*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp32f,1>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp32f,2>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp32f,3>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp32f,4>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp32fc,1>(const Npp32fc*,int,const Npp32fc*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp32fc,2>(const Npp32fc*,int,const Npp32fc*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp32fc,3>(const Npp32fc*,int,const Npp32fc*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp32fc,4>(const Npp32fc*,int,const Npp32fc*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp64f,1>(const Npp64f*,int,const Npp64f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp64f,2>(const Npp64f*,int,const Npp64f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp64f,3>(const Npp64f*,int,const Npp64f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageError<Npp64f,4>(const Npp64f*,int,const Npp64f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp8u,1>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp8u,2>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp8u,3>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp8u,4>(const Npp8u*,int,const Npp8u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp8s,1>(const Npp8s*,int,const Npp8s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp8s,2>(const Npp8s*,int,const Npp8s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp8s,3>(const Npp8s*,int,const Npp8s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp8s,4>(const Npp8s*,int,const Npp8s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp16u,1>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp16u,2>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp16u,3>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp16u,4>(const Npp16u*,int,const Npp16u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp16s,1>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp16s,2>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp16s,3>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp16s,4>(const Npp16s*,int,const Npp16s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp16sc,1>(const Npp16sc*,int,const Npp16sc*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp16sc,2>(const Npp16sc*,int,const Npp16sc*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp16sc,3>(const Npp16sc*,int,const Npp16sc*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp16sc,4>(const Npp16sc*,int,const Npp16sc*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp32u,1>(const Npp32u*,int,const Npp32u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp32u,2>(const Npp32u*,int,const Npp32u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp32u,3>(const Npp32u*,int,const Npp32u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp32u,4>(const Npp32u*,int,const Npp32u*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp32s,1>(const Npp32s*,int,const Npp32s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp32s,2>(const Npp32s*,int,const Npp32s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp32s,3>(const Npp32s*,int,const Npp32s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp32s,4>(const Npp32s*,int,const Npp32s*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp32sc,1>(const Npp32sc*,int,const Npp32sc*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp32sc,2>(const Npp32sc*,int,const Npp32sc*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp32sc,3>(const Npp32sc*,int,const Npp32sc*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp32sc,4>(const Npp32sc*,int,const Npp32sc*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp32f,1>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp32f,2>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp32f,3>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp32f,4>(const Npp32f*,int,const Npp32f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp32fc,1>(const Npp32fc*,int,const Npp32fc*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp32fc,2>(const Npp32fc*,int,const Npp32fc*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp32fc,3>(const Npp32fc*,int,const Npp32fc*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp32fc,4>(const Npp32fc*,int,const Npp32fc*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp64f,1>(const Npp64f*,int,const Npp64f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp64f,2>(const Npp64f*,int,const Npp64f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp64f,3>(const Npp64f*,int,const Npp64f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
template NppStatus runAverageRelativeError<Npp64f,4>(const Npp64f*,int,const Npp64f*,int,NppiSize,Npp8u*,Npp64f*,cudaStream_t);
}  // namespace statistics
}  // namespace nppi
