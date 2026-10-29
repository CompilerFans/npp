// nppiSum / nppiMean / nppiMean_StdDev executor kernels + explicit
// instantiation (see nppi_statistics_executor.h for the host surface).
#include "npp.h"
#include "nppi_statistics_executor.h"
#include <cmath>
#include <cuda_runtime.h>

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
}  // namespace statistics
}  // namespace nppi
