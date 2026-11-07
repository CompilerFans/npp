// Quality / error metric executors for this module (see the header for the
// semantics contract).  Reductions accumulate in double precision.
#include "nppi_quality_metrics_executor.h"

#include <cmath>
#include <cuda_runtime.h>
#include <vector>

namespace nppi {
namespace statistics {
namespace {

constexpr int kBlock = 256;

template <typename Acc> __device__ Acc block_reduce(Acc val) {
  __shared__ Acc shared[kBlock];
  shared[threadIdx.x] = val;
  __syncthreads();
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
    __syncthreads();
  }
  return shared[0];
}

__device__ float psnr_from_mse(double mse) {
  if (mse <= 0.0) return __int_as_float(0x7f800000);
  return (float)(10.0 * log10(65025.0 / mse));
}

template <typename D>
__global__ void qualityIndexSumsKernel(const D *pSrc1, int nSrc1Step, const D *pSrc2, int nSrc2Step, int width,
                                       int height, int channels, double *sums) {
  const int c = blockIdx.x;
  double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
  const long total = (long)width * height;
  for (long i = (long)threadIdx.x; i < total; i += blockDim.x) {
    const int y = (int)(i / width);
    const int x = (int)(i % width);
    const D *r1 = (const D *)((const char *)pSrc1 + (size_t)y * nSrc1Step);
    const D *r2 = (const D *)((const char *)pSrc2 + (size_t)y * nSrc2Step);
    const double a = (double)r1[(size_t)x * channels + c];
    const double b = (double)r2[(size_t)x * channels + c];
    sa += a;
    sb += b;
    saa += a * a;
    sbb += b * b;
    sab += a * b;
  }
  sa = block_reduce<double>(sa);
  sb = block_reduce<double>(sb);
  saa = block_reduce<double>(saa);
  sbb = block_reduce<double>(sbb);
  sab = block_reduce<double>(sab);
  if (threadIdx.x == 0) {
    sums[c * 5 + 0] = sa;
    sums[c * 5 + 1] = sb;
    sums[c * 5 + 2] = saa;
    sums[c * 5 + 3] = sbb;
    sums[c * 5 + 4] = sab;
  }
}

template <typename D>
__global__ void countInRange1Kernel(const D *pSrc, int nSrcStep, int width, int height, int *pCounts, D lo, D hi) {
  const long total = (long)width * height;
  long count = 0;
  for (long i = (long)threadIdx.x; i < total; i += blockDim.x) {
    const int y = (int)(i / width);
    const int x = (int)(i % width);
    const D *row = (const D *)((const char *)pSrc + (size_t)y * nSrcStep);
    const D v = row[x];
    if (v >= lo && v <= hi) ++count;
  }
  __shared__ long shared[kBlock];
  shared[threadIdx.x] = count;
  __syncthreads();
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) *pCounts = (int)shared[0];
}

// The per-channel bounds are host arrays (NPP contract); pass them to the
// kernel by value instead of dereferencing host memory from the device.
template <typename D> struct CountBounds {
  D lo[4];
  D hi[4];
};

template <typename D>
__global__ void countInRangeNKernel(const D *pSrc, int nSrcStep, int width, int height, int channels,
                                    CountBounds<D> bounds, int *aCounts) {
  const int c = blockIdx.x;
  const D lo = bounds.lo[c];
  const D hi = bounds.hi[c];
  const long total = (long)width * height;
  long count = 0;
  for (long i = (long)threadIdx.x; i < total; i += blockDim.x) {
    const int y = (int)(i / width);
    const int x = (int)(i % width);
    const D *row = (const D *)((const char *)pSrc + (size_t)y * nSrcStep);
    const D v = row[(size_t)x * channels + c];
    if (v >= lo && v <= hi) ++count;
  }
  __shared__ long shared[kBlock];
  shared[threadIdx.x] = count;
  __syncthreads();
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) aCounts[c] = (int)shared[0];
}

template <typename D, bool DO_PSNR>
__global__ void mseKernel(const D *pSrc1, int nSrc1Step, const D *pSrc2, int nSrc2Step, int width, int height,
                          int channels, Npp32f *pOut) {
  const int c = blockIdx.x;
  const long total = (long)width * height;
  double acc = 0.0;
  for (long i = (long)threadIdx.x; i < total; i += blockDim.x) {
    const int y = (int)(i / width);
    const int x = (int)(i % width);
    const D *r1 = (const D *)((const char *)pSrc1 + (size_t)y * nSrc1Step);
    const D *r2 = (const D *)((const char *)pSrc2 + (size_t)y * nSrc2Step);
    const double d = (double)r1[(size_t)x * channels + c] - (double)r2[(size_t)x * channels + c];
    acc += d * d;
  }
  acc = block_reduce<double>(acc);
  if (threadIdx.x == 0) {
    const double mse = acc / (double)total;
    pOut[c] = DO_PSNR ? psnr_from_mse(mse): (Npp32f)mse;
  }
}

template <typename D>
__global__ void histogramEvenKernel(const D *pSrc, int nSrcStep, int width, int height, int channels, int ch,
                                    Npp32s *pHist, int nLevels, Npp32s lo, Npp32s hi) {
  const long total = (long)width * height;
  const long span = (long)hi - (long)lo;
  const int nbins = nLevels - 1;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x * blockDim.x) {
    const int y = (int)(i / width);
    const int x = (int)(i % width);
    const D *row = (const D *)((const char *)pSrc + (size_t)y * nSrcStep);
    const double v = (double)row[(size_t)x * channels + ch];
    if (v < (double)lo || v >= (double)hi) continue;
    int bin = (int)(((v - (double)lo) * (double)nbins) / (double)span);
    if (bin >= nbins) bin = nbins - 1;
    if (bin < 0) bin = 0;
    atomicAdd((int *)&pHist[bin], 1);
  }
}

template <typename D, typename L>
__global__ void histogramRangeKernel(const D *pSrc, int nSrcStep, int width, int height, int channels, int ch,
                                     Npp32s *pHist, const L *pLevels, int nLevels) {
  const long total = (long)width * height;
  const int nbins = nLevels - 1;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x * blockDim.x) {
    const int y = (int)(i / width);
    const int x = (int)(i % width);
    const D *row = (const D *)((const char *)pSrc + (size_t)y * nSrcStep);
    const double v = (double)row[(size_t)x * channels + ch];
    if (!(v >= (double)pLevels[0]) || !(v < (double)pLevels[nLevels - 1])) continue;
    int bin = -1;
    for (int k = 0; k < nbins; ++k) {
      if (v >= (double)pLevels[k] && v < (double)pLevels[k + 1]) {
        bin = k;
        break;
      }
    }
    if (bin >= 0) atomicAdd((int *)&pHist[bin], 1);
  }
}

template <typename S>
__global__ void rectStdDevF32Kernel(const S *pSrc, int nSrcStep, const Npp64f *pSqr, int nSqrStep, Npp32f *pDst,
                                    int nDstStep, int width, int height, NppiRect rect) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const int x1 = x + rect.x, y1 = y + rect.y;
  const int x2 = x1 + rect.width, y2 = y1 + rect.height;
  const S *srcRow0 = (const S *)((const char *)pSrc + (size_t)y1 * nSrcStep);
  const S *srcRow1 = (const S *)((const char *)pSrc + (size_t)y2 * nSrcStep);
  const Npp64f *sqrRow0 = (const Npp64f *)((const char *)pSqr + (size_t)y1 * nSqrStep);
  const Npp64f *sqrRow1 = (const Npp64f *)((const char *)pSqr + (size_t)y2 * nSqrStep);
  const double sum = (double)srcRow1[x2] - (double)srcRow0[x2] - (double)srcRow1[x1] + (double)srcRow0[x1];
  const double sumSq = sqrRow1[x2] - sqrRow0[x2] - sqrRow1[x1] + sqrRow0[x1];
  const double count = (double)rect.width * rect.height;
  // (n*sumSq - sum^2)/n^2 keeps exact integer statistics exact, matching the
  // reference (the naive mean-of-squares form loses a ULP and shifts the
  // truncated 32s output by one).
  double var = (count * sumSq - sum * sum) / (count * count);
  if (var < 0.0) var = 0.0;
  Npp32f *dstRow = (Npp32f *)((char *)pDst + (size_t)y * nDstStep);
  dstRow[x] = (Npp32f)sqrt(var);
}

__global__ void rectStdDevS32Kernel(const Npp32s *pSrc, int nSrcStep, const Npp32s *pSqr, int nSqrStep, Npp32s *pDst,
                                    int nDstStep, int width, int height, NppiRect rect, int scale) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const int x1 = x + rect.x, y1 = y + rect.y;
  const int x2 = x1 + rect.width, y2 = y1 + rect.height;
  const Npp32s *srcRow0 = (const Npp32s *)((const char *)pSrc + (size_t)y1 * nSrcStep);
  const Npp32s *srcRow1 = (const Npp32s *)((const char *)pSrc + (size_t)y2 * nSrcStep);
  const Npp32s *sqrRow0 = (const Npp32s *)((const char *)pSqr + (size_t)y1 * nSqrStep);
  const Npp32s *sqrRow1 = (const Npp32s *)((const char *)pSqr + (size_t)y2 * nSqrStep);
  const double sum = (double)srcRow1[x2] - (double)srcRow0[x2] - (double)srcRow1[x1] + (double)srcRow0[x1];
  const double sumSq = (double)sqrRow1[x2] - (double)sqrRow0[x2] - (double)sqrRow1[x1] + (double)sqrRow0[x1];
  const double count = (double)rect.width * rect.height;
  double var = (count * sumSq - sum * sum) / (count * count);
  if (var < 0.0) var = 0.0;
  Npp32s v = (Npp32s)sqrt(var);
  if (scale > 0) {
    // NPP truncates the square root first, then divides by 2^sf with
    // round-half-to-even.
    const unsigned d = 1u << (unsigned)scale;
    const unsigned r = (unsigned)v & (d - 1u);
    unsigned q = (unsigned)v / d;
    if (r > d / 2u || (r == d / 2u && (q & 1u))) ++q;
    v = (Npp32s)q;
  } else if (scale < 0) {
    v = (Npp32s)((unsigned)v << (unsigned)(-scale));
  }
  Npp32s *dstRow = (Npp32s *)((char *)pDst + (size_t)y * nDstStep);
  dstRow[x] = v;
}

bool step_ok(int step, int width, int channels, int esize) { return step >= width * channels * esize; }

// Per-channel CountInRange / MSE scratch formula (reference): for 8u inputs the
// scratch is 1024 bytes per row below 128 rows and 12 bytes per row above;
// 32f always uses 12 bytes per row.
int rowScratchBytes(int height, bool is8u) {
  if (is8u && height < 128) return 1024 * height;
  return 12 * height;
}

// Batch scratch: 8 bytes per 32x128 block of the ROI, per colour channel.
int batchScratchBytes(NppiSize roi, int channels) {
  const long bx = (roi.width + 31) / 32;
  const long by = (roi.height + 127) / 128;
  return (int)(8 * bx * by * channels);
}

}  // namespace

// ---------------------------------------------------------------------------
// Buffer-size helpers (NVIDIA NPP 13.4 formulas measured with the reference sweep).
// ---------------------------------------------------------------------------
NppStatus qualityIndexGetBufferHostSize(NppiSize oSizeROI, int channels, NPP_BUFFER_SIZE_TYPE *hpBufferSize) {
  if (!hpBufferSize) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  const int h = oSizeROI.height;
  *hpBufferSize = channels * (4 * (h + (h - 1) / 256) + 20);
  return NPP_NO_ERROR;
}

NppStatus countInRangeGetBufferHostSize(NppiSize oSizeROI, int channels, bool alpha, int elemSize,
                                        NPP_BUFFER_SIZE_TYPE *hpBufferSize) {
  if (!hpBufferSize) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  *hpBufferSize = (alpha ? channels - 1: channels) * rowScratchBytes(oSizeROI.height, elemSize == 1);
  return NPP_NO_ERROR;
}

NppStatus mseGetBufferHostSize(NppiSize oSizeROI, int channels, NPP_BUFFER_SIZE_TYPE *hpBufferSize) {
  return countInRangeGetBufferHostSize(oSizeROI, channels, false, 1, hpBufferSize);
}

NppStatus psnrGetBufferHostSize(NppiSize oSizeROI, int channels, NPP_BUFFER_SIZE_TYPE *hpBufferSize) {
  return countInRangeGetBufferHostSize(oSizeROI, channels, false, 1, hpBufferSize);
}

NppStatus mseBatchGetBufferHostSize(NppiSize oSizeROI, int channels, NPP_BUFFER_SIZE_TYPE *hpBufferSize) {
  if (!hpBufferSize) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  *hpBufferSize = batchScratchBytes(oSizeROI, channels);
  return NPP_NO_ERROR;
}

NppStatus psnrBatchGetBufferHostSize(NppiSize oSizeROI, int channels, NPP_BUFFER_SIZE_TYPE *hpBufferSize) {
  return mseBatchGetBufferHostSize(oSizeROI, channels, hpBufferSize);
}

// ---------------------------------------------------------------------------
// QualityIndex
// ---------------------------------------------------------------------------
template <typename D>
NppStatus runQualityIndex(const D *pSrc1, int nSrc1Step, const D *pSrc2, int nSrc2Step, NppiSize oSizeROI,
                          int channels, bool alpha, Npp32f *pDst, Npp8u *pDeviceBuffer, cudaStream_t stream) {
  if (!pSrc1 || !pSrc2 || !pDst || !pDeviceBuffer) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  const int esize = (int)sizeof(D);
  if (!step_ok(nSrc1Step, oSizeROI.width, channels, esize) ||
      !step_ok(nSrc2Step, oSizeROI.width, channels, esize)) {
    return NPP_STEP_ERROR;
  }
  // A 4-channel step must be a multiple of the pixel size (the reference returns
  // NOT_EVEN_STEP_ERROR for a 32f AC4 step of 36; STEP_ERROR takes precedence
  // when the step is also too short).
  const int align = channels == 4 ? channels * esize: esize;
  if (nSrc1Step % align != 0 || nSrc2Step % align != 0) return NPP_NOT_EVEN_STEP_ERROR;
  const int nch = alpha ? channels - 1: channels;
  if (nch < 1 || nch > 4) return NPP_NUMBER_OF_CHANNELS_ERROR;

  double *dSums = nullptr;
  if (cudaMalloc(&dSums, sizeof(double) * 5 * nch) != cudaSuccess) return NPP_MEMORY_ALLOCATION_ERR;
  qualityIndexSumsKernel<D><<<nch, kBlock, 0, stream>>>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, oSizeROI.width,
                                                        oSizeROI.height, channels, dSums);
  cudaError_t err = cudaGetLastError();
  double hSums[5 * 4] = {0};
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(hSums, dSums, sizeof(double) * 5 * nch, cudaMemcpyDeviceToHost, stream);
  }
  if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
  cudaFree(dSums);
  if (err != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;

  const double n = (double)oSizeROI.width * oSizeROI.height;
  const double ma = hSums[0] / n;
  const double mb = hSums[1] / n;
  const double va = (hSums[2] - n * ma * ma) / (n - 1.0);
  const double vb = (hSums[3] - n * mb * mb) / (n - 1.0);
  for (int c = 0; c < nch; ++c) {
    const double ca = hSums[c * 5 + 0] / n;
    const double cb = hSums[c * 5 + 1] / n;
    const double vca = (hSums[c * 5 + 2] - n * ca * ca) / (n - 1.0);
    const double vcb = (hSums[c * 5 + 3] - n * cb * cb) / (n - 1.0);
    if (vca == 0.0 || vcb == 0.0) return NPP_QUALITY_INDEX_ERROR;
  }
  Npp32f hq[4] = {0, 0, 0, 0};
  const double denom = (ma * ma + mb * mb) * (va + vb);
  for (int c = 0; c < nch; ++c) {
    // Each channel's covariance uses its own channel means; the numerator
    // means and the (va+vb) denominator are taken from channel 0 (NPP quirk).
    const double ca = hSums[c * 5 + 0] / n;
    const double cb = hSums[c * 5 + 1] / n;
    const double cov = (hSums[c * 5 + 4] - n * ca * cb) / (n - 1.0);
    hq[c] = (Npp32f)(4.0 * cov * ma * mb / denom);
  }
  err = cudaMemcpyAsync(pDst, hq, sizeof(Npp32f) * nch, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

// ---------------------------------------------------------------------------
// CountInRange
// ---------------------------------------------------------------------------
template <typename D>
NppStatus runCountInRange1(const D *pSrc, int nSrcStep, NppiSize oSizeROI, int *pCounts, D nLowerBound,
                           D nUpperBound, Npp8u *pDeviceBuffer, cudaStream_t stream) {
  if (!pSrc || !pCounts || !pDeviceBuffer) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  if (!step_ok(nSrcStep, oSizeROI.width, 1, (int)sizeof(D))) return NPP_STEP_ERROR;
  if (!(nLowerBound <= nUpperBound)) return NPP_RANGE_ERROR;
  countInRange1Kernel<D><<<1, kBlock, 0, stream>>>(pSrc, nSrcStep, oSizeROI.width, oSizeROI.height, pCounts,
                                                   nLowerBound, nUpperBound);
  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

template <typename D>
NppStatus runCountInRangeN(const D *pSrc, int nSrcStep, NppiSize oSizeROI, int *aCounts, const D *aLowerBound,
                           const D *aUpperBound, int channels, bool alpha, Npp8u *pDeviceBuffer,
                           cudaStream_t stream) {
  if (!pSrc || !aCounts || !aLowerBound || !aUpperBound || !pDeviceBuffer) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  if (!step_ok(nSrcStep, oSizeROI.width, channels, (int)sizeof(D))) return NPP_STEP_ERROR;
  const int nch = alpha ? channels - 1: channels;
  for (int c = 0; c < nch; ++c) {
    if (!(aLowerBound[c] <= aUpperBound[c])) return NPP_RANGE_ERROR;
  }
  CountBounds<D> bounds{};
  for (int c = 0; c < nch; ++c) {
    bounds.lo[c] = aLowerBound[c];
    bounds.hi[c] = aUpperBound[c];
  }
  countInRangeNKernel<D><<<nch, kBlock, 0, stream>>>(pSrc, nSrcStep, oSizeROI.width, oSizeROI.height, channels,
                                                     bounds, aCounts);
  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

// ---------------------------------------------------------------------------
// MSE / PSNR
// ---------------------------------------------------------------------------
template <typename D>
NppStatus runMSE(const D *pSrc1, int nSrc1Step, const D *pSrc2, int nSrc2Step, NppiSize oSizeROI, int channels,
                 bool alpha, Npp32f *pMSE, Npp8u *pDeviceBuffer, cudaStream_t stream) {
  if (!pSrc1 || !pSrc2 || !pMSE || !pDeviceBuffer) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  if (!step_ok(nSrc1Step, oSizeROI.width, channels, (int)sizeof(D)) ||
      !step_ok(nSrc2Step, oSizeROI.width, channels, (int)sizeof(D))) {
    return NPP_STEP_ERROR;
  }
  const int nch = alpha ? channels - 1: channels;
  mseKernel<D, false><<<nch, kBlock, 0, stream>>>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, oSizeROI.width, oSizeROI.height,
                                                  channels, pMSE);
  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

template <typename D>
NppStatus runPSNR(const D *pSrc1, int nSrc1Step, const D *pSrc2, int nSrc2Step, NppiSize oSizeROI, int channels,
                  bool alpha, Npp32f *pPSNR, Npp8u *pDeviceBuffer, cudaStream_t stream) {
  if (!pSrc1 || !pSrc2 || !pPSNR || !pDeviceBuffer) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  if (!step_ok(nSrc1Step, oSizeROI.width, channels, (int)sizeof(D)) ||
      !step_ok(nSrc2Step, oSizeROI.width, channels, (int)sizeof(D))) {
    return NPP_STEP_ERROR;
  }
  const int nch = alpha ? channels - 1: channels;
  mseKernel<D, true><<<nch, kBlock, 0, stream>>>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, oSizeROI.width, oSizeROI.height,
                                                 channels, pPSNR);
  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

// ---------------------------------------------------------------------------
// Batch metrics
// ---------------------------------------------------------------------------
namespace {
template <bool DO_PSNR>
NppStatus runMetricBatch(const NppiImageDescriptor *pSrc1, const NppiImageDescriptor *pSrc2, int nBatchSize,
                         NppiSize oSizeROI, int channels, Npp32f *pOut,
                         const NppiBufferDescriptor *pDeviceBufferList, bool advanced, cudaStream_t stream) {
  (void)oSizeROI;
  if (!pSrc1 || !pSrc2 || !pOut || !pDeviceBufferList) return NPP_NULL_POINTER_ERROR;
  if (nBatchSize <= 0) return NPP_BAD_ARGUMENT_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  std::vector<NppiImageDescriptor> h1(nBatchSize), h2(nBatchSize);
  cudaError_t err = cudaMemcpyAsync(h1.data(), pSrc1, sizeof(NppiImageDescriptor) * nBatchSize, cudaMemcpyDeviceToHost,
                                    stream);
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(h2.data(), pSrc2, sizeof(NppiImageDescriptor) * nBatchSize, cudaMemcpyDeviceToHost, stream);
  }
  if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  for (int i = 0; i < nBatchSize; ++i) {
    const NppiSize roi = advanced ? h1[i].oSize: oSizeROI;
    if (roi.width <= 0 || roi.height <= 0) return NPP_SIZE_ERROR;
    if (h2[i].nStep < roi.width * channels || h1[i].nStep < roi.width * channels) return NPP_STEP_ERROR;
    mseKernel<Npp8u, DO_PSNR><<<channels, kBlock, 0, stream>>>(
        (const Npp8u *)h1[i].pData, h1[i].nStep, (const Npp8u *)h2[i].pData, h2[i].nStep, roi.width, roi.height,
        channels, pOut + (size_t)i * channels);
    if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
  return NPP_NO_ERROR;
}
}  // namespace

NppStatus runMSEBatch(const NppiImageDescriptor *pSrc1, const NppiImageDescriptor *pSrc2, int nBatchSize,
                      NppiSize oSizeROI, int channels, Npp32f *pMSE,
                      const NppiBufferDescriptor *pDeviceBufferList, bool advanced, cudaStream_t stream) {
  return runMetricBatch<false>(pSrc1, pSrc2, nBatchSize, oSizeROI, channels, pMSE, pDeviceBufferList, advanced,
                               stream);
}

NppStatus runPSNRBatch(const NppiImageDescriptor *pSrc1, const NppiImageDescriptor *pSrc2, int nBatchSize,
                       NppiSize oSizeROI, int channels, Npp32f *pPSNR,
                       const NppiBufferDescriptor *pDeviceBufferList, bool advanced, cudaStream_t stream) {
  return runMetricBatch<true>(pSrc1, pSrc2, nBatchSize, oSizeROI, channels, pPSNR, pDeviceBufferList, advanced,
                              stream);
}

// ---------------------------------------------------------------------------
// Histogram C3R / AC4R
// ---------------------------------------------------------------------------
namespace {
template <typename D>
void launchHistogramEven(const void *pSrc, int nSrcStep, NppiSize roi, Npp32s *const *pHist, const int *nLevels,
                         const Npp32s *lo, const Npp32s *hi, int channels, int nch, cudaStream_t stream) {
  const long total = (long)roi.width * roi.height;
  for (int c = 0; c < nch; ++c) {
    cudaMemsetAsync(pHist[c], 0, sizeof(Npp32s) * (size_t)(nLevels[c] - 1), stream);
    int blocks = (int)((total + kBlock - 1) / kBlock);
    if (blocks > 1024) blocks = 1024;
    histogramEvenKernel<D><<<blocks, kBlock, 0, stream>>>((const D *)pSrc, nSrcStep, roi.width, roi.height, channels,
                                                          c, pHist[c], nLevels[c], lo[c], hi[c]);
  }
}

template <typename D, typename L>
void launchHistogramRange(const void *pSrc, int nSrcStep, NppiSize roi, Npp32s *const *pHist, const void *const *pLevels,
                          const int *nLevels, int channels, int nch, cudaStream_t stream) {
  const long total = (long)roi.width * roi.height;
  for (int c = 0; c < nch; ++c) {
    cudaMemsetAsync(pHist[c], 0, sizeof(Npp32s) * (size_t)(nLevels[c] - 1), stream);
    int blocks = (int)((total + kBlock - 1) / kBlock);
    if (blocks > 1024) blocks = 1024;
    histogramRangeKernel<D, L><<<blocks, kBlock, 0, stream>>>((const D *)pSrc, nSrcStep, roi.width, roi.height,
                                                              channels, c, pHist[c], (const L *)pLevels[c],
                                                              nLevels[c]);
  }
}
}  // namespace

NppStatus runHistogramEven(const void *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *const *pHist,
                           const int *nLevels, const Npp32s *nLowerLevel, const Npp32s *nUpperLevel, int channels,
                           bool alpha, int srcKind, cudaStream_t stream) {
  if (!pSrc || !pHist || !nLevels || !nLowerLevel || !nUpperLevel) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  const int esize = srcKind == 0 ? 1: (srcKind == 3 ? 4: 2);
  if (!step_ok(nSrcStep, oSizeROI.width, channels, esize)) return NPP_STEP_ERROR;
  const int nch = alpha ? channels - 1: channels;
  for (int c = 0; c < nch; ++c) {
    if (!pHist[c]) return NPP_NULL_POINTER_ERROR;
    if (nLevels[c] < 2) return NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR;
    if (!(nLowerLevel[c] < nUpperLevel[c])) return NPP_RANGE_ERROR;
  }
  switch (srcKind) {
    case 0: launchHistogramEven<Npp8u>(pSrc, nSrcStep, oSizeROI, pHist, nLevels, nLowerLevel, nUpperLevel, channels, nch, stream); break;
    case 1: launchHistogramEven<Npp16u>(pSrc, nSrcStep, oSizeROI, pHist, nLevels, nLowerLevel, nUpperLevel, channels, nch, stream); break;
    case 2: launchHistogramEven<Npp16s>(pSrc, nSrcStep, oSizeROI, pHist, nLevels, nLowerLevel, nUpperLevel, channels, nch, stream); break;
    default: return NPP_DATA_TYPE_ERROR;
  }
  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus runHistogramRangeI(const void *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *const *pHist,
                             const Npp32s *const *pLevels, const int *nLevels, int channels, bool alpha, int srcKind,
                             cudaStream_t stream) {
  if (!pSrc || !pHist || !pLevels || !nLevels) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  const int esize = srcKind == 0 ? 1: 2;
  if (!step_ok(nSrcStep, oSizeROI.width, channels, esize)) return NPP_STEP_ERROR;
  const int nch = alpha ? channels - 1: channels;
  for (int c = 0; c < nch; ++c) {
    if (!pHist[c] || !pLevels[c]) return NPP_NULL_POINTER_ERROR;
    if (nLevels[c] < 2) return NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR;
  }
  switch (srcKind) {
    case 0: launchHistogramRange<Npp8u, Npp32s>(pSrc, nSrcStep, oSizeROI, pHist, (const void *const *)pLevels, nLevels, channels, nch, stream); break;
    case 1: launchHistogramRange<Npp16u, Npp32s>(pSrc, nSrcStep, oSizeROI, pHist, (const void *const *)pLevels, nLevels, channels, nch, stream); break;
    case 2: launchHistogramRange<Npp16s, Npp32s>(pSrc, nSrcStep, oSizeROI, pHist, (const void *const *)pLevels, nLevels, channels, nch, stream); break;
    default: return NPP_DATA_TYPE_ERROR;
  }
  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus runHistogramRangeF(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *const *pHist,
                             const Npp32f *const *pLevels, const int *nLevels, int channels, bool alpha,
                             cudaStream_t stream) {
  if (!pSrc || !pHist || !pLevels || !nLevels) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  if (!step_ok(nSrcStep, oSizeROI.width, channels, 4)) return NPP_STEP_ERROR;
  const int nch = alpha ? channels - 1: channels;
  for (int c = 0; c < nch; ++c) {
    if (!pHist[c] || !pLevels[c]) return NPP_NULL_POINTER_ERROR;
    if (nLevels[c] < 2) return NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR;
  }
  launchHistogramRange<Npp32f, Npp32f>(pSrc, nSrcStep, oSizeROI, pHist, (const void *const *)pLevels, nLevels,
                                       channels, nch, stream);
  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

// ---------------------------------------------------------------------------
// RectStdDev / even levels
// ---------------------------------------------------------------------------
namespace {
NppStatus validateRect(int nSrcStep, int nSqrStep, int nDstStep, int esize, int sqrEsize, NppiSize roi, NppiRect rect) {
  if (roi.width < 0 || roi.height < 0) return NPP_SIZE_ERROR;
  if (rect.width <= 0 || rect.height <= 0 || rect.x < 0 || rect.y < 0) return NPP_SIZE_ERROR;
  if (roi.width == 0 || roi.height == 0) return NPP_NO_ERROR;
  const int need = roi.width + rect.x + rect.width;
  if (nSrcStep < need * esize || nSqrStep < need * sqrEsize || nDstStep < roi.width * (int)sizeof(Npp32f)) {
    return NPP_STEP_ERROR;
  }
  return NPP_SUCCESS;
}
}  // namespace

NppStatus runRectStdDevF32(const Npp32f *pSrc, int nSrcStep, const Npp64f *pSqr, int nSqrStep, Npp32f *pDst,
                           int nDstStep, NppiSize oSizeROI, NppiRect oRect, cudaStream_t stream) {
  if (!pSrc || !pSqr || !pDst) return NPP_NULL_POINTER_ERROR;
  NppStatus st = validateRect(nSrcStep, nSqrStep, nDstStep, 4, 8, oSizeROI, oRect);
  if (st != NPP_SUCCESS || oSizeROI.width == 0 || oSizeROI.height == 0) return st;
  dim3 block(16, 16);
  dim3 grid((oSizeROI.width + 15) / 16, (oSizeROI.height + 15) / 16);
  rectStdDevF32Kernel<Npp32f><<<grid, block, 0, stream>>>(pSrc, nSrcStep, pSqr, nSqrStep, pDst, nDstStep,
                                                          oSizeROI.width, oSizeROI.height, oRect);
  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus runRectStdDevS32(const Npp32s *pSrc, int nSrcStep, const Npp32s *pSqr, int nSqrStep, Npp32s *pDst,
                           int nDstStep, NppiSize oSizeROI, NppiRect oRect, int nScaleFactor, cudaStream_t stream) {
  if (!pSrc || !pSqr || !pDst) return NPP_NULL_POINTER_ERROR;
  NppStatus st = validateRect(nSrcStep, nSqrStep, nDstStep, 4, 4, oSizeROI, oRect);
  if (st != NPP_SUCCESS || oSizeROI.width == 0 || oSizeROI.height == 0) return st;
  dim3 block(16, 16);
  dim3 grid((oSizeROI.width + 15) / 16, (oSizeROI.height + 15) / 16);
  rectStdDevS32Kernel<<<grid, block, 0, stream>>>(pSrc, nSrcStep, pSqr, nSqrStep, pDst, nDstStep, oSizeROI.width,
                                                  oSizeROI.height, oRect, nScaleFactor);
  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus evenLevelsHost(Npp32s *hpLevels, int nLevels, Npp32s nLowerLevel, Npp32s nUpperLevel) {
  if (!hpLevels) return NPP_NULL_POINTER_ERROR;
  if (nLevels < 2) return NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR;
  // NPP walks from the lower level, adding ceil(remaining_gap / remaining
  // steps) at each step.  The division is unsigned 32-bit, so a reversed
  // range wraps exactly like the reference (measured).
  Npp32s v = nLowerLevel;
  hpLevels[0] = v;
  for (int i = 1; i < nLevels; ++i) {
    const Npp32u num = (Npp32u)(nUpperLevel - v);
    const Npp32u den = (Npp32u)(nLevels - i);
    const Npp32u delta = num / den + (num % den ? 1u: 0u);
    v = (Npp32s)((Npp32u)v + delta);
    hpLevels[i] = v;
  }
  return NPP_NO_ERROR;
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------
template NppStatus runQualityIndex<Npp8u>(const Npp8u *, int, const Npp8u *, int, NppiSize, int, bool, Npp32f *,
                                          Npp8u *, cudaStream_t);
template NppStatus runQualityIndex<Npp16u>(const Npp16u *, int, const Npp16u *, int, NppiSize, int, bool, Npp32f *,
                                           Npp8u *, cudaStream_t);
template NppStatus runQualityIndex<Npp32f>(const Npp32f *, int, const Npp32f *, int, NppiSize, int, bool, Npp32f *,
                                           Npp8u *, cudaStream_t);

template NppStatus runCountInRange1<Npp8u>(const Npp8u *, int, NppiSize, int *, Npp8u, Npp8u, Npp8u *,
                                          cudaStream_t);
template NppStatus runCountInRange1<Npp32f>(const Npp32f *, int, NppiSize, int *, Npp32f, Npp32f, Npp8u *,
                                           cudaStream_t);
template NppStatus runCountInRangeN<Npp8u>(const Npp8u *, int, NppiSize, int *, const Npp8u *, const Npp8u *, int,
                                           bool, Npp8u *, cudaStream_t);
template NppStatus runCountInRangeN<Npp32f>(const Npp32f *, int, NppiSize, int *, const Npp32f *, const Npp32f *, int,
                                            bool, Npp8u *, cudaStream_t);

template NppStatus runMSE<Npp8u>(const Npp8u *, int, const Npp8u *, int, NppiSize, int, bool, Npp32f *, Npp8u *,
                                 cudaStream_t);
template NppStatus runPSNR<Npp8u>(const Npp8u *, int, const Npp8u *, int, NppiSize, int, bool, Npp32f *, Npp8u *,
                                  cudaStream_t);


}  // namespace statistics
}  // namespace nppi
