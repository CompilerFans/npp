// nppiIntegral / nppiSqrIntegral executor kernels + explicit instantiation
// (see nppi_integral_executor.h for the host surface).
#include "npp.h"
#include "nppi_integral_executor.h"
#include <cmath>
#include <cuda_runtime.h>
#include <cstdint>

namespace nppi {
namespace statistics {
namespace {

// Accumulator for the two-stage scan: int64 for integral destinations, double
// for floating point destinations, so 8-bit inputs stay exact in 32-bit
// destinations and the column pass re-reads no wider intermediate than needed.
template <typename T> struct ScanAcc;
template <> struct ScanAcc<Npp32s> { using type = int64_t; };
template <> struct ScanAcc<Npp32f> { using type = double; };
template <> struct ScanAcc<Npp64f> { using type = double; };

// Stage 1: one thread per source row j writes the running row prefix into
// destination row j+1 (columns 1..W) and zeroes destination column 0.  When
// HasSqr the same scan fills pSqr with the running prefix of pSrc^2.
template <typename SrcT, typename DstT, bool HasSqr, typename SqrT>
__global__ void integralRowScanKernel(const SrcT *pSrc, int nSrcStep, DstT *pDst, int nDstStep, SqrT *pSqr,
                                      int nSqrStep, int width, int height) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= height) {
    return;
  }
  const SrcT *srcRow = (const SrcT *)((const char *)pSrc + (size_t)j * nSrcStep);
  DstT *dstRow = (DstT *)((char *)pDst + (size_t)(j + 1) * nDstStep);
  typename ScanAcc<DstT>::type run = 0;
  dstRow[0] = DstT(0);
  for (int i = 0; i < width; ++i) {
    run += (typename ScanAcc<DstT>::type)srcRow[i];
    dstRow[i + 1] = (DstT)run;
  }
  if (HasSqr) {
    SqrT *sqrRow = (SqrT *)((char *)pSqr + (size_t)(j + 1) * nSqrStep);
    typename ScanAcc<SqrT>::type runSqr = 0;
    sqrRow[0] = SqrT(0);
    for (int i = 0; i < width; ++i) {
      const double v = (double)srcRow[i];
      runSqr += (typename ScanAcc<SqrT>::type)(v * v);
      sqrRow[i + 1] = (SqrT)runSqr;
    }
  }
}

// Stage 2: one thread per destination column i (0..W, inclusive) turns the row
// prefixes into full rectangle sums and adds nVal / nValSqr everywhere
// (destination row 0 is nVal / nValSqr).
template <typename DstT, bool HasSqr, typename SqrT>
__global__ void integralColScanKernel(DstT *pDst, int nDstStep, DstT nVal, SqrT *pSqr, int nSqrStep, SqrT nValSqr,
                                      int width, int height) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > width) {
    return;
  }
  {
    DstT *elem = (DstT *)((char *)pDst + (size_t)i * sizeof(DstT));
    *elem = nVal;
    typename ScanAcc<DstT>::type run = 0;
    for (int j = 1; j <= height; ++j) {
      elem = (DstT *)((char *)pDst + (size_t)j * nDstStep + (size_t)i * sizeof(DstT));
      run += (typename ScanAcc<DstT>::type)*elem;
      *elem = (DstT)(run + (typename ScanAcc<DstT>::type)nVal);
    }
  }
  if (HasSqr) {
    SqrT *elem = (SqrT *)((char *)pSqr + (size_t)i * sizeof(SqrT));
    *elem = nValSqr;
    typename ScanAcc<SqrT>::type run = 0;
    for (int j = 1; j <= height; ++j) {
      elem = (SqrT *)((char *)pSqr + (size_t)j * nSqrStep + (size_t)i * sizeof(SqrT));
      run += (typename ScanAcc<SqrT>::type)*elem;
      *elem = (SqrT)(run + (typename ScanAcc<SqrT>::type)nValSqr);
    }
  }
}

template <typename T> inline int rowBytes(int width) { return width * (int)sizeof(T); }

}  // namespace

template <typename SrcT, typename DstT>
NppStatus runIntegral(const SrcT *pSrc, int nSrcStep, DstT *pDst, int nDstStep, NppiSize oROI, DstT nVal,
                      cudaStream_t stream) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oROI.width < 0 || oROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  if (oROI.width == 0 || oROI.height == 0) {
    // Degenerate ROI: the (W+1) x (H+1) destination collapses to a single row
    // or column of nVal (5090-measured: still NPP_NO_ERROR).
    return NPP_NO_ERROR;
  }
  if (nSrcStep < rowBytes<SrcT>(oROI.width) || nDstStep < rowBytes<DstT>(oROI.width + 1)) {
    return NPP_STEP_ERROR;
  }

  const int height = oROI.height;
  const int width = oROI.width;
  integralRowScanKernel<SrcT, DstT, false, DstT><<<(height + 127) / 128, 128, 0, stream>>>(
      pSrc, nSrcStep, pDst, nDstStep, pDst, nDstStep, width, height);
  integralColScanKernel<DstT, false, DstT><<<(width + 127) / 128 + 1, 128, 0, stream>>>(pDst, nDstStep, nVal, pDst,
                                                                                        nDstStep, nVal, width, height);
  return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

template <typename SrcT, typename DstT, typename SqrT>
NppStatus runSqrIntegral(const SrcT *pSrc, int nSrcStep, DstT *pDst, int nDstStep, SqrT *pSqr, int nSqrStep,
                         NppiSize oSrcROI, DstT nVal, SqrT nValSqr, cudaStream_t stream) {
  if (!pSrc || !pDst || !pSqr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSrcROI.width < 0 || oSrcROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  if (oSrcROI.width == 0 || oSrcROI.height == 0) {
    return NPP_NO_ERROR;
  }
  if (nSrcStep < rowBytes<SrcT>(oSrcROI.width) || nDstStep < rowBytes<DstT>(oSrcROI.width + 1) ||
      nSqrStep < rowBytes<SqrT>(oSrcROI.width + 1)) {
    return NPP_STEP_ERROR;
  }

  const int height = oSrcROI.height;
  const int width = oSrcROI.width;
  integralRowScanKernel<SrcT, DstT, true, SqrT><<<(height + 127) / 128, 128, 0, stream>>>(
      pSrc, nSrcStep, pDst, nDstStep, pSqr, nSqrStep, width, height);
  integralColScanKernel<DstT, true, SqrT><<<(width + 127) / 128 + 1, 128, 0, stream>>>(pDst, nDstStep, nVal, pSqr,
                                                                                        nSqrStep, nValSqr, width,
                                                                                        height);
  return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

// Explicit instantiation for the nppiIntegral / nppiSqrIntegral variants.
template NppStatus runIntegral<Npp8u, Npp32s>(const Npp8u *, int, Npp32s *, int, NppiSize, Npp32s, cudaStream_t);
template NppStatus runIntegral<Npp8u, Npp32f>(const Npp8u *, int, Npp32f *, int, NppiSize, Npp32f, cudaStream_t);
template NppStatus runSqrIntegral<Npp8u, Npp32s, Npp32s>(const Npp8u *, int, Npp32s *, int, Npp32s *, int, NppiSize,
                                                         Npp32s, Npp32s, cudaStream_t);
template NppStatus runSqrIntegral<Npp8u, Npp32s, Npp64f>(const Npp8u *, int, Npp32s *, int, Npp64f *, int, NppiSize,
                                                         Npp32s, Npp64f, cudaStream_t);
template NppStatus runSqrIntegral<Npp8u, Npp32f, Npp64f>(const Npp8u *, int, Npp32f *, int, Npp64f *, int, NppiSize,
                                                         Npp32f, Npp64f, cudaStream_t);

}  // namespace statistics
}  // namespace nppi
