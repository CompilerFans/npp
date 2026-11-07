// nppiCrossCorr* / nppiSqrDistance* family executor kernels + host entry points.
// See nppi_crosscorr_executor.h for the measured semantics.
#include "npp.h"
#include "nppi_crosscorr_executor.h"
#include "npp_reduction.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <limits>

namespace {

using npp_reduce::kBlock;
using npp_reduce::numBlocks;

// Sfs store for 8u destinations: saturating trunc(value) * 2^-sf with the
// final store truncating fractional results; NaN -> 0, +-inf -> 255
// (reference-measured).
__device__ __forceinline__ Npp8u storeSfs8u(double v, int sf) {
  if (std::isnan(v)) return 0;
  const double scaled = std::trunc(ldexp(std::trunc(v), -sf));
  if (scaled < 0.0) return 0;
  if (scaled > 255.0) return 255;
  return (Npp8u)scaled;
}

template <typename D> __device__ __forceinline__ void storeValue(D *p, double v, int sf) {
  if constexpr (sizeof(D) == 1) {
    *p = storeSfs8u(v, sf);
  } else if constexpr (sizeof(D) == 4) {
    *p = (Npp32f)v;
  } else {
    *p = (Npp64f)v;
  }
}

// One thread per dst sample and channel.  The window is anchored per
// CrossCorrMode; out-of-ROI source samples contribute w = 0 while the template
// is always read in full, so N (= tw * th) counts padded positions for the
// NormLevel statistics.
template <typename S, typename D, int Channels, bool AlphaPreserve, nppi::statistics::CrossCorrMode M,
          nppi::statistics::CrossCorrOp Op>
__global__ void crossCorrKernel(const S *pSrc, int nSrcStep, int srcW, int srcH, const S *pTpl, int nTplStep,
                                int tplW, int tplH, D *pDst, int nDstStep, int dstW, int dstH, int nScaleFactor) {
  constexpr int nCh = Channels - (AlphaPreserve ? 1: 0);
  constexpr bool isNorm = (Op == nppi::statistics::CrossCorrOp::Norm);
  constexpr bool isLevel =
      (Op == nppi::statistics::CrossCorrOp::NormLevel || Op == nppi::statistics::CrossCorrOp::NormLevelAdvanced);
  constexpr bool isSqr = (Op == nppi::statistics::CrossCorrOp::SqrDistance);
  const int total = dstW * dstH * nCh;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = tid; i < total; i += gridDim.x * blockDim.x) {
    const int ci = i % nCh;
    const int px = (i / nCh) % dstW;
    const int py = i / (nCh * dstW);
    int wx, wy;  // window origin in the source image
    if (M == nppi::statistics::CrossCorrMode::Full) {
      wx = px - (tplW - 1);
      wy = py - (tplH - 1);
    } else if (M == nppi::statistics::CrossCorrMode::Same) {
      wx = px - tplW / 2;
      wy = py - tplH / 2;
    } else {
      wx = px;
      wy = py;
    }
    double rst = 0.0;  // sum(w * t) over in-ROI positions
    double rss = 0.0;  // sum(w * w) over in-ROI positions
    double rtt = 0.0;  // sum(t * t) over the full template
    double sw = 0.0;   // sum(w)
    double st = 0.0;   // sum(t) over the full template
    double sd = 0.0;   // sum((w - t)^2) over padded positions
    for (int v = 0; v < tplH; ++v) {
      const int sy = wy + v;
      const bool rowIn = (sy >= 0 && sy < srcH);
      const S *srow =
          rowIn ? reinterpret_cast<const S *>(reinterpret_cast<const char *>(pSrc) + sy * nSrcStep): pSrc;
      const S *trow = reinterpret_cast<const S *>(reinterpret_cast<const char *>(pTpl) + v * nTplStep);
      for (int u = 0; u < tplW; ++u) {
        const double t = (double)trow[u * Channels + ci];
        rtt += t * t;
        st += t;
        const int sx = wx + u;
        double w = 0.0;
        if (rowIn && sx >= 0 && sx < srcW) {
          w = (double)srow[sx * Channels + ci];
          sw += w;
          rss += w * w;
          rst += w * t;
        }
        if (isSqr) {
          const double d = w - t;
          sd += d * d;
        }
      }
    }
    double value = 0.0;
    if (isNorm || isSqr) {
      // reference-measured: for 64f destinations the denominator is rounded
      // through f32 (den = (double)sqrtf((float)(rss * rtt))); pure f64
      // arithmetic is ~4e-8 off the reference output values.
      double den;
      if constexpr (sizeof(D) == 8) {
        den = (double)std::sqrt((float)(rss * rtt));
      } else {
        den = std::sqrt(rss * rtt);
      }
      value = (isSqr ? sd: rst) / den;
    } else if (isLevel) {
      const double n = (double)tplW * (double)tplH;
      const double num = (n * rst - sw * st) / n;
      const double varW = (n * rss - sw * sw) / n;
      const double varT = (n * rtt - st * st) / n;
      const double prod = varW * varT;
      if (prod == 0.0) {
        // reference-measured sentinel: FLT_MAX for <= 4-byte dst (8u Sfs stores
        // 255), DBL_MAX for 64f dst; written to every dst pixel.
        value = (sizeof(D) <= 4) ? (double)std::numeric_limits<float>::max(): (double)std::numeric_limits<double>::max();
      } else {
        // reference-measured: for 64f destinations the denominator is rounded
        // through f32 (den = (double)sqrtf((float)(varW * varT))); pure f64
        // arithmetic is ~4e-8 off the reference output values.
        double den;
        if constexpr (sizeof(D) == 8) {
          den = (double)std::sqrt((float)prod);
        } else {
          den = std::sqrt(prod);
        }
        value = num / den;
      }
    } else {
      value = rst;  // plain valid correlation: raw dot product
    }
    D *drow = reinterpret_cast<D *>(reinterpret_cast<char *>(pDst) + py * nDstStep);
    storeValue<D>(drow + px * Channels + ci, value, nScaleFactor);
  }
}

}  // namespace

namespace nppi {
namespace statistics {

template <typename S, typename D, int Channels, bool AlphaPreserve, CrossCorrMode M, CrossCorrOp Op>
NppStatus runCrossCorr(const S *pSrc, int nSrcStep, NppiSize oSrcRoiSize, const S *pTpl, int nTplStep,
                       NppiSize oTplRoiSize, D *pDst, int nDstStep, int nScaleFactor, Npp8u *pDeviceBuffer,
                       Npp8u *pAdvancedScratchBuffer, cudaStream_t stream) {
  constexpr int esize = (int)sizeof(S);
  constexpr int dsize = (int)sizeof(D);
  // NPP's "even step" modulus is sizeof(element) * 2^v2(Channels): 8u C3
  // accepts odd steps (2^v2(3) = 1), C4 requires multiples of 4x the element
  // size (reference measurements: 8u C4 src step 25 -> -108, 8u C3 src step 19 -> 0).
  constexpr int chAlign = Channels & -Channels;
  constexpr int srcAlign = esize * chAlign;
  constexpr int dstAlign = dsize * chAlign;
  if constexpr (Op == CrossCorrOp::NormLevelAdvanced) {
    if (!pDeviceBuffer || !pAdvancedScratchBuffer) return NPP_NULL_POINTER_ERROR;
  }
  if (!pSrc) return NPP_NULL_POINTER_ERROR;
  if (oSrcRoiSize.width == 0 || oSrcRoiSize.height == 0) return NPP_NO_ERROR;
  if (oSrcRoiSize.width < 0 || oSrcRoiSize.height < 0) return NPP_SIZE_ERROR;
  if (oTplRoiSize.width == 0 || oTplRoiSize.height == 0) return NPP_NO_ERROR;
  if (oTplRoiSize.width < 0 || oTplRoiSize.height < 0) return NPP_SIZE_ERROR;
  // The source step is validated before the template/destination pointers
  // (reference: null dst + bad src step -> NPP_STEP_ERROR; null tpl + bad src
  // step -> NPP_STEP_ERROR).
  if ((unsigned)nSrcStep < (unsigned)(oSrcRoiSize.width * Channels * esize)) return NPP_STEP_ERROR;
  if (nSrcStep % srcAlign != 0) return NPP_NOT_EVEN_STEP_ERROR;
  if (!pTpl || !pDst) return NPP_NULL_POINTER_ERROR;
  int dstW, dstH;
  if (M == CrossCorrMode::Valid) {
    dstW = oSrcRoiSize.width - oTplRoiSize.width + 1;
    dstH = oSrcRoiSize.height - oTplRoiSize.height + 1;
    // A negative destination dimension is a size error; a zero one is the
    // empty-output no-op (reference: dst 5x-1 -> -6, 5x0 and 0x1 -> NO_ERROR).
    if (dstW < 0 || dstH < 0) return NPP_SIZE_ERROR;
    if (dstW == 0 || dstH == 0) return NPP_NO_ERROR;  // steps unchecked
  } else {
    dstW = (M == CrossCorrMode::Full) ? oSrcRoiSize.width + oTplRoiSize.width - 1: oSrcRoiSize.width;
    dstH = (M == CrossCorrMode::Full) ? oSrcRoiSize.height + oTplRoiSize.height - 1: oSrcRoiSize.height;
  }
  if ((unsigned)nDstStep < (unsigned)(dstW * Channels * dsize)) return NPP_STEP_ERROR;
  if (nDstStep % dstAlign != 0) return NPP_NOT_EVEN_STEP_ERROR;
  if ((unsigned)nTplStep < (unsigned)(oTplRoiSize.width * Channels * esize)) return NPP_STEP_ERROR;
  if (nTplStep % srcAlign != 0) return NPP_NOT_EVEN_STEP_ERROR;
  if constexpr (Op == CrossCorrOp::NormLevel) {
    // Before the Full/Same tpl>src geometry error (reference: null buffer + a
    // larger template -> NPP_NULL_POINTER_ERROR).
    if (!pDeviceBuffer) return NPP_NULL_POINTER_ERROR;
  }
  if (M != CrossCorrMode::Valid) {
    if (oTplRoiSize.width > oSrcRoiSize.width || oTplRoiSize.height > oSrcRoiSize.height)
      return NPP_SIZE_ERROR;
  }
  constexpr int nCh = Channels - (AlphaPreserve ? 1: 0);
  const int blocks = numBlocks(dstW * dstH * nCh);
  (void)pDeviceBuffer;
  (void)pAdvancedScratchBuffer;
  (void)cudaGetLastError();
  crossCorrKernel<S, D, Channels, AlphaPreserve, M, Op>
      <<<blocks, kBlock, 0, stream>>>(pSrc, nSrcStep, oSrcRoiSize.width, oSrcRoiSize.height, pTpl, nTplStep,
                                      oTplRoiSize.width, oTplRoiSize.height, pDst, nDstStep, dstW, dstH,
                                      nScaleFactor);
  if (cudaGetLastError() != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

// CrossCorr{Full,Same,Valid}_Norm (8u dst = Sfs)
template NppStatus runCrossCorr<Npp8u,Npp8u,1,false,CrossCorrMode::Full,CrossCorrOp::Norm>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp8u,3,false,CrossCorrMode::Full,CrossCorrOp::Norm>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp8u,4,false,CrossCorrMode::Full,CrossCorrOp::Norm>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp8u,4,true,CrossCorrMode::Full,CrossCorrOp::Norm>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,1,false,CrossCorrMode::Full,CrossCorrOp::Norm>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,3,false,CrossCorrMode::Full,CrossCorrOp::Norm>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,4,false,CrossCorrMode::Full,CrossCorrOp::Norm>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,4,true,CrossCorrMode::Full,CrossCorrOp::Norm>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,1,false,CrossCorrMode::Full,CrossCorrOp::Norm>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,3,false,CrossCorrMode::Full,CrossCorrOp::Norm>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,4,false,CrossCorrMode::Full,CrossCorrOp::Norm>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,4,true,CrossCorrMode::Full,CrossCorrOp::Norm>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,1,false,CrossCorrMode::Full,CrossCorrOp::Norm>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,3,false,CrossCorrMode::Full,CrossCorrOp::Norm>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,4,false,CrossCorrMode::Full,CrossCorrOp::Norm>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,4,true,CrossCorrMode::Full,CrossCorrOp::Norm>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,1,false,CrossCorrMode::Full,CrossCorrOp::Norm>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,3,false,CrossCorrMode::Full,CrossCorrOp::Norm>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,4,false,CrossCorrMode::Full,CrossCorrOp::Norm>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,4,true,CrossCorrMode::Full,CrossCorrOp::Norm>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp64f,Npp64f,1,false,CrossCorrMode::Full,CrossCorrOp::Norm>(const Npp64f*,int,NppiSize,const Npp64f*,int,NppiSize,Npp64f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp64f,Npp64f,3,false,CrossCorrMode::Full,CrossCorrOp::Norm>(const Npp64f*,int,NppiSize,const Npp64f*,int,NppiSize,Npp64f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp64f,Npp64f,4,false,CrossCorrMode::Full,CrossCorrOp::Norm>(const Npp64f*,int,NppiSize,const Npp64f*,int,NppiSize,Npp64f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp64f,Npp64f,4,true,CrossCorrMode::Full,CrossCorrOp::Norm>(const Npp64f*,int,NppiSize,const Npp64f*,int,NppiSize,Npp64f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp8u,1,false,CrossCorrMode::Same,CrossCorrOp::Norm>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp8u,3,false,CrossCorrMode::Same,CrossCorrOp::Norm>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp8u,4,false,CrossCorrMode::Same,CrossCorrOp::Norm>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp8u,4,true,CrossCorrMode::Same,CrossCorrOp::Norm>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,1,false,CrossCorrMode::Same,CrossCorrOp::Norm>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,3,false,CrossCorrMode::Same,CrossCorrOp::Norm>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,4,false,CrossCorrMode::Same,CrossCorrOp::Norm>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,4,true,CrossCorrMode::Same,CrossCorrOp::Norm>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,1,false,CrossCorrMode::Same,CrossCorrOp::Norm>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,3,false,CrossCorrMode::Same,CrossCorrOp::Norm>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,4,false,CrossCorrMode::Same,CrossCorrOp::Norm>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,4,true,CrossCorrMode::Same,CrossCorrOp::Norm>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,1,false,CrossCorrMode::Same,CrossCorrOp::Norm>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,3,false,CrossCorrMode::Same,CrossCorrOp::Norm>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,4,false,CrossCorrMode::Same,CrossCorrOp::Norm>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,4,true,CrossCorrMode::Same,CrossCorrOp::Norm>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,1,false,CrossCorrMode::Same,CrossCorrOp::Norm>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,3,false,CrossCorrMode::Same,CrossCorrOp::Norm>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,4,false,CrossCorrMode::Same,CrossCorrOp::Norm>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,4,true,CrossCorrMode::Same,CrossCorrOp::Norm>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp64f,Npp64f,1,false,CrossCorrMode::Same,CrossCorrOp::Norm>(const Npp64f*,int,NppiSize,const Npp64f*,int,NppiSize,Npp64f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp64f,Npp64f,3,false,CrossCorrMode::Same,CrossCorrOp::Norm>(const Npp64f*,int,NppiSize,const Npp64f*,int,NppiSize,Npp64f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp64f,Npp64f,4,false,CrossCorrMode::Same,CrossCorrOp::Norm>(const Npp64f*,int,NppiSize,const Npp64f*,int,NppiSize,Npp64f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp64f,Npp64f,4,true,CrossCorrMode::Same,CrossCorrOp::Norm>(const Npp64f*,int,NppiSize,const Npp64f*,int,NppiSize,Npp64f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp8u,1,false,CrossCorrMode::Valid,CrossCorrOp::Norm>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp8u,3,false,CrossCorrMode::Valid,CrossCorrOp::Norm>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp8u,4,false,CrossCorrMode::Valid,CrossCorrOp::Norm>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp8u,4,true,CrossCorrMode::Valid,CrossCorrOp::Norm>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,1,false,CrossCorrMode::Valid,CrossCorrOp::Norm>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,3,false,CrossCorrMode::Valid,CrossCorrOp::Norm>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,4,false,CrossCorrMode::Valid,CrossCorrOp::Norm>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,4,true,CrossCorrMode::Valid,CrossCorrOp::Norm>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,1,false,CrossCorrMode::Valid,CrossCorrOp::Norm>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,3,false,CrossCorrMode::Valid,CrossCorrOp::Norm>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,4,false,CrossCorrMode::Valid,CrossCorrOp::Norm>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,4,true,CrossCorrMode::Valid,CrossCorrOp::Norm>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,1,false,CrossCorrMode::Valid,CrossCorrOp::Norm>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,3,false,CrossCorrMode::Valid,CrossCorrOp::Norm>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,4,false,CrossCorrMode::Valid,CrossCorrOp::Norm>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,4,true,CrossCorrMode::Valid,CrossCorrOp::Norm>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,1,false,CrossCorrMode::Valid,CrossCorrOp::Norm>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,3,false,CrossCorrMode::Valid,CrossCorrOp::Norm>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,4,false,CrossCorrMode::Valid,CrossCorrOp::Norm>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,4,true,CrossCorrMode::Valid,CrossCorrOp::Norm>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp64f,Npp64f,1,false,CrossCorrMode::Valid,CrossCorrOp::Norm>(const Npp64f*,int,NppiSize,const Npp64f*,int,NppiSize,Npp64f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp64f,Npp64f,3,false,CrossCorrMode::Valid,CrossCorrOp::Norm>(const Npp64f*,int,NppiSize,const Npp64f*,int,NppiSize,Npp64f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp64f,Npp64f,4,false,CrossCorrMode::Valid,CrossCorrOp::Norm>(const Npp64f*,int,NppiSize,const Npp64f*,int,NppiSize,Npp64f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp64f,Npp64f,4,true,CrossCorrMode::Valid,CrossCorrOp::Norm>(const Npp64f*,int,NppiSize,const Npp64f*,int,NppiSize,Npp64f*,int,int,Npp8u*,Npp8u*,cudaStream_t);

// CrossCorr{Full,Same,Valid}_NormLevel
template NppStatus runCrossCorr<Npp8u,Npp8u,1,false,CrossCorrMode::Full,CrossCorrOp::NormLevel>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp8u,3,false,CrossCorrMode::Full,CrossCorrOp::NormLevel>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp8u,4,false,CrossCorrMode::Full,CrossCorrOp::NormLevel>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp8u,4,true,CrossCorrMode::Full,CrossCorrOp::NormLevel>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,1,false,CrossCorrMode::Full,CrossCorrOp::NormLevel>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,3,false,CrossCorrMode::Full,CrossCorrOp::NormLevel>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,4,false,CrossCorrMode::Full,CrossCorrOp::NormLevel>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,4,true,CrossCorrMode::Full,CrossCorrOp::NormLevel>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,1,false,CrossCorrMode::Full,CrossCorrOp::NormLevel>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,3,false,CrossCorrMode::Full,CrossCorrOp::NormLevel>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,4,false,CrossCorrMode::Full,CrossCorrOp::NormLevel>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,4,true,CrossCorrMode::Full,CrossCorrOp::NormLevel>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,1,false,CrossCorrMode::Full,CrossCorrOp::NormLevel>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,3,false,CrossCorrMode::Full,CrossCorrOp::NormLevel>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,4,false,CrossCorrMode::Full,CrossCorrOp::NormLevel>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,4,true,CrossCorrMode::Full,CrossCorrOp::NormLevel>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,1,false,CrossCorrMode::Full,CrossCorrOp::NormLevel>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,3,false,CrossCorrMode::Full,CrossCorrOp::NormLevel>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,4,false,CrossCorrMode::Full,CrossCorrOp::NormLevel>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,4,true,CrossCorrMode::Full,CrossCorrOp::NormLevel>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp64f,Npp64f,1,false,CrossCorrMode::Full,CrossCorrOp::NormLevel>(const Npp64f*,int,NppiSize,const Npp64f*,int,NppiSize,Npp64f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp64f,Npp64f,3,false,CrossCorrMode::Full,CrossCorrOp::NormLevel>(const Npp64f*,int,NppiSize,const Npp64f*,int,NppiSize,Npp64f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp64f,Npp64f,4,false,CrossCorrMode::Full,CrossCorrOp::NormLevel>(const Npp64f*,int,NppiSize,const Npp64f*,int,NppiSize,Npp64f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp64f,Npp64f,4,true,CrossCorrMode::Full,CrossCorrOp::NormLevel>(const Npp64f*,int,NppiSize,const Npp64f*,int,NppiSize,Npp64f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp8u,1,false,CrossCorrMode::Same,CrossCorrOp::NormLevel>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp8u,3,false,CrossCorrMode::Same,CrossCorrOp::NormLevel>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp8u,4,false,CrossCorrMode::Same,CrossCorrOp::NormLevel>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp8u,4,true,CrossCorrMode::Same,CrossCorrOp::NormLevel>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,1,false,CrossCorrMode::Same,CrossCorrOp::NormLevel>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,3,false,CrossCorrMode::Same,CrossCorrOp::NormLevel>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,4,false,CrossCorrMode::Same,CrossCorrOp::NormLevel>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,4,true,CrossCorrMode::Same,CrossCorrOp::NormLevel>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,1,false,CrossCorrMode::Same,CrossCorrOp::NormLevel>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,3,false,CrossCorrMode::Same,CrossCorrOp::NormLevel>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,4,false,CrossCorrMode::Same,CrossCorrOp::NormLevel>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,4,true,CrossCorrMode::Same,CrossCorrOp::NormLevel>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,1,false,CrossCorrMode::Same,CrossCorrOp::NormLevel>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,3,false,CrossCorrMode::Same,CrossCorrOp::NormLevel>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,4,false,CrossCorrMode::Same,CrossCorrOp::NormLevel>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,4,true,CrossCorrMode::Same,CrossCorrOp::NormLevel>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,1,false,CrossCorrMode::Same,CrossCorrOp::NormLevel>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,3,false,CrossCorrMode::Same,CrossCorrOp::NormLevel>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,4,false,CrossCorrMode::Same,CrossCorrOp::NormLevel>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,4,true,CrossCorrMode::Same,CrossCorrOp::NormLevel>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp64f,Npp64f,1,false,CrossCorrMode::Same,CrossCorrOp::NormLevel>(const Npp64f*,int,NppiSize,const Npp64f*,int,NppiSize,Npp64f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp64f,Npp64f,3,false,CrossCorrMode::Same,CrossCorrOp::NormLevel>(const Npp64f*,int,NppiSize,const Npp64f*,int,NppiSize,Npp64f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp64f,Npp64f,4,false,CrossCorrMode::Same,CrossCorrOp::NormLevel>(const Npp64f*,int,NppiSize,const Npp64f*,int,NppiSize,Npp64f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp64f,Npp64f,4,true,CrossCorrMode::Same,CrossCorrOp::NormLevel>(const Npp64f*,int,NppiSize,const Npp64f*,int,NppiSize,Npp64f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp8u,1,false,CrossCorrMode::Valid,CrossCorrOp::NormLevel>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp8u,3,false,CrossCorrMode::Valid,CrossCorrOp::NormLevel>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp8u,4,false,CrossCorrMode::Valid,CrossCorrOp::NormLevel>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp8u,4,true,CrossCorrMode::Valid,CrossCorrOp::NormLevel>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,1,false,CrossCorrMode::Valid,CrossCorrOp::NormLevel>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,3,false,CrossCorrMode::Valid,CrossCorrOp::NormLevel>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,4,false,CrossCorrMode::Valid,CrossCorrOp::NormLevel>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,4,true,CrossCorrMode::Valid,CrossCorrOp::NormLevel>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,1,false,CrossCorrMode::Valid,CrossCorrOp::NormLevel>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,3,false,CrossCorrMode::Valid,CrossCorrOp::NormLevel>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,4,false,CrossCorrMode::Valid,CrossCorrOp::NormLevel>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,4,true,CrossCorrMode::Valid,CrossCorrOp::NormLevel>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,1,false,CrossCorrMode::Valid,CrossCorrOp::NormLevel>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,3,false,CrossCorrMode::Valid,CrossCorrOp::NormLevel>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,4,false,CrossCorrMode::Valid,CrossCorrOp::NormLevel>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,4,true,CrossCorrMode::Valid,CrossCorrOp::NormLevel>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,1,false,CrossCorrMode::Valid,CrossCorrOp::NormLevel>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,3,false,CrossCorrMode::Valid,CrossCorrOp::NormLevel>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,4,false,CrossCorrMode::Valid,CrossCorrOp::NormLevel>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,4,true,CrossCorrMode::Valid,CrossCorrOp::NormLevel>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp64f,Npp64f,1,false,CrossCorrMode::Valid,CrossCorrOp::NormLevel>(const Npp64f*,int,NppiSize,const Npp64f*,int,NppiSize,Npp64f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp64f,Npp64f,3,false,CrossCorrMode::Valid,CrossCorrOp::NormLevel>(const Npp64f*,int,NppiSize,const Npp64f*,int,NppiSize,Npp64f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp64f,Npp64f,4,false,CrossCorrMode::Valid,CrossCorrOp::NormLevel>(const Npp64f*,int,NppiSize,const Npp64f*,int,NppiSize,Npp64f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp64f,Npp64f,4,true,CrossCorrMode::Valid,CrossCorrOp::NormLevel>(const Npp64f*,int,NppiSize,const Npp64f*,int,NppiSize,Npp64f*,int,int,Npp8u*,Npp8u*,cudaStream_t);

// CrossCorr{Full,Same,Valid}_NormLevelAdvanced (C1/C3/C4, no Sfs)
template NppStatus runCrossCorr<Npp8u,Npp32f,1,false,CrossCorrMode::Full,CrossCorrOp::NormLevelAdvanced>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,3,false,CrossCorrMode::Full,CrossCorrOp::NormLevelAdvanced>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,4,false,CrossCorrMode::Full,CrossCorrOp::NormLevelAdvanced>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,1,false,CrossCorrMode::Full,CrossCorrOp::NormLevelAdvanced>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,3,false,CrossCorrMode::Full,CrossCorrOp::NormLevelAdvanced>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,4,false,CrossCorrMode::Full,CrossCorrOp::NormLevelAdvanced>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,1,false,CrossCorrMode::Full,CrossCorrOp::NormLevelAdvanced>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,3,false,CrossCorrMode::Full,CrossCorrOp::NormLevelAdvanced>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,4,false,CrossCorrMode::Full,CrossCorrOp::NormLevelAdvanced>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,1,false,CrossCorrMode::Full,CrossCorrOp::NormLevelAdvanced>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,3,false,CrossCorrMode::Full,CrossCorrOp::NormLevelAdvanced>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,4,false,CrossCorrMode::Full,CrossCorrOp::NormLevelAdvanced>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp64f,Npp64f,1,false,CrossCorrMode::Full,CrossCorrOp::NormLevelAdvanced>(const Npp64f*,int,NppiSize,const Npp64f*,int,NppiSize,Npp64f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp64f,Npp64f,3,false,CrossCorrMode::Full,CrossCorrOp::NormLevelAdvanced>(const Npp64f*,int,NppiSize,const Npp64f*,int,NppiSize,Npp64f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp64f,Npp64f,4,false,CrossCorrMode::Full,CrossCorrOp::NormLevelAdvanced>(const Npp64f*,int,NppiSize,const Npp64f*,int,NppiSize,Npp64f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,1,false,CrossCorrMode::Same,CrossCorrOp::NormLevelAdvanced>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,3,false,CrossCorrMode::Same,CrossCorrOp::NormLevelAdvanced>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,4,false,CrossCorrMode::Same,CrossCorrOp::NormLevelAdvanced>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,1,false,CrossCorrMode::Same,CrossCorrOp::NormLevelAdvanced>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,3,false,CrossCorrMode::Same,CrossCorrOp::NormLevelAdvanced>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,4,false,CrossCorrMode::Same,CrossCorrOp::NormLevelAdvanced>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,1,false,CrossCorrMode::Same,CrossCorrOp::NormLevelAdvanced>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,3,false,CrossCorrMode::Same,CrossCorrOp::NormLevelAdvanced>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,4,false,CrossCorrMode::Same,CrossCorrOp::NormLevelAdvanced>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,1,false,CrossCorrMode::Same,CrossCorrOp::NormLevelAdvanced>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,3,false,CrossCorrMode::Same,CrossCorrOp::NormLevelAdvanced>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,4,false,CrossCorrMode::Same,CrossCorrOp::NormLevelAdvanced>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp64f,Npp64f,1,false,CrossCorrMode::Same,CrossCorrOp::NormLevelAdvanced>(const Npp64f*,int,NppiSize,const Npp64f*,int,NppiSize,Npp64f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp64f,Npp64f,3,false,CrossCorrMode::Same,CrossCorrOp::NormLevelAdvanced>(const Npp64f*,int,NppiSize,const Npp64f*,int,NppiSize,Npp64f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp64f,Npp64f,4,false,CrossCorrMode::Same,CrossCorrOp::NormLevelAdvanced>(const Npp64f*,int,NppiSize,const Npp64f*,int,NppiSize,Npp64f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,1,false,CrossCorrMode::Valid,CrossCorrOp::NormLevelAdvanced>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,3,false,CrossCorrMode::Valid,CrossCorrOp::NormLevelAdvanced>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,4,false,CrossCorrMode::Valid,CrossCorrOp::NormLevelAdvanced>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,1,false,CrossCorrMode::Valid,CrossCorrOp::NormLevelAdvanced>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,3,false,CrossCorrMode::Valid,CrossCorrOp::NormLevelAdvanced>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,4,false,CrossCorrMode::Valid,CrossCorrOp::NormLevelAdvanced>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,1,false,CrossCorrMode::Valid,CrossCorrOp::NormLevelAdvanced>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,3,false,CrossCorrMode::Valid,CrossCorrOp::NormLevelAdvanced>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,4,false,CrossCorrMode::Valid,CrossCorrOp::NormLevelAdvanced>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,1,false,CrossCorrMode::Valid,CrossCorrOp::NormLevelAdvanced>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,3,false,CrossCorrMode::Valid,CrossCorrOp::NormLevelAdvanced>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,4,false,CrossCorrMode::Valid,CrossCorrOp::NormLevelAdvanced>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp64f,Npp64f,1,false,CrossCorrMode::Valid,CrossCorrOp::NormLevelAdvanced>(const Npp64f*,int,NppiSize,const Npp64f*,int,NppiSize,Npp64f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp64f,Npp64f,3,false,CrossCorrMode::Valid,CrossCorrOp::NormLevelAdvanced>(const Npp64f*,int,NppiSize,const Npp64f*,int,NppiSize,Npp64f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp64f,Npp64f,4,false,CrossCorrMode::Valid,CrossCorrOp::NormLevelAdvanced>(const Npp64f*,int,NppiSize,const Npp64f*,int,NppiSize,Npp64f*,int,int,Npp8u*,Npp8u*,cudaStream_t);

// CrossCorrValid (plain dot product, C1)
template NppStatus runCrossCorr<Npp8u,Npp32f,1,false,CrossCorrMode::Valid,CrossCorrOp::Plain>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,1,false,CrossCorrMode::Valid,CrossCorrOp::Plain>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,1,false,CrossCorrMode::Valid,CrossCorrOp::Plain>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,1,false,CrossCorrMode::Valid,CrossCorrOp::Plain>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp64f,Npp64f,1,false,CrossCorrMode::Valid,CrossCorrOp::Plain>(const Npp64f*,int,NppiSize,const Npp64f*,int,NppiSize,Npp64f*,int,int,Npp8u*,Npp8u*,cudaStream_t);

// SqrDistance{Full,Same,Valid}_Norm (no 64f)
template NppStatus runCrossCorr<Npp8u,Npp8u,1,false,CrossCorrMode::Full,CrossCorrOp::SqrDistance>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp8u,3,false,CrossCorrMode::Full,CrossCorrOp::SqrDistance>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp8u,4,false,CrossCorrMode::Full,CrossCorrOp::SqrDistance>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp8u,4,true,CrossCorrMode::Full,CrossCorrOp::SqrDistance>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,1,false,CrossCorrMode::Full,CrossCorrOp::SqrDistance>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,3,false,CrossCorrMode::Full,CrossCorrOp::SqrDistance>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,4,false,CrossCorrMode::Full,CrossCorrOp::SqrDistance>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,4,true,CrossCorrMode::Full,CrossCorrOp::SqrDistance>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,1,false,CrossCorrMode::Full,CrossCorrOp::SqrDistance>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,3,false,CrossCorrMode::Full,CrossCorrOp::SqrDistance>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,4,false,CrossCorrMode::Full,CrossCorrOp::SqrDistance>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,4,true,CrossCorrMode::Full,CrossCorrOp::SqrDistance>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,1,false,CrossCorrMode::Full,CrossCorrOp::SqrDistance>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,3,false,CrossCorrMode::Full,CrossCorrOp::SqrDistance>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,4,false,CrossCorrMode::Full,CrossCorrOp::SqrDistance>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,4,true,CrossCorrMode::Full,CrossCorrOp::SqrDistance>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,1,false,CrossCorrMode::Full,CrossCorrOp::SqrDistance>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,3,false,CrossCorrMode::Full,CrossCorrOp::SqrDistance>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,4,false,CrossCorrMode::Full,CrossCorrOp::SqrDistance>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,4,true,CrossCorrMode::Full,CrossCorrOp::SqrDistance>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp8u,1,false,CrossCorrMode::Same,CrossCorrOp::SqrDistance>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp8u,3,false,CrossCorrMode::Same,CrossCorrOp::SqrDistance>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp8u,4,false,CrossCorrMode::Same,CrossCorrOp::SqrDistance>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp8u,4,true,CrossCorrMode::Same,CrossCorrOp::SqrDistance>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,1,false,CrossCorrMode::Same,CrossCorrOp::SqrDistance>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,3,false,CrossCorrMode::Same,CrossCorrOp::SqrDistance>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,4,false,CrossCorrMode::Same,CrossCorrOp::SqrDistance>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,4,true,CrossCorrMode::Same,CrossCorrOp::SqrDistance>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,1,false,CrossCorrMode::Same,CrossCorrOp::SqrDistance>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,3,false,CrossCorrMode::Same,CrossCorrOp::SqrDistance>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,4,false,CrossCorrMode::Same,CrossCorrOp::SqrDistance>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,4,true,CrossCorrMode::Same,CrossCorrOp::SqrDistance>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,1,false,CrossCorrMode::Same,CrossCorrOp::SqrDistance>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,3,false,CrossCorrMode::Same,CrossCorrOp::SqrDistance>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,4,false,CrossCorrMode::Same,CrossCorrOp::SqrDistance>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,4,true,CrossCorrMode::Same,CrossCorrOp::SqrDistance>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,1,false,CrossCorrMode::Same,CrossCorrOp::SqrDistance>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,3,false,CrossCorrMode::Same,CrossCorrOp::SqrDistance>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,4,false,CrossCorrMode::Same,CrossCorrOp::SqrDistance>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,4,true,CrossCorrMode::Same,CrossCorrOp::SqrDistance>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp8u,1,false,CrossCorrMode::Valid,CrossCorrOp::SqrDistance>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp8u,3,false,CrossCorrMode::Valid,CrossCorrOp::SqrDistance>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp8u,4,false,CrossCorrMode::Valid,CrossCorrOp::SqrDistance>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp8u,4,true,CrossCorrMode::Valid,CrossCorrOp::SqrDistance>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp8u*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,1,false,CrossCorrMode::Valid,CrossCorrOp::SqrDistance>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,3,false,CrossCorrMode::Valid,CrossCorrOp::SqrDistance>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,4,false,CrossCorrMode::Valid,CrossCorrOp::SqrDistance>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8u,Npp32f,4,true,CrossCorrMode::Valid,CrossCorrOp::SqrDistance>(const Npp8u*,int,NppiSize,const Npp8u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,1,false,CrossCorrMode::Valid,CrossCorrOp::SqrDistance>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,3,false,CrossCorrMode::Valid,CrossCorrOp::SqrDistance>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,4,false,CrossCorrMode::Valid,CrossCorrOp::SqrDistance>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp8s,Npp32f,4,true,CrossCorrMode::Valid,CrossCorrOp::SqrDistance>(const Npp8s*,int,NppiSize,const Npp8s*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,1,false,CrossCorrMode::Valid,CrossCorrOp::SqrDistance>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,3,false,CrossCorrMode::Valid,CrossCorrOp::SqrDistance>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,4,false,CrossCorrMode::Valid,CrossCorrOp::SqrDistance>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp16u,Npp32f,4,true,CrossCorrMode::Valid,CrossCorrOp::SqrDistance>(const Npp16u*,int,NppiSize,const Npp16u*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,1,false,CrossCorrMode::Valid,CrossCorrOp::SqrDistance>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,3,false,CrossCorrMode::Valid,CrossCorrOp::SqrDistance>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,4,false,CrossCorrMode::Valid,CrossCorrOp::SqrDistance>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);
template NppStatus runCrossCorr<Npp32f,Npp32f,4,true,CrossCorrMode::Valid,CrossCorrOp::SqrDistance>(const Npp32f*,int,NppiSize,const Npp32f*,int,NppiSize,Npp32f*,int,int,Npp8u*,Npp8u*,cudaStream_t);

}  // namespace statistics
}  // namespace nppi
