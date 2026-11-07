// c3 executor: batch color conversions (nppi_color_conversion.h).
//
// The batch entry points are host-driven: the descriptor arrays live in
// device memory, so each call copies them to the host and then launches one
// inverse-colour kernel per image pair.
//
// The pixel math is the one measured on NVIDIA NPP 13.4 (reference) for these
// inverse conversions.  Both families are single-precision evaluations with a
// fused multiply-add and C-style truncation after saturation to [0,255]:
//
//   YUV   (BT.601 full range):
//     R = Y + 1.13983 V,  G = Y - 0.39465 U - 0.58060 V,
//     B = Y + 2.03211 U
//   YCbCr (BT.601 video range):
//     Y' = 1.164 (Y - 16), R = Y' + 1.596 Cr,
//     G = Y' - 0.392 Cb - 0.813 Cr, B = Y' + 2.017 Cb
//
// The FMA evaluation order matters at rounding boundaries: it reproduces the
// reference bit-exactly for 4:2:0, 4:2:2 and 4:4:4 inputs (validated by
// dumping the reference 4:4:4/4:2:0/4:2:2 conversions over random planes).
//
// Batch semantics measured on the reference:
//   * nBatchSize <= 1 -> NPP_BAD_ARGUMENT_ERROR.
//   * plain variants use oSizeROI for every pair; _Advanced variants use
//     pDstBatchList[i].oSize (oMaxSizeROI is ignored by the reference).
//   * an odd ROI is converted up to the largest even rectangle and reported
//     as NPP_DOUBLE_SIZE_WARNING (4:2:2 shrinks the width, 4:2:0 both).
//   * per-image pointers and steps are not validated by the reference (an
//     invalid one faults its kernel).  This executor validates them instead
//     of touching memory and reports NPP_NULL_POINTER_ERROR / NPP_STEP_ERROR.
//
// Every plane is read with its own descriptor nStep (verified against the
// reference with distinct per-plane steps).
#include "nppi_c3_batch_executor.h"

#include <cuda_runtime.h>

#include <vector>

namespace nppi {
namespace c3batch {
namespace {

// How the sampling format constrains the ROI.
enum RoiShrink { kNoShrink = 0, kShrinkWidth = 1, kShrinkWidthHeight = 2 };

RoiShrink shrinkMode(Conversion conv) {
  switch (conv) {
  case kYCbCr420ToBGR:
  case kYCbCr420ToRGB:
  case kYUV420ToBGR:
  case kYUV420ToRGB:
    return kShrinkWidthHeight;
  case kYCbCr422ToBGR:
  case kYCbCr422ToRGB:
  case kYUV422ToBGR:
  case kYUV422ToRGB:
    return kShrinkWidth;
  default:
    return kNoShrink;
  }
}

bool isYuv(Conversion conv) {
  switch (conv) {
  case kYUV420ToBGR:
  case kYUV420ToRGB:
  case kYUV422ToBGR:
  case kYUV422ToRGB:
  case kYUVToBGRP3:
  case kYUVToRGBP3:
  case kYUVToBGRC3:
  case kYUVToRGBC3:
    return true;
  default:
    return false;
  }
}

bool isBgr(Conversion conv) {
  switch (conv) {
  case kYCbCr420ToBGR:
  case kYUV420ToBGR:
  case kYCbCr422ToBGR:
  case kYUV422ToBGR:
  case kYCbCrToBGRP3:
  case kYUVToBGRP3:
  case kYCbCrToBGRC3:
  case kYUVToBGRC3:
    return true;
  default:
    return false;
  }
}

// 0 = 4:4:4, 2 = 4:2:2, 3 = 4:2:0 (vertical subsampling factor 2 when == 3).
int subsampleCode(Conversion conv) {
  switch (conv) {
  case kYCbCr420ToBGR:
  case kYCbCr420ToRGB:
  case kYUV420ToBGR:
  case kYUV420ToRGB:
    return 3;
  case kYCbCr422ToBGR:
  case kYCbCr422ToRGB:
  case kYUV422ToBGR:
  case kYUV422ToRGB:
    return 2;
  default:
    return 0;
  }
}

__device__ __forceinline__ Npp8u saturateU8(float v) {
  v = fminf(fmaxf(v, 0.0f), 255.0f);
  return static_cast<Npp8u>(v);
}

// Inverse colour transform of one pixel (kYuv selects the full-range YUV
// matrix, otherwise the BT.601 video-range YCbCr matrix).
template <bool kYuv>
__device__ __forceinline__ void inversePixel(int y, int cb, int cr, Npp8u &r, Npp8u &g, Npp8u &b) {
  if (kYuv) {
    const float Y = static_cast<float>(y);
    const float U = static_cast<float>(cb - 128);
    const float V = static_cast<float>(cr - 128);
    r = saturateU8(fmaf(1.13983f, V, Y));
    g = saturateU8(fmaf(-0.58060f, V, fmaf(-0.39465f, U, Y)));
    b = saturateU8(fmaf(2.03211f, U, Y));
  } else {
    const float Y = 1.164f * (static_cast<float>(y) - 16.0f);
    const float Cb = static_cast<float>(cb - 128);
    const float Cr = static_cast<float>(cr - 128);
    r = saturateU8(fmaf(1.596f, Cr, Y));
    g = saturateU8(fmaf(-0.813f, Cr, fmaf(-0.392f, Cb, Y)));
    b = saturateU8(fmaf(2.017f, Cb, Y));
  }
}

// Planar source -> packed destination.  kSubX/kSubY are the chroma
// subsampling factors (1,1 for 4:4:4; 2,1 for 4:2:2; 2,2 for 4:2:0).
template <bool kBgr, bool kYuv, int kSubX, int kSubY>
__global__ void planarToPackedKernel(const Npp8u *pY, int nYStep, const Npp8u *pCb, int nCbStep, const Npp8u *pCr,
                                     int nCrStep, Npp8u *pDst, int nDstStep, int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const int cx = x / kSubX;
  const int cy = y / kSubY;
  const Npp8u *pSrcRowY = pY + static_cast<size_t>(y) * nYStep;
  const Npp8u *pSrcRowCb = pCb + static_cast<size_t>(cy) * nCbStep;
  const Npp8u *pSrcRowCr = pCr + static_cast<size_t>(cy) * nCrStep;
  Npp8u *pDstPixel = pDst + static_cast<size_t>(y) * nDstStep + 3 * x;
  Npp8u r, g, b;
  inversePixel<kYuv>(pSrcRowY[x], pSrcRowCb[cx], pSrcRowCr[cx], r, g, b);
  pDstPixel[0] = kBgr ? b: r;
  pDstPixel[1] = g;
  pDstPixel[2] = kBgr ? r: b;
}

// Packed 4:4:4 source -> packed destination.
template <bool kBgr, bool kYuv>
__global__ void packedKernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const Npp8u *pSrcPixel = pSrc + static_cast<size_t>(y) * nSrcStep + 3 * x;
  Npp8u *pDstPixel = pDst + static_cast<size_t>(y) * nDstStep + 3 * x;
  Npp8u r, g, b;
  inversePixel<kYuv>(pSrcPixel[0], pSrcPixel[1], pSrcPixel[2], r, g, b);
  pDstPixel[0] = kBgr ? b: r;
  pDstPixel[1] = g;
  pDstPixel[2] = kBgr ? r: b;
}

NppStatus validateDescriptor(const NppiImageDescriptor &d) {
  if (!d.pData) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (d.nStep <= 0) {
    return NPP_STEP_ERROR;
  }
  return NPP_NO_ERROR;
}

// Apply the sampling-format ROI constraint.  Returns the effective ROI and
// sets `warned` when an odd extent had to be dropped.
NppStatus effectiveRoi(Conversion conv, NppiSize roi, NppiSize &out, bool &warned) {
  if (roi.width <= 0 || roi.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  out = roi;
  const RoiShrink shrink = shrinkMode(conv);
  if (shrink != kNoShrink && (out.width & 1)) {
    out.width -= 1;
    warned = true;
  }
  if (shrink == kShrinkWidthHeight && (out.height & 1)) {
    out.height -= 1;
    warned = true;
  }
  if (out.width <= 0 || out.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  return NPP_NO_ERROR;
}

template <bool kBgr, bool kYuv, int kSubX, int kSubY>
NppStatus launchPlanar(const NppiImageDescriptor *const pSrc[3], const NppiImageDescriptor &dst, NppiSize roi,
                       cudaStream_t stream) {
  const dim3 block(16, 16);
  const dim3 grid((roi.width + 15) / 16, (roi.height + 15) / 16);
  // Drop any stale launch error left by the caller's stream setup.
  cudaGetLastError();
  planarToPackedKernel<kBgr, kYuv, kSubX, kSubY><<<grid, block, 0, stream>>>(
      static_cast<const Npp8u *>(pSrc[0]->pData), pSrc[0]->nStep, static_cast<const Npp8u *>(pSrc[1]->pData),
      pSrc[1]->nStep, static_cast<const Npp8u *>(pSrc[2]->pData), pSrc[2]->nStep,
      static_cast<Npp8u *>(dst.pData), dst.nStep, roi.width, roi.height);
  return (cudaGetLastError() == cudaSuccess) ? NPP_NO_ERROR: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

template <bool kBgr, bool kYuv>
NppStatus launchPacked(const NppiImageDescriptor &src, const NppiImageDescriptor &dst, NppiSize roi,
                       cudaStream_t stream) {
  const dim3 block(16, 16);
  const dim3 grid((roi.width + 15) / 16, (roi.height + 15) / 16);
  cudaGetLastError();
  packedKernel<kBgr, kYuv><<<grid, block, 0, stream>>>(static_cast<const Npp8u *>(src.pData), src.nStep,
                                                       static_cast<Npp8u *>(dst.pData), dst.nStep, roi.width,
                                                       roi.height);
  return (cudaGetLastError() == cudaSuccess) ? NPP_NO_ERROR: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus dispatchPlanar(Conversion conv, const NppiImageDescriptor *const pSrc[3], const NppiImageDescriptor &dst,
                         NppiSize roi, cudaStream_t stream) {
  const bool bgr = isBgr(conv);
  const bool yuv = isYuv(conv);
  switch (subsampleCode(conv)) {
  case 3:
    if (bgr && yuv) return launchPlanar<true, true, 2, 2>(pSrc, dst, roi, stream);
    if (bgr) return launchPlanar<true, false, 2, 2>(pSrc, dst, roi, stream);
    if (yuv) return launchPlanar<false, true, 2, 2>(pSrc, dst, roi, stream);
    return launchPlanar<false, false, 2, 2>(pSrc, dst, roi, stream);
  case 2:
    if (bgr && yuv) return launchPlanar<true, true, 2, 1>(pSrc, dst, roi, stream);
    if (bgr) return launchPlanar<true, false, 2, 1>(pSrc, dst, roi, stream);
    if (yuv) return launchPlanar<false, true, 2, 1>(pSrc, dst, roi, stream);
    return launchPlanar<false, false, 2, 1>(pSrc, dst, roi, stream);
  default:
    if (bgr && yuv) return launchPlanar<true, true, 1, 1>(pSrc, dst, roi, stream);
    if (bgr) return launchPlanar<true, false, 1, 1>(pSrc, dst, roi, stream);
    if (yuv) return launchPlanar<false, true, 1, 1>(pSrc, dst, roi, stream);
    return launchPlanar<false, false, 1, 1>(pSrc, dst, roi, stream);
  }
}

NppStatus dispatchPacked(Conversion conv, const NppiImageDescriptor &src, const NppiImageDescriptor &dst,
                         NppiSize roi, cudaStream_t stream) {
  const bool bgr = isBgr(conv);
  const bool yuv = isYuv(conv);
  if (bgr && yuv) return launchPacked<true, true>(src, dst, roi, stream);
  if (bgr) return launchPacked<true, false>(src, dst, roi, stream);
  if (yuv) return launchPacked<false, true>(src, dst, roi, stream);
  return launchPacked<false, false>(src, dst, roi, stream);
}

NppStatus copyDescriptors(const void *srcList, void *dstList, int nBatchSize, cudaStream_t stream,
                          std::vector<NppiImageDescriptor> &hSrc0, std::vector<NppiImageDescriptor> &hDst) {
  const size_t bytes = static_cast<size_t>(nBatchSize) * sizeof(NppiImageDescriptor);
  if (cudaMemcpyAsync(hSrc0.data(), srcList, bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaMemcpyAsync(hDst.data(), dstList, bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
    return NPP_MEMCPY_ERROR;
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
  return NPP_NO_ERROR;
}

} // namespace

NppStatus runPlanar(const NppiImageDescriptor *const pSrc[3], NppiImageDescriptor *pDst, int nBatchSize,
                    NppiSize oSizeROI, cudaStream_t stream, Conversion conv, int advanced) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pSrc[2] || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nBatchSize <= 1) {
    return NPP_BAD_ARGUMENT_ERROR;
  }
  const int nb = nBatchSize;
  std::vector<NppiImageDescriptor> hSrc0(static_cast<size_t>(nb)), hSrc1(static_cast<size_t>(nb)),
      hSrc2(static_cast<size_t>(nb)), hDst(static_cast<size_t>(nb));
  const size_t bytes = static_cast<size_t>(nb) * sizeof(NppiImageDescriptor);
  if (cudaMemcpyAsync(hSrc0.data(), pSrc[0], bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaMemcpyAsync(hSrc1.data(), pSrc[1], bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaMemcpyAsync(hSrc2.data(), pSrc[2], bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaMemcpyAsync(hDst.data(), pDst, bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
    return NPP_MEMCPY_ERROR;
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  bool warned = false;
  for (int i = 0; i < nb; ++i) {
    const NppiImageDescriptor *srcPlanes[3] = {&hSrc0[i], &hSrc1[i], &hSrc2[i]};
    for (int p = 0; p < 3; ++p) {
      NppStatus status = validateDescriptor(*srcPlanes[p]);
      if (status != NPP_NO_ERROR) {
        return status;
      }
    }
    NppStatus status = validateDescriptor(hDst[i]);
    if (status != NPP_NO_ERROR) {
      return status;
    }
    NppiSize roi{};
    status = effectiveRoi(conv, advanced ? hDst[i].oSize: oSizeROI, roi, warned);
    if (status != NPP_NO_ERROR) {
      return status;
    }
    status = dispatchPlanar(conv, srcPlanes, hDst[i], roi, stream);
    if (status == NPP_DOUBLE_SIZE_WARNING) {
      warned = true;
    } else if (status != NPP_NO_ERROR) {
      return status;
    }
  }
  return warned ? NPP_DOUBLE_SIZE_WARNING: NPP_NO_ERROR;
}

NppStatus runPacked(const NppiImageDescriptor *pSrc, NppiImageDescriptor *pDst, int nBatchSize, NppiSize oSizeROI,
                    cudaStream_t stream, Conversion conv, int advanced) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nBatchSize <= 1) {
    return NPP_BAD_ARGUMENT_ERROR;
  }
  const int nb = nBatchSize;
  std::vector<NppiImageDescriptor> hSrc(static_cast<size_t>(nb)), hDst(static_cast<size_t>(nb));
  NppStatus status = copyDescriptors(pSrc, pDst, nb, stream, hSrc, hDst);
  if (status != NPP_NO_ERROR) {
    return status;
  }

  bool warned = false;
  for (int i = 0; i < nb; ++i) {
    status = validateDescriptor(hSrc[i]);
    if (status != NPP_NO_ERROR) {
      return status;
    }
    status = validateDescriptor(hDst[i]);
    if (status != NPP_NO_ERROR) {
      return status;
    }
    NppiSize roi{};
    status = effectiveRoi(conv, advanced ? hDst[i].oSize: oSizeROI, roi, warned);
    if (status != NPP_NO_ERROR) {
      return status;
    }
    status = dispatchPacked(conv, hSrc[i], hDst[i], roi, stream);
    if (status == NPP_DOUBLE_SIZE_WARNING) {
      warned = true;
    } else if (status != NPP_NO_ERROR) {
      return status;
    }
  }
  return warned ? NPP_DOUBLE_SIZE_WARNING: NPP_NO_ERROR;
}

} // namespace c3batch
} // namespace nppi
