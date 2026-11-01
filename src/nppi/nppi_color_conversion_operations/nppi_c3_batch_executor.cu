// c3 package executor: batch color conversions (nppi_color_conversion.h).
//
// The batch entry points are host-driven: the descriptor arrays live in
// device memory, so each call copies them to the host and then runs the
// matching single-image conversion per pair.  This keeps the pixel contract
// bit-identical to the single-image families implemented in
// nppi_c3_conversion_executor.cu / nppi_c3_color_executor.cu and the older
// per-image entry points (nppi_ycbcr420_rgb.cpp, nppi_yuv422_rgb.cpp,
// nppi_rgb_yuv.cpp).
//
// Recorded NVIDIA NPP 13.4 (RTX 5090) semantics:
//   * nBatchSize <= 1 -> NPP_BAD_ARGUMENT_ERROR.
//   * plain variants use oSizeROI for every pair; _Advanced variants use
//     pDstBatchList[i].oSize (oMaxSizeROI is ignored by the reference).
//   * an odd ROI is converted up to the largest even rectangle and reported
//     as NPP_DOUBLE_SIZE_WARNING (4:2:2 shrinks the width, 4:2:0 both).
//   * per-image pointers/steps are not validated by the reference; invalid
//     ones fault its kernel.  This executor validates them instead of
//     touching memory, and reports NPP_NULL_POINTER_ERROR / NPP_STEP_ERROR.
#include "nppi_c3_batch_executor.h"

#include <cuda_runtime.h>

#include <vector>

// Per-plane-step 4:4:4 P3C3R kernels in nppi_rgb_yuv.cu (the public
// single-image entry points take one step for all three planes).
extern "C" {
cudaError_t nppiYUVToRGB_8u_P3C3R_Step3_kernel(const Npp8u *pSrcY, int nStepY, const Npp8u *pSrcU, int nStepU,
                                               const Npp8u *pSrcV, int nStepV, Npp8u *pDst, int nDstStep,
                                               NppiSize oSizeROI, int bgr, cudaStream_t stream);
cudaError_t nppiYCbCrToRGB_8u_P3C3R_Step3_kernel(const Npp8u *pSrcY, int nStepY, const Npp8u *pSrcCb, int nStepCb,
                                                 const Npp8u *pSrcCr, int nStepCr, Npp8u *pDst, int nDstStep,
                                                 NppiSize oSizeROI, int bgr, cudaStream_t stream);
}

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

NppStatus kernelStatus(cudaError_t err) {
  return (err == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
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

__global__ void swapRedBlueKernel(Npp8u *pDst, int nDstStep, int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  Npp8u *pixel = pDst + static_cast<size_t>(y) * nDstStep + 3 * x;
  const Npp8u tmp = pixel[0];
  pixel[0] = pixel[2];
  pixel[2] = tmp;
}

// The packed YUV/YCbCr -> BGR conversions have no packed single-image BGR
// entry point, so run the RGB conversion and exchange the R and B bytes.
NppStatus swapRedBlue(Npp8u *pDst, int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  const dim3 block(16, 16);
  const dim3 grid((oSizeROI.width + 15) / 16, (oSizeROI.height + 15) / 16);
  swapRedBlueKernel<<<grid, block, 0, stream>>>(pDst, nDstStep, oSizeROI.width, oSizeROI.height);
  return (cudaGetLastError() == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus convertPlanar(Conversion conv, const NppiImageDescriptor *const pSrc[3], const NppiImageDescriptor &dst,
                        NppiSize oSizeROI, cudaStream_t stream) {
  const Npp8u *pPlane[3] = {static_cast<const Npp8u *>(pSrc[0]->pData),
                            static_cast<const Npp8u *>(pSrc[1]->pData),
                            static_cast<const Npp8u *>(pSrc[2]->pData)};
  int aSrcStep[3] = {pSrc[0]->nStep, pSrc[1]->nStep, pSrc[2]->nStep};
  Npp8u *pDst = static_cast<Npp8u *>(dst.pData);
  NppStreamContext ctx{};
  ctx.hStream = stream;

  switch (conv) {
  case kYCbCr420ToBGR:
    return nppiYCbCr420ToBGR_8u_P3C3R_Ctx(pPlane, aSrcStep, pDst, dst.nStep, oSizeROI, ctx);
  case kYCbCr420ToRGB:
    return nppiYCbCr420ToRGB_8u_P3C3R_Ctx(pPlane, aSrcStep, pDst, dst.nStep, oSizeROI, ctx);
  case kYUV420ToBGR:
    return nppiYUV420ToBGR_8u_P3C3R_Ctx(pPlane, aSrcStep, pDst, dst.nStep, oSizeROI, ctx);
  case kYUV420ToRGB:
    return nppiYUV420ToRGB_8u_P3C3R_Ctx(pPlane, aSrcStep, pDst, dst.nStep, oSizeROI, ctx);
  case kYCbCr422ToBGR:
    return nppiYCbCr422ToBGR_8u_P3C3R_Ctx(pPlane, aSrcStep, pDst, dst.nStep, oSizeROI, ctx);
  case kYCbCr422ToRGB:
    return nppiYCbCr422ToRGB_8u_P3C3R_Ctx(pPlane, aSrcStep, pDst, dst.nStep, oSizeROI, ctx);
  case kYUV422ToBGR: {
    NppStatus status = nppiYUV422ToRGB_8u_P3C3R_Ctx(pPlane, aSrcStep, pDst, dst.nStep, oSizeROI, ctx);
    if (status != NPP_NO_ERROR) {
      return status;
    }
    return swapRedBlue(pDst, dst.nStep, oSizeROI, stream);
  }
  case kYUV422ToRGB:
    return nppiYUV422ToRGB_8u_P3C3R_Ctx(pPlane, aSrcStep, pDst, dst.nStep, oSizeROI, ctx);
  case kYCbCrToBGRP3:
    return kernelStatus(nppiYCbCrToRGB_8u_P3C3R_Step3_kernel(pPlane[0], aSrcStep[0], pPlane[1], aSrcStep[1],
                                                             pPlane[2], aSrcStep[2], pDst, dst.nStep, oSizeROI, 1,
                                                             stream));
  case kYCbCrToRGBP3:
    return kernelStatus(nppiYCbCrToRGB_8u_P3C3R_Step3_kernel(pPlane[0], aSrcStep[0], pPlane[1], aSrcStep[1],
                                                             pPlane[2], aSrcStep[2], pDst, dst.nStep, oSizeROI, 0,
                                                             stream));
  case kYUVToBGRP3:
    return kernelStatus(nppiYUVToRGB_8u_P3C3R_Step3_kernel(pPlane[0], aSrcStep[0], pPlane[1], aSrcStep[1],
                                                           pPlane[2], aSrcStep[2], pDst, dst.nStep, oSizeROI, 1,
                                                           stream));
  case kYUVToRGBP3:
    return kernelStatus(nppiYUVToRGB_8u_P3C3R_Step3_kernel(pPlane[0], aSrcStep[0], pPlane[1], aSrcStep[1],
                                                           pPlane[2], aSrcStep[2], pDst, dst.nStep, oSizeROI, 0,
                                                           stream));
  default:
    return NPP_BAD_ARGUMENT_ERROR;
  }
}

NppStatus convertPacked(Conversion conv, const NppiImageDescriptor &src, const NppiImageDescriptor &dst,
                        NppiSize oSizeROI, cudaStream_t stream) {
  const Npp8u *pSrc = static_cast<const Npp8u *>(src.pData);
  Npp8u *pDst = static_cast<Npp8u *>(dst.pData);
  NppStreamContext ctx{};
  ctx.hStream = stream;

  switch (conv) {
  case kYCbCrToRGBC3:
    return nppiYCbCrToRGB_8u_C3R_Ctx(pSrc, src.nStep, pDst, dst.nStep, oSizeROI, ctx);
  case kYCbCrToBGRC3: {
    NppStatus status = nppiYCbCrToRGB_8u_C3R_Ctx(pSrc, src.nStep, pDst, dst.nStep, oSizeROI, ctx);
    if (status != NPP_NO_ERROR) {
      return status;
    }
    return swapRedBlue(pDst, dst.nStep, oSizeROI, stream);
  }
  case kYUVToRGBC3:
    return nppiYUVToRGB_8u_C3R_Ctx(pSrc, src.nStep, pDst, dst.nStep, oSizeROI, ctx);
  case kYUVToBGRC3:
    return nppiYUVToBGR_8u_C3R_Ctx(pSrc, src.nStep, pDst, dst.nStep, oSizeROI, ctx);
  default:
    return NPP_BAD_ARGUMENT_ERROR;
  }
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

NppStatus copyDescriptors(const NppiImageDescriptor *src, NppiImageDescriptor *dst, int nBatchSize,
                          cudaStream_t stream, const NppiImageDescriptor *const aPlane[3],
                          std::vector<NppiImageDescriptor> &hSrc0, std::vector<NppiImageDescriptor> &hSrc1,
                          std::vector<NppiImageDescriptor> &hSrc2, std::vector<NppiImageDescriptor> &hDst) {
  const size_t bytes = static_cast<size_t>(nBatchSize) * sizeof(NppiImageDescriptor);
  if (aPlane) {
    if (cudaMemcpyAsync(hSrc0.data(), aPlane[0], bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaMemcpyAsync(hSrc1.data(), aPlane[1], bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaMemcpyAsync(hSrc2.data(), aPlane[2], bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
      return NPP_MEMCPY_ERROR;
    }
  } else if (cudaMemcpyAsync(hSrc0.data(), src, bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
    return NPP_MEMCPY_ERROR;
  }
  if (cudaMemcpyAsync(hDst.data(), dst, bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
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
  const NppiImageDescriptor *const aPlane[3] = {pSrc[0], pSrc[1], pSrc[2]};
  NppStatus status = copyDescriptors(pSrc[0], pDst, nb, stream, aPlane, hSrc0, hSrc1, hSrc2, hDst);
  if (status != NPP_NO_ERROR) {
    return status;
  }

  bool warned = false;
  for (int i = 0; i < nb; ++i) {
    const NppiImageDescriptor *srcPlanes[3] = {&hSrc0[i], &hSrc1[i], &hSrc2[i]};
    for (int p = 0; p < 3; ++p) {
      status = validateDescriptor(*srcPlanes[p]);
      if (status != NPP_NO_ERROR) {
        return status;
      }
    }
    status = validateDescriptor(hDst[i]);
    if (status != NPP_NO_ERROR) {
      return status;
    }
    NppiSize roi{};
    status = effectiveRoi(conv, advanced ? hDst[i].oSize : oSizeROI, roi, warned);
    if (status != NPP_NO_ERROR) {
      return status;
    }
    status = convertPlanar(conv, srcPlanes, hDst[i], roi, stream);
    if (status == NPP_DOUBLE_SIZE_WARNING) {
      warned = true;
    } else if (status != NPP_NO_ERROR) {
      return status;
    }
  }
  return warned ? NPP_DOUBLE_SIZE_WARNING : NPP_NO_ERROR;
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
  std::vector<NppiImageDescriptor> unused;
  NppStatus status = copyDescriptors(pSrc, pDst, nb, stream, nullptr, hSrc, unused, unused, hDst);
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
    status = effectiveRoi(conv, advanced ? hDst[i].oSize : oSizeROI, roi, warned);
    if (status != NPP_NO_ERROR) {
      return status;
    }
    status = convertPacked(conv, hSrc[i], hDst[i], roi, stream);
    if (status == NPP_DOUBLE_SIZE_WARNING) {
      warned = true;
    } else if (status != NPP_NO_ERROR) {
      return status;
    }
  }
  return warned ? NPP_DOUBLE_SIZE_WARNING : NPP_NO_ERROR;
}

} // namespace c3batch
} // namespace nppi
