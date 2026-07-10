#include "npp.h"
#include <cuda_runtime.h>

extern "C" {
cudaError_t nppiFilterBox_8u_CxR_kernel(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                        NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, int nChannels,
                                        int nFilteredChannels, cudaStream_t stream);
cudaError_t nppiFilterBox_16u_CxR_kernel(const Npp16u *pSrc, Npp32s nSrcStep, Npp16u *pDst, Npp32s nDstStep,
                                         NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, int nChannels,
                                         int nFilteredChannels, cudaStream_t stream);
cudaError_t nppiFilterBox_16s_CxR_kernel(const Npp16s *pSrc, Npp32s nSrcStep, Npp16s *pDst, Npp32s nDstStep,
                                         NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, int nChannels,
                                         int nFilteredChannels, cudaStream_t stream);
cudaError_t nppiFilterBox_32f_CxR_kernel(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                         NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, int nChannels,
                                         int nFilteredChannels, cudaStream_t stream);
cudaError_t nppiFilterBox_64f_CxR_kernel(const Npp64f *pSrc, Npp32s nSrcStep, Npp64f *pDst, Npp32s nDstStep,
                                         NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, int nChannels,
                                         int nFilteredChannels, cudaStream_t stream);
}

namespace {

template <typename T>
using FilterBoxKernel = cudaError_t (*)(const T *, Npp32s, T *, Npp32s, NppiSize, NppiSize, NppiPoint, int, int,
                                        cudaStream_t);

template <typename T>
NppStatus filterBoxGeneric(const T *pSrc, Npp32s nSrcStep, T *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                           NppiSize oMaskSize, NppiPoint oAnchor, int channels, int filteredChannels,
                           NppStreamContext nppStreamCtx, FilterBoxKernel<T> kernel) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (oMaskSize.width <= 0 || oMaskSize.height <= 0) {
    return NPP_MASK_SIZE_ERROR;
  }
  const long long rowBytes = static_cast<long long>(oSizeROI.width) * channels * sizeof(T);
  if (nSrcStep < rowBytes || nDstStep < rowBytes) {
    return NPP_STEP_ERROR;
  }
  if (oAnchor.x < 0 || oAnchor.x >= oMaskSize.width || oAnchor.y < 0 || oAnchor.y >= oMaskSize.height) {
    return NPP_ANCHOR_ERROR;
  }
  const cudaError_t status = kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, channels,
                                    filteredChannels, nppStreamCtx.hStream);
  return status == cudaSuccess ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStreamContext currentStreamContext() {
  NppStreamContext context{};
  nppGetStreamContext(&context);
  return context;
}

} // namespace

// FilterBox has no source-image size or border mode. The source pointer must have
// the halo required by the mask and anchor; FilterBoxBorder handles image edges.

NppStatus nppiFilterBox_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                   NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return filterBoxGeneric(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, 1, 1, nppStreamCtx,
                          nppiFilterBox_8u_CxR_kernel);
}

NppStatus nppiFilterBox_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                               NppiSize oMaskSize, NppiPoint oAnchor) {
  return nppiFilterBox_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor,
                                  currentStreamContext());
}

NppStatus nppiFilterBox_8u_C3R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                   NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor,
                                   NppStreamContext nppStreamCtx) {
  return filterBoxGeneric(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, 3, 3, nppStreamCtx,
                          nppiFilterBox_8u_CxR_kernel);
}

NppStatus nppiFilterBox_8u_C3R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                               NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor) {
  return nppiFilterBox_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor,
                                  currentStreamContext());
}

// nppiFilterBox_8u_C4R implementation
NppStatus nppiFilterBox_8u_C4R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                   NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return filterBoxGeneric(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, 4, 4, nppStreamCtx,
                          nppiFilterBox_8u_CxR_kernel);
}

NppStatus nppiFilterBox_8u_C4R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                               NppiSize oMaskSize, NppiPoint oAnchor) {
  return nppiFilterBox_8u_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor,
                                  currentStreamContext());
}

// nppiFilterBox_32f_C1R implementation
NppStatus nppiFilterBox_32f_C1R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                    NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor,
                                    NppStreamContext nppStreamCtx) {
  return filterBoxGeneric(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, 1, 1, nppStreamCtx,
                          nppiFilterBox_32f_CxR_kernel);
}

NppStatus nppiFilterBox_32f_C1R(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                NppiSize oMaskSize, NppiPoint oAnchor) {
  return nppiFilterBox_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor,
                                   currentStreamContext());
}

NppStatus nppiFilterBox_8u_AC4R_Ctx(const Npp8u *src, Npp32s srcStep, Npp8u *dst, Npp32s dstStep, NppiSize roi,
                                    NppiSize mask, NppiPoint anchor, NppStreamContext context) {
  return filterBoxGeneric(src, srcStep, dst, dstStep, roi, mask, anchor, 4, 3, context,
                          nppiFilterBox_8u_CxR_kernel);
}
NppStatus nppiFilterBox_8u_AC4R(const Npp8u *src, Npp32s srcStep, Npp8u *dst, Npp32s dstStep, NppiSize roi,
                                NppiSize mask, NppiPoint anchor) {
  return nppiFilterBox_8u_AC4R_Ctx(src, srcStep, dst, dstStep, roi, mask, anchor, currentStreamContext());
}

NppStatus nppiFilterBox_16u_C1R_Ctx(const Npp16u *src, Npp32s srcStep, Npp16u *dst, Npp32s dstStep, NppiSize roi,
                                    NppiSize mask, NppiPoint anchor, NppStreamContext context) {
  return filterBoxGeneric(src, srcStep, dst, dstStep, roi, mask, anchor, 1, 1, context,
                          nppiFilterBox_16u_CxR_kernel);
}
NppStatus nppiFilterBox_16u_C1R(const Npp16u *src, Npp32s srcStep, Npp16u *dst, Npp32s dstStep, NppiSize roi,
                                NppiSize mask, NppiPoint anchor) {
  return nppiFilterBox_16u_C1R_Ctx(src, srcStep, dst, dstStep, roi, mask, anchor, currentStreamContext());
}
NppStatus nppiFilterBox_16u_C3R_Ctx(const Npp16u *src, Npp32s srcStep, Npp16u *dst, Npp32s dstStep, NppiSize roi,
                                    NppiSize mask, NppiPoint anchor, NppStreamContext context) {
  return filterBoxGeneric(src, srcStep, dst, dstStep, roi, mask, anchor, 3, 3, context,
                          nppiFilterBox_16u_CxR_kernel);
}
NppStatus nppiFilterBox_16u_C3R(const Npp16u *src, Npp32s srcStep, Npp16u *dst, Npp32s dstStep, NppiSize roi,
                                NppiSize mask, NppiPoint anchor) {
  return nppiFilterBox_16u_C3R_Ctx(src, srcStep, dst, dstStep, roi, mask, anchor, currentStreamContext());
}
NppStatus nppiFilterBox_16u_C4R_Ctx(const Npp16u *src, Npp32s srcStep, Npp16u *dst, Npp32s dstStep, NppiSize roi,
                                    NppiSize mask, NppiPoint anchor, NppStreamContext context) {
  return filterBoxGeneric(src, srcStep, dst, dstStep, roi, mask, anchor, 4, 4, context,
                          nppiFilterBox_16u_CxR_kernel);
}
NppStatus nppiFilterBox_16u_C4R(const Npp16u *src, Npp32s srcStep, Npp16u *dst, Npp32s dstStep, NppiSize roi,
                                NppiSize mask, NppiPoint anchor) {
  return nppiFilterBox_16u_C4R_Ctx(src, srcStep, dst, dstStep, roi, mask, anchor, currentStreamContext());
}
NppStatus nppiFilterBox_16u_AC4R_Ctx(const Npp16u *src, Npp32s srcStep, Npp16u *dst, Npp32s dstStep, NppiSize roi,
                                     NppiSize mask, NppiPoint anchor, NppStreamContext context) {
  return filterBoxGeneric(src, srcStep, dst, dstStep, roi, mask, anchor, 4, 3, context,
                          nppiFilterBox_16u_CxR_kernel);
}
NppStatus nppiFilterBox_16u_AC4R(const Npp16u *src, Npp32s srcStep, Npp16u *dst, Npp32s dstStep, NppiSize roi,
                                 NppiSize mask, NppiPoint anchor) {
  return nppiFilterBox_16u_AC4R_Ctx(src, srcStep, dst, dstStep, roi, mask, anchor, currentStreamContext());
}

NppStatus nppiFilterBox_16s_C1R_Ctx(const Npp16s *src, Npp32s srcStep, Npp16s *dst, Npp32s dstStep, NppiSize roi,
                                    NppiSize mask, NppiPoint anchor, NppStreamContext context) {
  return filterBoxGeneric(src, srcStep, dst, dstStep, roi, mask, anchor, 1, 1, context,
                          nppiFilterBox_16s_CxR_kernel);
}
NppStatus nppiFilterBox_16s_C1R(const Npp16s *src, Npp32s srcStep, Npp16s *dst, Npp32s dstStep, NppiSize roi,
                                NppiSize mask, NppiPoint anchor) {
  return nppiFilterBox_16s_C1R_Ctx(src, srcStep, dst, dstStep, roi, mask, anchor, currentStreamContext());
}
NppStatus nppiFilterBox_16s_C3R_Ctx(const Npp16s *src, Npp32s srcStep, Npp16s *dst, Npp32s dstStep, NppiSize roi,
                                    NppiSize mask, NppiPoint anchor, NppStreamContext context) {
  return filterBoxGeneric(src, srcStep, dst, dstStep, roi, mask, anchor, 3, 3, context,
                          nppiFilterBox_16s_CxR_kernel);
}
NppStatus nppiFilterBox_16s_C3R(const Npp16s *src, Npp32s srcStep, Npp16s *dst, Npp32s dstStep, NppiSize roi,
                                NppiSize mask, NppiPoint anchor) {
  return nppiFilterBox_16s_C3R_Ctx(src, srcStep, dst, dstStep, roi, mask, anchor, currentStreamContext());
}
NppStatus nppiFilterBox_16s_C4R_Ctx(const Npp16s *src, Npp32s srcStep, Npp16s *dst, Npp32s dstStep, NppiSize roi,
                                    NppiSize mask, NppiPoint anchor, NppStreamContext context) {
  return filterBoxGeneric(src, srcStep, dst, dstStep, roi, mask, anchor, 4, 4, context,
                          nppiFilterBox_16s_CxR_kernel);
}
NppStatus nppiFilterBox_16s_C4R(const Npp16s *src, Npp32s srcStep, Npp16s *dst, Npp32s dstStep, NppiSize roi,
                                NppiSize mask, NppiPoint anchor) {
  return nppiFilterBox_16s_C4R_Ctx(src, srcStep, dst, dstStep, roi, mask, anchor, currentStreamContext());
}
NppStatus nppiFilterBox_16s_AC4R_Ctx(const Npp16s *src, Npp32s srcStep, Npp16s *dst, Npp32s dstStep, NppiSize roi,
                                     NppiSize mask, NppiPoint anchor, NppStreamContext context) {
  return filterBoxGeneric(src, srcStep, dst, dstStep, roi, mask, anchor, 4, 3, context,
                          nppiFilterBox_16s_CxR_kernel);
}
NppStatus nppiFilterBox_16s_AC4R(const Npp16s *src, Npp32s srcStep, Npp16s *dst, Npp32s dstStep, NppiSize roi,
                                 NppiSize mask, NppiPoint anchor) {
  return nppiFilterBox_16s_AC4R_Ctx(src, srcStep, dst, dstStep, roi, mask, anchor, currentStreamContext());
}

NppStatus nppiFilterBox_32f_C3R_Ctx(const Npp32f *src, Npp32s srcStep, Npp32f *dst, Npp32s dstStep, NppiSize roi,
                                    NppiSize mask, NppiPoint anchor, NppStreamContext context) {
  return filterBoxGeneric(src, srcStep, dst, dstStep, roi, mask, anchor, 3, 3, context,
                          nppiFilterBox_32f_CxR_kernel);
}
NppStatus nppiFilterBox_32f_C3R(const Npp32f *src, Npp32s srcStep, Npp32f *dst, Npp32s dstStep, NppiSize roi,
                                NppiSize mask, NppiPoint anchor) {
  return nppiFilterBox_32f_C3R_Ctx(src, srcStep, dst, dstStep, roi, mask, anchor, currentStreamContext());
}
NppStatus nppiFilterBox_32f_C4R_Ctx(const Npp32f *src, Npp32s srcStep, Npp32f *dst, Npp32s dstStep, NppiSize roi,
                                    NppiSize mask, NppiPoint anchor, NppStreamContext context) {
  return filterBoxGeneric(src, srcStep, dst, dstStep, roi, mask, anchor, 4, 4, context,
                          nppiFilterBox_32f_CxR_kernel);
}
NppStatus nppiFilterBox_32f_C4R(const Npp32f *src, Npp32s srcStep, Npp32f *dst, Npp32s dstStep, NppiSize roi,
                                NppiSize mask, NppiPoint anchor) {
  return nppiFilterBox_32f_C4R_Ctx(src, srcStep, dst, dstStep, roi, mask, anchor, currentStreamContext());
}
NppStatus nppiFilterBox_32f_AC4R_Ctx(const Npp32f *src, Npp32s srcStep, Npp32f *dst, Npp32s dstStep, NppiSize roi,
                                     NppiSize mask, NppiPoint anchor, NppStreamContext context) {
  return filterBoxGeneric(src, srcStep, dst, dstStep, roi, mask, anchor, 4, 3, context,
                          nppiFilterBox_32f_CxR_kernel);
}
NppStatus nppiFilterBox_32f_AC4R(const Npp32f *src, Npp32s srcStep, Npp32f *dst, Npp32s dstStep, NppiSize roi,
                                 NppiSize mask, NppiPoint anchor) {
  return nppiFilterBox_32f_AC4R_Ctx(src, srcStep, dst, dstStep, roi, mask, anchor, currentStreamContext());
}

NppStatus nppiFilterBox_64f_C1R_Ctx(const Npp64f *src, Npp32s srcStep, Npp64f *dst, Npp32s dstStep, NppiSize roi,
                                    NppiSize mask, NppiPoint anchor, NppStreamContext context) {
  return filterBoxGeneric(src, srcStep, dst, dstStep, roi, mask, anchor, 1, 1, context,
                          nppiFilterBox_64f_CxR_kernel);
}
NppStatus nppiFilterBox_64f_C1R(const Npp64f *src, Npp32s srcStep, Npp64f *dst, Npp32s dstStep, NppiSize roi,
                                NppiSize mask, NppiPoint anchor) {
  return nppiFilterBox_64f_C1R_Ctx(src, srcStep, dst, dstStep, roi, mask, anchor, currentStreamContext());
}
