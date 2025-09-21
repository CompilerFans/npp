#include "npp.h"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

// Forward declarations for mpp host func implementations
extern "C" {
NppStatus nppiFilterBoxBorder_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                              NppiPoint oSrcOffset, Npp8u *pDst, int nDstStep, NppiSize oDstSizeROI,
                                              NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType,
                                              NppStreamContext nppStreamCtx);
NppStatus nppiFilterBoxBorder_8u_C3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                              NppiPoint oSrcOffset, Npp8u *pDst, int nDstStep, NppiSize oDstSizeROI,
                                              NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType,
                                              NppStreamContext nppStreamCtx);
NppStatus nppiFilterBoxBorder_16s_C1R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                               NppiPoint oSrcOffset, Npp16s *pDst, int nDstStep, NppiSize oDstSizeROI,
                                               NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType,
                                               NppStreamContext nppStreamCtx);
NppStatus nppiFilterBoxBorder_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                               NppiPoint oSrcOffset, Npp32f *pDst, int nDstStep, NppiSize oDstSizeROI,
                                               NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType,
                                               NppStreamContext nppStreamCtx);
}

// Input validation helper
static inline NppStatus validateFilterBoxBorderInputs(const void *pSrc, int nSrcStep, NppiSize oSrcSizeROI, void *pDst,
                                                      int nDstStep, NppiSize oDstSizeROI, NppiSize oMaskSize,
                                                      NppiPoint oAnchor, NppiBorderType eBorderType) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSrcSizeROI.width <= 0 || oSrcSizeROI.height <= 0 || oDstSizeROI.width <= 0 || oDstSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (oMaskSize.width <= 0 || oMaskSize.height <= 0) {
    return NPP_MASK_SIZE_ERROR;
  }

  // Validate anchor point
  if (oAnchor.x < 0 || oAnchor.x >= oMaskSize.width || oAnchor.y < 0 || oAnchor.y >= oMaskSize.height) {
    return NPP_ANCHOR_ERROR;
  }

  // Validate border type
  if (eBorderType != NPP_BORDER_REPLICATE && eBorderType != NPP_BORDER_WRAP && eBorderType != NPP_BORDER_MIRROR &&
      eBorderType != NPP_BORDER_CONSTANT) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  return NPP_SUCCESS;
}

// 8-bit unsigned single channel box filter with border
NppStatus nppiFilterBoxBorder_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSizeROI, NppiPoint oSrcOffset,
                                         Npp8u *pDst, int nDstStep, NppiSize oDstSizeROI, NppiSize oMaskSize,
                                         NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  NppStatus status = validateFilterBoxBorderInputs(pSrc, nSrcStep, oSrcSizeROI, pDst, nDstStep, oDstSizeROI, oMaskSize,
                                                   oAnchor, eBorderType);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiFilterBoxBorder_8u_C1R_Ctx_impl(pSrc, nSrcStep, oSrcSizeROI, oSrcOffset, pDst, nDstStep, oDstSizeROI,
                                             oMaskSize, oAnchor, eBorderType, nppStreamCtx);
}

NppStatus nppiFilterBoxBorder_8u_C1R(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSizeROI, NppiPoint oSrcOffset,
                                     Npp8u *pDst, int nDstStep, NppiSize oDstSizeROI, NppiSize oMaskSize,
                                     NppiPoint oAnchor, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiFilterBoxBorder_8u_C1R_Ctx(pSrc, nSrcStep, oSrcSizeROI, oSrcOffset, pDst, nDstStep, oDstSizeROI, oMaskSize,
                                        oAnchor, eBorderType, nppStreamCtx);
}

// 8-bit unsigned three channel box filter with border
NppStatus nppiFilterBoxBorder_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSizeROI, NppiPoint oSrcOffset,
                                         Npp8u *pDst, int nDstStep, NppiSize oDstSizeROI, NppiSize oMaskSize,
                                         NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  NppStatus status = validateFilterBoxBorderInputs(pSrc, nSrcStep, oSrcSizeROI, pDst, nDstStep, oDstSizeROI, oMaskSize,
                                                   oAnchor, eBorderType);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiFilterBoxBorder_8u_C3R_Ctx_impl(pSrc, nSrcStep, oSrcSizeROI, oSrcOffset, pDst, nDstStep, oDstSizeROI,
                                             oMaskSize, oAnchor, eBorderType, nppStreamCtx);
}

NppStatus nppiFilterBoxBorder_8u_C3R(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSizeROI, NppiPoint oSrcOffset,
                                     Npp8u *pDst, int nDstStep, NppiSize oDstSizeROI, NppiSize oMaskSize,
                                     NppiPoint oAnchor, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiFilterBoxBorder_8u_C3R_Ctx(pSrc, nSrcStep, oSrcSizeROI, oSrcOffset, pDst, nDstStep, oDstSizeROI, oMaskSize,
                                        oAnchor, eBorderType, nppStreamCtx);
}

// 16-bit signed single channel box filter with border
NppStatus nppiFilterBoxBorder_16s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep, NppiSize oSrcSizeROI, NppiPoint oSrcOffset,
                                          Npp16s *pDst, int nDstStep, NppiSize oDstSizeROI, NppiSize oMaskSize,
                                          NppiPoint oAnchor, NppiBorderType eBorderType,
                                          NppStreamContext nppStreamCtx) {
  NppStatus status = validateFilterBoxBorderInputs(pSrc, nSrcStep, oSrcSizeROI, pDst, nDstStep, oDstSizeROI, oMaskSize,
                                                   oAnchor, eBorderType);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiFilterBoxBorder_16s_C1R_Ctx_impl(pSrc, nSrcStep, oSrcSizeROI, oSrcOffset, pDst, nDstStep, oDstSizeROI,
                                              oMaskSize, oAnchor, eBorderType, nppStreamCtx);
}

NppStatus nppiFilterBoxBorder_16s_C1R(const Npp16s *pSrc, int nSrcStep, NppiSize oSrcSizeROI, NppiPoint oSrcOffset,
                                      Npp16s *pDst, int nDstStep, NppiSize oDstSizeROI, NppiSize oMaskSize,
                                      NppiPoint oAnchor, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiFilterBoxBorder_16s_C1R_Ctx(pSrc, nSrcStep, oSrcSizeROI, oSrcOffset, pDst, nDstStep, oDstSizeROI,
                                         oMaskSize, oAnchor, eBorderType, nppStreamCtx);
}

// 32-bit float single channel box filter with border
NppStatus nppiFilterBoxBorder_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSizeROI, NppiPoint oSrcOffset,
                                          Npp32f *pDst, int nDstStep, NppiSize oDstSizeROI, NppiSize oMaskSize,
                                          NppiPoint oAnchor, NppiBorderType eBorderType,
                                          NppStreamContext nppStreamCtx) {
  NppStatus status = validateFilterBoxBorderInputs(pSrc, nSrcStep, oSrcSizeROI, pDst, nDstStep, oDstSizeROI, oMaskSize,
                                                   oAnchor, eBorderType);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiFilterBoxBorder_32f_C1R_Ctx_impl(pSrc, nSrcStep, oSrcSizeROI, oSrcOffset, pDst, nDstStep, oDstSizeROI,
                                              oMaskSize, oAnchor, eBorderType, nppStreamCtx);
}

NppStatus nppiFilterBoxBorder_32f_C1R(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSizeROI, NppiPoint oSrcOffset,
                                      Npp32f *pDst, int nDstStep, NppiSize oDstSizeROI, NppiSize oMaskSize,
                                      NppiPoint oAnchor, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiFilterBoxBorder_32f_C1R_Ctx(pSrc, nSrcStep, oSrcSizeROI, oSrcOffset, pDst, nDstStep, oDstSizeROI,
                                         oMaskSize, oAnchor, eBorderType, nppStreamCtx);
}
