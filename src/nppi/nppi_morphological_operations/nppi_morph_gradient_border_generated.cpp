// Generated API wrappers -- do not edit by hand.
// family: nppiMorphGradientBorder


#include "npp.h"
#include "nppi_morph_executor.h"

NppStatus nppiMorphGradientBorder_16s_C1R_Ctx(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::morph::runComposite<Npp16s, 1, 0>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, pBuffer, eBorderType, 2, nppStreamCtx.hStream);
}

NppStatus nppiMorphGradientBorder_16u_C1R_Ctx(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::morph::runComposite<Npp16u, 1, 0>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, pBuffer, eBorderType, 2, nppStreamCtx.hStream);
}

NppStatus nppiMorphGradientBorder_32f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::morph::runComposite<Npp32f, 1, 0>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, pBuffer, eBorderType, 2, nppStreamCtx.hStream);
}

NppStatus nppiMorphGradientBorder_32f_C3R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::morph::runComposite<Npp32f, 3, 0>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, pBuffer, eBorderType, 2, nppStreamCtx.hStream);
}

NppStatus nppiMorphGradientBorder_32f_C4R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::morph::runComposite<Npp32f, 4, 0>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, pBuffer, eBorderType, 2, nppStreamCtx.hStream);
}

NppStatus nppiMorphGradientBorder_8u_C1R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::morph::runComposite<Npp8u, 1, 0>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, pBuffer, eBorderType, 2, nppStreamCtx.hStream);
}

NppStatus nppiMorphGradientBorder_8u_C3R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::morph::runComposite<Npp8u, 3, 0>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, pBuffer, eBorderType, 2, nppStreamCtx.hStream);
}

NppStatus nppiMorphGradientBorder_8u_C4R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::morph::runComposite<Npp8u, 4, 0>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, pBuffer, eBorderType, 2, nppStreamCtx.hStream);
}

NppStatus nppiMorphGradientBorder_16s_C1R(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiMorphGradientBorder_16s_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, pBuffer, eBorderType, nppStreamCtx);
}

NppStatus nppiMorphGradientBorder_16u_C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiMorphGradientBorder_16u_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, pBuffer, eBorderType, nppStreamCtx);
}

NppStatus nppiMorphGradientBorder_32f_C1R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiMorphGradientBorder_32f_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, pBuffer, eBorderType, nppStreamCtx);
}

NppStatus nppiMorphGradientBorder_32f_C3R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiMorphGradientBorder_32f_C3R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, pBuffer, eBorderType, nppStreamCtx);
}

NppStatus nppiMorphGradientBorder_32f_C4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiMorphGradientBorder_32f_C4R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, pBuffer, eBorderType, nppStreamCtx);
}

NppStatus nppiMorphGradientBorder_8u_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiMorphGradientBorder_8u_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, pBuffer, eBorderType, nppStreamCtx);
}

NppStatus nppiMorphGradientBorder_8u_C3R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiMorphGradientBorder_8u_C3R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, pBuffer, eBorderType, nppStreamCtx);
}

NppStatus nppiMorphGradientBorder_8u_C4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiMorphGradientBorder_8u_C4R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, pBuffer, eBorderType, nppStreamCtx);
}
