// Generated API wrappers -- do not edit by hand.
// family: nppiGrayErodeBorder


#include "npp.h"
#include "nppi_morph_executor.h"

NppStatus nppiGrayErodeBorder_32f_C1R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32f * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::morph::runGray<Npp32f, Npp32f>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, eBorderType, nppi::morph::kErode, nppStreamCtx.hStream);
}

NppStatus nppiGrayErodeBorder_8u_C1R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32s * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::morph::runGray<Npp8u, Npp32s>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, eBorderType, nppi::morph::kErode, nppStreamCtx.hStream);
}

NppStatus nppiGrayErodeBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32f * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiGrayErodeBorder_32f_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, eBorderType, nppStreamCtx);
}

NppStatus nppiGrayErodeBorder_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32s * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiGrayErodeBorder_8u_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, eBorderType, nppStreamCtx);
}
