// Generated API wrappers -- do not edit by hand.
// family: nppiFilterSobelHorizMaskBorder


#include "npp.h"
#include "nppi_filter_derivatives_executor.h"

NppStatus nppiFilterSobelHorizMaskBorder_32f_C1R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterDerivMaskBorder<Npp32f, Npp32f, nppi::filter::kSobelHoriz>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eMaskSize, eBorderType, nppStreamCtx.hStream);
}

NppStatus nppiFilterSobelHorizMaskBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterSobelHorizMaskBorder_32f_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eMaskSize, eBorderType, nppStreamCtx);
}
