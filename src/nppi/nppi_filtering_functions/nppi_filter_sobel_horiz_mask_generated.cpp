// Generated API wrappers -- do not edit by hand.
// family: nppiFilterSobelHorizMask


#include "npp.h"
#include "nppi_filter_derivatives_executor.h"

NppStatus nppiFilterSobelHorizMask_32f_C1R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterDerivMask<Npp32f, Npp32f, nppi::filter::kSobelHoriz>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx.hStream);
}

NppStatus nppiFilterSobelHorizMask_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterSobelHorizMask_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx);
}
