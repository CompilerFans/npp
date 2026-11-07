// Generated API wrappers -- do not edit by hand.
// family: nppiFilterHarrisCornersBorder


#include "npp.h"
#include "nppi_filter_tail_executor.h"

NppStatus nppiFilterHarrisCornersBorder_8u32f_C1R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppiDifferentialKernel eFilterType, NppiMaskSize eMaskSize, NppiMaskSize eAvgWindowSize, Npp32f nK, Npp32f nScale, NppiBorderType eBorderType, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppi::filter::harrisCornersBorder(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eFilterType, eMaskSize, eAvgWindowSize, nK, nScale, eBorderType, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppiFilterHarrisCornersBorder_8u32f_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppiDifferentialKernel eFilterType, NppiMaskSize eMaskSize, NppiMaskSize eAvgWindowSize, Npp32f nK, Npp32f nScale, NppiBorderType eBorderType, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterHarrisCornersBorder_8u32f_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eFilterType, eMaskSize, eAvgWindowSize, nK, nScale, eBorderType, pDeviceBuffer, nppStreamCtx);
}
