// Generated API wrappers -- do not edit by hand.
// family: nppiFilterThresholdAdaptiveBoxBorder


#include "npp.h"
#include "nppi_filter_tail_executor.h"

NppStatus nppiFilterThresholdAdaptiveBoxBorder_8u_C1R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, Npp32f nDelta, Npp8u nValGT, Npp8u nValLE, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::filter::thresholdAdaptiveBoxBorder(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, oMaskSize, nDelta, nValGT, nValLE, eBorderType, nppStreamCtx.hStream);
}

NppStatus nppiFilterThresholdAdaptiveBoxBorder_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, Npp32f nDelta, Npp8u nValGT, Npp8u nValLE, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterThresholdAdaptiveBoxBorder_8u_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, oMaskSize, nDelta, nValGT, nValLE, eBorderType, nppStreamCtx);
}
