// Generated API wrappers -- do not edit by hand.
// family: nppiResizeHost


#include "npp.h"
#include "nppi_resize_executor.h"

NppStatus nppiGetResizeRect(NppiRect oSrcROI, NppiRect *pDstRect, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation) {
  return nppi::resize::getResizeRect(oSrcROI, pDstRect, nXFactor, nYFactor, nXShift, nYShift, eInterpolation);
}

NppStatus nppiGetResizeTiledSourceOffset(NppiRect oSrcRectROI, NppiRect oDstRectROI, NppiPoint * pNewSrcRectOffset) {
  return nppi::resize::getResizeTiledSourceOffset(oSrcRectROI, oDstRectROI, pNewSrcRectOffset);
}

NppStatus nppiResizeAdvancedGetBufferHostSize_8u_C1R(NppiSize oSrcROI, NppiSize oDstROI, int * hpBufferSize, int eInterpolationMode) {
  return nppi::resize::resizeAdvancedGetBufferHostSize(oSrcROI, oDstROI, hpBufferSize, eInterpolationMode);
}
