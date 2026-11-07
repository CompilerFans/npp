// Generated API wrappers -- do not edit by hand.
// family: nppiLUVToRGB


#include "npp.h"
#include "nppi_color_space_executor.h"

NppStatus nppiLUVToRGB_8u_AC4R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::color_space::luvToRgb(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 4, nppStreamCtx.hStream);
}

NppStatus nppiLUVToRGB_8u_C3R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::color_space::luvToRgb(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 3, nppStreamCtx.hStream);
}

NppStatus nppiLUVToRGB_8u_AC4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUVToRGB_8u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiLUVToRGB_8u_C3R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUVToRGB_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}
