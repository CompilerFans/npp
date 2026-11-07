// Generated API wrappers -- do not edit by hand.
// family: nppiCbYCr422ToBGR_709HDTV


#include "npp.h"
#include "nppi_c3_color_executor.h"

NppStatus nppiCbYCr422ToBGR_709HDTV_8u_C2C3R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::c3color::invC2(pSrc, nSrcStep, pDst, nDstStep, 3, 0, oSizeROI, nppStreamCtx.hStream);
}

NppStatus nppiCbYCr422ToBGR_709HDTV_8u_C2C4R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, Npp8u nAval, NppStreamContext nppStreamCtx) {
  return nppi::c3color::invC2(pSrc, nSrcStep, pDst, nDstStep, 4, nAval, oSizeROI, nppStreamCtx.hStream);
}

NppStatus nppiCbYCr422ToBGR_709HDTV_8u_C2C3R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiCbYCr422ToBGR_709HDTV_8u_C2C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCbYCr422ToBGR_709HDTV_8u_C2C4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, Npp8u nAval) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiCbYCr422ToBGR_709HDTV_8u_C2C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nAval, nppStreamCtx);
}
