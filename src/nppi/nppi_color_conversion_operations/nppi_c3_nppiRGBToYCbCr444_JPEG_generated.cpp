// Generated API wrappers -- do not edit by hand.
// family: nppiRGBToYCbCr444_JPEG


#include "npp.h"
#include "nppi_c3_conversion_executor.h"

NppStatus nppiRGBToYCbCr444_JPEG_8u_C3P3R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst[3], int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::c3conv::rgbToYCbCr444JpegC3P3(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
}

NppStatus nppiRGBToYCbCr444_JPEG_8u_P3R_Ctx(const Npp8u * const pSrc[3], int nSrcStep, Npp8u * pDst[3], int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::c3conv::rgbToYCbCr444JpegP3(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
}

NppStatus nppiRGBToYCbCr444_JPEG_8u_C3P3R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst[3], int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiRGBToYCbCr444_JPEG_8u_C3P3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiRGBToYCbCr444_JPEG_8u_P3R(const Npp8u * const pSrc[3], int nSrcStep, Npp8u * pDst[3], int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiRGBToYCbCr444_JPEG_8u_P3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}
