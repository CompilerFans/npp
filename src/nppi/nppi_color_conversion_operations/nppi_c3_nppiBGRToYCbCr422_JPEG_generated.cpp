// Generated API wrappers -- do not edit by hand.
// family: nppiBGRToYCbCr422_JPEG


#include "npp.h"
#include "nppi_c3_conversion_executor.h"

NppStatus nppiBGRToYCbCr422_JPEG_8u_C3P3R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst[3], int aDstStep[3], NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::c3conv::rgbToYCbCr422JpegC3P3(pSrc, nSrcStep, pDst, aDstStep, oSizeROI, nppStreamCtx.hStream, 1);
}

NppStatus nppiBGRToYCbCr422_JPEG_8u_P3R_Ctx(const Npp8u * const pSrc[3], int nSrcStep, Npp8u * pDst[3], int aDstStep[3], NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::c3conv::rgbToYCbCr422JpegP3(pSrc, nSrcStep, pDst, aDstStep, oSizeROI, nppStreamCtx.hStream, 1);
}

NppStatus nppiBGRToYCbCr422_JPEG_8u_C3P3R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst[3], int aDstStep[3], NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiBGRToYCbCr422_JPEG_8u_C3P3R_Ctx(pSrc, nSrcStep, pDst, aDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiBGRToYCbCr422_JPEG_8u_P3R(const Npp8u * const pSrc[3], int nSrcStep, Npp8u * pDst[3], int aDstStep[3], NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiBGRToYCbCr422_JPEG_8u_P3R_Ctx(pSrc, nSrcStep, pDst, aDstStep, oSizeROI, nppStreamCtx);
}
