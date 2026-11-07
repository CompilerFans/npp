// Generated API wrappers -- do not edit by hand.
// family: nppiYCbCr422ToRGB_JPEG


#include "npp.h"
#include "nppi_c3_conversion_executor.h"

NppStatus nppiYCbCr422ToRGB_JPEG_8u_P3C3R_Ctx(const Npp8u * const pSrc[3], int aSrcStep[3], Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::c3conv::yCbCr422ToRgbJpegP3C3(pSrc, aSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream, 0);
}

NppStatus nppiYCbCr422ToRGB_JPEG_8u_P3R_Ctx(const Npp8u * const pSrc[3], int aSrcStep[3], Npp8u * pDst[3], int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::c3conv::yCbCr422ToRgbJpegP3(pSrc, aSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream, 0);
}

NppStatus nppiYCbCr422ToRGB_JPEG_8u_P3C3R(const Npp8u * const pSrc[3], int aSrcStep[3], Npp8u * pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiYCbCr422ToRGB_JPEG_8u_P3C3R_Ctx(pSrc, aSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiYCbCr422ToRGB_JPEG_8u_P3R(const Npp8u * const pSrc[3], int aSrcStep[3], Npp8u * pDst[3], int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiYCbCr422ToRGB_JPEG_8u_P3R_Ctx(pSrc, aSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}
