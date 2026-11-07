// Generated API wrappers -- do not edit by hand.
// family: nppiBGRToYCbCr422


#include "npp.h"
#include "nppi_c3_conversion_executor.h"

NppStatus nppiBGRToYCbCr422_8u_AC4C2R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::c3conv::rgbToYCbCr422C3C2(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream, 4, 1, 0);
}

NppStatus nppiBGRToYCbCr422_8u_AC4P3R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst[3], int rDstStep[3], NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::c3conv::rgbToYCbCr422C3P3(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, nppStreamCtx.hStream, 4, 1);
}

NppStatus nppiBGRToYCbCr422_8u_C3C2R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::c3conv::rgbToYCbCr422C3C2(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream, 3, 1, 0);
}

NppStatus nppiBGRToYCbCr422_8u_C3P3R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst[3], int rDstStep[3], NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::c3conv::rgbToYCbCr422C3P3(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, nppStreamCtx.hStream, 3, 1);
}

NppStatus nppiBGRToYCbCr422_8u_AC4C2R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiBGRToYCbCr422_8u_AC4C2R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiBGRToYCbCr422_8u_AC4P3R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst[3], int rDstStep[3], NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiBGRToYCbCr422_8u_AC4P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiBGRToYCbCr422_8u_C3C2R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiBGRToYCbCr422_8u_C3C2R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiBGRToYCbCr422_8u_C3P3R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst[3], int rDstStep[3], NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiBGRToYCbCr422_8u_C3P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, nppStreamCtx);
}
