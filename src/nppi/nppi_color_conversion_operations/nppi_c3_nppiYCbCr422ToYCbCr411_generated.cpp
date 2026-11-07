// Generated API wrappers -- do not edit by hand.
// family: nppiYCbCr422ToYCbCr411


#include "npp.h"
#include "nppi_c3_yuv_layout_executor.h"

NppStatus nppiYCbCr422ToYCbCr411_8u_C2P2R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDstY, int nDstYStep, Npp8u * pDstCbCr, int nDstCbCrStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::c3yuv::c2ToP2(pSrc, nSrcStep, pDstY, nDstYStep, pDstCbCr, nDstCbCrStep, oSizeROI, nppStreamCtx.hStream, 0, 2, 1);
}

NppStatus nppiYCbCr422ToYCbCr411_8u_C2P3R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst[3], int rDstStep[3], NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::c3yuv::c2ToP3(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, nppStreamCtx.hStream, 0, 2, 1, 0);
}

NppStatus nppiYCbCr422ToYCbCr411_8u_P3P2R_Ctx(const Npp8u * const pSrc[3], int rSrcStep[3], Npp8u * pDstY, int nDstYStep, Npp8u * pDstCbCr, int nDstCbCrStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::c3yuv::p3ToP2(pSrc, rSrcStep, pDstY, nDstYStep, pDstCbCr, nDstCbCrStep, oSizeROI, nppStreamCtx.hStream, 0, 0);
}

NppStatus nppiYCbCr422ToYCbCr411_8u_P3R_Ctx(const Npp8u * const pSrc[3], int rSrcStep[3], Npp8u * pDst[3], int rDstStep[3], NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::c3yuv::p3Resample(pSrc, rSrcStep, pDst, rDstStep, oSizeROI, nppStreamCtx.hStream, 0, 0, 1);
}

NppStatus nppiYCbCr422ToYCbCr411_8u_C2P2R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDstY, int nDstYStep, Npp8u * pDstCbCr, int nDstCbCrStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiYCbCr422ToYCbCr411_8u_C2P2R_Ctx(pSrc, nSrcStep, pDstY, nDstYStep, pDstCbCr, nDstCbCrStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiYCbCr422ToYCbCr411_8u_C2P3R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst[3], int rDstStep[3], NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiYCbCr422ToYCbCr411_8u_C2P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiYCbCr422ToYCbCr411_8u_P3P2R(const Npp8u * const pSrc[3], int rSrcStep[3], Npp8u * pDstY, int nDstYStep, Npp8u * pDstCbCr, int nDstCbCrStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiYCbCr422ToYCbCr411_8u_P3P2R_Ctx(pSrc, rSrcStep, pDstY, nDstYStep, pDstCbCr, nDstCbCrStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiYCbCr422ToYCbCr411_8u_P3R(const Npp8u * const pSrc[3], int rSrcStep[3], Npp8u * pDst[3], int rDstStep[3], NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiYCbCr422ToYCbCr411_8u_P3R_Ctx(pSrc, rSrcStep, pDst, rDstStep, oSizeROI, nppStreamCtx);
}
