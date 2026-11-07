// Generated API wrappers -- do not edit by hand.
// family: nppiYCbCr411


#include "npp.h"
#include "nppi_c3_yuv_layout_executor.h"

NppStatus nppiYCbCr411_8u_P2P3R_Ctx(const Npp8u * pSrcY, int nSrcYStep, const Npp8u * pSrcCbCr, int nSrcCbCrStep, Npp8u * pDst[3], int rDstStep[3], NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::c3yuv::p2ToP3_411(pSrcY, nSrcYStep, pSrcCbCr, nSrcCbCrStep, pDst, rDstStep, oSizeROI, nppStreamCtx.hStream);
}

NppStatus nppiYCbCr411_8u_P3P2R_Ctx(const Npp8u * const pSrc[3], int rSrcStep[3], Npp8u * pDstY, int nDstYStep, Npp8u * pDstCbCr, int nDstCbCrStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::c3yuv::p3ToP2(pSrc, rSrcStep, pDstY, nDstYStep, pDstCbCr, nDstCbCrStep, oSizeROI, nppStreamCtx.hStream, 0, 2);
}

NppStatus nppiYCbCr411_8u_P2P3R(const Npp8u * pSrcY, int nSrcYStep, const Npp8u * pSrcCbCr, int nSrcCbCrStep, Npp8u * pDst[3], int rDstStep[3], NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiYCbCr411_8u_P2P3R_Ctx(pSrcY, nSrcYStep, pSrcCbCr, nSrcCbCrStep, pDst, rDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiYCbCr411_8u_P3P2R(const Npp8u * const pSrc[3], int rSrcStep[3], Npp8u * pDstY, int nDstYStep, Npp8u * pDstCbCr, int nDstCbCrStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiYCbCr411_8u_P3P2R_Ctx(pSrc, rSrcStep, pDstY, nDstYStep, pDstCbCr, nDstCbCrStep, oSizeROI, nppStreamCtx);
}
