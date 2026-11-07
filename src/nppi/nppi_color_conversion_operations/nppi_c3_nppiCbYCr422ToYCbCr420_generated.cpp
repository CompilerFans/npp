// Generated API wrappers -- do not edit by hand.
// family: nppiCbYCr422ToYCbCr420


#include "npp.h"
#include "nppi_c3_yuv_layout_executor.h"

NppStatus nppiCbYCr422ToYCbCr420_8u_C2P2R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDstY, int nDstYStep, Npp8u * pDstCbCr, int nDstCbCrStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::c3yuv::c2ToP2(pSrc, nSrcStep, pDstY, nDstYStep, pDstCbCr, nDstCbCrStep, oSizeROI, nppStreamCtx.hStream, 2, 1, 2);
}

NppStatus nppiCbYCr422ToYCbCr420_8u_C2P3R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst[3], int rDstStep[3], NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::c3yuv::c2ToP3(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, nppStreamCtx.hStream, 2, 1, 2, 0);
}

NppStatus nppiCbYCr422ToYCbCr420_8u_C2P2R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDstY, int nDstYStep, Npp8u * pDstCbCr, int nDstCbCrStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiCbYCr422ToYCbCr420_8u_C2P2R_Ctx(pSrc, nSrcStep, pDstY, nDstYStep, pDstCbCr, nDstCbCrStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCbYCr422ToYCbCr420_8u_C2P3R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst[3], int rDstStep[3], NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiCbYCr422ToYCbCr420_8u_C2P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, nppStreamCtx);
}
