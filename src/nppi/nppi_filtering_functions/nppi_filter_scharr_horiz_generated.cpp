// Generated API wrappers -- do not edit by hand.
// family: nppiFilterScharrHoriz


#include "npp.h"
#include "nppi_filter_derivatives_executor.h"

NppStatus nppiFilterScharrHoriz_32f_C1R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterDeriv<Npp32f, Npp32f, 1, false, nppi::filter::kScharrHoriz>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
}

NppStatus nppiFilterScharrHoriz_8s16s_C1R_Ctx(const Npp8s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterDeriv<Npp8s, Npp16s, 1, false, nppi::filter::kScharrHoriz>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
}

NppStatus nppiFilterScharrHoriz_8u16s_C1R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterDeriv<Npp8u, Npp16s, 1, false, nppi::filter::kScharrHoriz>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
}

NppStatus nppiFilterScharrHoriz_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterScharrHoriz_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiFilterScharrHoriz_8s16s_C1R(const Npp8s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterScharrHoriz_8s16s_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiFilterScharrHoriz_8u16s_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterScharrHoriz_8u16s_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}
