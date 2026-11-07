// Generated API wrappers -- do not edit by hand.
// family: nppiFilterSobelCross


#include "npp.h"
#include "nppi_filter_derivatives_executor.h"

NppStatus nppiFilterSobelCross_32f_C1R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterDerivMask<Npp32f, Npp32f, nppi::filter::kSobelCross>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx.hStream);
}

NppStatus nppiFilterSobelCross_8s16s_C1R_Ctx(const Npp8s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterDerivMask<Npp8s, Npp16s, nppi::filter::kSobelCross>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx.hStream);
}

NppStatus nppiFilterSobelCross_8u16s_C1R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterDerivMask<Npp8u, Npp16s, nppi::filter::kSobelCross>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx.hStream);
}

NppStatus nppiFilterSobelCross_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterSobelCross_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx);
}

NppStatus nppiFilterSobelCross_8s16s_C1R(const Npp8s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterSobelCross_8s16s_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx);
}

NppStatus nppiFilterSobelCross_8u16s_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterSobelCross_8u16s_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx);
}
