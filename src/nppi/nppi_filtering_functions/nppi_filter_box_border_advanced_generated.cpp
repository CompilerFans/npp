// Generated API wrappers -- do not edit by hand.
// family: nppiFilterBoxBorderAdvanced


#include "npp.h"
#include "nppi_filter_window_executor.h"

NppStatus nppiFilterBoxBorderAdvanced_16s_C1R_Ctx(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, Npp8u * pBuffer, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterBoxBorderAdvanced<Npp16s, 1>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, eBorderType, pBuffer, nppStreamCtx.hStream);
}

NppStatus nppiFilterBoxBorderAdvanced_16s_C3R_Ctx(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, Npp8u * pBuffer, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterBoxBorderAdvanced<Npp16s, 3>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, eBorderType, pBuffer, nppStreamCtx.hStream);
}

NppStatus nppiFilterBoxBorderAdvanced_16s_C4R_Ctx(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, Npp8u * pBuffer, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterBoxBorderAdvanced<Npp16s, 4>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, eBorderType, pBuffer, nppStreamCtx.hStream);
}

NppStatus nppiFilterBoxBorderAdvanced_16u_C1R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, Npp8u * pBuffer, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterBoxBorderAdvanced<Npp16u, 1>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, eBorderType, pBuffer, nppStreamCtx.hStream);
}

NppStatus nppiFilterBoxBorderAdvanced_16u_C3R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, Npp8u * pBuffer, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterBoxBorderAdvanced<Npp16u, 3>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, eBorderType, pBuffer, nppStreamCtx.hStream);
}

NppStatus nppiFilterBoxBorderAdvanced_16u_C4R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, Npp8u * pBuffer, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterBoxBorderAdvanced<Npp16u, 4>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, eBorderType, pBuffer, nppStreamCtx.hStream);
}

NppStatus nppiFilterBoxBorderAdvanced_32f_C1R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, Npp8u * pBuffer, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterBoxBorderAdvanced<Npp32f, 1>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, eBorderType, pBuffer, nppStreamCtx.hStream);
}

NppStatus nppiFilterBoxBorderAdvanced_32f_C3R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, Npp8u * pBuffer, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterBoxBorderAdvanced<Npp32f, 3>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, eBorderType, pBuffer, nppStreamCtx.hStream);
}

NppStatus nppiFilterBoxBorderAdvanced_32f_C4R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, Npp8u * pBuffer, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterBoxBorderAdvanced<Npp32f, 4>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, eBorderType, pBuffer, nppStreamCtx.hStream);
}

NppStatus nppiFilterBoxBorderAdvanced_64f_C1R_Ctx(const Npp64f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp64f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, Npp8u * pBuffer64, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterBoxBorderAdvanced<Npp64f, 1>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, eBorderType, pBuffer64, nppStreamCtx.hStream);
}

NppStatus nppiFilterBoxBorderAdvanced_8u_C1R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, Npp8u * pBuffer, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterBoxBorderAdvanced<Npp8u, 1>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, eBorderType, pBuffer, nppStreamCtx.hStream);
}

NppStatus nppiFilterBoxBorderAdvanced_8u_C3R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, Npp8u * pBuffer, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterBoxBorderAdvanced<Npp8u, 3>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, eBorderType, pBuffer, nppStreamCtx.hStream);
}

NppStatus nppiFilterBoxBorderAdvanced_8u_C4R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, Npp8u * pBuffer, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterBoxBorderAdvanced<Npp8u, 4>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, eBorderType, pBuffer, nppStreamCtx.hStream);
}
