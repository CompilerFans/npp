// Hand-written wrappers for nppiMinMaxIndx (irregular mask+COI signatures
// that the family generator cannot express). Calls the shared executor
// nppi::statistics::runMinMaxIndx<D,Channels>; buffer-size helpers forward to
// nppi::statistics::minMaxIndxGetBufferHostSize.
// family: nppiMinMaxIndx, nppiMinMaxIndxGetBufferHostSize
#include "npp.h"
#include "nppi_statistics_executor.h"

NppStatus nppiMinMaxIndx_8s_C1R_Ctx(const Npp8s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8s * pMinValue, Npp8s * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runMinMaxIndx<Npp8s,1>(pSrc, nSrcStep, nullptr, 0, oSizeROI, 0, pMinValue, pMaxValue, pMinIndex, pMaxIndex, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppiMinMaxIndx_8s_C1R(const Npp8s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8s * pMinValue, Npp8s * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status0 = nppGetStreamContext(&nppStreamCtx);
  if (status0 != NPP_NO_ERROR) return status0;
  return nppiMinMaxIndx_8s_C1R_Ctx(pSrc, nSrcStep, oSizeROI, pMinValue, pMaxValue, pMinIndex, pMaxIndex, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiMinMaxIndx_8s_C1MR_Ctx(const Npp8s * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, Npp8s * pMinValue, Npp8s * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runMinMaxIndx<Npp8s,1>(pSrc, nSrcStep, pMask, nMaskStep, oSizeROI, 0, pMinValue, pMaxValue, pMinIndex, pMaxIndex, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppiMinMaxIndx_8s_C1MR(const Npp8s * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, Npp8s * pMinValue, Npp8s * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status0 = nppGetStreamContext(&nppStreamCtx);
  if (status0 != NPP_NO_ERROR) return status0;
  return nppiMinMaxIndx_8s_C1MR_Ctx(pSrc, nSrcStep, pMask, nMaskStep, oSizeROI, pMinValue, pMaxValue, pMinIndex, pMaxIndex, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiMinMaxIndx_8s_C3CR_Ctx(const Npp8s * pSrc, int nSrcStep, NppiSize oSizeROI, int nCOI, Npp8s * pMinValue, Npp8s * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runMinMaxIndx<Npp8s,3>(pSrc, nSrcStep, nullptr, 0, oSizeROI, nCOI, pMinValue, pMaxValue, pMinIndex, pMaxIndex, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppiMinMaxIndx_8s_C3CR(const Npp8s * pSrc, int nSrcStep, NppiSize oSizeROI, int nCOI, Npp8s * pMinValue, Npp8s * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status0 = nppGetStreamContext(&nppStreamCtx);
  if (status0 != NPP_NO_ERROR) return status0;
  return nppiMinMaxIndx_8s_C3CR_Ctx(pSrc, nSrcStep, oSizeROI, nCOI, pMinValue, pMaxValue, pMinIndex, pMaxIndex, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiMinMaxIndx_8s_C3CMR_Ctx(const Npp8s * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, int nCOI, Npp8s * pMinValue, Npp8s * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runMinMaxIndx<Npp8s,3>(pSrc, nSrcStep, pMask, nMaskStep, oSizeROI, nCOI, pMinValue, pMaxValue, pMinIndex, pMaxIndex, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppiMinMaxIndx_8s_C3CMR(const Npp8s * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, int nCOI, Npp8s * pMinValue, Npp8s * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status0 = nppGetStreamContext(&nppStreamCtx);
  if (status0 != NPP_NO_ERROR) return status0;
  return nppiMinMaxIndx_8s_C3CMR_Ctx(pSrc, nSrcStep, pMask, nMaskStep, oSizeROI, nCOI, pMinValue, pMaxValue, pMinIndex, pMaxIndex, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiMinMaxIndx_8u_C1R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pMinValue, Npp8u * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runMinMaxIndx<Npp8u,1>(pSrc, nSrcStep, nullptr, 0, oSizeROI, 0, pMinValue, pMaxValue, pMinIndex, pMaxIndex, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppiMinMaxIndx_8u_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pMinValue, Npp8u * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status0 = nppGetStreamContext(&nppStreamCtx);
  if (status0 != NPP_NO_ERROR) return status0;
  return nppiMinMaxIndx_8u_C1R_Ctx(pSrc, nSrcStep, oSizeROI, pMinValue, pMaxValue, pMinIndex, pMaxIndex, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiMinMaxIndx_8u_C1MR_Ctx(const Npp8u * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, Npp8u * pMinValue, Npp8u * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runMinMaxIndx<Npp8u,1>(pSrc, nSrcStep, pMask, nMaskStep, oSizeROI, 0, pMinValue, pMaxValue, pMinIndex, pMaxIndex, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppiMinMaxIndx_8u_C1MR(const Npp8u * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, Npp8u * pMinValue, Npp8u * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status0 = nppGetStreamContext(&nppStreamCtx);
  if (status0 != NPP_NO_ERROR) return status0;
  return nppiMinMaxIndx_8u_C1MR_Ctx(pSrc, nSrcStep, pMask, nMaskStep, oSizeROI, pMinValue, pMaxValue, pMinIndex, pMaxIndex, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiMinMaxIndx_8u_C3CR_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, int nCOI, Npp8u * pMinValue, Npp8u * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runMinMaxIndx<Npp8u,3>(pSrc, nSrcStep, nullptr, 0, oSizeROI, nCOI, pMinValue, pMaxValue, pMinIndex, pMaxIndex, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppiMinMaxIndx_8u_C3CR(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, int nCOI, Npp8u * pMinValue, Npp8u * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status0 = nppGetStreamContext(&nppStreamCtx);
  if (status0 != NPP_NO_ERROR) return status0;
  return nppiMinMaxIndx_8u_C3CR_Ctx(pSrc, nSrcStep, oSizeROI, nCOI, pMinValue, pMaxValue, pMinIndex, pMaxIndex, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiMinMaxIndx_8u_C3CMR_Ctx(const Npp8u * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, int nCOI, Npp8u * pMinValue, Npp8u * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runMinMaxIndx<Npp8u,3>(pSrc, nSrcStep, pMask, nMaskStep, oSizeROI, nCOI, pMinValue, pMaxValue, pMinIndex, pMaxIndex, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppiMinMaxIndx_8u_C3CMR(const Npp8u * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, int nCOI, Npp8u * pMinValue, Npp8u * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status0 = nppGetStreamContext(&nppStreamCtx);
  if (status0 != NPP_NO_ERROR) return status0;
  return nppiMinMaxIndx_8u_C3CMR_Ctx(pSrc, nSrcStep, pMask, nMaskStep, oSizeROI, nCOI, pMinValue, pMaxValue, pMinIndex, pMaxIndex, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiMinMaxIndx_16u_C1R_Ctx(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp16u * pMinValue, Npp16u * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runMinMaxIndx<Npp16u,1>(pSrc, nSrcStep, nullptr, 0, oSizeROI, 0, pMinValue, pMaxValue, pMinIndex, pMaxIndex, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppiMinMaxIndx_16u_C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp16u * pMinValue, Npp16u * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status0 = nppGetStreamContext(&nppStreamCtx);
  if (status0 != NPP_NO_ERROR) return status0;
  return nppiMinMaxIndx_16u_C1R_Ctx(pSrc, nSrcStep, oSizeROI, pMinValue, pMaxValue, pMinIndex, pMaxIndex, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiMinMaxIndx_16u_C1MR_Ctx(const Npp16u * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, Npp16u * pMinValue, Npp16u * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runMinMaxIndx<Npp16u,1>(pSrc, nSrcStep, pMask, nMaskStep, oSizeROI, 0, pMinValue, pMaxValue, pMinIndex, pMaxIndex, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppiMinMaxIndx_16u_C1MR(const Npp16u * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, Npp16u * pMinValue, Npp16u * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status0 = nppGetStreamContext(&nppStreamCtx);
  if (status0 != NPP_NO_ERROR) return status0;
  return nppiMinMaxIndx_16u_C1MR_Ctx(pSrc, nSrcStep, pMask, nMaskStep, oSizeROI, pMinValue, pMaxValue, pMinIndex, pMaxIndex, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiMinMaxIndx_16u_C3CR_Ctx(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, int nCOI, Npp16u * pMinValue, Npp16u * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runMinMaxIndx<Npp16u,3>(pSrc, nSrcStep, nullptr, 0, oSizeROI, nCOI, pMinValue, pMaxValue, pMinIndex, pMaxIndex, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppiMinMaxIndx_16u_C3CR(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, int nCOI, Npp16u * pMinValue, Npp16u * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status0 = nppGetStreamContext(&nppStreamCtx);
  if (status0 != NPP_NO_ERROR) return status0;
  return nppiMinMaxIndx_16u_C3CR_Ctx(pSrc, nSrcStep, oSizeROI, nCOI, pMinValue, pMaxValue, pMinIndex, pMaxIndex, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiMinMaxIndx_16u_C3CMR_Ctx(const Npp16u * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, int nCOI, Npp16u * pMinValue, Npp16u * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runMinMaxIndx<Npp16u,3>(pSrc, nSrcStep, pMask, nMaskStep, oSizeROI, nCOI, pMinValue, pMaxValue, pMinIndex, pMaxIndex, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppiMinMaxIndx_16u_C3CMR(const Npp16u * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, int nCOI, Npp16u * pMinValue, Npp16u * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status0 = nppGetStreamContext(&nppStreamCtx);
  if (status0 != NPP_NO_ERROR) return status0;
  return nppiMinMaxIndx_16u_C3CMR_Ctx(pSrc, nSrcStep, pMask, nMaskStep, oSizeROI, nCOI, pMinValue, pMaxValue, pMinIndex, pMaxIndex, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiMinMaxIndx_32f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp32f * pMinValue, Npp32f * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runMinMaxIndx<Npp32f,1>(pSrc, nSrcStep, nullptr, 0, oSizeROI, 0, pMinValue, pMaxValue, pMinIndex, pMaxIndex, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppiMinMaxIndx_32f_C1R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp32f * pMinValue, Npp32f * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status0 = nppGetStreamContext(&nppStreamCtx);
  if (status0 != NPP_NO_ERROR) return status0;
  return nppiMinMaxIndx_32f_C1R_Ctx(pSrc, nSrcStep, oSizeROI, pMinValue, pMaxValue, pMinIndex, pMaxIndex, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiMinMaxIndx_32f_C1MR_Ctx(const Npp32f * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, Npp32f * pMinValue, Npp32f * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runMinMaxIndx<Npp32f,1>(pSrc, nSrcStep, pMask, nMaskStep, oSizeROI, 0, pMinValue, pMaxValue, pMinIndex, pMaxIndex, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppiMinMaxIndx_32f_C1MR(const Npp32f * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, Npp32f * pMinValue, Npp32f * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status0 = nppGetStreamContext(&nppStreamCtx);
  if (status0 != NPP_NO_ERROR) return status0;
  return nppiMinMaxIndx_32f_C1MR_Ctx(pSrc, nSrcStep, pMask, nMaskStep, oSizeROI, pMinValue, pMaxValue, pMinIndex, pMaxIndex, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiMinMaxIndx_32f_C3CR_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, int nCOI, Npp32f * pMinValue, Npp32f * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runMinMaxIndx<Npp32f,3>(pSrc, nSrcStep, nullptr, 0, oSizeROI, nCOI, pMinValue, pMaxValue, pMinIndex, pMaxIndex, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppiMinMaxIndx_32f_C3CR(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, int nCOI, Npp32f * pMinValue, Npp32f * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status0 = nppGetStreamContext(&nppStreamCtx);
  if (status0 != NPP_NO_ERROR) return status0;
  return nppiMinMaxIndx_32f_C3CR_Ctx(pSrc, nSrcStep, oSizeROI, nCOI, pMinValue, pMaxValue, pMinIndex, pMaxIndex, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiMinMaxIndx_32f_C3CMR_Ctx(const Npp32f * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, int nCOI, Npp32f * pMinValue, Npp32f * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runMinMaxIndx<Npp32f,3>(pSrc, nSrcStep, pMask, nMaskStep, oSizeROI, nCOI, pMinValue, pMaxValue, pMinIndex, pMaxIndex, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppiMinMaxIndx_32f_C3CMR(const Npp32f * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, int nCOI, Npp32f * pMinValue, Npp32f * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status0 = nppGetStreamContext(&nppStreamCtx);
  if (status0 != NPP_NO_ERROR) return status0;
  return nppiMinMaxIndx_32f_C3CMR_Ctx(pSrc, nSrcStep, pMask, nMaskStep, oSizeROI, nCOI, pMinValue, pMaxValue, pMinIndex, pMaxIndex, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiMinMaxIndxGetBufferHostSize_8s_C1R_Ctx(NppiSize oSizeROI, int * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppi::statistics::minMaxIndxGetBufferHostSize(oSizeROI, hpBufferSize);
}

NppStatus nppiMinMaxIndxGetBufferHostSize_8s_C1R(NppiSize oSizeROI, int * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status0 = nppGetStreamContext(&nppStreamCtx);
  if (status0 != NPP_NO_ERROR) return status0;
  return nppiMinMaxIndxGetBufferHostSize_8s_C1R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}

NppStatus nppiMinMaxIndxGetBufferHostSize_8s_C1MR_Ctx(NppiSize oSizeROI, int * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppi::statistics::minMaxIndxGetBufferHostSize(oSizeROI, hpBufferSize);
}

NppStatus nppiMinMaxIndxGetBufferHostSize_8s_C1MR(NppiSize oSizeROI, int * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status0 = nppGetStreamContext(&nppStreamCtx);
  if (status0 != NPP_NO_ERROR) return status0;
  return nppiMinMaxIndxGetBufferHostSize_8s_C1MR_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}

NppStatus nppiMinMaxIndxGetBufferHostSize_8s_C3CR_Ctx(NppiSize oSizeROI, int * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppi::statistics::minMaxIndxGetBufferHostSize(oSizeROI, hpBufferSize);
}

NppStatus nppiMinMaxIndxGetBufferHostSize_8s_C3CR(NppiSize oSizeROI, int * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status0 = nppGetStreamContext(&nppStreamCtx);
  if (status0 != NPP_NO_ERROR) return status0;
  return nppiMinMaxIndxGetBufferHostSize_8s_C3CR_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}

NppStatus nppiMinMaxIndxGetBufferHostSize_8s_C3CMR_Ctx(NppiSize oSizeROI, int * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppi::statistics::minMaxIndxGetBufferHostSize(oSizeROI, hpBufferSize);
}

NppStatus nppiMinMaxIndxGetBufferHostSize_8s_C3CMR(NppiSize oSizeROI, int * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status0 = nppGetStreamContext(&nppStreamCtx);
  if (status0 != NPP_NO_ERROR) return status0;
  return nppiMinMaxIndxGetBufferHostSize_8s_C3CMR_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}

NppStatus nppiMinMaxIndxGetBufferHostSize_8u_C1R_Ctx(NppiSize oSizeROI, int * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppi::statistics::minMaxIndxGetBufferHostSize(oSizeROI, hpBufferSize);
}

NppStatus nppiMinMaxIndxGetBufferHostSize_8u_C1R(NppiSize oSizeROI, int * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status0 = nppGetStreamContext(&nppStreamCtx);
  if (status0 != NPP_NO_ERROR) return status0;
  return nppiMinMaxIndxGetBufferHostSize_8u_C1R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}

NppStatus nppiMinMaxIndxGetBufferHostSize_8u_C1MR_Ctx(NppiSize oSizeROI, int * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppi::statistics::minMaxIndxGetBufferHostSize(oSizeROI, hpBufferSize);
}

NppStatus nppiMinMaxIndxGetBufferHostSize_8u_C1MR(NppiSize oSizeROI, int * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status0 = nppGetStreamContext(&nppStreamCtx);
  if (status0 != NPP_NO_ERROR) return status0;
  return nppiMinMaxIndxGetBufferHostSize_8u_C1MR_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}

NppStatus nppiMinMaxIndxGetBufferHostSize_8u_C3CR_Ctx(NppiSize oSizeROI, int * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppi::statistics::minMaxIndxGetBufferHostSize(oSizeROI, hpBufferSize);
}

NppStatus nppiMinMaxIndxGetBufferHostSize_8u_C3CR(NppiSize oSizeROI, int * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status0 = nppGetStreamContext(&nppStreamCtx);
  if (status0 != NPP_NO_ERROR) return status0;
  return nppiMinMaxIndxGetBufferHostSize_8u_C3CR_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}

NppStatus nppiMinMaxIndxGetBufferHostSize_8u_C3CMR_Ctx(NppiSize oSizeROI, int * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppi::statistics::minMaxIndxGetBufferHostSize(oSizeROI, hpBufferSize);
}

NppStatus nppiMinMaxIndxGetBufferHostSize_8u_C3CMR(NppiSize oSizeROI, int * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status0 = nppGetStreamContext(&nppStreamCtx);
  if (status0 != NPP_NO_ERROR) return status0;
  return nppiMinMaxIndxGetBufferHostSize_8u_C3CMR_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}

NppStatus nppiMinMaxIndxGetBufferHostSize_16u_C1R_Ctx(NppiSize oSizeROI, int * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppi::statistics::minMaxIndxGetBufferHostSize(oSizeROI, hpBufferSize);
}

NppStatus nppiMinMaxIndxGetBufferHostSize_16u_C1R(NppiSize oSizeROI, int * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status0 = nppGetStreamContext(&nppStreamCtx);
  if (status0 != NPP_NO_ERROR) return status0;
  return nppiMinMaxIndxGetBufferHostSize_16u_C1R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}

NppStatus nppiMinMaxIndxGetBufferHostSize_16u_C1MR_Ctx(NppiSize oSizeROI, int * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppi::statistics::minMaxIndxGetBufferHostSize(oSizeROI, hpBufferSize);
}

NppStatus nppiMinMaxIndxGetBufferHostSize_16u_C1MR(NppiSize oSizeROI, int * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status0 = nppGetStreamContext(&nppStreamCtx);
  if (status0 != NPP_NO_ERROR) return status0;
  return nppiMinMaxIndxGetBufferHostSize_16u_C1MR_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}

NppStatus nppiMinMaxIndxGetBufferHostSize_16u_C3CR_Ctx(NppiSize oSizeROI, int * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppi::statistics::minMaxIndxGetBufferHostSize(oSizeROI, hpBufferSize);
}

NppStatus nppiMinMaxIndxGetBufferHostSize_16u_C3CR(NppiSize oSizeROI, int * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status0 = nppGetStreamContext(&nppStreamCtx);
  if (status0 != NPP_NO_ERROR) return status0;
  return nppiMinMaxIndxGetBufferHostSize_16u_C3CR_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}

NppStatus nppiMinMaxIndxGetBufferHostSize_16u_C3CMR_Ctx(NppiSize oSizeROI, int * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppi::statistics::minMaxIndxGetBufferHostSize(oSizeROI, hpBufferSize);
}

NppStatus nppiMinMaxIndxGetBufferHostSize_16u_C3CMR(NppiSize oSizeROI, int * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status0 = nppGetStreamContext(&nppStreamCtx);
  if (status0 != NPP_NO_ERROR) return status0;
  return nppiMinMaxIndxGetBufferHostSize_16u_C3CMR_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}

NppStatus nppiMinMaxIndxGetBufferHostSize_32f_C1R_Ctx(NppiSize oSizeROI, int * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppi::statistics::minMaxIndxGetBufferHostSize(oSizeROI, hpBufferSize);
}

NppStatus nppiMinMaxIndxGetBufferHostSize_32f_C1R(NppiSize oSizeROI, int * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status0 = nppGetStreamContext(&nppStreamCtx);
  if (status0 != NPP_NO_ERROR) return status0;
  return nppiMinMaxIndxGetBufferHostSize_32f_C1R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}

NppStatus nppiMinMaxIndxGetBufferHostSize_32f_C1MR_Ctx(NppiSize oSizeROI, int * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppi::statistics::minMaxIndxGetBufferHostSize(oSizeROI, hpBufferSize);
}

NppStatus nppiMinMaxIndxGetBufferHostSize_32f_C1MR(NppiSize oSizeROI, int * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status0 = nppGetStreamContext(&nppStreamCtx);
  if (status0 != NPP_NO_ERROR) return status0;
  return nppiMinMaxIndxGetBufferHostSize_32f_C1MR_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}

NppStatus nppiMinMaxIndxGetBufferHostSize_32f_C3CR_Ctx(NppiSize oSizeROI, int * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppi::statistics::minMaxIndxGetBufferHostSize(oSizeROI, hpBufferSize);
}

NppStatus nppiMinMaxIndxGetBufferHostSize_32f_C3CR(NppiSize oSizeROI, int * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status0 = nppGetStreamContext(&nppStreamCtx);
  if (status0 != NPP_NO_ERROR) return status0;
  return nppiMinMaxIndxGetBufferHostSize_32f_C3CR_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}

NppStatus nppiMinMaxIndxGetBufferHostSize_32f_C3CMR_Ctx(NppiSize oSizeROI, int * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppi::statistics::minMaxIndxGetBufferHostSize(oSizeROI, hpBufferSize);
}

NppStatus nppiMinMaxIndxGetBufferHostSize_32f_C3CMR(NppiSize oSizeROI, int * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status0 = nppGetStreamContext(&nppStreamCtx);
  if (status0 != NPP_NO_ERROR) return status0;
  return nppiMinMaxIndxGetBufferHostSize_32f_C3CMR_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}

