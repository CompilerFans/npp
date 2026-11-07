// Generated API wrappers -- do not edit by hand.
// family: nppiWMSSSIM


#include "npp.h"
#include "nppi_ssim_executor.h"

NppStatus nppiWMSSSIM_8u_C1R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, NppiSize oSizeROI, Npp32f * pMSSSIM, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runWMSSSIM(pSrc1, nSrc1Step, pSrc2, nSrc2Step, oSizeROI, 1, pMSSSIM, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppiWMSSSIM_8u_C3R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, NppiSize oSizeROI, Npp32f * pMSSSIM, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runWMSSSIM(pSrc1, nSrc1Step, pSrc2, nSrc2Step, oSizeROI, 3, pMSSSIM, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppiWMSSSIM_8u_C1R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, NppiSize oSizeROI, Npp32f * pMSSSIM, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWMSSSIM_8u_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, oSizeROI, pMSSSIM, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiWMSSSIM_8u_C3R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, NppiSize oSizeROI, Npp32f * pMSSSIM, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWMSSSIM_8u_C3R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, oSizeROI, pMSSSIM, pDeviceBuffer, nppStreamCtx);
}

// NPP 13.4 addition: four-channel variant.  pMSSSIM receives one score per
// channel (measured: each equals the corresponding channel's 8u_C1R result).
NppStatus nppiWMSSSIM_8u_C4R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, NppiSize oSizeROI, Npp32f * pMSSSIM, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runWMSSSIM(pSrc1, nSrc1Step, pSrc2, nSrc2Step, oSizeROI, 4, pMSSSIM, pDeviceBuffer, nppStreamCtx.hStream);
}
