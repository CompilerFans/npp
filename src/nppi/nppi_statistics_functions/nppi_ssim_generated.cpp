// Generated API wrappers -- do not edit by hand.
// family: nppiSSIM


#include "npp.h"
#include "nppi_ssim_executor.h"

NppStatus nppiSSIM_8u_C1R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, NppiSize oSizeROI, Npp32f * pSSIM, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runSSIM(pSrc1, nSrc1Step, pSrc2, nSrc2Step, oSizeROI, 1, pSSIM, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppiSSIM_8u_C3R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, NppiSize oSizeROI, Npp32f * pSSIM, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runSSIM(pSrc1, nSrc1Step, pSrc2, nSrc2Step, oSizeROI, 3, pSSIM, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppiSSIM_8u_C1R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, NppiSize oSizeROI, Npp32f * pSSIM, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiSSIM_8u_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, oSizeROI, pSSIM, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiSSIM_8u_C3R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, NppiSize oSizeROI, Npp32f * pSSIM, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiSSIM_8u_C3R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, oSizeROI, pSSIM, pDeviceBuffer, nppStreamCtx);
}
