// Generated API wrappers -- do not edit by hand.
// family: nppiWMSSSIMGetBufferHostSize


#include "npp.h"
#include "nppi_ssim_executor.h"

NppStatus nppiWMSSSIMGetBufferHostSize_8u_C1R_Ctx(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppi::statistics::wmsssimGetBufferHostSize(oSizeROI, 1, hpBufferSize);
}

NppStatus nppiWMSSSIMGetBufferHostSize_8u_C3R_Ctx(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppi::statistics::wmsssimGetBufferHostSize(oSizeROI, 3, hpBufferSize);
}

NppStatus nppiWMSSSIMGetBufferHostSize_8u_C1R(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWMSSSIMGetBufferHostSize_8u_C1R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}

NppStatus nppiWMSSSIMGetBufferHostSize_8u_C3R(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWMSSSIMGetBufferHostSize_8u_C3R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NppStatus nppiWMSSSIMGetBufferHostSize_8u_C1R_Ctx(NppiSize oSizeROI, size_t * hpBufferSize, NppStreamContext nppStreamCtx) {
  int hpBufferSizeInt = 0;
  NppStatus status = nppiWMSSSIMGetBufferHostSize_8u_C1R_Ctx(oSizeROI, &hpBufferSizeInt, nppStreamCtx);
  if (status == NPP_NO_ERROR && hpBufferSize) *hpBufferSize = (size_t)hpBufferSizeInt;
  return status;
}
#endif

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NppStatus nppiWMSSSIMGetBufferHostSize_8u_C3R_Ctx(NppiSize oSizeROI, size_t * hpBufferSize, NppStreamContext nppStreamCtx) {
  int hpBufferSizeInt = 0;
  NppStatus status = nppiWMSSSIMGetBufferHostSize_8u_C3R_Ctx(oSizeROI, &hpBufferSizeInt, nppStreamCtx);
  if (status == NPP_NO_ERROR && hpBufferSize) *hpBufferSize = (size_t)hpBufferSizeInt;
  return status;
}
#endif

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NppStatus nppiWMSSSIMGetBufferHostSize_8u_C1R(NppiSize oSizeROI, size_t * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) return status;
  return nppiWMSSSIMGetBufferHostSize_8u_C1R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}
#endif

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NppStatus nppiWMSSSIMGetBufferHostSize_8u_C3R(NppiSize oSizeROI, size_t * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) return status;
  return nppiWMSSSIMGetBufferHostSize_8u_C3R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}
#endif

// NPP 13.4 additions.  The SSIM-family sizes are undocumented internals; the
// library reports a conservative requirement (see nppi_ssim_executor.cu), so
// these forward the same per-channel formula with four channels.
NppStatus nppiWMSSSIMGetBufferHostSize_8u_C4R_Ctx(NppiSize oSizeROI, size_t * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  NPP_BUFFER_SIZE_TYPE hpBufferSizeTmp = 0;
  NppStatus status = nppi::statistics::wmsssimGetBufferHostSize(oSizeROI, 4, &hpBufferSizeTmp);
  if (status == NPP_NO_ERROR && hpBufferSize) *hpBufferSize = (size_t)hpBufferSizeTmp;
  return status;
}

NppStatus nppiWMSSSIMBatchGetBufferHostSize_8u_C4R_Ctx(NppiSize oSizeROI, size_t * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  NPP_BUFFER_SIZE_TYPE hpBufferSizeTmp = 0;
  NppStatus status = nppi::statistics::wmsssimBatchGetBufferHostSize(oSizeROI, 4, &hpBufferSizeTmp);
  if (status == NPP_NO_ERROR && hpBufferSize) *hpBufferSize = (size_t)hpBufferSizeTmp;
  return status;
}
