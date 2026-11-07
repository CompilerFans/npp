// Generated API wrappers -- do not edit by hand.
// family: nppiWMSSSIMBatchGetBufferHostSize


#include "npp.h"
#include "nppi_ssim_executor.h"

NppStatus nppiWMSSSIMBatchGetBufferHostSize_8u_C1R_Ctx(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppi::statistics::wmsssimBatchGetBufferHostSize(oSizeROI, 1, hpBufferSize);
}

NppStatus nppiWMSSSIMBatchGetBufferHostSize_8u_C3R_Ctx(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppi::statistics::wmsssimBatchGetBufferHostSize(oSizeROI, 3, hpBufferSize);
}

NppStatus nppiWMSSSIMBatchGetBufferHostSize_8u_C1R(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWMSSSIMBatchGetBufferHostSize_8u_C1R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}

NppStatus nppiWMSSSIMBatchGetBufferHostSize_8u_C3R(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWMSSSIMBatchGetBufferHostSize_8u_C3R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NppStatus nppiWMSSSIMBatchGetBufferHostSize_8u_C1R_Ctx(NppiSize oSizeROI, size_t * hpBufferSize, NppStreamContext nppStreamCtx) {
  int hpBufferSizeInt = 0;
  NppStatus status = nppiWMSSSIMBatchGetBufferHostSize_8u_C1R_Ctx(oSizeROI, &hpBufferSizeInt, nppStreamCtx);
  if (status == NPP_NO_ERROR && hpBufferSize) *hpBufferSize = (size_t)hpBufferSizeInt;
  return status;
}
#endif

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NppStatus nppiWMSSSIMBatchGetBufferHostSize_8u_C3R_Ctx(NppiSize oSizeROI, size_t * hpBufferSize, NppStreamContext nppStreamCtx) {
  int hpBufferSizeInt = 0;
  NppStatus status = nppiWMSSSIMBatchGetBufferHostSize_8u_C3R_Ctx(oSizeROI, &hpBufferSizeInt, nppStreamCtx);
  if (status == NPP_NO_ERROR && hpBufferSize) *hpBufferSize = (size_t)hpBufferSizeInt;
  return status;
}
#endif

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NppStatus nppiWMSSSIMBatchGetBufferHostSize_8u_C1R(NppiSize oSizeROI, size_t * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) return status;
  return nppiWMSSSIMBatchGetBufferHostSize_8u_C1R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}
#endif

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NppStatus nppiWMSSSIMBatchGetBufferHostSize_8u_C3R(NppiSize oSizeROI, size_t * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) return status;
  return nppiWMSSSIMBatchGetBufferHostSize_8u_C3R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}
#endif
