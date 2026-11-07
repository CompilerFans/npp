// Generated API wrappers -- do not edit by hand.
// family: nppiSSIMGetBufferHostSize


#include "npp.h"
#include "nppi_ssim_executor.h"

NppStatus nppiSSIMGetBufferHostSize_8u_C1R_Ctx(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppi::statistics::ssimGetBufferHostSize(oSizeROI, 1, hpBufferSize);
}

NppStatus nppiSSIMGetBufferHostSize_8u_C3R_Ctx(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppi::statistics::ssimGetBufferHostSize(oSizeROI, 3, hpBufferSize);
}

NppStatus nppiSSIMGetBufferHostSize_8u_C1R(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiSSIMGetBufferHostSize_8u_C1R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}

NppStatus nppiSSIMGetBufferHostSize_8u_C3R(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiSSIMGetBufferHostSize_8u_C3R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NppStatus nppiSSIMGetBufferHostSize_8u_C1R_Ctx(NppiSize oSizeROI, size_t * hpBufferSize, NppStreamContext nppStreamCtx) {
  int hpBufferSizeInt = 0;
  NppStatus status = nppiSSIMGetBufferHostSize_8u_C1R_Ctx(oSizeROI, &hpBufferSizeInt, nppStreamCtx);
  if (status == NPP_NO_ERROR && hpBufferSize) *hpBufferSize = (size_t)hpBufferSizeInt;
  return status;
}
#endif

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NppStatus nppiSSIMGetBufferHostSize_8u_C3R_Ctx(NppiSize oSizeROI, size_t * hpBufferSize, NppStreamContext nppStreamCtx) {
  int hpBufferSizeInt = 0;
  NppStatus status = nppiSSIMGetBufferHostSize_8u_C3R_Ctx(oSizeROI, &hpBufferSizeInt, nppStreamCtx);
  if (status == NPP_NO_ERROR && hpBufferSize) *hpBufferSize = (size_t)hpBufferSizeInt;
  return status;
}
#endif

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NppStatus nppiSSIMGetBufferHostSize_8u_C1R(NppiSize oSizeROI, size_t * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) return status;
  return nppiSSIMGetBufferHostSize_8u_C1R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}
#endif

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NppStatus nppiSSIMGetBufferHostSize_8u_C3R(NppiSize oSizeROI, size_t * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) return status;
  return nppiSSIMGetBufferHostSize_8u_C3R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}
#endif
