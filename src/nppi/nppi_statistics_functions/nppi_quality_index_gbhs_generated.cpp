// Generated API wrappers -- do not edit by hand.
// family: nppiQualityIndexGetBufferHostSize


#include "npp.h"
#include "nppi_quality_metrics_executor.h"

NppStatus nppiQualityIndexGetBufferHostSize_16u32f_AC4R_Ctx(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppi::statistics::qualityIndexGetBufferHostSize(oSizeROI, 4, hpBufferSize);
}

NppStatus nppiQualityIndexGetBufferHostSize_16u32f_C1R_Ctx(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppi::statistics::qualityIndexGetBufferHostSize(oSizeROI, 1, hpBufferSize);
}

NppStatus nppiQualityIndexGetBufferHostSize_16u32f_C3R_Ctx(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppi::statistics::qualityIndexGetBufferHostSize(oSizeROI, 3, hpBufferSize);
}

NppStatus nppiQualityIndexGetBufferHostSize_32f_AC4R_Ctx(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppi::statistics::qualityIndexGetBufferHostSize(oSizeROI, 4, hpBufferSize);
}

NppStatus nppiQualityIndexGetBufferHostSize_32f_C1R_Ctx(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppi::statistics::qualityIndexGetBufferHostSize(oSizeROI, 1, hpBufferSize);
}

NppStatus nppiQualityIndexGetBufferHostSize_32f_C3R_Ctx(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppi::statistics::qualityIndexGetBufferHostSize(oSizeROI, 3, hpBufferSize);
}

NppStatus nppiQualityIndexGetBufferHostSize_8u32f_AC4R_Ctx(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppi::statistics::qualityIndexGetBufferHostSize(oSizeROI, 4, hpBufferSize);
}

NppStatus nppiQualityIndexGetBufferHostSize_8u32f_C1R_Ctx(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppi::statistics::qualityIndexGetBufferHostSize(oSizeROI, 1, hpBufferSize);
}

NppStatus nppiQualityIndexGetBufferHostSize_8u32f_C3R_Ctx(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppi::statistics::qualityIndexGetBufferHostSize(oSizeROI, 3, hpBufferSize);
}

NppStatus nppiQualityIndexGetBufferHostSize_16u32f_AC4R(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiQualityIndexGetBufferHostSize_16u32f_AC4R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}

NppStatus nppiQualityIndexGetBufferHostSize_16u32f_C1R(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiQualityIndexGetBufferHostSize_16u32f_C1R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}

NppStatus nppiQualityIndexGetBufferHostSize_16u32f_C3R(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiQualityIndexGetBufferHostSize_16u32f_C3R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}

NppStatus nppiQualityIndexGetBufferHostSize_32f_AC4R(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiQualityIndexGetBufferHostSize_32f_AC4R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}

NppStatus nppiQualityIndexGetBufferHostSize_32f_C1R(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiQualityIndexGetBufferHostSize_32f_C1R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}

NppStatus nppiQualityIndexGetBufferHostSize_32f_C3R(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiQualityIndexGetBufferHostSize_32f_C3R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}

NppStatus nppiQualityIndexGetBufferHostSize_8u32f_AC4R(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiQualityIndexGetBufferHostSize_8u32f_AC4R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}

NppStatus nppiQualityIndexGetBufferHostSize_8u32f_C1R(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiQualityIndexGetBufferHostSize_8u32f_C1R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}

NppStatus nppiQualityIndexGetBufferHostSize_8u32f_C3R(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiQualityIndexGetBufferHostSize_8u32f_C3R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NppStatus nppiQualityIndexGetBufferHostSize_16u32f_AC4R_Ctx(NppiSize oSizeROI, size_t * hpBufferSize, NppStreamContext nppStreamCtx) {
  int hpBufferSizeInt = 0;
  NppStatus status = nppiQualityIndexGetBufferHostSize_16u32f_AC4R_Ctx(oSizeROI, &hpBufferSizeInt, nppStreamCtx);
  if (status == NPP_NO_ERROR && hpBufferSize) *hpBufferSize = (size_t)hpBufferSizeInt;
  return status;
}
#endif

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NppStatus nppiQualityIndexGetBufferHostSize_16u32f_C1R_Ctx(NppiSize oSizeROI, size_t * hpBufferSize, NppStreamContext nppStreamCtx) {
  int hpBufferSizeInt = 0;
  NppStatus status = nppiQualityIndexGetBufferHostSize_16u32f_C1R_Ctx(oSizeROI, &hpBufferSizeInt, nppStreamCtx);
  if (status == NPP_NO_ERROR && hpBufferSize) *hpBufferSize = (size_t)hpBufferSizeInt;
  return status;
}
#endif

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NppStatus nppiQualityIndexGetBufferHostSize_16u32f_C3R_Ctx(NppiSize oSizeROI, size_t * hpBufferSize, NppStreamContext nppStreamCtx) {
  int hpBufferSizeInt = 0;
  NppStatus status = nppiQualityIndexGetBufferHostSize_16u32f_C3R_Ctx(oSizeROI, &hpBufferSizeInt, nppStreamCtx);
  if (status == NPP_NO_ERROR && hpBufferSize) *hpBufferSize = (size_t)hpBufferSizeInt;
  return status;
}
#endif

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NppStatus nppiQualityIndexGetBufferHostSize_32f_AC4R_Ctx(NppiSize oSizeROI, size_t * hpBufferSize, NppStreamContext nppStreamCtx) {
  int hpBufferSizeInt = 0;
  NppStatus status = nppiQualityIndexGetBufferHostSize_32f_AC4R_Ctx(oSizeROI, &hpBufferSizeInt, nppStreamCtx);
  if (status == NPP_NO_ERROR && hpBufferSize) *hpBufferSize = (size_t)hpBufferSizeInt;
  return status;
}
#endif

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NppStatus nppiQualityIndexGetBufferHostSize_32f_C1R_Ctx(NppiSize oSizeROI, size_t * hpBufferSize, NppStreamContext nppStreamCtx) {
  int hpBufferSizeInt = 0;
  NppStatus status = nppiQualityIndexGetBufferHostSize_32f_C1R_Ctx(oSizeROI, &hpBufferSizeInt, nppStreamCtx);
  if (status == NPP_NO_ERROR && hpBufferSize) *hpBufferSize = (size_t)hpBufferSizeInt;
  return status;
}
#endif

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NppStatus nppiQualityIndexGetBufferHostSize_32f_C3R_Ctx(NppiSize oSizeROI, size_t * hpBufferSize, NppStreamContext nppStreamCtx) {
  int hpBufferSizeInt = 0;
  NppStatus status = nppiQualityIndexGetBufferHostSize_32f_C3R_Ctx(oSizeROI, &hpBufferSizeInt, nppStreamCtx);
  if (status == NPP_NO_ERROR && hpBufferSize) *hpBufferSize = (size_t)hpBufferSizeInt;
  return status;
}
#endif

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NppStatus nppiQualityIndexGetBufferHostSize_8u32f_AC4R_Ctx(NppiSize oSizeROI, size_t * hpBufferSize, NppStreamContext nppStreamCtx) {
  int hpBufferSizeInt = 0;
  NppStatus status = nppiQualityIndexGetBufferHostSize_8u32f_AC4R_Ctx(oSizeROI, &hpBufferSizeInt, nppStreamCtx);
  if (status == NPP_NO_ERROR && hpBufferSize) *hpBufferSize = (size_t)hpBufferSizeInt;
  return status;
}
#endif

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NppStatus nppiQualityIndexGetBufferHostSize_8u32f_C1R_Ctx(NppiSize oSizeROI, size_t * hpBufferSize, NppStreamContext nppStreamCtx) {
  int hpBufferSizeInt = 0;
  NppStatus status = nppiQualityIndexGetBufferHostSize_8u32f_C1R_Ctx(oSizeROI, &hpBufferSizeInt, nppStreamCtx);
  if (status == NPP_NO_ERROR && hpBufferSize) *hpBufferSize = (size_t)hpBufferSizeInt;
  return status;
}
#endif

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NppStatus nppiQualityIndexGetBufferHostSize_8u32f_C3R_Ctx(NppiSize oSizeROI, size_t * hpBufferSize, NppStreamContext nppStreamCtx) {
  int hpBufferSizeInt = 0;
  NppStatus status = nppiQualityIndexGetBufferHostSize_8u32f_C3R_Ctx(oSizeROI, &hpBufferSizeInt, nppStreamCtx);
  if (status == NPP_NO_ERROR && hpBufferSize) *hpBufferSize = (size_t)hpBufferSizeInt;
  return status;
}
#endif

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NppStatus nppiQualityIndexGetBufferHostSize_16u32f_AC4R(NppiSize oSizeROI, size_t * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) return status;
  return nppiQualityIndexGetBufferHostSize_16u32f_AC4R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}
#endif

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NppStatus nppiQualityIndexGetBufferHostSize_16u32f_C1R(NppiSize oSizeROI, size_t * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) return status;
  return nppiQualityIndexGetBufferHostSize_16u32f_C1R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}
#endif

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NppStatus nppiQualityIndexGetBufferHostSize_16u32f_C3R(NppiSize oSizeROI, size_t * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) return status;
  return nppiQualityIndexGetBufferHostSize_16u32f_C3R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}
#endif

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NppStatus nppiQualityIndexGetBufferHostSize_32f_AC4R(NppiSize oSizeROI, size_t * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) return status;
  return nppiQualityIndexGetBufferHostSize_32f_AC4R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}
#endif

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NppStatus nppiQualityIndexGetBufferHostSize_32f_C1R(NppiSize oSizeROI, size_t * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) return status;
  return nppiQualityIndexGetBufferHostSize_32f_C1R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}
#endif

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NppStatus nppiQualityIndexGetBufferHostSize_32f_C3R(NppiSize oSizeROI, size_t * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) return status;
  return nppiQualityIndexGetBufferHostSize_32f_C3R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}
#endif

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NppStatus nppiQualityIndexGetBufferHostSize_8u32f_AC4R(NppiSize oSizeROI, size_t * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) return status;
  return nppiQualityIndexGetBufferHostSize_8u32f_AC4R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}
#endif

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NppStatus nppiQualityIndexGetBufferHostSize_8u32f_C1R(NppiSize oSizeROI, size_t * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) return status;
  return nppiQualityIndexGetBufferHostSize_8u32f_C1R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}
#endif

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NppStatus nppiQualityIndexGetBufferHostSize_8u32f_C3R(NppiSize oSizeROI, size_t * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) return status;
  return nppiQualityIndexGetBufferHostSize_8u32f_C3R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
}
#endif
