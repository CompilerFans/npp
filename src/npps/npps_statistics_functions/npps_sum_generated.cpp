// Generated API wrappers -- do not edit by hand.
// family: nppsSum


#include "npp.h"
#include "npps_statistics_executor.h"

NppStatus nppsSumGetBufferSize_16s32s_Sfs_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::sum_buffer_size<Npp16s>(nLength, hpBufferSize);
}

NppStatus nppsSumGetBufferSize_16s_Sfs_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::sum_buffer_size<Npp16s>(nLength, hpBufferSize);
}

NppStatus nppsSumGetBufferSize_16sc32sc_Sfs_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::sum_buffer_size<Npp16sc>(nLength, hpBufferSize);
}

NppStatus nppsSumGetBufferSize_16sc_Sfs_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::sum_buffer_size<Npp16sc>(nLength, hpBufferSize);
}

NppStatus nppsSumGetBufferSize_32f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::sum_buffer_size<Npp32f>(nLength, hpBufferSize);
}

NppStatus nppsSumGetBufferSize_32fc_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::sum_buffer_size<Npp32fc>(nLength, hpBufferSize);
}

NppStatus nppsSumGetBufferSize_32s_Sfs_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::sum_buffer_size<Npp32s>(nLength, hpBufferSize);
}

NppStatus nppsSumGetBufferSize_64f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::sum_buffer_size<Npp64f>(nLength, hpBufferSize);
}

NppStatus nppsSumGetBufferSize_64fc_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::sum_buffer_size<Npp64fc>(nLength, hpBufferSize);
}

NppStatus nppsSum_16s32s_Sfs_Ctx(const Npp16s * pSrc, NppSignalLength nLength, Npp32s * pSum, int nScaleFactor, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::sum<Npp16s, Npp32s>(pSrc, nLength, pSum, nScaleFactor, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsSum_16s_Sfs_Ctx(const Npp16s * pSrc, NppSignalLength nLength, Npp16s * pSum, int nScaleFactor, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::sum<Npp16s, Npp16s>(pSrc, nLength, pSum, nScaleFactor, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsSum_16sc32sc_Sfs_Ctx(const Npp16sc * pSrc, NppSignalLength nLength, Npp32sc * pSum, int nScaleFactor, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::sum<Npp16sc, Npp32sc>(pSrc, nLength, pSum, nScaleFactor, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsSum_16sc_Sfs_Ctx(const Npp16sc * pSrc, NppSignalLength nLength, Npp16sc * pSum, int nScaleFactor, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::sum<Npp16sc, Npp16sc>(pSrc, nLength, pSum, nScaleFactor, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsSum_32f_Ctx(const Npp32f * pSrc, NppSignalLength nLength, Npp32f * pSum, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::sum<Npp32f, Npp32f>(pSrc, nLength, pSum, 0, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsSum_32fc_Ctx(const Npp32fc * pSrc, NppSignalLength nLength, Npp32fc * pSum, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::sum<Npp32fc, Npp32fc>(pSrc, nLength, pSum, 0, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsSum_32s_Sfs_Ctx(const Npp32s * pSrc, NppSignalLength nLength, Npp32s * pSum, int nScaleFactor, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::sum<Npp32s, Npp32s>(pSrc, nLength, pSum, nScaleFactor, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsSum_64f_Ctx(const Npp64f * pSrc, NppSignalLength nLength, Npp64f * pSum, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::sum<Npp64f, Npp64f>(pSrc, nLength, pSum, 0, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsSum_64fc_Ctx(const Npp64fc * pSrc, NppSignalLength nLength, Npp64fc * pSum, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::sum<Npp64fc, Npp64fc>(pSrc, nLength, pSum, 0, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsSumGetBufferSize_16s32s_Sfs(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsSumGetBufferSize_16s32s_Sfs_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsSumGetBufferSize_16s_Sfs(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsSumGetBufferSize_16s_Sfs_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsSumGetBufferSize_16sc32sc_Sfs(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsSumGetBufferSize_16sc32sc_Sfs_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsSumGetBufferSize_16sc_Sfs(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsSumGetBufferSize_16sc_Sfs_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsSumGetBufferSize_32f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsSumGetBufferSize_32f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsSumGetBufferSize_32fc(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsSumGetBufferSize_32fc_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsSumGetBufferSize_32s_Sfs(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsSumGetBufferSize_32s_Sfs_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsSumGetBufferSize_64f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsSumGetBufferSize_64f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsSumGetBufferSize_64fc(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsSumGetBufferSize_64fc_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsSum_16s32s_Sfs(const Npp16s * pSrc, NppSignalLength nLength, Npp32s * pSum, int nScaleFactor, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsSum_16s32s_Sfs_Ctx(pSrc, nLength, pSum, nScaleFactor, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsSum_16s_Sfs(const Npp16s * pSrc, NppSignalLength nLength, Npp16s * pSum, int nScaleFactor, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsSum_16s_Sfs_Ctx(pSrc, nLength, pSum, nScaleFactor, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsSum_16sc32sc_Sfs(const Npp16sc * pSrc, NppSignalLength nLength, Npp32sc * pSum, int nScaleFactor, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsSum_16sc32sc_Sfs_Ctx(pSrc, nLength, pSum, nScaleFactor, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsSum_16sc_Sfs(const Npp16sc * pSrc, NppSignalLength nLength, Npp16sc * pSum, int nScaleFactor, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsSum_16sc_Sfs_Ctx(pSrc, nLength, pSum, nScaleFactor, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsSum_32f(const Npp32f * pSrc, NppSignalLength nLength, Npp32f * pSum, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsSum_32f_Ctx(pSrc, nLength, pSum, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsSum_32fc(const Npp32fc * pSrc, NppSignalLength nLength, Npp32fc * pSum, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsSum_32fc_Ctx(pSrc, nLength, pSum, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsSum_32s_Sfs(const Npp32s * pSrc, NppSignalLength nLength, Npp32s * pSum, int nScaleFactor, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsSum_32s_Sfs_Ctx(pSrc, nLength, pSum, nScaleFactor, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsSum_64f(const Npp64f * pSrc, NppSignalLength nLength, Npp64f * pSum, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsSum_64f_Ctx(pSrc, nLength, pSum, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsSum_64fc(const Npp64fc * pSrc, NppSignalLength nLength, Npp64fc * pSum, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsSum_64fc_Ctx(pSrc, nLength, pSum, pDeviceBuffer, nppStreamCtx);
}
