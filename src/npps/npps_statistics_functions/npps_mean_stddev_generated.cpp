// Generated API wrappers -- do not edit by hand.
// family: nppsMean


#include "npp.h"
#include "npps_statistics_executor.h"

NppStatus nppsMeanGetBufferSize_16s_Sfs_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::mean_buffer_size<Npp16s>(nLength, hpBufferSize);
}

NppStatus nppsMeanGetBufferSize_16sc_Sfs_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::mean_buffer_size<Npp16sc>(nLength, hpBufferSize);
}

NppStatus nppsMeanGetBufferSize_32f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::mean_buffer_size<Npp32f>(nLength, hpBufferSize);
}

NppStatus nppsMeanGetBufferSize_32fc_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::mean_buffer_size<Npp32fc>(nLength, hpBufferSize);
}

NppStatus nppsMeanGetBufferSize_32s_Sfs_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::mean_buffer_size<Npp32s>(nLength, hpBufferSize);
}

NppStatus nppsMeanGetBufferSize_64f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::mean_buffer_size<Npp64f>(nLength, hpBufferSize);
}

NppStatus nppsMeanGetBufferSize_64fc_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::mean_buffer_size<Npp64fc>(nLength, hpBufferSize);
}

NppStatus nppsMeanStdDevGetBufferSize_16s32s_Sfs_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::stddev_buffer_size<Npp16s>(nLength, hpBufferSize);
}

NppStatus nppsMeanStdDevGetBufferSize_16s_Sfs_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::stddev_buffer_size<Npp16s>(nLength, hpBufferSize);
}

NppStatus nppsMeanStdDevGetBufferSize_32f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::stddev_buffer_size<Npp32f>(nLength, hpBufferSize);
}

NppStatus nppsMeanStdDevGetBufferSize_64f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::stddev_buffer_size<Npp64f>(nLength, hpBufferSize);
}

NppStatus nppsMeanStdDev_16s32s_Sfs_Ctx(const Npp16s * pSrc, NppSignalLength nLength, Npp32s * pMean, Npp32s * pStdDev, int nScaleFactor, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::mean_stddev<Npp16s, Npp32s, Npp32s>(pSrc, nLength, pMean, pStdDev, nScaleFactor, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMeanStdDev_16s_Sfs_Ctx(const Npp16s * pSrc, NppSignalLength nLength, Npp16s * pMean, Npp16s * pStdDev, int nScaleFactor, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::mean_stddev<Npp16s, Npp16s, Npp16s>(pSrc, nLength, pMean, pStdDev, nScaleFactor, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMeanStdDev_32f_Ctx(const Npp32f * pSrc, NppSignalLength nLength, Npp32f * pMean, Npp32f * pStdDev, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::mean_stddev<Npp32f, Npp32f, Npp32f>(pSrc, nLength, pMean, pStdDev, 0, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMeanStdDev_64f_Ctx(const Npp64f * pSrc, NppSignalLength nLength, Npp64f * pMean, Npp64f * pStdDev, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::mean_stddev<Npp64f, Npp64f, Npp64f>(pSrc, nLength, pMean, pStdDev, 0, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMean_16s_Sfs_Ctx(const Npp16s * pSrc, NppSignalLength nLength, Npp16s * pMean, int nScaleFactor, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::mean<Npp16s, Npp16s>(pSrc, nLength, pMean, nScaleFactor, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMean_16sc_Sfs_Ctx(const Npp16sc * pSrc, NppSignalLength nLength, Npp16sc * pMean, int nScaleFactor, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::mean<Npp16sc, Npp16sc>(pSrc, nLength, pMean, nScaleFactor, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMean_32f_Ctx(const Npp32f * pSrc, NppSignalLength nLength, Npp32f * pMean, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::mean<Npp32f, Npp32f>(pSrc, nLength, pMean, 0, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMean_32fc_Ctx(const Npp32fc * pSrc, NppSignalLength nLength, Npp32fc * pMean, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::mean<Npp32fc, Npp32fc>(pSrc, nLength, pMean, 0, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMean_32s_Sfs_Ctx(const Npp32s * pSrc, NppSignalLength nLength, Npp32s * pMean, int nScaleFactor, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::mean<Npp32s, Npp32s>(pSrc, nLength, pMean, nScaleFactor, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMean_64f_Ctx(const Npp64f * pSrc, NppSignalLength nLength, Npp64f * pMean, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::mean<Npp64f, Npp64f>(pSrc, nLength, pMean, 0, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMean_64fc_Ctx(const Npp64fc * pSrc, NppSignalLength nLength, Npp64fc * pMean, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::mean<Npp64fc, Npp64fc>(pSrc, nLength, pMean, 0, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsStdDevGetBufferSize_16s32s_Sfs_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::stddev_buffer_size<Npp16s>(nLength, hpBufferSize);
}

NppStatus nppsStdDevGetBufferSize_16s_Sfs_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::stddev_buffer_size<Npp16s>(nLength, hpBufferSize);
}

NppStatus nppsStdDevGetBufferSize_32f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::stddev_buffer_size<Npp32f>(nLength, hpBufferSize);
}

NppStatus nppsStdDevGetBufferSize_64f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::stddev_buffer_size<Npp64f>(nLength, hpBufferSize);
}

NppStatus nppsStdDev_16s32s_Sfs_Ctx(const Npp16s * pSrc, NppSignalLength nLength, Npp32s * pStdDev, int nScaleFactor, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::stddev<Npp16s, Npp32s>(pSrc, nLength, pStdDev, nScaleFactor, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsStdDev_16s_Sfs_Ctx(const Npp16s * pSrc, NppSignalLength nLength, Npp16s * pStdDev, int nScaleFactor, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::stddev<Npp16s, Npp16s>(pSrc, nLength, pStdDev, nScaleFactor, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsStdDev_32f_Ctx(const Npp32f * pSrc, NppSignalLength nLength, Npp32f * pStdDev, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::stddev<Npp32f, Npp32f>(pSrc, nLength, pStdDev, 0, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsStdDev_64f_Ctx(const Npp64f * pSrc, NppSignalLength nLength, Npp64f * pStdDev, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::stddev<Npp64f, Npp64f>(pSrc, nLength, pStdDev, 0, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMeanGetBufferSize_16s_Sfs(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMeanGetBufferSize_16s_Sfs_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMeanGetBufferSize_16sc_Sfs(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMeanGetBufferSize_16sc_Sfs_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMeanGetBufferSize_32f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMeanGetBufferSize_32f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMeanGetBufferSize_32fc(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMeanGetBufferSize_32fc_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMeanGetBufferSize_32s_Sfs(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMeanGetBufferSize_32s_Sfs_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMeanGetBufferSize_64f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMeanGetBufferSize_64f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMeanGetBufferSize_64fc(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMeanGetBufferSize_64fc_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMeanStdDevGetBufferSize_16s32s_Sfs(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMeanStdDevGetBufferSize_16s32s_Sfs_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMeanStdDevGetBufferSize_16s_Sfs(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMeanStdDevGetBufferSize_16s_Sfs_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMeanStdDevGetBufferSize_32f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMeanStdDevGetBufferSize_32f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMeanStdDevGetBufferSize_64f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMeanStdDevGetBufferSize_64f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMeanStdDev_16s32s_Sfs(const Npp16s * pSrc, NppSignalLength nLength, Npp32s * pMean, Npp32s * pStdDev, int nScaleFactor, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMeanStdDev_16s32s_Sfs_Ctx(pSrc, nLength, pMean, pStdDev, nScaleFactor, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMeanStdDev_16s_Sfs(const Npp16s * pSrc, NppSignalLength nLength, Npp16s * pMean, Npp16s * pStdDev, int nScaleFactor, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMeanStdDev_16s_Sfs_Ctx(pSrc, nLength, pMean, pStdDev, nScaleFactor, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMeanStdDev_32f(const Npp32f * pSrc, NppSignalLength nLength, Npp32f * pMean, Npp32f * pStdDev, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMeanStdDev_32f_Ctx(pSrc, nLength, pMean, pStdDev, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMeanStdDev_64f(const Npp64f * pSrc, NppSignalLength nLength, Npp64f * pMean, Npp64f * pStdDev, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMeanStdDev_64f_Ctx(pSrc, nLength, pMean, pStdDev, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMean_16s_Sfs(const Npp16s * pSrc, NppSignalLength nLength, Npp16s * pMean, int nScaleFactor, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMean_16s_Sfs_Ctx(pSrc, nLength, pMean, nScaleFactor, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMean_16sc_Sfs(const Npp16sc * pSrc, NppSignalLength nLength, Npp16sc * pMean, int nScaleFactor, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMean_16sc_Sfs_Ctx(pSrc, nLength, pMean, nScaleFactor, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMean_32f(const Npp32f * pSrc, NppSignalLength nLength, Npp32f * pMean, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMean_32f_Ctx(pSrc, nLength, pMean, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMean_32fc(const Npp32fc * pSrc, NppSignalLength nLength, Npp32fc * pMean, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMean_32fc_Ctx(pSrc, nLength, pMean, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMean_32s_Sfs(const Npp32s * pSrc, NppSignalLength nLength, Npp32s * pMean, int nScaleFactor, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMean_32s_Sfs_Ctx(pSrc, nLength, pMean, nScaleFactor, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMean_64f(const Npp64f * pSrc, NppSignalLength nLength, Npp64f * pMean, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMean_64f_Ctx(pSrc, nLength, pMean, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMean_64fc(const Npp64fc * pSrc, NppSignalLength nLength, Npp64fc * pMean, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMean_64fc_Ctx(pSrc, nLength, pMean, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsStdDevGetBufferSize_16s32s_Sfs(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsStdDevGetBufferSize_16s32s_Sfs_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsStdDevGetBufferSize_16s_Sfs(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsStdDevGetBufferSize_16s_Sfs_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsStdDevGetBufferSize_32f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsStdDevGetBufferSize_32f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsStdDevGetBufferSize_64f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsStdDevGetBufferSize_64f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsStdDev_16s32s_Sfs(const Npp16s * pSrc, NppSignalLength nLength, Npp32s * pStdDev, int nScaleFactor, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsStdDev_16s32s_Sfs_Ctx(pSrc, nLength, pStdDev, nScaleFactor, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsStdDev_16s_Sfs(const Npp16s * pSrc, NppSignalLength nLength, Npp16s * pStdDev, int nScaleFactor, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsStdDev_16s_Sfs_Ctx(pSrc, nLength, pStdDev, nScaleFactor, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsStdDev_32f(const Npp32f * pSrc, NppSignalLength nLength, Npp32f * pStdDev, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsStdDev_32f_Ctx(pSrc, nLength, pStdDev, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsStdDev_64f(const Npp64f * pSrc, NppSignalLength nLength, Npp64f * pStdDev, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsStdDev_64f_Ctx(pSrc, nLength, pStdDev, pDeviceBuffer, nppStreamCtx);
}
