// Generated API wrappers -- do not edit by hand.
// family: nppsNorm_Inf


#include "npp.h"
#include "npps_statistics_executor.h"

NppStatus nppsNormDiffInfGetBufferSize_16s32f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::norm_buffer_size<Npp16s, npps::statistics::NormKind::Inf>(nLength, hpBufferSize);
}

NppStatus nppsNormDiffInfGetBufferSize_16s32s_Sfs_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::norm_buffer_size<Npp16s, npps::statistics::NormKind::Inf>(nLength, hpBufferSize);
}

NppStatus nppsNormDiffInfGetBufferSize_32f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::norm_buffer_size<Npp32f, npps::statistics::NormKind::Inf>(nLength, hpBufferSize);
}

NppStatus nppsNormDiffInfGetBufferSize_32fc32f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::norm_buffer_size<Npp32fc, npps::statistics::NormKind::Inf>(nLength, hpBufferSize);
}

NppStatus nppsNormDiffInfGetBufferSize_64f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::norm_buffer_size<Npp64f, npps::statistics::NormKind::Inf>(nLength, hpBufferSize);
}

NppStatus nppsNormDiffInfGetBufferSize_64fc64f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::norm_buffer_size<Npp64fc, npps::statistics::NormKind::Inf>(nLength, hpBufferSize);
}

NppStatus nppsNormDiff_Inf_16s32f_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, NppSignalLength nLength, Npp32f * pNorm, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::normdiff<Npp16s, npps::statistics::NormKind::Inf, Npp32f>(pSrc1, pSrc2, nLength, pNorm, 0, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsNormDiff_Inf_16s32s_Sfs_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, NppSignalLength nLength, Npp32s * pNorm, int nScaleFactor, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::normdiff<Npp16s, npps::statistics::NormKind::Inf, Npp32s>(pSrc1, pSrc2, nLength, pNorm, nScaleFactor, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsNormDiff_Inf_32f_Ctx(const Npp32f * pSrc1, const Npp32f * pSrc2, NppSignalLength nLength, Npp32f * pNorm, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::normdiff<Npp32f, npps::statistics::NormKind::Inf, Npp32f>(pSrc1, pSrc2, nLength, pNorm, 0, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsNormDiff_Inf_32fc32f_Ctx(const Npp32fc * pSrc1, const Npp32fc * pSrc2, NppSignalLength nLength, Npp32f * pNorm, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::normdiff<Npp32fc, npps::statistics::NormKind::Inf, Npp32f>(pSrc1, pSrc2, nLength, pNorm, 0, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsNormDiff_Inf_64f_Ctx(const Npp64f * pSrc1, const Npp64f * pSrc2, NppSignalLength nLength, Npp64f * pNorm, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::normdiff<Npp64f, npps::statistics::NormKind::Inf, Npp64f>(pSrc1, pSrc2, nLength, pNorm, 0, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsNormDiff_Inf_64fc64f_Ctx(const Npp64fc * pSrc1, const Npp64fc * pSrc2, NppSignalLength nLength, Npp64f * pNorm, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::normdiff<Npp64fc, npps::statistics::NormKind::Inf, Npp64f>(pSrc1, pSrc2, nLength, pNorm, 0, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsNormInfGetBufferSize_16s32f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::norm_buffer_size<Npp16s, npps::statistics::NormKind::Inf>(nLength, hpBufferSize);
}

NppStatus nppsNormInfGetBufferSize_16s32s_Sfs_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::norm_buffer_size<Npp16s, npps::statistics::NormKind::Inf>(nLength, hpBufferSize);
}

NppStatus nppsNormInfGetBufferSize_32f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::norm_buffer_size<Npp32f, npps::statistics::NormKind::Inf>(nLength, hpBufferSize);
}

NppStatus nppsNormInfGetBufferSize_32fc32f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::norm_buffer_size<Npp32fc, npps::statistics::NormKind::Inf>(nLength, hpBufferSize);
}

NppStatus nppsNormInfGetBufferSize_64f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::norm_buffer_size<Npp64f, npps::statistics::NormKind::Inf>(nLength, hpBufferSize);
}

NppStatus nppsNormInfGetBufferSize_64fc64f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::norm_buffer_size<Npp64fc, npps::statistics::NormKind::Inf>(nLength, hpBufferSize);
}

NppStatus nppsNorm_Inf_16s32f_Ctx(const Npp16s * pSrc, NppSignalLength nLength, Npp32f * pNorm, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::norm<Npp16s, npps::statistics::NormKind::Inf, Npp32f>(pSrc, nLength, pNorm, 0, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsNorm_Inf_16s32s_Sfs_Ctx(const Npp16s * pSrc, NppSignalLength nLength, Npp32s * pNorm, int nScaleFactor, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::norm<Npp16s, npps::statistics::NormKind::Inf, Npp32s>(pSrc, nLength, pNorm, nScaleFactor, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsNorm_Inf_32f_Ctx(const Npp32f * pSrc, NppSignalLength nLength, Npp32f * pNorm, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::norm<Npp32f, npps::statistics::NormKind::Inf, Npp32f>(pSrc, nLength, pNorm, 0, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsNorm_Inf_32fc32f_Ctx(const Npp32fc * pSrc, NppSignalLength nLength, Npp32f * pNorm, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::norm<Npp32fc, npps::statistics::NormKind::Inf, Npp32f>(pSrc, nLength, pNorm, 0, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsNorm_Inf_64f_Ctx(const Npp64f * pSrc, NppSignalLength nLength, Npp64f * pNorm, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::norm<Npp64f, npps::statistics::NormKind::Inf, Npp64f>(pSrc, nLength, pNorm, 0, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsNorm_Inf_64fc64f_Ctx(const Npp64fc * pSrc, NppSignalLength nLength, Npp64f * pNorm, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::norm<Npp64fc, npps::statistics::NormKind::Inf, Npp64f>(pSrc, nLength, pNorm, 0, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsNormDiffInfGetBufferSize_16s32f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsNormDiffInfGetBufferSize_16s32f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsNormDiffInfGetBufferSize_16s32s_Sfs(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsNormDiffInfGetBufferSize_16s32s_Sfs_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsNormDiffInfGetBufferSize_32f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsNormDiffInfGetBufferSize_32f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsNormDiffInfGetBufferSize_32fc32f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsNormDiffInfGetBufferSize_32fc32f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsNormDiffInfGetBufferSize_64f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsNormDiffInfGetBufferSize_64f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsNormDiffInfGetBufferSize_64fc64f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsNormDiffInfGetBufferSize_64fc64f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsNormDiff_Inf_16s32f(const Npp16s * pSrc1, const Npp16s * pSrc2, NppSignalLength nLength, Npp32f * pNorm, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsNormDiff_Inf_16s32f_Ctx(pSrc1, pSrc2, nLength, pNorm, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsNormDiff_Inf_16s32s_Sfs(const Npp16s * pSrc1, const Npp16s * pSrc2, NppSignalLength nLength, Npp32s * pNorm, int nScaleFactor, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsNormDiff_Inf_16s32s_Sfs_Ctx(pSrc1, pSrc2, nLength, pNorm, nScaleFactor, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsNormDiff_Inf_32f(const Npp32f * pSrc1, const Npp32f * pSrc2, NppSignalLength nLength, Npp32f * pNorm, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsNormDiff_Inf_32f_Ctx(pSrc1, pSrc2, nLength, pNorm, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsNormDiff_Inf_32fc32f(const Npp32fc * pSrc1, const Npp32fc * pSrc2, NppSignalLength nLength, Npp32f * pNorm, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsNormDiff_Inf_32fc32f_Ctx(pSrc1, pSrc2, nLength, pNorm, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsNormDiff_Inf_64f(const Npp64f * pSrc1, const Npp64f * pSrc2, NppSignalLength nLength, Npp64f * pNorm, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsNormDiff_Inf_64f_Ctx(pSrc1, pSrc2, nLength, pNorm, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsNormDiff_Inf_64fc64f(const Npp64fc * pSrc1, const Npp64fc * pSrc2, NppSignalLength nLength, Npp64f * pNorm, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsNormDiff_Inf_64fc64f_Ctx(pSrc1, pSrc2, nLength, pNorm, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsNormInfGetBufferSize_16s32f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsNormInfGetBufferSize_16s32f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsNormInfGetBufferSize_16s32s_Sfs(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsNormInfGetBufferSize_16s32s_Sfs_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsNormInfGetBufferSize_32f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsNormInfGetBufferSize_32f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsNormInfGetBufferSize_32fc32f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsNormInfGetBufferSize_32fc32f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsNormInfGetBufferSize_64f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsNormInfGetBufferSize_64f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsNormInfGetBufferSize_64fc64f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsNormInfGetBufferSize_64fc64f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsNorm_Inf_16s32f(const Npp16s * pSrc, NppSignalLength nLength, Npp32f * pNorm, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsNorm_Inf_16s32f_Ctx(pSrc, nLength, pNorm, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsNorm_Inf_16s32s_Sfs(const Npp16s * pSrc, NppSignalLength nLength, Npp32s * pNorm, int nScaleFactor, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsNorm_Inf_16s32s_Sfs_Ctx(pSrc, nLength, pNorm, nScaleFactor, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsNorm_Inf_32f(const Npp32f * pSrc, NppSignalLength nLength, Npp32f * pNorm, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsNorm_Inf_32f_Ctx(pSrc, nLength, pNorm, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsNorm_Inf_32fc32f(const Npp32fc * pSrc, NppSignalLength nLength, Npp32f * pNorm, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsNorm_Inf_32fc32f_Ctx(pSrc, nLength, pNorm, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsNorm_Inf_64f(const Npp64f * pSrc, NppSignalLength nLength, Npp64f * pNorm, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsNorm_Inf_64f_Ctx(pSrc, nLength, pNorm, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsNorm_Inf_64fc64f(const Npp64fc * pSrc, NppSignalLength nLength, Npp64f * pNorm, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsNorm_Inf_64fc64f_Ctx(pSrc, nLength, pNorm, pDeviceBuffer, nppStreamCtx);
}
