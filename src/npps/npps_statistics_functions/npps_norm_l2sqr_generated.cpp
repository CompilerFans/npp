// Generated API wrappers -- do not edit by hand.
// family: nppsNorm_L2Sqr


#include "npp.h"
#include "npps_statistics_executor.h"

NppStatus nppsNormDiffL2SqrGetBufferSize_16s64s_Sfs_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::norm_buffer_size<Npp16s, npps::statistics::NormKind::L2Sqr>(nLength, hpBufferSize);
}

NppStatus nppsNormDiff_L2Sqr_16s64s_Sfs_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, NppSignalLength nLength, Npp64s * pNorm, int nScaleFactor, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::normdiff<Npp16s, npps::statistics::NormKind::L2Sqr, Npp64s>(pSrc1, pSrc2, nLength, pNorm, nScaleFactor, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsNormL2SqrGetBufferSize_16s64s_Sfs_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::norm_buffer_size<Npp16s, npps::statistics::NormKind::L2Sqr>(nLength, hpBufferSize);
}

NppStatus nppsNorm_L2Sqr_16s64s_Sfs_Ctx(const Npp16s * pSrc, NppSignalLength nLength, Npp64s * pNorm, int nScaleFactor, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::norm<Npp16s, npps::statistics::NormKind::L2Sqr, Npp64s>(pSrc, nLength, pNorm, nScaleFactor, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsNormDiffL2SqrGetBufferSize_16s64s_Sfs(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsNormDiffL2SqrGetBufferSize_16s64s_Sfs_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsNormDiff_L2Sqr_16s64s_Sfs(const Npp16s * pSrc1, const Npp16s * pSrc2, NppSignalLength nLength, Npp64s * pNorm, int nScaleFactor, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsNormDiff_L2Sqr_16s64s_Sfs_Ctx(pSrc1, pSrc2, nLength, pNorm, nScaleFactor, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsNormL2SqrGetBufferSize_16s64s_Sfs(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsNormL2SqrGetBufferSize_16s64s_Sfs_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsNorm_L2Sqr_16s64s_Sfs(const Npp16s * pSrc, NppSignalLength nLength, Npp64s * pNorm, int nScaleFactor, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsNorm_L2Sqr_16s64s_Sfs_Ctx(pSrc, nLength, pNorm, nScaleFactor, pDeviceBuffer, nppStreamCtx);
}
