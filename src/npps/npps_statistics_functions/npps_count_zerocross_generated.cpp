// Generated API wrappers -- do not edit by hand.
// family: nppsZeroCrossing


#include "npp.h"
#include "npps_statistics_executor.h"

NppStatus nppsCountInRangeGetBufferSize_32s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::count_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsCountInRange_32s_Ctx(const Npp32s * pSrc, NppSignalLength nLength, int * pCounts, Npp32s nLowerBound, Npp32s nUpperBound, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::count_in_range(pSrc, nLength, pCounts, nLowerBound, nUpperBound, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsZeroCrossingGetBufferSize_16s32f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::zc_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsZeroCrossingGetBufferSize_32f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::zc_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsZeroCrossing_16s32f_Ctx(const Npp16s * pSrc, NppSignalLength nLength, Npp32f * pValZC, NppsZCType tZCType, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::zero_crossing<Npp16s>(pSrc, nLength, pValZC, tZCType, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsZeroCrossing_32f_Ctx(const Npp32f * pSrc, NppSignalLength nLength, Npp32f * pValZC, NppsZCType tZCType, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::zero_crossing<Npp32f>(pSrc, nLength, pValZC, tZCType, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsCountInRangeGetBufferSize_32s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsCountInRangeGetBufferSize_32s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsCountInRange_32s(const Npp32s * pSrc, NppSignalLength nLength, int * pCounts, Npp32s nLowerBound, Npp32s nUpperBound, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsCountInRange_32s_Ctx(pSrc, nLength, pCounts, nLowerBound, nUpperBound, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsZeroCrossingGetBufferSize_16s32f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsZeroCrossingGetBufferSize_16s32f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsZeroCrossingGetBufferSize_32f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsZeroCrossingGetBufferSize_32f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsZeroCrossing_16s32f(const Npp16s * pSrc, NppSignalLength nLength, Npp32f * pValZC, NppsZCType tZCType, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsZeroCrossing_16s32f_Ctx(pSrc, nLength, pValZC, tZCType, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsZeroCrossing_32f(const Npp32f * pSrc, NppSignalLength nLength, Npp32f * pValZC, NppsZCType tZCType, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsZeroCrossing_32f_Ctx(pSrc, nLength, pValZC, tZCType, pDeviceBuffer, nppStreamCtx);
}
