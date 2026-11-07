// Generated API wrappers -- do not edit by hand.
// family: nppsMinMax


#include "npp.h"
#include "npps_statistics_executor.h"

NppStatus nppsMaxAbsGetBufferSize_16s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp16s>(nLength, hpBufferSize);
}

NppStatus nppsMaxAbsGetBufferSize_32s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp32s>(nLength, hpBufferSize);
}

NppStatus nppsMaxAbsIndxGetBufferSize_16s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp16s>(nLength, hpBufferSize);
}

NppStatus nppsMaxAbsIndxGetBufferSize_32s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp32s>(nLength, hpBufferSize);
}

NppStatus nppsMaxAbsIndx_16s_Ctx(const Npp16s * pSrc, NppSignalLength nLength, Npp16s * pMaxAbs, int * pIndx, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp16s, true>(pSrc, nLength, nullptr, nullptr, pMaxAbs, pIndx, false, true, true, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaxAbsIndx_32s_Ctx(const Npp32s * pSrc, NppSignalLength nLength, Npp32s * pMaxAbs, int * pIndx, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp32s, true>(pSrc, nLength, nullptr, nullptr, pMaxAbs, pIndx, false, true, true, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaxAbs_16s_Ctx(const Npp16s * pSrc, NppSignalLength nLength, Npp16s * pMaxAbs, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp16s, true>(pSrc, nLength, nullptr, nullptr, pMaxAbs, nullptr, false, true, false, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaxAbs_32s_Ctx(const Npp32s * pSrc, NppSignalLength nLength, Npp32s * pMaxAbs, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp32s, true>(pSrc, nLength, nullptr, nullptr, pMaxAbs, nullptr, false, true, false, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaxGetBufferSize_16s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp16s>(nLength, hpBufferSize);
}

NppStatus nppsMaxGetBufferSize_32f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp32f>(nLength, hpBufferSize);
}

NppStatus nppsMaxGetBufferSize_32s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp32s>(nLength, hpBufferSize);
}

NppStatus nppsMaxGetBufferSize_64f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp64f>(nLength, hpBufferSize);
}

NppStatus nppsMaxIndxGetBufferSize_16s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp16s>(nLength, hpBufferSize);
}

NppStatus nppsMaxIndxGetBufferSize_32f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp32f>(nLength, hpBufferSize);
}

NppStatus nppsMaxIndxGetBufferSize_32s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp32s>(nLength, hpBufferSize);
}

NppStatus nppsMaxIndxGetBufferSize_64f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp64f>(nLength, hpBufferSize);
}

NppStatus nppsMaxIndx_16s_Ctx(const Npp16s * pSrc, NppSignalLength nLength, Npp16s * pMax, int * pIndx, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp16s, false>(pSrc, nLength, nullptr, nullptr, pMax, pIndx, false, true, true, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaxIndx_32f_Ctx(const Npp32f * pSrc, NppSignalLength nLength, Npp32f * pMax, int * pIndx, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp32f, false>(pSrc, nLength, nullptr, nullptr, pMax, pIndx, false, true, true, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaxIndx_32s_Ctx(const Npp32s * pSrc, NppSignalLength nLength, Npp32s * pMax, int * pIndx, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp32s, false>(pSrc, nLength, nullptr, nullptr, pMax, pIndx, false, true, true, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaxIndx_64f_Ctx(const Npp64f * pSrc, NppSignalLength nLength, Npp64f * pMax, int * pIndx, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp64f, false>(pSrc, nLength, nullptr, nullptr, pMax, pIndx, false, true, true, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMax_16s_Ctx(const Npp16s * pSrc, NppSignalLength nLength, Npp16s * pMax, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp16s, false>(pSrc, nLength, nullptr, nullptr, pMax, nullptr, false, true, false, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMax_32f_Ctx(const Npp32f * pSrc, NppSignalLength nLength, Npp32f * pMax, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp32f, false>(pSrc, nLength, nullptr, nullptr, pMax, nullptr, false, true, false, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMax_32s_Ctx(const Npp32s * pSrc, NppSignalLength nLength, Npp32s * pMax, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp32s, false>(pSrc, nLength, nullptr, nullptr, pMax, nullptr, false, true, false, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMax_64f_Ctx(const Npp64f * pSrc, NppSignalLength nLength, Npp64f * pMax, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp64f, false>(pSrc, nLength, nullptr, nullptr, pMax, nullptr, false, true, false, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMinAbsGetBufferSize_16s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp16s>(nLength, hpBufferSize);
}

NppStatus nppsMinAbsGetBufferSize_32s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp32s>(nLength, hpBufferSize);
}

NppStatus nppsMinAbsIndxGetBufferSize_16s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp16s>(nLength, hpBufferSize);
}

NppStatus nppsMinAbsIndxGetBufferSize_32s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp32s>(nLength, hpBufferSize);
}

NppStatus nppsMinAbsIndx_16s_Ctx(const Npp16s * pSrc, NppSignalLength nLength, Npp16s * pMinAbs, int * pIndx, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp16s, true>(pSrc, nLength, pMinAbs, pIndx, nullptr, nullptr, true, false, true, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMinAbsIndx_32s_Ctx(const Npp32s * pSrc, NppSignalLength nLength, Npp32s * pMinAbs, int * pIndx, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp32s, true>(pSrc, nLength, pMinAbs, pIndx, nullptr, nullptr, true, false, true, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMinAbs_16s_Ctx(const Npp16s * pSrc, NppSignalLength nLength, Npp16s * pMinAbs, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp16s, true>(pSrc, nLength, pMinAbs, nullptr, nullptr, nullptr, true, false, false, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMinAbs_32s_Ctx(const Npp32s * pSrc, NppSignalLength nLength, Npp32s * pMinAbs, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp32s, true>(pSrc, nLength, pMinAbs, nullptr, nullptr, nullptr, true, false, false, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMinGetBufferSize_16s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp16s>(nLength, hpBufferSize);
}

NppStatus nppsMinGetBufferSize_32f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp32f>(nLength, hpBufferSize);
}

NppStatus nppsMinGetBufferSize_32s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp32s>(nLength, hpBufferSize);
}

NppStatus nppsMinGetBufferSize_64f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp64f>(nLength, hpBufferSize);
}

NppStatus nppsMinIndxGetBufferSize_16s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp16s>(nLength, hpBufferSize);
}

NppStatus nppsMinIndxGetBufferSize_32f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp32f>(nLength, hpBufferSize);
}

NppStatus nppsMinIndxGetBufferSize_32s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp32s>(nLength, hpBufferSize);
}

NppStatus nppsMinIndxGetBufferSize_64f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp64f>(nLength, hpBufferSize);
}

NppStatus nppsMinIndx_16s_Ctx(const Npp16s * pSrc, NppSignalLength nLength, Npp16s * pMin, int * pIndx, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp16s, false>(pSrc, nLength, pMin, pIndx, nullptr, nullptr, true, false, true, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMinIndx_32f_Ctx(const Npp32f * pSrc, NppSignalLength nLength, Npp32f * pMin, int * pIndx, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp32f, false>(pSrc, nLength, pMin, pIndx, nullptr, nullptr, true, false, true, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMinIndx_32s_Ctx(const Npp32s * pSrc, NppSignalLength nLength, Npp32s * pMin, int * pIndx, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp32s, false>(pSrc, nLength, pMin, pIndx, nullptr, nullptr, true, false, true, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMinIndx_64f_Ctx(const Npp64f * pSrc, NppSignalLength nLength, Npp64f * pMin, int * pIndx, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp64f, false>(pSrc, nLength, pMin, pIndx, nullptr, nullptr, true, false, true, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMinMaxGetBufferSize_16s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp16s>(nLength, hpBufferSize);
}

NppStatus nppsMinMaxGetBufferSize_16u_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp16u>(nLength, hpBufferSize);
}

NppStatus nppsMinMaxGetBufferSize_32f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp32f>(nLength, hpBufferSize);
}

NppStatus nppsMinMaxGetBufferSize_32s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp32s>(nLength, hpBufferSize);
}

NppStatus nppsMinMaxGetBufferSize_32u_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp32u>(nLength, hpBufferSize);
}

NppStatus nppsMinMaxGetBufferSize_64f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp64f>(nLength, hpBufferSize);
}

NppStatus nppsMinMaxGetBufferSize_8u_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp8u>(nLength, hpBufferSize);
}

NppStatus nppsMinMaxIndxGetBufferSize_16s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp16s>(nLength, hpBufferSize);
}

NppStatus nppsMinMaxIndxGetBufferSize_16u_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp16u>(nLength, hpBufferSize);
}

NppStatus nppsMinMaxIndxGetBufferSize_32f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp32f>(nLength, hpBufferSize);
}

NppStatus nppsMinMaxIndxGetBufferSize_32s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp32s>(nLength, hpBufferSize);
}

NppStatus nppsMinMaxIndxGetBufferSize_32u_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp32u>(nLength, hpBufferSize);
}

NppStatus nppsMinMaxIndxGetBufferSize_64f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp64f>(nLength, hpBufferSize);
}

NppStatus nppsMinMaxIndxGetBufferSize_8u_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::extrema_buffer_size<Npp8u>(nLength, hpBufferSize);
}

NppStatus nppsMinMaxIndx_16s_Ctx(const Npp16s * pSrc, NppSignalLength nLength, Npp16s * pMin, int * pMinIndx, Npp16s * pMax, int * pMaxIndx, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp16s, false>(pSrc, nLength, pMin, pMinIndx, pMax, pMaxIndx, true, true, true, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMinMaxIndx_16u_Ctx(const Npp16u * pSrc, NppSignalLength nLength, Npp16u * pMin, int * pMinIndx, Npp16u * pMax, int * pMaxIndx, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp16u, false>(pSrc, nLength, pMin, pMinIndx, pMax, pMaxIndx, true, true, true, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMinMaxIndx_32f_Ctx(const Npp32f * pSrc, NppSignalLength nLength, Npp32f * pMin, int * pMinIndx, Npp32f * pMax, int * pMaxIndx, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp32f, false>(pSrc, nLength, pMin, pMinIndx, pMax, pMaxIndx, true, true, true, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMinMaxIndx_32s_Ctx(const Npp32s * pSrc, NppSignalLength nLength, Npp32s * pMin, int * pMinIndx, Npp32s * pMax, int * pMaxIndx, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp32s, false>(pSrc, nLength, pMin, pMinIndx, pMax, pMaxIndx, true, true, true, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMinMaxIndx_32u_Ctx(const Npp32u * pSrc, NppSignalLength nLength, Npp32u * pMin, int * pMinIndx, Npp32u * pMax, int * pMaxIndx, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp32u, false>(pSrc, nLength, pMin, pMinIndx, pMax, pMaxIndx, true, true, true, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMinMaxIndx_64f_Ctx(const Npp64f * pSrc, NppSignalLength nLength, Npp64f * pMin, int * pMinIndx, Npp64f * pMax, int * pMaxIndx, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp64f, false>(pSrc, nLength, pMin, pMinIndx, pMax, pMaxIndx, true, true, true, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMinMaxIndx_8u_Ctx(const Npp8u * pSrc, NppSignalLength nLength, Npp8u * pMin, int * pMinIndx, Npp8u * pMax, int * pMaxIndx, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp8u, false>(pSrc, nLength, pMin, pMinIndx, pMax, pMaxIndx, true, true, true, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMinMax_16s_Ctx(const Npp16s * pSrc, NppSignalLength nLength, Npp16s * pMin, Npp16s * pMax, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp16s, false>(pSrc, nLength, pMin, nullptr, pMax, nullptr, true, true, false, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMinMax_16u_Ctx(const Npp16u * pSrc, NppSignalLength nLength, Npp16u * pMin, Npp16u * pMax, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp16u, false>(pSrc, nLength, pMin, nullptr, pMax, nullptr, true, true, false, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMinMax_32f_Ctx(const Npp32f * pSrc, NppSignalLength nLength, Npp32f * pMin, Npp32f * pMax, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp32f, false>(pSrc, nLength, pMin, nullptr, pMax, nullptr, true, true, false, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMinMax_32s_Ctx(const Npp32s * pSrc, NppSignalLength nLength, Npp32s * pMin, Npp32s * pMax, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp32s, false>(pSrc, nLength, pMin, nullptr, pMax, nullptr, true, true, false, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMinMax_32u_Ctx(const Npp32u * pSrc, NppSignalLength nLength, Npp32u * pMin, Npp32u * pMax, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp32u, false>(pSrc, nLength, pMin, nullptr, pMax, nullptr, true, true, false, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMinMax_64f_Ctx(const Npp64f * pSrc, NppSignalLength nLength, Npp64f * pMin, Npp64f * pMax, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp64f, false>(pSrc, nLength, pMin, nullptr, pMax, nullptr, true, true, false, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMinMax_8u_Ctx(const Npp8u * pSrc, NppSignalLength nLength, Npp8u * pMin, Npp8u * pMax, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp8u, false>(pSrc, nLength, pMin, nullptr, pMax, nullptr, true, true, false, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMin_16s_Ctx(const Npp16s * pSrc, NppSignalLength nLength, Npp16s * pMin, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp16s, false>(pSrc, nLength, pMin, nullptr, nullptr, nullptr, true, false, false, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMin_32f_Ctx(const Npp32f * pSrc, NppSignalLength nLength, Npp32f * pMin, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp32f, false>(pSrc, nLength, pMin, nullptr, nullptr, nullptr, true, false, false, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMin_32s_Ctx(const Npp32s * pSrc, NppSignalLength nLength, Npp32s * pMin, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp32s, false>(pSrc, nLength, pMin, nullptr, nullptr, nullptr, true, false, false, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMin_64f_Ctx(const Npp64f * pSrc, NppSignalLength nLength, Npp64f * pMin, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::extrema<Npp64f, false>(pSrc, nLength, pMin, nullptr, nullptr, nullptr, true, false, false, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaxAbsGetBufferSize_16s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaxAbsGetBufferSize_16s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaxAbsGetBufferSize_32s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaxAbsGetBufferSize_32s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaxAbsIndxGetBufferSize_16s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaxAbsIndxGetBufferSize_16s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaxAbsIndxGetBufferSize_32s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaxAbsIndxGetBufferSize_32s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaxAbsIndx_16s(const Npp16s * pSrc, NppSignalLength nLength, Npp16s * pMaxAbs, int * pIndx, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaxAbsIndx_16s_Ctx(pSrc, nLength, pMaxAbs, pIndx, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaxAbsIndx_32s(const Npp32s * pSrc, NppSignalLength nLength, Npp32s * pMaxAbs, int * pIndx, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaxAbsIndx_32s_Ctx(pSrc, nLength, pMaxAbs, pIndx, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaxAbs_16s(const Npp16s * pSrc, NppSignalLength nLength, Npp16s * pMaxAbs, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaxAbs_16s_Ctx(pSrc, nLength, pMaxAbs, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaxAbs_32s(const Npp32s * pSrc, NppSignalLength nLength, Npp32s * pMaxAbs, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaxAbs_32s_Ctx(pSrc, nLength, pMaxAbs, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaxGetBufferSize_16s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaxGetBufferSize_16s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaxGetBufferSize_32f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaxGetBufferSize_32f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaxGetBufferSize_32s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaxGetBufferSize_32s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaxGetBufferSize_64f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaxGetBufferSize_64f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaxIndxGetBufferSize_16s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaxIndxGetBufferSize_16s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaxIndxGetBufferSize_32f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaxIndxGetBufferSize_32f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaxIndxGetBufferSize_32s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaxIndxGetBufferSize_32s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaxIndxGetBufferSize_64f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaxIndxGetBufferSize_64f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaxIndx_16s(const Npp16s * pSrc, NppSignalLength nLength, Npp16s * pMax, int * pIndx, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaxIndx_16s_Ctx(pSrc, nLength, pMax, pIndx, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaxIndx_32f(const Npp32f * pSrc, NppSignalLength nLength, Npp32f * pMax, int * pIndx, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaxIndx_32f_Ctx(pSrc, nLength, pMax, pIndx, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaxIndx_32s(const Npp32s * pSrc, NppSignalLength nLength, Npp32s * pMax, int * pIndx, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaxIndx_32s_Ctx(pSrc, nLength, pMax, pIndx, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaxIndx_64f(const Npp64f * pSrc, NppSignalLength nLength, Npp64f * pMax, int * pIndx, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaxIndx_64f_Ctx(pSrc, nLength, pMax, pIndx, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMax_16s(const Npp16s * pSrc, NppSignalLength nLength, Npp16s * pMax, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMax_16s_Ctx(pSrc, nLength, pMax, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMax_32f(const Npp32f * pSrc, NppSignalLength nLength, Npp32f * pMax, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMax_32f_Ctx(pSrc, nLength, pMax, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMax_32s(const Npp32s * pSrc, NppSignalLength nLength, Npp32s * pMax, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMax_32s_Ctx(pSrc, nLength, pMax, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMax_64f(const Npp64f * pSrc, NppSignalLength nLength, Npp64f * pMax, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMax_64f_Ctx(pSrc, nLength, pMax, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMinAbsGetBufferSize_16s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinAbsGetBufferSize_16s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMinAbsGetBufferSize_32s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinAbsGetBufferSize_32s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMinAbsIndxGetBufferSize_16s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinAbsIndxGetBufferSize_16s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMinAbsIndxGetBufferSize_32s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinAbsIndxGetBufferSize_32s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMinAbsIndx_16s(const Npp16s * pSrc, NppSignalLength nLength, Npp16s * pMinAbs, int * pIndx, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinAbsIndx_16s_Ctx(pSrc, nLength, pMinAbs, pIndx, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMinAbsIndx_32s(const Npp32s * pSrc, NppSignalLength nLength, Npp32s * pMinAbs, int * pIndx, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinAbsIndx_32s_Ctx(pSrc, nLength, pMinAbs, pIndx, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMinAbs_16s(const Npp16s * pSrc, NppSignalLength nLength, Npp16s * pMinAbs, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinAbs_16s_Ctx(pSrc, nLength, pMinAbs, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMinAbs_32s(const Npp32s * pSrc, NppSignalLength nLength, Npp32s * pMinAbs, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinAbs_32s_Ctx(pSrc, nLength, pMinAbs, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMinGetBufferSize_16s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinGetBufferSize_16s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMinGetBufferSize_32f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinGetBufferSize_32f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMinGetBufferSize_32s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinGetBufferSize_32s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMinGetBufferSize_64f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinGetBufferSize_64f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMinIndxGetBufferSize_16s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinIndxGetBufferSize_16s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMinIndxGetBufferSize_32f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinIndxGetBufferSize_32f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMinIndxGetBufferSize_32s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinIndxGetBufferSize_32s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMinIndxGetBufferSize_64f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinIndxGetBufferSize_64f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMinIndx_16s(const Npp16s * pSrc, NppSignalLength nLength, Npp16s * pMin, int * pIndx, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinIndx_16s_Ctx(pSrc, nLength, pMin, pIndx, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMinIndx_32f(const Npp32f * pSrc, NppSignalLength nLength, Npp32f * pMin, int * pIndx, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinIndx_32f_Ctx(pSrc, nLength, pMin, pIndx, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMinIndx_32s(const Npp32s * pSrc, NppSignalLength nLength, Npp32s * pMin, int * pIndx, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinIndx_32s_Ctx(pSrc, nLength, pMin, pIndx, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMinIndx_64f(const Npp64f * pSrc, NppSignalLength nLength, Npp64f * pMin, int * pIndx, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinIndx_64f_Ctx(pSrc, nLength, pMin, pIndx, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMinMaxGetBufferSize_16s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinMaxGetBufferSize_16s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMinMaxGetBufferSize_16u(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinMaxGetBufferSize_16u_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMinMaxGetBufferSize_32f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinMaxGetBufferSize_32f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMinMaxGetBufferSize_32s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinMaxGetBufferSize_32s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMinMaxGetBufferSize_32u(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinMaxGetBufferSize_32u_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMinMaxGetBufferSize_64f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinMaxGetBufferSize_64f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMinMaxGetBufferSize_8u(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinMaxGetBufferSize_8u_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMinMaxIndxGetBufferSize_16s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinMaxIndxGetBufferSize_16s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMinMaxIndxGetBufferSize_16u(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinMaxIndxGetBufferSize_16u_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMinMaxIndxGetBufferSize_32f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinMaxIndxGetBufferSize_32f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMinMaxIndxGetBufferSize_32s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinMaxIndxGetBufferSize_32s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMinMaxIndxGetBufferSize_32u(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinMaxIndxGetBufferSize_32u_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMinMaxIndxGetBufferSize_64f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinMaxIndxGetBufferSize_64f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMinMaxIndxGetBufferSize_8u(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinMaxIndxGetBufferSize_8u_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMinMaxIndx_16s(const Npp16s * pSrc, NppSignalLength nLength, Npp16s * pMin, int * pMinIndx, Npp16s * pMax, int * pMaxIndx, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinMaxIndx_16s_Ctx(pSrc, nLength, pMin, pMinIndx, pMax, pMaxIndx, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMinMaxIndx_16u(const Npp16u * pSrc, NppSignalLength nLength, Npp16u * pMin, int * pMinIndx, Npp16u * pMax, int * pMaxIndx, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinMaxIndx_16u_Ctx(pSrc, nLength, pMin, pMinIndx, pMax, pMaxIndx, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMinMaxIndx_32f(const Npp32f * pSrc, NppSignalLength nLength, Npp32f * pMin, int * pMinIndx, Npp32f * pMax, int * pMaxIndx, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinMaxIndx_32f_Ctx(pSrc, nLength, pMin, pMinIndx, pMax, pMaxIndx, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMinMaxIndx_32s(const Npp32s * pSrc, NppSignalLength nLength, Npp32s * pMin, int * pMinIndx, Npp32s * pMax, int * pMaxIndx, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinMaxIndx_32s_Ctx(pSrc, nLength, pMin, pMinIndx, pMax, pMaxIndx, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMinMaxIndx_32u(const Npp32u * pSrc, NppSignalLength nLength, Npp32u * pMin, int * pMinIndx, Npp32u * pMax, int * pMaxIndx, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinMaxIndx_32u_Ctx(pSrc, nLength, pMin, pMinIndx, pMax, pMaxIndx, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMinMaxIndx_64f(const Npp64f * pSrc, NppSignalLength nLength, Npp64f * pMin, int * pMinIndx, Npp64f * pMax, int * pMaxIndx, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinMaxIndx_64f_Ctx(pSrc, nLength, pMin, pMinIndx, pMax, pMaxIndx, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMinMaxIndx_8u(const Npp8u * pSrc, NppSignalLength nLength, Npp8u * pMin, int * pMinIndx, Npp8u * pMax, int * pMaxIndx, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinMaxIndx_8u_Ctx(pSrc, nLength, pMin, pMinIndx, pMax, pMaxIndx, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMinMax_16s(const Npp16s * pSrc, NppSignalLength nLength, Npp16s * pMin, Npp16s * pMax, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinMax_16s_Ctx(pSrc, nLength, pMin, pMax, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMinMax_16u(const Npp16u * pSrc, NppSignalLength nLength, Npp16u * pMin, Npp16u * pMax, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinMax_16u_Ctx(pSrc, nLength, pMin, pMax, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMinMax_32f(const Npp32f * pSrc, NppSignalLength nLength, Npp32f * pMin, Npp32f * pMax, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinMax_32f_Ctx(pSrc, nLength, pMin, pMax, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMinMax_32s(const Npp32s * pSrc, NppSignalLength nLength, Npp32s * pMin, Npp32s * pMax, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinMax_32s_Ctx(pSrc, nLength, pMin, pMax, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMinMax_32u(const Npp32u * pSrc, NppSignalLength nLength, Npp32u * pMin, Npp32u * pMax, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinMax_32u_Ctx(pSrc, nLength, pMin, pMax, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMinMax_64f(const Npp64f * pSrc, NppSignalLength nLength, Npp64f * pMin, Npp64f * pMax, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinMax_64f_Ctx(pSrc, nLength, pMin, pMax, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMinMax_8u(const Npp8u * pSrc, NppSignalLength nLength, Npp8u * pMin, Npp8u * pMax, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMinMax_8u_Ctx(pSrc, nLength, pMin, pMax, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMin_16s(const Npp16s * pSrc, NppSignalLength nLength, Npp16s * pMin, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMin_16s_Ctx(pSrc, nLength, pMin, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMin_32f(const Npp32f * pSrc, NppSignalLength nLength, Npp32f * pMin, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMin_32f_Ctx(pSrc, nLength, pMin, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMin_32s(const Npp32s * pSrc, NppSignalLength nLength, Npp32s * pMin, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMin_32s_Ctx(pSrc, nLength, pMin, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMin_64f(const Npp64f * pSrc, NppSignalLength nLength, Npp64f * pMin, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMin_64f_Ctx(pSrc, nLength, pMin, pDeviceBuffer, nppStreamCtx);
}
