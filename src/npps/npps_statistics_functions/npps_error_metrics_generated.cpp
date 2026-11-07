// Generated API wrappers -- do not edit by hand.
// family: nppsMaximumError


#include "npp.h"
#include "npps_statistics_executor.h"

NppStatus nppsAverageErrorGetBufferSize_16s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsAverageErrorGetBufferSize_16sc_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsAverageErrorGetBufferSize_16u_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsAverageErrorGetBufferSize_32f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsAverageErrorGetBufferSize_32fc_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsAverageErrorGetBufferSize_32s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsAverageErrorGetBufferSize_32sc_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsAverageErrorGetBufferSize_32u_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsAverageErrorGetBufferSize_64f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsAverageErrorGetBufferSize_64fc_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsAverageErrorGetBufferSize_64s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsAverageErrorGetBufferSize_64sc_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsAverageErrorGetBufferSize_8s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsAverageErrorGetBufferSize_8u_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsAverageError_16s_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp16s, npps::statistics::ErrKind::AvgAbs>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsAverageError_16sc_Ctx(const Npp16sc * pSrc1, const Npp16sc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp16sc, npps::statistics::ErrKind::AvgAbs>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsAverageError_16u_Ctx(const Npp16u * pSrc1, const Npp16u * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp16u, npps::statistics::ErrKind::AvgAbs>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsAverageError_32f_Ctx(const Npp32f * pSrc1, const Npp32f * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp32f, npps::statistics::ErrKind::AvgAbs>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsAverageError_32fc_Ctx(const Npp32fc * pSrc1, const Npp32fc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp32fc, npps::statistics::ErrKind::AvgAbs>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsAverageError_32s_Ctx(const Npp32s * pSrc1, const Npp32s * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp32s, npps::statistics::ErrKind::AvgAbs>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsAverageError_32sc_Ctx(const Npp32sc * pSrc1, const Npp32sc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp32sc, npps::statistics::ErrKind::AvgAbs>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsAverageError_32u_Ctx(const Npp32u * pSrc1, const Npp32u * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp32u, npps::statistics::ErrKind::AvgAbs>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsAverageError_64f_Ctx(const Npp64f * pSrc1, const Npp64f * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp64f, npps::statistics::ErrKind::AvgAbs>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsAverageError_64fc_Ctx(const Npp64fc * pSrc1, const Npp64fc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp64fc, npps::statistics::ErrKind::AvgAbs>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsAverageError_64s_Ctx(const Npp64s * pSrc1, const Npp64s * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp64s, npps::statistics::ErrKind::AvgAbs>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsAverageError_64sc_Ctx(const Npp64sc * pSrc1, const Npp64sc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp64sc, npps::statistics::ErrKind::AvgAbs>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsAverageError_8s_Ctx(const Npp8s * pSrc1, const Npp8s * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp8s, npps::statistics::ErrKind::AvgAbs>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsAverageError_8u_Ctx(const Npp8u * pSrc1, const Npp8u * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp8u, npps::statistics::ErrKind::AvgAbs>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsAverageRelativeErrorGetBufferSize_16s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsAverageRelativeErrorGetBufferSize_16sc_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsAverageRelativeErrorGetBufferSize_16u_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsAverageRelativeErrorGetBufferSize_32f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsAverageRelativeErrorGetBufferSize_32fc_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsAverageRelativeErrorGetBufferSize_32s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsAverageRelativeErrorGetBufferSize_32sc_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsAverageRelativeErrorGetBufferSize_32u_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsAverageRelativeErrorGetBufferSize_64f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsAverageRelativeErrorGetBufferSize_64fc_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsAverageRelativeErrorGetBufferSize_64s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsAverageRelativeErrorGetBufferSize_64sc_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsAverageRelativeErrorGetBufferSize_8s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsAverageRelativeErrorGetBufferSize_8u_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsAverageRelativeError_16s_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp16s, npps::statistics::ErrKind::AvgRel>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsAverageRelativeError_16sc_Ctx(const Npp16sc * pSrc1, const Npp16sc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp16sc, npps::statistics::ErrKind::AvgRel>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsAverageRelativeError_16u_Ctx(const Npp16u * pSrc1, const Npp16u * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp16u, npps::statistics::ErrKind::AvgRel>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsAverageRelativeError_32f_Ctx(const Npp32f * pSrc1, const Npp32f * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp32f, npps::statistics::ErrKind::AvgRel>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsAverageRelativeError_32fc_Ctx(const Npp32fc * pSrc1, const Npp32fc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp32fc, npps::statistics::ErrKind::AvgRel>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsAverageRelativeError_32s_Ctx(const Npp32s * pSrc1, const Npp32s * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp32s, npps::statistics::ErrKind::AvgRel>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsAverageRelativeError_32sc_Ctx(const Npp32sc * pSrc1, const Npp32sc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp32sc, npps::statistics::ErrKind::AvgRel>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsAverageRelativeError_32u_Ctx(const Npp32u * pSrc1, const Npp32u * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp32u, npps::statistics::ErrKind::AvgRel>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsAverageRelativeError_64f_Ctx(const Npp64f * pSrc1, const Npp64f * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp64f, npps::statistics::ErrKind::AvgRel>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsAverageRelativeError_64fc_Ctx(const Npp64fc * pSrc1, const Npp64fc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp64fc, npps::statistics::ErrKind::AvgRel>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsAverageRelativeError_64s_Ctx(const Npp64s * pSrc1, const Npp64s * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp64s, npps::statistics::ErrKind::AvgRel>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsAverageRelativeError_64sc_Ctx(const Npp64sc * pSrc1, const Npp64sc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp64sc, npps::statistics::ErrKind::AvgRel>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsAverageRelativeError_8s_Ctx(const Npp8s * pSrc1, const Npp8s * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp8s, npps::statistics::ErrKind::AvgRel>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsAverageRelativeError_8u_Ctx(const Npp8u * pSrc1, const Npp8u * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp8u, npps::statistics::ErrKind::AvgRel>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaximumErrorGetBufferSize_16s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsMaximumErrorGetBufferSize_16sc_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsMaximumErrorGetBufferSize_16u_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsMaximumErrorGetBufferSize_32f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsMaximumErrorGetBufferSize_32fc_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsMaximumErrorGetBufferSize_32s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsMaximumErrorGetBufferSize_32sc_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsMaximumErrorGetBufferSize_32u_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsMaximumErrorGetBufferSize_64f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsMaximumErrorGetBufferSize_64fc_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsMaximumErrorGetBufferSize_64s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsMaximumErrorGetBufferSize_64sc_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsMaximumErrorGetBufferSize_8s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsMaximumErrorGetBufferSize_8u_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsMaximumError_16s_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp16s, npps::statistics::ErrKind::MaxAbs>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaximumError_16sc_Ctx(const Npp16sc * pSrc1, const Npp16sc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp16sc, npps::statistics::ErrKind::MaxAbs>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaximumError_16u_Ctx(const Npp16u * pSrc1, const Npp16u * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp16u, npps::statistics::ErrKind::MaxAbs>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaximumError_32f_Ctx(const Npp32f * pSrc1, const Npp32f * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp32f, npps::statistics::ErrKind::MaxAbs>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaximumError_32fc_Ctx(const Npp32fc * pSrc1, const Npp32fc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp32fc, npps::statistics::ErrKind::MaxAbs>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaximumError_32s_Ctx(const Npp32s * pSrc1, const Npp32s * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp32s, npps::statistics::ErrKind::MaxAbs>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaximumError_32sc_Ctx(const Npp32sc * pSrc1, const Npp32sc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp32sc, npps::statistics::ErrKind::MaxAbs>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaximumError_32u_Ctx(const Npp32u * pSrc1, const Npp32u * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp32u, npps::statistics::ErrKind::MaxAbs>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaximumError_64f_Ctx(const Npp64f * pSrc1, const Npp64f * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp64f, npps::statistics::ErrKind::MaxAbs>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaximumError_64fc_Ctx(const Npp64fc * pSrc1, const Npp64fc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp64fc, npps::statistics::ErrKind::MaxAbs>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaximumError_64s_Ctx(const Npp64s * pSrc1, const Npp64s * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp64s, npps::statistics::ErrKind::MaxAbs>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaximumError_64sc_Ctx(const Npp64sc * pSrc1, const Npp64sc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp64sc, npps::statistics::ErrKind::MaxAbs>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaximumError_8s_Ctx(const Npp8s * pSrc1, const Npp8s * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp8s, npps::statistics::ErrKind::MaxAbs>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaximumError_8u_Ctx(const Npp8u * pSrc1, const Npp8u * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp8u, npps::statistics::ErrKind::MaxAbs>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaximumRelativeErrorGetBufferSize_16s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsMaximumRelativeErrorGetBufferSize_16sc_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsMaximumRelativeErrorGetBufferSize_16u_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsMaximumRelativeErrorGetBufferSize_32f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsMaximumRelativeErrorGetBufferSize_32fc_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsMaximumRelativeErrorGetBufferSize_32s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsMaximumRelativeErrorGetBufferSize_32sc_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsMaximumRelativeErrorGetBufferSize_32u_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsMaximumRelativeErrorGetBufferSize_64f_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsMaximumRelativeErrorGetBufferSize_64fc_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsMaximumRelativeErrorGetBufferSize_64s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsMaximumRelativeErrorGetBufferSize_64sc_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsMaximumRelativeErrorGetBufferSize_8s_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsMaximumRelativeErrorGetBufferSize_8u_Ctx(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return npps::statistics::err_buffer_size(nLength, hpBufferSize);
}

NppStatus nppsMaximumRelativeError_16s_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp16s, npps::statistics::ErrKind::MaxRel>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaximumRelativeError_16sc_Ctx(const Npp16sc * pSrc1, const Npp16sc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp16sc, npps::statistics::ErrKind::MaxRel>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaximumRelativeError_16u_Ctx(const Npp16u * pSrc1, const Npp16u * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp16u, npps::statistics::ErrKind::MaxRel>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaximumRelativeError_32f_Ctx(const Npp32f * pSrc1, const Npp32f * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp32f, npps::statistics::ErrKind::MaxRel>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaximumRelativeError_32fc_Ctx(const Npp32fc * pSrc1, const Npp32fc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp32fc, npps::statistics::ErrKind::MaxRel>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaximumRelativeError_32s_Ctx(const Npp32s * pSrc1, const Npp32s * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp32s, npps::statistics::ErrKind::MaxRel>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaximumRelativeError_32sc_Ctx(const Npp32sc * pSrc1, const Npp32sc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp32sc, npps::statistics::ErrKind::MaxRel>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaximumRelativeError_32u_Ctx(const Npp32u * pSrc1, const Npp32u * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp32u, npps::statistics::ErrKind::MaxRel>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaximumRelativeError_64f_Ctx(const Npp64f * pSrc1, const Npp64f * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp64f, npps::statistics::ErrKind::MaxRel>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaximumRelativeError_64fc_Ctx(const Npp64fc * pSrc1, const Npp64fc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp64fc, npps::statistics::ErrKind::MaxRel>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaximumRelativeError_64s_Ctx(const Npp64s * pSrc1, const Npp64s * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp64s, npps::statistics::ErrKind::MaxRel>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaximumRelativeError_64sc_Ctx(const Npp64sc * pSrc1, const Npp64sc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp64sc, npps::statistics::ErrKind::MaxRel>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaximumRelativeError_8s_Ctx(const Npp8s * pSrc1, const Npp8s * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp8s, npps::statistics::ErrKind::MaxRel>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsMaximumRelativeError_8u_Ctx(const Npp8u * pSrc1, const Npp8u * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return npps::statistics::error_metric<Npp8u, npps::statistics::ErrKind::MaxRel>(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsAverageErrorGetBufferSize_16s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageErrorGetBufferSize_16s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsAverageErrorGetBufferSize_16sc(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageErrorGetBufferSize_16sc_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsAverageErrorGetBufferSize_16u(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageErrorGetBufferSize_16u_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsAverageErrorGetBufferSize_32f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageErrorGetBufferSize_32f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsAverageErrorGetBufferSize_32fc(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageErrorGetBufferSize_32fc_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsAverageErrorGetBufferSize_32s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageErrorGetBufferSize_32s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsAverageErrorGetBufferSize_32sc(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageErrorGetBufferSize_32sc_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsAverageErrorGetBufferSize_32u(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageErrorGetBufferSize_32u_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsAverageErrorGetBufferSize_64f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageErrorGetBufferSize_64f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsAverageErrorGetBufferSize_64fc(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageErrorGetBufferSize_64fc_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsAverageErrorGetBufferSize_64s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageErrorGetBufferSize_64s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsAverageErrorGetBufferSize_64sc(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageErrorGetBufferSize_64sc_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsAverageErrorGetBufferSize_8s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageErrorGetBufferSize_8s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsAverageErrorGetBufferSize_8u(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageErrorGetBufferSize_8u_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsAverageError_16s(const Npp16s * pSrc1, const Npp16s * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageError_16s_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsAverageError_16sc(const Npp16sc * pSrc1, const Npp16sc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageError_16sc_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsAverageError_16u(const Npp16u * pSrc1, const Npp16u * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageError_16u_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsAverageError_32f(const Npp32f * pSrc1, const Npp32f * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageError_32f_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsAverageError_32fc(const Npp32fc * pSrc1, const Npp32fc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageError_32fc_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsAverageError_32s(const Npp32s * pSrc1, const Npp32s * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageError_32s_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsAverageError_32sc(const Npp32sc * pSrc1, const Npp32sc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageError_32sc_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsAverageError_32u(const Npp32u * pSrc1, const Npp32u * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageError_32u_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsAverageError_64f(const Npp64f * pSrc1, const Npp64f * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageError_64f_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsAverageError_64fc(const Npp64fc * pSrc1, const Npp64fc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageError_64fc_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsAverageError_64s(const Npp64s * pSrc1, const Npp64s * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageError_64s_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsAverageError_64sc(const Npp64sc * pSrc1, const Npp64sc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageError_64sc_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsAverageError_8s(const Npp8s * pSrc1, const Npp8s * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageError_8s_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsAverageError_8u(const Npp8u * pSrc1, const Npp8u * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageError_8u_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsAverageRelativeErrorGetBufferSize_16s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageRelativeErrorGetBufferSize_16s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsAverageRelativeErrorGetBufferSize_16sc(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageRelativeErrorGetBufferSize_16sc_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsAverageRelativeErrorGetBufferSize_16u(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageRelativeErrorGetBufferSize_16u_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsAverageRelativeErrorGetBufferSize_32f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageRelativeErrorGetBufferSize_32f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsAverageRelativeErrorGetBufferSize_32fc(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageRelativeErrorGetBufferSize_32fc_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsAverageRelativeErrorGetBufferSize_32s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageRelativeErrorGetBufferSize_32s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsAverageRelativeErrorGetBufferSize_32sc(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageRelativeErrorGetBufferSize_32sc_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsAverageRelativeErrorGetBufferSize_32u(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageRelativeErrorGetBufferSize_32u_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsAverageRelativeErrorGetBufferSize_64f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageRelativeErrorGetBufferSize_64f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsAverageRelativeErrorGetBufferSize_64fc(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageRelativeErrorGetBufferSize_64fc_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsAverageRelativeErrorGetBufferSize_64s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageRelativeErrorGetBufferSize_64s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsAverageRelativeErrorGetBufferSize_64sc(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageRelativeErrorGetBufferSize_64sc_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsAverageRelativeErrorGetBufferSize_8s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageRelativeErrorGetBufferSize_8s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsAverageRelativeErrorGetBufferSize_8u(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageRelativeErrorGetBufferSize_8u_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsAverageRelativeError_16s(const Npp16s * pSrc1, const Npp16s * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageRelativeError_16s_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsAverageRelativeError_16sc(const Npp16sc * pSrc1, const Npp16sc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageRelativeError_16sc_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsAverageRelativeError_16u(const Npp16u * pSrc1, const Npp16u * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageRelativeError_16u_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsAverageRelativeError_32f(const Npp32f * pSrc1, const Npp32f * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageRelativeError_32f_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsAverageRelativeError_32fc(const Npp32fc * pSrc1, const Npp32fc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageRelativeError_32fc_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsAverageRelativeError_32s(const Npp32s * pSrc1, const Npp32s * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageRelativeError_32s_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsAverageRelativeError_32sc(const Npp32sc * pSrc1, const Npp32sc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageRelativeError_32sc_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsAverageRelativeError_32u(const Npp32u * pSrc1, const Npp32u * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageRelativeError_32u_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsAverageRelativeError_64f(const Npp64f * pSrc1, const Npp64f * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageRelativeError_64f_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsAverageRelativeError_64fc(const Npp64fc * pSrc1, const Npp64fc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageRelativeError_64fc_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsAverageRelativeError_64s(const Npp64s * pSrc1, const Npp64s * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageRelativeError_64s_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsAverageRelativeError_64sc(const Npp64sc * pSrc1, const Npp64sc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageRelativeError_64sc_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsAverageRelativeError_8s(const Npp8s * pSrc1, const Npp8s * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageRelativeError_8s_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsAverageRelativeError_8u(const Npp8u * pSrc1, const Npp8u * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAverageRelativeError_8u_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaximumErrorGetBufferSize_16s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumErrorGetBufferSize_16s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaximumErrorGetBufferSize_16sc(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumErrorGetBufferSize_16sc_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaximumErrorGetBufferSize_16u(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumErrorGetBufferSize_16u_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaximumErrorGetBufferSize_32f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumErrorGetBufferSize_32f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaximumErrorGetBufferSize_32fc(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumErrorGetBufferSize_32fc_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaximumErrorGetBufferSize_32s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumErrorGetBufferSize_32s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaximumErrorGetBufferSize_32sc(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumErrorGetBufferSize_32sc_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaximumErrorGetBufferSize_32u(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumErrorGetBufferSize_32u_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaximumErrorGetBufferSize_64f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumErrorGetBufferSize_64f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaximumErrorGetBufferSize_64fc(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumErrorGetBufferSize_64fc_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaximumErrorGetBufferSize_64s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumErrorGetBufferSize_64s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaximumErrorGetBufferSize_64sc(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumErrorGetBufferSize_64sc_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaximumErrorGetBufferSize_8s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumErrorGetBufferSize_8s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaximumErrorGetBufferSize_8u(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumErrorGetBufferSize_8u_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaximumError_16s(const Npp16s * pSrc1, const Npp16s * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumError_16s_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaximumError_16sc(const Npp16sc * pSrc1, const Npp16sc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumError_16sc_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaximumError_16u(const Npp16u * pSrc1, const Npp16u * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumError_16u_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaximumError_32f(const Npp32f * pSrc1, const Npp32f * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumError_32f_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaximumError_32fc(const Npp32fc * pSrc1, const Npp32fc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumError_32fc_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaximumError_32s(const Npp32s * pSrc1, const Npp32s * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumError_32s_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaximumError_32sc(const Npp32sc * pSrc1, const Npp32sc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumError_32sc_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaximumError_32u(const Npp32u * pSrc1, const Npp32u * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumError_32u_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaximumError_64f(const Npp64f * pSrc1, const Npp64f * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumError_64f_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaximumError_64fc(const Npp64fc * pSrc1, const Npp64fc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumError_64fc_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaximumError_64s(const Npp64s * pSrc1, const Npp64s * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumError_64s_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaximumError_64sc(const Npp64sc * pSrc1, const Npp64sc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumError_64sc_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaximumError_8s(const Npp8s * pSrc1, const Npp8s * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumError_8s_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaximumError_8u(const Npp8u * pSrc1, const Npp8u * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumError_8u_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaximumRelativeErrorGetBufferSize_16s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumRelativeErrorGetBufferSize_16s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaximumRelativeErrorGetBufferSize_16sc(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumRelativeErrorGetBufferSize_16sc_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaximumRelativeErrorGetBufferSize_16u(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumRelativeErrorGetBufferSize_16u_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaximumRelativeErrorGetBufferSize_32f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumRelativeErrorGetBufferSize_32f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaximumRelativeErrorGetBufferSize_32fc(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumRelativeErrorGetBufferSize_32fc_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaximumRelativeErrorGetBufferSize_32s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumRelativeErrorGetBufferSize_32s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaximumRelativeErrorGetBufferSize_32sc(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumRelativeErrorGetBufferSize_32sc_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaximumRelativeErrorGetBufferSize_32u(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumRelativeErrorGetBufferSize_32u_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaximumRelativeErrorGetBufferSize_64f(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumRelativeErrorGetBufferSize_64f_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaximumRelativeErrorGetBufferSize_64fc(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumRelativeErrorGetBufferSize_64fc_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaximumRelativeErrorGetBufferSize_64s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumRelativeErrorGetBufferSize_64s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaximumRelativeErrorGetBufferSize_64sc(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumRelativeErrorGetBufferSize_64sc_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaximumRelativeErrorGetBufferSize_8s(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumRelativeErrorGetBufferSize_8s_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaximumRelativeErrorGetBufferSize_8u(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE * hpBufferSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumRelativeErrorGetBufferSize_8u_Ctx(nLength, hpBufferSize, nppStreamCtx);
}

NppStatus nppsMaximumRelativeError_16s(const Npp16s * pSrc1, const Npp16s * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumRelativeError_16s_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaximumRelativeError_16sc(const Npp16sc * pSrc1, const Npp16sc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumRelativeError_16sc_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaximumRelativeError_16u(const Npp16u * pSrc1, const Npp16u * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumRelativeError_16u_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaximumRelativeError_32f(const Npp32f * pSrc1, const Npp32f * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumRelativeError_32f_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaximumRelativeError_32fc(const Npp32fc * pSrc1, const Npp32fc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumRelativeError_32fc_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaximumRelativeError_32s(const Npp32s * pSrc1, const Npp32s * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumRelativeError_32s_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaximumRelativeError_32sc(const Npp32sc * pSrc1, const Npp32sc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumRelativeError_32sc_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaximumRelativeError_32u(const Npp32u * pSrc1, const Npp32u * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumRelativeError_32u_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaximumRelativeError_64f(const Npp64f * pSrc1, const Npp64f * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumRelativeError_64f_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaximumRelativeError_64fc(const Npp64fc * pSrc1, const Npp64fc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumRelativeError_64fc_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaximumRelativeError_64s(const Npp64s * pSrc1, const Npp64s * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumRelativeError_64s_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaximumRelativeError_64sc(const Npp64sc * pSrc1, const Npp64sc * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumRelativeError_64sc_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaximumRelativeError_8s(const Npp8s * pSrc1, const Npp8s * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumRelativeError_8s_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppsMaximumRelativeError_8u(const Npp8u * pSrc1, const Npp8u * pSrc2, NppSignalLength nLength, Npp64f * pDst, Npp8u * pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsMaximumRelativeError_8u_Ctx(pSrc1, pSrc2, nLength, pDst, pDeviceBuffer, nppStreamCtx);
}
