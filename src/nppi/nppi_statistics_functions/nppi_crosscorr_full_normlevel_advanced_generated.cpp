// Generated API wrappers -- do not edit by hand.
// family: nppiCrossCorrFull_NormLevelAdvanced


#include "npp.h"
#include "nppi_crosscorr_executor.h"

NppStatus nppiCrossCorrFull_NormLevelAdvanced_16u32f_C1R_Ctx(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp16u * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep, Npp8u * pDeviceBuffer, Npp8u * pAdvancedScratchBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp16u,Npp32f,1,0,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::NormLevelAdvanced>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, pDeviceBuffer, pAdvancedScratchBuffer, nppStreamCtx.hStream);
}

NppStatus nppiCrossCorrFull_NormLevelAdvanced_16u32f_C3R_Ctx(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp16u * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep, Npp8u * pDeviceBuffer, Npp8u * pAdvancedScratchBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp16u,Npp32f,3,0,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::NormLevelAdvanced>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, pDeviceBuffer, pAdvancedScratchBuffer, nppStreamCtx.hStream);
}

NppStatus nppiCrossCorrFull_NormLevelAdvanced_16u32f_C4R_Ctx(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp16u * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep, Npp8u * pDeviceBuffer, Npp8u * pAdvancedScratchBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp16u,Npp32f,4,0,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::NormLevelAdvanced>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, pDeviceBuffer, pAdvancedScratchBuffer, nppStreamCtx.hStream);
}

NppStatus nppiCrossCorrFull_NormLevelAdvanced_32f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp32f * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep, Npp8u * pDeviceBuffer, Npp8u * pAdvancedScratchBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp32f,Npp32f,1,0,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::NormLevelAdvanced>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, pDeviceBuffer, pAdvancedScratchBuffer, nppStreamCtx.hStream);
}

NppStatus nppiCrossCorrFull_NormLevelAdvanced_32f_C3R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp32f * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep, Npp8u * pDeviceBuffer, Npp8u * pAdvancedScratchBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp32f,Npp32f,3,0,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::NormLevelAdvanced>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, pDeviceBuffer, pAdvancedScratchBuffer, nppStreamCtx.hStream);
}

NppStatus nppiCrossCorrFull_NormLevelAdvanced_32f_C4R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp32f * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep, Npp8u * pDeviceBuffer, Npp8u * pAdvancedScratchBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp32f,Npp32f,4,0,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::NormLevelAdvanced>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, pDeviceBuffer, pAdvancedScratchBuffer, nppStreamCtx.hStream);
}

NppStatus nppiCrossCorrFull_NormLevelAdvanced_64f_C1R_Ctx(const Npp64f * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp64f * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp64f * pDst, int nDstStep, Npp8u * pDeviceBuffer, Npp8u * pAdvancedScratchBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp64f,Npp64f,1,0,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::NormLevelAdvanced>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, pDeviceBuffer, pAdvancedScratchBuffer, nppStreamCtx.hStream);
}

NppStatus nppiCrossCorrFull_NormLevelAdvanced_64f_C3R_Ctx(const Npp64f * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp64f * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp64f * pDst, int nDstStep, Npp8u * pDeviceBuffer, Npp8u * pAdvancedScratchBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp64f,Npp64f,3,0,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::NormLevelAdvanced>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, pDeviceBuffer, pAdvancedScratchBuffer, nppStreamCtx.hStream);
}

NppStatus nppiCrossCorrFull_NormLevelAdvanced_64f_C4R_Ctx(const Npp64f * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp64f * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp64f * pDst, int nDstStep, Npp8u * pDeviceBuffer, Npp8u * pAdvancedScratchBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp64f,Npp64f,4,0,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::NormLevelAdvanced>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, pDeviceBuffer, pAdvancedScratchBuffer, nppStreamCtx.hStream);
}

NppStatus nppiCrossCorrFull_NormLevelAdvanced_8s32f_C1R_Ctx(const Npp8s * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp8s * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep, Npp8u * pDeviceBuffer, Npp8u * pAdvancedScratchBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp8s,Npp32f,1,0,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::NormLevelAdvanced>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, pDeviceBuffer, pAdvancedScratchBuffer, nppStreamCtx.hStream);
}

NppStatus nppiCrossCorrFull_NormLevelAdvanced_8s32f_C3R_Ctx(const Npp8s * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp8s * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep, Npp8u * pDeviceBuffer, Npp8u * pAdvancedScratchBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp8s,Npp32f,3,0,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::NormLevelAdvanced>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, pDeviceBuffer, pAdvancedScratchBuffer, nppStreamCtx.hStream);
}

NppStatus nppiCrossCorrFull_NormLevelAdvanced_8s32f_C4R_Ctx(const Npp8s * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp8s * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep, Npp8u * pDeviceBuffer, Npp8u * pAdvancedScratchBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp8s,Npp32f,4,0,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::NormLevelAdvanced>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, pDeviceBuffer, pAdvancedScratchBuffer, nppStreamCtx.hStream);
}

NppStatus nppiCrossCorrFull_NormLevelAdvanced_8u32f_C1R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp8u * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep, Npp8u * pDeviceBuffer, Npp8u * pAdvancedScratchBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp8u,Npp32f,1,0,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::NormLevelAdvanced>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, pDeviceBuffer, pAdvancedScratchBuffer, nppStreamCtx.hStream);
}

NppStatus nppiCrossCorrFull_NormLevelAdvanced_8u32f_C3R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp8u * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep, Npp8u * pDeviceBuffer, Npp8u * pAdvancedScratchBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp8u,Npp32f,3,0,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::NormLevelAdvanced>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, pDeviceBuffer, pAdvancedScratchBuffer, nppStreamCtx.hStream);
}

NppStatus nppiCrossCorrFull_NormLevelAdvanced_8u32f_C4R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp8u * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep, Npp8u * pDeviceBuffer, Npp8u * pAdvancedScratchBuffer, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp8u,Npp32f,4,0,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::NormLevelAdvanced>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, pDeviceBuffer, pAdvancedScratchBuffer, nppStreamCtx.hStream);
}
