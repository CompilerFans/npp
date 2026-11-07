// Generated API wrappers -- do not edit by hand.
// family: nppiCrossCorrValid


#include "npp.h"
#include "nppi_crosscorr_executor.h"

NppStatus nppiCrossCorrValid_16u32f_C1R_Ctx(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp16u * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp16u,Npp32f,1,0,nppi::statistics::CrossCorrMode::Valid,nppi::statistics::CrossCorrOp::Plain>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, nullptr, nullptr, nppStreamCtx.hStream);
}

NppStatus nppiCrossCorrValid_32f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp32f * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp32f,Npp32f,1,0,nppi::statistics::CrossCorrMode::Valid,nppi::statistics::CrossCorrOp::Plain>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, nullptr, nullptr, nppStreamCtx.hStream);
}

NppStatus nppiCrossCorrValid_64f_C1R_Ctx(const Npp64f * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp64f * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp64f * pDst, int nDstStep, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp64f,Npp64f,1,0,nppi::statistics::CrossCorrMode::Valid,nppi::statistics::CrossCorrOp::Plain>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, nullptr, nullptr, nppStreamCtx.hStream);
}

NppStatus nppiCrossCorrValid_8s32f_C1R_Ctx(const Npp8s * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp8s * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp8s,Npp32f,1,0,nppi::statistics::CrossCorrMode::Valid,nppi::statistics::CrossCorrOp::Plain>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, nullptr, nullptr, nppStreamCtx.hStream);
}

NppStatus nppiCrossCorrValid_8u32f_C1R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp8u * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp8u,Npp32f,1,0,nppi::statistics::CrossCorrMode::Valid,nppi::statistics::CrossCorrOp::Plain>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, nullptr, nullptr, nppStreamCtx.hStream);
}

NppStatus nppiCrossCorrValid_16u32f_C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp16u * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiCrossCorrValid_16u32f_C1R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, nppStreamCtx);
}

NppStatus nppiCrossCorrValid_32f_C1R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp32f * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiCrossCorrValid_32f_C1R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, nppStreamCtx);
}

NppStatus nppiCrossCorrValid_64f_C1R(const Npp64f * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp64f * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp64f * pDst, int nDstStep) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiCrossCorrValid_64f_C1R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, nppStreamCtx);
}

NppStatus nppiCrossCorrValid_8s32f_C1R(const Npp8s * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp8s * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiCrossCorrValid_8s32f_C1R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, nppStreamCtx);
}

NppStatus nppiCrossCorrValid_8u32f_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp8u * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiCrossCorrValid_8u32f_C1R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, nppStreamCtx);
}
