// Generated API wrappers -- do not edit by hand.
// family: nppiSqrDistanceFull_Norm


#include "npp.h"
#include "nppi_crosscorr_executor.h"

NppStatus nppiSqrDistanceFull_Norm_16u32f_AC4R_Ctx(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp16u * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp16u,Npp32f,4,1,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::SqrDistance>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, nullptr, nullptr, nppStreamCtx.hStream);
}

NppStatus nppiSqrDistanceFull_Norm_16u32f_C1R_Ctx(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp16u * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp16u,Npp32f,1,0,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::SqrDistance>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, nullptr, nullptr, nppStreamCtx.hStream);
}

NppStatus nppiSqrDistanceFull_Norm_16u32f_C3R_Ctx(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp16u * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp16u,Npp32f,3,0,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::SqrDistance>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, nullptr, nullptr, nppStreamCtx.hStream);
}

NppStatus nppiSqrDistanceFull_Norm_16u32f_C4R_Ctx(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp16u * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp16u,Npp32f,4,0,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::SqrDistance>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, nullptr, nullptr, nppStreamCtx.hStream);
}

NppStatus nppiSqrDistanceFull_Norm_32f_AC4R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp32f * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp32f,Npp32f,4,1,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::SqrDistance>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, nullptr, nullptr, nppStreamCtx.hStream);
}

NppStatus nppiSqrDistanceFull_Norm_32f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp32f * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp32f,Npp32f,1,0,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::SqrDistance>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, nullptr, nullptr, nppStreamCtx.hStream);
}

NppStatus nppiSqrDistanceFull_Norm_32f_C3R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp32f * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp32f,Npp32f,3,0,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::SqrDistance>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, nullptr, nullptr, nppStreamCtx.hStream);
}

NppStatus nppiSqrDistanceFull_Norm_32f_C4R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp32f * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp32f,Npp32f,4,0,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::SqrDistance>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, nullptr, nullptr, nppStreamCtx.hStream);
}

NppStatus nppiSqrDistanceFull_Norm_8s32f_AC4R_Ctx(const Npp8s * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp8s * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp8s,Npp32f,4,1,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::SqrDistance>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, nullptr, nullptr, nppStreamCtx.hStream);
}

NppStatus nppiSqrDistanceFull_Norm_8s32f_C1R_Ctx(const Npp8s * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp8s * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp8s,Npp32f,1,0,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::SqrDistance>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, nullptr, nullptr, nppStreamCtx.hStream);
}

NppStatus nppiSqrDistanceFull_Norm_8s32f_C3R_Ctx(const Npp8s * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp8s * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp8s,Npp32f,3,0,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::SqrDistance>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, nullptr, nullptr, nppStreamCtx.hStream);
}

NppStatus nppiSqrDistanceFull_Norm_8s32f_C4R_Ctx(const Npp8s * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp8s * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp8s,Npp32f,4,0,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::SqrDistance>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, nullptr, nullptr, nppStreamCtx.hStream);
}

NppStatus nppiSqrDistanceFull_Norm_8u32f_AC4R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp8u * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp8u,Npp32f,4,1,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::SqrDistance>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, nullptr, nullptr, nppStreamCtx.hStream);
}

NppStatus nppiSqrDistanceFull_Norm_8u32f_C1R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp8u * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp8u,Npp32f,1,0,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::SqrDistance>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, nullptr, nullptr, nppStreamCtx.hStream);
}

NppStatus nppiSqrDistanceFull_Norm_8u32f_C3R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp8u * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp8u,Npp32f,3,0,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::SqrDistance>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, nullptr, nullptr, nppStreamCtx.hStream);
}

NppStatus nppiSqrDistanceFull_Norm_8u32f_C4R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp8u * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp8u,Npp32f,4,0,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::SqrDistance>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 0, nullptr, nullptr, nppStreamCtx.hStream);
}

NppStatus nppiSqrDistanceFull_Norm_8u_AC4RSfs_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp8u * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp8u * pDst, int nDstStep, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp8u,Npp8u,4,1,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::SqrDistance>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, nScaleFactor, nullptr, nullptr, nppStreamCtx.hStream);
}

NppStatus nppiSqrDistanceFull_Norm_8u_C1RSfs_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp8u * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp8u * pDst, int nDstStep, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp8u,Npp8u,1,0,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::SqrDistance>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, nScaleFactor, nullptr, nullptr, nppStreamCtx.hStream);
}

NppStatus nppiSqrDistanceFull_Norm_8u_C3RSfs_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp8u * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp8u * pDst, int nDstStep, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp8u,Npp8u,3,0,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::SqrDistance>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, nScaleFactor, nullptr, nullptr, nppStreamCtx.hStream);
}

NppStatus nppiSqrDistanceFull_Norm_8u_C4RSfs_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp8u * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp8u * pDst, int nDstStep, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCrossCorr<Npp8u,Npp8u,4,0,nppi::statistics::CrossCorrMode::Full,nppi::statistics::CrossCorrOp::SqrDistance>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, nScaleFactor, nullptr, nullptr, nppStreamCtx.hStream);
}

NppStatus nppiSqrDistanceFull_Norm_16u32f_AC4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp16u * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiSqrDistanceFull_Norm_16u32f_AC4R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, nppStreamCtx);
}

NppStatus nppiSqrDistanceFull_Norm_16u32f_C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp16u * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiSqrDistanceFull_Norm_16u32f_C1R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, nppStreamCtx);
}

NppStatus nppiSqrDistanceFull_Norm_16u32f_C3R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp16u * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiSqrDistanceFull_Norm_16u32f_C3R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, nppStreamCtx);
}

NppStatus nppiSqrDistanceFull_Norm_16u32f_C4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp16u * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiSqrDistanceFull_Norm_16u32f_C4R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, nppStreamCtx);
}

NppStatus nppiSqrDistanceFull_Norm_32f_AC4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp32f * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiSqrDistanceFull_Norm_32f_AC4R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, nppStreamCtx);
}

NppStatus nppiSqrDistanceFull_Norm_32f_C1R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp32f * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiSqrDistanceFull_Norm_32f_C1R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, nppStreamCtx);
}

NppStatus nppiSqrDistanceFull_Norm_32f_C3R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp32f * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiSqrDistanceFull_Norm_32f_C3R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, nppStreamCtx);
}

NppStatus nppiSqrDistanceFull_Norm_32f_C4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp32f * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiSqrDistanceFull_Norm_32f_C4R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, nppStreamCtx);
}

NppStatus nppiSqrDistanceFull_Norm_8s32f_AC4R(const Npp8s * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp8s * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiSqrDistanceFull_Norm_8s32f_AC4R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, nppStreamCtx);
}

NppStatus nppiSqrDistanceFull_Norm_8s32f_C1R(const Npp8s * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp8s * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiSqrDistanceFull_Norm_8s32f_C1R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, nppStreamCtx);
}

NppStatus nppiSqrDistanceFull_Norm_8s32f_C3R(const Npp8s * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp8s * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiSqrDistanceFull_Norm_8s32f_C3R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, nppStreamCtx);
}

NppStatus nppiSqrDistanceFull_Norm_8s32f_C4R(const Npp8s * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp8s * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiSqrDistanceFull_Norm_8s32f_C4R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, nppStreamCtx);
}

NppStatus nppiSqrDistanceFull_Norm_8u32f_AC4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp8u * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiSqrDistanceFull_Norm_8u32f_AC4R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, nppStreamCtx);
}

NppStatus nppiSqrDistanceFull_Norm_8u32f_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp8u * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiSqrDistanceFull_Norm_8u32f_C1R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, nppStreamCtx);
}

NppStatus nppiSqrDistanceFull_Norm_8u32f_C3R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp8u * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiSqrDistanceFull_Norm_8u32f_C3R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, nppStreamCtx);
}

NppStatus nppiSqrDistanceFull_Norm_8u32f_C4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp8u * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f * pDst, int nDstStep) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiSqrDistanceFull_Norm_8u32f_C4R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, nppStreamCtx);
}

NppStatus nppiSqrDistanceFull_Norm_8u_AC4RSfs(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp8u * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp8u * pDst, int nDstStep, int nScaleFactor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiSqrDistanceFull_Norm_8u_AC4RSfs_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSqrDistanceFull_Norm_8u_C1RSfs(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp8u * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp8u * pDst, int nDstStep, int nScaleFactor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiSqrDistanceFull_Norm_8u_C1RSfs_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSqrDistanceFull_Norm_8u_C3RSfs(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp8u * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp8u * pDst, int nDstStep, int nScaleFactor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiSqrDistanceFull_Norm_8u_C3RSfs_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, nScaleFactor, nppStreamCtx);
}

NppStatus nppiSqrDistanceFull_Norm_8u_C4RSfs(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcRoiSize, const Npp8u * pTpl, int nTplStep, NppiSize oTplRoiSize, Npp8u * pDst, int nDstStep, int nScaleFactor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiSqrDistanceFull_Norm_8u_C4RSfs_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, nScaleFactor, nppStreamCtx);
}
