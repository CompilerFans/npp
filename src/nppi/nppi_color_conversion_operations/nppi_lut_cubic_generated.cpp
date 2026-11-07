// Generated API wrappers -- do not edit by hand.
// family: nppiLUT_Cubic


#include "npp.h"
#include "nppi_lut_executor.h"

NppStatus nppiLUT_Cubic_16s_AC4IR_Ctx(Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32s * pValues[3], const Npp32s * pLevels[3], int nLevels[3], NppStreamContext nppStreamCtx) {
  return nppi::lut::runInPlace<Npp16s, 4, 1, nppi::lut::LutMode::Cubic>(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx.hStream);
}

NppStatus nppiLUT_Cubic_16s_AC4R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32s * pValues[3], const Npp32s * pLevels[3], int nLevels[3], NppStreamContext nppStreamCtx) {
  return nppi::lut::run<Npp16s, 4, 1, nppi::lut::LutMode::Cubic>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx.hStream);
}

NppStatus nppiLUT_Cubic_16s_C1IR_Ctx(Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32s * pValues, const Npp32s * pLevels, int nLevels, NppStreamContext nppStreamCtx) {
  return nppi::lut::runInPlace<Npp16s, 1, false, nppi::lut::LutMode::Cubic>(pSrcDst, nSrcDstStep, oSizeROI, &pValues, &pLevels, &nLevels, nppStreamCtx.hStream);
}

NppStatus nppiLUT_Cubic_16s_C1R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32s * pValues, const Npp32s * pLevels, int nLevels, NppStreamContext nppStreamCtx) {
  return nppi::lut::runC1<Npp16s, nppi::lut::LutMode::Cubic>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx.hStream);
}

NppStatus nppiLUT_Cubic_16s_C3IR_Ctx(Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32s * pValues[3], const Npp32s * pLevels[3], int nLevels[3], NppStreamContext nppStreamCtx) {
  return nppi::lut::runInPlace<Npp16s, 3, 0, nppi::lut::LutMode::Cubic>(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx.hStream);
}

NppStatus nppiLUT_Cubic_16s_C3R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32s * pValues[3], const Npp32s * pLevels[3], int nLevels[3], NppStreamContext nppStreamCtx) {
  return nppi::lut::run<Npp16s, 3, 0, nppi::lut::LutMode::Cubic>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx.hStream);
}

NppStatus nppiLUT_Cubic_16s_C4IR_Ctx(Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32s * pValues[4], const Npp32s * pLevels[4], int nLevels[4], NppStreamContext nppStreamCtx) {
  return nppi::lut::runInPlace<Npp16s, 4, 0, nppi::lut::LutMode::Cubic>(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx.hStream);
}

NppStatus nppiLUT_Cubic_16s_C4R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32s * pValues[4], const Npp32s * pLevels[4], int nLevels[4], NppStreamContext nppStreamCtx) {
  return nppi::lut::run<Npp16s, 4, 0, nppi::lut::LutMode::Cubic>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx.hStream);
}

NppStatus nppiLUT_Cubic_16u_AC4IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32s * pValues[3], const Npp32s * pLevels[3], int nLevels[3], NppStreamContext nppStreamCtx) {
  return nppi::lut::runInPlace<Npp16u, 4, 1, nppi::lut::LutMode::Cubic>(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx.hStream);
}

NppStatus nppiLUT_Cubic_16u_AC4R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32s * pValues[3], const Npp32s * pLevels[3], int nLevels[3], NppStreamContext nppStreamCtx) {
  return nppi::lut::run<Npp16u, 4, 1, nppi::lut::LutMode::Cubic>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx.hStream);
}

NppStatus nppiLUT_Cubic_16u_C1IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32s * pValues, const Npp32s * pLevels, int nLevels, NppStreamContext nppStreamCtx) {
  return nppi::lut::runInPlace<Npp16u, 1, false, nppi::lut::LutMode::Cubic>(pSrcDst, nSrcDstStep, oSizeROI, &pValues, &pLevels, &nLevels, nppStreamCtx.hStream);
}

NppStatus nppiLUT_Cubic_16u_C1R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32s * pValues, const Npp32s * pLevels, int nLevels, NppStreamContext nppStreamCtx) {
  return nppi::lut::runC1<Npp16u, nppi::lut::LutMode::Cubic>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx.hStream);
}

NppStatus nppiLUT_Cubic_16u_C3IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32s * pValues[3], const Npp32s * pLevels[3], int nLevels[3], NppStreamContext nppStreamCtx) {
  return nppi::lut::runInPlace<Npp16u, 3, 0, nppi::lut::LutMode::Cubic>(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx.hStream);
}

NppStatus nppiLUT_Cubic_16u_C3R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32s * pValues[3], const Npp32s * pLevels[3], int nLevels[3], NppStreamContext nppStreamCtx) {
  return nppi::lut::run<Npp16u, 3, 0, nppi::lut::LutMode::Cubic>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx.hStream);
}

NppStatus nppiLUT_Cubic_16u_C4IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32s * pValues[4], const Npp32s * pLevels[4], int nLevels[4], NppStreamContext nppStreamCtx) {
  return nppi::lut::runInPlace<Npp16u, 4, 0, nppi::lut::LutMode::Cubic>(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx.hStream);
}

NppStatus nppiLUT_Cubic_16u_C4R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32s * pValues[4], const Npp32s * pLevels[4], int nLevels[4], NppStreamContext nppStreamCtx) {
  return nppi::lut::run<Npp16u, 4, 0, nppi::lut::LutMode::Cubic>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx.hStream);
}

NppStatus nppiLUT_Cubic_32f_AC4IR_Ctx(Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f * pValues[3], const Npp32f * pLevels[3], int nLevels[3], NppStreamContext nppStreamCtx) {
  return nppi::lut::runInPlace<Npp32f, 4, 1, nppi::lut::LutMode::Cubic>(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx.hStream);
}

NppStatus nppiLUT_Cubic_32f_AC4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pValues[3], const Npp32f * pLevels[3], int nLevels[3], NppStreamContext nppStreamCtx) {
  return nppi::lut::run<Npp32f, 4, 1, nppi::lut::LutMode::Cubic>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx.hStream);
}

NppStatus nppiLUT_Cubic_32f_C1IR_Ctx(Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f * pValues, const Npp32f * pLevels, int nLevels, NppStreamContext nppStreamCtx) {
  return nppi::lut::runInPlace<Npp32f, 1, false, nppi::lut::LutMode::Cubic>(pSrcDst, nSrcDstStep, oSizeROI, &pValues, &pLevels, &nLevels, nppStreamCtx.hStream);
}

NppStatus nppiLUT_Cubic_32f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pValues, const Npp32f * pLevels, int nLevels, NppStreamContext nppStreamCtx) {
  return nppi::lut::runC1<Npp32f, nppi::lut::LutMode::Cubic>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx.hStream);
}

NppStatus nppiLUT_Cubic_32f_C3IR_Ctx(Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f * pValues[3], const Npp32f * pLevels[3], int nLevels[3], NppStreamContext nppStreamCtx) {
  return nppi::lut::runInPlace<Npp32f, 3, 0, nppi::lut::LutMode::Cubic>(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx.hStream);
}

NppStatus nppiLUT_Cubic_32f_C3R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pValues[3], const Npp32f * pLevels[3], int nLevels[3], NppStreamContext nppStreamCtx) {
  return nppi::lut::run<Npp32f, 3, 0, nppi::lut::LutMode::Cubic>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx.hStream);
}

NppStatus nppiLUT_Cubic_32f_C4IR_Ctx(Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f * pValues[4], const Npp32f * pLevels[4], int nLevels[4], NppStreamContext nppStreamCtx) {
  return nppi::lut::runInPlace<Npp32f, 4, 0, nppi::lut::LutMode::Cubic>(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx.hStream);
}

NppStatus nppiLUT_Cubic_32f_C4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pValues[4], const Npp32f * pLevels[4], int nLevels[4], NppStreamContext nppStreamCtx) {
  return nppi::lut::run<Npp32f, 4, 0, nppi::lut::LutMode::Cubic>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx.hStream);
}

NppStatus nppiLUT_Cubic_8u_AC4IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32s * pValues[3], const Npp32s * pLevels[3], int nLevels[3], NppStreamContext nppStreamCtx) {
  return nppi::lut::runInPlace<Npp8u, 4, 1, nppi::lut::LutMode::Cubic>(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx.hStream);
}

NppStatus nppiLUT_Cubic_8u_AC4R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32s * pValues[3], const Npp32s * pLevels[3], int nLevels[3], NppStreamContext nppStreamCtx) {
  return nppi::lut::run<Npp8u, 4, 1, nppi::lut::LutMode::Cubic>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx.hStream);
}

NppStatus nppiLUT_Cubic_8u_C1IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32s * pValues, const Npp32s * pLevels, int nLevels, NppStreamContext nppStreamCtx) {
  return nppi::lut::runInPlace<Npp8u, 1, false, nppi::lut::LutMode::Cubic>(pSrcDst, nSrcDstStep, oSizeROI, &pValues, &pLevels, &nLevels, nppStreamCtx.hStream);
}

NppStatus nppiLUT_Cubic_8u_C1R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32s * pValues, const Npp32s * pLevels, int nLevels, NppStreamContext nppStreamCtx) {
  return nppi::lut::runC1<Npp8u, nppi::lut::LutMode::Cubic>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx.hStream);
}

NppStatus nppiLUT_Cubic_8u_C3IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32s * pValues[3], const Npp32s * pLevels[3], int nLevels[3], NppStreamContext nppStreamCtx) {
  return nppi::lut::runInPlace<Npp8u, 3, 0, nppi::lut::LutMode::Cubic>(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx.hStream);
}

NppStatus nppiLUT_Cubic_8u_C3R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32s * pValues[3], const Npp32s * pLevels[3], int nLevels[3], NppStreamContext nppStreamCtx) {
  return nppi::lut::run<Npp8u, 3, 0, nppi::lut::LutMode::Cubic>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx.hStream);
}

NppStatus nppiLUT_Cubic_8u_C4IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32s * pValues[4], const Npp32s * pLevels[4], int nLevels[4], NppStreamContext nppStreamCtx) {
  return nppi::lut::runInPlace<Npp8u, 4, 0, nppi::lut::LutMode::Cubic>(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx.hStream);
}

NppStatus nppiLUT_Cubic_8u_C4R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32s * pValues[4], const Npp32s * pLevels[4], int nLevels[4], NppStreamContext nppStreamCtx) {
  return nppi::lut::run<Npp8u, 4, 0, nppi::lut::LutMode::Cubic>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx.hStream);
}

NppStatus nppiLUT_Cubic_16s_AC4IR(Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32s * pValues[3], const Npp32s * pLevels[3], int nLevels[3]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUT_Cubic_16s_AC4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Cubic_16s_AC4R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32s * pValues[3], const Npp32s * pLevels[3], int nLevels[3]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUT_Cubic_16s_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Cubic_16s_C1IR(Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32s * pValues, const Npp32s * pLevels, int nLevels) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUT_Cubic_16s_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Cubic_16s_C1R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32s * pValues, const Npp32s * pLevels, int nLevels) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUT_Cubic_16s_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Cubic_16s_C3IR(Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32s * pValues[3], const Npp32s * pLevels[3], int nLevels[3]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUT_Cubic_16s_C3IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Cubic_16s_C3R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32s * pValues[3], const Npp32s * pLevels[3], int nLevels[3]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUT_Cubic_16s_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Cubic_16s_C4IR(Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32s * pValues[4], const Npp32s * pLevels[4], int nLevels[4]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUT_Cubic_16s_C4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Cubic_16s_C4R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32s * pValues[4], const Npp32s * pLevels[4], int nLevels[4]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUT_Cubic_16s_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Cubic_16u_AC4IR(Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32s * pValues[3], const Npp32s * pLevels[3], int nLevels[3]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUT_Cubic_16u_AC4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Cubic_16u_AC4R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32s * pValues[3], const Npp32s * pLevels[3], int nLevels[3]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUT_Cubic_16u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Cubic_16u_C1IR(Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32s * pValues, const Npp32s * pLevels, int nLevels) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUT_Cubic_16u_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Cubic_16u_C1R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32s * pValues, const Npp32s * pLevels, int nLevels) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUT_Cubic_16u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Cubic_16u_C3IR(Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32s * pValues[3], const Npp32s * pLevels[3], int nLevels[3]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUT_Cubic_16u_C3IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Cubic_16u_C3R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32s * pValues[3], const Npp32s * pLevels[3], int nLevels[3]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUT_Cubic_16u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Cubic_16u_C4IR(Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32s * pValues[4], const Npp32s * pLevels[4], int nLevels[4]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUT_Cubic_16u_C4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Cubic_16u_C4R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32s * pValues[4], const Npp32s * pLevels[4], int nLevels[4]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUT_Cubic_16u_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Cubic_32f_AC4IR(Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f * pValues[3], const Npp32f * pLevels[3], int nLevels[3]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUT_Cubic_32f_AC4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Cubic_32f_AC4R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pValues[3], const Npp32f * pLevels[3], int nLevels[3]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUT_Cubic_32f_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Cubic_32f_C1IR(Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f * pValues, const Npp32f * pLevels, int nLevels) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUT_Cubic_32f_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Cubic_32f_C1R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pValues, const Npp32f * pLevels, int nLevels) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUT_Cubic_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Cubic_32f_C3IR(Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f * pValues[3], const Npp32f * pLevels[3], int nLevels[3]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUT_Cubic_32f_C3IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Cubic_32f_C3R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pValues[3], const Npp32f * pLevels[3], int nLevels[3]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUT_Cubic_32f_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Cubic_32f_C4IR(Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f * pValues[4], const Npp32f * pLevels[4], int nLevels[4]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUT_Cubic_32f_C4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Cubic_32f_C4R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pValues[4], const Npp32f * pLevels[4], int nLevels[4]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUT_Cubic_32f_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Cubic_8u_AC4IR(Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32s * pValues[3], const Npp32s * pLevels[3], int nLevels[3]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUT_Cubic_8u_AC4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Cubic_8u_AC4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32s * pValues[3], const Npp32s * pLevels[3], int nLevels[3]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUT_Cubic_8u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Cubic_8u_C1IR(Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32s * pValues, const Npp32s * pLevels, int nLevels) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUT_Cubic_8u_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Cubic_8u_C1R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32s * pValues, const Npp32s * pLevels, int nLevels) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUT_Cubic_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Cubic_8u_C3IR(Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32s * pValues[3], const Npp32s * pLevels[3], int nLevels[3]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUT_Cubic_8u_C3IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Cubic_8u_C3R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32s * pValues[3], const Npp32s * pLevels[3], int nLevels[3]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUT_Cubic_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Cubic_8u_C4IR(Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32s * pValues[4], const Npp32s * pLevels[4], int nLevels[4]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUT_Cubic_8u_C4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Cubic_8u_C4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32s * pValues[4], const Npp32s * pLevels[4], int nLevels[4]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUT_Cubic_8u_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}
