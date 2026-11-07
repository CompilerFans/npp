// Generated API wrappers -- do not edit by hand.
// family: nppiFloodFillGradientBoundary


#include "npp.h"
#include "nppi_flood_fill_executor.h"

NppStatus nppiFloodFillGradientBoundary_16u_C1IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, NppiPoint oSeed, Npp16u nMin, Npp16u nMax, const Npp16u nNewValue, const Npp16u nBoundaryValue, NppiNorm eNorm, NppiSize oSizeROI, NppiConnectedRegion * pConnectedRegion, Npp8u * pBuffer, NppStreamContext nppStreamCtx) {
  return nppi::floodfill::floodFillGradientBoundary<Npp16u, 1>(pSrcDst, nSrcDstStep, oSeed, &nMin, &nMax, &nNewValue, &nBoundaryValue, eNorm, oSizeROI, pConnectedRegion, pBuffer, nppStreamCtx.hStream);
}

NppStatus nppiFloodFillGradientBoundary_16u_C3IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, NppiPoint oSeed, Npp16u aMin[3], Npp16u aMax[3], const Npp16u aNewValues[3], const Npp16u aBoundaryValues[3], NppiNorm eNorm, NppiSize oSizeROI, NppiConnectedRegion * pConnectedRegion, Npp8u * pBuffer, NppStreamContext nppStreamCtx) {
  return nppi::floodfill::floodFillGradientBoundary<Npp16u, 3>(pSrcDst, nSrcDstStep, oSeed, aMin, aMax, aNewValues, aBoundaryValues, eNorm, oSizeROI, pConnectedRegion, pBuffer, nppStreamCtx.hStream);
}

NppStatus nppiFloodFillGradientBoundary_32u_C1IR_Ctx(Npp32u * pSrcDst, int nSrcDstStep, NppiPoint oSeed, Npp32u nMin, Npp32u nMax, const Npp32u nNewValue, const Npp32u nBoundaryValue, NppiNorm eNorm, NppiSize oSizeROI, NppiConnectedRegion * pConnectedRegion, Npp8u * pBuffer, NppStreamContext nppStreamCtx) {
  return nppi::floodfill::floodFillGradientBoundary<Npp32u, 1>(pSrcDst, nSrcDstStep, oSeed, &nMin, &nMax, &nNewValue, &nBoundaryValue, eNorm, oSizeROI, pConnectedRegion, pBuffer, nppStreamCtx.hStream);
}

NppStatus nppiFloodFillGradientBoundary_32u_C3IR_Ctx(Npp32u * pSrcDst, int nSrcDstStep, NppiPoint oSeed, Npp32u aMin[3], Npp32u aMax[3], const Npp32u aNewValues[3], const Npp32u aBoundaryValues[3], NppiNorm eNorm, NppiSize oSizeROI, NppiConnectedRegion * pConnectedRegion, Npp8u * pBuffer, NppStreamContext nppStreamCtx) {
  return nppi::floodfill::floodFillGradientBoundary<Npp32u, 3>(pSrcDst, nSrcDstStep, oSeed, aMin, aMax, aNewValues, aBoundaryValues, eNorm, oSizeROI, pConnectedRegion, pBuffer, nppStreamCtx.hStream);
}

NppStatus nppiFloodFillGradientBoundary_8u_C1IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, NppiPoint oSeed, Npp8u nMin, Npp8u nMax, const Npp8u nNewValue, const Npp8u nBoundaryValue, NppiNorm eNorm, NppiSize oSizeROI, NppiConnectedRegion * pConnectedRegion, Npp8u * pBuffer, NppStreamContext nppStreamCtx) {
  return nppi::floodfill::floodFillGradientBoundary<Npp8u, 1>(pSrcDst, nSrcDstStep, oSeed, &nMin, &nMax, &nNewValue, &nBoundaryValue, eNorm, oSizeROI, pConnectedRegion, pBuffer, nppStreamCtx.hStream);
}

NppStatus nppiFloodFillGradientBoundary_8u_C3IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, NppiPoint oSeed, Npp8u aMin[3], Npp8u aMax[3], const Npp8u aNewValues[3], const Npp8u aBoundaryValues[3], NppiNorm eNorm, NppiSize oSizeROI, NppiConnectedRegion * pConnectedRegion, Npp8u * pBuffer, NppStreamContext nppStreamCtx) {
  return nppi::floodfill::floodFillGradientBoundary<Npp8u, 3>(pSrcDst, nSrcDstStep, oSeed, aMin, aMax, aNewValues, aBoundaryValues, eNorm, oSizeROI, pConnectedRegion, pBuffer, nppStreamCtx.hStream);
}

NppStatus nppiFloodFillGradientBoundary_16u_C1IR(Npp16u * pSrcDst, int nSrcDstStep, NppiPoint oSeed, Npp16u nMin, Npp16u nMax, const Npp16u nNewValue, const Npp16u nBoundaryValue, NppiNorm eNorm, NppiSize oSizeROI, NppiConnectedRegion * pConnectedRegion, Npp8u * pBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFloodFillGradientBoundary_16u_C1IR_Ctx(pSrcDst, nSrcDstStep, oSeed, nMin, nMax, nNewValue, nBoundaryValue, eNorm, oSizeROI, pConnectedRegion, pBuffer, nppStreamCtx);
}

NppStatus nppiFloodFillGradientBoundary_16u_C3IR(Npp16u * pSrcDst, int nSrcDstStep, NppiPoint oSeed, Npp16u aMin[3], Npp16u aMax[3], const Npp16u aNewValues[3], const Npp16u aBoundaryValues[3], NppiNorm eNorm, NppiSize oSizeROI, NppiConnectedRegion * pConnectedRegion, Npp8u * pBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFloodFillGradientBoundary_16u_C3IR_Ctx(pSrcDst, nSrcDstStep, oSeed, aMin, aMax, aNewValues, aBoundaryValues, eNorm, oSizeROI, pConnectedRegion, pBuffer, nppStreamCtx);
}

NppStatus nppiFloodFillGradientBoundary_32u_C1IR(Npp32u * pSrcDst, int nSrcDstStep, NppiPoint oSeed, Npp32u nMin, Npp32u nMax, const Npp32u nNewValue, const Npp32u nBoundaryValue, NppiNorm eNorm, NppiSize oSizeROI, NppiConnectedRegion * pConnectedRegion, Npp8u * pBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFloodFillGradientBoundary_32u_C1IR_Ctx(pSrcDst, nSrcDstStep, oSeed, nMin, nMax, nNewValue, nBoundaryValue, eNorm, oSizeROI, pConnectedRegion, pBuffer, nppStreamCtx);
}

NppStatus nppiFloodFillGradientBoundary_32u_C3IR(Npp32u * pSrcDst, int nSrcDstStep, NppiPoint oSeed, Npp32u aMin[3], Npp32u aMax[3], const Npp32u aNewValues[3], const Npp32u aBoundaryValues[3], NppiNorm eNorm, NppiSize oSizeROI, NppiConnectedRegion * pConnectedRegion, Npp8u * pBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFloodFillGradientBoundary_32u_C3IR_Ctx(pSrcDst, nSrcDstStep, oSeed, aMin, aMax, aNewValues, aBoundaryValues, eNorm, oSizeROI, pConnectedRegion, pBuffer, nppStreamCtx);
}

NppStatus nppiFloodFillGradientBoundary_8u_C1IR(Npp8u * pSrcDst, int nSrcDstStep, NppiPoint oSeed, Npp8u nMin, Npp8u nMax, const Npp8u nNewValue, const Npp8u nBoundryValue, NppiNorm eNorm, NppiSize oSizeROI, NppiConnectedRegion * pConnectedRegion, Npp8u * pBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFloodFillGradientBoundary_8u_C1IR_Ctx(pSrcDst, nSrcDstStep, oSeed, nMin, nMax, nNewValue, nBoundryValue, eNorm, oSizeROI, pConnectedRegion, pBuffer, nppStreamCtx);
}

NppStatus nppiFloodFillGradientBoundary_8u_C3IR(Npp8u * pSrcDst, int nSrcDstStep, NppiPoint oSeed, Npp8u aMin[3], Npp8u aMax[3], const Npp8u aNewValues[3], const Npp8u aBoundaryValues[3], NppiNorm eNorm, NppiSize oSizeROI, NppiConnectedRegion * pConnectedRegion, Npp8u * pBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFloodFillGradientBoundary_8u_C3IR_Ctx(pSrcDst, nSrcDstStep, oSeed, aMin, aMax, aNewValues, aBoundaryValues, eNorm, oSizeROI, pConnectedRegion, pBuffer, nppStreamCtx);
}
