// Generated API wrappers -- do not edit by hand.
// family: nppiFloodFill


#include "npp.h"
#include "nppi_flood_fill_executor.h"

NppStatus nppiFloodFill_16u_C1IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, NppiPoint oSeed, const Npp16u nNewValue, NppiNorm eNorm, NppiSize oSizeROI, NppiConnectedRegion * pConnectedRegion, Npp8u * pBuffer, NppStreamContext nppStreamCtx) {
  return nppi::floodfill::floodFill<Npp16u, 1>(pSrcDst, nSrcDstStep, oSeed, &nNewValue, eNorm, oSizeROI, pConnectedRegion, pBuffer, nppStreamCtx.hStream);
}

NppStatus nppiFloodFill_16u_C3IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, NppiPoint oSeed, const Npp16u aNewValues[3], NppiNorm eNorm, NppiSize oSizeROI, NppiConnectedRegion * pConnectedRegion, Npp8u * pBuffer, NppStreamContext nppStreamCtx) {
  return nppi::floodfill::floodFill<Npp16u, 3>(pSrcDst, nSrcDstStep, oSeed, aNewValues, eNorm, oSizeROI, pConnectedRegion, pBuffer, nppStreamCtx.hStream);
}

NppStatus nppiFloodFill_32u_C1IR_Ctx(Npp32u * pSrcDst, int nSrcDstStep, NppiPoint oSeed, const Npp32u nNewValue, NppiNorm eNorm, NppiSize oSizeROI, NppiConnectedRegion * pConnectedRegion, Npp8u * pBuffer, NppStreamContext nppStreamCtx) {
  return nppi::floodfill::floodFill<Npp32u, 1>(pSrcDst, nSrcDstStep, oSeed, &nNewValue, eNorm, oSizeROI, pConnectedRegion, pBuffer, nppStreamCtx.hStream);
}

NppStatus nppiFloodFill_32u_C3IR_Ctx(Npp32u * pSrcDst, int nSrcDstStep, NppiPoint oSeed, const Npp32u aNewValues[3], NppiNorm eNorm, NppiSize oSizeROI, NppiConnectedRegion * pConnectedRegion, Npp8u * pBuffer, NppStreamContext nppStreamCtx) {
  return nppi::floodfill::floodFill<Npp32u, 3>(pSrcDst, nSrcDstStep, oSeed, aNewValues, eNorm, oSizeROI, pConnectedRegion, pBuffer, nppStreamCtx.hStream);
}

NppStatus nppiFloodFill_8u_C1IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, NppiPoint oSeed, const Npp8u nNewValue, NppiNorm eNorm, NppiSize oSizeROI, NppiConnectedRegion * pConnectedRegion, Npp8u * pBuffer, NppStreamContext nppStreamCtx) {
  return nppi::floodfill::floodFill<Npp8u, 1>(pSrcDst, nSrcDstStep, oSeed, &nNewValue, eNorm, oSizeROI, pConnectedRegion, pBuffer, nppStreamCtx.hStream);
}

NppStatus nppiFloodFill_8u_C3IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, NppiPoint oSeed, const Npp8u aNewValues[3], NppiNorm eNorm, NppiSize oSizeROI, NppiConnectedRegion * pConnectedRegion, Npp8u * pBuffer, NppStreamContext nppStreamCtx) {
  return nppi::floodfill::floodFill<Npp8u, 3>(pSrcDst, nSrcDstStep, oSeed, aNewValues, eNorm, oSizeROI, pConnectedRegion, pBuffer, nppStreamCtx.hStream);
}

NppStatus nppiFloodFill_16u_C1IR(Npp16u * pSrcDst, int nSrcDstStep, NppiPoint oSeed, const Npp16u nNewValue, NppiNorm eNorm, NppiSize oSizeROI, NppiConnectedRegion * pConnectedRegion, Npp8u * pBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFloodFill_16u_C1IR_Ctx(pSrcDst, nSrcDstStep, oSeed, nNewValue, eNorm, oSizeROI, pConnectedRegion, pBuffer, nppStreamCtx);
}

NppStatus nppiFloodFill_16u_C3IR(Npp16u * pSrcDst, int nSrcDstStep, NppiPoint oSeed, const Npp16u aNewValues[3], NppiNorm eNorm, NppiSize oSizeROI, NppiConnectedRegion * pConnectedRegion, Npp8u * pBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFloodFill_16u_C3IR_Ctx(pSrcDst, nSrcDstStep, oSeed, aNewValues, eNorm, oSizeROI, pConnectedRegion, pBuffer, nppStreamCtx);
}

NppStatus nppiFloodFill_32u_C1IR(Npp32u * pSrcDst, int nSrcDstStep, NppiPoint oSeed, const Npp32u nNewValue, NppiNorm eNorm, NppiSize oSizeROI, NppiConnectedRegion * pConnectedRegion, Npp8u * pBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFloodFill_32u_C1IR_Ctx(pSrcDst, nSrcDstStep, oSeed, nNewValue, eNorm, oSizeROI, pConnectedRegion, pBuffer, nppStreamCtx);
}

NppStatus nppiFloodFill_32u_C3IR(Npp32u * pSrcDst, int nSrcDstStep, NppiPoint oSeed, const Npp32u aNewValues[3], NppiNorm eNorm, NppiSize oSizeROI, NppiConnectedRegion * pConnectedRegion, Npp8u * pBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFloodFill_32u_C3IR_Ctx(pSrcDst, nSrcDstStep, oSeed, aNewValues, eNorm, oSizeROI, pConnectedRegion, pBuffer, nppStreamCtx);
}

NppStatus nppiFloodFill_8u_C1IR(Npp8u * pSrcDst, int nSrcDstStep, NppiPoint oSeed, const Npp8u nNewValue, NppiNorm eNorm, NppiSize oSizeROI, NppiConnectedRegion * pConnectedRegion, Npp8u * pBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFloodFill_8u_C1IR_Ctx(pSrcDst, nSrcDstStep, oSeed, nNewValue, eNorm, oSizeROI, pConnectedRegion, pBuffer, nppStreamCtx);
}

NppStatus nppiFloodFill_8u_C3IR(Npp8u * pSrcDst, int nSrcDstStep, NppiPoint oSeed, const Npp8u aNewValues[3], NppiNorm eNorm, NppiSize oSizeROI, NppiConnectedRegion * pConnectedRegion, Npp8u * pBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFloodFill_8u_C3IR_Ctx(pSrcDst, nSrcDstStep, oSeed, aNewValues, eNorm, oSizeROI, pConnectedRegion, pBuffer, nppStreamCtx);
}
