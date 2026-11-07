// Generated API wrappers -- do not edit by hand.
// family: nppiLUTPalette


#include "npp.h"
#include "nppi_lut_executor.h"

NppStatus nppiLUTPalette_16u32u_C1R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp32u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32u * pTable, int nBitSize, NppStreamContext nppStreamCtx) {
  return nppi::lut::paletteC1<Npp16u, Npp32u, Npp32u>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pTable, nBitSize, nppStreamCtx.hStream);
}

NppStatus nppiLUTPalette_16u8u_C1R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pTable, int nBitSize, NppStreamContext nppStreamCtx) {
  return nppi::lut::paletteC1<Npp16u, Npp8u, Npp8u>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pTable, nBitSize, nppStreamCtx.hStream);
}

NppStatus nppiLUTPalette_16u_C1R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp16u * pTable, int nBitSize, NppStreamContext nppStreamCtx) {
  return nppi::lut::paletteC1<Npp16u, Npp16u, Npp16u>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pTable, nBitSize, nppStreamCtx.hStream);
}

NppStatus nppiLUTPalette_8u32u_C1R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp32u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32u * pTable, int nBitSize, NppStreamContext nppStreamCtx) {
  return nppi::lut::paletteC1<Npp8u, Npp32u, Npp32u>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pTable, nBitSize, nppStreamCtx.hStream);
}

NppStatus nppiLUTPalette_8u_C1R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pTable, int nBitSize, NppStreamContext nppStreamCtx) {
  return nppi::lut::paletteC1<Npp8u, Npp8u, Npp8u>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pTable, nBitSize, nppStreamCtx.hStream);
}

NppStatus nppiLUTPalette_16u32u_C1R(const Npp16u * pSrc, int nSrcStep, Npp32u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32u * pTable, int nBitSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUTPalette_16u32u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pTable, nBitSize, nppStreamCtx);
}

NppStatus nppiLUTPalette_16u8u_C1R(const Npp16u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pTable, int nBitSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUTPalette_16u8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pTable, nBitSize, nppStreamCtx);
}

NppStatus nppiLUTPalette_16u_C1R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp16u * pTable, int nBitSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUTPalette_16u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pTable, nBitSize, nppStreamCtx);
}

NppStatus nppiLUTPalette_8u32u_C1R(const Npp8u * pSrc, int nSrcStep, Npp32u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32u * pTable, int nBitSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUTPalette_8u32u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pTable, nBitSize, nppStreamCtx);
}

NppStatus nppiLUTPalette_8u_C1R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pTable, int nBitSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUTPalette_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pTable, nBitSize, nppStreamCtx);
}
