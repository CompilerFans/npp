// Generated API wrappers -- do not edit by hand.
// family: nppiLUTPaletteSwap


#include "npp.h"
#include "nppi_lut_executor.h"

NppStatus nppiLUTPaletteSwap_16u_C3A0C4R_Ctx(const Npp16u * pSrc, int nSrcStep, int nAlphaValue, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp16u * pTables[3], int nBitSize, NppStreamContext nppStreamCtx) {
  return nppi::lut::paletteSwap<Npp16u>(pSrc, nSrcStep, nAlphaValue, pDst, nDstStep, oSizeROI, pTables, nBitSize, nppStreamCtx.hStream);
}

NppStatus nppiLUTPaletteSwap_8u_C3A0C4R_Ctx(const Npp8u * pSrc, int nSrcStep, int nAlphaValue, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pTables[3], int nBitSize, NppStreamContext nppStreamCtx) {
  return nppi::lut::paletteSwap<Npp8u>(pSrc, nSrcStep, nAlphaValue, pDst, nDstStep, oSizeROI, pTables, nBitSize, nppStreamCtx.hStream);
}

NppStatus nppiLUTPaletteSwap_16u_C3A0C4R(const Npp16u * pSrc, int nSrcStep, int nAlphaValue, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp16u * pTables[3], int nBitSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUTPaletteSwap_16u_C3A0C4R_Ctx(pSrc, nSrcStep, nAlphaValue, pDst, nDstStep, oSizeROI, pTables, nBitSize, nppStreamCtx);
}

NppStatus nppiLUTPaletteSwap_8u_C3A0C4R(const Npp8u * pSrc, int nSrcStep, int nAlphaValue, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pTables[3], int nBitSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUTPaletteSwap_8u_C3A0C4R_Ctx(pSrc, nSrcStep, nAlphaValue, pDst, nDstStep, oSizeROI, pTables, nBitSize, nppStreamCtx);
}
