// Generated API wrappers -- do not edit by hand.
// family: nppiAlphaCompColorKey


#include "npp.h"
#include "nppi_color_key_executor.h"

NppStatus nppiAlphaCompColorKey_8u_AC4R_Ctx(const Npp8u * pSrc1, int nSrc1Step, Npp8u nAlpha1, const Npp8u * pSrc2, int nSrc2Step, Npp8u nAlpha2, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, Npp8u nColorKeyConst[4], NppiAlphaOp nppAlphaOp, NppStreamContext nppStreamCtx) {
  return nppi::color_key::alphaCompColorKey<Npp8u, 4>(pSrc1, nSrc1Step, nAlpha1, pSrc2, nSrc2Step, nAlpha2, pDst, nDstStep, oSizeROI, nColorKeyConst, nppAlphaOp, nppStreamCtx.hStream);
}

NppStatus nppiAlphaCompColorKey_8u_AC4R(const Npp8u * pSrc1, int nSrc1Step, Npp8u nAlpha1, const Npp8u * pSrc2, int nSrc2Step, Npp8u nAlpha2, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, Npp8u nColorKeyConst[4], NppiAlphaOp nppAlphaOp) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiAlphaCompColorKey_8u_AC4R_Ctx(pSrc1, nSrc1Step, nAlpha1, pSrc2, nSrc2Step, nAlpha2, pDst, nDstStep, oSizeROI, nColorKeyConst, nppAlphaOp, nppStreamCtx);
}
