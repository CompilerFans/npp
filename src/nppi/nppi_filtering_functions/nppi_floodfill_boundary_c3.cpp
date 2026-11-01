// The public API header declares the 8u/16u C3 non-Ctx Boundary entry points
// twice: once with the scalar (C1) parameter list and once with the documented
// three-channel arrays.  The scalar declaration is a header mistake (NPP 13.4
// only has the array form); the array overloads are implemented here so that
// both the documented call and the _Ctx twin agree.
#include "npp.h"

NppStatus nppiFloodFillBoundary_8u_C3IR(Npp8u *pSrcDst, int nSrcDstStep, NppiPoint oSeed,
                                        const Npp8u aNewValues[3],
                                        const Npp8u aBoundaryValues[3], NppiNorm eNorm,
                                        NppiSize oSizeROI,
                                        NppiConnectedRegion *pConnectedRegion, Npp8u *pBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) return status;
  return nppiFloodFillBoundary_8u_C3IR_Ctx(pSrcDst, nSrcDstStep, oSeed, aNewValues, aBoundaryValues,
                                           eNorm, oSizeROI, pConnectedRegion, pBuffer, nppStreamCtx);
}

NppStatus nppiFloodFillBoundary_16u_C3IR(Npp16u *pSrcDst, int nSrcDstStep, NppiPoint oSeed,
                                         const Npp16u aNewValues[3],
                                         const Npp16u aBoundaryValues[3], NppiNorm eNorm,
                                         NppiSize oSizeROI,
                                         NppiConnectedRegion *pConnectedRegion, Npp8u *pBuffer) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) return status;
  return nppiFloodFillBoundary_16u_C3IR_Ctx(pSrcDst, nSrcDstStep, oSeed, aNewValues,
                                            aBoundaryValues, eNorm, oSizeROI, pConnectedRegion,
                                            pBuffer, nppStreamCtx);
}
