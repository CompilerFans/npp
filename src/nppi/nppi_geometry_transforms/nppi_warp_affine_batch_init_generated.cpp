// Generated API wrappers -- do not edit by hand.
// family: nppiWarpAffineBatchInit


#include "npp.h"
#include "nppi_warp_affine_executor.h"

NppStatus nppiWarpAffineBatchInit_Ctx(NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::geometry::warpAffineBatchInit(pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiWarpAffineBatchInit(NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpAffineBatchInit_Ctx(pBatchList, nBatchSize, nppStreamCtx);
}
