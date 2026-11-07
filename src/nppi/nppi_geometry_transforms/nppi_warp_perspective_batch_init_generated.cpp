// Generated API wrappers -- do not edit by hand.
// family: nppiWarpPerspectiveBatchInit


#include "npp.h"
#include "nppi_warp_perspective_executor.h"

NppStatus nppiWarpPerspectiveBatchInit_Ctx(NppiWarpPerspectiveBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::warp::batchInit(pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiWarpPerspectiveBatchInit(NppiWarpPerspectiveBatchCXR * pBatchList, unsigned int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpPerspectiveBatchInit_Ctx(pBatchList, nBatchSize, nppStreamCtx);
}
