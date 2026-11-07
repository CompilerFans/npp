// Generated API wrappers -- do not edit by hand.
// family: nppiColorTwistBatch_32fC


#include "npp.h"
#include "nppi_color_twist_executor.h"

NppStatus nppiColorTwistBatch_32fC_C4IR_Ctx(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI, NppiColorTwistBatchCXR * pBatchList, int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::colortwist::batchTwist<Npp32f, 4, 0, true, true>(nMin, nMax, oSizeROI, pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiColorTwistBatch_32fC_C4R_Ctx(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI, NppiColorTwistBatchCXR * pBatchList, int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::colortwist::batchTwist<Npp32f, 4, 0, false, true>(nMin, nMax, oSizeROI, pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiColorTwistBatch_32fC_C4IR(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI, NppiColorTwistBatchCXR * pBatchList, int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwistBatch_32fC_C4IR_Ctx(nMin, nMax, oSizeROI, pBatchList, nBatchSize, nppStreamCtx);
}

NppStatus nppiColorTwistBatch_32fC_C4R(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI, NppiColorTwistBatchCXR * pBatchList, int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwistBatch_32fC_C4R_Ctx(nMin, nMax, oSizeROI, pBatchList, nBatchSize, nppStreamCtx);
}
