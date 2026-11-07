// Generated API wrappers -- do not edit by hand.
// family: nppiColorTwistBatch32fC


#include "npp.h"
#include "nppi_color_twist_executor.h"

NppStatus nppiColorTwistBatch32fC_16f_C4IR_Ctx(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI, NppiColorTwistBatchCXR * pBatchList, int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::colortwist::batchTwist<Npp16f, 4, 0, true, true>(nMin, nMax, oSizeROI, pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiColorTwistBatch32fC_16f_C4R_Ctx(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI, NppiColorTwistBatchCXR * pBatchList, int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::colortwist::batchTwist<Npp16f, 4, 0, false, true>(nMin, nMax, oSizeROI, pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiColorTwistBatch32fC_8u_C4IR_Ctx(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI, NppiColorTwistBatchCXR * pBatchList, int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::colortwist::batchTwist<Npp8u, 4, 0, true, true>(nMin, nMax, oSizeROI, pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiColorTwistBatch32fC_8u_C4R_Ctx(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI, NppiColorTwistBatchCXR * pBatchList, int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::colortwist::batchTwist<Npp8u, 4, 0, false, true>(nMin, nMax, oSizeROI, pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiColorTwistBatch32fC_16f_C4IR(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI, NppiColorTwistBatchCXR * pBatchList, int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwistBatch32fC_16f_C4IR_Ctx(nMin, nMax, oSizeROI, pBatchList, nBatchSize, nppStreamCtx);
}

NppStatus nppiColorTwistBatch32fC_16f_C4R(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI, NppiColorTwistBatchCXR * pBatchList, int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwistBatch32fC_16f_C4R_Ctx(nMin, nMax, oSizeROI, pBatchList, nBatchSize, nppStreamCtx);
}

NppStatus nppiColorTwistBatch32fC_8u_C4IR(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI, NppiColorTwistBatchCXR * pBatchList, int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwistBatch32fC_8u_C4IR_Ctx(nMin, nMax, oSizeROI, pBatchList, nBatchSize, nppStreamCtx);
}

NppStatus nppiColorTwistBatch32fC_8u_C4R(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI, NppiColorTwistBatchCXR * pBatchList, int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwistBatch32fC_8u_C4R_Ctx(nMin, nMax, oSizeROI, pBatchList, nBatchSize, nppStreamCtx);
}
