// Generated API wrappers -- do not edit by hand.
// family: nppiYCbCrToRGBBatch


#include "npp.h"
#include "nppi_c3_batch_executor.h"

NppStatus nppiYCbCrToRGBBatch_8u_P3C3R_Advanced_Ctx(const NppiImageDescriptor * const pSrcBatchList[3], NppiImageDescriptor * pDstBatchList, int nBatchSize, NppiSize oMaxSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::c3batch::runPlanar(pSrcBatchList, pDstBatchList, nBatchSize, oMaxSizeROI, nppStreamCtx.hStream, nppi::c3batch::kYCbCrToRGBP3, 1);
}

NppStatus nppiYCbCrToRGBBatch_8u_P3C3R_Ctx(const NppiImageDescriptor * const pSrcBatchList[3], NppiImageDescriptor * pDstBatchList, int nBatchSize, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::c3batch::runPlanar(pSrcBatchList, pDstBatchList, nBatchSize, oSizeROI, nppStreamCtx.hStream, nppi::c3batch::kYCbCrToRGBP3, 0);
}

NppStatus nppiYCbCrToRGBBatch_8u_P3C3R(const NppiImageDescriptor * const pSrcBatchList[3], NppiImageDescriptor * pDstBatchList, int nBatchSize, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiYCbCrToRGBBatch_8u_P3C3R_Ctx(pSrcBatchList, pDstBatchList, nBatchSize, oSizeROI, nppStreamCtx);
}

NppStatus nppiYCbCrToRGBBatch_8u_P3C3R_Advanced(const NppiImageDescriptor * const pSrcBatchList[3], NppiImageDescriptor * pDstBatchList, int nBatchSize, NppiSize oMaxSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiYCbCrToRGBBatch_8u_P3C3R_Advanced_Ctx(pSrcBatchList, pDstBatchList, nBatchSize, oMaxSizeROI, nppStreamCtx);
}
