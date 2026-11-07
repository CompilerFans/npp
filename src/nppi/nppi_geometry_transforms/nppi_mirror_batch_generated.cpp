// Generated API wrappers -- do not edit by hand.
// family: nppiMirrorBatchR


#include "npp.h"
#include "nppi_mirror_executor.h"

NppStatus nppiMirrorBatch_32f_AC4R_Ctx(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR * pBatchList, int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::mirror::mirrorBatch<Npp32f, 4, 1>(oSizeROI, flip, pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiMirrorBatch_32f_C1R_Ctx(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR * pBatchList, int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::mirror::mirrorBatch<Npp32f, 1, 0>(oSizeROI, flip, pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiMirrorBatch_32f_C3R_Ctx(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR * pBatchList, int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::mirror::mirrorBatch<Npp32f, 3, 0>(oSizeROI, flip, pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiMirrorBatch_32f_C4R_Ctx(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR * pBatchList, int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::mirror::mirrorBatch<Npp32f, 4, 0>(oSizeROI, flip, pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiMirrorBatch_32f_AC4R(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR * pBatchList, int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiMirrorBatch_32f_AC4R_Ctx(oSizeROI, flip, pBatchList, nBatchSize, nppStreamCtx);
}

NppStatus nppiMirrorBatch_32f_C1R(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR * pBatchList, int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiMirrorBatch_32f_C1R_Ctx(oSizeROI, flip, pBatchList, nBatchSize, nppStreamCtx);
}

NppStatus nppiMirrorBatch_32f_C3R(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR * pBatchList, int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiMirrorBatch_32f_C3R_Ctx(oSizeROI, flip, pBatchList, nBatchSize, nppStreamCtx);
}

NppStatus nppiMirrorBatch_32f_C4R(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR * pBatchList, int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiMirrorBatch_32f_C4R_Ctx(oSizeROI, flip, pBatchList, nBatchSize, nppStreamCtx);
}
