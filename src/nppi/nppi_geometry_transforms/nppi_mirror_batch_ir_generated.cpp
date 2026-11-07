// Generated API wrappers -- do not edit by hand.
// family: nppiMirrorBatchIR


#include "npp.h"
#include "nppi_mirror_executor.h"

NppStatus nppiMirrorBatch_32f_AC4IR_Ctx(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR * pBatchList, int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::mirror::mirrorBatchInPlace<Npp32f, 4, 1>(oSizeROI, flip, pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiMirrorBatch_32f_C1IR_Ctx(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR * pBatchList, int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::mirror::mirrorBatchInPlace<Npp32f, 1, 0>(oSizeROI, flip, pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiMirrorBatch_32f_C3IR_Ctx(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR * pBatchList, int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::mirror::mirrorBatchInPlace<Npp32f, 3, 0>(oSizeROI, flip, pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiMirrorBatch_32f_C4IR_Ctx(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR * pBatchList, int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::mirror::mirrorBatchInPlace<Npp32f, 4, 0>(oSizeROI, flip, pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiMirrorBatch_32f_AC4IR(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR * pBatchList, int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiMirrorBatch_32f_AC4IR_Ctx(oSizeROI, flip, pBatchList, nBatchSize, nppStreamCtx);
}

NppStatus nppiMirrorBatch_32f_C1IR(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR * pBatchList, int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiMirrorBatch_32f_C1IR_Ctx(oSizeROI, flip, pBatchList, nBatchSize, nppStreamCtx);
}

NppStatus nppiMirrorBatch_32f_C3IR(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR * pBatchList, int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiMirrorBatch_32f_C3IR_Ctx(oSizeROI, flip, pBatchList, nBatchSize, nppStreamCtx);
}

NppStatus nppiMirrorBatch_32f_C4IR(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR * pBatchList, int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiMirrorBatch_32f_C4IR_Ctx(oSizeROI, flip, pBatchList, nBatchSize, nppStreamCtx);
}
