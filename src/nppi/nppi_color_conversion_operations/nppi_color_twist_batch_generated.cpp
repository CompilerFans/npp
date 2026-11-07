// Generated API wrappers -- do not edit by hand.
// family: nppiColorTwistBatch


#include "npp.h"
#include "nppi_color_twist_executor.h"

NppStatus nppiColorTwistBatch_32f_AC4IR_Ctx(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI, NppiColorTwistBatchCXR * pBatchList, int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::colortwist::batchTwist<Npp32f, 4, 1, true, false>(nMin, nMax, oSizeROI, pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiColorTwistBatch_32f_AC4R_Ctx(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI, NppiColorTwistBatchCXR * pBatchList, int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::colortwist::batchTwist<Npp32f, 4, 1, false, false>(nMin, nMax, oSizeROI, pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiColorTwistBatch_32f_C1IR_Ctx(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI, NppiColorTwistBatchCXR * pBatchList, int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::colortwist::batchTwist<Npp32f, 1, 0, true, false>(nMin, nMax, oSizeROI, pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiColorTwistBatch_32f_C1R_Ctx(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI, NppiColorTwistBatchCXR * pBatchList, int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::colortwist::batchTwist<Npp32f, 1, 0, false, false>(nMin, nMax, oSizeROI, pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiColorTwistBatch_32f_C3IR_Ctx(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI, NppiColorTwistBatchCXR * pBatchList, int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::colortwist::batchTwist<Npp32f, 3, 0, true, false>(nMin, nMax, oSizeROI, pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiColorTwistBatch_32f_C3R_Ctx(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI, NppiColorTwistBatchCXR * pBatchList, int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::colortwist::batchTwist<Npp32f, 3, 0, false, false>(nMin, nMax, oSizeROI, pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiColorTwistBatch_32f_C4IR_Ctx(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI, NppiColorTwistBatchCXR * pBatchList, int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::colortwist::batchTwist<Npp32f, 4, 0, true, false>(nMin, nMax, oSizeROI, pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiColorTwistBatch_32f_C4R_Ctx(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI, NppiColorTwistBatchCXR * pBatchList, int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::colortwist::batchTwist<Npp32f, 4, 0, false, false>(nMin, nMax, oSizeROI, pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiColorTwistBatch_32f_AC4IR(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI, NppiColorTwistBatchCXR * pBatchList, int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwistBatch_32f_AC4IR_Ctx(nMin, nMax, oSizeROI, pBatchList, nBatchSize, nppStreamCtx);
}

NppStatus nppiColorTwistBatch_32f_AC4R(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI, NppiColorTwistBatchCXR * pBatchList, int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwistBatch_32f_AC4R_Ctx(nMin, nMax, oSizeROI, pBatchList, nBatchSize, nppStreamCtx);
}

NppStatus nppiColorTwistBatch_32f_C1IR(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI, NppiColorTwistBatchCXR * pBatchList, int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwistBatch_32f_C1IR_Ctx(nMin, nMax, oSizeROI, pBatchList, nBatchSize, nppStreamCtx);
}

NppStatus nppiColorTwistBatch_32f_C1R(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI, NppiColorTwistBatchCXR * pBatchList, int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwistBatch_32f_C1R_Ctx(nMin, nMax, oSizeROI, pBatchList, nBatchSize, nppStreamCtx);
}

NppStatus nppiColorTwistBatch_32f_C3IR(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI, NppiColorTwistBatchCXR * pBatchList, int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwistBatch_32f_C3IR_Ctx(nMin, nMax, oSizeROI, pBatchList, nBatchSize, nppStreamCtx);
}

NppStatus nppiColorTwistBatch_32f_C3R(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI, NppiColorTwistBatchCXR * pBatchList, int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwistBatch_32f_C3R_Ctx(nMin, nMax, oSizeROI, pBatchList, nBatchSize, nppStreamCtx);
}

NppStatus nppiColorTwistBatch_32f_C4IR(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI, NppiColorTwistBatchCXR * pBatchList, int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwistBatch_32f_C4IR_Ctx(nMin, nMax, oSizeROI, pBatchList, nBatchSize, nppStreamCtx);
}

NppStatus nppiColorTwistBatch_32f_C4R(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI, NppiColorTwistBatchCXR * pBatchList, int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwistBatch_32f_C4R_Ctx(nMin, nMax, oSizeROI, pBatchList, nBatchSize, nppStreamCtx);
}
