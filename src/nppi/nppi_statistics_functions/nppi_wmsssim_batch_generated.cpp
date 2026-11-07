// Generated API wrappers -- do not edit by hand.
// family: nppiWMSSSIMBatch


#include "npp.h"
#include "nppi_ssim_executor.h"

NppStatus nppiWMSSSIMBatch_8u_C1R_Advanced_Ctx(const NppiImageDescriptor * pSrc1BatchList, const NppiImageDescriptor * pSrc2BatchList, int nBatchSize, NppiSize oMaxSizeROI, Npp32f * pWMSSSIM, NppiBufferDescriptor * pDeviceBufferList, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runWMSSSIMBatch(pSrc1BatchList, pSrc2BatchList, nBatchSize, oMaxSizeROI, 1, pWMSSSIM, true, nppStreamCtx.hStream);
}

NppStatus nppiWMSSSIMBatch_8u_C1R_Ctx(const NppiImageDescriptor * pSrc1BatchList, const NppiImageDescriptor * pSrc2BatchList, int nBatchSize, NppiSize oSizeROI, Npp32f * pWMSSSIM, NppiBufferDescriptor * pDeviceBufferList, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runWMSSSIMBatch(pSrc1BatchList, pSrc2BatchList, nBatchSize, oSizeROI, 1, pWMSSSIM, false, nppStreamCtx.hStream);
}

NppStatus nppiWMSSSIMBatch_8u_C3R_Advanced_Ctx(const NppiImageDescriptor * pSrc1BatchList, const NppiImageDescriptor * pSrc2BatchList, int nBatchSize, NppiSize oMaxSizeROI, Npp32f * pWMSSSIM, NppiBufferDescriptor * pDeviceBufferList, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runWMSSSIMBatch(pSrc1BatchList, pSrc2BatchList, nBatchSize, oMaxSizeROI, 3, pWMSSSIM, true, nppStreamCtx.hStream);
}

NppStatus nppiWMSSSIMBatch_8u_C3R_Ctx(const NppiImageDescriptor * pSrc1BatchList, const NppiImageDescriptor * pSrc2BatchList, int nBatchSize, NppiSize oSizeROI, Npp32f * pWMSSSIM, NppiBufferDescriptor * pDeviceBufferList, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runWMSSSIMBatch(pSrc1BatchList, pSrc2BatchList, nBatchSize, oSizeROI, 3, pWMSSSIM, false, nppStreamCtx.hStream);
}

NppStatus nppiWMSSSIMBatch_8u_C1R(const NppiImageDescriptor * pSrc1BatchList, const NppiImageDescriptor * pSrc2BatchList, int nBatchSize, NppiSize oSizeROI, Npp32f * pWMSSSIM, NppiBufferDescriptor * pDeviceBufferList) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWMSSSIMBatch_8u_C1R_Ctx(pSrc1BatchList, pSrc2BatchList, nBatchSize, oSizeROI, pWMSSSIM, pDeviceBufferList, nppStreamCtx);
}

NppStatus nppiWMSSSIMBatch_8u_C1R_Advanced(const NppiImageDescriptor * pSrc1BatchList, const NppiImageDescriptor * pSrc2BatchList, int nBatchSize, NppiSize oMaxSizeROI, Npp32f * pWMSSSIM, NppiBufferDescriptor * pDeviceBufferList) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWMSSSIMBatch_8u_C1R_Advanced_Ctx(pSrc1BatchList, pSrc2BatchList, nBatchSize, oMaxSizeROI, pWMSSSIM, pDeviceBufferList, nppStreamCtx);
}

NppStatus nppiWMSSSIMBatch_8u_C3R(const NppiImageDescriptor * pSrc1BatchList, const NppiImageDescriptor * pSrc2BatchList, int nBatchSize, NppiSize oSizeROI, Npp32f * pWMSSSIM, NppiBufferDescriptor * pDeviceBufferList) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWMSSSIMBatch_8u_C3R_Ctx(pSrc1BatchList, pSrc2BatchList, nBatchSize, oSizeROI, pWMSSSIM, pDeviceBufferList, nppStreamCtx);
}

NppStatus nppiWMSSSIMBatch_8u_C3R_Advanced(const NppiImageDescriptor * pSrc1BatchList, const NppiImageDescriptor * pSrc2BatchList, int nBatchSize, NppiSize oMaxSizeROI, Npp32f * pWMSSSIM, NppiBufferDescriptor * pDeviceBufferList) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWMSSSIMBatch_8u_C3R_Advanced_Ctx(pSrc1BatchList, pSrc2BatchList, nBatchSize, oMaxSizeROI, pWMSSSIM, pDeviceBufferList, nppStreamCtx);
}

// NPP 13.4 addition: four-channel batch variants (one score per channel per
// image pair).
NppStatus nppiWMSSSIMBatch_8u_C4R_Ctx(const NppiImageDescriptor * pSrc1BatchList, const NppiImageDescriptor * pSrc2BatchList, int nBatchSize, NppiSize oSizeROI, Npp32f * pWMSSSIM, NppiBufferDescriptor * pDeviceBufferList, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runWMSSSIMBatch(pSrc1BatchList, pSrc2BatchList, nBatchSize, oSizeROI, 4, pWMSSSIM, false, nppStreamCtx.hStream);
}

NppStatus nppiWMSSSIMBatch_8u_C4R_Advanced_Ctx(const NppiImageDescriptor * pSrc1BatchList, const NppiImageDescriptor * pSrc2BatchList, int nBatchSize, NppiSize oMaxSizeROI, Npp32f * pWMSSSIM, NppiBufferDescriptor * pDeviceBufferList, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runWMSSSIMBatch(pSrc1BatchList, pSrc2BatchList, nBatchSize, oMaxSizeROI, 4, pWMSSSIM, true, nppStreamCtx.hStream);
}
