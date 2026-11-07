// Generated API wrappers -- do not edit by hand.
// family: nppiPSNRBatch


#include "npp.h"
#include "nppi_quality_metrics_executor.h"

NppStatus nppiPSNRBatch_8u_C1R_Advanced_Ctx(const NppiImageDescriptor * pSrc1BatchList, const NppiImageDescriptor * pSrc2BatchList, int nBatchSize, NppiSize oMaxSizeROI, Npp32f * pPSNR, NppiBufferDescriptor * pDeviceBufferList, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runPSNRBatch(pSrc1BatchList, pSrc2BatchList, nBatchSize, oMaxSizeROI, 1, pPSNR, pDeviceBufferList, true, nppStreamCtx.hStream);
}

NppStatus nppiPSNRBatch_8u_C1R_Ctx(const NppiImageDescriptor * pSrc1BatchList, const NppiImageDescriptor * pSrc2BatchList, int nBatchSize, NppiSize oSizeROI, Npp32f * pPSNR, NppiBufferDescriptor * pDeviceBufferList, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runPSNRBatch(pSrc1BatchList, pSrc2BatchList, nBatchSize, oSizeROI, 1, pPSNR, pDeviceBufferList, false, nppStreamCtx.hStream);
}

NppStatus nppiPSNRBatch_8u_C3R_Advanced_Ctx(const NppiImageDescriptor * pSrc1BatchList, const NppiImageDescriptor * pSrc2BatchList, int nBatchSize, NppiSize oMaxSizeROI, Npp32f * pPSNR, NppiBufferDescriptor * pDeviceBufferList, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runPSNRBatch(pSrc1BatchList, pSrc2BatchList, nBatchSize, oMaxSizeROI, 3, pPSNR, pDeviceBufferList, true, nppStreamCtx.hStream);
}

NppStatus nppiPSNRBatch_8u_C3R_Ctx(const NppiImageDescriptor * pSrc1BatchList, const NppiImageDescriptor * pSrc2BatchList, int nBatchSize, NppiSize oSizeROI, Npp32f * pPSNR, NppiBufferDescriptor * pDeviceBufferList, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runPSNRBatch(pSrc1BatchList, pSrc2BatchList, nBatchSize, oSizeROI, 3, pPSNR, pDeviceBufferList, false, nppStreamCtx.hStream);
}

NppStatus nppiPSNRBatch_8u_C1R(const NppiImageDescriptor * pSrc1BatchList, const NppiImageDescriptor * pSrc2BatchList, int nBatchSize, NppiSize oSizeROI, Npp32f * pPSNR, NppiBufferDescriptor * pDeviceBufferList) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiPSNRBatch_8u_C1R_Ctx(pSrc1BatchList, pSrc2BatchList, nBatchSize, oSizeROI, pPSNR, pDeviceBufferList, nppStreamCtx);
}

NppStatus nppiPSNRBatch_8u_C1R_Advanced(const NppiImageDescriptor * pSrc1BatchList, const NppiImageDescriptor * pSrc2BatchList, int nBatchSize, NppiSize oMaxSizeROI, Npp32f * pPSNR, NppiBufferDescriptor * pDeviceBufferList) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiPSNRBatch_8u_C1R_Advanced_Ctx(pSrc1BatchList, pSrc2BatchList, nBatchSize, oMaxSizeROI, pPSNR, pDeviceBufferList, nppStreamCtx);
}

NppStatus nppiPSNRBatch_8u_C3R(const NppiImageDescriptor * pSrc1BatchList, const NppiImageDescriptor * pSrc2BatchList, int nBatchSize, NppiSize oSizeROI, Npp32f * pPSNR, NppiBufferDescriptor * pDeviceBufferList) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiPSNRBatch_8u_C3R_Ctx(pSrc1BatchList, pSrc2BatchList, nBatchSize, oSizeROI, pPSNR, pDeviceBufferList, nppStreamCtx);
}

NppStatus nppiPSNRBatch_8u_C3R_Advanced(const NppiImageDescriptor * pSrc1BatchList, const NppiImageDescriptor * pSrc2BatchList, int nBatchSize, NppiSize oMaxSizeROI, Npp32f * pPSNR, NppiBufferDescriptor * pDeviceBufferList) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiPSNRBatch_8u_C3R_Advanced_Ctx(pSrc1BatchList, pSrc2BatchList, nBatchSize, oMaxSizeROI, pPSNR, pDeviceBufferList, nppStreamCtx);
}
