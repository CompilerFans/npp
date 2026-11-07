// Generated API wrappers -- do not edit by hand.
// family: nppiWarpAffineBatch


#include "npp.h"
#include "nppi_warp_affine_executor.h"

NppStatus nppiWarpAffineBatch_16f_C1R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::geometry::warpAffineBatch<Npp16f, 1, 0>(oSmallestSrcSize, oSrcRectROI, oDstRectROI, eInterpolation, pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiWarpAffineBatch_16f_C3R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::geometry::warpAffineBatch<Npp16f, 3, 0>(oSmallestSrcSize, oSrcRectROI, oDstRectROI, eInterpolation, pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiWarpAffineBatch_16f_C4R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::geometry::warpAffineBatch<Npp16f, 4, 0>(oSmallestSrcSize, oSrcRectROI, oDstRectROI, eInterpolation, pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiWarpAffineBatch_32f_AC4R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::geometry::warpAffineBatch<Npp32f, 4, 1>(oSmallestSrcSize, oSrcRectROI, oDstRectROI, eInterpolation, pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiWarpAffineBatch_32f_C1R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::geometry::warpAffineBatch<Npp32f, 1, 0>(oSmallestSrcSize, oSrcRectROI, oDstRectROI, eInterpolation, pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiWarpAffineBatch_32f_C3R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::geometry::warpAffineBatch<Npp32f, 3, 0>(oSmallestSrcSize, oSrcRectROI, oDstRectROI, eInterpolation, pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiWarpAffineBatch_32f_C4R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::geometry::warpAffineBatch<Npp32f, 4, 0>(oSmallestSrcSize, oSrcRectROI, oDstRectROI, eInterpolation, pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiWarpAffineBatch_8u_AC4R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::geometry::warpAffineBatch<Npp8u, 4, 1>(oSmallestSrcSize, oSrcRectROI, oDstRectROI, eInterpolation, pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiWarpAffineBatch_8u_C1R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::geometry::warpAffineBatch<Npp8u, 1, 0>(oSmallestSrcSize, oSrcRectROI, oDstRectROI, eInterpolation, pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiWarpAffineBatch_8u_C3R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::geometry::warpAffineBatch<Npp8u, 3, 0>(oSmallestSrcSize, oSrcRectROI, oDstRectROI, eInterpolation, pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiWarpAffineBatch_8u_C4R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx) {
  return nppi::geometry::warpAffineBatch<Npp8u, 4, 0>(oSmallestSrcSize, oSrcRectROI, oDstRectROI, eInterpolation, pBatchList, nBatchSize, nppStreamCtx.hStream);
}

NppStatus nppiWarpAffineBatch_16f_C1R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpAffineBatch_16f_C1R_Ctx(oSmallestSrcSize, oSrcRectROI, oDstRectROI, eInterpolation, pBatchList, nBatchSize, nppStreamCtx);
}

NppStatus nppiWarpAffineBatch_16f_C3R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpAffineBatch_16f_C3R_Ctx(oSmallestSrcSize, oSrcRectROI, oDstRectROI, eInterpolation, pBatchList, nBatchSize, nppStreamCtx);
}

NppStatus nppiWarpAffineBatch_16f_C4R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpAffineBatch_16f_C4R_Ctx(oSmallestSrcSize, oSrcRectROI, oDstRectROI, eInterpolation, pBatchList, nBatchSize, nppStreamCtx);
}

NppStatus nppiWarpAffineBatch_32f_AC4R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpAffineBatch_32f_AC4R_Ctx(oSmallestSrcSize, oSrcRectROI, oDstRectROI, eInterpolation, pBatchList, nBatchSize, nppStreamCtx);
}

NppStatus nppiWarpAffineBatch_32f_C1R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpAffineBatch_32f_C1R_Ctx(oSmallestSrcSize, oSrcRectROI, oDstRectROI, eInterpolation, pBatchList, nBatchSize, nppStreamCtx);
}

NppStatus nppiWarpAffineBatch_32f_C3R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpAffineBatch_32f_C3R_Ctx(oSmallestSrcSize, oSrcRectROI, oDstRectROI, eInterpolation, pBatchList, nBatchSize, nppStreamCtx);
}

NppStatus nppiWarpAffineBatch_32f_C4R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpAffineBatch_32f_C4R_Ctx(oSmallestSrcSize, oSrcRectROI, oDstRectROI, eInterpolation, pBatchList, nBatchSize, nppStreamCtx);
}

NppStatus nppiWarpAffineBatch_8u_AC4R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpAffineBatch_8u_AC4R_Ctx(oSmallestSrcSize, oSrcRectROI, oDstRectROI, eInterpolation, pBatchList, nBatchSize, nppStreamCtx);
}

NppStatus nppiWarpAffineBatch_8u_C1R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpAffineBatch_8u_C1R_Ctx(oSmallestSrcSize, oSrcRectROI, oDstRectROI, eInterpolation, pBatchList, nBatchSize, nppStreamCtx);
}

NppStatus nppiWarpAffineBatch_8u_C3R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpAffineBatch_8u_C3R_Ctx(oSmallestSrcSize, oSrcRectROI, oDstRectROI, eInterpolation, pBatchList, nBatchSize, nppStreamCtx);
}

NppStatus nppiWarpAffineBatch_8u_C4R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpAffineBatch_8u_C4R_Ctx(oSmallestSrcSize, oSrcRectROI, oDstRectROI, eInterpolation, pBatchList, nBatchSize, nppStreamCtx);
}
