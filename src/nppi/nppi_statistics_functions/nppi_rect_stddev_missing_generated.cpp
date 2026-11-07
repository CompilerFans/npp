// Generated API wrappers -- do not edit by hand.
// family: nppiRectStdDev


#include "npp.h"
#include "nppi_quality_metrics_executor.h"

NppStatus nppiRectStdDev_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, const Npp64f *pSqr, int nSqrStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI, NppiRect oRect, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runRectStdDevF32(pSrc, nSrcStep, pSqr, nSqrStep, pDst, nDstStep, oSizeROI, oRect, nppStreamCtx.hStream);
}

NppStatus nppiRectStdDev_32s_C1RSfs_Ctx(const Npp32s *pSrc, int nSrcStep, const Npp32s *pSqr, int nSqrStep, Npp32s *pDst, int nDstStep, NppiSize oSizeROI, NppiRect oRect, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runRectStdDevS32(pSrc, nSrcStep, pSqr, nSqrStep, pDst, nDstStep, oSizeROI, oRect, nScaleFactor, nppStreamCtx.hStream);
}

NppStatus nppiRectStdDev_32f_C1R(const Npp32f *pSrc, int nSrcStep, const Npp64f *pSqr, int nSqrStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI, NppiRect oRect) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiRectStdDev_32f_C1R_Ctx(pSrc, nSrcStep, pSqr, nSqrStep, pDst, nDstStep, oSizeROI, oRect, nppStreamCtx);
}

NppStatus nppiRectStdDev_32s_C1RSfs(const Npp32s *pSrc, int nSrcStep, const Npp32s *pSqr, int nSqrStep, Npp32s *pDst, int nDstStep, NppiSize oSizeROI, NppiRect oRect, int nScaleFactor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiRectStdDev_32s_C1RSfs_Ctx(pSrc, nSrcStep, pSqr, nSqrStep, pDst, nDstStep, oSizeROI, oRect, nScaleFactor, nppStreamCtx);
}
