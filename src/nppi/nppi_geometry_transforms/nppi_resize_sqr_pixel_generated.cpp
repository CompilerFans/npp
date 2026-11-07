// Generated API wrappers -- do not edit by hand.
// family: nppiResizeSqrPixel


#include "npp.h"
#include "nppi_resize_executor.h"

NppStatus nppiResizeSqrPixel_16s_AC4R_Ctx(const Npp16s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16s * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::resize::sqrPixel<Npp16s, 4, 1>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiResizeSqrPixel_16s_C1R_Ctx(const Npp16s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16s * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::resize::sqrPixel<Npp16s, 1, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiResizeSqrPixel_16s_C3R_Ctx(const Npp16s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16s * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::resize::sqrPixel<Npp16s, 3, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiResizeSqrPixel_16s_C4R_Ctx(const Npp16s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16s * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::resize::sqrPixel<Npp16s, 4, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiResizeSqrPixel_16s_P3R_Ctx(const Npp16s * const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16s * pDst[3], int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::resize::sqrPixelPlanar<Npp16s, 3>(pSrc, nSrcStep, oSrcSize, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiResizeSqrPixel_16s_P4R_Ctx(const Npp16s * const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16s * pDst[4], int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::resize::sqrPixelPlanar<Npp16s, 4>(pSrc, nSrcStep, oSrcSize, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiResizeSqrPixel_16u_AC4R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16u * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::resize::sqrPixel<Npp16u, 4, 1>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiResizeSqrPixel_16u_C1R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16u * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::resize::sqrPixel<Npp16u, 1, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiResizeSqrPixel_16u_C3R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16u * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::resize::sqrPixel<Npp16u, 3, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiResizeSqrPixel_16u_C4R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16u * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::resize::sqrPixel<Npp16u, 4, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiResizeSqrPixel_16u_P3R_Ctx(const Npp16u * const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16u * pDst[3], int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::resize::sqrPixelPlanar<Npp16u, 3>(pSrc, nSrcStep, oSrcSize, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiResizeSqrPixel_16u_P4R_Ctx(const Npp16u * const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16u * pDst[4], int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::resize::sqrPixelPlanar<Npp16u, 4>(pSrc, nSrcStep, oSrcSize, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiResizeSqrPixel_32f_AC4R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32f * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::resize::sqrPixel<Npp32f, 4, 1>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiResizeSqrPixel_32f_C1R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32f * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::resize::sqrPixel<Npp32f, 1, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiResizeSqrPixel_32f_C3R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32f * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::resize::sqrPixel<Npp32f, 3, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiResizeSqrPixel_32f_C4R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32f * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::resize::sqrPixel<Npp32f, 4, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiResizeSqrPixel_32f_P3R_Ctx(const Npp32f * const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32f * pDst[3], int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::resize::sqrPixelPlanar<Npp32f, 3>(pSrc, nSrcStep, oSrcSize, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiResizeSqrPixel_32f_P4R_Ctx(const Npp32f * const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32f * pDst[4], int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::resize::sqrPixelPlanar<Npp32f, 4>(pSrc, nSrcStep, oSrcSize, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiResizeSqrPixel_64f_AC4R_Ctx(const Npp64f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp64f * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::resize::sqrPixel<Npp64f, 4, 1>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiResizeSqrPixel_64f_C1R_Ctx(const Npp64f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp64f * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::resize::sqrPixel<Npp64f, 1, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiResizeSqrPixel_64f_C3R_Ctx(const Npp64f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp64f * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::resize::sqrPixel<Npp64f, 3, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiResizeSqrPixel_64f_C4R_Ctx(const Npp64f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp64f * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::resize::sqrPixel<Npp64f, 4, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiResizeSqrPixel_64f_P3R_Ctx(const Npp64f * const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp64f * pDst[3], int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::resize::sqrPixelPlanar<Npp64f, 3>(pSrc, nSrcStep, oSrcSize, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiResizeSqrPixel_64f_P4R_Ctx(const Npp64f * const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp64f * pDst[4], int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::resize::sqrPixelPlanar<Npp64f, 4>(pSrc, nSrcStep, oSrcSize, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiResizeSqrPixel_8u_AC4R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::resize::sqrPixel<Npp8u, 4, 1>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiResizeSqrPixel_8u_C1R_Advanced_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, Npp8u * pBuffer, int eInterpolationMode, NppStreamContext nppStreamCtx) {
  return nppi::resize::sqrPixelAdvanced<Npp8u, 1, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, pBuffer, eInterpolationMode, nppStreamCtx.hStream);
}

NppStatus nppiResizeSqrPixel_8u_C1R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::resize::sqrPixel<Npp8u, 1, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiResizeSqrPixel_8u_C3R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::resize::sqrPixel<Npp8u, 3, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiResizeSqrPixel_8u_C4R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::resize::sqrPixel<Npp8u, 4, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiResizeSqrPixel_8u_P3R_Ctx(const Npp8u * const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u * pDst[3], int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::resize::sqrPixelPlanar<Npp8u, 3>(pSrc, nSrcStep, oSrcSize, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiResizeSqrPixel_8u_P4R_Ctx(const Npp8u * const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u * pDst[4], int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::resize::sqrPixelPlanar<Npp8u, 4>(pSrc, nSrcStep, oSrcSize, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiResizeSqrPixel_16s_AC4R(const Npp16s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16s * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiResizeSqrPixel_16s_AC4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx);
}

NppStatus nppiResizeSqrPixel_16s_C1R(const Npp16s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16s * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiResizeSqrPixel_16s_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx);
}

NppStatus nppiResizeSqrPixel_16s_C3R(const Npp16s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16s * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiResizeSqrPixel_16s_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx);
}

NppStatus nppiResizeSqrPixel_16s_C4R(const Npp16s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16s * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiResizeSqrPixel_16s_C4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx);
}

NppStatus nppiResizeSqrPixel_16s_P3R(const Npp16s * const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16s * pDst[3], int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiResizeSqrPixel_16s_P3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx);
}

NppStatus nppiResizeSqrPixel_16s_P4R(const Npp16s * const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16s * pDst[4], int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiResizeSqrPixel_16s_P4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx);
}

NppStatus nppiResizeSqrPixel_16u_AC4R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16u * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiResizeSqrPixel_16u_AC4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx);
}

NppStatus nppiResizeSqrPixel_16u_C1R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16u * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiResizeSqrPixel_16u_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx);
}

NppStatus nppiResizeSqrPixel_16u_C3R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16u * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiResizeSqrPixel_16u_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx);
}

NppStatus nppiResizeSqrPixel_16u_C4R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16u * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiResizeSqrPixel_16u_C4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx);
}

NppStatus nppiResizeSqrPixel_16u_P3R(const Npp16u * const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16u * pDst[3], int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiResizeSqrPixel_16u_P3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx);
}

NppStatus nppiResizeSqrPixel_16u_P4R(const Npp16u * const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16u * pDst[4], int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiResizeSqrPixel_16u_P4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx);
}

NppStatus nppiResizeSqrPixel_32f_AC4R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32f * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiResizeSqrPixel_32f_AC4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx);
}

NppStatus nppiResizeSqrPixel_32f_C1R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32f * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiResizeSqrPixel_32f_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx);
}

NppStatus nppiResizeSqrPixel_32f_C3R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32f * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiResizeSqrPixel_32f_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx);
}

NppStatus nppiResizeSqrPixel_32f_C4R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32f * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiResizeSqrPixel_32f_C4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx);
}

NppStatus nppiResizeSqrPixel_32f_P3R(const Npp32f * const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32f * pDst[3], int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiResizeSqrPixel_32f_P3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx);
}

NppStatus nppiResizeSqrPixel_32f_P4R(const Npp32f * const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32f * pDst[4], int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiResizeSqrPixel_32f_P4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx);
}

NppStatus nppiResizeSqrPixel_64f_AC4R(const Npp64f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp64f * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiResizeSqrPixel_64f_AC4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx);
}

NppStatus nppiResizeSqrPixel_64f_C1R(const Npp64f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp64f * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiResizeSqrPixel_64f_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx);
}

NppStatus nppiResizeSqrPixel_64f_C3R(const Npp64f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp64f * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiResizeSqrPixel_64f_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx);
}

NppStatus nppiResizeSqrPixel_64f_C4R(const Npp64f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp64f * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiResizeSqrPixel_64f_C4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx);
}

NppStatus nppiResizeSqrPixel_64f_P3R(const Npp64f * const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp64f * pDst[3], int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiResizeSqrPixel_64f_P3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx);
}

NppStatus nppiResizeSqrPixel_64f_P4R(const Npp64f * const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp64f * pDst[4], int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiResizeSqrPixel_64f_P4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx);
}

NppStatus nppiResizeSqrPixel_8u_AC4R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiResizeSqrPixel_8u_AC4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx);
}

NppStatus nppiResizeSqrPixel_8u_C1R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiResizeSqrPixel_8u_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx);
}

NppStatus nppiResizeSqrPixel_8u_C1R_Advanced(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, Npp8u * pBuffer, int eInterpolationMode) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiResizeSqrPixel_8u_C1R_Advanced_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, pBuffer, eInterpolationMode, nppStreamCtx);
}

NppStatus nppiResizeSqrPixel_8u_C3R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiResizeSqrPixel_8u_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx);
}

NppStatus nppiResizeSqrPixel_8u_C4R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u * pDst, int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiResizeSqrPixel_8u_C4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx);
}

NppStatus nppiResizeSqrPixel_8u_P3R(const Npp8u * const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u * pDst[3], int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiResizeSqrPixel_8u_P3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx);
}

NppStatus nppiResizeSqrPixel_8u_P4R(const Npp8u * const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u * pDst[4], int nDstStep, NppiRect oDstROI, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiResizeSqrPixel_8u_P4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx);
}
