// Generated API wrappers -- do not edit by hand.
// family: nppiFilterGaussPyramidLayerDownBorder


#include "npp.h"
#include "nppi_gauss_pyramid_executor.h"

NppStatus nppiFilterGaussPyramidLayerDownBorder_16u_C1R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, Npp32f nRate, const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::filter::gaussPyramidBorder<Npp16u, Npp16u, 1, nppi::filter::kPyramidDown>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, nRate, nFilterTaps, pKernel, eBorderType, nppStreamCtx.hStream);
}

NppStatus nppiFilterGaussPyramidLayerDownBorder_16u_C3R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, Npp32f nRate, const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::filter::gaussPyramidBorder<Npp16u, Npp16u, 3, nppi::filter::kPyramidDown>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, nRate, nFilterTaps, pKernel, eBorderType, nppStreamCtx.hStream);
}

NppStatus nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, Npp32f nRate, const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::filter::gaussPyramidBorder<Npp32f, Npp32f, 1, nppi::filter::kPyramidDown>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, nRate, nFilterTaps, pKernel, eBorderType, nppStreamCtx.hStream);
}

NppStatus nppiFilterGaussPyramidLayerDownBorder_32f_C3R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, Npp32f nRate, const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::filter::gaussPyramidBorder<Npp32f, Npp32f, 3, nppi::filter::kPyramidDown>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, nRate, nFilterTaps, pKernel, eBorderType, nppStreamCtx.hStream);
}

NppStatus nppiFilterGaussPyramidLayerDownBorder_8u_C1R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, Npp32f nRate, const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::filter::gaussPyramidBorder<Npp8u, Npp8u, 1, nppi::filter::kPyramidDown>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, nRate, nFilterTaps, pKernel, eBorderType, nppStreamCtx.hStream);
}

NppStatus nppiFilterGaussPyramidLayerDownBorder_8u_C3R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, Npp32f nRate, const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::filter::gaussPyramidBorder<Npp8u, Npp8u, 3, nppi::filter::kPyramidDown>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, nRate, nFilterTaps, pKernel, eBorderType, nppStreamCtx.hStream);
}

NppStatus nppiFilterGaussPyramidLayerDownBorder_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, Npp32f nRate, const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterGaussPyramidLayerDownBorder_16u_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, nRate, nFilterTaps, pKernel, eBorderType, nppStreamCtx);
}

NppStatus nppiFilterGaussPyramidLayerDownBorder_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, Npp32f nRate, const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterGaussPyramidLayerDownBorder_16u_C3R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, nRate, nFilterTaps, pKernel, eBorderType, nppStreamCtx);
}

NppStatus nppiFilterGaussPyramidLayerDownBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, Npp32f nRate, const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, nRate, nFilterTaps, pKernel, eBorderType, nppStreamCtx);
}

NppStatus nppiFilterGaussPyramidLayerDownBorder_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, Npp32f nRate, const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterGaussPyramidLayerDownBorder_32f_C3R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, nRate, nFilterTaps, pKernel, eBorderType, nppStreamCtx);
}

NppStatus nppiFilterGaussPyramidLayerDownBorder_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, Npp32f nRate, const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterGaussPyramidLayerDownBorder_8u_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, nRate, nFilterTaps, pKernel, eBorderType, nppStreamCtx);
}

NppStatus nppiFilterGaussPyramidLayerDownBorder_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, Npp32f nRate, const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterGaussPyramidLayerDownBorder_8u_C3R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, nRate, nFilterTaps, pKernel, eBorderType, nppStreamCtx);
}
