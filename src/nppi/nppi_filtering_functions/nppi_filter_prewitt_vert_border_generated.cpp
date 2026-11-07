// Generated API wrappers -- do not edit by hand.
// family: nppiFilterPrewittVertBorder


#include "npp.h"
#include "nppi_filter_derivatives_executor.h"

NppStatus nppiFilterPrewittVertBorder_16s_AC4R_Ctx(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterDerivBorder<Npp16s, Npp16s, 4, 1, nppi::filter::kPrewittVert>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx.hStream, true);
}

NppStatus nppiFilterPrewittVertBorder_16s_C1R_Ctx(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterDerivBorder<Npp16s, Npp16s, 1, 0, nppi::filter::kPrewittVert>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx.hStream, true);
}

NppStatus nppiFilterPrewittVertBorder_16s_C3R_Ctx(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterDerivBorder<Npp16s, Npp16s, 3, 0, nppi::filter::kPrewittVert>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx.hStream, true);
}

NppStatus nppiFilterPrewittVertBorder_16s_C4R_Ctx(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterDerivBorder<Npp16s, Npp16s, 4, 0, nppi::filter::kPrewittVert>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx.hStream, true);
}

NppStatus nppiFilterPrewittVertBorder_32f_AC4R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterDerivBorder<Npp32f, Npp32f, 4, 1, nppi::filter::kPrewittVert>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx.hStream, true);
}

NppStatus nppiFilterPrewittVertBorder_32f_C1R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterDerivBorder<Npp32f, Npp32f, 1, 0, nppi::filter::kPrewittVert>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx.hStream, true);
}

NppStatus nppiFilterPrewittVertBorder_32f_C3R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterDerivBorder<Npp32f, Npp32f, 3, 0, nppi::filter::kPrewittVert>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx.hStream, true);
}

NppStatus nppiFilterPrewittVertBorder_32f_C4R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterDerivBorder<Npp32f, Npp32f, 4, 0, nppi::filter::kPrewittVert>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx.hStream, true);
}

NppStatus nppiFilterPrewittVertBorder_8u_AC4R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterDerivBorder<Npp8u, Npp8u, 4, 1, nppi::filter::kPrewittVert>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx.hStream, true);
}

NppStatus nppiFilterPrewittVertBorder_8u_C1R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterDerivBorder<Npp8u, Npp8u, 1, 0, nppi::filter::kPrewittVert>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx.hStream, true);
}

NppStatus nppiFilterPrewittVertBorder_8u_C3R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterDerivBorder<Npp8u, Npp8u, 3, 0, nppi::filter::kPrewittVert>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx.hStream, true);
}

NppStatus nppiFilterPrewittVertBorder_8u_C4R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterDerivBorder<Npp8u, Npp8u, 4, 0, nppi::filter::kPrewittVert>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx.hStream, true);
}

NppStatus nppiFilterPrewittVertBorder_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterPrewittVertBorder_16s_AC4R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx);
}

NppStatus nppiFilterPrewittVertBorder_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterPrewittVertBorder_16s_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx);
}

NppStatus nppiFilterPrewittVertBorder_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterPrewittVertBorder_16s_C3R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx);
}

NppStatus nppiFilterPrewittVertBorder_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterPrewittVertBorder_16s_C4R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx);
}

NppStatus nppiFilterPrewittVertBorder_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterPrewittVertBorder_32f_AC4R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx);
}

NppStatus nppiFilterPrewittVertBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterPrewittVertBorder_32f_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx);
}

NppStatus nppiFilterPrewittVertBorder_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterPrewittVertBorder_32f_C3R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx);
}

NppStatus nppiFilterPrewittVertBorder_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterPrewittVertBorder_32f_C4R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx);
}

NppStatus nppiFilterPrewittVertBorder_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterPrewittVertBorder_8u_AC4R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx);
}

NppStatus nppiFilterPrewittVertBorder_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterPrewittVertBorder_8u_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx);
}

NppStatus nppiFilterPrewittVertBorder_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterPrewittVertBorder_8u_C3R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx);
}

NppStatus nppiFilterPrewittVertBorder_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterPrewittVertBorder_8u_C4R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx);
}
