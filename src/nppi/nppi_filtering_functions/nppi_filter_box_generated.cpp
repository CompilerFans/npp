// Generated API wrappers -- do not edit by hand.
// family: nppiFilterBox


#include "npp.h"
#include "nppi_filter_window_executor.h"

NppStatus nppiFilterBox_16s_AC4R_Ctx(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterBox<Npp16s, 4, 1>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilterBox_16s_C1R_Ctx(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterBox<Npp16s, 1, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilterBox_16s_C3R_Ctx(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterBox<Npp16s, 3, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilterBox_16s_C4R_Ctx(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterBox<Npp16s, 4, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilterBox_16u_AC4R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterBox<Npp16u, 4, 1>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilterBox_16u_C1R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterBox<Npp16u, 1, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilterBox_16u_C3R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterBox<Npp16u, 3, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilterBox_16u_C4R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterBox<Npp16u, 4, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilterBox_32f_AC4R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterBox<Npp32f, 4, 1>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilterBox_32f_C1R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterBox<Npp32f, 1, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilterBox_32f_C3R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterBox<Npp32f, 3, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilterBox_32f_C4R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterBox<Npp32f, 4, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilterBox_64f_C1R_Ctx(const Npp64f * pSrc, Npp32s nSrcStep, Npp64f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterBox<Npp64f, 1, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilterBox_8u_AC4R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterBox<Npp8u, 4, 1>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilterBox_8u_C1R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterBox<Npp8u, 1, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilterBox_8u_C3R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterBox<Npp8u, 3, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilterBox_8u_C4R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterBox<Npp8u, 4, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilterBox_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterBox_16s_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilterBox_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterBox_16s_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilterBox_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterBox_16s_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilterBox_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterBox_16s_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilterBox_16u_AC4R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterBox_16u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilterBox_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterBox_16u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilterBox_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterBox_16u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilterBox_16u_C4R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterBox_16u_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilterBox_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterBox_32f_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilterBox_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterBox_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilterBox_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterBox_32f_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilterBox_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterBox_32f_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilterBox_64f_C1R(const Npp64f * pSrc, Npp32s nSrcStep, Npp64f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterBox_64f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilterBox_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterBox_8u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilterBox_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterBox_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilterBox_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterBox_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilterBox_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterBox_8u_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx);
}
