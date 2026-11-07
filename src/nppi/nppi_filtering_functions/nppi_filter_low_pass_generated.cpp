// Generated API wrappers -- do not edit by hand.
// family: nppiFilterLowPass


#include "npp.h"
#include "nppi_filter_fixed_executor.h"

NppStatus nppiFilterLowPass_16s_AC4R_Ctx(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterFixedMask<Npp16s, Npp16s, 4, 1, nppi::filter::kLowPass>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx.hStream);
}

NppStatus nppiFilterLowPass_16s_C1R_Ctx(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterFixedMask<Npp16s, Npp16s, 1, 0, nppi::filter::kLowPass>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx.hStream);
}

NppStatus nppiFilterLowPass_16s_C3R_Ctx(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterFixedMask<Npp16s, Npp16s, 3, 0, nppi::filter::kLowPass>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx.hStream);
}

NppStatus nppiFilterLowPass_16s_C4R_Ctx(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterFixedMask<Npp16s, Npp16s, 4, 0, nppi::filter::kLowPass>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx.hStream);
}

NppStatus nppiFilterLowPass_16u_AC4R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterFixedMask<Npp16u, Npp16u, 4, 1, nppi::filter::kLowPass>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx.hStream);
}

NppStatus nppiFilterLowPass_16u_C1R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterFixedMask<Npp16u, Npp16u, 1, 0, nppi::filter::kLowPass>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx.hStream);
}

NppStatus nppiFilterLowPass_16u_C3R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterFixedMask<Npp16u, Npp16u, 3, 0, nppi::filter::kLowPass>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx.hStream);
}

NppStatus nppiFilterLowPass_16u_C4R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterFixedMask<Npp16u, Npp16u, 4, 0, nppi::filter::kLowPass>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx.hStream);
}

NppStatus nppiFilterLowPass_32f_AC4R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterFixedMask<Npp32f, Npp32f, 4, 1, nppi::filter::kLowPass>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx.hStream);
}

NppStatus nppiFilterLowPass_32f_C1R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterFixedMask<Npp32f, Npp32f, 1, 0, nppi::filter::kLowPass>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx.hStream);
}

NppStatus nppiFilterLowPass_32f_C3R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterFixedMask<Npp32f, Npp32f, 3, 0, nppi::filter::kLowPass>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx.hStream);
}

NppStatus nppiFilterLowPass_32f_C4R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterFixedMask<Npp32f, Npp32f, 4, 0, nppi::filter::kLowPass>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx.hStream);
}

NppStatus nppiFilterLowPass_8u_AC4R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterFixedMask<Npp8u, Npp8u, 4, 1, nppi::filter::kLowPass>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx.hStream);
}

NppStatus nppiFilterLowPass_8u_C1R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterFixedMask<Npp8u, Npp8u, 1, 0, nppi::filter::kLowPass>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx.hStream);
}

NppStatus nppiFilterLowPass_8u_C3R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterFixedMask<Npp8u, Npp8u, 3, 0, nppi::filter::kLowPass>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx.hStream);
}

NppStatus nppiFilterLowPass_8u_C4R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterFixedMask<Npp8u, Npp8u, 4, 0, nppi::filter::kLowPass>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx.hStream);
}

NppStatus nppiFilterLowPass_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterLowPass_16s_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx);
}

NppStatus nppiFilterLowPass_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterLowPass_16s_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx);
}

NppStatus nppiFilterLowPass_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterLowPass_16s_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx);
}

NppStatus nppiFilterLowPass_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterLowPass_16s_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx);
}

NppStatus nppiFilterLowPass_16u_AC4R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterLowPass_16u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx);
}

NppStatus nppiFilterLowPass_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterLowPass_16u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx);
}

NppStatus nppiFilterLowPass_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterLowPass_16u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx);
}

NppStatus nppiFilterLowPass_16u_C4R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterLowPass_16u_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx);
}

NppStatus nppiFilterLowPass_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterLowPass_32f_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx);
}

NppStatus nppiFilterLowPass_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterLowPass_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx);
}

NppStatus nppiFilterLowPass_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterLowPass_32f_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx);
}

NppStatus nppiFilterLowPass_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterLowPass_32f_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx);
}

NppStatus nppiFilterLowPass_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterLowPass_8u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx);
}

NppStatus nppiFilterLowPass_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterLowPass_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx);
}

NppStatus nppiFilterLowPass_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterLowPass_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx);
}

NppStatus nppiFilterLowPass_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilterLowPass_8u_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx);
}
