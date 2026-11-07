// Generated API wrappers -- do not edit by hand.
// family: nppiFilter


#include "npp.h"
#include "nppi_filter_executor.h"

NppStatus nppiFilter_16s_AC4R_Ctx(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterDiv<Npp16s, Npp16s, 4, 1>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nDivisor, nppStreamCtx.hStream);
}

NppStatus nppiFilter_16s_C1R_Ctx(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterDiv<Npp16s, Npp16s, 1, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nDivisor, nppStreamCtx.hStream);
}

NppStatus nppiFilter_16s_C3R_Ctx(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterDiv<Npp16s, Npp16s, 3, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nDivisor, nppStreamCtx.hStream);
}

NppStatus nppiFilter_16s_C4R_Ctx(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterDiv<Npp16s, Npp16s, 4, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nDivisor, nppStreamCtx.hStream);
}

NppStatus nppiFilter_16u_AC4R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterDiv<Npp16u, Npp16u, 4, 1>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nDivisor, nppStreamCtx.hStream);
}

NppStatus nppiFilter_16u_C1R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterDiv<Npp16u, Npp16u, 1, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nDivisor, nppStreamCtx.hStream);
}

NppStatus nppiFilter_16u_C3R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterDiv<Npp16u, Npp16u, 3, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nDivisor, nppStreamCtx.hStream);
}

NppStatus nppiFilter_16u_C4R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterDiv<Npp16u, Npp16u, 4, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nDivisor, nppStreamCtx.hStream);
}

NppStatus nppiFilter_32f_AC4R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp32f, Npp32f, 4, 1>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter_32f_C2R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp32f, Npp32f, 2, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter_32f_C3R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp32f, Npp32f, 3, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter_32f_C4R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp32f, Npp32f, 4, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter_64f_C1R_Ctx(const Npp64f * pSrc, Npp32s nSrcStep, Npp64f * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp64f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter64f<Npp64f, Npp64f, 1, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter_8u_AC4R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterDiv<Npp8u, Npp8u, 4, 1>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nDivisor, nppStreamCtx.hStream);
}

NppStatus nppiFilter_8u_C4R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filterDiv<Npp8u, Npp8u, 4, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nDivisor, nppStreamCtx.hStream);
}

NppStatus nppiFilter_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter_16s_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nDivisor, nppStreamCtx);
}

NppStatus nppiFilter_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter_16s_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nDivisor, nppStreamCtx);
}

NppStatus nppiFilter_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter_16s_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nDivisor, nppStreamCtx);
}

NppStatus nppiFilter_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter_16s_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nDivisor, nppStreamCtx);
}

NppStatus nppiFilter_16u_AC4R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter_16u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nDivisor, nppStreamCtx);
}

NppStatus nppiFilter_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter_16u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nDivisor, nppStreamCtx);
}

NppStatus nppiFilter_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter_16u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nDivisor, nppStreamCtx);
}

NppStatus nppiFilter_16u_C4R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter_16u_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nDivisor, nppStreamCtx);
}

NppStatus nppiFilter_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter_32f_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter_32f_C2R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter_32f_C2R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter_32f_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter_32f_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter_64f_C1R(const Npp64f * pSrc, Npp32s nSrcStep, Npp64f * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp64f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter_64f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter_8u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nDivisor, nppStreamCtx);
}

NppStatus nppiFilter_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter_8u_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nDivisor, nppStreamCtx);
}
