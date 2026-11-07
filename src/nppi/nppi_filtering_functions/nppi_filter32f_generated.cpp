// Generated API wrappers -- do not edit by hand.
// family: nppiFilter32f


#include "npp.h"
#include "nppi_filter_executor.h"

NppStatus nppiFilter32f_16f_C1R_Ctx(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp16f, Npp16f, 1, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter32f_16f_C3R_Ctx(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp16f, Npp16f, 3, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter32f_16f_C4R_Ctx(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp16f, Npp16f, 4, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter32f_16s_AC4R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp16s, Npp16s, 4, 1>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter32f_16s_C1R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp16s, Npp16s, 1, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter32f_16s_C3R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp16s, Npp16s, 3, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter32f_16s_C4R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp16s, Npp16s, 4, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter32f_16u_AC4R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp16u, Npp16u, 4, 1>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter32f_16u_C1R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp16u, Npp16u, 1, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter32f_16u_C3R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp16u, Npp16u, 3, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter32f_16u_C4R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp16u, Npp16u, 4, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter32f_32s_AC4R_Ctx(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp32s, Npp32s, 4, 1>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter32f_32s_C1R_Ctx(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp32s, Npp32s, 1, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter32f_32s_C3R_Ctx(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp32s, Npp32s, 3, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter32f_32s_C4R_Ctx(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp32s, Npp32s, 4, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter32f_8s16s_AC4R_Ctx(const Npp8s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp8s, Npp16s, 4, 1>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter32f_8s16s_C1R_Ctx(const Npp8s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp8s, Npp16s, 1, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter32f_8s16s_C3R_Ctx(const Npp8s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp8s, Npp16s, 3, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter32f_8s16s_C4R_Ctx(const Npp8s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp8s, Npp16s, 4, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter32f_8s_AC4R_Ctx(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp8s, Npp8s, 4, 1>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter32f_8s_C1R_Ctx(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp8s, Npp8s, 1, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter32f_8s_C2R_Ctx(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp8s, Npp8s, 2, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter32f_8s_C3R_Ctx(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp8s, Npp8s, 3, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter32f_8s_C4R_Ctx(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp8s, Npp8s, 4, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter32f_8u16s_AC4R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp8u, Npp16s, 4, 1>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter32f_8u16s_C1R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp8u, Npp16s, 1, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter32f_8u16s_C3R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp8u, Npp16s, 3, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter32f_8u16s_C4R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp8u, Npp16s, 4, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter32f_8u_AC4R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp8u, Npp8u, 4, 1>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter32f_8u_C1R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp8u, Npp8u, 1, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter32f_8u_C2R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp8u, Npp8u, 2, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter32f_8u_C3R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp8u, Npp8u, 3, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter32f_8u_C4R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp8u, Npp8u, 4, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx.hStream);
}

NppStatus nppiFilter32f_16f_C1R(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter32f_16f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter32f_16f_C3R(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter32f_16f_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter32f_16f_C4R(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter32f_16f_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter32f_16s_AC4R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter32f_16s_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter32f_16s_C1R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter32f_16s_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter32f_16s_C3R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter32f_16s_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter32f_16s_C4R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter32f_16s_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter32f_16u_AC4R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter32f_16u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter32f_16u_C1R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter32f_16u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter32f_16u_C3R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter32f_16u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter32f_16u_C4R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter32f_16u_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter32f_32s_AC4R(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter32f_32s_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter32f_32s_C1R(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter32f_32s_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter32f_32s_C3R(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter32f_32s_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter32f_32s_C4R(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter32f_32s_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter32f_8s16s_AC4R(const Npp8s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter32f_8s16s_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter32f_8s16s_C1R(const Npp8s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter32f_8s16s_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter32f_8s16s_C3R(const Npp8s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter32f_8s16s_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter32f_8s16s_C4R(const Npp8s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter32f_8s16s_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter32f_8s_AC4R(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter32f_8s_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter32f_8s_C1R(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter32f_8s_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter32f_8s_C2R(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter32f_8s_C2R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter32f_8s_C3R(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter32f_8s_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter32f_8s_C4R(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter32f_8s_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter32f_8u16s_AC4R(const Npp8u * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter32f_8u16s_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter32f_8u16s_C1R(const Npp8u * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter32f_8u16s_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter32f_8u16s_C3R(const Npp8u * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter32f_8u16s_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter32f_8u16s_C4R(const Npp8u * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter32f_8u16s_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter32f_8u_AC4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter32f_8u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter32f_8u_C1R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter32f_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter32f_8u_C2R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter32f_8u_C2R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter32f_8u_C3R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter32f_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}

NppStatus nppiFilter32f_8u_C4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiFilter32f_8u_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}
