// Generated API wrappers -- do not edit by hand.
// family: nppiErode3x3Border


#include "npp.h"
#include "nppi_morph_executor.h"

NppStatus nppiErode3x3Border_16u_AC4R_Ctx(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::morph::run3x3Border<Npp16u, 4, 1>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppi::morph::kErode, nppStreamCtx.hStream);
}

NppStatus nppiErode3x3Border_16u_C1R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::morph::run3x3Border<Npp16u, 1, 0>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppi::morph::kErode, nppStreamCtx.hStream);
}

NppStatus nppiErode3x3Border_16u_C3R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::morph::run3x3Border<Npp16u, 3, 0>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppi::morph::kErode, nppStreamCtx.hStream);
}

NppStatus nppiErode3x3Border_16u_C4R_Ctx(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::morph::run3x3Border<Npp16u, 4, 0>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppi::morph::kErode, nppStreamCtx.hStream);
}

NppStatus nppiErode3x3Border_32f_AC4R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::morph::run3x3Border<Npp32f, 4, 1>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppi::morph::kErode, nppStreamCtx.hStream);
}

NppStatus nppiErode3x3Border_32f_C1R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::morph::run3x3Border<Npp32f, 1, 0>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppi::morph::kErode, nppStreamCtx.hStream);
}

NppStatus nppiErode3x3Border_32f_C3R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::morph::run3x3Border<Npp32f, 3, 0>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppi::morph::kErode, nppStreamCtx.hStream);
}

NppStatus nppiErode3x3Border_32f_C4R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::morph::run3x3Border<Npp32f, 4, 0>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppi::morph::kErode, nppStreamCtx.hStream);
}

NppStatus nppiErode3x3Border_8u_AC4R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::morph::run3x3Border<Npp8u, 4, 1>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppi::morph::kErode, nppStreamCtx.hStream);
}

NppStatus nppiErode3x3Border_8u_C1R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::morph::run3x3Border<Npp8u, 1, 0>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppi::morph::kErode, nppStreamCtx.hStream);
}

NppStatus nppiErode3x3Border_8u_C3R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::morph::run3x3Border<Npp8u, 3, 0>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppi::morph::kErode, nppStreamCtx.hStream);
}

NppStatus nppiErode3x3Border_8u_C4R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::morph::run3x3Border<Npp8u, 4, 0>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppi::morph::kErode, nppStreamCtx.hStream);
}

NppStatus nppiErode3x3Border_16u_AC4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiErode3x3Border_16u_AC4R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx);
}

NppStatus nppiErode3x3Border_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiErode3x3Border_16u_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx);
}

NppStatus nppiErode3x3Border_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiErode3x3Border_16u_C3R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx);
}

NppStatus nppiErode3x3Border_16u_C4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiErode3x3Border_16u_C4R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx);
}

NppStatus nppiErode3x3Border_32f_AC4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiErode3x3Border_32f_AC4R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx);
}

NppStatus nppiErode3x3Border_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiErode3x3Border_32f_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx);
}

NppStatus nppiErode3x3Border_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiErode3x3Border_32f_C3R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx);
}

NppStatus nppiErode3x3Border_32f_C4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiErode3x3Border_32f_C4R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx);
}

NppStatus nppiErode3x3Border_8u_AC4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiErode3x3Border_8u_AC4R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx);
}

NppStatus nppiErode3x3Border_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiErode3x3Border_8u_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx);
}

NppStatus nppiErode3x3Border_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiErode3x3Border_8u_C3R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx);
}

NppStatus nppiErode3x3Border_8u_C4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiErode3x3Border_8u_C4R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, eBorderType, nppStreamCtx);
}
