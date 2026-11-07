// Generated API wrappers -- do not edit by hand.
// family: nppiGradientVectorScharrBorder


#include "npp.h"
#include "nppi_gradient_vector_executor.h"

NppStatus nppiGradientVectorScharrBorder_16s32f_C1R_Ctx(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDstX, int nDstXStep, Npp32f * pDstY, int nDstYStep, Npp32f * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::gv::gradientVectorBorder<Npp16s, Npp32f, 1, nppi::gv::kScharr>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDstX, nDstXStep, pDstY, nDstYStep, pDstMag, nDstMagStep, pDstAngle, nDstAngleStep, oSizeROI, eMaskSize, eNorm, eBorderType, nppStreamCtx.hStream);
}

NppStatus nppiGradientVectorScharrBorder_16s32f_C3C1R_Ctx(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDstX, int nDstXStep, Npp32f * pDstY, int nDstYStep, Npp32f * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::gv::gradientVectorBorder<Npp16s, Npp32f, 3, nppi::gv::kScharr>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDstX, nDstXStep, pDstY, nDstYStep, pDstMag, nDstMagStep, pDstAngle, nDstAngleStep, oSizeROI, eMaskSize, eNorm, eBorderType, nppStreamCtx.hStream);
}

NppStatus nppiGradientVectorScharrBorder_16u32f_C1R_Ctx(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDstX, int nDstXStep, Npp32f * pDstY, int nDstYStep, Npp32f * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::gv::gradientVectorBorder<Npp16u, Npp32f, 1, nppi::gv::kScharr>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDstX, nDstXStep, pDstY, nDstYStep, pDstMag, nDstMagStep, pDstAngle, nDstAngleStep, oSizeROI, eMaskSize, eNorm, eBorderType, nppStreamCtx.hStream);
}

NppStatus nppiGradientVectorScharrBorder_16u32f_C3C1R_Ctx(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDstX, int nDstXStep, Npp32f * pDstY, int nDstYStep, Npp32f * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::gv::gradientVectorBorder<Npp16u, Npp32f, 3, nppi::gv::kScharr>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDstX, nDstXStep, pDstY, nDstYStep, pDstMag, nDstMagStep, pDstAngle, nDstAngleStep, oSizeROI, eMaskSize, eNorm, eBorderType, nppStreamCtx.hStream);
}

NppStatus nppiGradientVectorScharrBorder_32f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDstX, int nDstXStep, Npp32f * pDstY, int nDstYStep, Npp32f * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::gv::gradientVectorBorder<Npp32f, Npp32f, 1, nppi::gv::kScharr>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDstX, nDstXStep, pDstY, nDstYStep, pDstMag, nDstMagStep, pDstAngle, nDstAngleStep, oSizeROI, eMaskSize, eNorm, eBorderType, nppStreamCtx.hStream);
}

NppStatus nppiGradientVectorScharrBorder_32f_C3C1R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDstX, int nDstXStep, Npp32f * pDstY, int nDstYStep, Npp32f * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::gv::gradientVectorBorder<Npp32f, Npp32f, 3, nppi::gv::kScharr>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDstX, nDstXStep, pDstY, nDstYStep, pDstMag, nDstMagStep, pDstAngle, nDstAngleStep, oSizeROI, eMaskSize, eNorm, eBorderType, nppStreamCtx.hStream);
}

NppStatus nppiGradientVectorScharrBorder_8u16s_C1R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDstX, int nDstXStep, Npp16s * pDstY, int nDstYStep, Npp16s * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::gv::gradientVectorBorder<Npp8u, Npp16s, 1, nppi::gv::kScharr>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDstX, nDstXStep, pDstY, nDstYStep, pDstMag, nDstMagStep, pDstAngle, nDstAngleStep, oSizeROI, eMaskSize, eNorm, eBorderType, nppStreamCtx.hStream);
}

NppStatus nppiGradientVectorScharrBorder_8u16s_C3C1R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDstX, int nDstXStep, Npp16s * pDstY, int nDstYStep, Npp16s * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppi::gv::gradientVectorBorder<Npp8u, Npp16s, 3, nppi::gv::kScharr>(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDstX, nDstXStep, pDstY, nDstYStep, pDstMag, nDstMagStep, pDstAngle, nDstAngleStep, oSizeROI, eMaskSize, eNorm, eBorderType, nppStreamCtx.hStream);
}

NppStatus nppiGradientVectorScharrBorder_16s32f_C1R(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDstX, int nDstXStep, Npp32f * pDstY, int nDstYStep, Npp32f * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiGradientVectorScharrBorder_16s32f_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDstX, nDstXStep, pDstY, nDstYStep, pDstMag, nDstMagStep, pDstAngle, nDstAngleStep, oSizeROI, eMaskSize, eNorm, eBorderType, nppStreamCtx);
}

NppStatus nppiGradientVectorScharrBorder_16s32f_C3C1R(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDstX, int nDstXStep, Npp32f * pDstY, int nDstYStep, Npp32f * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiGradientVectorScharrBorder_16s32f_C3C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDstX, nDstXStep, pDstY, nDstYStep, pDstMag, nDstMagStep, pDstAngle, nDstAngleStep, oSizeROI, eMaskSize, eNorm, eBorderType, nppStreamCtx);
}

NppStatus nppiGradientVectorScharrBorder_16u32f_C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDstX, int nDstXStep, Npp32f * pDstY, int nDstYStep, Npp32f * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiGradientVectorScharrBorder_16u32f_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDstX, nDstXStep, pDstY, nDstYStep, pDstMag, nDstMagStep, pDstAngle, nDstAngleStep, oSizeROI, eMaskSize, eNorm, eBorderType, nppStreamCtx);
}

NppStatus nppiGradientVectorScharrBorder_16u32f_C3C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDstX, int nDstXStep, Npp32f * pDstY, int nDstYStep, Npp32f * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiGradientVectorScharrBorder_16u32f_C3C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDstX, nDstXStep, pDstY, nDstYStep, pDstMag, nDstMagStep, pDstAngle, nDstAngleStep, oSizeROI, eMaskSize, eNorm, eBorderType, nppStreamCtx);
}

NppStatus nppiGradientVectorScharrBorder_32f_C1R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDstX, int nDstXStep, Npp32f * pDstY, int nDstYStep, Npp32f * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiGradientVectorScharrBorder_32f_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDstX, nDstXStep, pDstY, nDstYStep, pDstMag, nDstMagStep, pDstAngle, nDstAngleStep, oSizeROI, eMaskSize, eNorm, eBorderType, nppStreamCtx);
}

NppStatus nppiGradientVectorScharrBorder_32f_C3C1R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDstX, int nDstXStep, Npp32f * pDstY, int nDstYStep, Npp32f * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiGradientVectorScharrBorder_32f_C3C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDstX, nDstXStep, pDstY, nDstYStep, pDstMag, nDstMagStep, pDstAngle, nDstAngleStep, oSizeROI, eMaskSize, eNorm, eBorderType, nppStreamCtx);
}

NppStatus nppiGradientVectorScharrBorder_8u16s_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDstX, int nDstXStep, Npp16s * pDstY, int nDstYStep, Npp16s * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiGradientVectorScharrBorder_8u16s_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDstX, nDstXStep, pDstY, nDstYStep, pDstMag, nDstMagStep, pDstAngle, nDstAngleStep, oSizeROI, eMaskSize, eNorm, eBorderType, nppStreamCtx);
}

NppStatus nppiGradientVectorScharrBorder_8u16s_C3C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDstX, int nDstXStep, Npp16s * pDstY, int nDstYStep, Npp16s * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiGradientVectorScharrBorder_8u16s_C3C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDstX, nDstXStep, pDstY, nDstYStep, pDstMag, nDstMagStep, pDstAngle, nDstAngleStep, oSizeROI, eMaskSize, eNorm, eBorderType, nppStreamCtx);
}
