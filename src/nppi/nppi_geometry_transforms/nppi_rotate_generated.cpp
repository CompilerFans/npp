// Generated API wrappers -- do not edit by hand.
// family: nppiRotate


#include "npp.h"
#include "nppi_rotate_executor.h"

NppStatus nppiRotate_16u_AC4R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16u * pDst, int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::rotate::rotate<Npp16u, 4, 1>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nAngle, nShiftX, nShiftY, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiRotate_16u_C1R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16u * pDst, int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::rotate::rotate<Npp16u, 1, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nAngle, nShiftX, nShiftY, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiRotate_16u_C3R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16u * pDst, int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::rotate::rotate<Npp16u, 3, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nAngle, nShiftX, nShiftY, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiRotate_16u_C4R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16u * pDst, int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::rotate::rotate<Npp16u, 4, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nAngle, nShiftX, nShiftY, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiRotate_32f_AC4R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32f * pDst, int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::rotate::rotate<Npp32f, 4, 1>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nAngle, nShiftX, nShiftY, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiRotate_32f_C1R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32f * pDst, int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::rotate::rotate<Npp32f, 1, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nAngle, nShiftX, nShiftY, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiRotate_32f_C3R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32f * pDst, int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::rotate::rotate<Npp32f, 3, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nAngle, nShiftX, nShiftY, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiRotate_32f_C4R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32f * pDst, int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::rotate::rotate<Npp32f, 4, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nAngle, nShiftX, nShiftY, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiRotate_8u_AC4R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u * pDst, int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::rotate::rotate<Npp8u, 4, 1>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nAngle, nShiftX, nShiftY, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiRotate_8u_C1R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u * pDst, int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::rotate::rotate<Npp8u, 1, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nAngle, nShiftX, nShiftY, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiRotate_8u_C3R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u * pDst, int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::rotate::rotate<Npp8u, 3, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nAngle, nShiftX, nShiftY, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiRotate_8u_C4R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u * pDst, int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY, int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::rotate::rotate<Npp8u, 4, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nAngle, nShiftX, nShiftY, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiRotate_16u_AC4R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16u * pDst, int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiRotate_16u_AC4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nAngle, nShiftX, nShiftY, eInterpolation, nppStreamCtx);
}

NppStatus nppiRotate_16u_C1R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16u * pDst, int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiRotate_16u_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nAngle, nShiftX, nShiftY, eInterpolation, nppStreamCtx);
}

NppStatus nppiRotate_16u_C3R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16u * pDst, int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiRotate_16u_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nAngle, nShiftX, nShiftY, eInterpolation, nppStreamCtx);
}

NppStatus nppiRotate_16u_C4R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16u * pDst, int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiRotate_16u_C4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nAngle, nShiftX, nShiftY, eInterpolation, nppStreamCtx);
}

NppStatus nppiRotate_32f_AC4R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32f * pDst, int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiRotate_32f_AC4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nAngle, nShiftX, nShiftY, eInterpolation, nppStreamCtx);
}

NppStatus nppiRotate_32f_C1R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32f * pDst, int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiRotate_32f_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nAngle, nShiftX, nShiftY, eInterpolation, nppStreamCtx);
}

NppStatus nppiRotate_32f_C3R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32f * pDst, int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiRotate_32f_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nAngle, nShiftX, nShiftY, eInterpolation, nppStreamCtx);
}

NppStatus nppiRotate_32f_C4R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32f * pDst, int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiRotate_32f_C4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nAngle, nShiftX, nShiftY, eInterpolation, nppStreamCtx);
}

NppStatus nppiRotate_8u_AC4R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u * pDst, int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiRotate_8u_AC4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nAngle, nShiftX, nShiftY, eInterpolation, nppStreamCtx);
}

NppStatus nppiRotate_8u_C1R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u * pDst, int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiRotate_8u_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nAngle, nShiftX, nShiftY, eInterpolation, nppStreamCtx);
}

NppStatus nppiRotate_8u_C3R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u * pDst, int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiRotate_8u_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nAngle, nShiftX, nShiftY, eInterpolation, nppStreamCtx);
}

NppStatus nppiRotate_8u_C4R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u * pDst, int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiRotate_8u_C4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nAngle, nShiftX, nShiftY, eInterpolation, nppStreamCtx);
}
