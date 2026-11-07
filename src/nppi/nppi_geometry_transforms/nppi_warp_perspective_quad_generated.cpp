// Generated API wrappers -- do not edit by hand.
// family: nppiWarpPerspectiveQuad


#include "npp.h"
#include "nppi_warp_perspective_executor.h"

NppStatus nppiWarpPerspectiveQuad_16u_AC4R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp16u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::warp::warpQuad<Npp16u, 4, 1>(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiWarpPerspectiveQuad_16u_C1R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp16u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::warp::warpQuad<Npp16u, 1, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiWarpPerspectiveQuad_16u_C3R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp16u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::warp::warpQuad<Npp16u, 3, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiWarpPerspectiveQuad_16u_C4R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp16u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::warp::warpQuad<Npp16u, 4, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiWarpPerspectiveQuad_16u_P3R_Ctx(const Npp16u * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp16u * pDst[3], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::warp::warpQuadPlanar<Npp16u, 3>(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiWarpPerspectiveQuad_16u_P4R_Ctx(const Npp16u * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp16u * pDst[4], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::warp::warpQuadPlanar<Npp16u, 4>(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiWarpPerspectiveQuad_32f_AC4R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp32f * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::warp::warpQuad<Npp32f, 4, 1>(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiWarpPerspectiveQuad_32f_C1R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp32f * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::warp::warpQuad<Npp32f, 1, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiWarpPerspectiveQuad_32f_C3R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp32f * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::warp::warpQuad<Npp32f, 3, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiWarpPerspectiveQuad_32f_C4R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp32f * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::warp::warpQuad<Npp32f, 4, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiWarpPerspectiveQuad_32f_P3R_Ctx(const Npp32f * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp32f * pDst[3], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::warp::warpQuadPlanar<Npp32f, 3>(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiWarpPerspectiveQuad_32f_P4R_Ctx(const Npp32f * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp32f * pDst[4], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::warp::warpQuadPlanar<Npp32f, 4>(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiWarpPerspectiveQuad_32s_AC4R_Ctx(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp32s * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::warp::warpQuad<Npp32s, 4, 1>(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiWarpPerspectiveQuad_32s_C1R_Ctx(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp32s * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::warp::warpQuad<Npp32s, 1, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiWarpPerspectiveQuad_32s_C3R_Ctx(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp32s * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::warp::warpQuad<Npp32s, 3, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiWarpPerspectiveQuad_32s_C4R_Ctx(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp32s * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::warp::warpQuad<Npp32s, 4, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiWarpPerspectiveQuad_32s_P3R_Ctx(const Npp32s * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp32s * pDst[3], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::warp::warpQuadPlanar<Npp32s, 3>(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiWarpPerspectiveQuad_32s_P4R_Ctx(const Npp32s * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp32s * pDst[4], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::warp::warpQuadPlanar<Npp32s, 4>(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiWarpPerspectiveQuad_8u_AC4R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp8u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::warp::warpQuad<Npp8u, 4, 1>(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiWarpPerspectiveQuad_8u_C1R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp8u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::warp::warpQuad<Npp8u, 1, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiWarpPerspectiveQuad_8u_C3R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp8u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::warp::warpQuad<Npp8u, 3, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiWarpPerspectiveQuad_8u_C4R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp8u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::warp::warpQuad<Npp8u, 4, 0>(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiWarpPerspectiveQuad_8u_P3R_Ctx(const Npp8u * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp8u * pDst[3], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::warp::warpQuadPlanar<Npp8u, 3>(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiWarpPerspectiveQuad_8u_P4R_Ctx(const Npp8u * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp8u * pDst[4], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::warp::warpQuadPlanar<Npp8u, 4>(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx.hStream);
}

NppStatus nppiWarpPerspectiveQuad_16u_AC4R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp16u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpPerspectiveQuad_16u_AC4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveQuad_16u_C1R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp16u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpPerspectiveQuad_16u_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveQuad_16u_C3R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp16u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpPerspectiveQuad_16u_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveQuad_16u_C4R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp16u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpPerspectiveQuad_16u_C4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveQuad_16u_P3R(const Npp16u * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp16u * pDst[3], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpPerspectiveQuad_16u_P3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveQuad_16u_P4R(const Npp16u * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp16u * pDst[4], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpPerspectiveQuad_16u_P4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveQuad_32f_AC4R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp32f * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpPerspectiveQuad_32f_AC4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveQuad_32f_C1R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp32f * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpPerspectiveQuad_32f_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveQuad_32f_C3R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp32f * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpPerspectiveQuad_32f_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveQuad_32f_C4R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp32f * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpPerspectiveQuad_32f_C4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveQuad_32f_P3R(const Npp32f * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp32f * pDst[3], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpPerspectiveQuad_32f_P3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveQuad_32f_P4R(const Npp32f * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp32f * pDst[4], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpPerspectiveQuad_32f_P4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveQuad_32s_AC4R(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp32s * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpPerspectiveQuad_32s_AC4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveQuad_32s_C1R(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp32s * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpPerspectiveQuad_32s_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveQuad_32s_C3R(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp32s * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpPerspectiveQuad_32s_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveQuad_32s_C4R(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp32s * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpPerspectiveQuad_32s_C4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveQuad_32s_P3R(const Npp32s * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp32s * pDst[3], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpPerspectiveQuad_32s_P3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveQuad_32s_P4R(const Npp32s * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp32s * pDst[4], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpPerspectiveQuad_32s_P4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveQuad_8u_AC4R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp8u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpPerspectiveQuad_8u_AC4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveQuad_8u_C1R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp8u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpPerspectiveQuad_8u_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveQuad_8u_C3R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp8u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpPerspectiveQuad_8u_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveQuad_8u_C4R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp8u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpPerspectiveQuad_8u_C4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveQuad_8u_P3R(const Npp8u * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp8u * pDst[3], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpPerspectiveQuad_8u_P3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveQuad_8u_P4R(const Npp8u * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], Npp8u * pDst[4], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiWarpPerspectiveQuad_8u_P4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, eInterpolation, nppStreamCtx);
}
