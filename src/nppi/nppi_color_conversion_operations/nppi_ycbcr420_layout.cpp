#include "npp.h"
#include <cuda_runtime.h>

extern "C" {
cudaError_t nppiPlanar420ToSemiPlanar420_8u_P3P2R_kernel(const Npp8u *pSrcY, int nSrcYStep,
                                                         const Npp8u *pSrcChroma0, int nSrcChroma0Step,
                                                         const Npp8u *pSrcChroma1, int nSrcChroma1Step,
                                                         Npp8u *pDstY, int nDstYStep, Npp8u *pDstChroma01,
                                                         int nDstChroma01Step, NppiSize oSizeROI,
                                                         cudaStream_t stream);
cudaError_t nppiSemiPlanar420ToPlanar420_8u_P2P3R_kernel(const Npp8u *pSrcY, int nSrcYStep,
                                                         const Npp8u *pSrcChroma01, int nSrcChroma01Step,
                                                         Npp8u *pDstY, int nDstYStep, Npp8u *pDstChroma0,
                                                         int nDstChroma0Step, Npp8u *pDstChroma1,
                                                         int nDstChroma1Step, NppiSize oSizeROI,
                                                         cudaStream_t stream);
cudaError_t nppiPlanar420ToPlanar422_8u_P3P3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCb,
                                                     int nSrcCbStep, const Npp8u *pSrcCr, int nSrcCrStep,
                                                     Npp8u *pDstY, int nDstYStep, Npp8u *pDstCb, int nDstCbStep,
                                                     Npp8u *pDstCr, int nDstCrStep, NppiSize oSizeROI,
                                                     cudaStream_t stream);
cudaError_t nppiSemiPlanar420ToPlanar422_8u_P2P3R_kernel(const Npp8u *pSrcY, int nSrcYStep,
                                                         const Npp8u *pSrcCbCr, int nSrcCbCrStep, Npp8u *pDstY,
                                                         int nDstYStep, Npp8u *pDstCb, int nDstCbStep,
                                                         Npp8u *pDstCr, int nDstCrStep, NppiSize oSizeROI,
                                                         cudaStream_t stream);
cudaError_t nppiSemiPlanar420ToPacked422_8u_P2C2R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCbCr,
                                                         int nSrcCbCrStep, Npp8u *pDst, int nDstStep,
                                                         NppiSize oSizeROI, int packedMode, cudaStream_t stream);
cudaError_t nppiPlanarYCrCb420ToPackedCbYCr422_8u_P3C2R_kernel(const Npp8u *pSrcY, int nSrcYStep,
                                                               const Npp8u *pSrcCr, int nSrcCrStep,
                                                               const Npp8u *pSrcCb, int nSrcCbStep, Npp8u *pDst,
                                                               int nDstStep, NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiPlanarYCrCb420ToSemiPlanar411_8u_P3P2R_kernel(const Npp8u *pSrcY, int nSrcYStep,
                                                              const Npp8u *pSrcCr, int nSrcCrStep,
                                                              const Npp8u *pSrcCb, int nSrcCbStep, Npp8u *pDstY,
                                                              int nDstYStep, Npp8u *pDstCbCr, int nDstCbCrStep,
                                                              NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiSemiPlanar420ToPlanar411_8u_P2P3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCbCr,
                                                         int nSrcCbCrStep, Npp8u *pDstY, int nDstYStep,
                                                         Npp8u *pDstCb, int nDstCbStep, Npp8u *pDstCr,
                                                         int nDstCrStep, NppiSize oSizeROI, cudaStream_t stream);
}

namespace {

NppStatus validate420Roi(NppiSize oSizeROI) {
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_WRONG_INTERSECTION_ROI_ERROR;
  }
  if ((oSizeROI.width % 2) != 0 || (oSizeROI.height % 2) != 0) {
    return NPP_WRONG_INTERSECTION_ROI_ERROR;
  }
  return NPP_NO_ERROR;
}

NppStatus validate411DestinationRoi(NppiSize oSizeROI) {
  NppStatus status = validate420Roi(oSizeROI);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  if ((oSizeROI.width % 4) != 0) {
    return NPP_WRONG_INTERSECTION_ROI_ERROR;
  }
  return NPP_NO_ERROR;
}

NppStatus validatePlanar420ToSemiPlanarInputs(const Npp8u *const pSrc[3], const int rSrcStep[3], Npp8u *pDstY,
                                              int nDstYStep, Npp8u *pDstCbCr, int nDstCbCrStep,
                                              NppiSize oSizeROI) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pSrc[2] || !pDstY || !pDstCbCr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!rSrcStep || rSrcStep[0] <= 0 || rSrcStep[1] <= 0 || rSrcStep[2] <= 0 || nDstYStep <= 0 ||
      nDstCbCrStep <= 0) {
    return NPP_STEP_ERROR;
  }
  return validate420Roi(oSizeROI);
}

NppStatus validateSemiPlanar420ToPlanarInputs(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCbCr,
                                              int nSrcCbCrStep, Npp8u *pDst[3], int rDstStep[3], NppiSize oSizeROI) {
  if (!pSrcY || !pSrcCbCr || !pDst || !pDst[0] || !pDst[1] || !pDst[2]) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nSrcYStep <= 0 || nSrcCbCrStep <= 0 || !rDstStep || rDstStep[0] <= 0 || rDstStep[1] <= 0 ||
      rDstStep[2] <= 0) {
    return NPP_STEP_ERROR;
  }
  return validate420Roi(oSizeROI);
}

NppStatus validatePlanar420ToPlanar422Inputs(const Npp8u *const pSrc[3], const int rSrcStep[3], Npp8u *pDst[3],
                                             int nDstStep[3], NppiSize oSizeROI) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pSrc[2] || !pDst || !pDst[0] || !pDst[1] || !pDst[2]) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!rSrcStep || !nDstStep || rSrcStep[0] <= 0 || rSrcStep[1] <= 0 || rSrcStep[2] <= 0 || nDstStep[0] <= 0 ||
      nDstStep[1] <= 0 || nDstStep[2] <= 0) {
    return NPP_STEP_ERROR;
  }
  return validate420Roi(oSizeROI);
}

NppStatus validateSemiPlanar420ToPlanar422Inputs(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCbCr,
                                                 int nSrcCbCrStep, Npp8u *pDst[3], int rDstStep[3],
                                                 NppiSize oSizeROI) {
  if (!pSrcY || !pSrcCbCr || !pDst || !pDst[0] || !pDst[1] || !pDst[2]) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nSrcYStep <= 0 || nSrcCbCrStep <= 0 || !rDstStep || rDstStep[0] <= 0 || rDstStep[1] <= 0 ||
      rDstStep[2] <= 0) {
    return NPP_STEP_ERROR;
  }
  return validate420Roi(oSizeROI);
}

NppStatus validateSemiPlanar420ToPlanar411Inputs(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCbCr,
                                                 int nSrcCbCrStep, Npp8u *pDst[3], int rDstStep[3],
                                                 NppiSize oSizeROI) {
  if (!pSrcY || !pSrcCbCr || !pDst || !pDst[0] || !pDst[1] || !pDst[2]) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nSrcYStep <= 0 || nSrcCbCrStep <= 0 || !rDstStep || rDstStep[0] <= 0 || rDstStep[1] <= 0 ||
      rDstStep[2] <= 0) {
    return NPP_STEP_ERROR;
  }
  return validate411DestinationRoi(oSizeROI);
}

NppStatus validateSemiPlanar420ToPacked422Inputs(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCbCr,
                                                 int nSrcCbCrStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  if (!pSrcY || !pSrcCbCr || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nSrcYStep <= 0 || nSrcCbCrStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }
  return validate420Roi(oSizeROI);
}

NppStatus validatePlanarYCrCb420ToPacked422Inputs(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst,
                                                  int nDstStep, NppiSize oSizeROI) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pSrc[2] || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!rSrcStep || rSrcStep[0] <= 0 || rSrcStep[1] <= 0 || rSrcStep[2] <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }
  return validate420Roi(oSizeROI);
}

NppStatus validatePlanarYCrCb420ToSemiPlanar411Inputs(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDstY,
                                                      int nDstYStep, Npp8u *pDstCbCr, int nDstCbCrStep,
                                                      NppiSize oSizeROI) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pSrc[2] || !pDstY || !pDstCbCr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!rSrcStep || rSrcStep[0] <= 0 || rSrcStep[1] <= 0 || rSrcStep[2] <= 0 || nDstYStep <= 0 ||
      nDstCbCrStep <= 0) {
    return NPP_STEP_ERROR;
  }
  return validate411DestinationRoi(oSizeROI);
}

NppStatus toStatus(cudaError_t cudaStatus) {
  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

} // namespace

NppStatus nppiYCbCr420_8u_P3P2R_Ctx(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDstY, int nDstYStep,
                                    Npp8u *pDstCbCr, int nDstCbCrStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  NppStatus status =
      validatePlanar420ToSemiPlanarInputs(pSrc, rSrcStep, pDstY, nDstYStep, pDstCbCr, nDstCbCrStep, oSizeROI);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  return toStatus(nppiPlanar420ToSemiPlanar420_8u_P3P2R_kernel(pSrc[0], rSrcStep[0], pSrc[1], rSrcStep[1], pSrc[2],
                                                               rSrcStep[2], pDstY, nDstYStep, pDstCbCr, nDstCbCrStep,
                                                               oSizeROI, nppStreamCtx.hStream));
}

NppStatus nppiYCbCr420_8u_P3P2R(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDstY, int nDstYStep,
                                Npp8u *pDstCbCr, int nDstCbCrStep, NppiSize oSizeROI) {
  NppStreamContext ctx{};
  ctx.hStream = 0;
  return nppiYCbCr420_8u_P3P2R_Ctx(pSrc, rSrcStep, pDstY, nDstYStep, pDstCbCr, nDstCbCrStep, oSizeROI, ctx);
}

NppStatus nppiYCrCb420ToYCbCr420_8u_P3P2R_Ctx(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDstY,
                                              int nDstYStep, Npp8u *pDstCbCr, int nDstCbCrStep, NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx) {
  NppStatus status =
      validatePlanar420ToSemiPlanarInputs(pSrc, rSrcStep, pDstY, nDstYStep, pDstCbCr, nDstCbCrStep, oSizeROI);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  return toStatus(nppiPlanar420ToSemiPlanar420_8u_P3P2R_kernel(pSrc[0], rSrcStep[0], pSrc[2], rSrcStep[2], pSrc[1],
                                                               rSrcStep[1], pDstY, nDstYStep, pDstCbCr, nDstCbCrStep,
                                                               oSizeROI, nppStreamCtx.hStream));
}

NppStatus nppiYCrCb420ToYCbCr420_8u_P3P2R(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDstY, int nDstYStep,
                                          Npp8u *pDstCbCr, int nDstCbCrStep, NppiSize oSizeROI) {
  NppStreamContext ctx{};
  ctx.hStream = 0;
  return nppiYCrCb420ToYCbCr420_8u_P3P2R_Ctx(pSrc, rSrcStep, pDstY, nDstYStep, pDstCbCr, nDstCbCrStep, oSizeROI,
                                             ctx);
}

NppStatus nppiYCbCr420_8u_P2P3R_Ctx(const Npp8u *const pSrcY, int nSrcYStep, const Npp8u *pSrcCbCr,
                                    int nSrcCbCrStep, Npp8u *pDst[3], int rDstStep[3], NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  NppStatus status =
      validateSemiPlanar420ToPlanarInputs(pSrcY, nSrcYStep, pSrcCbCr, nSrcCbCrStep, pDst, rDstStep, oSizeROI);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  return toStatus(nppiSemiPlanar420ToPlanar420_8u_P2P3R_kernel(pSrcY, nSrcYStep, pSrcCbCr, nSrcCbCrStep, pDst[0],
                                                               rDstStep[0], pDst[1], rDstStep[1], pDst[2],
                                                               rDstStep[2], oSizeROI, nppStreamCtx.hStream));
}

NppStatus nppiYCbCr420_8u_P2P3R(const Npp8u *const pSrcY, int nSrcYStep, const Npp8u *pSrcCbCr, int nSrcCbCrStep,
                                Npp8u *pDst[3], int rDstStep[3], NppiSize oSizeROI) {
  NppStreamContext ctx{};
  ctx.hStream = 0;
  return nppiYCbCr420_8u_P2P3R_Ctx(pSrcY, nSrcYStep, pSrcCbCr, nSrcCbCrStep, pDst, rDstStep, oSizeROI, ctx);
}

NppStatus nppiYCbCr420ToYCrCb420_8u_P2P3R_Ctx(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCbCr,
                                              int nSrcCbCrStep, Npp8u *pDst[3], int rDstStep[3], NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx) {
  NppStatus status =
      validateSemiPlanar420ToPlanarInputs(pSrcY, nSrcYStep, pSrcCbCr, nSrcCbCrStep, pDst, rDstStep, oSizeROI);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  return toStatus(nppiSemiPlanar420ToPlanar420_8u_P2P3R_kernel(pSrcY, nSrcYStep, pSrcCbCr, nSrcCbCrStep, pDst[0],
                                                               rDstStep[0], pDst[2], rDstStep[2], pDst[1],
                                                               rDstStep[1], oSizeROI, nppStreamCtx.hStream));
}

NppStatus nppiYCbCr420ToYCrCb420_8u_P2P3R(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCbCr,
                                          int nSrcCbCrStep, Npp8u *pDst[3], int rDstStep[3], NppiSize oSizeROI) {
  NppStreamContext ctx{};
  ctx.hStream = 0;
  return nppiYCbCr420ToYCrCb420_8u_P2P3R_Ctx(pSrcY, nSrcYStep, pSrcCbCr, nSrcCbCrStep, pDst, rDstStep, oSizeROI,
                                             ctx);
}

NppStatus nppiYCbCr420ToYCbCr422_8u_P3R_Ctx(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst[3],
                                            int nDstStep[3], NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePlanar420ToPlanar422Inputs(pSrc, rSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  return toStatus(nppiPlanar420ToPlanar422_8u_P3P3R_kernel(pSrc[0], rSrcStep[0], pSrc[1], rSrcStep[1], pSrc[2],
                                                           rSrcStep[2], pDst[0], nDstStep[0], pDst[1], nDstStep[1],
                                                           pDst[2], nDstStep[2], oSizeROI, nppStreamCtx.hStream));
}

NppStatus nppiYCbCr420ToYCbCr422_8u_P3R(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst[3],
                                        int nDstStep[3], NppiSize oSizeROI) {
  NppStreamContext ctx{};
  ctx.hStream = 0;
  return nppiYCbCr420ToYCbCr422_8u_P3R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiYCbCr420ToYCbCr422_8u_P2P3R_Ctx(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCbCr,
                                              int nSrcCbCrStep, Npp8u *pDst[3], int rDstStep[3], NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx) {
  NppStatus status =
      validateSemiPlanar420ToPlanar422Inputs(pSrcY, nSrcYStep, pSrcCbCr, nSrcCbCrStep, pDst, rDstStep, oSizeROI);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  return toStatus(nppiSemiPlanar420ToPlanar422_8u_P2P3R_kernel(pSrcY, nSrcYStep, pSrcCbCr, nSrcCbCrStep, pDst[0],
                                                               rDstStep[0], pDst[1], rDstStep[1], pDst[2],
                                                               rDstStep[2], oSizeROI, nppStreamCtx.hStream));
}

NppStatus nppiYCbCr420ToYCbCr422_8u_P2P3R(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCbCr,
                                          int nSrcCbCrStep, Npp8u *pDst[3], int rDstStep[3], NppiSize oSizeROI) {
  NppStreamContext ctx{};
  ctx.hStream = 0;
  return nppiYCbCr420ToYCbCr422_8u_P2P3R_Ctx(pSrcY, nSrcYStep, pSrcCbCr, nSrcCbCrStep, pDst, rDstStep, oSizeROI,
                                             ctx);
}

NppStatus nppiYCbCr420ToYCbCr422_8u_P2C2R_Ctx(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCbCr,
                                              int nSrcCbCrStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx) {
  NppStatus status =
      validateSemiPlanar420ToPacked422Inputs(pSrcY, nSrcYStep, pSrcCbCr, nSrcCbCrStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  return toStatus(nppiSemiPlanar420ToPacked422_8u_P2C2R_kernel(pSrcY, nSrcYStep, pSrcCbCr, nSrcCbCrStep, pDst,
                                                               nDstStep, oSizeROI, 0, nppStreamCtx.hStream));
}

NppStatus nppiYCbCr420ToYCbCr422_8u_P2C2R(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCbCr,
                                          int nSrcCbCrStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx{};
  ctx.hStream = 0;
  return nppiYCbCr420ToYCbCr422_8u_P2C2R_Ctx(pSrcY, nSrcYStep, pSrcCbCr, nSrcCbCrStep, pDst, nDstStep, oSizeROI,
                                             ctx);
}

NppStatus nppiYCbCr420ToCbYCr422_8u_P2C2R_Ctx(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCbCr,
                                              int nSrcCbCrStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx) {
  NppStatus status =
      validateSemiPlanar420ToPacked422Inputs(pSrcY, nSrcYStep, pSrcCbCr, nSrcCbCrStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  return toStatus(nppiSemiPlanar420ToPacked422_8u_P2C2R_kernel(pSrcY, nSrcYStep, pSrcCbCr, nSrcCbCrStep, pDst,
                                                               nDstStep, oSizeROI, 1, nppStreamCtx.hStream));
}

NppStatus nppiYCbCr420ToCbYCr422_8u_P2C2R(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCbCr,
                                          int nSrcCbCrStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx{};
  ctx.hStream = 0;
  return nppiYCbCr420ToCbYCr422_8u_P2C2R_Ctx(pSrcY, nSrcYStep, pSrcCbCr, nSrcCbCrStep, pDst, nDstStep, oSizeROI,
                                             ctx);
}

NppStatus nppiYCrCb420ToCbYCr422_8u_P3C2R_Ctx(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                              NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePlanarYCrCb420ToPacked422Inputs(pSrc, rSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  return toStatus(nppiPlanarYCrCb420ToPackedCbYCr422_8u_P3C2R_kernel(pSrc[0], rSrcStep[0], pSrc[1], rSrcStep[1],
                                                                     pSrc[2], rSrcStep[2], pDst, nDstStep, oSizeROI,
                                                                     nppStreamCtx.hStream));
}

NppStatus nppiYCrCb420ToCbYCr422_8u_P3C2R(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                          NppiSize oSizeROI) {
  NppStreamContext ctx{};
  ctx.hStream = 0;
  return nppiYCrCb420ToCbYCr422_8u_P3C2R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiYCrCb420ToYCbCr411_8u_P3P2R_Ctx(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDstY,
                                              int nDstYStep, Npp8u *pDstCbCr, int nDstCbCrStep, NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx) {
  NppStatus status =
      validatePlanarYCrCb420ToSemiPlanar411Inputs(pSrc, rSrcStep, pDstY, nDstYStep, pDstCbCr, nDstCbCrStep, oSizeROI);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  return toStatus(nppiPlanarYCrCb420ToSemiPlanar411_8u_P3P2R_kernel(pSrc[0], rSrcStep[0], pSrc[1], rSrcStep[1],
                                                                    pSrc[2], rSrcStep[2], pDstY, nDstYStep, pDstCbCr,
                                                                    nDstCbCrStep, oSizeROI, nppStreamCtx.hStream));
}

NppStatus nppiYCrCb420ToYCbCr411_8u_P3P2R(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDstY, int nDstYStep,
                                          Npp8u *pDstCbCr, int nDstCbCrStep, NppiSize oSizeROI) {
  NppStreamContext ctx{};
  ctx.hStream = 0;
  return nppiYCrCb420ToYCbCr411_8u_P3P2R_Ctx(pSrc, rSrcStep, pDstY, nDstYStep, pDstCbCr, nDstCbCrStep, oSizeROI,
                                             ctx);
}

NppStatus nppiYCbCr420ToYCbCr411_8u_P3P2R_Ctx(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDstY,
                                              int nDstYStep, Npp8u *pDstCbCr, int nDstCbCrStep, NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx) {
  NppStatus status =
      validatePlanarYCrCb420ToSemiPlanar411Inputs(pSrc, rSrcStep, pDstY, nDstYStep, pDstCbCr, nDstCbCrStep, oSizeROI);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  return toStatus(nppiPlanarYCrCb420ToSemiPlanar411_8u_P3P2R_kernel(pSrc[0], rSrcStep[0], pSrc[2], rSrcStep[2],
                                                                    pSrc[1], rSrcStep[1], pDstY, nDstYStep, pDstCbCr,
                                                                    nDstCbCrStep, oSizeROI, nppStreamCtx.hStream));
}

NppStatus nppiYCbCr420ToYCbCr411_8u_P3P2R(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDstY, int nDstYStep,
                                          Npp8u *pDstCbCr, int nDstCbCrStep, NppiSize oSizeROI) {
  NppStreamContext ctx{};
  ctx.hStream = 0;
  return nppiYCbCr420ToYCbCr411_8u_P3P2R_Ctx(pSrc, rSrcStep, pDstY, nDstYStep, pDstCbCr, nDstCbCrStep, oSizeROI,
                                             ctx);
}

NppStatus nppiYCbCr420ToYCbCr411_8u_P2P3R_Ctx(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCbCr,
                                              int nSrcCbCrStep, Npp8u *pDst[3], int rDstStep[3], NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx) {
  NppStatus status =
      validateSemiPlanar420ToPlanar411Inputs(pSrcY, nSrcYStep, pSrcCbCr, nSrcCbCrStep, pDst, rDstStep, oSizeROI);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  return toStatus(nppiSemiPlanar420ToPlanar411_8u_P2P3R_kernel(pSrcY, nSrcYStep, pSrcCbCr, nSrcCbCrStep, pDst[0],
                                                               rDstStep[0], pDst[1], rDstStep[1], pDst[2],
                                                               rDstStep[2], oSizeROI, nppStreamCtx.hStream));
}

NppStatus nppiYCbCr420ToYCbCr411_8u_P2P3R(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCbCr,
                                          int nSrcCbCrStep, Npp8u *pDst[3], int rDstStep[3], NppiSize oSizeROI) {
  NppStreamContext ctx{};
  ctx.hStream = 0;
  return nppiYCbCr420ToYCbCr411_8u_P2P3R_Ctx(pSrcY, nSrcYStep, pSrcCbCr, nSrcCbCrStep, pDst, rDstStep, oSizeROI,
                                             ctx);
}
