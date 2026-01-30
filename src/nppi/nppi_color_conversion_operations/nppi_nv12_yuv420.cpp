#include "npp.h"
#include <cuda_runtime.h>

extern "C" {
cudaError_t nppiNV12ToYUV420_8u_P2P3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcUV, int nSrcUVStep,
                                             Npp8u *pDstY, int nDstYStep, Npp8u *pDstU, int nDstUStep,
                                             Npp8u *pDstV, int nDstVStep, NppiSize oSizeROI, cudaStream_t stream);
}

static NppStatus validateNV12ToYUV420Inputs(const Npp8u *const pSrc[2], int nSrcStep, Npp8u *pDst[3],
                                            int aDstStep[3], NppiSize oSizeROI) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pDst || !pDst[0] || !pDst[1] || !pDst[2]) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nSrcStep <= 0) {
    return NPP_STEP_ERROR;
  }
  if (!aDstStep || aDstStep[0] <= 0 || aDstStep[1] <= 0 || aDstStep[2] <= 0) {
    return NPP_STEP_ERROR;
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_WRONG_INTERSECTION_ROI_ERROR;
  }
  if ((oSizeROI.width % 2) != 0 || (oSizeROI.height % 2) != 0) {
    return NPP_WRONG_INTERSECTION_ROI_ERROR;
  }
  return NPP_NO_ERROR;
}

NppStatus nppiNV12ToYUV420_8u_P2P3R_Ctx(const Npp8u *const pSrc[2], int nSrcStep, Npp8u *pDst[3], int aDstStep[3],
                                        NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateNV12ToYUV420Inputs(pSrc, nSrcStep, pDst, aDstStep, oSizeROI);
  if (status != NPP_NO_ERROR) {
    return status;
  }

  cudaError_t cudaStatus = nppiNV12ToYUV420_8u_P2P3R_kernel(pSrc[0], nSrcStep, pSrc[1], nSrcStep, pDst[0], aDstStep[0],
                                                            pDst[1], aDstStep[1], pDst[2], aDstStep[2], oSizeROI,
                                                            nppStreamCtx.hStream);

  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiNV12ToYUV420_8u_P2P3R(const Npp8u *const pSrc[2], int nSrcStep, Npp8u *pDst[3], int aDstStep[3],
                                    NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiNV12ToYUV420_8u_P2P3R_Ctx(pSrc, nSrcStep, pDst, aDstStep, oSizeROI, ctx);
}
