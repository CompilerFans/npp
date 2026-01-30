#include "npp.h"
#include "nppi_color_conversion_validation.hpp"
#include <cuda_runtime.h>

extern "C" {
cudaError_t nppiRGBToHSV_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream);
cudaError_t nppiRGBToHSV_8u_AC4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                        cudaStream_t stream);
cudaError_t nppiHSVToRGB_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream);
cudaError_t nppiHSVToRGB_8u_AC4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                        cudaStream_t stream);
}

NppStatus nppiRGBToHSV_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiRGBToHSV_8u_C3R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiRGBToHSV_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiRGBToHSV_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiRGBToHSV_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus =
      nppiRGBToHSV_8u_AC4R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiRGBToHSV_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiRGBToHSV_8u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiHSVToRGB_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiHSVToRGB_8u_C3R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiHSVToRGB_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiHSVToRGB_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiHSVToRGB_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus =
      nppiHSVToRGB_8u_AC4R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiHSVToRGB_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiHSVToRGB_8u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}
