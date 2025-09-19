#include "npp.h"
#include <cstring>
#include <cuda_runtime.h>

// Forward declarations for CUDA kernels
extern "C" {
cudaError_t nppiAnd_8u_C1R_kernel(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiAndC_8u_C1R_kernel(const Npp8u *pSrc, int nSrcStep, const Npp8u nConstant, Npp8u *pDst, int nDstStep,
                                   NppiSize oSizeROI, cudaStream_t stream);
}

NppStatus nppiAnd_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  // 参数验证
  if (!pSrc1 || !pSrc2 || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (nSrc1Step < oSizeROI.width || nSrc2Step < oSizeROI.width || nDstStep < oSizeROI.width) {
    return NPP_STEP_ERROR;
  }

  // 调用CUDA内核
  cudaError_t cudaStatus =
      nppiAnd_8u_C1R_kernel(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);

  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAnd_8u_C1R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                         int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0; // 默认流
  return nppiAnd_8u_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiAndC_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp8u nConstant, Npp8u *pDst, int nDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  // 参数验证
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (nSrcStep < oSizeROI.width || nDstStep < oSizeROI.width) {
    return NPP_STEP_ERROR;
  }

  // 调用CUDA内核
  cudaError_t cudaStatus =
      nppiAndC_8u_C1R_kernel(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);

  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAndC_8u_C1R(const Npp8u *pSrc, int nSrcStep, const Npp8u nConstant, Npp8u *pDst, int nDstStep,
                          NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0; // 默认流
  return nppiAndC_8u_C1R_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, ctx);
}