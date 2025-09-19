#include "npp.h"
#include <cstring>
#include <cuda_runtime.h>

// Forward declarations for CUDA kernels
extern "C" {
cudaError_t nppiFilterBox_8u_C1R_kernel(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                        NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, cudaStream_t stream);
}

NppStatus nppiFilterBox_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                   NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  // 参数验证
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (oMaskSize.width <= 0 || oMaskSize.height <= 0) {
    return NPP_MASK_SIZE_ERROR;
  }
  if (oMaskSize.width > oSizeROI.width || oMaskSize.height > oSizeROI.height) {
    return NPP_MASK_SIZE_ERROR;
  }
  if (nSrcStep < oSizeROI.width || nDstStep < oSizeROI.width) {
    return NPP_STEP_ERROR;
  }
  // 验证锚点在掩码范围内
  if (oAnchor.x < 0 || oAnchor.x >= oMaskSize.width || oAnchor.y < 0 || oAnchor.y >= oMaskSize.height) {
    return NPP_ANCHOR_ERROR;
  }

  // 调用CUDA内核
  cudaError_t cudaStatus =
      nppiFilterBox_8u_C1R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx.hStream);

  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiFilterBox_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                               NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext ctx;
  ctx.hStream = 0; // 默认流
  return nppiFilterBox_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, ctx);
}