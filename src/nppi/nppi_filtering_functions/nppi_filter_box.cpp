#include "npp.h"
#include <cstring>
#include <cuda_runtime.h>

// Kernel declarations
extern "C" {
cudaError_t nppiFilterBox_8u_C1R_kernel(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                        NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, cudaStream_t stream);
cudaError_t nppiFilterBox_8u_C4R_kernel(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                        NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, cudaStream_t stream);
cudaError_t nppiFilterBox_32f_C1R_kernel(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                         NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, cudaStream_t stream);
}

NppStatus nppiFilterBox_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                   NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  // Parameter validation
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
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
  // Validate anchor within mask bounds
  if (oAnchor.x < 0 || oAnchor.x >= oMaskSize.width || oAnchor.y < 0 || oAnchor.y >= oMaskSize.height) {
    return NPP_ANCHOR_ERROR;
  }

  // Call GPU kernel
  cudaError_t cudaStatus =
      nppiFilterBox_8u_C1R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx.hStream);

  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiFilterBox_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                               NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext ctx;
  ctx.hStream = 0; // Default stream
  return nppiFilterBox_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, ctx);
}

// nppiFilterBox_8u_C4R implementation
NppStatus nppiFilterBox_8u_C4R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                   NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  // Parameter validation
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  if (oMaskSize.width <= 0 || oMaskSize.height <= 0) {
    return NPP_MASK_SIZE_ERROR;
  }
  if (oMaskSize.width > oSizeROI.width || oMaskSize.height > oSizeROI.height) {
    return NPP_MASK_SIZE_ERROR;
  }
  if (nSrcStep < oSizeROI.width * 4 || nDstStep < oSizeROI.width * 4) {
    return NPP_STEP_ERROR;
  }
  if (oAnchor.x < 0 || oAnchor.x >= oMaskSize.width || oAnchor.y < 0 || oAnchor.y >= oMaskSize.height) {
    return NPP_ANCHOR_ERROR;
  }

  // Call GPU kernel
  cudaError_t cudaStatus =
      nppiFilterBox_8u_C4R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx.hStream);

  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiFilterBox_8u_C4R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                               NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext ctx;
  ctx.hStream = 0; // Default stream
  return nppiFilterBox_8u_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, ctx);
}

// nppiFilterBox_32f_C1R implementation
NppStatus nppiFilterBox_32f_C1R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                    NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  // Parameter validation
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  if (oMaskSize.width <= 0 || oMaskSize.height <= 0) {
    return NPP_MASK_SIZE_ERROR;
  }
  if (oMaskSize.width > oSizeROI.width || oMaskSize.height > oSizeROI.height) {
    return NPP_MASK_SIZE_ERROR;
  }
  if (nSrcStep < static_cast<Npp32s>(oSizeROI.width * sizeof(Npp32f)) || nDstStep < static_cast<Npp32s>(oSizeROI.width * sizeof(Npp32f))) {
    return NPP_STEP_ERROR;
  }
  if (oAnchor.x < 0 || oAnchor.x >= oMaskSize.width || oAnchor.y < 0 || oAnchor.y >= oMaskSize.height) {
    return NPP_ANCHOR_ERROR;
  }

  // Call GPU kernel
  cudaError_t cudaStatus =
      nppiFilterBox_32f_C1R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx.hStream);

  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiFilterBox_32f_C1R(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext ctx;
  ctx.hStream = 0; // Default stream
  return nppiFilterBox_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, ctx);
}