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

// BOUNDARY HANDLING:
// This implementation supports two modes controlled by MPP_FILTERBOX_ZERO_PADDING macro:
//
// 1. If MPP_FILTERBOX_ZERO_PADDING is defined (zero padding mode):
//    - Out-of-bounds pixels are treated as zeros
//    - Safe for processing images without surrounding context
//    - Suitable for edge detection and filtering at image boundaries
//
// 2. If MPP_FILTERBOX_ZERO_PADDING is NOT defined (direct access mode, default):
//    - Assumes valid data exists outside the ROI boundaries
//    - The caller must ensure pSrc points to a location within a larger buffer
//    - Accessing pixels at negative offsets and beyond ROI boundaries must be safe
//    - Better performance due to no boundary checks
//
// For a ROI at position (x,y) with filter anchor at (ax,ay), the kernel may access:
// - Minimum offset: (x - ax, y - ay)
// - Maximum offset: (x + maskWidth - ax - 1, y + maskHeight - ay - 1)

NppStatus nppiFilterBox_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                   NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  // Parameter validation
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (oMaskSize.width <= 0 || oMaskSize.height <= 0) {
    return NPP_MASK_SIZE_ERROR;
  }
#ifdef MPP_FILTERBOX_ZERO_PADDING
  // In zero padding mode, mask can be larger than ROI
#else
  // In direct access mode, check if mask is larger than ROI
  if (oMaskSize.width > oSizeROI.width || oMaskSize.height > oSizeROI.height) {
    return NPP_MASK_SIZE_ERROR;
  }
#endif
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
#ifdef MPP_FILTERBOX_ZERO_PADDING
  // In zero padding mode, mask can be larger than ROI
#else
  if (oMaskSize.width > oSizeROI.width || oMaskSize.height > oSizeROI.height) {
    return NPP_MASK_SIZE_ERROR;
  }
#endif
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
NppStatus nppiFilterBox_32f_C1R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                    NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor,
                                    NppStreamContext nppStreamCtx) {
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
#ifdef MPP_FILTERBOX_ZERO_PADDING
  // In zero padding mode, mask can be larger than ROI
#else
  if (oMaskSize.width > oSizeROI.width || oMaskSize.height > oSizeROI.height) {
    return NPP_MASK_SIZE_ERROR;
  }
#endif
  if (nSrcStep < static_cast<Npp32s>(oSizeROI.width * sizeof(Npp32f)) ||
      nDstStep < static_cast<Npp32s>(oSizeROI.width * sizeof(Npp32f))) {
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
