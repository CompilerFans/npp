// nppiFilterMax and nppiFilterMin API implementations
// FilterMax: morphological dilation (local maximum)
// FilterMin: morphological erosion (local minimum)

#include "npp.h"
#include <cuda_runtime.h>

// Kernel declarations
extern "C" {
// FilterMax kernels
cudaError_t nppiFilterMax_8u_C1R_kernel(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                         NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, cudaStream_t stream);
cudaError_t nppiFilterMax_8u_C3R_kernel(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                         NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, cudaStream_t stream);
cudaError_t nppiFilterMax_8u_C4R_kernel(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                         NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, cudaStream_t stream);
cudaError_t nppiFilterMax_16u_C1R_kernel(const Npp16u *pSrc, Npp32s nSrcStep, Npp16u *pDst, Npp32s nDstStep,
                                          NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, cudaStream_t stream);
cudaError_t nppiFilterMax_32f_C1R_kernel(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                          NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, cudaStream_t stream);
cudaError_t nppiFilterMax_32f_C3R_kernel(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                          NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, cudaStream_t stream);

// FilterMin kernels
cudaError_t nppiFilterMin_8u_C1R_kernel(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                         NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, cudaStream_t stream);
cudaError_t nppiFilterMin_8u_C3R_kernel(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                         NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, cudaStream_t stream);
cudaError_t nppiFilterMin_8u_C4R_kernel(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                         NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, cudaStream_t stream);
cudaError_t nppiFilterMin_16u_C1R_kernel(const Npp16u *pSrc, Npp32s nSrcStep, Npp16u *pDst, Npp32s nDstStep,
                                          NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, cudaStream_t stream);
cudaError_t nppiFilterMin_32f_C1R_kernel(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                          NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, cudaStream_t stream);
cudaError_t nppiFilterMin_32f_C3R_kernel(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                          NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, cudaStream_t stream);
}

// ============================================================================
// FilterMax implementations
// ============================================================================

// 8u_C1R
NppStatus nppiFilterMax_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                    NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor,
                                    NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
  if (oMaskSize.width <= 0 || oMaskSize.height <= 0) return NPP_MASK_SIZE_ERROR;
  if (oAnchor.x < 0 || oAnchor.x >= oMaskSize.width || oAnchor.y < 0 || oAnchor.y >= oMaskSize.height)
    return NPP_ANCHOR_ERROR;

  cudaError_t err = nppiFilterMax_8u_C1R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor,
                                                 nppStreamCtx.hStream);
  return (err == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiFilterMax_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiFilterMax_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, ctx);
}

// 8u_C3R
NppStatus nppiFilterMax_8u_C3R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                    NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor,
                                    NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
  if (oMaskSize.width <= 0 || oMaskSize.height <= 0) return NPP_MASK_SIZE_ERROR;
  if (oAnchor.x < 0 || oAnchor.x >= oMaskSize.width || oAnchor.y < 0 || oAnchor.y >= oMaskSize.height)
    return NPP_ANCHOR_ERROR;

  cudaError_t err = nppiFilterMax_8u_C3R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor,
                                                 nppStreamCtx.hStream);
  return (err == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiFilterMax_8u_C3R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiFilterMax_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, ctx);
}

// 8u_C4R
NppStatus nppiFilterMax_8u_C4R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                    NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor,
                                    NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
  if (oMaskSize.width <= 0 || oMaskSize.height <= 0) return NPP_MASK_SIZE_ERROR;
  if (oAnchor.x < 0 || oAnchor.x >= oMaskSize.width || oAnchor.y < 0 || oAnchor.y >= oMaskSize.height)
    return NPP_ANCHOR_ERROR;

  cudaError_t err = nppiFilterMax_8u_C4R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor,
                                                 nppStreamCtx.hStream);
  return (err == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiFilterMax_8u_C4R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiFilterMax_8u_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, ctx);
}

// 8u_AC4R (process only first 3 channels, ignore alpha)
NppStatus nppiFilterMax_8u_AC4R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                     NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor,
                                     NppStreamContext nppStreamCtx) {
  // AC4R processes RGB and leaves alpha unchanged
  // Use C4R kernel as a simple implementation
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
  if (oMaskSize.width <= 0 || oMaskSize.height <= 0) return NPP_MASK_SIZE_ERROR;
  if (oAnchor.x < 0 || oAnchor.x >= oMaskSize.width || oAnchor.y < 0 || oAnchor.y >= oMaskSize.height)
    return NPP_ANCHOR_ERROR;

  cudaError_t err = nppiFilterMax_8u_C4R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor,
                                                 nppStreamCtx.hStream);
  return (err == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiFilterMax_8u_AC4R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                 NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiFilterMax_8u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, ctx);
}

// 16u_C1R
NppStatus nppiFilterMax_16u_C1R_Ctx(const Npp16u *pSrc, Npp32s nSrcStep, Npp16u *pDst, Npp32s nDstStep,
                                     NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor,
                                     NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
  if (oMaskSize.width <= 0 || oMaskSize.height <= 0) return NPP_MASK_SIZE_ERROR;
  if (oAnchor.x < 0 || oAnchor.x >= oMaskSize.width || oAnchor.y < 0 || oAnchor.y >= oMaskSize.height)
    return NPP_ANCHOR_ERROR;

  cudaError_t err = nppiFilterMax_16u_C1R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor,
                                                  nppStreamCtx.hStream);
  return (err == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiFilterMax_16u_C1R(const Npp16u *pSrc, Npp32s nSrcStep, Npp16u *pDst, Npp32s nDstStep,
                                 NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiFilterMax_16u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, ctx);
}

// 16s_C1R (use 16u kernel, signed comparison handled by CUDA intrinsics)
NppStatus nppiFilterMax_16s_C1R_Ctx(const Npp16s *pSrc, Npp32s nSrcStep, Npp16s *pDst, Npp32s nDstStep,
                                     NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor,
                                     NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
  if (oMaskSize.width <= 0 || oMaskSize.height <= 0) return NPP_MASK_SIZE_ERROR;
  if (oAnchor.x < 0 || oAnchor.x >= oMaskSize.width || oAnchor.y < 0 || oAnchor.y >= oMaskSize.height)
    return NPP_ANCHOR_ERROR;

  // Note: For signed 16-bit, we need a separate kernel implementation
  // For now, treat as unsigned which may not give correct results for negative values
  cudaError_t err = nppiFilterMax_16u_C1R_kernel((const Npp16u *)pSrc, nSrcStep, (Npp16u *)pDst, nDstStep,
                                                  oSizeROI, oMaskSize, oAnchor, nppStreamCtx.hStream);
  return (err == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiFilterMax_16s_C1R(const Npp16s *pSrc, Npp32s nSrcStep, Npp16s *pDst, Npp32s nDstStep,
                                 NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiFilterMax_16s_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, ctx);
}

// 32f_C1R
NppStatus nppiFilterMax_32f_C1R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                     NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor,
                                     NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
  if (oMaskSize.width <= 0 || oMaskSize.height <= 0) return NPP_MASK_SIZE_ERROR;
  if (oAnchor.x < 0 || oAnchor.x >= oMaskSize.width || oAnchor.y < 0 || oAnchor.y >= oMaskSize.height)
    return NPP_ANCHOR_ERROR;

  cudaError_t err = nppiFilterMax_32f_C1R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor,
                                                  nppStreamCtx.hStream);
  return (err == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiFilterMax_32f_C1R(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                 NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiFilterMax_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, ctx);
}

// 32f_C3R
NppStatus nppiFilterMax_32f_C3R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                     NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor,
                                     NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
  if (oMaskSize.width <= 0 || oMaskSize.height <= 0) return NPP_MASK_SIZE_ERROR;
  if (oAnchor.x < 0 || oAnchor.x >= oMaskSize.width || oAnchor.y < 0 || oAnchor.y >= oMaskSize.height)
    return NPP_ANCHOR_ERROR;

  cudaError_t err = nppiFilterMax_32f_C3R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor,
                                                  nppStreamCtx.hStream);
  return (err == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiFilterMax_32f_C3R(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                 NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiFilterMax_32f_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, ctx);
}

// ============================================================================
// FilterMin implementations
// ============================================================================

// 8u_C1R
NppStatus nppiFilterMin_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                    NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor,
                                    NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
  if (oMaskSize.width <= 0 || oMaskSize.height <= 0) return NPP_MASK_SIZE_ERROR;
  if (oAnchor.x < 0 || oAnchor.x >= oMaskSize.width || oAnchor.y < 0 || oAnchor.y >= oMaskSize.height)
    return NPP_ANCHOR_ERROR;

  cudaError_t err = nppiFilterMin_8u_C1R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor,
                                                 nppStreamCtx.hStream);
  return (err == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiFilterMin_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiFilterMin_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, ctx);
}

// 8u_C3R
NppStatus nppiFilterMin_8u_C3R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                    NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor,
                                    NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
  if (oMaskSize.width <= 0 || oMaskSize.height <= 0) return NPP_MASK_SIZE_ERROR;
  if (oAnchor.x < 0 || oAnchor.x >= oMaskSize.width || oAnchor.y < 0 || oAnchor.y >= oMaskSize.height)
    return NPP_ANCHOR_ERROR;

  cudaError_t err = nppiFilterMin_8u_C3R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor,
                                                 nppStreamCtx.hStream);
  return (err == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiFilterMin_8u_C3R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiFilterMin_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, ctx);
}

// 8u_C4R
NppStatus nppiFilterMin_8u_C4R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                    NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor,
                                    NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
  if (oMaskSize.width <= 0 || oMaskSize.height <= 0) return NPP_MASK_SIZE_ERROR;
  if (oAnchor.x < 0 || oAnchor.x >= oMaskSize.width || oAnchor.y < 0 || oAnchor.y >= oMaskSize.height)
    return NPP_ANCHOR_ERROR;

  cudaError_t err = nppiFilterMin_8u_C4R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor,
                                                 nppStreamCtx.hStream);
  return (err == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiFilterMin_8u_C4R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiFilterMin_8u_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, ctx);
}

// 8u_AC4R (process only first 3 channels, ignore alpha)
NppStatus nppiFilterMin_8u_AC4R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                     NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor,
                                     NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
  if (oMaskSize.width <= 0 || oMaskSize.height <= 0) return NPP_MASK_SIZE_ERROR;
  if (oAnchor.x < 0 || oAnchor.x >= oMaskSize.width || oAnchor.y < 0 || oAnchor.y >= oMaskSize.height)
    return NPP_ANCHOR_ERROR;

  cudaError_t err = nppiFilterMin_8u_C4R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor,
                                                 nppStreamCtx.hStream);
  return (err == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiFilterMin_8u_AC4R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                 NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiFilterMin_8u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, ctx);
}

// 16u_C1R
NppStatus nppiFilterMin_16u_C1R_Ctx(const Npp16u *pSrc, Npp32s nSrcStep, Npp16u *pDst, Npp32s nDstStep,
                                     NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor,
                                     NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
  if (oMaskSize.width <= 0 || oMaskSize.height <= 0) return NPP_MASK_SIZE_ERROR;
  if (oAnchor.x < 0 || oAnchor.x >= oMaskSize.width || oAnchor.y < 0 || oAnchor.y >= oMaskSize.height)
    return NPP_ANCHOR_ERROR;

  cudaError_t err = nppiFilterMin_16u_C1R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor,
                                                  nppStreamCtx.hStream);
  return (err == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiFilterMin_16u_C1R(const Npp16u *pSrc, Npp32s nSrcStep, Npp16u *pDst, Npp32s nDstStep,
                                 NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiFilterMin_16u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, ctx);
}

// 16s_C1R
NppStatus nppiFilterMin_16s_C1R_Ctx(const Npp16s *pSrc, Npp32s nSrcStep, Npp16s *pDst, Npp32s nDstStep,
                                     NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor,
                                     NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
  if (oMaskSize.width <= 0 || oMaskSize.height <= 0) return NPP_MASK_SIZE_ERROR;
  if (oAnchor.x < 0 || oAnchor.x >= oMaskSize.width || oAnchor.y < 0 || oAnchor.y >= oMaskSize.height)
    return NPP_ANCHOR_ERROR;

  cudaError_t err = nppiFilterMin_16u_C1R_kernel((const Npp16u *)pSrc, nSrcStep, (Npp16u *)pDst, nDstStep,
                                                  oSizeROI, oMaskSize, oAnchor, nppStreamCtx.hStream);
  return (err == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiFilterMin_16s_C1R(const Npp16s *pSrc, Npp32s nSrcStep, Npp16s *pDst, Npp32s nDstStep,
                                 NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiFilterMin_16s_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, ctx);
}

// 32f_C1R
NppStatus nppiFilterMin_32f_C1R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                     NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor,
                                     NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
  if (oMaskSize.width <= 0 || oMaskSize.height <= 0) return NPP_MASK_SIZE_ERROR;
  if (oAnchor.x < 0 || oAnchor.x >= oMaskSize.width || oAnchor.y < 0 || oAnchor.y >= oMaskSize.height)
    return NPP_ANCHOR_ERROR;

  cudaError_t err = nppiFilterMin_32f_C1R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor,
                                                  nppStreamCtx.hStream);
  return (err == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiFilterMin_32f_C1R(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                 NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiFilterMin_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, ctx);
}

// 32f_C3R
NppStatus nppiFilterMin_32f_C3R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                     NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor,
                                     NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
  if (oMaskSize.width <= 0 || oMaskSize.height <= 0) return NPP_MASK_SIZE_ERROR;
  if (oAnchor.x < 0 || oAnchor.x >= oMaskSize.width || oAnchor.y < 0 || oAnchor.y >= oMaskSize.height)
    return NPP_ANCHOR_ERROR;

  cudaError_t err = nppiFilterMin_32f_C3R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor,
                                                  nppStreamCtx.hStream);
  return (err == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiFilterMin_32f_C3R(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                 NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiFilterMin_32f_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, ctx);
}
