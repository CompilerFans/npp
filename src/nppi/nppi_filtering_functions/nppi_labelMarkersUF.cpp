#include "../../npp_internal.h"
#include "npp.h"
#include <cuda_runtime.h>

// Forward declarations for GPU functions (only _impl functions need extern "C")
extern "C" {
NppStatus nppiLabelMarkersUFGetBufferSize_32u_C1R_Ctx_impl(NppiSize oSizeROI, int *hpBufferSize);
NppStatus nppiLabelMarkersUF_8u32u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep,
                                                 Npp32u *pDst, int nDstStep,
                                                 NppiSize oSizeROI, NppiNorm eNorm,
                                                 Npp8u *pBuffer,
                                                 NppStreamContext nppStreamCtx);
NppStatus nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx_impl(
    const NppiImageDescriptor *pSrcBatchList,
    NppiImageDescriptor *pDstBatchList,
    int nBatchSize, NppiSize oMaxSizeROI,
    NppiNorm eNorm, NppStreamContext nppStreamCtx);
NppStatus nppiLabelMarkersUF_16u32u_C1R_Ctx_impl(const Npp16u *pSrc, int nSrcStep,
                                                  Npp32u *pDst, int nDstStep, NppiSize oSizeROI,
                                                  NppiNorm eNorm, Npp8u *pBuffer,
                                                  NppStreamContext nppStreamCtx);
NppStatus nppiLabelMarkersUF_32u_C1R_Ctx_impl(const Npp32u *pSrc, int nSrcStep, Npp32u *pDst,
                                               int nDstStep, NppiSize oSizeROI, NppiNorm eNorm,
                                               Npp8u *pBuffer, NppStreamContext nppStreamCtx);
NppStatus nppiLabelMarkersUFBatch_16u32u_C1R_Advanced_Ctx_impl(
    const NppiImageDescriptor *pSrcBatchList, NppiImageDescriptor *pDstBatchList, int nBatchSize,
    NppiSize oMaxSizeROI, NppiNorm eNorm, NppStreamContext nppStreamCtx);
NppStatus nppiLabelMarkersUFBatch_32u_C1R_Advanced_Ctx_impl(
    const NppiImageDescriptor *pSrcBatchList, NppiImageDescriptor *pDstBatchList, int nBatchSize,
    NppiSize oMaxSizeROI, NppiNorm eNorm, NppStreamContext nppStreamCtx);
}

namespace {

NppStatus labelMarkersValidateSingle(const void *pSrc, const Npp32u *pDst, const Npp8u *pBuffer,
                                     int nSrcStep, int nDstStep, NppiSize oSizeROI,
                                     NppiNorm eNorm, NppStreamContext ctx) {
  if (pSrc == nullptr || pDst == nullptr || pBuffer == nullptr) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
  if (nSrcStep <= 0 || nDstStep <= 0) return NPP_STEP_ERROR;
  if (eNorm != nppiNormL1 && eNorm != nppiNormInf) return NPP_BAD_ARGUMENT_ERROR;
  if (ctx.nCudaDeviceId < -1) return NPP_BAD_ARGUMENT_ERROR;
  return NPP_NO_ERROR;
}

// Batch statuses measured on the reference (NPP 13.4, measurement): a
// negative ROI or batch count is NPP_SIZE_ERROR, while a zero ROI or a zero
// batch count surfaces as NPP_CUDA_KERNEL_EXECUTION_ERROR; a batch size of 1 is
// accepted (the header says > 1, NPP does not enforce it) and an unknown
// NppiNorm is not rejected (it runs the 4-way path).
NppStatus labelMarkersValidateBatch(const NppiImageDescriptor *pSrcBatchList,
                                    const NppiImageDescriptor *pDstBatchList, int nBatchSize,
                                    NppiSize oMaxSizeROI, NppiNorm eNorm, NppStreamContext ctx) {
  (void)eNorm;
  if (pSrcBatchList == nullptr || pDstBatchList == nullptr) return NPP_NULL_POINTER_ERROR;
  if (oMaxSizeROI.width < 0 || oMaxSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (nBatchSize < 0) return NPP_SIZE_ERROR;
  if (nBatchSize == 0 || oMaxSizeROI.width == 0 || oMaxSizeROI.height == 0) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
  if (ctx.nCudaDeviceId < -1) return NPP_BAD_ARGUMENT_ERROR;
  return NPP_NO_ERROR;
}

template <typename Src>
NppStatus labelMarkersSingleDispatch(Src *pSrc, int nSrcStep, Npp32u *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppiNorm eNorm, Npp8u *pBuffer,
                                     NppStreamContext ctx,
                                     NppStatus (*impl)(const Src *, int, Npp32u *, int, NppiSize,
                                                       NppiNorm, Npp8u *, NppStreamContext)) {
  NppStatus st = labelMarkersValidateSingle(pSrc, pDst, pBuffer, nSrcStep, nDstStep, oSizeROI, eNorm,
                                            ctx);
  if (st != NPP_NO_ERROR) return st;
  return impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eNorm, pBuffer, ctx);
}

template <typename Src>
NppStatus labelMarkersBatchDispatch(const NppiImageDescriptor *pSrcBatchList,
                                    NppiImageDescriptor *pDstBatchList, int nBatchSize,
                                    NppiSize oMaxSizeROI, NppiNorm eNorm, NppStreamContext ctx,
                                    NppStatus (*impl)(const NppiImageDescriptor *,
                                                      NppiImageDescriptor *, int, NppiSize,
                                                      NppiNorm, NppStreamContext)) {
  NppStatus st =
      labelMarkersValidateBatch(pSrcBatchList, pDstBatchList, nBatchSize, oMaxSizeROI, eNorm, ctx);
  if (st != NPP_NO_ERROR) return st;
  return impl(pSrcBatchList, pDstBatchList, nBatchSize, oMaxSizeROI, eNorm, ctx);
}

}  // namespace

// Get required buffer size for label markers
NppStatus nppiLabelMarkersUFGetBufferSize_32u_C1R(NppiSize oSizeROI, int *hpBufferSize) {
  if (hpBufferSize == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  return nppiLabelMarkersUFGetBufferSize_32u_C1R_Ctx_impl(oSizeROI, hpBufferSize);
}

// Single image label markers using Union-Find (non-Ctx version)
NppStatus nppiLabelMarkersUF_8u32u_C1R(Npp8u *pSrc, int nSrcStep,
                                        Npp32u *pDst, int nDstStep,
                                        NppiSize oSizeROI, NppiNorm eNorm,
                                        Npp8u *pBuffer) {
  // Parameter validation
  if (pSrc == nullptr || pDst == nullptr || pBuffer == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  // Validate norm type (only L1 and Inf are supported)
  if (eNorm != nppiNormL1 && eNorm != nppiNormInf) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  NppStreamContext nppStreamCtx = nppCreateDefaultStreamContext();

  return nppiLabelMarkersUF_8u32u_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep,
                                                oSizeROI, eNorm, pBuffer, nppStreamCtx);
}

// Single image label markers using Union-Find (Ctx version)
NppStatus nppiLabelMarkersUF_8u32u_C1R_Ctx(Npp8u *pSrc, int nSrcStep,
                                            Npp32u *pDst, int nDstStep,
                                            NppiSize oSizeROI, NppiNorm eNorm,
                                            Npp8u *pBuffer,
                                            NppStreamContext nppStreamCtx) {
  // Parameter validation
  if (pSrc == nullptr || pDst == nullptr || pBuffer == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  // Validate norm type (only L1 and Inf are supported)
  if (eNorm != nppiNormL1 && eNorm != nppiNormInf) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  // Validate stream context
  if (nppStreamCtx.nCudaDeviceId < -1) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  return nppiLabelMarkersUF_8u32u_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep,
                                                oSizeROI, eNorm, pBuffer, nppStreamCtx);
}

// Batch image label markers using Union-Find (Advanced Ctx version)
NppStatus nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx(
    const NppiImageDescriptor *pSrcBatchList,
    NppiImageDescriptor *pDstBatchList,
    int nBatchSize, NppiSize oMaxSizeROI,
    NppiNorm eNorm, NppStreamContext nppStreamCtx) {
  // Parameter validation (see labelMarkersValidateBatch for the measured NPP
  // statuses).
  NppStatus vst = labelMarkersValidateBatch(pSrcBatchList, pDstBatchList, nBatchSize, oMaxSizeROI,
                                            eNorm, nppStreamCtx);
  if (vst != NPP_NO_ERROR) {
    return vst;
  }

  return nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx_impl(
      pSrcBatchList, pDstBatchList, nBatchSize, oMaxSizeROI, eNorm, nppStreamCtx);
}

// ---------------------------------------------------------------------------
// 16u / 32u source variants and the non-Advanced batch entry points.
// ---------------------------------------------------------------------------
NppStatus nppiLabelMarkersUF_16u32u_C1R_Ctx(Npp16u *pSrc, int nSrcStep, Npp32u *pDst,
                                            int nDstStep, NppiSize oSizeROI, NppiNorm eNorm,
                                            Npp8u *pBuffer, NppStreamContext nppStreamCtx) {
  return labelMarkersSingleDispatch<Npp16u>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eNorm,
                                            pBuffer, nppStreamCtx,
                                            nppiLabelMarkersUF_16u32u_C1R_Ctx_impl);
}

NppStatus nppiLabelMarkersUF_16u32u_C1R(Npp16u *pSrc, int nSrcStep, Npp32u *pDst, int nDstStep,
                                        NppiSize oSizeROI, NppiNorm eNorm, Npp8u *pBuffer) {
  return nppiLabelMarkersUF_16u32u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eNorm, pBuffer,
                                           nppCreateDefaultStreamContext());
}

NppStatus nppiLabelMarkersUF_32u_C1R_Ctx(Npp32u *pSrc, int nSrcStep, Npp32u *pDst, int nDstStep,
                                         NppiSize oSizeROI, NppiNorm eNorm, Npp8u *pBuffer,
                                         NppStreamContext nppStreamCtx) {
  return labelMarkersSingleDispatch<Npp32u>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eNorm,
                                            pBuffer, nppStreamCtx,
                                            nppiLabelMarkersUF_32u_C1R_Ctx_impl);
}

NppStatus nppiLabelMarkersUF_32u_C1R(Npp32u *pSrc, int nSrcStep, Npp32u *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppiNorm eNorm, Npp8u *pBuffer) {
  return nppiLabelMarkersUF_32u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eNorm, pBuffer,
                                        nppCreateDefaultStreamContext());
}

NppStatus nppiLabelMarkersUFBatch_16u32u_C1R_Advanced_Ctx(
    const NppiImageDescriptor *pSrcBatchList, NppiImageDescriptor *pDstBatchList, int nBatchSize,
    NppiSize oMaxSizeROI, NppiNorm eNorm, NppStreamContext nppStreamCtx) {
  return labelMarkersBatchDispatch<Npp16u>(pSrcBatchList, pDstBatchList, nBatchSize, oMaxSizeROI,
                                           eNorm, nppStreamCtx,
                                           nppiLabelMarkersUFBatch_16u32u_C1R_Advanced_Ctx_impl);
}

NppStatus nppiLabelMarkersUFBatch_8u32u_C1R_Advanced(const NppiImageDescriptor *pSrcBatchList,
                                                     NppiImageDescriptor *pDstBatchList, int nBatchSize,
                                                     NppiSize oMaxSizeROI, NppiNorm eNorm) {
  return nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx(pSrcBatchList, pDstBatchList, nBatchSize,
                                                        oMaxSizeROI, eNorm,
                                                        nppCreateDefaultStreamContext());
}

NppStatus nppiLabelMarkersUFBatch_16u32u_C1R_Advanced(const NppiImageDescriptor *pSrcBatchList,
                                                      NppiImageDescriptor *pDstBatchList,
                                                      int nBatchSize, NppiSize oMaxSizeROI,
                                                      NppiNorm eNorm) {
  return nppiLabelMarkersUFBatch_16u32u_C1R_Advanced_Ctx(
      pSrcBatchList, pDstBatchList, nBatchSize, oMaxSizeROI, eNorm,
      nppCreateDefaultStreamContext());
}

NppStatus nppiLabelMarkersUFBatch_32u_C1R_Advanced_Ctx(
    const NppiImageDescriptor *pSrcBatchList, NppiImageDescriptor *pDstBatchList, int nBatchSize,
    NppiSize oMaxSizeROI, NppiNorm eNorm, NppStreamContext nppStreamCtx) {
  return labelMarkersBatchDispatch<Npp32u>(pSrcBatchList, pDstBatchList, nBatchSize, oMaxSizeROI,
                                           eNorm, nppStreamCtx,
                                           nppiLabelMarkersUFBatch_32u_C1R_Advanced_Ctx_impl);
}

NppStatus nppiLabelMarkersUFBatch_32u_C1R_Advanced(const NppiImageDescriptor *pSrcBatchList,
                                                   NppiImageDescriptor *pDstBatchList,
                                                   int nBatchSize, NppiSize oMaxSizeROI,
                                                   NppiNorm eNorm) {
  return nppiLabelMarkersUFBatch_32u_C1R_Advanced_Ctx(
      pSrcBatchList, pDstBatchList, nBatchSize, oMaxSizeROI, eNorm,
      nppCreateDefaultStreamContext());
}

// Non-Advanced batch entry points: oSizeROI is the common per-image ROI, which
// is exactly the largest ROI the Advanced form expects.
NppStatus nppiLabelMarkersUFBatch_8u32u_C1R_Ctx(const NppiImageDescriptor *pSrcBatchList,
                                                NppiImageDescriptor *pDstBatchList, int nBatchSize,
                                                NppiSize oSizeROI, NppiNorm eNorm,
                                                NppStreamContext nppStreamCtx) {
  return nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx(pSrcBatchList, pDstBatchList, nBatchSize,
                                                        oSizeROI, eNorm, nppStreamCtx);
}

NppStatus nppiLabelMarkersUFBatch_8u32u_C1R(const NppiImageDescriptor *pSrcBatchList,
                                            NppiImageDescriptor *pDstBatchList, int nBatchSize,
                                            NppiSize oSizeROI, NppiNorm eNorm) {
  return nppiLabelMarkersUFBatch_8u32u_C1R_Ctx(pSrcBatchList, pDstBatchList, nBatchSize, oSizeROI,
                                               eNorm, nppCreateDefaultStreamContext());
}

NppStatus nppiLabelMarkersUFBatch_16u32u_C1R_Ctx(const NppiImageDescriptor *pSrcBatchList,
                                                 NppiImageDescriptor *pDstBatchList, int nBatchSize,
                                                 NppiSize oSizeROI, NppiNorm eNorm,
                                                 NppStreamContext nppStreamCtx) {
  return nppiLabelMarkersUFBatch_16u32u_C1R_Advanced_Ctx(pSrcBatchList, pDstBatchList, nBatchSize,
                                                         oSizeROI, eNorm, nppStreamCtx);
}

NppStatus nppiLabelMarkersUFBatch_16u32u_C1R(const NppiImageDescriptor *pSrcBatchList,
                                             NppiImageDescriptor *pDstBatchList, int nBatchSize,
                                             NppiSize oSizeROI, NppiNorm eNorm) {
  return nppiLabelMarkersUFBatch_16u32u_C1R_Ctx(pSrcBatchList, pDstBatchList, nBatchSize, oSizeROI,
                                                eNorm, nppCreateDefaultStreamContext());
}

NppStatus nppiLabelMarkersUFBatch_32u_C1R_Ctx(const NppiImageDescriptor *pSrcBatchList,
                                              NppiImageDescriptor *pDstBatchList, int nBatchSize,
                                              NppiSize oSizeROI, NppiNorm eNorm,
                                              NppStreamContext nppStreamCtx) {
  return nppiLabelMarkersUFBatch_32u_C1R_Advanced_Ctx(pSrcBatchList, pDstBatchList, nBatchSize,
                                                      oSizeROI, eNorm, nppStreamCtx);
}

NppStatus nppiLabelMarkersUFBatch_32u_C1R(const NppiImageDescriptor *pSrcBatchList,
                                          NppiImageDescriptor *pDstBatchList, int nBatchSize,
                                          NppiSize oSizeROI, NppiNorm eNorm) {
  return nppiLabelMarkersUFBatch_32u_C1R_Ctx(pSrcBatchList, pDstBatchList, nBatchSize, oSizeROI,
                                             eNorm, nppCreateDefaultStreamContext());
}
