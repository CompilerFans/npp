#include "npp.h"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

/**
 * NPP Image Filter Functions Implementation
 * Implements general 2D convolution nppiFilter functions
 */

// Forward declarations for mpp host func implementations
extern "C" {
NppStatus nppiFilter_8u_C1R_Ctx_cuda(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                     const Npp32s *pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor,
                                     NppStreamContext nppStreamCtx);
NppStatus nppiFilter_8u_C3R_Ctx_cuda(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                     const Npp32s *pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor,
                                     NppStreamContext nppStreamCtx);
NppStatus nppiFilter_32f_C1R_Ctx_cuda(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                      const Npp32f *pKernel, NppiSize oKernelSize, NppiPoint oAnchor,
                                      NppStreamContext nppStreamCtx);
}

// Input validation helper
static inline NppStatus validateFilterInputs(const void *pSrc, int nSrcStep, void *pDst, int nDstStep,
                                             NppiSize oSizeROI, const void *pKernel, NppiSize oKernelSize,
                                             NppiPoint oAnchor) {
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (!pSrc || !pDst || !pKernel) {
    return NPP_NULL_POINTER_ERROR;
  }

  // Validate kernel size
  if (oKernelSize.width <= 0 || oKernelSize.height <= 0 || oKernelSize.width % 2 == 0 || oKernelSize.height % 2 == 0) {
    return NPP_MASK_SIZE_ERROR;
  }

  // Validate anchor point
  if (oAnchor.x < 0 || oAnchor.x >= oKernelSize.width || oAnchor.y < 0 || oAnchor.y >= oKernelSize.height) {
    return NPP_ANCHOR_ERROR;
  }

  return NPP_SUCCESS;
}

// 8-bit unsigned single channel 2D convolution
NppStatus nppiFilter_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                const Npp32s *pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor,
                                NppStreamContext nppStreamCtx) {
  NppStatus status = validateFilterInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor);
  if (status != NPP_SUCCESS) {
    return status;
  }

  // Check divisor
  if (nDivisor == 0) {
    return NPP_DIVISOR_ERROR;
  }

  return nppiFilter_8u_C1R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nDivisor,
                                    nppStreamCtx);
}

NppStatus nppiFilter_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                            const Npp32s *pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiFilter_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nDivisor,
                               nppStreamCtx);
}

// 8-bit unsigned three channel 2D convolution
NppStatus nppiFilter_8u_C3R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                const Npp32s *pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor,
                                NppStreamContext nppStreamCtx) {
  NppStatus status = validateFilterInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor);
  if (status != NPP_SUCCESS) {
    return status;
  }

  if (nDivisor == 0) {
    return NPP_DIVISOR_ERROR;
  }

  return nppiFilter_8u_C3R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nDivisor,
                                    nppStreamCtx);
}

NppStatus nppiFilter_8u_C3R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                            const Npp32s *pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiFilter_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nDivisor,
                               nppStreamCtx);
}

// 32-bit float single channel 2D convolution
NppStatus nppiFilter_32f_C1R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                 const Npp32f *pKernel, NppiSize oKernelSize, NppiPoint oAnchor,
                                 NppStreamContext nppStreamCtx) {
  NppStatus status = validateFilterInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiFilter_32f_C1R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor,
                                     nppStreamCtx);
}

NppStatus nppiFilter_32f_C1R(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                             const Npp32f *pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiFilter_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}