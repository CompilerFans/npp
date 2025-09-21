#include "npp.h"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

/**
 * NPP Image Morphological Operations Implementation
 * Implements nppiErode3x3 and nppiDilate3x3 functions
 */

// Forward declarations for mpp host func implementations
extern "C" {
// 3x3 implementations
NppStatus nppiErode3x3_8u_C1R_Ctx_cuda(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx);
NppStatus nppiErode3x3_32f_C1R_Ctx_cuda(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                        NppStreamContext nppStreamCtx);
NppStatus nppiDilate3x3_8u_C1R_Ctx_cuda(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                        NppStreamContext nppStreamCtx);
NppStatus nppiDilate3x3_32f_C1R_Ctx_cuda(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                         NppiSize oSizeROI, NppStreamContext nppStreamCtx);

// General morphology implementations with arbitrary kernel support
NppStatus nppiErode_8u_C1R_Ctx_cuda(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                    const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx);
NppStatus nppiErode_8u_C4R_Ctx_cuda(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                    const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx);
NppStatus nppiErode_32f_C1R_Ctx_cuda(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                     const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx);
NppStatus nppiErode_32f_C4R_Ctx_cuda(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                     const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx);

NppStatus nppiDilate_8u_C1R_Ctx_cuda(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                     const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx);
NppStatus nppiDilate_8u_C4R_Ctx_cuda(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                     const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx);
NppStatus nppiDilate_32f_C1R_Ctx_cuda(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                      const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx);
NppStatus nppiDilate_32f_C4R_Ctx_cuda(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                      const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx);
}

// Input validation helper for morphological operations
static inline NppStatus validateMorphologyInputs(const void *pSrc, int nSrcStep, void *pDst, int nDstStep,
                                                 NppiSize oSizeROI) {
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  return NPP_SUCCESS;
}

// Extended validation for general morphological operations with kernels
static inline NppStatus validateMorphologyInputsWithKernel(const void *pSrc, int nSrcStep, void *pDst, int nDstStep,
                                                           NppiSize oSizeROI, const Npp8u *pMask, NppiSize oMaskSize,
                                                           NppiPoint oAnchor) {
  // Basic validation
  NppStatus status = validateMorphologyInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  // Mask validation
  if (!pMask) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oMaskSize.width <= 0 || oMaskSize.height <= 0) {
    return NPP_MASK_SIZE_ERROR;
  }

  // Anchor validation
  if (oAnchor.x < 0 || oAnchor.x >= oMaskSize.width || oAnchor.y < 0 || oAnchor.y >= oMaskSize.height) {
    return NPP_ANCHOR_ERROR;
  }

  return NPP_SUCCESS;
}

//=============================================================================
// Erosion Functions
//=============================================================================

// 8-bit unsigned single channel 3x3 erosion
NppStatus nppiErode3x3_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
  NppStatus status = validateMorphologyInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiErode3x3_8u_C1R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiErode3x3_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiErode3x3_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// 32-bit float single channel 3x3 erosion
NppStatus nppiErode3x3_32f_C1R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                   NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateMorphologyInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiErode3x3_32f_C1R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiErode3x3_32f_C1R(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiErode3x3_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

//=============================================================================
// Dilation Functions
//=============================================================================

// 8-bit unsigned single channel 3x3 dilation
NppStatus nppiDilate3x3_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  NppStatus status = validateMorphologyInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiDilate3x3_8u_C1R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiDilate3x3_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiDilate3x3_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// 32-bit float single channel 3x3 dilation
NppStatus nppiDilate3x3_32f_C1R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateMorphologyInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiDilate3x3_32f_C1R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiDilate3x3_32f_C1R(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiDilate3x3_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

//=============================================================================
// General Erosion Functions
//=============================================================================

// 8-bit unsigned single channel general erosion
NppStatus nppiErode_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                               const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  NppStatus status = validateMorphologyInputsWithKernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiErode_8u_C1R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, nppStreamCtx);
}

NppStatus nppiErode_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                          const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiErode_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, nppStreamCtx);
}

// 8-bit unsigned 4-channel general erosion
NppStatus nppiErode_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                               const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  NppStatus status = validateMorphologyInputsWithKernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiErode_8u_C4R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, nppStreamCtx);
}

NppStatus nppiErode_8u_C4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                          const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiErode_8u_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, nppStreamCtx);
}

// 32-bit float single channel general erosion
NppStatus nppiErode_32f_C1R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  NppStatus status = validateMorphologyInputsWithKernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiErode_32f_C1R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, nppStreamCtx);
}

NppStatus nppiErode_32f_C1R(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                           const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiErode_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, nppStreamCtx);
}

// 32-bit float 4-channel general erosion
NppStatus nppiErode_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  NppStatus status = validateMorphologyInputsWithKernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiErode_32f_C4R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, nppStreamCtx);
}

NppStatus nppiErode_32f_C4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                           const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiErode_32f_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, nppStreamCtx);
}

//=============================================================================
// General Dilation Functions
//=============================================================================

// 8-bit unsigned single channel general dilation
NppStatus nppiDilate_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  NppStatus status = validateMorphologyInputsWithKernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiDilate_8u_C1R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, nppStreamCtx);
}

NppStatus nppiDilate_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                           const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiDilate_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, nppStreamCtx);
}

// 8-bit unsigned 4-channel general dilation
NppStatus nppiDilate_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  NppStatus status = validateMorphologyInputsWithKernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiDilate_8u_C4R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, nppStreamCtx);
}

NppStatus nppiDilate_8u_C4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                           const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiDilate_8u_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, nppStreamCtx);
}

// 32-bit float single channel general dilation
NppStatus nppiDilate_32f_C1R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                 const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  NppStatus status = validateMorphologyInputsWithKernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiDilate_32f_C1R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, nppStreamCtx);
}

NppStatus nppiDilate_32f_C1R(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                            const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiDilate_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, nppStreamCtx);
}

// 32-bit float 4-channel general dilation
NppStatus nppiDilate_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                 const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
  NppStatus status = validateMorphologyInputsWithKernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiDilate_32f_C4R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, nppStreamCtx);
}

NppStatus nppiDilate_32f_C4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                            const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiDilate_32f_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pMask, oMaskSize, oAnchor, nppStreamCtx);
}