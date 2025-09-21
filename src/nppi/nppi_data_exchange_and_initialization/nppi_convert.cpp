#include "npp.h"
#include <cstring>
#include <cuda_runtime.h>

/**
 * NPP Image Convert Functions Implementation
 * Implements nppiConvert functions for data type conversion
 */

// Forward declarations for CUDA implementations
extern "C" {
NppStatus nppiConvert_8u32f_C1R_Ctx_cuda(const Npp8u *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
NppStatus nppiConvert_8u32f_C3R_Ctx_cuda(const Npp8u *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
}

/**
 * Helper function for parameter validation
 */
static inline NppStatus validateConvertInputs(const void *pSrc, int nSrcStep, void *pDst, int nDstStep,
                                              NppiSize oSizeROI) {
  if (!pSrc || !pDst)
    return NPP_NULL_POINTER_ERROR;
  if (nSrcStep <= 0 || nDstStep <= 0)
    return NPP_STEP_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0)
    return NPP_SIZE_ERROR;
  return NPP_SUCCESS;
}

/**
 * 8-bit unsigned to 32-bit float, single channel convert
 */
NppStatus nppiConvert_8u32f_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  // Parameter validation
  NppStatus status = validateConvertInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiConvert_8u32f_C1R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

/**
 * 8-bit unsigned to 32-bit float, single channel convert (no stream context)
 */
NppStatus nppiConvert_8u32f_C1R(const Npp8u *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI) {
  // Get default stream context
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  return nppiConvert_8u32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

/**
 * 8-bit unsigned to 32-bit float, three channel convert
 */
NppStatus nppiConvert_8u32f_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return nppiConvert_8u32f_C3R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

/**
 * 8-bit unsigned to 32-bit float, three channel convert (no stream context)
 */
NppStatus nppiConvert_8u32f_C3R(const Npp8u *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI) {
  // Get default stream context
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  return nppiConvert_8u32f_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}