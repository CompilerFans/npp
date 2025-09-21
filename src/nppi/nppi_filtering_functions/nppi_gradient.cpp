#include "../../npp_internal.h"
#include "npp.h"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

// Implementation file

// Forward declarations for mpp host func implementations
extern "C" {
// 带X/Y分量输出的版本（FilterBorderControlNPP使用）
NppStatus nppiGradientVectorPrewittBorder_8u16s_C1R_Ctx_cuda_xy(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize,
                                                                NppiPoint oSrcOffset, Npp16s *pDstX, int nDstXStep,
                                                                Npp16s *pDstY, int nDstYStep, Npp16s *pDstMag,
                                                                int nDstMagStep, Npp32f *pDstAngle, int nDstAngleStep,
                                                                NppiSize oSizeROI, NppiMaskSize eMaskSize,
                                                                NppiNorm eNorm, NppiBorderType eBorderType,
                                                                NppStreamContext nppStreamCtx);

// 原始版本（带mag/dir输出）
NppStatus nppiGradientVectorPrewittBorder_8u16s_C1R_Ctx_cuda_magdir(const Npp8u *pSrc, int nSrcStep,
                                                                    NppiSize oSrcSizeROI, NppiPoint oSrcOffset,
                                                                    Npp16s *pDstMag, int nDstMagStep, Npp16s *pDstDir,
                                                                    int nDstDirStep, NppiSize oDstSizeROI,
                                                                    NppiMaskSize eMaskSize, NppiBorderType eBorderType,
                                                                    NppStreamContext nppStreamCtx);
NppStatus nppiGradientVectorPrewittBorder_8u32f_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                                             NppiPoint oSrcOffset, Npp32f *pDstMag, int nDstMagStep,
                                                             Npp32f *pDstDir, int nDstDirStep, NppiSize oDstSizeROI,
                                                             NppiMaskSize eMaskSize, NppiBorderType eBorderType,
                                                             NppStreamContext nppStreamCtx);
NppStatus nppiGradientVectorPrewittBorder_16s_C1R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                                           NppiPoint oSrcOffset, Npp16s *pDstMag, int nDstMagStep,
                                                           Npp16s *pDstDir, int nDstDirStep, NppiSize oDstSizeROI,
                                                           NppiMaskSize eMaskSize, NppiBorderType eBorderType,
                                                           NppStreamContext nppStreamCtx);
}

// Input validation helper
static inline NppStatus validateGradientVectorInputs(const void *pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                                     void *pDstMag, int nDstMagStep, void *pDstDir, int nDstDirStep,
                                                     NppiSize oDstSizeROI, NppiMaskSize eMaskSize,
                                                     NppiBorderType eBorderType) {
  if (!pSrc || !pDstMag || !pDstDir) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSrcSizeROI.width <= 0 || oSrcSizeROI.height <= 0 || oDstSizeROI.width <= 0 || oDstSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstMagStep <= 0 || nDstDirStep <= 0) {
    return NPP_STEP_ERROR;
  }

  // Validate mask size
  if (eMaskSize != NPP_MASK_SIZE_3_X_3 && eMaskSize != NPP_MASK_SIZE_5_X_5) {
    return NPP_MASK_SIZE_ERROR;
  }

  // Validate border type
  if (eBorderType != NPP_BORDER_REPLICATE && eBorderType != NPP_BORDER_WRAP && eBorderType != NPP_BORDER_MIRROR &&
      eBorderType != NPP_BORDER_CONSTANT) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  return NPP_SUCCESS;
}

// 新版本 - 带X/Y分量输出（用于FilterBorderControlNPP）
NppStatus nppiGradientVectorPrewittBorder_8u16s_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize,
                                                        NppiPoint oSrcOffset, Npp16s *pDstX, int nDstXStep,
                                                        Npp16s *pDstY, int nDstYStep, Npp16s *pDstMag, int nDstMagStep,
                                                        Npp32f *pDstAngle, int nDstAngleStep, NppiSize oSizeROI,
                                                        NppiMaskSize eMaskSize, NppiNorm eNorm,
                                                        NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {

  // Parameter validation
  if (pSrc == nullptr || pDstX == nullptr || pDstY == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSrcSize.width <= 0 || oSrcSize.height <= 0 || oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstXStep <= 0 || nDstYStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (pDstMag != nullptr && nDstMagStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (pDstAngle != nullptr && nDstAngleStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (eMaskSize != NPP_MASK_SIZE_3_X_3 && eMaskSize != NPP_MASK_SIZE_5_X_5) {
    return NPP_MASK_SIZE_ERROR;
  }

  // CallGPU实现
  return nppiGradientVectorPrewittBorder_8u16s_C1R_Ctx_cuda_xy(
      pSrc, nSrcStep, oSrcSize, oSrcOffset, pDstX, nDstXStep, pDstY, nDstYStep, pDstMag, nDstMagStep, pDstAngle,
      nDstAngleStep, oSizeROI, eMaskSize, eNorm, eBorderType, nppStreamCtx);
}

// 不带上下文的版本
NppStatus nppiGradientVectorPrewittBorder_8u16s_C1R(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize,
                                                    NppiPoint oSrcOffset, Npp16s *pDstX, int nDstXStep, Npp16s *pDstY,
                                                    int nDstYStep, Npp16s *pDstMag, int nDstMagStep, Npp32f *pDstAngle,
                                                    int nDstAngleStep, NppiSize oSizeROI, NppiMaskSize eMaskSize,
                                                    NppiNorm eNorm, NppiBorderType eBorderType) {

  NppStreamContext nppStreamCtx = nppCreateDefaultStreamContext();
  nppStreamCtx.hStream = 0;

  return nppiGradientVectorPrewittBorder_8u16s_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDstX, nDstXStep, pDstY,
                                                       nDstYStep, pDstMag, nDstMagStep, pDstAngle, nDstAngleStep,
                                                       oSizeROI, eMaskSize, eNorm, eBorderType, nppStreamCtx);
}

// 原始版本 - 8-bit unsigned to 16-bit signed gradient vector with Prewitt operator (mag/dir输出)
NppStatus nppiGradientVectorPrewittBorder_8u16s_C1R_Ctx_magdir(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                                               NppiPoint oSrcOffset, Npp16s *pDstMag, int nDstMagStep,
                                                               Npp16s *pDstDir, int nDstDirStep, NppiSize oDstSizeROI,
                                                               NppiMaskSize eMaskSize, NppiBorderType eBorderType,
                                                               NppStreamContext nppStreamCtx) {
  NppStatus status = validateGradientVectorInputs(pSrc, nSrcStep, oSrcSizeROI, pDstMag, nDstMagStep, pDstDir,
                                                  nDstDirStep, oDstSizeROI, eMaskSize, eBorderType);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiGradientVectorPrewittBorder_8u16s_C1R_Ctx_cuda_magdir(pSrc, nSrcStep, oSrcSizeROI, oSrcOffset, pDstMag,
                                                                   nDstMagStep, pDstDir, nDstDirStep, oDstSizeROI,
                                                                   eMaskSize, eBorderType, nppStreamCtx);
}

NppStatus nppiGradientVectorPrewittBorder_8u16s_C1R_magdir(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                                           NppiPoint oSrcOffset, Npp16s *pDstMag, int nDstMagStep,
                                                           Npp16s *pDstDir, int nDstDirStep, NppiSize oDstSizeROI,
                                                           NppiMaskSize eMaskSize, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiGradientVectorPrewittBorder_8u16s_C1R_Ctx_magdir(pSrc, nSrcStep, oSrcSizeROI, oSrcOffset, pDstMag,
                                                              nDstMagStep, pDstDir, nDstDirStep, oDstSizeROI, eMaskSize,
                                                              eBorderType, nppStreamCtx);
}

// 8-bit unsigned to 32-bit float gradient vector with Prewitt operator
NppStatus nppiGradientVectorPrewittBorder_8u32f_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                                        NppiPoint oSrcOffset, Npp32f *pDstMag, int nDstMagStep,
                                                        Npp32f *pDstDir, int nDstDirStep, NppiSize oDstSizeROI,
                                                        NppiMaskSize eMaskSize, NppiBorderType eBorderType,
                                                        NppStreamContext nppStreamCtx) {
  NppStatus status = validateGradientVectorInputs(pSrc, nSrcStep, oSrcSizeROI, pDstMag, nDstMagStep, pDstDir,
                                                  nDstDirStep, oDstSizeROI, eMaskSize, eBorderType);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiGradientVectorPrewittBorder_8u32f_C1R_Ctx_impl(pSrc, nSrcStep, oSrcSizeROI, oSrcOffset, pDstMag,
                                                            nDstMagStep, pDstDir, nDstDirStep, oDstSizeROI, eMaskSize,
                                                            eBorderType, nppStreamCtx);
}

NppStatus nppiGradientVectorPrewittBorder_8u32f_C1R(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                                    NppiPoint oSrcOffset, Npp32f *pDstMag, int nDstMagStep,
                                                    Npp32f *pDstDir, int nDstDirStep, NppiSize oDstSizeROI,
                                                    NppiMaskSize eMaskSize, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiGradientVectorPrewittBorder_8u32f_C1R_Ctx(pSrc, nSrcStep, oSrcSizeROI, oSrcOffset, pDstMag, nDstMagStep,
                                                       pDstDir, nDstDirStep, oDstSizeROI, eMaskSize, eBorderType,
                                                       nppStreamCtx);
}

// 16-bit signed gradient vector with Prewitt operator
NppStatus nppiGradientVectorPrewittBorder_16s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                                      NppiPoint oSrcOffset, Npp16s *pDstMag, int nDstMagStep,
                                                      Npp16s *pDstDir, int nDstDirStep, NppiSize oDstSizeROI,
                                                      NppiMaskSize eMaskSize, NppiBorderType eBorderType,
                                                      NppStreamContext nppStreamCtx) {
  NppStatus status = validateGradientVectorInputs(pSrc, nSrcStep, oSrcSizeROI, pDstMag, nDstMagStep, pDstDir,
                                                  nDstDirStep, oDstSizeROI, eMaskSize, eBorderType);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiGradientVectorPrewittBorder_16s_C1R_Ctx_impl(pSrc, nSrcStep, oSrcSizeROI, oSrcOffset, pDstMag, nDstMagStep,
                                                          pDstDir, nDstDirStep, oDstSizeROI, eMaskSize, eBorderType,
                                                          nppStreamCtx);
}

NppStatus nppiGradientVectorPrewittBorder_16s_C1R(const Npp16s *pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                                  NppiPoint oSrcOffset, Npp16s *pDstMag, int nDstMagStep,
                                                  Npp16s *pDstDir, int nDstDirStep, NppiSize oDstSizeROI,
                                                  NppiMaskSize eMaskSize, NppiBorderType eBorderType) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiGradientVectorPrewittBorder_16s_C1R_Ctx(pSrc, nSrcStep, oSrcSizeROI, oSrcOffset, pDstMag, nDstMagStep,
                                                     pDstDir, nDstDirStep, oDstSizeROI, eMaskSize, eBorderType,
                                                     nppStreamCtx);
}