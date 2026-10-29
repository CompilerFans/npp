// Public wrappers for the fused AbsDiff+Threshold_GTVal entry points (p2.3).
#include "npp.h"

extern "C" {
NppStatus nppiFusedAbsDiff_Threshold_GTVal_Ctx_impl(NppDataType, NppiChannels, const void*, int,
                                                    const void*, int, void*, int, NppiSize,
                                                    const void*, const void*, NppStreamContext);
NppStatus nppiFusedAbsDiff_Threshold_GTVal_I_Ctx_impl(NppDataType, NppiChannels, void*, int,
                                                      const void*, int, NppiSize, const void*,
                                                      const void*, NppStreamContext);
}

static inline bool fusedTypeOk(NppDataType t) {
  return t == NPP_8U || t == NPP_16U || t == NPP_16S || t == NPP_32F;
}
static inline bool fusedChOk(NppiChannels ch) {
  return ch == NPP_CH_1 || ch == NPP_CH_3 || ch == NPP_CH_A4;
}

NppStatus nppiFusedAbsDiff_Threshold_GTVal_Ctx(NppDataType eSrcDstType, NppiChannels eSrcDstChannels,
                                               const void* pSrc1, int nSrc1Step,
                                               const void* pSrc2, int nSrc2Step,
                                               void* pDst, int nDstStep, NppiSize oSizeROI,
                                               const void* pThreshold, const void* pvalue,
                                               NppStreamContext nppStreamCtx) {
  if (!pSrc1 || !pSrc2 || !pDst || !pThreshold || !pvalue)
    return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0)
    return NPP_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0)
    return NPP_NO_ERROR;
  if (!fusedTypeOk(eSrcDstType) || !fusedChOk(eSrcDstChannels))
    return NPP_NOT_SUPPORTED_MODE_ERROR;
  return nppiFusedAbsDiff_Threshold_GTVal_Ctx_impl(
      eSrcDstType, eSrcDstChannels, pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep,
      oSizeROI, pThreshold, pvalue, nppStreamCtx);
}

NppStatus nppiFusedAbsDiff_Threshold_GTVal_I_Ctx(NppDataType eSrcDstType, NppiChannels eSrcDstChannels,
                                                 void* pSrcDst, int nSrcDstStep,
                                                 const void* pSrc2, int nSrc2Step,
                                                 NppiSize oSizeROI,
                                                 const void* pThreshold, const void* pvalue,
                                                 NppStreamContext nppStreamCtx) {
  if (!pSrcDst || !pSrc2 || !pThreshold || !pvalue)
    return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0)
    return NPP_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0)
    return NPP_NO_ERROR;
  if (!fusedTypeOk(eSrcDstType) || !fusedChOk(eSrcDstChannels))
    return NPP_NOT_SUPPORTED_MODE_ERROR;
  return nppiFusedAbsDiff_Threshold_GTVal_I_Ctx_impl(
      eSrcDstType, eSrcDstChannels, pSrcDst, nSrcDstStep, pSrc2, nSrc2Step, oSizeROI,
      pThreshold, pvalue, nppStreamCtx);
}
