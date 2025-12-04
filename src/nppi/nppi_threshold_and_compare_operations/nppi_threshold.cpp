#include "npp.h"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

// Forward declarations for mpp host func implementations

extern "C" {
NppStatus nppiThreshold_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                        const Npp8u nThreshold, NppCmpOp eComparisonOperation,
                                        NppStreamContext nppStreamCtx);
NppStatus nppiThreshold_8u_C1IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp8u nThreshold,
                                         NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);
NppStatus nppiThreshold_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                         NppiSize oSizeROI, const Npp32f nThreshold, NppCmpOp eComparisonOperation,
                                         NppStreamContext nppStreamCtx);
NppStatus nppiThreshold_LTVal_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                               NppiSize oSizeROI, Npp8u nThreshold, Npp8u nValue,
                                               NppStreamContext nppStreamCtx);
NppStatus nppiThreshold_LTVal_8u_C1IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                                Npp8u nThreshold, Npp8u nValue, NppStreamContext nppStreamCtx);
NppStatus nppiThreshold_GTVal_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                               NppiSize oSizeROI, Npp8u nThreshold, Npp8u nValue,
                                               NppStreamContext nppStreamCtx);
NppStatus nppiThreshold_GTVal_8u_C1IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                                Npp8u nThreshold, Npp8u nValue, NppStreamContext nppStreamCtx);
NppStatus nppiThreshold_LTValGTVal_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                                    NppiSize oSizeROI, Npp8u nThresholdLT, Npp8u nValueLT,
                                                    Npp8u nThresholdGT, Npp8u nValueGT, NppStreamContext nppStreamCtx);
NppStatus nppiThreshold_LTVal_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                                NppiSize oSizeROI, Npp32f nThreshold, Npp32f nValue,
                                                NppStreamContext nppStreamCtx);
// C3 declarations
NppStatus nppiThreshold_LTVal_8u_C3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                               NppiSize oSizeROI, const Npp8u *pThresholds, const Npp8u *pValues,
                                               NppStreamContext nppStreamCtx);
NppStatus nppiThreshold_LTVal_8u_C3IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                                const Npp8u *pThresholds, const Npp8u *pValues,
                                                NppStreamContext nppStreamCtx);
NppStatus nppiThreshold_GTVal_8u_C3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                               NppiSize oSizeROI, const Npp8u *pThresholds, const Npp8u *pValues,
                                               NppStreamContext nppStreamCtx);
NppStatus nppiThreshold_GTVal_8u_C3IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                                const Npp8u *pThresholds, const Npp8u *pValues,
                                                NppStreamContext nppStreamCtx);
// C4 declarations
NppStatus nppiThreshold_LTVal_8u_C4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                               NppiSize oSizeROI, const Npp8u *pThresholds, const Npp8u *pValues,
                                               NppStreamContext nppStreamCtx);
NppStatus nppiThreshold_LTVal_8u_C4IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                                const Npp8u *pThresholds, const Npp8u *pValues,
                                                NppStreamContext nppStreamCtx);
NppStatus nppiThreshold_GTVal_8u_C4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                               NppiSize oSizeROI, const Npp8u *pThresholds, const Npp8u *pValues,
                                               NppStreamContext nppStreamCtx);
NppStatus nppiThreshold_GTVal_8u_C4IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                                const Npp8u *pThresholds, const Npp8u *pValues,
                                                NppStreamContext nppStreamCtx);
}

// Input validation helper
static inline NppStatus validateThresholdInputs(const void *pSrc, int nSrcStep, void *pDst, int nDstStep,
                                                NppiSize oSizeROI, NppCmpOp eComparisonOperation,
                                                bool inPlace = false) {
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  if (inPlace) {
    if (nSrcStep <= 0) {
      return NPP_STEP_ERROR;
    }
    if (!pSrc) {
      return NPP_NULL_POINTER_ERROR;
    }
  } else {
    if (nSrcStep <= 0 || nDstStep <= 0) {
      return NPP_STEP_ERROR;
    }
    if (!pSrc || !pDst) {
      return NPP_NULL_POINTER_ERROR;
    }
  }

  // Validate comparison operation - only LESS and GREATER are supported for threshold
  if (eComparisonOperation != NPP_CMP_LESS && eComparisonOperation != NPP_CMP_GREATER) {
    return NPP_NOT_SUPPORTED_MODE_ERROR;
  }

  return NPP_SUCCESS;
}

// 8-bit unsigned single channel threshold (non-inplace)
NppStatus nppiThreshold_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   const Npp8u nThreshold, NppCmpOp eComparisonOperation,
                                   NppStreamContext nppStreamCtx) {
  NppStatus status = validateThresholdInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eComparisonOperation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiThreshold_8u_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nThreshold, eComparisonOperation,
                                       nppStreamCtx);
}

NppStatus nppiThreshold_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                               const Npp8u nThreshold, NppCmpOp eComparisonOperation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiThreshold_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nThreshold, eComparisonOperation,
                                  nppStreamCtx);
}

// 8-bit unsigned single channel threshold (inplace)
NppStatus nppiThreshold_8u_C1IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp8u nThreshold,
                                    NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateThresholdInputs(pSrcDst, nSrcDstStep, nullptr, 0, oSizeROI, eComparisonOperation, true);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiThreshold_8u_C1IR_Ctx_impl(pSrcDst, nSrcDstStep, oSizeROI, nThreshold, eComparisonOperation, nppStreamCtx);
}

NppStatus nppiThreshold_8u_C1IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp8u nThreshold,
                                NppCmpOp eComparisonOperation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiThreshold_8u_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nThreshold, eComparisonOperation, nppStreamCtx);
}

// 32-bit float single channel threshold (non-inplace)
NppStatus nppiThreshold_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                    const Npp32f nThreshold, NppCmpOp eComparisonOperation,
                                    NppStreamContext nppStreamCtx) {
  NppStatus status = validateThresholdInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eComparisonOperation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiThreshold_32f_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nThreshold, eComparisonOperation,
                                        nppStreamCtx);
}

NppStatus nppiThreshold_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                const Npp32f nThreshold, NppCmpOp eComparisonOperation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiThreshold_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nThreshold, eComparisonOperation,
                                   nppStreamCtx);
}

// Threshold_LTVal validation helper
static inline NppStatus validateThresholdValInputs(const void *pSrc, int nSrcStep, void *pDst, int nDstStep,
                                                   NppiSize oSizeROI, bool inPlace = false) {
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  if (inPlace) {
    if (nSrcStep <= 0) {
      return NPP_STEP_ERROR;
    }
    if (!pSrc) {
      return NPP_NULL_POINTER_ERROR;
    }
  } else {
    if (nSrcStep <= 0 || nDstStep <= 0) {
      return NPP_STEP_ERROR;
    }
    if (!pSrc || !pDst) {
      return NPP_NULL_POINTER_ERROR;
    }
  }
  return NPP_SUCCESS;
}

// Threshold_LTVal 8u C1R
NppStatus nppiThreshold_LTVal_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                          NppiSize oSizeROI, Npp8u nThreshold, Npp8u nValue,
                                          NppStreamContext nppStreamCtx) {
  NppStatus status = validateThresholdValInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiThreshold_LTVal_8u_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nThreshold, nValue, nppStreamCtx);
}

NppStatus nppiThreshold_LTVal_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, Npp8u nThreshold, Npp8u nValue) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiThreshold_LTVal_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nThreshold, nValue, nppStreamCtx);
}

// Threshold_LTVal 8u C1IR (inplace)
NppStatus nppiThreshold_LTVal_8u_C1IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                           Npp8u nThreshold, Npp8u nValue, NppStreamContext nppStreamCtx) {
  NppStatus status = validateThresholdValInputs(pSrcDst, nSrcDstStep, nullptr, 0, oSizeROI, true);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiThreshold_LTVal_8u_C1IR_Ctx_impl(pSrcDst, nSrcDstStep, oSizeROI, nThreshold, nValue, nppStreamCtx);
}

NppStatus nppiThreshold_LTVal_8u_C1IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                       Npp8u nThreshold, Npp8u nValue) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiThreshold_LTVal_8u_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nThreshold, nValue, nppStreamCtx);
}

// Threshold_GTVal 8u C1R
NppStatus nppiThreshold_GTVal_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                          NppiSize oSizeROI, Npp8u nThreshold, Npp8u nValue,
                                          NppStreamContext nppStreamCtx) {
  NppStatus status = validateThresholdValInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiThreshold_GTVal_8u_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nThreshold, nValue, nppStreamCtx);
}

NppStatus nppiThreshold_GTVal_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, Npp8u nThreshold, Npp8u nValue) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiThreshold_GTVal_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nThreshold, nValue, nppStreamCtx);
}

// Threshold_GTVal 8u C1IR (inplace)
NppStatus nppiThreshold_GTVal_8u_C1IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                           Npp8u nThreshold, Npp8u nValue, NppStreamContext nppStreamCtx) {
  NppStatus status = validateThresholdValInputs(pSrcDst, nSrcDstStep, nullptr, 0, oSizeROI, true);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiThreshold_GTVal_8u_C1IR_Ctx_impl(pSrcDst, nSrcDstStep, oSizeROI, nThreshold, nValue, nppStreamCtx);
}

NppStatus nppiThreshold_GTVal_8u_C1IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                       Npp8u nThreshold, Npp8u nValue) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiThreshold_GTVal_8u_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, nThreshold, nValue, nppStreamCtx);
}

// Threshold_LTValGTVal 8u C1R
NppStatus nppiThreshold_LTValGTVal_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                               NppiSize oSizeROI, Npp8u nThresholdLT, Npp8u nValueLT,
                                               Npp8u nThresholdGT, Npp8u nValueGT, NppStreamContext nppStreamCtx) {
  NppStatus status = validateThresholdValInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiThreshold_LTValGTVal_8u_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                   nThresholdLT, nValueLT, nThresholdGT, nValueGT, nppStreamCtx);
}

NppStatus nppiThreshold_LTValGTVal_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                           NppiSize oSizeROI, Npp8u nThresholdLT, Npp8u nValueLT,
                                           Npp8u nThresholdGT, Npp8u nValueGT) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiThreshold_LTValGTVal_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                              nThresholdLT, nValueLT, nThresholdGT, nValueGT, nppStreamCtx);
}

// Threshold_LTVal 32f C1R
NppStatus nppiThreshold_LTVal_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                           NppiSize oSizeROI, Npp32f nThreshold, Npp32f nValue,
                                           NppStreamContext nppStreamCtx) {
  NppStatus status = validateThresholdValInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiThreshold_LTVal_32f_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nThreshold, nValue, nppStreamCtx);
}

NppStatus nppiThreshold_LTVal_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                       NppiSize oSizeROI, Npp32f nThreshold, Npp32f nValue) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiThreshold_LTVal_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nThreshold, nValue, nppStreamCtx);
}

// C3 Threshold_LTVal wrappers
NppStatus nppiThreshold_LTVal_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                          NppiSize oSizeROI, const Npp8u pThresholds[3], const Npp8u pValues[3],
                                          NppStreamContext nppStreamCtx) {
  NppStatus status = validateThresholdValInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) return status;
  return nppiThreshold_LTVal_8u_C3R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pThresholds, pValues, nppStreamCtx);
}

NppStatus nppiThreshold_LTVal_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, const Npp8u pThresholds[3], const Npp8u pValues[3]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiThreshold_LTVal_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pThresholds, pValues, nppStreamCtx);
}

NppStatus nppiThreshold_LTVal_8u_C3IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                           const Npp8u pThresholds[3], const Npp8u pValues[3],
                                           NppStreamContext nppStreamCtx) {
  NppStatus status = validateThresholdValInputs(pSrcDst, nSrcDstStep, nullptr, 0, oSizeROI, true);
  if (status != NPP_SUCCESS) return status;
  return nppiThreshold_LTVal_8u_C3IR_Ctx_impl(pSrcDst, nSrcDstStep, oSizeROI, pThresholds, pValues, nppStreamCtx);
}

NppStatus nppiThreshold_LTVal_8u_C3IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                       const Npp8u pThresholds[3], const Npp8u pValues[3]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiThreshold_LTVal_8u_C3IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, pThresholds, pValues, nppStreamCtx);
}

// C3 Threshold_GTVal wrappers
NppStatus nppiThreshold_GTVal_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                          NppiSize oSizeROI, const Npp8u pThresholds[3], const Npp8u pValues[3],
                                          NppStreamContext nppStreamCtx) {
  NppStatus status = validateThresholdValInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) return status;
  return nppiThreshold_GTVal_8u_C3R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pThresholds, pValues, nppStreamCtx);
}

NppStatus nppiThreshold_GTVal_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, const Npp8u pThresholds[3], const Npp8u pValues[3]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiThreshold_GTVal_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pThresholds, pValues, nppStreamCtx);
}

NppStatus nppiThreshold_GTVal_8u_C3IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                           const Npp8u pThresholds[3], const Npp8u pValues[3],
                                           NppStreamContext nppStreamCtx) {
  NppStatus status = validateThresholdValInputs(pSrcDst, nSrcDstStep, nullptr, 0, oSizeROI, true);
  if (status != NPP_SUCCESS) return status;
  return nppiThreshold_GTVal_8u_C3IR_Ctx_impl(pSrcDst, nSrcDstStep, oSizeROI, pThresholds, pValues, nppStreamCtx);
}

NppStatus nppiThreshold_GTVal_8u_C3IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                       const Npp8u pThresholds[3], const Npp8u pValues[3]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiThreshold_GTVal_8u_C3IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, pThresholds, pValues, nppStreamCtx);
}

// C4 Threshold_LTVal wrappers
NppStatus nppiThreshold_LTVal_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                          NppiSize oSizeROI, const Npp8u pThresholds[4], const Npp8u pValues[4],
                                          NppStreamContext nppStreamCtx) {
  NppStatus status = validateThresholdValInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) return status;
  return nppiThreshold_LTVal_8u_C4R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pThresholds, pValues, nppStreamCtx);
}

NppStatus nppiThreshold_LTVal_8u_C4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, const Npp8u pThresholds[4], const Npp8u pValues[4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiThreshold_LTVal_8u_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pThresholds, pValues, nppStreamCtx);
}

NppStatus nppiThreshold_LTVal_8u_C4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                           const Npp8u pThresholds[4], const Npp8u pValues[4],
                                           NppStreamContext nppStreamCtx) {
  NppStatus status = validateThresholdValInputs(pSrcDst, nSrcDstStep, nullptr, 0, oSizeROI, true);
  if (status != NPP_SUCCESS) return status;
  return nppiThreshold_LTVal_8u_C4IR_Ctx_impl(pSrcDst, nSrcDstStep, oSizeROI, pThresholds, pValues, nppStreamCtx);
}

NppStatus nppiThreshold_LTVal_8u_C4IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                       const Npp8u pThresholds[4], const Npp8u pValues[4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiThreshold_LTVal_8u_C4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, pThresholds, pValues, nppStreamCtx);
}

// C4 Threshold_GTVal wrappers
NppStatus nppiThreshold_GTVal_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                          NppiSize oSizeROI, const Npp8u pThresholds[4], const Npp8u pValues[4],
                                          NppStreamContext nppStreamCtx) {
  NppStatus status = validateThresholdValInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) return status;
  return nppiThreshold_GTVal_8u_C4R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pThresholds, pValues, nppStreamCtx);
}

NppStatus nppiThreshold_GTVal_8u_C4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, const Npp8u pThresholds[4], const Npp8u pValues[4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiThreshold_GTVal_8u_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pThresholds, pValues, nppStreamCtx);
}

NppStatus nppiThreshold_GTVal_8u_C4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                           const Npp8u pThresholds[4], const Npp8u pValues[4],
                                           NppStreamContext nppStreamCtx) {
  NppStatus status = validateThresholdValInputs(pSrcDst, nSrcDstStep, nullptr, 0, oSizeROI, true);
  if (status != NPP_SUCCESS) return status;
  return nppiThreshold_GTVal_8u_C4IR_Ctx_impl(pSrcDst, nSrcDstStep, oSizeROI, pThresholds, pValues, nppStreamCtx);
}

NppStatus nppiThreshold_GTVal_8u_C4IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                       const Npp8u pThresholds[4], const Npp8u pValues[4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiThreshold_GTVal_8u_C4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, pThresholds, pValues, nppStreamCtx);
}
