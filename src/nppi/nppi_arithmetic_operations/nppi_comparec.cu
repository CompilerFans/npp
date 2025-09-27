#include "nppi_arithmetic_executor.h"
#include "nppi_arithmetic_ops.h"

using namespace nppi::arithmetic;

// Implementation functions using the unified executor
extern "C" {

// 8u C1 comparison operations
NppStatus nppiCompareC_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u nConstant, Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI, NppCmpOp eComparisonOperation,
                                       NppStreamContext nppStreamCtx) {
  CompareConstOp<Npp8u> op(nConstant, eComparisonOperation);
  return CompareOperationExecutor<Npp8u, 1, CompareConstOp<Npp8u>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                                            nppStreamCtx.hStream, op);
}

// 16s C1 comparison operations
NppStatus nppiCompareC_16s_C1R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s nConstant, Npp8u *pDst, int nDstStep,
                                        NppiSize oSizeROI, NppCmpOp eComparisonOperation,
                                        NppStreamContext nppStreamCtx) {
  CompareConstOp<Npp16s> op(nConstant, eComparisonOperation);
  return CompareOperationExecutor<Npp16s, 1, CompareConstOp<Npp16s>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                                              nppStreamCtx.hStream, op);
}

// 32f C1 comparison operations
NppStatus nppiCompareC_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f nConstant, Npp8u *pDst, int nDstStep,
                                        NppiSize oSizeROI, NppCmpOp eComparisonOperation,
                                        NppStreamContext nppStreamCtx) {
  CompareConstOp<Npp32f> op(nConstant, eComparisonOperation);
  return CompareOperationExecutor<Npp32f, 1, CompareConstOp<Npp32f>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                                              nppStreamCtx.hStream, op);
}

} // extern "C"

// Input validation helper
static inline NppStatus validateCompareCInputs(const void *pSrc, int nSrcStep, void *pDst, int nDstStep,
                                               NppiSize oSizeROI, NppCmpOp eComparisonOperation) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  // Validate comparison operation
  if (eComparisonOperation < NPP_CMP_LESS || eComparisonOperation > NPP_CMP_GREATER) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  return NPP_SUCCESS;
}

// Public API functions
NppStatus nppiCompareC_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u nConstant, Npp8u *pDst, int nDstStep,
                                  NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateCompareCInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eComparisonOperation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiCompareC_8u_C1R_Ctx_impl(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, eComparisonOperation,
                                      nppStreamCtx);
}

NppStatus nppiCompareC_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u nConstant, Npp8u *pDst, int nDstStep,
                              NppiSize oSizeROI, NppCmpOp eComparisonOperation) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiCompareC_8u_C1R_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, eComparisonOperation,
                                 nppStreamCtx);
}

NppStatus nppiCompareC_16s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s nConstant, Npp8u *pDst, int nDstStep,
                                   NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateCompareCInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eComparisonOperation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiCompareC_16s_C1R_Ctx_impl(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, eComparisonOperation,
                                       nppStreamCtx);
}

NppStatus nppiCompareC_16s_C1R(const Npp16s *pSrc, int nSrcStep, Npp16s nConstant, Npp8u *pDst, int nDstStep,
                               NppiSize oSizeROI, NppCmpOp eComparisonOperation) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiCompareC_16s_C1R_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, eComparisonOperation,
                                  nppStreamCtx);
}

NppStatus nppiCompareC_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f nConstant, Npp8u *pDst, int nDstStep,
                                   NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateCompareCInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eComparisonOperation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiCompareC_32f_C1R_Ctx_impl(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, eComparisonOperation,
                                       nppStreamCtx);
}

NppStatus nppiCompareC_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f nConstant, Npp8u *pDst, int nDstStep,
                               NppiSize oSizeROI, NppCmpOp eComparisonOperation) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiCompareC_32f_C1R_Ctx(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, eComparisonOperation,
                                  nppStreamCtx);
}