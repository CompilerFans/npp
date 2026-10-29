#include "npp.h"
#include <cstring>
#include <cuda_runtime.h>

extern "C" {
NppStatus nppiCompare_8u_C1R_Ctx_impl(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                      int nDstStep, NppiSize oSizeROI, NppCmpOp eCompOp, NppStreamContext nppStreamCtx);
NppStatus nppiCompare_16u_C1R_Ctx_impl(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step,
                                       Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppCmpOp eCompOp,
                                       NppStreamContext nppStreamCtx);
NppStatus nppiCompare_32f_C1R_Ctx_impl(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step,
                                       Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppCmpOp eCompOp,
                                       NppStreamContext nppStreamCtx);
// C3 impl declarations
NppStatus nppiCompare_8u_C3R_Ctx_impl(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                      int nDstStep, NppiSize oSizeROI, NppCmpOp eCompOp, NppStreamContext nppStreamCtx);
NppStatus nppiCompare_16u_C3R_Ctx_impl(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step,
                                       Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppCmpOp eCompOp,
                                       NppStreamContext nppStreamCtx);
NppStatus nppiCompare_32f_C3R_Ctx_impl(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step,
                                       Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppCmpOp eCompOp,
                                       NppStreamContext nppStreamCtx);
// C4 impl declarations
NppStatus nppiCompare_8u_C4R_Ctx_impl(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                      int nDstStep, NppiSize oSizeROI, NppCmpOp eCompOp, NppStreamContext nppStreamCtx);
NppStatus nppiCompare_16u_C4R_Ctx_impl(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step,
                                       Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppCmpOp eCompOp,
                                       NppStreamContext nppStreamCtx);
NppStatus nppiCompare_32f_C4R_Ctx_impl(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step,
                                       Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppCmpOp eCompOp,
                                       NppStreamContext nppStreamCtx);
}

static inline NppStatus validateCompareInputs(const void *pSrc1, int nSrc1Step, const void *pSrc2, int nSrc2Step,
                                              void *pDst, int nDstStep, NppiSize oSizeROI, NppCmpOp eCompOp) {
  // NPP does not validate eCompOp for compare: an illegal value short-circuits
  // to NPP_NO_ERROR before any pointer / ROI checks (confirmed on NPP 13.4).
  if (!(eCompOp == NPP_CMP_LESS || eCompOp == NPP_CMP_LESS_EQ || eCompOp == NPP_CMP_EQ ||
        eCompOp == NPP_CMP_GREATER_EQ || eCompOp == NPP_CMP_GREATER)) {
    return NPP_NO_ERROR;
  }
  if (!pSrc1 || !pSrc2 || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nSrc1Step <= 0 || nSrc2Step <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  return NPP_SUCCESS;
}

NppStatus nppiCompare_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI, NppCmpOp eCompOp, NppStreamContext nppStreamCtx) {
  NppStatus status = validateCompareInputs(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, eCompOp);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiCompare_8u_C1R_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, eCompOp,
                                     nppStreamCtx);
}

NppStatus nppiCompare_8u_C1R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI, NppCmpOp eCompOp) {
  NppStreamContext nppStreamCtx{};
  nppStreamCtx.hStream = nppGetStream();
  return nppiCompare_8u_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, eCompOp, nppStreamCtx);
}

NppStatus nppiCompare_16u_C1R_Ctx(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI, NppCmpOp eCompOp, NppStreamContext nppStreamCtx) {
  NppStatus status = validateCompareInputs(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, eCompOp);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiCompare_16u_C1R_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, eCompOp,
                                      nppStreamCtx);
}

NppStatus nppiCompare_16u_C1R(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI, NppCmpOp eCompOp) {
  NppStreamContext nppStreamCtx{};
  nppStreamCtx.hStream = nppGetStream();
  return nppiCompare_16u_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, eCompOp, nppStreamCtx);
}

NppStatus nppiCompare_32f_C1R_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI, NppCmpOp eCompOp, NppStreamContext nppStreamCtx) {
  NppStatus status = validateCompareInputs(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, eCompOp);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiCompare_32f_C1R_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, eCompOp,
                                      nppStreamCtx);
}

NppStatus nppiCompare_32f_C1R(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI, NppCmpOp eCompOp) {
  NppStreamContext nppStreamCtx{};
  nppStreamCtx.hStream = nppGetStream();
  return nppiCompare_32f_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, eCompOp, nppStreamCtx);
}

// C3 wrappers
NppStatus nppiCompare_8u_C3R_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI, NppCmpOp eCompOp, NppStreamContext nppStreamCtx) {
  NppStatus status = validateCompareInputs(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, eCompOp);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiCompare_8u_C3R_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, eCompOp,
                                     nppStreamCtx);
}

NppStatus nppiCompare_8u_C3R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI, NppCmpOp eCompOp) {
  NppStreamContext nppStreamCtx{};
  nppStreamCtx.hStream = nppGetStream();
  return nppiCompare_8u_C3R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, eCompOp, nppStreamCtx);
}

NppStatus nppiCompare_16u_C3R_Ctx(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI, NppCmpOp eCompOp, NppStreamContext nppStreamCtx) {
  NppStatus status = validateCompareInputs(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, eCompOp);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiCompare_16u_C3R_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, eCompOp,
                                      nppStreamCtx);
}

NppStatus nppiCompare_16u_C3R(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI, NppCmpOp eCompOp) {
  NppStreamContext nppStreamCtx{};
  nppStreamCtx.hStream = nppGetStream();
  return nppiCompare_16u_C3R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, eCompOp, nppStreamCtx);
}

NppStatus nppiCompare_32f_C3R_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI, NppCmpOp eCompOp, NppStreamContext nppStreamCtx) {
  NppStatus status = validateCompareInputs(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, eCompOp);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiCompare_32f_C3R_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, eCompOp,
                                      nppStreamCtx);
}

NppStatus nppiCompare_32f_C3R(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI, NppCmpOp eCompOp) {
  NppStreamContext nppStreamCtx{};
  nppStreamCtx.hStream = nppGetStream();
  return nppiCompare_32f_C3R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, eCompOp, nppStreamCtx);
}

// C4 wrappers
NppStatus nppiCompare_8u_C4R_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI, NppCmpOp eCompOp, NppStreamContext nppStreamCtx) {
  NppStatus status = validateCompareInputs(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, eCompOp);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiCompare_8u_C4R_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, eCompOp,
                                     nppStreamCtx);
}

NppStatus nppiCompare_8u_C4R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI, NppCmpOp eCompOp) {
  NppStreamContext nppStreamCtx{};
  nppStreamCtx.hStream = nppGetStream();
  return nppiCompare_8u_C4R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, eCompOp, nppStreamCtx);
}

NppStatus nppiCompare_16u_C4R_Ctx(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI, NppCmpOp eCompOp, NppStreamContext nppStreamCtx) {
  NppStatus status = validateCompareInputs(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, eCompOp);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiCompare_16u_C4R_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, eCompOp,
                                      nppStreamCtx);
}

NppStatus nppiCompare_16u_C4R(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI, NppCmpOp eCompOp) {
  NppStreamContext nppStreamCtx{};
  nppStreamCtx.hStream = nppGetStream();
  return nppiCompare_16u_C4R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, eCompOp, nppStreamCtx);
}

NppStatus nppiCompare_32f_C4R_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI, NppCmpOp eCompOp, NppStreamContext nppStreamCtx) {
  NppStatus status = validateCompareInputs(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, eCompOp);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiCompare_32f_C4R_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, eCompOp,
                                      nppStreamCtx);
}

NppStatus nppiCompare_32f_C4R(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI, NppCmpOp eCompOp) {
  NppStreamContext nppStreamCtx{};
  nppStreamCtx.hStream = nppGetStream();
  return nppiCompare_32f_C4R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, eCompOp, nppStreamCtx);
}
