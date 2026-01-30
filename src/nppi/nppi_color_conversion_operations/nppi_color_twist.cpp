#include "npp.h"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

// Forward declarations for mpp host func implementations

extern "C" {
NppStatus nppiColorTwist32f_8u_C3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                            NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                            NppStreamContext nppStreamCtx);
NppStatus nppiColorTwist32f_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                            NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                            NppStreamContext nppStreamCtx);
NppStatus nppiColorTwist32f_8u_C4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                            NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                            NppStreamContext nppStreamCtx);
NppStatus nppiColorTwist32f_8u_AC4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                             NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                             NppStreamContext nppStreamCtx);
NppStatus nppiColorTwist32f_8u_C2R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                            NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                            NppStreamContext nppStreamCtx);
NppStatus nppiColorTwist32f_8u_P3R_Ctx_impl(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *const pDst[3],
                                            int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                            NppStreamContext nppStreamCtx);
NppStatus nppiColorTwist32f_8u_C1IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                             const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiColorTwist32f_8u_C3IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                             const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiColorTwist32f_8u_C4IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                             const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiColorTwist32f_8u_C2IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                             const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiColorTwist32f_8u_AC4IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                              const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiColorTwist32f_8u_IP3R_Ctx_impl(Npp8u *const pSrcDst[3], int nSrcDstStep, NppiSize oSizeROI,
                                             const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiColorTwist32f_16u_C1R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
                                             NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                             NppStreamContext nppStreamCtx);
NppStatus nppiColorTwist32f_16u_C2R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
                                             NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                             NppStreamContext nppStreamCtx);
NppStatus nppiColorTwist32f_16u_C3R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
                                             NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                             NppStreamContext nppStreamCtx);
NppStatus nppiColorTwist32f_16u_AC4R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
                                              NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                              NppStreamContext nppStreamCtx);
NppStatus nppiColorTwist32f_16u_P3R_Ctx_impl(const Npp16u *const pSrc[3], int nSrcStep, Npp16u *const pDst[3],
                                             int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                             NppStreamContext nppStreamCtx);
NppStatus nppiColorTwist32f_16u_C1IR_Ctx_impl(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                              const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiColorTwist32f_16u_C2IR_Ctx_impl(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                              const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiColorTwist32f_16u_C3IR_Ctx_impl(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                              const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiColorTwist32f_16u_AC4IR_Ctx_impl(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                               const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiColorTwist32f_16u_IP3R_Ctx_impl(Npp16u *const pSrcDst[3], int nSrcDstStep, NppiSize oSizeROI,
                                              const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiColorTwist32f_16s_C1R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep,
                                             NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                             NppStreamContext nppStreamCtx);
NppStatus nppiColorTwist32f_16s_C2R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep,
                                             NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                             NppStreamContext nppStreamCtx);
NppStatus nppiColorTwist32f_16s_C3R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep,
                                             NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                             NppStreamContext nppStreamCtx);
NppStatus nppiColorTwist32f_16s_AC4R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep,
                                              NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                              NppStreamContext nppStreamCtx);
NppStatus nppiColorTwist32f_16s_P3R_Ctx_impl(const Npp16s *const pSrc[3], int nSrcStep, Npp16s *const pDst[3],
                                             int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                             NppStreamContext nppStreamCtx);
NppStatus nppiColorTwist32f_16s_C1IR_Ctx_impl(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                              const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiColorTwist32f_16s_C2IR_Ctx_impl(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                              const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiColorTwist32f_16s_C3IR_Ctx_impl(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                              const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiColorTwist32f_16s_AC4IR_Ctx_impl(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                               const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiColorTwist32f_16s_IP3R_Ctx_impl(Npp16s *const pSrcDst[3], int nSrcDstStep, NppiSize oSizeROI,
                                              const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
}

// Input validation helper
static inline NppStatus validateColorTwistInputs(const void *pSrc, int nSrcStep, void *pDst, int nDstStep,
                                                 NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (!aTwist) {
    return NPP_NULL_POINTER_ERROR;
  }

  return NPP_SUCCESS;
}

NppStatus nppiColorTwist32f_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                       const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorTwistInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiColorTwist32f_8u_C3R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorTwist32f_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                       const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorTwistInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiColorTwist32f_8u_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorTwist32f_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                       const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorTwistInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiColorTwist32f_8u_C4R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_C4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorTwist32f_8u_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                        NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                        NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorTwistInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiColorTwist32f_8u_AC4R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorTwist32f_8u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_C2R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                       const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorTwistInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiColorTwist32f_8u_C2R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_C2R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorTwist32f_8u_C2R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_C1IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                        const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorTwistInputs(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, aTwist);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiColorTwist32f_8u_C1IR_Ctx_impl(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_C1IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorTwist32f_8u_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_C3IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                        const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorTwistInputs(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, aTwist);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiColorTwist32f_8u_C3IR_Ctx_impl(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_C3IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorTwist32f_8u_C3IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_C4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                        const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorTwistInputs(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, aTwist);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiColorTwist32f_8u_C4IR_Ctx_impl(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_C4IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorTwist32f_8u_C4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_C2IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                        const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorTwistInputs(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, aTwist);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiColorTwist32f_8u_C2IR_Ctx_impl(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_C2IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorTwist32f_8u_C2IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_AC4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                         const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorTwistInputs(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, aTwist);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiColorTwist32f_8u_AC4IR_Ctx_impl(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_AC4IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                     const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorTwist32f_8u_AC4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_P3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *const pDst[3], int nDstStep,
                                       NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                       NppStreamContext nppStreamCtx) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pSrc[2] || !pDst || !pDst[0] || !pDst[1] || !pDst[2]) {
    return NPP_NULL_POINTER_ERROR;
  }

  NppStatus status = validateColorTwistInputs(pSrc[0], nSrcStep, pDst[0], nDstStep, oSizeROI, aTwist);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiColorTwist32f_8u_P3R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *const pDst[3], int nDstStep,
                                   NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorTwist32f_8u_P3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_IP3R_Ctx(Npp8u *const pSrcDst[3], int nSrcDstStep, NppiSize oSizeROI,
                                        const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  if (!pSrcDst || !pSrcDst[0] || !pSrcDst[1] || !pSrcDst[2]) {
    return NPP_NULL_POINTER_ERROR;
  }

  NppStatus status = validateColorTwistInputs(pSrcDst[0], nSrcDstStep, pSrcDst[0], nSrcDstStep, oSizeROI, aTwist);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiColorTwist32f_8u_IP3R_Ctx_impl(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_IP3R(Npp8u *const pSrcDst[3], int nSrcDstStep, NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorTwist32f_8u_IP3R_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
                                        NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                        NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorTwistInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiColorTwist32f_16u_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16u_C1R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorTwist32f_16u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16u_C2R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
                                        NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                        NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorTwistInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiColorTwist32f_16u_C2R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16u_C2R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorTwist32f_16u_C2R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
                                        NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                        NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorTwistInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiColorTwist32f_16u_C3R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16u_C3R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorTwist32f_16u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
                                         NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                         NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorTwistInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiColorTwist32f_16u_AC4R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16u_AC4R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
                                     NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorTwist32f_16u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16u_P3R_Ctx(const Npp16u *const pSrc[3], int nSrcStep, Npp16u *const pDst[3], int nDstStep,
                                        NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                        NppStreamContext nppStreamCtx) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pSrc[2] || !pDst || !pDst[0] || !pDst[1] || !pDst[2]) {
    return NPP_NULL_POINTER_ERROR;
  }
  NppStatus status = validateColorTwistInputs(pSrc[0], nSrcStep, pDst[0], nDstStep, oSizeROI, aTwist);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiColorTwist32f_16u_P3R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16u_P3R(const Npp16u *const pSrc[3], int nSrcStep, Npp16u *const pDst[3], int nDstStep,
                                    NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorTwist32f_16u_P3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16u_C1IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                         const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorTwistInputs(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, aTwist);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiColorTwist32f_16u_C1IR_Ctx_impl(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16u_C1IR(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                     const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorTwist32f_16u_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16u_C2IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                         const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorTwistInputs(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, aTwist);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiColorTwist32f_16u_C2IR_Ctx_impl(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16u_C2IR(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                     const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorTwist32f_16u_C2IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16u_C3IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                         const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorTwistInputs(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, aTwist);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiColorTwist32f_16u_C3IR_Ctx_impl(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16u_C3IR(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                     const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorTwist32f_16u_C3IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16u_AC4IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                          const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorTwistInputs(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, aTwist);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiColorTwist32f_16u_AC4IR_Ctx_impl(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16u_AC4IR(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorTwist32f_16u_AC4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16u_IP3R_Ctx(Npp16u *const pSrcDst[3], int nSrcDstStep, NppiSize oSizeROI,
                                         const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  if (!pSrcDst || !pSrcDst[0] || !pSrcDst[1] || !pSrcDst[2]) {
    return NPP_NULL_POINTER_ERROR;
  }
  NppStatus status = validateColorTwistInputs(pSrcDst[0], nSrcDstStep, pSrcDst[0], nSrcDstStep, oSizeROI, aTwist);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiColorTwist32f_16u_IP3R_Ctx_impl(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16u_IP3R(Npp16u *const pSrcDst[3], int nSrcDstStep, NppiSize oSizeROI,
                                     const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorTwist32f_16u_IP3R_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep,
                                        NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                        NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorTwistInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiColorTwist32f_16s_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16s_C1R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorTwist32f_16s_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16s_C2R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep,
                                        NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                        NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorTwistInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiColorTwist32f_16s_C2R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16s_C2R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorTwist32f_16s_C2R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16s_C3R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep,
                                        NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                        NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorTwistInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiColorTwist32f_16s_C3R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16s_C3R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorTwist32f_16s_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16s_AC4R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep,
                                         NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                         NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorTwistInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiColorTwist32f_16s_AC4R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16s_AC4R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep,
                                     NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorTwist32f_16s_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16s_P3R_Ctx(const Npp16s *const pSrc[3], int nSrcStep, Npp16s *const pDst[3], int nDstStep,
                                        NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                        NppStreamContext nppStreamCtx) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pSrc[2] || !pDst || !pDst[0] || !pDst[1] || !pDst[2]) {
    return NPP_NULL_POINTER_ERROR;
  }
  NppStatus status = validateColorTwistInputs(pSrc[0], nSrcStep, pDst[0], nDstStep, oSizeROI, aTwist);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiColorTwist32f_16s_P3R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16s_P3R(const Npp16s *const pSrc[3], int nSrcStep, Npp16s *const pDst[3], int nDstStep,
                                    NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorTwist32f_16s_P3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16s_C1IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                         const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorTwistInputs(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, aTwist);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiColorTwist32f_16s_C1IR_Ctx_impl(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16s_C1IR(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                     const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorTwist32f_16s_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16s_C2IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                         const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorTwistInputs(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, aTwist);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiColorTwist32f_16s_C2IR_Ctx_impl(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16s_C2IR(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                     const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorTwist32f_16s_C2IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16s_C3IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                         const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorTwistInputs(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, aTwist);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiColorTwist32f_16s_C3IR_Ctx_impl(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16s_C3IR(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                     const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorTwist32f_16s_C3IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16s_AC4IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                          const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorTwistInputs(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, aTwist);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiColorTwist32f_16s_AC4IR_Ctx_impl(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16s_AC4IR(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorTwist32f_16s_AC4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16s_IP3R_Ctx(Npp16s *const pSrcDst[3], int nSrcDstStep, NppiSize oSizeROI,
                                         const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  if (!pSrcDst || !pSrcDst[0] || !pSrcDst[1] || !pSrcDst[2]) {
    return NPP_NULL_POINTER_ERROR;
  }
  NppStatus status = validateColorTwistInputs(pSrcDst[0], nSrcDstStep, pSrcDst[0], nSrcDstStep, oSizeROI, aTwist);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiColorTwist32f_16s_IP3R_Ctx_impl(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16s_IP3R(Npp16s *const pSrcDst[3], int nSrcDstStep, NppiSize oSizeROI,
                                     const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorTwist32f_16s_IP3R_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}
