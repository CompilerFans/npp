#include "npp.h"
#include <cstring>
#include <cuda_runtime.h>

extern "C" {
// 16-bit unsigned
NppStatus nppiMirror_16u_C1R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oROI,
                                      NppiAxis flip, cudaStream_t stream);
NppStatus nppiMirror_16u_C1IR_Ctx_impl(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                       cudaStream_t stream);
NppStatus nppiMirror_16u_C3R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oROI,
                                      NppiAxis flip, cudaStream_t stream);
NppStatus nppiMirror_16u_C3IR_Ctx_impl(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                       cudaStream_t stream);
NppStatus nppiMirror_16u_C4R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oROI,
                                      NppiAxis flip, cudaStream_t stream);
NppStatus nppiMirror_16u_C4IR_Ctx_impl(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                       cudaStream_t stream);
// 32-bit float
NppStatus nppiMirror_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oROI,
                                      NppiAxis flip, cudaStream_t stream);
NppStatus nppiMirror_32f_C1IR_Ctx_impl(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                       cudaStream_t stream);
NppStatus nppiMirror_32f_C3R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oROI,
                                      NppiAxis flip, cudaStream_t stream);
NppStatus nppiMirror_32f_C3IR_Ctx_impl(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                       cudaStream_t stream);
NppStatus nppiMirror_32f_C4R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oROI,
                                      NppiAxis flip, cudaStream_t stream);
NppStatus nppiMirror_32f_C4IR_Ctx_impl(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                       cudaStream_t stream);
// 32-bit signed integer
NppStatus nppiMirror_32s_C1R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oROI,
                                      NppiAxis flip, cudaStream_t stream);
NppStatus nppiMirror_32s_C1IR_Ctx_impl(Npp32s *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                       cudaStream_t stream);
NppStatus nppiMirror_32s_C3R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oROI,
                                      NppiAxis flip, cudaStream_t stream);
NppStatus nppiMirror_32s_C3IR_Ctx_impl(Npp32s *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                       cudaStream_t stream);
NppStatus nppiMirror_32s_C4R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oROI,
                                      NppiAxis flip, cudaStream_t stream);
NppStatus nppiMirror_32s_C4IR_Ctx_impl(Npp32s *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                       cudaStream_t stream);
// 8-bit unsigned
NppStatus nppiMirror_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oROI,
                                     NppiAxis flip, cudaStream_t stream);
NppStatus nppiMirror_8u_C1IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                      cudaStream_t stream);
NppStatus nppiMirror_8u_C3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oROI,
                                     NppiAxis flip, cudaStream_t stream);
NppStatus nppiMirror_8u_C3IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                      cudaStream_t stream);
NppStatus nppiMirror_8u_C4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oROI,
                                     NppiAxis flip, cudaStream_t stream);
NppStatus nppiMirror_8u_C4IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                      cudaStream_t stream);
}

template <typename T>
static inline NppStatus validateMirrorInputs(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, NppiSize oROI,
                                             NppiAxis flip, int channels) {
  if (!pSrc || !pDst)
    return NPP_NULL_POINTER_ERROR;
  if (oROI.width <= 0 || oROI.height <= 0)
    return NPP_SIZE_ERROR;

  int minStep = oROI.width * channels * static_cast<int>(sizeof(T));
  if (nSrcStep < minStep || nDstStep < minStep)
    return NPP_STEP_ERROR;

  if (flip != NPP_HORIZONTAL_AXIS && flip != NPP_VERTICAL_AXIS && flip != NPP_BOTH_AXIS)
    return NPP_MIRROR_FLIP_ERROR;

  return NPP_SUCCESS;
}

template <typename T>
static inline NppStatus validateMirrorInputsInPlace(T *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                                    int channels) {
  if (!pSrcDst)
    return NPP_NULL_POINTER_ERROR;
  if (oROI.width <= 0 || oROI.height <= 0)
    return NPP_SIZE_ERROR;

  int minStep = oROI.width * channels * static_cast<int>(sizeof(T));
  if (nSrcDstStep < minStep)
    return NPP_STEP_ERROR;
  if (flip != NPP_HORIZONTAL_AXIS && flip != NPP_VERTICAL_AXIS && flip != NPP_BOTH_AXIS)
    return NPP_MIRROR_FLIP_ERROR;

  return NPP_SUCCESS;
}

// ============ 16-bit unsigned ============

// nppiMirror_16u_C1R_Ctx
NppStatus nppiMirror_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oROI,
                                 NppiAxis flip, NppStreamContext nppStreamCtx) {
  NppStatus status = validateMirrorInputs(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, 1);
  if (status != NPP_SUCCESS)
    return status;

  return nppiMirror_16u_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, nppStreamCtx.hStream);
}

// nppiMirror_16u_C1R
NppStatus nppiMirror_16u_C1R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oROI,
                             NppiAxis flip) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiMirror_16u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, nppStreamCtx);
}

// nppiMirror_16u_C1IR_Ctx
NppStatus nppiMirror_16u_C1IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                  NppStreamContext nppStreamCtx) {
  NppStatus status = validateMirrorInputsInPlace(pSrcDst, nSrcDstStep, oROI, flip, 1);
  if (status != NPP_SUCCESS)
    return status;

  return nppiMirror_16u_C1IR_Ctx_impl(pSrcDst, nSrcDstStep, oROI, flip, nppStreamCtx.hStream);
}

// nppiMirror_16u_C1IR
NppStatus nppiMirror_16u_C1IR(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiMirror_16u_C1IR_Ctx(pSrcDst, nSrcDstStep, oROI, flip, nppStreamCtx);
}

// nppiMirror_16u_C3R_Ctx
NppStatus nppiMirror_16u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oROI,
                                 NppiAxis flip, NppStreamContext nppStreamCtx) {
  NppStatus status = validateMirrorInputs(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, 3);
  if (status != NPP_SUCCESS)
    return status;

  return nppiMirror_16u_C3R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, nppStreamCtx.hStream);
}

// nppiMirror_16u_C3R
NppStatus nppiMirror_16u_C3R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oROI,
                             NppiAxis flip) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiMirror_16u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, nppStreamCtx);
}

// nppiMirror_16u_C3IR_Ctx
NppStatus nppiMirror_16u_C3IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                  NppStreamContext nppStreamCtx) {
  NppStatus status = validateMirrorInputsInPlace(pSrcDst, nSrcDstStep, oROI, flip, 3);
  if (status != NPP_SUCCESS)
    return status;

  return nppiMirror_16u_C3IR_Ctx_impl(pSrcDst, nSrcDstStep, oROI, flip, nppStreamCtx.hStream);
}

// nppiMirror_16u_C3IR
NppStatus nppiMirror_16u_C3IR(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiMirror_16u_C3IR_Ctx(pSrcDst, nSrcDstStep, oROI, flip, nppStreamCtx);
}

// nppiMirror_16u_C4R_Ctx
NppStatus nppiMirror_16u_C4R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oROI,
                                 NppiAxis flip, NppStreamContext nppStreamCtx) {
  NppStatus status = validateMirrorInputs(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, 4);
  if (status != NPP_SUCCESS)
    return status;

  return nppiMirror_16u_C4R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, nppStreamCtx.hStream);
}

// nppiMirror_16u_C4R
NppStatus nppiMirror_16u_C4R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oROI,
                             NppiAxis flip) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiMirror_16u_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, nppStreamCtx);
}

// nppiMirror_16u_C4IR_Ctx
NppStatus nppiMirror_16u_C4IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                  NppStreamContext nppStreamCtx) {
  NppStatus status = validateMirrorInputsInPlace(pSrcDst, nSrcDstStep, oROI, flip, 4);
  if (status != NPP_SUCCESS)
    return status;

  return nppiMirror_16u_C4IR_Ctx_impl(pSrcDst, nSrcDstStep, oROI, flip, nppStreamCtx.hStream);
}

// nppiMirror_16u_C4IR
NppStatus nppiMirror_16u_C4IR(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiMirror_16u_C4IR_Ctx(pSrcDst, nSrcDstStep, oROI, flip, nppStreamCtx);
}

// ============ 32-bit float  ============

// nppiMirror_32f_C1R_Ctx
NppStatus nppiMirror_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oROI,
                                 NppiAxis flip, NppStreamContext nppStreamCtx) {
  NppStatus status = validateMirrorInputs(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, 1);
  if (status != NPP_SUCCESS)
    return status;

  return nppiMirror_32f_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, nppStreamCtx.hStream);
}

// nppiMirror_32f_C1R
NppStatus nppiMirror_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oROI,
                             NppiAxis flip) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiMirror_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, nppStreamCtx);
}

// nppiMirror_32f_C1IR_Ctx
NppStatus nppiMirror_32f_C1IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                  NppStreamContext nppStreamCtx) {
  NppStatus status = validateMirrorInputsInPlace(pSrcDst, nSrcDstStep, oROI, flip, 1);
  if (status != NPP_SUCCESS)
    return status;

  return nppiMirror_32f_C1IR_Ctx_impl(pSrcDst, nSrcDstStep, oROI, flip, nppStreamCtx.hStream);
}

// nppiMirror_32f_C1IR
NppStatus nppiMirror_32f_C1IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiMirror_32f_C1IR_Ctx(pSrcDst, nSrcDstStep, oROI, flip, nppStreamCtx);
}

// nppiMirror_32f_C3R_Ctx
NppStatus nppiMirror_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oROI,
                                 NppiAxis flip, NppStreamContext nppStreamCtx) {
  NppStatus status = validateMirrorInputs(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, 3);
  if (status != NPP_SUCCESS)
    return status;

  return nppiMirror_32f_C3R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, nppStreamCtx.hStream);
}

// nppiMirror_32f_C3R
NppStatus nppiMirror_32f_C3R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oROI,
                             NppiAxis flip) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiMirror_32f_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, nppStreamCtx);
}

// nppiMirror_32f_C3IR_Ctx
NppStatus nppiMirror_32f_C3IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                  NppStreamContext nppStreamCtx) {
  NppStatus status = validateMirrorInputsInPlace(pSrcDst, nSrcDstStep, oROI, flip, 3);
  if (status != NPP_SUCCESS)
    return status;

  return nppiMirror_32f_C3IR_Ctx_impl(pSrcDst, nSrcDstStep, oROI, flip, nppStreamCtx.hStream);
}

// nppiMirror_32f_C3IR
NppStatus nppiMirror_32f_C3IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiMirror_32f_C3IR_Ctx(pSrcDst, nSrcDstStep, oROI, flip, nppStreamCtx);
}

// nppiMirror_32f_C4R_Ctx
NppStatus nppiMirror_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oROI,
                                 NppiAxis flip, NppStreamContext nppStreamCtx) {
  NppStatus status = validateMirrorInputs(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, 4);
  if (status != NPP_SUCCESS)
    return status;

  return nppiMirror_32f_C4R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, nppStreamCtx.hStream);
}

// nppiMirror_32f_C4R
NppStatus nppiMirror_32f_C4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oROI,
                             NppiAxis flip) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiMirror_32f_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, nppStreamCtx);
}

// nppiMirror_32f_C4IR_Ctx
NppStatus nppiMirror_32f_C4IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                  NppStreamContext nppStreamCtx) {
  NppStatus status = validateMirrorInputsInPlace(pSrcDst, nSrcDstStep, oROI, flip, 4);
  if (status != NPP_SUCCESS)
    return status;

  return nppiMirror_32f_C4IR_Ctx_impl(pSrcDst, nSrcDstStep, oROI, flip, nppStreamCtx.hStream);
}

// nppiMirror_32f_C4IR
NppStatus nppiMirror_32f_C4IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiMirror_32f_C4IR_Ctx(pSrcDst, nSrcDstStep, oROI, flip, nppStreamCtx);
}

// ============ 32-bit signed integer ============

// nppiMirror_32s_C1R_Ctx
NppStatus nppiMirror_32s_C1R_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oROI,
                                 NppiAxis flip, NppStreamContext nppStreamCtx) {
  NppStatus status = validateMirrorInputs(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, 1);
  if (status != NPP_SUCCESS)
    return status;

  return nppiMirror_32s_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, nppStreamCtx.hStream);
}

// nppiMirror_32s_C1R
NppStatus nppiMirror_32s_C1R(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oROI,
                             NppiAxis flip) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiMirror_32s_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, nppStreamCtx);
}

// nppiMirror_32s_C1IR_Ctx
NppStatus nppiMirror_32s_C1IR_Ctx(Npp32s *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                  NppStreamContext nppStreamCtx) {
  NppStatus status = validateMirrorInputsInPlace(pSrcDst, nSrcDstStep, oROI, flip, 1);
  if (status != NPP_SUCCESS)
    return status;

  return nppiMirror_32s_C1IR_Ctx_impl(pSrcDst, nSrcDstStep, oROI, flip, nppStreamCtx.hStream);
}

// nppiMirror_32s_C1IR
NppStatus nppiMirror_32s_C1IR(Npp32s *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiMirror_32s_C1IR_Ctx(pSrcDst, nSrcDstStep, oROI, flip, nppStreamCtx);
}

// nppiMirror_32s_C3R_Ctx
NppStatus nppiMirror_32s_C3R_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oROI,
                                 NppiAxis flip, NppStreamContext nppStreamCtx) {
  NppStatus status = validateMirrorInputs(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, 3);
  if (status != NPP_SUCCESS)
    return status;

  return nppiMirror_32s_C3R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, nppStreamCtx.hStream);
}

// nppiMirror_32s_C3R
NppStatus nppiMirror_32s_C3R(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oROI,
                             NppiAxis flip) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiMirror_32s_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, nppStreamCtx);
}

// nppiMirror_32s_C3IR_Ctx
NppStatus nppiMirror_32s_C3IR_Ctx(Npp32s *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                  NppStreamContext nppStreamCtx) {
  NppStatus status = validateMirrorInputsInPlace(pSrcDst, nSrcDstStep, oROI, flip, 3);
  if (status != NPP_SUCCESS)
    return status;

  return nppiMirror_32s_C3IR_Ctx_impl(pSrcDst, nSrcDstStep, oROI, flip, nppStreamCtx.hStream);
}

// nppiMirror_32s_C3IR
NppStatus nppiMirror_32s_C3IR(Npp32s *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiMirror_32s_C3IR_Ctx(pSrcDst, nSrcDstStep, oROI, flip, nppStreamCtx);
}

// nppiMirror_32s_C4R_Ctx
NppStatus nppiMirror_32s_C4R_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oROI,
                                 NppiAxis flip, NppStreamContext nppStreamCtx) {
  NppStatus status = validateMirrorInputs(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, 4);
  if (status != NPP_SUCCESS)
    return status;

  return nppiMirror_32s_C4R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, nppStreamCtx.hStream);
}

// nppiMirror_32s_C4R
NppStatus nppiMirror_32s_C4R(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oROI,
                             NppiAxis flip) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiMirror_32s_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, nppStreamCtx);
}

// nppiMirror_32s_C4IR_Ctx
NppStatus nppiMirror_32s_C4IR_Ctx(Npp32s *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                  NppStreamContext nppStreamCtx) {
  NppStatus status = validateMirrorInputsInPlace(pSrcDst, nSrcDstStep, oROI, flip, 4);
  if (status != NPP_SUCCESS)
    return status;

  return nppiMirror_32s_C4IR_Ctx_impl(pSrcDst, nSrcDstStep, oROI, flip, nppStreamCtx.hStream);
}

// nppiMirror_32s_C4IR
NppStatus nppiMirror_32s_C4IR(Npp32s *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiMirror_32s_C4IR_Ctx(pSrcDst, nSrcDstStep, oROI, flip, nppStreamCtx);
}

// ============ 8-bit unsigned 补充接口实现 ============
NppStatus nppiMirror_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oROI,
                                NppiAxis flip, NppStreamContext nppStreamCtx) {
  // Parameter validation
  NppStatus status = validateMirrorInputs(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, 1);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiMirror_8u_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, nppStreamCtx.hStream);
}

NppStatus nppiMirror_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oROI, NppiAxis flip) {
  // Get default stream context
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiMirror_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, nppStreamCtx);
}

// nppiMirror_8u_C1IR_Ctx
NppStatus nppiMirror_8u_C1IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                 NppStreamContext nppStreamCtx) {
  NppStatus status = validateMirrorInputsInPlace(pSrcDst, nSrcDstStep, oROI, flip, 1);
  if (status != NPP_SUCCESS)
    return status;

  return nppiMirror_8u_C1IR_Ctx_impl(pSrcDst, nSrcDstStep, oROI, flip, nppStreamCtx.hStream);
}

// nppiMirror_8u_C1IR
NppStatus nppiMirror_8u_C1IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiMirror_8u_C1IR_Ctx(pSrcDst, nSrcDstStep, oROI, flip, nppStreamCtx);
}

// nppiMirror_8u_C3R_Ctx
NppStatus nppiMirror_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oROI,
                                NppiAxis flip, NppStreamContext nppStreamCtx) {
  NppStatus status = validateMirrorInputs(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, 3);
  if (status != NPP_SUCCESS)
    return status;

  return nppiMirror_8u_C3R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, nppStreamCtx.hStream);
}

// nppiMirror_8u_C3R
NppStatus nppiMirror_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oROI, NppiAxis flip) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiMirror_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, nppStreamCtx);
}

// nppiMirror_8u_C3IR_Ctx
NppStatus nppiMirror_8u_C3IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                 NppStreamContext nppStreamCtx) {
  NppStatus status = validateMirrorInputsInPlace(pSrcDst, nSrcDstStep, oROI, flip, 3);
  if (status != NPP_SUCCESS)
    return status;

  return nppiMirror_8u_C3IR_Ctx_impl(pSrcDst, nSrcDstStep, oROI, flip, nppStreamCtx.hStream);
}

// nppiMirror_8u_C3IR
NppStatus nppiMirror_8u_C3IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiMirror_8u_C3IR_Ctx(pSrcDst, nSrcDstStep, oROI, flip, nppStreamCtx);
}

// nppiMirror_8u_C4R_Ctx
NppStatus nppiMirror_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oROI,
                                NppiAxis flip, NppStreamContext nppStreamCtx) {
  NppStatus status = validateMirrorInputs(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, 4);
  if (status != NPP_SUCCESS)
    return status;

  return nppiMirror_8u_C4R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, nppStreamCtx.hStream);
}

// nppiMirror_8u_C4R
NppStatus nppiMirror_8u_C4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oROI, NppiAxis flip) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiMirror_8u_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, nppStreamCtx);
}

// nppiMirror_8u_C4IR_Ctx
NppStatus nppiMirror_8u_C4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                 NppStreamContext nppStreamCtx) {
  NppStatus status = validateMirrorInputsInPlace(pSrcDst, nSrcDstStep, oROI, flip, 4);
  if (status != NPP_SUCCESS)
    return status;

  return nppiMirror_8u_C4IR_Ctx_impl(pSrcDst, nSrcDstStep, oROI, flip, nppStreamCtx.hStream);
}

// nppiMirror_8u_C4IR
NppStatus nppiMirror_8u_C4IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiMirror_8u_C4IR_Ctx(pSrcDst, nSrcDstStep, oROI, flip, nppStreamCtx);
}