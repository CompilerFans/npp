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

// Validation semantics measured against NVIDIA NPP 13.4 (reference):
//   * negative ROI dimensions -> NPP_SIZE_ERROR;
//   * step smaller than the row length -> NPP_STEP_ERROR;
//   * empty ROI (a zero dimension) -> NPP_NO_ERROR (verified with {0, 5});
//   * step not a multiple of the largest power of two dividing the pixel size
//     (8u C3 -> 1, 32f C3 -> 4, 16s C4 -> 8, 32f C4/AC4 -> 16) ->
//     NPP_NOT_EVEN_STEP_ERROR, checked before the null-pointer check;
//   * in-place variants additionally require even width and height, NVIDIA
//     NPP 13.4 returns NPP_SIZE_ERROR otherwise (see nppi_mirror_executor.h).
static inline bool nppMirrorStepAligned(int pixelBytes, int step) {
  const int align = pixelBytes & -pixelBytes;  // largest power-of-two divisor
  return (step % align) == 0;
}

template <typename T>
static inline NppStatus validateMirrorInputs(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, NppiSize oROI,
                                             NppiAxis flip, int channels) {
  if (oROI.width < 0 || oROI.height < 0)
    return NPP_SIZE_ERROR;

  int minStep = oROI.width * channels * static_cast<int>(sizeof(T));
  if (nSrcStep < minStep || nDstStep < minStep)
    return NPP_STEP_ERROR;

  if (oROI.width == 0 || oROI.height == 0)
    return NPP_NO_ERROR;

  if (!nppMirrorStepAligned(channels * static_cast<int>(sizeof(T)), nSrcStep) ||
      !nppMirrorStepAligned(channels * static_cast<int>(sizeof(T)), nDstStep))
    return NPP_NOT_EVEN_STEP_ERROR;

  if (!pSrc || !pDst)
    return NPP_NULL_POINTER_ERROR;

  if (flip != NPP_HORIZONTAL_AXIS && flip != NPP_VERTICAL_AXIS && flip != NPP_BOTH_AXIS)
    return NPP_MIRROR_FLIP_ERROR;

  return NPP_SUCCESS;
}

template <typename T>
static inline NppStatus validateMirrorInputsInPlace(T *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                                    int channels) {
  if (oROI.width < 0 || oROI.height < 0)
    return NPP_SIZE_ERROR;

  int minStep = oROI.width * channels * static_cast<int>(sizeof(T));
  if (nSrcDstStep < minStep)
    return NPP_STEP_ERROR;

  if (oROI.width == 0 || oROI.height == 0)
    return NPP_NO_ERROR;

  if (!nppMirrorStepAligned(channels * static_cast<int>(sizeof(T)), nSrcDstStep))
    return NPP_NOT_EVEN_STEP_ERROR;

  if (!pSrcDst)
    return NPP_NULL_POINTER_ERROR;

  // NVIDIA NPP 13.4 only accepts even width and height for in-place mirror;
  // the parity check fires before the flip check (measured in the reference implementation).
  if (oROI.width % 2 != 0 || oROI.height % 2 != 0)
    return NPP_SIZE_ERROR;

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