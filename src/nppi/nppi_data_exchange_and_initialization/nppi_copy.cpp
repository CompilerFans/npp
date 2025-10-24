#include "npp.h"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

// Forward declarations for implementation functions
extern "C" {
NppStatus nppiCopy_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
NppStatus nppiCopy_8u_C3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
NppStatus nppiCopy_8u_C4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
NppStatus nppiCopy_16u_C1R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
NppStatus nppiCopy_16u_C3R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
NppStatus nppiCopy_16u_C4R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
NppStatus nppiCopy_16s_C1R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
NppStatus nppiCopy_16s_C3R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
NppStatus nppiCopy_16s_C4R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
NppStatus nppiCopy_32s_C1R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
NppStatus nppiCopy_32s_C3R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
NppStatus nppiCopy_32s_C4R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
NppStatus nppiCopy_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
NppStatus nppiCopy_32f_C3R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
NppStatus nppiCopy_32f_C4R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
// C3P3R
NppStatus nppiCopy_8u_C3P3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *const pDst[3], int nDstStep,
                                     NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiCopy_16u_C3P3R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *const pDst[3], int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiCopy_16s_C3P3R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *const pDst[3], int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiCopy_32s_C3P3R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, Npp32s *const pDst[3], int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiCopy_32f_C3P3R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *const pDst[3], int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx);

// C4P4R
NppStatus nppiCopy_8u_C4P4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *const pDst[4], int nDstStep,
                                     NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiCopy_16u_C4P4R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *const pDst[4], int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiCopy_32f_C4P4R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *const pDst[4], int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx);

// P3C3R
NppStatus nppiCopy_8u_P3C3R_Ctx_impl(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiCopy_16u_P3C3R_Ctx_impl(const Npp16u *const pSrc[3], int nSrcStep, Npp16u *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiCopy_16s_P3C3R_Ctx_impl(const Npp16s *const pSrc[3], int nSrcStep, Npp16s *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiCopy_32s_P3C3R_Ctx_impl(const Npp32s *const pSrc[3], int nSrcStep, Npp32s *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiCopy_32f_P3C3R_Ctx_impl(const Npp32f *const pSrc[3], int nSrcStep, Npp32f *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx);

// P4C4R
NppStatus nppiCopy_8u_P4C4R_Ctx_impl(const Npp8u *const pSrc[4], int nSrcStep, Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiCopy_16u_P4C4R_Ctx_impl(const Npp16u *const pSrc[4], int nSrcStep, Npp16u *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiCopy_32f_P4C4R_Ctx_impl(const Npp32f *const pSrc[4], int nSrcStep, Npp32f *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx);
}

// Input validation helper
static inline NppStatus validateCopyInputs(const void *pSrc, int nSrcStep, void *pDst, int nDstStep,
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

// Macro to generate wrapper functions for a specific type and channel combination
#define NPPI_COPY_WRAPPER(TYPE, SUFFIX, CHANNEL)                                                                       \
  NppStatus nppiCopy_##SUFFIX##_##CHANNEL##_Ctx(const TYPE *pSrc, Npp32s nSrcStep, TYPE *pDst, Npp32s nDstStep,        \
                                                NppiSize oSizeROI, NppStreamContext nppStreamCtx) {                    \
    NppStatus status = validateCopyInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);                                   \
    if (status != NPP_SUCCESS)                                                                                         \
      return status;                                                                                                   \
    return nppiCopy_##SUFFIX##_##CHANNEL##_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);           \
  }                                                                                                                    \
                                                                                                                       \
  NppStatus nppiCopy_##SUFFIX##_##CHANNEL(const TYPE *pSrc, Npp32s nSrcStep, TYPE *pDst, Npp32s nDstStep,              \
                                          NppiSize oSizeROI) {                                                         \
    NppStreamContext nppStreamCtx;                                                                                     \
    nppStreamCtx.hStream = 0;                                                                                          \
    return nppiCopy_##SUFFIX##_##CHANNEL##_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);                \
  }

// Generate all wrapper functions using macro
// 8u variants
NPPI_COPY_WRAPPER(Npp8u, 8u, C1R)
NPPI_COPY_WRAPPER(Npp8u, 8u, C3R)
NPPI_COPY_WRAPPER(Npp8u, 8u, C4R)

// 16u variants
NPPI_COPY_WRAPPER(Npp16u, 16u, C1R)
NPPI_COPY_WRAPPER(Npp16u, 16u, C3R)
NPPI_COPY_WRAPPER(Npp16u, 16u, C4R)

// 16s variants
NPPI_COPY_WRAPPER(Npp16s, 16s, C1R)
NPPI_COPY_WRAPPER(Npp16s, 16s, C3R)
NPPI_COPY_WRAPPER(Npp16s, 16s, C4R)

// 32s variants
NPPI_COPY_WRAPPER(Npp32s, 32s, C1R)
NPPI_COPY_WRAPPER(Npp32s, 32s, C3R)
NPPI_COPY_WRAPPER(Npp32s, 32s, C4R)

// 32f variants
NPPI_COPY_WRAPPER(Npp32f, 32f, C1R)
NPPI_COPY_WRAPPER(Npp32f, 32f, C3R)
NPPI_COPY_WRAPPER(Npp32f, 32f, C4R)

#undef NPPI_COPY_WRAPPER

// Helper for validating planar copy inputs (packed to planar)
template <int CHANNELS>
static inline NppStatus validatePlanarInputs_CxPxR(const void *pSrc, int nSrcStep, void *const pDst[], int nDstStep,
                                                    NppiSize oSizeROI) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  for (int c = 0; c < CHANNELS; c++) {
    if (!pDst[c]) {
      return NPP_NULL_POINTER_ERROR;
    }
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  return NPP_SUCCESS;
}

// Helper for validating planar copy inputs (planar to packed)
template <int CHANNELS>
static inline NppStatus validatePlanarInputs_PxCxR(const void *const pSrc[], int nSrcStep, void *pDst, int nDstStep,
                                                    NppiSize oSizeROI) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  for (int c = 0; c < CHANNELS; c++) {
    if (!pSrc[c]) {
      return NPP_NULL_POINTER_ERROR;
    }
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  return NPP_SUCCESS;
}

// Macro for C3P3R/C4P4R wrapper functions
#define NPPI_COPY_CXPXR_WRAPPER(TYPE, SUFFIX, CHANNELS)                                                                  \
  NppStatus nppiCopy_##SUFFIX##_C##CHANNELS##P##CHANNELS##R_Ctx(const TYPE *pSrc, Npp32s nSrcStep,                       \
                                                                TYPE *const pDst[CHANNELS], Npp32s nDstStep,              \
                                                                NppiSize oSizeROI, NppStreamContext nppStreamCtx) {       \
    NppStatus status = validatePlanarInputs_CxPxR<CHANNELS>(pSrc, nSrcStep, (void *const *)pDst, nDstStep, oSizeROI);   \
    if (status != NPP_SUCCESS)                                                                                           \
      return status;                                                                                                     \
    return nppiCopy_##SUFFIX##_C##CHANNELS##P##CHANNELS##R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,            \
                                                                   nppStreamCtx);                                        \
  }                                                                                                                      \
                                                                                                                         \
  NppStatus nppiCopy_##SUFFIX##_C##CHANNELS##P##CHANNELS##R(const TYPE *pSrc, Npp32s nSrcStep,                           \
                                                            TYPE *const pDst[CHANNELS], Npp32s nDstStep,                 \
                                                            NppiSize oSizeROI) {                                         \
    NppStreamContext nppStreamCtx;                                                                                       \
    nppStreamCtx.hStream = 0;                                                                                            \
    return nppiCopy_##SUFFIX##_C##CHANNELS##P##CHANNELS##R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);  \
  }

// Macro for P3C3R/P4C4R wrapper functions
#define NPPI_COPY_PXCXR_WRAPPER(TYPE, SUFFIX, CHANNELS)                                                                  \
  NppStatus nppiCopy_##SUFFIX##_P##CHANNELS##C##CHANNELS##R_Ctx(const TYPE *const pSrc[CHANNELS], Npp32s nSrcStep,       \
                                                                TYPE *pDst, Npp32s nDstStep, NppiSize oSizeROI,          \
                                                                NppStreamContext nppStreamCtx) {                         \
    NppStatus status = validatePlanarInputs_PxCxR<CHANNELS>((const void *const *)pSrc, nSrcStep, pDst, nDstStep,        \
                                                             oSizeROI);                                                  \
    if (status != NPP_SUCCESS)                                                                                           \
      return status;                                                                                                     \
    return nppiCopy_##SUFFIX##_P##CHANNELS##C##CHANNELS##R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,            \
                                                                   nppStreamCtx);                                        \
  }                                                                                                                      \
                                                                                                                         \
  NppStatus nppiCopy_##SUFFIX##_P##CHANNELS##C##CHANNELS##R(const TYPE *const pSrc[CHANNELS], Npp32s nSrcStep,           \
                                                            TYPE *pDst, Npp32s nDstStep, NppiSize oSizeROI) {            \
    NppStreamContext nppStreamCtx;                                                                                       \
    nppStreamCtx.hStream = 0;                                                                                            \
    return nppiCopy_##SUFFIX##_P##CHANNELS##C##CHANNELS##R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);  \
  }

// C3P3R wrappers
NPPI_COPY_CXPXR_WRAPPER(Npp8u, 8u, 3)
NPPI_COPY_CXPXR_WRAPPER(Npp16u, 16u, 3)
NPPI_COPY_CXPXR_WRAPPER(Npp16s, 16s, 3)
NPPI_COPY_CXPXR_WRAPPER(Npp32s, 32s, 3)
NPPI_COPY_CXPXR_WRAPPER(Npp32f, 32f, 3)

// C4P4R wrappers
NPPI_COPY_CXPXR_WRAPPER(Npp8u, 8u, 4)
NPPI_COPY_CXPXR_WRAPPER(Npp16u, 16u, 4)
NPPI_COPY_CXPXR_WRAPPER(Npp32f, 32f, 4)

// P3C3R wrappers
NPPI_COPY_PXCXR_WRAPPER(Npp8u, 8u, 3)
NPPI_COPY_PXCXR_WRAPPER(Npp16u, 16u, 3)
NPPI_COPY_PXCXR_WRAPPER(Npp16s, 16s, 3)
NPPI_COPY_PXCXR_WRAPPER(Npp32s, 32s, 3)
NPPI_COPY_PXCXR_WRAPPER(Npp32f, 32f, 3)

// P4C4R wrappers
NPPI_COPY_PXCXR_WRAPPER(Npp8u, 8u, 4)
NPPI_COPY_PXCXR_WRAPPER(Npp16u, 16u, 4)
NPPI_COPY_PXCXR_WRAPPER(Npp32f, 32f, 4)

#undef NPPI_COPY_CXPXR_WRAPPER
#undef NPPI_COPY_PXCXR_WRAPPER
