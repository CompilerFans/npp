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
NppStatus nppiCopy_32f_C3P3R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *const pDst[3], int nDstStep,
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

// Special case: C3P3R (packed to planar)
NppStatus nppiCopy_32f_C3P3R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *const pDst[3], Npp32s nDstStep,
                                 NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  for (int c = 0; c < 3; c++) {
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

  return nppiCopy_32f_C3P3R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_32f_C3P3R(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *const pDst[3], Npp32s nDstStep,
                             NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiCopy_32f_C3P3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}
