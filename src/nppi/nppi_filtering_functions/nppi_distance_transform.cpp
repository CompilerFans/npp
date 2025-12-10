#include "../../npp_internal.h"
#include "npp.h"
#include <cuda_runtime.h>

// Declare GPU implementation functions
extern "C" {
NppStatus nppiDistanceTransformPBAGetBufferSize_impl(NppiSize oSizeROI, size_t *hpBufferSize);
NppStatus nppiDistanceTransformPBAGetAntialiasingBufferSize_impl(NppiSize oSizeROI, size_t *hpAntialiasingBufferSize);
NppStatus nppiSignedDistanceTransformPBAGetBufferSize_impl(NppiSize oSizeROI, size_t *hpBufferSize);
NppStatus nppiSignedDistanceTransformPBAGet64fBufferSize_impl(NppiSize oSizeROI, size_t *hpBufferSize);

NppStatus nppiDistanceTransformAbsPBA_8u16u_C1R_Ctx_impl(
    Npp8u *pSrc, int nSrcStep, Npp8u nMinSiteValue, Npp8u nMaxSiteValue, Npp16s *pDstVoronoi, int nDstVoronoiStep,
    Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep, Npp16u *pDstVoronoiAbsoluteManhattanDistances,
    int nDstVoronoiAbsoluteManhattanDistancesStep, Npp16u *pDstTransform, int nDstTransformStep, NppiSize oSizeROI,
    Npp8u *pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus nppiDistanceTransformPBA_8u32f_C1R_Ctx_impl(Npp8u *pSrc, int nSrcStep, Npp8u nMinSiteValue,
                                                      Npp8u nMaxSiteValue, Npp16s *pDstVoronoi, int nDstVoronoiStep,
                                                      Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,
                                                      Npp16s *pDstVoronoiRelativeManhattanDistances,
                                                      int nDstVoronoiRelativeManhattanDistancesStep,
                                                      Npp32f *pDstTransform, int nDstTransformStep, NppiSize oSizeROI,
                                                      Npp8u *pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus nppiDistanceTransformPBA_8u64f_C1R_Ctx_impl(
    Npp8u *pSrc, int nSrcStep, Npp8u nMinSiteValue, Npp8u nMaxSiteValue, Npp16s *pDstVoronoi, int nDstVoronoiStep,
    Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep, Npp16s *pDstVoronoiRelativeManhattanDistances,
    int nDstVoronoiRelativeManhattanDistancesStep, Npp64f *pDstTransform, int nDstTransformStep, NppiSize oSizeROI,
    Npp8u *pDeviceBuffer, Npp8u *pAntialiasingDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus nppiSignedDistanceTransformPBA_32f64f_C1R_Ctx_impl(
    Npp32f *pSrc, int nSrcStep, Npp32f nCutoffValue, Npp64f nSubPixelXShift, Npp64f nSubPixelYShift,
    Npp16s *pDstVoronoi, int nDstVoronoiStep, Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,
    Npp16s *pDstVoronoiRelativeManhattanDistances, int nDstVoronoiRelativeManhattanDistancesStep, Npp64f *pDstTransform,
    int nDstTransformStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp8u *pAntialiasingDeviceBuffer,
    NppStreamContext nppStreamCtx);

NppStatus nppiSignedDistanceTransformPBA_64f_C1R_Ctx_impl(
    Npp64f *pSrc, int nSrcStep, Npp64f nCutoffValue, Npp64f nSubPixelXShift, Npp64f nSubPixelYShift,
    Npp16s *pDstVoronoi, int nDstVoronoiStep, Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,
    Npp16s *pDstVoronoiRelativeManhattanDistances, int nDstVoronoiRelativeManhattanDistancesStep, Npp64f *pDstTransform,
    int nDstTransformStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp8u *pAntialiasingDeviceBuffer,
    NppStreamContext nppStreamCtx);
}

// Input validation helper
static inline NppStatus validateDistanceTransformInputs(NppiSize oSizeROI, size_t *hpBufferSize) {
  if (hpBufferSize == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  return NPP_SUCCESS;
}

static inline NppStatus validateDistanceTransformImageInputs(const void *pSrc, int nSrcStep, NppiSize oSizeROI,
                                                             Npp8u *pDeviceBuffer) {
  if (pSrc == nullptr || pDeviceBuffer == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (nSrcStep <= 0) {
    return NPP_STEP_ERROR;
  }
  return NPP_SUCCESS;
}

// Buffer size functions
NppStatus nppiDistanceTransformPBAGetBufferSize(NppiSize oSizeROI, size_t *hpBufferSize) {
  NppStatus status = validateDistanceTransformInputs(oSizeROI, hpBufferSize);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiDistanceTransformPBAGetBufferSize_impl(oSizeROI, hpBufferSize);
}

NppStatus nppiDistanceTransformPBAGetAntialiasingBufferSize(NppiSize oSizeROI, size_t *hpAntialiasingBufferSize) {
  NppStatus status = validateDistanceTransformInputs(oSizeROI, hpAntialiasingBufferSize);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiDistanceTransformPBAGetAntialiasingBufferSize_impl(oSizeROI, hpAntialiasingBufferSize);
}

NppStatus nppiSignedDistanceTransformPBAGetBufferSize(NppiSize oSizeROI, size_t *hpBufferSize) {
  NppStatus status = validateDistanceTransformInputs(oSizeROI, hpBufferSize);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiSignedDistanceTransformPBAGetBufferSize_impl(oSizeROI, hpBufferSize);
}

NppStatus nppiSignedDistanceTransformPBAGet64fBufferSize(NppiSize oSizeROI, size_t *hpBufferSize) {
  NppStatus status = validateDistanceTransformInputs(oSizeROI, hpBufferSize);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiSignedDistanceTransformPBAGet64fBufferSize_impl(oSizeROI, hpBufferSize);
}

// Distance transform 8u to 16u with absolute Manhattan distances
NppStatus nppiDistanceTransformAbsPBA_8u16u_C1R_Ctx(Npp8u *pSrc, int nSrcStep, Npp8u nMinSiteValue, Npp8u nMaxSiteValue,
                                                    Npp16s *pDstVoronoi, int nDstVoronoiStep,
                                                    Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,
                                                    Npp16u *pDstVoronoiAbsoluteManhattanDistances,
                                                    int nDstVoronoiAbsoluteManhattanDistancesStep,
                                                    Npp16u *pDstTransform, int nDstTransformStep, NppiSize oSizeROI,
                                                    Npp8u *pDeviceBuffer, NppStreamContext nppStreamCtx) {

  NppStatus status = validateDistanceTransformImageInputs(pSrc, nSrcStep, oSizeROI, pDeviceBuffer);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiDistanceTransformAbsPBA_8u16u_C1R_Ctx_impl(
      pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices,
      nDstVoronoiIndicesStep, pDstVoronoiAbsoluteManhattanDistances, nDstVoronoiAbsoluteManhattanDistancesStep,
      pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, nppStreamCtx);
}

// Distance transform 8u to 32f
NppStatus nppiDistanceTransformPBA_8u32f_C1R_Ctx(Npp8u *pSrc, int nSrcStep, Npp8u nMinSiteValue, Npp8u nMaxSiteValue,
                                                 Npp16s *pDstVoronoi, int nDstVoronoiStep, Npp16s *pDstVoronoiIndices,
                                                 int nDstVoronoiIndicesStep,
                                                 Npp16s *pDstVoronoiRelativeManhattanDistances,
                                                 int nDstVoronoiRelativeManhattanDistancesStep, Npp32f *pDstTransform,
                                                 int nDstTransformStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,
                                                 NppStreamContext nppStreamCtx) {

  NppStatus status = validateDistanceTransformImageInputs(pSrc, nSrcStep, oSizeROI, pDeviceBuffer);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiDistanceTransformPBA_8u32f_C1R_Ctx_impl(
      pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices,
      nDstVoronoiIndicesStep, pDstVoronoiRelativeManhattanDistances, nDstVoronoiRelativeManhattanDistancesStep,
      pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, nppStreamCtx);
}

// Distance transform 8u to 64f
NppStatus nppiDistanceTransformPBA_8u64f_C1R_Ctx(Npp8u *pSrc, int nSrcStep, Npp8u nMinSiteValue, Npp8u nMaxSiteValue,
                                                 Npp16s *pDstVoronoi, int nDstVoronoiStep, Npp16s *pDstVoronoiIndices,
                                                 int nDstVoronoiIndicesStep,
                                                 Npp16s *pDstVoronoiRelativeManhattanDistances,
                                                 int nDstVoronoiRelativeManhattanDistancesStep, Npp64f *pDstTransform,
                                                 int nDstTransformStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,
                                                 Npp8u *pAntialiasingDeviceBuffer, NppStreamContext nppStreamCtx) {

  NppStatus status = validateDistanceTransformImageInputs(pSrc, nSrcStep, oSizeROI, pDeviceBuffer);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiDistanceTransformPBA_8u64f_C1R_Ctx_impl(
      pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices,
      nDstVoronoiIndicesStep, pDstVoronoiRelativeManhattanDistances, nDstVoronoiRelativeManhattanDistancesStep,
      pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, pAntialiasingDeviceBuffer, nppStreamCtx);
}

// Signed distance transform 32f to 64f
NppStatus nppiSignedDistanceTransformPBA_32f64f_C1R_Ctx(
    Npp32f *pSrc, int nSrcStep, Npp32f nCutoffValue, Npp64f nSubPixelXShift, Npp64f nSubPixelYShift,
    Npp16s *pDstVoronoi, int nDstVoronoiStep, Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,
    Npp16s *pDstVoronoiRelativeManhattanDistances, int nDstVoronoiRelativeManhattanDistancesStep, Npp64f *pDstTransform,
    int nDstTransformStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp8u *pAntialiasingDeviceBuffer,
    NppStreamContext nppStreamCtx) {

  NppStatus status = validateDistanceTransformImageInputs(pSrc, nSrcStep, oSizeROI, pDeviceBuffer);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiSignedDistanceTransformPBA_32f64f_C1R_Ctx_impl(
      pSrc, nSrcStep, nCutoffValue, nSubPixelXShift, nSubPixelYShift, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices,
      nDstVoronoiIndicesStep, pDstVoronoiRelativeManhattanDistances, nDstVoronoiRelativeManhattanDistancesStep,
      pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, pAntialiasingDeviceBuffer, nppStreamCtx);
}

// Signed distance transform 64f to 64f
NppStatus nppiSignedDistanceTransformPBA_64f_C1R_Ctx(
    Npp64f *pSrc, int nSrcStep, Npp64f nCutoffValue, Npp64f nSubPixelXShift, Npp64f nSubPixelYShift,
    Npp16s *pDstVoronoi, int nDstVoronoiStep, Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,
    Npp16s *pDstVoronoiRelativeManhattanDistances, int nDstVoronoiRelativeManhattanDistancesStep, Npp64f *pDstTransform,
    int nDstTransformStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp8u *pAntialiasingDeviceBuffer,
    NppStreamContext nppStreamCtx) {

  NppStatus status = validateDistanceTransformImageInputs(pSrc, nSrcStep, oSizeROI, pDeviceBuffer);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiSignedDistanceTransformPBA_64f_C1R_Ctx_impl(
      pSrc, nSrcStep, nCutoffValue, nSubPixelXShift, nSubPixelYShift, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices,
      nDstVoronoiIndicesStep, pDstVoronoiRelativeManhattanDistances, nDstVoronoiRelativeManhattanDistancesStep,
      pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, pAntialiasingDeviceBuffer, nppStreamCtx);
}
