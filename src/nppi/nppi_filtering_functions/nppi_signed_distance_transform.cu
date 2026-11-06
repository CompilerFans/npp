// Signed distance transform (PBA family) — exact mask semantics decoded from
// NPP 13.4 libnppif.
//
// The reference runs the two PBA passes on outside (source <= 0, or < cutoff)
// and inside (source >= 1) pixels and then merges them with
// generateAdditionalRelativeShiftOutput. That output kernel does NOT emit the
// Euclidean distance: with the site displacement (dx,dy), the raw shift
// (sx,sy) and C = 30 + cutoff it computes
//
//   a = |dx| + sx, b = |dy| + sy
//   t = sqrt((max(a,b) - C)^2 + min(a,b)^2)
//
// and stores +t for pixels inside a region and -t outside. The 30 is a fixed
// band-reconstruction constant in the reference, independent of the ROI.
// Voronoi outputs:
//   signed: pDstVoronoi      = nearest outside pixel (self for outside pixels)
//           pDstVoronoiIndices = nearest opposite-class pixel (the one used
//                                by the transform)
//           relative manhattan = pixel - that opposite site
//   abs 32f: voronoi and indices = nearest inside pixel; relative = absolute
//            Manhattan distance; the transform is positive.
//   abs 64f: the reference ignores the AA buffer and emits the same signed
//            folded transform as the signed 64f entry point, with absolute
//            relative outputs.
// When there are no outside pixels the signed transform uses the dummy site
// (-1,0); when there are no inside pixels the signed outputs are all zero and
// the abs paths return NPP_NULL_POINTER_ERROR, as measured.
//
// Known limitations:
//   * fractional source values (0 < v < 1, v >= cutoff) feed a sub-pixel
//     antialiasing path in the reference; this executor treats them as
//     outside-class non-sites (0/1 masks are exact);
//   * the 32f64f abs entry point requires the antialiasing workspace, whose
//     reference values for binary masks follow the measured horizontal
//     distance rule implemented in aaAbsKernel (see the shared header); the
//     reference faults on open or fragmented masks, so only closed regions
//     are covered by the differential tests.
#include "npp.h"
#include "nppi_signed_distance_common.cuh"

#include <cmath>
#include <cuda_runtime.h>

using namespace nppi_sdt;

namespace {

template <typename T, typename OutT>
NppStatus validateAndRun(T *pSrc, int nSrcStep, T cutoff, OutT subX, OutT subY, Npp16s *pDstVoronoi,
                         int nDstVoronoiStep, Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,
                         Npp16u *pDstRelative, int nDstRelativeStep, OutT *pDstTransform, int nDstTransformStep,
                         NppiSize oSizeROI, Npp8u *pDeviceBuffer, int mode, bool antialiasingRequired,
                         Npp8u *pAntialiasingDeviceBuffer, cudaStream_t stream) {
  if (pSrc == nullptr || pDeviceBuffer == nullptr) return NPP_NULL_POINTER_ERROR;
  const bool useAA = pAntialiasingDeviceBuffer != nullptr;
  if (antialiasingRequired && !useAA) return NPP_NULL_POINTER_ERROR;
  // The antialiasing path requires the transform output (checked before the
  // ROI, as measured on the 5090).
  if (useAA && pDstTransform == nullptr) return NPP_NULL_POINTER_ERROR;
  if (!pDstVoronoi && !pDstVoronoiIndices && !pDstRelative && !pDstTransform) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width < kMinRoi || oSizeROI.height < kMinRoi) return NPP_SIZE_ERROR;

  const int width = oSizeROI.width, height = oSizeROI.height;
  if ((pDstVoronoi && nDstVoronoiStep < width * 4) ||
      (pDstVoronoiIndices && nDstVoronoiIndicesStep < width * 4) ||
      (pDstRelative && nDstRelativeStep < width * 4)) {
    return NPP_STEP_ERROR;
  }
  // Plain comparison: NaN passes, infinities do not.
  if (cutoff < (T)0 || cutoff >= (T)1) return NPP_RANGE_ERROR;

  int insideCount = 0;
  const cudaError_t err = runKernels<T, OutT>(pSrc, nSrcStep, cutoff, subX, subY, pDstVoronoi, nDstVoronoiStep,
                                              pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstRelative,
                                              nDstRelativeStep, pDstTransform, nDstTransformStep, width, height,
                                              pDeviceBuffer, mode, &insideCount, stream);
  if (err != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  if (insideCount == 0 && (mode == kAbsFolded || mode == kAaAbs)) return NPP_NULL_POINTER_ERROR;
  return NPP_NO_ERROR;
}

}  // namespace

NppStatus nppiSignedDistanceTransformPBA_32f_C1R_Ctx(
    Npp32f *pSrc, int nSrcStep, Npp32f nCutoffValue, Npp32f nSubPixelXShift, Npp32f nSubPixelYShift,
    Npp16s *pDstVoronoi, int nDstVoronoiStep, Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,
    Npp16s *pDstVoronoiRelativeManhattanDistances, int nDstVoronoiRelativeManhattanDistancesStep,
    Npp32f *pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,
    NppStreamContext nppStreamCtx) {
  return validateAndRun<Npp32f, Npp32f>(
      pSrc, nSrcStep, nCutoffValue, nSubPixelXShift, nSubPixelYShift, pDstVoronoi, nDstVoronoiStep,
      pDstVoronoiIndices, nDstVoronoiIndicesStep, (Npp16u *)pDstVoronoiRelativeManhattanDistances,
      nDstVoronoiRelativeManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer,
      kSignedFolded, false, nullptr, nppStreamCtx.hStream);
}

NppStatus nppiSignedDistanceTransformAbsPBA_32f_C1R_Ctx(
    Npp32f *pSrc, int nSrcStep, Npp32f nCutoffValue, Npp32f nSubPixelXShift, Npp32f nSubPixelYShift,
    Npp16s *pDstVoronoi, int nDstVoronoiStep, Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,
    Npp16u *pDstVoronoiAbsoluteManhattanDistances, int nDstVoronoiAbsoluteManhattanDistancesStep,
    Npp32f *pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,
    NppStreamContext nppStreamCtx) {
  return validateAndRun<Npp32f, Npp32f>(
      pSrc, nSrcStep, nCutoffValue, nSubPixelXShift, nSubPixelYShift, pDstVoronoi, nDstVoronoiStep,
      pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstVoronoiAbsoluteManhattanDistances,
      nDstVoronoiAbsoluteManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer,
      kAbsFolded, false, nullptr, nppStreamCtx.hStream);
}

NppStatus nppiSignedDistanceTransformAbsPBA_32f64f_C1R_Ctx(
    Npp32f *pSrc, int nSrcStep, Npp32f nCutoffValue, Npp64f nSubPixelXShift, Npp64f nSubPixelYShift,
    Npp16s *pDstVoronoi, int nDstVoronoiStep, Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,
    Npp16u *pDstVoronoiAbsoluteManhattanDistances, int nDstVoronoiAbsoluteManhattanDistancesStep,
    Npp64f *pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,
    Npp8u *pAntialiasingDeviceBuffer, NppStreamContext nppStreamCtx) {
  return validateAndRun<Npp32f, Npp64f>(
      pSrc, nSrcStep, nCutoffValue, nSubPixelXShift, nSubPixelYShift, pDstVoronoi, nDstVoronoiStep,
      pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstVoronoiAbsoluteManhattanDistances,
      nDstVoronoiAbsoluteManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer,
      kAaAbs, true, pAntialiasingDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppiSignedDistanceTransformAbsPBA_64f_C1R_Ctx(
    Npp64f *pSrc, int nSrcStep, Npp64f nCutoffValue, Npp64f nSubPixelXShift, Npp64f nSubPixelYShift,
    Npp16s *pDstVoronoi, int nDstVoronoiStep, Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,
    Npp16u *pDstVoronoiAbsoluteManhattanDistances, int nDstVoronoiAbsoluteManhattanDistancesStep,
    Npp64f *pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,
    Npp8u *pAntialiasingDeviceBuffer, NppStreamContext nppStreamCtx) {
  return validateAndRun<Npp64f, Npp64f>(
      pSrc, nSrcStep, nCutoffValue, nSubPixelXShift, nSubPixelYShift, pDstVoronoi, nDstVoronoiStep,
      pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstVoronoiAbsoluteManhattanDistances,
      nDstVoronoiAbsoluteManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer,
      kSignedAbsRel, false, pAntialiasingDeviceBuffer, nppStreamCtx.hStream);
}
