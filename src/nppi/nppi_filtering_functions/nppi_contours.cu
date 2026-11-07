// Contour interpolation / geometry-list functions — best-effort.
//
// These entry points are the tail of NPP's multi-call contour pipeline and
// consume the private direction/geometry structures produced by
// nppiCompressedMarkerLabelsUFInfo.  The exact marching-squares sub-pixel
// refinement and the geometry-list ordering were not decoded, so this file
// implements the documented data flow with simple, deterministic rules:
//
//  * nppiContoursImageMarchingSquaresInterpolation_{32f,64f}: a pixel is a
//    contour pixel when its contours-image value or its direction-info centre
//    direction is non-zero; its interpolated point is the pixel centre
//    (x, y), otherwise {0, 0}.  The per-label found/offset host lists and the
//    block-segment lists are initialised for the requested label range.
//  * nppiCompressedMarkerLabelsUFContoursOutputGeometryLists_C1R (a legacy
//    11.4 declaration that NPP 13.4 no longer exports): walks the requested
//    marker labels in order, assigns cumulative starting offsets from the
//    marker-info pixel counts, copies each contour's pixels-found count and
//    fills the host geometry entries with the image location derived from
//    the offset index.
//
// Validation: nFirst > nLast is NPP_BAD_ARGUMENT_ERROR and precedes the
// pointer checks; null required pointers are NPP_NULL_POINTER_ERROR; an empty
// ROI is a no-op for the interpolation kernels.
#include "npp.h"

#include <cuda_runtime.h>
#include <cstddef>

namespace {

template <typename PointT>
__global__ void contourInterpolateKernel(const Npp8u *pContoursImage, int nContoursImageStep, PointT *pInterp,
                                         int nInterpStep, const NppiContourPixelDirectionInfo *pDirection,
                                         int nDirectionStep, int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const Npp8u contour = pContoursImage[(size_t)y * nContoursImageStep + x];
  const NppiContourPixelDirectionInfo *dir =
      (const NppiContourPixelDirectionInfo *)((const char *)pDirection + (size_t)y * nDirectionStep) + x;
  const bool isContour = contour != 0 || dir->nContourDirectionCenterPixel != 0;
  PointT *out = (PointT *)((char *)pInterp + (size_t)y * nInterpStep) + x;
  if (isContour) {
    out->x = (decltype(out->x))x;
    out->y = (decltype(out->y))y;
  } else {
    out->x = (decltype(out->x))0;
    out->y = (decltype(out->y))0;
  }
}

template <typename PointT>
NppStatus contourInterpolate(Npp8u *pContoursImageDev, Npp32s nContoursImageStep, PointT *pInterpDev,
                             Npp32s nInterpStep, NppiContourPixelDirectionInfo *pDirectionDev,
                             Npp32s nDirectionStep, NppiContourPixelGeometryInfo *pGeometryDev,
                             NppiContourPixelGeometryInfo *pGeometryHost, PointT *pInterpGeometryDev,
                             Npp32u *pPixelsFoundHost, Npp32u *pStartOffsetDev, Npp32u *pStartOffsetHost,
                             Npp32u nTotalContourPixels, Npp32u nMaxMarkerLabelID, Npp32u nFirstID, Npp32u nLastID,
                             NppiContourBlockSegment *pBlockSegmentDev, NppiContourBlockSegment *pBlockSegmentHost,
                             NppiSize oSizeROI, cudaStream_t stream) {
  if (nFirstID > nLastID) return NPP_BAD_ARGUMENT_ERROR;
  if (pContoursImageDev == nullptr || pInterpDev == nullptr || pDirectionDev == nullptr ||
      pGeometryDev == nullptr || pGeometryHost == nullptr || pInterpGeometryDev == nullptr ||
      pPixelsFoundHost == nullptr || pStartOffsetDev == nullptr || pStartOffsetHost == nullptr ||
      pBlockSegmentDev == nullptr || pBlockSegmentHost == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;

  dim3 block(16, 16);
  dim3 grid((oSizeROI.width + 15) / 16, (oSizeROI.height + 15) / 16);
  contourInterpolateKernel<PointT><<<grid, block, 0, stream>>>(pContoursImageDev, nContoursImageStep, pInterpDev,
                                                               nInterpStep, pDirectionDev, nDirectionStep,
                                                               oSizeROI.width, oSizeROI.height);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;

  // Initialise the per-label bookkeeping for the requested range.
  const Npp32u first = nFirstID < nMaxMarkerLabelID ? nFirstID: nMaxMarkerLabelID;
  const Npp32u last = nLastID < nMaxMarkerLabelID ? nLastID: nMaxMarkerLabelID;
  for (Npp32u id = first; id <= last; ++id) {
    pPixelsFoundHost[id] = 0;
    pStartOffsetHost[id] = 0;
    pBlockSegmentHost[id].nMarkerLabelID = id;
    pBlockSegmentHost[id].nContourPixelCount = 0;
    pBlockSegmentHost[id].nContourStartingPixelOffset = 0;
    pBlockSegmentHost[id].nSegmentNum = 0;
  }
  if (cudaMemcpyAsync(pStartOffsetDev, pStartOffsetHost, (size_t)(last - first + 1) * sizeof(Npp32u),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess) {
    return NPP_MEMCPY_ERROR;
  }
  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

}  // namespace

NppStatus nppiContoursImageMarchingSquaresInterpolation_32f_C1R_Ctx(
    Npp8u *pContoursImageDev, Npp32s nContoursImageStep, NppiPoint32f *pContoursInterpolatedImageDev,
    Npp32s nContoursInterpolatedImageStep, NppiContourPixelDirectionInfo *pContoursDirectionImageDev,
    Npp32s nContoursDirectionImageStep, NppiContourPixelGeometryInfo *pContoursPixelGeometryListsDev,
    NppiContourPixelGeometryInfo *pContoursPixelGeometryListsHost,
    NppiPoint32f *pContoursInterpolatedGeometryListsDev, Npp32u *pContoursPixelsFoundListHost,
    Npp32u *pContoursPixelsStartingOffsetDev, Npp32u *pContoursPixelsStartingOffsetHost,
    Npp32u nTotalImagePixelContourCount, Npp32u nMaxMarkerLabelID, Npp32u nFirstContourGeometryListID,
    Npp32u nLastContourGeometryListID, NppiContourBlockSegment *pContoursBlockSegmentListDev,
    NppiContourBlockSegment *pContoursBlockSegmentListHost, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return contourInterpolate<NppiPoint32f>(
      pContoursImageDev, nContoursImageStep, pContoursInterpolatedImageDev, nContoursInterpolatedImageStep,
      pContoursDirectionImageDev, nContoursDirectionImageStep, pContoursPixelGeometryListsDev,
      pContoursPixelGeometryListsHost, pContoursInterpolatedGeometryListsDev, pContoursPixelsFoundListHost,
      pContoursPixelsStartingOffsetDev, pContoursPixelsStartingOffsetHost, nTotalImagePixelContourCount,
      nMaxMarkerLabelID, nFirstContourGeometryListID, nLastContourGeometryListID, pContoursBlockSegmentListDev,
      pContoursBlockSegmentListHost, oSizeROI, nppStreamCtx.hStream);
}

NppStatus nppiContoursImageMarchingSquaresInterpolation_64f_C1R_Ctx(
    Npp8u *pContoursImageDev, Npp32s nContoursImageStep, NppiPoint64f *pContoursInterpolatedImageDev,
    Npp32s nContoursInterpolatedImageStep, NppiContourPixelDirectionInfo *pContoursDirectionImageDev,
    Npp32s nContoursDirectionImageStep, NppiContourPixelGeometryInfo *pContoursPixelGeometryListsDev,
    NppiContourPixelGeometryInfo *pContoursPixelGeometryListsHost,
    NppiPoint64f *pContoursInterpolatedGeometryListsDev, Npp32u *pContoursPixelsFoundListHost,
    Npp32u *pContoursPixelsStartingOffsetDev, Npp32u *pContoursPixelsStartingOffsetHost,
    Npp32u nTotalImagePixelContourCount, Npp32u nMaxMarkerLabelID, Npp32u nFirstContourGeometryListID,
    Npp32u nLastContourGeometryListID, NppiContourBlockSegment *pContoursBlockSegmentListDev,
    NppiContourBlockSegment *pContoursBlockSegmentListHost, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return contourInterpolate<NppiPoint64f>(
      pContoursImageDev, nContoursImageStep, pContoursInterpolatedImageDev, nContoursInterpolatedImageStep,
      pContoursDirectionImageDev, nContoursDirectionImageStep, pContoursPixelGeometryListsDev,
      pContoursPixelGeometryListsHost, pContoursInterpolatedGeometryListsDev, pContoursPixelsFoundListHost,
      pContoursPixelsStartingOffsetDev, pContoursPixelsStartingOffsetHost, nTotalImagePixelContourCount,
      nMaxMarkerLabelID, nFirstContourGeometryListID, nLastContourGeometryListID, pContoursBlockSegmentListDev,
      pContoursBlockSegmentListHost, oSizeROI, nppStreamCtx.hStream);
}

// Host-side legacy geometry-list assembly (NPP 13.4 no longer exports this
// symbol; the declaration survives in the unified 11.4 API header).
NppStatus nppiCompressedMarkerLabelsUFContoursOutputGeometryLists_C1R(
    NppiContourPixelGeometryInfo *pContoursPixelGeometryListsHost,
    NppiCompressedMarkerLabelsInfo *pMarkerLabelsInfoListHost, Npp8u *pContoursGeometryImageHost,
    Npp32s nContoursGeometryImageStep, Npp32u *pContoursPixelCountsListHost,
    Npp32u *pContoursPixelsFoundListHost, Npp32u *pContoursPixelsStartingOffsetHost, Npp32u nMaxMarkerLabelID,
    Npp32u nFirstContourGeometryListID, Npp32u nLastContourGeometryListID,
    Npp32u bOutputInCounterclockwiseOrder, NppiSize oSizeROI) {
  if (nFirstContourGeometryListID > nLastContourGeometryListID) return NPP_BAD_ARGUMENT_ERROR;
  if (pContoursPixelGeometryListsHost == nullptr || pMarkerLabelsInfoListHost == nullptr ||
      pContoursGeometryImageHost == nullptr || pContoursPixelCountsListHost == nullptr ||
      pContoursPixelsFoundListHost == nullptr || pContoursPixelsStartingOffsetHost == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
  (void)bOutputInCounterclockwiseOrder;

  Npp32u offset = 0;
  for (Npp32u id = nFirstContourGeometryListID; id <= nLastContourGeometryListID && id < nMaxMarkerLabelID; ++id) {
    const Npp32u count = pMarkerLabelsInfoListHost[id].nContourPixelCount;
    const Npp32u found = pMarkerLabelsInfoListHost[id].nContourPixelsFound;
    pContoursPixelsStartingOffsetHost[id] = offset;
    pContoursPixelCountsListHost[id] = count;
    pContoursPixelsFoundListHost[id] = found;
    for (Npp32u i = 0; i < count; ++i) {
      const Npp32u index = offset + i;
      const NppiPoint location = {(Npp32s)(index % (Npp32u)oSizeROI.width),
                                  (Npp32s)((index / (Npp32u)oSizeROI.width) % (Npp32u)oSizeROI.height)};
      NppiContourPixelGeometryInfo &info = pContoursPixelGeometryListsHost[index];
      info.oContourOrderedGeometryLocation = location;
      info.oContourCenterPixelLocation = location;
      info.oContourPrevPixelLocation = location;
      info.oContourNextPixelLocation = location;
      info.nOrderIndex = (Npp32s)i;
      info.nReverseOrderIndex = (Npp32s)(count - i - 1);
      info.nFirstIndex = offset;
      info.nLastIndex = offset + count - 1;
    }
    offset += count;
  }
  return NPP_NO_ERROR;
}
