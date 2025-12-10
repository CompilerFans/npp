#include "../../npp_internal.h"
#include "npp.h"
#include <cuda_runtime.h>

// Declare GPU implementation functions
extern "C" {
NppStatus nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R_impl(unsigned int nMaxMarkerLabelID,
                                                                   unsigned int *hpBufferSize);

NppStatus nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx_impl(
    Npp32u *pCompressedMarkerLabels, Npp32s nCompressedMarkerLabelsStep, NppiSize oSizeROI,
    unsigned int nMaxMarkerLabelID, NppiCompressedMarkerLabelsInfo *pMarkerLabelsInfoList, Npp8u *pContoursImage,
    Npp32s nContoursImageStep, NppiContourPixelDirectionInfo *pContoursDirectionImage,
    Npp32s nContoursDirectionImageStep, NppiContourTotalsInfo *pContoursTotalsInfoHost,
    Npp32u *pContoursPixelCountsListDev, Npp32u *pContoursPixelCountsListHost, Npp32u *pContoursPixelStartingOffsetDev,
    Npp32u *pContoursPixelStartingOffsetHost, NppStreamContext nppStreamCtx);

NppStatus nppiCompressedMarkerLabelsUFGetGeometryListsSize_C1R_impl(Npp32u nMaxContourPixelGeometryInfoCount,
                                                                    Npp32u *hpBufferSize);

NppStatus nppiCompressedMarkerLabelsUFContoursGenerateGeometryLists_C1R_Ctx_impl(
    NppiCompressedMarkerLabelsInfo *pMarkerLabelsInfoListDev, NppiCompressedMarkerLabelsInfo *pMarkerLabelsInfoListHost,
    NppiContourPixelDirectionInfo *pContoursDirectionImageDev, Npp32s nContoursDirectionImageStep,
    NppiContourPixelGeometryInfo *pContoursPixelGeometryListsDev,
    NppiContourPixelGeometryInfo *pContoursPixelGeometryListsHost, Npp8u *pContoursGeometryImageHost,
    Npp32s nContoursGeometryImageStep, Npp32u *pContoursPixelCountsListDev, Npp32u *pContoursPixelsFoundListDev,
    Npp32u *pContoursPixelsFoundListHost, Npp32u *pContoursPixelsStartingOffsetDev,
    Npp32u *pContoursPixelsStartingOffsetHost, Npp32u nTotalImagePixelContourCount, Npp32u nMaxMarkerLabelID,
    Npp32u nFirstContourGeometryListID, Npp32u nLastContourGeometryListID,
    NppiContourBlockSegment *pContoursBlockSegmentListDev, NppiContourBlockSegment *pContoursBlockSegmentListHost,
    Npp32u bOutputInCounterclockwiseOrder, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus nppiCompressMarkerLabelsUFBatch_32u_C1IR_Advanced_Ctx_impl(
    NppiImageDescriptor *pSrcDstBatchList, NppiBufferDescriptor *pBufferList, unsigned int *pNewMaxLabelIDList,
    int nBatchSize, NppiSize oMaxSizeROI, int nLargestPerImageBufferSize, NppStreamContext nppStreamCtx);
}

// Input validation helpers
static inline NppStatus validateCompressedMarkerLabelsInputs(unsigned int nMaxMarkerLabelID,
                                                             unsigned int *hpBufferSize) {
  if (hpBufferSize == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nMaxMarkerLabelID == 0) {
    return NPP_BAD_ARGUMENT_ERROR;
  }
  return NPP_SUCCESS;
}

static inline NppStatus
validateCompressedMarkerLabelsInfoInputs(Npp32u *pCompressedMarkerLabels, Npp32s nCompressedMarkerLabelsStep,
                                         NppiSize oSizeROI, unsigned int nMaxMarkerLabelID,
                                         NppiCompressedMarkerLabelsInfo *pMarkerLabelsInfoList) {
  if (pCompressedMarkerLabels == nullptr || pMarkerLabelsInfoList == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (nCompressedMarkerLabelsStep <= 0) {
    return NPP_STEP_ERROR;
  }
  if (nMaxMarkerLabelID == 0) {
    return NPP_BAD_ARGUMENT_ERROR;
  }
  return NPP_SUCCESS;
}

// Get info list buffer size
NppStatus nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R(unsigned int nMaxMarkerLabelID,
                                                              unsigned int *hpBufferSize) {
  NppStatus status = validateCompressedMarkerLabelsInputs(nMaxMarkerLabelID, hpBufferSize);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R_impl(nMaxMarkerLabelID, hpBufferSize);
}

// Get compressed marker labels info
NppStatus nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(
    Npp32u *pCompressedMarkerLabels, Npp32s nCompressedMarkerLabelsStep, NppiSize oSizeROI,
    unsigned int nMaxMarkerLabelID, NppiCompressedMarkerLabelsInfo *pMarkerLabelsInfoList, Npp8u *pContoursImage,
    Npp32s nContoursImageStep, NppiContourPixelDirectionInfo *pContoursDirectionImage,
    Npp32s nContoursDirectionImageStep, NppiContourTotalsInfo *pContoursTotalsInfoHost,
    Npp32u *pContoursPixelCountsListDev, Npp32u *pContoursPixelCountsListHost, Npp32u *pContoursPixelStartingOffsetDev,
    Npp32u *pContoursPixelStartingOffsetHost, NppStreamContext nppStreamCtx) {

  NppStatus status = validateCompressedMarkerLabelsInfoInputs(pCompressedMarkerLabels, nCompressedMarkerLabelsStep,
                                                              oSizeROI, nMaxMarkerLabelID, pMarkerLabelsInfoList);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx_impl(
      pCompressedMarkerLabels, nCompressedMarkerLabelsStep, oSizeROI, nMaxMarkerLabelID, pMarkerLabelsInfoList,
      pContoursImage, nContoursImageStep, pContoursDirectionImage, nContoursDirectionImageStep, pContoursTotalsInfoHost,
      pContoursPixelCountsListDev, pContoursPixelCountsListHost, pContoursPixelStartingOffsetDev,
      pContoursPixelStartingOffsetHost, nppStreamCtx);
}

// Get geometry lists buffer size
NppStatus nppiCompressedMarkerLabelsUFGetGeometryListsSize_C1R(Npp32u nMaxContourPixelGeometryInfoCount,
                                                               Npp32u *hpBufferSize) {
  if (hpBufferSize == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  return nppiCompressedMarkerLabelsUFGetGeometryListsSize_C1R_impl(nMaxContourPixelGeometryInfoCount, hpBufferSize);
}

// Generate geometry lists
NppStatus nppiCompressedMarkerLabelsUFContoursGenerateGeometryLists_C1R_Ctx(
    NppiCompressedMarkerLabelsInfo *pMarkerLabelsInfoListDev, NppiCompressedMarkerLabelsInfo *pMarkerLabelsInfoListHost,
    NppiContourPixelDirectionInfo *pContoursDirectionImageDev, Npp32s nContoursDirectionImageStep,
    NppiContourPixelGeometryInfo *pContoursPixelGeometryListsDev,
    NppiContourPixelGeometryInfo *pContoursPixelGeometryListsHost, Npp8u *pContoursGeometryImageHost,
    Npp32s nContoursGeometryImageStep, Npp32u *pContoursPixelCountsListDev, Npp32u *pContoursPixelsFoundListDev,
    Npp32u *pContoursPixelsFoundListHost, Npp32u *pContoursPixelsStartingOffsetDev,
    Npp32u *pContoursPixelsStartingOffsetHost, Npp32u nTotalImagePixelContourCount, Npp32u nMaxMarkerLabelID,
    Npp32u nFirstContourGeometryListID, Npp32u nLastContourGeometryListID,
    NppiContourBlockSegment *pContoursBlockSegmentListDev, NppiContourBlockSegment *pContoursBlockSegmentListHost,
    Npp32u bOutputInCounterclockwiseOrder, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {

  if (pMarkerLabelsInfoListDev == nullptr || pMarkerLabelsInfoListHost == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (pContoursDirectionImageDev == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (nFirstContourGeometryListID >= nLastContourGeometryListID) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  return nppiCompressedMarkerLabelsUFContoursGenerateGeometryLists_C1R_Ctx_impl(
      pMarkerLabelsInfoListDev, pMarkerLabelsInfoListHost, pContoursDirectionImageDev, nContoursDirectionImageStep,
      pContoursPixelGeometryListsDev, pContoursPixelGeometryListsHost, pContoursGeometryImageHost,
      nContoursGeometryImageStep, pContoursPixelCountsListDev, pContoursPixelsFoundListDev,
      pContoursPixelsFoundListHost, pContoursPixelsStartingOffsetDev, pContoursPixelsStartingOffsetHost,
      nTotalImagePixelContourCount, nMaxMarkerLabelID, nFirstContourGeometryListID, nLastContourGeometryListID,
      pContoursBlockSegmentListDev, pContoursBlockSegmentListHost, bOutputInCounterclockwiseOrder, oSizeROI,
      nppStreamCtx);
}

// Batch compress marker labels (advanced)
NppStatus nppiCompressMarkerLabelsUFBatch_32u_C1IR_Advanced_Ctx(NppiImageDescriptor *pSrcDstBatchList,
                                                                NppiBufferDescriptor *pBufferList,
                                                                unsigned int *pNewMaxLabelIDList, int nBatchSize,
                                                                NppiSize oMaxSizeROI, int nLargestPerImageBufferSize,
                                                                NppStreamContext nppStreamCtx) {

  if (pSrcDstBatchList == nullptr || pBufferList == nullptr || pNewMaxLabelIDList == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nBatchSize <= 0) {
    return NPP_BAD_ARGUMENT_ERROR;
  }
  if (oMaxSizeROI.width <= 0 || oMaxSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (nLargestPerImageBufferSize <= 0) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  return nppiCompressMarkerLabelsUFBatch_32u_C1IR_Advanced_Ctx_impl(pSrcDstBatchList, pBufferList, pNewMaxLabelIDList,
                                                                    nBatchSize, oMaxSizeROI, nLargestPerImageBufferSize,
                                                                    nppStreamCtx);
}

// Non-context version
NppStatus nppiCompressMarkerLabelsUFBatch_32u_C1IR_Advanced(NppiImageDescriptor *pSrcDstBatchList,
                                                            NppiBufferDescriptor *pBufferList,
                                                            unsigned int *pNewMaxLabelIDList, int nBatchSize,
                                                            NppiSize oMaxSizeROI, int nLargestPerImageBufferSize) {

  NppStreamContext nppStreamCtx = nppCreateDefaultStreamContext();
  return nppiCompressMarkerLabelsUFBatch_32u_C1IR_Advanced_Ctx(pSrcDstBatchList, pBufferList, pNewMaxLabelIDList,
                                                               nBatchSize, oMaxSizeROI, nLargestPerImageBufferSize,
                                                               nppStreamCtx);
}
