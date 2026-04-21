#include "../../npp_internal.h"
#include "npp.h"
#include <cuda_runtime.h>
#include <vector>

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

NppStatus nppiCompressedMarkerLabelsUFGetContoursBlockSegmentListSize_C1R_impl(
    Npp32u *pContoursPixelCountsListHost, Npp32u nTotalImagePixelContourCount, Npp32u nCompressedLabelCount,
    Npp32u nFirstContourGeometryListID, Npp32u nLastContourGeometryListID, Npp32u *hpBufferSize);

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

static inline NppStatus validateCompressedMarkerLabelsInfoInputs(
    Npp32u *pCompressedMarkerLabels, Npp32s nCompressedMarkerLabelsStep, NppiSize oSizeROI,
    unsigned int nMaxMarkerLabelID, NppiCompressedMarkerLabelsInfo *pMarkerLabelsInfoList, Npp8u *pContoursImage,
    Npp32s nContoursImageStep, NppiContourPixelDirectionInfo *pContoursDirectionImage,
    Npp32s nContoursDirectionImageStep, NppiContourTotalsInfo *pContoursTotalsInfoHost,
    Npp32u *pContoursPixelCountsListDev, Npp32u *pContoursPixelCountsListHost,
    Npp32u *pContoursPixelStartingOffsetHost) {
  if (pCompressedMarkerLabels == nullptr || pMarkerLabelsInfoList == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (nCompressedMarkerLabelsStep <= 0) {
    return NPP_STEP_ERROR;
  }
  if (pContoursImage != nullptr && nContoursImageStep <= 0) {
    return NPP_STEP_ERROR;
  }
  if (pContoursDirectionImage != nullptr && nContoursDirectionImageStep <= 0) {
    return NPP_STEP_ERROR;
  }
  // When direction-image output is requested, this implementation mirrors NVIDIA's
  // contract by requiring the host/device contour metadata outputs together.
  // These arrays are consumed by downstream geometry-list generation paths.
  if (pContoursDirectionImage != nullptr &&
      (pContoursTotalsInfoHost == nullptr || pContoursPixelCountsListDev == nullptr ||
       pContoursPixelCountsListHost == nullptr || pContoursPixelStartingOffsetHost == nullptr)) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nMaxMarkerLabelID == 0) {
    return NPP_BAD_ARGUMENT_ERROR;
  }
  return NPP_SUCCESS;
}

static inline NppStatus validateCompressedMarkerLabelsGeometryInputs(
    NppiCompressedMarkerLabelsInfo *pMarkerLabelsInfoListDev, NppiCompressedMarkerLabelsInfo *pMarkerLabelsInfoListHost,
    NppiContourPixelDirectionInfo *pContoursDirectionImageDev, Npp32s nContoursDirectionImageStep, NppiSize oSizeROI,
    Npp32u nFirstContourGeometryListID, Npp32u nLastContourGeometryListID) {
  if (pMarkerLabelsInfoListDev == nullptr || pMarkerLabelsInfoListHost == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  // NVIDIA returns BAD_ARGUMENT (not NULL_POINTER) when the direction image is
  // omitted for geometry generation.
  if (pContoursDirectionImageDev == nullptr) {
    return NPP_BAD_ARGUMENT_ERROR;
  }
  if (nContoursDirectionImageStep <= 0) {
    return NPP_STEP_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (nFirstContourGeometryListID >= nLastContourGeometryListID) {
    return NPP_BAD_ARGUMENT_ERROR;
  }
  return NPP_SUCCESS;
}

template <typename T> struct ScopedCudaBuffer {
  T *ptr = nullptr;

  ~ScopedCudaBuffer() {
    if (ptr != nullptr) {
      cudaFree(ptr);
    }
  }

  T *get() const { return ptr; }
  T **out() { return &ptr; }
};

static inline Npp32u deriveTotalImagePixelContourCount(const Npp32u *pContoursPixelsStartingOffsetHost,
                                                       const Npp32u *pContoursPixelCountsListHost,
                                                       Npp32u nMaxMarkerLabelID) {
  // Preferred path: derive total via "last offset + last count".
  if (pContoursPixelsStartingOffsetHost != nullptr && pContoursPixelCountsListHost != nullptr) {
    return pContoursPixelsStartingOffsetHost[nMaxMarkerLabelID] + pContoursPixelCountsListHost[nMaxMarkerLabelID];
  }

  // Fallback for legacy call patterns that only provide contour counts.
  Npp32u nTotalImagePixelContourCount = 0;
  if (pContoursPixelCountsListHost != nullptr) {
    for (Npp32u label = 0; label <= nMaxMarkerLabelID; ++label) {
      nTotalImagePixelContourCount += pContoursPixelCountsListHost[label];
    }
  }
  return nTotalImagePixelContourCount;
}

template <typename T> static inline NppStatus allocateDeviceArray(ScopedCudaBuffer<T> &buffer, size_t count) {
  if (count == 0) {
    return NPP_SUCCESS;
  }
  if (cudaMalloc(reinterpret_cast<void **>(buffer.out()), count * sizeof(T)) != cudaSuccess) {
    return NPP_MEMORY_ALLOCATION_ERR;
  }
  return NPP_SUCCESS;
}

template <typename T>
static inline NppStatus allocateAndCopyHostArrayToDevice(ScopedCudaBuffer<T> &buffer, const T *host, size_t count) {
  if (host == nullptr || count == 0) {
    return NPP_SUCCESS;
  }

  NppStatus status = allocateDeviceArray(buffer, count);
  if (status != NPP_SUCCESS) {
    return status;
  }

  if (cudaMemcpy(buffer.get(), host, count * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
  return NPP_SUCCESS;
}

template <typename T> static inline NppStatus allocateAndZeroDeviceArray(ScopedCudaBuffer<T> &buffer, size_t count) {
  if (count == 0) {
    return NPP_SUCCESS;
  }

  NppStatus status = allocateDeviceArray(buffer, count);
  if (status != NPP_SUCCESS) {
    return status;
  }

  if (cudaMemset(buffer.get(), 0, count * sizeof(T)) != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
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

  NppStatus status = validateCompressedMarkerLabelsInfoInputs(
      pCompressedMarkerLabels, nCompressedMarkerLabelsStep, oSizeROI, nMaxMarkerLabelID, pMarkerLabelsInfoList,
      pContoursImage, nContoursImageStep, pContoursDirectionImage, nContoursDirectionImageStep, pContoursTotalsInfoHost,
      pContoursPixelCountsListDev, pContoursPixelCountsListHost, pContoursPixelStartingOffsetHost);
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

NppStatus nppiCompressedMarkerLabelsUFGetContoursBlockSegmentListSize_C1R(
    Npp32u *pContoursPixelCountsListHost, Npp32u nTotalImagePixelContourCount, Npp32u nCompressedLabelCount,
    Npp32u nFirstContourGeometryListID, Npp32u nLastContourGeometryListID, Npp32u *hpBufferSize) {
  if (pContoursPixelCountsListHost == nullptr || hpBufferSize == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nFirstContourGeometryListID >= nLastContourGeometryListID || nLastContourGeometryListID > nCompressedLabelCount) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  return nppiCompressedMarkerLabelsUFGetContoursBlockSegmentListSize_C1R_impl(
      pContoursPixelCountsListHost, nTotalImagePixelContourCount, nCompressedLabelCount, nFirstContourGeometryListID,
      nLastContourGeometryListID, hpBufferSize);
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

  NppStatus status = validateCompressedMarkerLabelsGeometryInputs(
      pMarkerLabelsInfoListDev, pMarkerLabelsInfoListHost, pContoursDirectionImageDev, nContoursDirectionImageStep,
      oSizeROI, nFirstContourGeometryListID, nLastContourGeometryListID);
  if (status != NPP_SUCCESS) {
    return status;
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
  if (nBatchSize <= 1) {
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

NppStatus nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(
    Npp32u *pCompressedMarkerLabels, Npp32s nCompressedMarkerLabelsStep, NppiSize oSizeROI,
    unsigned int nMaxMarkerLabelID, NppiCompressedMarkerLabelsInfo *pMarkerLabelsInfoList, Npp8u *pContoursImage,
    Npp32s nContoursImageStep, NppiContourPixelDirectionInfo *pContoursDirectionImage,
    Npp32s nContoursDirectionImageStep, NppiContourTotalsInfo *pContoursTotalsInfoHost,
    Npp32u *pContoursPixelCountsListDev, Npp32u *pContoursPixelCountsListHost, Npp32u *pContoursPixelStartingOffsetHost,
    NppStreamContext nppStreamCtx) {
  // Legacy overload: no device-side starting-offset list is exposed, so pass null
  // and keep host offsets as the source of truth.
  return nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(
      pCompressedMarkerLabels, nCompressedMarkerLabelsStep, oSizeROI, nMaxMarkerLabelID, pMarkerLabelsInfoList,
      pContoursImage, nContoursImageStep, pContoursDirectionImage, nContoursDirectionImageStep, pContoursTotalsInfoHost,
      pContoursPixelCountsListDev, pContoursPixelCountsListHost, nullptr, pContoursPixelStartingOffsetHost,
      nppStreamCtx);
}

NppStatus nppiCompressedMarkerLabelsUFContoursGenerateGeometryLists_C1R_Ctx(
    NppiCompressedMarkerLabelsInfo *pMarkerLabelsInfoListDev, NppiCompressedMarkerLabelsInfo *pMarkerLabelsInfoListHost,
    NppiContourPixelDirectionInfo *pContoursDirectionImageDev, Npp32s nContoursDirectionImageStep,
    NppiContourPixelGeometryInfo *pContoursPixelGeometryListsHost, Npp32u *pContoursPixelCountsListHost,
    Npp32u *pContoursPixelsFoundListHost, Npp32u *pContoursPixelsStartingOffsetHost, Npp32u nMaxMarkerLabelID,
    Npp32u nFirstContourGeometryListID, Npp32u nLastContourGeometryListID, NppiSize oSizeROI,
    NppStreamContext nppStreamCtx) {
  if (pContoursDirectionImageDev == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }

  // Legacy overload accepts host-side count/offset arrays. Reconstruct temporary
  // device buffers so the full implementation can run without code duplication.
  const Npp32u nTotalImagePixelContourCount = deriveTotalImagePixelContourCount(
      pContoursPixelsStartingOffsetHost, pContoursPixelCountsListHost, nMaxMarkerLabelID);

  ScopedCudaBuffer<NppiContourPixelGeometryInfo> pContoursPixelGeometryListsDev;
  ScopedCudaBuffer<Npp32u> pContoursPixelCountsListDev;
  ScopedCudaBuffer<Npp32u> pContoursPixelsFoundListDev;
  ScopedCudaBuffer<Npp32u> pContoursPixelsStartingOffsetDev;
  ScopedCudaBuffer<NppiContourBlockSegment> pContoursBlockSegmentListDev;
  std::vector<unsigned char> contoursBlockSegmentListHostStorage;
  NppiContourBlockSegment *pContoursBlockSegmentListHost = nullptr;

  NppStatus status =
      allocateDeviceArray(pContoursPixelGeometryListsDev, static_cast<size_t>(nTotalImagePixelContourCount));
  if (status != NPP_SUCCESS) {
    return status;
  }

  status = allocateAndCopyHostArrayToDevice(pContoursPixelCountsListDev, pContoursPixelCountsListHost,
                                            static_cast<size_t>(nMaxMarkerLabelID + 1));
  if (status != NPP_SUCCESS) {
    return status;
  }

  if (pContoursPixelsFoundListHost != nullptr) {
    status = allocateAndZeroDeviceArray(pContoursPixelsFoundListDev, static_cast<size_t>(nMaxMarkerLabelID + 1));
    if (status != NPP_SUCCESS) {
      return status;
    }
  }

  status = allocateAndCopyHostArrayToDevice(pContoursPixelsStartingOffsetDev, pContoursPixelsStartingOffsetHost,
                                            static_cast<size_t>(nMaxMarkerLabelID + 1));
  if (status != NPP_SUCCESS) {
    return status;
  }

  if (pContoursPixelCountsListHost != nullptr) {
    Npp32u contoursBlockSegmentListSize = 0;
    NppStatus blockSizeStatus = nppiCompressedMarkerLabelsUFGetContoursBlockSegmentListSize_C1R(
        pContoursPixelCountsListHost, nTotalImagePixelContourCount, nMaxMarkerLabelID + 1, nFirstContourGeometryListID,
        nLastContourGeometryListID, &contoursBlockSegmentListSize);
    if (blockSizeStatus != NPP_SUCCESS) {
      return blockSizeStatus;
    }
    if (contoursBlockSegmentListSize > 0) {
      contoursBlockSegmentListHostStorage.resize(contoursBlockSegmentListSize);
      pContoursBlockSegmentListHost =
          reinterpret_cast<NppiContourBlockSegment *>(contoursBlockSegmentListHostStorage.data());
      status = allocateDeviceArray(pContoursBlockSegmentListDev,
                                   contoursBlockSegmentListSize / sizeof(NppiContourBlockSegment));
      if (status != NPP_SUCCESS) {
        return status;
      }
    }
  }

  return nppiCompressedMarkerLabelsUFContoursGenerateGeometryLists_C1R_Ctx(
      pMarkerLabelsInfoListDev, pMarkerLabelsInfoListHost, pContoursDirectionImageDev, nContoursDirectionImageStep,
      pContoursPixelGeometryListsDev.get(), pContoursPixelGeometryListsHost, nullptr, 0,
      pContoursPixelCountsListDev.get(), pContoursPixelsFoundListDev.get(), pContoursPixelsFoundListHost,
      pContoursPixelsStartingOffsetDev.get(), pContoursPixelsStartingOffsetHost, nTotalImagePixelContourCount,
      nMaxMarkerLabelID, nFirstContourGeometryListID, nLastContourGeometryListID, pContoursBlockSegmentListDev.get(),
      pContoursBlockSegmentListHost, 0, oSizeROI, nppStreamCtx);
}
