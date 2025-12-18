#include "npp.h"
#include <algorithm>
#include <climits>
#include <cstring>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <map>
#include <vector>

// Contour direction constants (from nppdefs.h)
#define CONTOUR_DIR_SE 1
#define CONTOUR_DIR_S 2
#define CONTOUR_DIR_SW 4
#define CONTOUR_DIR_W 8
#define CONTOUR_DIR_E 16
#define CONTOUR_DIR_NE 32
#define CONTOUR_DIR_N 64
#define CONTOUR_DIR_NW 128

// Kernel to count pixels per label
__global__ void countPixelsPerLabel_kernel(const Npp32u *pMarkerLabels, int nStep, int width, int height,
                                           Npp32u *pPixelCounts, unsigned int nMaxLabelID) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const Npp32u *row = (const Npp32u *)((const char *)pMarkerLabels + y * nStep);
  Npp32u label = row[x];

  if (label > 0 && label <= nMaxLabelID) {
    atomicAdd(&pPixelCounts[label], 1);
  }
}

// Kernel to detect contour pixels
__global__ void detectContourPixels_kernel(const Npp32u *pMarkerLabels, int nStep, int width, int height,
                                           Npp8u *pContoursImage, int nContoursStep, Npp32u *pContourCounts,
                                           unsigned int nMaxLabelID) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const Npp32u *row = (const Npp32u *)((const char *)pMarkerLabels + y * nStep);
  Npp32u label = row[x];

  Npp8u *contourRow = pContoursImage ? (Npp8u *)((char *)pContoursImage + y * nContoursStep) : nullptr;

  if (label == 0) {
    if (contourRow)
      contourRow[x] = 0;
    return;
  }

  // Check if pixel is on contour (has neighbor with different label)
  bool isContour = false;
  int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1};
  int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1};

  for (int i = 0; i < 8; i++) {
    int nx = x + dx[i];
    int ny = y + dy[i];

    if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
      isContour = true;
      break;
    }

    const Npp32u *neighborRow = (const Npp32u *)((const char *)pMarkerLabels + ny * nStep);
    if (neighborRow[nx] != label) {
      isContour = true;
      break;
    }
  }

  if (contourRow) {
    contourRow[x] = isContour ? 255 : 0;
  }

  if (isContour && label <= nMaxLabelID) {
    atomicAdd(&pContourCounts[label], 1);
  }
}

// Kernel to compute contour direction info
__global__ void computeContourDirection_kernel(const Npp32u *pMarkerLabels, int nStep, int width, int height,
                                               NppiContourPixelDirectionInfo *pDirectionImage, int nDirectionStep,
                                               unsigned int nMaxLabelID) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const Npp32u *row = (const Npp32u *)((const char *)pMarkerLabels + y * nStep);
  Npp32u label = row[x];

  NppiContourPixelDirectionInfo *dirRow =
      (NppiContourPixelDirectionInfo *)((char *)pDirectionImage + y * nDirectionStep);
  NppiContourPixelDirectionInfo &info = dirRow[x];

  // Initialize
  info.nMarkerLabelID = label;
  info.nContourDirectionCenterPixel = 0;
  info.nContourInteriorDirectionCenterPixel = 0;
  info.nConnected = 0;
  info.nGeometryInfoIsValid = 0;

  if (label == 0)
    return;

  // Check all 8 neighbors and compute direction info
  Npp8u connected = 0;
  Npp8u interiorDir = 0;

  // Direction offsets: E, NE, N, NW, W, SW, S, SE
  int dx[] = {1, 1, 0, -1, -1, -1, 0, 1};
  int dy[] = {0, -1, -1, -1, 0, 1, 1, 1};
  Npp8u dirBits[] = {CONTOUR_DIR_E, CONTOUR_DIR_NE, CONTOUR_DIR_N, CONTOUR_DIR_NW,
                     CONTOUR_DIR_W, CONTOUR_DIR_SW, CONTOUR_DIR_S, CONTOUR_DIR_SE};

  for (int i = 0; i < 8; i++) {
    int nx = x + dx[i];
    int ny = y + dy[i];

    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
      const Npp32u *neighborRow = (const Npp32u *)((const char *)pMarkerLabels + ny * nStep);
      if (neighborRow[nx] == label) {
        connected |= dirBits[i];
      } else {
        // Direction to exterior
        interiorDir |= dirBits[(i + 4) % 8]; // Opposite direction points to interior
      }

      // Store neighbor locations
      switch (i) {
      case 0:
        info.nEast1.x = nx;
        info.nEast1.y = ny;
        info.nTest1EastConnected = (neighborRow[nx] == label);
        break;
      case 1:
        info.nNorthEast1.x = nx;
        info.nNorthEast1.y = ny;
        info.nTest1NorthEastConnected = (neighborRow[nx] == label);
        break;
      case 2:
        info.nNorth1.x = nx;
        info.nNorth1.y = ny;
        info.nTest1NorthConnected = (neighborRow[nx] == label);
        break;
      case 3:
        info.nNorthWest1.x = nx;
        info.nNorthWest1.y = ny;
        info.nTest1NorthWestConnected = (neighborRow[nx] == label);
        break;
      case 4:
        info.nWest1.x = nx;
        info.nWest1.y = ny;
        info.nTest1WestConnected = (neighborRow[nx] == label);
        break;
      case 5:
        info.nSouthWest1.x = nx;
        info.nSouthWest1.y = ny;
        info.nTest1SouthWestConnected = (neighborRow[nx] == label);
        break;
      case 6:
        info.nSouth1.x = nx;
        info.nSouth1.y = ny;
        info.nTest1SouthConnected = (neighborRow[nx] == label);
        break;
      case 7:
        info.nSouthEast1.x = nx;
        info.nSouthEast1.y = ny;
        info.nTest1SouthEastConnected = (neighborRow[nx] == label);
        break;
      }
    }
  }

  info.nConnected = connected;
  info.nContourInteriorDirectionCenterPixel = interiorDir;
}

// Kernel to update bounding box info for each label
__global__ void updateBoundingBox_kernel(const Npp32u *pMarkerLabels, int nStep, int width, int height, int *pMinX,
                                         int *pMinY, int *pMaxX, int *pMaxY, unsigned int nMaxLabelID) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const Npp32u *row = (const Npp32u *)((const char *)pMarkerLabels + y * nStep);
  Npp32u label = row[x];

  if (label == 0 || label > nMaxLabelID)
    return;

  // Update bounding box using atomic operations
  atomicMin(&pMinX[label], x);
  atomicMin(&pMinY[label], y);
  atomicMax(&pMaxX[label], x);
  atomicMax(&pMaxY[label], y);
}

// Kernel to find first contour pixel for each label
__global__ void findFirstContourPixel_kernel(const Npp8u *pContoursImage, int nContoursStep,
                                             const Npp32u *pMarkerLabels, int nStep, int width, int height,
                                             int *pFirstContourX, int *pFirstContourY, unsigned int nMaxLabelID) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const Npp8u *contourRow = (const Npp8u *)((const char *)pContoursImage + y * nContoursStep);
  if (contourRow[x] == 0)
    return; // Not a contour pixel

  const Npp32u *labelRow = (const Npp32u *)((const char *)pMarkerLabels + y * nStep);
  Npp32u label = labelRow[x];

  if (label == 0 || label > nMaxLabelID)
    return;

  // Encode position as y * width + x for comparison (topmost-leftmost wins)
  int encoded = y * width + x;
  atomicMin(&pFirstContourY[label], encoded); // Use encoded position to find topmost-leftmost
}

// Kernel for batch compression
__global__ void batchCompressLabels_kernel(Npp32u *pMarkerLabels, int nStep, int width, int height,
                                           Npp32u *pLabelMapping, unsigned int nMaxLabelID) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  Npp32u *row = (Npp32u *)((char *)pMarkerLabels + y * nStep);
  Npp32u label = row[x];

  if (label > 0 && label <= nMaxLabelID) {
    row[x] = pLabelMapping[label];
  }
}

extern "C" {

// Get info list buffer size
NppStatus nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R_impl(unsigned int nMaxMarkerLabelID,
                                                                   unsigned int *hpBufferSize) {
  // Need space for nMaxMarkerLabelID + 1 info structures (index 0 is unused)
  size_t size = (nMaxMarkerLabelID + 1) * sizeof(NppiCompressedMarkerLabelsInfo);
  *hpBufferSize = (unsigned int)((size + 255) & ~255); // Align to 256 bytes
  return NPP_SUCCESS;
}

// Get compressed marker labels info
NppStatus nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx_impl(
    Npp32u *pCompressedMarkerLabels, Npp32s nCompressedMarkerLabelsStep, NppiSize oSizeROI,
    unsigned int nMaxMarkerLabelID, NppiCompressedMarkerLabelsInfo *pMarkerLabelsInfoList, Npp8u *pContoursImage,
    Npp32s nContoursImageStep, NppiContourPixelDirectionInfo *pContoursDirectionImage,
    Npp32s nContoursDirectionImageStep, NppiContourTotalsInfo *pContoursTotalsInfoHost,
    Npp32u *pContoursPixelCountsListDev, Npp32u *pContoursPixelCountsListHost, Npp32u *pContoursPixelStartingOffsetDev,
    Npp32u *pContoursPixelStartingOffsetHost, NppStreamContext nppStreamCtx) {

  int width = oSizeROI.width;
  int height = oSizeROI.height;
  cudaStream_t stream = nppStreamCtx.hStream;

  // Clear any previous errors
  cudaGetLastError();

  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

  // Allocate temporary buffers for pixel and contour counts
  Npp32u *d_pixelCounts = nullptr;
  Npp32u *d_contourCounts = nullptr;
  size_t countSize = (nMaxMarkerLabelID + 1) * sizeof(Npp32u);

  cudaError_t err = cudaMalloc(&d_pixelCounts, countSize);
  if (err != cudaSuccess) {
    return NPP_MEMORY_ALLOCATION_ERR;
  }
  err = cudaMalloc(&d_contourCounts, countSize);
  if (err != cudaSuccess) {
    cudaFree(d_pixelCounts);
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  cudaMemsetAsync(d_pixelCounts, 0, countSize, stream);
  cudaMemsetAsync(d_contourCounts, 0, countSize, stream);

  // Allocate temporary buffers for bounding box and first contour pixel
  int *d_minX = nullptr, *d_minY = nullptr, *d_maxX = nullptr, *d_maxY = nullptr;
  int *d_firstContourPos = nullptr;
  size_t bboxSize = (nMaxMarkerLabelID + 1) * sizeof(int);

  err = cudaMalloc(&d_minX, bboxSize);
  if (err != cudaSuccess) {
    cudaFree(d_pixelCounts);
    cudaFree(d_contourCounts);
    return NPP_MEMORY_ALLOCATION_ERR;
  }
  err = cudaMalloc(&d_minY, bboxSize);
  if (err != cudaSuccess) {
    cudaFree(d_pixelCounts);
    cudaFree(d_contourCounts);
    cudaFree(d_minX);
    return NPP_MEMORY_ALLOCATION_ERR;
  }
  err = cudaMalloc(&d_maxX, bboxSize);
  if (err != cudaSuccess) {
    cudaFree(d_pixelCounts);
    cudaFree(d_contourCounts);
    cudaFree(d_minX);
    cudaFree(d_minY);
    return NPP_MEMORY_ALLOCATION_ERR;
  }
  err = cudaMalloc(&d_maxY, bboxSize);
  if (err != cudaSuccess) {
    cudaFree(d_pixelCounts);
    cudaFree(d_contourCounts);
    cudaFree(d_minX);
    cudaFree(d_minY);
    cudaFree(d_maxX);
    return NPP_MEMORY_ALLOCATION_ERR;
  }
  err = cudaMalloc(&d_firstContourPos, bboxSize);
  if (err != cudaSuccess) {
    cudaFree(d_pixelCounts);
    cudaFree(d_contourCounts);
    cudaFree(d_minX);
    cudaFree(d_minY);
    cudaFree(d_maxX);
    cudaFree(d_maxY);
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  // Initialize bounding box arrays with sentinel values
  std::vector<int> h_minInit(nMaxMarkerLabelID + 1, width);
  std::vector<int> h_maxInit(nMaxMarkerLabelID + 1, -1);
  std::vector<int> h_firstInit(nMaxMarkerLabelID + 1, INT_MAX);

  cudaMemcpyAsync(d_minX, h_minInit.data(), bboxSize, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_minY, h_minInit.data(), bboxSize, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_maxX, h_maxInit.data(), bboxSize, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_maxY, h_maxInit.data(), bboxSize, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_firstContourPos, h_firstInit.data(), bboxSize, cudaMemcpyHostToDevice, stream);

  // Helper lambda to cleanup and return error
  auto cleanupAndReturn = [&]() {
    cudaFree(d_pixelCounts);
    cudaFree(d_contourCounts);
    cudaFree(d_minX);
    cudaFree(d_minY);
    cudaFree(d_maxX);
    cudaFree(d_maxY);
    cudaFree(d_firstContourPos);
  };

  // Count pixels per label
  countPixelsPerLabel_kernel<<<gridSize, blockSize, 0, stream>>>(pCompressedMarkerLabels, nCompressedMarkerLabelsStep,
                                                                 width, height, d_pixelCounts, nMaxMarkerLabelID);

  // Detect contour pixels and count
  detectContourPixels_kernel<<<gridSize, blockSize, 0, stream>>>(pCompressedMarkerLabels, nCompressedMarkerLabelsStep,
                                                                 width, height, pContoursImage, nContoursImageStep,
                                                                 d_contourCounts, nMaxMarkerLabelID);

  // Compute contour direction info if requested
  if (pContoursDirectionImage != nullptr) {
    computeContourDirection_kernel<<<gridSize, blockSize, 0, stream>>>(
        pCompressedMarkerLabels, nCompressedMarkerLabelsStep, width, height, pContoursDirectionImage,
        nContoursDirectionImageStep, nMaxMarkerLabelID);
  }

  // Update bounding box
  updateBoundingBox_kernel<<<gridSize, blockSize, 0, stream>>>(pCompressedMarkerLabels, nCompressedMarkerLabelsStep,
                                                               width, height, d_minX, d_minY, d_maxX, d_maxY,
                                                               nMaxMarkerLabelID);

  // Find first contour pixel ( runs after contours image is filled)
  if (pContoursImage != nullptr) {
    findFirstContourPixel_kernel<<<gridSize, blockSize, 0, stream>>>(
        pContoursImage, nContoursImageStep, pCompressedMarkerLabels, nCompressedMarkerLabelsStep, width, height,
        nullptr, d_firstContourPos, nMaxMarkerLabelID);
  }

  // Sync and check for kernel errors
  cudaStreamSynchronize(stream);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    cleanupAndReturn();
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  // Copy counts from device to host
  std::vector<Npp32u> h_pixelCounts(nMaxMarkerLabelID + 1);
  std::vector<Npp32u> h_contourCounts(nMaxMarkerLabelID + 1);

  cudaMemcpy(h_pixelCounts.data(), d_pixelCounts, countSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_contourCounts.data(), d_contourCounts, countSize, cudaMemcpyDeviceToHost);

  // Copy bounding box and first contour pixel data from device
  std::vector<int> h_minX(nMaxMarkerLabelID + 1);
  std::vector<int> h_minY(nMaxMarkerLabelID + 1);
  std::vector<int> h_maxX(nMaxMarkerLabelID + 1);
  std::vector<int> h_maxY(nMaxMarkerLabelID + 1);
  std::vector<int> h_firstContourPos(nMaxMarkerLabelID + 1);

  cudaMemcpy(h_minX.data(), d_minX, bboxSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_minY.data(), d_minY, bboxSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_maxX.data(), d_maxX, bboxSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_maxY.data(), d_maxY, bboxSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_firstContourPos.data(), d_firstContourPos, bboxSize, cudaMemcpyDeviceToHost);

  // Build info list on host
  std::vector<NppiCompressedMarkerLabelsInfo> h_infoList(nMaxMarkerLabelID + 1);

  Npp32u totalContourPixels = 0;
  Npp32u longestContour = 0;
  Npp32u currentOffset = 0;

  for (unsigned int i = 0; i <= nMaxMarkerLabelID; i++) {
    h_infoList[i].nMarkerLabelPixelCount = h_pixelCounts[i];
    h_infoList[i].nContourPixelCount = h_contourCounts[i];
    h_infoList[i].nContourPixelsFound = 0;

    // Set bounding box
    // Note: NPP stores width/height as right/bottom coordinates (maxX/maxY), not actual dimensions
    if (h_minX[i] < width && h_maxX[i] >= 0) {
      h_infoList[i].oMarkerLabelBoundingBox.x = h_minX[i];
      h_infoList[i].oMarkerLabelBoundingBox.y = h_minY[i];
      h_infoList[i].oMarkerLabelBoundingBox.width = h_maxX[i];
      h_infoList[i].oMarkerLabelBoundingBox.height = h_maxY[i];
    } else {
      h_infoList[i].oMarkerLabelBoundingBox.x = 0;
      h_infoList[i].oMarkerLabelBoundingBox.y = 0;
      h_infoList[i].oMarkerLabelBoundingBox.width = 0;
      h_infoList[i].oMarkerLabelBoundingBox.height = 0;
    }

    // Decode first contour pixel position
    if (h_firstContourPos[i] < INT_MAX) {
      h_infoList[i].oContourFirstPixelLocation.y = h_firstContourPos[i] / width;
      h_infoList[i].oContourFirstPixelLocation.x = h_firstContourPos[i] % width;
    } else {
      h_infoList[i].oContourFirstPixelLocation.x = 0;
      h_infoList[i].oContourFirstPixelLocation.y = 0;
    }

    totalContourPixels += h_contourCounts[i];
    if (h_contourCounts[i] > longestContour) {
      longestContour = h_contourCounts[i];
    }

    // Set starting offset
    if (pContoursPixelStartingOffsetHost != nullptr) {
      pContoursPixelStartingOffsetHost[i] = currentOffset;
    }
    currentOffset += h_contourCounts[i];

    // Copy contour count to list
    if (pContoursPixelCountsListHost != nullptr) {
      pContoursPixelCountsListHost[i] = h_contourCounts[i];
    }
  }

  // Copy info list to device
  cudaMemcpy(pMarkerLabelsInfoList, h_infoList.data(), (nMaxMarkerLabelID + 1) * sizeof(NppiCompressedMarkerLabelsInfo),
             cudaMemcpyHostToDevice);

  // Copy counts and offsets to device if pointers provided
  if (pContoursPixelCountsListDev != nullptr) {
    cudaMemcpy(pContoursPixelCountsListDev, h_contourCounts.data(), countSize, cudaMemcpyHostToDevice);
  }
  if (pContoursPixelStartingOffsetDev != nullptr && pContoursPixelStartingOffsetHost != nullptr) {
    cudaMemcpy(pContoursPixelStartingOffsetDev, pContoursPixelStartingOffsetHost, countSize, cudaMemcpyHostToDevice);
  }

  // Fill totals info
  if (pContoursTotalsInfoHost != nullptr) {
    pContoursTotalsInfoHost->nTotalImagePixelContourCount = totalContourPixels;
    pContoursTotalsInfoHost->nLongestImageContourPixelCount = longestContour;
  }

  // Clean up
  cudaFree(d_pixelCounts);
  cudaFree(d_contourCounts);
  cudaFree(d_minX);
  cudaFree(d_minY);
  cudaFree(d_maxX);
  cudaFree(d_maxY);
  cudaFree(d_firstContourPos);

  return NPP_SUCCESS;
}

// Get geometry lists buffer size
NppStatus nppiCompressedMarkerLabelsUFGetGeometryListsSize_C1R_impl(Npp32u nMaxContourPixelGeometryInfoCount,
                                                                    Npp32u *hpBufferSize) {
  size_t size = nMaxContourPixelGeometryInfoCount * sizeof(NppiContourPixelGeometryInfo);
  *hpBufferSize = (Npp32u)((size + 255) & ~255);
  return NPP_SUCCESS;
}

// Generate geometry lists
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
    Npp32u bOutputInCounterclockwiseOrder, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {

  int width = oSizeROI.width;
  int height = oSizeROI.height;
  (void)nppStreamCtx; // Stream context reserved for future async support

  // Copy direction image to host for processing
  size_t dirImageSize = (size_t)width * height * sizeof(NppiContourPixelDirectionInfo);
  std::vector<NppiContourPixelDirectionInfo> h_dirImage(width * height);
  cudaMemcpy(h_dirImage.data(), pContoursDirectionImageDev, dirImageSize, cudaMemcpyDeviceToHost);

  // Process each contour in the requested range
  Npp32u geometryListOffset = 0;
  if (pContoursPixelsStartingOffsetHost != nullptr && nFirstContourGeometryListID > 0) {
    geometryListOffset = pContoursPixelsStartingOffsetHost[nFirstContourGeometryListID];
  }

  for (Npp32u labelID = nFirstContourGeometryListID;
       labelID < nLastContourGeometryListID && labelID <= nMaxMarkerLabelID; labelID++) {
    if (pMarkerLabelsInfoListHost == nullptr)
      continue;

    NppiCompressedMarkerLabelsInfo &info = pMarkerLabelsInfoListHost[labelID];
    if (info.nContourPixelCount == 0)
      continue;

    // Skip very large contours by default (performance optimization)
    if (info.nContourPixelCount > 256 * 1024 && nLastContourGeometryListID - nFirstContourGeometryListID > 1) {
      continue;
    }

    // Find starting pixel
    int startX = info.oContourFirstPixelLocation.x;
    int startY = info.oContourFirstPixelLocation.y;

    if (startX >= width || startY >= height)
      continue;

    // Trace contour
    int x = startX;
    int y = startY;
    Npp32u pixelCount = 0;
    Npp32u listOffset = geometryListOffset;

    // Direction to search next (start going clockwise from east)
    int searchDir = 0; // Start searching from east

    do {
      NppiContourPixelDirectionInfo &dirInfo = h_dirImage[y * width + x];
      if (dirInfo.nMarkerLabelID != labelID)
        break;

      // Store geometry info
      if (pContoursPixelGeometryListsHost != nullptr && listOffset < nTotalImagePixelContourCount) {
        NppiContourPixelGeometryInfo &geom = pContoursPixelGeometryListsHost[listOffset];
        geom.oContourCenterPixelLocation.x = x;
        geom.oContourCenterPixelLocation.y = y;
        geom.nOrderIndex = bOutputInCounterclockwiseOrder ? pixelCount : (info.nContourPixelCount - 1 - pixelCount);
        geom.nReverseOrderIndex =
            bOutputInCounterclockwiseOrder ? (info.nContourPixelCount - 1 - pixelCount) : pixelCount;

        // Store bounding box in first pixel
        if (pixelCount == 0) {
          geom.oContourPrevPixelLocation.x = info.oMarkerLabelBoundingBox.x;
          geom.oContourPrevPixelLocation.y = info.oMarkerLabelBoundingBox.y;
          geom.oContourNextPixelLocation.x = info.oMarkerLabelBoundingBox.width;
          geom.oContourNextPixelLocation.y = info.oMarkerLabelBoundingBox.height;
        }

        listOffset++;
      }

      // Mark pixel as visited
      dirInfo.nGeometryInfoIsValid = 1;
      pixelCount++;

      // Find next contour pixel (search in order based on direction)
      int dx[] = {1, 1, 0, -1, -1, -1, 0, 1};
      int dy[] = {0, 1, 1, 1, 0, -1, -1, -1};

      bool found = false;
      int startSearch = bOutputInCounterclockwiseOrder ? (searchDir + 6) % 8 : (searchDir + 2) % 8;

      for (int i = 0; i < 8; i++) {
        int dir = bOutputInCounterclockwiseOrder ? (startSearch + i) % 8 : (startSearch + 8 - i) % 8;
        int nx = x + dx[dir];
        int ny = y + dy[dir];

        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
          NppiContourPixelDirectionInfo &nextDir = h_dirImage[ny * width + nx];
          if (nextDir.nMarkerLabelID == labelID && nextDir.nGeometryInfoIsValid == 0) {
            // Check if this is a contour pixel
            bool isContour = false;
            for (int j = 0; j < 8; j++) {
              int nnx = nx + dx[j];
              int nny = ny + dy[j];
              if (nnx < 0 || nnx >= width || nny < 0 || nny >= height) {
                isContour = true;
                break;
              }
              if (h_dirImage[nny * width + nnx].nMarkerLabelID != labelID) {
                isContour = true;
                break;
              }
            }

            if (isContour) {
              x = nx;
              y = ny;
              searchDir = (dir + 4) % 8; // Reverse direction for next search
              found = true;
              break;
            }
          }
        }
      }

      if (!found)
        break;

    } while (x != startX || y != startY);

    info.nContourPixelsFound = pixelCount;
    geometryListOffset = listOffset;

    // Update found list
    if (pContoursPixelsFoundListHost != nullptr) {
      pContoursPixelsFoundListHost[labelID] = pixelCount;
    }
  }

  // Copy geometry lists to device
  if (pContoursPixelGeometryListsDev != nullptr && pContoursPixelGeometryListsHost != nullptr) {
    size_t copySize = nTotalImagePixelContourCount * sizeof(NppiContourPixelGeometryInfo);
    cudaMemcpy(pContoursPixelGeometryListsDev, pContoursPixelGeometryListsHost, copySize, cudaMemcpyHostToDevice);
  }

  // Copy found list to device
  if (pContoursPixelsFoundListDev != nullptr && pContoursPixelsFoundListHost != nullptr) {
    cudaMemcpy(pContoursPixelsFoundListDev, pContoursPixelsFoundListHost, (nMaxMarkerLabelID + 1) * sizeof(Npp32u),
               cudaMemcpyHostToDevice);
  }

  // Copy updated info list to device
  if (pMarkerLabelsInfoListDev != nullptr && pMarkerLabelsInfoListHost != nullptr) {
    cudaMemcpy(pMarkerLabelsInfoListDev, pMarkerLabelsInfoListHost,
               (nMaxMarkerLabelID + 1) * sizeof(NppiCompressedMarkerLabelsInfo), cudaMemcpyHostToDevice);
  }

  return NPP_SUCCESS;
}

// Batch compress marker labels (advanced)
NppStatus nppiCompressMarkerLabelsUFBatch_32u_C1IR_Advanced_Ctx_impl(
    NppiImageDescriptor *pSrcDstBatchListDev, NppiBufferDescriptor *pBufferListDev, unsigned int *pNewMaxLabelIDListDev,
    int nBatchSize, NppiSize oMaxSizeROI, int nLargestPerImageBufferSize, NppStreamContext nppStreamCtx) {

  cudaStream_t stream = nppStreamCtx.hStream;
  (void)nLargestPerImageBufferSize; // Not used in this implementation
  (void)oMaxSizeROI; // Not used in this implementation

  // Copy descriptor lists from device to host
  std::vector<NppiImageDescriptor> h_imgList(nBatchSize);
  std::vector<NppiBufferDescriptor> h_bufList(nBatchSize);
  std::vector<unsigned int> h_newMaxLabelIDList(nBatchSize);

  cudaMemcpy(h_imgList.data(), pSrcDstBatchListDev, nBatchSize * sizeof(NppiImageDescriptor), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_bufList.data(), pBufferListDev, nBatchSize * sizeof(NppiBufferDescriptor), cudaMemcpyDeviceToHost);

  for (int batch = 0; batch < nBatchSize; batch++) {
    NppiImageDescriptor &img = h_imgList[batch];
    NppiBufferDescriptor &buf = h_bufList[batch];

    int width = img.oSize.width;
    int height = img.oSize.height;
    int numPixels = width * height;
    Npp32u *pMarkerLabels = (Npp32u *)img.pData;
    int nStep = img.nStep;

    // Max label ID is width * height (from nppiLabelMarkersUF output)
    unsigned int nMaxMarkerLabelID = (unsigned int)numPixels;

    // Use provided buffer for label mapping on device
    Npp32u *pLabelMapping = (Npp32u *)buf.pData;

    // Read image to host
    std::vector<Npp32u> h_image(numPixels);
    cudaMemcpy(h_image.data(), pMarkerLabels, numPixels * sizeof(Npp32u), cudaMemcpyDeviceToHost);

    // Use a map for sparse label tracking (more memory efficient)
    std::map<Npp32u, Npp32u> labelMap;

    // Collect unique labels (including label 0, which is valid)
    for (int i = 0; i < numPixels; i++) {
      Npp32u label = h_image[i];
      if (label <= nMaxMarkerLabelID) {
        labelMap[label] = 0; // Mark as seen
      }
    }

    // Assign compressed labels
    unsigned int newLabelID = 1;
    for (auto &kv : labelMap) {
      kv.second = newLabelID++;
    }

    h_newMaxLabelIDList[batch] = newLabelID - 1;

    // Create full mapping array for GPU (initialize to 0)
    std::vector<Npp32u> h_mapping(nMaxMarkerLabelID + 1, 0);
    for (const auto &kv : labelMap) {
      h_mapping[kv.first] = kv.second;
    }

    // Copy mapping to device
    cudaMemcpyAsync(pLabelMapping, h_mapping.data(), (nMaxMarkerLabelID + 1) * sizeof(Npp32u),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    // Apply mapping
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    batchCompressLabels_kernel<<<gridSize, blockSize, 0, stream>>>(pMarkerLabels, nStep, width, height, pLabelMapping,
                                                                   nMaxMarkerLabelID);
    cudaStreamSynchronize(stream);
  }

  // Copy results back to device
  cudaMemcpy(pNewMaxLabelIDListDev, h_newMaxLabelIDList.data(), nBatchSize * sizeof(unsigned int), cudaMemcpyHostToDevice);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

} // extern "C"
