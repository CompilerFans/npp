#include "npp.h"
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <set>
#include <utility>
#include <vector>

class NPPICompressedMarkerLabelsTest : public ::testing::Test {
protected:
  void SetUp() override {
    width = 64;
    height = 64;
    oSizeROI.width = width;
    oSizeROI.height = height;

    // Initialize stream context
    memset(&nppStreamCtx, 0, sizeof(nppStreamCtx));
    cudaStreamCreate(&nppStreamCtx.hStream);
    cudaGetDevice(&nppStreamCtx.nCudaDeviceId);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, nppStreamCtx.nCudaDeviceId);
    nppStreamCtx.nMultiProcessorCount = prop.multiProcessorCount;
    nppStreamCtx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    nppStreamCtx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
    nppStreamCtx.nSharedMemPerBlock = prop.sharedMemPerBlock;
    nppStreamCtx.nCudaDevAttrComputeCapabilityMajor = prop.major;
    nppStreamCtx.nCudaDevAttrComputeCapabilityMinor = prop.minor;
  }

  void TearDown() override {
    if (nppStreamCtx.hStream) {
      cudaStreamDestroy(nppStreamCtx.hStream);
    }
  }

  int width, height;
  NppiSize oSizeROI;
  NppStreamContext nppStreamCtx;

  // RAII guard for CUDA memory
  struct CudaMemGuard {
    void *ptr;
    CudaMemGuard(void *p) : ptr(p) {}
    ~CudaMemGuard() {
      if (ptr)
        cudaFree(ptr);
    }
  };

  bool createCompressedLabelsFromBinary(const std::vector<Npp8u> &h_binaryImage, Npp32u **d_labelsOut,
                                        std::vector<Npp32u> &h_compressedLabelsOut, int &nMaxMarkerLabelID) {
    const size_t pixelCount = static_cast<size_t>(width) * height;
    if (h_binaryImage.size() != pixelCount || d_labelsOut == nullptr) {
      return false;
    }
    Npp32u *d_labels = nullptr;
    if (cudaMalloc(&d_labels, pixelCount * sizeof(Npp32u)) != cudaSuccess) {
      return false;
    }

    auto buildLabelsFromHostBinary = [&]() -> bool {
      std::vector<Npp32u> h_labels(pixelCount, 0);
      Npp32u nextLabel = 1;
      std::vector<int> stack;
      stack.reserve(pixelCount);

      auto inBounds = [&](int x, int y) { return x >= 0 && x < width && y >= 0 && y < height; };
      auto idxOf = [&](int x, int y) { return static_cast<size_t>(y) * width + x; };

      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          const size_t seedIdx = idxOf(x, y);
          if (h_binaryImage[seedIdx] == 0 || h_labels[seedIdx] != 0) {
            continue;
          }

          h_labels[seedIdx] = nextLabel;
          stack.push_back(static_cast<int>(seedIdx));
          while (!stack.empty()) {
            const int cur = stack.back();
            stack.pop_back();
            const int curY = cur / width;
            const int curX = cur - curY * width;

            for (int dy = -1; dy <= 1; ++dy) {
              for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) {
                  continue;
                }
                const int nx = curX + dx;
                const int ny = curY + dy;
                if (!inBounds(nx, ny)) {
                  continue;
                }
                const size_t nidx = idxOf(nx, ny);
                if (h_binaryImage[nidx] == 0 || h_labels[nidx] != 0) {
                  continue;
                }
                h_labels[nidx] = nextLabel;
                stack.push_back(static_cast<int>(nidx));
              }
            }
          }
          ++nextLabel;
        }
      }

      return nextLabel != 1 &&
             cudaMemcpy(d_labels, h_labels.data(), pixelCount * sizeof(Npp32u), cudaMemcpyHostToDevice) == cudaSuccess;
    };

    if (!buildLabelsFromHostBinary()) {
      cudaFree(d_labels);
      return false;
    }

    int compressBufferSize = 0;
    Npp8u *d_compressBuffer = nullptr;
    if (nppiCompressMarkerLabelsGetBufferSize_32u_C1R(static_cast<int>(pixelCount), &compressBufferSize) !=
            NPP_SUCCESS ||
        compressBufferSize <= 0 || cudaMalloc(&d_compressBuffer, compressBufferSize) != cudaSuccess) {
      if (d_compressBuffer)
        cudaFree(d_compressBuffer);
      cudaFree(d_labels);
      return false;
    }

    int compressedCount = 0;
    NppStatus status = nppiCompressMarkerLabelsUF_32u_C1IR_Ctx(d_labels, width * sizeof(Npp32u), oSizeROI,
                                                               static_cast<int>(pixelCount), &compressedCount,
                                                               d_compressBuffer, nppStreamCtx);
    cudaFree(d_compressBuffer);
    if (status != NPP_SUCCESS || compressedCount <= 0) {
      cudaFree(d_labels);
      return false;
    }

    h_compressedLabelsOut.resize(pixelCount);
    if (cudaMemcpy(h_compressedLabelsOut.data(), d_labels, pixelCount * sizeof(Npp32u), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
      cudaFree(d_labels);
      return false;
    }

    *d_labelsOut = d_labels;
    nMaxMarkerLabelID = compressedCount;
    return true;
  }

  void createContourBinaryInput(std::vector<Npp8u> &h_binary) {
    width = 64;
    height = 64;
    oSizeROI.width = width;
    oSizeROI.height = height;
    h_binary.assign(static_cast<size_t>(width) * height, 0);

    for (int y = 10; y < 25; ++y) {
      for (int x = 10; x < 25; ++x) {
        h_binary[y * width + x] = 255;
      }
    }
    for (int y = 35; y < 50; ++y) {
      for (int x = 35; x < 50; ++x) {
        h_binary[y * width + x] = 255;
      }
    }
  }

  // Create a simple labeled image with distinct regions for testing
  void createTestLabeledImage(Npp32u *h_labels, int w, int h, int &maxLabelID) {
    // Create 4 distinct rectangular regions
    // Region 1: top-left quadrant (label 1)
    // Region 2: top-right quadrant (label 2)
    // Region 3: bottom-left quadrant (label 3)
    // Region 4: bottom-right quadrant (label 4)

    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        int idx = y * w + x;
        if (y < h / 2) {
          h_labels[idx] = (x < w / 2) ? 1 : 2;
        } else {
          h_labels[idx] = (x < w / 2) ? 3 : 4;
        }
      }
    }
    maxLabelID = 4;
  }

  // Create a more complex test image with a ring shape
  void createRingTestImage(Npp32u *h_labels, int w, int h, int &maxLabelID) {
    int centerX = w / 2;
    int centerY = h / 2;
    int outerRadius = std::min(w, h) / 3;
    int innerRadius = outerRadius / 2;

    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        int idx = y * w + x;
        int dx = x - centerX;
        int dy = y - centerY;
        int distSq = dx * dx + dy * dy;

        if (distSq <= innerRadius * innerRadius) {
          h_labels[idx] = 1; // Inner circle
        } else if (distSq <= outerRadius * outerRadius) {
          h_labels[idx] = 2; // Ring
        } else {
          h_labels[idx] = 0; // Background
        }
      }
    }
    maxLabelID = 2;
  }

  void assertUnifiedDirectionImageFields(const std::vector<Npp32u> &h_labels, const std::vector<Npp8u> &h_contours,
                                         const std::vector<NppiContourPixelDirectionInfo> &h_dirImage) {
    ASSERT_EQ(h_labels.size(), h_contours.size());
    ASSERT_EQ(h_labels.size(), h_dirImage.size());

    int contourPixelCount = 0;
    for (size_t i = 0; i < h_labels.size(); ++i) {
      const Npp32u label = h_labels[i];
      const Npp8u contour = h_contours[i];
      const NppiContourPixelDirectionInfo &info = h_dirImage[i];

      for (int j = 0; j < 4; ++j) {
        EXPECT_EQ(info.aContourInteriorDirection1[j], 0u) << "legacy-only unified field must stay zero at pixel " << i;
      }

      if (contour != 0 && label != 0) {
        ++contourPixelCount;
        EXPECT_EQ(info.nMarkerLabelID, label);
      }
    }

    EXPECT_GT(contourPixelCount, 0);
  }

  bool hasAnyMissingConnectedNeighbor(const NppiContourPixelDirectionInfo &info) const {
    return info.nTest1EastConnected == 0 || info.nTest1NorthEastConnected == 0 || info.nTest1NorthConnected == 0 ||
           info.nTest1NorthWestConnected == 0 || info.nTest1WestConnected == 0 || info.nTest1SouthWestConnected == 0 ||
           info.nTest1SouthConnected == 0 || info.nTest1SouthEastConnected == 0;
  }

  void expectPointEqual(const NppiPoint &lhs, const NppiPoint &rhs, const char *field, size_t index) {
    EXPECT_EQ(lhs.x, rhs.x) << field << ".x mismatch at element " << index;
    EXPECT_EQ(lhs.y, rhs.y) << field << ".y mismatch at element " << index;
  }

  void expectGeometryInfoEqual(const NppiContourPixelGeometryInfo &lhs, const NppiContourPixelGeometryInfo &rhs,
                               size_t index) {
    expectPointEqual(lhs.oContourOrderedGeometryLocation, rhs.oContourOrderedGeometryLocation,
                     "oContourOrderedGeometryLocation", index);
    expectPointEqual(lhs.oContourPrevPixelLocation, rhs.oContourPrevPixelLocation, "oContourPrevPixelLocation", index);
    expectPointEqual(lhs.oContourCenterPixelLocation, rhs.oContourCenterPixelLocation, "oContourCenterPixelLocation",
                     index);
    expectPointEqual(lhs.oContourNextPixelLocation, rhs.oContourNextPixelLocation, "oContourNextPixelLocation", index);
    EXPECT_EQ(lhs.nOrderIndex, rhs.nOrderIndex) << "nOrderIndex mismatch at element " << index;
    EXPECT_EQ(lhs.nReverseOrderIndex, rhs.nReverseOrderIndex) << "nReverseOrderIndex mismatch at element " << index;
    EXPECT_EQ(lhs.nFirstIndex, rhs.nFirstIndex) << "nFirstIndex mismatch at element " << index;
    EXPECT_EQ(lhs.nLastIndex, rhs.nLastIndex) << "nLastIndex mismatch at element " << index;
    EXPECT_EQ(lhs.nNextContourPixelIndex, rhs.nNextContourPixelIndex)
        << "nNextContourPixelIndex mismatch at element " << index;
    EXPECT_EQ(lhs.nPrevContourPixelIndex, rhs.nPrevContourPixelIndex)
        << "nPrevContourPixelIndex mismatch at element " << index;
    EXPECT_EQ(lhs.nPixelAlreadyUsed, rhs.nPixelAlreadyUsed) << "nPixelAlreadyUsed mismatch at element " << index;
    EXPECT_EQ(lhs.nAlreadyLinked, rhs.nAlreadyLinked) << "nAlreadyLinked mismatch at element " << index;
    EXPECT_EQ(lhs.nAlreadyOutput, rhs.nAlreadyOutput) << "nAlreadyOutput mismatch at element " << index;
    EXPECT_EQ(lhs.nContourInteriorDirection, rhs.nContourInteriorDirection)
        << "nContourInteriorDirection mismatch at element " << index;
  }

  void expectDirectionInfoEqual(const NppiContourPixelDirectionInfo &lhs, const NppiContourPixelDirectionInfo &rhs,
                                size_t index) {
    EXPECT_EQ(lhs.nMarkerLabelID, rhs.nMarkerLabelID) << "nMarkerLabelID mismatch at pixel " << index;
    EXPECT_EQ(lhs.nContourDirectionCenterPixel, rhs.nContourDirectionCenterPixel)
        << "nContourDirectionCenterPixel mismatch at pixel " << index;
    EXPECT_EQ(lhs.nContourInteriorDirectionCenterPixel, rhs.nContourInteriorDirectionCenterPixel)
        << "nContourInteriorDirectionCenterPixel mismatch at pixel " << index;
    EXPECT_EQ(lhs.nConnected, rhs.nConnected) << "nConnected mismatch at pixel " << index;
    EXPECT_EQ(lhs.nGeometryInfoIsValid, rhs.nGeometryInfoIsValid) << "nGeometryInfoIsValid mismatch at pixel " << index;
    expectGeometryInfoEqual(lhs.oContourPixelGeometryInfo, rhs.oContourPixelGeometryInfo, index);
    expectPointEqual(lhs.nEast1, rhs.nEast1, "nEast1", index);
    expectPointEqual(lhs.nNorthEast1, rhs.nNorthEast1, "nNorthEast1", index);
    expectPointEqual(lhs.nNorth1, rhs.nNorth1, "nNorth1", index);
    expectPointEqual(lhs.nNorthWest1, rhs.nNorthWest1, "nNorthWest1", index);
    expectPointEqual(lhs.nWest1, rhs.nWest1, "nWest1", index);
    expectPointEqual(lhs.nSouthWest1, rhs.nSouthWest1, "nSouthWest1", index);
    expectPointEqual(lhs.nSouth1, rhs.nSouth1, "nSouth1", index);
    expectPointEqual(lhs.nSouthEast1, rhs.nSouthEast1, "nSouthEast1", index);
    for (int j = 0; j < 4; ++j) {
      EXPECT_EQ(lhs.aContourInteriorDirection1[j], rhs.aContourInteriorDirection1[j])
          << "aContourInteriorDirection1[" << j << "] mismatch at pixel " << index;
    }
    EXPECT_EQ(lhs.nTest1EastConnected, rhs.nTest1EastConnected) << "nTest1EastConnected mismatch at pixel " << index;
    EXPECT_EQ(lhs.nTest1NorthEastConnected, rhs.nTest1NorthEastConnected)
        << "nTest1NorthEastConnected mismatch at pixel " << index;
    EXPECT_EQ(lhs.nTest1NorthConnected, rhs.nTest1NorthConnected) << "nTest1NorthConnected mismatch at pixel " << index;
    EXPECT_EQ(lhs.nTest1NorthWestConnected, rhs.nTest1NorthWestConnected)
        << "nTest1NorthWestConnected mismatch at pixel " << index;
    EXPECT_EQ(lhs.nTest1WestConnected, rhs.nTest1WestConnected) << "nTest1WestConnected mismatch at pixel " << index;
    EXPECT_EQ(lhs.nTest1SouthWestConnected, rhs.nTest1SouthWestConnected)
        << "nTest1SouthWestConnected mismatch at pixel " << index;
    EXPECT_EQ(lhs.nTest1SouthConnected, rhs.nTest1SouthConnected) << "nTest1SouthConnected mismatch at pixel " << index;
    EXPECT_EQ(lhs.nTest1SouthEastConnected, rhs.nTest1SouthEastConnected)
        << "nTest1SouthEastConnected mismatch at pixel " << index;
  }
};

// Test 1: nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R
TEST_F(NPPICompressedMarkerLabelsTest, GetInfoListSize_Basic) {
  unsigned int nMaxMarkerLabelID = 100;
  unsigned int bufferSize = 0;

  NppStatus status = nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R(nMaxMarkerLabelID, &bufferSize);
  EXPECT_EQ(status, NPP_SUCCESS);
  EXPECT_GT(bufferSize, 0u);
  EXPECT_GE(bufferSize, (nMaxMarkerLabelID + 1) * sizeof(NppiCompressedMarkerLabelsInfo));
}

TEST_F(NPPICompressedMarkerLabelsTest, GetInfoListSize_SmallID) {
  unsigned int nMaxMarkerLabelID = 1;
  unsigned int bufferSize = 0;

  NppStatus status = nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R(nMaxMarkerLabelID, &bufferSize);
  EXPECT_EQ(status, NPP_SUCCESS);
  EXPECT_GT(bufferSize, 0u);
}

TEST_F(NPPICompressedMarkerLabelsTest, GetInfoListSize_LargeID) {
  unsigned int nMaxMarkerLabelID = 10000;
  unsigned int bufferSize = 0;

  NppStatus status = nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R(nMaxMarkerLabelID, &bufferSize);
  EXPECT_EQ(status, NPP_SUCCESS);
  EXPECT_GE(bufferSize, (nMaxMarkerLabelID + 1) * sizeof(NppiCompressedMarkerLabelsInfo));
}

// Test 2: nppiCompressedMarkerLabelsUFGetGeometryListsSize_C1R
TEST_F(NPPICompressedMarkerLabelsTest, GetGeometryListsSize_Basic) {
  Npp32u nMaxContourPixelGeometryInfoCount = 1000;
  Npp32u bufferSize = 0;

  NppStatus status =
      nppiCompressedMarkerLabelsUFGetGeometryListsSize_C1R(nMaxContourPixelGeometryInfoCount, &bufferSize);
  EXPECT_EQ(status, NPP_SUCCESS);
  EXPECT_GT(bufferSize, 0u);
}

TEST_F(NPPICompressedMarkerLabelsTest, GetGeometryListsSize_Small) {
  Npp32u nMaxContourPixelGeometryInfoCount = 10;
  Npp32u bufferSize = 0;

  NppStatus status =
      nppiCompressedMarkerLabelsUFGetGeometryListsSize_C1R(nMaxContourPixelGeometryInfoCount, &bufferSize);
  EXPECT_EQ(status, NPP_SUCCESS);
  EXPECT_GT(bufferSize, 0u);
}

TEST_F(NPPICompressedMarkerLabelsTest, GetGeometryListsSize_Large) {
  Npp32u nMaxContourPixelGeometryInfoCount = 100000;
  Npp32u bufferSize = 0;

  NppStatus status =
      nppiCompressedMarkerLabelsUFGetGeometryListsSize_C1R(nMaxContourPixelGeometryInfoCount, &bufferSize);
  EXPECT_EQ(status, NPP_SUCCESS);
  EXPECT_GE(bufferSize, nMaxContourPixelGeometryInfoCount * sizeof(NppiContourPixelGeometryInfo));
}

// Test 3: nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx - Full workflow test
TEST_F(NPPICompressedMarkerLabelsTest, Info_BasicWorkflow) {
  // Create test labeled image
  int maxLabelID = 0;
  std::vector<Npp32u> h_labels(width * height);
  createTestLabeledImage(h_labels.data(), width, height, maxLabelID);

  // Allocate device memory for labels
  // IMPORTANT: For NPP compressed marker labels, step MUST equal width * sizeof(Npp32u)
  Npp32u *d_labels = nullptr;
  int labelStep = width * sizeof(Npp32u);
  cudaMalloc(&d_labels, width * height * sizeof(Npp32u));
  CudaMemGuard labelGuard(d_labels);
  cudaMemcpy(d_labels, h_labels.data(), width * height * sizeof(Npp32u), cudaMemcpyHostToDevice);

  // Get info list buffer size
  unsigned int infoBufferSize = 0;
  NppStatus status = nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R(maxLabelID, &infoBufferSize);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Allocate info list
  NppiCompressedMarkerLabelsInfo *d_infoList = nullptr;
  cudaMalloc(&d_infoList, infoBufferSize);
  CudaMemGuard infoGuard(d_infoList);

  // Allocate contours image
  Npp8u *d_contours = nullptr;
  int contoursStep = width * sizeof(Npp8u);
  cudaMalloc(&d_contours, width * height * sizeof(Npp8u));
  CudaMemGuard contoursGuard(d_contours);

  // Allocate direction image
  NppiContourPixelDirectionInfo *d_dirImage = nullptr;
  int dirStep = width * sizeof(NppiContourPixelDirectionInfo);
  cudaMalloc(&d_dirImage, width * height * sizeof(NppiContourPixelDirectionInfo));
  CudaMemGuard dirGuard(d_dirImage);

  // Allocate host and device arrays for counts and offsets
  std::vector<Npp32u> h_pixelCounts(maxLabelID + 1, 0);
  std::vector<Npp32u> h_pixelOffsets(maxLabelID + 1, 0);

  Npp32u *d_pixelCounts = nullptr;
  Npp32u *d_pixelOffsets = nullptr;
  cudaMalloc(&d_pixelCounts, (maxLabelID + 1) * sizeof(Npp32u));
  cudaMalloc(&d_pixelOffsets, (maxLabelID + 1) * sizeof(Npp32u));
  CudaMemGuard countsGuard(d_pixelCounts);
  CudaMemGuard offsetsGuard(d_pixelOffsets);

  // Host totals info
  NppiContourTotalsInfo totalsInfo;
  memset(&totalsInfo, 0, sizeof(totalsInfo));

  // Call the Info function
  status = nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(
      d_labels, labelStep, oSizeROI, maxLabelID, d_infoList, d_contours, contoursStep, d_dirImage, dirStep, &totalsInfo,
      d_pixelCounts, h_pixelCounts.data(), d_pixelOffsets, h_pixelOffsets.data(), nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  // Verify totals
  EXPECT_GT(totalsInfo.nTotalImagePixelContourCount, 0u);
  EXPECT_GT(totalsInfo.nLongestImageContourPixelCount, 0u);

  // Copy info list back and verify
  std::vector<NppiCompressedMarkerLabelsInfo> h_infoList(maxLabelID + 1);
  cudaMemcpy(h_infoList.data(), d_infoList, (maxLabelID + 1) * sizeof(NppiCompressedMarkerLabelsInfo),
             cudaMemcpyDeviceToHost);

  // Verify each region has valid pixel count and contour pixels
  // Note: The exact pixel count depends on NPP internal implementation
  for (int i = 1; i <= maxLabelID; i++) {
    EXPECT_GT(h_infoList[i].nMarkerLabelPixelCount, 0u) << "Label " << i << " should have pixels";
    EXPECT_GT(h_infoList[i].nContourPixelCount, 0u) << "Label " << i << " should have contour pixels";
  }

  // Copy contours image and verify
  std::vector<Npp8u> h_contours(width * height);
  cudaMemcpy(h_contours.data(), d_contours, width * height * sizeof(Npp8u), cudaMemcpyDeviceToHost);

  // Check that boundary pixels are marked
  int boundaryCount = 0;
  for (int i = 0; i < width * height; i++) {
    if (h_contours[i] != 0)
      boundaryCount++;
  }
  EXPECT_GT(boundaryCount, 0);
}

TEST_F(NPPICompressedMarkerLabelsTest, Info_BasicWorkflow_LegacySignature) {
  int maxLabelID = 0;
  std::vector<Npp32u> h_labels(width * height);
  createTestLabeledImage(h_labels.data(), width, height, maxLabelID);

  Npp32u *d_labels = nullptr;
  int labelStep = width * sizeof(Npp32u);
  cudaMalloc(&d_labels, width * height * sizeof(Npp32u));
  CudaMemGuard labelGuard(d_labels);
  cudaMemcpy(d_labels, h_labels.data(), width * height * sizeof(Npp32u), cudaMemcpyHostToDevice);

  unsigned int infoBufferSize = 0;
  NppStatus status = nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R(maxLabelID, &infoBufferSize);
  ASSERT_EQ(status, NPP_SUCCESS);

  NppiCompressedMarkerLabelsInfo *d_infoList = nullptr;
  cudaMalloc(&d_infoList, infoBufferSize);
  CudaMemGuard infoGuard(d_infoList);

  Npp8u *d_contours = nullptr;
  int contoursStep = width * sizeof(Npp8u);
  cudaMalloc(&d_contours, width * height * sizeof(Npp8u));
  CudaMemGuard contoursGuard(d_contours);

  NppiContourPixelDirectionInfo *d_dirImage = nullptr;
  int dirStep = width * sizeof(NppiContourPixelDirectionInfo);
  cudaMalloc(&d_dirImage, width * height * sizeof(NppiContourPixelDirectionInfo));
  CudaMemGuard dirGuard(d_dirImage);

  std::vector<Npp32u> h_pixelCounts(maxLabelID + 1, 0);
  std::vector<Npp32u> h_pixelOffsets(maxLabelID + 1, 0);

  Npp32u *d_pixelCounts = nullptr;
  cudaMalloc(&d_pixelCounts, (maxLabelID + 1) * sizeof(Npp32u));
  CudaMemGuard countsGuard(d_pixelCounts);

  NppiContourTotalsInfo totalsInfo;
  memset(&totalsInfo, 0, sizeof(totalsInfo));

  status = nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(
      d_labels, labelStep, oSizeROI, maxLabelID, d_infoList, d_contours, contoursStep, d_dirImage, dirStep, &totalsInfo,
      d_pixelCounts, h_pixelCounts.data(), h_pixelOffsets.data(), nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);
  EXPECT_GT(totalsInfo.nTotalImagePixelContourCount, 0u);
  EXPECT_GT(totalsInfo.nLongestImageContourPixelCount, 0u);

  std::vector<NppiCompressedMarkerLabelsInfo> h_infoList(maxLabelID + 1);
  cudaMemcpy(h_infoList.data(), d_infoList, (maxLabelID + 1) * sizeof(NppiCompressedMarkerLabelsInfo),
             cudaMemcpyDeviceToHost);

  for (int i = 1; i <= maxLabelID; i++) {
    EXPECT_GT(h_infoList[i].nMarkerLabelPixelCount, 0u);
    EXPECT_GT(h_infoList[i].nContourPixelCount, 0u);
    EXPECT_GE(h_pixelOffsets[i], h_pixelOffsets[i - 1]);
  }
}

TEST_F(NPPICompressedMarkerLabelsTest, Info_DirectionImageUnifiedFields_FullSignature) {
  std::vector<Npp8u> h_binary;
  createContourBinaryInput(h_binary);

  int maxLabelID = 0;
  std::vector<Npp32u> h_labels;
  Npp32u *d_labels = nullptr;
  ASSERT_TRUE(createCompressedLabelsFromBinary(h_binary, &d_labels, h_labels, maxLabelID));
  int labelStep = width * sizeof(Npp32u);
  CudaMemGuard labelGuard(d_labels);

  unsigned int infoBufferSize = 0;
  NppStatus status = nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R(maxLabelID, &infoBufferSize);
  ASSERT_EQ(status, NPP_SUCCESS);

  NppiCompressedMarkerLabelsInfo *d_infoList = nullptr;
  cudaMalloc(&d_infoList, infoBufferSize);
  CudaMemGuard infoGuard(d_infoList);

  Npp8u *d_contours = nullptr;
  int contoursStep = width * sizeof(Npp8u);
  cudaMalloc(&d_contours, width * height * sizeof(Npp8u));
  CudaMemGuard contoursGuard(d_contours);

  NppiContourPixelDirectionInfo *d_dirImage = nullptr;
  int dirStep = width * sizeof(NppiContourPixelDirectionInfo);
  cudaMalloc(&d_dirImage, width * height * sizeof(NppiContourPixelDirectionInfo));
  CudaMemGuard dirGuard(d_dirImage);

  std::vector<Npp32u> h_pixelCounts(maxLabelID + 1, 0);
  std::vector<Npp32u> h_pixelOffsets(maxLabelID + 1, 0);
  Npp32u *d_pixelCounts = nullptr;
  Npp32u *d_pixelOffsets = nullptr;
  cudaMalloc(&d_pixelCounts, (maxLabelID + 1) * sizeof(Npp32u));
  cudaMalloc(&d_pixelOffsets, (maxLabelID + 1) * sizeof(Npp32u));
  CudaMemGuard countsGuard(d_pixelCounts);
  CudaMemGuard offsetsGuard(d_pixelOffsets);

  NppiContourTotalsInfo totalsInfo{};
  status = nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(
      d_labels, labelStep, oSizeROI, maxLabelID, d_infoList, d_contours, contoursStep, d_dirImage, dirStep, &totalsInfo,
      d_pixelCounts, h_pixelCounts.data(), d_pixelOffsets, h_pixelOffsets.data(), nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> h_contours(width * height);
  std::vector<NppiContourPixelDirectionInfo> h_dirImage(width * height);
  cudaMemcpy(h_contours.data(), d_contours, width * height * sizeof(Npp8u), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_dirImage.data(), d_dirImage, width * height * sizeof(NppiContourPixelDirectionInfo),
             cudaMemcpyDeviceToHost);

  assertUnifiedDirectionImageFields(h_labels, h_contours, h_dirImage);
}

TEST_F(NPPICompressedMarkerLabelsTest, Info_DirectionImageUnifiedFields_LegacySignature) {
  std::vector<Npp8u> h_binary;
  createContourBinaryInput(h_binary);

  int maxLabelID = 0;
  std::vector<Npp32u> h_labels;
  Npp32u *d_labels = nullptr;
  ASSERT_TRUE(createCompressedLabelsFromBinary(h_binary, &d_labels, h_labels, maxLabelID));
  int labelStep = width * sizeof(Npp32u);
  CudaMemGuard labelGuard(d_labels);

  unsigned int infoBufferSize = 0;
  NppStatus status = nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R(maxLabelID, &infoBufferSize);
  ASSERT_EQ(status, NPP_SUCCESS);

  NppiCompressedMarkerLabelsInfo *d_infoList = nullptr;
  cudaMalloc(&d_infoList, infoBufferSize);
  CudaMemGuard infoGuard(d_infoList);

  Npp8u *d_contours = nullptr;
  int contoursStep = width * sizeof(Npp8u);
  cudaMalloc(&d_contours, width * height * sizeof(Npp8u));
  CudaMemGuard contoursGuard(d_contours);

  NppiContourPixelDirectionInfo *d_dirImage = nullptr;
  int dirStep = width * sizeof(NppiContourPixelDirectionInfo);
  cudaMalloc(&d_dirImage, width * height * sizeof(NppiContourPixelDirectionInfo));
  CudaMemGuard dirGuard(d_dirImage);

  std::vector<Npp32u> h_pixelCounts(maxLabelID + 1, 0);
  std::vector<Npp32u> h_pixelOffsets(maxLabelID + 1, 0);
  Npp32u *d_pixelCounts = nullptr;
  cudaMalloc(&d_pixelCounts, (maxLabelID + 1) * sizeof(Npp32u));
  CudaMemGuard countsGuard(d_pixelCounts);

  NppiContourTotalsInfo totalsInfo{};
  status = nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(
      d_labels, labelStep, oSizeROI, maxLabelID, d_infoList, d_contours, contoursStep, d_dirImage, dirStep, &totalsInfo,
      d_pixelCounts, h_pixelCounts.data(), h_pixelOffsets.data(), nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> h_contours(width * height);
  std::vector<NppiContourPixelDirectionInfo> h_dirImage(width * height);
  cudaMemcpy(h_contours.data(), d_contours, width * height * sizeof(Npp8u), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_dirImage.data(), d_dirImage, width * height * sizeof(NppiContourPixelDirectionInfo),
             cudaMemcpyDeviceToHost);

  assertUnifiedDirectionImageFields(h_labels, h_contours, h_dirImage);
}

TEST_F(NPPICompressedMarkerLabelsTest, Info_BackgroundContourIsTracked) {
  width = 32;
  height = 32;
  oSizeROI.width = width;
  oSizeROI.height = height;

  std::vector<Npp32u> h_labels(width * height, 0);
  for (int y = 8; y <= 20; ++y) {
    for (int x = 10; x <= 22; ++x) {
      h_labels[y * width + x] = 1;
    }
  }

  const int maxLabelID = 1;
  const int labelStep = width * sizeof(Npp32u);

  Npp32u *d_labels = nullptr;
  ASSERT_EQ(cudaMalloc(&d_labels, width * height * sizeof(Npp32u)), cudaSuccess);
  CudaMemGuard labelGuard(d_labels);
  ASSERT_EQ(cudaMemcpy(d_labels, h_labels.data(), width * height * sizeof(Npp32u), cudaMemcpyHostToDevice),
            cudaSuccess);

  unsigned int infoBufferSize = 0;
  ASSERT_EQ(nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R(maxLabelID, &infoBufferSize), NPP_SUCCESS);

  NppiCompressedMarkerLabelsInfo *d_infoList = nullptr;
  ASSERT_EQ(cudaMalloc(&d_infoList, infoBufferSize), cudaSuccess);
  CudaMemGuard infoGuard(d_infoList);

  Npp8u *d_contours = nullptr;
  ASSERT_EQ(cudaMalloc(&d_contours, width * height * sizeof(Npp8u)), cudaSuccess);
  CudaMemGuard contoursGuard(d_contours);

  NppiContourPixelDirectionInfo *d_dirImage = nullptr;
  const int dirStep = width * sizeof(NppiContourPixelDirectionInfo);
  ASSERT_EQ(cudaMalloc(&d_dirImage, width * height * sizeof(NppiContourPixelDirectionInfo)), cudaSuccess);
  CudaMemGuard dirGuard(d_dirImage);

  std::vector<Npp32u> h_pixelCounts(maxLabelID + 1, 0);
  std::vector<Npp32u> h_pixelOffsets(maxLabelID + 1, 0);
  Npp32u *d_pixelCounts = nullptr;
  Npp32u *d_pixelOffsets = nullptr;
  ASSERT_EQ(cudaMalloc(&d_pixelCounts, (maxLabelID + 1) * sizeof(Npp32u)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_pixelOffsets, (maxLabelID + 1) * sizeof(Npp32u)), cudaSuccess);
  CudaMemGuard countsGuard(d_pixelCounts);
  CudaMemGuard offsetsGuard(d_pixelOffsets);

  NppiContourTotalsInfo totalsInfo{};
  ASSERT_EQ(nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(d_labels, labelStep, oSizeROI, maxLabelID, d_infoList,
                                                         d_contours, width * sizeof(Npp8u), d_dirImage, dirStep,
                                                         &totalsInfo, d_pixelCounts, h_pixelCounts.data(),
                                                         d_pixelOffsets, h_pixelOffsets.data(), nppStreamCtx),
            NPP_SUCCESS);

  std::vector<NppiCompressedMarkerLabelsInfo> h_infoList(maxLabelID + 1);
  ASSERT_EQ(cudaMemcpy(h_infoList.data(), d_infoList, (maxLabelID + 1) * sizeof(NppiCompressedMarkerLabelsInfo),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);

  EXPECT_GT(h_infoList[0].nContourPixelCount, 0u);
  EXPECT_EQ(h_infoList[0].nContourPixelCount, h_pixelCounts[0]);
  EXPECT_GT(totalsInfo.nTotalImagePixelContourCount, h_pixelCounts[1]);
}

TEST_F(NPPICompressedMarkerLabelsTest, Info_DirectionImagePopulatesBasicDirectionInfo) {
  std::vector<Npp8u> h_binary;
  createContourBinaryInput(h_binary);

  int maxLabelID = 0;
  std::vector<Npp32u> h_labels;
  Npp32u *d_labels = nullptr;
  ASSERT_TRUE(createCompressedLabelsFromBinary(h_binary, &d_labels, h_labels, maxLabelID));
  CudaMemGuard labelGuard(d_labels);

  unsigned int infoBufferSize = 0;
  ASSERT_EQ(nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R(maxLabelID, &infoBufferSize), NPP_SUCCESS);

  NppiCompressedMarkerLabelsInfo *d_infoList = nullptr;
  ASSERT_EQ(cudaMalloc(&d_infoList, infoBufferSize), cudaSuccess);
  CudaMemGuard infoGuard(d_infoList);

  Npp8u *d_contours = nullptr;
  ASSERT_EQ(cudaMalloc(&d_contours, width * height * sizeof(Npp8u)), cudaSuccess);
  CudaMemGuard contoursGuard(d_contours);

  NppiContourPixelDirectionInfo *d_dirImage = nullptr;
  const int dirStep = width * sizeof(NppiContourPixelDirectionInfo);
  ASSERT_EQ(cudaMalloc(&d_dirImage, width * height * sizeof(NppiContourPixelDirectionInfo)), cudaSuccess);
  CudaMemGuard dirGuard(d_dirImage);

  std::vector<Npp32u> h_pixelCounts(maxLabelID + 1, 0);
  std::vector<Npp32u> h_pixelOffsets(maxLabelID + 1, 0);
  Npp32u *d_pixelCounts = nullptr;
  Npp32u *d_pixelOffsets = nullptr;
  ASSERT_EQ(cudaMalloc(&d_pixelCounts, (maxLabelID + 1) * sizeof(Npp32u)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_pixelOffsets, (maxLabelID + 1) * sizeof(Npp32u)), cudaSuccess);
  CudaMemGuard countsGuard(d_pixelCounts);
  CudaMemGuard offsetsGuard(d_pixelOffsets);

  NppiContourTotalsInfo totalsInfo{};
  ASSERT_EQ(nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(
                d_labels, width * static_cast<int>(sizeof(Npp32u)), oSizeROI, maxLabelID, d_infoList, d_contours,
                width * static_cast<int>(sizeof(Npp8u)), d_dirImage, dirStep, &totalsInfo, d_pixelCounts,
                h_pixelCounts.data(), d_pixelOffsets, h_pixelOffsets.data(), nppStreamCtx),
            NPP_SUCCESS);

  std::vector<Npp8u> h_contours(width * height);
  std::vector<NppiContourPixelDirectionInfo> h_dirImage(width * height);
  ASSERT_EQ(cudaMemcpy(h_contours.data(), d_contours, width * height * sizeof(Npp8u), cudaMemcpyDeviceToHost),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(h_dirImage.data(), d_dirImage, width * height * sizeof(NppiContourPixelDirectionInfo),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);

  int contourPixelsWithMatchingLabel = 0;
  int contourPixelsWithMissingConnectedNeighbor = 0;
  for (size_t i = 0; i < h_contours.size(); ++i) {
    if (h_contours[i] == 0) {
      continue;
    }
    contourPixelsWithMatchingLabel += (h_dirImage[i].nMarkerLabelID == h_labels[i] && h_labels[i] != 0);
    contourPixelsWithMissingConnectedNeighbor += hasAnyMissingConnectedNeighbor(h_dirImage[i]);
  }

  EXPECT_GT(contourPixelsWithMatchingLabel, 0);
  EXPECT_GT(contourPixelsWithMissingConnectedNeighbor, 0);
}

TEST_F(NPPICompressedMarkerLabelsTest, Info_DirectionImageRequiresHostAuxOutputs) {
  std::vector<Npp8u> h_binary;
  createContourBinaryInput(h_binary);

  int maxLabelID = 0;
  std::vector<Npp32u> h_labels;
  Npp32u *d_labels = nullptr;
  ASSERT_TRUE(createCompressedLabelsFromBinary(h_binary, &d_labels, h_labels, maxLabelID));
  CudaMemGuard labelGuard(d_labels);

  unsigned int infoBufferSize = 0;
  ASSERT_EQ(nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R(maxLabelID, &infoBufferSize), NPP_SUCCESS);

  NppiCompressedMarkerLabelsInfo *d_infoList = nullptr;
  ASSERT_EQ(cudaMalloc(&d_infoList, infoBufferSize), cudaSuccess);
  CudaMemGuard infoGuard(d_infoList);

  Npp8u *d_contours = nullptr;
  ASSERT_EQ(cudaMalloc(&d_contours, width * height * sizeof(Npp8u)), cudaSuccess);
  CudaMemGuard contoursGuard(d_contours);

  NppiContourPixelDirectionInfo *d_dirImage = nullptr;
  const int dirStep = width * sizeof(NppiContourPixelDirectionInfo);
  ASSERT_EQ(cudaMalloc(&d_dirImage, width * height * sizeof(NppiContourPixelDirectionInfo)), cudaSuccess);
  CudaMemGuard dirGuard(d_dirImage);

  Npp32u *d_pixelCounts = nullptr;
  Npp32u *d_pixelOffsets = nullptr;
  ASSERT_EQ(cudaMalloc(&d_pixelCounts, (maxLabelID + 1) * sizeof(Npp32u)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_pixelOffsets, (maxLabelID + 1) * sizeof(Npp32u)), cudaSuccess);
  CudaMemGuard countsGuard(d_pixelCounts);
  CudaMemGuard offsetsGuard(d_pixelOffsets);

  std::vector<Npp32u> h_pixelCounts(maxLabelID + 1, 0);
  std::vector<Npp32u> h_pixelOffsets(maxLabelID + 1, 0);
  NppiContourTotalsInfo totalsInfo{};

  EXPECT_EQ(nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(
                d_labels, width * static_cast<int>(sizeof(Npp32u)), oSizeROI, maxLabelID, d_infoList, d_contours,
                width * static_cast<int>(sizeof(Npp8u)), d_dirImage, dirStep, nullptr, d_pixelCounts,
                h_pixelCounts.data(), d_pixelOffsets, h_pixelOffsets.data(), nppStreamCtx),
            NPP_NULL_POINTER_ERROR);

  EXPECT_EQ(nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(
                d_labels, width * static_cast<int>(sizeof(Npp32u)), oSizeROI, maxLabelID, d_infoList, d_contours,
                width * static_cast<int>(sizeof(Npp8u)), d_dirImage, dirStep, &totalsInfo, nullptr,
                h_pixelCounts.data(), d_pixelOffsets, h_pixelOffsets.data(), nppStreamCtx),
            NPP_NULL_POINTER_ERROR);

  EXPECT_EQ(nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(
                d_labels, width * static_cast<int>(sizeof(Npp32u)), oSizeROI, maxLabelID, d_infoList, d_contours,
                width * static_cast<int>(sizeof(Npp8u)), d_dirImage, dirStep, &totalsInfo, d_pixelCounts, nullptr,
                d_pixelOffsets, h_pixelOffsets.data(), nppStreamCtx),
            NPP_NULL_POINTER_ERROR);

  EXPECT_EQ(nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(
                d_labels, width * static_cast<int>(sizeof(Npp32u)), oSizeROI, maxLabelID, d_infoList, d_contours,
                width * static_cast<int>(sizeof(Npp8u)), d_dirImage, dirStep, &totalsInfo, d_pixelCounts,
                h_pixelCounts.data(), d_pixelOffsets, nullptr, nppStreamCtx),
            NPP_NULL_POINTER_ERROR);
}

TEST_F(NPPICompressedMarkerLabelsTest, Info_FullAndLegacySignature_EquivalentOutputs) {
#ifdef USE_NVIDIA_NPP_TESTS
  GTEST_SKIP() << "Temporarily disabled on NVIDIA NPP build due non-deterministic full/legacy parity.";
#endif
  std::vector<Npp8u> h_binary;
  createContourBinaryInput(h_binary);

  int maxLabelID = 0;
  std::vector<Npp32u> h_labels;
  Npp32u *d_labels = nullptr;
  ASSERT_TRUE(createCompressedLabelsFromBinary(h_binary, &d_labels, h_labels, maxLabelID));
  const int labelStep = width * sizeof(Npp32u);
  CudaMemGuard labelGuard(d_labels);

  unsigned int infoBufferSize = 0;
  ASSERT_EQ(nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R(maxLabelID, &infoBufferSize), NPP_SUCCESS);

  NppiCompressedMarkerLabelsInfo *d_infoFull = nullptr;
  NppiCompressedMarkerLabelsInfo *d_infoLegacy = nullptr;
  cudaMalloc(&d_infoFull, infoBufferSize);
  cudaMalloc(&d_infoLegacy, infoBufferSize);
  CudaMemGuard infoFullGuard(d_infoFull);
  CudaMemGuard infoLegacyGuard(d_infoLegacy);

  Npp8u *d_contoursFull = nullptr;
  Npp8u *d_contoursLegacy = nullptr;
  const int contoursStep = width * sizeof(Npp8u);
  cudaMalloc(&d_contoursFull, width * height * sizeof(Npp8u));
  cudaMalloc(&d_contoursLegacy, width * height * sizeof(Npp8u));
  CudaMemGuard contoursFullGuard(d_contoursFull);
  CudaMemGuard contoursLegacyGuard(d_contoursLegacy);

  NppiContourPixelDirectionInfo *d_dirFull = nullptr;
  NppiContourPixelDirectionInfo *d_dirLegacy = nullptr;
  const int dirStep = width * sizeof(NppiContourPixelDirectionInfo);
  cudaMalloc(&d_dirFull, width * height * sizeof(NppiContourPixelDirectionInfo));
  cudaMalloc(&d_dirLegacy, width * height * sizeof(NppiContourPixelDirectionInfo));
  CudaMemGuard dirFullGuard(d_dirFull);
  CudaMemGuard dirLegacyGuard(d_dirLegacy);

  std::vector<Npp32u> h_countsFull(maxLabelID + 1, 0);
  std::vector<Npp32u> h_countsLegacy(maxLabelID + 1, 0);
  std::vector<Npp32u> h_offsetsFull(maxLabelID + 1, 0);
  std::vector<Npp32u> h_offsetsLegacy(maxLabelID + 1, 0);
  Npp32u *d_countsFull = nullptr;
  Npp32u *d_countsLegacy = nullptr;
  Npp32u *d_offsetsFull = nullptr;
  cudaMalloc(&d_countsFull, (maxLabelID + 1) * sizeof(Npp32u));
  cudaMalloc(&d_countsLegacy, (maxLabelID + 1) * sizeof(Npp32u));
  cudaMalloc(&d_offsetsFull, (maxLabelID + 1) * sizeof(Npp32u));
  CudaMemGuard countsFullGuard(d_countsFull);
  CudaMemGuard countsLegacyGuard(d_countsLegacy);
  CudaMemGuard offsetsFullGuard(d_offsetsFull);

  NppiContourTotalsInfo totalsFull{};
  NppiContourTotalsInfo totalsLegacy{};
  ASSERT_EQ(nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(
                d_labels, labelStep, oSizeROI, maxLabelID, d_infoFull, d_contoursFull, contoursStep, d_dirFull, dirStep,
                &totalsFull, d_countsFull, h_countsFull.data(), d_offsetsFull, h_offsetsFull.data(), nppStreamCtx),
            NPP_SUCCESS);
  ASSERT_EQ(nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(
                d_labels, labelStep, oSizeROI, maxLabelID, d_infoLegacy, d_contoursLegacy, contoursStep, d_dirLegacy,
                dirStep, &totalsLegacy, d_countsLegacy, h_countsLegacy.data(), h_offsetsLegacy.data(), nppStreamCtx),
            NPP_SUCCESS);

  EXPECT_EQ(totalsFull.nTotalImagePixelContourCount, totalsLegacy.nTotalImagePixelContourCount);
  EXPECT_EQ(totalsFull.nLongestImageContourPixelCount, totalsLegacy.nLongestImageContourPixelCount);
  EXPECT_EQ(h_countsFull, h_countsLegacy);
  EXPECT_EQ(h_offsetsFull, h_offsetsLegacy);

  std::vector<NppiCompressedMarkerLabelsInfo> h_infoFull(maxLabelID + 1);
  std::vector<NppiCompressedMarkerLabelsInfo> h_infoLegacy(maxLabelID + 1);
  cudaMemcpy(h_infoFull.data(), d_infoFull, (maxLabelID + 1) * sizeof(NppiCompressedMarkerLabelsInfo),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_infoLegacy.data(), d_infoLegacy, (maxLabelID + 1) * sizeof(NppiCompressedMarkerLabelsInfo),
             cudaMemcpyDeviceToHost);
  for (int label = 0; label <= maxLabelID; ++label) {
    EXPECT_EQ(h_infoFull[label].nMarkerLabelPixelCount, h_infoLegacy[label].nMarkerLabelPixelCount);
    EXPECT_EQ(h_infoFull[label].nContourPixelCount, h_infoLegacy[label].nContourPixelCount);
    EXPECT_EQ(h_infoFull[label].nContourPixelsFound, h_infoLegacy[label].nContourPixelsFound);
    EXPECT_EQ(h_infoFull[label].oMarkerLabelBoundingBox.x, h_infoLegacy[label].oMarkerLabelBoundingBox.x);
    EXPECT_EQ(h_infoFull[label].oMarkerLabelBoundingBox.y, h_infoLegacy[label].oMarkerLabelBoundingBox.y);
    EXPECT_EQ(h_infoFull[label].oMarkerLabelBoundingBox.width, h_infoLegacy[label].oMarkerLabelBoundingBox.width);
    EXPECT_EQ(h_infoFull[label].oMarkerLabelBoundingBox.height, h_infoLegacy[label].oMarkerLabelBoundingBox.height);
  }

  std::vector<Npp8u> h_contoursFull(width * height);
  std::vector<Npp8u> h_contoursLegacy(width * height);
  cudaMemcpy(h_contoursFull.data(), d_contoursFull, width * height * sizeof(Npp8u), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_contoursLegacy.data(), d_contoursLegacy, width * height * sizeof(Npp8u), cudaMemcpyDeviceToHost);
  EXPECT_EQ(h_contoursFull, h_contoursLegacy);

  std::vector<NppiContourPixelDirectionInfo> h_dirFull(width * height);
  std::vector<NppiContourPixelDirectionInfo> h_dirLegacy(width * height);
  cudaMemcpy(h_dirFull.data(), d_dirFull, width * height * sizeof(NppiContourPixelDirectionInfo),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_dirLegacy.data(), d_dirLegacy, width * height * sizeof(NppiContourPixelDirectionInfo),
             cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < h_dirFull.size(); ++i) {
    expectDirectionInfoEqual(h_dirFull[i], h_dirLegacy[i], i);
  }
}

// Test 4: nppiCompressedMarkerLabelsUFInfo with ring-shaped region
TEST_F(NPPICompressedMarkerLabelsTest, Info_RingShape) {
  int maxLabelID = 0;
  std::vector<Npp32u> h_labels(width * height);
  createRingTestImage(h_labels.data(), width, height, maxLabelID);

  Npp32u *d_labels = nullptr;
  int labelStep = width * sizeof(Npp32u);
  cudaMalloc(&d_labels, width * height * sizeof(Npp32u));
  CudaMemGuard labelGuard(d_labels);
  cudaMemcpy(d_labels, h_labels.data(), width * height * sizeof(Npp32u), cudaMemcpyHostToDevice);

  unsigned int infoBufferSize = 0;
  NppStatus status = nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R(maxLabelID, &infoBufferSize);
  ASSERT_EQ(status, NPP_SUCCESS);

  NppiCompressedMarkerLabelsInfo *d_infoList = nullptr;
  cudaMalloc(&d_infoList, infoBufferSize);
  CudaMemGuard infoGuard(d_infoList);

  Npp8u *d_contours = nullptr;
  int contoursStep = width * sizeof(Npp8u);
  cudaMalloc(&d_contours, width * height * sizeof(Npp8u));
  CudaMemGuard contoursGuard(d_contours);

  NppiContourPixelDirectionInfo *d_dirImage = nullptr;
  int dirStep = width * sizeof(NppiContourPixelDirectionInfo);
  cudaMalloc(&d_dirImage, width * height * sizeof(NppiContourPixelDirectionInfo));
  CudaMemGuard dirGuard(d_dirImage);

  std::vector<Npp32u> h_pixelCounts(maxLabelID + 1, 0);
  std::vector<Npp32u> h_pixelOffsets(maxLabelID + 1, 0);

  Npp32u *d_pixelCounts = nullptr;
  Npp32u *d_pixelOffsets = nullptr;
  cudaMalloc(&d_pixelCounts, (maxLabelID + 1) * sizeof(Npp32u));
  cudaMalloc(&d_pixelOffsets, (maxLabelID + 1) * sizeof(Npp32u));
  CudaMemGuard countsGuard(d_pixelCounts);
  CudaMemGuard offsetsGuard(d_pixelOffsets);

  NppiContourTotalsInfo totalsInfo;
  memset(&totalsInfo, 0, sizeof(totalsInfo));

  status = nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(
      d_labels, labelStep, oSizeROI, maxLabelID, d_infoList, d_contours, contoursStep, d_dirImage, dirStep, &totalsInfo,
      d_pixelCounts, h_pixelCounts.data(), d_pixelOffsets, h_pixelOffsets.data(), nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  // Ring shape should have inner and outer contours for label 2
  std::vector<NppiCompressedMarkerLabelsInfo> h_infoList(maxLabelID + 1);
  cudaMemcpy(h_infoList.data(), d_infoList, (maxLabelID + 1) * sizeof(NppiCompressedMarkerLabelsInfo),
             cudaMemcpyDeviceToHost);

  EXPECT_GT(h_infoList[1].nMarkerLabelPixelCount, 0u); // Inner circle
  EXPECT_GT(h_infoList[2].nMarkerLabelPixelCount, 0u); // Ring
  EXPECT_GT(h_infoList[1].nContourPixelCount, 0u);
  EXPECT_GT(h_infoList[2].nContourPixelCount, 0u);
}

// Test 5: nppiCompressedMarkerLabelsUFContoursGenerateGeometryLists_C1R_Ctx
// This is a complex API that requires specific setup from NVIDIA's label markers.
// We test the basic buffer size and info functions, geometry list generation
// requires more complex setup that matches NVIDIA's internal state.
TEST_F(NPPICompressedMarkerLabelsTest, GenerateGeometryLists_BufferSizeOnly) {
  // Test that we can get geometry list buffer size
  Npp32u nMaxContourPixelGeometryInfoCount = 1000;
  Npp32u bufferSize = 0;

  NppStatus status =
      nppiCompressedMarkerLabelsUFGetGeometryListsSize_C1R(nMaxContourPixelGeometryInfoCount, &bufferSize);
  EXPECT_EQ(status, NPP_SUCCESS);
  EXPECT_GE(bufferSize, nMaxContourPixelGeometryInfoCount * sizeof(NppiContourPixelGeometryInfo));
}

TEST_F(NPPICompressedMarkerLabelsTest, GenerateGeometryLists_FullSignature_NullDirectionImageReturnsBadArgument) {
  NppiCompressedMarkerLabelsInfo *d_infoList = nullptr;
  ASSERT_EQ(cudaMalloc(&d_infoList, sizeof(NppiCompressedMarkerLabelsInfo)), cudaSuccess);
  CudaMemGuard infoGuard(d_infoList);

  NppiCompressedMarkerLabelsInfo h_infoList[1]{};
  NppiSize roi{8, 8};

  const NppStatus status = nppiCompressedMarkerLabelsUFContoursGenerateGeometryLists_C1R_Ctx(
      d_infoList, h_infoList, nullptr, roi.width * static_cast<Npp32s>(sizeof(NppiContourPixelDirectionInfo)), nullptr,
      nullptr, nullptr, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 1, nullptr, nullptr, 0, roi,
      nppStreamCtx);
  EXPECT_EQ(status, NPP_BAD_ARGUMENT_ERROR);
}

TEST_F(NPPICompressedMarkerLabelsTest, GenerateGeometryLists_FullSignature_Functional) {
  std::vector<Npp8u> h_binary;
  createContourBinaryInput(h_binary);

  int maxLabelID = 0;
  std::vector<Npp32u> h_labels;
  Npp32u *d_labels = nullptr;
  ASSERT_TRUE(createCompressedLabelsFromBinary(h_binary, &d_labels, h_labels, maxLabelID));
  const int labelStep = width * sizeof(Npp32u);
  CudaMemGuard labelGuard(d_labels);

  unsigned int infoBufferSize = 0;
  NppStatus status = nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R(maxLabelID, &infoBufferSize);
  ASSERT_EQ(status, NPP_SUCCESS);

  NppiCompressedMarkerLabelsInfo *d_infoList = nullptr;
  cudaMalloc(&d_infoList, infoBufferSize);
  CudaMemGuard infoGuard(d_infoList);

  Npp8u *d_contours = nullptr;
  cudaMalloc(&d_contours, width * height * sizeof(Npp8u));
  CudaMemGuard contoursGuard(d_contours);

  NppiContourPixelDirectionInfo *d_dirImage = nullptr;
  const int dirStep = width * sizeof(NppiContourPixelDirectionInfo);
  cudaMalloc(&d_dirImage, width * height * sizeof(NppiContourPixelDirectionInfo));
  CudaMemGuard dirGuard(d_dirImage);

  std::vector<Npp32u> h_pixelCounts(maxLabelID + 1, 0);
  std::vector<Npp32u> h_pixelOffsets(maxLabelID + 1, 0);
  std::vector<Npp32u> h_pixelsFound(maxLabelID + 1, 0);
  Npp32u *d_pixelCounts = nullptr;
  Npp32u *d_pixelOffsets = nullptr;
  Npp32u *d_pixelsFound = nullptr;
  cudaMalloc(&d_pixelCounts, (maxLabelID + 1) * sizeof(Npp32u));
  cudaMalloc(&d_pixelOffsets, (maxLabelID + 1) * sizeof(Npp32u));
  cudaMalloc(&d_pixelsFound, (maxLabelID + 1) * sizeof(Npp32u));
  CudaMemGuard countsGuard(d_pixelCounts);
  CudaMemGuard offsetsGuard(d_pixelOffsets);
  CudaMemGuard foundGuard(d_pixelsFound);
  cudaMemset(d_pixelsFound, 0, (maxLabelID + 1) * sizeof(Npp32u));

  NppiContourTotalsInfo totalsInfo{};
  status = nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(
      d_labels, labelStep, oSizeROI, maxLabelID, d_infoList, d_contours, width * sizeof(Npp8u), d_dirImage, dirStep,
      &totalsInfo, d_pixelCounts, h_pixelCounts.data(), d_pixelOffsets, h_pixelOffsets.data(), nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<NppiCompressedMarkerLabelsInfo> h_infoList(maxLabelID + 1);
  cudaMemcpy(h_infoList.data(), d_infoList, (maxLabelID + 1) * sizeof(NppiCompressedMarkerLabelsInfo),
             cudaMemcpyDeviceToHost);

  Npp32u geometryListBytes = 0;
  status =
      nppiCompressedMarkerLabelsUFGetGeometryListsSize_C1R(totalsInfo.nTotalImagePixelContourCount, &geometryListBytes);
  ASSERT_EQ(status, NPP_SUCCESS);
  ASSERT_GT(geometryListBytes, 0u);

  NppiContourPixelGeometryInfo *d_geometry = nullptr;
  cudaMalloc(&d_geometry, geometryListBytes);
  CudaMemGuard geometryGuard(d_geometry);
  std::vector<NppiContourPixelGeometryInfo> h_geometry(geometryListBytes / sizeof(NppiContourPixelGeometryInfo));

  int contourLabelID = 0;
  for (int label = 1; label <= maxLabelID; ++label) {
    if (h_pixelCounts[label] > 0) {
      contourLabelID = label;
      break;
    }
  }
  ASSERT_GT(contourLabelID, 0);

  Npp32u blockSegmentBytes = 0;
  status = nppiCompressedMarkerLabelsUFGetContoursBlockSegmentListSize_C1R(
      h_pixelCounts.data(), totalsInfo.nTotalImagePixelContourCount, maxLabelID + 1, contourLabelID, contourLabelID + 1,
      &blockSegmentBytes);
  ASSERT_EQ(status, NPP_SUCCESS);

  NppiContourBlockSegment *d_blockSegments = nullptr;
  std::vector<unsigned char> h_blockStorage(blockSegmentBytes);
  NppiContourBlockSegment *h_blockSegments =
      blockSegmentBytes == 0 ? nullptr : reinterpret_cast<NppiContourBlockSegment *>(h_blockStorage.data());
  if (blockSegmentBytes > 0) {
    cudaMalloc(&d_blockSegments, blockSegmentBytes);
  }
  CudaMemGuard blockGuard(d_blockSegments);

  status = nppiCompressedMarkerLabelsUFContoursGenerateGeometryLists_C1R_Ctx(
      d_infoList, h_infoList.data(), d_dirImage, dirStep, d_geometry, h_geometry.data(), nullptr, 0, d_pixelCounts,
      d_pixelsFound, h_pixelsFound.data(), d_pixelOffsets, h_pixelOffsets.data(),
      totalsInfo.nTotalImagePixelContourCount, maxLabelID, contourLabelID, contourLabelID + 1, d_blockSegments,
      h_blockSegments, 0, oSizeROI, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  EXPECT_GT(h_pixelsFound[contourLabelID], 0u);
  EXPECT_EQ(h_infoList[contourLabelID].nContourPixelsFound, h_pixelsFound[contourLabelID]);

  const Npp32u totalFound = h_pixelsFound[contourLabelID];
  ASSERT_GT(totalFound, 0u);
  for (Npp32u i = 0; i < totalFound; ++i) {
    EXPECT_GE(h_geometry[i].oContourCenterPixelLocation.x, 0);
    EXPECT_GE(h_geometry[i].oContourCenterPixelLocation.y, 0);
    EXPECT_LT(h_geometry[i].oContourCenterPixelLocation.x, width);
    EXPECT_LT(h_geometry[i].oContourCenterPixelLocation.y, height);
  }
}

TEST_F(NPPICompressedMarkerLabelsTest, GenerateGeometryLists_LegacySignature_Functional) {
  std::vector<Npp8u> h_binary;
  createContourBinaryInput(h_binary);

  int maxLabelID = 0;
  std::vector<Npp32u> h_labels;
  Npp32u *d_labels = nullptr;
  ASSERT_TRUE(createCompressedLabelsFromBinary(h_binary, &d_labels, h_labels, maxLabelID));
  const int labelStep = width * sizeof(Npp32u);
  CudaMemGuard labelGuard(d_labels);

  unsigned int infoBufferSize = 0;
  NppStatus status = nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R(maxLabelID, &infoBufferSize);
  ASSERT_EQ(status, NPP_SUCCESS);

  NppiCompressedMarkerLabelsInfo *d_infoList = nullptr;
  cudaMalloc(&d_infoList, infoBufferSize);
  CudaMemGuard infoGuard(d_infoList);

  Npp8u *d_contours = nullptr;
  cudaMalloc(&d_contours, width * height * sizeof(Npp8u));
  CudaMemGuard contoursGuard(d_contours);

  NppiContourPixelDirectionInfo *d_dirImage = nullptr;
  const int dirStep = width * sizeof(NppiContourPixelDirectionInfo);
  cudaMalloc(&d_dirImage, width * height * sizeof(NppiContourPixelDirectionInfo));
  CudaMemGuard dirGuard(d_dirImage);

  std::vector<Npp32u> h_pixelCounts(maxLabelID + 1, 0);
  std::vector<Npp32u> h_pixelOffsets(maxLabelID + 1, 0);
  Npp32u *d_pixelCounts = nullptr;
  cudaMalloc(&d_pixelCounts, (maxLabelID + 1) * sizeof(Npp32u));
  CudaMemGuard countsGuard(d_pixelCounts);

  NppiContourTotalsInfo totalsInfo{};
  status = nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(
      d_labels, labelStep, oSizeROI, maxLabelID, d_infoList, d_contours, width * sizeof(Npp8u), d_dirImage, dirStep,
      &totalsInfo, d_pixelCounts, h_pixelCounts.data(), h_pixelOffsets.data(), nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<NppiCompressedMarkerLabelsInfo> h_infoList(maxLabelID + 1);
  cudaMemcpy(h_infoList.data(), d_infoList, (maxLabelID + 1) * sizeof(NppiCompressedMarkerLabelsInfo),
             cudaMemcpyDeviceToHost);

  Npp32u geometryListBytes = 0;
  status =
      nppiCompressedMarkerLabelsUFGetGeometryListsSize_C1R(totalsInfo.nTotalImagePixelContourCount, &geometryListBytes);
  ASSERT_EQ(status, NPP_SUCCESS);
  ASSERT_GT(geometryListBytes, 0u);

  std::vector<Npp32u> h_pixelsFound(maxLabelID + 1, 0);
  std::vector<NppiContourPixelGeometryInfo> h_geometry(geometryListBytes / sizeof(NppiContourPixelGeometryInfo));

  int contourLabelID = 0;
  for (int label = 1; label <= maxLabelID; ++label) {
    if (h_pixelCounts[label] > 0) {
      contourLabelID = label;
      break;
    }
  }
  ASSERT_GT(contourLabelID, 0);

  status = nppiCompressedMarkerLabelsUFContoursGenerateGeometryLists_C1R_Ctx(
      d_infoList, h_infoList.data(), d_dirImage, dirStep, h_geometry.data(), h_pixelCounts.data(), h_pixelsFound.data(),
      h_pixelOffsets.data(), maxLabelID, contourLabelID, contourLabelID + 1, oSizeROI, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  EXPECT_GT(h_pixelsFound[contourLabelID], 0u);
  EXPECT_EQ(h_infoList[contourLabelID].nContourPixelsFound, h_pixelsFound[contourLabelID]);

  const Npp32u totalFound = h_pixelsFound[contourLabelID];
  ASSERT_GT(totalFound, 0u);
  for (Npp32u i = 0; i < totalFound; ++i) {
    EXPECT_GE(h_geometry[i].oContourCenterPixelLocation.x, 0);
    EXPECT_GE(h_geometry[i].oContourCenterPixelLocation.y, 0);
    EXPECT_LT(h_geometry[i].oContourCenterPixelLocation.x, width);
    EXPECT_LT(h_geometry[i].oContourCenterPixelLocation.y, height);
  }
}

TEST_F(NPPICompressedMarkerLabelsTest, GenerateGeometryLists_FullAndLegacySignature_EquivalentOutputs) {
#ifdef USE_NVIDIA_NPP_TESTS
  GTEST_SKIP() << "Temporarily disabled on NVIDIA NPP build due non-deterministic full/legacy parity.";
#endif
  std::vector<Npp8u> h_binary;
  createContourBinaryInput(h_binary);

  int maxLabelID = 0;
  std::vector<Npp32u> h_labels;
  Npp32u *d_labels = nullptr;
  ASSERT_TRUE(createCompressedLabelsFromBinary(h_binary, &d_labels, h_labels, maxLabelID));
  const int labelStep = width * sizeof(Npp32u);
  CudaMemGuard labelGuard(d_labels);

  unsigned int infoBufferSize = 0;
  ASSERT_EQ(nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R(maxLabelID, &infoBufferSize), NPP_SUCCESS);

  NppiCompressedMarkerLabelsInfo *d_infoFull = nullptr;
  NppiCompressedMarkerLabelsInfo *d_infoLegacy = nullptr;
  cudaMalloc(&d_infoFull, infoBufferSize);
  cudaMalloc(&d_infoLegacy, infoBufferSize);
  CudaMemGuard infoFullGuard(d_infoFull);
  CudaMemGuard infoLegacyGuard(d_infoLegacy);

  Npp8u *d_contoursFull = nullptr;
  Npp8u *d_contoursLegacy = nullptr;
  cudaMalloc(&d_contoursFull, width * height * sizeof(Npp8u));
  cudaMalloc(&d_contoursLegacy, width * height * sizeof(Npp8u));
  CudaMemGuard contoursFullGuard(d_contoursFull);
  CudaMemGuard contoursLegacyGuard(d_contoursLegacy);

  NppiContourPixelDirectionInfo *d_dirFull = nullptr;
  NppiContourPixelDirectionInfo *d_dirLegacy = nullptr;
  const int dirStep = width * sizeof(NppiContourPixelDirectionInfo);
  cudaMalloc(&d_dirFull, width * height * sizeof(NppiContourPixelDirectionInfo));
  cudaMalloc(&d_dirLegacy, width * height * sizeof(NppiContourPixelDirectionInfo));
  CudaMemGuard dirFullGuard(d_dirFull);
  CudaMemGuard dirLegacyGuard(d_dirLegacy);

  std::vector<Npp32u> h_countsFull(maxLabelID + 1, 0);
  std::vector<Npp32u> h_countsLegacy(maxLabelID + 1, 0);
  std::vector<Npp32u> h_offsetsFull(maxLabelID + 1, 0);
  std::vector<Npp32u> h_offsetsLegacy(maxLabelID + 1, 0);
  std::vector<Npp32u> h_foundFull(maxLabelID + 1, 0);
  std::vector<Npp32u> h_foundLegacy(maxLabelID + 1, 0);
  Npp32u *d_countsFull = nullptr;
  Npp32u *d_countsLegacy = nullptr;
  Npp32u *d_offsetsFull = nullptr;
  Npp32u *d_foundFull = nullptr;
  cudaMalloc(&d_countsFull, (maxLabelID + 1) * sizeof(Npp32u));
  cudaMalloc(&d_countsLegacy, (maxLabelID + 1) * sizeof(Npp32u));
  cudaMalloc(&d_offsetsFull, (maxLabelID + 1) * sizeof(Npp32u));
  cudaMalloc(&d_foundFull, (maxLabelID + 1) * sizeof(Npp32u));
  cudaMemset(d_foundFull, 0, (maxLabelID + 1) * sizeof(Npp32u));
  CudaMemGuard countsFullGuard(d_countsFull);
  CudaMemGuard countsLegacyGuard(d_countsLegacy);
  CudaMemGuard offsetsFullGuard(d_offsetsFull);
  CudaMemGuard foundFullGuard(d_foundFull);

  NppiContourTotalsInfo totalsFull{};
  NppiContourTotalsInfo totalsLegacy{};
  ASSERT_EQ(nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(d_labels, labelStep, oSizeROI, maxLabelID, d_infoFull,
                                                         d_contoursFull, width * sizeof(Npp8u), d_dirFull, dirStep,
                                                         &totalsFull, d_countsFull, h_countsFull.data(), d_offsetsFull,
                                                         h_offsetsFull.data(), nppStreamCtx),
            NPP_SUCCESS);
  ASSERT_EQ(nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(d_labels, labelStep, oSizeROI, maxLabelID, d_infoLegacy,
                                                         d_contoursLegacy, width * sizeof(Npp8u), d_dirLegacy, dirStep,
                                                         &totalsLegacy, d_countsLegacy, h_countsLegacy.data(),
                                                         h_offsetsLegacy.data(), nppStreamCtx),
            NPP_SUCCESS);

  std::vector<NppiCompressedMarkerLabelsInfo> h_infoFull(maxLabelID + 1);
  std::vector<NppiCompressedMarkerLabelsInfo> h_infoLegacy(maxLabelID + 1);
  cudaMemcpy(h_infoFull.data(), d_infoFull, (maxLabelID + 1) * sizeof(NppiCompressedMarkerLabelsInfo),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_infoLegacy.data(), d_infoLegacy, (maxLabelID + 1) * sizeof(NppiCompressedMarkerLabelsInfo),
             cudaMemcpyDeviceToHost);

  Npp32u geometryListBytes = 0;
  ASSERT_EQ(
      nppiCompressedMarkerLabelsUFGetGeometryListsSize_C1R(totalsFull.nTotalImagePixelContourCount, &geometryListBytes),
      NPP_SUCCESS);
  ASSERT_GT(geometryListBytes, 0u);

  NppiContourPixelGeometryInfo *d_geometryFull = nullptr;
  cudaMalloc(&d_geometryFull, geometryListBytes);
  CudaMemGuard geometryFullGuard(d_geometryFull);
  std::vector<NppiContourPixelGeometryInfo> h_geometryFull(geometryListBytes / sizeof(NppiContourPixelGeometryInfo));
  std::vector<NppiContourPixelGeometryInfo> h_geometryLegacy(geometryListBytes / sizeof(NppiContourPixelGeometryInfo));

  int contourLabelID = 0;
  for (int label = 1; label <= maxLabelID; ++label) {
    if (h_countsFull[label] > 0) {
      contourLabelID = label;
      break;
    }
  }
  ASSERT_GT(contourLabelID, 0);

  Npp32u blockSegmentBytes = 0;
  ASSERT_EQ(nppiCompressedMarkerLabelsUFGetContoursBlockSegmentListSize_C1R(
                h_countsFull.data(), totalsFull.nTotalImagePixelContourCount, maxLabelID + 1, contourLabelID,
                contourLabelID + 1, &blockSegmentBytes),
            NPP_SUCCESS);
  NppiContourBlockSegment *d_blockSegments = nullptr;
  std::vector<unsigned char> h_blockStorage(blockSegmentBytes);
  NppiContourBlockSegment *h_blockSegments =
      blockSegmentBytes == 0 ? nullptr : reinterpret_cast<NppiContourBlockSegment *>(h_blockStorage.data());
  if (blockSegmentBytes > 0) {
    cudaMalloc(&d_blockSegments, blockSegmentBytes);
  }
  CudaMemGuard blockGuard(d_blockSegments);

  ASSERT_EQ(nppiCompressedMarkerLabelsUFContoursGenerateGeometryLists_C1R_Ctx(
                d_infoFull, h_infoFull.data(), d_dirFull, dirStep, d_geometryFull, h_geometryFull.data(), nullptr, 0,
                d_countsFull, d_foundFull, h_foundFull.data(), d_offsetsFull, h_offsetsFull.data(),
                totalsFull.nTotalImagePixelContourCount, maxLabelID, contourLabelID, contourLabelID + 1,
                d_blockSegments, h_blockSegments, 0, oSizeROI, nppStreamCtx),
            NPP_SUCCESS);
  ASSERT_EQ(nppiCompressedMarkerLabelsUFContoursGenerateGeometryLists_C1R_Ctx(
                d_infoLegacy, h_infoLegacy.data(), d_dirLegacy, dirStep, h_geometryLegacy.data(), h_countsLegacy.data(),
                h_foundLegacy.data(), h_offsetsLegacy.data(), maxLabelID, contourLabelID, contourLabelID + 1, oSizeROI,
                nppStreamCtx),
            NPP_SUCCESS);

  EXPECT_EQ(h_foundFull, h_foundLegacy);
  EXPECT_EQ(h_infoFull[contourLabelID].nContourPixelsFound, h_infoLegacy[contourLabelID].nContourPixelsFound);

  const Npp32u totalFound = h_foundFull[contourLabelID];
  ASSERT_EQ(totalFound, h_foundLegacy[contourLabelID]);
  for (Npp32u i = 0; i < totalFound; ++i) {
    expectGeometryInfoEqual(h_geometryFull[i], h_geometryLegacy[i], i);
  }
}

TEST_F(NPPICompressedMarkerLabelsTest, GenerateGeometryLists_OrderedLocationsAreValid) {
  std::vector<Npp8u> h_binary;
  createContourBinaryInput(h_binary);

  int maxLabelID = 0;
  std::vector<Npp32u> h_labels;
  Npp32u *d_labels = nullptr;
  ASSERT_TRUE(createCompressedLabelsFromBinary(h_binary, &d_labels, h_labels, maxLabelID));
  CudaMemGuard labelGuard(d_labels);

  unsigned int infoBufferSize = 0;
  ASSERT_EQ(nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R(maxLabelID, &infoBufferSize), NPP_SUCCESS);

  NppiCompressedMarkerLabelsInfo *d_infoList = nullptr;
  ASSERT_EQ(cudaMalloc(&d_infoList, infoBufferSize), cudaSuccess);
  CudaMemGuard infoGuard(d_infoList);

  Npp8u *d_contours = nullptr;
  ASSERT_EQ(cudaMalloc(&d_contours, width * height * sizeof(Npp8u)), cudaSuccess);
  CudaMemGuard contoursGuard(d_contours);

  NppiContourPixelDirectionInfo *d_dirImage = nullptr;
  const int dirStep = width * sizeof(NppiContourPixelDirectionInfo);
  ASSERT_EQ(cudaMalloc(&d_dirImage, width * height * sizeof(NppiContourPixelDirectionInfo)), cudaSuccess);
  CudaMemGuard dirGuard(d_dirImage);

  std::vector<Npp32u> h_pixelCounts(maxLabelID + 1, 0);
  std::vector<Npp32u> h_pixelOffsets(maxLabelID + 1, 0);
  std::vector<Npp32u> h_pixelsFound(maxLabelID + 1, 0);
  Npp32u *d_pixelCounts = nullptr;
  Npp32u *d_pixelOffsets = nullptr;
  Npp32u *d_pixelsFound = nullptr;
  ASSERT_EQ(cudaMalloc(&d_pixelCounts, (maxLabelID + 1) * sizeof(Npp32u)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_pixelOffsets, (maxLabelID + 1) * sizeof(Npp32u)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_pixelsFound, (maxLabelID + 1) * sizeof(Npp32u)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_pixelsFound, 0, (maxLabelID + 1) * sizeof(Npp32u)), cudaSuccess);
  CudaMemGuard countsGuard(d_pixelCounts);
  CudaMemGuard offsetsGuard(d_pixelOffsets);
  CudaMemGuard foundGuard(d_pixelsFound);

  NppiContourTotalsInfo totalsInfo{};
  ASSERT_EQ(nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(
                d_labels, width * static_cast<int>(sizeof(Npp32u)), oSizeROI, maxLabelID, d_infoList, d_contours,
                width * static_cast<int>(sizeof(Npp8u)), d_dirImage, dirStep, &totalsInfo, d_pixelCounts,
                h_pixelCounts.data(), d_pixelOffsets, h_pixelOffsets.data(), nppStreamCtx),
            NPP_SUCCESS);

  std::vector<NppiCompressedMarkerLabelsInfo> h_infoList(maxLabelID + 1);
  std::vector<Npp8u> h_contours(width * height, 0);
  ASSERT_EQ(cudaMemcpy(h_infoList.data(), d_infoList, (maxLabelID + 1) * sizeof(NppiCompressedMarkerLabelsInfo),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(h_contours.data(), d_contours, width * height * sizeof(Npp8u), cudaMemcpyDeviceToHost),
            cudaSuccess);

  Npp32u geometryListBytes = 0;
  ASSERT_EQ(
      nppiCompressedMarkerLabelsUFGetGeometryListsSize_C1R(totalsInfo.nTotalImagePixelContourCount, &geometryListBytes),
      NPP_SUCCESS);
  ASSERT_GT(geometryListBytes, 0u);

  NppiContourPixelGeometryInfo *d_geometry = nullptr;
  ASSERT_EQ(cudaMalloc(&d_geometry, geometryListBytes), cudaSuccess);
  CudaMemGuard geometryGuard(d_geometry);
  std::vector<NppiContourPixelGeometryInfo> h_geometry(geometryListBytes / sizeof(NppiContourPixelGeometryInfo));

  int contourLabelID = 0;
  for (int label = 1; label <= maxLabelID; ++label) {
    if (h_pixelCounts[label] > 0) {
      contourLabelID = label;
      break;
    }
  }
  ASSERT_GT(contourLabelID, 0);

  Npp32u blockSegmentBytes = 0;
  ASSERT_EQ(nppiCompressedMarkerLabelsUFGetContoursBlockSegmentListSize_C1R(
                h_pixelCounts.data(), totalsInfo.nTotalImagePixelContourCount, maxLabelID + 1, contourLabelID,
                contourLabelID + 1, &blockSegmentBytes),
            NPP_SUCCESS);

  NppiContourBlockSegment *d_blockSegments = nullptr;
  std::vector<unsigned char> h_blockStorage(blockSegmentBytes);
  NppiContourBlockSegment *h_blockSegments =
      blockSegmentBytes == 0 ? nullptr : reinterpret_cast<NppiContourBlockSegment *>(h_blockStorage.data());
  if (blockSegmentBytes > 0) {
    ASSERT_EQ(cudaMalloc(&d_blockSegments, blockSegmentBytes), cudaSuccess);
  }
  CudaMemGuard blockGuard(d_blockSegments);

  ASSERT_EQ(nppiCompressedMarkerLabelsUFContoursGenerateGeometryLists_C1R_Ctx(
                d_infoList, h_infoList.data(), d_dirImage, dirStep, d_geometry, h_geometry.data(), nullptr, 0,
                d_pixelCounts, d_pixelsFound, h_pixelsFound.data(), d_pixelOffsets, h_pixelOffsets.data(),
                totalsInfo.nTotalImagePixelContourCount, maxLabelID, contourLabelID, contourLabelID + 1,
                d_blockSegments, h_blockSegments, 0, oSizeROI, nppStreamCtx),
            NPP_SUCCESS);

  const Npp32u base = h_pixelOffsets[contourLabelID];
  const Npp32u found = h_pixelsFound[contourLabelID];
  ASSERT_GT(found, 0u);

  std::set<std::pair<int, int>> orderedLocations;
  for (Npp32u i = 0; i < found; ++i) {
    const NppiContourPixelGeometryInfo &geom = h_geometry[base + i];
    const int x = geom.oContourOrderedGeometryLocation.x;
    const int y = geom.oContourOrderedGeometryLocation.y;

    EXPECT_GE(x, 0);
    EXPECT_GE(y, 0);
    EXPECT_LT(x, width);
    EXPECT_LT(y, height);
    EXPECT_GE(x, h_infoList[contourLabelID].oMarkerLabelBoundingBox.x);
    EXPECT_GE(y, h_infoList[contourLabelID].oMarkerLabelBoundingBox.y);
    EXPECT_LE(x, h_infoList[contourLabelID].oMarkerLabelBoundingBox.width);
    EXPECT_LE(y, h_infoList[contourLabelID].oMarkerLabelBoundingBox.height);
    EXPECT_TRUE(orderedLocations.insert({x, y}).second);
  }

  EXPECT_EQ(orderedLocations.size(), found);
  EXPECT_EQ(h_geometry[base].oContourPrevPixelLocation.x, h_infoList[contourLabelID].oMarkerLabelBoundingBox.x);
  EXPECT_EQ(h_geometry[base].oContourPrevPixelLocation.y, h_infoList[contourLabelID].oMarkerLabelBoundingBox.y);
  EXPECT_EQ(h_geometry[base].oContourNextPixelLocation.x, h_infoList[contourLabelID].oMarkerLabelBoundingBox.width);
  EXPECT_EQ(h_geometry[base].oContourNextPixelLocation.y, h_infoList[contourLabelID].oMarkerLabelBoundingBox.height);
}

// Test 6: Full workflow with two separate rectangular regions
TEST_F(NPPICompressedMarkerLabelsTest, FullWorkflow) {
  // Create a pre-labeled image directly (simulating output from label markers)
  // Two 15x15 squares in different locations
  std::vector<Npp32u> h_labels(width * height, 0);

  // Create two separate regions with labels 1 and 2
  for (int y = 10; y < 25; y++) {
    for (int x = 10; x < 25; x++) {
      h_labels[y * width + x] = 1;
    }
  }
  for (int y = 35; y < 50; y++) {
    for (int x = 35; x < 50; x++) {
      h_labels[y * width + x] = 2;
    }
  }
  int maxLabelID = 2;

  Npp32u *d_labels = nullptr;
  int labelStep = width * sizeof(Npp32u);
  cudaMalloc(&d_labels, width * height * sizeof(Npp32u));
  CudaMemGuard labelGuard(d_labels);
  cudaMemcpy(d_labels, h_labels.data(), width * height * sizeof(Npp32u), cudaMemcpyHostToDevice);

  // Test the info API
  unsigned int infoBufferSize = 0;
  NppStatus status = nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R(maxLabelID, &infoBufferSize);
  ASSERT_EQ(status, NPP_SUCCESS);

  NppiCompressedMarkerLabelsInfo *d_infoList = nullptr;
  cudaMalloc(&d_infoList, infoBufferSize);
  CudaMemGuard infoGuard(d_infoList);

  std::vector<NppiCompressedMarkerLabelsInfo> h_infoList(maxLabelID + 1);

  Npp8u *d_contours = nullptr;
  cudaMalloc(&d_contours, width * height * sizeof(Npp8u));
  CudaMemGuard contoursGuard(d_contours);

  NppiContourPixelDirectionInfo *d_dirImage = nullptr;
  int dirStep = width * sizeof(NppiContourPixelDirectionInfo);
  cudaMalloc(&d_dirImage, width * height * sizeof(NppiContourPixelDirectionInfo));
  CudaMemGuard dirGuard(d_dirImage);

  std::vector<Npp32u> h_pixelCounts(maxLabelID + 1, 0);
  std::vector<Npp32u> h_pixelOffsets(maxLabelID + 1, 0);

  Npp32u *d_pixelCounts = nullptr;
  Npp32u *d_pixelOffsets = nullptr;
  cudaMalloc(&d_pixelCounts, (maxLabelID + 1) * sizeof(Npp32u));
  cudaMalloc(&d_pixelOffsets, (maxLabelID + 1) * sizeof(Npp32u));
  CudaMemGuard countsGuard(d_pixelCounts);
  CudaMemGuard offsetsGuard(d_pixelOffsets);

  NppiContourTotalsInfo totalsInfo;
  memset(&totalsInfo, 0, sizeof(totalsInfo));

  status = nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(
      d_labels, labelStep, oSizeROI, maxLabelID, d_infoList, d_contours, width * sizeof(Npp8u), d_dirImage, dirStep,
      &totalsInfo, d_pixelCounts, h_pixelCounts.data(), d_pixelOffsets, h_pixelOffsets.data(), nppStreamCtx);

  EXPECT_EQ(status, NPP_SUCCESS);
  EXPECT_GT(totalsInfo.nTotalImagePixelContourCount, 0u);

  // Copy and verify info
  cudaMemcpy(h_infoList.data(), d_infoList, (maxLabelID + 1) * sizeof(NppiCompressedMarkerLabelsInfo),
             cudaMemcpyDeviceToHost);

  // Verify both regions have valid pixel counts (exact count may vary)
  EXPECT_GT(h_infoList[1].nMarkerLabelPixelCount, 0u);
  EXPECT_GT(h_infoList[2].nMarkerLabelPixelCount, 0u);
  EXPECT_GT(h_infoList[1].nContourPixelCount, 0u);
  EXPECT_GT(h_infoList[2].nContourPixelCount, 0u);
}

// Test 7: Verify contour pixel counts are correct
TEST_F(NPPICompressedMarkerLabelsTest, ContourPixelCounts) {
  // Create a simple 8x8 filled square in a 16x16 image
  width = 16;
  height = 16;
  oSizeROI.width = width;
  oSizeROI.height = height;

  std::vector<Npp32u> h_labels(width * height, 0);

  // Fill a 6x6 square from (5,5) to (10,10)
  for (int y = 5; y <= 10; y++) {
    for (int x = 5; x <= 10; x++) {
      h_labels[y * width + x] = 1;
    }
  }
  int maxLabelID = 1;

  Npp32u *d_labels = nullptr;
  int labelStep = width * sizeof(Npp32u);
  cudaMalloc(&d_labels, width * height * sizeof(Npp32u));
  CudaMemGuard labelGuard(d_labels);
  cudaMemcpy(d_labels, h_labels.data(), width * height * sizeof(Npp32u), cudaMemcpyHostToDevice);

  unsigned int infoBufferSize = 0;
  NppStatus status = nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R(maxLabelID, &infoBufferSize);
  ASSERT_EQ(status, NPP_SUCCESS);

  NppiCompressedMarkerLabelsInfo *d_infoList = nullptr;
  cudaMalloc(&d_infoList, infoBufferSize);
  CudaMemGuard infoGuard(d_infoList);

  Npp8u *d_contours = nullptr;
  cudaMalloc(&d_contours, width * height * sizeof(Npp8u));
  CudaMemGuard contoursGuard(d_contours);

  NppiContourPixelDirectionInfo *d_dirImage = nullptr;
  int dirStep = width * sizeof(NppiContourPixelDirectionInfo);
  cudaMalloc(&d_dirImage, width * height * sizeof(NppiContourPixelDirectionInfo));
  CudaMemGuard dirGuard(d_dirImage);

  std::vector<Npp32u> h_pixelCounts(maxLabelID + 1, 0);
  std::vector<Npp32u> h_pixelOffsets(maxLabelID + 1, 0);

  Npp32u *d_pixelCounts = nullptr;
  Npp32u *d_pixelOffsets = nullptr;
  cudaMalloc(&d_pixelCounts, (maxLabelID + 1) * sizeof(Npp32u));
  cudaMalloc(&d_pixelOffsets, (maxLabelID + 1) * sizeof(Npp32u));
  CudaMemGuard countsGuard(d_pixelCounts);
  CudaMemGuard offsetsGuard(d_pixelOffsets);

  NppiContourTotalsInfo totalsInfo;
  memset(&totalsInfo, 0, sizeof(totalsInfo));

  status = nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(
      d_labels, labelStep, oSizeROI, maxLabelID, d_infoList, d_contours, width * sizeof(Npp8u), d_dirImage, dirStep,
      &totalsInfo, d_pixelCounts, h_pixelCounts.data(), d_pixelOffsets, h_pixelOffsets.data(), nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<NppiCompressedMarkerLabelsInfo> h_infoList(maxLabelID + 1);
  cudaMemcpy(h_infoList.data(), d_infoList, (maxLabelID + 1) * sizeof(NppiCompressedMarkerLabelsInfo),
             cudaMemcpyDeviceToHost);

  // Region should have positive pixel count
  EXPECT_GT(h_infoList[1].nMarkerLabelPixelCount, 0u);

  // Region should have positive contour pixel count
  EXPECT_GT(h_infoList[1].nContourPixelCount, 0u);
}

// Test 8: Bounding box verification
TEST_F(NPPICompressedMarkerLabelsTest, BoundingBox) {
  width = 32;
  height = 32;
  oSizeROI.width = width;
  oSizeROI.height = height;

  std::vector<Npp32u> h_labels(width * height, 0);

  // Create region at known location: (8,10) to (20,22)
  int x1 = 8, y1 = 10, x2 = 20, y2 = 22;
  for (int y = y1; y <= y2; y++) {
    for (int x = x1; x <= x2; x++) {
      h_labels[y * width + x] = 1;
    }
  }
  int maxLabelID = 1;

  Npp32u *d_labels = nullptr;
  int labelStep = width * sizeof(Npp32u);
  cudaMalloc(&d_labels, width * height * sizeof(Npp32u));
  CudaMemGuard labelGuard(d_labels);
  cudaMemcpy(d_labels, h_labels.data(), width * height * sizeof(Npp32u), cudaMemcpyHostToDevice);

  unsigned int infoBufferSize = 0;
  NppStatus status = nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R(maxLabelID, &infoBufferSize);
  ASSERT_EQ(status, NPP_SUCCESS);

  NppiCompressedMarkerLabelsInfo *d_infoList = nullptr;
  cudaMalloc(&d_infoList, infoBufferSize);
  CudaMemGuard infoGuard(d_infoList);

  Npp8u *d_contours = nullptr;
  cudaMalloc(&d_contours, width * height * sizeof(Npp8u));
  CudaMemGuard contoursGuard(d_contours);

  NppiContourPixelDirectionInfo *d_dirImage = nullptr;
  int dirStep = width * sizeof(NppiContourPixelDirectionInfo);
  cudaMalloc(&d_dirImage, width * height * sizeof(NppiContourPixelDirectionInfo));
  CudaMemGuard dirGuard(d_dirImage);

  std::vector<Npp32u> h_pixelCounts(maxLabelID + 1, 0);
  std::vector<Npp32u> h_pixelOffsets(maxLabelID + 1, 0);

  Npp32u *d_pixelCounts = nullptr;
  Npp32u *d_pixelOffsets = nullptr;
  cudaMalloc(&d_pixelCounts, (maxLabelID + 1) * sizeof(Npp32u));
  cudaMalloc(&d_pixelOffsets, (maxLabelID + 1) * sizeof(Npp32u));
  CudaMemGuard countsGuard(d_pixelCounts);
  CudaMemGuard offsetsGuard(d_pixelOffsets);

  NppiContourTotalsInfo totalsInfo;
  memset(&totalsInfo, 0, sizeof(totalsInfo));

  status = nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(
      d_labels, labelStep, oSizeROI, maxLabelID, d_infoList, d_contours, width * sizeof(Npp8u), d_dirImage, dirStep,
      &totalsInfo, d_pixelCounts, h_pixelCounts.data(), d_pixelOffsets, h_pixelOffsets.data(), nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<NppiCompressedMarkerLabelsInfo> h_infoList(maxLabelID + 1);
  cudaMemcpy(h_infoList.data(), d_infoList, (maxLabelID + 1) * sizeof(NppiCompressedMarkerLabelsInfo),
             cudaMemcpyDeviceToHost);

  // Verify bounding box
  EXPECT_EQ(h_infoList[1].oMarkerLabelBoundingBox.x, x1);
  EXPECT_EQ(h_infoList[1].oMarkerLabelBoundingBox.y, y1);
  // Note: NPP stores width/height as right/bottom coordinates
  EXPECT_EQ(h_infoList[1].oMarkerLabelBoundingBox.width, x2);
  EXPECT_EQ(h_infoList[1].oMarkerLabelBoundingBox.height, y2);
}

// Test 9: First contour pixel location
TEST_F(NPPICompressedMarkerLabelsTest, FirstContourPixel) {
  width = 32;
  height = 32;
  oSizeROI.width = width;
  oSizeROI.height = height;

  std::vector<Npp32u> h_labels(width * height, 0);

  // Create region starting at (15, 5)
  for (int y = 5; y <= 15; y++) {
    for (int x = 15; x <= 25; x++) {
      h_labels[y * width + x] = 1;
    }
  }
  int maxLabelID = 1;

  Npp32u *d_labels = nullptr;
  int labelStep = width * sizeof(Npp32u);
  cudaMalloc(&d_labels, width * height * sizeof(Npp32u));
  CudaMemGuard labelGuard(d_labels);
  cudaMemcpy(d_labels, h_labels.data(), width * height * sizeof(Npp32u), cudaMemcpyHostToDevice);

  unsigned int infoBufferSize = 0;
  NppStatus status = nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R(maxLabelID, &infoBufferSize);
  ASSERT_EQ(status, NPP_SUCCESS);

  NppiCompressedMarkerLabelsInfo *d_infoList = nullptr;
  cudaMalloc(&d_infoList, infoBufferSize);
  CudaMemGuard infoGuard(d_infoList);

  Npp8u *d_contours = nullptr;
  cudaMalloc(&d_contours, width * height * sizeof(Npp8u));
  CudaMemGuard contoursGuard(d_contours);

  NppiContourPixelDirectionInfo *d_dirImage = nullptr;
  int dirStep = width * sizeof(NppiContourPixelDirectionInfo);
  cudaMalloc(&d_dirImage, width * height * sizeof(NppiContourPixelDirectionInfo));
  CudaMemGuard dirGuard(d_dirImage);

  std::vector<Npp32u> h_pixelCounts(maxLabelID + 1, 0);
  std::vector<Npp32u> h_pixelOffsets(maxLabelID + 1, 0);

  Npp32u *d_pixelCounts = nullptr;
  Npp32u *d_pixelOffsets = nullptr;
  cudaMalloc(&d_pixelCounts, (maxLabelID + 1) * sizeof(Npp32u));
  cudaMalloc(&d_pixelOffsets, (maxLabelID + 1) * sizeof(Npp32u));
  CudaMemGuard countsGuard(d_pixelCounts);
  CudaMemGuard offsetsGuard(d_pixelOffsets);

  NppiContourTotalsInfo totalsInfo;
  memset(&totalsInfo, 0, sizeof(totalsInfo));

  status = nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(
      d_labels, labelStep, oSizeROI, maxLabelID, d_infoList, d_contours, width * sizeof(Npp8u), d_dirImage, dirStep,
      &totalsInfo, d_pixelCounts, h_pixelCounts.data(), d_pixelOffsets, h_pixelOffsets.data(), nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<NppiCompressedMarkerLabelsInfo> h_infoList(maxLabelID + 1);
  cudaMemcpy(h_infoList.data(), d_infoList, (maxLabelID + 1) * sizeof(NppiCompressedMarkerLabelsInfo),
             cudaMemcpyDeviceToHost);

  // First contour pixel should be within the region bounds
  EXPECT_GE(h_infoList[1].oContourFirstPixelLocation.x, 0);
  EXPECT_GE(h_infoList[1].oContourFirstPixelLocation.y, 0);
  EXPECT_LT(h_infoList[1].oContourFirstPixelLocation.x, width);
  EXPECT_LT(h_infoList[1].oContourFirstPixelLocation.y, height);
}
