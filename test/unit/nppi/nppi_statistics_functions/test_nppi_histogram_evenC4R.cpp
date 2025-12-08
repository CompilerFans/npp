#include "npp.h"
#include "framework/npp_test_base.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using namespace npp_functional_test;

/**
 * @brief NPP HistogramEven Tests
 *
 * Tests for nppiHistogramEven_16u_C4R and nppiHistogramEven_16s_C4R families.
 * These tests verify basic histogram computation on 4-channel 16-bit images.
 */
class NPPIHistogramEvenC4RTest : public NppTestBase {
protected:
  void SetUp() override {
    NppTestBase::SetUp();
    width = 8;
    height = 8;
    roi.width = width;
    roi.height = height;
  }

  int width, height;
  NppiSize roi;
};

/**
 * @brief Test nppiHistogramEven_16u_C4R basic functionality
 *
 * Verifies histogram computation for 16-bit unsigned 4-channel images.
 */
TEST_F(NPPIHistogramEvenC4RTest, HistogramEven_16u_C4R) {
  size_t dataSize = width * height * 4;

  // Generate test data with known distribution
  std::vector<Npp16u> srcData(dataSize);
  for (size_t i = 0; i < dataSize; i += 4) {
    srcData[i + 0] = 100;  // Channel 0
    srcData[i + 1] = 200;  // Channel 1
    srcData[i + 2] = 300;  // Channel 2
    srcData[i + 3] = 400;  // Channel 3
  }

  NppImageMemory<Npp16u> src(width, height, 4);
  src.copyFromHost(srcData);

  // Ensure data is copied before histogram computation
  cudaDeviceSynchronize();

  // Setup histogram parameters
  int nLevels[4] = {257, 257, 257, 257};  // nLevels must be nBins + 1
  Npp32s nLowerLevel[4] = {0, 0, 0, 0};
  Npp32s nUpperLevel[4] = {512, 512, 512, 512};

  // Allocate histogram buffers
  DeviceMemory<Npp32s> hist0, hist1, hist2, hist3;
  hist0.allocate(nLevels[0] - 1);
  hist1.allocate(nLevels[1] - 1);
  hist2.allocate(nLevels[2] - 1);
  hist3.allocate(nLevels[3] - 1);

  Npp32s *pHist[4] = {hist0.get(), hist1.get(), hist2.get(), hist3.get()};

  // Get buffer size
  SIZE_TYPE bufferSize = 0;
  NppStatus status = nppiHistogramEvenGetBufferSize_16u_C4R(roi, nLevels, &bufferSize);
  EXPECT_EQ(status, NPP_SUCCESS);

  DeviceMemory<Npp8u> deviceBuffer;
  deviceBuffer.allocate(bufferSize);

  // Compute histogram
  status = nppiHistogramEven_16u_C4R(
      src.get(), src.step(),
      roi, pHist, nLevels, nLowerLevel, nUpperLevel,
      deviceBuffer.get()
  );

  EXPECT_EQ(status, NPP_SUCCESS);

  // Synchronize before reading results (NPP functions are asynchronous)
  cudaDeviceSynchronize();

  // Verify histogram results
  std::vector<Npp32s> histData0, histData1, histData2, histData3;
  hist0.copyToHost(histData0);
  hist1.copyToHost(histData1);
  hist2.copyToHost(histData2);
  hist3.copyToHost(histData3);

  // All pixels in channel 0 have value 100
  // bin = (value - lowerLevel) * (nLevels - 1) / (upperLevel - lowerLevel)
  int expectedBin0 = ((100 - nLowerLevel[0]) * (nLevels[0] - 1)) / (nUpperLevel[0] - nLowerLevel[0]);
  if (expectedBin0 >= nLevels[0] - 1) expectedBin0 = nLevels[0] - 2;
  EXPECT_EQ(histData0[expectedBin0], width * height) << "All pixels should be in bin " << expectedBin0;

  // Verify other bins are empty for channel 0
  int totalCount0 = 0;
  for (int i = 0; i < nLevels[0] - 1; i++) {
    totalCount0 += histData0[i];
  }
  EXPECT_EQ(totalCount0, width * height) << "Total histogram count should equal number of pixels";

  // All pixels in channel 1 have value 200
  int expectedBin1 = ((200 - nLowerLevel[1]) * (nLevels[1] - 1)) / (nUpperLevel[1] - nLowerLevel[1]);
  if (expectedBin1 >= nLevels[1] - 1) expectedBin1 = nLevels[1] - 2;
  EXPECT_EQ(histData1[expectedBin1], width * height) << "All pixels should be in bin " << expectedBin1;

  // All pixels in channel 2 have value 300
  int expectedBin2 = ((300 - nLowerLevel[2]) * (nLevels[2] - 1)) / (nUpperLevel[2] - nLowerLevel[2]);
  if (expectedBin2 >= nLevels[2] - 1) expectedBin2 = nLevels[2] - 2;
  EXPECT_EQ(histData2[expectedBin2], width * height) << "All pixels should be in bin " << expectedBin2;

  // All pixels in channel 3 have value 400
  int expectedBin3 = ((400 - nLowerLevel[3]) * (nLevels[3] - 1)) / (nUpperLevel[3] - nLowerLevel[3]);
  if (expectedBin3 >= nLevels[3] - 1) expectedBin3 = nLevels[3] - 2;
  EXPECT_EQ(histData3[expectedBin3], width * height) << "All pixels should be in bin " << expectedBin3;
}

/**
 * @brief Test nppiHistogramEven_16u_C4R_Ctx with stream context
 *
 * Verifies the context-aware version that supports custom CUDA streams.
 */
TEST_F(NPPIHistogramEvenC4RTest, HistogramEven_16u_C4R_Ctx) {
  size_t dataSize = width * height * 4;

  std::vector<Npp16u> srcData(dataSize);
  for (size_t i = 0; i < dataSize; i += 4) {
    srcData[i + 0] = 100;
    srcData[i + 1] = 200;
    srcData[i + 2] = 300;
    srcData[i + 3] = 400;
  }

  NppImageMemory<Npp16u> src(width, height, 4);
  src.copyFromHost(srcData);

  // Ensure data is copied before histogram computation
  cudaDeviceSynchronize();

  int nLevels[4] = {257, 257, 257, 257};  // nLevels must be nBins + 1
  Npp32s nLowerLevel[4] = {0, 0, 0, 0};
  Npp32s nUpperLevel[4] = {512, 512, 512, 512};

  DeviceMemory<Npp32s> hist0, hist1, hist2, hist3;
  hist0.allocate(nLevels[0] - 1);
  hist1.allocate(nLevels[1] - 1);
  hist2.allocate(nLevels[2] - 1);
  hist3.allocate(nLevels[3] - 1);

  Npp32s *pHist[4] = {hist0.get(), hist1.get(), hist2.get(), hist3.get()};

  SIZE_TYPE bufferSize = 0;
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;

  NppStatus status = nppiHistogramEvenGetBufferSize_16u_C4R_Ctx(roi, nLevels, &bufferSize, nppStreamCtx);
  EXPECT_EQ(status, NPP_SUCCESS);

  DeviceMemory<Npp8u> deviceBuffer;
  deviceBuffer.allocate(bufferSize);

  status = nppiHistogramEven_16u_C4R_Ctx(
      src.get(), src.step(),
      roi, pHist, nLevels, nLowerLevel, nUpperLevel,
      deviceBuffer.get(), nppStreamCtx
  );

  EXPECT_EQ(status, NPP_SUCCESS);

  // Synchronize before reading results (NPP functions are asynchronous)
  cudaDeviceSynchronize();

  // Verify histogram results for all channels
  std::vector<Npp32s> histData0, histData1, histData2, histData3;
  hist0.copyToHost(histData0);
  hist1.copyToHost(histData1);
  hist2.copyToHost(histData2);
  hist3.copyToHost(histData3);

  // Channel 0: value 100
  int expectedBin0 = ((100 - nLowerLevel[0]) * (nLevels[0] - 1)) / (nUpperLevel[0] - nLowerLevel[0]);
  if (expectedBin0 >= nLevels[0] - 1) expectedBin0 = nLevels[0] - 2;
  EXPECT_EQ(histData0[expectedBin0], width * height) << "Channel 0: All pixels should be in bin " << expectedBin0;

  // Channel 1: value 200
  int expectedBin1 = ((200 - nLowerLevel[1]) * (nLevels[1] - 1)) / (nUpperLevel[1] - nLowerLevel[1]);
  if (expectedBin1 >= nLevels[1] - 1) expectedBin1 = nLevels[1] - 2;
  EXPECT_EQ(histData1[expectedBin1], width * height) << "Channel 1: All pixels should be in bin " << expectedBin1;

  // Channel 2: value 300
  int expectedBin2 = ((300 - nLowerLevel[2]) * (nLevels[2] - 1)) / (nUpperLevel[2] - nLowerLevel[2]);
  if (expectedBin2 >= nLevels[2] - 1) expectedBin2 = nLevels[2] - 2;
  EXPECT_EQ(histData2[expectedBin2], width * height) << "Channel 2: All pixels should be in bin " << expectedBin2;

  // Channel 3: value 400
  int expectedBin3 = ((400 - nLowerLevel[3]) * (nLevels[3] - 1)) / (nUpperLevel[3] - nLowerLevel[3]);
  if (expectedBin3 >= nLevels[3] - 1) expectedBin3 = nLevels[3] - 2;
  EXPECT_EQ(histData3[expectedBin3], width * height) << "Channel 3: All pixels should be in bin " << expectedBin3;

  // Verify total counts
  int totalCount0 = 0, totalCount1 = 0, totalCount2 = 0, totalCount3 = 0;
  for (int i = 0; i < nLevels[0] - 1; i++) totalCount0 += histData0[i];
  for (int i = 0; i < nLevels[1] - 1; i++) totalCount1 += histData1[i];
  for (int i = 0; i < nLevels[2] - 1; i++) totalCount2 += histData2[i];
  for (int i = 0; i < nLevels[3] - 1; i++) totalCount3 += histData3[i];

  EXPECT_EQ(totalCount0, width * height) << "Channel 0: Total count mismatch";
  EXPECT_EQ(totalCount1, width * height) << "Channel 1: Total count mismatch";
  EXPECT_EQ(totalCount2, width * height) << "Channel 2: Total count mismatch";
  EXPECT_EQ(totalCount3, width * height) << "Channel 3: Total count mismatch";
}

/**
 * @brief Test nppiHistogramEven_16s_C4R basic functionality
 *
 * Verifies histogram computation for 16-bit signed 4-channel images.
 * Note: Using Npp16u with signed interpretation since Npp16s doesn't support C4.
 */
TEST_F(NPPIHistogramEvenC4RTest, HistogramEven_16s_C4R) {
  size_t dataSize = width * height * 4;

  // Generate test data with signed values (stored as unsigned for C4 support)
  std::vector<Npp16u> srcData(dataSize);
  for (size_t i = 0; i < dataSize; i += 4) {
    srcData[i + 0] = static_cast<Npp16u>(static_cast<Npp16s>(-100));  // Channel 0
    srcData[i + 1] = 0;     // Channel 1
    srcData[i + 2] = 100;   // Channel 2
    srcData[i + 3] = 200;   // Channel 3
  }

  NppImageMemory<Npp16u> src(width, height, 4);
  src.copyFromHost(srcData);

  // Ensure data is copied before histogram computation
  cudaDeviceSynchronize();

  // Setup histogram parameters for signed data
  int nLevels[4] = {257, 257, 257, 257};  // nLevels must be nBins + 1
  Npp32s nLowerLevel[4] = {-256, -256, -256, -256};
  Npp32s nUpperLevel[4] = {256, 256, 256, 256};

  DeviceMemory<Npp32s> hist0, hist1, hist2, hist3;
  hist0.allocate(nLevels[0] - 1);
  hist1.allocate(nLevels[1] - 1);
  hist2.allocate(nLevels[2] - 1);
  hist3.allocate(nLevels[3] - 1);

  Npp32s *pHist[4] = {hist0.get(), hist1.get(), hist2.get(), hist3.get()};

  SIZE_TYPE bufferSize = 0;
  NppStatus status = nppiHistogramEvenGetBufferSize_16s_C4R(roi, nLevels, &bufferSize);
  EXPECT_EQ(status, NPP_SUCCESS);

  DeviceMemory<Npp8u> deviceBuffer;
  deviceBuffer.allocate(bufferSize);

  status = nppiHistogramEven_16s_C4R(
      reinterpret_cast<const Npp16s*>(src.get()), src.step(),
      roi, pHist, nLevels, nLowerLevel, nUpperLevel,
      deviceBuffer.get()
  );

  EXPECT_EQ(status, NPP_SUCCESS);

  // Synchronize before reading results (NPP functions are asynchronous)
  cudaDeviceSynchronize();

  // Verify histogram results for all channels
  std::vector<Npp32s> histData0, histData1, histData2, histData3;
  hist0.copyToHost(histData0);
  hist1.copyToHost(histData1);
  hist2.copyToHost(histData2);
  hist3.copyToHost(histData3);

  // Channel 0: value -100 (stored as unsigned, interpreted as signed)
  // For signed 16-bit: -100 is stored as 0xFF9C (65436 in unsigned)
  Npp16s signedVal0 = -100;
  int pixelValue0 = static_cast<int>(signedVal0);
  int expectedBin0 = ((pixelValue0 - nLowerLevel[0]) * (nLevels[0] - 1)) / (nUpperLevel[0] - nLowerLevel[0]);
  if (expectedBin0 >= nLevels[0] - 1) expectedBin0 = nLevels[0] - 2;
  if (expectedBin0 < 0) expectedBin0 = 0;
  EXPECT_EQ(histData0[expectedBin0], width * height) << "Channel 0: All pixels with value -100 should be in bin " << expectedBin0;

  // Channel 1: value 0
  int expectedBin1 = ((0 - nLowerLevel[1]) * (nLevels[1] - 1)) / (nUpperLevel[1] - nLowerLevel[1]);
  if (expectedBin1 >= nLevels[1] - 1) expectedBin1 = nLevels[1] - 2;
  EXPECT_EQ(histData1[expectedBin1], width * height) << "Channel 1: All pixels with value 0 should be in bin " << expectedBin1;

  // Channel 2: value 100
  int expectedBin2 = ((100 - nLowerLevel[2]) * (nLevels[2] - 1)) / (nUpperLevel[2] - nLowerLevel[2]);
  if (expectedBin2 >= nLevels[2] - 1) expectedBin2 = nLevels[2] - 2;
  EXPECT_EQ(histData2[expectedBin2], width * height) << "Channel 2: All pixels with value 100 should be in bin " << expectedBin2;

  // Channel 3: value 200
  int expectedBin3 = ((200 - nLowerLevel[3]) * (nLevels[3] - 1)) / (nUpperLevel[3] - nLowerLevel[3]);
  if (expectedBin3 >= nLevels[3] - 1) expectedBin3 = nLevels[3] - 2;
  EXPECT_EQ(histData3[expectedBin3], width * height) << "Channel 3: All pixels with value 200 should be in bin " << expectedBin3;

  // Verify total counts for all channels
  int totalCount0 = 0, totalCount1 = 0, totalCount2 = 0, totalCount3 = 0;
  for (int i = 0; i < nLevels[0] - 1; i++) totalCount0 += histData0[i];
  for (int i = 0; i < nLevels[1] - 1; i++) totalCount1 += histData1[i];
  for (int i = 0; i < nLevels[2] - 1; i++) totalCount2 += histData2[i];
  for (int i = 0; i < nLevels[3] - 1; i++) totalCount3 += histData3[i];

  EXPECT_EQ(totalCount0, width * height) << "Channel 0: Total count should equal number of pixels";
  EXPECT_EQ(totalCount1, width * height) << "Channel 1: Total count should equal number of pixels";
  EXPECT_EQ(totalCount2, width * height) << "Channel 2: Total count should equal number of pixels";
  EXPECT_EQ(totalCount3, width * height) << "Channel 3: Total count should equal number of pixels";
}

/**
 * @brief Test nppiHistogramEven_16s_C4R_Ctx with stream context
 *
 * Verifies the context-aware version for signed 16-bit data.
 * Note: Using Npp16u with signed interpretation since Npp16s doesn't support C4.
 */
TEST_F(NPPIHistogramEvenC4RTest, HistogramEven_16s_C4R_Ctx) {
  size_t dataSize = width * height * 4;

  std::vector<Npp16u> srcData(dataSize);
  for (size_t i = 0; i < dataSize; i += 4) {
    srcData[i + 0] = static_cast<Npp16u>(static_cast<Npp16s>(-100));
    srcData[i + 1] = 0;
    srcData[i + 2] = 100;
    srcData[i + 3] = 200;
  }

  NppImageMemory<Npp16u> src(width, height, 4);
  src.copyFromHost(srcData);

  // Ensure data is copied before histogram computation
  cudaDeviceSynchronize();

  int nLevels[4] = {257, 257, 257, 257};  // nLevels must be nBins + 1
  Npp32s nLowerLevel[4] = {-256, -256, -256, -256};
  Npp32s nUpperLevel[4] = {256, 256, 256, 256};

  DeviceMemory<Npp32s> hist0, hist1, hist2, hist3;
  hist0.allocate(nLevels[0] - 1);
  hist1.allocate(nLevels[1] - 1);
  hist2.allocate(nLevels[2] - 1);
  hist3.allocate(nLevels[3] - 1);

  Npp32s *pHist[4] = {hist0.get(), hist1.get(), hist2.get(), hist3.get()};

  SIZE_TYPE bufferSize = 0;
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;

  NppStatus status = nppiHistogramEvenGetBufferSize_16s_C4R_Ctx(roi, nLevels, &bufferSize, nppStreamCtx);
  EXPECT_EQ(status, NPP_SUCCESS);

  DeviceMemory<Npp8u> deviceBuffer;
  deviceBuffer.allocate(bufferSize);

  status = nppiHistogramEven_16s_C4R_Ctx(
      reinterpret_cast<const Npp16s*>(src.get()), src.step(),
      roi, pHist, nLevels, nLowerLevel, nUpperLevel,
      deviceBuffer.get(), nppStreamCtx
  );

  EXPECT_EQ(status, NPP_SUCCESS);

  // Synchronize before reading results (NPP functions are asynchronous)
  cudaDeviceSynchronize();

  // Verify histogram results for all channels
  std::vector<Npp32s> histData0, histData1, histData2, histData3;
  hist0.copyToHost(histData0);
  hist1.copyToHost(histData1);
  hist2.copyToHost(histData2);
  hist3.copyToHost(histData3);

  // Channel 0: value -100
  Npp16s signedVal0 = -100;
  int pixelValue0 = static_cast<int>(signedVal0);
  int expectedBin0 = ((pixelValue0 - nLowerLevel[0]) * (nLevels[0] - 1)) / (nUpperLevel[0] - nLowerLevel[0]);
  if (expectedBin0 >= nLevels[0] - 1) expectedBin0 = nLevels[0] - 2;
  if (expectedBin0 < 0) expectedBin0 = 0;
  EXPECT_EQ(histData0[expectedBin0], width * height) << "Channel 0: All pixels with value -100 should be in bin " << expectedBin0;

  // Channel 1: value 0
  int expectedBin1 = ((0 - nLowerLevel[1]) * (nLevels[1] - 1)) / (nUpperLevel[1] - nLowerLevel[1]);
  if (expectedBin1 >= nLevels[1] - 1) expectedBin1 = nLevels[1] - 2;
  EXPECT_EQ(histData1[expectedBin1], width * height) << "Channel 1: All pixels with value 0 should be in bin " << expectedBin1;

  // Channel 2: value 100
  int expectedBin2 = ((100 - nLowerLevel[2]) * (nLevels[2] - 1)) / (nUpperLevel[2] - nLowerLevel[2]);
  if (expectedBin2 >= nLevels[2] - 1) expectedBin2 = nLevels[2] - 2;
  EXPECT_EQ(histData2[expectedBin2], width * height) << "Channel 2: All pixels with value 100 should be in bin " << expectedBin2;

  // Channel 3: value 200
  int expectedBin3 = ((200 - nLowerLevel[3]) * (nLevels[3] - 1)) / (nUpperLevel[3] - nLowerLevel[3]);
  if (expectedBin3 >= nLevels[3] - 1) expectedBin3 = nLevels[3] - 2;
  EXPECT_EQ(histData3[expectedBin3], width * height) << "Channel 3: All pixels with value 200 should be in bin " << expectedBin3;

  // Verify total counts for all channels
  int totalCount0 = 0, totalCount1 = 0, totalCount2 = 0, totalCount3 = 0;
  for (int i = 0; i < nLevels[0] - 1; i++) totalCount0 += histData0[i];
  for (int i = 0; i < nLevels[1] - 1; i++) totalCount1 += histData1[i];
  for (int i = 0; i < nLevels[2] - 1; i++) totalCount2 += histData2[i];
  for (int i = 0; i < nLevels[3] - 1; i++) totalCount3 += histData3[i];

  EXPECT_EQ(totalCount0, width * height) << "Channel 0: Total count should equal number of pixels";
  EXPECT_EQ(totalCount1, width * height) << "Channel 1: Total count should equal number of pixels";
  EXPECT_EQ(totalCount2, width * height) << "Channel 2: Total count should equal number of pixels";
  EXPECT_EQ(totalCount3, width * height) << "Channel 3: Total count should equal number of pixels";

  // Verify buffer size consistency between regular and Ctx versions
  SIZE_TYPE bufferSizeRegular = 0;
  NppStatus statusRegular = nppiHistogramEvenGetBufferSize_16s_C4R(roi, nLevels, &bufferSizeRegular);
  EXPECT_EQ(statusRegular, NPP_SUCCESS);
  EXPECT_EQ(bufferSize, bufferSizeRegular) << "Buffer sizes should match between Ctx and regular versions";
}
