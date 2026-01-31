#include "npp.h"
#include "framework/npp_test_base.h"
#include <gtest/gtest.h>
#include <vector>

using namespace npp_functional_test;

class NPPISwapChannelsTest : public NppTestBase {
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

// C4R tests (4 channel to 4 channel)
TEST_F(NPPISwapChannelsTest, SwapChannels_8u_C4R) {
  size_t dataSize = width * height * 4;
  std::vector<Npp8u> srcData(dataSize);
  for (size_t i = 0; i < dataSize; i += 4) {
    srcData[i + 0] = 10;
    srcData[i + 1] = 20;
    srcData[i + 2] = 30;
    srcData[i + 3] = 40;
  }

  NppImageMemory<Npp8u> src(width, height, 4);
  NppImageMemory<Npp8u> dst(width, height, 4);
  src.copyFromHost(srcData);

  int aDstOrder[4] = {2, 1, 0, 3};
  NppStatus status = nppiSwapChannels_8u_C4R(
      src.get(), src.step(), dst.get(), dst.step(), roi, aDstOrder);

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> dstData;
  dst.copyToHost(dstData);
  EXPECT_EQ(dstData[0], 30);
  EXPECT_EQ(dstData[1], 20);
  EXPECT_EQ(dstData[2], 10);
  EXPECT_EQ(dstData[3], 40);
}

TEST_F(NPPISwapChannelsTest, SwapChannels_8u_C4R_Ctx) {
  size_t dataSize = width * height * 4;
  std::vector<Npp8u> srcData(dataSize);
  for (size_t i = 0; i < dataSize; i += 4) {
    srcData[i + 0] = 10;
    srcData[i + 1] = 20;
    srcData[i + 2] = 30;
    srcData[i + 3] = 40;
  }

  NppImageMemory<Npp8u> src(width, height, 4);
  NppImageMemory<Npp8u> dst(width, height, 4);
  src.copyFromHost(srcData);

  int aDstOrder[4] = {2, 1, 0, 3};
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;

  NppStatus status = nppiSwapChannels_8u_C4R_Ctx(
      src.get(), src.step(), dst.get(), dst.step(), roi, aDstOrder, nppStreamCtx);

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> dstData;
  dst.copyToHost(dstData);
  EXPECT_EQ(dstData[0], 30);
  EXPECT_EQ(dstData[1], 20);
  EXPECT_EQ(dstData[2], 10);
  EXPECT_EQ(dstData[3], 40);
}

// C4IR tests (4 channel in-place)
TEST_F(NPPISwapChannelsTest, SwapChannels_8u_C4IR) {
  size_t dataSize = width * height * 4;
  std::vector<Npp8u> srcData(dataSize);
  for (size_t i = 0; i < dataSize; i += 4) {
    srcData[i + 0] = 10;
    srcData[i + 1] = 20;
    srcData[i + 2] = 30;
    srcData[i + 3] = 40;
  }

  NppImageMemory<Npp8u> srcDst(width, height, 4);
  srcDst.copyFromHost(srcData);

  int aDstOrder[4] = {2, 1, 0, 3};
  NppStatus status = nppiSwapChannels_8u_C4IR(srcDst.get(), srcDst.step(), roi, aDstOrder);

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> dstData;
  srcDst.copyToHost(dstData);
  EXPECT_EQ(dstData[0], 30);
  EXPECT_EQ(dstData[1], 20);
  EXPECT_EQ(dstData[2], 10);
  EXPECT_EQ(dstData[3], 40);
}

TEST_F(NPPISwapChannelsTest, SwapChannels_8u_C4IR_Ctx) {
  size_t dataSize = width * height * 4;
  std::vector<Npp8u> srcData(dataSize);
  for (size_t i = 0; i < dataSize; i += 4) {
    srcData[i + 0] = 10;
    srcData[i + 1] = 20;
    srcData[i + 2] = 30;
    srcData[i + 3] = 40;
  }

  NppImageMemory<Npp8u> srcDst(width, height, 4);
  srcDst.copyFromHost(srcData);

  int aDstOrder[4] = {2, 1, 0, 3};
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;

  NppStatus status = nppiSwapChannels_8u_C4IR_Ctx(
      srcDst.get(), srcDst.step(), roi, aDstOrder, nppStreamCtx);

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> dstData;
  srcDst.copyToHost(dstData);
  EXPECT_EQ(dstData[0], 30);
  EXPECT_EQ(dstData[1], 20);
  EXPECT_EQ(dstData[2], 10);
  EXPECT_EQ(dstData[3], 40);
}

// C3R tests (3 channel to 3 channel)
TEST_F(NPPISwapChannelsTest, SwapChannels_8u_C3R) {
  size_t dataSize = width * height * 3;
  std::vector<Npp8u> srcData(dataSize);
  for (size_t i = 0; i < dataSize; i += 3) {
    srcData[i + 0] = 11;
    srcData[i + 1] = 22;
    srcData[i + 2] = 33;
  }

  NppImageMemory<Npp8u> src(width, height, 3);
  NppImageMemory<Npp8u> dst(width, height, 3);
  src.copyFromHost(srcData);

  int aDstOrder[3] = {2, 1, 0};
  NppStatus status = nppiSwapChannels_8u_C3R(
      src.get(), src.step(), dst.get(), dst.step(), roi, aDstOrder);

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> dstData;
  dst.copyToHost(dstData);
  EXPECT_EQ(dstData[0], 33);
  EXPECT_EQ(dstData[1], 22);
  EXPECT_EQ(dstData[2], 11);
}

// C3IR tests (3 channel in-place)
TEST_F(NPPISwapChannelsTest, SwapChannels_16u_C3IR) {
  size_t dataSize = width * height * 3;
  std::vector<Npp16u> srcData(dataSize);
  for (size_t i = 0; i < dataSize; i += 3) {
    srcData[i + 0] = 1000;
    srcData[i + 1] = 2000;
    srcData[i + 2] = 3000;
  }

  NppImageMemory<Npp16u> srcDst(width, height, 3);
  srcDst.copyFromHost(srcData);

  int aDstOrder[3] = {2, 1, 0};
  NppStatus status = nppiSwapChannels_16u_C3IR(srcDst.get(), srcDst.step(), roi, aDstOrder);

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16u> dstData;
  srcDst.copyToHost(dstData);
  EXPECT_EQ(dstData[0], 3000);
  EXPECT_EQ(dstData[1], 2000);
  EXPECT_EQ(dstData[2], 1000);
}

// C4C3R tests (4 channel to 3 channel)
TEST_F(NPPISwapChannelsTest, SwapChannels_32f_C4C3R) {
  size_t srcSize = width * height * 4;
  std::vector<Npp32f> srcData(srcSize);
  for (size_t i = 0; i < srcSize; i += 4) {
    srcData[i + 0] = 1.0f;
    srcData[i + 1] = 2.0f;
    srcData[i + 2] = 3.0f;
    srcData[i + 3] = 4.0f;
  }

  NppImageMemory<Npp32f> src(width, height, 4);
  NppImageMemory<Npp32f> dst(width, height, 3);
  src.copyFromHost(srcData);

  int aDstOrder[3] = {2, 1, 0};
  NppStatus status = nppiSwapChannels_32f_C4C3R(
      src.get(), src.step(), dst.get(), dst.step(), roi, aDstOrder);

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> dstData;
  dst.copyToHost(dstData);
  EXPECT_FLOAT_EQ(dstData[0], 3.0f);
  EXPECT_FLOAT_EQ(dstData[1], 2.0f);
  EXPECT_FLOAT_EQ(dstData[2], 1.0f);
}

// C3C4R tests (3 channel to 4 channel with fill value)
TEST_F(NPPISwapChannelsTest, SwapChannels_8u_C3C4R) {
  size_t srcSize = width * height * 3;
  std::vector<Npp8u> srcData(srcSize);
  for (size_t i = 0; i < srcSize; i += 3) {
    srcData[i + 0] = 11;
    srcData[i + 1] = 22;
    srcData[i + 2] = 33;
  }

  NppImageMemory<Npp8u> src(width, height, 3);
  NppImageMemory<Npp8u> dst(width, height, 4);
  src.copyFromHost(srcData);

  int aDstOrder[4] = {2, 1, 0, 3};
  Npp8u fillValue = 255;
  NppStatus status = nppiSwapChannels_8u_C3C4R(
      src.get(), src.step(), dst.get(), dst.step(), roi, aDstOrder, fillValue);

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> dstData;
  dst.copyToHost(dstData);
  EXPECT_EQ(dstData[0], 33);
  EXPECT_EQ(dstData[1], 22);
  EXPECT_EQ(dstData[2], 11);
  EXPECT_EQ(dstData[3], 255);
}

// AC4R tests (4 channel with alpha preserved)
TEST_F(NPPISwapChannelsTest, SwapChannels_32s_AC4R) {
  size_t dataSize = width * height * 4;
  std::vector<Npp32s> srcData(dataSize);
  for (size_t i = 0; i < dataSize; i += 4) {
    srcData[i + 0] = 100;
    srcData[i + 1] = 200;
    srcData[i + 2] = 300;
    srcData[i + 3] = 999;
  }

  NppImageMemory<Npp32s> src(width, height, 4);
  NppImageMemory<Npp32s> dst(width, height, 4);
  src.copyFromHost(srcData);

  int aDstOrder[3] = {2, 1, 0};
  NppStatus status = nppiSwapChannels_32s_AC4R(
      src.get(), src.step(), dst.get(), dst.step(), roi, aDstOrder);

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32s> dstData;
  dst.copyToHost(dstData);
  EXPECT_EQ(dstData[0], 300);
  EXPECT_EQ(dstData[1], 200);
  EXPECT_EQ(dstData[2], 100);
  // AC4R preserves alpha channel
  EXPECT_EQ(dstData[3], 999);
}

// Test multiple data types - use 16u instead of 16s for C4
TEST_F(NPPISwapChannelsTest, SwapChannels_16u_C4R) {
  size_t dataSize = width * height * 4;
  std::vector<Npp16u> srcData(dataSize);
  for (size_t i = 0; i < dataSize; i += 4) {
    srcData[i + 0] = 100;
    srcData[i + 1] = 200;
    srcData[i + 2] = 300;
    srcData[i + 3] = 400;
  }

  NppImageMemory<Npp16u> src(width, height, 4);
  NppImageMemory<Npp16u> dst(width, height, 4);
  src.copyFromHost(srcData);

  int aDstOrder[4] = {3, 2, 1, 0};
  NppStatus status = nppiSwapChannels_16u_C4R(
      src.get(), src.step(), dst.get(), dst.step(), roi, aDstOrder);

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16u> dstData;
  dst.copyToHost(dstData);
  EXPECT_EQ(dstData[0], 400);
  EXPECT_EQ(dstData[1], 300);
  EXPECT_EQ(dstData[2], 200);
  EXPECT_EQ(dstData[3], 100);
}

TEST_F(NPPISwapChannelsTest, SwapChannels_32f_C3IR) {
  size_t dataSize = width * height * 3;
  std::vector<Npp32f> srcData(dataSize);
  for (size_t i = 0; i < dataSize; i += 3) {
    srcData[i + 0] = 0.1f;
    srcData[i + 1] = 0.5f;
    srcData[i + 2] = 0.9f;
  }

  NppImageMemory<Npp32f> srcDst(width, height, 3);
  srcDst.copyFromHost(srcData);

  int aDstOrder[3] = {1, 0, 2};
  NppStatus status = nppiSwapChannels_32f_C3IR(srcDst.get(), srcDst.step(), roi, aDstOrder);

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> dstData;
  srcDst.copyToHost(dstData);
  EXPECT_FLOAT_EQ(dstData[0], 0.5f);
  EXPECT_FLOAT_EQ(dstData[1], 0.1f);
  EXPECT_FLOAT_EQ(dstData[2], 0.9f);
}
