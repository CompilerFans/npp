#include "npp.h"
#include "framework/npp_test_base.h"
#include <gtest/gtest.h>
#include <vector>

using namespace npp_functional_test;

/**
 * @brief NPP SwapChannels Tests
 *
 * Tests for nppiSwapChannels_8u_C4R and nppiSwapChannels_8u_C4IR families.
 * These tests verify basic channel swapping operations on 4-channel images.
 */
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

/**
 * @brief Test nppiSwapChannels_8u_C4R basic functionality
 *
 * Verifies RGBA to BGRA conversion (swap red and blue channels).
 */
TEST_F(NPPISwapChannelsTest, SwapChannels_8u_C4R) {
  size_t dataSize = width * height * 4;

  // Generate simple test pattern with identifiable channel values
  std::vector<Npp8u> srcData(dataSize);
  for (size_t i = 0; i < dataSize; i += 4) {
    srcData[i + 0] = 10;  // R
    srcData[i + 1] = 20;  // G
    srcData[i + 2] = 30;  // B
    srcData[i + 3] = 40;  // A
  }

  NppImageMemory<Npp8u> src(width, height, 4);
  NppImageMemory<Npp8u> dst(width, height, 4);

  src.copyFromHost(srcData);

  // Swap R and B channels: RGBA -> BGRA
  int aDstOrder[4] = {2, 1, 0, 3};

  NppStatus status = nppiSwapChannels_8u_C4R(
      src.get(), src.step(),
      dst.get(), dst.step(),
      roi, aDstOrder
  );

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> dstData;
  dst.copyToHost(dstData);

  // Verify first pixel: should be BGRA = [30, 20, 10, 40]
  EXPECT_EQ(dstData[0], 30);  // B
  EXPECT_EQ(dstData[1], 20);  // G
  EXPECT_EQ(dstData[2], 10);  // R
  EXPECT_EQ(dstData[3], 40);  // A
}

/**
 * @brief Test nppiSwapChannels_8u_C4R_Ctx with stream context
 *
 * Verifies the context-aware version that supports custom CUDA streams.
 */
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
      src.get(), src.step(),
      dst.get(), dst.step(),
      roi, aDstOrder, nppStreamCtx
  );

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> dstData;
  dst.copyToHost(dstData);

  EXPECT_EQ(dstData[0], 30);
  EXPECT_EQ(dstData[1], 20);
  EXPECT_EQ(dstData[2], 10);
  EXPECT_EQ(dstData[3], 40);
}

/**
 * @brief Test nppiSwapChannels_8u_C4IR in-place operation
 *
 * Verifies in-place channel swapping where source and destination are the same buffer.
 */
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

  NppStatus status = nppiSwapChannels_8u_C4IR(
      srcDst.get(), srcDst.step(),
      roi, aDstOrder
  );

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> dstData;
  srcDst.copyToHost(dstData);

  EXPECT_EQ(dstData[0], 30);
  EXPECT_EQ(dstData[1], 20);
  EXPECT_EQ(dstData[2], 10);
  EXPECT_EQ(dstData[3], 40);
}

/**
 * @brief Test nppiSwapChannels_8u_C4IR_Ctx with stream context
 *
 * Verifies the in-place context-aware version that supports custom CUDA streams.
 */
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
      srcDst.get(), srcDst.step(),
      roi, aDstOrder, nppStreamCtx
  );

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> dstData;
  srcDst.copyToHost(dstData);

  EXPECT_EQ(dstData[0], 30);
  EXPECT_EQ(dstData[1], 20);
  EXPECT_EQ(dstData[2], 10);
  EXPECT_EQ(dstData[3], 40);
}
