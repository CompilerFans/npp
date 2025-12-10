#include "npp.h"
#include "framework/npp_test_base.h"
#include <gtest/gtest.h>
#include <vector>

using namespace npp_functional_test;

/**
 * @brief NPP CopyConstBorder Tests
 *
 * Tests for nppiCopyConstBorder_8u_C1R family of functions.
 * These tests verify copying image data with constant value borders.
 */
class NPPICopyConstBorderTest : public NppTestBase {
protected:
  void SetUp() override {
    NppTestBase::SetUp();
    width = 4;
    height = 4;
    srcRoi.width = width;
    srcRoi.height = height;
  }

  int width, height;
  NppiSize srcRoi;
};

/**
 * @brief Test nppiCopyConstBorder_8u_C1R basic functionality
 *
 * Verifies basic border copying with constant value.
 */
TEST_F(NPPICopyConstBorderTest, CopyConstBorder_8u_C1R) {
  size_t srcDataSize = width * height;

  // Generate simple test pattern
  std::vector<Npp8u> srcData(srcDataSize);
  for (size_t i = 0; i < srcDataSize; i++) {
    srcData[i] = static_cast<Npp8u>((i * 10) % 256);
  }

  // Destination with borders: 2 pixels on each side
  int topBorder = 2;
  int leftBorder = 2;
  int dstWidth = width + leftBorder + 2;   // left + right
  int dstHeight = height + topBorder + 2;  // top + bottom
  NppiSize dstRoi = {dstWidth, dstHeight};
  Npp8u borderValue = 255;

  NppImageMemory<Npp8u> src(width, height, 1);
  NppImageMemory<Npp8u> dst(dstWidth, dstHeight, 1);

  src.copyFromHost(srcData);

  NppStatus status = nppiCopyConstBorder_8u_C1R(
      src.get(), src.step(), srcRoi,
      dst.get(), dst.step(), dstRoi,
      topBorder, leftBorder, borderValue
  );

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> dstData;
  dst.copyToHost(dstData);

  // Verify border pixels (top-left corner should be border value)
  EXPECT_EQ(dstData[0], borderValue);
  EXPECT_EQ(dstData[1], borderValue);

  // Verify center pixel (should be from source)
  int centerIdx = (topBorder + 1) * dstWidth + (leftBorder + 1);
  int srcCenterIdx = 1 * width + 1;
  EXPECT_EQ(dstData[centerIdx], srcData[srcCenterIdx]);
}

/**
 * @brief Test nppiCopyConstBorder_8u_C1R_Ctx with stream context
 *
 * Verifies the context-aware version that supports custom CUDA streams.
 */
TEST_F(NPPICopyConstBorderTest, CopyConstBorder_8u_C1R_Ctx) {
  size_t srcDataSize = width * height;

  std::vector<Npp8u> srcData(srcDataSize);
  for (size_t i = 0; i < srcDataSize; i++) {
    srcData[i] = static_cast<Npp8u>((i * 10) % 256);
  }

  int topBorder = 2;
  int leftBorder = 2;
  int dstWidth = width + leftBorder + 2;
  int dstHeight = height + topBorder + 2;
  NppiSize dstRoi = {dstWidth, dstHeight};
  Npp8u borderValue = 255;

  NppImageMemory<Npp8u> src(width, height, 1);
  NppImageMemory<Npp8u> dst(dstWidth, dstHeight, 1);

  src.copyFromHost(srcData);

  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;

  NppStatus status = nppiCopyConstBorder_8u_C1R_Ctx(
      src.get(), src.step(), srcRoi,
      dst.get(), dst.step(), dstRoi,
      topBorder, leftBorder, borderValue,
      nppStreamCtx
  );

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> dstData;
  dst.copyToHost(dstData);

  // Verify border pixels
  EXPECT_EQ(dstData[0], borderValue);
  EXPECT_EQ(dstData[1], borderValue);

  // Verify center pixel
  int centerIdx = (topBorder + 1) * dstWidth + (leftBorder + 1);
  int srcCenterIdx = 1 * width + 1;
  EXPECT_EQ(dstData[centerIdx], srcData[srcCenterIdx]);
}
