#include "npp_test_base.h"

using namespace npp_functional_test;

class ResizeC1RPrecisionTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }
  void TearDown() override { NppTestBase::TearDown(); }
};

// Nearest Neighbor C1R precision tests

TEST_F(ResizeC1RPrecisionTest, NN_8u_C1R_ExactUpsample2x) {
  const int srcWidth = 4, srcHeight = 4;
  const int dstWidth = 8, dstHeight = 8;

  std::vector<Npp8u> srcData(srcWidth * srcHeight);

  // Fill with distinct values
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      srcData[y * srcWidth + x] = (Npp8u)((y * srcWidth + x) * 15);
    }
  }

  NppImageMemory<Npp8u> src(srcWidth, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status =
      nppiResize_8u_C1R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI, NPPI_INTER_NN);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight);
  dst.copyToHost(resultData);

  // Verify corner mapping
  EXPECT_EQ(resultData[0], srcData[0]) << "Top-left corner should preserve value";
  EXPECT_EQ(resultData[dstWidth - 1], srcData[srcWidth - 1]) << "Top-right corner";
  EXPECT_EQ(resultData[(dstHeight - 1) * dstWidth], srcData[(srcHeight - 1) * srcWidth]) << "Bottom-left corner";
}

TEST_F(ResizeC1RPrecisionTest, NN_8u_C1R_BlockPattern) {
  const int srcWidth = 4, srcHeight = 4;
  const int dstWidth = 8, dstHeight = 8;

  std::vector<Npp8u> srcData(srcWidth * srcHeight);

  // Create 2x2 blocks with values 50, 100, 150, 200
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int block_id = (y / 2) * 2 + (x / 2);
      srcData[y * srcWidth + x] = (Npp8u)(block_id * 50 + 50);
    }
  }

  NppImageMemory<Npp8u> src(srcWidth, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status =
      nppiResize_8u_C1R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI, NPPI_INTER_NN);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight);
  dst.copyToHost(resultData);

  // All values should be one of the block values (no interpolation)
  for (int i = 0; i < dstWidth * dstHeight; i++) {
    Npp8u val = resultData[i];
    EXPECT_TRUE(val == 50 || val == 100 || val == 150 || val == 200)
        << "NN should only select source values, not interpolate. Got: " << (int)val;
  }
}

TEST_F(ResizeC1RPrecisionTest, NN_8u_C1R_Downsampling) {
  const int srcWidth = 8, srcHeight = 8;
  const int dstWidth = 4, dstHeight = 4;

  std::vector<Npp8u> srcData(srcWidth * srcHeight);

  // Create checkerboard pattern
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      srcData[y * srcWidth + x] = ((x + y) % 2) ? 200 : 80;
    }
  }

  NppImageMemory<Npp8u> src(srcWidth, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status =
      nppiResize_8u_C1R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI, NPPI_INTER_NN);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight);
  dst.copyToHost(resultData);

  // All values should be either 80 or 200 (no averaging)
  for (int i = 0; i < dstWidth * dstHeight; i++) {
    Npp8u val = resultData[i];
    EXPECT_TRUE(val == 80 || val == 200) << "NN downsampling should pick nearest pixel, not average. Got: " << (int)val;
  }
}

// Linear Interpolation C1R precision tests

TEST_F(ResizeC1RPrecisionTest, Linear_8u_C1R_CornerPreservation) {
  const int srcWidth = 4, srcHeight = 4;
  const int dstWidth = 8, dstHeight = 8;

  std::vector<Npp8u> srcData(srcWidth * srcHeight);

  // Create known corner values
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      srcData[y * srcWidth + x] = (Npp8u)(x * 60 + y * 15);
    }
  }

  NppImageMemory<Npp8u> src(srcWidth, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C1R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_LINEAR);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight);
  dst.copyToHost(resultData);

  // Corners should approximately preserve source values
  EXPECT_NEAR(resultData[0], srcData[0], 5) << "Top-left corner";
  EXPECT_NEAR(resultData[dstWidth - 1], srcData[srcWidth - 1], 5) << "Top-right corner";
  EXPECT_NEAR(resultData[(dstHeight - 1) * dstWidth], srcData[(srcHeight - 1) * srcWidth], 5) << "Bottom-left";
  EXPECT_NEAR(resultData[(dstHeight - 1) * dstWidth + (dstWidth - 1)],
              srcData[(srcHeight - 1) * srcWidth + (srcWidth - 1)], 5)
      << "Bottom-right";
}

TEST_F(ResizeC1RPrecisionTest, Linear_8u_C1R_HorizontalGradient) {
  const int srcWidth = 4, srcHeight = 4;
  const int dstWidth = 8, dstHeight = 4;

  std::vector<Npp8u> srcData(srcWidth * srcHeight);

  // Create horizontal gradient: 0, 85, 170, 255
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      srcData[y * srcWidth + x] = (Npp8u)(x * 255 / (srcWidth - 1));
    }
  }

  NppImageMemory<Npp8u> src(srcWidth, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C1R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_LINEAR);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight);
  dst.copyToHost(resultData);

  // Verify monotonic horizontal gradient in each row
  for (int y = 0; y < dstHeight; y++) {
    for (int x = 1; x < dstWidth - 1; x++) {
      int idx = y * dstWidth + x;
      int prev_idx = y * dstWidth + (x - 1);
      EXPECT_GE(resultData[idx], resultData[prev_idx] - 2)
          << "Horizontal gradient should be monotonic at (" << x << "," << y << ")";
    }
  }
}

TEST_F(ResizeC1RPrecisionTest, Linear_8u_C1R_VerticalGradient) {
  const int srcWidth = 4, srcHeight = 4;
  const int dstWidth = 4, dstHeight = 8;

  std::vector<Npp8u> srcData(srcWidth * srcHeight);

  // Create vertical gradient
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      srcData[y * srcWidth + x] = (Npp8u)(y * 255 / (srcHeight - 1));
    }
  }

  NppImageMemory<Npp8u> src(srcWidth, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C1R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_LINEAR);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight);
  dst.copyToHost(resultData);

  // Verify monotonic vertical gradient in each column
  for (int x = 0; x < dstWidth; x++) {
    for (int y = 1; y < dstHeight - 1; y++) {
      int idx = y * dstWidth + x;
      int prev_idx = (y - 1) * dstWidth + x;
      EXPECT_GE(resultData[idx], resultData[prev_idx] - 2)
          << "Vertical gradient should be monotonic at (" << x << "," << y << ")";
    }
  }
}

TEST_F(ResizeC1RPrecisionTest, Linear_8u_C1R_StepFunctionInterpolation) {
  const int srcWidth = 4, srcHeight = 4;
  const int dstWidth = 7, dstHeight = 4;

  std::vector<Npp8u> srcData(srcWidth * srcHeight);

  // Create step function: left half 60, right half 200
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      srcData[y * srcWidth + x] = (x < srcWidth / 2) ? 60 : 200;
    }
  }

  NppImageMemory<Npp8u> src(srcWidth, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C1R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_LINEAR);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight);
  dst.copyToHost(resultData);

  // Left side should be close to 60, right side close to 200
  for (int y = 0; y < dstHeight; y++) {
    EXPECT_NEAR(resultData[y * dstWidth + 0], 60, 15) << "Left edge at y=" << y;
    EXPECT_NEAR(resultData[y * dstWidth + (dstWidth - 1)], 200, 15) << "Right edge at y=" << y;

    // Middle should show interpolation
    int mid_idx = y * dstWidth + (dstWidth / 2);
    EXPECT_GT(resultData[mid_idx], 60) << "Middle should be interpolated at y=" << y;
    EXPECT_LT(resultData[mid_idx], 200) << "Middle should be interpolated at y=" << y;
  }
}
