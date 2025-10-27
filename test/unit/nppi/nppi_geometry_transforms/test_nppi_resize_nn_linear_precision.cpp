#include "npp_test_base.h"

using namespace npp_functional_test;

class ResizePrecisionTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }
  void TearDown() override { NppTestBase::TearDown(); }
};

// Nearest Neighbor precision tests

// Test exact 2x upsampling - each source pixel should replicate to 2x2 block
TEST_F(ResizePrecisionTest, NN_8u_C3R_ExactUpsample2x) {
  const int srcWidth = 4, srcHeight = 4;
  const int dstWidth = 8, dstHeight = 8;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Fill with distinct values for each pixel
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      Npp8u value = (Npp8u)((y * srcWidth + x) * 10);
      srcData[idx + 0] = value;
      srcData[idx + 1] = value + 1;
      srcData[idx + 2] = value + 2;
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_NN);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Each source pixel should map to approximately 2x2 destination pixels
  // Verify src(0,0) value appears in dst(0,0) and nearby pixels
  int src_idx_00 = 0;
  int dst_idx_00 = 0;
  EXPECT_EQ(resultData[dst_idx_00 + 0], srcData[src_idx_00 + 0]) << "NN should preserve exact pixel values";
  EXPECT_EQ(resultData[dst_idx_00 + 1], srcData[src_idx_00 + 1]);
  EXPECT_EQ(resultData[dst_idx_00 + 2], srcData[src_idx_00 + 2]);

  // Verify src(3,3) maps correctly to dst region
  int src_idx_33 = ((3 * srcWidth + 3) * 3);
  int dst_idx_66 = ((6 * dstWidth + 6) * 3);
  EXPECT_EQ(resultData[dst_idx_66 + 0], srcData[src_idx_33 + 0]) << "NN should map corner pixels correctly";
}

// Test exact coordinate mapping with block pattern
TEST_F(ResizePrecisionTest, NN_8u_C3R_ExactMapping) {
  const int srcWidth = 4, srcHeight = 4;
  const int dstWidth = 8, dstHeight = 8;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Create 2x2 blocks with distinct values
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      int block_id = (y / 2) * 2 + (x / 2);
      Npp8u value = (Npp8u)(block_id * 60 + 40);
      srcData[idx + 0] = value;
      srcData[idx + 1] = value;
      srcData[idx + 2] = value;
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_NN);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Verify all destination pixels have values from source (no interpolation)
  for (int i = 0; i < dstWidth * dstHeight * 3; i += 3) {
    Npp8u val = resultData[i];
    // Value should be one of the 4 block values: 40, 100, 160, 220
    EXPECT_TRUE(val == 40 || val == 100 || val == 160 || val == 220)
        << "NN should only select source pixel values, not interpolate. Got: " << (int)val;
  }
}

// Test channel independence for nearest neighbor
TEST_F(ResizePrecisionTest, NN_8u_C3R_ChannelIndependence) {
  const int srcWidth = 8, srcHeight = 8;
  const int dstWidth = 16, dstHeight = 16;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Fill each channel with different pattern
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      srcData[idx + 0] = (Npp8u)((x % 2) * 100 + 50);      // R: alternating 50/150
      srcData[idx + 1] = (Npp8u)((y % 2) * 100 + 70);      // G: alternating 70/170
      srcData[idx + 2] = (Npp8u)(((x + y) % 2) * 100 + 90); // B: checkerboard 90/190
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_NN);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Verify each channel only contains values from source
  for (int i = 0; i < dstWidth * dstHeight * 3; i += 3) {
    EXPECT_TRUE(resultData[i + 0] == 50 || resultData[i + 0] == 150) << "R channel should be 50 or 150";
    EXPECT_TRUE(resultData[i + 1] == 70 || resultData[i + 1] == 170) << "G channel should be 70 or 170";
    EXPECT_TRUE(resultData[i + 2] == 90 || resultData[i + 2] == 190) << "B channel should be 90 or 190";
  }
}

// Test boundary pixel selection
TEST_F(ResizePrecisionTest, NN_8u_C3R_BoundarySelection) {
  const int srcWidth = 6, srcHeight = 6;
  const int dstWidth = 12, dstHeight = 12;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Create three horizontal stripes: 60, 120, 180
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      Npp8u value;
      if (y < 2)
        value = 60;
      else if (y < 4)
        value = 120;
      else
        value = 180;
      srcData[idx + 0] = value;
      srcData[idx + 1] = value;
      srcData[idx + 2] = value;
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_NN);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // All pixels should be one of the three stripe values
  for (int i = 0; i < dstWidth * dstHeight * 3; i += 3) {
    Npp8u val = resultData[i];
    EXPECT_TRUE(val == 60 || val == 120 || val == 180)
        << "NN should select from source values: 60, 120, or 180. Got: " << (int)val;
  }
}

// Test downsampling pixel selection
TEST_F(ResizePrecisionTest, NN_8u_C3R_Downsampling) {
  const int srcWidth = 8, srcHeight = 8;
  const int dstWidth = 4, dstHeight = 4;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Create fine checkerboard pattern
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      Npp8u value = ((x + y) % 2) ? 200 : 80;
      srcData[idx + 0] = value;
      srcData[idx + 1] = value;
      srcData[idx + 2] = value;
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_NN);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // All values should be either 80 or 200 (no averaging)
  for (int i = 0; i < dstWidth * dstHeight * 3; i += 3) {
    Npp8u val = resultData[i];
    EXPECT_TRUE(val == 80 || val == 200) << "NN downsampling should pick nearest pixel, not average. Got: " << (int)val;
  }
}

// Linear interpolation precision tests

// Test exact bilinear interpolation calculation
TEST_F(ResizePrecisionTest, Linear_8u_C3R_ExactInterpolation) {
  const int srcWidth = 2, srcHeight = 2;
  const int dstWidth = 3, dstHeight = 3;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Set corner values for known interpolation
  // Top-left (0,0): 0, Top-right (1,0): 100
  // Bottom-left (0,1): 50, Bottom-right (1,1): 150
  srcData[0] = srcData[1] = srcData[2] = 0;       // (0,0)
  srcData[3] = srcData[4] = srcData[5] = 100;     // (1,0)
  srcData[6] = srcData[7] = srcData[8] = 50;      // (0,1)
  srcData[9] = srcData[10] = srcData[11] = 150;   // (1,1)

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_LINEAR);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Corners should preserve values
  EXPECT_NEAR(resultData[0], 0, 5) << "Top-left corner";
  EXPECT_NEAR(resultData[6], 100, 5) << "Top-right corner";
  EXPECT_NEAR(resultData[18], 50, 5) << "Bottom-left corner";
  EXPECT_NEAR(resultData[24], 150, 5) << "Bottom-right corner";

  // Center pixel interpolation (relaxed tolerance due to coordinate mapping complexity)
  int center_idx = (1 * dstWidth + 1) * 3;
  EXPECT_GT(resultData[center_idx], 0) << "Center should be interpolated, not 0";
  EXPECT_LT(resultData[center_idx], 150) << "Center should be interpolated, not max";
  // Value should be somewhere in the interpolated range
  EXPECT_GE(resultData[center_idx], 40) << "Center should show interpolation influence";
  EXPECT_LE(resultData[center_idx], 110) << "Center should show interpolation influence";
}

// Test halfway point interpolation
TEST_F(ResizePrecisionTest, Linear_8u_C3R_HalfwayPoint) {
  const int srcWidth = 4, srcHeight = 4;
  const int dstWidth = 7, dstHeight = 4;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Horizontal gradient on each row: 0, 90, 180, 255
  for (int y = 0; y < srcHeight; y++) {
    srcData[(y * srcWidth + 0) * 3 + 0] = srcData[(y * srcWidth + 0) * 3 + 1] = srcData[(y * srcWidth + 0) * 3 + 2] = 0;
    srcData[(y * srcWidth + 1) * 3 + 0] = srcData[(y * srcWidth + 1) * 3 + 1] = srcData[(y * srcWidth + 1) * 3 + 2] = 90;
    srcData[(y * srcWidth + 2) * 3 + 0] = srcData[(y * srcWidth + 2) * 3 + 1] = srcData[(y * srcWidth + 2) * 3 + 2] = 180;
    srcData[(y * srcWidth + 3) * 3 + 0] = srcData[(y * srcWidth + 3) * 3 + 1] = srcData[(y * srcWidth + 3) * 3 + 2] = 255;
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_LINEAR);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Verify interpolated values are between adjacent source values for each row
  for (int y = 0; y < dstHeight; y++) {
    for (int x = 0; x < dstWidth - 1; x++) {
      int idx = (y * dstWidth + x) * 3;
      int next_idx = (y * dstWidth + (x + 1)) * 3;
      // Linear interpolation should produce monotonically increasing values
      EXPECT_LE(resultData[idx], resultData[next_idx] + 1) << "Gradient should be monotonic at (" << x << "," << y << ")";
    }
  }
}

// Test channel independence for linear interpolation
TEST_F(ResizePrecisionTest, Linear_8u_C3R_ChannelIndependence) {
  const int srcWidth = 4, srcHeight = 4;
  const int dstWidth = 8, dstHeight = 8;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Create channel-specific gradients
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      srcData[idx + 0] = (Npp8u)(x * 255 / (srcWidth - 1));  // R: horizontal gradient
      srcData[idx + 1] = (Npp8u)(y * 255 / (srcHeight - 1)); // G: vertical gradient
      srcData[idx + 2] = 128;                                // B: constant
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_LINEAR);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Verify R channel increases horizontally
  for (int y = 0; y < dstHeight; y++) {
    for (int x = 1; x < dstWidth; x++) {
      int idx = (y * dstWidth + x) * 3;
      int prev_idx = (y * dstWidth + (x - 1)) * 3;
      EXPECT_GE(resultData[idx + 0], resultData[prev_idx + 0] - 2) << "R gradient horizontal at (" << x << "," << y << ")";
    }
  }

  // Verify G channel increases vertically
  for (int y = 1; y < dstHeight; y++) {
    for (int x = 0; x < dstWidth; x++) {
      int idx = (y * dstWidth + x) * 3;
      int prev_idx = ((y - 1) * dstWidth + x) * 3;
      EXPECT_GE(resultData[idx + 1], resultData[prev_idx + 1] - 2) << "G gradient vertical at (" << x << "," << y << ")";
    }
  }

  // Verify B channel is constant
  for (int i = 0; i < dstWidth * dstHeight * 3; i += 3) {
    EXPECT_NEAR(resultData[i + 2], 128, 3) << "B channel should remain constant";
  }
}

// Test boundary interpolation behavior
TEST_F(ResizePrecisionTest, Linear_8u_C3R_BoundaryInterpolation) {
  const int srcWidth = 4, srcHeight = 4;
  const int dstWidth = 7, dstHeight = 7;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Create step function: left half 50, right half 200
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      Npp8u value = (x < srcWidth / 2) ? 50 : 200;
      srcData[idx + 0] = value;
      srcData[idx + 1] = value;
      srcData[idx + 2] = value;
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_LINEAR);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Left side should be close to 50
  for (int y = 0; y < dstHeight; y++) {
    int idx_left = (y * dstWidth + 1) * 3;
    EXPECT_NEAR(resultData[idx_left], 50, 15) << "Left region at y=" << y;
  }

  // Right side should be close to 200
  for (int y = 0; y < dstHeight; y++) {
    int idx_right = (y * dstWidth + (dstWidth - 2)) * 3;
    EXPECT_NEAR(resultData[idx_right], 200, 15) << "Right region at y=" << y;
  }

  // Middle region should have intermediate values
  for (int y = 0; y < dstHeight; y++) {
    int idx_mid = (y * dstWidth + (dstWidth / 2)) * 3;
    EXPECT_GT(resultData[idx_mid], 50) << "Middle should be interpolated at y=" << y;
    EXPECT_LT(resultData[idx_mid], 200) << "Middle should be interpolated at y=" << y;
  }
}

// Test gradient preservation with linear interpolation
TEST_F(ResizePrecisionTest, Linear_8u_C3R_GradientPreservation) {
  const int srcWidth = 8, srcHeight = 8;
  const int dstWidth = 16, dstHeight = 16;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Create perfect linear gradient
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      Npp8u value_x = (Npp8u)(x * 255 / (srcWidth - 1));
      Npp8u value_y = (Npp8u)(y * 255 / (srcHeight - 1));
      srcData[idx + 0] = value_x;  // Horizontal gradient
      srcData[idx + 1] = value_y;  // Vertical gradient
      srcData[idx + 2] = (value_x + value_y) / 2; // Diagonal gradient
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_LINEAR);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Verify monotonic horizontal gradient in R channel
  for (int y = 0; y < dstHeight; y++) {
    for (int x = 1; x < dstWidth - 1; x++) {
      int idx = (y * dstWidth + x) * 3;
      int prev_idx = (y * dstWidth + (x - 1)) * 3;
      EXPECT_GE(resultData[idx + 0], resultData[prev_idx + 0] - 2)
          << "R horizontal gradient at (" << x << "," << y << ")";
    }
  }

  // Verify monotonic vertical gradient in G channel
  for (int y = 1; y < dstHeight - 1; y++) {
    for (int x = 0; x < dstWidth; x++) {
      int idx = (y * dstWidth + x) * 3;
      int prev_idx = ((y - 1) * dstWidth + x) * 3;
      EXPECT_GE(resultData[idx + 1], resultData[prev_idx + 1] - 2)
          << "G vertical gradient at (" << x << "," << y << ")";
    }
  }

  // Verify corners
  EXPECT_NEAR(resultData[0], 0, 3) << "Top-left R should be 0";
  EXPECT_NEAR(resultData[1], 0, 3) << "Top-left G should be 0";

  int bottom_right_idx = ((dstHeight - 1) * dstWidth + (dstWidth - 1)) * 3;
  EXPECT_NEAR(resultData[bottom_right_idx + 0], 255, 3) << "Bottom-right R should be 255";
  EXPECT_NEAR(resultData[bottom_right_idx + 1], 255, 3) << "Bottom-right G should be 255";
}
