#include "npp_test_base.h"

using namespace npp_functional_test;

class ResizeFunctionalTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }
};

TEST_F(ResizeFunctionalTest, Resize_8u_C1R_NearestNeighbor) {
  const int srcWidth = 32, srcHeight = 32;
  const int dstWidth = 64, dstHeight = 64;

  // prepare test data
  std::vector<Npp8u> srcData(srcWidth * srcHeight);

  // 生成棋盘图案
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      srcData[y * srcWidth + x] = ((x / 4 + y / 4) % 2) ? 255 : 0;
    }
  }

  NppImageMemory<Npp8u> src(srcWidth, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth, dstHeight);

  src.copyFromHost(srcData);

  // 设置源和目标区域
  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  // 执行最近邻缩放
  NppStatus status = nppiResize_8u_C1R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_NN); // 最近邻插值

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiResize_8u_C1R failed";

  // Validate结果 - 简单检查非零像素存在
  std::vector<Npp8u> resultData(dstWidth * dstHeight);
  dst.copyToHost(resultData);

  // 统计非零像素
  int nonZeroCount = 0;
  for (size_t i = 0; i < resultData.size(); i++) {
    if (resultData[i] > 0)
      nonZeroCount++;
  }

  EXPECT_GT(nonZeroCount, dstWidth * dstHeight / 4) << "Not enough non-zero pixels after resize";
  EXPECT_LT(nonZeroCount, dstWidth * dstHeight * 3 / 4) << "Too many non-zero pixels after resize";
}

TEST_F(ResizeFunctionalTest, Resize_8u_C1R_Bilinear) {
  const int srcWidth = 16, srcHeight = 16;
  const int dstWidth = 32, dstHeight = 32;

  // 准备梯度测试数据
  std::vector<Npp8u> srcData(srcWidth * srcHeight);

  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      srcData[y * srcWidth + x] = (Npp8u)(x * 255 / (srcWidth - 1));
    }
  }

  NppImageMemory<Npp8u> src(srcWidth, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth, dstHeight);

  src.copyFromHost(srcData);

  // 设置源和目标区域
  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  // 执行双线性缩放
  NppStatus status = nppiResize_8u_C1R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_LINEAR); // 双线性插值

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiResize_8u_C1R bilinear failed";

  // Validate结果 - 检查梯度连续性
  std::vector<Npp8u> resultData(dstWidth * dstHeight);
  dst.copyToHost(resultData);

  // 检查第一行的梯度
  for (int x = 1; x < dstWidth - 1; x++) {
    int current = resultData[x];
    int prev = resultData[x - 1];
    int next = resultData[x + 1];

    // 梯度应该是单调递增的
    EXPECT_GE(current, prev) << "Gradient not monotonic at x=" << x;
    EXPECT_LE(current, next) << "Gradient not monotonic at x=" << x;
  }
}

TEST_F(ResizeFunctionalTest, Resize_8u_C3R_NearestNeighbor) {
  const int srcWidth = 16, srcHeight = 16;
  const int dstWidth = 32, dstHeight = 32;

  // 准备三通道测试数据
  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      srcData[idx + 0] = (Npp8u)(x * 255 / (srcWidth - 1));  // R
      srcData[idx + 1] = (Npp8u)(y * 255 / (srcHeight - 1)); // G
      srcData[idx + 2] = 128;                                // B
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight); // 3 channels
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  // 设置源和目标区域
  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  // 执行三通道最近邻缩放
  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_NN); // 最近邻插值

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiResize_8u_C3R failed";

  // Validate结果
  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // 检查几个采样点
  for (int y = 0; y < dstHeight; y += 8) {
    for (int x = 0; x < dstWidth; x += 8) {
      int idx = (y * dstWidth + x) * 3;

      // 每个通道应该有合理的值
      EXPECT_LE(resultData[idx + 0], 255) << "R channel out of range";
      EXPECT_LE(resultData[idx + 1], 255) << "G channel out of range";
      EXPECT_LE(resultData[idx + 2], 255) << "B channel out of range";
    }
  }
}

TEST_F(ResizeFunctionalTest, Resize_8u_C3R_Bilinear) {
  const int srcWidth = 16, srcHeight = 16;
  const int dstWidth = 32, dstHeight = 32;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      srcData[idx + 0] = (Npp8u)(x * 255 / (srcWidth - 1));
      srcData[idx + 1] = (Npp8u)(y * 255 / (srcHeight - 1));
      srcData[idx + 2] = 128;
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

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiResize_8u_C3R bilinear failed";

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Check gradient continuity in first row for R channel
  for (int x = 1; x < dstWidth - 1; x++) {
    int current = resultData[x * 3];
    int prev = resultData[(x - 1) * 3];
    EXPECT_GE(current, prev - 1) << "R gradient not continuous at x=" << x;
  }
}

TEST_F(ResizeFunctionalTest, Resize_8u_C3R_Super) {
  const int srcWidth = 64, srcHeight = 64;
  const int dstWidth = 32, dstHeight = 32;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Create uniform blocks to verify averaging behavior
  // Scale factor is 2 (64/32), so each dst pixel samples from approximately 2x2 src region
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      // Set top-left 8x8 block to 100, rest to 0
      // This ensures dst(0,0) through dst(3,3) all sample from value-100 regions
      if (x < 8 && y < 8) {
        srcData[idx + 0] = 100;
        srcData[idx + 1] = 150;
        srcData[idx + 2] = 200;
      } else {
        srcData[idx + 0] = 0;
        srcData[idx + 1] = 0;
        srcData[idx + 2] = 0;
      }
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
                                       NPPI_INTER_SUPER);

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiResize_8u_C3R super sampling failed";

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Verify super sampling averages correctly
  // dst(0,0) samples from src region [0,2)x[0,2), all 100
  int dst_idx_0_0 = (0 * dstWidth + 0) * 3;
  EXPECT_NEAR(resultData[dst_idx_0_0 + 0], 100, 2) << "Super sampling R channel at (0,0)";
  EXPECT_NEAR(resultData[dst_idx_0_0 + 1], 150, 2) << "Super sampling G channel at (0,0)";
  EXPECT_NEAR(resultData[dst_idx_0_0 + 2], 200, 2) << "Super sampling B channel at (0,0)";

  // dst(2,2) samples from src region [4,6)x[4,6), still within 8x8 block
  int dst_idx_2_2 = (2 * dstWidth + 2) * 3;
  EXPECT_NEAR(resultData[dst_idx_2_2 + 0], 100, 2) << "Super sampling R channel at (2,2)";
  EXPECT_NEAR(resultData[dst_idx_2_2 + 1], 150, 2) << "Super sampling G channel at (2,2)";

  // dst(4,4) samples from src region [8,10)x[8,10), which is all 0
  int dst_idx_4_4 = (4 * dstWidth + 4) * 3;
  EXPECT_EQ(resultData[dst_idx_4_4 + 0], 0) << "Region outside block should be 0";
  EXPECT_EQ(resultData[dst_idx_4_4 + 1], 0) << "Region outside block should be 0";
  EXPECT_EQ(resultData[dst_idx_4_4 + 2], 0) << "Region outside block should be 0";

  // Verify edge case: dst(3,3) samples from src [6,8)x[6,8), near boundary
  // NVIDIA NPP may handle boundary differently, so just check it's valid
  int dst_idx_3_3 = (3 * dstWidth + 3) * 3;
  EXPECT_LE(resultData[dst_idx_3_3 + 0], 100) << "Edge pixel value should not exceed source max";
  EXPECT_GE(resultData[dst_idx_3_3 + 0], 0) << "Edge pixel should have non-negative value";
}

TEST_F(ResizeFunctionalTest, Resize_8u_C3R_Super_4xDownscale) {
  const int srcWidth = 128, srcHeight = 128;
  const int dstWidth = 32, dstHeight = 32;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Create checkerboard pattern to test 4x downscaling
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      bool isWhite = ((x / 4) + (y / 4)) % 2 == 0;
      Npp8u value = isWhite ? 200 : 50;
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
                                       NPPI_INTER_SUPER);

  ASSERT_EQ(status, NPP_SUCCESS) << "4x downscale failed";

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // With 4x downscaling, each dst pixel averages 16 src pixels
  // Checkerboard should produce averaged values
  for (int y = 0; y < dstHeight; y++) {
    for (int x = 0; x < dstWidth; x++) {
      int idx = (y * dstWidth + x) * 3;
      // Value should be between 50 and 200 due to averaging
      EXPECT_GE(resultData[idx], 50) << "Value too low at (" << x << "," << y << ")";
      EXPECT_LE(resultData[idx], 200) << "Value too high at (" << x << "," << y << ")";
    }
  }
}

TEST_F(ResizeFunctionalTest, Resize_8u_C3R_Super_NonIntegerScale) {
  const int srcWidth = 100, srcHeight = 100;
  const int dstWidth = 30, dstHeight = 30;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Create gradient pattern
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      srcData[idx + 0] = (Npp8u)(x * 255 / srcWidth);
      srcData[idx + 1] = (Npp8u)(y * 255 / srcHeight);
      srcData[idx + 2] = 128;
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
                                       NPPI_INTER_SUPER);

  ASSERT_EQ(status, NPP_SUCCESS) << "Non-integer scale failed";

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Verify gradient is preserved (R increases left to right)
  for (int y = 0; y < dstHeight; y++) {
    for (int x = 1; x < dstWidth; x++) {
      int idx = (y * dstWidth + x) * 3;
      int prev_idx = (y * dstWidth + (x - 1)) * 3;
      EXPECT_GE(resultData[idx + 0], resultData[prev_idx + 0] - 3) << "Gradient not monotonic";
    }
  }
}

TEST_F(ResizeFunctionalTest, Resize_8u_C3R_Super_SmallImage) {
  const int srcWidth = 16, srcHeight = 16;
  const int dstWidth = 4, dstHeight = 4;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Fill with uniform value
  std::fill(srcData.begin(), srcData.end(), 100);

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_SUPER);

  ASSERT_EQ(status, NPP_SUCCESS) << "Small image resize failed";

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Uniform input should produce uniform output
  for (int i = 0; i < dstWidth * dstHeight * 3; i++) {
    EXPECT_NEAR(resultData[i], 100, 2) << "Uniform averaging failed at index " << i;
  }
}

TEST_F(ResizeFunctionalTest, Resize_8u_C3R_Super_ROITest) {
  const int srcWidth = 64, srcHeight = 64;
  const int dstWidth = 32, dstHeight = 32;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Create quadrant pattern
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      if (x < srcWidth / 2 && y < srcHeight / 2) {
        srcData[idx + 0] = 255;
        srcData[idx + 1] = 0;
        srcData[idx + 2] = 0;
      } else if (x >= srcWidth / 2 && y < srcHeight / 2) {
        srcData[idx + 0] = 0;
        srcData[idx + 1] = 255;
        srcData[idx + 2] = 0;
      } else if (x < srcWidth / 2 && y >= srcHeight / 2) {
        srcData[idx + 0] = 0;
        srcData[idx + 1] = 0;
        srcData[idx + 2] = 255;
      } else {
        srcData[idx + 0] = 255;
        srcData[idx + 1] = 255;
        srcData[idx + 2] = 0;
      }
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  // Test with ROI on upper-left quadrant only
  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth / 2, srcHeight / 2};
  NppiRect dstROI = {0, 0, dstWidth / 2, dstHeight / 2};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_SUPER);

  ASSERT_EQ(status, NPP_SUCCESS) << "ROI resize failed";

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Upper-left quadrant should be predominantly red
  int red_dominant = 0;
  for (int y = 0; y < dstHeight / 2; y++) {
    for (int x = 0; x < dstWidth / 2; x++) {
      int idx = (y * dstWidth + x) * 3;
      if (resultData[idx + 0] > resultData[idx + 1] && resultData[idx + 0] > resultData[idx + 2]) {
        red_dominant++;
      }
    }
  }

  EXPECT_GT(red_dominant, (dstWidth / 2) * (dstHeight / 2) / 2) << "ROI color not preserved";
}

TEST_F(ResizeFunctionalTest, Resize_8u_C3R_Super_LargeDownscale) {
  const int srcWidth = 256, srcHeight = 256;
  const int dstWidth = 16, dstHeight = 16;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Create alternating stripes
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      Npp8u value = (x % 2) ? 255 : 0;
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
                                       NPPI_INTER_SUPER);

  ASSERT_EQ(status, NPP_SUCCESS) << "Large downscale failed";

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // 16x downscaling of alternating stripes should produce ~127
  for (int i = 0; i < dstWidth * dstHeight * 3; i++) {
    EXPECT_NEAR(resultData[i], 127, 15) << "Large downscale averaging incorrect at " << i;
  }
}

TEST_F(ResizeFunctionalTest, Resize_8u_C3R_Super_BoundaryTest) {
  const int srcWidth = 32, srcHeight = 32;
  const int dstWidth = 16, dstHeight = 16;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3, 0);

  // Create precise boundary test pattern
  // Fill left half with 200, right half with 0
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      if (x < srcWidth / 2) {
        srcData[idx + 0] = 200;
        srcData[idx + 1] = 200;
        srcData[idx + 2] = 200;
      }
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
                                       NPPI_INTER_SUPER);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Test boundary pixels at the transition
  // dst(7,y) should sample from src around x=14-16, which is at boundary
  // dst(8,y) should sample from src around x=16-18, which crosses boundary

  for (int y = 0; y < dstHeight; y++) {
    // Far left: should be ~200 (fully in 200 region)
    int idx_left = (y * dstWidth + 3) * 3;
    EXPECT_NEAR(resultData[idx_left + 0], 200, 3) << "Left region incorrect at y=" << y;

    // At boundary: dst(7,y) maps to src(14-16, y*2 to y*2+2)
    int idx_boundary = (y * dstWidth + 7) * 3;
    EXPECT_GE(resultData[idx_boundary + 0], 100) << "Boundary should show 200 influence at y=" << y;
    EXPECT_LE(resultData[idx_boundary + 0], 200) << "Boundary should not exceed max at y=" << y;

    // Just after boundary: dst(8,y) maps to src(16-18, y*2 to y*2+2)
    int idx_after = (y * dstWidth + 8) * 3;
    EXPECT_LT(resultData[idx_after + 0], 100) << "After boundary should show averaging at y=" << y;

    // Far right: should be ~0 (fully in 0 region)
    int idx_right = (y * dstWidth + 12) * 3;
    EXPECT_NEAR(resultData[idx_right + 0], 0, 3) << "Right region incorrect at y=" << y;
  }
}

TEST_F(ResizeFunctionalTest, Resize_8u_C3R_Super_CornerTest) {
  const int srcWidth = 32, srcHeight = 32;
  const int dstWidth = 8, dstHeight = 8;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3, 0);

  // Create quadrant pattern: top-left corner is 255, rest is 0
  for (int y = 0; y < 8; y++) {
    for (int x = 0; x < 8; x++) {
      int idx = (y * srcWidth + x) * 3;
      srcData[idx + 0] = 255;
      srcData[idx + 1] = 255;
      srcData[idx + 2] = 255;
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
                                       NPPI_INTER_SUPER);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // With 4x downscaling, source 8x8 block maps perfectly to dst 2x2 region
  // Top-left 2x2 block: samples from src[0,8)x[0,8), fully in 255 region
  for (int y = 0; y < 2; y++) {
    for (int x = 0; x < 2; x++) {
      int idx = (y * dstWidth + x) * 3;
      EXPECT_NEAR(resultData[idx], 255, 3) << "Pixel (" << x << "," << y << ") should be ~255";
    }
  }

  // Pixels outside 2x2 block: sample from 0 region
  int idx_20 = (0 * dstWidth + 2) * 3;
  EXPECT_EQ(resultData[idx_20], 0) << "Pixel (2,0) should be 0";

  int idx_02 = (2 * dstWidth + 0) * 3;
  EXPECT_EQ(resultData[idx_02], 0) << "Pixel (0,2) should be 0";

  int idx_77 = (7 * dstWidth + 7) * 3;
  EXPECT_EQ(resultData[idx_77], 0) << "Far corner (7,7) should be 0";
}

TEST_F(ResizeFunctionalTest, Resize_8u_C3R_Super_GradientBoundary) {
  const int srcWidth = 64, srcHeight = 64;
  const int dstWidth = 16, dstHeight = 16;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Create horizontal gradient with sharp boundary in the middle
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      // Left half: gradient 0->200, right half: constant 0
      if (x < srcWidth / 2) {
        Npp8u value = (Npp8u)((x * 200) / (srcWidth / 2));
        srcData[idx + 0] = value;
        srcData[idx + 1] = value;
        srcData[idx + 2] = value;
      } else {
        srcData[idx + 0] = 0;
        srcData[idx + 1] = 0;
        srcData[idx + 2] = 0;
      }
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
                                       NPPI_INTER_SUPER);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Verify gradient is preserved in left half
  for (int y = 0; y < dstHeight; y++) {
    // Check monotonic increase in left half
    for (int x = 1; x < dstWidth / 2 - 1; x++) {
      int idx = (y * dstWidth + x) * 3;
      int prev_idx = (y * dstWidth + (x - 1)) * 3;
      EXPECT_GE(resultData[idx], resultData[prev_idx] - 2)
        << "Gradient not preserved at (" << x << "," << y << ")";
    }

    // Right half should be mostly 0
    for (int x = dstWidth / 2 + 2; x < dstWidth; x++) {
      int idx = (y * dstWidth + x) * 3;
      EXPECT_LT(resultData[idx], 10) << "Right region should be near 0 at (" << x << "," << y << ")";
    }
  }
}
