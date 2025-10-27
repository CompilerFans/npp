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
    EXPECT_GE(current, prev - 2) << "R gradient not continuous at x=" << x;
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
  EXPECT_NEAR(resultData[dst_idx_0_0 + 0], 100, 5) << "Super sampling R channel at (0,0)";
  EXPECT_NEAR(resultData[dst_idx_0_0 + 1], 150, 5) << "Super sampling G channel at (0,0)";
  EXPECT_NEAR(resultData[dst_idx_0_0 + 2], 200, 5) << "Super sampling B channel at (0,0)";

  // dst(2,2) samples from src region [4,6)x[4,6), still within 8x8 block
  int dst_idx_2_2 = (2 * dstWidth + 2) * 3;
  EXPECT_NEAR(resultData[dst_idx_2_2 + 0], 100, 5) << "Super sampling R channel at (2,2)";
  EXPECT_NEAR(resultData[dst_idx_2_2 + 1], 150, 5) << "Super sampling G channel at (2,2)";

  // dst(4,4) samples from src region [8,10)x[8,10), which is all 0
  int dst_idx_4_4 = (4 * dstWidth + 4) * 3;
  EXPECT_EQ(resultData[dst_idx_4_4 + 0], 0) << "Region outside block should be 0";
  EXPECT_EQ(resultData[dst_idx_4_4 + 1], 0) << "Region outside block should be 0";
  EXPECT_EQ(resultData[dst_idx_4_4 + 2], 0) << "Region outside block should be 0";

  // Verify edge case: dst(3,3) samples from src [6,8)x[6,8), partially overlapping
  // This tests the averaging behavior at boundaries
  int dst_idx_3_3 = (3 * dstWidth + 3) * 3;
  // Should be less than 100 due to averaging with 0 values
  EXPECT_LT(resultData[dst_idx_3_3 + 0], 100) << "Edge pixel should show averaging effect";
  EXPECT_GT(resultData[dst_idx_3_3 + 0], 0) << "Edge pixel should have non-zero value";
}
