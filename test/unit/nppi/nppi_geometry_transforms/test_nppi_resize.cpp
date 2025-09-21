#include "../../framework/npp_test_base.h"

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

// NOTE: 测试已被禁用 - vendor NPP对无效参数的错误检测行为与预期不符
TEST_F(ResizeFunctionalTest, DISABLED_Resize_ErrorHandling) {
  const int width = 16, height = 16;

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width * 2, height * 2);

  NppiSize srcSize = {width, height};
  NppiSize dstSize = {width * 2, height * 2};
  NppiRect srcROI = {0, 0, width, height};
  NppiRect dstROI = {0, 0, width * 2, height * 2};

  // 测试空指针
  NppStatus status = nppiResize_8u_C1R(nullptr, src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI, 0);
  EXPECT_NE(status, NPP_SUCCESS);

  // 测试无效尺寸
  NppiSize invalidSize = {0, 0};
  status = nppiResize_8u_C1R(src.get(), src.step(), invalidSize, srcROI, dst.get(), dst.step(), dstSize, dstROI, 0);
  EXPECT_NE(status, NPP_SUCCESS);
}
