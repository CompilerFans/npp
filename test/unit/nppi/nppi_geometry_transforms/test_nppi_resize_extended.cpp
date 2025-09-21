// Implementation file

#include "../../framework/npp_test_base.h"
#include <algorithm>
#include <cmath>

using namespace npp_functional_test;

class ResizeExtendedTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }
};

// 测试16位无符号整数单通道缩放
TEST_F(ResizeExtendedTest, Resize_16u_C1R_Ctx_LinearInterpolation) {
  const int srcWidth = 32, srcHeight = 32;
  const int dstWidth = 64, dstHeight = 64;

  // prepare test data - 渐变图案
  std::vector<Npp16u> srcData(srcWidth * srcHeight);
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      srcData[y * srcWidth + x] = static_cast<Npp16u>((x + y) * 1000);
    }
  }

  NppImageMemory<Npp16u> src(srcWidth, srcHeight);
  NppImageMemory<Npp16u> dst(dstWidth, dstHeight);

  src.copyFromHost(srcData);

  // 设置参数
  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  // 获取流上下文
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  // 执行缩放
  NppStatus status = nppiResize_16u_C1R_Ctx(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize,
                                            dstROI, NPPI_INTER_LINEAR, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  // 同步并获取结果
  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp16u> dstData(dstWidth * dstHeight);
  dst.copyToHost(dstData);

  // Validate结果不为全零
  bool hasNonZero = false;
  for (const auto &val : dstData) {
    if (val != 0) {
      hasNonZero = true;
      break;
    }
  }
  ASSERT_TRUE(hasNonZero);
}

// 测试32位浮点单通道缩放
TEST_F(ResizeExtendedTest, Resize_32f_C1R_Ctx_CubicInterpolation) {
  const int srcWidth = 32, srcHeight = 32;
  const int dstWidth = 48, dstHeight = 48;

  // prepare test data - 正弦波图案
  std::vector<Npp32f> srcData(srcWidth * srcHeight);
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      srcData[y * srcWidth + x] = sinf(x * 0.2f) * cosf(y * 0.2f);
    }
  }

  NppImageMemory<Npp32f> src(srcWidth, srcHeight);
  NppImageMemory<Npp32f> dst(dstWidth, dstHeight);

  src.copyFromHost(srcData);

  // 设置参数
  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  // 获取流上下文
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  // 执行缩放
  NppStatus status = nppiResize_32f_C1R_Ctx(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize,
                                            dstROI, NPPI_INTER_CUBIC, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  // 同步并获取结果
  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp32f> dstData(dstWidth * dstHeight);
  dst.copyToHost(dstData);

  // Validate结果包含有效数据
  float maxVal = *std::max_element(dstData.begin(), dstData.end());
  float minVal = *std::min_element(dstData.begin(), dstData.end());
  ASSERT_GT(maxVal - minVal, 0.1f); // 确保有数据变化
}

// 测试32位浮点三通道缩放
TEST_F(ResizeExtendedTest, Resize_32f_C3R_Ctx_LinearInterpolation) {
  const int srcWidth = 32, srcHeight = 32;
  const int dstWidth = 64, dstHeight = 64;

  // prepare test data - 彩色渐变
  std::vector<Npp32f> srcData(srcWidth * srcHeight * 3);
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      srcData[idx + 0] = static_cast<float>(x) / srcWidth;                          // R
      srcData[idx + 1] = static_cast<float>(y) / srcHeight;                         // G
      srcData[idx + 2] = 1.0f - static_cast<float>(x + y) / (srcWidth + srcHeight); // B
    }
  }

  NppImageMemory<Npp32f> src(srcWidth, srcHeight, 3);
  NppImageMemory<Npp32f> dst(dstWidth, dstHeight, 3);

  src.copyFromHost(srcData);

  // 设置参数
  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  // 获取流上下文
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  // 执行缩放
  NppStatus status = nppiResize_32f_C3R_Ctx(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize,
                                            dstROI, NPPI_INTER_LINEAR, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  // 同步并获取结果
  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp32f> dstData(dstWidth * dstHeight * 3);
  dst.copyToHost(dstData);

  // ValidateRGB通道都有有效数据
  float rMax = 0, gMax = 0, bMax = 0;
  for (int i = 0; i < dstWidth * dstHeight; i++) {
    rMax = std::max(rMax, dstData[i * 3 + 0]);
    gMax = std::max(gMax, dstData[i * 3 + 1]);
    bMax = std::max(bMax, dstData[i * 3 + 2]);
  }

  ASSERT_GT(rMax, 0.5f);
  ASSERT_GT(gMax, 0.5f);
  ASSERT_GT(bMax, 0.1f);
}