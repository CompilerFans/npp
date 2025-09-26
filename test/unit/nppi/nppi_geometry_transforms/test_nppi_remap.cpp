#include "npp_test_base.h"
#include <algorithm>
#include <cmath>

using namespace npp_functional_test;

class RemapFunctionalTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }
};

// 测试8位单通道重映射
TEST_F(RemapFunctionalTest, Remap_8u_C1R_Ctx_Identity) {
  const int width = 64, height = 64;

  // prepare test data - 渐变图案
  std::vector<Npp8u> srcData(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      srcData[y * width + x] = static_cast<Npp8u>((x + y) % 256);
    }
  }

  // 准备映射表 - 恒等映射
  std::vector<Npp32f> xMapData(width * height);
  std::vector<Npp32f> yMapData(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      xMapData[idx] = static_cast<float>(x);
      yMapData[idx] = static_cast<float>(y);
    }
  }

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);
  NppImageMemory<Npp32f> xMap(width, height);
  NppImageMemory<Npp32f> yMap(width, height);

  src.copyFromHost(srcData);
  xMap.copyFromHost(xMapData);
  yMap.copyFromHost(yMapData);

  // 设置参数
  NppiSize srcSize = {width, height};
  NppiRect srcROI = {0, 0, width, height};
  NppiSize dstSizeROI = {width, height};

  // 获取流上下文
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  // 执行重映射
  NppStatus status = nppiRemap_8u_C1R_Ctx(src.get(), srcSize, src.step(), srcROI, xMap.get(), xMap.step(), yMap.get(),
                                          yMap.step(), dst.get(), dst.step(), dstSizeROI, NPPI_INTER_NN, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  // 同步并获取结果
  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp8u> dstData(width * height);
  dst.copyToHost(dstData);

  // Validate恒等映射 - 输出应该与输入相同
  for (int i = 0; i < width * height; i++) {
    ASSERT_EQ(srcData[i], dstData[i]);
  }
}

// 测试16位无符号整数三通道重映射
TEST_F(RemapFunctionalTest, Remap_16u_C3R_Ctx_Mirror) {
  const int width = 32, height = 32;

  // prepare test data - RGB渐变
  std::vector<Npp16u> srcData(width * height * 3);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * 3;
      srcData[idx + 0] = static_cast<Npp16u>(x * 2000); // R
      srcData[idx + 1] = static_cast<Npp16u>(y * 2000); // G
      srcData[idx + 2] = static_cast<Npp16u>(30000);    // B
    }
  }

  // 准备映射表 - 水平镜像
  std::vector<Npp32f> xMapData(width * height);
  std::vector<Npp32f> yMapData(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      xMapData[idx] = static_cast<float>(width - 1 - x);
      yMapData[idx] = static_cast<float>(y);
    }
  }

  NppImageMemory<Npp16u> src(width, height, 3);
  NppImageMemory<Npp16u> dst(width, height, 3);
  NppImageMemory<Npp32f> xMap(width, height);
  NppImageMemory<Npp32f> yMap(width, height);

  src.copyFromHost(srcData);
  xMap.copyFromHost(xMapData);
  yMap.copyFromHost(yMapData);

  // 设置参数
  NppiSize srcSize = {width, height};
  NppiRect srcROI = {0, 0, width, height};
  NppiSize dstSizeROI = {width, height};

  // 获取流上下文
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  // 执行重映射
  NppStatus status = nppiRemap_16u_C3R_Ctx(src.get(), srcSize, src.step(), srcROI, xMap.get(), xMap.step(), yMap.get(),
                                           yMap.step(), dst.get(), dst.step(), dstSizeROI, NPPI_INTER_NN, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  // 同步并获取结果
  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp16u> dstData(width * height * 3);
  dst.copyToHost(dstData);

  //
  int firstIdx = 0;
  int lastIdx = (width - 1) * 3;
  ASSERT_EQ(srcData[firstIdx], dstData[lastIdx]);         // R通道
  ASSERT_EQ(srcData[firstIdx + 1], dstData[lastIdx + 1]); // G通道
}

// 测试32位浮点单通道重映射 - 旋转
TEST_F(RemapFunctionalTest, Remap_32f_C1R_Ctx_Rotation) {
  const int width = 64, height = 64;

  // prepare test data - 中心有一个正方形
  std::vector<Npp32f> srcData(width * height, 0.0f);
  for (int y = 20; y < 40; y++) {
    for (int x = 20; x < 40; x++) {
      srcData[y * width + x] = 1.0f;
    }
  }

  // 准备映射表 - 绕中心旋转90度
  std::vector<Npp32f> xMapData(width * height);
  std::vector<Npp32f> yMapData(width * height);
  float cx = width / 2.0f;
  float cy = height / 2.0f;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      // 90度逆时针旋转
      float dx = x - cx;
      float dy = y - cy;
      float newX = -dy + cx;
      float newY = dx + cy;

      int idx = y * width + x;
      xMapData[idx] = newX;
      yMapData[idx] = newY;
    }
  }

  NppImageMemory<Npp32f> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);
  NppImageMemory<Npp32f> xMap(width, height);
  NppImageMemory<Npp32f> yMap(width, height);

  src.copyFromHost(srcData);
  xMap.copyFromHost(xMapData);
  yMap.copyFromHost(yMapData);

  // 设置参数
  NppiSize srcSize = {width, height};
  NppiRect srcROI = {0, 0, width, height};
  NppiSize dstSizeROI = {width, height};

  // 获取流上下文
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  // 执行重映射
  NppStatus status =
      nppiRemap_32f_C1R_Ctx(src.get(), srcSize, src.step(), srcROI, xMap.get(), xMap.step(), yMap.get(), yMap.step(),
                            dst.get(), dst.step(), dstSizeROI, NPPI_INTER_LINEAR, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  // 同步并获取结果
  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp32f> dstData(width * height);
  dst.copyToHost(dstData);

  // Validate旋转 - 检查是否有非零值
  float sum = 0.0f;
  for (const auto &val : dstData) {
    sum += val;
  }
  ASSERT_GT(sum, 100.0f); // 确保有数据被映射
}
