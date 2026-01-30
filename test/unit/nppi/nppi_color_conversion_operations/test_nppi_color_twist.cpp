#include "npp_test_base.h"
#include <algorithm>
#include <cmath>

using namespace npp_functional_test;

class ColorTwistFunctionalTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }
};

// 测试8位三通道ColorTwist - 恒等变换
TEST_F(ColorTwistFunctionalTest, ColorTwist32f_8u_C3R_Ctx_Identity) {
  const int width = 32, height = 32;

  // prepare test data - RGB渐变
  std::vector<Npp8u> srcData(width * height * 3);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * 3;
      srcData[idx + 0] = static_cast<Npp8u>(x * 255 / width);                  // R
      srcData[idx + 1] = static_cast<Npp8u>(y * 255 / height);                 // G
      srcData[idx + 2] = static_cast<Npp8u>((x + y) * 255 / (width + height)); // B
    }
  }

  NppImageMemory<Npp8u> src(width, height, 3);
  NppImageMemory<Npp8u> dst(width, height, 3);

  src.copyFromHost(srcData);

  // 设置恒等变换矩阵
  Npp32f aTwist[3][4] = {
      {1.0f, 0.0f, 0.0f, 0.0f}, // R' = R
      {0.0f, 1.0f, 0.0f, 0.0f}, // G' = G
      {0.0f, 0.0f, 1.0f, 0.0f}  // B' = B
  };

  // 设置参数
  NppiSize oSizeROI = {width, height};

  // 获取流上下文
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  // 执行ColorTwist
  NppStatus status =
      nppiColorTwist32f_8u_C3R_Ctx(src.get(), src.step(), dst.get(), dst.step(), oSizeROI, aTwist, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  // 同步并获取结果
  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp8u> dstData(width * height * 3);
  dst.copyToHost(dstData);

  // Validate恒等变换 - 输出应该与输入相同
  for (int i = 0; i < width * height * 3; i++) {
    ASSERT_EQ(srcData[i], dstData[i]);
  }
}

// 测试8位三通道ColorTwist - 通道交换
TEST_F(ColorTwistFunctionalTest, ColorTwist32f_8u_C3R_Ctx_SwapChannels) {
  const int width = 32, height = 32;

  // prepare test data - 纯色通道
  std::vector<Npp8u> srcData(width * height * 3);
  for (int i = 0; i < width * height; i++) {
    srcData[i * 3 + 0] = 255; // 纯红
    srcData[i * 3 + 1] = 0;   // 无绿
    srcData[i * 3 + 2] = 0;   // 无蓝
  }

  NppImageMemory<Npp8u> src(width, height, 3);
  NppImageMemory<Npp8u> dst(width, height, 3);

  src.copyFromHost(srcData);

  // 设置通道交换矩阵 (RGB -> BGR)
  Npp32f aTwist[3][4] = {
      {0.0f, 0.0f, 1.0f, 0.0f}, // R' = B
      {0.0f, 1.0f, 0.0f, 0.0f}, // G' = G
      {1.0f, 0.0f, 0.0f, 0.0f}  // B' = R
  };

  // 设置参数
  NppiSize oSizeROI = {width, height};

  // 获取流上下文
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  // 执行ColorTwist
  NppStatus status =
      nppiColorTwist32f_8u_C3R_Ctx(src.get(), src.step(), dst.get(), dst.step(), oSizeROI, aTwist, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  // 同步并获取结果
  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp8u> dstData(width * height * 3);
  dst.copyToHost(dstData);

  // Validate通道交换 - 红色应该变成蓝色
  for (int i = 0; i < width * height; i++) {
    ASSERT_EQ(dstData[i * 3 + 0], 0);   // R' = 0
    ASSERT_EQ(dstData[i * 3 + 1], 0);   // G' = 0
    ASSERT_EQ(dstData[i * 3 + 2], 255); // B' = 255
  }
}

// 测试8位单通道ColorTwist - 亮度调整
TEST_F(ColorTwistFunctionalTest, ColorTwist32f_8u_C1R_Ctx_Brightness) {
  const int width = 64, height = 64;

  // prepare test data - 灰度渐变
  std::vector<Npp8u> srcData(width * height);
  for (int i = 0; i < width * height; i++) {
    srcData[i] = static_cast<Npp8u>(i % 256);
  }

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  src.copyFromHost(srcData);

  // 设置亮度增加矩阵（增加50）
  Npp32f aTwist[3][4] = {
      {1.0f, 0.0f, 0.0f, 50.0f}, // Gray' = Gray + 50
      {0.0f, 0.0f, 0.0f, 0.0f},  // 未使用
      {0.0f, 0.0f, 0.0f, 0.0f}   // 未使用
  };

  // 设置参数
  NppiSize oSizeROI = {width, height};

  // 获取流上下文
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  // 执行ColorTwist
  NppStatus status =
      nppiColorTwist32f_8u_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), oSizeROI, aTwist, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  // 同步并获取结果
  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp8u> dstData(width * height);
  dst.copyToHost(dstData);

  // Validate亮度调整
  for (int i = 0; i < width * height; i++) {
    int expected = std::min(255, static_cast<int>(srcData[i]) + 50);
    ASSERT_NEAR(dstData[i], expected, 1); // 允许舍入误差
  }
}

// 测试8位四通道ColorTwist - 恒等变换并保留alpha
TEST_F(ColorTwistFunctionalTest, ColorTwist32f_8u_C4R_Ctx_Identity) {
  const int width = 32, height = 32;

  std::vector<Npp8u> srcData(width * height * 4);
  for (int i = 0; i < width * height; i++) {
    srcData[i * 4 + 0] = static_cast<Npp8u>(i % 256);
    srcData[i * 4 + 1] = static_cast<Npp8u>((i * 2) % 256);
    srcData[i * 4 + 2] = static_cast<Npp8u>((i * 3) % 256);
    srcData[i * 4 + 3] = static_cast<Npp8u>((i * 5) % 256);
  }

  NppImageMemory<Npp8u> src(width, height, 4);
  NppImageMemory<Npp8u> dst(width, height, 4);
  src.copyFromHost(srcData);

  Npp32f aTwist[3][4] = {
      {1.0f, 0.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, 1.0f, 0.0f},
  };

  NppiSize oSizeROI = {width, height};
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  NppStatus status =
      nppiColorTwist32f_8u_C4R_Ctx(src.get(), src.step(), dst.get(), dst.step(), oSizeROI, aTwist, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp8u> dstData(width * height * 4);
  dst.copyToHost(dstData);

  for (int i = 0; i < width * height; i++) {
    ASSERT_EQ(dstData[i * 4 + 0], srcData[i * 4 + 0]);
    ASSERT_EQ(dstData[i * 4 + 1], srcData[i * 4 + 1]);
    ASSERT_EQ(dstData[i * 4 + 2], srcData[i * 4 + 2]);
    ASSERT_EQ(dstData[i * 4 + 3], srcData[i * 4 + 3]);
  }
}

// 测试8位四通道AC4 ColorTwist - 恒等变换，alpha清零
TEST_F(ColorTwistFunctionalTest, ColorTwist32f_8u_AC4R_Ctx_Identity) {
  const int width = 32, height = 32;

  std::vector<Npp8u> srcData(width * height * 4);
  for (int i = 0; i < width * height; i++) {
    srcData[i * 4 + 0] = static_cast<Npp8u>(i % 256);
    srcData[i * 4 + 1] = static_cast<Npp8u>((i * 2) % 256);
    srcData[i * 4 + 2] = static_cast<Npp8u>((i * 3) % 256);
    srcData[i * 4 + 3] = static_cast<Npp8u>((i * 7) % 256);
  }

  NppImageMemory<Npp8u> src(width, height, 4);
  NppImageMemory<Npp8u> dst(width, height, 4);
  src.copyFromHost(srcData);

  Npp32f aTwist[3][4] = {
      {1.0f, 0.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, 1.0f, 0.0f},
  };

  NppiSize oSizeROI = {width, height};
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  NppStatus status =
      nppiColorTwist32f_8u_AC4R_Ctx(src.get(), src.step(), dst.get(), dst.step(), oSizeROI, aTwist, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp8u> dstData(width * height * 4);
  dst.copyToHost(dstData);

  for (int i = 0; i < width * height; i++) {
    ASSERT_EQ(dstData[i * 4 + 0], srcData[i * 4 + 0]);
    ASSERT_EQ(dstData[i * 4 + 1], srcData[i * 4 + 1]);
    ASSERT_EQ(dstData[i * 4 + 2], srcData[i * 4 + 2]);
    ASSERT_EQ(dstData[i * 4 + 3], 0);
  }
}

// 测试8位三通道ColorTwist in-place
TEST_F(ColorTwistFunctionalTest, ColorTwist32f_8u_C3IR_Ctx_Brightness) {
  const int width = 32, height = 32;

  std::vector<Npp8u> srcData(width * height * 3);
  for (int i = 0; i < width * height; i++) {
    srcData[i * 3 + 0] = static_cast<Npp8u>(i % 256);
    srcData[i * 3 + 1] = static_cast<Npp8u>((i * 2) % 256);
    srcData[i * 3 + 2] = static_cast<Npp8u>((i * 3) % 256);
  }

  NppImageMemory<Npp8u> src(width, height, 3);
  src.copyFromHost(srcData);

  Npp32f aTwist[3][4] = {
      {1.0f, 0.0f, 0.0f, 10.0f},
      {0.0f, 1.0f, 0.0f, 20.0f},
      {0.0f, 0.0f, 1.0f, 30.0f},
  };

  NppiSize oSizeROI = {width, height};
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  NppStatus status = nppiColorTwist32f_8u_C3IR_Ctx(src.get(), src.step(), oSizeROI, aTwist, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp8u> dstData(width * height * 3);
  src.copyToHost(dstData);

  for (int i = 0; i < width * height; i++) {
    int r = std::min(255, static_cast<int>(srcData[i * 3 + 0]) + 10);
    int g = std::min(255, static_cast<int>(srcData[i * 3 + 1]) + 20);
    int b = std::min(255, static_cast<int>(srcData[i * 3 + 2]) + 30);
    ASSERT_NEAR(dstData[i * 3 + 0], r, 1);
    ASSERT_NEAR(dstData[i * 3 + 1], g, 1);
    ASSERT_NEAR(dstData[i * 3 + 2], b, 1);
  }
}

// 测试8位四通道ColorTwist in-place (alpha应保持不变)
TEST_F(ColorTwistFunctionalTest, ColorTwist32f_8u_AC4IR_Ctx_Brightness) {
  const int width = 32, height = 32;

  std::vector<Npp8u> srcData(width * height * 4);
  for (int i = 0; i < width * height; i++) {
    srcData[i * 4 + 0] = static_cast<Npp8u>(i % 256);
    srcData[i * 4 + 1] = static_cast<Npp8u>((i * 2) % 256);
    srcData[i * 4 + 2] = static_cast<Npp8u>((i * 3) % 256);
    srcData[i * 4 + 3] = static_cast<Npp8u>((i * 7) % 256);
  }

  NppImageMemory<Npp8u> src(width, height, 4);
  src.copyFromHost(srcData);

  Npp32f aTwist[3][4] = {
      {1.0f, 0.0f, 0.0f, 10.0f},
      {0.0f, 1.0f, 0.0f, 20.0f},
      {0.0f, 0.0f, 1.0f, 30.0f},
  };

  NppiSize oSizeROI = {width, height};
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  NppStatus status = nppiColorTwist32f_8u_AC4IR_Ctx(src.get(), src.step(), oSizeROI, aTwist, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp8u> dstData(width * height * 4);
  src.copyToHost(dstData);

  for (int i = 0; i < width * height; i++) {
    int r = std::min(255, static_cast<int>(srcData[i * 4 + 0]) + 10);
    int g = std::min(255, static_cast<int>(srcData[i * 4 + 1]) + 20);
    int b = std::min(255, static_cast<int>(srcData[i * 4 + 2]) + 30);
    ASSERT_NEAR(dstData[i * 4 + 0], r, 1);
    ASSERT_NEAR(dstData[i * 4 + 1], g, 1);
    ASSERT_NEAR(dstData[i * 4 + 2], b, 1);
    ASSERT_EQ(dstData[i * 4 + 3], srcData[i * 4 + 3]);
  }
}
