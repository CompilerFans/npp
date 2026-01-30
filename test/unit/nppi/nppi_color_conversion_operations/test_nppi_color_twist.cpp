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

// 测试8位双通道ColorTwist - 恒等变换
TEST_F(ColorTwistFunctionalTest, ColorTwist32f_8u_C2R_Ctx_Identity) {
  const int width = 32, height = 32;

  std::vector<Npp8u> srcData(width * height * 2);
  for (int i = 0; i < width * height; i++) {
    srcData[i * 2 + 0] = static_cast<Npp8u>(i % 256);
    srcData[i * 2 + 1] = static_cast<Npp8u>((i * 3) % 256);
  }

  NppImageMemory<Npp8u> src(width, height, 2);
  NppImageMemory<Npp8u> dst(width, height, 2);
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
      nppiColorTwist32f_8u_C2R_Ctx(src.get(), src.step(), dst.get(), dst.step(), oSizeROI, aTwist, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp8u> dstData(width * height * 2);
  dst.copyToHost(dstData);

  for (int i = 0; i < width * height * 2; i++) {
    ASSERT_EQ(srcData[i], dstData[i]);
  }
}

// 测试8位双通道ColorTwist in-place
TEST_F(ColorTwistFunctionalTest, ColorTwist32f_8u_C2IR_Ctx_Brightness) {
  const int width = 32, height = 32;

  std::vector<Npp8u> srcData(width * height * 2);
  for (int i = 0; i < width * height; i++) {
    srcData[i * 2 + 0] = static_cast<Npp8u>(i % 256);
    srcData[i * 2 + 1] = static_cast<Npp8u>((i * 3) % 256);
  }

  NppImageMemory<Npp8u> src(width, height, 2);
  src.copyFromHost(srcData);

  Npp32f aTwist[3][4] = {
      {1.0f, 0.0f, 0.0f, 10.0f},
      {0.0f, 1.0f, 0.0f, 20.0f},
      {0.0f, 0.0f, 1.0f, 0.0f},
  };

  NppiSize oSizeROI = {width, height};
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  NppStatus status = nppiColorTwist32f_8u_C2IR_Ctx(src.get(), src.step(), oSizeROI, aTwist, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp8u> dstData(width * height * 2);
  src.copyToHost(dstData);

  for (int i = 0; i < width * height; i++) {
    int c0 = std::min(255, static_cast<int>(srcData[i * 2 + 0]) + 10);
    int c1 = std::min(255, static_cast<int>(srcData[i * 2 + 1]) + 20);
    ASSERT_NEAR(dstData[i * 2 + 0], c0, 1);
    ASSERT_NEAR(dstData[i * 2 + 1], c1, 1);
  }
}

// 测试8位平面三通道ColorTwist - 恒等变换
TEST_F(ColorTwistFunctionalTest, ColorTwist32f_8u_P3R_Ctx_Identity) {
  const int width = 32, height = 32;

  std::vector<Npp8u> srcR(width * height);
  std::vector<Npp8u> srcG(width * height);
  std::vector<Npp8u> srcB(width * height);
  for (int i = 0; i < width * height; i++) {
    srcR[i] = static_cast<Npp8u>(i % 256);
    srcG[i] = static_cast<Npp8u>((i * 2) % 256);
    srcB[i] = static_cast<Npp8u>((i * 3) % 256);
  }

  NppImageMemory<Npp8u> srcPlaneR(width, height, 1);
  NppImageMemory<Npp8u> srcPlaneG(width, height, 1);
  NppImageMemory<Npp8u> srcPlaneB(width, height, 1);
  NppImageMemory<Npp8u> dstPlaneR(width, height, 1);
  NppImageMemory<Npp8u> dstPlaneG(width, height, 1);
  NppImageMemory<Npp8u> dstPlaneB(width, height, 1);

  srcPlaneR.copyFromHost(srcR);
  srcPlaneG.copyFromHost(srcG);
  srcPlaneB.copyFromHost(srcB);

  const Npp8u *pSrc[3] = {srcPlaneR.get(), srcPlaneG.get(), srcPlaneB.get()};
  Npp8u *pDst[3] = {dstPlaneR.get(), dstPlaneG.get(), dstPlaneB.get()};

  Npp32f aTwist[3][4] = {
      {1.0f, 0.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, 1.0f, 0.0f},
  };

  NppiSize oSizeROI = {width, height};
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  NppStatus status = nppiColorTwist32f_8u_P3R_Ctx(pSrc, srcPlaneR.step(), pDst, dstPlaneR.step(), oSizeROI, aTwist,
                                                  nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp8u> dstR(width * height);
  std::vector<Npp8u> dstG(width * height);
  std::vector<Npp8u> dstB(width * height);
  dstPlaneR.copyToHost(dstR);
  dstPlaneG.copyToHost(dstG);
  dstPlaneB.copyToHost(dstB);

  for (int i = 0; i < width * height; i++) {
    ASSERT_EQ(dstR[i], srcR[i]);
    ASSERT_EQ(dstG[i], srcG[i]);
    ASSERT_EQ(dstB[i], srcB[i]);
  }
}

// 测试8位平面三通道ColorTwist in-place
TEST_F(ColorTwistFunctionalTest, ColorTwist32f_8u_IP3R_Ctx_Brightness) {
  const int width = 32, height = 32;

  std::vector<Npp8u> srcR(width * height);
  std::vector<Npp8u> srcG(width * height);
  std::vector<Npp8u> srcB(width * height);
  for (int i = 0; i < width * height; i++) {
    srcR[i] = static_cast<Npp8u>(i % 256);
    srcG[i] = static_cast<Npp8u>((i * 2) % 256);
    srcB[i] = static_cast<Npp8u>((i * 3) % 256);
  }

  NppImageMemory<Npp8u> planeR(width, height, 1);
  NppImageMemory<Npp8u> planeG(width, height, 1);
  NppImageMemory<Npp8u> planeB(width, height, 1);
  planeR.copyFromHost(srcR);
  planeG.copyFromHost(srcG);
  planeB.copyFromHost(srcB);

  Npp8u *pSrcDst[3] = {planeR.get(), planeG.get(), planeB.get()};

  Npp32f aTwist[3][4] = {
      {1.0f, 0.0f, 0.0f, 10.0f},
      {0.0f, 1.0f, 0.0f, 20.0f},
      {0.0f, 0.0f, 1.0f, 30.0f},
  };

  NppiSize oSizeROI = {width, height};
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  NppStatus status =
      nppiColorTwist32f_8u_IP3R_Ctx(pSrcDst, planeR.step(), oSizeROI, aTwist, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp8u> dstR(width * height);
  std::vector<Npp8u> dstG(width * height);
  std::vector<Npp8u> dstB(width * height);
  planeR.copyToHost(dstR);
  planeG.copyToHost(dstG);
  planeB.copyToHost(dstB);

  for (int i = 0; i < width * height; i++) {
    int r = std::min(255, static_cast<int>(srcR[i]) + 10);
    int g = std::min(255, static_cast<int>(srcG[i]) + 20);
    int b = std::min(255, static_cast<int>(srcB[i]) + 30);
    ASSERT_NEAR(dstR[i], r, 1);
    ASSERT_NEAR(dstG[i], g, 1);
    ASSERT_NEAR(dstB[i], b, 1);
  }
}

// 测试16位无符号三通道ColorTwist - 恒等变换
TEST_F(ColorTwistFunctionalTest, ColorTwist32f_16u_C3R_Ctx_Identity) {
  const int width = 32, height = 32;

  std::vector<Npp16u> srcData(width * height * 3);
  for (int i = 0; i < width * height; i++) {
    srcData[i * 3 + 0] = static_cast<Npp16u>((i * 5) % 65535);
    srcData[i * 3 + 1] = static_cast<Npp16u>((i * 7) % 65535);
    srcData[i * 3 + 2] = static_cast<Npp16u>((i * 11) % 65535);
  }

  NppImageMemory<Npp16u> src(width, height, 3);
  NppImageMemory<Npp16u> dst(width, height, 3);
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
      nppiColorTwist32f_16u_C3R_Ctx(src.get(), src.step(), dst.get(), dst.step(), oSizeROI, aTwist, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp16u> dstData(width * height * 3);
  dst.copyToHost(dstData);

  for (int i = 0; i < width * height * 3; i++) {
    ASSERT_EQ(srcData[i], dstData[i]);
  }
}

// 测试16位无符号AC4 ColorTwist - 恒等变换 (alpha置零)
TEST_F(ColorTwistFunctionalTest, ColorTwist32f_16u_AC4R_Ctx_Identity) {
  const int width = 32, height = 32;

  std::vector<Npp16u> srcData(width * height * 4);
  for (int i = 0; i < width * height; i++) {
    srcData[i * 4 + 0] = static_cast<Npp16u>((i * 5) % 65535);
    srcData[i * 4 + 1] = static_cast<Npp16u>((i * 7) % 65535);
    srcData[i * 4 + 2] = static_cast<Npp16u>((i * 11) % 65535);
    srcData[i * 4 + 3] = static_cast<Npp16u>((i * 13) % 65535);
  }

  NppImageMemory<Npp16u> src(width, height, 4);
  NppImageMemory<Npp16u> dst(width, height, 4);
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
      nppiColorTwist32f_16u_AC4R_Ctx(src.get(), src.step(), dst.get(), dst.step(), oSizeROI, aTwist, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp16u> dstData(width * height * 4);
  dst.copyToHost(dstData);

  for (int i = 0; i < width * height; i++) {
    ASSERT_EQ(dstData[i * 4 + 0], srcData[i * 4 + 0]);
    ASSERT_EQ(dstData[i * 4 + 1], srcData[i * 4 + 1]);
    ASSERT_EQ(dstData[i * 4 + 2], srcData[i * 4 + 2]);
    ASSERT_EQ(dstData[i * 4 + 3], 0);
  }
}

// 测试16位无符号三通道ColorTwist in-place
TEST_F(ColorTwistFunctionalTest, ColorTwist32f_16u_C3IR_Ctx_Brightness) {
  const int width = 32, height = 32;

  std::vector<Npp16u> srcData(width * height * 3);
  for (int i = 0; i < width * height; i++) {
    srcData[i * 3 + 0] = static_cast<Npp16u>((i * 5) % 65535);
    srcData[i * 3 + 1] = static_cast<Npp16u>((i * 7) % 65535);
    srcData[i * 3 + 2] = static_cast<Npp16u>((i * 11) % 65535);
  }

  NppImageMemory<Npp16u> src(width, height, 3);
  src.copyFromHost(srcData);

  Npp32f aTwist[3][4] = {
      {1.0f, 0.0f, 0.0f, 10.0f},
      {0.0f, 1.0f, 0.0f, 20.0f},
      {0.0f, 0.0f, 1.0f, 30.0f},
  };

  NppiSize oSizeROI = {width, height};
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  NppStatus status = nppiColorTwist32f_16u_C3IR_Ctx(src.get(), src.step(), oSizeROI, aTwist, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp16u> dstData(width * height * 3);
  src.copyToHost(dstData);

  for (int i = 0; i < width * height; i++) {
    int r = std::min(65535, static_cast<int>(srcData[i * 3 + 0]) + 10);
    int g = std::min(65535, static_cast<int>(srcData[i * 3 + 1]) + 20);
    int b = std::min(65535, static_cast<int>(srcData[i * 3 + 2]) + 30);
    ASSERT_NEAR(dstData[i * 3 + 0], r, 1);
    ASSERT_NEAR(dstData[i * 3 + 1], g, 1);
    ASSERT_NEAR(dstData[i * 3 + 2], b, 1);
  }
}

// 测试16位有符号三通道ColorTwist - 恒等变换
TEST_F(ColorTwistFunctionalTest, ColorTwist32f_16s_C3R_Ctx_Identity) {
  const int width = 32, height = 32;

  std::vector<Npp16s> srcData(width * height * 3);
  for (int i = 0; i < width * height; i++) {
    srcData[i * 3 + 0] = static_cast<Npp16s>((i * 5) % 1024 - 512);
    srcData[i * 3 + 1] = static_cast<Npp16s>((i * 7) % 1024 - 512);
    srcData[i * 3 + 2] = static_cast<Npp16s>((i * 11) % 1024 - 512);
  }

  size_t srcStepBytes = 0;
  size_t dstStepBytes = 0;
  Npp16s *d_src = nullptr;
  Npp16s *d_dst = nullptr;
  ASSERT_EQ(cudaMallocPitch(reinterpret_cast<void **>(&d_src), &srcStepBytes, width * sizeof(Npp16s) * 3, height),
            cudaSuccess);
  ASSERT_EQ(cudaMallocPitch(reinterpret_cast<void **>(&d_dst), &dstStepBytes, width * sizeof(Npp16s) * 3, height),
            cudaSuccess);

  cudaMemcpy2D(d_src, srcStepBytes, srcData.data(), width * sizeof(Npp16s) * 3, width * sizeof(Npp16s) * 3, height,
               cudaMemcpyHostToDevice);

  Npp32f aTwist[3][4] = {
      {1.0f, 0.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, 1.0f, 0.0f},
  };

  NppiSize oSizeROI = {width, height};
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  NppStatus status = nppiColorTwist32f_16s_C3R_Ctx(d_src, static_cast<int>(srcStepBytes), d_dst,
                                                   static_cast<int>(dstStepBytes), oSizeROI, aTwist, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp16s> dstData(width * height * 3);
  cudaMemcpy2D(dstData.data(), width * sizeof(Npp16s) * 3, d_dst, dstStepBytes, width * sizeof(Npp16s) * 3, height,
               cudaMemcpyDeviceToHost);

  for (int i = 0; i < width * height * 3; i++) {
    ASSERT_EQ(srcData[i], dstData[i]);
  }

  cudaFree(d_src);
  cudaFree(d_dst);
}

// 测试16位有符号三通道ColorTwist in-place
TEST_F(ColorTwistFunctionalTest, ColorTwist32f_16s_C3IR_Ctx_Brightness) {
  const int width = 32, height = 32;

  std::vector<Npp16s> srcData(width * height * 3);
  for (int i = 0; i < width * height; i++) {
    srcData[i * 3 + 0] = static_cast<Npp16s>((i * 5) % 1024 - 512);
    srcData[i * 3 + 1] = static_cast<Npp16s>((i * 7) % 1024 - 512);
    srcData[i * 3 + 2] = static_cast<Npp16s>((i * 11) % 1024 - 512);
  }

  size_t srcStepBytes = 0;
  Npp16s *d_src = nullptr;
  ASSERT_EQ(cudaMallocPitch(reinterpret_cast<void **>(&d_src), &srcStepBytes, width * sizeof(Npp16s) * 3, height),
            cudaSuccess);

  cudaMemcpy2D(d_src, srcStepBytes, srcData.data(), width * sizeof(Npp16s) * 3, width * sizeof(Npp16s) * 3, height,
               cudaMemcpyHostToDevice);

  Npp32f aTwist[3][4] = {
      {1.0f, 0.0f, 0.0f, 10.0f},
      {0.0f, 1.0f, 0.0f, 20.0f},
      {0.0f, 0.0f, 1.0f, 30.0f},
  };

  NppiSize oSizeROI = {width, height};
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  NppStatus status =
      nppiColorTwist32f_16s_C3IR_Ctx(d_src, static_cast<int>(srcStepBytes), oSizeROI, aTwist, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp16s> dstData(width * height * 3);
  cudaMemcpy2D(dstData.data(), width * sizeof(Npp16s) * 3, d_src, srcStepBytes, width * sizeof(Npp16s) * 3, height,
               cudaMemcpyDeviceToHost);

  for (int i = 0; i < width * height; i++) {
    int r = std::min(32767, static_cast<int>(srcData[i * 3 + 0]) + 10);
    int g = std::min(32767, static_cast<int>(srcData[i * 3 + 1]) + 20);
    int b = std::min(32767, static_cast<int>(srcData[i * 3 + 2]) + 30);
    ASSERT_NEAR(dstData[i * 3 + 0], r, 1);
    ASSERT_NEAR(dstData[i * 3 + 1], g, 1);
    ASSERT_NEAR(dstData[i * 3 + 2], b, 1);
  }

  cudaFree(d_src);
}
