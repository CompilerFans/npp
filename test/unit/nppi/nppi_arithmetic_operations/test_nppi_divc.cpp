#include "npp_test_base.h"
#include <algorithm>
#include <cmath>

using namespace npp_functional_test;

class DivCFunctionalTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }
};

// 测试8位单通道除法（带缩放因子）
TEST_F(DivCFunctionalTest, DivC_8u_C1RSfs_Ctx_Basic) {
  const int width = 64, height = 64;
  const Npp8u divisor = 2;
  const int nScaleFactor = 0;

  // prepare test data
  std::vector<Npp8u> srcData(width * height);
  for (int i = 0; i < width * height; i++) {
    srcData[i] = static_cast<Npp8u>((i % 200) + 50); // 50-249bounds
  }

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  src.copyFromHost(srcData);

  // 设置参数
  NppiSize oSizeROI = {width, height};

  // 获取流上下文
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  // 执行除法
  NppStatus status = nppiDivC_8u_C1RSfs_Ctx(src.get(), src.step(), divisor, dst.get(), dst.step(), oSizeROI,
                                            nScaleFactor, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  // 同步并获取结果
  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp8u> dstData(width * height);
  dst.copyToHost(dstData);

  // Validate除法结果（考虑舍入差异）
  for (int i = 0; i < width * height; i++) {
    Npp8u expected = static_cast<Npp8u>(srcData[i] / divisor);
    // NPP可能使用不同的舍入方式，允许±1的误差
    ASSERT_TRUE(abs((int)dstData[i] - (int)expected) <= 1)
        << "At index " << i << ": expected " << (int)expected << " but got " << (int)dstData[i]
        << " (src=" << (int)srcData[i] << ", divisor=" << (int)divisor << ")";
  }
}

// 测试8位三通道除法（带缩放因子）
TEST_F(DivCFunctionalTest, DivC_8u_C3RSfs_Ctx_ColorImage) {
  const int width = 32, height = 32;
  const Npp8u aDivisor[3] = {2, 4, 8}; // RGB各通道不同的除数
  const int nScaleFactor = 0;

  // prepare test data
  std::vector<Npp8u> srcData(width * height * 3);
  for (int i = 0; i < width * height; i++) {
    srcData[i * 3 + 0] = static_cast<Npp8u>(200); // R
    srcData[i * 3 + 1] = static_cast<Npp8u>(160); // G
    srcData[i * 3 + 2] = static_cast<Npp8u>(240); // B
  }

  NppImageMemory<Npp8u> src(width, height, 3);
  NppImageMemory<Npp8u> dst(width, height, 3);

  src.copyFromHost(srcData);

  // 设置参数
  NppiSize oSizeROI = {width, height};

  // 获取流上下文
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  // 执行除法
  NppStatus status = nppiDivC_8u_C3RSfs_Ctx(src.get(), src.step(), aDivisor, dst.get(), dst.step(), oSizeROI,
                                            nScaleFactor, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  // 同步并获取结果
  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp8u> dstData(width * height * 3);
  dst.copyToHost(dstData);

  // Validate除法结果
  for (int i = 0; i < width * height; i++) {
    ASSERT_EQ(dstData[i * 3 + 0], 200 / 2); // R通道
    ASSERT_EQ(dstData[i * 3 + 1], 160 / 4); // G通道
    ASSERT_EQ(dstData[i * 3 + 2], 240 / 8); // B通道
  }
}

// 测试32位浮点单通道除法
TEST_F(DivCFunctionalTest, DivC_32f_C1R_Ctx_FloatingPoint) {
  const int width = 48, height = 48;
  const Npp32f divisor = 3.14159f;

  // prepare test data
  std::vector<Npp32f> srcData(width * height);
  for (int i = 0; i < width * height; i++) {
    srcData[i] = static_cast<float>(i + 1); // 1, 2, 3, ...
  }

  NppImageMemory<Npp32f> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  src.copyFromHost(srcData);

  // 设置参数
  NppiSize oSizeROI = {width, height};

  // 获取流上下文
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  // 执行除法
  NppStatus status =
      nppiDivC_32f_C1R_Ctx(src.get(), src.step(), divisor, dst.get(), dst.step(), oSizeROI, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  // 同步并获取结果
  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp32f> dstData(width * height);
  dst.copyToHost(dstData);

  // Validate除法结果
  for (int i = 0; i < width * height; i++) {
    float expected = srcData[i] / divisor;
    ASSERT_NEAR(dstData[i], expected, 1e-6f);
  }
}

// 测试缩放因子的效果
TEST_F(DivCFunctionalTest, DivC_8u_C1RSfs_Ctx_ScaleFactor) {
  const int width = 16, height = 16;
  const Npp8u divisor = 2;
  const int nScaleFactor = 1; // 结果左移1位（乘以2）

  // prepare test data
  std::vector<Npp8u> srcData(width * height);
  srcData[0] = 100; // 测试值
  srcData[1] = 200;
  srcData[2] = 50;

  for (int i = 3; i < width * height; i++) {
    srcData[i] = static_cast<Npp8u>(i % 200);
  }

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  src.copyFromHost(srcData);

  // 设置参数
  NppiSize oSizeROI = {width, height};

  // 获取流上下文
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  // 执行除法
  NppStatus status = nppiDivC_8u_C1RSfs_Ctx(src.get(), src.step(), divisor, dst.get(), dst.step(), oSizeROI,
                                            nScaleFactor, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  // 同步并获取结果
  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp8u> dstData(width * height);
  dst.copyToHost(dstData);

  // Validate缩放因子效果 - NPP中缩放因子是右移
  ASSERT_EQ(dstData[0], 25); // (100 / 2) >> 1 = 25
  ASSERT_EQ(dstData[1], 50); // (200 / 2) >> 1 = 50
  ASSERT_EQ(dstData[2], 12); // (50 / 2) >> 1 = 12
}

// 测试除零保护
TEST_F(DivCFunctionalTest, DivC_32f_C1R_Ctx_DivisionBySmallNumber) {
  const int width = 8, height = 8;
  const Npp32f divisor = 1e-6f;

  // prepare test data
  std::vector<Npp32f> srcData(width * height, 1.0f);

  NppImageMemory<Npp32f> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  src.copyFromHost(srcData);

  // 设置参数
  NppiSize oSizeROI = {width, height};

  // 获取流上下文
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  // 执行除法
  NppStatus status =
      nppiDivC_32f_C1R_Ctx(src.get(), src.step(), divisor, dst.get(), dst.step(), oSizeROI, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  // 同步并获取结果
  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp32f> dstData(width * height);
  dst.copyToHost(dstData);

  // Validate结果不是NaN或无穷大
  for (const auto &val : dstData) {
    ASSERT_FALSE(std::isnan(val));
    ASSERT_TRUE(std::isfinite(val));
  }
}
