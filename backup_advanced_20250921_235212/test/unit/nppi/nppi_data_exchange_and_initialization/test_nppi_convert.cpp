// Implementation file

#include "../../framework/npp_test_base.h"
#include <algorithm>
#include <cmath>

using namespace npp_functional_test;

class ConvertFunctionalTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }
};

// 测试8位无符号整数到32位浮点数转换
TEST_F(ConvertFunctionalTest, Convert_8u32f_C1R_Ctx_Basic) {
  const int width = 64, height = 64;

  // prepare test data - 全bounds8位值
  std::vector<Npp8u> srcData(width * height);
  for (int i = 0; i < width * height; i++) {
    srcData[i] = static_cast<Npp8u>(i % 256);
  }

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  src.copyFromHost(srcData);

  // 设置参数
  NppiSize oSizeROI = {width, height};

  // 获取流上下文
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  // 执行转换
  NppStatus status = nppiConvert_8u32f_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), oSizeROI, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  // 同步并获取结果
  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp32f> dstData(width * height);
  dst.copyToHost(dstData);

  // Validate转换结果
  for (int i = 0; i < width * height; i++) {
    ASSERT_FLOAT_EQ(dstData[i], static_cast<float>(srcData[i]));
  }
}

// 测试边界值转换
TEST_F(ConvertFunctionalTest, Convert_8u32f_C1R_Ctx_BoundaryValues) {
  const int width = 16, height = 16;

  // prepare test data - 边界值
  std::vector<Npp8u> srcData(width * height);
  srcData[0] = 0;   // 最小值
  srcData[1] = 255; // 最大值
  srcData[2] = 1;   // 接近最小值
  srcData[3] = 254; // 接近最大值
  srcData[4] = 128; // 中间值

  // 填充剩余数据
  for (int i = 5; i < width * height; i++) {
    srcData[i] = static_cast<Npp8u>(i % 256);
  }

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  src.copyFromHost(srcData);

  // 设置参数
  NppiSize oSizeROI = {width, height};

  // 获取流上下文
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  // 执行转换
  NppStatus status = nppiConvert_8u32f_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), oSizeROI, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  // 同步并获取结果
  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp32f> dstData(width * height);
  dst.copyToHost(dstData);

  // Validate边界值
  ASSERT_FLOAT_EQ(dstData[0], 0.0f);
  ASSERT_FLOAT_EQ(dstData[1], 255.0f);
  ASSERT_FLOAT_EQ(dstData[2], 1.0f);
  ASSERT_FLOAT_EQ(dstData[3], 254.0f);
  ASSERT_FLOAT_EQ(dstData[4], 128.0f);
}

// test roi part trans
TEST_F(ConvertFunctionalTest, Convert_8u32f_C1R_Ctx_PartialROI) {
  const int width = 32, height = 32;

  // prepare test data
  std::vector<Npp8u> srcData(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      srcData[y * width + x] = static_cast<Npp8u>((x + y) % 256);
    }
  }

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  src.copyFromHost(srcData);

  // 设置ROI为中心区域
  NppiSize oSizeROI = {16, 16};
  int xOffset = 8;
  int yOffset = 8;

  // 获取流上下文
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  // 执行ROI转换
  Npp8u *pSrcROI = src.get() + yOffset * src.step() / sizeof(Npp8u) + xOffset;
  Npp32f *pDstROI = dst.get() + yOffset * dst.step() / sizeof(Npp32f) + xOffset;

  NppStatus status = nppiConvert_8u32f_C1R_Ctx(pSrcROI, src.step(), pDstROI, dst.step(), oSizeROI, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  // 同步并获取结果
  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp32f> dstData(width * height);
  dst.copyToHost(dstData);

  // ValidateROI内外的值
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      if (x >= xOffset && x < xOffset + oSizeROI.width && y >= yOffset && y < yOffset + oSizeROI.height) {
        // ROI内应该有转换后的值
        ASSERT_FLOAT_EQ(dstData[idx], static_cast<float>(srcData[idx]));
      } else {
        // ROI外应该保持0
        ASSERT_FLOAT_EQ(dstData[idx], 0.0f);
      }
    }
  }
}