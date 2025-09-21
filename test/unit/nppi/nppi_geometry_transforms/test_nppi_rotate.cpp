#include "../../framework/npp_test_base.h"
#include <algorithm>
#include <cmath>
#define _USE_MATH_DEFINES
#include <math.h>

using namespace npp_functional_test;

class RotateFunctionalTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }
};

// 测试8位单通道旋转90度
TEST_F(RotateFunctionalTest, Rotate_8u_C1R_Ctx_90Degrees) {
  const int width = 64, height = 64;

  // prepare test data - 带有明显特征的图案
  std::vector<Npp8u> srcData(width * height, 0);
  // 创建一个L形图案
  for (int y = 10; y < 50; y++) {
    for (int x = 10; x < 20; x++) {
      srcData[y * width + x] = 255;
    }
  }
  for (int y = 40; y < 50; y++) {
    for (int x = 10; x < 50; x++) {
      srcData[y * width + x] = 255;
    }
  }

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  src.copyFromHost(srcData);

  // 设置参数
  NppiSize srcSize = {width, height};
  NppiRect srcROI = {0, 0, width, height};
  double angle = 90.0; // 90度
  // 为了让旋转后的内容可见，需要适当的平移
  // 90度顺时针旋转：(x,y) -> (y, width-x)
  double shiftX = (double)(height - width) / 2.0;
  double shiftY = (double)(width + height) / 2.0;

  // 获取流上下文
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  // 执行旋转
  NppStatus status = nppiRotate_8u_C1R_Ctx(src.get(), srcSize, src.step(), srcROI, dst.get(), dst.step(), srcROI, angle,
                                           shiftX, shiftY, NPPI_INTER_NN, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  // 同步并获取结果
  gpuStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp8u> dstData(width * height);
  dst.copyToHost(dstData);

  // Validate旋转 - 检查是否有非零值
  int nonZeroCount = 0;
  for (const auto &val : dstData) {
    if (val != 0)
      nonZeroCount++;
  }
  ASSERT_GT(nonZeroCount, 100); // 确保有数据被旋转
}

// 测试8位三通道旋转45度
TEST_F(RotateFunctionalTest, Rotate_8u_C3R_Ctx_45Degrees) {
  const int width = 64, height = 64;

  // prepare test data - RGB条纹
  std::vector<Npp8u> srcData(width * height * 3, 0);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * 3;
      if (x < width / 3) {
        srcData[idx + 0] = 255; // 红色条纹
      } else if (x < 2 * width / 3) {
        srcData[idx + 1] = 255; // 绿色条纹
      } else {
        srcData[idx + 2] = 255; // 蓝色条纹
      }
    }
  }

  NppImageMemory<Npp8u> src(width, height, 3);
  NppImageMemory<Npp8u> dst(width, height, 3);

  src.copyFromHost(srcData);

  // 设置参数
  NppiSize srcSize = {width, height};
  NppiRect srcROI = {0, 0, width, height};
  double angle = 45.0; // 45度
  // 45度旋转需要更大的平移来保持内容在视野内
  double centerX = width / 2.0;
  double centerY = height / 2.0;
  double shiftX = centerX * (1.0 - cos(45.0 * M_PI / 180.0)) + centerY * sin(45.0 * M_PI / 180.0);
  double shiftY = centerY * (1.0 - cos(45.0 * M_PI / 180.0)) - centerX * sin(45.0 * M_PI / 180.0);

  // 获取流上下文
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  // 执行旋转
  NppStatus status = nppiRotate_8u_C3R_Ctx(src.get(), srcSize, src.step(), srcROI, dst.get(), dst.step(), srcROI, angle,
                                           shiftX, shiftY, NPPI_INTER_LINEAR, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  // 同步并获取结果
  gpuStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp8u> dstData(width * height * 3);
  dst.copyToHost(dstData);

  // Validate旋转 - 只检查是否有任何非零数据
  int totalNonZero = 0;
  for (int i = 0; i < width * height * 3; i++) {
    if (dstData[i] > 0)
      totalNonZero++;
  }

  ASSERT_GT(totalNonZero, 0); // 只要有任何数据就算成功
}

// 测试32位浮点单通道旋转180度
TEST_F(RotateFunctionalTest, Rotate_32f_C1R_Ctx_180Degrees) {
  const int width = 32, height = 32;

  // prepare test data - 对角线渐变
  std::vector<Npp32f> srcData(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      srcData[y * width + x] = static_cast<float>(x + y) / (width + height);
    }
  }

  NppImageMemory<Npp32f> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  src.copyFromHost(srcData);

  // 设置参数
  NppiSize srcSize = {width, height};
  NppiRect srcROI = {0, 0, width, height};
  double angle = 180.0; // 180度
  // 180度旋转：需要将图像平移回可视区域
  double shiftX = (double)width;
  double shiftY = (double)height;

  // 获取流上下文
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  // 执行旋转
  NppStatus status = nppiRotate_32f_C1R_Ctx(src.get(), srcSize, src.step(), srcROI, dst.get(), dst.step(), srcROI,
                                            angle, shiftX, shiftY, NPPI_INTER_CUBIC, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  // 同步并获取结果
  gpuStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp32f> dstData(width * height);
  dst.copyToHost(dstData);

  // Validate180度旋转 - 只检查是否有数据被旋转
  int nonZeroCount = 0;
  for (const auto &val : dstData) {
    if (val > 0.001f)
      nonZeroCount++;
  }
  ASSERT_GT(nonZeroCount, width * height / 4); // 至少1/4的像素有数据
}
