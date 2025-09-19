/**
 * @file test_nppi_filterGauss_simple.cpp
 * @brief NPP 高斯滤波函数简单测试
 */

#include "../../framework/npp_test_base.h"

using namespace npp_functional_test;

class SimpleGaussianFilterTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }
};

// 最简单的功能测试
TEST_F(SimpleGaussianFilterTest, BasicFunctionality) {
  const int width = 8, height = 8;

  // 创建简单的测试图像
  std::vector<Npp8u> srcData(width * height, 128);
  srcData[width / 2 * width + width / 2] = 255; // 中心一个白点

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiFilterGauss_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, NPP_MASK_SIZE_3_X_3);

  EXPECT_EQ(status, NPP_SUCCESS) << "nppiFilterGauss_8u_C1R should succeed";

  if (status == NPP_SUCCESS) {
    std::vector<Npp8u> resultData(width * height);
    dst.copyToHost(resultData);

    // 简单验证：结果数据应该不全是原值
    bool hasChanged = false;
    for (int i = 0; i < width * height; i++) {
      if (resultData[i] != srcData[i]) {
        hasChanged = true;
        break;
      }
    }

    EXPECT_TRUE(hasChanged) << "Gaussian filter should modify the image";
  }
}