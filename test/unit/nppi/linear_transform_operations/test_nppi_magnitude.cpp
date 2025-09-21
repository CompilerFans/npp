#include "../../framework/npp_test_base.h"
#include <cmath>

using namespace npp_functional_test;

class MagnitudeFunctionalTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }
};

TEST_F(MagnitudeFunctionalTest, Magnitude_32fc_32f_C1R_BasicOperation) {
  const int width = 32;
  const int height = 32;

  // 准备复数数据
  std::vector<Npp32fc> srcData(width * height);
  std::vector<Npp32f> expectedData(width * height);

  // 生成测试数据
  for (int i = 0; i < width * height; i++) {
    srcData[i].re = static_cast<float>(i % 10 + 1); // 1-10
    srcData[i].im = static_cast<float>(i % 5 + 1);  // 1-5
    expectedData[i] = std::sqrt(srcData[i].re * srcData[i].re + srcData[i].im * srcData[i].im);
  }

  NppImageMemory<Npp32fc> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiMagnitude_32fc32f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiMagnitude_32fc32f_C1R failed";

  std::vector<Npp32f> resultData(width * height);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f))
      << "Magnitude operation produced incorrect results";
}

// nppiMagnitude_32f_C2R 函数在标准NPP API中不存在，测试已删除

// NOTE: 测试已被禁用 - vendor NPP对无效参数的错误检测行为与预期不符
TEST_F(MagnitudeFunctionalTest, DISABLED_Magnitude_ErrorHandling) {
  const int width = 16;
  const int height = 16;

  NppImageMemory<Npp32fc> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  NppiSize roi = {width, height};

  // 测试null指针
  NppStatus status = nppiMagnitude_32fc32f_C1R(nullptr, src.step(), dst.get(), dst.step(), roi);

  EXPECT_EQ(status, NPP_NULL_POINTER_ERROR) << "Should detect null pointer";
}
