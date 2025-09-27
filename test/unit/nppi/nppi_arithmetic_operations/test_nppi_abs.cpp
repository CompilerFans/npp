#include "npp_test_base.h"

using namespace npp_functional_test;

class AbsFunctionalTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }
};

// 注意：nppiAbs_8s_C1R函数在标准NPP API中不存在，已删除相关测试

// ==================== 32位浮点数测试 ====================

TEST_F(AbsFunctionalTest, Abs_32f_C1R_BasicOperation) {
  const int width = 32;
  const int height = 32;

  // prepare test data
  std::vector<Npp32f> srcData(width * height);
  std::vector<Npp32f> expectedData(width * height);

  TestDataGenerator::generateRandom(srcData, -100.0f, 100.0f, 12345);

  // 计算期望结果
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = std::abs(srcData[i]);
  }

  // 分配GPU内存
  NppImageMemory<Npp32f> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  // 复制数据到GPU
  src.copyFromHost(srcData);

  // 执行Abs操作
  NppiSize roi = {width, height};
  NppStatus status = nppiAbs_32f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_32f_C1R failed";

  // Validate结果
  std::vector<Npp32f> resultData(width * height);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f))
      << "Abs 32f operation produced incorrect results";
}

// ==================== In-place 操作测试 ====================

TEST_F(AbsFunctionalTest, Abs_32f_C1IR_InPlaceOperation) {
  const int width = 16;
  const int height = 16;

  // prepare test data
  std::vector<Npp32f> srcData(width * height);
  std::vector<Npp32f> expectedData(width * height);

  TestDataGenerator::generateRandom(srcData, -50.0f, 50.0f, 54321);

  // 计算期望结果
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = std::abs(srcData[i]);
  }

  // 分配GPU内存
  NppImageMemory<Npp32f> srcDst(width, height);

  // 复制数据到GPU
  srcDst.copyFromHost(srcData);

  // 执行In-place Abs操作
  NppiSize roi = {width, height};
  NppStatus status = nppiAbs_32f_C1IR(srcDst.get(), srcDst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_32f_C1IR failed";

  // Validate结果
  std::vector<Npp32f> resultData(width * height);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f))
      << "In-place Abs operation produced incorrect results";
}
