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

  // Expected result
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = std::abs(srcData[i]);
  }

  // Allocate GPU memory
  NppImageMemory<Npp32f> srcDst(width, height);

  // Copy data to GPU
  srcDst.copyFromHost(srcData);

  // Execute In-place Abs operation
  NppiSize roi = {width, height};
  NppStatus status = nppiAbs_32f_C1IR(srcDst.get(), srcDst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_32f_C1IR failed";

  // Validate result
  std::vector<Npp32f> resultData(width * height);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f))
      << "In-place Abs operation produced incorrect results";
}

// ==================== Context version tests ====================

TEST_F(AbsFunctionalTest, Abs_32f_C1R_Ctx_BasicOperation) {
  const int width = 32;
  const int height = 32;

  std::vector<Npp32f> srcData(width * height);
  std::vector<Npp32f> expectedData(width * height);

  TestDataGenerator::generateRandom(srcData, -100.0f, 100.0f, 12345);

  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = std::abs(srcData[i]);
  }

  NppImageMemory<Npp32f> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  NppStatus status = nppiAbs_32f_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, nppStreamCtx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_32f_C1R_Ctx failed";

  std::vector<Npp32f> resultData(width * height);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f))
      << "Abs 32f Ctx operation produced incorrect results";
}

TEST_F(AbsFunctionalTest, Abs_32f_C1IR_Ctx_InPlaceOperation) {
  const int width = 16;
  const int height = 16;

  std::vector<Npp32f> srcData(width * height);
  std::vector<Npp32f> expectedData(width * height);

  TestDataGenerator::generateRandom(srcData, -50.0f, 50.0f, 54321);

  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = std::abs(srcData[i]);
  }

  NppImageMemory<Npp32f> srcDst(width, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  NppStatus status = nppiAbs_32f_C1IR_Ctx(srcDst.get(), srcDst.step(), roi, nppStreamCtx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_32f_C1IR_Ctx failed";

  std::vector<Npp32f> resultData(width * height);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f))
      << "In-place Abs Ctx operation produced incorrect results";
}

// ==================== 16s (signed 16-bit) tests ====================

TEST_F(AbsFunctionalTest, Abs_16s_C1R_BasicOperation) {
  const int width = 32;
  const int height = 32;

  std::vector<Npp16s> srcData(width * height);
  std::vector<Npp16s> expectedData(width * height);

  for (size_t i = 0; i < srcData.size(); i++) {
    srcData[i] = static_cast<Npp16s>((i % 201) - 100); // Range: -100 to 100
  }

  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = static_cast<Npp16s>(std::abs(srcData[i]));
  }

  NppImageMemory<Npp16s> src(width, height);
  NppImageMemory<Npp16s> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiAbs_16s_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_16s_C1R failed";

  std::vector<Npp16s> resultData(width * height);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData)) << "Abs 16s operation produced incorrect results";
}

TEST_F(AbsFunctionalTest, Abs_16s_C1IR_InPlaceOperation) {
  const int width = 16;
  const int height = 16;

  std::vector<Npp16s> srcData(width * height);
  std::vector<Npp16s> expectedData(width * height);

  for (size_t i = 0; i < srcData.size(); i++) {
    srcData[i] = static_cast<Npp16s>((i % 101) - 50); // Range: -50 to 50
  }

  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = static_cast<Npp16s>(std::abs(srcData[i]));
  }

  NppImageMemory<Npp16s> srcDst(width, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiAbs_16s_C1IR(srcDst.get(), srcDst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_16s_C1IR failed";

  std::vector<Npp16s> resultData(width * height);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData))
      << "In-place Abs 16s operation produced incorrect results";
}

// ==================== Multi-channel 32f tests ====================

TEST_F(AbsFunctionalTest, Abs_32f_C3R_BasicOperation) {
  const int width = 16;
  const int height = 16;
  const int channels = 3;

  std::vector<Npp32f> srcData(width * height * channels);
  std::vector<Npp32f> expectedData(width * height * channels);

  TestDataGenerator::generateRandom(srcData, -100.0f, 100.0f, 11111);

  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = std::abs(srcData[i]);
  }

  NppImageMemory<Npp32f> src(width * channels, height);
  NppImageMemory<Npp32f> dst(width * channels, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiAbs_32f_C3R(src.get(), src.step(), dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_32f_C3R failed";

  std::vector<Npp32f> resultData(width * height * channels);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f))
      << "Abs 32f C3 operation produced incorrect results";
}

TEST_F(AbsFunctionalTest, Abs_32f_C3IR_InPlaceOperation) {
  const int width = 16;
  const int height = 16;
  const int channels = 3;

  std::vector<Npp32f> srcData(width * height * channels);
  std::vector<Npp32f> expectedData(width * height * channels);

  TestDataGenerator::generateRandom(srcData, -50.0f, 50.0f, 22222);

  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = std::abs(srcData[i]);
  }

  NppImageMemory<Npp32f> srcDst(width * channels, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiAbs_32f_C3IR(srcDst.get(), srcDst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_32f_C3IR failed";

  std::vector<Npp32f> resultData(width * height * channels);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f))
      << "In-place Abs 32f C3 operation produced incorrect results";
}

TEST_F(AbsFunctionalTest, Abs_32f_C4R_BasicOperation) {
  const int width = 16;
  const int height = 16;
  const int channels = 4;

  std::vector<Npp32f> srcData(width * height * channels);
  std::vector<Npp32f> expectedData(width * height * channels);

  TestDataGenerator::generateRandom(srcData, -100.0f, 100.0f, 33333);

  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = std::abs(srcData[i]);
  }

  NppImageMemory<Npp32f> src(width * channels, height);
  NppImageMemory<Npp32f> dst(width * channels, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiAbs_32f_C4R(src.get(), src.step(), dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_32f_C4R failed";

  std::vector<Npp32f> resultData(width * height * channels);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f))
      << "Abs 32f C4 operation produced incorrect results";
}

TEST_F(AbsFunctionalTest, Abs_32f_C4IR_InPlaceOperation) {
  const int width = 16;
  const int height = 16;
  const int channels = 4;

  std::vector<Npp32f> srcData(width * height * channels);
  std::vector<Npp32f> expectedData(width * height * channels);

  TestDataGenerator::generateRandom(srcData, -50.0f, 50.0f, 44444);

  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = std::abs(srcData[i]);
  }

  NppImageMemory<Npp32f> srcDst(width * channels, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiAbs_32f_C4IR(srcDst.get(), srcDst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_32f_C4IR failed";

  std::vector<Npp32f> resultData(width * height * channels);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f))
      << "In-place Abs 32f C4 operation produced incorrect results";
}

// ==================== Multi-channel 16s tests ====================

TEST_F(AbsFunctionalTest, Abs_16s_C3R_BasicOperation) {
  const int width = 16;
  const int height = 16;
  const int channels = 3;

  std::vector<Npp16s> srcData(width * height * channels);
  std::vector<Npp16s> expectedData(width * height * channels);

  for (size_t i = 0; i < srcData.size(); i++) {
    srcData[i] = static_cast<Npp16s>((i % 201) - 100);
  }

  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = static_cast<Npp16s>(std::abs(srcData[i]));
  }

  NppImageMemory<Npp16s> src(width * channels, height);
  NppImageMemory<Npp16s> dst(width * channels, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiAbs_16s_C3R(src.get(), src.step(), dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_16s_C3R failed";

  std::vector<Npp16s> resultData(width * height * channels);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData))
      << "Abs 16s C3 operation produced incorrect results";
}

TEST_F(AbsFunctionalTest, Abs_16s_C4R_BasicOperation) {
  const int width = 16;
  const int height = 16;
  const int channels = 4;

  std::vector<Npp16s> srcData(width * height * channels);
  std::vector<Npp16s> expectedData(width * height * channels);

  for (size_t i = 0; i < srcData.size(); i++) {
    srcData[i] = static_cast<Npp16s>((i % 201) - 100);
  }

  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = static_cast<Npp16s>(std::abs(srcData[i]));
  }

  NppImageMemory<Npp16s> src(width * channels, height);
  NppImageMemory<Npp16s> dst(width * channels, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiAbs_16s_C4R(src.get(), src.step(), dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_16s_C4R failed";

  std::vector<Npp16s> resultData(width * height * channels);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData))
      << "Abs 16s C4 operation produced incorrect results";
}

// ==================== Context version tests for 16s ====================

TEST_F(AbsFunctionalTest, Abs_16s_C1R_Ctx_BasicOperation) {
  const int width = 32;
  const int height = 32;

  std::vector<Npp16s> srcData(width * height);
  std::vector<Npp16s> expectedData(width * height);

  for (size_t i = 0; i < srcData.size(); i++) {
    srcData[i] = static_cast<Npp16s>((i % 201) - 100);
  }

  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = static_cast<Npp16s>(std::abs(srcData[i]));
  }

  NppImageMemory<Npp16s> src(width, height);
  NppImageMemory<Npp16s> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  NppStatus status = nppiAbs_16s_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, nppStreamCtx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_16s_C1R_Ctx failed";

  std::vector<Npp16s> resultData(width * height);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData))
      << "Abs 16s C1R Ctx operation produced incorrect results";
}

TEST_F(AbsFunctionalTest, Abs_16s_C1IR_Ctx_InPlaceOperation) {
  const int width = 16;
  const int height = 16;

  std::vector<Npp16s> srcData(width * height);
  std::vector<Npp16s> expectedData(width * height);

  for (size_t i = 0; i < srcData.size(); i++) {
    srcData[i] = static_cast<Npp16s>((i % 101) - 50);
  }

  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = static_cast<Npp16s>(std::abs(srcData[i]));
  }

  NppImageMemory<Npp16s> srcDst(width, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  NppStatus status = nppiAbs_16s_C1IR_Ctx(srcDst.get(), srcDst.step(), roi, nppStreamCtx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_16s_C1IR_Ctx failed";

  std::vector<Npp16s> resultData(width * height);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData))
      << "In-place Abs 16s Ctx operation produced incorrect results";
}

TEST_F(AbsFunctionalTest, Abs_16s_C3R_Ctx_BasicOperation) {
  const int width = 16;
  const int height = 16;
  const int channels = 3;

  std::vector<Npp16s> srcData(width * height * channels);
  std::vector<Npp16s> expectedData(width * height * channels);

  for (size_t i = 0; i < srcData.size(); i++) {
    srcData[i] = static_cast<Npp16s>((i % 201) - 100);
  }

  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = static_cast<Npp16s>(std::abs(srcData[i]));
  }

  NppImageMemory<Npp16s> src(width * channels, height);
  NppImageMemory<Npp16s> dst(width * channels, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  NppStatus status = nppiAbs_16s_C3R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, nppStreamCtx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_16s_C3R_Ctx failed";

  std::vector<Npp16s> resultData(width * height * channels);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData))
      << "Abs 16s C3R Ctx operation produced incorrect results";
}

TEST_F(AbsFunctionalTest, Abs_16s_C4R_Ctx_BasicOperation) {
  const int width = 16;
  const int height = 16;
  const int channels = 4;

  std::vector<Npp16s> srcData(width * height * channels);
  std::vector<Npp16s> expectedData(width * height * channels);

  for (size_t i = 0; i < srcData.size(); i++) {
    srcData[i] = static_cast<Npp16s>((i % 201) - 100);
  }

  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = static_cast<Npp16s>(std::abs(srcData[i]));
  }

  NppImageMemory<Npp16s> src(width * channels, height);
  NppImageMemory<Npp16s> dst(width * channels, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  NppStatus status = nppiAbs_16s_C4R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, nppStreamCtx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_16s_C4R_Ctx failed";

  std::vector<Npp16s> resultData(width * height * channels);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData))
      << "Abs 16s C4R Ctx operation produced incorrect results";
}

// ==================== Context version tests for 32f ====================

TEST_F(AbsFunctionalTest, Abs_32f_C3R_Ctx_BasicOperation) {
  const int width = 16;
  const int height = 16;
  const int channels = 3;

  std::vector<Npp32f> srcData(width * height * channels);
  std::vector<Npp32f> expectedData(width * height * channels);

  TestDataGenerator::generateRandom(srcData, -100.0f, 100.0f, 11111);

  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = std::abs(srcData[i]);
  }

  NppImageMemory<Npp32f> src(width * channels, height);
  NppImageMemory<Npp32f> dst(width * channels, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  NppStatus status = nppiAbs_32f_C3R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, nppStreamCtx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_32f_C3R_Ctx failed";

  std::vector<Npp32f> resultData(width * height * channels);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f))
      << "Abs 32f C3R Ctx operation produced incorrect results";
}

TEST_F(AbsFunctionalTest, Abs_32f_C3IR_Ctx_InPlaceOperation) {
  const int width = 16;
  const int height = 16;
  const int channels = 3;

  std::vector<Npp32f> srcData(width * height * channels);
  std::vector<Npp32f> expectedData(width * height * channels);

  TestDataGenerator::generateRandom(srcData, -50.0f, 50.0f, 22222);

  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = std::abs(srcData[i]);
  }

  NppImageMemory<Npp32f> srcDst(width * channels, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  NppStatus status = nppiAbs_32f_C3IR_Ctx(srcDst.get(), srcDst.step(), roi, nppStreamCtx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_32f_C3IR_Ctx failed";

  std::vector<Npp32f> resultData(width * height * channels);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f))
      << "In-place Abs 32f C3 Ctx operation produced incorrect results";
}

TEST_F(AbsFunctionalTest, Abs_32f_C4R_Ctx_BasicOperation) {
  const int width = 16;
  const int height = 16;
  const int channels = 4;

  std::vector<Npp32f> srcData(width * height * channels);
  std::vector<Npp32f> expectedData(width * height * channels);

  TestDataGenerator::generateRandom(srcData, -100.0f, 100.0f, 33333);

  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = std::abs(srcData[i]);
  }

  NppImageMemory<Npp32f> src(width * channels, height);
  NppImageMemory<Npp32f> dst(width * channels, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  NppStatus status = nppiAbs_32f_C4R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, nppStreamCtx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_32f_C4R_Ctx failed";

  std::vector<Npp32f> resultData(width * height * channels);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f))
      << "Abs 32f C4R Ctx operation produced incorrect results";
}

TEST_F(AbsFunctionalTest, Abs_32f_C4IR_Ctx_InPlaceOperation) {
  const int width = 16;
  const int height = 16;
  const int channels = 4;

  std::vector<Npp32f> srcData(width * height * channels);
  std::vector<Npp32f> expectedData(width * height * channels);

  TestDataGenerator::generateRandom(srcData, -50.0f, 50.0f, 44444);

  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = std::abs(srcData[i]);
  }

  NppImageMemory<Npp32f> srcDst(width * channels, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  NppStatus status = nppiAbs_32f_C4IR_Ctx(srcDst.get(), srcDst.step(), roi, nppStreamCtx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_32f_C4IR_Ctx failed";

  std::vector<Npp32f> resultData(width * height * channels);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f))
      << "In-place Abs 32f C4 Ctx operation produced incorrect results";
}

// ==================== 16s In-place and multi-channel in-place tests ====================

TEST_F(AbsFunctionalTest, Abs_16s_C3IR_InPlaceOperation) {
  const int width = 16;
  const int height = 16;
  const int channels = 3;

  std::vector<Npp16s> srcData(width * height * channels);
  std::vector<Npp16s> expectedData(width * height * channels);

  for (size_t i = 0; i < srcData.size(); i++) {
    srcData[i] = static_cast<Npp16s>((i % 201) - 100);
  }

  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = static_cast<Npp16s>(std::abs(srcData[i]));
  }

  NppImageMemory<Npp16s> srcDst(width * channels, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiAbs_16s_C3IR(srcDst.get(), srcDst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_16s_C3IR failed";

  std::vector<Npp16s> resultData(width * height * channels);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData))
      << "In-place Abs 16s C3 operation produced incorrect results";
}

TEST_F(AbsFunctionalTest, Abs_16s_C3IR_Ctx_InPlaceOperation) {
  const int width = 16;
  const int height = 16;
  const int channels = 3;

  std::vector<Npp16s> srcData(width * height * channels);
  std::vector<Npp16s> expectedData(width * height * channels);

  for (size_t i = 0; i < srcData.size(); i++) {
    srcData[i] = static_cast<Npp16s>((i % 201) - 100);
  }

  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = static_cast<Npp16s>(std::abs(srcData[i]));
  }

  NppImageMemory<Npp16s> srcDst(width * channels, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  NppStatus status = nppiAbs_16s_C3IR_Ctx(srcDst.get(), srcDst.step(), roi, nppStreamCtx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_16s_C3IR_Ctx failed";

  std::vector<Npp16s> resultData(width * height * channels);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData))
      << "In-place Abs 16s C3 Ctx operation produced incorrect results";
}

TEST_F(AbsFunctionalTest, Abs_16s_C4IR_InPlaceOperation) {
  const int width = 16;
  const int height = 16;
  const int channels = 4;

  std::vector<Npp16s> srcData(width * height * channels);
  std::vector<Npp16s> expectedData(width * height * channels);

  for (size_t i = 0; i < srcData.size(); i++) {
    srcData[i] = static_cast<Npp16s>((i % 201) - 100);
  }

  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = static_cast<Npp16s>(std::abs(srcData[i]));
  }

  NppImageMemory<Npp16s> srcDst(width * channels, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiAbs_16s_C4IR(srcDst.get(), srcDst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_16s_C4IR failed";

  std::vector<Npp16s> resultData(width * height * channels);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData))
      << "In-place Abs 16s C4 operation produced incorrect results";
}

TEST_F(AbsFunctionalTest, Abs_16s_C4IR_Ctx_InPlaceOperation) {
  const int width = 16;
  const int height = 16;
  const int channels = 4;

  std::vector<Npp16s> srcData(width * height * channels);
  std::vector<Npp16s> expectedData(width * height * channels);

  for (size_t i = 0; i < srcData.size(); i++) {
    srcData[i] = static_cast<Npp16s>((i % 201) - 100);
  }

  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = static_cast<Npp16s>(std::abs(srcData[i]));
  }

  NppImageMemory<Npp16s> srcDst(width * channels, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  NppStatus status = nppiAbs_16s_C4IR_Ctx(srcDst.get(), srcDst.step(), roi, nppStreamCtx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_16s_C4IR_Ctx failed";

  std::vector<Npp16s> resultData(width * height * channels);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData))
      << "In-place Abs 16s C4 Ctx operation produced incorrect results";
}

// ==================== AC4 (Alpha Channel 4) tests ====================

// AC4 processes only the first 3 channels, leaving the alpha channel unchanged

TEST_F(AbsFunctionalTest, Abs_16s_AC4R_BasicOperation) {
  const int width = 16;
  const int height = 16;
  const int channels = 4;

  std::vector<Npp16s> srcData(width * height * channels);

  for (size_t i = 0; i < srcData.size(); i++) {
    srcData[i] = static_cast<Npp16s>((i % 201) - 100);
  }

  NppImageMemory<Npp16s> src(width * channels, height);
  NppImageMemory<Npp16s> dst(width * channels, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiAbs_16s_AC4R(src.get(), src.step(), dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_16s_AC4R failed";

  std::vector<Npp16s> resultData(width * height * channels);
  dst.copyToHost(resultData);

  // AC4: verify only first 3 channels, alpha is unspecified for non-in-place
  for (size_t i = 0; i < width * height; i++) {
    EXPECT_EQ(resultData[i * 4 + 0], static_cast<Npp16s>(std::abs(srcData[i * 4 + 0])));
    EXPECT_EQ(resultData[i * 4 + 1], static_cast<Npp16s>(std::abs(srcData[i * 4 + 1])));
    EXPECT_EQ(resultData[i * 4 + 2], static_cast<Npp16s>(std::abs(srcData[i * 4 + 2])));
    // Alpha channel not verified - AC4R does not guarantee alpha in dst
  }
}

TEST_F(AbsFunctionalTest, Abs_16s_AC4R_Ctx_BasicOperation) {
  const int width = 16;
  const int height = 16;
  const int channels = 4;

  std::vector<Npp16s> srcData(width * height * channels);

  for (size_t i = 0; i < srcData.size(); i++) {
    srcData[i] = static_cast<Npp16s>((i % 201) - 100);
  }

  NppImageMemory<Npp16s> src(width * channels, height);
  NppImageMemory<Npp16s> dst(width * channels, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  ctx.hStream = 0;
  NppStatus status = nppiAbs_16s_AC4R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_16s_AC4R_Ctx failed";

  std::vector<Npp16s> resultData(width * height * channels);
  dst.copyToHost(resultData);

  // AC4: verify only first 3 channels
  for (size_t i = 0; i < width * height; i++) {
    EXPECT_EQ(resultData[i * 4 + 0], static_cast<Npp16s>(std::abs(srcData[i * 4 + 0])));
    EXPECT_EQ(resultData[i * 4 + 1], static_cast<Npp16s>(std::abs(srcData[i * 4 + 1])));
    EXPECT_EQ(resultData[i * 4 + 2], static_cast<Npp16s>(std::abs(srcData[i * 4 + 2])));
  }
}

TEST_F(AbsFunctionalTest, Abs_16s_AC4IR_InPlaceOperation) {
  const int width = 16;
  const int height = 16;
  const int channels = 4;

  std::vector<Npp16s> srcData(width * height * channels);
  std::vector<Npp16s> expectedData(width * height * channels);

  for (size_t i = 0; i < srcData.size(); i++) {
    srcData[i] = static_cast<Npp16s>((i % 201) - 100);
  }

  for (size_t i = 0; i < width * height; i++) {
    expectedData[i * 4 + 0] = static_cast<Npp16s>(std::abs(srcData[i * 4 + 0]));
    expectedData[i * 4 + 1] = static_cast<Npp16s>(std::abs(srcData[i * 4 + 1]));
    expectedData[i * 4 + 2] = static_cast<Npp16s>(std::abs(srcData[i * 4 + 2]));
    expectedData[i * 4 + 3] = srcData[i * 4 + 3];
  }

  NppImageMemory<Npp16s> srcDst(width * channels, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiAbs_16s_AC4IR(srcDst.get(), srcDst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_16s_AC4IR failed";

  std::vector<Npp16s> resultData(width * height * channels);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData))
      << "In-place Abs 16s AC4 operation produced incorrect results";
}

TEST_F(AbsFunctionalTest, Abs_16s_AC4IR_Ctx_InPlaceOperation) {
  const int width = 16;
  const int height = 16;
  const int channels = 4;

  std::vector<Npp16s> srcData(width * height * channels);
  std::vector<Npp16s> expectedData(width * height * channels);

  for (size_t i = 0; i < srcData.size(); i++) {
    srcData[i] = static_cast<Npp16s>((i % 201) - 100);
  }

  for (size_t i = 0; i < width * height; i++) {
    expectedData[i * 4 + 0] = static_cast<Npp16s>(std::abs(srcData[i * 4 + 0]));
    expectedData[i * 4 + 1] = static_cast<Npp16s>(std::abs(srcData[i * 4 + 1]));
    expectedData[i * 4 + 2] = static_cast<Npp16s>(std::abs(srcData[i * 4 + 2]));
    expectedData[i * 4 + 3] = srcData[i * 4 + 3];
  }

  NppImageMemory<Npp16s> srcDst(width * channels, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  ctx.hStream = 0;
  NppStatus status = nppiAbs_16s_AC4IR_Ctx(srcDst.get(), srcDst.step(), roi, ctx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_16s_AC4IR_Ctx failed";

  std::vector<Npp16s> resultData(width * height * channels);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData))
      << "In-place Abs 16s AC4 Ctx operation produced incorrect results";
}

// 32f AC4 tests

TEST_F(AbsFunctionalTest, Abs_32f_AC4R_BasicOperation) {
  const int width = 16;
  const int height = 16;
  const int channels = 4;

  std::vector<Npp32f> srcData(width * height * channels);

  TestDataGenerator::generateRandom(srcData, -100.0f, 100.0f, 55555);

  NppImageMemory<Npp32f> src(width * channels, height);
  NppImageMemory<Npp32f> dst(width * channels, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiAbs_32f_AC4R(src.get(), src.step(), dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_32f_AC4R failed";

  std::vector<Npp32f> resultData(width * height * channels);
  dst.copyToHost(resultData);

  // AC4: verify only first 3 channels
  for (size_t i = 0; i < width * height; i++) {
    EXPECT_NEAR(resultData[i * 4 + 0], std::abs(srcData[i * 4 + 0]), 1e-5f);
    EXPECT_NEAR(resultData[i * 4 + 1], std::abs(srcData[i * 4 + 1]), 1e-5f);
    EXPECT_NEAR(resultData[i * 4 + 2], std::abs(srcData[i * 4 + 2]), 1e-5f);
  }
}

TEST_F(AbsFunctionalTest, Abs_32f_AC4R_Ctx_BasicOperation) {
  const int width = 16;
  const int height = 16;
  const int channels = 4;

  std::vector<Npp32f> srcData(width * height * channels);

  TestDataGenerator::generateRandom(srcData, -100.0f, 100.0f, 55555);

  NppImageMemory<Npp32f> src(width * channels, height);
  NppImageMemory<Npp32f> dst(width * channels, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  ctx.hStream = 0;
  NppStatus status = nppiAbs_32f_AC4R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_32f_AC4R_Ctx failed";

  std::vector<Npp32f> resultData(width * height * channels);
  dst.copyToHost(resultData);

  // AC4: verify only first 3 channels
  for (size_t i = 0; i < width * height; i++) {
    EXPECT_NEAR(resultData[i * 4 + 0], std::abs(srcData[i * 4 + 0]), 1e-5f);
    EXPECT_NEAR(resultData[i * 4 + 1], std::abs(srcData[i * 4 + 1]), 1e-5f);
    EXPECT_NEAR(resultData[i * 4 + 2], std::abs(srcData[i * 4 + 2]), 1e-5f);
  }
}

TEST_F(AbsFunctionalTest, Abs_32f_AC4IR_InPlaceOperation) {
  const int width = 16;
  const int height = 16;
  const int channels = 4;

  std::vector<Npp32f> srcData(width * height * channels);
  std::vector<Npp32f> expectedData(width * height * channels);

  TestDataGenerator::generateRandom(srcData, -50.0f, 50.0f, 66666);

  for (size_t i = 0; i < width * height; i++) {
    expectedData[i * 4 + 0] = std::abs(srcData[i * 4 + 0]);
    expectedData[i * 4 + 1] = std::abs(srcData[i * 4 + 1]);
    expectedData[i * 4 + 2] = std::abs(srcData[i * 4 + 2]);
    expectedData[i * 4 + 3] = srcData[i * 4 + 3];
  }

  NppImageMemory<Npp32f> srcDst(width * channels, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiAbs_32f_AC4IR(srcDst.get(), srcDst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_32f_AC4IR failed";

  std::vector<Npp32f> resultData(width * height * channels);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f))
      << "In-place Abs 32f AC4 operation produced incorrect results";
}

TEST_F(AbsFunctionalTest, Abs_32f_AC4IR_Ctx_InPlaceOperation) {
  const int width = 16;
  const int height = 16;
  const int channels = 4;

  std::vector<Npp32f> srcData(width * height * channels);
  std::vector<Npp32f> expectedData(width * height * channels);

  TestDataGenerator::generateRandom(srcData, -50.0f, 50.0f, 66666);

  for (size_t i = 0; i < width * height; i++) {
    expectedData[i * 4 + 0] = std::abs(srcData[i * 4 + 0]);
    expectedData[i * 4 + 1] = std::abs(srcData[i * 4 + 1]);
    expectedData[i * 4 + 2] = std::abs(srcData[i * 4 + 2]);
    expectedData[i * 4 + 3] = srcData[i * 4 + 3];
  }

  NppImageMemory<Npp32f> srcDst(width * channels, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  ctx.hStream = 0;
  NppStatus status = nppiAbs_32f_AC4IR_Ctx(srcDst.get(), srcDst.step(), roi, ctx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_32f_AC4IR_Ctx failed";

  std::vector<Npp32f> resultData(width * height * channels);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f))
      << "In-place Abs 32f AC4 Ctx operation produced incorrect results";
}

// ==================== 16f (half-precision float) tests ====================

TEST_F(AbsFunctionalTest, Abs_16f_C1R_BasicOperation) {
  const int width = 32;
  const int height = 32;

  std::vector<Npp16f> srcData(width * height);
  std::vector<Npp16f> expectedData(width * height);

  TestDataGenerator::generateRandom16f(srcData, -100.0f, 100.0f, 12345);

  for (size_t i = 0; i < expectedData.size(); i++) {
    float val = npp16f_to_float_host(srcData[i]);
    expectedData[i] = float_to_npp16f_host(std::fabs(val));
  }

  NppImageMemory<Npp16f> src(width, height);
  NppImageMemory<Npp16f> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiAbs_16f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_16f_C1R failed";

  std::vector<Npp16f> resultData(width * height);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-3f))
      << "Abs 16f C1R operation produced incorrect results";
}

TEST_F(AbsFunctionalTest, Abs_16f_C1R_Ctx_BasicOperation) {
  const int width = 32;
  const int height = 32;

  std::vector<Npp16f> srcData(width * height);
  std::vector<Npp16f> expectedData(width * height);

  TestDataGenerator::generateRandom16f(srcData, -100.0f, 100.0f, 12345);

  for (size_t i = 0; i < expectedData.size(); i++) {
    float val = npp16f_to_float_host(srcData[i]);
    expectedData[i] = float_to_npp16f_host(std::fabs(val));
  }

  NppImageMemory<Npp16f> src(width, height);
  NppImageMemory<Npp16f> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status = nppiAbs_16f_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_16f_C1R_Ctx failed";

  std::vector<Npp16f> resultData(width * height);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-3f))
      << "Abs 16f C1R Ctx operation produced incorrect results";
}

TEST_F(AbsFunctionalTest, Abs_16f_C1IR_InPlaceOperation) {
  const int width = 16;
  const int height = 16;

  std::vector<Npp16f> srcData(width * height);
  std::vector<Npp16f> expectedData(width * height);

  TestDataGenerator::generateRandom16f(srcData, -50.0f, 50.0f, 54321);

  for (size_t i = 0; i < expectedData.size(); i++) {
    float val = npp16f_to_float_host(srcData[i]);
    expectedData[i] = float_to_npp16f_host(std::fabs(val));
  }

  NppImageMemory<Npp16f> srcDst(width, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiAbs_16f_C1IR(srcDst.get(), srcDst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_16f_C1IR failed";

  std::vector<Npp16f> resultData(width * height);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-3f))
      << "In-place Abs 16f C1 operation produced incorrect results";
}

TEST_F(AbsFunctionalTest, Abs_16f_C1IR_Ctx_InPlaceOperation) {
  const int width = 16;
  const int height = 16;

  std::vector<Npp16f> srcData(width * height);
  std::vector<Npp16f> expectedData(width * height);

  TestDataGenerator::generateRandom16f(srcData, -50.0f, 50.0f, 54321);

  for (size_t i = 0; i < expectedData.size(); i++) {
    float val = npp16f_to_float_host(srcData[i]);
    expectedData[i] = float_to_npp16f_host(std::fabs(val));
  }

  NppImageMemory<Npp16f> srcDst(width, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status = nppiAbs_16f_C1IR_Ctx(srcDst.get(), srcDst.step(), roi, ctx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_16f_C1IR_Ctx failed";

  std::vector<Npp16f> resultData(width * height);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-3f))
      << "In-place Abs 16f C1 Ctx operation produced incorrect results";
}

// 16f C3 tests
TEST_F(AbsFunctionalTest, Abs_16f_C3R_BasicOperation) {
  const int width = 32;
  const int height = 32;
  const int channels = 3;

  std::vector<Npp16f> srcData(width * height * channels);
  std::vector<Npp16f> expectedData(width * height * channels);

  TestDataGenerator::generateRandom16f(srcData, -100.0f, 100.0f, 33333);

  for (size_t i = 0; i < expectedData.size(); i++) {
    float val = npp16f_to_float_host(srcData[i]);
    expectedData[i] = float_to_npp16f_host(std::fabs(val));
  }

  NppImageMemory<Npp16f> src(width * channels, height);
  NppImageMemory<Npp16f> dst(width * channels, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiAbs_16f_C3R(src.get(), src.step(), dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_16f_C3R failed";

  std::vector<Npp16f> resultData(width * height * channels);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-3f))
      << "Abs 16f C3R operation produced incorrect results";
}

TEST_F(AbsFunctionalTest, Abs_16f_C3R_Ctx_BasicOperation) {
  const int width = 32;
  const int height = 32;
  const int channels = 3;

  std::vector<Npp16f> srcData(width * height * channels);
  std::vector<Npp16f> expectedData(width * height * channels);

  TestDataGenerator::generateRandom16f(srcData, -100.0f, 100.0f, 33333);

  for (size_t i = 0; i < expectedData.size(); i++) {
    float val = npp16f_to_float_host(srcData[i]);
    expectedData[i] = float_to_npp16f_host(std::fabs(val));
  }

  NppImageMemory<Npp16f> src(width * channels, height);
  NppImageMemory<Npp16f> dst(width * channels, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status = nppiAbs_16f_C3R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_16f_C3R_Ctx failed";

  std::vector<Npp16f> resultData(width * height * channels);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-3f))
      << "Abs 16f C3R Ctx operation produced incorrect results";
}

TEST_F(AbsFunctionalTest, Abs_16f_C3IR_InPlaceOperation) {
  const int width = 16;
  const int height = 16;
  const int channels = 3;

  std::vector<Npp16f> srcData(width * height * channels);
  std::vector<Npp16f> expectedData(width * height * channels);

  TestDataGenerator::generateRandom16f(srcData, -50.0f, 50.0f, 44444);

  for (size_t i = 0; i < expectedData.size(); i++) {
    float val = npp16f_to_float_host(srcData[i]);
    expectedData[i] = float_to_npp16f_host(std::fabs(val));
  }

  NppImageMemory<Npp16f> srcDst(width * channels, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiAbs_16f_C3IR(srcDst.get(), srcDst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_16f_C3IR failed";

  std::vector<Npp16f> resultData(width * height * channels);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-3f))
      << "In-place Abs 16f C3 operation produced incorrect results";
}

TEST_F(AbsFunctionalTest, Abs_16f_C3IR_Ctx_InPlaceOperation) {
  const int width = 16;
  const int height = 16;
  const int channels = 3;

  std::vector<Npp16f> srcData(width * height * channels);
  std::vector<Npp16f> expectedData(width * height * channels);

  TestDataGenerator::generateRandom16f(srcData, -50.0f, 50.0f, 44444);

  for (size_t i = 0; i < expectedData.size(); i++) {
    float val = npp16f_to_float_host(srcData[i]);
    expectedData[i] = float_to_npp16f_host(std::fabs(val));
  }

  NppImageMemory<Npp16f> srcDst(width * channels, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status = nppiAbs_16f_C3IR_Ctx(srcDst.get(), srcDst.step(), roi, ctx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_16f_C3IR_Ctx failed";

  std::vector<Npp16f> resultData(width * height * channels);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-3f))
      << "In-place Abs 16f C3 Ctx operation produced incorrect results";
}

// 16f C4 tests
TEST_F(AbsFunctionalTest, Abs_16f_C4R_BasicOperation) {
  const int width = 32;
  const int height = 32;
  const int channels = 4;

  std::vector<Npp16f> srcData(width * height * channels);
  std::vector<Npp16f> expectedData(width * height * channels);

  TestDataGenerator::generateRandom16f(srcData, -100.0f, 100.0f, 55555);

  for (size_t i = 0; i < expectedData.size(); i++) {
    float val = npp16f_to_float_host(srcData[i]);
    expectedData[i] = float_to_npp16f_host(std::fabs(val));
  }

  NppImageMemory<Npp16f> src(width * channels, height);
  NppImageMemory<Npp16f> dst(width * channels, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiAbs_16f_C4R(src.get(), src.step(), dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_16f_C4R failed";

  std::vector<Npp16f> resultData(width * height * channels);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-3f))
      << "Abs 16f C4R operation produced incorrect results";
}

TEST_F(AbsFunctionalTest, Abs_16f_C4R_Ctx_BasicOperation) {
  const int width = 32;
  const int height = 32;
  const int channels = 4;

  std::vector<Npp16f> srcData(width * height * channels);
  std::vector<Npp16f> expectedData(width * height * channels);

  TestDataGenerator::generateRandom16f(srcData, -100.0f, 100.0f, 55555);

  for (size_t i = 0; i < expectedData.size(); i++) {
    float val = npp16f_to_float_host(srcData[i]);
    expectedData[i] = float_to_npp16f_host(std::fabs(val));
  }

  NppImageMemory<Npp16f> src(width * channels, height);
  NppImageMemory<Npp16f> dst(width * channels, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status = nppiAbs_16f_C4R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_16f_C4R_Ctx failed";

  std::vector<Npp16f> resultData(width * height * channels);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-3f))
      << "Abs 16f C4R Ctx operation produced incorrect results";
}

TEST_F(AbsFunctionalTest, Abs_16f_C4IR_InPlaceOperation) {
  const int width = 16;
  const int height = 16;
  const int channels = 4;

  std::vector<Npp16f> srcData(width * height * channels);
  std::vector<Npp16f> expectedData(width * height * channels);

  TestDataGenerator::generateRandom16f(srcData, -50.0f, 50.0f, 66666);

  for (size_t i = 0; i < expectedData.size(); i++) {
    float val = npp16f_to_float_host(srcData[i]);
    expectedData[i] = float_to_npp16f_host(std::fabs(val));
  }

  NppImageMemory<Npp16f> srcDst(width * channels, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiAbs_16f_C4IR(srcDst.get(), srcDst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_16f_C4IR failed";

  std::vector<Npp16f> resultData(width * height * channels);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-3f))
      << "In-place Abs 16f C4 operation produced incorrect results";
}

TEST_F(AbsFunctionalTest, Abs_16f_C4IR_Ctx_InPlaceOperation) {
  const int width = 16;
  const int height = 16;
  const int channels = 4;

  std::vector<Npp16f> srcData(width * height * channels);
  std::vector<Npp16f> expectedData(width * height * channels);

  TestDataGenerator::generateRandom16f(srcData, -50.0f, 50.0f, 66666);

  for (size_t i = 0; i < expectedData.size(); i++) {
    float val = npp16f_to_float_host(srcData[i]);
    expectedData[i] = float_to_npp16f_host(std::fabs(val));
  }

  NppImageMemory<Npp16f> srcDst(width * channels, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status = nppiAbs_16f_C4IR_Ctx(srcDst.get(), srcDst.step(), roi, ctx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_16f_C4IR_Ctx failed";

  std::vector<Npp16f> resultData(width * height * channels);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-3f))
      << "In-place Abs 16f C4 Ctx operation produced incorrect results";
}
