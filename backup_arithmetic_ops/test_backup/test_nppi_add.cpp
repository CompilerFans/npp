#include "npp_test_base.h"

using namespace npp_functional_test;

class AddFunctionalTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }
};

// ==================== 8位无符号整数测试 ====================

TEST_F(AddFunctionalTest, Add_8u_C1RSfs_BasicOperation) {
  const int width = 64;
  const int height = 64;
  const int scaleFactor = 0;

  // prepare test data
  std::vector<Npp8u> srcData1(width * height);
  std::vector<Npp8u> srcData2(width * height);
  std::vector<Npp8u> expectedData(width * height);

  TestDataGenerator::generateConstant(srcData1, static_cast<Npp8u>(50));
  TestDataGenerator::generateConstant(srcData2, static_cast<Npp8u>(30));
  TestDataGenerator::generateConstant(expectedData, static_cast<Npp8u>(80));

  // 分配GPU内存
  NppImageMemory<Npp8u> src1(width, height);
  NppImageMemory<Npp8u> src2(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  // 复制数据到GPU
  src1.copyFromHost(srcData1);
  src2.copyFromHost(srcData2);

  // 执行Add操作
  NppiSize roi = {width, height};
  NppStatus status =
      nppiAdd_8u_C1RSfs(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, scaleFactor);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAdd_8u_C1RSfs failed";

  // Validate结果
  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData)) << "Add operation produced incorrect results";
}

TEST_F(AddFunctionalTest, Add_8u_C1RSfs_WithScaling) {
  const int width = 32;
  const int height = 32;
  const int scaleFactor = 2; // 右移2位，除以4

  // prepare test data
  std::vector<Npp8u> srcData1(width * height);
  std::vector<Npp8u> srcData2(width * height);
  std::vector<Npp8u> expectedData(width * height);

  TestDataGenerator::generateConstant(srcData1, static_cast<Npp8u>(100));
  TestDataGenerator::generateConstant(srcData2, static_cast<Npp8u>(60));
  // 期望结果: (100 + 60 + 2) >> 2 = 162 >> 2 = 40
  TestDataGenerator::generateConstant(expectedData, static_cast<Npp8u>(40));

  // 分配GPU内存
  NppImageMemory<Npp8u> src1(width, height);
  NppImageMemory<Npp8u> src2(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  // 复制数据到GPU
  src1.copyFromHost(srcData1);
  src2.copyFromHost(srcData2);

  // 执行Add操作
  NppiSize roi = {width, height};
  NppStatus status =
      nppiAdd_8u_C1RSfs(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, scaleFactor);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAdd_8u_C1RSfs with scaling failed";

  // Validate结果
  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData))
      << "Add operation with scaling produced incorrect results";
}

TEST_F(AddFunctionalTest, Add_8u_C1RSfs_SaturationHandling) {
  const int width = 16;
  const int height = 16;
  const int scaleFactor = 0;

  // prepare test data - 会导致饱和的情况
  std::vector<Npp8u> srcData1(width * height);
  std::vector<Npp8u> srcData2(width * height);
  std::vector<Npp8u> expectedData(width * height);

  TestDataGenerator::generateConstant(srcData1, static_cast<Npp8u>(200));
  TestDataGenerator::generateConstant(srcData2, static_cast<Npp8u>(100));
  // 期望结果: 200 + 100 = 300，饱和到255
  TestDataGenerator::generateConstant(expectedData, static_cast<Npp8u>(255));

  // 分配GPU内存
  NppImageMemory<Npp8u> src1(width, height);
  NppImageMemory<Npp8u> src2(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  // 复制数据到GPU
  src1.copyFromHost(srcData1);
  src2.copyFromHost(srcData2);

  // 执行Add操作
  NppiSize roi = {width, height};
  NppStatus status =
      nppiAdd_8u_C1RSfs(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, scaleFactor);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAdd_8u_C1RSfs saturation test failed";

  // Validate结果
  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData)) << "Add operation saturation handling failed";
}

// ==================== 32位浮点数测试 ====================

TEST_F(AddFunctionalTest, Add_32f_C1R_BasicOperation) {
  const int width = 64;
  const int height = 64;

  // prepare test data
  std::vector<Npp32f> srcData1(width * height);
  std::vector<Npp32f> srcData2(width * height);
  std::vector<Npp32f> expectedData(width * height);

  TestDataGenerator::generateConstant(srcData1, 1.5f);
  TestDataGenerator::generateConstant(srcData2, 2.5f);
  TestDataGenerator::generateConstant(expectedData, 4.0f);

  // 分配GPU内存
  NppImageMemory<Npp32f> src1(width, height);
  NppImageMemory<Npp32f> src2(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  // 复制数据到GPU
  src1.copyFromHost(srcData1);
  src2.copyFromHost(srcData2);

  // 执行Add操作
  NppiSize roi = {width, height};
  NppStatus status = nppiAdd_32f_C1R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAdd_32f_C1R failed";

  // Validate结果
  std::vector<Npp32f> resultData(width * height);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f))
      << "Add 32f operation produced incorrect results";
}

TEST_F(AddFunctionalTest, Add_32f_C1R_RandomData) {
  const int width = 128;
  const int height = 128;
  const unsigned seed = 12345; // 固定种子保证可重现性

  // 准备随机测试数据
  std::vector<Npp32f> srcData1(width * height);
  std::vector<Npp32f> srcData2(width * height);
  std::vector<Npp32f> expectedData(width * height);

  TestDataGenerator::generateRandom(srcData1, -100.0f, 100.0f, seed);
  TestDataGenerator::generateRandom(srcData2, -100.0f, 100.0f, seed + 1);

  // 计算期望结果
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = srcData1[i] + srcData2[i];
  }

  // 分配GPU内存
  NppImageMemory<Npp32f> src1(width, height);
  NppImageMemory<Npp32f> src2(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  // 复制数据到GPU
  src1.copyFromHost(srcData1);
  src2.copyFromHost(srcData2);

  // 执行Add操作
  NppiSize roi = {width, height};
  NppStatus status = nppiAdd_32f_C1R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAdd_32f_C1R with random data failed";

  // Validate结果
  std::vector<Npp32f> resultData(width * height);
  dst.copyToHost(resultData);

  auto [isMatch, mismatchIndex] = ResultValidator::findFirstMismatch(resultData, expectedData, 1e-5f);
  EXPECT_TRUE(isMatch) << "First mismatch at index " << mismatchIndex << ", expected: " << expectedData[mismatchIndex]
                       << ", got: " << resultData[mismatchIndex];
}

// ==================== In-place 操作测试 ====================

TEST_F(AddFunctionalTest, Add_32f_C1IR_InPlaceOperation) {
  const int width = 32;
  const int height = 32;

  // prepare test data
  std::vector<Npp32f> srcData(width * height);
  std::vector<Npp32f> srcDstData(width * height);
  std::vector<Npp32f> expectedData(width * height);

  TestDataGenerator::generateConstant(srcData, 3.0f);
  TestDataGenerator::generateConstant(srcDstData, 2.0f);
  TestDataGenerator::generateConstant(expectedData, 5.0f);

  // 分配GPU内存
  NppImageMemory<Npp32f> src(width, height);
  NppImageMemory<Npp32f> srcDst(width, height);

  // 复制数据到GPU
  src.copyFromHost(srcData);
  srcDst.copyFromHost(srcDstData);

  // 执行In-place Add操作
  NppiSize roi = {width, height};
  NppStatus status = nppiAdd_32f_C1IR(src.get(), src.step(), srcDst.get(), srcDst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAdd_32f_C1IR failed";

  // Validate结果
  std::vector<Npp32f> resultData(width * height);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f))
      << "In-place Add operation produced incorrect results";
}

// ==================== 错误处理测试 ====================

// Test error handling for null pointers
TEST_F(AddFunctionalTest, DISABLED_Add_ErrorHandling_NullPointer) {
  const int width = 16;
  const int height = 16;

  NppImageMemory<Npp32f> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  NppiSize roi = {width, height};

  // 测试null指针
  NppStatus status = nppiAdd_32f_C1R(nullptr, src.step(), src.get(), src.step(), dst.get(), dst.step(), roi);

  EXPECT_EQ(status, NPP_NULL_POINTER_ERROR) << "Should detect null pointer";
}

// Test error handling for invalid ROI
TEST_F(AddFunctionalTest, DISABLED_Add_ErrorHandling_InvalidROI) {
  const int width = 16;
  const int height = 16;

  NppImageMemory<Npp32f> src1(width, height);
  NppImageMemory<Npp32f> src2(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  // 测试无效的ROI
  NppiSize roi = {0, height}; // 宽度为0
  NppStatus status = nppiAdd_32f_C1R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);

  EXPECT_EQ(status, NPP_NO_ERROR) << "vendor NPP returns success for zero-size ROI";
}

// ==================== 边界情况测试 ====================

TEST_F(AddFunctionalTest, Add_BoundaryConditions_SmallImage) {
  const int width = 1;
  const int height = 1;

  // 最小图像尺寸测试
  std::vector<Npp32f> srcData1(1, 10.0f);
  std::vector<Npp32f> srcData2(1, 20.0f);

  NppImageMemory<Npp32f> src1(width, height);
  NppImageMemory<Npp32f> src2(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  src1.copyFromHost(srcData1);
  src2.copyFromHost(srcData2);

  NppiSize roi = {width, height};
  NppStatus status = nppiAdd_32f_C1R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "1x1 image should work";

  std::vector<Npp32f> result(1);
  dst.copyToHost(result);

  EXPECT_FLOAT_EQ(result[0], 30.0f) << "1x1 image addition failed";
}

TEST_F(AddFunctionalTest, Add_BoundaryConditions_LargeImage) {
  // 跳过大图像测试，如果GPU内存不足
  if (!hasComputeCapability(3, 5)) {
    GTEST_SKIP() << "GPU compute capability too low for large image test";
  }

  const int width = 2048;
  const int height = 2048;

  // 大图像测试
  NppImageMemory<Npp32f> src1(width, height);
  NppImageMemory<Npp32f> src2(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  // 填充常数值
  src1.fill(1.0f);
  src2.fill(2.0f);

  NppiSize roi = {width, height};
  NppStatus status = nppiAdd_32f_C1R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);

  EXPECT_EQ(status, NPP_NO_ERROR) << "Large image addition should work";

  // 仅检查一小部分结果以Validate正确性
  std::vector<Npp32f> sample(100);
  cudaMemcpy2D(sample.data(), 10 * sizeof(Npp32f), dst.get(), dst.step(), 10 * sizeof(Npp32f), 10,
               cudaMemcpyDeviceToHost);

  for (int i = 0; i < 100; i++) {
    EXPECT_FLOAT_EQ(sample[i], 3.0f) << "Large image sample check failed";
  }
}
