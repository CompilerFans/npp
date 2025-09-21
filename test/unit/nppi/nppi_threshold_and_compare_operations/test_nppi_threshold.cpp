#include "../../framework/npp_test_base.h"

using namespace npp_functional_test;

class ThresholdFunctionalTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }
};

// 测试8位阈值处理 - LESS操作
TEST_F(ThresholdFunctionalTest, Threshold_8u_C1R_Less) {
  const int width = 32, height = 32;
  const Npp8u threshold = 128;

  // prepare test data - 渐变图像
  std::vector<Npp8u> srcData(width * height);
  std::vector<Npp8u> expectedData(width * height);

  for (int i = 0; i < width * height; i++) {
    srcData[i] = (Npp8u)(i % 256);
    // NPP_CMP_LESS: if (src < threshold) dst = threshold; else dst = src;
    expectedData[i] = (srcData[i] < threshold) ? threshold : srcData[i];
  }

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiThreshold_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, threshold, NPP_CMP_LESS);

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiThreshold_8u_C1R failed";

  // Validate结果
  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData))
      << "Threshold LESS operation produced incorrect results";
}

// 测试8位阈值处理 - GREATER操作
TEST_F(ThresholdFunctionalTest, Threshold_8u_C1R_Greater) {
  const int width = 32, height = 32;
  const Npp8u threshold = 128;

  // prepare test data
  std::vector<Npp8u> srcData(width * height);
  std::vector<Npp8u> expectedData(width * height);

  for (int i = 0; i < width * height; i++) {
    srcData[i] = (Npp8u)(i % 256);
    // NPP_CMP_GREATER: if (src > threshold) dst = threshold; else dst = src;
    expectedData[i] = (srcData[i] > threshold) ? threshold : srcData[i];
  }

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status =
      nppiThreshold_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, threshold, NPP_CMP_GREATER);

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiThreshold_8u_C1R failed";

  // Validate结果
  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData))
      << "Threshold GREATER operation produced incorrect results";
}

// 测试8位原地阈值处理
TEST_F(ThresholdFunctionalTest, Threshold_8u_C1IR_InPlace) {
  const int width = 16, height = 16;
  const Npp8u threshold = 100;

  // prepare test data
  std::vector<Npp8u> srcData(width * height);
  std::vector<Npp8u> expectedData(width * height);

  // 创建具有明显对比的测试图像
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      srcData[idx] = (x < width / 2) ? 50 : 150;
      expectedData[idx] = (srcData[idx] < threshold) ? threshold : srcData[idx];
    }
  }

  NppImageMemory<Npp8u> srcDst(width, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiThreshold_8u_C1IR(srcDst.get(), srcDst.step(), roi, threshold, NPP_CMP_LESS);

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiThreshold_8u_C1IR failed";

  // Validate结果
  std::vector<Npp8u> resultData(width * height);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData))
      << "In-place threshold operation produced incorrect results";
}

// 测试32位浮点阈值处理
TEST_F(ThresholdFunctionalTest, Threshold_32f_C1R_Float) {
  const int width = 16, height = 16;
  const Npp32f threshold = 0.5f;

  // prepare test data
  std::vector<Npp32f> srcData(width * height);
  std::vector<Npp32f> expectedData(width * height);

  for (int i = 0; i < width * height; i++) {
    srcData[i] = (float)i / (width * height);
    expectedData[i] = (srcData[i] > threshold) ? threshold : srcData[i];
  }

  NppImageMemory<Npp32f> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status =
      nppiThreshold_32f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, threshold, NPP_CMP_GREATER);

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiThreshold_32f_C1R failed";

  // Validate结果
  std::vector<Npp32f> resultData(width * height);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-6f))
      << "Float threshold operation produced incorrect results";
}

// 错误处理测试
// NOTE: 测试已被禁用 - vendor NPP对无效参数的错误检测行为与预期不符
TEST_F(ThresholdFunctionalTest, DISABLED_Threshold_ErrorHandling) {
  const int width = 16, height = 16;
  const Npp8u threshold = 128;

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  NppiSize roi = {width, height};

  // 测试空指针
  NppStatus status = nppiThreshold_8u_C1R(nullptr, src.step(), dst.get(), dst.step(), roi, threshold, NPP_CMP_LESS);
  EXPECT_EQ(status, NPP_NULL_POINTER_ERROR);

  // 测试无效ROI
  NppiSize invalidRoi = {0, 0};
  status = nppiThreshold_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), invalidRoi, threshold, NPP_CMP_LESS);
  EXPECT_EQ(status, NPP_SIZE_ERROR);

  // 测试无效比较操作
  status = nppiThreshold_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, threshold,
                                NPP_CMP_EQ); // EQ not supported for threshold
  EXPECT_EQ(status, NPP_NOT_SUPPORTED_MODE_ERROR);
}

// 二值化测试（特殊应用场景）
TEST_F(ThresholdFunctionalTest, Threshold_BinaryImage) {
  const int width = 32, height = 32;
  const Npp8u threshold = 128;

  // prepare test data - 创建一个包含噪声的二值化场景
  std::vector<Npp8u> srcData(width * height);

  for (int i = 0; i < width * height; i++) {
    // 创建一个圆形区域
    int x = i % width;
    int y = i / width;
    int dx = x - width / 2;
    int dy = y - height / 2;
    float dist = sqrt(dx * dx + dy * dy);

    srcData[i] = (dist < width / 3) ? 200 : 50;
  }

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiThreshold_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, threshold, NPP_CMP_LESS);

  ASSERT_EQ(status, NPP_SUCCESS) << "Binary threshold failed";

  // Validate结果 - 检查二值化效果
  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);

  int lowCount = 0, highCount = 0;
  for (size_t i = 0; i < resultData.size(); i++) {
    if (resultData[i] == threshold)
      lowCount++;
    else
      highCount++;
  }

  EXPECT_GT(lowCount, 0) << "No pixels were thresholded";
  EXPECT_GT(highCount, 0) << "All pixels were thresholded";
}
