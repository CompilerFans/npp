/**
 * @file test_nppi_sub.cpp
 * @brief NPP 减法函数测试
 */

#include "../../../common/npp_test_utils.h"
#include "../../framework/npp_test_base.h"

using namespace npp_functional_test;

class SubFunctionalTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }
};

// NOTE: 此测试被禁用 - NVIDIA NPP的nppiSub_8u_C1RSfs函数存在严重缺陷
// 该函数总是返回0而非正确的减法结果，这是NVIDIA NPP库的已知问题
TEST_F(SubFunctionalTest, Sub_8u_C1RSfs_BasicOperation) {
  const int width = 32;
  const int height = 32;
  const int scaleFactor = 0;

  std::vector<Npp8u> srcData1(width * height);
  std::vector<Npp8u> srcData2(width * height);
  std::vector<Npp8u> expectedData(width * height);

  // NVIDIA NPP subtraction: pDst = pSrc2 - pSrc1, not pSrc1 - pSrc2
  // So to get 100-30=70, we need src1=30, src2=100
  TestDataGenerator::generateConstant(srcData1, static_cast<Npp8u>(30));     // subtrahend
  TestDataGenerator::generateConstant(srcData2, static_cast<Npp8u>(100));    // minuend
  TestDataGenerator::generateConstant(expectedData, static_cast<Npp8u>(70)); // 100-30=70

  NppImageMemory<Npp8u> src1(width, height);
  NppImageMemory<Npp8u> src2(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  src1.copyFromHost(srcData1);
  src2.copyFromHost(srcData2);

  NppiSize roi = {width, height};
  NppStatus status =
      nppiSub_8u_C1RSfs(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, scaleFactor);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiSub_8u_C1RSfs failed";

  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);

  // Verify results using precision control system for sub operation
  for (size_t i = 0; i < resultData.size(); i++) {
    NPP_EXPECT_ARITHMETIC_EQUAL(resultData[i], expectedData[i], "nppiSub_8u_C1RSfs")
        << "Sub operation mismatch at index " << i << ": got " << (int)resultData[i] << ", expected "
        << (int)expectedData[i];
  }
}

// NOTE: 此测试被禁用 - NVIDIA NPP的减法函数存在参数顺序和行为问题
// 需要进一步研究其确切行为模式
TEST_F(SubFunctionalTest, Sub_32f_C1R_BasicOperation) {
  const int width = 32;
  const int height = 32;

  std::vector<Npp32f> srcData1(width * height);
  std::vector<Npp32f> srcData2(width * height);
  std::vector<Npp32f> expectedData(width * height);

  // NVIDIA NPP subtraction: pDst = pSrc2 - pSrc1, not pSrc1 - pSrc2
  // So to get 10.5-3.5=7.0, we need src1=3.5, src2=10.5
  TestDataGenerator::generateConstant(srcData1, 3.5f);     // subtrahend
  TestDataGenerator::generateConstant(srcData2, 10.5f);    // minuend
  TestDataGenerator::generateConstant(expectedData, 7.0f); // 10.5-3.5=7.0

  NppImageMemory<Npp32f> src1(width, height);
  NppImageMemory<Npp32f> src2(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  src1.copyFromHost(srcData1);
  src2.copyFromHost(srcData2);

  NppiSize roi = {width, height};
  NppStatus status = nppiSub_32f_C1R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiSub_32f_C1R failed";

  std::vector<Npp32f> resultData(width * height);
  dst.copyToHost(resultData);

  // Verify results using precision control system for sub operation
  for (size_t i = 0; i < resultData.size(); i++) {
    NPP_EXPECT_ARITHMETIC_EQUAL(resultData[i], expectedData[i], "nppiSub_32f_C1R")
        << "Sub 32f operation mismatch at index " << i << ": got " << resultData[i] << ", expected " << expectedData[i];
  }
}