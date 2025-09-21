// Implementation file

#include "../../framework/npp_test_base.h"

using namespace npp_functional_test;

class MulFunctionalTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }
};

TEST_F(MulFunctionalTest, Mul_8u_C1RSfs_BasicOperation) {
  const int width = 32;
  const int height = 32;
  const int scaleFactor = 0;

  std::vector<Npp8u> srcData1(width * height);
  std::vector<Npp8u> srcData2(width * height);
  std::vector<Npp8u> expectedData(width * height);

  TestDataGenerator::generateConstant(srcData1, static_cast<Npp8u>(10));
  TestDataGenerator::generateConstant(srcData2, static_cast<Npp8u>(5));
  TestDataGenerator::generateConstant(expectedData, static_cast<Npp8u>(50));

  NppImageMemory<Npp8u> src1(width, height);
  NppImageMemory<Npp8u> src2(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  src1.copyFromHost(srcData1);
  src2.copyFromHost(srcData2);

  NppiSize roi = {width, height};
  NppStatus status =
      nppiMul_8u_C1RSfs(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, scaleFactor);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiMul_8u_C1RSfs failed";

  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData)) << "Mul operation produced incorrect results";
}

TEST_F(MulFunctionalTest, Mul_32f_C1R_BasicOperation) {
  const int width = 32;
  const int height = 32;

  std::vector<Npp32f> srcData1(width * height);
  std::vector<Npp32f> srcData2(width * height);
  std::vector<Npp32f> expectedData(width * height);

  TestDataGenerator::generateConstant(srcData1, 2.5f);
  TestDataGenerator::generateConstant(srcData2, 4.0f);
  TestDataGenerator::generateConstant(expectedData, 10.0f);

  NppImageMemory<Npp32f> src1(width, height);
  NppImageMemory<Npp32f> src2(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  src1.copyFromHost(srcData1);
  src2.copyFromHost(srcData2);

  NppiSize roi = {width, height};
  NppStatus status = nppiMul_32f_C1R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiMul_32f_C1R failed";

  std::vector<Npp32f> resultData(width * height);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f))
      << "Mul 32f operation produced incorrect results";
}