#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== AbsDiffC 8u C1 TEST_P ====================

struct AbsDiffC8uParam {
  int width;
  int height;
  Npp8u constant;
  bool use_ctx;
  std::string name;
};

class AbsDiffC8uParamTest : public NppTestBase, public ::testing::WithParamInterface<AbsDiffC8uParam> {};

TEST_P(AbsDiffC8uParamTest, AbsDiffC_8u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp8u constant = param.constant;

  std::vector<Npp8u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  std::vector<Npp8u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::abs_diff_c<Npp8u>(srcData[i], constant);
  }

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiAbsDiffC_8u_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, constant, ctx);
  } else {
    status = nppiAbsDiffC_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, constant);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(AbsDiffC8u, AbsDiffC8uParamTest,
                         ::testing::Values(AbsDiffC8uParam{32, 32, 50, false, "32x32_c50_noCtx"},
                                           AbsDiffC8uParam{32, 32, 50, true, "32x32_c50_Ctx"},
                                           AbsDiffC8uParam{32, 32, 128, false, "32x32_c128_noCtx"},
                                           AbsDiffC8uParam{64, 64, 100, false, "64x64_c100_noCtx"}),
                         [](const ::testing::TestParamInfo<AbsDiffC8uParam> &info) { return info.param.name; });

// ==================== AbsDiffC 16u C1 TEST_P ====================

struct AbsDiffC16uParam {
  int width;
  int height;
  Npp16u constant;
  bool use_ctx;
  std::string name;
};

class AbsDiffC16uParamTest : public NppTestBase, public ::testing::WithParamInterface<AbsDiffC16uParam> {};

TEST_P(AbsDiffC16uParamTest, AbsDiffC_16u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp16u constant = param.constant;

  std::vector<Npp16u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);

  std::vector<Npp16u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::abs_diff_c<Npp16u>(srcData[i], constant);
  }

  NppImageMemory<Npp16u> src(width, height);
  NppImageMemory<Npp16u> dst(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiAbsDiffC_16u_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, constant, ctx);
  } else {
    status = nppiAbsDiffC_16u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, constant);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp16u> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(AbsDiffC16u, AbsDiffC16uParamTest,
                         ::testing::Values(AbsDiffC16uParam{32, 32, 1000, false, "32x32_c1000_noCtx"},
                                           AbsDiffC16uParam{32, 32, 1000, true, "32x32_c1000_Ctx"},
                                           AbsDiffC16uParam{32, 32, 32768, false, "32x32_c32768_noCtx"},
                                           AbsDiffC16uParam{64, 64, 5000, false, "64x64_c5000_noCtx"}),
                         [](const ::testing::TestParamInfo<AbsDiffC16uParam> &info) { return info.param.name; });

// ==================== AbsDiffC 32f C1 TEST_P ====================

struct AbsDiffC32fParam {
  int width;
  int height;
  Npp32f constant;
  bool use_ctx;
  std::string name;
};

class AbsDiffC32fParamTest : public NppTestBase, public ::testing::WithParamInterface<AbsDiffC32fParam> {};

TEST_P(AbsDiffC32fParamTest, AbsDiffC_32f_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp32f constant = param.constant;

  std::vector<Npp32f> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, -100.0f, 100.0f, 12345);

  std::vector<Npp32f> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::abs_diff_c<Npp32f>(srcData[i], constant);
  }

  NppImageMemory<Npp32f> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiAbsDiffC_32f_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, constant, ctx);
  } else {
    status = nppiAbsDiffC_32f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, constant);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32f> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
}

INSTANTIATE_TEST_SUITE_P(AbsDiffC32f, AbsDiffC32fParamTest,
                         ::testing::Values(AbsDiffC32fParam{32, 32, 25.5f, false, "32x32_c25p5_noCtx"},
                                           AbsDiffC32fParam{32, 32, 25.5f, true, "32x32_c25p5_Ctx"},
                                           AbsDiffC32fParam{32, 32, -50.0f, false, "32x32_cm50_noCtx"},
                                           AbsDiffC32fParam{64, 64, 10.0f, false, "64x64_c10_noCtx"}),
                         [](const ::testing::TestParamInfo<AbsDiffC32fParam> &info) { return info.param.name; });
