#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== MaxEvery 8u C1 TEST_P ====================

struct MaxEvery8uParam {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class MaxEvery8uParamTest : public NppTestBase, public ::testing::WithParamInterface<MaxEvery8uParam> {};

TEST_P(MaxEvery8uParamTest, MaxEvery_8u_C1IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height);
  std::vector<Npp8u> dstData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 54321);

  std::vector<Npp8u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::max_every<Npp8u>(srcData[i], dstData[i]);
  }

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);
  src.copyFromHost(srcData);
  dst.copyFromHost(dstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiMaxEvery_8u_C1IR_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMaxEvery_8u_C1IR(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(MaxEvery8u, MaxEvery8uParamTest,
                         ::testing::Values(MaxEvery8uParam{32, 32, false, "32x32_noCtx"},
                                           MaxEvery8uParam{32, 32, true, "32x32_Ctx"},
                                           MaxEvery8uParam{64, 64, false, "64x64_noCtx"},
                                           MaxEvery8uParam{64, 64, true, "64x64_Ctx"}),
                         [](const ::testing::TestParamInfo<MaxEvery8uParam> &info) { return info.param.name; });

// ==================== MaxEvery 16u C1 TEST_P ====================

struct MaxEvery16uParam {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class MaxEvery16uParamTest : public NppTestBase, public ::testing::WithParamInterface<MaxEvery16uParam> {};

TEST_P(MaxEvery16uParamTest, MaxEvery_16u_C1IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcData(width * height);
  std::vector<Npp16u> dstData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 54321);

  std::vector<Npp16u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::max_every<Npp16u>(srcData[i], dstData[i]);
  }

  NppImageMemory<Npp16u> src(width, height);
  NppImageMemory<Npp16u> dst(width, height);
  src.copyFromHost(srcData);
  dst.copyFromHost(dstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiMaxEvery_16u_C1IR_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMaxEvery_16u_C1IR(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp16u> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(MaxEvery16u, MaxEvery16uParamTest,
                         ::testing::Values(MaxEvery16uParam{32, 32, false, "32x32_noCtx"},
                                           MaxEvery16uParam{32, 32, true, "32x32_Ctx"},
                                           MaxEvery16uParam{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<MaxEvery16uParam> &info) { return info.param.name; });

// ==================== MaxEvery 32f C1 TEST_P ====================

struct MaxEvery32fParam {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class MaxEvery32fParamTest : public NppTestBase, public ::testing::WithParamInterface<MaxEvery32fParam> {};

TEST_P(MaxEvery32fParamTest, MaxEvery_32f_C1IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> srcData(width * height);
  std::vector<Npp32f> dstData(width * height);
  TestDataGenerator::generateRandom(srcData, -100.0f, 100.0f, 12345);
  TestDataGenerator::generateRandom(dstData, -100.0f, 100.0f, 54321);

  std::vector<Npp32f> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::max_every<Npp32f>(srcData[i], dstData[i]);
  }

  NppImageMemory<Npp32f> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);
  src.copyFromHost(srcData);
  dst.copyFromHost(dstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiMaxEvery_32f_C1IR_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMaxEvery_32f_C1IR(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32f> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
}

INSTANTIATE_TEST_SUITE_P(MaxEvery32f, MaxEvery32fParamTest,
                         ::testing::Values(MaxEvery32fParam{32, 32, false, "32x32_noCtx"},
                                           MaxEvery32fParam{32, 32, true, "32x32_Ctx"},
                                           MaxEvery32fParam{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<MaxEvery32fParam> &info) { return info.param.name; });
