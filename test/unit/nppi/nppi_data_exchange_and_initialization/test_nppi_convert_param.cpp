#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Convert 8u to 16u TEST_P ====================

struct Convert8u16uParam {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class Convert8u16uParamTest : public NppTestBase, public ::testing::WithParamInterface<Convert8u16uParam> {};

TEST_P(Convert8u16uParamTest, Convert_8u16u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  std::vector<Npp16u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::convert<Npp16u, Npp8u>(srcData[i]);
  }

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp16u> dst(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiConvert_8u16u_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiConvert_8u16u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp16u> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(Convert8u16u, Convert8u16uParamTest,
                         ::testing::Values(Convert8u16uParam{32, 32, false, "32x32_noCtx"},
                                           Convert8u16uParam{32, 32, true, "32x32_Ctx"},
                                           Convert8u16uParam{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<Convert8u16uParam> &info) { return info.param.name; });

// ==================== Convert 8u to 32f TEST_P ====================

struct Convert8u32fParam {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class Convert8u32fParamTest : public NppTestBase, public ::testing::WithParamInterface<Convert8u32fParam> {};

TEST_P(Convert8u32fParamTest, Convert_8u32f_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  std::vector<Npp32f> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::convert<Npp32f, Npp8u>(srcData[i]);
  }

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiConvert_8u32f_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiConvert_8u32f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32f> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
}

INSTANTIATE_TEST_SUITE_P(Convert8u32f, Convert8u32fParamTest,
                         ::testing::Values(Convert8u32fParam{32, 32, false, "32x32_noCtx"},
                                           Convert8u32fParam{32, 32, true, "32x32_Ctx"},
                                           Convert8u32fParam{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<Convert8u32fParam> &info) { return info.param.name; });

// ==================== Convert 32f to 8u TEST_P ====================

struct Convert32f8uParam {
  int width;
  int height;
  NppRoundMode rndMode;
  bool use_ctx;
  std::string name;
};

class Convert32f8uParamTest : public NppTestBase, public ::testing::WithParamInterface<Convert32f8uParam> {};

TEST_P(Convert32f8uParamTest, Convert_32f8u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, 0.0f, 255.0f, 12345);

  std::vector<Npp8u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::convert<Npp8u, Npp32f>(srcData[i]);
  }

  NppImageMemory<Npp32f> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiConvert_32f8u_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, param.rndMode, ctx);
  } else {
    status = nppiConvert_32f8u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, param.rndMode);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(Convert32f8u, Convert32f8uParamTest,
                         ::testing::Values(Convert32f8uParam{32, 32, NPP_RND_NEAR, false, "32x32_RndNear_noCtx"},
                                           Convert32f8uParam{32, 32, NPP_RND_NEAR, true, "32x32_RndNear_Ctx"},
                                           Convert32f8uParam{64, 64, NPP_RND_NEAR, false, "64x64_RndNear_noCtx"}),
                         [](const ::testing::TestParamInfo<Convert32f8uParam> &info) { return info.param.name; });

// ==================== Convert 16u to 32f TEST_P ====================

struct Convert16u32fParam {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class Convert16u32fParamTest : public NppTestBase, public ::testing::WithParamInterface<Convert16u32fParam> {};

TEST_P(Convert16u32fParamTest, Convert_16u32f_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);

  std::vector<Npp32f> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::convert<Npp32f, Npp16u>(srcData[i]);
  }

  NppImageMemory<Npp16u> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiConvert_16u32f_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiConvert_16u32f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32f> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
}

INSTANTIATE_TEST_SUITE_P(Convert16u32f, Convert16u32fParamTest,
                         ::testing::Values(Convert16u32fParam{32, 32, false, "32x32_noCtx"},
                                           Convert16u32fParam{32, 32, true, "32x32_Ctx"},
                                           Convert16u32fParam{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<Convert16u32fParam> &info) { return info.param.name; });
