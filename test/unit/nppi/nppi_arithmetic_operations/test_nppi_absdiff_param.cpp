#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== AbsDiff 8u C1 TEST_P ====================

struct AbsDiff8uParam {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class AbsDiff8uParamTest : public NppTestBase, public ::testing::WithParamInterface<AbsDiff8uParam> {};

TEST_P(AbsDiff8uParamTest, AbsDiff_8u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> src1Data(width * height);
  std::vector<Npp8u> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 54321);

  std::vector<Npp8u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::abs_diff<Npp8u>(src1Data[i], src2Data[i]);
  }

  NppImageMemory<Npp8u> src1(width, height);
  NppImageMemory<Npp8u> src2(width, height);
  NppImageMemory<Npp8u> dst(width, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiAbsDiff_8u_C1R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiAbsDiff_8u_C1R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(AbsDiff8u, AbsDiff8uParamTest,
                         ::testing::Values(AbsDiff8uParam{32, 32, false, "32x32_noCtx"},
                                           AbsDiff8uParam{32, 32, true, "32x32_Ctx"},
                                           AbsDiff8uParam{64, 64, false, "64x64_noCtx"},
                                           AbsDiff8uParam{64, 64, true, "64x64_Ctx"}),
                         [](const ::testing::TestParamInfo<AbsDiff8uParam> &info) { return info.param.name; });

// ==================== AbsDiff 16u C1 TEST_P ====================

struct AbsDiff16uParam {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class AbsDiff16uParamTest : public NppTestBase, public ::testing::WithParamInterface<AbsDiff16uParam> {};

TEST_P(AbsDiff16uParamTest, AbsDiff_16u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> src1Data(width * height);
  std::vector<Npp16u> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 54321);

  std::vector<Npp16u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::abs_diff<Npp16u>(src1Data[i], src2Data[i]);
  }

  NppImageMemory<Npp16u> src1(width, height);
  NppImageMemory<Npp16u> src2(width, height);
  NppImageMemory<Npp16u> dst(width, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiAbsDiff_16u_C1R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiAbsDiff_16u_C1R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp16u> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(AbsDiff16u, AbsDiff16uParamTest,
                         ::testing::Values(AbsDiff16uParam{32, 32, false, "32x32_noCtx"},
                                           AbsDiff16uParam{32, 32, true, "32x32_Ctx"},
                                           AbsDiff16uParam{64, 64, false, "64x64_noCtx"},
                                           AbsDiff16uParam{64, 64, true, "64x64_Ctx"}),
                         [](const ::testing::TestParamInfo<AbsDiff16uParam> &info) { return info.param.name; });

// ==================== AbsDiff 32f C1 TEST_P ====================

struct AbsDiff32fParam {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class AbsDiff32fParamTest : public NppTestBase, public ::testing::WithParamInterface<AbsDiff32fParam> {};

TEST_P(AbsDiff32fParamTest, AbsDiff_32f_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> src1Data(width * height);
  std::vector<Npp32f> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, -100.0f, 100.0f, 12345);
  TestDataGenerator::generateRandom(src2Data, -100.0f, 100.0f, 54321);

  std::vector<Npp32f> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::abs_diff<Npp32f>(src1Data[i], src2Data[i]);
  }

  NppImageMemory<Npp32f> src1(width, height);
  NppImageMemory<Npp32f> src2(width, height);
  NppImageMemory<Npp32f> dst(width, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiAbsDiff_32f_C1R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiAbsDiff_32f_C1R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32f> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
}

INSTANTIATE_TEST_SUITE_P(AbsDiff32f, AbsDiff32fParamTest,
                         ::testing::Values(AbsDiff32fParam{32, 32, false, "32x32_noCtx"},
                                           AbsDiff32fParam{32, 32, true, "32x32_Ctx"},
                                           AbsDiff32fParam{64, 64, false, "64x64_noCtx"},
                                           AbsDiff32fParam{64, 64, true, "64x64_Ctx"}),
                         [](const ::testing::TestParamInfo<AbsDiff32fParam> &info) { return info.param.name; });

// ==================== AbsDiff 8u C3 TEST_P ====================

struct AbsDiff8uC3Param {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class AbsDiff8uC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<AbsDiff8uC3Param> {};

TEST_P(AbsDiff8uC3ParamTest, AbsDiff_8u_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 3;

  std::vector<Npp8u> src1Data(width * height * channels);
  std::vector<Npp8u> src2Data(width * height * channels);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 54321);

  std::vector<Npp8u> expectedData(width * height * channels);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::abs_diff<Npp8u>(src1Data[i], src2Data[i]);
  }

  NppImageMemory<Npp8u> src1(width * channels, height);
  NppImageMemory<Npp8u> src2(width * channels, height);
  NppImageMemory<Npp8u> dst(width * channels, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiAbsDiff_8u_C3R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiAbsDiff_8u_C3R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(width * height * channels);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(AbsDiff8uC3, AbsDiff8uC3ParamTest,
                         ::testing::Values(AbsDiff8uC3Param{32, 32, false, "32x32_noCtx"},
                                           AbsDiff8uC3Param{32, 32, true, "32x32_Ctx"},
                                           AbsDiff8uC3Param{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AbsDiff8uC3Param> &info) { return info.param.name; });

// ==================== AbsDiff 8u C4 TEST_P ====================

struct AbsDiff8uC4Param {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class AbsDiff8uC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<AbsDiff8uC4Param> {};

TEST_P(AbsDiff8uC4ParamTest, AbsDiff_8u_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;

  std::vector<Npp8u> src1Data(width * height * channels);
  std::vector<Npp8u> src2Data(width * height * channels);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 54321);

  std::vector<Npp8u> expectedData(width * height * channels);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::abs_diff<Npp8u>(src1Data[i], src2Data[i]);
  }

  NppImageMemory<Npp8u> src1(width * channels, height);
  NppImageMemory<Npp8u> src2(width * channels, height);
  NppImageMemory<Npp8u> dst(width * channels, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiAbsDiff_8u_C4R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiAbsDiff_8u_C4R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(width * height * channels);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(AbsDiff8uC4, AbsDiff8uC4ParamTest,
                         ::testing::Values(AbsDiff8uC4Param{32, 32, false, "32x32_noCtx"},
                                           AbsDiff8uC4Param{32, 32, true, "32x32_Ctx"},
                                           AbsDiff8uC4Param{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AbsDiff8uC4Param> &info) { return info.param.name; });
