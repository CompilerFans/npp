#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Set 8u C1 TEST_P ====================

struct Set8uParam {
  int width;
  int height;
  Npp8u value;
  bool use_ctx;
  std::string name;
};

class Set8uParamTest : public NppTestBase, public ::testing::WithParamInterface<Set8uParam> {};

TEST_P(Set8uParamTest, Set_8u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp8u value = param.value;

  std::vector<Npp8u> expectedData(width * height, value);

  NppImageMemory<Npp8u> dst(width, height);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiSet_8u_C1R_Ctx(value, dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiSet_8u_C1R(value, dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(Set8u, Set8uParamTest,
                         ::testing::Values(Set8uParam{32, 32, 128, false, "32x32_v128_noCtx"},
                                           Set8uParam{32, 32, 128, true, "32x32_v128_Ctx"},
                                           Set8uParam{64, 64, 0, false, "64x64_v0_noCtx"},
                                           Set8uParam{64, 64, 255, false, "64x64_v255_noCtx"}),
                         [](const ::testing::TestParamInfo<Set8uParam> &info) { return info.param.name; });

// ==================== Set 16u C1 TEST_P ====================

struct Set16uParam {
  int width;
  int height;
  Npp16u value;
  bool use_ctx;
  std::string name;
};

class Set16uParamTest : public NppTestBase, public ::testing::WithParamInterface<Set16uParam> {};

TEST_P(Set16uParamTest, Set_16u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp16u value = param.value;

  std::vector<Npp16u> expectedData(width * height, value);

  NppImageMemory<Npp16u> dst(width, height);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiSet_16u_C1R_Ctx(value, dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiSet_16u_C1R(value, dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp16u> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(Set16u, Set16uParamTest,
                         ::testing::Values(Set16uParam{32, 32, 32768, false, "32x32_v32768_noCtx"},
                                           Set16uParam{32, 32, 32768, true, "32x32_v32768_Ctx"},
                                           Set16uParam{64, 64, 0, false, "64x64_v0_noCtx"}),
                         [](const ::testing::TestParamInfo<Set16uParam> &info) { return info.param.name; });

// ==================== Set 32f C1 TEST_P ====================

struct Set32fParam {
  int width;
  int height;
  Npp32f value;
  bool use_ctx;
  std::string name;
};

class Set32fParamTest : public NppTestBase, public ::testing::WithParamInterface<Set32fParam> {};

TEST_P(Set32fParamTest, Set_32f_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp32f value = param.value;

  std::vector<Npp32f> expectedData(width * height, value);

  NppImageMemory<Npp32f> dst(width, height);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiSet_32f_C1R_Ctx(value, dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiSet_32f_C1R(value, dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32f> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
}

INSTANTIATE_TEST_SUITE_P(Set32f, Set32fParamTest,
                         ::testing::Values(Set32fParam{32, 32, 3.14159f, false, "32x32_vPi_noCtx"},
                                           Set32fParam{32, 32, 3.14159f, true, "32x32_vPi_Ctx"},
                                           Set32fParam{64, 64, 0.0f, false, "64x64_v0_noCtx"},
                                           Set32fParam{64, 64, -1.5f, false, "64x64_vNeg_noCtx"}),
                         [](const ::testing::TestParamInfo<Set32fParam> &info) { return info.param.name; });

// ==================== Set 8u C3 TEST_P ====================

struct Set8uC3Param {
  int width;
  int height;
  Npp8u values[3];
  bool use_ctx;
  std::string name;
};

class Set8uC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<Set8uC3Param> {};

TEST_P(Set8uC3ParamTest, Set_8u_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> expectedData(width * height * 3);
  for (int i = 0; i < width * height; i++) {
    expectedData[i * 3 + 0] = param.values[0];
    expectedData[i * 3 + 1] = param.values[1];
    expectedData[i * 3 + 2] = param.values[2];
  }

  NppImageMemory<Npp8u> dst(width * 3, height);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiSet_8u_C3R_Ctx(param.values, dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiSet_8u_C3R(param.values, dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(width * height * 3);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(Set8uC3, Set8uC3ParamTest,
                         ::testing::Values(Set8uC3Param{32, 32, {255, 128, 64}, false, "32x32_RGB_noCtx"},
                                           Set8uC3Param{32, 32, {255, 128, 64}, true, "32x32_RGB_Ctx"},
                                           Set8uC3Param{64, 64, {0, 0, 0}, false, "64x64_Black_noCtx"}),
                         [](const ::testing::TestParamInfo<Set8uC3Param> &info) { return info.param.name; });

// ==================== Set 8u C4 TEST_P ====================

struct Set8uC4Param {
  int width;
  int height;
  Npp8u values[4];
  bool use_ctx;
  std::string name;
};

class Set8uC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<Set8uC4Param> {};

TEST_P(Set8uC4ParamTest, Set_8u_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> expectedData(width * height * 4);
  for (int i = 0; i < width * height; i++) {
    expectedData[i * 4 + 0] = param.values[0];
    expectedData[i * 4 + 1] = param.values[1];
    expectedData[i * 4 + 2] = param.values[2];
    expectedData[i * 4 + 3] = param.values[3];
  }

  NppImageMemory<Npp8u> dst(width * 4, height);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiSet_8u_C4R_Ctx(param.values, dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiSet_8u_C4R(param.values, dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(width * height * 4);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(Set8uC4, Set8uC4ParamTest,
                         ::testing::Values(Set8uC4Param{32, 32, {255, 128, 64, 255}, false, "32x32_RGBA_noCtx"},
                                           Set8uC4Param{32, 32, {255, 128, 64, 255}, true, "32x32_RGBA_Ctx"},
                                           Set8uC4Param{64, 64, {0, 0, 0, 255}, false, "64x64_BlackA_noCtx"}),
                         [](const ::testing::TestParamInfo<Set8uC4Param> &info) { return info.param.name; });
