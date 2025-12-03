#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Not 8u C1 TEST_P ====================

struct Not8uParam {
  int width;
  int height;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Not8uParamTest : public NppTestBase, public ::testing::WithParamInterface<Not8uParam> {};

TEST_P(Not8uParamTest, Not_8u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  std::vector<Npp8u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::not_val<Npp8u>(srcData[i]);
  }

  NppImageMemory<Npp8u> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiNot_8u_C1IR_Ctx(src.get(), src.step(), roi, ctx);
    } else {
      status = nppiNot_8u_C1IR(src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(width * height);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp8u> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiNot_8u_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiNot_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Not8u, Not8uParamTest,
                         ::testing::Values(Not8uParam{32, 32, false, false, "32x32_noCtx"},
                                           Not8uParam{32, 32, true, false, "32x32_Ctx"},
                                           Not8uParam{32, 32, false, true, "32x32_InPlace"},
                                           Not8uParam{32, 32, true, true, "32x32_InPlace_Ctx"},
                                           Not8uParam{64, 64, false, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<Not8uParam> &info) { return info.param.name; });

// ==================== Not 8u C3 TEST_P ====================

struct Not8uC3Param {
  int width;
  int height;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Not8uC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<Not8uC3Param> {};

TEST_P(Not8uC3ParamTest, Not_8u_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 3;

  std::vector<Npp8u> srcData(width * height * channels);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  std::vector<Npp8u> expectedData(width * height * channels);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::not_val<Npp8u>(srcData[i]);
  }

  NppImageMemory<Npp8u> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiNot_8u_C3IR_Ctx(src.get(), src.step(), roi, ctx);
    } else {
      status = nppiNot_8u_C3IR(src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(width * height * channels);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp8u> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiNot_8u_C3R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiNot_8u_C3R(src.get(), src.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(width * height * channels);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Not8uC3, Not8uC3ParamTest,
                         ::testing::Values(Not8uC3Param{32, 32, false, false, "32x32_noCtx"},
                                           Not8uC3Param{32, 32, true, false, "32x32_Ctx"},
                                           Not8uC3Param{32, 32, false, true, "32x32_InPlace"},
                                           Not8uC3Param{32, 32, true, true, "32x32_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<Not8uC3Param> &info) { return info.param.name; });

// ==================== Not 8u C4 TEST_P ====================

struct Not8uC4Param {
  int width;
  int height;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Not8uC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<Not8uC4Param> {};

TEST_P(Not8uC4ParamTest, Not_8u_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;

  std::vector<Npp8u> srcData(width * height * channels);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  std::vector<Npp8u> expectedData(width * height * channels);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::not_val<Npp8u>(srcData[i]);
  }

  NppImageMemory<Npp8u> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiNot_8u_C4IR_Ctx(src.get(), src.step(), roi, ctx);
    } else {
      status = nppiNot_8u_C4IR(src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(width * height * channels);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp8u> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiNot_8u_C4R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiNot_8u_C4R(src.get(), src.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(width * height * channels);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Not8uC4, Not8uC4ParamTest,
                         ::testing::Values(Not8uC4Param{32, 32, false, false, "32x32_noCtx"},
                                           Not8uC4Param{32, 32, true, false, "32x32_Ctx"},
                                           Not8uC4Param{32, 32, false, true, "32x32_InPlace"},
                                           Not8uC4Param{32, 32, true, true, "32x32_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<Not8uC4Param> &info) { return info.param.name; });
