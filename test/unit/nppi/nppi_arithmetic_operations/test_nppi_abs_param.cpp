#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Abs 16s C1 TEST_P ====================

struct Abs16sParam {
  int width;
  int height;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Abs16sParamTest : public NppTestBase, public ::testing::WithParamInterface<Abs16sParam> {};

TEST_P(Abs16sParamTest, Abs_16s_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16s> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(-32768), static_cast<Npp16s>(32767), 12345);

  std::vector<Npp16s> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::abs_val<Npp16s>(srcData[i]);
  }

  NppImageMemory<Npp16s> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiAbs_16s_C1IR_Ctx(src.get(), src.step(), roi, ctx);
    } else {
      status = nppiAbs_16s_C1IR(src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16s> resultData(width * height);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16s> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiAbs_16s_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiAbs_16s_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16s> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Abs16s, Abs16sParamTest,
                         ::testing::Values(Abs16sParam{32, 32, false, false, "32x32_noCtx"},
                                           Abs16sParam{32, 32, true, false, "32x32_Ctx"},
                                           Abs16sParam{32, 32, false, true, "32x32_InPlace"},
                                           Abs16sParam{64, 64, false, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<Abs16sParam> &info) { return info.param.name; });

// ==================== Abs 32f C1 TEST_P ====================

struct Abs32fParam {
  int width;
  int height;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Abs32fParamTest : public NppTestBase, public ::testing::WithParamInterface<Abs32fParam> {};

TEST_P(Abs32fParamTest, Abs_32f_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, -100.0f, 100.0f, 12345);

  std::vector<Npp32f> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::abs_val<Npp32f>(srcData[i]);
  }

  NppImageMemory<Npp32f> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiAbs_32f_C1IR_Ctx(src.get(), src.step(), roi, ctx);
    } else {
      status = nppiAbs_32f_C1IR(src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> resultData(width * height);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
  } else {
    NppImageMemory<Npp32f> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiAbs_32f_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiAbs_32f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
  }
}

INSTANTIATE_TEST_SUITE_P(Abs32f, Abs32fParamTest,
                         ::testing::Values(Abs32fParam{32, 32, false, false, "32x32_noCtx"},
                                           Abs32fParam{32, 32, true, false, "32x32_Ctx"},
                                           Abs32fParam{32, 32, false, true, "32x32_InPlace"},
                                           Abs32fParam{64, 64, false, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<Abs32fParam> &info) { return info.param.name; });
