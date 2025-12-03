#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== LShiftC 8u C1 TEST_P ====================

struct LShiftC8uParam {
  int width;
  int height;
  Npp32u shift;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class LShiftC8uParamTest : public NppTestBase, public ::testing::WithParamInterface<LShiftC8uParam> {};

TEST_P(LShiftC8uParamTest, LShiftC_8u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp32u shift = param.shift;

  std::vector<Npp8u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  std::vector<Npp8u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::lshift_c<Npp8u>(srcData[i], shift);
  }

  NppImageMemory<Npp8u> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiLShiftC_8u_C1IR_Ctx(shift, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiLShiftC_8u_C1IR(shift, src.get(), src.step(), roi);
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
      status = nppiLShiftC_8u_C1R_Ctx(src.get(), src.step(), shift, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiLShiftC_8u_C1R(src.get(), src.step(), shift, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(LShiftC8u, LShiftC8uParamTest,
                         ::testing::Values(LShiftC8uParam{32, 32, 2, false, false, "32x32_s2_noCtx"},
                                           LShiftC8uParam{32, 32, 2, true, false, "32x32_s2_Ctx"},
                                           LShiftC8uParam{32, 32, 4, false, true, "32x32_s4_InPlace"},
                                           LShiftC8uParam{32, 32, 4, true, true, "32x32_s4_InPlace_Ctx"},
                                           LShiftC8uParam{64, 64, 3, false, false, "64x64_s3_noCtx"}),
                         [](const ::testing::TestParamInfo<LShiftC8uParam> &info) { return info.param.name; });

// ==================== LShiftC 16u C1 TEST_P ====================

struct LShiftC16uParam {
  int width;
  int height;
  Npp32u shift;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class LShiftC16uParamTest : public NppTestBase, public ::testing::WithParamInterface<LShiftC16uParam> {};

TEST_P(LShiftC16uParamTest, LShiftC_16u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp32u shift = param.shift;

  std::vector<Npp16u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);

  std::vector<Npp16u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::lshift_c<Npp16u>(srcData[i], shift);
  }

  NppImageMemory<Npp16u> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiLShiftC_16u_C1IR_Ctx(shift, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiLShiftC_16u_C1IR(shift, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(width * height);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16u> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiLShiftC_16u_C1R_Ctx(src.get(), src.step(), shift, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiLShiftC_16u_C1R(src.get(), src.step(), shift, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(LShiftC16u, LShiftC16uParamTest,
                         ::testing::Values(LShiftC16uParam{32, 32, 4, false, false, "32x32_s4_noCtx"},
                                           LShiftC16uParam{32, 32, 4, true, false, "32x32_s4_Ctx"},
                                           LShiftC16uParam{32, 32, 8, false, true, "32x32_s8_InPlace"},
                                           LShiftC16uParam{64, 64, 6, false, false, "64x64_s6_noCtx"}),
                         [](const ::testing::TestParamInfo<LShiftC16uParam> &info) { return info.param.name; });

