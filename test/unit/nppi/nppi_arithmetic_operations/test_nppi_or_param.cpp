#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Or 8u C1 TEST_P ====================

struct Or8uParam {
  int width;
  int height;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Or8uParamTest : public NppTestBase, public ::testing::WithParamInterface<Or8uParam> {};

TEST_P(Or8uParamTest, Or_8u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> src1Data(width * height);
  std::vector<Npp8u> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 54321);

  std::vector<Npp8u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::or_val<Npp8u>(src1Data[i], src2Data[i]);
  }

  NppImageMemory<Npp8u> src1(width, height);
  NppImageMemory<Npp8u> src2(width, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiOr_8u_C1IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx);
    } else {
      status = nppiOr_8u_C1IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(width * height);
    src2.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp8u> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiOr_8u_C1R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiOr_8u_C1R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Or8u, Or8uParamTest,
                         ::testing::Values(Or8uParam{32, 32, false, false, "32x32_noCtx"},
                                           Or8uParam{32, 32, true, false, "32x32_Ctx"},
                                           Or8uParam{32, 32, false, true, "32x32_InPlace"},
                                           Or8uParam{32, 32, true, true, "32x32_InPlace_Ctx"},
                                           Or8uParam{64, 64, false, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<Or8uParam> &info) { return info.param.name; });

// ==================== OrC 8u C1 TEST_P ====================

struct OrC8uParam {
  int width;
  int height;
  Npp8u constant;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class OrC8uParamTest : public NppTestBase, public ::testing::WithParamInterface<OrC8uParam> {};

TEST_P(OrC8uParamTest, OrC_8u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp8u constant = param.constant;

  std::vector<Npp8u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  std::vector<Npp8u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::or_c<Npp8u>(srcData[i], constant);
  }

  NppImageMemory<Npp8u> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiOrC_8u_C1IR_Ctx(constant, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiOrC_8u_C1IR(constant, src.get(), src.step(), roi);
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
      status = nppiOrC_8u_C1R_Ctx(src.get(), src.step(), constant, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiOrC_8u_C1R(src.get(), src.step(), constant, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(OrC8u, OrC8uParamTest,
                         ::testing::Values(OrC8uParam{32, 32, 0xF0, false, false, "32x32_cF0_noCtx"},
                                           OrC8uParam{32, 32, 0xF0, true, false, "32x32_cF0_Ctx"},
                                           OrC8uParam{32, 32, 0x0F, false, true, "32x32_c0F_InPlace"},
                                           OrC8uParam{32, 32, 0x0F, true, true, "32x32_c0F_InPlace_Ctx"},
                                           OrC8uParam{64, 64, 0xAA, false, false, "64x64_cAA_noCtx"}),
                         [](const ::testing::TestParamInfo<OrC8uParam> &info) { return info.param.name; });

// ==================== Or 8u C3 TEST_P ====================

class Or8uC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<Or8uParam> {};

TEST_P(Or8uC3ParamTest, Or_8u_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 3;
  const int total = width * height * channels;

  std::vector<Npp8u> src1Data(total);
  std::vector<Npp8u> src2Data(total);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 54321);

  std::vector<Npp8u> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::or_val<Npp8u>(src1Data[i], src2Data[i]);
  }

  NppImageMemory<Npp8u> src1(width * channels, height);
  NppImageMemory<Npp8u> src2(width * channels, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiOr_8u_C3IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx);
    } else {
      status = nppiOr_8u_C3IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    src2.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp8u> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiOr_8u_C3R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiOr_8u_C3R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Or8uC3, Or8uC3ParamTest,
                         ::testing::Values(Or8uParam{32, 32, false, false, "32x32_noCtx"},
                                           Or8uParam{32, 32, true, false, "32x32_Ctx"},
                                           Or8uParam{32, 32, false, true, "32x32_InPlace"},
                                           Or8uParam{32, 32, true, true, "32x32_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<Or8uParam> &info) { return info.param.name; });

// ==================== Or 8u C4 TEST_P ====================

class Or8uC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<Or8uParam> {};

TEST_P(Or8uC4ParamTest, Or_8u_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  std::vector<Npp8u> src1Data(total);
  std::vector<Npp8u> src2Data(total);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 54321);

  std::vector<Npp8u> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::or_val<Npp8u>(src1Data[i], src2Data[i]);
  }

  NppImageMemory<Npp8u> src1(width * channels, height);
  NppImageMemory<Npp8u> src2(width * channels, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiOr_8u_C4IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx);
    } else {
      status = nppiOr_8u_C4IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    src2.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp8u> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiOr_8u_C4R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiOr_8u_C4R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Or8uC4, Or8uC4ParamTest,
                         ::testing::Values(Or8uParam{32, 32, false, false, "32x32_noCtx"},
                                           Or8uParam{32, 32, true, false, "32x32_Ctx"},
                                           Or8uParam{32, 32, false, true, "32x32_InPlace"},
                                           Or8uParam{32, 32, true, true, "32x32_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<Or8uParam> &info) { return info.param.name; });
