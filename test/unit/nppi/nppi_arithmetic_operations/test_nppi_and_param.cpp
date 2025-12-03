#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== And 8u C1 TEST_P ====================

struct And8uParam {
  int width;
  int height;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class And8uParamTest : public NppTestBase, public ::testing::WithParamInterface<And8uParam> {};

TEST_P(And8uParamTest, And_8u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> src1Data(width * height);
  std::vector<Npp8u> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 54321);

  std::vector<Npp8u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::and_val<Npp8u>(src1Data[i], src2Data[i]);
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
      status = nppiAnd_8u_C1IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx);
    } else {
      status = nppiAnd_8u_C1IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
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
      status = nppiAnd_8u_C1R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiAnd_8u_C1R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(And8u, And8uParamTest,
                         ::testing::Values(And8uParam{32, 32, false, false, "32x32_noCtx"},
                                           And8uParam{32, 32, true, false, "32x32_Ctx"},
                                           And8uParam{32, 32, false, true, "32x32_InPlace"},
                                           And8uParam{32, 32, true, true, "32x32_InPlace_Ctx"},
                                           And8uParam{64, 64, false, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<And8uParam> &info) { return info.param.name; });

// ==================== AndC 8u C1 TEST_P ====================

struct AndC8uParam {
  int width;
  int height;
  Npp8u constant;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class AndC8uParamTest : public NppTestBase, public ::testing::WithParamInterface<AndC8uParam> {};

TEST_P(AndC8uParamTest, AndC_8u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp8u constant = param.constant;

  std::vector<Npp8u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  std::vector<Npp8u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::and_c<Npp8u>(srcData[i], constant);
  }

  NppImageMemory<Npp8u> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiAndC_8u_C1IR_Ctx(constant, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiAndC_8u_C1IR(constant, src.get(), src.step(), roi);
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
      status = nppiAndC_8u_C1R_Ctx(src.get(), src.step(), constant, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiAndC_8u_C1R(src.get(), src.step(), constant, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(AndC8u, AndC8uParamTest,
                         ::testing::Values(AndC8uParam{32, 32, 0xF0, false, false, "32x32_cF0_noCtx"},
                                           AndC8uParam{32, 32, 0xF0, true, false, "32x32_cF0_Ctx"},
                                           AndC8uParam{32, 32, 0x0F, false, true, "32x32_c0F_InPlace"},
                                           AndC8uParam{32, 32, 0x0F, true, true, "32x32_c0F_InPlace_Ctx"},
                                           AndC8uParam{64, 64, 0xAA, false, false, "64x64_cAA_noCtx"}),
                         [](const ::testing::TestParamInfo<AndC8uParam> &info) { return info.param.name; });

// ==================== And 8u C3 TEST_P ====================

class And8uC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<And8uParam> {};

TEST_P(And8uC3ParamTest, And_8u_C3R) {
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
    expectedData[i] = expect::and_val<Npp8u>(src1Data[i], src2Data[i]);
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
      status = nppiAnd_8u_C3IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx);
    } else {
      status = nppiAnd_8u_C3IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
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
      status = nppiAnd_8u_C3R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiAnd_8u_C3R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(And8uC3, And8uC3ParamTest,
                         ::testing::Values(And8uParam{32, 32, false, false, "32x32_noCtx"},
                                           And8uParam{32, 32, true, false, "32x32_Ctx"},
                                           And8uParam{32, 32, false, true, "32x32_InPlace"},
                                           And8uParam{32, 32, true, true, "32x32_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<And8uParam> &info) { return info.param.name; });

// ==================== And 8u C4 TEST_P ====================

class And8uC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<And8uParam> {};

TEST_P(And8uC4ParamTest, And_8u_C4R) {
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
    expectedData[i] = expect::and_val<Npp8u>(src1Data[i], src2Data[i]);
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
      status = nppiAnd_8u_C4IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx);
    } else {
      status = nppiAnd_8u_C4IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
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
      status = nppiAnd_8u_C4R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiAnd_8u_C4R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(And8uC4, And8uC4ParamTest,
                         ::testing::Values(And8uParam{32, 32, false, false, "32x32_noCtx"},
                                           And8uParam{32, 32, true, false, "32x32_Ctx"},
                                           And8uParam{32, 32, false, true, "32x32_InPlace"},
                                           And8uParam{32, 32, true, true, "32x32_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<And8uParam> &info) { return info.param.name; });

// ==================== AndC 8u C3 TEST_P ====================

class AndC8uC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<AndC8uParam> {};

TEST_P(AndC8uC3ParamTest, AndC_8u_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp8u constant = param.constant;
  const int channels = 3;
  const int total = width * height * channels;

  const Npp8u aConstants[3] = {constant, constant, constant};

  std::vector<Npp8u> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  std::vector<Npp8u> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::and_c<Npp8u>(srcData[i], aConstants[i % 3]);
  }

  NppImageMemory<Npp8u> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiAndC_8u_C3IR_Ctx(aConstants, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiAndC_8u_C3IR(aConstants, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp8u> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiAndC_8u_C3R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiAndC_8u_C3R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(AndC8uC3, AndC8uC3ParamTest,
                         ::testing::Values(AndC8uParam{32, 32, 0xF0, false, false, "32x32_cF0_noCtx"},
                                           AndC8uParam{32, 32, 0xF0, true, false, "32x32_cF0_Ctx"},
                                           AndC8uParam{32, 32, 0x0F, false, true, "32x32_c0F_InPlace"},
                                           AndC8uParam{32, 32, 0x0F, true, true, "32x32_c0F_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<AndC8uParam> &info) { return info.param.name; });

// ==================== AndC 8u C4 TEST_P ====================

class AndC8uC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<AndC8uParam> {};

TEST_P(AndC8uC4ParamTest, AndC_8u_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp8u constant = param.constant;
  const int channels = 4;
  const int total = width * height * channels;

  const Npp8u aConstants[4] = {constant, constant, constant, constant};

  std::vector<Npp8u> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  std::vector<Npp8u> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::and_c<Npp8u>(srcData[i], aConstants[i % 4]);
  }

  NppImageMemory<Npp8u> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiAndC_8u_C4IR_Ctx(aConstants, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiAndC_8u_C4IR(aConstants, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp8u> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiAndC_8u_C4R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiAndC_8u_C4R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(AndC8uC4, AndC8uC4ParamTest,
                         ::testing::Values(AndC8uParam{32, 32, 0xF0, false, false, "32x32_cF0_noCtx"},
                                           AndC8uParam{32, 32, 0xF0, true, false, "32x32_cF0_Ctx"},
                                           AndC8uParam{32, 32, 0x0F, false, true, "32x32_c0F_InPlace"},
                                           AndC8uParam{32, 32, 0x0F, true, true, "32x32_c0F_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<AndC8uParam> &info) { return info.param.name; });

// ==================== And 16u C1 TEST_P ====================

struct And16uParam {
  int width;
  int height;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class And16uC1ParamTest : public NppTestBase, public ::testing::WithParamInterface<And16uParam> {};

TEST_P(And16uC1ParamTest, And_16u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> src1Data(width * height);
  std::vector<Npp16u> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 54321);

  std::vector<Npp16u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::and_val<Npp16u>(src1Data[i], src2Data[i]);
  }

  NppImageMemory<Npp16u> src1(width, height);
  NppImageMemory<Npp16u> src2(width, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiAnd_16u_C1IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx);
    } else {
      status = nppiAnd_16u_C1IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(width * height);
    src2.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16u> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiAnd_16u_C1R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiAnd_16u_C1R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(And16uC1, And16uC1ParamTest,
                         ::testing::Values(And16uParam{32, 32, false, false, "32x32_noCtx"},
                                           And16uParam{32, 32, true, false, "32x32_Ctx"},
                                           And16uParam{32, 32, false, true, "32x32_InPlace"},
                                           And16uParam{32, 32, true, true, "32x32_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<And16uParam> &info) { return info.param.name; });

// ==================== And 16u C3 TEST_P ====================

class And16uC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<And16uParam> {};

TEST_P(And16uC3ParamTest, And_16u_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 3;
  const int total = width * height * channels;

  std::vector<Npp16u> src1Data(total);
  std::vector<Npp16u> src2Data(total);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 54321);

  std::vector<Npp16u> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::and_val<Npp16u>(src1Data[i], src2Data[i]);
  }

  NppImageMemory<Npp16u> src1(width * channels, height);
  NppImageMemory<Npp16u> src2(width * channels, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiAnd_16u_C3IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx);
    } else {
      status = nppiAnd_16u_C3IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    src2.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16u> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiAnd_16u_C3R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiAnd_16u_C3R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(And16uC3, And16uC3ParamTest,
                         ::testing::Values(And16uParam{32, 32, false, false, "32x32_noCtx"},
                                           And16uParam{32, 32, true, false, "32x32_Ctx"},
                                           And16uParam{32, 32, false, true, "32x32_InPlace"},
                                           And16uParam{32, 32, true, true, "32x32_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<And16uParam> &info) { return info.param.name; });
