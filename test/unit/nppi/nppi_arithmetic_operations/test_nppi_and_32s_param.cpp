#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== And 32s TEST_P ====================

struct And32sParam {
  int width;
  int height;
  bool use_ctx;
  bool in_place;
  std::string name;
};

// ==================== And 32s C1 TEST_P ====================

class And32sC1ParamTest : public NppTestBase, public ::testing::WithParamInterface<And32sParam> {};

TEST_P(And32sC1ParamTest, And_32s_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32s> src1Data(width * height);
  std::vector<Npp32s> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp32s>(-1000000), static_cast<Npp32s>(1000000), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp32s>(-1000000), static_cast<Npp32s>(1000000), 54321);

  std::vector<Npp32s> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = src1Data[i] & src2Data[i];
  }

  NppImageMemory<Npp32s> src1(width, height);
  NppImageMemory<Npp32s> src2(width, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      nppGetStreamContext(&ctx);
      status = nppiAnd_32s_C1IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx);
    } else {
      status = nppiAnd_32s_C1IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32s> resultData(width * height);
    src2.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp32s> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      nppGetStreamContext(&ctx);
      status = nppiAnd_32s_C1R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiAnd_32s_C1R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32s> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(And32sC1, And32sC1ParamTest,
                         ::testing::Values(And32sParam{32, 32, false, false, "32x32_noCtx"},
                                           And32sParam{32, 32, true, false, "32x32_Ctx"},
                                           And32sParam{32, 32, false, true, "32x32_InPlace"},
                                           And32sParam{32, 32, true, true, "32x32_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<And32sParam> &info) { return info.param.name; });

// ==================== And 32s C3 TEST_P ====================

class And32sC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<And32sParam> {};

TEST_P(And32sC3ParamTest, And_32s_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 3;
  const int total = width * height * channels;

  std::vector<Npp32s> src1Data(total);
  std::vector<Npp32s> src2Data(total);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp32s>(-1000000), static_cast<Npp32s>(1000000), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp32s>(-1000000), static_cast<Npp32s>(1000000), 54321);

  std::vector<Npp32s> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = src1Data[i] & src2Data[i];
  }

  NppImageMemory<Npp32s> src1(width * channels, height);
  NppImageMemory<Npp32s> src2(width * channels, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      nppGetStreamContext(&ctx);
      status = nppiAnd_32s_C3IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx);
    } else {
      status = nppiAnd_32s_C3IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32s> resultData(total);
    src2.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp32s> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      nppGetStreamContext(&ctx);
      status = nppiAnd_32s_C3R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiAnd_32s_C3R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32s> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(And32sC3, And32sC3ParamTest,
                         ::testing::Values(And32sParam{32, 32, false, false, "32x32_noCtx"},
                                           And32sParam{32, 32, true, false, "32x32_Ctx"},
                                           And32sParam{32, 32, false, true, "32x32_InPlace"},
                                           And32sParam{32, 32, true, true, "32x32_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<And32sParam> &info) { return info.param.name; });

// ==================== And 32s C4 TEST_P ====================

class And32sC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<And32sParam> {};

TEST_P(And32sC4ParamTest, And_32s_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  std::vector<Npp32s> src1Data(total);
  std::vector<Npp32s> src2Data(total);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp32s>(-1000000), static_cast<Npp32s>(1000000), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp32s>(-1000000), static_cast<Npp32s>(1000000), 54321);

  std::vector<Npp32s> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = src1Data[i] & src2Data[i];
  }

  NppImageMemory<Npp32s> src1(width * channels, height);
  NppImageMemory<Npp32s> src2(width * channels, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      nppGetStreamContext(&ctx);
      status = nppiAnd_32s_C4IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx);
    } else {
      status = nppiAnd_32s_C4IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32s> resultData(total);
    src2.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp32s> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      nppGetStreamContext(&ctx);
      status = nppiAnd_32s_C4R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiAnd_32s_C4R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32s> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(And32sC4, And32sC4ParamTest,
                         ::testing::Values(And32sParam{32, 32, false, false, "32x32_noCtx"},
                                           And32sParam{32, 32, true, false, "32x32_Ctx"},
                                           And32sParam{32, 32, false, true, "32x32_InPlace"},
                                           And32sParam{32, 32, true, true, "32x32_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<And32sParam> &info) { return info.param.name; });

// ==================== And 32s AC4 TEST_P ====================

class And32sAC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<And32sParam> {};

TEST_P(And32sAC4ParamTest, And_32s_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  std::vector<Npp32s> src1Data(total);
  std::vector<Npp32s> src2Data(total);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp32s>(-1000000), static_cast<Npp32s>(1000000), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp32s>(-1000000), static_cast<Npp32s>(1000000), 54321);

  std::vector<Npp32s> expectedData(total);
  for (int i = 0; i < total; i++) {
    if (i % 4 == 3) {
      expectedData[i] = src2Data[i];  // Alpha unchanged
    } else {
      expectedData[i] = src1Data[i] & src2Data[i];
    }
  }

  NppImageMemory<Npp32s> src1(width * channels, height);
  NppImageMemory<Npp32s> src2(width * channels, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      nppGetStreamContext(&ctx);
      status = nppiAnd_32s_AC4IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx);
    } else {
      status = nppiAnd_32s_AC4IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32s> resultData(total);
    src2.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp32s> dst(width * channels, height);
    // Pre-copy src2 to dst to preserve alpha channel
    cudaMemcpy2D(dst.get(), dst.step(), src2.get(), src2.step(),
                 width * channels * sizeof(Npp32s), height, cudaMemcpyDeviceToDevice);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      nppGetStreamContext(&ctx);
      status = nppiAnd_32s_AC4R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiAnd_32s_AC4R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32s> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(And32sAC4, And32sAC4ParamTest,
                         ::testing::Values(And32sParam{32, 32, false, false, "32x32_noCtx"},
                                           And32sParam{32, 32, true, false, "32x32_Ctx"},
                                           And32sParam{32, 32, false, true, "32x32_InPlace"},
                                           And32sParam{32, 32, true, true, "32x32_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<And32sParam> &info) { return info.param.name; });

// ==================== AndC 32s C1 TEST_P ====================

struct AndC32sParam {
  int width;
  int height;
  Npp32s constant;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class AndC32sC1ParamTest : public NppTestBase, public ::testing::WithParamInterface<AndC32sParam> {};

TEST_P(AndC32sC1ParamTest, AndC_32s_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp32s constant = param.constant;

  std::vector<Npp32s> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp32s>(-1000000), static_cast<Npp32s>(1000000), 12345);

  std::vector<Npp32s> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = srcData[i] & constant;
  }

  NppImageMemory<Npp32s> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      nppGetStreamContext(&ctx);
      status = nppiAndC_32s_C1IR_Ctx(constant, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiAndC_32s_C1IR(constant, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32s> resultData(width * height);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp32s> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      nppGetStreamContext(&ctx);
      status = nppiAndC_32s_C1R_Ctx(src.get(), src.step(), constant, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiAndC_32s_C1R(src.get(), src.step(), constant, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32s> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(AndC32sC1, AndC32sC1ParamTest,
                         ::testing::Values(AndC32sParam{32, 32, static_cast<Npp32s>(0xFFFF0000), false, false, "32x32_cFFFF0000_noCtx"},
                                           AndC32sParam{32, 32, static_cast<Npp32s>(0xFFFF0000), true, false, "32x32_cFFFF0000_Ctx"},
                                           AndC32sParam{32, 32, static_cast<Npp32s>(0x0000FFFF), false, true, "32x32_c0000FFFF_InPlace"},
                                           AndC32sParam{32, 32, static_cast<Npp32s>(0x0000FFFF), true, true, "32x32_c0000FFFF_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<AndC32sParam> &info) { return info.param.name; });

// ==================== AndC 32s C3 TEST_P ====================

class AndC32sC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<AndC32sParam> {};

TEST_P(AndC32sC3ParamTest, AndC_32s_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp32s constant = param.constant;
  const int channels = 3;
  const int total = width * height * channels;

  const Npp32s aConstants[3] = {constant, constant, constant};

  std::vector<Npp32s> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp32s>(-1000000), static_cast<Npp32s>(1000000), 12345);

  std::vector<Npp32s> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = srcData[i] & aConstants[i % 3];
  }

  NppImageMemory<Npp32s> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      nppGetStreamContext(&ctx);
      status = nppiAndC_32s_C3IR_Ctx(aConstants, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiAndC_32s_C3IR(aConstants, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32s> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp32s> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      nppGetStreamContext(&ctx);
      status = nppiAndC_32s_C3R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiAndC_32s_C3R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32s> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(AndC32sC3, AndC32sC3ParamTest,
                         ::testing::Values(AndC32sParam{32, 32, static_cast<Npp32s>(0xFFFF0000), false, false, "32x32_cFFFF0000_noCtx"},
                                           AndC32sParam{32, 32, static_cast<Npp32s>(0xFFFF0000), true, false, "32x32_cFFFF0000_Ctx"},
                                           AndC32sParam{32, 32, static_cast<Npp32s>(0x0000FFFF), false, true, "32x32_c0000FFFF_InPlace"},
                                           AndC32sParam{32, 32, static_cast<Npp32s>(0x0000FFFF), true, true, "32x32_c0000FFFF_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<AndC32sParam> &info) { return info.param.name; });

// ==================== AndC 32s C4 TEST_P ====================

class AndC32sC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<AndC32sParam> {};

TEST_P(AndC32sC4ParamTest, AndC_32s_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp32s constant = param.constant;
  const int channels = 4;
  const int total = width * height * channels;

  const Npp32s aConstants[4] = {constant, constant, constant, constant};

  std::vector<Npp32s> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp32s>(-1000000), static_cast<Npp32s>(1000000), 12345);

  std::vector<Npp32s> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = srcData[i] & aConstants[i % 4];
  }

  NppImageMemory<Npp32s> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      nppGetStreamContext(&ctx);
      status = nppiAndC_32s_C4IR_Ctx(aConstants, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiAndC_32s_C4IR(aConstants, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32s> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp32s> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      nppGetStreamContext(&ctx);
      status = nppiAndC_32s_C4R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiAndC_32s_C4R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32s> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(AndC32sC4, AndC32sC4ParamTest,
                         ::testing::Values(AndC32sParam{32, 32, static_cast<Npp32s>(0xFFFF0000), false, false, "32x32_cFFFF0000_noCtx"},
                                           AndC32sParam{32, 32, static_cast<Npp32s>(0xFFFF0000), true, false, "32x32_cFFFF0000_Ctx"},
                                           AndC32sParam{32, 32, static_cast<Npp32s>(0x0000FFFF), false, true, "32x32_c0000FFFF_InPlace"},
                                           AndC32sParam{32, 32, static_cast<Npp32s>(0x0000FFFF), true, true, "32x32_c0000FFFF_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<AndC32sParam> &info) { return info.param.name; });

// ==================== AndC 32s AC4 TEST_P ====================

class AndC32sAC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<AndC32sParam> {};

TEST_P(AndC32sAC4ParamTest, AndC_32s_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp32s constant = param.constant;
  const int channels = 4;
  const int total = width * height * channels;

  const Npp32s aConstants[3] = {constant, constant, constant};

  std::vector<Npp32s> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp32s>(-1000000), static_cast<Npp32s>(1000000), 12345);

  std::vector<Npp32s> expectedData(total);
  for (int i = 0; i < total; i++) {
    int ch = i % 4;
    if (ch == 3) {
      expectedData[i] = srcData[i];  // Alpha unchanged
    } else {
      expectedData[i] = srcData[i] & aConstants[ch];
    }
  }

  NppImageMemory<Npp32s> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      nppGetStreamContext(&ctx);
      status = nppiAndC_32s_AC4IR_Ctx(aConstants, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiAndC_32s_AC4IR(aConstants, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32s> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp32s> dst(width * channels, height);
    // Pre-copy src to dst to preserve alpha channel
    cudaMemcpy2D(dst.get(), dst.step(), src.get(), src.step(),
                 width * channels * sizeof(Npp32s), height, cudaMemcpyDeviceToDevice);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      nppGetStreamContext(&ctx);
      status = nppiAndC_32s_AC4R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiAndC_32s_AC4R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32s> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(AndC32sAC4, AndC32sAC4ParamTest,
                         ::testing::Values(AndC32sParam{32, 32, static_cast<Npp32s>(0xFFFF0000), false, false, "32x32_cFFFF0000_noCtx"},
                                           AndC32sParam{32, 32, static_cast<Npp32s>(0xFFFF0000), true, false, "32x32_cFFFF0000_Ctx"},
                                           AndC32sParam{32, 32, static_cast<Npp32s>(0x0000FFFF), false, true, "32x32_c0000FFFF_InPlace"},
                                           AndC32sParam{32, 32, static_cast<Npp32s>(0x0000FFFF), true, true, "32x32_c0000FFFF_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<AndC32sParam> &info) { return info.param.name; });
