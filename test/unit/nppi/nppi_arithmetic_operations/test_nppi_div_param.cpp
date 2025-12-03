#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Div 32f TEST_P ====================

struct Div32fParam {
  int width;
  int height;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Div32fParamTest : public NppTestBase, public ::testing::WithParamInterface<Div32fParam> {};

TEST_P(Div32fParamTest, Div_32f_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> src1Data(width * height);
  std::vector<Npp32f> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, 1.0f, 100.0f, 12345);
  TestDataGenerator::generateRandom(src2Data, 1.0f, 10.0f, 54321);

  std::vector<Npp32f> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    // NPP Div: dst = src2 / src1
    expectedData[i] = expect::div<Npp32f>(src2Data[i], src1Data[i]);
  }

  NppImageMemory<Npp32f> src1(width, height);
  NppImageMemory<Npp32f> src2(width, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiDiv_32f_C1IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx);
    } else {
      status = nppiDiv_32f_C1IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> resultData(width * height);
    src2.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-4f));
  } else {
    NppImageMemory<Npp32f> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiDiv_32f_C1R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiDiv_32f_C1R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-4f));
  }
}

INSTANTIATE_TEST_SUITE_P(Div32f, Div32fParamTest,
                         ::testing::Values(Div32fParam{32, 32, false, false, "32x32_noCtx"},
                                           Div32fParam{32, 32, true, false, "32x32_Ctx"},
                                           Div32fParam{32, 32, false, true, "32x32_InPlace"},
                                           Div32fParam{32, 32, true, true, "32x32_InPlace_Ctx"},
                                           Div32fParam{64, 64, false, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<Div32fParam> &info) { return info.param.name; });

// ==================== Div 8u with scale factor TEST_P ====================

struct Div8uSfsParam {
  int width;
  int height;
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Div8uSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<Div8uSfsParam> {};

TEST_P(Div8uSfsParamTest, Div_8u_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int scaleFactor = param.scaleFactor;

  std::vector<Npp8u> src1Data(width * height);
  std::vector<Npp8u> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(10), static_cast<Npp8u>(200), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8u>(1), static_cast<Npp8u>(10), 54321);

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
      status = nppiDiv_8u_C1IRSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiDiv_8u_C1IRSfs(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(width * height);
    src2.copyToHost(resultData);
    // Tolerance-based verification for integer division
    // NPP Div: dst = src2 / src1
    for (size_t i = 0; i < resultData.size(); i++) {
      int expected = src2Data[i] / src1Data[i];
      EXPECT_LE(std::abs(static_cast<int>(resultData[i]) - expected), 1);
    }
  } else {
    NppImageMemory<Npp8u> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiDiv_8u_C1RSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                     scaleFactor, ctx);
    } else {
      status = nppiDiv_8u_C1RSfs(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(width * height);
    dst.copyToHost(resultData);
    // Tolerance-based verification for integer division
    // NPP Div: dst = src2 / src1
    for (size_t i = 0; i < resultData.size(); i++) {
      int expected = src2Data[i] / src1Data[i];
      EXPECT_LE(std::abs(static_cast<int>(resultData[i]) - expected), 1);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Div8uSfs, Div8uSfsParamTest,
                         ::testing::Values(Div8uSfsParam{32, 32, 0, false, false, "32x32_sfs0_noCtx"},
                                           Div8uSfsParam{32, 32, 0, true, false, "32x32_sfs0_Ctx"},
                                           Div8uSfsParam{32, 32, 0, false, true, "32x32_sfs0_InPlace"},
                                           Div8uSfsParam{32, 32, 0, true, true, "32x32_sfs0_InPlace_Ctx"},
                                           Div8uSfsParam{64, 64, 0, false, false, "64x64_sfs0_noCtx"}),
                         [](const ::testing::TestParamInfo<Div8uSfsParam> &info) { return info.param.name; });

// ==================== Div 16u with scale factor TEST_P ====================

struct Div16uSfsParam {
  int width;
  int height;
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Div16uSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<Div16uSfsParam> {};

TEST_P(Div16uSfsParamTest, Div_16u_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int scaleFactor = param.scaleFactor;

  std::vector<Npp16u> src1Data(width * height);
  std::vector<Npp16u> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(100), static_cast<Npp16u>(10000), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(1), static_cast<Npp16u>(100), 54321);

  NppImageMemory<Npp16u> src1(width, height);
  NppImageMemory<Npp16u> src2(width, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiDiv_16u_C1IRSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiDiv_16u_C1IRSfs(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(width * height);
    src2.copyToHost(resultData);
    // Tolerance-based verification
    // NPP Div: dst = src2 / src1
    for (size_t i = 0; i < resultData.size(); i++) {
      int expected = src2Data[i] / src1Data[i];
      EXPECT_LE(std::abs(static_cast<int>(resultData[i]) - expected), 1);
    }
  } else {
    NppImageMemory<Npp16u> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiDiv_16u_C1RSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                      scaleFactor, ctx);
    } else {
      status = nppiDiv_16u_C1RSfs(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(width * height);
    dst.copyToHost(resultData);
    // Tolerance-based verification
    // NPP Div: dst = src2 / src1
    for (size_t i = 0; i < resultData.size(); i++) {
      int expected = src2Data[i] / src1Data[i];
      EXPECT_LE(std::abs(static_cast<int>(resultData[i]) - expected), 1);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Div16uSfs, Div16uSfsParamTest,
                         ::testing::Values(Div16uSfsParam{32, 32, 0, false, false, "32x32_sfs0_noCtx"},
                                           Div16uSfsParam{32, 32, 0, true, false, "32x32_sfs0_Ctx"},
                                           Div16uSfsParam{32, 32, 0, false, true, "32x32_sfs0_InPlace"},
                                           Div16uSfsParam{32, 32, 0, true, true, "32x32_sfs0_InPlace_Ctx"},
                                           Div16uSfsParam{64, 64, 0, false, false, "64x64_sfs0_noCtx"}),
                         [](const ::testing::TestParamInfo<Div16uSfsParam> &info) { return info.param.name; });

// ==================== Div 16s with scale factor TEST_P ====================

struct Div16sSfsParam {
  int width;
  int height;
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Div16sSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<Div16sSfsParam> {};

TEST_P(Div16sSfsParamTest, Div_16s_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int scaleFactor = param.scaleFactor;

  std::vector<Npp16s> src1Data(width * height);
  std::vector<Npp16s> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16s>(10), static_cast<Npp16s>(1000), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16s>(1), static_cast<Npp16s>(10), 54321);

  NppImageMemory<Npp16s> src1(width, height);
  NppImageMemory<Npp16s> src2(width, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiDiv_16s_C1IRSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiDiv_16s_C1IRSfs(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16s> resultData(width * height);
    src2.copyToHost(resultData);
    // Tolerance-based verification
    // NPP Div: dst = src2 / src1
    for (size_t i = 0; i < resultData.size(); i++) {
      int expected = src2Data[i] / src1Data[i];
      EXPECT_LE(std::abs(static_cast<int>(resultData[i]) - expected), 1);
    }
  } else {
    NppImageMemory<Npp16s> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiDiv_16s_C1RSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                      scaleFactor, ctx);
    } else {
      status = nppiDiv_16s_C1RSfs(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16s> resultData(width * height);
    dst.copyToHost(resultData);
    // Tolerance-based verification
    // NPP Div: dst = src2 / src1
    for (size_t i = 0; i < resultData.size(); i++) {
      int expected = src2Data[i] / src1Data[i];
      EXPECT_LE(std::abs(static_cast<int>(resultData[i]) - expected), 1);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Div16sSfs, Div16sSfsParamTest,
                         ::testing::Values(Div16sSfsParam{32, 32, 0, false, false, "32x32_sfs0_noCtx"},
                                           Div16sSfsParam{32, 32, 0, true, false, "32x32_sfs0_Ctx"},
                                           Div16sSfsParam{32, 32, 0, false, true, "32x32_sfs0_InPlace"},
                                           Div16sSfsParam{32, 32, 0, true, true, "32x32_sfs0_InPlace_Ctx"},
                                           Div16sSfsParam{64, 64, 0, false, false, "64x64_sfs0_noCtx"}),
                         [](const ::testing::TestParamInfo<Div16sSfsParam> &info) { return info.param.name; });

// ==================== Div 32f C3 TEST_P ====================

struct Div32fC3Param {
  int width;
  int height;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Div32fC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<Div32fC3Param> {};

TEST_P(Div32fC3ParamTest, Div_32f_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 3;

  std::vector<Npp32f> src1Data(width * height * channels);
  std::vector<Npp32f> src2Data(width * height * channels);
  TestDataGenerator::generateRandom(src1Data, 1.0f, 100.0f, 12345);
  TestDataGenerator::generateRandom(src2Data, 1.0f, 10.0f, 54321);

  std::vector<Npp32f> expectedData(width * height * channels);
  for (size_t i = 0; i < expectedData.size(); i++) {
    // NPP Div: dst = src2 / src1
    expectedData[i] = expect::div<Npp32f>(src2Data[i], src1Data[i]);
  }

  NppImageMemory<Npp32f> src1(width * channels, height);
  NppImageMemory<Npp32f> src2(width * channels, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiDiv_32f_C3IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx);
    } else {
      status = nppiDiv_32f_C3IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> resultData(width * height * channels);
    src2.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-4f));
  } else {
    NppImageMemory<Npp32f> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiDiv_32f_C3R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiDiv_32f_C3R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> resultData(width * height * channels);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-4f));
  }
}

INSTANTIATE_TEST_SUITE_P(Div32fC3, Div32fC3ParamTest,
                         ::testing::Values(Div32fC3Param{32, 32, false, false, "32x32_noCtx"},
                                           Div32fC3Param{32, 32, true, false, "32x32_Ctx"},
                                           Div32fC3Param{32, 32, false, true, "32x32_InPlace"},
                                           Div32fC3Param{32, 32, true, true, "32x32_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<Div32fC3Param> &info) { return info.param.name; });

// ==================== Div 32f C4 TEST_P ====================

struct Div32fC4Param {
  int width;
  int height;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Div32fC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<Div32fC4Param> {};

TEST_P(Div32fC4ParamTest, Div_32f_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;

  std::vector<Npp32f> src1Data(width * height * channels);
  std::vector<Npp32f> src2Data(width * height * channels);
  TestDataGenerator::generateRandom(src1Data, 1.0f, 100.0f, 12345);
  TestDataGenerator::generateRandom(src2Data, 1.0f, 10.0f, 54321);

  std::vector<Npp32f> expectedData(width * height * channels);
  for (size_t i = 0; i < expectedData.size(); i++) {
    // NPP Div: dst = src2 / src1
    expectedData[i] = expect::div<Npp32f>(src2Data[i], src1Data[i]);
  }

  NppImageMemory<Npp32f> src1(width * channels, height);
  NppImageMemory<Npp32f> src2(width * channels, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiDiv_32f_C4IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx);
    } else {
      status = nppiDiv_32f_C4IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> resultData(width * height * channels);
    src2.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-4f));
  } else {
    NppImageMemory<Npp32f> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiDiv_32f_C4R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiDiv_32f_C4R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> resultData(width * height * channels);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-4f));
  }
}

INSTANTIATE_TEST_SUITE_P(Div32fC4, Div32fC4ParamTest,
                         ::testing::Values(Div32fC4Param{32, 32, false, false, "32x32_noCtx"},
                                           Div32fC4Param{32, 32, true, false, "32x32_Ctx"},
                                           Div32fC4Param{32, 32, false, true, "32x32_InPlace"},
                                           Div32fC4Param{32, 32, true, true, "32x32_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<Div32fC4Param> &info) { return info.param.name; });

// ==================== Div 16u C3 with scale factor TEST_P ====================

class Div16uC3SfsParamTest : public NppTestBase, public ::testing::WithParamInterface<Div16uSfsParam> {};

TEST_P(Div16uC3SfsParamTest, Div_16u_C3RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int scaleFactor = param.scaleFactor;
  const int channels = 3;
  const int total = width * height * channels;

  std::vector<Npp16u> src1Data(total);
  std::vector<Npp16u> src2Data(total);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(100), static_cast<Npp16u>(10000), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(1), static_cast<Npp16u>(100), 54321);

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
      status = nppiDiv_16u_C3IRSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiDiv_16u_C3IRSfs(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    src2.copyToHost(resultData);
    for (size_t i = 0; i < resultData.size(); i++) {
      int expected = src2Data[i] / src1Data[i];
      EXPECT_LE(std::abs(static_cast<int>(resultData[i]) - expected), 1);
    }
  } else {
    NppImageMemory<Npp16u> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiDiv_16u_C3RSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                      scaleFactor, ctx);
    } else {
      status = nppiDiv_16u_C3RSfs(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    dst.copyToHost(resultData);
    for (size_t i = 0; i < resultData.size(); i++) {
      int expected = src2Data[i] / src1Data[i];
      EXPECT_LE(std::abs(static_cast<int>(resultData[i]) - expected), 1);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Div16uC3Sfs, Div16uC3SfsParamTest,
                         ::testing::Values(Div16uSfsParam{32, 32, 0, false, false, "32x32_sfs0_noCtx"},
                                           Div16uSfsParam{32, 32, 0, true, false, "32x32_sfs0_Ctx"},
                                           Div16uSfsParam{32, 32, 0, false, true, "32x32_sfs0_InPlace"},
                                           Div16uSfsParam{32, 32, 0, true, true, "32x32_sfs0_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<Div16uSfsParam> &info) { return info.param.name; });

// ==================== Div 16s C3 with scale factor TEST_P ====================

class Div16sC3SfsParamTest : public NppTestBase, public ::testing::WithParamInterface<Div16sSfsParam> {};

TEST_P(Div16sC3SfsParamTest, Div_16s_C3RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int scaleFactor = param.scaleFactor;
  const int channels = 3;
  const int total = width * height * channels;

  std::vector<Npp16s> src1Data(total);
  std::vector<Npp16s> src2Data(total);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16s>(10), static_cast<Npp16s>(1000), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16s>(1), static_cast<Npp16s>(10), 54321);

  NppImageMemory<Npp16s> src1(width * channels, height);
  NppImageMemory<Npp16s> src2(width * channels, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiDiv_16s_C3IRSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiDiv_16s_C3IRSfs(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16s> resultData(total);
    src2.copyToHost(resultData);
    for (size_t i = 0; i < resultData.size(); i++) {
      int expected = src2Data[i] / src1Data[i];
      EXPECT_LE(std::abs(static_cast<int>(resultData[i]) - expected), 1);
    }
  } else {
    NppImageMemory<Npp16s> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiDiv_16s_C3RSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                      scaleFactor, ctx);
    } else {
      status = nppiDiv_16s_C3RSfs(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16s> resultData(total);
    dst.copyToHost(resultData);
    for (size_t i = 0; i < resultData.size(); i++) {
      int expected = src2Data[i] / src1Data[i];
      EXPECT_LE(std::abs(static_cast<int>(resultData[i]) - expected), 1);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Div16sC3Sfs, Div16sC3SfsParamTest,
                         ::testing::Values(Div16sSfsParam{32, 32, 0, false, false, "32x32_sfs0_noCtx"},
                                           Div16sSfsParam{32, 32, 0, true, false, "32x32_sfs0_Ctx"},
                                           Div16sSfsParam{32, 32, 0, false, true, "32x32_sfs0_InPlace"},
                                           Div16sSfsParam{32, 32, 0, true, true, "32x32_sfs0_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<Div16sSfsParam> &info) { return info.param.name; });
