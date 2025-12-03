#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== DivC 32f TEST_P ====================

struct DivC32fParam {
  int width;
  int height;
  Npp32f constant;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class DivC32fParamTest : public NppTestBase, public ::testing::WithParamInterface<DivC32fParam> {};

TEST_P(DivC32fParamTest, DivC_32f_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp32f constant = param.constant;

  std::vector<Npp32f> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, 1.0f, 100.0f, 12345);

  std::vector<Npp32f> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::div_c<Npp32f>(srcData[i], constant);
  }

  NppImageMemory<Npp32f> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiDivC_32f_C1IR_Ctx(constant, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiDivC_32f_C1IR(constant, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> resultData(width * height);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-4f));
  } else {
    NppImageMemory<Npp32f> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiDivC_32f_C1R_Ctx(src.get(), src.step(), constant, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiDivC_32f_C1R(src.get(), src.step(), constant, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-4f));
  }
}

INSTANTIATE_TEST_SUITE_P(DivC32f, DivC32fParamTest,
                         ::testing::Values(DivC32fParam{32, 32, 2.5f, false, false, "32x32_noCtx"},
                                           DivC32fParam{32, 32, 2.5f, true, false, "32x32_Ctx"},
                                           DivC32fParam{32, 32, 4.0f, false, true, "32x32_InPlace"},
                                           DivC32fParam{32, 32, 4.0f, true, true, "32x32_InPlace_Ctx"},
                                           DivC32fParam{64, 64, 5.0f, false, false, "64x64_noCtx"},
                                           DivC32fParam{64, 64, 5.0f, true, false, "64x64_Ctx"}),
                         [](const ::testing::TestParamInfo<DivC32fParam> &info) { return info.param.name; });

// ==================== DivC 8u with scale factor TEST_P ====================

struct DivC8uSfsParam {
  int width;
  int height;
  Npp8u constant;
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class DivC8uSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DivC8uSfsParam> {};

TEST_P(DivC8uSfsParamTest, DivC_8u_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp8u constant = param.constant;
  const int scaleFactor = param.scaleFactor;

  std::vector<Npp8u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(10), static_cast<Npp8u>(200), 12345);

  NppImageMemory<Npp8u> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiDivC_8u_C1IRSfs_Ctx(constant, src.get(), src.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiDivC_8u_C1IRSfs(constant, src.get(), src.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(width * height);
    src.copyToHost(resultData);
    // Verify division results with tolerance for rounding differences
    for (size_t i = 0; i < resultData.size(); i++) {
      int expected = srcData[i] / constant;
      EXPECT_LE(std::abs(static_cast<int>(resultData[i]) - expected), 1);
    }
  } else {
    NppImageMemory<Npp8u> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiDivC_8u_C1RSfs_Ctx(src.get(), src.step(), constant, dst.get(), dst.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiDivC_8u_C1RSfs(src.get(), src.step(), constant, dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(width * height);
    dst.copyToHost(resultData);
    // Verify division results with tolerance for rounding differences
    for (size_t i = 0; i < resultData.size(); i++) {
      int expected = srcData[i] / constant;
      EXPECT_LE(std::abs(static_cast<int>(resultData[i]) - expected), 1);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(DivC8uSfs, DivC8uSfsParamTest,
                         ::testing::Values(DivC8uSfsParam{32, 32, 2, 0, false, false, "32x32_sfs0_noCtx"},
                                           DivC8uSfsParam{32, 32, 2, 0, true, false, "32x32_sfs0_Ctx"},
                                           DivC8uSfsParam{32, 32, 4, 0, false, true, "32x32_sfs0_InPlace"},
                                           DivC8uSfsParam{32, 32, 4, 0, true, true, "32x32_sfs0_InPlace_Ctx"},
                                           DivC8uSfsParam{64, 64, 5, 0, false, false, "64x64_sfs0_noCtx"}),
                         [](const ::testing::TestParamInfo<DivC8uSfsParam> &info) { return info.param.name; });

// ==================== DivC 16u with scale factor TEST_P ====================

struct DivC16uSfsParam {
  int width;
  int height;
  Npp16u constant;
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class DivC16uSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DivC16uSfsParam> {};

TEST_P(DivC16uSfsParamTest, DivC_16u_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp16u constant = param.constant;
  const int scaleFactor = param.scaleFactor;

  std::vector<Npp16u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(100), static_cast<Npp16u>(30000), 12345);

  NppImageMemory<Npp16u> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiDivC_16u_C1IRSfs_Ctx(constant, src.get(), src.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiDivC_16u_C1IRSfs(constant, src.get(), src.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(width * height);
    src.copyToHost(resultData);
    for (size_t i = 0; i < resultData.size(); i++) {
      int expected = srcData[i] / constant;
      EXPECT_LE(std::abs(static_cast<int>(resultData[i]) - expected), 1);
    }
  } else {
    NppImageMemory<Npp16u> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiDivC_16u_C1RSfs_Ctx(src.get(), src.step(), constant, dst.get(), dst.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiDivC_16u_C1RSfs(src.get(), src.step(), constant, dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(width * height);
    dst.copyToHost(resultData);
    for (size_t i = 0; i < resultData.size(); i++) {
      int expected = srcData[i] / constant;
      EXPECT_LE(std::abs(static_cast<int>(resultData[i]) - expected), 1);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(DivC16uSfs, DivC16uSfsParamTest,
                         ::testing::Values(DivC16uSfsParam{32, 32, 10, 0, false, false, "32x32_sfs0_noCtx"},
                                           DivC16uSfsParam{32, 32, 10, 0, true, false, "32x32_sfs0_Ctx"},
                                           DivC16uSfsParam{32, 32, 5, 0, false, true, "32x32_sfs0_InPlace"},
                                           DivC16uSfsParam{32, 32, 5, 0, true, true, "32x32_sfs0_InPlace_Ctx"},
                                           DivC16uSfsParam{64, 64, 20, 0, false, false, "64x64_sfs0_noCtx"}),
                         [](const ::testing::TestParamInfo<DivC16uSfsParam> &info) { return info.param.name; });

// ==================== DivC 16s with scale factor TEST_P ====================

struct DivC16sSfsParam {
  int width;
  int height;
  Npp16s constant;
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class DivC16sSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DivC16sSfsParam> {};

TEST_P(DivC16sSfsParamTest, DivC_16s_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp16s constant = param.constant;
  const int scaleFactor = param.scaleFactor;

  std::vector<Npp16s> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(10), static_cast<Npp16s>(1000), 12345);

  NppImageMemory<Npp16s> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiDivC_16s_C1IRSfs_Ctx(constant, src.get(), src.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiDivC_16s_C1IRSfs(constant, src.get(), src.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16s> resultData(width * height);
    src.copyToHost(resultData);
    for (size_t i = 0; i < resultData.size(); i++) {
      int expected = srcData[i] / constant;
      EXPECT_LE(std::abs(static_cast<int>(resultData[i]) - expected), 1);
    }
  } else {
    NppImageMemory<Npp16s> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiDivC_16s_C1RSfs_Ctx(src.get(), src.step(), constant, dst.get(), dst.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiDivC_16s_C1RSfs(src.get(), src.step(), constant, dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16s> resultData(width * height);
    dst.copyToHost(resultData);
    for (size_t i = 0; i < resultData.size(); i++) {
      int expected = srcData[i] / constant;
      EXPECT_LE(std::abs(static_cast<int>(resultData[i]) - expected), 1);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(DivC16sSfs, DivC16sSfsParamTest,
                         ::testing::Values(DivC16sSfsParam{32, 32, 10, 0, false, false, "32x32_sfs0_noCtx"},
                                           DivC16sSfsParam{32, 32, 10, 0, true, false, "32x32_sfs0_Ctx"},
                                           DivC16sSfsParam{32, 32, 5, 0, false, true, "32x32_sfs0_InPlace"},
                                           DivC16sSfsParam{32, 32, 5, 0, true, true, "32x32_sfs0_InPlace_Ctx"},
                                           DivC16sSfsParam{64, 64, 20, 0, false, false, "64x64_sfs0_noCtx"}),
                         [](const ::testing::TestParamInfo<DivC16sSfsParam> &info) { return info.param.name; });

// ==================== DivC 16f Tests ====================

class DivC16fTest : public NppTestBase {};

TEST_F(DivC16fTest, DivC_16f_C1R_BasicOperation) {
  const int width = 32;
  const int height = 32;
  const Npp32f constant = 2.5f;

  std::vector<Npp16f> srcData(width * height);
  TestDataGenerator::generateRandom16f(srcData, 1.0f, 50.0f, 12345);

  std::vector<Npp16f> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    float val = npp16f_to_float_host(srcData[i]);
    float result = val / constant;
    expectedData[i] = float_to_npp16f_host(result);
  }

  NppImageMemory<Npp16f> src(width, height);
  NppImageMemory<Npp16f> dst(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiDivC_16f_C1R(src.get(), src.step(), constant, dst.get(), dst.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp16f> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f));
}

TEST_F(DivC16fTest, DivC_16f_C1R_Ctx_BasicOperation) {
  const int width = 32;
  const int height = 32;
  const Npp32f constant = 2.5f;

  std::vector<Npp16f> srcData(width * height);
  TestDataGenerator::generateRandom16f(srcData, 1.0f, 50.0f, 12345);

  std::vector<Npp16f> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    float val = npp16f_to_float_host(srcData[i]);
    float result = val / constant;
    expectedData[i] = float_to_npp16f_host(result);
  }

  NppImageMemory<Npp16f> src(width, height);
  NppImageMemory<Npp16f> dst(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  ctx.hStream = 0;
  NppStatus status = nppiDivC_16f_C1R_Ctx(src.get(), src.step(), constant, dst.get(), dst.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp16f> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f));
}

TEST_F(DivC16fTest, DivC_16f_C1IR_BasicOperation) {
  const int width = 32;
  const int height = 32;
  const Npp32f constant = 2.5f;

  std::vector<Npp16f> srcData(width * height);
  TestDataGenerator::generateRandom16f(srcData, 1.0f, 50.0f, 12345);

  std::vector<Npp16f> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    float val = npp16f_to_float_host(srcData[i]);
    float result = val / constant;
    expectedData[i] = float_to_npp16f_host(result);
  }

  NppImageMemory<Npp16f> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiDivC_16f_C1IR(constant, src.get(), src.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp16f> resultData(width * height);
  src.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f));
}

TEST_F(DivC16fTest, DivC_16f_C1IR_Ctx_BasicOperation) {
  const int width = 32;
  const int height = 32;
  const Npp32f constant = 2.5f;

  std::vector<Npp16f> srcData(width * height);
  TestDataGenerator::generateRandom16f(srcData, 1.0f, 50.0f, 12345);

  std::vector<Npp16f> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    float val = npp16f_to_float_host(srcData[i]);
    float result = val / constant;
    expectedData[i] = float_to_npp16f_host(result);
  }

  NppImageMemory<Npp16f> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  ctx.hStream = 0;
  NppStatus status = nppiDivC_16f_C1IR_Ctx(constant, src.get(), src.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp16f> resultData(width * height);
  src.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f));
}

TEST_F(DivC16fTest, DivC_16f_C3R_BasicOperation) {
  const int width = 32;
  const int height = 32;
  const int channels = 3;
  const Npp32f constants[3] = {2.0f, 2.5f, 3.0f};

  std::vector<Npp16f> srcData(width * height * channels);
  TestDataGenerator::generateRandom16f(srcData, 1.0f, 50.0f, 12345);

  std::vector<Npp16f> expectedData(width * height * channels);
  for (int i = 0; i < width * height; i++) {
    for (int c = 0; c < channels; c++) {
      float val = npp16f_to_float_host(srcData[i * channels + c]);
      float result = val / constants[c];
      expectedData[i * channels + c] = float_to_npp16f_host(result);
    }
  }

  NppImageMemory<Npp16f> src(width * channels, height);
  NppImageMemory<Npp16f> dst(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiDivC_16f_C3R(src.get(), src.step(), constants, dst.get(), dst.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp16f> resultData(width * height * channels);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f));
}

TEST_F(DivC16fTest, DivC_16f_C3R_Ctx_BasicOperation) {
  const int width = 32;
  const int height = 32;
  const int channels = 3;
  const Npp32f constants[3] = {2.0f, 2.5f, 3.0f};

  std::vector<Npp16f> srcData(width * height * channels);
  TestDataGenerator::generateRandom16f(srcData, 1.0f, 50.0f, 12345);

  std::vector<Npp16f> expectedData(width * height * channels);
  for (int i = 0; i < width * height; i++) {
    for (int c = 0; c < channels; c++) {
      float val = npp16f_to_float_host(srcData[i * channels + c]);
      float result = val / constants[c];
      expectedData[i * channels + c] = float_to_npp16f_host(result);
    }
  }

  NppImageMemory<Npp16f> src(width * channels, height);
  NppImageMemory<Npp16f> dst(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  ctx.hStream = 0;
  NppStatus status = nppiDivC_16f_C3R_Ctx(src.get(), src.step(), constants, dst.get(), dst.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp16f> resultData(width * height * channels);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f));
}

TEST_F(DivC16fTest, DivC_16f_C3IR_BasicOperation) {
  const int width = 32;
  const int height = 32;
  const int channels = 3;
  const Npp32f constants[3] = {2.0f, 2.5f, 3.0f};

  std::vector<Npp16f> srcData(width * height * channels);
  TestDataGenerator::generateRandom16f(srcData, 1.0f, 50.0f, 12345);

  std::vector<Npp16f> expectedData(width * height * channels);
  for (int i = 0; i < width * height; i++) {
    for (int c = 0; c < channels; c++) {
      float val = npp16f_to_float_host(srcData[i * channels + c]);
      float result = val / constants[c];
      expectedData[i * channels + c] = float_to_npp16f_host(result);
    }
  }

  NppImageMemory<Npp16f> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiDivC_16f_C3IR(constants, src.get(), src.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp16f> resultData(width * height * channels);
  src.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f));
}

TEST_F(DivC16fTest, DivC_16f_C3IR_Ctx_BasicOperation) {
  const int width = 32;
  const int height = 32;
  const int channels = 3;
  const Npp32f constants[3] = {2.0f, 2.5f, 3.0f};

  std::vector<Npp16f> srcData(width * height * channels);
  TestDataGenerator::generateRandom16f(srcData, 1.0f, 50.0f, 12345);

  std::vector<Npp16f> expectedData(width * height * channels);
  for (int i = 0; i < width * height; i++) {
    for (int c = 0; c < channels; c++) {
      float val = npp16f_to_float_host(srcData[i * channels + c]);
      float result = val / constants[c];
      expectedData[i * channels + c] = float_to_npp16f_host(result);
    }
  }

  NppImageMemory<Npp16f> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  ctx.hStream = 0;
  NppStatus status = nppiDivC_16f_C3IR_Ctx(constants, src.get(), src.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp16f> resultData(width * height * channels);
  src.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f));
}

TEST_F(DivC16fTest, DivC_16f_C4R_BasicOperation) {
  const int width = 32;
  const int height = 32;
  const int channels = 4;
  const Npp32f constants[4] = {2.0f, 2.5f, 3.0f, 4.0f};

  std::vector<Npp16f> srcData(width * height * channels);
  TestDataGenerator::generateRandom16f(srcData, 1.0f, 50.0f, 12345);

  std::vector<Npp16f> expectedData(width * height * channels);
  for (int i = 0; i < width * height; i++) {
    for (int c = 0; c < channels; c++) {
      float val = npp16f_to_float_host(srcData[i * channels + c]);
      float result = val / constants[c];
      expectedData[i * channels + c] = float_to_npp16f_host(result);
    }
  }

  NppImageMemory<Npp16f> src(width * channels, height);
  NppImageMemory<Npp16f> dst(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiDivC_16f_C4R(src.get(), src.step(), constants, dst.get(), dst.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp16f> resultData(width * height * channels);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f));
}

TEST_F(DivC16fTest, DivC_16f_C4R_Ctx_BasicOperation) {
  const int width = 32;
  const int height = 32;
  const int channels = 4;
  const Npp32f constants[4] = {2.0f, 2.5f, 3.0f, 4.0f};

  std::vector<Npp16f> srcData(width * height * channels);
  TestDataGenerator::generateRandom16f(srcData, 1.0f, 50.0f, 12345);

  std::vector<Npp16f> expectedData(width * height * channels);
  for (int i = 0; i < width * height; i++) {
    for (int c = 0; c < channels; c++) {
      float val = npp16f_to_float_host(srcData[i * channels + c]);
      float result = val / constants[c];
      expectedData[i * channels + c] = float_to_npp16f_host(result);
    }
  }

  NppImageMemory<Npp16f> src(width * channels, height);
  NppImageMemory<Npp16f> dst(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  ctx.hStream = 0;
  NppStatus status = nppiDivC_16f_C4R_Ctx(src.get(), src.step(), constants, dst.get(), dst.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp16f> resultData(width * height * channels);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f));
}

TEST_F(DivC16fTest, DivC_16f_C4IR_BasicOperation) {
  const int width = 32;
  const int height = 32;
  const int channels = 4;
  const Npp32f constants[4] = {2.0f, 2.5f, 3.0f, 4.0f};

  std::vector<Npp16f> srcData(width * height * channels);
  TestDataGenerator::generateRandom16f(srcData, 1.0f, 50.0f, 12345);

  std::vector<Npp16f> expectedData(width * height * channels);
  for (int i = 0; i < width * height; i++) {
    for (int c = 0; c < channels; c++) {
      float val = npp16f_to_float_host(srcData[i * channels + c]);
      float result = val / constants[c];
      expectedData[i * channels + c] = float_to_npp16f_host(result);
    }
  }

  NppImageMemory<Npp16f> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiDivC_16f_C4IR(constants, src.get(), src.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp16f> resultData(width * height * channels);
  src.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f));
}

TEST_F(DivC16fTest, DivC_16f_C4IR_Ctx_BasicOperation) {
  const int width = 32;
  const int height = 32;
  const int channels = 4;
  const Npp32f constants[4] = {2.0f, 2.5f, 3.0f, 4.0f};

  std::vector<Npp16f> srcData(width * height * channels);
  TestDataGenerator::generateRandom16f(srcData, 1.0f, 50.0f, 12345);

  std::vector<Npp16f> expectedData(width * height * channels);
  for (int i = 0; i < width * height; i++) {
    for (int c = 0; c < channels; c++) {
      float val = npp16f_to_float_host(srcData[i * channels + c]);
      float result = val / constants[c];
      expectedData[i * channels + c] = float_to_npp16f_host(result);
    }
  }

  NppImageMemory<Npp16f> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  ctx.hStream = 0;
  NppStatus status = nppiDivC_16f_C4IR_Ctx(constants, src.get(), src.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp16f> resultData(width * height * channels);
  src.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f));
}
