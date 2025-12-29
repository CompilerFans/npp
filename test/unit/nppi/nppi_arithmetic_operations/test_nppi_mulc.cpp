#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== MulC 32f TEST_P ====================

struct MulC32fParam {
  int width;
  int height;
  Npp32f constant;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class MulC32fParamTest : public NppTestBase, public ::testing::WithParamInterface<MulC32fParam> {};

TEST_P(MulC32fParamTest, MulC_32f_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp32f constant = param.constant;

  std::vector<Npp32f> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, -50.0f, 50.0f, 12345);

  std::vector<Npp32f> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::mul_c<Npp32f>(srcData[i], constant);
  }

  NppImageMemory<Npp32f> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      nppGetStreamContext(&ctx);
      status = nppiMulC_32f_C1IR_Ctx(constant, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiMulC_32f_C1IR(constant, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> resultData(width * height);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-4f));
  } else {
    NppImageMemory<Npp32f> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      nppGetStreamContext(&ctx);
      status = nppiMulC_32f_C1R_Ctx(src.get(), src.step(), constant, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiMulC_32f_C1R(src.get(), src.step(), constant, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-4f));
  }
}

INSTANTIATE_TEST_SUITE_P(MulC32f, MulC32fParamTest,
                         ::testing::Values(MulC32fParam{32, 32, 2.5f, false, false, "32x32_noCtx"},
                                           MulC32fParam{32, 32, 2.5f, true, false, "32x32_Ctx"},
                                           MulC32fParam{32, 32, 1.5f, false, true, "32x32_InPlace"},
                                           MulC32fParam{32, 32, 1.5f, true, true, "32x32_InPlace_Ctx"},
                                           MulC32fParam{64, 64, 3.0f, false, false, "64x64_noCtx"},
                                           MulC32fParam{64, 64, 3.0f, true, false, "64x64_Ctx"}),
                         [](const ::testing::TestParamInfo<MulC32fParam> &info) { return info.param.name; });

// ==================== MulC 8u with scale factor TEST_P ====================

struct MulC8uSfsParam {
  int width;
  int height;
  Npp8u constant;
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class MulC8uSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<MulC8uSfsParam> {};

TEST_P(MulC8uSfsParamTest, MulC_8u_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp8u constant = param.constant;
  const int scaleFactor = param.scaleFactor;

  std::vector<Npp8u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(1), static_cast<Npp8u>(100), 12345);

  std::vector<Npp8u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::mul_c_sfs<Npp8u>(srcData[i], constant, scaleFactor);
  }

  NppImageMemory<Npp8u> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      nppGetStreamContext(&ctx);
      status = nppiMulC_8u_C1IRSfs_Ctx(constant, src.get(), src.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiMulC_8u_C1IRSfs(constant, src.get(), src.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(width * height);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp8u> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      nppGetStreamContext(&ctx);
      status = nppiMulC_8u_C1RSfs_Ctx(src.get(), src.step(), constant, dst.get(), dst.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiMulC_8u_C1RSfs(src.get(), src.step(), constant, dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(MulC8uSfs, MulC8uSfsParamTest,
                         ::testing::Values(MulC8uSfsParam{32, 32, 2, 0, false, false, "32x32_sfs0_noCtx"},
                                           MulC8uSfsParam{32, 32, 2, 0, true, false, "32x32_sfs0_Ctx"},
                                           MulC8uSfsParam{32, 32, 2, 0, false, true, "32x32_sfs0_InPlace"},
                                           MulC8uSfsParam{32, 32, 2, 0, true, true, "32x32_sfs0_InPlace_Ctx"},
                                           MulC8uSfsParam{64, 64, 2, 0, false, false, "64x64_sfs0_noCtx"}),
                         [](const ::testing::TestParamInfo<MulC8uSfsParam> &info) { return info.param.name; });

// ==================== MulC 16u with scale factor TEST_P ====================

struct MulC16uSfsParam {
  int width;
  int height;
  Npp16u constant;
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class MulC16uSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<MulC16uSfsParam> {};

TEST_P(MulC16uSfsParamTest, MulC_16u_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp16u constant = param.constant;
  const int scaleFactor = param.scaleFactor;

  std::vector<Npp16u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(1), static_cast<Npp16u>(1000), 12345);

  std::vector<Npp16u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::mul_c_sfs<Npp16u>(srcData[i], constant, scaleFactor);
  }

  NppImageMemory<Npp16u> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      nppGetStreamContext(&ctx);
      status = nppiMulC_16u_C1IRSfs_Ctx(constant, src.get(), src.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiMulC_16u_C1IRSfs(constant, src.get(), src.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(width * height);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16u> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      nppGetStreamContext(&ctx);
      status = nppiMulC_16u_C1RSfs_Ctx(src.get(), src.step(), constant, dst.get(), dst.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiMulC_16u_C1RSfs(src.get(), src.step(), constant, dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(MulC16uSfs, MulC16uSfsParamTest,
                         ::testing::Values(MulC16uSfsParam{32, 32, 10, 0, false, false, "32x32_sfs0_noCtx"},
                                           MulC16uSfsParam{32, 32, 10, 0, true, false, "32x32_sfs0_Ctx"},
                                           MulC16uSfsParam{32, 32, 5, 0, false, true, "32x32_sfs0_InPlace"},
                                           MulC16uSfsParam{32, 32, 5, 0, true, true, "32x32_sfs0_InPlace_Ctx"},
                                           MulC16uSfsParam{64, 64, 20, 0, false, false, "64x64_sfs0_noCtx"}),
                         [](const ::testing::TestParamInfo<MulC16uSfsParam> &info) { return info.param.name; });

// ==================== MulC 16s with scale factor TEST_P ====================

struct MulC16sSfsParam {
  int width;
  int height;
  Npp16s constant;
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class MulC16sSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<MulC16sSfsParam> {};

TEST_P(MulC16sSfsParamTest, MulC_16s_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp16s constant = param.constant;
  const int scaleFactor = param.scaleFactor;

  std::vector<Npp16s> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(-100), static_cast<Npp16s>(100), 12345);

  std::vector<Npp16s> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::mul_c_sfs<Npp16s>(srcData[i], constant, scaleFactor);
  }

  NppImageMemory<Npp16s> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      nppGetStreamContext(&ctx);
      status = nppiMulC_16s_C1IRSfs_Ctx(constant, src.get(), src.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiMulC_16s_C1IRSfs(constant, src.get(), src.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16s> resultData(width * height);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16s> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      nppGetStreamContext(&ctx);
      status = nppiMulC_16s_C1RSfs_Ctx(src.get(), src.step(), constant, dst.get(), dst.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiMulC_16s_C1RSfs(src.get(), src.step(), constant, dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16s> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(MulC16sSfs, MulC16sSfsParamTest,
                         ::testing::Values(MulC16sSfsParam{32, 32, 10, 0, false, false, "32x32_sfs0_noCtx"},
                                           MulC16sSfsParam{32, 32, 10, 0, true, false, "32x32_sfs0_Ctx"},
                                           MulC16sSfsParam{32, 32, 5, 0, false, true, "32x32_sfs0_InPlace"},
                                           MulC16sSfsParam{32, 32, 5, 0, true, true, "32x32_sfs0_InPlace_Ctx"},
                                           MulC16sSfsParam{64, 64, 20, 0, false, false, "64x64_sfs0_noCtx"}),
                         [](const ::testing::TestParamInfo<MulC16sSfsParam> &info) { return info.param.name; });

// ==================== MulC 16f (half-precision float) tests ====================

class MulC16fTest : public NppTestBase {};

TEST_F(MulC16fTest, MulC_16f_C1R_BasicOperation) {
  const int width = 32;
  const int height = 32;
  const Npp32f constant = 2.5f;

  std::vector<Npp16f> srcData(width * height);
  std::vector<Npp16f> expectedData(width * height);

  TestDataGenerator::generateRandom16f(srcData, -20.0f, 20.0f, 12345);

  for (size_t i = 0; i < expectedData.size(); i++) {
    float val = npp16f_to_float_host(srcData[i]);
    expectedData[i] = float_to_npp16f_host(val * constant);
  }

  NppImageMemory<Npp16f> src(width, height);
  NppImageMemory<Npp16f> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiMulC_16f_C1R(src.get(), src.step(), constant, dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiMulC_16f_C1R failed";

  std::vector<Npp16f> resultData(width * height);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f))
      << "MulC 16f C1R operation produced incorrect results";
}

TEST_F(MulC16fTest, MulC_16f_C1R_Ctx_BasicOperation) {
  const int width = 32;
  const int height = 32;
  const Npp32f constant = 2.5f;

  std::vector<Npp16f> srcData(width * height);
  std::vector<Npp16f> expectedData(width * height);

  TestDataGenerator::generateRandom16f(srcData, -20.0f, 20.0f, 12345);

  for (size_t i = 0; i < expectedData.size(); i++) {
    float val = npp16f_to_float_host(srcData[i]);
    expectedData[i] = float_to_npp16f_host(val * constant);
  }

  NppImageMemory<Npp16f> src(width, height);
  NppImageMemory<Npp16f> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status = nppiMulC_16f_C1R_Ctx(src.get(), src.step(), constant, dst.get(), dst.step(), roi, ctx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiMulC_16f_C1R_Ctx failed";

  std::vector<Npp16f> resultData(width * height);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f))
      << "MulC 16f C1R Ctx operation produced incorrect results";
}

TEST_F(MulC16fTest, MulC_16f_C1IR_InPlaceOperation) {
  const int width = 32;
  const int height = 32;
  const Npp32f constant = 1.5f;

  std::vector<Npp16f> srcData(width * height);
  std::vector<Npp16f> expectedData(width * height);

  TestDataGenerator::generateRandom16f(srcData, -20.0f, 20.0f, 54321);

  for (size_t i = 0; i < expectedData.size(); i++) {
    float val = npp16f_to_float_host(srcData[i]);
    expectedData[i] = float_to_npp16f_host(val * constant);
  }

  NppImageMemory<Npp16f> srcDst(width, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiMulC_16f_C1IR(constant, srcDst.get(), srcDst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiMulC_16f_C1IR failed";

  std::vector<Npp16f> resultData(width * height);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f))
      << "In-place MulC 16f C1 operation produced incorrect results";
}

TEST_F(MulC16fTest, MulC_16f_C1IR_Ctx_InPlaceOperation) {
  const int width = 32;
  const int height = 32;
  const Npp32f constant = 1.5f;

  std::vector<Npp16f> srcData(width * height);
  std::vector<Npp16f> expectedData(width * height);

  TestDataGenerator::generateRandom16f(srcData, -20.0f, 20.0f, 54321);

  for (size_t i = 0; i < expectedData.size(); i++) {
    float val = npp16f_to_float_host(srcData[i]);
    expectedData[i] = float_to_npp16f_host(val * constant);
  }

  NppImageMemory<Npp16f> srcDst(width, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status = nppiMulC_16f_C1IR_Ctx(constant, srcDst.get(), srcDst.step(), roi, ctx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiMulC_16f_C1IR_Ctx failed";

  std::vector<Npp16f> resultData(width * height);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f))
      << "In-place MulC 16f C1 Ctx operation produced incorrect results";
}

// 16f C3 tests
TEST_F(MulC16fTest, MulC_16f_C3R_BasicOperation) {
  const int width = 32;
  const int height = 32;
  const int channels = 3;
  const Npp32f aConstants[3] = {1.5f, 2.0f, 2.5f};

  std::vector<Npp16f> srcData(width * height * channels);
  std::vector<Npp16f> expectedData(width * height * channels);

  TestDataGenerator::generateRandom16f(srcData, -20.0f, 20.0f, 33333);

  for (size_t i = 0; i < width * height; i++) {
    for (int c = 0; c < channels; c++) {
      float val = npp16f_to_float_host(srcData[i * channels + c]);
      expectedData[i * channels + c] = float_to_npp16f_host(val * aConstants[c]);
    }
  }

  NppImageMemory<Npp16f> src(width * channels, height);
  NppImageMemory<Npp16f> dst(width * channels, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiMulC_16f_C3R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiMulC_16f_C3R failed";

  std::vector<Npp16f> resultData(width * height * channels);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f))
      << "MulC 16f C3R operation produced incorrect results";
}

TEST_F(MulC16fTest, MulC_16f_C3R_Ctx_BasicOperation) {
  const int width = 32;
  const int height = 32;
  const int channels = 3;
  const Npp32f aConstants[3] = {1.5f, 2.0f, 2.5f};

  std::vector<Npp16f> srcData(width * height * channels);
  std::vector<Npp16f> expectedData(width * height * channels);

  TestDataGenerator::generateRandom16f(srcData, -20.0f, 20.0f, 33333);

  for (size_t i = 0; i < width * height; i++) {
    for (int c = 0; c < channels; c++) {
      float val = npp16f_to_float_host(srcData[i * channels + c]);
      expectedData[i * channels + c] = float_to_npp16f_host(val * aConstants[c]);
    }
  }

  NppImageMemory<Npp16f> src(width * channels, height);
  NppImageMemory<Npp16f> dst(width * channels, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status = nppiMulC_16f_C3R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiMulC_16f_C3R_Ctx failed";

  std::vector<Npp16f> resultData(width * height * channels);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f))
      << "MulC 16f C3R Ctx operation produced incorrect results";
}

TEST_F(MulC16fTest, MulC_16f_C3IR_InPlaceOperation) {
  const int width = 32;
  const int height = 32;
  const int channels = 3;
  const Npp32f aConstants[3] = {1.5f, 2.0f, 2.5f};

  std::vector<Npp16f> srcData(width * height * channels);
  std::vector<Npp16f> expectedData(width * height * channels);

  TestDataGenerator::generateRandom16f(srcData, -20.0f, 20.0f, 44444);

  for (size_t i = 0; i < width * height; i++) {
    for (int c = 0; c < channels; c++) {
      float val = npp16f_to_float_host(srcData[i * channels + c]);
      expectedData[i * channels + c] = float_to_npp16f_host(val * aConstants[c]);
    }
  }

  NppImageMemory<Npp16f> srcDst(width * channels, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiMulC_16f_C3IR(aConstants, srcDst.get(), srcDst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiMulC_16f_C3IR failed";

  std::vector<Npp16f> resultData(width * height * channels);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f))
      << "In-place MulC 16f C3 operation produced incorrect results";
}

TEST_F(MulC16fTest, MulC_16f_C3IR_Ctx_InPlaceOperation) {
  const int width = 32;
  const int height = 32;
  const int channels = 3;
  const Npp32f aConstants[3] = {1.5f, 2.0f, 2.5f};

  std::vector<Npp16f> srcData(width * height * channels);
  std::vector<Npp16f> expectedData(width * height * channels);

  TestDataGenerator::generateRandom16f(srcData, -20.0f, 20.0f, 44444);

  for (size_t i = 0; i < width * height; i++) {
    for (int c = 0; c < channels; c++) {
      float val = npp16f_to_float_host(srcData[i * channels + c]);
      expectedData[i * channels + c] = float_to_npp16f_host(val * aConstants[c]);
    }
  }

  NppImageMemory<Npp16f> srcDst(width * channels, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status = nppiMulC_16f_C3IR_Ctx(aConstants, srcDst.get(), srcDst.step(), roi, ctx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiMulC_16f_C3IR_Ctx failed";

  std::vector<Npp16f> resultData(width * height * channels);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f))
      << "In-place MulC 16f C3 Ctx operation produced incorrect results";
}

// 16f C4 tests
TEST_F(MulC16fTest, MulC_16f_C4R_BasicOperation) {
  const int width = 32;
  const int height = 32;
  const int channels = 4;
  const Npp32f aConstants[4] = {1.5f, 2.0f, 2.5f, 3.0f};

  std::vector<Npp16f> srcData(width * height * channels);
  std::vector<Npp16f> expectedData(width * height * channels);

  TestDataGenerator::generateRandom16f(srcData, -20.0f, 20.0f, 55555);

  for (size_t i = 0; i < width * height; i++) {
    for (int c = 0; c < channels; c++) {
      float val = npp16f_to_float_host(srcData[i * channels + c]);
      expectedData[i * channels + c] = float_to_npp16f_host(val * aConstants[c]);
    }
  }

  NppImageMemory<Npp16f> src(width * channels, height);
  NppImageMemory<Npp16f> dst(width * channels, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiMulC_16f_C4R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiMulC_16f_C4R failed";

  std::vector<Npp16f> resultData(width * height * channels);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f))
      << "MulC 16f C4R operation produced incorrect results";
}

TEST_F(MulC16fTest, MulC_16f_C4R_Ctx_BasicOperation) {
  const int width = 32;
  const int height = 32;
  const int channels = 4;
  const Npp32f aConstants[4] = {1.5f, 2.0f, 2.5f, 3.0f};

  std::vector<Npp16f> srcData(width * height * channels);
  std::vector<Npp16f> expectedData(width * height * channels);

  TestDataGenerator::generateRandom16f(srcData, -20.0f, 20.0f, 55555);

  for (size_t i = 0; i < width * height; i++) {
    for (int c = 0; c < channels; c++) {
      float val = npp16f_to_float_host(srcData[i * channels + c]);
      expectedData[i * channels + c] = float_to_npp16f_host(val * aConstants[c]);
    }
  }

  NppImageMemory<Npp16f> src(width * channels, height);
  NppImageMemory<Npp16f> dst(width * channels, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status = nppiMulC_16f_C4R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiMulC_16f_C4R_Ctx failed";

  std::vector<Npp16f> resultData(width * height * channels);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f))
      << "MulC 16f C4R Ctx operation produced incorrect results";
}

TEST_F(MulC16fTest, MulC_16f_C4IR_InPlaceOperation) {
  const int width = 32;
  const int height = 32;
  const int channels = 4;
  const Npp32f aConstants[4] = {1.5f, 2.0f, 2.5f, 3.0f};

  std::vector<Npp16f> srcData(width * height * channels);
  std::vector<Npp16f> expectedData(width * height * channels);

  TestDataGenerator::generateRandom16f(srcData, -20.0f, 20.0f, 66666);

  for (size_t i = 0; i < width * height; i++) {
    for (int c = 0; c < channels; c++) {
      float val = npp16f_to_float_host(srcData[i * channels + c]);
      expectedData[i * channels + c] = float_to_npp16f_host(val * aConstants[c]);
    }
  }

  NppImageMemory<Npp16f> srcDst(width * channels, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiMulC_16f_C4IR(aConstants, srcDst.get(), srcDst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiMulC_16f_C4IR failed";

  std::vector<Npp16f> resultData(width * height * channels);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f))
      << "In-place MulC 16f C4 operation produced incorrect results";
}

TEST_F(MulC16fTest, MulC_16f_C4IR_Ctx_InPlaceOperation) {
  const int width = 32;
  const int height = 32;
  const int channels = 4;
  const Npp32f aConstants[4] = {1.5f, 2.0f, 2.5f, 3.0f};

  std::vector<Npp16f> srcData(width * height * channels);
  std::vector<Npp16f> expectedData(width * height * channels);

  TestDataGenerator::generateRandom16f(srcData, -20.0f, 20.0f, 66666);

  for (size_t i = 0; i < width * height; i++) {
    for (int c = 0; c < channels; c++) {
      float val = npp16f_to_float_host(srcData[i * channels + c]);
      expectedData[i * channels + c] = float_to_npp16f_host(val * aConstants[c]);
    }
  }

  NppImageMemory<Npp16f> srcDst(width * channels, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status = nppiMulC_16f_C4IR_Ctx(aConstants, srcDst.get(), srcDst.step(), roi, ctx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiMulC_16f_C4IR_Ctx failed";

  std::vector<Npp16f> resultData(width * height * channels);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f))
      << "In-place MulC 16f C4 Ctx operation produced incorrect results";
}
