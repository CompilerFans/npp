#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Mul 32f TEST_P ====================

struct Mul32fParam {
  int width;
  int height;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Mul32fParamTest : public NppTestBase, public ::testing::WithParamInterface<Mul32fParam> {};

TEST_P(Mul32fParamTest, Mul_32f_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> src1Data(width * height);
  std::vector<Npp32f> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, -10.0f, 10.0f, 12345);
  TestDataGenerator::generateRandom(src2Data, -10.0f, 10.0f, 54321);

  std::vector<Npp32f> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::mul<Npp32f>(src1Data[i], src2Data[i]);
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
      status = nppiMul_32f_C1IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx);
    } else {
      status = nppiMul_32f_C1IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
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
      status = nppiMul_32f_C1R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiMul_32f_C1R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-4f));
  }
}

INSTANTIATE_TEST_SUITE_P(Mul32f, Mul32fParamTest,
                         ::testing::Values(Mul32fParam{32, 32, false, false, "32x32_noCtx"},
                                           Mul32fParam{32, 32, true, false, "32x32_Ctx"},
                                           Mul32fParam{32, 32, false, true, "32x32_InPlace"},
                                           Mul32fParam{32, 32, true, true, "32x32_InPlace_Ctx"},
                                           Mul32fParam{64, 64, false, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<Mul32fParam> &info) { return info.param.name; });

// ==================== Mul 8u with scale factor TEST_P ====================

struct Mul8uSfsParam {
  int width;
  int height;
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Mul8uSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<Mul8uSfsParam> {};

TEST_P(Mul8uSfsParamTest, Mul_8u_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int scaleFactor = param.scaleFactor;

  std::vector<Npp8u> src1Data(width * height);
  std::vector<Npp8u> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(1), static_cast<Npp8u>(15), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8u>(1), static_cast<Npp8u>(15), 54321);

  std::vector<Npp8u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::mul_sfs<Npp8u>(src1Data[i], src2Data[i], scaleFactor);
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
      status = nppiMul_8u_C1IRSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiMul_8u_C1IRSfs(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor);
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
      status = nppiMul_8u_C1RSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                     scaleFactor, ctx);
    } else {
      status = nppiMul_8u_C1RSfs(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Mul8uSfs, Mul8uSfsParamTest,
                         ::testing::Values(Mul8uSfsParam{32, 32, 0, false, false, "32x32_sfs0_noCtx"},
                                           Mul8uSfsParam{32, 32, 0, true, false, "32x32_sfs0_Ctx"},
                                           Mul8uSfsParam{32, 32, 0, false, true, "32x32_sfs0_InPlace"},
                                           Mul8uSfsParam{32, 32, 0, true, true, "32x32_sfs0_InPlace_Ctx"},
                                           Mul8uSfsParam{64, 64, 0, false, false, "64x64_sfs0_noCtx"}),
                         [](const ::testing::TestParamInfo<Mul8uSfsParam> &info) { return info.param.name; });

// ==================== Mul 16u with scale factor TEST_P ====================

struct Mul16uSfsParam {
  int width;
  int height;
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Mul16uSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<Mul16uSfsParam> {};

TEST_P(Mul16uSfsParamTest, Mul_16u_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int scaleFactor = param.scaleFactor;

  std::vector<Npp16u> src1Data(width * height);
  std::vector<Npp16u> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(1), static_cast<Npp16u>(100), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(1), static_cast<Npp16u>(100), 54321);

  std::vector<Npp16u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::mul_sfs<Npp16u>(src1Data[i], src2Data[i], scaleFactor);
  }

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
      status = nppiMul_16u_C1IRSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiMul_16u_C1IRSfs(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(width * height);
    src2.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16u> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiMul_16u_C1RSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                      scaleFactor, ctx);
    } else {
      status = nppiMul_16u_C1RSfs(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Mul16uSfs, Mul16uSfsParamTest,
                         ::testing::Values(Mul16uSfsParam{32, 32, 0, false, false, "32x32_sfs0_noCtx"},
                                           Mul16uSfsParam{32, 32, 0, true, false, "32x32_sfs0_Ctx"},
                                           Mul16uSfsParam{32, 32, 0, false, true, "32x32_sfs0_InPlace"},
                                           Mul16uSfsParam{32, 32, 0, true, true, "32x32_sfs0_InPlace_Ctx"},
                                           Mul16uSfsParam{64, 64, 0, false, false, "64x64_sfs0_noCtx"}),
                         [](const ::testing::TestParamInfo<Mul16uSfsParam> &info) { return info.param.name; });

// ==================== Mul 16s with scale factor TEST_P ====================

struct Mul16sSfsParam {
  int width;
  int height;
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Mul16sSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<Mul16sSfsParam> {};

TEST_P(Mul16sSfsParamTest, Mul_16s_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int scaleFactor = param.scaleFactor;

  std::vector<Npp16s> src1Data(width * height);
  std::vector<Npp16s> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16s>(-50), static_cast<Npp16s>(50), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16s>(-50), static_cast<Npp16s>(50), 54321);

  std::vector<Npp16s> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::mul_sfs<Npp16s>(src1Data[i], src2Data[i], scaleFactor);
  }

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
      status = nppiMul_16s_C1IRSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiMul_16s_C1IRSfs(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16s> resultData(width * height);
    src2.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16s> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiMul_16s_C1RSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                      scaleFactor, ctx);
    } else {
      status = nppiMul_16s_C1RSfs(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16s> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Mul16sSfs, Mul16sSfsParamTest,
                         ::testing::Values(Mul16sSfsParam{32, 32, 0, false, false, "32x32_sfs0_noCtx"},
                                           Mul16sSfsParam{32, 32, 0, true, false, "32x32_sfs0_Ctx"},
                                           Mul16sSfsParam{32, 32, 0, false, true, "32x32_sfs0_InPlace"},
                                           Mul16sSfsParam{32, 32, 0, true, true, "32x32_sfs0_InPlace_Ctx"},
                                           Mul16sSfsParam{64, 64, 0, false, false, "64x64_sfs0_noCtx"}),
                         [](const ::testing::TestParamInfo<Mul16sSfsParam> &info) { return info.param.name; });

// ==================== Mul 32f C3 TEST_P ====================

struct Mul32fC3Param {
  int width;
  int height;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Mul32fC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<Mul32fC3Param> {};

TEST_P(Mul32fC3ParamTest, Mul_32f_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 3;

  std::vector<Npp32f> src1Data(width * height * channels);
  std::vector<Npp32f> src2Data(width * height * channels);
  TestDataGenerator::generateRandom(src1Data, -10.0f, 10.0f, 12345);
  TestDataGenerator::generateRandom(src2Data, -10.0f, 10.0f, 54321);

  std::vector<Npp32f> expectedData(width * height * channels);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::mul<Npp32f>(src1Data[i], src2Data[i]);
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
      status = nppiMul_32f_C3IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx);
    } else {
      status = nppiMul_32f_C3IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
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
      status = nppiMul_32f_C3R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiMul_32f_C3R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> resultData(width * height * channels);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-4f));
  }
}

INSTANTIATE_TEST_SUITE_P(Mul32fC3, Mul32fC3ParamTest,
                         ::testing::Values(Mul32fC3Param{32, 32, false, false, "32x32_noCtx"},
                                           Mul32fC3Param{32, 32, true, false, "32x32_Ctx"},
                                           Mul32fC3Param{32, 32, false, true, "32x32_InPlace"},
                                           Mul32fC3Param{32, 32, true, true, "32x32_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<Mul32fC3Param> &info) { return info.param.name; });

// ==================== Mul 32f C4 TEST_P ====================

struct Mul32fC4Param {
  int width;
  int height;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Mul32fC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<Mul32fC4Param> {};

TEST_P(Mul32fC4ParamTest, Mul_32f_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;

  std::vector<Npp32f> src1Data(width * height * channels);
  std::vector<Npp32f> src2Data(width * height * channels);
  TestDataGenerator::generateRandom(src1Data, -10.0f, 10.0f, 12345);
  TestDataGenerator::generateRandom(src2Data, -10.0f, 10.0f, 54321);

  std::vector<Npp32f> expectedData(width * height * channels);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::mul<Npp32f>(src1Data[i], src2Data[i]);
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
      status = nppiMul_32f_C4IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx);
    } else {
      status = nppiMul_32f_C4IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
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
      status = nppiMul_32f_C4R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiMul_32f_C4R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> resultData(width * height * channels);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-4f));
  }
}

INSTANTIATE_TEST_SUITE_P(Mul32fC4, Mul32fC4ParamTest,
                         ::testing::Values(Mul32fC4Param{32, 32, false, false, "32x32_noCtx"},
                                           Mul32fC4Param{32, 32, true, false, "32x32_Ctx"},
                                           Mul32fC4Param{32, 32, false, true, "32x32_InPlace"},
                                           Mul32fC4Param{32, 32, true, true, "32x32_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<Mul32fC4Param> &info) { return info.param.name; });
