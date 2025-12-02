#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== SubC 32f TEST_P ====================

struct SubC32fParam {
  int width;
  int height;
  Npp32f constant;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class SubC32fParamTest : public NppTestBase, public ::testing::WithParamInterface<SubC32fParam> {};

TEST_P(SubC32fParamTest, SubC_32f_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp32f constant = param.constant;

  std::vector<Npp32f> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, -50.0f, 50.0f, 12345);

  std::vector<Npp32f> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::sub_c<Npp32f>(srcData[i], constant);
  }

  NppImageMemory<Npp32f> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiSubC_32f_C1IR_Ctx(constant, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiSubC_32f_C1IR(constant, src.get(), src.step(), roi);
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
      status = nppiSubC_32f_C1R_Ctx(src.get(), src.step(), constant, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiSubC_32f_C1R(src.get(), src.step(), constant, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
  }
}

INSTANTIATE_TEST_SUITE_P(SubC32f, SubC32fParamTest,
                         ::testing::Values(SubC32fParam{32, 32, 10.5f, false, false, "32x32_noCtx"},
                                           SubC32fParam{32, 32, 10.5f, true, false, "32x32_Ctx"},
                                           SubC32fParam{32, 32, 15.0f, false, true, "32x32_InPlace"},
                                           SubC32fParam{32, 32, 15.0f, true, true, "32x32_InPlace_Ctx"},
                                           SubC32fParam{64, 64, 20.0f, false, false, "64x64_noCtx"},
                                           SubC32fParam{64, 64, 20.0f, true, false, "64x64_Ctx"}),
                         [](const ::testing::TestParamInfo<SubC32fParam> &info) { return info.param.name; });

// ==================== SubC 8u with scale factor TEST_P ====================

struct SubC8uSfsParam {
  int width;
  int height;
  Npp8u constant;
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class SubC8uSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SubC8uSfsParam> {};

TEST_P(SubC8uSfsParamTest, SubC_8u_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp8u constant = param.constant;
  const int scaleFactor = param.scaleFactor;

  std::vector<Npp8u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(50), static_cast<Npp8u>(200), 12345);

  std::vector<Npp8u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::sub_c_sfs<Npp8u>(srcData[i], constant, scaleFactor);
  }

  NppImageMemory<Npp8u> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiSubC_8u_C1IRSfs_Ctx(constant, src.get(), src.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiSubC_8u_C1IRSfs(constant, src.get(), src.step(), roi, scaleFactor);
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
      status = nppiSubC_8u_C1RSfs_Ctx(src.get(), src.step(), constant, dst.get(), dst.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiSubC_8u_C1RSfs(src.get(), src.step(), constant, dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(SubC8uSfs, SubC8uSfsParamTest,
                         ::testing::Values(SubC8uSfsParam{32, 32, 30, 0, false, false, "32x32_sfs0_noCtx"},
                                           SubC8uSfsParam{32, 32, 30, 0, true, false, "32x32_sfs0_Ctx"},
                                           SubC8uSfsParam{32, 32, 20, 0, false, true, "32x32_sfs0_InPlace"},
                                           SubC8uSfsParam{32, 32, 20, 0, true, true, "32x32_sfs0_InPlace_Ctx"},
                                           SubC8uSfsParam{64, 64, 50, 0, false, false, "64x64_sfs0_noCtx"}),
                         [](const ::testing::TestParamInfo<SubC8uSfsParam> &info) { return info.param.name; });

// ==================== SubC 16u with scale factor TEST_P ====================

struct SubC16uSfsParam {
  int width;
  int height;
  Npp16u constant;
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class SubC16uSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SubC16uSfsParam> {};

TEST_P(SubC16uSfsParamTest, SubC_16u_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp16u constant = param.constant;
  const int scaleFactor = param.scaleFactor;

  std::vector<Npp16u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(2000), static_cast<Npp16u>(30000), 12345);

  std::vector<Npp16u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::sub_c_sfs<Npp16u>(srcData[i], constant, scaleFactor);
  }

  NppImageMemory<Npp16u> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiSubC_16u_C1IRSfs_Ctx(constant, src.get(), src.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiSubC_16u_C1IRSfs(constant, src.get(), src.step(), roi, scaleFactor);
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
      status = nppiSubC_16u_C1RSfs_Ctx(src.get(), src.step(), constant, dst.get(), dst.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiSubC_16u_C1RSfs(src.get(), src.step(), constant, dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(SubC16uSfs, SubC16uSfsParamTest,
                         ::testing::Values(SubC16uSfsParam{32, 32, 1000, 0, false, false, "32x32_sfs0_noCtx"},
                                           SubC16uSfsParam{32, 32, 1000, 0, true, false, "32x32_sfs0_Ctx"},
                                           SubC16uSfsParam{32, 32, 500, 0, false, true, "32x32_sfs0_InPlace"},
                                           SubC16uSfsParam{32, 32, 500, 0, true, true, "32x32_sfs0_InPlace_Ctx"},
                                           SubC16uSfsParam{64, 64, 2000, 0, false, false, "64x64_sfs0_noCtx"}),
                         [](const ::testing::TestParamInfo<SubC16uSfsParam> &info) { return info.param.name; });
