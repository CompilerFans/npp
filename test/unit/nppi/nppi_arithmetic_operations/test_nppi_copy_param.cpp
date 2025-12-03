#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Copy 8u C1 TEST_P ====================

struct Copy8uParam {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class Copy8uParamTest : public NppTestBase, public ::testing::WithParamInterface<Copy8uParam> {};

TEST_P(Copy8uParamTest, Copy_8u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiCopy_8u_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiCopy_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, srcData));
}

INSTANTIATE_TEST_SUITE_P(Copy8u, Copy8uParamTest,
                         ::testing::Values(Copy8uParam{32, 32, false, "32x32_noCtx"},
                                           Copy8uParam{32, 32, true, "32x32_Ctx"},
                                           Copy8uParam{64, 64, false, "64x64_noCtx"},
                                           Copy8uParam{64, 64, true, "64x64_Ctx"}),
                         [](const ::testing::TestParamInfo<Copy8uParam> &info) { return info.param.name; });

// ==================== Copy 16u C1 TEST_P ====================

struct Copy16uParam {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class Copy16uParamTest : public NppTestBase, public ::testing::WithParamInterface<Copy16uParam> {};

TEST_P(Copy16uParamTest, Copy_16u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);

  NppImageMemory<Npp16u> src(width, height);
  NppImageMemory<Npp16u> dst(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiCopy_16u_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiCopy_16u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp16u> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, srcData));
}

INSTANTIATE_TEST_SUITE_P(Copy16u, Copy16uParamTest,
                         ::testing::Values(Copy16uParam{32, 32, false, "32x32_noCtx"},
                                           Copy16uParam{32, 32, true, "32x32_Ctx"},
                                           Copy16uParam{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<Copy16uParam> &info) { return info.param.name; });

// ==================== Copy 32f C1 TEST_P ====================

struct Copy32fParam {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class Copy32fParamTest : public NppTestBase, public ::testing::WithParamInterface<Copy32fParam> {};

TEST_P(Copy32fParamTest, Copy_32f_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, -100.0f, 100.0f, 12345);

  NppImageMemory<Npp32f> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiCopy_32f_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiCopy_32f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32f> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, srcData, 1e-5f));
}

INSTANTIATE_TEST_SUITE_P(Copy32f, Copy32fParamTest,
                         ::testing::Values(Copy32fParam{32, 32, false, "32x32_noCtx"},
                                           Copy32fParam{32, 32, true, "32x32_Ctx"},
                                           Copy32fParam{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<Copy32fParam> &info) { return info.param.name; });

// ==================== Copy 8u C3 TEST_P ====================

struct Copy8uC3Param {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class Copy8uC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<Copy8uC3Param> {};

TEST_P(Copy8uC3ParamTest, Copy_8u_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height * 3);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  NppImageMemory<Npp8u> src(width * 3, height);
  NppImageMemory<Npp8u> dst(width * 3, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiCopy_8u_C3R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiCopy_8u_C3R(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(width * height * 3);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, srcData));
}

INSTANTIATE_TEST_SUITE_P(Copy8uC3, Copy8uC3ParamTest,
                         ::testing::Values(Copy8uC3Param{32, 32, false, "32x32_noCtx"},
                                           Copy8uC3Param{32, 32, true, "32x32_Ctx"},
                                           Copy8uC3Param{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<Copy8uC3Param> &info) { return info.param.name; });

// ==================== Copy 8u C4 TEST_P ====================

struct Copy8uC4Param {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class Copy8uC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<Copy8uC4Param> {};

TEST_P(Copy8uC4ParamTest, Copy_8u_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height * 4);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  NppImageMemory<Npp8u> src(width * 4, height);
  NppImageMemory<Npp8u> dst(width * 4, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiCopy_8u_C4R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiCopy_8u_C4R(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(width * height * 4);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, srcData));
}

INSTANTIATE_TEST_SUITE_P(Copy8uC4, Copy8uC4ParamTest,
                         ::testing::Values(Copy8uC4Param{32, 32, false, "32x32_noCtx"},
                                           Copy8uC4Param{32, 32, true, "32x32_Ctx"},
                                           Copy8uC4Param{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<Copy8uC4Param> &info) { return info.param.name; });
