#include "npp_test_base.h"

using namespace npp_functional_test;

namespace {
template <typename T> T min_every(T a, T b) { return (a < b) ? a : b; }
} // namespace

// ==================== MinEvery 8u C1 TEST_P ====================

struct MinEvery8uParam {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class MinEvery8uParamTest : public NppTestBase, public ::testing::WithParamInterface<MinEvery8uParam> {};

TEST_P(MinEvery8uParamTest, MinEvery_8u_C1IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height);
  std::vector<Npp8u> dstData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 54321);

  std::vector<Npp8u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = min_every<Npp8u>(srcData[i], dstData[i]);
  }

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);
  src.copyFromHost(srcData);
  dst.copyFromHost(dstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiMinEvery_8u_C1IR_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMinEvery_8u_C1IR(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(MinEvery8u, MinEvery8uParamTest,
                         ::testing::Values(MinEvery8uParam{32, 32, false, "32x32_noCtx"},
                                           MinEvery8uParam{32, 32, true, "32x32_Ctx"},
                                           MinEvery8uParam{64, 64, false, "64x64_noCtx"},
                                           MinEvery8uParam{64, 64, true, "64x64_Ctx"}),
                         [](const ::testing::TestParamInfo<MinEvery8uParam> &info) { return info.param.name; });

// ==================== MinEvery 16u C1 TEST_P ====================

struct MinEvery16uParam {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class MinEvery16uParamTest : public NppTestBase, public ::testing::WithParamInterface<MinEvery16uParam> {};

TEST_P(MinEvery16uParamTest, MinEvery_16u_C1IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcData(width * height);
  std::vector<Npp16u> dstData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 54321);

  std::vector<Npp16u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = min_every<Npp16u>(srcData[i], dstData[i]);
  }

  NppImageMemory<Npp16u> src(width, height);
  NppImageMemory<Npp16u> dst(width, height);
  src.copyFromHost(srcData);
  dst.copyFromHost(dstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiMinEvery_16u_C1IR_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMinEvery_16u_C1IR(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp16u> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(MinEvery16u, MinEvery16uParamTest,
                         ::testing::Values(MinEvery16uParam{32, 32, false, "32x32_noCtx"},
                                           MinEvery16uParam{32, 32, true, "32x32_Ctx"},
                                           MinEvery16uParam{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<MinEvery16uParam> &info) { return info.param.name; });

// ==================== MinEvery 32f C1 TEST_P ====================

struct MinEvery32fParam {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class MinEvery32fParamTest : public NppTestBase, public ::testing::WithParamInterface<MinEvery32fParam> {};

TEST_P(MinEvery32fParamTest, MinEvery_32f_C1IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> srcData(width * height);
  std::vector<Npp32f> dstData(width * height);
  TestDataGenerator::generateRandom(srcData, -100.0f, 100.0f, 12345);
  TestDataGenerator::generateRandom(dstData, -100.0f, 100.0f, 54321);

  std::vector<Npp32f> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = min_every<Npp32f>(srcData[i], dstData[i]);
  }

  NppImageMemory<Npp32f> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);
  src.copyFromHost(srcData);
  dst.copyFromHost(dstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiMinEvery_32f_C1IR_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMinEvery_32f_C1IR(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32f> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
}

INSTANTIATE_TEST_SUITE_P(MinEvery32f, MinEvery32fParamTest,
                         ::testing::Values(MinEvery32fParam{32, 32, false, "32x32_noCtx"},
                                           MinEvery32fParam{32, 32, true, "32x32_Ctx"},
                                           MinEvery32fParam{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<MinEvery32fParam> &info) { return info.param.name; });

// ==================== MinEvery 8u C3 TEST_P ====================

struct MinEvery8uC3Param {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class MinEvery8uC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<MinEvery8uC3Param> {};

TEST_P(MinEvery8uC3ParamTest, MinEvery_8u_C3IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 3;
  const int total = width * height * channels;

  std::vector<Npp8u> srcData(total);
  std::vector<Npp8u> dstData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 54321);

  std::vector<Npp8u> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = min_every<Npp8u>(srcData[i], dstData[i]);
  }

  NppImageMemory<Npp8u> src(width * channels, height);
  NppImageMemory<Npp8u> dst(width * channels, height);
  src.copyFromHost(srcData);
  dst.copyFromHost(dstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiMinEvery_8u_C3IR_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMinEvery_8u_C3IR(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(total);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(MinEvery8uC3, MinEvery8uC3ParamTest,
                         ::testing::Values(MinEvery8uC3Param{32, 32, false, "32x32_noCtx"},
                                           MinEvery8uC3Param{32, 32, true, "32x32_Ctx"},
                                           MinEvery8uC3Param{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<MinEvery8uC3Param> &info) { return info.param.name; });

// ==================== MinEvery 8u C4 TEST_P ====================

struct MinEvery8uC4Param {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class MinEvery8uC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<MinEvery8uC4Param> {};

TEST_P(MinEvery8uC4ParamTest, MinEvery_8u_C4IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  std::vector<Npp8u> srcData(total);
  std::vector<Npp8u> dstData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 54321);

  std::vector<Npp8u> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = min_every<Npp8u>(srcData[i], dstData[i]);
  }

  NppImageMemory<Npp8u> src(width * channels, height);
  NppImageMemory<Npp8u> dst(width * channels, height);
  src.copyFromHost(srcData);
  dst.copyFromHost(dstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiMinEvery_8u_C4IR_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMinEvery_8u_C4IR(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(total);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(MinEvery8uC4, MinEvery8uC4ParamTest,
                         ::testing::Values(MinEvery8uC4Param{32, 32, false, "32x32_noCtx"},
                                           MinEvery8uC4Param{32, 32, true, "32x32_Ctx"},
                                           MinEvery8uC4Param{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<MinEvery8uC4Param> &info) { return info.param.name; });

// ==================== MinEvery 16u C3 TEST_P ====================

struct MinEvery16uC3Param {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class MinEvery16uC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<MinEvery16uC3Param> {};

TEST_P(MinEvery16uC3ParamTest, MinEvery_16u_C3IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 3;
  const int total = width * height * channels;

  std::vector<Npp16u> srcData(total);
  std::vector<Npp16u> dstData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 54321);

  std::vector<Npp16u> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = min_every<Npp16u>(srcData[i], dstData[i]);
  }

  NppImageMemory<Npp16u> src(width * channels, height);
  NppImageMemory<Npp16u> dst(width * channels, height);
  src.copyFromHost(srcData);
  dst.copyFromHost(dstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiMinEvery_16u_C3IR_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMinEvery_16u_C3IR(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp16u> resultData(total);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(MinEvery16uC3, MinEvery16uC3ParamTest,
                         ::testing::Values(MinEvery16uC3Param{32, 32, false, "32x32_noCtx"},
                                           MinEvery16uC3Param{32, 32, true, "32x32_Ctx"},
                                           MinEvery16uC3Param{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<MinEvery16uC3Param> &info) { return info.param.name; });

// ==================== MinEvery 16u C4 TEST_P ====================

struct MinEvery16uC4Param {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class MinEvery16uC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<MinEvery16uC4Param> {};

TEST_P(MinEvery16uC4ParamTest, MinEvery_16u_C4IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  std::vector<Npp16u> srcData(total);
  std::vector<Npp16u> dstData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 54321);

  std::vector<Npp16u> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = min_every<Npp16u>(srcData[i], dstData[i]);
  }

  NppImageMemory<Npp16u> src(width * channels, height);
  NppImageMemory<Npp16u> dst(width * channels, height);
  src.copyFromHost(srcData);
  dst.copyFromHost(dstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiMinEvery_16u_C4IR_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMinEvery_16u_C4IR(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp16u> resultData(total);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(MinEvery16uC4, MinEvery16uC4ParamTest,
                         ::testing::Values(MinEvery16uC4Param{32, 32, false, "32x32_noCtx"},
                                           MinEvery16uC4Param{32, 32, true, "32x32_Ctx"},
                                           MinEvery16uC4Param{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<MinEvery16uC4Param> &info) { return info.param.name; });

// ==================== MinEvery 32f C3 TEST_P ====================

struct MinEvery32fC3Param {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class MinEvery32fC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<MinEvery32fC3Param> {};

TEST_P(MinEvery32fC3ParamTest, MinEvery_32f_C3IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 3;
  const int total = width * height * channels;

  std::vector<Npp32f> srcData(total);
  std::vector<Npp32f> dstData(total);
  TestDataGenerator::generateRandom(srcData, -100.0f, 100.0f, 12345);
  TestDataGenerator::generateRandom(dstData, -100.0f, 100.0f, 54321);

  std::vector<Npp32f> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = min_every<Npp32f>(srcData[i], dstData[i]);
  }

  NppImageMemory<Npp32f> src(width * channels, height);
  NppImageMemory<Npp32f> dst(width * channels, height);
  src.copyFromHost(srcData);
  dst.copyFromHost(dstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiMinEvery_32f_C3IR_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMinEvery_32f_C3IR(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32f> resultData(total);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
}

INSTANTIATE_TEST_SUITE_P(MinEvery32fC3, MinEvery32fC3ParamTest,
                         ::testing::Values(MinEvery32fC3Param{32, 32, false, "32x32_noCtx"},
                                           MinEvery32fC3Param{32, 32, true, "32x32_Ctx"},
                                           MinEvery32fC3Param{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<MinEvery32fC3Param> &info) { return info.param.name; });

// ==================== MinEvery 32f C4 TEST_P ====================

struct MinEvery32fC4Param {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class MinEvery32fC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<MinEvery32fC4Param> {};

TEST_P(MinEvery32fC4ParamTest, MinEvery_32f_C4IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  std::vector<Npp32f> srcData(total);
  std::vector<Npp32f> dstData(total);
  TestDataGenerator::generateRandom(srcData, -100.0f, 100.0f, 12345);
  TestDataGenerator::generateRandom(dstData, -100.0f, 100.0f, 54321);

  std::vector<Npp32f> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = min_every<Npp32f>(srcData[i], dstData[i]);
  }

  NppImageMemory<Npp32f> src(width * channels, height);
  NppImageMemory<Npp32f> dst(width * channels, height);
  src.copyFromHost(srcData);
  dst.copyFromHost(dstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiMinEvery_32f_C4IR_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMinEvery_32f_C4IR(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32f> resultData(total);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
}

INSTANTIATE_TEST_SUITE_P(MinEvery32fC4, MinEvery32fC4ParamTest,
                         ::testing::Values(MinEvery32fC4Param{32, 32, false, "32x32_noCtx"},
                                           MinEvery32fC4Param{32, 32, true, "32x32_Ctx"},
                                           MinEvery32fC4Param{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<MinEvery32fC4Param> &info) { return info.param.name; });
