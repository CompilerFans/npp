#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== MaxEvery 8u C1 TEST_P ====================

struct MaxEvery8uParam {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class MaxEvery8uParamTest : public NppTestBase, public ::testing::WithParamInterface<MaxEvery8uParam> {};

TEST_P(MaxEvery8uParamTest, MaxEvery_8u_C1IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height);
  std::vector<Npp8u> dstData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 54321);

  std::vector<Npp8u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::max_every<Npp8u>(srcData[i], dstData[i]);
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
    status = nppiMaxEvery_8u_C1IR_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMaxEvery_8u_C1IR(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(MaxEvery8u, MaxEvery8uParamTest,
                         ::testing::Values(MaxEvery8uParam{32, 32, false, "32x32_noCtx"},
                                           MaxEvery8uParam{32, 32, true, "32x32_Ctx"},
                                           MaxEvery8uParam{64, 64, false, "64x64_noCtx"},
                                           MaxEvery8uParam{64, 64, true, "64x64_Ctx"}),
                         [](const ::testing::TestParamInfo<MaxEvery8uParam> &info) { return info.param.name; });

// ==================== MaxEvery 16u C1 TEST_P ====================

struct MaxEvery16uParam {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class MaxEvery16uParamTest : public NppTestBase, public ::testing::WithParamInterface<MaxEvery16uParam> {};

TEST_P(MaxEvery16uParamTest, MaxEvery_16u_C1IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcData(width * height);
  std::vector<Npp16u> dstData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 54321);

  std::vector<Npp16u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::max_every<Npp16u>(srcData[i], dstData[i]);
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
    status = nppiMaxEvery_16u_C1IR_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMaxEvery_16u_C1IR(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp16u> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(MaxEvery16u, MaxEvery16uParamTest,
                         ::testing::Values(MaxEvery16uParam{32, 32, false, "32x32_noCtx"},
                                           MaxEvery16uParam{32, 32, true, "32x32_Ctx"},
                                           MaxEvery16uParam{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<MaxEvery16uParam> &info) { return info.param.name; });

// ==================== MaxEvery 32f C1 TEST_P ====================

struct MaxEvery32fParam {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class MaxEvery32fParamTest : public NppTestBase, public ::testing::WithParamInterface<MaxEvery32fParam> {};

TEST_P(MaxEvery32fParamTest, MaxEvery_32f_C1IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> srcData(width * height);
  std::vector<Npp32f> dstData(width * height);
  TestDataGenerator::generateRandom(srcData, -100.0f, 100.0f, 12345);
  TestDataGenerator::generateRandom(dstData, -100.0f, 100.0f, 54321);

  std::vector<Npp32f> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::max_every<Npp32f>(srcData[i], dstData[i]);
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
    status = nppiMaxEvery_32f_C1IR_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMaxEvery_32f_C1IR(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32f> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
}

INSTANTIATE_TEST_SUITE_P(MaxEvery32f, MaxEvery32fParamTest,
                         ::testing::Values(MaxEvery32fParam{32, 32, false, "32x32_noCtx"},
                                           MaxEvery32fParam{32, 32, true, "32x32_Ctx"},
                                           MaxEvery32fParam{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<MaxEvery32fParam> &info) { return info.param.name; });

// ==================== MaxEvery 8u C3 TEST_P ====================

struct MaxEvery8uC3Param {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class MaxEvery8uC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<MaxEvery8uC3Param> {};

TEST_P(MaxEvery8uC3ParamTest, MaxEvery_8u_C3IR) {
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
    expectedData[i] = expect::max_every<Npp8u>(srcData[i], dstData[i]);
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
    status = nppiMaxEvery_8u_C3IR_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMaxEvery_8u_C3IR(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(total);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(MaxEvery8uC3, MaxEvery8uC3ParamTest,
                         ::testing::Values(MaxEvery8uC3Param{32, 32, false, "32x32_noCtx"},
                                           MaxEvery8uC3Param{32, 32, true, "32x32_Ctx"},
                                           MaxEvery8uC3Param{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<MaxEvery8uC3Param> &info) { return info.param.name; });

// ==================== MaxEvery 8u C4 TEST_P ====================

struct MaxEvery8uC4Param {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class MaxEvery8uC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<MaxEvery8uC4Param> {};

TEST_P(MaxEvery8uC4ParamTest, MaxEvery_8u_C4IR) {
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
    expectedData[i] = expect::max_every<Npp8u>(srcData[i], dstData[i]);
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
    status = nppiMaxEvery_8u_C4IR_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMaxEvery_8u_C4IR(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(total);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(MaxEvery8uC4, MaxEvery8uC4ParamTest,
                         ::testing::Values(MaxEvery8uC4Param{32, 32, false, "32x32_noCtx"},
                                           MaxEvery8uC4Param{32, 32, true, "32x32_Ctx"},
                                           MaxEvery8uC4Param{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<MaxEvery8uC4Param> &info) { return info.param.name; });

// ==================== MaxEvery 16u C3 TEST_P ====================

struct MaxEvery16uC3Param {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class MaxEvery16uC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<MaxEvery16uC3Param> {};

TEST_P(MaxEvery16uC3ParamTest, MaxEvery_16u_C3IR) {
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
    expectedData[i] = expect::max_every<Npp16u>(srcData[i], dstData[i]);
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
    status = nppiMaxEvery_16u_C3IR_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMaxEvery_16u_C3IR(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp16u> resultData(total);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(MaxEvery16uC3, MaxEvery16uC3ParamTest,
                         ::testing::Values(MaxEvery16uC3Param{32, 32, false, "32x32_noCtx"},
                                           MaxEvery16uC3Param{32, 32, true, "32x32_Ctx"},
                                           MaxEvery16uC3Param{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<MaxEvery16uC3Param> &info) { return info.param.name; });

// ==================== MaxEvery 16u C4 TEST_P ====================

struct MaxEvery16uC4Param {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class MaxEvery16uC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<MaxEvery16uC4Param> {};

TEST_P(MaxEvery16uC4ParamTest, MaxEvery_16u_C4IR) {
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
    expectedData[i] = expect::max_every<Npp16u>(srcData[i], dstData[i]);
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
    status = nppiMaxEvery_16u_C4IR_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMaxEvery_16u_C4IR(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp16u> resultData(total);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(MaxEvery16uC4, MaxEvery16uC4ParamTest,
                         ::testing::Values(MaxEvery16uC4Param{32, 32, false, "32x32_noCtx"},
                                           MaxEvery16uC4Param{32, 32, true, "32x32_Ctx"},
                                           MaxEvery16uC4Param{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<MaxEvery16uC4Param> &info) { return info.param.name; });

// ==================== MaxEvery 32f C3 TEST_P ====================

struct MaxEvery32fC3Param {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class MaxEvery32fC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<MaxEvery32fC3Param> {};

TEST_P(MaxEvery32fC3ParamTest, MaxEvery_32f_C3IR) {
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
    expectedData[i] = expect::max_every<Npp32f>(srcData[i], dstData[i]);
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
    status = nppiMaxEvery_32f_C3IR_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMaxEvery_32f_C3IR(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32f> resultData(total);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
}

INSTANTIATE_TEST_SUITE_P(MaxEvery32fC3, MaxEvery32fC3ParamTest,
                         ::testing::Values(MaxEvery32fC3Param{32, 32, false, "32x32_noCtx"},
                                           MaxEvery32fC3Param{32, 32, true, "32x32_Ctx"},
                                           MaxEvery32fC3Param{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<MaxEvery32fC3Param> &info) { return info.param.name; });

// ==================== MaxEvery 32f C4 TEST_P ====================

struct MaxEvery32fC4Param {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class MaxEvery32fC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<MaxEvery32fC4Param> {};

TEST_P(MaxEvery32fC4ParamTest, MaxEvery_32f_C4IR) {
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
    expectedData[i] = expect::max_every<Npp32f>(srcData[i], dstData[i]);
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
    status = nppiMaxEvery_32f_C4IR_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMaxEvery_32f_C4IR(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32f> resultData(total);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
}

INSTANTIATE_TEST_SUITE_P(MaxEvery32fC4, MaxEvery32fC4ParamTest,
                         ::testing::Values(MaxEvery32fC4Param{32, 32, false, "32x32_noCtx"},
                                           MaxEvery32fC4Param{32, 32, true, "32x32_Ctx"},
                                           MaxEvery32fC4Param{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<MaxEvery32fC4Param> &info) { return info.param.name; });

// ==================== MaxEvery 16s C3 TEST_P ====================

struct MaxEvery16sC3Param {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class MaxEvery16sC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<MaxEvery16sC3Param> {};

TEST_P(MaxEvery16sC3ParamTest, MaxEvery_16s_C3IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 3;
  const int total = width * height * channels;

  std::vector<Npp16s> srcData(total);
  std::vector<Npp16s> dstData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(-32768), static_cast<Npp16s>(32767), 12345);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp16s>(-32768), static_cast<Npp16s>(32767), 54321);

  std::vector<Npp16s> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::max_every<Npp16s>(srcData[i], dstData[i]);
  }

  NppImageMemory<Npp16s> src(width * channels, height);
  NppImageMemory<Npp16s> dst(width * channels, height);
  src.copyFromHost(srcData);
  dst.copyFromHost(dstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiMaxEvery_16s_C3IR_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMaxEvery_16s_C3IR(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp16s> resultData(total);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(MaxEvery16sC3, MaxEvery16sC3ParamTest,
                         ::testing::Values(MaxEvery16sC3Param{32, 32, false, "32x32_noCtx"},
                                           MaxEvery16sC3Param{32, 32, true, "32x32_Ctx"},
                                           MaxEvery16sC3Param{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<MaxEvery16sC3Param> &info) { return info.param.name; });

// ==================== MaxEvery 16s C4 TEST_P ====================

struct MaxEvery16sC4Param {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class MaxEvery16sC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<MaxEvery16sC4Param> {};

TEST_P(MaxEvery16sC4ParamTest, MaxEvery_16s_C4IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  std::vector<Npp16s> srcData(total);
  std::vector<Npp16s> dstData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(-32768), static_cast<Npp16s>(32767), 12345);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp16s>(-32768), static_cast<Npp16s>(32767), 54321);

  std::vector<Npp16s> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::max_every<Npp16s>(srcData[i], dstData[i]);
  }

  NppImageMemory<Npp16s> src(width * channels, height);
  NppImageMemory<Npp16s> dst(width * channels, height);
  src.copyFromHost(srcData);
  dst.copyFromHost(dstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiMaxEvery_16s_C4IR_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMaxEvery_16s_C4IR(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp16s> resultData(total);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(MaxEvery16sC4, MaxEvery16sC4ParamTest,
                         ::testing::Values(MaxEvery16sC4Param{32, 32, false, "32x32_noCtx"},
                                           MaxEvery16sC4Param{32, 32, true, "32x32_Ctx"},
                                           MaxEvery16sC4Param{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<MaxEvery16sC4Param> &info) { return info.param.name; });
