#include "npp_test_base.h"

using namespace npp_functional_test;

// ==================== SwapChannels 8u C3 TEST_P ====================

struct SwapChannels8uC3Param {
  int width;
  int height;
  int order[3];
  bool use_ctx;
  std::string name;
};

class SwapChannels8uC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<SwapChannels8uC3Param> {};

TEST_P(SwapChannels8uC3ParamTest, SwapChannels_8u_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height * 3);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  // Compute expected swapped data
  std::vector<Npp8u> expectedData(width * height * 3);
  for (int i = 0; i < width * height; i++) {
    expectedData[i * 3 + 0] = srcData[i * 3 + param.order[0]];
    expectedData[i * 3 + 1] = srcData[i * 3 + param.order[1]];
    expectedData[i * 3 + 2] = srcData[i * 3 + param.order[2]];
  }

  NppImageMemory<Npp8u> src(width * 3, height);
  NppImageMemory<Npp8u> dst(width * 3, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiSwapChannels_8u_C3R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, param.order, ctx);
  } else {
    status = nppiSwapChannels_8u_C3R(src.get(), src.step(), dst.get(), dst.step(), roi, param.order);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(width * height * 3);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(SwapChannels8uC3, SwapChannels8uC3ParamTest,
                         ::testing::Values(SwapChannels8uC3Param{32, 32, {2, 1, 0}, false, "32x32_BGR_noCtx"},
                                           SwapChannels8uC3Param{32, 32, {2, 1, 0}, true, "32x32_BGR_Ctx"},
                                           SwapChannels8uC3Param{64, 64, {1, 2, 0}, false, "64x64_GBR_noCtx"},
                                           SwapChannels8uC3Param{64, 64, {0, 2, 1}, false, "64x64_RBG_noCtx"}),
                         [](const ::testing::TestParamInfo<SwapChannels8uC3Param> &info) { return info.param.name; });

// ==================== SwapChannels 8u C4 TEST_P ====================

struct SwapChannels8uC4Param {
  int width;
  int height;
  int order[4];
  bool use_ctx;
  std::string name;
};

class SwapChannels8uC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<SwapChannels8uC4Param> {};

TEST_P(SwapChannels8uC4ParamTest, SwapChannels_8u_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height * 4);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  // Compute expected swapped data
  std::vector<Npp8u> expectedData(width * height * 4);
  for (int i = 0; i < width * height; i++) {
    expectedData[i * 4 + 0] = srcData[i * 4 + param.order[0]];
    expectedData[i * 4 + 1] = srcData[i * 4 + param.order[1]];
    expectedData[i * 4 + 2] = srcData[i * 4 + param.order[2]];
    expectedData[i * 4 + 3] = srcData[i * 4 + param.order[3]];
  }

  NppImageMemory<Npp8u> src(width * 4, height);
  NppImageMemory<Npp8u> dst(width * 4, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiSwapChannels_8u_C4R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, param.order, ctx);
  } else {
    status = nppiSwapChannels_8u_C4R(src.get(), src.step(), dst.get(), dst.step(), roi, param.order);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(width * height * 4);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(SwapChannels8uC4, SwapChannels8uC4ParamTest,
                         ::testing::Values(SwapChannels8uC4Param{32, 32, {2, 1, 0, 3}, false, "32x32_BGRA_noCtx"},
                                           SwapChannels8uC4Param{32, 32, {2, 1, 0, 3}, true, "32x32_BGRA_Ctx"},
                                           SwapChannels8uC4Param{64, 64, {3, 2, 1, 0}, false, "64x64_ABGR_noCtx"}),
                         [](const ::testing::TestParamInfo<SwapChannels8uC4Param> &info) { return info.param.name; });

// ==================== SwapChannels 32f C3 TEST_P ====================

struct SwapChannels32fC3Param {
  int width;
  int height;
  int order[3];
  bool use_ctx;
  std::string name;
};

class SwapChannels32fC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<SwapChannels32fC3Param> {};

TEST_P(SwapChannels32fC3ParamTest, SwapChannels_32f_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> srcData(width * height * 3);
  TestDataGenerator::generateRandom(srcData, -100.0f, 100.0f, 12345);

  std::vector<Npp32f> expectedData(width * height * 3);
  for (int i = 0; i < width * height; i++) {
    expectedData[i * 3 + 0] = srcData[i * 3 + param.order[0]];
    expectedData[i * 3 + 1] = srcData[i * 3 + param.order[1]];
    expectedData[i * 3 + 2] = srcData[i * 3 + param.order[2]];
  }

  NppImageMemory<Npp32f> src(width * 3, height);
  NppImageMemory<Npp32f> dst(width * 3, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiSwapChannels_32f_C3R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, param.order, ctx);
  } else {
    status = nppiSwapChannels_32f_C3R(src.get(), src.step(), dst.get(), dst.step(), roi, param.order);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32f> resultData(width * height * 3);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
}

INSTANTIATE_TEST_SUITE_P(SwapChannels32fC3, SwapChannels32fC3ParamTest,
                         ::testing::Values(SwapChannels32fC3Param{32, 32, {2, 1, 0}, false, "32x32_BGR_noCtx"},
                                           SwapChannels32fC3Param{32, 32, {2, 1, 0}, true, "32x32_BGR_Ctx"},
                                           SwapChannels32fC3Param{64, 64, {1, 2, 0}, false, "64x64_GBR_noCtx"}),
                         [](const ::testing::TestParamInfo<SwapChannels32fC3Param> &info) { return info.param.name; });
