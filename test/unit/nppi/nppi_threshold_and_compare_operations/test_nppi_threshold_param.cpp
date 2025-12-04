#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Threshold_LTVal 8u C1 TEST_P ====================

struct ThresholdLTVal8uParam {
  int width;
  int height;
  Npp8u threshold;
  Npp8u value;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class ThresholdLTVal8uParamTest : public NppTestBase, public ::testing::WithParamInterface<ThresholdLTVal8uParam> {};

TEST_P(ThresholdLTVal8uParamTest, Threshold_LTVal_8u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp8u threshold = param.threshold;
  const Npp8u value = param.value;

  std::vector<Npp8u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  std::vector<Npp8u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::threshold_lt<Npp8u>(srcData[i], threshold, value);
  }

  NppImageMemory<Npp8u> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiThreshold_LTVal_8u_C1IR_Ctx(src.get(), src.step(), roi, threshold, value, ctx);
    } else {
      status = nppiThreshold_LTVal_8u_C1IR(src.get(), src.step(), roi, threshold, value);
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
      status = nppiThreshold_LTVal_8u_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, threshold, value, ctx);
    } else {
      status = nppiThreshold_LTVal_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, threshold, value);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(ThresholdLTVal8u, ThresholdLTVal8uParamTest,
                         ::testing::Values(ThresholdLTVal8uParam{32, 32, 128, 0, false, false, "32x32_t128_v0_noCtx"},
                                           ThresholdLTVal8uParam{32, 32, 128, 0, true, false, "32x32_t128_v0_Ctx"},
                                           ThresholdLTVal8uParam{32, 32, 100, 50, false, true, "32x32_t100_v50_InPlace"},
                                           ThresholdLTVal8uParam{64, 64, 64, 0, false, false, "64x64_t64_v0_noCtx"}),
                         [](const ::testing::TestParamInfo<ThresholdLTVal8uParam> &info) { return info.param.name; });

// ==================== Threshold_GTVal 8u C1 TEST_P ====================

struct ThresholdGTVal8uParam {
  int width;
  int height;
  Npp8u threshold;
  Npp8u value;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class ThresholdGTVal8uParamTest : public NppTestBase, public ::testing::WithParamInterface<ThresholdGTVal8uParam> {};

TEST_P(ThresholdGTVal8uParamTest, Threshold_GTVal_8u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp8u threshold = param.threshold;
  const Npp8u value = param.value;

  std::vector<Npp8u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  std::vector<Npp8u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::threshold_gt<Npp8u>(srcData[i], threshold, value);
  }

  NppImageMemory<Npp8u> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiThreshold_GTVal_8u_C1IR_Ctx(src.get(), src.step(), roi, threshold, value, ctx);
    } else {
      status = nppiThreshold_GTVal_8u_C1IR(src.get(), src.step(), roi, threshold, value);
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
      status = nppiThreshold_GTVal_8u_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, threshold, value, ctx);
    } else {
      status = nppiThreshold_GTVal_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, threshold, value);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(ThresholdGTVal8u, ThresholdGTVal8uParamTest,
                         ::testing::Values(ThresholdGTVal8uParam{32, 32, 128, 255, false, false, "32x32_t128_v255_noCtx"},
                                           ThresholdGTVal8uParam{32, 32, 128, 255, true, false, "32x32_t128_v255_Ctx"},
                                           ThresholdGTVal8uParam{32, 32, 200, 200, false, true, "32x32_t200_v200_InPlace"},
                                           ThresholdGTVal8uParam{64, 64, 192, 255, false, false, "64x64_t192_v255_noCtx"}),
                         [](const ::testing::TestParamInfo<ThresholdGTVal8uParam> &info) { return info.param.name; });

// ==================== Threshold_LTValGTVal 8u C1 TEST_P ====================

struct ThresholdLTValGTVal8uParam {
  int width;
  int height;
  Npp8u thresholdLT;
  Npp8u valueLT;
  Npp8u thresholdGT;
  Npp8u valueGT;
  bool use_ctx;
  std::string name;
};

class ThresholdLTValGTVal8uParamTest : public NppTestBase, public ::testing::WithParamInterface<ThresholdLTValGTVal8uParam> {};

TEST_P(ThresholdLTValGTVal8uParamTest, Threshold_LTValGTVal_8u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  std::vector<Npp8u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::threshold_ltgt<Npp8u>(srcData[i], param.thresholdLT, param.valueLT, param.thresholdGT, param.valueGT);
  }

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiThreshold_LTValGTVal_8u_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi,
                                                   param.thresholdLT, param.valueLT, param.thresholdGT, param.valueGT, ctx);
  } else {
    status = nppiThreshold_LTValGTVal_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi,
                                               param.thresholdLT, param.valueLT, param.thresholdGT, param.valueGT);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(ThresholdLTValGTVal8u, ThresholdLTValGTVal8uParamTest,
                         ::testing::Values(ThresholdLTValGTVal8uParam{32, 32, 64, 0, 192, 255, false, "32x32_noCtx"},
                                           ThresholdLTValGTVal8uParam{32, 32, 64, 0, 192, 255, true, "32x32_Ctx"},
                                           ThresholdLTValGTVal8uParam{64, 64, 50, 0, 200, 255, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<ThresholdLTValGTVal8uParam> &info) { return info.param.name; });

// ==================== Threshold_LTVal 32f C1 TEST_P ====================

struct ThresholdLTVal32fParam {
  int width;
  int height;
  Npp32f threshold;
  Npp32f value;
  bool use_ctx;
  std::string name;
};

class ThresholdLTVal32fParamTest : public NppTestBase, public ::testing::WithParamInterface<ThresholdLTVal32fParam> {};

TEST_P(ThresholdLTVal32fParamTest, Threshold_LTVal_32f_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp32f threshold = param.threshold;
  const Npp32f value = param.value;

  std::vector<Npp32f> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, -100.0f, 100.0f, 12345);

  std::vector<Npp32f> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::threshold_lt<Npp32f>(srcData[i], threshold, value);
  }

  NppImageMemory<Npp32f> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiThreshold_LTVal_32f_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, threshold, value, ctx);
  } else {
    status = nppiThreshold_LTVal_32f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, threshold, value);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32f> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
}

INSTANTIATE_TEST_SUITE_P(ThresholdLTVal32f, ThresholdLTVal32fParamTest,
                         ::testing::Values(ThresholdLTVal32fParam{32, 32, 0.0f, -100.0f, false, "32x32_t0_vNeg_noCtx"},
                                           ThresholdLTVal32fParam{32, 32, 0.0f, -100.0f, true, "32x32_t0_vNeg_Ctx"},
                                           ThresholdLTVal32fParam{64, 64, -50.0f, -100.0f, false, "64x64_tNeg_vNeg_noCtx"}),
                         [](const ::testing::TestParamInfo<ThresholdLTVal32fParam> &info) { return info.param.name; });

// ==================== Threshold_LTVal 8u C3 TEST_P ====================

struct ThresholdLTVal8uC3Param {
  int width;
  int height;
  Npp8u thresholds[3];
  Npp8u values[3];
  bool use_ctx;
  bool in_place;
  std::string name;
};

class ThresholdLTVal8uC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<ThresholdLTVal8uC3Param> {};

TEST_P(ThresholdLTVal8uC3ParamTest, Threshold_LTVal_8u_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 3;
  const int total = width * height * channels;

  std::vector<Npp8u> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  std::vector<Npp8u> expectedData(total);
  for (int i = 0; i < width * height; i++) {
    for (int c = 0; c < channels; c++) {
      int idx = i * channels + c;
      expectedData[idx] = (srcData[idx] < param.thresholds[c]) ? param.values[c] : srcData[idx];
    }
  }

  NppImageMemory<Npp8u> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiThreshold_LTVal_8u_C3IR_Ctx(src.get(), src.step(), roi, param.thresholds, param.values, ctx);
    } else {
      status = nppiThreshold_LTVal_8u_C3IR(src.get(), src.step(), roi, param.thresholds, param.values);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp8u> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiThreshold_LTVal_8u_C3R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, param.thresholds, param.values, ctx);
    } else {
      status = nppiThreshold_LTVal_8u_C3R(src.get(), src.step(), dst.get(), dst.step(), roi, param.thresholds, param.values);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(ThresholdLTVal8uC3, ThresholdLTVal8uC3ParamTest,
                         ::testing::Values(ThresholdLTVal8uC3Param{32, 32, {128, 64, 192}, {0, 0, 0}, false, false, "32x32_noCtx"},
                                           ThresholdLTVal8uC3Param{32, 32, {128, 64, 192}, {0, 0, 0}, true, false, "32x32_Ctx"},
                                           ThresholdLTVal8uC3Param{32, 32, {100, 100, 100}, {50, 50, 50}, false, true, "32x32_InPlace"},
                                           ThresholdLTVal8uC3Param{32, 32, {100, 100, 100}, {50, 50, 50}, true, true, "32x32_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<ThresholdLTVal8uC3Param> &info) { return info.param.name; });

// ==================== Threshold_GTVal 8u C3 TEST_P ====================

class ThresholdGTVal8uC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<ThresholdLTVal8uC3Param> {};

TEST_P(ThresholdGTVal8uC3ParamTest, Threshold_GTVal_8u_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 3;
  const int total = width * height * channels;

  std::vector<Npp8u> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  std::vector<Npp8u> expectedData(total);
  for (int i = 0; i < width * height; i++) {
    for (int c = 0; c < channels; c++) {
      int idx = i * channels + c;
      expectedData[idx] = (srcData[idx] > param.thresholds[c]) ? param.values[c] : srcData[idx];
    }
  }

  NppImageMemory<Npp8u> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiThreshold_GTVal_8u_C3IR_Ctx(src.get(), src.step(), roi, param.thresholds, param.values, ctx);
    } else {
      status = nppiThreshold_GTVal_8u_C3IR(src.get(), src.step(), roi, param.thresholds, param.values);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp8u> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiThreshold_GTVal_8u_C3R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, param.thresholds, param.values, ctx);
    } else {
      status = nppiThreshold_GTVal_8u_C3R(src.get(), src.step(), dst.get(), dst.step(), roi, param.thresholds, param.values);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(ThresholdGTVal8uC3, ThresholdGTVal8uC3ParamTest,
                         ::testing::Values(ThresholdLTVal8uC3Param{32, 32, {128, 64, 192}, {255, 255, 255}, false, false, "32x32_noCtx"},
                                           ThresholdLTVal8uC3Param{32, 32, {128, 64, 192}, {255, 255, 255}, true, false, "32x32_Ctx"},
                                           ThresholdLTVal8uC3Param{32, 32, {200, 200, 200}, {200, 200, 200}, false, true, "32x32_InPlace"},
                                           ThresholdLTVal8uC3Param{32, 32, {200, 200, 200}, {200, 200, 200}, true, true, "32x32_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<ThresholdLTVal8uC3Param> &info) { return info.param.name; });
