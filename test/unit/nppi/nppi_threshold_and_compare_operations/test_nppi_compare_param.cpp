#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Compare 8u C1 TEST_P ====================

struct Compare8uParam {
  int width;
  int height;
  NppCmpOp cmpOp;
  bool use_ctx;
  std::string name;
};

class Compare8uParamTest : public NppTestBase, public ::testing::WithParamInterface<Compare8uParam> {};

TEST_P(Compare8uParamTest, Compare_8u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const NppCmpOp cmpOp = param.cmpOp;

  std::vector<Npp8u> src1Data(width * height);
  std::vector<Npp8u> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 54321);

  std::vector<Npp8u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    switch (cmpOp) {
    case NPP_CMP_LESS:
      expectedData[i] = expect::compare_lt<Npp8u>(src1Data[i], src2Data[i]);
      break;
    case NPP_CMP_LESS_EQ:
      expectedData[i] = expect::compare_le<Npp8u>(src1Data[i], src2Data[i]);
      break;
    case NPP_CMP_EQ:
      expectedData[i] = expect::compare_eq<Npp8u>(src1Data[i], src2Data[i]);
      break;
    case NPP_CMP_GREATER_EQ:
      expectedData[i] = expect::compare_ge<Npp8u>(src1Data[i], src2Data[i]);
      break;
    case NPP_CMP_GREATER:
      expectedData[i] = expect::compare_gt<Npp8u>(src1Data[i], src2Data[i]);
      break;
    default:
      FAIL() << "Unknown comparison operator";
    }
  }

  NppImageMemory<Npp8u> src1(width, height);
  NppImageMemory<Npp8u> src2(width, height);
  NppImageMemory<Npp8u> dst(width, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiCompare_8u_C1R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, cmpOp, ctx);
  } else {
    status = nppiCompare_8u_C1R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, cmpOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(Compare8u, Compare8uParamTest,
                         ::testing::Values(Compare8uParam{32, 32, NPP_CMP_LESS, false, "32x32_LT_noCtx"},
                                           Compare8uParam{32, 32, NPP_CMP_LESS, true, "32x32_LT_Ctx"},
                                           Compare8uParam{32, 32, NPP_CMP_LESS_EQ, false, "32x32_LE_noCtx"},
                                           Compare8uParam{32, 32, NPP_CMP_EQ, false, "32x32_EQ_noCtx"},
                                           Compare8uParam{32, 32, NPP_CMP_GREATER_EQ, false, "32x32_GE_noCtx"},
                                           Compare8uParam{32, 32, NPP_CMP_GREATER, false, "32x32_GT_noCtx"},
                                           Compare8uParam{64, 64, NPP_CMP_LESS, false, "64x64_LT_noCtx"}),
                         [](const ::testing::TestParamInfo<Compare8uParam> &info) { return info.param.name; });

// ==================== Compare 16u C1 TEST_P ====================

struct Compare16uParam {
  int width;
  int height;
  NppCmpOp cmpOp;
  bool use_ctx;
  std::string name;
};

class Compare16uParamTest : public NppTestBase, public ::testing::WithParamInterface<Compare16uParam> {};

TEST_P(Compare16uParamTest, Compare_16u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const NppCmpOp cmpOp = param.cmpOp;

  std::vector<Npp16u> src1Data(width * height);
  std::vector<Npp16u> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 54321);

  std::vector<Npp8u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    switch (cmpOp) {
    case NPP_CMP_LESS:
      expectedData[i] = expect::compare_lt<Npp16u>(src1Data[i], src2Data[i]);
      break;
    case NPP_CMP_LESS_EQ:
      expectedData[i] = expect::compare_le<Npp16u>(src1Data[i], src2Data[i]);
      break;
    case NPP_CMP_EQ:
      expectedData[i] = expect::compare_eq<Npp16u>(src1Data[i], src2Data[i]);
      break;
    case NPP_CMP_GREATER_EQ:
      expectedData[i] = expect::compare_ge<Npp16u>(src1Data[i], src2Data[i]);
      break;
    case NPP_CMP_GREATER:
      expectedData[i] = expect::compare_gt<Npp16u>(src1Data[i], src2Data[i]);
      break;
    default:
      FAIL() << "Unknown comparison operator";
    }
  }

  NppImageMemory<Npp16u> src1(width, height);
  NppImageMemory<Npp16u> src2(width, height);
  NppImageMemory<Npp8u> dst(width, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiCompare_16u_C1R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, cmpOp, ctx);
  } else {
    status = nppiCompare_16u_C1R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, cmpOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(Compare16u, Compare16uParamTest,
                         ::testing::Values(Compare16uParam{32, 32, NPP_CMP_LESS, false, "32x32_LT_noCtx"},
                                           Compare16uParam{32, 32, NPP_CMP_EQ, false, "32x32_EQ_noCtx"},
                                           Compare16uParam{32, 32, NPP_CMP_GREATER, false, "32x32_GT_noCtx"}),
                         [](const ::testing::TestParamInfo<Compare16uParam> &info) { return info.param.name; });

// ==================== Compare 32f C1 TEST_P ====================

struct Compare32fParam {
  int width;
  int height;
  NppCmpOp cmpOp;
  bool use_ctx;
  std::string name;
};

class Compare32fParamTest : public NppTestBase, public ::testing::WithParamInterface<Compare32fParam> {};

TEST_P(Compare32fParamTest, Compare_32f_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const NppCmpOp cmpOp = param.cmpOp;

  std::vector<Npp32f> src1Data(width * height);
  std::vector<Npp32f> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, -100.0f, 100.0f, 12345);
  TestDataGenerator::generateRandom(src2Data, -100.0f, 100.0f, 54321);

  std::vector<Npp8u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    switch (cmpOp) {
    case NPP_CMP_LESS:
      expectedData[i] = expect::compare_lt<Npp32f>(src1Data[i], src2Data[i]);
      break;
    case NPP_CMP_LESS_EQ:
      expectedData[i] = expect::compare_le<Npp32f>(src1Data[i], src2Data[i]);
      break;
    case NPP_CMP_EQ:
      expectedData[i] = expect::compare_eq<Npp32f>(src1Data[i], src2Data[i]);
      break;
    case NPP_CMP_GREATER_EQ:
      expectedData[i] = expect::compare_ge<Npp32f>(src1Data[i], src2Data[i]);
      break;
    case NPP_CMP_GREATER:
      expectedData[i] = expect::compare_gt<Npp32f>(src1Data[i], src2Data[i]);
      break;
    default:
      FAIL() << "Unknown comparison operator";
    }
  }

  NppImageMemory<Npp32f> src1(width, height);
  NppImageMemory<Npp32f> src2(width, height);
  NppImageMemory<Npp8u> dst(width, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiCompare_32f_C1R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, cmpOp, ctx);
  } else {
    status = nppiCompare_32f_C1R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, cmpOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(Compare32f, Compare32fParamTest,
                         ::testing::Values(Compare32fParam{32, 32, NPP_CMP_LESS, false, "32x32_LT_noCtx"},
                                           Compare32fParam{32, 32, NPP_CMP_EQ, false, "32x32_EQ_noCtx"},
                                           Compare32fParam{32, 32, NPP_CMP_GREATER, false, "32x32_GT_noCtx"}),
                         [](const ::testing::TestParamInfo<Compare32fParam> &info) { return info.param.name; });

// ==================== Compare 8u C3 TEST_P ====================

class Compare8uC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<Compare8uParam> {};

TEST_P(Compare8uC3ParamTest, Compare_8u_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 3;
  const NppCmpOp cmpOp = param.cmpOp;

  std::vector<Npp8u> src1Data(width * height * channels);
  std::vector<Npp8u> src2Data(width * height * channels);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 54321);

  std::vector<Npp8u> expectedData(width * height);
  for (int i = 0; i < width * height; i++) {
    bool allMatch = true;
    for (int c = 0; c < channels; c++) {
      int idx = i * channels + c;
      bool match = false;
      switch (cmpOp) {
      case NPP_CMP_LESS:
        match = src1Data[idx] < src2Data[idx];
        break;
      case NPP_CMP_LESS_EQ:
        match = src1Data[idx] <= src2Data[idx];
        break;
      case NPP_CMP_EQ:
        match = src1Data[idx] == src2Data[idx];
        break;
      case NPP_CMP_GREATER_EQ:
        match = src1Data[idx] >= src2Data[idx];
        break;
      case NPP_CMP_GREATER:
        match = src1Data[idx] > src2Data[idx];
        break;
      default:
        break;
      }
      allMatch = allMatch && match;
    }
    expectedData[i] = allMatch ? 255 : 0;
  }

  NppImageMemory<Npp8u> src1(width * channels, height);
  NppImageMemory<Npp8u> src2(width * channels, height);
  NppImageMemory<Npp8u> dst(width, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiCompare_8u_C3R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, cmpOp, ctx);
  } else {
    status = nppiCompare_8u_C3R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, cmpOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(Compare8uC3, Compare8uC3ParamTest,
                         ::testing::Values(Compare8uParam{32, 32, NPP_CMP_LESS, false, "32x32_LT_noCtx"},
                                           Compare8uParam{32, 32, NPP_CMP_LESS, true, "32x32_LT_Ctx"},
                                           Compare8uParam{32, 32, NPP_CMP_EQ, false, "32x32_EQ_noCtx"},
                                           Compare8uParam{32, 32, NPP_CMP_GREATER, false, "32x32_GT_noCtx"}),
                         [](const ::testing::TestParamInfo<Compare8uParam> &info) { return info.param.name; });

// ==================== Compare 8u C4 TEST_P ====================

class Compare8uC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<Compare8uParam> {};

TEST_P(Compare8uC4ParamTest, Compare_8u_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const NppCmpOp cmpOp = param.cmpOp;

  std::vector<Npp8u> src1Data(width * height * channels);
  std::vector<Npp8u> src2Data(width * height * channels);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 54321);

  std::vector<Npp8u> expectedData(width * height);
  for (int i = 0; i < width * height; i++) {
    bool allMatch = true;
    for (int c = 0; c < channels; c++) {
      int idx = i * channels + c;
      bool match = false;
      switch (cmpOp) {
      case NPP_CMP_LESS:
        match = src1Data[idx] < src2Data[idx];
        break;
      case NPP_CMP_LESS_EQ:
        match = src1Data[idx] <= src2Data[idx];
        break;
      case NPP_CMP_EQ:
        match = src1Data[idx] == src2Data[idx];
        break;
      case NPP_CMP_GREATER_EQ:
        match = src1Data[idx] >= src2Data[idx];
        break;
      case NPP_CMP_GREATER:
        match = src1Data[idx] > src2Data[idx];
        break;
      default:
        break;
      }
      allMatch = allMatch && match;
    }
    expectedData[i] = allMatch ? 255 : 0;
  }

  NppImageMemory<Npp8u> src1(width * channels, height);
  NppImageMemory<Npp8u> src2(width * channels, height);
  NppImageMemory<Npp8u> dst(width, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiCompare_8u_C4R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, cmpOp, ctx);
  } else {
    status = nppiCompare_8u_C4R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, cmpOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(Compare8uC4, Compare8uC4ParamTest,
                         ::testing::Values(Compare8uParam{32, 32, NPP_CMP_LESS, false, "32x32_LT_noCtx"},
                                           Compare8uParam{32, 32, NPP_CMP_LESS, true, "32x32_LT_Ctx"},
                                           Compare8uParam{32, 32, NPP_CMP_EQ, false, "32x32_EQ_noCtx"},
                                           Compare8uParam{32, 32, NPP_CMP_GREATER, false, "32x32_GT_noCtx"}),
                         [](const ::testing::TestParamInfo<Compare8uParam> &info) { return info.param.name; });

// ==================== Compare 16u C3 TEST_P ====================

class Compare16uC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<Compare16uParam> {};

TEST_P(Compare16uC3ParamTest, Compare_16u_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 3;
  const NppCmpOp cmpOp = param.cmpOp;

  std::vector<Npp16u> src1Data(width * height * channels);
  std::vector<Npp16u> src2Data(width * height * channels);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 54321);

  std::vector<Npp8u> expectedData(width * height);
  for (int i = 0; i < width * height; i++) {
    bool allMatch = true;
    for (int c = 0; c < channels; c++) {
      int idx = i * channels + c;
      bool match = false;
      switch (cmpOp) {
      case NPP_CMP_LESS:
        match = src1Data[idx] < src2Data[idx];
        break;
      case NPP_CMP_LESS_EQ:
        match = src1Data[idx] <= src2Data[idx];
        break;
      case NPP_CMP_EQ:
        match = src1Data[idx] == src2Data[idx];
        break;
      case NPP_CMP_GREATER_EQ:
        match = src1Data[idx] >= src2Data[idx];
        break;
      case NPP_CMP_GREATER:
        match = src1Data[idx] > src2Data[idx];
        break;
      default:
        break;
      }
      allMatch = allMatch && match;
    }
    expectedData[i] = allMatch ? 255 : 0;
  }

  NppImageMemory<Npp16u> src1(width * channels, height);
  NppImageMemory<Npp16u> src2(width * channels, height);
  NppImageMemory<Npp8u> dst(width, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiCompare_16u_C3R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, cmpOp, ctx);
  } else {
    status = nppiCompare_16u_C3R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, cmpOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(Compare16uC3, Compare16uC3ParamTest,
                         ::testing::Values(Compare16uParam{32, 32, NPP_CMP_LESS, false, "32x32_LT_noCtx"},
                                           Compare16uParam{32, 32, NPP_CMP_EQ, false, "32x32_EQ_noCtx"},
                                           Compare16uParam{32, 32, NPP_CMP_GREATER, false, "32x32_GT_noCtx"}),
                         [](const ::testing::TestParamInfo<Compare16uParam> &info) { return info.param.name; });

// ==================== Compare 16u C4 TEST_P ====================

class Compare16uC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<Compare16uParam> {};

TEST_P(Compare16uC4ParamTest, Compare_16u_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const NppCmpOp cmpOp = param.cmpOp;

  std::vector<Npp16u> src1Data(width * height * channels);
  std::vector<Npp16u> src2Data(width * height * channels);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 54321);

  std::vector<Npp8u> expectedData(width * height);
  for (int i = 0; i < width * height; i++) {
    bool allMatch = true;
    for (int c = 0; c < channels; c++) {
      int idx = i * channels + c;
      bool match = false;
      switch (cmpOp) {
      case NPP_CMP_LESS:
        match = src1Data[idx] < src2Data[idx];
        break;
      case NPP_CMP_LESS_EQ:
        match = src1Data[idx] <= src2Data[idx];
        break;
      case NPP_CMP_EQ:
        match = src1Data[idx] == src2Data[idx];
        break;
      case NPP_CMP_GREATER_EQ:
        match = src1Data[idx] >= src2Data[idx];
        break;
      case NPP_CMP_GREATER:
        match = src1Data[idx] > src2Data[idx];
        break;
      default:
        break;
      }
      allMatch = allMatch && match;
    }
    expectedData[i] = allMatch ? 255 : 0;
  }

  NppImageMemory<Npp16u> src1(width * channels, height);
  NppImageMemory<Npp16u> src2(width * channels, height);
  NppImageMemory<Npp8u> dst(width, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiCompare_16u_C4R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, cmpOp, ctx);
  } else {
    status = nppiCompare_16u_C4R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, cmpOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(Compare16uC4, Compare16uC4ParamTest,
                         ::testing::Values(Compare16uParam{32, 32, NPP_CMP_LESS, false, "32x32_LT_noCtx"},
                                           Compare16uParam{32, 32, NPP_CMP_EQ, false, "32x32_EQ_noCtx"},
                                           Compare16uParam{32, 32, NPP_CMP_GREATER, false, "32x32_GT_noCtx"}),
                         [](const ::testing::TestParamInfo<Compare16uParam> &info) { return info.param.name; });

// ==================== Compare 32f C3 TEST_P ====================

class Compare32fC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<Compare32fParam> {};

TEST_P(Compare32fC3ParamTest, Compare_32f_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 3;
  const NppCmpOp cmpOp = param.cmpOp;

  std::vector<Npp32f> src1Data(width * height * channels);
  std::vector<Npp32f> src2Data(width * height * channels);
  TestDataGenerator::generateRandom(src1Data, -100.0f, 100.0f, 12345);
  TestDataGenerator::generateRandom(src2Data, -100.0f, 100.0f, 54321);

  std::vector<Npp8u> expectedData(width * height);
  for (int i = 0; i < width * height; i++) {
    bool allMatch = true;
    for (int c = 0; c < channels; c++) {
      int idx = i * channels + c;
      bool match = false;
      switch (cmpOp) {
      case NPP_CMP_LESS:
        match = src1Data[idx] < src2Data[idx];
        break;
      case NPP_CMP_LESS_EQ:
        match = src1Data[idx] <= src2Data[idx];
        break;
      case NPP_CMP_EQ:
        match = src1Data[idx] == src2Data[idx];
        break;
      case NPP_CMP_GREATER_EQ:
        match = src1Data[idx] >= src2Data[idx];
        break;
      case NPP_CMP_GREATER:
        match = src1Data[idx] > src2Data[idx];
        break;
      default:
        break;
      }
      allMatch = allMatch && match;
    }
    expectedData[i] = allMatch ? 255 : 0;
  }

  NppImageMemory<Npp32f> src1(width * channels, height);
  NppImageMemory<Npp32f> src2(width * channels, height);
  NppImageMemory<Npp8u> dst(width, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiCompare_32f_C3R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, cmpOp, ctx);
  } else {
    status = nppiCompare_32f_C3R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, cmpOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(Compare32fC3, Compare32fC3ParamTest,
                         ::testing::Values(Compare32fParam{32, 32, NPP_CMP_LESS, false, "32x32_LT_noCtx"},
                                           Compare32fParam{32, 32, NPP_CMP_EQ, false, "32x32_EQ_noCtx"},
                                           Compare32fParam{32, 32, NPP_CMP_GREATER, false, "32x32_GT_noCtx"}),
                         [](const ::testing::TestParamInfo<Compare32fParam> &info) { return info.param.name; });

// ==================== Compare 32f C4 TEST_P ====================

class Compare32fC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<Compare32fParam> {};

TEST_P(Compare32fC4ParamTest, Compare_32f_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const NppCmpOp cmpOp = param.cmpOp;

  std::vector<Npp32f> src1Data(width * height * channels);
  std::vector<Npp32f> src2Data(width * height * channels);
  TestDataGenerator::generateRandom(src1Data, -100.0f, 100.0f, 12345);
  TestDataGenerator::generateRandom(src2Data, -100.0f, 100.0f, 54321);

  std::vector<Npp8u> expectedData(width * height);
  for (int i = 0; i < width * height; i++) {
    bool allMatch = true;
    for (int c = 0; c < channels; c++) {
      int idx = i * channels + c;
      bool match = false;
      switch (cmpOp) {
      case NPP_CMP_LESS:
        match = src1Data[idx] < src2Data[idx];
        break;
      case NPP_CMP_LESS_EQ:
        match = src1Data[idx] <= src2Data[idx];
        break;
      case NPP_CMP_EQ:
        match = src1Data[idx] == src2Data[idx];
        break;
      case NPP_CMP_GREATER_EQ:
        match = src1Data[idx] >= src2Data[idx];
        break;
      case NPP_CMP_GREATER:
        match = src1Data[idx] > src2Data[idx];
        break;
      default:
        break;
      }
      allMatch = allMatch && match;
    }
    expectedData[i] = allMatch ? 255 : 0;
  }

  NppImageMemory<Npp32f> src1(width * channels, height);
  NppImageMemory<Npp32f> src2(width * channels, height);
  NppImageMemory<Npp8u> dst(width, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiCompare_32f_C4R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, cmpOp, ctx);
  } else {
    status = nppiCompare_32f_C4R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, cmpOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(Compare32fC4, Compare32fC4ParamTest,
                         ::testing::Values(Compare32fParam{32, 32, NPP_CMP_LESS, false, "32x32_LT_noCtx"},
                                           Compare32fParam{32, 32, NPP_CMP_EQ, false, "32x32_EQ_noCtx"},
                                           Compare32fParam{32, 32, NPP_CMP_GREATER, false, "32x32_GT_noCtx"}),
                         [](const ::testing::TestParamInfo<Compare32fParam> &info) { return info.param.name; });
