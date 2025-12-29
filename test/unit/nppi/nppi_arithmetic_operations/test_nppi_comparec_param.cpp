#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== CompareC 8u C1 TEST_P ====================

struct CompareC8uParam {
  int width;
  int height;
  Npp8u constant;
  NppCmpOp cmpOp;
  bool use_ctx;
  std::string name;
};

class CompareC8uParamTest : public NppTestBase, public ::testing::WithParamInterface<CompareC8uParam> {};

TEST_P(CompareC8uParamTest, CompareC_8u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp8u constant = param.constant;
  const NppCmpOp cmpOp = param.cmpOp;

  std::vector<Npp8u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  std::vector<Npp8u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    switch (cmpOp) {
    case NPP_CMP_LESS:
      expectedData[i] = expect::compare_lt<Npp8u>(srcData[i], constant);
      break;
    case NPP_CMP_LESS_EQ:
      expectedData[i] = expect::compare_le<Npp8u>(srcData[i], constant);
      break;
    case NPP_CMP_EQ:
      expectedData[i] = expect::compare_eq<Npp8u>(srcData[i], constant);
      break;
    case NPP_CMP_GREATER_EQ:
      expectedData[i] = expect::compare_ge<Npp8u>(srcData[i], constant);
      break;
    case NPP_CMP_GREATER:
      expectedData[i] = expect::compare_gt<Npp8u>(srcData[i], constant);
      break;
    default:
      FAIL() << "Unknown comparison operator";
    }
  }

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiCompareC_8u_C1R_Ctx(src.get(), src.step(), constant, dst.get(), dst.step(), roi, cmpOp, ctx);
  } else {
    status = nppiCompareC_8u_C1R(src.get(), src.step(), constant, dst.get(), dst.step(), roi, cmpOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(CompareC8u, CompareC8uParamTest,
                         ::testing::Values(CompareC8uParam{32, 32, 128, NPP_CMP_LESS, false, "32x32_c128_LT_noCtx"},
                                           CompareC8uParam{32, 32, 128, NPP_CMP_LESS, true, "32x32_c128_LT_Ctx"},
                                           CompareC8uParam{32, 32, 100, NPP_CMP_LESS_EQ, false, "32x32_c100_LE_noCtx"},
                                           CompareC8uParam{32, 32, 100, NPP_CMP_EQ, false, "32x32_c100_EQ_noCtx"},
                                           CompareC8uParam{32, 32, 50, NPP_CMP_GREATER_EQ, false, "32x32_c50_GE_noCtx"},
                                           CompareC8uParam{32, 32, 50, NPP_CMP_GREATER, false, "32x32_c50_GT_noCtx"},
                                           CompareC8uParam{64, 64, 128, NPP_CMP_LESS, false, "64x64_c128_LT_noCtx"}),
                         [](const ::testing::TestParamInfo<CompareC8uParam> &info) { return info.param.name; });

// ==================== CompareC 16u C1 TEST_P ====================

struct CompareC16uParam {
  int width;
  int height;
  Npp16u constant;
  NppCmpOp cmpOp;
  bool use_ctx;
  std::string name;
};

class CompareC16uParamTest : public NppTestBase, public ::testing::WithParamInterface<CompareC16uParam> {};

TEST_P(CompareC16uParamTest, CompareC_16u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp16u constant = param.constant;
  const NppCmpOp cmpOp = param.cmpOp;

  std::vector<Npp16u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);

  std::vector<Npp8u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    switch (cmpOp) {
    case NPP_CMP_LESS:
      expectedData[i] = expect::compare_lt<Npp16u>(srcData[i], constant);
      break;
    case NPP_CMP_EQ:
      expectedData[i] = expect::compare_eq<Npp16u>(srcData[i], constant);
      break;
    case NPP_CMP_GREATER:
      expectedData[i] = expect::compare_gt<Npp16u>(srcData[i], constant);
      break;
    default:
      FAIL() << "Unknown comparison operator";
    }
  }

  NppImageMemory<Npp16u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiCompareC_16u_C1R_Ctx(src.get(), src.step(), constant, dst.get(), dst.step(), roi, cmpOp, ctx);
  } else {
    status = nppiCompareC_16u_C1R(src.get(), src.step(), constant, dst.get(), dst.step(), roi, cmpOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(
    CompareC16u, CompareC16uParamTest,
    ::testing::Values(CompareC16uParam{32, 32, 32768, NPP_CMP_LESS, false, "32x32_c32768_LT_noCtx"},
                      CompareC16uParam{32, 32, 32768, NPP_CMP_EQ, false, "32x32_c32768_EQ_noCtx"},
                      CompareC16uParam{32, 32, 32768, NPP_CMP_GREATER, false, "32x32_c32768_GT_noCtx"}),
    [](const ::testing::TestParamInfo<CompareC16uParam> &info) { return info.param.name; });

// ==================== CompareC 32f C1 TEST_P ====================

struct CompareC32fParam {
  int width;
  int height;
  Npp32f constant;
  NppCmpOp cmpOp;
  bool use_ctx;
  std::string name;
};

class CompareC32fParamTest : public NppTestBase, public ::testing::WithParamInterface<CompareC32fParam> {};

TEST_P(CompareC32fParamTest, CompareC_32f_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp32f constant = param.constant;
  const NppCmpOp cmpOp = param.cmpOp;

  std::vector<Npp32f> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, -100.0f, 100.0f, 12345);

  std::vector<Npp8u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    switch (cmpOp) {
    case NPP_CMP_LESS:
      expectedData[i] = expect::compare_lt<Npp32f>(srcData[i], constant);
      break;
    case NPP_CMP_EQ:
      expectedData[i] = expect::compare_eq<Npp32f>(srcData[i], constant);
      break;
    case NPP_CMP_GREATER:
      expectedData[i] = expect::compare_gt<Npp32f>(srcData[i], constant);
      break;
    default:
      FAIL() << "Unknown comparison operator";
    }
  }

  NppImageMemory<Npp32f> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiCompareC_32f_C1R_Ctx(src.get(), src.step(), constant, dst.get(), dst.step(), roi, cmpOp, ctx);
  } else {
    status = nppiCompareC_32f_C1R(src.get(), src.step(), constant, dst.get(), dst.step(), roi, cmpOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(CompareC32f, CompareC32fParamTest,
                         ::testing::Values(CompareC32fParam{32, 32, 0.0f, NPP_CMP_LESS, false, "32x32_c0_LT_noCtx"},
                                           CompareC32fParam{32, 32, 0.0f, NPP_CMP_EQ, false, "32x32_c0_EQ_noCtx"},
                                           CompareC32fParam{32, 32, 0.0f, NPP_CMP_GREATER, false, "32x32_c0_GT_noCtx"}),
                         [](const ::testing::TestParamInfo<CompareC32fParam> &info) { return info.param.name; });
