#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Transpose 8u C1 TEST_P ====================

struct Transpose8uParam {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class Transpose8uParamTest : public NppTestBase, public ::testing::WithParamInterface<Transpose8uParam> {};

TEST_P(Transpose8uParamTest, Transpose_8u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  // Compute expected transposed data
  std::vector<Npp8u> expectedData(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      expectedData[x * height + y] = srcData[y * width + x];
    }
  }

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(height, width);  // Note: swapped dimensions
  src.copyFromHost(srcData);

  NppiSize srcRoi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiTranspose_8u_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), srcRoi, ctx);
  } else {
    status = nppiTranspose_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), srcRoi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(Transpose8u, Transpose8uParamTest,
                         ::testing::Values(Transpose8uParam{32, 32, false, "32x32_noCtx"},
                                           Transpose8uParam{32, 32, true, "32x32_Ctx"},
                                           Transpose8uParam{64, 64, false, "64x64_noCtx"},
                                           Transpose8uParam{64, 32, false, "64x32_noCtx"},
                                           Transpose8uParam{32, 64, false, "32x64_noCtx"}),
                         [](const ::testing::TestParamInfo<Transpose8uParam> &info) { return info.param.name; });

// ==================== Transpose 16u C1 TEST_P ====================

struct Transpose16uParam {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class Transpose16uParamTest : public NppTestBase, public ::testing::WithParamInterface<Transpose16uParam> {};

TEST_P(Transpose16uParamTest, Transpose_16u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);

  std::vector<Npp16u> expectedData(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      expectedData[x * height + y] = srcData[y * width + x];
    }
  }

  NppImageMemory<Npp16u> src(width, height);
  NppImageMemory<Npp16u> dst(height, width);
  src.copyFromHost(srcData);

  NppiSize srcRoi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiTranspose_16u_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), srcRoi, ctx);
  } else {
    status = nppiTranspose_16u_C1R(src.get(), src.step(), dst.get(), dst.step(), srcRoi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp16u> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(Transpose16u, Transpose16uParamTest,
                         ::testing::Values(Transpose16uParam{32, 32, false, "32x32_noCtx"},
                                           Transpose16uParam{32, 32, true, "32x32_Ctx"},
                                           Transpose16uParam{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<Transpose16uParam> &info) { return info.param.name; });

// ==================== Transpose 32f C1 TEST_P ====================

struct Transpose32fParam {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

class Transpose32fParamTest : public NppTestBase, public ::testing::WithParamInterface<Transpose32fParam> {};

TEST_P(Transpose32fParamTest, Transpose_32f_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, -100.0f, 100.0f, 12345);

  std::vector<Npp32f> expectedData(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      expectedData[x * height + y] = srcData[y * width + x];
    }
  }

  NppImageMemory<Npp32f> src(width, height);
  NppImageMemory<Npp32f> dst(height, width);
  src.copyFromHost(srcData);

  NppiSize srcRoi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiTranspose_32f_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), srcRoi, ctx);
  } else {
    status = nppiTranspose_32f_C1R(src.get(), src.step(), dst.get(), dst.step(), srcRoi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32f> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
}

INSTANTIATE_TEST_SUITE_P(Transpose32f, Transpose32fParamTest,
                         ::testing::Values(Transpose32fParam{32, 32, false, "32x32_noCtx"},
                                           Transpose32fParam{32, 32, true, "32x32_Ctx"},
                                           Transpose32fParam{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<Transpose32fParam> &info) { return info.param.name; });
