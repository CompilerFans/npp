#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Mirror 8u C1 TEST_P ====================

struct Mirror8uParam {
  int width;
  int height;
  NppiAxis flip;
  bool use_ctx;
  std::string name;
};

class Mirror8uParamTest : public NppTestBase, public ::testing::WithParamInterface<Mirror8uParam> {};

TEST_P(Mirror8uParamTest, Mirror_8u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const NppiAxis flip = param.flip;

  std::vector<Npp8u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  // Compute expected mirrored data
  std::vector<Npp8u> expectedData(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int srcX = x, srcY = y;
      if (flip == NPP_HORIZONTAL_AXIS || flip == NPP_BOTH_AXIS) {
        srcY = height - 1 - y;
      }
      if (flip == NPP_VERTICAL_AXIS || flip == NPP_BOTH_AXIS) {
        srcX = width - 1 - x;
      }
      expectedData[y * width + x] = srcData[srcY * width + srcX];
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
    status = nppiMirror_8u_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, flip, ctx);
  } else {
    status = nppiMirror_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, flip);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(Mirror8u, Mirror8uParamTest,
                         ::testing::Values(Mirror8uParam{32, 32, NPP_HORIZONTAL_AXIS, false, "32x32_H_noCtx"},
                                           Mirror8uParam{32, 32, NPP_HORIZONTAL_AXIS, true, "32x32_H_Ctx"},
                                           Mirror8uParam{32, 32, NPP_VERTICAL_AXIS, false, "32x32_V_noCtx"},
                                           Mirror8uParam{32, 32, NPP_BOTH_AXIS, false, "32x32_Both_noCtx"},
                                           Mirror8uParam{64, 64, NPP_HORIZONTAL_AXIS, false, "64x64_H_noCtx"}),
                         [](const ::testing::TestParamInfo<Mirror8uParam> &info) { return info.param.name; });

// ==================== Mirror 8u C3 TEST_P ====================

struct Mirror8uC3Param {
  int width;
  int height;
  NppiAxis flip;
  bool use_ctx;
  std::string name;
};

class Mirror8uC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<Mirror8uC3Param> {};

TEST_P(Mirror8uC3ParamTest, Mirror_8u_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const NppiAxis flip = param.flip;

  std::vector<Npp8u> srcData(width * height * 3);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  std::vector<Npp8u> expectedData(width * height * 3);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int srcX = x, srcY = y;
      if (flip == NPP_HORIZONTAL_AXIS || flip == NPP_BOTH_AXIS) {
        srcY = height - 1 - y;
      }
      if (flip == NPP_VERTICAL_AXIS || flip == NPP_BOTH_AXIS) {
        srcX = width - 1 - x;
      }
      for (int c = 0; c < 3; c++) {
        expectedData[(y * width + x) * 3 + c] = srcData[(srcY * width + srcX) * 3 + c];
      }
    }
  }

  NppImageMemory<Npp8u> src(width * 3, height);
  NppImageMemory<Npp8u> dst(width * 3, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiMirror_8u_C3R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, flip, ctx);
  } else {
    status = nppiMirror_8u_C3R(src.get(), src.step(), dst.get(), dst.step(), roi, flip);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultData(width * height * 3);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
}

INSTANTIATE_TEST_SUITE_P(Mirror8uC3, Mirror8uC3ParamTest,
                         ::testing::Values(Mirror8uC3Param{32, 32, NPP_HORIZONTAL_AXIS, false, "32x32_H_noCtx"},
                                           Mirror8uC3Param{32, 32, NPP_VERTICAL_AXIS, false, "32x32_V_noCtx"},
                                           Mirror8uC3Param{64, 64, NPP_BOTH_AXIS, false, "64x64_Both_noCtx"}),
                         [](const ::testing::TestParamInfo<Mirror8uC3Param> &info) { return info.param.name; });

// ==================== Mirror 32f C1 TEST_P ====================

struct Mirror32fParam {
  int width;
  int height;
  NppiAxis flip;
  bool use_ctx;
  std::string name;
};

class Mirror32fParamTest : public NppTestBase, public ::testing::WithParamInterface<Mirror32fParam> {};

TEST_P(Mirror32fParamTest, Mirror_32f_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const NppiAxis flip = param.flip;

  std::vector<Npp32f> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, -100.0f, 100.0f, 12345);

  std::vector<Npp32f> expectedData(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int srcX = x, srcY = y;
      if (flip == NPP_HORIZONTAL_AXIS || flip == NPP_BOTH_AXIS) {
        srcY = height - 1 - y;
      }
      if (flip == NPP_VERTICAL_AXIS || flip == NPP_BOTH_AXIS) {
        srcX = width - 1 - x;
      }
      expectedData[y * width + x] = srcData[srcY * width + srcX];
    }
  }

  NppImageMemory<Npp32f> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    status = nppiMirror_32f_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, flip, ctx);
  } else {
    status = nppiMirror_32f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, flip);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32f> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
}

INSTANTIATE_TEST_SUITE_P(Mirror32f, Mirror32fParamTest,
                         ::testing::Values(Mirror32fParam{32, 32, NPP_HORIZONTAL_AXIS, false, "32x32_H_noCtx"},
                                           Mirror32fParam{32, 32, NPP_VERTICAL_AXIS, false, "32x32_V_noCtx"},
                                           Mirror32fParam{64, 64, NPP_BOTH_AXIS, false, "64x64_Both_noCtx"}),
                         [](const ::testing::TestParamInfo<Mirror32fParam> &info) { return info.param.name; });
