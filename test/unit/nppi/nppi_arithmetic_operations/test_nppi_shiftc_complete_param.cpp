#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Common Param Structures ====================

struct ShiftCParam {
  int width;
  int height;
  Npp32u shiftAmount;
  bool use_ctx;
  bool in_place;
  std::string name;
};

// ==================== LShiftC 8u AC4 TEST_P ====================

class LShiftC8uAC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<ShiftCParam> {};

TEST_P(LShiftC8uAC4ParamTest, LShiftC_8u_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  const Npp32u aConstants[3] = {param.shiftAmount, param.shiftAmount, param.shiftAmount};

  std::vector<Npp8u> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  std::vector<Npp8u> expectedData(total);
  for (int i = 0; i < width * height; i++) {
    for (int c = 0; c < 3; c++) {
      expectedData[i * 4 + c] = static_cast<Npp8u>(srcData[i * 4 + c] << aConstants[c]);
    }
    expectedData[i * 4 + 3] = srcData[i * 4 + 3]; // Alpha unchanged
  }

  NppImageMemory<Npp8u> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiLShiftC_8u_AC4IR_Ctx(aConstants, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiLShiftC_8u_AC4IR(aConstants, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp8u> dst(width * channels, height);
    cudaMemcpy2D(dst.get(), dst.step(), src.get(), src.step(),
                 width * channels * sizeof(Npp8u), height, cudaMemcpyDeviceToDevice);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiLShiftC_8u_AC4R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiLShiftC_8u_AC4R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(LShiftC8uAC4, LShiftC8uAC4ParamTest,
                         ::testing::Values(ShiftCParam{32, 32, 2, false, false, "32x32_s2_noCtx"},
                                           ShiftCParam{32, 32, 2, true, false, "32x32_s2_Ctx"},
                                           ShiftCParam{32, 32, 4, false, true, "32x32_s4_InPlace"},
                                           ShiftCParam{32, 32, 4, true, true, "32x32_s4_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<ShiftCParam> &info) { return info.param.name; });

// ==================== LShiftC 16u AC4 TEST_P ====================

class LShiftC16uAC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<ShiftCParam> {};

TEST_P(LShiftC16uAC4ParamTest, LShiftC_16u_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  const Npp32u aConstants[3] = {param.shiftAmount, param.shiftAmount, param.shiftAmount};

  std::vector<Npp16u> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);

  std::vector<Npp16u> expectedData(total);
  for (int i = 0; i < width * height; i++) {
    for (int c = 0; c < 3; c++) {
      expectedData[i * 4 + c] = static_cast<Npp16u>(srcData[i * 4 + c] << aConstants[c]);
    }
    expectedData[i * 4 + 3] = srcData[i * 4 + 3];
  }

  NppImageMemory<Npp16u> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiLShiftC_16u_AC4IR_Ctx(aConstants, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiLShiftC_16u_AC4IR(aConstants, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16u> dst(width * channels, height);
    cudaMemcpy2D(dst.get(), dst.step(), src.get(), src.step(),
                 width * channels * sizeof(Npp16u), height, cudaMemcpyDeviceToDevice);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiLShiftC_16u_AC4R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiLShiftC_16u_AC4R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(LShiftC16uAC4, LShiftC16uAC4ParamTest,
                         ::testing::Values(ShiftCParam{32, 32, 4, false, false, "32x32_s4_noCtx"},
                                           ShiftCParam{32, 32, 4, true, false, "32x32_s4_Ctx"},
                                           ShiftCParam{32, 32, 8, false, true, "32x32_s8_InPlace"},
                                           ShiftCParam{32, 32, 8, true, true, "32x32_s8_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<ShiftCParam> &info) { return info.param.name; });

// ==================== LShiftC 32s C1/C3/C4/AC4 TEST_P ====================

class LShiftC32sC1ParamTest : public NppTestBase, public ::testing::WithParamInterface<ShiftCParam> {};

TEST_P(LShiftC32sC1ParamTest, LShiftC_32s_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int total = width * height;

  std::vector<Npp32s> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp32s>(0), static_cast<Npp32s>(0x7FFFFFFF), 12345);

  std::vector<Npp32s> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = srcData[i] << param.shiftAmount;
  }

  NppImageMemory<Npp32s> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiLShiftC_32s_C1IR_Ctx(param.shiftAmount, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiLShiftC_32s_C1IR(param.shiftAmount, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32s> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp32s> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiLShiftC_32s_C1R_Ctx(src.get(), src.step(), param.shiftAmount, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiLShiftC_32s_C1R(src.get(), src.step(), param.shiftAmount, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32s> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(LShiftC32sC1, LShiftC32sC1ParamTest,
                         ::testing::Values(ShiftCParam{32, 32, 4, false, false, "32x32_s4_noCtx"},
                                           ShiftCParam{32, 32, 4, true, false, "32x32_s4_Ctx"},
                                           ShiftCParam{32, 32, 8, false, true, "32x32_s8_InPlace"},
                                           ShiftCParam{32, 32, 8, true, true, "32x32_s8_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<ShiftCParam> &info) { return info.param.name; });

// ==================== LShiftC 32s C3 TEST_P ====================

class LShiftC32sC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<ShiftCParam> {};

TEST_P(LShiftC32sC3ParamTest, LShiftC_32s_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 3;
  const int total = width * height * channels;

  const Npp32u aConstants[3] = {param.shiftAmount, param.shiftAmount, param.shiftAmount};

  std::vector<Npp32s> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp32s>(0), static_cast<Npp32s>(0x7FFFFFFF), 12345);

  std::vector<Npp32s> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = srcData[i] << aConstants[i % 3];
  }

  NppImageMemory<Npp32s> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiLShiftC_32s_C3IR_Ctx(aConstants, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiLShiftC_32s_C3IR(aConstants, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32s> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp32s> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiLShiftC_32s_C3R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiLShiftC_32s_C3R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32s> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(LShiftC32sC3, LShiftC32sC3ParamTest,
                         ::testing::Values(ShiftCParam{32, 32, 4, false, false, "32x32_s4_noCtx"},
                                           ShiftCParam{32, 32, 4, true, false, "32x32_s4_Ctx"},
                                           ShiftCParam{32, 32, 8, false, true, "32x32_s8_InPlace"},
                                           ShiftCParam{32, 32, 8, true, true, "32x32_s8_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<ShiftCParam> &info) { return info.param.name; });

// ==================== LShiftC 32s C4 TEST_P ====================

class LShiftC32sC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<ShiftCParam> {};

TEST_P(LShiftC32sC4ParamTest, LShiftC_32s_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  const Npp32u aConstants[4] = {param.shiftAmount, param.shiftAmount, param.shiftAmount, param.shiftAmount};

  std::vector<Npp32s> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp32s>(0), static_cast<Npp32s>(0x7FFFFFFF), 12345);

  std::vector<Npp32s> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = srcData[i] << aConstants[i % 4];
  }

  NppImageMemory<Npp32s> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiLShiftC_32s_C4IR_Ctx(aConstants, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiLShiftC_32s_C4IR(aConstants, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32s> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp32s> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiLShiftC_32s_C4R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiLShiftC_32s_C4R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32s> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(LShiftC32sC4, LShiftC32sC4ParamTest,
                         ::testing::Values(ShiftCParam{32, 32, 4, false, false, "32x32_s4_noCtx"},
                                           ShiftCParam{32, 32, 4, true, false, "32x32_s4_Ctx"},
                                           ShiftCParam{32, 32, 8, false, true, "32x32_s8_InPlace"},
                                           ShiftCParam{32, 32, 8, true, true, "32x32_s8_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<ShiftCParam> &info) { return info.param.name; });

// ==================== LShiftC 32s AC4 TEST_P ====================

class LShiftC32sAC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<ShiftCParam> {};

TEST_P(LShiftC32sAC4ParamTest, LShiftC_32s_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  const Npp32u aConstants[3] = {param.shiftAmount, param.shiftAmount, param.shiftAmount};

  std::vector<Npp32s> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp32s>(0), static_cast<Npp32s>(0x7FFFFFFF), 12345);

  std::vector<Npp32s> expectedData(total);
  for (int i = 0; i < width * height; i++) {
    for (int c = 0; c < 3; c++) {
      expectedData[i * 4 + c] = srcData[i * 4 + c] << aConstants[c];
    }
    expectedData[i * 4 + 3] = srcData[i * 4 + 3];
  }

  NppImageMemory<Npp32s> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiLShiftC_32s_AC4IR_Ctx(aConstants, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiLShiftC_32s_AC4IR(aConstants, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32s> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp32s> dst(width * channels, height);
    cudaMemcpy2D(dst.get(), dst.step(), src.get(), src.step(),
                 width * channels * sizeof(Npp32s), height, cudaMemcpyDeviceToDevice);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiLShiftC_32s_AC4R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiLShiftC_32s_AC4R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32s> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(LShiftC32sAC4, LShiftC32sAC4ParamTest,
                         ::testing::Values(ShiftCParam{32, 32, 4, false, false, "32x32_s4_noCtx"},
                                           ShiftCParam{32, 32, 4, true, false, "32x32_s4_Ctx"},
                                           ShiftCParam{32, 32, 8, false, true, "32x32_s8_InPlace"},
                                           ShiftCParam{32, 32, 8, true, true, "32x32_s8_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<ShiftCParam> &info) { return info.param.name; });

// ==================== RShiftC 8u AC4 TEST_P ====================

class RShiftC8uAC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<ShiftCParam> {};

TEST_P(RShiftC8uAC4ParamTest, RShiftC_8u_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  const Npp32u aConstants[3] = {param.shiftAmount, param.shiftAmount, param.shiftAmount};

  std::vector<Npp8u> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  std::vector<Npp8u> expectedData(total);
  for (int i = 0; i < width * height; i++) {
    for (int c = 0; c < 3; c++) {
      expectedData[i * 4 + c] = static_cast<Npp8u>(srcData[i * 4 + c] >> aConstants[c]);
    }
    expectedData[i * 4 + 3] = srcData[i * 4 + 3];
  }

  NppImageMemory<Npp8u> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiRShiftC_8u_AC4IR_Ctx(aConstants, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiRShiftC_8u_AC4IR(aConstants, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp8u> dst(width * channels, height);
    cudaMemcpy2D(dst.get(), dst.step(), src.get(), src.step(),
                 width * channels * sizeof(Npp8u), height, cudaMemcpyDeviceToDevice);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiRShiftC_8u_AC4R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiRShiftC_8u_AC4R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(RShiftC8uAC4, RShiftC8uAC4ParamTest,
                         ::testing::Values(ShiftCParam{32, 32, 2, false, false, "32x32_s2_noCtx"},
                                           ShiftCParam{32, 32, 2, true, false, "32x32_s2_Ctx"},
                                           ShiftCParam{32, 32, 4, false, true, "32x32_s4_InPlace"},
                                           ShiftCParam{32, 32, 4, true, true, "32x32_s4_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<ShiftCParam> &info) { return info.param.name; });

// Note: RShiftC_8s tests are skipped because NppImageMemory<Npp8s> is not supported
// The 8s shift operations are already tested in test_nppi_shift_param.cpp

// ==================== RShiftC 16s C1 TEST_P ====================
// Note: NppImageMemory<Npp16s> only supports C1. C3/C4/AC4 tests are skipped.
// Those operations are already tested in test_nppi_shift_param.cpp

class RShiftC16sC1ParamTest : public NppTestBase, public ::testing::WithParamInterface<ShiftCParam> {};

TEST_P(RShiftC16sC1ParamTest, RShiftC_16s_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int total = width * height;

  std::vector<Npp16s> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(-32768), static_cast<Npp16s>(32767), 12345);

  std::vector<Npp16s> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = srcData[i] >> param.shiftAmount;
  }

  NppImageMemory<Npp16s> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiRShiftC_16s_C1IR_Ctx(param.shiftAmount, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiRShiftC_16s_C1IR(param.shiftAmount, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16s> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16s> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiRShiftC_16s_C1R_Ctx(src.get(), src.step(), param.shiftAmount, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiRShiftC_16s_C1R(src.get(), src.step(), param.shiftAmount, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16s> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(RShiftC16sC1, RShiftC16sC1ParamTest,
                         ::testing::Values(ShiftCParam{32, 32, 4, false, false, "32x32_s4_noCtx"},
                                           ShiftCParam{32, 32, 4, true, false, "32x32_s4_Ctx"},
                                           ShiftCParam{32, 32, 8, false, true, "32x32_s8_InPlace"},
                                           ShiftCParam{32, 32, 8, true, true, "32x32_s8_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<ShiftCParam> &info) { return info.param.name; });

// ==================== RShiftC 16u AC4 TEST_P ====================

class RShiftC16uAC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<ShiftCParam> {};

TEST_P(RShiftC16uAC4ParamTest, RShiftC_16u_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  const Npp32u aConstants[3] = {param.shiftAmount, param.shiftAmount, param.shiftAmount};

  std::vector<Npp16u> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);

  std::vector<Npp16u> expectedData(total);
  for (int i = 0; i < width * height; i++) {
    for (int c = 0; c < 3; c++) {
      expectedData[i * 4 + c] = static_cast<Npp16u>(srcData[i * 4 + c] >> aConstants[c]);
    }
    expectedData[i * 4 + 3] = srcData[i * 4 + 3];
  }

  NppImageMemory<Npp16u> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiRShiftC_16u_AC4IR_Ctx(aConstants, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiRShiftC_16u_AC4IR(aConstants, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16u> dst(width * channels, height);
    cudaMemcpy2D(dst.get(), dst.step(), src.get(), src.step(),
                 width * channels * sizeof(Npp16u), height, cudaMemcpyDeviceToDevice);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiRShiftC_16u_AC4R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiRShiftC_16u_AC4R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(RShiftC16uAC4, RShiftC16uAC4ParamTest,
                         ::testing::Values(ShiftCParam{32, 32, 4, false, false, "32x32_s4_noCtx"},
                                           ShiftCParam{32, 32, 4, true, false, "32x32_s4_Ctx"},
                                           ShiftCParam{32, 32, 8, false, true, "32x32_s8_InPlace"},
                                           ShiftCParam{32, 32, 8, true, true, "32x32_s8_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<ShiftCParam> &info) { return info.param.name; });

// ==================== RShiftC 32s Complete TEST_P ====================

class RShiftC32sC1ParamTest : public NppTestBase, public ::testing::WithParamInterface<ShiftCParam> {};

TEST_P(RShiftC32sC1ParamTest, RShiftC_32s_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int total = width * height;

  std::vector<Npp32s> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp32s>(-0x7FFFFFFF), static_cast<Npp32s>(0x7FFFFFFF), 12345);

  std::vector<Npp32s> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = srcData[i] >> param.shiftAmount;
  }

  NppImageMemory<Npp32s> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiRShiftC_32s_C1IR_Ctx(param.shiftAmount, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiRShiftC_32s_C1IR(param.shiftAmount, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32s> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp32s> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiRShiftC_32s_C1R_Ctx(src.get(), src.step(), param.shiftAmount, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiRShiftC_32s_C1R(src.get(), src.step(), param.shiftAmount, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32s> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(RShiftC32sC1, RShiftC32sC1ParamTest,
                         ::testing::Values(ShiftCParam{32, 32, 8, false, false, "32x32_s8_noCtx"},
                                           ShiftCParam{32, 32, 8, true, false, "32x32_s8_Ctx"},
                                           ShiftCParam{32, 32, 16, false, true, "32x32_s16_InPlace"},
                                           ShiftCParam{32, 32, 16, true, true, "32x32_s16_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<ShiftCParam> &info) { return info.param.name; });

// ==================== RShiftC 32s C3 TEST_P ====================

class RShiftC32sC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<ShiftCParam> {};

TEST_P(RShiftC32sC3ParamTest, RShiftC_32s_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 3;
  const int total = width * height * channels;

  const Npp32u aConstants[3] = {param.shiftAmount, param.shiftAmount, param.shiftAmount};

  std::vector<Npp32s> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp32s>(-0x7FFFFFFF), static_cast<Npp32s>(0x7FFFFFFF), 12345);

  std::vector<Npp32s> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = srcData[i] >> aConstants[i % 3];
  }

  NppImageMemory<Npp32s> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiRShiftC_32s_C3IR_Ctx(aConstants, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiRShiftC_32s_C3IR(aConstants, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32s> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp32s> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiRShiftC_32s_C3R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiRShiftC_32s_C3R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32s> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(RShiftC32sC3, RShiftC32sC3ParamTest,
                         ::testing::Values(ShiftCParam{32, 32, 8, false, false, "32x32_s8_noCtx"},
                                           ShiftCParam{32, 32, 8, true, false, "32x32_s8_Ctx"},
                                           ShiftCParam{32, 32, 16, false, true, "32x32_s16_InPlace"},
                                           ShiftCParam{32, 32, 16, true, true, "32x32_s16_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<ShiftCParam> &info) { return info.param.name; });

// ==================== RShiftC 32s C4 TEST_P ====================

class RShiftC32sC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<ShiftCParam> {};

TEST_P(RShiftC32sC4ParamTest, RShiftC_32s_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  const Npp32u aConstants[4] = {param.shiftAmount, param.shiftAmount, param.shiftAmount, param.shiftAmount};

  std::vector<Npp32s> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp32s>(-0x7FFFFFFF), static_cast<Npp32s>(0x7FFFFFFF), 12345);

  std::vector<Npp32s> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = srcData[i] >> aConstants[i % 4];
  }

  NppImageMemory<Npp32s> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiRShiftC_32s_C4IR_Ctx(aConstants, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiRShiftC_32s_C4IR(aConstants, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32s> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp32s> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiRShiftC_32s_C4R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiRShiftC_32s_C4R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32s> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(RShiftC32sC4, RShiftC32sC4ParamTest,
                         ::testing::Values(ShiftCParam{32, 32, 8, false, false, "32x32_s8_noCtx"},
                                           ShiftCParam{32, 32, 8, true, false, "32x32_s8_Ctx"},
                                           ShiftCParam{32, 32, 16, false, true, "32x32_s16_InPlace"},
                                           ShiftCParam{32, 32, 16, true, true, "32x32_s16_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<ShiftCParam> &info) { return info.param.name; });

// ==================== RShiftC 32s AC4 TEST_P ====================

class RShiftC32sAC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<ShiftCParam> {};

TEST_P(RShiftC32sAC4ParamTest, RShiftC_32s_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  const Npp32u aConstants[3] = {param.shiftAmount, param.shiftAmount, param.shiftAmount};

  std::vector<Npp32s> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp32s>(-0x7FFFFFFF), static_cast<Npp32s>(0x7FFFFFFF), 12345);

  std::vector<Npp32s> expectedData(total);
  for (int i = 0; i < width * height; i++) {
    for (int c = 0; c < 3; c++) {
      expectedData[i * 4 + c] = srcData[i * 4 + c] >> aConstants[c];
    }
    expectedData[i * 4 + 3] = srcData[i * 4 + 3];
  }

  NppImageMemory<Npp32s> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiRShiftC_32s_AC4IR_Ctx(aConstants, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiRShiftC_32s_AC4IR(aConstants, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32s> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp32s> dst(width * channels, height);
    cudaMemcpy2D(dst.get(), dst.step(), src.get(), src.step(),
                 width * channels * sizeof(Npp32s), height, cudaMemcpyDeviceToDevice);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiRShiftC_32s_AC4R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiRShiftC_32s_AC4R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32s> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(RShiftC32sAC4, RShiftC32sAC4ParamTest,
                         ::testing::Values(ShiftCParam{32, 32, 8, false, false, "32x32_s8_noCtx"},
                                           ShiftCParam{32, 32, 8, true, false, "32x32_s8_Ctx"},
                                           ShiftCParam{32, 32, 16, false, true, "32x32_s16_InPlace"},
                                           ShiftCParam{32, 32, 16, true, true, "32x32_s16_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<ShiftCParam> &info) { return info.param.name; });
