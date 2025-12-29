#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Common Param Structures ====================

struct BitwiseC8uAC4Param {
  int width;
  int height;
  Npp8u constant;
  bool use_ctx;
  bool in_place;
  std::string name;
};

struct BitwiseC16uAC4Param {
  int width;
  int height;
  Npp16u constant;
  bool use_ctx;
  bool in_place;
  std::string name;
};

// ==================== AndC 8u AC4 TEST_P ====================

class AndC8uAC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<BitwiseC8uAC4Param> {};

TEST_P(AndC8uAC4ParamTest, AndC_8u_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  const Npp8u aConstants[3] = {param.constant, param.constant, param.constant};

  std::vector<Npp8u> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  std::vector<Npp8u> expectedData(total);
  for (int i = 0; i < width * height; i++) {
    for (int c = 0; c < 3; c++) {
      expectedData[i * 4 + c] = expect::and_c<Npp8u>(srcData[i * 4 + c], aConstants[c]);
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
      status = nppiAndC_8u_AC4IR_Ctx(aConstants, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiAndC_8u_AC4IR(aConstants, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp8u> dst(width * channels, height);
    // Pre-copy src to dst to initialize alpha channel
    cudaMemcpy2D(dst.get(), dst.step(), src.get(), src.step(), width * channels * sizeof(Npp8u), height,
                 cudaMemcpyDeviceToDevice);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiAndC_8u_AC4R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiAndC_8u_AC4R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(AndC8uAC4, AndC8uAC4ParamTest,
                         ::testing::Values(BitwiseC8uAC4Param{32, 32, 0xF0, false, false, "32x32_cF0_noCtx"},
                                           BitwiseC8uAC4Param{32, 32, 0xF0, true, false, "32x32_cF0_Ctx"},
                                           BitwiseC8uAC4Param{32, 32, 0x0F, false, true, "32x32_c0F_InPlace"},
                                           BitwiseC8uAC4Param{32, 32, 0x0F, true, true, "32x32_c0F_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<BitwiseC8uAC4Param> &info) { return info.param.name; });

// ==================== AndC 16u C1/C3 TEST_P ====================

class AndC16uC1ParamTest : public NppTestBase, public ::testing::WithParamInterface<BitwiseC16uAC4Param> {};

TEST_P(AndC16uC1ParamTest, AndC_16u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int total = width * height;

  std::vector<Npp16u> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);

  std::vector<Npp16u> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::and_c<Npp16u>(srcData[i], param.constant);
  }

  NppImageMemory<Npp16u> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiAndC_16u_C1IR_Ctx(param.constant, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiAndC_16u_C1IR(param.constant, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16u> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiAndC_16u_C1R_Ctx(src.get(), src.step(), param.constant, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiAndC_16u_C1R(src.get(), src.step(), param.constant, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(AndC16uC1, AndC16uC1ParamTest,
                         ::testing::Values(BitwiseC16uAC4Param{32, 32, 0xFF00, false, false, "32x32_cFF00_noCtx"},
                                           BitwiseC16uAC4Param{32, 32, 0xFF00, true, false, "32x32_cFF00_Ctx"},
                                           BitwiseC16uAC4Param{32, 32, 0x00FF, false, true, "32x32_c00FF_InPlace"},
                                           BitwiseC16uAC4Param{32, 32, 0x00FF, true, true, "32x32_c00FF_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<BitwiseC16uAC4Param> &info) { return info.param.name; });

// ==================== AndC 16u C3 TEST_P ====================

class AndC16uC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<BitwiseC16uAC4Param> {};

TEST_P(AndC16uC3ParamTest, AndC_16u_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 3;
  const int total = width * height * channels;

  const Npp16u aConstants[3] = {param.constant, param.constant, param.constant};

  std::vector<Npp16u> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);

  std::vector<Npp16u> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::and_c<Npp16u>(srcData[i], aConstants[i % 3]);
  }

  NppImageMemory<Npp16u> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiAndC_16u_C3IR_Ctx(aConstants, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiAndC_16u_C3IR(aConstants, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16u> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiAndC_16u_C3R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiAndC_16u_C3R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(AndC16uC3, AndC16uC3ParamTest,
                         ::testing::Values(BitwiseC16uAC4Param{32, 32, 0xFF00, false, false, "32x32_cFF00_noCtx"},
                                           BitwiseC16uAC4Param{32, 32, 0xFF00, true, false, "32x32_cFF00_Ctx"},
                                           BitwiseC16uAC4Param{32, 32, 0x00FF, false, true, "32x32_c00FF_InPlace"},
                                           BitwiseC16uAC4Param{32, 32, 0x00FF, true, true, "32x32_c00FF_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<BitwiseC16uAC4Param> &info) { return info.param.name; });

// ==================== AndC 16u C4 TEST_P ====================

class AndC16uC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<BitwiseC16uAC4Param> {};

TEST_P(AndC16uC4ParamTest, AndC_16u_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  const Npp16u aConstants[4] = {param.constant, param.constant, param.constant, param.constant};

  std::vector<Npp16u> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);

  std::vector<Npp16u> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::and_c<Npp16u>(srcData[i], aConstants[i % 4]);
  }

  NppImageMemory<Npp16u> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiAndC_16u_C4IR_Ctx(aConstants, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiAndC_16u_C4IR(aConstants, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16u> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiAndC_16u_C4R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiAndC_16u_C4R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(AndC16uC4, AndC16uC4ParamTest,
                         ::testing::Values(BitwiseC16uAC4Param{32, 32, 0xFF00, false, false, "32x32_cFF00_noCtx"},
                                           BitwiseC16uAC4Param{32, 32, 0xFF00, true, false, "32x32_cFF00_Ctx"},
                                           BitwiseC16uAC4Param{32, 32, 0x00FF, false, true, "32x32_c00FF_InPlace"},
                                           BitwiseC16uAC4Param{32, 32, 0x00FF, true, true, "32x32_c00FF_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<BitwiseC16uAC4Param> &info) { return info.param.name; });

// ==================== AndC 16u AC4 TEST_P ====================

class AndC16uAC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<BitwiseC16uAC4Param> {};

TEST_P(AndC16uAC4ParamTest, AndC_16u_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  const Npp16u aConstants[3] = {param.constant, param.constant, param.constant};

  std::vector<Npp16u> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);

  std::vector<Npp16u> expectedData(total);
  for (int i = 0; i < width * height; i++) {
    for (int c = 0; c < 3; c++) {
      expectedData[i * 4 + c] = expect::and_c<Npp16u>(srcData[i * 4 + c], aConstants[c]);
    }
    expectedData[i * 4 + 3] = srcData[i * 4 + 3]; // Alpha unchanged
  }

  NppImageMemory<Npp16u> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiAndC_16u_AC4IR_Ctx(aConstants, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiAndC_16u_AC4IR(aConstants, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16u> dst(width * channels, height);
    cudaMemcpy2D(dst.get(), dst.step(), src.get(), src.step(), width * channels * sizeof(Npp16u), height,
                 cudaMemcpyDeviceToDevice);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiAndC_16u_AC4R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiAndC_16u_AC4R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(AndC16uAC4, AndC16uAC4ParamTest,
                         ::testing::Values(BitwiseC16uAC4Param{32, 32, 0xFF00, false, false, "32x32_cFF00_noCtx"},
                                           BitwiseC16uAC4Param{32, 32, 0xFF00, true, false, "32x32_cFF00_Ctx"},
                                           BitwiseC16uAC4Param{32, 32, 0x00FF, false, true, "32x32_c00FF_InPlace"},
                                           BitwiseC16uAC4Param{32, 32, 0x00FF, true, true, "32x32_c00FF_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<BitwiseC16uAC4Param> &info) { return info.param.name; });

// ==================== OrC 8u AC4 TEST_P ====================

class OrC8uAC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<BitwiseC8uAC4Param> {};

TEST_P(OrC8uAC4ParamTest, OrC_8u_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  const Npp8u aConstants[3] = {param.constant, param.constant, param.constant};

  std::vector<Npp8u> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  std::vector<Npp8u> expectedData(total);
  for (int i = 0; i < width * height; i++) {
    for (int c = 0; c < 3; c++) {
      expectedData[i * 4 + c] = expect::or_c<Npp8u>(srcData[i * 4 + c], aConstants[c]);
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
      status = nppiOrC_8u_AC4IR_Ctx(aConstants, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiOrC_8u_AC4IR(aConstants, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp8u> dst(width * channels, height);
    cudaMemcpy2D(dst.get(), dst.step(), src.get(), src.step(), width * channels * sizeof(Npp8u), height,
                 cudaMemcpyDeviceToDevice);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiOrC_8u_AC4R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiOrC_8u_AC4R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(OrC8uAC4, OrC8uAC4ParamTest,
                         ::testing::Values(BitwiseC8uAC4Param{32, 32, 0xF0, false, false, "32x32_cF0_noCtx"},
                                           BitwiseC8uAC4Param{32, 32, 0xF0, true, false, "32x32_cF0_Ctx"},
                                           BitwiseC8uAC4Param{32, 32, 0x0F, false, true, "32x32_c0F_InPlace"},
                                           BitwiseC8uAC4Param{32, 32, 0x0F, true, true, "32x32_c0F_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<BitwiseC8uAC4Param> &info) { return info.param.name; });

// ==================== OrC 16u C1 TEST_P ====================

class OrC16uC1ParamTest : public NppTestBase, public ::testing::WithParamInterface<BitwiseC16uAC4Param> {};

TEST_P(OrC16uC1ParamTest, OrC_16u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int total = width * height;

  std::vector<Npp16u> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);

  std::vector<Npp16u> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::or_c<Npp16u>(srcData[i], param.constant);
  }

  NppImageMemory<Npp16u> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiOrC_16u_C1IR_Ctx(param.constant, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiOrC_16u_C1IR(param.constant, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16u> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiOrC_16u_C1R_Ctx(src.get(), src.step(), param.constant, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiOrC_16u_C1R(src.get(), src.step(), param.constant, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(OrC16uC1, OrC16uC1ParamTest,
                         ::testing::Values(BitwiseC16uAC4Param{32, 32, 0xFF00, false, false, "32x32_cFF00_noCtx"},
                                           BitwiseC16uAC4Param{32, 32, 0xFF00, true, false, "32x32_cFF00_Ctx"},
                                           BitwiseC16uAC4Param{32, 32, 0x00FF, false, true, "32x32_c00FF_InPlace"},
                                           BitwiseC16uAC4Param{32, 32, 0x00FF, true, true, "32x32_c00FF_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<BitwiseC16uAC4Param> &info) { return info.param.name; });

// ==================== OrC 16u C3 TEST_P ====================

class OrC16uC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<BitwiseC16uAC4Param> {};

TEST_P(OrC16uC3ParamTest, OrC_16u_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 3;
  const int total = width * height * channels;

  const Npp16u aConstants[3] = {param.constant, param.constant, param.constant};

  std::vector<Npp16u> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);

  std::vector<Npp16u> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::or_c<Npp16u>(srcData[i], aConstants[i % 3]);
  }

  NppImageMemory<Npp16u> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiOrC_16u_C3IR_Ctx(aConstants, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiOrC_16u_C3IR(aConstants, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16u> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiOrC_16u_C3R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiOrC_16u_C3R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(OrC16uC3, OrC16uC3ParamTest,
                         ::testing::Values(BitwiseC16uAC4Param{32, 32, 0xFF00, false, false, "32x32_cFF00_noCtx"},
                                           BitwiseC16uAC4Param{32, 32, 0xFF00, true, false, "32x32_cFF00_Ctx"},
                                           BitwiseC16uAC4Param{32, 32, 0x00FF, false, true, "32x32_c00FF_InPlace"},
                                           BitwiseC16uAC4Param{32, 32, 0x00FF, true, true, "32x32_c00FF_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<BitwiseC16uAC4Param> &info) { return info.param.name; });

// ==================== OrC 16u C4 TEST_P ====================

class OrC16uC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<BitwiseC16uAC4Param> {};

TEST_P(OrC16uC4ParamTest, OrC_16u_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  const Npp16u aConstants[4] = {param.constant, param.constant, param.constant, param.constant};

  std::vector<Npp16u> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);

  std::vector<Npp16u> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::or_c<Npp16u>(srcData[i], aConstants[i % 4]);
  }

  NppImageMemory<Npp16u> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiOrC_16u_C4IR_Ctx(aConstants, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiOrC_16u_C4IR(aConstants, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16u> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiOrC_16u_C4R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiOrC_16u_C4R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(OrC16uC4, OrC16uC4ParamTest,
                         ::testing::Values(BitwiseC16uAC4Param{32, 32, 0xFF00, false, false, "32x32_cFF00_noCtx"},
                                           BitwiseC16uAC4Param{32, 32, 0xFF00, true, false, "32x32_cFF00_Ctx"},
                                           BitwiseC16uAC4Param{32, 32, 0x00FF, false, true, "32x32_c00FF_InPlace"},
                                           BitwiseC16uAC4Param{32, 32, 0x00FF, true, true, "32x32_c00FF_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<BitwiseC16uAC4Param> &info) { return info.param.name; });

// ==================== OrC 16u AC4 TEST_P ====================

class OrC16uAC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<BitwiseC16uAC4Param> {};

TEST_P(OrC16uAC4ParamTest, OrC_16u_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  const Npp16u aConstants[3] = {param.constant, param.constant, param.constant};

  std::vector<Npp16u> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);

  std::vector<Npp16u> expectedData(total);
  for (int i = 0; i < width * height; i++) {
    for (int c = 0; c < 3; c++) {
      expectedData[i * 4 + c] = expect::or_c<Npp16u>(srcData[i * 4 + c], aConstants[c]);
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
      status = nppiOrC_16u_AC4IR_Ctx(aConstants, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiOrC_16u_AC4IR(aConstants, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16u> dst(width * channels, height);
    cudaMemcpy2D(dst.get(), dst.step(), src.get(), src.step(), width * channels * sizeof(Npp16u), height,
                 cudaMemcpyDeviceToDevice);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiOrC_16u_AC4R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiOrC_16u_AC4R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(OrC16uAC4, OrC16uAC4ParamTest,
                         ::testing::Values(BitwiseC16uAC4Param{32, 32, 0xFF00, false, false, "32x32_cFF00_noCtx"},
                                           BitwiseC16uAC4Param{32, 32, 0xFF00, true, false, "32x32_cFF00_Ctx"},
                                           BitwiseC16uAC4Param{32, 32, 0x00FF, false, true, "32x32_c00FF_InPlace"},
                                           BitwiseC16uAC4Param{32, 32, 0x00FF, true, true, "32x32_c00FF_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<BitwiseC16uAC4Param> &info) { return info.param.name; });

// ==================== XorC 8u AC4 TEST_P ====================

class XorC8uAC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<BitwiseC8uAC4Param> {};

TEST_P(XorC8uAC4ParamTest, XorC_8u_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  const Npp8u aConstants[3] = {param.constant, param.constant, param.constant};

  std::vector<Npp8u> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  std::vector<Npp8u> expectedData(total);
  for (int i = 0; i < width * height; i++) {
    for (int c = 0; c < 3; c++) {
      expectedData[i * 4 + c] = expect::xor_c<Npp8u>(srcData[i * 4 + c], aConstants[c]);
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
      status = nppiXorC_8u_AC4IR_Ctx(aConstants, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiXorC_8u_AC4IR(aConstants, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp8u> dst(width * channels, height);
    cudaMemcpy2D(dst.get(), dst.step(), src.get(), src.step(), width * channels * sizeof(Npp8u), height,
                 cudaMemcpyDeviceToDevice);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiXorC_8u_AC4R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiXorC_8u_AC4R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(XorC8uAC4, XorC8uAC4ParamTest,
                         ::testing::Values(BitwiseC8uAC4Param{32, 32, 0xF0, false, false, "32x32_cF0_noCtx"},
                                           BitwiseC8uAC4Param{32, 32, 0xF0, true, false, "32x32_cF0_Ctx"},
                                           BitwiseC8uAC4Param{32, 32, 0x0F, false, true, "32x32_c0F_InPlace"},
                                           BitwiseC8uAC4Param{32, 32, 0x0F, true, true, "32x32_c0F_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<BitwiseC8uAC4Param> &info) { return info.param.name; });

// ==================== XorC 16u C1 TEST_P ====================

class XorC16uC1ParamTest : public NppTestBase, public ::testing::WithParamInterface<BitwiseC16uAC4Param> {};

TEST_P(XorC16uC1ParamTest, XorC_16u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int total = width * height;

  std::vector<Npp16u> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);

  std::vector<Npp16u> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::xor_c<Npp16u>(srcData[i], param.constant);
  }

  NppImageMemory<Npp16u> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiXorC_16u_C1IR_Ctx(param.constant, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiXorC_16u_C1IR(param.constant, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16u> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiXorC_16u_C1R_Ctx(src.get(), src.step(), param.constant, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiXorC_16u_C1R(src.get(), src.step(), param.constant, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(XorC16uC1, XorC16uC1ParamTest,
                         ::testing::Values(BitwiseC16uAC4Param{32, 32, 0xFF00, false, false, "32x32_cFF00_noCtx"},
                                           BitwiseC16uAC4Param{32, 32, 0xFF00, true, false, "32x32_cFF00_Ctx"},
                                           BitwiseC16uAC4Param{32, 32, 0x00FF, false, true, "32x32_c00FF_InPlace"},
                                           BitwiseC16uAC4Param{32, 32, 0x00FF, true, true, "32x32_c00FF_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<BitwiseC16uAC4Param> &info) { return info.param.name; });

// ==================== XorC 16u C3 TEST_P ====================

class XorC16uC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<BitwiseC16uAC4Param> {};

TEST_P(XorC16uC3ParamTest, XorC_16u_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 3;
  const int total = width * height * channels;

  const Npp16u aConstants[3] = {param.constant, param.constant, param.constant};

  std::vector<Npp16u> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);

  std::vector<Npp16u> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::xor_c<Npp16u>(srcData[i], aConstants[i % 3]);
  }

  NppImageMemory<Npp16u> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiXorC_16u_C3IR_Ctx(aConstants, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiXorC_16u_C3IR(aConstants, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16u> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiXorC_16u_C3R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiXorC_16u_C3R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(XorC16uC3, XorC16uC3ParamTest,
                         ::testing::Values(BitwiseC16uAC4Param{32, 32, 0xFF00, false, false, "32x32_cFF00_noCtx"},
                                           BitwiseC16uAC4Param{32, 32, 0xFF00, true, false, "32x32_cFF00_Ctx"},
                                           BitwiseC16uAC4Param{32, 32, 0x00FF, false, true, "32x32_c00FF_InPlace"},
                                           BitwiseC16uAC4Param{32, 32, 0x00FF, true, true, "32x32_c00FF_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<BitwiseC16uAC4Param> &info) { return info.param.name; });

// ==================== XorC 16u C4 TEST_P ====================

class XorC16uC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<BitwiseC16uAC4Param> {};

TEST_P(XorC16uC4ParamTest, XorC_16u_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  const Npp16u aConstants[4] = {param.constant, param.constant, param.constant, param.constant};

  std::vector<Npp16u> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);

  std::vector<Npp16u> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::xor_c<Npp16u>(srcData[i], aConstants[i % 4]);
  }

  NppImageMemory<Npp16u> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiXorC_16u_C4IR_Ctx(aConstants, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiXorC_16u_C4IR(aConstants, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16u> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiXorC_16u_C4R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiXorC_16u_C4R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(XorC16uC4, XorC16uC4ParamTest,
                         ::testing::Values(BitwiseC16uAC4Param{32, 32, 0xFF00, false, false, "32x32_cFF00_noCtx"},
                                           BitwiseC16uAC4Param{32, 32, 0xFF00, true, false, "32x32_cFF00_Ctx"},
                                           BitwiseC16uAC4Param{32, 32, 0x00FF, false, true, "32x32_c00FF_InPlace"},
                                           BitwiseC16uAC4Param{32, 32, 0x00FF, true, true, "32x32_c00FF_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<BitwiseC16uAC4Param> &info) { return info.param.name; });

// ==================== XorC 16u AC4 TEST_P ====================

class XorC16uAC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<BitwiseC16uAC4Param> {};

TEST_P(XorC16uAC4ParamTest, XorC_16u_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  const Npp16u aConstants[3] = {param.constant, param.constant, param.constant};

  std::vector<Npp16u> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);

  std::vector<Npp16u> expectedData(total);
  for (int i = 0; i < width * height; i++) {
    for (int c = 0; c < 3; c++) {
      expectedData[i * 4 + c] = expect::xor_c<Npp16u>(srcData[i * 4 + c], aConstants[c]);
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
      status = nppiXorC_16u_AC4IR_Ctx(aConstants, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiXorC_16u_AC4IR(aConstants, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16u> dst(width * channels, height);
    cudaMemcpy2D(dst.get(), dst.step(), src.get(), src.step(), width * channels * sizeof(Npp16u), height,
                 cudaMemcpyDeviceToDevice);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiXorC_16u_AC4R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiXorC_16u_AC4R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(XorC16uAC4, XorC16uAC4ParamTest,
                         ::testing::Values(BitwiseC16uAC4Param{32, 32, 0xFF00, false, false, "32x32_cFF00_noCtx"},
                                           BitwiseC16uAC4Param{32, 32, 0xFF00, true, false, "32x32_cFF00_Ctx"},
                                           BitwiseC16uAC4Param{32, 32, 0x00FF, false, true, "32x32_c00FF_InPlace"},
                                           BitwiseC16uAC4Param{32, 32, 0x00FF, true, true, "32x32_c00FF_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<BitwiseC16uAC4Param> &info) { return info.param.name; });
