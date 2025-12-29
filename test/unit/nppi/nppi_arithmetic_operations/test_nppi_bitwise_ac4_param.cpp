#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Common Param Structure ====================

struct BitwiseAC4Param {
  int width;
  int height;
  bool use_ctx;
  bool in_place;
  std::string name;
};

// ==================== And 8u AC4 TEST_P ====================

class And8uAC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<BitwiseAC4Param> {};

TEST_P(And8uAC4ParamTest, And_8u_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  std::vector<Npp8u> src1Data(total);
  std::vector<Npp8u> src2Data(total);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 54321);

  // AC4: only first 3 channels processed, alpha unchanged
  std::vector<Npp8u> expectedData(total);
  for (int i = 0; i < width * height; i++) {
    for (int c = 0; c < 3; c++) {
      expectedData[i * 4 + c] = expect::and_val<Npp8u>(src1Data[i * 4 + c], src2Data[i * 4 + c]);
    }
    // Alpha channel: for in-place, src2 alpha unchanged; for non-inplace, copy from src1
    expectedData[i * 4 + 3] = param.in_place ? src2Data[i * 4 + 3] : src1Data[i * 4 + 3];
  }

  NppImageMemory<Npp8u> src1(width * channels, height);
  NppImageMemory<Npp8u> src2(width * channels, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiAnd_8u_AC4IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx);
    } else {
      status = nppiAnd_8u_AC4IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    src2.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp8u> dst(width * channels, height);
    // Pre-copy src1 to dst to initialize alpha channel
    cudaMemcpy2D(dst.get(), dst.step(), src1.get(), src1.step(), width * channels * sizeof(Npp8u), height,
                 cudaMemcpyDeviceToDevice);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiAnd_8u_AC4R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiAnd_8u_AC4R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(And8uAC4, And8uAC4ParamTest,
                         ::testing::Values(BitwiseAC4Param{32, 32, false, false, "32x32_noCtx"},
                                           BitwiseAC4Param{32, 32, true, false, "32x32_Ctx"},
                                           BitwiseAC4Param{32, 32, false, true, "32x32_InPlace"},
                                           BitwiseAC4Param{32, 32, true, true, "32x32_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<BitwiseAC4Param> &info) { return info.param.name; });

// ==================== And 16u AC4 TEST_P ====================

class And16uAC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<BitwiseAC4Param> {};

TEST_P(And16uAC4ParamTest, And_16u_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  std::vector<Npp16u> src1Data(total);
  std::vector<Npp16u> src2Data(total);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 54321);

  std::vector<Npp16u> expectedData(total);
  for (int i = 0; i < width * height; i++) {
    for (int c = 0; c < 3; c++) {
      expectedData[i * 4 + c] = expect::and_val<Npp16u>(src1Data[i * 4 + c], src2Data[i * 4 + c]);
    }
    expectedData[i * 4 + 3] = param.in_place ? src2Data[i * 4 + 3] : src1Data[i * 4 + 3];
  }

  NppImageMemory<Npp16u> src1(width * channels, height);
  NppImageMemory<Npp16u> src2(width * channels, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiAnd_16u_AC4IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx);
    } else {
      status = nppiAnd_16u_AC4IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    src2.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16u> dst(width * channels, height);
    cudaMemcpy2D(dst.get(), dst.step(), src1.get(), src1.step(), width * channels * sizeof(Npp16u), height,
                 cudaMemcpyDeviceToDevice);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiAnd_16u_AC4R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiAnd_16u_AC4R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(And16uAC4, And16uAC4ParamTest,
                         ::testing::Values(BitwiseAC4Param{32, 32, false, false, "32x32_noCtx"},
                                           BitwiseAC4Param{32, 32, true, false, "32x32_Ctx"},
                                           BitwiseAC4Param{32, 32, false, true, "32x32_InPlace"},
                                           BitwiseAC4Param{32, 32, true, true, "32x32_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<BitwiseAC4Param> &info) { return info.param.name; });

// ==================== And 16u C4 TEST_P ====================

class And16uC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<BitwiseAC4Param> {};

TEST_P(And16uC4ParamTest, And_16u_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  std::vector<Npp16u> src1Data(total);
  std::vector<Npp16u> src2Data(total);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 54321);

  std::vector<Npp16u> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::and_val<Npp16u>(src1Data[i], src2Data[i]);
  }

  NppImageMemory<Npp16u> src1(width * channels, height);
  NppImageMemory<Npp16u> src2(width * channels, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiAnd_16u_C4IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx);
    } else {
      status = nppiAnd_16u_C4IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    src2.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16u> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiAnd_16u_C4R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiAnd_16u_C4R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(And16uC4, And16uC4ParamTest,
                         ::testing::Values(BitwiseAC4Param{32, 32, false, false, "32x32_noCtx"},
                                           BitwiseAC4Param{32, 32, true, false, "32x32_Ctx"},
                                           BitwiseAC4Param{32, 32, false, true, "32x32_InPlace"},
                                           BitwiseAC4Param{32, 32, true, true, "32x32_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<BitwiseAC4Param> &info) { return info.param.name; });

// ==================== Or 8u AC4 TEST_P ====================

class Or8uAC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<BitwiseAC4Param> {};

TEST_P(Or8uAC4ParamTest, Or_8u_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  std::vector<Npp8u> src1Data(total);
  std::vector<Npp8u> src2Data(total);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 54321);

  std::vector<Npp8u> expectedData(total);
  for (int i = 0; i < width * height; i++) {
    for (int c = 0; c < 3; c++) {
      expectedData[i * 4 + c] = expect::or_val<Npp8u>(src1Data[i * 4 + c], src2Data[i * 4 + c]);
    }
    expectedData[i * 4 + 3] = param.in_place ? src2Data[i * 4 + 3] : src1Data[i * 4 + 3];
  }

  NppImageMemory<Npp8u> src1(width * channels, height);
  NppImageMemory<Npp8u> src2(width * channels, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiOr_8u_AC4IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx);
    } else {
      status = nppiOr_8u_AC4IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    src2.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp8u> dst(width * channels, height);
    cudaMemcpy2D(dst.get(), dst.step(), src1.get(), src1.step(), width * channels * sizeof(Npp8u), height,
                 cudaMemcpyDeviceToDevice);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiOr_8u_AC4R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiOr_8u_AC4R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Or8uAC4, Or8uAC4ParamTest,
                         ::testing::Values(BitwiseAC4Param{32, 32, false, false, "32x32_noCtx"},
                                           BitwiseAC4Param{32, 32, true, false, "32x32_Ctx"},
                                           BitwiseAC4Param{32, 32, false, true, "32x32_InPlace"},
                                           BitwiseAC4Param{32, 32, true, true, "32x32_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<BitwiseAC4Param> &info) { return info.param.name; });

// ==================== Or 16u AC4 TEST_P ====================

class Or16uAC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<BitwiseAC4Param> {};

TEST_P(Or16uAC4ParamTest, Or_16u_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  std::vector<Npp16u> src1Data(total);
  std::vector<Npp16u> src2Data(total);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 54321);

  std::vector<Npp16u> expectedData(total);
  for (int i = 0; i < width * height; i++) {
    for (int c = 0; c < 3; c++) {
      expectedData[i * 4 + c] = expect::or_val<Npp16u>(src1Data[i * 4 + c], src2Data[i * 4 + c]);
    }
    expectedData[i * 4 + 3] = param.in_place ? src2Data[i * 4 + 3] : src1Data[i * 4 + 3];
  }

  NppImageMemory<Npp16u> src1(width * channels, height);
  NppImageMemory<Npp16u> src2(width * channels, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiOr_16u_AC4IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx);
    } else {
      status = nppiOr_16u_AC4IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    src2.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16u> dst(width * channels, height);
    cudaMemcpy2D(dst.get(), dst.step(), src1.get(), src1.step(), width * channels * sizeof(Npp16u), height,
                 cudaMemcpyDeviceToDevice);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiOr_16u_AC4R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiOr_16u_AC4R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Or16uAC4, Or16uAC4ParamTest,
                         ::testing::Values(BitwiseAC4Param{32, 32, false, false, "32x32_noCtx"},
                                           BitwiseAC4Param{32, 32, true, false, "32x32_Ctx"},
                                           BitwiseAC4Param{32, 32, false, true, "32x32_InPlace"},
                                           BitwiseAC4Param{32, 32, true, true, "32x32_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<BitwiseAC4Param> &info) { return info.param.name; });

// ==================== Or 16u C4 TEST_P ====================

class Or16uC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<BitwiseAC4Param> {};

TEST_P(Or16uC4ParamTest, Or_16u_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  std::vector<Npp16u> src1Data(total);
  std::vector<Npp16u> src2Data(total);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 54321);

  std::vector<Npp16u> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::or_val<Npp16u>(src1Data[i], src2Data[i]);
  }

  NppImageMemory<Npp16u> src1(width * channels, height);
  NppImageMemory<Npp16u> src2(width * channels, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiOr_16u_C4IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx);
    } else {
      status = nppiOr_16u_C4IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    src2.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16u> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiOr_16u_C4R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiOr_16u_C4R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Or16uC4, Or16uC4ParamTest,
                         ::testing::Values(BitwiseAC4Param{32, 32, false, false, "32x32_noCtx"},
                                           BitwiseAC4Param{32, 32, true, false, "32x32_Ctx"},
                                           BitwiseAC4Param{32, 32, false, true, "32x32_InPlace"},
                                           BitwiseAC4Param{32, 32, true, true, "32x32_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<BitwiseAC4Param> &info) { return info.param.name; });

// ==================== Xor 8u AC4 TEST_P ====================

class Xor8uAC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<BitwiseAC4Param> {};

TEST_P(Xor8uAC4ParamTest, Xor_8u_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  std::vector<Npp8u> src1Data(total);
  std::vector<Npp8u> src2Data(total);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 54321);

  std::vector<Npp8u> expectedData(total);
  for (int i = 0; i < width * height; i++) {
    for (int c = 0; c < 3; c++) {
      expectedData[i * 4 + c] = expect::xor_val<Npp8u>(src1Data[i * 4 + c], src2Data[i * 4 + c]);
    }
    expectedData[i * 4 + 3] = param.in_place ? src2Data[i * 4 + 3] : src1Data[i * 4 + 3];
  }

  NppImageMemory<Npp8u> src1(width * channels, height);
  NppImageMemory<Npp8u> src2(width * channels, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiXor_8u_AC4IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx);
    } else {
      status = nppiXor_8u_AC4IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    src2.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp8u> dst(width * channels, height);
    cudaMemcpy2D(dst.get(), dst.step(), src1.get(), src1.step(), width * channels * sizeof(Npp8u), height,
                 cudaMemcpyDeviceToDevice);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiXor_8u_AC4R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiXor_8u_AC4R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Xor8uAC4, Xor8uAC4ParamTest,
                         ::testing::Values(BitwiseAC4Param{32, 32, false, false, "32x32_noCtx"},
                                           BitwiseAC4Param{32, 32, true, false, "32x32_Ctx"},
                                           BitwiseAC4Param{32, 32, false, true, "32x32_InPlace"},
                                           BitwiseAC4Param{32, 32, true, true, "32x32_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<BitwiseAC4Param> &info) { return info.param.name; });

// ==================== Xor 16u AC4 TEST_P ====================

class Xor16uAC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<BitwiseAC4Param> {};

TEST_P(Xor16uAC4ParamTest, Xor_16u_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  std::vector<Npp16u> src1Data(total);
  std::vector<Npp16u> src2Data(total);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 54321);

  std::vector<Npp16u> expectedData(total);
  for (int i = 0; i < width * height; i++) {
    for (int c = 0; c < 3; c++) {
      expectedData[i * 4 + c] = expect::xor_val<Npp16u>(src1Data[i * 4 + c], src2Data[i * 4 + c]);
    }
    expectedData[i * 4 + 3] = param.in_place ? src2Data[i * 4 + 3] : src1Data[i * 4 + 3];
  }

  NppImageMemory<Npp16u> src1(width * channels, height);
  NppImageMemory<Npp16u> src2(width * channels, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiXor_16u_AC4IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx);
    } else {
      status = nppiXor_16u_AC4IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    src2.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16u> dst(width * channels, height);
    cudaMemcpy2D(dst.get(), dst.step(), src1.get(), src1.step(), width * channels * sizeof(Npp16u), height,
                 cudaMemcpyDeviceToDevice);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiXor_16u_AC4R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiXor_16u_AC4R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Xor16uAC4, Xor16uAC4ParamTest,
                         ::testing::Values(BitwiseAC4Param{32, 32, false, false, "32x32_noCtx"},
                                           BitwiseAC4Param{32, 32, true, false, "32x32_Ctx"},
                                           BitwiseAC4Param{32, 32, false, true, "32x32_InPlace"},
                                           BitwiseAC4Param{32, 32, true, true, "32x32_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<BitwiseAC4Param> &info) { return info.param.name; });

// ==================== Xor 16u C4 TEST_P ====================

class Xor16uC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<BitwiseAC4Param> {};

TEST_P(Xor16uC4ParamTest, Xor_16u_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  std::vector<Npp16u> src1Data(total);
  std::vector<Npp16u> src2Data(total);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 54321);

  std::vector<Npp16u> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::xor_val<Npp16u>(src1Data[i], src2Data[i]);
  }

  NppImageMemory<Npp16u> src1(width * channels, height);
  NppImageMemory<Npp16u> src2(width * channels, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiXor_16u_C4IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx);
    } else {
      status = nppiXor_16u_C4IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    src2.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16u> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiXor_16u_C4R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiXor_16u_C4R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Xor16uC4, Xor16uC4ParamTest,
                         ::testing::Values(BitwiseAC4Param{32, 32, false, false, "32x32_noCtx"},
                                           BitwiseAC4Param{32, 32, true, false, "32x32_Ctx"},
                                           BitwiseAC4Param{32, 32, false, true, "32x32_InPlace"},
                                           BitwiseAC4Param{32, 32, true, true, "32x32_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<BitwiseAC4Param> &info) { return info.param.name; });

// ==================== Not 8u AC4 TEST_P ====================

class Not8uAC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<BitwiseAC4Param> {};

TEST_P(Not8uAC4ParamTest, Not_8u_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  std::vector<Npp8u> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  std::vector<Npp8u> expectedData(total);
  for (int i = 0; i < width * height; i++) {
    for (int c = 0; c < 3; c++) {
      expectedData[i * 4 + c] = expect::not_val<Npp8u>(srcData[i * 4 + c]);
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
      status = nppiNot_8u_AC4IR_Ctx(src.get(), src.step(), roi, ctx);
    } else {
      status = nppiNot_8u_AC4IR(src.get(), src.step(), roi);
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
      status = nppiNot_8u_AC4R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiNot_8u_AC4R(src.get(), src.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Not8uAC4, Not8uAC4ParamTest,
                         ::testing::Values(BitwiseAC4Param{32, 32, false, false, "32x32_noCtx"},
                                           BitwiseAC4Param{32, 32, true, false, "32x32_Ctx"},
                                           BitwiseAC4Param{32, 32, false, true, "32x32_InPlace"},
                                           BitwiseAC4Param{32, 32, true, true, "32x32_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<BitwiseAC4Param> &info) { return info.param.name; });
