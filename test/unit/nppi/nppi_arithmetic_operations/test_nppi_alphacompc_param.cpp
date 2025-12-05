#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Common Param Structures ====================

struct AlphaCompCParam {
  int width;
  int height;
  NppiAlphaOp alphaOp;
  bool use_ctx;
  std::string name;
};

// ==================== AlphaCompC 8u C1R TEST_P ====================

class AlphaCompC8uC1RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaCompCParam> {};

TEST_P(AlphaCompC8uC1RParamTest, AlphaCompC_8u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> src1Data(width * height);
  std::vector<Npp8u> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 54321);

  NppImageMemory<Npp8u> src1(width, height);
  NppImageMemory<Npp8u> src2(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  Npp8u alpha1 = 128;
  Npp8u alpha2 = 200;

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaCompC_8u_C1R_Ctx(src1.get(), src1.step(), alpha1, src2.get(), src2.step(), alpha2, dst.get(),
                                       dst.step(), roi, param.alphaOp, ctx);
  } else {
    status = nppiAlphaCompC_8u_C1R(src1.get(), src1.step(), alpha1, src2.get(), src2.step(), alpha2, dst.get(),
                                   dst.step(), roi, param.alphaOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaCompC8uC1R, AlphaCompC8uC1RParamTest,
                         ::testing::Values(AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OVER, false, "32x32_Over_noCtx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OVER, true, "32x32_Over_Ctx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_IN, false, "32x32_In_noCtx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OUT, false, "32x32_Out_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaCompCParam> &info) { return info.param.name; });

// ==================== AlphaCompC 8u C3R TEST_P ====================

class AlphaCompC8uC3RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaCompCParam> {};

TEST_P(AlphaCompC8uC3RParamTest, AlphaCompC_8u_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 3;

  std::vector<Npp8u> src1Data(width * height * channels);
  std::vector<Npp8u> src2Data(width * height * channels);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 54321);

  NppImageMemory<Npp8u> src1(width, height, 3);
  NppImageMemory<Npp8u> src2(width, height, 3);
  NppImageMemory<Npp8u> dst(width, height, 3);

  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  Npp8u alpha1 = 128;
  Npp8u alpha2 = 200;

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaCompC_8u_C3R_Ctx(src1.get(), src1.step(), alpha1, src2.get(), src2.step(), alpha2, dst.get(),
                                       dst.step(), roi, param.alphaOp, ctx);
  } else {
    status = nppiAlphaCompC_8u_C3R(src1.get(), src1.step(), alpha1, src2.get(), src2.step(), alpha2, dst.get(),
                                   dst.step(), roi, param.alphaOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaCompC8uC3R, AlphaCompC8uC3RParamTest,
                         ::testing::Values(AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OVER, false, "32x32_Over_noCtx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OVER, true, "32x32_Over_Ctx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_IN, false, "32x32_In_noCtx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OUT, false, "32x32_Out_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaCompCParam> &info) { return info.param.name; });

// ==================== AlphaCompC 8u C4R TEST_P ====================

class AlphaCompC8uC4RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaCompCParam> {};

TEST_P(AlphaCompC8uC4RParamTest, AlphaCompC_8u_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;

  std::vector<Npp8u> src1Data(width * height * channels);
  std::vector<Npp8u> src2Data(width * height * channels);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 54321);

  NppImageMemory<Npp8u> src1(width, height, 4);
  NppImageMemory<Npp8u> src2(width, height, 4);
  NppImageMemory<Npp8u> dst(width, height, 4);

  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  Npp8u alpha1 = 128;
  Npp8u alpha2 = 200;

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaCompC_8u_C4R_Ctx(src1.get(), src1.step(), alpha1, src2.get(), src2.step(), alpha2, dst.get(),
                                       dst.step(), roi, param.alphaOp, ctx);
  } else {
    status = nppiAlphaCompC_8u_C4R(src1.get(), src1.step(), alpha1, src2.get(), src2.step(), alpha2, dst.get(),
                                   dst.step(), roi, param.alphaOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaCompC8uC4R, AlphaCompC8uC4RParamTest,
                         ::testing::Values(AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OVER, false, "32x32_Over_noCtx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OVER, true, "32x32_Over_Ctx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_IN, false, "32x32_In_noCtx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OUT, false, "32x32_Out_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaCompCParam> &info) { return info.param.name; });

// ==================== AlphaCompC 8u AC4R TEST_P ====================

class AlphaCompC8uAC4RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaCompCParam> {};

TEST_P(AlphaCompC8uAC4RParamTest, AlphaCompC_8u_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;

  std::vector<Npp8u> src1Data(width * height * channels);
  std::vector<Npp8u> src2Data(width * height * channels);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 54321);

  NppImageMemory<Npp8u> src1(width, height, 4);
  NppImageMemory<Npp8u> src2(width, height, 4);
  NppImageMemory<Npp8u> dst(width, height, 4);

  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  Npp8u alpha1 = 128;
  Npp8u alpha2 = 200;

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaCompC_8u_AC4R_Ctx(src1.get(), src1.step(), alpha1, src2.get(), src2.step(), alpha2, dst.get(),
                                        dst.step(), roi, param.alphaOp, ctx);
  } else {
    status = nppiAlphaCompC_8u_AC4R(src1.get(), src1.step(), alpha1, src2.get(), src2.step(), alpha2, dst.get(),
                                    dst.step(), roi, param.alphaOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaCompC8uAC4R, AlphaCompC8uAC4RParamTest,
                         ::testing::Values(AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OVER, false, "32x32_Over_noCtx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OVER, true, "32x32_Over_Ctx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_IN, false, "32x32_In_noCtx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OUT, false, "32x32_Out_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaCompCParam> &info) { return info.param.name; });

// ==================== AlphaCompC 16u C1R TEST_P ====================

class AlphaCompC16uC1RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaCompCParam> {};

TEST_P(AlphaCompC16uC1RParamTest, AlphaCompC_16u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> src1Data(width * height);
  std::vector<Npp16u> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 54321);

  NppImageMemory<Npp16u> src1(width, height);
  NppImageMemory<Npp16u> src2(width, height);
  NppImageMemory<Npp16u> dst(width, height);

  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  Npp16u alpha1 = 32768;
  Npp16u alpha2 = 50000;

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaCompC_16u_C1R_Ctx(src1.get(), src1.step(), alpha1, src2.get(), src2.step(), alpha2, dst.get(),
                                        dst.step(), roi, param.alphaOp, ctx);
  } else {
    status = nppiAlphaCompC_16u_C1R(src1.get(), src1.step(), alpha1, src2.get(), src2.step(), alpha2, dst.get(),
                                    dst.step(), roi, param.alphaOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaCompC16uC1R, AlphaCompC16uC1RParamTest,
                         ::testing::Values(AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OVER, false, "32x32_Over_noCtx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OVER, true, "32x32_Over_Ctx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_IN, false, "32x32_In_noCtx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OUT, false, "32x32_Out_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaCompCParam> &info) { return info.param.name; });

// ==================== AlphaCompC 16u C3R TEST_P ====================

class AlphaCompC16uC3RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaCompCParam> {};

TEST_P(AlphaCompC16uC3RParamTest, AlphaCompC_16u_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 3;

  std::vector<Npp16u> src1Data(width * height * channels);
  std::vector<Npp16u> src2Data(width * height * channels);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 54321);

  NppImageMemory<Npp16u> src1(width, height, 3);
  NppImageMemory<Npp16u> src2(width, height, 3);
  NppImageMemory<Npp16u> dst(width, height, 3);

  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  Npp16u alpha1 = 32768;
  Npp16u alpha2 = 50000;

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaCompC_16u_C3R_Ctx(src1.get(), src1.step(), alpha1, src2.get(), src2.step(), alpha2, dst.get(),
                                        dst.step(), roi, param.alphaOp, ctx);
  } else {
    status = nppiAlphaCompC_16u_C3R(src1.get(), src1.step(), alpha1, src2.get(), src2.step(), alpha2, dst.get(),
                                    dst.step(), roi, param.alphaOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaCompC16uC3R, AlphaCompC16uC3RParamTest,
                         ::testing::Values(AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OVER, false, "32x32_Over_noCtx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OVER, true, "32x32_Over_Ctx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_IN, false, "32x32_In_noCtx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OUT, false, "32x32_Out_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaCompCParam> &info) { return info.param.name; });

// ==================== AlphaCompC 16u C4R TEST_P ====================

class AlphaCompC16uC4RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaCompCParam> {};

TEST_P(AlphaCompC16uC4RParamTest, AlphaCompC_16u_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;

  std::vector<Npp16u> src1Data(width * height * channels);
  std::vector<Npp16u> src2Data(width * height * channels);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 54321);

  NppImageMemory<Npp16u> src1(width, height, 4);
  NppImageMemory<Npp16u> src2(width, height, 4);
  NppImageMemory<Npp16u> dst(width, height, 4);

  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  Npp16u alpha1 = 32768;
  Npp16u alpha2 = 50000;

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaCompC_16u_C4R_Ctx(src1.get(), src1.step(), alpha1, src2.get(), src2.step(), alpha2, dst.get(),
                                        dst.step(), roi, param.alphaOp, ctx);
  } else {
    status = nppiAlphaCompC_16u_C4R(src1.get(), src1.step(), alpha1, src2.get(), src2.step(), alpha2, dst.get(),
                                    dst.step(), roi, param.alphaOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaCompC16uC4R, AlphaCompC16uC4RParamTest,
                         ::testing::Values(AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OVER, false, "32x32_Over_noCtx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OVER, true, "32x32_Over_Ctx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_IN, false, "32x32_In_noCtx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OUT, false, "32x32_Out_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaCompCParam> &info) { return info.param.name; });

// ==================== AlphaCompC 16u AC4R TEST_P ====================

class AlphaCompC16uAC4RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaCompCParam> {};

TEST_P(AlphaCompC16uAC4RParamTest, AlphaCompC_16u_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;

  std::vector<Npp16u> src1Data(width * height * channels);
  std::vector<Npp16u> src2Data(width * height * channels);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 54321);

  NppImageMemory<Npp16u> src1(width, height, 4);
  NppImageMemory<Npp16u> src2(width, height, 4);
  NppImageMemory<Npp16u> dst(width, height, 4);

  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  Npp16u alpha1 = 32768;
  Npp16u alpha2 = 50000;

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaCompC_16u_AC4R_Ctx(src1.get(), src1.step(), alpha1, src2.get(), src2.step(), alpha2, dst.get(),
                                         dst.step(), roi, param.alphaOp, ctx);
  } else {
    status = nppiAlphaCompC_16u_AC4R(src1.get(), src1.step(), alpha1, src2.get(), src2.step(), alpha2, dst.get(),
                                     dst.step(), roi, param.alphaOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaCompC16uAC4R, AlphaCompC16uAC4RParamTest,
                         ::testing::Values(AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OVER, false, "32x32_Over_noCtx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OVER, true, "32x32_Over_Ctx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_IN, false, "32x32_In_noCtx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OUT, false, "32x32_Out_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaCompCParam> &info) { return info.param.name; });

// ==================== AlphaCompC 32f C1R TEST_P ====================

class AlphaCompC32fC1RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaCompCParam> {};

TEST_P(AlphaCompC32fC1RParamTest, AlphaCompC_32f_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> src1Data(width * height);
  std::vector<Npp32f> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, 0.0f, 1.0f, 12345);
  TestDataGenerator::generateRandom(src2Data, 0.0f, 1.0f, 54321);

  NppImageMemory<Npp32f> src1(width, height);
  NppImageMemory<Npp32f> src2(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  Npp32f alpha1 = 0.5f;
  Npp32f alpha2 = 0.8f;

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaCompC_32f_C1R_Ctx(src1.get(), src1.step(), alpha1, src2.get(), src2.step(), alpha2, dst.get(),
                                        dst.step(), roi, param.alphaOp, ctx);
  } else {
    status = nppiAlphaCompC_32f_C1R(src1.get(), src1.step(), alpha1, src2.get(), src2.step(), alpha2, dst.get(),
                                    dst.step(), roi, param.alphaOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaCompC32fC1R, AlphaCompC32fC1RParamTest,
                         ::testing::Values(AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OVER, false, "32x32_Over_noCtx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OVER, true, "32x32_Over_Ctx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_IN, false, "32x32_In_noCtx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OUT, false, "32x32_Out_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaCompCParam> &info) { return info.param.name; });

// ==================== AlphaCompC 16s C1R TEST_P ====================

class AlphaCompC16sC1RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaCompCParam> {};

TEST_P(AlphaCompC16sC1RParamTest, AlphaCompC_16s_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16s> src1Data(width * height);
  std::vector<Npp16s> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16s>(0), static_cast<Npp16s>(255), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16s>(0), static_cast<Npp16s>(255), 54321);

  NppImageMemory<Npp16s> src1(width, height);
  NppImageMemory<Npp16s> src2(width, height);
  NppImageMemory<Npp16s> dst(width, height);

  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  Npp16s alpha1 = 128;
  Npp16s alpha2 = 200;

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaCompC_16s_C1R_Ctx(src1.get(), src1.step(), alpha1, src2.get(), src2.step(), alpha2, dst.get(),
                                        dst.step(), roi, param.alphaOp, ctx);
  } else {
    status = nppiAlphaCompC_16s_C1R(src1.get(), src1.step(), alpha1, src2.get(), src2.step(), alpha2, dst.get(),
                                    dst.step(), roi, param.alphaOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaCompC16sC1R, AlphaCompC16sC1RParamTest,
                         ::testing::Values(AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OVER, false, "32x32_Over_noCtx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OVER, true, "32x32_Over_Ctx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_IN, false, "32x32_In_noCtx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OUT, false, "32x32_Out_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaCompCParam> &info) { return info.param.name; });

// ==================== AlphaCompC 8s C1R TEST_P ====================

class AlphaCompC8sC1RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaCompCParam> {};

TEST_P(AlphaCompC8sC1RParamTest, AlphaCompC_8s_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8s> src1Data(width * height);
  std::vector<Npp8s> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8s>(0), static_cast<Npp8s>(127), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8s>(0), static_cast<Npp8s>(127), 54321);

  Npp8s *d_src1 = nullptr;
  Npp8s *d_src2 = nullptr;
  Npp8s *d_dst = nullptr;
  int src1Step, src2Step, dstStep;

  d_src1 = reinterpret_cast<Npp8s *>(nppiMalloc_8u_C1(width, height, &src1Step));
  d_src2 = reinterpret_cast<Npp8s *>(nppiMalloc_8u_C1(width, height, &src2Step));
  d_dst = reinterpret_cast<Npp8s *>(nppiMalloc_8u_C1(width, height, &dstStep));

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_src2, nullptr);
  ASSERT_NE(d_dst, nullptr);

  int hostStep = width * sizeof(Npp8s);
  cudaMemcpy2D(d_src1, src1Step, src1Data.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_src2, src2Step, src2Data.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  Npp8s alpha1 = 64;
  Npp8s alpha2 = 100;

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaCompC_8s_C1R_Ctx(d_src1, src1Step, alpha1, d_src2, src2Step, alpha2, d_dst, dstStep, roi,
                                       param.alphaOp, ctx);
  } else {
    status =
        nppiAlphaCompC_8s_C1R(d_src1, src1Step, alpha1, d_src2, src2Step, alpha2, d_dst, dstStep, roi, param.alphaOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  nppiFree(d_src1);
  nppiFree(d_src2);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(AlphaCompC8sC1R, AlphaCompC8sC1RParamTest,
                         ::testing::Values(AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OVER, false, "32x32_Over_noCtx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OVER, true, "32x32_Over_Ctx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_IN, false, "32x32_In_noCtx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OUT, false, "32x32_Out_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaCompCParam> &info) { return info.param.name; });

// ==================== AlphaCompC 32s C1R TEST_P ====================

class AlphaCompC32sC1RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaCompCParam> {};

TEST_P(AlphaCompC32sC1RParamTest, AlphaCompC_32s_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32s> src1Data(width * height);
  std::vector<Npp32s> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp32s>(0), static_cast<Npp32s>(255), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp32s>(0), static_cast<Npp32s>(255), 54321);

  NppImageMemory<Npp32s> src1(width, height);
  NppImageMemory<Npp32s> src2(width, height);
  NppImageMemory<Npp32s> dst(width, height);

  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  Npp32s alpha1 = 128;
  Npp32s alpha2 = 200;

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaCompC_32s_C1R_Ctx(src1.get(), src1.step(), alpha1, src2.get(), src2.step(), alpha2, dst.get(),
                                        dst.step(), roi, param.alphaOp, ctx);
  } else {
    status = nppiAlphaCompC_32s_C1R(src1.get(), src1.step(), alpha1, src2.get(), src2.step(), alpha2, dst.get(),
                                    dst.step(), roi, param.alphaOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaCompC32sC1R, AlphaCompC32sC1RParamTest,
                         ::testing::Values(AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OVER, false, "32x32_Over_noCtx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OVER, true, "32x32_Over_Ctx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_IN, false, "32x32_In_noCtx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OUT, false, "32x32_Out_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaCompCParam> &info) { return info.param.name; });

// ==================== AlphaCompC 32u C1R TEST_P ====================

class AlphaCompC32uC1RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaCompCParam> {};

TEST_P(AlphaCompC32uC1RParamTest, AlphaCompC_32u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32u> src1Data(width * height);
  std::vector<Npp32u> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp32u>(0), static_cast<Npp32u>(255), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp32u>(0), static_cast<Npp32u>(255), 54321);

  Npp32u *d_src1 = nullptr;
  Npp32u *d_src2 = nullptr;
  Npp32u *d_dst = nullptr;
  int src1Step, src2Step, dstStep;

  d_src1 = reinterpret_cast<Npp32u *>(nppiMalloc_32s_C1(width, height, &src1Step));
  d_src2 = reinterpret_cast<Npp32u *>(nppiMalloc_32s_C1(width, height, &src2Step));
  d_dst = reinterpret_cast<Npp32u *>(nppiMalloc_32s_C1(width, height, &dstStep));

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_src2, nullptr);
  ASSERT_NE(d_dst, nullptr);

  int hostStep = width * sizeof(Npp32u);
  cudaMemcpy2D(d_src1, src1Step, src1Data.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_src2, src2Step, src2Data.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  Npp32u alpha1 = 128;
  Npp32u alpha2 = 200;

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaCompC_32u_C1R_Ctx(d_src1, src1Step, alpha1, d_src2, src2Step, alpha2, d_dst, dstStep, roi,
                                        param.alphaOp, ctx);
  } else {
    status =
        nppiAlphaCompC_32u_C1R(d_src1, src1Step, alpha1, d_src2, src2Step, alpha2, d_dst, dstStep, roi, param.alphaOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  nppiFree(d_src1);
  nppiFree(d_src2);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(AlphaCompC32uC1R, AlphaCompC32uC1RParamTest,
                         ::testing::Values(AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OVER, false, "32x32_Over_noCtx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OVER, true, "32x32_Over_Ctx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_IN, false, "32x32_In_noCtx"},
                                           AlphaCompCParam{32, 32, NPPI_OP_ALPHA_OUT, false, "32x32_Out_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaCompCParam> &info) { return info.param.name; });
