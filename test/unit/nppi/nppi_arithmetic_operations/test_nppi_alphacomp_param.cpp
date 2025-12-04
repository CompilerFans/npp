#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Common Param Structures ====================

struct AlphaCompParam {
  int width;
  int height;
  NppiAlphaOp alphaOp;
  bool use_ctx;
  std::string name;
};

// ==================== AlphaComp 8u AC1R TEST_P ====================

class AlphaComp8uAC1RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaCompParam> {};

TEST_P(AlphaComp8uAC1RParamTest, AlphaComp_8u_AC1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 2; // AC1 has 2 channels: color + alpha

  std::vector<Npp8u> src1Data(width * height * channels);
  std::vector<Npp8u> src2Data(width * height * channels);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 54321);

  Npp8u *d_src1 = nullptr;
  Npp8u *d_src2 = nullptr;
  Npp8u *d_dst = nullptr;
  int src1Step, src2Step, dstStep;

  d_src1 = nppiMalloc_8u_C2(width, height, &src1Step);
  d_src2 = nppiMalloc_8u_C2(width, height, &src2Step);
  d_dst = nppiMalloc_8u_C2(width, height, &dstStep);

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_src2, nullptr);
  ASSERT_NE(d_dst, nullptr);

  int hostStep = width * channels * sizeof(Npp8u);
  cudaMemcpy2D(d_src1, src1Step, src1Data.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_src2, src2Step, src2Data.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaComp_8u_AC1R_Ctx(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, roi, param.alphaOp, ctx);
  } else {
    status = nppiAlphaComp_8u_AC1R(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, roi, param.alphaOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  nppiFree(d_src1);
  nppiFree(d_src2);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(AlphaComp8uAC1R, AlphaComp8uAC1RParamTest,
                         ::testing::Values(AlphaCompParam{32, 32, NPPI_OP_ALPHA_OVER, false, "32x32_Over_noCtx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_OVER, true, "32x32_Over_Ctx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_IN, false, "32x32_In_noCtx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_OUT, false, "32x32_Out_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaCompParam> &info) { return info.param.name; });

// ==================== AlphaComp 16u AC1R TEST_P ====================

class AlphaComp16uAC1RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaCompParam> {};

TEST_P(AlphaComp16uAC1RParamTest, AlphaComp_16u_AC1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 2;

  std::vector<Npp16u> src1Data(width * height * channels);
  std::vector<Npp16u> src2Data(width * height * channels);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 54321);

  Npp16u *d_src1 = nullptr;
  Npp16u *d_src2 = nullptr;
  Npp16u *d_dst = nullptr;
  int src1Step, src2Step, dstStep;

  // Use C3 allocation for width * 2 channels (treating 2 channels as if part of C3)
  d_src1 = reinterpret_cast<Npp16u*>(nppiMalloc_16u_C1(width * channels, height, &src1Step));
  d_src2 = reinterpret_cast<Npp16u*>(nppiMalloc_16u_C1(width * channels, height, &src2Step));
  d_dst = reinterpret_cast<Npp16u*>(nppiMalloc_16u_C1(width * channels, height, &dstStep));

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_src2, nullptr);
  ASSERT_NE(d_dst, nullptr);

  int hostStep = width * channels * sizeof(Npp16u);
  cudaMemcpy2D(d_src1, src1Step, src1Data.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_src2, src2Step, src2Data.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaComp_16u_AC1R_Ctx(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, roi, param.alphaOp, ctx);
  } else {
    status = nppiAlphaComp_16u_AC1R(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, roi, param.alphaOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  nppiFree(d_src1);
  nppiFree(d_src2);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(AlphaComp16uAC1R, AlphaComp16uAC1RParamTest,
                         ::testing::Values(AlphaCompParam{32, 32, NPPI_OP_ALPHA_OVER, false, "32x32_Over_noCtx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_OVER, true, "32x32_Over_Ctx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_IN, false, "32x32_In_noCtx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_OUT, false, "32x32_Out_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaCompParam> &info) { return info.param.name; });

// ==================== AlphaComp 32f AC1R TEST_P ====================

class AlphaComp32fAC1RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaCompParam> {};

TEST_P(AlphaComp32fAC1RParamTest, AlphaComp_32f_AC1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 2;

  std::vector<Npp32f> src1Data(width * height * channels);
  std::vector<Npp32f> src2Data(width * height * channels);
  TestDataGenerator::generateRandom(src1Data, 0.0f, 1.0f, 12345);
  TestDataGenerator::generateRandom(src2Data, 0.0f, 1.0f, 54321);

  Npp32f *d_src1 = nullptr;
  Npp32f *d_src2 = nullptr;
  Npp32f *d_dst = nullptr;
  int src1Step, src2Step, dstStep;

  d_src1 = reinterpret_cast<Npp32f*>(nppiMalloc_32f_C1(width * channels, height, &src1Step));
  d_src2 = reinterpret_cast<Npp32f*>(nppiMalloc_32f_C1(width * channels, height, &src2Step));
  d_dst = reinterpret_cast<Npp32f*>(nppiMalloc_32f_C1(width * channels, height, &dstStep));

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_src2, nullptr);
  ASSERT_NE(d_dst, nullptr);

  int hostStep = width * channels * sizeof(Npp32f);
  cudaMemcpy2D(d_src1, src1Step, src1Data.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_src2, src2Step, src2Data.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaComp_32f_AC1R_Ctx(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, roi, param.alphaOp, ctx);
  } else {
    status = nppiAlphaComp_32f_AC1R(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, roi, param.alphaOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  nppiFree(d_src1);
  nppiFree(d_src2);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(AlphaComp32fAC1R, AlphaComp32fAC1RParamTest,
                         ::testing::Values(AlphaCompParam{32, 32, NPPI_OP_ALPHA_OVER, false, "32x32_Over_noCtx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_OVER, true, "32x32_Over_Ctx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_IN, false, "32x32_In_noCtx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_OUT, false, "32x32_Out_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaCompParam> &info) { return info.param.name; });

// ==================== AlphaComp 32s AC1R TEST_P ====================

class AlphaComp32sAC1RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaCompParam> {};

TEST_P(AlphaComp32sAC1RParamTest, AlphaComp_32s_AC1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 2;

  std::vector<Npp32s> src1Data(width * height * channels);
  std::vector<Npp32s> src2Data(width * height * channels);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp32s>(0), static_cast<Npp32s>(255), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp32s>(0), static_cast<Npp32s>(255), 54321);

  Npp32s *d_src1 = nullptr;
  Npp32s *d_src2 = nullptr;
  Npp32s *d_dst = nullptr;
  int src1Step, src2Step, dstStep;

  d_src1 = reinterpret_cast<Npp32s*>(nppiMalloc_32s_C1(width * channels, height, &src1Step));
  d_src2 = reinterpret_cast<Npp32s*>(nppiMalloc_32s_C1(width * channels, height, &src2Step));
  d_dst = reinterpret_cast<Npp32s*>(nppiMalloc_32s_C1(width * channels, height, &dstStep));

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_src2, nullptr);
  ASSERT_NE(d_dst, nullptr);

  int hostStep = width * channels * sizeof(Npp32s);
  cudaMemcpy2D(d_src1, src1Step, src1Data.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_src2, src2Step, src2Data.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaComp_32s_AC1R_Ctx(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, roi, param.alphaOp, ctx);
  } else {
    status = nppiAlphaComp_32s_AC1R(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, roi, param.alphaOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  nppiFree(d_src1);
  nppiFree(d_src2);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(AlphaComp32sAC1R, AlphaComp32sAC1RParamTest,
                         ::testing::Values(AlphaCompParam{32, 32, NPPI_OP_ALPHA_OVER, false, "32x32_Over_noCtx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_OVER, true, "32x32_Over_Ctx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_IN, false, "32x32_In_noCtx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_OUT, false, "32x32_Out_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaCompParam> &info) { return info.param.name; });

// ==================== AlphaComp 32u AC1R TEST_P ====================

class AlphaComp32uAC1RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaCompParam> {};

TEST_P(AlphaComp32uAC1RParamTest, AlphaComp_32u_AC1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 2;

  std::vector<Npp32u> src1Data(width * height * channels);
  std::vector<Npp32u> src2Data(width * height * channels);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp32u>(0), static_cast<Npp32u>(255), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp32u>(0), static_cast<Npp32u>(255), 54321);

  Npp32u *d_src1 = nullptr;
  Npp32u *d_src2 = nullptr;
  Npp32u *d_dst = nullptr;
  int src1Step, src2Step, dstStep;

  // Use 32s allocation for 32u
  d_src1 = reinterpret_cast<Npp32u*>(nppiMalloc_32s_C1(width * channels, height, &src1Step));
  d_src2 = reinterpret_cast<Npp32u*>(nppiMalloc_32s_C1(width * channels, height, &src2Step));
  d_dst = reinterpret_cast<Npp32u*>(nppiMalloc_32s_C1(width * channels, height, &dstStep));

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_src2, nullptr);
  ASSERT_NE(d_dst, nullptr);

  int hostStep = width * channels * sizeof(Npp32u);
  cudaMemcpy2D(d_src1, src1Step, src1Data.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_src2, src2Step, src2Data.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaComp_32u_AC1R_Ctx(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, roi, param.alphaOp, ctx);
  } else {
    status = nppiAlphaComp_32u_AC1R(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, roi, param.alphaOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  nppiFree(d_src1);
  nppiFree(d_src2);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(AlphaComp32uAC1R, AlphaComp32uAC1RParamTest,
                         ::testing::Values(AlphaCompParam{32, 32, NPPI_OP_ALPHA_OVER, false, "32x32_Over_noCtx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_OVER, true, "32x32_Over_Ctx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_IN, false, "32x32_In_noCtx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_OUT, false, "32x32_Out_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaCompParam> &info) { return info.param.name; });

// ==================== AlphaComp 32u AC4R TEST_P ====================

class AlphaComp32uAC4RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaCompParam> {};

TEST_P(AlphaComp32uAC4RParamTest, AlphaComp_32u_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;

  std::vector<Npp32u> src1Data(width * height * channels);
  std::vector<Npp32u> src2Data(width * height * channels);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp32u>(0), static_cast<Npp32u>(255), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp32u>(0), static_cast<Npp32u>(255), 54321);

  Npp32u *d_src1 = nullptr;
  Npp32u *d_src2 = nullptr;
  Npp32u *d_dst = nullptr;
  int src1Step, src2Step, dstStep;

  d_src1 = reinterpret_cast<Npp32u*>(nppiMalloc_32s_C4(width, height, &src1Step));
  d_src2 = reinterpret_cast<Npp32u*>(nppiMalloc_32s_C4(width, height, &src2Step));
  d_dst = reinterpret_cast<Npp32u*>(nppiMalloc_32s_C4(width, height, &dstStep));

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_src2, nullptr);
  ASSERT_NE(d_dst, nullptr);

  int hostStep = width * channels * sizeof(Npp32u);
  cudaMemcpy2D(d_src1, src1Step, src1Data.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_src2, src2Step, src2Data.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaComp_32u_AC4R_Ctx(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, roi, param.alphaOp, ctx);
  } else {
    status = nppiAlphaComp_32u_AC4R(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, roi, param.alphaOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  nppiFree(d_src1);
  nppiFree(d_src2);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(AlphaComp32uAC4R, AlphaComp32uAC4RParamTest,
                         ::testing::Values(AlphaCompParam{32, 32, NPPI_OP_ALPHA_OVER, false, "32x32_Over_noCtx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_OVER, true, "32x32_Over_Ctx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_IN, false, "32x32_In_noCtx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_OUT, false, "32x32_Out_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaCompParam> &info) { return info.param.name; });

// ==================== AlphaComp 16s AC1R TEST_P ====================

class AlphaComp16sAC1RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaCompParam> {};

TEST_P(AlphaComp16sAC1RParamTest, AlphaComp_16s_AC1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 2;

  std::vector<Npp16s> src1Data(width * height * channels);
  std::vector<Npp16s> src2Data(width * height * channels);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16s>(0), static_cast<Npp16s>(255), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16s>(0), static_cast<Npp16s>(255), 54321);

  NppImageMemory<Npp16s> src1(width * channels, height);
  NppImageMemory<Npp16s> src2(width * channels, height);
  NppImageMemory<Npp16s> dst(width * channels, height);

  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaComp_16s_AC1R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(),
                                         dst.get(), dst.step(), roi, param.alphaOp, ctx);
  } else {
    status = nppiAlphaComp_16s_AC1R(src1.get(), src1.step(), src2.get(), src2.step(),
                                     dst.get(), dst.step(), roi, param.alphaOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaComp16sAC1R, AlphaComp16sAC1RParamTest,
                         ::testing::Values(AlphaCompParam{32, 32, NPPI_OP_ALPHA_OVER, false, "32x32_Over_noCtx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_OVER, true, "32x32_Over_Ctx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_IN, false, "32x32_In_noCtx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_OUT, false, "32x32_Out_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaCompParam> &info) { return info.param.name; });

// ==================== AlphaComp 8s AC1R TEST_P ====================
// Note: Npp8s uses NppImageMemory via reinterpret_cast from 8u

class AlphaComp8sAC1RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaCompParam> {};

TEST_P(AlphaComp8sAC1RParamTest, AlphaComp_8s_AC1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 2;

  std::vector<Npp8s> src1Data(width * height * channels);
  std::vector<Npp8s> src2Data(width * height * channels);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8s>(0), static_cast<Npp8s>(127), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8s>(0), static_cast<Npp8s>(127), 54321);

  Npp8s *d_src1 = nullptr;
  Npp8s *d_src2 = nullptr;
  Npp8s *d_dst = nullptr;
  int src1Step, src2Step, dstStep;

  d_src1 = reinterpret_cast<Npp8s*>(nppiMalloc_8u_C2(width, height, &src1Step));
  d_src2 = reinterpret_cast<Npp8s*>(nppiMalloc_8u_C2(width, height, &src2Step));
  d_dst = reinterpret_cast<Npp8s*>(nppiMalloc_8u_C2(width, height, &dstStep));

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_src2, nullptr);
  ASSERT_NE(d_dst, nullptr);

  int hostStep = width * channels * sizeof(Npp8s);
  cudaMemcpy2D(d_src1, src1Step, src1Data.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_src2, src2Step, src2Data.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaComp_8s_AC1R_Ctx(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, roi, param.alphaOp, ctx);
  } else {
    status = nppiAlphaComp_8s_AC1R(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, roi, param.alphaOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  nppiFree(d_src1);
  nppiFree(d_src2);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(AlphaComp8sAC1R, AlphaComp8sAC1RParamTest,
                         ::testing::Values(AlphaCompParam{32, 32, NPPI_OP_ALPHA_OVER, false, "32x32_Over_noCtx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_OVER, true, "32x32_Over_Ctx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_IN, false, "32x32_In_noCtx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_OUT, false, "32x32_Out_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaCompParam> &info) { return info.param.name; });

// ==================== AlphaComp 8u AC4R TEST_P ====================

class AlphaComp8uAC4RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaCompParam> {};

TEST_P(AlphaComp8uAC4RParamTest, AlphaComp_8u_AC4R) {
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

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaComp_8u_AC4R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(),
                                        dst.get(), dst.step(), roi, param.alphaOp, ctx);
  } else {
    status = nppiAlphaComp_8u_AC4R(src1.get(), src1.step(), src2.get(), src2.step(),
                                    dst.get(), dst.step(), roi, param.alphaOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaComp8uAC4R, AlphaComp8uAC4RParamTest,
                         ::testing::Values(AlphaCompParam{32, 32, NPPI_OP_ALPHA_OVER, false, "32x32_Over_noCtx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_OVER, true, "32x32_Over_Ctx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_IN, false, "32x32_In_noCtx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_OUT, false, "32x32_Out_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaCompParam> &info) { return info.param.name; });

// ==================== AlphaComp 16u AC4R TEST_P ====================

class AlphaComp16uAC4RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaCompParam> {};

TEST_P(AlphaComp16uAC4RParamTest, AlphaComp_16u_AC4R) {
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

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaComp_16u_AC4R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(),
                                         dst.get(), dst.step(), roi, param.alphaOp, ctx);
  } else {
    status = nppiAlphaComp_16u_AC4R(src1.get(), src1.step(), src2.get(), src2.step(),
                                     dst.get(), dst.step(), roi, param.alphaOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaComp16uAC4R, AlphaComp16uAC4RParamTest,
                         ::testing::Values(AlphaCompParam{32, 32, NPPI_OP_ALPHA_OVER, false, "32x32_Over_noCtx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_OVER, true, "32x32_Over_Ctx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_IN, false, "32x32_In_noCtx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_OUT, false, "32x32_Out_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaCompParam> &info) { return info.param.name; });

// ==================== AlphaComp 32f AC4R TEST_P ====================

class AlphaComp32fAC4RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaCompParam> {};

TEST_P(AlphaComp32fAC4RParamTest, AlphaComp_32f_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;

  std::vector<Npp32f> src1Data(width * height * channels);
  std::vector<Npp32f> src2Data(width * height * channels);
  TestDataGenerator::generateRandom(src1Data, 0.0f, 1.0f, 12345);
  TestDataGenerator::generateRandom(src2Data, 0.0f, 1.0f, 54321);

  NppImageMemory<Npp32f> src1(width, height, 4);
  NppImageMemory<Npp32f> src2(width, height, 4);
  NppImageMemory<Npp32f> dst(width, height, 4);

  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaComp_32f_AC4R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(),
                                         dst.get(), dst.step(), roi, param.alphaOp, ctx);
  } else {
    status = nppiAlphaComp_32f_AC4R(src1.get(), src1.step(), src2.get(), src2.step(),
                                     dst.get(), dst.step(), roi, param.alphaOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaComp32fAC4R, AlphaComp32fAC4RParamTest,
                         ::testing::Values(AlphaCompParam{32, 32, NPPI_OP_ALPHA_OVER, false, "32x32_Over_noCtx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_OVER, true, "32x32_Over_Ctx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_IN, false, "32x32_In_noCtx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_OUT, false, "32x32_Out_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaCompParam> &info) { return info.param.name; });

// ==================== AlphaComp 32s AC4R TEST_P ====================

class AlphaComp32sAC4RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaCompParam> {};

TEST_P(AlphaComp32sAC4RParamTest, AlphaComp_32s_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;

  std::vector<Npp32s> src1Data(width * height * channels);
  std::vector<Npp32s> src2Data(width * height * channels);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp32s>(0), static_cast<Npp32s>(255), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp32s>(0), static_cast<Npp32s>(255), 54321);

  NppImageMemory<Npp32s> src1(width, height, 4);
  NppImageMemory<Npp32s> src2(width, height, 4);
  NppImageMemory<Npp32s> dst(width, height, 4);

  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaComp_32s_AC4R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(),
                                         dst.get(), dst.step(), roi, param.alphaOp, ctx);
  } else {
    status = nppiAlphaComp_32s_AC4R(src1.get(), src1.step(), src2.get(), src2.step(),
                                     dst.get(), dst.step(), roi, param.alphaOp);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaComp32sAC4R, AlphaComp32sAC4RParamTest,
                         ::testing::Values(AlphaCompParam{32, 32, NPPI_OP_ALPHA_OVER, false, "32x32_Over_noCtx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_OVER, true, "32x32_Over_Ctx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_IN, false, "32x32_In_noCtx"},
                                           AlphaCompParam{32, 32, NPPI_OP_ALPHA_OUT, false, "32x32_Out_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaCompParam> &info) { return info.param.name; });
