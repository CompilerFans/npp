// RShiftC 8s/16s parameterized tests for nppi arithmetic operations
#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Common Param Structures ====================

struct RShiftCParam {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

// Helper to allocate 8s memory with step
static Npp8s *allocate8s(int width, int height, int channels, int &step) {
  size_t pitch;
  Npp8s *ptr;
  cudaMallocPitch(&ptr, &pitch, width * channels * sizeof(Npp8s), height);
  step = static_cast<int>(pitch);
  return ptr;
}

// ==================== RShiftC 8s C1R TEST_P ====================

class RShiftC8sC1RParamTest : public NppTestBase, public ::testing::WithParamInterface<RShiftCParam> {};

TEST_P(RShiftC8sC1RParamTest, RShiftC_8s_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int srcStep, dstStep;
  Npp8s *d_src = allocate8s(width, height, 1, srcStep);
  Npp8s *d_dst = allocate8s(width, height, 1, dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp8s> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8s>(-128), static_cast<Npp8s>(127), 12345);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * sizeof(Npp8s), width * sizeof(Npp8s), height,
               cudaMemcpyHostToDevice);

  Npp32u nConstant = 2;
  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiRShiftC_8s_C1R_Ctx(d_src, srcStep, nConstant, d_dst, dstStep, roi, ctx);
  } else {
    status = nppiRShiftC_8s_C1R(d_src, srcStep, nConstant, d_dst, dstStep, roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  cudaFree(d_src);
  cudaFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(RShiftC8sC1R, RShiftC8sC1RParamTest,
                         ::testing::Values(RShiftCParam{32, 32, false, "32x32_noCtx"},
                                           RShiftCParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<RShiftCParam> &info) { return info.param.name; });

// ==================== RShiftC 8s C1IR TEST_P ====================

class RShiftC8sC1IRParamTest : public NppTestBase, public ::testing::WithParamInterface<RShiftCParam> {};

TEST_P(RShiftC8sC1IRParamTest, RShiftC_8s_C1IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int dstStep;
  Npp8s *d_dst = allocate8s(width, height, 1, dstStep);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp8s> dstData(width * height);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp8s>(-128), static_cast<Npp8s>(127), 12345);
  cudaMemcpy2D(d_dst, dstStep, dstData.data(), width * sizeof(Npp8s), width * sizeof(Npp8s), height,
               cudaMemcpyHostToDevice);

  Npp32u nConstant = 2;
  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiRShiftC_8s_C1IR_Ctx(nConstant, d_dst, dstStep, roi, ctx);
  } else {
    status = nppiRShiftC_8s_C1IR(nConstant, d_dst, dstStep, roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  cudaFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(RShiftC8sC1IR, RShiftC8sC1IRParamTest,
                         ::testing::Values(RShiftCParam{32, 32, false, "32x32_noCtx"},
                                           RShiftCParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<RShiftCParam> &info) { return info.param.name; });

// ==================== RShiftC 8s C3R TEST_P ====================

class RShiftC8sC3RParamTest : public NppTestBase, public ::testing::WithParamInterface<RShiftCParam> {};

TEST_P(RShiftC8sC3RParamTest, RShiftC_8s_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int srcStep, dstStep;
  Npp8s *d_src = allocate8s(width, height, 3, srcStep);
  Npp8s *d_dst = allocate8s(width, height, 3, dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp8s> srcData(width * height * 3);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8s>(-128), static_cast<Npp8s>(127), 12345);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * 3 * sizeof(Npp8s), width * 3 * sizeof(Npp8s), height,
               cudaMemcpyHostToDevice);

  Npp32u aConstants[3] = {1, 2, 3};
  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiRShiftC_8s_C3R_Ctx(d_src, srcStep, aConstants, d_dst, dstStep, roi, ctx);
  } else {
    status = nppiRShiftC_8s_C3R(d_src, srcStep, aConstants, d_dst, dstStep, roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  cudaFree(d_src);
  cudaFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(RShiftC8sC3R, RShiftC8sC3RParamTest,
                         ::testing::Values(RShiftCParam{32, 32, false, "32x32_noCtx"},
                                           RShiftCParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<RShiftCParam> &info) { return info.param.name; });

// ==================== RShiftC 8s C3IR TEST_P ====================

class RShiftC8sC3IRParamTest : public NppTestBase, public ::testing::WithParamInterface<RShiftCParam> {};

TEST_P(RShiftC8sC3IRParamTest, RShiftC_8s_C3IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int dstStep;
  Npp8s *d_dst = allocate8s(width, height, 3, dstStep);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp8s> dstData(width * height * 3);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp8s>(-128), static_cast<Npp8s>(127), 12345);
  cudaMemcpy2D(d_dst, dstStep, dstData.data(), width * 3 * sizeof(Npp8s), width * 3 * sizeof(Npp8s), height,
               cudaMemcpyHostToDevice);

  Npp32u aConstants[3] = {1, 2, 3};
  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiRShiftC_8s_C3IR_Ctx(aConstants, d_dst, dstStep, roi, ctx);
  } else {
    status = nppiRShiftC_8s_C3IR(aConstants, d_dst, dstStep, roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  cudaFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(RShiftC8sC3IR, RShiftC8sC3IRParamTest,
                         ::testing::Values(RShiftCParam{32, 32, false, "32x32_noCtx"},
                                           RShiftCParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<RShiftCParam> &info) { return info.param.name; });

// ==================== RShiftC 8s C4R TEST_P ====================

class RShiftC8sC4RParamTest : public NppTestBase, public ::testing::WithParamInterface<RShiftCParam> {};

TEST_P(RShiftC8sC4RParamTest, RShiftC_8s_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int srcStep, dstStep;
  Npp8s *d_src = allocate8s(width, height, 4, srcStep);
  Npp8s *d_dst = allocate8s(width, height, 4, dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp8s> srcData(width * height * 4);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8s>(-128), static_cast<Npp8s>(127), 12345);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * 4 * sizeof(Npp8s), width * 4 * sizeof(Npp8s), height,
               cudaMemcpyHostToDevice);

  Npp32u aConstants[4] = {1, 2, 3, 4};
  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiRShiftC_8s_C4R_Ctx(d_src, srcStep, aConstants, d_dst, dstStep, roi, ctx);
  } else {
    status = nppiRShiftC_8s_C4R(d_src, srcStep, aConstants, d_dst, dstStep, roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  cudaFree(d_src);
  cudaFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(RShiftC8sC4R, RShiftC8sC4RParamTest,
                         ::testing::Values(RShiftCParam{32, 32, false, "32x32_noCtx"},
                                           RShiftCParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<RShiftCParam> &info) { return info.param.name; });

// ==================== RShiftC 8s C4IR TEST_P ====================

class RShiftC8sC4IRParamTest : public NppTestBase, public ::testing::WithParamInterface<RShiftCParam> {};

TEST_P(RShiftC8sC4IRParamTest, RShiftC_8s_C4IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int dstStep;
  Npp8s *d_dst = allocate8s(width, height, 4, dstStep);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp8s> dstData(width * height * 4);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp8s>(-128), static_cast<Npp8s>(127), 12345);
  cudaMemcpy2D(d_dst, dstStep, dstData.data(), width * 4 * sizeof(Npp8s), width * 4 * sizeof(Npp8s), height,
               cudaMemcpyHostToDevice);

  Npp32u aConstants[4] = {1, 2, 3, 4};
  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiRShiftC_8s_C4IR_Ctx(aConstants, d_dst, dstStep, roi, ctx);
  } else {
    status = nppiRShiftC_8s_C4IR(aConstants, d_dst, dstStep, roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  cudaFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(RShiftC8sC4IR, RShiftC8sC4IRParamTest,
                         ::testing::Values(RShiftCParam{32, 32, false, "32x32_noCtx"},
                                           RShiftCParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<RShiftCParam> &info) { return info.param.name; });

// ==================== RShiftC 8s AC4R TEST_P ====================

class RShiftC8sAC4RParamTest : public NppTestBase, public ::testing::WithParamInterface<RShiftCParam> {};

TEST_P(RShiftC8sAC4RParamTest, RShiftC_8s_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int srcStep, dstStep;
  Npp8s *d_src = allocate8s(width, height, 4, srcStep);
  Npp8s *d_dst = allocate8s(width, height, 4, dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp8s> srcData(width * height * 4);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8s>(-128), static_cast<Npp8s>(127), 12345);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * 4 * sizeof(Npp8s), width * 4 * sizeof(Npp8s), height,
               cudaMemcpyHostToDevice);

  Npp32u aConstants[3] = {1, 2, 3};
  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiRShiftC_8s_AC4R_Ctx(d_src, srcStep, aConstants, d_dst, dstStep, roi, ctx);
  } else {
    status = nppiRShiftC_8s_AC4R(d_src, srcStep, aConstants, d_dst, dstStep, roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  cudaFree(d_src);
  cudaFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(RShiftC8sAC4R, RShiftC8sAC4RParamTest,
                         ::testing::Values(RShiftCParam{32, 32, false, "32x32_noCtx"},
                                           RShiftCParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<RShiftCParam> &info) { return info.param.name; });

// ==================== RShiftC 8s AC4IR TEST_P ====================

class RShiftC8sAC4IRParamTest : public NppTestBase, public ::testing::WithParamInterface<RShiftCParam> {};

TEST_P(RShiftC8sAC4IRParamTest, RShiftC_8s_AC4IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int dstStep;
  Npp8s *d_dst = allocate8s(width, height, 4, dstStep);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp8s> dstData(width * height * 4);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp8s>(-128), static_cast<Npp8s>(127), 12345);
  cudaMemcpy2D(d_dst, dstStep, dstData.data(), width * 4 * sizeof(Npp8s), width * 4 * sizeof(Npp8s), height,
               cudaMemcpyHostToDevice);

  Npp32u aConstants[3] = {1, 2, 3};
  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiRShiftC_8s_AC4IR_Ctx(aConstants, d_dst, dstStep, roi, ctx);
  } else {
    status = nppiRShiftC_8s_AC4IR(aConstants, d_dst, dstStep, roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  cudaFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(RShiftC8sAC4IR, RShiftC8sAC4IRParamTest,
                         ::testing::Values(RShiftCParam{32, 32, false, "32x32_noCtx"},
                                           RShiftCParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<RShiftCParam> &info) { return info.param.name; });

// ==================== RShiftC 16s C3R TEST_P ====================

class RShiftC16sC3RParamTest : public NppTestBase, public ::testing::WithParamInterface<RShiftCParam> {};

TEST_P(RShiftC16sC3RParamTest, RShiftC_16s_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int srcStep, dstStep;
  Npp16s *d_src = nppiMalloc_16s_C1(width * 3, height, &srcStep);
  Npp16s *d_dst = nppiMalloc_16s_C1(width * 3, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> srcData(width * height * 3);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(-32768), static_cast<Npp16s>(32767), 12345);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * 3 * sizeof(Npp16s), width * 3 * sizeof(Npp16s), height,
               cudaMemcpyHostToDevice);

  Npp32u aConstants[3] = {1, 2, 3};
  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiRShiftC_16s_C3R_Ctx(d_src, srcStep, aConstants, d_dst, dstStep, roi, ctx);
  } else {
    status = nppiRShiftC_16s_C3R(d_src, srcStep, aConstants, d_dst, dstStep, roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  nppiFree(d_src);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(RShiftC16sC3R, RShiftC16sC3RParamTest,
                         ::testing::Values(RShiftCParam{32, 32, false, "32x32_noCtx"},
                                           RShiftCParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<RShiftCParam> &info) { return info.param.name; });

// ==================== RShiftC 16s C3IR TEST_P ====================

class RShiftC16sC3IRParamTest : public NppTestBase, public ::testing::WithParamInterface<RShiftCParam> {};

TEST_P(RShiftC16sC3IRParamTest, RShiftC_16s_C3IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int dstStep;
  Npp16s *d_dst = nppiMalloc_16s_C1(width * 3, height, &dstStep);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> dstData(width * height * 3);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp16s>(-32768), static_cast<Npp16s>(32767), 12345);
  cudaMemcpy2D(d_dst, dstStep, dstData.data(), width * 3 * sizeof(Npp16s), width * 3 * sizeof(Npp16s), height,
               cudaMemcpyHostToDevice);

  Npp32u aConstants[3] = {1, 2, 3};
  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiRShiftC_16s_C3IR_Ctx(aConstants, d_dst, dstStep, roi, ctx);
  } else {
    status = nppiRShiftC_16s_C3IR(aConstants, d_dst, dstStep, roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(RShiftC16sC3IR, RShiftC16sC3IRParamTest,
                         ::testing::Values(RShiftCParam{32, 32, false, "32x32_noCtx"},
                                           RShiftCParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<RShiftCParam> &info) { return info.param.name; });

// ==================== RShiftC 16s C4R TEST_P ====================

class RShiftC16sC4RParamTest : public NppTestBase, public ::testing::WithParamInterface<RShiftCParam> {};

TEST_P(RShiftC16sC4RParamTest, RShiftC_16s_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int srcStep, dstStep;
  Npp16s *d_src = nppiMalloc_16s_C4(width, height, &srcStep);
  Npp16s *d_dst = nppiMalloc_16s_C4(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> srcData(width * height * 4);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(-32768), static_cast<Npp16s>(32767), 12345);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * 4 * sizeof(Npp16s), width * 4 * sizeof(Npp16s), height,
               cudaMemcpyHostToDevice);

  Npp32u aConstants[4] = {1, 2, 3, 4};
  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiRShiftC_16s_C4R_Ctx(d_src, srcStep, aConstants, d_dst, dstStep, roi, ctx);
  } else {
    status = nppiRShiftC_16s_C4R(d_src, srcStep, aConstants, d_dst, dstStep, roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  nppiFree(d_src);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(RShiftC16sC4R, RShiftC16sC4RParamTest,
                         ::testing::Values(RShiftCParam{32, 32, false, "32x32_noCtx"},
                                           RShiftCParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<RShiftCParam> &info) { return info.param.name; });

// ==================== RShiftC 16s C4IR TEST_P ====================

class RShiftC16sC4IRParamTest : public NppTestBase, public ::testing::WithParamInterface<RShiftCParam> {};

TEST_P(RShiftC16sC4IRParamTest, RShiftC_16s_C4IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int dstStep;
  Npp16s *d_dst = nppiMalloc_16s_C4(width, height, &dstStep);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> dstData(width * height * 4);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp16s>(-32768), static_cast<Npp16s>(32767), 12345);
  cudaMemcpy2D(d_dst, dstStep, dstData.data(), width * 4 * sizeof(Npp16s), width * 4 * sizeof(Npp16s), height,
               cudaMemcpyHostToDevice);

  Npp32u aConstants[4] = {1, 2, 3, 4};
  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiRShiftC_16s_C4IR_Ctx(aConstants, d_dst, dstStep, roi, ctx);
  } else {
    status = nppiRShiftC_16s_C4IR(aConstants, d_dst, dstStep, roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(RShiftC16sC4IR, RShiftC16sC4IRParamTest,
                         ::testing::Values(RShiftCParam{32, 32, false, "32x32_noCtx"},
                                           RShiftCParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<RShiftCParam> &info) { return info.param.name; });

// ==================== RShiftC 16s AC4R TEST_P ====================

class RShiftC16sAC4RParamTest : public NppTestBase, public ::testing::WithParamInterface<RShiftCParam> {};

TEST_P(RShiftC16sAC4RParamTest, RShiftC_16s_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int srcStep, dstStep;
  Npp16s *d_src = nppiMalloc_16s_C4(width, height, &srcStep);
  Npp16s *d_dst = nppiMalloc_16s_C4(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> srcData(width * height * 4);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(-32768), static_cast<Npp16s>(32767), 12345);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * 4 * sizeof(Npp16s), width * 4 * sizeof(Npp16s), height,
               cudaMemcpyHostToDevice);

  Npp32u aConstants[3] = {1, 2, 3};
  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiRShiftC_16s_AC4R_Ctx(d_src, srcStep, aConstants, d_dst, dstStep, roi, ctx);
  } else {
    status = nppiRShiftC_16s_AC4R(d_src, srcStep, aConstants, d_dst, dstStep, roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  nppiFree(d_src);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(RShiftC16sAC4R, RShiftC16sAC4RParamTest,
                         ::testing::Values(RShiftCParam{32, 32, false, "32x32_noCtx"},
                                           RShiftCParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<RShiftCParam> &info) { return info.param.name; });

// ==================== RShiftC 16s AC4IR TEST_P ====================

class RShiftC16sAC4IRParamTest : public NppTestBase, public ::testing::WithParamInterface<RShiftCParam> {};

TEST_P(RShiftC16sAC4IRParamTest, RShiftC_16s_AC4IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int dstStep;
  Npp16s *d_dst = nppiMalloc_16s_C4(width, height, &dstStep);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> dstData(width * height * 4);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp16s>(-32768), static_cast<Npp16s>(32767), 12345);
  cudaMemcpy2D(d_dst, dstStep, dstData.data(), width * 4 * sizeof(Npp16s), width * 4 * sizeof(Npp16s), height,
               cudaMemcpyHostToDevice);

  Npp32u aConstants[3] = {1, 2, 3};
  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiRShiftC_16s_AC4IR_Ctx(aConstants, d_dst, dstStep, roi, ctx);
  } else {
    status = nppiRShiftC_16s_AC4IR(aConstants, d_dst, dstStep, roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(RShiftC16sAC4IR, RShiftC16sAC4IRParamTest,
                         ::testing::Values(RShiftCParam{32, 32, false, "32x32_noCtx"},
                                           RShiftCParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<RShiftCParam> &info) { return info.param.name; });
