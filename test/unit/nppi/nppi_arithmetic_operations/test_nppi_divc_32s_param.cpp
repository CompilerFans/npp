// DivC 32s C1/C3 RSfs parameterized tests for nppi arithmetic operations
#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Common Param Structures ====================

struct DivC32sParam {
  int width;
  int height;
  int scaleFactor;
  bool use_ctx;
  std::string name;
};

// ==================== DivC 32s C1RSfs TEST_P ====================

class DivC32sC1RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DivC32sParam> {};

TEST_P(DivC32sC1RSfsParamTest, DivC_32s_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int srcStep, dstStep;
  Npp32s *d_src = nppiMalloc_32s_C1(width, height, &srcStep);
  Npp32s *d_dst = nppiMalloc_32s_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp32s> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp32s>(-10000), static_cast<Npp32s>(10000), 12345);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * sizeof(Npp32s), width * sizeof(Npp32s), height,
               cudaMemcpyHostToDevice);

  Npp32s nConstant = 10;
  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiDivC_32s_C1RSfs_Ctx(d_src, srcStep, nConstant, d_dst, dstStep, roi, param.scaleFactor, ctx);
  } else {
    status = nppiDivC_32s_C1RSfs(d_src, srcStep, nConstant, d_dst, dstStep, roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  nppiFree(d_src);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(DivC32sC1RSfs, DivC32sC1RSfsParamTest,
                         ::testing::Values(DivC32sParam{32, 32, 0, false, "32x32_noCtx"},
                                           DivC32sParam{32, 32, 0, true, "32x32_Ctx"},
                                           DivC32sParam{64, 64, 2, false, "64x64_s2"}),
                         [](const ::testing::TestParamInfo<DivC32sParam> &info) { return info.param.name; });

// ==================== DivC 32s C3RSfs TEST_P ====================

class DivC32sC3RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DivC32sParam> {};

TEST_P(DivC32sC3RSfsParamTest, DivC_32s_C3RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int srcStep, dstStep;
  Npp32s *d_src = nppiMalloc_32s_C3(width, height, &srcStep);
  Npp32s *d_dst = nppiMalloc_32s_C3(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  const int total = width * height * 3;
  std::vector<Npp32s> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp32s>(-10000), static_cast<Npp32s>(10000), 12345);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * 3 * sizeof(Npp32s), width * 3 * sizeof(Npp32s), height,
               cudaMemcpyHostToDevice);

  Npp32s aConstants[3] = {5, 10, 15};
  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiDivC_32s_C3RSfs_Ctx(d_src, srcStep, aConstants, d_dst, dstStep, roi, param.scaleFactor, ctx);
  } else {
    status = nppiDivC_32s_C3RSfs(d_src, srcStep, aConstants, d_dst, dstStep, roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  nppiFree(d_src);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(DivC32sC3RSfs, DivC32sC3RSfsParamTest,
                         ::testing::Values(DivC32sParam{32, 32, 0, false, "32x32_noCtx"},
                                           DivC32sParam{32, 32, 0, true, "32x32_Ctx"},
                                           DivC32sParam{64, 64, 2, false, "64x64_s2"}),
                         [](const ::testing::TestParamInfo<DivC32sParam> &info) { return info.param.name; });
