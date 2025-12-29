// AddC/SubC 16s C1 parameterized tests for nppi arithmetic operations
#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Common Param Structures ====================

struct AddSubC16sParam {
  int width;
  int height;
  int scaleFactor;
  bool use_ctx;
  std::string name;
};

// ==================== AddC 16s C1RSfs TEST_P ====================

class AddC16sC1RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<AddSubC16sParam> {};

TEST_P(AddC16sC1RSfsParamTest, AddC_16s_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int srcStep, dstStep;
  Npp16s *d_src = nppiMalloc_16s_C1(width, height, &srcStep);
  Npp16s *d_dst = nppiMalloc_16s_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(-1000), static_cast<Npp16s>(1000), 12345);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * sizeof(Npp16s), width * sizeof(Npp16s), height,
               cudaMemcpyHostToDevice);

  Npp16s nConstant = 100;
  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAddC_16s_C1RSfs_Ctx(d_src, srcStep, nConstant, d_dst, dstStep, roi, param.scaleFactor, ctx);
  } else {
    status = nppiAddC_16s_C1RSfs(d_src, srcStep, nConstant, d_dst, dstStep, roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  nppiFree(d_src);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(AddC16sC1RSfs, AddC16sC1RSfsParamTest,
                         ::testing::Values(AddSubC16sParam{32, 32, 0, false, "32x32_noCtx"},
                                           AddSubC16sParam{32, 32, 0, true, "32x32_Ctx"},
                                           AddSubC16sParam{64, 64, 2, false, "64x64_s2"}),
                         [](const ::testing::TestParamInfo<AddSubC16sParam> &info) { return info.param.name; });

// ==================== SubC 16s C1RSfs TEST_P ====================

class SubC16sC1RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<AddSubC16sParam> {};

TEST_P(SubC16sC1RSfsParamTest, SubC_16s_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int srcStep, dstStep;
  Npp16s *d_src = nppiMalloc_16s_C1(width, height, &srcStep);
  Npp16s *d_dst = nppiMalloc_16s_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(-1000), static_cast<Npp16s>(1000), 12345);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * sizeof(Npp16s), width * sizeof(Npp16s), height,
               cudaMemcpyHostToDevice);

  Npp16s nConstant = 50;
  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSubC_16s_C1RSfs_Ctx(d_src, srcStep, nConstant, d_dst, dstStep, roi, param.scaleFactor, ctx);
  } else {
    status = nppiSubC_16s_C1RSfs(d_src, srcStep, nConstant, d_dst, dstStep, roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  nppiFree(d_src);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(SubC16sC1RSfs, SubC16sC1RSfsParamTest,
                         ::testing::Values(AddSubC16sParam{32, 32, 0, false, "32x32_noCtx"},
                                           AddSubC16sParam{32, 32, 0, true, "32x32_Ctx"},
                                           AddSubC16sParam{64, 64, 2, false, "64x64_s2"}),
                         [](const ::testing::TestParamInfo<AddSubC16sParam> &info) { return info.param.name; });
