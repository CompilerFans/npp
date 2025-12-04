// Sqrt 32f multi-channel parameterized tests for nppi arithmetic operations
#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Common Param Structures ====================

struct Sqrt32fMCParam {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

// ==================== Sqrt 32f C1IR TEST_P ====================

class Sqrt32fC1IRParamTest : public NppTestBase, public ::testing::WithParamInterface<Sqrt32fMCParam> {};

TEST_P(Sqrt32fC1IRParamTest, Sqrt_32f_C1IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> srcDstData(width * height);
  TestDataGenerator::generateRandom(srcDstData, 1.0f, 1000.0f, 12345);

  NppImageMemory<Npp32f> srcDst(width, height, 1);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqrt_32f_C1IR_Ctx(srcDst.get(), srcDst.step(), roi, ctx);
  } else {
    status = nppiSqrt_32f_C1IR(srcDst.get(), srcDst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqrt32fC1IR, Sqrt32fC1IRParamTest,
                         ::testing::Values(Sqrt32fMCParam{32, 32, false, "32x32_noCtx"},
                                           Sqrt32fMCParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<Sqrt32fMCParam> &info) { return info.param.name; });

// ==================== Sqrt 32f C3R TEST_P ====================

class Sqrt32fC3RParamTest : public NppTestBase, public ::testing::WithParamInterface<Sqrt32fMCParam> {};

TEST_P(Sqrt32fC3RParamTest, Sqrt_32f_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> srcData(width * height * 3);
  TestDataGenerator::generateRandom(srcData, 1.0f, 1000.0f, 12345);

  NppImageMemory<Npp32f> src(width, height, 3);
  NppImageMemory<Npp32f> dst(width, height, 3);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqrt_32f_C3R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiSqrt_32f_C3R(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqrt32fC3R, Sqrt32fC3RParamTest,
                         ::testing::Values(Sqrt32fMCParam{32, 32, false, "32x32_noCtx"},
                                           Sqrt32fMCParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<Sqrt32fMCParam> &info) { return info.param.name; });

// ==================== Sqrt 32f C3IR TEST_P ====================

class Sqrt32fC3IRParamTest : public NppTestBase, public ::testing::WithParamInterface<Sqrt32fMCParam> {};

TEST_P(Sqrt32fC3IRParamTest, Sqrt_32f_C3IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> srcDstData(width * height * 3);
  TestDataGenerator::generateRandom(srcDstData, 1.0f, 1000.0f, 12345);

  NppImageMemory<Npp32f> srcDst(width, height, 3);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqrt_32f_C3IR_Ctx(srcDst.get(), srcDst.step(), roi, ctx);
  } else {
    status = nppiSqrt_32f_C3IR(srcDst.get(), srcDst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqrt32fC3IR, Sqrt32fC3IRParamTest,
                         ::testing::Values(Sqrt32fMCParam{32, 32, false, "32x32_noCtx"},
                                           Sqrt32fMCParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<Sqrt32fMCParam> &info) { return info.param.name; });

// ==================== Sqrt 32f C4R TEST_P ====================

class Sqrt32fC4RParamTest : public NppTestBase, public ::testing::WithParamInterface<Sqrt32fMCParam> {};

TEST_P(Sqrt32fC4RParamTest, Sqrt_32f_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> srcData(width * height * 4);
  TestDataGenerator::generateRandom(srcData, 1.0f, 1000.0f, 12345);

  NppImageMemory<Npp32f> src(width, height, 4);
  NppImageMemory<Npp32f> dst(width, height, 4);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqrt_32f_C4R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiSqrt_32f_C4R(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqrt32fC4R, Sqrt32fC4RParamTest,
                         ::testing::Values(Sqrt32fMCParam{32, 32, false, "32x32_noCtx"},
                                           Sqrt32fMCParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<Sqrt32fMCParam> &info) { return info.param.name; });

// ==================== Sqrt 32f C4IR TEST_P ====================

class Sqrt32fC4IRParamTest : public NppTestBase, public ::testing::WithParamInterface<Sqrt32fMCParam> {};

TEST_P(Sqrt32fC4IRParamTest, Sqrt_32f_C4IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> srcDstData(width * height * 4);
  TestDataGenerator::generateRandom(srcDstData, 1.0f, 1000.0f, 12345);

  NppImageMemory<Npp32f> srcDst(width, height, 4);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqrt_32f_C4IR_Ctx(srcDst.get(), srcDst.step(), roi, ctx);
  } else {
    status = nppiSqrt_32f_C4IR(srcDst.get(), srcDst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqrt32fC4IR, Sqrt32fC4IRParamTest,
                         ::testing::Values(Sqrt32fMCParam{32, 32, false, "32x32_noCtx"},
                                           Sqrt32fMCParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<Sqrt32fMCParam> &info) { return info.param.name; });
