// Sqr parameterized tests for nppi arithmetic operations
#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Common Param Structures ====================

struct SqrParam {
  int width;
  int height;
  int scaleFactor;
  bool use_ctx;
  std::string name;
};

// ==================== Sqr 8u AC4RSfs TEST_P ====================

class Sqr8uAC4RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrParam> {};

TEST_P(Sqr8uAC4RSfsParamTest, Sqr_8u_AC4RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height * 4);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(1), static_cast<Npp8u>(15), 12345);

  NppImageMemory<Npp8u> src(width, height, 4);
  NppImageMemory<Npp8u> dst(width, height, 4);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqr_8u_AC4RSfs_Ctx(src.get(), src.step(), dst.get(), dst.step(),
                                     roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqr_8u_AC4RSfs(src.get(), src.step(), dst.get(), dst.step(),
                                 roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqr8uAC4RSfs, Sqr8uAC4RSfsParamTest,
                         ::testing::Values(SqrParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrParam> &info) { return info.param.name; });

// ==================== Sqr 8u AC4IRSfs TEST_P ====================

class Sqr8uAC4IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrParam> {};

TEST_P(Sqr8uAC4IRSfsParamTest, Sqr_8u_AC4IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcDstData(width * height * 4);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp8u>(1), static_cast<Npp8u>(15), 12345);

  NppImageMemory<Npp8u> srcDst(width, height, 4);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqr_8u_AC4IRSfs_Ctx(srcDst.get(), srcDst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqr_8u_AC4IRSfs(srcDst.get(), srcDst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqr8uAC4IRSfs, Sqr8uAC4IRSfsParamTest,
                         ::testing::Values(SqrParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrParam> &info) { return info.param.name; });

// ==================== Sqr 16u AC4RSfs TEST_P ====================

class Sqr16uAC4RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrParam> {};

TEST_P(Sqr16uAC4RSfsParamTest, Sqr_16u_AC4RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcData(width * height * 4);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(1), static_cast<Npp16u>(255), 12345);

  NppImageMemory<Npp16u> src(width, height, 4);
  NppImageMemory<Npp16u> dst(width, height, 4);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqr_16u_AC4RSfs_Ctx(src.get(), src.step(), dst.get(), dst.step(),
                                      roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqr_16u_AC4RSfs(src.get(), src.step(), dst.get(), dst.step(),
                                  roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqr16uAC4RSfs, Sqr16uAC4RSfsParamTest,
                         ::testing::Values(SqrParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrParam> &info) { return info.param.name; });

// ==================== Sqr 16u AC4IRSfs TEST_P ====================

class Sqr16uAC4IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrParam> {};

TEST_P(Sqr16uAC4IRSfsParamTest, Sqr_16u_AC4IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcDstData(width * height * 4);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp16u>(1), static_cast<Npp16u>(255), 12345);

  NppImageMemory<Npp16u> srcDst(width, height, 4);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqr_16u_AC4IRSfs_Ctx(srcDst.get(), srcDst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqr_16u_AC4IRSfs(srcDst.get(), srcDst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqr16uAC4IRSfs, Sqr16uAC4IRSfsParamTest,
                         ::testing::Values(SqrParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrParam> &info) { return info.param.name; });

// ==================== Sqr 16s AC4RSfs TEST_P ====================

class Sqr16sAC4RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrParam> {};

TEST_P(Sqr16sAC4RSfsParamTest, Sqr_16s_AC4RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int srcStep, dstStep;
  Npp16s* d_src = nppiMalloc_16s_C1(width * 4, height, &srcStep);
  Npp16s* d_dst = nppiMalloc_16s_C1(width * 4, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> srcData(width * height * 4);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(1), static_cast<Npp16s>(180), 12345);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * 4 * sizeof(Npp16s),
               width * 4 * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqr_16s_AC4RSfs_Ctx(d_src, srcStep, d_dst, dstStep,
                                      roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqr_16s_AC4RSfs(d_src, srcStep, d_dst, dstStep,
                                  roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  nppiFree(d_src);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(Sqr16sAC4RSfs, Sqr16sAC4RSfsParamTest,
                         ::testing::Values(SqrParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrParam> &info) { return info.param.name; });

// ==================== Sqr 16s AC4IRSfs TEST_P ====================

class Sqr16sAC4IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrParam> {};

TEST_P(Sqr16sAC4IRSfsParamTest, Sqr_16s_AC4IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int dstStep;
  Npp16s* d_dst = nppiMalloc_16s_C1(width * 4, height, &dstStep);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> dstData(width * height * 4);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp16s>(1), static_cast<Npp16s>(180), 12345);
  cudaMemcpy2D(d_dst, dstStep, dstData.data(), width * 4 * sizeof(Npp16s),
               width * 4 * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqr_16s_AC4IRSfs_Ctx(d_dst, dstStep, roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqr_16s_AC4IRSfs(d_dst, dstStep, roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(Sqr16sAC4IRSfs, Sqr16sAC4IRSfsParamTest,
                         ::testing::Values(SqrParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrParam> &info) { return info.param.name; });

// ==================== Sqr 32f AC4R TEST_P ====================

class Sqr32fAC4RParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrParam> {};

TEST_P(Sqr32fAC4RParamTest, Sqr_32f_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> srcData(width * height * 4);
  TestDataGenerator::generateRandom(srcData, 1.0f, 10.0f, 12345);

  NppImageMemory<Npp32f> src(width, height, 4);
  NppImageMemory<Npp32f> dst(width, height, 4);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqr_32f_AC4R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiSqr_32f_AC4R(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqr32fAC4R, Sqr32fAC4RParamTest,
                         ::testing::Values(SqrParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrParam> &info) { return info.param.name; });

// ==================== Sqr 32f AC4IR TEST_P ====================

class Sqr32fAC4IRParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrParam> {};

TEST_P(Sqr32fAC4IRParamTest, Sqr_32f_AC4IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> dstData(width * height * 4);
  TestDataGenerator::generateRandom(dstData, 1.0f, 10.0f, 12345);

  NppImageMemory<Npp32f> dst(width, height, 4);
  dst.copyFromHost(dstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqr_32f_AC4IR_Ctx(dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiSqr_32f_AC4IR(dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqr32fAC4IR, Sqr32fAC4IRParamTest,
                         ::testing::Values(SqrParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrParam> &info) { return info.param.name; });
