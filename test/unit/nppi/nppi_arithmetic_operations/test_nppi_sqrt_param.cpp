// Sqrt parameterized tests for nppi arithmetic operations
#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Common Param Structures ====================

struct SqrtParam {
  int width;
  int height;
  int scaleFactor;
  bool use_ctx;
  std::string name;
};

// ==================== Sqrt 8u C1RSfs TEST_P ====================

class Sqrt8uC1RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrtParam> {};

TEST_P(Sqrt8uC1RSfsParamTest, Sqrt_8u_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(1), static_cast<Npp8u>(255), 12345);

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqrt_8u_C1RSfs_Ctx(src.get(), src.step(), dst.get(), dst.step(),
                                     roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqrt_8u_C1RSfs(src.get(), src.step(), dst.get(), dst.step(),
                                 roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqrt8uC1RSfs, Sqrt8uC1RSfsParamTest,
                         ::testing::Values(SqrtParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrtParam{32, 32, 0, true, "32x32_Ctx"},
                                           SqrtParam{64, 64, 1, false, "64x64_sf1_noCtx"}),
                         [](const ::testing::TestParamInfo<SqrtParam> &info) { return info.param.name; });

// ==================== Sqrt 8u C1IRSfs TEST_P ====================

class Sqrt8uC1IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrtParam> {};

TEST_P(Sqrt8uC1IRSfsParamTest, Sqrt_8u_C1IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcDstData(width * height);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp8u>(1), static_cast<Npp8u>(255), 12345);

  NppImageMemory<Npp8u> srcDst(width, height);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqrt_8u_C1IRSfs_Ctx(srcDst.get(), srcDst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqrt_8u_C1IRSfs(srcDst.get(), srcDst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqrt8uC1IRSfs, Sqrt8uC1IRSfsParamTest,
                         ::testing::Values(SqrtParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrtParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrtParam> &info) { return info.param.name; });

// ==================== Sqrt 8u C3RSfs TEST_P ====================

class Sqrt8uC3RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrtParam> {};

TEST_P(Sqrt8uC3RSfsParamTest, Sqrt_8u_C3RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height * 3);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(1), static_cast<Npp8u>(255), 12345);

  NppImageMemory<Npp8u> src(width, height, 3);
  NppImageMemory<Npp8u> dst(width, height, 3);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqrt_8u_C3RSfs_Ctx(src.get(), src.step(), dst.get(), dst.step(),
                                     roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqrt_8u_C3RSfs(src.get(), src.step(), dst.get(), dst.step(),
                                 roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqrt8uC3RSfs, Sqrt8uC3RSfsParamTest,
                         ::testing::Values(SqrtParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrtParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrtParam> &info) { return info.param.name; });

// ==================== Sqrt 8u C3IRSfs TEST_P ====================

class Sqrt8uC3IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrtParam> {};

TEST_P(Sqrt8uC3IRSfsParamTest, Sqrt_8u_C3IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcDstData(width * height * 3);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp8u>(1), static_cast<Npp8u>(255), 12345);

  NppImageMemory<Npp8u> srcDst(width, height, 3);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqrt_8u_C3IRSfs_Ctx(srcDst.get(), srcDst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqrt_8u_C3IRSfs(srcDst.get(), srcDst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqrt8uC3IRSfs, Sqrt8uC3IRSfsParamTest,
                         ::testing::Values(SqrtParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrtParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrtParam> &info) { return info.param.name; });

// ==================== Sqrt 8u AC4RSfs TEST_P ====================

class Sqrt8uAC4RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrtParam> {};

TEST_P(Sqrt8uAC4RSfsParamTest, Sqrt_8u_AC4RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height * 4);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(1), static_cast<Npp8u>(255), 12345);

  NppImageMemory<Npp8u> src(width, height, 4);
  NppImageMemory<Npp8u> dst(width, height, 4);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqrt_8u_AC4RSfs_Ctx(src.get(), src.step(), dst.get(), dst.step(),
                                      roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqrt_8u_AC4RSfs(src.get(), src.step(), dst.get(), dst.step(),
                                  roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqrt8uAC4RSfs, Sqrt8uAC4RSfsParamTest,
                         ::testing::Values(SqrtParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrtParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrtParam> &info) { return info.param.name; });

// ==================== Sqrt 8u AC4IRSfs TEST_P ====================

class Sqrt8uAC4IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrtParam> {};

TEST_P(Sqrt8uAC4IRSfsParamTest, Sqrt_8u_AC4IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcDstData(width * height * 4);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp8u>(1), static_cast<Npp8u>(255), 12345);

  NppImageMemory<Npp8u> srcDst(width, height, 4);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqrt_8u_AC4IRSfs_Ctx(srcDst.get(), srcDst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqrt_8u_AC4IRSfs(srcDst.get(), srcDst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqrt8uAC4IRSfs, Sqrt8uAC4IRSfsParamTest,
                         ::testing::Values(SqrtParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrtParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrtParam> &info) { return info.param.name; });

// ==================== Sqrt 16u C1RSfs TEST_P ====================

class Sqrt16uC1RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrtParam> {};

TEST_P(Sqrt16uC1RSfsParamTest, Sqrt_16u_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(1), static_cast<Npp16u>(65535), 12345);

  NppImageMemory<Npp16u> src(width, height);
  NppImageMemory<Npp16u> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqrt_16u_C1RSfs_Ctx(src.get(), src.step(), dst.get(), dst.step(),
                                      roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqrt_16u_C1RSfs(src.get(), src.step(), dst.get(), dst.step(),
                                  roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqrt16uC1RSfs, Sqrt16uC1RSfsParamTest,
                         ::testing::Values(SqrtParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrtParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrtParam> &info) { return info.param.name; });

// ==================== Sqrt 16u C1IRSfs TEST_P ====================

class Sqrt16uC1IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrtParam> {};

TEST_P(Sqrt16uC1IRSfsParamTest, Sqrt_16u_C1IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcDstData(width * height);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp16u>(1), static_cast<Npp16u>(65535), 12345);

  NppImageMemory<Npp16u> srcDst(width, height);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqrt_16u_C1IRSfs_Ctx(srcDst.get(), srcDst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqrt_16u_C1IRSfs(srcDst.get(), srcDst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqrt16uC1IRSfs, Sqrt16uC1IRSfsParamTest,
                         ::testing::Values(SqrtParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrtParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrtParam> &info) { return info.param.name; });

// ==================== Sqrt 16u C3RSfs TEST_P ====================

class Sqrt16uC3RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrtParam> {};

TEST_P(Sqrt16uC3RSfsParamTest, Sqrt_16u_C3RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcData(width * height * 3);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(1), static_cast<Npp16u>(65535), 12345);

  NppImageMemory<Npp16u> src(width, height, 3);
  NppImageMemory<Npp16u> dst(width, height, 3);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqrt_16u_C3RSfs_Ctx(src.get(), src.step(), dst.get(), dst.step(),
                                      roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqrt_16u_C3RSfs(src.get(), src.step(), dst.get(), dst.step(),
                                  roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqrt16uC3RSfs, Sqrt16uC3RSfsParamTest,
                         ::testing::Values(SqrtParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrtParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrtParam> &info) { return info.param.name; });

// ==================== Sqrt 16u C3IRSfs TEST_P ====================

class Sqrt16uC3IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrtParam> {};

TEST_P(Sqrt16uC3IRSfsParamTest, Sqrt_16u_C3IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcDstData(width * height * 3);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp16u>(1), static_cast<Npp16u>(65535), 12345);

  NppImageMemory<Npp16u> srcDst(width, height, 3);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqrt_16u_C3IRSfs_Ctx(srcDst.get(), srcDst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqrt_16u_C3IRSfs(srcDst.get(), srcDst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqrt16uC3IRSfs, Sqrt16uC3IRSfsParamTest,
                         ::testing::Values(SqrtParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrtParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrtParam> &info) { return info.param.name; });

// ==================== Sqrt 16u AC4RSfs TEST_P ====================

class Sqrt16uAC4RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrtParam> {};

TEST_P(Sqrt16uAC4RSfsParamTest, Sqrt_16u_AC4RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcData(width * height * 4);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(1), static_cast<Npp16u>(65535), 12345);

  NppImageMemory<Npp16u> src(width, height, 4);
  NppImageMemory<Npp16u> dst(width, height, 4);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqrt_16u_AC4RSfs_Ctx(src.get(), src.step(), dst.get(), dst.step(),
                                       roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqrt_16u_AC4RSfs(src.get(), src.step(), dst.get(), dst.step(),
                                   roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqrt16uAC4RSfs, Sqrt16uAC4RSfsParamTest,
                         ::testing::Values(SqrtParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrtParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrtParam> &info) { return info.param.name; });

// ==================== Sqrt 16u AC4IRSfs TEST_P ====================

class Sqrt16uAC4IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrtParam> {};

TEST_P(Sqrt16uAC4IRSfsParamTest, Sqrt_16u_AC4IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcDstData(width * height * 4);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp16u>(1), static_cast<Npp16u>(65535), 12345);

  NppImageMemory<Npp16u> srcDst(width, height, 4);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqrt_16u_AC4IRSfs_Ctx(srcDst.get(), srcDst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqrt_16u_AC4IRSfs(srcDst.get(), srcDst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqrt16uAC4IRSfs, Sqrt16uAC4IRSfsParamTest,
                         ::testing::Values(SqrtParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrtParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrtParam> &info) { return info.param.name; });

// ==================== Sqrt 16s C1RSfs TEST_P ====================

class Sqrt16sC1RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrtParam> {};

TEST_P(Sqrt16sC1RSfsParamTest, Sqrt_16s_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int srcStep, dstStep;
  Npp16s* d_src = nppiMalloc_16s_C1(width, height, &srcStep);
  Npp16s* d_dst = nppiMalloc_16s_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(1), static_cast<Npp16s>(32767), 12345);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * sizeof(Npp16s),
               width * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqrt_16s_C1RSfs_Ctx(d_src, srcStep, d_dst, dstStep,
                                      roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqrt_16s_C1RSfs(d_src, srcStep, d_dst, dstStep,
                                  roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  nppiFree(d_src);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(Sqrt16sC1RSfs, Sqrt16sC1RSfsParamTest,
                         ::testing::Values(SqrtParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrtParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrtParam> &info) { return info.param.name; });

// ==================== Sqrt 16s C1IRSfs TEST_P ====================

class Sqrt16sC1IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrtParam> {};

TEST_P(Sqrt16sC1IRSfsParamTest, Sqrt_16s_C1IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int dstStep;
  Npp16s* d_dst = nppiMalloc_16s_C1(width, height, &dstStep);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> dstData(width * height);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp16s>(1), static_cast<Npp16s>(32767), 12345);
  cudaMemcpy2D(d_dst, dstStep, dstData.data(), width * sizeof(Npp16s),
               width * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqrt_16s_C1IRSfs_Ctx(d_dst, dstStep, roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqrt_16s_C1IRSfs(d_dst, dstStep, roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(Sqrt16sC1IRSfs, Sqrt16sC1IRSfsParamTest,
                         ::testing::Values(SqrtParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrtParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrtParam> &info) { return info.param.name; });

// ==================== Sqrt 16s C3RSfs TEST_P ====================

class Sqrt16sC3RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrtParam> {};

TEST_P(Sqrt16sC3RSfsParamTest, Sqrt_16s_C3RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int srcStep, dstStep;
  Npp16s* d_src = nppiMalloc_16s_C1(width * 3, height, &srcStep);
  Npp16s* d_dst = nppiMalloc_16s_C1(width * 3, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> srcData(width * height * 3);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(1), static_cast<Npp16s>(32767), 12345);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * 3 * sizeof(Npp16s),
               width * 3 * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqrt_16s_C3RSfs_Ctx(d_src, srcStep, d_dst, dstStep,
                                      roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqrt_16s_C3RSfs(d_src, srcStep, d_dst, dstStep,
                                  roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  nppiFree(d_src);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(Sqrt16sC3RSfs, Sqrt16sC3RSfsParamTest,
                         ::testing::Values(SqrtParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrtParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrtParam> &info) { return info.param.name; });

// ==================== Sqrt 16s C3IRSfs TEST_P ====================

class Sqrt16sC3IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrtParam> {};

TEST_P(Sqrt16sC3IRSfsParamTest, Sqrt_16s_C3IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int dstStep;
  Npp16s* d_dst = nppiMalloc_16s_C1(width * 3, height, &dstStep);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> dstData(width * height * 3);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp16s>(1), static_cast<Npp16s>(32767), 12345);
  cudaMemcpy2D(d_dst, dstStep, dstData.data(), width * 3 * sizeof(Npp16s),
               width * 3 * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqrt_16s_C3IRSfs_Ctx(d_dst, dstStep, roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqrt_16s_C3IRSfs(d_dst, dstStep, roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(Sqrt16sC3IRSfs, Sqrt16sC3IRSfsParamTest,
                         ::testing::Values(SqrtParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrtParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrtParam> &info) { return info.param.name; });

// ==================== Sqrt 16s AC4RSfs TEST_P ====================

class Sqrt16sAC4RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrtParam> {};

TEST_P(Sqrt16sAC4RSfsParamTest, Sqrt_16s_AC4RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int srcStep, dstStep;
  Npp16s* d_src = nppiMalloc_16s_C1(width * 4, height, &srcStep);
  Npp16s* d_dst = nppiMalloc_16s_C1(width * 4, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> srcData(width * height * 4);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(1), static_cast<Npp16s>(32767), 12345);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * 4 * sizeof(Npp16s),
               width * 4 * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqrt_16s_AC4RSfs_Ctx(d_src, srcStep, d_dst, dstStep,
                                       roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqrt_16s_AC4RSfs(d_src, srcStep, d_dst, dstStep,
                                   roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  nppiFree(d_src);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(Sqrt16sAC4RSfs, Sqrt16sAC4RSfsParamTest,
                         ::testing::Values(SqrtParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrtParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrtParam> &info) { return info.param.name; });

// ==================== Sqrt 16s AC4IRSfs TEST_P ====================

class Sqrt16sAC4IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrtParam> {};

TEST_P(Sqrt16sAC4IRSfsParamTest, Sqrt_16s_AC4IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int dstStep;
  Npp16s* d_dst = nppiMalloc_16s_C1(width * 4, height, &dstStep);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> dstData(width * height * 4);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp16s>(1), static_cast<Npp16s>(32767), 12345);
  cudaMemcpy2D(d_dst, dstStep, dstData.data(), width * 4 * sizeof(Npp16s),
               width * 4 * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqrt_16s_AC4IRSfs_Ctx(d_dst, dstStep, roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqrt_16s_AC4IRSfs(d_dst, dstStep, roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(Sqrt16sAC4IRSfs, Sqrt16sAC4IRSfsParamTest,
                         ::testing::Values(SqrtParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrtParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrtParam> &info) { return info.param.name; });

// ==================== Sqrt 32f AC4R TEST_P ====================

class Sqrt32fAC4RParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrtParam> {};

TEST_P(Sqrt32fAC4RParamTest, Sqrt_32f_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> srcData(width * height * 4);
  TestDataGenerator::generateRandom(srcData, 1.0f, 100.0f, 12345);

  NppImageMemory<Npp32f> src(width, height, 4);
  NppImageMemory<Npp32f> dst(width, height, 4);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqrt_32f_AC4R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiSqrt_32f_AC4R(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqrt32fAC4R, Sqrt32fAC4RParamTest,
                         ::testing::Values(SqrtParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrtParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrtParam> &info) { return info.param.name; });

// ==================== Sqrt 32f AC4IR TEST_P ====================

class Sqrt32fAC4IRParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrtParam> {};

TEST_P(Sqrt32fAC4IRParamTest, Sqrt_32f_AC4IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> dstData(width * height * 4);
  TestDataGenerator::generateRandom(dstData, 1.0f, 100.0f, 12345);

  NppImageMemory<Npp32f> dst(width, height, 4);
  dst.copyFromHost(dstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqrt_32f_AC4IR_Ctx(dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiSqrt_32f_AC4IR(dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqrt32fAC4IR, Sqrt32fAC4IRParamTest,
                         ::testing::Values(SqrtParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrtParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrtParam> &info) { return info.param.name; });
