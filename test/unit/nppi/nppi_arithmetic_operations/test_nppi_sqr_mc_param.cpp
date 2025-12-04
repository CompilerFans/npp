// Sqr multi-channel parameterized tests for nppi arithmetic operations
#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Common Param Structures ====================

struct SqrMCParam {
  int width;
  int height;
  int scaleFactor;
  bool use_ctx;
  std::string name;
};

// ==================== Sqr 8u C1RSfs TEST_P ====================

class Sqr8uC1RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrMCParam> {};

TEST_P(Sqr8uC1RSfsParamTest, Sqr_8u_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(1), static_cast<Npp8u>(15), 12345);

  NppImageMemory<Npp8u> src(width, height, 1);
  NppImageMemory<Npp8u> dst(width, height, 1);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqr_8u_C1RSfs_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqr_8u_C1RSfs(src.get(), src.step(), dst.get(), dst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqr8uC1RSfs, Sqr8uC1RSfsParamTest,
                         ::testing::Values(SqrMCParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrMCParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrMCParam> &info) { return info.param.name; });

// ==================== Sqr 8u C1IRSfs TEST_P ====================

class Sqr8uC1IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrMCParam> {};

TEST_P(Sqr8uC1IRSfsParamTest, Sqr_8u_C1IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcDstData(width * height);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp8u>(1), static_cast<Npp8u>(15), 12345);

  NppImageMemory<Npp8u> srcDst(width, height, 1);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqr_8u_C1IRSfs_Ctx(srcDst.get(), srcDst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqr_8u_C1IRSfs(srcDst.get(), srcDst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqr8uC1IRSfs, Sqr8uC1IRSfsParamTest,
                         ::testing::Values(SqrMCParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrMCParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrMCParam> &info) { return info.param.name; });

// ==================== Sqr 8u C3RSfs TEST_P ====================

class Sqr8uC3RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrMCParam> {};

TEST_P(Sqr8uC3RSfsParamTest, Sqr_8u_C3RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height * 3);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(1), static_cast<Npp8u>(15), 12345);

  NppImageMemory<Npp8u> src(width, height, 3);
  NppImageMemory<Npp8u> dst(width, height, 3);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqr_8u_C3RSfs_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqr_8u_C3RSfs(src.get(), src.step(), dst.get(), dst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqr8uC3RSfs, Sqr8uC3RSfsParamTest,
                         ::testing::Values(SqrMCParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrMCParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrMCParam> &info) { return info.param.name; });

// ==================== Sqr 8u C3IRSfs TEST_P ====================

class Sqr8uC3IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrMCParam> {};

TEST_P(Sqr8uC3IRSfsParamTest, Sqr_8u_C3IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcDstData(width * height * 3);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp8u>(1), static_cast<Npp8u>(15), 12345);

  NppImageMemory<Npp8u> srcDst(width, height, 3);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqr_8u_C3IRSfs_Ctx(srcDst.get(), srcDst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqr_8u_C3IRSfs(srcDst.get(), srcDst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqr8uC3IRSfs, Sqr8uC3IRSfsParamTest,
                         ::testing::Values(SqrMCParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrMCParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrMCParam> &info) { return info.param.name; });

// ==================== Sqr 8u C4RSfs TEST_P ====================

class Sqr8uC4RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrMCParam> {};

TEST_P(Sqr8uC4RSfsParamTest, Sqr_8u_C4RSfs) {
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
    status = nppiSqr_8u_C4RSfs_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqr_8u_C4RSfs(src.get(), src.step(), dst.get(), dst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqr8uC4RSfs, Sqr8uC4RSfsParamTest,
                         ::testing::Values(SqrMCParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrMCParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrMCParam> &info) { return info.param.name; });

// ==================== Sqr 8u C4IRSfs TEST_P ====================

class Sqr8uC4IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrMCParam> {};

TEST_P(Sqr8uC4IRSfsParamTest, Sqr_8u_C4IRSfs) {
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
    status = nppiSqr_8u_C4IRSfs_Ctx(srcDst.get(), srcDst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqr_8u_C4IRSfs(srcDst.get(), srcDst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqr8uC4IRSfs, Sqr8uC4IRSfsParamTest,
                         ::testing::Values(SqrMCParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrMCParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrMCParam> &info) { return info.param.name; });

// ==================== Sqr 16u C1RSfs TEST_P ====================

class Sqr16uC1RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrMCParam> {};

TEST_P(Sqr16uC1RSfsParamTest, Sqr_16u_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(1), static_cast<Npp16u>(255), 12345);

  NppImageMemory<Npp16u> src(width, height, 1);
  NppImageMemory<Npp16u> dst(width, height, 1);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqr_16u_C1RSfs_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqr_16u_C1RSfs(src.get(), src.step(), dst.get(), dst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqr16uC1RSfs, Sqr16uC1RSfsParamTest,
                         ::testing::Values(SqrMCParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrMCParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrMCParam> &info) { return info.param.name; });

// ==================== Sqr 16u C1IRSfs TEST_P ====================

class Sqr16uC1IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrMCParam> {};

TEST_P(Sqr16uC1IRSfsParamTest, Sqr_16u_C1IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcDstData(width * height);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp16u>(1), static_cast<Npp16u>(255), 12345);

  NppImageMemory<Npp16u> srcDst(width, height, 1);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqr_16u_C1IRSfs_Ctx(srcDst.get(), srcDst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqr_16u_C1IRSfs(srcDst.get(), srcDst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqr16uC1IRSfs, Sqr16uC1IRSfsParamTest,
                         ::testing::Values(SqrMCParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrMCParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrMCParam> &info) { return info.param.name; });

// ==================== Sqr 16u C3RSfs TEST_P ====================

class Sqr16uC3RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrMCParam> {};

TEST_P(Sqr16uC3RSfsParamTest, Sqr_16u_C3RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcData(width * height * 3);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(1), static_cast<Npp16u>(255), 12345);

  NppImageMemory<Npp16u> src(width, height, 3);
  NppImageMemory<Npp16u> dst(width, height, 3);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqr_16u_C3RSfs_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqr_16u_C3RSfs(src.get(), src.step(), dst.get(), dst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqr16uC3RSfs, Sqr16uC3RSfsParamTest,
                         ::testing::Values(SqrMCParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrMCParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrMCParam> &info) { return info.param.name; });

// ==================== Sqr 16u C3IRSfs TEST_P ====================

class Sqr16uC3IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrMCParam> {};

TEST_P(Sqr16uC3IRSfsParamTest, Sqr_16u_C3IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcDstData(width * height * 3);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp16u>(1), static_cast<Npp16u>(255), 12345);

  NppImageMemory<Npp16u> srcDst(width, height, 3);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqr_16u_C3IRSfs_Ctx(srcDst.get(), srcDst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqr_16u_C3IRSfs(srcDst.get(), srcDst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqr16uC3IRSfs, Sqr16uC3IRSfsParamTest,
                         ::testing::Values(SqrMCParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrMCParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrMCParam> &info) { return info.param.name; });

// ==================== Sqr 16u C4RSfs TEST_P ====================

class Sqr16uC4RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrMCParam> {};

TEST_P(Sqr16uC4RSfsParamTest, Sqr_16u_C4RSfs) {
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
    status = nppiSqr_16u_C4RSfs_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqr_16u_C4RSfs(src.get(), src.step(), dst.get(), dst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqr16uC4RSfs, Sqr16uC4RSfsParamTest,
                         ::testing::Values(SqrMCParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrMCParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrMCParam> &info) { return info.param.name; });

// ==================== Sqr 16u C4IRSfs TEST_P ====================

class Sqr16uC4IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrMCParam> {};

TEST_P(Sqr16uC4IRSfsParamTest, Sqr_16u_C4IRSfs) {
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
    status = nppiSqr_16u_C4IRSfs_Ctx(srcDst.get(), srcDst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqr_16u_C4IRSfs(srcDst.get(), srcDst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqr16uC4IRSfs, Sqr16uC4IRSfsParamTest,
                         ::testing::Values(SqrMCParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrMCParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrMCParam> &info) { return info.param.name; });

// ==================== Sqr 16s C1RSfs TEST_P ====================

class Sqr16sC1RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrMCParam> {};

TEST_P(Sqr16sC1RSfsParamTest, Sqr_16s_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int srcStep, dstStep;
  Npp16s* d_src = nppiMalloc_16s_C1(width, height, &srcStep);
  Npp16s* d_dst = nppiMalloc_16s_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(1), static_cast<Npp16s>(180), 12345);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * sizeof(Npp16s),
               width * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqr_16s_C1RSfs_Ctx(d_src, srcStep, d_dst, dstStep, roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqr_16s_C1RSfs(d_src, srcStep, d_dst, dstStep, roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  nppiFree(d_src);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(Sqr16sC1RSfs, Sqr16sC1RSfsParamTest,
                         ::testing::Values(SqrMCParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrMCParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrMCParam> &info) { return info.param.name; });

// ==================== Sqr 16s C1IRSfs TEST_P ====================

class Sqr16sC1IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrMCParam> {};

TEST_P(Sqr16sC1IRSfsParamTest, Sqr_16s_C1IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int dstStep;
  Npp16s* d_dst = nppiMalloc_16s_C1(width, height, &dstStep);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> dstData(width * height);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp16s>(1), static_cast<Npp16s>(180), 12345);
  cudaMemcpy2D(d_dst, dstStep, dstData.data(), width * sizeof(Npp16s),
               width * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqr_16s_C1IRSfs_Ctx(d_dst, dstStep, roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqr_16s_C1IRSfs(d_dst, dstStep, roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(Sqr16sC1IRSfs, Sqr16sC1IRSfsParamTest,
                         ::testing::Values(SqrMCParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrMCParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrMCParam> &info) { return info.param.name; });

// ==================== Sqr 16s C3RSfs TEST_P ====================

class Sqr16sC3RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrMCParam> {};

TEST_P(Sqr16sC3RSfsParamTest, Sqr_16s_C3RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int srcStep, dstStep;
  Npp16s* d_src = nppiMalloc_16s_C1(width * 3, height, &srcStep);
  Npp16s* d_dst = nppiMalloc_16s_C1(width * 3, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> srcData(width * height * 3);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(1), static_cast<Npp16s>(180), 12345);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * 3 * sizeof(Npp16s),
               width * 3 * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqr_16s_C3RSfs_Ctx(d_src, srcStep, d_dst, dstStep, roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqr_16s_C3RSfs(d_src, srcStep, d_dst, dstStep, roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  nppiFree(d_src);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(Sqr16sC3RSfs, Sqr16sC3RSfsParamTest,
                         ::testing::Values(SqrMCParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrMCParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrMCParam> &info) { return info.param.name; });

// ==================== Sqr 16s C3IRSfs TEST_P ====================

class Sqr16sC3IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrMCParam> {};

TEST_P(Sqr16sC3IRSfsParamTest, Sqr_16s_C3IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int dstStep;
  Npp16s* d_dst = nppiMalloc_16s_C1(width * 3, height, &dstStep);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> dstData(width * height * 3);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp16s>(1), static_cast<Npp16s>(180), 12345);
  cudaMemcpy2D(d_dst, dstStep, dstData.data(), width * 3 * sizeof(Npp16s),
               width * 3 * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqr_16s_C3IRSfs_Ctx(d_dst, dstStep, roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqr_16s_C3IRSfs(d_dst, dstStep, roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(Sqr16sC3IRSfs, Sqr16sC3IRSfsParamTest,
                         ::testing::Values(SqrMCParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrMCParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrMCParam> &info) { return info.param.name; });

// ==================== Sqr 16s C4RSfs TEST_P ====================

class Sqr16sC4RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrMCParam> {};

TEST_P(Sqr16sC4RSfsParamTest, Sqr_16s_C4RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int srcStep, dstStep;
  Npp16s* d_src = nppiMalloc_16s_C4(width, height, &srcStep);
  Npp16s* d_dst = nppiMalloc_16s_C4(width, height, &dstStep);
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
    status = nppiSqr_16s_C4RSfs_Ctx(d_src, srcStep, d_dst, dstStep, roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqr_16s_C4RSfs(d_src, srcStep, d_dst, dstStep, roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  nppiFree(d_src);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(Sqr16sC4RSfs, Sqr16sC4RSfsParamTest,
                         ::testing::Values(SqrMCParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrMCParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrMCParam> &info) { return info.param.name; });

// ==================== Sqr 16s C4IRSfs TEST_P ====================

class Sqr16sC4IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrMCParam> {};

TEST_P(Sqr16sC4IRSfsParamTest, Sqr_16s_C4IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int dstStep;
  Npp16s* d_dst = nppiMalloc_16s_C4(width, height, &dstStep);
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
    status = nppiSqr_16s_C4IRSfs_Ctx(d_dst, dstStep, roi, param.scaleFactor, ctx);
  } else {
    status = nppiSqr_16s_C4IRSfs(d_dst, dstStep, roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(Sqr16sC4IRSfs, Sqr16sC4IRSfsParamTest,
                         ::testing::Values(SqrMCParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrMCParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrMCParam> &info) { return info.param.name; });

// ==================== Sqr 32f C1R TEST_P ====================

class Sqr32fC1RParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrMCParam> {};

TEST_P(Sqr32fC1RParamTest, Sqr_32f_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, 1.0f, 10.0f, 12345);

  NppImageMemory<Npp32f> src(width, height, 1);
  NppImageMemory<Npp32f> dst(width, height, 1);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqr_32f_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiSqr_32f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqr32fC1R, Sqr32fC1RParamTest,
                         ::testing::Values(SqrMCParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrMCParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrMCParam> &info) { return info.param.name; });

// ==================== Sqr 32f C1IR TEST_P ====================

class Sqr32fC1IRParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrMCParam> {};

TEST_P(Sqr32fC1IRParamTest, Sqr_32f_C1IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> dstData(width * height);
  TestDataGenerator::generateRandom(dstData, 1.0f, 10.0f, 12345);

  NppImageMemory<Npp32f> dst(width, height, 1);
  dst.copyFromHost(dstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqr_32f_C1IR_Ctx(dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiSqr_32f_C1IR(dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqr32fC1IR, Sqr32fC1IRParamTest,
                         ::testing::Values(SqrMCParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrMCParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrMCParam> &info) { return info.param.name; });

// ==================== Sqr 32f C3R TEST_P ====================

class Sqr32fC3RParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrMCParam> {};

TEST_P(Sqr32fC3RParamTest, Sqr_32f_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> srcData(width * height * 3);
  TestDataGenerator::generateRandom(srcData, 1.0f, 10.0f, 12345);

  NppImageMemory<Npp32f> src(width, height, 3);
  NppImageMemory<Npp32f> dst(width, height, 3);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqr_32f_C3R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiSqr_32f_C3R(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqr32fC3R, Sqr32fC3RParamTest,
                         ::testing::Values(SqrMCParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrMCParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrMCParam> &info) { return info.param.name; });

// ==================== Sqr 32f C3IR TEST_P ====================

class Sqr32fC3IRParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrMCParam> {};

TEST_P(Sqr32fC3IRParamTest, Sqr_32f_C3IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> dstData(width * height * 3);
  TestDataGenerator::generateRandom(dstData, 1.0f, 10.0f, 12345);

  NppImageMemory<Npp32f> dst(width, height, 3);
  dst.copyFromHost(dstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiSqr_32f_C3IR_Ctx(dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiSqr_32f_C3IR(dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqr32fC3IR, Sqr32fC3IRParamTest,
                         ::testing::Values(SqrMCParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrMCParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrMCParam> &info) { return info.param.name; });

// ==================== Sqr 32f C4R TEST_P ====================

class Sqr32fC4RParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrMCParam> {};

TEST_P(Sqr32fC4RParamTest, Sqr_32f_C4R) {
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
    status = nppiSqr_32f_C4R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiSqr_32f_C4R(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqr32fC4R, Sqr32fC4RParamTest,
                         ::testing::Values(SqrMCParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrMCParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrMCParam> &info) { return info.param.name; });

// ==================== Sqr 32f C4IR TEST_P ====================

class Sqr32fC4IRParamTest : public NppTestBase, public ::testing::WithParamInterface<SqrMCParam> {};

TEST_P(Sqr32fC4IRParamTest, Sqr_32f_C4IR) {
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
    status = nppiSqr_32f_C4IR_Ctx(dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiSqr_32f_C4IR(dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Sqr32fC4IR, Sqr32fC4IRParamTest,
                         ::testing::Values(SqrMCParam{32, 32, 0, false, "32x32_noCtx"},
                                           SqrMCParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<SqrMCParam> &info) { return info.param.name; });
