// Exp/Ln parameterized tests for nppi arithmetic operations
#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Common Param Structures ====================

struct ExpLnParam {
  int width;
  int height;
  int scaleFactor;
  bool use_ctx;
  std::string name;
};

// ==================== Exp 8u C1RSfs TEST_P ====================

class Exp8uC1RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<ExpLnParam> {};

TEST_P(Exp8uC1RSfsParamTest, Exp_8u_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(5), 12345);

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiExp_8u_C1RSfs_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiExp_8u_C1RSfs(src.get(), src.step(), dst.get(), dst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Exp8uC1RSfs, Exp8uC1RSfsParamTest,
                         ::testing::Values(ExpLnParam{32, 32, 0, false, "32x32_noCtx"},
                                           ExpLnParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<ExpLnParam> &info) { return info.param.name; });

// ==================== Exp 8u C1IRSfs TEST_P ====================

class Exp8uC1IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<ExpLnParam> {};

TEST_P(Exp8uC1IRSfsParamTest, Exp_8u_C1IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcDstData(width * height);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp8u>(0), static_cast<Npp8u>(5), 12345);

  NppImageMemory<Npp8u> srcDst(width, height);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiExp_8u_C1IRSfs_Ctx(srcDst.get(), srcDst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiExp_8u_C1IRSfs(srcDst.get(), srcDst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Exp8uC1IRSfs, Exp8uC1IRSfsParamTest,
                         ::testing::Values(ExpLnParam{32, 32, 0, false, "32x32_noCtx"},
                                           ExpLnParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<ExpLnParam> &info) { return info.param.name; });

// ==================== Exp 8u C3RSfs TEST_P ====================

class Exp8uC3RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<ExpLnParam> {};

TEST_P(Exp8uC3RSfsParamTest, Exp_8u_C3RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height * 3);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(5), 12345);

  NppImageMemory<Npp8u> src(width, height, 3);
  NppImageMemory<Npp8u> dst(width, height, 3);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiExp_8u_C3RSfs_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiExp_8u_C3RSfs(src.get(), src.step(), dst.get(), dst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Exp8uC3RSfs, Exp8uC3RSfsParamTest,
                         ::testing::Values(ExpLnParam{32, 32, 0, false, "32x32_noCtx"},
                                           ExpLnParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<ExpLnParam> &info) { return info.param.name; });

// ==================== Exp 8u C3IRSfs TEST_P ====================

class Exp8uC3IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<ExpLnParam> {};

TEST_P(Exp8uC3IRSfsParamTest, Exp_8u_C3IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcDstData(width * height * 3);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp8u>(0), static_cast<Npp8u>(5), 12345);

  NppImageMemory<Npp8u> srcDst(width, height, 3);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiExp_8u_C3IRSfs_Ctx(srcDst.get(), srcDst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiExp_8u_C3IRSfs(srcDst.get(), srcDst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Exp8uC3IRSfs, Exp8uC3IRSfsParamTest,
                         ::testing::Values(ExpLnParam{32, 32, 0, false, "32x32_noCtx"},
                                           ExpLnParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<ExpLnParam> &info) { return info.param.name; });

// ==================== Exp 16u C1RSfs TEST_P ====================

class Exp16uC1RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<ExpLnParam> {};

TEST_P(Exp16uC1RSfsParamTest, Exp_16u_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(10), 12345);

  NppImageMemory<Npp16u> src(width, height);
  NppImageMemory<Npp16u> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiExp_16u_C1RSfs_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiExp_16u_C1RSfs(src.get(), src.step(), dst.get(), dst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Exp16uC1RSfs, Exp16uC1RSfsParamTest,
                         ::testing::Values(ExpLnParam{32, 32, 0, false, "32x32_noCtx"},
                                           ExpLnParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<ExpLnParam> &info) { return info.param.name; });

// ==================== Exp 16u C1IRSfs TEST_P ====================

class Exp16uC1IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<ExpLnParam> {};

TEST_P(Exp16uC1IRSfsParamTest, Exp_16u_C1IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcDstData(width * height);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp16u>(0), static_cast<Npp16u>(10), 12345);

  NppImageMemory<Npp16u> srcDst(width, height);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiExp_16u_C1IRSfs_Ctx(srcDst.get(), srcDst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiExp_16u_C1IRSfs(srcDst.get(), srcDst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Exp16uC1IRSfs, Exp16uC1IRSfsParamTest,
                         ::testing::Values(ExpLnParam{32, 32, 0, false, "32x32_noCtx"},
                                           ExpLnParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<ExpLnParam> &info) { return info.param.name; });

// ==================== Exp 16u C3RSfs TEST_P ====================

class Exp16uC3RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<ExpLnParam> {};

TEST_P(Exp16uC3RSfsParamTest, Exp_16u_C3RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcData(width * height * 3);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(10), 12345);

  NppImageMemory<Npp16u> src(width, height, 3);
  NppImageMemory<Npp16u> dst(width, height, 3);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiExp_16u_C3RSfs_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiExp_16u_C3RSfs(src.get(), src.step(), dst.get(), dst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Exp16uC3RSfs, Exp16uC3RSfsParamTest,
                         ::testing::Values(ExpLnParam{32, 32, 0, false, "32x32_noCtx"},
                                           ExpLnParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<ExpLnParam> &info) { return info.param.name; });

// ==================== Exp 16u C3IRSfs TEST_P ====================

class Exp16uC3IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<ExpLnParam> {};

TEST_P(Exp16uC3IRSfsParamTest, Exp_16u_C3IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcDstData(width * height * 3);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp16u>(0), static_cast<Npp16u>(10), 12345);

  NppImageMemory<Npp16u> srcDst(width, height, 3);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiExp_16u_C3IRSfs_Ctx(srcDst.get(), srcDst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiExp_16u_C3IRSfs(srcDst.get(), srcDst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Exp16uC3IRSfs, Exp16uC3IRSfsParamTest,
                         ::testing::Values(ExpLnParam{32, 32, 0, false, "32x32_noCtx"},
                                           ExpLnParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<ExpLnParam> &info) { return info.param.name; });

// ==================== Exp 16s C1RSfs TEST_P ====================

class Exp16sC1RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<ExpLnParam> {};

TEST_P(Exp16sC1RSfsParamTest, Exp_16s_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int srcStep, dstStep;
  Npp16s *d_src = nppiMalloc_16s_C1(width, height, &srcStep);
  Npp16s *d_dst = nppiMalloc_16s_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(0), static_cast<Npp16s>(10), 12345);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * sizeof(Npp16s), width * sizeof(Npp16s), height,
               cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiExp_16s_C1RSfs_Ctx(d_src, srcStep, d_dst, dstStep, roi, param.scaleFactor, ctx);
  } else {
    status = nppiExp_16s_C1RSfs(d_src, srcStep, d_dst, dstStep, roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  nppiFree(d_src);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(Exp16sC1RSfs, Exp16sC1RSfsParamTest,
                         ::testing::Values(ExpLnParam{32, 32, 0, false, "32x32_noCtx"},
                                           ExpLnParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<ExpLnParam> &info) { return info.param.name; });

// ==================== Exp 16s C1IRSfs TEST_P ====================

class Exp16sC1IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<ExpLnParam> {};

TEST_P(Exp16sC1IRSfsParamTest, Exp_16s_C1IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int dstStep;
  Npp16s *d_dst = nppiMalloc_16s_C1(width, height, &dstStep);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> dstData(width * height);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp16s>(0), static_cast<Npp16s>(10), 12345);
  cudaMemcpy2D(d_dst, dstStep, dstData.data(), width * sizeof(Npp16s), width * sizeof(Npp16s), height,
               cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiExp_16s_C1IRSfs_Ctx(d_dst, dstStep, roi, param.scaleFactor, ctx);
  } else {
    status = nppiExp_16s_C1IRSfs(d_dst, dstStep, roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(Exp16sC1IRSfs, Exp16sC1IRSfsParamTest,
                         ::testing::Values(ExpLnParam{32, 32, 0, false, "32x32_noCtx"},
                                           ExpLnParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<ExpLnParam> &info) { return info.param.name; });

// ==================== Exp 16s C3RSfs TEST_P ====================

class Exp16sC3RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<ExpLnParam> {};

TEST_P(Exp16sC3RSfsParamTest, Exp_16s_C3RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int srcStep, dstStep;
  Npp16s *d_src = nppiMalloc_16s_C1(width * 3, height, &srcStep);
  Npp16s *d_dst = nppiMalloc_16s_C1(width * 3, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> srcData(width * height * 3);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(0), static_cast<Npp16s>(10), 12345);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * 3 * sizeof(Npp16s), width * 3 * sizeof(Npp16s), height,
               cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiExp_16s_C3RSfs_Ctx(d_src, srcStep, d_dst, dstStep, roi, param.scaleFactor, ctx);
  } else {
    status = nppiExp_16s_C3RSfs(d_src, srcStep, d_dst, dstStep, roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  nppiFree(d_src);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(Exp16sC3RSfs, Exp16sC3RSfsParamTest,
                         ::testing::Values(ExpLnParam{32, 32, 0, false, "32x32_noCtx"},
                                           ExpLnParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<ExpLnParam> &info) { return info.param.name; });

// ==================== Exp 16s C3IRSfs TEST_P ====================

class Exp16sC3IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<ExpLnParam> {};

TEST_P(Exp16sC3IRSfsParamTest, Exp_16s_C3IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int dstStep;
  Npp16s *d_dst = nppiMalloc_16s_C1(width * 3, height, &dstStep);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> dstData(width * height * 3);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp16s>(0), static_cast<Npp16s>(10), 12345);
  cudaMemcpy2D(d_dst, dstStep, dstData.data(), width * 3 * sizeof(Npp16s), width * 3 * sizeof(Npp16s), height,
               cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiExp_16s_C3IRSfs_Ctx(d_dst, dstStep, roi, param.scaleFactor, ctx);
  } else {
    status = nppiExp_16s_C3IRSfs(d_dst, dstStep, roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(Exp16sC3IRSfs, Exp16sC3IRSfsParamTest,
                         ::testing::Values(ExpLnParam{32, 32, 0, false, "32x32_noCtx"},
                                           ExpLnParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<ExpLnParam> &info) { return info.param.name; });

// ==================== Ln 8u C1RSfs TEST_P ====================

class Ln8uC1RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<ExpLnParam> {};

TEST_P(Ln8uC1RSfsParamTest, Ln_8u_C1RSfs) {
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
    status = nppiLn_8u_C1RSfs_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiLn_8u_C1RSfs(src.get(), src.step(), dst.get(), dst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Ln8uC1RSfs, Ln8uC1RSfsParamTest,
                         ::testing::Values(ExpLnParam{32, 32, 0, false, "32x32_noCtx"},
                                           ExpLnParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<ExpLnParam> &info) { return info.param.name; });

// ==================== Ln 8u C1IRSfs TEST_P ====================

class Ln8uC1IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<ExpLnParam> {};

TEST_P(Ln8uC1IRSfsParamTest, Ln_8u_C1IRSfs) {
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
    status = nppiLn_8u_C1IRSfs_Ctx(srcDst.get(), srcDst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiLn_8u_C1IRSfs(srcDst.get(), srcDst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Ln8uC1IRSfs, Ln8uC1IRSfsParamTest,
                         ::testing::Values(ExpLnParam{32, 32, 0, false, "32x32_noCtx"},
                                           ExpLnParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<ExpLnParam> &info) { return info.param.name; });

// ==================== Ln 8u C3RSfs TEST_P ====================

class Ln8uC3RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<ExpLnParam> {};

TEST_P(Ln8uC3RSfsParamTest, Ln_8u_C3RSfs) {
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
    status = nppiLn_8u_C3RSfs_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiLn_8u_C3RSfs(src.get(), src.step(), dst.get(), dst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Ln8uC3RSfs, Ln8uC3RSfsParamTest,
                         ::testing::Values(ExpLnParam{32, 32, 0, false, "32x32_noCtx"},
                                           ExpLnParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<ExpLnParam> &info) { return info.param.name; });

// ==================== Ln 8u C3IRSfs TEST_P ====================

class Ln8uC3IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<ExpLnParam> {};

TEST_P(Ln8uC3IRSfsParamTest, Ln_8u_C3IRSfs) {
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
    status = nppiLn_8u_C3IRSfs_Ctx(srcDst.get(), srcDst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiLn_8u_C3IRSfs(srcDst.get(), srcDst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Ln8uC3IRSfs, Ln8uC3IRSfsParamTest,
                         ::testing::Values(ExpLnParam{32, 32, 0, false, "32x32_noCtx"},
                                           ExpLnParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<ExpLnParam> &info) { return info.param.name; });

// ==================== Ln 16u C1RSfs TEST_P ====================

class Ln16uC1RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<ExpLnParam> {};

TEST_P(Ln16uC1RSfsParamTest, Ln_16u_C1RSfs) {
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
    status = nppiLn_16u_C1RSfs_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiLn_16u_C1RSfs(src.get(), src.step(), dst.get(), dst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Ln16uC1RSfs, Ln16uC1RSfsParamTest,
                         ::testing::Values(ExpLnParam{32, 32, 0, false, "32x32_noCtx"},
                                           ExpLnParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<ExpLnParam> &info) { return info.param.name; });

// ==================== Ln 16u C1IRSfs TEST_P ====================

class Ln16uC1IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<ExpLnParam> {};

TEST_P(Ln16uC1IRSfsParamTest, Ln_16u_C1IRSfs) {
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
    status = nppiLn_16u_C1IRSfs_Ctx(srcDst.get(), srcDst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiLn_16u_C1IRSfs(srcDst.get(), srcDst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Ln16uC1IRSfs, Ln16uC1IRSfsParamTest,
                         ::testing::Values(ExpLnParam{32, 32, 0, false, "32x32_noCtx"},
                                           ExpLnParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<ExpLnParam> &info) { return info.param.name; });

// ==================== Ln 16u C3RSfs TEST_P ====================

class Ln16uC3RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<ExpLnParam> {};

TEST_P(Ln16uC3RSfsParamTest, Ln_16u_C3RSfs) {
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
    status = nppiLn_16u_C3RSfs_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiLn_16u_C3RSfs(src.get(), src.step(), dst.get(), dst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Ln16uC3RSfs, Ln16uC3RSfsParamTest,
                         ::testing::Values(ExpLnParam{32, 32, 0, false, "32x32_noCtx"},
                                           ExpLnParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<ExpLnParam> &info) { return info.param.name; });

// ==================== Ln 16u C3IRSfs TEST_P ====================

class Ln16uC3IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<ExpLnParam> {};

TEST_P(Ln16uC3IRSfsParamTest, Ln_16u_C3IRSfs) {
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
    status = nppiLn_16u_C3IRSfs_Ctx(srcDst.get(), srcDst.step(), roi, param.scaleFactor, ctx);
  } else {
    status = nppiLn_16u_C3IRSfs(srcDst.get(), srcDst.step(), roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(Ln16uC3IRSfs, Ln16uC3IRSfsParamTest,
                         ::testing::Values(ExpLnParam{32, 32, 0, false, "32x32_noCtx"},
                                           ExpLnParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<ExpLnParam> &info) { return info.param.name; });

// ==================== Ln 16s C1RSfs TEST_P ====================

class Ln16sC1RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<ExpLnParam> {};

TEST_P(Ln16sC1RSfsParamTest, Ln_16s_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int srcStep, dstStep;
  Npp16s *d_src = nppiMalloc_16s_C1(width, height, &srcStep);
  Npp16s *d_dst = nppiMalloc_16s_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(1), static_cast<Npp16s>(32767), 12345);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * sizeof(Npp16s), width * sizeof(Npp16s), height,
               cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiLn_16s_C1RSfs_Ctx(d_src, srcStep, d_dst, dstStep, roi, param.scaleFactor, ctx);
  } else {
    status = nppiLn_16s_C1RSfs(d_src, srcStep, d_dst, dstStep, roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  nppiFree(d_src);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(Ln16sC1RSfs, Ln16sC1RSfsParamTest,
                         ::testing::Values(ExpLnParam{32, 32, 0, false, "32x32_noCtx"},
                                           ExpLnParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<ExpLnParam> &info) { return info.param.name; });

// ==================== Ln 16s C1IRSfs TEST_P ====================

class Ln16sC1IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<ExpLnParam> {};

TEST_P(Ln16sC1IRSfsParamTest, Ln_16s_C1IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int dstStep;
  Npp16s *d_dst = nppiMalloc_16s_C1(width, height, &dstStep);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> dstData(width * height);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp16s>(1), static_cast<Npp16s>(32767), 12345);
  cudaMemcpy2D(d_dst, dstStep, dstData.data(), width * sizeof(Npp16s), width * sizeof(Npp16s), height,
               cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiLn_16s_C1IRSfs_Ctx(d_dst, dstStep, roi, param.scaleFactor, ctx);
  } else {
    status = nppiLn_16s_C1IRSfs(d_dst, dstStep, roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(Ln16sC1IRSfs, Ln16sC1IRSfsParamTest,
                         ::testing::Values(ExpLnParam{32, 32, 0, false, "32x32_noCtx"},
                                           ExpLnParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<ExpLnParam> &info) { return info.param.name; });

// ==================== Ln 16s C3RSfs TEST_P ====================

class Ln16sC3RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<ExpLnParam> {};

TEST_P(Ln16sC3RSfsParamTest, Ln_16s_C3RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int srcStep, dstStep;
  Npp16s *d_src = nppiMalloc_16s_C1(width * 3, height, &srcStep);
  Npp16s *d_dst = nppiMalloc_16s_C1(width * 3, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> srcData(width * height * 3);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(1), static_cast<Npp16s>(32767), 12345);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * 3 * sizeof(Npp16s), width * 3 * sizeof(Npp16s), height,
               cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiLn_16s_C3RSfs_Ctx(d_src, srcStep, d_dst, dstStep, roi, param.scaleFactor, ctx);
  } else {
    status = nppiLn_16s_C3RSfs(d_src, srcStep, d_dst, dstStep, roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  nppiFree(d_src);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(Ln16sC3RSfs, Ln16sC3RSfsParamTest,
                         ::testing::Values(ExpLnParam{32, 32, 0, false, "32x32_noCtx"},
                                           ExpLnParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<ExpLnParam> &info) { return info.param.name; });

// ==================== Ln 16s C3IRSfs TEST_P ====================

class Ln16sC3IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<ExpLnParam> {};

TEST_P(Ln16sC3IRSfsParamTest, Ln_16s_C3IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int dstStep;
  Npp16s *d_dst = nppiMalloc_16s_C1(width * 3, height, &dstStep);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> dstData(width * height * 3);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp16s>(1), static_cast<Npp16s>(32767), 12345);
  cudaMemcpy2D(d_dst, dstStep, dstData.data(), width * 3 * sizeof(Npp16s), width * 3 * sizeof(Npp16s), height,
               cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiLn_16s_C3IRSfs_Ctx(d_dst, dstStep, roi, param.scaleFactor, ctx);
  } else {
    status = nppiLn_16s_C3IRSfs(d_dst, dstStep, roi, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(Ln16sC3IRSfs, Ln16sC3IRSfsParamTest,
                         ::testing::Values(ExpLnParam{32, 32, 0, false, "32x32_noCtx"},
                                           ExpLnParam{32, 32, 0, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<ExpLnParam> &info) { return info.param.name; });
