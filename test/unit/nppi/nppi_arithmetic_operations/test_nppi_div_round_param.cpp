#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Common Param Structures ====================

struct DivRoundParam {
  int width;
  int height;
  int scaleFactor;
  NppRoundMode roundMode;
  bool use_ctx;
  std::string name;
};

// ==================== Div_Round 8u C1RSfs TEST_P ====================

class DivRound8uC1RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DivRoundParam> {};

TEST_P(DivRound8uC1RSfsParamTest, Div_Round_8u_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> src1Data(width * height);
  std::vector<Npp8u> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(1), static_cast<Npp8u>(255), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8u>(1), static_cast<Npp8u>(255), 54321);

  NppImageMemory<Npp8u> src1(width, height);
  NppImageMemory<Npp8u> src2(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiDiv_Round_8u_C1RSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                         param.roundMode, param.scaleFactor, ctx);
  } else {
    status = nppiDiv_Round_8u_C1RSfs(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                     param.roundMode, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(DivRound8uC1RSfs, DivRound8uC1RSfsParamTest,
                         ::testing::Values(DivRoundParam{32, 32, 0, NPP_RND_NEAR, false, "32x32_near_noCtx"},
                                           DivRoundParam{32, 32, 0, NPP_RND_NEAR, true, "32x32_near_Ctx"},
                                           DivRoundParam{32, 32, 0, NPP_RND_ZERO, false, "32x32_zero_noCtx"},
                                           DivRoundParam{64, 64, 2, NPP_RND_NEAR, false, "64x64_sf2_noCtx"}),
                         [](const ::testing::TestParamInfo<DivRoundParam> &info) { return info.param.name; });

// ==================== Div_Round 8u C1IRSfs TEST_P ====================

class DivRound8uC1IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DivRoundParam> {};

TEST_P(DivRound8uC1IRSfsParamTest, Div_Round_8u_C1IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> src1Data(width * height);
  std::vector<Npp8u> srcDstData(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(1), static_cast<Npp8u>(255), 12345);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp8u>(1), static_cast<Npp8u>(255), 54321);

  NppImageMemory<Npp8u> src1(width, height);
  NppImageMemory<Npp8u> srcDst(width, height);

  src1.copyFromHost(src1Data);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiDiv_Round_8u_C1IRSfs_Ctx(src1.get(), src1.step(), srcDst.get(), srcDst.step(), roi, param.roundMode,
                                          param.scaleFactor, ctx);
  } else {
    status = nppiDiv_Round_8u_C1IRSfs(src1.get(), src1.step(), srcDst.get(), srcDst.step(), roi, param.roundMode,
                                      param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(DivRound8uC1IRSfs, DivRound8uC1IRSfsParamTest,
                         ::testing::Values(DivRoundParam{32, 32, 0, NPP_RND_NEAR, false, "32x32_near_noCtx"},
                                           DivRoundParam{32, 32, 0, NPP_RND_NEAR, true, "32x32_near_Ctx"},
                                           DivRoundParam{32, 32, 0, NPP_RND_ZERO, false, "32x32_zero_noCtx"}),
                         [](const ::testing::TestParamInfo<DivRoundParam> &info) { return info.param.name; });

// ==================== Div_Round 8u C3RSfs TEST_P ====================

class DivRound8uC3RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DivRoundParam> {};

TEST_P(DivRound8uC3RSfsParamTest, Div_Round_8u_C3RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> src1Data(width * height * 3);
  std::vector<Npp8u> src2Data(width * height * 3);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(1), static_cast<Npp8u>(255), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8u>(1), static_cast<Npp8u>(255), 54321);

  NppImageMemory<Npp8u> src1(width, height, 3);
  NppImageMemory<Npp8u> src2(width, height, 3);
  NppImageMemory<Npp8u> dst(width, height, 3);

  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiDiv_Round_8u_C3RSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                         param.roundMode, param.scaleFactor, ctx);
  } else {
    status = nppiDiv_Round_8u_C3RSfs(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                     param.roundMode, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(DivRound8uC3RSfs, DivRound8uC3RSfsParamTest,
                         ::testing::Values(DivRoundParam{32, 32, 0, NPP_RND_NEAR, false, "32x32_near_noCtx"},
                                           DivRoundParam{32, 32, 0, NPP_RND_NEAR, true, "32x32_near_Ctx"}),
                         [](const ::testing::TestParamInfo<DivRoundParam> &info) { return info.param.name; });

// ==================== Div_Round 8u C3IRSfs TEST_P ====================

class DivRound8uC3IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DivRoundParam> {};

TEST_P(DivRound8uC3IRSfsParamTest, Div_Round_8u_C3IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> src1Data(width * height * 3);
  std::vector<Npp8u> srcDstData(width * height * 3);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(1), static_cast<Npp8u>(255), 12345);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp8u>(1), static_cast<Npp8u>(255), 54321);

  NppImageMemory<Npp8u> src1(width, height, 3);
  NppImageMemory<Npp8u> srcDst(width, height, 3);

  src1.copyFromHost(src1Data);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiDiv_Round_8u_C3IRSfs_Ctx(src1.get(), src1.step(), srcDst.get(), srcDst.step(), roi, param.roundMode,
                                          param.scaleFactor, ctx);
  } else {
    status = nppiDiv_Round_8u_C3IRSfs(src1.get(), src1.step(), srcDst.get(), srcDst.step(), roi, param.roundMode,
                                      param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(DivRound8uC3IRSfs, DivRound8uC3IRSfsParamTest,
                         ::testing::Values(DivRoundParam{32, 32, 0, NPP_RND_NEAR, false, "32x32_near_noCtx"},
                                           DivRoundParam{32, 32, 0, NPP_RND_NEAR, true, "32x32_near_Ctx"}),
                         [](const ::testing::TestParamInfo<DivRoundParam> &info) { return info.param.name; });

// ==================== Div_Round 8u C4RSfs TEST_P ====================

class DivRound8uC4RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DivRoundParam> {};

TEST_P(DivRound8uC4RSfsParamTest, Div_Round_8u_C4RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> src1Data(width * height * 4);
  std::vector<Npp8u> src2Data(width * height * 4);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(1), static_cast<Npp8u>(255), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8u>(1), static_cast<Npp8u>(255), 54321);

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
    status = nppiDiv_Round_8u_C4RSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                         param.roundMode, param.scaleFactor, ctx);
  } else {
    status = nppiDiv_Round_8u_C4RSfs(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                     param.roundMode, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(DivRound8uC4RSfs, DivRound8uC4RSfsParamTest,
                         ::testing::Values(DivRoundParam{32, 32, 0, NPP_RND_NEAR, false, "32x32_near_noCtx"},
                                           DivRoundParam{32, 32, 0, NPP_RND_NEAR, true, "32x32_near_Ctx"}),
                         [](const ::testing::TestParamInfo<DivRoundParam> &info) { return info.param.name; });

// ==================== Div_Round 8u C4IRSfs TEST_P ====================

class DivRound8uC4IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DivRoundParam> {};

TEST_P(DivRound8uC4IRSfsParamTest, Div_Round_8u_C4IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> src1Data(width * height * 4);
  std::vector<Npp8u> srcDstData(width * height * 4);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(1), static_cast<Npp8u>(255), 12345);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp8u>(1), static_cast<Npp8u>(255), 54321);

  NppImageMemory<Npp8u> src1(width, height, 4);
  NppImageMemory<Npp8u> srcDst(width, height, 4);

  src1.copyFromHost(src1Data);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiDiv_Round_8u_C4IRSfs_Ctx(src1.get(), src1.step(), srcDst.get(), srcDst.step(), roi, param.roundMode,
                                          param.scaleFactor, ctx);
  } else {
    status = nppiDiv_Round_8u_C4IRSfs(src1.get(), src1.step(), srcDst.get(), srcDst.step(), roi, param.roundMode,
                                      param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(DivRound8uC4IRSfs, DivRound8uC4IRSfsParamTest,
                         ::testing::Values(DivRoundParam{32, 32, 0, NPP_RND_NEAR, false, "32x32_near_noCtx"},
                                           DivRoundParam{32, 32, 0, NPP_RND_NEAR, true, "32x32_near_Ctx"}),
                         [](const ::testing::TestParamInfo<DivRoundParam> &info) { return info.param.name; });

// ==================== Div_Round 8u AC4RSfs TEST_P ====================

class DivRound8uAC4RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DivRoundParam> {};

TEST_P(DivRound8uAC4RSfsParamTest, Div_Round_8u_AC4RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> src1Data(width * height * 4);
  std::vector<Npp8u> src2Data(width * height * 4);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(1), static_cast<Npp8u>(255), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8u>(1), static_cast<Npp8u>(255), 54321);

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
    status = nppiDiv_Round_8u_AC4RSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                          param.roundMode, param.scaleFactor, ctx);
  } else {
    status = nppiDiv_Round_8u_AC4RSfs(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                      param.roundMode, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(DivRound8uAC4RSfs, DivRound8uAC4RSfsParamTest,
                         ::testing::Values(DivRoundParam{32, 32, 0, NPP_RND_NEAR, false, "32x32_near_noCtx"},
                                           DivRoundParam{32, 32, 0, NPP_RND_NEAR, true, "32x32_near_Ctx"}),
                         [](const ::testing::TestParamInfo<DivRoundParam> &info) { return info.param.name; });

// ==================== Div_Round 8u AC4IRSfs TEST_P ====================

class DivRound8uAC4IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DivRoundParam> {};

TEST_P(DivRound8uAC4IRSfsParamTest, Div_Round_8u_AC4IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> src1Data(width * height * 4);
  std::vector<Npp8u> srcDstData(width * height * 4);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(1), static_cast<Npp8u>(255), 12345);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp8u>(1), static_cast<Npp8u>(255), 54321);

  NppImageMemory<Npp8u> src1(width, height, 4);
  NppImageMemory<Npp8u> srcDst(width, height, 4);

  src1.copyFromHost(src1Data);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiDiv_Round_8u_AC4IRSfs_Ctx(src1.get(), src1.step(), srcDst.get(), srcDst.step(), roi, param.roundMode,
                                           param.scaleFactor, ctx);
  } else {
    status = nppiDiv_Round_8u_AC4IRSfs(src1.get(), src1.step(), srcDst.get(), srcDst.step(), roi, param.roundMode,
                                       param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(DivRound8uAC4IRSfs, DivRound8uAC4IRSfsParamTest,
                         ::testing::Values(DivRoundParam{32, 32, 0, NPP_RND_NEAR, false, "32x32_near_noCtx"},
                                           DivRoundParam{32, 32, 0, NPP_RND_NEAR, true, "32x32_near_Ctx"}),
                         [](const ::testing::TestParamInfo<DivRoundParam> &info) { return info.param.name; });

// ==================== Div_Round 16u C1RSfs TEST_P ====================

class DivRound16uC1RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DivRoundParam> {};

TEST_P(DivRound16uC1RSfsParamTest, Div_Round_16u_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> src1Data(width * height);
  std::vector<Npp16u> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(1), static_cast<Npp16u>(65535), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(1), static_cast<Npp16u>(65535), 54321);

  NppImageMemory<Npp16u> src1(width, height);
  NppImageMemory<Npp16u> src2(width, height);
  NppImageMemory<Npp16u> dst(width, height);

  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiDiv_Round_16u_C1RSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                          param.roundMode, param.scaleFactor, ctx);
  } else {
    status = nppiDiv_Round_16u_C1RSfs(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                      param.roundMode, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(DivRound16uC1RSfs, DivRound16uC1RSfsParamTest,
                         ::testing::Values(DivRoundParam{32, 32, 0, NPP_RND_NEAR, false, "32x32_near_noCtx"},
                                           DivRoundParam{32, 32, 0, NPP_RND_NEAR, true, "32x32_near_Ctx"}),
                         [](const ::testing::TestParamInfo<DivRoundParam> &info) { return info.param.name; });

// ==================== Div_Round 16u C1IRSfs TEST_P ====================

class DivRound16uC1IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DivRoundParam> {};

TEST_P(DivRound16uC1IRSfsParamTest, Div_Round_16u_C1IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> src1Data(width * height);
  std::vector<Npp16u> srcDstData(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(1), static_cast<Npp16u>(65535), 12345);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp16u>(1), static_cast<Npp16u>(65535), 54321);

  NppImageMemory<Npp16u> src1(width, height);
  NppImageMemory<Npp16u> srcDst(width, height);

  src1.copyFromHost(src1Data);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiDiv_Round_16u_C1IRSfs_Ctx(src1.get(), src1.step(), srcDst.get(), srcDst.step(), roi, param.roundMode,
                                           param.scaleFactor, ctx);
  } else {
    status = nppiDiv_Round_16u_C1IRSfs(src1.get(), src1.step(), srcDst.get(), srcDst.step(), roi, param.roundMode,
                                       param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(DivRound16uC1IRSfs, DivRound16uC1IRSfsParamTest,
                         ::testing::Values(DivRoundParam{32, 32, 0, NPP_RND_NEAR, false, "32x32_near_noCtx"},
                                           DivRoundParam{32, 32, 0, NPP_RND_NEAR, true, "32x32_near_Ctx"}),
                         [](const ::testing::TestParamInfo<DivRoundParam> &info) { return info.param.name; });

// ==================== Div_Round 16u C3RSfs TEST_P ====================

class DivRound16uC3RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DivRoundParam> {};

TEST_P(DivRound16uC3RSfsParamTest, Div_Round_16u_C3RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> src1Data(width * height * 3);
  std::vector<Npp16u> src2Data(width * height * 3);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(1), static_cast<Npp16u>(65535), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(1), static_cast<Npp16u>(65535), 54321);

  NppImageMemory<Npp16u> src1(width, height, 3);
  NppImageMemory<Npp16u> src2(width, height, 3);
  NppImageMemory<Npp16u> dst(width, height, 3);

  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiDiv_Round_16u_C3RSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                          param.roundMode, param.scaleFactor, ctx);
  } else {
    status = nppiDiv_Round_16u_C3RSfs(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                      param.roundMode, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(DivRound16uC3RSfs, DivRound16uC3RSfsParamTest,
                         ::testing::Values(DivRoundParam{32, 32, 0, NPP_RND_NEAR, false, "32x32_near_noCtx"},
                                           DivRoundParam{32, 32, 0, NPP_RND_NEAR, true, "32x32_near_Ctx"}),
                         [](const ::testing::TestParamInfo<DivRoundParam> &info) { return info.param.name; });

// ==================== Div_Round 16u C3IRSfs TEST_P ====================

class DivRound16uC3IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DivRoundParam> {};

TEST_P(DivRound16uC3IRSfsParamTest, Div_Round_16u_C3IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> src1Data(width * height * 3);
  std::vector<Npp16u> srcDstData(width * height * 3);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(1), static_cast<Npp16u>(65535), 12345);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp16u>(1), static_cast<Npp16u>(65535), 54321);

  NppImageMemory<Npp16u> src1(width, height, 3);
  NppImageMemory<Npp16u> srcDst(width, height, 3);

  src1.copyFromHost(src1Data);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiDiv_Round_16u_C3IRSfs_Ctx(src1.get(), src1.step(), srcDst.get(), srcDst.step(), roi, param.roundMode,
                                           param.scaleFactor, ctx);
  } else {
    status = nppiDiv_Round_16u_C3IRSfs(src1.get(), src1.step(), srcDst.get(), srcDst.step(), roi, param.roundMode,
                                       param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(DivRound16uC3IRSfs, DivRound16uC3IRSfsParamTest,
                         ::testing::Values(DivRoundParam{32, 32, 0, NPP_RND_NEAR, false, "32x32_near_noCtx"},
                                           DivRoundParam{32, 32, 0, NPP_RND_NEAR, true, "32x32_near_Ctx"}),
                         [](const ::testing::TestParamInfo<DivRoundParam> &info) { return info.param.name; });

// ==================== Div_Round 16u C4RSfs TEST_P ====================

class DivRound16uC4RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DivRoundParam> {};

TEST_P(DivRound16uC4RSfsParamTest, Div_Round_16u_C4RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> src1Data(width * height * 4);
  std::vector<Npp16u> src2Data(width * height * 4);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(1), static_cast<Npp16u>(65535), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(1), static_cast<Npp16u>(65535), 54321);

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
    status = nppiDiv_Round_16u_C4RSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                          param.roundMode, param.scaleFactor, ctx);
  } else {
    status = nppiDiv_Round_16u_C4RSfs(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                      param.roundMode, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(DivRound16uC4RSfs, DivRound16uC4RSfsParamTest,
                         ::testing::Values(DivRoundParam{32, 32, 0, NPP_RND_NEAR, false, "32x32_near_noCtx"},
                                           DivRoundParam{32, 32, 0, NPP_RND_NEAR, true, "32x32_near_Ctx"}),
                         [](const ::testing::TestParamInfo<DivRoundParam> &info) { return info.param.name; });

// ==================== Div_Round 16u C4IRSfs TEST_P ====================

class DivRound16uC4IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DivRoundParam> {};

TEST_P(DivRound16uC4IRSfsParamTest, Div_Round_16u_C4IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> src1Data(width * height * 4);
  std::vector<Npp16u> srcDstData(width * height * 4);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(1), static_cast<Npp16u>(65535), 12345);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp16u>(1), static_cast<Npp16u>(65535), 54321);

  NppImageMemory<Npp16u> src1(width, height, 4);
  NppImageMemory<Npp16u> srcDst(width, height, 4);

  src1.copyFromHost(src1Data);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiDiv_Round_16u_C4IRSfs_Ctx(src1.get(), src1.step(), srcDst.get(), srcDst.step(), roi, param.roundMode,
                                           param.scaleFactor, ctx);
  } else {
    status = nppiDiv_Round_16u_C4IRSfs(src1.get(), src1.step(), srcDst.get(), srcDst.step(), roi, param.roundMode,
                                       param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(DivRound16uC4IRSfs, DivRound16uC4IRSfsParamTest,
                         ::testing::Values(DivRoundParam{32, 32, 0, NPP_RND_NEAR, false, "32x32_near_noCtx"},
                                           DivRoundParam{32, 32, 0, NPP_RND_NEAR, true, "32x32_near_Ctx"}),
                         [](const ::testing::TestParamInfo<DivRoundParam> &info) { return info.param.name; });

// ==================== Div_Round 16u AC4RSfs TEST_P ====================

class DivRound16uAC4RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DivRoundParam> {};

TEST_P(DivRound16uAC4RSfsParamTest, Div_Round_16u_AC4RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> src1Data(width * height * 4);
  std::vector<Npp16u> src2Data(width * height * 4);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(1), static_cast<Npp16u>(65535), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(1), static_cast<Npp16u>(65535), 54321);

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
    status = nppiDiv_Round_16u_AC4RSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                           param.roundMode, param.scaleFactor, ctx);
  } else {
    status = nppiDiv_Round_16u_AC4RSfs(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                       param.roundMode, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(DivRound16uAC4RSfs, DivRound16uAC4RSfsParamTest,
                         ::testing::Values(DivRoundParam{32, 32, 0, NPP_RND_NEAR, false, "32x32_near_noCtx"},
                                           DivRoundParam{32, 32, 0, NPP_RND_NEAR, true, "32x32_near_Ctx"}),
                         [](const ::testing::TestParamInfo<DivRoundParam> &info) { return info.param.name; });

// ==================== Div_Round 16u AC4IRSfs TEST_P ====================

class DivRound16uAC4IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DivRoundParam> {};

TEST_P(DivRound16uAC4IRSfsParamTest, Div_Round_16u_AC4IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> src1Data(width * height * 4);
  std::vector<Npp16u> srcDstData(width * height * 4);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(1), static_cast<Npp16u>(65535), 12345);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp16u>(1), static_cast<Npp16u>(65535), 54321);

  NppImageMemory<Npp16u> src1(width, height, 4);
  NppImageMemory<Npp16u> srcDst(width, height, 4);

  src1.copyFromHost(src1Data);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiDiv_Round_16u_AC4IRSfs_Ctx(src1.get(), src1.step(), srcDst.get(), srcDst.step(), roi, param.roundMode,
                                            param.scaleFactor, ctx);
  } else {
    status = nppiDiv_Round_16u_AC4IRSfs(src1.get(), src1.step(), srcDst.get(), srcDst.step(), roi, param.roundMode,
                                        param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(DivRound16uAC4IRSfs, DivRound16uAC4IRSfsParamTest,
                         ::testing::Values(DivRoundParam{32, 32, 0, NPP_RND_NEAR, false, "32x32_near_noCtx"},
                                           DivRoundParam{32, 32, 0, NPP_RND_NEAR, true, "32x32_near_Ctx"}),
                         [](const ::testing::TestParamInfo<DivRoundParam> &info) { return info.param.name; });

// ==================== Div_Round 16s C1RSfs TEST_P ====================

class DivRound16sC1RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DivRoundParam> {};

TEST_P(DivRound16sC1RSfsParamTest, Div_Round_16s_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16s> src1Data(width * height);
  std::vector<Npp16s> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16s>(1), static_cast<Npp16s>(32767), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16s>(1), static_cast<Npp16s>(32767), 54321);

  NppImageMemory<Npp16s> src1(width, height);
  NppImageMemory<Npp16s> src2(width, height);
  NppImageMemory<Npp16s> dst(width, height);

  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiDiv_Round_16s_C1RSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                          param.roundMode, param.scaleFactor, ctx);
  } else {
    status = nppiDiv_Round_16s_C1RSfs(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                      param.roundMode, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(DivRound16sC1RSfs, DivRound16sC1RSfsParamTest,
                         ::testing::Values(DivRoundParam{32, 32, 0, NPP_RND_NEAR, false, "32x32_near_noCtx"},
                                           DivRoundParam{32, 32, 0, NPP_RND_NEAR, true, "32x32_near_Ctx"}),
                         [](const ::testing::TestParamInfo<DivRoundParam> &info) { return info.param.name; });

// ==================== Div_Round 16s C1IRSfs TEST_P ====================

class DivRound16sC1IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DivRoundParam> {};

TEST_P(DivRound16sC1IRSfsParamTest, Div_Round_16s_C1IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16s> src1Data(width * height);
  std::vector<Npp16s> srcDstData(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16s>(1), static_cast<Npp16s>(32767), 12345);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp16s>(1), static_cast<Npp16s>(32767), 54321);

  NppImageMemory<Npp16s> src1(width, height);
  NppImageMemory<Npp16s> srcDst(width, height);

  src1.copyFromHost(src1Data);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiDiv_Round_16s_C1IRSfs_Ctx(src1.get(), src1.step(), srcDst.get(), srcDst.step(), roi, param.roundMode,
                                           param.scaleFactor, ctx);
  } else {
    status = nppiDiv_Round_16s_C1IRSfs(src1.get(), src1.step(), srcDst.get(), srcDst.step(), roi, param.roundMode,
                                       param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(DivRound16sC1IRSfs, DivRound16sC1IRSfsParamTest,
                         ::testing::Values(DivRoundParam{32, 32, 0, NPP_RND_NEAR, false, "32x32_near_noCtx"},
                                           DivRoundParam{32, 32, 0, NPP_RND_NEAR, true, "32x32_near_Ctx"}),
                         [](const ::testing::TestParamInfo<DivRoundParam> &info) { return info.param.name; });

// ==================== Div_Round 16s C3RSfs TEST_P ====================

class DivRound16sC3RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DivRoundParam> {};

TEST_P(DivRound16sC3RSfsParamTest, Div_Round_16s_C3RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  // Note: NppImageMemory<Npp16s> only supports C1
  Npp16s *d_src1 = nullptr;
  Npp16s *d_src2 = nullptr;
  Npp16s *d_dst = nullptr;
  int src1Step, src2Step, dstStep;

  d_src1 = reinterpret_cast<Npp16s *>(nppiMalloc_16s_C1(width * 3, height, &src1Step));
  d_src2 = reinterpret_cast<Npp16s *>(nppiMalloc_16s_C1(width * 3, height, &src2Step));
  d_dst = reinterpret_cast<Npp16s *>(nppiMalloc_16s_C1(width * 3, height, &dstStep));

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_src2, nullptr);
  ASSERT_NE(d_dst, nullptr);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiDiv_Round_16s_C3RSfs_Ctx(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, roi, param.roundMode,
                                          param.scaleFactor, ctx);
  } else {
    status = nppiDiv_Round_16s_C3RSfs(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, roi, param.roundMode,
                                      param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  nppiFree(d_src1);
  nppiFree(d_src2);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(DivRound16sC3RSfs, DivRound16sC3RSfsParamTest,
                         ::testing::Values(DivRoundParam{32, 32, 0, NPP_RND_NEAR, false, "32x32_near_noCtx"},
                                           DivRoundParam{32, 32, 0, NPP_RND_NEAR, true, "32x32_near_Ctx"}),
                         [](const ::testing::TestParamInfo<DivRoundParam> &info) { return info.param.name; });

// ==================== Div_Round 16s C3IRSfs TEST_P ====================

class DivRound16sC3IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DivRoundParam> {};

TEST_P(DivRound16sC3IRSfsParamTest, Div_Round_16s_C3IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  Npp16s *d_src1 = nullptr;
  Npp16s *d_srcDst = nullptr;
  int src1Step, srcDstStep;

  d_src1 = reinterpret_cast<Npp16s *>(nppiMalloc_16s_C1(width * 3, height, &src1Step));
  d_srcDst = reinterpret_cast<Npp16s *>(nppiMalloc_16s_C1(width * 3, height, &srcDstStep));

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_srcDst, nullptr);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiDiv_Round_16s_C3IRSfs_Ctx(d_src1, src1Step, d_srcDst, srcDstStep, roi, param.roundMode,
                                           param.scaleFactor, ctx);
  } else {
    status = nppiDiv_Round_16s_C3IRSfs(d_src1, src1Step, d_srcDst, srcDstStep, roi, param.roundMode, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  nppiFree(d_src1);
  nppiFree(d_srcDst);
}

INSTANTIATE_TEST_SUITE_P(DivRound16sC3IRSfs, DivRound16sC3IRSfsParamTest,
                         ::testing::Values(DivRoundParam{32, 32, 0, NPP_RND_NEAR, false, "32x32_near_noCtx"},
                                           DivRoundParam{32, 32, 0, NPP_RND_NEAR, true, "32x32_near_Ctx"}),
                         [](const ::testing::TestParamInfo<DivRoundParam> &info) { return info.param.name; });

// ==================== Div_Round 16s C4RSfs TEST_P ====================

class DivRound16sC4RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DivRoundParam> {};

TEST_P(DivRound16sC4RSfsParamTest, Div_Round_16s_C4RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  Npp16s *d_src1 = nullptr;
  Npp16s *d_src2 = nullptr;
  Npp16s *d_dst = nullptr;
  int src1Step, src2Step, dstStep;

  d_src1 = reinterpret_cast<Npp16s *>(nppiMalloc_16s_C1(width * 4, height, &src1Step));
  d_src2 = reinterpret_cast<Npp16s *>(nppiMalloc_16s_C1(width * 4, height, &src2Step));
  d_dst = reinterpret_cast<Npp16s *>(nppiMalloc_16s_C1(width * 4, height, &dstStep));

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_src2, nullptr);
  ASSERT_NE(d_dst, nullptr);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiDiv_Round_16s_C4RSfs_Ctx(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, roi, param.roundMode,
                                          param.scaleFactor, ctx);
  } else {
    status = nppiDiv_Round_16s_C4RSfs(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, roi, param.roundMode,
                                      param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  nppiFree(d_src1);
  nppiFree(d_src2);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(DivRound16sC4RSfs, DivRound16sC4RSfsParamTest,
                         ::testing::Values(DivRoundParam{32, 32, 0, NPP_RND_NEAR, false, "32x32_near_noCtx"},
                                           DivRoundParam{32, 32, 0, NPP_RND_NEAR, true, "32x32_near_Ctx"}),
                         [](const ::testing::TestParamInfo<DivRoundParam> &info) { return info.param.name; });

// ==================== Div_Round 16s C4IRSfs TEST_P ====================

class DivRound16sC4IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DivRoundParam> {};

TEST_P(DivRound16sC4IRSfsParamTest, Div_Round_16s_C4IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  Npp16s *d_src1 = nullptr;
  Npp16s *d_srcDst = nullptr;
  int src1Step, srcDstStep;

  d_src1 = reinterpret_cast<Npp16s *>(nppiMalloc_16s_C1(width * 4, height, &src1Step));
  d_srcDst = reinterpret_cast<Npp16s *>(nppiMalloc_16s_C1(width * 4, height, &srcDstStep));

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_srcDst, nullptr);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiDiv_Round_16s_C4IRSfs_Ctx(d_src1, src1Step, d_srcDst, srcDstStep, roi, param.roundMode,
                                           param.scaleFactor, ctx);
  } else {
    status = nppiDiv_Round_16s_C4IRSfs(d_src1, src1Step, d_srcDst, srcDstStep, roi, param.roundMode, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  nppiFree(d_src1);
  nppiFree(d_srcDst);
}

INSTANTIATE_TEST_SUITE_P(DivRound16sC4IRSfs, DivRound16sC4IRSfsParamTest,
                         ::testing::Values(DivRoundParam{32, 32, 0, NPP_RND_NEAR, false, "32x32_near_noCtx"},
                                           DivRoundParam{32, 32, 0, NPP_RND_NEAR, true, "32x32_near_Ctx"}),
                         [](const ::testing::TestParamInfo<DivRoundParam> &info) { return info.param.name; });

// ==================== Div_Round 16s AC4RSfs TEST_P ====================

class DivRound16sAC4RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DivRoundParam> {};

TEST_P(DivRound16sAC4RSfsParamTest, Div_Round_16s_AC4RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  Npp16s *d_src1 = nullptr;
  Npp16s *d_src2 = nullptr;
  Npp16s *d_dst = nullptr;
  int src1Step, src2Step, dstStep;

  d_src1 = reinterpret_cast<Npp16s *>(nppiMalloc_16s_C1(width * 4, height, &src1Step));
  d_src2 = reinterpret_cast<Npp16s *>(nppiMalloc_16s_C1(width * 4, height, &src2Step));
  d_dst = reinterpret_cast<Npp16s *>(nppiMalloc_16s_C1(width * 4, height, &dstStep));

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_src2, nullptr);
  ASSERT_NE(d_dst, nullptr);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiDiv_Round_16s_AC4RSfs_Ctx(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, roi, param.roundMode,
                                           param.scaleFactor, ctx);
  } else {
    status = nppiDiv_Round_16s_AC4RSfs(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, roi, param.roundMode,
                                       param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  nppiFree(d_src1);
  nppiFree(d_src2);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(DivRound16sAC4RSfs, DivRound16sAC4RSfsParamTest,
                         ::testing::Values(DivRoundParam{32, 32, 0, NPP_RND_NEAR, false, "32x32_near_noCtx"},
                                           DivRoundParam{32, 32, 0, NPP_RND_NEAR, true, "32x32_near_Ctx"}),
                         [](const ::testing::TestParamInfo<DivRoundParam> &info) { return info.param.name; });

// ==================== Div_Round 16s AC4IRSfs TEST_P ====================

class DivRound16sAC4IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DivRoundParam> {};

TEST_P(DivRound16sAC4IRSfsParamTest, Div_Round_16s_AC4IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  Npp16s *d_src1 = nullptr;
  Npp16s *d_srcDst = nullptr;
  int src1Step, srcDstStep;

  d_src1 = reinterpret_cast<Npp16s *>(nppiMalloc_16s_C1(width * 4, height, &src1Step));
  d_srcDst = reinterpret_cast<Npp16s *>(nppiMalloc_16s_C1(width * 4, height, &srcDstStep));

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_srcDst, nullptr);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiDiv_Round_16s_AC4IRSfs_Ctx(d_src1, src1Step, d_srcDst, srcDstStep, roi, param.roundMode,
                                            param.scaleFactor, ctx);
  } else {
    status =
        nppiDiv_Round_16s_AC4IRSfs(d_src1, src1Step, d_srcDst, srcDstStep, roi, param.roundMode, param.scaleFactor);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  nppiFree(d_src1);
  nppiFree(d_srcDst);
}

INSTANTIATE_TEST_SUITE_P(DivRound16sAC4IRSfs, DivRound16sAC4IRSfsParamTest,
                         ::testing::Values(DivRoundParam{32, 32, 0, NPP_RND_NEAR, false, "32x32_near_noCtx"},
                                           DivRoundParam{32, 32, 0, NPP_RND_NEAR, true, "32x32_near_Ctx"}),
                         [](const ::testing::TestParamInfo<DivRoundParam> &info) { return info.param.name; });
