#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Common Param Structures ====================

struct AlphaPremulParam {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

// ==================== AlphaPremul 8u AC4R TEST_P ====================

class AlphaPremul8uAC4RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaPremulParam> {};

TEST_P(AlphaPremul8uAC4RParamTest, AlphaPremul_8u_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;

  std::vector<Npp8u> srcData(width * height * channels);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  NppImageMemory<Npp8u> src(width, height, 4);
  NppImageMemory<Npp8u> dst(width, height, 4);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaPremul_8u_AC4R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiAlphaPremul_8u_AC4R(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaPremul8uAC4R, AlphaPremul8uAC4RParamTest,
                         ::testing::Values(AlphaPremulParam{32, 32, false, "32x32_noCtx"},
                                           AlphaPremulParam{32, 32, true, "32x32_Ctx"},
                                           AlphaPremulParam{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaPremulParam> &info) { return info.param.name; });

// ==================== AlphaPremul 8u AC4IR TEST_P ====================

class AlphaPremul8uAC4IRParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaPremulParam> {};

TEST_P(AlphaPremul8uAC4IRParamTest, AlphaPremul_8u_AC4IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;

  std::vector<Npp8u> srcData(width * height * channels);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  NppImageMemory<Npp8u> srcDst(width, height, 4);

  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaPremul_8u_AC4IR_Ctx(srcDst.get(), srcDst.step(), roi, ctx);
  } else {
    status = nppiAlphaPremul_8u_AC4IR(srcDst.get(), srcDst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaPremul8uAC4IR, AlphaPremul8uAC4IRParamTest,
                         ::testing::Values(AlphaPremulParam{32, 32, false, "32x32_noCtx"},
                                           AlphaPremulParam{32, 32, true, "32x32_Ctx"},
                                           AlphaPremulParam{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaPremulParam> &info) { return info.param.name; });

// ==================== AlphaPremul 16u AC4R TEST_P ====================

class AlphaPremul16uAC4RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaPremulParam> {};

TEST_P(AlphaPremul16uAC4RParamTest, AlphaPremul_16u_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;

  std::vector<Npp16u> srcData(width * height * channels);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);

  NppImageMemory<Npp16u> src(width, height, 4);
  NppImageMemory<Npp16u> dst(width, height, 4);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaPremul_16u_AC4R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiAlphaPremul_16u_AC4R(src.get(), src.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaPremul16uAC4R, AlphaPremul16uAC4RParamTest,
                         ::testing::Values(AlphaPremulParam{32, 32, false, "32x32_noCtx"},
                                           AlphaPremulParam{32, 32, true, "32x32_Ctx"},
                                           AlphaPremulParam{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaPremulParam> &info) { return info.param.name; });

// ==================== AlphaPremul 16u AC4IR TEST_P ====================

class AlphaPremul16uAC4IRParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaPremulParam> {};

TEST_P(AlphaPremul16uAC4IRParamTest, AlphaPremul_16u_AC4IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;

  std::vector<Npp16u> srcData(width * height * channels);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);

  NppImageMemory<Npp16u> srcDst(width, height, 4);

  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaPremul_16u_AC4IR_Ctx(srcDst.get(), srcDst.step(), roi, ctx);
  } else {
    status = nppiAlphaPremul_16u_AC4IR(srcDst.get(), srcDst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaPremul16uAC4IR, AlphaPremul16uAC4IRParamTest,
                         ::testing::Values(AlphaPremulParam{32, 32, false, "32x32_noCtx"},
                                           AlphaPremulParam{32, 32, true, "32x32_Ctx"},
                                           AlphaPremulParam{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaPremulParam> &info) { return info.param.name; });

// ==================== AlphaPremulC Param Structures ====================

struct AlphaPremulCParam {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

// ==================== AlphaPremulC 8u C1R TEST_P ====================

class AlphaPremulC8uC1RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaPremulCParam> {};

TEST_P(AlphaPremulC8uC1RParamTest, AlphaPremulC_8u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  src.copyFromHost(srcData);

  Npp8u alpha = 128;

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaPremulC_8u_C1R_Ctx(src.get(), src.step(), alpha, dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiAlphaPremulC_8u_C1R(src.get(), src.step(), alpha, dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaPremulC8uC1R, AlphaPremulC8uC1RParamTest,
                         ::testing::Values(AlphaPremulCParam{32, 32, false, "32x32_noCtx"},
                                           AlphaPremulCParam{32, 32, true, "32x32_Ctx"},
                                           AlphaPremulCParam{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaPremulCParam> &info) { return info.param.name; });

// ==================== AlphaPremulC 8u C1IR TEST_P ====================

class AlphaPremulC8uC1IRParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaPremulCParam> {};

TEST_P(AlphaPremulC8uC1IRParamTest, AlphaPremulC_8u_C1IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  NppImageMemory<Npp8u> srcDst(width, height);

  srcDst.copyFromHost(srcData);

  Npp8u alpha = 128;

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaPremulC_8u_C1IR_Ctx(alpha, srcDst.get(), srcDst.step(), roi, ctx);
  } else {
    status = nppiAlphaPremulC_8u_C1IR(alpha, srcDst.get(), srcDst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaPremulC8uC1IR, AlphaPremulC8uC1IRParamTest,
                         ::testing::Values(AlphaPremulCParam{32, 32, false, "32x32_noCtx"},
                                           AlphaPremulCParam{32, 32, true, "32x32_Ctx"},
                                           AlphaPremulCParam{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaPremulCParam> &info) { return info.param.name; });

// ==================== AlphaPremulC 8u C3R TEST_P ====================

class AlphaPremulC8uC3RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaPremulCParam> {};

TEST_P(AlphaPremulC8uC3RParamTest, AlphaPremulC_8u_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 3;

  std::vector<Npp8u> srcData(width * height * channels);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  NppImageMemory<Npp8u> src(width, height, 3);
  NppImageMemory<Npp8u> dst(width, height, 3);

  src.copyFromHost(srcData);

  Npp8u alpha = 128;

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaPremulC_8u_C3R_Ctx(src.get(), src.step(), alpha, dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiAlphaPremulC_8u_C3R(src.get(), src.step(), alpha, dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaPremulC8uC3R, AlphaPremulC8uC3RParamTest,
                         ::testing::Values(AlphaPremulCParam{32, 32, false, "32x32_noCtx"},
                                           AlphaPremulCParam{32, 32, true, "32x32_Ctx"},
                                           AlphaPremulCParam{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaPremulCParam> &info) { return info.param.name; });

// ==================== AlphaPremulC 8u C3IR TEST_P ====================

class AlphaPremulC8uC3IRParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaPremulCParam> {};

TEST_P(AlphaPremulC8uC3IRParamTest, AlphaPremulC_8u_C3IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 3;

  std::vector<Npp8u> srcData(width * height * channels);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  NppImageMemory<Npp8u> srcDst(width, height, 3);

  srcDst.copyFromHost(srcData);

  Npp8u alpha = 128;

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaPremulC_8u_C3IR_Ctx(alpha, srcDst.get(), srcDst.step(), roi, ctx);
  } else {
    status = nppiAlphaPremulC_8u_C3IR(alpha, srcDst.get(), srcDst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaPremulC8uC3IR, AlphaPremulC8uC3IRParamTest,
                         ::testing::Values(AlphaPremulCParam{32, 32, false, "32x32_noCtx"},
                                           AlphaPremulCParam{32, 32, true, "32x32_Ctx"},
                                           AlphaPremulCParam{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaPremulCParam> &info) { return info.param.name; });

// ==================== AlphaPremulC 8u C4R TEST_P ====================

class AlphaPremulC8uC4RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaPremulCParam> {};

TEST_P(AlphaPremulC8uC4RParamTest, AlphaPremulC_8u_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;

  std::vector<Npp8u> srcData(width * height * channels);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  NppImageMemory<Npp8u> src(width, height, 4);
  NppImageMemory<Npp8u> dst(width, height, 4);

  src.copyFromHost(srcData);

  Npp8u alpha = 128;

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaPremulC_8u_C4R_Ctx(src.get(), src.step(), alpha, dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiAlphaPremulC_8u_C4R(src.get(), src.step(), alpha, dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaPremulC8uC4R, AlphaPremulC8uC4RParamTest,
                         ::testing::Values(AlphaPremulCParam{32, 32, false, "32x32_noCtx"},
                                           AlphaPremulCParam{32, 32, true, "32x32_Ctx"},
                                           AlphaPremulCParam{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaPremulCParam> &info) { return info.param.name; });

// ==================== AlphaPremulC 8u C4IR TEST_P ====================

class AlphaPremulC8uC4IRParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaPremulCParam> {};

TEST_P(AlphaPremulC8uC4IRParamTest, AlphaPremulC_8u_C4IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;

  std::vector<Npp8u> srcData(width * height * channels);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  NppImageMemory<Npp8u> srcDst(width, height, 4);

  srcDst.copyFromHost(srcData);

  Npp8u alpha = 128;

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaPremulC_8u_C4IR_Ctx(alpha, srcDst.get(), srcDst.step(), roi, ctx);
  } else {
    status = nppiAlphaPremulC_8u_C4IR(alpha, srcDst.get(), srcDst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaPremulC8uC4IR, AlphaPremulC8uC4IRParamTest,
                         ::testing::Values(AlphaPremulCParam{32, 32, false, "32x32_noCtx"},
                                           AlphaPremulCParam{32, 32, true, "32x32_Ctx"},
                                           AlphaPremulCParam{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaPremulCParam> &info) { return info.param.name; });

// ==================== AlphaPremulC 8u AC4R TEST_P ====================

class AlphaPremulC8uAC4RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaPremulCParam> {};

TEST_P(AlphaPremulC8uAC4RParamTest, AlphaPremulC_8u_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;

  std::vector<Npp8u> srcData(width * height * channels);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  NppImageMemory<Npp8u> src(width, height, 4);
  NppImageMemory<Npp8u> dst(width, height, 4);

  src.copyFromHost(srcData);

  Npp8u alpha = 128;

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaPremulC_8u_AC4R_Ctx(src.get(), src.step(), alpha, dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiAlphaPremulC_8u_AC4R(src.get(), src.step(), alpha, dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaPremulC8uAC4R, AlphaPremulC8uAC4RParamTest,
                         ::testing::Values(AlphaPremulCParam{32, 32, false, "32x32_noCtx"},
                                           AlphaPremulCParam{32, 32, true, "32x32_Ctx"},
                                           AlphaPremulCParam{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaPremulCParam> &info) { return info.param.name; });

// ==================== AlphaPremulC 8u AC4IR TEST_P ====================

class AlphaPremulC8uAC4IRParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaPremulCParam> {};

TEST_P(AlphaPremulC8uAC4IRParamTest, AlphaPremulC_8u_AC4IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;

  std::vector<Npp8u> srcData(width * height * channels);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);

  NppImageMemory<Npp8u> srcDst(width, height, 4);

  srcDst.copyFromHost(srcData);

  Npp8u alpha = 128;

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaPremulC_8u_AC4IR_Ctx(alpha, srcDst.get(), srcDst.step(), roi, ctx);
  } else {
    status = nppiAlphaPremulC_8u_AC4IR(alpha, srcDst.get(), srcDst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaPremulC8uAC4IR, AlphaPremulC8uAC4IRParamTest,
                         ::testing::Values(AlphaPremulCParam{32, 32, false, "32x32_noCtx"},
                                           AlphaPremulCParam{32, 32, true, "32x32_Ctx"},
                                           AlphaPremulCParam{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaPremulCParam> &info) { return info.param.name; });

// ==================== AlphaPremulC 16u C1R TEST_P ====================

class AlphaPremulC16uC1RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaPremulCParam> {};

TEST_P(AlphaPremulC16uC1RParamTest, AlphaPremulC_16u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);

  NppImageMemory<Npp16u> src(width, height);
  NppImageMemory<Npp16u> dst(width, height);

  src.copyFromHost(srcData);

  Npp16u alpha = 32768;

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaPremulC_16u_C1R_Ctx(src.get(), src.step(), alpha, dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiAlphaPremulC_16u_C1R(src.get(), src.step(), alpha, dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaPremulC16uC1R, AlphaPremulC16uC1RParamTest,
                         ::testing::Values(AlphaPremulCParam{32, 32, false, "32x32_noCtx"},
                                           AlphaPremulCParam{32, 32, true, "32x32_Ctx"},
                                           AlphaPremulCParam{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaPremulCParam> &info) { return info.param.name; });

// ==================== AlphaPremulC 16u C1IR TEST_P ====================

class AlphaPremulC16uC1IRParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaPremulCParam> {};

TEST_P(AlphaPremulC16uC1IRParamTest, AlphaPremulC_16u_C1IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);

  NppImageMemory<Npp16u> srcDst(width, height);

  srcDst.copyFromHost(srcData);

  Npp16u alpha = 32768;

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaPremulC_16u_C1IR_Ctx(alpha, srcDst.get(), srcDst.step(), roi, ctx);
  } else {
    status = nppiAlphaPremulC_16u_C1IR(alpha, srcDst.get(), srcDst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaPremulC16uC1IR, AlphaPremulC16uC1IRParamTest,
                         ::testing::Values(AlphaPremulCParam{32, 32, false, "32x32_noCtx"},
                                           AlphaPremulCParam{32, 32, true, "32x32_Ctx"},
                                           AlphaPremulCParam{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaPremulCParam> &info) { return info.param.name; });

// ==================== AlphaPremulC 16u C3R TEST_P ====================

class AlphaPremulC16uC3RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaPremulCParam> {};

TEST_P(AlphaPremulC16uC3RParamTest, AlphaPremulC_16u_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 3;

  std::vector<Npp16u> srcData(width * height * channels);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);

  NppImageMemory<Npp16u> src(width, height, 3);
  NppImageMemory<Npp16u> dst(width, height, 3);

  src.copyFromHost(srcData);

  Npp16u alpha = 32768;

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaPremulC_16u_C3R_Ctx(src.get(), src.step(), alpha, dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiAlphaPremulC_16u_C3R(src.get(), src.step(), alpha, dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaPremulC16uC3R, AlphaPremulC16uC3RParamTest,
                         ::testing::Values(AlphaPremulCParam{32, 32, false, "32x32_noCtx"},
                                           AlphaPremulCParam{32, 32, true, "32x32_Ctx"},
                                           AlphaPremulCParam{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaPremulCParam> &info) { return info.param.name; });

// ==================== AlphaPremulC 16u C3IR TEST_P ====================

class AlphaPremulC16uC3IRParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaPremulCParam> {};

TEST_P(AlphaPremulC16uC3IRParamTest, AlphaPremulC_16u_C3IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 3;

  std::vector<Npp16u> srcData(width * height * channels);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);

  NppImageMemory<Npp16u> srcDst(width, height, 3);

  srcDst.copyFromHost(srcData);

  Npp16u alpha = 32768;

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaPremulC_16u_C3IR_Ctx(alpha, srcDst.get(), srcDst.step(), roi, ctx);
  } else {
    status = nppiAlphaPremulC_16u_C3IR(alpha, srcDst.get(), srcDst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaPremulC16uC3IR, AlphaPremulC16uC3IRParamTest,
                         ::testing::Values(AlphaPremulCParam{32, 32, false, "32x32_noCtx"},
                                           AlphaPremulCParam{32, 32, true, "32x32_Ctx"},
                                           AlphaPremulCParam{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaPremulCParam> &info) { return info.param.name; });

// ==================== AlphaPremulC 16u C4R TEST_P ====================

class AlphaPremulC16uC4RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaPremulCParam> {};

TEST_P(AlphaPremulC16uC4RParamTest, AlphaPremulC_16u_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;

  std::vector<Npp16u> srcData(width * height * channels);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);

  NppImageMemory<Npp16u> src(width, height, 4);
  NppImageMemory<Npp16u> dst(width, height, 4);

  src.copyFromHost(srcData);

  Npp16u alpha = 32768;

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaPremulC_16u_C4R_Ctx(src.get(), src.step(), alpha, dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiAlphaPremulC_16u_C4R(src.get(), src.step(), alpha, dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaPremulC16uC4R, AlphaPremulC16uC4RParamTest,
                         ::testing::Values(AlphaPremulCParam{32, 32, false, "32x32_noCtx"},
                                           AlphaPremulCParam{32, 32, true, "32x32_Ctx"},
                                           AlphaPremulCParam{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaPremulCParam> &info) { return info.param.name; });

// ==================== AlphaPremulC 16u C4IR TEST_P ====================

class AlphaPremulC16uC4IRParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaPremulCParam> {};

TEST_P(AlphaPremulC16uC4IRParamTest, AlphaPremulC_16u_C4IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;

  std::vector<Npp16u> srcData(width * height * channels);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);

  NppImageMemory<Npp16u> srcDst(width, height, 4);

  srcDst.copyFromHost(srcData);

  Npp16u alpha = 32768;

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaPremulC_16u_C4IR_Ctx(alpha, srcDst.get(), srcDst.step(), roi, ctx);
  } else {
    status = nppiAlphaPremulC_16u_C4IR(alpha, srcDst.get(), srcDst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaPremulC16uC4IR, AlphaPremulC16uC4IRParamTest,
                         ::testing::Values(AlphaPremulCParam{32, 32, false, "32x32_noCtx"},
                                           AlphaPremulCParam{32, 32, true, "32x32_Ctx"},
                                           AlphaPremulCParam{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaPremulCParam> &info) { return info.param.name; });

// ==================== AlphaPremulC 16u AC4R TEST_P ====================

class AlphaPremulC16uAC4RParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaPremulCParam> {};

TEST_P(AlphaPremulC16uAC4RParamTest, AlphaPremulC_16u_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;

  std::vector<Npp16u> srcData(width * height * channels);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);

  NppImageMemory<Npp16u> src(width, height, 4);
  NppImageMemory<Npp16u> dst(width, height, 4);

  src.copyFromHost(srcData);

  Npp16u alpha = 32768;

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaPremulC_16u_AC4R_Ctx(src.get(), src.step(), alpha, dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiAlphaPremulC_16u_AC4R(src.get(), src.step(), alpha, dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaPremulC16uAC4R, AlphaPremulC16uAC4RParamTest,
                         ::testing::Values(AlphaPremulCParam{32, 32, false, "32x32_noCtx"},
                                           AlphaPremulCParam{32, 32, true, "32x32_Ctx"},
                                           AlphaPremulCParam{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaPremulCParam> &info) { return info.param.name; });

// ==================== AlphaPremulC 16u AC4IR TEST_P ====================

class AlphaPremulC16uAC4IRParamTest : public NppTestBase, public ::testing::WithParamInterface<AlphaPremulCParam> {};

TEST_P(AlphaPremulC16uAC4IRParamTest, AlphaPremulC_16u_AC4IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;

  std::vector<Npp16u> srcData(width * height * channels);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);

  NppImageMemory<Npp16u> srcDst(width, height, 4);

  srcDst.copyFromHost(srcData);

  Npp16u alpha = 32768;

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAlphaPremulC_16u_AC4IR_Ctx(alpha, srcDst.get(), srcDst.step(), roi, ctx);
  } else {
    status = nppiAlphaPremulC_16u_AC4IR(alpha, srcDst.get(), srcDst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AlphaPremulC16uAC4IR, AlphaPremulC16uAC4IRParamTest,
                         ::testing::Values(AlphaPremulCParam{32, 32, false, "32x32_noCtx"},
                                           AlphaPremulCParam{32, 32, true, "32x32_Ctx"},
                                           AlphaPremulCParam{64, 64, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AlphaPremulCParam> &info) { return info.param.name; });
