// MulCScale parameterized tests for nppi arithmetic operations
#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Common Param Structures ====================

struct MulCScaleParam {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

// ==================== MulCScale 8u C1R TEST_P ====================

class MulCScale8uC1RParamTest : public NppTestBase, public ::testing::WithParamInterface<MulCScaleParam> {};

TEST_P(MulCScale8uC1RParamTest, MulCScale_8u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(1), static_cast<Npp8u>(255), 12345);

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  Npp8u nConstant = 2;
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiMulCScale_8u_C1R_Ctx(src.get(), src.step(), nConstant, dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMulCScale_8u_C1R(src.get(), src.step(), nConstant, dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(MulCScale8uC1R, MulCScale8uC1RParamTest,
                         ::testing::Values(MulCScaleParam{32, 32, false, "32x32_noCtx"},
                                           MulCScaleParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<MulCScaleParam> &info) { return info.param.name; });

// ==================== MulCScale 8u C1IR TEST_P ====================

class MulCScale8uC1IRParamTest : public NppTestBase, public ::testing::WithParamInterface<MulCScaleParam> {};

TEST_P(MulCScale8uC1IRParamTest, MulCScale_8u_C1IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcDstData(width * height);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp8u>(1), static_cast<Npp8u>(255), 12345);

  NppImageMemory<Npp8u> srcDst(width, height);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  Npp8u nConstant = 2;
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiMulCScale_8u_C1IR_Ctx(nConstant, srcDst.get(), srcDst.step(), roi, ctx);
  } else {
    status = nppiMulCScale_8u_C1IR(nConstant, srcDst.get(), srcDst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(MulCScale8uC1IR, MulCScale8uC1IRParamTest,
                         ::testing::Values(MulCScaleParam{32, 32, false, "32x32_noCtx"},
                                           MulCScaleParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<MulCScaleParam> &info) { return info.param.name; });

// ==================== MulCScale 8u C3R TEST_P ====================

class MulCScale8uC3RParamTest : public NppTestBase, public ::testing::WithParamInterface<MulCScaleParam> {};

TEST_P(MulCScale8uC3RParamTest, MulCScale_8u_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height * 3);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(1), static_cast<Npp8u>(255), 12345);

  NppImageMemory<Npp8u> src(width, height, 3);
  NppImageMemory<Npp8u> dst(width, height, 3);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  Npp8u aConstants[3] = {2, 3, 4};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiMulCScale_8u_C3R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMulCScale_8u_C3R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(MulCScale8uC3R, MulCScale8uC3RParamTest,
                         ::testing::Values(MulCScaleParam{32, 32, false, "32x32_noCtx"},
                                           MulCScaleParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<MulCScaleParam> &info) { return info.param.name; });

// ==================== MulCScale 8u C3IR TEST_P ====================

class MulCScale8uC3IRParamTest : public NppTestBase, public ::testing::WithParamInterface<MulCScaleParam> {};

TEST_P(MulCScale8uC3IRParamTest, MulCScale_8u_C3IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcDstData(width * height * 3);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp8u>(1), static_cast<Npp8u>(255), 12345);

  NppImageMemory<Npp8u> srcDst(width, height, 3);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  Npp8u aConstants[3] = {2, 3, 4};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiMulCScale_8u_C3IR_Ctx(aConstants, srcDst.get(), srcDst.step(), roi, ctx);
  } else {
    status = nppiMulCScale_8u_C3IR(aConstants, srcDst.get(), srcDst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(MulCScale8uC3IR, MulCScale8uC3IRParamTest,
                         ::testing::Values(MulCScaleParam{32, 32, false, "32x32_noCtx"},
                                           MulCScaleParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<MulCScaleParam> &info) { return info.param.name; });

// ==================== MulCScale 8u C4R TEST_P ====================

class MulCScale8uC4RParamTest : public NppTestBase, public ::testing::WithParamInterface<MulCScaleParam> {};

TEST_P(MulCScale8uC4RParamTest, MulCScale_8u_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height * 4);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(1), static_cast<Npp8u>(255), 12345);

  NppImageMemory<Npp8u> src(width, height, 4);
  NppImageMemory<Npp8u> dst(width, height, 4);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  Npp8u aConstants[4] = {2, 3, 4, 5};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiMulCScale_8u_C4R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMulCScale_8u_C4R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(MulCScale8uC4R, MulCScale8uC4RParamTest,
                         ::testing::Values(MulCScaleParam{32, 32, false, "32x32_noCtx"},
                                           MulCScaleParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<MulCScaleParam> &info) { return info.param.name; });

// ==================== MulCScale 8u C4IR TEST_P ====================

class MulCScale8uC4IRParamTest : public NppTestBase, public ::testing::WithParamInterface<MulCScaleParam> {};

TEST_P(MulCScale8uC4IRParamTest, MulCScale_8u_C4IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcDstData(width * height * 4);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp8u>(1), static_cast<Npp8u>(255), 12345);

  NppImageMemory<Npp8u> srcDst(width, height, 4);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  Npp8u aConstants[4] = {2, 3, 4, 5};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiMulCScale_8u_C4IR_Ctx(aConstants, srcDst.get(), srcDst.step(), roi, ctx);
  } else {
    status = nppiMulCScale_8u_C4IR(aConstants, srcDst.get(), srcDst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(MulCScale8uC4IR, MulCScale8uC4IRParamTest,
                         ::testing::Values(MulCScaleParam{32, 32, false, "32x32_noCtx"},
                                           MulCScaleParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<MulCScaleParam> &info) { return info.param.name; });

// ==================== MulCScale 8u AC4R TEST_P ====================

class MulCScale8uAC4RParamTest : public NppTestBase, public ::testing::WithParamInterface<MulCScaleParam> {};

TEST_P(MulCScale8uAC4RParamTest, MulCScale_8u_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height * 4);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(1), static_cast<Npp8u>(255), 12345);

  NppImageMemory<Npp8u> src(width, height, 4);
  NppImageMemory<Npp8u> dst(width, height, 4);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  Npp8u aConstants[3] = {2, 3, 4};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiMulCScale_8u_AC4R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMulCScale_8u_AC4R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(MulCScale8uAC4R, MulCScale8uAC4RParamTest,
                         ::testing::Values(MulCScaleParam{32, 32, false, "32x32_noCtx"},
                                           MulCScaleParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<MulCScaleParam> &info) { return info.param.name; });

// ==================== MulCScale 8u AC4IR TEST_P ====================

class MulCScale8uAC4IRParamTest : public NppTestBase, public ::testing::WithParamInterface<MulCScaleParam> {};

TEST_P(MulCScale8uAC4IRParamTest, MulCScale_8u_AC4IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcDstData(width * height * 4);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp8u>(1), static_cast<Npp8u>(255), 12345);

  NppImageMemory<Npp8u> srcDst(width, height, 4);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  Npp8u aConstants[3] = {2, 3, 4};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiMulCScale_8u_AC4IR_Ctx(aConstants, srcDst.get(), srcDst.step(), roi, ctx);
  } else {
    status = nppiMulCScale_8u_AC4IR(aConstants, srcDst.get(), srcDst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(MulCScale8uAC4IR, MulCScale8uAC4IRParamTest,
                         ::testing::Values(MulCScaleParam{32, 32, false, "32x32_noCtx"},
                                           MulCScaleParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<MulCScaleParam> &info) { return info.param.name; });

// ==================== MulCScale 16u C1R TEST_P ====================

class MulCScale16uC1RParamTest : public NppTestBase, public ::testing::WithParamInterface<MulCScaleParam> {};

TEST_P(MulCScale16uC1RParamTest, MulCScale_16u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(1), static_cast<Npp16u>(65535), 12345);

  NppImageMemory<Npp16u> src(width, height);
  NppImageMemory<Npp16u> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  Npp16u nConstant = 2;
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiMulCScale_16u_C1R_Ctx(src.get(), src.step(), nConstant, dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMulCScale_16u_C1R(src.get(), src.step(), nConstant, dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(MulCScale16uC1R, MulCScale16uC1RParamTest,
                         ::testing::Values(MulCScaleParam{32, 32, false, "32x32_noCtx"},
                                           MulCScaleParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<MulCScaleParam> &info) { return info.param.name; });

// ==================== MulCScale 16u C1IR TEST_P ====================

class MulCScale16uC1IRParamTest : public NppTestBase, public ::testing::WithParamInterface<MulCScaleParam> {};

TEST_P(MulCScale16uC1IRParamTest, MulCScale_16u_C1IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcDstData(width * height);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp16u>(1), static_cast<Npp16u>(65535), 12345);

  NppImageMemory<Npp16u> srcDst(width, height);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  Npp16u nConstant = 2;
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiMulCScale_16u_C1IR_Ctx(nConstant, srcDst.get(), srcDst.step(), roi, ctx);
  } else {
    status = nppiMulCScale_16u_C1IR(nConstant, srcDst.get(), srcDst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(MulCScale16uC1IR, MulCScale16uC1IRParamTest,
                         ::testing::Values(MulCScaleParam{32, 32, false, "32x32_noCtx"},
                                           MulCScaleParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<MulCScaleParam> &info) { return info.param.name; });

// ==================== MulCScale 16u C3R TEST_P ====================

class MulCScale16uC3RParamTest : public NppTestBase, public ::testing::WithParamInterface<MulCScaleParam> {};

TEST_P(MulCScale16uC3RParamTest, MulCScale_16u_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcData(width * height * 3);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(1), static_cast<Npp16u>(65535), 12345);

  NppImageMemory<Npp16u> src(width, height, 3);
  NppImageMemory<Npp16u> dst(width, height, 3);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  Npp16u aConstants[3] = {2, 3, 4};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiMulCScale_16u_C3R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMulCScale_16u_C3R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(MulCScale16uC3R, MulCScale16uC3RParamTest,
                         ::testing::Values(MulCScaleParam{32, 32, false, "32x32_noCtx"},
                                           MulCScaleParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<MulCScaleParam> &info) { return info.param.name; });

// ==================== MulCScale 16u C3IR TEST_P ====================

class MulCScale16uC3IRParamTest : public NppTestBase, public ::testing::WithParamInterface<MulCScaleParam> {};

TEST_P(MulCScale16uC3IRParamTest, MulCScale_16u_C3IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcDstData(width * height * 3);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp16u>(1), static_cast<Npp16u>(65535), 12345);

  NppImageMemory<Npp16u> srcDst(width, height, 3);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  Npp16u aConstants[3] = {2, 3, 4};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiMulCScale_16u_C3IR_Ctx(aConstants, srcDst.get(), srcDst.step(), roi, ctx);
  } else {
    status = nppiMulCScale_16u_C3IR(aConstants, srcDst.get(), srcDst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(MulCScale16uC3IR, MulCScale16uC3IRParamTest,
                         ::testing::Values(MulCScaleParam{32, 32, false, "32x32_noCtx"},
                                           MulCScaleParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<MulCScaleParam> &info) { return info.param.name; });

// ==================== MulCScale 16u C4R TEST_P ====================

class MulCScale16uC4RParamTest : public NppTestBase, public ::testing::WithParamInterface<MulCScaleParam> {};

TEST_P(MulCScale16uC4RParamTest, MulCScale_16u_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcData(width * height * 4);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(1), static_cast<Npp16u>(65535), 12345);

  NppImageMemory<Npp16u> src(width, height, 4);
  NppImageMemory<Npp16u> dst(width, height, 4);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  Npp16u aConstants[4] = {2, 3, 4, 5};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiMulCScale_16u_C4R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMulCScale_16u_C4R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(MulCScale16uC4R, MulCScale16uC4RParamTest,
                         ::testing::Values(MulCScaleParam{32, 32, false, "32x32_noCtx"},
                                           MulCScaleParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<MulCScaleParam> &info) { return info.param.name; });

// ==================== MulCScale 16u C4IR TEST_P ====================

class MulCScale16uC4IRParamTest : public NppTestBase, public ::testing::WithParamInterface<MulCScaleParam> {};

TEST_P(MulCScale16uC4IRParamTest, MulCScale_16u_C4IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcDstData(width * height * 4);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp16u>(1), static_cast<Npp16u>(65535), 12345);

  NppImageMemory<Npp16u> srcDst(width, height, 4);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  Npp16u aConstants[4] = {2, 3, 4, 5};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiMulCScale_16u_C4IR_Ctx(aConstants, srcDst.get(), srcDst.step(), roi, ctx);
  } else {
    status = nppiMulCScale_16u_C4IR(aConstants, srcDst.get(), srcDst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(MulCScale16uC4IR, MulCScale16uC4IRParamTest,
                         ::testing::Values(MulCScaleParam{32, 32, false, "32x32_noCtx"},
                                           MulCScaleParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<MulCScaleParam> &info) { return info.param.name; });

// ==================== MulCScale 16u AC4R TEST_P ====================

class MulCScale16uAC4RParamTest : public NppTestBase, public ::testing::WithParamInterface<MulCScaleParam> {};

TEST_P(MulCScale16uAC4RParamTest, MulCScale_16u_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcData(width * height * 4);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(1), static_cast<Npp16u>(65535), 12345);

  NppImageMemory<Npp16u> src(width, height, 4);
  NppImageMemory<Npp16u> dst(width, height, 4);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  Npp16u aConstants[3] = {2, 3, 4};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiMulCScale_16u_AC4R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMulCScale_16u_AC4R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(MulCScale16uAC4R, MulCScale16uAC4RParamTest,
                         ::testing::Values(MulCScaleParam{32, 32, false, "32x32_noCtx"},
                                           MulCScaleParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<MulCScaleParam> &info) { return info.param.name; });

// ==================== MulCScale 16u AC4IR TEST_P ====================

class MulCScale16uAC4IRParamTest : public NppTestBase, public ::testing::WithParamInterface<MulCScaleParam> {};

TEST_P(MulCScale16uAC4IRParamTest, MulCScale_16u_AC4IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcDstData(width * height * 4);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp16u>(1), static_cast<Npp16u>(65535), 12345);

  NppImageMemory<Npp16u> srcDst(width, height, 4);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  Npp16u aConstants[3] = {2, 3, 4};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiMulCScale_16u_AC4IR_Ctx(aConstants, srcDst.get(), srcDst.step(), roi, ctx);
  } else {
    status = nppiMulCScale_16u_AC4IR(aConstants, srcDst.get(), srcDst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(MulCScale16uAC4IR, MulCScale16uAC4IRParamTest,
                         ::testing::Values(MulCScaleParam{32, 32, false, "32x32_noCtx"},
                                           MulCScaleParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<MulCScaleParam> &info) { return info.param.name; });
