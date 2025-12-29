// MulScale parameterized tests for nppi arithmetic operations
#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Common Param Structures ====================

struct MulScaleParam {
  int width;
  int height;
  bool use_ctx;
  std::string name;
};

// ==================== MulScale 8u C1 TEST_P ====================

class MulScale8uC1RParamTest : public NppTestBase, public ::testing::WithParamInterface<MulScaleParam> {};

TEST_P(MulScale8uC1RParamTest, MulScale_8u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> src1Data(width * height);
  std::vector<Npp8u> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(1), static_cast<Npp8u>(200), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8u>(1), static_cast<Npp8u>(200), 67890);

  NppImageMemory<Npp8u> src1(width, height, 1);
  NppImageMemory<Npp8u> src2(width, height, 1);
  NppImageMemory<Npp8u> dst(width, height, 1);

  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiMulScale_8u_C1R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMulScale_8u_C1R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(MulScale8uC1R, MulScale8uC1RParamTest,
                         ::testing::Values(MulScaleParam{32, 32, false, "32x32_noCtx"},
                                           MulScaleParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<MulScaleParam> &info) { return info.param.name; });

// ==================== MulScale 8u C1IR TEST_P ====================

class MulScale8uC1IRParamTest : public NppTestBase, public ::testing::WithParamInterface<MulScaleParam> {};

TEST_P(MulScale8uC1IRParamTest, MulScale_8u_C1IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height);
  std::vector<Npp8u> srcDstData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(1), static_cast<Npp8u>(200), 12345);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp8u>(1), static_cast<Npp8u>(200), 67890);

  NppImageMemory<Npp8u> src(width, height, 1);
  NppImageMemory<Npp8u> srcDst(width, height, 1);

  src.copyFromHost(srcData);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiMulScale_8u_C1IR_Ctx(src.get(), src.step(), srcDst.get(), srcDst.step(), roi, ctx);
  } else {
    status = nppiMulScale_8u_C1IR(src.get(), src.step(), srcDst.get(), srcDst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(MulScale8uC1IR, MulScale8uC1IRParamTest,
                         ::testing::Values(MulScaleParam{32, 32, false, "32x32_noCtx"},
                                           MulScaleParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<MulScaleParam> &info) { return info.param.name; });

// ==================== MulScale 8u C3 TEST_P ====================

class MulScale8uC3RParamTest : public NppTestBase, public ::testing::WithParamInterface<MulScaleParam> {};

TEST_P(MulScale8uC3RParamTest, MulScale_8u_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> src1Data(width * height * 3);
  std::vector<Npp8u> src2Data(width * height * 3);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(1), static_cast<Npp8u>(200), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8u>(1), static_cast<Npp8u>(200), 67890);

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
    status = nppiMulScale_8u_C3R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMulScale_8u_C3R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(MulScale8uC3R, MulScale8uC3RParamTest,
                         ::testing::Values(MulScaleParam{32, 32, false, "32x32_noCtx"},
                                           MulScaleParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<MulScaleParam> &info) { return info.param.name; });

// ==================== MulScale 8u C3IR TEST_P ====================

class MulScale8uC3IRParamTest : public NppTestBase, public ::testing::WithParamInterface<MulScaleParam> {};

TEST_P(MulScale8uC3IRParamTest, MulScale_8u_C3IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height * 3);
  std::vector<Npp8u> srcDstData(width * height * 3);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(1), static_cast<Npp8u>(200), 12345);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp8u>(1), static_cast<Npp8u>(200), 67890);

  NppImageMemory<Npp8u> src(width, height, 3);
  NppImageMemory<Npp8u> srcDst(width, height, 3);

  src.copyFromHost(srcData);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiMulScale_8u_C3IR_Ctx(src.get(), src.step(), srcDst.get(), srcDst.step(), roi, ctx);
  } else {
    status = nppiMulScale_8u_C3IR(src.get(), src.step(), srcDst.get(), srcDst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(MulScale8uC3IR, MulScale8uC3IRParamTest,
                         ::testing::Values(MulScaleParam{32, 32, false, "32x32_noCtx"},
                                           MulScaleParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<MulScaleParam> &info) { return info.param.name; });

// ==================== MulScale 8u C4 TEST_P ====================

class MulScale8uC4RParamTest : public NppTestBase, public ::testing::WithParamInterface<MulScaleParam> {};

TEST_P(MulScale8uC4RParamTest, MulScale_8u_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> src1Data(width * height * 4);
  std::vector<Npp8u> src2Data(width * height * 4);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(1), static_cast<Npp8u>(200), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8u>(1), static_cast<Npp8u>(200), 67890);

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
    status = nppiMulScale_8u_C4R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMulScale_8u_C4R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(MulScale8uC4R, MulScale8uC4RParamTest,
                         ::testing::Values(MulScaleParam{32, 32, false, "32x32_noCtx"},
                                           MulScaleParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<MulScaleParam> &info) { return info.param.name; });

// ==================== MulScale 8u C4IR TEST_P ====================

class MulScale8uC4IRParamTest : public NppTestBase, public ::testing::WithParamInterface<MulScaleParam> {};

TEST_P(MulScale8uC4IRParamTest, MulScale_8u_C4IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height * 4);
  std::vector<Npp8u> srcDstData(width * height * 4);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(1), static_cast<Npp8u>(200), 12345);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp8u>(1), static_cast<Npp8u>(200), 67890);

  NppImageMemory<Npp8u> src(width, height, 4);
  NppImageMemory<Npp8u> srcDst(width, height, 4);

  src.copyFromHost(srcData);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiMulScale_8u_C4IR_Ctx(src.get(), src.step(), srcDst.get(), srcDst.step(), roi, ctx);
  } else {
    status = nppiMulScale_8u_C4IR(src.get(), src.step(), srcDst.get(), srcDst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(MulScale8uC4IR, MulScale8uC4IRParamTest,
                         ::testing::Values(MulScaleParam{32, 32, false, "32x32_noCtx"},
                                           MulScaleParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<MulScaleParam> &info) { return info.param.name; });

// ==================== MulScale 8u AC4 TEST_P ====================

class MulScale8uAC4RParamTest : public NppTestBase, public ::testing::WithParamInterface<MulScaleParam> {};

TEST_P(MulScale8uAC4RParamTest, MulScale_8u_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> src1Data(width * height * 4);
  std::vector<Npp8u> src2Data(width * height * 4);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(1), static_cast<Npp8u>(200), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8u>(1), static_cast<Npp8u>(200), 67890);

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
    status =
        nppiMulScale_8u_AC4R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMulScale_8u_AC4R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(MulScale8uAC4R, MulScale8uAC4RParamTest,
                         ::testing::Values(MulScaleParam{32, 32, false, "32x32_noCtx"},
                                           MulScaleParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<MulScaleParam> &info) { return info.param.name; });

// ==================== MulScale 8u AC4IR TEST_P ====================

class MulScale8uAC4IRParamTest : public NppTestBase, public ::testing::WithParamInterface<MulScaleParam> {};

TEST_P(MulScale8uAC4IRParamTest, MulScale_8u_AC4IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height * 4);
  std::vector<Npp8u> srcDstData(width * height * 4);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(1), static_cast<Npp8u>(200), 12345);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp8u>(1), static_cast<Npp8u>(200), 67890);

  NppImageMemory<Npp8u> src(width, height, 4);
  NppImageMemory<Npp8u> srcDst(width, height, 4);

  src.copyFromHost(srcData);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiMulScale_8u_AC4IR_Ctx(src.get(), src.step(), srcDst.get(), srcDst.step(), roi, ctx);
  } else {
    status = nppiMulScale_8u_AC4IR(src.get(), src.step(), srcDst.get(), srcDst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(MulScale8uAC4IR, MulScale8uAC4IRParamTest,
                         ::testing::Values(MulScaleParam{32, 32, false, "32x32_noCtx"},
                                           MulScaleParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<MulScaleParam> &info) { return info.param.name; });

// ==================== MulScale 16u C1 TEST_P ====================

class MulScale16uC1RParamTest : public NppTestBase, public ::testing::WithParamInterface<MulScaleParam> {};

TEST_P(MulScale16uC1RParamTest, MulScale_16u_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> src1Data(width * height);
  std::vector<Npp16u> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(1), static_cast<Npp16u>(50000), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(1), static_cast<Npp16u>(50000), 67890);

  NppImageMemory<Npp16u> src1(width, height, 1);
  NppImageMemory<Npp16u> src2(width, height, 1);
  NppImageMemory<Npp16u> dst(width, height, 1);

  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status =
        nppiMulScale_16u_C1R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMulScale_16u_C1R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(MulScale16uC1R, MulScale16uC1RParamTest,
                         ::testing::Values(MulScaleParam{32, 32, false, "32x32_noCtx"},
                                           MulScaleParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<MulScaleParam> &info) { return info.param.name; });

// ==================== MulScale 16u C1IR TEST_P ====================

class MulScale16uC1IRParamTest : public NppTestBase, public ::testing::WithParamInterface<MulScaleParam> {};

TEST_P(MulScale16uC1IRParamTest, MulScale_16u_C1IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcData(width * height);
  std::vector<Npp16u> srcDstData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(1), static_cast<Npp16u>(50000), 12345);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp16u>(1), static_cast<Npp16u>(50000), 67890);

  NppImageMemory<Npp16u> src(width, height, 1);
  NppImageMemory<Npp16u> srcDst(width, height, 1);

  src.copyFromHost(srcData);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiMulScale_16u_C1IR_Ctx(src.get(), src.step(), srcDst.get(), srcDst.step(), roi, ctx);
  } else {
    status = nppiMulScale_16u_C1IR(src.get(), src.step(), srcDst.get(), srcDst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(MulScale16uC1IR, MulScale16uC1IRParamTest,
                         ::testing::Values(MulScaleParam{32, 32, false, "32x32_noCtx"},
                                           MulScaleParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<MulScaleParam> &info) { return info.param.name; });

// ==================== MulScale 16u C3 TEST_P ====================

class MulScale16uC3RParamTest : public NppTestBase, public ::testing::WithParamInterface<MulScaleParam> {};

TEST_P(MulScale16uC3RParamTest, MulScale_16u_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> src1Data(width * height * 3);
  std::vector<Npp16u> src2Data(width * height * 3);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(1), static_cast<Npp16u>(50000), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(1), static_cast<Npp16u>(50000), 67890);

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
    status =
        nppiMulScale_16u_C3R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMulScale_16u_C3R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(MulScale16uC3R, MulScale16uC3RParamTest,
                         ::testing::Values(MulScaleParam{32, 32, false, "32x32_noCtx"},
                                           MulScaleParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<MulScaleParam> &info) { return info.param.name; });

// ==================== MulScale 16u C3IR TEST_P ====================

class MulScale16uC3IRParamTest : public NppTestBase, public ::testing::WithParamInterface<MulScaleParam> {};

TEST_P(MulScale16uC3IRParamTest, MulScale_16u_C3IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcData(width * height * 3);
  std::vector<Npp16u> srcDstData(width * height * 3);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(1), static_cast<Npp16u>(50000), 12345);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp16u>(1), static_cast<Npp16u>(50000), 67890);

  NppImageMemory<Npp16u> src(width, height, 3);
  NppImageMemory<Npp16u> srcDst(width, height, 3);

  src.copyFromHost(srcData);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiMulScale_16u_C3IR_Ctx(src.get(), src.step(), srcDst.get(), srcDst.step(), roi, ctx);
  } else {
    status = nppiMulScale_16u_C3IR(src.get(), src.step(), srcDst.get(), srcDst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(MulScale16uC3IR, MulScale16uC3IRParamTest,
                         ::testing::Values(MulScaleParam{32, 32, false, "32x32_noCtx"},
                                           MulScaleParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<MulScaleParam> &info) { return info.param.name; });

// ==================== MulScale 16u C4 TEST_P ====================

class MulScale16uC4RParamTest : public NppTestBase, public ::testing::WithParamInterface<MulScaleParam> {};

TEST_P(MulScale16uC4RParamTest, MulScale_16u_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> src1Data(width * height * 4);
  std::vector<Npp16u> src2Data(width * height * 4);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(1), static_cast<Npp16u>(50000), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(1), static_cast<Npp16u>(50000), 67890);

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
    status =
        nppiMulScale_16u_C4R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMulScale_16u_C4R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(MulScale16uC4R, MulScale16uC4RParamTest,
                         ::testing::Values(MulScaleParam{32, 32, false, "32x32_noCtx"},
                                           MulScaleParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<MulScaleParam> &info) { return info.param.name; });

// ==================== MulScale 16u C4IR TEST_P ====================

class MulScale16uC4IRParamTest : public NppTestBase, public ::testing::WithParamInterface<MulScaleParam> {};

TEST_P(MulScale16uC4IRParamTest, MulScale_16u_C4IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcData(width * height * 4);
  std::vector<Npp16u> srcDstData(width * height * 4);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(1), static_cast<Npp16u>(50000), 12345);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp16u>(1), static_cast<Npp16u>(50000), 67890);

  NppImageMemory<Npp16u> src(width, height, 4);
  NppImageMemory<Npp16u> srcDst(width, height, 4);

  src.copyFromHost(srcData);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiMulScale_16u_C4IR_Ctx(src.get(), src.step(), srcDst.get(), srcDst.step(), roi, ctx);
  } else {
    status = nppiMulScale_16u_C4IR(src.get(), src.step(), srcDst.get(), srcDst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(MulScale16uC4IR, MulScale16uC4IRParamTest,
                         ::testing::Values(MulScaleParam{32, 32, false, "32x32_noCtx"},
                                           MulScaleParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<MulScaleParam> &info) { return info.param.name; });

// ==================== MulScale 16u AC4 TEST_P ====================

class MulScale16uAC4RParamTest : public NppTestBase, public ::testing::WithParamInterface<MulScaleParam> {};

TEST_P(MulScale16uAC4RParamTest, MulScale_16u_AC4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> src1Data(width * height * 4);
  std::vector<Npp16u> src2Data(width * height * 4);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(1), static_cast<Npp16u>(50000), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(1), static_cast<Npp16u>(50000), 67890);

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
    status =
        nppiMulScale_16u_AC4R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
  } else {
    status = nppiMulScale_16u_AC4R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(MulScale16uAC4R, MulScale16uAC4RParamTest,
                         ::testing::Values(MulScaleParam{32, 32, false, "32x32_noCtx"},
                                           MulScaleParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<MulScaleParam> &info) { return info.param.name; });

// ==================== MulScale 16u AC4IR TEST_P ====================

class MulScale16uAC4IRParamTest : public NppTestBase, public ::testing::WithParamInterface<MulScaleParam> {};

TEST_P(MulScale16uAC4IRParamTest, MulScale_16u_AC4IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcData(width * height * 4);
  std::vector<Npp16u> srcDstData(width * height * 4);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(1), static_cast<Npp16u>(50000), 12345);
  TestDataGenerator::generateRandom(srcDstData, static_cast<Npp16u>(1), static_cast<Npp16u>(50000), 67890);

  NppImageMemory<Npp16u> src(width, height, 4);
  NppImageMemory<Npp16u> srcDst(width, height, 4);

  src.copyFromHost(srcData);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiMulScale_16u_AC4IR_Ctx(src.get(), src.step(), srcDst.get(), srcDst.step(), roi, ctx);
  } else {
    status = nppiMulScale_16u_AC4IR(src.get(), src.step(), srcDst.get(), srcDst.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(MulScale16uAC4IR, MulScale16uAC4IRParamTest,
                         ::testing::Values(MulScaleParam{32, 32, false, "32x32_noCtx"},
                                           MulScaleParam{32, 32, true, "32x32_Ctx"}),
                         [](const ::testing::TestParamInfo<MulScaleParam> &info) { return info.param.name; });
