#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Sub 32f TEST_P ====================

struct Sub32fParam {
  int width;
  int height;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Sub32fParamTest : public NppTestBase, public ::testing::WithParamInterface<Sub32fParam> {};

TEST_P(Sub32fParamTest, Sub_32f_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> src1Data(width * height);
  std::vector<Npp32f> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, -50.0f, 50.0f, 12345);
  TestDataGenerator::generateRandom(src2Data, -50.0f, 50.0f, 54321);

  std::vector<Npp32f> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    // NPP Sub: dst = src2 - src1
    expectedData[i] = expect::sub<Npp32f>(src2Data[i], src1Data[i]);
  }

  NppImageMemory<Npp32f> src1(width, height);
  NppImageMemory<Npp32f> src2(width, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiSub_32f_C1IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx);
    } else {
      status = nppiSub_32f_C1IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> resultData(width * height);
    src2.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
  } else {
    NppImageMemory<Npp32f> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiSub_32f_C1R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiSub_32f_C1R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
  }
}

INSTANTIATE_TEST_SUITE_P(Sub32f, Sub32fParamTest,
                         ::testing::Values(Sub32fParam{32, 32, false, false, "32x32_noCtx"},
                                           Sub32fParam{32, 32, true, false, "32x32_Ctx"},
                                           Sub32fParam{32, 32, false, true, "32x32_InPlace"},
                                           Sub32fParam{32, 32, true, true, "32x32_InPlace_Ctx"},
                                           Sub32fParam{64, 64, false, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<Sub32fParam> &info) { return info.param.name; });

// ==================== Sub 8u with scale factor TEST_P ====================

struct Sub8uSfsParam {
  int width;
  int height;
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Sub8uSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<Sub8uSfsParam> {};

TEST_P(Sub8uSfsParamTest, Sub_8u_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int scaleFactor = param.scaleFactor;

  std::vector<Npp8u> src1Data(width * height);
  std::vector<Npp8u> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(50), static_cast<Npp8u>(200), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8u>(1), static_cast<Npp8u>(50), 54321);

  std::vector<Npp8u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    // NPP Sub: dst = src2 - src1
    expectedData[i] = expect::sub_sfs<Npp8u>(src2Data[i], src1Data[i], scaleFactor);
  }

  NppImageMemory<Npp8u> src1(width, height);
  NppImageMemory<Npp8u> src2(width, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiSub_8u_C1IRSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiSub_8u_C1IRSfs(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(width * height);
    src2.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp8u> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiSub_8u_C1RSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                     scaleFactor, ctx);
    } else {
      status =
          nppiSub_8u_C1RSfs(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Sub8uSfs, Sub8uSfsParamTest,
                         ::testing::Values(Sub8uSfsParam{32, 32, 0, false, false, "32x32_sfs0_noCtx"},
                                           Sub8uSfsParam{32, 32, 0, true, false, "32x32_sfs0_Ctx"},
                                           Sub8uSfsParam{32, 32, 0, false, true, "32x32_sfs0_InPlace"},
                                           Sub8uSfsParam{32, 32, 0, true, true, "32x32_sfs0_InPlace_Ctx"},
                                           Sub8uSfsParam{64, 64, 0, false, false, "64x64_sfs0_noCtx"}),
                         [](const ::testing::TestParamInfo<Sub8uSfsParam> &info) { return info.param.name; });

// ==================== Sub 16u with scale factor TEST_P ====================

struct Sub16uSfsParam {
  int width;
  int height;
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Sub16uSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<Sub16uSfsParam> {};

TEST_P(Sub16uSfsParamTest, Sub_16u_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int scaleFactor = param.scaleFactor;

  std::vector<Npp16u> src1Data(width * height);
  std::vector<Npp16u> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(500), static_cast<Npp16u>(2000), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(1), static_cast<Npp16u>(500), 54321);

  std::vector<Npp16u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    // NPP Sub: dst = src2 - src1
    expectedData[i] = expect::sub_sfs<Npp16u>(src2Data[i], src1Data[i], scaleFactor);
  }

  NppImageMemory<Npp16u> src1(width, height);
  NppImageMemory<Npp16u> src2(width, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiSub_16u_C1IRSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiSub_16u_C1IRSfs(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(width * height);
    src2.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16u> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiSub_16u_C1RSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                      scaleFactor, ctx);
    } else {
      status =
          nppiSub_16u_C1RSfs(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Sub16uSfs, Sub16uSfsParamTest,
                         ::testing::Values(Sub16uSfsParam{32, 32, 0, false, false, "32x32_sfs0_noCtx"},
                                           Sub16uSfsParam{32, 32, 0, true, false, "32x32_sfs0_Ctx"},
                                           Sub16uSfsParam{32, 32, 0, false, true, "32x32_sfs0_InPlace"},
                                           Sub16uSfsParam{32, 32, 0, true, true, "32x32_sfs0_InPlace_Ctx"},
                                           Sub16uSfsParam{64, 64, 0, false, false, "64x64_sfs0_noCtx"}),
                         [](const ::testing::TestParamInfo<Sub16uSfsParam> &info) { return info.param.name; });

// ==================== Sub 16s with scale factor TEST_P ====================

struct Sub16sSfsParam {
  int width;
  int height;
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Sub16sSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<Sub16sSfsParam> {};

TEST_P(Sub16sSfsParamTest, Sub_16s_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int scaleFactor = param.scaleFactor;

  std::vector<Npp16s> src1Data(width * height);
  std::vector<Npp16s> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16s>(-100), static_cast<Npp16s>(100), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16s>(-50), static_cast<Npp16s>(50), 54321);

  std::vector<Npp16s> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    // NPP Sub: dst = src2 - src1
    expectedData[i] = expect::sub_sfs<Npp16s>(src2Data[i], src1Data[i], scaleFactor);
  }

  NppImageMemory<Npp16s> src1(width, height);
  NppImageMemory<Npp16s> src2(width, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiSub_16s_C1IRSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiSub_16s_C1IRSfs(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16s> resultData(width * height);
    src2.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16s> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiSub_16s_C1RSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                      scaleFactor, ctx);
    } else {
      status =
          nppiSub_16s_C1RSfs(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16s> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Sub16sSfs, Sub16sSfsParamTest,
                         ::testing::Values(Sub16sSfsParam{32, 32, 0, false, false, "32x32_sfs0_noCtx"},
                                           Sub16sSfsParam{32, 32, 0, true, false, "32x32_sfs0_Ctx"},
                                           Sub16sSfsParam{32, 32, 0, false, true, "32x32_sfs0_InPlace"},
                                           Sub16sSfsParam{32, 32, 0, true, true, "32x32_sfs0_InPlace_Ctx"},
                                           Sub16sSfsParam{64, 64, 0, false, false, "64x64_sfs0_noCtx"}),
                         [](const ::testing::TestParamInfo<Sub16sSfsParam> &info) { return info.param.name; });

// ==================== Sub 32f C3 TEST_P ====================

struct Sub32fC3Param {
  int width;
  int height;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Sub32fC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<Sub32fC3Param> {};

TEST_P(Sub32fC3ParamTest, Sub_32f_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 3;

  std::vector<Npp32f> src1Data(width * height * channels);
  std::vector<Npp32f> src2Data(width * height * channels);
  TestDataGenerator::generateRandom(src1Data, -50.0f, 50.0f, 12345);
  TestDataGenerator::generateRandom(src2Data, -50.0f, 50.0f, 54321);

  std::vector<Npp32f> expectedData(width * height * channels);
  for (size_t i = 0; i < expectedData.size(); i++) {
    // NPP Sub: dst = src2 - src1
    expectedData[i] = expect::sub<Npp32f>(src2Data[i], src1Data[i]);
  }

  NppImageMemory<Npp32f> src1(width * channels, height);
  NppImageMemory<Npp32f> src2(width * channels, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiSub_32f_C3IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx);
    } else {
      status = nppiSub_32f_C3IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> resultData(width * height * channels);
    src2.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
  } else {
    NppImageMemory<Npp32f> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiSub_32f_C3R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiSub_32f_C3R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> resultData(width * height * channels);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
  }
}

INSTANTIATE_TEST_SUITE_P(Sub32fC3, Sub32fC3ParamTest,
                         ::testing::Values(Sub32fC3Param{32, 32, false, false, "32x32_noCtx"},
                                           Sub32fC3Param{32, 32, true, false, "32x32_Ctx"},
                                           Sub32fC3Param{32, 32, false, true, "32x32_InPlace"},
                                           Sub32fC3Param{32, 32, true, true, "32x32_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<Sub32fC3Param> &info) { return info.param.name; });

// ==================== Sub 32f C4 TEST_P ====================

struct Sub32fC4Param {
  int width;
  int height;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Sub32fC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<Sub32fC4Param> {};

TEST_P(Sub32fC4ParamTest, Sub_32f_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;

  std::vector<Npp32f> src1Data(width * height * channels);
  std::vector<Npp32f> src2Data(width * height * channels);
  TestDataGenerator::generateRandom(src1Data, -50.0f, 50.0f, 12345);
  TestDataGenerator::generateRandom(src2Data, -50.0f, 50.0f, 54321);

  std::vector<Npp32f> expectedData(width * height * channels);
  for (size_t i = 0; i < expectedData.size(); i++) {
    // NPP Sub: dst = src2 - src1
    expectedData[i] = expect::sub<Npp32f>(src2Data[i], src1Data[i]);
  }

  NppImageMemory<Npp32f> src1(width * channels, height);
  NppImageMemory<Npp32f> src2(width * channels, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiSub_32f_C4IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx);
    } else {
      status = nppiSub_32f_C4IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> resultData(width * height * channels);
    src2.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
  } else {
    NppImageMemory<Npp32f> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiSub_32f_C4R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiSub_32f_C4R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> resultData(width * height * channels);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
  }
}

INSTANTIATE_TEST_SUITE_P(Sub32fC4, Sub32fC4ParamTest,
                         ::testing::Values(Sub32fC4Param{32, 32, false, false, "32x32_noCtx"},
                                           Sub32fC4Param{32, 32, true, false, "32x32_Ctx"},
                                           Sub32fC4Param{32, 32, false, true, "32x32_InPlace"},
                                           Sub32fC4Param{32, 32, true, true, "32x32_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<Sub32fC4Param> &info) { return info.param.name; });

// ==================== Sub 16u C3 with scale factor TEST_P ====================

class Sub16uC3SfsParamTest : public NppTestBase, public ::testing::WithParamInterface<Sub16uSfsParam> {};

TEST_P(Sub16uC3SfsParamTest, Sub_16u_C3RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int scaleFactor = param.scaleFactor;
  const int channels = 3;
  const int total = width * height * channels;

  std::vector<Npp16u> src1Data(total);
  std::vector<Npp16u> src2Data(total);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(500), static_cast<Npp16u>(2000), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(1), static_cast<Npp16u>(500), 54321);

  std::vector<Npp16u> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    // NPP Sub: dst = src2 - src1
    expectedData[i] = expect::sub_sfs<Npp16u>(src2Data[i], src1Data[i], scaleFactor);
  }

  NppImageMemory<Npp16u> src1(width * channels, height);
  NppImageMemory<Npp16u> src2(width * channels, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiSub_16u_C3IRSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiSub_16u_C3IRSfs(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    src2.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16u> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiSub_16u_C3RSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                      scaleFactor, ctx);
    } else {
      status =
          nppiSub_16u_C3RSfs(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Sub16uC3Sfs, Sub16uC3SfsParamTest,
                         ::testing::Values(Sub16uSfsParam{32, 32, 0, false, false, "32x32_sfs0_noCtx"},
                                           Sub16uSfsParam{32, 32, 0, true, false, "32x32_sfs0_Ctx"},
                                           Sub16uSfsParam{32, 32, 0, false, true, "32x32_sfs0_InPlace"},
                                           Sub16uSfsParam{32, 32, 0, true, true, "32x32_sfs0_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<Sub16uSfsParam> &info) { return info.param.name; });

// ==================== Sub 16s C3 with scale factor TEST_P ====================

class Sub16sC3SfsParamTest : public NppTestBase, public ::testing::WithParamInterface<Sub16sSfsParam> {};

TEST_P(Sub16sC3SfsParamTest, Sub_16s_C3RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int scaleFactor = param.scaleFactor;
  const int channels = 3;
  const int total = width * height * channels;

  std::vector<Npp16s> src1Data(total);
  std::vector<Npp16s> src2Data(total);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16s>(-100), static_cast<Npp16s>(100), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16s>(-50), static_cast<Npp16s>(50), 54321);

  std::vector<Npp16s> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    // NPP Sub: dst = src2 - src1
    expectedData[i] = expect::sub_sfs<Npp16s>(src2Data[i], src1Data[i], scaleFactor);
  }

  NppImageMemory<Npp16s> src1(width * channels, height);
  NppImageMemory<Npp16s> src2(width * channels, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiSub_16s_C3IRSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiSub_16s_C3IRSfs(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16s> resultData(total);
    src2.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16s> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiSub_16s_C3RSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                      scaleFactor, ctx);
    } else {
      status =
          nppiSub_16s_C3RSfs(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16s> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Sub16sC3Sfs, Sub16sC3SfsParamTest,
                         ::testing::Values(Sub16sSfsParam{32, 32, 0, false, false, "32x32_sfs0_noCtx"},
                                           Sub16sSfsParam{32, 32, 0, true, false, "32x32_sfs0_Ctx"},
                                           Sub16sSfsParam{32, 32, 0, false, true, "32x32_sfs0_InPlace"},
                                           Sub16sSfsParam{32, 32, 0, true, true, "32x32_sfs0_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<Sub16sSfsParam> &info) { return info.param.name; });
