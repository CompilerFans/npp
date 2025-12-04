#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Add 32f TEST_P ====================

struct Add32fParam {
  int width;
  int height;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Add32fParamTest : public NppTestBase, public ::testing::WithParamInterface<Add32fParam> {};

TEST_P(Add32fParamTest, Add_32f_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> src1Data(width * height);
  std::vector<Npp32f> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, -50.0f, 50.0f, 12345);
  TestDataGenerator::generateRandom(src2Data, -50.0f, 50.0f, 54321);

  std::vector<Npp32f> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::add<Npp32f>(src1Data[i], src2Data[i]);
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
      status = nppiAdd_32f_C1IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx);
    } else {
      status = nppiAdd_32f_C1IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
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
      status = nppiAdd_32f_C1R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiAdd_32f_C1R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
  }
}

INSTANTIATE_TEST_SUITE_P(Add32f, Add32fParamTest,
                         ::testing::Values(Add32fParam{32, 32, false, false, "32x32_noCtx"},
                                           Add32fParam{32, 32, true, false, "32x32_Ctx"},
                                           Add32fParam{32, 32, false, true, "32x32_InPlace"},
                                           Add32fParam{32, 32, true, true, "32x32_InPlace_Ctx"},
                                           Add32fParam{64, 64, false, false, "64x64_noCtx"},
                                           Add32fParam{64, 64, true, false, "64x64_Ctx"}),
                         [](const ::testing::TestParamInfo<Add32fParam> &info) { return info.param.name; });

// ==================== Add 8u with scale factor TEST_P ====================

struct Add8uSfsParam {
  int width;
  int height;
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Add8uSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<Add8uSfsParam> {};

TEST_P(Add8uSfsParamTest, Add_8u_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int scaleFactor = param.scaleFactor;

  std::vector<Npp8u> src1Data(width * height);
  std::vector<Npp8u> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(1), static_cast<Npp8u>(100), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8u>(1), static_cast<Npp8u>(100), 54321);

  std::vector<Npp8u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::add_sfs<Npp8u>(src1Data[i], src2Data[i], scaleFactor);
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
      status = nppiAdd_8u_C1IRSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiAdd_8u_C1IRSfs(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor);
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
      status = nppiAdd_8u_C1RSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                     scaleFactor, ctx);
    } else {
      status =
          nppiAdd_8u_C1RSfs(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Add8uSfs, Add8uSfsParamTest,
                         ::testing::Values(Add8uSfsParam{32, 32, 0, false, false, "32x32_sfs0_noCtx"},
                                           Add8uSfsParam{32, 32, 0, true, false, "32x32_sfs0_Ctx"},
                                           Add8uSfsParam{32, 32, 0, false, true, "32x32_sfs0_InPlace"},
                                           Add8uSfsParam{32, 32, 0, true, true, "32x32_sfs0_InPlace_Ctx"},
                                           Add8uSfsParam{64, 64, 0, false, false, "64x64_sfs0_noCtx"}),
                         [](const ::testing::TestParamInfo<Add8uSfsParam> &info) { return info.param.name; });

// ==================== Add 16u with scale factor TEST_P ====================

struct Add16uSfsParam {
  int width;
  int height;
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Add16uSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<Add16uSfsParam> {};

TEST_P(Add16uSfsParamTest, Add_16u_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int scaleFactor = param.scaleFactor;

  std::vector<Npp16u> src1Data(width * height);
  std::vector<Npp16u> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(1), static_cast<Npp16u>(1000), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(1), static_cast<Npp16u>(1000), 54321);

  std::vector<Npp16u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::add_sfs<Npp16u>(src1Data[i], src2Data[i], scaleFactor);
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
      status = nppiAdd_16u_C1IRSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiAdd_16u_C1IRSfs(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor);
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
      status = nppiAdd_16u_C1RSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                      scaleFactor, ctx);
    } else {
      status =
          nppiAdd_16u_C1RSfs(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Add16uSfs, Add16uSfsParamTest,
                         ::testing::Values(Add16uSfsParam{32, 32, 0, false, false, "32x32_sfs0_noCtx"},
                                           Add16uSfsParam{32, 32, 0, true, false, "32x32_sfs0_Ctx"},
                                           Add16uSfsParam{32, 32, 0, false, true, "32x32_sfs0_InPlace"},
                                           Add16uSfsParam{32, 32, 0, true, true, "32x32_sfs0_InPlace_Ctx"},
                                           Add16uSfsParam{64, 64, 0, false, false, "64x64_sfs0_noCtx"}),
                         [](const ::testing::TestParamInfo<Add16uSfsParam> &info) { return info.param.name; });

// ==================== Add 16s with scale factor TEST_P ====================

struct Add16sSfsParam {
  int width;
  int height;
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Add16sSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<Add16sSfsParam> {};

TEST_P(Add16sSfsParamTest, Add_16s_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int scaleFactor = param.scaleFactor;

  std::vector<Npp16s> src1Data(width * height);
  std::vector<Npp16s> src2Data(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16s>(-100), static_cast<Npp16s>(100), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16s>(-100), static_cast<Npp16s>(100), 54321);

  std::vector<Npp16s> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::add_sfs<Npp16s>(src1Data[i], src2Data[i], scaleFactor);
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
      status = nppiAdd_16s_C1IRSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiAdd_16s_C1IRSfs(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor);
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
      status = nppiAdd_16s_C1RSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                      scaleFactor, ctx);
    } else {
      status =
          nppiAdd_16s_C1RSfs(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16s> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Add16sSfs, Add16sSfsParamTest,
                         ::testing::Values(Add16sSfsParam{32, 32, 0, false, false, "32x32_sfs0_noCtx"},
                                           Add16sSfsParam{32, 32, 0, true, false, "32x32_sfs0_Ctx"},
                                           Add16sSfsParam{32, 32, 0, false, true, "32x32_sfs0_InPlace"},
                                           Add16sSfsParam{32, 32, 0, true, true, "32x32_sfs0_InPlace_Ctx"},
                                           Add16sSfsParam{64, 64, 0, false, false, "64x64_sfs0_noCtx"}),
                         [](const ::testing::TestParamInfo<Add16sSfsParam> &info) { return info.param.name; });

// ==================== Add 32f C3 TEST_P ====================

struct Add32fC3Param {
  int width;
  int height;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Add32fC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<Add32fC3Param> {};

TEST_P(Add32fC3ParamTest, Add_32f_C3R) {
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
    expectedData[i] = expect::add<Npp32f>(src1Data[i], src2Data[i]);
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
      status = nppiAdd_32f_C3IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx);
    } else {
      status = nppiAdd_32f_C3IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
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
      status = nppiAdd_32f_C3R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiAdd_32f_C3R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> resultData(width * height * channels);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
  }
}

INSTANTIATE_TEST_SUITE_P(Add32fC3, Add32fC3ParamTest,
                         ::testing::Values(Add32fC3Param{32, 32, false, false, "32x32_noCtx"},
                                           Add32fC3Param{32, 32, true, false, "32x32_Ctx"},
                                           Add32fC3Param{32, 32, false, true, "32x32_InPlace"},
                                           Add32fC3Param{32, 32, true, true, "32x32_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<Add32fC3Param> &info) { return info.param.name; });

// ==================== Add 32f C4 TEST_P ====================

struct Add32fC4Param {
  int width;
  int height;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Add32fC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<Add32fC4Param> {};

TEST_P(Add32fC4ParamTest, Add_32f_C4R) {
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
    expectedData[i] = expect::add<Npp32f>(src1Data[i], src2Data[i]);
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
      status = nppiAdd_32f_C4IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx);
    } else {
      status = nppiAdd_32f_C4IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
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
      status = nppiAdd_32f_C4R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiAdd_32f_C4R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> resultData(width * height * channels);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
  }
}

INSTANTIATE_TEST_SUITE_P(Add32fC4, Add32fC4ParamTest,
                         ::testing::Values(Add32fC4Param{32, 32, false, false, "32x32_noCtx"},
                                           Add32fC4Param{32, 32, true, false, "32x32_Ctx"},
                                           Add32fC4Param{32, 32, false, true, "32x32_InPlace"},
                                           Add32fC4Param{32, 32, true, true, "32x32_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<Add32fC4Param> &info) { return info.param.name; });

// ==================== Add 16u C3 with scale factor TEST_P ====================

class Add16uC3SfsParamTest : public NppTestBase, public ::testing::WithParamInterface<Add16uSfsParam> {};

TEST_P(Add16uC3SfsParamTest, Add_16u_C3RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int scaleFactor = param.scaleFactor;
  const int channels = 3;
  const int total = width * height * channels;

  std::vector<Npp16u> src1Data(total);
  std::vector<Npp16u> src2Data(total);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(1), static_cast<Npp16u>(1000), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(1), static_cast<Npp16u>(1000), 54321);

  std::vector<Npp16u> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::add_sfs<Npp16u>(src1Data[i], src2Data[i], scaleFactor);
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
      status = nppiAdd_16u_C3IRSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiAdd_16u_C3IRSfs(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor);
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
      status = nppiAdd_16u_C3RSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                      scaleFactor, ctx);
    } else {
      status =
          nppiAdd_16u_C3RSfs(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Add16uC3Sfs, Add16uC3SfsParamTest,
                         ::testing::Values(Add16uSfsParam{32, 32, 0, false, false, "32x32_sfs0_noCtx"},
                                           Add16uSfsParam{32, 32, 0, true, false, "32x32_sfs0_Ctx"},
                                           Add16uSfsParam{32, 32, 0, false, true, "32x32_sfs0_InPlace"},
                                           Add16uSfsParam{32, 32, 0, true, true, "32x32_sfs0_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<Add16uSfsParam> &info) { return info.param.name; });

// ==================== Add 16s C3 with scale factor TEST_P ====================

class Add16sC3SfsParamTest : public NppTestBase, public ::testing::WithParamInterface<Add16sSfsParam> {};

TEST_P(Add16sC3SfsParamTest, Add_16s_C3RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int scaleFactor = param.scaleFactor;
  const int channels = 3;
  const int total = width * height * channels;

  std::vector<Npp16s> src1Data(total);
  std::vector<Npp16s> src2Data(total);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16s>(-100), static_cast<Npp16s>(100), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16s>(-100), static_cast<Npp16s>(100), 54321);

  std::vector<Npp16s> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::add_sfs<Npp16s>(src1Data[i], src2Data[i], scaleFactor);
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
      status = nppiAdd_16s_C3IRSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiAdd_16s_C3IRSfs(src1.get(), src1.step(), src2.get(), src2.step(), roi, scaleFactor);
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
      status = nppiAdd_16s_C3RSfs_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi,
                                      scaleFactor, ctx);
    } else {
      status =
          nppiAdd_16s_C3RSfs(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16s> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Add16sC3Sfs, Add16sC3SfsParamTest,
                         ::testing::Values(Add16sSfsParam{32, 32, 0, false, false, "32x32_sfs0_noCtx"},
                                           Add16sSfsParam{32, 32, 0, true, false, "32x32_sfs0_Ctx"},
                                           Add16sSfsParam{32, 32, 0, false, true, "32x32_sfs0_InPlace"},
                                           Add16sSfsParam{32, 32, 0, true, true, "32x32_sfs0_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<Add16sSfsParam> &info) { return info.param.name; });
