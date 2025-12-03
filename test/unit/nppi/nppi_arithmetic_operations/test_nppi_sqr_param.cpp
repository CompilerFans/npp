#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Sqr 8u with scale factor TEST_P ====================

struct Sqr8uSfsParam {
  int width;
  int height;
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Sqr8uSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<Sqr8uSfsParam> {};

TEST_P(Sqr8uSfsParamTest, Sqr_8u_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int scaleFactor = param.scaleFactor;

  std::vector<Npp8u> srcData(width * height);
  // Use smaller values to avoid overflow with sfs=0
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(15), 12345);

  std::vector<Npp8u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::sqr_sfs<Npp8u>(srcData[i], scaleFactor);
  }

  NppImageMemory<Npp8u> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiSqr_8u_C1IRSfs_Ctx(src.get(), src.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiSqr_8u_C1IRSfs(src.get(), src.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(width * height);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp8u> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiSqr_8u_C1RSfs_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiSqr_8u_C1RSfs(src.get(), src.step(), dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Sqr8uSfs, Sqr8uSfsParamTest,
                         ::testing::Values(Sqr8uSfsParam{32, 32, 0, false, false, "32x32_sfs0_noCtx"},
                                           Sqr8uSfsParam{32, 32, 0, true, false, "32x32_sfs0_Ctx"},
                                           Sqr8uSfsParam{32, 32, 0, false, true, "32x32_sfs0_InPlace"},
                                           Sqr8uSfsParam{32, 32, 0, true, true, "32x32_sfs0_InPlace_Ctx"},
                                           Sqr8uSfsParam{64, 64, 0, false, false, "64x64_sfs0_noCtx"}),
                         [](const ::testing::TestParamInfo<Sqr8uSfsParam> &info) { return info.param.name; });

// ==================== Sqr 16u with scale factor TEST_P ====================

struct Sqr16uSfsParam {
  int width;
  int height;
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Sqr16uSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<Sqr16uSfsParam> {};

TEST_P(Sqr16uSfsParamTest, Sqr_16u_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int scaleFactor = param.scaleFactor;

  std::vector<Npp16u> srcData(width * height);
  // Use smaller values to avoid overflow
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(255), 12345);

  std::vector<Npp16u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::sqr_sfs<Npp16u>(srcData[i], scaleFactor);
  }

  NppImageMemory<Npp16u> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiSqr_16u_C1IRSfs_Ctx(src.get(), src.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiSqr_16u_C1IRSfs(src.get(), src.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(width * height);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16u> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiSqr_16u_C1RSfs_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiSqr_16u_C1RSfs(src.get(), src.step(), dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Sqr16uSfs, Sqr16uSfsParamTest,
                         ::testing::Values(Sqr16uSfsParam{32, 32, 0, false, false, "32x32_sfs0_noCtx"},
                                           Sqr16uSfsParam{32, 32, 0, true, false, "32x32_sfs0_Ctx"},
                                           Sqr16uSfsParam{32, 32, 0, false, true, "32x32_sfs0_InPlace"},
                                           Sqr16uSfsParam{32, 32, 0, true, true, "32x32_sfs0_InPlace_Ctx"},
                                           Sqr16uSfsParam{64, 64, 0, false, false, "64x64_sfs0_noCtx"}),
                         [](const ::testing::TestParamInfo<Sqr16uSfsParam> &info) { return info.param.name; });

// ==================== Sqr 16s with scale factor TEST_P ====================

struct Sqr16sSfsParam {
  int width;
  int height;
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Sqr16sSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<Sqr16sSfsParam> {};

TEST_P(Sqr16sSfsParamTest, Sqr_16s_C1RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int scaleFactor = param.scaleFactor;

  std::vector<Npp16s> srcData(width * height);
  // Use smaller values to avoid overflow
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(-100), static_cast<Npp16s>(100), 12345);

  std::vector<Npp16s> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::sqr_sfs<Npp16s>(srcData[i], scaleFactor);
  }

  NppImageMemory<Npp16s> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiSqr_16s_C1IRSfs_Ctx(src.get(), src.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiSqr_16s_C1IRSfs(src.get(), src.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16s> resultData(width * height);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16s> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiSqr_16s_C1RSfs_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiSqr_16s_C1RSfs(src.get(), src.step(), dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16s> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Sqr16sSfs, Sqr16sSfsParamTest,
                         ::testing::Values(Sqr16sSfsParam{32, 32, 0, false, false, "32x32_sfs0_noCtx"},
                                           Sqr16sSfsParam{32, 32, 0, true, false, "32x32_sfs0_Ctx"},
                                           Sqr16sSfsParam{32, 32, 0, false, true, "32x32_sfs0_InPlace"},
                                           Sqr16sSfsParam{32, 32, 0, true, true, "32x32_sfs0_InPlace_Ctx"},
                                           Sqr16sSfsParam{64, 64, 0, false, false, "64x64_sfs0_noCtx"}),
                         [](const ::testing::TestParamInfo<Sqr16sSfsParam> &info) { return info.param.name; });

// ==================== Sqr 32f TEST_P ====================

struct Sqr32fParam {
  int width;
  int height;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Sqr32fParamTest : public NppTestBase, public ::testing::WithParamInterface<Sqr32fParam> {};

TEST_P(Sqr32fParamTest, Sqr_32f_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, -10.0f, 10.0f, 12345);

  std::vector<Npp32f> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::sqr<Npp32f>(srcData[i]);
  }

  NppImageMemory<Npp32f> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiSqr_32f_C1IR_Ctx(src.get(), src.step(), roi, ctx);
    } else {
      status = nppiSqr_32f_C1IR(src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> resultData(width * height);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
  } else {
    NppImageMemory<Npp32f> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiSqr_32f_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiSqr_32f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
  }
}

INSTANTIATE_TEST_SUITE_P(Sqr32f, Sqr32fParamTest,
                         ::testing::Values(Sqr32fParam{32, 32, false, false, "32x32_noCtx"},
                                           Sqr32fParam{32, 32, true, false, "32x32_Ctx"},
                                           Sqr32fParam{32, 32, false, true, "32x32_InPlace"},
                                           Sqr32fParam{32, 32, true, true, "32x32_InPlace_Ctx"},
                                           Sqr32fParam{64, 64, false, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<Sqr32fParam> &info) { return info.param.name; });

// ==================== Sqr 32f C3 TEST_P ====================

class Sqr32fC3ParamTest : public NppTestBase, public ::testing::WithParamInterface<Sqr32fParam> {};

TEST_P(Sqr32fC3ParamTest, Sqr_32f_C3R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 3;
  const int total = width * height * channels;

  std::vector<Npp32f> srcData(total);
  TestDataGenerator::generateRandom(srcData, -10.0f, 10.0f, 12345);

  std::vector<Npp32f> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::sqr<Npp32f>(srcData[i]);
  }

  NppImageMemory<Npp32f> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiSqr_32f_C3IR_Ctx(src.get(), src.step(), roi, ctx);
    } else {
      status = nppiSqr_32f_C3IR(src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
  } else {
    NppImageMemory<Npp32f> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiSqr_32f_C3R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiSqr_32f_C3R(src.get(), src.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
  }
}

INSTANTIATE_TEST_SUITE_P(Sqr32fC3, Sqr32fC3ParamTest,
                         ::testing::Values(Sqr32fParam{32, 32, false, false, "32x32_noCtx"},
                                           Sqr32fParam{32, 32, true, false, "32x32_Ctx"},
                                           Sqr32fParam{32, 32, false, true, "32x32_InPlace"},
                                           Sqr32fParam{32, 32, true, true, "32x32_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<Sqr32fParam> &info) { return info.param.name; });

// ==================== Sqr 32f C4 TEST_P ====================

class Sqr32fC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<Sqr32fParam> {};

TEST_P(Sqr32fC4ParamTest, Sqr_32f_C4R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int channels = 4;
  const int total = width * height * channels;

  std::vector<Npp32f> srcData(total);
  TestDataGenerator::generateRandom(srcData, -10.0f, 10.0f, 12345);

  std::vector<Npp32f> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::sqr<Npp32f>(srcData[i]);
  }

  NppImageMemory<Npp32f> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiSqr_32f_C4IR_Ctx(src.get(), src.step(), roi, ctx);
    } else {
      status = nppiSqr_32f_C4IR(src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
  } else {
    NppImageMemory<Npp32f> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiSqr_32f_C4R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiSqr_32f_C4R(src.get(), src.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
  }
}

INSTANTIATE_TEST_SUITE_P(Sqr32fC4, Sqr32fC4ParamTest,
                         ::testing::Values(Sqr32fParam{32, 32, false, false, "32x32_noCtx"},
                                           Sqr32fParam{32, 32, true, false, "32x32_Ctx"},
                                           Sqr32fParam{32, 32, false, true, "32x32_InPlace"},
                                           Sqr32fParam{32, 32, true, true, "32x32_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<Sqr32fParam> &info) { return info.param.name; });

// ==================== Sqr 8u C3 with scale factor TEST_P ====================

class Sqr8uC3SfsParamTest : public NppTestBase, public ::testing::WithParamInterface<Sqr8uSfsParam> {};

TEST_P(Sqr8uC3SfsParamTest, Sqr_8u_C3RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int scaleFactor = param.scaleFactor;
  const int channels = 3;
  const int total = width * height * channels;

  std::vector<Npp8u> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(15), 12345);

  std::vector<Npp8u> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::sqr_sfs<Npp8u>(srcData[i], scaleFactor);
  }

  NppImageMemory<Npp8u> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiSqr_8u_C3IRSfs_Ctx(src.get(), src.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiSqr_8u_C3IRSfs(src.get(), src.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp8u> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiSqr_8u_C3RSfs_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiSqr_8u_C3RSfs(src.get(), src.step(), dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Sqr8uC3Sfs, Sqr8uC3SfsParamTest,
                         ::testing::Values(Sqr8uSfsParam{32, 32, 0, false, false, "32x32_sfs0_noCtx"},
                                           Sqr8uSfsParam{32, 32, 0, true, false, "32x32_sfs0_Ctx"},
                                           Sqr8uSfsParam{32, 32, 0, false, true, "32x32_sfs0_InPlace"},
                                           Sqr8uSfsParam{32, 32, 0, true, true, "32x32_sfs0_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<Sqr8uSfsParam> &info) { return info.param.name; });

// ==================== Sqr 8u C4 with scale factor TEST_P ====================

class Sqr8uC4SfsParamTest : public NppTestBase, public ::testing::WithParamInterface<Sqr8uSfsParam> {};

TEST_P(Sqr8uC4SfsParamTest, Sqr_8u_C4RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int scaleFactor = param.scaleFactor;
  const int channels = 4;
  const int total = width * height * channels;

  std::vector<Npp8u> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(15), 12345);

  std::vector<Npp8u> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::sqr_sfs<Npp8u>(srcData[i], scaleFactor);
  }

  NppImageMemory<Npp8u> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiSqr_8u_C4IRSfs_Ctx(src.get(), src.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiSqr_8u_C4IRSfs(src.get(), src.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp8u> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiSqr_8u_C4RSfs_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiSqr_8u_C4RSfs(src.get(), src.step(), dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Sqr8uC4Sfs, Sqr8uC4SfsParamTest,
                         ::testing::Values(Sqr8uSfsParam{32, 32, 0, false, false, "32x32_sfs0_noCtx"},
                                           Sqr8uSfsParam{32, 32, 0, true, false, "32x32_sfs0_Ctx"},
                                           Sqr8uSfsParam{32, 32, 0, false, true, "32x32_sfs0_InPlace"},
                                           Sqr8uSfsParam{32, 32, 0, true, true, "32x32_sfs0_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<Sqr8uSfsParam> &info) { return info.param.name; });

// ==================== Sqr 16u C3 with scale factor TEST_P ====================

class Sqr16uC3SfsParamTest : public NppTestBase, public ::testing::WithParamInterface<Sqr16uSfsParam> {};

TEST_P(Sqr16uC3SfsParamTest, Sqr_16u_C3RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int scaleFactor = param.scaleFactor;
  const int channels = 3;
  const int total = width * height * channels;

  std::vector<Npp16u> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(255), 12345);

  std::vector<Npp16u> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::sqr_sfs<Npp16u>(srcData[i], scaleFactor);
  }

  NppImageMemory<Npp16u> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiSqr_16u_C3IRSfs_Ctx(src.get(), src.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiSqr_16u_C3IRSfs(src.get(), src.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16u> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiSqr_16u_C3RSfs_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiSqr_16u_C3RSfs(src.get(), src.step(), dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Sqr16uC3Sfs, Sqr16uC3SfsParamTest,
                         ::testing::Values(Sqr16uSfsParam{32, 32, 0, false, false, "32x32_sfs0_noCtx"},
                                           Sqr16uSfsParam{32, 32, 0, true, false, "32x32_sfs0_Ctx"},
                                           Sqr16uSfsParam{32, 32, 0, false, true, "32x32_sfs0_InPlace"},
                                           Sqr16uSfsParam{32, 32, 0, true, true, "32x32_sfs0_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<Sqr16uSfsParam> &info) { return info.param.name; });

// ==================== Sqr 16u C4 with scale factor TEST_P ====================

class Sqr16uC4SfsParamTest : public NppTestBase, public ::testing::WithParamInterface<Sqr16uSfsParam> {};

TEST_P(Sqr16uC4SfsParamTest, Sqr_16u_C4RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int scaleFactor = param.scaleFactor;
  const int channels = 4;
  const int total = width * height * channels;

  std::vector<Npp16u> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(255), 12345);

  std::vector<Npp16u> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::sqr_sfs<Npp16u>(srcData[i], scaleFactor);
  }

  NppImageMemory<Npp16u> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiSqr_16u_C4IRSfs_Ctx(src.get(), src.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiSqr_16u_C4IRSfs(src.get(), src.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16u> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiSqr_16u_C4RSfs_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiSqr_16u_C4RSfs(src.get(), src.step(), dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Sqr16uC4Sfs, Sqr16uC4SfsParamTest,
                         ::testing::Values(Sqr16uSfsParam{32, 32, 0, false, false, "32x32_sfs0_noCtx"},
                                           Sqr16uSfsParam{32, 32, 0, true, false, "32x32_sfs0_Ctx"},
                                           Sqr16uSfsParam{32, 32, 0, false, true, "32x32_sfs0_InPlace"},
                                           Sqr16uSfsParam{32, 32, 0, true, true, "32x32_sfs0_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<Sqr16uSfsParam> &info) { return info.param.name; });

// ==================== Sqr 16s C3 with scale factor TEST_P ====================

class Sqr16sC3SfsParamTest : public NppTestBase, public ::testing::WithParamInterface<Sqr16sSfsParam> {};

TEST_P(Sqr16sC3SfsParamTest, Sqr_16s_C3RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int scaleFactor = param.scaleFactor;
  const int channels = 3;
  const int total = width * height * channels;

  std::vector<Npp16s> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(-100), static_cast<Npp16s>(100), 12345);

  std::vector<Npp16s> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::sqr_sfs<Npp16s>(srcData[i], scaleFactor);
  }

  NppImageMemory<Npp16s> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiSqr_16s_C3IRSfs_Ctx(src.get(), src.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiSqr_16s_C3IRSfs(src.get(), src.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16s> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16s> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiSqr_16s_C3RSfs_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiSqr_16s_C3RSfs(src.get(), src.step(), dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16s> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Sqr16sC3Sfs, Sqr16sC3SfsParamTest,
                         ::testing::Values(Sqr16sSfsParam{32, 32, 0, false, false, "32x32_sfs0_noCtx"},
                                           Sqr16sSfsParam{32, 32, 0, true, false, "32x32_sfs0_Ctx"},
                                           Sqr16sSfsParam{32, 32, 0, false, true, "32x32_sfs0_InPlace"},
                                           Sqr16sSfsParam{32, 32, 0, true, true, "32x32_sfs0_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<Sqr16sSfsParam> &info) { return info.param.name; });

// ==================== Sqr 16s C4 with scale factor TEST_P ====================

class Sqr16sC4SfsParamTest : public NppTestBase, public ::testing::WithParamInterface<Sqr16sSfsParam> {};

TEST_P(Sqr16sC4SfsParamTest, Sqr_16s_C4RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const int scaleFactor = param.scaleFactor;
  const int channels = 4;
  const int total = width * height * channels;

  std::vector<Npp16s> srcData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(-100), static_cast<Npp16s>(100), 12345);

  std::vector<Npp16s> expectedData(total);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::sqr_sfs<Npp16s>(srcData[i], scaleFactor);
  }

  NppImageMemory<Npp16s> src(width * channels, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiSqr_16s_C4IRSfs_Ctx(src.get(), src.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiSqr_16s_C4IRSfs(src.get(), src.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16s> resultData(total);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  } else {
    NppImageMemory<Npp16s> dst(width * channels, height);
    if (param.use_ctx) {
      NppStreamContext ctx{};
      ctx.hStream = 0;
      status = nppiSqr_16s_C4RSfs_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiSqr_16s_C4RSfs(src.get(), src.step(), dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16s> resultData(total);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData));
  }
}

INSTANTIATE_TEST_SUITE_P(Sqr16sC4Sfs, Sqr16sC4SfsParamTest,
                         ::testing::Values(Sqr16sSfsParam{32, 32, 0, false, false, "32x32_sfs0_noCtx"},
                                           Sqr16sSfsParam{32, 32, 0, true, false, "32x32_sfs0_Ctx"},
                                           Sqr16sSfsParam{32, 32, 0, false, true, "32x32_sfs0_InPlace"},
                                           Sqr16sSfsParam{32, 32, 0, true, true, "32x32_sfs0_InPlace_Ctx"}),
                         [](const ::testing::TestParamInfo<Sqr16sSfsParam> &info) { return info.param.name; });
