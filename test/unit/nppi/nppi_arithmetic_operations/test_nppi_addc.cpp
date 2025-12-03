#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

class AddcFunctionalTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }
};

TEST_F(AddcFunctionalTest, AddC_8u_C1RSfs_BasicOperation) {
  const int width = 32;
  const int height = 32;
  const int scaleFactor = 0;
  const Npp8u nConstant = 50;

  std::vector<Npp8u> srcData(width * height);
  std::vector<Npp8u> expectedData(width * height);

  TestDataGenerator::generateConstant(srcData, static_cast<Npp8u>(30));
  TestDataGenerator::generateConstant(expectedData, static_cast<Npp8u>(80));

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiAddC_8u_C1RSfs(src.get(), src.step(), nConstant, dst.get(), dst.step(), roi, scaleFactor);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAddC_8u_C1RSfs failed";

  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData)) << "AddC operation produced incorrect results";
}

TEST_F(AddcFunctionalTest, AddC_32f_C1R_BasicOperation) {
  const int width = 32;
  const int height = 32;
  const Npp32f nConstant = 7.5f;

  std::vector<Npp32f> srcData(width * height);
  std::vector<Npp32f> expectedData(width * height);

  TestDataGenerator::generateConstant(srcData, 2.5f);
  TestDataGenerator::generateConstant(expectedData, 10.0f);

  NppImageMemory<Npp32f> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiAddC_32f_C1R(src.get(), src.step(), nConstant, dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAddC_32f_C1R failed";

  std::vector<Npp32f> resultData(width * height);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f))
      << "AddC 32f operation produced incorrect results";
}

// ==================== Tests using framework ====================

// Helper class for AddC with scale factor tests
class AddCSfsTest : public ConstOpParamTest<Npp8u, Npp8u> {};

TEST_F(AddCSfsTest, AddC_8u_C1RSfs_Framework) {
  const Npp8u constant = 30;
  const int scaleFactor = 0;
  auto expectFunc = [constant, scaleFactor](Npp8u x, Npp8u) { return expect::add_c_sfs<Npp8u>(x, constant, scaleFactor); };

  ConstOpTestConfig config = {32, 32, 1, false, "AddC_8u_C1RSfs_Framework"};
  auto nppFunc = [scaleFactor](const Npp8u *src, int srcStep, Npp8u *dst, int dstStep, NppiSize roi, Npp8u c) {
    return nppiAddC_8u_C1RSfs(src, srcStep, c, dst, dstStep, roi, scaleFactor);
  };
  runConstTest(config, constant, expectFunc, nppFunc, nullptr);
}

TEST_F(AddCSfsTest, AddC_8u_C1RSfs_Ctx_Framework) {
  const Npp8u constant = 30;
  const int scaleFactor = 0;
  auto expectFunc = [constant, scaleFactor](Npp8u x, Npp8u) { return expect::add_c_sfs<Npp8u>(x, constant, scaleFactor); };

  ConstOpTestConfig config = {32, 32, 1, true, "AddC_8u_C1RSfs_Ctx_Framework"};
  auto nppFunc = [scaleFactor](const Npp8u *src, int srcStep, Npp8u *dst, int dstStep, NppiSize roi, Npp8u c) {
    return nppiAddC_8u_C1RSfs(src, srcStep, c, dst, dstStep, roi, scaleFactor);
  };
  auto nppFuncCtx = [scaleFactor](const Npp8u *src, int srcStep, Npp8u *dst, int dstStep, NppiSize roi, Npp8u c,
                                  NppStreamContext ctx) {
    return nppiAddC_8u_C1RSfs_Ctx(src, srcStep, c, dst, dstStep, roi, scaleFactor, ctx);
  };
  runConstTest(config, constant, expectFunc, nppFunc, nppFuncCtx);
}

// 32f AddC tests using framework
class AddC32fTest : public ConstOpParamTest<Npp32f, Npp32f> {};

TEST_F(AddC32fTest, AddC_32f_C1R_Framework) {
  const Npp32f constant = 10.5f;
  auto expectFunc = [](Npp32f x, Npp32f c) { return expect::add_c<Npp32f>(x, c); };

  ConstOpTestConfig config = {32, 32, 1, false, "AddC_32f_C1R_Framework"};
  auto nppFunc = [](const Npp32f *src, int srcStep, Npp32f *dst, int dstStep, NppiSize roi, Npp32f c) {
    return nppiAddC_32f_C1R(src, srcStep, c, dst, dstStep, roi);
  };
  runConstTest(config, constant, expectFunc, nppFunc, nullptr);
}

TEST_F(AddC32fTest, AddC_32f_C1R_Ctx_Framework) {
  const Npp32f constant = 10.5f;
  auto expectFunc = [](Npp32f x, Npp32f c) { return expect::add_c<Npp32f>(x, c); };

  ConstOpTestConfig config = {32, 32, 1, true, "AddC_32f_C1R_Ctx_Framework"};
  auto nppFunc = [](const Npp32f *src, int srcStep, Npp32f *dst, int dstStep, NppiSize roi, Npp32f c) {
    return nppiAddC_32f_C1R(src, srcStep, c, dst, dstStep, roi);
  };
  auto nppFuncCtx = [](const Npp32f *src, int srcStep, Npp32f *dst, int dstStep, NppiSize roi, Npp32f c,
                       NppStreamContext ctx) { return nppiAddC_32f_C1R_Ctx(src, srcStep, c, dst, dstStep, roi, ctx); };
  runConstTest(config, constant, expectFunc, nppFunc, nppFuncCtx);
}

TEST_F(AddC32fTest, AddC_32f_C1IR_Framework) {
  const int width = 32;
  const int height = 32;
  const Npp32f constant = 15.0f;

  std::vector<Npp32f> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, -50.0f, 50.0f, 12345);

  std::vector<Npp32f> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::add_c<Npp32f>(srcData[i], constant);
  }

  NppImageMemory<Npp32f> srcDst(width, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiAddC_32f_C1IR(constant, srcDst.get(), srcDst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAddC_32f_C1IR failed";

  std::vector<Npp32f> resultData(width * height);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f))
      << "In-place AddC 32f operation produced incorrect results";
}

TEST_F(AddC32fTest, AddC_32f_C1IR_Ctx_Framework) {
  const int width = 32;
  const int height = 32;
  const Npp32f constant = 15.0f;

  std::vector<Npp32f> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, -50.0f, 50.0f, 12345);

  std::vector<Npp32f> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::add_c<Npp32f>(srcData[i], constant);
  }

  NppImageMemory<Npp32f> srcDst(width, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  NppStatus status = nppiAddC_32f_C1IR_Ctx(constant, srcDst.get(), srcDst.step(), roi, nppStreamCtx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAddC_32f_C1IR_Ctx failed";

  std::vector<Npp32f> resultData(width * height);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f))
      << "In-place AddC 32f Ctx operation produced incorrect results";
}

// 16u AddC with scale factor tests
class AddC16uSfsTest : public NppTestBase {};

TEST_F(AddC16uSfsTest, AddC_16u_C1RSfs_Framework) {
  const int width = 32;
  const int height = 32;
  const Npp16u constant = 1000;
  const int scaleFactor = 0;

  std::vector<Npp16u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(30000), 12345);

  std::vector<Npp16u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::add_c_sfs<Npp16u>(srcData[i], constant, scaleFactor);
  }

  NppImageMemory<Npp16u> src(width, height);
  NppImageMemory<Npp16u> dst(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiAddC_16u_C1RSfs(src.get(), src.step(), constant, dst.get(), dst.step(), roi, scaleFactor);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAddC_16u_C1RSfs failed";

  std::vector<Npp16u> resultData(width * height);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData))
      << "AddC 16u operation produced incorrect results";
}

TEST_F(AddC16uSfsTest, AddC_16u_C1RSfs_Ctx_Framework) {
  const int width = 32;
  const int height = 32;
  const Npp16u constant = 1000;
  const int scaleFactor = 0;

  std::vector<Npp16u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(30000), 12345);

  std::vector<Npp16u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::add_c_sfs<Npp16u>(srcData[i], constant, scaleFactor);
  }

  NppImageMemory<Npp16u> src(width, height);
  NppImageMemory<Npp16u> dst(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  NppStatus status =
      nppiAddC_16u_C1RSfs_Ctx(src.get(), src.step(), constant, dst.get(), dst.step(), roi, scaleFactor, nppStreamCtx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAddC_16u_C1RSfs_Ctx failed";

  std::vector<Npp16u> resultData(width * height);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData))
      << "AddC 16u Ctx operation produced incorrect results";
}

TEST_F(AddC16uSfsTest, AddC_16u_C1IRSfs_Framework) {
  const int width = 32;
  const int height = 32;
  const Npp16u constant = 500;
  const int scaleFactor = 0;

  std::vector<Npp16u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(30000), 12345);

  std::vector<Npp16u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::add_c_sfs<Npp16u>(srcData[i], constant, scaleFactor);
  }

  NppImageMemory<Npp16u> srcDst(width, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiAddC_16u_C1IRSfs(constant, srcDst.get(), srcDst.step(), roi, scaleFactor);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAddC_16u_C1IRSfs failed";

  std::vector<Npp16u> resultData(width * height);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData))
      << "In-place AddC 16u operation produced incorrect results";
}

TEST_F(AddC16uSfsTest, AddC_16u_C1IRSfs_Ctx_Framework) {
  const int width = 32;
  const int height = 32;
  const Npp16u constant = 500;
  const int scaleFactor = 0;

  std::vector<Npp16u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(30000), 12345);

  std::vector<Npp16u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::add_c_sfs<Npp16u>(srcData[i], constant, scaleFactor);
  }

  NppImageMemory<Npp16u> srcDst(width, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  NppStatus status = nppiAddC_16u_C1IRSfs_Ctx(constant, srcDst.get(), srcDst.step(), roi, scaleFactor, nppStreamCtx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAddC_16u_C1IRSfs_Ctx failed";

  std::vector<Npp16u> resultData(width * height);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData))
      << "In-place AddC 16u Ctx operation produced incorrect results";
}

// 8u In-place AddC tests
class AddC8uSfsTest : public NppTestBase {};

TEST_F(AddC8uSfsTest, AddC_8u_C1IRSfs_Framework) {
  const int width = 32;
  const int height = 32;
  const Npp8u constant = 20;
  const int scaleFactor = 0;

  std::vector<Npp8u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(200), 12345);

  std::vector<Npp8u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::add_c_sfs<Npp8u>(srcData[i], constant, scaleFactor);
  }

  NppImageMemory<Npp8u> srcDst(width, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiAddC_8u_C1IRSfs(constant, srcDst.get(), srcDst.step(), roi, scaleFactor);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAddC_8u_C1IRSfs failed";

  std::vector<Npp8u> resultData(width * height);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData))
      << "In-place AddC 8u operation produced incorrect results";
}

TEST_F(AddC8uSfsTest, AddC_8u_C1IRSfs_Ctx_Framework) {
  const int width = 32;
  const int height = 32;
  const Npp8u constant = 20;
  const int scaleFactor = 0;

  std::vector<Npp8u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(200), 12345);

  std::vector<Npp8u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::add_c_sfs<Npp8u>(srcData[i], constant, scaleFactor);
  }

  NppImageMemory<Npp8u> srcDst(width, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  NppStatus status = nppiAddC_8u_C1IRSfs_Ctx(constant, srcDst.get(), srcDst.step(), roi, scaleFactor, nppStreamCtx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAddC_8u_C1IRSfs_Ctx failed";

  std::vector<Npp8u> resultData(width * height);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData))
      << "In-place AddC 8u Ctx operation produced incorrect results";
}

// ==================== TEST_P parameterized tests ====================

// Parameter structure for AddC 32f tests
struct AddC32fParam {
  int width;
  int height;
  Npp32f constant;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class AddC32fParamTest : public NppTestBase, public ::testing::WithParamInterface<AddC32fParam> {};

TEST_P(AddC32fParamTest, AddC_32f_Parameterized) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp32f constant = param.constant;

  std::vector<Npp32f> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, -50.0f, 50.0f, 12345);

  std::vector<Npp32f> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::add_c<Npp32f>(srcData[i], constant);
  }

  NppImageMemory<Npp32f> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiAddC_32f_C1IR_Ctx(constant, src.get(), src.step(), roi, ctx);
    } else {
      status = nppiAddC_32f_C1IR(constant, src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR) << "NPP AddC in-place failed";

    std::vector<Npp32f> resultData(width * height);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f)) << "Result mismatch";
  } else {
    NppImageMemory<Npp32f> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiAddC_32f_C1R_Ctx(src.get(), src.step(), constant, dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiAddC_32f_C1R(src.get(), src.step(), constant, dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR) << "NPP AddC failed";

    std::vector<Npp32f> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f)) << "Result mismatch";
  }
}

INSTANTIATE_TEST_SUITE_P(AddC32f, AddC32fParamTest,
                         ::testing::Values(AddC32fParam{32, 32, 10.5f, false, false, "32x32_noCtx_noDst"},
                                           AddC32fParam{32, 32, 10.5f, true, false, "32x32_Ctx_noDst"},
                                           AddC32fParam{32, 32, 15.0f, false, true, "32x32_noCtx_InPlace"},
                                           AddC32fParam{32, 32, 15.0f, true, true, "32x32_Ctx_InPlace"},
                                           AddC32fParam{64, 64, 20.0f, false, false, "64x64_noCtx"},
                                           AddC32fParam{64, 64, 20.0f, true, false, "64x64_Ctx"}),
                         [](const ::testing::TestParamInfo<AddC32fParam> &info) { return info.param.name; });

// Parameter structure for AddC 8u with scale factor tests
struct AddC8uSfsParam {
  int width;
  int height;
  Npp8u constant;
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class AddC8uSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<AddC8uSfsParam> {};

TEST_P(AddC8uSfsParamTest, AddC_8u_Sfs_Parameterized) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp8u constant = param.constant;
  const int scaleFactor = param.scaleFactor;

  std::vector<Npp8u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(200), 12345);

  std::vector<Npp8u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::add_c_sfs<Npp8u>(srcData[i], constant, scaleFactor);
  }

  NppImageMemory<Npp8u> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiAddC_8u_C1IRSfs_Ctx(constant, src.get(), src.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiAddC_8u_C1IRSfs(constant, src.get(), src.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR) << "NPP AddC in-place failed";

    std::vector<Npp8u> resultData(width * height);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData)) << "Result mismatch";
  } else {
    NppImageMemory<Npp8u> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiAddC_8u_C1RSfs_Ctx(src.get(), src.step(), constant, dst.get(), dst.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiAddC_8u_C1RSfs(src.get(), src.step(), constant, dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR) << "NPP AddC failed";

    std::vector<Npp8u> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData)) << "Result mismatch";
  }
}

INSTANTIATE_TEST_SUITE_P(AddC8uSfs, AddC8uSfsParamTest,
                         ::testing::Values(AddC8uSfsParam{32, 32, 30, 0, false, false, "32x32_sfs0_noCtx"},
                                           AddC8uSfsParam{32, 32, 30, 0, true, false, "32x32_sfs0_Ctx"},
                                           AddC8uSfsParam{32, 32, 20, 0, false, true, "32x32_sfs0_InPlace"},
                                           AddC8uSfsParam{32, 32, 20, 0, true, true, "32x32_sfs0_InPlace_Ctx"},
                                           AddC8uSfsParam{64, 64, 50, 0, false, false, "64x64_sfs0_noCtx"}),
                         [](const ::testing::TestParamInfo<AddC8uSfsParam> &info) { return info.param.name; });

// Parameter structure for AddC 16u with scale factor tests
struct AddC16uSfsParam {
  int width;
  int height;
  Npp16u constant;
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class AddC16uSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<AddC16uSfsParam> {};

TEST_P(AddC16uSfsParamTest, AddC_16u_Sfs_Parameterized) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const Npp16u constant = param.constant;
  const int scaleFactor = param.scaleFactor;

  std::vector<Npp16u> srcData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(30000), 12345);

  std::vector<Npp16u> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::add_c_sfs<Npp16u>(srcData[i], constant, scaleFactor);
  }

  NppImageMemory<Npp16u> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiAddC_16u_C1IRSfs_Ctx(constant, src.get(), src.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiAddC_16u_C1IRSfs(constant, src.get(), src.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR) << "NPP AddC in-place failed";

    std::vector<Npp16u> resultData(width * height);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData)) << "Result mismatch";
  } else {
    NppImageMemory<Npp16u> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiAddC_16u_C1RSfs_Ctx(src.get(), src.step(), constant, dst.get(), dst.step(), roi, scaleFactor, ctx);
    } else {
      status = nppiAddC_16u_C1RSfs(src.get(), src.step(), constant, dst.get(), dst.step(), roi, scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR) << "NPP AddC failed";

    std::vector<Npp16u> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData)) << "Result mismatch";
  }
}

INSTANTIATE_TEST_SUITE_P(AddC16uSfs, AddC16uSfsParamTest,
                         ::testing::Values(AddC16uSfsParam{32, 32, 1000, 0, false, false, "32x32_sfs0_noCtx"},
                                           AddC16uSfsParam{32, 32, 1000, 0, true, false, "32x32_sfs0_Ctx"},
                                           AddC16uSfsParam{32, 32, 500, 0, false, true, "32x32_sfs0_InPlace"},
                                           AddC16uSfsParam{32, 32, 500, 0, true, true, "32x32_sfs0_InPlace_Ctx"},
                                           AddC16uSfsParam{64, 64, 2000, 0, false, false, "64x64_sfs0_noCtx"}),
                         [](const ::testing::TestParamInfo<AddC16uSfsParam> &info) { return info.param.name; });

// ==================== AddC 16f (half-precision float) tests ====================

class AddC16fTest : public NppTestBase {};

TEST_F(AddC16fTest, AddC_16f_C1R_BasicOperation) {
  const int width = 32;
  const int height = 32;
  const Npp32f constant = 10.5f;

  std::vector<Npp16f> srcData(width * height);
  std::vector<Npp16f> expectedData(width * height);

  TestDataGenerator::generateRandom16f(srcData, -50.0f, 50.0f, 12345);

  for (size_t i = 0; i < expectedData.size(); i++) {
    float val = npp16f_to_float_host(srcData[i]);
    expectedData[i] = float_to_npp16f_host(val + constant);
  }

  NppImageMemory<Npp16f> src(width, height);
  NppImageMemory<Npp16f> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiAddC_16f_C1R(src.get(), src.step(), constant, dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAddC_16f_C1R failed";

  std::vector<Npp16f> resultData(width * height);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f))
      << "AddC 16f C1R operation produced incorrect results";
}

TEST_F(AddC16fTest, AddC_16f_C1R_Ctx_BasicOperation) {
  const int width = 32;
  const int height = 32;
  const Npp32f constant = 10.5f;

  std::vector<Npp16f> srcData(width * height);
  std::vector<Npp16f> expectedData(width * height);

  TestDataGenerator::generateRandom16f(srcData, -50.0f, 50.0f, 12345);

  for (size_t i = 0; i < expectedData.size(); i++) {
    float val = npp16f_to_float_host(srcData[i]);
    expectedData[i] = float_to_npp16f_host(val + constant);
  }

  NppImageMemory<Npp16f> src(width, height);
  NppImageMemory<Npp16f> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  ctx.hStream = 0;
  NppStatus status = nppiAddC_16f_C1R_Ctx(src.get(), src.step(), constant, dst.get(), dst.step(), roi, ctx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAddC_16f_C1R_Ctx failed";

  std::vector<Npp16f> resultData(width * height);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f))
      << "AddC 16f C1R Ctx operation produced incorrect results";
}

TEST_F(AddC16fTest, AddC_16f_C1IR_InPlaceOperation) {
  const int width = 32;
  const int height = 32;
  const Npp32f constant = 15.0f;

  std::vector<Npp16f> srcData(width * height);
  std::vector<Npp16f> expectedData(width * height);

  TestDataGenerator::generateRandom16f(srcData, -50.0f, 50.0f, 54321);

  for (size_t i = 0; i < expectedData.size(); i++) {
    float val = npp16f_to_float_host(srcData[i]);
    expectedData[i] = float_to_npp16f_host(val + constant);
  }

  NppImageMemory<Npp16f> srcDst(width, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiAddC_16f_C1IR(constant, srcDst.get(), srcDst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAddC_16f_C1IR failed";

  std::vector<Npp16f> resultData(width * height);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f))
      << "In-place AddC 16f C1 operation produced incorrect results";
}

TEST_F(AddC16fTest, AddC_16f_C1IR_Ctx_InPlaceOperation) {
  const int width = 32;
  const int height = 32;
  const Npp32f constant = 15.0f;

  std::vector<Npp16f> srcData(width * height);
  std::vector<Npp16f> expectedData(width * height);

  TestDataGenerator::generateRandom16f(srcData, -50.0f, 50.0f, 54321);

  for (size_t i = 0; i < expectedData.size(); i++) {
    float val = npp16f_to_float_host(srcData[i]);
    expectedData[i] = float_to_npp16f_host(val + constant);
  }

  NppImageMemory<Npp16f> srcDst(width, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  ctx.hStream = 0;
  NppStatus status = nppiAddC_16f_C1IR_Ctx(constant, srcDst.get(), srcDst.step(), roi, ctx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAddC_16f_C1IR_Ctx failed";

  std::vector<Npp16f> resultData(width * height);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f))
      << "In-place AddC 16f C1 Ctx operation produced incorrect results";
}

// 16f C3 tests
TEST_F(AddC16fTest, AddC_16f_C3R_BasicOperation) {
  const int width = 32;
  const int height = 32;
  const int channels = 3;
  const Npp32f aConstants[3] = {5.0f, 10.0f, 15.0f};

  std::vector<Npp16f> srcData(width * height * channels);
  std::vector<Npp16f> expectedData(width * height * channels);

  TestDataGenerator::generateRandom16f(srcData, -50.0f, 50.0f, 33333);

  for (size_t i = 0; i < width * height; i++) {
    for (int c = 0; c < channels; c++) {
      float val = npp16f_to_float_host(srcData[i * channels + c]);
      expectedData[i * channels + c] = float_to_npp16f_host(val + aConstants[c]);
    }
  }

  NppImageMemory<Npp16f> src(width * channels, height);
  NppImageMemory<Npp16f> dst(width * channels, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiAddC_16f_C3R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAddC_16f_C3R failed";

  std::vector<Npp16f> resultData(width * height * channels);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f))
      << "AddC 16f C3R operation produced incorrect results";
}

TEST_F(AddC16fTest, AddC_16f_C3R_Ctx_BasicOperation) {
  const int width = 32;
  const int height = 32;
  const int channels = 3;
  const Npp32f aConstants[3] = {5.0f, 10.0f, 15.0f};

  std::vector<Npp16f> srcData(width * height * channels);
  std::vector<Npp16f> expectedData(width * height * channels);

  TestDataGenerator::generateRandom16f(srcData, -50.0f, 50.0f, 33333);

  for (size_t i = 0; i < width * height; i++) {
    for (int c = 0; c < channels; c++) {
      float val = npp16f_to_float_host(srcData[i * channels + c]);
      expectedData[i * channels + c] = float_to_npp16f_host(val + aConstants[c]);
    }
  }

  NppImageMemory<Npp16f> src(width * channels, height);
  NppImageMemory<Npp16f> dst(width * channels, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  ctx.hStream = 0;
  NppStatus status = nppiAddC_16f_C3R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAddC_16f_C3R_Ctx failed";

  std::vector<Npp16f> resultData(width * height * channels);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f))
      << "AddC 16f C3R Ctx operation produced incorrect results";
}

TEST_F(AddC16fTest, AddC_16f_C3IR_InPlaceOperation) {
  const int width = 32;
  const int height = 32;
  const int channels = 3;
  const Npp32f aConstants[3] = {5.0f, 10.0f, 15.0f};

  std::vector<Npp16f> srcData(width * height * channels);
  std::vector<Npp16f> expectedData(width * height * channels);

  TestDataGenerator::generateRandom16f(srcData, -50.0f, 50.0f, 44444);

  for (size_t i = 0; i < width * height; i++) {
    for (int c = 0; c < channels; c++) {
      float val = npp16f_to_float_host(srcData[i * channels + c]);
      expectedData[i * channels + c] = float_to_npp16f_host(val + aConstants[c]);
    }
  }

  NppImageMemory<Npp16f> srcDst(width * channels, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiAddC_16f_C3IR(aConstants, srcDst.get(), srcDst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAddC_16f_C3IR failed";

  std::vector<Npp16f> resultData(width * height * channels);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f))
      << "In-place AddC 16f C3 operation produced incorrect results";
}

TEST_F(AddC16fTest, AddC_16f_C3IR_Ctx_InPlaceOperation) {
  const int width = 32;
  const int height = 32;
  const int channels = 3;
  const Npp32f aConstants[3] = {5.0f, 10.0f, 15.0f};

  std::vector<Npp16f> srcData(width * height * channels);
  std::vector<Npp16f> expectedData(width * height * channels);

  TestDataGenerator::generateRandom16f(srcData, -50.0f, 50.0f, 44444);

  for (size_t i = 0; i < width * height; i++) {
    for (int c = 0; c < channels; c++) {
      float val = npp16f_to_float_host(srcData[i * channels + c]);
      expectedData[i * channels + c] = float_to_npp16f_host(val + aConstants[c]);
    }
  }

  NppImageMemory<Npp16f> srcDst(width * channels, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  ctx.hStream = 0;
  NppStatus status = nppiAddC_16f_C3IR_Ctx(aConstants, srcDst.get(), srcDst.step(), roi, ctx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAddC_16f_C3IR_Ctx failed";

  std::vector<Npp16f> resultData(width * height * channels);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f))
      << "In-place AddC 16f C3 Ctx operation produced incorrect results";
}

// 16f C4 tests
TEST_F(AddC16fTest, AddC_16f_C4R_BasicOperation) {
  const int width = 32;
  const int height = 32;
  const int channels = 4;
  const Npp32f aConstants[4] = {5.0f, 10.0f, 15.0f, 20.0f};

  std::vector<Npp16f> srcData(width * height * channels);
  std::vector<Npp16f> expectedData(width * height * channels);

  TestDataGenerator::generateRandom16f(srcData, -50.0f, 50.0f, 55555);

  for (size_t i = 0; i < width * height; i++) {
    for (int c = 0; c < channels; c++) {
      float val = npp16f_to_float_host(srcData[i * channels + c]);
      expectedData[i * channels + c] = float_to_npp16f_host(val + aConstants[c]);
    }
  }

  NppImageMemory<Npp16f> src(width * channels, height);
  NppImageMemory<Npp16f> dst(width * channels, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiAddC_16f_C4R(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAddC_16f_C4R failed";

  std::vector<Npp16f> resultData(width * height * channels);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f))
      << "AddC 16f C4R operation produced incorrect results";
}

TEST_F(AddC16fTest, AddC_16f_C4R_Ctx_BasicOperation) {
  const int width = 32;
  const int height = 32;
  const int channels = 4;
  const Npp32f aConstants[4] = {5.0f, 10.0f, 15.0f, 20.0f};

  std::vector<Npp16f> srcData(width * height * channels);
  std::vector<Npp16f> expectedData(width * height * channels);

  TestDataGenerator::generateRandom16f(srcData, -50.0f, 50.0f, 55555);

  for (size_t i = 0; i < width * height; i++) {
    for (int c = 0; c < channels; c++) {
      float val = npp16f_to_float_host(srcData[i * channels + c]);
      expectedData[i * channels + c] = float_to_npp16f_host(val + aConstants[c]);
    }
  }

  NppImageMemory<Npp16f> src(width * channels, height);
  NppImageMemory<Npp16f> dst(width * channels, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  ctx.hStream = 0;
  NppStatus status = nppiAddC_16f_C4R_Ctx(src.get(), src.step(), aConstants, dst.get(), dst.step(), roi, ctx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAddC_16f_C4R_Ctx failed";

  std::vector<Npp16f> resultData(width * height * channels);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f))
      << "AddC 16f C4R Ctx operation produced incorrect results";
}

TEST_F(AddC16fTest, AddC_16f_C4IR_InPlaceOperation) {
  const int width = 32;
  const int height = 32;
  const int channels = 4;
  const Npp32f aConstants[4] = {5.0f, 10.0f, 15.0f, 20.0f};

  std::vector<Npp16f> srcData(width * height * channels);
  std::vector<Npp16f> expectedData(width * height * channels);

  TestDataGenerator::generateRandom16f(srcData, -50.0f, 50.0f, 66666);

  for (size_t i = 0; i < width * height; i++) {
    for (int c = 0; c < channels; c++) {
      float val = npp16f_to_float_host(srcData[i * channels + c]);
      expectedData[i * channels + c] = float_to_npp16f_host(val + aConstants[c]);
    }
  }

  NppImageMemory<Npp16f> srcDst(width * channels, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiAddC_16f_C4IR(aConstants, srcDst.get(), srcDst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAddC_16f_C4IR failed";

  std::vector<Npp16f> resultData(width * height * channels);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f))
      << "In-place AddC 16f C4 operation produced incorrect results";
}

TEST_F(AddC16fTest, AddC_16f_C4IR_Ctx_InPlaceOperation) {
  const int width = 32;
  const int height = 32;
  const int channels = 4;
  const Npp32f aConstants[4] = {5.0f, 10.0f, 15.0f, 20.0f};

  std::vector<Npp16f> srcData(width * height * channels);
  std::vector<Npp16f> expectedData(width * height * channels);

  TestDataGenerator::generateRandom16f(srcData, -50.0f, 50.0f, 66666);

  for (size_t i = 0; i < width * height; i++) {
    for (int c = 0; c < channels; c++) {
      float val = npp16f_to_float_host(srcData[i * channels + c]);
      expectedData[i * channels + c] = float_to_npp16f_host(val + aConstants[c]);
    }
  }

  NppImageMemory<Npp16f> srcDst(width * channels, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  ctx.hStream = 0;
  NppStatus status = nppiAddC_16f_C4IR_Ctx(aConstants, srcDst.get(), srcDst.step(), roi, ctx);

  ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAddC_16f_C4IR_Ctx failed";

  std::vector<Npp16f> resultData(width * height * channels);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual16f(resultData, expectedData, 1e-2f))
      << "In-place AddC 16f C4 Ctx operation produced incorrect results";
}
