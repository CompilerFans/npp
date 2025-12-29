#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Common Param Structures ====================

struct AddAccumParam {
  int width;
  int height;
  bool use_mask;
  bool use_ctx;
  std::string name;
};

// ==================== AddProduct 8u32f C1IR TEST_P ====================

class AddProduct8u32fC1IRParamTest : public NppTestBase, public ::testing::WithParamInterface<AddAccumParam> {};

TEST_P(AddProduct8u32fC1IRParamTest, AddProduct_8u32f_C1IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> src1Data(width * height);
  std::vector<Npp8u> src2Data(width * height);
  std::vector<Npp32f> accumData(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(0), static_cast<Npp8u>(10), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8u>(0), static_cast<Npp8u>(10), 54321);
  TestDataGenerator::generateRandom(accumData, 0.0f, 100.0f, 11111);

  NppImageMemory<Npp8u> src1(width, height);
  NppImageMemory<Npp8u> src2(width, height);
  NppImageMemory<Npp32f> accum(width, height);

  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);
  accum.copyFromHost(accumData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAddProduct_8u32f_C1IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), accum.get(), accum.step(),
                                           roi, ctx);
  } else {
    status =
        nppiAddProduct_8u32f_C1IR(src1.get(), src1.step(), src2.get(), src2.step(), accum.get(), accum.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AddProduct8u32fC1IR, AddProduct8u32fC1IRParamTest,
                         ::testing::Values(AddAccumParam{32, 32, false, false, "32x32_noCtx"},
                                           AddAccumParam{32, 32, false, true, "32x32_Ctx"},
                                           AddAccumParam{64, 64, false, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AddAccumParam> &info) { return info.param.name; });

// ==================== AddProduct 8u32f C1IMR TEST_P ====================

class AddProduct8u32fC1IMRParamTest : public NppTestBase, public ::testing::WithParamInterface<AddAccumParam> {};

TEST_P(AddProduct8u32fC1IMRParamTest, AddProduct_8u32f_C1IMR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> src1Data(width * height);
  std::vector<Npp8u> src2Data(width * height);
  std::vector<Npp8u> maskData(width * height);
  std::vector<Npp32f> accumData(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(0), static_cast<Npp8u>(10), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8u>(0), static_cast<Npp8u>(10), 54321);
  TestDataGenerator::generateRandom(maskData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 99999);
  TestDataGenerator::generateRandom(accumData, 0.0f, 100.0f, 11111);

  NppImageMemory<Npp8u> src1(width, height);
  NppImageMemory<Npp8u> src2(width, height);
  NppImageMemory<Npp8u> mask(width, height);
  NppImageMemory<Npp32f> accum(width, height);

  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);
  mask.copyFromHost(maskData);
  accum.copyFromHost(accumData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAddProduct_8u32f_C1IMR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), mask.get(), mask.step(),
                                            accum.get(), accum.step(), roi, ctx);
  } else {
    status = nppiAddProduct_8u32f_C1IMR(src1.get(), src1.step(), src2.get(), src2.step(), mask.get(), mask.step(),
                                        accum.get(), accum.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AddProduct8u32fC1IMR, AddProduct8u32fC1IMRParamTest,
                         ::testing::Values(AddAccumParam{32, 32, true, false, "32x32_noCtx"},
                                           AddAccumParam{32, 32, true, true, "32x32_Ctx"},
                                           AddAccumParam{64, 64, true, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AddAccumParam> &info) { return info.param.name; });

// ==================== AddProduct 16u32f C1IR TEST_P ====================

class AddProduct16u32fC1IRParamTest : public NppTestBase, public ::testing::WithParamInterface<AddAccumParam> {};

TEST_P(AddProduct16u32fC1IRParamTest, AddProduct_16u32f_C1IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> src1Data(width * height);
  std::vector<Npp16u> src2Data(width * height);
  std::vector<Npp32f> accumData(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(0), static_cast<Npp16u>(100), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(0), static_cast<Npp16u>(100), 54321);
  TestDataGenerator::generateRandom(accumData, 0.0f, 10000.0f, 11111);

  NppImageMemory<Npp16u> src1(width, height);
  NppImageMemory<Npp16u> src2(width, height);
  NppImageMemory<Npp32f> accum(width, height);

  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);
  accum.copyFromHost(accumData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAddProduct_16u32f_C1IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), accum.get(), accum.step(),
                                            roi, ctx);
  } else {
    status =
        nppiAddProduct_16u32f_C1IR(src1.get(), src1.step(), src2.get(), src2.step(), accum.get(), accum.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AddProduct16u32fC1IR, AddProduct16u32fC1IRParamTest,
                         ::testing::Values(AddAccumParam{32, 32, false, false, "32x32_noCtx"},
                                           AddAccumParam{32, 32, false, true, "32x32_Ctx"},
                                           AddAccumParam{64, 64, false, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AddAccumParam> &info) { return info.param.name; });

// ==================== AddProduct 16u32f C1IMR TEST_P ====================

class AddProduct16u32fC1IMRParamTest : public NppTestBase, public ::testing::WithParamInterface<AddAccumParam> {};

TEST_P(AddProduct16u32fC1IMRParamTest, AddProduct_16u32f_C1IMR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> src1Data(width * height);
  std::vector<Npp16u> src2Data(width * height);
  std::vector<Npp8u> maskData(width * height);
  std::vector<Npp32f> accumData(width * height);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(0), static_cast<Npp16u>(100), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(0), static_cast<Npp16u>(100), 54321);
  TestDataGenerator::generateRandom(maskData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 99999);
  TestDataGenerator::generateRandom(accumData, 0.0f, 10000.0f, 11111);

  NppImageMemory<Npp16u> src1(width, height);
  NppImageMemory<Npp16u> src2(width, height);
  NppImageMemory<Npp8u> mask(width, height);
  NppImageMemory<Npp32f> accum(width, height);

  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);
  mask.copyFromHost(maskData);
  accum.copyFromHost(accumData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAddProduct_16u32f_C1IMR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), mask.get(), mask.step(),
                                             accum.get(), accum.step(), roi, ctx);
  } else {
    status = nppiAddProduct_16u32f_C1IMR(src1.get(), src1.step(), src2.get(), src2.step(), mask.get(), mask.step(),
                                         accum.get(), accum.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AddProduct16u32fC1IMR, AddProduct16u32fC1IMRParamTest,
                         ::testing::Values(AddAccumParam{32, 32, true, false, "32x32_noCtx"},
                                           AddAccumParam{32, 32, true, true, "32x32_Ctx"},
                                           AddAccumParam{64, 64, true, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AddAccumParam> &info) { return info.param.name; });

// ==================== AddProduct 16f C1IR TEST_P ====================

class AddProduct16fC1IRParamTest : public NppTestBase, public ::testing::WithParamInterface<AddAccumParam> {};

TEST_P(AddProduct16fC1IRParamTest, AddProduct_16f_C1IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16f> src1Data(width * height);
  std::vector<Npp16f> src2Data(width * height);
  std::vector<Npp16f> accumData(width * height);

  Npp16f *d_src1 = nullptr;
  Npp16f *d_src2 = nullptr;
  Npp16f *d_accum = nullptr;
  int src1Step, src2Step, accumStep;

  d_src1 = reinterpret_cast<Npp16f *>(nppiMalloc_16u_C1(width, height, &src1Step));
  d_src2 = reinterpret_cast<Npp16f *>(nppiMalloc_16u_C1(width, height, &src2Step));
  d_accum = reinterpret_cast<Npp16f *>(nppiMalloc_16u_C1(width, height, &accumStep));

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_src2, nullptr);
  ASSERT_NE(d_accum, nullptr);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAddProduct_16f_C1IR_Ctx(d_src1, src1Step, d_src2, src2Step, d_accum, accumStep, roi, ctx);
  } else {
    status = nppiAddProduct_16f_C1IR(d_src1, src1Step, d_src2, src2Step, d_accum, accumStep, roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  nppiFree(d_src1);
  nppiFree(d_src2);
  nppiFree(d_accum);
}

// Note: 16f_C1IR_Ctx may return NPP_NOT_SUPPORTED_MODE_ERROR on some GPUs
INSTANTIATE_TEST_SUITE_P(AddProduct16fC1IR, AddProduct16fC1IRParamTest,
                         ::testing::Values(AddAccumParam{32, 32, false, false, "32x32_noCtx"},
                                           AddAccumParam{64, 64, false, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AddAccumParam> &info) { return info.param.name; });

// ==================== AddProduct 32f C1IR TEST_P ====================

class AddProduct32fC1IRParamTest : public NppTestBase, public ::testing::WithParamInterface<AddAccumParam> {};

TEST_P(AddProduct32fC1IRParamTest, AddProduct_32f_C1IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> src1Data(width * height);
  std::vector<Npp32f> src2Data(width * height);
  std::vector<Npp32f> accumData(width * height);
  TestDataGenerator::generateRandom(src1Data, 0.0f, 1.0f, 12345);
  TestDataGenerator::generateRandom(src2Data, 0.0f, 1.0f, 54321);
  TestDataGenerator::generateRandom(accumData, 0.0f, 100.0f, 11111);

  NppImageMemory<Npp32f> src1(width, height);
  NppImageMemory<Npp32f> src2(width, height);
  NppImageMemory<Npp32f> accum(width, height);

  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);
  accum.copyFromHost(accumData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAddProduct_32f_C1IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), accum.get(), accum.step(),
                                         roi, ctx);
  } else {
    status = nppiAddProduct_32f_C1IR(src1.get(), src1.step(), src2.get(), src2.step(), accum.get(), accum.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AddProduct32fC1IR, AddProduct32fC1IRParamTest,
                         ::testing::Values(AddAccumParam{32, 32, false, false, "32x32_noCtx"},
                                           AddAccumParam{32, 32, false, true, "32x32_Ctx"},
                                           AddAccumParam{64, 64, false, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AddAccumParam> &info) { return info.param.name; });

// ==================== AddProduct 32f C1IMR TEST_P ====================

class AddProduct32fC1IMRParamTest : public NppTestBase, public ::testing::WithParamInterface<AddAccumParam> {};

TEST_P(AddProduct32fC1IMRParamTest, AddProduct_32f_C1IMR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> src1Data(width * height);
  std::vector<Npp32f> src2Data(width * height);
  std::vector<Npp8u> maskData(width * height);
  std::vector<Npp32f> accumData(width * height);
  TestDataGenerator::generateRandom(src1Data, 0.0f, 1.0f, 12345);
  TestDataGenerator::generateRandom(src2Data, 0.0f, 1.0f, 54321);
  TestDataGenerator::generateRandom(maskData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 99999);
  TestDataGenerator::generateRandom(accumData, 0.0f, 100.0f, 11111);

  NppImageMemory<Npp32f> src1(width, height);
  NppImageMemory<Npp32f> src2(width, height);
  NppImageMemory<Npp8u> mask(width, height);
  NppImageMemory<Npp32f> accum(width, height);

  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);
  mask.copyFromHost(maskData);
  accum.copyFromHost(accumData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAddProduct_32f_C1IMR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), mask.get(), mask.step(),
                                          accum.get(), accum.step(), roi, ctx);
  } else {
    status = nppiAddProduct_32f_C1IMR(src1.get(), src1.step(), src2.get(), src2.step(), mask.get(), mask.step(),
                                      accum.get(), accum.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AddProduct32fC1IMR, AddProduct32fC1IMRParamTest,
                         ::testing::Values(AddAccumParam{32, 32, true, false, "32x32_noCtx"},
                                           AddAccumParam{32, 32, true, true, "32x32_Ctx"},
                                           AddAccumParam{64, 64, true, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AddAccumParam> &info) { return info.param.name; });

// ==================== AddSquare 8u32f C1IR TEST_P ====================

class AddSquare8u32fC1IRParamTest : public NppTestBase, public ::testing::WithParamInterface<AddAccumParam> {};

TEST_P(AddSquare8u32fC1IRParamTest, AddSquare_8u32f_C1IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height);
  std::vector<Npp32f> accumData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(10), 12345);
  TestDataGenerator::generateRandom(accumData, 0.0f, 100.0f, 11111);

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp32f> accum(width, height);

  src.copyFromHost(srcData);
  accum.copyFromHost(accumData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAddSquare_8u32f_C1IR_Ctx(src.get(), src.step(), accum.get(), accum.step(), roi, ctx);
  } else {
    status = nppiAddSquare_8u32f_C1IR(src.get(), src.step(), accum.get(), accum.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AddSquare8u32fC1IR, AddSquare8u32fC1IRParamTest,
                         ::testing::Values(AddAccumParam{32, 32, false, false, "32x32_noCtx"},
                                           AddAccumParam{32, 32, false, true, "32x32_Ctx"},
                                           AddAccumParam{64, 64, false, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AddAccumParam> &info) { return info.param.name; });

// ==================== AddSquare 8u32f C1IMR TEST_P ====================

class AddSquare8u32fC1IMRParamTest : public NppTestBase, public ::testing::WithParamInterface<AddAccumParam> {};

TEST_P(AddSquare8u32fC1IMRParamTest, AddSquare_8u32f_C1IMR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height);
  std::vector<Npp8u> maskData(width * height);
  std::vector<Npp32f> accumData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(10), 12345);
  TestDataGenerator::generateRandom(maskData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 99999);
  TestDataGenerator::generateRandom(accumData, 0.0f, 100.0f, 11111);

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> mask(width, height);
  NppImageMemory<Npp32f> accum(width, height);

  src.copyFromHost(srcData);
  mask.copyFromHost(maskData);
  accum.copyFromHost(accumData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAddSquare_8u32f_C1IMR_Ctx(src.get(), src.step(), mask.get(), mask.step(), accum.get(), accum.step(),
                                           roi, ctx);
  } else {
    status = nppiAddSquare_8u32f_C1IMR(src.get(), src.step(), mask.get(), mask.step(), accum.get(), accum.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AddSquare8u32fC1IMR, AddSquare8u32fC1IMRParamTest,
                         ::testing::Values(AddAccumParam{32, 32, true, false, "32x32_noCtx"},
                                           AddAccumParam{32, 32, true, true, "32x32_Ctx"},
                                           AddAccumParam{64, 64, true, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AddAccumParam> &info) { return info.param.name; });

// ==================== AddSquare 16u32f C1IR TEST_P ====================

class AddSquare16u32fC1IRParamTest : public NppTestBase, public ::testing::WithParamInterface<AddAccumParam> {};

TEST_P(AddSquare16u32fC1IRParamTest, AddSquare_16u32f_C1IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcData(width * height);
  std::vector<Npp32f> accumData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(100), 12345);
  TestDataGenerator::generateRandom(accumData, 0.0f, 10000.0f, 11111);

  NppImageMemory<Npp16u> src(width, height);
  NppImageMemory<Npp32f> accum(width, height);

  src.copyFromHost(srcData);
  accum.copyFromHost(accumData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAddSquare_16u32f_C1IR_Ctx(src.get(), src.step(), accum.get(), accum.step(), roi, ctx);
  } else {
    status = nppiAddSquare_16u32f_C1IR(src.get(), src.step(), accum.get(), accum.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AddSquare16u32fC1IR, AddSquare16u32fC1IRParamTest,
                         ::testing::Values(AddAccumParam{32, 32, false, false, "32x32_noCtx"},
                                           AddAccumParam{32, 32, false, true, "32x32_Ctx"},
                                           AddAccumParam{64, 64, false, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AddAccumParam> &info) { return info.param.name; });

// ==================== AddSquare 16u32f C1IMR TEST_P ====================

class AddSquare16u32fC1IMRParamTest : public NppTestBase, public ::testing::WithParamInterface<AddAccumParam> {};

TEST_P(AddSquare16u32fC1IMRParamTest, AddSquare_16u32f_C1IMR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcData(width * height);
  std::vector<Npp8u> maskData(width * height);
  std::vector<Npp32f> accumData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(100), 12345);
  TestDataGenerator::generateRandom(maskData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 99999);
  TestDataGenerator::generateRandom(accumData, 0.0f, 10000.0f, 11111);

  NppImageMemory<Npp16u> src(width, height);
  NppImageMemory<Npp8u> mask(width, height);
  NppImageMemory<Npp32f> accum(width, height);

  src.copyFromHost(srcData);
  mask.copyFromHost(maskData);
  accum.copyFromHost(accumData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAddSquare_16u32f_C1IMR_Ctx(src.get(), src.step(), mask.get(), mask.step(), accum.get(), accum.step(),
                                            roi, ctx);
  } else {
    status = nppiAddSquare_16u32f_C1IMR(src.get(), src.step(), mask.get(), mask.step(), accum.get(), accum.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AddSquare16u32fC1IMR, AddSquare16u32fC1IMRParamTest,
                         ::testing::Values(AddAccumParam{32, 32, true, false, "32x32_noCtx"},
                                           AddAccumParam{32, 32, true, true, "32x32_Ctx"},
                                           AddAccumParam{64, 64, true, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AddAccumParam> &info) { return info.param.name; });

// ==================== AddSquare 32f C1IR TEST_P ====================

class AddSquare32fC1IRParamTest : public NppTestBase, public ::testing::WithParamInterface<AddAccumParam> {};

TEST_P(AddSquare32fC1IRParamTest, AddSquare_32f_C1IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> srcData(width * height);
  std::vector<Npp32f> accumData(width * height);
  TestDataGenerator::generateRandom(srcData, 0.0f, 1.0f, 12345);
  TestDataGenerator::generateRandom(accumData, 0.0f, 100.0f, 11111);

  NppImageMemory<Npp32f> src(width, height);
  NppImageMemory<Npp32f> accum(width, height);

  src.copyFromHost(srcData);
  accum.copyFromHost(accumData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAddSquare_32f_C1IR_Ctx(src.get(), src.step(), accum.get(), accum.step(), roi, ctx);
  } else {
    status = nppiAddSquare_32f_C1IR(src.get(), src.step(), accum.get(), accum.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AddSquare32fC1IR, AddSquare32fC1IRParamTest,
                         ::testing::Values(AddAccumParam{32, 32, false, false, "32x32_noCtx"},
                                           AddAccumParam{32, 32, false, true, "32x32_Ctx"},
                                           AddAccumParam{64, 64, false, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AddAccumParam> &info) { return info.param.name; });

// ==================== AddSquare 32f C1IMR TEST_P ====================

class AddSquare32fC1IMRParamTest : public NppTestBase, public ::testing::WithParamInterface<AddAccumParam> {};

TEST_P(AddSquare32fC1IMRParamTest, AddSquare_32f_C1IMR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> srcData(width * height);
  std::vector<Npp8u> maskData(width * height);
  std::vector<Npp32f> accumData(width * height);
  TestDataGenerator::generateRandom(srcData, 0.0f, 1.0f, 12345);
  TestDataGenerator::generateRandom(maskData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 99999);
  TestDataGenerator::generateRandom(accumData, 0.0f, 100.0f, 11111);

  NppImageMemory<Npp32f> src(width, height);
  NppImageMemory<Npp8u> mask(width, height);
  NppImageMemory<Npp32f> accum(width, height);

  src.copyFromHost(srcData);
  mask.copyFromHost(maskData);
  accum.copyFromHost(accumData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAddSquare_32f_C1IMR_Ctx(src.get(), src.step(), mask.get(), mask.step(), accum.get(), accum.step(), roi,
                                         ctx);
  } else {
    status = nppiAddSquare_32f_C1IMR(src.get(), src.step(), mask.get(), mask.step(), accum.get(), accum.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AddSquare32fC1IMR, AddSquare32fC1IMRParamTest,
                         ::testing::Values(AddAccumParam{32, 32, true, false, "32x32_noCtx"},
                                           AddAccumParam{32, 32, true, true, "32x32_Ctx"},
                                           AddAccumParam{64, 64, true, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<AddAccumParam> &info) { return info.param.name; });

// ==================== AddWeighted Param Structures ====================

struct AddWeightedParam {
  int width;
  int height;
  Npp32f alpha;
  bool use_mask;
  bool use_ctx;
  std::string name;
};

// ==================== AddWeighted 8u32f C1IR TEST_P ====================

class AddWeighted8u32fC1IRParamTest : public NppTestBase, public ::testing::WithParamInterface<AddWeightedParam> {};

TEST_P(AddWeighted8u32fC1IRParamTest, AddWeighted_8u32f_C1IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height);
  std::vector<Npp32f> accumData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);
  TestDataGenerator::generateRandom(accumData, 0.0f, 255.0f, 11111);

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp32f> accum(width, height);

  src.copyFromHost(srcData);
  accum.copyFromHost(accumData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAddWeighted_8u32f_C1IR_Ctx(src.get(), src.step(), accum.get(), accum.step(), roi, param.alpha, ctx);
  } else {
    status = nppiAddWeighted_8u32f_C1IR(src.get(), src.step(), accum.get(), accum.step(), roi, param.alpha);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AddWeighted8u32fC1IR, AddWeighted8u32fC1IRParamTest,
                         ::testing::Values(AddWeightedParam{32, 32, 0.5f, false, false, "32x32_a05_noCtx"},
                                           AddWeightedParam{32, 32, 0.5f, false, true, "32x32_a05_Ctx"},
                                           AddWeightedParam{64, 64, 0.2f, false, false, "64x64_a02_noCtx"}),
                         [](const ::testing::TestParamInfo<AddWeightedParam> &info) { return info.param.name; });

// ==================== AddWeighted 8u32f C1IMR TEST_P ====================

class AddWeighted8u32fC1IMRParamTest : public NppTestBase, public ::testing::WithParamInterface<AddWeightedParam> {};

TEST_P(AddWeighted8u32fC1IMRParamTest, AddWeighted_8u32f_C1IMR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp8u> srcData(width * height);
  std::vector<Npp8u> maskData(width * height);
  std::vector<Npp32f> accumData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);
  TestDataGenerator::generateRandom(maskData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 99999);
  TestDataGenerator::generateRandom(accumData, 0.0f, 255.0f, 11111);

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> mask(width, height);
  NppImageMemory<Npp32f> accum(width, height);

  src.copyFromHost(srcData);
  mask.copyFromHost(maskData);
  accum.copyFromHost(accumData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAddWeighted_8u32f_C1IMR_Ctx(src.get(), src.step(), mask.get(), mask.step(), accum.get(), accum.step(),
                                             roi, param.alpha, ctx);
  } else {
    status = nppiAddWeighted_8u32f_C1IMR(src.get(), src.step(), mask.get(), mask.step(), accum.get(), accum.step(), roi,
                                         param.alpha);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AddWeighted8u32fC1IMR, AddWeighted8u32fC1IMRParamTest,
                         ::testing::Values(AddWeightedParam{32, 32, 0.5f, true, false, "32x32_a05_noCtx"},
                                           AddWeightedParam{32, 32, 0.5f, true, true, "32x32_a05_Ctx"},
                                           AddWeightedParam{64, 64, 0.2f, true, false, "64x64_a02_noCtx"}),
                         [](const ::testing::TestParamInfo<AddWeightedParam> &info) { return info.param.name; });

// ==================== AddWeighted 16u32f C1IR TEST_P ====================

class AddWeighted16u32fC1IRParamTest : public NppTestBase, public ::testing::WithParamInterface<AddWeightedParam> {};

TEST_P(AddWeighted16u32fC1IRParamTest, AddWeighted_16u32f_C1IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcData(width * height);
  std::vector<Npp32f> accumData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);
  TestDataGenerator::generateRandom(accumData, 0.0f, 65535.0f, 11111);

  NppImageMemory<Npp16u> src(width, height);
  NppImageMemory<Npp32f> accum(width, height);

  src.copyFromHost(srcData);
  accum.copyFromHost(accumData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAddWeighted_16u32f_C1IR_Ctx(src.get(), src.step(), accum.get(), accum.step(), roi, param.alpha, ctx);
  } else {
    status = nppiAddWeighted_16u32f_C1IR(src.get(), src.step(), accum.get(), accum.step(), roi, param.alpha);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AddWeighted16u32fC1IR, AddWeighted16u32fC1IRParamTest,
                         ::testing::Values(AddWeightedParam{32, 32, 0.5f, false, false, "32x32_a05_noCtx"},
                                           AddWeightedParam{32, 32, 0.5f, false, true, "32x32_a05_Ctx"},
                                           AddWeightedParam{64, 64, 0.2f, false, false, "64x64_a02_noCtx"}),
                         [](const ::testing::TestParamInfo<AddWeightedParam> &info) { return info.param.name; });

// ==================== AddWeighted 16u32f C1IMR TEST_P ====================

class AddWeighted16u32fC1IMRParamTest : public NppTestBase, public ::testing::WithParamInterface<AddWeightedParam> {};

TEST_P(AddWeighted16u32fC1IMRParamTest, AddWeighted_16u32f_C1IMR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp16u> srcData(width * height);
  std::vector<Npp8u> maskData(width * height);
  std::vector<Npp32f> accumData(width * height);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);
  TestDataGenerator::generateRandom(maskData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 99999);
  TestDataGenerator::generateRandom(accumData, 0.0f, 65535.0f, 11111);

  NppImageMemory<Npp16u> src(width, height);
  NppImageMemory<Npp8u> mask(width, height);
  NppImageMemory<Npp32f> accum(width, height);

  src.copyFromHost(srcData);
  mask.copyFromHost(maskData);
  accum.copyFromHost(accumData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAddWeighted_16u32f_C1IMR_Ctx(src.get(), src.step(), mask.get(), mask.step(), accum.get(), accum.step(),
                                              roi, param.alpha, ctx);
  } else {
    status = nppiAddWeighted_16u32f_C1IMR(src.get(), src.step(), mask.get(), mask.step(), accum.get(), accum.step(),
                                          roi, param.alpha);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AddWeighted16u32fC1IMR, AddWeighted16u32fC1IMRParamTest,
                         ::testing::Values(AddWeightedParam{32, 32, 0.5f, true, false, "32x32_a05_noCtx"},
                                           AddWeightedParam{32, 32, 0.5f, true, true, "32x32_a05_Ctx"},
                                           AddWeightedParam{64, 64, 0.2f, true, false, "64x64_a02_noCtx"}),
                         [](const ::testing::TestParamInfo<AddWeightedParam> &info) { return info.param.name; });

// ==================== AddWeighted 32f C1IR TEST_P ====================

class AddWeighted32fC1IRParamTest : public NppTestBase, public ::testing::WithParamInterface<AddWeightedParam> {};

TEST_P(AddWeighted32fC1IRParamTest, AddWeighted_32f_C1IR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> srcData(width * height);
  std::vector<Npp32f> accumData(width * height);
  TestDataGenerator::generateRandom(srcData, 0.0f, 1.0f, 12345);
  TestDataGenerator::generateRandom(accumData, 0.0f, 1.0f, 11111);

  NppImageMemory<Npp32f> src(width, height);
  NppImageMemory<Npp32f> accum(width, height);

  src.copyFromHost(srcData);
  accum.copyFromHost(accumData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAddWeighted_32f_C1IR_Ctx(src.get(), src.step(), accum.get(), accum.step(), roi, param.alpha, ctx);
  } else {
    status = nppiAddWeighted_32f_C1IR(src.get(), src.step(), accum.get(), accum.step(), roi, param.alpha);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AddWeighted32fC1IR, AddWeighted32fC1IRParamTest,
                         ::testing::Values(AddWeightedParam{32, 32, 0.5f, false, false, "32x32_a05_noCtx"},
                                           AddWeightedParam{32, 32, 0.5f, false, true, "32x32_a05_Ctx"},
                                           AddWeightedParam{64, 64, 0.2f, false, false, "64x64_a02_noCtx"}),
                         [](const ::testing::TestParamInfo<AddWeightedParam> &info) { return info.param.name; });

// ==================== AddWeighted 32f C1IMR TEST_P ====================

class AddWeighted32fC1IMRParamTest : public NppTestBase, public ::testing::WithParamInterface<AddWeightedParam> {};

TEST_P(AddWeighted32fC1IMRParamTest, AddWeighted_32f_C1IMR) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> srcData(width * height);
  std::vector<Npp8u> maskData(width * height);
  std::vector<Npp32f> accumData(width * height);
  TestDataGenerator::generateRandom(srcData, 0.0f, 1.0f, 12345);
  TestDataGenerator::generateRandom(maskData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 99999);
  TestDataGenerator::generateRandom(accumData, 0.0f, 1.0f, 11111);

  NppImageMemory<Npp32f> src(width, height);
  NppImageMemory<Npp8u> mask(width, height);
  NppImageMemory<Npp32f> accum(width, height);

  src.copyFromHost(srcData);
  mask.copyFromHost(maskData);
  accum.copyFromHost(accumData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.use_ctx) {
    NppStreamContext ctx{};
    ctx.hStream = 0;
    status = nppiAddWeighted_32f_C1IMR_Ctx(src.get(), src.step(), mask.get(), mask.step(), accum.get(), accum.step(),
                                           roi, param.alpha, ctx);
  } else {
    status = nppiAddWeighted_32f_C1IMR(src.get(), src.step(), mask.get(), mask.step(), accum.get(), accum.step(), roi,
                                       param.alpha);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AddWeighted32fC1IMR, AddWeighted32fC1IMRParamTest,
                         ::testing::Values(AddWeightedParam{32, 32, 0.5f, true, false, "32x32_a05_noCtx"},
                                           AddWeightedParam{32, 32, 0.5f, true, true, "32x32_a05_Ctx"},
                                           AddWeightedParam{64, 64, 0.2f, true, false, "64x64_a02_noCtx"}),
                         [](const ::testing::TestParamInfo<AddWeightedParam> &info) { return info.param.name; });
