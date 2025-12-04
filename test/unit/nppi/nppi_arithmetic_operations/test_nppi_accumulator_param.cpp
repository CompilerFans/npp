#include "npp_test_base.h"
#include <cmath>

using namespace npp_functional_test;

// Parameter structure for accumulator operations (AddProduct, AddSquare, AddWeighted)
struct AccumulatorParam {
  int width;
  int height;
  bool use_ctx;
  bool use_mask;
  std::string name() const {
    std::string result = std::to_string(width) + "x" + std::to_string(height);
    if (use_mask) result += "_Mask";
    if (use_ctx) result += "_Ctx";
    return result;
  }
};

static const std::vector<AccumulatorParam> kAccumulatorParams = {
    {32, 32, false, false}, {32, 32, true, false},
    {64, 64, false, false}, {64, 64, true, false},
    {32, 32, false, true},  {32, 32, true, true},
};

// ==================== AddProduct 8u32f C1IR ====================
class AddProduct8u32fParamTest : public NppTestBase, public ::testing::WithParamInterface<AccumulatorParam> {};

TEST_P(AddProduct8u32fParamTest, AddProduct_8u32f_C1IR) {
  const auto& p = GetParam();
  const int total = p.width * p.height;

  std::vector<Npp8u> src1Data(total), src2Data(total);
  std::vector<Npp32f> accData(total), expectedData(total);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp8u>(0), static_cast<Npp8u>(100), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp8u>(0), static_cast<Npp8u>(100), 54321);
  TestDataGenerator::generateRandom(accData, 0.0f, 1000.0f, 11111);

  std::vector<Npp8u> maskData;
  if (p.use_mask) {
    maskData.resize(total);
    for (int i = 0; i < total; i++) {
      maskData[i] = (i % 3 == 0) ? 255 : 0;
    }
  }

  for (int i = 0; i < total; i++) {
    if (p.use_mask && maskData[i] == 0) {
      expectedData[i] = accData[i];
    } else {
      expectedData[i] = accData[i] + static_cast<Npp32f>(src1Data[i]) * static_cast<Npp32f>(src2Data[i]);
    }
  }

  NppImageMemory<Npp8u> src1(p.width, p.height);
  NppImageMemory<Npp8u> src2(p.width, p.height);
  NppImageMemory<Npp32f> acc(p.width, p.height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);
  acc.copyFromHost(accData);

  NppiSize roi = {p.width, p.height};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status;

  if (p.use_mask) {
    NppImageMemory<Npp8u> mask(p.width, p.height);
    mask.copyFromHost(maskData);
    status = p.use_ctx
        ? nppiAddProduct_8u32f_C1IMR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(),
                                          mask.get(), mask.step(), acc.get(), acc.step(), roi, ctx)
        : nppiAddProduct_8u32f_C1IMR(src1.get(), src1.step(), src2.get(), src2.step(),
                                      mask.get(), mask.step(), acc.get(), acc.step(), roi);
  } else {
    status = p.use_ctx
        ? nppiAddProduct_8u32f_C1IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(),
                                         acc.get(), acc.step(), roi, ctx)
        : nppiAddProduct_8u32f_C1IR(src1.get(), src1.step(), src2.get(), src2.step(),
                                     acc.get(), acc.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32f> result(total);
  acc.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expectedData, 1e-3f));
}

INSTANTIATE_TEST_SUITE_P(AddProduct8u32f, AddProduct8u32fParamTest, ::testing::ValuesIn(kAccumulatorParams),
    [](const auto& info) { return info.param.name(); });

// ==================== AddProduct 16u32f C1IR ====================
class AddProduct16u32fParamTest : public NppTestBase, public ::testing::WithParamInterface<AccumulatorParam> {};

TEST_P(AddProduct16u32fParamTest, AddProduct_16u32f_C1IR) {
  const auto& p = GetParam();
  if (p.use_mask) return;  // 16u version doesn't have mask variant
  const int total = p.width * p.height;

  std::vector<Npp16u> src1Data(total), src2Data(total);
  std::vector<Npp32f> accData(total), expectedData(total);
  TestDataGenerator::generateRandom(src1Data, static_cast<Npp16u>(0), static_cast<Npp16u>(1000), 12345);
  TestDataGenerator::generateRandom(src2Data, static_cast<Npp16u>(0), static_cast<Npp16u>(1000), 54321);
  TestDataGenerator::generateRandom(accData, 0.0f, 100000.0f, 11111);

  for (int i = 0; i < total; i++) {
    expectedData[i] = accData[i] + static_cast<Npp32f>(src1Data[i]) * static_cast<Npp32f>(src2Data[i]);
  }

  NppImageMemory<Npp16u> src1(p.width, p.height);
  NppImageMemory<Npp16u> src2(p.width, p.height);
  NppImageMemory<Npp32f> acc(p.width, p.height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);
  acc.copyFromHost(accData);

  NppiSize roi = {p.width, p.height};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status = p.use_ctx
      ? nppiAddProduct_16u32f_C1IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(),
                                        acc.get(), acc.step(), roi, ctx)
      : nppiAddProduct_16u32f_C1IR(src1.get(), src1.step(), src2.get(), src2.step(),
                                    acc.get(), acc.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32f> result(total);
  acc.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expectedData, 1e-1f));
}

INSTANTIATE_TEST_SUITE_P(AddProduct16u32f, AddProduct16u32fParamTest, ::testing::ValuesIn(kAccumulatorParams),
    [](const auto& info) { return info.param.name(); });

// ==================== AddSquare 8u32f C1IR ====================
class AddSquare8u32fParamTest : public NppTestBase, public ::testing::WithParamInterface<AccumulatorParam> {};

TEST_P(AddSquare8u32fParamTest, AddSquare_8u32f_C1IR) {
  const auto& p = GetParam();
  const int total = p.width * p.height;

  std::vector<Npp8u> srcData(total);
  std::vector<Npp32f> accData(total), expectedData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(100), 12345);
  TestDataGenerator::generateRandom(accData, 0.0f, 1000.0f, 11111);

  std::vector<Npp8u> maskData;
  if (p.use_mask) {
    maskData.resize(total);
    for (int i = 0; i < total; i++) {
      maskData[i] = (i % 3 == 0) ? 255 : 0;
    }
  }

  for (int i = 0; i < total; i++) {
    if (p.use_mask && maskData[i] == 0) {
      expectedData[i] = accData[i];
    } else {
      expectedData[i] = accData[i] + static_cast<Npp32f>(srcData[i]) * static_cast<Npp32f>(srcData[i]);
    }
  }

  NppImageMemory<Npp8u> src(p.width, p.height);
  NppImageMemory<Npp32f> acc(p.width, p.height);
  src.copyFromHost(srcData);
  acc.copyFromHost(accData);

  NppiSize roi = {p.width, p.height};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status;

  if (p.use_mask) {
    NppImageMemory<Npp8u> mask(p.width, p.height);
    mask.copyFromHost(maskData);
    status = p.use_ctx
        ? nppiAddSquare_8u32f_C1IMR_Ctx(src.get(), src.step(), mask.get(), mask.step(),
                                         acc.get(), acc.step(), roi, ctx)
        : nppiAddSquare_8u32f_C1IMR(src.get(), src.step(), mask.get(), mask.step(),
                                     acc.get(), acc.step(), roi);
  } else {
    status = p.use_ctx
        ? nppiAddSquare_8u32f_C1IR_Ctx(src.get(), src.step(), acc.get(), acc.step(), roi, ctx)
        : nppiAddSquare_8u32f_C1IR(src.get(), src.step(), acc.get(), acc.step(), roi);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32f> result(total);
  acc.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expectedData, 1e-3f));
}

INSTANTIATE_TEST_SUITE_P(AddSquare8u32f, AddSquare8u32fParamTest, ::testing::ValuesIn(kAccumulatorParams),
    [](const auto& info) { return info.param.name(); });

// ==================== AddSquare 16u32f C1IR ====================
class AddSquare16u32fParamTest : public NppTestBase, public ::testing::WithParamInterface<AccumulatorParam> {};

TEST_P(AddSquare16u32fParamTest, AddSquare_16u32f_C1IR) {
  const auto& p = GetParam();
  if (p.use_mask) return;  // 16u version doesn't have mask variant
  const int total = p.width * p.height;

  std::vector<Npp16u> srcData(total);
  std::vector<Npp32f> accData(total), expectedData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(1000), 12345);
  TestDataGenerator::generateRandom(accData, 0.0f, 100000.0f, 11111);

  for (int i = 0; i < total; i++) {
    expectedData[i] = accData[i] + static_cast<Npp32f>(srcData[i]) * static_cast<Npp32f>(srcData[i]);
  }

  NppImageMemory<Npp16u> src(p.width, p.height);
  NppImageMemory<Npp32f> acc(p.width, p.height);
  src.copyFromHost(srcData);
  acc.copyFromHost(accData);

  NppiSize roi = {p.width, p.height};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status = p.use_ctx
      ? nppiAddSquare_16u32f_C1IR_Ctx(src.get(), src.step(), acc.get(), acc.step(), roi, ctx)
      : nppiAddSquare_16u32f_C1IR(src.get(), src.step(), acc.get(), acc.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32f> result(total);
  acc.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expectedData, 1e-1f));
}

INSTANTIATE_TEST_SUITE_P(AddSquare16u32f, AddSquare16u32fParamTest, ::testing::ValuesIn(kAccumulatorParams),
    [](const auto& info) { return info.param.name(); });

// ==================== AddWeighted 8u32f C1IR ====================
struct AddWeightedParam {
  int width;
  int height;
  Npp32f alpha;
  bool use_ctx;
  bool use_mask;
  std::string name() const {
    std::string result = std::to_string(width) + "x" + std::to_string(height);
    result += "_alpha" + std::to_string(static_cast<int>(alpha * 100));
    if (use_mask) result += "_Mask";
    if (use_ctx) result += "_Ctx";
    return result;
  }
};

static const std::vector<AddWeightedParam> kAddWeightedParams = {
    {32, 32, 0.5f, false, false}, {32, 32, 0.5f, true, false},
    {32, 32, 0.25f, false, false}, {32, 32, 0.75f, true, false},
    {32, 32, 0.5f, false, true}, {32, 32, 0.5f, true, true},
};

class AddWeighted8u32fParamTest : public NppTestBase, public ::testing::WithParamInterface<AddWeightedParam> {};

TEST_P(AddWeighted8u32fParamTest, AddWeighted_8u32f_C1IR) {
  const auto& p = GetParam();
  const int total = p.width * p.height;

  std::vector<Npp8u> srcData(total);
  std::vector<Npp32f> accData(total), expectedData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp8u>(0), static_cast<Npp8u>(255), 12345);
  TestDataGenerator::generateRandom(accData, 0.0f, 255.0f, 11111);

  std::vector<Npp8u> maskData;
  if (p.use_mask) {
    maskData.resize(total);
    for (int i = 0; i < total; i++) {
      maskData[i] = (i % 3 == 0) ? 255 : 0;
    }
  }

  for (int i = 0; i < total; i++) {
    if (p.use_mask && maskData[i] == 0) {
      expectedData[i] = accData[i];
    } else {
      expectedData[i] = accData[i] * (1.0f - p.alpha) + static_cast<Npp32f>(srcData[i]) * p.alpha;
    }
  }

  NppImageMemory<Npp8u> src(p.width, p.height);
  NppImageMemory<Npp32f> acc(p.width, p.height);
  src.copyFromHost(srcData);
  acc.copyFromHost(accData);

  NppiSize roi = {p.width, p.height};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status;

  if (p.use_mask) {
    NppImageMemory<Npp8u> mask(p.width, p.height);
    mask.copyFromHost(maskData);
    status = p.use_ctx
        ? nppiAddWeighted_8u32f_C1IMR_Ctx(src.get(), src.step(), mask.get(), mask.step(),
                                           acc.get(), acc.step(), roi, p.alpha, ctx)
        : nppiAddWeighted_8u32f_C1IMR(src.get(), src.step(), mask.get(), mask.step(),
                                       acc.get(), acc.step(), roi, p.alpha);
  } else {
    status = p.use_ctx
        ? nppiAddWeighted_8u32f_C1IR_Ctx(src.get(), src.step(), acc.get(), acc.step(), roi, p.alpha, ctx)
        : nppiAddWeighted_8u32f_C1IR(src.get(), src.step(), acc.get(), acc.step(), roi, p.alpha);
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32f> result(total);
  acc.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expectedData, 1e-3f));
}

INSTANTIATE_TEST_SUITE_P(AddWeighted8u32f, AddWeighted8u32fParamTest, ::testing::ValuesIn(kAddWeightedParams),
    [](const auto& info) { return info.param.name(); });

// ==================== AddWeighted 16u32f C1IR ====================
class AddWeighted16u32fParamTest : public NppTestBase, public ::testing::WithParamInterface<AddWeightedParam> {};

TEST_P(AddWeighted16u32fParamTest, AddWeighted_16u32f_C1IR) {
  const auto& p = GetParam();
  if (p.use_mask) return;  // 16u version doesn't have mask variant
  const int total = p.width * p.height;

  std::vector<Npp16u> srcData(total);
  std::vector<Npp32f> accData(total), expectedData(total);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16u>(0), static_cast<Npp16u>(65535), 12345);
  TestDataGenerator::generateRandom(accData, 0.0f, 65535.0f, 11111);

  for (int i = 0; i < total; i++) {
    expectedData[i] = accData[i] * (1.0f - p.alpha) + static_cast<Npp32f>(srcData[i]) * p.alpha;
  }

  NppImageMemory<Npp16u> src(p.width, p.height);
  NppImageMemory<Npp32f> acc(p.width, p.height);
  src.copyFromHost(srcData);
  acc.copyFromHost(accData);

  NppiSize roi = {p.width, p.height};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status = p.use_ctx
      ? nppiAddWeighted_16u32f_C1IR_Ctx(src.get(), src.step(), acc.get(), acc.step(), roi, p.alpha, ctx)
      : nppiAddWeighted_16u32f_C1IR(src.get(), src.step(), acc.get(), acc.step(), roi, p.alpha);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32f> result(total);
  acc.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expectedData, 1e-1f));
}

INSTANTIATE_TEST_SUITE_P(AddWeighted16u32f, AddWeighted16u32fParamTest, ::testing::ValuesIn(kAddWeightedParams),
    [](const auto& info) { return info.param.name(); });

// ==================== AddWeighted 32f C1IR ====================
class AddWeighted32fParamTest : public NppTestBase, public ::testing::WithParamInterface<AddWeightedParam> {};

TEST_P(AddWeighted32fParamTest, AddWeighted_32f_C1IR) {
  const auto& p = GetParam();
  if (p.use_mask) return;  // 32f version doesn't have mask variant
  const int total = p.width * p.height;

  std::vector<Npp32f> srcData(total);
  std::vector<Npp32f> accData(total), expectedData(total);
  TestDataGenerator::generateRandom(srcData, 0.0f, 1.0f, 12345);
  TestDataGenerator::generateRandom(accData, 0.0f, 1.0f, 11111);

  for (int i = 0; i < total; i++) {
    expectedData[i] = accData[i] * (1.0f - p.alpha) + srcData[i] * p.alpha;
  }

  NppImageMemory<Npp32f> src(p.width, p.height);
  NppImageMemory<Npp32f> acc(p.width, p.height);
  src.copyFromHost(srcData);
  acc.copyFromHost(accData);

  NppiSize roi = {p.width, p.height};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status = p.use_ctx
      ? nppiAddWeighted_32f_C1IR_Ctx(src.get(), src.step(), acc.get(), acc.step(), roi, p.alpha, ctx)
      : nppiAddWeighted_32f_C1IR(src.get(), src.step(), acc.get(), acc.step(), roi, p.alpha);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32f> result(total);
  acc.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expectedData, 1e-5f));
}

INSTANTIATE_TEST_SUITE_P(AddWeighted32f, AddWeighted32fParamTest, ::testing::ValuesIn(kAddWeightedParams),
    [](const auto& info) { return info.param.name(); });
