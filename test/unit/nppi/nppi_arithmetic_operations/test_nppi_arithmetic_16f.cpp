#include "npp_test_base.h"
#include <functional>

using namespace npp_functional_test;

// Parameter structure for 16f binary operation tests
struct Arith16fParam {
  int channels;
  bool use_ctx;
  bool in_place;
  std::string name() const {
    std::string result = "C" + std::to_string(channels);
    result += in_place ? "IR" : "R";
    if (use_ctx)
      result += "_Ctx";
    return result;
  }
};

// Base test class for 16f binary operations
class Arith16fParamTest : public NppTestBase, public ::testing::WithParamInterface<Arith16fParam> {
protected:
  static constexpr int kWidth = 32;
  static constexpr int kHeight = 32;

  template <typename BinaryOp>
  void runBinaryTest(
      BinaryOp op, std::function<NppStatus(const Npp16f *, int, const Npp16f *, int, Npp16f *, int, NppiSize)> nppR,
      std::function<NppStatus(const Npp16f *, int, const Npp16f *, int, Npp16f *, int, NppiSize, NppStreamContext)>
          nppR_Ctx,
      std::function<NppStatus(const Npp16f *, int, Npp16f *, int, NppiSize)> nppIR,
      std::function<NppStatus(const Npp16f *, int, Npp16f *, int, NppiSize, NppStreamContext)> nppIR_Ctx) {
    const auto &p = GetParam();
    const int total = kWidth * kHeight * p.channels;

    std::vector<Npp16f> src1Data(total), src2Data(total), expected(total);
    TestDataGenerator::generateRandom16f(src1Data, -20.0f, 20.0f, 12345);
    TestDataGenerator::generateRandom16f(src2Data, -20.0f, 20.0f, 54321);

    for (int i = 0; i < total; i++) {
      float v1 = npp16f_to_float_host(src1Data[i]);
      float v2 = npp16f_to_float_host(src2Data[i]);
      expected[i] = float_to_npp16f_host(op(v1, v2));
    }

    NppImageMemory<Npp16f> src1(kWidth * p.channels, kHeight);
    NppImageMemory<Npp16f> src2(kWidth * p.channels, kHeight);
    src1.copyFromHost(src1Data);
    src2.copyFromHost(src2Data);

    NppiSize roi = {kWidth, kHeight};
    NppStreamContext ctx;
    nppGetStreamContext(&ctx);
    NppStatus status;
    std::vector<Npp16f> result(total);

    if (p.in_place) {
      status = p.use_ctx ? nppIR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx)
                         : nppIR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
      ASSERT_EQ(status, NPP_NO_ERROR);
      src2.copyToHost(result);
    } else {
      NppImageMemory<Npp16f> dst(kWidth * p.channels, kHeight);
      status = p.use_ctx ? nppR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx)
                         : nppR(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
      ASSERT_EQ(status, NPP_NO_ERROR);
      dst.copyToHost(result);
    }

    EXPECT_TRUE(ResultValidator::arraysEqual16f(result, expected, 1e-2f));
  }

  template <typename UnaryConstOp>
  void runConstTest(
      UnaryConstOp op, const Npp32f *constants,
      std::function<NppStatus(const Npp16f *, int, Npp32f, Npp16f *, int, NppiSize)> nppC1R,
      std::function<NppStatus(const Npp16f *, int, Npp32f, Npp16f *, int, NppiSize, NppStreamContext)> nppC1R_Ctx,
      std::function<NppStatus(Npp32f, Npp16f *, int, NppiSize)> nppC1IR,
      std::function<NppStatus(Npp32f, Npp16f *, int, NppiSize, NppStreamContext)> nppC1IR_Ctx,
      std::function<NppStatus(const Npp16f *, int, const Npp32f *, Npp16f *, int, NppiSize)> nppC3R,
      std::function<NppStatus(const Npp16f *, int, const Npp32f *, Npp16f *, int, NppiSize, NppStreamContext)>
          nppC3R_Ctx,
      std::function<NppStatus(const Npp32f *, Npp16f *, int, NppiSize)> nppC3IR,
      std::function<NppStatus(const Npp32f *, Npp16f *, int, NppiSize, NppStreamContext)> nppC3IR_Ctx,
      std::function<NppStatus(const Npp16f *, int, const Npp32f *, Npp16f *, int, NppiSize)> nppC4R = nullptr,
      std::function<NppStatus(const Npp16f *, int, const Npp32f *, Npp16f *, int, NppiSize, NppStreamContext)>
          nppC4R_Ctx = nullptr,
      std::function<NppStatus(const Npp32f *, Npp16f *, int, NppiSize)> nppC4IR = nullptr,
      std::function<NppStatus(const Npp32f *, Npp16f *, int, NppiSize, NppStreamContext)> nppC4IR_Ctx = nullptr) {
    const auto &p = GetParam();
    const int total = kWidth * kHeight * p.channels;

    std::vector<Npp16f> srcData(total), expected(total);
    TestDataGenerator::generateRandom16f(srcData, 1.0f, 50.0f, 12345);

    for (int i = 0; i < total; i++) {
      float v = npp16f_to_float_host(srcData[i]);
      expected[i] = float_to_npp16f_host(op(v, constants[i % p.channels]));
    }

    NppImageMemory<Npp16f> src(kWidth * p.channels, kHeight);
    src.copyFromHost(srcData);

    NppiSize roi = {kWidth, kHeight};
    NppStreamContext ctx;
    nppGetStreamContext(&ctx);
    NppStatus status;
    std::vector<Npp16f> result(total);

    if (p.channels == 1) {
      if (p.in_place) {
        status = p.use_ctx ? nppC1IR_Ctx(constants[0], src.get(), src.step(), roi, ctx)
                           : nppC1IR(constants[0], src.get(), src.step(), roi);
        ASSERT_EQ(status, NPP_NO_ERROR);
        src.copyToHost(result);
      } else {
        NppImageMemory<Npp16f> dst(kWidth, kHeight);
        status = p.use_ctx ? nppC1R_Ctx(src.get(), src.step(), constants[0], dst.get(), dst.step(), roi, ctx)
                           : nppC1R(src.get(), src.step(), constants[0], dst.get(), dst.step(), roi);
        ASSERT_EQ(status, NPP_NO_ERROR);
        dst.copyToHost(result);
      }
    } else if (p.channels == 3) {
      if (p.in_place) {
        status = p.use_ctx ? nppC3IR_Ctx(constants, src.get(), src.step(), roi, ctx)
                           : nppC3IR(constants, src.get(), src.step(), roi);
        ASSERT_EQ(status, NPP_NO_ERROR);
        src.copyToHost(result);
      } else {
        NppImageMemory<Npp16f> dst(kWidth * p.channels, kHeight);
        status = p.use_ctx ? nppC3R_Ctx(src.get(), src.step(), constants, dst.get(), dst.step(), roi, ctx)
                           : nppC3R(src.get(), src.step(), constants, dst.get(), dst.step(), roi);
        ASSERT_EQ(status, NPP_NO_ERROR);
        dst.copyToHost(result);
      }
    } else {
      // C4
      if (p.in_place) {
        status = p.use_ctx ? nppC4IR_Ctx(constants, src.get(), src.step(), roi, ctx)
                           : nppC4IR(constants, src.get(), src.step(), roi);
        ASSERT_EQ(status, NPP_NO_ERROR);
        src.copyToHost(result);
      } else {
        NppImageMemory<Npp16f> dst(kWidth * p.channels, kHeight);
        status = p.use_ctx ? nppC4R_Ctx(src.get(), src.step(), constants, dst.get(), dst.step(), roi, ctx)
                           : nppC4R(src.get(), src.step(), constants, dst.get(), dst.step(), roi);
        ASSERT_EQ(status, NPP_NO_ERROR);
        dst.copyToHost(result);
      }
    }

    EXPECT_TRUE(ResultValidator::arraysEqual16f(result, expected, 1e-2f));
  }
};

// Parameter values
static const std::vector<Arith16fParam> kArith16fParams = {
    {1, false, false}, {1, true, false}, {1, false, true},  {1, true, true},  {3, false, false}, {3, true, false},
    {3, false, true},  {3, true, true},  {4, false, false}, {4, true, false}, {4, false, true},  {4, true, true},
};

// ==================== Add_16f ====================
class Add16fParamTest : public Arith16fParamTest {};

TEST_P(Add16fParamTest, Add_16f) {
  const auto &p = GetParam();
  auto addOp = [](float a, float b) { return a + b; };

  if (p.channels == 1) {
    runBinaryTest(addOp, nppiAdd_16f_C1R, nppiAdd_16f_C1R_Ctx, nppiAdd_16f_C1IR, nppiAdd_16f_C1IR_Ctx);
  } else if (p.channels == 3) {
    runBinaryTest(addOp, nppiAdd_16f_C3R, nppiAdd_16f_C3R_Ctx, nppiAdd_16f_C3IR, nppiAdd_16f_C3IR_Ctx);
  } else {
    runBinaryTest(addOp, nppiAdd_16f_C4R, nppiAdd_16f_C4R_Ctx, nppiAdd_16f_C4IR, nppiAdd_16f_C4IR_Ctx);
  }
}

INSTANTIATE_TEST_SUITE_P(Add16f, Add16fParamTest, ::testing::ValuesIn(kArith16fParams),
                         [](const auto &info) { return info.param.name(); });

// ==================== Sub_16f ====================
class Sub16fParamTest : public Arith16fParamTest {};

TEST_P(Sub16fParamTest, Sub_16f) {
  const auto &p = GetParam();
  // NPP Sub: dst = src2 - src1
  auto subOp = [](float a, float b) { return b - a; };

  if (p.channels == 1) {
    runBinaryTest(subOp, nppiSub_16f_C1R, nppiSub_16f_C1R_Ctx, nppiSub_16f_C1IR, nppiSub_16f_C1IR_Ctx);
  } else if (p.channels == 3) {
    runBinaryTest(subOp, nppiSub_16f_C3R, nppiSub_16f_C3R_Ctx, nppiSub_16f_C3IR, nppiSub_16f_C3IR_Ctx);
  } else {
    runBinaryTest(subOp, nppiSub_16f_C4R, nppiSub_16f_C4R_Ctx, nppiSub_16f_C4IR, nppiSub_16f_C4IR_Ctx);
  }
}

INSTANTIATE_TEST_SUITE_P(Sub16f, Sub16fParamTest, ::testing::ValuesIn(kArith16fParams),
                         [](const auto &info) { return info.param.name(); });

// ==================== Mul_16f ====================
class Mul16fParamTest : public Arith16fParamTest {};

TEST_P(Mul16fParamTest, Mul_16f) {
  const auto &p = GetParam();
  auto mulOp = [](float a, float b) { return a * b; };

  if (p.channels == 1) {
    runBinaryTest(mulOp, nppiMul_16f_C1R, nppiMul_16f_C1R_Ctx, nppiMul_16f_C1IR, nppiMul_16f_C1IR_Ctx);
  } else if (p.channels == 3) {
    runBinaryTest(mulOp, nppiMul_16f_C3R, nppiMul_16f_C3R_Ctx, nppiMul_16f_C3IR, nppiMul_16f_C3IR_Ctx);
  } else {
    runBinaryTest(mulOp, nppiMul_16f_C4R, nppiMul_16f_C4R_Ctx, nppiMul_16f_C4IR, nppiMul_16f_C4IR_Ctx);
  }
}

INSTANTIATE_TEST_SUITE_P(Mul16f, Mul16fParamTest, ::testing::ValuesIn(kArith16fParams),
                         [](const auto &info) { return info.param.name(); });

// ==================== Div_16f ====================
class Div16fParamTest : public Arith16fParamTest {};

TEST_P(Div16fParamTest, Div_16f) {
  const auto &p = GetParam();
  // NPP Div: dst = src2 / src1
  auto divOp = [](float a, float b) { return (a != 0.0f) ? b / a : 0.0f; };

  // Use non-zero values for src1 to avoid division by zero
  const int total = kWidth * kHeight * p.channels;
  std::vector<Npp16f> src1Data(total), src2Data(total), expected(total);
  TestDataGenerator::generateRandom16f(src1Data, 1.0f, 20.0f, 12345); // Avoid zero
  TestDataGenerator::generateRandom16f(src2Data, -20.0f, 20.0f, 54321);

  for (int i = 0; i < total; i++) {
    float v1 = npp16f_to_float_host(src1Data[i]);
    float v2 = npp16f_to_float_host(src2Data[i]);
    expected[i] = float_to_npp16f_host(divOp(v1, v2));
  }

  NppImageMemory<Npp16f> src1(kWidth * p.channels, kHeight);
  NppImageMemory<Npp16f> src2(kWidth * p.channels, kHeight);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {kWidth, kHeight};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status;
  std::vector<Npp16f> result(total);

  if (p.in_place) {
    if (p.channels == 1) {
      status = p.use_ctx ? nppiDiv_16f_C1IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx)
                         : nppiDiv_16f_C1IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
    } else if (p.channels == 3) {
      status = p.use_ctx ? nppiDiv_16f_C3IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx)
                         : nppiDiv_16f_C3IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
    } else {
      status = p.use_ctx ? nppiDiv_16f_C4IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx)
                         : nppiDiv_16f_C4IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);
    src2.copyToHost(result);
  } else {
    NppImageMemory<Npp16f> dst(kWidth * p.channels, kHeight);
    if (p.channels == 1) {
      status =
          p.use_ctx
              ? nppiDiv_16f_C1R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx)
              : nppiDiv_16f_C1R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    } else if (p.channels == 3) {
      status =
          p.use_ctx
              ? nppiDiv_16f_C3R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx)
              : nppiDiv_16f_C3R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    } else {
      status =
          p.use_ctx
              ? nppiDiv_16f_C4R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx)
              : nppiDiv_16f_C4R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);
    dst.copyToHost(result);
  }

  EXPECT_TRUE(ResultValidator::arraysEqual16f(result, expected, 1e-2f));
}

INSTANTIATE_TEST_SUITE_P(Div16f, Div16fParamTest, ::testing::ValuesIn(kArith16fParams),
                         [](const auto &info) { return info.param.name(); });

// ==================== Sqr_16f ====================
class Sqr16fParamTest : public Arith16fParamTest {};

TEST_P(Sqr16fParamTest, Sqr_16f) {
  const auto &p = GetParam();
  const int total = kWidth * kHeight * p.channels;

  std::vector<Npp16f> srcData(total), expected(total);
  TestDataGenerator::generateRandom16f(srcData, -10.0f, 10.0f, 12345);

  for (int i = 0; i < total; i++) {
    float v = npp16f_to_float_host(srcData[i]);
    expected[i] = float_to_npp16f_host(v * v);
  }

  NppImageMemory<Npp16f> src(kWidth * p.channels, kHeight);
  src.copyFromHost(srcData);

  NppiSize roi = {kWidth, kHeight};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status;
  std::vector<Npp16f> result(total);

  if (p.in_place) {
    if (p.channels == 1) {
      status = p.use_ctx ? nppiSqr_16f_C1IR_Ctx(src.get(), src.step(), roi, ctx)
                         : nppiSqr_16f_C1IR(src.get(), src.step(), roi);
    } else if (p.channels == 3) {
      status = p.use_ctx ? nppiSqr_16f_C3IR_Ctx(src.get(), src.step(), roi, ctx)
                         : nppiSqr_16f_C3IR(src.get(), src.step(), roi);
    } else {
      status = p.use_ctx ? nppiSqr_16f_C4IR_Ctx(src.get(), src.step(), roi, ctx)
                         : nppiSqr_16f_C4IR(src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);
    src.copyToHost(result);
  } else {
    NppImageMemory<Npp16f> dst(kWidth * p.channels, kHeight);
    if (p.channels == 1) {
      status = p.use_ctx ? nppiSqr_16f_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx)
                         : nppiSqr_16f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);
    } else if (p.channels == 3) {
      status = p.use_ctx ? nppiSqr_16f_C3R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx)
                         : nppiSqr_16f_C3R(src.get(), src.step(), dst.get(), dst.step(), roi);
    } else {
      status = p.use_ctx ? nppiSqr_16f_C4R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx)
                         : nppiSqr_16f_C4R(src.get(), src.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);
    dst.copyToHost(result);
  }

  EXPECT_TRUE(ResultValidator::arraysEqual16f(result, expected, 1e-2f));
}

INSTANTIATE_TEST_SUITE_P(Sqr16f, Sqr16fParamTest, ::testing::ValuesIn(kArith16fParams),
                         [](const auto &info) { return info.param.name(); });

// ==================== Sqrt_16f ====================
class Sqrt16fParamTest : public Arith16fParamTest {};

TEST_P(Sqrt16fParamTest, Sqrt_16f) {
  const auto &p = GetParam();
  const int total = kWidth * kHeight * p.channels;

  std::vector<Npp16f> srcData(total), expected(total);
  TestDataGenerator::generateRandom16f(srcData, 0.1f, 100.0f, 12345); // Positive values only

  for (int i = 0; i < total; i++) {
    float v = npp16f_to_float_host(srcData[i]);
    expected[i] = float_to_npp16f_host(std::sqrt(v));
  }

  NppImageMemory<Npp16f> src(kWidth * p.channels, kHeight);
  src.copyFromHost(srcData);

  NppiSize roi = {kWidth, kHeight};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status;
  std::vector<Npp16f> result(total);

  if (p.in_place) {
    if (p.channels == 1) {
      status = p.use_ctx ? nppiSqrt_16f_C1IR_Ctx(src.get(), src.step(), roi, ctx)
                         : nppiSqrt_16f_C1IR(src.get(), src.step(), roi);
    } else if (p.channels == 3) {
      status = p.use_ctx ? nppiSqrt_16f_C3IR_Ctx(src.get(), src.step(), roi, ctx)
                         : nppiSqrt_16f_C3IR(src.get(), src.step(), roi);
    } else {
      status = p.use_ctx ? nppiSqrt_16f_C4IR_Ctx(src.get(), src.step(), roi, ctx)
                         : nppiSqrt_16f_C4IR(src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);
    src.copyToHost(result);
  } else {
    NppImageMemory<Npp16f> dst(kWidth * p.channels, kHeight);
    if (p.channels == 1) {
      status = p.use_ctx ? nppiSqrt_16f_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx)
                         : nppiSqrt_16f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);
    } else if (p.channels == 3) {
      status = p.use_ctx ? nppiSqrt_16f_C3R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx)
                         : nppiSqrt_16f_C3R(src.get(), src.step(), dst.get(), dst.step(), roi);
    } else {
      status = p.use_ctx ? nppiSqrt_16f_C4R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx)
                         : nppiSqrt_16f_C4R(src.get(), src.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);
    dst.copyToHost(result);
  }

  EXPECT_TRUE(ResultValidator::arraysEqual16f(result, expected, 1e-2f));
}

INSTANTIATE_TEST_SUITE_P(Sqrt16f, Sqrt16fParamTest, ::testing::ValuesIn(kArith16fParams),
                         [](const auto &info) { return info.param.name(); });

// ==================== AbsDiff_16f ====================
class AbsDiff16fParamTest : public Arith16fParamTest {};

TEST_P(AbsDiff16fParamTest, AbsDiff_16f) {
  const auto &p = GetParam();
  if (p.in_place)
    return; // AbsDiff has no in-place variant

  auto absDiffOp = [](float a, float b) { return std::fabs(a - b); };

  const int total = kWidth * kHeight * p.channels;
  std::vector<Npp16f> src1Data(total), src2Data(total), expected(total);
  TestDataGenerator::generateRandom16f(src1Data, -50.0f, 50.0f, 12345);
  TestDataGenerator::generateRandom16f(src2Data, -50.0f, 50.0f, 54321);

  for (int i = 0; i < total; i++) {
    float v1 = npp16f_to_float_host(src1Data[i]);
    float v2 = npp16f_to_float_host(src2Data[i]);
    expected[i] = float_to_npp16f_host(absDiffOp(v1, v2));
  }

  NppImageMemory<Npp16f> src1(kWidth * p.channels, kHeight);
  NppImageMemory<Npp16f> src2(kWidth * p.channels, kHeight);
  NppImageMemory<Npp16f> dst(kWidth * p.channels, kHeight);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);

  NppiSize roi = {kWidth, kHeight};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status;

  if (p.channels == 1) {
    status =
        p.use_ctx
            ? nppiAbsDiff_16f_C1R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx)
            : nppiAbsDiff_16f_C1R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
  } else {
    return; // AbsDiff_16f only has C1R variant
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp16f> result(total);
  dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual16f(result, expected, 1e-2f));
}

// Only test C1R for AbsDiff_16f (no C3/C4 variants)
static const std::vector<Arith16fParam> kAbsDiff16fParams = {
    {1, false, false},
    {1, true, false},
};

INSTANTIATE_TEST_SUITE_P(AbsDiff16f, AbsDiff16fParamTest, ::testing::ValuesIn(kAbsDiff16fParams),
                         [](const auto &info) { return info.param.name(); });

// ==================== Ln_16f ====================
// C1 and C3 only
static const std::vector<Arith16fParam> kLn16fParams = {
    {1, false, false}, {1, true, false}, {1, false, true}, {1, true, true},
    {3, false, false}, {3, true, false}, {3, false, true}, {3, true, true},
};

class Ln16fParamTest : public Arith16fParamTest {};

TEST_P(Ln16fParamTest, Ln_16f) {
  const auto &p = GetParam();
  const int total = kWidth * kHeight * p.channels;

  std::vector<Npp16f> srcData(total), expected(total);
  TestDataGenerator::generateRandom16f(srcData, 0.1f, 100.0f, 12345); // Positive values only

  for (int i = 0; i < total; i++) {
    float v = npp16f_to_float_host(srcData[i]);
    expected[i] = float_to_npp16f_host(std::log(v));
  }

  NppImageMemory<Npp16f> src(kWidth * p.channels, kHeight);
  src.copyFromHost(srcData);

  NppiSize roi = {kWidth, kHeight};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status;
  std::vector<Npp16f> result(total);

  if (p.in_place) {
    if (p.channels == 1) {
      status = p.use_ctx ? nppiLn_16f_C1IR_Ctx(src.get(), src.step(), roi, ctx)
                         : nppiLn_16f_C1IR(src.get(), src.step(), roi);
    } else {
      status = p.use_ctx ? nppiLn_16f_C3IR_Ctx(src.get(), src.step(), roi, ctx)
                         : nppiLn_16f_C3IR(src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);
    src.copyToHost(result);
  } else {
    NppImageMemory<Npp16f> dst(kWidth * p.channels, kHeight);
    if (p.channels == 1) {
      status = p.use_ctx ? nppiLn_16f_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx)
                         : nppiLn_16f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);
    } else {
      status = p.use_ctx ? nppiLn_16f_C3R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx)
                         : nppiLn_16f_C3R(src.get(), src.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);
    dst.copyToHost(result);
  }

  EXPECT_TRUE(ResultValidator::arraysEqual16f(result, expected, 1e-2f));
}

INSTANTIATE_TEST_SUITE_P(Ln16f, Ln16fParamTest, ::testing::ValuesIn(kLn16fParams),
                         [](const auto &info) { return info.param.name(); });

// ==================== AddProduct_16f_C1IR ====================
class AddProduct16fTest : public NppTestBase {};

TEST_F(AddProduct16fTest, AddProduct_16f_C1IR_BasicOperation) {
  const int width = 32;
  const int height = 32;
  const int total = width * height;

  std::vector<Npp16f> src1Data(total), src2Data(total), srcDstData(total), expected(total);
  TestDataGenerator::generateRandom16f(src1Data, -5.0f, 5.0f, 12345);
  TestDataGenerator::generateRandom16f(src2Data, -5.0f, 5.0f, 54321);
  TestDataGenerator::generateRandom16f(srcDstData, -10.0f, 10.0f, 11111);

  for (int i = 0; i < total; i++) {
    float v1 = npp16f_to_float_host(src1Data[i]);
    float v2 = npp16f_to_float_host(src2Data[i]);
    float acc = npp16f_to_float_host(srcDstData[i]);
    expected[i] = float_to_npp16f_host(acc + v1 * v2);
  }

  NppImageMemory<Npp16f> src1(width, height);
  NppImageMemory<Npp16f> src2(width, height);
  NppImageMemory<Npp16f> srcDst(width, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStatus status =
      nppiAddProduct_16f_C1IR(src1.get(), src1.step(), src2.get(), src2.step(), srcDst.get(), srcDst.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp16f> result(total);
  srcDst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual16f(result, expected, 1e-1f));
}

TEST_F(AddProduct16fTest, AddProduct_16f_C1IR_Ctx_BasicOperation) {
  const int width = 32;
  const int height = 32;
  const int total = width * height;

  std::vector<Npp16f> src1Data(total), src2Data(total), srcDstData(total), expected(total);
  TestDataGenerator::generateRandom16f(src1Data, -5.0f, 5.0f, 12345);
  TestDataGenerator::generateRandom16f(src2Data, -5.0f, 5.0f, 54321);
  TestDataGenerator::generateRandom16f(srcDstData, -10.0f, 10.0f, 11111);

  for (int i = 0; i < total; i++) {
    float v1 = npp16f_to_float_host(src1Data[i]);
    float v2 = npp16f_to_float_host(src2Data[i]);
    float acc = npp16f_to_float_host(srcDstData[i]);
    expected[i] = float_to_npp16f_host(acc + v1 * v2);
  }

  NppImageMemory<Npp16f> src1(width, height);
  NppImageMemory<Npp16f> src2(width, height);
  NppImageMemory<Npp16f> srcDst(width, height);
  src1.copyFromHost(src1Data);
  src2.copyFromHost(src2Data);
  srcDst.copyFromHost(srcDstData);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status = nppiAddProduct_16f_C1IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), srcDst.get(),
                                                 srcDst.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp16f> result(total);
  srcDst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual16f(result, expected, 1e-1f));
}

// ==================== Abs_16f ====================
class Abs16fParamTest : public Arith16fParamTest {};

TEST_P(Abs16fParamTest, Abs_16f) {
  const auto &p = GetParam();
  const int total = kWidth * kHeight * p.channels;

  std::vector<Npp16f> srcData(total), expected(total);
  TestDataGenerator::generateRandom16f(srcData, -50.0f, 50.0f, 12345);

  for (int i = 0; i < total; i++) {
    float v = npp16f_to_float_host(srcData[i]);
    expected[i] = float_to_npp16f_host(std::fabs(v));
  }

  NppImageMemory<Npp16f> src(kWidth * p.channels, kHeight);
  src.copyFromHost(srcData);

  NppiSize roi = {kWidth, kHeight};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status;
  std::vector<Npp16f> result(total);

  if (p.in_place) {
    if (p.channels == 1) {
      status = p.use_ctx ? nppiAbs_16f_C1IR_Ctx(src.get(), src.step(), roi, ctx)
                         : nppiAbs_16f_C1IR(src.get(), src.step(), roi);
    } else if (p.channels == 3) {
      status = p.use_ctx ? nppiAbs_16f_C3IR_Ctx(src.get(), src.step(), roi, ctx)
                         : nppiAbs_16f_C3IR(src.get(), src.step(), roi);
    } else {
      status = p.use_ctx ? nppiAbs_16f_C4IR_Ctx(src.get(), src.step(), roi, ctx)
                         : nppiAbs_16f_C4IR(src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);
    src.copyToHost(result);
  } else {
    NppImageMemory<Npp16f> dst(kWidth * p.channels, kHeight);
    if (p.channels == 1) {
      status = p.use_ctx ? nppiAbs_16f_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx)
                         : nppiAbs_16f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);
    } else if (p.channels == 3) {
      status = p.use_ctx ? nppiAbs_16f_C3R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx)
                         : nppiAbs_16f_C3R(src.get(), src.step(), dst.get(), dst.step(), roi);
    } else {
      status = p.use_ctx ? nppiAbs_16f_C4R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx)
                         : nppiAbs_16f_C4R(src.get(), src.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);
    dst.copyToHost(result);
  }

  EXPECT_TRUE(ResultValidator::arraysEqual16f(result, expected, 1e-2f));
}

INSTANTIATE_TEST_SUITE_P(Abs16f, Abs16fParamTest, ::testing::ValuesIn(kArith16fParams),
                         [](const auto &info) { return info.param.name(); });

// ==================== AddC_16f ====================
class AddC16fParamTest : public Arith16fParamTest {};

TEST_P(AddC16fParamTest, AddC_16f) {
  const Npp32f constants[4] = {2.5f, 3.5f, 4.5f, 5.5f};
  auto addCOp = [](float v, float c) { return v + c; };

  runConstTest(addCOp, constants, nppiAddC_16f_C1R, nppiAddC_16f_C1R_Ctx, nppiAddC_16f_C1IR, nppiAddC_16f_C1IR_Ctx,
               nppiAddC_16f_C3R, nppiAddC_16f_C3R_Ctx, nppiAddC_16f_C3IR, nppiAddC_16f_C3IR_Ctx, nppiAddC_16f_C4R,
               nppiAddC_16f_C4R_Ctx, nppiAddC_16f_C4IR, nppiAddC_16f_C4IR_Ctx);
}

INSTANTIATE_TEST_SUITE_P(AddC16f, AddC16fParamTest, ::testing::ValuesIn(kArith16fParams),
                         [](const auto &info) { return info.param.name(); });

// ==================== SubC_16f ====================
class SubC16fParamTest : public Arith16fParamTest {};

TEST_P(SubC16fParamTest, SubC_16f) {
  const Npp32f constants[4] = {2.5f, 3.5f, 4.5f, 5.5f};
  auto subCOp = [](float v, float c) { return v - c; };

  runConstTest(subCOp, constants, nppiSubC_16f_C1R, nppiSubC_16f_C1R_Ctx, nppiSubC_16f_C1IR, nppiSubC_16f_C1IR_Ctx,
               nppiSubC_16f_C3R, nppiSubC_16f_C3R_Ctx, nppiSubC_16f_C3IR, nppiSubC_16f_C3IR_Ctx, nppiSubC_16f_C4R,
               nppiSubC_16f_C4R_Ctx, nppiSubC_16f_C4IR, nppiSubC_16f_C4IR_Ctx);
}

INSTANTIATE_TEST_SUITE_P(SubC16f, SubC16fParamTest, ::testing::ValuesIn(kArith16fParams),
                         [](const auto &info) { return info.param.name(); });

// ==================== MulC_16f ====================
class MulC16fParamTest : public Arith16fParamTest {};

TEST_P(MulC16fParamTest, MulC_16f) {
  const Npp32f constants[4] = {2.0f, 1.5f, 0.5f, 3.0f};
  auto mulCOp = [](float v, float c) { return v * c; };

  runConstTest(mulCOp, constants, nppiMulC_16f_C1R, nppiMulC_16f_C1R_Ctx, nppiMulC_16f_C1IR, nppiMulC_16f_C1IR_Ctx,
               nppiMulC_16f_C3R, nppiMulC_16f_C3R_Ctx, nppiMulC_16f_C3IR, nppiMulC_16f_C3IR_Ctx, nppiMulC_16f_C4R,
               nppiMulC_16f_C4R_Ctx, nppiMulC_16f_C4IR, nppiMulC_16f_C4IR_Ctx);
}

INSTANTIATE_TEST_SUITE_P(MulC16f, MulC16fParamTest, ::testing::ValuesIn(kArith16fParams),
                         [](const auto &info) { return info.param.name(); });

// ==================== DivC_16f ====================
class DivC16fParamTest : public Arith16fParamTest {};

TEST_P(DivC16fParamTest, DivC_16f) {
  const Npp32f constants[4] = {2.0f, 2.5f, 3.0f, 4.0f};
  auto divCOp = [](float v, float c) { return v / c; };

  runConstTest(divCOp, constants, nppiDivC_16f_C1R, nppiDivC_16f_C1R_Ctx, nppiDivC_16f_C1IR, nppiDivC_16f_C1IR_Ctx,
               nppiDivC_16f_C3R, nppiDivC_16f_C3R_Ctx, nppiDivC_16f_C3IR, nppiDivC_16f_C3IR_Ctx, nppiDivC_16f_C4R,
               nppiDivC_16f_C4R_Ctx, nppiDivC_16f_C4IR, nppiDivC_16f_C4IR_Ctx);
}

INSTANTIATE_TEST_SUITE_P(DivC16f, DivC16fParamTest, ::testing::ValuesIn(kArith16fParams),
                         [](const auto &info) { return info.param.name(); });

// ==================== DeviceC operations ====================
// These operations take device pointers for constants instead of host values

class DeviceC16fParamTest : public Arith16fParamTest {
protected:
  template <typename OpFunc>
  void runDeviceCTest(
      OpFunc op, const Npp32f *hostConstants,
      std::function<NppStatus(const Npp16f *, int, const Npp32f *, Npp16f *, int, NppiSize, NppStreamContext)> nppC1R,
      std::function<NppStatus(const Npp32f *, Npp16f *, int, NppiSize, NppStreamContext)> nppC1IR,
      std::function<NppStatus(const Npp16f *, int, const Npp32f *, Npp16f *, int, NppiSize, NppStreamContext)> nppC3R,
      std::function<NppStatus(const Npp32f *, Npp16f *, int, NppiSize, NppStreamContext)> nppC3IR,
      std::function<NppStatus(const Npp16f *, int, const Npp32f *, Npp16f *, int, NppiSize, NppStreamContext)> nppC4R,
      std::function<NppStatus(const Npp32f *, Npp16f *, int, NppiSize, NppStreamContext)> nppC4IR) {
    const auto &p = GetParam();
    const int total = kWidth * kHeight * p.channels;

    std::vector<Npp16f> srcData(total), expected(total);
    TestDataGenerator::generateRandom16f(srcData, 1.0f, 50.0f, 12345);

    for (int i = 0; i < total; i++) {
      float v = npp16f_to_float_host(srcData[i]);
      expected[i] = float_to_npp16f_host(op(v, hostConstants[i % p.channels]));
    }

    NppImageMemory<Npp16f> src(kWidth * p.channels, kHeight);
    src.copyFromHost(srcData);

    // Allocate device constant
    Npp32f *d_constants;
    cudaMalloc(&d_constants, p.channels * sizeof(Npp32f));
    cudaMemcpy(d_constants, hostConstants, p.channels * sizeof(Npp32f), cudaMemcpyHostToDevice);

    NppiSize roi = {kWidth, kHeight};
    NppStreamContext ctx;
    nppGetStreamContext(&ctx);
    NppStatus status;
    std::vector<Npp16f> result(total);

    if (p.in_place) {
      if (p.channels == 1) {
        status = nppC1IR(d_constants, src.get(), src.step(), roi, ctx);
      } else if (p.channels == 3) {
        status = nppC3IR(d_constants, src.get(), src.step(), roi, ctx);
      } else {
        status = nppC4IR(d_constants, src.get(), src.step(), roi, ctx);
      }
      ASSERT_EQ(status, NPP_NO_ERROR);
      src.copyToHost(result);
    } else {
      NppImageMemory<Npp16f> dst(kWidth * p.channels, kHeight);
      if (p.channels == 1) {
        status = nppC1R(src.get(), src.step(), d_constants, dst.get(), dst.step(), roi, ctx);
      } else if (p.channels == 3) {
        status = nppC3R(src.get(), src.step(), d_constants, dst.get(), dst.step(), roi, ctx);
      } else {
        status = nppC4R(src.get(), src.step(), d_constants, dst.get(), dst.step(), roi, ctx);
      }
      ASSERT_EQ(status, NPP_NO_ERROR);
      dst.copyToHost(result);
    }

    cudaFree(d_constants);
    EXPECT_TRUE(ResultValidator::arraysEqual16f(result, expected, 1e-2f));
  }
};

// ==================== AddDeviceC_16f ====================
class AddDeviceC16fParamTest : public DeviceC16fParamTest {};

TEST_P(AddDeviceC16fParamTest, AddDeviceC_16f) {
  const Npp32f constants[4] = {2.5f, 3.5f, 4.5f, 5.5f};
  auto addOp = [](float v, float c) { return v + c; };

  runDeviceCTest(addOp, constants, nppiAddDeviceC_16f_C1R_Ctx, nppiAddDeviceC_16f_C1IR_Ctx, nppiAddDeviceC_16f_C3R_Ctx,
                 nppiAddDeviceC_16f_C3IR_Ctx, nppiAddDeviceC_16f_C4R_Ctx, nppiAddDeviceC_16f_C4IR_Ctx);
}

INSTANTIATE_TEST_SUITE_P(AddDeviceC16f, AddDeviceC16fParamTest, ::testing::ValuesIn(kArith16fParams),
                         [](const auto &info) { return info.param.name(); });

// ==================== SubDeviceC_16f ====================
class SubDeviceC16fParamTest : public DeviceC16fParamTest {};

TEST_P(SubDeviceC16fParamTest, SubDeviceC_16f) {
  const Npp32f constants[4] = {1.5f, 2.5f, 3.5f, 4.5f};
  auto subOp = [](float v, float c) { return v - c; };

  runDeviceCTest(subOp, constants, nppiSubDeviceC_16f_C1R_Ctx, nppiSubDeviceC_16f_C1IR_Ctx, nppiSubDeviceC_16f_C3R_Ctx,
                 nppiSubDeviceC_16f_C3IR_Ctx, nppiSubDeviceC_16f_C4R_Ctx, nppiSubDeviceC_16f_C4IR_Ctx);
}

INSTANTIATE_TEST_SUITE_P(SubDeviceC16f, SubDeviceC16fParamTest, ::testing::ValuesIn(kArith16fParams),
                         [](const auto &info) { return info.param.name(); });

// ==================== MulDeviceC_16f ====================
class MulDeviceC16fParamTest : public DeviceC16fParamTest {};

TEST_P(MulDeviceC16fParamTest, MulDeviceC_16f) {
  const Npp32f constants[4] = {2.0f, 1.5f, 0.5f, 3.0f};
  auto mulOp = [](float v, float c) { return v * c; };

  runDeviceCTest(mulOp, constants, nppiMulDeviceC_16f_C1R_Ctx, nppiMulDeviceC_16f_C1IR_Ctx, nppiMulDeviceC_16f_C3R_Ctx,
                 nppiMulDeviceC_16f_C3IR_Ctx, nppiMulDeviceC_16f_C4R_Ctx, nppiMulDeviceC_16f_C4IR_Ctx);
}

INSTANTIATE_TEST_SUITE_P(MulDeviceC16f, MulDeviceC16fParamTest, ::testing::ValuesIn(kArith16fParams),
                         [](const auto &info) { return info.param.name(); });

// ==================== DivDeviceC_16f ====================
class DivDeviceC16fParamTest : public DeviceC16fParamTest {};

TEST_P(DivDeviceC16fParamTest, DivDeviceC_16f) {
  const Npp32f constants[4] = {2.0f, 2.5f, 3.0f, 4.0f};
  auto divOp = [](float v, float c) { return v / c; };

  runDeviceCTest(divOp, constants, nppiDivDeviceC_16f_C1R_Ctx, nppiDivDeviceC_16f_C1IR_Ctx, nppiDivDeviceC_16f_C3R_Ctx,
                 nppiDivDeviceC_16f_C3IR_Ctx, nppiDivDeviceC_16f_C4R_Ctx, nppiDivDeviceC_16f_C4IR_Ctx);
}

INSTANTIATE_TEST_SUITE_P(DivDeviceC16f, DivDeviceC16fParamTest, ::testing::ValuesIn(kArith16fParams),
                         [](const auto &info) { return info.param.name(); });
