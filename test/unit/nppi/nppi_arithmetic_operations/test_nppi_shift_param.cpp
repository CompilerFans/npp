#include "npp_test_base.h"
#include <functional>

using namespace npp_functional_test;

// Parameter structure for shift operation tests
struct ShiftParam {
  int channels;
  bool use_ctx;
  bool in_place;
  std::string name() const {
    std::string result = "C" + std::to_string(channels);
    result += in_place ? "IR" : "R";
    if (use_ctx) result += "_Ctx";
    return result;
  }
};

// Base test class for shift operations
class ShiftParamTest : public NppTestBase, public ::testing::WithParamInterface<ShiftParam> {
protected:
  static constexpr int kWidth = 32;
  static constexpr int kHeight = 32;

  template <typename T>
  void runRShiftCTest(Npp32u shiftVal,
                      std::function<NppStatus(const T*, int, Npp32u, T*, int, NppiSize)> nppC1R,
                      std::function<NppStatus(const T*, int, Npp32u, T*, int, NppiSize, NppStreamContext)> nppC1R_Ctx,
                      std::function<NppStatus(Npp32u, T*, int, NppiSize)> nppC1IR,
                      std::function<NppStatus(Npp32u, T*, int, NppiSize, NppStreamContext)> nppC1IR_Ctx,
                      std::function<NppStatus(const T*, int, const Npp32u*, T*, int, NppiSize)> nppC3R,
                      std::function<NppStatus(const T*, int, const Npp32u*, T*, int, NppiSize, NppStreamContext)> nppC3R_Ctx,
                      std::function<NppStatus(const Npp32u*, T*, int, NppiSize)> nppC3IR,
                      std::function<NppStatus(const Npp32u*, T*, int, NppiSize, NppStreamContext)> nppC3IR_Ctx,
                      std::function<NppStatus(const T*, int, const Npp32u*, T*, int, NppiSize)> nppC4R,
                      std::function<NppStatus(const T*, int, const Npp32u*, T*, int, NppiSize, NppStreamContext)> nppC4R_Ctx,
                      std::function<NppStatus(const Npp32u*, T*, int, NppiSize)> nppC4IR,
                      std::function<NppStatus(const Npp32u*, T*, int, NppiSize, NppStreamContext)> nppC4IR_Ctx) {
    const auto& p = GetParam();
    const int total = kWidth * kHeight * p.channels;

    std::vector<T> srcData(total), expected(total);
    TestDataGenerator::generateRandom(srcData, static_cast<T>(0), std::numeric_limits<T>::max(), 12345);

    Npp32u shifts[4] = {shiftVal, shiftVal, shiftVal, shiftVal};
    for (int i = 0; i < total; i++) {
      expected[i] = static_cast<T>(srcData[i] >> shifts[i % p.channels]);
    }

    NppImageMemory<T> src(kWidth * p.channels, kHeight);
    src.copyFromHost(srcData);

    NppiSize roi = {kWidth, kHeight};
    NppStreamContext ctx{}; ctx.hStream = 0;
    NppStatus status;
    std::vector<T> result(total);

    if (p.in_place) {
      if (p.channels == 1) {
        status = p.use_ctx ? nppC1IR_Ctx(shiftVal, src.get(), src.step(), roi, ctx)
                           : nppC1IR(shiftVal, src.get(), src.step(), roi);
      } else if (p.channels == 3) {
        status = p.use_ctx ? nppC3IR_Ctx(shifts, src.get(), src.step(), roi, ctx)
                           : nppC3IR(shifts, src.get(), src.step(), roi);
      } else {
        status = p.use_ctx ? nppC4IR_Ctx(shifts, src.get(), src.step(), roi, ctx)
                           : nppC4IR(shifts, src.get(), src.step(), roi);
      }
      ASSERT_EQ(status, NPP_NO_ERROR);
      src.copyToHost(result);
    } else {
      NppImageMemory<T> dst(kWidth * p.channels, kHeight);
      if (p.channels == 1) {
        status = p.use_ctx ? nppC1R_Ctx(src.get(), src.step(), shiftVal, dst.get(), dst.step(), roi, ctx)
                           : nppC1R(src.get(), src.step(), shiftVal, dst.get(), dst.step(), roi);
      } else if (p.channels == 3) {
        status = p.use_ctx ? nppC3R_Ctx(src.get(), src.step(), shifts, dst.get(), dst.step(), roi, ctx)
                           : nppC3R(src.get(), src.step(), shifts, dst.get(), dst.step(), roi);
      } else {
        status = p.use_ctx ? nppC4R_Ctx(src.get(), src.step(), shifts, dst.get(), dst.step(), roi, ctx)
                           : nppC4R(src.get(), src.step(), shifts, dst.get(), dst.step(), roi);
      }
      ASSERT_EQ(status, NPP_NO_ERROR);
      dst.copyToHost(result);
    }

    EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
  }

  template <typename T>
  void runLShiftCTest(Npp32u shiftVal,
                      std::function<NppStatus(const T*, int, Npp32u, T*, int, NppiSize)> nppC1R,
                      std::function<NppStatus(const T*, int, Npp32u, T*, int, NppiSize, NppStreamContext)> nppC1R_Ctx,
                      std::function<NppStatus(Npp32u, T*, int, NppiSize)> nppC1IR,
                      std::function<NppStatus(Npp32u, T*, int, NppiSize, NppStreamContext)> nppC1IR_Ctx,
                      std::function<NppStatus(const T*, int, const Npp32u*, T*, int, NppiSize)> nppC3R,
                      std::function<NppStatus(const T*, int, const Npp32u*, T*, int, NppiSize, NppStreamContext)> nppC3R_Ctx,
                      std::function<NppStatus(const Npp32u*, T*, int, NppiSize)> nppC3IR,
                      std::function<NppStatus(const Npp32u*, T*, int, NppiSize, NppStreamContext)> nppC3IR_Ctx,
                      std::function<NppStatus(const T*, int, const Npp32u*, T*, int, NppiSize)> nppC4R,
                      std::function<NppStatus(const T*, int, const Npp32u*, T*, int, NppiSize, NppStreamContext)> nppC4R_Ctx,
                      std::function<NppStatus(const Npp32u*, T*, int, NppiSize)> nppC4IR,
                      std::function<NppStatus(const Npp32u*, T*, int, NppiSize, NppStreamContext)> nppC4IR_Ctx) {
    const auto& p = GetParam();
    const int total = kWidth * kHeight * p.channels;

    // Use smaller values to avoid overflow
    std::vector<T> srcData(total), expected(total);
    T maxVal = std::numeric_limits<T>::max() >> (shiftVal + 1);
    TestDataGenerator::generateRandom(srcData, static_cast<T>(0), maxVal, 12345);

    Npp32u shifts[4] = {shiftVal, shiftVal, shiftVal, shiftVal};
    for (int i = 0; i < total; i++) {
      expected[i] = static_cast<T>(srcData[i] << shifts[i % p.channels]);
    }

    NppImageMemory<T> src(kWidth * p.channels, kHeight);
    src.copyFromHost(srcData);

    NppiSize roi = {kWidth, kHeight};
    NppStreamContext ctx{}; ctx.hStream = 0;
    NppStatus status;
    std::vector<T> result(total);

    if (p.in_place) {
      if (p.channels == 1) {
        status = p.use_ctx ? nppC1IR_Ctx(shiftVal, src.get(), src.step(), roi, ctx)
                           : nppC1IR(shiftVal, src.get(), src.step(), roi);
      } else if (p.channels == 3) {
        status = p.use_ctx ? nppC3IR_Ctx(shifts, src.get(), src.step(), roi, ctx)
                           : nppC3IR(shifts, src.get(), src.step(), roi);
      } else {
        status = p.use_ctx ? nppC4IR_Ctx(shifts, src.get(), src.step(), roi, ctx)
                           : nppC4IR(shifts, src.get(), src.step(), roi);
      }
      ASSERT_EQ(status, NPP_NO_ERROR);
      src.copyToHost(result);
    } else {
      NppImageMemory<T> dst(kWidth * p.channels, kHeight);
      if (p.channels == 1) {
        status = p.use_ctx ? nppC1R_Ctx(src.get(), src.step(), shiftVal, dst.get(), dst.step(), roi, ctx)
                           : nppC1R(src.get(), src.step(), shiftVal, dst.get(), dst.step(), roi);
      } else if (p.channels == 3) {
        status = p.use_ctx ? nppC3R_Ctx(src.get(), src.step(), shifts, dst.get(), dst.step(), roi, ctx)
                           : nppC3R(src.get(), src.step(), shifts, dst.get(), dst.step(), roi);
      } else {
        status = p.use_ctx ? nppC4R_Ctx(src.get(), src.step(), shifts, dst.get(), dst.step(), roi, ctx)
                           : nppC4R(src.get(), src.step(), shifts, dst.get(), dst.step(), roi);
      }
      ASSERT_EQ(status, NPP_NO_ERROR);
      dst.copyToHost(result);
    }

    EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
  }
};

// Parameter values
static const std::vector<ShiftParam> kShiftParams = {
    {1, false, false}, {1, true, false}, {1, false, true}, {1, true, true},
    {3, false, false}, {3, true, false}, {3, false, true}, {3, true, true},
    {4, false, false}, {4, true, false}, {4, false, true}, {4, true, true},
};

// ==================== RShiftC_8u ====================
class RShiftC8uTest : public ShiftParamTest {};

TEST_P(RShiftC8uTest, RShiftC_8u) {
  runRShiftCTest<Npp8u>(2,
      nppiRShiftC_8u_C1R, nppiRShiftC_8u_C1R_Ctx, nppiRShiftC_8u_C1IR, nppiRShiftC_8u_C1IR_Ctx,
      nppiRShiftC_8u_C3R, nppiRShiftC_8u_C3R_Ctx, nppiRShiftC_8u_C3IR, nppiRShiftC_8u_C3IR_Ctx,
      nppiRShiftC_8u_C4R, nppiRShiftC_8u_C4R_Ctx, nppiRShiftC_8u_C4IR, nppiRShiftC_8u_C4IR_Ctx);
}

INSTANTIATE_TEST_SUITE_P(RShiftC8u, RShiftC8uTest, ::testing::ValuesIn(kShiftParams),
    [](const auto& info) { return info.param.name(); });

// ==================== RShiftC_16u ====================
class RShiftC16uTest : public ShiftParamTest {};

TEST_P(RShiftC16uTest, RShiftC_16u) {
  runRShiftCTest<Npp16u>(4,
      nppiRShiftC_16u_C1R, nppiRShiftC_16u_C1R_Ctx, nppiRShiftC_16u_C1IR, nppiRShiftC_16u_C1IR_Ctx,
      nppiRShiftC_16u_C3R, nppiRShiftC_16u_C3R_Ctx, nppiRShiftC_16u_C3IR, nppiRShiftC_16u_C3IR_Ctx,
      nppiRShiftC_16u_C4R, nppiRShiftC_16u_C4R_Ctx, nppiRShiftC_16u_C4IR, nppiRShiftC_16u_C4IR_Ctx);
}

INSTANTIATE_TEST_SUITE_P(RShiftC16u, RShiftC16uTest, ::testing::ValuesIn(kShiftParams),
    [](const auto& info) { return info.param.name(); });

// ==================== LShiftC_8u ====================
class LShiftC8uTest : public ShiftParamTest {};

TEST_P(LShiftC8uTest, LShiftC_8u) {
  runLShiftCTest<Npp8u>(2,
      nppiLShiftC_8u_C1R, nppiLShiftC_8u_C1R_Ctx, nppiLShiftC_8u_C1IR, nppiLShiftC_8u_C1IR_Ctx,
      nppiLShiftC_8u_C3R, nppiLShiftC_8u_C3R_Ctx, nppiLShiftC_8u_C3IR, nppiLShiftC_8u_C3IR_Ctx,
      nppiLShiftC_8u_C4R, nppiLShiftC_8u_C4R_Ctx, nppiLShiftC_8u_C4IR, nppiLShiftC_8u_C4IR_Ctx);
}

INSTANTIATE_TEST_SUITE_P(LShiftC8u, LShiftC8uTest, ::testing::ValuesIn(kShiftParams),
    [](const auto& info) { return info.param.name(); });

// ==================== LShiftC_16u ====================
class LShiftC16uTest : public ShiftParamTest {};

TEST_P(LShiftC16uTest, LShiftC_16u) {
  runLShiftCTest<Npp16u>(4,
      nppiLShiftC_16u_C1R, nppiLShiftC_16u_C1R_Ctx, nppiLShiftC_16u_C1IR, nppiLShiftC_16u_C1IR_Ctx,
      nppiLShiftC_16u_C3R, nppiLShiftC_16u_C3R_Ctx, nppiLShiftC_16u_C3IR, nppiLShiftC_16u_C3IR_Ctx,
      nppiLShiftC_16u_C4R, nppiLShiftC_16u_C4R_Ctx, nppiLShiftC_16u_C4IR, nppiLShiftC_16u_C4IR_Ctx);
}

INSTANTIATE_TEST_SUITE_P(LShiftC16u, LShiftC16uTest, ::testing::ValuesIn(kShiftParams),
    [](const auto& info) { return info.param.name(); });
