#include "npp_test_base.h"
#include <functional>

using namespace npp_functional_test;

// Parameter structure for MulScale operation tests
struct MulScaleParam {
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

// Base test class for MulScale operations
class MulScaleParamTest : public NppTestBase, public ::testing::WithParamInterface<MulScaleParam> {
protected:
  static constexpr int kWidth = 32;
  static constexpr int kHeight = 32;

  template <typename T>
  void runMulScaleTest(
      std::function<NppStatus(const T *, int, const T *, int, T *, int, NppiSize)> nppC1R,
      std::function<NppStatus(const T *, int, const T *, int, T *, int, NppiSize, NppStreamContext)> nppC1R_Ctx,
      std::function<NppStatus(const T *, int, T *, int, NppiSize)> nppC1IR,
      std::function<NppStatus(const T *, int, T *, int, NppiSize, NppStreamContext)> nppC1IR_Ctx,
      std::function<NppStatus(const T *, int, const T *, int, T *, int, NppiSize)> nppC3R,
      std::function<NppStatus(const T *, int, const T *, int, T *, int, NppiSize, NppStreamContext)> nppC3R_Ctx,
      std::function<NppStatus(const T *, int, T *, int, NppiSize)> nppC3IR,
      std::function<NppStatus(const T *, int, T *, int, NppiSize, NppStreamContext)> nppC3IR_Ctx,
      std::function<NppStatus(const T *, int, const T *, int, T *, int, NppiSize)> nppC4R,
      std::function<NppStatus(const T *, int, const T *, int, T *, int, NppiSize, NppStreamContext)> nppC4R_Ctx,
      std::function<NppStatus(const T *, int, T *, int, NppiSize)> nppC4IR,
      std::function<NppStatus(const T *, int, T *, int, NppiSize, NppStreamContext)> nppC4IR_Ctx) {
    const auto &p = GetParam();
    const int total = kWidth * kHeight * p.channels;

    // MulScale: dst = (src1 * src2) / maxVal
    // maxVal is 255 for 8u, 65535 for 16u
    T maxVal = std::numeric_limits<T>::max();

    std::vector<T> src1Data(total), src2Data(total), expected(total);
    // Use smaller values to avoid overflow in expected calculation
    T halfMax = maxVal / 2;
    TestDataGenerator::generateRandom(src1Data, static_cast<T>(0), halfMax, 12345);
    TestDataGenerator::generateRandom(src2Data, static_cast<T>(0), halfMax, 54321);

    for (int i = 0; i < total; i++) {
      // Use 64-bit for intermediate calculation
      uint64_t product = static_cast<uint64_t>(src1Data[i]) * static_cast<uint64_t>(src2Data[i]);
      expected[i] = static_cast<T>(product / maxVal);
    }

    NppImageMemory<T> src1(kWidth * p.channels, kHeight);
    NppImageMemory<T> src2(kWidth * p.channels, kHeight);
    src1.copyFromHost(src1Data);
    src2.copyFromHost(src2Data);

    NppiSize roi = {kWidth, kHeight};
    NppStreamContext ctx{};
    ctx.hStream = 0;
    NppStatus status;
    std::vector<T> result(total);

    if (p.in_place) {
      if (p.channels == 1) {
        status = p.use_ctx ? nppC1IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx)
                           : nppC1IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
      } else if (p.channels == 3) {
        status = p.use_ctx ? nppC3IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx)
                           : nppC3IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
      } else {
        status = p.use_ctx ? nppC4IR_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), roi, ctx)
                           : nppC4IR(src1.get(), src1.step(), src2.get(), src2.step(), roi);
      }
      ASSERT_EQ(status, NPP_NO_ERROR);
      src2.copyToHost(result);
    } else {
      NppImageMemory<T> dst(kWidth * p.channels, kHeight);
      if (p.channels == 1) {
        status = p.use_ctx
                     ? nppC1R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx)
                     : nppC1R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
      } else if (p.channels == 3) {
        status = p.use_ctx
                     ? nppC3R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx)
                     : nppC3R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
      } else {
        status = p.use_ctx
                     ? nppC4R_Ctx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx)
                     : nppC4R(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
      }
      ASSERT_EQ(status, NPP_NO_ERROR);
      dst.copyToHost(result);
    }

    EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
  }
};

// Parameter values
static const std::vector<MulScaleParam> kMulScaleParams = {
    {1, false, false}, {1, true, false}, {1, false, true},  {1, true, true},  {3, false, false}, {3, true, false},
    {3, false, true},  {3, true, true},  {4, false, false}, {4, true, false}, {4, false, true},  {4, true, true},
};

// ==================== MulScale_8u ====================
class MulScale8uParamTest : public MulScaleParamTest {};

TEST_P(MulScale8uParamTest, MulScale_8u) {
  runMulScaleTest<Npp8u>(nppiMulScale_8u_C1R, nppiMulScale_8u_C1R_Ctx, nppiMulScale_8u_C1IR, nppiMulScale_8u_C1IR_Ctx,
                         nppiMulScale_8u_C3R, nppiMulScale_8u_C3R_Ctx, nppiMulScale_8u_C3IR, nppiMulScale_8u_C3IR_Ctx,
                         nppiMulScale_8u_C4R, nppiMulScale_8u_C4R_Ctx, nppiMulScale_8u_C4IR, nppiMulScale_8u_C4IR_Ctx);
}

INSTANTIATE_TEST_SUITE_P(MulScale8u, MulScale8uParamTest, ::testing::ValuesIn(kMulScaleParams),
                         [](const auto &info) { return info.param.name(); });

// ==================== MulScale_16u ====================
class MulScale16uParamTest : public MulScaleParamTest {};

TEST_P(MulScale16uParamTest, MulScale_16u) {
  runMulScaleTest<Npp16u>(nppiMulScale_16u_C1R, nppiMulScale_16u_C1R_Ctx, nppiMulScale_16u_C1IR,
                          nppiMulScale_16u_C1IR_Ctx, nppiMulScale_16u_C3R, nppiMulScale_16u_C3R_Ctx,
                          nppiMulScale_16u_C3IR, nppiMulScale_16u_C3IR_Ctx, nppiMulScale_16u_C4R,
                          nppiMulScale_16u_C4R_Ctx, nppiMulScale_16u_C4IR, nppiMulScale_16u_C4IR_Ctx);
}

INSTANTIATE_TEST_SUITE_P(MulScale16u, MulScale16uParamTest, ::testing::ValuesIn(kMulScaleParams),
                         [](const auto &info) { return info.param.name(); });
