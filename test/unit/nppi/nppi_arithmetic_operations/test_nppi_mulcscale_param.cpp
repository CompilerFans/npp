#include "npp_test_base.h"
#include <functional>

using namespace npp_functional_test;

// Parameter structure for MulCScale operation tests
struct MulCScaleParam {
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

// Base test class for MulCScale operations
class MulCScaleParamTest : public NppTestBase, public ::testing::WithParamInterface<MulCScaleParam> {
protected:
  static constexpr int kWidth = 32;
  static constexpr int kHeight = 32;

  template <typename T>
  void
  runMulCScaleTest(T constVal, std::function<NppStatus(const T *, int, T, T *, int, NppiSize)> nppC1R,
                   std::function<NppStatus(const T *, int, T, T *, int, NppiSize, NppStreamContext)> nppC1R_Ctx,
                   std::function<NppStatus(T, T *, int, NppiSize)> nppC1IR,
                   std::function<NppStatus(T, T *, int, NppiSize, NppStreamContext)> nppC1IR_Ctx,
                   std::function<NppStatus(const T *, int, const T *, T *, int, NppiSize)> nppC3R,
                   std::function<NppStatus(const T *, int, const T *, T *, int, NppiSize, NppStreamContext)> nppC3R_Ctx,
                   std::function<NppStatus(const T *, T *, int, NppiSize)> nppC3IR,
                   std::function<NppStatus(const T *, T *, int, NppiSize, NppStreamContext)> nppC3IR_Ctx,
                   std::function<NppStatus(const T *, int, const T *, T *, int, NppiSize)> nppC4R,
                   std::function<NppStatus(const T *, int, const T *, T *, int, NppiSize, NppStreamContext)> nppC4R_Ctx,
                   std::function<NppStatus(const T *, T *, int, NppiSize)> nppC4IR,
                   std::function<NppStatus(const T *, T *, int, NppiSize, NppStreamContext)> nppC4IR_Ctx) {
    const auto &p = GetParam();
    const int total = kWidth * kHeight * p.channels;

    // MulCScale: dst = (src * constant) / maxVal
    // maxVal is 255 for 8u, 65535 for 16u
    T maxVal = std::numeric_limits<T>::max();

    std::vector<T> srcData(total), expected(total);
    // Use smaller values to avoid overflow in expected calculation
    T halfMax = maxVal / 2;
    TestDataGenerator::generateRandom(srcData, static_cast<T>(0), halfMax, 12345);

    T constants[4] = {constVal, constVal, constVal, constVal};
    for (int i = 0; i < total; i++) {
      // Use 64-bit for intermediate calculation
      uint64_t product = static_cast<uint64_t>(srcData[i]) * static_cast<uint64_t>(constants[i % p.channels]);
      expected[i] = static_cast<T>(product / maxVal);
    }

    NppImageMemory<T> src(kWidth * p.channels, kHeight);
    src.copyFromHost(srcData);

    NppiSize roi = {kWidth, kHeight};
    NppStreamContext ctx{};
    ctx.hStream = 0;
    NppStatus status;
    std::vector<T> result(total);

    if (p.in_place) {
      if (p.channels == 1) {
        status = p.use_ctx ? nppC1IR_Ctx(constVal, src.get(), src.step(), roi, ctx)
                           : nppC1IR(constVal, src.get(), src.step(), roi);
      } else if (p.channels == 3) {
        status = p.use_ctx ? nppC3IR_Ctx(constants, src.get(), src.step(), roi, ctx)
                           : nppC3IR(constants, src.get(), src.step(), roi);
      } else {
        status = p.use_ctx ? nppC4IR_Ctx(constants, src.get(), src.step(), roi, ctx)
                           : nppC4IR(constants, src.get(), src.step(), roi);
      }
      ASSERT_EQ(status, NPP_NO_ERROR);
      src.copyToHost(result);
    } else {
      NppImageMemory<T> dst(kWidth * p.channels, kHeight);
      if (p.channels == 1) {
        status = p.use_ctx ? nppC1R_Ctx(src.get(), src.step(), constVal, dst.get(), dst.step(), roi, ctx)
                           : nppC1R(src.get(), src.step(), constVal, dst.get(), dst.step(), roi);
      } else if (p.channels == 3) {
        status = p.use_ctx ? nppC3R_Ctx(src.get(), src.step(), constants, dst.get(), dst.step(), roi, ctx)
                           : nppC3R(src.get(), src.step(), constants, dst.get(), dst.step(), roi);
      } else {
        status = p.use_ctx ? nppC4R_Ctx(src.get(), src.step(), constants, dst.get(), dst.step(), roi, ctx)
                           : nppC4R(src.get(), src.step(), constants, dst.get(), dst.step(), roi);
      }
      ASSERT_EQ(status, NPP_NO_ERROR);
      dst.copyToHost(result);
    }

    EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
  }
};

// Parameter values
static const std::vector<MulCScaleParam> kMulCScaleParams = {
    {1, false, false}, {1, true, false}, {1, false, true},  {1, true, true},  {3, false, false}, {3, true, false},
    {3, false, true},  {3, true, true},  {4, false, false}, {4, true, false}, {4, false, true},  {4, true, true},
};

// ==================== MulCScale_8u ====================
class MulCScale8uParamTest : public MulCScaleParamTest {};

TEST_P(MulCScale8uParamTest, MulCScale_8u) {
  runMulCScaleTest<Npp8u>(static_cast<Npp8u>(128), nppiMulCScale_8u_C1R, nppiMulCScale_8u_C1R_Ctx,
                          nppiMulCScale_8u_C1IR, nppiMulCScale_8u_C1IR_Ctx, nppiMulCScale_8u_C3R,
                          nppiMulCScale_8u_C3R_Ctx, nppiMulCScale_8u_C3IR, nppiMulCScale_8u_C3IR_Ctx,
                          nppiMulCScale_8u_C4R, nppiMulCScale_8u_C4R_Ctx, nppiMulCScale_8u_C4IR,
                          nppiMulCScale_8u_C4IR_Ctx);
}

INSTANTIATE_TEST_SUITE_P(MulCScale8u, MulCScale8uParamTest, ::testing::ValuesIn(kMulCScaleParams),
                         [](const auto &info) { return info.param.name(); });

// ==================== MulCScale_16u ====================
class MulCScale16uParamTest : public MulCScaleParamTest {};

TEST_P(MulCScale16uParamTest, MulCScale_16u) {
  runMulCScaleTest<Npp16u>(static_cast<Npp16u>(32768), nppiMulCScale_16u_C1R, nppiMulCScale_16u_C1R_Ctx,
                           nppiMulCScale_16u_C1IR, nppiMulCScale_16u_C1IR_Ctx, nppiMulCScale_16u_C3R,
                           nppiMulCScale_16u_C3R_Ctx, nppiMulCScale_16u_C3IR, nppiMulCScale_16u_C3IR_Ctx,
                           nppiMulCScale_16u_C4R, nppiMulCScale_16u_C4R_Ctx, nppiMulCScale_16u_C4IR,
                           nppiMulCScale_16u_C4IR_Ctx);
}

INSTANTIATE_TEST_SUITE_P(MulCScale16u, MulCScale16uParamTest, ::testing::ValuesIn(kMulCScaleParams),
                         [](const auto &info) { return info.param.name(); });
