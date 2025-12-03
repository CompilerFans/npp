#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Sqrt 32f TEST_P ====================

struct Sqrt32fParam {
  int width;
  int height;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Sqrt32fParamTest : public NppTestBase, public ::testing::WithParamInterface<Sqrt32fParam> {};

TEST_P(Sqrt32fParamTest, Sqrt_32f_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> srcData(width * height);
  // Use only non-negative values for sqrt
  TestDataGenerator::generateRandom(srcData, 0.0f, 100.0f, 12345);

  std::vector<Npp32f> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::sqrt_val<Npp32f>(srcData[i]);
  }

  NppImageMemory<Npp32f> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiSqrt_32f_C1IR_Ctx(src.get(), src.step(), roi, ctx);
    } else {
      status = nppiSqrt_32f_C1IR(src.get(), src.step(), roi);
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
      status = nppiSqrt_32f_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiSqrt_32f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f));
  }
}

INSTANTIATE_TEST_SUITE_P(Sqrt32f, Sqrt32fParamTest,
                         ::testing::Values(Sqrt32fParam{32, 32, false, false, "32x32_noCtx"},
                                           Sqrt32fParam{32, 32, true, false, "32x32_Ctx"},
                                           Sqrt32fParam{32, 32, false, true, "32x32_InPlace"},
                                           Sqrt32fParam{32, 32, true, true, "32x32_InPlace_Ctx"},
                                           Sqrt32fParam{64, 64, false, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<Sqrt32fParam> &info) { return info.param.name; });
