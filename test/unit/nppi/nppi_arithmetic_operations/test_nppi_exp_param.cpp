#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Exp 32f C1 TEST_P ====================

struct Exp32fParam {
  int width;
  int height;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Exp32fParamTest : public NppTestBase, public ::testing::WithParamInterface<Exp32fParam> {};

TEST_P(Exp32fParamTest, Exp_32f_C1R) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  std::vector<Npp32f> srcData(width * height);
  // Use small values to avoid overflow
  TestDataGenerator::generateRandom(srcData, -2.0f, 2.0f, 12345);

  std::vector<Npp32f> expectedData(width * height);
  for (size_t i = 0; i < expectedData.size(); i++) {
    expectedData[i] = expect::exp_val<Npp32f>(srcData[i]);
  }

  NppImageMemory<Npp32f> src(width, height);
  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status;

  if (param.in_place) {
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiExp_32f_C1IR_Ctx(src.get(), src.step(), roi, ctx);
    } else {
      status = nppiExp_32f_C1IR(src.get(), src.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> resultData(width * height);
    src.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-4f));
  } else {
    NppImageMemory<Npp32f> dst(width, height);
    if (param.use_ctx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppiExp_32f_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppiExp_32f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-4f));
  }
}

INSTANTIATE_TEST_SUITE_P(Exp32f, Exp32fParamTest,
                         ::testing::Values(Exp32fParam{32, 32, false, false, "32x32_noCtx"},
                                           Exp32fParam{32, 32, true, false, "32x32_Ctx"},
                                           Exp32fParam{32, 32, false, true, "32x32_InPlace"},
                                           Exp32fParam{32, 32, true, true, "32x32_InPlace_Ctx"},
                                           Exp32fParam{64, 64, false, false, "64x64_noCtx"}),
                         [](const ::testing::TestParamInfo<Exp32fParam> &info) { return info.param.name; });
