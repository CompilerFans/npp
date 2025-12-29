#include "npp_test_base.h"
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <npp.h>
#include <random>
#include <vector>

using namespace npp_functional_test;

// ==================== AddC Complex Types Tests (16sc, 32sc, 32fc) ====================

// Test parameters for complex AddC operations
struct AddCComplexParam {
  std::string name;
  int channels; // 1, 3, 4, or -4 for AC4
  bool inplace;
  bool useCtx;
};

// 16sc (16-bit signed complex) tests
class AddC16scTest : public NppTestBase, public ::testing::WithParamInterface<AddCComplexParam> {};

TEST_P(AddC16scTest, ComplexAdd) {
  const auto &p = GetParam();
  const int width = 32, height = 32;
  const int actualChannels = (p.channels == -4) ? 4 : p.channels;
  const int total = width * height * actualChannels;

  std::vector<Npp16sc> src(total), expected(total);
  std::mt19937 gen(12345);
  std::uniform_int_distribution<int> dist(-1000, 1000);
  for (int i = 0; i < total; i++) {
    src[i].re = static_cast<Npp16s>(dist(gen));
    src[i].im = static_cast<Npp16s>(dist(gen));
  }

  // Constants array - for AC4, only 3 constants are used
  const int constCount = (p.channels == -4) ? 3 : p.channels;
  std::vector<Npp16sc> aConstants(constCount);
  for (int i = 0; i < constCount; i++) {
    aConstants[i].re = static_cast<Npp16s>(10 * (i + 1));
    aConstants[i].im = static_cast<Npp16s>(5 * (i + 1));
  }

  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int ch = i % actualChannels;
    if (p.channels == -4 && ch == 3) {
      // AC4: alpha channel unchanged
      expected[i] = src[i];
    } else {
      int constIdx = (p.channels == -4) ? ch : ch;
      int sumRe = src[i].re + aConstants[constIdx].re;
      int sumIm = src[i].im + aConstants[constIdx].im;
      expected[i].re = static_cast<Npp16s>(std::min(32767, std::max(-32768, sumRe >> scaleFactor)));
      expected[i].im = static_cast<Npp16s>(std::min(32767, std::max(-32768, sumIm >> scaleFactor)));
    }
  }

  NppImageMemory<Npp16sc> d_src(width * actualChannels, height);
  NppImageMemory<Npp16sc> d_dst(width * actualChannels, height);
  d_src.copyFromHost(src);
  if (p.channels == -4) {
    d_dst.copyFromHost(src); // For AC4, init dst with src for alpha preservation
  }

  NppiSize roi = {width, height};
  NppStatus status;
  NppStreamContext ctx;
  if (p.useCtx) {
    nppGetStreamContext(&ctx);
  }

  if (p.inplace) {
    d_dst.copyFromHost(src);
    if (p.channels == 1) {
      status = p.useCtx ? nppiAddC_16sc_C1IRSfs_Ctx(aConstants[0], d_dst.get(), d_dst.step(), roi, scaleFactor, ctx)
                        : nppiAddC_16sc_C1IRSfs(aConstants[0], d_dst.get(), d_dst.step(), roi, scaleFactor);
    } else if (p.channels == 3) {
      status = p.useCtx ? nppiAddC_16sc_C3IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx)
                        : nppiAddC_16sc_C3IRSfs(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor);
    } else if (p.channels == -4) {
      status = p.useCtx
                   ? nppiAddC_16sc_AC4IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx)
                   : nppiAddC_16sc_AC4IRSfs(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor);
    }
  } else {
    if (p.channels == 1) {
      status = p.useCtx ? nppiAddC_16sc_C1RSfs_Ctx(d_src.get(), d_src.step(), aConstants[0], d_dst.get(), d_dst.step(),
                                                   roi, scaleFactor, ctx)
                        : nppiAddC_16sc_C1RSfs(d_src.get(), d_src.step(), aConstants[0], d_dst.get(), d_dst.step(), roi,
                                               scaleFactor);
    } else if (p.channels == 3) {
      status = p.useCtx ? nppiAddC_16sc_C3RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(),
                                                   d_dst.step(), roi, scaleFactor, ctx)
                        : nppiAddC_16sc_C3RSfs(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(),
                                               roi, scaleFactor);
    } else if (p.channels == -4) {
      status = p.useCtx ? nppiAddC_16sc_AC4RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(),
                                                    d_dst.step(), roi, scaleFactor, ctx)
                        : nppiAddC_16sc_AC4RSfs(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(),
                                                roi, scaleFactor);
    }
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp16sc> result(total);
  d_dst.copyToHost(result);

  for (int i = 0; i < total; i++) {
    EXPECT_EQ(result[i].re, expected[i].re) << "Real mismatch at " << i;
    EXPECT_EQ(result[i].im, expected[i].im) << "Imag mismatch at " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(
    AddC16scSuite, AddC16scTest,
    ::testing::Values(AddCComplexParam{"C1R", 1, false, false}, AddCComplexParam{"C1R_Ctx", 1, false, true},
                      AddCComplexParam{"C1IR", 1, true, false}, AddCComplexParam{"C1IR_Ctx", 1, true, true},
                      AddCComplexParam{"C3R", 3, false, false}, AddCComplexParam{"C3R_Ctx", 3, false, true},
                      AddCComplexParam{"C3IR", 3, true, false}, AddCComplexParam{"C3IR_Ctx", 3, true, true},
                      AddCComplexParam{"AC4R", -4, false, false}, AddCComplexParam{"AC4R_Ctx", -4, false, true},
                      AddCComplexParam{"AC4IR", -4, true, false}, AddCComplexParam{"AC4IR_Ctx", -4, true, true}),
    [](const ::testing::TestParamInfo<AddCComplexParam> &info) { return info.param.name; });

// 32sc (32-bit signed complex) tests
class AddC32scTest : public NppTestBase, public ::testing::WithParamInterface<AddCComplexParam> {};

TEST_P(AddC32scTest, ComplexAdd) {
  const auto &p = GetParam();
  const int width = 32, height = 32;
  const int actualChannels = (p.channels == -4) ? 4 : p.channels;
  const int total = width * height * actualChannels;

  std::vector<Npp32sc> src(total), expected(total);
  std::mt19937 gen(12345);
  std::uniform_int_distribution<int> dist(-100000, 100000);
  for (int i = 0; i < total; i++) {
    src[i].re = dist(gen);
    src[i].im = dist(gen);
  }

  const int constCount = (p.channels == -4) ? 3 : p.channels;
  std::vector<Npp32sc> aConstants(constCount);
  for (int i = 0; i < constCount; i++) {
    aConstants[i].re = 100 * (i + 1);
    aConstants[i].im = 50 * (i + 1);
  }

  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int ch = i % actualChannels;
    if (p.channels == -4 && ch == 3) {
      expected[i] = src[i];
    } else {
      int constIdx = (p.channels == -4) ? ch : ch;
      int64_t sumRe = (int64_t)src[i].re + aConstants[constIdx].re;
      int64_t sumIm = (int64_t)src[i].im + aConstants[constIdx].im;
      expected[i].re = static_cast<Npp32s>(sumRe >> scaleFactor);
      expected[i].im = static_cast<Npp32s>(sumIm >> scaleFactor);
    }
  }

  NppImageMemory<Npp32sc> d_src(width * actualChannels, height);
  NppImageMemory<Npp32sc> d_dst(width * actualChannels, height);
  d_src.copyFromHost(src);
  if (p.channels == -4) {
    d_dst.copyFromHost(src);
  }

  NppiSize roi = {width, height};
  NppStatus status;
  NppStreamContext ctx;
  if (p.useCtx) {
    nppGetStreamContext(&ctx);
  }

  if (p.inplace) {
    d_dst.copyFromHost(src);
    if (p.channels == 1) {
      status = p.useCtx ? nppiAddC_32sc_C1IRSfs_Ctx(aConstants[0], d_dst.get(), d_dst.step(), roi, scaleFactor, ctx)
                        : nppiAddC_32sc_C1IRSfs(aConstants[0], d_dst.get(), d_dst.step(), roi, scaleFactor);
    } else if (p.channels == 3) {
      status = p.useCtx ? nppiAddC_32sc_C3IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx)
                        : nppiAddC_32sc_C3IRSfs(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor);
    } else if (p.channels == -4) {
      status = p.useCtx
                   ? nppiAddC_32sc_AC4IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx)
                   : nppiAddC_32sc_AC4IRSfs(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor);
    }
  } else {
    if (p.channels == 1) {
      status = p.useCtx ? nppiAddC_32sc_C1RSfs_Ctx(d_src.get(), d_src.step(), aConstants[0], d_dst.get(), d_dst.step(),
                                                   roi, scaleFactor, ctx)
                        : nppiAddC_32sc_C1RSfs(d_src.get(), d_src.step(), aConstants[0], d_dst.get(), d_dst.step(), roi,
                                               scaleFactor);
    } else if (p.channels == 3) {
      status = p.useCtx ? nppiAddC_32sc_C3RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(),
                                                   d_dst.step(), roi, scaleFactor, ctx)
                        : nppiAddC_32sc_C3RSfs(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(),
                                               roi, scaleFactor);
    } else if (p.channels == -4) {
      status = p.useCtx ? nppiAddC_32sc_AC4RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(),
                                                    d_dst.step(), roi, scaleFactor, ctx)
                        : nppiAddC_32sc_AC4RSfs(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(),
                                                roi, scaleFactor);
    }
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32sc> result(total);
  d_dst.copyToHost(result);

  for (int i = 0; i < total; i++) {
    EXPECT_EQ(result[i].re, expected[i].re) << "Real mismatch at " << i;
    EXPECT_EQ(result[i].im, expected[i].im) << "Imag mismatch at " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(
    AddC32scSuite, AddC32scTest,
    ::testing::Values(AddCComplexParam{"C1R", 1, false, false}, AddCComplexParam{"C1R_Ctx", 1, false, true},
                      AddCComplexParam{"C1IR", 1, true, false}, AddCComplexParam{"C1IR_Ctx", 1, true, true},
                      AddCComplexParam{"C3R", 3, false, false}, AddCComplexParam{"C3R_Ctx", 3, false, true},
                      AddCComplexParam{"C3IR", 3, true, false}, AddCComplexParam{"C3IR_Ctx", 3, true, true},
                      AddCComplexParam{"AC4R", -4, false, false}, AddCComplexParam{"AC4R_Ctx", -4, false, true},
                      AddCComplexParam{"AC4IR", -4, true, false}, AddCComplexParam{"AC4IR_Ctx", -4, true, true}),
    [](const ::testing::TestParamInfo<AddCComplexParam> &info) { return info.param.name; });

// 32fc (32-bit float complex) tests
class AddC32fcTest : public NppTestBase, public ::testing::WithParamInterface<AddCComplexParam> {};

TEST_P(AddC32fcTest, ComplexAdd) {
  const auto &p = GetParam();
  const int width = 32, height = 32;
  const int actualChannels = (p.channels == -4) ? 4 : p.channels;
  const int total = width * height * actualChannels;

  std::vector<Npp32fc> src(total), expected(total);
  std::mt19937 gen(12345);
  std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
  for (int i = 0; i < total; i++) {
    src[i].re = dist(gen);
    src[i].im = dist(gen);
  }

  const int constCount = (p.channels == -4) ? 3 : p.channels;
  std::vector<Npp32fc> aConstants(constCount);
  for (int i = 0; i < constCount; i++) {
    aConstants[i].re = 1.5f * (i + 1);
    aConstants[i].im = 0.5f * (i + 1);
  }

  for (int i = 0; i < total; i++) {
    int ch = i % actualChannels;
    if (p.channels == -4 && ch == 3) {
      expected[i] = src[i];
    } else {
      int constIdx = (p.channels == -4) ? ch : ch;
      expected[i].re = src[i].re + aConstants[constIdx].re;
      expected[i].im = src[i].im + aConstants[constIdx].im;
    }
  }

  NppImageMemory<Npp32fc> d_src(width * actualChannels, height);
  NppImageMemory<Npp32fc> d_dst(width * actualChannels, height);
  d_src.copyFromHost(src);
  if (p.channels == -4) {
    d_dst.copyFromHost(src);
  }

  NppiSize roi = {width, height};
  NppStatus status;
  NppStreamContext ctx;
  if (p.useCtx) {
    nppGetStreamContext(&ctx);
  }

  if (p.inplace) {
    d_dst.copyFromHost(src);
    if (p.channels == 1) {
      status = p.useCtx ? nppiAddC_32fc_C1IR_Ctx(aConstants[0], d_dst.get(), d_dst.step(), roi, ctx)
                        : nppiAddC_32fc_C1IR(aConstants[0], d_dst.get(), d_dst.step(), roi);
    } else if (p.channels == 3) {
      status = p.useCtx ? nppiAddC_32fc_C3IR_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, ctx)
                        : nppiAddC_32fc_C3IR(aConstants.data(), d_dst.get(), d_dst.step(), roi);
    } else if (p.channels == 4) {
      status = p.useCtx ? nppiAddC_32fc_C4IR_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, ctx)
                        : nppiAddC_32fc_C4IR(aConstants.data(), d_dst.get(), d_dst.step(), roi);
    } else if (p.channels == -4) {
      status = p.useCtx ? nppiAddC_32fc_AC4IR_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, ctx)
                        : nppiAddC_32fc_AC4IR(aConstants.data(), d_dst.get(), d_dst.step(), roi);
    }
  } else {
    if (p.channels == 1) {
      status = p.useCtx ? nppiAddC_32fc_C1R_Ctx(d_src.get(), d_src.step(), aConstants[0], d_dst.get(), d_dst.step(),
                                                roi, ctx)
                        : nppiAddC_32fc_C1R(d_src.get(), d_src.step(), aConstants[0], d_dst.get(), d_dst.step(), roi);
    } else if (p.channels == 3) {
      status =
          p.useCtx
              ? nppiAddC_32fc_C3R_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi, ctx)
              : nppiAddC_32fc_C3R(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi);
    } else if (p.channels == 4) {
      status =
          p.useCtx
              ? nppiAddC_32fc_C4R_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi, ctx)
              : nppiAddC_32fc_C4R(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi);
    } else if (p.channels == -4) {
      status = p.useCtx
                   ? nppiAddC_32fc_AC4R_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(),
                                            roi, ctx)
                   : nppiAddC_32fc_AC4R(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi);
    }
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32fc> result(total);
  d_dst.copyToHost(result);

  for (int i = 0; i < total; i++) {
    EXPECT_NEAR(result[i].re, expected[i].re, 1e-4f) << "Real mismatch at " << i;
    EXPECT_NEAR(result[i].im, expected[i].im, 1e-4f) << "Imag mismatch at " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(
    AddC32fcSuite, AddC32fcTest,
    ::testing::Values(AddCComplexParam{"C1R", 1, false, false}, AddCComplexParam{"C1R_Ctx", 1, false, true},
                      AddCComplexParam{"C1IR", 1, true, false}, AddCComplexParam{"C1IR_Ctx", 1, true, true},
                      AddCComplexParam{"C3R", 3, false, false}, AddCComplexParam{"C3R_Ctx", 3, false, true},
                      AddCComplexParam{"C3IR", 3, true, false}, AddCComplexParam{"C3IR_Ctx", 3, true, true},
                      AddCComplexParam{"C4R", 4, false, false}, AddCComplexParam{"C4R_Ctx", 4, false, true},
                      AddCComplexParam{"C4IR", 4, true, false}, AddCComplexParam{"C4IR_Ctx", 4, true, true},
                      AddCComplexParam{"AC4R", -4, false, false}, AddCComplexParam{"AC4R_Ctx", -4, false, true},
                      AddCComplexParam{"AC4IR", -4, true, false}, AddCComplexParam{"AC4IR_Ctx", -4, true, true}),
    [](const ::testing::TestParamInfo<AddCComplexParam> &info) { return info.param.name; });

// ==================== AddC Ctx variants for integer types ====================

struct AddCIntCtxParam {
  std::string name;
  std::string dtype; // "8u", "16u", "16s", "32s", "32f"
  int channels;      // 1, 3, 4, or -4 for AC4
  bool inplace;
};

class AddCIntCtxTest : public NppTestBase, public ::testing::WithParamInterface<AddCIntCtxParam> {};

TEST_P(AddCIntCtxTest, CtxVariant) {
  const auto &p = GetParam();
  const int width = 32, height = 32;
  const int actualChannels = (p.channels == -4) ? 4 : p.channels;

  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppiSize roi = {width, height};
  NppStatus status = NPP_NO_ERROR;

  if (p.dtype == "8u") {
    const int total = width * height * actualChannels;
    std::vector<Npp8u> src(total), expected(total);
    TestDataGenerator::generateRandom(src, (Npp8u)0, (Npp8u)200, 12345);

    const int constCount = (p.channels == -4) ? 3 : actualChannels;
    std::vector<Npp8u> aConstants(constCount);
    for (int i = 0; i < constCount; i++) {
      aConstants[i] = static_cast<Npp8u>(10 * (i + 1));
    }

    int scaleFactor = 0;
    for (int i = 0; i < total; i++) {
      int ch = i % actualChannels;
      if (p.channels == -4 && ch == 3) {
        expected[i] = src[i];
      } else {
        int constIdx = (p.channels == -4) ? ch : ch;
        int sum = src[i] + aConstants[constIdx];
        expected[i] = static_cast<Npp8u>(std::min(255, std::max(0, sum >> scaleFactor)));
      }
    }

    NppImageMemory<Npp8u> d_src(width * actualChannels, height);
    NppImageMemory<Npp8u> d_dst(width * actualChannels, height);
    d_src.copyFromHost(src);
    if (p.channels == -4)
      d_dst.copyFromHost(src);

    if (p.inplace) {
      d_dst.copyFromHost(src);
      if (p.channels == 3) {
        status = nppiAddC_8u_C3IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      } else if (p.channels == 4) {
        status = nppiAddC_8u_C4IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      } else if (p.channels == -4) {
        status = nppiAddC_8u_AC4IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      }
    } else {
      if (p.channels == 3) {
        status = nppiAddC_8u_C3RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi,
                                        scaleFactor, ctx);
      } else if (p.channels == 4) {
        status = nppiAddC_8u_C4RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi,
                                        scaleFactor, ctx);
      } else if (p.channels == -4) {
        status = nppiAddC_8u_AC4RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi,
                                         scaleFactor, ctx);
      }
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> result(total);
    d_dst.copyToHost(result);
    EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
  } else if (p.dtype == "16u") {
    const int total = width * height * actualChannels;
    std::vector<Npp16u> src(total), expected(total);
    TestDataGenerator::generateRandom(src, (Npp16u)0, (Npp16u)60000, 12345);

    const int constCount = (p.channels == -4) ? 3 : actualChannels;
    std::vector<Npp16u> aConstants(constCount);
    for (int i = 0; i < constCount; i++) {
      aConstants[i] = static_cast<Npp16u>(100 * (i + 1));
    }

    int scaleFactor = 0;
    for (int i = 0; i < total; i++) {
      int ch = i % actualChannels;
      if (p.channels == -4 && ch == 3) {
        expected[i] = src[i];
      } else {
        int constIdx = (p.channels == -4) ? ch : ch;
        int sum = src[i] + aConstants[constIdx];
        expected[i] = static_cast<Npp16u>(std::min(65535, std::max(0, sum >> scaleFactor)));
      }
    }

    NppImageMemory<Npp16u> d_src(width * actualChannels, height);
    NppImageMemory<Npp16u> d_dst(width * actualChannels, height);
    d_src.copyFromHost(src);
    if (p.channels == -4)
      d_dst.copyFromHost(src);

    if (p.inplace) {
      d_dst.copyFromHost(src);
      if (p.channels == 3) {
        status = nppiAddC_16u_C3IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      } else if (p.channels == 4) {
        status = nppiAddC_16u_C4IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      } else if (p.channels == -4) {
        status = nppiAddC_16u_AC4IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      }
    } else {
      if (p.channels == 3) {
        status = nppiAddC_16u_C3RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi,
                                         scaleFactor, ctx);
      } else if (p.channels == 4) {
        status = nppiAddC_16u_C4RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi,
                                         scaleFactor, ctx);
      } else if (p.channels == -4) {
        status = nppiAddC_16u_AC4RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi,
                                          scaleFactor, ctx);
      }
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> result(total);
    d_dst.copyToHost(result);
    EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
  } else if (p.dtype == "16s") {
    const int total = width * height * actualChannels;
    std::vector<Npp16s> src(total), expected(total);
    TestDataGenerator::generateRandom(src, (Npp16s)-10000, (Npp16s)10000, 12345);

    const int constCount = (p.channels == -4) ? 3 : actualChannels;
    std::vector<Npp16s> aConstants(constCount);
    for (int i = 0; i < constCount; i++) {
      aConstants[i] = static_cast<Npp16s>(100 * (i + 1) * ((i % 2 == 0) ? 1 : -1));
    }

    int scaleFactor = 0;
    for (int i = 0; i < total; i++) {
      int ch = i % actualChannels;
      if (p.channels == -4 && ch == 3) {
        expected[i] = src[i];
      } else {
        int constIdx = (p.channels == -4) ? ch : ch;
        int sum = src[i] + aConstants[constIdx];
        expected[i] = static_cast<Npp16s>(std::min(32767, std::max(-32768, sum >> scaleFactor)));
      }
    }

    NppImageMemory<Npp16s> d_src(width * actualChannels, height);
    NppImageMemory<Npp16s> d_dst(width * actualChannels, height);
    d_src.copyFromHost(src);
    if (p.channels == -4)
      d_dst.copyFromHost(src);

    if (p.inplace) {
      d_dst.copyFromHost(src);
      if (p.channels == 1) {
        status = nppiAddC_16s_C1IRSfs_Ctx(aConstants[0], d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      } else if (p.channels == 3) {
        status = nppiAddC_16s_C3IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      } else if (p.channels == 4) {
        status = nppiAddC_16s_C4IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      } else if (p.channels == -4) {
        status = nppiAddC_16s_AC4IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      }
    } else {
      if (p.channels == 1) {
        status = nppiAddC_16s_C1RSfs_Ctx(d_src.get(), d_src.step(), aConstants[0], d_dst.get(), d_dst.step(), roi,
                                         scaleFactor, ctx);
      } else if (p.channels == 3) {
        status = nppiAddC_16s_C3RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi,
                                         scaleFactor, ctx);
      } else if (p.channels == 4) {
        status = nppiAddC_16s_C4RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi,
                                         scaleFactor, ctx);
      } else if (p.channels == -4) {
        status = nppiAddC_16s_AC4RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi,
                                          scaleFactor, ctx);
      }
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16s> result(total);
    d_dst.copyToHost(result);
    EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
  } else if (p.dtype == "32s") {
    const int total = width * height * actualChannels;
    std::vector<Npp32s> src(total), expected(total);
    TestDataGenerator::generateRandom(src, (Npp32s)-100000, (Npp32s)100000, 12345);

    const int constCount = (p.channels == -4) ? 3 : actualChannels;
    std::vector<Npp32s> aConstants(constCount);
    for (int i = 0; i < constCount; i++) {
      aConstants[i] = 100 * (i + 1) * ((i % 2 == 0) ? 1 : -1);
    }

    int scaleFactor = 0;
    for (int i = 0; i < total; i++) {
      int ch = i % actualChannels;
      if (p.channels == -4 && ch == 3) {
        expected[i] = src[i];
      } else {
        int constIdx = (p.channels == -4) ? ch : ch;
        int64_t sum = (int64_t)src[i] + aConstants[constIdx];
        expected[i] = static_cast<Npp32s>(sum >> scaleFactor);
      }
    }

    NppImageMemory<Npp32s> d_src(width * actualChannels, height);
    NppImageMemory<Npp32s> d_dst(width * actualChannels, height);
    d_src.copyFromHost(src);
    if (p.channels == -4)
      d_dst.copyFromHost(src);

    if (p.inplace) {
      d_dst.copyFromHost(src);
      if (p.channels == 1) {
        status = nppiAddC_32s_C1IRSfs_Ctx(aConstants[0], d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      } else if (p.channels == 3) {
        status = nppiAddC_32s_C3IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      }
    } else {
      if (p.channels == 1) {
        status = nppiAddC_32s_C1RSfs_Ctx(d_src.get(), d_src.step(), aConstants[0], d_dst.get(), d_dst.step(), roi,
                                         scaleFactor, ctx);
      } else if (p.channels == 3) {
        status = nppiAddC_32s_C3RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi,
                                         scaleFactor, ctx);
      }
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32s> result(total);
    d_dst.copyToHost(result);
    EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
  } else if (p.dtype == "32f") {
    const int total = width * height * actualChannels;
    std::vector<Npp32f> src(total), expected(total);
    TestDataGenerator::generateRandom(src, -100.0f, 100.0f, 12345);

    const int constCount = (p.channels == -4) ? 3 : actualChannels;
    std::vector<Npp32f> aConstants(constCount);
    for (int i = 0; i < constCount; i++) {
      aConstants[i] = 1.5f * (i + 1) * ((i % 2 == 0) ? 1 : -1);
    }

    for (int i = 0; i < total; i++) {
      int ch = i % actualChannels;
      if (p.channels == -4 && ch == 3) {
        expected[i] = src[i];
      } else {
        int constIdx = (p.channels == -4) ? ch : ch;
        expected[i] = src[i] + aConstants[constIdx];
      }
    }

    NppImageMemory<Npp32f> d_src(width * actualChannels, height);
    NppImageMemory<Npp32f> d_dst(width * actualChannels, height);
    d_src.copyFromHost(src);
    if (p.channels == -4)
      d_dst.copyFromHost(src);

    if (p.inplace) {
      d_dst.copyFromHost(src);
      if (p.channels == 3) {
        status = nppiAddC_32f_C3IR_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, ctx);
      } else if (p.channels == 4) {
        status = nppiAddC_32f_C4IR_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, ctx);
      } else if (p.channels == -4) {
        status = nppiAddC_32f_AC4IR_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, ctx);
      }
    } else {
      if (p.channels == 3) {
        status =
            nppiAddC_32f_C3R_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi, ctx);
      } else if (p.channels == 4) {
        status =
            nppiAddC_32f_C4R_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi, ctx);
      } else if (p.channels == -4) {
        status =
            nppiAddC_32f_AC4R_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi, ctx);
      }
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> result(total);
    d_dst.copyToHost(result);
    EXPECT_TRUE(ResultValidator::arraysEqual(result, expected, 1e-5f));
  }
}

INSTANTIATE_TEST_SUITE_P(
    AddCIntCtxSuite, AddCIntCtxTest,
    ::testing::Values(
        // 8u Ctx variants
        AddCIntCtxParam{"8u_C3RSfs_Ctx", "8u", 3, false}, AddCIntCtxParam{"8u_C4RSfs_Ctx", "8u", 4, false},
        AddCIntCtxParam{"8u_AC4RSfs_Ctx", "8u", -4, false}, AddCIntCtxParam{"8u_C3IRSfs_Ctx", "8u", 3, true},
        AddCIntCtxParam{"8u_C4IRSfs_Ctx", "8u", 4, true}, AddCIntCtxParam{"8u_AC4IRSfs_Ctx", "8u", -4, true},
        // 16u Ctx variants
        AddCIntCtxParam{"16u_C3RSfs_Ctx", "16u", 3, false}, AddCIntCtxParam{"16u_C4RSfs_Ctx", "16u", 4, false},
        AddCIntCtxParam{"16u_AC4RSfs_Ctx", "16u", -4, false}, AddCIntCtxParam{"16u_C3IRSfs_Ctx", "16u", 3, true},
        AddCIntCtxParam{"16u_C4IRSfs_Ctx", "16u", 4, true}, AddCIntCtxParam{"16u_AC4IRSfs_Ctx", "16u", -4, true},
        // 16s Ctx variants
        AddCIntCtxParam{"16s_C1RSfs_Ctx", "16s", 1, false}, AddCIntCtxParam{"16s_C3RSfs_Ctx", "16s", 3, false},
        AddCIntCtxParam{"16s_C4RSfs_Ctx", "16s", 4, false}, AddCIntCtxParam{"16s_AC4RSfs_Ctx", "16s", -4, false},
        AddCIntCtxParam{"16s_C1IRSfs_Ctx", "16s", 1, true}, AddCIntCtxParam{"16s_C3IRSfs_Ctx", "16s", 3, true},
        AddCIntCtxParam{"16s_C4IRSfs_Ctx", "16s", 4, true}, AddCIntCtxParam{"16s_AC4IRSfs_Ctx", "16s", -4, true},
        // 32s Ctx variants
        AddCIntCtxParam{"32s_C1RSfs_Ctx", "32s", 1, false}, AddCIntCtxParam{"32s_C3RSfs_Ctx", "32s", 3, false},
        AddCIntCtxParam{"32s_C1IRSfs_Ctx", "32s", 1, true}, AddCIntCtxParam{"32s_C3IRSfs_Ctx", "32s", 3, true},
        // 32f Ctx variants
        AddCIntCtxParam{"32f_C3R_Ctx", "32f", 3, false}, AddCIntCtxParam{"32f_C4R_Ctx", "32f", 4, false},
        AddCIntCtxParam{"32f_AC4R_Ctx", "32f", -4, false}, AddCIntCtxParam{"32f_C3IR_Ctx", "32f", 3, true},
        AddCIntCtxParam{"32f_C4IR_Ctx", "32f", 4, true}, AddCIntCtxParam{"32f_AC4IR_Ctx", "32f", -4, true}),
    [](const ::testing::TestParamInfo<AddCIntCtxParam> &info) { return info.param.name; });

// ==================== AddC Non-Ctx integer inplace variants (not yet tested) ====================

// 16s non-Ctx inplace
class AddC16sInplaceTest : public NppTestBase {};

TEST_F(AddC16sInplaceTest, AddC_16s_C1IRSfs) {
  const int width = 32, height = 32;
  const int total = width * height;

  std::vector<Npp16s> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp16s)-10000, (Npp16s)10000, 12345);

  Npp16s nConstant = 500;
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int sum = src[i] + nConstant;
    expected[i] = static_cast<Npp16s>(std::min(32767, std::max(-32768, sum >> scaleFactor)));
  }

  NppImageMemory<Npp16s> d_srcdst(width, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAddC_16s_C1IRSfs(nConstant, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp16s> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(AddC16sInplaceTest, AddC_16s_C3IRSfs) {
  const int width = 32, height = 32, channels = 3;
  const int total = width * height * channels;

  std::vector<Npp16s> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp16s)-10000, (Npp16s)10000, 12345);

  Npp16s aConstants[3] = {100, -200, 300};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int ch = i % 3;
    int sum = src[i] + aConstants[ch];
    expected[i] = static_cast<Npp16s>(std::min(32767, std::max(-32768, sum >> scaleFactor)));
  }

  NppImageMemory<Npp16s> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAddC_16s_C3IRSfs(aConstants, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp16s> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(AddC16sInplaceTest, AddC_16s_C4IRSfs) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp16s> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp16s)-10000, (Npp16s)10000, 12345);

  Npp16s aConstants[4] = {100, -200, 300, -400};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int ch = i % 4;
    int sum = src[i] + aConstants[ch];
    expected[i] = static_cast<Npp16s>(std::min(32767, std::max(-32768, sum >> scaleFactor)));
  }

  NppImageMemory<Npp16s> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAddC_16s_C4IRSfs(aConstants, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp16s> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(AddC16sInplaceTest, AddC_16s_AC4IRSfs) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp16s> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp16s)-10000, (Npp16s)10000, 12345);

  Npp16s aConstants[3] = {100, -200, 300};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int ch = i % 4;
    if (ch == 3) {
      expected[i] = src[i];
    } else {
      int sum = src[i] + aConstants[ch];
      expected[i] = static_cast<Npp16s>(std::min(32767, std::max(-32768, sum >> scaleFactor)));
    }
  }

  NppImageMemory<Npp16s> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAddC_16s_AC4IRSfs(aConstants, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp16s> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

// 16u inplace tests
class AddC16uInplaceTest : public NppTestBase {};

TEST_F(AddC16uInplaceTest, AddC_16u_C3IRSfs) {
  const int width = 32, height = 32, channels = 3;
  const int total = width * height * channels;

  std::vector<Npp16u> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp16u)0, (Npp16u)60000, 12345);

  Npp16u aConstants[3] = {100, 200, 300};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int ch = i % 3;
    int sum = src[i] + aConstants[ch];
    expected[i] = static_cast<Npp16u>(std::min(65535, std::max(0, sum >> scaleFactor)));
  }

  NppImageMemory<Npp16u> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAddC_16u_C3IRSfs(aConstants, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp16u> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(AddC16uInplaceTest, AddC_16u_C4IRSfs) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp16u> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp16u)0, (Npp16u)60000, 12345);

  Npp16u aConstants[4] = {100, 200, 300, 400};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int ch = i % 4;
    int sum = src[i] + aConstants[ch];
    expected[i] = static_cast<Npp16u>(std::min(65535, std::max(0, sum >> scaleFactor)));
  }

  NppImageMemory<Npp16u> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAddC_16u_C4IRSfs(aConstants, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp16u> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(AddC16uInplaceTest, AddC_16u_AC4IRSfs) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp16u> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp16u)0, (Npp16u)60000, 12345);

  Npp16u aConstants[3] = {100, 200, 300};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int ch = i % 4;
    if (ch == 3) {
      expected[i] = src[i];
    } else {
      int sum = src[i] + aConstants[ch];
      expected[i] = static_cast<Npp16u>(std::min(65535, std::max(0, sum >> scaleFactor)));
    }
  }

  NppImageMemory<Npp16u> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAddC_16u_AC4IRSfs(aConstants, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp16u> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

// 32s inplace tests
class AddC32sInplaceTest : public NppTestBase {};

TEST_F(AddC32sInplaceTest, AddC_32s_C1IRSfs) {
  const int width = 32, height = 32;
  const int total = width * height;

  std::vector<Npp32s> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp32s)-100000, (Npp32s)100000, 12345);

  Npp32s nConstant = 500;
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int64_t sum = (int64_t)src[i] + nConstant;
    expected[i] = static_cast<Npp32s>(sum >> scaleFactor);
  }

  NppImageMemory<Npp32s> d_srcdst(width, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAddC_32s_C1IRSfs(nConstant, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp32s> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(AddC32sInplaceTest, AddC_32s_C3IRSfs) {
  const int width = 32, height = 32, channels = 3;
  const int total = width * height * channels;

  std::vector<Npp32s> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp32s)-100000, (Npp32s)100000, 12345);

  Npp32s aConstants[3] = {100, -200, 300};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int ch = i % 3;
    int64_t sum = (int64_t)src[i] + aConstants[ch];
    expected[i] = static_cast<Npp32s>(sum >> scaleFactor);
  }

  NppImageMemory<Npp32s> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAddC_32s_C3IRSfs(aConstants, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp32s> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}
