#include "npp_test_base.h"
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <npp.h>
#include <random>
#include <vector>

using namespace npp_functional_test;

// ==================== SubC Complex Types Tests (16sc, 32sc, 32fc) ====================

struct SubCComplexParam {
  std::string name;
  int channels;
  bool inplace;
  bool useCtx;
};

// 16sc tests
class SubC16scTest : public NppTestBase, public ::testing::WithParamInterface<SubCComplexParam> {};

TEST_P(SubC16scTest, ComplexSub) {
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
      expected[i] = src[i];
    } else {
      int constIdx = ch;
      int diffRe = src[i].re - aConstants[constIdx].re;
      int diffIm = src[i].im - aConstants[constIdx].im;
      expected[i].re = static_cast<Npp16s>(std::min(32767, std::max(-32768, diffRe >> scaleFactor)));
      expected[i].im = static_cast<Npp16s>(std::min(32767, std::max(-32768, diffIm >> scaleFactor)));
    }
  }

  NppImageMemory<Npp16sc> d_src(width * actualChannels, height);
  NppImageMemory<Npp16sc> d_dst(width * actualChannels, height);
  d_src.copyFromHost(src);
  if (p.channels == -4)
    d_dst.copyFromHost(src);

  NppiSize roi = {width, height};
  NppStatus status;
  NppStreamContext ctx;
  if (p.useCtx)
    nppGetStreamContext(&ctx);

  if (p.inplace) {
    d_dst.copyFromHost(src);
    if (p.channels == 1) {
      status = p.useCtx ? nppiSubC_16sc_C1IRSfs_Ctx(aConstants[0], d_dst.get(), d_dst.step(), roi, scaleFactor, ctx)
                        : nppiSubC_16sc_C1IRSfs(aConstants[0], d_dst.get(), d_dst.step(), roi, scaleFactor);
    } else if (p.channels == 3) {
      status = p.useCtx ? nppiSubC_16sc_C3IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx)
                        : nppiSubC_16sc_C3IRSfs(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor);
    } else if (p.channels == -4) {
      status = p.useCtx
                   ? nppiSubC_16sc_AC4IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx)
                   : nppiSubC_16sc_AC4IRSfs(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor);
    }
  } else {
    if (p.channels == 1) {
      status = p.useCtx ? nppiSubC_16sc_C1RSfs_Ctx(d_src.get(), d_src.step(), aConstants[0], d_dst.get(), d_dst.step(),
                                                   roi, scaleFactor, ctx)
                        : nppiSubC_16sc_C1RSfs(d_src.get(), d_src.step(), aConstants[0], d_dst.get(), d_dst.step(), roi,
                                               scaleFactor);
    } else if (p.channels == 3) {
      status = p.useCtx ? nppiSubC_16sc_C3RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(),
                                                   d_dst.step(), roi, scaleFactor, ctx)
                        : nppiSubC_16sc_C3RSfs(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(),
                                               roi, scaleFactor);
    } else if (p.channels == -4) {
      status = p.useCtx ? nppiSubC_16sc_AC4RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(),
                                                    d_dst.step(), roi, scaleFactor, ctx)
                        : nppiSubC_16sc_AC4RSfs(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(),
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
    SubC16scSuite, SubC16scTest,
    ::testing::Values(SubCComplexParam{"C1R", 1, false, false}, SubCComplexParam{"C1R_Ctx", 1, false, true},
                      SubCComplexParam{"C1IR", 1, true, false}, SubCComplexParam{"C1IR_Ctx", 1, true, true},
                      SubCComplexParam{"C3R", 3, false, false}, SubCComplexParam{"C3R_Ctx", 3, false, true},
                      SubCComplexParam{"C3IR", 3, true, false}, SubCComplexParam{"C3IR_Ctx", 3, true, true},
                      SubCComplexParam{"AC4R", -4, false, false}, SubCComplexParam{"AC4R_Ctx", -4, false, true},
                      SubCComplexParam{"AC4IR", -4, true, false}, SubCComplexParam{"AC4IR_Ctx", -4, true, true}),
    [](const ::testing::TestParamInfo<SubCComplexParam> &info) { return info.param.name; });

// 32sc tests
class SubC32scTest : public NppTestBase, public ::testing::WithParamInterface<SubCComplexParam> {};

TEST_P(SubC32scTest, ComplexSub) {
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
      int64_t diffRe = (int64_t)src[i].re - aConstants[ch].re;
      int64_t diffIm = (int64_t)src[i].im - aConstants[ch].im;
      expected[i].re = static_cast<Npp32s>(diffRe >> scaleFactor);
      expected[i].im = static_cast<Npp32s>(diffIm >> scaleFactor);
    }
  }

  NppImageMemory<Npp32sc> d_src(width * actualChannels, height);
  NppImageMemory<Npp32sc> d_dst(width * actualChannels, height);
  d_src.copyFromHost(src);
  if (p.channels == -4)
    d_dst.copyFromHost(src);

  NppiSize roi = {width, height};
  NppStatus status;
  NppStreamContext ctx;
  if (p.useCtx)
    nppGetStreamContext(&ctx);

  if (p.inplace) {
    d_dst.copyFromHost(src);
    if (p.channels == 1) {
      status = p.useCtx ? nppiSubC_32sc_C1IRSfs_Ctx(aConstants[0], d_dst.get(), d_dst.step(), roi, scaleFactor, ctx)
                        : nppiSubC_32sc_C1IRSfs(aConstants[0], d_dst.get(), d_dst.step(), roi, scaleFactor);
    } else if (p.channels == 3) {
      status = p.useCtx ? nppiSubC_32sc_C3IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx)
                        : nppiSubC_32sc_C3IRSfs(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor);
    } else if (p.channels == -4) {
      status = p.useCtx
                   ? nppiSubC_32sc_AC4IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx)
                   : nppiSubC_32sc_AC4IRSfs(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor);
    }
  } else {
    if (p.channels == 1) {
      status = p.useCtx ? nppiSubC_32sc_C1RSfs_Ctx(d_src.get(), d_src.step(), aConstants[0], d_dst.get(), d_dst.step(),
                                                   roi, scaleFactor, ctx)
                        : nppiSubC_32sc_C1RSfs(d_src.get(), d_src.step(), aConstants[0], d_dst.get(), d_dst.step(), roi,
                                               scaleFactor);
    } else if (p.channels == 3) {
      status = p.useCtx ? nppiSubC_32sc_C3RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(),
                                                   d_dst.step(), roi, scaleFactor, ctx)
                        : nppiSubC_32sc_C3RSfs(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(),
                                               roi, scaleFactor);
    } else if (p.channels == -4) {
      status = p.useCtx ? nppiSubC_32sc_AC4RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(),
                                                    d_dst.step(), roi, scaleFactor, ctx)
                        : nppiSubC_32sc_AC4RSfs(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(),
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
    SubC32scSuite, SubC32scTest,
    ::testing::Values(SubCComplexParam{"C1R", 1, false, false}, SubCComplexParam{"C1R_Ctx", 1, false, true},
                      SubCComplexParam{"C1IR", 1, true, false}, SubCComplexParam{"C1IR_Ctx", 1, true, true},
                      SubCComplexParam{"C3R", 3, false, false}, SubCComplexParam{"C3R_Ctx", 3, false, true},
                      SubCComplexParam{"C3IR", 3, true, false}, SubCComplexParam{"C3IR_Ctx", 3, true, true},
                      SubCComplexParam{"AC4R", -4, false, false}, SubCComplexParam{"AC4R_Ctx", -4, false, true},
                      SubCComplexParam{"AC4IR", -4, true, false}, SubCComplexParam{"AC4IR_Ctx", -4, true, true}),
    [](const ::testing::TestParamInfo<SubCComplexParam> &info) { return info.param.name; });

// 32fc tests
class SubC32fcTest : public NppTestBase, public ::testing::WithParamInterface<SubCComplexParam> {};

TEST_P(SubC32fcTest, ComplexSub) {
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
      expected[i].re = src[i].re - aConstants[ch].re;
      expected[i].im = src[i].im - aConstants[ch].im;
    }
  }

  NppImageMemory<Npp32fc> d_src(width * actualChannels, height);
  NppImageMemory<Npp32fc> d_dst(width * actualChannels, height);
  d_src.copyFromHost(src);
  if (p.channels == -4)
    d_dst.copyFromHost(src);

  NppiSize roi = {width, height};
  NppStatus status;
  NppStreamContext ctx;
  if (p.useCtx)
    nppGetStreamContext(&ctx);

  if (p.inplace) {
    d_dst.copyFromHost(src);
    if (p.channels == 1) {
      status = p.useCtx ? nppiSubC_32fc_C1IR_Ctx(aConstants[0], d_dst.get(), d_dst.step(), roi, ctx)
                        : nppiSubC_32fc_C1IR(aConstants[0], d_dst.get(), d_dst.step(), roi);
    } else if (p.channels == 3) {
      status = p.useCtx ? nppiSubC_32fc_C3IR_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, ctx)
                        : nppiSubC_32fc_C3IR(aConstants.data(), d_dst.get(), d_dst.step(), roi);
    } else if (p.channels == 4) {
      status = p.useCtx ? nppiSubC_32fc_C4IR_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, ctx)
                        : nppiSubC_32fc_C4IR(aConstants.data(), d_dst.get(), d_dst.step(), roi);
    } else if (p.channels == -4) {
      status = p.useCtx ? nppiSubC_32fc_AC4IR_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, ctx)
                        : nppiSubC_32fc_AC4IR(aConstants.data(), d_dst.get(), d_dst.step(), roi);
    }
  } else {
    if (p.channels == 1) {
      status = p.useCtx ? nppiSubC_32fc_C1R_Ctx(d_src.get(), d_src.step(), aConstants[0], d_dst.get(), d_dst.step(),
                                                roi, ctx)
                        : nppiSubC_32fc_C1R(d_src.get(), d_src.step(), aConstants[0], d_dst.get(), d_dst.step(), roi);
    } else if (p.channels == 3) {
      status =
          p.useCtx
              ? nppiSubC_32fc_C3R_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi, ctx)
              : nppiSubC_32fc_C3R(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi);
    } else if (p.channels == 4) {
      status =
          p.useCtx
              ? nppiSubC_32fc_C4R_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi, ctx)
              : nppiSubC_32fc_C4R(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi);
    } else if (p.channels == -4) {
      status = p.useCtx
                   ? nppiSubC_32fc_AC4R_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(),
                                            roi, ctx)
                   : nppiSubC_32fc_AC4R(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi);
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
    SubC32fcSuite, SubC32fcTest,
    ::testing::Values(SubCComplexParam{"C1R", 1, false, false}, SubCComplexParam{"C1R_Ctx", 1, false, true},
                      SubCComplexParam{"C1IR", 1, true, false}, SubCComplexParam{"C1IR_Ctx", 1, true, true},
                      SubCComplexParam{"C3R", 3, false, false}, SubCComplexParam{"C3R_Ctx", 3, false, true},
                      SubCComplexParam{"C3IR", 3, true, false}, SubCComplexParam{"C3IR_Ctx", 3, true, true},
                      SubCComplexParam{"C4R", 4, false, false}, SubCComplexParam{"C4R_Ctx", 4, false, true},
                      SubCComplexParam{"C4IR", 4, true, false}, SubCComplexParam{"C4IR_Ctx", 4, true, true},
                      SubCComplexParam{"AC4R", -4, false, false}, SubCComplexParam{"AC4R_Ctx", -4, false, true},
                      SubCComplexParam{"AC4IR", -4, true, false}, SubCComplexParam{"AC4IR_Ctx", -4, true, true}),
    [](const ::testing::TestParamInfo<SubCComplexParam> &info) { return info.param.name; });

// ==================== SubC Ctx variants for integer types ====================

struct SubCIntCtxParam {
  std::string name;
  std::string dtype;
  int channels;
  bool inplace;
};

class SubCIntCtxTest : public NppTestBase, public ::testing::WithParamInterface<SubCIntCtxParam> {};

TEST_P(SubCIntCtxTest, CtxVariant) {
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
    TestDataGenerator::generateRandom(src, (Npp8u)50, (Npp8u)255, 12345);

    const int constCount = (p.channels == -4) ? 3 : actualChannels;
    std::vector<Npp8u> aConstants(constCount);
    for (int i = 0; i < constCount; i++)
      aConstants[i] = static_cast<Npp8u>(10 * (i + 1));

    int scaleFactor = 0;
    for (int i = 0; i < total; i++) {
      int ch = i % actualChannels;
      if (p.channels == -4 && ch == 3) {
        expected[i] = src[i];
      } else {
        int diff = src[i] - aConstants[ch];
        expected[i] = static_cast<Npp8u>(std::min(255, std::max(0, diff >> scaleFactor)));
      }
    }

    NppImageMemory<Npp8u> d_src(width * actualChannels, height);
    NppImageMemory<Npp8u> d_dst(width * actualChannels, height);
    d_src.copyFromHost(src);
    if (p.channels == -4)
      d_dst.copyFromHost(src);

    if (p.inplace) {
      d_dst.copyFromHost(src);
      if (p.channels == 3)
        status = nppiSubC_8u_C3IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      else if (p.channels == 4)
        status = nppiSubC_8u_C4IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      else if (p.channels == -4)
        status = nppiSubC_8u_AC4IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
    } else {
      if (p.channels == 3)
        status = nppiSubC_8u_C3RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi,
                                        scaleFactor, ctx);
      else if (p.channels == 4)
        status = nppiSubC_8u_C4RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi,
                                        scaleFactor, ctx);
      else if (p.channels == -4)
        status = nppiSubC_8u_AC4RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi,
                                         scaleFactor, ctx);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> result(total);
    d_dst.copyToHost(result);
    EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
  } else if (p.dtype == "16u") {
    const int total = width * height * actualChannels;
    std::vector<Npp16u> src(total), expected(total);
    TestDataGenerator::generateRandom(src, (Npp16u)1000, (Npp16u)65000, 12345);

    const int constCount = (p.channels == -4) ? 3 : actualChannels;
    std::vector<Npp16u> aConstants(constCount);
    for (int i = 0; i < constCount; i++)
      aConstants[i] = static_cast<Npp16u>(100 * (i + 1));

    int scaleFactor = 0;
    for (int i = 0; i < total; i++) {
      int ch = i % actualChannels;
      if (p.channels == -4 && ch == 3) {
        expected[i] = src[i];
      } else {
        int diff = src[i] - aConstants[ch];
        expected[i] = static_cast<Npp16u>(std::min(65535, std::max(0, diff >> scaleFactor)));
      }
    }

    NppImageMemory<Npp16u> d_src(width * actualChannels, height);
    NppImageMemory<Npp16u> d_dst(width * actualChannels, height);
    d_src.copyFromHost(src);
    if (p.channels == -4)
      d_dst.copyFromHost(src);

    if (p.inplace) {
      d_dst.copyFromHost(src);
      if (p.channels == 3)
        status = nppiSubC_16u_C3IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      else if (p.channels == 4)
        status = nppiSubC_16u_C4IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      else if (p.channels == -4)
        status = nppiSubC_16u_AC4IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
    } else {
      if (p.channels == 3)
        status = nppiSubC_16u_C3RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi,
                                         scaleFactor, ctx);
      else if (p.channels == 4)
        status = nppiSubC_16u_C4RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi,
                                         scaleFactor, ctx);
      else if (p.channels == -4)
        status = nppiSubC_16u_AC4RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi,
                                          scaleFactor, ctx);
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
    for (int i = 0; i < constCount; i++)
      aConstants[i] = static_cast<Npp16s>(100 * (i + 1) * ((i % 2 == 0) ? 1 : -1));

    int scaleFactor = 0;
    for (int i = 0; i < total; i++) {
      int ch = i % actualChannels;
      if (p.channels == -4 && ch == 3) {
        expected[i] = src[i];
      } else {
        int diff = src[i] - aConstants[ch];
        expected[i] = static_cast<Npp16s>(std::min(32767, std::max(-32768, diff >> scaleFactor)));
      }
    }

    NppImageMemory<Npp16s> d_src(width * actualChannels, height);
    NppImageMemory<Npp16s> d_dst(width * actualChannels, height);
    d_src.copyFromHost(src);
    if (p.channels == -4)
      d_dst.copyFromHost(src);

    if (p.inplace) {
      d_dst.copyFromHost(src);
      if (p.channels == 1)
        status = nppiSubC_16s_C1IRSfs_Ctx(aConstants[0], d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      else if (p.channels == 3)
        status = nppiSubC_16s_C3IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      else if (p.channels == 4)
        status = nppiSubC_16s_C4IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      else if (p.channels == -4)
        status = nppiSubC_16s_AC4IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
    } else {
      if (p.channels == 1)
        status = nppiSubC_16s_C1RSfs_Ctx(d_src.get(), d_src.step(), aConstants[0], d_dst.get(), d_dst.step(), roi,
                                         scaleFactor, ctx);
      else if (p.channels == 3)
        status = nppiSubC_16s_C3RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi,
                                         scaleFactor, ctx);
      else if (p.channels == 4)
        status = nppiSubC_16s_C4RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi,
                                         scaleFactor, ctx);
      else if (p.channels == -4)
        status = nppiSubC_16s_AC4RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi,
                                          scaleFactor, ctx);
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
    for (int i = 0; i < constCount; i++)
      aConstants[i] = 100 * (i + 1) * ((i % 2 == 0) ? 1 : -1);

    int scaleFactor = 0;
    for (int i = 0; i < total; i++) {
      int ch = i % actualChannels;
      if (p.channels == -4 && ch == 3) {
        expected[i] = src[i];
      } else {
        int64_t diff = (int64_t)src[i] - aConstants[ch];
        expected[i] = static_cast<Npp32s>(diff >> scaleFactor);
      }
    }

    NppImageMemory<Npp32s> d_src(width * actualChannels, height);
    NppImageMemory<Npp32s> d_dst(width * actualChannels, height);
    d_src.copyFromHost(src);

    if (p.inplace) {
      d_dst.copyFromHost(src);
      if (p.channels == 1)
        status = nppiSubC_32s_C1IRSfs_Ctx(aConstants[0], d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      else if (p.channels == 3)
        status = nppiSubC_32s_C3IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
    } else {
      if (p.channels == 1)
        status = nppiSubC_32s_C1RSfs_Ctx(d_src.get(), d_src.step(), aConstants[0], d_dst.get(), d_dst.step(), roi,
                                         scaleFactor, ctx);
      else if (p.channels == 3)
        status = nppiSubC_32s_C3RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi,
                                         scaleFactor, ctx);
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
    for (int i = 0; i < constCount; i++)
      aConstants[i] = 1.5f * (i + 1) * ((i % 2 == 0) ? 1 : -1);

    for (int i = 0; i < total; i++) {
      int ch = i % actualChannels;
      if (p.channels == -4 && ch == 3) {
        expected[i] = src[i];
      } else {
        expected[i] = src[i] - aConstants[ch];
      }
    }

    NppImageMemory<Npp32f> d_src(width * actualChannels, height);
    NppImageMemory<Npp32f> d_dst(width * actualChannels, height);
    d_src.copyFromHost(src);
    if (p.channels == -4)
      d_dst.copyFromHost(src);

    if (p.inplace) {
      d_dst.copyFromHost(src);
      if (p.channels == 3)
        status = nppiSubC_32f_C3IR_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, ctx);
      else if (p.channels == 4)
        status = nppiSubC_32f_C4IR_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, ctx);
      else if (p.channels == -4)
        status = nppiSubC_32f_AC4IR_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, ctx);
    } else {
      if (p.channels == 3)
        status =
            nppiSubC_32f_C3R_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi, ctx);
      else if (p.channels == 4)
        status =
            nppiSubC_32f_C4R_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi, ctx);
      else if (p.channels == -4)
        status =
            nppiSubC_32f_AC4R_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi, ctx);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> result(total);
    d_dst.copyToHost(result);
    EXPECT_TRUE(ResultValidator::arraysEqual(result, expected, 1e-5f));
  }
}

INSTANTIATE_TEST_SUITE_P(
    SubCIntCtxSuite, SubCIntCtxTest,
    ::testing::Values(
        SubCIntCtxParam{"8u_C3RSfs_Ctx", "8u", 3, false}, SubCIntCtxParam{"8u_C4RSfs_Ctx", "8u", 4, false},
        SubCIntCtxParam{"8u_AC4RSfs_Ctx", "8u", -4, false}, SubCIntCtxParam{"8u_C3IRSfs_Ctx", "8u", 3, true},
        SubCIntCtxParam{"8u_C4IRSfs_Ctx", "8u", 4, true}, SubCIntCtxParam{"8u_AC4IRSfs_Ctx", "8u", -4, true},
        SubCIntCtxParam{"16u_C3RSfs_Ctx", "16u", 3, false}, SubCIntCtxParam{"16u_C4RSfs_Ctx", "16u", 4, false},
        SubCIntCtxParam{"16u_AC4RSfs_Ctx", "16u", -4, false}, SubCIntCtxParam{"16u_C3IRSfs_Ctx", "16u", 3, true},
        SubCIntCtxParam{"16u_C4IRSfs_Ctx", "16u", 4, true}, SubCIntCtxParam{"16u_AC4IRSfs_Ctx", "16u", -4, true},
        SubCIntCtxParam{"16s_C1RSfs_Ctx", "16s", 1, false}, SubCIntCtxParam{"16s_C3RSfs_Ctx", "16s", 3, false},
        SubCIntCtxParam{"16s_C4RSfs_Ctx", "16s", 4, false}, SubCIntCtxParam{"16s_AC4RSfs_Ctx", "16s", -4, false},
        SubCIntCtxParam{"16s_C1IRSfs_Ctx", "16s", 1, true}, SubCIntCtxParam{"16s_C3IRSfs_Ctx", "16s", 3, true},
        SubCIntCtxParam{"16s_C4IRSfs_Ctx", "16s", 4, true}, SubCIntCtxParam{"16s_AC4IRSfs_Ctx", "16s", -4, true},
        SubCIntCtxParam{"32s_C1RSfs_Ctx", "32s", 1, false}, SubCIntCtxParam{"32s_C3RSfs_Ctx", "32s", 3, false},
        SubCIntCtxParam{"32s_C1IRSfs_Ctx", "32s", 1, true}, SubCIntCtxParam{"32s_C3IRSfs_Ctx", "32s", 3, true},
        SubCIntCtxParam{"32f_C3R_Ctx", "32f", 3, false}, SubCIntCtxParam{"32f_C4R_Ctx", "32f", 4, false},
        SubCIntCtxParam{"32f_AC4R_Ctx", "32f", -4, false}, SubCIntCtxParam{"32f_C3IR_Ctx", "32f", 3, true},
        SubCIntCtxParam{"32f_C4IR_Ctx", "32f", 4, true}, SubCIntCtxParam{"32f_AC4IR_Ctx", "32f", -4, true}),
    [](const ::testing::TestParamInfo<SubCIntCtxParam> &info) { return info.param.name; });

// ==================== SubC Non-Ctx integer inplace variants ====================

class SubC16sInplaceTest : public NppTestBase {};

TEST_F(SubC16sInplaceTest, SubC_16s_C1IRSfs) {
  const int width = 32, height = 32, total = width * height;
  std::vector<Npp16s> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp16s)-10000, (Npp16s)10000, 12345);

  Npp16s nConstant = 500;
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int diff = src[i] - nConstant;
    expected[i] = static_cast<Npp16s>(std::min(32767, std::max(-32768, diff >> scaleFactor)));
  }

  NppImageMemory<Npp16s> d_srcdst(width, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiSubC_16s_C1IRSfs(nConstant, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp16s> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(SubC16sInplaceTest, SubC_16s_C3IRSfs) {
  const int width = 32, height = 32, channels = 3, total = width * height * channels;
  std::vector<Npp16s> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp16s)-10000, (Npp16s)10000, 12345);

  Npp16s aConstants[3] = {100, -200, 300};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int diff = src[i] - aConstants[i % 3];
    expected[i] = static_cast<Npp16s>(std::min(32767, std::max(-32768, diff >> scaleFactor)));
  }

  NppImageMemory<Npp16s> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiSubC_16s_C3IRSfs(aConstants, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp16s> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(SubC16sInplaceTest, SubC_16s_C4IRSfs) {
  const int width = 32, height = 32, channels = 4, total = width * height * channels;
  std::vector<Npp16s> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp16s)-10000, (Npp16s)10000, 12345);

  Npp16s aConstants[4] = {100, -200, 300, -400};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int diff = src[i] - aConstants[i % 4];
    expected[i] = static_cast<Npp16s>(std::min(32767, std::max(-32768, diff >> scaleFactor)));
  }

  NppImageMemory<Npp16s> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiSubC_16s_C4IRSfs(aConstants, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp16s> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(SubC16sInplaceTest, SubC_16s_AC4IRSfs) {
  const int width = 32, height = 32, channels = 4, total = width * height * channels;
  std::vector<Npp16s> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp16s)-10000, (Npp16s)10000, 12345);

  Npp16s aConstants[3] = {100, -200, 300};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int ch = i % 4;
    if (ch == 3) {
      expected[i] = src[i];
    } else {
      int diff = src[i] - aConstants[ch];
      expected[i] = static_cast<Npp16s>(std::min(32767, std::max(-32768, diff >> scaleFactor)));
    }
  }

  NppImageMemory<Npp16s> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiSubC_16s_AC4IRSfs(aConstants, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp16s> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

class SubC16uInplaceTest : public NppTestBase {};

TEST_F(SubC16uInplaceTest, SubC_16u_C3IRSfs) {
  const int width = 32, height = 32, channels = 3, total = width * height * channels;
  std::vector<Npp16u> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp16u)1000, (Npp16u)65000, 12345);

  Npp16u aConstants[3] = {100, 200, 300};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int diff = src[i] - aConstants[i % 3];
    expected[i] = static_cast<Npp16u>(std::min(65535, std::max(0, diff >> scaleFactor)));
  }

  NppImageMemory<Npp16u> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiSubC_16u_C3IRSfs(aConstants, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp16u> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(SubC16uInplaceTest, SubC_16u_C4IRSfs) {
  const int width = 32, height = 32, channels = 4, total = width * height * channels;
  std::vector<Npp16u> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp16u)1000, (Npp16u)65000, 12345);

  Npp16u aConstants[4] = {100, 200, 300, 400};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int diff = src[i] - aConstants[i % 4];
    expected[i] = static_cast<Npp16u>(std::min(65535, std::max(0, diff >> scaleFactor)));
  }

  NppImageMemory<Npp16u> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiSubC_16u_C4IRSfs(aConstants, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp16u> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(SubC16uInplaceTest, SubC_16u_AC4IRSfs) {
  const int width = 32, height = 32, channels = 4, total = width * height * channels;
  std::vector<Npp16u> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp16u)1000, (Npp16u)65000, 12345);

  Npp16u aConstants[3] = {100, 200, 300};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int ch = i % 4;
    if (ch == 3) {
      expected[i] = src[i];
    } else {
      int diff = src[i] - aConstants[ch];
      expected[i] = static_cast<Npp16u>(std::min(65535, std::max(0, diff >> scaleFactor)));
    }
  }

  NppImageMemory<Npp16u> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiSubC_16u_AC4IRSfs(aConstants, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp16u> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

class SubC32sInplaceTest : public NppTestBase {};

TEST_F(SubC32sInplaceTest, SubC_32s_C1IRSfs) {
  const int width = 32, height = 32, total = width * height;
  std::vector<Npp32s> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp32s)-100000, (Npp32s)100000, 12345);

  Npp32s nConstant = 500;
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int64_t diff = (int64_t)src[i] - nConstant;
    expected[i] = static_cast<Npp32s>(diff >> scaleFactor);
  }

  NppImageMemory<Npp32s> d_srcdst(width, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiSubC_32s_C1IRSfs(nConstant, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp32s> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(SubC32sInplaceTest, SubC_32s_C3IRSfs) {
  const int width = 32, height = 32, channels = 3, total = width * height * channels;
  std::vector<Npp32s> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp32s)-100000, (Npp32s)100000, 12345);

  Npp32s aConstants[3] = {100, -200, 300};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int64_t diff = (int64_t)src[i] - aConstants[i % 3];
    expected[i] = static_cast<Npp32s>(diff >> scaleFactor);
  }

  NppImageMemory<Npp32s> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiSubC_32s_C3IRSfs(aConstants, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp32s> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}
