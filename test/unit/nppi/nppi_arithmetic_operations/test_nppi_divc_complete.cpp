#include <gtest/gtest.h>
#include <npp.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <cmath>
#include "npp_test_base.h"

using namespace npp_functional_test;

// ==================== DivC Complex Types Tests (16sc, 32sc, 32fc) ====================

struct DivCComplexParam {
  std::string name;
  int channels;
  bool inplace;
  bool useCtx;
};

// 16sc tests - complex division: (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c^2+d^2)
class DivC16scTest : public NppTestBase,
                     public ::testing::WithParamInterface<DivCComplexParam> {};

TEST_P(DivC16scTest, ComplexDiv) {
  const auto& p = GetParam();
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
    aConstants[i].re = static_cast<Npp16s>(2 * (i + 1));
    aConstants[i].im = static_cast<Npp16s>(i + 1);
  }

  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int ch = i % actualChannels;
    if (p.channels == -4 && ch == 3) {
      expected[i] = src[i];
    } else {
      // Complex division: (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c^2+d^2)
      float denom = (float)(aConstants[ch].re * aConstants[ch].re + aConstants[ch].im * aConstants[ch].im);
      if (denom != 0) {
        float quotRe = (src[i].re * aConstants[ch].re + src[i].im * aConstants[ch].im) / denom;
        float quotIm = (src[i].im * aConstants[ch].re - src[i].re * aConstants[ch].im) / denom;
        expected[i].re = static_cast<Npp16s>(std::round(quotRe));
        expected[i].im = static_cast<Npp16s>(std::round(quotIm));
      } else {
        expected[i].re = 0;
        expected[i].im = 0;
      }
    }
  }

  NppImageMemory<Npp16sc> d_src(width * actualChannels, height);
  NppImageMemory<Npp16sc> d_dst(width * actualChannels, height);
  d_src.copyFromHost(src);
  if (p.channels == -4) d_dst.copyFromHost(src);

  NppiSize roi = {width, height};
  NppStatus status;
  NppStreamContext ctx;
  if (p.useCtx) nppGetStreamContext(&ctx);

  if (p.inplace) {
    d_dst.copyFromHost(src);
    if (p.channels == 1) {
      status = p.useCtx ? nppiDivC_16sc_C1IRSfs_Ctx(aConstants[0], d_dst.get(), d_dst.step(), roi, scaleFactor, ctx)
                        : nppiDivC_16sc_C1IRSfs(aConstants[0], d_dst.get(), d_dst.step(), roi, scaleFactor);
    } else if (p.channels == 3) {
      status = p.useCtx ? nppiDivC_16sc_C3IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx)
                        : nppiDivC_16sc_C3IRSfs(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor);
    } else if (p.channels == -4) {
      status = p.useCtx ? nppiDivC_16sc_AC4IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx)
                        : nppiDivC_16sc_AC4IRSfs(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor);
    }
  } else {
    if (p.channels == 1) {
      status = p.useCtx ? nppiDivC_16sc_C1RSfs_Ctx(d_src.get(), d_src.step(), aConstants[0], d_dst.get(), d_dst.step(), roi, scaleFactor, ctx)
                        : nppiDivC_16sc_C1RSfs(d_src.get(), d_src.step(), aConstants[0], d_dst.get(), d_dst.step(), roi, scaleFactor);
    } else if (p.channels == 3) {
      status = p.useCtx ? nppiDivC_16sc_C3RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx)
                        : nppiDivC_16sc_C3RSfs(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor);
    } else if (p.channels == -4) {
      status = p.useCtx ? nppiDivC_16sc_AC4RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx)
                        : nppiDivC_16sc_AC4RSfs(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor);
    }
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp16sc> result(total);
  d_dst.copyToHost(result);

  for (int i = 0; i < total; i++) {
    EXPECT_LE(std::abs(result[i].re - expected[i].re), 1) << "Real mismatch at " << i;
    EXPECT_LE(std::abs(result[i].im - expected[i].im), 1) << "Imag mismatch at " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(
    DivC16scSuite, DivC16scTest,
    ::testing::Values(
        DivCComplexParam{"C1R", 1, false, false},
        DivCComplexParam{"C1R_Ctx", 1, false, true},
        DivCComplexParam{"C1IR", 1, true, false},
        DivCComplexParam{"C1IR_Ctx", 1, true, true},
        DivCComplexParam{"C3R", 3, false, false},
        DivCComplexParam{"C3R_Ctx", 3, false, true},
        DivCComplexParam{"C3IR", 3, true, false},
        DivCComplexParam{"C3IR_Ctx", 3, true, true},
        DivCComplexParam{"AC4R", -4, false, false},
        DivCComplexParam{"AC4R_Ctx", -4, false, true},
        DivCComplexParam{"AC4IR", -4, true, false},
        DivCComplexParam{"AC4IR_Ctx", -4, true, true}
    ),
    [](const ::testing::TestParamInfo<DivCComplexParam>& info) { return info.param.name; }
);

// 32sc tests
class DivC32scTest : public NppTestBase,
                     public ::testing::WithParamInterface<DivCComplexParam> {};

TEST_P(DivC32scTest, ComplexDiv) {
  const auto& p = GetParam();
  const int width = 32, height = 32;
  const int actualChannels = (p.channels == -4) ? 4 : p.channels;
  const int total = width * height * actualChannels;

  std::vector<Npp32sc> src(total), expected(total);
  std::mt19937 gen(12345);
  std::uniform_int_distribution<int> dist(-10000, 10000);
  for (int i = 0; i < total; i++) {
    src[i].re = dist(gen);
    src[i].im = dist(gen);
  }

  const int constCount = (p.channels == -4) ? 3 : p.channels;
  std::vector<Npp32sc> aConstants(constCount);
  for (int i = 0; i < constCount; i++) {
    aConstants[i].re = 2 * (i + 1);
    aConstants[i].im = i + 1;
  }

  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int ch = i % actualChannels;
    if (p.channels == -4 && ch == 3) {
      expected[i] = src[i];
    } else {
      double denom = (double)aConstants[ch].re * aConstants[ch].re + (double)aConstants[ch].im * aConstants[ch].im;
      if (denom != 0) {
        double quotRe = ((double)src[i].re * aConstants[ch].re + (double)src[i].im * aConstants[ch].im) / denom;
        double quotIm = ((double)src[i].im * aConstants[ch].re - (double)src[i].re * aConstants[ch].im) / denom;
        expected[i].re = static_cast<Npp32s>(std::round(quotRe));
        expected[i].im = static_cast<Npp32s>(std::round(quotIm));
      } else {
        expected[i].re = 0;
        expected[i].im = 0;
      }
    }
  }

  NppImageMemory<Npp32sc> d_src(width * actualChannels, height);
  NppImageMemory<Npp32sc> d_dst(width * actualChannels, height);
  d_src.copyFromHost(src);
  if (p.channels == -4) d_dst.copyFromHost(src);

  NppiSize roi = {width, height};
  NppStatus status;
  NppStreamContext ctx;
  if (p.useCtx) nppGetStreamContext(&ctx);

  if (p.inplace) {
    d_dst.copyFromHost(src);
    if (p.channels == 1) {
      status = p.useCtx ? nppiDivC_32sc_C1IRSfs_Ctx(aConstants[0], d_dst.get(), d_dst.step(), roi, scaleFactor, ctx)
                        : nppiDivC_32sc_C1IRSfs(aConstants[0], d_dst.get(), d_dst.step(), roi, scaleFactor);
    } else if (p.channels == 3) {
      status = p.useCtx ? nppiDivC_32sc_C3IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx)
                        : nppiDivC_32sc_C3IRSfs(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor);
    } else if (p.channels == -4) {
      status = p.useCtx ? nppiDivC_32sc_AC4IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx)
                        : nppiDivC_32sc_AC4IRSfs(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor);
    }
  } else {
    if (p.channels == 1) {
      status = p.useCtx ? nppiDivC_32sc_C1RSfs_Ctx(d_src.get(), d_src.step(), aConstants[0], d_dst.get(), d_dst.step(), roi, scaleFactor, ctx)
                        : nppiDivC_32sc_C1RSfs(d_src.get(), d_src.step(), aConstants[0], d_dst.get(), d_dst.step(), roi, scaleFactor);
    } else if (p.channels == 3) {
      status = p.useCtx ? nppiDivC_32sc_C3RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx)
                        : nppiDivC_32sc_C3RSfs(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor);
    } else if (p.channels == -4) {
      status = p.useCtx ? nppiDivC_32sc_AC4RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx)
                        : nppiDivC_32sc_AC4RSfs(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor);
    }
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32sc> result(total);
  d_dst.copyToHost(result);

  for (int i = 0; i < total; i++) {
    EXPECT_LE(std::abs(result[i].re - expected[i].re), 1) << "Real mismatch at " << i;
    EXPECT_LE(std::abs(result[i].im - expected[i].im), 1) << "Imag mismatch at " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(
    DivC32scSuite, DivC32scTest,
    ::testing::Values(
        DivCComplexParam{"C1R", 1, false, false},
        DivCComplexParam{"C1R_Ctx", 1, false, true},
        DivCComplexParam{"C1IR", 1, true, false},
        DivCComplexParam{"C1IR_Ctx", 1, true, true},
        DivCComplexParam{"C3R", 3, false, false},
        DivCComplexParam{"C3R_Ctx", 3, false, true},
        DivCComplexParam{"C3IR", 3, true, false},
        DivCComplexParam{"C3IR_Ctx", 3, true, true},
        DivCComplexParam{"AC4R", -4, false, false},
        DivCComplexParam{"AC4R_Ctx", -4, false, true},
        DivCComplexParam{"AC4IR", -4, true, false},
        DivCComplexParam{"AC4IR_Ctx", -4, true, true}
    ),
    [](const ::testing::TestParamInfo<DivCComplexParam>& info) { return info.param.name; }
);

// 32fc tests
class DivC32fcTest : public NppTestBase,
                     public ::testing::WithParamInterface<DivCComplexParam> {};

TEST_P(DivC32fcTest, ComplexDiv) {
  const auto& p = GetParam();
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
    aConstants[i].re = 2.0f * (i + 1);
    aConstants[i].im = 1.0f * (i + 1);
  }

  for (int i = 0; i < total; i++) {
    int ch = i % actualChannels;
    if (p.channels == -4 && ch == 3) {
      expected[i] = src[i];
    } else {
      float denom = aConstants[ch].re * aConstants[ch].re + aConstants[ch].im * aConstants[ch].im;
      expected[i].re = (src[i].re * aConstants[ch].re + src[i].im * aConstants[ch].im) / denom;
      expected[i].im = (src[i].im * aConstants[ch].re - src[i].re * aConstants[ch].im) / denom;
    }
  }

  NppImageMemory<Npp32fc> d_src(width * actualChannels, height);
  NppImageMemory<Npp32fc> d_dst(width * actualChannels, height);
  d_src.copyFromHost(src);
  if (p.channels == -4) d_dst.copyFromHost(src);

  NppiSize roi = {width, height};
  NppStatus status;
  NppStreamContext ctx;
  if (p.useCtx) nppGetStreamContext(&ctx);

  if (p.inplace) {
    d_dst.copyFromHost(src);
    if (p.channels == 1) {
      status = p.useCtx ? nppiDivC_32fc_C1IR_Ctx(aConstants[0], d_dst.get(), d_dst.step(), roi, ctx)
                        : nppiDivC_32fc_C1IR(aConstants[0], d_dst.get(), d_dst.step(), roi);
    } else if (p.channels == 3) {
      status = p.useCtx ? nppiDivC_32fc_C3IR_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, ctx)
                        : nppiDivC_32fc_C3IR(aConstants.data(), d_dst.get(), d_dst.step(), roi);
    } else if (p.channels == 4) {
      status = p.useCtx ? nppiDivC_32fc_C4IR_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, ctx)
                        : nppiDivC_32fc_C4IR(aConstants.data(), d_dst.get(), d_dst.step(), roi);
    } else if (p.channels == -4) {
      status = p.useCtx ? nppiDivC_32fc_AC4IR_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, ctx)
                        : nppiDivC_32fc_AC4IR(aConstants.data(), d_dst.get(), d_dst.step(), roi);
    }
  } else {
    if (p.channels == 1) {
      status = p.useCtx ? nppiDivC_32fc_C1R_Ctx(d_src.get(), d_src.step(), aConstants[0], d_dst.get(), d_dst.step(), roi, ctx)
                        : nppiDivC_32fc_C1R(d_src.get(), d_src.step(), aConstants[0], d_dst.get(), d_dst.step(), roi);
    } else if (p.channels == 3) {
      status = p.useCtx ? nppiDivC_32fc_C3R_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi, ctx)
                        : nppiDivC_32fc_C3R(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi);
    } else if (p.channels == 4) {
      status = p.useCtx ? nppiDivC_32fc_C4R_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi, ctx)
                        : nppiDivC_32fc_C4R(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi);
    } else if (p.channels == -4) {
      status = p.useCtx ? nppiDivC_32fc_AC4R_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi, ctx)
                        : nppiDivC_32fc_AC4R(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi);
    }
  }
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32fc> result(total);
  d_dst.copyToHost(result);

  for (int i = 0; i < total; i++) {
    EXPECT_NEAR(result[i].re, expected[i].re, 1e-3f) << "Real mismatch at " << i;
    EXPECT_NEAR(result[i].im, expected[i].im, 1e-3f) << "Imag mismatch at " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(
    DivC32fcSuite, DivC32fcTest,
    ::testing::Values(
        DivCComplexParam{"C1R", 1, false, false},
        DivCComplexParam{"C1R_Ctx", 1, false, true},
        DivCComplexParam{"C1IR", 1, true, false},
        DivCComplexParam{"C1IR_Ctx", 1, true, true},
        DivCComplexParam{"C3R", 3, false, false},
        DivCComplexParam{"C3R_Ctx", 3, false, true},
        DivCComplexParam{"C3IR", 3, true, false},
        DivCComplexParam{"C3IR_Ctx", 3, true, true},
        DivCComplexParam{"C4R", 4, false, false},
        DivCComplexParam{"C4R_Ctx", 4, false, true},
        DivCComplexParam{"C4IR", 4, true, false},
        DivCComplexParam{"C4IR_Ctx", 4, true, true},
        DivCComplexParam{"AC4R", -4, false, false},
        DivCComplexParam{"AC4R_Ctx", -4, false, true},
        DivCComplexParam{"AC4IR", -4, true, false},
        DivCComplexParam{"AC4IR_Ctx", -4, true, true}
    ),
    [](const ::testing::TestParamInfo<DivCComplexParam>& info) { return info.param.name; }
);

// ==================== DivC Ctx variants for integer types ====================

struct DivCIntCtxParam {
  std::string name;
  std::string dtype;
  int channels;
  bool inplace;
};

class DivCIntCtxTest : public NppTestBase,
                       public ::testing::WithParamInterface<DivCIntCtxParam> {};

TEST_P(DivCIntCtxTest, CtxVariant) {
  const auto& p = GetParam();
  const int width = 32, height = 32;
  const int actualChannels = (p.channels == -4) ? 4 : p.channels;

  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppiSize roi = {width, height};
  NppStatus status = NPP_NO_ERROR;

  if (p.dtype == "8u") {
    const int total = width * height * actualChannels;
    std::vector<Npp8u> src(total), expected(total);
    TestDataGenerator::generateRandom(src, (Npp8u)0, (Npp8u)255, 12345);

    const int constCount = (p.channels == -4) ? 3 : actualChannels;
    std::vector<Npp8u> aConstants(constCount);
    for (int i = 0; i < constCount; i++) aConstants[i] = static_cast<Npp8u>(2 + i);

    int scaleFactor = 0;
    for (int i = 0; i < total; i++) {
      int ch = i % actualChannels;
      if (p.channels == -4 && ch == 3) {
        expected[i] = src[i];
      } else {
        int quot = (aConstants[ch] == 0) ? 0 : src[i] / aConstants[ch];
        expected[i] = static_cast<Npp8u>(std::min(255, std::max(0, quot >> scaleFactor)));
      }
    }

    NppImageMemory<Npp8u> d_src(width * actualChannels, height);
    NppImageMemory<Npp8u> d_dst(width * actualChannels, height);
    d_src.copyFromHost(src);
    if (p.channels == -4) d_dst.copyFromHost(src);

    if (p.inplace) {
      d_dst.copyFromHost(src);
      if (p.channels == 3) status = nppiDivC_8u_C3IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      else if (p.channels == 4) status = nppiDivC_8u_C4IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      else if (p.channels == -4) status = nppiDivC_8u_AC4IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
    } else {
      if (p.channels == 3) status = nppiDivC_8u_C3RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      else if (p.channels == 4) status = nppiDivC_8u_C4RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      else if (p.channels == -4) status = nppiDivC_8u_AC4RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> result(total);
    d_dst.copyToHost(result);
    for (size_t i = 0; i < result.size(); i++) {
      EXPECT_LE(std::abs(static_cast<int>(result[i]) - static_cast<int>(expected[i])), 1)
        << "Mismatch at index " << i;
    }
  }
  else if (p.dtype == "16u") {
    const int total = width * height * actualChannels;
    std::vector<Npp16u> src(total), expected(total);
    TestDataGenerator::generateRandom(src, (Npp16u)0, (Npp16u)65000, 12345);

    const int constCount = (p.channels == -4) ? 3 : actualChannels;
    std::vector<Npp16u> aConstants(constCount);
    for (int i = 0; i < constCount; i++) aConstants[i] = static_cast<Npp16u>(10 + i * 10);

    int scaleFactor = 0;
    for (int i = 0; i < total; i++) {
      int ch = i % actualChannels;
      if (p.channels == -4 && ch == 3) {
        expected[i] = src[i];
      } else {
        int quot = (aConstants[ch] == 0) ? 0 : src[i] / aConstants[ch];
        expected[i] = static_cast<Npp16u>(std::min(65535, std::max(0, quot >> scaleFactor)));
      }
    }

    NppImageMemory<Npp16u> d_src(width * actualChannels, height);
    NppImageMemory<Npp16u> d_dst(width * actualChannels, height);
    d_src.copyFromHost(src);
    if (p.channels == -4) d_dst.copyFromHost(src);

    if (p.inplace) {
      d_dst.copyFromHost(src);
      if (p.channels == 3) status = nppiDivC_16u_C3IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      else if (p.channels == 4) status = nppiDivC_16u_C4IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      else if (p.channels == -4) status = nppiDivC_16u_AC4IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
    } else {
      if (p.channels == 3) status = nppiDivC_16u_C3RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      else if (p.channels == 4) status = nppiDivC_16u_C4RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      else if (p.channels == -4) status = nppiDivC_16u_AC4RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16u> result(total);
    d_dst.copyToHost(result);
    for (size_t i = 0; i < result.size(); i++) {
      EXPECT_LE(std::abs(static_cast<int>(result[i]) - static_cast<int>(expected[i])), 1)
        << "Mismatch at index " << i;
    }
  }
  else if (p.dtype == "16s") {
    const int total = width * height * actualChannels;
    std::vector<Npp16s> src(total), expected(total);
    TestDataGenerator::generateRandom(src, (Npp16s)-10000, (Npp16s)10000, 12345);

    const int constCount = (p.channels == -4) ? 3 : actualChannels;
    std::vector<Npp16s> aConstants(constCount);
    for (int i = 0; i < constCount; i++) aConstants[i] = static_cast<Npp16s>(10 * (i + 1) * ((i % 2 == 0) ? 1 : -1));

    int scaleFactor = 0;
    for (int i = 0; i < total; i++) {
      int ch = i % actualChannels;
      if (p.channels == -4 && ch == 3) {
        expected[i] = src[i];
      } else {
        int quot = (aConstants[ch] == 0) ? 0 : src[i] / aConstants[ch];
        expected[i] = static_cast<Npp16s>(std::min(32767, std::max(-32768, quot >> scaleFactor)));
      }
    }

    NppImageMemory<Npp16s> d_src(width * actualChannels, height);
    NppImageMemory<Npp16s> d_dst(width * actualChannels, height);
    d_src.copyFromHost(src);
    if (p.channels == -4) d_dst.copyFromHost(src);

    if (p.inplace) {
      d_dst.copyFromHost(src);
      if (p.channels == 1) status = nppiDivC_16s_C1IRSfs_Ctx(aConstants[0], d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      else if (p.channels == 3) status = nppiDivC_16s_C3IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      else if (p.channels == 4) status = nppiDivC_16s_C4IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      else if (p.channels == -4) status = nppiDivC_16s_AC4IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
    } else {
      if (p.channels == 1) status = nppiDivC_16s_C1RSfs_Ctx(d_src.get(), d_src.step(), aConstants[0], d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      else if (p.channels == 3) status = nppiDivC_16s_C3RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      else if (p.channels == 4) status = nppiDivC_16s_C4RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      else if (p.channels == -4) status = nppiDivC_16s_AC4RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp16s> result(total);
    d_dst.copyToHost(result);
    for (size_t i = 0; i < result.size(); i++) {
      EXPECT_LE(std::abs(static_cast<int>(result[i]) - static_cast<int>(expected[i])), 1)
        << "Mismatch at index " << i;
    }
  }
  else if (p.dtype == "32s") {
    const int total = width * height * actualChannels;
    std::vector<Npp32s> src(total), expected(total);
    TestDataGenerator::generateRandom(src, (Npp32s)-100000, (Npp32s)100000, 12345);

    const int constCount = (p.channels == -4) ? 3 : actualChannels;
    std::vector<Npp32s> aConstants(constCount);
    for (int i = 0; i < constCount; i++) aConstants[i] = 10 * (i + 1) * ((i % 2 == 0) ? 1 : -1);

    int scaleFactor = 0;
    for (int i = 0; i < total; i++) {
      int ch = i % actualChannels;
      if (p.channels == -4 && ch == 3) {
        expected[i] = src[i];
      } else {
        int64_t quot = (aConstants[ch] == 0) ? 0 : src[i] / aConstants[ch];
        expected[i] = static_cast<Npp32s>(quot >> scaleFactor);
      }
    }

    NppImageMemory<Npp32s> d_src(width * actualChannels, height);
    NppImageMemory<Npp32s> d_dst(width * actualChannels, height);
    d_src.copyFromHost(src);

    if (p.inplace) {
      d_dst.copyFromHost(src);
      if (p.channels == 1) status = nppiDivC_32s_C1IRSfs_Ctx(aConstants[0], d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      else if (p.channels == 3) status = nppiDivC_32s_C3IRSfs_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
    } else {
      if (p.channels == 1) status = nppiDivC_32s_C1RSfs_Ctx(d_src.get(), d_src.step(), aConstants[0], d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
      else if (p.channels == 3) status = nppiDivC_32s_C3RSfs_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi, scaleFactor, ctx);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32s> result(total);
    d_dst.copyToHost(result);
    EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
  }
  else if (p.dtype == "32f") {
    const int total = width * height * actualChannels;
    std::vector<Npp32f> src(total), expected(total);
    TestDataGenerator::generateRandom(src, -100.0f, 100.0f, 12345);

    const int constCount = (p.channels == -4) ? 3 : actualChannels;
    std::vector<Npp32f> aConstants(constCount);
    for (int i = 0; i < constCount; i++) aConstants[i] = 2.0f * (i + 1) * ((i % 2 == 0) ? 1 : -1);

    for (int i = 0; i < total; i++) {
      int ch = i % actualChannels;
      if (p.channels == -4 && ch == 3) {
        expected[i] = src[i];
      } else {
        expected[i] = src[i] / aConstants[ch];
      }
    }

    NppImageMemory<Npp32f> d_src(width * actualChannels, height);
    NppImageMemory<Npp32f> d_dst(width * actualChannels, height);
    d_src.copyFromHost(src);
    if (p.channels == -4) d_dst.copyFromHost(src);

    if (p.inplace) {
      d_dst.copyFromHost(src);
      if (p.channels == 3) status = nppiDivC_32f_C3IR_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, ctx);
      else if (p.channels == 4) status = nppiDivC_32f_C4IR_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, ctx);
      else if (p.channels == -4) status = nppiDivC_32f_AC4IR_Ctx(aConstants.data(), d_dst.get(), d_dst.step(), roi, ctx);
    } else {
      if (p.channels == 3) status = nppiDivC_32f_C3R_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi, ctx);
      else if (p.channels == 4) status = nppiDivC_32f_C4R_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi, ctx);
      else if (p.channels == -4) status = nppiDivC_32f_AC4R_Ctx(d_src.get(), d_src.step(), aConstants.data(), d_dst.get(), d_dst.step(), roi, ctx);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp32f> result(total);
    d_dst.copyToHost(result);
    EXPECT_TRUE(ResultValidator::arraysEqual(result, expected, 1e-4f));
  }
}

INSTANTIATE_TEST_SUITE_P(
    DivCIntCtxSuite, DivCIntCtxTest,
    ::testing::Values(
        DivCIntCtxParam{"8u_C3RSfs_Ctx", "8u", 3, false},
        DivCIntCtxParam{"8u_C4RSfs_Ctx", "8u", 4, false},
        DivCIntCtxParam{"8u_AC4RSfs_Ctx", "8u", -4, false},
        DivCIntCtxParam{"8u_C3IRSfs_Ctx", "8u", 3, true},
        DivCIntCtxParam{"8u_C4IRSfs_Ctx", "8u", 4, true},
        DivCIntCtxParam{"8u_AC4IRSfs_Ctx", "8u", -4, true},
        DivCIntCtxParam{"16u_C3RSfs_Ctx", "16u", 3, false},
        DivCIntCtxParam{"16u_C4RSfs_Ctx", "16u", 4, false},
        DivCIntCtxParam{"16u_AC4RSfs_Ctx", "16u", -4, false},
        DivCIntCtxParam{"16u_C3IRSfs_Ctx", "16u", 3, true},
        DivCIntCtxParam{"16u_C4IRSfs_Ctx", "16u", 4, true},
        DivCIntCtxParam{"16u_AC4IRSfs_Ctx", "16u", -4, true},
        DivCIntCtxParam{"16s_C1RSfs_Ctx", "16s", 1, false},
        DivCIntCtxParam{"16s_C3RSfs_Ctx", "16s", 3, false},
        DivCIntCtxParam{"16s_C4RSfs_Ctx", "16s", 4, false},
        DivCIntCtxParam{"16s_AC4RSfs_Ctx", "16s", -4, false},
        DivCIntCtxParam{"16s_C1IRSfs_Ctx", "16s", 1, true},
        DivCIntCtxParam{"16s_C3IRSfs_Ctx", "16s", 3, true},
        DivCIntCtxParam{"16s_C4IRSfs_Ctx", "16s", 4, true},
        DivCIntCtxParam{"16s_AC4IRSfs_Ctx", "16s", -4, true},
        DivCIntCtxParam{"32s_C1RSfs_Ctx", "32s", 1, false},
        DivCIntCtxParam{"32s_C3RSfs_Ctx", "32s", 3, false},
        DivCIntCtxParam{"32s_C1IRSfs_Ctx", "32s", 1, true},
        DivCIntCtxParam{"32s_C3IRSfs_Ctx", "32s", 3, true},
        DivCIntCtxParam{"32f_C3R_Ctx", "32f", 3, false},
        DivCIntCtxParam{"32f_C4R_Ctx", "32f", 4, false},
        DivCIntCtxParam{"32f_AC4R_Ctx", "32f", -4, false},
        DivCIntCtxParam{"32f_C3IR_Ctx", "32f", 3, true},
        DivCIntCtxParam{"32f_C4IR_Ctx", "32f", 4, true},
        DivCIntCtxParam{"32f_AC4IR_Ctx", "32f", -4, true}
    ),
    [](const ::testing::TestParamInfo<DivCIntCtxParam>& info) { return info.param.name; }
);

// ==================== DivC Non-Ctx integer inplace variants ====================

class DivC16sInplaceTest : public NppTestBase {};

TEST_F(DivC16sInplaceTest, DivC_16s_C1IRSfs) {
  const int width = 32, height = 32, total = width * height;
  std::vector<Npp16s> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp16s)-10000, (Npp16s)10000, 12345);

  Npp16s nConstant = 5;
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int quot = src[i] / nConstant;
    expected[i] = static_cast<Npp16s>(std::min(32767, std::max(-32768, quot >> scaleFactor)));
  }

  NppImageMemory<Npp16s> d_srcdst(width, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiDivC_16s_C1IRSfs(nConstant, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp16s> result(total);
  d_srcdst.copyToHost(result);
  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_LE(std::abs(static_cast<int>(result[i]) - static_cast<int>(expected[i])), 1);
  }
}

TEST_F(DivC16sInplaceTest, DivC_16s_C3IRSfs) {
  const int width = 32, height = 32, channels = 3, total = width * height * channels;
  std::vector<Npp16s> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp16s)-10000, (Npp16s)10000, 12345);

  Npp16s aConstants[3] = {10, -20, 30};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int quot = (aConstants[i % 3] == 0) ? 0 : src[i] / aConstants[i % 3];
    expected[i] = static_cast<Npp16s>(std::min(32767, std::max(-32768, quot >> scaleFactor)));
  }

  NppImageMemory<Npp16s> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiDivC_16s_C3IRSfs(aConstants, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp16s> result(total);
  d_srcdst.copyToHost(result);
  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_LE(std::abs(static_cast<int>(result[i]) - static_cast<int>(expected[i])), 1);
  }
}

TEST_F(DivC16sInplaceTest, DivC_16s_C4IRSfs) {
  const int width = 32, height = 32, channels = 4, total = width * height * channels;
  std::vector<Npp16s> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp16s)-10000, (Npp16s)10000, 12345);

  Npp16s aConstants[4] = {10, -20, 30, -40};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int quot = (aConstants[i % 4] == 0) ? 0 : src[i] / aConstants[i % 4];
    expected[i] = static_cast<Npp16s>(std::min(32767, std::max(-32768, quot >> scaleFactor)));
  }

  NppImageMemory<Npp16s> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiDivC_16s_C4IRSfs(aConstants, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp16s> result(total);
  d_srcdst.copyToHost(result);
  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_LE(std::abs(static_cast<int>(result[i]) - static_cast<int>(expected[i])), 1);
  }
}

TEST_F(DivC16sInplaceTest, DivC_16s_AC4IRSfs) {
  const int width = 32, height = 32, channels = 4, total = width * height * channels;
  std::vector<Npp16s> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp16s)-10000, (Npp16s)10000, 12345);

  Npp16s aConstants[3] = {10, -20, 30};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int ch = i % 4;
    if (ch == 3) {
      expected[i] = src[i];
    } else {
      int quot = (aConstants[ch] == 0) ? 0 : src[i] / aConstants[ch];
      expected[i] = static_cast<Npp16s>(std::min(32767, std::max(-32768, quot >> scaleFactor)));
    }
  }

  NppImageMemory<Npp16s> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiDivC_16s_AC4IRSfs(aConstants, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp16s> result(total);
  d_srcdst.copyToHost(result);
  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_LE(std::abs(static_cast<int>(result[i]) - static_cast<int>(expected[i])), 1);
  }
}

class DivC16uInplaceTest : public NppTestBase {};

TEST_F(DivC16uInplaceTest, DivC_16u_C3IRSfs) {
  const int width = 32, height = 32, channels = 3, total = width * height * channels;
  std::vector<Npp16u> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp16u)0, (Npp16u)65000, 12345);

  Npp16u aConstants[3] = {10, 20, 30};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int quot = (aConstants[i % 3] == 0) ? 0 : src[i] / aConstants[i % 3];
    expected[i] = static_cast<Npp16u>(std::min(65535, std::max(0, quot >> scaleFactor)));
  }

  NppImageMemory<Npp16u> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiDivC_16u_C3IRSfs(aConstants, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp16u> result(total);
  d_srcdst.copyToHost(result);
  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_LE(std::abs(static_cast<int>(result[i]) - static_cast<int>(expected[i])), 1);
  }
}

TEST_F(DivC16uInplaceTest, DivC_16u_C4IRSfs) {
  const int width = 32, height = 32, channels = 4, total = width * height * channels;
  std::vector<Npp16u> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp16u)0, (Npp16u)65000, 12345);

  Npp16u aConstants[4] = {10, 20, 30, 40};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int quot = (aConstants[i % 4] == 0) ? 0 : src[i] / aConstants[i % 4];
    expected[i] = static_cast<Npp16u>(std::min(65535, std::max(0, quot >> scaleFactor)));
  }

  NppImageMemory<Npp16u> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiDivC_16u_C4IRSfs(aConstants, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp16u> result(total);
  d_srcdst.copyToHost(result);
  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_LE(std::abs(static_cast<int>(result[i]) - static_cast<int>(expected[i])), 1);
  }
}

TEST_F(DivC16uInplaceTest, DivC_16u_AC4IRSfs) {
  const int width = 32, height = 32, channels = 4, total = width * height * channels;
  std::vector<Npp16u> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp16u)0, (Npp16u)65000, 12345);

  Npp16u aConstants[3] = {10, 20, 30};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int ch = i % 4;
    if (ch == 3) {
      expected[i] = src[i];
    } else {
      int quot = (aConstants[ch] == 0) ? 0 : src[i] / aConstants[ch];
      expected[i] = static_cast<Npp16u>(std::min(65535, std::max(0, quot >> scaleFactor)));
    }
  }

  NppImageMemory<Npp16u> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiDivC_16u_AC4IRSfs(aConstants, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp16u> result(total);
  d_srcdst.copyToHost(result);
  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_LE(std::abs(static_cast<int>(result[i]) - static_cast<int>(expected[i])), 1);
  }
}

class DivC32sInplaceTest : public NppTestBase {};

TEST_F(DivC32sInplaceTest, DivC_32s_C1IRSfs) {
  const int width = 32, height = 32, total = width * height;
  std::vector<Npp32s> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp32s)-100000, (Npp32s)100000, 12345);

  Npp32s nConstant = 5;
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int64_t quot = src[i] / nConstant;
    expected[i] = static_cast<Npp32s>(quot >> scaleFactor);
  }

  NppImageMemory<Npp32s> d_srcdst(width, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiDivC_32s_C1IRSfs(nConstant, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp32s> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(DivC32sInplaceTest, DivC_32s_C3IRSfs) {
  const int width = 32, height = 32, channels = 3, total = width * height * channels;
  std::vector<Npp32s> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp32s)-100000, (Npp32s)100000, 12345);

  Npp32s aConstants[3] = {10, -20, 30};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int64_t quot = (aConstants[i % 3] == 0) ? 0 : src[i] / aConstants[i % 3];
    expected[i] = static_cast<Npp32s>(quot >> scaleFactor);
  }

  NppImageMemory<Npp32s> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiDivC_32s_C3IRSfs(aConstants, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp32s> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}
