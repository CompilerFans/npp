#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ============================================================================
// Mul 16f (half precision float) - C1/C3/C4 channels
// ============================================================================

struct Mul16fCompleteParam {
  int width;
  int height;
  int channels;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Mul16fCompleteParamTest : public NppTestBase, public ::testing::WithParamInterface<Mul16fCompleteParam> {};

TEST_P(Mul16fCompleteParamTest, Mul_16f_Complete) {
  const auto &p = GetParam();
  const int total = p.width * p.height * p.channels;

  std::vector<Npp32f> src1_f(total), src2_f(total);
  TestDataGenerator::generateRandom(src1_f, -5.0f, 5.0f, 12345);
  TestDataGenerator::generateRandom(src2_f, -5.0f, 5.0f, 54321);

  std::vector<Npp16f> src1(total), src2(total), expected(total);
  for (int i = 0; i < total; i++) {
    src1[i] = float_to_npp16f_host(src1_f[i]);
    src2[i] = float_to_npp16f_host(src2_f[i]);
    expected[i] = float_to_npp16f_host(src1_f[i] * src2_f[i]);
  }

  NppImageMemory<Npp16f> d_src1(p.width * p.channels, p.height);
  NppImageMemory<Npp16f> d_src2(p.width * p.channels, p.height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {p.width, p.height};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status = NPP_ERROR;

  if (p.in_place) {
    if (p.channels == 1) {
      status = p.use_ctx ? nppiMul_16f_C1IR_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, ctx)
                         : nppiMul_16f_C1IR(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi);
    } else if (p.channels == 3) {
      status = p.use_ctx ? nppiMul_16f_C3IR_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, ctx)
                         : nppiMul_16f_C3IR(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi);
    } else if (p.channels == 4) {
      status = p.use_ctx ? nppiMul_16f_C4IR_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, ctx)
                         : nppiMul_16f_C4IR(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);
    std::vector<Npp16f> result(total);
    d_src2.copyToHost(result);
    EXPECT_TRUE(ResultValidator::arraysEqual16f(result, expected, 0.5f));
  } else {
    NppImageMemory<Npp16f> d_dst(p.width * p.channels, p.height);
    if (p.channels == 1) {
      status = p.use_ctx ? nppiMul_16f_C1R_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                               d_dst.step(), roi, ctx)
                         : nppiMul_16f_C1R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                           d_dst.step(), roi);
    } else if (p.channels == 3) {
      status = p.use_ctx ? nppiMul_16f_C3R_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                               d_dst.step(), roi, ctx)
                         : nppiMul_16f_C3R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                           d_dst.step(), roi);
    } else if (p.channels == 4) {
      status = p.use_ctx ? nppiMul_16f_C4R_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                               d_dst.step(), roi, ctx)
                         : nppiMul_16f_C4R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                           d_dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);
    std::vector<Npp16f> result(total);
    d_dst.copyToHost(result);
    EXPECT_TRUE(ResultValidator::arraysEqual16f(result, expected, 0.5f));
  }
}

INSTANTIATE_TEST_SUITE_P(
    Mul16fComplete, Mul16fCompleteParamTest,
    ::testing::Values(
        Mul16fCompleteParam{32, 32, 1, false, false, "C1R"}, Mul16fCompleteParam{32, 32, 1, true, false, "C1R_Ctx"},
        Mul16fCompleteParam{32, 32, 1, false, true, "C1IR"}, Mul16fCompleteParam{32, 32, 1, true, true, "C1IR_Ctx"},
        Mul16fCompleteParam{32, 32, 3, false, false, "C3R"}, Mul16fCompleteParam{32, 32, 3, true, false, "C3R_Ctx"},
        Mul16fCompleteParam{32, 32, 3, false, true, "C3IR"}, Mul16fCompleteParam{32, 32, 3, true, true, "C3IR_Ctx"},
        Mul16fCompleteParam{32, 32, 4, false, false, "C4R"}, Mul16fCompleteParam{32, 32, 4, true, false, "C4R_Ctx"},
        Mul16fCompleteParam{32, 32, 4, false, true, "C4IR"}, Mul16fCompleteParam{32, 32, 4, true, true, "C4IR_Ctx"}),
    [](const ::testing::TestParamInfo<Mul16fCompleteParam> &info) { return info.param.name; });

// ============================================================================
// Mul 32fc (complex float) - C1/C3/C4/AC4 channels
// ============================================================================

struct Mul32fcParam {
  int width;
  int height;
  int channels; // 1, 3, 4, or -4 for AC4
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Mul32fcParamTest : public NppTestBase, public ::testing::WithParamInterface<Mul32fcParam> {};

TEST_P(Mul32fcParamTest, Mul_32fc) {
  const auto &p = GetParam();
  const int ch = (p.channels < 0) ? 4 : p.channels;
  const int total = p.width * p.height * ch;

  std::vector<Npp32fc> src1(total), src2(total), expected(total);
  for (int i = 0; i < total; i++) {
    src1[i].re = static_cast<Npp32f>((i % 20) - 10);
    src1[i].im = static_cast<Npp32f>((i % 10) - 5);
    src2[i].re = static_cast<Npp32f>(((i + 7) % 20) - 10);
    src2[i].im = static_cast<Npp32f>(((i + 3) % 10) - 5);
    if (p.channels == -4 && (i % 4 == 3)) {
      expected[i] = src2[i]; // AC4: alpha unchanged
    } else {
      // Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
      expected[i].re = src1[i].re * src2[i].re - src1[i].im * src2[i].im;
      expected[i].im = src1[i].re * src2[i].im + src1[i].im * src2[i].re;
    }
  }

  NppImageMemory<Npp32fc> d_src1(p.width * ch, p.height);
  NppImageMemory<Npp32fc> d_src2(p.width * ch, p.height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {p.width, p.height};
  NppStreamContext ctx{};
  NppStatus status = NPP_ERROR;

  if (p.in_place) {
    if (p.channels == 1) {
      status = p.use_ctx ? nppiMul_32fc_C1IR_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, ctx)
                         : nppiMul_32fc_C1IR(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi);
    } else if (p.channels == 3) {
      status = p.use_ctx ? nppiMul_32fc_C3IR_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, ctx)
                         : nppiMul_32fc_C3IR(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi);
    } else if (p.channels == 4) {
      status = p.use_ctx ? nppiMul_32fc_C4IR_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, ctx)
                         : nppiMul_32fc_C4IR(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi);
    } else if (p.channels == -4) {
      status = p.use_ctx ? nppiMul_32fc_AC4IR_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, ctx)
                         : nppiMul_32fc_AC4IR(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);
    std::vector<Npp32fc> result(total);
    d_src2.copyToHost(result);
    for (int i = 0; i < total; i++) {
      EXPECT_NEAR(result[i].re, expected[i].re, 1e-3f) << "re mismatch at " << i;
      EXPECT_NEAR(result[i].im, expected[i].im, 1e-3f) << "im mismatch at " << i;
    }
  } else {
    NppImageMemory<Npp32fc> d_dst(p.width * ch, p.height);
    if (p.channels == -4)
      d_dst.copyFromHost(src2);
    if (p.channels == 1) {
      status = p.use_ctx ? nppiMul_32fc_C1R_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                d_dst.step(), roi, ctx)
                         : nppiMul_32fc_C1R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                            d_dst.step(), roi);
    } else if (p.channels == 3) {
      status = p.use_ctx ? nppiMul_32fc_C3R_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                d_dst.step(), roi, ctx)
                         : nppiMul_32fc_C3R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                            d_dst.step(), roi);
    } else if (p.channels == 4) {
      status = p.use_ctx ? nppiMul_32fc_C4R_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                d_dst.step(), roi, ctx)
                         : nppiMul_32fc_C4R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                            d_dst.step(), roi);
    } else if (p.channels == -4) {
      status = p.use_ctx ? nppiMul_32fc_AC4R_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                 d_dst.step(), roi, ctx)
                         : nppiMul_32fc_AC4R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                             d_dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);
    std::vector<Npp32fc> result(total);
    d_dst.copyToHost(result);
    for (int i = 0; i < total; i++) {
      EXPECT_NEAR(result[i].re, expected[i].re, 1e-3f) << "re mismatch at " << i;
      EXPECT_NEAR(result[i].im, expected[i].im, 1e-3f) << "im mismatch at " << i;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Mul32fc, Mul32fcParamTest,
    ::testing::Values(Mul32fcParam{32, 32, 1, false, false, "C1R"}, Mul32fcParam{32, 32, 1, true, false, "C1R_Ctx"},
                      Mul32fcParam{32, 32, 1, false, true, "C1IR"}, Mul32fcParam{32, 32, 1, true, true, "C1IR_Ctx"},
                      Mul32fcParam{32, 32, 3, false, false, "C3R"}, Mul32fcParam{32, 32, 3, true, false, "C3R_Ctx"},
                      Mul32fcParam{32, 32, 3, false, true, "C3IR"}, Mul32fcParam{32, 32, 3, true, true, "C3IR_Ctx"},
                      Mul32fcParam{32, 32, 4, false, false, "C4R"}, Mul32fcParam{32, 32, 4, true, false, "C4R_Ctx"},
                      Mul32fcParam{32, 32, 4, false, true, "C4IR"}, Mul32fcParam{32, 32, 4, true, true, "C4IR_Ctx"},
                      Mul32fcParam{32, 32, -4, false, false, "AC4R"}, Mul32fcParam{32, 32, -4, true, false, "AC4R_Ctx"},
                      Mul32fcParam{32, 32, -4, false, true, "AC4IR"},
                      Mul32fcParam{32, 32, -4, true, true, "AC4IR_Ctx"}),
    [](const ::testing::TestParamInfo<Mul32fcParam> &info) { return info.param.name; });

// ============================================================================
// Mul 16sc (complex 16-bit signed with scale)
// ============================================================================

struct Mul16scParam {
  int width;
  int height;
  int channels; // 1, 3, or -4 for AC4
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Mul16scParamTest : public NppTestBase, public ::testing::WithParamInterface<Mul16scParam> {};

TEST_P(Mul16scParamTest, Mul_16sc) {
  const auto &p = GetParam();
  const int ch = (p.channels < 0) ? 4 : p.channels;
  const int total = p.width * p.height * ch;

  std::vector<Npp16sc> src1(total), src2(total), expected(total);
  for (int i = 0; i < total; i++) {
    src1[i].re = static_cast<Npp16s>((i % 40) - 20);
    src1[i].im = static_cast<Npp16s>((i % 20) - 10);
    src2[i].re = static_cast<Npp16s>(((i + 7) % 40) - 20);
    src2[i].im = static_cast<Npp16s>(((i + 3) % 20) - 10);
    if (p.channels == -4 && (i % 4 == 3)) {
      expected[i] = src2[i];
    } else {
      int re = src1[i].re * src2[i].re - src1[i].im * src2[i].im;
      int im = src1[i].re * src2[i].im + src1[i].im * src2[i].re;
      if (p.scaleFactor > 0) {
        re = (re + (1 << (p.scaleFactor - 1))) >> p.scaleFactor;
        im = (im + (1 << (p.scaleFactor - 1))) >> p.scaleFactor;
      }
      expected[i].re = static_cast<Npp16s>(std::max(-32768, std::min(32767, re)));
      expected[i].im = static_cast<Npp16s>(std::max(-32768, std::min(32767, im)));
    }
  }

  NppImageMemory<Npp16sc> d_src1(p.width * ch, p.height);
  NppImageMemory<Npp16sc> d_src2(p.width * ch, p.height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {p.width, p.height};
  NppStreamContext ctx{};
  NppStatus status = NPP_ERROR;

  if (p.in_place) {
    if (p.channels == 1) {
      status = p.use_ctx
                   ? nppiMul_16sc_C1IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                              p.scaleFactor, ctx)
                   : nppiMul_16sc_C1IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
    } else if (p.channels == 3) {
      status = p.use_ctx
                   ? nppiMul_16sc_C3IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                              p.scaleFactor, ctx)
                   : nppiMul_16sc_C3IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
    } else if (p.channels == -4) {
      status = p.use_ctx ? nppiMul_16sc_AC4IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                     p.scaleFactor, ctx)
                         : nppiMul_16sc_AC4IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                 p.scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);
    std::vector<Npp16sc> result(total);
    d_src2.copyToHost(result);
    for (int i = 0; i < total; i++) {
      EXPECT_LE(std::abs(result[i].re - expected[i].re), 1) << "re mismatch at " << i;
      EXPECT_LE(std::abs(result[i].im - expected[i].im), 1) << "im mismatch at " << i;
    }
  } else {
    NppImageMemory<Npp16sc> d_dst(p.width * ch, p.height);
    if (p.channels == -4)
      d_dst.copyFromHost(src2);
    if (p.channels == 1) {
      status = p.use_ctx ? nppiMul_16sc_C1RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                   d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                         : nppiMul_16sc_C1RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                               d_dst.step(), roi, p.scaleFactor);
    } else if (p.channels == 3) {
      status = p.use_ctx ? nppiMul_16sc_C3RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                   d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                         : nppiMul_16sc_C3RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                               d_dst.step(), roi, p.scaleFactor);
    } else if (p.channels == -4) {
      status = p.use_ctx ? nppiMul_16sc_AC4RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                    d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                         : nppiMul_16sc_AC4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                d_dst.step(), roi, p.scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);
    std::vector<Npp16sc> result(total);
    d_dst.copyToHost(result);
    for (int i = 0; i < total; i++) {
      EXPECT_LE(std::abs(result[i].re - expected[i].re), 1) << "re mismatch at " << i;
      EXPECT_LE(std::abs(result[i].im - expected[i].im), 1) << "im mismatch at " << i;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Mul16sc, Mul16scParamTest,
    ::testing::Values(
        Mul16scParam{32, 32, 1, 0, false, false, "C1RSfs"}, Mul16scParam{32, 32, 1, 0, true, false, "C1RSfs_Ctx"},
        Mul16scParam{32, 32, 1, 0, false, true, "C1IRSfs"}, Mul16scParam{32, 32, 1, 0, true, true, "C1IRSfs_Ctx"},
        Mul16scParam{32, 32, 3, 0, false, false, "C3RSfs"}, Mul16scParam{32, 32, 3, 0, true, false, "C3RSfs_Ctx"},
        Mul16scParam{32, 32, 3, 0, false, true, "C3IRSfs"}, Mul16scParam{32, 32, 3, 0, true, true, "C3IRSfs_Ctx"},
        Mul16scParam{32, 32, -4, 0, false, false, "AC4RSfs"}, Mul16scParam{32, 32, -4, 0, true, false, "AC4RSfs_Ctx"},
        Mul16scParam{32, 32, -4, 0, false, true, "AC4IRSfs"}, Mul16scParam{32, 32, -4, 0, true, true, "AC4IRSfs_Ctx"}),
    [](const ::testing::TestParamInfo<Mul16scParam> &info) { return info.param.name; });

// ============================================================================
// Mul 32sc (complex 32-bit signed with scale)
// ============================================================================

struct Mul32scParam {
  int width;
  int height;
  int channels;
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Mul32scParamTest : public NppTestBase, public ::testing::WithParamInterface<Mul32scParam> {};

TEST_P(Mul32scParamTest, Mul_32sc) {
  const auto &p = GetParam();
  const int ch = (p.channels < 0) ? 4 : p.channels;
  const int total = p.width * p.height * ch;

  std::vector<Npp32sc> src1(total), src2(total), expected(total);
  for (int i = 0; i < total; i++) {
    src1[i].re = static_cast<Npp32s>((i % 400) - 200);
    src1[i].im = static_cast<Npp32s>((i % 200) - 100);
    src2[i].re = static_cast<Npp32s>(((i + 70) % 400) - 200);
    src2[i].im = static_cast<Npp32s>(((i + 30) % 200) - 100);
    if (p.channels == -4 && (i % 4 == 3)) {
      expected[i] = src2[i];
    } else {
      int64_t re = static_cast<int64_t>(src1[i].re) * src2[i].re - static_cast<int64_t>(src1[i].im) * src2[i].im;
      int64_t im = static_cast<int64_t>(src1[i].re) * src2[i].im + static_cast<int64_t>(src1[i].im) * src2[i].re;
      if (p.scaleFactor > 0) {
        re = (re + (1LL << (p.scaleFactor - 1))) >> p.scaleFactor;
        im = (im + (1LL << (p.scaleFactor - 1))) >> p.scaleFactor;
      }
      expected[i].re = static_cast<Npp32s>(re);
      expected[i].im = static_cast<Npp32s>(im);
    }
  }

  NppImageMemory<Npp32sc> d_src1(p.width * ch, p.height);
  NppImageMemory<Npp32sc> d_src2(p.width * ch, p.height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {p.width, p.height};
  NppStreamContext ctx{};
  NppStatus status = NPP_ERROR;

  if (p.in_place) {
    if (p.channels == 1) {
      status = p.use_ctx
                   ? nppiMul_32sc_C1IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                              p.scaleFactor, ctx)
                   : nppiMul_32sc_C1IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
    } else if (p.channels == 3) {
      status = p.use_ctx
                   ? nppiMul_32sc_C3IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                              p.scaleFactor, ctx)
                   : nppiMul_32sc_C3IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
    } else if (p.channels == -4) {
      status = p.use_ctx ? nppiMul_32sc_AC4IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                     p.scaleFactor, ctx)
                         : nppiMul_32sc_AC4IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                 p.scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);
    std::vector<Npp32sc> result(total);
    d_src2.copyToHost(result);
    for (int i = 0; i < total; i++) {
      EXPECT_LE(std::abs(result[i].re - expected[i].re), 1) << "re mismatch at " << i;
      EXPECT_LE(std::abs(result[i].im - expected[i].im), 1) << "im mismatch at " << i;
    }
  } else {
    NppImageMemory<Npp32sc> d_dst(p.width * ch, p.height);
    if (p.channels == -4)
      d_dst.copyFromHost(src2);
    if (p.channels == 1) {
      status = p.use_ctx ? nppiMul_32sc_C1RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                   d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                         : nppiMul_32sc_C1RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                               d_dst.step(), roi, p.scaleFactor);
    } else if (p.channels == 3) {
      status = p.use_ctx ? nppiMul_32sc_C3RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                   d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                         : nppiMul_32sc_C3RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                               d_dst.step(), roi, p.scaleFactor);
    } else if (p.channels == -4) {
      status = p.use_ctx ? nppiMul_32sc_AC4RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                    d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                         : nppiMul_32sc_AC4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                d_dst.step(), roi, p.scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);
    std::vector<Npp32sc> result(total);
    d_dst.copyToHost(result);
    for (int i = 0; i < total; i++) {
      EXPECT_LE(std::abs(result[i].re - expected[i].re), 1) << "re mismatch at " << i;
      EXPECT_LE(std::abs(result[i].im - expected[i].im), 1) << "im mismatch at " << i;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Mul32sc, Mul32scParamTest,
    ::testing::Values(
        Mul32scParam{32, 32, 1, 0, false, false, "C1RSfs"}, Mul32scParam{32, 32, 1, 0, true, false, "C1RSfs_Ctx"},
        Mul32scParam{32, 32, 1, 0, false, true, "C1IRSfs"}, Mul32scParam{32, 32, 1, 0, true, true, "C1IRSfs_Ctx"},
        Mul32scParam{32, 32, 3, 0, false, false, "C3RSfs"}, Mul32scParam{32, 32, 3, 0, true, false, "C3RSfs_Ctx"},
        Mul32scParam{32, 32, 3, 0, false, true, "C3IRSfs"}, Mul32scParam{32, 32, 3, 0, true, true, "C3IRSfs_Ctx"},
        Mul32scParam{32, 32, -4, 0, false, false, "AC4RSfs"}, Mul32scParam{32, 32, -4, 0, true, false, "AC4RSfs_Ctx"},
        Mul32scParam{32, 32, -4, 0, false, true, "AC4IRSfs"}, Mul32scParam{32, 32, -4, 0, true, true, "AC4IRSfs_Ctx"}),
    [](const ::testing::TestParamInfo<Mul32scParam> &info) { return info.param.name; });

// ============================================================================
// Mul remaining integer variants (8u/16u/16s/32s) - C3/C4/AC4 with Ctx and IR
// ============================================================================

struct MulIntSfsCompleteParam {
  int width;
  int height;
  int channels; // 3, 4, or -4 for AC4
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string dtype; // "8u", "16u", "16s", "32s"
  std::string name;
};

class MulIntSfsCompleteParamTest : public NppTestBase, public ::testing::WithParamInterface<MulIntSfsCompleteParam> {};

TEST_P(MulIntSfsCompleteParamTest, Mul_IntSfs_Complete) {
  const auto &p = GetParam();
  const int ch = (p.channels < 0) ? 4 : p.channels;
  const int total = p.width * p.height * ch;

  NppiSize roi = {p.width, p.height};
  NppStreamContext ctx{};
  NppStatus status = NPP_ERROR;

  if (p.dtype == "8u") {
    std::vector<Npp8u> src1(total), src2(total), expected(total);
    TestDataGenerator::generateRandom(src1, (Npp8u)1, (Npp8u)15, 12345);
    TestDataGenerator::generateRandom(src2, (Npp8u)1, (Npp8u)15, 54321);
    for (int i = 0; i < total; i++) {
      if (p.channels == -4 && (i % 4 == 3)) {
        expected[i] = src2[i];
      } else {
        expected[i] = expect::mul_sfs<Npp8u>(src1[i], src2[i], p.scaleFactor);
      }
    }
    NppImageMemory<Npp8u> d_src1(p.width * ch, p.height);
    NppImageMemory<Npp8u> d_src2(p.width * ch, p.height);
    d_src1.copyFromHost(src1);
    d_src2.copyFromHost(src2);

    if (p.in_place) {
      if (p.channels == 3) {
        status = p.use_ctx
                     ? nppiMul_8u_C3IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                              p.scaleFactor, ctx)
                     : nppiMul_8u_C3IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
      } else if (p.channels == 4) {
        status = p.use_ctx
                     ? nppiMul_8u_C4IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                              p.scaleFactor, ctx)
                     : nppiMul_8u_C4IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
      } else if (p.channels == -4) {
        status = p.use_ctx ? nppiMul_8u_AC4IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                     p.scaleFactor, ctx)
                           : nppiMul_8u_AC4IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                 p.scaleFactor);
      }
      ASSERT_EQ(status, NPP_NO_ERROR);
      std::vector<Npp8u> result(total);
      d_src2.copyToHost(result);
      EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
    } else {
      NppImageMemory<Npp8u> d_dst(p.width * ch, p.height);
      if (p.channels == -4)
        d_dst.copyFromHost(src2);
      if (p.channels == 3) {
        status = p.use_ctx ? nppiMul_8u_C3RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                   d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                           : nppiMul_8u_C3RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                               d_dst.step(), roi, p.scaleFactor);
      } else if (p.channels == 4) {
        status = p.use_ctx ? nppiMul_8u_C4RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                   d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                           : nppiMul_8u_C4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                               d_dst.step(), roi, p.scaleFactor);
      } else if (p.channels == -4) {
        status = p.use_ctx ? nppiMul_8u_AC4RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                    d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                           : nppiMul_8u_AC4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                d_dst.step(), roi, p.scaleFactor);
      }
      ASSERT_EQ(status, NPP_NO_ERROR);
      std::vector<Npp8u> result(total);
      d_dst.copyToHost(result);
      EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
    }
  } else if (p.dtype == "16u") {
    std::vector<Npp16u> src1(total), src2(total), expected(total);
    TestDataGenerator::generateRandom(src1, (Npp16u)1, (Npp16u)100, 12345);
    TestDataGenerator::generateRandom(src2, (Npp16u)1, (Npp16u)100, 54321);
    for (int i = 0; i < total; i++) {
      if (p.channels == -4 && (i % 4 == 3)) {
        expected[i] = src2[i];
      } else {
        expected[i] = expect::mul_sfs<Npp16u>(src1[i], src2[i], p.scaleFactor);
      }
    }
    NppImageMemory<Npp16u> d_src1(p.width * ch, p.height);
    NppImageMemory<Npp16u> d_src2(p.width * ch, p.height);
    d_src1.copyFromHost(src1);
    d_src2.copyFromHost(src2);

    if (p.in_place) {
      if (p.channels == 4) {
        status = p.use_ctx ? nppiMul_16u_C4IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                     p.scaleFactor, ctx)
                           : nppiMul_16u_C4IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                 p.scaleFactor);
      } else if (p.channels == -4) {
        status = p.use_ctx ? nppiMul_16u_AC4IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                      p.scaleFactor, ctx)
                           : nppiMul_16u_AC4IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                  p.scaleFactor);
      }
      ASSERT_EQ(status, NPP_NO_ERROR);
      std::vector<Npp16u> result(total);
      d_src2.copyToHost(result);
      EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
    } else {
      NppImageMemory<Npp16u> d_dst(p.width * ch, p.height);
      if (p.channels == -4)
        d_dst.copyFromHost(src2);
      if (p.channels == 4) {
        status = p.use_ctx ? nppiMul_16u_C4RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                    d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                           : nppiMul_16u_C4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                d_dst.step(), roi, p.scaleFactor);
      } else if (p.channels == -4) {
        status = p.use_ctx ? nppiMul_16u_AC4RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                     d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                           : nppiMul_16u_AC4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                 d_dst.step(), roi, p.scaleFactor);
      }
      ASSERT_EQ(status, NPP_NO_ERROR);
      std::vector<Npp16u> result(total);
      d_dst.copyToHost(result);
      EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
    }
  } else if (p.dtype == "16s") {
    std::vector<Npp16s> src1(total), src2(total), expected(total);
    TestDataGenerator::generateRandom(src1, (Npp16s)-50, (Npp16s)50, 12345);
    TestDataGenerator::generateRandom(src2, (Npp16s)-50, (Npp16s)50, 54321);
    for (int i = 0; i < total; i++) {
      if (p.channels == -4 && (i % 4 == 3)) {
        expected[i] = src2[i];
      } else {
        expected[i] = expect::mul_sfs<Npp16s>(src1[i], src2[i], p.scaleFactor);
      }
    }
    NppImageMemory<Npp16s> d_src1(p.width * ch, p.height);
    NppImageMemory<Npp16s> d_src2(p.width * ch, p.height);
    d_src1.copyFromHost(src1);
    d_src2.copyFromHost(src2);

    if (p.in_place) {
      if (p.channels == 4) {
        status = p.use_ctx ? nppiMul_16s_C4IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                     p.scaleFactor, ctx)
                           : nppiMul_16s_C4IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                 p.scaleFactor);
      } else if (p.channels == -4) {
        status = p.use_ctx ? nppiMul_16s_AC4IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                      p.scaleFactor, ctx)
                           : nppiMul_16s_AC4IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                  p.scaleFactor);
      }
      ASSERT_EQ(status, NPP_NO_ERROR);
      std::vector<Npp16s> result(total);
      d_src2.copyToHost(result);
      EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
    } else {
      NppImageMemory<Npp16s> d_dst(p.width * ch, p.height);
      if (p.channels == -4)
        d_dst.copyFromHost(src2);
      if (p.channels == 4) {
        status = p.use_ctx ? nppiMul_16s_C4RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                    d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                           : nppiMul_16s_C4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                d_dst.step(), roi, p.scaleFactor);
      } else if (p.channels == -4) {
        status = p.use_ctx ? nppiMul_16s_AC4RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                     d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                           : nppiMul_16s_AC4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                 d_dst.step(), roi, p.scaleFactor);
      }
      ASSERT_EQ(status, NPP_NO_ERROR);
      std::vector<Npp16s> result(total);
      d_dst.copyToHost(result);
      EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
    }
  } else if (p.dtype == "32s") {
    std::vector<Npp32s> src1(total), src2(total), expected(total);
    TestDataGenerator::generateRandom(src1, (Npp32s)-100, (Npp32s)100, 12345);
    TestDataGenerator::generateRandom(src2, (Npp32s)-100, (Npp32s)100, 54321);
    for (int i = 0; i < total; i++) {
      int64_t prod = static_cast<int64_t>(src1[i]) * src2[i];
      if (p.scaleFactor > 0) {
        prod = (prod + (1LL << (p.scaleFactor - 1))) >> p.scaleFactor;
      }
      expected[i] = static_cast<Npp32s>(prod);
    }
    NppImageMemory<Npp32s> d_src1(p.width * ch, p.height);
    NppImageMemory<Npp32s> d_src2(p.width * ch, p.height);
    d_src1.copyFromHost(src1);
    d_src2.copyFromHost(src2);

    if (p.in_place) {
      if (p.channels == 1) {
        status = p.use_ctx ? nppiMul_32s_C1IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                     p.scaleFactor, ctx)
                           : nppiMul_32s_C1IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                 p.scaleFactor);
      } else if (p.channels == 3) {
        status = p.use_ctx ? nppiMul_32s_C3IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                     p.scaleFactor, ctx)
                           : nppiMul_32s_C3IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                 p.scaleFactor);
      }
      ASSERT_EQ(status, NPP_NO_ERROR);
      std::vector<Npp32s> result(total);
      d_src2.copyToHost(result);
      for (size_t i = 0; i < result.size(); i++) {
        EXPECT_LE(std::abs(result[i] - expected[i]), 1) << "mismatch at " << i;
      }
    } else {
      NppImageMemory<Npp32s> d_dst(p.width * ch, p.height);
      if (p.channels == 1) {
        status = p.use_ctx ? nppiMul_32s_C1RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                    d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                           : nppiMul_32s_C1RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                d_dst.step(), roi, p.scaleFactor);
      } else if (p.channels == 3) {
        status = p.use_ctx ? nppiMul_32s_C3RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                    d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                           : nppiMul_32s_C3RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                d_dst.step(), roi, p.scaleFactor);
      }
      ASSERT_EQ(status, NPP_NO_ERROR);
      std::vector<Npp32s> result(total);
      d_dst.copyToHost(result);
      for (size_t i = 0; i < result.size(); i++) {
        EXPECT_LE(std::abs(result[i] - expected[i]), 1) << "mismatch at " << i;
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(MulIntSfsComplete, MulIntSfsCompleteParamTest,
                         ::testing::Values(
                             // 8u C3/C4/AC4 variants
                             MulIntSfsCompleteParam{32, 32, 3, 0, true, false, "8u", "8u_C3RSfs_Ctx"},
                             MulIntSfsCompleteParam{32, 32, 3, 0, true, true, "8u", "8u_C3IRSfs_Ctx"},
                             MulIntSfsCompleteParam{32, 32, 4, 0, true, false, "8u", "8u_C4RSfs_Ctx"},
                             MulIntSfsCompleteParam{32, 32, 4, 0, false, true, "8u", "8u_C4IRSfs"},
                             MulIntSfsCompleteParam{32, 32, 4, 0, true, true, "8u", "8u_C4IRSfs_Ctx"},
                             MulIntSfsCompleteParam{32, 32, -4, 0, true, false, "8u", "8u_AC4RSfs_Ctx"},
                             MulIntSfsCompleteParam{32, 32, -4, 0, false, true, "8u", "8u_AC4IRSfs"},
                             MulIntSfsCompleteParam{32, 32, -4, 0, true, true, "8u", "8u_AC4IRSfs_Ctx"},
                             // 16u C4/AC4 variants
                             MulIntSfsCompleteParam{32, 32, 4, 0, true, false, "16u", "16u_C4RSfs_Ctx"},
                             MulIntSfsCompleteParam{32, 32, 4, 0, false, true, "16u", "16u_C4IRSfs"},
                             MulIntSfsCompleteParam{32, 32, 4, 0, true, true, "16u", "16u_C4IRSfs_Ctx"},
                             MulIntSfsCompleteParam{32, 32, -4, 0, true, false, "16u", "16u_AC4RSfs_Ctx"},
                             MulIntSfsCompleteParam{32, 32, -4, 0, false, true, "16u", "16u_AC4IRSfs"},
                             MulIntSfsCompleteParam{32, 32, -4, 0, true, true, "16u", "16u_AC4IRSfs_Ctx"},
                             // 16s C4/AC4 variants
                             MulIntSfsCompleteParam{32, 32, 4, 0, true, false, "16s", "16s_C4RSfs_Ctx"},
                             MulIntSfsCompleteParam{32, 32, 4, 0, false, true, "16s", "16s_C4IRSfs"},
                             MulIntSfsCompleteParam{32, 32, 4, 0, true, true, "16s", "16s_C4IRSfs_Ctx"},
                             MulIntSfsCompleteParam{32, 32, -4, 0, true, false, "16s", "16s_AC4RSfs_Ctx"},
                             MulIntSfsCompleteParam{32, 32, -4, 0, false, true, "16s", "16s_AC4IRSfs"},
                             MulIntSfsCompleteParam{32, 32, -4, 0, true, true, "16s", "16s_AC4IRSfs_Ctx"},
                             // 32s C1/C3 variants
                             MulIntSfsCompleteParam{32, 32, 1, 0, true, false, "32s", "32s_C1RSfs_Ctx"},
                             MulIntSfsCompleteParam{32, 32, 1, 0, false, true, "32s", "32s_C1IRSfs"},
                             MulIntSfsCompleteParam{32, 32, 1, 0, true, true, "32s", "32s_C1IRSfs_Ctx"},
                             MulIntSfsCompleteParam{32, 32, 3, 0, true, false, "32s", "32s_C3RSfs_Ctx"},
                             MulIntSfsCompleteParam{32, 32, 3, 0, false, true, "32s", "32s_C3IRSfs"},
                             MulIntSfsCompleteParam{32, 32, 3, 0, true, true, "32s", "32s_C3IRSfs_Ctx"}),
                         [](const ::testing::TestParamInfo<MulIntSfsCompleteParam> &info) { return info.param.name; });

// ============================================================================
// Mul 32f AC4 variants
// ============================================================================

struct Mul32fAC4Param {
  int width;
  int height;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Mul32fAC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<Mul32fAC4Param> {};

TEST_P(Mul32fAC4ParamTest, Mul_32f_AC4) {
  const auto &p = GetParam();
  const int channels = 4;
  const int total = p.width * p.height * channels;

  std::vector<Npp32f> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, -10.0f, 10.0f, 12345);
  TestDataGenerator::generateRandom(src2, -10.0f, 10.0f, 54321);
  for (int i = 0; i < total; i++) {
    if (i % 4 == 3) {
      expected[i] = src2[i];
    } else {
      expected[i] = src1[i] * src2[i];
    }
  }

  NppImageMemory<Npp32f> d_src1(p.width * channels, p.height);
  NppImageMemory<Npp32f> d_src2(p.width * channels, p.height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {p.width, p.height};
  NppStreamContext ctx{};
  NppStatus status;

  if (p.in_place) {
    status = p.use_ctx ? nppiMul_32f_AC4IR_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, ctx)
                       : nppiMul_32f_AC4IR(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi);
    ASSERT_EQ(status, NPP_NO_ERROR);
    std::vector<Npp32f> result(total);
    d_src2.copyToHost(result);
    EXPECT_TRUE(ResultValidator::arraysEqual(result, expected, 1e-4f));
  } else {
    NppImageMemory<Npp32f> d_dst(p.width * channels, p.height);
    d_dst.copyFromHost(src2);
    status = p.use_ctx ? nppiMul_32f_AC4R_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                              d_dst.step(), roi, ctx)
                       : nppiMul_32f_AC4R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                          d_dst.step(), roi);
    ASSERT_EQ(status, NPP_NO_ERROR);
    std::vector<Npp32f> result(total);
    d_dst.copyToHost(result);
    EXPECT_TRUE(ResultValidator::arraysEqual(result, expected, 1e-4f));
  }
}

INSTANTIATE_TEST_SUITE_P(Mul32fAC4, Mul32fAC4ParamTest,
                         ::testing::Values(Mul32fAC4Param{32, 32, true, false, "AC4R_Ctx"},
                                           Mul32fAC4Param{32, 32, false, true, "AC4IR"},
                                           Mul32fAC4Param{32, 32, true, true, "AC4IR_Ctx"}),
                         [](const ::testing::TestParamInfo<Mul32fAC4Param> &info) { return info.param.name; });

// ============================================================================
// Mul_32s_C1R and Mul_32s_C1R_Ctx - non-Sfs variants
// ============================================================================

TEST(Mul32sC1RTest, Basic) {
  const int width = 32, height = 32;
  const int total = width * height;

  std::vector<Npp32s> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp32s)-100, (Npp32s)100, 12345);
  TestDataGenerator::generateRandom(src2, (Npp32s)-100, (Npp32s)100, 54321);
  for (int i = 0; i < total; i++) {
    expected[i] = src1[i] * src2[i];
  }

  NppImageMemory<Npp32s> d_src1(width, height);
  NppImageMemory<Npp32s> d_src2(width, height);
  NppImageMemory<Npp32s> d_dst(width, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {width, height};

  NppStatus status =
      nppiMul_32s_C1R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32s> result(total);
  d_dst.copyToHost(result);
  for (int i = 0; i < total; i++) {
    EXPECT_EQ(result[i], expected[i]) << "mismatch at " << i;
  }
}

TEST(Mul32sC1RCtxTest, Basic) {
  const int width = 32, height = 32;
  const int total = width * height;

  std::vector<Npp32s> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp32s)-100, (Npp32s)100, 12345);
  TestDataGenerator::generateRandom(src2, (Npp32s)-100, (Npp32s)100, 54321);
  for (int i = 0; i < total; i++) {
    expected[i] = src1[i] * src2[i];
  }

  NppImageMemory<Npp32s> d_src1(width, height);
  NppImageMemory<Npp32s> d_src2(width, height);
  NppImageMemory<Npp32s> d_dst(width, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {width, height};
  NppStreamContext ctx{};

  NppStatus status = nppiMul_32s_C1R_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                         d_dst.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32s> result(total);
  d_dst.copyToHost(result);
  for (int i = 0; i < total; i++) {
    EXPECT_EQ(result[i], expected[i]) << "mismatch at " << i;
  }
}
