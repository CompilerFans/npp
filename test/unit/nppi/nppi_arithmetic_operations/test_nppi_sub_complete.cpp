#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ============================================================================
// Sub 16f (half precision float) - C1/C3/C4 channels
// ============================================================================

struct Sub16fCompleteParam {
  int width;
  int height;
  int channels;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Sub16fCompleteParamTest : public NppTestBase, public ::testing::WithParamInterface<Sub16fCompleteParam> {};

TEST_P(Sub16fCompleteParamTest, Sub_16f_Complete) {
  const auto &p = GetParam();
  const int total = p.width * p.height * p.channels;

  std::vector<Npp32f> src1_f(total), src2_f(total);
  TestDataGenerator::generateRandom(src1_f, -10.0f, 10.0f, 12345);
  TestDataGenerator::generateRandom(src2_f, -10.0f, 10.0f, 54321);

  std::vector<Npp16f> src1(total), src2(total), expected(total);
  for (int i = 0; i < total; i++) {
    src1[i] = float_to_npp16f_host(src1_f[i]);
    src2[i] = float_to_npp16f_host(src2_f[i]);
    expected[i] = float_to_npp16f_host(src2_f[i] - src1_f[i]);
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
      status = p.use_ctx ? nppiSub_16f_C1IR_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, ctx)
                         : nppiSub_16f_C1IR(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi);
    } else if (p.channels == 3) {
      status = p.use_ctx ? nppiSub_16f_C3IR_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, ctx)
                         : nppiSub_16f_C3IR(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi);
    } else if (p.channels == 4) {
      status = p.use_ctx ? nppiSub_16f_C4IR_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, ctx)
                         : nppiSub_16f_C4IR(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);
    std::vector<Npp16f> result(total);
    d_src2.copyToHost(result);
    EXPECT_TRUE(ResultValidator::arraysEqual16f(result, expected, 0.1f));
  } else {
    NppImageMemory<Npp16f> d_dst(p.width * p.channels, p.height);
    if (p.channels == 1) {
      status = p.use_ctx ? nppiSub_16f_C1R_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                               d_dst.step(), roi, ctx)
                         : nppiSub_16f_C1R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                           d_dst.step(), roi);
    } else if (p.channels == 3) {
      status = p.use_ctx ? nppiSub_16f_C3R_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                               d_dst.step(), roi, ctx)
                         : nppiSub_16f_C3R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                           d_dst.step(), roi);
    } else if (p.channels == 4) {
      status = p.use_ctx ? nppiSub_16f_C4R_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                               d_dst.step(), roi, ctx)
                         : nppiSub_16f_C4R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                           d_dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);
    std::vector<Npp16f> result(total);
    d_dst.copyToHost(result);
    EXPECT_TRUE(ResultValidator::arraysEqual16f(result, expected, 0.1f));
  }
}

INSTANTIATE_TEST_SUITE_P(
    Sub16fComplete, Sub16fCompleteParamTest,
    ::testing::Values(
        Sub16fCompleteParam{32, 32, 1, false, false, "C1R"}, Sub16fCompleteParam{32, 32, 1, true, false, "C1R_Ctx"},
        Sub16fCompleteParam{32, 32, 1, false, true, "C1IR"}, Sub16fCompleteParam{32, 32, 1, true, true, "C1IR_Ctx"},
        Sub16fCompleteParam{32, 32, 3, false, false, "C3R"}, Sub16fCompleteParam{32, 32, 3, true, false, "C3R_Ctx"},
        Sub16fCompleteParam{32, 32, 3, false, true, "C3IR"}, Sub16fCompleteParam{32, 32, 3, true, true, "C3IR_Ctx"},
        Sub16fCompleteParam{32, 32, 4, false, false, "C4R"}, Sub16fCompleteParam{32, 32, 4, true, false, "C4R_Ctx"},
        Sub16fCompleteParam{32, 32, 4, false, true, "C4IR"}, Sub16fCompleteParam{32, 32, 4, true, true, "C4IR_Ctx"}),
    [](const ::testing::TestParamInfo<Sub16fCompleteParam> &info) { return info.param.name; });

// ============================================================================
// Sub 32fc (complex float) - C1/C3/C4/AC4 channels
// ============================================================================

struct Sub32fcParam {
  int width;
  int height;
  int channels;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Sub32fcParamTest : public NppTestBase, public ::testing::WithParamInterface<Sub32fcParam> {};

TEST_P(Sub32fcParamTest, Sub_32fc) {
  const auto &p = GetParam();
  const int ch = (p.channels < 0) ? 4 : p.channels;
  const int total = p.width * p.height * ch;

  std::vector<Npp32fc> src1(total), src2(total), expected(total);
  for (int i = 0; i < total; i++) {
    src1[i].re = static_cast<Npp32f>((i % 100) - 50);
    src1[i].im = static_cast<Npp32f>((i % 50) - 25);
    src2[i].re = static_cast<Npp32f>(((i + 30) % 100) - 50);
    src2[i].im = static_cast<Npp32f>(((i + 15) % 50) - 25);
    if (p.channels == -4 && (i % 4 == 3)) {
      expected[i] = src2[i];
    } else {
      expected[i].re = src2[i].re - src1[i].re;
      expected[i].im = src2[i].im - src1[i].im;
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
      status = p.use_ctx ? nppiSub_32fc_C1IR_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, ctx)
                         : nppiSub_32fc_C1IR(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi);
    } else if (p.channels == 3) {
      status = p.use_ctx ? nppiSub_32fc_C3IR_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, ctx)
                         : nppiSub_32fc_C3IR(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi);
    } else if (p.channels == 4) {
      status = p.use_ctx ? nppiSub_32fc_C4IR_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, ctx)
                         : nppiSub_32fc_C4IR(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi);
    } else if (p.channels == -4) {
      status = p.use_ctx ? nppiSub_32fc_AC4IR_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, ctx)
                         : nppiSub_32fc_AC4IR(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);
    std::vector<Npp32fc> result(total);
    d_src2.copyToHost(result);
    for (int i = 0; i < total; i++) {
      EXPECT_NEAR(result[i].re, expected[i].re, 1e-4f) << "re mismatch at " << i;
      EXPECT_NEAR(result[i].im, expected[i].im, 1e-4f) << "im mismatch at " << i;
    }
  } else {
    NppImageMemory<Npp32fc> d_dst(p.width * ch, p.height);
    if (p.channels == -4)
      d_dst.copyFromHost(src2);
    if (p.channels == 1) {
      status = p.use_ctx ? nppiSub_32fc_C1R_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                d_dst.step(), roi, ctx)
                         : nppiSub_32fc_C1R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                            d_dst.step(), roi);
    } else if (p.channels == 3) {
      status = p.use_ctx ? nppiSub_32fc_C3R_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                d_dst.step(), roi, ctx)
                         : nppiSub_32fc_C3R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                            d_dst.step(), roi);
    } else if (p.channels == 4) {
      status = p.use_ctx ? nppiSub_32fc_C4R_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                d_dst.step(), roi, ctx)
                         : nppiSub_32fc_C4R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                            d_dst.step(), roi);
    } else if (p.channels == -4) {
      status = p.use_ctx ? nppiSub_32fc_AC4R_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                 d_dst.step(), roi, ctx)
                         : nppiSub_32fc_AC4R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                             d_dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);
    std::vector<Npp32fc> result(total);
    d_dst.copyToHost(result);
    for (int i = 0; i < total; i++) {
      EXPECT_NEAR(result[i].re, expected[i].re, 1e-4f) << "re mismatch at " << i;
      EXPECT_NEAR(result[i].im, expected[i].im, 1e-4f) << "im mismatch at " << i;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Sub32fc, Sub32fcParamTest,
    ::testing::Values(Sub32fcParam{32, 32, 1, false, false, "C1R"}, Sub32fcParam{32, 32, 1, true, false, "C1R_Ctx"},
                      Sub32fcParam{32, 32, 1, false, true, "C1IR"}, Sub32fcParam{32, 32, 1, true, true, "C1IR_Ctx"},
                      Sub32fcParam{32, 32, 3, false, false, "C3R"}, Sub32fcParam{32, 32, 3, true, false, "C3R_Ctx"},
                      Sub32fcParam{32, 32, 3, false, true, "C3IR"}, Sub32fcParam{32, 32, 3, true, true, "C3IR_Ctx"},
                      Sub32fcParam{32, 32, 4, false, false, "C4R"}, Sub32fcParam{32, 32, 4, true, false, "C4R_Ctx"},
                      Sub32fcParam{32, 32, 4, false, true, "C4IR"}, Sub32fcParam{32, 32, 4, true, true, "C4IR_Ctx"},
                      Sub32fcParam{32, 32, -4, false, false, "AC4R"}, Sub32fcParam{32, 32, -4, true, false, "AC4R_Ctx"},
                      Sub32fcParam{32, 32, -4, false, true, "AC4IR"},
                      Sub32fcParam{32, 32, -4, true, true, "AC4IR_Ctx"}),
    [](const ::testing::TestParamInfo<Sub32fcParam> &info) { return info.param.name; });

// ============================================================================
// Sub 16sc/32sc (complex integer with scale) - C1/C3/AC4 channels
// ============================================================================

struct Sub16scParam {
  int width;
  int height;
  int channels;
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Sub16scParamTest : public NppTestBase, public ::testing::WithParamInterface<Sub16scParam> {};

TEST_P(Sub16scParamTest, Sub_16sc) {
  const auto &p = GetParam();
  const int ch = (p.channels < 0) ? 4 : p.channels;
  const int total = p.width * p.height * ch;

  std::vector<Npp16sc> src1(total), src2(total), expected(total);
  for (int i = 0; i < total; i++) {
    src1[i].re = static_cast<Npp16s>((i % 200) - 100);
    src1[i].im = static_cast<Npp16s>((i % 100) - 50);
    src2[i].re = static_cast<Npp16s>(((i + 30) % 200) - 100);
    src2[i].im = static_cast<Npp16s>(((i + 15) % 100) - 50);
    if (p.channels == -4 && (i % 4 == 3)) {
      expected[i] = src2[i];
    } else {
      int re = src2[i].re - src1[i].re;
      int im = src2[i].im - src1[i].im;
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
                   ? nppiSub_16sc_C1IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                              p.scaleFactor, ctx)
                   : nppiSub_16sc_C1IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
    } else if (p.channels == 3) {
      status = p.use_ctx
                   ? nppiSub_16sc_C3IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                              p.scaleFactor, ctx)
                   : nppiSub_16sc_C3IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
    } else if (p.channels == -4) {
      status = p.use_ctx ? nppiSub_16sc_AC4IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                     p.scaleFactor, ctx)
                         : nppiSub_16sc_AC4IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
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
      status = p.use_ctx ? nppiSub_16sc_C1RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                   d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                         : nppiSub_16sc_C1RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                               d_dst.step(), roi, p.scaleFactor);
    } else if (p.channels == 3) {
      status = p.use_ctx ? nppiSub_16sc_C3RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                   d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                         : nppiSub_16sc_C3RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                               d_dst.step(), roi, p.scaleFactor);
    } else if (p.channels == -4) {
      status = p.use_ctx ? nppiSub_16sc_AC4RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                    d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                         : nppiSub_16sc_AC4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
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
    Sub16sc, Sub16scParamTest,
    ::testing::Values(
        Sub16scParam{32, 32, 1, 0, false, false, "C1RSfs"}, Sub16scParam{32, 32, 1, 0, true, false, "C1RSfs_Ctx"},
        Sub16scParam{32, 32, 1, 0, false, true, "C1IRSfs"}, Sub16scParam{32, 32, 1, 0, true, true, "C1IRSfs_Ctx"},
        Sub16scParam{32, 32, 3, 0, false, false, "C3RSfs"}, Sub16scParam{32, 32, 3, 0, true, false, "C3RSfs_Ctx"},
        Sub16scParam{32, 32, 3, 0, false, true, "C3IRSfs"}, Sub16scParam{32, 32, 3, 0, true, true, "C3IRSfs_Ctx"},
        Sub16scParam{32, 32, -4, 0, false, false, "AC4RSfs"}, Sub16scParam{32, 32, -4, 0, true, false, "AC4RSfs_Ctx"},
        Sub16scParam{32, 32, -4, 0, false, true, "AC4IRSfs"}, Sub16scParam{32, 32, -4, 0, true, true, "AC4IRSfs_Ctx"}),
    [](const ::testing::TestParamInfo<Sub16scParam> &info) { return info.param.name; });

// ============================================================================
// Sub 32sc (complex 32-bit integer with scale)
// ============================================================================

struct Sub32scParam {
  int width;
  int height;
  int channels;
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Sub32scParamTest : public NppTestBase, public ::testing::WithParamInterface<Sub32scParam> {};

TEST_P(Sub32scParamTest, Sub_32sc) {
  const auto &p = GetParam();
  const int ch = (p.channels < 0) ? 4 : p.channels;
  const int total = p.width * p.height * ch;

  std::vector<Npp32sc> src1(total), src2(total), expected(total);
  for (int i = 0; i < total; i++) {
    src1[i].re = static_cast<Npp32s>((i % 2000) - 1000);
    src1[i].im = static_cast<Npp32s>((i % 1000) - 500);
    src2[i].re = static_cast<Npp32s>(((i + 300) % 2000) - 1000);
    src2[i].im = static_cast<Npp32s>(((i + 150) % 1000) - 500);
    if (p.channels == -4 && (i % 4 == 3)) {
      expected[i] = src2[i];
    } else {
      int64_t re = static_cast<int64_t>(src2[i].re) - src1[i].re;
      int64_t im = static_cast<int64_t>(src2[i].im) - src1[i].im;
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
                   ? nppiSub_32sc_C1IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                              p.scaleFactor, ctx)
                   : nppiSub_32sc_C1IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
    } else if (p.channels == 3) {
      status = p.use_ctx
                   ? nppiSub_32sc_C3IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                              p.scaleFactor, ctx)
                   : nppiSub_32sc_C3IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
    } else if (p.channels == -4) {
      status = p.use_ctx ? nppiSub_32sc_AC4IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                     p.scaleFactor, ctx)
                         : nppiSub_32sc_AC4IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
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
      status = p.use_ctx ? nppiSub_32sc_C1RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                   d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                         : nppiSub_32sc_C1RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                               d_dst.step(), roi, p.scaleFactor);
    } else if (p.channels == 3) {
      status = p.use_ctx ? nppiSub_32sc_C3RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                   d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                         : nppiSub_32sc_C3RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                               d_dst.step(), roi, p.scaleFactor);
    } else if (p.channels == -4) {
      status = p.use_ctx ? nppiSub_32sc_AC4RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                    d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                         : nppiSub_32sc_AC4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
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
    Sub32sc, Sub32scParamTest,
    ::testing::Values(
        Sub32scParam{32, 32, 1, 0, false, false, "C1RSfs"}, Sub32scParam{32, 32, 1, 0, true, false, "C1RSfs_Ctx"},
        Sub32scParam{32, 32, 1, 0, false, true, "C1IRSfs"}, Sub32scParam{32, 32, 1, 0, true, true, "C1IRSfs_Ctx"},
        Sub32scParam{32, 32, 3, 0, false, false, "C3RSfs"}, Sub32scParam{32, 32, 3, 0, true, false, "C3RSfs_Ctx"},
        Sub32scParam{32, 32, 3, 0, false, true, "C3IRSfs"}, Sub32scParam{32, 32, 3, 0, true, true, "C3IRSfs_Ctx"},
        Sub32scParam{32, 32, -4, 0, false, false, "AC4RSfs"}, Sub32scParam{32, 32, -4, 0, true, false, "AC4RSfs_Ctx"},
        Sub32scParam{32, 32, -4, 0, false, true, "AC4IRSfs"}, Sub32scParam{32, 32, -4, 0, true, true, "AC4IRSfs_Ctx"}),
    [](const ::testing::TestParamInfo<Sub32scParam> &info) { return info.param.name; });

// ============================================================================
// Sub remaining integer variants (16u/16s) with Ctx and C4/AC4
// ============================================================================

struct SubIntSfsParam {
  int width;
  int height;
  int channels;
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string dtype;
  std::string name;
};

class SubIntSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<SubIntSfsParam> {};

TEST_P(SubIntSfsParamTest, Sub_IntSfs) {
  const auto &p = GetParam();
  const int ch = (p.channels < 0) ? 4 : p.channels;
  const int total = p.width * p.height * ch;

  NppiSize roi = {p.width, p.height};
  NppStreamContext ctx{};
  NppStatus status = NPP_ERROR;

  if (p.dtype == "16u") {
    std::vector<Npp16u> src1(total), src2(total), expected(total);
    TestDataGenerator::generateRandom(src1, (Npp16u)0, (Npp16u)500, 12345);
    TestDataGenerator::generateRandom(src2, (Npp16u)500, (Npp16u)1000, 54321);
    for (int i = 0; i < total; i++) {
      if (p.channels == -4 && (i % 4 == 3)) {
        expected[i] = src2[i];
      } else {
        expected[i] = expect::sub_sfs<Npp16u>(src2[i], src1[i], p.scaleFactor);
      }
    }
    NppImageMemory<Npp16u> d_src1(p.width * ch, p.height);
    NppImageMemory<Npp16u> d_src2(p.width * ch, p.height);
    d_src1.copyFromHost(src1);
    d_src2.copyFromHost(src2);

    if (p.in_place) {
      if (p.channels == 4) {
        status = p.use_ctx ? nppiSub_16u_C4IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                     p.scaleFactor, ctx)
                           : nppiSub_16u_C4IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                 p.scaleFactor);
      } else if (p.channels == -4) {
        status = p.use_ctx ? nppiSub_16u_AC4IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                      p.scaleFactor, ctx)
                           : nppiSub_16u_AC4IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
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
        status = p.use_ctx ? nppiSub_16u_C4RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                    d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                           : nppiSub_16u_C4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                d_dst.step(), roi, p.scaleFactor);
      } else if (p.channels == -4) {
        status = p.use_ctx ? nppiSub_16u_AC4RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                     d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                           : nppiSub_16u_AC4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                 d_dst.step(), roi, p.scaleFactor);
      }
      ASSERT_EQ(status, NPP_NO_ERROR);
      std::vector<Npp16u> result(total);
      d_dst.copyToHost(result);
      EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
    }
  } else if (p.dtype == "16s") {
    std::vector<Npp16s> src1(total), src2(total), expected(total);
    TestDataGenerator::generateRandom(src1, (Npp16s)-500, (Npp16s)500, 12345);
    TestDataGenerator::generateRandom(src2, (Npp16s)-500, (Npp16s)500, 54321);
    for (int i = 0; i < total; i++) {
      if (p.channels == -4 && (i % 4 == 3)) {
        expected[i] = src2[i];
      } else {
        expected[i] = expect::sub_sfs<Npp16s>(src2[i], src1[i], p.scaleFactor);
      }
    }
    NppImageMemory<Npp16s> d_src1(p.width * ch, p.height);
    NppImageMemory<Npp16s> d_src2(p.width * ch, p.height);
    d_src1.copyFromHost(src1);
    d_src2.copyFromHost(src2);

    if (p.in_place) {
      if (p.channels == 4) {
        status = p.use_ctx ? nppiSub_16s_C4IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                     p.scaleFactor, ctx)
                           : nppiSub_16s_C4IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                 p.scaleFactor);
      } else if (p.channels == -4) {
        status = p.use_ctx ? nppiSub_16s_AC4IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                      p.scaleFactor, ctx)
                           : nppiSub_16s_AC4IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
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
        status = p.use_ctx ? nppiSub_16s_C4RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                    d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                           : nppiSub_16s_C4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                d_dst.step(), roi, p.scaleFactor);
      } else if (p.channels == -4) {
        status = p.use_ctx ? nppiSub_16s_AC4RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                     d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                           : nppiSub_16s_AC4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                 d_dst.step(), roi, p.scaleFactor);
      }
      ASSERT_EQ(status, NPP_NO_ERROR);
      std::vector<Npp16s> result(total);
      d_dst.copyToHost(result);
      EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(SubIntSfs, SubIntSfsParamTest,
                         ::testing::Values(
                             // 16u variants
                             SubIntSfsParam{32, 32, 4, 0, false, false, "16u", "16u_C4RSfs"},
                             SubIntSfsParam{32, 32, 4, 0, true, false, "16u", "16u_C4RSfs_Ctx"},
                             SubIntSfsParam{32, 32, 4, 0, false, true, "16u", "16u_C4IRSfs"},
                             SubIntSfsParam{32, 32, 4, 0, true, true, "16u", "16u_C4IRSfs_Ctx"},
                             SubIntSfsParam{32, 32, -4, 0, true, false, "16u", "16u_AC4RSfs_Ctx"},
                             SubIntSfsParam{32, 32, -4, 0, false, true, "16u", "16u_AC4IRSfs"},
                             SubIntSfsParam{32, 32, -4, 0, true, true, "16u", "16u_AC4IRSfs_Ctx"},
                             // 16s variants
                             SubIntSfsParam{32, 32, 4, 0, false, false, "16s", "16s_C4RSfs"},
                             SubIntSfsParam{32, 32, 4, 0, true, false, "16s", "16s_C4RSfs_Ctx"},
                             SubIntSfsParam{32, 32, 4, 0, false, true, "16s", "16s_C4IRSfs"},
                             SubIntSfsParam{32, 32, 4, 0, true, true, "16s", "16s_C4IRSfs_Ctx"},
                             SubIntSfsParam{32, 32, -4, 0, true, false, "16s", "16s_AC4RSfs_Ctx"},
                             SubIntSfsParam{32, 32, -4, 0, false, true, "16s", "16s_AC4IRSfs"},
                             SubIntSfsParam{32, 32, -4, 0, true, true, "16s", "16s_AC4IRSfs_Ctx"}),
                         [](const ::testing::TestParamInfo<SubIntSfsParam> &info) { return info.param.name; });

// ============================================================================
// Sub 8u variants (C3/C4/AC4)
// ============================================================================

struct Sub8uSfsCompleteParam {
  int width;
  int height;
  int channels;
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Sub8uSfsCompleteParamTest : public NppTestBase, public ::testing::WithParamInterface<Sub8uSfsCompleteParam> {};

TEST_P(Sub8uSfsCompleteParamTest, Sub_8u_Sfs_Complete) {
  const auto &p = GetParam();
  const int ch = (p.channels < 0) ? 4 : p.channels;
  const int total = p.width * p.height * ch;

  std::vector<Npp8u> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp8u)0, (Npp8u)100, 12345);
  TestDataGenerator::generateRandom(src2, (Npp8u)100, (Npp8u)200, 54321);
  for (int i = 0; i < total; i++) {
    if (p.channels == -4 && (i % 4 == 3)) {
      expected[i] = src2[i];
    } else {
      expected[i] = expect::sub_sfs<Npp8u>(src2[i], src1[i], p.scaleFactor);
    }
  }

  NppImageMemory<Npp8u> d_src1(p.width * ch, p.height);
  NppImageMemory<Npp8u> d_src2(p.width * ch, p.height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {p.width, p.height};
  NppStreamContext ctx{};
  NppStatus status = NPP_ERROR;

  if (p.in_place) {
    if (p.channels == 3) {
      status = p.use_ctx
                   ? nppiSub_8u_C3IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                            p.scaleFactor, ctx)
                   : nppiSub_8u_C3IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
    } else if (p.channels == 4) {
      status = p.use_ctx
                   ? nppiSub_8u_C4IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                            p.scaleFactor, ctx)
                   : nppiSub_8u_C4IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
    } else if (p.channels == -4) {
      status = p.use_ctx
                   ? nppiSub_8u_AC4IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                             p.scaleFactor, ctx)
                   : nppiSub_8u_AC4IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
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
      status = p.use_ctx ? nppiSub_8u_C3RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                 d_dst.step(), roi, p.scaleFactor, ctx)
                         : nppiSub_8u_C3RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                             d_dst.step(), roi, p.scaleFactor);
    } else if (p.channels == 4) {
      status = p.use_ctx ? nppiSub_8u_C4RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                 d_dst.step(), roi, p.scaleFactor, ctx)
                         : nppiSub_8u_C4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                             d_dst.step(), roi, p.scaleFactor);
    } else if (p.channels == -4) {
      status = p.use_ctx ? nppiSub_8u_AC4RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                  d_dst.step(), roi, p.scaleFactor, ctx)
                         : nppiSub_8u_AC4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                              d_dst.step(), roi, p.scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);
    std::vector<Npp8u> result(total);
    d_dst.copyToHost(result);
    EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
  }
}

INSTANTIATE_TEST_SUITE_P(Sub8uSfsComplete, Sub8uSfsCompleteParamTest,
                         ::testing::Values(Sub8uSfsCompleteParam{32, 32, 3, 0, true, false, "C3RSfs_Ctx"},
                                           Sub8uSfsCompleteParam{32, 32, 3, 0, false, true, "C3IRSfs"},
                                           Sub8uSfsCompleteParam{32, 32, 3, 0, true, true, "C3IRSfs_Ctx"},
                                           Sub8uSfsCompleteParam{32, 32, 4, 0, false, false, "C4RSfs"},
                                           Sub8uSfsCompleteParam{32, 32, 4, 0, true, false, "C4RSfs_Ctx"},
                                           Sub8uSfsCompleteParam{32, 32, 4, 0, false, true, "C4IRSfs"},
                                           Sub8uSfsCompleteParam{32, 32, 4, 0, true, true, "C4IRSfs_Ctx"},
                                           Sub8uSfsCompleteParam{32, 32, -4, 0, true, false, "AC4RSfs_Ctx"},
                                           Sub8uSfsCompleteParam{32, 32, -4, 0, false, true, "AC4IRSfs"},
                                           Sub8uSfsCompleteParam{32, 32, -4, 0, true, true, "AC4IRSfs_Ctx"}),
                         [](const ::testing::TestParamInfo<Sub8uSfsCompleteParam> &info) { return info.param.name; });

// ============================================================================
// Sub 32s variants (C1/C3/C4)
// ============================================================================

struct Sub32sSfsParam {
  int width;
  int height;
  int channels;
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Sub32sSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<Sub32sSfsParam> {};

TEST_P(Sub32sSfsParamTest, Sub_32s_Sfs) {
  const auto &p = GetParam();
  const int ch = (p.channels < 0) ? 4 : p.channels;
  const int total = p.width * p.height * ch;

  std::vector<Npp32s> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp32s)-10000, (Npp32s)10000, 12345);
  TestDataGenerator::generateRandom(src2, (Npp32s)-10000, (Npp32s)10000, 54321);
  for (int i = 0; i < total; i++) {
    expected[i] = src2[i] - src1[i];
  }

  NppImageMemory<Npp32s> d_src1(p.width * ch, p.height);
  NppImageMemory<Npp32s> d_src2(p.width * ch, p.height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {p.width, p.height};
  NppStreamContext ctx{};
  NppStatus status = NPP_ERROR;

  if (p.in_place) {
    if (p.channels == 1) {
      status = p.use_ctx
                   ? nppiSub_32s_C1IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                             p.scaleFactor, ctx)
                   : nppiSub_32s_C1IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
    } else if (p.channels == 3) {
      status = p.use_ctx
                   ? nppiSub_32s_C3IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                             p.scaleFactor, ctx)
                   : nppiSub_32s_C3IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);
    std::vector<Npp32s> result(total);
    d_src2.copyToHost(result);
    for (int i = 0; i < total; i++) {
      EXPECT_LE(std::abs(result[i] - expected[i]), 1) << "mismatch at " << i;
    }
  } else {
    NppImageMemory<Npp32s> d_dst(p.width * ch, p.height);
    if (p.channels == 1) {
      status = p.use_ctx ? nppiSub_32s_C1RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                  d_dst.step(), roi, p.scaleFactor, ctx)
                         : nppiSub_32s_C1RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                              d_dst.step(), roi, p.scaleFactor);
    } else if (p.channels == 3) {
      status = p.use_ctx ? nppiSub_32s_C3RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                  d_dst.step(), roi, p.scaleFactor, ctx)
                         : nppiSub_32s_C3RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                              d_dst.step(), roi, p.scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);
    std::vector<Npp32s> result(total);
    d_dst.copyToHost(result);
    for (int i = 0; i < total; i++) {
      EXPECT_LE(std::abs(result[i] - expected[i]), 1) << "mismatch at " << i;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Sub32sSfs, Sub32sSfsParamTest,
                         ::testing::Values(Sub32sSfsParam{32, 32, 1, 0, true, false, "C1RSfs_Ctx"},
                                           Sub32sSfsParam{32, 32, 1, 0, false, true, "C1IRSfs"},
                                           Sub32sSfsParam{32, 32, 1, 0, true, true, "C1IRSfs_Ctx"},
                                           Sub32sSfsParam{32, 32, 3, 0, true, false, "C3RSfs_Ctx"},
                                           Sub32sSfsParam{32, 32, 3, 0, false, true, "C3IRSfs"},
                                           Sub32sSfsParam{32, 32, 3, 0, true, true, "C3IRSfs_Ctx"}),
                         [](const ::testing::TestParamInfo<Sub32sSfsParam> &info) { return info.param.name; });

// ============================================================================
// Sub_32s_C1R (non-Sfs version)
// ============================================================================

TEST(Sub32sC1RTest, Basic) {
  const int width = 32, height = 32;
  const int total = width * height;

  std::vector<Npp32s> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp32s)-10000, (Npp32s)10000, 12345);
  TestDataGenerator::generateRandom(src2, (Npp32s)-10000, (Npp32s)10000, 54321);
  for (int i = 0; i < total; i++) {
    expected[i] = src2[i] - src1[i];
  }

  NppImageMemory<Npp32s> d_src1(width, height);
  NppImageMemory<Npp32s> d_src2(width, height);
  NppImageMemory<Npp32s> d_dst(width, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {width, height};
  NppStatus status =
      nppiSub_32s_C1R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32s> result(total);
  d_dst.copyToHost(result);
  for (int i = 0; i < total; i++) {
    EXPECT_LE(std::abs(result[i] - expected[i]), 1) << "mismatch at " << i;
  }
}

// ============================================================================
// Sub float AC4 variants
// ============================================================================

struct Sub32fAC4Param {
  int width;
  int height;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Sub32fAC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<Sub32fAC4Param> {};

TEST_P(Sub32fAC4ParamTest, Sub_32f_AC4) {
  const auto &p = GetParam();
  const int channels = 4;
  const int total = p.width * p.height * channels;

  std::vector<Npp32f> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, -50.0f, 50.0f, 12345);
  TestDataGenerator::generateRandom(src2, -50.0f, 50.0f, 54321);
  for (int i = 0; i < total; i++) {
    if (i % 4 == 3) {
      expected[i] = src2[i];
    } else {
      expected[i] = src2[i] - src1[i];
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
    status = p.use_ctx ? nppiSub_32f_AC4IR_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, ctx)
                       : nppiSub_32f_AC4IR(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi);
    ASSERT_EQ(status, NPP_NO_ERROR);
    std::vector<Npp32f> result(total);
    d_src2.copyToHost(result);
    EXPECT_TRUE(ResultValidator::arraysEqual(result, expected, 1e-5f));
  } else {
    NppImageMemory<Npp32f> d_dst(p.width * channels, p.height);
    d_dst.copyFromHost(src2);
    status = p.use_ctx ? nppiSub_32f_AC4R_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                              d_dst.step(), roi, ctx)
                       : nppiSub_32f_AC4R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                          d_dst.step(), roi);
    ASSERT_EQ(status, NPP_NO_ERROR);
    std::vector<Npp32f> result(total);
    d_dst.copyToHost(result);
    EXPECT_TRUE(ResultValidator::arraysEqual(result, expected, 1e-5f));
  }
}

INSTANTIATE_TEST_SUITE_P(Sub32fAC4, Sub32fAC4ParamTest,
                         ::testing::Values(Sub32fAC4Param{32, 32, false, false, "AC4R"},
                                           Sub32fAC4Param{32, 32, true, false, "AC4R_Ctx"},
                                           Sub32fAC4Param{32, 32, false, true, "AC4IR"},
                                           Sub32fAC4Param{32, 32, true, true, "AC4IR_Ctx"}),
                         [](const ::testing::TestParamInfo<Sub32fAC4Param> &info) { return info.param.name; });

// ============================================================================
// Sub_32s_C1R_Ctx
// ============================================================================

TEST(Sub32sC1RCtxTest, Basic) {
  const int width = 32, height = 32;
  const int total = width * height;

  std::vector<Npp32s> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp32s)-10000, (Npp32s)10000, 12345);
  TestDataGenerator::generateRandom(src2, (Npp32s)-10000, (Npp32s)10000, 54321);
  for (int i = 0; i < total; i++) {
    expected[i] = src2[i] - src1[i];
  }

  NppImageMemory<Npp32s> d_src1(width, height);
  NppImageMemory<Npp32s> d_src2(width, height);
  NppImageMemory<Npp32s> d_dst(width, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {width, height};
  NppStreamContext ctx{};

  NppStatus status = nppiSub_32s_C1R_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                         d_dst.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32s> result(total);
  d_dst.copyToHost(result);
  for (int i = 0; i < total; i++) {
    EXPECT_LE(std::abs(result[i] - expected[i]), 1) << "mismatch at " << i;
  }
}
