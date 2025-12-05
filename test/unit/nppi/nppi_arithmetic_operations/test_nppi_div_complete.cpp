#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ============================================================================
// Div 32fc (complex float) - C1/C3/C4/AC4 channels
// Note: NPP Div computes dst = src2 / src1
// ============================================================================

struct Div32fcParam {
  int width;
  int height;
  int channels; // 1, 3, 4, or -4 for AC4
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Div32fcParamTest : public NppTestBase, public ::testing::WithParamInterface<Div32fcParam> {};

TEST_P(Div32fcParamTest, Div_32fc) {
  const auto &p = GetParam();
  const int ch = (p.channels < 0) ? 4 : p.channels;
  const int total = p.width * p.height * ch;

  std::vector<Npp32fc> src1(total), src2(total), expected(total);
  for (int i = 0; i < total; i++) {
    // Avoid division by zero - use non-zero complex values
    src1[i].re = static_cast<Npp32f>((i % 10) + 1);
    src1[i].im = static_cast<Npp32f>((i % 5) + 1);
    src2[i].re = static_cast<Npp32f>(((i + 7) % 20) - 10);
    src2[i].im = static_cast<Npp32f>(((i + 3) % 10) - 5);
    if (p.channels == -4 && (i % 4 == 3)) {
      expected[i] = src2[i]; // AC4: alpha unchanged
    } else {
      // Complex division: (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c^2+d^2)
      float denom = src1[i].re * src1[i].re + src1[i].im * src1[i].im;
      expected[i].re = (src2[i].re * src1[i].re + src2[i].im * src1[i].im) / denom;
      expected[i].im = (src2[i].im * src1[i].re - src2[i].re * src1[i].im) / denom;
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
      status = p.use_ctx ? nppiDiv_32fc_C1IR_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, ctx)
                         : nppiDiv_32fc_C1IR(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi);
    } else if (p.channels == 3) {
      status = p.use_ctx ? nppiDiv_32fc_C3IR_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, ctx)
                         : nppiDiv_32fc_C3IR(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi);
    } else if (p.channels == 4) {
      status = p.use_ctx ? nppiDiv_32fc_C4IR_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, ctx)
                         : nppiDiv_32fc_C4IR(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi);
    } else if (p.channels == -4) {
      status = p.use_ctx ? nppiDiv_32fc_AC4IR_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, ctx)
                         : nppiDiv_32fc_AC4IR(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi);
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
      status = p.use_ctx ? nppiDiv_32fc_C1R_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                d_dst.step(), roi, ctx)
                         : nppiDiv_32fc_C1R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                            d_dst.step(), roi);
    } else if (p.channels == 3) {
      status = p.use_ctx ? nppiDiv_32fc_C3R_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                d_dst.step(), roi, ctx)
                         : nppiDiv_32fc_C3R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                            d_dst.step(), roi);
    } else if (p.channels == 4) {
      status = p.use_ctx ? nppiDiv_32fc_C4R_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                d_dst.step(), roi, ctx)
                         : nppiDiv_32fc_C4R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                            d_dst.step(), roi);
    } else if (p.channels == -4) {
      status = p.use_ctx ? nppiDiv_32fc_AC4R_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                 d_dst.step(), roi, ctx)
                         : nppiDiv_32fc_AC4R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
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
    Div32fc, Div32fcParamTest,
    ::testing::Values(Div32fcParam{32, 32, 1, false, false, "C1R"}, Div32fcParam{32, 32, 1, true, false, "C1R_Ctx"},
                      Div32fcParam{32, 32, 1, false, true, "C1IR"}, Div32fcParam{32, 32, 1, true, true, "C1IR_Ctx"},
                      Div32fcParam{32, 32, 3, false, false, "C3R"}, Div32fcParam{32, 32, 3, true, false, "C3R_Ctx"},
                      Div32fcParam{32, 32, 3, false, true, "C3IR"}, Div32fcParam{32, 32, 3, true, true, "C3IR_Ctx"},
                      Div32fcParam{32, 32, 4, false, false, "C4R"}, Div32fcParam{32, 32, 4, true, false, "C4R_Ctx"},
                      Div32fcParam{32, 32, 4, false, true, "C4IR"}, Div32fcParam{32, 32, 4, true, true, "C4IR_Ctx"},
                      Div32fcParam{32, 32, -4, false, false, "AC4R"}, Div32fcParam{32, 32, -4, true, false, "AC4R_Ctx"},
                      Div32fcParam{32, 32, -4, false, true, "AC4IR"},
                      Div32fcParam{32, 32, -4, true, true, "AC4IR_Ctx"}),
    [](const ::testing::TestParamInfo<Div32fcParam> &info) { return info.param.name; });

// ============================================================================
// Div 16sc (complex 16-bit signed with scale)
// ============================================================================

struct Div16scParam {
  int width;
  int height;
  int channels; // 1, 3, or -4 for AC4
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Div16scParamTest : public NppTestBase, public ::testing::WithParamInterface<Div16scParam> {};

TEST_P(Div16scParamTest, Div_16sc) {
  const auto &p = GetParam();
  const int ch = (p.channels < 0) ? 4 : p.channels;
  const int total = p.width * p.height * ch;

  std::vector<Npp16sc> src1(total), src2(total), expected(total);
  for (int i = 0; i < total; i++) {
    // Avoid division by zero
    src1[i].re = static_cast<Npp16s>((i % 10) + 5);
    src1[i].im = static_cast<Npp16s>((i % 5) + 1);
    src2[i].re = static_cast<Npp16s>(((i + 7) % 40) - 20);
    src2[i].im = static_cast<Npp16s>(((i + 3) % 20) - 10);
    if (p.channels == -4 && (i % 4 == 3)) {
      expected[i] = src2[i];
    } else {
      // Complex division with scale factor
      int denom = src1[i].re * src1[i].re + src1[i].im * src1[i].im;
      int re = (src2[i].re * src1[i].re + src2[i].im * src1[i].im);
      int im = (src2[i].im * src1[i].re - src2[i].re * src1[i].im);
      // Apply scale and division
      if (p.scaleFactor > 0) {
        re = ((re * (1 << p.scaleFactor)) + denom / 2) / denom;
        im = ((im * (1 << p.scaleFactor)) + denom / 2) / denom;
      } else {
        re = (re + denom / 2) / denom;
        im = (im + denom / 2) / denom;
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
                   ? nppiDiv_16sc_C1IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                              p.scaleFactor, ctx)
                   : nppiDiv_16sc_C1IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
    } else if (p.channels == 3) {
      status = p.use_ctx
                   ? nppiDiv_16sc_C3IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                              p.scaleFactor, ctx)
                   : nppiDiv_16sc_C3IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
    } else if (p.channels == -4) {
      status = p.use_ctx ? nppiDiv_16sc_AC4IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                     p.scaleFactor, ctx)
                         : nppiDiv_16sc_AC4IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                 p.scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);
    std::vector<Npp16sc> result(total);
    d_src2.copyToHost(result);
    for (int i = 0; i < total; i++) {
      EXPECT_LE(std::abs(result[i].re - expected[i].re), 2) << "re mismatch at " << i;
      EXPECT_LE(std::abs(result[i].im - expected[i].im), 2) << "im mismatch at " << i;
    }
  } else {
    NppImageMemory<Npp16sc> d_dst(p.width * ch, p.height);
    if (p.channels == -4)
      d_dst.copyFromHost(src2);
    if (p.channels == 1) {
      status = p.use_ctx ? nppiDiv_16sc_C1RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                   d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                         : nppiDiv_16sc_C1RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                               d_dst.step(), roi, p.scaleFactor);
    } else if (p.channels == 3) {
      status = p.use_ctx ? nppiDiv_16sc_C3RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                   d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                         : nppiDiv_16sc_C3RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                               d_dst.step(), roi, p.scaleFactor);
    } else if (p.channels == -4) {
      status = p.use_ctx ? nppiDiv_16sc_AC4RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                    d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                         : nppiDiv_16sc_AC4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                d_dst.step(), roi, p.scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);
    std::vector<Npp16sc> result(total);
    d_dst.copyToHost(result);
    for (int i = 0; i < total; i++) {
      EXPECT_LE(std::abs(result[i].re - expected[i].re), 2) << "re mismatch at " << i;
      EXPECT_LE(std::abs(result[i].im - expected[i].im), 2) << "im mismatch at " << i;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Div16sc, Div16scParamTest,
    ::testing::Values(
        Div16scParam{32, 32, 1, 0, false, false, "C1RSfs"}, Div16scParam{32, 32, 1, 0, true, false, "C1RSfs_Ctx"},
        Div16scParam{32, 32, 1, 0, false, true, "C1IRSfs"}, Div16scParam{32, 32, 1, 0, true, true, "C1IRSfs_Ctx"},
        Div16scParam{32, 32, 3, 0, false, false, "C3RSfs"}, Div16scParam{32, 32, 3, 0, true, false, "C3RSfs_Ctx"},
        Div16scParam{32, 32, 3, 0, false, true, "C3IRSfs"}, Div16scParam{32, 32, 3, 0, true, true, "C3IRSfs_Ctx"},
        Div16scParam{32, 32, -4, 0, false, false, "AC4RSfs"}, Div16scParam{32, 32, -4, 0, true, false, "AC4RSfs_Ctx"},
        Div16scParam{32, 32, -4, 0, false, true, "AC4IRSfs"}, Div16scParam{32, 32, -4, 0, true, true, "AC4IRSfs_Ctx"}),
    [](const ::testing::TestParamInfo<Div16scParam> &info) { return info.param.name; });

// ============================================================================
// Div 32sc (complex 32-bit signed with scale)
// ============================================================================

struct Div32scParam {
  int width;
  int height;
  int channels;
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Div32scParamTest : public NppTestBase, public ::testing::WithParamInterface<Div32scParam> {};

TEST_P(Div32scParamTest, Div_32sc) {
  const auto &p = GetParam();
  const int ch = (p.channels < 0) ? 4 : p.channels;
  const int total = p.width * p.height * ch;

  std::vector<Npp32sc> src1(total), src2(total), expected(total);
  for (int i = 0; i < total; i++) {
    // Avoid division by zero
    src1[i].re = static_cast<Npp32s>((i % 100) + 10);
    src1[i].im = static_cast<Npp32s>((i % 50) + 5);
    src2[i].re = static_cast<Npp32s>(((i + 70) % 400) - 200);
    src2[i].im = static_cast<Npp32s>(((i + 30) % 200) - 100);
    if (p.channels == -4 && (i % 4 == 3)) {
      expected[i] = src2[i];
    } else {
      // Complex division
      int64_t denom = static_cast<int64_t>(src1[i].re) * src1[i].re + static_cast<int64_t>(src1[i].im) * src1[i].im;
      int64_t re = static_cast<int64_t>(src2[i].re) * src1[i].re + static_cast<int64_t>(src2[i].im) * src1[i].im;
      int64_t im = static_cast<int64_t>(src2[i].im) * src1[i].re - static_cast<int64_t>(src2[i].re) * src1[i].im;
      if (p.scaleFactor > 0) {
        re = ((re << p.scaleFactor) + denom / 2) / denom;
        im = ((im << p.scaleFactor) + denom / 2) / denom;
      } else {
        re = (re + denom / 2) / denom;
        im = (im + denom / 2) / denom;
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
                   ? nppiDiv_32sc_C1IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                              p.scaleFactor, ctx)
                   : nppiDiv_32sc_C1IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
    } else if (p.channels == 3) {
      status = p.use_ctx
                   ? nppiDiv_32sc_C3IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                              p.scaleFactor, ctx)
                   : nppiDiv_32sc_C3IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
    } else if (p.channels == -4) {
      status = p.use_ctx ? nppiDiv_32sc_AC4IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                     p.scaleFactor, ctx)
                         : nppiDiv_32sc_AC4IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                 p.scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);
    std::vector<Npp32sc> result(total);
    d_src2.copyToHost(result);
    for (int i = 0; i < total; i++) {
      EXPECT_LE(std::abs(result[i].re - expected[i].re), 2) << "re mismatch at " << i;
      EXPECT_LE(std::abs(result[i].im - expected[i].im), 2) << "im mismatch at " << i;
    }
  } else {
    NppImageMemory<Npp32sc> d_dst(p.width * ch, p.height);
    if (p.channels == -4)
      d_dst.copyFromHost(src2);
    if (p.channels == 1) {
      status = p.use_ctx ? nppiDiv_32sc_C1RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                   d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                         : nppiDiv_32sc_C1RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                               d_dst.step(), roi, p.scaleFactor);
    } else if (p.channels == 3) {
      status = p.use_ctx ? nppiDiv_32sc_C3RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                   d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                         : nppiDiv_32sc_C3RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                               d_dst.step(), roi, p.scaleFactor);
    } else if (p.channels == -4) {
      status = p.use_ctx ? nppiDiv_32sc_AC4RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                    d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                         : nppiDiv_32sc_AC4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                d_dst.step(), roi, p.scaleFactor);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);
    std::vector<Npp32sc> result(total);
    d_dst.copyToHost(result);
    for (int i = 0; i < total; i++) {
      EXPECT_LE(std::abs(result[i].re - expected[i].re), 2) << "re mismatch at " << i;
      EXPECT_LE(std::abs(result[i].im - expected[i].im), 2) << "im mismatch at " << i;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Div32sc, Div32scParamTest,
    ::testing::Values(
        Div32scParam{32, 32, 1, 0, false, false, "C1RSfs"}, Div32scParam{32, 32, 1, 0, true, false, "C1RSfs_Ctx"},
        Div32scParam{32, 32, 1, 0, false, true, "C1IRSfs"}, Div32scParam{32, 32, 1, 0, true, true, "C1IRSfs_Ctx"},
        Div32scParam{32, 32, 3, 0, false, false, "C3RSfs"}, Div32scParam{32, 32, 3, 0, true, false, "C3RSfs_Ctx"},
        Div32scParam{32, 32, 3, 0, false, true, "C3IRSfs"}, Div32scParam{32, 32, 3, 0, true, true, "C3IRSfs_Ctx"},
        Div32scParam{32, 32, -4, 0, false, false, "AC4RSfs"}, Div32scParam{32, 32, -4, 0, true, false, "AC4RSfs_Ctx"},
        Div32scParam{32, 32, -4, 0, false, true, "AC4IRSfs"}, Div32scParam{32, 32, -4, 0, true, true, "AC4IRSfs_Ctx"}),
    [](const ::testing::TestParamInfo<Div32scParam> &info) { return info.param.name; });

// ============================================================================
// Div remaining integer variants (8u/16u/16s/32s) - C3/C4/AC4 with Ctx and IR
// ============================================================================

struct DivIntSfsCompleteParam {
  int width;
  int height;
  int channels; // 3, 4, or -4 for AC4
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string dtype; // "8u", "16u", "16s", "32s"
  std::string name;
};

class DivIntSfsCompleteParamTest : public NppTestBase, public ::testing::WithParamInterface<DivIntSfsCompleteParam> {};

TEST_P(DivIntSfsCompleteParamTest, Div_IntSfs_Complete) {
  const auto &p = GetParam();
  const int ch = (p.channels < 0) ? 4 : p.channels;
  const int total = p.width * p.height * ch;

  NppiSize roi = {p.width, p.height};
  NppStreamContext ctx{};
  NppStatus status = NPP_ERROR;

  if (p.dtype == "8u") {
    std::vector<Npp8u> src1(total), src2(total);
    TestDataGenerator::generateRandom(src1, (Npp8u)10, (Npp8u)200, 12345);
    TestDataGenerator::generateRandom(src2, (Npp8u)1, (Npp8u)10, 54321);
    NppImageMemory<Npp8u> d_src1(p.width * ch, p.height);
    NppImageMemory<Npp8u> d_src2(p.width * ch, p.height);
    d_src1.copyFromHost(src1);
    d_src2.copyFromHost(src2);

    if (p.in_place) {
      if (p.channels == 3) {
        status = p.use_ctx
                     ? nppiDiv_8u_C3IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                              p.scaleFactor, ctx)
                     : nppiDiv_8u_C3IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
      } else if (p.channels == 4) {
        status = p.use_ctx
                     ? nppiDiv_8u_C4IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                              p.scaleFactor, ctx)
                     : nppiDiv_8u_C4IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
      } else if (p.channels == -4) {
        status = p.use_ctx ? nppiDiv_8u_AC4IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                     p.scaleFactor, ctx)
                           : nppiDiv_8u_AC4IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                 p.scaleFactor);
      }
      ASSERT_EQ(status, NPP_NO_ERROR);
      std::vector<Npp8u> result(total);
      d_src2.copyToHost(result);
      // Tolerance-based verification for integer division
      for (int i = 0; i < total; i++) {
        if (p.channels == -4 && (i % 4 == 3))
          continue; // Skip alpha
        int expected = src2[i] / src1[i];
        EXPECT_LE(std::abs(static_cast<int>(result[i]) - expected), 1);
      }
    } else {
      NppImageMemory<Npp8u> d_dst(p.width * ch, p.height);
      if (p.channels == -4)
        d_dst.copyFromHost(src2);
      if (p.channels == 3) {
        status = p.use_ctx ? nppiDiv_8u_C3RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                   d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                           : nppiDiv_8u_C3RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                               d_dst.step(), roi, p.scaleFactor);
      } else if (p.channels == 4) {
        status = p.use_ctx ? nppiDiv_8u_C4RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                   d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                           : nppiDiv_8u_C4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                               d_dst.step(), roi, p.scaleFactor);
      } else if (p.channels == -4) {
        status = p.use_ctx ? nppiDiv_8u_AC4RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                    d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                           : nppiDiv_8u_AC4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                d_dst.step(), roi, p.scaleFactor);
      }
      ASSERT_EQ(status, NPP_NO_ERROR);
      std::vector<Npp8u> result(total);
      d_dst.copyToHost(result);
      for (int i = 0; i < total; i++) {
        if (p.channels == -4 && (i % 4 == 3))
          continue;
        int expected = src2[i] / src1[i];
        EXPECT_LE(std::abs(static_cast<int>(result[i]) - expected), 1);
      }
    }
  } else if (p.dtype == "16u") {
    std::vector<Npp16u> src1(total), src2(total);
    TestDataGenerator::generateRandom(src1, (Npp16u)100, (Npp16u)10000, 12345);
    TestDataGenerator::generateRandom(src2, (Npp16u)1, (Npp16u)100, 54321);
    NppImageMemory<Npp16u> d_src1(p.width * ch, p.height);
    NppImageMemory<Npp16u> d_src2(p.width * ch, p.height);
    d_src1.copyFromHost(src1);
    d_src2.copyFromHost(src2);

    if (p.in_place) {
      if (p.channels == 4) {
        status = p.use_ctx ? nppiDiv_16u_C4IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                     p.scaleFactor, ctx)
                           : nppiDiv_16u_C4IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                 p.scaleFactor);
      } else if (p.channels == -4) {
        status = p.use_ctx ? nppiDiv_16u_AC4IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                      p.scaleFactor, ctx)
                           : nppiDiv_16u_AC4IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                  p.scaleFactor);
      }
      ASSERT_EQ(status, NPP_NO_ERROR);
      std::vector<Npp16u> result(total);
      d_src2.copyToHost(result);
      for (int i = 0; i < total; i++) {
        if (p.channels == -4 && (i % 4 == 3))
          continue;
        int expected = src2[i] / src1[i];
        EXPECT_LE(std::abs(static_cast<int>(result[i]) - expected), 1);
      }
    } else {
      NppImageMemory<Npp16u> d_dst(p.width * ch, p.height);
      if (p.channels == -4)
        d_dst.copyFromHost(src2);
      if (p.channels == 4) {
        status = p.use_ctx ? nppiDiv_16u_C4RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                    d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                           : nppiDiv_16u_C4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                d_dst.step(), roi, p.scaleFactor);
      } else if (p.channels == -4) {
        status = p.use_ctx ? nppiDiv_16u_AC4RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                     d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                           : nppiDiv_16u_AC4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                 d_dst.step(), roi, p.scaleFactor);
      }
      ASSERT_EQ(status, NPP_NO_ERROR);
      std::vector<Npp16u> result(total);
      d_dst.copyToHost(result);
      for (int i = 0; i < total; i++) {
        if (p.channels == -4 && (i % 4 == 3))
          continue;
        int expected = src2[i] / src1[i];
        EXPECT_LE(std::abs(static_cast<int>(result[i]) - expected), 1);
      }
    }
  } else if (p.dtype == "16s") {
    std::vector<Npp16s> src1(total), src2(total);
    TestDataGenerator::generateRandom(src1, (Npp16s)10, (Npp16s)1000, 12345);
    TestDataGenerator::generateRandom(src2, (Npp16s)1, (Npp16s)10, 54321);
    NppImageMemory<Npp16s> d_src1(p.width * ch, p.height);
    NppImageMemory<Npp16s> d_src2(p.width * ch, p.height);
    d_src1.copyFromHost(src1);
    d_src2.copyFromHost(src2);

    if (p.in_place) {
      if (p.channels == 4) {
        status = p.use_ctx ? nppiDiv_16s_C4IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                     p.scaleFactor, ctx)
                           : nppiDiv_16s_C4IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                 p.scaleFactor);
      } else if (p.channels == -4) {
        status = p.use_ctx ? nppiDiv_16s_AC4IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                      p.scaleFactor, ctx)
                           : nppiDiv_16s_AC4IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                  p.scaleFactor);
      }
      ASSERT_EQ(status, NPP_NO_ERROR);
      std::vector<Npp16s> result(total);
      d_src2.copyToHost(result);
      for (int i = 0; i < total; i++) {
        if (p.channels == -4 && (i % 4 == 3))
          continue;
        int expected = src2[i] / src1[i];
        EXPECT_LE(std::abs(static_cast<int>(result[i]) - expected), 1);
      }
    } else {
      NppImageMemory<Npp16s> d_dst(p.width * ch, p.height);
      if (p.channels == -4)
        d_dst.copyFromHost(src2);
      if (p.channels == 4) {
        status = p.use_ctx ? nppiDiv_16s_C4RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                    d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                           : nppiDiv_16s_C4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                d_dst.step(), roi, p.scaleFactor);
      } else if (p.channels == -4) {
        status = p.use_ctx ? nppiDiv_16s_AC4RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                     d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                           : nppiDiv_16s_AC4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                 d_dst.step(), roi, p.scaleFactor);
      }
      ASSERT_EQ(status, NPP_NO_ERROR);
      std::vector<Npp16s> result(total);
      d_dst.copyToHost(result);
      for (int i = 0; i < total; i++) {
        if (p.channels == -4 && (i % 4 == 3))
          continue;
        int expected = src2[i] / src1[i];
        EXPECT_LE(std::abs(static_cast<int>(result[i]) - expected), 1);
      }
    }
  } else if (p.dtype == "32s") {
    std::vector<Npp32s> src1(total), src2(total);
    TestDataGenerator::generateRandom(src1, (Npp32s)10, (Npp32s)1000, 12345);
    TestDataGenerator::generateRandom(src2, (Npp32s)1, (Npp32s)100, 54321);
    NppImageMemory<Npp32s> d_src1(p.width * ch, p.height);
    NppImageMemory<Npp32s> d_src2(p.width * ch, p.height);
    d_src1.copyFromHost(src1);
    d_src2.copyFromHost(src2);

    if (p.in_place) {
      if (p.channels == 1) {
        status = p.use_ctx ? nppiDiv_32s_C1IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                     p.scaleFactor, ctx)
                           : nppiDiv_32s_C1IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                 p.scaleFactor);
      } else if (p.channels == 3) {
        status = p.use_ctx ? nppiDiv_32s_C3IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                     p.scaleFactor, ctx)
                           : nppiDiv_32s_C3IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi,
                                                 p.scaleFactor);
      }
      ASSERT_EQ(status, NPP_NO_ERROR);
      std::vector<Npp32s> result(total);
      d_src2.copyToHost(result);
      for (int i = 0; i < total; i++) {
        int expected = src2[i] / src1[i];
        EXPECT_LE(std::abs(result[i] - expected), 1);
      }
    } else {
      NppImageMemory<Npp32s> d_dst(p.width * ch, p.height);
      if (p.channels == 1) {
        status = p.use_ctx ? nppiDiv_32s_C1RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                    d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                           : nppiDiv_32s_C1RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                d_dst.step(), roi, p.scaleFactor);
      } else if (p.channels == 3) {
        status = p.use_ctx ? nppiDiv_32s_C3RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                                    d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                           : nppiDiv_32s_C3RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                                d_dst.step(), roi, p.scaleFactor);
      }
      ASSERT_EQ(status, NPP_NO_ERROR);
      std::vector<Npp32s> result(total);
      d_dst.copyToHost(result);
      for (int i = 0; i < total; i++) {
        int expected = src2[i] / src1[i];
        EXPECT_LE(std::abs(result[i] - expected), 1);
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(DivIntSfsComplete, DivIntSfsCompleteParamTest,
                         ::testing::Values(
                             // 8u C3/C4/AC4 variants
                             DivIntSfsCompleteParam{32, 32, 3, 0, true, false, "8u", "8u_C3RSfs_Ctx"},
                             DivIntSfsCompleteParam{32, 32, 3, 0, true, true, "8u", "8u_C3IRSfs_Ctx"},
                             DivIntSfsCompleteParam{32, 32, 4, 0, true, false, "8u", "8u_C4RSfs_Ctx"},
                             DivIntSfsCompleteParam{32, 32, 4, 0, false, true, "8u", "8u_C4IRSfs"},
                             DivIntSfsCompleteParam{32, 32, 4, 0, true, true, "8u", "8u_C4IRSfs_Ctx"},
                             DivIntSfsCompleteParam{32, 32, -4, 0, true, false, "8u", "8u_AC4RSfs_Ctx"},
                             DivIntSfsCompleteParam{32, 32, -4, 0, false, true, "8u", "8u_AC4IRSfs"},
                             DivIntSfsCompleteParam{32, 32, -4, 0, true, true, "8u", "8u_AC4IRSfs_Ctx"},
                             // 16u C4/AC4 variants
                             DivIntSfsCompleteParam{32, 32, 4, 0, true, false, "16u", "16u_C4RSfs_Ctx"},
                             DivIntSfsCompleteParam{32, 32, 4, 0, false, true, "16u", "16u_C4IRSfs"},
                             DivIntSfsCompleteParam{32, 32, 4, 0, true, true, "16u", "16u_C4IRSfs_Ctx"},
                             DivIntSfsCompleteParam{32, 32, -4, 0, true, false, "16u", "16u_AC4RSfs_Ctx"},
                             DivIntSfsCompleteParam{32, 32, -4, 0, false, true, "16u", "16u_AC4IRSfs"},
                             DivIntSfsCompleteParam{32, 32, -4, 0, true, true, "16u", "16u_AC4IRSfs_Ctx"},
                             // 16s C4/AC4 variants
                             DivIntSfsCompleteParam{32, 32, 4, 0, true, false, "16s", "16s_C4RSfs_Ctx"},
                             DivIntSfsCompleteParam{32, 32, 4, 0, false, true, "16s", "16s_C4IRSfs"},
                             DivIntSfsCompleteParam{32, 32, 4, 0, true, true, "16s", "16s_C4IRSfs_Ctx"},
                             DivIntSfsCompleteParam{32, 32, -4, 0, true, false, "16s", "16s_AC4RSfs_Ctx"},
                             DivIntSfsCompleteParam{32, 32, -4, 0, false, true, "16s", "16s_AC4IRSfs"},
                             DivIntSfsCompleteParam{32, 32, -4, 0, true, true, "16s", "16s_AC4IRSfs_Ctx"},
                             // 32s C1/C3 variants
                             DivIntSfsCompleteParam{32, 32, 1, 0, false, false, "32s", "32s_C1RSfs"},
                             DivIntSfsCompleteParam{32, 32, 1, 0, true, false, "32s", "32s_C1RSfs_Ctx"},
                             DivIntSfsCompleteParam{32, 32, 1, 0, false, true, "32s", "32s_C1IRSfs"},
                             DivIntSfsCompleteParam{32, 32, 1, 0, true, true, "32s", "32s_C1IRSfs_Ctx"},
                             DivIntSfsCompleteParam{32, 32, 3, 0, false, false, "32s", "32s_C3RSfs"},
                             DivIntSfsCompleteParam{32, 32, 3, 0, true, false, "32s", "32s_C3RSfs_Ctx"},
                             DivIntSfsCompleteParam{32, 32, 3, 0, false, true, "32s", "32s_C3IRSfs"},
                             DivIntSfsCompleteParam{32, 32, 3, 0, true, true, "32s", "32s_C3IRSfs_Ctx"}),
                         [](const ::testing::TestParamInfo<DivIntSfsCompleteParam> &info) { return info.param.name; });

// ============================================================================
// Div 32f AC4 variants
// ============================================================================

struct Div32fAC4Param {
  int width;
  int height;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Div32fAC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<Div32fAC4Param> {};

TEST_P(Div32fAC4ParamTest, Div_32f_AC4) {
  const auto &p = GetParam();
  const int channels = 4;
  const int total = p.width * p.height * channels;

  std::vector<Npp32f> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, 1.0f, 100.0f, 12345);
  TestDataGenerator::generateRandom(src2, 1.0f, 10.0f, 54321);
  for (int i = 0; i < total; i++) {
    if (i % 4 == 3) {
      expected[i] = src2[i];
    } else {
      expected[i] = src2[i] / src1[i];
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
    status = p.use_ctx ? nppiDiv_32f_AC4IR_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, ctx)
                       : nppiDiv_32f_AC4IR(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi);
    ASSERT_EQ(status, NPP_NO_ERROR);
    std::vector<Npp32f> result(total);
    d_src2.copyToHost(result);
    EXPECT_TRUE(ResultValidator::arraysEqual(result, expected, 1e-4f));
  } else {
    NppImageMemory<Npp32f> d_dst(p.width * channels, p.height);
    d_dst.copyFromHost(src2);
    status = p.use_ctx ? nppiDiv_32f_AC4R_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                              d_dst.step(), roi, ctx)
                       : nppiDiv_32f_AC4R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                          d_dst.step(), roi);
    ASSERT_EQ(status, NPP_NO_ERROR);
    std::vector<Npp32f> result(total);
    d_dst.copyToHost(result);
    EXPECT_TRUE(ResultValidator::arraysEqual(result, expected, 1e-4f));
  }
}

INSTANTIATE_TEST_SUITE_P(Div32fAC4, Div32fAC4ParamTest,
                         ::testing::Values(Div32fAC4Param{32, 32, true, false, "AC4R_Ctx"},
                                           Div32fAC4Param{32, 32, false, true, "AC4IR"},
                                           Div32fAC4Param{32, 32, true, true, "AC4IR_Ctx"}),
                         [](const ::testing::TestParamInfo<Div32fAC4Param> &info) { return info.param.name; });

// ============================================================================
// Div_32s_C1R and Div_32s_C1R_Ctx - non-Sfs variants
// ============================================================================

TEST(Div32sC1RTest, Basic) {
  const int width = 32, height = 32;
  const int total = width * height;

  std::vector<Npp32s> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp32s)10, (Npp32s)1000, 12345);
  TestDataGenerator::generateRandom(src2, (Npp32s)1, (Npp32s)100, 54321);
  for (int i = 0; i < total; i++) {
    expected[i] = src2[i] / src1[i];
  }

  NppImageMemory<Npp32s> d_src1(width, height);
  NppImageMemory<Npp32s> d_src2(width, height);
  NppImageMemory<Npp32s> d_dst(width, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {width, height};

  NppStatus status =
      nppiDiv_32s_C1R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32s> result(total);
  d_dst.copyToHost(result);
  for (int i = 0; i < total; i++) {
    EXPECT_LE(std::abs(result[i] - expected[i]), 1) << "mismatch at " << i;
  }
}

TEST(Div32sC1RCtxTest, Basic) {
  const int width = 32, height = 32;
  const int total = width * height;

  std::vector<Npp32s> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp32s)10, (Npp32s)1000, 12345);
  TestDataGenerator::generateRandom(src2, (Npp32s)1, (Npp32s)100, 54321);
  for (int i = 0; i < total; i++) {
    expected[i] = src2[i] / src1[i];
  }

  NppImageMemory<Npp32s> d_src1(width, height);
  NppImageMemory<Npp32s> d_src2(width, height);
  NppImageMemory<Npp32s> d_dst(width, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {width, height};
  NppStreamContext ctx{};

  NppStatus status = nppiDiv_32s_C1R_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(),
                                         d_dst.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32s> result(total);
  d_dst.copyToHost(result);
  for (int i = 0; i < total; i++) {
    EXPECT_LE(std::abs(result[i] - expected[i]), 1) << "mismatch at " << i;
  }
}
