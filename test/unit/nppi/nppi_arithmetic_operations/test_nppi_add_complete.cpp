#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ============================================================================
// Add 16f (half precision float) - C1/C3/C4 channels
// ============================================================================

struct Add16fCompleteParam {
  int width;
  int height;
  int channels;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Add16fCompleteParamTest : public NppTestBase, public ::testing::WithParamInterface<Add16fCompleteParam> {};

TEST_P(Add16fCompleteParamTest, Add_16f_Complete) {
  const auto& p = GetParam();
  const int total = p.width * p.height * p.channels;

  std::vector<Npp32f> src1_f(total), src2_f(total);
  TestDataGenerator::generateRandom(src1_f, -10.0f, 10.0f, 12345);
  TestDataGenerator::generateRandom(src2_f, -10.0f, 10.0f, 54321);

  std::vector<Npp16f> src1(total), src2(total), expected(total);
  for (int i = 0; i < total; i++) {
    src1[i] = float_to_npp16f_host(src1_f[i]);
    src2[i] = float_to_npp16f_host(src2_f[i]);
    expected[i] = float_to_npp16f_host(src1_f[i] + src2_f[i]);
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
      status = p.use_ctx ? nppiAdd_16f_C1IR_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, ctx)
                         : nppiAdd_16f_C1IR(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi);
    } else if (p.channels == 3) {
      status = p.use_ctx ? nppiAdd_16f_C3IR_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, ctx)
                         : nppiAdd_16f_C3IR(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi);
    } else if (p.channels == 4) {
      status = p.use_ctx ? nppiAdd_16f_C4IR_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, ctx)
                         : nppiAdd_16f_C4IR(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);
    std::vector<Npp16f> result(total);
    d_src2.copyToHost(result);
    EXPECT_TRUE(ResultValidator::arraysEqual16f(result, expected, 0.1f));
  } else {
    NppImageMemory<Npp16f> d_dst(p.width * p.channels, p.height);
    if (p.channels == 1) {
      status = p.use_ctx ? nppiAdd_16f_C1R_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, ctx)
                         : nppiAdd_16f_C1R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi);
    } else if (p.channels == 3) {
      status = p.use_ctx ? nppiAdd_16f_C3R_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, ctx)
                         : nppiAdd_16f_C3R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi);
    } else if (p.channels == 4) {
      status = p.use_ctx ? nppiAdd_16f_C4R_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, ctx)
                         : nppiAdd_16f_C4R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi);
    }
    ASSERT_EQ(status, NPP_NO_ERROR);
    std::vector<Npp16f> result(total);
    d_dst.copyToHost(result);
    EXPECT_TRUE(ResultValidator::arraysEqual16f(result, expected, 0.1f));
  }
}

INSTANTIATE_TEST_SUITE_P(Add16fComplete, Add16fCompleteParamTest, ::testing::Values(
  Add16fCompleteParam{32, 32, 1, false, false, "C1R"},
  Add16fCompleteParam{32, 32, 1, true, false, "C1R_Ctx"},
  Add16fCompleteParam{32, 32, 1, false, true, "C1IR"},
  Add16fCompleteParam{32, 32, 1, true, true, "C1IR_Ctx"},
  Add16fCompleteParam{32, 32, 3, false, false, "C3R"},
  Add16fCompleteParam{32, 32, 3, true, false, "C3R_Ctx"},
  Add16fCompleteParam{32, 32, 3, false, true, "C3IR"},
  Add16fCompleteParam{32, 32, 3, true, true, "C3IR_Ctx"},
  Add16fCompleteParam{32, 32, 4, false, false, "C4R"},
  Add16fCompleteParam{32, 32, 4, true, false, "C4R_Ctx"},
  Add16fCompleteParam{32, 32, 4, false, true, "C4IR"},
  Add16fCompleteParam{32, 32, 4, true, true, "C4IR_Ctx"}
), [](const ::testing::TestParamInfo<Add16fCompleteParam>& info) { return info.param.name; });

// ============================================================================
// Add 32fc (complex float) - C1/C3/C4/AC4 channels
// ============================================================================

struct Add32fcParam {
  int width;
  int height;
  int channels;  // 1, 3, 4, or -4 for AC4
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Add32fcParamTest : public NppTestBase, public ::testing::WithParamInterface<Add32fcParam> {};

TEST_P(Add32fcParamTest, Add_32fc) {
  const auto& p = GetParam();
  const int ch = (p.channels < 0) ? 4 : p.channels;
  const int total = p.width * p.height * ch;

  std::vector<Npp32fc> src1(total), src2(total), expected(total);
  for (int i = 0; i < total; i++) {
    src1[i].re = static_cast<Npp32f>((i % 100) - 50);
    src1[i].im = static_cast<Npp32f>((i % 50) - 25);
    src2[i].re = static_cast<Npp32f>(((i + 30) % 100) - 50);
    src2[i].im = static_cast<Npp32f>(((i + 15) % 50) - 25);
    if (p.channels == -4 && (i % 4 == 3)) {
      expected[i] = src2[i];  // AC4: alpha unchanged
    } else {
      expected[i].re = src1[i].re + src2[i].re;
      expected[i].im = src1[i].im + src2[i].im;
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
      status = p.use_ctx ? nppiAdd_32fc_C1IR_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, ctx)
                         : nppiAdd_32fc_C1IR(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi);
    } else if (p.channels == 3) {
      status = p.use_ctx ? nppiAdd_32fc_C3IR_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, ctx)
                         : nppiAdd_32fc_C3IR(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi);
    } else if (p.channels == 4) {
      status = p.use_ctx ? nppiAdd_32fc_C4IR_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, ctx)
                         : nppiAdd_32fc_C4IR(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi);
    } else if (p.channels == -4) {
      status = p.use_ctx ? nppiAdd_32fc_AC4IR_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, ctx)
                         : nppiAdd_32fc_AC4IR(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi);
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
    if (p.channels == -4) d_dst.copyFromHost(src2);
    if (p.channels == 1) {
      status = p.use_ctx ? nppiAdd_32fc_C1R_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, ctx)
                         : nppiAdd_32fc_C1R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi);
    } else if (p.channels == 3) {
      status = p.use_ctx ? nppiAdd_32fc_C3R_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, ctx)
                         : nppiAdd_32fc_C3R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi);
    } else if (p.channels == 4) {
      status = p.use_ctx ? nppiAdd_32fc_C4R_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, ctx)
                         : nppiAdd_32fc_C4R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi);
    } else if (p.channels == -4) {
      status = p.use_ctx ? nppiAdd_32fc_AC4R_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, ctx)
                         : nppiAdd_32fc_AC4R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi);
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

INSTANTIATE_TEST_SUITE_P(Add32fc, Add32fcParamTest, ::testing::Values(
  Add32fcParam{32, 32, 1, false, false, "C1R"},
  Add32fcParam{32, 32, 1, true, false, "C1R_Ctx"},
  Add32fcParam{32, 32, 1, false, true, "C1IR"},
  Add32fcParam{32, 32, 1, true, true, "C1IR_Ctx"},
  Add32fcParam{32, 32, 3, false, false, "C3R"},
  Add32fcParam{32, 32, 3, true, false, "C3R_Ctx"},
  Add32fcParam{32, 32, 3, false, true, "C3IR"},
  Add32fcParam{32, 32, 3, true, true, "C3IR_Ctx"},
  Add32fcParam{32, 32, 4, false, false, "C4R"},
  Add32fcParam{32, 32, 4, true, false, "C4R_Ctx"},
  Add32fcParam{32, 32, 4, false, true, "C4IR"},
  Add32fcParam{32, 32, 4, true, true, "C4IR_Ctx"},
  Add32fcParam{32, 32, -4, false, false, "AC4R"},
  Add32fcParam{32, 32, -4, true, false, "AC4R_Ctx"},
  Add32fcParam{32, 32, -4, false, true, "AC4IR"},
  Add32fcParam{32, 32, -4, true, true, "AC4IR_Ctx"}
), [](const ::testing::TestParamInfo<Add32fcParam>& info) { return info.param.name; });

// ============================================================================
// Add 16sc/32sc (complex integer with scale) - C1/C3/AC4 channels
// ============================================================================

struct Add16scParam {
  int width;
  int height;
  int channels;  // 1, 3, or -4 for AC4
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Add16scParamTest : public NppTestBase, public ::testing::WithParamInterface<Add16scParam> {};

TEST_P(Add16scParamTest, Add_16sc) {
  const auto& p = GetParam();
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
      int re = src1[i].re + src2[i].re;
      int im = src1[i].im + src2[i].im;
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
      status = p.use_ctx ? nppiAdd_16sc_C1IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor, ctx)
                         : nppiAdd_16sc_C1IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
    } else if (p.channels == 3) {
      status = p.use_ctx ? nppiAdd_16sc_C3IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor, ctx)
                         : nppiAdd_16sc_C3IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
    } else if (p.channels == -4) {
      status = p.use_ctx ? nppiAdd_16sc_AC4IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor, ctx)
                         : nppiAdd_16sc_AC4IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
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
    if (p.channels == -4) d_dst.copyFromHost(src2);
    if (p.channels == 1) {
      status = p.use_ctx ? nppiAdd_16sc_C1RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                         : nppiAdd_16sc_C1RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, p.scaleFactor);
    } else if (p.channels == 3) {
      status = p.use_ctx ? nppiAdd_16sc_C3RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                         : nppiAdd_16sc_C3RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, p.scaleFactor);
    } else if (p.channels == -4) {
      status = p.use_ctx ? nppiAdd_16sc_AC4RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                         : nppiAdd_16sc_AC4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, p.scaleFactor);
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

INSTANTIATE_TEST_SUITE_P(Add16sc, Add16scParamTest, ::testing::Values(
  Add16scParam{32, 32, 1, 0, false, false, "C1RSfs"},
  Add16scParam{32, 32, 1, 0, true, false, "C1RSfs_Ctx"},
  Add16scParam{32, 32, 1, 0, false, true, "C1IRSfs"},
  Add16scParam{32, 32, 1, 0, true, true, "C1IRSfs_Ctx"},
  Add16scParam{32, 32, 3, 0, false, false, "C3RSfs"},
  Add16scParam{32, 32, 3, 0, true, false, "C3RSfs_Ctx"},
  Add16scParam{32, 32, 3, 0, false, true, "C3IRSfs"},
  Add16scParam{32, 32, 3, 0, true, true, "C3IRSfs_Ctx"},
  Add16scParam{32, 32, -4, 0, false, false, "AC4RSfs"},
  Add16scParam{32, 32, -4, 0, true, false, "AC4RSfs_Ctx"},
  Add16scParam{32, 32, -4, 0, false, true, "AC4IRSfs"},
  Add16scParam{32, 32, -4, 0, true, true, "AC4IRSfs_Ctx"}
), [](const ::testing::TestParamInfo<Add16scParam>& info) { return info.param.name; });

// ============================================================================
// Add 32sc (complex 32-bit integer with scale)
// ============================================================================

struct Add32scParam {
  int width;
  int height;
  int channels;
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Add32scParamTest : public NppTestBase, public ::testing::WithParamInterface<Add32scParam> {};

TEST_P(Add32scParamTest, Add_32sc) {
  const auto& p = GetParam();
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
      int64_t re = static_cast<int64_t>(src1[i].re) + src2[i].re;
      int64_t im = static_cast<int64_t>(src1[i].im) + src2[i].im;
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
      status = p.use_ctx ? nppiAdd_32sc_C1IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor, ctx)
                         : nppiAdd_32sc_C1IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
    } else if (p.channels == 3) {
      status = p.use_ctx ? nppiAdd_32sc_C3IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor, ctx)
                         : nppiAdd_32sc_C3IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
    } else if (p.channels == -4) {
      status = p.use_ctx ? nppiAdd_32sc_AC4IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor, ctx)
                         : nppiAdd_32sc_AC4IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
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
    if (p.channels == -4) d_dst.copyFromHost(src2);
    if (p.channels == 1) {
      status = p.use_ctx ? nppiAdd_32sc_C1RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                         : nppiAdd_32sc_C1RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, p.scaleFactor);
    } else if (p.channels == 3) {
      status = p.use_ctx ? nppiAdd_32sc_C3RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                         : nppiAdd_32sc_C3RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, p.scaleFactor);
    } else if (p.channels == -4) {
      status = p.use_ctx ? nppiAdd_32sc_AC4RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                         : nppiAdd_32sc_AC4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, p.scaleFactor);
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

INSTANTIATE_TEST_SUITE_P(Add32sc, Add32scParamTest, ::testing::Values(
  Add32scParam{32, 32, 1, 0, false, false, "C1RSfs"},
  Add32scParam{32, 32, 1, 0, true, false, "C1RSfs_Ctx"},
  Add32scParam{32, 32, 1, 0, false, true, "C1IRSfs"},
  Add32scParam{32, 32, 1, 0, true, true, "C1IRSfs_Ctx"},
  Add32scParam{32, 32, 3, 0, false, false, "C3RSfs"},
  Add32scParam{32, 32, 3, 0, true, false, "C3RSfs_Ctx"},
  Add32scParam{32, 32, 3, 0, false, true, "C3IRSfs"},
  Add32scParam{32, 32, 3, 0, true, true, "C3IRSfs_Ctx"},
  Add32scParam{32, 32, -4, 0, false, false, "AC4RSfs"},
  Add32scParam{32, 32, -4, 0, true, false, "AC4RSfs_Ctx"},
  Add32scParam{32, 32, -4, 0, false, true, "AC4IRSfs"},
  Add32scParam{32, 32, -4, 0, true, true, "AC4IRSfs_Ctx"}
), [](const ::testing::TestParamInfo<Add32scParam>& info) { return info.param.name; });

// ============================================================================
// Add remaining integer variants (8u/16u/16s/32s) with Ctx and C4/AC4
// ============================================================================

struct AddIntSfsParam {
  int width;
  int height;
  int channels;  // 3, 4, or -4 for AC4
  int scaleFactor;
  bool use_ctx;
  bool in_place;
  std::string dtype;  // "8u", "16u", "16s", "32s"
  std::string name;
};

class AddIntSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<AddIntSfsParam> {};

TEST_P(AddIntSfsParamTest, Add_IntSfs) {
  const auto& p = GetParam();
  const int ch = (p.channels < 0) ? 4 : p.channels;
  const int total = p.width * p.height * ch;

  NppiSize roi = {p.width, p.height};
  NppStreamContext ctx{};
  NppStatus status = NPP_ERROR;

  if (p.dtype == "8u") {
    std::vector<Npp8u> src1(total), src2(total), expected(total);
    TestDataGenerator::generateRandom(src1, (Npp8u)0, (Npp8u)100, 12345);
    TestDataGenerator::generateRandom(src2, (Npp8u)0, (Npp8u)100, 54321);
    for (int i = 0; i < total; i++) {
      if (p.channels == -4 && (i % 4 == 3)) {
        expected[i] = src2[i];
      } else {
        expected[i] = expect::add_sfs<Npp8u>(src1[i], src2[i], p.scaleFactor);
      }
    }
    NppImageMemory<Npp8u> d_src1(p.width * ch, p.height);
    NppImageMemory<Npp8u> d_src2(p.width * ch, p.height);
    d_src1.copyFromHost(src1);
    d_src2.copyFromHost(src2);

    if (p.in_place) {
      if (p.channels == 3) {
        status = p.use_ctx ? nppiAdd_8u_C3IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor, ctx)
                           : nppiAdd_8u_C3IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
      } else if (p.channels == 4) {
        status = p.use_ctx ? nppiAdd_8u_C4IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor, ctx)
                           : nppiAdd_8u_C4IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
      } else if (p.channels == -4) {
        status = p.use_ctx ? nppiAdd_8u_AC4IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor, ctx)
                           : nppiAdd_8u_AC4IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
      }
      ASSERT_EQ(status, NPP_NO_ERROR);
      std::vector<Npp8u> result(total);
      d_src2.copyToHost(result);
      EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
    } else {
      NppImageMemory<Npp8u> d_dst(p.width * ch, p.height);
      if (p.channels == -4) d_dst.copyFromHost(src2);
      if (p.channels == 3) {
        status = p.use_ctx ? nppiAdd_8u_C3RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                           : nppiAdd_8u_C3RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, p.scaleFactor);
      } else if (p.channels == 4) {
        status = p.use_ctx ? nppiAdd_8u_C4RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                           : nppiAdd_8u_C4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, p.scaleFactor);
      } else if (p.channels == -4) {
        status = p.use_ctx ? nppiAdd_8u_AC4RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                           : nppiAdd_8u_AC4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, p.scaleFactor);
      }
      ASSERT_EQ(status, NPP_NO_ERROR);
      std::vector<Npp8u> result(total);
      d_dst.copyToHost(result);
      EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
    }
  } else if (p.dtype == "16u") {
    std::vector<Npp16u> src1(total), src2(total), expected(total);
    TestDataGenerator::generateRandom(src1, (Npp16u)0, (Npp16u)1000, 12345);
    TestDataGenerator::generateRandom(src2, (Npp16u)0, (Npp16u)1000, 54321);
    for (int i = 0; i < total; i++) {
      if (p.channels == -4 && (i % 4 == 3)) {
        expected[i] = src2[i];
      } else {
        expected[i] = expect::add_sfs<Npp16u>(src1[i], src2[i], p.scaleFactor);
      }
    }
    NppImageMemory<Npp16u> d_src1(p.width * ch, p.height);
    NppImageMemory<Npp16u> d_src2(p.width * ch, p.height);
    d_src1.copyFromHost(src1);
    d_src2.copyFromHost(src2);

    if (p.in_place) {
      if (p.channels == 4) {
        status = p.use_ctx ? nppiAdd_16u_C4IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor, ctx)
                           : nppiAdd_16u_C4IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
      } else if (p.channels == -4) {
        status = p.use_ctx ? nppiAdd_16u_AC4IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor, ctx)
                           : nppiAdd_16u_AC4IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
      }
      ASSERT_EQ(status, NPP_NO_ERROR);
      std::vector<Npp16u> result(total);
      d_src2.copyToHost(result);
      EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
    } else {
      NppImageMemory<Npp16u> d_dst(p.width * ch, p.height);
      if (p.channels == -4) d_dst.copyFromHost(src2);
      if (p.channels == 4) {
        status = p.use_ctx ? nppiAdd_16u_C4RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                           : nppiAdd_16u_C4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, p.scaleFactor);
      } else if (p.channels == -4) {
        status = p.use_ctx ? nppiAdd_16u_AC4RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                           : nppiAdd_16u_AC4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, p.scaleFactor);
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
        expected[i] = expect::add_sfs<Npp16s>(src1[i], src2[i], p.scaleFactor);
      }
    }
    NppImageMemory<Npp16s> d_src1(p.width * ch, p.height);
    NppImageMemory<Npp16s> d_src2(p.width * ch, p.height);
    d_src1.copyFromHost(src1);
    d_src2.copyFromHost(src2);

    if (p.in_place) {
      if (p.channels == 4) {
        status = p.use_ctx ? nppiAdd_16s_C4IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor, ctx)
                           : nppiAdd_16s_C4IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
      } else if (p.channels == -4) {
        status = p.use_ctx ? nppiAdd_16s_AC4IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor, ctx)
                           : nppiAdd_16s_AC4IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
      }
      ASSERT_EQ(status, NPP_NO_ERROR);
      std::vector<Npp16s> result(total);
      d_src2.copyToHost(result);
      EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
    } else {
      NppImageMemory<Npp16s> d_dst(p.width * ch, p.height);
      if (p.channels == -4) d_dst.copyFromHost(src2);
      if (p.channels == 4) {
        status = p.use_ctx ? nppiAdd_16s_C4RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                           : nppiAdd_16s_C4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, p.scaleFactor);
      } else if (p.channels == -4) {
        status = p.use_ctx ? nppiAdd_16s_AC4RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                           : nppiAdd_16s_AC4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, p.scaleFactor);
      }
      ASSERT_EQ(status, NPP_NO_ERROR);
      std::vector<Npp16s> result(total);
      d_dst.copyToHost(result);
      EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
    }
  } else if (p.dtype == "32s") {
    std::vector<Npp32s> src1(total), src2(total), expected(total);
    TestDataGenerator::generateRandom(src1, (Npp32s)-10000, (Npp32s)10000, 12345);
    TestDataGenerator::generateRandom(src2, (Npp32s)-10000, (Npp32s)10000, 54321);
    for (int i = 0; i < total; i++) {
      int64_t sum = static_cast<int64_t>(src1[i]) + src2[i];
      expected[i] = static_cast<Npp32s>(sum);
    }
    NppImageMemory<Npp32s> d_src1(p.width * ch, p.height);
    NppImageMemory<Npp32s> d_src2(p.width * ch, p.height);
    d_src1.copyFromHost(src1);
    d_src2.copyFromHost(src2);

    if (p.in_place) {
      if (p.channels == 1) {
        status = p.use_ctx ? nppiAdd_32s_C1IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor, ctx)
                           : nppiAdd_32s_C1IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
      } else if (p.channels == 3) {
        status = p.use_ctx ? nppiAdd_32s_C3IRSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor, ctx)
                           : nppiAdd_32s_C3IRSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, p.scaleFactor);
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
        status = p.use_ctx ? nppiAdd_32s_C1RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                           : nppiAdd_32s_C1RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, p.scaleFactor);
      } else if (p.channels == 3) {
        status = p.use_ctx ? nppiAdd_32s_C3RSfs_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, p.scaleFactor, ctx)
                           : nppiAdd_32s_C3RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, p.scaleFactor);
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

// Add float AC4 variants
struct Add32fAC4Param {
  int width;
  int height;
  bool use_ctx;
  bool in_place;
  std::string name;
};

class Add32fAC4ParamTest : public NppTestBase, public ::testing::WithParamInterface<Add32fAC4Param> {};

TEST_P(Add32fAC4ParamTest, Add_32f_AC4) {
  const auto& p = GetParam();
  const int channels = 4;
  const int total = p.width * p.height * channels;

  std::vector<Npp32f> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, -50.0f, 50.0f, 12345);
  TestDataGenerator::generateRandom(src2, -50.0f, 50.0f, 54321);
  for (int i = 0; i < total; i++) {
    if (i % 4 == 3) {
      expected[i] = src2[i];
    } else {
      expected[i] = src1[i] + src2[i];
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
    status = p.use_ctx ? nppiAdd_32f_AC4IR_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi, ctx)
                       : nppiAdd_32f_AC4IR(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), roi);
    ASSERT_EQ(status, NPP_NO_ERROR);
    std::vector<Npp32f> result(total);
    d_src2.copyToHost(result);
    EXPECT_TRUE(ResultValidator::arraysEqual(result, expected, 1e-5f));
  } else {
    NppImageMemory<Npp32f> d_dst(p.width * channels, p.height);
    d_dst.copyFromHost(src2);
    status = p.use_ctx ? nppiAdd_32f_AC4R_Ctx(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi, ctx)
                       : nppiAdd_32f_AC4R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi);
    ASSERT_EQ(status, NPP_NO_ERROR);
    std::vector<Npp32f> result(total);
    d_dst.copyToHost(result);
    EXPECT_TRUE(ResultValidator::arraysEqual(result, expected, 1e-5f));
  }
}

INSTANTIATE_TEST_SUITE_P(Add32fAC4, Add32fAC4ParamTest, ::testing::Values(
  Add32fAC4Param{32, 32, false, false, "AC4R"},
  Add32fAC4Param{32, 32, true, false, "AC4R_Ctx"},
  Add32fAC4Param{32, 32, false, true, "AC4IR"},
  Add32fAC4Param{32, 32, true, true, "AC4IR_Ctx"}
), [](const ::testing::TestParamInfo<Add32fAC4Param>& info) { return info.param.name; });

INSTANTIATE_TEST_SUITE_P(AddIntSfs, AddIntSfsParamTest, ::testing::Values(
  // 8u variants
  AddIntSfsParam{32, 32, 3, 0, true, false, "8u", "8u_C3RSfs_Ctx"},
  AddIntSfsParam{32, 32, 3, 0, true, true, "8u", "8u_C3IRSfs_Ctx"},
  AddIntSfsParam{32, 32, 4, 0, false, false, "8u", "8u_C4RSfs"},
  AddIntSfsParam{32, 32, 4, 0, true, false, "8u", "8u_C4RSfs_Ctx"},
  AddIntSfsParam{32, 32, 4, 0, false, true, "8u", "8u_C4IRSfs"},
  AddIntSfsParam{32, 32, 4, 0, true, true, "8u", "8u_C4IRSfs_Ctx"},
  AddIntSfsParam{32, 32, -4, 0, true, false, "8u", "8u_AC4RSfs_Ctx"},
  AddIntSfsParam{32, 32, -4, 0, false, true, "8u", "8u_AC4IRSfs"},
  AddIntSfsParam{32, 32, -4, 0, true, true, "8u", "8u_AC4IRSfs_Ctx"},
  // 16u variants
  AddIntSfsParam{32, 32, 4, 0, false, false, "16u", "16u_C4RSfs"},
  AddIntSfsParam{32, 32, 4, 0, true, false, "16u", "16u_C4RSfs_Ctx"},
  AddIntSfsParam{32, 32, 4, 0, false, true, "16u", "16u_C4IRSfs"},
  AddIntSfsParam{32, 32, 4, 0, true, true, "16u", "16u_C4IRSfs_Ctx"},
  AddIntSfsParam{32, 32, -4, 0, true, false, "16u", "16u_AC4RSfs_Ctx"},
  AddIntSfsParam{32, 32, -4, 0, false, true, "16u", "16u_AC4IRSfs"},
  AddIntSfsParam{32, 32, -4, 0, true, true, "16u", "16u_AC4IRSfs_Ctx"},
  // 16s variants
  AddIntSfsParam{32, 32, 4, 0, false, false, "16s", "16s_C4RSfs"},
  AddIntSfsParam{32, 32, 4, 0, true, false, "16s", "16s_C4RSfs_Ctx"},
  AddIntSfsParam{32, 32, 4, 0, false, true, "16s", "16s_C4IRSfs"},
  AddIntSfsParam{32, 32, 4, 0, true, true, "16s", "16s_C4IRSfs_Ctx"},
  AddIntSfsParam{32, 32, -4, 0, true, false, "16s", "16s_AC4RSfs_Ctx"},
  AddIntSfsParam{32, 32, -4, 0, false, true, "16s", "16s_AC4IRSfs"},
  AddIntSfsParam{32, 32, -4, 0, true, true, "16s", "16s_AC4IRSfs_Ctx"},
  // 32s variants
  AddIntSfsParam{32, 32, 1, 0, true, false, "32s", "32s_C1R_Ctx"},
  AddIntSfsParam{32, 32, 1, 0, true, false, "32s", "32s_C1RSfs_Ctx"},
  AddIntSfsParam{32, 32, 1, 0, false, true, "32s", "32s_C1IRSfs"},
  AddIntSfsParam{32, 32, 1, 0, true, true, "32s", "32s_C1IRSfs_Ctx"},
  AddIntSfsParam{32, 32, 3, 0, true, false, "32s", "32s_C3RSfs_Ctx"},
  AddIntSfsParam{32, 32, 3, 0, false, true, "32s", "32s_C3IRSfs"},
  AddIntSfsParam{32, 32, 3, 0, true, true, "32s", "32s_C3IRSfs_Ctx"}
), [](const ::testing::TestParamInfo<AddIntSfsParam>& info) { return info.param.name; });

// ============================================================================
// Add_32s_C1R_Ctx - the only remaining Add variant
// ============================================================================

TEST(Add32sC1RCtxTest, Basic) {
  const int width = 32, height = 32;
  const int total = width * height;

  std::vector<Npp32s> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp32s)-10000, (Npp32s)10000, 12345);
  TestDataGenerator::generateRandom(src2, (Npp32s)-10000, (Npp32s)10000, 54321);
  for (int i = 0; i < total; i++) {
    expected[i] = src1[i] + src2[i];
  }

  NppImageMemory<Npp32s> d_src1(width, height);
  NppImageMemory<Npp32s> d_src2(width, height);
  NppImageMemory<Npp32s> d_dst(width, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {width, height};
  NppStreamContext ctx{};

  NppStatus status = nppiAdd_32s_C1R_Ctx(
      d_src1.get(), d_src1.step(),
      d_src2.get(), d_src2.step(),
      d_dst.get(), d_dst.step(),
      roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32s> result(total);
  d_dst.copyToHost(result);
  for (int i = 0; i < total; i++) {
    EXPECT_LE(std::abs(result[i] - expected[i]), 1) << "mismatch at " << i;
  }
}
