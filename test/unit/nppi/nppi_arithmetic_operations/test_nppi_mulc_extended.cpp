#include "npp_test_base.h"

using namespace npp_functional_test;

// ==================== MulC 8u C3/C4/AC4 with scale ====================
class MulC8uMultiChannelTest : public NppTestBase {};

TEST_F(MulC8uMultiChannelTest, MulC_8u_C3RSfs) {
  const int width = 32, height = 32, channels = 3;
  const int total = width * height * channels;

  std::vector<Npp8u> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp8u)0, (Npp8u)100, 12345);

  Npp8u aConstants[3] = {2, 3, 2};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int ch = i % 3;
    int prod = src[i] * aConstants[ch];
    expected[i] = static_cast<Npp8u>(std::min(255, std::max(0, prod >> scaleFactor)));
  }

  NppImageMemory<Npp8u> d_src(width * channels, height);
  NppImageMemory<Npp8u> d_dst(width * channels, height);
  d_src.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiMulC_8u_C3RSfs(d_src.get(), d_src.step(), aConstants, d_dst.get(), d_dst.step(), roi, scaleFactor),
            NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(MulC8uMultiChannelTest, MulC_8u_C4RSfs) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp8u> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp8u)0, (Npp8u)100, 12345);

  Npp8u aConstants[4] = {2, 3, 2, 1};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int ch = i % 4;
    int prod = src[i] * aConstants[ch];
    expected[i] = static_cast<Npp8u>(std::min(255, std::max(0, prod >> scaleFactor)));
  }

  NppImageMemory<Npp8u> d_src(width * channels, height);
  NppImageMemory<Npp8u> d_dst(width * channels, height);
  d_src.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiMulC_8u_C4RSfs(d_src.get(), d_src.step(), aConstants, d_dst.get(), d_dst.step(), roi, scaleFactor),
            NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(MulC8uMultiChannelTest, MulC_8u_AC4RSfs) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp8u> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp8u)0, (Npp8u)100, 12345);

  Npp8u aConstants[3] = {2, 3, 2};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int ch = i % 4;
    if (ch == 3) {
      expected[i] = src[i];
    } else {
      int prod = src[i] * aConstants[ch];
      expected[i] = static_cast<Npp8u>(std::min(255, std::max(0, prod >> scaleFactor)));
    }
  }

  NppImageMemory<Npp8u> d_src(width * channels, height);
  NppImageMemory<Npp8u> d_dst(width * channels, height);
  d_src.copyFromHost(src);
  d_dst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiMulC_8u_AC4RSfs(d_src.get(), d_src.step(), aConstants, d_dst.get(), d_dst.step(), roi, scaleFactor),
            NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

// ==================== MulC 16u C3/C4/AC4 with scale ====================
class MulC16uMultiChannelTest : public NppTestBase {};

TEST_F(MulC16uMultiChannelTest, MulC_16u_C3RSfs) {
  const int width = 32, height = 32, channels = 3;
  const int total = width * height * channels;

  std::vector<Npp16u> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp16u)0, (Npp16u)1000, 12345);

  Npp16u aConstants[3] = {2, 3, 2};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int ch = i % 3;
    int prod = src[i] * aConstants[ch];
    expected[i] = static_cast<Npp16u>(std::min(65535, std::max(0, prod >> scaleFactor)));
  }

  NppImageMemory<Npp16u> d_src(width * channels, height);
  NppImageMemory<Npp16u> d_dst(width * channels, height);
  d_src.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiMulC_16u_C3RSfs(d_src.get(), d_src.step(), aConstants, d_dst.get(), d_dst.step(), roi, scaleFactor),
            NPP_NO_ERROR);

  std::vector<Npp16u> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(MulC16uMultiChannelTest, MulC_16u_C4RSfs) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp16u> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp16u)0, (Npp16u)1000, 12345);

  Npp16u aConstants[4] = {2, 3, 2, 1};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int ch = i % 4;
    int prod = src[i] * aConstants[ch];
    expected[i] = static_cast<Npp16u>(std::min(65535, std::max(0, prod >> scaleFactor)));
  }

  NppImageMemory<Npp16u> d_src(width * channels, height);
  NppImageMemory<Npp16u> d_dst(width * channels, height);
  d_src.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiMulC_16u_C4RSfs(d_src.get(), d_src.step(), aConstants, d_dst.get(), d_dst.step(), roi, scaleFactor),
            NPP_NO_ERROR);

  std::vector<Npp16u> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(MulC16uMultiChannelTest, MulC_16u_AC4RSfs) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp16u> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp16u)0, (Npp16u)1000, 12345);

  Npp16u aConstants[3] = {2, 3, 2};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int ch = i % 4;
    if (ch == 3) {
      expected[i] = src[i];
    } else {
      int prod = src[i] * aConstants[ch];
      expected[i] = static_cast<Npp16u>(std::min(65535, std::max(0, prod >> scaleFactor)));
    }
  }

  NppImageMemory<Npp16u> d_src(width * channels, height);
  NppImageMemory<Npp16u> d_dst(width * channels, height);
  d_src.copyFromHost(src);
  d_dst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiMulC_16u_AC4RSfs(d_src.get(), d_src.step(), aConstants, d_dst.get(), d_dst.step(), roi, scaleFactor),
            NPP_NO_ERROR);

  std::vector<Npp16u> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

// ==================== MulC 16s C3/C4/AC4 with scale ====================
class MulC16sMultiChannelTest : public NppTestBase {};

TEST_F(MulC16sMultiChannelTest, MulC_16s_C3RSfs) {
  const int width = 32, height = 32, channels = 3;
  const int total = width * height * channels;

  std::vector<Npp16s> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp16s)-500, (Npp16s)500, 12345);

  Npp16s aConstants[3] = {2, -3, 2};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int ch = i % 3;
    int prod = src[i] * aConstants[ch];
    expected[i] = static_cast<Npp16s>(std::min(32767, std::max(-32768, prod >> scaleFactor)));
  }

  NppImageMemory<Npp16s> d_src(width * channels, height);
  NppImageMemory<Npp16s> d_dst(width * channels, height);
  d_src.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiMulC_16s_C3RSfs(d_src.get(), d_src.step(), aConstants, d_dst.get(), d_dst.step(), roi, scaleFactor),
            NPP_NO_ERROR);

  std::vector<Npp16s> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(MulC16sMultiChannelTest, MulC_16s_C4RSfs) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp16s> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp16s)-500, (Npp16s)500, 12345);

  Npp16s aConstants[4] = {2, -3, 2, -1};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int ch = i % 4;
    int prod = src[i] * aConstants[ch];
    expected[i] = static_cast<Npp16s>(std::min(32767, std::max(-32768, prod >> scaleFactor)));
  }

  NppImageMemory<Npp16s> d_src(width * channels, height);
  NppImageMemory<Npp16s> d_dst(width * channels, height);
  d_src.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiMulC_16s_C4RSfs(d_src.get(), d_src.step(), aConstants, d_dst.get(), d_dst.step(), roi, scaleFactor),
            NPP_NO_ERROR);

  std::vector<Npp16s> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(MulC16sMultiChannelTest, MulC_16s_AC4RSfs) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp16s> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp16s)-500, (Npp16s)500, 12345);

  Npp16s aConstants[3] = {2, -3, 2};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int ch = i % 4;
    if (ch == 3) {
      expected[i] = src[i];
    } else {
      int prod = src[i] * aConstants[ch];
      expected[i] = static_cast<Npp16s>(std::min(32767, std::max(-32768, prod >> scaleFactor)));
    }
  }

  NppImageMemory<Npp16s> d_src(width * channels, height);
  NppImageMemory<Npp16s> d_dst(width * channels, height);
  d_src.copyFromHost(src);
  d_dst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiMulC_16s_AC4RSfs(d_src.get(), d_src.step(), aConstants, d_dst.get(), d_dst.step(), roi, scaleFactor),
            NPP_NO_ERROR);

  std::vector<Npp16s> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

// ==================== MulC 32f C3/C4/AC4 ====================
class MulC32fMultiChannelTest : public NppTestBase {};

TEST_F(MulC32fMultiChannelTest, MulC_32f_C3R) {
  const int width = 32, height = 32, channels = 3;
  const int total = width * height * channels;

  std::vector<Npp32f> src(total), expected(total);
  TestDataGenerator::generateRandom(src, -100.0f, 100.0f, 12345);

  Npp32f aConstants[3] = {1.5f, -2.5f, 3.5f};
  for (int i = 0; i < total; i++) {
    int ch = i % 3;
    expected[i] = src[i] * aConstants[ch];
  }

  NppImageMemory<Npp32f> d_src(width * channels, height);
  NppImageMemory<Npp32f> d_dst(width * channels, height);
  d_src.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiMulC_32f_C3R(d_src.get(), d_src.step(), aConstants, d_dst.get(), d_dst.step(), roi), NPP_NO_ERROR);

  std::vector<Npp32f> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected, 1e-4f));
}

TEST_F(MulC32fMultiChannelTest, MulC_32f_C4R) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp32f> src(total), expected(total);
  TestDataGenerator::generateRandom(src, -100.0f, 100.0f, 12345);

  Npp32f aConstants[4] = {1.5f, -2.5f, 3.5f, -4.5f};
  for (int i = 0; i < total; i++) {
    int ch = i % 4;
    expected[i] = src[i] * aConstants[ch];
  }

  NppImageMemory<Npp32f> d_src(width * channels, height);
  NppImageMemory<Npp32f> d_dst(width * channels, height);
  d_src.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiMulC_32f_C4R(d_src.get(), d_src.step(), aConstants, d_dst.get(), d_dst.step(), roi), NPP_NO_ERROR);

  std::vector<Npp32f> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected, 1e-4f));
}

TEST_F(MulC32fMultiChannelTest, MulC_32f_AC4R) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp32f> src(total), expected(total);
  TestDataGenerator::generateRandom(src, -100.0f, 100.0f, 12345);

  Npp32f aConstants[3] = {1.5f, -2.5f, 3.5f};
  for (int i = 0; i < total; i++) {
    int ch = i % 4;
    if (ch == 3) {
      expected[i] = src[i];
    } else {
      expected[i] = src[i] * aConstants[ch];
    }
  }

  NppImageMemory<Npp32f> d_src(width * channels, height);
  NppImageMemory<Npp32f> d_dst(width * channels, height);
  d_src.copyFromHost(src);
  d_dst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiMulC_32f_AC4R(d_src.get(), d_src.step(), aConstants, d_dst.get(), d_dst.step(), roi), NPP_NO_ERROR);

  std::vector<Npp32f> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected, 1e-4f));
}

// ==================== MulC 32s ====================
class MulC32sTest : public NppTestBase {};

TEST_F(MulC32sTest, MulC_32s_C1RSfs) {
  const int width = 32, height = 32;
  const int total = width * height;

  std::vector<Npp32s> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp32s)-10000, (Npp32s)10000, 12345);

  Npp32s nConstant = 5;
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int64_t prod = (int64_t)src[i] * nConstant;
    expected[i] = static_cast<Npp32s>(prod >> scaleFactor);
  }

  NppImageMemory<Npp32s> d_src(width, height);
  NppImageMemory<Npp32s> d_dst(width, height);
  d_src.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiMulC_32s_C1RSfs(d_src.get(), d_src.step(), nConstant, d_dst.get(), d_dst.step(), roi, scaleFactor),
            NPP_NO_ERROR);

  std::vector<Npp32s> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(MulC32sTest, MulC_32s_C3RSfs) {
  const int width = 32, height = 32, channels = 3;
  const int total = width * height * channels;

  std::vector<Npp32s> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp32s)-10000, (Npp32s)10000, 12345);

  Npp32s aConstants[3] = {2, -3, 2};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int ch = i % 3;
    int64_t prod = (int64_t)src[i] * aConstants[ch];
    expected[i] = static_cast<Npp32s>(prod >> scaleFactor);
  }

  NppImageMemory<Npp32s> d_src(width * channels, height);
  NppImageMemory<Npp32s> d_dst(width * channels, height);
  d_src.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiMulC_32s_C3RSfs(d_src.get(), d_src.step(), aConstants, d_dst.get(), d_dst.step(), roi, scaleFactor),
            NPP_NO_ERROR);

  std::vector<Npp32s> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

// ==================== MulC inplace ====================
class MulC8uInplaceTest : public NppTestBase {};

TEST_F(MulC8uInplaceTest, MulC_8u_C1IRSfs) {
  const int width = 32, height = 32;
  const int total = width * height;

  std::vector<Npp8u> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp8u)0, (Npp8u)100, 12345);

  Npp8u nConstant = 2;
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int prod = src[i] * nConstant;
    expected[i] = static_cast<Npp8u>(std::min(255, std::max(0, prod >> scaleFactor)));
  }

  NppImageMemory<Npp8u> d_srcdst(width, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiMulC_8u_C1IRSfs(nConstant, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(MulC8uInplaceTest, MulC_8u_C3IRSfs) {
  const int width = 32, height = 32, channels = 3;
  const int total = width * height * channels;

  std::vector<Npp8u> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp8u)0, (Npp8u)100, 12345);

  Npp8u aConstants[3] = {2, 3, 2};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int ch = i % 3;
    int prod = src[i] * aConstants[ch];
    expected[i] = static_cast<Npp8u>(std::min(255, std::max(0, prod >> scaleFactor)));
  }

  NppImageMemory<Npp8u> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiMulC_8u_C3IRSfs(aConstants, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(MulC8uInplaceTest, MulC_8u_C4IRSfs) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp8u> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp8u)0, (Npp8u)100, 12345);

  Npp8u aConstants[4] = {2, 3, 2, 1};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int ch = i % 4;
    int prod = src[i] * aConstants[ch];
    expected[i] = static_cast<Npp8u>(std::min(255, std::max(0, prod >> scaleFactor)));
  }

  NppImageMemory<Npp8u> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiMulC_8u_C4IRSfs(aConstants, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(MulC8uInplaceTest, MulC_8u_AC4IRSfs) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp8u> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp8u)0, (Npp8u)100, 12345);

  Npp8u aConstants[3] = {2, 3, 2};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int ch = i % 4;
    if (ch == 3) {
      expected[i] = src[i];
    } else {
      int prod = src[i] * aConstants[ch];
      expected[i] = static_cast<Npp8u>(std::min(255, std::max(0, prod >> scaleFactor)));
    }
  }

  NppImageMemory<Npp8u> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiMulC_8u_AC4IRSfs(aConstants, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

// ==================== MulC 32f inplace ====================
class MulC32fInplaceTest : public NppTestBase {};

TEST_F(MulC32fInplaceTest, MulC_32f_C1IR) {
  const int width = 32, height = 32;
  const int total = width * height;

  std::vector<Npp32f> src(total), expected(total);
  TestDataGenerator::generateRandom(src, -100.0f, 100.0f, 12345);

  Npp32f nConstant = 2.5f;
  for (int i = 0; i < total; i++) {
    expected[i] = src[i] * nConstant;
  }

  NppImageMemory<Npp32f> d_srcdst(width, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiMulC_32f_C1IR(nConstant, d_srcdst.get(), d_srcdst.step(), roi), NPP_NO_ERROR);

  std::vector<Npp32f> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected, 1e-4f));
}

TEST_F(MulC32fInplaceTest, MulC_32f_C3IR) {
  const int width = 32, height = 32, channels = 3;
  const int total = width * height * channels;

  std::vector<Npp32f> src(total), expected(total);
  TestDataGenerator::generateRandom(src, -100.0f, 100.0f, 12345);

  Npp32f aConstants[3] = {1.5f, -2.5f, 3.5f};
  for (int i = 0; i < total; i++) {
    int ch = i % 3;
    expected[i] = src[i] * aConstants[ch];
  }

  NppImageMemory<Npp32f> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiMulC_32f_C3IR(aConstants, d_srcdst.get(), d_srcdst.step(), roi), NPP_NO_ERROR);

  std::vector<Npp32f> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected, 1e-4f));
}

TEST_F(MulC32fInplaceTest, MulC_32f_C4IR) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp32f> src(total), expected(total);
  TestDataGenerator::generateRandom(src, -100.0f, 100.0f, 12345);

  Npp32f aConstants[4] = {1.5f, -2.5f, 3.5f, -4.5f};
  for (int i = 0; i < total; i++) {
    int ch = i % 4;
    expected[i] = src[i] * aConstants[ch];
  }

  NppImageMemory<Npp32f> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiMulC_32f_C4IR(aConstants, d_srcdst.get(), d_srcdst.step(), roi), NPP_NO_ERROR);

  std::vector<Npp32f> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected, 1e-4f));
}

TEST_F(MulC32fInplaceTest, MulC_32f_AC4IR) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp32f> src(total), expected(total);
  TestDataGenerator::generateRandom(src, -100.0f, 100.0f, 12345);

  Npp32f aConstants[3] = {1.5f, -2.5f, 3.5f};
  for (int i = 0; i < total; i++) {
    int ch = i % 4;
    if (ch == 3) {
      expected[i] = src[i];
    } else {
      expected[i] = src[i] * aConstants[ch];
    }
  }

  NppImageMemory<Npp32f> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiMulC_32f_AC4IR(aConstants, d_srcdst.get(), d_srcdst.step(), roi), NPP_NO_ERROR);

  std::vector<Npp32f> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected, 1e-4f));
}

// ==================== MulScale ====================
class MulScale8uTest : public NppTestBase {};

TEST_F(MulScale8uTest, MulScale_8u_C1R) {
  const int width = 32, height = 32;
  const int total = width * height;

  std::vector<Npp8u> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp8u)0, (Npp8u)255, 12345);
  TestDataGenerator::generateRandom(src2, (Npp8u)0, (Npp8u)255, 54321);

  for (int i = 0; i < total; i++) {
    int prod = src1[i] * src2[i];
    expected[i] = static_cast<Npp8u>((prod + 128) / 255);
  }

  NppImageMemory<Npp8u> d_src1(width, height);
  NppImageMemory<Npp8u> d_src2(width, height);
  NppImageMemory<Npp8u> d_dst(width, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(
      nppiMulScale_8u_C1R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi),
      NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_dst.copyToHost(result);
  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_LE(std::abs(static_cast<int>(result[i]) - static_cast<int>(expected[i])), 1)
        << "MulScale mismatch at index " << i;
  }
}

TEST_F(MulScale8uTest, MulScale_8u_C3R) {
  const int width = 32, height = 32, channels = 3;
  const int total = width * height * channels;

  std::vector<Npp8u> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp8u)0, (Npp8u)255, 12345);
  TestDataGenerator::generateRandom(src2, (Npp8u)0, (Npp8u)255, 54321);

  for (int i = 0; i < total; i++) {
    int prod = src1[i] * src2[i];
    expected[i] = static_cast<Npp8u>((prod + 128) / 255);
  }

  NppImageMemory<Npp8u> d_src1(width * channels, height);
  NppImageMemory<Npp8u> d_src2(width * channels, height);
  NppImageMemory<Npp8u> d_dst(width * channels, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(
      nppiMulScale_8u_C3R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi),
      NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_dst.copyToHost(result);
  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_LE(std::abs(static_cast<int>(result[i]) - static_cast<int>(expected[i])), 1)
        << "MulScale mismatch at index " << i;
  }
}

TEST_F(MulScale8uTest, MulScale_8u_C4R) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp8u> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp8u)0, (Npp8u)255, 12345);
  TestDataGenerator::generateRandom(src2, (Npp8u)0, (Npp8u)255, 54321);

  for (int i = 0; i < total; i++) {
    int prod = src1[i] * src2[i];
    expected[i] = static_cast<Npp8u>((prod + 128) / 255);
  }

  NppImageMemory<Npp8u> d_src1(width * channels, height);
  NppImageMemory<Npp8u> d_src2(width * channels, height);
  NppImageMemory<Npp8u> d_dst(width * channels, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(
      nppiMulScale_8u_C4R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi),
      NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_dst.copyToHost(result);
  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_LE(std::abs(static_cast<int>(result[i]) - static_cast<int>(expected[i])), 1)
        << "MulScale mismatch at index " << i;
  }
}

TEST_F(MulScale8uTest, MulScale_8u_AC4R) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp8u> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp8u)0, (Npp8u)255, 12345);
  TestDataGenerator::generateRandom(src2, (Npp8u)0, (Npp8u)255, 54321);

  for (int i = 0; i < total; i++) {
    if (i % 4 == 3) {
      expected[i] = src2[i];
    } else {
      int prod = src1[i] * src2[i];
      expected[i] = static_cast<Npp8u>((prod + 128) / 255);
    }
  }

  NppImageMemory<Npp8u> d_src1(width * channels, height);
  NppImageMemory<Npp8u> d_src2(width * channels, height);
  NppImageMemory<Npp8u> d_dst(width * channels, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);
  d_dst.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(
      nppiMulScale_8u_AC4R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi),
      NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_dst.copyToHost(result);
  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_LE(std::abs(static_cast<int>(result[i]) - static_cast<int>(expected[i])), 1)
        << "MulScale mismatch at index " << i;
  }
}
