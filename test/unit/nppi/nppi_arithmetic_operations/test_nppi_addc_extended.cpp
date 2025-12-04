#include "npp_test_base.h"

using namespace npp_functional_test;

// ==================== AddC 8u C3/C4/AC4 with scale ====================
class AddC8uMultiChannelTest : public NppTestBase {};

TEST_F(AddC8uMultiChannelTest, AddC_8u_C3RSfs) {
  const int width = 32, height = 32, channels = 3;
  const int total = width * height * channels;

  std::vector<Npp8u> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp8u)0, (Npp8u)200, 12345);

  Npp8u aConstants[3] = {10, 20, 30};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int ch = i % 3;
    int sum = src[i] + aConstants[ch];
    expected[i] = static_cast<Npp8u>(std::min(255, std::max(0, sum >> scaleFactor)));
  }

  NppImageMemory<Npp8u> d_src(width * channels, height);
  NppImageMemory<Npp8u> d_dst(width * channels, height);
  d_src.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAddC_8u_C3RSfs(d_src.get(), d_src.step(), aConstants,
                               d_dst.get(), d_dst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(AddC8uMultiChannelTest, AddC_8u_C4RSfs) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp8u> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp8u)0, (Npp8u)200, 12345);

  Npp8u aConstants[4] = {10, 20, 30, 40};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int ch = i % 4;
    int sum = src[i] + aConstants[ch];
    expected[i] = static_cast<Npp8u>(std::min(255, std::max(0, sum >> scaleFactor)));
  }

  NppImageMemory<Npp8u> d_src(width * channels, height);
  NppImageMemory<Npp8u> d_dst(width * channels, height);
  d_src.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAddC_8u_C4RSfs(d_src.get(), d_src.step(), aConstants,
                               d_dst.get(), d_dst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(AddC8uMultiChannelTest, AddC_8u_AC4RSfs) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp8u> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp8u)0, (Npp8u)200, 12345);

  Npp8u aConstants[3] = {10, 20, 30};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int ch = i % 4;
    if (ch == 3) {
      expected[i] = src[i];
    } else {
      int sum = src[i] + aConstants[ch];
      expected[i] = static_cast<Npp8u>(std::min(255, std::max(0, sum >> scaleFactor)));
    }
  }

  NppImageMemory<Npp8u> d_src(width * channels, height);
  NppImageMemory<Npp8u> d_dst(width * channels, height);
  d_src.copyFromHost(src);
  d_dst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAddC_8u_AC4RSfs(d_src.get(), d_src.step(), aConstants,
                                d_dst.get(), d_dst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

// ==================== AddC 16u C3/C4/AC4 with scale ====================
class AddC16uMultiChannelTest : public NppTestBase {};

TEST_F(AddC16uMultiChannelTest, AddC_16u_C3RSfs) {
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

  NppImageMemory<Npp16u> d_src(width * channels, height);
  NppImageMemory<Npp16u> d_dst(width * channels, height);
  d_src.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAddC_16u_C3RSfs(d_src.get(), d_src.step(), aConstants,
                                d_dst.get(), d_dst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp16u> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(AddC16uMultiChannelTest, AddC_16u_C4RSfs) {
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

  NppImageMemory<Npp16u> d_src(width * channels, height);
  NppImageMemory<Npp16u> d_dst(width * channels, height);
  d_src.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAddC_16u_C4RSfs(d_src.get(), d_src.step(), aConstants,
                                d_dst.get(), d_dst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp16u> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(AddC16uMultiChannelTest, AddC_16u_AC4RSfs) {
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

  NppImageMemory<Npp16u> d_src(width * channels, height);
  NppImageMemory<Npp16u> d_dst(width * channels, height);
  d_src.copyFromHost(src);
  d_dst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAddC_16u_AC4RSfs(d_src.get(), d_src.step(), aConstants,
                                 d_dst.get(), d_dst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp16u> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

// ==================== AddC 16s C3/C4/AC4 with scale ====================
class AddC16sMultiChannelTest : public NppTestBase {};

TEST_F(AddC16sMultiChannelTest, AddC_16s_C3RSfs) {
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

  NppImageMemory<Npp16s> d_src(width * channels, height);
  NppImageMemory<Npp16s> d_dst(width * channels, height);
  d_src.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAddC_16s_C3RSfs(d_src.get(), d_src.step(), aConstants,
                                d_dst.get(), d_dst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp16s> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(AddC16sMultiChannelTest, AddC_16s_C4RSfs) {
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

  NppImageMemory<Npp16s> d_src(width * channels, height);
  NppImageMemory<Npp16s> d_dst(width * channels, height);
  d_src.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAddC_16s_C4RSfs(d_src.get(), d_src.step(), aConstants,
                                d_dst.get(), d_dst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp16s> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(AddC16sMultiChannelTest, AddC_16s_AC4RSfs) {
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

  NppImageMemory<Npp16s> d_src(width * channels, height);
  NppImageMemory<Npp16s> d_dst(width * channels, height);
  d_src.copyFromHost(src);
  d_dst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAddC_16s_AC4RSfs(d_src.get(), d_src.step(), aConstants,
                                 d_dst.get(), d_dst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp16s> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

// ==================== AddC 32f C3/C4/AC4 ====================
class AddC32fMultiChannelTest : public NppTestBase {};

TEST_F(AddC32fMultiChannelTest, AddC_32f_C3R) {
  const int width = 32, height = 32, channels = 3;
  const int total = width * height * channels;

  std::vector<Npp32f> src(total), expected(total);
  TestDataGenerator::generateRandom(src, -100.0f, 100.0f, 12345);

  Npp32f aConstants[3] = {1.5f, -2.5f, 3.5f};
  for (int i = 0; i < total; i++) {
    int ch = i % 3;
    expected[i] = src[i] + aConstants[ch];
  }

  NppImageMemory<Npp32f> d_src(width * channels, height);
  NppImageMemory<Npp32f> d_dst(width * channels, height);
  d_src.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAddC_32f_C3R(d_src.get(), d_src.step(), aConstants,
                             d_dst.get(), d_dst.step(), roi), NPP_NO_ERROR);

  std::vector<Npp32f> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected, 1e-5f));
}

TEST_F(AddC32fMultiChannelTest, AddC_32f_C4R) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp32f> src(total), expected(total);
  TestDataGenerator::generateRandom(src, -100.0f, 100.0f, 12345);

  Npp32f aConstants[4] = {1.5f, -2.5f, 3.5f, -4.5f};
  for (int i = 0; i < total; i++) {
    int ch = i % 4;
    expected[i] = src[i] + aConstants[ch];
  }

  NppImageMemory<Npp32f> d_src(width * channels, height);
  NppImageMemory<Npp32f> d_dst(width * channels, height);
  d_src.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAddC_32f_C4R(d_src.get(), d_src.step(), aConstants,
                             d_dst.get(), d_dst.step(), roi), NPP_NO_ERROR);

  std::vector<Npp32f> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected, 1e-5f));
}

TEST_F(AddC32fMultiChannelTest, AddC_32f_AC4R) {
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
      expected[i] = src[i] + aConstants[ch];
    }
  }

  NppImageMemory<Npp32f> d_src(width * channels, height);
  NppImageMemory<Npp32f> d_dst(width * channels, height);
  d_src.copyFromHost(src);
  d_dst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAddC_32f_AC4R(d_src.get(), d_src.step(), aConstants,
                              d_dst.get(), d_dst.step(), roi), NPP_NO_ERROR);

  std::vector<Npp32f> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected, 1e-5f));
}

// ==================== AddC 32s ====================
class AddC32sTest : public NppTestBase {};

TEST_F(AddC32sTest, AddC_32s_C1RSfs) {
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

  NppImageMemory<Npp32s> d_src(width, height);
  NppImageMemory<Npp32s> d_dst(width, height);
  d_src.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAddC_32s_C1RSfs(d_src.get(), d_src.step(), nConstant,
                                d_dst.get(), d_dst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp32s> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(AddC32sTest, AddC_32s_C3RSfs) {
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

  NppImageMemory<Npp32s> d_src(width * channels, height);
  NppImageMemory<Npp32s> d_dst(width * channels, height);
  d_src.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAddC_32s_C3RSfs(d_src.get(), d_src.step(), aConstants,
                                d_dst.get(), d_dst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp32s> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

// ==================== AddC inplace ====================
class AddC8uInplaceTest : public NppTestBase {};

TEST_F(AddC8uInplaceTest, AddC_8u_C1IRSfs) {
  const int width = 32, height = 32;
  const int total = width * height;

  std::vector<Npp8u> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp8u)0, (Npp8u)200, 12345);

  Npp8u nConstant = 25;
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int sum = src[i] + nConstant;
    expected[i] = static_cast<Npp8u>(std::min(255, std::max(0, sum >> scaleFactor)));
  }

  NppImageMemory<Npp8u> d_srcdst(width, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAddC_8u_C1IRSfs(nConstant, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(AddC8uInplaceTest, AddC_8u_C3IRSfs) {
  const int width = 32, height = 32, channels = 3;
  const int total = width * height * channels;

  std::vector<Npp8u> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp8u)0, (Npp8u)200, 12345);

  Npp8u aConstants[3] = {10, 20, 30};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int ch = i % 3;
    int sum = src[i] + aConstants[ch];
    expected[i] = static_cast<Npp8u>(std::min(255, std::max(0, sum >> scaleFactor)));
  }

  NppImageMemory<Npp8u> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAddC_8u_C3IRSfs(aConstants, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(AddC8uInplaceTest, AddC_8u_C4IRSfs) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp8u> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp8u)0, (Npp8u)200, 12345);

  Npp8u aConstants[4] = {10, 20, 30, 40};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int ch = i % 4;
    int sum = src[i] + aConstants[ch];
    expected[i] = static_cast<Npp8u>(std::min(255, std::max(0, sum >> scaleFactor)));
  }

  NppImageMemory<Npp8u> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAddC_8u_C4IRSfs(aConstants, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(AddC8uInplaceTest, AddC_8u_AC4IRSfs) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp8u> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp8u)0, (Npp8u)200, 12345);

  Npp8u aConstants[3] = {10, 20, 30};
  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int ch = i % 4;
    if (ch == 3) {
      expected[i] = src[i];
    } else {
      int sum = src[i] + aConstants[ch];
      expected[i] = static_cast<Npp8u>(std::min(255, std::max(0, sum >> scaleFactor)));
    }
  }

  NppImageMemory<Npp8u> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAddC_8u_AC4IRSfs(aConstants, d_srcdst.get(), d_srcdst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

// ==================== AddC 32f inplace ====================
class AddC32fInplaceTest : public NppTestBase {};

TEST_F(AddC32fInplaceTest, AddC_32f_C1IR) {
  const int width = 32, height = 32;
  const int total = width * height;

  std::vector<Npp32f> src(total), expected(total);
  TestDataGenerator::generateRandom(src, -100.0f, 100.0f, 12345);

  Npp32f nConstant = 5.5f;
  for (int i = 0; i < total; i++) {
    expected[i] = src[i] + nConstant;
  }

  NppImageMemory<Npp32f> d_srcdst(width, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAddC_32f_C1IR(nConstant, d_srcdst.get(), d_srcdst.step(), roi), NPP_NO_ERROR);

  std::vector<Npp32f> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected, 1e-5f));
}

TEST_F(AddC32fInplaceTest, AddC_32f_C3IR) {
  const int width = 32, height = 32, channels = 3;
  const int total = width * height * channels;

  std::vector<Npp32f> src(total), expected(total);
  TestDataGenerator::generateRandom(src, -100.0f, 100.0f, 12345);

  Npp32f aConstants[3] = {1.5f, -2.5f, 3.5f};
  for (int i = 0; i < total; i++) {
    int ch = i % 3;
    expected[i] = src[i] + aConstants[ch];
  }

  NppImageMemory<Npp32f> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAddC_32f_C3IR(aConstants, d_srcdst.get(), d_srcdst.step(), roi), NPP_NO_ERROR);

  std::vector<Npp32f> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected, 1e-5f));
}

TEST_F(AddC32fInplaceTest, AddC_32f_C4IR) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp32f> src(total), expected(total);
  TestDataGenerator::generateRandom(src, -100.0f, 100.0f, 12345);

  Npp32f aConstants[4] = {1.5f, -2.5f, 3.5f, -4.5f};
  for (int i = 0; i < total; i++) {
    int ch = i % 4;
    expected[i] = src[i] + aConstants[ch];
  }

  NppImageMemory<Npp32f> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAddC_32f_C4IR(aConstants, d_srcdst.get(), d_srcdst.step(), roi), NPP_NO_ERROR);

  std::vector<Npp32f> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected, 1e-5f));
}

TEST_F(AddC32fInplaceTest, AddC_32f_AC4IR) {
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
      expected[i] = src[i] + aConstants[ch];
    }
  }

  NppImageMemory<Npp32f> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAddC_32f_AC4IR(aConstants, d_srcdst.get(), d_srcdst.step(), roi), NPP_NO_ERROR);

  std::vector<Npp32f> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected, 1e-5f));
}
