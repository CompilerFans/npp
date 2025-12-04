#include "npp_test_base.h"

using namespace npp_functional_test;

// ==================== Mul 8u C3/C4/AC4 with scale ====================
class Mul8uMultiChannelTest : public NppTestBase {};

TEST_F(Mul8uMultiChannelTest, Mul_8u_C3RSfs) {
  const int width = 32, height = 32, channels = 3;
  const int total = width * height * channels;

  std::vector<Npp8u> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp8u)0, (Npp8u)15, 12345);
  TestDataGenerator::generateRandom(src2, (Npp8u)0, (Npp8u)15, 54321);

  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int prod = src1[i] * src2[i];
    expected[i] = static_cast<Npp8u>(std::min(255, std::max(0, prod >> scaleFactor)));
  }

  NppImageMemory<Npp8u> d_src1(width * channels, height);
  NppImageMemory<Npp8u> d_src2(width * channels, height);
  NppImageMemory<Npp8u> d_dst(width * channels, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiMul_8u_C3RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                              d_dst.get(), d_dst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_dst.copyToHost(result);
  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_LE(std::abs(static_cast<int>(result[i]) - static_cast<int>(expected[i])), 1)
      << "Mul mismatch at index " << i;
  }
}

TEST_F(Mul8uMultiChannelTest, Mul_8u_C4RSfs) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp8u> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp8u)0, (Npp8u)15, 12345);
  TestDataGenerator::generateRandom(src2, (Npp8u)0, (Npp8u)15, 54321);

  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    int prod = src1[i] * src2[i];
    expected[i] = static_cast<Npp8u>(std::min(255, std::max(0, prod >> scaleFactor)));
  }

  NppImageMemory<Npp8u> d_src1(width * channels, height);
  NppImageMemory<Npp8u> d_src2(width * channels, height);
  NppImageMemory<Npp8u> d_dst(width * channels, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiMul_8u_C4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                              d_dst.get(), d_dst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_dst.copyToHost(result);
  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_LE(std::abs(static_cast<int>(result[i]) - static_cast<int>(expected[i])), 1)
      << "Mul mismatch at index " << i;
  }
}

TEST_F(Mul8uMultiChannelTest, Mul_8u_AC4RSfs) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp8u> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp8u)0, (Npp8u)15, 12345);
  TestDataGenerator::generateRandom(src2, (Npp8u)0, (Npp8u)15, 54321);

  int scaleFactor = 0;
  for (int i = 0; i < total; i++) {
    if (i % 4 == 3) {
      expected[i] = src2[i];
    } else {
      int prod = src1[i] * src2[i];
      expected[i] = static_cast<Npp8u>(std::min(255, std::max(0, prod >> scaleFactor)));
    }
  }

  NppImageMemory<Npp8u> d_src1(width * channels, height);
  NppImageMemory<Npp8u> d_src2(width * channels, height);
  NppImageMemory<Npp8u> d_dst(width * channels, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);
  d_dst.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiMul_8u_AC4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                               d_dst.get(), d_dst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_dst.copyToHost(result);
  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_LE(std::abs(static_cast<int>(result[i]) - static_cast<int>(expected[i])), 1)
      << "Mul mismatch at index " << i;
  }
}

// ==================== Mul 16u C4/AC4 with scale ====================
class Mul16uMultiChannelTest : public NppTestBase {};

TEST_F(Mul16uMultiChannelTest, Mul_16u_C4RSfs) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp16u> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp16u)0, (Npp16u)255, 12345);
  TestDataGenerator::generateRandom(src2, (Npp16u)0, (Npp16u)255, 54321);

  int scaleFactor = 8;
  for (int i = 0; i < total; i++) {
    int prod = src1[i] * src2[i];
    expected[i] = static_cast<Npp16u>(std::min(65535, std::max(0, (prod + (1 << (scaleFactor - 1))) >> scaleFactor)));
  }

  NppImageMemory<Npp16u> d_src1(width * channels, height);
  NppImageMemory<Npp16u> d_src2(width * channels, height);
  NppImageMemory<Npp16u> d_dst(width * channels, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiMul_16u_C4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                               d_dst.get(), d_dst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp16u> result(total);
  d_dst.copyToHost(result);
  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_LE(std::abs(static_cast<int>(result[i]) - static_cast<int>(expected[i])), 1)
      << "Mul mismatch at index " << i;
  }
}

TEST_F(Mul16uMultiChannelTest, Mul_16u_AC4RSfs) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp16u> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp16u)0, (Npp16u)255, 12345);
  TestDataGenerator::generateRandom(src2, (Npp16u)0, (Npp16u)255, 54321);

  int scaleFactor = 8;
  for (int i = 0; i < total; i++) {
    if (i % 4 == 3) {
      expected[i] = src2[i];
    } else {
      int prod = src1[i] * src2[i];
      expected[i] = static_cast<Npp16u>(std::min(65535, std::max(0, (prod + (1 << (scaleFactor - 1))) >> scaleFactor)));
    }
  }

  NppImageMemory<Npp16u> d_src1(width * channels, height);
  NppImageMemory<Npp16u> d_src2(width * channels, height);
  NppImageMemory<Npp16u> d_dst(width * channels, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);
  d_dst.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiMul_16u_AC4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                d_dst.get(), d_dst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp16u> result(total);
  d_dst.copyToHost(result);
  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_LE(std::abs(static_cast<int>(result[i]) - static_cast<int>(expected[i])), 1)
      << "Mul mismatch at index " << i;
  }
}

// ==================== Mul 16s C4/AC4 with scale ====================
class Mul16sMultiChannelTest : public NppTestBase {};

TEST_F(Mul16sMultiChannelTest, Mul_16s_C4RSfs) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp16s> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp16s)-100, (Npp16s)100, 12345);
  TestDataGenerator::generateRandom(src2, (Npp16s)-100, (Npp16s)100, 54321);

  int scaleFactor = 7;
  for (int i = 0; i < total; i++) {
    int prod = src1[i] * src2[i];
    expected[i] = static_cast<Npp16s>(std::min(32767, std::max(-32768, (prod + (1 << (scaleFactor - 1))) >> scaleFactor)));
  }

  NppImageMemory<Npp16s> d_src1(width * channels, height);
  NppImageMemory<Npp16s> d_src2(width * channels, height);
  NppImageMemory<Npp16s> d_dst(width * channels, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiMul_16s_C4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                               d_dst.get(), d_dst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp16s> result(total);
  d_dst.copyToHost(result);
  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_LE(std::abs(static_cast<int>(result[i]) - static_cast<int>(expected[i])), 1)
      << "Mul mismatch at index " << i;
  }
}

TEST_F(Mul16sMultiChannelTest, Mul_16s_AC4RSfs) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp16s> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp16s)-100, (Npp16s)100, 12345);
  TestDataGenerator::generateRandom(src2, (Npp16s)-100, (Npp16s)100, 54321);

  int scaleFactor = 7;
  for (int i = 0; i < total; i++) {
    if (i % 4 == 3) {
      expected[i] = src2[i];
    } else {
      int prod = src1[i] * src2[i];
      expected[i] = static_cast<Npp16s>(std::min(32767, std::max(-32768, (prod + (1 << (scaleFactor - 1))) >> scaleFactor)));
    }
  }

  NppImageMemory<Npp16s> d_src1(width * channels, height);
  NppImageMemory<Npp16s> d_src2(width * channels, height);
  NppImageMemory<Npp16s> d_dst(width * channels, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);
  d_dst.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiMul_16s_AC4RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                                d_dst.get(), d_dst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp16s> result(total);
  d_dst.copyToHost(result);
  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_LE(std::abs(static_cast<int>(result[i]) - static_cast<int>(expected[i])), 1)
      << "Mul mismatch at index " << i;
  }
}

// ==================== Mul 32f C3/C4/AC4 ====================
class Mul32fMultiChannelTest : public NppTestBase {};

TEST_F(Mul32fMultiChannelTest, Mul_32f_C3R) {
  const int width = 32, height = 32, channels = 3;
  const int total = width * height * channels;

  std::vector<Npp32f> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, -10.0f, 10.0f, 12345);
  TestDataGenerator::generateRandom(src2, -10.0f, 10.0f, 54321);

  for (int i = 0; i < total; i++) {
    expected[i] = src1[i] * src2[i];
  }

  NppImageMemory<Npp32f> d_src1(width * channels, height);
  NppImageMemory<Npp32f> d_src2(width * channels, height);
  NppImageMemory<Npp32f> d_dst(width * channels, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiMul_32f_C3R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                            d_dst.get(), d_dst.step(), roi), NPP_NO_ERROR);

  std::vector<Npp32f> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected, 1e-4f));
}

TEST_F(Mul32fMultiChannelTest, Mul_32f_C4R) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp32f> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, -10.0f, 10.0f, 12345);
  TestDataGenerator::generateRandom(src2, -10.0f, 10.0f, 54321);

  for (int i = 0; i < total; i++) {
    expected[i] = src1[i] * src2[i];
  }

  NppImageMemory<Npp32f> d_src1(width * channels, height);
  NppImageMemory<Npp32f> d_src2(width * channels, height);
  NppImageMemory<Npp32f> d_dst(width * channels, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiMul_32f_C4R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                            d_dst.get(), d_dst.step(), roi), NPP_NO_ERROR);

  std::vector<Npp32f> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected, 1e-4f));
}

TEST_F(Mul32fMultiChannelTest, Mul_32f_AC4R) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

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

  NppImageMemory<Npp32f> d_src1(width * channels, height);
  NppImageMemory<Npp32f> d_src2(width * channels, height);
  NppImageMemory<Npp32f> d_dst(width * channels, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);
  d_dst.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiMul_32f_AC4R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                             d_dst.get(), d_dst.step(), roi), NPP_NO_ERROR);

  std::vector<Npp32f> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected, 1e-4f));
}

// ==================== Mul 32s ====================
class Mul32sTest : public NppTestBase {};

TEST_F(Mul32sTest, Mul_32s_C1RSfs) {
  const int width = 32, height = 32;
  const int total = width * height;

  std::vector<Npp32s> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp32s)-1000, (Npp32s)1000, 12345);
  TestDataGenerator::generateRandom(src2, (Npp32s)-1000, (Npp32s)1000, 54321);

  int scaleFactor = 10;
  for (int i = 0; i < total; i++) {
    int64_t prod = (int64_t)src1[i] * src2[i];
    expected[i] = static_cast<Npp32s>((prod + (1LL << (scaleFactor - 1))) >> scaleFactor);
  }

  NppImageMemory<Npp32s> d_src1(width, height);
  NppImageMemory<Npp32s> d_src2(width, height);
  NppImageMemory<Npp32s> d_dst(width, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiMul_32s_C1RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                               d_dst.get(), d_dst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp32s> result(total);
  d_dst.copyToHost(result);
  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_LE(std::abs(static_cast<int>(result[i]) - static_cast<int>(expected[i])), 1)
      << "Mul mismatch at index " << i;
  }
}

TEST_F(Mul32sTest, Mul_32s_C3RSfs) {
  const int width = 32, height = 32, channels = 3;
  const int total = width * height * channels;

  std::vector<Npp32s> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp32s)-1000, (Npp32s)1000, 12345);
  TestDataGenerator::generateRandom(src2, (Npp32s)-1000, (Npp32s)1000, 54321);

  int scaleFactor = 10;
  for (int i = 0; i < total; i++) {
    int64_t prod = (int64_t)src1[i] * src2[i];
    expected[i] = static_cast<Npp32s>((prod + (1LL << (scaleFactor - 1))) >> scaleFactor);
  }

  NppImageMemory<Npp32s> d_src1(width * channels, height);
  NppImageMemory<Npp32s> d_src2(width * channels, height);
  NppImageMemory<Npp32s> d_dst(width * channels, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiMul_32s_C3RSfs(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(),
                               d_dst.get(), d_dst.step(), roi, scaleFactor), NPP_NO_ERROR);

  std::vector<Npp32s> result(total);
  d_dst.copyToHost(result);
  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_LE(std::abs(static_cast<int>(result[i]) - static_cast<int>(expected[i])), 1)
      << "Mul mismatch at index " << i;
  }
}
