#include "npp_test_base.h"

using namespace npp_functional_test;

// ==================== And 8u C3/C4/AC4 ====================
class And8uMultiChannelTest : public NppTestBase {};

TEST_F(And8uMultiChannelTest, And_8u_C3R) {
  const int width = 32, height = 32, channels = 3;
  const int total = width * height * channels;

  std::vector<Npp8u> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp8u)0, (Npp8u)255, 12345);
  TestDataGenerator::generateRandom(src2, (Npp8u)0, (Npp8u)255, 54321);

  for (int i = 0; i < total; i++) {
    expected[i] = src1[i] & src2[i];
  }

  NppImageMemory<Npp8u> d_src1(width * channels, height);
  NppImageMemory<Npp8u> d_src2(width * channels, height);
  NppImageMemory<Npp8u> d_dst(width * channels, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAnd_8u_C3R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi),
            NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(And8uMultiChannelTest, And_8u_C4R) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp8u> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp8u)0, (Npp8u)255, 12345);
  TestDataGenerator::generateRandom(src2, (Npp8u)0, (Npp8u)255, 54321);

  for (int i = 0; i < total; i++) {
    expected[i] = src1[i] & src2[i];
  }

  NppImageMemory<Npp8u> d_src1(width * channels, height);
  NppImageMemory<Npp8u> d_src2(width * channels, height);
  NppImageMemory<Npp8u> d_dst(width * channels, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAnd_8u_C4R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi),
            NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(And8uMultiChannelTest, And_8u_AC4R) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp8u> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp8u)0, (Npp8u)255, 12345);
  TestDataGenerator::generateRandom(src2, (Npp8u)0, (Npp8u)255, 54321);

  for (int i = 0; i < total; i++) {
    if (i % 4 == 3) {
      expected[i] = src2[i];
    } else {
      expected[i] = src1[i] & src2[i];
    }
  }

  NppImageMemory<Npp8u> d_src1(width * channels, height);
  NppImageMemory<Npp8u> d_src2(width * channels, height);
  NppImageMemory<Npp8u> d_dst(width * channels, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);
  d_dst.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAnd_8u_AC4R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi),
            NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

// ==================== And 16u C3/C4/AC4 ====================
class And16uMultiChannelTest : public NppTestBase {};

TEST_F(And16uMultiChannelTest, And_16u_C3R) {
  const int width = 32, height = 32, channels = 3;
  const int total = width * height * channels;

  std::vector<Npp16u> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp16u)0, (Npp16u)65535, 12345);
  TestDataGenerator::generateRandom(src2, (Npp16u)0, (Npp16u)65535, 54321);

  for (int i = 0; i < total; i++) {
    expected[i] = src1[i] & src2[i];
  }

  NppImageMemory<Npp16u> d_src1(width * channels, height);
  NppImageMemory<Npp16u> d_src2(width * channels, height);
  NppImageMemory<Npp16u> d_dst(width * channels, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAnd_16u_C3R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi),
            NPP_NO_ERROR);

  std::vector<Npp16u> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(And16uMultiChannelTest, And_16u_C4R) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp16u> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp16u)0, (Npp16u)65535, 12345);
  TestDataGenerator::generateRandom(src2, (Npp16u)0, (Npp16u)65535, 54321);

  for (int i = 0; i < total; i++) {
    expected[i] = src1[i] & src2[i];
  }

  NppImageMemory<Npp16u> d_src1(width * channels, height);
  NppImageMemory<Npp16u> d_src2(width * channels, height);
  NppImageMemory<Npp16u> d_dst(width * channels, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAnd_16u_C4R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi),
            NPP_NO_ERROR);

  std::vector<Npp16u> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(And16uMultiChannelTest, And_16u_AC4R) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp16u> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp16u)0, (Npp16u)65535, 12345);
  TestDataGenerator::generateRandom(src2, (Npp16u)0, (Npp16u)65535, 54321);

  for (int i = 0; i < total; i++) {
    if (i % 4 == 3) {
      expected[i] = src2[i];
    } else {
      expected[i] = src1[i] & src2[i];
    }
  }

  NppImageMemory<Npp16u> d_src1(width * channels, height);
  NppImageMemory<Npp16u> d_src2(width * channels, height);
  NppImageMemory<Npp16u> d_dst(width * channels, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);
  d_dst.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAnd_16u_AC4R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi),
            NPP_NO_ERROR);

  std::vector<Npp16u> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

// ==================== Or 8u C3/C4/AC4 ====================
class Or8uMultiChannelTest : public NppTestBase {};

TEST_F(Or8uMultiChannelTest, Or_8u_C3R) {
  const int width = 32, height = 32, channels = 3;
  const int total = width * height * channels;

  std::vector<Npp8u> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp8u)0, (Npp8u)255, 12345);
  TestDataGenerator::generateRandom(src2, (Npp8u)0, (Npp8u)255, 54321);

  for (int i = 0; i < total; i++) {
    expected[i] = src1[i] | src2[i];
  }

  NppImageMemory<Npp8u> d_src1(width * channels, height);
  NppImageMemory<Npp8u> d_src2(width * channels, height);
  NppImageMemory<Npp8u> d_dst(width * channels, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiOr_8u_C3R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi),
            NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(Or8uMultiChannelTest, Or_8u_C4R) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp8u> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp8u)0, (Npp8u)255, 12345);
  TestDataGenerator::generateRandom(src2, (Npp8u)0, (Npp8u)255, 54321);

  for (int i = 0; i < total; i++) {
    expected[i] = src1[i] | src2[i];
  }

  NppImageMemory<Npp8u> d_src1(width * channels, height);
  NppImageMemory<Npp8u> d_src2(width * channels, height);
  NppImageMemory<Npp8u> d_dst(width * channels, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiOr_8u_C4R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi),
            NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(Or8uMultiChannelTest, Or_8u_AC4R) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp8u> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp8u)0, (Npp8u)255, 12345);
  TestDataGenerator::generateRandom(src2, (Npp8u)0, (Npp8u)255, 54321);

  for (int i = 0; i < total; i++) {
    if (i % 4 == 3) {
      expected[i] = src2[i];
    } else {
      expected[i] = src1[i] | src2[i];
    }
  }

  NppImageMemory<Npp8u> d_src1(width * channels, height);
  NppImageMemory<Npp8u> d_src2(width * channels, height);
  NppImageMemory<Npp8u> d_dst(width * channels, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);
  d_dst.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiOr_8u_AC4R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi),
            NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

// ==================== Or 16u C3/C4/AC4 ====================
class Or16uMultiChannelTest : public NppTestBase {};

TEST_F(Or16uMultiChannelTest, Or_16u_C3R) {
  const int width = 32, height = 32, channels = 3;
  const int total = width * height * channels;

  std::vector<Npp16u> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp16u)0, (Npp16u)65535, 12345);
  TestDataGenerator::generateRandom(src2, (Npp16u)0, (Npp16u)65535, 54321);

  for (int i = 0; i < total; i++) {
    expected[i] = src1[i] | src2[i];
  }

  NppImageMemory<Npp16u> d_src1(width * channels, height);
  NppImageMemory<Npp16u> d_src2(width * channels, height);
  NppImageMemory<Npp16u> d_dst(width * channels, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiOr_16u_C3R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi),
            NPP_NO_ERROR);

  std::vector<Npp16u> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(Or16uMultiChannelTest, Or_16u_C4R) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp16u> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp16u)0, (Npp16u)65535, 12345);
  TestDataGenerator::generateRandom(src2, (Npp16u)0, (Npp16u)65535, 54321);

  for (int i = 0; i < total; i++) {
    expected[i] = src1[i] | src2[i];
  }

  NppImageMemory<Npp16u> d_src1(width * channels, height);
  NppImageMemory<Npp16u> d_src2(width * channels, height);
  NppImageMemory<Npp16u> d_dst(width * channels, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiOr_16u_C4R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi),
            NPP_NO_ERROR);

  std::vector<Npp16u> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(Or16uMultiChannelTest, Or_16u_AC4R) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp16u> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp16u)0, (Npp16u)65535, 12345);
  TestDataGenerator::generateRandom(src2, (Npp16u)0, (Npp16u)65535, 54321);

  for (int i = 0; i < total; i++) {
    if (i % 4 == 3) {
      expected[i] = src2[i];
    } else {
      expected[i] = src1[i] | src2[i];
    }
  }

  NppImageMemory<Npp16u> d_src1(width * channels, height);
  NppImageMemory<Npp16u> d_src2(width * channels, height);
  NppImageMemory<Npp16u> d_dst(width * channels, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);
  d_dst.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiOr_16u_AC4R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi),
            NPP_NO_ERROR);

  std::vector<Npp16u> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

// ==================== Xor 8u C3/C4/AC4 ====================
class Xor8uMultiChannelTest : public NppTestBase {};

TEST_F(Xor8uMultiChannelTest, Xor_8u_C3R) {
  const int width = 32, height = 32, channels = 3;
  const int total = width * height * channels;

  std::vector<Npp8u> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp8u)0, (Npp8u)255, 12345);
  TestDataGenerator::generateRandom(src2, (Npp8u)0, (Npp8u)255, 54321);

  for (int i = 0; i < total; i++) {
    expected[i] = src1[i] ^ src2[i];
  }

  NppImageMemory<Npp8u> d_src1(width * channels, height);
  NppImageMemory<Npp8u> d_src2(width * channels, height);
  NppImageMemory<Npp8u> d_dst(width * channels, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiXor_8u_C3R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi),
            NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(Xor8uMultiChannelTest, Xor_8u_C4R) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp8u> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp8u)0, (Npp8u)255, 12345);
  TestDataGenerator::generateRandom(src2, (Npp8u)0, (Npp8u)255, 54321);

  for (int i = 0; i < total; i++) {
    expected[i] = src1[i] ^ src2[i];
  }

  NppImageMemory<Npp8u> d_src1(width * channels, height);
  NppImageMemory<Npp8u> d_src2(width * channels, height);
  NppImageMemory<Npp8u> d_dst(width * channels, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiXor_8u_C4R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi),
            NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(Xor8uMultiChannelTest, Xor_8u_AC4R) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp8u> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp8u)0, (Npp8u)255, 12345);
  TestDataGenerator::generateRandom(src2, (Npp8u)0, (Npp8u)255, 54321);

  for (int i = 0; i < total; i++) {
    if (i % 4 == 3) {
      expected[i] = src2[i];
    } else {
      expected[i] = src1[i] ^ src2[i];
    }
  }

  NppImageMemory<Npp8u> d_src1(width * channels, height);
  NppImageMemory<Npp8u> d_src2(width * channels, height);
  NppImageMemory<Npp8u> d_dst(width * channels, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);
  d_dst.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiXor_8u_AC4R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi),
            NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

// ==================== Xor 16u C3/C4/AC4 ====================
class Xor16uMultiChannelTest : public NppTestBase {};

TEST_F(Xor16uMultiChannelTest, Xor_16u_C3R) {
  const int width = 32, height = 32, channels = 3;
  const int total = width * height * channels;

  std::vector<Npp16u> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp16u)0, (Npp16u)65535, 12345);
  TestDataGenerator::generateRandom(src2, (Npp16u)0, (Npp16u)65535, 54321);

  for (int i = 0; i < total; i++) {
    expected[i] = src1[i] ^ src2[i];
  }

  NppImageMemory<Npp16u> d_src1(width * channels, height);
  NppImageMemory<Npp16u> d_src2(width * channels, height);
  NppImageMemory<Npp16u> d_dst(width * channels, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiXor_16u_C3R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi),
            NPP_NO_ERROR);

  std::vector<Npp16u> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(Xor16uMultiChannelTest, Xor_16u_C4R) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp16u> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp16u)0, (Npp16u)65535, 12345);
  TestDataGenerator::generateRandom(src2, (Npp16u)0, (Npp16u)65535, 54321);

  for (int i = 0; i < total; i++) {
    expected[i] = src1[i] ^ src2[i];
  }

  NppImageMemory<Npp16u> d_src1(width * channels, height);
  NppImageMemory<Npp16u> d_src2(width * channels, height);
  NppImageMemory<Npp16u> d_dst(width * channels, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiXor_16u_C4R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi),
            NPP_NO_ERROR);

  std::vector<Npp16u> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(Xor16uMultiChannelTest, Xor_16u_AC4R) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp16u> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp16u)0, (Npp16u)65535, 12345);
  TestDataGenerator::generateRandom(src2, (Npp16u)0, (Npp16u)65535, 54321);

  for (int i = 0; i < total; i++) {
    if (i % 4 == 3) {
      expected[i] = src2[i];
    } else {
      expected[i] = src1[i] ^ src2[i];
    }
  }

  NppImageMemory<Npp16u> d_src1(width * channels, height);
  NppImageMemory<Npp16u> d_src2(width * channels, height);
  NppImageMemory<Npp16u> d_dst(width * channels, height);
  d_src1.copyFromHost(src1);
  d_src2.copyFromHost(src2);
  d_dst.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiXor_16u_AC4R(d_src1.get(), d_src1.step(), d_src2.get(), d_src2.step(), d_dst.get(), d_dst.step(), roi),
            NPP_NO_ERROR);

  std::vector<Npp16u> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

// ==================== Not 8u C3/C4/AC4 ====================
class Not8uMultiChannelTest : public NppTestBase {};

TEST_F(Not8uMultiChannelTest, Not_8u_C3R) {
  const int width = 32, height = 32, channels = 3;
  const int total = width * height * channels;

  std::vector<Npp8u> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp8u)0, (Npp8u)255, 12345);

  for (int i = 0; i < total; i++) {
    expected[i] = ~src[i];
  }

  NppImageMemory<Npp8u> d_src(width * channels, height);
  NppImageMemory<Npp8u> d_dst(width * channels, height);
  d_src.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiNot_8u_C3R(d_src.get(), d_src.step(), d_dst.get(), d_dst.step(), roi), NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(Not8uMultiChannelTest, Not_8u_C4R) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp8u> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp8u)0, (Npp8u)255, 12345);

  for (int i = 0; i < total; i++) {
    expected[i] = ~src[i];
  }

  NppImageMemory<Npp8u> d_src(width * channels, height);
  NppImageMemory<Npp8u> d_dst(width * channels, height);
  d_src.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiNot_8u_C4R(d_src.get(), d_src.step(), d_dst.get(), d_dst.step(), roi), NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(Not8uMultiChannelTest, Not_8u_AC4R) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp8u> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp8u)0, (Npp8u)255, 12345);

  for (int i = 0; i < total; i++) {
    if (i % 4 == 3) {
      expected[i] = src[i];
    } else {
      expected[i] = ~src[i];
    }
  }

  NppImageMemory<Npp8u> d_src(width * channels, height);
  NppImageMemory<Npp8u> d_dst(width * channels, height);
  d_src.copyFromHost(src);
  d_dst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiNot_8u_AC4R(d_src.get(), d_src.step(), d_dst.get(), d_dst.step(), roi), NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_dst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

// ==================== Not inplace ====================
class Not8uInplaceTest : public NppTestBase {};

TEST_F(Not8uInplaceTest, Not_8u_C1IR) {
  const int width = 32, height = 32;
  const int total = width * height;

  std::vector<Npp8u> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp8u)0, (Npp8u)255, 12345);

  for (int i = 0; i < total; i++) {
    expected[i] = ~src[i];
  }

  NppImageMemory<Npp8u> d_srcdst(width, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiNot_8u_C1IR(d_srcdst.get(), d_srcdst.step(), roi), NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(Not8uInplaceTest, Not_8u_C3IR) {
  const int width = 32, height = 32, channels = 3;
  const int total = width * height * channels;

  std::vector<Npp8u> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp8u)0, (Npp8u)255, 12345);

  for (int i = 0; i < total; i++) {
    expected[i] = ~src[i];
  }

  NppImageMemory<Npp8u> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiNot_8u_C3IR(d_srcdst.get(), d_srcdst.step(), roi), NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(Not8uInplaceTest, Not_8u_C4IR) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp8u> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp8u)0, (Npp8u)255, 12345);

  for (int i = 0; i < total; i++) {
    expected[i] = ~src[i];
  }

  NppImageMemory<Npp8u> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiNot_8u_C4IR(d_srcdst.get(), d_srcdst.step(), roi), NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(Not8uInplaceTest, Not_8u_AC4IR) {
  const int width = 32, height = 32, channels = 4;
  const int total = width * height * channels;

  std::vector<Npp8u> src(total), expected(total);
  TestDataGenerator::generateRandom(src, (Npp8u)0, (Npp8u)255, 12345);

  for (int i = 0; i < total; i++) {
    if (i % 4 == 3) {
      expected[i] = src[i];
    } else {
      expected[i] = ~src[i];
    }
  }

  NppImageMemory<Npp8u> d_srcdst(width * channels, height);
  d_srcdst.copyFromHost(src);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiNot_8u_AC4IR(d_srcdst.get(), d_srcdst.step(), roi), NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

// ==================== And/Or/Xor inplace ====================
class BitwiseInplaceTest : public NppTestBase {};

TEST_F(BitwiseInplaceTest, And_8u_C1IR) {
  const int width = 32, height = 32;
  const int total = width * height;

  std::vector<Npp8u> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp8u)0, (Npp8u)255, 12345);
  TestDataGenerator::generateRandom(src2, (Npp8u)0, (Npp8u)255, 54321);

  for (int i = 0; i < total; i++) {
    expected[i] = src1[i] & src2[i];
  }

  NppImageMemory<Npp8u> d_src(width, height);
  NppImageMemory<Npp8u> d_srcdst(width, height);
  d_src.copyFromHost(src1);
  d_srcdst.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiAnd_8u_C1IR(d_src.get(), d_src.step(), d_srcdst.get(), d_srcdst.step(), roi), NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(BitwiseInplaceTest, Or_8u_C1IR) {
  const int width = 32, height = 32;
  const int total = width * height;

  std::vector<Npp8u> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp8u)0, (Npp8u)255, 12345);
  TestDataGenerator::generateRandom(src2, (Npp8u)0, (Npp8u)255, 54321);

  for (int i = 0; i < total; i++) {
    expected[i] = src1[i] | src2[i];
  }

  NppImageMemory<Npp8u> d_src(width, height);
  NppImageMemory<Npp8u> d_srcdst(width, height);
  d_src.copyFromHost(src1);
  d_srcdst.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiOr_8u_C1IR(d_src.get(), d_src.step(), d_srcdst.get(), d_srcdst.step(), roi), NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}

TEST_F(BitwiseInplaceTest, Xor_8u_C1IR) {
  const int width = 32, height = 32;
  const int total = width * height;

  std::vector<Npp8u> src1(total), src2(total), expected(total);
  TestDataGenerator::generateRandom(src1, (Npp8u)0, (Npp8u)255, 12345);
  TestDataGenerator::generateRandom(src2, (Npp8u)0, (Npp8u)255, 54321);

  for (int i = 0; i < total; i++) {
    expected[i] = src1[i] ^ src2[i];
  }

  NppImageMemory<Npp8u> d_src(width, height);
  NppImageMemory<Npp8u> d_srcdst(width, height);
  d_src.copyFromHost(src1);
  d_srcdst.copyFromHost(src2);

  NppiSize roi = {width, height};
  ASSERT_EQ(nppiXor_8u_C1IR(d_src.get(), d_src.step(), d_srcdst.get(), d_srcdst.step(), roi), NPP_NO_ERROR);

  std::vector<Npp8u> result(total);
  d_srcdst.copyToHost(result);
  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}
