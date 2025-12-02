#pragma once

#include "npp.h"
#include "npp_test_base.h"
#include <cmath>
#include <functional>
#include <gtest/gtest.h>
#include <string>
#include <type_traits>
#include <vector>

namespace npp_arithmetic_test {

using namespace npp_functional_test;

// Expected value calculator functor types
template <typename T> using UnaryExpectFunc = std::function<T(T)>;
template <typename T> using BinaryExpectFunc = std::function<T(T, T)>;
template <typename T, typename ConstT = T> using UnaryConstExpectFunc = std::function<T(T, ConstT)>;
template <typename T, typename AccT = Npp32f> using TernaryInplaceExpectFunc = std::function<AccT(T, T, AccT)>;

// Common expected value functors
namespace expect {

template <typename T> T abs_val(T x) {
  if constexpr (std::is_floating_point_v<T>) {
    return std::fabs(x);
  } else {
    return static_cast<T>(std::abs(static_cast<int>(x)));
  }
}

template <typename T> T abs_diff(T a, T b) {
  if constexpr (std::is_floating_point_v<T>) {
    return std::fabs(a - b);
  } else if constexpr (std::is_unsigned_v<T>) {
    return (a > b) ? (a - b) : (b - a);
  } else {
    return static_cast<T>(std::abs(static_cast<int>(a) - static_cast<int>(b)));
  }
}

template <typename T, typename ConstT = T> T abs_diff_c(T x, ConstT c) {
  if constexpr (std::is_floating_point_v<T>) {
    return std::fabs(static_cast<T>(x) - static_cast<T>(c));
  } else if constexpr (std::is_unsigned_v<T>) {
    T tc = static_cast<T>(c);
    return (x > tc) ? (x - tc) : (tc - x);
  } else {
    return static_cast<T>(std::abs(static_cast<int>(x) - static_cast<int>(c)));
  }
}

template <typename T, typename AccT = Npp32f> AccT add_product(T a, T b, AccT acc) {
  return acc + static_cast<AccT>(a) * static_cast<AccT>(b);
}

template <typename T, typename AccT = Npp32f> AccT add_square(T x, AccT acc) {
  return acc + static_cast<AccT>(x) * static_cast<AccT>(x);
}

template <typename T, typename AccT = Npp32f> AccT add_weighted(T x, AccT acc, AccT alpha) {
  return acc * (static_cast<AccT>(1) - alpha) + static_cast<AccT>(x) * alpha;
}

// AddC: dst = saturate(src + constant)
template <typename T, typename ConstT = T> T add_c(T x, ConstT c) {
  if constexpr (std::is_floating_point_v<T>) {
    return static_cast<T>(x) + static_cast<T>(c);
  } else if constexpr (std::is_same_v<T, Npp8u>) {
    int result = static_cast<int>(x) + static_cast<int>(c);
    return static_cast<T>(std::max(0, std::min(255, result)));
  } else if constexpr (std::is_same_v<T, Npp16u>) {
    int result = static_cast<int>(x) + static_cast<int>(c);
    return static_cast<T>(std::max(0, std::min(65535, result)));
  } else if constexpr (std::is_same_v<T, Npp16s>) {
    int result = static_cast<int>(x) + static_cast<int>(c);
    return static_cast<T>(std::max(-32768, std::min(32767, result)));
  } else {
    return static_cast<T>(x + c);
  }
}

// AddC with scale factor: dst = saturate((src + constant) >> scaleFactor)
template <typename T, typename ConstT = T> T add_c_sfs(T x, ConstT c, int scaleFactor) {
  if constexpr (std::is_same_v<T, Npp8u>) {
    int result = static_cast<int>(x) + static_cast<int>(c);
    if (scaleFactor > 0) {
      result = (result + (1 << (scaleFactor - 1))) >> scaleFactor;
    }
    return static_cast<T>(std::max(0, std::min(255, result)));
  } else if constexpr (std::is_same_v<T, Npp16u>) {
    int result = static_cast<int>(x) + static_cast<int>(c);
    if (scaleFactor > 0) {
      result = (result + (1 << (scaleFactor - 1))) >> scaleFactor;
    }
    return static_cast<T>(std::max(0, std::min(65535, result)));
  } else if constexpr (std::is_same_v<T, Npp16s>) {
    int result = static_cast<int>(x) + static_cast<int>(c);
    if (scaleFactor > 0) {
      result = (result + (1 << (scaleFactor - 1))) >> scaleFactor;
    }
    return static_cast<T>(std::max(-32768, std::min(32767, result)));
  } else {
    return static_cast<T>(x + c);
  }
}

} // namespace expect

// Test parameter structures
struct UnaryOpTestConfig {
  int width;
  int height;
  int channels;
  bool use_ctx;
  bool in_place;
  std::string test_name;
};

struct BinaryOpTestConfig {
  int width;
  int height;
  int channels;
  bool use_ctx;
  std::string test_name;
};

struct ConstOpTestConfig {
  int width;
  int height;
  int channels;
  bool use_ctx;
  std::string test_name;
};

// Helper functions for min/max values - defined first so they can be used by generateTestData
template <typename T> constexpr T getMinValue() {
  if constexpr (std::is_same_v<T, Npp8u>) {
    return 0;
  } else if constexpr (std::is_same_v<T, Npp16u>) {
    return 0;
  } else if constexpr (std::is_same_v<T, Npp16s>) {
    return -100;
  } else if constexpr (std::is_same_v<T, Npp32f>) {
    return -100.0f;
  } else if constexpr (std::is_same_v<T, Npp64f>) {
    return -100.0;
  }
  return T{};
}

template <typename T> constexpr T getMaxValue() {
  if constexpr (std::is_same_v<T, Npp8u>) {
    return 255;
  } else if constexpr (std::is_same_v<T, Npp16u>) {
    return 65535;
  } else if constexpr (std::is_same_v<T, Npp16s>) {
    return 100;
  } else if constexpr (std::is_same_v<T, Npp32f>) {
    return 100.0f;
  } else if constexpr (std::is_same_v<T, Npp64f>) {
    return 100.0;
  }
  return T{};
}

// Helper to generate test data
template <typename T> std::vector<T> generateTestData(int width, int height, int channels, unsigned seed = 12345) {
  std::vector<T> data(width * height * channels);
  TestDataGenerator::generateRandom(data, getMinValue<T>(), getMaxValue<T>(), seed);
  return data;
}

// Parameterized test base class for unary operations (e.g., Abs)
template <typename T> class UnaryOpParamTest : public NppTestBase {
protected:
  void runUnaryTest(const UnaryOpTestConfig &config, UnaryExpectFunc<T> expectFunc,
                    std::function<NppStatus(const T *, int, T *, int, NppiSize)> nppFunc,
                    std::function<NppStatus(const T *, int, T *, int, NppiSize, NppStreamContext)> nppFuncCtx = nullptr,
                    std::function<NppStatus(T *, int, NppiSize)> nppFuncInplace = nullptr,
                    std::function<NppStatus(T *, int, NppiSize, NppStreamContext)> nppFuncInplaceCtx = nullptr) {

    const int width = config.width;
    const int height = config.height;
    const int channels = config.channels;

    std::vector<T> srcData(width * height * channels);
    TestDataGenerator::generateRandom(srcData, getMinValue<T>(), getMaxValue<T>(), 12345);

    std::vector<T> expectedData(width * height * channels);
    for (size_t i = 0; i < expectedData.size(); i++) {
      expectedData[i] = expectFunc(srcData[i]);
    }

    NppImageMemory<T> src(width * channels, height);
    src.copyFromHost(srcData);

    NppiSize roi = {width, height};
    NppStatus status;

    if (config.in_place) {
      if (config.use_ctx && nppFuncInplaceCtx) {
        NppStreamContext ctx;
        ctx.hStream = 0;
        status = nppFuncInplaceCtx(src.get(), src.step(), roi, ctx);
      } else if (nppFuncInplace) {
        status = nppFuncInplace(src.get(), src.step(), roi);
      } else {
        FAIL() << "In-place function not provided";
      }
      ASSERT_EQ(status, NPP_NO_ERROR) << "NPP function failed";

      std::vector<T> resultData(width * height * channels);
      src.copyToHost(resultData);

      EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, getTolerance<T>()))
          << "Result mismatch for " << config.test_name;
    } else {
      NppImageMemory<T> dst(width * channels, height);

      if (config.use_ctx && nppFuncCtx) {
        NppStreamContext ctx;
        ctx.hStream = 0;
        status = nppFuncCtx(src.get(), src.step(), dst.get(), dst.step(), roi, ctx);
      } else if (nppFunc) {
        status = nppFunc(src.get(), src.step(), dst.get(), dst.step(), roi);
      } else {
        FAIL() << "Non-inplace function not provided";
      }
      ASSERT_EQ(status, NPP_NO_ERROR) << "NPP function failed";

      std::vector<T> resultData(width * height * channels);
      dst.copyToHost(resultData);

      EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, getTolerance<T>()))
          << "Result mismatch for " << config.test_name;
    }
  }

  template <typename U> static constexpr U getTolerance() {
    if constexpr (std::is_floating_point_v<U>) {
      return static_cast<U>(1e-5);
    }
    return U{0};
  }
};

// Parameterized test base class for binary operations (e.g., AbsDiff)
template <typename T> class BinaryOpParamTest : public NppTestBase {
protected:
  void runBinaryTest(const BinaryOpTestConfig &config, BinaryExpectFunc<T> expectFunc,
                     std::function<NppStatus(const T *, int, const T *, int, T *, int, NppiSize)> nppFunc,
                     std::function<NppStatus(const T *, int, const T *, int, T *, int, NppiSize, NppStreamContext)>
                         nppFuncCtx = nullptr) {

    const int width = config.width;
    const int height = config.height;
    const int channels = config.channels;

    std::vector<T> src1Data(width * height * channels);
    std::vector<T> src2Data(width * height * channels);
    TestDataGenerator::generateRandom(src1Data, getMinValue<T>(), getMaxValue<T>(), 12345);
    TestDataGenerator::generateRandom(src2Data, getMinValue<T>(), getMaxValue<T>(), 54321);

    std::vector<T> expectedData(width * height * channels);
    for (size_t i = 0; i < expectedData.size(); i++) {
      expectedData[i] = expectFunc(src1Data[i], src2Data[i]);
    }

    NppImageMemory<T> src1(width * channels, height);
    NppImageMemory<T> src2(width * channels, height);
    NppImageMemory<T> dst(width * channels, height);

    src1.copyFromHost(src1Data);
    src2.copyFromHost(src2Data);

    NppiSize roi = {width, height};
    NppStatus status;

    if (config.use_ctx && nppFuncCtx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppFuncCtx(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi, ctx);
    } else {
      status = nppFunc(src1.get(), src1.step(), src2.get(), src2.step(), dst.get(), dst.step(), roi);
    }

    ASSERT_EQ(status, NPP_NO_ERROR) << "NPP function failed";

    std::vector<T> resultData(width * height * channels);
    dst.copyToHost(resultData);

    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, getTolerance<T>()))
        << "Result mismatch for " << config.test_name;
  }

  template <typename U> static constexpr U getTolerance() {
    if constexpr (std::is_floating_point_v<U>) {
      return static_cast<U>(1e-5);
    }
    return U{0};
  }
};

// Parameterized test base class for constant operations (e.g., AbsDiffC)
template <typename T, typename ConstT = T> class ConstOpParamTest : public NppTestBase {
protected:
  void runConstTest(
      const ConstOpTestConfig &config, ConstT constant, UnaryConstExpectFunc<T, ConstT> expectFunc,
      std::function<NppStatus(const T *, int, T *, int, NppiSize, ConstT)> nppFunc,
      std::function<NppStatus(const T *, int, T *, int, NppiSize, ConstT, NppStreamContext)> nppFuncCtx = nullptr) {

    const int width = config.width;
    const int height = config.height;
    const int channels = config.channels;

    std::vector<T> srcData(width * height * channels);
    TestDataGenerator::generateRandom(srcData, getMinValue<T>(), getMaxValue<T>(), 12345);

    std::vector<T> expectedData(width * height * channels);
    for (size_t i = 0; i < expectedData.size(); i++) {
      expectedData[i] = expectFunc(srcData[i], constant);
    }

    NppImageMemory<T> src(width * channels, height);
    NppImageMemory<T> dst(width * channels, height);

    src.copyFromHost(srcData);

    NppiSize roi = {width, height};
    NppStatus status;

    if (config.use_ctx && nppFuncCtx) {
      NppStreamContext ctx;
      ctx.hStream = 0;
      status = nppFuncCtx(src.get(), src.step(), dst.get(), dst.step(), roi, constant, ctx);
    } else {
      status = nppFunc(src.get(), src.step(), dst.get(), dst.step(), roi, constant);
    }

    ASSERT_EQ(status, NPP_NO_ERROR) << "NPP function failed";

    std::vector<T> resultData(width * height * channels);
    dst.copyToHost(resultData);

    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, getTolerance<T>()))
        << "Result mismatch for " << config.test_name;
  }

  template <typename U> static constexpr U getTolerance() {
    if constexpr (std::is_floating_point_v<U>) {
      return static_cast<U>(1e-5);
    }
    return U{0};
  }
};

// Macro helpers for defining tests with less boilerplate
#define DEFINE_UNARY_OP_TEST(TestClass, TestName, Type, Width, Height, Channels, UseCtx, InPlace, ExpectFunc, NppFunc, \
                             NppFuncCtx, NppFuncIP, NppFuncIPCtx)                                                      \
  TEST_F(TestClass, TestName) {                                                                                        \
    UnaryOpTestConfig config = {Width, Height, Channels, UseCtx, InPlace, #TestName};                                  \
    runUnaryTest(config, ExpectFunc, NppFunc, NppFuncCtx, NppFuncIP, NppFuncIPCtx);                                    \
  }

#define DEFINE_BINARY_OP_TEST(TestClass, TestName, Type, Width, Height, Channels, UseCtx, ExpectFunc, NppFunc,         \
                              NppFuncCtx)                                                                              \
  TEST_F(TestClass, TestName) {                                                                                        \
    BinaryOpTestConfig config = {Width, Height, Channels, UseCtx, #TestName};                                          \
    runBinaryTest(config, ExpectFunc, NppFunc, NppFuncCtx);                                                            \
  }

#define DEFINE_CONST_OP_TEST(TestClass, TestName, Type, Width, Height, Channels, UseCtx, Constant, ExpectFunc,         \
                             NppFunc, NppFuncCtx)                                                                      \
  TEST_F(TestClass, TestName) {                                                                                        \
    ConstOpTestConfig config = {Width, Height, Channels, UseCtx, #TestName};                                           \
    runConstTest(config, Constant, ExpectFunc, NppFunc, NppFuncCtx);                                                   \
  }

} // namespace npp_arithmetic_test
