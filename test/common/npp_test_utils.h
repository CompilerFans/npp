#pragma once

#include <algorithm>
#include <cmath>

namespace NPPTestUtils {

// Get tolerance for integer types
inline int GetIntegerTolerance() {
#ifdef NPP_STRICT_TESTING
  return 0; // Strict mode: no difference allowed
#else
  return 1; // Tolerant mode: allow 1 unit difference
#endif
}

// Precision-controlled comparison function
template <typename T> bool IsEqual(T actual, T expected) {
#ifdef NPP_STRICT_TESTING
  // Strict mode: require exact equality
  return actual == expected;
#else
  // Tolerant mode: allow small differences
  if constexpr (std::is_floating_point_v<T>) {
    // Floating point: use relative error
    T abs_actual = std::abs(actual);
    T abs_expected = std::abs(expected);
    T max_val = std::max(abs_actual, abs_expected);

    if (max_val < 1e-7f) {
      // For very small values, use absolute error
      return std::abs(actual - expected) <= 1e-7f;
    } else {
      // Use relative error of 0.1%
      return std::abs(actual - expected) <= max_val * 0.001f;
    }
  } else {
    // Integer types: allow small differences
    return std::abs(static_cast<int>(actual) - static_cast<int>(expected)) <= GetIntegerTolerance();
  }
#endif
}

// Get tolerance for floating point types
template <typename T> T GetFloatTolerance() {
#ifdef NPP_STRICT_TESTING
  return T(0); // Strict mode: no difference allowed
#else
  if constexpr (std::is_same_v<T, float>) {
    return 1e-5f; // 32-bit float: 1e-5
  } else {
    return 1e-10; // 64-bit float: 1e-10
  }
#endif
}

// Special tolerance for arithmetic functions - handle differences in algorithm implementations
template <typename T> bool IsArithmeticEqual(T actual, T expected, const char *func_name = nullptr) {
#ifdef NPP_STRICT_TESTING
  return actual == expected;
#else
  // Set different tolerance based on function type
  if (func_name) {
    if (strstr(func_name, "Exp") || strstr(func_name, "exp")) {
      // Exponential functions: larger tolerance due to algorithm differences
      if constexpr (std::is_floating_point_v<T>) {
        return std::abs(actual - expected) <= std::max(std::abs(expected) * 0.01f, T(0.01f));
      } else {
        // 8-bit integer exponential: allow larger differences
        return std::abs(static_cast<int>(actual) - static_cast<int>(expected)) <= 5;
      }
    } else if (strstr(func_name, "Ln") || strstr(func_name, "ln") || strstr(func_name, "Log")) {
      // Logarithmic functions: medium tolerance
      if constexpr (std::is_floating_point_v<T>) {
        return std::abs(actual - expected) <= std::max(std::abs(expected) * 0.005f, T(0.005f));
      } else {
        return std::abs(static_cast<int>(actual) - static_cast<int>(expected)) <= 3;
      }
    } else if (strstr(func_name, "Sqrt") || strstr(func_name, "sqrt")) {
      // Square root: small tolerance
      if constexpr (std::is_floating_point_v<T>) {
        return std::abs(actual - expected) <= std::max(std::abs(expected) * 0.002f, T(0.001f));
      } else {
        return std::abs(static_cast<int>(actual) - static_cast<int>(expected)) <= 2;
      }
    }
  }

  // Default to standard tolerance
  return IsEqual(actual, expected);
#endif
}

// Get test mode description
inline const char *GetTestModeDescription() {
#ifdef NPP_STRICT_TESTING
  return "Strict Testing Mode (exact match required)";
#else
  return "Tolerant Testing Mode (minor differences allowed)";
#endif
}

// Macros for test output
#define NPP_EXPECT_EQUAL(actual, expected)                                                                             \
  EXPECT_TRUE(NPPTestUtils::IsEqual(actual, expected))                                                                 \
      << "Values differ: actual=" << actual << ", expected=" << expected                                               \
      << " (Test mode: " << NPPTestUtils::GetTestModeDescription() << ")"

#define NPP_EXPECT_ARITHMETIC_EQUAL(actual, expected, func_name)                                                       \
  EXPECT_TRUE(NPPTestUtils::IsArithmeticEqual(actual, expected, func_name))                                            \
      << "Arithmetic function " << func_name << " values differ: actual=" << actual << ", expected=" << expected       \
      << " (Test mode: " << NPPTestUtils::GetTestModeDescription() << ")"

} // namespace NPPTestUtils