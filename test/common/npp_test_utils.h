#pragma once

#include <cmath>
#include <algorithm>

namespace NPPTestUtils {

// 获取整数类型的容差
inline int GetIntegerTolerance() {
#ifdef NPP_STRICT_TESTING
    return 0;  // 严格模式：不允许差异
#else
    return 1;  // 宽容模式：允许1个单位的差异
#endif
}

// 精度控制的比较函数
template<typename T>
bool IsEqual(T actual, T expected) {
#ifdef NPP_STRICT_TESTING
    // 严格模式：要求完全相等
    return actual == expected;
#else
    // 宽容模式：允许小的差异
    if constexpr (std::is_floating_point_v<T>) {
        // 浮点数：使用相对误差
        T abs_actual = std::abs(actual);
        T abs_expected = std::abs(expected);
        T max_val = std::max(abs_actual, abs_expected);
        
        if (max_val < 1e-7f) {
            // 对于非常小的数值，使用绝对误差
            return std::abs(actual - expected) <= 1e-7f;
        } else {
            // 使用相对误差 0.1%
            return std::abs(actual - expected) <= max_val * 0.001f;
        }
    } else {
        // 整数类型：允许小的差异
        return std::abs(static_cast<int>(actual) - static_cast<int>(expected)) <= GetIntegerTolerance();
    }
#endif
}

// 获取浮点类型的容差
template<typename T>
T GetFloatTolerance() {
#ifdef NPP_STRICT_TESTING
    return T(0);  // 严格模式：不允许差异
#else
    if constexpr (std::is_same_v<T, float>) {
        return 1e-5f;  // 32位浮点：1e-5
    } else {
        return 1e-10;  // 64位浮点：1e-10
    }
#endif
}

// 算术函数特殊容差 - 用于处理不同算法实现的差异
template<typename T>
bool IsArithmeticEqual(T actual, T expected, const char* func_name = nullptr) {
#ifdef NPP_STRICT_TESTING
    return actual == expected;
#else
    // 根据函数类型设置不同的容差
    if (func_name) {
        if (strstr(func_name, "Exp") || strstr(func_name, "exp")) {
            // 指数函数：较大容差因为算法差异
            if constexpr (std::is_floating_point_v<T>) {
                return std::abs(actual - expected) <= std::max(std::abs(expected) * 0.01f, T(0.01f));
            } else {
                // 8位整数指数：允许更大差异
                return std::abs(static_cast<int>(actual) - static_cast<int>(expected)) <= 5;
            }
        } else if (strstr(func_name, "Ln") || strstr(func_name, "ln") || strstr(func_name, "Log")) {
            // 对数函数：中等容差
            if constexpr (std::is_floating_point_v<T>) {
                return std::abs(actual - expected) <= std::max(std::abs(expected) * 0.005f, T(0.005f));
            } else {
                return std::abs(static_cast<int>(actual) - static_cast<int>(expected)) <= 3;
            }
        } else if (strstr(func_name, "Sqrt") || strstr(func_name, "sqrt")) {
            // 平方根：小容差
            if constexpr (std::is_floating_point_v<T>) {
                return std::abs(actual - expected) <= std::max(std::abs(expected) * 0.002f, T(0.001f));
            } else {
                return std::abs(static_cast<int>(actual) - static_cast<int>(expected)) <= 2;
            }
        }
    }
    
    // 默认使用标准容差
    return IsEqual(actual, expected);
#endif
}

// 获取测试模式描述
inline const char* GetTestModeDescription() {
#ifdef NPP_STRICT_TESTING
    return "Strict Testing Mode (exact match required)";
#else
    return "Tolerant Testing Mode (minor differences allowed)";
#endif
}

// 用于测试输出的宏
#define NPP_EXPECT_EQUAL(actual, expected) \
    EXPECT_TRUE(NPPTestUtils::IsEqual(actual, expected)) \
        << "Values differ: actual=" << actual << ", expected=" << expected \
        << " (Test mode: " << NPPTestUtils::GetTestModeDescription() << ")"

#define NPP_EXPECT_ARITHMETIC_EQUAL(actual, expected, func_name) \
    EXPECT_TRUE(NPPTestUtils::IsArithmeticEqual(actual, expected, func_name)) \
        << "Arithmetic function " << func_name << " values differ: actual=" << actual \
        << ", expected=" << expected << " (Test mode: " << NPPTestUtils::GetTestModeDescription() << ")"

} // namespace NPPTestUtils