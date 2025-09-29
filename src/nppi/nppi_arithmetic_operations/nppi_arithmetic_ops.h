#pragma once

#include "npp.h"
#include <cuda_runtime.h>
#include <limits>
#include <type_traits>

namespace nppi {
namespace arithmetic {

// Type-specific saturating cast utility
template <typename T> __device__ __host__ inline T saturate_cast(double value) {
  if constexpr (std::is_same_v<T, Npp8u>) {
    return static_cast<T>(value < 0.0 ? 0.0 : (value > 255.0 ? 255.0 : value));
  } else if constexpr (std::is_same_v<T, Npp16u>) {
    return static_cast<T>(value < 0.0 ? 0.0 : (value > 65535.0 ? 65535.0 : value));
  } else if constexpr (std::is_same_v<T, Npp16s>) {
    return static_cast<T>(value < -32768.0 ? -32768.0 : (value > 32767.0 ? 32767.0 : value));
  } else if constexpr (std::is_same_v<T, Npp32s>) {
    return static_cast<T>(value < -2147483648.0 ? -2147483648.0 : (value > 2147483647.0 ? 2147483647.0 : value));
  } else if constexpr (std::is_same_v<T, Npp32f>) {
    return static_cast<T>(value);
  }
  return static_cast<T>(value);
}

// ============================================================================
// Binary Operation Functors
// ============================================================================

template <typename T> struct AddOp {
  __device__ __host__ T operator()(T a, T b, int scaleFactor = 0) const {
    double result = static_cast<double>(a) + static_cast<double>(b);
    if (scaleFactor > 0 && std::is_integral_v<T>) {
      result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
    }
    return saturate_cast<T>(result);
  }
};

template <typename T> struct SubOp {
  __device__ __host__ T operator()(T a, T b, int scaleFactor = 0) const {
    // NPP Sub convention: pSrc2 - pSrc1, so b - a
    double result = static_cast<double>(b) - static_cast<double>(a);
    if (scaleFactor > 0 && std::is_integral_v<T>) {
      result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
    }
    return saturate_cast<T>(result);
  }
};

template <typename T> struct MulOp {
  __device__ __host__ T operator()(T a, T b, int scaleFactor = 0) const {
    double result = static_cast<double>(a) * static_cast<double>(b);
    if (scaleFactor > 0 && std::is_integral_v<T>) {
      result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
    }
    return saturate_cast<T>(result);
  }
};

template <typename T> struct DivOp {
  __device__ __host__ T operator()(T a, T b, int scaleFactor = 0) const {
    if (a == T(0)) {
      // Division by zero - saturate to maximum value for integers
      if constexpr (std::is_same_v<T, Npp8u>)
        return T(255);
      else if constexpr (std::is_same_v<T, Npp16u>)
        return T(65535);
      else if constexpr (std::is_same_v<T, Npp16s>)
        return T(32767);
      else
        return T(0);
    }
    // NPP Div convention: pSrc2 / pSrc1, so b / a
    double result = static_cast<double>(b) / static_cast<double>(a);
    if (scaleFactor > 0 && std::is_integral_v<T>) {
      result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
    }
    return saturate_cast<T>(result);
  }
};

template <typename T> struct AndOp {
  __device__ __host__ T operator()(T a, T b, int = 0) const {
    if constexpr (std::is_integral_v<T>) {
      return a & b;
    }
    return T(0);
  }
};

template <typename T> struct OrOp {
  __device__ __host__ T operator()(T a, T b, int = 0) const {
    if constexpr (std::is_integral_v<T>) {
      return a | b;
    }
    return T(0);
  }
};

template <typename T> struct XorOp {
  __device__ __host__ T operator()(T a, T b, int = 0) const {
    if constexpr (std::is_integral_v<T>) {
      return a ^ b;
    }
    return T(0);
  }
};

template <typename T> struct AbsDiffOp {
  __device__ __host__ T operator()(T a, T b, int scaleFactor = 0) const {
    double result = std::abs(static_cast<double>(a) - static_cast<double>(b));
    if (scaleFactor > 0 && std::is_integral_v<T>) {
      result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
    }
    return saturate_cast<T>(result);
  }
};

template <typename T> struct MulScaleOp {
  __device__ __host__ T operator()(T a, T b, int = 0) const {
    if constexpr (std::is_same_v<T, Npp8u>) {
      // For 8-bit: result = (a * b) / 255
      double result = (static_cast<double>(a) * static_cast<double>(b)) / 255.0;
      return saturate_cast<T>(result);
    } else if constexpr (std::is_same_v<T, Npp16u>) {
      // For 16-bit: result = (a * b) / 65535
      double result = (static_cast<double>(a) * static_cast<double>(b)) / 65535.0;
      return saturate_cast<T>(result);
    } else {
      // For other types, fallback to regular multiplication
      return saturate_cast<T>(static_cast<double>(a) * static_cast<double>(b));
    }
  }
};

template <typename T> struct MaxOp {
  __device__ __host__ T operator()(T a, T b, int = 0) const { return a > b ? a : b; }
};

template <typename T> struct MinOp {
  __device__ __host__ T operator()(T a, T b, int = 0) const { return a < b ? a : b; }
};

template <typename T> struct MaxEveryOp {
  __device__ __host__ T operator()(T a, T b, int = 0) const { return a > b ? a : b; }
};

template <typename T> struct MinEveryOp {
  __device__ __host__ T operator()(T a, T b, int = 0) const { return a < b ? a : b; }
};

// ============================================================================
// Unary Operation Functors
// ============================================================================

template <typename T> struct AbsOp {
  __device__ __host__ T operator()(T a, T = T(0), int = 0) const { return std::abs(a); }
};

template <typename T> struct SqrOp {
  __device__ __host__ T operator()(T a, T = T(0), int scaleFactor = 0) const {
    double result = static_cast<double>(a) * static_cast<double>(a);
    if (scaleFactor > 0 && std::is_integral_v<T>) {
      result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
    }
    return saturate_cast<T>(result);
  }
};

template <typename T> struct SqrtOp {
  __device__ __host__ T operator()(T a, T = T(0), int scaleFactor = 0) const {
    double result = std::sqrt(static_cast<double>(a));
    if (scaleFactor > 0 && std::is_integral_v<T>) {
      // NVIDIA NPP uses truncation for scale factors
      result = result / (1 << scaleFactor);
    } else if (std::is_integral_v<T>) {
      // For no scaling, use rounding to match NVIDIA NPP behavior
      result = result + 0.5;
    }
    return saturate_cast<T>(result);
  }
};

template <typename T> struct ExpOp {
  __device__ __host__ T operator()(T a, T = T(0), int scaleFactor = 0) const {
    double result = std::exp(static_cast<double>(a));
    if (scaleFactor > 0 && std::is_integral_v<T>) {
      result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
    }
    return saturate_cast<T>(result);
  }
};

template <typename T> struct LnOp {
  __device__ __host__ T operator()(T a, T = T(0), int scaleFactor = 0) const {
    if constexpr (std::is_same_v<T, Npp32f>) {
      // Handle special values for floating point to match vendor NPP behavior
      if (a == T(0)) {
        return -std::numeric_limits<T>::infinity();
      } else if (a < T(0)) {
        return std::numeric_limits<T>::quiet_NaN();
      }
      return std::log(a);
    } else {
      // For integer types
      if (a <= T(0)) {
        return T(0);
      }
      double result = std::log(static_cast<double>(a));
      if (scaleFactor > 0) {
        result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
      }
      return saturate_cast<T>(result);
    }
  }
};

template <typename T> struct NotOp {
  __device__ __host__ T operator()(T a, T = T(0), int = 0) const {
    if constexpr (std::is_integral_v<T>) {
      return ~a;
    }
    return T(0);
  }
};

// ============================================================================
// Shift Operation Functors
// ============================================================================

template <typename T> class LShiftConstOp {
private:
  int shiftCount;

public:
  __device__ __host__ LShiftConstOp(int shift) : shiftCount(shift) {}

  __device__ __host__ T operator()(T a, T = T(0), int = 0) const {
    if constexpr (std::is_integral_v<T>) {
      if (shiftCount >= 0 && shiftCount < 32) {
        return static_cast<T>(a << shiftCount);
      }
    }
    return a;
  }
};

template <typename T> class RShiftConstOp {
private:
  int shiftCount;

public:
  __device__ __host__ RShiftConstOp(int shift) : shiftCount(shift) {}

  __device__ __host__ T operator()(T a, T = T(0), int = 0) const {
    if constexpr (std::is_integral_v<T>) {
      if (shiftCount >= 0 && shiftCount < 32) {
        return static_cast<T>(a >> shiftCount);
      }
    }
    return a;
  }
};

// ============================================================================
// Constant Operation Functors (for operations with constants)
// ============================================================================

template <typename T> class AddConstOp {
private:
  T constant;

public:
  __device__ __host__ AddConstOp(T c) : constant(c) {}

  __device__ __host__ T operator()(T a, T = T(0), int scaleFactor = 0) const {
    double result = static_cast<double>(a) + static_cast<double>(constant);
    if (scaleFactor > 0 && std::is_integral_v<T>) {
      result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
    }
    return saturate_cast<T>(result);
  }
};

template <typename T> class SubConstOp {
private:
  T constant;

public:
  __device__ __host__ SubConstOp(T c) : constant(c) {}

  __device__ __host__ T operator()(T a, T = T(0), int scaleFactor = 0) const {
    double result = static_cast<double>(a) - static_cast<double>(constant);
    if (scaleFactor > 0 && std::is_integral_v<T>) {
      result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
    }
    return saturate_cast<T>(result);
  }
};

template <typename T> class MulConstOp {
private:
  T constant;

public:
  __device__ __host__ MulConstOp(T c) : constant(c) {}

  __device__ __host__ T operator()(T a, T = T(0), int scaleFactor = 0) const {
    double result = static_cast<double>(a) * static_cast<double>(constant);
    if (scaleFactor > 0 && std::is_integral_v<T>) {
      result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
    }
    return saturate_cast<T>(result);
  }
};

template <typename T> class AndConstOp {
private:
  T constant;

public:
  __device__ __host__ AndConstOp(T c) : constant(c) {}

  __device__ __host__ T operator()(T a, T = T(0), int = 0) const {
    if constexpr (std::is_integral_v<T>) {
      return a & constant;
    }
    return T(0);
  }
};

template <typename T> class OrConstOp {
private:
  T constant;

public:
  __device__ __host__ OrConstOp(T c) : constant(c) {}

  __device__ __host__ T operator()(T a, T = T(0), int = 0) const {
    if constexpr (std::is_integral_v<T>) {
      return a | constant;
    }
    return T(0);
  }
};

template <typename T> class XorConstOp {
private:
  T constant;

public:
  __device__ __host__ XorConstOp(T c) : constant(c) {}

  __device__ __host__ T operator()(T a, T = T(0), int = 0) const {
    if constexpr (std::is_integral_v<T>) {
      return a ^ constant;
    }
    return T(0);
  }
};

template <typename T> class AbsDiffConstOp {
private:
  T constant;

public:
  __device__ __host__ AbsDiffConstOp(T c) : constant(c) {}

  __device__ __host__ T operator()(T a, T = T(0), int scaleFactor = 0) const {
    double result = std::abs(static_cast<double>(a) - static_cast<double>(constant));
    if (scaleFactor > 0 && std::is_integral_v<T>) {
      result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
    }
    return saturate_cast<T>(result);
  }
};

// ============================================================================
// Comparison Operation Functors (output Npp8u)
// ============================================================================

template <typename T> class CompareConstOp {
private:
  T constant;
  NppCmpOp operation;

public:
  __device__ __host__ CompareConstOp(T c, NppCmpOp op) : constant(c), operation(op) {}

  __device__ __host__ Npp8u operator()(T a, T = T(0), int = 0) const {
    bool result = false;
    double src_val = static_cast<double>(a);
    double const_val = static_cast<double>(constant);

    switch (operation) {
    case NPP_CMP_LESS:
      result = src_val < const_val;
      break;
    case NPP_CMP_LESS_EQ:
      result = src_val <= const_val;
      break;
    case NPP_CMP_EQ:
      result = src_val == const_val;
      break;
    case NPP_CMP_GREATER_EQ:
      result = src_val >= const_val;
      break;
    case NPP_CMP_GREATER:
      result = src_val > const_val;
      break;
    default:
      result = false;
      break;
    }
    return result ? Npp8u(255) : Npp8u(0);
  }
};

// ============================================================================
// Complex Operation Functors
// ============================================================================

template <typename T> struct AddProductOp {
  __device__ __host__ T operator()(T a, T b, T c, int scaleFactor = 0) const {
    // Result = a + (b * c)
    double product = static_cast<double>(b) * static_cast<double>(c);
    double result = static_cast<double>(a) + product;
    if (scaleFactor > 0 && std::is_integral_v<T>) {
      result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
    }
    return saturate_cast<T>(result);
  }
};

template <typename T> struct AddSquareOp {
  __device__ __host__ T operator()(T a, T b, int scaleFactor = 0) const {
    // Result = a + (b * b)
    double square = static_cast<double>(b) * static_cast<double>(b);
    double result = static_cast<double>(a) + square;
    if (scaleFactor > 0 && std::is_integral_v<T>) {
      result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
    }
    return saturate_cast<T>(result);
  }
};

template <typename T> class AddWeightedOp {
private:
  T alpha1, alpha2;

public:
  __device__ __host__ AddWeightedOp(T a1, T a2) : alpha1(a1), alpha2(a2) {}

  __device__ __host__ T operator()(T a, T b, int scaleFactor = 0) const {
    // Result = alpha1 * a + alpha2 * b
    double result =
        static_cast<double>(alpha1) * static_cast<double>(a) + static_cast<double>(alpha2) * static_cast<double>(b);
    if (scaleFactor > 0 && std::is_integral_v<T>) {
      result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
    }
    return saturate_cast<T>(result);
  }
};

// ============================================================================
// Alpha Composition Operations
// ============================================================================

template <typename T> struct AlphaCompOp {
  __device__ __host__ T operator()(T src1, T alpha1, T src2, T alpha2, int = 0) const {
    // Standard alpha composition: result = src1 * alpha1 + src2 * alpha2 * (1 - alpha1)
    if constexpr (std::is_same_v<T, Npp8u>) {
      double a1 = static_cast<double>(alpha1) / 255.0;
      double a2 = static_cast<double>(alpha2) / 255.0;
      double result = static_cast<double>(src1) * a1 + static_cast<double>(src2) * a2 * (1.0 - a1);
      return saturate_cast<T>(result);
    } else if constexpr (std::is_same_v<T, Npp32f>) {
      double result = static_cast<double>(src1) * static_cast<double>(alpha1) +
                      static_cast<double>(src2) * static_cast<double>(alpha2) * (1.0 - static_cast<double>(alpha1));
      return static_cast<T>(result);
    }
    return src1; // Fallback
  }
};

template <typename T> class AlphaCompConstOp {
private:
  T constant_alpha;

public:
  __device__ __host__ AlphaCompConstOp(T alpha) : constant_alpha(alpha) {}

  __device__ __host__ T operator()(T src1, T src2, int = 0) const {
    if constexpr (std::is_same_v<T, Npp8u>) {
      double alpha = static_cast<double>(constant_alpha) / 255.0;
      double result = static_cast<double>(src1) * alpha + static_cast<double>(src2) * (1.0 - alpha);
      return saturate_cast<T>(result);
    } else if constexpr (std::is_same_v<T, Npp32f>) {
      double alpha = static_cast<double>(constant_alpha);
      double result = static_cast<double>(src1) * alpha + static_cast<double>(src2) * (1.0 - alpha);
      return static_cast<T>(result);
    }
    return src1; // Fallback
  }
};

template <typename T> struct AlphaPremulOp {
  __device__ __host__ T operator()(T src, T alpha, int = 0) const {
    // Alpha premultiplication: result = src * alpha
    if constexpr (std::is_same_v<T, Npp8u>) {
      double result = static_cast<double>(src) * static_cast<double>(alpha) / 255.0;
      return saturate_cast<T>(result);
    } else if constexpr (std::is_same_v<T, Npp32f>) {
      double result = static_cast<double>(src) * static_cast<double>(alpha);
      return static_cast<T>(result);
    }
    return src; // Fallback
  }
};

template <typename T> class AlphaPremulConstOp {
private:
  T constant_alpha;

public:
  __device__ __host__ AlphaPremulConstOp(T alpha) : constant_alpha(alpha) {}

  __device__ __host__ T operator()(T src, T = T(0), int = 0) const {
    if constexpr (std::is_same_v<T, Npp8u>) {
      double result = static_cast<double>(src) * static_cast<double>(constant_alpha) / 255.0;
      return saturate_cast<T>(result);
    } else if constexpr (std::is_same_v<T, Npp32f>) {
      double result = static_cast<double>(src) * static_cast<double>(constant_alpha);
      return static_cast<T>(result);
    }
    return src; // Fallback
  }
};

// DivConstOp: division by constant with proper zero handling
template <typename T> class DivConstOp {
private:
  T constant;

public:
  __device__ __host__ DivConstOp(T c) : constant(c) {}

  __device__ __host__ T operator()(T a, T = T(0), int scaleFactor = 0) const {
    if (constant == T(0)) {
      // Division by zero: return maximum value for the type
      if constexpr (std::is_integral_v<T>) {
        if constexpr (std::is_signed_v<T>) {
          return (a >= T(0)) ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
        } else {
          return std::numeric_limits<T>::max();
        }
      } else {
        return (a >= T(0)) ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
      }
    }

    double result = static_cast<double>(a) / static_cast<double>(constant);

    if (scaleFactor > 0 && std::is_integral_v<T>) {
      // For division, NPP uses right shift (truncation), not rounding division
      result = result / (1 << scaleFactor);
    }

    return saturate_cast<T>(result);
  }
};

} // namespace arithmetic
} // namespace nppi