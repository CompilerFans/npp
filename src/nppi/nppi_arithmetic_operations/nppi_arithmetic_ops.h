#pragma once

#include "npp.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>
#include <type_traits>

namespace nppi {
namespace arithmetic {

// ============================================================================
// Type traits for complex number detection
// ============================================================================
template <typename T> struct is_complex : std::false_type {};
template <> struct is_complex<Npp16sc> : std::true_type {};
template <> struct is_complex<Npp32sc> : std::true_type {};
template <> struct is_complex<Npp32fc> : std::true_type {};

template <typename T> inline constexpr bool is_complex_v = is_complex<T>::value;

// ============================================================================
// Type traits for 16f (half precision) type detection
// ============================================================================
template <typename T> struct is_16f : std::false_type {};
template <> struct is_16f<Npp16f> : std::true_type {};

template <typename T> inline constexpr bool is_16f_v = is_16f<T>::value;

// ============================================================================
// Npp16f conversion utilities
// ============================================================================
__device__ __forceinline__ __half npp16f_to_half(Npp16f val) {
  return __ushort_as_half(static_cast<unsigned short>(val.fp16));
}

__device__ __forceinline__ Npp16f half_to_npp16f(__half val) {
  Npp16f result;
  result.fp16 = static_cast<short>(__half_as_ushort(val));
  return result;
}

__device__ __forceinline__ Npp16f float_to_npp16f(float val) { return half_to_npp16f(__float2half(val)); }

__device__ __forceinline__ float npp16f_to_float(Npp16f val) { return __half2float(npp16f_to_half(val)); }

// Get the component type of a complex number
template <typename T> struct complex_component { using type = T; };
template <> struct complex_component<Npp16sc> { using type = Npp16s; };
template <> struct complex_component<Npp32sc> { using type = Npp32s; };
template <> struct complex_component<Npp32fc> { using type = Npp32f; };

template <typename T> using complex_component_t = typename complex_component<T>::type;

// ============================================================================
// Type-specific saturating cast utility
// ============================================================================
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

// Saturating cast for complex types - saturates real and imaginary components separately
template <typename T> __device__ __host__ inline T saturate_cast_complex(double re, double im) {
  using ComponentT = complex_component_t<T>;
  T result;
  result.re = saturate_cast<ComponentT>(re);
  result.im = saturate_cast<ComponentT>(im);
  return result;
}

// ============================================================================
// Binary Operation Functors
// ============================================================================

template <typename T> struct AddOp {
  __device__ __host__ T operator()(T a, T b, int scaleFactor = 0) const {
    if constexpr (is_16f_v<T>) {
      __half ha = npp16f_to_half(a);
      __half hb = npp16f_to_half(b);
      return half_to_npp16f(__hadd(ha, hb));
    } else if constexpr (is_complex_v<T>) {
      // Complex addition: (a.re + a.im*i) + (b.re + b.im*i)
      double re = static_cast<double>(a.re) + static_cast<double>(b.re);
      double im = static_cast<double>(a.im) + static_cast<double>(b.im);
      if (scaleFactor > 0) {
        double scale = 1.0 / (1 << scaleFactor);
        re = (re + (1 << (scaleFactor - 1))) * scale;
        im = (im + (1 << (scaleFactor - 1))) * scale;
      }
      return saturate_cast_complex<T>(re, im);
    } else {
      double result = static_cast<double>(a) + static_cast<double>(b);
      if (scaleFactor > 0 && std::is_integral_v<T>) {
        result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
      }
      return saturate_cast<T>(result);
    }
  }
};

template <typename T> struct SubOp {
  __device__ __host__ T operator()(T a, T b, int scaleFactor = 0) const {
    // NPP Sub convention: pSrc2 - pSrc1, so b - a
    if constexpr (is_16f_v<T>) {
      __half ha = npp16f_to_half(a);
      __half hb = npp16f_to_half(b);
      return half_to_npp16f(__hsub(hb, ha));
    } else if constexpr (is_complex_v<T>) {
      // Complex subtraction: (b.re + b.im*i) - (a.re + a.im*i)
      double re = static_cast<double>(b.re) - static_cast<double>(a.re);
      double im = static_cast<double>(b.im) - static_cast<double>(a.im);
      if (scaleFactor > 0) {
        double scale = 1.0 / (1 << scaleFactor);
        re = (re + (1 << (scaleFactor - 1))) * scale;
        im = (im + (1 << (scaleFactor - 1))) * scale;
      }
      return saturate_cast_complex<T>(re, im);
    } else {
      double result = static_cast<double>(b) - static_cast<double>(a);
      if (scaleFactor > 0 && std::is_integral_v<T>) {
        result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
      }
      return saturate_cast<T>(result);
    }
  }
};

template <typename T> struct MulOp {
  __device__ __host__ T operator()(T a, T b, int scaleFactor = 0) const {
    if constexpr (is_16f_v<T>) {
      __half ha = npp16f_to_half(a);
      __half hb = npp16f_to_half(b);
      return half_to_npp16f(__hmul(ha, hb));
    } else if constexpr (is_complex_v<T>) {
      // Complex multiplication: (a.re + a.im*i) * (b.re + b.im*i)
      // = (a.re*b.re - a.im*b.im) + (a.re*b.im + a.im*b.re)*i
      double re =
          static_cast<double>(a.re) * static_cast<double>(b.re) - static_cast<double>(a.im) * static_cast<double>(b.im);
      double im =
          static_cast<double>(a.re) * static_cast<double>(b.im) + static_cast<double>(a.im) * static_cast<double>(b.re);
      if (scaleFactor > 0) {
        double scale = 1.0 / (1 << scaleFactor);
        re = (re + (1 << (scaleFactor - 1))) * scale;
        im = (im + (1 << (scaleFactor - 1))) * scale;
      }
      return saturate_cast_complex<T>(re, im);
    } else {
      double result = static_cast<double>(a) * static_cast<double>(b);
      if (scaleFactor > 0 && std::is_integral_v<T>) {
        result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
      }
      return saturate_cast<T>(result);
    }
  }
};

template <typename T> struct DivOp {
  __device__ __host__ T operator()(T a, T b, int scaleFactor = 0) const {
    // NPP Div convention: pSrc2 / pSrc1, so b / a
    if constexpr (is_16f_v<T>) {
      __half ha = npp16f_to_half(a);
      __half hb = npp16f_to_half(b);
      return half_to_npp16f(__hdiv(hb, ha));
    } else if constexpr (is_complex_v<T>) {
      // Complex division: b / a = (b.re + b.im*i) / (a.re + a.im*i)
      // = ((b.re*a.re + b.im*a.im) + (b.im*a.re - b.re*a.im)*i) / (a.re² + a.im²)
      double denom =
          static_cast<double>(a.re) * static_cast<double>(a.re) + static_cast<double>(a.im) * static_cast<double>(a.im);
      if (denom == 0.0) {
        T result;
        result.re = 0;
        result.im = 0;
        return result;
      }
      double re = (static_cast<double>(b.re) * static_cast<double>(a.re) +
                   static_cast<double>(b.im) * static_cast<double>(a.im)) /
                  denom;
      double im = (static_cast<double>(b.im) * static_cast<double>(a.re) -
                   static_cast<double>(b.re) * static_cast<double>(a.im)) /
                  denom;
      if (scaleFactor > 0) {
        double scale = 1.0 / (1 << scaleFactor);
        re = (re + (1 << (scaleFactor - 1))) * scale;
        im = (im + (1 << (scaleFactor - 1))) * scale;
      }
      return saturate_cast_complex<T>(re, im);
    } else {
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
      double result = static_cast<double>(b) / static_cast<double>(a);
      if (scaleFactor > 0 && std::is_integral_v<T>) {
        result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
      }
      return saturate_cast<T>(result);
    }
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
    if constexpr (is_16f_v<T>) {
      __half ha = npp16f_to_half(a);
      __half hb = npp16f_to_half(b);
      __half diff = __hsub(ha, hb);
      // Use __habs if available, otherwise manual abs
      __half result = __habs(diff);
      return half_to_npp16f(result);
    } else {
      double result = std::abs(static_cast<double>(a) - static_cast<double>(b));
      if (scaleFactor > 0 && std::is_integral_v<T>) {
        result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
      }
      return saturate_cast<T>(result);
    }
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
  __device__ __host__ T operator()(T a, T = T{}, int = 0) const {
    if constexpr (is_16f_v<T>) {
      __half ha = npp16f_to_half(a);
      return half_to_npp16f(__habs(ha));
    } else {
      return std::abs(a);
    }
  }
};

template <typename T> struct SqrOp {
  __device__ __host__ T operator()(T a, T = T{}, int scaleFactor = 0) const {
    if constexpr (is_16f_v<T>) {
      __half ha = npp16f_to_half(a);
      return half_to_npp16f(__hmul(ha, ha));
    } else {
      double result = static_cast<double>(a) * static_cast<double>(a);
      if (scaleFactor > 0 && std::is_integral_v<T>) {
        result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
      }
      return saturate_cast<T>(result);
    }
  }
};

template <typename T> struct SqrtOp {
  __device__ __host__ T operator()(T a, T = T{}, int scaleFactor = 0) const {
    if constexpr (is_16f_v<T>) {
      __half ha = npp16f_to_half(a);
      return half_to_npp16f(hsqrt(ha));
    } else {
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
  __device__ __host__ T operator()(T a, T = T{}, int scaleFactor = 0) const {
    if constexpr (is_16f_v<T>) {
      __half ha = npp16f_to_half(a);
      return half_to_npp16f(hlog(ha));
    } else if constexpr (std::is_same_v<T, Npp32f>) {
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
// Multi-Channel Constant Shift Operation Functors
// ============================================================================
template <typename T> struct LeftShiftStrategy {
  __device__ __host__ static T apply(T value, Npp32u shiftCount) {
    if constexpr (std::is_same_v<T, Npp8u> || std::is_same_v<T, Npp8s>) {
      return (shiftCount < 8) ? (value << shiftCount) : 0;
    } else if constexpr (std::is_same_v<T, Npp16u> || std::is_same_v<T, Npp16s>) {
      return (shiftCount < 16) ? (value << shiftCount) : 0;
    } else if constexpr (std::is_same_v<T, Npp32u> || std::is_same_v<T, Npp32s>) {
      return (shiftCount < 32) ? (value << shiftCount) : 0;
    } else {
      return value;
    }
  }
};

template <typename T> struct RightShiftStrategy {
  __device__ __host__ static T apply(T value, Npp32u shiftCount) {
    if constexpr (std::is_same_v<T, Npp8u> || std::is_same_v<T, Npp8s>) {
      return (shiftCount < 8) ? (value >> shiftCount) : 0;
    } else if constexpr (std::is_same_v<T, Npp16u> || std::is_same_v<T, Npp16s>) {
      return (shiftCount < 16) ? (value >> shiftCount) : 0;
    } else if constexpr (std::is_same_v<T, Npp32u> || std::is_same_v<T, Npp32s>) {
      return (shiftCount < 32) ? (value >> shiftCount) : 0;
    } else {
      return value;
    }
  }
};

template <typename T, int Channels, template <typename> class ShiftStrategy> class ShiftConstMultiOp {
private:
  Npp32u constants[Channels];

public:
  __device__ __host__ ShiftConstMultiOp(const Npp32u aConstants[Channels]) {
    for (int i = 0; i < Channels; i++) {
      constants[i] = aConstants[i];
    }
  }

  __device__ __host__ T operator()(T a, T = T(0), int = 0) const { return a; }

  __device__ __host__ Npp32u getConstant(int channel) const {
    if (channel < Channels) {
      return constants[channel];
    }
    return 0;
  }

  __device__ __host__ T applyShift(T value, int channel) const {
    return ShiftStrategy<T>::apply(value, getConstant(channel));
  }
};

template <typename T, int Channels> using LShiftConstMultiOp = ShiftConstMultiOp<T, Channels, LeftShiftStrategy>;

template <typename T, int Channels> using RShiftConstMultiOp = ShiftConstMultiOp<T, Channels, RightShiftStrategy>;

// ============================================================================
// Constant Operation Functors (for operations with constants)
// ============================================================================

template <typename T> class AddConstOp {
private:
  T constant;

public:
  __device__ __host__ AddConstOp(T c) : constant(c) {}

  __device__ __host__ T operator()(T a, T = T{}, int scaleFactor = 0) const {
    if constexpr (is_complex_v<T>) {
      double re = static_cast<double>(a.re) + static_cast<double>(constant.re);
      double im = static_cast<double>(a.im) + static_cast<double>(constant.im);
      if (scaleFactor > 0) {
        double scale = 1.0 / (1 << scaleFactor);
        re = (re + (1 << (scaleFactor - 1))) * scale;
        im = (im + (1 << (scaleFactor - 1))) * scale;
      }
      return saturate_cast_complex<T>(re, im);
    } else {
      double result = static_cast<double>(a) + static_cast<double>(constant);
      if (scaleFactor > 0 && std::is_integral_v<T>) {
        result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
      }
      return saturate_cast<T>(result);
    }
  }
};

template <typename T> class SubConstOp {
private:
  T constant;

public:
  __device__ __host__ SubConstOp(T c) : constant(c) {}

  __device__ __host__ T operator()(T a, T = T{}, int scaleFactor = 0) const {
    if constexpr (is_complex_v<T>) {
      double re = static_cast<double>(a.re) - static_cast<double>(constant.re);
      double im = static_cast<double>(a.im) - static_cast<double>(constant.im);
      if (scaleFactor > 0) {
        double scale = 1.0 / (1 << scaleFactor);
        re = (re + (1 << (scaleFactor - 1))) * scale;
        im = (im + (1 << (scaleFactor - 1))) * scale;
      }
      return saturate_cast_complex<T>(re, im);
    } else {
      double result = static_cast<double>(a) - static_cast<double>(constant);
      if (scaleFactor > 0 && std::is_integral_v<T>) {
        result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
      }
      return saturate_cast<T>(result);
    }
  }
};

template <typename T> class MulConstOp {
private:
  T constant;

public:
  __device__ __host__ MulConstOp(T c) : constant(c) {}

  __device__ __host__ T operator()(T a, T = T{}, int scaleFactor = 0) const {
    if constexpr (is_complex_v<T>) {
      // Complex multiplication: (a.re + a.im*i) * (c.re + c.im*i)
      // = (a.re*c.re - a.im*c.im) + (a.re*c.im + a.im*c.re)*i
      double re = static_cast<double>(a.re) * static_cast<double>(constant.re) -
                  static_cast<double>(a.im) * static_cast<double>(constant.im);
      double im = static_cast<double>(a.re) * static_cast<double>(constant.im) +
                  static_cast<double>(a.im) * static_cast<double>(constant.re);
      if (scaleFactor > 0) {
        double scale = 1.0 / (1 << scaleFactor);
        re = (re + (1 << (scaleFactor - 1))) * scale;
        im = (im + (1 << (scaleFactor - 1))) * scale;
      }
      return saturate_cast_complex<T>(re, im);
    } else {
      double result = static_cast<double>(a) * static_cast<double>(constant);
      if (scaleFactor > 0 && std::is_integral_v<T>) {
        result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
      }
      return saturate_cast<T>(result);
    }
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

// MulCScaleOp: multiply by constant with scale (result = (src * constant) / max_value)
template <typename T> class MulCScaleOp {
private:
  T constant;

public:
  __device__ __host__ MulCScaleOp(T c) : constant(c) {}

  __device__ __host__ T operator()(T a, T = T(0), int = 0) const {
    if constexpr (std::is_same_v<T, Npp8u>) {
      // For 8-bit: result = (a * constant) / 255
      double result = (static_cast<double>(a) * static_cast<double>(constant)) / 255.0;
      return saturate_cast<T>(result);
    } else if constexpr (std::is_same_v<T, Npp16u>) {
      // For 16-bit: result = (a * constant) / 65535
      double result = (static_cast<double>(a) * static_cast<double>(constant)) / 65535.0;
      return saturate_cast<T>(result);
    } else {
      // For other types, fallback to regular multiplication
      return saturate_cast<T>(static_cast<double>(a) * static_cast<double>(constant));
    }
  }
};

// DivRoundOp: division with rounding mode support
template <typename T> struct DivRoundOp {
  __device__ __host__ T operator()(T a, T b, int scaleFactor = 0, NppRoundMode rndMode = NPP_RND_NEAR) const {
    // NPP Div convention: pSrc2 / pSrc1, so b / a
    if (a == T(0)) {
      // Division by zero - saturate to maximum value for integers
      if constexpr (std::is_same_v<T, Npp8u>)
        return T(255);
      else if constexpr (std::is_same_v<T, Npp16u>)
        return T(65535);
      else if constexpr (std::is_same_v<T, Npp16s>)
        return (b >= T(0)) ? T(32767) : T(-32768);
      else
        return T(0);
    }

    double result = static_cast<double>(b) / static_cast<double>(a);

    // Apply scale factor first
    if (scaleFactor != 0) {
      if (scaleFactor > 0) {
        result = result / static_cast<double>(1 << scaleFactor);
      } else {
        result = result * static_cast<double>(1 << (-scaleFactor));
      }
    }

    // Apply rounding mode
    // Note: NPP_RND_NEAR == NPP_ROUND_NEAREST_TIES_TO_EVEN (aliases)
    // Note: NPP_RND_FINANCIAL == NPP_ROUND_NEAREST_TIES_AWAY_FROM_ZERO (aliases)
    // Note: NPP_RND_ZERO == NPP_ROUND_TOWARD_ZERO (aliases)
    switch (rndMode) {
    case NPP_RND_NEAR: {
      // Round to nearest even (banker's rounding)
      double floor_val = std::floor(result);
      double frac = result - floor_val;
      if (frac > 0.5) {
        result = floor_val + 1.0;
      } else if (frac < 0.5) {
        result = floor_val;
      } else {
        // Exactly 0.5 - round to nearest even
        int floor_int = static_cast<int>(floor_val);
        result = (floor_int % 2 == 0) ? floor_val : floor_val + 1.0;
      }
      break;
    }
    case NPP_RND_FINANCIAL:
      // Round away from zero at 0.5
      if (result >= 0) {
        result = std::floor(result + 0.5);
      } else {
        result = std::ceil(result - 0.5);
      }
      break;
    case NPP_RND_ZERO:
    default:
      // Truncate toward zero
      if (result >= 0) {
        result = std::floor(result);
      } else {
        result = std::ceil(result);
      }
      break;
    }

    return saturate_cast<T>(result);
  }
};

// DivConstOp: division by constant with proper zero handling
template <typename T> class DivConstOp {
private:
  T constant;

public:
  __device__ __host__ DivConstOp(T c) : constant(c) {}

  __device__ __host__ T operator()(T a, T = T{}, int scaleFactor = 0) const {
    if constexpr (is_complex_v<T>) {
      // Complex division: (a.re + a.im*i) / (c.re + c.im*i)
      // = [(a.re*c.re + a.im*c.im) + (a.im*c.re - a.re*c.im)*i] / (c.re^2 + c.im^2)
      double cre = static_cast<double>(constant.re);
      double cim = static_cast<double>(constant.im);
      double denom = cre * cre + cim * cim;
      if (denom == 0.0) {
        T result;
        result.re = 0;
        result.im = 0;
        return result;
      }
      double are = static_cast<double>(a.re);
      double aim = static_cast<double>(a.im);
      double re = (are * cre + aim * cim) / denom;
      double im = (aim * cre - are * cim) / denom;
      if (scaleFactor > 0) {
        double scale = 1.0 / (1 << scaleFactor);
        re = (re + (1 << (scaleFactor - 1))) * scale;
        im = (im + (1 << (scaleFactor - 1))) * scale;
      }
      return saturate_cast_complex<T>(re, im);
    } else {
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
  }
};

// ============================================================================
// 16f Constant Operation Functors
// Note: For 16f operations, constant is Npp32f type
// ============================================================================

class AddConst16fOp {
private:
  Npp32f constant_f32; // Store as float, convert to half on device

public:
  __device__ __host__ AddConst16fOp(Npp32f c) : constant_f32(c) {}

  __device__ Npp16f operator()(Npp16f a, Npp16f = Npp16f{}, int = 0) const {
    __half ha = npp16f_to_half(a);
    __half hc = __float2half(constant_f32);
    return half_to_npp16f(__hadd(ha, hc));
  }
};

class SubConst16fOp {
private:
  Npp32f constant_f32;

public:
  __device__ __host__ SubConst16fOp(Npp32f c) : constant_f32(c) {}

  __device__ Npp16f operator()(Npp16f a, Npp16f = Npp16f{}, int = 0) const {
    __half ha = npp16f_to_half(a);
    __half hc = __float2half(constant_f32);
    return half_to_npp16f(__hsub(ha, hc));
  }
};

class MulConst16fOp {
private:
  Npp32f constant_f32;

public:
  __device__ __host__ MulConst16fOp(Npp32f c) : constant_f32(c) {}

  __device__ Npp16f operator()(Npp16f a, Npp16f = Npp16f{}, int = 0) const {
    __half ha = npp16f_to_half(a);
    __half hc = __float2half(constant_f32);
    return half_to_npp16f(__hmul(ha, hc));
  }
};

class DivConst16fOp {
private:
  Npp32f constant_f32;

public:
  __device__ __host__ DivConst16fOp(Npp32f c) : constant_f32(c) {}

  __device__ Npp16f operator()(Npp16f a, Npp16f = Npp16f{}, int = 0) const {
    __half ha = npp16f_to_half(a);
    __half hc = __float2half(constant_f32);
    return half_to_npp16f(__hdiv(ha, hc));
  }
};

// 16f DeviceC operation functors (constant is on device)
class AddDeviceConst16fOp {
private:
  const Npp32f *pConstant;

public:
  __device__ __host__ AddDeviceConst16fOp(const Npp32f *c) : pConstant(c) {}

  __device__ Npp16f operator()(Npp16f a, Npp16f = Npp16f{}, int = 0) const {
    __half ha = npp16f_to_half(a);
    __half c = __float2half(*pConstant);
    return half_to_npp16f(__hadd(ha, c));
  }
};

class SubDeviceConst16fOp {
private:
  const Npp32f *pConstant;

public:
  __device__ __host__ SubDeviceConst16fOp(const Npp32f *c) : pConstant(c) {}

  __device__ Npp16f operator()(Npp16f a, Npp16f = Npp16f{}, int = 0) const {
    __half ha = npp16f_to_half(a);
    __half c = __float2half(*pConstant);
    return half_to_npp16f(__hsub(ha, c));
  }
};

class MulDeviceConst16fOp {
private:
  const Npp32f *pConstant;

public:
  __device__ __host__ MulDeviceConst16fOp(const Npp32f *c) : pConstant(c) {}

  __device__ Npp16f operator()(Npp16f a, Npp16f = Npp16f{}, int = 0) const {
    __half ha = npp16f_to_half(a);
    __half c = __float2half(*pConstant);
    return half_to_npp16f(__hmul(ha, c));
  }
};

class DivDeviceConst16fOp {
private:
  const Npp32f *pConstant;

public:
  __device__ __host__ DivDeviceConst16fOp(const Npp32f *c) : pConstant(c) {}

  __device__ Npp16f operator()(Npp16f a, Npp16f = Npp16f{}, int = 0) const {
    __half ha = npp16f_to_half(a);
    __half c = __float2half(*pConstant);
    return half_to_npp16f(__hdiv(ha, c));
  }
};

// 16f AddProduct operation: dst = dst + (src1 * src2)
struct AddProduct16fOp {
  __device__ Npp16f operator()(Npp16f dst, Npp16f src1, Npp16f src2, int = 0) const {
    __half hd = npp16f_to_half(dst);
    __half ha = npp16f_to_half(src1);
    __half hb = npp16f_to_half(src2);
    return half_to_npp16f(__hadd(hd, __hmul(ha, hb)));
  }
};

} // namespace arithmetic
} // namespace nppi