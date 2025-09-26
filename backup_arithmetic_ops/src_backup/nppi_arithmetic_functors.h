#ifndef NPPI_ARITHMETIC_FUNCTORS_H
#define NPPI_ARITHMETIC_FUNCTORS_H

#include "npp.h"
#include <cuda_runtime.h>
#include <type_traits>
#include <cmath>
#include <functional> // For standard functors

namespace nppi {
namespace functors {

// Generic saturation utilities
template<typename T>
__device__ __host__ T saturate_cast(double value);

// Specializations for common types
template<>
__device__ __host__ inline Npp8u saturate_cast<Npp8u>(double value) {
    return static_cast<Npp8u>(value < 0.0 ? 0.0 : (value > 255.0 ? 255.0 : value));
}

template<>
__device__ __host__ inline Npp8s saturate_cast<Npp8s>(double value) {
    return static_cast<Npp8s>(value < -128.0 ? -128.0 : (value > 127.0 ? 127.0 : value));
}

template<>
__device__ __host__ inline Npp16u saturate_cast<Npp16u>(double value) {
    return static_cast<Npp16u>(value < 0.0 ? 0.0 : (value > 65535.0 ? 65535.0 : value));
}

template<>
__device__ __host__ inline Npp16s saturate_cast<Npp16s>(double value) {
    return static_cast<Npp16s>(value < -32768.0 ? -32768.0 : (value > 32767.0 ? 32767.0 : value));
}

template<>
__device__ __host__ inline Npp32f saturate_cast<Npp32f>(double value) {
    return static_cast<Npp32f>(value);
}

template<>
__device__ __host__ inline Npp32s saturate_cast<Npp32s>(double value) {
    return static_cast<Npp32s>(value < -2147483648.0 ? -2147483648.0 : 
                                (value > 2147483647.0 ? 2147483647.0 : value));
}

// Generic functor wrapper that adds scaling and saturation to standard functors
template<typename T, typename StdFunctor>
struct ScaledFunctor {
    __device__ __host__ T operator()(T a, T b, int scaleFactor = 0) const {
        StdFunctor op;
        double result = static_cast<double>(op(static_cast<double>(a), static_cast<double>(b)));
        
        if (scaleFactor > 0 && std::is_integral_v<T>) {
            result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
        }
        return saturate_cast<T>(result);
    }
};

// Binary operation functors using standard library functors
template<typename T>
using AddFunctor = ScaledFunctor<T, std::plus<double>>;

template<typename T>
using SubFunctor = ScaledFunctor<T, std::minus<double>>;

template<typename T>
using MulFunctor = ScaledFunctor<T, std::multiplies<double>>;

// Special division functor with zero-division protection
template<typename T>
struct DivFunctor {
    __device__ __host__ T operator()(T a, T b, int scaleFactor = 0) const {
        if (b == T(0)) return T(0); // Handle division by zero
        double result = static_cast<double>(a) / static_cast<double>(b);
        if (scaleFactor > 0 && std::is_integral_v<T>) {
            result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
        }
        return saturate_cast<T>(result);
    }
};

template<typename T>
struct AbsDiffFunctor {
    __device__ __host__ T operator()(T a, T b, int scaleFactor = 0) const {
        double result = fabs(static_cast<double>(a) - static_cast<double>(b));
        if (scaleFactor > 0 && std::is_integral_v<T>) {
            result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
        }
        return saturate_cast<T>(result);
    }
};

// Logical operation functors using standard library functors
template<typename T>
struct AndFunctor {
    __device__ __host__ T operator()(T a, T b, int = 0) const {
        static_assert(std::is_integral_v<T>, "Bitwise AND only supported for integral types");
        std::bit_and<T> op;
        return op(a, b);
    }
};

template<typename T>
struct OrFunctor {
    __device__ __host__ T operator()(T a, T b, int = 0) const {
        static_assert(std::is_integral_v<T>, "Bitwise OR only supported for integral types");
        std::bit_or<T> op;
        return op(a, b);
    }
};

template<typename T>
struct XorFunctor {
    __device__ __host__ T operator()(T a, T b, int = 0) const {
        static_assert(std::is_integral_v<T>, "Bitwise XOR only supported for integral types");
        std::bit_xor<T> op;
        return op(a, b);
    }
};

// Generic constant operation functor using standard library functors
template<typename T, typename BinaryFunctor>
struct ConstFunctor {
    T constant;
    __device__ __host__ ConstFunctor(T c) : constant(c) {}
    __device__ __host__ T operator()(T a, int scaleFactor = 0) const {
        BinaryFunctor binOp;
        double result = static_cast<double>(binOp(static_cast<double>(a), static_cast<double>(constant)));
        if (scaleFactor > 0 && std::is_integral_v<T>) {
            result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
        }
        return saturate_cast<T>(result);
    }
};

// Type aliases for common constant operations using standard functors
template<typename T>
using AddConstFunctor = ConstFunctor<T, std::plus<double>>;

template<typename T>
using SubConstFunctor = ConstFunctor<T, std::minus<double>>;

template<typename T>
using MulConstFunctor = ConstFunctor<T, std::multiplies<double>>;

// Special division constant functor with zero-division protection
template<typename T>
struct DivConstFunctor {
    T constant;
    __device__ __host__ DivConstFunctor(T c) : constant(c) {}
    __device__ __host__ T operator()(T a, int scaleFactor = 0) const {
        if (constant == T(0)) return T(0);
        double result = static_cast<double>(a) / static_cast<double>(constant);
        if (scaleFactor > 0 && std::is_integral_v<T>) {
            result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
        }
        return saturate_cast<T>(result);
    }
};

// Generic unary operation functor wrapper
template<typename T, typename UnaryMathFunc>
struct UnaryMathFunctor {
    __device__ __host__ T operator()(T a, int scaleFactor = 0) const {
        UnaryMathFunc func;
        double result = func(static_cast<double>(a));
        if (scaleFactor > 0 && std::is_integral_v<T>) {
            result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
        }
        return saturate_cast<T>(result);
    }
};

// Unary operation functors
template<typename T>
struct AbsFunctor {
    __device__ __host__ T operator()(T a, int = 0) const {
        if constexpr (std::is_unsigned_v<T>) {
            return a;
        } else {
            // Use standard library abs for consistency
            return saturate_cast<T>(std::abs(static_cast<double>(a)));
        }
    }
};

// Math function wrappers
struct SquareFunctor {
    __device__ __host__ double operator()(double x) const { return x * x; }
};

struct SqrtFunctor_ {
    __device__ __host__ double operator()(double x) const { return sqrt(x); }
};

struct ExpFunctor_ {
    __device__ __host__ double operator()(double x) const { return exp(x); }
};

struct LogFunctor_ {
    __device__ __host__ double operator()(double x) const { 
        return (x > 0) ? log(x) : 0.0; // Handle invalid input 
    }
};

// Type aliases using the generic unary functor wrapper
template<typename T>
using SqrFunctor = UnaryMathFunctor<T, SquareFunctor>;

template<typename T>
using SqrtFunctor = UnaryMathFunctor<T, SqrtFunctor_>;

template<typename T>
using ExpFunctor = UnaryMathFunctor<T, ExpFunctor_>;

template<typename T>
using LnFunctor = UnaryMathFunctor<T, LogFunctor_>;

} // namespace functors
} // namespace nppi

#endif // NPPI_ARITHMETIC_FUNCTORS_H