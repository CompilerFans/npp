#pragma once

#include "npp.h"
#include "npp_version_compat.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

// Helper functions for Npp16f (half-precision float) type
inline unsigned short npp_half_as_ushort_host(__half h) {
    return static_cast<unsigned short>(static_cast<__half_raw>(h).x);
  }
  
  inline __half npp_ushort_as_half_host(unsigned short v) {
    __half_raw raw{};
    raw.x = v;
    return __half(raw);
  }
  
  inline Npp16f float_to_npp16f_host(float val) {
    Npp16f result;
    __half h = __float2half(val);
    result.fp16 = static_cast<short>(npp_half_as_ushort_host(h));
    return result;
  }
  
  inline float npp16f_to_float_host(Npp16f val) {
    __half h = npp_ushort_as_half_host(static_cast<unsigned short>(val.fp16));
    return __half2float(h);
  }
  
  // Type trait to check if T is Npp16f
  template <typename T> struct is_npp16f : std::false_type {};
  template <> struct is_npp16f<Npp16f> : std::true_type {};
  template <typename T> constexpr bool is_npp16f_v = is_npp16f<T>::value;
  
  #if CUDA_SDK_AT_LEAST(12, 8)
  #define SIZE_TYPE size_t
  #else
  #define SIZE_TYPE int
  #endif