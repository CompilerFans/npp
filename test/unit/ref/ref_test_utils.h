#pragma once

// Helpers for the generated reference test suite (tools/gen_ref_tests.py).

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <cuda_runtime.h>

#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

namespace ref {

using npp_functional_test::DeviceMemory;

inline size_t align16(size_t n) { return (n + 15) & ~size_t(15); }

inline NppStreamContext ref_current_context() {
  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  return ctx;
}

template <typename T> constexpr npp_diff::ValueKind kind_for() {
  using K = npp_diff::ValueKind;
  if constexpr (std::is_same_v<T, Npp8u>) return K::U8;
  else if constexpr (std::is_same_v<T, Npp8s>) return K::I8;
  else if constexpr (std::is_same_v<T, Npp16u>) return K::U16;
  else if constexpr (std::is_same_v<T, Npp16s>) return K::I16;
  else if constexpr (std::is_same_v<T, Npp16f>) return K::F16;
  else if constexpr (std::is_same_v<T, Npp32u>) return K::U32;
  else if constexpr (std::is_same_v<T, Npp32s>) return K::I32;
  else if constexpr (std::is_same_v<T, Npp32f>) return K::F32;
  else if constexpr (std::is_same_v<T, Npp64u>) return K::U64;
  else if constexpr (std::is_same_v<T, Npp64s>) return K::I64;
  else if constexpr (std::is_same_v<T, Npp64f>) return K::F64;
  else if constexpr (std::is_same_v<T, Npp32fc>) return K::CF32;
  else if constexpr (std::is_same_v<T, Npp64fc>) return K::CF64;
  else if constexpr (std::is_same_v<T, Npp16sc>) return K::CF32;
  else if constexpr (std::is_same_v<T, Npp32sc>) return K::CF32;
  else if constexpr (std::is_same_v<T, Npp8uc>) return K::CF32;
  else if constexpr (std::is_same_v<T, Npp16uc>) return K::CF32;
  else if constexpr (std::is_same_v<T, Npp32uc>) return K::CF32;
  else return K::Bytes;
}

template <typename T> T ref_value(npp_diff::Rng &rng) {
  if constexpr (std::is_same_v<T, Npp16f>) {
    return npp_diff::to_npp16f(rng.unit_float() * 2.0f - 1.0f);
  } else if constexpr (std::is_same_v<T, Npp32fc>) {
    Npp32fc v;
    v.re = rng.unit_float() * 2.0f - 1.0f;
    v.im = rng.unit_float() * 2.0f - 1.0f;
    return v;
  } else if constexpr (std::is_same_v<T, Npp64fc>) {
    Npp64fc v;
    v.re = rng.unit_double() * 2.0 - 1.0;
    v.im = rng.unit_double() * 2.0 - 1.0;
    return v;
  } else if constexpr (std::is_same_v<T, Npp16sc> || std::is_same_v<T, Npp32sc> ||
                       std::is_same_v<T, Npp8uc> ||
                       std::is_same_v<T, Npp16uc> || std::is_same_v<T, Npp32uc>) {
    T v;
    v.re = static_cast<decltype(v.re)>(rng.next() % 16);
    v.im = static_cast<decltype(v.im)>(rng.next() % 16);
    return v;
  } else if constexpr (std::is_floating_point_v<T>) {
    return static_cast<T>(rng.unit_double() * 2.0 - 1.0);
  } else if constexpr (std::is_integral_v<T>) {
    return static_cast<T>(rng.next() % 16);
  } else {
    return T{};
  }
}

template <typename T>
std::unique_ptr<DeviceMemory<T>> ref_device(size_t count, uint64_t seed, bool fill = true) {
  auto mem = std::make_unique<DeviceMemory<T>>(count);
  if (fill) {
    std::vector<T> host(count);
    npp_diff::Rng rng(seed);
    for (auto &v : host) v = ref_value<T>(rng);
    mem->copyFromHost(host);
  } else {
    cudaMemset(mem->get(), 0xCD, count * sizeof(T));
  }
  return mem;
}

template <typename T> void ref_capture(npp_diff::Case &c, const T *ptr, size_t count) {
  c.device_output<T>(ptr, count, kind_for<T>());
}

inline void ref_capture_bytes(npp_diff::Case &c, const void *ptr, size_t bytes) {
  c.device_output<uint8_t>(reinterpret_cast<const uint8_t *>(ptr), bytes, npp_diff::ValueKind::Bytes);
}

// Batch descriptor structs embed device pointers, which are cudaMalloc
// addresses and differ between runs; a raw capture of a struct list would
// flake in verify.  ref_capture_batch_structs() copies the list back to
// host, nulls the pointer fields, and captures the deterministic remainder
// (steps, sizes and the transformed-coefficient block that the
// nppi*BatchInit entry points write inside each struct).  Every captured
// struct type must provide a RefBatchPointerFields specialization that
// nulls its pointer fields; the undefined primary template makes an
// unspecialized capture a compile error instead of a silent pointer leak.
template <typename T> struct RefBatchPointerFields;

template <> struct RefBatchPointerFields<NppiWarpAffineBatchCXR> {
  static void mask(NppiWarpAffineBatchCXR &s) {
    s.pSrc = nullptr;
    s.pDst = nullptr;
    s.pCoeffs = nullptr;
  }
};

template <> struct RefBatchPointerFields<NppiWarpPerspectiveBatchCXR> {
  static void mask(NppiWarpPerspectiveBatchCXR &s) {
    s.pSrc = nullptr;
    s.pDst = nullptr;
    s.pCoeffs = nullptr;
  }
};

template <typename T>
void ref_capture_batch_structs(npp_diff::Case &c, const T *dev, size_t count) {
  std::vector<T> host(count);
  if (count > 0 && dev != nullptr) {
    cudaDeviceSynchronize();
    cudaError_t err = cudaMemcpy(host.data(), dev, count * sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      ADD_FAILURE() << "cudaMemcpy(D2H) failed for batch structs: " << cudaGetErrorString(err);
    }
  }
  for (auto &s : host) RefBatchPointerFields<T>::mask(s);
  c.host_output<uint8_t>(reinterpret_cast<const uint8_t *>(host.data()),
                         host.size() * sizeof(T), npp_diff::ValueKind::Bytes);
}

} // namespace ref
