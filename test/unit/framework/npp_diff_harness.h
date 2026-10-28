#pragma once

// Differential test harness for NVIDIA NPP vs mcMPP.
//
// Record mode (NVIDIA NPP reference, normally on the RTX 5090):
//   NPP_RECORD_DIR=/path ./unit_tests --gtest_filter='*Generated*'
//
// Verify mode (mcMPP build):
//   NPP_VERIFY_DIR=/path ./unit_tests --gtest_filter='*Generated*'
//
// The harness stores the semantic NppStatus name (enum numeric values differ
// between NVIDIA NPP 13 and the mcMPP headers) plus the raw output buffers.
// Integer outputs are compared exactly; floating outputs use an explicit
// per-output tolerance.  Inputs are regenerated deterministically, so only a
// hash is stored for debugging.
//
// NPP 13 removed the non-Ctx entry points.  Generated tests therefore call
// `_Ctx` when `USE_NVIDIA_NPP_TESTS` is defined (reference build) and exercise
// both `_Ctx` and the plain entry point in the mcMPP build against the same
// golden case.

#include "npp.h"
#include "npp_version_compat.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#ifdef USE_NVIDIA_NPP_TESTS
#define NPP_DIFF_HAS_NONCTX 0
#else
#define NPP_DIFF_HAS_NONCTX 1
#endif

namespace npp_diff {

enum class ValueKind {
  Bytes,
  U8,
  I8,
  U16,
  I16,
  U32,
  I32,
  U64,
  I64,
  I16Tol,
  I32Tol,
  I64Tol,
  F16,
  F32,
  F64,
  CF32,
  CF64,
};

inline size_t element_size(ValueKind kind) {
  switch (kind) {
  case ValueKind::Bytes:
  case ValueKind::U8:
  case ValueKind::I8:
    return 1;
  case ValueKind::U16:
  case ValueKind::I16:
  case ValueKind::I16Tol:
  case ValueKind::F16:
    return 2;
  case ValueKind::U32:
  case ValueKind::I32:
  case ValueKind::I32Tol:
  case ValueKind::F32:
    return 4;
  case ValueKind::U64:
  case ValueKind::I64:
  case ValueKind::I64Tol:
  case ValueKind::F64:
  case ValueKind::CF32:
    return 8;
  case ValueKind::CF64:
    return 16;
  }
  return 1;
}

inline const char *kind_name(ValueKind kind) {
  switch (kind) {
  case ValueKind::Bytes: return "bytes";
  case ValueKind::U8: return "u8";
  case ValueKind::I8: return "i8";
  case ValueKind::U16: return "u16";
  case ValueKind::I16: return "i16";
  case ValueKind::U32: return "u32";
  case ValueKind::I32: return "i32";
  case ValueKind::U64: return "u64";
  case ValueKind::I64: return "i64";
  case ValueKind::I16Tol: return "i16tol";
  case ValueKind::I32Tol: return "i32tol";
  case ValueKind::I64Tol: return "i64tol";
  case ValueKind::F16: return "f16";
  case ValueKind::F32: return "f32";
  case ValueKind::F64: return "f64";
  case ValueKind::CF32: return "cf32";
  case ValueKind::CF64: return "cf64";
  }
  return "bytes";
}

inline ValueKind kind_from_name(const std::string &name) {
  if (name == "u8") return ValueKind::U8;
  if (name == "i8") return ValueKind::I8;
  if (name == "u16") return ValueKind::U16;
  if (name == "i16") return ValueKind::I16;
  if (name == "u32") return ValueKind::U32;
  if (name == "i32") return ValueKind::I32;
  if (name == "u64") return ValueKind::U64;
  if (name == "i64") return ValueKind::I64;
  if (name == "i16tol") return ValueKind::I16Tol;
  if (name == "i32tol") return ValueKind::I32Tol;
  if (name == "i64tol") return ValueKind::I64Tol;
  if (name == "f16") return ValueKind::F16;
  if (name == "f32") return ValueKind::F32;
  if (name == "f64") return ValueKind::F64;
  if (name == "cf32") return ValueKind::CF32;
  if (name == "cf64") return ValueKind::CF64;
  return ValueKind::Bytes;
}

// Status names are the comparison key: NPP 13 and mcMPP header enum values
// are not guaranteed to agree numerically.  Every name below exists in both
// API-unified/nppdefs.h and NVIDIA NPP 13.4 (the reference only adds two new
// names, which fall through to the numeric fallback).
inline std::string status_name(NppStatus status) {
  if (status == NPP_NO_ERROR) return "NPP_NO_ERROR";
  if (status == NPP_ERROR) return "NPP_ERROR";
  if (status == NPP_ERROR_RESERVED) return "NPP_ERROR_RESERVED";
  if (status == NPP_NOT_IMPLEMENTED_ERROR) return "NPP_NOT_IMPLEMENTED_ERROR";
  if (status == NPP_NO_MEMORY_ERROR) return "NPP_NO_MEMORY_ERROR";
  if (status == NPP_BAD_ARGUMENT_ERROR) return "NPP_BAD_ARGUMENT_ERROR";
  if (status == NPP_SIZE_ERROR) return "NPP_SIZE_ERROR";
  if (status == NPP_RANGE_ERROR) return "NPP_RANGE_ERROR";
  if (status == NPP_NULL_POINTER_ERROR) return "NPP_NULL_POINTER_ERROR";
  if (status == NPP_MEMORY_ALLOCATION_ERR) return "NPP_MEMORY_ALLOCATION_ERR";
  if (status == NPP_DIVIDE_BY_ZERO_ERROR) return "NPP_DIVIDE_BY_ZERO_ERROR";
  if (status == NPP_OUT_OFF_RANGE_ERROR) return "NPP_OUT_OFF_RANGE_ERROR";
  if (status == NPP_DATA_TYPE_ERROR) return "NPP_DATA_TYPE_ERROR";
  if (status == NPP_SCALE_RANGE_ERROR) return "NPP_SCALE_RANGE_ERROR";
  if (status == NPP_STEP_ERROR) return "NPP_STEP_ERROR";
  if (status == NPP_FFT_ORDER_ERROR) return "NPP_FFT_ORDER_ERROR";
  if (status == NPP_FFT_FLAG_ERROR) return "NPP_FFT_FLAG_ERROR";
  if (status == NPP_CONTEXT_MATCH_ERROR) return "NPP_CONTEXT_MATCH_ERROR";
  if (status == NPP_THRESHOLD_ERROR) return "NPP_THRESHOLD_ERROR";
  if (status == NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR) return "NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR";
  if (status == NPP_MOMENT_00_ZERO_ERROR) return "NPP_MOMENT_00_ZERO_ERROR";
  if (status == NPP_MIRROR_FLIP_ERROR) return "NPP_MIRROR_FLIP_ERROR";
  if (status == NPP_INTERPOLATION_ERROR) return "NPP_INTERPOLATION_ERROR";
  if (status == NPP_RESIZE_FACTOR_ERROR) return "NPP_RESIZE_FACTOR_ERROR";
  if (status == NPP_MASK_SIZE_ERROR) return "NPP_MASK_SIZE_ERROR";
  if (status == NPP_ANCHOR_ERROR) return "NPP_ANCHOR_ERROR";
  if (status == NPP_STRIDE_ERROR) return "NPP_STRIDE_ERROR";
  if (status == NPP_CHANNEL_ERROR) return "NPP_CHANNEL_ERROR";
  if (status == NPP_DIVISOR_ERROR) return "NPP_DIVISOR_ERROR";
  if (status == NPP_COI_ERROR) return "NPP_COI_ERROR";
  if (status == NPP_NUMBER_OF_CHANNELS_ERROR) return "NPP_NUMBER_OF_CHANNELS_ERROR";
  if (status == NPP_COEFFICIENT_ERROR) return "NPP_COEFFICIENT_ERROR";
  if (status == NPP_RECTANGLE_ERROR) return "NPP_RECTANGLE_ERROR";
  if (status == NPP_QUADRANGLE_ERROR) return "NPP_QUADRANGLE_ERROR";
  if (status == NPP_ZERO_MASK_VALUE_ERROR) return "NPP_ZERO_MASK_VALUE_ERROR";
  if (status == NPP_CHANNEL_ORDER_ERROR) return "NPP_CHANNEL_ORDER_ERROR";
  if (status == NPP_CORRUPTED_DATA_ERROR) return "NPP_CORRUPTED_DATA_ERROR";
  if (status == NPP_LUT_NUMBER_OF_LEVELS_ERROR) return "NPP_LUT_NUMBER_OF_LEVELS_ERROR";
  if (status == NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR) return "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR";
  if (status == NPP_NOT_EVEN_STEP_ERROR) return "NPP_NOT_EVEN_STEP_ERROR";
  if (status == NPP_OVERFLOW_ERROR) return "NPP_OVERFLOW_ERROR";
  if (status == NPP_RESIZE_NO_OPERATION_ERROR) return "NPP_RESIZE_NO_OPERATION_ERROR";
  if (status == NPP_QUALITY_INDEX_ERROR) return "NPP_QUALITY_INDEX_ERROR";
  if (status == NPP_ROUND_MODE_NOT_SUPPORTED_ERROR) return "NPP_ROUND_MODE_NOT_SUPPORTED_ERROR";
  if (status == NPP_CUDA_KERNEL_EXECUTION_ERROR) return "NPP_CUDA_KERNEL_EXECUTION_ERROR";
  if (status == NPP_ALIGNMENT_ERROR) return "NPP_ALIGNMENT_ERROR";
  if (status == NPP_MEMCPY_ERROR) return "NPP_MEMCPY_ERROR";
  if (status == NPP_MEMSET_ERROR) return "NPP_MEMSET_ERROR";
  if (status == NPP_MEMFREE_ERROR) return "NPP_MEMFREE_ERROR";
  if (status == NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR) return "NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR";
  if (status == NPP_WRONG_INTERSECTION_ROI_ERROR) return "NPP_WRONG_INTERSECTION_ROI_ERROR";
  if (status == NPP_TEXTURE_BIND_ERROR) return "NPP_TEXTURE_BIND_ERROR";
  if (status == NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY) return "NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY";
  if (status == NPP_ZC_MODE_NOT_SUPPORTED_ERROR) return "NPP_ZC_MODE_NOT_SUPPORTED_ERROR";
  if (status == NPP_LUT_PALETTE_BITSIZE_ERROR) return "NPP_LUT_PALETTE_BITSIZE_ERROR";
  if (status == NPP_INVALID_DEVICE_POINTER_ERROR) return "NPP_INVALID_DEVICE_POINTER_ERROR";
  if (status == NPP_INVALID_HOST_POINTER_ERROR) return "NPP_INVALID_HOST_POINTER_ERROR";
  if (status == NPP_NOT_SUPPORTED_MODE_ERROR) return "NPP_NOT_SUPPORTED_MODE_ERROR";
  if (status == NPP_NO_OPERATION_WARNING) return "NPP_NO_OPERATION_WARNING";
  if (status == NPP_DIVIDE_BY_ZERO_WARNING) return "NPP_DIVIDE_BY_ZERO_WARNING";
  if (status == NPP_AFFINE_QUAD_INCORRECT_WARNING) return "NPP_AFFINE_QUAD_INCORRECT_WARNING";
  if (status == NPP_WRONG_INTERSECTION_ROI_WARNING) return "NPP_WRONG_INTERSECTION_ROI_WARNING";
  if (status == NPP_WRONG_INTERSECTION_QUAD_WARNING) return "NPP_WRONG_INTERSECTION_QUAD_WARNING";
  if (status == NPP_DOUBLE_SIZE_WARNING) return "NPP_DOUBLE_SIZE_WARNING";
  if (status == NPP_MISALIGNED_DST_ROI_WARNING) return "NPP_MISALIGNED_DST_ROI_WARNING";
  char buf[32];
  std::snprintf(buf, sizeof(buf), "UNKNOWN(%d)", static_cast<int>(status));
  return buf;
}

// Some runtimes leave a benign stale error behind (for example after a tiny
// cudaMalloc).  NPP kernels that call cudaGetLastError() after launch would
// report it as their own failure, so generated tests clear the slot right
// before invoking the API under test.
inline void clear_cuda_error() { (void)cudaGetLastError(); }

inline bool recording() {
  const char *dir = std::getenv("NPP_RECORD_DIR");
  return dir != nullptr && dir[0] != '\0';
}

inline bool verifying() {
  const char *dir = std::getenv("NPP_VERIFY_DIR");
  return dir != nullptr && dir[0] != '\0';
}

inline std::string record_root() {
  const char *dir = std::getenv("NPP_RECORD_DIR");
  return dir ? dir : "";
}

inline std::string verify_root() {
  const char *dir = std::getenv("NPP_VERIFY_DIR");
  return dir ? dir : "";
}

inline std::string sanitize_path(const std::string &id) {
  std::string out;
  out.reserve(id.size());
  for (char c : id) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '/' || c == '_' || c == '-' || c == '.') {
      out.push_back(c);
    } else {
      out.push_back('_');
    }
  }
  return out;
}

inline uint64_t fnv1a(const void *data, size_t bytes) {
  const uint8_t *p = static_cast<const uint8_t *>(data);
  uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < bytes; ++i) {
    h ^= p[i];
    h *= 1099511628211ULL;
  }
  return h;
}

// Deterministic input generation shared by record and verify runs.
class Rng {
public:
  explicit Rng(uint64_t seed) : state_(seed ? seed : 0x9e3779b97f4a7c15ULL) {}

  uint64_t next() {
    uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }

  // Uniform in [lo, hi] for integer types.
  template <typename T> T integer(T lo, T hi) {
    uint64_t span = static_cast<uint64_t>(hi) - static_cast<uint64_t>(lo);
    if (span == ~0ULL) {
      return static_cast<T>(next());
    }
    return static_cast<T>(static_cast<uint64_t>(lo) + next() % (span + 1));
  }

  float unit_float() {
    // 24 random mantissa bits in [0, 1)
    return static_cast<float>(next() >> 40) / 16777216.0f;
  }

  double unit_double() {
    return static_cast<double>(next() >> 11) / 9007199254740992.0;
  }

private:
  uint64_t state_;
};

inline Npp16f to_npp16f(float value) {
  Npp16f result;
  __half h = __float2half(value);
  result.fp16 = static_cast<short>(__half_raw(h).x);
  return result;
}

template <typename T> inline void fill_uniform(std::vector<T> &data, Rng &rng) {
  if constexpr (std::is_same_v<T, Npp16f>) {
    for (auto &v : data) {
      float f = rng.unit_float() * 4.0f - 2.0f;
      v = to_npp16f(f);
    }
  } else if constexpr (std::is_floating_point_v<T>) {
    for (auto &v : data) {
      v = static_cast<T>(rng.unit_double() * 4.0 - 2.0);
    }
  } else if constexpr (std::is_integral_v<T>) {
    using U = std::make_unsigned_t<T>;
    for (auto &v : data) {
      v = static_cast<T>(rng.integer<U>(0, static_cast<U>(~U{0})));
    }
  }
}

template <typename T> inline std::vector<T> make_uniform(size_t count, uint64_t seed) {
  std::vector<T> data(count);
  Rng rng(seed);
  fill_uniform(data, rng);
  return data;
}

struct OutputRecord {
  std::vector<uint8_t> bytes;
  ValueKind kind = ValueKind::Bytes;
  size_t count = 0;
  double rtol = 1e-5;
  double atol = 1e-5;
};

inline bool close_double(double a, double b, double rtol, double atol) {
  if (std::isnan(a) && std::isnan(b)) return true;
  if (std::isinf(a) || std::isinf(b)) return a == b;
  return std::fabs(a - b) <= atol + rtol * std::fabs(b);
}

inline bool compare_values(ValueKind kind, const uint8_t *a, const uint8_t *b, size_t count, double rtol,
                           double atol, size_t *mismatch) {
  const size_t esize = element_size(kind);
  for (size_t i = 0; i < count; ++i) {
    const uint8_t *pa = a + i * esize;
    const uint8_t *pb = b + i * esize;
    bool ok = true;
    switch (kind) {
    case ValueKind::Bytes:
    case ValueKind::U8:
    case ValueKind::I8:
      ok = std::memcmp(pa, pb, 1) == 0;
      break;
    case ValueKind::U16:
    case ValueKind::I16:
      ok = std::memcmp(pa, pb, 2) == 0;
      break;
    case ValueKind::U32:
    case ValueKind::I32:
      ok = std::memcmp(pa, pb, 4) == 0;
      break;
    case ValueKind::U64:
    case ValueKind::I64:
      ok = std::memcmp(pa, pb, 8) == 0;
      break;
    case ValueKind::I16Tol: {
      int16_t va, vb;
      std::memcpy(&va, pa, 2);
      std::memcpy(&vb, pb, 2);
      ok = close_double(va, vb, rtol, atol);
      break;
    }
    case ValueKind::I32Tol: {
      int32_t va, vb;
      std::memcpy(&va, pa, 4);
      std::memcpy(&vb, pb, 4);
      ok = close_double(va, vb, rtol, atol);
      break;
    }
    case ValueKind::I64Tol: {
      int64_t va, vb;
      std::memcpy(&va, pa, 8);
      std::memcpy(&vb, pb, 8);
      ok = close_double((double)va, (double)vb, rtol, atol);
      break;
    }
    case ValueKind::F16: {
      __half ha, hb;
      std::memcpy(&ha, pa, 2);
      std::memcpy(&hb, pb, 2);
      ok = close_double(__half2float(ha), __half2float(hb), rtol, atol);
      break;
    }
    case ValueKind::F32: {
      float fa, fb;
      std::memcpy(&fa, pa, 4);
      std::memcpy(&fb, pb, 4);
      ok = close_double(fa, fb, rtol, atol);
      break;
    }
    case ValueKind::F64: {
      double da, db;
      std::memcpy(&da, pa, 8);
      std::memcpy(&db, pb, 8);
      ok = close_double(da, db, rtol, atol);
      break;
    }
    case ValueKind::CF32: {
      float fa[2], fb[2];
      std::memcpy(fa, pa, 8);
      std::memcpy(fb, pb, 8);
      ok = close_double(fa[0], fb[0], rtol, atol) && close_double(fa[1], fb[1], rtol, atol);
      break;
    }
    case ValueKind::CF64: {
      double da[2], db[2];
      std::memcpy(da, pa, 16);
      std::memcpy(db, pb, 16);
      ok = close_double(da[0], db[0], rtol, atol) && close_double(da[1], db[1], rtol, atol);
      break;
    }
    }
    if (!ok) {
      if (mismatch) *mismatch = i;
      return false;
    }
  }
  return true;
}

class Case {
public:
  Case(std::string id, std::string function) : id_(std::move(id)), function_(std::move(function)) {}

  Case &expect_status(NppStatus status) {
    expected_status_ = status;
    has_expected_status_ = true;
    return *this;
  }

  Case &status(NppStatus status) {
    actual_status_ = status;
    has_status_ = true;
    return *this;
  }

  Case &input(const void *data, size_t bytes) {
    input_hashes_.push_back(fnv1a(data, bytes));
    return *this;
  }

  // Capture a device buffer.
  template <typename T>
  Case &device_output(const T *ptr, size_t count, ValueKind kind, double rtol = 1e-5, double atol = 1e-5) {
    OutputRecord rec;
    rec.kind = kind;
    rec.count = count;
    rec.rtol = rtol;
    rec.atol = atol;
    rec.bytes.resize(count * element_size(kind));
    if (count > 0 && ptr != nullptr) {
      cudaDeviceSynchronize();
      cudaError_t err = cudaMemcpy(rec.bytes.data(), ptr, rec.bytes.size(), cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) {
        ADD_FAILURE() << "cudaMemcpy(D2H) failed for case " << id_ << ": " << cudaGetErrorString(err);
        rec.bytes.assign(rec.bytes.size(), 0);
      }
    }
    outputs_.push_back(std::move(rec));
    return *this;
  }

  // Capture a host pointer (buffer-size helpers, affine coefficients, ...).
  template <typename T>
  Case &host_output(const T *ptr, size_t count, ValueKind kind, double rtol = 1e-5, double atol = 1e-5) {
    OutputRecord rec;
    rec.kind = kind;
    rec.count = count;
    rec.rtol = rtol;
    rec.atol = atol;
    rec.bytes.resize(count * element_size(kind));
    if (count > 0 && ptr != nullptr) {
      std::memcpy(rec.bytes.data(), ptr, rec.bytes.size());
    }
    outputs_.push_back(std::move(rec));
    return *this;
  }

  void finish() {
    if (recording() && verifying()) {
      ADD_FAILURE() << "set only one of NPP_RECORD_DIR / NPP_VERIFY_DIR";
      return;
    }
    if (recording()) {
      record();
    } else if (verifying()) {
      verify();
    } else if (has_expected_status_) {
      ASSERT_TRUE(has_status_) << "case " << id_ << " did not record a status";
      EXPECT_EQ(status_name(actual_status_), status_name(expected_status_)) << "case " << id_;
    }
  }

private:
  std::string meta_text() const {
    std::ostringstream os;
    os << "case=" << id_ << "\n";
    os << "function=" << function_ << "\n";
    os << "status=" << (has_status_ ? status_name(actual_status_) : "NONE") << "\n";
    os << "inputs=" << input_hashes_.size() << "\n";
    for (size_t i = 0; i < input_hashes_.size(); ++i) {
      os << "input_hash" << i << "=" << input_hashes_[i] << "\n";
    }
    os << "outputs=" << outputs_.size() << "\n";
    for (size_t i = 0; i < outputs_.size(); ++i) {
      os << "out" << i << "=" << kind_name(outputs_[i].kind) << "," << outputs_[i].count << ","
         << outputs_[i].bytes.size() << "\n";
    }
    return os.str();
  }

  void record() const {
    const std::string rel = sanitize_path(id_);
    const std::filesystem::path meta = std::filesystem::path(record_root()) / (rel + ".meta");
    const std::filesystem::path bin = std::filesystem::path(record_root()) / (rel + ".bin");
    std::error_code ec;
    std::filesystem::create_directories(meta.parent_path(), ec);
    {
      std::ofstream m(meta, std::ios::binary);
      m << meta_text();
      ASSERT_TRUE(m.good()) << "cannot write " << meta;
    }
    {
      std::ofstream b(bin, std::ios::binary);
      for (const auto &o : outputs_) {
        b.write(reinterpret_cast<const char *>(o.bytes.data()), static_cast<std::streamsize>(o.bytes.size()));
      }
      ASSERT_TRUE(b.good()) << "cannot write " << bin;
    }
  }

  void verify() const {
    const std::string rel = sanitize_path(id_);
    const std::filesystem::path meta = std::filesystem::path(verify_root()) / (rel + ".meta");
    const std::filesystem::path bin = std::filesystem::path(verify_root()) / (rel + ".bin");
    if (!std::filesystem::exists(meta)) {
      FAIL() << "missing golden " << meta << " (record on the 5090 first)";
      return;
    }
    std::ifstream m(meta);
    ASSERT_TRUE(m.good()) << "cannot read " << meta;

    std::string line, golden_status = "NONE";
    std::vector<ValueKind> kinds;
    std::vector<size_t> counts, sizes;
    size_t total_bytes = 0;
    while (std::getline(m, line)) {
      if (line.rfind("status=", 0) == 0) {
        golden_status = line.substr(7);
      } else if (line.rfind("out", 0) == 0) {
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string spec = line.substr(eq + 1);
        const auto c1 = spec.find(',');
        const auto c2 = spec.find(',', c1 + 1);
        if (c1 == std::string::npos || c2 == std::string::npos) continue;
        kinds.push_back(kind_from_name(spec.substr(0, c1)));
        counts.push_back(std::stoull(spec.substr(c1 + 1, c2 - c1 - 1)));
        const size_t bytes = std::stoull(spec.substr(c2 + 1));
        sizes.push_back(bytes);
        total_bytes += bytes;
      }
    }

    std::vector<uint8_t> golden(total_bytes);
    {
      std::ifstream b(bin, std::ios::binary);
      ASSERT_TRUE(b.good()) << "cannot read " << bin;
      b.read(reinterpret_cast<char *>(golden.data()), static_cast<std::streamsize>(golden.size()));
      ASSERT_EQ(static_cast<size_t>(b.gcount()), golden.size()) << "short golden " << bin;
    }

    ASSERT_TRUE(has_status_) << "case " << id_ << " did not record a status";
    EXPECT_EQ(status_name(actual_status_), golden_status) << "status mismatch for case " << id_;

    // Compare only when the reference succeeded; error-path cases are fully
    // covered by the status comparison.
    if (golden_status != "NPP_NO_ERROR" && golden_status != "NPP_SUCCESS") {
      return;
    }
    ASSERT_EQ(outputs_.size(), kinds.size()) << "output count mismatch for case " << id_;
    size_t offset = 0;
    for (size_t i = 0; i < outputs_.size(); ++i) {
      ASSERT_EQ(outputs_[i].kind, kinds[i]) << "kind mismatch for case " << id_ << " output " << i;
      ASSERT_EQ(outputs_[i].count, counts[i]) << "count mismatch for case " << id_ << " output " << i;
      ASSERT_EQ(outputs_[i].bytes.size(), sizes[i]) << "size mismatch for case " << id_ << " output " << i;
      size_t mismatch = 0;
      const bool ok = compare_values(outputs_[i].kind, outputs_[i].bytes.data(), golden.data() + offset,
                                     outputs_[i].count, outputs_[i].rtol, outputs_[i].atol, &mismatch);
      if (!ok) {
        std::ostringstream msg;
        msg << "value mismatch for case " << id_ << " output " << i << " at element " << mismatch;
        EXPECT_TRUE(ok) << msg.str();
      }
      offset += sizes[i];
    }
  }

  std::string id_;
  std::string function_;
  NppStatus actual_status_{};
  NppStatus expected_status_{};
  bool has_status_ = false;
  bool has_expected_status_ = false;
  std::vector<uint64_t> input_hashes_;
  std::vector<OutputRecord> outputs_;
};

// Convenience: build a stable case id from family/variant/case tokens.
inline std::string make_id(std::initializer_list<std::string> parts) {
  std::string out;
  for (const auto &p : parts) {
    if (!out.empty()) out += "/";
    out += p;
  }
  return out;
}

} // namespace npp_diff
