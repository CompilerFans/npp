#pragma once

// Numerical conformance test framework (independent of the existing diff and
// reference suites).
//
// Expected values are computed on the CPU in-process; there are no golden
// records and no status/error-code assertions.  A DUT call that returns
// non-success marks that subcase as skipped (it is not a failure).  The same
// sources build against the in-tree MPP library and, on the RTX 5090, against
// the closed-source NVIDIA NPP 13.4 library (USE_NVIDIA_NPP), so a green run
// there validates the CPU references and tolerances themselves.
//
// The framework focuses on numerical behaviour across dtype, layout, size,
// step and value regime.  Add a family by adding one TEST per API: allocate
// nothing, just provide the _Ctx call and the CPU reference.

#include "npp.h"
#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace num {

using npp_diff::Rng;
using npp_functional_test::DeviceMemory;

// ---------------------------------------------------------------------------
// Element kind mapping (mirrors test/unit/ref/ref_test_utils.h kind_for).
// ---------------------------------------------------------------------------
template <typename T> constexpr npp_diff::ValueKind kind_of() {
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
  else return K::Bytes;
}

inline size_t align16(size_t n) { return (n + 15) & ~size_t(15); }

// ---------------------------------------------------------------------------
// Deterministic input regimes.
// ---------------------------------------------------------------------------
enum Regime { kSmall, kWide, kExtremes };

inline uint64_t mix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

inline uint64_t case_seed(const char *id, int w, int h, int sstride, int dstride, int regime,
                          int param) {
  uint64_t h64 = 1469598103934665603ull;
  for (const char *p = id; *p; ++p) {
    h64 ^= (uint8_t)*p;
    h64 *= 1099511628211ull;
  }
  h64 = mix64(h64 ^ (uint64_t)w);
  h64 = mix64(h64 ^ ((uint64_t)h << 8));
  h64 = mix64(h64 ^ ((uint64_t)sstride << 16));
  h64 = mix64(h64 ^ ((uint64_t)dstride << 24));
  h64 = mix64(h64 ^ ((uint64_t)regime << 32));
  h64 = mix64(h64 ^ ((uint64_t)param << 40));
  return h64;
}

template <typename T> inline T regime_value(Rng &rng, Regime r, size_t index) {
  using L = std::numeric_limits<T>;
  if constexpr (std::is_same_v<T, Npp16f>) {
    float v;
    if (r == kSmall) v = (float)((int)(rng.next() % 9) - 4);
    else if (r == kWide) v = rng.unit_float() * 2000.0f - 1000.0f;
    else {
      static const float ext[] = {0.0f, -0.0f, 1.0f, -1.0f, 65504.0f, -65504.0f,
                                  6.1035156e-5f};
      v = ext[index % (sizeof(ext) / sizeof(ext[0]))];
    }
    return npp_diff::to_npp16f(v);
  } else if constexpr (std::is_same_v<T, Npp32fc> || std::is_same_v<T, Npp64fc> ||
                       std::is_same_v<T, Npp16sc> || std::is_same_v<T, Npp32sc> ||
                       std::is_same_v<T, Npp64sc> || std::is_same_v<T, Npp8uc> ||
                       std::is_same_v<T, Npp16uc> || std::is_same_v<T, Npp32uc>) {
    using R = decltype(T{}.re);
    T v;
    Rng r2((uint64_t)index * 0x9e3779b97f4a7c15ull ^ (uint64_t)r);
    if (r == kSmall) {
      v.re = (R)((int)(r2.next() % 9) - 4);
      v.im = (R)((int)(r2.next() % 9) - 4);
    } else {
      v.re = (R)((int)(r2.next() % 2001) - 1000);
      v.im = (R)((int)(r2.next() % 2001) - 1000);
    }
    return v;
  } else if constexpr (std::is_floating_point_v<T>) {
    T v;
    if (r == kSmall) {
      v = (T)((int)(rng.next() % 9) - 4) * (T)0.25;
    } else if (r == kWide) {
      v = (T)(rng.unit_double() * 2000.0 - 1000.0);
    } else {
      static const T ext[] = {T(0), T(-0.0), T(1), T(-1), std::numeric_limits<T>::max(),
                              std::numeric_limits<T>::lowest(),
                              std::numeric_limits<T>::infinity(),
                              -std::numeric_limits<T>::infinity(),
                              std::numeric_limits<T>::quiet_NaN(),
                              std::numeric_limits<T>::denorm_min()};
      v = ext[index % (sizeof(ext) / sizeof(ext[0]))];
    }
    return v;
  } else {
    // Integral.
    if (r == kSmall) {
      if constexpr (std::is_signed_v<T>) return (T)((int)(rng.next() % 17) - 8);
      else return (T)(rng.next() % 16);
    } else if (r == kWide) {
      if constexpr (sizeof(T) == 8) {
        return (T)rng.next();
      } else if constexpr (std::is_signed_v<T>) {
        const uint64_t span =
            (uint64_t)((int64_t)L::max() - (int64_t)L::lowest() + 1);
        return (T)((int64_t)(rng.next() % span) + (int64_t)L::lowest());
      } else {
        return (T)(rng.next() % ((uint64_t)L::max() + 1));
      }
    } else {
      static const T ext[] = {T(0), T(1), L::max(), L::lowest(), (T)(L::max() - 1)};
      return ext[index % (sizeof(ext) / sizeof(ext[0]))];
    }
  }
}

// ---------------------------------------------------------------------------
// Case matrix.
// ---------------------------------------------------------------------------
struct Case {
  int w, h;
  int src_pad;  // extra bytes added to the packed row stride
  int dst_pad;
  Regime regime;
  int param;  // scale factor / mode / enum value (family-specific)
  std::string id() const;
};

inline std::string Case::id() const {
  char buf[160];
  std::snprintf(buf, sizeof(buf), "%dx%d/sp%d/dp%d/r%d/p%d", w, h, src_pad, dst_pad, (int)regime,
                param);
  return buf;
}

// Number of parameters is family-specific (scale factor, mask, ...).  The
// driver is invoked once per parameter value by the caller.
inline std::vector<Case> make_cases(int param, int size_filter = 0) {
  static const Regime regimes[] = {kSmall, kWide, kExtremes};
  static const int sizes[][2] = {{1, 1}, {3, 5}, {17, 13}, {64, 64}, {1, 257}, {31, 17}};
  static const int pads[] = {0, 16, 32};
  std::vector<Case> v;
  for (int si = 0; si < (int)(sizeof(sizes) / sizeof(sizes[0])); ++si) {
    if (size_filter >= 0 && si != size_filter) continue;
    for (int pad : pads) {
      for (Regime r : regimes) {
        Case c{};
        c.w = sizes[si][0];
        c.h = sizes[si][1];
        c.src_pad = pad;
        c.dst_pad = pad;
        c.regime = r;
        c.param = param;
        v.push_back(c);
      }
    }
  }
  return v;
}

// ---------------------------------------------------------------------------
// Failure accounting / reporting.
// ---------------------------------------------------------------------------
struct Stats {
  uint64_t executed = 0;
  uint64_t skipped = 0;
  uint64_t skipped_not_implemented = 0;
  uint64_t mismatched_cases = 0;
  uint64_t compared_elements = 0;
};

inline Stats &stats() {
  static Stats s;
  return s;
}

inline bool report_enabled() {
  const char *v = std::getenv("NUM_REPORT");
  return v != nullptr && v[0] != '\0' && v[0] != '0';
}

inline std::string status_name_safe(NppStatus st) { return npp_diff::status_name(st); }

// Debug formatting for failure messages (F16/F32/F64/integers/complex).
inline std::string debug_value(npp_diff::ValueKind kind, const uint8_t *p) {
  char buf[96];
  switch (kind) {
    case npp_diff::ValueKind::F16: {
      __half h;
      std::memcpy(&h, p, 2);
      std::snprintf(buf, sizeof(buf), "%.9g", __half2float(h));
      return buf;
    }
    case npp_diff::ValueKind::F32: {
      float v;
      std::memcpy(&v, p, 4);
      std::snprintf(buf, sizeof(buf), "%.9g", v);
      return buf;
    }
    case npp_diff::ValueKind::F64: {
      double v;
      std::memcpy(&v, p, 8);
      std::snprintf(buf, sizeof(buf), "%.17g", v);
      return buf;
    }
    case npp_diff::ValueKind::I32:
    case npp_diff::ValueKind::I32Tol: {
      int32_t v;
      std::memcpy(&v, p, 4);
      std::snprintf(buf, sizeof(buf), "%d", v);
      return buf;
    }
    case npp_diff::ValueKind::U32: {
      uint32_t v;
      std::memcpy(&v, p, 4);
      std::snprintf(buf, sizeof(buf), "%u", v);
      return buf;
    }
    case npp_diff::ValueKind::U64: {
      uint64_t v;
      std::memcpy(&v, p, 8);
      std::snprintf(buf, sizeof(buf), "%llu", (unsigned long long)v);
      return buf;
    }
    case npp_diff::ValueKind::I64: {
      int64_t v;
      std::memcpy(&v, p, 8);
      std::snprintf(buf, sizeof(buf), "%lld", (long long)v);
      return buf;
    }
    case npp_diff::ValueKind::I16:
    case npp_diff::ValueKind::I16Tol: {
      int16_t v;
      std::memcpy(&v, p, 2);
      std::snprintf(buf, sizeof(buf), "%d", (int)v);
      return buf;
    }
    case npp_diff::ValueKind::U16: {
      uint16_t v;
      std::memcpy(&v, p, 2);
      std::snprintf(buf, sizeof(buf), "%u", (unsigned)v);
      return buf;
    }
    case npp_diff::ValueKind::I8:
    case npp_diff::ValueKind::U8:
    case npp_diff::ValueKind::Bytes: {
      std::snprintf(buf, sizeof(buf), "%u", (unsigned)*p);
      return buf;
    }
    default:
      return "?";
  }
}


// ---------------------------------------------------------------------------
// Binary elementwise driver (out-of-place): dst = f(src1, src2, param)
// ---------------------------------------------------------------------------
template <typename S, typename D, int C, int A, class Call, class Ref>
void run_binary(const char *op_id, Call call, Ref ref, double rtol, double atol,
                const std::vector<int> &params) {
  NppStreamContext ctx{};
  NppStatus cst = nppGetStreamContext(&ctx);
  if (cst != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << op_id << "] nppGetStreamContext: " << status_name_safe(cst);
    return;
  }

  uint64_t executed = 0, skipped = 0, skipped_ni = 0, mismatched = 0, compared = 0;
  std::string first_failure;

  for (int param : params) {
    for (const Case &c : make_cases(param)) {
      const int ss = (int)align16((size_t)c.w * C * sizeof(S)) + c.src_pad;
      const int ds = (int)align16((size_t)c.w * C * sizeof(D)) + c.dst_pad;
      const size_t sbytes = (size_t)ss * c.h;
      const size_t dbytes = (size_t)ds * c.h;

      const uint64_t seed = case_seed(op_id, c.w, c.h, ss, ds, (int)c.regime, param);
      Rng rng(seed);
      std::vector<uint8_t> h1(sbytes), h2(sbytes), hd(dbytes), want(dbytes, 0x5A);
      {
        S *p1 = reinterpret_cast<S *>(h1.data());
        S *p2 = reinterpret_cast<S *>(h2.data());
        const size_t n1 = sbytes / sizeof(S);
        for (size_t i = 0; i < n1; ++i) {
          p1[i] = regime_value<S>(rng, c.regime, i);
          p2[i] = regime_value<S>(rng, c.regime, i + 1);
        }
        if (sbytes % sizeof(S)) std::memset(h1.data() + n1 * sizeof(S), 0, sbytes - n1 * sizeof(S));
        if (sbytes % sizeof(S)) std::memset(h2.data() + n1 * sizeof(S), 0, sbytes - n1 * sizeof(S));
      }

      DeviceMemory<uint8_t> d1(sbytes), d2(sbytes), dd(dbytes);
      d1.copyFromHost(h1);
      d2.copyFromHost(h2);
      dd.copyFromHost(hd);

      npp_diff::clear_cuda_error();
      NppStatus st = call(reinterpret_cast<const S *>(d1.get()), ss,
                          reinterpret_cast<const S *>(d2.get()), ss,
                          reinterpret_cast<D *>(dd.get()), ds, NppiSize{c.w, c.h}, param, ctx);
      if (st != NPP_NO_ERROR) {
        ++skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++skipped_ni;
        continue;
      }
      cudaError_t se = cudaDeviceSynchronize();
      ASSERT_EQ(se, cudaSuccess) << "[" << op_id << "] " << c.id() << " sync: "
                                 << cudaGetErrorString(se);
      dd.copyToHost(hd);

      // CPU reference over the ROI (active channels only), leaving the
      // remaining destination bytes at the sentinel so alpha preservation is
      // checked as well.
      D *pd = reinterpret_cast<D *>(want.data());
      for (int y = 0; y < c.h; ++y) {
        for (int x = 0; x < c.w; ++x) {
          for (int ch = 0; ch < C; ++ch) {
            const size_t idx = (size_t)y * (ds / sizeof(D)) + (size_t)x * C + ch;
            // Inactive channels (AC4 alpha): the 5090 arithmetic kernels write
            // 0 rather than preserving the destination alpha.
            if (ch >= A) {
              pd[idx] = D(0);
              continue;
            }
            const S a = reinterpret_cast<const S *>(h1.data())[(size_t)y * (ss / sizeof(S)) +
                                                               (size_t)x * C + ch];
            const S b = reinterpret_cast<const S *>(h2.data())[(size_t)y * (ss / sizeof(S)) +
                                                               (size_t)x * C + ch];
            pd[idx] = ref(a, b, param);
          }
        }
      }

      ++executed;
      const npp_diff::ValueKind kind = kind_of<D>();
      const size_t es = npp_diff::element_size(kind);
      for (int y = 0; y < c.h; ++y) {
        for (int x = 0; x < c.w; ++x) {
          for (int ch = 0; ch < C; ++ch) {
            const size_t off = (size_t)y * ds + ((size_t)x * C + ch) * es;
            const S av = reinterpret_cast<const S *>(h1.data())[(size_t)y * (ss / sizeof(S)) +
                                                                 (size_t)x * C + ch];
            const S bv = reinterpret_cast<const S *>(h2.data())[(size_t)y * (ss / sizeof(S)) +
                                                                 (size_t)x * C + ch];
            ++compared;
            if (!npp_diff::compare_values(kind, hd.data() + off, want.data() + off, 1, rtol, atol,
                                          nullptr)) {
              if (mismatched < 8) {
                char buf[320];
                std::snprintf(buf, sizeof(buf),
                              "[%s] %s mismatch at (x=%d,y=%d,ch=%d): got=%s want=%s a=%.17g "
                              "b=%.17g p=%d (rtol=%g atol=%g)",
                              op_id, c.id().c_str(), x, y, ch,
                              debug_value(kind, hd.data() + off).c_str(),
                              debug_value(kind, want.data() + off).c_str(), (double)av, (double)bv,
                              param, rtol, atol);
                if (first_failure.empty()) first_failure = buf;
                if (std::getenv("NUM_DEBUG_DEVICE") != nullptr) {
                  S dv1 = S(0), dv2 = S(0);
                  const size_t src_off = (size_t)y * ss + ((size_t)x * C + ch) * sizeof(S);
                  cudaMemcpy(&dv1, d1.get() + src_off, sizeof(S), cudaMemcpyDeviceToHost);
                  cudaMemcpy(&dv2, d2.get() + src_off, sizeof(S), cudaMemcpyDeviceToHost);
                  std::fprintf(stderr, "DEV %s %s at (%d,%d,%d): dev_a=%.17g dev_b=%.17g\n",
                               op_id, c.id().c_str(), x, y, ch, (double)dv1, (double)dv2);
                }
                EXPECT_TRUE(false) << buf;
              }
              ++mismatched;
              break;
            }
          }
          if (mismatched >= 8) break;
        }
        if (mismatched >= 8) break;
      }
    }
  }

  stats().executed += executed;
  stats().skipped += skipped;
  stats().skipped_not_implemented += skipped_ni;
  stats().mismatched_cases += mismatched;
  stats().compared_elements += compared;

  if (executed == 0 && skipped > 0) {
    GTEST_SKIP() << "[" << op_id << "] all " << skipped
                 << " subcases returned non-success (not-implemented=" << skipped_ni << ")";
    return;
  }
  if (mismatched > 8) {
    ADD_FAILURE() << "[" << op_id << "] total mismatched cases: " << mismatched
                  << " (only the first 8 are shown)";
  }
  if (report_enabled()) {
    std::fprintf(stderr, "[NUM] %-48s executed=%llu skipped=%llu notimpl=%llu mismatch=%llu\n",
                 op_id, (unsigned long long)executed, (unsigned long long)skipped,
                 (unsigned long long)skipped_ni, (unsigned long long)mismatched);
  }
}

}  // namespace num
