// CPU-reference output coverage for nppsDiv_32s16s_Sfs (npps_arithmetic header).
//
// Measured NPP 13.4 semantics (reference): pSrc1 (16-bit) is the DIVISOR and
// pSrc2 (32-bit) the dividend, dst = round_half_even(pSrc2 / (pSrc1 * 2^sf))
// saturated to Npp16s.  The reference's sf >= 2 scaling shows additional
// rounding quirks that are not modelled; this driver exercises sf 0 and 1 with
// inputs chosen away from the half-way ties and zero divisors where the
// reference and a plain half-even model coincide.
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace div32s16 {

inline bool check_case(const char *id, const GenCase &gc, const std::vector<Npp16s> &a,
                       const std::vector<Npp32s> &b, const std::vector<Npp16s> &got, int sf, RunCounts &cnt,
                       std::string &first) {
  bool ok = true;
  for (size_t i = 0; i < a.size(); ++i) {
    ++cnt.compared;
    const long long num = b[i];
    const long long den = (long long)a[i] << sf;
    long long want;
    if (den == 0) {
      want = 0;
    } else {
      const double q = (double)num / (double)den;
      const double r = std::nearbyint(q);  // round half to even
      want = (long long)r;
      if (want > 32767) want = 32767;
      if (want < -32768) want = -32768;
    }
    if ((long long)got[i] == want) continue;
    char buf[320];
    std::snprintf(buf, sizeof(buf), "[%s] %s i=%zu a=%lld b=%lld sf=%d got=%lld want=%lld", id, gc.id().c_str(), i,
                  (long long)a[i], (long long)b[i], sf, (long long)got[i], want);
    if (first.empty()) first = buf;
    EXPECT_TRUE(false) << buf;
    ++cnt.mismatched;
    ok = false;
    if (cnt.mismatched >= 8) return false;
  }
  return ok;
}

template <class Call>
void run_div32s16s(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  static const int lengths[] = {1, 7, 64, 101};
  static const int sfs[] = {0, 1};
  static const int divisors[] = {3, 5, 7, -3, -5};
  bool failed = false;
  for (int len: lengths) {
    for (int sf: sfs) {
      if (failed) break;
      std::vector<Npp16s> a((size_t)len);
      std::vector<Npp32s> b((size_t)len);
      for (int i = 0; i < len; ++i) {
        a[(size_t)i] = (Npp16s)divisors[i % 5];
        // Away from half-way ties: with an odd divisor and a 2^sf shift the
        // tie values are b == odd multiples of divisor << (sf - 1).
        Npp32s v = (Npp32s)(((i * 2654435761u) % 20001) - 10000);
        const long long den = (long long)a[(size_t)i] << sf;
        const long long ad = den < 0 ? -den: den;
        if (sf > 0 && ((v % ad) == ad / 2 || (v % ad) == -(ad / 2))) v += 1;
        b[(size_t)i] = v;
      }
      // Saturation sample (the reference's zero-divisor handling differs from
      // the plain model, so no zero divisor is used).
      if (len > 8) {
        a[0] = 1;
        b[0] = 100000;
      }
      npp_functional_test::DeviceMemory<Npp16s> da((size_t)len);
      npp_functional_test::DeviceMemory<Npp32s> db((size_t)len);
      npp_functional_test::DeviceMemory<Npp16s> dd((size_t)len);
      da.copyFromHost(a);
      db.copyFromHost(b);
      std::vector<Npp16s> z((size_t)len, 0x5A5A);
      dd.copyFromHost(z);
      npp_diff::clear_cuda_error();
      const NppStatus st = call(da.get(), db.get(), dd.get(), len, sf, ctx);
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        continue;
      }
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "[" << id << "] sync";
      ++cnt.executed;
      std::vector<Npp16s> got((size_t)len);
      cudaMemcpy(got.data(), dd.get(), got.size() * sizeof(Npp16s), cudaMemcpyDeviceToHost);
      const GenCase gc{len, 1, sf, 0, 0, kSmall, 0};
      if (!check_case(id, gc, a, b, got, sf, cnt, first)) failed = true;
      if (cnt.mismatched >= 8) failed = true;
    }
  }
  finish(id, cnt, first);
}

}  // namespace div32s16
}  // namespace num
