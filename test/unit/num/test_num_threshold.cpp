// Numerical conformance: nppiThreshold / nppiCompare families.
//
// NPP semantics (validated by the 5090 gate): threshold sets the pixel to the
// threshold value when (src OP threshold) is true, otherwise copies the source;
// compare writes an 8u_C1 mask (0xFF true / 0x00 false) and requires the
// condition for every active channel.  No status assertions: a non-success DUT
// status skips the subcase.

#include "num_test_utils.h"

#include <array>

namespace {

using num::align16;
using num::Case;
using num::case_seed;
using num::debug_value;
using num::kind_of;
using num::make_cases;
using num::regime_value;
using num::Regime;
using num::Rng;
using npp_functional_test::DeviceMemory;

constexpr NppCmpOp kThresholdOps[] = {NPP_CMP_LESS, NPP_CMP_GREATER};
constexpr NppCmpOp kCompareOps[] = {NPP_CMP_LESS, NPP_CMP_LESS_EQ, NPP_CMP_EQ,
                                    NPP_CMP_GREATER_EQ, NPP_CMP_GREATER};

template <typename T, int C, int A, class Call>
void run_threshold_scalar(const char *op_id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << op_id << "] no stream context";
    return;
  }
  uint64_t executed = 0, skipped = 0, ni = 0, mismatched = 0;
  for (NppCmpOp op : kThresholdOps) {
    for (const Case &c : make_cases((int)op)) {
      const int ss = (int)align16((size_t)c.w * C * sizeof(T)) + c.src_pad;
      const int ds = (int)align16((size_t)c.w * C * sizeof(T)) + c.dst_pad;
      const size_t sbytes = (size_t)ss * c.h, dbytes = (size_t)ds * c.h;
      Rng rng(case_seed(op_id, c.w, c.h, ss, ds, (int)c.regime, (int)op));
      std::vector<uint8_t> hs(sbytes), hd(dbytes, 0x5A), want(dbytes, 0x5A);
      T *ps = reinterpret_cast<T *>(hs.data());
      for (size_t i = 0; i < sbytes / sizeof(T); ++i)
        ps[i] = regime_value<T>(rng, c.regime, i);
      // Deterministic threshold: every third case uses the source pixel at
      // (0,0) (boundary), otherwise a regime value.
      T thr = (c.w * c.h) % 3 == 0 ? ps[0] : regime_value<T>(rng, c.regime, 7);
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x)
          for (int ch = 0; ch < A; ++ch) {
            const T v = ps[(size_t)y * (ss / sizeof(T)) + (size_t)x * C + ch];
            const bool cond = op == NPP_CMP_LESS ? (v < thr) : (v > thr);
            reinterpret_cast<T *>(want.data())[(size_t)y * (ds / sizeof(T)) + (size_t)x * C + ch] =
                cond ? thr : v;
          }
      DeviceMemory<uint8_t> dsr(sbytes), dds(dbytes);
      dsr.copyFromHost(hs);
      dds.copyFromHost(hd);
      npp_diff::clear_cuda_error();
      NppStatus st = call(reinterpret_cast<const T *>(dsr.get()), ss,
                          reinterpret_cast<T *>(dds.get()), ds, NppiSize{c.w, c.h}, thr, op, ctx);
      if (st != NPP_NO_ERROR) {
        ++skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++ni;
        continue;
      }
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      dds.copyToHost(hd);
      ++executed;
      const auto kind = kind_of<T>();
      const size_t es = npp_diff::element_size(kind);
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x)
          for (int ch = 0; ch < C; ++ch) {
            const size_t off = (size_t)y * ds + ((size_t)x * C + ch) * es;
            if (!npp_diff::compare_values(kind, hd.data() + off, want.data() + off, 1, 0, 0, nullptr)) {
              if (mismatched < 6)
                EXPECT_TRUE(false) << "[" << op_id << "] " << c.id() << " op=" << (int)op
                                   << " at (" << x << "," << y << "," << ch << "): got="
                                   << debug_value(kind, hd.data() + off) << " want="
                                   << debug_value(kind, want.data() + off);
              ++mismatched;
            }
          }
    }
  }
  num::stats().executed += executed;
  num::stats().skipped += skipped;
  num::stats().skipped_not_implemented += ni;
  num::stats().mismatched_cases += mismatched;
  if (executed == 0 && skipped > 0) {
    GTEST_SKIP() << "[" << op_id << "] all subcases returned non-success";
    return;
  }
  if (num::report_enabled())
    std::fprintf(stderr, "[NUM] %-48s executed=%llu skipped=%llu notimpl=%llu mismatch=%llu\n",
                 op_id, (unsigned long long)executed, (unsigned long long)skipped,
                 (unsigned long long)ni, (unsigned long long)mismatched);
}

// Per-channel threshold arrays (C3 / AC4).
template <typename T, int C, int A, class Call>
void run_threshold_array(const char *op_id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << op_id << "] no stream context";
    return;
  }
  uint64_t executed = 0, skipped = 0, ni = 0, mismatched = 0;
  for (NppCmpOp op : kThresholdOps) {
    for (const Case &c : make_cases((int)op)) {
      const int ss = (int)align16((size_t)c.w * C * sizeof(T)) + c.src_pad;
      const int ds = (int)align16((size_t)c.w * C * sizeof(T)) + c.dst_pad;
      const size_t sbytes = (size_t)ss * c.h, dbytes = (size_t)ds * c.h;
      Rng rng(case_seed(op_id, c.w, c.h, ss, ds, (int)c.regime, (int)op));
      std::vector<uint8_t> hs(sbytes), hd(dbytes, 0x5A), want(dbytes, 0x5A);
      T *ps = reinterpret_cast<T *>(hs.data());
      for (size_t i = 0; i < sbytes / sizeof(T); ++i)
        ps[i] = regime_value<T>(rng, c.regime, i);
      T thr[3];
      for (int ch = 0; ch < 3; ++ch)
        thr[ch] = (c.w * c.h) % 3 == 0 ? ps[ch] : regime_value<T>(rng, c.regime, 11 + ch);
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x)
          for (int ch = 0; ch < A; ++ch) {
            const T v = ps[(size_t)y * (ss / sizeof(T)) + (size_t)x * C + ch];
            const bool cond = op == NPP_CMP_LESS ? (v < thr[ch]) : (v > thr[ch]);
            reinterpret_cast<T *>(want.data())[(size_t)y * (ds / sizeof(T)) + (size_t)x * C + ch] =
                cond ? thr[ch] : v;
          }
      DeviceMemory<uint8_t> dsr(sbytes), dds(dbytes);
      dsr.copyFromHost(hs);
      dds.copyFromHost(hd);
      npp_diff::clear_cuda_error();
      NppStatus st = call(reinterpret_cast<const T *>(dsr.get()), ss,
                          reinterpret_cast<T *>(dds.get()), ds, NppiSize{c.w, c.h}, thr, op, ctx);
      if (st != NPP_NO_ERROR) {
        ++skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++ni;
        continue;
      }
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      dds.copyToHost(hd);
      ++executed;
      const auto kind = kind_of<T>();
      const size_t es = npp_diff::element_size(kind);
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x)
          for (int ch = 0; ch < C; ++ch) {
            const size_t off = (size_t)y * ds + ((size_t)x * C + ch) * es;
            if (!npp_diff::compare_values(kind, hd.data() + off, want.data() + off, 1, 0, 0, nullptr)) {
              if (mismatched < 6)
                EXPECT_TRUE(false) << "[" << op_id << "] " << c.id() << " op=" << (int)op
                                   << " at (" << x << "," << y << "," << ch << "): got="
                                   << debug_value(kind, hd.data() + off) << " want="
                                   << debug_value(kind, want.data() + off);
              ++mismatched;
            }
          }
    }
  }
  num::stats().executed += executed;
  num::stats().skipped += skipped;
  num::stats().skipped_not_implemented += ni;
  num::stats().mismatched_cases += mismatched;
  if (executed == 0 && skipped > 0) {
    GTEST_SKIP() << "[" << op_id << "] all subcases returned non-success";
    return;
  }
  if (num::report_enabled())
    std::fprintf(stderr, "[NUM] %-48s executed=%llu skipped=%llu notimpl=%llu mismatch=%llu\n",
                 op_id, (unsigned long long)executed, (unsigned long long)skipped,
                 (unsigned long long)ni, (unsigned long long)mismatched);
}

// Compare: two images -> 8u_C1 mask (0xFF true, 0x00 false), AND over the
// active channels.
template <typename T, int C, int A, class Call>
void run_compare(const char *op_id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << op_id << "] no stream context";
    return;
  }
  uint64_t executed = 0, skipped = 0, ni = 0, mismatched = 0;
  for (NppCmpOp op : kCompareOps) {
    for (const Case &c : make_cases((int)op)) {
      const int ss = (int)align16((size_t)c.w * C * sizeof(T)) + c.src_pad;
      const int ms = (int)align16((size_t)c.w) + c.dst_pad;
      const size_t sbytes = (size_t)ss * c.h, mbytes = (size_t)ms * c.h;
      Rng rng(case_seed(op_id, c.w, c.h, ss, ms, (int)c.regime, (int)op));
      std::vector<uint8_t> h1(sbytes), h2(sbytes), hm(mbytes, 0x5A), want(mbytes, 0x5A);
      T *p1 = reinterpret_cast<T *>(h1.data());
      T *p2 = reinterpret_cast<T *>(h2.data());
      for (size_t i = 0; i < sbytes / sizeof(T); ++i) {
        p1[i] = regime_value<T>(rng, c.regime, i);
        p2[i] = regime_value<T>(rng, c.regime, i + 1);
      }
      if ((c.w * c.h) % 4 == 0) std::memcpy(h2.data(), h1.data(), sbytes);  // equality coverage
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x) {
          bool all = true;
          for (int ch = 0; ch < A; ++ch) {
            const T a = p1[(size_t)y * (ss / sizeof(T)) + (size_t)x * C + ch];
            const T b = p2[(size_t)y * (ss / sizeof(T)) + (size_t)x * C + ch];
            bool cnd = false;
            switch (op) {
              case NPP_CMP_LESS: cnd = a < b; break;
              case NPP_CMP_LESS_EQ: cnd = a <= b; break;
              case NPP_CMP_EQ: cnd = a == b; break;
              case NPP_CMP_GREATER_EQ: cnd = a >= b; break;
              default: cnd = a > b; break;
            }
            all = all && cnd;
          }
          want[(size_t)y * ms + x] = all ? 0xFF : 0x00;
        }
      DeviceMemory<uint8_t> d1(sbytes), d2(sbytes), dm(mbytes);
      d1.copyFromHost(h1);
      d2.copyFromHost(h2);
      dm.copyFromHost(hm);
      npp_diff::clear_cuda_error();
      NppStatus st = call(reinterpret_cast<const T *>(d1.get()), ss,
                          reinterpret_cast<const T *>(d2.get()), ss,
                          reinterpret_cast<Npp8u *>(dm.get()), ms, NppiSize{c.w, c.h}, op, ctx);
      if (st != NPP_NO_ERROR) {
        ++skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++ni;
        continue;
      }
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      dm.copyToHost(hm);
      ++executed;
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x) {
          const size_t off = (size_t)y * ms + x;
          if (hm[off] != want[off]) {
            if (mismatched < 6)
              EXPECT_TRUE(false) << "[" << op_id << "] " << c.id() << " op=" << (int)op << " at ("
                                 << x << "," << y << "): got=" << (unsigned)hm[off]
                                 << " want=" << (unsigned)want[off];
            ++mismatched;
          }
        }
    }
  }
  num::stats().executed += executed;
  num::stats().skipped += skipped;
  num::stats().skipped_not_implemented += ni;
  num::stats().mismatched_cases += mismatched;
  if (executed == 0 && skipped > 0) {
    GTEST_SKIP() << "[" << op_id << "] all subcases returned non-success";
    return;
  }
  if (num::report_enabled())
    std::fprintf(stderr, "[NUM] %-48s executed=%llu skipped=%llu notimpl=%llu mismatch=%llu\n",
                 op_id, (unsigned long long)executed, (unsigned long long)skipped,
                 (unsigned long long)ni, (unsigned long long)mismatched);
}

}  // namespace

#define NUM_THR_SCALAR(NAME, T, C, A, CALL)                                     \
  TEST(NumThreshold, NAME) {                                                    \
    run_threshold_scalar<T, C, A>(                                              \
        #NAME,                                                                  \
        [](const T *s, int ss, T *d, int ds, NppiSize r, T thr, NppCmpOp op,     \
           NppStreamContext c) { return CALL; });                               \
  }
#define NUM_THR_ARRAY(NAME, T, C, A, CALL)                                      \
  TEST(NumThreshold, NAME) {                                                    \
    run_threshold_array<T, C, A>(                                               \
        #NAME,                                                                  \
        [](const T *s, int ss, T *d, int ds, NppiSize r, const T *thr, NppCmpOp op, \
           NppStreamContext c) { return CALL; });                               \
  }
#define NUM_CMP(NAME, T, C, A, CALL)                                            \
  TEST(NumCompare, NAME) {                                                      \
    run_compare<T, C, A>(                                                       \
        #NAME,                                                                  \
        [](const T *a, int sa, const T *b, int sb, Npp8u *d, int ds, NppiSize r, \
           NppCmpOp op, NppStreamContext c) { return CALL; });                  \
  }

NUM_THR_ARRAY(Threshold_8u_C3R, Npp8u, 3, 3,
              (nppiThreshold_8u_C3R_Ctx(s, ss, d, ds, r, thr, op, c)))
NUM_THR_ARRAY(Threshold_8u_AC4R, Npp8u, 4, 3,
              (nppiThreshold_8u_AC4R_Ctx(s, ss, d, ds, r, thr, op, c)))
NUM_THR_SCALAR(Threshold_16u_C1R, Npp16u, 1, 1,
               (nppiThreshold_16u_C1R_Ctx(s, ss, d, ds, r, thr, op, c)))
NUM_THR_SCALAR(Threshold_16s_C1R, Npp16s, 1, 1,
               (nppiThreshold_16s_C1R_Ctx(s, ss, d, ds, r, thr, op, c)))
NUM_THR_SCALAR(Threshold_32f_C1R, Npp32f, 1, 1,
               (nppiThreshold_32f_C1R_Ctx(s, ss, d, ds, r, thr, op, c)))

NUM_CMP(Compare_8u_C1R, Npp8u, 1, 1,
        (nppiCompare_8u_C1R_Ctx(a, sa, b, sb, d, ds, r, op, c)))
NUM_CMP(Compare_8u_C3R, Npp8u, 3, 3,
        (nppiCompare_8u_C3R_Ctx(a, sa, b, sb, d, ds, r, op, c)))
NUM_CMP(Compare_8u_C4R, Npp8u, 4, 4,
        (nppiCompare_8u_C4R_Ctx(a, sa, b, sb, d, ds, r, op, c)))
NUM_CMP(Compare_8u_AC4R, Npp8u, 4, 3,
        (nppiCompare_8u_AC4R_Ctx(a, sa, b, sb, d, ds, r, op, c)))
NUM_CMP(Compare_16u_C1R, Npp16u, 1, 1,
        (nppiCompare_16u_C1R_Ctx(a, sa, b, sb, d, ds, r, op, c)))
NUM_CMP(Compare_16s_C1R, Npp16s, 1, 1,
        (nppiCompare_16s_C1R_Ctx(a, sa, b, sb, d, ds, r, op, c)))
NUM_CMP(Compare_32f_C1R, Npp32f, 1, 1,
        (nppiCompare_32f_C1R_Ctx(a, sa, b, sb, d, ds, r, op, c)))

// NOTE: keep the 8u C1 scalar-threshold test last.  On the closed-source NPP
// runtime, running it before the C3/AC4 array-threshold variants leaves NPP
// returning success without writing the destination (measured on the 5090;
// ordering-dependent, alone each test passes).
NUM_THR_SCALAR(Threshold_8u_C1R, Npp8u, 1, 1,
               (nppiThreshold_8u_C1R_Ctx(s, ss, d, ds, r, thr, op, c)))
