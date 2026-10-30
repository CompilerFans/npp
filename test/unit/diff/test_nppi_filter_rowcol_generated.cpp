// Differential tests for package f12: nppiFilterRow / nppiFilterColumn
// (1D separable filters without border handling): integer kernels with
// divisor (8u/16u/16s C1/C3/C4/AC4), 32f kernels (32f C1/C3/C4/AC4) and
// 64f_C1R.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiFilterRowColDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppiFilterRowColDiff.*'
//
// Kernels and float sources are dyadic (integer multiples of 1/8) and 16-bit
// sources use their full range: every float partial sum stays below 2^24, so
// the accumulation is exact regardless of the implementation's summation
// order.  Integer-kernel variants use int32 ring arithmetic (wrap once,
// divide with C truncation, saturate).

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

using namespace npp_diff;
using namespace npp_functional_test;
namespace {

template <typename T> DeviceMemory<T> &arena(std::size_t n) {
  static std::vector<std::unique_ptr<DeviceMemory<T>>> v;
  v.push_back(std::make_unique<DeviceMemory<T>>(n ? n : 1));
  return *v.back();
}

NppStreamContext current_ctx() {
  NppStreamContext c{};
  NppStatus st = nppGetStreamContext(&c);
  EXPECT_EQ(st, NPP_NO_ERROR);
  return c;
}

int align_step(int step, int es) { return ((step + es - 1) / es) * es; }

// Integral sources use their full range; floating-point sources are dyadic
// (multiples of 1/8).  extreme > 0 sprinkles the type extremes.
template <typename T> std::vector<T> make_src(size_t n, uint64_t seed, int extreme = 0) {
  Rng rng(seed);
  std::vector<T> v(n);
  size_t i = 0;
  for (auto &x : v) {
    if constexpr (std::is_integral_v<T>) {
      x = rng.integer<T>(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
      if (extreme > 0) {
        if (i % (size_t)extreme == 0) x = std::numeric_limits<T>::min();
        else if (i % (size_t)extreme == (size_t)extreme / 2) x = std::numeric_limits<T>::max();
      }
    } else {
      x = static_cast<T>(rng.integer<int>(-64, 64)) * static_cast<T>(0.125);
      if (extreme > 0) {
        if (i % (size_t)extreme == 0) x = static_cast<T>(-4096.0);
        else if (i % (size_t)extreme == (size_t)extreme / 2) x = static_cast<T>(4096.0);
      }
    }
    ++i;
  }
  return v;
}

std::vector<Npp32s> make_ki(size_t n, uint64_t seed) {
  Rng rng(seed);
  std::vector<Npp32s> v(n);
  for (auto &x : v) x = rng.integer<int>(-8, 8);
  return v;
}

std::vector<Npp32s> make_kc(size_t n, Npp32s value) { return std::vector<Npp32s>(n, value); }

std::vector<Npp32f> make_kf(size_t n, uint64_t seed) {
  Rng rng(seed);
  std::vector<Npp32f> v(n);
  for (auto &x : v) x = static_cast<float>(rng.integer<int>(-8, 8)) * 0.125f;
  return v;
}

std::vector<Npp64f> make_kd(size_t n, uint64_t seed) {
  Rng rng(seed);
  std::vector<Npp64f> v(n);
  for (auto &x : v) x = static_cast<double>(rng.integer<int>(-8, 8)) * 0.125;
  return v;
}

template <typename D> ValueKind dst_kind() {
  if constexpr (std::is_same_v<D, Npp8u>) return ValueKind::U8;
  else if constexpr (std::is_same_v<D, Npp16u>) return ValueKind::U16;
  else if constexpr (std::is_same_v<D, Npp16s>) return ValueKind::I16;
  else if constexpr (std::is_same_v<D, Npp32f>) return ValueKind::F32;
  else return ValueKind::F64;
}

template <typename F>
void status_case(const std::string &family, const char *fname, const std::string &key, F &&f) {
  clear_cuda_error();
  NppStatus st = f();
  Case c(family + "/" + key, fname);
  c.status(st);
  c.finish();
}

constexpr int kMargin = 8;  // raw taps with |anchor| <= 4 and mask <= 5 stay inside

// 1D data case: the ROI sits inside a (W+2*margin)x(H+2*margin) buffer so the
// borderless raw reads stay inside the allocation.  dstep < 0 is used
// verbatim with the dst pointer moved to the end of the buffer.
template <typename S, typename D, typename K, typename F>
void rc_data(const std::string &family, const char *fname, const std::string &key, int W, int H,
             int ch, int mask, int anchor, int margin, uint64_t seed, int extreme, ValueKind kind,
             int dstep, const std::vector<K> &hk, int sstep, F &&f) {
  const int sPx = ch * (int)sizeof(S);
  const int dPx = ch * (int)sizeof(D);
  const int sStride = sstep ? align_step(sstep, sPx) : align_step((W + 2 * margin) * sPx + 3, sPx);
  const int dStride =
      dstep < 0 ? -dstep : (dstep ? align_step(dstep, dPx) : align_step(W * dPx + 3, dPx));
  const int allocStride = align_step((W + 2 * margin) * sPx + 3, sPx);
  const size_t se = ((size_t)(H + 2 * margin) * allocStride + (size_t)(W + 2 * margin) * sPx) / sizeof(S);
  const size_t de = (size_t)H * dStride / sizeof(D);
  const size_t dz = de ? de : 1;  // keep empty-ROI destinations non-null
  std::vector<S> hs = make_src<S>(se, seed + 11, extreme);
  std::vector<D> hd = make_uniform<D>(dz, seed + 29);
  DeviceMemory<S> &s = arena<S>(se);
  DeviceMemory<D> &d = arena<D>(dz);
  DeviceMemory<K> &k = arena<K>(hk.size());
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  k.copyFromHost(hk);
  clear_cuda_error();
  const S *ps = s.get() + (size_t)margin * (sStride / sizeof(S)) + (size_t)margin * ch;
  D *pd = (dstep < 0 && H > 0) ? d.get() + (size_t)(H - 1) * (dStride / sizeof(D)) : d.get();
  NppStatus st = f(ps, sStride, pd, dstep < 0 ? -dStride : dStride, NppiSize{W, H}, k.get(), mask,
                   anchor, current_ctx());
  Case c(family + "/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(S));
  c.input(hk.data(), hk.size() * sizeof(K));
  c.status(st);
  c.device_output<D>(d.get(), dz, kind);
  c.finish();
}

template <typename S, typename D, typename K, typename F>
void rc(const std::string &family, const char *fname, const std::string &key, int W, int H, int ch,
        int mask, int anchor, uint64_t seed, ValueKind kind, const std::vector<K> &hk, F &&f) {
  rc_data<S, D, K, F>(family, fname, key, W, H, ch, mask, anchor, kMargin, seed, 0, kind, 0, hk, 0,
                      std::forward<F>(f));
}

}  // namespace

class NppiFilterRowColDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
  int case_id_ = 0;
};

// ---------------------------------------------------------------------------
// Sweeps: every dtype x layout, mask 5, anchor 2, divisor 7.
// ---------------------------------------------------------------------------
#define ROW_DIV_RUN(S, TAG, CH)                                                                \
  rc<S, S, Npp32s>("nppiFilterRow", "nppiFilterRow_" #TAG, "sweep/" #TAG, 17, 13, CH, 5, 2,    \
                   0xE100 + case_id_++, dst_kind<S>(), make_ki(5, 0xE1FF),                     \
                   [](const S *s, int ss, S *d, int ds, NppiSize o, const Npp32s *k, int m,    \
                      int a, NppStreamContext c) {                                             \
                     return nppiFilterRow_##TAG##_Ctx(s, ss, d, ds, o, k, m, a, 7, c);          \
                   })
#define COL_DIV_RUN(S, TAG, CH)                                                                \
  rc<S, S, Npp32s>("nppiFilterColumn", "nppiFilterColumn_" #TAG, "sweep/" #TAG, 17, 13, CH, 5, 2, \
                   0xE200 + case_id_++, dst_kind<S>(), make_ki(5, 0xE2FF),                     \
                   [](const S *s, int ss, S *d, int ds, NppiSize o, const Npp32s *k, int m,    \
                      int a, NppStreamContext c) {                                             \
                     return nppiFilterColumn_##TAG##_Ctx(s, ss, d, ds, o, k, m, a, 7, c);       \
                   })
#define ROW_F32_RUN(TAG, CH)                                                                   \
  rc<Npp32f, Npp32f, Npp32f>("nppiFilterRow", "nppiFilterRow_" #TAG, "sweep/" #TAG, 17, 13, CH, 5, 2, \
                            0xE300 + case_id_++, dst_kind<Npp32f>(), make_kf(5, 0xE3FF),       \
                            [](const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize o,         \
                               const Npp32f *k, int m, int a, NppStreamContext c) {             \
                              return nppiFilterRow_##TAG##_Ctx(s, ss, d, ds, o, k, m, a, c);    \
                            })
#define COL_F32_RUN(TAG, CH)                                                                   \
  rc<Npp32f, Npp32f, Npp32f>("nppiFilterColumn", "nppiFilterColumn_" #TAG, "sweep/" #TAG, 17, 13, CH, 5, 2, \
                            0xE400 + case_id_++, dst_kind<Npp32f>(), make_kf(5, 0xE4FF),       \
                            [](const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize o,         \
                               const Npp32f *k, int m, int a, NppStreamContext c) {             \
                              return nppiFilterColumn_##TAG##_Ctx(s, ss, d, ds, o, k, m, a, c); \
                            })

#define DIV_SWEEP(FN)                                                                          \
  FN(Npp8u, 8u_C1R, 1);                                                                        \
  FN(Npp8u, 8u_C3R, 3);                                                                        \
  FN(Npp8u, 8u_C4R, 4);                                                                        \
  FN(Npp8u, 8u_AC4R, 4);                                                                       \
  FN(Npp16u, 16u_C1R, 1);                                                                      \
  FN(Npp16u, 16u_C3R, 3);                                                                      \
  FN(Npp16u, 16u_C4R, 4);                                                                      \
  FN(Npp16u, 16u_AC4R, 4);                                                                     \
  FN(Npp16s, 16s_C1R, 1);                                                                      \
  FN(Npp16s, 16s_C3R, 3);                                                                      \
  FN(Npp16s, 16s_C4R, 4);                                                                      \
  FN(Npp16s, 16s_AC4R, 4)

TEST_F(NppiFilterRowColDiff, DivSweep) { DIV_SWEEP(ROW_DIV_RUN); }
TEST_F(NppiFilterRowColDiff, ColumnDivSweep) { DIV_SWEEP(COL_DIV_RUN); }

TEST_F(NppiFilterRowColDiff, FloatSweep) {
  ROW_F32_RUN(32f_C1R, 1);
  ROW_F32_RUN(32f_C3R, 3);
  ROW_F32_RUN(32f_C4R, 4);
  ROW_F32_RUN(32f_AC4R, 4);
  COL_F32_RUN(32f_C1R, 1);
  COL_F32_RUN(32f_C3R, 3);
  COL_F32_RUN(32f_C4R, 4);
  COL_F32_RUN(32f_AC4R, 4);
  rc<Npp64f, Npp64f, Npp64f>("nppiFilterRow", "nppiFilterRow_64f_C1R", "sweep/64f_C1R", 17, 13, 1,
                             5, 2, 0xE500 + case_id_++, dst_kind<Npp64f>(), make_kd(5, 0xE5FF),
                             [](const Npp64f *s, int ss, Npp64f *d, int ds, NppiSize o,
                                const Npp64f *k, int m, int a, NppStreamContext c) {
                               return nppiFilterRow_64f_C1R_Ctx(s, ss, d, ds, o, k, m, a, c);
                             });
  rc<Npp64f, Npp64f, Npp64f>("nppiFilterColumn", "nppiFilterColumn_64f_C1R", "sweep/64f_C1R", 17,
                             13, 1, 5, 2, 0xE600 + case_id_++, dst_kind<Npp64f>(),
                             make_kd(5, 0xE6FF),
                             [](const Npp64f *s, int ss, Npp64f *d, int ds, NppiSize o,
                                const Npp64f *k, int m, int a, NppStreamContext c) {
                               return nppiFilterColumn_64f_C1R_Ctx(s, ss, d, ds, o, k, m, a, c);
                             });
}

// ---------------------------------------------------------------------------
// Geometry: anchors (including 1-W / 1-H limits and anchor == mask), mask
// sizes (even, larger than the ROI, zero, negative), empty ROIs, strides,
// divisors.
// ---------------------------------------------------------------------------
TEST_F(NppiFilterRowColDiff, DivGeometry) {
  auto row = [&](const char *tag, int W, int H, int mask, int anchor, int margin, uint64_t seed,
                 const std::vector<Npp32s> &hk, int div = 7, int dstep = 0, int sstep = 0,
                 int extreme = 0) {
    rc_data<Npp16u, Npp16u, Npp32s>(
        "nppiFilterRow", "nppiFilterRow_16u_C1R", std::string("geom/row/") + tag, W, H, 1, mask,
        anchor, margin, seed, extreme, dst_kind<Npp16u>(), dstep, hk, sstep,
        [div](const Npp16u *s, int ss, Npp16u *d, int ds, NppiSize o, const Npp32s *k, int m,
              int a, NppStreamContext c) {
          return nppiFilterRow_16u_C1R_Ctx(s, ss, d, ds, o, k, m, a, div, c);
        });
  };
  auto col = [&](const char *tag, int W, int H, int mask, int anchor, int margin, uint64_t seed,
                 const std::vector<Npp32s> &hk, int div = 7, int dstep = 0, int sstep = 0,
                 int extreme = 0) {
    rc_data<Npp16u, Npp16u, Npp32s>(
        "nppiFilterColumn", "nppiFilterColumn_16u_C1R", std::string("geom/col/") + tag, W, H, 1,
        mask, anchor, margin, seed, extreme, dst_kind<Npp16u>(), dstep, hk, sstep,
        [div](const Npp16u *s, int ss, Npp16u *d, int ds, NppiSize o, const Npp32s *k, int m,
              int a, NppStreamContext c) {
          return nppiFilterColumn_16u_C1R_Ctx(s, ss, d, ds, o, k, m, a, div, c);
        });
  };
  const int m = kMargin;
  row("a0", 17, 13, 5, 0, m, 0x7101, make_ki(5, 0x7101));
  row("amax", 17, 13, 5, 5, m, 0x7102, make_ki(5, 0x7102));  // anchor == mask
  row("aneg", 17, 13, 5, -3, m, 0x7103, make_ki(5, 0x7103));
  row("aext", 5, 5, 3, -4, m, 0x7104, make_ki(3, 0x7104));  // anchor == 1-W
  row("asym", 17, 13, 3, 0, m, 0x7105, std::vector<Npp32s>{1, 2, 4});
  row("mask1", 17, 13, 1, 0, m, 0x7106, make_ki(1, 0x7106));
  row("mask2", 17, 13, 2, 1, m, 0x7107, make_ki(2, 0x7107));  // even mask
  row("mask9", 17, 13, 9, 4, m, 0x7108, make_ki(9, 0x7108));
  row("mask23", 17, 13, 23, 11, 12, 0x7109, make_ki(23, 0x7109));  // mask > ROI width
  row("mask0", 17, 13, 0, 0, m, 0x710A, make_ki(1, 0x710A));
  row("small", 1, 1, 3, 1, m, 0x710B, make_ki(3, 0x710B));
  row("wide", 33, 1, 5, 2, m, 0x710C, make_ki(5, 0x710C));
  row("tall", 1, 33, 5, 2, m, 0x710D, make_ki(5, 0x710D));
  row("emptyw", 0, 13, 3, 1, m, 0x710E, make_ki(3, 0x710E));
  row("emptyh", 17, 0, 5, 2, m, 0x710F, make_ki(5, 0x710F));
  row("negstep", 17, 13, 5, 2, m, 0x7110, make_ki(5, 0x7110), 7, -48);
  row("bigstep", 17, 13, 5, 2, m, 0x7111, make_ki(5, 0x7111), 7, 98);
  row("sstep_small", 8, 8, 3, 1, m, 0x7112, make_ki(3, 0x7112), 7, 0, 2);
  row("div1", 17, 13, 5, 2, m, 0x7113, make_ki(5, 0x7113), 1);
  row("divneg", 17, 13, 5, 2, m, 0x7114, make_ki(5, 0x7114), -3);
  row("divbig", 17, 13, 5, 2, m, 0x7115, make_ki(5, 0x7115), 1000003);
  col("a0", 17, 13, 5, 0, m, 0x7121, make_ki(5, 0x7121));
  col("amax", 17, 13, 5, 5, m, 0x7122, make_ki(5, 0x7122));
  col("aneg", 17, 13, 5, -3, m, 0x7123, make_ki(5, 0x7123));
  col("aext", 5, 5, 3, -4, m, 0x7124, make_ki(3, 0x7124));  // anchor == 1-H
  col("asym", 17, 13, 3, 0, m, 0x7125, std::vector<Npp32s>{1, 2, 4});
  col("mask1", 17, 13, 1, 0, m, 0x7126, make_ki(1, 0x7126));
  col("mask2", 17, 13, 2, 1, m, 0x7127, make_ki(2, 0x7127));
  col("mask9", 17, 13, 9, 4, m, 0x7128, make_ki(9, 0x7128));
  col("mask23", 13, 17, 23, 11, 12, 0x7129, make_ki(23, 0x7129));  // mask > ROI height
  col("mask0", 17, 13, 0, 0, m, 0x712A, make_ki(1, 0x712A));
  col("small", 1, 1, 3, 1, m, 0x712B, make_ki(3, 0x712B));
  col("wide", 33, 1, 5, 2, m, 0x712C, make_ki(5, 0x712C));
  col("tall", 1, 33, 5, 2, m, 0x712D, make_ki(5, 0x712D));
  col("emptyw", 0, 13, 3, 1, m, 0x712E, make_ki(3, 0x712E));
  col("emptyh", 17, 0, 3, 1, m, 0x712F, make_ki(3, 0x712F));
  col("negstep", 17, 13, 5, 2, m, 0x7130, make_ki(5, 0x7130), 7, -48);
  col("divneg", 17, 13, 5, 2, m, 0x7131, make_ki(5, 0x7131), -3);
}

TEST_F(NppiFilterRowColDiff, FloatGeometry) {
  auto row32 = [&](const char *tag, int W, int H, int mask, int anchor, int margin, uint64_t seed,
                   const std::vector<Npp32f> &hk, int dstep = 0, int sstep = 0) {
    rc_data<Npp32f, Npp32f, Npp32f>(
        "nppiFilterRow", "nppiFilterRow_32f_C3R", std::string("geom/32f_row/") + tag, W, H, 3, mask,
        anchor, margin, seed, 0, dst_kind<Npp32f>(), dstep, hk, sstep,
        [](const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize o, const Npp32f *k, int m, int a,
           NppStreamContext c) {
          return nppiFilterRow_32f_C3R_Ctx(s, ss, d, ds, o, k, m, a, c);
        });
  };
  auto col32 = [&](const char *tag, int W, int H, int mask, int anchor, int margin, uint64_t seed,
                   const std::vector<Npp32f> &hk) {
    rc_data<Npp32f, Npp32f, Npp32f>(
        "nppiFilterColumn", "nppiFilterColumn_32f_C3R", std::string("geom/32f_col/") + tag, W, H, 3,
        mask, anchor, margin, seed, 0, dst_kind<Npp32f>(), 0, hk, 0,
        [](const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize o, const Npp32f *k, int m, int a,
           NppStreamContext c) {
          return nppiFilterColumn_32f_C3R_Ctx(s, ss, d, ds, o, k, m, a, c);
        });
  };
  const int m = kMargin;
  row32("a0", 17, 13, 5, 0, m, 0x7201, make_kf(5, 0x7201));
  row32("amax", 17, 13, 5, 5, m, 0x7202, make_kf(5, 0x7202));
  row32("aext", 5, 5, 3, -4, m, 0x7203, make_kf(3, 0x7203));
  row32("asym", 17, 13, 3, 0, m, 0x7204, std::vector<Npp32f>{1.0f, 2.0f, 4.0f});
  row32("mask2", 17, 13, 2, 1, m, 0x7205, make_kf(2, 0x7205));
  row32("mask9", 17, 13, 9, 4, m, 0x7206, make_kf(9, 0x7206));
  row32("mask0", 17, 13, 0, 0, m, 0x7207, make_kf(1, 0x7207));
  row32("emptyw", 0, 13, 3, 1, m, 0x7208, make_kf(3, 0x7208));
  row32("negstep", 17, 13, 5, 2, m, 0x7209, make_kf(5, 0x7209), -48);
  row32("sstep_small", 8, 8, 3, 1, m, 0x720A, make_kf(3, 0x720A), 0, 4);
  col32("a0", 17, 13, 5, 0, m, 0x7211, make_kf(5, 0x7211));
  col32("amax", 17, 13, 5, 5, m, 0x7212, make_kf(5, 0x7212));
  col32("aext", 5, 5, 3, -4, m, 0x7213, make_kf(3, 0x7213));
  col32("asym", 17, 13, 3, 0, m, 0x7214, std::vector<Npp32f>{1.0f, 2.0f, 4.0f});
  col32("mask0", 17, 13, 0, 0, m, 0x7215, make_kf(1, 0x7215));
  col32("emptyh", 17, 0, 3, 1, m, 0x7216, make_kf(3, 0x7216));
  rc<Npp64f, Npp64f, Npp64f>("nppiFilterRow", "nppiFilterRow_64f_C1R", "geom/64f_row/asym", 17, 13,
                             1, 3, 0, 0x7301, dst_kind<Npp64f>(),
                             std::vector<Npp64f>{1.0, 2.0, 4.0},
                             [](const Npp64f *s, int ss, Npp64f *d, int ds, NppiSize o,
                                const Npp64f *k, int m, int a, NppStreamContext c) {
                               return nppiFilterRow_64f_C1R_Ctx(s, ss, d, ds, o, k, m, a, c);
                             });
  rc<Npp64f, Npp64f, Npp64f>("nppiFilterRow", "nppiFilterRow_64f_C1R", "geom/64f_row/mask0", 17,
                             13, 1, 0, 0, 0x7302, dst_kind<Npp64f>(), make_kd(1, 0x7302),
                             [](const Npp64f *s, int ss, Npp64f *d, int ds, NppiSize o,
                                const Npp64f *k, int m, int a, NppStreamContext c) {
                               return nppiFilterRow_64f_C1R_Ctx(s, ss, d, ds, o, k, m, a, c);
                             });
  rc<Npp64f, Npp64f, Npp64f>("nppiFilterColumn", "nppiFilterColumn_64f_C1R", "geom/64f_col/asym",
                             17, 13, 1, 3, 0, 0x7303, dst_kind<Npp64f>(),
                             std::vector<Npp64f>{1.0, 2.0, 4.0},
                             [](const Npp64f *s, int ss, Npp64f *d, int ds, NppiSize o,
                                const Npp64f *k, int m, int a, NppStreamContext c) {
                               return nppiFilterColumn_64f_C1R_Ctx(s, ss, d, ds, o, k, m, a, c);
                             });
  rc<Npp64f, Npp64f, Npp64f>("nppiFilterColumn", "nppiFilterColumn_64f_C1R", "geom/64f_col/mask0",
                             17, 13, 1, 0, 0, 0x7304, dst_kind<Npp64f>(), make_kd(1, 0x7304),
                             [](const Npp64f *s, int ss, Npp64f *d, int ds, NppiSize o,
                                const Npp64f *k, int m, int a, NppStreamContext c) {
                               return nppiFilterColumn_64f_C1R_Ctx(s, ss, d, ds, o, k, m, a, c);
                             });
}

// AC4 alpha handling on the separable families.
TEST_F(NppiFilterRowColDiff, LayoutGeometry) {
  auto ac4row = [&](const char *tag, int mask, int anchor, uint64_t seed,
                    const std::vector<Npp32s> &hk, int div = 7) {
    rc<Npp16s, Npp16s, Npp32s>("nppiFilterRow", "nppiFilterRow_16s_AC4R",
                               std::string("geom/16s_AC4_row/") + tag, 17, 13, 4, mask, anchor,
                               seed, dst_kind<Npp16s>(), hk,
                               [div](const Npp16s *s, int ss, Npp16s *d, int ds, NppiSize o,
                                     const Npp32s *k, int m, int a, NppStreamContext c) {
                                 return nppiFilterRow_16s_AC4R_Ctx(s, ss, d, ds, o, k, m, a, div, c);
                               });
  };
  auto ac4col = [&](const char *tag, int mask, int anchor, uint64_t seed,
                    const std::vector<Npp32s> &hk, int div = 7) {
    rc<Npp16s, Npp16s, Npp32s>("nppiFilterColumn", "nppiFilterColumn_16s_AC4R",
                               std::string("geom/16s_AC4_col/") + tag, 17, 13, 4, mask, anchor,
                               seed, dst_kind<Npp16s>(), hk,
                               [div](const Npp16s *s, int ss, Npp16s *d, int ds, NppiSize o,
                                     const Npp32s *k, int m, int a, NppStreamContext c) {
                                 return nppiFilterColumn_16s_AC4R_Ctx(s, ss, d, ds, o, k, m, a, div,
                                                                      c);
                               });
  };
  ac4row("a0", 5, 0, 0x7401, make_ki(5, 0x7401));
  ac4row("amax", 5, 5, 0x7402, make_ki(5, 0x7402));
  ac4row("asym", 3, 0, 0x7403, std::vector<Npp32s>{1, 2, 4});
  ac4row("mask0", 0, 0, 0x7404, make_ki(1, 0x7404));
  ac4row("emptyw", 3, 1, 0x7405, make_ki(3, 0x7405));
  ac4row("divneg", 5, 2, 0x7406, make_ki(5, 0x7406), -3);
  ac4col("a0", 5, 0, 0x7411, make_ki(5, 0x7411));
  ac4col("mask0", 0, 0, 0x7412, make_ki(1, 0x7412));
  ac4col("divneg", 5, 2, 0x7413, make_ki(5, 0x7413), -3);
}

// ---------------------------------------------------------------------------
// Integer ring arithmetic on single-tap and constant kernels.
// ---------------------------------------------------------------------------
TEST_F(NppiFilterRowColDiff, DivArithmetic) {
  auto prow = [&](const char *tag, const std::vector<Npp32s> &hk, int mask, int anchor, int div,
                  int extreme) {
    rc_data<Npp16s, Npp16s, Npp32s>(
        "nppiFilterRow", "nppiFilterRow_16s_C1R", std::string("arith/row/") + tag, 9, 7, 1, mask,
        anchor, kMargin, 0x7500 + case_id_++, extreme, dst_kind<Npp16s>(), 0, hk, 0,
        [div](const Npp16s *s, int ss, Npp16s *d, int ds, NppiSize o, const Npp32s *k, int m,
              int a, NppStreamContext c) {
          return nppiFilterRow_16s_C1R_Ctx(s, ss, d, ds, o, k, m, a, div, c);
        });
  };
  auto pcol = [&](const char *tag, const std::vector<Npp32s> &hk, int mask, int anchor, int div,
                  int extreme) {
    rc_data<Npp16s, Npp16s, Npp32s>(
        "nppiFilterColumn", "nppiFilterColumn_16s_C1R", std::string("arith/col/") + tag, 9, 7, 1,
        mask, anchor, kMargin, 0x7600 + case_id_++, extreme, dst_kind<Npp16s>(), 0, hk, 0,
        [div](const Npp16s *s, int ss, Npp16s *d, int ds, NppiSize o, const Npp32s *k, int m,
              int a, NppStreamContext c) {
          return nppiFilterColumn_16s_C1R_Ctx(s, ss, d, ds, o, k, m, a, div, c);
        });
  };
  prow("wrap", make_kc(1, 1 << 20), 1, 0, 1, 0);
  prow("wrapmin", make_kc(1, -2147483647 - 1), 1, 0, 1, 8);
  prow("divm1", make_kc(1, 1), 1, 0, -1, 8);
  prow("trunc", make_kc(1, 4), 1, 0, 3, 8);
  prow("truncneg", make_kc(1, -7), 1, 0, 2, 0);
  prow("sat", make_kc(9, 4096), 9, 4, 1, 0);
  pcol("wrap", make_kc(1, 1 << 20), 1, 0, 1, 0);
  pcol("divm1", make_kc(1, 1), 1, 0, -1, 8);
  pcol("trunc", make_kc(1, 4), 1, 0, 3, 8);
  pcol("sat", make_kc(9, 4096), 9, 4, 1, 0);
  auto p8row = [&](const char *tag, Npp32s kv, int div) {
    rc<Npp8u, Npp8u, Npp32s>("nppiFilterRow", "nppiFilterRow_8u_C4R",
                             std::string("arith/8u_C4_row/") + tag, 9, 7, 4, 1, 0,
                             0x7700 + case_id_++, dst_kind<Npp8u>(), std::vector<Npp32s>{kv},
                             [div](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o,
                                   const Npp32s *k, int m, int a, NppStreamContext c) {
                               return nppiFilterRow_8u_C4R_Ctx(s, ss, d, ds, o, k, m, a, div, c);
                             });
  };
  p8row("sat_hi", 64, 1);
  p8row("sat_lo", -64, 1);
  p8row("trunc", 1, 3);
  rc<Npp8u, Npp8u, Npp32s>("nppiFilterColumn", "nppiFilterColumn_8u_C4R", "arith/8u_C4_col/sat_hi",
                           9, 7, 4, 1, 0, 0x7710, dst_kind<Npp8u>(), std::vector<Npp32s>{64},
                           [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o,
                              const Npp32s *k, int m, int a, NppStreamContext c) {
                             return nppiFilterColumn_8u_C4R_Ctx(s, ss, d, ds, o, k, m, a, 1, c);
                           });
}

// ---------------------------------------------------------------------------
// Error statuses and validation order.
// ---------------------------------------------------------------------------
TEST_F(NppiFilterRowColDiff, DivErrors) {
  const auto hk = make_ki(5, 0x7801);
  DeviceMemory<Npp16s> &s = arena<Npp16s>(4096);
  DeviceMemory<Npp16s> &d = arena<Npp16s>(4096);
  DeviceMemory<Npp32s> &k = arena<Npp32s>(5);
  k.copyFromHost(hk);
  const Npp16s *ps = s.get();
  Npp16s *pd = d.get();
  const Npp32s *pk = k.get();
  auto ctx = current_ctx();
  const NppiSize roi{8, 8};
  auto row = [&](const Npp16s *src, int ss, Npp16s *dst, int ds, NppiSize o, int m, int a,
                 int div) { return nppiFilterRow_16s_C1R_Ctx(src, ss, dst, ds, o, pk, m, a, div, ctx); };
  auto col = [&](const Npp16s *src, int ss, Npp16s *dst, int ds, NppiSize o, int m, int a,
                 int div) {
    return nppiFilterColumn_16s_C1R_Ctx(src, ss, dst, ds, o, pk, m, a, div, ctx);
  };
  status_case("nppiFilterRow", "nppiFilterRow_16s_C1R", "err/null_src",
              [&] { return row(nullptr, 16, pd, 16, roi, 3, 1, 7); });
  status_case("nppiFilterRow", "nppiFilterRow_16s_C1R", "err/null_dst",
              [&] { return row(ps, 16, nullptr, 16, roi, 3, 1, 7); });
  status_case("nppiFilterRow", "nppiFilterRow_16s_C1R", "err/neg_roi",
              [&] { return row(ps, 16, pd, 16, NppiSize{-1, 8}, 3, 1, 7); });
  status_case("nppiFilterRow", "nppiFilterRow_16s_C1R", "err/neg_roi_vs_anchor",
              [&] { return row(ps, 16, pd, 16, NppiSize{-1, 8}, 3, 9, 7); });
  status_case("nppiFilterRow", "nppiFilterRow_16s_C1R", "err/anchor_high",
              [&] { return row(ps, 16, pd, 16, roi, 3, 4, 7); });
  status_case("nppiFilterRow", "nppiFilterRow_16s_C1R", "err/anchor_low",
              [&] { return row(ps, 16, pd, 16, roi, 3, -8, 7); });
  status_case("nppiFilterRow", "nppiFilterRow_16s_C1R", "err/anchor_high_vs_step",
              [&] { return row(ps, 16, pd, 15, roi, 3, 4, 7); });
  status_case("nppiFilterRow", "nppiFilterRow_16s_C1R", "err/anchor_high_vs_div0",
              [&] { return row(ps, 16, pd, 16, roi, 3, 4, 0); });
  status_case("nppiFilterRow", "nppiFilterRow_16s_C1R", "err/step_small",
              [&] { return row(ps, 16, pd, 15, roi, 3, 1, 7); });
  status_case("nppiFilterRow", "nppiFilterRow_16s_C1R", "err/step_zero",
              [&] { return row(ps, 16, pd, 0, roi, 3, 1, 7); });
  status_case("nppiFilterRow", "nppiFilterRow_16s_C1R", "err/step_small_vs_div0",
              [&] { return row(ps, 16, pd, 15, roi, 3, 1, 0); });
  status_case("nppiFilterRow", "nppiFilterRow_16s_C1R", "err/div0",
              [&] { return row(ps, 16, pd, 16, roi, 3, 1, 0); });
  status_case("nppiFilterRow", "nppiFilterRow_16s_C1R", "err/div0_vs_negroi",
              [&] { return row(ps, 16, pd, 16, NppiSize{-1, 8}, 3, 1, 0); });
  status_case("nppiFilterRow", "nppiFilterRow_16s_C1R", "err/div0_vs_empty",
              [&] { return row(ps, 16, pd, 16, NppiSize{0, 8}, 3, 1, 0); });
  status_case("nppiFilterRow", "nppiFilterRow_16s_C1R", "err/div0_vs_mask0",
              [&] { return row(ps, 16, pd, 16, roi, 0, 0, 0); });
  status_case("nppiFilterRow", "nppiFilterRow_16s_C1R", "err/mask_neg_anchor0",
              [&] { return row(ps, 16, pd, 16, roi, -1, 0, 7); });
  status_case("nppiFilterRow", "nppiFilterRow_16s_C1R", "ok/mask_neg_anchor_ok",
              [&] { return row(ps, 16, pd, 16, roi, -1, -1, 7); });
  status_case("nppiFilterRow", "nppiFilterRow_16s_C1R", "ok/neg_step",
              [&] { return row(ps, 16, pd + 3000, -64, roi, 3, 1, 7); });
  status_case("nppiFilterRow", "nppiFilterRow_16s_C1R", "ok/empty_roi",
              [&] { return row(ps, 16, pd, 16, NppiSize{0, 8}, 3, 1, 7); });
  status_case("nppiFilterColumn", "nppiFilterColumn_16s_C1R", "err/null_src",
              [&] { return col(nullptr, 16, pd, 16, roi, 3, 1, 7); });
  status_case("nppiFilterColumn", "nppiFilterColumn_16s_C1R", "err/null_dst",
              [&] { return col(ps, 16, nullptr, 16, roi, 3, 1, 7); });
  status_case("nppiFilterColumn", "nppiFilterColumn_16s_C1R", "err/anchor_high",
              [&] { return col(ps, 16, pd, 16, roi, 3, 4, 7); });
  status_case("nppiFilterColumn", "nppiFilterColumn_16s_C1R", "err/anchor_low",
              [&] { return col(ps, 16, pd, 16, roi, 3, -8, 7); });
  status_case("nppiFilterColumn", "nppiFilterColumn_16s_C1R", "err/neg_roi",
              [&] { return col(ps, 16, pd, 16, NppiSize{8, -1}, 3, 1, 7); });
  status_case("nppiFilterColumn", "nppiFilterColumn_16s_C1R", "err/step_small",
              [&] { return col(ps, 16, pd, 15, roi, 3, 1, 7); });
  status_case("nppiFilterColumn", "nppiFilterColumn_16s_C1R", "err/div0",
              [&] { return col(ps, 16, pd, 16, roi, 3, 1, 0); });
  status_case("nppiFilterColumn", "nppiFilterColumn_16s_C1R", "err/div0_vs_empty",
              [&] { return col(ps, 16, pd, 16, NppiSize{8, 0}, 3, 1, 0); });
  status_case("nppiFilterColumn", "nppiFilterColumn_16s_C1R", "err/mask_neg_anchor0",
              [&] { return col(ps, 16, pd, 16, roi, -1, 0, 7); });
  status_case("nppiFilterColumn", "nppiFilterColumn_16s_C1R", "ok/mask_neg_anchor_ok",
              [&] { return col(ps, 16, pd, 16, roi, -1, -1, 7); });
  status_case("nppiFilterColumn", "nppiFilterColumn_16s_C1R", "ok/neg_step",
              [&] { return col(ps, 16, pd + 3000, -64, roi, 3, 1, 7); });
  status_case("nppiFilterColumn", "nppiFilterColumn_16s_C1R", "ok/empty_roi",
              [&] { return col(ps, 16, pd, 16, NppiSize{8, 0}, 3, 1, 7); });
}

TEST_F(NppiFilterRowColDiff, FloatErrors) {
  DeviceMemory<Npp32f> &s = arena<Npp32f>(4096);
  DeviceMemory<Npp32f> &d = arena<Npp32f>(4096);
  DeviceMemory<Npp32f> &k = arena<Npp32f>(5);
  k.copyFromHost(make_kf(5, 0x7901));
  const Npp32f *ps = s.get();
  Npp32f *pd = d.get();
  const Npp32f *pk = k.get();
  auto ctx = current_ctx();
  const NppiSize roi{8, 8};
  auto row = [&](const Npp32f *src, int ss, Npp32f *dst, int ds, NppiSize o, int m, int a) {
    return nppiFilterRow_32f_C3R_Ctx(src, ss, dst, ds, o, pk, m, a, ctx);
  };
  auto col = [&](const Npp32f *src, int ss, Npp32f *dst, int ds, NppiSize o, int m, int a) {
    return nppiFilterColumn_32f_C3R_Ctx(src, ss, dst, ds, o, pk, m, a, ctx);
  };
  status_case("nppiFilterRow", "nppiFilterRow_32f_C3R", "err/null_src",
              [&] { return row(nullptr, 96, pd, 96, roi, 3, 1); });
  status_case("nppiFilterRow", "nppiFilterRow_32f_C3R", "err/anchor_high",
              [&] { return row(ps, 96, pd, 96, roi, 3, 4); });
  status_case("nppiFilterRow", "nppiFilterRow_32f_C3R", "err/anchor_low",
              [&] { return row(ps, 96, pd, 96, roi, 3, -8); });
  status_case("nppiFilterRow", "nppiFilterRow_32f_C3R", "err/anchor_high_vs_step",
              [&] { return row(ps, 96, pd, 95, roi, 3, 4); });
  status_case("nppiFilterRow", "nppiFilterRow_32f_C3R", "err/step_small",
              [&] { return row(ps, 96, pd, 95, roi, 3, 1); });
  status_case("nppiFilterRow", "nppiFilterRow_32f_C3R", "err/step_zero",
              [&] { return row(ps, 96, pd, 0, roi, 3, 1); });
  status_case("nppiFilterRow", "nppiFilterRow_32f_C3R", "ok/mask_neg_anchor_ok",
              [&] { return row(ps, 96, pd, 96, roi, -1, -1); });
  status_case("nppiFilterRow", "nppiFilterRow_32f_C3R", "ok/neg_step",
              [&] { return row(ps, 96, pd + 3000, -192, roi, 3, 1); });
  status_case("nppiFilterRow", "nppiFilterRow_32f_C3R", "ok/empty_roi",
              [&] { return row(ps, 96, pd, 96, NppiSize{0, 8}, 3, 1); });
  status_case("nppiFilterColumn", "nppiFilterColumn_32f_C3R", "err/null_dst",
              [&] { return col(ps, 96, nullptr, 96, roi, 3, 1); });
  status_case("nppiFilterColumn", "nppiFilterColumn_32f_C3R", "err/anchor_high",
              [&] { return col(ps, 96, pd, 96, roi, 3, 4); });
  status_case("nppiFilterColumn", "nppiFilterColumn_32f_C3R", "err/step_small",
              [&] { return col(ps, 96, pd, 95, roi, 3, 1); });
  status_case("nppiFilterColumn", "nppiFilterColumn_32f_C3R", "ok/empty_roi",
              [&] { return col(ps, 96, pd, 96, NppiSize{8, 0}, 3, 1); });
  DeviceMemory<Npp64f> &s64 = arena<Npp64f>(4096);
  DeviceMemory<Npp64f> &d64 = arena<Npp64f>(4096);
  DeviceMemory<Npp64f> &k64 = arena<Npp64f>(5);
  k64.copyFromHost(make_kd(5, 0x7902));
  const Npp64f *ps64 = s64.get();
  Npp64f *pd64 = d64.get();
  const Npp64f *pk64 = k64.get();
  status_case("nppiFilterRow", "nppiFilterRow_64f_C1R", "err/null_src", [&] {
    return nppiFilterRow_64f_C1R_Ctx(nullptr, 64, pd64, 64, roi, pk64, 3, 1, ctx);
  });
  status_case("nppiFilterRow", "nppiFilterRow_64f_C1R", "err/anchor_high", [&] {
    return nppiFilterRow_64f_C1R_Ctx(ps64, 64, pd64, 64, roi, pk64, 3, 4, ctx);
  });
  status_case("nppiFilterRow", "nppiFilterRow_64f_C1R", "err/step_small", [&] {
    return nppiFilterRow_64f_C1R_Ctx(ps64, 64, pd64, 63, roi, pk64, 3, 1, ctx);
  });
  status_case("nppiFilterRow", "nppiFilterRow_64f_C1R", "ok/empty_roi", [&] {
    return nppiFilterRow_64f_C1R_Ctx(ps64, 64, pd64, 64, NppiSize{0, 8}, pk64, 3, 1, ctx);
  });
  status_case("nppiFilterColumn", "nppiFilterColumn_64f_C1R", "err/anchor_high", [&] {
    return nppiFilterColumn_64f_C1R_Ctx(ps64, 64, pd64, 64, roi, pk64, 3, 4, ctx);
  });
  status_case("nppiFilterColumn", "nppiFilterColumn_64f_C1R", "ok/mask0", [&] {
    return nppiFilterColumn_64f_C1R_Ctx(ps64, 64, pd64, 64, roi, pk64, 0, 0, ctx);
  });
}
