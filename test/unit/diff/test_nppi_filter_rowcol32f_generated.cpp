// Differential tests for package f1: nppiFilterRow32f / nppiFilterColumn32f
// (separable 1D filters, 32f kernel, no border) and
// nppiFilterRowBorder32f / nppiFilterColumnBorder32f (with border handling).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiFilterRowCol32fDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppiFilterRowCol32fDiff.*'
//
// Kernels are dyadic (integer multiples of 1/8) and 16-bit sources use their
// full range: every float partial sum stays below 2^24, so the accumulation is
// exact regardless of the implementation's summation order.

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
  v.push_back(std::make_unique<DeviceMemory<T>>(n));
  return *v.back();
}

NppStreamContext current_ctx() {
  NppStreamContext c{};
  NppStatus st = nppGetStreamContext(&c);
  EXPECT_EQ(st, NPP_NO_ERROR);
  return c;
}

int align_step(int step, int es) { return ((step + es - 1) / es) * es; }

template <typename T> std::vector<T> make_src(size_t n, uint64_t seed) {
  Rng rng(seed);
  std::vector<T> v(n);
  for (auto &x : v) {
    if constexpr (std::is_integral_v<T>) {
      x = rng.integer<T>(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
    }
  }
  return v;
}

std::vector<Npp32f> make_kf(size_t n, uint64_t seed) {
  Rng rng(seed);
  std::vector<Npp32f> v(n);
  for (auto &x : v) x = static_cast<float>(rng.integer<int>(-8, 8)) * 0.125f;
  return v;
}

template <typename D> ValueKind dst_kind() {
  if constexpr (std::is_same_v<D, Npp8u>) return ValueKind::U8;
  else if constexpr (std::is_same_v<D, Npp16u>) return ValueKind::U16;
  else return ValueKind::I16;
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

// ---------------------------------------------------------------------------
// Row/column without border: raw reads may leave the ROI, so the source ROI
// sits inside a (W+2M)x(H+2M) buffer.
// ---------------------------------------------------------------------------
template <typename S, typename D, typename F>
void rc_data(const std::string &family, const char *fname, const std::string &key, int W, int H,
             int ch, int mask, int anchor, uint64_t seed, ValueKind kind,
             const std::vector<Npp32f> &hk, F &&f) {
  const int sPx = ch * (int)sizeof(S);
  const int dPx = ch * (int)sizeof(D);
  const int sStep = align_step((W + 2 * kMargin) * sPx + 3, sPx);
  const int dStep = align_step(W * dPx + 3, dPx);
  const size_t se = (size_t)(H + 2 * kMargin) * sStep / sizeof(S);
  const size_t de = (size_t)H * dStep / sizeof(D);
  std::vector<S> hs = make_src<S>(se, seed + 11);
  std::vector<D> hd = make_uniform<D>(de, seed + 29);
  DeviceMemory<S> &s = arena<S>(se);
  DeviceMemory<D> &d = arena<D>(de);
  DeviceMemory<Npp32f> &k = arena<Npp32f>(hk.size());
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  k.copyFromHost(hk);
  clear_cuda_error();
  const S *ps = s.get() + (size_t)kMargin * (sStep / sizeof(S)) + (size_t)kMargin * ch;
  NppStatus st = f(ps, sStep, d.get(), dStep, NppiSize{W, H}, k.get(), mask, anchor, current_ctx());
  Case c(family + "/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(S));
  c.input(hk.data(), hk.size() * sizeof(Npp32f));
  c.status(st);
  c.device_output<D>(d.get(), de, kind);
  c.finish();
}

// ---------------------------------------------------------------------------
// Row/column with replicate border: reads are clamped into the source image,
// and the non-filtered axis is clamped as well (measured NPP 13.4 behaviour).
// ---------------------------------------------------------------------------
template <typename S, typename D, typename F>
void rcb_data(const std::string &family, const char *fname, const std::string &key, int W, int H,
              int ch, int sw, int sh, int offx, int offy, int mask, int anchor, uint64_t seed,
              ValueKind kind, const std::vector<Npp32f> &hk, F &&f) {
  const int sPx = ch * (int)sizeof(S);
  const int dPx = ch * (int)sizeof(D);
  const int sStep = align_step(sw * sPx + 3, sPx);
  const int dStep = align_step(W * dPx + 3, dPx);
  const size_t se = (size_t)sh * sStep / sizeof(S);
  const size_t de = (size_t)H * dStep / sizeof(D);
  std::vector<S> hs = make_src<S>(se, seed + 11);
  std::vector<D> hd = make_uniform<D>(de, seed + 29);
  DeviceMemory<S> &s = arena<S>(se);
  DeviceMemory<D> &d = arena<D>(de);
  DeviceMemory<Npp32f> &k = arena<Npp32f>(hk.size());
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  k.copyFromHost(hk);
  clear_cuda_error();
  const S *ps = s.get() + (size_t)offy * (sStep / sizeof(S)) + (size_t)offx * ch;
  NppStatus st = f(ps, sStep, NppiSize{sw, sh}, NppiPoint{offx, offy}, d.get(), dStep,
                   NppiSize{W, H}, k.get(), mask, anchor, NPP_BORDER_REPLICATE, current_ctx());
  Case c(family + "/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(S));
  c.input(hk.data(), hk.size() * sizeof(Npp32f));
  c.status(st);
  c.device_output<D>(d.get(), de, kind);
  c.finish();
}

#define ROW_RUN(S, D, TAG, CH)                                                              \
  rc_data<S, D>("nppiFilterRow32f", "nppiFilterRow32f_" #TAG, "sweep/" #TAG, 17, 13, CH, 5, 2, \
                0xF600 + case_id_++, dst_kind<D>(), make_kf(5, 0xF6FF),                     \
                [](const S *s, int ss, D *d, int ds, NppiSize o, const Npp32f *k, int m,     \
                   int a, NppStreamContext c) {                                              \
                  return nppiFilterRow32f_##TAG##_Ctx(s, ss, d, ds, o, k, m, a, c);          \
                })
#define COL_RUN(S, D, TAG, CH)                                                              \
  rc_data<S, D>("nppiFilterColumn32f", "nppiFilterColumn32f_" #TAG, "sweep/" #TAG, 17, 13, CH, 5, 2, \
                0xF700 + case_id_++, dst_kind<D>(), make_kf(5, 0xF7FF),                     \
                [](const S *s, int ss, D *d, int ds, NppiSize o, const Npp32f *k, int m,     \
                   int a, NppStreamContext c) {                                              \
                  return nppiFilterColumn32f_##TAG##_Ctx(s, ss, d, ds, o, k, m, a, c);       \
                })
#define ROWB_RUN(S, D, TAG, CH)                                                             \
  rcb_data<S, D>("nppiFilterRowBorder32f", "nppiFilterRowBorder32f_" #TAG, "sweep/" #TAG, 7, 7, CH, 9, 9, \
                 2, 3, 5, 2, 0xF800 + case_id_++, dst_kind<D>(), make_kf(5, 0xF8FF),         \
                 [](const S *s, int ss, NppiSize sz, NppiPoint so, D *d, int ds,            \
                    NppiSize o, const Npp32f *k, int m, int a, NppiBorderType bt,           \
                    NppStreamContext c) {                                                   \
                   return nppiFilterRowBorder32f_##TAG##_Ctx(s, ss, sz, so, d, ds, o, k, m, a, bt, c); \
                 })
#define COLB_RUN(S, D, TAG, CH)                                                             \
  rcb_data<S, D>("nppiFilterColumnBorder32f", "nppiFilterColumnBorder32f_" #TAG, "sweep/" #TAG, 7, 7, CH, \
                 9, 9, 2, 3, 5, 2, 0xF900 + case_id_++, dst_kind<D>(), make_kf(5, 0xF9FF),   \
                 [](const S *s, int ss, NppiSize sz, NppiPoint so, D *d, int ds,            \
                    NppiSize o, const Npp32f *k, int m, int a, NppiBorderType bt,           \
                    NppStreamContext c) {                                                   \
                   return nppiFilterColumnBorder32f_##TAG##_Ctx(s, ss, sz, so, d, ds, o, k, m, a, bt, c); \
                 })

}  // namespace

class NppiFilterRowCol32fDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
  int case_id_ = 0;
};

#define ROWCOL_SWEEP(FN)                                                                   \
  FN(Npp8u, Npp8u, 8u_C1R, 1);                                                             \
  FN(Npp8u, Npp8u, 8u_C3R, 3);                                                             \
  FN(Npp8u, Npp8u, 8u_C4R, 4);                                                             \
  FN(Npp8u, Npp8u, 8u_AC4R, 4);                                                            \
  FN(Npp16u, Npp16u, 16u_C1R, 1);                                                          \
  FN(Npp16u, Npp16u, 16u_C3R, 3);                                                          \
  FN(Npp16u, Npp16u, 16u_C4R, 4);                                                          \
  FN(Npp16u, Npp16u, 16u_AC4R, 4);                                                         \
  FN(Npp16s, Npp16s, 16s_C1R, 1);                                                          \
  FN(Npp16s, Npp16s, 16s_C3R, 3);                                                          \
  FN(Npp16s, Npp16s, 16s_C4R, 4);                                                          \
  FN(Npp16s, Npp16s, 16s_AC4R, 4)

TEST_F(NppiFilterRowCol32fDiff, Row32fSweep) { ROWCOL_SWEEP(ROW_RUN); }
TEST_F(NppiFilterRowCol32fDiff, Column32fSweep) { ROWCOL_SWEEP(COL_RUN); }
TEST_F(NppiFilterRowCol32fDiff, RowBorder32fSweep) { ROWCOL_SWEEP(ROWB_RUN); }
TEST_F(NppiFilterRowCol32fDiff, ColumnBorder32fSweep) { ROWCOL_SWEEP(COLB_RUN); }

// Anchor / mask / geometry variations for the non-border row/column filters.
TEST_F(NppiFilterRowCol32fDiff, RowColGeometry) {
  const auto k3 = make_kf(3, 0x6001);
  const auto k5 = make_kf(5, 0x6002);
  const auto k1 = make_kf(1, 0x6003);
  const auto k0 = make_kf(1, 0x6004);  // non-null buffer; a 0 mask never reads it
  auto row = [&](const char *tag, int W, int H, int ch, int m, int a, uint64_t seed,
                 const std::vector<Npp32f> &hk) {
    rc_data<Npp8u, Npp8u>("nppiFilterRow32f", "nppiFilterRow32f_8u_C3R",
                          std::string("row/") + tag + "/m" + std::to_string(m) + "/a" +
                              std::to_string(a),
                          W, H, ch, m, a, seed, ValueKind::U8, hk,
                          [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, const Npp32f *k,
                             int mm, int aa, NppStreamContext c) {
                            return nppiFilterRow32f_8u_C3R_Ctx(s, ss, d, ds, o, k, mm, aa, c);
                          });
  };
  auto col = [&](const char *tag, int W, int H, int ch, int m, int a, uint64_t seed,
                 const std::vector<Npp32f> &hk) {
    rc_data<Npp8u, Npp8u>("nppiFilterColumn32f", "nppiFilterColumn32f_8u_C3R",
                          std::string("col/") + tag + "/m" + std::to_string(m) + "/a" +
                              std::to_string(a),
                          W, H, ch, m, a, seed, ValueKind::U8, hk,
                          [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, const Npp32f *k,
                             int mm, int aa, NppStreamContext c) {
                            return nppiFilterColumn32f_8u_C3R_Ctx(s, ss, d, ds, o, k, mm, aa, c);
                          });
  };
  row("a0", 8, 8, 3, 3, 0, 0x6011, k3);
  row("amax", 8, 8, 3, 3, 3, 0x6012, k3);
  row("aneg", 8, 8, 3, 3, -4, 0x6013, k3);
  row("mask1", 8, 8, 3, 1, 0, 0x6014, k1);
  row("mask0", 8, 8, 3, 0, 0, 0x6015, k0);   // zero mask writes zeros
  row("small", 7, 5, 3, 5, 2, 0x6016, k5);
  row("one", 1, 1, 3, 3, 3, 0x6017, k3);     // 1x1 ROI, anchor at the limit
  row("wide", 33, 17, 3, 3, 1, 0x6018, k3);
  col("a0", 8, 8, 3, 3, 0, 0x6021, k3);
  col("amax", 8, 8, 3, 3, 3, 0x6022, k3);
  col("aneg", 8, 8, 3, 3, -4, 0x6023, k3);
  col("mask1", 8, 8, 3, 1, 0, 0x6024, k1);
  col("mask0", 8, 8, 3, 0, 0, 0x6025, k0);
  col("tall", 17, 33, 3, 5, 2, 0x6026, k5);
  col("one", 1, 1, 3, 3, 3, 0x6027, k3);
  // Empty ROIs (anchor >= 1 in the empty dimension).
  row("emptyw", 0, 4, 3, 3, 1, 0x6031, k3);
  col("emptyh", 4, 0, 3, 3, 1, 0x6032, k3);
}

// Truncation and saturation for the separable families (16u/16s included).
TEST_F(NppiFilterRowCol32fDiff, RowColTruncSaturate) {
  const auto k15 = std::vector<Npp32f>{1.5f};
  const auto k2 = std::vector<Npp32f>{2.0f};
  const auto km2 = std::vector<Npp32f>{-2.0f};
  const auto k4 = std::vector<Npp32f>{4.0f};
  // Row: single-tap kernels are pure per-pixel scaling.
  rc_data<Npp16s, Npp16s>("nppiFilterRow32f", "nppiFilterRow32f_16s_C1R", "sat/row16s", 6, 6, 1, 1,
                          0, 0x6101, ValueKind::I16, k2, [](const Npp16s *s, int ss, Npp16s *d,
                          int ds, NppiSize o, const Npp32f *k, int m, int a, NppStreamContext c) {
                            return nppiFilterRow32f_16s_C1R_Ctx(s, ss, d, ds, o, k, m, a, c);
                          });
  rc_data<Npp16u, Npp16u>("nppiFilterRow32f", "nppiFilterRow32f_16u_C1R", "sat/row16u", 6, 6, 1, 1,
                          0, 0x6102, ValueKind::U16, k4, [](const Npp16u *s, int ss, Npp16u *d,
                          int ds, NppiSize o, const Npp32f *k, int m, int a, NppStreamContext c) {
                            return nppiFilterRow32f_16u_C1R_Ctx(s, ss, d, ds, o, k, m, a, c);
                          });
  rc_data<Npp8u, Npp8u>("nppiFilterRow32f", "nppiFilterRow32f_8u_C1R", "trunc/row8u", 6, 6, 1, 1,
                        0, 0x6103, ValueKind::U8, std::vector<Npp32f>{0.5f},
                        [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, const Npp32f *k,
                           int m, int a, NppStreamContext c) {
                          return nppiFilterRow32f_8u_C1R_Ctx(s, ss, d, ds, o, k, m, a, c);
                        });
  rc_data<Npp16s, Npp16s>("nppiFilterColumn32f", "nppiFilterColumn32f_16s_C1R", "sat/col16s", 6, 6,
                          1, 1, 0, 0x6104, ValueKind::I16, km2, [](const Npp16s *s, int ss,
                          Npp16s *d, int ds, NppiSize o, const Npp32f *k, int m, int a,
                          NppStreamContext c) {
                            return nppiFilterColumn32f_16s_C1R_Ctx(s, ss, d, ds, o, k, m, a, c);
                          });
  rc_data<Npp16u, Npp16u>("nppiFilterColumn32f", "nppiFilterColumn32f_16u_C1R", "sat/col16u", 6, 6,
                          1, 1, 0, 0x6105, ValueKind::U16, std::vector<Npp32f>{3.0f},
                          [](const Npp16u *s, int ss, Npp16u *d, int ds, NppiSize o,
                             const Npp32f *k, int m, int a, NppStreamContext c) {
                            return nppiFilterColumn32f_16u_C1R_Ctx(s, ss, d, ds, o, k, m, a, c);
                          });
  rcb_data<Npp16u, Npp16u>("nppiFilterRowBorder32f", "nppiFilterRowBorder32f_16u_C1R", "sat/rowb16u",
                           6, 6, 1, 8, 8, 1, 1, 1, 0, 0x6106, ValueKind::U16, k4,
                          [](const Npp16u *s, int ss, NppiSize sz, NppiPoint so, Npp16u *d, int ds,
                             NppiSize o, const Npp32f *k, int m, int a, NppiBorderType bt,
                             NppStreamContext c) {
                            return nppiFilterRowBorder32f_16u_C1R_Ctx(s, ss, sz, so, d, ds, o, k, m,
                                                                       a, bt, c);
                          });
  rcb_data<Npp16s, Npp16s>("nppiFilterColumnBorder32f", "nppiFilterColumnBorder32f_16s_C1R",
                           "sat/colb16s", 6, 6, 1, 8, 8, 1, 1, 1, 0, 0x6107, ValueKind::I16, k15,
                          [](const Npp16s *s, int ss, NppiSize sz, NppiPoint so, Npp16s *d, int ds,
                             NppiSize o, const Npp32f *k, int m, int a, NppiBorderType bt,
                             NppStreamContext c) {
                            return nppiFilterColumnBorder32f_16s_C1R_Ctx(s, ss, sz, so, d, ds, o, k,
                                                                          m, a, bt, c);
                          });
}

// Border variants: offsets, ROI beyond the source, negative/zero masks
// (masks and anchors are NOT validated for the border families).
TEST_F(NppiFilterRowCol32fDiff, RowColBorderGeometry) {
  const auto k3 = make_kf(3, 0x6201);
  auto rowb = [&](const char *tag, int W, int H, int ch, int sw, int sh, int ox, int oy, int m,
                  int a, uint64_t seed, const auto &hk) {
    rcb_data<Npp8u, Npp8u>("nppiFilterRowBorder32f", "nppiFilterRowBorder32f_8u_C3R",
                           std::string("rowb/") + tag, W, H, ch, sw, sh, ox, oy, m, a, seed,
                           ValueKind::U8, hk,
                           [](const Npp8u *s, int ss, NppiSize sz, NppiPoint so, Npp8u *d, int ds,
                              NppiSize o, const Npp32f *k, int mm, int aa, NppiBorderType bt,
                              NppStreamContext c) {
                             return nppiFilterRowBorder32f_8u_C3R_Ctx(s, ss, sz, so, d, ds, o, k,
                                                                      mm, aa, bt, c);
                           });
  };
  auto colb = [&](const char *tag, int W, int H, int ch, int sw, int sh, int ox, int oy, int m,
                  int a, uint64_t seed, const auto &hk) {
    rcb_data<Npp8u, Npp8u>("nppiFilterColumnBorder32f", "nppiFilterColumnBorder32f_8u_C3R",
                           std::string("colb/") + tag, W, H, ch, sw, sh, ox, oy, m, a, seed,
                           ValueKind::U8, hk,
                           [](const Npp8u *s, int ss, NppiSize sz, NppiPoint so, Npp8u *d, int ds,
                              NppiSize o, const Npp32f *k, int mm, int aa, NppiBorderType bt,
                              NppStreamContext c) {
                             return nppiFilterColumnBorder32f_8u_C3R_Ctx(s, ss, sz, so, d, ds, o,
                                                                         k, mm, aa, bt, c);
                           });
  };
  rowb("offset", 5, 5, 3, 12, 10, 4, 6, 3, 1, 0x6211, k3);
  rowb("beyond", 9, 9, 3, 4, 4, 1, 1, 3, 1, 0x6212, k3);   // ROI larger than the source
  rowb("last_off", 3, 3, 3, 8, 8, 7, 7, 3, 1, 0x6213, k3); // offset in the last pixel
  rowb("anchor9", 4, 4, 3, 6, 6, 1, 1, 3, 9, 0x6214, k3);  // anchors are not validated
  rowb("mask0", 4, 4, 3, 6, 6, 1, 1, 0, 0, 0x6215, make_kf(1, 0x6215));  // zero mask -> zeros
  rowb("mask_neg", 4, 4, 3, 6, 6, 1, 1, -3, 0, 0x6216, make_kf(1, 0x6216));  // negative mask
  rowb("empty", 0, 4, 3, 6, 6, 1, 1, 3, 1, 0x6217, k3);    // empty ROI, untouched dst
  colb("offset", 5, 5, 3, 12, 10, 4, 6, 3, 1, 0x6221, k3);
  colb("beyond", 9, 9, 3, 4, 4, 1, 1, 3, 1, 0x6222, k3);
  colb("anchor9", 4, 4, 3, 6, 6, 1, 1, 3, 9, 0x6223, k3);
  colb("mask0", 4, 4, 3, 6, 6, 1, 1, 0, 0, 0x6224, make_kf(1, 0x6224));
  colb("empty", 4, 0, 3, 6, 6, 1, 1, 3, 1, 0x6225, k3);
}

// Error statuses for all four separable families.
TEST_F(NppiFilterRowCol32fDiff, RowColErrors) {
  const auto hk = make_kf(3, 0x6301);
  DeviceMemory<Npp8u> &s = arena<Npp8u>(4096);
  DeviceMemory<Npp8u> &d = arena<Npp8u>(4096);
  DeviceMemory<Npp32f> &k = arena<Npp32f>(3);
  k.copyFromHost(hk);
  // Status probes only: raw taps outside the ROI must stay inside the
  // allocation (MXC faults on reads before the buffer), so start at a margin.
  const Npp8u *ps = s.get() + 64;
  Npp8u *pd = d.get();
  const Npp32f *pk = k.get();
  auto ctx = current_ctx();
  const NppiSize roi{8, 8};
  const NppiSize sz{9, 9};
  const NppiPoint off{2, 3};

  status_case("nppiFilterRow32f", "nppiFilterRow32f_8u_C1R", "err/null_src", [&] {
    return nppiFilterRow32f_8u_C1R_Ctx(nullptr, 16, pd, 16, roi, pk, 3, 1, ctx);
  });
  status_case("nppiFilterRow32f", "nppiFilterRow32f_8u_C1R", "err/anchor_high", [&] {
    return nppiFilterRow32f_8u_C1R_Ctx(ps, 16, pd, 16, roi, pk, 3, 4, ctx);
  });
  status_case("nppiFilterRow32f", "nppiFilterRow32f_8u_C1R", "err/anchor_low", [&] {
    return nppiFilterRow32f_8u_C1R_Ctx(ps, 16, pd, 16, roi, pk, 3, -8, ctx);
  });
  status_case("nppiFilterRow32f", "nppiFilterRow32f_8u_C1R", "err/neg_roi", [&] {
    return nppiFilterRow32f_8u_C1R_Ctx(ps, 16, pd, 16, NppiSize{-1, 8}, pk, 3, 1, ctx);
  });
  status_case("nppiFilterRow32f", "nppiFilterRow32f_8u_C1R", "err/step_small", [&] {
    return nppiFilterRow32f_8u_C1R_Ctx(ps, 16, pd, 7, roi, pk, 3, 1, ctx);
  });
  status_case("nppiFilterColumn32f", "nppiFilterColumn32f_8u_C1R", "err/anchor_low", [&] {
    return nppiFilterColumn32f_8u_C1R_Ctx(ps, 16, pd, 16, roi, pk, 3, -8, ctx);
  });
  status_case("nppiFilterColumn32f", "nppiFilterColumn32f_8u_C1R", "err/neg_mask", [&] {
    return nppiFilterColumn32f_8u_C1R_Ctx(ps, 16, pd, 16, roi, pk, -1, 0, ctx);
  });
  status_case("nppiFilterRowBorder32f", "nppiFilterRowBorder32f_8u_C1R", "err/bad_border", [&] {
    return nppiFilterRowBorder32f_8u_C1R_Ctx(ps, 16, sz, off, pd, 16, roi, pk, 3, 1,
                                             NPP_BORDER_CONSTANT, ctx);
  });
  status_case("nppiFilterRowBorder32f", "nppiFilterRowBorder32f_8u_C1R", "err/src_size_zero", [&] {
    return nppiFilterRowBorder32f_8u_C1R_Ctx(ps, 16, NppiSize{0, 9}, off, pd, 16, roi, pk, 3, 1,
                                             NPP_BORDER_REPLICATE, ctx);
  });
  status_case("nppiFilterRowBorder32f", "nppiFilterRowBorder32f_8u_C1R",
              "err/neg_roi_beats_border", [&] {
    return nppiFilterRowBorder32f_8u_C1R_Ctx(ps, 16, sz, off, pd, 16, NppiSize{-1, 7}, pk, 3, 1,
                                             NPP_BORDER_CONSTANT, ctx);
  });
  status_case("nppiFilterRowBorder32f", "nppiFilterRowBorder32f_8u_C1R", "err/offset_oob", [&] {
    return nppiFilterRowBorder32f_8u_C1R_Ctx(ps, 16, sz, NppiPoint{9, 3}, pd, 16, roi, pk, 3, 1,
                                             NPP_BORDER_REPLICATE, ctx);
  });
  status_case("nppiFilterRowBorder32f", "nppiFilterRowBorder32f_8u_C1R", "err/step_small", [&] {
    return nppiFilterRowBorder32f_8u_C1R_Ctx(ps, 16, sz, off, pd, 7, roi, pk, 3, 1,
                                             NPP_BORDER_REPLICATE, ctx);
  });
  status_case("nppiFilterColumnBorder32f", "nppiFilterColumnBorder32f_8u_C1R", "err/bad_border", [&] {
    return nppiFilterColumnBorder32f_8u_C1R_Ctx(ps, 16, sz, off, pd, 16, roi, pk, 3, 1,
                                                NPP_BORDER_CONSTANT, ctx);
  });
  status_case("nppiFilterColumnBorder32f", "nppiFilterColumnBorder32f_8u_C1R", "err/null_dst", [&] {
    return nppiFilterColumnBorder32f_8u_C1R_Ctx(ps, 16, sz, off, nullptr, 16, roi, pk, 3, 1,
                                                NPP_BORDER_REPLICATE, ctx);
  });
  status_case("nppiFilterColumnBorder32f", "nppiFilterColumnBorder32f_8u_C1R", "err/offset_neg", [&] {
    return nppiFilterColumnBorder32f_8u_C1R_Ctx(ps, 16, sz, NppiPoint{2, -1}, pd, 16, roi, pk, 3, 1,
                                                NPP_BORDER_REPLICATE, ctx);
  });
}
