// Differential tests for package f12: nppiFilterRowBorder /
// nppiFilterColumnBorder (1D separable filters with border handling):
// integer kernels with divisor (8u/16u/16s) and 32f kernels, C1/C3/C4/AC4.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiFilterRowColBorderDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppiFilterRowColBorderDiff.*'
//
// Kernels and float sources are dyadic (integer multiples of 1/8) and 16-bit
// sources use their full range: every float partial sum stays below 2^24, so
// the accumulation is exact regardless of the implementation's summation
// order.  Integer-kernel variants use int32 ring arithmetic (wrap once,
// divide with C truncation, saturate).  Masks and anchors are NOT validated
// for the border families (measured NPP 13.4 behaviour); the replicate border
// clamps reads in both axes.

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

template <typename D> ValueKind dst_kind() {
  if constexpr (std::is_same_v<D, Npp8u>) return ValueKind::U8;
  else if constexpr (std::is_same_v<D, Npp16u>) return ValueKind::U16;
  else if constexpr (std::is_same_v<D, Npp16s>) return ValueKind::I16;
  else return ValueKind::F32;
}

template <typename F>
void status_case(const std::string &family, const char *fname, const std::string &key, F &&f) {
  clear_cuda_error();
  NppStatus st = f();
  Case c(family + "/" + key, fname);
  c.status(st);
  c.finish();
}

// Border data case: reads are clamped into the (sw x sh) source image.
template <typename S, typename D, typename K, typename F>
void rcb_data(const std::string &family, const char *fname, const std::string &key, int W, int H,
              int ch, int sw, int sh, int offx, int offy, int mask, int anchor, uint64_t seed,
              int extreme, ValueKind kind, int dstep, const std::vector<K> &hk, F &&f) {
  const int sPx = ch * (int)sizeof(S);
  const int dPx = ch * (int)sizeof(D);
  const int sStep = align_step(sw * sPx + 3, sPx);
  const int dStride =
      dstep < 0 ? -dstep : (dstep ? align_step(dstep, dPx) : align_step(W * dPx + 3, dPx));
  const size_t se = (size_t)sh * sStep / sizeof(S);
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
  const S *ps = s.get() + (size_t)offy * (sStep / sizeof(S)) + (size_t)offx * ch;
  D *pd = (dstep < 0 && H > 0) ? d.get() + (size_t)(H - 1) * (dStride / sizeof(D)) : d.get();
  NppStatus st = f(ps, sStep, NppiSize{sw, sh}, NppiPoint{offx, offy}, pd,
                   dstep < 0 ? -dStride : dStride, NppiSize{W, H}, k.get(), mask, anchor,
                   NPP_BORDER_REPLICATE, current_ctx());
  Case c(family + "/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(S));
  c.input(hk.data(), hk.size() * sizeof(K));
  c.status(st);
  c.device_output<D>(d.get(), dz, kind);
  c.finish();
}

template <typename S, typename D, typename K, typename F>
void rcb(const std::string &family, const char *fname, const std::string &key, int W, int H, int ch,
         int sw, int sh, int offx, int offy, int mask, int anchor, uint64_t seed, ValueKind kind,
         const std::vector<K> &hk, F &&f) {
  rcb_data<S, D, K, F>(family, fname, key, W, H, ch, sw, sh, offx, offy, mask, anchor, seed, 0,
                       kind, 0, hk, std::forward<F>(f));
}

}  // namespace

class NppiFilterRowColBorderDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
  int case_id_ = 0;
};

// ---------------------------------------------------------------------------
// Sweeps: every dtype x layout of the two families, mask 5, anchor 2,
// divisor 7, 7x7 ROI in a 9x9 source at offset (2,3).
// ---------------------------------------------------------------------------
#define RB_DIV_RUN(S, TAG, CH)                                                                 \
  rcb<S, S, Npp32s>("nppiFilterRowBorder", "nppiFilterRowBorder_" #TAG, "sweep/" #TAG, 7, 7, CH, \
                    9, 9, 2, 3, 5, 2, 0xC100 + case_id_++, dst_kind<S>(), make_ki(5, 0xC1FF),   \
                    [](const S *s, int ss, NppiSize sz, NppiPoint so, S *d, int ds, NppiSize o, \
                       const Npp32s *k, int m, int a, int div, NppiBorderType bt,              \
                       NppStreamContext c) {                                                   \
                      return nppiFilterRowBorder_##TAG##_Ctx(s, ss, sz, so, d, ds, o, k, m, a,  \
                                                             div, bt, c);                      \
                    })
#define CB_DIV_RUN(S, TAG, CH)                                                                 \
  rcb<S, S, Npp32s>("nppiFilterColumnBorder", "nppiFilterColumnBorder_" #TAG, "sweep/" #TAG, 7, \
                    7, CH, 9, 9, 2, 3, 5, 2, 0xC200 + case_id_++, dst_kind<S>(), make_ki(5, 0xC2FF), \
                    [](const S *s, int ss, NppiSize sz, NppiPoint so, S *d, int ds, NppiSize o, \
                       const Npp32s *k, int m, int a, int div, NppiBorderType bt,              \
                       NppStreamContext c) {                                                   \
                      return nppiFilterColumnBorder_##TAG##_Ctx(s, ss, sz, so, d, ds, o, k, m,  \
                                                                a, div, bt, c);                 \
                    })
#define RB_F32_RUN(TAG, CH)                                                                    \
  rcb<Npp32f, Npp32f, Npp32f>("nppiFilterRowBorder", "nppiFilterRowBorder_" #TAG, "sweep/" #TAG, \
                              7, 7, CH, 9, 9, 2, 3, 5, 2, 0xC300 + case_id_++,                 \
                              dst_kind<Npp32f>(), make_kf(5, 0xC3FF),                          \
                              [](const Npp32f *s, int ss, NppiSize sz, NppiPoint so, Npp32f *d, \
                                 int ds, NppiSize o, const Npp32f *k, int m, int a,            \
                                 NppiBorderType bt, NppStreamContext c) {                      \
                                return nppiFilterRowBorder_##TAG##_Ctx(s, ss, sz, so, d, ds, o, \
                                                                        k, m, a, bt, c);        \
                              })
#define CB_F32_RUN(TAG, CH)                                                                    \
  rcb<Npp32f, Npp32f, Npp32f>("nppiFilterColumnBorder", "nppiFilterColumnBorder_" #TAG,        \
                              "sweep/" #TAG, 7, 7, CH, 9, 9, 2, 3, 5, 2, 0xC400 + case_id_++,  \
                              dst_kind<Npp32f>(), make_kf(5, 0xC4FF),                          \
                              [](const Npp32f *s, int ss, NppiSize sz, NppiPoint so, Npp32f *d, \
                                 int ds, NppiSize o, const Npp32f *k, int m, int a,            \
                                 NppiBorderType bt, NppStreamContext c) {                      \
                                return nppiFilterColumnBorder_##TAG##_Ctx(s, ss, sz, so, d, ds, \
                                                                           o, k, m, a, bt, c);  \
                              })

#define B_DIV_SWEEP(FN)                                                                        \
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

#define B_F32_SWEEP(FN)                                                                        \
  FN(32f_C1R, 1);                                                                              \
  FN(32f_C3R, 3);                                                                              \
  FN(32f_C4R, 4);                                                                              \
  FN(32f_AC4R, 4)

TEST_F(NppiFilterRowColBorderDiff, DivSweep) { B_DIV_SWEEP(RB_DIV_RUN); }
TEST_F(NppiFilterRowColBorderDiff, ColumnDivSweep) { B_DIV_SWEEP(CB_DIV_RUN); }
TEST_F(NppiFilterRowColBorderDiff, FloatSweep) { B_F32_SWEEP(RB_F32_RUN); }
TEST_F(NppiFilterRowColBorderDiff, ColumnFloatSweep) { B_F32_SWEEP(CB_F32_RUN); }

// ---------------------------------------------------------------------------
// Geometry: offsets, ROI beyond the source, unvalidated anchors/masks, empty
// masks, negative masks, strides, divisors.
// ---------------------------------------------------------------------------
TEST_F(NppiFilterRowColBorderDiff, DivGeometry) {
  auto rowb = [&](const char *tag, int W, int H, int sw, int sh, int ox, int oy, int mask, int anchor,
                  uint64_t seed, const std::vector<Npp32s> &hk, int div = 7, int dstep = 0) {
    rcb_data<Npp16u, Npp16u, Npp32s>(
        "nppiFilterRowBorder", "nppiFilterRowBorder_16u_C1R", std::string("geom/rowb/") + tag, W,
        H, 1, sw, sh, ox, oy, mask, anchor, seed, 0, dst_kind<Npp16u>(), dstep, hk,
        [div](const Npp16u *s, int ss, NppiSize sz, NppiPoint so, Npp16u *d, int ds, NppiSize o,
              const Npp32s *k, int m, int a, NppiBorderType bt, NppStreamContext c) {
          return nppiFilterRowBorder_16u_C1R_Ctx(s, ss, sz, so, d, ds, o, k, m, a, div, bt, c);
        });
  };
  auto colb = [&](const char *tag, int W, int H, int sw, int sh, int ox, int oy, int mask, int anchor,
                  uint64_t seed, const std::vector<Npp32s> &hk, int div = 7, int dstep = 0) {
    rcb_data<Npp16u, Npp16u, Npp32s>(
        "nppiFilterColumnBorder", "nppiFilterColumnBorder_16u_C1R", std::string("geom/colb/") + tag,
        W, H, 1, sw, sh, ox, oy, mask, anchor, seed, 0, dst_kind<Npp16u>(), dstep, hk,
        [div](const Npp16u *s, int ss, NppiSize sz, NppiPoint so, Npp16u *d, int ds, NppiSize o,
              const Npp32s *k, int m, int a, NppiBorderType bt, NppStreamContext c) {
          return nppiFilterColumnBorder_16u_C1R_Ctx(s, ss, sz, so, d, ds, o, k, m, a, div, bt, c);
        });
  };
  rowb("offset", 5, 5, 12, 10, 4, 6, 3, 1, 0xB101, make_ki(3, 0xB101));
  rowb("beyond", 9, 9, 4, 4, 1, 1, 3, 1, 0xB102, make_ki(3, 0xB102));   // ROI larger than source
  rowb("last_off", 3, 3, 8, 8, 7, 7, 3, 1, 0xB103, make_ki(3, 0xB103)); // offset in last pixel
  rowb("offset0", 5, 5, 8, 8, 0, 0, 5, 2, 0xB104, make_ki(5, 0xB104));
  rowb("anchor9", 4, 4, 6, 6, 1, 1, 3, 9, 0xB105, make_ki(3, 0xB105));  // anchors not validated
  rowb("anchor_neg", 4, 4, 6, 6, 1, 1, 3, -2, 0xB106, make_ki(3, 0xB106));
  rowb("mask0", 4, 4, 6, 6, 1, 1, 0, 0, 0xB107, make_ki(1, 0xB107));    // zero mask -> zeros
  rowb("mask_neg", 4, 4, 6, 6, 1, 1, -3, 0, 0xB108, make_ki(1, 0xB108)); // negative mask
  rowb("mask2", 6, 6, 8, 8, 1, 1, 2, 0, 0xB109, make_ki(2, 0xB109));
  rowb("mask9", 4, 4, 6, 6, 1, 1, 9, 4, 0xB10A, make_ki(9, 0xB10A));
  rowb("src1x1", 4, 4, 1, 1, 0, 0, 3, 1, 0xB10B, make_ki(3, 0xB10B));
  rowb("small", 1, 1, 5, 5, 2, 2, 3, 1, 0xB10C, make_ki(3, 0xB10C));
  rowb("wide", 33, 1, 40, 8, 1, 3, 5, 2, 0xB10D, make_ki(5, 0xB10D));
  rowb("tall", 1, 33, 8, 40, 3, 1, 5, 2, 0xB10E, make_ki(5, 0xB10E));
  rowb("emptyw", 0, 4, 6, 6, 1, 1, 3, 1, 0xB10F, make_ki(3, 0xB10F));
  rowb("emptyh", 4, 0, 6, 6, 1, 1, 3, 1, 0xB110, make_ki(3, 0xB110));
  rowb("negstep", 4, 4, 6, 6, 1, 1, 3, 1, 0xB111, make_ki(3, 0xB111), 7, -48);
  rowb("bigstep", 4, 4, 6, 6, 1, 1, 3, 1, 0xB112, make_ki(3, 0xB112), 7, 98);
  rowb("div1", 5, 5, 8, 8, 1, 1, 5, 2, 0xB113, make_ki(5, 0xB113), 1);
  rowb("divneg", 5, 5, 8, 8, 1, 1, 5, 2, 0xB114, make_ki(5, 0xB114), -3);
  rowb("divbig", 5, 5, 8, 8, 1, 1, 5, 2, 0xB115, make_ki(5, 0xB115), 1000003);
  colb("offset", 5, 5, 12, 10, 4, 6, 3, 1, 0xB121, make_ki(3, 0xB121));
  colb("beyond", 9, 9, 4, 4, 1, 1, 3, 1, 0xB122, make_ki(3, 0xB122));
  colb("last_off", 3, 3, 8, 8, 7, 7, 3, 1, 0xB123, make_ki(3, 0xB123));
  colb("offset0", 5, 5, 8, 8, 0, 0, 5, 2, 0xB124, make_ki(5, 0xB124));
  colb("anchor9", 4, 4, 6, 6, 1, 1, 3, 9, 0xB125, make_ki(3, 0xB125));
  colb("anchor_neg", 4, 4, 6, 6, 1, 1, 3, -2, 0xB126, make_ki(3, 0xB126));
  colb("mask0", 4, 4, 6, 6, 1, 1, 0, 0, 0xB127, make_ki(1, 0xB127));
  colb("mask_neg", 4, 4, 6, 6, 1, 1, -3, 0, 0xB128, make_ki(1, 0xB128));
  colb("mask2", 6, 6, 8, 8, 1, 1, 2, 0, 0xB129, make_ki(2, 0xB129));
  colb("mask9", 4, 4, 6, 6, 1, 1, 9, 4, 0xB12A, make_ki(9, 0xB12A));
  colb("src1x1", 4, 4, 1, 1, 0, 0, 3, 1, 0xB12B, make_ki(3, 0xB12B));
  colb("emptyw", 0, 4, 6, 6, 1, 1, 3, 1, 0xB12C, make_ki(3, 0xB12C));
  colb("emptyh", 4, 0, 6, 6, 1, 1, 3, 1, 0xB12D, make_ki(3, 0xB12D));
  colb("negstep", 4, 4, 6, 6, 1, 1, 3, 1, 0xB12E, make_ki(3, 0xB12E), 7, -48);
  colb("divneg", 5, 5, 8, 8, 1, 1, 5, 2, 0xB12F, make_ki(5, 0xB12F), -3);
}

TEST_F(NppiFilterRowColBorderDiff, FloatGeometry) {
  auto rowb32 = [&](const char *tag, int W, int H, int sw, int sh, int ox, int oy, int mask,
                    int anchor, uint64_t seed, const std::vector<Npp32f> &hk, int dstep = 0) {
    rcb_data<Npp32f, Npp32f, Npp32f>(
        "nppiFilterRowBorder", "nppiFilterRowBorder_32f_C3R", std::string("geom/32f_rowb/") + tag,
        W, H, 3, sw, sh, ox, oy, mask, anchor, seed, 0, dst_kind<Npp32f>(), dstep, hk,
        [](const Npp32f *s, int ss, NppiSize sz, NppiPoint so, Npp32f *d, int ds, NppiSize o,
           const Npp32f *k, int m, int a, NppiBorderType bt, NppStreamContext c) {
          return nppiFilterRowBorder_32f_C3R_Ctx(s, ss, sz, so, d, ds, o, k, m, a, bt, c);
        });
  };
  auto colb32 = [&](const char *tag, int W, int H, int sw, int sh, int ox, int oy, int mask,
                    int anchor, uint64_t seed, const std::vector<Npp32f> &hk) {
    rcb_data<Npp32f, Npp32f, Npp32f>(
        "nppiFilterColumnBorder", "nppiFilterColumnBorder_32f_C3R",
        std::string("geom/32f_colb/") + tag, W, H, 3, sw, sh, ox, oy, mask, anchor, seed, 0,
        dst_kind<Npp32f>(), 0, hk,
        [](const Npp32f *s, int ss, NppiSize sz, NppiPoint so, Npp32f *d, int ds, NppiSize o,
           const Npp32f *k, int m, int a, NppiBorderType bt, NppStreamContext c) {
          return nppiFilterColumnBorder_32f_C3R_Ctx(s, ss, sz, so, d, ds, o, k, m, a, bt, c);
        });
  };
  rowb32("offset", 5, 5, 12, 10, 4, 6, 3, 1, 0xB201, make_kf(3, 0xB201));
  rowb32("beyond", 9, 9, 4, 4, 1, 1, 3, 1, 0xB202, make_kf(3, 0xB202));
  rowb32("anchor9", 4, 4, 6, 6, 1, 1, 3, 9, 0xB203, make_kf(3, 0xB203));
  rowb32("anchor_neg", 4, 4, 6, 6, 1, 1, 3, -2, 0xB204, make_kf(3, 0xB204));
  rowb32("mask0", 4, 4, 6, 6, 1, 1, 0, 0, 0xB205, make_kf(1, 0xB205));
  rowb32("mask_neg", 4, 4, 6, 6, 1, 1, -3, 0, 0xB206, make_kf(1, 0xB206));
  rowb32("mask2", 6, 6, 8, 8, 1, 1, 2, 0, 0xB207, make_kf(2, 0xB207));
  rowb32("emptyw", 0, 4, 6, 6, 1, 1, 3, 1, 0xB208, make_kf(3, 0xB208));
  rowb32("negstep", 4, 4, 6, 6, 1, 1, 3, 1, 0xB209, make_kf(3, 0xB209), -48);
  colb32("offset", 5, 5, 12, 10, 4, 6, 3, 1, 0xB211, make_kf(3, 0xB211));
  colb32("beyond", 9, 9, 4, 4, 1, 1, 3, 1, 0xB212, make_kf(3, 0xB212));
  colb32("mask0", 4, 4, 6, 6, 1, 1, 0, 0, 0xB213, make_kf(1, 0xB213));
  colb32("mask_neg", 4, 4, 6, 6, 1, 1, -3, 0, 0xB214, make_kf(1, 0xB214));
  colb32("emptyh", 4, 0, 6, 6, 1, 1, 3, 1, 0xB215, make_kf(3, 0xB215));
}

// AC4 alpha handling on the border families.
TEST_F(NppiFilterRowColBorderDiff, LayoutGeometry) {
  auto ac4row = [&](const char *tag, int mask, int anchor, uint64_t seed,
                    const std::vector<Npp32s> &hk, int div = 7) {
    rcb<Npp16s, Npp16s, Npp32s>("nppiFilterRowBorder", "nppiFilterRowBorder_16s_AC4R",
                                std::string("geom/16s_AC4_rowb/") + tag, 7, 7, 4, 9, 9, 2, 3, mask,
                                anchor, seed, dst_kind<Npp16s>(), hk,
                                [div](const Npp16s *s, int ss, NppiSize sz, NppiPoint so, Npp16s *d,
                                      int ds, NppiSize o, const Npp32s *k, int m, int a,
                                      NppiBorderType bt, NppStreamContext c) {
                                  return nppiFilterRowBorder_16s_AC4R_Ctx(s, ss, sz, so, d, ds, o,
                                                                           k, m, a, div, bt, c);
                                });
  };
  auto ac4col = [&](const char *tag, int mask, int anchor, uint64_t seed,
                    const std::vector<Npp32s> &hk, int div = 7) {
    rcb<Npp16s, Npp16s, Npp32s>("nppiFilterColumnBorder", "nppiFilterColumnBorder_16s_AC4R",
                                std::string("geom/16s_AC4_colb/") + tag, 7, 7, 4, 9, 9, 2, 3, mask,
                                anchor, seed, dst_kind<Npp16s>(), hk,
                                [div](const Npp16s *s, int ss, NppiSize sz, NppiPoint so, Npp16s *d,
                                      int ds, NppiSize o, const Npp32s *k, int m, int a,
                                      NppiBorderType bt, NppStreamContext c) {
                                  return nppiFilterColumnBorder_16s_AC4R_Ctx(s, ss, sz, so, d, ds,
                                                                             o, k, m, a, div, bt, c);
                                });
  };
  ac4row("mask0", 0, 0, 0xB301, make_ki(1, 0xB301));
  ac4row("mask_neg", -3, 0, 0xB302, make_ki(1, 0xB302));
  ac4row("mask3", 3, 1, 0xB303, make_ki(3, 0xB303));
  ac4row("divneg", 5, 2, 0xB304, make_ki(5, 0xB304), -3);
  ac4col("mask0", 0, 0, 0xB311, make_ki(1, 0xB311));
  ac4col("divneg", 5, 2, 0xB312, make_ki(5, 0xB312), -3);
}

// ---------------------------------------------------------------------------
// Integer ring arithmetic on the border variants.
// ---------------------------------------------------------------------------
TEST_F(NppiFilterRowColBorderDiff, DivArithmetic) {
  auto prow = [&](const char *tag, const std::vector<Npp32s> &hk, int mask, int anchor, int div,
                  int extreme) {
    rcb_data<Npp16s, Npp16s, Npp32s>(
        "nppiFilterRowBorder", "nppiFilterRowBorder_16s_C1R", std::string("arith/rowb/") + tag, 7,
        5, 1, 9, 7, 2, 1, mask, anchor, 0xB400 + case_id_++, extreme, dst_kind<Npp16s>(), 0, hk,
        [div](const Npp16s *s, int ss, NppiSize sz, NppiPoint so, Npp16s *d, int ds, NppiSize o,
              const Npp32s *k, int m, int a, NppiBorderType bt, NppStreamContext c) {
          return nppiFilterRowBorder_16s_C1R_Ctx(s, ss, sz, so, d, ds, o, k, m, a, div, bt, c);
        });
  };
  auto pcol = [&](const char *tag, const std::vector<Npp32s> &hk, int mask, int anchor, int div,
                  int extreme) {
    rcb_data<Npp16s, Npp16s, Npp32s>(
        "nppiFilterColumnBorder", "nppiFilterColumnBorder_16s_C1R",
        std::string("arith/colb/") + tag, 7, 5, 1, 9, 7, 2, 1, mask, anchor, 0xB500 + case_id_++,
        extreme, dst_kind<Npp16s>(), 0, hk,
        [div](const Npp16s *s, int ss, NppiSize sz, NppiPoint so, Npp16s *d, int ds, NppiSize o,
              const Npp32s *k, int m, int a, NppiBorderType bt, NppStreamContext c) {
          return nppiFilterColumnBorder_16s_C1R_Ctx(s, ss, sz, so, d, ds, o, k, m, a, div, bt, c);
        });
  };
  prow("wrap", make_kc(1, 1 << 20), 1, 0, 1, 0);
  prow("wrapmin", make_kc(1, -2147483647 - 1), 1, 0, 1, 8);
  prow("divm1", make_kc(1, 1), 1, 0, -1, 8);
  prow("trunc", make_kc(1, 4), 1, 0, 3, 8);
  prow("sat", make_kc(9, 4096), 9, 4, 1, 0);
  pcol("wrap", make_kc(1, 1 << 20), 1, 0, 1, 0);
  pcol("divm1", make_kc(1, 1), 1, 0, -1, 8);
  pcol("trunc", make_kc(1, 4), 1, 0, 3, 8);
  pcol("sat", make_kc(9, 4096), 9, 4, 1, 0);
}

// ---------------------------------------------------------------------------
// Error statuses and validation order (null -> srcSize -> roi -> border mode
// -> srcOffset -> divisor -> step).
// ---------------------------------------------------------------------------
TEST_F(NppiFilterRowColBorderDiff, DivErrors) {
  const auto hk = make_ki(5, 0xB601);
  DeviceMemory<Npp16s> &s = arena<Npp16s>(4096);
  DeviceMemory<Npp16s> &d = arena<Npp16s>(4096);
  DeviceMemory<Npp32s> &k = arena<Npp32s>(5);
  k.copyFromHost(hk);
  const Npp16s *ps = s.get();
  Npp16s *pd = d.get();
  const Npp32s *pk = k.get();
  auto ctx = current_ctx();
  const NppiSize roi{8, 8};
  const NppiSize sz{9, 9};
  const NppiPoint off{2, 3};
  auto row = [&](const Npp16s *src, int ss, NppiSize szb, NppiPoint ofb, Npp16s *dst, int ds,
                 NppiSize o, int m, int a, int div, NppiBorderType bt) {
    return nppiFilterRowBorder_16s_C1R_Ctx(src, ss, szb, ofb, dst, ds, o, pk, m, a, div, bt, ctx);
  };
  auto col = [&](const Npp16s *src, int ss, NppiSize szb, NppiPoint ofb, Npp16s *dst, int ds,
                 NppiSize o, int m, int a, int div, NppiBorderType bt) {
    return nppiFilterColumnBorder_16s_C1R_Ctx(src, ss, szb, ofb, dst, ds, o, pk, m, a, div, bt, ctx);
  };
  status_case("nppiFilterRowBorder", "nppiFilterRowBorder_16s_C1R", "err/null_src",
              [&] { return row(nullptr, 16, sz, off, pd, 16, roi, 3, 1, 7, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterRowBorder", "nppiFilterRowBorder_16s_C1R", "err/null_dst",
              [&] { return row(ps, 16, sz, off, nullptr, 16, roi, 3, 1, 7, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterRowBorder", "nppiFilterRowBorder_16s_C1R", "err/src_size_zero",
              [&] { return row(ps, 16, NppiSize{0, 9}, off, pd, 16, roi, 3, 1, 7, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterRowBorder", "nppiFilterRowBorder_16s_C1R", "err/src_size_neg",
              [&] { return row(ps, 16, NppiSize{-1, 9}, off, pd, 16, roi, 3, 1, 7, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterRowBorder", "nppiFilterRowBorder_16s_C1R", "err/neg_roi",
              [&] { return row(ps, 16, sz, off, pd, 16, NppiSize{-1, 7}, 3, 1, 7, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterRowBorder", "nppiFilterRowBorder_16s_C1R", "err/neg_roi_beats_border",
              [&] { return row(ps, 16, sz, off, pd, 16, NppiSize{-1, 7}, 3, 1, 7, NPP_BORDER_CONSTANT); });
  status_case("nppiFilterRowBorder", "nppiFilterRowBorder_16s_C1R", "err/bad_border",
              [&] { return row(ps, 16, sz, off, pd, 16, roi, 3, 1, 7, NPP_BORDER_CONSTANT); });
  status_case("nppiFilterRowBorder", "nppiFilterRowBorder_16s_C1R", "err/border_wrap",
              [&] { return row(ps, 16, sz, off, pd, 16, roi, 3, 1, 7, NPP_BORDER_WRAP); });
  status_case("nppiFilterRowBorder", "nppiFilterRowBorder_16s_C1R", "err/border_beats_offset",
              [&] { return row(ps, 16, sz, NppiPoint{9, 3}, pd, 16, roi, 3, 1, 7, NPP_BORDER_CONSTANT); });
  status_case("nppiFilterRowBorder", "nppiFilterRowBorder_16s_C1R", "err/offset_oob",
              [&] { return row(ps, 16, sz, NppiPoint{9, 3}, pd, 16, roi, 3, 1, 7, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterRowBorder", "nppiFilterRowBorder_16s_C1R", "err/offset_neg",
              [&] { return row(ps, 16, sz, NppiPoint{2, -1}, pd, 16, roi, 3, 1, 7, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterRowBorder", "nppiFilterRowBorder_16s_C1R", "err/offset_vs_div0",
              [&] { return row(ps, 16, sz, NppiPoint{9, 3}, pd, 16, roi, 3, 1, 0, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterRowBorder", "nppiFilterRowBorder_16s_C1R", "err/div0",
              [&] { return row(ps, 16, sz, off, pd, 16, roi, 3, 1, 0, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterRowBorder", "nppiFilterRowBorder_16s_C1R", "err/div0_vs_step",
              [&] { return row(ps, 16, sz, off, pd, 15, roi, 3, 1, 0, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterRowBorder", "nppiFilterRowBorder_16s_C1R", "err/div0_vs_empty",
              [&] { return row(ps, 16, sz, off, pd, 16, NppiSize{0, 7}, 3, 1, 0, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterRowBorder", "nppiFilterRowBorder_16s_C1R", "err/div0_vs_mask0",
              [&] { return row(ps, 16, sz, off, pd, 16, roi, 0, 0, 0, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterRowBorder", "nppiFilterRowBorder_16s_C1R", "err/step_small",
              [&] { return row(ps, 16, sz, off, pd, 15, roi, 3, 1, 7, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterRowBorder", "nppiFilterRowBorder_16s_C1R", "err/step_zero",
              [&] { return row(ps, 16, sz, off, pd, 0, roi, 3, 1, 7, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterRowBorder", "nppiFilterRowBorder_16s_C1R", "ok/empty_roi",
              [&] { return row(ps, 16, sz, off, pd, 16, NppiSize{0, 7}, 3, 1, 7, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterRowBorder", "nppiFilterRowBorder_16s_C1R", "ok/mask0",
              [&] { return row(ps, 16, sz, off, pd, 16, roi, 0, 0, 7, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterRowBorder", "nppiFilterRowBorder_16s_C1R", "ok/neg_step",
              [&] { return row(ps, 16, sz, off, pd + 3000, -64, roi, 3, 1, 7, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterColumnBorder", "nppiFilterColumnBorder_16s_C1R", "err/null_src",
              [&] { return col(nullptr, 16, sz, off, pd, 16, roi, 3, 1, 7, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterColumnBorder", "nppiFilterColumnBorder_16s_C1R", "err/bad_border",
              [&] { return col(ps, 16, sz, off, pd, 16, roi, 3, 1, 7, NPP_BORDER_CONSTANT); });
  status_case("nppiFilterColumnBorder", "nppiFilterColumnBorder_16s_C1R", "err/src_size_zero",
              [&] { return col(ps, 16, NppiSize{9, 0}, off, pd, 16, roi, 3, 1, 7, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterColumnBorder", "nppiFilterColumnBorder_16s_C1R", "err/offset_oob",
              [&] { return col(ps, 16, sz, NppiPoint{3, 9}, pd, 16, roi, 3, 1, 7, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterColumnBorder", "nppiFilterColumnBorder_16s_C1R", "err/div0",
              [&] { return col(ps, 16, sz, off, pd, 16, roi, 3, 1, 0, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterColumnBorder", "nppiFilterColumnBorder_16s_C1R", "err/step_small",
              [&] { return col(ps, 16, sz, off, pd, 15, roi, 3, 1, 7, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterColumnBorder", "nppiFilterColumnBorder_16s_C1R", "ok/empty_roi",
              [&] { return col(ps, 16, sz, off, pd, 16, NppiSize{7, 0}, 3, 1, 7, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterColumnBorder", "nppiFilterColumnBorder_16s_C1R", "ok/mask0",
              [&] { return col(ps, 16, sz, off, pd, 16, roi, 0, 0, 7, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterColumnBorder", "nppiFilterColumnBorder_16s_C1R", "ok/neg_step",
              [&] { return col(ps, 16, sz, off, pd + 3000, -64, roi, 3, 1, 7, NPP_BORDER_REPLICATE); });
}

TEST_F(NppiFilterRowColBorderDiff, FloatErrors) {
  DeviceMemory<Npp32f> &s = arena<Npp32f>(4096);
  DeviceMemory<Npp32f> &d = arena<Npp32f>(4096);
  DeviceMemory<Npp32f> &k = arena<Npp32f>(5);
  k.copyFromHost(make_kf(5, 0xB701));
  const Npp32f *ps = s.get();
  Npp32f *pd = d.get();
  const Npp32f *pk = k.get();
  auto ctx = current_ctx();
  const NppiSize roi{8, 8};
  const NppiSize sz{9, 9};
  const NppiPoint off{2, 3};
  auto row = [&](const Npp32f *src, int ss, NppiSize szb, NppiPoint ofb, Npp32f *dst, int ds,
                 NppiSize o, int m, int a, NppiBorderType bt) {
    return nppiFilterRowBorder_32f_C3R_Ctx(src, ss, szb, ofb, dst, ds, o, pk, m, a, bt, ctx);
  };
  auto col = [&](const Npp32f *src, int ss, NppiSize szb, NppiPoint ofb, Npp32f *dst, int ds,
                 NppiSize o, int m, int a, NppiBorderType bt) {
    return nppiFilterColumnBorder_32f_C3R_Ctx(src, ss, szb, ofb, dst, ds, o, pk, m, a, bt, ctx);
  };
  status_case("nppiFilterRowBorder", "nppiFilterRowBorder_32f_C3R", "err/null_src",
              [&] { return row(nullptr, 96, sz, off, pd, 96, roi, 3, 1, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterRowBorder", "nppiFilterRowBorder_32f_C3R", "err/null_dst",
              [&] { return row(ps, 96, sz, off, nullptr, 96, roi, 3, 1, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterRowBorder", "nppiFilterRowBorder_32f_C3R", "err/src_size_zero",
              [&] { return row(ps, 96, NppiSize{0, 9}, off, pd, 96, roi, 3, 1, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterRowBorder", "nppiFilterRowBorder_32f_C3R", "err/neg_roi_beats_border",
              [&] { return row(ps, 96, sz, off, pd, 96, NppiSize{-1, 7}, 3, 1, NPP_BORDER_CONSTANT); });
  status_case("nppiFilterRowBorder", "nppiFilterRowBorder_32f_C3R", "err/bad_border",
              [&] { return row(ps, 96, sz, off, pd, 96, roi, 3, 1, NPP_BORDER_CONSTANT); });
  status_case("nppiFilterRowBorder", "nppiFilterRowBorder_32f_C3R", "err/offset_oob",
              [&] { return row(ps, 96, sz, NppiPoint{9, 3}, pd, 96, roi, 3, 1, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterRowBorder", "nppiFilterRowBorder_32f_C3R", "err/offset_neg",
              [&] { return row(ps, 96, sz, NppiPoint{2, -1}, pd, 96, roi, 3, 1, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterRowBorder", "nppiFilterRowBorder_32f_C3R", "err/step_small",
              [&] { return row(ps, 96, sz, off, pd, 95, roi, 3, 1, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterRowBorder", "nppiFilterRowBorder_32f_C3R", "ok/empty_roi",
              [&] { return row(ps, 96, sz, off, pd, 96, NppiSize{0, 7}, 3, 1, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterRowBorder", "nppiFilterRowBorder_32f_C3R", "ok/mask0",
              [&] { return row(ps, 96, sz, off, pd, 96, roi, 0, 0, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterRowBorder", "nppiFilterRowBorder_32f_C3R", "ok/neg_step",
              [&] { return row(ps, 96, sz, off, pd + 3000, -192, roi, 3, 1, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterColumnBorder", "nppiFilterColumnBorder_32f_C3R", "err/null_dst",
              [&] { return col(ps, 96, sz, off, nullptr, 96, roi, 3, 1, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterColumnBorder", "nppiFilterColumnBorder_32f_C3R", "err/bad_border",
              [&] { return col(ps, 96, sz, off, pd, 96, roi, 3, 1, NPP_BORDER_CONSTANT); });
  status_case("nppiFilterColumnBorder", "nppiFilterColumnBorder_32f_C3R", "err/offset_oob",
              [&] { return col(ps, 96, sz, NppiPoint{3, 9}, pd, 96, roi, 3, 1, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterColumnBorder", "nppiFilterColumnBorder_32f_C3R", "err/step_small",
              [&] { return col(ps, 96, sz, off, pd, 95, roi, 3, 1, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterColumnBorder", "nppiFilterColumnBorder_32f_C3R", "ok/empty_roi",
              [&] { return col(ps, 96, sz, off, pd, 96, NppiSize{7, 0}, 3, 1, NPP_BORDER_REPLICATE); });
  status_case("nppiFilterColumnBorder", "nppiFilterColumnBorder_32f_C3R", "ok/mask0",
              [&] { return col(ps, 96, sz, off, pd, 96, roi, 0, 0, NPP_BORDER_REPLICATE); });
}
