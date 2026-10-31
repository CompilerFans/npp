// Differential tests for package f12: nppiFilter (2D convolution).
//
// Covers the integer-kernel variants with divisor (8u/16u/16s, C1/C3/C4/AC4),
// the 32f-kernel variants (32f C1/C2/C3/C4/AC4) and 64f_C1R.  The three
// functions that were already implemented (8u_C1R, 8u_C3R, 32f_C1R) are
// probed as well so their semantics can be compared against NPP 13.4.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiFilter2dDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppiFilter2dDiff.*'
//
// Kernels and float sources are dyadic (integer multiples of 1/8), and 16-bit
// sources use their full range: every float partial sum stays below 2^24, so
// the accumulation is exact regardless of the implementation's summation
// order.  Integer-kernel variants use int32 ring arithmetic (wrap once,
// divide with C truncation, saturate), which the wrap/sat probes pin down.

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
// (multiples of 1/8).  extreme > 0 sprinkles the type extremes: for integer
// types min/max, for floats +-4096 (so products stay exactly representable).
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

constexpr int kMargin = 8;  // raw taps with |anchor| <= 4 and kernel <= 5 stay inside

// 2D data case: the ROI sits inside a (W+2*margin)x(H+2*margin) buffer, so
// the borderless raw reads stay inside the allocation.  sstep/dstep override
// the computed strides when nonzero (dstep < 0 is used verbatim and the dst
// pointer is moved to the end of the buffer so negative row offsets stay
// inside the allocation).
template <typename S, typename D, typename K, typename F>
void f2_data(const std::string &family, const char *fname, const std::string &key, int W, int H,
             int ch, int kw, int kh, int ax, int ay, int margin, uint64_t seed, int extreme,
             ValueKind kind, int dstep, const std::vector<K> &hk, int sstep, F &&f) {
  const int sPx = ch * (int)sizeof(S);
  const int dPx = ch * (int)sizeof(D);
  const int sStride = sstep ? align_step(sstep, sPx) : align_step((W + 2 * margin) * sPx + 3, sPx);
  const int dStride =
      dstep < 0 ? -dstep : (dstep ? align_step(dstep, dPx) : align_step(W * dPx + 3, dPx));
  // Allocation always uses the (W+2*margin)-wide geometry so that a small
  // stride override (overlapping rows) keeps every read inside the buffer.
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
  NppStatus st = f(ps, sStride, pd, dstep < 0 ? -dStride : dStride, NppiSize{W, H}, k.get(),
                   NppiSize{kw, kh}, NppiPoint{ax, ay}, current_ctx());
  Case c(family + "/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(S));
  c.input(hk.data(), hk.size() * sizeof(K));
  c.status(st);
  c.device_output<D>(d.get(), dz, kind);
  c.finish();
}

template <typename S, typename D, typename K, typename F>
void f2(const std::string &family, const char *fname, const std::string &key, int W, int H, int ch,
        int kw, int kh, int ax, int ay, int margin, uint64_t seed, ValueKind kind,
        const std::vector<K> &hk, F &&f) {
  f2_data<S, D, K, F>(family, fname, key, W, H, ch, kw, kh, ax, ay, margin, seed, 0, kind, 0, hk,
                      0, std::forward<F>(f));
}

}  // namespace

class NppiFilter2dDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
  int case_id_ = 0;
};

// ---------------------------------------------------------------------------
// Sweeps: every dtype x layout of the family, 5x5 kernel, divisor 7.
// ---------------------------------------------------------------------------
#define DIV_RUN(S, TAG, CH)                                                                    \
  f2<S, S, Npp32s>("nppiFilter", "nppiFilter_" #TAG, "sweep/" #TAG, 17, 13, CH, 5, 5, 2, 2,    \
                   kMargin, 0xF100 + case_id_++, dst_kind<S>(), make_ki(25, 0xF1FF),           \
                   [](const S *s, int ss, S *d, int ds, NppiSize o, const Npp32s *k,           \
                      NppiSize ks, NppiPoint a, NppStreamContext c) {                          \
                     return nppiFilter_##TAG##_Ctx(s, ss, d, ds, o, k, ks, a, 7, c);            \
                   })
#define F32_RUN(TAG, CH)                                                                       \
  f2<Npp32f, Npp32f, Npp32f>("nppiFilter", "nppiFilter_" #TAG, "sweep/" #TAG, 17, 13, CH, 5, 5, 2, \
                             2, kMargin, 0xF200 + case_id_++, dst_kind<Npp32f>(),              \
                             make_kf(25, 0xF2FF),                                              \
                             [](const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize o,        \
                                const Npp32f *k, NppiSize ks, NppiPoint a, NppStreamContext c) { \
                               return nppiFilter_##TAG##_Ctx(s, ss, d, ds, o, k, ks, a, c);     \
                             })

TEST_F(NppiFilter2dDiff, DivSweep) {
  DIV_RUN(Npp8u, 8u_C4R, 4);
  DIV_RUN(Npp8u, 8u_AC4R, 4);
  DIV_RUN(Npp16u, 16u_C1R, 1);
  DIV_RUN(Npp16u, 16u_C3R, 3);
  DIV_RUN(Npp16u, 16u_C4R, 4);
  DIV_RUN(Npp16u, 16u_AC4R, 4);
  DIV_RUN(Npp16s, 16s_C1R, 1);
  DIV_RUN(Npp16s, 16s_C3R, 3);
  DIV_RUN(Npp16s, 16s_C4R, 4);
  DIV_RUN(Npp16s, 16s_AC4R, 4);
  // Already implemented ("real") functions: probes for semantic comparison.
  DIV_RUN(Npp8u, 8u_C1R, 1);
  DIV_RUN(Npp8u, 8u_C3R, 3);
}

TEST_F(NppiFilter2dDiff, FloatSweep) {
  F32_RUN(32f_C1R, 1);
  F32_RUN(32f_C2R, 2);
  F32_RUN(32f_C3R, 3);
  F32_RUN(32f_C4R, 4);
  F32_RUN(32f_AC4R, 4);
  f2<Npp64f, Npp64f, Npp64f>("nppiFilter", "nppiFilter_64f_C1R", "sweep/64f_C1R", 17, 13, 1, 5, 5,
                             2, 2, kMargin, 0xF300 + case_id_++, dst_kind<Npp64f>(),
                             make_kd(25, 0xF3FF),
                             [](const Npp64f *s, int ss, Npp64f *d, int ds, NppiSize o,
                                const Npp64f *k, NppiSize ks, NppiPoint a, NppStreamContext c) {
                               return nppiFilter_64f_C1R_Ctx(s, ss, d, ds, o, k, ks, a, c);
                             });
}

// ---------------------------------------------------------------------------
// Geometry: anchors (including the 1-W limit and anchor == kernel size),
// kernel shapes (including even sizes, 1xN/Nx1, wider than the ROI), empty
// kernels, empty ROIs, stride overrides, divisors.
// ---------------------------------------------------------------------------
TEST_F(NppiFilter2dDiff, DivGeometry) {
  auto g = [&](const char *tag, int W, int H, int kw, int kh, int ax, int ay, int margin,
               uint64_t seed, const std::vector<Npp32s> &hk, int div = 7, int dstep = 0,
               int sstep = 0, int extreme = 0) {
    f2_data<Npp16u, Npp16u, Npp32s>(
        "nppiFilter", "nppiFilter_16u_C1R", std::string("geom/16u/") + tag, W, H, 1, kw, kh, ax,
        ay, margin, seed, extreme, dst_kind<Npp16u>(), dstep, hk, sstep,
        [div](const Npp16u *s, int ss, Npp16u *d, int ds, NppiSize o, const Npp32s *k, NppiSize ks,
              NppiPoint a, NppStreamContext c) {
          return nppiFilter_16u_C1R_Ctx(s, ss, d, ds, o, k, ks, a, div, c);
        });
  };
  const int m = kMargin;
  g("a0", 17, 13, 5, 5, 0, 0, m, 0x6101, make_ki(25, 0x6101));
  g("amax", 17, 13, 5, 5, 5, 5, m, 0x6102, make_ki(25, 0x6102));  // anchor == kernel size
  g("aneg", 17, 13, 5, 5, -3, -3, m, 0x6103, make_ki(25, 0x6103));
  g("aext", 5, 5, 3, 3, -4, -4, m, 0x6104, make_ki(9, 0x6104));  // anchor == 1-W
  g("asym", 17, 13, 3, 1, 0, 0, m, 0x6105, std::vector<Npp32s>{1, 2, 4});  // mirror probe
  g("k11", 17, 13, 1, 1, 0, 0, m, 0x6106, make_ki(1, 0x6106));
  g("k15", 17, 13, 1, 5, 0, 2, m, 0x6107, make_ki(5, 0x6107));
  g("k51", 17, 13, 5, 1, 2, 0, m, 0x6108, make_ki(5, 0x6108));
  g("k37", 17, 13, 3, 7, 1, 3, m, 0x6109, make_ki(21, 0x6109));
  g("k22", 17, 13, 2, 2, 1, 1, m, 0x610A, make_ki(4, 0x610A));  // even kernel size
  g("k99", 17, 13, 9, 9, 4, 4, m, 0x610B, make_ki(81, 0x610B));
  g("kwide", 13, 9, 23, 3, 11, 1, 12, 0x610C, make_ki(69, 0x610C));  // kernel wider than ROI
  g("ktall", 9, 13, 3, 23, 1, 11, 12, 0x610D, make_ki(69, 0x610D));
  g("kevw", 17, 13, 0, 5, 0, 2, m, 0x610E, make_ki(5, 0x610E));   // empty kernel width
  g("kevh", 17, 13, 5, 0, 2, 0, m, 0x610F, make_ki(5, 0x610F));   // empty kernel height
  g("kev", 17, 13, 0, 0, 0, 0, m, 0x6110, make_ki(1, 0x6110));
  g("knegw", 17, 13, -1, 5, -1, 2, m, 0x6111, make_ki(5, 0x6111));  // negative kernel width
  g("knegh", 17, 13, 5, -1, 2, -1, m, 0x6112, make_ki(5, 0x6112));
  g("small", 1, 1, 3, 3, 1, 1, m, 0x6113, make_ki(9, 0x6113));
  g("wide", 33, 1, 5, 5, 2, 2, m, 0x6114, make_ki(25, 0x6114));
  g("tall", 1, 33, 5, 5, 2, 2, m, 0x6115, make_ki(25, 0x6115));
  g("emptyw", 0, 13, 5, 5, 2, 2, m, 0x6116, make_ki(25, 0x6116));
  g("emptyw_a0", 0, 13, 3, 3, 0, 0, m, 0x6117, make_ki(9, 0x6117));  // anchor 0 with W == 0
  g("emptyh", 17, 0, 5, 5, 2, 2, m, 0x6118, make_ki(25, 0x6118));
  g("negstep", 17, 13, 5, 5, 2, 2, m, 0x6119, make_ki(25, 0x6119), 7, -48);
  g("bigstep", 17, 13, 5, 5, 2, 2, m, 0x611A, make_ki(25, 0x611A), 7, 98);
  g("sstep_small", 8, 8, 3, 3, 1, 1, m, 0x611B, make_ki(9, 0x611B), 7, 0, 2);  // overlapping rows
  g("div1", 17, 13, 5, 5, 2, 2, m, 0x611C, make_ki(25, 0x611C), 1);
  g("divneg", 17, 13, 5, 5, 2, 2, m, 0x611D, make_ki(25, 0x611D), -3);
  g("divbig", 17, 13, 5, 5, 2, 2, m, 0x611E, make_ki(25, 0x611E), 1000003);
}

TEST_F(NppiFilter2dDiff, FloatGeometry) {
  auto g32 = [&](const char *tag, int W, int H, int kw, int kh, int ax, int ay, int margin,
                 uint64_t seed, const std::vector<Npp32f> &hk, int dstep = 0, int sstep = 0) {
    f2_data<Npp32f, Npp32f, Npp32f>(
        "nppiFilter", "nppiFilter_32f_C3R", std::string("geom/32f/") + tag, W, H, 3, kw, kh, ax,
        ay, margin, seed, 0, dst_kind<Npp32f>(), dstep, hk, sstep,
        [](const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize o, const Npp32f *k, NppiSize ks,
           NppiPoint a, NppStreamContext c) {
          return nppiFilter_32f_C3R_Ctx(s, ss, d, ds, o, k, ks, a, c);
        });
  };
  const int m = kMargin;
  g32("a0", 17, 13, 5, 5, 0, 0, m, 0x6201, make_kf(25, 0x6201));
  g32("amax", 17, 13, 5, 5, 5, 5, m, 0x6202, make_kf(25, 0x6202));
  g32("aneg", 17, 13, 5, 5, -3, -3, m, 0x6203, make_kf(25, 0x6203));
  g32("aext", 5, 5, 3, 3, -4, -4, m, 0x6204, make_kf(9, 0x6204));
  g32("asym", 17, 13, 3, 1, 0, 0, m, 0x6205, std::vector<Npp32f>{1.0f, 2.0f, 4.0f});
  g32("k22", 17, 13, 2, 2, 1, 1, m, 0x6206, make_kf(4, 0x6206));
  g32("k99", 17, 13, 9, 9, 4, 4, m, 0x6207, make_kf(81, 0x6207));
  g32("kwide", 13, 9, 23, 3, 11, 1, 12, 0x6208, make_kf(69, 0x6208));
  g32("kevw", 17, 13, 0, 5, 0, 2, m, 0x6209, make_kf(5, 0x6209));
  g32("kevh", 17, 13, 5, 0, 2, 0, m, 0x620A, make_kf(5, 0x620A));
  g32("knegw", 17, 13, -1, 5, -1, 2, m, 0x620B, make_kf(5, 0x620B));
  g32("small", 1, 1, 3, 3, 1, 1, m, 0x620C, make_kf(9, 0x620C));
  g32("emptyw", 0, 13, 5, 5, 2, 2, m, 0x620D, make_kf(25, 0x620D));
  g32("emptyh", 17, 0, 5, 5, 2, 2, m, 0x620E, make_kf(25, 0x620E));
  g32("negstep", 17, 13, 5, 5, 2, 2, m, 0x620F, make_kf(25, 0x620F), -252);
  g32("bigstep", 17, 13, 5, 5, 2, 2, m, 0x6210, make_kf(25, 0x6210), 264);
  g32("sstep_small", 8, 8, 3, 3, 1, 1, m, 0x6211, make_kf(9, 0x6211), 0, 4);
  f2<Npp64f, Npp64f, Npp64f>("nppiFilter", "nppiFilter_64f_C1R", "geom/64f/a0", 17, 13, 1, 5, 5, 0,
                             0, kMargin, 0x6301, dst_kind<Npp64f>(), make_kd(25, 0x6301),
                             [](const Npp64f *s, int ss, Npp64f *d, int ds, NppiSize o,
                                const Npp64f *k, NppiSize ks, NppiPoint a, NppStreamContext c) {
                               return nppiFilter_64f_C1R_Ctx(s, ss, d, ds, o, k, ks, a, c);
                             });
  f2<Npp64f, Npp64f, Npp64f>("nppiFilter", "nppiFilter_64f_C1R", "geom/64f/amax", 17, 13, 1, 5, 5,
                             5, 5, kMargin, 0x6302, dst_kind<Npp64f>(), make_kd(25, 0x6302),
                             [](const Npp64f *s, int ss, Npp64f *d, int ds, NppiSize o,
                                const Npp64f *k, NppiSize ks, NppiPoint a, NppStreamContext c) {
                               return nppiFilter_64f_C1R_Ctx(s, ss, d, ds, o, k, ks, a, c);
                             });
  f2<Npp64f, Npp64f, Npp64f>("nppiFilter", "nppiFilter_64f_C1R", "geom/64f/asym", 17, 13, 1, 3, 1,
                             0, 0, kMargin, 0x6303, dst_kind<Npp64f>(),
                             std::vector<Npp64f>{1.0, 2.0, 4.0},
                             [](const Npp64f *s, int ss, Npp64f *d, int ds, NppiSize o,
                                const Npp64f *k, NppiSize ks, NppiPoint a, NppStreamContext c) {
                               return nppiFilter_64f_C1R_Ctx(s, ss, d, ds, o, k, ks, a, c);
                             });
  f2<Npp64f, Npp64f, Npp64f>("nppiFilter", "nppiFilter_64f_C1R", "geom/64f/k22", 17, 13, 1, 2, 2,
                             1, 1, kMargin, 0x6304, dst_kind<Npp64f>(), make_kd(4, 0x6304),
                             [](const Npp64f *s, int ss, Npp64f *d, int ds, NppiSize o,
                                const Npp64f *k, NppiSize ks, NppiPoint a, NppStreamContext c) {
                               return nppiFilter_64f_C1R_Ctx(s, ss, d, ds, o, k, ks, a, c);
                             });
  f2<Npp64f, Npp64f, Npp64f>("nppiFilter", "nppiFilter_64f_C1R", "geom/64f/emptyw", 0, 13, 1, 5,
                             5, 2, 2, kMargin, 0x6305, dst_kind<Npp64f>(), make_kd(25, 0x6305),
                             [](const Npp64f *s, int ss, Npp64f *d, int ds, NppiSize o,
                                const Npp64f *k, NppiSize ks, NppiPoint a, NppStreamContext c) {
                               return nppiFilter_64f_C1R_Ctx(s, ss, d, ds, o, k, ks, a, c);
                             });
  // 2D 64f degenerate-kernel probes: empty taps along one axis only.
  f2<Npp64f, Npp64f, Npp64f>("nppiFilter", "nppiFilter_64f_C1R", "geom/64f/kevw", 17, 13, 1, 0, 5,
                             0, 2, kMargin, 0x6306, dst_kind<Npp64f>(), make_kd(5, 0x6306),
                             [](const Npp64f *s, int ss, Npp64f *d, int ds, NppiSize o,
                                const Npp64f *k, NppiSize ks, NppiPoint a, NppStreamContext c) {
                               return nppiFilter_64f_C1R_Ctx(s, ss, d, ds, o, k, ks, a, c);
                             });
  f2<Npp64f, Npp64f, Npp64f>("nppiFilter", "nppiFilter_64f_C1R", "geom/64f/kevh", 17, 13, 1, 5, 0,
                             2, 0, kMargin, 0x6307, dst_kind<Npp64f>(), make_kd(5, 0x6307),
                             [](const Npp64f *s, int ss, Npp64f *d, int ds, NppiSize o,
                                const Npp64f *k, NppiSize ks, NppiPoint a, NppStreamContext c) {
                               return nppiFilter_64f_C1R_Ctx(s, ss, d, ds, o, k, ks, a, c);
                             });
}

// AC4 alpha handling, C4 saturation and a couple of extra geometry points.
TEST_F(NppiFilter2dDiff, LayoutGeometry) {
  auto ac4 = [&](const char *tag, int W, int H, int kw, int kh, int ax, int ay, uint64_t seed,
                 const std::vector<Npp32s> &hk, int div = 7) {
    f2<Npp16s, Npp16s, Npp32s>("nppiFilter", "nppiFilter_16s_AC4R",
                               std::string("geom/16s_AC4/") + tag, W, H, 4, kw, kh, ax, ay,
                               kMargin, seed, dst_kind<Npp16s>(), hk,
                               [div](const Npp16s *s, int ss, Npp16s *d, int ds, NppiSize o,
                                     const Npp32s *k, NppiSize ks, NppiPoint a,
                                     NppStreamContext c) {
                                 return nppiFilter_16s_AC4R_Ctx(s, ss, d, ds, o, k, ks, a, div, c);
                               });
  };
  ac4("a0", 17, 13, 5, 5, 0, 0, 0x6401, make_ki(25, 0x6401));
  ac4("amax", 17, 13, 5, 5, 5, 5, 0x6402, make_ki(25, 0x6402));
  ac4("asym", 17, 13, 3, 1, 0, 0, 0x6403, std::vector<Npp32s>{1, 2, 4});
  ac4("kevw", 17, 13, 0, 5, 0, 2, 0x6404, make_ki(5, 0x6404));
  ac4("kevh", 17, 13, 5, 0, 2, 0, 0x6405, make_ki(5, 0x6405));
  ac4("emptyw", 0, 13, 5, 5, 2, 2, 0x6406, make_ki(25, 0x6406));
  ac4("divneg", 17, 13, 5, 5, 2, 2, 0x6407, make_ki(25, 0x6407), -3);
  f2<Npp8u, Npp8u, Npp32s>("nppiFilter", "nppiFilter_8u_C4R", "geom/8u_C4/k22", 17, 13, 4, 2, 2, 1,
                           1, kMargin, 0x6411, dst_kind<Npp8u>(), make_ki(4, 0x6411),
                           [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o,
                              const Npp32s *k, NppiSize ks, NppiPoint a, NppStreamContext c) {
                             return nppiFilter_8u_C4R_Ctx(s, ss, d, ds, o, k, ks, a, 7, c);
                           });
  f2<Npp8u, Npp8u, Npp32s>("nppiFilter", "nppiFilter_8u_C4R", "geom/8u_C4/small", 1, 1, 4, 3, 3, 1,
                           1, kMargin, 0x6412, dst_kind<Npp8u>(), make_ki(9, 0x6412),
                           [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o,
                              const Npp32s *k, NppiSize ks, NppiPoint a, NppStreamContext c) {
                             return nppiFilter_8u_C4R_Ctx(s, ss, d, ds, o, k, ks, a, 7, c);
                           });
  f2<Npp8u, Npp8u, Npp32s>("nppiFilter", "nppiFilter_8u_C4R", "geom/8u_C4/divneg", 17, 13, 4, 5, 5,
                           2, 2, kMargin, 0x6413, dst_kind<Npp8u>(), make_ki(25, 0x6413),
                           [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o,
                              const Npp32s *k, NppiSize ks, NppiPoint a, NppStreamContext c) {
                             return nppiFilter_8u_C4R_Ctx(s, ss, d, ds, o, k, ks, a, -3, c);
                           });
  // Legacy ("real") functions: anchor == kernel size and even kernels are
  // rejected by the legacy validation but accepted by the NPP anchor rule.
  f2<Npp8u, Npp8u, Npp32s>("nppiFilter", "nppiFilter_8u_C1R", "geom/8u_C1/a0", 17, 13, 1, 5, 5, 0,
                           0, kMargin, 0x6421, dst_kind<Npp8u>(), make_ki(25, 0x6421),
                           [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o,
                              const Npp32s *k, NppiSize ks, NppiPoint a, NppStreamContext c) {
                             return nppiFilter_8u_C1R_Ctx(s, ss, d, ds, o, k, ks, a, 7, c);
                           });
  f2<Npp8u, Npp8u, Npp32s>("nppiFilter", "nppiFilter_8u_C1R", "geom/8u_C1/asym", 17, 13, 1, 3, 1,
                           0, 0, kMargin, 0x6422, dst_kind<Npp8u>(), std::vector<Npp32s>{1, 2, 4},
                           [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o,
                              const Npp32s *k, NppiSize ks, NppiPoint a, NppStreamContext c) {
                             return nppiFilter_8u_C1R_Ctx(s, ss, d, ds, o, k, ks, a, 7, c);
                           });
  f2<Npp8u, Npp8u, Npp32s>("nppiFilter", "nppiFilter_8u_C1R", "geom/8u_C1/anchor_eq_kw", 17, 13, 1,
                           3, 3, 1, 1, kMargin, 0x6423, dst_kind<Npp8u>(), make_ki(9, 0x6423),
                           [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o,
                              const Npp32s *k, NppiSize ks, NppiPoint a, NppStreamContext c) {
                             return nppiFilter_8u_C1R_Ctx(s, ss, d, ds, o, k, ks, a, 7, c);
                           });
  f2<Npp8u, Npp8u, Npp32s>("nppiFilter", "nppiFilter_8u_C1R", "geom/8u_C1/k22", 17, 13, 1, 2, 2, 1,
                           1, kMargin, 0x6424, dst_kind<Npp8u>(), make_ki(4, 0x6424),
                           [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o,
                              const Npp32s *k, NppiSize ks, NppiPoint a, NppStreamContext c) {
                             return nppiFilter_8u_C1R_Ctx(s, ss, d, ds, o, k, ks, a, 7, c);
                           });
  f2<Npp32f, Npp32f, Npp32f>("nppiFilter", "nppiFilter_32f_C1R", "geom/32f_C1/asym", 17, 13, 1, 3,
                             1, 0, 0, kMargin, 0x6425, dst_kind<Npp32f>(),
                             std::vector<Npp32f>{1.0f, 2.0f, 4.0f},
                             [](const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize o,
                                const Npp32f *k, NppiSize ks, NppiPoint a, NppStreamContext c) {
                               return nppiFilter_32f_C1R_Ctx(s, ss, d, ds, o, k, ks, a, c);
                             });
  f2<Npp32f, Npp32f, Npp32f>("nppiFilter", "nppiFilter_32f_C1R", "geom/32f_C1/a0", 17, 13, 1, 5, 5,
                             0, 0, kMargin, 0x6426, dst_kind<Npp32f>(), make_kf(25, 0x6426),
                             [](const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize o,
                                const Npp32f *k, NppiSize ks, NppiPoint a, NppStreamContext c) {
                               return nppiFilter_32f_C1R_Ctx(s, ss, d, ds, o, k, ks, a, c);
                             });
  // 32f_C1R shape probes: whether reads past the ROI are clamped into the ROI
  // or taken raw from the surrounding arena depends on the kernel shape
  // (5x5@a(0,0) clamps, 3x1@a(0,0) reads raw).  These pin the boundary.
  auto g1c = [&](const char *tag, int kw, int kh, int ax, int ay, uint64_t seed, size_t nk) {
    f2<Npp32f, Npp32f, Npp32f>("nppiFilter", "nppiFilter_32f_C1R",
                               std::string("geom/32f_C1/") + tag, 17, 13, 1, kw, kh, ax, ay,
                               kMargin, seed, dst_kind<Npp32f>(), make_kf(nk, seed),
                               [](const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize o,
                                  const Npp32f *k, NppiSize ks, NppiPoint a, NppStreamContext c) {
                                 return nppiFilter_32f_C1R_Ctx(s, ss, d, ds, o, k, ks, a, c);
                               });
  };
  g1c("p13", 1, 3, 0, 0, 0x6430, 3);
  g1c("p31", 3, 1, 0, 0, 0x6431, 3);
  g1c("p22", 2, 2, 0, 0, 0x6432, 4);
  g1c("p33", 3, 3, 0, 0, 0x6433, 9);
  g1c("p44", 4, 4, 0, 0, 0x6434, 16);
  g1c("p51", 5, 1, 0, 0, 0x6435, 5);
  g1c("p15", 1, 5, 0, 0, 0x6436, 5);
  g1c("p53", 5, 3, 0, 0, 0x6437, 15);
  g1c("p35", 3, 5, 0, 0, 0x6438, 15);
  g1c("p33c", 3, 3, 2, 2, 0x6439, 9);
  g1c("p55c", 5, 5, 2, 2, 0x643A, 25);
}

// ---------------------------------------------------------------------------
// Integer ring arithmetic: int32 wrap, INT32_MIN / -1, truncation toward
// zero, saturation.
// ---------------------------------------------------------------------------
TEST_F(NppiFilter2dDiff, DivArithmetic) {
  auto p16s = [&](const char *tag, const std::vector<Npp32s> &hk, int kw, int kh, int ax, int ay,
                  int div, int extreme) {
    f2_data<Npp16s, Npp16s, Npp32s>(
        "nppiFilter", "nppiFilter_16s_C1R", std::string("arith/16s/") + tag, 9, 7, 1, kw, kh, ax,
        ay, kMargin, 0x6500 + case_id_++, extreme, dst_kind<Npp16s>(), 0, hk, 0,
        [div](const Npp16s *s, int ss, Npp16s *d, int ds, NppiSize o, const Npp32s *k, NppiSize ks,
              NppiPoint a, NppStreamContext c) {
          return nppiFilter_16s_C1R_Ctx(s, ss, d, ds, o, k, ks, a, div, c);
        });
  };
  p16s("wrap", make_kc(1, 1 << 20), 1, 1, 0, 0, 1, 0);         // wraps int32
  p16s("wrapmin", make_kc(1, -2147483647 - 1), 1, 1, 0, 0, 1, 8);  // INT32_MIN taps
  p16s("divm1", make_kc(1, 1), 1, 1, 0, 0, -1, 8);             // INT32_MIN / -1
  p16s("divm1b", make_kc(1, -1), 1, 1, 0, 0, -1, 8);
  p16s("trunc", make_kc(1, 4), 1, 1, 0, 0, 3, 8);
  p16s("truncneg", make_kc(1, -7), 1, 1, 0, 0, 2, 0);
  p16s("sat", make_kc(9, 4096), 3, 3, 1, 1, 1, 0);             // saturates after wrap
  f2_data<Npp16u, Npp16u, Npp32s>(
      "nppiFilter", "nppiFilter_16u_C1R", "arith/16u/sat", 9, 7, 1, 1, 1, 0, 0, kMargin, 0x6521,
      0, dst_kind<Npp16u>(), 0, make_kc(1, 1024), 0,
      [](const Npp16u *s, int ss, Npp16u *d, int ds, NppiSize o, const Npp32s *k, NppiSize ks,
         NppiPoint a, NppStreamContext c) {
        return nppiFilter_16u_C1R_Ctx(s, ss, d, ds, o, k, ks, a, 1, c);
      });
  f2_data<Npp16u, Npp16u, Npp32s>(
      "nppiFilter", "nppiFilter_16u_C1R", "arith/16u/bigdiv", 9, 7, 1, 1, 1, 0, 0, kMargin, 0x6522,
      0, dst_kind<Npp16u>(), 0, make_kc(1, 1000000), 0,
      [](const Npp16u *s, int ss, Npp16u *d, int ds, NppiSize o, const Npp32s *k, NppiSize ks,
         NppiPoint a, NppStreamContext c) {
        return nppiFilter_16u_C1R_Ctx(s, ss, d, ds, o, k, ks, a, 999983, c);
      });
  auto p8 = [&](const char *tag, Npp32s kv) {
    f2<Npp8u, Npp8u, Npp32s>("nppiFilter", "nppiFilter_8u_C4R", std::string("arith/8u_C4/") + tag,
                             9, 7, 4, 1, 1, 0, 0, kMargin, 0x6530 + case_id_++, dst_kind<Npp8u>(),
                             std::vector<Npp32s>{kv},
                             [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o,
                                const Npp32s *k, NppiSize ks, NppiPoint a, NppStreamContext c) {
                               return nppiFilter_8u_C4R_Ctx(s, ss, d, ds, o, k, ks, a, 1, c);
                             });
  };
  p8("sat_hi", 64);
  p8("sat_lo", -64);
  p8("trunc", 1);
}

// ---------------------------------------------------------------------------
// Error statuses and validation order.
// ---------------------------------------------------------------------------
TEST_F(NppiFilter2dDiff, DivErrors) {
  const auto hk = make_ki(9, 0x6601);
  DeviceMemory<Npp16s> &s = arena<Npp16s>(4096);
  DeviceMemory<Npp16s> &d = arena<Npp16s>(4096);
  DeviceMemory<Npp32s> &k = arena<Npp32s>(9);
  k.copyFromHost(hk);
  // Status probes only: raw taps outside the ROI must stay inside the
  // allocation (MXC faults on reads before the buffer), so start at a margin.
  const Npp16s *ps = s.get() + 64;
  Npp16s *pd = d.get();
  const Npp32s *pk = k.get();
  auto ctx = current_ctx();
  const NppiSize roi{8, 8};
  auto call = [&](const Npp16s *src, int ss, Npp16s *dst, int ds, NppiSize o, NppiSize ks,
                  NppiPoint a, int div) {
    return nppiFilter_16s_C1R_Ctx(src, ss, dst, ds, o, pk, ks, a, div, ctx);
  };
  status_case("nppiFilter", "nppiFilter_16s_C1R", "err/16s/null_src",
              [&] { return call(nullptr, 16, pd, 16, roi, {3, 3}, {1, 1}, 7); });
  status_case("nppiFilter", "nppiFilter_16s_C1R", "err/16s/null_dst",
              [&] { return call(ps, 16, nullptr, 16, roi, {3, 3}, {1, 1}, 7); });
  status_case("nppiFilter", "nppiFilter_16s_C1R", "err/16s/neg_roi",
              [&] { return call(ps, 16, pd, 16, NppiSize{-1, 8}, {3, 3}, {1, 1}, 7); });
  status_case("nppiFilter", "nppiFilter_16s_C1R", "err/16s/neg_roi_vs_anchor",
              [&] { return call(ps, 16, pd, 16, NppiSize{-1, 8}, {3, 3}, {9, 9}, 7); });
  status_case("nppiFilter", "nppiFilter_16s_C1R", "err/16s/anchor_high",
              [&] { return call(ps, 16, pd, 16, roi, {3, 3}, {4, 1}, 7); });
  status_case("nppiFilter", "nppiFilter_16s_C1R", "err/16s/anchor_low",
              [&] { return call(ps, 16, pd, 16, roi, {3, 3}, {-8, 1}, 7); });
  status_case("nppiFilter", "nppiFilter_16s_C1R", "err/16s/anchor_high_vs_step",
              [&] { return call(ps, 16, pd, 15, roi, {3, 3}, {4, 1}, 7); });
  status_case("nppiFilter", "nppiFilter_16s_C1R", "err/16s/anchor_high_vs_div0",
              [&] { return call(ps, 16, pd, 16, roi, {3, 3}, {4, 1}, 0); });
  status_case("nppiFilter", "nppiFilter_16s_C1R", "err/16s/step_small",
              [&] { return call(ps, 16, pd, 15, roi, {3, 3}, {1, 1}, 7); });
  status_case("nppiFilter", "nppiFilter_16s_C1R", "err/16s/step_zero",
              [&] { return call(ps, 16, pd, 0, roi, {3, 3}, {1, 1}, 7); });
  status_case("nppiFilter", "nppiFilter_16s_C1R", "err/16s/step_small_vs_div0",
              [&] { return call(ps, 16, pd, 15, roi, {3, 3}, {1, 1}, 0); });
  status_case("nppiFilter", "nppiFilter_16s_C1R", "err/16s/div0",
              [&] { return call(ps, 16, pd, 16, roi, {3, 3}, {1, 1}, 0); });
  status_case("nppiFilter", "nppiFilter_16s_C1R", "err/16s/div0_vs_negroi",
              [&] { return call(ps, 16, pd, 16, NppiSize{-1, 8}, {3, 3}, {1, 1}, 0); });
  status_case("nppiFilter", "nppiFilter_16s_C1R", "err/16s/div0_vs_empty",
              [&] { return call(ps, 16, pd, 16, NppiSize{0, 8}, {3, 3}, {1, 1}, 0); });
  status_case("nppiFilter", "nppiFilter_16s_C1R", "err/16s/div0_vs_kernel0",
              [&] { return call(ps, 16, pd, 16, roi, {0, 3}, {0, 1}, 0); });
  status_case("nppiFilter", "nppiFilter_16s_C1R", "err/16s/kernel_neg_anchor0",
              [&] { return call(ps, 16, pd, 16, roi, {-1, 3}, {0, 1}, 7); });
  status_case("nppiFilter", "nppiFilter_16s_C1R", "ok/16s/kernel_neg_anchor_ok",
              [&] { return call(ps, 16, pd, 16, roi, {-1, 3}, {-1, 1}, 7); });
  status_case("nppiFilter", "nppiFilter_16s_C1R", "ok/16s/neg_step",
              [&] { return call(ps, 16, pd + 3000, -64, roi, {3, 3}, {1, 1}, 7); });
  status_case("nppiFilter", "nppiFilter_16s_C1R", "ok/16s/empty_roi",
              [&] { return call(ps, 16, pd, 16, NppiSize{0, 8}, {3, 3}, {1, 1}, 7); });
  status_case("nppiFilter", "nppiFilter_16s_C1R", "ok/16s/empty_roi_h",
              [&] { return call(ps, 16, pd, 16, NppiSize{8, 0}, {3, 3}, {1, 1}, 7); });
}

TEST_F(NppiFilter2dDiff, FloatErrors) {
  const auto hk = make_kf(9, 0x6701);
  DeviceMemory<Npp32f> &s = arena<Npp32f>(4096);
  DeviceMemory<Npp32f> &d = arena<Npp32f>(4096);
  DeviceMemory<Npp32f> &k = arena<Npp32f>(9);
  k.copyFromHost(hk);
  const Npp32f *ps = s.get() + 64;
  Npp32f *pd = d.get();
  const Npp32f *pk = k.get();
  auto ctx = current_ctx();
  const NppiSize roi{8, 8};
  auto call = [&](const Npp32f *src, int ss, Npp32f *dst, int ds, NppiSize o, NppiSize ks,
                  NppiPoint a) {
    return nppiFilter_32f_C3R_Ctx(src, ss, dst, ds, o, pk, ks, a, ctx);
  };
  status_case("nppiFilter", "nppiFilter_32f_C3R", "err/32f/null_src",
              [&] { return call(nullptr, 96, pd, 96, roi, {3, 3}, {1, 1}); });
  status_case("nppiFilter", "nppiFilter_32f_C3R", "err/32f/null_dst",
              [&] { return call(ps, 96, nullptr, 96, roi, {3, 3}, {1, 1}); });
  status_case("nppiFilter", "nppiFilter_32f_C3R", "err/32f/neg_roi",
              [&] { return call(ps, 96, pd, 96, NppiSize{-1, 8}, {3, 3}, {1, 1}); });
  status_case("nppiFilter", "nppiFilter_32f_C3R", "err/32f/anchor_high",
              [&] { return call(ps, 96, pd, 96, roi, {3, 3}, {4, 1}); });
  status_case("nppiFilter", "nppiFilter_32f_C3R", "err/32f/anchor_low",
              [&] { return call(ps, 96, pd, 96, roi, {3, 3}, {-8, 1}); });
  status_case("nppiFilter", "nppiFilter_32f_C3R", "err/32f/anchor_high_vs_step",
              [&] { return call(ps, 96, pd, 95, roi, {3, 3}, {4, 1}); });
  status_case("nppiFilter", "nppiFilter_32f_C3R", "err/32f/step_small",
              [&] { return call(ps, 96, pd, 95, roi, {3, 3}, {1, 1}); });
  status_case("nppiFilter", "nppiFilter_32f_C3R", "err/32f/step_zero",
              [&] { return call(ps, 96, pd, 0, roi, {3, 3}, {1, 1}); });
  status_case("nppiFilter", "nppiFilter_32f_C3R", "ok/32f/kernel_neg_anchor_ok",
              [&] { return call(ps, 96, pd, 96, roi, {-1, 3}, {-1, 1}); });
  status_case("nppiFilter", "nppiFilter_32f_C3R", "ok/32f/neg_step",
              [&] { return call(ps, 96, pd + 3000, -192, roi, {3, 3}, {1, 1}); });
  status_case("nppiFilter", "nppiFilter_32f_C3R", "ok/32f/empty_roi",
              [&] { return call(ps, 96, pd, 96, NppiSize{0, 8}, {3, 3}, {1, 1}); });
  DeviceMemory<Npp64f> &s64 = arena<Npp64f>(4096);
  DeviceMemory<Npp64f> &d64 = arena<Npp64f>(4096);
  DeviceMemory<Npp64f> &k64 = arena<Npp64f>(9);
  k64.copyFromHost(make_kd(9, 0x6702));
  const Npp64f *ps64 = s64.get() + 64;
  Npp64f *pd64 = d64.get();
  const Npp64f *pk64 = k64.get();
  status_case("nppiFilter", "nppiFilter_64f_C1R", "err/64f/null_src", [&] {
    return nppiFilter_64f_C1R_Ctx(nullptr, 64, pd64, 64, roi, pk64, {3, 3}, {1, 1}, ctx);
  });
  status_case("nppiFilter", "nppiFilter_64f_C1R", "err/64f/neg_roi", [&] {
    return nppiFilter_64f_C1R_Ctx(ps64, 64, pd64, 64, NppiSize{-1, 8}, pk64, {3, 3}, {1, 1}, ctx);
  });
  status_case("nppiFilter", "nppiFilter_64f_C1R", "err/64f/anchor_high", [&] {
    return nppiFilter_64f_C1R_Ctx(ps64, 64, pd64, 64, roi, pk64, {3, 3}, {4, 1}, ctx);
  });
  status_case("nppiFilter", "nppiFilter_64f_C1R", "err/64f/step_small", [&] {
    return nppiFilter_64f_C1R_Ctx(ps64, 64, pd64, 63, roi, pk64, {3, 3}, {1, 1}, ctx);
  });
  status_case("nppiFilter", "nppiFilter_64f_C1R", "ok/64f/kernel0", [&] {
    return nppiFilter_64f_C1R_Ctx(ps64, 64, pd64, 64, roi, pk64, {0, 3}, {0, 1}, ctx);
  });
  status_case("nppiFilter", "nppiFilter_64f_C1R", "ok/64f/empty_roi", [&] {
    return nppiFilter_64f_C1R_Ctx(ps64, 64, pd64, 64, NppiSize{0, 8}, pk64, {3, 3}, {1, 1}, ctx);
  });
  // 2D 64f anchor lower-bound probes (roi is 8x8, W = 8): 1-W = -7.
  status_case("nppiFilter", "nppiFilter_64f_C1R", "err/64f/anchor_low", [&] {
    return nppiFilter_64f_C1R_Ctx(ps64, 64, pd64, 64, roi, pk64, {3, 3}, {-8, 1}, ctx);
  });
  status_case("nppiFilter", "nppiFilter_64f_C1R", "ok/64f/anchor_1w", [&] {
    return nppiFilter_64f_C1R_Ctx(ps64, 64, pd64, 64, roi, pk64, {3, 3}, {-7, 1}, ctx);
  });
  // Divisor errors on an AC4 variant.
  DeviceMemory<Npp16u> &s2 = arena<Npp16u>(4096);
  DeviceMemory<Npp16u> &d2 = arena<Npp16u>(4096);
  DeviceMemory<Npp32s> &k2 = arena<Npp32s>(9);
  k2.copyFromHost(make_ki(9, 0x6702));
  const Npp16u *ps2 = s2.get() + 64;
  Npp16u *pd2 = d2.get();
  const Npp32s *pk2 = k2.get();
  status_case("nppiFilter", "nppiFilter_16u_AC4R", "err/16u_AC4/div0", [&] {
    return nppiFilter_16u_AC4R_Ctx(ps2, 32, pd2, 32, roi, pk2, {3, 3}, {1, 1}, 0, ctx);
  });
  status_case("nppiFilter", "nppiFilter_16u_AC4R", "err/16u_AC4/div0_vs_empty", [&] {
    return nppiFilter_16u_AC4R_Ctx(ps2, 32, pd2, 32, NppiSize{0, 8}, pk2, {3, 3}, {1, 1}, 0, ctx);
  });
  status_case("nppiFilter", "nppiFilter_16u_AC4R", "err/16u_AC4/step_small", [&] {
    return nppiFilter_16u_AC4R_Ctx(ps2, 32, pd2, 31, roi, pk2, {3, 3}, {1, 1}, 7, ctx);
  });
  status_case("nppiFilter", "nppiFilter_16u_AC4R", "err/16u_AC4/anchor_low", [&] {
    return nppiFilter_16u_AC4R_Ctx(ps2, 32, pd2, 32, roi, pk2, {3, 3}, {-8, 1}, 7, ctx);
  });
}

TEST_F(NppiFilter2dDiff, LegacyErrors) {
  // Probes for the already implemented 8u functions: the legacy validation
  // rejects even kernel sizes, anchors outside the kernel and non-positive
  // steps; NPP 13.4 may differ.
  const auto hk = make_ki(9, 0x6801);
  DeviceMemory<Npp8u> &s = arena<Npp8u>(4096);
  DeviceMemory<Npp8u> &d = arena<Npp8u>(4096);
  DeviceMemory<Npp32s> &k = arena<Npp32s>(9);
  k.copyFromHost(hk);
  const Npp8u *ps = s.get() + 64;
  Npp8u *pd = d.get();
  const Npp32s *pk = k.get();
  auto ctx = current_ctx();
  const NppiSize roi{8, 8};
  auto call = [&](const Npp8u *src, int ss, Npp8u *dst, int ds, NppiSize o, NppiSize ks,
                  NppiPoint a, int div) {
    return nppiFilter_8u_C1R_Ctx(src, ss, dst, ds, o, pk, ks, a, div, ctx);
  };
  status_case("nppiFilter", "nppiFilter_8u_C1R", "err/8u/even_kernel",
              [&] { return call(ps, 16, pd, 16, roi, {2, 2}, {1, 1}, 7); });
  status_case("nppiFilter", "nppiFilter_8u_C1R", "err/8u/anchor_eq_kw",
              [&] { return call(ps, 16, pd, 16, roi, {3, 3}, {3, 1}, 7); });
  status_case("nppiFilter", "nppiFilter_8u_C1R", "err/8u/anchor_neg",
              [&] { return call(ps, 16, pd, 16, roi, {3, 3}, {-1, 1}, 7); });
  status_case("nppiFilter", "nppiFilter_8u_C1R", "err/8u/step_zero",
              [&] { return call(ps, 16, pd, 0, roi, {3, 3}, {1, 1}, 7); });
  status_case("nppiFilter", "nppiFilter_8u_C1R", "ok/8u/neg_step",
              [&] { return call(ps, 16, pd + 3000, -32, roi, {3, 3}, {1, 1}, 7); });
  status_case("nppiFilter", "nppiFilter_8u_C1R", "ok/8u/empty_roi",
              [&] { return call(ps, 16, pd, 16, NppiSize{0, 8}, {3, 3}, {1, 1}, 7); });
  status_case("nppiFilter", "nppiFilter_8u_C1R", "ok/8u/kernel0",
              [&] { return call(ps, 16, pd, 16, roi, {0, 3}, {0, 1}, 7); });
}
