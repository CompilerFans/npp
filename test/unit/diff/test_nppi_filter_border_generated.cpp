// Differential tests for package f1: nppiFilterBorder32f (2D filter with a
// 32f kernel and border handling), nppiFilterBorder_32f (32f source) and
// nppiFilterBorder (32s kernel + divisor).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiFilterBorderDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppiFilterBorderDiff.*'
//
// Only NPP_BORDER_REPLICATE is supported by these families (NVIDIA NPP 13.4
// returns NPP_NOT_SUPPORTED_MODE_ERROR for every other mode).  32s sources for
// the 32f-kernel family are bounded and float kernels are dyadic so that all
// float accumulations are exact (see test_nppi_filter32f_generated.cpp).

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
    if constexpr (std::is_same_v<T, Npp32f>) {
      x = static_cast<float>(rng.integer<int>(-64, 63)) * 0.0625f;
    } else if constexpr (std::is_same_v<T, Npp32s>) {
      x = rng.integer<Npp32s>(-1000, 1000);
    } else {
      x = rng.integer<T>(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
    }
  }
  return v;
}

template <typename T> std::vector<T> make_const(size_t n, T value) {
  return std::vector<T>(n, value);
}

std::vector<Npp32f> make_kf(size_t n, uint64_t seed) {
  Rng rng(seed);
  std::vector<Npp32f> v(n);
  for (auto &x : v) x = static_cast<float>(rng.integer<int>(-8, 8)) * 0.125f;
  return v;
}

std::vector<Npp32s> make_ki(size_t n, uint64_t seed) {
  Rng rng(seed);
  std::vector<Npp32s> v(n);
  for (auto &x : v) x = rng.integer<Npp32s>(-3, 3);
  return v;
}

template <typename D> ValueKind dst_kind() {
  if constexpr (std::is_same_v<D, Npp8u>) return ValueKind::U8;
  else if constexpr (std::is_same_v<D, Npp8s>) return ValueKind::I8;
  else if constexpr (std::is_same_v<D, Npp16u>) return ValueKind::U16;
  else if constexpr (std::is_same_v<D, Npp16s>) return ValueKind::I16;
  else if constexpr (std::is_same_v<D, Npp32s>) return ValueKind::I32;
  else return ValueKind::F32;
}

template <typename F>
void status_case(const char *fname, const std::string &key, F &&f) {
  clear_cuda_error();
  NppStatus st = f();
  Case c(std::string(fname) + "/" + key, fname);
  c.status(st);
  c.finish();
}

// ---------------------------------------------------------------------------
// 32f-kernel border family (nppiFilterBorder32f / nppiFilterBorder_32f).
// ---------------------------------------------------------------------------
template <typename S, typename D, typename F>
void fb32_data(const char *fname, const std::string &key, int W, int H, int ch, int sw, int sh,
               int offx, int offy, NppiSize ksz, NppiPoint an, uint64_t seed, ValueKind kind,
               const std::vector<Npp32f> &hk, F &&f) {
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
                   NppiSize{W, H}, k.get(), ksz, an, NPP_BORDER_REPLICATE, current_ctx());
  Case c(std::string(fname) + "/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(S));
  c.input(hk.data(), hk.size() * sizeof(Npp32f));
  c.status(st);
  c.device_output<D>(d.get(), de, kind);
  c.finish();
}

#define FB32_RUN(S, D, TAG, CH)                                                            \
  fb32_data<S, D>("nppiFilterBorder32f_" #TAG, "sweep/" #TAG, 7, 7, CH, 9, 9, 2, 3,        \
                  NppiSize{3, 3}, NppiPoint{1, 1}, 0xF300 + case_id_++, dst_kind<D>(),     \
                  make_kf(9, 0xF3FF),                                                      \
                  [](const S *s, int ss, NppiSize sz, NppiPoint so, D *d, int ds,          \
                     NppiSize o, const Npp32f *k, NppiSize kz, NppiPoint a, NppiBorderType bt, \
                     NppStreamContext c) {                                                 \
                    return nppiFilterBorder32f_##TAG##_Ctx(s, ss, sz, so, d, ds, o, k, kz, a, bt, c); \
                  })

#define FB32_RUNX(S, D, TAG, CH, W, H, SW, SH, OX, OY, KS, AX, AY, SEED, HK)               \
  fb32_data<S, D>("nppiFilterBorder32f_" #TAG, \
                  #TAG "/" #W "x" #H "/src" #SW "x" #SH "/off" #OX "," #OY "/a" #AX "," #AY "/s" #SEED, \
                  W, H, CH, SW, SH, OX, OY, NppiSize{KS, KS}, NppiPoint{AX, AY}, SEED,     \
                  dst_kind<D>(), HK,                                                       \
                  [](const S *s, int ss, NppiSize sz, NppiPoint so, D *d, int ds,          \
                     NppiSize o, const Npp32f *k, NppiSize kz, NppiPoint a, NppiBorderType bt, \
                     NppStreamContext c) {                                                 \
                    return nppiFilterBorder32f_##TAG##_Ctx(s, ss, sz, so, d, ds, o, k, kz, a, bt, c); \
                  })

#define FBRUN_32F(TAG, CH)                                                                 \
  fb32_data<Npp32f, Npp32f>("nppiFilterBorder_" #TAG, "sweep32f/" #TAG, 7, 7, CH, 9, 9, 2, 3, \
                            NppiSize{3, 3}, NppiPoint{1, 1}, 0xF400 + case_id_++,          \
                            ValueKind::F32, make_kf(9, 0xF4FF),                             \
                            [](const Npp32f *s, int ss, NppiSize sz, NppiPoint so, Npp32f *d, int ds, \
                               NppiSize o, const Npp32f *k, NppiSize kz, NppiPoint a,      \
                               NppiBorderType bt, NppStreamContext c) {                    \
                              return nppiFilterBorder_##TAG##_Ctx(s, ss, sz, so, d, ds, o, k, kz, a, bt, c); \
                            })

// ---------------------------------------------------------------------------
// Integer-kernel + divisor border family (nppiFilterBorder).
// ---------------------------------------------------------------------------
template <typename S, typename D, typename F>
void fbd_data(const char *fname, const std::string &key, int W, int H, int ch, int sw, int sh,
              int offx, int offy, NppiSize ksz, NppiPoint an, int divisor, uint64_t seed,
              ValueKind kind, const std::vector<Npp32s> &hk, F &&f) {
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
  DeviceMemory<Npp32s> &k = arena<Npp32s>(hk.size());
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  k.copyFromHost(hk);
  clear_cuda_error();
  const S *ps = s.get() + (size_t)offy * (sStep / sizeof(S)) + (size_t)offx * ch;
  NppStatus st = f(ps, sStep, NppiSize{sw, sh}, NppiPoint{offx, offy}, d.get(), dStep,
                   NppiSize{W, H}, k.get(), ksz, an, divisor, NPP_BORDER_REPLICATE, current_ctx());
  Case c(std::string(fname) + "/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(S));
  c.input(hk.data(), hk.size() * sizeof(Npp32s));
  c.status(st);
  c.device_output<D>(d.get(), de, kind);
  c.finish();
}

const int kDivs[5] = {1, 2, 3, -2, 7};

#define FBD_RUN(S, D, TAG, CH)                                                             \
  fbd_data<S, D>("nppiFilterBorder_" #TAG, "sweep/" #TAG, 7, 7, CH, 9, 9, 2, 3,            \
                 NppiSize{3, 3}, NppiPoint{1, 1}, kDivs[case_id_ % 5], 0xF500 + case_id_++, \
                 dst_kind<D>(), make_ki(9, 0xF5FF),                                        \
                 [](const S *s, int ss, NppiSize sz, NppiPoint so, D *d, int ds,           \
                    NppiSize o, const Npp32s *k, NppiSize kz, NppiPoint a, int dv,         \
                    NppiBorderType bt, NppStreamContext c) {                               \
                   return nppiFilterBorder_##TAG##_Ctx(s, ss, sz, so, d, ds, o, k, kz, a, dv, bt, c); \
                 })

#define FBD_RUNX(S, D, TAG, CH, W, H, SW, SH, OX, OY, KS, AX, AY, DV, SEED, HK)            \
  fbd_data<S, D>("nppiFilterBorder_" #TAG, \
                 #TAG "/" #W "x" #H "/src" #SW "x" #SH "/off" #OX "," #OY "/a" #AX "," #AY "/d" #DV "/s" #SEED, \
                 W, H, CH, SW, SH, OX, OY, NppiSize{KS, KS}, NppiPoint{AX, AY}, DV, SEED,  \
                 dst_kind<D>(), HK,                                                        \
                 [](const S *s, int ss, NppiSize sz, NppiPoint so, D *d, int ds,           \
                    NppiSize o, const Npp32s *k, NppiSize kz, NppiPoint a, int dv,         \
                    NppiBorderType bt, NppStreamContext c) {                               \
                   return nppiFilterBorder_##TAG##_Ctx(s, ss, sz, so, d, ds, o, k, kz, a, dv, bt, c); \
                 })

}  // namespace

class NppiFilterBorderDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
  int case_id_ = 0;
};

// ===== nppiFilterBorder32f: every non-16f variant =========================
TEST_F(NppiFilterBorderDiff, Border32fSweep) {
  FB32_RUN(Npp8u, Npp8u, 8u_C1R, 1);
  FB32_RUN(Npp8u, Npp8u, 8u_C2R, 2);
  FB32_RUN(Npp8u, Npp8u, 8u_C3R, 3);
  FB32_RUN(Npp8u, Npp8u, 8u_C4R, 4);
  FB32_RUN(Npp8u, Npp8u, 8u_AC4R, 4);
  FB32_RUN(Npp8s, Npp8s, 8s_C1R, 1);
  FB32_RUN(Npp8s, Npp8s, 8s_C2R, 2);
  FB32_RUN(Npp8s, Npp8s, 8s_C3R, 3);
  FB32_RUN(Npp8s, Npp8s, 8s_C4R, 4);
  FB32_RUN(Npp8s, Npp8s, 8s_AC4R, 4);
  FB32_RUN(Npp16u, Npp16u, 16u_C1R, 1);
  FB32_RUN(Npp16u, Npp16u, 16u_C3R, 3);
  FB32_RUN(Npp16u, Npp16u, 16u_C4R, 4);
  FB32_RUN(Npp16u, Npp16u, 16u_AC4R, 4);
  FB32_RUN(Npp16s, Npp16s, 16s_C1R, 1);
  FB32_RUN(Npp16s, Npp16s, 16s_C3R, 3);
  FB32_RUN(Npp16s, Npp16s, 16s_C4R, 4);
  FB32_RUN(Npp16s, Npp16s, 16s_AC4R, 4);
  FB32_RUN(Npp32s, Npp32s, 32s_C1R, 1);
  FB32_RUN(Npp32s, Npp32s, 32s_C3R, 3);
  FB32_RUN(Npp32s, Npp32s, 32s_C4R, 4);
  FB32_RUN(Npp32s, Npp32s, 32s_AC4R, 4);
  FB32_RUN(Npp8u, Npp16s, 8u16s_C1R, 1);
  FB32_RUN(Npp8u, Npp16s, 8u16s_C3R, 3);
  FB32_RUN(Npp8u, Npp16s, 8u16s_C4R, 4);
  FB32_RUN(Npp8u, Npp16s, 8u16s_AC4R, 4);
  FB32_RUN(Npp8s, Npp16s, 8s16s_C1R, 1);
  FB32_RUN(Npp8s, Npp16s, 8s16s_C3R, 3);
  FB32_RUN(Npp8s, Npp16s, 8s16s_C4R, 4);
  FB32_RUN(Npp8s, Npp16s, 8s16s_AC4R, 4);
}

// nppiFilterBorder_32f (float sources, float kernel, no divisor).
TEST_F(NppiFilterBorderDiff, Border32fSourceSweep) {
  FBRUN_32F(32f_C1R, 1);
  FBRUN_32F(32f_C2R, 2);
  FBRUN_32F(32f_C3R, 3);
  FBRUN_32F(32f_C4R, 4);
  FBRUN_32F(32f_AC4R, 4);
}

// Offsets / ROI beyond the source / unvalidated anchors / kernel shapes.
TEST_F(NppiFilterBorderDiff, Border32fGeometry) {
  const auto k3 = make_kf(9, 0x5101);
  const auto k1 = make_kf(1, 0x5102);
  const auto k0 = make_kf(1, 0x5103);  // non-null buffer, 0x0 kernel never reads it
  // ROI fully inside a larger source, non-zero offset.
  FB32_RUNX(Npp8u, Npp8u, 8u_C3R, 3, 5, 5, 12, 10, 4, 6, 3, 1, 1, 0x5104, k3);
  // ROI extending beyond the source on every side: replicate clamping.
  FB32_RUNX(Npp8u, Npp8u, 8u_C3R, 3, 9, 9, 4, 4, 1, 1, 3, 1, 1, 0x5105, k3);
  // Offset in the last valid source pixel.
  FB32_RUNX(Npp8u, Npp8u, 8u_C1R, 1, 3, 3, 8, 8, 7, 7, 3, 1, 1, 0x5106, k3);
  // Anchors are NOT validated for the border families (any value is legal).
  FB32_RUNX(Npp8u, Npp8u, 8u_C1R, 1, 4, 4, 6, 6, 1, 1, 3, 9, 9, 0x5107, k3);
  FB32_RUNX(Npp16s, Npp16s, 16s_C1R, 1, 4, 4, 6, 6, 1, 1, 3, -7, 0, 0x5108, k3);
  // 1x1 and 0x0 kernels (0x0 writes zeros).
  FB32_RUNX(Npp8u, Npp8u, 8u_C1R, 1, 4, 4, 6, 6, 1, 1, 1, 0, 0, 0x5109, k1);
  FB32_RUNX(Npp8u, Npp8u, 8u_C1R, 1, 4, 4, 6, 6, 1, 1, 0, 0, 0, 0x510A, k0);
  // 32s saturate through a single large tap.
  FB32_RUNX(Npp32s, Npp32s, 32s_C1R, 1, 4, 4, 6, 6, 1, 1, 1, 0, 0, 0x510B,
            std::vector<Npp32f>{4000000.0f});
  // Mixed dtype, negative kernel taps.
  FB32_RUNX(Npp8s, Npp16s, 8s16s_C3R, 3, 4, 4, 6, 6, 1, 1, 3, 1, 1, 0x510C, k3);
  // Empty ROIs succeed and leave the destination untouched.
  FB32_RUNX(Npp8u, Npp8u, 8u_C1R, 1, 0, 4, 6, 6, 1, 1, 3, 0, 0, 0x510D, k3);
  FB32_RUNX(Npp8u, Npp8u, 8u_C1R, 1, 4, 0, 6, 6, 1, 1, 3, 0, 0, 0x510E, k3);
  FB32_RUNX(Npp8u, Npp8u, 8u_C1R, 1, 0, 0, 6, 6, 1, 1, 3, 0, 0, 0x510F, k3);
}

// nppiFilterBorder32f validation order and error statuses.
TEST_F(NppiFilterBorderDiff, Border32fErrors) {
  const auto hk = make_kf(9, 0x5201);
  DeviceMemory<Npp8u> &s = arena<Npp8u>(4096);
  DeviceMemory<Npp8u> &d = arena<Npp8u>(4096);
  DeviceMemory<Npp32f> &k = arena<Npp32f>(9);
  k.copyFromHost(hk);
  // Status probes only: the replicate border origin sits at pSrc - offset, so
  // the source must start at a margin (MXC faults on reads before the buffer).
  const Npp8u *ps = s.get() + 256;
  Npp8u *pd = d.get();
  // Allocate the negative-step target up front (see NppiFilter32fDiff.Errors).
  Npp8u *pd2 = arena<Npp8u>(8192).get() + 512;
  const Npp32f *pk = k.get();
  auto ctx = current_ctx();
  const NppiSize sz{9, 9};
  const NppiSize roi{7, 7};
  const NppiSize ksz{3, 3};
  const NppiPoint an{1, 1};
  const NppiPoint off{2, 3};

  status_case("nppiFilterBorder32f_8u_C1R", "err/null_src", [&] {
    return nppiFilterBorder32f_8u_C1R_Ctx(nullptr, 16, sz, off, pd, 16, roi, pk, ksz, an,
                                          NPP_BORDER_REPLICATE, ctx);
  });
  status_case("nppiFilterBorder32f_8u_C1R", "err/null_dst", [&] {
    return nppiFilterBorder32f_8u_C1R_Ctx(ps, 16, sz, off, nullptr, 16, roi, pk, ksz, an,
                                          NPP_BORDER_REPLICATE, ctx);
  });
  status_case("nppiFilterBorder32f_8u_C1R", "err/src_size_zero", [&] {
    return nppiFilterBorder32f_8u_C1R_Ctx(ps, 16, NppiSize{0, 9}, off, pd, 16, roi, pk, ksz, an,
                                          NPP_BORDER_REPLICATE, ctx);
  });
  status_case("nppiFilterBorder32f_8u_C1R", "err/src_size_neg", [&] {
    return nppiFilterBorder32f_8u_C1R_Ctx(ps, 16, NppiSize{-1, 9}, off, pd, 16, roi, pk, ksz, an,
                                          NPP_BORDER_REPLICATE, ctx);
  });
  status_case("nppiFilterBorder32f_8u_C1R", "err/neg_roi", [&] {
    return nppiFilterBorder32f_8u_C1R_Ctx(ps, 16, sz, off, pd, 16, NppiSize{-1, 7}, pk, ksz, an,
                                          NPP_BORDER_REPLICATE, ctx);
  });
  status_case("nppiFilterBorder32f_8u_C1R", "err/bad_border", [&] {
    return nppiFilterBorder32f_8u_C1R_Ctx(ps, 16, sz, off, pd, 16, roi, pk, ksz, an,
                                          NPP_BORDER_CONSTANT, ctx);
  });
  status_case("nppiFilterBorder32f_8u_C1R", "err/undef_border", [&] {
    return nppiFilterBorder32f_8u_C1R_Ctx(ps, 16, sz, off, pd, 16, roi, pk, ksz, an,
                                          NPP_BORDER_UNDEFINED, ctx);
  });
  status_case("nppiFilterBorder32f_8u_C1R", "err/size_beats_border", [&] {
    return nppiFilterBorder32f_8u_C1R_Ctx(ps, 16, NppiSize{0, 9}, off, pd, 16, roi, pk, ksz, an,
                                          NPP_BORDER_CONSTANT, ctx);
  });
  status_case("nppiFilterBorder32f_8u_C1R", "err/neg_roi_beats_border", [&] {
    return nppiFilterBorder32f_8u_C1R_Ctx(ps, 16, sz, off, pd, 16, NppiSize{-1, 7}, pk, ksz, an,
                                          NPP_BORDER_CONSTANT, ctx);
  });
  status_case("nppiFilterBorder32f_8u_C1R", "err/offset_x_oob", [&] {
    return nppiFilterBorder32f_8u_C1R_Ctx(ps, 16, sz, NppiPoint{9, 3}, pd, 16, roi, pk, ksz, an,
                                          NPP_BORDER_REPLICATE, ctx);
  });
  status_case("nppiFilterBorder32f_8u_C1R", "err/offset_neg", [&] {
    return nppiFilterBorder32f_8u_C1R_Ctx(ps, 16, sz, NppiPoint{-1, 3}, pd, 16, roi, pk, ksz, an,
                                          NPP_BORDER_REPLICATE, ctx);
  });
  status_case("nppiFilterBorder32f_8u_C1R", "err/offset_empty_roi", [&] {
    return nppiFilterBorder32f_8u_C1R_Ctx(ps, 16, sz, NppiPoint{9, 3}, pd, 16, NppiSize{0, 7}, pk,
                                          ksz, an, NPP_BORDER_REPLICATE, ctx);
  });
  status_case("nppiFilterBorder32f_8u_C1R", "err/step_small", [&] {
    return nppiFilterBorder32f_8u_C1R_Ctx(ps, 16, sz, off, pd, 6, roi, pk, ksz, an,
                                          NPP_BORDER_REPLICATE, ctx);
  });
  status_case("nppiFilterBorder32f_8u_C1R", "err/step_neg", [&] {
    return nppiFilterBorder32f_8u_C1R_Ctx(ps, 16, sz, off, pd2, -16, roi, pk, ksz, an,
                                          NPP_BORDER_REPLICATE, ctx);
  });
}

// nppiFilterBorder32f 16f variants: refused before any validation.
TEST_F(NppiFilterBorderDiff, Border32fHalfUnsupported) {
  DeviceMemory<Npp8u> &buf = arena<Npp8u>(64);
  DeviceMemory<Npp32f> &k = arena<Npp32f>(9);
  k.copyFromHost(make_kf(9, 0x5301));
  const auto ps = reinterpret_cast<const Npp16f *>(buf.get());
  auto pd = reinterpret_cast<Npp16f *>(buf.get() + 32);
  auto ctx = current_ctx();
  status_case("nppiFilterBorder32f_16f_C1R", "16f/c1", [&] {
    return nppiFilterBorder32f_16f_C1R_Ctx(ps, 8, NppiSize{4, 4}, NppiPoint{0, 0}, pd, 8,
                                           NppiSize{4, 4}, k.get(), NppiSize{3, 3}, NppiPoint{1, 1},
                                           NPP_BORDER_REPLICATE, ctx);
  });
  status_case("nppiFilterBorder32f_16f_C3R", "16f/c3", [&] {
    return nppiFilterBorder32f_16f_C3R_Ctx(ps, 24, NppiSize{4, 4}, NppiPoint{0, 0}, pd, 24,
                                           NppiSize{4, 4}, k.get(), NppiSize{3, 3}, NppiPoint{1, 1},
                                           NPP_BORDER_REPLICATE, ctx);
  });
  status_case("nppiFilterBorder32f_16f_C4R", "16f/c4", [&] {
    return nppiFilterBorder32f_16f_C4R_Ctx(ps, 32, NppiSize{4, 4}, NppiPoint{0, 0}, pd, 32,
                                           NppiSize{4, 4}, k.get(), NppiSize{3, 3}, NppiPoint{1, 1},
                                           NPP_BORDER_REPLICATE, ctx);
  });
}

// ===== nppiFilterBorder: integer kernel + divisor =========================
TEST_F(NppiFilterBorderDiff, BorderDivSweep) {
  FBD_RUN(Npp8u, Npp8u, 8u_C1R, 1);
  FBD_RUN(Npp8u, Npp8u, 8u_C3R, 3);
  FBD_RUN(Npp8u, Npp8u, 8u_C4R, 4);
  FBD_RUN(Npp8u, Npp8u, 8u_AC4R, 4);
  FBD_RUN(Npp16u, Npp16u, 16u_C1R, 1);
  FBD_RUN(Npp16u, Npp16u, 16u_C3R, 3);
  FBD_RUN(Npp16u, Npp16u, 16u_C4R, 4);
  FBD_RUN(Npp16u, Npp16u, 16u_AC4R, 4);
  FBD_RUN(Npp16s, Npp16s, 16s_C1R, 1);
  FBD_RUN(Npp16s, Npp16s, 16s_C3R, 3);
  FBD_RUN(Npp16s, Npp16s, 16s_C4R, 4);
  FBD_RUN(Npp16s, Npp16s, 16s_AC4R, 4);
}

// Division truncation toward zero (random signed/unsigned sources exercise
// negative dividends and divisors exactly: -7/2 = -3, -7/-2 = 3, 9/-2 = -4)
// and saturating quotients.  The int32 wrap-around case with a constant
// 65535 source lives in BorderDivWrapConst below.
TEST_F(NppiFilterBorderDiff, BorderDivArithmetic) {
  const auto k1 = std::vector<Npp32s>{1};
  const auto k3 = std::vector<Npp32s>{3};
  FBD_RUNX(Npp16s, Npp16s, 16s_C1R, 1, 4, 4, 6, 6, 1, 1, 1, 0, 0, 2, 0x5401, k1);
  FBD_RUNX(Npp16s, Npp16s, 16s_C1R, 1, 4, 4, 6, 6, 1, 1, 1, 0, 0, -2, 0x5402, k1);
  FBD_RUNX(Npp8u, Npp8u, 8u_C1R, 1, 4, 4, 6, 6, 1, 1, 1, 0, 0, 2, 0x5403, k1);
  FBD_RUNX(Npp8u, Npp8u, 8u_C1R, 1, 4, 4, 6, 6, 1, 1, 1, 0, 0, 1, 0x5404, k3);
  FBD_RUNX(Npp16s, Npp16s, 16s_C1R, 1, 4, 4, 6, 6, 1, 1, 1, 0, 0, -1, 0x5405, k3);
}

// Dedicated int32-wrap case with a constant 65535 source (exact expected
// behaviour: 9 taps of 65535*100000 wrap around int32, then saturate).
TEST_F(NppiFilterBorderDiff, BorderDivWrapConst) {
  const std::vector<Npp32s> hk = make_const<Npp32s>(9, 100000);
  const int sw = 6, sh = 6, W = 4, H = 4, ch = 1;
  const int sStep = align_step(sw * ch, ch);
  const int dStep = align_step(W * ch, ch);
  const size_t se = (size_t)sh * sStep;
  const size_t de = (size_t)H * dStep;
  std::vector<Npp16u> hs = make_const<Npp16u>(se, 65535);
  std::vector<Npp16u> hd = make_uniform<Npp16u>(de, 0x5451);
  DeviceMemory<Npp16u> &s = arena<Npp16u>(se);
  DeviceMemory<Npp16u> &d = arena<Npp16u>(de);
  DeviceMemory<Npp32s> &k = arena<Npp32s>(9);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  k.copyFromHost(hk);
  clear_cuda_error();
  const Npp16u *ps = s.get() + (size_t)1 * (sStep / sizeof(Npp16u)) + 1;
  NppStatus st = nppiFilterBorder_16u_C1R_Ctx(ps, sStep, NppiSize{sw, sh}, NppiPoint{1, 1},
                                              d.get(), dStep, NppiSize{W, H}, k.get(),
                                              NppiSize{3, 3}, NppiPoint{1, 1}, 1,
                                              NPP_BORDER_REPLICATE, current_ctx());
  Case c("nppiFilterBorder_16u_C1R/16u_C1R/wrap_const", "nppiFilterBorder_16u_C1R");
  c.input(hs.data(), hs.size() * sizeof(Npp16u));
  c.input(hk.data(), hk.size() * sizeof(Npp32s));
  c.status(st);
  c.device_output<Npp16u>(d.get(), de, ValueKind::U16);
  c.finish();
}

// nppiFilterBorder error statuses (divisor, mode, offset, size ordering).
TEST_F(NppiFilterBorderDiff, BorderDivErrors) {
  const auto hk = make_ki(9, 0x5501);
  DeviceMemory<Npp8u> &s = arena<Npp8u>(4096);
  DeviceMemory<Npp8u> &d = arena<Npp8u>(4096);
  DeviceMemory<Npp32s> &k = arena<Npp32s>(9);
  k.copyFromHost(hk);
  // Status probes only: the replicate border origin sits at pSrc - offset, so
  // the source must start at a margin (MXC faults on reads before the buffer).
  const Npp8u *ps = s.get() + 256;
  Npp8u *pd = d.get();
  const Npp32s *pk = k.get();
  auto ctx = current_ctx();
  const NppiSize sz{9, 9};
  const NppiSize roi{7, 7};
  const NppiSize ksz{3, 3};
  const NppiPoint an{1, 1};
  const NppiPoint off{2, 3};

  status_case("nppiFilterBorder_8u_C1R", "err/div_zero", [&] {
    return nppiFilterBorder_8u_C1R_Ctx(ps, 16, sz, off, pd, 16, roi, pk, ksz, an, 0,
                                       NPP_BORDER_REPLICATE, ctx);
  });
  status_case("nppiFilterBorder_8u_C1R", "err/mode_beats_div", [&] {
    return nppiFilterBorder_8u_C1R_Ctx(ps, 16, sz, off, pd, 16, roi, pk, ksz, an, 0,
                                       NPP_BORDER_CONSTANT, ctx);
  });
  status_case("nppiFilterBorder_8u_C1R", "err/div_beats_offset", [&] {
    return nppiFilterBorder_8u_C1R_Ctx(ps, 16, sz, NppiPoint{9, 3}, pd, 16, roi, pk, ksz, an, 0,
                                       NPP_BORDER_REPLICATE, ctx);
  });
  status_case("nppiFilterBorder_8u_C1R", "err/offset_y_oob", [&] {
    return nppiFilterBorder_8u_C1R_Ctx(ps, 16, sz, NppiPoint{2, 9}, pd, 16, roi, pk, ksz, an, 1,
                                       NPP_BORDER_REPLICATE, ctx);
  });
  status_case("nppiFilterBorder_8u_C1R", "err/null_src", [&] {
    return nppiFilterBorder_8u_C1R_Ctx(nullptr, 16, sz, off, pd, 16, roi, pk, ksz, an, 1,
                                       NPP_BORDER_REPLICATE, ctx);
  });
  status_case("nppiFilterBorder_8u_C1R", "err/empty_roi_div0", [&] {
    // divisor is checked even for an empty ROI
    return nppiFilterBorder_8u_C1R_Ctx(ps, 16, sz, off, pd, 16, NppiSize{0, 7}, pk, ksz, an, 0,
                                       NPP_BORDER_REPLICATE, ctx);
  });
}
