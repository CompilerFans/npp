// Differential tests for package f1: nppiFilter32f (general 2D filter with a
// 32f kernel, no border handling).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiFilter32fDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppiFilter32fDiff.*'
//
// Exactness strategy: source values for 32s are bounded (+-1000) and 32f
// sources are multiples of 1/16, kernels are integer multiples of 1/8, so
// every float partial sum stays below 2^24 and is exact regardless of the
// summation order used by the implementation.

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

// Bounded source values (see the exactness note at the top of the file).
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

// Dyadic float kernels: integer multiples of 1/8 in [-1, 1].
std::vector<Npp32f> make_kf(size_t n, uint64_t seed) {
  Rng rng(seed);
  std::vector<Npp32f> v(n);
  for (auto &x : v) x = static_cast<float>(rng.integer<int>(-8, 8)) * 0.125f;
  return v;
}

template <typename F>
void status_case(const char *fname, const std::string &key, F &&f) {
  clear_cuda_error();
  NppStatus st = f();
  Case c("nppiFilter32f/" + key, fname);
  c.status(st);
  c.finish();
}

// Data case: source ROI WxH sits inside a (W+2M)x(H+2M) buffer with an 8
// pixel margin so that raw (unclamped) taps with |anchor| <= 4 and kernels up
// to 5x5 stay inside the allocation.
constexpr int kMargin = 8;

template <typename S, typename D, typename F>
void f2_data(const char *fname, const std::string &key, int W, int H, int ch, NppiSize ksz,
             NppiPoint an, uint64_t seed, ValueKind kind, const std::vector<Npp32f> &hk, F &&f) {
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
  NppStatus st = f(ps, sStep, d.get(), dStep, NppiSize{W, H}, k.get(), ksz, an, current_ctx());
  Case c("nppiFilter32f/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(S));
  c.input(hk.data(), hk.size() * sizeof(Npp32f));
  c.status(st);
  c.device_output<D>(d.get(), de, kind);
  c.finish();
}

template <typename D>
ValueKind dst_kind() {
  if constexpr (std::is_same_v<D, Npp8u>) return ValueKind::U8;
  else if constexpr (std::is_same_v<D, Npp8s>) return ValueKind::I8;
  else if constexpr (std::is_same_v<D, Npp16u>) return ValueKind::U16;
  else if constexpr (std::is_same_v<D, Npp16s>) return ValueKind::I16;
  else if constexpr (std::is_same_v<D, Npp32s>) return ValueKind::I32;
  else return ValueKind::F32;
}

#define F32_RUN(S, D, TAG, CH)                                                            \
  f2_data<S, D>("nppiFilter32f_" #TAG, "sweep/" #TAG, 17, 13, CH, NppiSize{3, 3},         \
                NppiPoint{1, 1}, 0xF100 + case_id_++, dst_kind<D>(), make_kf(9, 0xF2FF),  \
                [](const S *s, int ss, D *d, int ds, NppiSize o, const Npp32f *k,         \
                   NppiSize kz, NppiPoint a, NppStreamContext c) {                         \
                  return nppiFilter32f_##TAG##_Ctx(s, ss, d, ds, o, k, kz, a, c);         \
                })

#define F32_RUN_SEED(S, D, TAG, CH, W, H, KS, AX, AY, SEED, HK)                            \
  f2_data<S, D>("nppiFilter32f_" #TAG, \
                #TAG "/" #W "x" #H "/k" #KS "/a" #AX "," #AY "/s" #SEED, W, H, \
                CH, NppiSize{KS, KS}, NppiPoint{AX, AY}, SEED, dst_kind<D>(), HK,          \
                [](const S *s, int ss, D *d, int ds, NppiSize o, const Npp32f *k,          \
                   NppiSize kz, NppiPoint a, NppStreamContext c) {                          \
                  return nppiFilter32f_##TAG##_Ctx(s, ss, d, ds, o, k, kz, a, c);          \
                })

}  // namespace

class NppiFilter32fDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
  int case_id_ = 0;
};

// Every non-16f variant, one geometry each (17x13 ROI, 3x3 kernel).
TEST_F(NppiFilter32fDiff, Sweep) {
  F32_RUN(Npp8u, Npp8u, 8u_C1R, 1);
  F32_RUN(Npp8u, Npp8u, 8u_C2R, 2);
  F32_RUN(Npp8u, Npp8u, 8u_C3R, 3);
  F32_RUN(Npp8u, Npp8u, 8u_C4R, 4);
  F32_RUN(Npp8u, Npp8u, 8u_AC4R, 4);
  F32_RUN(Npp8s, Npp8s, 8s_C1R, 1);
  F32_RUN(Npp8s, Npp8s, 8s_C2R, 2);
  F32_RUN(Npp8s, Npp8s, 8s_C3R, 3);
  F32_RUN(Npp8s, Npp8s, 8s_C4R, 4);
  F32_RUN(Npp8s, Npp8s, 8s_AC4R, 4);
  F32_RUN(Npp16u, Npp16u, 16u_C1R, 1);
  F32_RUN(Npp16u, Npp16u, 16u_C3R, 3);
  F32_RUN(Npp16u, Npp16u, 16u_C4R, 4);
  F32_RUN(Npp16u, Npp16u, 16u_AC4R, 4);
  F32_RUN(Npp16s, Npp16s, 16s_C1R, 1);
  F32_RUN(Npp16s, Npp16s, 16s_C3R, 3);
  F32_RUN(Npp16s, Npp16s, 16s_C4R, 4);
  F32_RUN(Npp16s, Npp16s, 16s_AC4R, 4);
  F32_RUN(Npp32s, Npp32s, 32s_C1R, 1);
  F32_RUN(Npp32s, Npp32s, 32s_C3R, 3);
  F32_RUN(Npp32s, Npp32s, 32s_C4R, 4);
  F32_RUN(Npp32s, Npp32s, 32s_AC4R, 4);
  F32_RUN(Npp8u, Npp16s, 8u16s_C1R, 1);
  F32_RUN(Npp8u, Npp16s, 8u16s_C3R, 3);
  F32_RUN(Npp8u, Npp16s, 8u16s_C4R, 4);
  F32_RUN(Npp8u, Npp16s, 8u16s_AC4R, 4);
  F32_RUN(Npp8s, Npp16s, 8s16s_C1R, 1);
  F32_RUN(Npp8s, Npp16s, 8s16s_C3R, 3);
  F32_RUN(Npp8s, Npp16s, 8s16s_C4R, 4);
  F32_RUN(Npp8s, Npp16s, 8s16s_AC4R, 4);
}

// Anchor / kernel-size / geometry variations (8u C3, 16s C1, 32s AC4).
TEST_F(NppiFilter32fDiff, GeometryAndAnchors) {
  const auto k3 = make_kf(9, 0x3001);
  const auto k5 = make_kf(25, 0x3002);
  const auto k1 = std::vector<Npp32f>{0.5f};
  const auto k0 = make_kf(1, 0x3003);  // buffer is non-null; a 0x0 kernel never reads it
  F32_RUN_SEED(Npp8u, Npp8u, 8u_C3R, 3, 8, 8, 3, 0, 0, 0x3011, k3);       // anchor 0,0
  F32_RUN_SEED(Npp8u, Npp8u, 8u_C3R, 3, 8, 8, 3, 3, 3, 0x3012, k3);       // anchor at corner
  F32_RUN_SEED(Npp8u, Npp8u, 8u_C3R, 3, 8, 8, 3, -4, -4, 0x3013, k3);     // negative anchor
  F32_RUN_SEED(Npp8u, Npp8u, 8u_C3R, 3, 7, 5, 3, 2, 1, 0x3014, k3);       // small ROI
  F32_RUN_SEED(Npp8u, Npp8u, 8u_C3R, 3, 1, 1, 3, 3, 3, 0x3015, k3);       // 1x1 ROI
  F32_RUN_SEED(Npp8u, Npp8u, 8u_C3R, 3, 33, 17, 5, 2, 1, 0x3016, k5);     // 5x5 kernel
  F32_RUN_SEED(Npp8u, Npp8u, 8u_C1R, 1, 8, 8, 1, 0, 0, 0x3017, k1);       // 1x1 kernel
  F32_RUN_SEED(Npp8u, Npp8u, 8u_C1R, 1, 8, 8, 0, 0, 0, 0x3018, k0);       // 0x0 kernel -> zeros
  F32_RUN_SEED(Npp16s, Npp16s, 16s_C1R, 1, 12, 9, 3, 1, 1, 0x3021, k3);
  F32_RUN_SEED(Npp16s, Npp16s, 16s_C1R, 1, 12, 9, 3, -2, 2, 0x3022, k3);
  F32_RUN_SEED(Npp32s, Npp32s, 32s_AC4R, 4, 9, 7, 3, 1, 1, 0x3031, k3);
  F32_RUN_SEED(Npp32s, Npp32s, 32s_AC4R, 4, 9, 7, 3, 3, 0, 0x3032, k3);
}

// Truncation and saturation of the float accumulation into integer dsts.
TEST_F(NppiFilter32fDiff, TruncAndSaturate) {
  const std::vector<Npp32f> k15{1.5f};
  const std::vector<Npp32f> k05{0.5f};
  const std::vector<Npp32f> km2{-2.0f};
  const std::vector<Npp32f> k4{4.0f};
  const std::vector<Npp32f> k4e6{4000000.0f};
  F32_RUN_SEED(Npp8s, Npp8s, 8s_C1R, 1, 6, 6, 1, 0, 0, 0x3101, k15);   // -3*1.5 -> -4 (trunc)
  F32_RUN_SEED(Npp8u, Npp8u, 8u_C1R, 1, 6, 6, 1, 0, 0, 0x3102, k05);   // 255*0.5 -> 127
  F32_RUN_SEED(Npp8u, Npp8u, 8u_C1R, 1, 6, 6, 1, 0, 0, 0x3103, k4);    // saturate 255
  F32_RUN_SEED(Npp8s, Npp8s, 8s_C1R, 1, 6, 6, 1, 0, 0, 0x3104, km2);   // saturate -128
  F32_RUN_SEED(Npp16u, Npp16u, 16u_C1R, 1, 6, 6, 1, 0, 0, 0x3105, std::vector<Npp32f>{3.0f});  // 65535
  F32_RUN_SEED(Npp16s, Npp16s, 16s_C1R, 1, 6, 6, 1, 0, 0, 0x3106, std::vector<Npp32f>{2.0f});  // -32768
  F32_RUN_SEED(Npp16s, Npp16s, 16s_C1R, 1, 6, 6, 1, 0, 0, 0x3107, km2);                          // +32767
  F32_RUN_SEED(Npp32s, Npp32s, 32s_C1R, 1, 6, 6, 1, 0, 0, 0x3108, k4e6);  // +-4e9 -> +-INT32_MAX
  F32_RUN_SEED(Npp8u, Npp16s, 8u16s_C1R, 1, 6, 6, 1, 0, 0, 0x3109, k15);  // 250*1.5 -> 375
  F32_RUN_SEED(Npp8s, Npp16s, 8s16s_C1R, 1, 6, 6, 1, 0, 0, 0x310A, std::vector<Npp32f>{-3.0f});
}

// Empty ROIs succeed without touching the destination (anchor >= 1 required
// when the corresponding ROI dimension is 0).
TEST_F(NppiFilter32fDiff, EmptyRoi) {
  const auto k3 = make_kf(9, 0x3201);
  F32_RUN_SEED(Npp8u, Npp8u, 8u_C1R, 1, 0, 4, 3, 1, 1, 0x3201, k3);
  F32_RUN_SEED(Npp8u, Npp8u, 8u_C1R, 1, 4, 0, 3, 1, 1, 0x3202, k3);
  F32_RUN_SEED(Npp8u, Npp8u, 8u_C1R, 1, 0, 0, 3, 1, 1, 0x3203, k3);
}

// Validation order and error statuses (8u C1 representative).
TEST_F(NppiFilter32fDiff, Errors) {
  const auto hk = make_kf(9, 0x3301);
  DeviceMemory<Npp8u> &s = arena<Npp8u>(4096);
  DeviceMemory<Npp8u> &d = arena<Npp8u>(4096);
  DeviceMemory<Npp32f> &k = arena<Npp32f>(9);
  k.copyFromHost(hk);
  const Npp8u *ps = s.get();
  Npp8u *pd = d.get();
  const Npp32f *pk = k.get();
  auto ctx = current_ctx();

  status_case("nppiFilter32f_8u_C1R", "err/null_src", [&] {
    return nppiFilter32f_8u_C1R_Ctx(nullptr, 16, pd, 16, NppiSize{4, 4}, pk, NppiSize{3, 3},
                                    NppiPoint{1, 1}, ctx);
  });
  status_case("nppiFilter32f_8u_C1R", "err/null_dst", [&] {
    return nppiFilter32f_8u_C1R_Ctx(ps, 16, nullptr, 16, NppiSize{4, 4}, pk, NppiSize{3, 3},
                                    NppiPoint{1, 1}, ctx);
  });
  status_case("nppiFilter32f_8u_C1R", "err/neg_roi", [&] {
    return nppiFilter32f_8u_C1R_Ctx(ps, 16, pd, 16, NppiSize{-1, 4}, pk, NppiSize{3, 3},
                                    NppiPoint{1, 1}, ctx);
  });
  status_case("nppiFilter32f_8u_C1R", "err/anchor_high", [&] {
    return nppiFilter32f_8u_C1R_Ctx(ps, 16, pd, 16, NppiSize{4, 4}, pk, NppiSize{3, 3},
                                    NppiPoint{4, 1}, ctx);
  });
  status_case("nppiFilter32f_8u_C1R", "err/anchor_low", [&] {
    return nppiFilter32f_8u_C1R_Ctx(ps, 16, pd, 16, NppiSize{4, 4}, pk, NppiSize{3, 3},
                                    NppiPoint{-4, 1}, ctx);
  });
  status_case("nppiFilter32f_8u_C1R", "err/anchor_neg_kernel", [&] {
    return nppiFilter32f_8u_C1R_Ctx(ps, 16, pd, 16, NppiSize{4, 4}, pk, NppiSize{-1, 3},
                                    NppiPoint{0, 1}, ctx);
  });
  status_case("nppiFilter32f_8u_C1R", "err/step_small", [&] {
    return nppiFilter32f_8u_C1R_Ctx(ps, 16, pd, 3, NppiSize{4, 4}, pk, NppiSize{3, 3},
                                    NppiPoint{1, 1}, ctx);
  });
  status_case("nppiFilter32f_8u_C1R", "err/step_neg", [&] {
    // Negative dst steps pass the unsigned comparison; point into the middle
    // of a large arena so the (successful) launch stays in bounds.
    Npp8u *pd2 = arena<Npp8u>(8192).get() + 512;
    return nppiFilter32f_8u_C1R_Ctx(ps, 16, pd2, -16, NppiSize{4, 4}, pk, NppiSize{3, 3},
                                    NppiPoint{1, 1}, ctx);
  });
  status_case("nppiFilter32f_8u_C1R", "err/neg_roi_beats_anchor", [&] {
    return nppiFilter32f_8u_C1R_Ctx(ps, 16, pd, 16, NppiSize{-1, 4}, pk, NppiSize{3, 3},
                                    NppiPoint{99, 1}, ctx);
  });
}

// 16f variants are refused with NOT_SUFFICIENT_COMPUTE_CAPABILITY before any
// other validation (matches NVIDIA NPP 13.4 on sm_120).
TEST_F(NppiFilter32fDiff, HalfUnsupported) {
  DeviceMemory<Npp8u> &buf = arena<Npp8u>(64);
  DeviceMemory<Npp32f> &k = arena<Npp32f>(9);
  k.copyFromHost(make_kf(9, 0x3401));
  const auto ps = reinterpret_cast<const Npp16f *>(buf.get());
  auto pd = reinterpret_cast<Npp16f *>(buf.get() + 32);
  auto ctx = current_ctx();
  status_case("nppiFilter32f_16f_C1R", "16f/c1", [&] {
    return nppiFilter32f_16f_C1R_Ctx(ps, 8, pd, 8, NppiSize{4, 4}, k.get(), NppiSize{3, 3},
                                     NppiPoint{1, 1}, ctx);
  });
  status_case("nppiFilter32f_16f_C3R", "16f/c3", [&] {
    return nppiFilter32f_16f_C3R_Ctx(ps, 24, pd, 24, NppiSize{4, 4}, k.get(), NppiSize{3, 3},
                                     NppiPoint{1, 1}, ctx);
  });
  status_case("nppiFilter32f_16f_C4R", "16f/c4", [&] {
    return nppiFilter32f_16f_C4R_Ctx(ps, 32, pd, 32, NppiSize{4, 4}, k.get(), NppiSize{3, 3},
                                     NppiPoint{1, 1}, ctx);
  });
}
