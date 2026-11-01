// Differential tests for package f9: nppiFilterBox / BoxBorder / Min / Max
// (+Border) / Median (+Border) and the two median GetBufferSize families.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiFilter*Diff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppiFilter*Diff.*'
//
// Sources sit inside a margin large enough for the raw (non-border) window
// reads; destinations are prefilled with 0xAB so the AC4 alpha channel (which
// must stay untouched) is compared as well.
#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

using namespace npp_diff;
using namespace npp_functional_test;
namespace {

constexpr int kW = 9;       // ROI width
constexpr int kH = 7;       // ROI height
constexpr int kMargin = 10; // covers |anchor/mask| up to 9

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

int align_up(int v, int a) { return ((v + a - 1) / a) * a; }

// Integral sources use their full range; floating sources are dyadic
// (multiples of 1/8) so medians/means are exactly representable.  extreme > 0
// sprinkles the type extremes so box saturation is exercised.
template <typename T> std::vector<T> make_src(size_t n, uint64_t seed, int extreme = 0) {
  Rng rng(seed);
  std::vector<T> v(n);
  size_t i = 0;
  for (auto &x : v) {
    if constexpr (std::is_same_v<T, Npp64f>) {
      x = static_cast<T>(rng.integer<int>(-64, 64)) * 0.125;
    } else if constexpr (std::is_floating_point_v<T>) {
      x = static_cast<T>(rng.integer<int>(-64, 64)) * static_cast<T>(0.125);
    } else {
      x = rng.integer<T>(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
      if (extreme > 0) {
        if (i % (size_t)extreme == 0) x = std::numeric_limits<T>::min();
        else if (i % (size_t)extreme == (size_t)extreme / 2) x = std::numeric_limits<T>::max();
      }
    }
    ++i;
  }
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

// Plain (non-border) data case.  The source buffer is a margin-sized image
// and pSrc points at the ROI origin inside it, so the raw window taps stay in
// the allocation.
template <typename S, typename F>
void plain_data(const std::string &family, const char *fname, const std::string &key, int ch,
                NppiSize mask, NppiPoint anchor, uint64_t seed, int extreme, F &&call) {
  const int px = ch * (int)sizeof(S);
  const int stride = align_up((kW + 2 * kMargin) * px, 16);
  const size_t sElems = (size_t)(kH + 2 * kMargin) * (stride / sizeof(S)) + 4;
  const int dstride = align_up(kW * px, 16);
  const size_t dElems = (size_t)kH * (dstride / sizeof(S));
  const size_t dBytes = (size_t)dstride * kH;

  std::vector<S> hs = make_src<S>(sElems, seed, extreme);
  DeviceMemory<S> &s = arena<S>(sElems);
  DeviceMemory<S> &d = arena<S>(dElems ? dElems : 1);
  s.copyFromHost(hs);
  std::vector<unsigned char> fill(dBytes, 0xAB);
  cudaMemcpy(d.get(), fill.data(), dBytes, cudaMemcpyHostToDevice);

  S *ps = s.get() + (size_t)kMargin * (stride / sizeof(S)) + (size_t)kMargin * ch;
  clear_cuda_error();
  NppStatus st = call(ps, stride, d.get(), dstride, NppiSize{kW, kH}, mask, anchor,
                      current_ctx());

  Case c(family + "/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(S));
  c.status(st);
  c.device_output<S>(d.get(), dElems, dst_kind<S>(), 1e-5, 1e-5);
  c.finish();
}

// Border data case: pSrc is the image origin (pSrc - oSrcOffset is the image
// corner contract); the allocation carries a margin around the image.
template <typename S, typename F>
void border_data(const std::string &family, const char *fname, const std::string &key, int ch,
                 NppiSize srcSize, NppiPoint srcOffset, NppiSize mask, NppiPoint anchor,
                 NppiBorderType border, uint64_t seed, int extreme, F &&call) {
  const int px = ch * (int)sizeof(S);
  const int stride = align_up((srcSize.width + 2 * kMargin) * px, 16);
  const size_t sElems = (size_t)(srcSize.height + 2 * kMargin) * (stride / sizeof(S)) + 4;
  const int dstride = align_up(kW * px, 16);
  const size_t dElems = (size_t)kH * (dstride / sizeof(S));
  const size_t dBytes = (size_t)dstride * kH;

  std::vector<S> hs = make_src<S>(sElems, seed, extreme);
  DeviceMemory<S> &s = arena<S>(sElems);
  DeviceMemory<S> &d = arena<S>(dElems ? dElems : 1);
  s.copyFromHost(hs);
  std::vector<unsigned char> fill(dBytes, 0xAB);
  cudaMemcpy(d.get(), fill.data(), dBytes, cudaMemcpyHostToDevice);

  S *ps = s.get() + (size_t)kMargin * (stride / sizeof(S)) + (size_t)kMargin * ch;
  clear_cuda_error();
  NppStatus st = call(ps, stride, srcSize, srcOffset, d.get(), dstride, NppiSize{kW, kH}, mask,
                      anchor, border, current_ctx());

  Case c(family + "/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(S));
  c.status(st);
  c.device_output<S>(d.get(), dElems, dst_kind<S>(), 1e-5, 1e-5);
  c.finish();
}

// ---------------------------------------------------------------------------
// per-family runners
// ---------------------------------------------------------------------------
#define BOX_PLAIN(S, TAG, CH)                                                                  \
  plain_data<S>("nppiFilterBox", "nppiFilterBox_" #TAG, "sweep/" #TAG, CH, NppiSize{3, 3},     \
                NppiPoint{1, 1}, 0x1000u + (uint64_t)CH, 0,                                    \
                [](const S *s, int ss, S *d, int ds, NppiSize r, NppiSize m, NppiPoint a,      \
                   NppStreamContext c) {                                                       \
                  return nppiFilterBox_##TAG##_Ctx(s, ss, d, ds, r, m, a, c);                  \
                });                                                                            \
  static_assert(true, "");
#define BOX_BORDER(S, TAG, CH)                                                                 \
  border_data<S>("nppiFilterBoxBorder", "nppiFilterBoxBorder_" #TAG, "sweep/" #TAG, CH,        \
                 NppiSize{13, 11}, NppiPoint{2, 3}, NppiSize{3, 3}, NppiPoint{1, 1},           \
                 NPP_BORDER_REPLICATE, 0x2000u + (uint64_t)CH, 0,                             \
                 [](const S *s, int ss, NppiSize sz, NppiPoint o, S *d, int ds, NppiSize r,    \
                    NppiSize m, NppiPoint a, NppiBorderType b, NppStreamContext c) {           \
                   return nppiFilterBoxBorder_##TAG##_Ctx(s, ss, sz, o, d, ds, r, m, a, b, c); \
                 });                                                                           \
  static_assert(true, "");
#define MIN_PLAIN(S, TAG, CH)                                                                  \
  plain_data<S>("nppiFilterMin", "nppiFilterMin_" #TAG, "sweep/" #TAG, CH, NppiSize{3, 3},     \
                NppiPoint{1, 1}, 0x3000u + (uint64_t)CH, 0,                                    \
                [](const S *s, int ss, S *d, int ds, NppiSize r, NppiSize m, NppiPoint a,      \
                   NppStreamContext c) {                                                       \
                  return nppiFilterMin_##TAG##_Ctx(s, ss, d, ds, r, m, a, c);                  \
                });                                                                            \
  static_assert(true, "");
#define MAX_PLAIN(S, TAG, CH)                                                                  \
  plain_data<S>("nppiFilterMax", "nppiFilterMax_" #TAG, "sweep/" #TAG, CH, NppiSize{3, 3},     \
                NppiPoint{1, 1}, 0x4000u + (uint64_t)CH, 0,                                    \
                [](const S *s, int ss, S *d, int ds, NppiSize r, NppiSize m, NppiPoint a,      \
                   NppStreamContext c) {                                                       \
                  return nppiFilterMax_##TAG##_Ctx(s, ss, d, ds, r, m, a, c);                  \
                });                                                                            \
  static_assert(true, "");
#define MIN_BORDER(S, TAG, CH)                                                                 \
  border_data<S>("nppiFilterMinBorder", "nppiFilterMinBorder_" #TAG, "sweep/" #TAG, CH,        \
                 NppiSize{13, 11}, NppiPoint{2, 3}, NppiSize{3, 3}, NppiPoint{1, 1},           \
                 NPP_BORDER_REPLICATE, 0x5000u + (uint64_t)CH, 0,                             \
                 [](const S *s, int ss, NppiSize sz, NppiPoint o, S *d, int ds, NppiSize r,    \
                    NppiSize m, NppiPoint a, NppiBorderType b, NppStreamContext c) {           \
                   return nppiFilterMinBorder_##TAG##_Ctx(s, ss, sz, o, d, ds, r, m, a, b, c); \
                 });                                                                           \
  static_assert(true, "");
#define MAX_BORDER(S, TAG, CH)                                                                 \
  border_data<S>("nppiFilterMaxBorder", "nppiFilterMaxBorder_" #TAG, "sweep/" #TAG, CH,        \
                 NppiSize{13, 11}, NppiPoint{2, 3}, NppiSize{3, 3}, NppiPoint{1, 1},           \
                 NPP_BORDER_REPLICATE, 0x6000u + (uint64_t)CH, 0,                             \
                 [](const S *s, int ss, NppiSize sz, NppiPoint o, S *d, int ds, NppiSize r,    \
                    NppiSize m, NppiPoint a, NppiBorderType b, NppStreamContext c) {           \
                   return nppiFilterMaxBorder_##TAG##_Ctx(s, ss, sz, o, d, ds, r, m, a, b, c); \
                 });                                                                           \
  static_assert(true, "");
#define MEDIAN_PLAIN(S, TAG, CH)                                                               \
  plain_data<S>("nppiFilterMedian", "nppiFilterMedian_" #TAG, "sweep/" #TAG, CH, NppiSize{3, 3},\
                NppiPoint{1, 1}, 0x7000u + (uint64_t)CH, 0,                                    \
                [](const S *s, int ss, S *d, int ds, NppiSize r, NppiSize m, NppiPoint a,      \
                   NppStreamContext c) {                                                       \
                  return nppiFilterMedian_##TAG##_Ctx(s, ss, d, ds, r, m, a, nullptr, c);      \
                });                                                                            \
  static_assert(true, "");
#define MEDIAN_BORDER(S, TAG, CH)                                                              \
  border_data<S>("nppiFilterMedianBorder", "nppiFilterMedianBorder_" #TAG, "sweep/" #TAG, CH,  \
                 NppiSize{13, 11}, NppiPoint{2, 3}, NppiSize{3, 3}, NppiPoint{1, 1},           \
                 NPP_BORDER_REPLICATE, 0x8000u + (uint64_t)CH, 0,                             \
                 [](const S *s, int ss, NppiSize sz, NppiPoint o, S *d, int ds, NppiSize r,    \
                    NppiSize m, NppiPoint a, NppiBorderType b, NppStreamContext c) {           \
                   return nppiFilterMedianBorder_##TAG##_Ctx(s, ss, sz, o, d, ds, r, m, a,      \
                                                             nullptr, b, c);                   \
                 });                                                                           \
  static_assert(true, "");

#define ALL_VARIANTS(FN)                                                                       \
  FN(Npp8u, 8u_C1R, 1)                                                                         \
  FN(Npp8u, 8u_C3R, 3)                                                                         \
  FN(Npp8u, 8u_C4R, 4)                                                                         \
  FN(Npp8u, 8u_AC4R, 4)                                                                        \
  FN(Npp16u, 16u_C1R, 1)                                                                       \
  FN(Npp16u, 16u_C3R, 3)                                                                       \
  FN(Npp16u, 16u_C4R, 4)                                                                       \
  FN(Npp16u, 16u_AC4R, 4)                                                                      \
  FN(Npp16s, 16s_C1R, 1)                                                                       \
  FN(Npp16s, 16s_C3R, 3)                                                                       \
  FN(Npp16s, 16s_C4R, 4)                                                                       \
  FN(Npp16s, 16s_AC4R, 4)                                                                      \
  FN(Npp32f, 32f_C1R, 1)                                                                       \
  FN(Npp32f, 32f_C3R, 3)                                                                       \
  FN(Npp32f, 32f_C4R, 4)                                                                       \
  FN(Npp32f, 32f_AC4R, 4)                                                                        \

}  // namespace

class NppiFilterBoxDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

TEST_F(NppiFilterBoxDiff, Sweep) {
  ALL_VARIANTS(BOX_PLAIN);
  plain_data<Npp8u>("nppiFilterBox", "nppiFilterBox_8u_C1R", "geometry/5x3_a21", 1,
                    NppiSize{5, 3}, NppiPoint{2, 1}, 0x1A01, 0,
                    [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize r, NppiSize m,
                       NppiPoint a, NppStreamContext c) {
                      return nppiFilterBox_8u_C1R_Ctx(s, ss, d, ds, r, m, a, c);
                    });
  plain_data<Npp8u>("nppiFilterBox", "nppiFilterBox_8u_C1R", "geometry/2x2_a10", 1,
                    NppiSize{2, 2}, NppiPoint{1, 0}, 0x1A02, 0,
                    [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize r, NppiSize m,
                       NppiPoint a, NppStreamContext c) {
                      return nppiFilterBox_8u_C1R_Ctx(s, ss, d, ds, r, m, a, c);
                    });
  plain_data<Npp64f>("nppiFilterBox", "nppiFilterBox_64f_C1R", "sweep/64f_C1R", 1,
                     NppiSize{3, 3}, NppiPoint{1, 1}, 0x1A04, 0,
                     [](const Npp64f *s, int ss, Npp64f *d, int ds, NppiSize r, NppiSize m,
                        NppiPoint a, NppStreamContext c) {
                       return nppiFilterBox_64f_C1R_Ctx(s, ss, d, ds, r, m, a, c);
                     });
  plain_data<Npp16s>("nppiFilterBox", "nppiFilterBox_16s_C1R", "extreme/sat", 1,
                     NppiSize{5, 5}, NppiPoint{2, 2}, 0x1A03, 7,
                     [](const Npp16s *s, int ss, Npp16s *d, int ds, NppiSize r, NppiSize m,
                        NppiPoint a, NppStreamContext c) {
                       return nppiFilterBox_16s_C1R_Ctx(s, ss, d, ds, r, m, a, c);
                     });
}

TEST_F(NppiFilterBoxDiff, BorderSweep) {
  ALL_VARIANTS(BOX_BORDER);
  // MIRROR (4) and the undocumented value 5 are accepted and behave as the
  // two reflection conventions.
  border_data<Npp8u>("nppiFilterBoxBorder", "nppiFilterBoxBorder_8u_C1R", "border/mirror", 1,
                     NppiSize{13, 11}, NppiPoint{2, 3}, NppiSize{5, 5}, NppiPoint{2, 2},
                     NPP_BORDER_MIRROR, 0x2A01, 0,
                     [](const Npp8u *s, int ss, NppiSize sz, NppiPoint o, Npp8u *d, int ds,
                        NppiSize r, NppiSize m, NppiPoint a, NppiBorderType b,
                        NppStreamContext c) {
                       return nppiFilterBoxBorder_8u_C1R_Ctx(s, ss, sz, o, d, ds, r, m, a, b, c);
                     });
  border_data<Npp8u>("nppiFilterBoxBorder", "nppiFilterBoxBorder_8u_C1R", "border/mirror5", 1,
                     NppiSize{13, 11}, NppiPoint{2, 3}, NppiSize{3, 3}, NppiPoint{1, 1},
                     (NppiBorderType)5, 0x2A02, 0,
                     [](const Npp8u *s, int ss, NppiSize sz, NppiPoint o, Npp8u *d, int ds,
                        NppiSize r, NppiSize m, NppiPoint a, NppiBorderType b,
                        NppStreamContext c) {
                       return nppiFilterBoxBorder_8u_C1R_Ctx(s, ss, sz, o, d, ds, r, m, a, b, c);
                     });
  border_data<Npp8u>("nppiFilterBoxBorder", "nppiFilterBoxBorder_8u_C1R", "geometry/off0", 1,
                     NppiSize{13, 11}, NppiPoint{0, 0}, NppiSize{4, 2}, NppiPoint{3, 0},
                     NPP_BORDER_REPLICATE, 0x2A03, 0,
                     [](const Npp8u *s, int ss, NppiSize sz, NppiPoint o, Npp8u *d, int ds,
                        NppiSize r, NppiSize m, NppiPoint a, NppiBorderType b,
                        NppStreamContext c) {
                       return nppiFilterBoxBorder_8u_C1R_Ctx(s, ss, sz, o, d, ds, r, m, a, b, c);
                     });
}

class NppiFilterMinDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};
TEST_F(NppiFilterMinDiff, Sweep) {
  ALL_VARIANTS(MIN_PLAIN);
  plain_data<Npp16s>("nppiFilterMin", "nppiFilterMin_16s_C1R", "geometry/5x3_a21", 1,
                     NppiSize{5, 3}, NppiPoint{2, 1}, 0x3A01, 0,
                     [](const Npp16s *s, int ss, Npp16s *d, int ds, NppiSize r, NppiSize m,
                        NppiPoint a, NppStreamContext c) {
                       return nppiFilterMin_16s_C1R_Ctx(s, ss, d, ds, r, m, a, c);
                     });
  plain_data<Npp8u>("nppiFilterMin", "nppiFilterMin_8u_C3R", "geometry/2x2_a10", 3,
                    NppiSize{2, 2}, NppiPoint{1, 0}, 0x3A02, 0,
                    [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize r, NppiSize m,
                       NppiPoint a, NppStreamContext c) {
                      return nppiFilterMin_8u_C3R_Ctx(s, ss, d, ds, r, m, a, c);
                    });
}
class NppiFilterMaxDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};
TEST_F(NppiFilterMaxDiff, Sweep) {
  ALL_VARIANTS(MAX_PLAIN);
  plain_data<Npp16s>("nppiFilterMax", "nppiFilterMax_16s_C1R", "geometry/5x3_a21", 1,
                     NppiSize{5, 3}, NppiPoint{2, 1}, 0x4A01, 0,
                     [](const Npp16s *s, int ss, Npp16s *d, int ds, NppiSize r, NppiSize m,
                        NppiPoint a, NppStreamContext c) {
                       return nppiFilterMax_16s_C1R_Ctx(s, ss, d, ds, r, m, a, c);
                     });
  plain_data<Npp16u>("nppiFilterMax", "nppiFilterMax_16u_C1R", "anchor/zero", 1,
                     NppiSize{3, 3}, NppiPoint{0, 0}, 0x4A02, 0,
                     [](const Npp16u *s, int ss, Npp16u *d, int ds, NppiSize r, NppiSize m,
                        NppiPoint a, NppStreamContext c) {
                       return nppiFilterMax_16u_C1R_Ctx(s, ss, d, ds, r, m, a, c);
                     });
}
class NppiFilterMinBorderDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};
TEST_F(NppiFilterMinBorderDiff, Sweep) { ALL_VARIANTS(MIN_BORDER); }
class NppiFilterMaxBorderDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};
TEST_F(NppiFilterMaxBorderDiff, Sweep) { ALL_VARIANTS(MAX_BORDER); }

class NppiFilterMedianDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};
TEST_F(NppiFilterMedianDiff, Sweep) {
  ALL_VARIANTS(MEDIAN_PLAIN);
  // even tap count: the two middle samples are averaged (trunc toward zero)
  plain_data<Npp8u>("nppiFilterMedian", "nppiFilterMedian_8u_C1R", "geometry/5x3_a21", 1,
                    NppiSize{5, 3}, NppiPoint{2, 1}, 0x7A01, 0,
                    [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize r, NppiSize m,
                       NppiPoint a, NppStreamContext c) {
                      return nppiFilterMedian_8u_C1R_Ctx(s, ss, d, ds, r, m, a, nullptr, c);
                    });
  plain_data<Npp16s>("nppiFilterMedian", "nppiFilterMedian_16s_C1R", "geometry/3x5_a11", 1,
                     NppiSize{3, 5}, NppiPoint{1, 2}, 0x7A02, 0,
                     [](const Npp16s *s, int ss, Npp16s *d, int ds, NppiSize r, NppiSize m,
                        NppiPoint a, NppStreamContext c) {
                       return nppiFilterMedian_16s_C1R_Ctx(s, ss, d, ds, r, m, a, nullptr, c);
                     });
  plain_data<Npp32f>("nppiFilterMedian", "nppiFilterMedian_32f_C1R", "geometry/5x5_a22", 1,
                     NppiSize{5, 5}, NppiPoint{2, 2}, 0x7A03, 0,
                     [](const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize r, NppiSize m,
                        NppiPoint a, NppStreamContext c) {
                       return nppiFilterMedian_32f_C1R_Ctx(s, ss, d, ds, r, m, a, nullptr, c);
                     });
}

class NppiFilterMedianBorderDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};
TEST_F(NppiFilterMedianBorderDiff, Sweep) {
  ALL_VARIANTS(MEDIAN_BORDER);
  border_data<Npp8u>("nppiFilterMedianBorder", "nppiFilterMedianBorder_8u_C1R", "border/mirror",
                     1, NppiSize{13, 11}, NppiPoint{2, 3}, NppiSize{3, 1}, NppiPoint{1, 0},
                     NPP_BORDER_MIRROR, 0x8A01, 0,
                     [](const Npp8u *s, int ss, NppiSize sz, NppiPoint o, Npp8u *d, int ds,
                        NppiSize r, NppiSize m, NppiPoint a, NppiBorderType b,
                        NppStreamContext c) {
                       return nppiFilterMedianBorder_8u_C1R_Ctx(s, ss, sz, o, d, ds, r, m, a,
                                                               nullptr, b, c);
                     });
  border_data<Npp16s>("nppiFilterMedianBorder", "nppiFilterMedianBorder_16s_C1R", "geometry/3x5_a12",
                      1, NppiSize{13, 11}, NppiPoint{2, 3}, NppiSize{3, 5}, NppiPoint{1, 2},
                      NPP_BORDER_REPLICATE, 0x8A02, 0,
                      [](const Npp16s *s, int ss, NppiSize sz, NppiPoint o, Npp16s *d, int ds,
                         NppiSize r, NppiSize m, NppiPoint a, NppiBorderType b,
                         NppStreamContext c) {
                        return nppiFilterMedianBorder_16s_C1R_Ctx(s, ss, sz, o, d, ds, r, m, a,
                                                                  nullptr, b, c);
                      });
}

// ---------------------------------------------------------------------------
// buffered median path (mask 16x16 => GetBufferSize > 0 for every config)
// ---------------------------------------------------------------------------
namespace {

template <typename S, typename BufFn, typename Call>
void median_buffered(const std::string &family, const char *fname, const std::string &key, int ch,
                     NppiSize mask, NppiPoint anchor, uint64_t seed, BufFn &&buf_size,
                     Call &&call) {
  const int px = ch * (int)sizeof(S);
  const int stride = align_up((kW + 2 * kMargin) * px, 16);
  const size_t sElems = (size_t)(kH + 2 * kMargin) * (stride / sizeof(S)) + 4;
  const int dstride = align_up(kW * px, 16);
  const size_t dElems = (size_t)kH * (dstride / sizeof(S));
  const size_t dBytes = (size_t)dstride * kH;

  Npp32u bsz = 0;
  NppStatus bst = buf_size(mask, &bsz);
  EXPECT_EQ(bst, NPP_NO_ERROR);

  std::vector<S> hs = make_src<S>(sElems, seed, 0);
  DeviceMemory<S> &s = arena<S>(sElems);
  DeviceMemory<S> &d = arena<S>(dElems ? dElems : 1);
  DeviceMemory<Npp8u> &buf = arena<Npp8u>(bsz ? bsz : 1);
  s.copyFromHost(hs);
  std::vector<unsigned char> fill(dBytes, 0xAB);
  cudaMemcpy(d.get(), fill.data(), dBytes, cudaMemcpyHostToDevice);

  S *ps = s.get() + (size_t)kMargin * (stride / sizeof(S)) + (size_t)kMargin * ch;
  clear_cuda_error();
  NppStatus st = call(ps, stride, d.get(), dstride, NppiSize{kW, kH}, mask, anchor, buf.get(),
                      current_ctx());

  Case c(family + "/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(S));
  c.status(st);
  // NPP's 16s even-median tie rounding is not self-consistent (the same
  // middle-sum -327 rounds both ways across windows), so allow 1 LSB there.
  ValueKind out_kind = dst_kind<S>();
  double atol = 1e-5;
  if constexpr (std::is_same_v<S, Npp16s>) {
    out_kind = ValueKind::I16Tol;
    atol = 1.0;
  }
  c.device_output<S>(d.get(), dElems, out_kind, 1e-5, atol);
  c.finish();
}

#define MEDIAN_BUF(TAG, S)                                                                     \
  median_buffered<S>(                                                                          \
      "nppiFilterMedian", "nppiFilterMedian_" #TAG, "buffered/" #TAG "/16x16", 1, NppiSize{16, 16},     \
      NppiPoint{8, 8}, 0x9100u,                                                                \
      [](NppiSize m, Npp32u *b) {                                                              \
        return nppiFilterMedianGetBufferSize_##TAG##_Ctx(NppiSize{kW, kH}, m, b, current_ctx());\
      },                                                                                     \
      [](const S *s, int ss, S *d, int ds, NppiSize r, NppiSize m, NppiPoint a, Npp8u *b,      \
         NppStreamContext c) { return nppiFilterMedian_##TAG##_Ctx(s, ss, d, ds, r, m, a, b, c); })

}  // namespace

TEST_F(NppiFilterMedianDiff, BufferedPath) {
  MEDIAN_BUF(8u_C1R, Npp8u);
  MEDIAN_BUF(16u_C1R, Npp16u);
  MEDIAN_BUF(16s_C1R, Npp16s);
  MEDIAN_BUF(32f_C1R, Npp32f);
}

// ---------------------------------------------------------------------------
// error / validation paths (status-only goldens)
// ---------------------------------------------------------------------------
namespace {
template <typename T> T *devbuf(size_t n) { return arena<T>(n).get(); }
}  // namespace

TEST_F(NppiFilterBoxDiff, Validation) {
  const std::string fam = "nppiFilterBox";
  const char *fn = "nppiFilterBox_8u_C1R";
  Npp8u *s = devbuf<Npp8u>(4096);
  Npp8u *d = devbuf<Npp8u>(4096);
  Npp8u *s16 = devbuf<Npp8u>(4096);
  NppStreamContext c = current_ctx();
  const NppiSize roi{9, 7}, m3{3, 3};
  const NppiPoint a1{1, 1};
  status_case(fam, fn, "null_src", [&] { return nppiFilterBox_8u_C1R_Ctx(nullptr, 32, d, 32, roi, m3, a1, c); });
  status_case(fam, fn, "null_dst", [&] { return nppiFilterBox_8u_C1R_Ctx(s, 32, nullptr, 32, roi, m3, a1, c); });
  status_case(fam, fn, "roi_zero", [&] { return nppiFilterBox_8u_C1R_Ctx(s, 32, d, 32, NppiSize{0, 7}, m3, a1, c); });
  status_case(fam, fn, "roi_neg", [&] { return nppiFilterBox_8u_C1R_Ctx(s, 32, d, 32, NppiSize{-3, 7}, m3, a1, c); });
  status_case(fam, fn, "mask_zero", [&] { return nppiFilterBox_8u_C1R_Ctx(s, 32, d, 32, roi, NppiSize{0, 0}, a1, c); });
  status_case(fam, fn, "mask_neg", [&] { return nppiFilterBox_8u_C1R_Ctx(s, 32, d, 32, roi, NppiSize{-3, 3}, a1, c); });
  status_case(fam, fn, "mask_gt_roi", [&] { return nppiFilterBox_8u_C1R_Ctx(s, 32, d, 32, {4, 4}, {9, 5}, a1, c); });
  status_case(fam, fn, "dstep_small", [&] { return nppiFilterBox_8u_C1R_Ctx(s, 32, d, 8, roi, m3, a1, c); });
  status_case(fam, fn, "anchor_gt", [&] { return nppiFilterBox_8u_C1R_Ctx(s, 32, d, 32, roi, m3, NppiPoint{3, 1}, c); });
  status_case(fam, "nppiFilterBox_16u_C1R", "roi_zero_16u", [&] {
    return nppiFilterBox_16u_C1R_Ctx((Npp16u *)s16, 32, (Npp16u *)d, 32, NppiSize{0, 7}, m3, a1, c);
  });
  status_case(fam, "nppiFilterBox_8u_C4R", "roi_zero_c4", [&] {
    return nppiFilterBox_8u_C4R_Ctx(s16, 32, d, 32, NppiSize{0, 7}, m3, a1, c);
  });
  status_case(fam, "nppiFilterBox_8u_C1R", "mask_gt_roi_ok", [&] {
    return nppiFilterBox_8u_C1R_Ctx(s, 32, d, 32, NppiSize{9, 7}, NppiSize{11, 9}, NppiPoint{5, 4}, c);
  });
  const char *fn16 = "nppiFilterBox_16u_C1R";
  status_case(fam, fn16, "dstep_odd_16u", [&] { return nppiFilterBox_16u_C1R_Ctx((Npp16u *)s16, 32, (Npp16u *)d, 19, roi, m3, a1, c); });
  status_case(fam, fn16, "dstep_neg_16u", [&] { return nppiFilterBox_16u_C1R_Ctx((Npp16u *)s16, 32, (Npp16u *)d, -1, NppiSize{1, 1}, m3, a1, c); });
}

class NppiFilterBoxBorderDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

TEST_F(NppiFilterBoxBorderDiff, Validation) {
  const std::string fam = "nppiFilterBoxBorder";
  const char *fn = "nppiFilterBoxBorder_8u_C1R";
  Npp8u *s = devbuf<Npp8u>(4096);
  Npp8u *d = devbuf<Npp8u>(4096);
  NppStreamContext c = current_ctx();
  const NppiSize src{13, 11}, roi{9, 7}, m3{3, 3};
  const NppiPoint off{2, 3}, a1{1, 1};
  auto call = [&](NppiSize sr, NppiPoint o, int ds, NppiBorderType b) {
    return nppiFilterBoxBorder_8u_C1R_Ctx(s, 32, sr, o, d, ds, roi, m3, a1, b, c);
  };
  status_case(fam, fn, "null_src", [&] { return nppiFilterBoxBorder_8u_C1R_Ctx(nullptr, 32, src, off, d, 32, roi, m3, a1, NPP_BORDER_REPLICATE, c); });
  status_case(fam, fn, "border_undef", [&] { return call(src, off, 32, NPP_BORDER_UNDEFINED); });
  status_case(fam, fn, "border_const", [&] { return call(src, off, 32, NPP_BORDER_CONSTANT); });
  status_case(fam, fn, "border_wrap", [&] { return call(src, off, 32, NPP_BORDER_WRAP); });
  status_case(fam, fn, "border_mirror_ok", [&] { return call(src, off, 32, NPP_BORDER_MIRROR); });
  status_case(fam, fn, "border_5_ok", [&] { return call(src, off, 32, (NppiBorderType)5); });
  status_case(fam, fn, "border_mirror_mask4x2", [&] {
    return nppiFilterBoxBorder_8u_C1R_Ctx(s, 32, src, off, d, 32, roi, NppiSize{4, 2}, a1,
                                          NPP_BORDER_MIRROR, c);
  });
  status_case(fam, fn, "border_5_mask3x5", [&] {
    return nppiFilterBoxBorder_8u_C1R_Ctx(s, 32, src, off, d, 32, roi, NppiSize{3, 5}, a1,
                                          (NppiBorderType)5, c);
  });
  status_case(fam, fn, "srcsize_zero", [&] { return call(NppiSize{0, 0}, off, 32, NPP_BORDER_REPLICATE); });
  status_case(fam, fn, "srcsize_neg", [&] { return call(NppiSize{-3, 7}, off, 32, NPP_BORDER_REPLICATE); });
  status_case(fam, fn, "offset_oob", [&] { return call(src, NppiPoint{20, 3}, 32, NPP_BORDER_REPLICATE); });
  status_case(fam, fn, "offset_neg", [&] { return call(src, NppiPoint{-1, 3}, 32, NPP_BORDER_REPLICATE); });
  status_case(fam, fn, "mask_zero", [&] { return nppiFilterBoxBorder_8u_C1R_Ctx(s, 32, src, off, d, 32, roi, NppiSize{0, 0}, a1, NPP_BORDER_REPLICATE, c); });
  status_case(fam, fn, "dstep_small", [&] { return call(src, off, 8, NPP_BORDER_REPLICATE); });
  status_case(fam, fn, "roi_zero", [&] { return nppiFilterBoxBorder_8u_C1R_Ctx(s, 32, src, off, d, 32, NppiSize{0, 7}, m3, a1, NPP_BORDER_REPLICATE, c); });
  status_case(fam, "nppiFilterBoxBorder_16u_C1R", "roi_zero_16u", [&] {
    return nppiFilterBoxBorder_16u_C1R_Ctx((Npp16u *)s, 32, src, off, (Npp16u *)d, 32,
                                           NppiSize{0, 7}, m3, a1, NPP_BORDER_REPLICATE, c);
  });
  status_case(fam, fn, "mask_gt_roi_ok", [&] {
    return nppiFilterBoxBorder_8u_C1R_Ctx(s, 32, src, off, d, 32, NppiSize{9, 7},
                                          NppiSize{11, 9}, NppiPoint{5, 4},
                                          NPP_BORDER_REPLICATE, c);
  });
  status_case(fam, fn, "anchor_gt", [&] { return nppiFilterBoxBorder_8u_C1R_Ctx(s, 32, src, off, d, 32, roi, m3, NppiPoint{3, 1}, NPP_BORDER_REPLICATE, c); });
}

TEST_F(NppiFilterMinDiff, Validation) {
  const std::string fam = "nppiFilterMin";
  const char *fn = "nppiFilterMin_8u_C1R";
  Npp8u *s = devbuf<Npp8u>(4096);
  Npp8u *d = devbuf<Npp8u>(4096);
  NppStreamContext c = current_ctx();
  const NppiSize roi{9, 7}, m3{3, 3};
  const NppiPoint a1{1, 1};
  status_case(fam, fn, "null_src", [&] { return nppiFilterMin_8u_C1R_Ctx(nullptr, 32, d, 32, roi, m3, a1, c); });
  status_case(fam, fn, "roi_zero", [&] { return nppiFilterMin_8u_C1R_Ctx(s, 32, d, 32, NppiSize{0, 7}, m3, a1, c); });
  status_case(fam, fn, "roi_neg", [&] { return nppiFilterMin_8u_C1R_Ctx(s, 32, d, 32, NppiSize{-3, 7}, m3, a1, c); });
  status_case(fam, fn, "mask_zero", [&] { return nppiFilterMin_8u_C1R_Ctx(s, 32, d, 32, roi, NppiSize{0, 0}, a1, c); });
  status_case(fam, fn, "mask_gt_roi", [&] { return nppiFilterMin_8u_C1R_Ctx(s, 32, d, 32, {4, 4}, {9, 5}, a1, c); });
  status_case(fam, fn, "dstep_small", [&] { return nppiFilterMin_8u_C1R_Ctx(s, 32, d, 8, roi, m3, a1, c); });
  status_case(fam, fn, "anchor_gt", [&] { return nppiFilterMin_8u_C1R_Ctx(s, 32, d, 32, roi, m3, NppiPoint{3, 1}, c); });
}

TEST_F(NppiFilterMaxDiff, Validation) {
  const std::string fam = "nppiFilterMax";
  const char *fn = "nppiFilterMax_8u_C1R";
  Npp8u *s = devbuf<Npp8u>(4096);
  Npp8u *d = devbuf<Npp8u>(4096);
  NppStreamContext c = current_ctx();
  const NppiSize roi{9, 7}, m3{3, 3};
  const NppiPoint a1{1, 1};
  status_case(fam, fn, "null_dst", [&] { return nppiFilterMax_8u_C1R_Ctx(s, 32, nullptr, 32, roi, m3, a1, c); });
  status_case(fam, fn, "roi_zero", [&] { return nppiFilterMax_8u_C1R_Ctx(s, 32, d, 32, NppiSize{9, 0}, m3, a1, c); });
  status_case(fam, fn, "mask_zero_w", [&] { return nppiFilterMax_8u_C1R_Ctx(s, 32, d, 32, roi, NppiSize{0, 3}, a1, c); });
  status_case(fam, fn, "mask_gt_roi", [&] { return nppiFilterMax_8u_C1R_Ctx(s, 32, d, 32, {4, 4}, {9, 5}, a1, c); });
  status_case(fam, fn, "dstep_small", [&] { return nppiFilterMax_8u_C1R_Ctx(s, 32, d, 8, roi, m3, a1, c); });
  status_case(fam, fn, "anchor_gt", [&] { return nppiFilterMax_8u_C1R_Ctx(s, 32, d, 32, roi, m3, NppiPoint{3, 1}, c); });
}

TEST_F(NppiFilterMinBorderDiff, Validation) {
  const std::string fam = "nppiFilterMinBorder";
  const char *fn = "nppiFilterMinBorder_8u_C1R";
  Npp8u *s = devbuf<Npp8u>(4096);
  Npp8u *d = devbuf<Npp8u>(4096);
  NppStreamContext c = current_ctx();
  const NppiSize src{13, 11}, roi{9, 7}, m3{3, 3};
  const NppiPoint off{2, 3}, a1{1, 1};
  auto call = [&](NppiSize sr, NppiPoint o, int ds, NppiBorderType b) {
    return nppiFilterMinBorder_8u_C1R_Ctx(s, 32, sr, o, d, ds, roi, m3, a1, b, c);
  };
  status_case(fam, fn, "border_undef", [&] { return call(src, off, 32, NPP_BORDER_UNDEFINED); });
  status_case(fam, fn, "border_const", [&] { return call(src, off, 32, NPP_BORDER_CONSTANT); });
  status_case(fam, fn, "border_wrap", [&] { return call(src, off, 32, NPP_BORDER_WRAP); });
  status_case(fam, fn, "border_mirror", [&] { return call(src, off, 32, NPP_BORDER_MIRROR); });
  status_case(fam, fn, "border_5", [&] { return call(src, off, 32, (NppiBorderType)5); });
  status_case(fam, fn, "roi_zero", [&] { return nppiFilterMinBorder_8u_C1R_Ctx(s, 32, src, off, d, 32, NppiSize{0, 7}, m3, a1, NPP_BORDER_REPLICATE, c); });
  status_case(fam, fn, "srcsize_zero", [&] { return call(NppiSize{0, 0}, off, 32, NPP_BORDER_REPLICATE); });
  status_case(fam, fn, "offset_oob", [&] { return call(src, NppiPoint{20, 3}, 32, NPP_BORDER_REPLICATE); });
  status_case(fam, fn, "mask_zero", [&] { return nppiFilterMinBorder_8u_C1R_Ctx(s, 32, src, off, d, 32, roi, NppiSize{0, 0}, a1, NPP_BORDER_REPLICATE, c); });
  status_case(fam, fn, "dstep_small", [&] { return call(src, off, 8, NPP_BORDER_REPLICATE); });
  status_case(fam, fn, "anchor_gt", [&] { return nppiFilterMinBorder_8u_C1R_Ctx(s, 32, src, off, d, 32, roi, m3, NppiPoint{3, 1}, NPP_BORDER_REPLICATE, c); });
}

TEST_F(NppiFilterMaxBorderDiff, Validation) {
  const std::string fam = "nppiFilterMaxBorder";
  const char *fn = "nppiFilterMaxBorder_8u_C1R";
  Npp8u *s = devbuf<Npp8u>(4096);
  Npp8u *d = devbuf<Npp8u>(4096);
  NppStreamContext c = current_ctx();
  const NppiSize src{13, 11}, roi{9, 7}, m3{3, 3};
  const NppiPoint off{2, 3}, a1{1, 1};
  auto call = [&](NppiPoint o, NppiBorderType b) {
    return nppiFilterMaxBorder_8u_C1R_Ctx(s, 32, src, o, d, 32, roi, m3, a1, b, c);
  };
  status_case(fam, fn, "border_const", [&] { return call(off, NPP_BORDER_CONSTANT); });
  status_case(fam, fn, "border_mirror", [&] { return call(off, NPP_BORDER_MIRROR); });
  status_case(fam, fn, "offset_neg", [&] { return call(NppiPoint{-1, 3}, NPP_BORDER_REPLICATE); });
  status_case(fam, fn, "mask_zero", [&] { return nppiFilterMaxBorder_8u_C1R_Ctx(s, 32, src, off, d, 32, roi, NppiSize{0, 0}, a1, NPP_BORDER_REPLICATE, c); });
  status_case(fam, fn, "anchor_gt", [&] { return nppiFilterMaxBorder_8u_C1R_Ctx(s, 32, src, off, d, 32, roi, m3, NppiPoint{3, 1}, NPP_BORDER_REPLICATE, c); });
}

TEST_F(NppiFilterMedianDiff, Validation) {
  const std::string fam = "nppiFilterMedian";
  const char *fn = "nppiFilterMedian_8u_C1R";
  Npp8u *s = devbuf<Npp8u>(4096);
  Npp8u *d = devbuf<Npp8u>(4096);
  NppStreamContext c = current_ctx();
  const NppiSize roi{9, 7}, m3{3, 3};
  const NppiPoint a1{1, 1};
  status_case(fam, fn, "null_src", [&] { return nppiFilterMedian_8u_C1R_Ctx(nullptr, 32, d, 32, roi, m3, a1, nullptr, c); });
  status_case(fam, fn, "roi_zero", [&] { return nppiFilterMedian_8u_C1R_Ctx(s, 32, d, 32, NppiSize{0, 7}, m3, a1, nullptr, c); });
  status_case(fam, fn, "roi_neg", [&] { return nppiFilterMedian_8u_C1R_Ctx(s, 32, d, 32, NppiSize{-3, 7}, m3, a1, nullptr, c); });
  status_case(fam, fn, "mask_zero", [&] { return nppiFilterMedian_8u_C1R_Ctx(s, 32, d, 32, roi, NppiSize{0, 0}, a1, nullptr, c); });
  status_case(fam, fn, "mask_gt_roi", [&] { return nppiFilterMedian_8u_C1R_Ctx(s, 32, d, 32, {4, 4}, {9, 5}, a1, nullptr, c); });
  status_case(fam, fn, "dstep_small", [&] { return nppiFilterMedian_8u_C1R_Ctx(s, 32, d, 8, roi, m3, a1, nullptr, c); });
  status_case(fam, fn, "anchor_gt", [&] { return nppiFilterMedian_8u_C1R_Ctx(s, 32, d, 32, roi, m3, NppiPoint{3, 1}, nullptr, c); });
  const char *fn16 = "nppiFilterMedian_16u_C1R";
  status_case(fam, fn16, "dstep_neg_16u_ok", [&] {
    return nppiFilterMedian_16u_C1R_Ctx((Npp16u *)s, 32, (Npp16u *)d + 8, -2, NppiSize{1, 1}, m3, a1, nullptr, c);
  });
}

TEST_F(NppiFilterMedianBorderDiff, Validation) {
  const std::string fam = "nppiFilterMedianBorder";
  const char *fn = "nppiFilterMedianBorder_8u_C1R";
  Npp8u *s = devbuf<Npp8u>(4096);
  Npp8u *d = devbuf<Npp8u>(4096);
  NppStreamContext c = current_ctx();
  const NppiSize src{13, 11}, roi{9, 7}, m3{3, 3};
  const NppiPoint off{2, 3}, a1{1, 1};
  auto call = [&](NppiSize sr, NppiPoint o, int ds, NppiBorderType b) {
    return nppiFilterMedianBorder_8u_C1R_Ctx(s, 32, sr, o, d, ds, roi, m3, a1, nullptr, b, c);
  };
  status_case(fam, fn, "border_undef", [&] { return call(src, off, 32, NPP_BORDER_UNDEFINED); });
  status_case(fam, fn, "border_5", [&] { return call(src, off, 32, (NppiBorderType)5); });
  status_case(fam, fn, "border_mirror_ok", [&] { return call(src, off, 32, NPP_BORDER_MIRROR); });
  status_case(fam, fn, "roi_zero_ok", [&] { return nppiFilterMedianBorder_8u_C1R_Ctx(s, 32, src, off, d, 32, NppiSize{0, 7}, m3, a1, nullptr, NPP_BORDER_CONSTANT, c); });
  status_case(fam, fn, "mask_zero_ok", [&] { return nppiFilterMedianBorder_8u_C1R_Ctx(s, 32, src, off, d, 32, roi, NppiSize{0, 0}, a1, nullptr, NPP_BORDER_CONSTANT, c); });
  status_case(fam, fn, "srcsize_zero", [&] { return call(NppiSize{0, 0}, off, 32, NPP_BORDER_REPLICATE); });
  status_case(fam, fn, "offset_oob", [&] { return call(src, NppiPoint{20, 3}, 32, NPP_BORDER_REPLICATE); });
  status_case(fam, fn, "dstep_small", [&] { return call(src, off, 8, NPP_BORDER_REPLICATE); });
  const char *fn16 = "nppiFilterMedianBorder_16u_C1R";
  status_case(fam, fn16, "dstep_odd_16u", [&] {
    return nppiFilterMedianBorder_16u_C1R_Ctx((Npp16u *)s, 32, src, off, (Npp16u *)d, 19, roi, m3, a1, nullptr, NPP_BORDER_REPLICATE, c);
  });
  status_case(fam, fn, "anchor_gt", [&] { return nppiFilterMedianBorder_8u_C1R_Ctx(s, 32, src, off, d, 32, roi, m3, NppiPoint{3, 1}, nullptr, NPP_BORDER_REPLICATE, c); });
}

// ---------------------------------------------------------------------------
// median GetBufferSize families
// ---------------------------------------------------------------------------
namespace {

const NppiSize kBufMasks[] = {
    {1, 1},   {3, 3},   {5, 5},   {7, 7},   {8, 8},   {9, 9},   {10, 10},
    {11, 11}, {12, 12}, {13, 13}, {14, 14}, {15, 15}, {16, 16}, {17, 17},
    {20, 20}, {24, 24}, {25, 25}, {32, 32}, {33, 33}, {4, 16},  {9, 16},
    {15, 16}, {2, 33},  {32, 2},  {1, 40},  {40, 1},  {40, 40},
};

template <typename F>
void bufsize_case(const std::string &family, const char *fname, const std::string &key, F &&f) {
  Npp32u sz = 0xDEADBEEFu;
  clear_cuda_error();
  NppStatus st = f(&sz);
  Case c(family + "/" + key, fname);
  c.status(st);
  c.host_output<Npp32u>(&sz, 1, ValueKind::U32);
  c.finish();
}

}  // namespace

class NppiFilterMedianGetBufferSizeDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

#define BUFSIZE_PLAIN(TAG)                                                                     \
  {                                                                                            \
    const char *fn = "nppiFilterMedianGetBufferSize_" #TAG;                                    \
    for (const auto &m : kBufMasks) {                                                          \
      std::string key = std::string(#TAG) + "/mask/" + std::to_string(m.width) + "x" +        \
                        std::to_string(m.height);                                              \
      bufsize_case("nppiFilterMedianGetBufferSize", fn, key, [&](Npp32u *p) {                  \
        return nppiFilterMedianGetBufferSize_##TAG##_Ctx(NppiSize{7, 7}, m, p, current_ctx()); \
      });                                                                                     \
    }                                                                                         \
  }

TEST_F(NppiFilterMedianGetBufferSizeDiff, Sweep) {
  BUFSIZE_PLAIN(8u_C1R);
  BUFSIZE_PLAIN(8u_C3R);
  BUFSIZE_PLAIN(8u_C4R);
  BUFSIZE_PLAIN(8u_AC4R);
  BUFSIZE_PLAIN(16u_C1R);
  BUFSIZE_PLAIN(16u_C3R);
  BUFSIZE_PLAIN(16u_C4R);
  BUFSIZE_PLAIN(16u_AC4R);
  BUFSIZE_PLAIN(16s_C1R);
  BUFSIZE_PLAIN(16s_C3R);
  BUFSIZE_PLAIN(16s_C4R);
  BUFSIZE_PLAIN(16s_AC4R);
  BUFSIZE_PLAIN(32f_C1R);
  BUFSIZE_PLAIN(32f_C3R);
  BUFSIZE_PLAIN(32f_C4R);
  BUFSIZE_PLAIN(32f_AC4R);
}

TEST_F(NppiFilterMedianGetBufferSizeDiff, Validation) {
  const std::string fam = "nppiFilterMedianGetBufferSize";
  const char *fn = "nppiFilterMedianGetBufferSize_8u_C1R";
  NppStreamContext c = current_ctx();
  bufsize_case(fam, fn, "roi_neg", [&](Npp32u *p) {
    return nppiFilterMedianGetBufferSize_8u_C1R_Ctx(NppiSize{-2, 5}, NppiSize{3, 3}, p, c);
  });
  bufsize_case(fam, fn, "roi_zero", [&](Npp32u *p) {
    return nppiFilterMedianGetBufferSize_8u_C1R_Ctx(NppiSize{0, 5}, NppiSize{3, 3}, p, c);
  });
  bufsize_case(fam, fn, "mask_zero", [&](Npp32u *p) {
    return nppiFilterMedianGetBufferSize_8u_C1R_Ctx(NppiSize{7, 7}, NppiSize{0, 0}, p, c);
  });
  bufsize_case(fam, fn, "mask_neg", [&](Npp32u *p) {
    return nppiFilterMedianGetBufferSize_8u_C1R_Ctx(NppiSize{7, 7}, NppiSize{-3, 3}, p, c);
  });
  bufsize_case(fam, fn, "mask_neg_h", [&](Npp32u *p) {
    return nppiFilterMedianGetBufferSize_8u_C1R_Ctx(NppiSize{7, 7}, NppiSize{3, -3}, p, c);
  });
  bufsize_case(fam, fn, "mask_gt_roi", [&](Npp32u *p) {
    return nppiFilterMedianGetBufferSize_8u_C1R_Ctx(NppiSize{4, 4}, NppiSize{16, 16}, p, c);
  });
  bufsize_case(fam, "nppiFilterMedianGetBufferSize_16u_C1R", "mask_neg_16u", [&](Npp32u *p) {
    return nppiFilterMedianGetBufferSize_16u_C1R_Ctx(NppiSize{7, 7}, NppiSize{-3, 3}, p, c);
  });
}

class NppiFilterMedianBorderGetBufferSizeDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

#define BUFSIZE_BORDER(TAG)                                                                    \
  {                                                                                            \
    const char *fn = "nppiFilterMedianBorderGetBufferSize_" #TAG;                              \
    for (const auto &m : kBufMasks) {                                                          \
      std::string key = std::string(#TAG) + "/mask/" + std::to_string(m.width) + "x" +        \
                        std::to_string(m.height);                                              \
      bufsize_case("nppiFilterMedianBorderGetBufferSize", fn, key, [&](Npp32u *p) {            \
        return nppiFilterMedianBorderGetBufferSize_##TAG##_Ctx(NppiSize{7, 7}, m, p,           \
                                                               NPP_BORDER_REPLICATE,           \
                                                               current_ctx());                 \
      });                                                                                     \
    }                                                                                         \
  }

TEST_F(NppiFilterMedianBorderGetBufferSizeDiff, Sweep) {
  BUFSIZE_BORDER(8u_C1R);
  BUFSIZE_BORDER(8u_C3R);
  BUFSIZE_BORDER(8u_C4R);
  BUFSIZE_BORDER(8u_AC4R);
  BUFSIZE_BORDER(16u_C1R);
  BUFSIZE_BORDER(16u_C3R);
  BUFSIZE_BORDER(16u_C4R);
  BUFSIZE_BORDER(16u_AC4R);
  BUFSIZE_BORDER(16s_C1R);
  BUFSIZE_BORDER(16s_C3R);
  BUFSIZE_BORDER(16s_C4R);
  BUFSIZE_BORDER(16s_AC4R);
  BUFSIZE_BORDER(32f_C1R);
  BUFSIZE_BORDER(32f_C3R);
  BUFSIZE_BORDER(32f_C4R);
  BUFSIZE_BORDER(32f_AC4R);
}

TEST_F(NppiFilterMedianBorderGetBufferSizeDiff, Validation) {
  const std::string fam = "nppiFilterMedianBorderGetBufferSize";
  const char *fn = "nppiFilterMedianBorderGetBufferSize_8u_C1R";
  NppStreamContext c = current_ctx();
  const int types[] = {0, 1, 2, 3, 4, 5, 7};
  for (int t : types) {
    bufsize_case(fam, fn, std::string("bordertype/") + std::to_string(t), [&](Npp32u *p) {
      return nppiFilterMedianBorderGetBufferSize_8u_C1R_Ctx(NppiSize{7, 7}, NppiSize{16, 16}, p,
                                                            (NppiBorderType)t, c);
    });
  }
  bufsize_case(fam, fn, "roi_neg", [&](Npp32u *p) {
    return nppiFilterMedianBorderGetBufferSize_8u_C1R_Ctx(NppiSize{-2, 5}, NppiSize{3, 3}, p,
                                                          NPP_BORDER_REPLICATE, c);
  });
  bufsize_case(fam, fn, "roi_zero", [&](Npp32u *p) {
    return nppiFilterMedianBorderGetBufferSize_8u_C1R_Ctx(NppiSize{0, 5}, NppiSize{3, 3}, p,
                                                          NPP_BORDER_REPLICATE, c);
  });
  bufsize_case(fam, fn, "mask_zero", [&](Npp32u *p) {
    return nppiFilterMedianBorderGetBufferSize_8u_C1R_Ctx(NppiSize{7, 7}, NppiSize{0, 0}, p,
                                                          NPP_BORDER_REPLICATE, c);
  });
  bufsize_case(fam, fn, "mask_neg", [&](Npp32u *p) {
    return nppiFilterMedianBorderGetBufferSize_8u_C1R_Ctx(NppiSize{7, 7}, NppiSize{-3, 3}, p,
                                                          NPP_BORDER_REPLICATE, c);
  });
  bufsize_case(fam, "nppiFilterMedianBorderGetBufferSize_16u_C1R", "mask_neg_16u", [&](Npp32u *p) {
    return nppiFilterMedianBorderGetBufferSize_16u_C1R_Ctx(NppiSize{7, 7}, NppiSize{-3, 3}, p,
                                                           NPP_BORDER_REPLICATE, c);
  });
}
