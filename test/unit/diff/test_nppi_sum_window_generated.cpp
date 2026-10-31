// Differential tests for package f12: nppiSumWindowRow / nppiSumWindowColumn
// (and their Border variants): 8u/16u/16s sources, 32f destination, C1/C3/C4.
// The already implemented 8u32f_C1R functions are probed as well.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiSumWindowDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppiSumWindowDiff.*'
//
// The tests pin down the open semantic questions of this family: the window
// direction (filter convention [x+ax-mask+1, x+ax] vs the legacy
// [x-ax, x-ax+mask-1]), whether out-of-ROI taps are clipped or read raw,
// anchor/mask validation, empty ROIs and stride handling.

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

template <typename T> std::vector<T> make_src(size_t n, uint64_t seed) {
  Rng rng(seed);
  std::vector<T> v(n);
  for (auto &x : v) {
    if constexpr (std::is_integral_v<T>) {
      x = rng.integer<T>(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
    } else {
      x = static_cast<T>(rng.integer<int>(-64, 64)) * static_cast<T>(0.125);
    }
  }
  return v;
}

template <typename F>
void status_case(const std::string &family, const char *fname, const std::string &key, F &&f) {
  clear_cuda_error();
  NppStatus st = f();
  Case c(family + "/" + key, fname);
  c.status(st);
  c.finish();
}

constexpr int kMargin = 8;

// Non-border data case: the ROI sits inside a (W+2*margin)x(H+2*margin)
// buffer so both clipped and raw window reads stay inside the allocation.
template <typename S, typename F>
void sw_data(const std::string &family, const char *fname, const std::string &key, int W, int H,
             int ch, int mask, int anchor, int margin, uint64_t seed, int dstep, int sstep, F &&f) {
  const int sPx = ch * (int)sizeof(S);
  const int dPx = ch * (int)sizeof(Npp32f);
  const int sStride = sstep ? align_step(sstep, sPx) : align_step((W + 2 * margin) * sPx + 3, sPx);
  const int dStride = dstep < 0 ? -dstep
      : (dstep ? align_step(dstep, dPx) : align_step(W * dPx + 3, dPx));
  const int allocStride = align_step((W + 2 * margin) * sPx + 3, sPx);
  const size_t se = ((size_t)(H + 2 * margin) * allocStride + (size_t)(W + 2 * margin) * sPx) / sizeof(S);
  const size_t de = (size_t)H * dStride / sizeof(Npp32f);
  const size_t dz = de ? de : 1;  // keep empty-ROI destinations non-null
  std::vector<S> hs = make_src<S>(se, seed + 11);
  std::vector<Npp32f> hd = make_uniform<Npp32f>(dz, seed + 29);
  DeviceMemory<S> &s = arena<S>(se);
  DeviceMemory<Npp32f> &d = arena<Npp32f>(dz);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  clear_cuda_error();
  const S *ps = s.get() + (size_t)margin * (sStride / sizeof(S)) + (size_t)margin * ch;
  Npp32f *pd = (dstep < 0 && H > 0) ? d.get() + (size_t)(H - 1) * (dStride / sizeof(Npp32f)) : d.get();
  NppStatus st = f(ps, sStride, pd, dstep < 0 ? -dStride : dStride, NppiSize{W, H}, mask, anchor,
                   current_ctx());
  Case c(family + "/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(S));
  c.status(st);
  c.device_output<Npp32f>(d.get(), dz, ValueKind::F32);
  c.finish();
}

// Border data case: reads are clamped into the (sw x sh) source image.
template <typename S, typename F>
void swb_data(const std::string &family, const char *fname, const std::string &key, int W, int H,
              int ch, int sw, int sh, int offx, int offy, int mask, int anchor, uint64_t seed,
              int dstep, F &&f) {
  const int sPx = ch * (int)sizeof(S);
  const int dPx = ch * (int)sizeof(Npp32f);
  const int sStep = align_step(sw * sPx + 3, sPx);
  const int dStride = dstep < 0 ? -dstep
      : (dstep ? align_step(dstep, dPx) : align_step(W * dPx + 3, dPx));
  const size_t se = (size_t)sh * sStep / sizeof(S);
  const size_t de = (size_t)H * dStride / sizeof(Npp32f);
  const size_t dz = de ? de : 1;  // keep empty-ROI destinations non-null
  std::vector<S> hs = make_src<S>(se, seed + 11);
  std::vector<Npp32f> hd = make_uniform<Npp32f>(dz, seed + 29);
  DeviceMemory<S> &s = arena<S>(se);
  DeviceMemory<Npp32f> &d = arena<Npp32f>(dz);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  clear_cuda_error();
  const S *ps = s.get() + (size_t)offy * (sStep / sizeof(S)) + (size_t)offx * ch;
  Npp32f *pd = (dstep < 0 && H > 0) ? d.get() + (size_t)(H - 1) * (dStride / sizeof(Npp32f)) : d.get();
  NppStatus st = f(ps, sStep, NppiSize{sw, sh}, NppiPoint{offx, offy}, pd,
                   dstep < 0 ? -dStride : dStride, NppiSize{W, H}, mask, anchor,
                   NPP_BORDER_REPLICATE, current_ctx());
  Case c(family + "/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(S));
  c.status(st);
  c.device_output<Npp32f>(d.get(), dz, ValueKind::F32);
  c.finish();
}

}  // namespace

class NppiSumWindowDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
  int case_id_ = 0;
};

// ---------------------------------------------------------------------------
// Sweeps: every dtype x layout of the four families, mask 5, anchor 2.
// ---------------------------------------------------------------------------
#define SW_ROW_RUN(S, TAG, CH)                                                                 \
  sw_data<S>("nppiSumWindowRow", "nppiSumWindowRow_" #TAG, "sweep/" #TAG, 17, 13, CH, 5, 2,    \
             kMargin, 0xD100 + case_id_++, 0, 0,                                               \
             [](const S *s, int ss, Npp32f *d, int ds, NppiSize o, int m, int a,               \
                NppStreamContext c) {                                                          \
               return nppiSumWindowRow_##TAG##_Ctx(s, ss, d, ds, o, m, a, c);                  \
             })
#define SW_COL_RUN(S, TAG, CH)                                                                 \
  sw_data<S>("nppiSumWindowColumn", "nppiSumWindowColumn_" #TAG, "sweep/" #TAG, 17, 13, CH, 5, 2, \
             kMargin, 0xD200 + case_id_++, 0, 0,                                               \
             [](const S *s, int ss, Npp32f *d, int ds, NppiSize o, int m, int a,               \
                NppStreamContext c) {                                                          \
               return nppiSumWindowColumn_##TAG##_Ctx(s, ss, d, ds, o, m, a, c);                \
             })
#define SWB_ROW_RUN(S, TAG, CH)                                                                \
  swb_data<S>("nppiSumWindowRowBorder", "nppiSumWindowRowBorder_" #TAG, "sweep/" #TAG, 7, 7, CH, \
              9, 9, 2, 3, 5, 2, 0xD300 + case_id_++, 0,                                         \
              [](const S *s, int ss, NppiSize sz, NppiPoint so, Npp32f *d, int ds, NppiSize o, \
                 int m, int a, NppiBorderType bt, NppStreamContext c) {                        \
                return nppiSumWindowRowBorder_##TAG##_Ctx(s, ss, sz, so, d, ds, o, m, a, bt, c); \
              })
#define SWB_COL_RUN(S, TAG, CH)                                                                \
  swb_data<S>("nppiSumWindowColumnBorder", "nppiSumWindowColumnBorder_" #TAG, "sweep/" #TAG, 7, \
              7, CH, 9, 9, 2, 3, 5, 2, 0xD400 + case_id_++, 0,                                 \
              [](const S *s, int ss, NppiSize sz, NppiPoint so, Npp32f *d, int ds, NppiSize o, \
                 int m, int a, NppiBorderType bt, NppStreamContext c) {                        \
                return nppiSumWindowColumnBorder_##TAG##_Ctx(s, ss, sz, so, d, ds, o, m, a, bt, \
                                                             c);                              \
              })

#define SW_SWEEP(FN)                                                                           \
  FN(Npp8u, 8u32f_C1R, 1);                                                                     \
  FN(Npp8u, 8u32f_C3R, 3);                                                                     \
  FN(Npp8u, 8u32f_C4R, 4);                                                                     \
  FN(Npp16u, 16u32f_C1R, 1);                                                                   \
  FN(Npp16u, 16u32f_C3R, 3);                                                                   \
  FN(Npp16u, 16u32f_C4R, 4);                                                                   \
  FN(Npp16s, 16s32f_C1R, 1);                                                                   \
  FN(Npp16s, 16s32f_C3R, 3);                                                                   \
  FN(Npp16s, 16s32f_C4R, 4)

TEST_F(NppiSumWindowDiff, RowSweep) { SW_SWEEP(SW_ROW_RUN); }
TEST_F(NppiSumWindowDiff, ColumnSweep) { SW_SWEEP(SW_COL_RUN); }
TEST_F(NppiSumWindowDiff, RowBorderSweep) { SW_SWEEP(SWB_ROW_RUN); }
TEST_F(NppiSumWindowDiff, ColumnBorderSweep) { SW_SWEEP(SWB_COL_RUN); }

// ---------------------------------------------------------------------------
// Geometry for the non-border families: window direction and clipping.
// ---------------------------------------------------------------------------
TEST_F(NppiSumWindowDiff, Geometry) {
  auto row = [&](const char *tag, int W, int H, int mask, int anchor, int margin, uint64_t seed,
                 int dstep = 0, int sstep = 0) {
    sw_data<Npp16u>("nppiSumWindowRow", "nppiSumWindowRow_16u32f_C1R",
                    std::string("geom/row/") + tag, W, H, 1, mask, anchor, margin, seed, dstep,
                    sstep,
                    [](const Npp16u *s, int ss, Npp32f *d, int ds, NppiSize o, int m, int a,
                       NppStreamContext c) {
                      return nppiSumWindowRow_16u32f_C1R_Ctx(s, ss, d, ds, o, m, a, c);
                    });
  };
  auto col = [&](const char *tag, int W, int H, int mask, int anchor, int margin, uint64_t seed,
                 int dstep = 0, int sstep = 0) {
    sw_data<Npp16s>("nppiSumWindowColumn", "nppiSumWindowColumn_16s32f_C1R",
                    std::string("geom/col/") + tag, W, H, 1, mask, anchor, margin, seed, dstep,
                    sstep,
                    [](const Npp16s *s, int ss, Npp32f *d, int ds, NppiSize o, int m, int a,
                       NppStreamContext c) {
                      return nppiSumWindowColumn_16s32f_C1R_Ctx(s, ss, d, ds, o, m, a, c);
                    });
  };
  const int m = kMargin;
  row("a0", 17, 13, 5, 0, m, 0xD501);        // window direction probe
  row("amax", 17, 13, 5, 5, m, 0xD502);      // anchor == mask
  row("aneg", 17, 13, 5, -3, m, 0xD503);
  row("aext", 5, 5, 3, -4, m, 0xD504);       // anchor == 1-W
  row("mask1", 17, 13, 1, 0, m, 0xD505);
  row("mask2", 17, 13, 2, 0, m, 0xD506);     // even mask, direction probe
  row("mask9", 17, 13, 9, 4, m, 0xD507);
  row("mask23", 17, 13, 23, 11, 12, 0xD508); // mask > ROI width (clipping)
  row("mask0", 17, 13, 0, 0, m, 0xD509);     // zero mask
  row("mask_neg", 17, 13, -3, 0, m, 0xD50A); // negative mask, anchor 0
  row("mask_neg_ok", 17, 13, -3, -3, m, 0xD50B);
  row("small", 1, 1, 3, 1, m, 0xD50C);
  row("wide", 33, 1, 5, 2, m, 0xD50D);
  row("tall", 1, 33, 5, 2, m, 0xD50E);
  row("emptyw", 0, 13, 3, 1, m, 0xD50F);
  row("emptyh", 17, 0, 5, 2, m, 0xD510);
  row("negstep", 17, 13, 5, 2, m, 0xD511, -96);
  row("bigstep", 17, 13, 5, 2, m, 0xD512, 98);
  row("sstep_small", 8, 8, 3, 1, m, 0xD513, 0, 2);
  col("a0", 17, 13, 5, 0, m, 0xD521);
  col("amax", 17, 13, 5, 5, m, 0xD522);
  col("aext", 5, 5, 3, -4, m, 0xD523);
  col("mask2", 17, 13, 2, 0, m, 0xD524);
  col("mask9", 17, 13, 9, 4, m, 0xD525);
  col("mask23", 13, 17, 23, 11, 12, 0xD526);
  col("mask0", 17, 13, 0, 0, m, 0xD527);
  col("emptyh", 17, 0, 3, 1, m, 0xD528);
  col("negstep", 17, 13, 5, 2, m, 0xD529, -96);
  // Legacy ("real") 8u32f_C1R probes.
  auto row8 = [&](const char *tag, int W, int H, int mask, int anchor, uint64_t seed) {
    sw_data<Npp8u>("nppiSumWindowRow", "nppiSumWindowRow_8u32f_C1R",
                   std::string("geom/8u_row/") + tag, W, H, 1, mask, anchor, kMargin, seed, 0, 0,
                   [](const Npp8u *s, int ss, Npp32f *d, int ds, NppiSize o, int m, int a,
                      NppStreamContext c) {
                     return nppiSumWindowRow_8u32f_C1R_Ctx(s, ss, d, ds, o, m, a, c);
                   });
  };
  auto col8 = [&](const char *tag, int W, int H, int mask, int anchor, uint64_t seed) {
    sw_data<Npp8u>("nppiSumWindowColumn", "nppiSumWindowColumn_8u32f_C1R",
                   std::string("geom/8u_col/") + tag, W, H, 1, mask, anchor, kMargin, seed, 0, 0,
                   [](const Npp8u *s, int ss, Npp32f *d, int ds, NppiSize o, int m, int a,
                      NppStreamContext c) {
                     return nppiSumWindowColumn_8u32f_C1R_Ctx(s, ss, d, ds, o, m, a, c);
                   });
  };
  row8("a0", 17, 13, 5, 0, 0xD531);
  row8("mask0", 17, 13, 0, 0, 0xD532);
  row8("anchor_eq_mask", 17, 13, 3, 3, 0xD533);
  // Negative-anchor probes: the window is [p-a, p-a+mask-1] (anchor sign
  // inverted w.r.t. the filters).  With a < 0 the window extends to the right
  // of the pixel; the last columns of the ROI deviate from that rule in a way
  // that is characterised by these probes (width/mask/anchor/margin sweep).
  auto prow = [&](const char *tag, int W, int H, int mask, int anchor, uint64_t seed,
                  int margin = 16) {
    sw_data<Npp16u>("nppiSumWindowRow", "nppiSumWindowRow_16u32f_C1R",
                    std::string("probe/row/") + tag, W, H, 1, mask, anchor, margin, seed, 0, 0,
                    [](const Npp16u *s, int ss, Npp32f *d, int ds, NppiSize o, int m, int a,
                       NppStreamContext c) {
                      return nppiSumWindowRow_16u32f_C1R_Ctx(s, ss, d, ds, o, m, a, c);
                    });
  };
  prow("a_neg1", 17, 13, 5, -1, 0xD701);
  prow("a_neg2", 17, 13, 5, -2, 0xD702);
  prow("a_neg4", 17, 13, 5, -4, 0xD703);
  prow("a_neg6", 17, 13, 5, -6, 0xD704);
  prow("m3_a_neg3", 17, 13, 3, -3, 0xD705);
  prow("m9_a_neg3", 17, 13, 9, -3, 0xD706);
  prow("m7_a_neg4", 17, 13, 7, -4, 0xD707);
  prow("m2_a_neg3", 17, 13, 2, -3, 0xD708);
  prow("w9_a_neg3", 9, 13, 5, -3, 0xD709);
  prow("w15_a_neg3", 15, 13, 5, -3, 0xD70A);
  prow("w16_a_neg3", 16, 13, 5, -3, 0xD70B);
  prow("w18_a_neg3", 18, 13, 5, -3, 0xD70C);
  prow("w24_a_neg3", 24, 13, 5, -3, 0xD70D);
  prow("w31_a_neg3", 31, 13, 5, -3, 0xD70E);
  prow("w32_a_neg3", 32, 13, 5, -3, 0xD70F);
  prow("w33_a_neg3", 33, 13, 5, -3, 0xD710);
  prow("w40_a_neg3", 40, 13, 5, -3, 0xD711);
  prow("a_neg3_m8", 17, 13, 5, -3, 0xD712, 8);
  prow("a_neg3_m24", 17, 13, 5, -3, 0xD713, 24);
  prow("a_neg3_h1", 17, 1, 5, -3, 0xD714);
  row8("anchor_neg", 17, 13, 3, -1, 0xD534);
  row8("emptyw", 0, 13, 3, 1, 0xD535);
  row8("negstep", 17, 13, 5, 2, 0xD536);
  col8("a0", 17, 13, 5, 0, 0xD537);
  col8("mask0", 17, 13, 0, 0, 0xD538);
  col8("emptyh", 17, 0, 3, 1, 0xD539);
  col8("negstep", 17, 13, 5, 2, 0xD53A);
}

// ---------------------------------------------------------------------------
// Geometry for the border families.
// ---------------------------------------------------------------------------
TEST_F(NppiSumWindowDiff, BorderGeometry) {
  auto rowb = [&](const char *tag, int W, int H, int sw, int sh, int ox, int oy, int mask, int anchor,
                  uint64_t seed, int dstep = 0) {
    swb_data<Npp16u>("nppiSumWindowRowBorder", "nppiSumWindowRowBorder_16u32f_C1R",
                     std::string("geom/rowb/") + tag, W, H, 1, sw, sh, ox, oy, mask, anchor, seed,
                     dstep,
                     [](const Npp16u *s, int ss, NppiSize sz, NppiPoint so, Npp32f *d, int ds,
                        NppiSize o, int m, int a, NppiBorderType bt, NppStreamContext c) {
                       return nppiSumWindowRowBorder_16u32f_C1R_Ctx(s, ss, sz, so, d, ds, o, m, a,
                                                                    bt, c);
                     });
  };
  auto colb = [&](const char *tag, int W, int H, int sw, int sh, int ox, int oy, int mask, int anchor,
                  uint64_t seed) {
    swb_data<Npp16s>("nppiSumWindowColumnBorder", "nppiSumWindowColumnBorder_16s32f_C1R",
                     std::string("geom/colb/") + tag, W, H, 1, sw, sh, ox, oy, mask, anchor, seed, 0,
                     [](const Npp16s *s, int ss, NppiSize sz, NppiPoint so, Npp32f *d, int ds,
                        NppiSize o, int m, int a, NppiBorderType bt, NppStreamContext c) {
                       return nppiSumWindowColumnBorder_16s32f_C1R_Ctx(s, ss, sz, so, d, ds, o, m,
                                                                       a, bt, c);
                     });
  };
  rowb("offset", 5, 5, 12, 10, 4, 6, 3, 1, 0xD601);
  rowb("beyond", 9, 9, 4, 4, 1, 1, 3, 1, 0xD602);   // ROI larger than the source
  rowb("last_off", 3, 3, 8, 8, 7, 7, 3, 1, 0xD603); // offset in the last pixel
  rowb("offset0", 5, 5, 8, 8, 0, 0, 5, 2, 0xD604);
  rowb("anchor9", 4, 4, 6, 6, 1, 1, 3, 9, 0xD605);  // anchors are not validated
  rowb("anchor_neg", 4, 4, 6, 6, 1, 1, 3, -2, 0xD606);
  rowb("mask0", 4, 4, 6, 6, 1, 1, 0, 0, 0xD607);
  rowb("mask_neg", 4, 4, 6, 6, 1, 1, -3, 0, 0xD608);
  rowb("mask2", 6, 6, 8, 8, 1, 1, 2, 0, 0xD609);
  rowb("mask9", 4, 4, 6, 6, 1, 1, 9, 4, 0xD60A);
  rowb("src1x1", 4, 4, 1, 1, 0, 0, 3, 1, 0xD60B);
  rowb("emptyw", 0, 4, 6, 6, 1, 1, 3, 1, 0xD60C);
  rowb("emptyh", 4, 0, 6, 6, 1, 1, 3, 1, 0xD60D);
  rowb("negstep", 4, 4, 6, 6, 1, 1, 3, 1, 0xD60E, -48);
  colb("offset", 5, 5, 12, 10, 4, 6, 3, 1, 0xD621);
  colb("beyond", 9, 9, 4, 4, 1, 1, 3, 1, 0xD622);
  colb("last_off", 3, 3, 8, 8, 7, 7, 3, 1, 0xD623);
  colb("anchor9", 4, 4, 6, 6, 1, 1, 3, 9, 0xD624);
  colb("mask0", 4, 4, 6, 6, 1, 1, 0, 0, 0xD625);
  colb("mask_neg", 4, 4, 6, 6, 1, 1, -3, 0, 0xD626);
  colb("mask9", 4, 4, 6, 6, 1, 1, 9, 4, 0xD627);
  colb("src1x1", 4, 4, 1, 1, 0, 0, 3, 1, 0xD628);
  colb("emptyh", 4, 0, 6, 6, 1, 1, 3, 1, 0xD629);
}

// ---------------------------------------------------------------------------
// Error statuses and validation order.
// ---------------------------------------------------------------------------
TEST_F(NppiSumWindowDiff, Errors) {
  DeviceMemory<Npp16u> &s = arena<Npp16u>(4096);
  DeviceMemory<Npp32f> &d = arena<Npp32f>(4096);
  const Npp16u *ps = s.get();
  Npp32f *pd = d.get();
  auto ctx = current_ctx();
  const NppiSize roi{8, 8};
  const NppiSize sz{9, 9};
  const NppiPoint off{2, 3};
  auto row = [&](const Npp16u *src, int ss, Npp32f *dst, int ds, NppiSize o, int m, int a) {
    return nppiSumWindowRow_16u32f_C1R_Ctx(src, ss, dst, ds, o, m, a, ctx);
  };
  auto col = [&](const Npp16u *src, int ss, Npp32f *dst, int ds, NppiSize o, int m, int a) {
    return nppiSumWindowColumn_16u32f_C1R_Ctx(src, ss, dst, ds, o, m, a, ctx);
  };
  status_case("nppiSumWindowRow", "nppiSumWindowRow_16u32f_C1R", "err/null_src",
              [&] { return row(nullptr, 16, pd, 32, roi, 3, 1); });
  status_case("nppiSumWindowRow", "nppiSumWindowRow_16u32f_C1R", "err/null_dst",
              [&] { return row(ps, 16, nullptr, 32, roi, 3, 1); });
  status_case("nppiSumWindowRow", "nppiSumWindowRow_16u32f_C1R", "err/neg_roi",
              [&] { return row(ps, 16, pd, 32, NppiSize{-1, 8}, 3, 1); });
  status_case("nppiSumWindowRow", "nppiSumWindowRow_16u32f_C1R", "err/anchor_high",
              [&] { return row(ps, 16, pd, 32, roi, 3, 4); });
  status_case("nppiSumWindowRow", "nppiSumWindowRow_16u32f_C1R", "err/anchor_low",
              [&] { return row(ps, 16, pd, 32, roi, 3, -8); });
  // Lower-anchor-bound discriminators for W=8: candidate bounds are
  // a >= 1-W, a >= 2-W, a >= 1-mask and a >= -W/2.
  status_case("nppiSumWindowRow", "nppiSumWindowRow_16u32f_C1R", "bound/m3_a_neg2",
              [&] { return row(ps, 16, pd, 32, roi, 3, -2); });
  status_case("nppiSumWindowRow", "nppiSumWindowRow_16u32f_C1R", "bound/m3_a_neg3",
              [&] { return row(ps, 16, pd, 32, roi, 3, -3); });
  status_case("nppiSumWindowRow", "nppiSumWindowRow_16u32f_C1R", "bound/m3_a_neg5",
              [&] { return row(ps, 16, pd, 32, roi, 3, -5); });
  status_case("nppiSumWindowRow", "nppiSumWindowRow_16u32f_C1R", "bound/m3_a_neg6",
              [&] { return row(ps, 16, pd, 32, roi, 3, -6); });
  status_case("nppiSumWindowRow", "nppiSumWindowRow_16u32f_C1R", "bound/m3_a_neg7",
              [&] { return row(ps, 16, pd, 32, roi, 3, -7); });
  status_case("nppiSumWindowRow", "nppiSumWindowRow_16u32f_C1R", "bound/m5_a_neg5",
              [&] { return row(ps, 16, pd, 32, roi, 5, -5); });
  status_case("nppiSumWindowRow", "nppiSumWindowRow_16u32f_C1R", "bound/m1_a_neg1",
              [&] { return row(ps, 16, pd, 32, roi, 1, -1); });
  status_case("nppiSumWindowRow", "nppiSumWindowRow_16u32f_C1R", "bound/w5_m3_a_neg3",
              [&] { return row(ps, 16, pd, 32, NppiSize{5, 8}, 3, -3); });
  status_case("nppiSumWindowRow", "nppiSumWindowRow_16u32f_C1R", "err/anchor_high_vs_step",
              [&] { return row(ps, 16, pd, 31, roi, 3, 4); });
  status_case("nppiSumWindowRow", "nppiSumWindowRow_16u32f_C1R", "err/step_small",
              [&] { return row(ps, 16, pd, 31, roi, 3, 1); });
  status_case("nppiSumWindowRow", "nppiSumWindowRow_16u32f_C1R", "err/step_zero",
              [&] { return row(ps, 16, pd, 0, roi, 3, 1); });
  status_case("nppiSumWindowRow", "nppiSumWindowRow_16u32f_C1R", "ok/empty_roi",
              [&] { return row(ps, 16, pd, 32, NppiSize{0, 8}, 3, 1); });
  status_case("nppiSumWindowRow", "nppiSumWindowRow_16u32f_C1R", "ok/mask0",
              [&] { return row(ps, 16, pd, 32, roi, 0, 0); });
  status_case("nppiSumWindowRow", "nppiSumWindowRow_16u32f_C1R", "ok/neg_step",
              [&] { return row(ps, 16, pd + 3000, -64, roi, 3, 1); });
  status_case("nppiSumWindowColumn", "nppiSumWindowColumn_16u32f_C1R", "err/null_src",
              [&] { return col(nullptr, 16, pd, 32, roi, 3, 1); });
  status_case("nppiSumWindowColumn", "nppiSumWindowColumn_16u32f_C1R", "err/anchor_high",
              [&] { return col(ps, 16, pd, 32, roi, 3, 4); });
  status_case("nppiSumWindowColumn", "nppiSumWindowColumn_16u32f_C1R", "err/neg_roi_h",
              [&] { return col(ps, 16, pd, 32, NppiSize{8, -1}, 3, 1); });
  status_case("nppiSumWindowColumn", "nppiSumWindowColumn_16u32f_C1R", "err/step_small",
              [&] { return col(ps, 16, pd, 31, roi, 3, 1); });
  status_case("nppiSumWindowColumn", "nppiSumWindowColumn_16u32f_C1R", "ok/empty_roi",
              [&] { return col(ps, 16, pd, 32, NppiSize{8, 0}, 3, 1); });
  status_case("nppiSumWindowColumn", "nppiSumWindowColumn_16u32f_C1R", "ok/mask0",
              [&] { return col(ps, 16, pd, 32, roi, 0, 0); });
  // Border families.
  auto rowb = [&](const Npp16u *src, int ss, NppiSize szb, NppiPoint ofb, Npp32f *dst, int ds,
                  NppiSize o, int m, int a, NppiBorderType bt) {
    return nppiSumWindowRowBorder_16u32f_C1R_Ctx(src, ss, szb, ofb, dst, ds, o, m, a, bt, ctx);
  };
  auto colb = [&](const Npp16u *src, int ss, NppiSize szb, NppiPoint ofb, Npp32f *dst, int ds,
                  NppiSize o, int m, int a, NppiBorderType bt) {
    return nppiSumWindowColumnBorder_16u32f_C1R_Ctx(src, ss, szb, ofb, dst, ds, o, m, a, bt, ctx);
  };
  status_case("nppiSumWindowRowBorder", "nppiSumWindowRowBorder_16u32f_C1R", "err/null_src",
              [&] { return rowb(nullptr, 16, sz, off, pd, 32, roi, 3, 1, NPP_BORDER_REPLICATE); });
  status_case("nppiSumWindowRowBorder", "nppiSumWindowRowBorder_16u32f_C1R", "err/null_dst",
              [&] { return rowb(ps, 16, sz, off, nullptr, 32, roi, 3, 1, NPP_BORDER_REPLICATE); });
  status_case("nppiSumWindowRowBorder", "nppiSumWindowRowBorder_16u32f_C1R", "err/bad_border",
              [&] { return rowb(ps, 16, sz, off, pd, 32, roi, 3, 1, NPP_BORDER_CONSTANT); });
  status_case("nppiSumWindowRowBorder", "nppiSumWindowRowBorder_16u32f_C1R", "err/border_wrap",
              [&] { return rowb(ps, 16, sz, off, pd, 32, roi, 3, 1, NPP_BORDER_WRAP); });
  status_case("nppiSumWindowRowBorder", "nppiSumWindowRowBorder_16u32f_C1R", "err/src_size_zero",
              [&] { return rowb(ps, 16, NppiSize{0, 9}, off, pd, 32, roi, 3, 1, NPP_BORDER_REPLICATE); });
  status_case("nppiSumWindowRowBorder", "nppiSumWindowRowBorder_16u32f_C1R", "err/src_size_neg",
              [&] { return rowb(ps, 16, NppiSize{-1, 9}, off, pd, 32, roi, 3, 1, NPP_BORDER_REPLICATE); });
  status_case("nppiSumWindowRowBorder", "nppiSumWindowRowBorder_16u32f_C1R",
              "err/neg_roi_beats_border",
              [&] { return rowb(ps, 16, sz, off, pd, 32, NppiSize{-1, 7}, 3, 1, NPP_BORDER_CONSTANT); });
  status_case("nppiSumWindowRowBorder", "nppiSumWindowRowBorder_16u32f_C1R", "err/offset_oob",
              [&] { return rowb(ps, 16, sz, NppiPoint{9, 3}, pd, 32, roi, 3, 1, NPP_BORDER_REPLICATE); });
  status_case("nppiSumWindowRowBorder", "nppiSumWindowRowBorder_16u32f_C1R", "err/offset_neg",
              [&] { return rowb(ps, 16, sz, NppiPoint{2, -1}, pd, 32, roi, 3, 1, NPP_BORDER_REPLICATE); });
  status_case("nppiSumWindowRowBorder", "nppiSumWindowRowBorder_16u32f_C1R", "err/offset_vs_step",
              [&] { return rowb(ps, 16, sz, NppiPoint{9, 3}, pd, 31, roi, 3, 1, NPP_BORDER_REPLICATE); });
  status_case("nppiSumWindowRowBorder", "nppiSumWindowRowBorder_16u32f_C1R", "err/step_small",
              [&] { return rowb(ps, 16, sz, off, pd, 31, roi, 3, 1, NPP_BORDER_REPLICATE); });
  status_case("nppiSumWindowRowBorder", "nppiSumWindowRowBorder_16u32f_C1R", "ok/empty_roi",
              [&] { return rowb(ps, 16, sz, off, pd, 32, NppiSize{0, 7}, 3, 1, NPP_BORDER_REPLICATE); });
  status_case("nppiSumWindowRowBorder", "nppiSumWindowRowBorder_16u32f_C1R", "ok/mask0",
              [&] { return rowb(ps, 16, sz, off, pd, 32, roi, 0, 0, NPP_BORDER_REPLICATE); });
  status_case("nppiSumWindowRowBorder", "nppiSumWindowRowBorder_16u32f_C1R", "ok/neg_step",
              [&] { return rowb(ps, 16, sz, off, pd + 3000, -64, roi, 3, 1, NPP_BORDER_REPLICATE); });
  status_case("nppiSumWindowColumnBorder", "nppiSumWindowColumnBorder_16u32f_C1R", "err/bad_border",
              [&] { return colb(ps, 16, sz, off, pd, 32, roi, 3, 1, NPP_BORDER_CONSTANT); });
  status_case("nppiSumWindowColumnBorder", "nppiSumWindowColumnBorder_16u32f_C1R", "err/null_dst",
              [&] { return colb(ps, 16, sz, off, nullptr, 32, roi, 3, 1, NPP_BORDER_REPLICATE); });
  status_case("nppiSumWindowColumnBorder", "nppiSumWindowColumnBorder_16u32f_C1R", "err/offset_oob",
              [&] { return colb(ps, 16, sz, NppiPoint{3, 9}, pd, 32, roi, 3, 1, NPP_BORDER_REPLICATE); });
  status_case("nppiSumWindowColumnBorder", "nppiSumWindowColumnBorder_16u32f_C1R", "err/step_small",
              [&] { return colb(ps, 16, sz, off, pd, 31, roi, 3, 1, NPP_BORDER_REPLICATE); });
  status_case("nppiSumWindowColumnBorder", "nppiSumWindowColumnBorder_16u32f_C1R", "ok/empty_roi",
              [&] { return colb(ps, 16, sz, off, pd, 32, NppiSize{7, 0}, 3, 1, NPP_BORDER_REPLICATE); });
  // Legacy ("real") 8u32f_C1R validation probes.
  auto row8 = [&](const Npp8u *src, int ss, Npp32f *dst, int ds, NppiSize o, int m, int a) {
    return nppiSumWindowRow_8u32f_C1R_Ctx(src, ss, dst, ds, o, m, a, ctx);
  };
  DeviceMemory<Npp8u> &s8 = arena<Npp8u>(4096);
  const Npp8u *ps8 = s8.get();
  status_case("nppiSumWindowRow", "nppiSumWindowRow_8u32f_C1R", "err/8u/anchor_eq_mask",
              [&] { return row8(ps8, 16, pd, 32, roi, 3, 3); });
  status_case("nppiSumWindowRow", "nppiSumWindowRow_8u32f_C1R", "err/8u/anchor_neg",
              [&] { return row8(ps8, 16, pd, 32, roi, 3, -1); });
  status_case("nppiSumWindowRow", "nppiSumWindowRow_8u32f_C1R", "err/8u/mask_neg",
              [&] { return row8(ps8, 16, pd, 32, roi, -3, 0); });
  status_case("nppiSumWindowRow", "nppiSumWindowRow_8u32f_C1R", "err/8u/empty_roi",
              [&] { return row8(ps8, 16, pd, 32, NppiSize{0, 8}, 3, 1); });
  status_case("nppiSumWindowRow", "nppiSumWindowRow_8u32f_C1R", "err/8u/step_zero",
              [&] { return row8(ps8, 16, pd, 0, roi, 3, 1); });
  status_case("nppiSumWindowRow", "nppiSumWindowRow_8u32f_C1R", "ok/8u/neg_step",
              [&] { return row8(ps8, 16, pd + 3000, -64, roi, 3, 1); });
}
