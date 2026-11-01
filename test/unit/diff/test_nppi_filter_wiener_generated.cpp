// Differential tests for package f10: nppiFilterWienerBorder.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiFilterWienerDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records ./diff_tests \
//     --gtest_filter='NppiFilterWienerDiff.*'
//
// Measured semantics (5090, NPP 13.4; libnppif SASS + probes
// tools/probe_f10/fit/wiener_fit2.cu, wiener_fit3.cu, wiener_shape.cu):
//  - only NPP_BORDER_REPLICATE is supported; reads clamp to the source image
//    bounds relative to pSrc: [-off, srcSize-1-off];
//  - the statistics window is NPP's quirky loop: for mask height 1 the mean
//    visits row y+ay and the variance row y+ay-1 with columns ax-(w-1)..ax;
//    otherwise there are h-1 rows starting at dy=ay going down, the first row
//    visiting all w columns and later rows only w-1 (dropping the leftmost).
//    N = w*h, so the variance can be negative; duplicated clamped cells count
//    multiple times.  AC4 is the exception: it uses the plain w x h rectangle
//    for both mean and variance (K == N);
//  - mean = sum*(1/N); var = sum(fma(v,v,-mean*mean))*(1/N) (the SASS order);
//  - aNoise == 0 outputs the variance; integer dtypes scale the noise by the
//    dynamic range squared (256-1)^2 / (32768-1)^2, 32f uses it directly;
//    dst = mean if var <= noise, else mean + (var-noise)/var*(src-mean);
//  - integer destinations truncate toward zero and saturate;
//  - AC4 computes channels 0..2 only and leaves the alpha channel untouched;
//  - validation order: aNoise range (over the used channels, OUT_OFF_RANGE) ->
//    null src/dst -> negative ROI -> srcSize -> offset -> border type ->
//    mask <= 0 -> anchor (anchor.x >= width or anchor.y >= height ->
//    OUT_OFF_RANGE; negative anchors pass) -> zero-size ROI -> dst step.  A null aNoise segfaults on NPP 13.4 and is
//    therefore not exercised here.
//
// Integer outputs use a 1-LSB tolerance (float summation order); the probes
// matched exactly for 8u/16s on random data, with occasional +/-1 on 16s.

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <algorithm>
#include <memory>
#include <string>
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

template <typename T> ValueKind kind_of();
template <> ValueKind kind_of<Npp8u>() { return ValueKind::U8Tol; }
template <> ValueKind kind_of<Npp16s>() { return ValueKind::I16Tol; }
template <> ValueKind kind_of<Npp32f>() { return ValueKind::F32; }

template <typename T> void tolerance(double &rtol, double &atol) {
  if constexpr (std::is_floating_point_v<T>) {
    rtol = 1e-3;
    atol = 1e-3;
  } else {
    rtol = 0.0;
    atol = 1.0;
  }
}

// ---------------------------------------------------------------- helpers

template <typename T, typename F>
void wiener_ok(const std::string &fname, const std::string &key, int channels, int srcW, int srcH, int offX,
               int offY, int W, int H, int mw, int mh, int ax, int ay, const std::vector<float> &noise,
               int pad, uint64_t seed, F f) {
  const int spx = channels * (int)sizeof(T);
  const int sStep = align_step(srcW * spx + pad, spx);
  const int dpx = channels * (int)sizeof(T);
  const int dAlign = channels == 4 ? dpx : (int)sizeof(T);
  const int dStep = align_step(W * dpx + pad, dAlign);
  const size_t se = (size_t)srcH * sStep / sizeof(T);
  const size_t de = (size_t)H * dStep / sizeof(T);
  std::vector<T> hs = make_uniform<T>(se, seed + 11);
  std::vector<T> hd = make_uniform<T>(de, seed + 29);
  DeviceMemory<T> &s = arena<T>(se);
  DeviceMemory<T> &d = arena<T>(de);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  const T *pSrc = s.get() + (size_t)offY * (sStep / sizeof(T)) + (size_t)offX * channels;
  std::vector<float> nz(noise);
  nz.resize(std::max<size_t>(nz.size(), 4), 0.f);
  clear_cuda_error();
  NppStatus st = f(pSrc, sStep, NppiSize{srcW, srcH}, NppiPoint{offX, offY}, d.get(), dStep, NppiSize{W, H},
                   NppiSize{mw, mh}, NppiPoint{ax, ay}, nz.data(), NPP_BORDER_REPLICATE, current_ctx());
  double rtol, atol;
  tolerance<T>(rtol, atol);
  Case c("nppiFilterWiener/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(T));
  c.status(st);
  c.device_output<T>(d.get(), de, kind_of<T>(), rtol, atol);
  c.finish();
}

template <typename F> void only_status(const std::string &fname, const std::string &key, F f) {
  clear_cuda_error();
  NppStatus st = f();
  Case c("nppiFilterWiener/" + key, fname);
  c.status(st);
  c.finish();
}

// --------------------------------------------------------- error matrices

template <typename T, typename F>
void wiener_errors(const std::string &fam, const std::string &suffix, int channels, T *s, T *d, F f) {
  NppStreamContext ctx = current_ctx();
  const std::string fn = fam + suffix;
  const NppiSize sz{7, 7};
  const NppiPoint off{2, 2};
  const NppiSize roi{4, 4};
  const NppiSize zroi{0, 0};
  const int dpx = channels * (int)sizeof(T);
  const int dAlign = channels == 4 ? dpx : (int)sizeof(T);
  const NppiSize good{3, 3};
  const NppiPoint ga{1, 1};
  std::vector<float> noise4(std::max(channels, 4), 0.5f);
  std::vector<float> noise1 = noise4;
  noise1[0] = 1.5f;
  std::vector<float> noiseLast = noise4;
  noiseLast[channels - 1] = -0.25f;
  float *nz = noise4.data();
  float *bad0 = noise1.data();
  float *badlast = noiseLast.data();
  const NppiBorderType rep = NPP_BORDER_REPLICATE;
  const NppiBorderType badB = NPP_BORDER_CONSTANT;
  only_status(fn, fam + "/err/badnoise0" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, roi, good, ga, bad0, rep, ctx); });
  only_status(fn, fam + "/err/badnoise_last" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, roi, good, ga, badlast, rep, ctx); });
  // noise range is validated before every other check
  only_status(fn, fam + "/err/badnoise_null" + suffix,
              [&] { return f(nullptr, 16, sz, off, d, 16, roi, good, ga, bad0, rep, ctx); });
  only_status(fn, fam + "/err/badnoise_negroi" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, NppiSize{4, -1}, good, ga, bad0, rep, ctx); });
  only_status(fn, fam + "/err/badnoise_border" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, roi, good, ga, bad0, badB, ctx); });
  only_status(fn, fam + "/err/badnoise_mask0" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, roi, NppiSize{0, 0}, ga, bad0, rep, ctx); });
  only_status(fn, fam + "/err/nullsrc" + suffix,
              [&] { return f(nullptr, 16, sz, off, d, 16, roi, good, ga, nz, rep, ctx); });
  only_status(fn, fam + "/err/nulldst" + suffix,
              [&] { return f(s, 16, sz, off, nullptr, 16, roi, good, ga, nz, rep, ctx); });
  only_status(fn, fam + "/err/negroi" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, NppiSize{4, -1}, good, ga, nz, rep, ctx); });
  only_status(fn, fam + "/err/srcsize0" + suffix,
              [&] { return f(s, 16, NppiSize{0, 5}, off, d, 16, roi, good, ga, nz, rep, ctx); });
  only_status(fn, fam + "/err/offneg" + suffix,
              [&] { return f(s, 16, sz, NppiPoint{-1, 0}, d, 16, roi, good, ga, nz, rep, ctx); });
  only_status(fn, fam + "/err/off_eq_w" + suffix,
              [&] { return f(s, 16, sz, NppiPoint{7, 2}, d, 16, roi, good, ga, nz, rep, ctx); });
  only_status(fn, fam + "/err/badborder" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, roi, good, ga, nz, badB, ctx); });
  only_status(fn, fam + "/err/mask0" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, roi, NppiSize{0, 3}, ga, nz, rep, ctx); });
  only_status(fn, fam + "/err/maskneg" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, roi, NppiSize{-3, 3}, ga, nz, rep, ctx); });
  only_status(fn, fam + "/err/mask0_badanchor" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, roi, NppiSize{0, 0}, NppiPoint{9, 9}, nz, rep, ctx); });
  only_status(fn, fam + "/err/badanchor_x" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, roi, good, NppiPoint{3, 1}, nz, rep, ctx); });
  only_status(fn, fam + "/err/badanchor_y" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, roi, good, NppiPoint{1, 3}, nz, rep, ctx); });
  only_status(fn, fam + "/err/anchor_x_last" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, roi, good, NppiPoint{2, 1}, nz, rep, ctx); });
  only_status(fn, fam + "/err/roizero" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, zroi, good, ga, nz, rep, ctx); });
  only_status(fn, fam + "/err/roizero_badanchor" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, zroi, good, NppiPoint{9, 9}, nz, rep, ctx); });
  only_status(fn, fam + "/err/roizero_badborder" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, zroi, good, ga, nz, badB, ctx); });
  only_status(fn, fam + "/err/roizero_badstep" + suffix,
              [&] { return f(s, 16, sz, off, d, 3, zroi, good, ga, nz, rep, ctx); });
  only_status(fn, fam + "/err/badanchor_stepsmall" + suffix,
              [&] { return f(s, 16, sz, off, d, 4 * dpx - 1, roi, good, NppiPoint{9, 9}, nz, rep, ctx); });
  only_status(fn, fam + "/err/stepsmall" + suffix,
              [&] { return f(s, 16, sz, off, d, 4 * dpx - 1, roi, good, ga, nz, rep, ctx); });
  if (dAlign > 1)
    only_status(fn, fam + "/err/stepodd" + suffix,
                [&] { return f(s, 16, sz, off, d, 4 * dpx + 1, roi, good, ga, nz, rep, ctx); });
}

}  // namespace

class NppiFilterWienerDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

// ========================= nppiFilterWienerBorder =========================
TEST_F(NppiFilterWienerDiff, WienerBorder) {
  // mask / anchor / noise matrix on C1 for each supported dtype
  struct Case {
    int mw, mh, ax, ay;
    float n;
    const char *tag;
  } cases[] = {
      {3, 3, 1, 1, 1.0f, "m33n1"},    {3, 3, 1, 1, 0.0f, "m33n0"},   {3, 3, 1, 1, 0.1f, "m33n01"},
      {3, 3, 1, 1, 0.5f, "m33n05"},   {5, 5, 2, 2, 1.0f, "m55n1"},   {5, 5, 2, 2, 0.25f, "m55n025"},
      {7, 7, 3, 3, 1.0f, "m77n1"},    {4, 4, 1, 1, 0.25f, "m44n025"}, {6, 6, 2, 2, 0.1f, "m66n01"},
      {2, 2, 0, 0, 1.0f, "m22n1"},    {1, 1, 0, 0, 1.0f, "m11n1"},   {3, 5, 1, 2, 0.5f, "m35n05"},
      {5, 3, 2, 1, 0.5f, "m53n05"},   {1, 3, 0, 1, 1.0f, "m13n1"},   {3, 1, 1, 0, 1.0f, "m31n1"},
      {1, 3, 0, 0, 0.0f, "m13a00n0"}, {3, 1, 0, 0, 0.1f, "m31a00n01"}, {3, 3, 0, 0, 0.1f, "m33a00n01"},
      {3, 3, 2, 2, 0.1f, "m33a22n01"}, {3, 3, 2, 0, 0.1f, "m33a20n01"},
  };
  for (const Case &c : cases) {
    wiener_ok<Npp32f>("nppiFilterWienerBorder_32f_C1R_Ctx",
                      std::string("nppiFilterWienerBorder/32f/C1/") + c.tag, 1, 15, 13, 2, 3, 11, 9,
                      c.mw, c.mh, c.ax, c.ay, {c.n}, 3, __LINE__, nppiFilterWienerBorder_32f_C1R_Ctx);
    wiener_ok<Npp8u>("nppiFilterWienerBorder_8u_C1R_Ctx",
                     std::string("nppiFilterWienerBorder/8u/C1/") + c.tag, 1, 15, 13, 2, 3, 11, 9,
                     c.mw, c.mh, c.ax, c.ay, {c.n}, 3, __LINE__, nppiFilterWienerBorder_8u_C1R_Ctx);
    wiener_ok<Npp16s>("nppiFilterWienerBorder_16s_C1R_Ctx",
                      std::string("nppiFilterWienerBorder/16s/C1/") + c.tag, 1, 15, 13, 2, 3, 11, 9,
                      c.mw, c.mh, c.ax, c.ay, {c.n}, 3, __LINE__, nppiFilterWienerBorder_16s_C1R_Ctx);
  }
  // geometry edges: corner offset, overhang, tiny source, padded steps
  wiener_ok<Npp32f>("nppiFilterWienerBorder_32f_C1R_Ctx", "nppiFilterWienerBorder/32f/C1/corner", 1, 13,
                    13, 0, 0, 9, 9, 3, 3, 1, 1, {0.5f}, 3, __LINE__,
                    nppiFilterWienerBorder_32f_C1R_Ctx);
  wiener_ok<Npp32f>("nppiFilterWienerBorder_32f_C1R_Ctx", "nppiFilterWienerBorder/32f/C1/overhang", 1,
                    13, 13, 10, 10, 5, 5, 5, 5, 2, 2, {0.25f}, 3, __LINE__,
                    nppiFilterWienerBorder_32f_C1R_Ctx);
  wiener_ok<Npp8u>("nppiFilterWienerBorder_8u_C1R_Ctx", "nppiFilterWienerBorder/8u/C1/tiny", 1, 2, 2,
                   1, 1, 4, 4, 3, 3, 1, 1, {1.f}, 3, __LINE__, nppiFilterWienerBorder_8u_C1R_Ctx);
  wiener_ok<Npp16s>("nppiFilterWienerBorder_16s_C1R_Ctx", "nppiFilterWienerBorder/16s/C1/pad", 1, 17,
                    15, 2, 2, 11, 9, 5, 5, 2, 2, {0.1f}, 7, __LINE__,
                    nppiFilterWienerBorder_16s_C1R_Ctx);
  // negative anchors pass validation (and shift the window)
  wiener_ok<Npp32f>("nppiFilterWienerBorder_32f_C1R_Ctx", "nppiFilterWienerBorder/32f/C1/anchor_neg",
                    1, 15, 13, 2, 3, 11, 9, 3, 3, -2, -1, {0.5f}, 3, __LINE__,
                    nppiFilterWienerBorder_32f_C1R_Ctx);
  // wide / tall masks with a large anchor component
  wiener_ok<Npp32f>("nppiFilterWienerBorder_32f_C1R_Ctx", "nppiFilterWienerBorder/32f/C1/m1x100", 1,
                    15, 13, 2, 3, 11, 9, 1, 8, 0, 7, {0.5f}, 3, __LINE__,
                    nppiFilterWienerBorder_32f_C1R_Ctx);

  // per-channel noise + layout variants
  const std::vector<float> n3 = {0.5f, 0.1f, 1.0f};
  const std::vector<float> n4 = {0.5f, 0.1f, 0.5f, 0.25f};
  wiener_ok<Npp8u>("nppiFilterWienerBorder_8u_C3R_Ctx", "nppiFilterWienerBorder/8u/C3/m33", 3, 15, 13,
                   2, 3, 11, 9, 3, 3, 1, 1, n3, 3, __LINE__, nppiFilterWienerBorder_8u_C3R_Ctx);
  wiener_ok<Npp16s>("nppiFilterWienerBorder_16s_C3R_Ctx", "nppiFilterWienerBorder/16s/C3/m55", 3, 15,
                    13, 2, 3, 11, 9, 5, 5, 2, 2, n3, 3, __LINE__,
                    nppiFilterWienerBorder_16s_C3R_Ctx);
  wiener_ok<Npp32f>("nppiFilterWienerBorder_32f_C3R_Ctx", "nppiFilterWienerBorder/32f/C3/m44", 3, 15,
                    13, 2, 3, 11, 9, 4, 4, 1, 1, n3, 3, __LINE__,
                    nppiFilterWienerBorder_32f_C3R_Ctx);
  wiener_ok<Npp8u>("nppiFilterWienerBorder_8u_C4R_Ctx", "nppiFilterWienerBorder/8u/C4/m33", 4, 15, 13,
                   2, 3, 11, 9, 3, 3, 1, 1, n4, 3, __LINE__, nppiFilterWienerBorder_8u_C4R_Ctx);
  wiener_ok<Npp32f>("nppiFilterWienerBorder_32f_C4R_Ctx", "nppiFilterWienerBorder/32f/C4/m55", 4, 15,
                    13, 2, 3, 11, 9, 5, 5, 2, 2, n4, 3, __LINE__,
                    nppiFilterWienerBorder_32f_C4R_Ctx);
  wiener_ok<Npp8u>("nppiFilterWienerBorder_8u_AC4R_Ctx", "nppiFilterWienerBorder/8u/AC4/m33", 4, 15,
                   13, 2, 3, 11, 9, 3, 3, 1, 1, n3, 3, __LINE__, nppiFilterWienerBorder_8u_AC4R_Ctx);
  wiener_ok<Npp16s>("nppiFilterWienerBorder_16s_AC4R_Ctx", "nppiFilterWienerBorder/16s/AC4/m33", 4,
                    15, 13, 2, 3, 11, 9, 3, 3, 1, 1, n3, 3, __LINE__,
                    nppiFilterWienerBorder_16s_AC4R_Ctx);
  wiener_ok<Npp32f>("nppiFilterWienerBorder_32f_AC4R_Ctx", "nppiFilterWienerBorder/32f/AC4/m77", 4, 15,
                    13, 2, 3, 11, 9, 7, 7, 3, 3, n3, 3, __LINE__,
                    nppiFilterWienerBorder_32f_AC4R_Ctx);

  wiener_errors<Npp8u>("nppiFilterWienerBorder", "_8u_C1R_Ctx", 1, arena<Npp8u>(64).get(),
                       arena<Npp8u>(64).get(), nppiFilterWienerBorder_8u_C1R_Ctx);
  wiener_errors<Npp16s>("nppiFilterWienerBorder", "_16s_C1R_Ctx", 1, arena<Npp16s>(64).get(),
                        arena<Npp16s>(64).get(), nppiFilterWienerBorder_16s_C1R_Ctx);
  wiener_errors<Npp32f>("nppiFilterWienerBorder", "_32f_C1R_Ctx", 1, arena<Npp32f>(64).get(),
                        arena<Npp32f>(64).get(), nppiFilterWienerBorder_32f_C1R_Ctx);
  wiener_errors<Npp8u>("nppiFilterWienerBorder", "_8u_AC4R_Ctx", 4, arena<Npp8u>(64).get(),
                       arena<Npp8u>(64).get(), nppiFilterWienerBorder_8u_AC4R_Ctx);
  wiener_errors<Npp16s>("nppiFilterWienerBorder", "_16s_C4R_Ctx", 4, arena<Npp16s>(64).get(),
                        arena<Npp16s>(64).get(), nppiFilterWienerBorder_16s_C4R_Ctx);
}
