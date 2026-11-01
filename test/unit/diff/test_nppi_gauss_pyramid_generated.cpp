// Differential tests for package f13 family 3: Gauss pyramid layers
// (nppiFilterGaussPyramidLayerDownBorder, 12 functions).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiGaussPyramidDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppiGaussPyramidDiff.*'
//
// Measured semantics (5090, NPP 13.4, probes f13_probe15/16/17/24/25):
//  - down: dst pixel j samples the source at j*rate + (t - center) per tap,
//    center = (taps-1)/2 (integer division), with a bilinear interpolation of
//    the fractional coordinate; taps are applied per axis and are NOT
//    renormalised
//  - REPLICATE clamps the tap coordinate and MIRROR reflects it (reflect-101:
//    -1 -> 1, dim -> dim-2); only those two border types are accepted
//  - coordinates are pSrc-relative: pSrc points at oSrcOffset inside the
//    oSrcSize image, so the clamp window is [-offset, srcSize-1-offset]
//  - 8u/16u accumulate in float and truncate toward zero (then saturate);
//    32f passes through
//  - validation order: null (-8) -> oSrcSize <= 0 (-6) -> negative ROI (-6) ->
//    bad offset (-11) -> bad border (-14) -> taps < 3 or >= 100 (-33) ->
//    rate outside (1, 10] (-5) -> zero ROI (0, no step check) -> dst step too
//    small (-16) -> dst step not a multiple of sizeof(dst) (-108); the source
//    step is never validated
//
// Integer outputs are compared with atol=1 (NPP's float pipeline and ours can
// differ in the last bit before the truncation); float outputs use 1e-4.

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <cmath>
#include <memory>
#include <string>
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
template <> ValueKind kind_of<Npp8u>() { return ValueKind::U8; }
template <> ValueKind kind_of<Npp16u>() { return ValueKind::U16; }
template <> ValueKind kind_of<Npp32f>() { return ValueKind::F32; }

template <typename T> double kind_atol();
template <> double kind_atol<Npp8u>() { return 1.0; }
template <> double kind_atol<Npp16u>() { return 1.0; }
template <> double kind_atol<Npp32f>() { return 1e-4; }

template <typename T> double kind_rtol();
template <> double kind_rtol<Npp8u>() { return 0.0; }
template <> double kind_rtol<Npp16u>() { return 0.0; }
template <> double kind_rtol<Npp32f>() { return 1e-5; }

// ------------------------------------------------------------------ helpers

// Happy path. Source image is srcH rows of srcW pixels; pSrc is inset by the
// requested offset (the NPP convention), so border reads stay inside the
// image. The destination buffer is junk-seeded and its backing store is
// allocated row by row with the same step used for the call.
template <typename S, typename D, typename F>
void pyr_ok(const std::string &fname, const std::string &key, int ch, int srcW, int srcH, int offX,
            int offY, int W, int H, int pad, uint64_t seed, float rate, int taps,
            const std::vector<float> &kern, NppiBorderType border, F f) {
  const int spx = ch * (int)sizeof(S);
  const int sStep = align_step(srcW * spx + pad, spx);
  const int dStep = align_step(W * ch * (int)sizeof(D) + pad, (int)sizeof(D));
  const size_t se = (size_t)srcH * sStep / sizeof(S);
  const size_t de = (size_t)H * dStep / sizeof(D);
  std::vector<S> hs = make_uniform<S>(se, seed + 11);
  std::vector<D> hd = make_uniform<D>(de, seed + 29);
  DeviceMemory<S> &s = arena<S>(se);
  DeviceMemory<D> &d = arena<D>(de);
  DeviceMemory<Npp32f> &k = arena<Npp32f>(kern.size());
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  k.copyFromHost(kern);
  const S *pSrc = s.get() + (size_t)offY * (sStep / sizeof(S)) + (size_t)offX * ch;
  clear_cuda_error();
  NppStatus st = f(pSrc, sStep, NppiSize{srcW, srcH}, NppiPoint{offX, offY}, d.get(), dStep,
                   NppiSize{W, H}, rate, taps, k.get(), border, current_ctx());
  Case c("nppiGaussPyramid/" + fname + "/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(S));
  c.status(st);
  c.device_output<D>(d.get(), de, kind_of<D>(), kind_rtol<D>(), kind_atol<D>());
  c.finish();
}

// Status-only case.
template <typename F> void only_status(const std::string &fname, const std::string &key, F f) {
  clear_cuda_error();
  NppStatus st = f();
  Case c("nppiGaussPyramid/" + fname + "/" + key, fname);
  c.status(st);
  c.finish();
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) ADD_FAILURE() << "async CUDA fault after " << key << ": "
                                      << cudaGetErrorString(e);
}

// ------------------------------------------------------------------ cases

template <typename S, typename D, typename F>
void down_values(const std::string &fam, int ch, F f, uint64_t seed) {
  const std::vector<float> k3 = {0.25f, 0.5f, 0.25f};
  const std::vector<float> k3u = {1.f / 3.f, 1.f / 3.f, 1.f / 3.f};
  const std::vector<float> k5 = {1.f / 16, 4.f / 16, 6.f / 16, 4.f / 16, 1.f / 16};
  const std::vector<float> k4 = {0.1f, 0.2f, 0.3f, 0.4f};
  const int srcW = 24, srcH = 18;
  struct RateCase { float rate; int taps, dw, dh; };
  const RateCase rates[] = {
      {1.5f, 3, 16, 12}, {2.0f, 3, 12, 9}, {2.5f, 3, 9, 7},
      {3.0f, 3, 8, 6},  {3.7f, 3, 6, 4},  {1.25f, 3, 19, 14},
  };
  int idx = 0;
  for (const auto &rc : rates) {
    const std::vector<float> &kern = rc.taps == 5 ? k5 : (idx % 2 ? k3u : k3);
    for (NppiBorderType border : {NPP_BORDER_REPLICATE, NPP_BORDER_MIRROR}) {
      const std::string key = "r" + std::to_string(rc.rate).substr(0, 3) +
                              (border == NPP_BORDER_MIRROR ? "/mirror/" : "/replicate/") +
                              std::to_string(ch) + "ch";
      pyr_ok<S, D>(fam, key, ch, srcW, srcH, 0, 0, rc.dw, rc.dh, idx % 3, seed + 100 * idx,
                   rc.rate, rc.taps, kern, border, f);
      ++idx;
    }
  }
  // 5 taps, 4 taps (even) and an offset in the interior
  pyr_ok<S, D>(fam, "k5/5ch", ch, srcW, srcH, 0, 0, 12, 9, 1, seed + 7, 2.0f, 5, k5,
               NPP_BORDER_REPLICATE, f);
  pyr_ok<S, D>(fam, "k5mirror/5ch", ch, srcW, srcH, 0, 0, 12, 9, 2, seed + 8, 2.0f, 5, k5,
               NPP_BORDER_MIRROR, f);
  pyr_ok<S, D>(fam, "even4/4ch", ch, srcW, srcH, 0, 0, 10, 8, 0, seed + 9, 2.0f, 4, k4,
               NPP_BORDER_REPLICATE, f);
  pyr_ok<S, D>(fam, "off2_3/3ch", ch, srcW, srcH, 2, 3, 8, 6, 3, seed + 10, 2.0f, 3, k3,
               NPP_BORDER_REPLICATE, f);
  pyr_ok<S, D>(fam, "off3mirror/3ch", ch, srcW, srcH, 3, 2, 8, 6, 0, seed + 11, 2.5f, 3, k5,
               NPP_BORDER_MIRROR, f);
  // ROI overhanging the source image: reads clamp, every ROI pixel is written
  pyr_ok<S, D>(fam, "overhang/3ch", ch, srcW, srcH, 0, 0, 16, 12, 2, seed + 12, 1.5f, 3, k3,
               NPP_BORDER_REPLICATE, f);
  // rate 10 (the upper bound is accepted)
  pyr_ok<S, D>(fam, "rate10/3ch", ch, srcW, srcH, 0, 0, 3, 2, 0, seed + 13, 10.0f, 3, k3u,
               NPP_BORDER_REPLICATE, f);
}

template <typename S, typename D, typename F>
void down_errors_typed(const std::string &fam, int ch, F f, uint64_t seed) {
  const std::vector<float> k3 = {0.25f, 0.5f, 0.25f};
  const int W = 12, H = 8;
  const int sStep = align_step(W * ch * (int)sizeof(S), ch * (int)sizeof(S));
  const int dRow = 4 * ch * (int)sizeof(D);
  const size_t se = (size_t)H * sStep / sizeof(S);
  const size_t de = (size_t)6 * align_step(W * ch * (int)sizeof(D), (int)sizeof(D)) / sizeof(D);
  std::vector<S> hs = make_uniform<S>(se, seed + 3);
  std::vector<D> hd = make_uniform<D>(de, seed + 5);
  DeviceMemory<S> &s = arena<S>(se);
  DeviceMemory<D> &d = arena<D>(de);
  DeviceMemory<Npp32f> &k = arena<Npp32f>(3);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  k.copyFromHost(k3);
  const S *pSrc = s.get();
  const Npp32f *pK = k.get();
  auto run = [&](const S *src, int sstep, NppiSize ss, NppiPoint off, D *dst, int dstep,
                 NppiSize roi, float rate, int taps, const Npp32f *pk, NppiBorderType bt) {
    return f(src, sstep, ss, off, dst, dstep, roi, rate, taps, pk, bt, current_ctx());
  };
  using namespace std::placeholders;
  const std::string suf = "/" + std::to_string(ch) + "ch";
  auto S_ok = [&](const char *tag, auto call) {
    only_status(fam + suf, tag, [&]() { return call(); });
  };
  (void)S_ok;
  // happy status baseline
  only_status(fam + suf, "ok", [&]() {
    return run(pSrc, sStep, {W, H}, {0, 0}, d.get(), dRow, {4, 4}, 2.0f, 3, pK,
               NPP_BORDER_REPLICATE);
  });
  only_status(fam + suf, "null_src", [&]() {
    return run(nullptr, sStep, {W, H}, {0, 0}, d.get(), dRow, {4, 4}, 2.0f, 3, pK,
               NPP_BORDER_REPLICATE);
  });
  only_status(fam + suf, "null_dst", [&]() {
    return run(pSrc, sStep, {W, H}, {0, 0}, nullptr, dRow, {4, 4}, 2.0f, 3, pK,
               NPP_BORDER_REPLICATE);
  });
  only_status(fam + suf, "null_kernel", [&]() {
    return run(pSrc, sStep, {W, H}, {0, 0}, d.get(), dRow, {4, 4}, 2.0f, 3, nullptr,
               NPP_BORDER_REPLICATE);
  });
  only_status(fam + suf, "srcsize0", [&]() {
    return run(pSrc, sStep, {0, H}, {0, 0}, d.get(), dRow, {4, 4}, 2.0f, 3, pK,
               NPP_BORDER_REPLICATE);
  });
  only_status(fam + suf, "roi_neg", [&]() {
    return run(pSrc, sStep, {W, H}, {0, 0}, d.get(), dRow, {-1, 4}, 2.0f, 3, pK,
               NPP_BORDER_REPLICATE);
  });
  only_status(fam + suf, "roi_zero", [&]() {
    return run(pSrc, sStep, {W, H}, {0, 0}, d.get(), dRow, {0, 0}, 2.0f, 3, pK,
               NPP_BORDER_REPLICATE);
  });
  only_status(fam + suf, "roi_zero_badrate", [&]() {
    return run(pSrc, sStep, {W, H}, {0, 0}, d.get(), dRow, {0, 4}, 1.0f, 3, pK,
               NPP_BORDER_REPLICATE);
  });
  only_status(fam + suf, "roi_zero_badtaps", [&]() {
    return run(pSrc, sStep, {W, H}, {0, 0}, d.get(), dRow, {0, 4}, 2.0f, 1, pK,
               NPP_BORDER_REPLICATE);
  });
  only_status(fam + suf, "roi_zero_badborder", [&]() {
    return run(pSrc, sStep, {W, H}, {0, 0}, d.get(), dRow, {0, 4}, 2.0f, 3, pK,
               NPP_BORDER_CONSTANT);
  });
  only_status(fam + suf, "roi_zero_badstep", [&]() {
    return run(pSrc, sStep, {W, H}, {0, 0}, d.get(), 1, {0, 4}, 2.0f, 3, pK,
               NPP_BORDER_REPLICATE);
  });
  only_status(fam + suf, "off_neg", [&]() {
    return run(pSrc, sStep, {W, H}, {-1, 0}, d.get(), dRow, {4, 4}, 2.0f, 3, pK,
               NPP_BORDER_REPLICATE);
  });
  only_status(fam + suf, "off_eq_size", [&]() {
    return run(pSrc, sStep, {W, H}, {W, 0}, d.get(), dRow, {4, 4}, 2.0f, 3, pK,
               NPP_BORDER_REPLICATE);
  });
  only_status(fam + suf, "border_constant", [&]() {
    return run(pSrc, sStep, {W, H}, {0, 0}, d.get(), dRow, {4, 4}, 2.0f, 3, pK,
               NPP_BORDER_CONSTANT);
  });
  only_status(fam + suf, "taps1", [&]() {
    return run(pSrc, sStep, {W, H}, {0, 0}, d.get(), dRow, {4, 4}, 2.0f, 1, pK,
               NPP_BORDER_REPLICATE);
  });
  only_status(fam + suf, "taps2", [&]() {
    return run(pSrc, sStep, {W, H}, {0, 0}, d.get(), dRow, {4, 4}, 2.0f, 2, pK,
               NPP_BORDER_REPLICATE);
  });
  only_status(fam + suf, "taps100", [&]() {
    return run(pSrc, sStep, {W, H}, {0, 0}, d.get(), dRow, {4, 4}, 2.0f, 100, pK,
               NPP_BORDER_REPLICATE);
  });
  only_status(fam + suf, "rate_1", [&]() {
    return run(pSrc, sStep, {W, H}, {0, 0}, d.get(), dRow, {4, 4}, 1.0f, 3, pK,
               NPP_BORDER_REPLICATE);
  });
  only_status(fam + suf, "rate_0p9", [&]() {
    return run(pSrc, sStep, {W, H}, {0, 0}, d.get(), dRow, {4, 4}, 0.9f, 3, pK,
               NPP_BORDER_REPLICATE);
  });
  only_status(fam + suf, "rate_10p1", [&]() {
    return run(pSrc, sStep, {W, H}, {0, 0}, d.get(), dRow, {4, 4}, 10.1f, 3, pK,
               NPP_BORDER_REPLICATE);
  });
  only_status(fam + suf, "step_small", [&]() {
    return run(pSrc, sStep, {W, H}, {0, 0}, d.get(), dRow - 1, {4, 4}, 2.0f, 3, pK,
               NPP_BORDER_REPLICATE);
  });
  only_status(fam + suf, "step_unaligned", [&]() {
    return run(pSrc, sStep, {W, H}, {0, 0}, d.get(), dRow + 1, {4, 4}, 2.0f, 3, pK,
               NPP_BORDER_REPLICATE);
  });
  only_status(fam + suf, "srcstep0", [&]() {
    return run(pSrc, 0, {W, H}, {0, 0}, d.get(), dRow, {4, 4}, 2.0f, 3, pK,
               NPP_BORDER_REPLICATE);
  });
  only_status(fam + suf, "badrate_badtaps", [&]() {
    return run(pSrc, sStep, {W, H}, {0, 0}, d.get(), dRow, {4, 4}, 1.0f, 1, pK,
               NPP_BORDER_REPLICATE);
  });
  only_status(fam + suf, "badrate_badborder", [&]() {
    return run(pSrc, sStep, {W, H}, {0, 0}, d.get(), dRow, {4, 4}, 1.0f, 3, pK,
               NPP_BORDER_CONSTANT);
  });
  only_status(fam + suf, "badtaps_badborder", [&]() {
    return run(pSrc, sStep, {W, H}, {0, 0}, d.get(), dRow, {4, 4}, 2.0f, 1, pK,
               NPP_BORDER_CONSTANT);
  });
  only_status(fam + suf, "badborder_badstep", [&]() {
    return run(pSrc, sStep, {W, H}, {0, 0}, d.get(), 1, {4, 4}, 2.0f, 3, pK,
               NPP_BORDER_CONSTANT);
  });
  only_status(fam + suf, "offneg_badrate", [&]() {
    return run(pSrc, sStep, {W, H}, {-1, 0}, d.get(), dRow, {4, 4}, 1.0f, 3, pK,
               NPP_BORDER_REPLICATE);
  });
  only_status(fam + suf, "offneg_badborder", [&]() {
    return run(pSrc, sStep, {W, H}, {-1, 0}, d.get(), dRow, {4, 4}, 2.0f, 3, pK,
               NPP_BORDER_CONSTANT);
  });
  cudaDeviceSynchronize();
}

}  // namespace

TEST(NppiGaussPyramidDiff, Down) {
  down_values<Npp8u, Npp8u>("nppiFilterGaussPyramidLayerDownBorder_8u_C1R", 1,
                            [](const Npp8u *s, int ss, NppiSize sz, NppiPoint o, Npp8u *d, int ds,
                               NppiSize roi, float r, int t, const Npp32f *k, NppiBorderType b,
                               NppStreamContext c) {
                              return nppiFilterGaussPyramidLayerDownBorder_8u_C1R_Ctx(
                                  s, ss, sz, o, d, ds, roi, r, t, k, b, c);
                            },
                            0x51);
  down_values<Npp16u, Npp16u>("nppiFilterGaussPyramidLayerDownBorder_16u_C1R", 1,
                              [](const Npp16u *s, int ss, NppiSize sz, NppiPoint o, Npp16u *d,
                                 int ds, NppiSize roi, float r, int t, const Npp32f *k,
                                 NppiBorderType b, NppStreamContext c) {
                                return nppiFilterGaussPyramidLayerDownBorder_16u_C1R_Ctx(
                                    s, ss, sz, o, d, ds, roi, r, t, k, b, c);
                              },
                              0x52);
  down_values<Npp32f, Npp32f>("nppiFilterGaussPyramidLayerDownBorder_32f_C1R", 1,
                              [](const Npp32f *s, int ss, NppiSize sz, NppiPoint o, Npp32f *d,
                                 int ds, NppiSize roi, float r, int t, const Npp32f *k,
                                 NppiBorderType b, NppStreamContext c) {
                                return nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx(
                                    s, ss, sz, o, d, ds, roi, r, t, k, b, c);
                              },
                              0x53);
  down_values<Npp8u, Npp8u>("nppiFilterGaussPyramidLayerDownBorder_8u_C3R", 3,
                            [](const Npp8u *s, int ss, NppiSize sz, NppiPoint o, Npp8u *d, int ds,
                               NppiSize roi, float r, int t, const Npp32f *k, NppiBorderType b,
                               NppStreamContext c) {
                              return nppiFilterGaussPyramidLayerDownBorder_8u_C3R_Ctx(
                                  s, ss, sz, o, d, ds, roi, r, t, k, b, c);
                            },
                            0x54);
  down_values<Npp16u, Npp16u>("nppiFilterGaussPyramidLayerDownBorder_16u_C3R", 3,
                              [](const Npp16u *s, int ss, NppiSize sz, NppiPoint o, Npp16u *d,
                                 int ds, NppiSize roi, float r, int t, const Npp32f *k,
                                 NppiBorderType b, NppStreamContext c) {
                                return nppiFilterGaussPyramidLayerDownBorder_16u_C3R_Ctx(
                                    s, ss, sz, o, d, ds, roi, r, t, k, b, c);
                              },
                              0x55);
  down_values<Npp32f, Npp32f>("nppiFilterGaussPyramidLayerDownBorder_32f_C3R", 3,
                              [](const Npp32f *s, int ss, NppiSize sz, NppiPoint o, Npp32f *d,
                                 int ds, NppiSize roi, float r, int t, const Npp32f *k,
                                 NppiBorderType b, NppStreamContext c) {
                                return nppiFilterGaussPyramidLayerDownBorder_32f_C3R_Ctx(
                                    s, ss, sz, o, d, ds, roi, r, t, k, b, c);
                              },
                              0x56);
}

TEST(NppiGaussPyramidDiff, DownErrors) {
  down_errors_typed<Npp8u, Npp8u>("nppiFilterGaussPyramidLayerDownBorder_8u_C1R", 1,
                                  [](const Npp8u *s, int ss, NppiSize sz, NppiPoint o, Npp8u *d,
                                     int ds, NppiSize roi, float r, int t, const Npp32f *k,
                                     NppiBorderType b, NppStreamContext c) {
                                    return nppiFilterGaussPyramidLayerDownBorder_8u_C1R_Ctx(
                                        s, ss, sz, o, d, ds, roi, r, t, k, b, c);
                                  },
                                  0x61);
  down_errors_typed<Npp16u, Npp16u>("nppiFilterGaussPyramidLayerDownBorder_16u_C1R", 1,
                                    [](const Npp16u *s, int ss, NppiSize sz, NppiPoint o, Npp16u *d,
                                       int ds, NppiSize roi, float r, int t, const Npp32f *k,
                                       NppiBorderType b, NppStreamContext c) {
                                      return nppiFilterGaussPyramidLayerDownBorder_16u_C1R_Ctx(
                                          s, ss, sz, o, d, ds, roi, r, t, k, b, c);
                                    },
                                    0x62);
  down_errors_typed<Npp32f, Npp32f>("nppiFilterGaussPyramidLayerDownBorder_32f_C1R", 1,
                                    [](const Npp32f *s, int ss, NppiSize sz, NppiPoint o, Npp32f *d,
                                       int ds, NppiSize roi, float r, int t, const Npp32f *k,
                                       NppiBorderType b, NppStreamContext c) {
                                      return nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx(
                                          s, ss, sz, o, d, ds, roi, r, t, k, b, c);
                                    },
                                    0x63);
  down_errors_typed<Npp32f, Npp32f>("nppiFilterGaussPyramidLayerDownBorder_32f_C3R", 3,
                                    [](const Npp32f *s, int ss, NppiSize sz, NppiPoint o, Npp32f *d,
                                       int ds, NppiSize roi, float r, int t, const Npp32f *k,
                                       NppiBorderType b, NppStreamContext c) {
                                      return nppiFilterGaussPyramidLayerDownBorder_32f_C3R_Ctx(
                                          s, ss, sz, o, d, ds, roi, r, t, k, b, c);
                                    },
                                    0x64);
}
