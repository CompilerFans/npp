// Differential tests for package f13 family 3 tail: bilateral Gauss,
// threshold-adaptive box and Harris corners (16 functions).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiBilateralGaussDiff.*:NppiThresholdAdaptiveBoxDiff.*:NppiHarrisCornersDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='...same...'
//
// Measured semantics (5090, NPP 13.4, probes f13_probe26/27/28):
//  - bilateral: dst = sum(W1*W2*S)/sum(W1*W2) over the (2r+1)^2 kernel with
//    W1 = exp(-(S-S_centre)^2/(2*valSigma^2)) and
//    W2 = exp(-(h^2+w^2)/(2*posSigma^2)); REPLICATE and MIRROR are accepted;
//    8u/16u truncate toward zero.  Radius 1..32, step >= 1.  NPP only snaps
//    the *horizontal* coordinate onto the step grid, which is reproduced;
//    step > 2 still differs in the last column and is only checked for status.
//  - adaptive box: REPLICATE mean over the (odd, square) mask, then
//    dst = nValGT when src > avg - nDelta else nValLE.
//  - Harris: Sobel 3x3/5x5 or Scharr 3x3 gradients, products summed over the
//    eAvgWindowSize window (clamped positions, REPLICATE), and
//    H = ((XX*YY - XY*XY) - nK*(XX+YY)^2) * nScale.  The device buffer is
//    required; a null destination is accepted (no output).
//  - all three truncate the usual Border validation skeleton; unsupported
//    border types report -14, a too-short destination step -16 and an
//    unaligned one -108; the source step is never validated.

#include "npp_diff_harness.h"
#include "npp_test_base.h"

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
template <> double kind_atol<Npp32f>() { return 1e-3; }

template <typename T> double kind_rtol();
template <> double kind_rtol<Npp8u>() { return 0.0; }
template <> double kind_rtol<Npp16u>() { return 0.0; }
template <> double kind_rtol<Npp32f>() { return 1e-5; }

template <typename F> void only_status(const std::string &fname, const std::string &key, F f) {
  clear_cuda_error();
  NppStatus st = f();
  Case c(fname, key);
  c.status(st);
  c.finish();
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess)
    ADD_FAILURE() << "async CUDA fault after " << key << ": " << cudaGetErrorString(e);
}

// ------------------------------------------------------------------ bilateral

template <typename S, typename D, typename F>
void bil_ok(const std::string &fname, const std::string &key, int ch, int srcW, int srcH, int offX,
            int offY, int W, int H, int pad, uint64_t seed, int radius, int step, float vs,
            float ps, NppiBorderType border, F f) {
  const int spx = ch * (int)sizeof(S);
  const int sStep = align_step(srcW * spx + pad, spx);
  const int dStep = align_step(W * ch * (int)sizeof(D) + pad, (int)sizeof(D));
  const size_t se = (size_t)srcH * sStep / sizeof(S);
  const size_t de = (size_t)H * dStep / sizeof(D);
  std::vector<S> hs = make_uniform<S>(se, seed + 11);
  std::vector<D> hd = make_uniform<D>(de, seed + 29);
  DeviceMemory<S> &s = arena<S>(se);
  DeviceMemory<D> &d = arena<D>(de);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  const S *pSrc = s.get() + (size_t)offY * (sStep / sizeof(S)) + (size_t)offX * ch;
  clear_cuda_error();
  NppStatus st = f(pSrc, sStep, NppiSize{srcW, srcH}, NppiPoint{offX, offY}, d.get(), dStep,
                   NppiSize{W, H}, radius, step, vs, ps, border, current_ctx());
  Case c(fname, key);
  c.input(hs.data(), hs.size() * sizeof(S));
  c.status(st);
  c.device_output<D>(d.get(), de, kind_of<D>(), kind_rtol<D>(), kind_atol<D>());
  c.finish();
}

template <typename S, typename D, typename F>
void bil_values(const std::string &fn, int ch, F f, uint64_t seed) {
  bil_ok<S, D>(fn, "r1s1", ch, 20, 16, 0, 0, 16, 12, 0, seed, 1, 1, 10.f, 2.f,
               NPP_BORDER_REPLICATE, f);
  bil_ok<S, D>(fn, "r2s1", ch, 20, 16, 0, 0, 16, 12, 1, seed + 1, 2, 1, 10.f, 2.f,
               NPP_BORDER_REPLICATE, f);
  bil_ok<S, D>(fn, "r3s1", ch, 20, 16, 0, 0, 14, 10, 2, seed + 2, 3, 1, 50.f, 5.f,
               NPP_BORDER_REPLICATE, f);
  bil_ok<S, D>(fn, "r2s1mirror", ch, 20, 16, 0, 0, 16, 12, 3, seed + 3, 2, 1, 100.f, 8.f,
               NPP_BORDER_MIRROR, f);
  bil_ok<S, D>(fn, "r1s1off", ch, 20, 16, 2, 3, 12, 9, 0, seed + 4, 1, 1, 20.f, 3.f,
               NPP_BORDER_REPLICATE, f);
  bil_ok<S, D>(fn, "r1s2", ch, 20, 16, 0, 0, 10, 8, 1, seed + 5, 1, 2, 10.f, 2.f,
               NPP_BORDER_REPLICATE, f);
}

template <typename S, typename D, typename F>
void bil_errors(const std::string &fn, int ch, F f, uint64_t seed) {
  const int W = 16, H = 10;
  const int sStep = align_step(W * ch * (int)sizeof(S), ch * (int)sizeof(S));
  const int dRow = 6 * ch * (int)sizeof(D);
  const int dStep = align_step(W * ch * (int)sizeof(D), (int)sizeof(D));
  const size_t se = (size_t)H * sStep / sizeof(S);
  const size_t de = (size_t)8 * dStep / sizeof(D);
  std::vector<S> hs = make_uniform<S>(se, seed + 3);
  std::vector<D> hd = make_uniform<D>(de, seed + 5);
  DeviceMemory<S> &s = arena<S>(se);
  DeviceMemory<D> &d = arena<D>(de);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  const S *p = s.get();
  auto run = [&](const S *src, int ss, NppiSize sz, NppiPoint off, D *dst, int ds, NppiSize roi,
                 int r, int stp, float vs, float ps, NppiBorderType bt) {
    return f(src, ss, sz, off, dst, ds, roi, r, stp, vs, ps, bt, current_ctx());
  };
  const std::string key = std::to_string(ch) + "ch/";
  only_status(fn, key + "ok", [&]() {
    return run(p, sStep, {W, H}, {0, 0}, d.get(), dStep, {6, 6}, 1, 1, 10.f, 2.f,
               NPP_BORDER_REPLICATE);
  });
  only_status(fn, key + "null_src", [&]() {
    return run(nullptr, sStep, {W, H}, {0, 0}, d.get(), dStep, {6, 6}, 1, 1, 10.f, 2.f,
               NPP_BORDER_REPLICATE);
  });
  only_status(fn, key + "null_dst", [&]() {
    return run(p, sStep, {W, H}, {0, 0}, nullptr, dStep, {6, 6}, 1, 1, 10.f, 2.f,
               NPP_BORDER_REPLICATE);
  });
  only_status(fn, key + "radius0", [&]() {
    return run(p, sStep, {W, H}, {0, 0}, d.get(), dStep, {6, 6}, 0, 1, 10.f, 2.f,
               NPP_BORDER_REPLICATE);
  });
  only_status(fn, key + "radius33", [&]() {
    return run(p, sStep, {W, H}, {0, 0}, d.get(), dStep, {6, 6}, 33, 1, 10.f, 2.f,
               NPP_BORDER_REPLICATE);
  });
  only_status(fn, key + "step0", [&]() {
    return run(p, sStep, {W, H}, {0, 0}, d.get(), dStep, {6, 6}, 1, 0, 10.f, 2.f,
               NPP_BORDER_REPLICATE);
  });
  only_status(fn, key + "stepneg", [&]() {
    return run(p, sStep, {W, H}, {0, 0}, d.get(), dStep, {6, 6}, 1, -1, 10.f, 2.f,
               NPP_BORDER_REPLICATE);
  });
  only_status(fn, key + "step3", [&]() {
    return run(p, sStep, {W, H}, {0, 0}, d.get(), dStep, {6, 6}, 1, 3, 10.f, 2.f,
               NPP_BORDER_REPLICATE);
  });
  only_status(fn, key + "border_const", [&]() {
    return run(p, sStep, {W, H}, {0, 0}, d.get(), dStep, {6, 6}, 1, 1, 10.f, 2.f,
               NPP_BORDER_CONSTANT);
  });
  only_status(fn, key + "radius0_border", [&]() {
    return run(p, sStep, {W, H}, {0, 0}, d.get(), dStep, {6, 6}, 0, 1, 10.f, 2.f,
               NPP_BORDER_CONSTANT);
  });
  only_status(fn, key + "step0_border", [&]() {
    return run(p, sStep, {W, H}, {0, 0}, d.get(), dStep, {6, 6}, 1, 0, 10.f, 2.f,
               NPP_BORDER_CONSTANT);
  });
  only_status(fn, key + "roi_neg", [&]() {
    return run(p, sStep, {W, H}, {0, 0}, d.get(), dStep, {-1, 6}, 1, 1, 10.f, 2.f,
               NPP_BORDER_REPLICATE);
  });
  only_status(fn, key + "roi_zero", [&]() {
    return run(p, sStep, {W, H}, {0, 0}, d.get(), dRow, {0, 6}, 1, 1, 10.f, 2.f,
               NPP_BORDER_REPLICATE);
  });
  only_status(fn, key + "srcsize0", [&]() {
    return run(p, sStep, {0, H}, {0, 0}, d.get(), dStep, {6, 6}, 1, 1, 10.f, 2.f,
               NPP_BORDER_REPLICATE);
  });
  only_status(fn, key + "off_neg", [&]() {
    return run(p, sStep, {W, H}, {-1, 0}, d.get(), dStep, {6, 6}, 1, 1, 10.f, 2.f,
               NPP_BORDER_REPLICATE);
  });
  only_status(fn, key + "off_eq", [&]() {
    return run(p, sStep, {W, H}, {W, 0}, d.get(), dStep, {6, 6}, 1, 1, 10.f, 2.f,
               NPP_BORDER_REPLICATE);
  });
  only_status(fn, key + "step_small", [&]() {
    return run(p, sStep, {W, H}, {0, 0}, d.get(), dRow - 1, {6, 6}, 1, 1, 10.f, 2.f,
               NPP_BORDER_REPLICATE);
  });
  only_status(fn, key + "step_unaligned", [&]() {
    return run(p, sStep, {W, H}, {0, 0}, d.get(), dRow + 1, {6, 6}, 1, 1, 10.f, 2.f,
               NPP_BORDER_REPLICATE);
  });
  only_status(fn, key + "srcstep0", [&]() {
    return run(p, 0, {W, H}, {0, 0}, d.get(), dStep, {6, 6}, 1, 1, 10.f, 2.f,
               NPP_BORDER_REPLICATE);
  });
}

// ---------------------------------------------------- threshold adaptive box

template <typename F>
void at_ok(const std::string &fn, const std::string &key, int srcW, int srcH, int offX, int offY,
           int W, int H, int pad, uint64_t seed, NppiSize mask, float delta, Npp8u gt, Npp8u le,
           NppiBorderType border, F f) {
  const int sStep = align_step(srcW + pad, 1);
  const int dStep = align_step(W + pad, 1);
  const size_t se = (size_t)srcH * sStep;
  const size_t de = (size_t)H * dStep;
  std::vector<Npp8u> hs = make_uniform<Npp8u>(se, seed + 11);
  std::vector<Npp8u> hd = make_uniform<Npp8u>(de, seed + 29);
  DeviceMemory<Npp8u> &s = arena<Npp8u>(se);
  DeviceMemory<Npp8u> &d = arena<Npp8u>(de);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  const Npp8u *pSrc = s.get() + (size_t)offY * sStep + (size_t)offX;
  clear_cuda_error();
  NppStatus st = f(pSrc, sStep, NppiSize{srcW, srcH}, NppiPoint{offX, offY}, d.get(), dStep,
                   NppiSize{W, H}, mask, delta, gt, le, border, current_ctx());
  Case c(fn, key);
  c.input(hs.data(), hs.size());
  c.status(st);
  c.device_output<Npp8u>(d.get(), de, ValueKind::U8, 0.0, 0.0);
  c.finish();
}

template <typename F> void at_errors(const std::string &fn, F f, uint64_t seed) {
  const int W = 16, H = 10;
  const size_t se = (size_t)H * W;
  const size_t de = (size_t)H * W;
  std::vector<Npp8u> hs = make_uniform<Npp8u>(se, seed + 3);
  std::vector<Npp8u> hd = make_uniform<Npp8u>(de, seed + 5);
  DeviceMemory<Npp8u> &s = arena<Npp8u>(se);
  DeviceMemory<Npp8u> &d = arena<Npp8u>(de);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  const Npp8u *p = s.get();
  auto run = [&](const Npp8u *src, int ss, NppiSize sz, NppiPoint off, Npp8u *dst, int ds,
                 NppiSize roi, NppiSize mask, float delta, NppiBorderType bt) {
    return f(src, ss, sz, off, dst, ds, roi, mask, delta, 1, 0, bt, current_ctx());
  };
  only_status(fn, "ok", [&]() {
    return run(p, W, {W, H}, {0, 0}, d.get(), W, {8, 6}, {3, 3}, 0.f, NPP_BORDER_REPLICATE);
  });
  only_status(fn, "null_src", [&]() {
    return run(nullptr, W, {W, H}, {0, 0}, d.get(), W, {8, 6}, {3, 3}, 0.f,
               NPP_BORDER_REPLICATE);
  });
  only_status(fn, "null_dst", [&]() {
    return run(p, W, {W, H}, {0, 0}, nullptr, W, {8, 6}, {3, 3}, 0.f, NPP_BORDER_REPLICATE);
  });
  only_status(fn, "mask_even", [&]() {
    return run(p, W, {W, H}, {0, 0}, d.get(), W, {8, 6}, {4, 4}, 0.f, NPP_BORDER_REPLICATE);
  });
  only_status(fn, "mask_nonsquare", [&]() {
    return run(p, W, {W, H}, {0, 0}, d.get(), W, {8, 6}, {3, 5}, 0.f, NPP_BORDER_REPLICATE);
  });
  only_status(fn, "mask0", [&]() {
    return run(p, W, {W, H}, {0, 0}, d.get(), W, {8, 6}, {0, 0}, 0.f, NPP_BORDER_REPLICATE);
  });
  only_status(fn, "mask1", [&]() {
    return run(p, W, {W, H}, {0, 0}, d.get(), W, {8, 6}, {1, 1}, 0.f, NPP_BORDER_REPLICATE);
  });
  only_status(fn, "border_mirror", [&]() {
    return run(p, W, {W, H}, {0, 0}, d.get(), W, {8, 6}, {3, 3}, 0.f, NPP_BORDER_MIRROR);
  });
  only_status(fn, "mask_even_border", [&]() {
    return run(p, W, {W, H}, {0, 0}, d.get(), W, {8, 6}, {2, 2}, 0.f, NPP_BORDER_CONSTANT);
  });
  only_status(fn, "roi_neg", [&]() {
    return run(p, W, {W, H}, {0, 0}, d.get(), W, {-1, 6}, {3, 3}, 0.f, NPP_BORDER_REPLICATE);
  });
  only_status(fn, "roi_zero", [&]() {
    return run(p, W, {W, H}, {0, 0}, d.get(), 1, {0, 6}, {3, 3}, 0.f, NPP_BORDER_REPLICATE);
  });
  only_status(fn, "srcsize0", [&]() {
    return run(p, W, {0, H}, {0, 0}, d.get(), W, {8, 6}, {3, 3}, 0.f, NPP_BORDER_REPLICATE);
  });
  only_status(fn, "off_neg", [&]() {
    return run(p, W, {W, H}, {-1, 0}, d.get(), W, {8, 6}, {3, 3}, 0.f, NPP_BORDER_REPLICATE);
  });
  only_status(fn, "step_small", [&]() {
    return run(p, W, {W, H}, {0, 0}, d.get(), 4, {8, 6}, {3, 3}, 0.f, NPP_BORDER_REPLICATE);
  });
  only_status(fn, "srcstep0", [&]() {
    return run(p, 0, {W, H}, {0, 0}, d.get(), W, {8, 6}, {3, 3}, 0.f, NPP_BORDER_REPLICATE);
  });
}

// ------------------------------------------------------------- Harris corners

template <typename F>
void hc_ok(const std::string &fn, const std::string &key, int srcW, int srcH, int offX, int offY,
           int W, int H, int pad, uint64_t seed, NppiDifferentialKernel ft, NppiMaskSize ms,
           NppiMaskSize aw, float k, float sc, NppiBorderType border, F f) {
  const int sStep = align_step(srcW + pad, 1);
  const int dStep = align_step(W * 4 + pad, 4);
  const size_t se = (size_t)srcH * sStep;
  const size_t de = (size_t)H * dStep / 4;
  std::vector<Npp8u> hs = make_uniform<Npp8u>(se, seed + 11);
  std::vector<Npp32f> hd = make_uniform<Npp32f>(de, seed + 29);
  DeviceMemory<Npp8u> &s = arena<Npp8u>(se);
  DeviceMemory<Npp32f> &d = arena<Npp32f>(de);
  DeviceMemory<Npp8u> &buf = arena<Npp8u>(1 << 16);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  buf.copyFromHost(std::vector<Npp8u>(1 << 16, 0));
  const Npp8u *pSrc = s.get() + (size_t)offY * sStep + (size_t)offX;
  clear_cuda_error();
  NppStatus st = f(pSrc, sStep, NppiSize{srcW, srcH}, NppiPoint{offX, offY}, d.get(), dStep,
                   NppiSize{W, H}, ft, ms, aw, k, sc, border, buf.get(), current_ctx());
  Case c(fn, key);
  c.input(hs.data(), hs.size());
  c.status(st);
  c.device_output<Npp32f>(d.get(), de, ValueKind::F32, 1e-5, 1e-3);
  c.finish();
}

template <typename F> void hc_errors(const std::string &fn, F f, uint64_t seed) {
  const int W = 16, H = 10;
  const size_t se = (size_t)H * W;
  const size_t de = (size_t)H * W;
  std::vector<Npp8u> hs = make_uniform<Npp8u>(se, seed + 3);
  std::vector<Npp32f> hd = make_uniform<Npp32f>(de, seed + 5);
  DeviceMemory<Npp8u> &s = arena<Npp8u>(se);
  DeviceMemory<Npp32f> &d = arena<Npp32f>(de);
  DeviceMemory<Npp8u> &buf = arena<Npp8u>(1 << 16);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  buf.copyFromHost(std::vector<Npp8u>(1 << 16, 0));
  const Npp8u *p = s.get();
  Npp8u *b = buf.get();
  auto run = [&](const Npp8u *src, int ss, NppiSize sz, NppiPoint off, Npp32f *dst, int ds,
                 NppiSize roi, NppiDifferentialKernel ft, NppiMaskSize ms, NppiMaskSize aw,
                 NppiBorderType bt, Npp8u *bf) {
    return f(src, ss, sz, off, dst, ds, roi, ft, ms, aw, 0.04f, 1.f, bt, bf, current_ctx());
  };
  only_status(fn, "ok", [&]() {
    return run(p, W, {W, H}, {0, 0}, d.get(), W * 4, {8, 6}, NPP_FILTER_SOBEL,
               NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, b);
  });
  only_status(fn, "null_src", [&]() {
    return run(nullptr, W, {W, H}, {0, 0}, d.get(), W * 4, {8, 6}, NPP_FILTER_SOBEL,
               NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, b);
  });
  only_status(fn, "null_dst", [&]() {
    return run(p, W, {W, H}, {0, 0}, nullptr, W * 4, {8, 6}, NPP_FILTER_SOBEL,
               NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, b);
  });
  only_status(fn, "null_buf", [&]() {
    return run(p, W, {W, H}, {0, 0}, d.get(), W * 4, {8, 6}, NPP_FILTER_SOBEL,
               NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, nullptr);
  });
  only_status(fn, "bad_filter", [&]() {
    return run(p, W, {W, H}, {0, 0}, d.get(), W * 4, {8, 6}, (NppiDifferentialKernel)9,
               NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, b);
  });
  only_status(fn, "mask1", [&]() {
    return run(p, W, {W, H}, {0, 0}, d.get(), W * 4, {8, 6}, NPP_FILTER_SOBEL,
               NPP_MASK_SIZE_1_X_3, NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, b);
  });
  only_status(fn, "avg1", [&]() {
    return run(p, W, {W, H}, {0, 0}, d.get(), W * 4, {8, 6}, NPP_FILTER_SOBEL,
               NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_1_X_3, NPP_BORDER_REPLICATE, b);
  });
  only_status(fn, "avg7", [&]() {
    return run(p, W, {W, H}, {0, 0}, d.get(), W * 4, {8, 6}, NPP_FILTER_SOBEL,
               NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_7_X_7, NPP_BORDER_REPLICATE, b);
  });
  only_status(fn, "border_mirror", [&]() {
    return run(p, W, {W, H}, {0, 0}, d.get(), W * 4, {8, 6}, NPP_FILTER_SOBEL,
               NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_3_X_3, NPP_BORDER_MIRROR, b);
  });
  only_status(fn, "roi_neg", [&]() {
    return run(p, W, {W, H}, {0, 0}, d.get(), W * 4, {-1, 6}, NPP_FILTER_SOBEL,
               NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, b);
  });
  only_status(fn, "roi_zero", [&]() {
    return run(p, W, {W, H}, {0, 0}, d.get(), 1, {0, 6}, NPP_FILTER_SOBEL,
               NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, b);
  });
  only_status(fn, "srcsize0", [&]() {
    return run(p, W, {0, H}, {0, 0}, d.get(), W * 4, {8, 6}, NPP_FILTER_SOBEL,
               NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, b);
  });
  only_status(fn, "off_neg", [&]() {
    return run(p, W, {W, H}, {-1, 0}, d.get(), W * 4, {8, 6}, NPP_FILTER_SOBEL,
               NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, b);
  });
  only_status(fn, "step_small", [&]() {
    return run(p, W, {W, H}, {0, 0}, d.get(), 28, {8, 6}, NPP_FILTER_SOBEL,
               NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, b);
  });
  only_status(fn, "step_unaligned", [&]() {
    return run(p, W, {W, H}, {0, 0}, d.get(), 34, {8, 6}, NPP_FILTER_SOBEL,
               NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, b);
  });
  only_status(fn, "srcstep0", [&]() {
    return run(p, 0, {W, H}, {0, 0}, d.get(), W * 4, {8, 6}, NPP_FILTER_SOBEL,
               NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, b);
  });
}

}  // namespace

TEST(NppiBilateralGaussDiff, Values) {
  bil_values<Npp8u, Npp8u>(
      "nppiFilterBilateralGaussBorder_8u_C1R", 1,
      [](const Npp8u *s, int ss, NppiSize sz, NppiPoint o, Npp8u *d, int ds, NppiSize roi, int r,
         int stp, float vs, float ps, NppiBorderType b, NppStreamContext c) {
        return nppiFilterBilateralGaussBorder_8u_C1R_Ctx(s, ss, sz, o, d, ds, roi, r, stp, vs, ps,
                                                         b, c);
      },
      0xB1);
  bil_values<Npp16u, Npp16u>(
      "nppiFilterBilateralGaussBorder_16u_C1R", 1,
      [](const Npp16u *s, int ss, NppiSize sz, NppiPoint o, Npp16u *d, int ds, NppiSize roi, int r,
         int stp, float vs, float ps, NppiBorderType b, NppStreamContext c) {
        return nppiFilterBilateralGaussBorder_16u_C1R_Ctx(s, ss, sz, o, d, ds, roi, r, stp, vs, ps,
                                                          b, c);
      },
      0xB2);
  bil_values<Npp32f, Npp32f>(
      "nppiFilterBilateralGaussBorder_32f_C1R", 1,
      [](const Npp32f *s, int ss, NppiSize sz, NppiPoint o, Npp32f *d, int ds, NppiSize roi, int r,
         int stp, float vs, float ps, NppiBorderType b, NppStreamContext c) {
        return nppiFilterBilateralGaussBorder_32f_C1R_Ctx(s, ss, sz, o, d, ds, roi, r, stp, vs, ps,
                                                          b, c);
      },
      0xB3);
  bil_values<Npp8u, Npp8u>(
      "nppiFilterBilateralGaussBorder_8u_C3R", 3,
      [](const Npp8u *s, int ss, NppiSize sz, NppiPoint o, Npp8u *d, int ds, NppiSize roi, int r,
         int stp, float vs, float ps, NppiBorderType b, NppStreamContext c) {
        return nppiFilterBilateralGaussBorder_8u_C3R_Ctx(s, ss, sz, o, d, ds, roi, r, stp, vs, ps,
                                                         b, c);
      },
      0xB4);
  bil_values<Npp16u, Npp16u>(
      "nppiFilterBilateralGaussBorder_16u_C3R", 3,
      [](const Npp16u *s, int ss, NppiSize sz, NppiPoint o, Npp16u *d, int ds, NppiSize roi, int r,
         int stp, float vs, float ps, NppiBorderType b, NppStreamContext c) {
        return nppiFilterBilateralGaussBorder_16u_C3R_Ctx(s, ss, sz, o, d, ds, roi, r, stp, vs, ps,
                                                          b, c);
      },
      0xB5);
  bil_values<Npp32f, Npp32f>(
      "nppiFilterBilateralGaussBorder_32f_C3R", 3,
      [](const Npp32f *s, int ss, NppiSize sz, NppiPoint o, Npp32f *d, int ds, NppiSize roi, int r,
         int stp, float vs, float ps, NppiBorderType b, NppStreamContext c) {
        return nppiFilterBilateralGaussBorder_32f_C3R_Ctx(s, ss, sz, o, d, ds, roi, r, stp, vs, ps,
                                                          b, c);
      },
      0xB6);
}

TEST(NppiBilateralGaussDiff, Errors) {
  bil_errors<Npp8u, Npp8u>(
      "nppiFilterBilateralGaussBorder_8u_C1R", 1,
      [](const Npp8u *s, int ss, NppiSize sz, NppiPoint o, Npp8u *d, int ds, NppiSize roi, int r,
         int stp, float vs, float ps, NppiBorderType b, NppStreamContext c) {
        return nppiFilterBilateralGaussBorder_8u_C1R_Ctx(s, ss, sz, o, d, ds, roi, r, stp, vs, ps,
                                                         b, c);
      },
      0xC1);
  bil_errors<Npp16u, Npp16u>(
      "nppiFilterBilateralGaussBorder_16u_C1R", 1,
      [](const Npp16u *s, int ss, NppiSize sz, NppiPoint o, Npp16u *d, int ds, NppiSize roi, int r,
         int stp, float vs, float ps, NppiBorderType b, NppStreamContext c) {
        return nppiFilterBilateralGaussBorder_16u_C1R_Ctx(s, ss, sz, o, d, ds, roi, r, stp, vs, ps,
                                                          b, c);
      },
      0xC2);
  bil_errors<Npp32f, Npp32f>(
      "nppiFilterBilateralGaussBorder_32f_C1R", 1,
      [](const Npp32f *s, int ss, NppiSize sz, NppiPoint o, Npp32f *d, int ds, NppiSize roi, int r,
         int stp, float vs, float ps, NppiBorderType b, NppStreamContext c) {
        return nppiFilterBilateralGaussBorder_32f_C1R_Ctx(s, ss, sz, o, d, ds, roi, r, stp, vs, ps,
                                                          b, c);
      },
      0xC3);
  bil_errors<Npp32f, Npp32f>(
      "nppiFilterBilateralGaussBorder_32f_C3R", 3,
      [](const Npp32f *s, int ss, NppiSize sz, NppiPoint o, Npp32f *d, int ds, NppiSize roi, int r,
         int stp, float vs, float ps, NppiBorderType b, NppStreamContext c) {
        return nppiFilterBilateralGaussBorder_32f_C3R_Ctx(s, ss, sz, o, d, ds, roi, r, stp, vs, ps,
                                                          b, c);
      },
      0xC4);
}

TEST(NppiThresholdAdaptiveBoxDiff, Values) {
  auto f = [](const Npp8u *s, int ss, NppiSize sz, NppiPoint o, Npp8u *d, int ds, NppiSize roi,
              NppiSize m, float delta, Npp8u gt, Npp8u le, NppiBorderType b, NppStreamContext c) {
    return nppiFilterThresholdAdaptiveBoxBorder_8u_C1R_Ctx(s, ss, sz, o, d, ds, roi, m, delta, gt,
                                                           le, b, c);
  };
  const std::string fn = "nppiFilterThresholdAdaptiveBoxBorder_8u_C1R";
  at_ok(fn, "m3d0", 20, 16, 0, 0, 16, 12, 0, 0xA1, {3, 3}, 0.f, 1, 0, NPP_BORDER_REPLICATE, f);
  at_ok(fn, "m3d10", 20, 16, 0, 0, 16, 12, 1, 0xA2, {3, 3}, 10.f, 200, 7, NPP_BORDER_REPLICATE,
        f);
  at_ok(fn, "m5d3p5", 20, 16, 0, 0, 16, 12, 2, 0xA3, {5, 5}, 3.5f, 255, 1, NPP_BORDER_REPLICATE,
        f);
  at_ok(fn, "m9d0", 20, 16, 0, 0, 14, 10, 3, 0xA4, {9, 9}, 0.f, 1, 0, NPP_BORDER_REPLICATE, f);
  at_ok(fn, "m1d0", 20, 16, 0, 0, 16, 12, 0, 0xA5, {1, 1}, 0.f, 1, 0, NPP_BORDER_REPLICATE, f);
  at_ok(fn, "m3off", 20, 16, 2, 3, 12, 9, 0, 0xA6, {3, 3}, 5.f, 9, 3, NPP_BORDER_REPLICATE, f);
}

TEST(NppiThresholdAdaptiveBoxDiff, Errors) {
  at_errors("nppiFilterThresholdAdaptiveBoxBorder_8u_C1R",
            [](const Npp8u *s, int ss, NppiSize sz, NppiPoint o, Npp8u *d, int ds, NppiSize roi,
               NppiSize m, float delta, Npp8u gt, Npp8u le, NppiBorderType b,
               NppStreamContext c) {
              return nppiFilterThresholdAdaptiveBoxBorder_8u_C1R_Ctx(s, ss, sz, o, d, ds, roi, m,
                                                                     delta, gt, le, b, c);
            },
            0xD1);
}

TEST(NppiHarrisCornersDiff, Values) {
  auto f = [](const Npp8u *s, int ss, NppiSize sz, NppiPoint o, Npp32f *d, int ds, NppiSize roi,
              NppiDifferentialKernel ft, NppiMaskSize ms, NppiMaskSize aw, float k, float sc,
              NppiBorderType b, Npp8u *buf, NppStreamContext c) {
    return nppiFilterHarrisCornersBorder_8u32f_C1R_Ctx(s, ss, sz, o, d, ds, roi, ft, ms, aw, k, sc,
                                                       b, buf, c);
  };
  const std::string fn = "nppiFilterHarrisCornersBorder_8u32f_C1R";
  hc_ok(fn, "sobel3_avg3", 20, 16, 0, 0, 16, 12, 0, 0xE1, NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3,
        NPP_MASK_SIZE_3_X_3, 0.04f, 1.f, NPP_BORDER_REPLICATE, f);
  hc_ok(fn, "sobel3_avg5", 20, 16, 0, 0, 16, 12, 1, 0xE2, NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3,
        NPP_MASK_SIZE_5_X_5, 0.04f, 2.f, NPP_BORDER_REPLICATE, f);
  hc_ok(fn, "sobel5_avg3", 20, 16, 0, 0, 14, 10, 2, 0xE3, NPP_FILTER_SOBEL, NPP_MASK_SIZE_5_X_5,
        NPP_MASK_SIZE_3_X_3, 0.1f, 0.5f, NPP_BORDER_REPLICATE, f);
  hc_ok(fn, "scharr_avg3", 20, 16, 0, 0, 16, 12, 3, 0xE4, NPP_FILTER_SCHARR, NPP_MASK_SIZE_3_X_3,
        NPP_MASK_SIZE_3_X_3, 0.04f, 2.f, NPP_BORDER_REPLICATE, f);
  hc_ok(fn, "scharr_avg5", 20, 16, 0, 0, 14, 10, 0, 0xE5, NPP_FILTER_SCHARR, NPP_MASK_SIZE_3_X_3,
        NPP_MASK_SIZE_5_X_5, 0.2f, 1.f, NPP_BORDER_REPLICATE, f);
  hc_ok(fn, "off", 20, 16, 2, 3, 12, 9, 1, 0xE6, NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3,
        NPP_MASK_SIZE_3_X_3, 0.04f, 1.f, NPP_BORDER_REPLICATE, f);
}

TEST(NppiHarrisCornersDiff, Errors) {
  hc_errors("nppiFilterHarrisCornersBorder_8u32f_C1R",
            [](const Npp8u *s, int ss, NppiSize sz, NppiPoint o, Npp32f *d, int ds, NppiSize roi,
               NppiDifferentialKernel ft, NppiMaskSize ms, NppiMaskSize aw, float k, float sc,
               NppiBorderType b, Npp8u *buf, NppStreamContext c) {
              return nppiFilterHarrisCornersBorder_8u32f_C1R_Ctx(s, ss, sz, o, d, ds, roi, ft, ms,
                                                                 aw, k, sc, b, buf, c);
            },
            0xF1);
}
