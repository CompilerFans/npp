// Differential tests for package m1: morphological operations
// (nppiDilate / nppiErode / *3x3 / *Border variants, Morph composite ops,
// Gray dilate/erode, MorphGetBufferSize; 16 families / 292 functions).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiMorphDiff.*'
// Risky validation probes (null mask / null scratch buffer / degenerate mask
// geometry) live in a separate fixture so a crash cannot lose the main goldens:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiMorphProbeDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records ./diff_tests \
//     --gtest_filter='NppiMorphDiff.*:NppiMorphProbeDiff.*'
//
// Case design notes:
//  - plain variants receive a source buffer that is larger than the ROI by the
//    full mask reach on every side (pSrc inset), because NVIDIA documents that
//    avoiding \ref sampling_beyond_image_boundaries is the caller's job; the
//    margin data is random so clamped vs raw reads are distinguishable
//  - border variants place pSrc at an offset inside a srcW x srcH image and
//    use NPP_BORDER_REPLICATE (other border types are probed as errors)
//  - every OK case hashes source, destination-initial and mask contents, and
//    captures the whole destination buffer (row padding and, for AC4, the
//    untouched alpha channel are part of the golden)
//  - composite ops size their scratch buffer through the matching
//    nppiMorphGetBufferSize_* call, which is recorded as its own golden
//  - seeds are derived from the case id, so non-Ctx adapter cases can reuse a
//    _Ctx golden by replaying the same id

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
template <> ValueKind kind_of<Npp16s>() { return ValueKind::I16; }
template <> ValueKind kind_of<Npp32s>() { return ValueKind::I32; }
template <> ValueKind kind_of<Npp32f>() { return ValueKind::F32; }
template <> ValueKind kind_of<Npp64f>() { return ValueKind::F64; }

// Deterministic seed for a case id (stable across record/verify builds).
uint64_t seed_of(const std::string &key) {
  uint64_t h = 1469598103934665603ULL;
  for (char c : key) {
    h ^= (unsigned char)c;
    h *= 1099511628211ULL;
  }
  return h & 0x7fffffffULL;
}

// Device copies of the structuring elements used across the suite.
const std::vector<Npp8u> &maskBox3() {
  static const std::vector<Npp8u> m{1, 1, 1, 1, 1, 1, 1, 1, 1};
  return m;
}
const std::vector<Npp8u> &maskCross3() {
  static const std::vector<Npp8u> m{0, 1, 0, 1, 1, 1, 0, 1, 0};
  return m;
}
const std::vector<Npp8u> &maskZero3() {
  static const std::vector<Npp8u> m{0, 0, 0, 0, 0, 0, 0, 0, 0};
  return m;
}
const std::vector<Npp8u> &maskOne1() {
  static const std::vector<Npp8u> m{1};
  return m;
}
const std::vector<Npp8u> &mask5x3() {
  // 5 wide x 3 high, anchor (1, 1); asymmetric taps
  static const std::vector<Npp8u> m{1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1};
  return m;
}
const std::vector<Npp32s> &maskGray8u() {
  static const std::vector<Npp32s> m{0, -5, 200, 1, -1, 3, 0, 2, -3};
  return m;
}
const std::vector<Npp32f> &maskGray32f() {
  static const std::vector<Npp32f> m{0.0f, -2.0f, 1.5f, 2.0f, -1.0f, 0.5f, 1.0f, -0.5f, 3.0f};
  return m;
}

template <typename M> const M *dev_mask(const std::vector<M> &h) {
  DeviceMemory<M> &m = arena<M>(h.size());
  m.copyFromHost(h);
  return m.get();
}

// Zero-initialized device buffer (validation probes that may actually launch).
template <typename T> T *zero_arena(std::size_t n) {
  DeviceMemory<T> &d = arena<T>(n);
  std::vector<T> z(n, T(0));
  d.copyFromHost(z);
  return d.get();
}

// ---------------------------------------------------------------- ok helpers

// Plain masked morphology: source buffer is (W+ml+mr) x (H+mt+mb) with the
// margins derived from the mask reach; pSrc inset by (ml, mt).
template <typename T, typename M, typename F>
void morph_plain_ok(const std::string &fname, const std::string &key, int channels, int W, int H,
                    const std::vector<M> &mask, NppiSize maskSize, NppiPoint anchor, int pad, F f) {
  const int ml = anchor.x, mr = maskSize.width - 1 - anchor.x;
  const int mt = anchor.y, mb = maskSize.height - 1 - anchor.y;
  const int spx = channels * (int)sizeof(T);
  const int bw = W + ml + mr, bh = H + mt + mb;
  const int sStep = align_step(bw * spx, spx);
  const int dpx = channels * (int)sizeof(T);
  const int dAlign = channels == 4 ? dpx : (int)sizeof(T);
  const int dStep = align_step(W * dpx + pad, dAlign);
  const size_t se = (size_t)bh * sStep / sizeof(T);
  const size_t de = (size_t)H * dStep / sizeof(T);
  const uint64_t seed = seed_of(key);
  std::vector<T> hs = make_uniform<T>(se, seed + 11);
  std::vector<T> hd = make_uniform<T>(de, seed + 29);
  DeviceMemory<T> &s = arena<T>(se);
  DeviceMemory<T> &d = arena<T>(de);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  const M *pMask = dev_mask(mask);
  const T *pSrc = s.get() + (size_t)mt * (sStep / sizeof(T)) + (size_t)ml * channels;
  clear_cuda_error();
  NppStatus st = f(pSrc, sStep, d.get(), dStep, NppiSize{W, H}, pMask, maskSize, anchor, current_ctx());
  Case c(key, fname);
  c.input(hs.data(), hs.size() * sizeof(T));
  c.input(hd.data(), hd.size() * sizeof(T));
  c.input(mask.data(), mask.size() * sizeof(M));
  c.status(st);
  c.device_output<T>(d.get(), de, kind_of<T>(), 1e-6, 1e-6);
  c.finish();
}

// Plain 3x3 morphology (implicit full 3x3 structuring element, margin 1).
template <typename T, typename F>
void morph3x3_plain_ok(const std::string &fname, const std::string &key, int channels, int W, int H, int pad, F f) {
  const int spx = channels * (int)sizeof(T);
  const int bw = W + 2, bh = H + 2;
  const int sStep = align_step(bw * spx, spx);
  const int dpx = channels * (int)sizeof(T);
  const int dAlign = channels == 4 ? dpx : (int)sizeof(T);
  const int dStep = align_step(W * dpx + pad, dAlign);
  const size_t se = (size_t)bh * sStep / sizeof(T);
  const size_t de = (size_t)H * dStep / sizeof(T);
  const uint64_t seed = seed_of(key);
  std::vector<T> hs = make_uniform<T>(se, seed + 11);
  std::vector<T> hd = make_uniform<T>(de, seed + 29);
  DeviceMemory<T> &s = arena<T>(se);
  DeviceMemory<T> &d = arena<T>(de);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  const T *pSrc = s.get() + (size_t)(sStep / sizeof(T)) + (size_t)channels;
  clear_cuda_error();
  NppStatus st = f(pSrc, sStep, d.get(), dStep, NppiSize{W, H}, current_ctx());
  Case c(key, fname);
  c.input(hs.data(), hs.size() * sizeof(T));
  c.input(hd.data(), hd.size() * sizeof(T));
  c.status(st);
  c.device_output<T>(d.get(), de, kind_of<T>(), 1e-6, 1e-6);
  c.finish();
}

// Border morphology: image srcW x srcH, pSrc = image + (offX, offY) pixels.
template <typename T, typename M, typename F>
void morph_border_ok(const std::string &fname, const std::string &key, int channels, int srcW, int srcH, int offX,
                     int offY, int W, int H, const std::vector<M> &mask, NppiSize maskSize, NppiPoint anchor, int pad,
                     F f) {
  const int spx = channels * (int)sizeof(T);
  const int sStep = align_step(srcW * spx + pad, spx);
  const int dpx = channels * (int)sizeof(T);
  const int dAlign = channels == 4 ? dpx : (int)sizeof(T);
  const int dStep = align_step(W * dpx + pad, dAlign);
  const size_t se = (size_t)srcH * sStep / sizeof(T);
  const size_t de = (size_t)H * dStep / sizeof(T);
  const uint64_t seed = seed_of(key);
  std::vector<T> hs = make_uniform<T>(se, seed + 11);
  std::vector<T> hd = make_uniform<T>(de, seed + 29);
  DeviceMemory<T> &s = arena<T>(se);
  DeviceMemory<T> &d = arena<T>(de);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  const M *pMask = dev_mask(mask);
  const T *pSrc = s.get() + (size_t)offY * (sStep / sizeof(T)) + (size_t)offX * channels;
  clear_cuda_error();
  NppStatus st = f(pSrc, sStep, NppiSize{srcW, srcH}, NppiPoint{offX, offY}, d.get(), dStep, NppiSize{W, H}, pMask,
                   maskSize, anchor, NPP_BORDER_REPLICATE, current_ctx());
  Case c(key, fname);
  c.input(hs.data(), hs.size() * sizeof(T));
  c.input(hd.data(), hd.size() * sizeof(T));
  c.input(mask.data(), mask.size() * sizeof(M));
  c.status(st);
  c.device_output<T>(d.get(), de, kind_of<T>(), 1e-6, 1e-6);
  c.finish();
}

// Border 3x3 morphology (no mask/anchor parameters).
template <typename T, typename F>
void morph3x3_border_ok(const std::string &fname, const std::string &key, int channels, int srcW, int srcH, int offX,
                        int offY, int W, int H, int pad, F f) {
  const int spx = channels * (int)sizeof(T);
  const int sStep = align_step(srcW * spx + pad, spx);
  const int dpx = channels * (int)sizeof(T);
  const int dAlign = channels == 4 ? dpx : (int)sizeof(T);
  const int dStep = align_step(W * dpx + pad, dAlign);
  const size_t se = (size_t)srcH * sStep / sizeof(T);
  const size_t de = (size_t)H * dStep / sizeof(T);
  const uint64_t seed = seed_of(key);
  std::vector<T> hs = make_uniform<T>(se, seed + 11);
  std::vector<T> hd = make_uniform<T>(de, seed + 29);
  DeviceMemory<T> &s = arena<T>(se);
  DeviceMemory<T> &d = arena<T>(de);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  const T *pSrc = s.get() + (size_t)offY * (sStep / sizeof(T)) + (size_t)offX * channels;
  clear_cuda_error();
  NppStatus st = f(pSrc, sStep, NppiSize{srcW, srcH}, NppiPoint{offX, offY}, d.get(), dStep, NppiSize{W, H},
                   NPP_BORDER_REPLICATE, current_ctx());
  Case c(key, fname);
  c.input(hs.data(), hs.size() * sizeof(T));
  c.input(hd.data(), hd.size() * sizeof(T));
  c.status(st);
  c.device_output<T>(d.get(), de, kind_of<T>(), 1e-6, 1e-6);
  c.finish();
}

// Composite morphology (Open/Close/Gradient/TopHat/BlackHat): the scratch
// buffer is sized through `gbs` and its size is recorded as its own golden.
template <typename T, typename F, typename G>
void morph_composite_ok(const std::string &fname, const std::string &key, int channels, int srcW, int srcH, int offX,
                        int offY, int W, int H, const std::vector<Npp8u> &mask, NppiSize maskSize, NppiPoint anchor,
                        int pad, F f, G gbs) {
  int nbytes = 0;
  clear_cuda_error();
  NppStatus gs = gbs(NppiSize{W, H}, &nbytes);
  {
    Case c("nppiMorphGetBufferSize/bufsize/" + key, fname);
    c.status(gs);
    c.host_output<int>(&nbytes, 1, ValueKind::I32);
    c.finish();
  }
  ASSERT_GT(nbytes, 0) << key << ": buffer size must be positive";
  const int spx = channels * (int)sizeof(T);
  const int sStep = align_step(srcW * spx + pad, spx);
  const int dpx = channels * (int)sizeof(T);
  const int dAlign = channels == 4 ? dpx : (int)sizeof(T);
  const int dStep = align_step(W * dpx + pad, dAlign);
  const size_t se = (size_t)srcH * sStep / sizeof(T);
  const size_t de = (size_t)H * dStep / sizeof(T);
  const uint64_t seed = seed_of(key);
  std::vector<T> hs = make_uniform<T>(se, seed + 11);
  std::vector<T> hd = make_uniform<T>(de, seed + 29);
  DeviceMemory<T> &s = arena<T>(se);
  DeviceMemory<T> &d = arena<T>(de);
  DeviceMemory<Npp8u> &buf = arena<Npp8u>((size_t)nbytes);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  std::vector<Npp8u> zb((size_t)nbytes, 0);
  buf.copyFromHost(zb);
  const Npp8u *pMask = dev_mask(mask);
  const T *pSrc = s.get() + (size_t)offY * (sStep / sizeof(T)) + (size_t)offX * channels;
  clear_cuda_error();
  NppStatus st = f(pSrc, sStep, NppiSize{srcW, srcH}, NppiPoint{offX, offY}, d.get(), dStep, NppiSize{W, H}, pMask,
                   maskSize, anchor, buf.get(), NPP_BORDER_REPLICATE, current_ctx());
  Case c(key, fname);
  c.input(hs.data(), hs.size() * sizeof(T));
  c.input(hd.data(), hd.size() * sizeof(T));
  c.input(mask.data(), mask.size() * sizeof(Npp8u));
  c.status(st);
  c.device_output<T>(d.get(), de, kind_of<T>(), 1e-6, 1e-6);
  c.finish();
}

// Gray dilate/erode: signed mask of type M (Npp32s for 8u, Npp32f for 32f).
template <typename T, typename M, typename F>
void gray_ok(const std::string &fname, const std::string &key, int srcW, int srcH, int offX, int offY, int W, int H,
             const std::vector<M> &mask, NppiSize maskSize, NppiPoint anchor, int pad, F f) {
  const int spx = (int)sizeof(T);
  const int sStep = align_step(srcW * spx + pad, spx);
  const int dStep = align_step(W * spx + pad, spx);
  const size_t se = (size_t)srcH * sStep / sizeof(T);
  const size_t de = (size_t)H * dStep / sizeof(T);
  const uint64_t seed = seed_of(key);
  std::vector<T> hs = make_uniform<T>(se, seed + 11);
  std::vector<T> hd = make_uniform<T>(de, seed + 29);
  DeviceMemory<T> &s = arena<T>(se);
  DeviceMemory<T> &d = arena<T>(de);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  const M *pMask = dev_mask(mask);
  const T *pSrc = s.get() + (size_t)offY * (sStep / sizeof(T)) + (size_t)offX;
  clear_cuda_error();
  NppStatus st = f(pSrc, sStep, NppiSize{srcW, srcH}, NppiPoint{offX, offY}, d.get(), dStep, NppiSize{W, H}, pMask,
                   maskSize, anchor, NPP_BORDER_REPLICATE, current_ctx());
  Case c(key, fname);
  c.input(hs.data(), hs.size() * sizeof(T));
  c.input(hd.data(), hd.size() * sizeof(T));
  c.input(mask.data(), mask.size() * sizeof(M));
  c.status(st);
  c.device_output<T>(d.get(), de, kind_of<T>(), 1e-6, 1e-6);
  c.finish();
}

// Buffer-size helper golden (host output, exact).
template <typename F> void gbs_case(const std::string &fname, const std::string &key, NppiSize roi, F f) {
  int size = 0;
  clear_cuda_error();
  NppStatus st = f(roi, &size);
  Case c(key, fname);
  c.status(st);
  c.host_output<int>(&size, 1, ValueKind::I32);
  c.finish();
}

// Status-only case: the functor fully binds the call.
template <typename F> void only_status(const std::string &fname, const std::string &key, F f) {
  clear_cuda_error();
  NppStatus st = f();
  Case c(key, fname);
  c.status(st);
  c.finish();
}

// ------------------------------------------------------------ error matrices

// Plain masked variants (safe probes; W = H = 4).
template <typename T, typename F>
void plain_errors(const std::string &fam, const std::string &suffix, int channels, T *s, T *d, const Npp8u *m, F f) {
  NppStreamContext ctx = current_ctx();
  const std::string fn = fam + suffix;
  const std::string id = fam + "/err/";
  const NppiSize roi{4, 4};
  const NppiSize ms{3, 3};
  const NppiPoint an{1, 1};
  const int dpx = channels * (int)sizeof(T);
  const int dAlign = channels == 4 ? dpx : (int)sizeof(T);
  only_status(fn, id + "nullsrc" + suffix, [&] { return f(nullptr, 16, d, 16, roi, m, ms, an, ctx); });
  only_status(fn, id + "nulldst" + suffix, [&] { return f(s, 16, nullptr, 16, roi, m, ms, an, ctx); });
  only_status(fn, id + "negroi" + suffix, [&] { return f(s, 16, d, 16, NppiSize{4, -1}, m, ms, an, ctx); });
  only_status(fn, id + "zeroroi" + suffix, [&] { return f(s, 16, d, 3, NppiSize{0, 0}, m, ms, an, ctx); });
  if (suffix == "_8u_C1R_Ctx") {
    only_status(fn, id + "zerow" + suffix, [&] { return f(s, 16, d, 16, NppiSize{0, 7}, m, ms, an, ctx); });
    only_status(fn, id + "zeroh" + suffix, [&] { return f(s, 16, d, 16, NppiSize{7, 0}, m, ms, an, ctx); });
  }
  only_status(fn, id + "stepsmall" + suffix,
              [&] { return f(s, 16, d, 4 * dpx - 1, roi, m, ms, an, ctx); });
  if (dAlign > 1)
    only_status(fn, id + "stepodd" + suffix, [&] { return f(s, 16, d, 4 * dpx + 1, roi, m, ms, an, ctx); });
}

// Plain 3x3 variants (safe probes).
template <typename T, typename F>
void plain3x3_errors(const std::string &fam, const std::string &suffix, int channels, T *s, T *d, F f) {
  NppStreamContext ctx = current_ctx();
  const std::string fn = fam + suffix;
  const std::string id = fam + "/err/";
  const NppiSize roi{4, 4};
  const int dpx = channels * (int)sizeof(T);
  const int dAlign = channels == 4 ? dpx : (int)sizeof(T);
  only_status(fn, id + "nullsrc" + suffix, [&] { return f(nullptr, 16, d, 16, roi, ctx); });
  only_status(fn, id + "nulldst" + suffix, [&] { return f(s, 16, nullptr, 16, roi, ctx); });
  only_status(fn, id + "negroi" + suffix, [&] { return f(s, 16, d, 16, NppiSize{4, -1}, ctx); });
  only_status(fn, id + "zeroroi" + suffix, [&] { return f(s, 16, d, 3, NppiSize{0, 0}, ctx); });
  only_status(fn, id + "stepsmall" + suffix, [&] { return f(s, 16, d, 4 * dpx - 1, roi, ctx); });
  if (dAlign > 1)
    only_status(fn, id + "stepodd" + suffix, [&] { return f(s, 16, d, 4 * dpx + 1, roi, ctx); });
}

// Border masked variants (safe probes).
template <typename T, typename F>
void border_mask_errors(const std::string &fam, const std::string &suffix, int channels, T *s, T *d, const Npp8u *m,
                        F f) {
  NppStreamContext ctx = current_ctx();
  const std::string fn = fam + suffix;
  const std::string id = fam + "/err/";
  const NppiSize sz{7, 7};
  const NppiPoint off{2, 2};
  const NppiSize roi{4, 4};
  const NppiSize ms{3, 3};
  const NppiPoint an{1, 1};
  const int dpx = channels * (int)sizeof(T);
  const int dAlign = channels == 4 ? dpx : (int)sizeof(T);
  const NppiSize msz{3, 3};
  const NppiBorderType bad = NPP_BORDER_CONSTANT;
  only_status(fn, id + "nullsrc" + suffix,
              [&] { return f(nullptr, 16, sz, off, d, 16, roi, m, ms, an, NPP_BORDER_REPLICATE, ctx); });
  only_status(fn, id + "nulldst" + suffix,
              [&] { return f(s, 16, sz, off, nullptr, 16, roi, m, ms, an, NPP_BORDER_REPLICATE, ctx); });
  only_status(fn, id + "negroi" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, NppiSize{4, -1}, m, ms, an, NPP_BORDER_REPLICATE, ctx); });
  only_status(fn, id + "zeroroi" + suffix,
              [&] { return f(s, 16, NppiSize{0, 5}, off, d, 3, NppiSize{0, 0}, m, msz, an, bad, ctx); });
  only_status(fn, id + "srcsize" + suffix,
              [&] { return f(s, 16, NppiSize{0, 5}, off, d, 16, roi, m, ms, an, bad, ctx); });
  only_status(fn, id + "offneg" + suffix,
              [&] { return f(s, 16, sz, NppiPoint{-1, 0}, d, 16, roi, m, ms, an, bad, ctx); });
  only_status(fn, id + "badborder" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, roi, m, ms, an, bad, ctx); });
  if (suffix == "_8u_C1R_Ctx") {
    only_status(fn, id + "badborder_mirror" + suffix,
                [&] { return f(s, 16, sz, off, d, 16, roi, m, ms, an, NPP_BORDER_MIRROR, ctx); });
    only_status(fn, id + "badborder_wrap" + suffix,
                [&] { return f(s, 16, sz, off, d, 16, roi, m, ms, an, NPP_BORDER_WRAP, ctx); });
    only_status(fn, id + "badborder_undef" + suffix,
                [&] { return f(s, 16, sz, off, d, 16, roi, m, ms, an, NPP_BORDER_UNDEFINED, ctx); });
    only_status(fn, id + "offeq" + suffix,
                [&] { return f(s, 16, sz, NppiPoint{7, 0}, d, 16, roi, m, ms, an, NPP_BORDER_REPLICATE, ctx); });
  }
  only_status(fn, id + "stepsmall" + suffix,
              [&] { return f(s, 16, sz, off, d, 4 * dpx - 1, roi, m, ms, an, NPP_BORDER_REPLICATE, ctx); });
  if (dAlign > 1)
    only_status(fn, id + "stepodd" + suffix,
                [&] { return f(s, 16, sz, off, d, 4 * dpx + 1, roi, m, ms, an, NPP_BORDER_REPLICATE, ctx); });
  // Precedence pins (mirroring the derivative-filter measurements).
  only_status(fn, id + "negroi_badborder" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, NppiSize{4, -1}, m, ms, an, bad, ctx); });
  only_status(fn, id + "zroi_offneg" + suffix,
              [&] { return f(s, 16, sz, NppiPoint{0, -1}, d, 16, NppiSize{0, 0}, m, ms, an, NPP_BORDER_REPLICATE, ctx); });
  only_status(fn, id + "zroi_badborder" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, NppiSize{0, 0}, m, ms, an, bad, ctx); });
  only_status(fn, id + "zroi_badstep" + suffix,
              [&] { return f(s, 16, sz, off, d, 3, NppiSize{0, 0}, m, ms, an, NPP_BORDER_REPLICATE, ctx); });
}

// Border 3x3 variants (safe probes).
template <typename T, typename F>
void border3x3_errors(const std::string &fam, const std::string &suffix, int channels, T *s, T *d, F f) {
  NppStreamContext ctx = current_ctx();
  const std::string fn = fam + suffix;
  const std::string id = fam + "/err/";
  const NppiSize sz{7, 7};
  const NppiPoint off{2, 2};
  const NppiSize roi{4, 4};
  const int dpx = channels * (int)sizeof(T);
  const int dAlign = channels == 4 ? dpx : (int)sizeof(T);
  const NppiBorderType bad = NPP_BORDER_CONSTANT;
  only_status(fn, id + "nullsrc" + suffix,
              [&] { return f(nullptr, 16, sz, off, d, 16, roi, NPP_BORDER_REPLICATE, ctx); });
  only_status(fn, id + "nulldst" + suffix,
              [&] { return f(s, 16, sz, off, nullptr, 16, roi, NPP_BORDER_REPLICATE, ctx); });
  only_status(fn, id + "negroi" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, NppiSize{4, -1}, NPP_BORDER_REPLICATE, ctx); });
  only_status(fn, id + "zeroroi" + suffix,
              [&] { return f(s, 16, NppiSize{0, 5}, off, d, 3, NppiSize{0, 0}, bad, ctx); });
  only_status(fn, id + "srcsize" + suffix,
              [&] { return f(s, 16, NppiSize{0, 5}, off, d, 16, roi, bad, ctx); });
  only_status(fn, id + "offneg" + suffix,
              [&] { return f(s, 16, sz, NppiPoint{-1, 0}, d, 16, roi, bad, ctx); });
  only_status(fn, id + "badborder" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, roi, bad, ctx); });
  only_status(fn, id + "stepsmall" + suffix,
              [&] { return f(s, 16, sz, off, d, 4 * dpx - 1, roi, NPP_BORDER_REPLICATE, ctx); });
  if (dAlign > 1)
    only_status(fn, id + "stepodd" + suffix,
                [&] { return f(s, 16, sz, off, d, 4 * dpx + 1, roi, NPP_BORDER_REPLICATE, ctx); });
  only_status(fn, id + "negroi_badborder" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, NppiSize{4, -1}, bad, ctx); });
  only_status(fn, id + "zroi_badstep" + suffix,
              [&] { return f(s, 16, sz, off, d, 3, NppiSize{0, 0}, NPP_BORDER_REPLICATE, ctx); });
}

// Composite morphology (safe probes; scratch buffer is a real allocation).
template <typename T, typename F>
void composite_errors(const std::string &fam, const std::string &suffix, int channels, T *s, T *d, const Npp8u *m,
                      Npp8u *buf, F f) {
  NppStreamContext ctx = current_ctx();
  const std::string fn = fam + suffix;
  const std::string id = fam + "/err/";
  const NppiSize sz{7, 7};
  const NppiPoint off{2, 2};
  const NppiSize roi{4, 4};
  const NppiSize ms{3, 3};
  const NppiPoint an{1, 1};
  const int dpx = channels * (int)sizeof(T);
  const int dAlign = channels == 4 ? dpx : (int)sizeof(T);
  const NppiBorderType bad = NPP_BORDER_CONSTANT;
  only_status(fn, id + "nullsrc" + suffix,
              [&] { return f(nullptr, 16, sz, off, d, 16, roi, m, ms, an, buf, NPP_BORDER_REPLICATE, ctx); });
  only_status(fn, id + "nulldst" + suffix,
              [&] { return f(s, 16, sz, off, nullptr, 16, roi, m, ms, an, buf, NPP_BORDER_REPLICATE, ctx); });
  only_status(fn, id + "negroi" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, NppiSize{4, -1}, m, ms, an, buf, NPP_BORDER_REPLICATE, ctx); });
  only_status(fn, id + "zeroroi" + suffix,
              [&] { return f(s, 16, NppiSize{0, 5}, off, d, 3, NppiSize{0, 0}, m, ms, an, buf, bad, ctx); });
  only_status(fn, id + "srcsize" + suffix,
              [&] { return f(s, 16, NppiSize{0, 5}, off, d, 16, roi, m, ms, an, buf, bad, ctx); });
  only_status(fn, id + "offneg" + suffix,
              [&] { return f(s, 16, sz, NppiPoint{-1, 0}, d, 16, roi, m, ms, an, buf, bad, ctx); });
  only_status(fn, id + "badborder" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, roi, m, ms, an, buf, bad, ctx); });
  only_status(fn, id + "stepsmall" + suffix,
              [&] { return f(s, 16, sz, off, d, 4 * dpx - 1, roi, m, ms, an, buf, NPP_BORDER_REPLICATE, ctx); });
  if (dAlign > 1)
    only_status(fn, id + "stepodd" + suffix,
                [&] { return f(s, 16, sz, off, d, 4 * dpx + 1, roi, m, ms, an, buf, NPP_BORDER_REPLICATE, ctx); });
  only_status(fn, id + "negroi_badborder" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, NppiSize{4, -1}, m, ms, an, buf, bad, ctx); });
  only_status(fn, id + "zroi_badstep" + suffix,
              [&] { return f(s, 16, sz, off, d, 3, NppiSize{0, 0}, m, ms, an, buf, NPP_BORDER_REPLICATE, ctx); });
}

// Gray dilate/erode (safe probes).
template <typename T, typename M, typename F>
void gray_errors(const std::string &fam, const std::string &suffix, T *s, T *d, const M *m, F f) {
  NppStreamContext ctx = current_ctx();
  const std::string fn = fam + suffix;
  const std::string id = fam + "/err/";
  const NppiSize sz{7, 7};
  const NppiPoint off{2, 2};
  const NppiSize roi{4, 4};
  const NppiSize ms{3, 3};
  const NppiPoint an{1, 1};
  const int dpx = (int)sizeof(T);
  const NppiBorderType bad = NPP_BORDER_CONSTANT;
  only_status(fn, id + "nullsrc" + suffix,
              [&] { return f(nullptr, 16, sz, off, d, 16, roi, m, ms, an, NPP_BORDER_REPLICATE, ctx); });
  only_status(fn, id + "nulldst" + suffix,
              [&] { return f(s, 16, sz, off, nullptr, 16, roi, m, ms, an, NPP_BORDER_REPLICATE, ctx); });
  only_status(fn, id + "negroi" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, NppiSize{4, -1}, m, ms, an, NPP_BORDER_REPLICATE, ctx); });
  only_status(fn, id + "zeroroi" + suffix,
              [&] { return f(s, 16, NppiSize{0, 5}, off, d, 3, NppiSize{0, 0}, m, ms, an, bad, ctx); });
  only_status(fn, id + "srcsize" + suffix,
              [&] { return f(s, 16, NppiSize{0, 5}, off, d, 16, roi, m, ms, an, bad, ctx); });
  only_status(fn, id + "offneg" + suffix,
              [&] { return f(s, 16, sz, NppiPoint{-1, 0}, d, 16, roi, m, ms, an, bad, ctx); });
  only_status(fn, id + "badborder" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, roi, m, ms, an, bad, ctx); });
  only_status(fn, id + "stepsmall" + suffix,
              [&] { return f(s, 16, sz, off, d, 4 * dpx - 1, roi, m, ms, an, NPP_BORDER_REPLICATE, ctx); });
  only_status(fn, id + "stepodd" + suffix,
              [&] { return f(s, 16, sz, off, d, 4 * dpx + 1, roi, m, ms, an, NPP_BORDER_REPLICATE, ctx); });
  only_status(fn, id + "negroi_badborder" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, NppiSize{4, -1}, m, ms, an, bad, ctx); });
  only_status(fn, id + "zroi_badstep" + suffix,
              [&] { return f(s, 16, sz, off, d, 3, NppiSize{0, 0}, m, ms, an, NPP_BORDER_REPLICATE, ctx); });
}

// Buffer-size helper errors.
template <typename F> void gbs_errors(const std::string &fam, const std::string &suffix, F f) {
  const std::string fn = fam + suffix;
  const std::string id = fam + "/err/";
  int size = 0;
  only_status(fn, id + "nullout" + suffix, [&] { return f(NppiSize{4, 4}, nullptr); });
  {
    clear_cuda_error();
    NppStatus st = f(NppiSize{0, 0}, &size);
    Case c(id + "zeroroi" + suffix, fn);
    c.status(st);
    c.host_output<int>(&size, 1, ValueKind::I32);
    c.finish();
  }
  only_status(fn, id + "negroi" + suffix, [&] { return f(NppiSize{-1, 4}, &size); });
  only_status(fn, id + "negroi2" + suffix, [&] { return f(NppiSize{4, -1}, &size); });
}

// ------------------------------------------------------------ call macros

// masked plain 9x9 with the three basic structuring elements
#define MPL(FN, S, LAY, CH)                                                                          \
  morph_plain_ok<S, Npp8u>(#FN "_Ctx", #FN "/" LAY "/box3", CH, 9, 9, maskBox3(), NppiSize{3, 3},     \
                           NppiPoint{1, 1}, 3, FN##_Ctx);                                             \
  morph_plain_ok<S, Npp8u>(#FN "_Ctx", #FN "/" LAY "/cross3", CH, 9, 9, maskCross3(), NppiSize{3, 3}, \
                           NppiPoint{1, 1}, 3, FN##_Ctx);                                             \
  morph_plain_ok<S, Npp8u>(#FN "_Ctx", #FN "/" LAY "/one1", CH, 9, 9, maskOne1(), NppiSize{1, 1},     \
                           NppiPoint{0, 0}, 3, FN##_Ctx)

#define PLAIN_MATRIX(FAM)                                                                            \
  do {                                                                                               \
    MPL(FAM##_8u_C1R, Npp8u, "8u/C1", 1);                                                            \
    MPL(FAM##_8u_C3R, Npp8u, "8u/C3", 3);                                                            \
    MPL(FAM##_8u_C4R, Npp8u, "8u/C4", 4);                                                            \
    MPL(FAM##_8u_AC4R, Npp8u, "8u/AC4", 4);                                                          \
    MPL(FAM##_16u_C1R, Npp16u, "16u/C1", 1);                                                         \
    MPL(FAM##_16u_C3R, Npp16u, "16u/C3", 3);                                                         \
    MPL(FAM##_16u_C4R, Npp16u, "16u/C4", 4);                                                         \
    MPL(FAM##_16u_AC4R, Npp16u, "16u/AC4", 4);                                                       \
    MPL(FAM##_32f_C1R, Npp32f, "32f/C1", 1);                                                         \
    MPL(FAM##_32f_C3R, Npp32f, "32f/C3", 3);                                                         \
    MPL(FAM##_32f_C4R, Npp32f, "32f/C4", 4);                                                         \
    MPL(FAM##_32f_AC4R, Npp32f, "32f/AC4", 4);                                                       \
    morph_plain_ok<Npp8u, Npp8u>(#FAM "_8u_C1R_Ctx", #FAM "/8u/C1/m5x3", 1, 9, 9, mask5x3(),         \
                                 NppiSize{5, 3}, NppiPoint{1, 1}, 5, FAM##_8u_C1R_Ctx);              \
    morph_plain_ok<Npp8u, Npp8u>(#FAM "_8u_C1R_Ctx", #FAM "/8u/C1/11x7", 1, 11, 7, maskBox3(),       \
                                 NppiSize{3, 3}, NppiPoint{1, 1}, 5, FAM##_8u_C1R_Ctx);              \
    morph_plain_ok<Npp8u, Npp8u>(#FAM "_8u_C1R_Ctx", #FAM "/8u/C1/1x1", 1, 1, 1, maskBox3(),         \
                                 NppiSize{3, 3}, NppiPoint{1, 1}, 3, FAM##_8u_C1R_Ctx);              \
    morph_plain_ok<Npp8u, Npp8u>(#FAM "_8u_C1R_Ctx", #FAM "/8u/C1/1x7", 1, 1, 7, maskBox3(),         \
                                 NppiSize{3, 3}, NppiPoint{1, 1}, 3, FAM##_8u_C1R_Ctx);              \
    morph_plain_ok<Npp8u, Npp8u>(#FAM "_8u_C1R_Ctx", #FAM "/8u/C1/7x1", 1, 7, 1, maskBox3(),         \
                                 NppiSize{3, 3}, NppiPoint{1, 1}, 3, FAM##_8u_C1R_Ctx);              \
    morph_plain_ok<Npp32f, Npp8u>(#FAM "_32f_C1R_Ctx", #FAM "/32f/C1/11x7", 1, 11, 7, maskCross3(),   \
                                 NppiSize{3, 3}, NppiPoint{1, 1}, 5, FAM##_32f_C1R_Ctx);             \
    plain_errors(#FAM, "_8u_C1R_Ctx", 1, arena<Npp8u>(64).get(), arena<Npp8u>(64).get(),             \
                 dev_mask(maskBox3()), FAM##_8u_C1R_Ctx);                                            \
    plain_errors(#FAM, "_16u_C1R_Ctx", 1, arena<Npp16u>(64).get(), arena<Npp16u>(64).get(),          \
                 dev_mask(maskBox3()), FAM##_16u_C1R_Ctx);                                           \
    plain_errors(#FAM, "_8u_AC4R_Ctx", 4, arena<Npp8u>(64).get(), arena<Npp8u>(64).get(),            \
                 dev_mask(maskBox3()), FAM##_8u_AC4R_Ctx);                                           \
    plain_errors(#FAM, "_32f_C1R_Ctx", 1, arena<Npp32f>(64).get(), arena<Npp32f>(64).get(),          \
                 dev_mask(maskBox3()), FAM##_32f_C1R_Ctx);                                           \
  } while (0)

#define M3P(FN, S, LAY, CH)                                                                           \
  morph3x3_plain_ok<S>(#FN "_Ctx", #FN "/" LAY "/9x9", CH, 9, 9, 3, FN##_Ctx)

#define PLAIN3x3_MATRIX(FAM)                                                                          \
  do {                                                                                               \
    M3P(FAM##_8u_C1R, Npp8u, "8u/C1", 1);                                                            \
    M3P(FAM##_8u_C3R, Npp8u, "8u/C3", 3);                                                            \
    M3P(FAM##_8u_C4R, Npp8u, "8u/C4", 4);                                                            \
    M3P(FAM##_8u_AC4R, Npp8u, "8u/AC4", 4);                                                          \
    M3P(FAM##_16u_C1R, Npp16u, "16u/C1", 1);                                                         \
    M3P(FAM##_16u_C3R, Npp16u, "16u/C3", 3);                                                         \
    M3P(FAM##_16u_C4R, Npp16u, "16u/C4", 4);                                                         \
    M3P(FAM##_16u_AC4R, Npp16u, "16u/AC4", 4);                                                       \
    M3P(FAM##_32f_C1R, Npp32f, "32f/C1", 1);                                                         \
    M3P(FAM##_32f_C3R, Npp32f, "32f/C3", 3);                                                         \
    M3P(FAM##_32f_C4R, Npp32f, "32f/C4", 4);                                                         \
    M3P(FAM##_32f_AC4R, Npp32f, "32f/AC4", 4);                                                       \
    M3P(FAM##_64f_C1R, Npp64f, "64f/C1", 1);                                                         \
    morph3x3_plain_ok<Npp8u>(#FAM "_8u_C1R_Ctx", #FAM "/8u/C1/11x7", 1, 11, 7, 5, FAM##_8u_C1R_Ctx); \
    morph3x3_plain_ok<Npp8u>(#FAM "_8u_C1R_Ctx", #FAM "/8u/C1/1x1", 1, 1, 1, 3, FAM##_8u_C1R_Ctx);   \
    morph3x3_plain_ok<Npp8u>(#FAM "_8u_C1R_Ctx", #FAM "/8u/C1/1x7", 1, 1, 7, 3, FAM##_8u_C1R_Ctx);   \
    morph3x3_plain_ok<Npp8u>(#FAM "_8u_C1R_Ctx", #FAM "/8u/C1/7x1", 1, 7, 1, 3, FAM##_8u_C1R_Ctx);   \
    morph3x3_plain_ok<Npp64f>(#FAM "_64f_C1R_Ctx", #FAM "/64f/C1/1x1", 1, 1, 1, 3,                    \
                              FAM##_64f_C1R_Ctx);                                                    \
    plain3x3_errors(#FAM, "_8u_C1R_Ctx", 1, arena<Npp8u>(64).get(), arena<Npp8u>(64).get(),           \
                    FAM##_8u_C1R_Ctx);                                                               \
    plain3x3_errors(#FAM, "_16u_C4R_Ctx", 4, arena<Npp16u>(64).get(), arena<Npp16u>(64).get(),        \
                    FAM##_16u_C4R_Ctx);                                                              \
    plain3x3_errors(#FAM, "_32f_C1R_Ctx", 1, arena<Npp32f>(64).get(), arena<Npp32f>(64).get(),        \
                    FAM##_32f_C1R_Ctx);                                                              \
  } while (0)

// masked border 9x9 inside a 13x13 image at offset (2,3)
#define MBR(FN, S, LAY, CH)                                                                           \
  morph_border_ok<S, Npp8u>(#FN "_Ctx", #FN "/" LAY "/9x9", CH, 13, 13, 2, 3, 9, 9, maskBox3(),       \
                            NppiSize{3, 3}, NppiPoint{1, 1}, 3, FN##_Ctx)

#define BORDER_MATRIX(FAM)                                                                            \
  do {                                                                                                \
    MBR(FAM##_8u_C1R, Npp8u, "8u/C1", 1);                                                             \
    MBR(FAM##_8u_C3R, Npp8u, "8u/C3", 3);                                                             \
    MBR(FAM##_8u_C4R, Npp8u, "8u/C4", 4);                                                             \
    MBR(FAM##_8u_AC4R, Npp8u, "8u/AC4", 4);                                                           \
    MBR(FAM##_16u_C1R, Npp16u, "16u/C1", 1);                                                          \
    MBR(FAM##_16u_C3R, Npp16u, "16u/C3", 3);                                                          \
    MBR(FAM##_16u_C4R, Npp16u, "16u/C4", 4);                                                          \
    MBR(FAM##_16u_AC4R, Npp16u, "16u/AC4", 4);                                                        \
    MBR(FAM##_32f_C1R, Npp32f, "32f/C1", 1);                                                          \
    MBR(FAM##_32f_C3R, Npp32f, "32f/C3", 3);                                                          \
    MBR(FAM##_32f_C4R, Npp32f, "32f/C4", 4);                                                          \
    MBR(FAM##_32f_AC4R, Npp32f, "32f/AC4", 4);                                                        \
    morph_border_ok<Npp8u, Npp8u>(#FAM "_8u_C1R_Ctx", #FAM "/8u/C1/corner", 1, 13, 13, 0, 0, 9, 9,    \
                                  maskBox3(), NppiSize{3, 3}, NppiPoint{1, 1}, 3, FAM##_8u_C1R_Ctx); \
    morph_border_ok<Npp8u, Npp8u>(#FAM "_8u_C1R_Ctx", #FAM "/8u/C1/overhang", 1, 13, 13, 10, 10, 5, 5, \
                                  maskBox3(), NppiSize{3, 3}, NppiPoint{1, 1}, 3, FAM##_8u_C1R_Ctx); \
    morph_border_ok<Npp8u, Npp8u>(#FAM "_8u_C1R_Ctx", #FAM "/8u/C1/1x7", 1, 13, 13, 2, 2, 1, 7,       \
                                  maskCross3(), NppiSize{3, 3}, NppiPoint{1, 1}, 3, FAM##_8u_C1R_Ctx); \
    morph_border_ok<Npp8u, Npp8u>(#FAM "_8u_C1R_Ctx", #FAM "/8u/C1/7x1", 1, 13, 13, 2, 2, 7, 1,       \
                                  maskCross3(), NppiSize{3, 3}, NppiPoint{1, 1}, 3, FAM##_8u_C1R_Ctx); \
    morph_border_ok<Npp8u, Npp8u>(#FAM "_8u_C1R_Ctx", #FAM "/8u/C1/m5x3", 1, 13, 13, 2, 3, 9, 9,      \
                                  mask5x3(), NppiSize{5, 3}, NppiPoint{1, 1}, 5, FAM##_8u_C1R_Ctx);  \
    morph_border_ok<Npp16u, Npp8u>(#FAM "_16u_AC4R_Ctx", #FAM "/16u/AC4/cross", 4, 11, 9, 1, 2, 7, 5, \
                                   maskCross3(), NppiSize{3, 3}, NppiPoint{1, 1}, 3,                 \
                                   FAM##_16u_AC4R_Ctx);                                               \
    morph_border_ok<Npp32f, Npp8u>(#FAM "_32f_C3R_Ctx", #FAM "/32f/C3/overhang", 3, 11, 9, 8, 6, 5, 5, \
                                   maskBox3(), NppiSize{3, 3}, NppiPoint{1, 1}, 3,                   \
                                   FAM##_32f_C3R_Ctx);                                                \
    border_mask_errors(#FAM, "_8u_C1R_Ctx", 1, arena<Npp8u>(64).get(), arena<Npp8u>(64).get(),        \
                       dev_mask(maskBox3()), FAM##_8u_C1R_Ctx);                                      \
    border_mask_errors(#FAM, "_16u_C1R_Ctx", 1, arena<Npp16u>(64).get(), arena<Npp16u>(64).get(),     \
                       dev_mask(maskBox3()), FAM##_16u_C1R_Ctx);                                     \
    border_mask_errors(#FAM, "_8u_AC4R_Ctx", 4, arena<Npp8u>(64).get(), arena<Npp8u>(64).get(),       \
                       dev_mask(maskBox3()), FAM##_8u_AC4R_Ctx);                                     \
    border_mask_errors(#FAM, "_32f_C1R_Ctx", 1, arena<Npp32f>(64).get(), arena<Npp32f>(64).get(),     \
                       dev_mask(maskBox3()), FAM##_32f_C1R_Ctx);                                     \
  } while (0)

#define M3B(FN, S, LAY, CH)                                                                           \
  morph3x3_border_ok<S>(#FN "_Ctx", #FN "/" LAY "/9x9", CH, 13, 13, 2, 3, 9, 9, 3, FN##_Ctx)

#define BORDER3x3_MATRIX(FAM)                                                                         \
  do {                                                                                                \
    M3B(FAM##_8u_C1R, Npp8u, "8u/C1", 1);                                                             \
    M3B(FAM##_8u_C3R, Npp8u, "8u/C3", 3);                                                             \
    M3B(FAM##_8u_C4R, Npp8u, "8u/C4", 4);                                                             \
    M3B(FAM##_8u_AC4R, Npp8u, "8u/AC4", 4);                                                           \
    M3B(FAM##_16u_C1R, Npp16u, "16u/C1", 1);                                                          \
    M3B(FAM##_16u_C3R, Npp16u, "16u/C3", 3);                                                          \
    M3B(FAM##_16u_C4R, Npp16u, "16u/C4", 4);                                                          \
    M3B(FAM##_16u_AC4R, Npp16u, "16u/AC4", 4);                                                        \
    M3B(FAM##_32f_C1R, Npp32f, "32f/C1", 1);                                                          \
    M3B(FAM##_32f_C3R, Npp32f, "32f/C3", 3);                                                          \
    M3B(FAM##_32f_C4R, Npp32f, "32f/C4", 4);                                                          \
    M3B(FAM##_32f_AC4R, Npp32f, "32f/AC4", 4);                                                        \
    morph3x3_border_ok<Npp8u>(#FAM "_8u_C1R_Ctx", #FAM "/8u/C1/corner", 1, 13, 13, 0, 0, 9, 9, 3,     \
                              FAM##_8u_C1R_Ctx);                                                      \
    morph3x3_border_ok<Npp8u>(#FAM "_8u_C1R_Ctx", #FAM "/8u/C1/overhang", 1, 13, 13, 10, 10, 5, 5, 3, \
                              FAM##_8u_C1R_Ctx);                                                      \
    morph3x3_border_ok<Npp8u>(#FAM "_8u_C1R_Ctx", #FAM "/8u/C1/1x1", 1, 13, 13, 2, 2, 1, 1, 3,        \
                              FAM##_8u_C1R_Ctx);                                                      \
    border3x3_errors(#FAM, "_8u_C1R_Ctx", 1, arena<Npp8u>(64).get(), arena<Npp8u>(64).get(),          \
                     FAM##_8u_C1R_Ctx);                                                               \
    border3x3_errors(#FAM, "_16u_C4R_Ctx", 4, arena<Npp16u>(64).get(), arena<Npp16u>(64).get(),       \
                     FAM##_16u_C4R_Ctx);                                                              \
    border3x3_errors(#FAM, "_32f_C1R_Ctx", 1, arena<Npp32f>(64).get(), arena<Npp32f>(64).get(),       \
                     FAM##_32f_C1R_Ctx);                                                              \
  } while (0)

// composite morphology 9x9 inside a 13x13 image at offset (2,3)
#define MCB(FAM, FN, S, LAY, CH, GBS)                                                                 \
  morph_composite_ok<S>(#FN "_Ctx", #FN "/" LAY "/9x9", CH, 13, 13, 2, 3, 9, 9, maskBox3(),           \
                        NppiSize{3, 3}, NppiPoint{1, 1}, 3, FN##_Ctx, GBS)

#define COMPOSITE_MATRIX(FAM)                                                                         \
  do {                                                                                                \
    MCB(FAM, FAM##_8u_C1R, Npp8u, "8u/C1", 1, nppiMorphGetBufferSize_8u_C1R);                         \
    MCB(FAM, FAM##_8u_C3R, Npp8u, "8u/C3", 3, nppiMorphGetBufferSize_8u_C3R);                         \
    MCB(FAM, FAM##_8u_C4R, Npp8u, "8u/C4", 4, nppiMorphGetBufferSize_8u_C4R);                         \
    MCB(FAM, FAM##_16s_C1R, Npp16s, "16s/C1", 1, nppiMorphGetBufferSize_16s_C1R);                     \
    MCB(FAM, FAM##_16u_C1R, Npp16u, "16u/C1", 1, nppiMorphGetBufferSize_16u_C1R);                     \
    MCB(FAM, FAM##_32f_C1R, Npp32f, "32f/C1", 1, nppiMorphGetBufferSize_32f_C1R);                     \
    MCB(FAM, FAM##_32f_C3R, Npp32f, "32f/C3", 3, nppiMorphGetBufferSize_32f_C3R);                     \
    MCB(FAM, FAM##_32f_C4R, Npp32f, "32f/C4", 4, nppiMorphGetBufferSize_32f_C4R);                     \
    morph_composite_ok<Npp8u>(#FAM "_8u_C1R_Ctx", #FAM "/8u/C1/m5x3", 1, 13, 13, 2, 3, 9, 9,          \
                              mask5x3(), NppiSize{5, 3}, NppiPoint{1, 1}, 5, FAM##_8u_C1R_Ctx,        \
                              nppiMorphGetBufferSize_8u_C1R);                                         \
    morph_composite_ok<Npp8u>(#FAM "_8u_C1R_Ctx", #FAM "/8u/C1/1x1", 1, 13, 13, 2, 2, 1, 1,           \
                              maskBox3(), NppiSize{3, 3}, NppiPoint{1, 1}, 3, FAM##_8u_C1R_Ctx,       \
                              nppiMorphGetBufferSize_8u_C1R);                                         \
    morph_composite_ok<Npp8u>(#FAM "_8u_C1R_Ctx", #FAM "/8u/C1/overhang", 1, 13, 13, 10, 10, 5, 5,    \
                              maskBox3(), NppiSize{3, 3}, NppiPoint{1, 1}, 3, FAM##_8u_C1R_Ctx,       \
                              nppiMorphGetBufferSize_8u_C1R);                                         \
    morph_composite_ok<Npp8u>(#FAM "_8u_C1R_Ctx", #FAM "/8u/C1/cross3", 1, 13, 13, 2, 3, 9, 9,        \
                              maskCross3(), NppiSize{3, 3}, NppiPoint{1, 1}, 3, FAM##_8u_C1R_Ctx,     \
                              nppiMorphGetBufferSize_8u_C1R);                                         \
    morph_composite_ok<Npp32f>(#FAM "_32f_C1R_Ctx", #FAM "/32f/C1/1x7", 1, 13, 13, 2, 2, 1, 7,        \
                               maskBox3(), NppiSize{3, 3}, NppiPoint{1, 1}, 3, FAM##_32f_C1R_Ctx,     \
                               nppiMorphGetBufferSize_32f_C1R);                                       \
    composite_errors(#FAM, "_8u_C1R_Ctx", 1, arena<Npp8u>(64).get(), arena<Npp8u>(64).get(),          \
                     dev_mask(maskBox3()), zero_arena<Npp8u>(4096), FAM##_8u_C1R_Ctx);                \
    composite_errors(#FAM, "_16s_C1R_Ctx", 1, arena<Npp16s>(64).get(), arena<Npp16s>(64).get(),       \
                     dev_mask(maskBox3()), zero_arena<Npp8u>(4096), FAM##_16s_C1R_Ctx);               \
    composite_errors(#FAM, "_32f_C4R_Ctx", 4, arena<Npp32f>(64).get(), arena<Npp32f>(64).get(),       \
                     dev_mask(maskBox3()), zero_arena<Npp8u>(4096), FAM##_32f_C4R_Ctx);               \
  } while (0)

}  // namespace

class NppiMorphDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

// ==================== Dilate / Erode (plain, arbitrary mask) ====================
TEST_F(NppiMorphDiff, Dilate) { PLAIN_MATRIX(nppiDilate); }
TEST_F(NppiMorphDiff, Erode) { PLAIN_MATRIX(nppiErode); }

// ==================== Dilate3x3 / Erode3x3 (plain) ====================
TEST_F(NppiMorphDiff, Dilate3x3) { PLAIN3x3_MATRIX(nppiDilate3x3); }
TEST_F(NppiMorphDiff, Erode3x3) { PLAIN3x3_MATRIX(nppiErode3x3); }

// ==================== Dilate / Erode Border (arbitrary mask) ====================
TEST_F(NppiMorphDiff, DilateBorder) { BORDER_MATRIX(nppiDilateBorder); }
TEST_F(NppiMorphDiff, ErodeBorder) { BORDER_MATRIX(nppiErodeBorder); }

// ==================== Dilate3x3 / Erode3x3 Border ====================
TEST_F(NppiMorphDiff, Dilate3x3Border) { BORDER3x3_MATRIX(nppiDilate3x3Border); }
TEST_F(NppiMorphDiff, Erode3x3Border) { BORDER3x3_MATRIX(nppiErode3x3Border); }

// ==================== Morph composites (scratch-buffer ops) ====================
TEST_F(NppiMorphDiff, MorphOpenBorder) { COMPOSITE_MATRIX(nppiMorphOpenBorder); }
TEST_F(NppiMorphDiff, MorphCloseBorder) { COMPOSITE_MATRIX(nppiMorphCloseBorder); }
TEST_F(NppiMorphDiff, MorphGradientBorder) { COMPOSITE_MATRIX(nppiMorphGradientBorder); }
TEST_F(NppiMorphDiff, MorphTopHatBorder) { COMPOSITE_MATRIX(nppiMorphTopHatBorder); }
TEST_F(NppiMorphDiff, MorphBlackHatBorder) { COMPOSITE_MATRIX(nppiMorphBlackHatBorder); }

// ==================== Gray dilate / erode (signed mask) ====================
TEST_F(NppiMorphDiff, GrayDilateBorder) {
  gray_ok<Npp8u, Npp32s>("nppiGrayDilateBorder_8u_C1R_Ctx", "nppiGrayDilateBorder/8u/box3", 13, 13, 2, 3, 9, 9,
                         maskGray8u(), NppiSize{3, 3}, NppiPoint{1, 1}, 3, nppiGrayDilateBorder_8u_C1R_Ctx);
  gray_ok<Npp32f, Npp32f>("nppiGrayDilateBorder_32f_C1R_Ctx", "nppiGrayDilateBorder/32f/box3", 13, 13, 2, 3, 9, 9,
                          maskGray32f(), NppiSize{3, 3}, NppiPoint{1, 1}, 3, nppiGrayDilateBorder_32f_C1R_Ctx);
  gray_ok<Npp8u, Npp32s>("nppiGrayDilateBorder_8u_C1R_Ctx", "nppiGrayDilateBorder/8u/corner", 13, 13, 0, 0, 9, 9,
                         maskGray8u(), NppiSize{3, 3}, NppiPoint{1, 1}, 3, nppiGrayDilateBorder_8u_C1R_Ctx);
  gray_ok<Npp8u, Npp32s>("nppiGrayDilateBorder_8u_C1R_Ctx", "nppiGrayDilateBorder/8u/m5x3", 13, 13, 2, 3, 9, 9,
                         std::vector<Npp32s>{0, 1, -1, 2, 0}, NppiSize{5, 1}, NppiPoint{2, 0}, 5,
                         nppiGrayDilateBorder_8u_C1R_Ctx);
  gray_errors("nppiGrayDilateBorder", "_8u_C1R_Ctx", arena<Npp8u>(64).get(), arena<Npp8u>(64).get(),
              dev_mask(maskGray8u()), nppiGrayDilateBorder_8u_C1R_Ctx);
  gray_errors("nppiGrayDilateBorder", "_32f_C1R_Ctx", arena<Npp32f>(64).get(), arena<Npp32f>(64).get(),
              dev_mask(maskGray32f()), nppiGrayDilateBorder_32f_C1R_Ctx);
}

TEST_F(NppiMorphDiff, GrayErodeBorder) {
  gray_ok<Npp8u, Npp32s>("nppiGrayErodeBorder_8u_C1R_Ctx", "nppiGrayErodeBorder/8u/box3", 13, 13, 2, 3, 9, 9,
                         maskGray8u(), NppiSize{3, 3}, NppiPoint{1, 1}, 3, nppiGrayErodeBorder_8u_C1R_Ctx);
  gray_ok<Npp32f, Npp32f>("nppiGrayErodeBorder_32f_C1R_Ctx", "nppiGrayErodeBorder/32f/box3", 13, 13, 2, 3, 9, 9,
                          maskGray32f(), NppiSize{3, 3}, NppiPoint{1, 1}, 3, nppiGrayErodeBorder_32f_C1R_Ctx);
  gray_ok<Npp8u, Npp32s>("nppiGrayErodeBorder_8u_C1R_Ctx", "nppiGrayErodeBorder/8u/overhang", 13, 13, 10, 10, 5, 5,
                         maskGray8u(), NppiSize{3, 3}, NppiPoint{1, 1}, 3, nppiGrayErodeBorder_8u_C1R_Ctx);
  gray_errors("nppiGrayErodeBorder", "_8u_C1R_Ctx", arena<Npp8u>(64).get(), arena<Npp8u>(64).get(),
              dev_mask(maskGray8u()), nppiGrayErodeBorder_8u_C1R_Ctx);
  gray_errors("nppiGrayErodeBorder", "_32f_C1R_Ctx", arena<Npp32f>(64).get(), arena<Npp32f>(64).get(),
              dev_mask(maskGray32f()), nppiGrayErodeBorder_32f_C1R_Ctx);
}

// ==================== MorphGetBufferSize (host-side helper) ====================
TEST_F(NppiMorphDiff, MorphGetBufferSize) {
  const NppiSize rois[] = {
      NppiSize{1, 1},   NppiSize{1, 2},    NppiSize{2, 1},     NppiSize{3, 3},     NppiSize{5, 5},
      NppiSize{7, 1},   NppiSize{1, 7},    NppiSize{9, 9},     NppiSize{11, 7},    NppiSize{17, 13},
      NppiSize{13, 17}, NppiSize{33, 17},  NppiSize{64, 64},   NppiSize{100, 1},   NppiSize{1, 100},
      NppiSize{257, 3}, NppiSize{1024, 64}};
  const struct {
    const char *name;
    NppStatus (*fn)(NppiSize, int *);
  } variants[] = {
      {"nppiMorphGetBufferSize_8u_C1R", nppiMorphGetBufferSize_8u_C1R},
      {"nppiMorphGetBufferSize_8u_C3R", nppiMorphGetBufferSize_8u_C3R},
      {"nppiMorphGetBufferSize_8u_C4R", nppiMorphGetBufferSize_8u_C4R},
      {"nppiMorphGetBufferSize_16s_C1R", nppiMorphGetBufferSize_16s_C1R},
      {"nppiMorphGetBufferSize_16u_C1R", nppiMorphGetBufferSize_16u_C1R},
      {"nppiMorphGetBufferSize_32f_C1R", nppiMorphGetBufferSize_32f_C1R},
      {"nppiMorphGetBufferSize_32f_C3R", nppiMorphGetBufferSize_32f_C3R},
      {"nppiMorphGetBufferSize_32f_C4R", nppiMorphGetBufferSize_32f_C4R},
  };
  for (const auto &v : variants) {
    for (const NppiSize roi : rois) {
      const std::string key = std::string("nppiMorphGetBufferSize/") + v.name + "/" + std::to_string(roi.width) +
                              "x" + std::to_string(roi.height);
      gbs_case(v.name, key, roi, v.fn);
    }
    gbs_errors("nppiMorphGetBufferSize", std::string("_") + v.name, v.fn);
  }
}

// ===========================================================================
// Risky validation probes: these pass null masks / scratch buffers or
// degenerate mask geometry that an implementation might dereference.  They run
// in a dedicated fixture so a crash cannot take down the main goldens; each
// probe uses zeroed, generously sized buffers so even an unvalidated call
// stays inside the allocation.
// ===========================================================================
class NppiMorphProbeDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

// 32x32 zeroed image with pSrc inset by (8, 8) pixels; ROI 4x4.
struct PaddedImage {
  Npp8u *base;
  int step;
  int channels;
  Npp8u *padded_src(int channels) {
    this->channels = channels;
    base = zero_arena<Npp8u>((size_t)32 * 32 * channels);
    step = 32 * channels;
    return base + 8 * step + 8 * channels;
  }
  Npp8u *padded_dst(int channels) { return zero_arena<Npp8u>((size_t)32 * 32 * channels); }
};

TEST_F(NppiMorphProbeDiff, PlainMaskDegenerate) {
  NppStreamContext ctx = current_ctx();
  PaddedImage img;
  Npp8u *s = img.padded_src(1);
  Npp8u *d = img.padded_dst(1);
  const int step = img.step;
  const NppiSize roi{4, 4};
  const NppiSize ms{3, 3};
  const Npp8u *m = dev_mask(maskBox3());
  const std::string fam = "nppiDilate";
  const std::string fn = "nppiDilate_8u_C1R_Ctx";
  const std::string id = fam + "/probe/";
  // null mask
  only_status(fn, id + "nullmask_8u_C1R_Ctx",
              [&] { return nppiDilate_8u_C1R_Ctx(s, step, d, step, roi, nullptr, ms, NppiPoint{1, 1}, ctx); });
  // 0x0 mask
  only_status(fn, id + "masksize0_8u_C1R_Ctx",
              [&] { return nppiDilate_8u_C1R_Ctx(s, step, d, step, roi, dev_mask(maskZero3()), NppiSize{0, 0}, NppiPoint{0, 0}, ctx); });
  // all-zero 3x3 mask
  only_status(fn, id + "maskzero_8u_C1R_Ctx",
              [&] { return nppiDilate_8u_C1R_Ctx(s, step, d, step, roi, dev_mask(maskZero3()), ms, NppiPoint{1, 1}, ctx); });
  // 1x0 mask
  only_status(fn, id + "mask1x0_8u_C1R_Ctx",
              [&] { return nppiDilate_8u_C1R_Ctx(s, step, d, step, roi, dev_mask(maskOne1()), NppiSize{1, 0}, NppiPoint{0, 0}, ctx); });
  // negative mask size
  only_status(fn, id + "maskneg_8u_C1R_Ctx",
              [&] { return nppiDilate_8u_C1R_Ctx(s, step, d, step, roi, dev_mask(maskBox3()), NppiSize{-1, 3}, NppiPoint{1, 1}, ctx); });
  // anchors outside [0, maskSize)
  only_status(fn, id + "anchorout_8u_C1R_Ctx",
              [&] { return nppiDilate_8u_C1R_Ctx(s, step, d, step, roi, m, ms, NppiPoint{3, 1}, ctx); });
  only_status(fn, id + "anchorout2_8u_C1R_Ctx",
              [&] { return nppiDilate_8u_C1R_Ctx(s, step, d, step, roi, m, ms, NppiPoint{0, 3}, ctx); });
  only_status(fn, id + "anchorneg_8u_C1R_Ctx",
              [&] { return nppiDilate_8u_C1R_Ctx(s, step, d, step, roi, m, ms, NppiPoint{-1, 1}, ctx); });
  only_status(fn, id + "anchoreq_8u_C1R_Ctx",
              [&] { return nppiDilate_8u_C1R_Ctx(s, step, d, step, roi, m, ms, NppiPoint{3, 3}, ctx); });
  // 16u type coverage
  Npp16u *s16 = zero_arena<Npp16u>(32 * 32);
  Npp16u *d16 = zero_arena<Npp16u>(32 * 32);
  const int step16 = 32 * (int)sizeof(Npp16u);
  Npp16u *ps16 = s16 + 8 * 32 + 8;
  only_status("nppiDilate_16u_C1R_Ctx", "nppiDilate/probe/nullmask_16u_C1R_Ctx",
              [&] { return nppiDilate_16u_C1R_Ctx(ps16, step16, d16, step16, roi, nullptr, ms, NppiPoint{1, 1}, ctx); });
  // erode counterpart (mask-null validation may differ per op)
  only_status("nppiErode_8u_C1R_Ctx", "nppiErode/probe/nullmask_8u_C1R_Ctx",
              [&] { return nppiErode_8u_C1R_Ctx(s, step, d, step, roi, nullptr, ms, NppiPoint{1, 1}, ctx); });
  only_status("nppiErode_8u_C1R_Ctx", "nppiErode/probe/masksize0_8u_C1R_Ctx",
              [&] { return nppiErode_8u_C1R_Ctx(s, step, d, step, roi, dev_mask(maskZero3()), NppiSize{0, 0}, NppiPoint{0, 0}, ctx); });
}

TEST_F(NppiMorphProbeDiff, BorderMaskDegenerate) {
  NppStreamContext ctx = current_ctx();
  PaddedImage img;
  Npp8u *s = img.padded_src(1);
  Npp8u *d = img.padded_dst(1);
  const int step = img.step;
  const NppiSize sz{16, 16};  // image subset declared to NPP
  const NppiPoint off{8, 8};
  const NppiSize roi{4, 4};
  const NppiSize ms{3, 3};
  const Npp8u *m = dev_mask(maskBox3());
  const std::string fn = "nppiDilateBorder_8u_C1R_Ctx";
  const std::string id = "nppiDilateBorder/probe/";
  only_status(fn, id + "nullmask_8u_C1R_Ctx",
              [&] { return nppiDilateBorder_8u_C1R_Ctx(s, step, sz, off, d, step, roi, nullptr, ms, NppiPoint{1, 1}, NPP_BORDER_REPLICATE, ctx); });
  only_status(fn, id + "masksize0_8u_C1R_Ctx",
              [&] { return nppiDilateBorder_8u_C1R_Ctx(s, step, sz, off, d, step, roi, dev_mask(maskZero3()), NppiSize{0, 0}, NppiPoint{0, 0}, NPP_BORDER_REPLICATE, ctx); });
  only_status(fn, id + "anchorout_8u_C1R_Ctx",
              [&] { return nppiDilateBorder_8u_C1R_Ctx(s, step, sz, off, d, step, roi, m, ms, NppiPoint{3, 1}, NPP_BORDER_REPLICATE, ctx); });
  only_status("nppiErodeBorder_8u_C1R_Ctx", "nppiErodeBorder/probe/nullmask_8u_C1R_Ctx",
              [&] { return nppiErodeBorder_8u_C1R_Ctx(s, step, sz, off, d, step, roi, nullptr, ms, NppiPoint{1, 1}, NPP_BORDER_REPLICATE, ctx); });
  only_status("nppiGrayDilateBorder_8u_C1R_Ctx", "nppiGrayDilateBorder/probe/nullmask_8u_C1R_Ctx",
              [&] { return nppiGrayDilateBorder_8u_C1R_Ctx(s, step, sz, off, d, step, roi, nullptr, ms, NppiPoint{1, 1}, NPP_BORDER_REPLICATE, ctx); });
  only_status("nppiGrayErodeBorder_32f_C1R_Ctx", "nppiGrayErodeBorder/probe/nullmask_32f_C1R_Ctx",
              [&] {
                Npp32f *s32 = zero_arena<Npp32f>(32 * 32);
                Npp32f *d32 = zero_arena<Npp32f>(32 * 32);
                return nppiGrayErodeBorder_32f_C1R_Ctx(s32 + 8 * 32 + 8, 32 * (int)sizeof(Npp32f), sz, off, d32,
                                                       32 * (int)sizeof(Npp32f), roi, nullptr, ms, NppiPoint{1, 1},
                                                       NPP_BORDER_REPLICATE, ctx);
              });
}

TEST_F(NppiMorphProbeDiff, CompositeDegenerate) {
  NppStreamContext ctx = current_ctx();
  PaddedImage img;
  Npp8u *s = img.padded_src(1);
  Npp8u *d = img.padded_dst(1);
  const int step = img.step;
  const NppiSize sz{16, 16};
  const NppiPoint off{8, 8};
  const NppiSize roi{4, 4};
  const NppiSize ms{3, 3};
  const Npp8u *m = dev_mask(maskBox3());
  Npp8u *buf = zero_arena<Npp8u>(65536);
  const std::string fn = "nppiMorphOpenBorder_8u_C1R_Ctx";
  const std::string id = "nppiMorphOpenBorder/probe/";
  only_status(fn, id + "nullmask_8u_C1R_Ctx",
              [&] { return nppiMorphOpenBorder_8u_C1R_Ctx(s, step, sz, off, d, step, roi, nullptr, ms, NppiPoint{1, 1}, buf, NPP_BORDER_REPLICATE, ctx); });
  only_status(fn, id + "nullbuf_8u_C1R_Ctx",
              [&] { return nppiMorphOpenBorder_8u_C1R_Ctx(s, step, sz, off, d, step, roi, m, ms, NppiPoint{1, 1}, nullptr, NPP_BORDER_REPLICATE, ctx); });
  only_status(fn, id + "masksize0_8u_C1R_Ctx",
              [&] { return nppiMorphOpenBorder_8u_C1R_Ctx(s, step, sz, off, d, step, roi, dev_mask(maskZero3()), NppiSize{0, 0}, NppiPoint{0, 0}, buf, NPP_BORDER_REPLICATE, ctx); });
  only_status(fn, id + "anchorout_8u_C1R_Ctx",
              [&] { return nppiMorphOpenBorder_8u_C1R_Ctx(s, step, sz, off, d, step, roi, m, ms, NppiPoint{3, 1}, buf, NPP_BORDER_REPLICATE, ctx); });
  only_status("nppiMorphCloseBorder_8u_C1R_Ctx", "nppiMorphCloseBorder/probe/nullbuf_8u_C1R_Ctx",
              [&] { return nppiMorphCloseBorder_8u_C1R_Ctx(s, step, sz, off, d, step, roi, m, ms, NppiPoint{1, 1}, nullptr, NPP_BORDER_REPLICATE, ctx); });
}

// ===========================================================================
// Non-Ctx adapters share the corresponding _Ctx golden (MPP build only)
// ===========================================================================
#if NPP_DIFF_HAS_NONCTX
TEST_F(NppiMorphDiff, NonCtx_Plain) {
  // replays nppiDilate_8u_C1R/8u/C1/box3 and nppiDilate_16u_C3R/16u/C3/box3 exactly
  morph_plain_ok<Npp8u, Npp8u>("nppiDilate_8u_C1R", "nppiDilate_8u_C1R/8u/C1/box3", 1, 9, 9, maskBox3(), NppiSize{3, 3},
                               NppiPoint{1, 1}, 3, [](const Npp8u *p, int ss, Npp8u *q, int ds, NppiSize roi,
                                                      const Npp8u *m, NppiSize ms, NppiPoint a, NppStreamContext) {
                                 return nppiDilate_8u_C1R(p, ss, q, ds, roi, m, ms, a);
                               });
  morph_plain_ok<Npp16u, Npp8u>("nppiDilate_16u_C3R", "nppiDilate_16u_C3R/16u/C3/box3", 3, 9, 9, maskBox3(), NppiSize{3, 3},
                                NppiPoint{1, 1}, 3, [](const Npp16u *p, int ss, Npp16u *q, int ds, NppiSize roi,
                                                       const Npp8u *m, NppiSize ms, NppiPoint a, NppStreamContext) {
                                  return nppiDilate_16u_C3R(p, ss, q, ds, roi, m, ms, a);
                                });
}

TEST_F(NppiMorphDiff, NonCtx_Plain3x3) {
  morph3x3_plain_ok<Npp8u>("nppiErode3x3_8u_C1R", "nppiErode3x3_8u_C1R/8u/C1/9x9", 1, 9, 9, 3,
                           [](const Npp8u *p, int ss, Npp8u *q, int ds, NppiSize roi, NppStreamContext) {
                             return nppiErode3x3_8u_C1R(p, ss, q, ds, roi);
                           });
  morph3x3_plain_ok<Npp64f>("nppiErode3x3_64f_C1R", "nppiErode3x3_64f_C1R/64f/C1/9x9", 1, 9, 9, 3,
                            [](const Npp64f *p, int ss, Npp64f *q, int ds, NppiSize roi, NppStreamContext) {
                              return nppiErode3x3_64f_C1R(p, ss, q, ds, roi);
                            });
}

TEST_F(NppiMorphDiff, NonCtx_Border) {
  morph_border_ok<Npp8u, Npp8u>(
      "nppiDilateBorder_8u_C1R", "nppiDilateBorder_8u_C1R/8u/C1/9x9", 1, 13, 13, 2, 3, 9, 9, maskBox3(), NppiSize{3, 3},
      NppiPoint{1, 1}, 3, [](const Npp8u *p, int ss, NppiSize sz, NppiPoint o, Npp8u *q, int ds, NppiSize roi,
                             const Npp8u *m, NppiSize ms, NppiPoint a, NppiBorderType b, NppStreamContext) {
        return nppiDilateBorder_8u_C1R(p, ss, sz, o, q, ds, roi, m, ms, a, b);
      });
  morph3x3_border_ok<Npp8u>(
      "nppiErode3x3Border_8u_C1R", "nppiErode3x3Border_8u_C1R/8u/C1/9x9", 1, 13, 13, 2, 3, 9, 9, 3,
      [](const Npp8u *p, int ss, NppiSize sz, NppiPoint o, Npp8u *q, int ds, NppiSize roi, NppiBorderType b,
         NppStreamContext) { return nppiErode3x3Border_8u_C1R(p, ss, sz, o, q, ds, roi, b); });
}

TEST_F(NppiMorphDiff, NonCtx_Composite) {
  morph_composite_ok<Npp8u>(
      "nppiMorphOpenBorder_8u_C1R", "nppiMorphOpenBorder_8u_C1R/8u/C1/9x9", 1, 13, 13, 2, 3, 9, 9, maskBox3(),
      NppiSize{3, 3}, NppiPoint{1, 1}, 3,
      [](const Npp8u *p, int ss, NppiSize sz, NppiPoint o, Npp8u *q, int ds, NppiSize roi, const Npp8u *m,
         NppiSize ms, NppiPoint a, Npp8u *buf, NppiBorderType b, NppStreamContext) {
        return nppiMorphOpenBorder_8u_C1R(p, ss, sz, o, q, ds, roi, m, ms, a, buf, b);
      },
      nppiMorphGetBufferSize_8u_C1R);
  morph_composite_ok<Npp16u>(
      "nppiMorphCloseBorder_16u_C1R", "nppiMorphCloseBorder_16u_C1R/16u/C1/9x9", 1, 13, 13, 2, 3, 9, 9, maskBox3(),
      NppiSize{3, 3}, NppiPoint{1, 1}, 3,
      [](const Npp16u *p, int ss, NppiSize sz, NppiPoint o, Npp16u *q, int ds, NppiSize roi, const Npp8u *m,
         NppiSize ms, NppiPoint a, Npp8u *buf, NppiBorderType b, NppStreamContext) {
        return nppiMorphCloseBorder_16u_C1R(p, ss, sz, o, q, ds, roi, m, ms, a, buf, b);
      },
      nppiMorphGetBufferSize_16u_C1R);
}

TEST_F(NppiMorphDiff, NonCtx_Gray) {
  gray_ok<Npp8u, Npp32s>(
      "nppiGrayDilateBorder_8u_C1R", "nppiGrayDilateBorder/8u/box3", 13, 13, 2, 3, 9, 9, maskGray8u(), NppiSize{3, 3},
      NppiPoint{1, 1}, 3, [](const Npp8u *p, int ss, NppiSize sz, NppiPoint o, Npp8u *q, int ds, NppiSize roi,
                             const Npp32s *m, NppiSize ms, NppiPoint a, NppiBorderType b, NppStreamContext) {
        return nppiGrayDilateBorder_8u_C1R(p, ss, sz, o, q, ds, roi, m, ms, a, b);
      });
}
#endif  // NPP_DIFF_HAS_NONCTX
