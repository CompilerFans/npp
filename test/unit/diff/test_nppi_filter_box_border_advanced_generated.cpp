// Differential tests for package f9: nppiFilterBoxBorderAdvanced (13 _Ctx
// entry points, REPLICATE only).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiFilterBoxBorderAdvancedDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppiFilterBoxBorderAdvancedDiff.*'
//
// MEASURED SEMANTICS (scratch/f9_semantics.md section 10): NVIDIA 13.4's
// "advanced" kernel only produces the documented replicate-box result when
// oSizeROI.width == oSrcSize.width and oSrcOffset.x == 0.  Every other
// geometry reads a window that drifts by oSizeROI.width per row index (the
// kernel uses the ROI width as a row stride into the source), so the reference
// disagrees with the documented contract there.  The value cases below are
// therefore restricted to the geometry where the reference agrees; the
// validation cases cover the full measured error contract.
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

constexpr int kMargin = 8;  // margin around the source image inside the allocation

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

template <typename T> std::vector<T> make_src(size_t n, uint64_t seed) {
  Rng rng(seed);
  std::vector<T> v(n);
  for (auto &x : v) {
    if constexpr (std::is_floating_point_v<T>) {
      x = static_cast<T>(rng.integer<int>(-64, 64)) * static_cast<T>(0.125);
    } else {
      x = rng.integer<T>(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
    }
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

// Data case: the ROI equals the source size (the geometry where the reference
// implements the documented contract), the source image sits inside a margin
// allocation and pSrc points at the image origin + oSrcOffset.
template <typename S, typename F>
void adv_data(const char *fname, const std::string &key, int ch, NppiSize srcSize,
              NppiPoint srcOffset, NppiSize mask, NppiPoint anchor, NppiBorderType border,
              uint64_t seed, F &&call) {
  const int px = ch * (int)sizeof(S);
  const int stride = align_up((srcSize.width + 2 * kMargin) * px, 16);
  const size_t sElems = (size_t)(srcSize.height + 2 * kMargin) * (stride / sizeof(S)) + 4;
  const int dstride = align_up(srcSize.width * px, 16);
  const size_t dElems = (size_t)(dstride / sizeof(S)) * srcSize.height;
  const size_t dBytes = (size_t)dstride * srcSize.height;
  const size_t bufBytes = (size_t)(srcSize.width * srcSize.height * ch + 2) * 8;

  std::vector<S> hs = make_src<S>(sElems, seed);
  DeviceMemory<S> &s = arena<S>(sElems);
  DeviceMemory<S> &d = arena<S>(dElems ? dElems : 1);
  DeviceMemory<Npp8u> &buf = arena<Npp8u>(bufBytes ? bufBytes : 1);
  s.copyFromHost(hs);
  std::vector<unsigned char> fill(dBytes, 0xAB);
  cudaMemcpy(d.get(), fill.data(), dBytes, cudaMemcpyHostToDevice);
  cudaMemset(buf.get(), 0, bufBytes);

  S *ps = s.get() + (size_t)(kMargin + srcOffset.y) * (stride / sizeof(S)) +
          (size_t)(kMargin + srcOffset.x) * ch;
  clear_cuda_error();
  NppStatus st = call(ps, stride, srcSize, srcOffset, d.get(), dstride, srcSize, mask, anchor,
                      border, buf.get(), current_ctx());

  Case c(std::string("nppiFilterBoxBorderAdvanced/") + key, fname);
  c.input(hs.data(), hs.size() * sizeof(S));
  c.status(st);
  c.device_output<S>(d.get(), dElems, dst_kind<S>(), 1e-5, 1e-5);
  c.finish();
}

#define ADV_DATA(S, TAG, CH)                                                                   \
  adv_data<S>("nppiFilterBoxBorderAdvanced_" #TAG, "sweep/" #TAG, CH, NppiSize{13, 11},       \
              NppiPoint{0, 0}, NppiSize{3, 3}, NppiPoint{1, 1}, NPP_BORDER_REPLICATE,         \
              0x9100u + (uint64_t)CH,                                                          \
              [](const S *s, int ss, NppiSize sz, NppiPoint o, S *d, int ds, NppiSize r,      \
                 NppiSize m, NppiPoint a, NppiBorderType b, Npp8u *p, NppStreamContext c) {   \
                return nppiFilterBoxBorderAdvanced_##TAG##_Ctx(s, ss, sz, o, d, ds, r, m, a,   \
                                                               b, p, c);                      \
              });                                                                              \
  adv_data<S>("nppiFilterBoxBorderAdvanced_" #TAG, "mask5/" #TAG, CH, NppiSize{13, 11},       \
              NppiPoint{0, 0}, NppiSize{5, 5}, NppiPoint{2, 2}, NPP_BORDER_REPLICATE,         \
              0x9200u + (uint64_t)CH,                                                          \
              [](const S *s, int ss, NppiSize sz, NppiPoint o, S *d, int ds, NppiSize r,      \
                 NppiSize m, NppiPoint a, NppiBorderType b, Npp8u *p, NppStreamContext c) {   \
                return nppiFilterBoxBorderAdvanced_##TAG##_Ctx(s, ss, sz, o, d, ds, r, m, a,   \
                                                               b, p, c);                      \
              });                                                                              \
  adv_data<S>("nppiFilterBoxBorderAdvanced_" #TAG, "small/" #TAG, CH, NppiSize{8, 6},         \
              NppiPoint{0, 0}, NppiSize{3, 3}, NppiPoint{1, 1}, NPP_BORDER_REPLICATE,         \
              0x9300u + (uint64_t)CH,                                                          \
              [](const S *s, int ss, NppiSize sz, NppiPoint o, S *d, int ds, NppiSize r,      \
                 NppiSize m, NppiPoint a, NppiBorderType b, Npp8u *p, NppStreamContext c) {   \
                return nppiFilterBoxBorderAdvanced_##TAG##_Ctx(s, ss, sz, o, d, ds, r, m, a,   \
                                                               b, p, c);                      \
              });                                                                              \
  static_assert(true, "");

#define ADV_ALL(FN)                                                                            \
  FN(Npp8u, 8u_C1R, 1)                                                                         \
  FN(Npp8u, 8u_C3R, 3)                                                                         \
  FN(Npp8u, 8u_C4R, 4)                                                                         \
  FN(Npp16u, 16u_C1R, 1)                                                                       \
  FN(Npp16u, 16u_C3R, 3)                                                                       \
  FN(Npp16u, 16u_C4R, 4)                                                                       \
  FN(Npp16s, 16s_C1R, 1)                                                                       \
  FN(Npp16s, 16s_C3R, 3)                                                                       \
  FN(Npp16s, 16s_C4R, 4)                                                                       \
  FN(Npp32f, 32f_C1R, 1)                                                                       \
  FN(Npp32f, 32f_C3R, 3)                                                                       \
  FN(Npp32f, 32f_C4R, 4)                                                                       \
  FN(Npp64f, 64f_C1R, 1)

}  // namespace

class NppiFilterBoxBorderAdvancedDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

TEST_F(NppiFilterBoxBorderAdvancedDiff, Sweep) {
  ADV_ALL(ADV_DATA);

  // Non-square mask and a sub-image anchor inside the clean geometry.
  adv_data<Npp8u>("nppiFilterBoxBorderAdvanced_8u_C1R", "mask5x3/8u_C1R", 1, NppiSize{13, 11},
                  NppiPoint{0, 0}, NppiSize{5, 3}, NppiPoint{2, 1}, NPP_BORDER_REPLICATE, 0x9401u,
                  [](const Npp8u *s, int ss, NppiSize sz, NppiPoint o, Npp8u *d, int ds,
                     NppiSize r, NppiSize m, NppiPoint a, NppiBorderType b, Npp8u *p,
                     NppStreamContext c) {
                    return nppiFilterBoxBorderAdvanced_8u_C1R_Ctx(s, ss, sz, o, d, ds, r, m, a, b,
                                                                  p, c);
                  });
  adv_data<Npp32f>("nppiFilterBoxBorderAdvanced_32f_C3R", "mask7x3/32f_C3R", 3, NppiSize{11, 9},
                   NppiPoint{0, 0}, NppiSize{7, 3}, NppiPoint{3, 1}, NPP_BORDER_REPLICATE, 0x9402u,
                   [](const Npp32f *s, int ss, NppiSize sz, NppiPoint o, Npp32f *d, int ds,
                      NppiSize r, NppiSize m, NppiPoint a, NppiBorderType b, Npp8u *p,
                      NppStreamContext c) {
                     return nppiFilterBoxBorderAdvanced_32f_C3R_Ctx(s, ss, sz, o, d, ds, r, m, a, b,
                                                                   p, c);
                   });
}

TEST_F(NppiFilterBoxBorderAdvancedDiff, Validation) {
  const std::string fam = "nppiFilterBoxBorderAdvanced";
  const char *fn = "nppiFilterBoxBorderAdvanced_8u_C1R";
  Npp8u *s = arena<Npp8u>(4096).get();
  Npp8u *d = arena<Npp8u>(4096).get();
  Npp8u *b = arena<Npp8u>(4096).get();
  NppStreamContext c = current_ctx();
  const NppiSize src{16, 12}, m3{3, 3};
  const NppiPoint off{0, 0}, a1{1, 1};
  auto call = [&](const Npp8u *ps, int ss, NppiSize sz, NppiPoint o, Npp8u *pd, int ds,
                  NppiSize r, NppiSize m, NppiPoint a, NppiBorderType bt, Npp8u *pb) {
    return nppiFilterBoxBorderAdvanced_8u_C1R_Ctx(ps, ss, sz, o, pd, ds, r, m, a, bt, pb, c);
  };
  status_case(fam, fn, "null_src", [&] { return call(nullptr, 16, src, off, d, 16, src, m3, a1, NPP_BORDER_REPLICATE, b); });
  status_case(fam, fn, "null_dst", [&] { return call(s, 16, src, off, nullptr, 16, src, m3, a1, NPP_BORDER_REPLICATE, b); });
  status_case(fam, fn, "null_buf", [&] { return call(s, 16, src, off, d, 16, src, m3, a1, NPP_BORDER_REPLICATE, nullptr); });
  status_case(fam, fn, "roi_neg_w", [&] { return call(s, 16, src, off, d, 16, NppiSize{-3, 7}, m3, a1, NPP_BORDER_REPLICATE, b); });
  status_case(fam, fn, "roi_neg_h", [&] { return call(s, 16, src, off, d, 16, NppiSize{7, -3}, m3, a1, NPP_BORDER_REPLICATE, b); });
  status_case(fam, fn, "roi_empty", [&] { return call(s, 16, src, off, d, 16, NppiSize{0, 0}, m3, a1, NPP_BORDER_REPLICATE, b); });
  status_case(fam, fn, "roi_zero_w", [&] { return call(s, 16, src, off, d, 16, NppiSize{0, 7}, m3, a1, NPP_BORDER_REPLICATE, b); });
  status_case(fam, fn, "roi_zero_h", [&] { return call(s, 16, src, off, d, 16, NppiSize{7, 0}, m3, a1, NPP_BORDER_REPLICATE, b); });
  status_case(fam, fn, "mask_zero", [&] { return call(s, 16, src, off, d, 16, src, NppiSize{0, 0}, a1, NPP_BORDER_REPLICATE, b); });
  status_case(fam, fn, "mask_neg", [&] { return call(s, 16, src, off, d, 16, src, NppiSize{-3, 3}, a1, NPP_BORDER_REPLICATE, b); });
  status_case(fam, fn, "anchor_oob", [&] { return call(s, 16, src, off, d, 16, src, m3, NppiPoint{5, 5}, NPP_BORDER_REPLICATE, b); });
  status_case(fam, fn, "dstep_small", [&] { return call(s, 16, src, off, d, 8, src, m3, a1, NPP_BORDER_REPLICATE, b); });
  status_case(fam, fn, "dstep_even_w_ok", [&] { return call(s, 16, src, off, d, 8, NppiSize{12, 12}, m3, a1, NPP_BORDER_REPLICATE, b); });
  status_case(fam, fn, "dstep_even_w_ok4", [&] { return call(s, 16, src, off, d, 4, NppiSize{16, 12}, m3, a1, NPP_BORDER_REPLICATE, b); });
  status_case(fam, fn, "dstep_even_w_small", [&] { return call(s, 16, src, off, d, 2, NppiSize{12, 12}, m3, a1, NPP_BORDER_REPLICATE, b); });
  status_case(fam, fn, "dstep_odd_width", [&] { return call(s, 16, src, off, d, 6, NppiSize{7, 5}, m3, a1, NPP_BORDER_REPLICATE, b); });
  status_case(fam, fn, "dstep_odd_width_ok", [&] { return call(s, 16, src, off, d, 7, NppiSize{7, 5}, m3, a1, NPP_BORDER_REPLICATE, b); });
  status_case(fam, fn, "dstep_zero", [&] { return call(s, 16, src, off, d, 0, src, m3, a1, NPP_BORDER_REPLICATE, b); });
  status_case(fam, fn, "dstep_neg", [&] { return call(s, 16, src, off, d, -16, src, m3, a1, NPP_BORDER_REPLICATE, b); });
  status_case(fam, fn, "sstep_small", [&] { return call(s, 8, src, off, d, 16, src, m3, a1, NPP_BORDER_REPLICATE, b); });
  status_case(fam, fn, "border_const", [&] { return call(s, 16, src, off, d, 16, src, m3, a1, NPP_BORDER_CONSTANT, b); });
  status_case(fam, fn, "border_wrap", [&] { return call(s, 16, src, off, d, 16, src, m3, a1, NPP_BORDER_WRAP, b); });
  status_case(fam, fn, "border_mirror", [&] { return call(s, 16, src, off, d, 16, src, m3, a1, NPP_BORDER_MIRROR, b); });
  status_case(fam, fn, "border_none", [&] { return call(s, 16, src, off, d, 16, src, m3, a1, NPP_BORDER_NONE, b); });
  status_case(fam, fn, "border_5", [&] { return call(s, 16, src, off, d, 16, src, m3, a1, (NppiBorderType)5, b); });
  status_case(fam, fn, "srcsize_zero", [&] { return call(s, 16, NppiSize{0, 0}, off, d, 16, src, m3, a1, NPP_BORDER_REPLICATE, b); });
  status_case(fam, fn, "srcsize_neg", [&] { return call(s, 16, NppiSize{-3, 7}, off, d, 16, src, m3, a1, NPP_BORDER_REPLICATE, b); });
  status_case(fam, fn, "offset_oob", [&] { return call(s, 16, src, NppiPoint{16, 0}, d, 16, src, m3, a1, NPP_BORDER_REPLICATE, b); });
  status_case(fam, fn, "offset_neg", [&] { return call(s, 16, src, NppiPoint{0, -1}, d, 16, src, m3, a1, NPP_BORDER_REPLICATE, b); });
  status_case(fam, fn, "roi_gt_src", [&] { return call(s, 16, src, off, d, 16, NppiSize{32, 32}, m3, a1, NPP_BORDER_REPLICATE, b); });
  // precedence samples measured on the 5090
  status_case(fam, fn, "nullsrc_and_negroi", [&] { return call(nullptr, 16, src, off, d, 16, NppiSize{-3, 7}, m3, a1, NPP_BORDER_REPLICATE, b); });
  status_case(fam, fn, "nullbuf_and_negroi", [&] { return call(s, 16, src, off, d, 16, NppiSize{-3, 7}, m3, a1, NPP_BORDER_REPLICATE, nullptr); });
  status_case(fam, fn, "nullbuf_and_mirror", [&] { return call(s, 16, src, off, d, 16, src, m3, a1, NPP_BORDER_MIRROR, nullptr); });
  status_case(fam, fn, "badoff_and_mirror", [&] { return call(s, 16, src, NppiPoint{99, 0}, d, 16, src, m3, a1, NPP_BORDER_MIRROR, b); });
  status_case(fam, fn, "negroi_and_badoff", [&] { return call(s, 16, src, NppiPoint{99, 0}, d, 16, NppiSize{-3, 7}, m3, a1, NPP_BORDER_REPLICATE, b); });
  status_case(fam, fn, "srcsize0_and_nullbuf", [&] { return call(s, 16, NppiSize{0, 0}, off, d, 16, src, m3, a1, NPP_BORDER_REPLICATE, nullptr); });
  status_case(fam, fn, "empty_and_dstep0", [&] { return call(s, 16, src, off, d, 0, NppiSize{0, 0}, m3, a1, NPP_BORDER_REPLICATE, b); });
  status_case(fam, fn, "zerow_and_dstep0", [&] { return call(s, 16, src, off, d, 0, NppiSize{0, 7}, m3, a1, NPP_BORDER_REPLICATE, b); });
  status_case(fam, fn, "zeroh_and_dstep0", [&] { return call(s, 16, src, off, d, 0, NppiSize{7, 0}, m3, a1, NPP_BORDER_REPLICATE, b); });
  // other layouts: alignment + mask/anchor freedoms + 64f acceptance
  status_case(fam, "nppiFilterBoxBorderAdvanced_16u_C3R", "dstep_unaligned_16u_c3", [&] {
    return nppiFilterBoxBorderAdvanced_16u_C3R_Ctx((Npp16u *)s, 16, src, off, (Npp16u *)d, 97,
                                                   src, m3, a1, NPP_BORDER_REPLICATE, b, c);
  });
  status_case(fam, "nppiFilterBoxBorderAdvanced_16u_C3R", "roi_zero_w_16u_c3", [&] {
    return nppiFilterBoxBorderAdvanced_16u_C3R_Ctx((Npp16u *)s, 16, src, off, (Npp16u *)d, 16,
                                                   NppiSize{0, 7}, m3, a1, NPP_BORDER_REPLICATE, b, c);
  });
  status_case(fam, "nppiFilterBoxBorderAdvanced_32f_C4R", "dstep_unaligned_32f_c4", [&] {
    return nppiFilterBoxBorderAdvanced_32f_C4R_Ctx((Npp32f *)s, 128, src, off, (Npp32f *)d, 132,
                                                   NppiSize{8, 6}, NppiSize{11, 9}, NppiPoint{5, 4},
                                                   NPP_BORDER_REPLICATE, b, c);
  });
  status_case(fam, "nppiFilterBoxBorderAdvanced_64f_C1R", "accept_64f", [&] {
    return nppiFilterBoxBorderAdvanced_64f_C1R_Ctx((Npp64f *)s, 16, src, off, (Npp64f *)d, 64,
                                                   NppiSize{2, 2}, m3, a1, NPP_BORDER_REPLICATE, b, c);
  });
  status_case(fam, "nppiFilterBoxBorderAdvanced_64f_C1R", "border_mirror_64f", [&] {
    return nppiFilterBoxBorderAdvanced_64f_C1R_Ctx((Npp64f *)s, 16, src, off, (Npp64f *)d, 64,
                                                   NppiSize{2, 2}, m3, a1, NPP_BORDER_MIRROR, b, c);
  });
}
