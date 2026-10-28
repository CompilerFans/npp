// Differential tests for package p2.2b: nppiScale (bit-depth scaling) and
// nppiCopyConstBorder / nppiCopyReplicateBorder / nppiCopyWrapBorder.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiScaleBorderDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppiScaleBorderDiff.*'

#include "npp_diff_harness.h"
#include "npp_test_base.h"

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

// Scale: generic 2D helper; the caller binds hint / nMin-nMax through the functor.
template <typename S, typename D, typename F>
void scale_run(const char *fname, const std::string &key, int width, int height, int channels, int pad,
               uint64_t seed, ValueKind kind, F &&f) {
  const int es = (int)sizeof(D);
  const int sStep = align_step(width * channels * (int)sizeof(S) + pad, (int)sizeof(S));
  const int dStep = align_step(width * channels * es + pad, es);
  const size_t se = (size_t)height * sStep / sizeof(S);
  const size_t de = (size_t)height * dStep / es;
  std::vector<S> hs = make_uniform<S>(se, seed + 11);
  std::vector<D> hd = make_uniform<D>(de, seed + 29);
  DeviceMemory<S> &s = arena<S>(se);
  DeviceMemory<D> &d = arena<D>(de);
  s.copyFromHost(hs); d.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st = f(s.get(), sStep, d.get(), dStep, NppiSize{width, height}, current_ctx());
  Case c("nppiScale/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(S));
  c.status(st);
  c.device_output<D>(d.get(), de, kind, 1e-5, 1e-6);
  c.finish();
}

// Border copy: generic 2D helper; the functor binds the value (scalar or array).
template <typename T, typename F>
void border_run(const char *fname, const std::string &key, int channels, int width, int height, int pad,
                int top, int left, int dw, int dh, const T *bval, uint64_t seed, ValueKind kind, F &&f) {
  const int es = (int)sizeof(T);
  // NPP requires image row steps to be multiples of the pixel size
  // (channels*elemSize); align each row step accordingly.
  const int px = channels * es;
  const int sStep = align_step(width * px + pad, px);
  const int dStep = align_step(dw * px + pad, px);
  const size_t se = (size_t)height * sStep / es;
  const size_t de = (size_t)dh * dStep / es;
  std::vector<T> hs = make_uniform<T>(se, seed + 11);
  std::vector<T> hd = make_uniform<T>(de, seed + 29);
  DeviceMemory<T> &s = arena<T>(se);
  DeviceMemory<T> &d = arena<T>(de);
  s.copyFromHost(hs); d.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st = f(s.get(), sStep, NppiSize{width, height}, d.get(), dStep, NppiSize{dw, dh}, top, left,
                   bval, current_ctx());
  Case c("nppiBorder/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(T));
  c.status(st);
  c.device_output<T>(d.get(), de, kind, 1e-5, 1e-6);
  c.finish();
}

}  // namespace

class NppiScaleBorderDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

// ===== nppiScale: hint families (lower bit depth -> 8u) ============
TEST_F(NppiScaleBorderDiff, Scale_16S8U) {
  scale_run<Npp16s, Npp8u>("nppiScale_16s8u_C1R", "16s8u/C1/1x1", 1, 1, 1, 16, 2000, ValueKind::U8,
      [](const Npp16s *s, int ss, Npp8u *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiScale_16s8u_C1R_Ctx(s, ss, d, ds, o, (NppHintAlgorithm)0, c); });
  scale_run<Npp16s, Npp8u>("nppiScale_16s8u_C3R", "16s8u/C3/17x13", 17, 13, 3, 16, 2001, ValueKind::U8,
      [](const Npp16s *s, int ss, Npp8u *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiScale_16s8u_C3R_Ctx(s, ss, d, ds, o, (NppHintAlgorithm)0, c); });
  scale_run<Npp16s, Npp8u>("nppiScale_16s8u_C4R", "16s8u/C4/64x32", 64, 32, 4, 32, 2002, ValueKind::U8,
      [](const Npp16s *s, int ss, Npp8u *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiScale_16s8u_C4R_Ctx(s, ss, d, ds, o, (NppHintAlgorithm)0, c); });
}

TEST_F(NppiScaleBorderDiff, Scale_16U8U) {
  scale_run<Npp16u, Npp8u>("nppiScale_16u8u_C1R", "16u8u/C1/1x1", 1, 1, 1, 16, 2000, ValueKind::U8,
      [](const Npp16u *s, int ss, Npp8u *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiScale_16u8u_C1R_Ctx(s, ss, d, ds, o, (NppHintAlgorithm)0, c); });
  scale_run<Npp16u, Npp8u>("nppiScale_16u8u_C3R", "16u8u/C3/17x13", 17, 13, 3, 16, 2001, ValueKind::U8,
      [](const Npp16u *s, int ss, Npp8u *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiScale_16u8u_C3R_Ctx(s, ss, d, ds, o, (NppHintAlgorithm)0, c); });
  scale_run<Npp16u, Npp8u>("nppiScale_16u8u_C4R", "16u8u/C4/64x32", 64, 32, 4, 32, 2002, ValueKind::U8,
      [](const Npp16u *s, int ss, Npp8u *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiScale_16u8u_C4R_Ctx(s, ss, d, ds, o, (NppHintAlgorithm)0, c); });
}

TEST_F(NppiScaleBorderDiff, Scale_32S8U) {
  scale_run<Npp32s, Npp8u>("nppiScale_32s8u_C1R", "32s8u/C1/1x1", 1, 1, 1, 16, 2000, ValueKind::U8,
      [](const Npp32s *s, int ss, Npp8u *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiScale_32s8u_C1R_Ctx(s, ss, d, ds, o, (NppHintAlgorithm)0, c); });
  scale_run<Npp32s, Npp8u>("nppiScale_32s8u_C3R", "32s8u/C3/17x13", 17, 13, 3, 16, 2001, ValueKind::U8,
      [](const Npp32s *s, int ss, Npp8u *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiScale_32s8u_C3R_Ctx(s, ss, d, ds, o, (NppHintAlgorithm)0, c); });
  scale_run<Npp32s, Npp8u>("nppiScale_32s8u_C4R", "32s8u/C4/64x32", 64, 32, 4, 32, 2002, ValueKind::U8,
      [](const Npp32s *s, int ss, Npp8u *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiScale_32s8u_C4R_Ctx(s, ss, d, ds, o, (NppHintAlgorithm)0, c); });
}

// ===== nppiScale: raise to higher integer bit depth =================
TEST_F(NppiScaleBorderDiff, Scale_8U16U) {
  scale_run<Npp8u, Npp16u>("nppiScale_8u16u_C1R", "8u16u/C1/1x1", 1, 1, 1, 16, 2100, ValueKind::U16,
      [](const Npp8u *s, int ss, Npp16u *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiScale_8u16u_C1R_Ctx(s, ss, d, ds, o, c); });
  scale_run<Npp8u, Npp16u>("nppiScale_8u16u_C3R", "8u16u/C3/17x13", 17, 13, 3, 16, 2101, ValueKind::U16,
      [](const Npp8u *s, int ss, Npp16u *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiScale_8u16u_C3R_Ctx(s, ss, d, ds, o, c); });
  scale_run<Npp8u, Npp16u>("nppiScale_8u16u_AC4R", "8u16u/AC4/64x32", 64, 32, 4, 32, 2102, ValueKind::U16,
      [](const Npp8u *s, int ss, Npp16u *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiScale_8u16u_AC4R_Ctx(s, ss, d, ds, o, c); });
}

TEST_F(NppiScaleBorderDiff, Scale_8U16S) {
  scale_run<Npp8u, Npp16s>("nppiScale_8u16s_C1R", "8u16s/C1/1x1", 1, 1, 1, 16, 2100, ValueKind::I16,
      [](const Npp8u *s, int ss, Npp16s *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiScale_8u16s_C1R_Ctx(s, ss, d, ds, o, c); });
  scale_run<Npp8u, Npp16s>("nppiScale_8u16s_C3R", "8u16s/C3/17x13", 17, 13, 3, 16, 2101, ValueKind::I16,
      [](const Npp8u *s, int ss, Npp16s *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiScale_8u16s_C3R_Ctx(s, ss, d, ds, o, c); });
  scale_run<Npp8u, Npp16s>("nppiScale_8u16s_AC4R", "8u16s/AC4/64x32", 64, 32, 4, 32, 2102, ValueKind::I16,
      [](const Npp8u *s, int ss, Npp16s *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiScale_8u16s_AC4R_Ctx(s, ss, d, ds, o, c); });
}

TEST_F(NppiScaleBorderDiff, Scale_8U32S) {
  scale_run<Npp8u, Npp32s>("nppiScale_8u32s_C1R", "8u32s/C1/1x1", 1, 1, 1, 16, 2100, ValueKind::I32,
      [](const Npp8u *s, int ss, Npp32s *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiScale_8u32s_C1R_Ctx(s, ss, d, ds, o, c); });
  scale_run<Npp8u, Npp32s>("nppiScale_8u32s_C3R", "8u32s/C3/17x13", 17, 13, 3, 16, 2101, ValueKind::I32,
      [](const Npp8u *s, int ss, Npp32s *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiScale_8u32s_C3R_Ctx(s, ss, d, ds, o, c); });
  scale_run<Npp8u, Npp32s>("nppiScale_8u32s_AC4R", "8u32s/AC4/64x32", 64, 32, 4, 32, 2102, ValueKind::I32,
      [](const Npp8u *s, int ss, Npp32s *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiScale_8u32s_AC4R_Ctx(s, ss, d, ds, o, c); });
}

// ===== nppiScale: 8u<->32f with nMin/nMax ===========================
TEST_F(NppiScaleBorderDiff, Scale_8u32f) {
  const float mm[3][2] = {{0.f, 255.f}, {0.f, 100.f}, {-10.f, 200.f}};
  int seed = 2200;
  for (int i = 0; i < 3; ++i) {
    scale_run<Npp8u, Npp32f>("nppiScale_8u32f_C1R", "8u32f/C1/17x13/m" + std::to_string(i), 17, 13, 1, 16, seed++, ValueKind::F32,
        [&](const Npp8u *s, int ss, Npp32f *d, int ds, NppiSize o, NppStreamContext c) {
          return nppiScale_8u32f_C1R_Ctx(s, ss, d, ds, o, mm[i][0], mm[i][1], c); });
  }
}
TEST_F(NppiScaleBorderDiff, Scale_32f8u) {
  const float mm[3][2] = {{0.f, 255.f}, {0.f, 100.f}, {-10.f, 200.f}};
  int seed = 2210;
  for (int i = 0; i < 3; ++i) {
    scale_run<Npp32f, Npp8u>("nppiScale_32f8u_C3R", "32f8u/C3/17x13/m" + std::to_string(i), 17, 13, 3, 16, seed++, ValueKind::U8,
        [&](const Npp32f *s, int ss, Npp8u *d, int ds, NppiSize o, NppStreamContext c) {
          return nppiScale_32f8u_C3R_Ctx(s, ss, d, ds, o, mm[i][0], mm[i][1], c); });
  }
}

// ===== nppiCopyConstBorder ===========================================
TEST_F(NppiScaleBorderDiff, Border_Const) {
  static const Npp8u b1[1] = {17};
  static const Npp8u b3[3] = {11, 22, 33};
  // 8u C1R: 17x13 src, dst 21x17 (top/left=2, right/bottom=2/2).
  border_run<Npp8u>("nppiCopyConstBorder_8u_C1R", "const/8u/C1/21x17", 1, 17, 13, 16, 2, 2, 21, 17, b1, 3000, ValueKind::U8,
      [](const Npp8u *s, int ss, NppiSize so, Npp8u *d, int ds, NppiSize dso, int t, int l, const Npp8u *b, NppStreamContext c) {
        return nppiCopyConstBorder_8u_C1R_Ctx(s, ss, so, d, ds, dso, t, l, b[0], c); });
  // 8u C3R: 17x13 src, dst 20x16 (top/left=3, right/bottom=0).
  border_run<Npp8u>("nppiCopyConstBorder_8u_C3R", "const/8u/C3/20x16", 3, 17, 13, 16, 3, 3, 20, 16, b3, 3001, ValueKind::U8,
      [](const Npp8u *s, int ss, NppiSize so, Npp8u *d, int ds, NppiSize dso, int t, int l, const Npp8u *b, NppStreamContext c) {
        return nppiCopyConstBorder_8u_C3R_Ctx(s, ss, so, d, ds, dso, t, l, b, c); });
  // 16u C4R: 5x7 src, dst 7x9 (top/left=1, right/bottom=1/1).
  static const Npp16u b4[4] = {1000, 2000, 3000, 4000};
  border_run<Npp16u>("nppiCopyConstBorder_16u_C4R", "const/16u/C4/7x9", 4, 5, 7, 8, 1, 1, 7, 9, b4, 3002, ValueKind::U16,
      [](const Npp16u *s, int ss, NppiSize so, Npp16u *d, int ds, NppiSize dso, int t, int l, const Npp16u *b, NppStreamContext c) {
        return nppiCopyConstBorder_16u_C4R_Ctx(s, ss, so, d, ds, dso, t, l, b, c); });
  // 32f AC4R: alpha preserved.
  static const Npp32f ba[3] = {0.25f, -0.5f, 1.5f};
  border_run<Npp32f>("nppiCopyConstBorder_32f_AC4R", "const/32f/AC4/7x9", 4, 5, 7, 8, 1, 1, 7, 9, ba, 3003, ValueKind::F32,
      [](const Npp32f *s, int ss, NppiSize so, Npp32f *d, int ds, NppiSize dso, int t, int l, const Npp32f *b, NppStreamContext c) {
        return nppiCopyConstBorder_32f_AC4R_Ctx(s, ss, so, d, ds, dso, t, l, b, c); });
  // 32s C1R: no border offset (top=left=0).
  static const Npp32s bs[1] = {-1};
  border_run<Npp32s>("nppiCopyConstBorder_32s_C1R", "const/32s/C1/noborder", 1, 64, 32, 16, 0, 0, 64, 32, bs, 3004, ValueKind::I32,
      [](const Npp32s *s, int ss, NppiSize so, Npp32s *d, int ds, NppiSize dso, int t, int l, const Npp32s *b, NppStreamContext c) {
        return nppiCopyConstBorder_32s_C1R_Ctx(s, ss, so, d, ds, dso, t, l, b[0], c); });
}

TEST_F(NppiScaleBorderDiff, Border_Replicate) {
  border_run<Npp8u>("nppiCopyReplicateBorder_8u_C1R", "rep/8u/C1/21x17", 1, 17, 13, 16, 2, 2, 21, 17, nullptr, 3100, ValueKind::U8,
      [](const Npp8u *s, int ss, NppiSize so, Npp8u *d, int ds, NppiSize dso, int t, int l, const Npp8u *, NppStreamContext c) {
        return nppiCopyReplicateBorder_8u_C1R_Ctx(s, ss, so, d, ds, dso, t, l, c); });
  border_run<Npp16s>("nppiCopyReplicateBorder_16s_C3R", "rep/16s/C3/23x19", 3, 17, 13, 16, 3, 3, 23, 19, nullptr, 3101, ValueKind::I16,
      [](const Npp16s *s, int ss, NppiSize so, Npp16s *d, int ds, NppiSize dso, int t, int l, const Npp16s *, NppStreamContext c) {
        return nppiCopyReplicateBorder_16s_C3R_Ctx(s, ss, so, d, ds, dso, t, l, c); });
  border_run<Npp32f>("nppiCopyReplicateBorder_32f_AC4R", "rep/32f/AC4/7x9", 4, 5, 7, 8, 1, 1, 7, 9, nullptr, 3102, ValueKind::F32,
      [](const Npp32f *s, int ss, NppiSize so, Npp32f *d, int ds, NppiSize dso, int t, int l, const Npp32f *, NppStreamContext c) {
        return nppiCopyReplicateBorder_32f_AC4R_Ctx(s, ss, so, d, ds, dso, t, l, c); });
  border_run<Npp16u>("nppiCopyReplicateBorder_16u_C4R", "rep/16u/C4/noborder", 4, 1, 1, 8, 0, 0, 1, 1, nullptr, 3103, ValueKind::U16,
      [](const Npp16u *s, int ss, NppiSize so, Npp16u *d, int ds, NppiSize dso, int t, int l, const Npp16u *, NppStreamContext c) {
        return nppiCopyReplicateBorder_16u_C4R_Ctx(s, ss, so, d, ds, dso, t, l, c); });
}

TEST_F(NppiScaleBorderDiff, Border_Wrap) {
  border_run<Npp8u>("nppiCopyWrapBorder_8u_C1R", "wrap/8u/C1/21x17", 1, 17, 13, 16, 2, 2, 21, 17, nullptr, 3200, ValueKind::U8,
      [](const Npp8u *s, int ss, NppiSize so, Npp8u *d, int ds, NppiSize dso, int t, int l, const Npp8u *, NppStreamContext c) {
        return nppiCopyWrapBorder_8u_C1R_Ctx(s, ss, so, d, ds, dso, t, l, c); });
  border_run<Npp16u>("nppiCopyWrapBorder_16u_C3R", "wrap/16u/C3/23x19", 3, 17, 13, 16, 3, 3, 23, 19, nullptr, 3201, ValueKind::U16,
      [](const Npp16u *s, int ss, NppiSize so, Npp16u *d, int ds, NppiSize dso, int t, int l, const Npp16u *, NppStreamContext c) {
        return nppiCopyWrapBorder_16u_C3R_Ctx(s, ss, so, d, ds, dso, t, l, c); });
  border_run<Npp32s>("nppiCopyWrapBorder_32s_AC4R", "wrap/32s/AC4/7x9", 4, 5, 7, 8, 1, 1, 7, 9, nullptr, 3202, ValueKind::I32,
      [](const Npp32s *s, int ss, NppiSize so, Npp32s *d, int ds, NppiSize dso, int t, int l, const Npp32s *, NppStreamContext c) {
        return nppiCopyWrapBorder_32s_AC4R_Ctx(s, ss, so, d, ds, dso, t, l, c); });
  border_run<Npp32f>("nppiCopyWrapBorder_32f_C1R", "wrap/32f/C1/noborder", 1, 1, 1, 4, 0, 0, 1, 1, nullptr, 3203, ValueKind::F32,
      [](const Npp32f *s, int ss, NppiSize so, Npp32f *d, int ds, NppiSize dso, int t, int l, const Npp32f *, NppStreamContext c) {
        return nppiCopyWrapBorder_32f_C1R_Ctx(s, ss, so, d, ds, dso, t, l, c); });
}

// ===== Border/Scale contract (status semantics) =======================
TEST_F(NppiScaleBorderDiff, Contract) {
  int seed = 3300;
  { // const border: negative top border
    std::vector<Npp8u> hs = make_uniform<Npp8u>(64, seed + 11);
    std::vector<Npp8u> hd = make_uniform<Npp8u>(128, seed + 29);
    DeviceMemory<Npp8u> &s = arena<Npp8u>(64);
    DeviceMemory<Npp8u> &d = arena<Npp8u>(128);
    s.copyFromHost(hs); d.copyFromHost(hd);
    clear_cuda_error();
    NppStatus st = nppiCopyConstBorder_8u_C1R_Ctx(s.get(), 8, NppiSize{4, 4}, d.get(), 8, NppiSize{8, 8}, -1, 2, 0, current_ctx());
    Case c("nppiBorder/negTop", "nppiCopyConstBorder_8u_C1R"); c.status(st); c.finish();
  }
  { // const border: empty destination ROI
    std::vector<Npp8u> hs = make_uniform<Npp8u>(64, seed + 31);
    std::vector<Npp8u> hd = make_uniform<Npp8u>(128, seed + 41);
    DeviceMemory<Npp8u> &s = arena<Npp8u>(64);
    DeviceMemory<Npp8u> &d = arena<Npp8u>(128);
    s.copyFromHost(hs); d.copyFromHost(hd);
    clear_cuda_error();
    NppStatus st = nppiCopyConstBorder_8u_C1R_Ctx(s.get(), 8, NppiSize{4, 4}, d.get(), 8, NppiSize{0, 8}, 1, 2, 0, current_ctx());
    Case c("nppiBorder/emptyDst", "nppiCopyConstBorder_8u_C1R"); c.status(st); c.finish();
  }
  { // const border: null destination
    std::vector<Npp8u> hs = make_uniform<Npp8u>(64, seed + 51);
    DeviceMemory<Npp8u> &s = arena<Npp8u>(64);
    s.copyFromHost(hs);
    clear_cuda_error();
    NppStatus st = nppiCopyConstBorder_8u_C1R_Ctx(s.get(), 8, NppiSize{4, 4}, nullptr, 8, NppiSize{8, 8}, 1, 2, 0, current_ctx());
    Case c("nppiBorder/nullDst", "nppiCopyConstBorder_8u_C1R"); c.status(st); c.finish();
  }
  { // wrap border: border exceeds destination ROI
    std::vector<Npp8u> hs = make_uniform<Npp8u>(64, seed + 61);
    std::vector<Npp8u> hd = make_uniform<Npp8u>(128, seed + 71);
    DeviceMemory<Npp8u> &s = arena<Npp8u>(64);
    DeviceMemory<Npp8u> &d = arena<Npp8u>(128);
    s.copyFromHost(hs); d.copyFromHost(hd);
    clear_cuda_error();
    NppStatus st = nppiCopyWrapBorder_8u_C1R_Ctx(s.get(), 8, NppiSize{4, 4}, d.get(), 8, NppiSize{6, 6}, 1, 2, current_ctx());
    Case c("nppiBorder/oversize", "nppiCopyWrapBorder_8u_C1R"); c.status(st); c.finish();
  }
  { // replicate border: insufficient dst step
    std::vector<Npp8u> hs = make_uniform<Npp8u>(64, seed + 81);
    std::vector<Npp8u> hd = make_uniform<Npp8u>(128, seed + 91);
    DeviceMemory<Npp8u> &s = arena<Npp8u>(64);
    DeviceMemory<Npp8u> &d = arena<Npp8u>(128);
    s.copyFromHost(hs); d.copyFromHost(hd);
    clear_cuda_error();
    NppStatus st = nppiCopyReplicateBorder_8u_C1R_Ctx(s.get(), 8, NppiSize{4, 4}, d.get(), 4, NppiSize{8, 8}, 1, 2, current_ctx());
    Case c("nppiBorder/shortDstStep", "nppiCopyReplicateBorder_8u_C1R"); c.status(st); c.finish();
  }
  { // scale: nMax <= nMin
    std::vector<Npp8u> hs = make_uniform<Npp8u>(64, seed + 101);
    std::vector<Npp32f> hd = make_uniform<Npp32f>(128, seed + 111);
    DeviceMemory<Npp8u> &s = arena<Npp8u>(64);
    DeviceMemory<Npp32f> &d = arena<Npp32f>(128);
    s.copyFromHost(hs); d.copyFromHost(hd);
    clear_cuda_error();
    NppStatus st = nppiScale_8u32f_C1R_Ctx(s.get(), 8, d.get(), 16, NppiSize{4, 4}, 10.f, 5.f, current_ctx());
    Case c("nppiScale/badirange", "nppiScale_8u32f_C1R"); c.status(st); c.finish();
  }
  { // scale: empty ROI
    std::vector<Npp8u> hs = make_uniform<Npp8u>(64, 999);
    std::vector<Npp16u> hd = make_uniform<Npp16u>(128, 1000);
    DeviceMemory<Npp8u> &s = arena<Npp8u>(64);
    DeviceMemory<Npp16u> &d = arena<Npp16u>(128);
    s.copyFromHost(hs); d.copyFromHost(hd);
    clear_cuda_error();
    NppStatus st = nppiScale_8u16u_C1R_Ctx(s.get(), 8, d.get(), 16, NppiSize{0, 4}, current_ctx());
    Case c("nppiScale/emptyRoi", "nppiScale_8u16u_C1R"); c.status(st); c.finish();
  }
}
