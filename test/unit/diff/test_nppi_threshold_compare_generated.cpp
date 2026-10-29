// Differential tests for package p2.3: NPPI threshold and compare operations.
//   Suite: NppiThresholdCompareDiff
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiThresholdCompareDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppiThresholdCompareDiff.*'

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

// Threshold, out of place.  The functor binds the family-specific scalar/array params.
template <typename T, typename F>
void thresh_run(const char *fname, const std::string &key, int ch, int width, int height, int pad,
                uint64_t seed, ValueKind kind, F &&f) {
  const int es = (int)sizeof(T);
  const int px = ch * es;
  const int sStep = align_step(width * px + pad, px);
  const int dStep = align_step(width * px + pad, es);
  const size_t se = (size_t)height * sStep / es;
  const size_t de = (size_t)height * dStep / es;
  std::vector<T> hs = make_uniform<T>(se, seed + 11);
  std::vector<T> hd = make_uniform<T>(de, seed + 29);
  DeviceMemory<T> &s = arena<T>(se);
  DeviceMemory<T> &d = arena<T>(de);
  s.copyFromHost(hs); d.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st = f(s.get(), sStep, d.get(), dStep, NppiSize{width, height}, current_ctx());
  Case c("nppiThresh/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(T));
  c.status(st);
  c.device_output<T>(d.get(), de, kind, 0.0, 0.0);
  c.finish();
}

// Threshold, in place.
template <typename T, typename F>
void thresh_run_i(const char *fname, const std::string &key, int ch, int width, int height, int pad,
                  uint64_t seed, ValueKind kind, F &&f) {
  const int es = (int)sizeof(T);
  const int px = ch * es;
  const int sStep = align_step(width * px + pad, px);
  const size_t se = (size_t)height * sStep / es;
  std::vector<T> hv = make_uniform<T>(se, seed + 11);
  DeviceMemory<T> &d = arena<T>(se);
  d.copyFromHost(hv);
  clear_cuda_error();
  NppStatus st = f(d.get(), sStep, NppiSize{width, height}, current_ctx());
  Case c("nppiThresh/" + key, fname);
  c.input(hv.data(), hv.size() * sizeof(T));
  c.status(st);
  c.device_output<T>(d.get(), se, kind, 0.0, 0.0);
  c.finish();
}

// Compare mask output.
template <typename T, typename F>
void cmp_run(const char *fname, const std::string &key, int ch, int width, int height, int pad,
             uint64_t seed, F &&f) {
  const int es = (int)sizeof(T);
  const int px = ch * es;
  const int sStep = align_step(width * px + pad, px);
  const int dStep = align_step(width + pad, 1);
  const size_t se = (size_t)height * sStep / es;
  const size_t de = (size_t)height * dStep;
  std::vector<T> h1 = make_uniform<T>(se, seed + 11);
  std::vector<T> h2 = make_uniform<T>(se, seed + 29);
  std::vector<Npp8u> hd = make_uniform<Npp8u>(de, seed + 37);
  DeviceMemory<T> &a1 = arena<T>(se);
  DeviceMemory<T> &a2 = arena<T>(se);
  DeviceMemory<Npp8u> &d = arena<Npp8u>(de);
  a1.copyFromHost(h1); a2.copyFromHost(h2); d.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st = f(a1.get(), sStep, a2.get(), sStep, d.get(), dStep, NppiSize{width, height}, current_ctx());
  Case c("nppiCmp/" + key, fname);
  c.input(h1.data(), h1.size() * sizeof(T));
  c.status(st);
  c.device_output<Npp8u>(d.get(), de, ValueKind::U8, 0.0, 0.0);
  c.finish();
}

}  // namespace

class NppiThresholdCompareDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

// ===== nppiThreshold =======
TEST_F(NppiThresholdCompareDiff, Thresh_Thr) {
  { // 8u C1 17x13
    thresh_run<Npp8u>("nppiThreshold_8u_C1R", "Thr/8u/C1/17x13", 1, 17, 13, 16, 2000, ValueKind::U8,
      [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiThreshold_8u_C1R_Ctx(s, ss, d, ds, o, 100, NPP_CMP_GREATER, c); });
  }
  { // 16s C3 64x32
    static const Npp16s thr[3] = {1000, 3000, -2000};
    thresh_run<Npp16s>("nppiThreshold_16s_C3R", "Thr/16s/C3/64x32", 3, 64, 32, 32, 2001, ValueKind::I16,
      [](const Npp16s *s, int ss, Npp16s *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiThreshold_16s_C3R_Ctx(s, ss, d, ds, o, thr, NPP_CMP_GREATER, c); });
  }
  { // 32f AC4 1x1 alpha
    static const Npp32f thr[3] = {-0.5f, 0.0f, 0.5f};
    thresh_run<Npp32f>("nppiThreshold_32f_AC4R", "Thr/32f/AC4/1x1", 4, 1, 1, 16, 2002, ValueKind::F32,
      [](const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiThreshold_32f_AC4R_Ctx(s, ss, d, ds, o, thr, NPP_CMP_GREATER, c); });
  }
  { // 16u AC4 in place (IR)
    static const Npp16u thr[3] = {100, 20000, 40000};
    thresh_run_i<Npp16u>("nppiThreshold_16u_AC4IR", "Thr/16u/AC4IR/64x32", 4, 64, 32, 32, 2003, ValueKind::U16,
      [](Npp16u *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiThreshold_16u_AC4IR_Ctx(d, ds, o, thr, NPP_CMP_GREATER, c); });
  }
}
// ===== nppiThreshold_GT =======
TEST_F(NppiThresholdCompareDiff, Thresh_ThrGt) {
  { // 8u C1 17x13
    thresh_run<Npp8u>("nppiThreshold_GT_8u_C1R", "ThrGt/8u/C1/17x13", 1, 17, 13, 16, 2010, ValueKind::U8,
      [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiThreshold_GT_8u_C1R_Ctx(s, ss, d, ds, o, 100, c); });
  }
  { // 16s C3 64x32
    static const Npp16s thr[3] = {1000, 3000, -2000};
    thresh_run<Npp16s>("nppiThreshold_GT_16s_C3R", "ThrGt/16s/C3/64x32", 3, 64, 32, 32, 2011, ValueKind::I16,
      [](const Npp16s *s, int ss, Npp16s *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiThreshold_GT_16s_C3R_Ctx(s, ss, d, ds, o, thr, c); });
  }
  { // 32f AC4 1x1 alpha
    static const Npp32f thr[3] = {-0.5f, 0.0f, 0.5f};
    thresh_run<Npp32f>("nppiThreshold_GT_32f_AC4R", "ThrGt/32f/AC4/1x1", 4, 1, 1, 16, 2012, ValueKind::F32,
      [](const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiThreshold_GT_32f_AC4R_Ctx(s, ss, d, ds, o, thr, c); });
  }
  { // 16u AC4 in place (IR)
    static const Npp16u thr[3] = {100, 20000, 40000};
    thresh_run_i<Npp16u>("nppiThreshold_GT_16u_AC4IR", "ThrGt/16u/AC4IR/64x32", 4, 64, 32, 32, 2013, ValueKind::U16,
      [](Npp16u *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiThreshold_GT_16u_AC4IR_Ctx(d, ds, o, thr, c); });
  }
}
// ===== nppiThreshold_LT =======
TEST_F(NppiThresholdCompareDiff, Thresh_ThrLt) {
  { // 8u C1 17x13
    thresh_run<Npp8u>("nppiThreshold_LT_8u_C1R", "ThrLt/8u/C1/17x13", 1, 17, 13, 16, 2020, ValueKind::U8,
      [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiThreshold_LT_8u_C1R_Ctx(s, ss, d, ds, o, 100, c); });
  }
  { // 16s C3 64x32
    static const Npp16s thr[3] = {1000, 3000, -2000};
    thresh_run<Npp16s>("nppiThreshold_LT_16s_C3R", "ThrLt/16s/C3/64x32", 3, 64, 32, 32, 2021, ValueKind::I16,
      [](const Npp16s *s, int ss, Npp16s *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiThreshold_LT_16s_C3R_Ctx(s, ss, d, ds, o, thr, c); });
  }
  { // 32f AC4 1x1 alpha
    static const Npp32f thr[3] = {-0.5f, 0.0f, 0.5f};
    thresh_run<Npp32f>("nppiThreshold_LT_32f_AC4R", "ThrLt/32f/AC4/1x1", 4, 1, 1, 16, 2022, ValueKind::F32,
      [](const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiThreshold_LT_32f_AC4R_Ctx(s, ss, d, ds, o, thr, c); });
  }
  { // 16u AC4 in place (IR)
    static const Npp16u thr[3] = {100, 20000, 40000};
    thresh_run_i<Npp16u>("nppiThreshold_LT_16u_AC4IR", "ThrLt/16u/AC4IR/64x32", 4, 64, 32, 32, 2023, ValueKind::U16,
      [](Npp16u *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiThreshold_LT_16u_AC4IR_Ctx(d, ds, o, thr, c); });
  }
}
// ===== nppiThreshold_GTVal =======
TEST_F(NppiThresholdCompareDiff, Thresh_ThrGtVal) {
  { // 8u C1 17x13
    thresh_run<Npp8u>("nppiThreshold_GTVal_8u_C1R", "ThrGtVal/8u/C1/17x13", 1, 17, 13, 16, 2030, ValueKind::U8,
      [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiThreshold_GTVal_8u_C1R_Ctx(s, ss, d, ds, o, 100, 200, c); });
  }
  { // 16s C3 64x32
    static const Npp16s thr[3] = {1000, 3000, -2000};
    static const Npp16s val[3] = {7, 250, -100};
    thresh_run<Npp16s>("nppiThreshold_GTVal_16s_C3R", "ThrGtVal/16s/C3/64x32", 3, 64, 32, 32, 2031, ValueKind::I16,
      [](const Npp16s *s, int ss, Npp16s *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiThreshold_GTVal_16s_C3R_Ctx(s, ss, d, ds, o, thr, val, c); });
  }
  { // 32f AC4 1x1 alpha
    static const Npp32f thr[3] = {-0.5f, 0.0f, 0.5f};
    static const Npp32f val[3] = {0.25f, -0.25f, 1.0f};
    thresh_run<Npp32f>("nppiThreshold_GTVal_32f_AC4R", "ThrGtVal/32f/AC4/1x1", 4, 1, 1, 16, 2032, ValueKind::F32,
      [](const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiThreshold_GTVal_32f_AC4R_Ctx(s, ss, d, ds, o, thr, val, c); });
  }
  { // 16u AC4 in place (IR)
    static const Npp16u thr[3] = {100, 20000, 40000};
    static const Npp16u val[3] = {7, 8, 9};
    thresh_run_i<Npp16u>("nppiThreshold_GTVal_16u_AC4IR", "ThrGtVal/16u/AC4IR/64x32", 4, 64, 32, 32, 2033, ValueKind::U16,
      [](Npp16u *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiThreshold_GTVal_16u_AC4IR_Ctx(d, ds, o, thr, val, c); });
  }
}
// ===== nppiThreshold_LTVal =======
TEST_F(NppiThresholdCompareDiff, Thresh_ThrLtVal) {
  { // 8u C1 17x13
    thresh_run<Npp8u>("nppiThreshold_LTVal_8u_C1R", "ThrLtVal/8u/C1/17x13", 1, 17, 13, 16, 2040, ValueKind::U8,
      [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiThreshold_LTVal_8u_C1R_Ctx(s, ss, d, ds, o, 100, 200, c); });
  }
  { // 16s C3 64x32
    static const Npp16s thr[3] = {1000, 3000, -2000};
    static const Npp16s val[3] = {7, 250, -100};
    thresh_run<Npp16s>("nppiThreshold_LTVal_16s_C3R", "ThrLtVal/16s/C3/64x32", 3, 64, 32, 32, 2041, ValueKind::I16,
      [](const Npp16s *s, int ss, Npp16s *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiThreshold_LTVal_16s_C3R_Ctx(s, ss, d, ds, o, thr, val, c); });
  }
  { // 32f AC4 1x1 alpha
    static const Npp32f thr[3] = {-0.5f, 0.0f, 0.5f};
    static const Npp32f val[3] = {0.25f, -0.25f, 1.0f};
    thresh_run<Npp32f>("nppiThreshold_LTVal_32f_AC4R", "ThrLtVal/32f/AC4/1x1", 4, 1, 1, 16, 2042, ValueKind::F32,
      [](const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiThreshold_LTVal_32f_AC4R_Ctx(s, ss, d, ds, o, thr, val, c); });
  }
  { // 16u AC4 in place (IR)
    static const Npp16u thr[3] = {100, 20000, 40000};
    static const Npp16u val[3] = {7, 8, 9};
    thresh_run_i<Npp16u>("nppiThreshold_LTVal_16u_AC4IR", "ThrLtVal/16u/AC4IR/64x32", 4, 64, 32, 32, 2043, ValueKind::U16,
      [](Npp16u *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiThreshold_LTVal_16u_AC4IR_Ctx(d, ds, o, thr, val, c); });
  }
}
// ===== nppiThreshold_Val =======
TEST_F(NppiThresholdCompareDiff, Thresh_ThrVal) {
  { // 8u C1 17x13
    thresh_run<Npp8u>("nppiThreshold_Val_8u_C1R", "ThrVal/8u/C1/17x13", 1, 17, 13, 16, 2050, ValueKind::U8,
      [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiThreshold_Val_8u_C1R_Ctx(s, ss, d, ds, o, 100, 200, NPP_CMP_LESS, c); });
  }
  { // 16s C3 64x32
    static const Npp16s thr[3] = {1000, 3000, -2000};
    static const Npp16s val[3] = {7, 250, -100};
    thresh_run<Npp16s>("nppiThreshold_Val_16s_C3R", "ThrVal/16s/C3/64x32", 3, 64, 32, 32, 2051, ValueKind::I16,
      [](const Npp16s *s, int ss, Npp16s *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiThreshold_Val_16s_C3R_Ctx(s, ss, d, ds, o, thr, val, NPP_CMP_LESS, c); });
  }
  { // 32f AC4 1x1 alpha
    static const Npp32f thr[3] = {-0.5f, 0.0f, 0.5f};
    static const Npp32f val[3] = {0.25f, -0.25f, 1.0f};
    thresh_run<Npp32f>("nppiThreshold_Val_32f_AC4R", "ThrVal/32f/AC4/1x1", 4, 1, 1, 16, 2052, ValueKind::F32,
      [](const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiThreshold_Val_32f_AC4R_Ctx(s, ss, d, ds, o, thr, val, NPP_CMP_LESS, c); });
  }
  { // 16u AC4 in place (IR)
    static const Npp16u thr[3] = {100, 20000, 40000};
    static const Npp16u val[3] = {7, 8, 9};
    thresh_run_i<Npp16u>("nppiThreshold_Val_16u_AC4IR", "ThrVal/16u/AC4IR/64x32", 4, 64, 32, 32, 2053, ValueKind::U16,
      [](Npp16u *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiThreshold_Val_16u_AC4IR_Ctx(d, ds, o, thr, val, NPP_CMP_LESS, c); });
  }
}
// ===== nppiThreshold_LTValGTVal =======
TEST_F(NppiThresholdCompareDiff, Thresh_ThrLtValGtVal) {
  { // 8u C1 17x13
    thresh_run<Npp8u>("nppiThreshold_LTValGTVal_8u_C1R", "ThrLtValGtVal/8u/C1/17x13", 1, 17, 13, 16, 2060, ValueKind::U8,
      [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiThreshold_LTValGTVal_8u_C1R_Ctx(s, ss, d, ds, o, 50, 90, 150, 180, c); });
  }
  { // 16s C3 64x32
    static const Npp16s tlt[3] = {100, 2000, 4000};
    static const Npp16s vlt[3] = {1, 2, 3};
    static const Npp16s tgt[3] = {8000, 12000, 16000};
    static const Npp16s vgt[3] = {9, 8, 7};
    thresh_run<Npp16s>("nppiThreshold_LTValGTVal_16s_C3R", "ThrLtValGtVal/16s/C3/64x32", 3, 64, 32, 32, 2061, ValueKind::I16,
      [](const Npp16s *s, int ss, Npp16s *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiThreshold_LTValGTVal_16s_C3R_Ctx(s, ss, d, ds, o, tlt, vlt, tgt, vgt, c); });
  }
  { // 32f AC4 1x1 alpha
    static const Npp32f vgt[3] = {0.1f, 0.2f, 0.3f}; static const Npp32f vlt[3] = {0.4f, 0.5f, 0.6f};
    static const Npp32f tlt[3] = {-1.0f, 0.0f, 1.0f};
    static const Npp32f tgt[3] = {2.0f, 3.0f, 4.0f};
    thresh_run<Npp32f>("nppiThreshold_LTValGTVal_32f_AC4R", "ThrLtValGtVal/32f/AC4/1x1", 4, 1, 1, 16, 2062, ValueKind::F32,
      [](const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiThreshold_LTValGTVal_32f_AC4R_Ctx(s, ss, d, ds, o, tlt, vlt, tgt, vgt, c); });
  }
  { // 16u AC4 in place (IR)
    static const Npp16u tlt[3] = {100, 20000, 40000};
    static const Npp16u vlt[3] = {1, 2, 3};
    static const Npp16u tgt[3] = {50000, 60000, 65535};
    static const Npp16u vgt[3] = {5, 6, 7};
    thresh_run_i<Npp16u>("nppiThreshold_LTValGTVal_16u_AC4IR", "ThrLtValGtVal/16u/AC4IR/64x32", 4, 64, 32, 32, 2063, ValueKind::U16,
      [](Npp16u *d, int ds, NppiSize o, NppStreamContext c) {
        return nppiThreshold_LTValGTVal_16u_AC4IR_Ctx(d, ds, o, tlt, vlt, tgt, vgt, c); });
  }
}
// ===== nppiCompare =======
TEST_F(NppiThresholdCompareDiff, Cmp_Compare) {
  { // 8u C1 17x13 compare NPP_CMP_LESS
    cmp_run<Npp8u>("nppiCompare_8u_C1R", "cmp/NPP_CMP_LESS/C1/17x13", 1, 17, 13, 16, 4000,
      [](const Npp8u *a, int as, const Npp8u *b, int bs, Npp8u *d, int ds, NppiSize o, NppStreamContext c){
        return nppiCompare_8u_C1R_Ctx(a, as, b, bs, d, ds, o, NPP_CMP_LESS, c); });
  }
  { // 8u C1 17x13 compare NPP_CMP_LESS_EQ
    cmp_run<Npp8u>("nppiCompare_8u_C1R", "cmp/NPP_CMP_LESS_EQ/C1/17x13", 1, 17, 13, 16, 4001,
      [](const Npp8u *a, int as, const Npp8u *b, int bs, Npp8u *d, int ds, NppiSize o, NppStreamContext c){
        return nppiCompare_8u_C1R_Ctx(a, as, b, bs, d, ds, o, NPP_CMP_LESS_EQ, c); });
  }
  { // 8u C1 17x13 compare NPP_CMP_EQ
    cmp_run<Npp8u>("nppiCompare_8u_C1R", "cmp/NPP_CMP_EQ/C1/17x13", 1, 17, 13, 16, 4002,
      [](const Npp8u *a, int as, const Npp8u *b, int bs, Npp8u *d, int ds, NppiSize o, NppStreamContext c){
        return nppiCompare_8u_C1R_Ctx(a, as, b, bs, d, ds, o, NPP_CMP_EQ, c); });
  }
  { // 8u C1 17x13 compare NPP_CMP_GREATER_EQ
    cmp_run<Npp8u>("nppiCompare_8u_C1R", "cmp/NPP_CMP_GREATER_EQ/C1/17x13", 1, 17, 13, 16, 4003,
      [](const Npp8u *a, int as, const Npp8u *b, int bs, Npp8u *d, int ds, NppiSize o, NppStreamContext c){
        return nppiCompare_8u_C1R_Ctx(a, as, b, bs, d, ds, o, NPP_CMP_GREATER_EQ, c); });
  }
  { // 8u C1 17x13 compare NPP_CMP_GREATER
    cmp_run<Npp8u>("nppiCompare_8u_C1R", "cmp/NPP_CMP_GREATER/C1/17x13", 1, 17, 13, 16, 4004,
      [](const Npp8u *a, int as, const Npp8u *b, int bs, Npp8u *d, int ds, NppiSize o, NppStreamContext c){
        return nppiCompare_8u_C1R_Ctx(a, as, b, bs, d, ds, o, NPP_CMP_GREATER, c); });
  }
  { // 16u C3 64x32 EQ
    cmp_run<Npp16u>("nppiCompare_16u_C3R", "cmp/EQ/C3/64x32", 3, 64, 32, 32, 4010,
      [](const Npp16u *a, int as, const Npp16u *b, int bs, Npp8u *d, int ds, NppiSize o, NppStreamContext c){
        return nppiCompare_16u_C3R_Ctx(a, as, b, bs, d, ds, o, NPP_CMP_EQ, c); });
  }
  { // 16s C4 17x13 GE
    cmp_run<Npp16s>("nppiCompare_16s_C4R", "cmp/GE/C4/17x13", 4, 17, 13, 16, 4011,
      [](const Npp16s *a, int as, const Npp16s *b, int bs, Npp8u *d, int ds, NppiSize o, NppStreamContext c){
        return nppiCompare_16s_C4R_Ctx(a, as, b, bs, d, ds, o, NPP_CMP_GREATER_EQ, c); });
  }
  { // 32f AC4 1x1 LE
    cmp_run<Npp32f>("nppiCompare_32f_AC4R", "cmp/LE/AC4/1x1", 4, 1, 1, 16, 4012,
      [](const Npp32f *a, int as, const Npp32f *b, int bs, Npp8u *d, int ds, NppiSize o, NppStreamContext c){
        return nppiCompare_32f_AC4R_Ctx(a, as, b, bs, d, ds, o, NPP_CMP_LESS_EQ, c); });
  }
}
// ===== nppiCompareC =======
TEST_F(NppiThresholdCompareDiff, Cmp_CompareC) {
  { // 8u C1 const 17x13
    cmp_run<Npp8u>("nppiCompareC_8u_C1R", "cc/GT/C1/17x13", 1, 17, 13, 16, 4020,
      [](const Npp8u *a, int as, const Npp8u *, int, Npp8u *d, int ds, NppiSize o, NppStreamContext c){
        return nppiCompareC_8u_C1R_Ctx(a, as, 100, d, ds, o, NPP_CMP_GREATER, c); });
  }
  { // 16s C3 const array 64x32
    static const Npp16s cst[3] = {-2000, 5000, 100};
    cmp_run<Npp16s>("nppiCompareC_16s_C3R", "cc/LT/C3/64x32", 3, 64, 32, 32, 4021,
      [](const Npp16s *a, int as, const Npp16s *, int, Npp8u *d, int ds, NppiSize o, NppStreamContext c){
        return nppiCompareC_16s_C3R_Ctx(a, as, cst, d, ds, o, NPP_CMP_LESS, c); });
  }
  { // 16u C4 const array 1x1
    static const Npp16u cst[4] = {100, 200, 300, 400};
    cmp_run<Npp16u>("nppiCompareC_16u_C4R", "cc/EQ/C4/1x1", 4, 1, 1, 16, 4022,
      [](const Npp16u *a, int as, const Npp16u *, int, Npp8u *d, int ds, NppiSize o, NppStreamContext c){
        return nppiCompareC_16u_C4R_Ctx(a, as, cst, d, ds, o, NPP_CMP_EQ, c); });
  }
  { // 32f AC4 const array 17x13
    static const Npp32f cst[4] = {0.5f, -0.5f, 1.5f, 2.0f};
    cmp_run<Npp32f>("nppiCompareC_32f_AC4R", "cc/GE/AC4/17x13", 4, 17, 13, 16, 4023,
      [](const Npp32f *a, int as, const Npp32f *, int, Npp8u *d, int ds, NppiSize o, NppStreamContext c){
        return nppiCompareC_32f_AC4R_Ctx(a, as, cst, d, ds, o, NPP_CMP_GREATER_EQ, c); });
  }
}
// ===== nppiCompareEqualEps / EqualEpsC =======
TEST_F(NppiThresholdCompareDiff, Cmp_EqualEps) {
  { // 32f C1 eps 0.01 17x13
    cmp_run<Npp32f>("nppiCompareEqualEps_32f_C1R", "eps/0.01/C1/17x13", 1, 17, 13, 16, 4030,
      [](const Npp32f *a, int as, const Npp32f *b, int bs, Npp8u *d, int ds, NppiSize o, NppStreamContext c){
        return nppiCompareEqualEps_32f_C1R_Ctx(a, as, b, bs, d, ds, o, 0.01f, c); });
  }
  { // 32f C3 eps 0.0 64x32
    cmp_run<Npp32f>("nppiCompareEqualEps_32f_C3R", "eps/0/C3/64x32", 3, 64, 32, 32, 4031,
      [](const Npp32f *a, int as, const Npp32f *b, int bs, Npp8u *d, int ds, NppiSize o, NppStreamContext c){
        return nppiCompareEqualEps_32f_C3R_Ctx(a, as, b, bs, d, ds, o, 0.0f, c); });
  }
  { // 32f AC4 eps 1.0 17x13
    cmp_run<Npp32f>("nppiCompareEqualEps_32f_AC4R", "eps/1/AC4/17x13", 4, 17, 13, 16, 4032,
      [](const Npp32f *a, int as, const Npp32f *b, int bs, Npp8u *d, int ds, NppiSize o, NppStreamContext c){
        return nppiCompareEqualEps_32f_AC4R_Ctx(a, as, b, bs, d, ds, o, 1.0f, c); });
  }
  { // 32f C4 eps 0.0 1x1
    cmp_run<Npp32f>("nppiCompareEqualEps_32f_C4R", "eps/0/C4/1x1", 4, 1, 1, 16, 4033,
      [](const Npp32f *a, int as, const Npp32f *b, int bs, Npp8u *d, int ds, NppiSize o, NppStreamContext c){
        return nppiCompareEqualEps_32f_C4R_Ctx(a, as, b, bs, d, ds, o, 0.0f, c); });
  }
  { // 32f C1 EqualEpsC const 0.5
    cmp_run<Npp32f>("nppiCompareEqualEpsC_32f_C1R", "epsc/0.5/C1/17x13", 1, 17, 13, 16, 4034,
      [](const Npp32f *a, int as, const Npp32f *, int, Npp8u *d, int ds, NppiSize o, NppStreamContext c){
        return nppiCompareEqualEpsC_32f_C1R_Ctx(a, as, 0.25f, d, ds, o, 0.5f, c); });
  }
  { // 32f C3 EqualEpsC const array 2.0 64x32
    static const Npp32f cst[3] = {0.0f, 1.0f, -1.0f};
    cmp_run<Npp32f>("nppiCompareEqualEpsC_32f_C3R", "epsc/2/C3/64x32", 3, 64, 32, 32, 4035,
      [](const Npp32f *a, int as, const Npp32f *, int, Npp8u *d, int ds, NppiSize o, NppStreamContext c){
        return nppiCompareEqualEpsC_32f_C3R_Ctx(a, as, cst, d, ds, o, 2.0f, c); });
  }
  { // 32f AC4 EqualEpsC const 64x32
    static const Npp32f cst[4] = {0.1f, -0.1f, 1.0f, 0.5f};
    cmp_run<Npp32f>("nppiCompareEqualEpsC_32f_AC4R", "epsc/0/C4/64x32", 4, 64, 32, 32, 4036,
      [](const Npp32f *a, int as, const Npp32f *, int, Npp8u *d, int ds, NppiSize o, NppStreamContext c){
        return nppiCompareEqualEpsC_32f_AC4R_Ctx(a, as, cst, d, ds, o, 0.0f, c); });
  }
}
// ===== Contract / status semantics =======
TEST_F(NppiThresholdCompareDiff, Contract) {
  int seed = 4100;
  { // threshold GT: null pointer
    std::vector<Npp8u> hd = make_uniform<Npp8u>(1024, seed + 1);
    DeviceMemory<Npp8u> &d = arena<Npp8u>(1024); d.copyFromHost(hd);
    clear_cuda_error();
    NppStatus st = nppiThreshold_GT_8u_C1R_Ctx(nullptr, 8, d.get(), 8, NppiSize{4, 4}, 100, current_ctx());
    Case c("nppiThresh/nullPtr", "nppiThreshold_GT_8u_C1R"); c.status(st); c.finish();
  }
  { // threshold GT: negative size
    std::vector<Npp8u> hs = make_uniform<Npp8u>(1024, seed + 11); std::vector<Npp8u> hd = make_uniform<Npp8u>(1024, seed + 21);
    DeviceMemory<Npp8u> &s = arena<Npp8u>(1024); DeviceMemory<Npp8u> &d = arena<Npp8u>(1024);
    s.copyFromHost(hs); d.copyFromHost(hd);
    clear_cuda_error();
    NppStatus st = nppiThreshold_GT_8u_C1R_Ctx(s.get(), 8, d.get(), 8, NppiSize{-4, 4}, 100, current_ctx());
    Case c("nppiThresh/negSize", "nppiThreshold_GT_8u_C1R"); c.status(st); c.finish();
  }
  { // threshold GT: empty ROI (no-op)
    std::vector<Npp8u> hs = make_uniform<Npp8u>(1024, seed + 31); std::vector<Npp8u> hd = make_uniform<Npp8u>(1024, seed + 41);
    DeviceMemory<Npp8u> &s = arena<Npp8u>(1024); DeviceMemory<Npp8u> &d = arena<Npp8u>(1024);
    s.copyFromHost(hs); d.copyFromHost(hd);
    clear_cuda_error();
    NppStatus st = nppiThreshold_GT_8u_C1R_Ctx(s.get(), 8, d.get(), 8, NppiSize{0, 4}, 100, current_ctx());
    Case c("nppiThresh/emptyRoi", "nppiThreshold_GT_8u_C1R"); c.status(st); c.finish();
  }
  { // threshold GT: zero step
    std::vector<Npp8u> hs = make_uniform<Npp8u>(1024, seed + 51); std::vector<Npp8u> hd = make_uniform<Npp8u>(1024, seed + 61);
    DeviceMemory<Npp8u> &s = arena<Npp8u>(1024); DeviceMemory<Npp8u> &d = arena<Npp8u>(1024);
    s.copyFromHost(hs); d.copyFromHost(hd);
    clear_cuda_error();
    NppStatus st = nppiThreshold_GT_8u_C1R_Ctx(s.get(), 0, d.get(), 8, NppiSize{4, 4}, 100, current_ctx());
    Case c("nppiThresh/zeroStep", "nppiThreshold_GT_8u_C1R"); c.status(st); c.finish();
  }
  { // threshold GT: insufficient step
    std::vector<Npp8u> hs = make_uniform<Npp8u>(1024, seed + 71); std::vector<Npp8u> hd = make_uniform<Npp8u>(1024, seed + 81);
    DeviceMemory<Npp8u> &s = arena<Npp8u>(1024); DeviceMemory<Npp8u> &d = arena<Npp8u>(1024);
    s.copyFromHost(hs); d.copyFromHost(hd);
    clear_cuda_error();
    NppStatus st = nppiThreshold_GT_8u_C1R_Ctx(s.get(), 4, d.get(), 8, NppiSize{8, 4}, 100, current_ctx());
    Case c("nppiThresh/shortStep", "nppiThreshold_GT_8u_C1R"); c.status(st); c.finish();
  }
  { // threshold base: illegal comparison operation
    std::vector<Npp16u> hs = make_uniform<Npp16u>(1024, seed + 91); std::vector<Npp16u> hd = make_uniform<Npp16u>(1024, seed + 101);
    DeviceMemory<Npp16u> &s = arena<Npp16u>(1024); DeviceMemory<Npp16u> &d = arena<Npp16u>(1024);
    s.copyFromHost(hs); d.copyFromHost(hd);
    clear_cuda_error();
    NppStatus st = nppiThreshold_16u_C1R_Ctx(s.get(), 8, d.get(), 8, NppiSize{4, 4}, 100, (NppCmpOp)99, current_ctx());
    Case c("nppiThresh/badOp", "nppiThreshold_16u_C1R"); c.status(st); c.finish();
  }
  { // compare: null dst
    std::vector<Npp8u> h = make_uniform<Npp8u>(1024, seed + 111);
    DeviceMemory<Npp8u> &s = arena<Npp8u>(1024); s.copyFromHost(h);
    clear_cuda_error();
    NppStatus st = nppiCompare_8u_C1R_Ctx(s.get(), 8, s.get(), 8, nullptr, 8, NppiSize{4, 4}, NPP_CMP_LESS, current_ctx());
    Case c("nppiCmp/nullDst", "nppiCompare_8u_C1R"); c.status(st); c.finish();
  }
  { // compare: invalid comparison op
    std::vector<Npp8u> h = make_uniform<Npp8u>(1024, seed + 121);
    DeviceMemory<Npp8u> &s = arena<Npp8u>(1024); s.copyFromHost(h);
    clear_cuda_error();
    NppStatus st = nppiCompare_8u_C1R_Ctx(s.get(), 8, s.get(), 8, nullptr, 8, NppiSize{4, 4}, (NppCmpOp)99, current_ctx());
    Case c("nppiCmp/badOp", "nppiCompare_8u_C1R"); c.status(st); c.finish();
  }
}