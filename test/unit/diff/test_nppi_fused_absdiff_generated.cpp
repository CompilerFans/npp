// Differential tests for the fused AbsDiff + Threshold_GTVal entry points (p2.3).
//   Suite: NppiThresholdCompareDiff
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiThresholdCompareDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppiThresholdCompareDiff.*'

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <memory>
#include <vector>
#include <cstdint>

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
  EXPECT_EQ(nppGetStreamContext(&c), NPP_NO_ERROR);
  return c;
}

int align_step(int step, int es) { return ((step + es - 1) / es) * es; }

// Non-inplace fused absdiff+threshold. threshold/value arrays live in host
// memory for the duration of the call (replicated on the GPU by the kernel).
template <typename T, typename F>
void fused_run(const char *fname, const std::string &key, NppDataType dt, NppiChannels chn,
               int ch, int width, int height, int pad, uint64_t seed, ValueKind kind,
               const T *thr, const T *val, F &&f) {
  const int es = (int)sizeof(T);
  const int px = ch * es;
  const int sStep = align_step(width * px + pad, px);
  const int dStep = align_step(width * px + pad, es);
  const size_t se = (size_t)height * sStep / es;
  const size_t de = (size_t)height * dStep / es;
  std::vector<T> h1 = make_uniform<T>(se, seed + 11);
  std::vector<T> h2 = make_uniform<T>(se, seed + 29);
  std::vector<T> hd = make_uniform<T>(de, seed + 37);
  DeviceMemory<T> &a1 = arena<T>(se);
  DeviceMemory<T> &a2 = arena<T>(se);
  DeviceMemory<T> &d = arena<T>(de);
  a1.copyFromHost(h1); a2.copyFromHost(h2); d.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st = f(a1.get(), sStep, a2.get(), sStep, d.get(), dStep, NppiSize{width, height});
  Case c("nppiFused/" + key, fname);
  c.input(h1.data(), h1.size() * sizeof(T));
  c.input(h2.data(), h2.size() * sizeof(T));
  c.status(st);
  c.device_output<T>(d.get(), de, kind, 0.0, 0.0);
  c.finish();
}

// In-place fused absdiff+threshold. pSrcDst is both src1 and dst.
template <typename T, typename F>
void fused_run_i(const char *fname, const std::string &key, NppDataType dt, NppiChannels chn,
                 int ch, int width, int height, int pad, uint64_t seed, ValueKind kind,
                 const T *thr, const T *val, F &&f) {
  const int es = (int)sizeof(T);
  const int px = ch * es;
  const int sStep = align_step(width * px + pad, px);
  const size_t se = (size_t)height * sStep / es;
  std::vector<T> hv = make_uniform<T>(se, seed + 11);
  std::vector<T> h2 = make_uniform<T>(se, seed + 29);
  DeviceMemory<T> &d = arena<T>(se);
  DeviceMemory<T> &a2 = arena<T>(se);
  d.copyFromHost(hv); a2.copyFromHost(h2);
  clear_cuda_error();
  NppStatus st = f(d.get(), sStep, a2.get(), sStep, NppiSize{width, height});
  Case c("nppiFused/" + key, fname);
  c.input(hv.data(), hv.size() * sizeof(T));
  c.input(h2.data(), h2.size() * sizeof(T));
  c.status(st);
  c.device_output<T>(d.get(), se, kind, 0.0, 0.0);
  c.finish();
}

}  // namespace

class NppiFusedAbsDiffDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

TEST_F(NppiFusedAbsDiffDiff, Fused_AbsDiffThrGtVal) {
  { // 8u CH_1 17x13
    Npp8u thr = 60, val = 200;
    fused_run<Npp8u>("nppiFusedAbsDiff_Threshold_GTVal", "fused/8u/CH1/17x13", NPP_8U, NPP_CH_1,
                     1, 17, 13, 16, 5030, ValueKind::U8, &thr, &val,
      [&](const Npp8u *s1, int ss1, const Npp8u *s2, int ss2, Npp8u *d, int ds, NppiSize o) {
        return nppiFusedAbsDiff_Threshold_GTVal_Ctx(NPP_8U, NPP_CH_1, s1, ss1, s2, ss2, d, ds, o,
                                                    &thr, &val, current_ctx()); });
  }
  { // 16u CH_3 64x32
    static const Npp16u thr[3] = {100, 2000, 4000};
    static const Npp16u val[3] = {7, 8, 9};
    fused_run<Npp16u>("nppiFusedAbsDiff_Threshold_GTVal", "fused/16u/CH3/64x32", NPP_16U, NPP_CH_3,
                     3, 64, 32, 32, 5031, ValueKind::U16, thr, val,
      [&](const Npp16u *s1, int ss1, const Npp16u *s2, int ss2, Npp16u *d, int ds, NppiSize o) {
        return nppiFusedAbsDiff_Threshold_GTVal_Ctx(NPP_16U, NPP_CH_3, s1, ss1, s2, ss2, d, ds, o,
                                                    thr, val, current_ctx()); });
  }
  { // 16s CH_1 17x13 (signed absdiff)
    Npp16s thr = 1000, val = -500;
    fused_run<Npp16s>("nppiFusedAbsDiff_Threshold_GTVal", "fused/16s/CH1/17x13", NPP_16S, NPP_CH_1,
                     1, 17, 13, 16, 5032, ValueKind::I16, &thr, &val,
      [&](const Npp16s *s1, int ss1, const Npp16s *s2, int ss2, Npp16s *d, int ds, NppiSize o) {
        return nppiFusedAbsDiff_Threshold_GTVal_Ctx(NPP_16S, NPP_CH_1, s1, ss1, s2, ss2, d, ds, o,
                                                    &thr, &val, current_ctx()); });
  }
  { // 32f CH_A4 1x1 alpha copy-through
    static const Npp32f thr[3] = {0.5f, 1.0f, 1.5f};
    static const Npp32f val[3] = {-1.0f, 2.0f, 0.5f};
    fused_run<Npp32f>("nppiFusedAbsDiff_Threshold_GTVal", "fused/32f/A4/1x1", NPP_32F, NPP_CH_A4,
                     4, 1, 1, 16, 5033, ValueKind::F32, thr, val,
      [&](const Npp32f *s1, int ss1, const Npp32f *s2, int ss2, Npp32f *d, int ds, NppiSize o) {
        return nppiFusedAbsDiff_Threshold_GTVal_Ctx(NPP_32F, NPP_CH_A4, s1, ss1, s2, ss2, d, ds, o,
                                                    thr, val, current_ctx()); });
  }
  { // In-place 8u CH_1 9x7
    Npp8u thr = 30, val = 200;
    fused_run_i<Npp8u>("nppiFusedAbsDiff_Threshold_GTVal_I", "fusedI/8u/CH1/9x7", NPP_8U, NPP_CH_1,
                      1, 9, 7, 16, 5040, ValueKind::U8, &thr, &val,
      [&](Npp8u *d, int ds, const Npp8u *s2, int ss2, NppiSize o) {
        return nppiFusedAbsDiff_Threshold_GTVal_I_Ctx(NPP_8U, NPP_CH_1, d, ds, s2, ss2, o,
                                                      &thr, &val, current_ctx()); });
  }
  { // In-place 16u CH_A4 17x13
    static const Npp16u thr[3] = {100, 2000, 4000};
    static const Npp16u val[3] = {1, 2, 3};
    fused_run_i<Npp16u>("nppiFusedAbsDiff_Threshold_GTVal_I", "fusedI/16u/A4/17x13", NPP_16U, NPP_CH_A4,
                       4, 17, 13, 16, 5041, ValueKind::U16, thr, val,
      [&](Npp16u *d, int ds, const Npp16u *s2, int ss2, NppiSize o) {
        return nppiFusedAbsDiff_Threshold_GTVal_I_Ctx(NPP_16U, NPP_CH_A4, d, ds, s2, ss2, o,
                                                      thr, val, current_ctx()); });
  }
}
