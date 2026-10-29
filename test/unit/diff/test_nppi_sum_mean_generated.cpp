// Differential tests for nppiSum / nppiMean / nppiMean_StdDev (package p3.1a1).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiSumMeanDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppiSumMeanDiff.*'
//
// NVIDIA NPP 13 only exports the _Ctx entry points; the reference matrix uses
// those.
//
// Reduction outputs are device doubles (Npp64f) or Npp64s (8u64s) and are
// captured with device_output into ValueKind::F64 / ValueKind::I64.  The
// buffer-size helpers are only checked for status.

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <cstdint>
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

NppStreamContext current_context() {
  NppStreamContext ctx{};
  NppStatus status = nppGetStreamContext(&ctx);
  EXPECT_EQ(status, NPP_NO_ERROR) << "nppGetStreamContext failed";
  return ctx;
}

inline int align_step(int step, int esize) { return ((step + esize - 1) / esize) * esize; }

inline DeviceMemory<Npp8u> &mask_arena(int width, int height, int maskStep) {
  std::vector<Npp8u> hm((size_t)height * maskStep, 0);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) hm[(size_t)y * maskStep + x] = ((x + y) % 2 == 0) ? 1 : 0;
  DeviceMemory<Npp8u> &d = arena<Npp8u>((size_t)height * maskStep);
  d.copyFromHost(hm);
  return d;
}

inline Npp8u *scratch() { return arena<Npp8u>(1 << 20).get(); }

// Plain (unmasked) C1/C3/C4/AC4 sum or mean.
template <typename D, typename Out, int NCh, int OutCount>
void plain_reduce(const char *fname, const std::string &idBase, int width, int height, int pad, uint64_t seed,
                  ValueKind vk, NppStatus (*f)(const D *, int, NppiSize, Npp8u *, Out *, NppStreamContext),
                  double rtol = 1e-5, double atol = 1e-5) {
  const int se = (int)sizeof(D);
  const int step = align_step(width * NCh * se + pad, se);
  const size_t stepElems = (size_t)step / se;
  const size_t elems = (size_t)height * stepElems;
  std::vector<D> hs = make_uniform<D>(elems, seed);
  DeviceMemory<D> &s = arena<D>(elems);
  DeviceMemory<Out> &o = arena<Out>(OutCount);
  s.copyFromHost(hs);
  Case c(idBase + "_" + std::to_string(width) + "x" + std::to_string(height), fname);
  clear_cuda_error();
  NppStatus st = f(s.get(), step, NppiSize{width, height}, scratch(), o.get(), current_context());
  c.input(hs.data(), hs.size() * sizeof(D));
  c.status(st);
  c.device_output<Out>(o.get(), OutCount, vk, rtol, atol);
  c.finish();
}

// Masked single-channel mean (C1MR).
template <typename D>
void mask1_mean(const char *fname, const std::string &idBase, int width, int height, uint64_t seed,
                NppStatus (*f)(const D *, int, const Npp8u *, int, NppiSize, Npp8u *, Npp64f *, NppStreamContext)) {
  const int se = (int)sizeof(D);
  const int step = align_step(width * se + 4, se);
  const int maskStep = width + 4;
  std::vector<D> hs = make_uniform<D>((size_t)height * (step / se), seed);
  DeviceMemory<D> &s = arena<D>(height * (size_t)(step / se));
  s.copyFromHost(hs);
  DeviceMemory<Npp8u> &md = mask_arena(width, height, maskStep);
  DeviceMemory<Npp64f> &m = arena<Npp64f>(1);
  Case c(idBase + "_" + std::to_string(width) + "x" + std::to_string(height), fname);
  clear_cuda_error();
  NppStatus st = f(s.get(), step, md.get(), maskStep, NppiSize{width, height}, scratch(), m.get(), current_context());
  c.input(hs.data(), hs.size() * sizeof(D));
  c.status(st);
  c.device_output(m.get(), 1, ValueKind::F64);
  c.finish();
}

// Masked three-channel mean, channel selected by nCOI (C3CMR).
template <typename D>
void mask_meancoi(const char *fname, const std::string &idBase, int width, int height, int nCOI, uint64_t seed,
                  NppStatus (*f)(const D *, int, const Npp8u *, int, NppiSize, int, Npp8u *, Npp64f *,
                                 NppStreamContext)) {
  const int se = (int)sizeof(D);
  const int step = align_step(width * 3 * se + 4, se);
  const int maskStep = width + 4;
  std::vector<D> hs = make_uniform<D>((size_t)height * (step / se), seed);
  DeviceMemory<D> &s = arena<D>(height * (size_t)(step / se));
  s.copyFromHost(hs);
  DeviceMemory<Npp8u> &md = mask_arena(width, height, maskStep);
  DeviceMemory<Npp64f> &m = arena<Npp64f>(1);
  Case c(idBase + "_" + std::to_string(width) + "x" + std::to_string(height) + "_c" + std::to_string(nCOI), fname);
  clear_cuda_error();
  NppStatus stt = f(s.get(), step, md.get(), maskStep, NppiSize{width, height}, nCOI, scratch(), m.get(),
                     current_context());
  c.input(hs.data(), hs.size() * sizeof(D));
  c.status(stt);
  c.device_output(m.get(), 1, ValueKind::F64);
  c.finish();
}

// Mean_StdDev single channel (C1R).
template <typename D>
void ms1(const char *fname, const char *idBase, int width, int height, uint64_t seed,
         NppStatus (*f)(const D *, int, NppiSize, Npp8u *, Npp64f *, Npp64f *, NppStreamContext)) {
  const int se = (int)sizeof(D);
  const int step = align_step(width * se + 4, se);
  std::vector<D> hs = make_uniform<D>((size_t)height * (step / se), seed);
  DeviceMemory<D> &s = arena<D>(height * (size_t)(step / se));
  s.copyFromHost(hs);
  DeviceMemory<Npp64f> &m = arena<Npp64f>(1);
  DeviceMemory<Npp64f> &v = arena<Npp64f>(1);
  Case c(std::string(idBase) + "_" + std::to_string(width) + "x" + std::to_string(height), fname);
  clear_cuda_error();
  NppStatus st = f(s.get(), step, NppiSize{width, height}, scratch(), m.get(), v.get(), current_context());
  c.input(hs.data(), hs.size() * sizeof(D));
  c.status(st);
  c.device_output(m.get(), 1, ValueKind::F64);
  c.device_output(v.get(), 1, ValueKind::F64);
  c.finish();
}

// Mean_StdDev three-channel one selected by nCOI (C3CR).
template <typename D>
void ms_coi(const char *fname, const char *idBase, int width, int height, int nCOI, uint64_t seed,
            NppStatus (*f)(const D *, int, NppiSize, int, Npp8u *, Npp64f *, Npp64f *, NppStreamContext)) {
  const int se = (int)sizeof(D);
  const int step = align_step(width * 3 * se + 4, se);
  std::vector<D> hs = make_uniform<D>((size_t)height * (step / se), seed);
  DeviceMemory<D> &s = arena<D>(height * (size_t)(step / se));
  s.copyFromHost(hs);
  DeviceMemory<Npp64f> &m = arena<Npp64f>(1);
  DeviceMemory<Npp64f> &v = arena<Npp64f>(1);
  Case c(std::string(idBase) + "_" + std::to_string(width) + "x" + std::to_string(height) + "_c" +
             std::to_string(nCOI), fname);
  clear_cuda_error();
  NppStatus st = f(s.get(), step, NppiSize{width, height}, nCOI, scratch(), m.get(), v.get(), current_context());
  c.input(hs.data(), hs.size() * sizeof(D));
  c.status(st);
  c.device_output(m.get(), 1, ValueKind::F64);
  c.device_output(v.get(), 1, ValueKind::F64);
  c.finish();
}

// Mean_StdDev masked single channel (C1MR).
template <typename D>
void ms_m1(const char *fname, const char *idBase, int width, int height, uint64_t seed,
           NppStatus (*f)(const D *, int, const Npp8u *, int, NppiSize, Npp8u *, Npp64f *, Npp64f *, NppStreamContext)) {
  const int se = (int)sizeof(D);
  const int step = align_step(width * se + 4, se);
  const int maskStep = width + 4;
  std::vector<D> hs = make_uniform<D>((size_t)height * (step / se), seed);
  DeviceMemory<D> &s = arena<D>(height * (size_t)(step / se));
  s.copyFromHost(hs);
  DeviceMemory<Npp8u> &md = mask_arena(width, height, maskStep);
  DeviceMemory<Npp64f> &m = arena<Npp64f>(1);
  DeviceMemory<Npp64f> &v = arena<Npp64f>(1);
  Case c(std::string(idBase) + "_" + std::to_string(width) + "x" + std::to_string(height), fname);
  clear_cuda_error();
  NppStatus st = f(s.get(), step, md.get(), maskStep, NppiSize{width, height}, scratch(), m.get(), v.get(), current_context());
  c.input(hs.data(), hs.size() * sizeof(D));
  c.status(st);
  c.device_output(m.get(), 1, ValueKind::F64);
  c.device_output(v.get(), 1, ValueKind::F64);
  c.finish();
}

// Mean_StdDev masked three-channel one selected by nCOI (C3CMR).
template <typename D>
void ms_mc(const char *fname, const char *idBase, int width, int height, int nCOI, uint64_t seed,
           NppStatus (*f)(const D *, int, const Npp8u *, int, NppiSize, int, Npp8u *, Npp64f *, Npp64f *, NppStreamContext)) {
  const int se = (int)sizeof(D);
  const int step = align_step(width * 3 * se + 4, se);
  const int maskStep = width + 4;
  std::vector<D> hs = make_uniform<D>((size_t)height * (step / se), seed);
  DeviceMemory<D> &s = arena<D>(height * (size_t)(step / se));
  s.copyFromHost(hs);
  DeviceMemory<Npp8u> &md = mask_arena(width, height, maskStep);
  DeviceMemory<Npp64f> &m = arena<Npp64f>(1);
  DeviceMemory<Npp64f> &v = arena<Npp64f>(1);
  Case c(std::string(idBase) + "_" + std::to_string(width) + "x" + std::to_string(height) + "_c" +
             std::to_string(nCOI), fname);
  clear_cuda_error();
  NppStatus st = f(s.get(), step, md.get(), maskStep, NppiSize{width, height}, nCOI, scratch(), m.get(), v.get(), current_context());
  c.input(hs.data(), hs.size() * sizeof(D));
  c.status(st);
  c.device_output(m.get(), 1, ValueKind::F64);
  c.device_output(v.get(), 1, ValueKind::F64);
  c.finish();
}

}  // namespace

class NppiSumMeanDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

// ============================ nppiSum =======================================

TEST_F(NppiSumMeanDiff, Sum_8u_C1R) {
  plain_reduce<Npp8u, Npp64f, 1, 1>("nppiSum_8u_C1R_Ctx", "sum/8u/C1R", 17, 13, 16, 0x811, ValueKind::F64,
                                     &nppiSum_8u_C1R_Ctx);
  plain_reduce<Npp8u, Npp64f, 1, 1>("nppiSum_8u_C1R_Ctx", "sum/8u/C1R", 1, 1, 8, 0x812, ValueKind::F64,
                                     &nppiSum_8u_C1R_Ctx);
}
TEST_F(NppiSumMeanDiff, Sum_8u_C3R) {
  plain_reduce<Npp8u, Npp64f, 3, 3>("nppiSum_8u_C3R_Ctx", "sum/8u/C3R", 17, 13, 16, 0x2001, ValueKind::F64,
                                     &nppiSum_8u_C3R_Ctx);
}
TEST_F(NppiSumMeanDiff, Sum_8u_C4R) {
  plain_reduce<Npp8u, Npp64f, 4, 4>("nppiSum_8u_C4R_Ctx", "sum/8u/C4R", 17, 13, 16, 0x2002, ValueKind::F64,
                                     &nppiSum_8u_C4R_Ctx);
}
TEST_F(NppiSumMeanDiff, Sum_8u_AC4R) {
  plain_reduce<Npp8u, Npp64f, 4, 3>("nppiSum_8u_AC4R_Ctx", "sum/8u/AC4R", 17, 13, 16, 0x2003, ValueKind::F64,
                                     &nppiSum_8u_AC4R_Ctx);
}
TEST_F(NppiSumMeanDiff, Sum_8u64s_C1R) {
  plain_reduce<Npp8u, Npp64s, 1, 1>("nppiSum_8u64s_C1R_Ctx", "sum/8u64s/C1R", 17, 13, 16, 0x3001, ValueKind::I64,
                                     &nppiSum_8u64s_C1R_Ctx, 0.0, 0.0);
}
TEST_F(NppiSumMeanDiff, Sum_8u64s_C4R) {
  plain_reduce<Npp8u, Npp64s, 4, 4>("nppiSum_8u64s_C4R_Ctx", "sum/8u64s/C4R", 64, 32, 32, 0x3002, ValueKind::I64,
                                     &nppiSum_8u64s_C4R_Ctx, 0.0, 0.0);
}
TEST_F(NppiSumMeanDiff, Sum_16u_C1R) {
  plain_reduce<Npp16u, Npp64f, 1, 1>("nppiSum_16u_C1R_Ctx", "sum/16u/C1R", 17, 13, 16, 0x4001, ValueKind::F64,
                                     &nppiSum_16u_C1R_Ctx);
}
TEST_F(NppiSumMeanDiff, Sum_16u_AC4R) {
  plain_reduce<Npp16u, Npp64f, 4, 3>("nppiSum_16u_AC4R_Ctx", "sum/16u/AC4R", 17, 13, 16, 0x4002, ValueKind::F64,
                                     &nppiSum_16u_AC4R_Ctx);
}
TEST_F(NppiSumMeanDiff, Sum_16s_C1R) {
  plain_reduce<Npp16s, Npp64f, 1, 1>("nppiSum_16s_C1R_Ctx", "sum/16s/C1R", 17, 13, 16, 0x5001, ValueKind::F64,
                                     &nppiSum_16s_C1R_Ctx);
}
TEST_F(NppiSumMeanDiff, Sum_16s_C4R) {
  plain_reduce<Npp16s, Npp64f, 4, 4>("nppiSum_16s_C4R_Ctx", "sum/16s/C4R", 17, 13, 16, 0x5002, ValueKind::F64,
                                     &nppiSum_16s_C4R_Ctx);
}
TEST_F(NppiSumMeanDiff, Sum_32f_C1R) {
  plain_reduce<Npp32f, Npp64f, 1, 1>("nppiSum_32f_C1R_Ctx", "sum/32f/C1R", 64, 32, 32, 0x6001, ValueKind::F64,
                                     &nppiSum_32f_C1R_Ctx);
}
TEST_F(NppiSumMeanDiff, Sum_32f_AC4R) {
  plain_reduce<Npp32f, Npp64f, 4, 3>("nppiSum_32f_AC4R_Ctx", "sum/32f/AC4R", 17, 13, 16, 0x6002, ValueKind::F64,
                                     &nppiSum_32f_AC4R_Ctx);
}

// ============================ nppiMean =====================================
// mean/8u/C1R_17x13 routes through the pre-existing hand-written impl in
// nppi_mean_stddev.cpp; it already has unit-test coverage, so it is not part of
// this package's differential suite (NVIDIA-parity verified elsewhere).
TEST_F(NppiSumMeanDiff, Mean_8u_C3R) {
  plain_reduce<Npp8u, Npp64f, 3, 3>("nppiMean_8u_C3R_Ctx", "mean/8u/C3R", 17, 13, 16, 0x7002, ValueKind::F64,
                                    &nppiMean_8u_C3R_Ctx);
}
TEST_F(NppiSumMeanDiff, Mean_8u_AC4R) {
  plain_reduce<Npp8u, Npp64f, 4, 3>("nppiMean_8u_AC4R_Ctx", "mean/8u/AC4R", 17, 13, 16, 0x7003, ValueKind::F64,
                                    &nppiMean_8u_AC4R_Ctx);
}
TEST_F(NppiSumMeanDiff, Mean_8u_C1MR) {
  mask1_mean<Npp8u>("nppiMean_8u_C1MR_Ctx", "mean/8u/C1MR", 17, 13, 0x7101, &nppiMean_8u_C1MR_Ctx);
}
TEST_F(NppiSumMeanDiff, Mean_8u_C3CMR) {
  mask_meancoi<Npp8u>("nppiMean_8u_C3CMR_Ctx", "mean/8u/C3CMR", 17, 13, 1, 0x7102, &nppiMean_8u_C3CMR_Ctx);
}
TEST_F(NppiSumMeanDiff, Mean_16u_C1R) {
  plain_reduce<Npp16u, Npp64f, 1, 1>("nppiMean_16u_C1R_Ctx", "mean/16u/C1R", 17, 13, 16, 0x8001, ValueKind::F64,
                                     &nppiMean_16u_C1R_Ctx);
}
TEST_F(NppiSumMeanDiff, Mean_16u_C4R) {
  plain_reduce<Npp16u, Npp64f, 4, 4>("nppiMean_16u_C4R_Ctx", "mean/16u/C4R", 17, 13, 16, 0x8002, ValueKind::F64,
                                     &nppiMean_16u_C4R_Ctx);
}
TEST_F(NppiSumMeanDiff, Mean_16u_C3CMR) {
  mask_meancoi<Npp16u>("nppiMean_16u_C3CMR_Ctx", "mean/16u/C3CMR", 1, 1, 2, 0x8201, &nppiMean_16u_C3CMR_Ctx);
}
TEST_F(NppiSumMeanDiff, Mean_16s_C1R) {
  plain_reduce<Npp16s, Npp64f, 1, 1>("nppiMean_16s_C1R_Ctx", "mean/16s/C1R", 17, 13, 16, 0x8301, ValueKind::F64,
                                     &nppiMean_16s_C1R_Ctx);
}
TEST_F(NppiSumMeanDiff, Mean_32f_C1R) {
  plain_reduce<Npp32f, Npp64f, 1, 1>("nppiMean_32f_C1R_Ctx", "mean/32f/C1R", 17, 13, 16, 0x8401, ValueKind::F64,
                                     &nppiMean_32f_C1R_Ctx);
}
TEST_F(NppiSumMeanDiff, Mean_32f_AC4R) {
  plain_reduce<Npp32f, Npp64f, 4, 3>("nppiMean_32f_AC4R_Ctx", "mean/32f/AC4R", 17, 13, 16, 0x8402, ValueKind::F64,
                                     &nppiMean_32f_AC4R_Ctx);
}
TEST_F(NppiSumMeanDiff, Mean_8s_C1MR) {
  mask1_mean<Npp8s>("nppiMean_8s_C1MR_Ctx", "mean/8s/C1MR", 17, 13, 0x9101, &nppiMean_8s_C1MR_Ctx);
}
TEST_F(NppiSumMeanDiff, Mean_8s_C3CMR) {
  mask_meancoi<Npp8s>("nppiMean_8s_C3CMR_Ctx", "mean/8s/C3CMR", 17, 13, 0, 0x9102, &nppiMean_8s_C3CMR_Ctx);
}

// ============================ nppiMean_StdDev ==============================

TEST_F(NppiSumMeanDiff, MeanStdDev_8u_C3CR) {
  ms_coi<Npp8u>("nppiMean_StdDev_8u_C3CR_Ctx", "msd/8u/C3CR", 17, 13, 1, 0xa002, &nppiMean_StdDev_8u_C3CR_Ctx);
}
TEST_F(NppiSumMeanDiff, MeanStdDev_8u_C3CMR) {
  ms_mc<Npp8u>("nppiMean_StdDev_8u_C3CMR_Ctx", "msd/8u/C3CMR", 17, 13, 2, 0xa003, &nppiMean_StdDev_8u_C3CMR_Ctx);
}
TEST_F(NppiSumMeanDiff, MeanStdDev_16u_C1R) {
  ms1<Npp16u>("nppiMean_StdDev_16u_C1R_Ctx", "msd/16u/C1R", 17, 13, 0xa101, &nppiMean_StdDev_16u_C1R_Ctx);
  ms1<Npp16u>("nppiMean_StdDev_16u_C1R_Ctx", "msd/16u/C1R", 1, 1, 0xa102, &nppiMean_StdDev_16u_C1R_Ctx);
}
TEST_F(NppiSumMeanDiff, MeanStdDev_16u_C3CR) {
  ms_coi<Npp16u>("nppiMean_StdDev_16u_C3CR_Ctx", "msd/16u/C3CR", 17, 13, 0, 0xa103, &nppiMean_StdDev_16u_C3CR_Ctx);
}
TEST_F(NppiSumMeanDiff, MeanStdDev_16u_C1MR) {
  ms_m1<Npp16u>("nppiMean_StdDev_16u_C1MR_Ctx", "msd/16u/C1MR", 17, 13, 0xa104, &nppiMean_StdDev_16u_C1MR_Ctx);
}
TEST_F(NppiSumMeanDiff, MeanStdDev_8s_C1R) {
  ms1<Npp8s>("nppiMean_StdDev_8s_C1R_Ctx", "msd/8s/C1R", 17, 13, 0xa301, &nppiMean_StdDev_8s_C1R_Ctx);
}
TEST_F(NppiSumMeanDiff, MeanStdDev_8s_C3CR) {
  ms_coi<Npp8s>("nppiMean_StdDev_8s_C3CR_Ctx", "msd/8s/C3CR", 17, 13, 1, 0xa302, &nppiMean_StdDev_8s_C3CR_Ctx);
}
TEST_F(NppiSumMeanDiff, MeanStdDev_8s_C3CMR) {
  ms_mc<Npp8s>("nppiMean_StdDev_8s_C3CMR_Ctx", "msd/8s/C3CMR", 17, 13, 1, 0xa303, &nppiMean_StdDev_8s_C3CMR_Ctx);
}

// ============================ buffer-size helpers ==========================

TEST_F(NppiSumMeanDiff, BufferSize_SumGetBufferHostSize) {
  SIZE_TYPE n = 0;
  NppStatus st = nppiSumGetBufferHostSize_8u_C4R_Ctx(NppiSize{17, 13}, &n, current_context());
  Case c("bufferSize/sum_8u_C4R", "nppiSumGetBufferHostSize_8u_C4R_Ctx");
  c.input(&n, sizeof(n));
  c.status(st);
  c.finish();
}
TEST_F(NppiSumMeanDiff, BufferSize_MeanGetBufferHostSize) {
  SIZE_TYPE n = 0;
  NppStatus st = nppiMeanGetBufferHostSize_8u_AC4R_Ctx(NppiSize{17, 13}, &n, current_context());
  Case c("bufferSize/mean_8u_AC4R", "nppiMeanGetBufferHostSize_8u_AC4R_Ctx");
  c.input(&n, sizeof(n));
  c.status(st);
  c.finish();
}
TEST_F(NppiSumMeanDiff, BufferSize_MeanStdDevGetBufferHostSize) {
  SIZE_TYPE n = 0;
  NppStatus st = nppiMeanStdDevGetBufferHostSize_8u_C3CR_Ctx(NppiSize{17, 13}, &n, current_context());
  Case c("bufferSize/msd_8u_C3CR", "nppiMeanStdDevGetBufferHostSize_8u_C3CR_Ctx");
  c.input(&n, sizeof(n));
  c.status(st);
  c.finish();
}

// ============================ non-Ctx adapters ==============================
#if NPP_DIFF_HAS_NONCTX
// Non-Ctx adapters reproduce an existing _Ctx matrix case (NVIDIA records goldens
// only for _Ctx entry points), so reuse the matching golden id and identical
// inputs; the non-Ctx forwarder must produce the same result.
template <typename D, typename Out, int NCh, int OutCount>
void nonctx_reduce(const char *fname, const std::string &idBase, int width, int height, int pad, uint64_t seed,
                   ValueKind vk, NppStatus (*f)(const D *, int, NppiSize, Npp8u *, Out *)) {
  const int se = (int)sizeof(D);
  const int step = align_step(width * NCh * se + pad, se);
  const size_t stepElems = (size_t)step / se;
  const size_t elems = (size_t)height * stepElems;
  std::vector<D> hs = make_uniform<D>(elems, seed);
  DeviceMemory<D> &s = arena<D>(elems);
  DeviceMemory<Out> &o = arena<Out>(OutCount);
  s.copyFromHost(hs);
  Case c(idBase + "_" + std::to_string(width) + "x" + std::to_string(height), fname);
  clear_cuda_error();
  NppStatus st = f(s.get(), step, NppiSize{width, height}, scratch(), o.get());
  c.input(hs.data(), hs.size() * sizeof(D));
  c.status(st);
  c.device_output<Out>(o.get(), OutCount, vk);
  c.finish();
}

TEST_F(NppiSumMeanDiff, NonCtx_Sum_8u_C1R) {
  nonctx_reduce<Npp8u, Npp64f, 1, 1>("nppiSum_8u_C1R", "sum/8u/C1R", 17, 13, 16, 0x811, ValueKind::F64,
                                     &nppiSum_8u_C1R);
}
TEST_F(NppiSumMeanDiff, NonCtx_Mean) {
  nonctx_reduce<Npp16u, Npp64f, 1, 1>("nppiMean_16u_C1R", "mean/16u/C1R", 17, 13, 16, 0x8001, ValueKind::F64,
                                      &nppiMean_16u_C1R);
}
#endif

// ============================ contract ======================================
// Null pointers, empty / negative ROI, step too short.
// The status is captured via Case so the golden records the actual 5090/NVIDIA
// status name and verify compares against that same name. Scenarios that fault
// the NVIDIA device (e.g. out-of-range nCOI for masked mean) are not probed,
// because a device fault would abort recording.
TEST_F(NppiSumMeanDiff, Contract_Sum_8u_C1R) {
  DeviceMemory<Npp8u> &s = arena<Npp8u>(16);
  DeviceMemory<Npp64f> &o = arena<Npp64f>(1);
  NppStreamContext ctx = current_context();
  {
    clear_cuda_error();
    NppStatus st = nppiSum_8u_C1R_Ctx(nullptr, 16, NppiSize{2, 2}, scratch(), o.get(), ctx);
    Case c("nppiSum/8u_C1R/null_src", "nppiSum_8u_C1R");
    c.status(st);
    c.finish();
  }
  {
    clear_cuda_error();
    NppStatus st = nppiSum_8u_C1R_Ctx(s.get(), 16, NppiSize{2, 2}, scratch(), nullptr, ctx);
    Case c("nppiSum/8u_C1R/null_psum", "nppiSum_8u_C1R");
    c.status(st);
    c.finish();
  }
  {
    clear_cuda_error();
    NppStatus st = nppiSum_8u_C1R_Ctx(s.get(), 16, NppiSize{0, 2}, scratch(), o.get(), ctx);
    Case c("nppiSum/8u_C1R/zro_roi", "nppiSum_8u_C1R");
    c.status(st);
    c.finish();
  }
  {
    clear_cuda_error();
    NppStatus st = nppiSum_8u_C1R_Ctx(s.get(), 16, NppiSize{-1, 2}, scratch(), o.get(), ctx);
    Case c("nppiSum/8u_C1R/neg_size", "nppiSum_8u_C1R");
    c.status(st);
    c.finish();
  }
  {
    clear_cuda_error();
    NppStatus st = nppiSum_8u_C1R_Ctx(s.get(), 2, NppiSize{4, 2}, scratch(), o.get(), ctx);
    Case c("nppiSum/8u_C1R/short_step", "nppiSum_8u_C1R");
    c.status(st);
    c.finish();
  }
}

