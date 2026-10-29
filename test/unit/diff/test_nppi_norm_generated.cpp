// Differential tests for nppiNorm / nppiNormDiff L1/L2/Inf (package p3.1b1).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   tools/remote5090.sh record 'NppiNormDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppiNormDiff.*'
//
// NVIDIA NPP 13 only exports the _Ctx entry points; the reference matrix uses
// those.  Norm / NormDiff outputs are device doubles (Npp64f) captured with
// ValueKind::F64.  The buffer-size helpers are only checked for status; exact
// bytes are not a contract.

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

// Plain (unmasked) C1/C3/C4/AC4 Norm.  srcCh = channels per source pixel
// (1/3/4), outCh = scalars written (1 / 3 / 3 / 4).  ROI sizes are deliberately
// varied so the 1x1, 17x13 and padded 64x32 shapes are all covered.
template <typename D>
void norm_plain(const char *fname, const std::string &idBase, int width, int height, int pad, int srcCh, int outCh,
                uint64_t seed, NppStatus (*f)(const D *, int, NppiSize, Npp64f *, Npp8u *, NppStreamContext),
                double rtol = 1e-5, double atol = 1e-5) {
  const int se = (int)sizeof(D);
  const int step = align_step(width * srcCh * se + pad, se);
  const size_t stepElems = (size_t)step / se;
  const size_t elems = (size_t)height * stepElems;
  std::vector<D> hs = make_uniform<D>(elems, seed);
  DeviceMemory<D> &s = arena<D>(elems);
  DeviceMemory<Npp64f> &o = arena<Npp64f>(outCh);
  s.copyFromHost(hs);
  Case c(idBase + "_" + std::to_string(width) + "x" + std::to_string(height), fname);
  clear_cuda_error();
  NppStatus st = f(s.get(), step, NppiSize{width, height}, o.get(), scratch(), current_context());
  c.input(hs.data(), hs.size() * sizeof(D));
  c.status(st);
  c.device_output(o.get(), outCh, ValueKind::F64, rtol, atol);
  c.finish();
}

// Two-source NormDiff: norm of (pSrc1 - pSrc2).
template <typename D>
void normdiff_plain(const char *fname, const std::string &idBase, int width, int height, int pad, int srcCh, int outCh,
                    uint64_t seed, NppStatus (*f)(const D *, int, const D *, int, NppiSize, Npp64f *, Npp8u *,
                                                  NppStreamContext),
                    double rtol = 1e-5, double atol = 1e-5) {
  const int se = (int)sizeof(D);
  const int step = align_step(width * srcCh * se + pad, se);
  const size_t stepElems = (size_t)step / se;
  const size_t elems = (size_t)height * stepElems;
  std::vector<D> h1 = make_uniform<D>(elems, seed);
  std::vector<D> h2 = make_uniform<D>(elems, seed ^ 0x9E3779B97F4A7C15ULL);
  DeviceMemory<D> &s1 = arena<D>(elems);
  DeviceMemory<D> &s2 = arena<D>(elems);
  DeviceMemory<Npp64f> &o = arena<Npp64f>(outCh);
  s1.copyFromHost(h1);
  s2.copyFromHost(h2);
  Case c(idBase + "_" + std::to_string(width) + "x" + std::to_string(height), fname);
  clear_cuda_error();
  NppStatus st = f(s1.get(), step, s2.get(), step, NppiSize{width, height}, o.get(), scratch(), current_context());
  c.input(h1.data(), h1.size() * sizeof(D));
  c.input(h2.data(), h2.size() * sizeof(D));
  c.status(st);
  c.device_output(o.get(), outCh, ValueKind::F64, rtol, atol);
  c.finish();
}

// Masked C1MR single-channel norm.
template <typename D>
void norm_mask1(const char *fname, const std::string &idBase, int width, int height, uint64_t seed,
                NppStatus (*f)(const D *, int, const Npp8u *, int, NppiSize, Npp64f *, Npp8u *, NppStreamContext)) {
  const int se = (int)sizeof(D);
  const int step = align_step(width * se + 4, se);
  const int maskStep = width + 4;
  std::vector<D> hs = make_uniform<D>((size_t)height * (step / se), seed);
  DeviceMemory<D> &s = arena<D>(height * (size_t)(step / se));
  s.copyFromHost(hs);
  DeviceMemory<Npp8u> &md = mask_arena(width, height, maskStep);
  DeviceMemory<Npp64f> &o = arena<Npp64f>(1);
  Case c(idBase + "_" + std::to_string(width) + "x" + std::to_string(height), fname);
  clear_cuda_error();
  NppStatus st = f(s.get(), step, md.get(), maskStep, NppiSize{width, height}, o.get(), scratch(), current_context());
  c.input(hs.data(), hs.size() * sizeof(D));
  c.status(st);
  c.device_output(o.get(), 1, ValueKind::F64);
  c.finish();
}

// Masked C3CMR (three-channel) single-channel-by-nCOI norm.
template <typename D>
void norm_maskcoi(const char *fname, const std::string &idBase, int width, int height, int nCOI, uint64_t seed,
                  NppStatus (*f)(const D *, int, const Npp8u *, int, NppiSize, int, Npp64f *, Npp8u *,
                                 NppStreamContext)) {
  const int se = (int)sizeof(D);
  const int step = align_step(width * 3 * se + 4, se);
  const int maskStep = width + 4;
  std::vector<D> hs = make_uniform<D>((size_t)height * (step / se), seed);
  DeviceMemory<D> &s = arena<D>(height * (size_t)(step / se));
  s.copyFromHost(hs);
  DeviceMemory<Npp8u> &md = mask_arena(width, height, maskStep);
  DeviceMemory<Npp64f> &o = arena<Npp64f>(1);
  Case c(idBase + "_" + std::to_string(width) + "x" + std::to_string(height) + "_c" + std::to_string(nCOI), fname);
  clear_cuda_error();
  NppStatus st = f(s.get(), step, md.get(), maskStep, NppiSize{width, height}, nCOI, o.get(), scratch(),
                   current_context());
  c.input(hs.data(), hs.size() * sizeof(D));
  c.status(st);
  c.device_output(o.get(), 1, ValueKind::F64);
  c.finish();
}

// Two-source masked C1MR NormDiff.
template <typename D>
void normdiff_mask1(const char *fname, const std::string &idBase, int width, int height, uint64_t seed,
                    NppStatus (*f)(const D *, int, const D *, int, const Npp8u *, int, NppiSize, Npp64f *, Npp8u *,
                                   NppStreamContext)) {
  const int se = (int)sizeof(D);
  const int step = align_step(width * se + 4, se);
  const int maskStep = width + 4;
  std::vector<D> h1 = make_uniform<D>((size_t)height * (step / se), seed);
  std::vector<D> h2 = make_uniform<D>((size_t)height * (step / se), seed ^ 0xBF58476D1CE4E5B9ULL);
  DeviceMemory<D> &s1 = arena<D>(height * (size_t)(step / se));
  DeviceMemory<D> &s2 = arena<D>(height * (size_t)(step / se));
  s1.copyFromHost(h1);
  s2.copyFromHost(h2);
  DeviceMemory<Npp8u> &md = mask_arena(width, height, maskStep);
  DeviceMemory<Npp64f> &o = arena<Npp64f>(1);
  Case c(idBase + "_" + std::to_string(width) + "x" + std::to_string(height), fname);
  clear_cuda_error();
  NppStatus st =
      f(s1.get(), step, s2.get(), step, md.get(), maskStep, NppiSize{width, height}, o.get(), scratch(), current_context());
  c.input(h1.data(), h1.size() * sizeof(D));
  c.input(h2.data(), h2.size() * sizeof(D));
  c.status(st);
  c.device_output(o.get(), 1, ValueKind::F64);
  c.finish();
}

// Two-source masked C3CMR NormDiff (channel by nCOI).
template <typename D>
void normdiff_maskcoi(const char *fname, const std::string &idBase, int width, int height, int nCOI, uint64_t seed,
                      NppStatus (*f)(const D *, int, const D *, int, const Npp8u *, int, NppiSize, int, Npp64f *,
                                     Npp8u *, NppStreamContext)) {
  const int se = (int)sizeof(D);
  const int step = align_step(width * 3 * se + 4, se);
  const int maskStep = width + 4;
  std::vector<D> h1 = make_uniform<D>((size_t)height * (step / se), seed);
  std::vector<D> h2 = make_uniform<D>((size_t)height * (step / se), seed ^ 0xD1B54A32D192ED03ULL);
  DeviceMemory<D> &s1 = arena<D>(height * (size_t)(step / se));
  DeviceMemory<D> &s2 = arena<D>(height * (size_t)(step / se));
  s1.copyFromHost(h1);
  s2.copyFromHost(h2);
  DeviceMemory<Npp8u> &md = mask_arena(width, height, maskStep);
  DeviceMemory<Npp64f> &o = arena<Npp64f>(1);
  Case c(idBase + "_" + std::to_string(width) + "x" + std::to_string(height) + "_c" + std::to_string(nCOI), fname);
  clear_cuda_error();
  NppStatus st = f(s1.get(), step, s2.get(), step, md.get(), maskStep, NppiSize{width, height}, nCOI, o.get(),
                   scratch(), current_context());
  c.input(h1.data(), h1.size() * sizeof(D));
  c.input(h2.data(), h2.size() * sizeof(D));
  c.status(st);
  c.device_output(o.get(), 1, ValueKind::F64);
  c.finish();
}

}  // namespace

class NppiNormDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

// ============================ nppiNorm L1 ==================================
TEST_F(NppiNormDiff, Norm_L1) {
  norm_plain<Npp8u>("nppiNorm_L1_8u_C1R_Ctx", "norm/L1/8u/C1R", 17, 13, 16, 1, 1, 0x1001, &nppiNorm_L1_8u_C1R_Ctx);
  norm_plain<Npp8u>("nppiNorm_L1_8u_C1R_Ctx", "norm/L1/8u/C1R", 1, 1, 8, 1, 1, 0x1002, &nppiNorm_L1_8u_C1R_Ctx);
  norm_plain<Npp32f>("nppiNorm_L1_32f_C3R_Ctx", "norm/L1/32f/C3R", 17, 13, 16, 3, 3, 0x1003, &nppiNorm_L1_32f_C3R_Ctx);
  norm_plain<Npp16u>("nppiNorm_L1_16u_C4R_Ctx", "norm/L1/16u/C4R", 64, 32, 32, 4, 4, 0x1004, &nppiNorm_L1_16u_C4R_Ctx);
  norm_plain<Npp16s>("nppiNorm_L1_16s_AC4R_Ctx", "norm/L1/16s/AC4R", 17, 13, 16, 4, 3, 0x1005, &nppiNorm_L1_16s_AC4R_Ctx);
}
TEST_F(NppiNormDiff, Norm_L1_Masked) {
  norm_mask1<Npp16u>("nppiNorm_L1_16u_C1MR_Ctx", "norm/L1/16u/C1MR", 17, 13, 0x1101, &nppiNorm_L1_16u_C1MR_Ctx);
  norm_maskcoi<Npp32f>("nppiNorm_L1_32f_C3CMR_Ctx", "norm/L1/32f/C3CMR", 17, 13, 2, 0x1102, &nppiNorm_L1_32f_C3CMR_Ctx);
}

// ============================ nppiNorm L2 ==================================
TEST_F(NppiNormDiff, Norm_L2) {
  norm_plain<Npp8u>("nppiNorm_L2_8u_C1R_Ctx", "norm/L2/8u/C1R", 17, 13, 16, 1, 1, 0x2001, &nppiNorm_L2_8u_C1R_Ctx);
  norm_plain<Npp32f>("nppiNorm_L2_32f_C3R_Ctx", "norm/L2/32f/C3R", 64, 32, 32, 3, 3, 0x2002, &nppiNorm_L2_32f_C3R_Ctx);
  norm_plain<Npp16u>("nppiNorm_L2_16u_C4R_Ctx", "norm/L2/16u/C4R", 17, 13, 16, 4, 4, 0x2003, &nppiNorm_L2_16u_C4R_Ctx);
  norm_plain<Npp16s>("nppiNorm_L2_16s_AC4R_Ctx", "norm/L2/16s/AC4R", 1, 1, 8, 4, 3, 0x2004, &nppiNorm_L2_16s_AC4R_Ctx);
}
TEST_F(NppiNormDiff, Norm_L2_Masked) {
  norm_mask1<Npp8u>("nppiNorm_L2_8u_C1MR_Ctx", "norm/L2/8u/C1MR", 17, 13, 0x2101, &nppiNorm_L2_8u_C1MR_Ctx);
  norm_maskcoi<Npp16u>("nppiNorm_L2_16u_C3CMR_Ctx", "norm/L2/16u/C3CMR", 17, 13, 1, 0x2102, &nppiNorm_L2_16u_C3CMR_Ctx);
}

// ============================ nppiNorm Inf =================================
TEST_F(NppiNormDiff, Norm_Inf) {
  norm_plain<Npp8u>("nppiNorm_Inf_8u_C1R_Ctx", "norm/Inf/8u/C1R", 17, 13, 16, 1, 1, 0x3001, &nppiNorm_Inf_8u_C1R_Ctx);
  norm_plain<Npp8u>("nppiNorm_Inf_8u_C1R_Ctx", "norm/Inf/8u/C1R", 1, 1, 8, 1, 1, 0x3002, &nppiNorm_Inf_8u_C1R_Ctx);
  norm_plain<Npp32s>("nppiNorm_Inf_32s_C1R_Ctx", "norm/Inf/32s/C1R", 17, 13, 16, 1, 1, 0x3003, &nppiNorm_Inf_32s_C1R_Ctx);
  norm_plain<Npp32f>("nppiNorm_Inf_32f_C4R_Ctx", "norm/Inf/32f/C4R", 64, 32, 32, 4, 4, 0x3004, &nppiNorm_Inf_32f_C4R_Ctx);
  norm_plain<Npp16u>("nppiNorm_Inf_16u_AC4R_Ctx", "norm/Inf/16u/AC4R", 17, 13, 16, 4, 3, 0x3005, &nppiNorm_Inf_16u_AC4R_Ctx);
}
TEST_F(NppiNormDiff, Norm_Inf_Masked) {
  norm_mask1<Npp8u>("nppiNorm_Inf_8u_C1MR_Ctx", "norm/Inf/8u/C1MR", 17, 13, 0x3101, &nppiNorm_Inf_8u_C1MR_Ctx);
  norm_maskcoi<Npp32f>("nppiNorm_Inf_32f_C3CMR_Ctx", "norm/Inf/32f/C3CMR", 17, 13, 1, 0x3102, &nppiNorm_Inf_32f_C3CMR_Ctx);
}

// ============================ nppiNormDiff L1 ==============================
TEST_F(NppiNormDiff, NormDiff_L1) {
  normdiff_plain<Npp8u>("nppiNormDiff_L1_8u_C1R_Ctx", "ndiff/L1/8u/C1R", 17, 13, 16, 1, 1, 0x4001,
                        &nppiNormDiff_L1_8u_C1R_Ctx);
  normdiff_plain<Npp32f>("nppiNormDiff_L1_32f_C3R_Ctx", "ndiff/L1/32f/C3R", 17, 13, 16, 3, 3, 0x4002,
                         &nppiNormDiff_L1_32f_C3R_Ctx);
  normdiff_plain<Npp16u>("nppiNormDiff_L1_16u_C4R_Ctx", "ndiff/L1/16u/C4R", 64, 32, 32, 4, 4, 0x4003,
                         &nppiNormDiff_L1_16u_C4R_Ctx);
  normdiff_plain<Npp16s>("nppiNormDiff_L1_16s_AC4R_Ctx", "ndiff/L1/16s/AC4R", 1, 1, 8, 4, 3, 0x4004,
                         &nppiNormDiff_L1_16s_AC4R_Ctx);
}
TEST_F(NppiNormDiff, NormDiff_L1_Masked) {
  normdiff_mask1<Npp16u>("nppiNormDiff_L1_16u_C1MR_Ctx", "ndiff/L1/16u/C1MR", 17, 13, 0x4101,
                         &nppiNormDiff_L1_16u_C1MR_Ctx);
  normdiff_maskcoi<Npp32f>("nppiNormDiff_L1_32f_C3CMR_Ctx", "ndiff/L1/32f/C3CMR", 17, 13, 2, 0x4102,
                           &nppiNormDiff_L1_32f_C3CMR_Ctx);
}

// ============================ nppiNormDiff L2 ==============================
TEST_F(NppiNormDiff, NormDiff_L2) {
  normdiff_plain<Npp8u>("nppiNormDiff_L2_8u_C1R_Ctx", "ndiff/L2/8u/C1R", 17, 13, 16, 1, 1, 0x5001,
                        &nppiNormDiff_L2_8u_C1R_Ctx);
  normdiff_plain<Npp32f>("nppiNormDiff_L2_32f_C3R_Ctx", "ndiff/L2/32f/C3R", 64, 32, 32, 3, 3, 0x5002,
                         &nppiNormDiff_L2_32f_C3R_Ctx);
  normdiff_plain<Npp16u>("nppiNormDiff_L2_16u_C4R_Ctx", "ndiff/L2/16u/C4R", 17, 13, 16, 4, 4, 0x5003,
                         &nppiNormDiff_L2_16u_C4R_Ctx);
  normdiff_plain<Npp16s>("nppiNormDiff_L2_16s_AC4R_Ctx", "ndiff/L2/16s/AC4R", 17, 13, 16, 4, 3, 0x5004,
                         &nppiNormDiff_L2_16s_AC4R_Ctx);
}
TEST_F(NppiNormDiff, NormDiff_L2_Masked) {
  normdiff_mask1<Npp8u>("nppiNormDiff_L2_8u_C1MR_Ctx", "ndiff/L2/8u/C1MR", 17, 13, 0x5101,
                        &nppiNormDiff_L2_8u_C1MR_Ctx);
  normdiff_maskcoi<Npp16u>("nppiNormDiff_L2_16u_C3CMR_Ctx", "ndiff/L2/16u/C3CMR", 17, 13, 1, 0x5102,
                           &nppiNormDiff_L2_16u_C3CMR_Ctx);
}

// ============================ nppiNormDiff Inf =============================
TEST_F(NppiNormDiff, NormDiff_Inf) {
  normdiff_plain<Npp8u>("nppiNormDiff_Inf_8u_C1R_Ctx", "ndiff/Inf/8u/C1R", 17, 13, 16, 1, 1, 0x6001,
                        &nppiNormDiff_Inf_8u_C1R_Ctx);
  normdiff_plain<Npp32f>("nppiNormDiff_Inf_32f_C3R_Ctx", "ndiff/Inf/32f/C3R", 17, 13, 16, 3, 3, 0x6002,
                         &nppiNormDiff_Inf_32f_C3R_Ctx);
  normdiff_plain<Npp16u>("nppiNormDiff_Inf_16u_C4R_Ctx", "ndiff/Inf/16u/C4R", 64, 32, 32, 4, 4, 0x6003,
                         &nppiNormDiff_Inf_16u_C4R_Ctx);
  normdiff_plain<Npp8u>("nppiNormDiff_Inf_8u_AC4R_Ctx", "ndiff/Inf/8u/AC4R", 1, 1, 8, 4, 3, 0x6004,
                        &nppiNormDiff_Inf_8u_AC4R_Ctx);
}
TEST_F(NppiNormDiff, NormDiff_Inf_Masked) {
  normdiff_mask1<Npp8u>("nppiNormDiff_Inf_8u_C1MR_Ctx", "ndiff/Inf/8u/C1MR", 17, 13, 0x6101,
                        &nppiNormDiff_Inf_8u_C1MR_Ctx);
  normdiff_maskcoi<Npp32f>("nppiNormDiff_Inf_32f_C3CMR_Ctx", "ndiff/Inf/32f/C3CMR", 17, 13, 3, 0x6102,
                           &nppiNormDiff_Inf_32f_C3CMR_Ctx);
}

// ============================ buffer-size helpers (status only) ============
TEST_F(NppiNormDiff, BufferSize_Norms) {
  SIZE_TYPE n = 0;
  NppStreamContext ctx = current_context();
  {
    NppStatus st = nppiNormL1GetBufferHostSize_8u_C4R_Ctx(NppiSize{17, 13}, &n, ctx);
    Case c("bufferSize/normL1_8u_C4R", "nppiNormL1GetBufferHostSize_8u_C4R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiNormL2GetBufferHostSize_16u_C1R_Ctx(NppiSize{64, 32}, &n, ctx);
    Case c("bufferSize/normL2_16u_C1R", "nppiNormL2GetBufferHostSize_16u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiNormInfGetBufferHostSize_32f_AC4R_Ctx(NppiSize{17, 13}, &n, ctx);
    Case c("bufferSize/normInf_32f_AC4R", "nppiNormInfGetBufferHostSize_32f_AC4R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiNormDiffL1GetBufferHostSize_16s_C3R_Ctx(NppiSize{17, 13}, &n, ctx);
    Case c("bufferSize/ndiffL1_16s_C3R", "nppiNormDiffL1GetBufferHostSize_16s_C3R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiNormDiffL2GetBufferHostSize_8u_C4R_Ctx(NppiSize{64, 32}, &n, ctx);
    Case c("bufferSize/ndiffL2_8u_C4R", "nppiNormDiffL2GetBufferHostSize_8u_C4R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiNormDiffInfGetBufferHostSize_32f_C1R_Ctx(NppiSize{17, 13}, &n, ctx);
    Case c("bufferSize/ndiffInf_32f_C1R", "nppiNormDiffInfGetBufferHostSize_32f_C1R_Ctx");
    c.status(st); c.finish();
  }
}

// ============================ non-Ctx adapters =============================
#if NPP_DIFF_HAS_NONCTX
TEST_F(NppiNormDiff, NonCtx_Norm) {
  // Reproduce the existing Norm_L1_16u_C4R golden so both entry points agree.
  const int se = (int)sizeof(Npp16u);
  const int step = align_step(64 * 4 * se + 32, se);
  const size_t elems = (size_t)32 * (step / se);
  std::vector<Npp16u> hs = make_uniform<Npp16u>(elems, 0x1004);
  DeviceMemory<Npp16u> &s = arena<Npp16u>(elems);
  DeviceMemory<Npp64f> &o = arena<Npp64f>(4);
  s.copyFromHost(hs);
  Case c("norm/L1/16u/C4R_64x32", "nppiNorm_L1_16u_C4R");  // reuses the golden id above
  clear_cuda_error();
  NppStatus st = nppiNorm_L1_16u_C4R(s.get(), step, NppiSize{64, 32}, o.get(), scratch());
  c.input(hs.data(), hs.size() * sizeof(Npp16u));
  c.status(st);
  c.device_output(o.get(), 4, ValueKind::F64);
  c.finish();
}
#endif

// ============================ contract =====================================
// Null pointers, empty / negative ROI, step too short.  Status is captured via
// Case so the golden records the actual 5090/NVIDIA status name and verify
// compares against that same name.  Scenarios that fault the NVIDIA device are
// not probed (P3.1a2/a3 already identified those).
TEST_F(NppiNormDiff, Contract_Norm) {
  DeviceMemory<Npp8u> &s = arena<Npp8u>(16);
  DeviceMemory<Npp64f> &o = arena<Npp64f>(1);
  NppStreamContext ctx = current_context();
  {
    NppStatus st = nppiNorm_L1_8u_C1R_Ctx(nullptr, 16, NppiSize{2, 2}, o.get(), scratch(), ctx);
    Case c("norm/contract/null_src", "nppiNorm_L1_8u_C1R");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiNorm_L1_8u_C1R_Ctx(s.get(), 16, NppiSize{2, 2}, nullptr, scratch(), ctx);
    Case c("norm/contract/null_out", "nppiNorm_L1_8u_C1R");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiNorm_L1_8u_C1R_Ctx(s.get(), 16, NppiSize{0, 2}, o.get(), scratch(), ctx);
    Case c("norm/contract/zro_roi", "nppiNorm_L1_8u_C1R");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiNorm_L1_8u_C1R_Ctx(s.get(), 16, NppiSize{-1, 2}, o.get(), scratch(), ctx);
    Case c("norm/contract/neg_size", "nppiNorm_L1_8u_C1R");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiNorm_L1_8u_C1R_Ctx(s.get(), 2, NppiSize{4, 2}, o.get(), scratch(), ctx);
    Case c("norm/contract/short_step", "nppiNorm_L1_8u_C1R");
    c.status(st); c.finish();
  }
}

TEST_F(NppiNormDiff, Contract_NormDiff) {
  DeviceMemory<Npp8u> &s1 = arena<Npp8u>(16);
  DeviceMemory<Npp8u> &s2 = arena<Npp8u>(16);
  DeviceMemory<Npp64f> &o = arena<Npp64f>(1);
  NppStreamContext ctx = current_context();
  {
    NppStatus st = nppiNormDiff_Inf_8u_C1R_Ctx(s1.get(), 16, nullptr, 16, NppiSize{2, 2}, o.get(), scratch(), ctx);
    Case c("ndiff/contract/null_src2", "nppiNormDiff_Inf_8u_C1R");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiNormDiff_Inf_8u_C1R_Ctx(nullptr, 16, s2.get(), 16, NppiSize{2, 2}, o.get(), scratch(), ctx);
    Case c("ndiff/contract/null_src1", "nppiNormDiff_Inf_8u_C1R");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiNormDiff_Inf_8u_C1R_Ctx(s1.get(), 16, s2.get(), 16, NppiSize{0, 2}, o.get(), scratch(), ctx);
    Case c("ndiff/contract/zro_roi", "nppiNormDiff_Inf_8u_C1R");
    c.status(st); c.finish();
  }
}
