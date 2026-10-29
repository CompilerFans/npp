// Differential tests for nppiNormRel L1/L2/Inf, nppiDotProd and the
// NormLevel buffer-size helpers (package p3.1b2).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   tools/remote5090.sh record 'NppiNormRelDotProdDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppiNormRelDotProdDiff.*'
//
// NormRel = norm(pSrc1 - pSrc2) / norm(pSrc2); DotProd = per-channel sum of
// pSrc1 * pSrc2.  Both write device doubles (Npp64f), captured with the
// plain two-source helper below.  Buffer-size helpers are only checked for
// status; exact bytes are not a contract.

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

inline Npp8u *scratch() { return arena<Npp8u>(4 << 20).get(); }

// Two-source, per-channel Npp64f output (NormRel and DotProd share this
// signature): C1R writes 1 scalar, C3/C4 write srcCh scalars, AC4 writes
// srcCh-1 (alpha ignored).  id is the function name recorded for golden.
template <typename D>
void twosrc_plain(const char *fname, const std::string &idBase, int width, int height, int pad, int srcCh, int outCh,
                  uint64_t seed, NppStatus (*f)(const D *, int, const D *, int, NppiSize, Npp64f *, Npp8u *,
                                                NppStreamContext),
                  double rtol = 1e-5, double atol = 1e-5) {
  const int se = (int)sizeof(D);
  const int step = align_step(width * srcCh * se + pad, se);
  const size_t elems = (size_t)height * (size_t)(step / se);
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

// Masked C1MR single-channel two-source norm.
template <typename D>
void twosrc_mask1(const char *fname, const std::string &idBase, int width, int height, uint64_t seed,
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

// Two-source masked C3CMR (channel selected by nCOI).
template <typename D>
void twosrc_maskcoi(const char *fname, const std::string &idBase, int width, int height, int nCOI, uint64_t seed,
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

class NppiNormRelDotProdDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

// ============================ nppiNormRel L1 ================================
TEST_F(NppiNormRelDotProdDiff, NormRel_L1) {
  twosrc_plain<Npp8u>("nppiNormRel_L1_8u_C1R_Ctx", "normrel/L1/8u/C1R", 17, 13, 16, 1, 1, 0x101,
                      &nppiNormRel_L1_8u_C1R_Ctx);
  twosrc_plain<Npp8u>("nppiNormRel_L1_8u_C1R_Ctx", "normrel/L1/8u/C1R", 1, 1, 8, 1, 1, 0x102,
                      &nppiNormRel_L1_8u_C1R_Ctx);
  twosrc_plain<Npp32f>("nppiNormRel_L1_32f_C3R_Ctx", "normrel/L1/32f/C3R", 17, 13, 16, 3, 3, 0x103,
                       &nppiNormRel_L1_32f_C3R_Ctx);
  twosrc_plain<Npp16u>("nppiNormRel_L1_16u_C4R_Ctx", "normrel/L1/16u/C4R", 64, 32, 32, 4, 4, 0x104,
                       &nppiNormRel_L1_16u_C4R_Ctx);
  twosrc_plain<Npp16s>("nppiNormRel_L1_16s_AC4R_Ctx", "normrel/L1/16s/AC4R", 17, 13, 16, 4, 3, 0x105,
                       &nppiNormRel_L1_16s_AC4R_Ctx);
}
TEST_F(NppiNormRelDotProdDiff, NormRel_L1_Masked) {
  twosrc_mask1<Npp16u>("nppiNormRel_L1_16u_C1MR_Ctx", "normrel/L1/16u/C1MR", 17, 13, 0x111,
                       &nppiNormRel_L1_16u_C1MR_Ctx);
  twosrc_maskcoi<Npp32f>("nppiNormRel_L1_32f_C3CMR_Ctx", "normrel/L1/32f/C3CMR", 17, 13, 2, 0x112,
                         &nppiNormRel_L1_32f_C3CMR_Ctx);
  twosrc_mask1<Npp8s>("nppiNormRel_L1_8s_C1MR_Ctx", "normrel/L1/8s/C1MR", 64, 32, 0x113, &nppiNormRel_L1_8s_C1MR_Ctx);
  twosrc_maskcoi<Npp8u>("nppiNormRel_L1_8u_C3CMR_Ctx", "normrel/L1/8u/C3CMR", 17, 13, 1, 0x114,
                        &nppiNormRel_L1_8u_C3CMR_Ctx);
}

// ============================ nppiNormRel L2 ================================
TEST_F(NppiNormRelDotProdDiff, NormRel_L2) {
  twosrc_plain<Npp8u>("nppiNormRel_L2_8u_C1R_Ctx", "normrel/L2/8u/C1R", 17, 13, 16, 1, 1, 0x201,
                      &nppiNormRel_L2_8u_C1R_Ctx);
  twosrc_plain<Npp32f>("nppiNormRel_L2_32f_C3R_Ctx", "normrel/L2/32f/C3R", 64, 32, 32, 3, 3, 0x202,
                       &nppiNormRel_L2_32f_C3R_Ctx);
  twosrc_plain<Npp16u>("nppiNormRel_L2_16u_C4R_Ctx", "normrel/L2/16u/C4R", 17, 13, 16, 4, 4, 0x203,
                       &nppiNormRel_L2_16u_C4R_Ctx);
  twosrc_plain<Npp16s>("nppiNormRel_L2_16s_AC4R_Ctx", "normrel/L2/16s/AC4R", 1, 1, 8, 4, 3, 0x204,
                       &nppiNormRel_L2_16s_AC4R_Ctx);
}
TEST_F(NppiNormRelDotProdDiff, NormRel_L2_Masked) {
  twosrc_mask1<Npp8u>("nppiNormRel_L2_8u_C1MR_Ctx", "normrel/L2/8u/C1MR", 17, 13, 0x211, &nppiNormRel_L2_8u_C1MR_Ctx);
  twosrc_maskcoi<Npp16u>("nppiNormRel_L2_16u_C3CMR_Ctx", "normrel/L2/16u/C3CMR", 17, 13, 1, 0x212,
                         &nppiNormRel_L2_16u_C3CMR_Ctx);
  twosrc_mask1<Npp8s>("nppiNormRel_L2_8s_C1MR_Ctx", "normrel/L2/8s/C1MR", 17, 13, 0x213, &nppiNormRel_L2_8s_C1MR_Ctx);
}

// ============================ nppiNormRel Inf ===============================
TEST_F(NppiNormRelDotProdDiff, NormRel_Inf) {
  twosrc_plain<Npp8u>("nppiNormRel_Inf_8u_C1R_Ctx", "normrel/Inf/8u/C1R", 17, 13, 16, 1, 1, 0x301,
                      &nppiNormRel_Inf_8u_C1R_Ctx);
  twosrc_plain<Npp32f>("nppiNormRel_Inf_32f_C3R_Ctx", "normrel/Inf/32f/C3R", 17, 13, 16, 3, 3, 0x302,
                       &nppiNormRel_Inf_32f_C3R_Ctx);
  twosrc_plain<Npp16u>("nppiNormRel_Inf_16u_C4R_Ctx", "normrel/Inf/16u/C4R", 64, 32, 32, 4, 4, 0x303,
                       &nppiNormRel_Inf_16u_C4R_Ctx);
  twosrc_plain<Npp8u>("nppiNormRel_Inf_8u_AC4R_Ctx", "normrel/Inf/8u/AC4R", 1, 1, 8, 4, 3, 0x304,
                      &nppiNormRel_Inf_8u_AC4R_Ctx);
}
TEST_F(NppiNormRelDotProdDiff, NormRel_Inf_Masked) {
  twosrc_mask1<Npp16u>("nppiNormRel_Inf_16u_C1MR_Ctx", "normrel/Inf/16u/C1MR", 17, 13, 0x311,
                       &nppiNormRel_Inf_16u_C1MR_Ctx);
  twosrc_maskcoi<Npp32f>("nppiNormRel_Inf_32f_C3CMR_Ctx", "normrel/Inf/32f/C3CMR", 17, 13, 3, 0x312,
                         &nppiNormRel_Inf_32f_C3CMR_Ctx);
  twosrc_maskcoi<Npp8u>("nppiNormRel_Inf_8u_C3CMR_Ctx", "normrel/Inf/8u/C3CMR", 17, 13, 1, 0x313,
                        &nppiNormRel_Inf_8u_C3CMR_Ctx);
}

// ============================ nppiDotProd ===================================
TEST_F(NppiNormRelDotProdDiff, DotProd_Numerics) {
  twosrc_plain<Npp8u>("nppiDotProd_8u64f_C1R_Ctx", "dotprod/C1/8u", 17, 13, 16, 1, 1, 0x401, &nppiDotProd_8u64f_C1R_Ctx);
  twosrc_plain<Npp8s>("nppiDotProd_8s64f_C3R_Ctx", "dotprod/C3/8s", 17, 13, 16, 3, 3, 0x402, &nppiDotProd_8s64f_C3R_Ctx);
  twosrc_plain<Npp16u>("nppiDotProd_16u64f_C4R_Ctx", "dotprod/C4/16u", 64, 32, 32, 4, 4, 0x403, &nppiDotProd_16u64f_C4R_Ctx);
  twosrc_plain<Npp16s>("nppiDotProd_16s64f_AC4R_Ctx", "dotprod/AC4/16s", 17, 13, 16, 4, 3, 0x404, &nppiDotProd_16s64f_AC4R_Ctx);
  twosrc_plain<Npp32u>("nppiDotProd_32u64f_C1R_Ctx", "dotprod/C1/32u", 1, 1, 8, 1, 1, 0x405, &nppiDotProd_32u64f_C1R_Ctx);
  twosrc_plain<Npp32s>("nppiDotProd_32s64f_C3R_Ctx", "dotprod/C3/32s", 64, 32, 32, 3, 3, 0x406, &nppiDotProd_32s64f_C3R_Ctx);
  twosrc_plain<Npp32f>("nppiDotProd_32f64f_C4R_Ctx", "dotprod/C4/32f", 17, 13, 16, 4, 4, 0x407, &nppiDotProd_32f64f_C4R_Ctx);
}

// ============================ buffer-size helpers (status only) =============
TEST_F(NppiNormRelDotProdDiff, BufferSize_Helpers) {
  SIZE_TYPE n = 0;
  NppStreamContext ctx = current_context();
  {
    NppStatus st = nppiNormRelL1GetBufferHostSize_8u_C4R_Ctx(NppiSize{17, 13}, &n, ctx);
    Case c("bufferSize/normRelL1_8u_C4R", "nppiNormRelL1GetBufferHostSize_8u_C4R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiNormRelL2GetBufferHostSize_16u_C1R_Ctx(NppiSize{64, 32}, &n, ctx);
    Case c("bufferSize/normRelL2_16u_C1R", "nppiNormRelL2GetBufferHostSize_16u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiNormRelInfGetBufferHostSize_32f_AC4R_Ctx(NppiSize{17, 13}, &n, ctx);
    Case c("bufferSize/normRelInf_32f_AC4R", "nppiNormRelInfGetBufferHostSize_32f_AC4R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiDotProdGetBufferHostSize_32f64f_C3R_Ctx(NppiSize{17, 13}, &n, ctx);
    Case c("bufferSize/dotprod_32f_C3R", "nppiDotProdGetBufferHostSize_32f64f_C3R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiFullNormLevelGetBufferHostSize_32f_AC4R_Ctx(NppiSize{17, 13}, &n, ctx);
    Case c("bufferSize/fullNormLevel_32f_AC4R", "nppiFullNormLevelGetBufferHostSize_32f_AC4R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiSameNormLevelGetBufferHostSize_16u32f_C1R_Ctx(NppiSize{64, 32}, &n, ctx);
    Case c("bufferSize/sameNormLevel_16u32f_C1R", "nppiSameNormLevelGetBufferHostSize_16u32f_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiValidNormLevelGetBufferHostSize_8u_C3RSfs_Ctx(NppiSize{17, 13}, &n, ctx);
    Case c("bufferSize/validNormLevel_8u_C3RSfs", "nppiValidNormLevelGetBufferHostSize_8u_C3RSfs_Ctx");
    c.status(st); c.finish();
  }
}

// ============================ contract =====================================
// Illegal-input probes follow the P3.1b1 convention: only scenarios that do
// not fault the NVIDIA device are recorded (null src, null out, zero / negative
// ROI, short step).  Status is captured via Case so verify compares the golden
// status name returned by the 5090.
TEST_F(NppiNormRelDotProdDiff, Contract_NormRel) {
  DeviceMemory<Npp8u> &s1 = arena<Npp8u>(16);
  DeviceMemory<Npp8u> &s2 = arena<Npp8u>(16);
  DeviceMemory<Npp64f> &o = arena<Npp64f>(1);
  NppStreamContext ctx = current_context();
  {
    NppStatus st = nppiNormRel_L1_8u_C1R_Ctx(s1.get(), 16, nullptr, 16, NppiSize{2, 2}, o.get(), scratch(), ctx);
    Case c("normrel/contract/null_src2", "nppiNormRel_L1_8u_C1R");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiNormRel_L1_8u_C1R_Ctx(nullptr, 16, s2.get(), 16, NppiSize{2, 2}, o.get(), scratch(), ctx);
    Case c("normrel/contract/null_src1", "nppiNormRel_L1_8u_C1R");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiNormRel_L1_8u_C1R_Ctx(s1.get(), 16, s2.get(), 16, NppiSize{2, 2}, nullptr, scratch(), ctx);
    Case c("normrel/contract/null_out", "nppiNormRel_L1_8u_C1R");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiNormRel_L1_8u_C1R_Ctx(s1.get(), 16, s2.get(), 16, NppiSize{0, 2}, o.get(), scratch(), ctx);
    Case c("normrel/contract/zro_roi", "nppiNormRel_L1_8u_C1R");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiNormRel_L1_8u_C1R_Ctx(s1.get(), 16, s2.get(), 16, NppiSize{-1, 2}, o.get(), scratch(), ctx);
    Case c("normrel/contract/neg_size", "nppiNormRel_L1_8u_C1R");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiNormRel_L1_8u_C1R_Ctx(s1.get(), 2, s2.get(), 16, NppiSize{4, 2}, o.get(), scratch(), ctx);
    Case c("normrel/contract/short_step", "nppiNormRel_L1_8u_C1R");
    c.status(st); c.finish();
  }
}

TEST_F(NppiNormRelDotProdDiff, Contract_DotProd) {
  DeviceMemory<Npp8u> &s1 = arena<Npp8u>(16);
  DeviceMemory<Npp8u> &s2 = arena<Npp8u>(16);
  DeviceMemory<Npp64f> &o = arena<Npp64f>(1);
  NppStreamContext ctx = current_context();
  {
    NppStatus st = nppiDotProd_8u64f_C1R_Ctx(nullptr, 16, s2.get(), 16, NppiSize{2, 2}, o.get(), scratch(), ctx);
    Case c("dotprod/contract/null_src1", "nppiDotProd_8u64f_C1R");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiDotProd_8u64f_C1R_Ctx(s1.get(), 16, s2.get(), 16, NppiSize{0, 2}, o.get(), scratch(), ctx);
    Case c("dotprod/contract/zro_roi", "nppiDotProd_8u64f_C1R");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiDotProd_8u64f_C1R_Ctx(s1.get(), 2, s2.get(), 16, NppiSize{4, 2}, o.get(), scratch(), ctx);
    Case c("dotprod/contract/short_step", "nppiDotProd_8u64f_C1R");
    c.status(st); c.finish();
  }
}
