// Differential tests for nppiQualityIndex and its GetBufferHostSize helpers
// (package s2).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   tools/remote5090.sh record 'NppiQualityIndexDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppiQualityIndexDiff.*'
//
// QualityIndex (documented formula):
//   Q = 4*sigma_MN*Mtil*Ntil / ((Mtil^2+Ntil^2)(sigma_M^2+sigma_N^2))
// with the N-1 (W*H-1) denominators for the variance/covariance terms.
// pDst receives one Npp32f per processed channel (AC4R ignores alpha).
// NPP_QUALITY_INDEX_ERROR is returned when either image is constant.
//
// Output arity is pinned with sentinel-prefilled destination buffers: the
// number of floats that change from the sentinel is the number NPP actually
// wrote, so record and verify agree even on the untouched lanes.

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

inline Npp8u *scratch() { return arena<Npp8u>(8 << 20).get(); }

// Sentinel used to prefill output buffers (an ordinary, easily-spotted float).
constexpr float kSentinel = -1.0e30f;

template <typename T> void prefill(DeviceMemory<T> &mem, T value) {
  std::vector<T> host(mem.size(), value);
  mem.copyFromHost(host);
}

// Generic two-source Npp32f metric.  nOut floats are captured; entries NPP
// does not write keep the sentinel value.
template <typename D>
void metric(const char *fname, const std::string &idBase, int width, int height, int pad, int srcCh, uint64_t seed,
            NppStatus (*f)(const D *, int, const D *, int, NppiSize, Npp32f *, Npp8u *, NppStreamContext),
            size_t nOut, double rtol = 1e-5, double atol = 1e-5, Npp8u *scratchOverride = nullptr) {
  const int se = (int)sizeof(D);
  const int step = align_step(width * srcCh * se + pad, se);
  const size_t elems = (size_t)height * (size_t)(step / se);
  std::vector<D> h1 = make_uniform<D>(elems, seed);
  std::vector<D> h2 = make_uniform<D>(elems, seed ^ 0x9E3779B97F4A7C15ULL);
  DeviceMemory<D> &s1 = arena<D>(elems);
  DeviceMemory<D> &s2 = arena<D>(elems);
  DeviceMemory<Npp32f> &o = arena<Npp32f>(nOut);
  s1.copyFromHost(h1);
  s2.copyFromHost(h2);
  prefill(o, kSentinel);
  Case c(idBase + "_" + std::to_string(width) + "x" + std::to_string(height), fname);
  clear_cuda_error();
  NppStatus st = f(s1.get(), step, s2.get(), step, NppiSize{width, height}, o.get(),
                   scratchOverride ? scratchOverride : scratch(), current_context());
  c.input(h1.data(), h1.size() * sizeof(D));
  c.input(h2.data(), h2.size() * sizeof(D));
  c.status(st);
  c.device_output(o.get(), nOut, ValueKind::F32, rtol, atol);
  c.finish();
}

// Caller-crafted two-source variant (constant images, alpha probes).
template <typename D>
void metric_data(const char *fname, const std::string &id, const std::vector<D> &h1, const std::vector<D> &h2,
                 int width, int height, int srcCh,
                 NppStatus (*f)(const D *, int, const D *, int, NppiSize, Npp32f *, Npp8u *, NppStreamContext),
                 size_t nOut, double rtol = 1e-5, double atol = 1e-5) {
  const int se = (int)sizeof(D);
  const int step = align_step(width * srcCh * se + 8, se);
  ASSERT_EQ(h1.size(), (size_t)height * (size_t)(step / se));
  ASSERT_EQ(h2.size(), h1.size());
  DeviceMemory<D> &s1 = arena<D>(h1.size());
  DeviceMemory<D> &s2 = arena<D>(h2.size());
  DeviceMemory<Npp32f> &o = arena<Npp32f>(nOut);
  s1.copyFromHost(h1);
  s2.copyFromHost(h2);
  prefill(o, kSentinel);
  Case c(id, fname);
  clear_cuda_error();
  NppStatus st = f(s1.get(), step, s2.get(), step, NppiSize{width, height}, o.get(), scratch(), current_context());
  c.input(h1.data(), h1.size() * sizeof(D));
  c.input(h2.data(), h2.size() * sizeof(D));
  c.status(st);
  c.device_output(o.get(), nOut, ValueKind::F32, rtol, atol);
  c.finish();
}

// Crafted constant image of a packed layout: every channel of every padded
// row holds `value` (padding bytes are never read).
template <typename D> std::vector<D> constant_image(int width, int height, int srcCh, D value) {
  const int se = (int)sizeof(D);
  const int step = align_step(width * srcCh * se + 8, se);
  std::vector<D> v((size_t)height * (size_t)(step / se), D{});
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      for (int c = 0; c < srcCh; ++c) v[(size_t)y * (step / se) + (size_t)x * srcCh + c] = value;
  return v;
}

// Element count matching constant_image()/metric_data() row padding.
inline size_t padded_elems(int width, int height, int srcCh, int esize, int pad = 8) {
  const int step = align_step(width * srcCh * esize + pad, esize);
  return (size_t)height * (size_t)(step / esize);
}

}  // namespace

class NppiQualityIndexDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

// ============================== nppiQualityIndex ============================
// One RNG case per dtype x layout, plus 1x1 and 64x32 padded ROIs.  The
// multi-channel cases capture 4 floats (sentinel-prefilled) so the per-channel
// output arity is pinned by the golden.
TEST_F(NppiQualityIndexDiff, QualityIndex_Plain) {
  metric<Npp8u>("nppiQualityIndex_8u32f_C1R_Ctx", "qIdx/8u32f/C1R", 17, 13, 16, 1, 0x501,
                &nppiQualityIndex_8u32f_C1R_Ctx, 1);
  metric<Npp8u>("nppiQualityIndex_8u32f_C1R_Ctx", "qIdx/8u32f/C1R", 1, 1, 8, 1, 0x502,
                &nppiQualityIndex_8u32f_C1R_Ctx, 1);
  metric<Npp8u>("nppiQualityIndex_8u32f_C3R_Ctx", "qIdx/8u32f/C3R", 17, 13, 16, 3, 0x503,
                &nppiQualityIndex_8u32f_C3R_Ctx, 4);
  metric<Npp8u>("nppiQualityIndex_8u32f_AC4R_Ctx", "qIdx/8u32f/AC4R", 17, 13, 16, 4, 0x504,
                &nppiQualityIndex_8u32f_AC4R_Ctx, 4);
  metric<Npp8u>("nppiQualityIndex_8u32f_AC4R_Ctx", "qIdx/8u32f/AC4R", 64, 32, 32, 4, 0x505,
                &nppiQualityIndex_8u32f_AC4R_Ctx, 4);
  metric<Npp16u>("nppiQualityIndex_16u32f_C1R_Ctx", "qIdx/16u32f/C1R", 17, 13, 16, 1, 0x511,
                 &nppiQualityIndex_16u32f_C1R_Ctx, 1);
  metric<Npp16u>("nppiQualityIndex_16u32f_C3R_Ctx", "qIdx/16u32f/C3R", 17, 13, 16, 3, 0x512,
                 &nppiQualityIndex_16u32f_C3R_Ctx, 4);
  metric<Npp16u>("nppiQualityIndex_16u32f_AC4R_Ctx", "qIdx/16u32f/AC4R", 64, 32, 32, 4, 0x513,
                 &nppiQualityIndex_16u32f_AC4R_Ctx, 4);
  metric<Npp32f>("nppiQualityIndex_32f_C1R_Ctx", "qIdx/32f/C1R", 17, 13, 16, 1, 0x521,
                 &nppiQualityIndex_32f_C1R_Ctx, 1);
  metric<Npp32f>("nppiQualityIndex_32f_C1R_Ctx", "qIdx/32f/C1R", 1, 1, 8, 1, 0x522,
                 &nppiQualityIndex_32f_C1R_Ctx, 1);
  metric<Npp32f>("nppiQualityIndex_32f_C3R_Ctx", "qIdx/32f/C3R", 17, 13, 16, 3, 0x523,
                 &nppiQualityIndex_32f_C3R_Ctx, 4);
  metric<Npp32f>("nppiQualityIndex_32f_AC4R_Ctx", "qIdx/32f/AC4R", 64, 32, 32, 4, 0x524,
                 &nppiQualityIndex_32f_AC4R_Ctx, 4);
}

// ========================= crafted formula / error ==========================
// Identical images give Q = 1 exactly.  A hand-computed 4x1 case pins the
// N-1 denominators (population denominators would give a different value).
// Constant images must return NPP_QUALITY_INDEX_ERROR.
TEST_F(NppiQualityIndexDiff, QualityIndex_Crafted) {
  {  // identical 8u images -> Q = 1.0
    const size_t n = padded_elems(17, 13, 1, 1);
    metric_data<Npp8u>("nppiQualityIndex_8u32f_C1R_Ctx", "qIdx/8u32f/C1R_identical_17x13", make_uniform<Npp8u>(n, 0x601),
                       make_uniform<Npp8u>(n, 0x601), 17, 13, 1, &nppiQualityIndex_8u32f_C1R_Ctx, 1);
  }
  {  // hand-computed 8u C1 4x1: M=(1,2,3,4) N=(1,2,3,5)
     // Mtil=2.5 Ntil=2.75 sM2=5/3 sN2=8.75/3 sMN=7.625/3
     // Q = 4*(7.625/3)*2.5*2.75/((2.5^2+2.75^2)*((5+8.75)/3)) ~ 1.1040
    std::vector<Npp8u> h1 = constant_image<Npp8u>(4, 1, 1, Npp8u{0});
    std::vector<Npp8u> h2 = constant_image<Npp8u>(4, 1, 1, Npp8u{0});
    const Npp8u a[4] = {1, 2, 3, 4}, b[4] = {1, 2, 3, 5};
    for (int x = 0; x < 4; ++x) {
      h1[x] = a[x];
      h2[x] = b[x];
    }
    metric_data<Npp8u>("nppiQualityIndex_8u32f_C1R_Ctx", "qIdx/8u32f/C1R_formula_4x1", h1, h2, 4, 1, 1,
                       &nppiQualityIndex_8u32f_C1R_Ctx, 1);
  }
  {  // constant 8u image -> NPP_QUALITY_INDEX_ERROR (src1 constant)
    metric_data<Npp8u>("nppiQualityIndex_8u32f_C1R_Ctx", "qIdx/8u32f/C1R_const_src1", constant_image<Npp8u>(17, 13, 1, Npp8u{7}),
                       make_uniform<Npp8u>(padded_elems(17, 13, 1, 1), 0x602), 17, 13, 1,
                       &nppiQualityIndex_8u32f_C1R_Ctx, 1);
  }
  {  // constant 32f image -> NPP_QUALITY_INDEX_ERROR (src2 constant)
    metric_data<Npp32f>("nppiQualityIndex_32f_C1R_Ctx", "qIdx/32f/C1R_const_src2",
                        make_uniform<Npp32f>(padded_elems(17, 13, 1, 4), 0x603),
                        constant_image<Npp32f>(17, 13, 1, Npp32f{-1.5f}), 17, 13, 1,
                        &nppiQualityIndex_32f_C1R_Ctx, 1);
  }
  {  // 1x1 ROI: both images trivially constant -> NPP_QUALITY_INDEX_ERROR
    metric_data<Npp8u>("nppiQualityIndex_8u32f_C1R_Ctx", "qIdx/8u32f/C1R_roi_1x1", constant_image<Npp8u>(1, 1, 1, Npp8u{3}),
                       constant_image<Npp8u>(1, 1, 1, Npp8u{9}), 1, 1, 1, &nppiQualityIndex_8u32f_C1R_Ctx, 1);
  }
  {  // AC4R: identical RGB, wildly different alpha -> alpha must be ignored
    std::vector<Npp8u> h1 = constant_image<Npp8u>(17, 13, 4, Npp8u{0});
    std::vector<Npp8u> h2 = constant_image<Npp8u>(17, 13, 4, Npp8u{0});
    const int se = 1, srcCh = 4, width = 17, height = 13;
    const int step = align_step(width * srcCh * se + 8, se);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < 3; ++c) {
          h1[(size_t)y * (step / se) + (size_t)x * srcCh + c] = (Npp8u)(x * 7 + y * 11 + c);
          h2[(size_t)y * (step / se) + (size_t)x * srcCh + c] = (Npp8u)(x * 7 + y * 11 + c);
        }
        h1[(size_t)y * (step / se) + (size_t)x * srcCh + 3] = 0;
        h2[(size_t)y * (step / se) + (size_t)x * srcCh + 3] = 255;
      }
    }
    metric_data<Npp8u>("nppiQualityIndex_8u32f_AC4R_Ctx", "qIdx/8u32f/AC4R_alpha_ignored", h1, h2, 17, 13, 4,
                       &nppiQualityIndex_8u32f_AC4R_Ctx, 4);
  }
  {  // C3R per-channel identity: only the red channel differs -> Q[0]<1, Q[1]=Q[2]=1
    std::vector<Npp8u> h1 = constant_image<Npp8u>(8, 2, 3, Npp8u{0});
    std::vector<Npp8u> h2 = constant_image<Npp8u>(8, 2, 3, Npp8u{0});
    const int se = 1, srcCh = 3, width = 8, height = 2;
    const int step = align_step(width * srcCh * se + 8, se);
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x) {
        const Npp8u r = (Npp8u)(x * 13 + y * 5);
        for (int c = 0; c < 3; ++c) {
          h1[(size_t)y * (step / se) + (size_t)x * srcCh + c] = r;
          h2[(size_t)y * (step / se) + (size_t)x * srcCh + c] = (c == 0) ? (Npp8u)(r + 1) : r;
        }
      }
    metric_data<Npp8u>("nppiQualityIndex_8u32f_C3R_Ctx", "qIdx/8u32f/C3R_per_channel", h1, h2, 8, 2, 3,
                       &nppiQualityIndex_8u32f_C3R_Ctx, 4);
  }
}

// ==================== buffer-size helpers (status only) =====================
TEST_F(NppiQualityIndexDiff, BufferSize_Helpers) {
  NppStreamContext ctx = current_context();
  struct Probe {
    const char *id;
    const char *fn;
    NppStatus (*f)(NppiSize, SIZE_TYPE *, NppStreamContext);
    NppiSize roi;
  };
  const Probe probes[] = {
      {"qIdx/buf/8u32f_C1R", "nppiQualityIndexGetBufferHostSize_8u32f_C1R_Ctx",
       &nppiQualityIndexGetBufferHostSize_8u32f_C1R_Ctx, NppiSize{17, 13}},
      {"qIdx/buf/8u32f_C3R", "nppiQualityIndexGetBufferHostSize_8u32f_C3R_Ctx",
       &nppiQualityIndexGetBufferHostSize_8u32f_C3R_Ctx, NppiSize{17, 13}},
      {"qIdx/buf/8u32f_AC4R", "nppiQualityIndexGetBufferHostSize_8u32f_AC4R_Ctx",
       &nppiQualityIndexGetBufferHostSize_8u32f_AC4R_Ctx, NppiSize{64, 32}},
      {"qIdx/buf/16u32f_C1R", "nppiQualityIndexGetBufferHostSize_16u32f_C1R_Ctx",
       &nppiQualityIndexGetBufferHostSize_16u32f_C1R_Ctx, NppiSize{17, 13}},
      {"qIdx/buf/16u32f_C3R", "nppiQualityIndexGetBufferHostSize_16u32f_C3R_Ctx",
       &nppiQualityIndexGetBufferHostSize_16u32f_C3R_Ctx, NppiSize{17, 13}},
      {"qIdx/buf/16u32f_AC4R", "nppiQualityIndexGetBufferHostSize_16u32f_AC4R_Ctx",
       &nppiQualityIndexGetBufferHostSize_16u32f_AC4R_Ctx, NppiSize{1, 1}},
      {"qIdx/buf/32f_C1R", "nppiQualityIndexGetBufferHostSize_32f_C1R_Ctx",
       &nppiQualityIndexGetBufferHostSize_32f_C1R_Ctx, NppiSize{17, 13}},
      {"qIdx/buf/32f_C3R", "nppiQualityIndexGetBufferHostSize_32f_C3R_Ctx",
       &nppiQualityIndexGetBufferHostSize_32f_C3R_Ctx, NppiSize{17, 13}},
      {"qIdx/buf/32f_AC4R", "nppiQualityIndexGetBufferHostSize_32f_AC4R_Ctx",
       &nppiQualityIndexGetBufferHostSize_32f_AC4R_Ctx, NppiSize{64, 32}},
  };
  for (const Probe &p : probes) {
    SIZE_TYPE n = 0;
    NppStatus st = p.f(p.roi, &n, ctx);
    Case c(p.id, p.fn);
    c.status(st);
    c.host_output(&n, 1, ValueKind::Bytes);
    c.finish();
  }
}

// Buffer-size helper contract: null out pointer, zero ROI, negative ROI.
TEST_F(NppiQualityIndexDiff, BufferSize_Contract) {
  NppStreamContext ctx = current_context();
  {
    SIZE_TYPE n = 0;
    NppStatus st = nppiQualityIndexGetBufferHostSize_8u32f_C1R_Ctx(NppiSize{17, 13}, static_cast<SIZE_TYPE *>(nullptr), ctx);
    Case c("qIdx/buf_contract/null_ptr", "nppiQualityIndexGetBufferHostSize_8u32f_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    SIZE_TYPE n = 0;
    NppStatus st = nppiQualityIndexGetBufferHostSize_8u32f_C1R_Ctx(NppiSize{0, 2}, &n, ctx);
    Case c("qIdx/buf_contract/zro_roi", "nppiQualityIndexGetBufferHostSize_8u32f_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    SIZE_TYPE n = 0;
    NppStatus st = nppiQualityIndexGetBufferHostSize_8u32f_C1R_Ctx(NppiSize{-1, 2}, &n, ctx);
    Case c("qIdx/buf_contract/neg_size", "nppiQualityIndexGetBufferHostSize_8u32f_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    SIZE_TYPE n = 0;
    NppStatus st = nppiQualityIndexGetBufferHostSize_32f_C3R_Ctx(NppiSize{17, 13}, static_cast<SIZE_TYPE *>(nullptr), ctx);
    Case c("qIdx/buf_contract/null_ptr_c3", "nppiQualityIndexGetBufferHostSize_32f_C3R_Ctx");
    c.status(st); c.finish();
  }
}

// ==================== buffer-size sufficiency ===============================
TEST_F(NppiQualityIndexDiff, BufferSize_Sufficiency) {
  NppStreamContext ctx = current_context();
  {
    SIZE_TYPE sz = 0;
    ASSERT_EQ(nppiQualityIndexGetBufferHostSize_8u32f_C1R_Ctx(NppiSize{64, 32}, &sz, ctx), NPP_NO_ERROR);
    ASSERT_GT(sz, (SIZE_TYPE)0);
    metric<Npp8u>("nppiQualityIndex_8u32f_C1R_Ctx", "qIdx/8u32f/C1R_exactbuf", 64, 32, 32, 1, 0x701,
                  &nppiQualityIndex_8u32f_C1R_Ctx, 1, 1e-5, 1e-5, arena<Npp8u>(sz).get());
  }
  {
    SIZE_TYPE sz = 0;
    ASSERT_EQ(nppiQualityIndexGetBufferHostSize_16u32f_AC4R_Ctx(NppiSize{64, 32}, &sz, ctx), NPP_NO_ERROR);
    ASSERT_GT(sz, (SIZE_TYPE)0);
    metric<Npp16u>("nppiQualityIndex_16u32f_AC4R_Ctx", "qIdx/16u32f/AC4R_exactbuf", 64, 32, 32, 4, 0x702,
                   &nppiQualityIndex_16u32f_AC4R_Ctx, 4, 1e-5, 1e-5, arena<Npp8u>(sz).get());
  }
  {
    SIZE_TYPE sz = 0;
    ASSERT_EQ(nppiQualityIndexGetBufferHostSize_32f_C3R_Ctx(NppiSize{17, 13}, &sz, ctx), NPP_NO_ERROR);
    ASSERT_GT(sz, (SIZE_TYPE)0);
    metric<Npp32f>("nppiQualityIndex_32f_C3R_Ctx", "qIdx/32f/C3R_exactbuf", 17, 13, 16, 3, 0x703,
                   &nppiQualityIndex_32f_C3R_Ctx, 4, 1e-5, 1e-5, arena<Npp8u>(sz).get());
  }
}

// ============================ non-Ctx adapters ==============================
#if NPP_DIFF_HAS_NONCTX
TEST_F(NppiQualityIndexDiff, NonCtx_QualityIndex) {
  // Reproduce the qIdx/8u32f/C1R_17x13 golden so both entry points agree.
  const int se = (int)sizeof(Npp8u);
  const int step = align_step(17 * se + 16, se);
  const size_t elems = (size_t)13 * (step / se);
  std::vector<Npp8u> h1 = make_uniform<Npp8u>(elems, 0x501);
  std::vector<Npp8u> h2 = make_uniform<Npp8u>(elems, 0x501 ^ 0x9E3779B97F4A7C15ULL);
  DeviceMemory<Npp8u> &s1 = arena<Npp8u>(elems);
  DeviceMemory<Npp8u> &s2 = arena<Npp8u>(elems);
  DeviceMemory<Npp32f> &o = arena<Npp32f>(1);
  s1.copyFromHost(h1);
  s2.copyFromHost(h2);
  prefill(o, kSentinel);
  Case c("qIdx/8u32f/C1R_17x13", "nppiQualityIndex_8u32f_C1R");  // reuses the golden id above
  clear_cuda_error();
  NppStatus st = nppiQualityIndex_8u32f_C1R(s1.get(), step, s2.get(), step, NppiSize{17, 13}, o.get(), scratch());
  c.input(h1.data(), h1.size() * sizeof(Npp8u));
  c.input(h2.data(), h2.size() * sizeof(Npp8u));
  c.status(st);
  c.device_output(o.get(), 1, ValueKind::F32);
  c.finish();
}

TEST_F(NppiQualityIndexDiff, NonCtx_QualityIndexC3) {
  // Reproduce the qIdx/8u32f/C3R_17x13 golden via the non-Ctx entry point.
  const int se = (int)sizeof(Npp8u);
  const int step = align_step(17 * 3 * se + 16, se);
  const size_t elems = (size_t)13 * (step / se);
  std::vector<Npp8u> h1 = make_uniform<Npp8u>(elems, 0x503);
  std::vector<Npp8u> h2 = make_uniform<Npp8u>(elems, 0x503 ^ 0x9E3779B97F4A7C15ULL);
  DeviceMemory<Npp8u> &s1 = arena<Npp8u>(elems);
  DeviceMemory<Npp8u> &s2 = arena<Npp8u>(elems);
  DeviceMemory<Npp32f> &o = arena<Npp32f>(4);
  s1.copyFromHost(h1);
  s2.copyFromHost(h2);
  prefill(o, kSentinel);
  Case c("qIdx/8u32f/C3R_17x13", "nppiQualityIndex_8u32f_C3R");
  clear_cuda_error();
  NppStatus st = nppiQualityIndex_8u32f_C3R(s1.get(), step, s2.get(), step, NppiSize{17, 13}, o.get(), scratch());
  c.input(h1.data(), h1.size() * sizeof(Npp8u));
  c.input(h2.data(), h2.size() * sizeof(Npp8u));
  c.status(st);
  c.device_output(o.get(), 4, ValueKind::F32);
  c.finish();
}
#endif

// ============================ contract ======================================
TEST_F(NppiQualityIndexDiff, Contract_QualityIndex) {
  DeviceMemory<Npp8u> &s1 = arena<Npp8u>(64);
  DeviceMemory<Npp8u> &s2 = arena<Npp8u>(64);
  DeviceMemory<Npp32f> &o = arena<Npp32f>(4);
  prefill(s1, Npp8u{0});
  prefill(s2, Npp8u{0});
  NppStreamContext ctx = current_context();
  {
    NppStatus st = nppiQualityIndex_8u32f_C1R_Ctx(nullptr, 16, s2.get(), 16, NppiSize{2, 2}, o.get(), scratch(), ctx);
    Case c("qIdx/contract/null_src1", "nppiQualityIndex_8u32f_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiQualityIndex_8u32f_C1R_Ctx(s1.get(), 16, nullptr, 16, NppiSize{2, 2}, o.get(), scratch(), ctx);
    Case c("qIdx/contract/null_src2", "nppiQualityIndex_8u32f_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiQualityIndex_8u32f_C1R_Ctx(s1.get(), 16, s2.get(), 16, NppiSize{2, 2}, nullptr, scratch(), ctx);
    Case c("qIdx/contract/null_out", "nppiQualityIndex_8u32f_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiQualityIndex_8u32f_C1R_Ctx(s1.get(), 16, s2.get(), 16, NppiSize{2, 2}, o.get(), nullptr, ctx);
    Case c("qIdx/contract/null_buf", "nppiQualityIndex_8u32f_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiQualityIndex_8u32f_C1R_Ctx(s1.get(), 16, s2.get(), 16, NppiSize{0, 2}, o.get(), scratch(), ctx);
    Case c("qIdx/contract/zro_roi", "nppiQualityIndex_8u32f_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiQualityIndex_8u32f_C1R_Ctx(s1.get(), 16, s2.get(), 16, NppiSize{-1, 2}, o.get(), scratch(), ctx);
    Case c("qIdx/contract/neg_size", "nppiQualityIndex_8u32f_C1R_Ctx");
    c.status(st); c.finish();
  }
  {  // step < width*channels -> step error
    NppStatus st = nppiQualityIndex_8u32f_C1R_Ctx(s1.get(), 2, s2.get(), 16, NppiSize{4, 2}, o.get(), scratch(), ctx);
    Case c("qIdx/contract/short_step", "nppiQualityIndex_8u32f_C1R_Ctx");
    c.status(st); c.finish();
  }
  {  // src2 step too short
    NppStatus st = nppiQualityIndex_8u32f_C1R_Ctx(s1.get(), 16, s2.get(), 2, NppiSize{4, 2}, o.get(), scratch(), ctx);
    Case c("qIdx/contract/short_step2", "nppiQualityIndex_8u32f_C1R_Ctx");
    c.status(st); c.finish();
  }
  {  // 32f AC4 step not a multiple of 16 bytes
    DeviceMemory<Npp32f> &t1 = arena<Npp32f>(64);
    DeviceMemory<Npp32f> &t2 = arena<Npp32f>(64);
    prefill(t1, 0.5f);
    prefill(t2, 0.5f);
    NppStatus st =
        nppiQualityIndex_32f_AC4R_Ctx(t1.get(), 36, t2.get(), 36, NppiSize{2, 2}, o.get(), scratch(), ctx);
    Case c("qIdx/contract/noteven_step_c4", "nppiQualityIndex_32f_AC4R_Ctx");
    c.status(st); c.finish();
  }
  {  // 16u C1 step not a multiple of 2
    DeviceMemory<Npp16u> &t1 = arena<Npp16u>(64);
    DeviceMemory<Npp16u> &t2 = arena<Npp16u>(64);
    prefill(t1, Npp16u{5});
    prefill(t2, Npp16u{5});
    NppStatus st =
        nppiQualityIndex_16u32f_C1R_Ctx(t1.get(), 3, t2.get(), 3, NppiSize{2, 2}, o.get(), scratch(), ctx);
    Case c("qIdx/contract/noteven_step_c1_16u", "nppiQualityIndex_16u32f_C1R_Ctx");
    c.status(st); c.finish();
  }
}
