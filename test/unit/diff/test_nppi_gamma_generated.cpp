// Differential tests for the nppiGammaFwd / nppiGammaInv planar variants
// (package c4).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiGammaDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records ./diff_tests \
//     --gtest_filter='NppiGammaDiff.*'
//
// Semantics pinned here (NVIDIA NPP 13.4, measured on the 5090):
//   * Each ROI byte of each of the three planes is mapped through the family
//     LUT; the ramp case below pins all 256 entries of both tables.  Row
//     padding past the ROI is never written (dst sentinel bytes).
//   * 8u planar steps have no alignment requirement (a step of width+1 is
//     accepted).
//   * Host validation: null plane-0 pointer -> NPP_NULL_POINTER_ERROR,
//     negative ROI -> NPP_SIZE_ERROR, empty ROI -> NPP_NO_ERROR even with a
//     bad step, step == 0 or step < width -> NPP_STEP_ERROR (checked for the
//     src and dst steps independently).
// NVIDIA only null-checks plane 0 (a null plane 1/2 pointer returns
// NPP_NO_ERROR and then faults the context) and launches with negative steps
// walking out-of-bounds rows; mcMPP validates all planes and does not launch
// on negative steps.  Neither case is covered by goldens.

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace npp_diff;
using namespace npp_functional_test;

namespace {

// Keep every device buffer alive for the process lifetime: the maca allocator
// can hand out just-freed pointers and then reject them in a later kernel.
template <typename T> DeviceMemory<T> &arena(std::size_t n) {
  static std::vector<std::unique_ptr<DeviceMemory<T>>> v;
  v.push_back(std::make_unique<DeviceMemory<T>>(n));
  return *v.back();
}

NppStreamContext current_context() {
  NppStreamContext ctx;
  EXPECT_EQ(nppGetStreamContext(&ctx), NPP_NO_ERROR) << "nppGetStreamContext failed";
  return ctx;
}

ValueKind u8_kind() { return ValueKind::U8; }

const Npp8u kSentinel = 0xEE;

struct Planes {
  DeviceMemory<Npp8u> *p[3];
};

Planes make_planes(std::size_t bytes, Npp8u fill) {
  Planes pl;
  for (int i = 0; i < 3; ++i) {
    pl.p[i] = &arena<Npp8u>(bytes);
    EXPECT_EQ(cudaMemset(pl.p[i]->get(), fill, bytes * sizeof(Npp8u)), cudaSuccess);
  }
  return pl;
}

// Out-of-place planar case: three uniform-random source planes, sentinel dst.
void p3_case(const std::string &id, const char *fname, int width, int height, int pad, uint64_t seed, bool inverse,
             bool nonctx = false) {
  const int step = width + pad;
  const size_t elems = (size_t)height * step;
  std::vector<Npp8u> hs[3];
  Planes src = make_planes(elems, 0);
  Planes dst = make_planes(elems, kSentinel);
  const Npp8u *sp[3];
  Npp8u *dp[3];
  for (int i = 0; i < 3; ++i) {
    hs[i] = make_uniform<Npp8u>(elems, seed + 100 * i);
    src.p[i]->copyFromHost(hs[i]);
    sp[i] = src.p[i]->get();
    dp[i] = dst.p[i]->get();
  }
  NppiSize o{width, height};
  clear_cuda_error();
  NppStatus st;
#if NPP_DIFF_HAS_NONCTX
  if (nonctx) {
    st = inverse ? nppiGammaInv_8u_P3R(sp, step, dp, step, o) : nppiGammaFwd_8u_P3R(sp, step, dp, step, o);
  } else
#endif
  {
    st = inverse ? nppiGammaInv_8u_P3R_Ctx(sp, step, dp, step, o, current_context())
                 : nppiGammaFwd_8u_P3R_Ctx(sp, step, dp, step, o, current_context());
  }
  Case c("nppiGamma/" + id, fname);
  for (int i = 0; i < 3; ++i)
    c.input(hs[i].data(), hs[i].size());
  c.status(st);
  for (int i = 0; i < 3; ++i)
    c.device_output<Npp8u>(dp[i], elems, u8_kind());
  c.finish();
}

// In-place planar case: sentinel padding proves only the ROI is rewritten.
void ip3_case(const std::string &id, const char *fname, int width, int height, int pad, uint64_t seed, bool inverse,
              bool nonctx = false) {
  const int step = width + pad;
  const size_t elems = (size_t)height * step;
  std::vector<Npp8u> hs[3];
  Planes buf = make_planes(elems, kSentinel);
  Npp8u *bp[3];
  for (int i = 0; i < 3; ++i) {
    hs[i] = make_uniform<Npp8u>(elems, seed + 100 * i);
    for (int y = 0; y < height; ++y)
      for (int x = width; x < step; ++x)
        hs[i][(size_t)y * step + x] = kSentinel;
    buf.p[i]->copyFromHost(hs[i]);
    bp[i] = buf.p[i]->get();
  }
  NppiSize o{width, height};
  clear_cuda_error();
  NppStatus st;
#if NPP_DIFF_HAS_NONCTX
  if (nonctx) {
    st = inverse ? nppiGammaInv_8u_IP3R(bp, step, o) : nppiGammaFwd_8u_IP3R(bp, step, o);
  } else
#endif
  {
    st = inverse ? nppiGammaInv_8u_IP3R_Ctx(bp, step, o, current_context())
                 : nppiGammaFwd_8u_IP3R_Ctx(bp, step, o, current_context());
  }
  Case c("nppiGamma/" + id, fname);
  for (int i = 0; i < 3; ++i)
    c.input(hs[i].data(), hs[i].size());
  c.status(st);
  for (int i = 0; i < 3; ++i)
    c.device_output<Npp8u>(bp[i], elems, u8_kind());
  c.finish();
}

// Error-path case for the out-of-place entry points.  which: 0 = null src
// plane 0, 1 = null dst plane 0.
void p3_error(const std::string &id, const char *fname, Npp8u *which, NppiSize roi, int srcStep, int dstStep,
              bool inverse) {
  // Allocate for the largest step so padded/negative-adjacent rows stay in
  // bounds for the reference implementation.
  int span = 5;
  if (srcStep > span)
    span = srcStep;
  if (dstStep > span)
    span = dstStep;
  const size_t elems = (size_t)3 * span;
  std::vector<Npp8u> hs = make_uniform<Npp8u>(elems, 4242);
  Planes src = make_planes(elems, 0);
  Planes dst = make_planes(elems, kSentinel);
  const Npp8u *sp[3];
  Npp8u *dp[3];
  for (int i = 0; i < 3; ++i) {
    src.p[i]->copyFromHost(hs);
    sp[i] = src.p[i]->get();
    dp[i] = dst.p[i]->get();
  }
  if (which == (Npp8u *)1)
    sp[0] = nullptr;
  if (which == (Npp8u *)3)
    dp[0] = nullptr;
  NppiSize o = roi;
  clear_cuda_error();
  NppStatus st = inverse ? nppiGammaInv_8u_P3R_Ctx(sp, srcStep, dp, dstStep, o, current_context())
                         : nppiGammaFwd_8u_P3R_Ctx(sp, srcStep, dp, dstStep, o, current_context());
  Case c("nppiGamma/" + id, fname);
  c.status(st);
  for (int i = 0; i < 3; ++i)
    c.device_output<Npp8u>(dp[i], elems, u8_kind());
  c.finish();
}

// Error-path case for the in-place entry point.  which: 0 = null plane 0.
void ip3_error(const std::string &id, const char *fname, Npp8u *which, NppiSize roi, int step, bool inverse) {
  // Allocate for the largest step so padded rows stay in bounds for the
  // reference implementation.
  const int span = step > 5 ? step : 5;
  const size_t elems = (size_t)3 * span;
  std::vector<Npp8u> hs = make_uniform<Npp8u>(elems, 1717);
  Planes buf = make_planes(elems, kSentinel);
  Npp8u *bp[3];
  for (int i = 0; i < 3; ++i) {
    buf.p[i]->copyFromHost(hs);
    bp[i] = buf.p[i]->get();
  }
  if (which == (Npp8u *)1)
    bp[0] = nullptr;
  NppiSize o = roi;
  clear_cuda_error();
  NppStatus st = inverse ? nppiGammaInv_8u_IP3R_Ctx(bp, step, o, current_context())
                         : nppiGammaFwd_8u_IP3R_Ctx(bp, step, o, current_context());
  Case c("nppiGamma/" + id, fname);
  c.status(st);
  for (int i = 0; i < 3; ++i)
    c.device_output<Npp8u>(bp[i], elems, u8_kind());
  c.finish();
}

}  // namespace

class NppiGammaDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

TEST_F(NppiGammaDiff, Fwd_P3R) {
  // Ramp 0..255 in every plane pins all 256 forward LUT entries.
  const int step = 256;
  const size_t elems = step;
  Planes src = make_planes(elems, 0);
  Planes dst = make_planes(elems, kSentinel);
  const Npp8u *sp[3];
  Npp8u *dp[3];
  std::vector<Npp8u> ramp(elems);
  for (size_t k = 0; k < elems; ++k)
    ramp[k] = (Npp8u)k;
  for (int i = 0; i < 3; ++i) {
    src.p[i]->copyFromHost(ramp);
    sp[i] = src.p[i]->get();
    dp[i] = dst.p[i]->get();
  }
  NppiSize o{256, 1};
  clear_cuda_error();
  NppStatus st = nppiGammaFwd_8u_P3R_Ctx(sp, step, dp, step, o, current_context());
  Case c("nppiGamma/fwd_ramp256", "nppiGammaFwd_8u_P3R");
  for (int i = 0; i < 3; ++i)
    c.input(ramp.data(), ramp.size());
  c.status(st);
  for (int i = 0; i < 3; ++i)
    c.device_output<Npp8u>(dp[i], elems, u8_kind());
  c.finish();

  p3_case("fwd_plain17x13", "nppiGammaFwd_8u_P3R", 17, 13, 7, 5101, false);
  p3_case("fwd_tight1x1", "nppiGammaFwd_8u_P3R", 1, 1, 0, 5102, false);
}

TEST_F(NppiGammaDiff, Fwd_IP3R) {
  ip3_case("fwd_ip_plain17x13", "nppiGammaFwd_8u_IP3R", 17, 13, 7, 5201, false);
  ip3_case("fwd_ip_tight33x9", "nppiGammaFwd_8u_IP3R", 33, 9, 0, 5202, false);
}

TEST_F(NppiGammaDiff, Inv_P3R) {
  // Ramp 0..255 in every plane pins all 256 inverse LUT entries.
  const int step = 256;
  const size_t elems = step;
  Planes src = make_planes(elems, 0);
  Planes dst = make_planes(elems, kSentinel);
  const Npp8u *sp[3];
  Npp8u *dp[3];
  std::vector<Npp8u> ramp(elems);
  for (size_t k = 0; k < elems; ++k)
    ramp[k] = (Npp8u)k;
  for (int i = 0; i < 3; ++i) {
    src.p[i]->copyFromHost(ramp);
    sp[i] = src.p[i]->get();
    dp[i] = dst.p[i]->get();
  }
  NppiSize o{256, 1};
  clear_cuda_error();
  NppStatus st = nppiGammaInv_8u_P3R_Ctx(sp, step, dp, step, o, current_context());
  Case c("nppiGamma/inv_ramp256", "nppiGammaInv_8u_P3R");
  for (int i = 0; i < 3; ++i)
    c.input(ramp.data(), ramp.size());
  c.status(st);
  for (int i = 0; i < 3; ++i)
    c.device_output<Npp8u>(dp[i], elems, u8_kind());
  c.finish();

  p3_case("inv_plain17x13", "nppiGammaInv_8u_P3R", 17, 13, 7, 5301, true);
  p3_case("inv_tight1x1", "nppiGammaInv_8u_P3R", 1, 1, 0, 5302, true);
}

TEST_F(NppiGammaDiff, Inv_IP3R) {
  ip3_case("inv_ip_plain17x13", "nppiGammaInv_8u_IP3R", 17, 13, 7, 5401, true);
  ip3_case("inv_ip_tight33x9", "nppiGammaInv_8u_IP3R", 33, 9, 0, 5402, true);
}

TEST_F(NppiGammaDiff, Errors_P3R) {
  p3_error("err/null_src0", "nppiGammaFwd_8u_P3R", (Npp8u *)1, {5, 3}, 5, 5, false);
  p3_error("err/null_dst0", "nppiGammaFwd_8u_P3R", (Npp8u *)3, {5, 3}, 5, 5, false);
  p3_error("err/neg_w", "nppiGammaFwd_8u_P3R", nullptr, {-1, 3}, 5, 5, false);
  p3_error("err/neg_h", "nppiGammaFwd_8u_P3R", nullptr, {5, -2}, 5, 5, false);
  p3_error("err/zero_w", "nppiGammaFwd_8u_P3R", nullptr, {0, 3}, 5, 5, false);
  p3_error("err/zero_h", "nppiGammaFwd_8u_P3R", nullptr, {5, 0}, 5, 5, false);
  // Empty ROI short-circuits the step checks (a bad step still returns 0).
  p3_error("err/zero_roi_bad_step", "nppiGammaFwd_8u_P3R", nullptr, {0, 0}, 3, 3, false);
  p3_error("err/zero_step_src", "nppiGammaFwd_8u_P3R", nullptr, {5, 3}, 0, 5, false);
  p3_error("err/zero_step_dst", "nppiGammaFwd_8u_P3R", nullptr, {5, 3}, 5, 0, false);
  p3_error("err/small_step_src", "nppiGammaFwd_8u_P3R", nullptr, {5, 3}, 4, 5, false);
  p3_error("err/small_step_dst", "nppiGammaFwd_8u_P3R", nullptr, {5, 3}, 5, 4, false);
  // 8u planar has no step alignment requirement; width+1 is a valid padding.
  p3_error("err/step6", "nppiGammaFwd_8u_P3R", nullptr, {5, 3}, 6, 6, false);
  p3_error("err/big_step", "nppiGammaFwd_8u_P3R", nullptr, {5, 3}, 512, 512, false);
  // Same validation for the inverse direction.
  p3_error("err/inv_zero_step", "nppiGammaInv_8u_P3R", nullptr, {5, 3}, 0, 5, true);
  p3_error("err/inv_neg_w", "nppiGammaInv_8u_P3R", nullptr, {-5, 3}, 5, 5, true);
}

TEST_F(NppiGammaDiff, Errors_IP3R) {
  ip3_error("err/ip_null", "nppiGammaFwd_8u_IP3R", (Npp8u *)1, {5, 3}, 5, false);
  ip3_error("err/ip_neg_w", "nppiGammaFwd_8u_IP3R", nullptr, {-1, 3}, 5, false);
  ip3_error("err/ip_zero_roi_bad_step", "nppiGammaFwd_8u_IP3R", nullptr, {0, 0}, 3, false);
  ip3_error("err/ip_zero_step", "nppiGammaFwd_8u_IP3R", nullptr, {5, 3}, 0, false);
  ip3_error("err/ip_small_step", "nppiGammaFwd_8u_IP3R", nullptr, {5, 3}, 4, false);
  ip3_error("err/ip_step6", "nppiGammaFwd_8u_IP3R", nullptr, {5, 3}, 6, false);
  ip3_error("err/ip_inv_zero_step", "nppiGammaInv_8u_IP3R", nullptr, {5, 3}, 0, true);
}

#if NPP_DIFF_HAS_NONCTX
// The non-Ctx entry points replay Ctx cases (same ids and inputs), so the
// recorded goldens cover both call paths.
TEST_F(NppiGammaDiff, NonCtx) {
  p3_case("fwd_plain17x13", "nppiGammaFwd_8u_P3R", 17, 13, 7, 5101, false, /*nonctx=*/true);
  p3_case("inv_plain17x13", "nppiGammaInv_8u_P3R", 17, 13, 7, 5301, true, /*nonctx=*/true);
  ip3_case("fwd_ip_plain17x13", "nppiGammaFwd_8u_IP3R", 17, 13, 7, 5201, false, /*nonctx=*/true);
}
#endif  // NPP_DIFF_HAS_NONCTX
