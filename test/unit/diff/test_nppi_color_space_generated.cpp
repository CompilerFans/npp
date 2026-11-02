// Differential tests for the CIE color-space families of package c4:
//   nppiRGBToXYZ, nppiXYZToRGB, nppiRGBToLUV, nppiLUVToRGB (8u C3R / AC4R).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiColorSpaceDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records ./diff_tests \
//     --gtest_filter='NppiColorSpaceDiff.*'
//
// Semantics pinned here (NVIDIA NPP 13.4, 5090): the conversions follow the
// formulas published in nppi_color_conversion.h with 8-bit truncation; the
// AC4 destination alpha is preserved and row padding past the ROI is never
// written.  Error paths: null pointer -> NPP_NULL_POINTER_ERROR (checked
// before the ROI), negative width/height -> NPP_SIZE_ERROR, empty ROI ->
// NPP_NO_ERROR (short-circuits the step checks), step < ROI row length ->
// NPP_STEP_ERROR, AC4 step % 4 != 0 -> NPP_NOT_EVEN_STEP_ERROR (C3 has no
// alignment requirement), negative step -> NPP_NO_ERROR (no launch).

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <cstdint>
#include <memory>
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
  NppStreamContext ctx{};
  EXPECT_EQ(nppGetStreamContext(&ctx), NPP_NO_ERROR) << "nppGetStreamContext failed";
  return ctx;
}

ValueKind u8_kind() { return ValueKind::U8; }

using CtxFn = NppStatus (*)(const Npp8u *, int, Npp8u *, int, NppiSize, NppStreamContext);
using PlainFn = NppStatus (*)(const Npp8u *, int, Npp8u *, int, NppiSize);

struct Family {
  const char *name;
  CtxFn ctx3;
  CtxFn ctx4;
};

Family families[] = {
    {"nppiRGBToXYZ", nppiRGBToXYZ_8u_C3R_Ctx, nppiRGBToXYZ_8u_AC4R_Ctx},
    {"nppiXYZToRGB", nppiXYZToRGB_8u_C3R_Ctx, nppiXYZToRGB_8u_AC4R_Ctx},
    {"nppiRGBToLUV", nppiRGBToLUV_8u_C3R_Ctx, nppiRGBToLUV_8u_AC4R_Ctx},
    {"nppiLUVToRGB", nppiLUVToRGB_8u_C3R_Ctx, nppiLUVToRGB_8u_AC4R_Ctx},
};

#if NPP_DIFF_HAS_NONCTX
PlainFn plain_fns[] = {
    nppiRGBToXYZ_8u_C3R, nppiRGBToXYZ_8u_AC4R,
    nppiXYZToRGB_8u_C3R, nppiXYZToRGB_8u_AC4R,
    nppiRGBToLUV_8u_C3R, nppiRGBToLUV_8u_AC4R,
    nppiLUVToRGB_8u_C3R, nppiLUVToRGB_8u_AC4R,
};
#endif

// Conv case: random source, destination pre-filled with a different random
// pattern; the whole (padded) buffer is captured so the tests also prove that
// the row padding and the AC4 alpha channel are untouched.
void conv_case(int family, const std::string &idBase, int width, int height, int channels, int pad, uint64_t seed,
               bool nonctx = false) {
  const int step = width * channels + pad;
  const size_t elems = (size_t)height * step;
  std::vector<Npp8u> hs = make_uniform<Npp8u>(elems, seed + 11);
  std::vector<Npp8u> hd = make_uniform<Npp8u>(elems, seed + 37);
  DeviceMemory<Npp8u> &s = arena<Npp8u>(elems);
  DeviceMemory<Npp8u> &d = arena<Npp8u>(elems);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  NppiSize o{width, height};
  clear_cuda_error();
  NppStatus st;
  const Family &f = families[family];
#if NPP_DIFF_HAS_NONCTX
  if (nonctx) {
    PlainFn plain = plain_fns[2 * family + (channels == 4 ? 1 : 0)];
    st = plain(s.get(), step, d.get(), step, o);
  } else
#endif
  {
    st = (channels == 4) ? f.ctx4(s.get(), step, d.get(), step, o, current_context())
                         : f.ctx3(s.get(), step, d.get(), step, o, current_context());
  }
  const std::string id = std::string(f.name) + "/" + idBase + std::to_string(width) + "x" + std::to_string(height) +
                         "c" + std::to_string(channels);
  Case c(id, std::string(f.name) + (channels == 4 ? "_8u_AC4R" : "_8u_C3R"));
  c.input(hs.data(), hs.size());
  c.input(hd.data(), hd.size());
  c.status(st);
  c.device_output<Npp8u>(d.get(), elems, u8_kind());
  c.finish();
}

// Error paths.  nullIndex picks which pointer is replaced by nullptr (0 = none).
void error_case(int family, const std::string &id, int channels, int nullIndex, NppiSize roi, int step) {
  const int width = 5, height = 3;
  const int stepBytes = step > 0 ? step : width * channels;
  const size_t elems = (size_t)height * stepBytes;
  std::vector<Npp8u> hs = make_uniform<Npp8u>(elems, 4242);
  std::vector<Npp8u> hd = make_uniform<Npp8u>(elems, 2424);
  DeviceMemory<Npp8u> &s = arena<Npp8u>(elems);
  DeviceMemory<Npp8u> &d = arena<Npp8u>(elems);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  Npp8u *pSrc = (nullIndex == 1) ? nullptr : s.get();
  Npp8u *pDst = (nullIndex == 2) ? nullptr : d.get();
  const Family &f = families[family];
  clear_cuda_error();
  NppStatus st = (channels == 4) ? f.ctx4(pSrc, step, pDst, step, roi, current_context())
                                 : f.ctx3(pSrc, step, pDst, step, roi, current_context());
  Case c(std::string(f.name) + "/c" + std::to_string(channels) + "_" + id,
         std::string(f.name) + (channels == 4 ? "_8u_AC4R" : "_8u_C3R"));
  c.status(st);
  c.device_output<Npp8u>(d.get(), elems, u8_kind());
  c.finish();
}

void error_matrix(int family) {
  // Null pointers win over the ROI checks (measured on the 5090).
  error_case(family, "null_src", 3, 1, {-1, 3}, 0);
  error_case(family, "null_dst", 3, 2, {-1, 3}, 0);
  error_case(family, "null_src", 4, 1, {5, 3}, 0);
  error_case(family, "null_dst", 4, 2, {5, 3}, 0);
  // Negative ROI.
  error_case(family, "neg_w", 3, 0, {-1, 3}, 0);
  error_case(family, "neg_h", 3, 0, {5, -2}, 0);
  error_case(family, "neg_w", 4, 0, {-1, 3}, 0);
  // Empty ROI short-circuits the step checks.
  error_case(family, "zero_roi_bad_step", 3, 0, {0, 0}, 1);
  error_case(family, "zero_h_bad_step", 3, 0, {5, 0}, 0);
  error_case(family, "zero_roi_bad_step", 4, 0, {0, 0}, 1);
  // Step checks.
  error_case(family, "small_step", 3, 0, {5, 3}, 14);
  error_case(family, "zero_step", 3, 0, {5, 3}, 0);
  error_case(family, "exact_step", 3, 0, {5, 3}, 15);
  error_case(family, "small_step", 4, 0, {5, 3}, 19);
  error_case(family, "zero_step", 4, 0, {5, 3}, 0);
  // AC4 alignment: step >= width*4 but not a multiple of 4.
  error_case(family, "odd_step", 4, 0, {5, 3}, 22);
  // C3 has no step alignment requirement.
  error_case(family, "odd_step", 3, 0, {5, 3}, 17);
}

}  // namespace

class NppiColorSpaceDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

TEST_F(NppiColorSpaceDiff, RGBToXYZ_Roi) {
  conv_case(0, "plain", 64, 32, 3, 16, 1101);
  conv_case(0, "plain", 17, 13, 3, 13, 1102);
  conv_case(0, "tight", 1, 1, 3, 0, 1103);
  conv_case(0, "tight", 33, 9, 3, 0, 1104);
  conv_case(0, "plain", 64, 32, 4, 16, 1201);
  conv_case(0, "plain", 17, 13, 4, 12, 1202);
  conv_case(0, "tight", 1, 1, 4, 0, 1203);
  conv_case(0, "tight", 33, 9, 4, 0, 1204);
}

TEST_F(NppiColorSpaceDiff, XYZToRGB_Roi) {
  conv_case(1, "plain", 64, 32, 3, 16, 2101);
  conv_case(1, "plain", 17, 13, 3, 13, 2102);
  conv_case(1, "tight", 1, 1, 3, 0, 2103);
  conv_case(1, "tight", 33, 9, 3, 0, 2104);
  conv_case(1, "plain", 64, 32, 4, 16, 2201);
  conv_case(1, "plain", 17, 13, 4, 12, 2202);
  conv_case(1, "tight", 1, 1, 4, 0, 2203);
  conv_case(1, "tight", 33, 9, 4, 0, 2204);
}

TEST_F(NppiColorSpaceDiff, RGBToLUV_Roi) {
  conv_case(2, "plain", 64, 32, 3, 16, 3101);
  conv_case(2, "plain", 17, 13, 3, 13, 3102);
  conv_case(2, "tight", 1, 1, 3, 0, 3103);
  conv_case(2, "tight", 33, 9, 3, 0, 3104);
  conv_case(2, "plain", 64, 32, 4, 16, 3201);
  conv_case(2, "plain", 17, 13, 4, 12, 3202);
  conv_case(2, "tight", 1, 1, 4, 0, 3203);
  conv_case(2, "tight", 33, 9, 4, 0, 3204);
}

TEST_F(NppiColorSpaceDiff, LUVToRGB_Roi) {
  conv_case(3, "plain", 64, 32, 3, 16, 4101);
  conv_case(3, "plain", 17, 13, 3, 13, 4102);
  conv_case(3, "tight", 1, 1, 3, 0, 4103);
  conv_case(3, "tight", 33, 9, 3, 0, 4104);
  conv_case(3, "plain", 64, 32, 4, 16, 4201);
  conv_case(3, "plain", 17, 13, 4, 12, 4202);
  conv_case(3, "tight", 1, 1, 4, 0, 4203);
  conv_case(3, "tight", 33, 9, 4, 0, 4204);
}

TEST_F(NppiColorSpaceDiff, RGBToXYZ_Errors) { error_matrix(0); }
TEST_F(NppiColorSpaceDiff, XYZToRGB_Errors) { error_matrix(1); }
TEST_F(NppiColorSpaceDiff, RGBToLUV_Errors) { error_matrix(2); }
TEST_F(NppiColorSpaceDiff, LUVToRGB_Errors) { error_matrix(3); }

#if NPP_DIFF_HAS_NONCTX
// The non-Ctx entry points replay the Ctx ROI cases (same ids and inputs), so
// the recorded goldens cover both call paths.
TEST_F(NppiColorSpaceDiff, NonCtx_Replay) {
  conv_case(0, "plain", 17, 13, 3, 13, 1102, /*nonctx=*/true);
  conv_case(0, "plain", 17, 13, 4, 12, 1202, /*nonctx=*/true);
  conv_case(1, "plain", 17, 13, 3, 13, 2102, /*nonctx=*/true);
  conv_case(1, "plain", 17, 13, 4, 12, 2202, /*nonctx=*/true);
  conv_case(2, "plain", 17, 13, 3, 13, 3102, /*nonctx=*/true);
  conv_case(2, "plain", 17, 13, 4, 12, 3202, /*nonctx=*/true);
  conv_case(3, "plain", 17, 13, 3, 13, 4102, /*nonctx=*/true);
  conv_case(3, "plain", 17, 13, 4, 12, 4202, /*nonctx=*/true);
}
#endif  // NPP_DIFF_HAS_NONCTX
