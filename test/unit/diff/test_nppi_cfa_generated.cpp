// Differential tests for the Bayer CFA demosaic functions of package c4:
//   nppiCFAToRGB_32u_C1C3R_Ctx
//   nppiCFAToRGBA_{8u,16u,32u}_C1AC4R[_Ctx]
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiCfaDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records ./diff_tests \
//     --gtest_filter='NppiCfaDiff.*'
//
// Semantics pinned here (NVIDIA NPP 13.4, 5090): integer bilinear light
// field with chroma-correlated green and parity-dependent diagonal pairing
// (see src/.../nppi_cfa_executor.h), source borders mirrored, destination
// written from its origin.  The oSrcROI width/height fields act as exclusive
// right/bottom edges (NVIDIA quirk), so the produced size is
// (width - x) x (height - y).  Error paths: null -> NPP_NULL_POINTER_ERROR,
// bad interpolation mode / grid -> NPP_BAD_ARGUMENT_ERROR, negative origin or
// ROI past oSrcSize -> NPP_OUT_OFF_RANGE_ERROR, non-positive or odd ROI
// (including odd origin) -> NPP_SIZE_ERROR, destination step below the row or
// not a whole number of pixels -> NPP_STEP_ERROR / NPP_NOT_EVEN_STEP_ERROR,
// negative step -> NPP_NO_ERROR with no write.

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <cstdint>
#include <memory>
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
  EXPECT_EQ(nppGetStreamContext(&ctx), NPP_NO_ERROR) << "nppGetStreamContext failed";
  return ctx;
}

template <typename T> ValueKind kind_for();
template <> ValueKind kind_for<Npp8u>() { return ValueKind::U8; }
template <> ValueKind kind_for<Npp16u>() { return ValueKind::U16; }
template <> ValueKind kind_for<Npp32u>() { return ValueKind::U32; }

constexpr int kSrcW = 16;
constexpr int kSrcH = 16;

// RGB (3-channel) case: 32-bit only.
void cfa_to_rgb_case(const std::string &id, NppiRect roi, int grid, int srcPadBytes, int dstPadBytes, uint64_t seed) {
  const int srcStep = kSrcW * 4 + srcPadBytes;
  const int dstStep = kSrcW * 3 * 4 + dstPadBytes;  // multiple of the 12-byte pixel
  const size_t srcElems = (size_t)(srcStep / 4) * kSrcH;
  const size_t dstElems = (size_t)(dstStep / 4) * kSrcH;
  std::vector<Npp32u> hs = make_uniform<Npp32u>(srcElems, seed + 3);
  std::vector<Npp32u> hd = make_uniform<Npp32u>(dstElems, seed + 7);
  DeviceMemory<Npp32u> &s = arena<Npp32u>(srcElems);
  DeviceMemory<Npp32u> &d = arena<Npp32u>(dstElems);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st = nppiCFAToRGB_32u_C1C3R_Ctx(s.get(), srcStep, NppiSize{kSrcW, kSrcH}, roi, d.get(), dstStep,
                                             (NppiBayerGridPosition)grid, NPPI_INTER_UNDEFINED, current_context());
  Case c("nppiCFAToRGB/32u_" + id, "nppiCFAToRGB_32u_C1C3R");
  c.input(hs.data(), hs.size() * 4);
  c.input(hd.data(), hd.size() * 4);
  c.status(st);
  c.device_output<Npp32u>(d.get(), dstElems, ValueKind::U32);
  c.finish();
}

// RGBA case for T = Npp8u / Npp16u / Npp32u.  The destination step is always
// a whole number of 4-element pixels (NPP requires that).
template <typename T>
void cfa_to_rgba_case(const std::string &id, NppiRect roi, int grid, int srcPadElems, int dstPadPixels,
                      uint64_t seed, bool nonctx = false) {
  const int srcStep = (kSrcW + srcPadElems) * (int)sizeof(T);
  const int dstStep = (kSrcW * 4 + dstPadPixels * 4) * (int)sizeof(T);
  const size_t srcElems = (size_t)(srcStep / sizeof(T)) * kSrcH;
  const size_t dstElems = (size_t)(dstStep / sizeof(T)) * kSrcH;
  std::vector<T> hs = make_uniform<T>(srcElems, seed + 3);
  std::vector<T> hd = make_uniform<T>(dstElems, seed + 7);
  DeviceMemory<T> &s = arena<T>(srcElems);
  DeviceMemory<T> &d = arena<T>(dstElems);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  const T alpha = (T)(sizeof(T) == 1 ? 201 : (sizeof(T) == 2 ? 5150 : 0xDEADBEEFu));
  clear_cuda_error();
  NppStatus st;
  const std::string fname = std::string("nppiCFAToRGBA_") + (sizeof(T) == 1 ? "8u" : (sizeof(T) == 2 ? "16u" : "32u")) +
                            "_C1AC4R";
#if NPP_DIFF_HAS_NONCTX
  if constexpr (sizeof(T) == 1) {
    if (nonctx) {
      st = nppiCFAToRGBA_8u_C1AC4R(s.get(), srcStep, NppiSize{kSrcW, kSrcH}, roi, d.get(), dstStep,
                                   (NppiBayerGridPosition)grid, NPPI_INTER_UNDEFINED, (Npp8u)alpha);
    } else {
      st = nppiCFAToRGBA_8u_C1AC4R_Ctx(s.get(), srcStep, NppiSize{kSrcW, kSrcH}, roi, d.get(), dstStep,
                                       (NppiBayerGridPosition)grid, NPPI_INTER_UNDEFINED, alpha, current_context());
    }
  } else if constexpr (sizeof(T) == 2) {
    if (nonctx) {
      st = nppiCFAToRGBA_16u_C1AC4R(s.get(), srcStep, NppiSize{kSrcW, kSrcH}, roi, d.get(), dstStep,
                                    (NppiBayerGridPosition)grid, NPPI_INTER_UNDEFINED, (Npp16u)alpha);
    } else {
      st = nppiCFAToRGBA_16u_C1AC4R_Ctx(s.get(), srcStep, NppiSize{kSrcW, kSrcH}, roi, d.get(), dstStep,
                                        (NppiBayerGridPosition)grid, NPPI_INTER_UNDEFINED, alpha, current_context());
    }
  } else {
    st = nppiCFAToRGBA_32u_C1AC4R_Ctx(s.get(), srcStep, NppiSize{kSrcW, kSrcH}, roi, d.get(), dstStep,
                                      (NppiBayerGridPosition)grid, NPPI_INTER_UNDEFINED, alpha, current_context());
  }
  (void)nonctx;
#else
  if constexpr (sizeof(T) == 1) {
    st = nppiCFAToRGBA_8u_C1AC4R_Ctx(s.get(), srcStep, NppiSize{kSrcW, kSrcH}, roi, d.get(), dstStep,
                                     (NppiBayerGridPosition)grid, NPPI_INTER_UNDEFINED, alpha, current_context());
  } else if constexpr (sizeof(T) == 2) {
    st = nppiCFAToRGBA_16u_C1AC4R_Ctx(s.get(), srcStep, NppiSize{kSrcW, kSrcH}, roi, d.get(), dstStep,
                                      (NppiBayerGridPosition)grid, NPPI_INTER_UNDEFINED, alpha, current_context());
  } else {
    st = nppiCFAToRGBA_32u_C1AC4R_Ctx(s.get(), srcStep, NppiSize{kSrcW, kSrcH}, roi, d.get(), dstStep,
                                      (NppiBayerGridPosition)grid, NPPI_INTER_UNDEFINED, alpha, current_context());
  }
#endif
  Case c("nppiCFAToRGBA/8u_16u_32u/" + id, fname);
  c.input(hs.data(), hs.size() * sizeof(T));
  c.input(hd.data(), hd.size() * sizeof(T));
  c.status(st);
  c.device_output<T>(d.get(), dstElems, kind_for<T>());
  c.finish();
}

// Error paths for the 8u RGBA variant.
void cfa_to_rgba_error(const std::string &id, const Npp8u *src, Npp8u *dst, NppiRect roi, int dstStep,
                       NppiInterpolationMode interp = NPPI_INTER_UNDEFINED,
                       NppiBayerGridPosition grid = NPPI_BAYER_RGGB) {
  const int srcStep = kSrcW;
  const size_t srcElems = (size_t)srcStep * kSrcH;
  const int allocStep = dstStep > 0 ? dstStep : 32;
  const size_t dstElems = (size_t)allocStep * kSrcH;
  std::vector<Npp8u> hs = make_uniform<Npp8u>(srcElems, 91);
  std::vector<Npp8u> hd = make_uniform<Npp8u>(dstElems, 92);
  DeviceMemory<Npp8u> &s = arena<Npp8u>(srcElems);
  DeviceMemory<Npp8u> &d = arena<Npp8u>(dstElems);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  const Npp8u *pSrc = (src == (const Npp8u *)1) ? nullptr : s.get();
  Npp8u *pDst = (dst == (Npp8u *)1) ? nullptr : d.get();
  clear_cuda_error();
  NppStatus st = nppiCFAToRGBA_8u_C1AC4R_Ctx(pSrc, srcStep, NppiSize{kSrcW, kSrcH}, roi, pDst, dstStep, grid,
                                             interp, 33, current_context());
  Case c("nppiCFAToRGBA/8u_" + id, "nppiCFAToRGBA_8u_C1AC4R");
  c.status(st);
  c.device_output<Npp8u>(d.get(), dstElems, ValueKind::U8);
  c.finish();
}

}  // namespace

class NppiCfaDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

TEST_F(NppiCfaDiff, CfaToRgb32u_Roi) {
  for (int grid = 0; grid < 4; ++grid) {
    cfa_to_rgb_case("full_g" + std::to_string(grid), NppiRect{0, 0, kSrcW, kSrcH}, grid, 0, 4, 7100 + grid);
  }
  cfa_to_rgb_case("offset_g1", NppiRect{2, 2, 8, 8}, 1, 4, 8, 7201);
  cfa_to_rgb_case("offset_g2", NppiRect{2, 2, 10, 10}, 2, 0, 0, 7202);
  cfa_to_rgb_case("offset16_g3", NppiRect{4, 4, 12, 12}, 3, 0, 16, 7203);
}

TEST_F(NppiCfaDiff, CfaToRgba8u_Roi) {
  for (int grid = 0; grid < 4; ++grid) {
    cfa_to_rgba_case<Npp8u>("8u_full_g" + std::to_string(grid), NppiRect{0, 0, kSrcW, kSrcH}, grid, 0, 0,
                            7300 + grid);
  }
  cfa_to_rgba_case<Npp8u>("8u_offset_g1", NppiRect{2, 2, 8, 8}, 1, 1, 1, 7311);
  cfa_to_rgba_case<Npp8u>("8u_offset_g3", NppiRect{2, 2, 10, 10}, 3, 0, 2, 7312);
}

TEST_F(NppiCfaDiff, CfaToRgba16u_Roi) {
  for (int grid = 0; grid < 4; ++grid) {
    cfa_to_rgba_case<Npp16u>("16u_full_g" + std::to_string(grid), NppiRect{0, 0, kSrcW, kSrcH}, grid, 1, 1,
                             7400 + grid);
  }
  cfa_to_rgba_case<Npp16u>("16u_offset_g2", NppiRect{2, 2, 10, 10}, 2, 0, 2, 7411);
}

TEST_F(NppiCfaDiff, CfaToRgba32u_Roi) {
  for (int grid = 0; grid < 4; ++grid) {
    cfa_to_rgba_case<Npp32u>("32u_full_g" + std::to_string(grid), NppiRect{0, 0, kSrcW, kSrcH}, grid, 1, 1,
                             7500 + grid);
  }
  cfa_to_rgba_case<Npp32u>("32u_offset_g1", NppiRect{2, 2, 10, 10}, 1, 0, 2, 7511);
}

TEST_F(NppiCfaDiff, CfaToRgba8u_Errors) {
  cfa_to_rgba_error("null_src", (const Npp8u *)1, nullptr, {0, 0, 8, 8}, 32);
  cfa_to_rgba_error("null_dst", nullptr, (Npp8u *)1, {0, 0, 8, 8}, 32);
  cfa_to_rgba_error("bad_interp", nullptr, nullptr, {0, 0, 8, 8}, 32, NPPI_INTER_LINEAR);
  cfa_to_rgba_error("bad_grid", nullptr, nullptr, {0, 0, 8, 8}, 32, NPPI_INTER_UNDEFINED,
                    (NppiBayerGridPosition)9);
  cfa_to_rgba_error("neg_origin", nullptr, nullptr, {-2, 0, 8, 8}, 32);
  cfa_to_rgba_error("odd_w", nullptr, nullptr, {0, 0, 7, 8}, 32);
  cfa_to_rgba_error("odd_h", nullptr, nullptr, {0, 0, 8, 7}, 32);
  cfa_to_rgba_error("odd_origin", nullptr, nullptr, {1, 0, 8, 8}, 32);
  cfa_to_rgba_error("zero_roi", nullptr, nullptr, {0, 0, 0, 0}, 32);
  cfa_to_rgba_error("past_src", nullptr, nullptr, {8, 8, 12, 12}, 32);
  cfa_to_rgba_error("small_step", nullptr, nullptr, {0, 0, 8, 8}, 28);
  cfa_to_rgba_error("unaligned_step", nullptr, nullptr, {0, 0, 8, 8}, 34);
  // A negative destination step makes NVIDIA launch and write row 0 at the
  // buffer origin (and the remaining rows out of bounds); mcMPP returns
  // NPP_NO_ERROR without launching, so this deviation is not golden-tested.
}

#if NPP_DIFF_HAS_NONCTX
TEST_F(NppiCfaDiff, NonCtx_Replay) {
  cfa_to_rgba_case<Npp8u>("8u_offset_g1", NppiRect{2, 2, 8, 8}, 1, 1, 1, 7311, /*nonctx=*/true);
  cfa_to_rgba_case<Npp16u>("16u_offset_g2", NppiRect{2, 2, 10, 10}, 2, 0, 2, 7411, /*nonctx=*/true);
}
#endif  // NPP_DIFF_HAS_NONCTX
