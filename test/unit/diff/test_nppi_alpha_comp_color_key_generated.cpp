// Differential tests for the nppiAlphaCompColorKey family (package c4).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiAlphaCompColorKeyDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records ./diff_tests \
//     --gtest_filter='NppiAlphaCompColorKeyDiff.*'
//
// Semantics pinned here (NVIDIA NPP 13.4, measured on the 5090):
//   * Source-1 pixels whose FOUR channels equal the color key take the
//     source-2 pixel verbatim (alpha included).  All other pixels blend their
//     RGB channels with the Porter-Duff operator; the destination ALPHA of a
//     blended pixel is NOT written.  Row padding is never written.
//   * The blend is the measured fixed-point model (round-half-down divisions,
//     alpha 255 promoted to 256, XOR coefficients clamped at 255) implemented
//     in nppi_color_key_executor.cu; the alpha-pair sweep below pins it.
//   * Premultiplied operators (6..12) and out-of-range operator values (<0 or
//     >5) are complete no-ops returning NPP_NO_ERROR *before* any validation
//     (they succeed even with null pointers or bad steps).
//   * Error paths: null pointer -> NPP_NULL_POINTER_ERROR, negative ROI ->
//     NPP_SIZE_ERROR, empty ROI -> NPP_NO_ERROR (short-circuits the step
//     checks), step == 0 or step < ROI row length -> NPP_STEP_ERROR, step not
//     a multiple of 4 (AC4 pixels) -> NPP_NOT_EVEN_STEP_ERROR.
// Negative steps are deliberately not recorded: NVIDIA returns NPP_NO_ERROR
// and launches a kernel that walks out-of-bounds rows (nondeterministic
// output); mcMPP returns NPP_NO_ERROR without launching.

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
  NppStreamContext ctx{};
  EXPECT_EQ(nppGetStreamContext(&ctx), NPP_NO_ERROR) << "nppGetStreamContext failed";
  return ctx;
}

ValueKind u8_kind() { return ValueKind::U8; }

const char *kFname = "nppiAlphaCompColorKey_8u_AC4R";

// Blend case: uniform-random sources, a 4-channel key with a distinctive alpha
// so key matches require all four channels.  Pixels with (x+y)%4==0 match the
// key exactly, pixels with (x+y)%4==1 match only RGB (near miss: alpha off by
// one) and must go through the blend path.  The whole buffer is captured so
// untouched alpha and padding stay visible.
void blend_case(const std::string &id, int width, int height, int pad, int a1, int a2, int op, uint64_t seed,
                bool nonctx = false) {
  const int channels = 4;
  const int step = width * channels + pad; // pad keeps the 4-byte alignment
  const size_t elems = (size_t)height * step;
  std::vector<Npp8u> hs1 = make_uniform<Npp8u>(elems, seed + 11);
  std::vector<Npp8u> hs2 = make_uniform<Npp8u>(elems, seed + 29);
  std::vector<Npp8u> hd = make_uniform<Npp8u>(elems, seed + 37);
  std::vector<Npp8u> key = {(Npp8u)((seed * 3) & 0xFF), (Npp8u)((seed * 5) & 0xFF), (Npp8u)((seed * 7) & 0xFF),
                            (Npp8u)200};
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t base = (size_t)y * step + (size_t)x * channels;
      if ((x + y) % 4 == 0) {
        for (int c = 0; c < channels; ++c) {
          hs1[base + c] = key[c];
        }
      } else if ((x + y) % 4 == 1) { // near miss: RGB matches, alpha differs
        for (int c = 0; c < 3; ++c) {
          hs1[base + c] = key[c];
        }
        hs1[base + 3] = (Npp8u)201;
      }
    }
  }
  DeviceMemory<Npp8u> &s1 = arena<Npp8u>(elems);
  DeviceMemory<Npp8u> &s2 = arena<Npp8u>(elems);
  DeviceMemory<Npp8u> &d = arena<Npp8u>(elems);
  s1.copyFromHost(hs1);
  s2.copyFromHost(hs2);
  d.copyFromHost(hd);
  NppiSize o{width, height};
  clear_cuda_error();
  NppStatus st =
      nonctx ? nppiAlphaCompColorKey_8u_AC4R(s1.get(), step, (Npp8u)a1, s2.get(), step, (Npp8u)a2, d.get(), step, o,
                                             key.data(), (NppiAlphaOp)op)
             : nppiAlphaCompColorKey_8u_AC4R_Ctx(s1.get(), step, (Npp8u)a1, s2.get(), step, (Npp8u)a2, d.get(), step, o,
                                                 key.data(), (NppiAlphaOp)op, current_context());
  Case c("nppiAlphaCompColorKey/" + id, kFname);
  c.input(hs1.data(), hs1.size());
  c.input(hs2.data(), hs2.size());
  c.input(key.data(), key.size());
  c.status(st);
  c.device_output<Npp8u>(d.get(), elems, u8_kind());
  c.finish();
}

// Error / no-op case: fixed 5x3 ROI, controllable pointers, ROI and steps.
void error_case(const std::string &id, Npp8u *nullPtr, NppiSize roi, int step, int op, bool nonctx = false) {
  const int width = 5, height = 3, channels = 4;
  const int stepBytes = step > 0 ? step : width * channels;
  const size_t elems = (size_t)height * stepBytes;
  std::vector<Npp8u> hs1 = make_uniform<Npp8u>(elems, 1234);
  std::vector<Npp8u> hs2 = make_uniform<Npp8u>(elems, 4321);
  std::vector<Npp8u> hd = make_uniform<Npp8u>(elems, 999);
  std::vector<Npp8u> key = {7, 8, 9, 10};
  DeviceMemory<Npp8u> &s1 = arena<Npp8u>(elems);
  DeviceMemory<Npp8u> &s2 = arena<Npp8u>(elems);
  DeviceMemory<Npp8u> &d = arena<Npp8u>(elems);
  s1.copyFromHost(hs1);
  s2.copyFromHost(hs2);
  d.copyFromHost(hd);
  Npp8u *pSrc1 = (nullPtr == (Npp8u *)1) ? nullptr : s1.get();
  Npp8u *pSrc2 = (nullPtr == (Npp8u *)2) ? nullptr : s2.get();
  Npp8u *pDst = (nullPtr == (Npp8u *)3) ? nullptr : d.get();
  NppiSize o = roi;
  clear_cuda_error();
  NppStatus st =
      nonctx ? nppiAlphaCompColorKey_8u_AC4R(pSrc1, step, 200, pSrc2, step, 100, pDst, step, o, key.data(), (NppiAlphaOp)op)
             : nppiAlphaCompColorKey_8u_AC4R_Ctx(pSrc1, step, 200, pSrc2, step, 100, pDst, step, o, key.data(),
                                                 (NppiAlphaOp)op, current_context());
  Case c("nppiAlphaCompColorKey/" + id, kFname);
  c.input(hs1.data(), hs1.size());
  c.input(hs2.data(), hs2.size());
  c.input(key.data(), key.size());
  c.status(st);
  c.device_output<Npp8u>(d.get(), elems, u8_kind());
  c.finish();
}

}  // namespace

class NppiAlphaCompColorKeyDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

// One sweep per Porter-Duff operator.  The alpha pairs pin the coefficient
// quantization: 255/0 corners (XOR clamps at 255 there), 255->256 promotion,
// and mixed quantized products.
TEST_F(NppiAlphaCompColorKeyDiff, OpOver) {
  const int pairs[][2] = {{255, 255}, {255, 0}, {0, 255}, {0, 0}, {128, 128}, {200, 100},
                          {100, 200}, {1, 255},  {255, 1}, {64, 192}, {192, 64}, {255, 128}};
  for (size_t i = 0; i < sizeof(pairs) / sizeof(pairs[0]); ++i) {
    blend_case("over_p" + std::to_string(pairs[i][0]) + "_" + std::to_string(pairs[i][1]), 19, 7, 8, pairs[i][0],
               pairs[i][1], NPPI_OP_ALPHA_OVER, 5000 + i);
  }
  blend_case("over_tight", 17, 13, 0, 173, 91, NPPI_OP_ALPHA_OVER, 5100);
  blend_case("over_small", 1, 1, 0, 255, 255, NPPI_OP_ALPHA_OVER, 5101);
}

TEST_F(NppiAlphaCompColorKeyDiff, OpIn) {
  const int pairs[][2] = {{255, 255}, {255, 0}, {0, 255}, {128, 128}, {200, 100}, {100, 200},
                          {1, 255},    {255, 1}, {64, 192}, {77, 88}, {255, 128}, {128, 255}};
  for (size_t i = 0; i < sizeof(pairs) / sizeof(pairs[0]); ++i) {
    blend_case("in_p" + std::to_string(pairs[i][0]) + "_" + std::to_string(pairs[i][1]), 19, 7, 8, pairs[i][0],
               pairs[i][1], NPPI_OP_ALPHA_IN, 6000 + i);
  }
}

TEST_F(NppiAlphaCompColorKeyDiff, OpOut) {
  const int pairs[][2] = {{255, 255}, {255, 0}, {0, 255}, {128, 128}, {200, 100}, {100, 200},
                          {1, 255},    {255, 1}, {64, 192}, {77, 88}, {255, 128}, {128, 255}};
  for (size_t i = 0; i < sizeof(pairs) / sizeof(pairs[0]); ++i) {
    blend_case("out_p" + std::to_string(pairs[i][0]) + "_" + std::to_string(pairs[i][1]), 19, 7, 8, pairs[i][0],
               pairs[i][1], NPPI_OP_ALPHA_OUT, 7000 + i);
  }
}

TEST_F(NppiAlphaCompColorKeyDiff, OpAtop) {
  const int pairs[][2] = {{255, 255}, {255, 0}, {0, 255}, {128, 128}, {200, 100}, {100, 200},
                          {1, 255},    {255, 1}, {64, 192}, {77, 88}, {255, 128}, {128, 255}};
  for (size_t i = 0; i < sizeof(pairs) / sizeof(pairs[0]); ++i) {
    blend_case("atop_p" + std::to_string(pairs[i][0]) + "_" + std::to_string(pairs[i][1]), 19, 7, 8, pairs[i][0],
               pairs[i][1], NPPI_OP_ALPHA_ATOP, 8000 + i);
  }
}

TEST_F(NppiAlphaCompColorKeyDiff, OpXor) {
  // The 255/0 and 0/255 corners are the ones where XOR's coefficients clamp
  // at 255 while OUT keeps 256 - both are covered below.
  const int pairs[][2] = {{255, 255}, {255, 0}, {0, 255}, {0, 0}, {128, 128}, {200, 100},
                          {100, 200}, {1, 255},  {255, 1}, {64, 192}, {192, 64}, {254, 1}};
  for (size_t i = 0; i < sizeof(pairs) / sizeof(pairs[0]); ++i) {
    blend_case("xor_p" + std::to_string(pairs[i][0]) + "_" + std::to_string(pairs[i][1]), 19, 7, 8, pairs[i][0],
               pairs[i][1], NPPI_OP_ALPHA_XOR, 9000 + i);
  }
}

TEST_F(NppiAlphaCompColorKeyDiff, OpPlus) {
  // PLUS saturates at 255 (e.g. alpha 255/255 with bright sources).
  const int pairs[][2] = {{255, 255}, {255, 0}, {0, 255}, {0, 0}, {128, 128}, {200, 100},
                          {100, 200}, {1, 255},  {255, 1}, {64, 192}, {192, 64}, {240, 240}};
  for (size_t i = 0; i < sizeof(pairs) / sizeof(pairs[0]); ++i) {
    blend_case("plus_p" + std::to_string(pairs[i][0]) + "_" + std::to_string(pairs[i][1]), 19, 7, 8, pairs[i][0],
               pairs[i][1], NPPI_OP_ALPHA_PLUS, 10000 + i);
  }
}

TEST_F(NppiAlphaCompColorKeyDiff, Errors) {
  error_case("null_src1", (Npp8u *)1, {5, 3}, 0, NPPI_OP_ALPHA_OVER);
  error_case("null_src2", (Npp8u *)2, {5, 3}, 0, NPPI_OP_ALPHA_OVER);
  error_case("null_dst", (Npp8u *)3, {5, 3}, 0, NPPI_OP_ALPHA_OVER);
  error_case("neg_w", nullptr, {-1, 3}, 0, NPPI_OP_ALPHA_OVER);
  error_case("neg_h", nullptr, {5, -2}, 0, NPPI_OP_ALPHA_OVER);
  error_case("zero_w", nullptr, {0, 3}, 0, NPPI_OP_ALPHA_OVER);
  error_case("zero_h", nullptr, {5, 0}, 0, NPPI_OP_ALPHA_OVER);
  // Empty ROI short-circuits the step checks (bad steps still return 0).
  error_case("zero_roi_bad_step", nullptr, {0, 0}, 7, NPPI_OP_ALPHA_OVER);
  error_case("zero_step", nullptr, {5, 3}, 0, NPPI_OP_ALPHA_OVER);
  error_case("small_step", nullptr, {5, 3}, 19, NPPI_OP_ALPHA_OVER);  // 19 < 20
  error_case("odd_step", nullptr, {5, 3}, 21, NPPI_OP_ALPHA_OVER);    // >= 20, not %4
  error_case("odd_step2", nullptr, {5, 3}, 22, NPPI_OP_ALPHA_OVER);
  error_case("aligned_step", nullptr, {5, 3}, 24, NPPI_OP_ALPHA_OVER); // >= 20, %4 == 0
  error_case("big_step", nullptr, {5, 3}, 512, NPPI_OP_ALPHA_OVER);
}

// Premultiplied (6..12) and out-of-range operators are complete no-ops that
// skip every validation, including null pointers and bad steps.
TEST_F(NppiAlphaCompColorKeyDiff, NoopOperators) {
  const int ops[] = {6, 7, 8, 9, 10, 11, 12, 13, 42, -1, -3};
  for (size_t i = 0; i < sizeof(ops) / sizeof(ops[0]); ++i) {
    const std::string suffix = std::to_string(ops[i]);
    std::string id = (ops[i] < 0 ? "negm" + suffix : "op" + suffix);
    error_case(id + "_valid", nullptr, {5, 3}, 0, ops[i]);
  }
  error_case("premul_null", (Npp8u *)1, {5, 3}, 0, 6);
  error_case("premul_neg_roi", nullptr, {-5, 3}, 0, 6);
  error_case("premul_bad_step", nullptr, {5, 3}, 19, 6);
  error_case("premul_zero_step", nullptr, {5, 3}, 0, 11);
  error_case("invalid_null", (Npp8u *)3, {5, 3}, 0, 99);
}

#if NPP_DIFF_HAS_NONCTX
// The non-Ctx entry point replays the same inputs under the Ctx case ids, so
// the recorded goldens cover both call paths.
TEST_F(NppiAlphaCompColorKeyDiff, NonCtx) {
  // Same call as the OpOver sweep entry over_p200_100 (seed 5005).
  blend_case("over_p200_100", 19, 7, 8, 200, 100, NPPI_OP_ALPHA_OVER, 5005, /*nonctx=*/true);
  // Same call as the OpXor sweep entry xor_p255_0 (seed 9001).
  blend_case("xor_p255_0", 19, 7, 8, 255, 0, NPPI_OP_ALPHA_XOR, 9001, /*nonctx=*/true);
  // Premultiplied operator through the non-Ctx entry point is still a no-op
  // (same as the op11_valid golden).
  error_case("op11_valid", nullptr, {5, 3}, 0, 11, /*nonctx=*/true);
}
#endif  // NPP_DIFF_HAS_NONCTX

