// Differential tests for the nppiCompColorKey family (package c4).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiCompColorKeyDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records ./diff_tests \
//     --gtest_filter='NppiCompColorKeyDiff.*'
//
// Semantics pinned here (NVIDIA NPP 13.4, 5090): a source-1 pixel whose every
// channel equals the color key is replaced by the source-2 pixel; C4 compares
// all four channels (alpha included).  Padding past the ROI is never written.
// Error paths: null pointer -> NPP_NULL_POINTER_ERROR, negative ROI ->
// NPP_SIZE_ERROR, empty ROI -> NPP_NO_ERROR (no write), step < ROI row length
// -> NPP_STEP_ERROR.

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

inline int align_step(int step, int esize) { return ((step + esize - 1) / esize) * esize; }

// ROI case: two uniform-random sources, a key drawn from the source-1 pixels so
// that some pixels match, padded row strides, whole buffer captured (proves the
// padding is untouched).
void roi_case(const char *fname, const std::string &idBase, int width, int height, int channels, int pad,
              uint64_t seed, bool nonctx = false) {
  const int esize = 1;
  const int step = align_step(width * channels * esize + pad, esize);
  const size_t elems = (size_t)height * step / esize;
  std::vector<Npp8u> hs1 = make_uniform<Npp8u>(elems, seed + 11);
  std::vector<Npp8u> hs2 = make_uniform<Npp8u>(elems, seed + 29);
  std::vector<Npp8u> hd = make_uniform<Npp8u>(elems, seed + 37);
  // Force a deterministic set of matching pixels (every 7th pixel).
  std::vector<Npp8u> key(channels);
  for (int c = 0; c < channels; ++c) {
    key[c] = (Npp8u)((seed * (c + 3)) & 0xFF);
  }
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if ((x + y) % 5 != 2) {
        continue;
      }
      const size_t base = ((size_t)y * step + (size_t)x * channels) / esize;
      for (int c = 0; c < channels; ++c) {
        hs1[base + c] = key[c];
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
  NppStatus st;
  if (channels == 1) {
    st = nonctx ? nppiCompColorKey_8u_C1R(s1.get(), step, s2.get(), step, d.get(), step, o, key[0])
                : nppiCompColorKey_8u_C1R_Ctx(s1.get(), step, s2.get(), step, d.get(), step, o, key[0],
                                              current_context());
  } else if (channels == 3) {
    st = nonctx ? nppiCompColorKey_8u_C3R(s1.get(), step, s2.get(), step, d.get(), step, o, key.data())
                : nppiCompColorKey_8u_C3R_Ctx(s1.get(), step, s2.get(), step, d.get(), step, o, key.data(),
                                              current_context());
  } else {
    st = nonctx ? nppiCompColorKey_8u_C4R(s1.get(), step, s2.get(), step, d.get(), step, o, key.data())
                : nppiCompColorKey_8u_C4R_Ctx(s1.get(), step, s2.get(), step, d.get(), step, o, key.data(),
                                              current_context());
  }
  const std::string id = "nppiCompColorKey/" + idBase + std::to_string(width) + "x" + std::to_string(height) + "c" +
                         std::to_string(channels);
  Case c(id, fname);
  c.input(hs1.data(), hs1.size());
  c.input(hs2.data(), hs2.size());
  c.input(key.data(), key.size());
  c.status(st);
  c.device_output<Npp8u>(d.get(), elems, u8_kind());
  c.finish();
}

void error_case(const char *fname, const std::string &id, int channels, Npp8u *nullPtr, NppiSize roi, int step) {
  const int width = 5, height = 3;
  const int stepBytes = step > 0 ? step : width * channels;
  const size_t elems = (size_t)height * stepBytes;
  std::vector<Npp8u> hs1 = make_uniform<Npp8u>(elems, 1234);
  std::vector<Npp8u> hs2 = make_uniform<Npp8u>(elems, 4321);
  std::vector<Npp8u> hd = make_uniform<Npp8u>(elems, 999);
  std::vector<Npp8u> key(channels, 7);
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
  NppStatus st;
  if (channels == 1) {
    st = nppiCompColorKey_8u_C1R_Ctx(pSrc1, step, pSrc2, step, pDst, step, o, key[0], current_context());
  } else if (channels == 3) {
    st = nppiCompColorKey_8u_C3R_Ctx(pSrc1, step, pSrc2, step, pDst, step, o, key.data(), current_context());
  } else {
    st = nppiCompColorKey_8u_C4R_Ctx(pSrc1, step, pSrc2, step, pDst, step, o, key.data(), current_context());
  }
  Case c("nppiCompColorKey/" + id, fname);
  c.status(st);
  c.device_output<Npp8u>(d.get(), elems, u8_kind());
  c.finish();
}

}  // namespace

class NppiCompColorKeyDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

TEST_F(NppiCompColorKeyDiff, C1R_Roi) {
  roi_case("nppiCompColorKey_8u_C1R", "plain", 64, 32, 1, 32, 101);
  roi_case("nppiCompColorKey_8u_C1R", "plain", 17, 13, 1, 16, 102);
  roi_case("nppiCompColorKey_8u_C1R", "tight", 1, 1, 1, 0, 103);
}

TEST_F(NppiCompColorKeyDiff, C3R_Roi) {
  roi_case("nppiCompColorKey_8u_C3R", "plain", 17, 13, 3, 16, 201);
  roi_case("nppiCompColorKey_8u_C3R", "tight", 33, 9, 3, 0, 202);
}

TEST_F(NppiCompColorKeyDiff, C4R_Roi) {
  roi_case("nppiCompColorKey_8u_C4R", "plain", 17, 13, 4, 16, 301);
  roi_case("nppiCompColorKey_8u_C4R", "tight", 9, 5, 4, 0, 302);
}

TEST_F(NppiCompColorKeyDiff, C1R_Errors) {
  error_case("nppiCompColorKey_8u_C1R", "c1/null_src1", 1, (Npp8u *)1, {5, 3}, 0);
  error_case("nppiCompColorKey_8u_C1R", "c1/null_src2", 1, (Npp8u *)2, {5, 3}, 0);
  error_case("nppiCompColorKey_8u_C1R", "c1/null_dst", 1, (Npp8u *)3, {5, 3}, 0);
  error_case("nppiCompColorKey_8u_C1R", "c1/neg_w", 1, nullptr, {-1, 3}, 0);
  error_case("nppiCompColorKey_8u_C1R", "c1/neg_h", 1, nullptr, {5, -2}, 0);
  error_case("nppiCompColorKey_8u_C1R", "c1/zero_w", 1, nullptr, {0, 3}, 0);
  error_case("nppiCompColorKey_8u_C1R", "c1/zero_h", 1, nullptr, {5, 0}, 0);
  error_case("nppiCompColorKey_8u_C1R", "c1/small_step", 1, nullptr, {5, 3}, 4);
  error_case("nppiCompColorKey_8u_C1R", "c1/zero_step", 1, nullptr, {5, 3}, 0);
  error_case("nppiCompColorKey_8u_C1R", "c1/big_step", 1, nullptr, {5, 3}, 512);
}

TEST_F(NppiCompColorKeyDiff, C3R_Errors) {
  error_case("nppiCompColorKey_8u_C3R", "c3/null_dst", 3, (Npp8u *)3, {5, 3}, 0);
  error_case("nppiCompColorKey_8u_C3R", "c3/neg_w", 3, nullptr, {-1, 3}, 0);
  error_case("nppiCompColorKey_8u_C3R", "c3/zero_roi", 3, nullptr, {0, 0}, 0);
  error_case("nppiCompColorKey_8u_C3R", "c3/small_step", 3, nullptr, {5, 3}, 13);
  error_case("nppiCompColorKey_8u_C3R", "c3/zero_step", 3, nullptr, {5, 3}, 0);
}

TEST_F(NppiCompColorKeyDiff, C4R_Errors) {
  error_case("nppiCompColorKey_8u_C4R", "c4/null_src1", 4, (Npp8u *)1, {5, 3}, 0);
  error_case("nppiCompColorKey_8u_C4R", "c4/neg_h", 4, nullptr, {5, -1}, 0);
  error_case("nppiCompColorKey_8u_C4R", "c4/small_step", 4, nullptr, {5, 3}, 17);
  error_case("nppiCompColorKey_8u_C4R", "c4/zero_step", 4, nullptr, {5, 3}, 0);
}

#if NPP_DIFF_HAS_NONCTX
// The non-Ctx entry points replay the Ctx ROI cases (same ids and inputs), so
// the recorded goldens cover both call paths.
TEST_F(NppiCompColorKeyDiff, NonCtx_C1R) {
  roi_case("nppiCompColorKey_8u_C1R", "plain", 17, 13, 1, 16, 102, /*nonctx=*/true);
}

TEST_F(NppiCompColorKeyDiff, NonCtx_C3R_C4R) {
  roi_case("nppiCompColorKey_8u_C3R", "plain", 17, 13, 3, 16, 201, /*nonctx=*/true);
  roi_case("nppiCompColorKey_8u_C4R", "plain", 17, 13, 4, 16, 301, /*nonctx=*/true);
}
#endif  // NPP_DIFF_HAS_NONCTX

