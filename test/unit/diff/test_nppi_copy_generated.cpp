// Differential tests for the nppiCopy variants package (p2.1b): packed AC4R,
// masked (MR) and channel-conversion (C1Cx / CxC1 / CxCR) nppiCopy functions.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiCopyDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppiCopyDiff.*'
//
// Padded row strides capture the row-end padding, proving data outside
// oSizeROI (including the untouched alpha channel for AC4) is preserved.

#include "npp_diff_harness.h"
#include "npp_test_base.h"

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

NppStreamContext current_ctx() {
  NppStreamContext c{};
  NppStatus st = nppGetStreamContext(&c);
  EXPECT_EQ(st, NPP_NO_ERROR);
  return c;
}

int align_step(int step, int es) { return ((step + es - 1) / es) * es; }

template <typename D> ValueKind out_kind() {
  if constexpr (std::is_same_v<D, Npp8s>) return ValueKind::I8;
  else if constexpr (std::is_same_v<D, Npp8u>) return ValueKind::U8;
  else if constexpr (std::is_same_v<D, Npp16s>) return ValueKind::I16;
  else if constexpr (std::is_same_v<D, Npp16u>) return ValueKind::U16;
  else if constexpr (std::is_same_v<D, Npp32s>) return ValueKind::I32;
  else if constexpr (std::is_same_v<D, Npp32f>) return ValueKind::F32;
  else if constexpr (std::is_same_v<D, Npp16f>) return ValueKind::F16;
  else if constexpr (std::is_same_v<D, Npp16sc>) return ValueKind::I16Tol;
  else if constexpr (std::is_same_v<D, Npp32sc>) return ValueKind::I32Tol;
  else if constexpr (std::is_same_v<D, Npp32fc>) return ValueKind::CF32;
  else return ValueKind::Bytes;
}

// Same per-pixel stride on source and destination (plain / AC4 / selected-channel).
template <typename T>
void roi_stride(const char *fname, const std::string &idBase, int width, int height, int stride, int pad,
                NppStatus (*f)(const T *, int, T *, int, NppiSize, NppStreamContext), uint64_t seed) {
  const int es = (int)sizeof(T);
  const int sStep = align_step(width * stride * es + pad, es);
  const int dStep = align_step(width * stride * es + pad, es);
  const size_t se = (size_t)height * sStep / es;
  const size_t de = (size_t)height * dStep / es;
  std::vector<T> hs = make_uniform<T>(se, seed + 11);
  std::vector<T> hd = make_uniform<T>(de, seed + 29);
  DeviceMemory<T> &s = arena<T>(se);
  DeviceMemory<T> &d = arena<T>(de);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  NppiSize o{width, height};
  clear_cuda_error();
  NppStatus st = f(s.get(), sStep, d.get(), dStep, o, current_ctx());
  std::string id = "nppiCopy/" + idBase + std::to_string(width) + "x" + std::to_string(height);
  Case c(id, fname);
  c.input(hs.data(), hs.size() * sizeof(T));
  c.status(st);
  c.device_output<T>(d.get(), de, out_kind<T>(), 1e-5, 1e-6);
  c.finish();
}

// Insert / extract channel conversion (different per-pivot strides).
template <typename T>
void roi_conv(const char *fname, const std::string &idBase, int width, int height, int srcStride, int dstStride,
              int pad, NppStatus (*f)(const T *, int, T *, int, NppiSize, NppStreamContext), uint64_t seed) {
  const int es = (int)sizeof(T);
  const int sStep = align_step(width * srcStride * es + pad, es);
  const int dStep = align_step(width * dstStride * es + pad, es);
  const size_t se = (size_t)height * sStep / es;
  const size_t de = (size_t)height * dStep / es;
  std::vector<T> hs = make_uniform<T>(se, seed + 11);
  std::vector<T> hd = make_uniform<T>(de, seed + 29);
  DeviceMemory<T> &s = arena<T>(se);
  DeviceMemory<T> &d = arena<T>(de);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  NppiSize o{width, height};
  clear_cuda_error();
  NppStatus st = f(s.get(), sStep, d.get(), dStep, o, current_ctx());
  std::string id = "nppiCopy/" + idBase + std::to_string(width) + "x" + std::to_string(height);
  Case c(id, fname);
  c.input(hs.data(), hs.size() * sizeof(T));
  c.status(st);
  c.device_output<T>(d.get(), de, out_kind<T>(), 1e-5, 1e-6);
  c.finish();
}

// Masked copy with an 8u mask (1 byte per pixel).
template <typename T>
void roi_masked(const char *fname, const std::string &idBase, int width, int height, int stride, int pad,
                NppStatus (*f)(const T *, int, T *, int, NppiSize, const Npp8u *, int, NppStreamContext),
                uint64_t seed) {
  const int es = (int)sizeof(T);
  const int sStep = align_step(width * stride * es + pad, es);
  const int dStep = align_step(width * stride * es + pad, es);
  const size_t se = (size_t)height * sStep / es;
  const size_t de = (size_t)height * dStep / es;
  const int mStep = width + 4;
  const size_t me = (size_t)height * mStep;
  std::vector<T> hs = make_uniform<T>(se, seed + 11);
  std::vector<T> hd = make_uniform<T>(de, seed + 29);
  std::vector<Npp8u> hm = make_uniform<Npp8u>(me, seed + 7);
  for (auto &v : hm) v = (Npp8u)(v & 3);  // mix of 0 and non-zero
  DeviceMemory<T> &s = arena<T>(se);
  DeviceMemory<T> &d = arena<T>(de);
  DeviceMemory<Npp8u> &m = arena<Npp8u>(me);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  m.copyFromHost(hm);
  NppiSize o{width, height};
  clear_cuda_error();
  NppStatus st = f(s.get(), sStep, d.get(), dStep, o, m.get(), mStep, current_ctx());
  std::string id = "nppiCopy/" + idBase + std::to_string(width) + "x" + std::to_string(height);
  Case c(id, fname);
  c.input(hs.data(), hs.size() * sizeof(T));
  c.input(hm.data(), hm.size());
  c.status(st);
  c.device_output<T>(d.get(), de, out_kind<T>(), 1e-5, 1e-6);
  c.finish();
}

}  // namespace

class NppiCopyDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

// ===== Plain packed (new dtypes) =============================================
TEST_F(NppiCopyDiff, Plain_16f) {
  roi_stride("nppiCopy_16f_C1R", "16f_C1R", 17, 13, 1, 16, &nppiCopy_16f_C1R_Ctx, 100);
  roi_stride("nppiCopy_16f_C1R", "16f_C1R", 64, 32, 1, 31, &nppiCopy_16f_C1R_Ctx, 101);
  roi_stride("nppiCopy_16f_C3R", "16f_C3R", 1, 1, 3, 16, &nppiCopy_16f_C3R_Ctx, 110);
  roi_stride("nppiCopy_16f_C3R", "16f_C3R", 17, 13, 3, 16, &nppiCopy_16f_C3R_Ctx, 111);
  roi_stride("nppiCopy_16f_C4R", "16f_C4R", 64, 32, 4, 32, &nppiCopy_16f_C4R_Ctx, 120);
}
TEST_F(NppiCopyDiff, Plain_8s) {
  roi_stride("nppiCopy_8s_C1R", "8s_C1R", 17, 13, 1, 16, &nppiCopy_8s_C1R_Ctx, 130);
  roi_stride("nppiCopy_8s_C2R", "8s_C2R", 1, 1, 2, 16, &nppiCopy_8s_C2R_Ctx, 131);
  roi_stride("nppiCopy_8s_C3R", "8s_C3R", 17, 13, 3, 16, &nppiCopy_8s_C3R_Ctx, 132);
  roi_stride("nppiCopy_8s_C4R", "8s_C4R", 64, 32, 4, 32, &nppiCopy_8s_C4R_Ctx, 133);
}
TEST_F(NppiCopyDiff, Plain_complex) {
  roi_stride("nppiCopy_16sc_C2R", "16sc_C2R", 17, 13, 2, 16, &nppiCopy_16sc_C2R_Ctx, 140);
  roi_stride("nppiCopy_16sc_C4R", "16sc_C4R", 1, 1, 4, 16, &nppiCopy_16sc_C4R_Ctx, 141);
  roi_stride("nppiCopy_32sc_C1R", "32sc_C1R", 17, 13, 1, 16, &nppiCopy_32sc_C1R_Ctx, 142);
  roi_stride("nppiCopy_32fc_C1R", "32fc_C1R", 17, 13, 1, 16, &nppiCopy_32fc_C1R_Ctx, 143);
  roi_stride("nppiCopy_32fc_C3R", "32fc_C3R", 64, 32, 3, 16, &nppiCopy_32fc_C3R_Ctx, 144);
}

// ===== AC4R (copies 3 of 4 channels) ========================================
TEST_F(NppiCopyDiff, AC4) {
  roi_stride("nppiCopy_8u_AC4R", "8u_AC4R", 17, 13, 4, 16, &nppiCopy_8u_AC4R_Ctx, 200);
  roi_stride("nppiCopy_8u_AC4R", "8u_AC4R", 64, 32, 4, 32, &nppiCopy_8u_AC4R_Ctx, 201);
  roi_stride("nppiCopy_8s_AC4R", "8s_AC4R", 17, 13, 4, 16, &nppiCopy_8s_AC4R_Ctx, 202);
  roi_stride("nppiCopy_16s_AC4R", "16s_AC4R", 64, 32, 4, 32, &nppiCopy_16s_AC4R_Ctx, 203);
  roi_stride("nppiCopy_16u_AC4R", "16u_AC4R", 17, 13, 4, 16, &nppiCopy_16u_AC4R_Ctx, 204);
  roi_stride("nppiCopy_32s_AC4R", "32s_AC4R", 1, 1, 4, 16, &nppiCopy_32s_AC4R_Ctx, 205);
  roi_stride("nppiCopy_32f_AC4R", "32f_AC4R", 17, 13, 4, 16, &nppiCopy_32f_AC4R_Ctx, 206);
}

// ===== Channel conversion ====================================================
TEST_F(NppiCopyDiff, Insert_C1C2) {
  roi_conv("nppiCopy_32f_C1C2R", "32f_C1C2R", 17, 13, 1, 2, 16, &nppiCopy_32f_C1C2R_Ctx, 300);
}
TEST_F(NppiCopyDiff, Insert_C1C3) {
  roi_conv("nppiCopy_8u_C1C3R", "8u_C1C3R", 17, 13, 1, 3, 16, &nppiCopy_8u_C1C3R_Ctx, 310);
  roi_conv("nppiCopy_16s_C1C3R", "16s_C1C3R", 64, 32, 1, 3, 32, &nppiCopy_16s_C1C3R_Ctx, 311);
  roi_conv("nppiCopy_32f_C1C3R", "32f_C1C3R", 1, 1, 1, 3, 16, &nppiCopy_32f_C1C3R_Ctx, 312);
}
TEST_F(NppiCopyDiff, Insert_C1C4) {
  roi_conv("nppiCopy_8u_C1C4R", "8u_C1C4R", 1, 1, 1, 4, 16, &nppiCopy_8u_C1C4R_Ctx, 320);
  roi_conv("nppiCopy_32f_C1C4R", "32f_C1C4R", 17, 13, 1, 4, 16, &nppiCopy_32f_C1C4R_Ctx, 321);
}
TEST_F(NppiCopyDiff, Extract_CxC1) {
  roi_conv("nppiCopy_32f_C2C1R", "32f_C2C1R", 17, 13, 2, 1, 16, &nppiCopy_32f_C2C1R_Ctx, 330);
  roi_conv("nppiCopy_8u_C3C1R", "8u_C3C1R", 17, 13, 3, 1, 16, &nppiCopy_8u_C3C1R_Ctx, 331);
  roi_conv("nppiCopy_16u_C4C1R", "16u_C4C1R", 64, 32, 4, 1, 32, &nppiCopy_16u_C4C1R_Ctx, 332);
  roi_conv("nppiCopy_32s_C4C1R", "32s_C4C1R", 1, 1, 4, 1, 16, &nppiCopy_32s_C4C1R_Ctx, 333);
}
TEST_F(NppiCopyDiff, SelectedChannel) {
  roi_stride("nppiCopy_8u_C3CR", "8u_C3CR", 17, 13, 3, 16, &nppiCopy_8u_C3CR_Ctx, 340);
  roi_stride("nppiCopy_32f_C4CR", "32f_C4CR", 64, 32, 4, 32, &nppiCopy_32f_C4CR_Ctx, 341);
}

// ===== Masked (MR) ===========================================================
TEST_F(NppiCopyDiff, Masked_C1) {
  roi_masked("nppiCopy_8u_C1MR", "8u_C1MR", 17, 13, 1, 16, &nppiCopy_8u_C1MR_Ctx, 400);
  roi_masked("nppiCopy_16s_C1MR", "16s_C1MR", 64, 32, 1, 32, &nppiCopy_16s_C1MR_Ctx, 401);
}
TEST_F(NppiCopyDiff, Masked_C3) {
  roi_masked("nppiCopy_8u_C3MR", "8u_C3MR", 64, 32, 3, 32, &nppiCopy_8u_C3MR_Ctx, 410);
}
TEST_F(NppiCopyDiff, Masked_C4) {
  roi_masked("nppiCopy_32s_C4MR", "32s_C4MR", 1, 1, 4, 16, &nppiCopy_32s_C4MR_Ctx, 420);
  roi_masked("nppiCopy_32s_C4MR", "32s_C4MR", 17, 13, 4, 16, &nppiCopy_32s_C4MR_Ctx, 421);
}
TEST_F(NppiCopyDiff, Masked_AC4MR) {
  roi_masked("nppiCopy_16u_AC4MR", "16u_AC4MR", 17, 13, 4, 16, &nppiCopy_16u_AC4MR_Ctx, 430);
  roi_masked("nppiCopy_32f_AC4MR", "32f_AC4MR", 64, 32, 4, 32, &nppiCopy_32f_AC4MR_Ctx, 431);
}




// ===== Contract (error paths; only the NppStatus name is compared) ==========
TEST_F(NppiCopyDiff, Contract_Null) {
  clear_cuda_error();
  NppStatus st = nppiCopy_8u_AC4R_Ctx(nullptr, 16, nullptr, 16, NppiSize{1, 1}, current_ctx());
  Case c("nppiCopy/8u_AC4R/null", "nppiCopy_8u_AC4R");
  c.status(st);
  c.finish();
}

TEST_F(NppiCopyDiff, Contract_NullMask) {
  DeviceMemory<Npp8u> &s = arena<Npp8u>(16);
  DeviceMemory<Npp8u> &d = arena<Npp8u>(16);
  s.copyFromHost(std::vector<Npp8u>(16, 1));
  d.copyFromHost(std::vector<Npp8u>(16, 2));
  clear_cuda_error();
  NppStatus st = nppiCopy_8u_C1MR_Ctx(s.get(), 16, d.get(), 16, NppiSize{4, 1}, nullptr, 4, current_ctx());
  Case c("nppiCopy/8u_C1MR/nullmask", "nppiCopy_8u_C1MR");
  c.status(st);
  c.finish();
}

TEST_F(NppiCopyDiff, Contract_EmptyROI) {
  DeviceMemory<Npp32f> &s = arena<Npp32f>(8);
  DeviceMemory<Npp32f> &d = arena<Npp32f>(8);
  s.copyFromHost(std::vector<Npp32f>(8, 1.f));
  d.copyFromHost(std::vector<Npp32f>(8, 2.f));
  clear_cuda_error();
  NppStatus st = nppiCopy_32f_AC4R_Ctx(s.get(), 16, d.get(), 16, NppiSize{0, 1}, current_ctx());
  Case c("nppiCopy/32f_AC4R/empty", "nppiCopy_32f_AC4R");
  c.status(st);
  c.finish();
}

TEST_F(NppiCopyDiff, Contract_NegativeSize) {
  DeviceMemory<Npp16f> &s = arena<Npp16f>(4);
  DeviceMemory<Npp16f> &d = arena<Npp16f>(4);
  s.copyFromHost(std::vector<Npp16f>(4, Npp16f{}));
  d.copyFromHost(std::vector<Npp16f>(4, Npp16f{}));
  clear_cuda_error();
  NppStatus st = nppiCopy_16f_C1R_Ctx(s.get(), 8, d.get(), 8, NppiSize{-1, 2}, current_ctx());
  Case c("nppiCopy/16f_C1R/neg", "nppiCopy_16f_C1R");
  c.status(st);
  c.finish();
}

TEST_F(NppiCopyDiff, Contract_ShortStep) {
  DeviceMemory<Npp8u> &s = arena<Npp8u>(8);
  DeviceMemory<Npp8u> &d = arena<Npp8u>(8);
  s.copyFromHost(std::vector<Npp8u>(8, 1));
  d.copyFromHost(std::vector<Npp8u>(8, 2));
  clear_cuda_error();
  // width=4, 4ch, 1B -> min step 16; give 8 -> reference step error.
  NppStatus st = nppiCopy_8u_AC4R_Ctx(s.get(), 8, d.get(), 8, NppiSize{4, 1}, current_ctx());
  Case c("nppiCopy/8u_AC4R/shortstep", "nppiCopy_8u_AC4R");
  c.status(st);
  c.finish();
}

// ===== Non-Ctx adapters (reproduce _Ctx matrix cases -> same golden) ========
#if NPP_DIFF_HAS_NONCTX
TEST_F(NppiCopyDiff, NonCtx_16f_C4) {
  // Reproduces roi_stride(...16f_C4R, 64,32,4,32, seed=120)
  const int width = 64, height = 32, stride = 4, pad = 32;
  const int es = (int)sizeof(Npp16f);
  const int sStep = ((width * stride * es + pad + es - 1) / es) * es;
  const int dStep = sStep;
  const size_t se = (size_t)height * sStep / es;
  std::vector<Npp16f> hs = make_uniform<Npp16f>(se, 120 + 11);
  std::vector<Npp16f> hd = make_uniform<Npp16f>(se, 120 + 29);
  DeviceMemory<Npp16f> &s = arena<Npp16f>(se);
  DeviceMemory<Npp16f> &d = arena<Npp16f>(se);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st = nppiCopy_16f_C4R(s.get(), sStep, d.get(), dStep, NppiSize{width, height});
  Case c("nppiCopy/16f_C4R64x32", "nppiCopy_16f_C4R");
  c.status(st);
  c.device_output<Npp16f>(d.get(), se, ValueKind::F16, 1e-5, 1e-6);
  c.finish();
}
TEST_F(NppiCopyDiff, NonCtx_8u_C3MR) {
  // Reproduces roi_masked 8u_C3MR 64,32,3,32 seed=410
  const int width = 64, height = 32, stride = 3, pad = 32;
  const int es = (int)sizeof(Npp8u);
  const int sStep = ((width * stride * es + pad + es - 1) / es) * es;
  const int mStep = width + 4;
  const size_t se = (size_t)height * sStep / es;
  const size_t me = (size_t)height * mStep;
  std::vector<Npp8u> hs = make_uniform<Npp8u>(se, 410 + 11);
  std::vector<Npp8u> hd = make_uniform<Npp8u>(se, 410 + 29);
  std::vector<Npp8u> hm = make_uniform<Npp8u>(me, 410 + 7);
  for (auto &v : hm) v = (Npp8u)(v & 3);
  DeviceMemory<Npp8u> &s = arena<Npp8u>(se);
  DeviceMemory<Npp8u> &d = arena<Npp8u>(se);
  DeviceMemory<Npp8u> &m = arena<Npp8u>(me);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  m.copyFromHost(hm);
  clear_cuda_error();
  NppStatus st = nppiCopy_8u_C3MR(s.get(), sStep, d.get(), sStep, NppiSize{width, height}, m.get(), mStep);
  Case c("nppiCopy/8u_C3MR64x32", "nppiCopy_8u_C3MR");
  c.status(st);
  c.device_output<Npp8u>(d.get(), se, ValueKind::U8, 1e-5, 1e-6);
  c.finish();
}
TEST_F(NppiCopyDiff, NonCtx_32f_C1C3) {
  // Reproduces roi_conv 32f_C1C3R 1,1,1,3,16 seed=312
  const int width = 1, height = 1, srcStride = 1, dstStride = 3, pad = 16;
  const int es = (int)sizeof(Npp32f);
  const int sStep = ((width * srcStride * es + pad + es - 1) / es) * es;
  const int dStep = ((width * dstStride * es + pad + es - 1) / es) * es;
  const size_t se = (size_t)height * sStep / es;
  const size_t de = (size_t)height * dStep / es;
  std::vector<Npp32f> hs = make_uniform<Npp32f>(se, 312 + 11);
  std::vector<Npp32f> hd = make_uniform<Npp32f>(de, 312 + 29);
  DeviceMemory<Npp32f> &s = arena<Npp32f>(se);
  DeviceMemory<Npp32f> &d = arena<Npp32f>(de);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st = nppiCopy_32f_C1C3R(s.get(), sStep, d.get(), dStep, NppiSize{width, height});
  Case c("nppiCopy/32f_C1C3R1x1", "nppiCopy_32f_C1C3R");
  c.status(st);
  c.device_output<Npp32f>(d.get(), de, ValueKind::F32, 1e-5, 1e-6);
  c.finish();
}
#endif
