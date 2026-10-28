// Differential tests for the nppiConvert family (p2.1a).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiConvertDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppiConvertDiff.*'
//
// 2D ROI semantics are exercised with padded row strides.  The whole output
// buffer (ROI + row-end padding) is captured and compared, which also proves
// that data outside oSizeROI is left untouched.
//
// NVIDIA NPP 13 only exports _Ctx entry points; the matrix uses those.  A
// NPP_DIFF_HAS_NONCTX-guarded block additionally checks the non-Ctx adapters.

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

using namespace npp_diff;
using namespace npp_functional_test;

namespace {

// Allocate a device buffer that stays alive for the process lifetime. The maca
// device-memory allocator can hand out pointers into regions it has just freed
// and then reject those pointers when a later kernel runs ("bad ptr"), leaving
// the destination untouched. Keeping every test buffer alive avoids that
// free/reuse path (test-only; all blocks are released at process exit).
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

template <typename D>
ValueKind out_kind() {
  if constexpr (std::is_same_v<D, Npp8s>) return ValueKind::I8;
  else if constexpr (std::is_same_v<D, Npp8u>) return ValueKind::U8;
  else if constexpr (std::is_same_v<D, Npp16s>) return ValueKind::I16;
  else if constexpr (std::is_same_v<D, Npp16u>) return ValueKind::U16;
  else if constexpr (std::is_same_v<D, Npp32s>) return ValueKind::I32;
  else if constexpr (std::is_same_v<D, Npp32u>) return ValueKind::U32;
  else if constexpr (std::is_same_v<D, Npp32f>) return ValueKind::F32;
  else if constexpr (std::is_same_v<D, Npp16f>) return ValueKind::F16;
  else return ValueKind::Bytes;
}

inline int align_step(int step, int esize) { return ((step + esize - 1) / esize) * esize; }

std::vector<Npp32f> boundary_floats() {
  return {-1e30f, -std::numeric_limits<Npp32f>::infinity(), -3.75f, -1.5f, -0.5f, -0.0f, 0.0f,
          0.5f,    1.5f, 3.75f, std::numeric_limits<Npp32f>::infinity(),
          std::numeric_limits<Npp32f>::quiet_NaN(), 255.6f, 299.0f, -1.0f};
}

// Fill a full element buffer byte-by-byte from source data then pad the tail of
// each row with a deterministic sentinel, so the reference and this implementation
// see identical out-of-ROI bytes.
template <typename A>
void fill_src(std::vector<A> &out, int height, int stepElems, int widthElems, const std::vector<A> &roi) {
  Rng rng(0xABCD + widthElems * 17 + height);
  for (auto &e : out) e = static_cast<A>(rng.next());
  for (int y = 0; y < height; ++y) {
    const size_t base = (size_t)y * stepElems;
    for (int x = 0; x < widthElems && (size_t)x < roi.size(); ++x) out[base + x] = roi[x];
  }
}

// ---- Plain convert (no round mode / no scale) --------------------------------
template <typename A, typename D>
void roi_plain(const char *fname, const std::string &idBase, int width, int height, int channels, int pad,
               NppStatus (*f)(const A *, int, D *, int, NppiSize, NppStreamContext), uint64_t seed) {
  const int se = (int)sizeof(A), de = (int)sizeof(D);
  const int srcStep = align_step(width * channels * se + pad, se);
  const int dstStep = align_step(width * channels * de + pad, de);
  const size_t srcElems = (size_t)height * srcStep / se;
  const size_t dstElems = (size_t)height * dstStep / de;
  std::vector<A> hs = make_uniform<A>(srcElems, seed + 11);
  std::vector<D> hd = make_uniform<D>(dstElems, seed + 29);
  DeviceMemory<A> &s = arena<A>(srcElems);
  DeviceMemory<D> &d = arena<D>(dstElems);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  NppiSize o{width, height};
  clear_cuda_error();
  NppStatus st = f(s.get(), srcStep, d.get(), dstStep, o, current_context());
  const std::string id = "nppiConvert/" + idBase + std::to_string(width) + "x" + std::to_string(height) + "c" +
                         std::to_string(channels);
  Case c(id, fname);
  c.input(hs.data(), hs.size() * sizeof(A));
  c.status(st);
  c.device_output<D>(d.get(), dstElems, out_kind<D>(), 1e-5, 1e-6);
  c.finish();
}

// ---- Round-only convert (eRoundMode, no scale) ------------------------------
template <typename A, typename D>
void roi_rnd(const char *fname, const std::string &idBase, int width, int height, int channels, NppRoundMode mode,
             NppStatus (*f)(const A *, int, D *, int, NppiSize, NppRoundMode, NppStreamContext), uint64_t seed) {
  const int se = (int)sizeof(A), de = (int)sizeof(D);
  const int srcStep = align_step(width * channels * se + 16, se);
  const int dstStep = align_step(width * channels * de + 16, de);
  const size_t srcElems = (size_t)height * srcStep / se;
  const size_t dstElems = (size_t)height * dstStep / de;
  std::vector<A> hs = make_uniform<A>(srcElems, seed + 13);
  std::vector<D> hd = make_uniform<D>(dstElems, seed + 31);
  DeviceMemory<A> &s = arena<A>(srcElems);
  DeviceMemory<D> &d = arena<D>(dstElems);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  NppiSize o{width, height};
  clear_cuda_error();
  NppStatus st = f(s.get(), srcStep, d.get(), dstStep, o, mode, current_context());
  Case c("nppiConvert/" + idBase + std::to_string(width) + "x" + std::to_string(height) + "c" +
             std::to_string(channels),
         fname);
  c.input(hs.data(), hs.size() * sizeof(A));
  c.status(st);
  c.device_output<D>(d.get(), dstElems, out_kind<D>(), 1e-5, 1e-6);
  c.finish();
}

// ---- Round + scale (eRoundMode + nScaleFactor), C1 integer dst ---------------
template <typename A, typename D>
void roi_rndsfs(const char *fname, const std::string &idBase, int width, int height, NppRoundMode mode, int sf,
                NppStatus (*f)(const A *, int, D *, int, NppiSize, NppRoundMode, int, NppStreamContext),
                uint64_t seed) {
  const int se = (int)sizeof(A), de = (int)sizeof(D);
  const int srcStep = width * se;
  const int dstStep = width * de;
  const size_t srcElems = (size_t)height * srcStep / se;
  const size_t dstElems = (size_t)height * dstStep / de;
  std::vector<A> hs = make_uniform<A>(srcElems, seed + 17);
  std::vector<D> hd = make_uniform<D>(dstElems, seed + 19);
  DeviceMemory<A> &s = arena<A>(srcElems);
  DeviceMemory<D> &d = arena<D>(dstElems);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  NppiSize o{width, height};
  clear_cuda_error();
  NppStatus st = f(s.get(), srcStep, d.get(), dstStep, o, mode, sf, current_context());
  Case c("nppiConvert/" + idBase + "c1", fname);
  c.input(hs.data(), hs.size() * sizeof(A));
  c.status(st);
  c.device_output<D>(d.get(), dstElems, out_kind<D>(), 1e-5, 1e-6);
  c.finish();
}

}  // namespace

class NppiConvertDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

// ===========================================================================
// Plain converts (int->int saturating, int/16f -> float)
// ===========================================================================
TEST_F(NppiConvertDiff, Plain_16s16u_C1Rs) {
  roi_plain("nppiConvert_16s16u_C1Rs", "16s16u", 64, 32, 1, 32, &nppiConvert_16s16u_C1Rs_Ctx, 101);
  roi_plain("nppiConvert_16s16u_C1Rs", "16s16u", 17, 1, 1, 16, &nppiConvert_16s16u_C1Rs_Ctx, 100);
}
TEST_F(NppiConvertDiff, Plain_16s32s_C3R) {
  roi_plain("nppiConvert_16s32s_C3R", "16s32s", 17, 13, 3, 16, &nppiConvert_16s32s_C3R_Ctx, 200);
}
TEST_F(NppiConvertDiff, Plain_16s32s_C4R) {
  roi_plain("nppiConvert_16s32s_C4R", "16s32s", 1, 1, 4, 16, &nppiConvert_16s32s_C4R_Ctx, 202);
}
TEST_F(NppiConvertDiff, Plain_32s8s_AC4R) {
  roi_plain("nppiConvert_32s8s_AC4R", "32s8s", 17, 13, 4, 16, &nppiConvert_32s8s_AC4R_Ctx, 300);
}
TEST_F(NppiConvertDiff, Plain_8s8u_C1Rs) {
  roi_plain("nppiConvert_8s8u_C1Rs", "8s8u", 17, 13, 1, 16, &nppiConvert_8s8u_C1Rs_Ctx, 400);
}
TEST_F(NppiConvertDiff, Plain_8u16s_C1R) {
  roi_plain("nppiConvert_8u16s_C1R", "8u16s", 17, 13, 1, 16, &nppiConvert_8u16s_C1R_Ctx, 500);
}
TEST_F(NppiConvertDiff, Plain_8u16s_C3R) {
  roi_plain("nppiConvert_8u16s_C3R", "8u16s", 64, 32, 3, 32, &nppiConvert_8u16s_C3R_Ctx, 501);
}
TEST_F(NppiConvertDiff, Plain_16s32f_C3R) {
  roi_plain("nppiConvert_16s32f_C3R", "16s32f", 17, 13, 3, 16, &nppiConvert_16s32f_C3R_Ctx, 600);
}
TEST_F(NppiConvertDiff, Plain_16u32f_C4R) {
  roi_plain("nppiConvert_16u32f_C4R", "16u32f", 17, 13, 4, 16, &nppiConvert_16u32f_C4R_Ctx, 700);
}
TEST_F(NppiConvertDiff, Plain_32s32f_C1R) {
  roi_plain("nppiConvert_32s32f_C1R", "32s32f", 1, 1, 1, 16, &nppiConvert_32s32f_C1R_Ctx, 800);
  roi_plain("nppiConvert_32s32f_C1R", "32s32f", 64, 32, 1, 32, &nppiConvert_32s32f_C1R_Ctx, 801);
}
TEST_F(NppiConvertDiff, Plain_8s32f_AC4R) {
  roi_plain("nppiConvert_8s32f_AC4R", "8s32f", 17, 13, 4, 16, &nppiConvert_8s32f_AC4R_Ctx, 900);
}
TEST_F(NppiConvertDiff, Plain_16f32f_C1R) {
  roi_plain("nppiConvert_16f32f_C1R", "16f32f", 17, 13, 1, 16, &nppiConvert_16f32f_C1R_Ctx, 1000);
}
TEST_F(NppiConvertDiff, Plain_16f32f_C3R) {
  roi_plain("nppiConvert_16f32f_C3R", "16f32f", 64, 32, 3, 32, &nppiConvert_16f32f_C3R_Ctx, 1001);
}
TEST_F(NppiConvertDiff, Plain_16f32f_C4R) {
  roi_plain("nppiConvert_16f32f_C4R", "16f32f", 1, 1, 4, 16, &nppiConvert_16f32f_C4R_Ctx, 1002);
}
TEST_F(NppiConvertDiff, Plain_16f32f_AC4R) {
  roi_plain("nppiConvert_16f32f_AC4R", "16f32f", 17, 13, 4, 16, &nppiConvert_16f32f_AC4R_Ctx, 1003);
}

// ===========================================================================
// Rounded converts (float -> int, float -> 16f)
// ===========================================================================
TEST_F(NppiConvertDiff, Rnd_32f16s_C1R) {
  roi_rnd("nppiConvert_32f16s_C1R", "32f16s", 17, 13, 1, NPP_RND_NEAR, &nppiConvert_32f16s_C1R_Ctx, 1100);
}
TEST_F(NppiConvertDiff, Rnd_32f16s_C3R) {
  roi_rnd("nppiConvert_32f16s_C3R", "32f16s", 17, 13, 3, NPP_RND_NEAR, &nppiConvert_32f16s_C3R_Ctx, 1101);
}
TEST_F(NppiConvertDiff, Rnd_32f16u_C4R) {
  roi_rnd("nppiConvert_32f16u_C4R", "32f16u", 17, 13, 4, NPP_RND_FINANCIAL, &nppiConvert_32f16u_C4R_Ctx, 1200);
}
TEST_F(NppiConvertDiff, Rnd_32f8s_C4R) {
  roi_rnd("nppiConvert_32f8s_C4R", "32f8s", 17, 13, 4, NPP_RND_ZERO, &nppiConvert_32f8s_C4R_Ctx, 1300);
}
TEST_F(NppiConvertDiff, Rnd_32f16f_C1) {
  roi_rnd("nppiConvert_32f16f_C1R", "32f16f", 17, 13, 1, NPP_RND_NEAR, &nppiConvert_32f16f_C1R_Ctx, 1400);
}


// ===========================================================================
// Boundary values + round modes (C1, crafted inputs incl. NaN / +/-Inf)
// ===========================================================================
namespace {
void boundary_32f8s(const std::string &idBase, NppRoundMode mode) {
  std::vector<Npp32f> src = boundary_floats();
  const int width = (int)src.size();
  DeviceMemory<Npp32f> &s = arena<Npp32f>((size_t)width);
  DeviceMemory<Npp8s> &d = arena<Npp8s>((size_t)width);
  std::vector<Npp8s> hd((size_t)width, (Npp8s)0x5A);
  s.copyFromHost(src);
  d.copyFromHost(hd);
  NppiSize o{width, 1};
  clear_cuda_error();
  NppStatus st = nppiConvert_32f8s_C1R_Ctx(s.get(), width * (int)sizeof(Npp32f), d.get(), width * (int)sizeof(Npp8s),
                                           o, mode, current_context());
  Case c("nppiConvert/" + idBase, "nppiConvert_32f8s_C1R");
  c.input(src.data(), src.size() * sizeof(Npp32f));
  c.status(st);
  c.device_output<Npp8s>(d.get(), (size_t)width, ValueKind::I8, 1e-5, 1e-6);
  c.finish();
}
void boundary_32f16u(const std::string &idBase, NppRoundMode mode) {
  std::vector<Npp32f> src = boundary_floats();
  const int width = (int)src.size();
  DeviceMemory<Npp32f> &s = arena<Npp32f>((size_t)width);
  DeviceMemory<Npp16u> &d = arena<Npp16u>((size_t)width);
  std::vector<Npp16u> hd((size_t)width, (Npp16u)0x5A);
  s.copyFromHost(src);
  d.copyFromHost(hd);
  NppiSize o{width, 1};
  clear_cuda_error();
  NppStatus st = nppiConvert_32f16u_C1R_Ctx(s.get(), width * (int)sizeof(Npp32f), d.get(), width * (int)sizeof(Npp16u),
                                            o, mode, current_context());
  Case c("nppiConvert/" + idBase, "nppiConvert_32f16u_C1R");
  c.input(src.data(), src.size() * sizeof(Npp32f));
  c.status(st);
  c.device_output<Npp16u>(d.get(), (size_t)width, ValueKind::U16, 1e-5, 1e-6);
  c.finish();
}
}  // namespace
TEST_F(NppiConvertDiff, Rnd32f8s_Near) { boundary_32f8s("32f8s_boundary_near", NPP_RND_NEAR); }
TEST_F(NppiConvertDiff, Rnd32f8s_Fin)  { boundary_32f8s("32f8s_boundary_fin", NPP_RND_FINANCIAL); }
TEST_F(NppiConvertDiff, Rnd32f8s_Zero) { boundary_32f8s("32f8s_boundary_zero", NPP_RND_ZERO); }
TEST_F(NppiConvertDiff, Rnd32f16u_Fin) { boundary_32f16u("32f16u_boundary_fin", NPP_RND_FINANCIAL); }

// ===========================================================================
// Round + scale (C1, integer dst)
// ===========================================================================
TEST_F(NppiConvertDiff, Sfs_32f32u) { roi_rndsfs("nppiConvert_32f32u_C1RSfs", "32f32u", 17, 1, NPP_RND_NEAR, 8,
                                                 &nppiConvert_32f32u_C1RSfs_Ctx, 1500); }
TEST_F(NppiConvertDiff, Sfs_32s16s) { roi_rndsfs("nppiConvert_32s16s_C1RSfs", "32s16s", 17, 13, NPP_RND_NEAR, 4,
                                                 &nppiConvert_32s16s_C1RSfs_Ctx, 1600); }
TEST_F(NppiConvertDiff, Sfs_16u16s) { roi_rndsfs("nppiConvert_16u16s_C1RSfs", "16u16s", 17, 13, NPP_RND_FINANCIAL, 1,
                                                 &nppiConvert_16u16s_C1RSfs_Ctx, 1700); }
TEST_F(NppiConvertDiff, Sfs_16s8s) { roi_rndsfs("nppiConvert_16s8s_C1RSfs", "16s8s", 17, 13, NPP_RND_NEAR, 4,
                                                 &nppiConvert_16s8s_C1RSfs_Ctx, 1800); }
TEST_F(NppiConvertDiff, Sfs_32u8s) { roi_rndsfs("nppiConvert_32u8s_C1RSfs", "32u8s", 17, 13, NPP_RND_FINANCIAL, 8,
                                                 &nppiConvert_32u8s_C1RSfs_Ctx, 1900); }
TEST_F(NppiConvertDiff, Sfs_8u8s) { roi_rndsfs("nppiConvert_8u8s_C1RSfs", "8u8s", 17, 13, NPP_RND_NEAR, 1,
                                               &nppiConvert_8u8s_C1RSfs_Ctx, 2000); }

// ===========================================================================
// Contract (error paths)
// ===========================================================================
TEST_F(NppiConvertDiff, Contract_Null) {
  clear_cuda_error();
  NppStatus st = nppiConvert_16s32s_C1R_Ctx(nullptr, 4, nullptr, 4, NppiSize{1, 1}, current_context());
  Case c("nppiConvert/16s32s/null", "nppiConvert_16s32s_C1R");
  c.status(st);
  c.finish();
}
TEST_F(NppiConvertDiff, Contract_Empty) {
  Npp16s s1 = 3;
  DeviceMemory<Npp16s> &s = arena<Npp16s>(1);
  DeviceMemory<Npp32s> &d = arena<Npp32s>(1);
  s.copyFromHost({s1});
  d.copyFromHost({(Npp32s)7});
  clear_cuda_error();
  NppStatus st = nppiConvert_16s32s_C1R_Ctx(s.get(), 2, d.get(), 4, NppiSize{0, 1}, current_context());
  Case c("nppiConvert/16s32s/empty", "nppiConvert_16s32s_C1R");
  c.status(st);
  c.finish();
}
TEST_F(NppiConvertDiff, Contract_NegativeSize) {
  DeviceMemory<Npp16s> &s = arena<Npp16s>(1);
  DeviceMemory<Npp32s> &d = arena<Npp32s>(1);
  s.copyFromHost({(Npp16s)1});
  d.copyFromHost({(Npp32s)1});
  clear_cuda_error();
  NppStatus st = nppiConvert_16s32s_C1R_Ctx(s.get(), 2, d.get(), 4, NppiSize{-1, 2}, current_context());
  Case c("nppiConvert/16s32s/neg", "nppiConvert_16s32s_C1R");
  c.status(st);
  c.finish();
}
TEST_F(NppiConvertDiff, Contract_ShortStep) {
  DeviceMemory<Npp8u> &s = arena<Npp8u>(8);
  DeviceMemory<Npp16u> &d = arena<Npp16u>(8);
  std::vector<Npp8u> vs(8, 1);
  std::vector<Npp16u> vd(8, 0);
  s.copyFromHost(vs);
  d.copyFromHost(vd);
  clear_cuda_error();
  // Need 17*1*1 = 17 bytes for src step; use 8 -> NPP_STRIDE_ERROR.
  NppStatus st = nppiConvert_8u16u_C4R_Ctx(s.get(), 8, d.get(), 8, NppiSize{17, 1}, current_context());
  Case c("nppiConvert/8u16u/shortstep", "nppiConvert_8u16u_C4R");
  c.status(st);
  c.finish();
}

// ===========================================================================
// Non-Ctx adapters share the same golden (MPP build only)
// ===========================================================================
#if NPP_DIFF_HAS_NONCTX
// These reproduce the exact host inputs of the corresponding _Ctx matrix cases,
// so they share the same recorded golden and simply exercise the non-Ctx
// adapter's stream acquisition + forwarding path.
TEST_F(NppiConvertDiff, NonCtx_16s32s) {
  const int width = 17, height = 13, channels = 3;
  const int se = sizeof(Npp16s), de = sizeof(Npp32s);
  const int srcStep = ((width * channels * se + 16 + se - 1) / se) * se;
  const int dstStep = ((width * channels * de + 16 + de - 1) / de) * de;
  const size_t srcElems = (size_t)height * srcStep / se;
  const size_t dstElems = (size_t)height * dstStep / de;
  std::vector<Npp16s> hs = make_uniform<Npp16s>(srcElems, 200 + 11);
  std::vector<Npp32s> hd = make_uniform<Npp32s>(dstElems, 200 + 29);
  DeviceMemory<Npp16s> &s = arena<Npp16s>(srcElems);
  DeviceMemory<Npp32s> &d = arena<Npp32s>(dstElems);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st = nppiConvert_16s32s_C3R(s.get(), srcStep, d.get(), dstStep, NppiSize{width, height});
  Case c("nppiConvert/16s32s17x13c3", "nppiConvert_16s32s_C3R");
  c.status(st);
  c.device_output<Npp32s>(d.get(), dstElems, ValueKind::I32, 1e-5, 1e-6);
  c.finish();
}
TEST_F(NppiConvertDiff, NonCtx_16f32f) {
  const int width = 17, height = 13, channels = 1;
  const int se = sizeof(Npp16f), de = sizeof(Npp32f);
  const int srcStep = ((width * channels * se + 16 + se - 1) / se) * se;
  const int dstStep = ((width * channels * de + 16 + de - 1) / de) * de;
  const size_t srcElems = (size_t)height * srcStep / se;
  const size_t dstElems = (size_t)height * dstStep / de;
  std::vector<Npp16f> hs = make_uniform<Npp16f>(srcElems, 1000 + 11);
  std::vector<Npp32f> hd = make_uniform<Npp32f>(dstElems, 1000 + 29);
  DeviceMemory<Npp16f> &s = arena<Npp16f>(srcElems);
  DeviceMemory<Npp32f> &d = arena<Npp32f>(dstElems);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st = nppiConvert_16f32f_C1R(s.get(), srcStep, d.get(), dstStep, NppiSize{width, height});
  Case c("nppiConvert/16f32f17x13c1", "nppiConvert_16f32f_C1R");
  c.status(st);
  c.device_output<Npp32f>(d.get(), dstElems, ValueKind::F32, 1e-5, 1e-6);
  c.finish();
}
#endif

