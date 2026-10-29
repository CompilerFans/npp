// Differential tests for nppiMinIndx / nppiMaxIndx / nppiMinMaxIndx /
// nppiMinEvery_AC4 / nppiMaxEvery_AC4 (package p3.1a3).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiIndexMinMaxDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppiIndexMinMaxDiff.*'
//
// Index reductions write a value (dtype) plus Npp32s x/y position arrays on
// device; MinMaxIndx writes scalar min/max plus NppiPoint positions.  The
// indexed (position) outputs are captured with ValueKind::I32.  NVIDIA NPP 13
// only exports the _Ctx entry points so the reference matrix uses those.  As
// in P3.1a2, invalid-ROI / error inputs are not probed for the index
// reductions (NPP 13 leaves deferred device faults); contract coverage uses
// minimal valid 1x1 round-trips plus buffer-size status probes.

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

inline Npp8u *scratch() { return arena<Npp8u>(1 << 20).get(); }

template <typename D> ValueKind indx_kind();
template <> ValueKind indx_kind<Npp8u>() { return ValueKind::U8; }
template <> ValueKind indx_kind<Npp16u>() { return ValueKind::U16; }
template <> ValueKind indx_kind<Npp16s>() { return ValueKind::I16; }
template <> ValueKind indx_kind<Npp8s>() { return ValueKind::I8; }
template <> ValueKind indx_kind<Npp32f>() { return ValueKind::F32; }

// ---- MinIndx / MaxIndx helpers ----
template <typename D, int NCh, int OutCount>
void extr_indx(const char *fname, const std::string &idBase, int width, int height, int pad, uint64_t seed,
               NppStatus (*f)(const D *, int, NppiSize, Npp8u *, D *, int *, int *, NppStreamContext)) {
  const int se = (int)sizeof(D);
  const int step = align_step(width * NCh * se + pad, se);
  const size_t stepElems = (size_t)step / se;
  const size_t elems = (size_t)height * stepElems;
  std::vector<D> hs = make_uniform<D>(elems, seed);
  DeviceMemory<D> &s = arena<D>(elems);
  DeviceMemory<D> &val = arena<D>(OutCount);
  DeviceMemory<int> &ix = arena<int>(OutCount);
  DeviceMemory<int> &iy = arena<int>(OutCount);
  s.copyFromHost(hs);
  Case c(idBase + "_" + std::to_string(width) + "x" + std::to_string(height), fname);
  clear_cuda_error();
  NppStatus st = f(s.get(), step, NppiSize{width, height}, scratch(), val.get(), ix.get(), iy.get(), current_context());
  c.input(hs.data(), hs.size() * sizeof(D));
  c.status(st);
  c.device_output(val.get(), OutCount, indx_kind<D>(), 0.0, 0.0);
  c.device_output(ix.get(), OutCount, ValueKind::I32, 0.0, 0.0);
  c.device_output(iy.get(), OutCount, ValueKind::I32, 0.0, 0.0);
  c.finish();
}

// Tie-break: single-channel source provided by the caller.
template <typename D>
void extr_indx_built(const char *fname, const std::string &idBase, const std::vector<D> &hs, int width, int height,
                     NppStatus (*f)(const D *, int, NppiSize, Npp8u *, D *, int *, int *, NppStreamContext)) {
  DeviceMemory<D> &s = arena<D>(hs.size());
  DeviceMemory<D> &val = arena<D>(1);
  DeviceMemory<int> &ix = arena<int>(1);
  DeviceMemory<int> &iy = arena<int>(1);
  s.copyFromHost(hs);
  Case c(idBase + "_tie", fname);
  clear_cuda_error();
  NppStatus st = f(s.get(), (int)width * (int)sizeof(D), NppiSize{width, height}, scratch(), val.get(), ix.get(),
                   iy.get(), current_context());
  c.input(hs.data(), hs.size() * sizeof(D));
  c.status(st);
  c.device_output(val.get(), 1, indx_kind<D>(), 0.0, 0.0);
  c.device_output(ix.get(), 1, ValueKind::I32, 0.0, 0.0);
  c.device_output(iy.get(), 1, ValueKind::I32, 0.0, 0.0);
  c.finish();
}

// ---- MinMaxIndx shape helpers ----
template <typename D>
void mm_c1(const char *fname, const std::string &idBase, int width, int height, int pad, uint64_t seed,
           NppStatus (*f)(const D *, int, NppiSize, D *, D *, NppiPoint *, NppiPoint *, Npp8u *, NppStreamContext)) {
  const int se = (int)sizeof(D);
  const int step = align_step(width * 1 * se + pad, se);
  const size_t stepElems = (size_t)step / se;
  const size_t elems = (size_t)height * stepElems;
  std::vector<D> hs = make_uniform<D>(elems, seed);
  DeviceMemory<D> &ds = arena<D>(elems);
  DeviceMemory<D> &mn = arena<D>(1);
  DeviceMemory<D> &mx = arena<D>(1);
  DeviceMemory<int> &mi = arena<int>(2);
  DeviceMemory<int> &xi = arena<int>(2);
  ds.copyFromHost(hs);
  Case c(idBase + "_" + std::to_string(width) + "x" + std::to_string(height), fname);
  clear_cuda_error();
  NppStatus st = f(ds.get(), step, NppiSize{width, height}, mn.get(), mx.get(), (NppiPoint *)mi.get(),
                   (NppiPoint *)xi.get(), scratch(), current_context());
  c.input(hs.data(), hs.size() * sizeof(D));
  c.status(st);
  c.device_output(mn.get(), 1, indx_kind<D>(), 0.0, 0.0);
  c.device_output(mx.get(), 1, indx_kind<D>(), 0.0, 0.0);
  c.device_output(mi.get(), 2, ValueKind::I32, 0.0, 0.0);
  c.device_output(xi.get(), 2, ValueKind::I32, 0.0, 0.0);
  c.finish();
}

template <typename D>
void mm_c1m(const char *fname, const std::string &idBase, int width, int height, int pad, uint64_t seed,
            NppStatus (*f)(const D *, int, const Npp8u *, int, NppiSize, D *, D *, NppiPoint *, NppiPoint *, Npp8u *,
                           NppStreamContext)) {
  const int se = (int)sizeof(D);
  const int step = align_step(width * 1 * se + pad, se);
  const size_t stepElems = (size_t)step / se;
  const size_t elems = (size_t)height * stepElems;
  std::vector<D> hs = make_uniform<D>(elems, seed);
  std::vector<Npp8u> mask((size_t)width * height);
  for (int i = 0; i < (int)mask.size(); ++i) mask[i] = (i % 3 == 0) ? 0 : 255;
  DeviceMemory<D> &ds = arena<D>(elems);
  DeviceMemory<Npp8u> &dm = arena<Npp8u>(mask.size());
  DeviceMemory<D> &mn = arena<D>(1);
  DeviceMemory<D> &mx = arena<D>(1);
  DeviceMemory<int> &mi = arena<int>(2);
  DeviceMemory<int> &xi = arena<int>(2);
  ds.copyFromHost(hs);
  dm.copyFromHost(mask);
  Case c(idBase + "_" + std::to_string(width) + "x" + std::to_string(height), fname);
  clear_cuda_error();
  NppStatus st = f(ds.get(), step, dm.get(), width, NppiSize{width, height}, mn.get(), mx.get(), (NppiPoint *)mi.get(),
                   (NppiPoint *)xi.get(), scratch(), current_context());
  c.input(hs.data(), hs.size() * sizeof(D));
  c.status(st);
  c.device_output(mn.get(), 1, indx_kind<D>(), 0.0, 0.0);
  c.device_output(mx.get(), 1, indx_kind<D>(), 0.0, 0.0);
  c.device_output(mi.get(), 2, ValueKind::I32, 0.0, 0.0);
  c.device_output(xi.get(), 2, ValueKind::I32, 0.0, 0.0);
  c.finish();
}

template <typename D>
void mm_c3c(const char *fname, const std::string &idBase, int width, int height, int pad, uint64_t seed, int nCOI,
            NppStatus (*f)(const D *, int, NppiSize, int, D *, D *, NppiPoint *, NppiPoint *, Npp8u *,
                           NppStreamContext)) {
  const int se = (int)sizeof(D);
  const int step = align_step(width * 3 * se + pad, se);
  const size_t stepElems = (size_t)step / se;
  const size_t elems = (size_t)height * stepElems;
  std::vector<D> hs = make_uniform<D>(elems, seed);
  DeviceMemory<D> &ds = arena<D>(elems);
  DeviceMemory<D> &mn = arena<D>(1);
  DeviceMemory<D> &mx = arena<D>(1);
  DeviceMemory<int> &mi = arena<int>(2);
  DeviceMemory<int> &xi = arena<int>(2);
  ds.copyFromHost(hs);
  Case c(idBase + "_" + std::to_string(width) + "x" + std::to_string(height), fname);
  clear_cuda_error();
  NppStatus st = f(ds.get(), step, NppiSize{width, height}, nCOI, mn.get(), mx.get(), (NppiPoint *)mi.get(),
                   (NppiPoint *)xi.get(), scratch(), current_context());
  c.input(hs.data(), hs.size() * sizeof(D));
  c.status(st);
  c.device_output(mn.get(), 1, indx_kind<D>(), 0.0, 0.0);
  c.device_output(mx.get(), 1, indx_kind<D>(), 0.0, 0.0);
  c.device_output(mi.get(), 2, ValueKind::I32, 0.0, 0.0);
  c.device_output(xi.get(), 2, ValueKind::I32, 0.0, 0.0);
  c.finish();
}

template <typename D>
void mm_c3cm(const char *fname, const std::string &idBase, int width, int height, int pad, uint64_t seed, int nCOI,
             NppStatus (*f)(const D *, int, const Npp8u *, int, NppiSize, int, D *, D *, NppiPoint *, NppiPoint *,
                            Npp8u *, NppStreamContext)) {
  const int se = (int)sizeof(D);
  const int step = align_step(width * 3 * se + pad, se);
  const size_t stepElems = (size_t)step / se;
  const size_t elems = (size_t)height * stepElems;
  std::vector<D> hs = make_uniform<D>(elems, seed);
  std::vector<Npp8u> mask((size_t)width * height);
  for (int i = 0; i < (int)mask.size(); ++i) mask[i] = (i % 4 == 0) ? 0 : 255;
  DeviceMemory<D> &ds = arena<D>(elems);
  DeviceMemory<Npp8u> &dm = arena<Npp8u>(mask.size());
  DeviceMemory<D> &mn = arena<D>(1);
  DeviceMemory<D> &mx = arena<D>(1);
  DeviceMemory<int> &mi = arena<int>(2);
  DeviceMemory<int> &xi = arena<int>(2);
  ds.copyFromHost(hs);
  dm.copyFromHost(mask);
  Case c(idBase + "_" + std::to_string(width) + "x" + std::to_string(height), fname);
  clear_cuda_error();
  NppStatus st = f(ds.get(), step, dm.get(), width, NppiSize{width, height}, nCOI, mn.get(), mx.get(),
                   (NppiPoint *)mi.get(), (NppiPoint *)xi.get(), scratch(), current_context());
  c.input(hs.data(), hs.size() * sizeof(D));
  c.input(mask.data(), mask.size() * sizeof(Npp8u));
  c.status(st);
  c.device_output(mn.get(), 1, indx_kind<D>(), 0.0, 0.0);
  c.device_output(mx.get(), 1, indx_kind<D>(), 0.0, 0.0);
  c.device_output(mi.get(), 2, ValueKind::I32, 0.0, 0.0);
  c.device_output(xi.get(), 2, ValueKind::I32, 0.0, 0.0);
  c.finish();
}

// ---- MinEvery_AC4 / MaxEvery_AC4: elementwise in-place min/max merge ----
template <typename D, int Channels>
void every_ac4(const char *fname, const std::string &idBase, int width, int height, int pad, uint64_t seed,
               NppStatus (*f)(const D *, int, D *, int, NppiSize, NppStreamContext)) {
  const int se = (int)sizeof(D);
  const int step = align_step(width * Channels * se + pad, se);
  const size_t stepElems = (size_t)step / se;
  const size_t elems = (size_t)height * stepElems;
  std::vector<D> src = make_uniform<D>(elems, seed);
  std::vector<D> dst = make_uniform<D>(elems, seed ^ 0xdeadbeefULL);
  DeviceMemory<D> &ds_s = arena<D>(elems);
  DeviceMemory<D> &ds_d = arena<D>(elems);
  ds_s.copyFromHost(src);
  ds_d.copyFromHost(dst);
  Case c(idBase + "_" + std::to_string(width) + "x" + std::to_string(height), fname);
  clear_cuda_error();
  NppStatus st = f(ds_s.get(), step, ds_d.get(), step, NppiSize{width, height}, current_context());
  c.input(src.data(), src.size() * sizeof(D));
  c.input(dst.data(), dst.size() * sizeof(D));
  c.status(st);
  c.device_output(ds_d.get(), (size_t)height * stepElems, indx_kind<D>(), 0.0, 0.0);
  c.finish();
}

}  // namespace

class NppiIndexMinMaxDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

// ============================ nppiMinIndx ====================================
TEST_F(NppiIndexMinMaxDiff, MinIndx_8u) {
  extr_indx<Npp8u, 1, 1>("nppiMinIndx_8u_C1R_Ctx", "MinIndx/8u/C1R", 17, 13, 16, 0x3c1d5a2bULL, &nppiMinIndx_8u_C1R_Ctx);
  extr_indx<Npp8u, 3, 3>("nppiMinIndx_8u_C3R_Ctx", "MinIndx/8u/C3R", 64, 32, 16, 0x71c2e094ULL, &nppiMinIndx_8u_C3R_Ctx);
  extr_indx<Npp8u, 4, 4>("nppiMinIndx_8u_C4R_Ctx", "MinIndx/8u/C4R", 17, 13, 16, 0x821f4cd6ULL, &nppiMinIndx_8u_C4R_Ctx);
  extr_indx<Npp8u, 4, 3>("nppiMinIndx_8u_AC4R_Ctx", "MinIndx/8u/AC4R", 64, 32, 16, 0x5a30b97ULL, &nppiMinIndx_8u_AC4R_Ctx);
  extr_indx<Npp8u, 1, 1>("nppiMinIndx_8u_C1R_Ctx", "MinIndx/8u/C1R_1x1", 1, 1, 0, 0x7d1e65ULL, &nppiMinIndx_8u_C1R_Ctx);
  std::vector<Npp8u> tie_min(6 * 4, 200);
  tie_min[0 * 6 + 2] = 7;  // first minimum
  tie_min[3 * 6 + 3] = 7;  // later minimum
  extr_indx_built<Npp8u>("nppiMinIndx_8u_C1R_Ctx", "MinIndx/8u/C1R/tie", tie_min, 6, 4, &nppiMinIndx_8u_C1R_Ctx);
}
TEST_F(NppiIndexMinMaxDiff, MinIndx_16u) {
  extr_indx<Npp16u, 1, 1>("nppiMinIndx_16u_C1R_Ctx", "MinIndx/16u/C1R", 64, 32, 16, 0x40a11d2cULL, &nppiMinIndx_16u_C1R_Ctx);
  extr_indx<Npp16u, 3, 3>("nppiMinIndx_16u_C3R_Ctx", "MinIndx/16u/C3R", 17, 13, 16, 0xc0ffee12ULL, &nppiMinIndx_16u_C3R_Ctx);
  extr_indx<Npp16u, 4, 4>("nppiMinIndx_16u_C4R_Ctx", "MinIndx/16u/C4R", 17, 13, 16, 0x71b2e343ULL, &nppiMinIndx_16u_C4R_Ctx);
  extr_indx<Npp16u, 4, 3>("nppiMinIndx_16u_AC4R_Ctx", "MinIndx/16u/AC4R", 64, 32, 16, 0xe5f08911ULL, &nppiMinIndx_16u_AC4R_Ctx);
}
TEST_F(NppiIndexMinMaxDiff, MinIndx_16s) {
  extr_indx<Npp16s, 1, 1>("nppiMinIndx_16s_C1R_Ctx", "MinIndx/16s/C1R", 17, 13, 16, 0x5c8a6002ULL, &nppiMinIndx_16s_C1R_Ctx);
  extr_indx<Npp16s, 3, 3>("nppiMinIndx_16s_C3R_Ctx", "MinIndx/16s/C3R", 64, 32, 16, 0x7332fcc4ULL, &nppiMinIndx_16s_C3R_Ctx);
  extr_indx<Npp16s, 4, 4>("nppiMinIndx_16s_C4R_Ctx", "MinIndx/16s/C4R", 17, 13, 16, 0x9fe88455ULL, &nppiMinIndx_16s_C4R_Ctx);
  extr_indx<Npp16s, 4, 3>("nppiMinIndx_16s_AC4R_Ctx", "MinIndx/16s/AC4R", 1, 1, 0, 0x2baa90e1ULL, &nppiMinIndx_16s_AC4R_Ctx);
}
TEST_F(NppiIndexMinMaxDiff, MinIndx_32f) {
  extr_indx<Npp32f, 1, 1>("nppiMinIndx_32f_C1R_Ctx", "MinIndx/32f/C1R", 64, 32, 16, 0x77bd9b0cULL, &nppiMinIndx_32f_C1R_Ctx);
  extr_indx<Npp32f, 3, 3>("nppiMinIndx_32f_C3R_Ctx", "MinIndx/32f/C3R", 17, 13, 16, 0x8e4a3a2dULL, &nppiMinIndx_32f_C3R_Ctx);
  extr_indx<Npp32f, 4, 4>("nppiMinIndx_32f_C4R_Ctx", "MinIndx/32f/C4R", 64, 32, 16, 0xbb49ceb8ULL, &nppiMinIndx_32f_C4R_Ctx);
  extr_indx<Npp32f, 4, 3>("nppiMinIndx_32f_AC4R_Ctx", "MinIndx/32f/AC4R", 17, 13, 16, 0x30d875a1ULL, &nppiMinIndx_32f_AC4R_Ctx);
  std::vector<Npp32f> tie(6, 0.0f);
  tie[1] = -3.0f; tie[4] = -3.0f;  // equal minima, first at x=1
  extr_indx_built<Npp32f>("nppiMinIndx_32f_C1R_Ctx", "MinIndx/32f/C1R/tie", tie, 6, 1, &nppiMinIndx_32f_C1R_Ctx);
}

// ============================ nppiMaxIndx ======================================
TEST_F(NppiIndexMinMaxDiff, MaxIndx_8u) {
  extr_indx<Npp8u, 1, 1>("nppiMaxIndx_8u_C1R_Ctx", "MaxIndx/8u/C1R", 17, 13, 16, 0x5f2a9180ULL, &nppiMaxIndx_8u_C1R_Ctx);
  extr_indx<Npp8u, 3, 3>("nppiMaxIndx_8u_C3R_Ctx", "MaxIndx/8u/C3R", 64, 32, 16, 0x1b2c3d4eULL, &nppiMaxIndx_8u_C3R_Ctx);
  extr_indx<Npp8u, 4, 4>("nppiMaxIndx_8u_C4R_Ctx", "MaxIndx/8u/C4R", 17, 13, 16, 0x2c3d4e5fULL, &nppiMaxIndx_8u_C4R_Ctx);
  extr_indx<Npp8u, 4, 3>("nppiMaxIndx_8u_AC4R_Ctx", "MaxIndx/8u/AC4R", 64, 32, 16, 0x9e44ee66ULL, &nppiMaxIndx_8u_AC4R_Ctx);
  std::vector<Npp8u> tie_max(6 * 4, 1);
  tie_max[0 * 6 + 5] = 250;  // first maximum
  tie_max[3 * 6 + 0] = 250;  // later maximum
  extr_indx_built<Npp8u>("nppiMaxIndx_8u_C1R_Ctx", "MaxIndx/8u/C1R/tie", tie_max, 6, 4, &nppiMaxIndx_8u_C1R_Ctx);
}
TEST_F(NppiIndexMinMaxDiff, MaxIndx_16u) {
  extr_indx<Npp16u, 1, 1>("nppiMaxIndx_16u_C1R_Ctx", "MaxIndx/16u/C1R", 64, 32, 16, 0x7721f0aaULL, &nppiMaxIndx_16u_C1R_Ctx);
  extr_indx<Npp16u, 3, 3>("nppiMaxIndx_16u_C3R_Ctx", "MaxIndx/16u/C3R", 17, 13, 16, 0x8100cc2eULL, &nppiMaxIndx_16u_C3R_Ctx);
  extr_indx<Npp16u, 4, 4>("nppiMaxIndx_16u_C4R_Ctx", "MaxIndx/16u/C4R", 17, 13, 16, 0x0badf00dULL, &nppiMaxIndx_16u_C4R_Ctx);
  extr_indx<Npp16u, 4, 3>("nppiMaxIndx_16u_AC4R_Ctx", "MaxIndx/16u/AC4R", 64, 32, 16, 0xdef10badULL, &nppiMaxIndx_16u_AC4R_Ctx);
}
TEST_F(NppiIndexMinMaxDiff, MaxIndx_16s) {
  extr_indx<Npp16s, 1, 1>("nppiMaxIndx_16s_C1R_Ctx", "MaxIndx/16s/C1R", 17, 13, 16, 0x61a3e9c7ULL, &nppiMaxIndx_16s_C1R_Ctx);
  extr_indx<Npp16s, 3, 3>("nppiMaxIndx_16s_C3R_Ctx", "MaxIndx/16s/C3R", 64, 32, 16, 0xcb3d5f1aULL, &nppiMaxIndx_16s_C3R_Ctx);
  extr_indx<Npp16s, 4, 4>("nppiMaxIndx_16s_C4R_Ctx", "MaxIndx/16s/C4R", 17, 13, 16, 0xe70a9c2dULL, &nppiMaxIndx_16s_C4R_Ctx);
  extr_indx<Npp16s, 4, 3>("nppiMaxIndx_16s_AC4R_Ctx", "MaxIndx/16s/AC4R", 1, 1, 0, 0x5b0c1a2eULL, &nppiMaxIndx_16s_AC4R_Ctx);
}
TEST_F(NppiIndexMinMaxDiff, MaxIndx_32f) {
  extr_indx<Npp32f, 1, 1>("nppiMaxIndx_32f_C1R_Ctx", "MaxIndx/32f/C1R", 64, 32, 16, 0x42c1f5a6ULL, &nppiMaxIndx_32f_C1R_Ctx);
  extr_indx<Npp32f, 3, 3>("nppiMaxIndx_32f_C3R_Ctx", "MaxIndx/32f/C3R", 17, 13, 16, 0x90d4e7f2ULL, &nppiMaxIndx_32f_C3R_Ctx);
  extr_indx<Npp32f, 4, 4>("nppiMaxIndx_32f_C4R_Ctx", "MaxIndx/32f/C4R", 64, 32, 16, 0x11e2a3b4ULL, &nppiMaxIndx_32f_C4R_Ctx);
  extr_indx<Npp32f, 4, 3>("nppiMaxIndx_32f_AC4R_Ctx", "MaxIndx/32f/AC4R", 17, 13, 16, 0x66c5d8eaULL, &nppiMaxIndx_32f_AC4R_Ctx);
}

// ============================ nppiMinMaxIndx ==================================
TEST_F(NppiIndexMinMaxDiff, MinMaxIndx_8u) {
  mm_c1<Npp8u>("nppiMinMaxIndx_8u_C1R_Ctx", "MinMaxIndx/8u/C1R", 17, 13, 16, 0x713400aaULL, &nppiMinMaxIndx_8u_C1R_Ctx);
  mm_c1m<Npp8u>("nppiMinMaxIndx_8u_C1MR_Ctx", "MinMaxIndx/8u/C1MR", 17, 13, 16, 0x2045eefaULL, &nppiMinMaxIndx_8u_C1MR_Ctx);
  mm_c3c<Npp8u>("nppiMinMaxIndx_8u_C3CR_Ctx", "MinMaxIndx/8u/C3CR", 17, 13, 16, 0xcc55ee11ULL, 2, &nppiMinMaxIndx_8u_C3CR_Ctx);
  mm_c3cm<Npp8u>("nppiMinMaxIndx_8u_C3CMR_Ctx", "MinMaxIndx/8u/C3CMR", 17, 13, 16, 0x9d1c2b3aULL, 1, &nppiMinMaxIndx_8u_C3CMR_Ctx);
}
TEST_F(NppiIndexMinMaxDiff, MinMaxIndx_16u) {
  mm_c1<Npp16u>("nppiMinMaxIndx_16u_C1R_Ctx", "MinMaxIndx/16u/C1R", 17, 13, 16, 0x55aa66bbULL, &nppiMinMaxIndx_16u_C1R_Ctx);
  mm_c1m<Npp16u>("nppiMinMaxIndx_16u_C1MR_Ctx", "MinMaxIndx/16u/C1MR", 17, 13, 16, 0x7799ee33ULL, &nppiMinMaxIndx_16u_C1MR_Ctx);
  mm_c3c<Npp16u>("nppiMinMaxIndx_16u_C3CR_Ctx", "MinMaxIndx/16u/C3CR", 17, 13, 16, 0x1234abcdULL, 3, &nppiMinMaxIndx_16u_C3CR_Ctx);
  mm_c3cm<Npp16u>("nppiMinMaxIndx_16u_C3CMR_Ctx", "MinMaxIndx/16u/C3CMR", 17, 13, 16, 0x5678efaaULL, 2, &nppiMinMaxIndx_16u_C3CMR_Ctx);
}
TEST_F(NppiIndexMinMaxDiff, MinMaxIndx_32f) {
  mm_c1<Npp32f>("nppiMinMaxIndx_32f_C1R_Ctx", "MinMaxIndx/32f/C1R", 17, 13, 16, 0xdeadbeefULL, &nppiMinMaxIndx_32f_C1R_Ctx);
  mm_c1m<Npp32f>("nppiMinMaxIndx_32f_C1MR_Ctx", "MinMaxIndx/32f/C1MR", 17, 13, 16, 0x3241c0deULL, &nppiMinMaxIndx_32f_C1MR_Ctx);
  mm_c3c<Npp32f>("nppiMinMaxIndx_32f_C3CR_Ctx", "MinMaxIndx/32f/C3CR", 17, 13, 16, 0xa1b2c3d4ULL, 1, &nppiMinMaxIndx_32f_C3CR_Ctx);
  mm_c3cm<Npp32f>("nppiMinMaxIndx_32f_C3CMR_Ctx", "MinMaxIndx/32f/C3CMR", 17, 13, 16, 0x9012f3e4ULL, 3, &nppiMinMaxIndx_32f_C3CMR_Ctx);
}
TEST_F(NppiIndexMinMaxDiff, MinMaxIndx_8s) {
  mm_c1<Npp8s>("nppiMinMaxIndx_8s_C1R_Ctx", "MinMaxIndx/8s/C1R", 17, 13, 16, 0x7f6e5d4cULL, &nppiMinMaxIndx_8s_C1R_Ctx);
  mm_c1m<Npp8s>("nppiMinMaxIndx_8s_C1MR_Ctx", "MinMaxIndx/8s/C1MR", 17, 13, 16, 0x8a1b2c3dULL, &nppiMinMaxIndx_8s_C1MR_Ctx);
  mm_c3c<Npp8s>("nppiMinMaxIndx_8s_C3CR_Ctx", "MinMaxIndx/8s/C3CR", 17, 13, 16, 0x9e0f1a2bULL, 2, &nppiMinMaxIndx_8s_C3CR_Ctx);
  mm_c3cm<Npp8s>("nppiMinMaxIndx_8s_C3CMR_Ctx", "MinMaxIndx/8s/C3CMR", 17, 13, 16, 0x1a2b3c4dULL, 1, &nppiMinMaxIndx_8s_C3CMR_Ctx);
}

// ============================ MinEvery_AC4 / MaxEvery_AC4 ============================
TEST_F(NppiIndexMinMaxDiff, MinEvery_AC4) {
  every_ac4<Npp8u, 4>("nppiMinEvery_8u_AC4IR_Ctx", "MinEvery/8u/AC4IR", 17, 13, 16, 0x1f2e3d4cULL, &nppiMinEvery_8u_AC4IR_Ctx);
  every_ac4<Npp16u, 4>("nppiMinEvery_16u_AC4IR_Ctx", "MinEvery/16u/AC4IR", 17, 13, 16, 0x21c3b4a5ULL, &nppiMinEvery_16u_AC4IR_Ctx);
  every_ac4<Npp16s, 4>("nppiMinEvery_16s_AC4IR_Ctx", "MinEvery/16s/AC4IR", 17, 13, 16, 0x305f2e1dULL, &nppiMinEvery_16s_AC4IR_Ctx);
  every_ac4<Npp32f, 4>("nppiMinEvery_32f_AC4IR_Ctx", "MinEvery/32f/AC4IR", 17, 13, 16, 0x49a8b7c6ULL, &nppiMinEvery_32f_AC4IR_Ctx);
}
TEST_F(NppiIndexMinMaxDiff, MaxEvery_AC4) {
  every_ac4<Npp8u, 4>("nppiMaxEvery_8u_AC4IR_Ctx", "MaxEvery/8u/AC4IR", 17, 13, 16, 0x77e5d4c3ULL, &nppiMaxEvery_8u_AC4IR_Ctx);
  every_ac4<Npp16u, 4>("nppiMaxEvery_16u_AC4IR_Ctx", "MaxEvery/16u/AC4IR", 17, 13, 16, 0x66d3c2b1ULL, &nppiMaxEvery_16u_AC4IR_Ctx);
  every_ac4<Npp16s, 4>("nppiMaxEvery_16s_AC4IR_Ctx", "MaxEvery/16s/AC4IR", 17, 13, 16, 0x55c1b0a9ULL, &nppiMaxEvery_16s_AC4IR_Ctx);
  every_ac4<Npp32f, 4>("nppiMaxEvery_32f_AC4IR_Ctx", "MaxEvery/32f/AC4IR", 17, 13, 16, 0x44a91827ULL, &nppiMaxEvery_32f_AC4IR_Ctx);
}

// ============================ buffer-size helpers ============================
TEST_F(NppiIndexMinMaxDiff, BufferSize_MinIndx) {
  SIZE_TYPE n = 0;
  NppStatus st = nppiMinIndxGetBufferHostSize_8u_C1R_Ctx(NppiSize{17, 13}, &n, current_context());
  Case c("bufferSize/minIndx_8u_C1R", "nppiMinIndxGetBufferHostSize_8u_C1R_Ctx");
  c.input(&n, sizeof(n));
  c.status(st);
  c.finish();
}
TEST_F(NppiIndexMinMaxDiff, BufferSize_MaxIndx) {
  SIZE_TYPE n = 0;
  NppStatus st = nppiMaxIndxGetBufferHostSize_16s_C4R_Ctx(NppiSize{64, 32}, &n, current_context());
  Case c("bufferSize/maxIndx_16s_C4R", "nppiMaxIndxGetBufferHostSize_16s_C4R_Ctx");
  c.input(&n, sizeof(n));
  c.status(st);
  c.finish();
}
TEST_F(NppiIndexMinMaxDiff, BufferSize_MinMaxIndx) {
  SIZE_TYPE n = 0;
  NppStatus st = nppiMinMaxIndxGetBufferHostSize_32f_C3CMR_Ctx(NppiSize{64, 32}, &n, current_context());
  Case c("bufferSize/minmaxIndx_32f_C3CMR", "nppiMinMaxIndxGetBufferHostSize_32f_C3CMR_Ctx");
  c.input(&n, sizeof(n));
  c.status(st);
  c.finish();
}

// Allocate with the returned size and run inside it.
TEST_F(NppiIndexMinMaxDiff, BufferAllocRun) {
  SIZE_TYPE n = 0;
  NppiSize roi{64, 32};
  NppStreamContext ctx = current_context();
  NppStatus st = nppiMinIndxGetBufferHostSize_32f_AC4R_Ctx(roi, &n, ctx);
  if (st == NPP_NO_ERROR) {
    DeviceMemory<Npp8u> &b = arena<Npp8u>((size_t)n);
    const size_t elems = (size_t)64 * 32 * 4;
    std::vector<Npp32f> hs = make_uniform<Npp32f>(elems, 0x12345678ULL);
    DeviceMemory<Npp32f> &s = arena<Npp32f>(elems);
    s.copyFromHost(hs);
    DeviceMemory<Npp32f> &val = arena<Npp32f>(3);
    DeviceMemory<int> &ix = arena<int>(3);
    DeviceMemory<int> &iy = arena<int>(3);
    clear_cuda_error();
    NppStatus runst = nppiMinIndx_32f_AC4R_Ctx(s.get(), (int)(64 * 4 * 4), roi, b.get(), val.get(), ix.get(), iy.get(), ctx);
    Case c2("bufferAllocRun/minIndx_32f_AC4R", "nppiMinIndx_32f_AC4R_Ctx");
    c2.status(runst);
    c2.device_output(val.get(), 3, ValueKind::F32, 0.0, 0.0);
    c2.device_output(ix.get(), 3, ValueKind::I32, 0.0, 0.0);
    c2.device_output(iy.get(), 3, ValueKind::I32, 0.0, 0.0);
    c2.input(hs.data(), hs.size() * sizeof(Npp32f));
    c2.finish();
  }
}

// Minimal valid 1x1 round-trips (safe contract probes; error inputs avoid
// NVIDIA 13 deferred faults as noted above).
TEST_F(NppiIndexMinMaxDiff, Contract_1x1) {
  extr_indx<Npp8u, 1, 1>("nppiMinIndx_8u_C1R_Ctx", "Contract/MinIndx/8u/C1R", 1, 1, 0, 0x1ULL, &nppiMinIndx_8u_C1R_Ctx);
  extr_indx<Npp8u, 1, 1>("nppiMaxIndx_8u_C1R_Ctx", "Contract/MaxIndx/8u/C1R", 1, 1, 0, 0x2ULL, &nppiMaxIndx_8u_C1R_Ctx);
  mm_c1<Npp8u>("nppiMinMaxIndx_8u_C1R_Ctx", "Contract/MinMaxIndx/8u/C1R", 1, 1, 0, 0x3ULL, &nppiMinMaxIndx_8u_C1R_Ctx);
  every_ac4<Npp8u, 4>("nppiMinEvery_8u_AC4IR_Ctx", "Contract/MinEvery/8u/AC4IR", 1, 1, 0, 0x4ULL, &nppiMinEvery_8u_AC4IR_Ctx);
}



