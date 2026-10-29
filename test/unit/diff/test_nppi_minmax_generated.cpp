// Differential tests for nppiMin / nppiMax / nppiMinMax (package p3.1a2).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiMinMaxDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppiMinMaxDiff.*'
//
// NVIDIA NPP 13 only exports the _Ctx entry points; the reference matrix uses
// those.  Min/Max/MinMax write device scalars or per-channel arrays whose type
// always equals the source dtype, captured with the matching exact ValueKind.
// Buffer-size helpers are only checked for status; the "allocate then run" case
// confirms the returned size is usable.

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

// Single-output extreme reduction (Min or Max).  NCh = packed channels per
// pixel; OutCount = number of reduced output channels (NCh for C1/C3/C4, 3 for
// AC4).
template <typename D, int NCh, int OutCount>
void extreme(const char *fname, const std::string &idBase, int width, int height, int pad, uint64_t seed_v,
             ValueKind vk, NppStatus (*f)(const D *, int, NppiSize, Npp8u *, D *, NppStreamContext)) {
  const int se = (int)sizeof(D);
  const int step = align_step(width * NCh * se + pad, se);
  const size_t stepElems = (size_t)step / se;
  const size_t elems = (size_t)height * stepElems;
  std::vector<D> hs = make_uniform<D>(elems, seed_v);
  DeviceMemory<D> &s = arena<D>(elems);
  DeviceMemory<D> &o = arena<D>(OutCount);
  s.copyFromHost(hs);
  Case c(idBase + "_" + std::to_string(width) + "x" + std::to_string(height), fname);
  clear_cuda_error();
  NppStatus st = f(s.get(), step, NppiSize{width, height}, scratch(), o.get(), current_context());
  c.input(hs.data(), hs.size() * sizeof(D));
  c.status(st);
  c.device_output(o.get(), OutCount, vk, 0.0, 0.0);
  c.finish();
}

// MinMax pair reduction (two output arrays of OutCount each).
template <typename D, int NCh, int OutCount>
void minmax(const char *fname, const std::string &idBase, int width, int height, int pad, uint64_t seed_v,
            ValueKind vk, NppStatus (*f)(const D *, int, NppiSize, D *, D *, Npp8u *, NppStreamContext)) {
  const int se = (int)sizeof(D);
  const int step = align_step(width * NCh * se + pad, se);
  const size_t stepElems = (size_t)step / se;
  const size_t elems = (size_t)height * stepElems;
  std::vector<D> hs = make_uniform<D>(elems, seed_v);
  DeviceMemory<D> &s = arena<D>(elems);
  DeviceMemory<D> &mn = arena<D>(OutCount);
  DeviceMemory<D> &mx = arena<D>(OutCount);
  s.copyFromHost(hs);
  Case c(idBase + "_" + std::to_string(width) + "x" + std::to_string(height), fname);
  clear_cuda_error();
  NppStatus st = f(s.get(), step, NppiSize{width, height}, mn.get(), mx.get(), scratch(), current_context());
  c.input(hs.data(), hs.size() * sizeof(D));
  c.status(st);
  c.device_output(mn.get(), OutCount, vk, 0.0, 0.0);
  c.device_output(mx.get(), OutCount, vk, 0.0, 0.0);
  c.finish();
}

}  // namespace

class NppiMinMaxDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

// ============================ nppiMin ========================================
TEST_F(NppiMinMaxDiff, Min_8u) {
  extreme<Npp8u, 1, 1>("nppiMin_8u_C1R_Ctx", "Min/8u/C1R", 17, 13, 16, 0x14e1ed4b, ValueKind::U8, &nppiMin_8u_C1R_Ctx);
  extreme<Npp8u, 3, 3>("nppiMin_8u_C3R_Ctx", "Min/8u/C3R", 17, 13, 16, 0xedbaacf0, ValueKind::U8, &nppiMin_8u_C3R_Ctx);
  extreme<Npp8u, 4, 4>("nppiMin_8u_C4R_Ctx", "Min/8u/C4R", 64, 32, 32, 0x41617a89, ValueKind::U8, &nppiMin_8u_C4R_Ctx);
  extreme<Npp8u, 4, 3>("nppiMin_8u_AC4R_Ctx", "Min/8u/AC4R", 1, 1, 8, 0x24ee383f, ValueKind::U8, &nppiMin_8u_AC4R_Ctx);
}
TEST_F(NppiMinMaxDiff, Min_16u) {
  extreme<Npp16u, 1, 1>("nppiMin_16u_C1R_Ctx", "Min/16u/C1R", 17, 13, 16, 0xe84fde64, ValueKind::U16, &nppiMin_16u_C1R_Ctx);
  extreme<Npp16u, 3, 3>("nppiMin_16u_C3R_Ctx", "Min/16u/C3R", 17, 13, 16, 0x5fc779a2, ValueKind::U16, &nppiMin_16u_C3R_Ctx);
  extreme<Npp16u, 4, 4>("nppiMin_16u_C4R_Ctx", "Min/16u/C4R", 64, 32, 32, 0xf11c5ee5, ValueKind::U16, &nppiMin_16u_C4R_Ctx);
  extreme<Npp16u, 4, 3>("nppiMin_16u_AC4R_Ctx", "Min/16u/AC4R", 1, 1, 8, 0x8d8fa12f, ValueKind::U16, &nppiMin_16u_AC4R_Ctx);
}
TEST_F(NppiMinMaxDiff, Min_16s) {
  extreme<Npp16s, 1, 1>("nppiMin_16s_C1R_Ctx", "Min/16s/C1R", 17, 13, 16, 0x637029aa, ValueKind::I16, &nppiMin_16s_C1R_Ctx);
  extreme<Npp16s, 3, 3>("nppiMin_16s_C3R_Ctx", "Min/16s/C3R", 17, 13, 16, 0x6bf88e6c, ValueKind::I16, &nppiMin_16s_C3R_Ctx);
  extreme<Npp16s, 4, 4>("nppiMin_16s_C4R_Ctx", "Min/16s/C4R", 64, 32, 32, 0xc323990d, ValueKind::I16, &nppiMin_16s_C4R_Ctx);
  extreme<Npp16s, 4, 3>("nppiMin_16s_AC4R_Ctx", "Min/16s/AC4R", 1, 1, 8, 0xa6b056c3, ValueKind::I16, &nppiMin_16s_AC4R_Ctx);
}
TEST_F(NppiMinMaxDiff, Min_32f) {
  extreme<Npp32f, 1, 1>("nppiMin_32f_C1R_Ctx", "Min/32f/C1R", 17, 13, 16, 0x0c135170, ValueKind::F32, &nppiMin_32f_C1R_Ctx);
  extreme<Npp32f, 3, 3>("nppiMin_32f_C3R_Ctx", "Min/32f/C3R", 17, 13, 16, 0x949bb632, ValueKind::F32, &nppiMin_32f_C3R_Ctx);
  extreme<Npp32f, 4, 4>("nppiMin_32f_C4R_Ctx", "Min/32f/C4R", 64, 32, 32, 0x9a807147, ValueKind::F32, &nppiMin_32f_C4R_Ctx);
  extreme<Npp32f, 4, 3>("nppiMin_32f_AC4R_Ctx", "Min/32f/AC4R", 1, 1, 8, 0x7e0d2efd, ValueKind::F32, &nppiMin_32f_AC4R_Ctx);
}
// ============================ nppiMax ========================================
TEST_F(NppiMinMaxDiff, Max_8u) {
  extreme<Npp8u, 1, 1>("nppiMax_8u_C1R_Ctx", "Max/8u/C1R", 17, 13, 16, 0xcf739014, ValueKind::U8, &nppiMax_8u_C1R_Ctx);
  extreme<Npp8u, 3, 3>("nppiMax_8u_C3R_Ctx", "Max/8u/C3R", 17, 13, 16, 0x52102a4f, ValueKind::U8, &nppiMax_8u_C3R_Ctx);
  extreme<Npp8u, 4, 4>("nppiMax_8u_C4R_Ctx", "Max/8u/C4R", 64, 32, 32, 0xdd0bfd2a, ValueKind::U8, &nppiMax_8u_C4R_Ctx);
  extreme<Npp8u, 4, 3>("nppiMax_8u_AC4R_Ctx", "Max/8u/AC4R", 1, 1, 8, 0xbf674520, ValueKind::U8, &nppiMax_8u_AC4R_Ctx);
}
TEST_F(NppiMinMaxDiff, Max_16u) {
  extreme<Npp16u, 1, 1>("nppiMax_16u_C1R_Ctx", "Max/16u/C1R", 17, 13, 16, 0x4e0e5dec, ValueKind::U16, &nppiMax_16u_C1R_Ctx);
  extreme<Npp16u, 3, 3>("nppiMax_16u_C3R_Ctx", "Max/16u/C3R", 17, 13, 16, 0xcb71c3b1, ValueKind::U16, &nppiMax_16u_C3R_Ctx);
  extreme<Npp16u, 4, 4>("nppiMax_16u_C4R_Ctx", "Max/16u/C4R", 64, 32, 32, 0x057214d6, ValueKind::U16, &nppiMax_16u_C4R_Ctx);
  extreme<Npp16u, 4, 3>("nppiMax_16u_AC4R_Ctx", "Max/16u/AC4R", 1, 1, 8, 0x5e1aa8e0, ValueKind::U16, &nppiMax_16u_AC4R_Ctx);
}
TEST_F(NppiMinMaxDiff, Max_16s) {
  extreme<Npp16s, 1, 1>("nppiMax_16s_C1R_Ctx", "Max/16s/C1R", 17, 13, 16, 0xf7c5df9b, ValueKind::I16, &nppiMax_16s_C1R_Ctx);
  extreme<Npp16s, 3, 3>("nppiMax_16s_C3R_Ctx", "Max/16s/C3R", 17, 13, 16, 0x804e445d, ValueKind::I16, &nppiMax_16s_C3R_Ctx);
  extreme<Npp16s, 4, 4>("nppiMax_16s_C4R_Ctx", "Max/16s/C4R", 64, 32, 32, 0xaecde31c, ValueKind::I16, &nppiMax_16s_C4R_Ctx);
  extreme<Npp16s, 4, 3>("nppiMax_16s_AC4R_Ctx", "Max/16s/AC4R", 1, 1, 8, 0x925aa0d2, ValueKind::I16, &nppiMax_16s_AC4R_Ctx);
}
TEST_F(NppiMinMaxDiff, Max_32f) {
  extreme<Npp32f, 1, 1>("nppiMax_32f_C1R_Ctx", "Max/32f/C1R", 17, 13, 16, 0x2cc02830, ValueKind::F32, &nppiMax_32f_C1R_Ctx);
  extreme<Npp32f, 3, 3>("nppiMax_32f_C3R_Ctx", "Max/32f/C3R", 17, 13, 16, 0x2437c36e, ValueKind::F32, &nppiMax_32f_C3R_Ctx);
  extreme<Npp32f, 4, 4>("nppiMax_32f_C4R_Ctx", "Max/32f/C4R", 64, 32, 32, 0xacac1519, ValueKind::F32, &nppiMax_32f_C4R_Ctx);
  extreme<Npp32f, 4, 3>("nppiMax_32f_AC4R_Ctx", "Max/32f/AC4R", 1, 1, 8, 0x491f5763, ValueKind::F32, &nppiMax_32f_AC4R_Ctx);
}
// ============================ nppiMinMax ========================================
TEST_F(NppiMinMaxDiff, MinMax_8u) {
  minmax<Npp8u, 1, 1>("nppiMinMax_8u_C1R_Ctx", "MinMax/8u/C1R", 17, 13, 16, 0x126cc016, ValueKind::U8, &nppiMinMax_8u_C1R_Ctx);
  minmax<Npp8u, 3, 3>("nppiMinMax_8u_C3R_Ctx", "MinMax/8u/C3R", 17, 13, 16, 0x89e45b54, ValueKind::U8, &nppiMinMax_8u_C3R_Ctx);
  minmax<Npp8u, 4, 4>("nppiMinMax_8u_C4R_Ctx", "MinMax/8u/C4R", 64, 32, 32, 0xc6ff7d33, ValueKind::U8, &nppiMinMax_8u_C4R_Ctx);
  minmax<Npp8u, 4, 3>("nppiMinMax_8u_AC4R_Ctx", "MinMax/8u/AC4R", 1, 1, 8, 0x6372bf7d, ValueKind::U8, &nppiMinMax_8u_AC4R_Ctx);
}
TEST_F(NppiMinMaxDiff, MinMax_16u) {
  minmax<Npp16u, 1, 1>("nppiMinMax_16u_C1R_Ctx", "MinMax/16u/C1R", 17, 13, 16, 0x5feee6a8, ValueKind::U16, &nppiMinMax_16u_C1R_Ctx);
  minmax<Npp16u, 3, 3>("nppiMinMax_16u_C3R_Ctx", "MinMax/16u/C3R", 17, 13, 16, 0x576681e6, ValueKind::U16, &nppiMinMax_16u_C3R_Ctx);
  minmax<Npp16u, 4, 4>("nppiMinMax_16u_C4R_Ctx", "MinMax/16u/C4R", 64, 32, 32, 0x797d56a1, ValueKind::U16, &nppiMinMax_16u_C4R_Ctx);
  minmax<Npp16u, 4, 3>("nppiMinMax_16u_AC4R_Ctx", "MinMax/16u/AC4R", 1, 1, 8, 0x15f098eb, ValueKind::U16, &nppiMinMax_16u_AC4R_Ctx);
}
TEST_F(NppiMinMaxDiff, MinMax_16s) {
  minmax<Npp16s, 1, 1>("nppiMinMax_16s_C1R_Ctx", "MinMax/16s/C1R", 17, 13, 16, 0xebd12166, ValueKind::I16, &nppiMinMax_16s_C1R_Ctx);
  minmax<Npp16s, 3, 3>("nppiMinMax_16s_C3R_Ctx", "MinMax/16s/C3R", 17, 13, 16, 0x6e6dbba1, ValueKind::I16, &nppiMinMax_16s_C3R_Ctx);
  minmax<Npp16s, 4, 4>("nppiMinMax_16s_C4R_Ctx", "MinMax/16s/C4R", 64, 32, 32, 0xc0ae6bd8, ValueKind::I16, &nppiMinMax_16s_C4R_Ctx);
  minmax<Npp16s, 4, 3>("nppiMinMax_16s_AC4R_Ctx", "MinMax/16s/AC4R", 1, 1, 8, 0xa43b298e, ValueKind::I16, &nppiMinMax_16s_AC4R_Ctx);
}
TEST_F(NppiMinMaxDiff, MinMax_32f) {
  minmax<Npp32f, 1, 1>("nppiMinMax_32f_C1R_Ctx", "MinMax/32f/C1R", 17, 13, 16, 0x474b199b, ValueKind::F32, &nppiMinMax_32f_C1R_Ctx);
  minmax<Npp32f, 3, 3>("nppiMinMax_32f_C3R_Ctx", "MinMax/32f/C3R", 17, 13, 16, 0x36184c2a, ValueKind::F32, &nppiMinMax_32f_C3R_Ctx);
  minmax<Npp32f, 4, 4>("nppiMinMax_32f_C4R_Ctx", "MinMax/32f/C4R", 64, 32, 32, 0x9acb8c5d, ValueKind::F32, &nppiMinMax_32f_C4R_Ctx);
  minmax<Npp32f, 4, 3>("nppiMinMax_32f_AC4R_Ctx", "MinMax/32f/AC4R", 1, 1, 8, 0x373ecea7, ValueKind::F32, &nppiMinMax_32f_AC4R_Ctx);
}

// ============================ buffer-size helpers ===========================
TEST_F(NppiMinMaxDiff, BufferSize_Min) {
  SIZE_TYPE n = 0;
  NppStatus st = nppiMinGetBufferHostSize_8u_C4R_Ctx(NppiSize{17, 13}, &n, current_context());
  Case c("bufferSize/min_8u_C4R", "nppiMinGetBufferHostSize_8u_C4R_Ctx");
  c.input(&n, sizeof(n));
  c.status(st);
  c.finish();
}
TEST_F(NppiMinMaxDiff, BufferSize_Max) {
  SIZE_TYPE n = 0;
  NppStatus st = nppiMaxGetBufferHostSize_8u_C4R_Ctx(NppiSize{17, 13}, &n, current_context());
  Case c("bufferSize/max_8u_C4R", "nppiMaxGetBufferHostSize_8u_C4R_Ctx");
  c.input(&n, sizeof(n));
  c.status(st);
  c.finish();
}
TEST_F(NppiMinMaxDiff, BufferSize_MinMax) {
  SIZE_TYPE n = 0;
  NppStatus st = nppiMinMaxGetBufferHostSize_8u_C4R_Ctx(NppiSize{17, 13}, &n, current_context());
  Case c("bufferSize/minmax_8u_C4R", "nppiMinMaxGetBufferHostSize_8u_C4R_Ctx");
  c.input(&n, sizeof(n));
  c.status(st);
  c.finish();
}

// Allocate with the returned size, then run inside it: the helper size must be
// a usable scratch allocation and the reduction must succeed.
TEST_F(NppiMinMaxDiff, BufferAllocRun) {
  SIZE_TYPE n = 0;
  NppiSize roi{64, 32};
  NppStreamContext ctx = current_context();
  NppStatus st = nppiMinMaxGetBufferHostSize_32f_AC4R_Ctx(roi, &n, ctx);
  if (st == NPP_NO_ERROR) {
    DeviceMemory<Npp8u> &b = arena<Npp8u>((size_t)n);
    const size_t elems = (size_t)64 * 32 * 4;
    std::vector<Npp32f> hs = make_uniform<Npp32f>(elems, 0xBEEF);
    DeviceMemory<Npp32f> &s = arena<Npp32f>(elems);
    s.copyFromHost(hs);
    DeviceMemory<Npp32f> &mn = arena<Npp32f>(3);
    DeviceMemory<Npp32f> &mx = arena<Npp32f>(3);
    clear_cuda_error();
    NppStatus runst = nppiMinMax_32f_AC4R_Ctx(s.get(), (int)(64 * 4 * 4), roi, mn.get(), mx.get(), b.get(), ctx);
    Case c2("bufferAllocRun/minmax_32f_AC4R", "nppiMinMax_32f_AC4R_Ctx");
    c2.status(runst);
    c2.device_output(mn.get(), 3, ValueKind::F32, 0.0, 0.0);
    c2.device_output(mx.get(), 3, ValueKind::F32, 0.0, 0.0);
    c2.finish();
  }
}

#if NPP_DIFF_HAS_NONCTX
template <typename D, int NCh, int OutCount>
void nonctx_extreme(const char *fname, const std::string &idBase, int width, int height, int pad, uint64_t seed_v,
                    ValueKind vk, NppStatus (*f)(const D *, int, NppiSize, Npp8u *, D *)) {
  const int se = (int)sizeof(D);
  const int step = align_step(width * NCh * se + pad, se);
  const size_t stepElems = (size_t)step / se;
  const size_t elems = (size_t)height * stepElems;
  std::vector<D> hs = make_uniform<D>(elems, seed_v);
  DeviceMemory<D> &s = arena<D>(elems);
  DeviceMemory<D> &o = arena<D>(OutCount);
  s.copyFromHost(hs);
  Case c(idBase + "_" + std::to_string(width) + "x" + std::to_string(height), fname);
  clear_cuda_error();
  NppStatus st = f(s.get(), step, NppiSize{width, height}, scratch(), o.get());
  c.input(hs.data(), hs.size() * sizeof(D));
  c.status(st);
  c.device_output(o.get(), OutCount, vk, 0.0, 0.0);
  c.finish();
}

TEST_F(NppiMinMaxDiff, NonCtx_Min_8u_C1R) {
  nonctx_extreme<Npp8u, 1, 1>("nppiMin_8u_C1R", "NonCtx/Min/8u/C1R", 17, 13, 16, 0x101, ValueKind::U8, &nppiMin_8u_C1R);
}
TEST_F(NppiMinMaxDiff, NonCtx_Max_16s_C1R) {
  nonctx_extreme<Npp16s, 1, 1>("nppiMax_16s_C1R", "NonCtx/Max/16s/C1R", 17, 13, 16, 0x3001, ValueKind::I16, &nppiMax_16s_C1R);
}
#endif

// ============================ contract ======================================
// Null pointers, empty / negative ROI, step too short. The status is captured
// via Case so the golden records the actual 5090/NVIDIA status name and verify
// compares against that same name.
TEST_F(NppiMinMaxDiff, Contract_Min_8u_C1R) {
  DeviceMemory<Npp8u> &s = arena<Npp8u>(16);
  DeviceMemory<Npp8u> &o = arena<Npp8u>(1);
  NppStreamContext ctx = current_context();
  {
    clear_cuda_error();
    NppStatus st = nppiMin_8u_C1R_Ctx(nullptr, 16, NppiSize{2, 2}, scratch(), o.get(), ctx);
    Case c("min/8u/C1R/null_src", "nppiMin_8u_C1R"); c.status(st); c.finish();
  }
  {
    clear_cuda_error();
    NppStatus st = nppiMin_8u_C1R_Ctx(s.get(), 16, NppiSize{2, 2}, scratch(), nullptr, ctx);
    Case c("min/8u/C1R/null_out", "nppiMin_8u_C1R"); c.status(st); c.finish();
  }
  {
    clear_cuda_error();
    NppStatus st = nppiMin_8u_C1R_Ctx(s.get(), 16, NppiSize{0, 2}, scratch(), o.get(), ctx);
    Case c("min/8u/C1R/zro_roi", "nppiMin_8u_C1R"); c.status(st); c.finish();
  }
  {
    clear_cuda_error();
    NppStatus st = nppiMin_8u_C1R_Ctx(s.get(), 16, NppiSize{-1, 2}, scratch(), o.get(), ctx);
    Case c("min/8u/C1R/neg_size", "nppiMin_8u_C1R"); c.status(st); c.finish();
  }
  {
    clear_cuda_error();
    NppStatus st = nppiMin_8u_C1R_Ctx(s.get(), 2, NppiSize{4, 2}, scratch(), o.get(), ctx);
    Case c("min/8u/C1R/short_step", "nppiMin_8u_C1R"); c.status(st); c.finish();
  }
}
TEST_F(NppiMinMaxDiff, Contract_Max_8u_C1R) {
  DeviceMemory<Npp8u> &s = arena<Npp8u>(16);
  DeviceMemory<Npp8u> &o = arena<Npp8u>(1);
  NppStreamContext ctx = current_context();
  {
    clear_cuda_error();
    NppStatus st = nppiMax_8u_C1R_Ctx(nullptr, 16, NppiSize{2, 2}, scratch(), o.get(), ctx);
    Case c("max/8u/C1R/null_src", "nppiMax_8u_C1R"); c.status(st); c.finish();
  }
  {
    clear_cuda_error();
    NppStatus st = nppiMax_8u_C1R_Ctx(s.get(), 16, NppiSize{0, 0}, scratch(), o.get(), ctx);
    Case c("max/8u/C1R/zro_roi", "nppiMax_8u_C1R"); c.status(st); c.finish();
  }
  {
    clear_cuda_error();
    NppStatus st = nppiMax_8u_C1R_Ctx(s.get(), 16, NppiSize{-1, 2}, scratch(), o.get(), ctx);
    Case c("max/8u/C1R/neg_size", "nppiMax_8u_C1R"); c.status(st); c.finish();
  }
  {
    clear_cuda_error();
    NppStatus st = nppiMax_8u_C1R_Ctx(s.get(), 3, NppiSize{4, 2}, scratch(), o.get(), ctx);
    Case c("max/8u/C1R/short_step", "nppiMax_8u_C1R"); c.status(st); c.finish();
  }
}
TEST_F(NppiMinMaxDiff, Contract_MinMax_8u_C1R) {
  DeviceMemory<Npp8u> &s = arena<Npp8u>(16);
  DeviceMemory<Npp8u> &mn = arena<Npp8u>(1);
  DeviceMemory<Npp8u> &mx = arena<Npp8u>(1);
  NppStreamContext ctx = current_context();
  {
    clear_cuda_error();
    NppStatus st = nppiMinMax_8u_C1R_Ctx(nullptr, 16, NppiSize{2, 2}, mn.get(), mx.get(), scratch(), ctx);
    Case c("minmax/8u/C1R/null_src", "nppiMinMax_8u_C1R"); c.status(st); c.finish();
  }
  {
    clear_cuda_error();
    NppStatus st = nppiMinMax_8u_C1R_Ctx(s.get(), 16, NppiSize{1, 1}, mn.get(), mx.get(), scratch(), ctx);
    Case c2("minmax/8u/C1R_1x1", "nppiMinMax_8u_C1R"); c2.status(st);
    c2.device_output(mn.get(), 1, ValueKind::U8, 0.0, 0.0);
    c2.device_output(mx.get(), 1, ValueKind::U8, 0.0, 0.0);
    c2.finish();
  }
  {
    clear_cuda_error();
    NppStatus st = nppiMinMax_8u_C1R_Ctx(s.get(), 16, NppiSize{-1, 2}, mn.get(), mx.get(), scratch(), ctx);
    Case c("minmax/8u/C1R/neg_size", "nppiMinMax_8u_C1R"); c.status(st); c.finish();
  }
  {
    clear_cuda_error();
    NppStatus st = nppiMinMax_8u_C1R_Ctx(s.get(), 2, NppiSize{4, 2}, mn.get(), mx.get(), scratch(), ctx);
    Case c("minmax/8u/C1R/short_step", "nppiMinMax_8u_C1R"); c.status(st); c.finish();
  }
}
