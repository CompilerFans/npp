// Differential tests for package f13 family 4: the nppiFloodFill* families
// (FloodFill / Boundary / Range / Gradient and their *Boundary forms, 72
// functions) plus nppiFloodFillGetBufferSize.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4 (only the _Ctx entry points
// exist there):
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiFloodFillDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=... LD_LIBRARY_PATH=build/src ./build/diff_tests \
//     --gtest_filter='NppiFloodFillDiff.*'
//
// Measured semantics (probes f13_probe58..63, see f13_notes.md):
//  - the region is the connected component of the seed: value equality for
//    FloodFill/Boundary, [nMin,nMax] per channel for Range, and
//    [seed-nMin, seed+nMax] (lower bound clamped at 0) for Gradient; the seed
//    is always part of the region; L1 = 4-way, Inf = 8-way connectivity;
//  - the *Boundary forms do not stop at nBoundaryValue: they write it into the
//    region pixels that have a neighbour outside the region (ROI edges do not
//    count) and nNewValue into the rest;
//  - C3 seed values are read at `y*step + x*sizeof(T)` (single-channel
//    addressing), so a seed whose column is not a multiple of 3 reads another
//    pixel and only the seed itself is filled ("quirk" case below);
//  - GetBufferSize = 4*width*height + 4*width + 64 for every type/layout.
#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <memory>
#include <string>
#include <vector>

using namespace npp_diff;
using namespace npp_functional_test;
namespace {

constexpr int W = 24, H = 18;

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

template <typename T> std::vector<T> picture_t() {
  std::vector<T> h((size_t)W * H);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) h[(size_t)y * W + x] = (T)(((x / 6 + y / 6) % 3) * 40);
  for (int i = 0; i < 8; ++i) h[(size_t)(2 + i) * W + (2 + i)] = (T)200;
  for (int x = 0; x < W; ++x) h[(size_t)12 * W + x] = (T)160;
  for (int y = 15; y < H; ++y)
    for (int x = 18; x < W; ++x) h[(size_t)y * W + x] = (T)200;
  return h;
}

std::vector<Npp8u> picture_c3() {
  std::vector<Npp8u> h((size_t)W * H * 3);
  auto put = [&](int x, int y, int v) {
    size_t i = ((size_t)y * W + x) * 3;
    h[i] = (Npp8u)v;
    h[i + 1] = (Npp8u)(v + 5);
    h[i + 2] = (Npp8u)(v + 9);
  };
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) put(x, y, ((x / 6 + y / 6) % 3) * 40);
  for (int i = 0; i < 8; ++i) put(2 + i, 2 + i, 200);
  for (int x = 0; x < W; ++x) put(x, 12, 160);
  for (int y = 15; y < H; ++y)
    for (int x = 18; x < W; ++x) put(x, y, 200);
  return h;
}

// Runs a C1 case: copies the picture into a device arena, calls `f` and records
// the status plus the whole (in-place) image.
template <typename T, typename F>
void case_c1(const std::string &fn, const std::string &key, const std::vector<T> &host, F f,
             ValueKind kind, NppiConnectedRegion *region = nullptr) {
  DeviceMemory<T> &d = arena<T>(host.size());
  DeviceMemory<Npp8u> &buf = arena<Npp8u>(1 << 16);
  d.copyFromHost(host);
  const NppStreamContext ctx = current_ctx();
  clear_cuda_error();
  NppStatus st = f(d.get(), (int)sizeof(T) * W, NppiSize{W, H}, buf.get(), region, ctx);
  Case c(fn + "/" + key, fn);
  c.status(st);
  c.device_output<T>(d.get(), host.size(), kind, 0.0, 0.0);
  c.finish();
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) ADD_FAILURE() << "async CUDA fault after " << key << ": " << cudaGetErrorString(e);
}

template <typename T, typename F>
void case_c3(const std::string &fn, const std::string &key, const std::vector<T> &host, F f,
             ValueKind kind) {
  DeviceMemory<T> &d = arena<T>(host.size());
  DeviceMemory<Npp8u> &buf = arena<Npp8u>(1 << 16);
  d.copyFromHost(host);
  const NppStreamContext ctx = current_ctx();
  clear_cuda_error();
  NppStatus st = f(d.get(), (int)sizeof(T) * W * 3, NppiSize{W, H}, buf.get(), nullptr, ctx);
  Case c(fn + "/" + key, fn);
  c.status(st);
  c.device_output<T>(d.get(), host.size(), kind, 0.0, 0.0);
  c.finish();
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) ADD_FAILURE() << "async CUDA fault after " << key << ": " << cudaGetErrorString(e);
}

template <typename F> void only_status(const std::string &fn, const std::string &key, F f) {
  clear_cuda_error();
  NppStatus st = f();
  Case c(fn + "/" + key, fn);
  c.status(st);
  c.finish();
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) ADD_FAILURE() << "async CUDA fault after " << key << ": " << cudaGetErrorString(e);
}

std::vector<Npp16u> picture_c3_16() {
  std::vector<Npp16u> h((size_t)W * H * 3);
  auto put = [&](int x, int y, int v) {
    size_t i = ((size_t)y * W + x) * 3;
    h[i] = (Npp16u)(v * 100);
    h[i + 1] = (Npp16u)(v * 100 + 7);
    h[i + 2] = (Npp16u)(v * 100 + 13);
  };
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) put(x, y, ((x / 6 + y / 6) % 3) * 40);
  for (int i = 0; i < 8; ++i) put(2 + i, 2 + i, 200);
  for (int x = 0; x < W; ++x) put(x, 12, 160);
  for (int y = 15; y < H; ++y)
    for (int x = 18; x < W; ++x) put(x, y, 200);
  return h;
}

}  // namespace

TEST(NppiFloodFillDiff, BufferSize) {
  const NppiSize rois[4] = {{W, H}, {1, 1}, {7, 3}, {100, 64}};
  for (const auto &roi : rois) {
    int size = -1;
    clear_cuda_error();
    NppStatus st = nppiFloodFillGetBufferSize(roi, &size);
    Case c("nppiFloodFillGetBufferSize/" + std::to_string(roi.width) + "x" +
               std::to_string(roi.height),
           "nppiFloodFillGetBufferSize");
    c.status(st);
    c.host_output(&size, 1, ValueKind::I32, 0.0, 0.0);
    c.finish();
  }
  only_status("nppiFloodFillGetBufferSize", "null",
              [] { return nppiFloodFillGetBufferSize(NppiSize{4, 4}, nullptr); });
  only_status("nppiFloodFillGetBufferSize", "zero-roi", [] {
    int size = -1;
    return nppiFloodFillGetBufferSize(NppiSize{0, 4}, &size);
  });
}

TEST(NppiFloodFillDiff, Plain) {
  const std::vector<Npp8u> p8 = picture_t<Npp8u>();
  const std::vector<Npp16u> p16 = picture_t<Npp16u>();
  const std::vector<Npp32u> p32 = picture_t<Npp32u>();
  for (int norm = 0; norm < 2; ++norm) {
    const NppiNorm e = norm ? nppiNormInf : nppiNormL1;
    const std::string n = norm ? "Inf" : "L1";
    case_c1<Npp8u>("nppiFloodFill_8u_C1IR", n, p8,
                   [e](Npp8u *s, int ss, NppiSize r, Npp8u *b, NppiConnectedRegion *reg,
                       NppStreamContext c) {
                     return nppiFloodFill_8u_C1IR_Ctx(s, ss, NppiPoint{3, 3}, 77, e, r, reg, b, c);
                   },
                   ValueKind::U8);
    case_c1<Npp16u>("nppiFloodFill_16u_C1IR", n, p16,
                    [e](Npp16u *s, int ss, NppiSize r, Npp8u *b, NppiConnectedRegion *reg,
                        NppStreamContext c) {
                      return nppiFloodFill_16u_C1IR_Ctx(s, ss, NppiPoint{19, 3}, 7, e, r, reg, b, c);
                    },
                    ValueKind::U16);
    case_c1<Npp32u>("nppiFloodFill_32u_C1IR", n, p32,
                    [e](Npp32u *s, int ss, NppiSize r, Npp8u *b, NppiConnectedRegion *reg,
                        NppStreamContext c) {
                      return nppiFloodFill_32u_C1IR_Ctx(s, ss, NppiPoint{13, 8}, 1, e, r, reg, b, c);
                    },
                    ValueKind::U32);
  }
  // C3: seed column 3 is a multiple of 3, so the single-channel seed address
  // still lands inside the same colour block.
  const std::vector<Npp8u> c3 = picture_c3();
  case_c3<Npp8u>("nppiFloodFill_8u_C3IR", "L1", c3,
                 [](Npp8u *s, int ss, NppiSize r, Npp8u *b, NppiConnectedRegion *reg,
                    NppStreamContext c) {
                   const Npp8u nv[3] = {77, 78, 79};
                   return nppiFloodFill_8u_C3IR_Ctx(s, ss, NppiPoint{3, 15}, nv, nppiNormL1, r, reg,
                                                    b, c);
                 },
                 ValueKind::U8);
  // quirk: seed (4,3) reads pixel (3,1) instead, so only the seed is filled
  case_c3<Npp8u>("nppiFloodFill_8u_C3IR", "L1/quirk", c3,
                 [](Npp8u *s, int ss, NppiSize r, Npp8u *b, NppiConnectedRegion *reg,
                    NppStreamContext c) {
                   const Npp8u nv[3] = {77, 78, 79};
                   return nppiFloodFill_8u_C3IR_Ctx(s, ss, NppiPoint{4, 3}, nv, nppiNormInf, r, reg,
                                                    b, c);
                 },
                 ValueKind::U8);
}

TEST(NppiFloodFillDiff, Boundary) {
  const std::vector<Npp8u> p8 = picture_t<Npp8u>();
  const std::vector<Npp32u> p32 = picture_t<Npp32u>();
  for (int norm = 0; norm < 2; ++norm) {
    const NppiNorm e = norm ? nppiNormInf : nppiNormL1;
    const std::string n = norm ? "Inf" : "L1";
    case_c1<Npp8u>("nppiFloodFillBoundary_8u_C1IR", n, p8,
                   [e](Npp8u *s, int ss, NppiSize r, Npp8u *b, NppiConnectedRegion *reg,
                       NppStreamContext c) {
                     return nppiFloodFillBoundary_8u_C1IR_Ctx(s, ss, NppiPoint{3, 3}, 33, 90, e, r,
                                                              reg, b, c);
                   },
                   ValueKind::U8);
    case_c1<Npp32u>("nppiFloodFillBoundary_32u_C1IR", n, p32,
                    [e](Npp32u *s, int ss, NppiSize r, Npp8u *b, NppiConnectedRegion *reg,
                        NppStreamContext c) {
                      return nppiFloodFillBoundary_32u_C1IR_Ctx(s, ss, NppiPoint{19, 3}, 33, 250, e,
                                                                r, reg, b, c);
                    },
                    ValueKind::U32);
  }
  const std::vector<Npp8u> c3 = picture_c3();
  case_c3<Npp8u>("nppiFloodFillBoundary_8u_C3IR", "L1", c3,
                 [](Npp8u *s, int ss, NppiSize r, Npp8u *b, NppiConnectedRegion *reg,
                    NppStreamContext c) {
                   const Npp8u nv[3] = {33, 34, 35}, bv[3] = {90, 91, 92};
                   return nppiFloodFillBoundary_8u_C3IR_Ctx(s, ss, NppiPoint{3, 15}, nv, bv,
                                                            nppiNormL1, r, reg, b, c);
                 },
                 ValueKind::U8);
  const std::vector<Npp16u> c316 = picture_c3_16();
  case_c3<Npp16u>("nppiFloodFillBoundary_16u_C3IR", "Inf/quirk", c316,
                  [](Npp16u *s, int ss, NppiSize r, Npp8u *b, NppiConnectedRegion *reg,
                     NppStreamContext c) {
                    const Npp16u nv[3] = {33, 34, 35}, bv[3] = {9000, 9001, 9002};
                    return nppiFloodFillBoundary_16u_C3IR_Ctx(s, ss, NppiPoint{4, 3}, nv, bv,
                                                              nppiNormInf, r, reg, b, c);
                  },
                  ValueKind::U16);
}

TEST(NppiFloodFillDiff, Range) {
  const std::vector<Npp8u> p8 = picture_t<Npp8u>();
  const std::vector<Npp16u> p16 = picture_t<Npp16u>();
  const std::vector<Npp32u> p32 = picture_t<Npp32u>();
  for (int norm = 0; norm < 2; ++norm) {
    const NppiNorm e = norm ? nppiNormInf : nppiNormL1;
    const std::string n = norm ? "Inf" : "L1";
    case_c1<Npp8u>("nppiFloodFillRange_8u_C1IR", n, p8,
                   [e](Npp8u *s, int ss, NppiSize r, Npp8u *b, NppiConnectedRegion *reg,
                       NppStreamContext c) {
                     return nppiFloodFillRange_8u_C1IR_Ctx(s, ss, NppiPoint{1, 1}, 40, 120, 3, e, r,
                                                           reg, b, c);
                   },
                   ValueKind::U8);
    case_c1<Npp16u>("nppiFloodFillRange_16u_C1IR", n, p16,
                    [e](Npp16u *s, int ss, NppiSize r, Npp8u *b, NppiConnectedRegion *reg,
                        NppStreamContext c) {
                      return nppiFloodFillRange_16u_C1IR_Ctx(s, ss, NppiPoint{13, 8}, 8000, 9000,
                                                             7, e, r, reg, b, c);
                    },
                    ValueKind::U16);
  }
  case_c1<Npp32u>("nppiFloodFillRange_32u_C1IR", "seed/outside", p32,
                  [](Npp32u *s, int ss, NppiSize r, Npp8u *b, NppiConnectedRegion *reg,
                     NppStreamContext c) {
                    return nppiFloodFillRange_32u_C1IR_Ctx(s, ss, NppiPoint{3, 3}, 900, 950, 5,
                                                           nppiNormL1, r, reg, b, c);
                  },
                  ValueKind::U32);
  const std::vector<Npp8u> c3 = picture_c3();
  case_c3<Npp8u>("nppiFloodFillRange_8u_C3IR", "L1", c3,
                 [](Npp8u *s, int ss, NppiSize r, Npp8u *b, NppiConnectedRegion *reg,
                    NppStreamContext c) {
                   Npp8u mn[3] = {40, 45, 49}, mx[3] = {120, 125, 129};
                   const Npp8u nv[3] = {3, 4, 5};
                   return nppiFloodFillRange_8u_C3IR_Ctx(s, ss, NppiPoint{3, 15}, mn, mx, nv,
                                                         nppiNormL1, r, reg, b, c);
                 },
                 ValueKind::U8);
}

TEST(NppiFloodFillDiff, Gradient) {
  const std::vector<Npp8u> p8 = picture_t<Npp8u>();
  const std::vector<Npp32u> p32 = picture_t<Npp32u>();
  for (int norm = 0; norm < 2; ++norm) {
    const NppiNorm e = norm ? nppiNormInf : nppiNormL1;
    const std::string n = norm ? "Inf" : "L1";
    case_c1<Npp8u>("nppiFloodFillGradient_8u_C1IR", n, p8,
                   [e](Npp8u *s, int ss, NppiSize r, Npp8u *b, NppiConnectedRegion *reg,
                       NppStreamContext c) {
                     return nppiFloodFillGradient_8u_C1IR_Ctx(s, ss, NppiPoint{1, 1}, 20, 40, 3, e,
                                                              r, reg, b, c);
                   },
                   ValueKind::U8);
  }
  // seed 40, nMin 60 -> the lower bound clamps to 0 (unsigned underflow)
  case_c1<Npp8u>("nppiFloodFillGradient_8u_C1IR", "underflow", p8,
                 [](Npp8u *s, int ss, NppiSize r, Npp8u *b, NppiConnectedRegion *reg,
                    NppStreamContext c) {
                   return nppiFloodFillGradient_8u_C1IR_Ctx(s, ss, NppiPoint{1, 1}, 60, 20, 3,
                                                            nppiNormL1, r, reg, b, c);
                 },
                 ValueKind::U8);
  case_c1<Npp32u>("nppiFloodFillGradient_32u_C1IR", "Inf", p32,
                  [](Npp32u *s, int ss, NppiSize r, Npp8u *b, NppiConnectedRegion *reg,
                     NppStreamContext c) {
                    return nppiFloodFillGradient_32u_C1IR_Ctx(s, ss, NppiPoint{13, 8}, 30, 30, 5,
                                                              nppiNormInf, r, reg, b, c);
                  },
                  ValueKind::U32);
  const std::vector<Npp8u> c3 = picture_c3();
  case_c3<Npp8u>("nppiFloodFillGradient_8u_C3IR", "L1", c3,
                 [](Npp8u *s, int ss, NppiSize r, Npp8u *b, NppiConnectedRegion *reg,
                    NppStreamContext c) {
                   Npp8u mn[3] = {20, 20, 20}, mx[3] = {45, 45, 45};
                   const Npp8u nv[3] = {3, 4, 5};
                   return nppiFloodFillGradient_8u_C3IR_Ctx(s, ss, NppiPoint{3, 15}, mn, mx, nv,
                                                            nppiNormL1, r, reg, b, c);
                 },
                 ValueKind::U8);
}

TEST(NppiFloodFillDiff, RangeBoundary) {
  const std::vector<Npp8u> p8 = picture_t<Npp8u>();
  const std::vector<Npp16u> p16 = picture_t<Npp16u>();
  case_c1<Npp8u>("nppiFloodFillRangeBoundary_8u_C1IR", "L1", p8,
                 [](Npp8u *s, int ss, NppiSize r, Npp8u *b, NppiConnectedRegion *reg,
                    NppStreamContext c) {
                   return nppiFloodFillRangeBoundary_8u_C1IR_Ctx(s, ss, NppiPoint{1, 1}, 40, 120,
                                                                 33, 200, nppiNormL1, r, reg, b, c);
                 },
                 ValueKind::U8);
  case_c1<Npp16u>("nppiFloodFillRangeBoundary_16u_C1IR", "Inf", p16,
                  [](Npp16u *s, int ss, NppiSize r, Npp8u *b, NppiConnectedRegion *reg,
                     NppStreamContext c) {
                    return nppiFloodFillRangeBoundary_16u_C1IR_Ctx(s, ss, NppiPoint{13, 8}, 8000,
                                                                   9000, 33, 30000, nppiNormInf, r,
                                                                   reg, b, c);
                  },
                  ValueKind::U16);
  const std::vector<Npp8u> c3 = picture_c3();
  case_c3<Npp8u>("nppiFloodFillRangeBoundary_8u_C3IR", "L1", c3,
                 [](Npp8u *s, int ss, NppiSize r, Npp8u *b, NppiConnectedRegion *reg,
                    NppStreamContext c) {
                   Npp8u mn[3] = {40, 45, 49}, mx[3] = {120, 125, 129};
                   const Npp8u nv[3] = {3, 4, 5}, bv[3] = {90, 91, 92};
                   return nppiFloodFillRangeBoundary_8u_C3IR_Ctx(s, ss, NppiPoint{3, 15}, mn, mx,
                                                                  nv, bv, nppiNormL1, r, reg, b, c);
                 },
                 ValueKind::U8);
}

TEST(NppiFloodFillDiff, GradientBoundary) {
  const std::vector<Npp8u> p8 = picture_t<Npp8u>();
  const std::vector<Npp32u> p32 = picture_t<Npp32u>();
  case_c1<Npp8u>("nppiFloodFillGradientBoundary_8u_C1IR", "L1", p8,
                 [](Npp8u *s, int ss, NppiSize r, Npp8u *b, NppiConnectedRegion *reg,
                    NppStreamContext c) {
                   return nppiFloodFillGradientBoundary_8u_C1IR_Ctx(s, ss, NppiPoint{1, 1}, 20, 40,
                                                                    33, 200, nppiNormL1, r, reg, b,
                                                                    c);
                 },
                 ValueKind::U8);
  case_c1<Npp32u>("nppiFloodFillGradientBoundary_32u_C1IR", "Inf", p32,
                  [](Npp32u *s, int ss, NppiSize r, Npp8u *b, NppiConnectedRegion *reg,
                     NppStreamContext c) {
                    return nppiFloodFillGradientBoundary_32u_C1IR_Ctx(s, ss, NppiPoint{13, 8}, 30,
                                                                      30, 33, 900, nppiNormInf, r,
                                                                      reg, b, c);
                  },
                  ValueKind::U32);
  const std::vector<Npp8u> c3 = picture_c3();
  case_c3<Npp8u>("nppiFloodFillGradientBoundary_8u_C3IR", "L1", c3,
                 [](Npp8u *s, int ss, NppiSize r, Npp8u *b, NppiConnectedRegion *reg,
                    NppStreamContext c) {
                   Npp8u mn[3] = {20, 20, 20}, mx[3] = {45, 45, 45};
                   const Npp8u nv[3] = {3, 4, 5}, bv[3] = {90, 91, 92};
                   return nppiFloodFillGradientBoundary_8u_C3IR_Ctx(s, ss, NppiPoint{3, 15}, mn, mx,
                                                                    nv, bv, nppiNormL1, r, reg, b, c);
                 },
                 ValueKind::U8);
}

TEST(NppiFloodFillDiff, Errors) {
  std::vector<Npp8u> p = picture_t<Npp8u>();
  DeviceMemory<Npp8u> &d = arena<Npp8u>(p.size());
  DeviceMemory<Npp8u> &buf = arena<Npp8u>(1 << 16);
  d.copyFromHost(p);
  const NppStreamContext ctx = current_ctx();
  Npp8u *src = d.get();
  Npp8u *b = buf.get();
  const NppiSize roi{W, H};

  only_status("nppiFloodFill_8u_C1IR", "null-src",
              [&] { return nppiFloodFill_8u_C1IR_Ctx(nullptr, W, {1, 1}, 3, nppiNormL1, roi, nullptr, b, ctx); });
  only_status("nppiFloodFill_8u_C1IR", "null-buffer",
              [&] { return nppiFloodFill_8u_C1IR_Ctx(src, W, {1, 1}, 3, nppiNormL1, roi, nullptr, nullptr, ctx); });
  only_status("nppiFloodFill_8u_C1IR", "zero-roi",
              [&] { return nppiFloodFill_8u_C1IR_Ctx(src, W, {1, 1}, 3, nppiNormL1, {0, 0}, nullptr, b, ctx); });
  only_status("nppiFloodFill_8u_C1IR", "negative-roi",
              [&] { return nppiFloodFill_8u_C1IR_Ctx(src, W, {1, 1}, 3, nppiNormL1, {-2, 4}, nullptr, b, ctx); });
  only_status("nppiFloodFill_8u_C1IR", "seed-x-negative",
              [&] { return nppiFloodFill_8u_C1IR_Ctx(src, W, {-1, 1}, 3, nppiNormL1, roi, nullptr, b, ctx); });
  only_status("nppiFloodFill_8u_C1IR", "seed-y-negative",
              [&] { return nppiFloodFill_8u_C1IR_Ctx(src, W, {1, -1}, 3, nppiNormL1, roi, nullptr, b, ctx); });
  only_status("nppiFloodFill_8u_C1IR", "seed-x-eq-width",
              [&] { return nppiFloodFill_8u_C1IR_Ctx(src, W, {W, 1}, 3, nppiNormL1, roi, nullptr, b, ctx); });
  only_status("nppiFloodFill_8u_C1IR", "seed-y-eq-height",
              [&] { return nppiFloodFill_8u_C1IR_Ctx(src, W, {1, H}, 3, nppiNormL1, roi, nullptr, b, ctx); });
  only_status("nppiFloodFill_8u_C1IR", "norm-l2",
              [&] { return nppiFloodFill_8u_C1IR_Ctx(src, W, {1, 1}, 3, nppiNormL2, roi, nullptr, b, ctx); });
  only_status("nppiFloodFill_8u_C1IR", "norm-7",
              [&] { return nppiFloodFill_8u_C1IR_Ctx(src, W, {1, 1}, 3, (NppiNorm)7, roi, nullptr, b, ctx); });
  only_status("nppiFloodFill_8u_C1IR", "order-null-src-zero-roi",
              [&] { return nppiFloodFill_8u_C1IR_Ctx(nullptr, W, {1, 1}, 3, nppiNormL1, {0, 0}, nullptr, b, ctx); });
  only_status("nppiFloodFill_8u_C1IR", "order-null-buffer-zero-roi",
              [&] { return nppiFloodFill_8u_C1IR_Ctx(src, W, {1, 1}, 3, nppiNormL1, {0, 0}, nullptr, nullptr, ctx); });
  only_status("nppiFloodFill_8u_C1IR", "order-seed-before-norm",
              [&] { return nppiFloodFill_8u_C1IR_Ctx(src, W, {-1, 1}, 3, nppiNormL2, roi, nullptr, b, ctx); });
  only_status("nppiFloodFill_8u_C1IR", "order-roi-before-seed",
              [&] { return nppiFloodFill_8u_C1IR_Ctx(src, W, {-1, 1}, 3, nppiNormL1, {-2, 3}, nullptr, b, ctx); });
  only_status("nppiFloodFill_8u_C1IR", "step-zero",
              [&] { return nppiFloodFill_8u_C1IR_Ctx(src, 0, {1, 1}, 3, nppiNormL1, roi, nullptr, b, ctx); });
  only_status("nppiFloodFill_8u_C3IR", "null-src",
              [&] { return nppiFloodFill_8u_C3IR_Ctx(nullptr, 3 * W, {1, 1}, nullptr, nppiNormL1, roi, nullptr, b, ctx); });
  only_status("nppiFloodFillRange_8u_C1IR", "min-above-max",
              [&] { return nppiFloodFillRange_8u_C1IR_Ctx(src, W, {1, 1}, 9, 2, 3, nppiNormL1, roi, nullptr, b, ctx); });
  only_status("nppiFloodFillRange_8u_C1IR", "null-buffer",
              [&] { return nppiFloodFillRange_8u_C1IR_Ctx(src, W, {1, 1}, 0, 10, 3, nppiNormL1, roi, nullptr, nullptr, ctx); });
  only_status("nppiFloodFillGradient_8u_C1IR", "min-above-max-ok",
              [&] { return nppiFloodFillGradient_8u_C1IR_Ctx(src, W, {1, 1}, 9, 2, 3, nppiNormL1, roi, nullptr, b, ctx); });
  only_status("nppiFloodFillGradientBoundary_8u_C1IR", "min-above-max-ok",
              [&] { return nppiFloodFillGradientBoundary_8u_C1IR_Ctx(src, W, {1, 1}, 9, 2, 3, 0, nppiNormL1, roi, nullptr, b, ctx); });
  only_status("nppiFloodFillBoundary_8u_C1IR", "null-src",
              [&] { return nppiFloodFillBoundary_8u_C1IR_Ctx(nullptr, W, {1, 1}, 3, 9, nppiNormL1, roi, nullptr, b, ctx); });
  only_status("nppiFloodFillRangeBoundary_32u_C1IR", "null-buffer",
              [&] { return nppiFloodFillRangeBoundary_32u_C1IR_Ctx((Npp32u *)nullptr, 4 * W, {1, 1}, 0, 10, 3, 9, nppiNormL1, roi, nullptr, nullptr, ctx); });
}
