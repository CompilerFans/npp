// Differential tests for the c3 YUV/YCbCr sampling-format and layout
// conversion families (nppi_color_conversion.h):
//
//   nppiYCbCr422 {C2P3R, P3C2R}
//   nppiCbYCr422ToYCbCr422 {C2R, C2P3R}
//   nppiCbYCr422ToYCbCr411 {C2P3R}
//   nppiCbYCr422ToYCbCr420 {C2P3R, C2P2R}
//   nppiCbYCr422ToYCrCb420 {C2P3R}
//   nppiYCbCr411 {P2P3R, P3P2R}
//   nppiYCbCr411ToYCbCr420 {P2P3R, P3P2R, P3R}
//   nppiYCbCr411ToYCbCr422 {P2C2R, P2P3R, P3C2R, P3R}
//   nppiYCbCr411ToYCrCb420 {P2P3R}
//   nppiYCbCr411ToYCrCb422 {P3C2R, P3R}
//   nppiYCbCr422ToYCbCr411 {C2P2R, C2P3R, P3P2R, P3R}
//   nppiYCbCr422ToYCbCr420 {C2P2R, C2P3R, P3P2R, P3R}
//   nppiYCbCr422ToYCrCb420 {C2P3R}
//   nppiYCbCr422ToYCrCb422 {C2R, P3C2R}
//   nppiYCrCb420ToYCbCr422 {P3C2R, P3R}
//   nppiYCrCb422ToYCbCr411/420/422 {C2P3R}
//
// Measured NVIDIA NPP 13.4 semantics (RTX 5090):
//   * C2 rows are [Y0 Cb Y1 Cr]; CbYCr422 rows are [Cb Y0 Cr Y1].
//   * Subsampling never averages: 4:2:0 keeps even source rows, 4:1:1 keeps
//     every second chroma column (one chroma sample per four pixels).
//   * 4:1:1 -> 4:2:0 interleaves the two source chroma rows into one
//     destination row at x = 2*c + (y & 1).
//   * Only luma/packed strides are validated against the ROI width.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiC3YuvLayoutDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records ./diff_tests --gtest_filter='NppiC3YuvLayoutDiff.*'

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

using namespace npp_diff;
using namespace npp_functional_test;

namespace {

NppStreamContext current_ctx() {
  NppStreamContext c{};
  NppStatus st = nppGetStreamContext(&c);
  EXPECT_EQ(st, NPP_NO_ERROR);
  return c;
}

struct Planes {
  std::vector<std::unique_ptr<DeviceMemory<Npp8u>>> dev;
  std::vector<Npp8u *> ptrs;
};

Planes make_planes(const std::vector<size_t> &bytes, uint64_t seed) {
  Planes p;
  for (size_t i = 0; i < bytes.size(); ++i) {
    std::vector<Npp8u> host = make_uniform<Npp8u>(bytes[i], seed + 7 * i);
    p.dev.push_back(std::make_unique<DeviceMemory<Npp8u>>(bytes[i]));
    p.dev.back()->copyFromHost(host);
    p.ptrs.push_back(p.dev.back()->get());
  }
  return p;
}

void capture_planes(Case &c, const Planes &p, const std::vector<size_t> &bytes) {
  for (size_t i = 0; i < p.dev.size(); ++i) {
    c.device_output<Npp8u>(p.dev[i]->get(), bytes[i], ValueKind::U8);
  }
}

const Npp8u **sbuf(const Planes &p) { return const_cast<const Npp8u **>(p.ptrs.data()); }

struct Dims {
  int cw;
  int ch;
};

// Chroma plane dimensions produced by px/py decimation factors.
Dims chroma_dims(int w, int h, int px, int py) {
  return Dims{px == 2 ? w / 4 : w / 2, py == 2 ? h / 2 : h};
}

// ---------- packed C2 -> planar P3 ----------
void c2_p3(const char *fname, const std::string &key, int w, int h, int px, int py,
           NppStatus (*f)(const Npp8u *, int, Npp8u **, int *, NppiSize, NppStreamContext), uint64_t seed) {
  const int sStep = 2 * w + 6;
  const Dims c = chroma_dims(w, h, px, py);
  const int cyStep = w + 5;
  const int ccStep = std::max(c.cw, 1) + 6;
  Planes dst = make_planes({(size_t)cyStep * h, (size_t)ccStep * h, (size_t)ccStep * h}, seed + 29);
  int steps[3] = {cyStep, ccStep, ccStep};
  std::vector<Npp8u> hs = make_uniform<Npp8u>((size_t)sStep * h, seed + 11);
  DeviceMemory<Npp8u> src((size_t)sStep * h);
  src.copyFromHost(hs);
  clear_cuda_error();
  NppStatus st = f(src.get(), sStep, dst.ptrs.data(), steps, NppiSize{w, h}, current_ctx());
  Case cs("nppiC3YuvLayout/" + key, fname);
  cs.status(st);
  cs.device_output<Npp8u>(dst.dev[0]->get(), (size_t)cyStep * h, ValueKind::U8);
  cs.device_output<Npp8u>(dst.dev[1]->get(), (size_t)ccStep * c.ch, ValueKind::U8);
  cs.device_output<Npp8u>(dst.dev[2]->get(), (size_t)ccStep * c.ch, ValueKind::U8);
  cs.finish();
}

// ---------- packed C2 -> planar Y + interleaved CbCr ----------
void c2_p2(const char *fname, const std::string &key, int w, int h, int px, int py,
           NppStatus (*f)(const Npp8u *, int, Npp8u *, int, Npp8u *, int, NppiSize, NppStreamContext), uint64_t seed) {
  const int sStep = 2 * w + 6;
  const Dims c = chroma_dims(w, h, px, py);
  const int dyStep = w + 5;
  const int dcStep = 2 * std::max(c.cw, 1) + 6;
  std::vector<Npp8u> hs = make_uniform<Npp8u>((size_t)sStep * h, seed + 11);
  DeviceMemory<Npp8u> src((size_t)sStep * h);
  src.copyFromHost(hs);
  DeviceMemory<Npp8u> dy((size_t)dyStep * h);
  dy.copyFromHost(make_uniform<Npp8u>((size_t)dyStep * h, seed + 29));
  DeviceMemory<Npp8u> dc((size_t)dcStep * h);
  dc.copyFromHost(make_uniform<Npp8u>((size_t)dcStep * h, seed + 31));
  clear_cuda_error();
  NppStatus st = f(src.get(), sStep, dy.get(), dyStep, dc.get(), dcStep, NppiSize{w, h}, current_ctx());
  Case cs("nppiC3YuvLayout/" + key, fname);
  cs.status(st);
  cs.device_output<Npp8u>(dy.get(), (size_t)dyStep * h, ValueKind::U8);
  cs.device_output<Npp8u>(dc.get(), (size_t)dcStep * c.ch, ValueKind::U8);
  cs.finish();
}

// ---------- planar P3 -> packed C2 ----------
// srcMode: 0 = 4:2:2 source chroma plane, 1 = 4:1:1, 2 = 4:2:0.
void p3_c2(const char *fname, const std::string &key, int w, int h, int srcMode,
           NppStatus (*f)(const Npp8u *const *, int *, Npp8u *, int, NppiSize, NppStreamContext), uint64_t seed) {
  const int yStep = w + 5;
  const int cw = srcMode == 1 ? w / 4 : w / 2;
  const int ch = srcMode == 2 ? h / 2 : h;
  const int cStep = std::max(cw, 1) + 6;
  const int dStep = 2 * w + 6;
  Planes src = make_planes({(size_t)yStep * h, (size_t)cStep * ch, (size_t)cStep * ch}, seed + 11);
  std::vector<Npp8u> hd = make_uniform<Npp8u>((size_t)dStep * h, seed + 29);
  DeviceMemory<Npp8u> dst((size_t)dStep * h);
  dst.copyFromHost(hd);
  int steps[3] = {yStep, cStep, cStep};
  clear_cuda_error();
  NppStatus st = f(sbuf(src), steps, dst.get(), dStep, NppiSize{w, h}, current_ctx());
  Case cs("nppiC3YuvLayout/" + key, fname);
  cs.status(st);
  cs.device_output<Npp8u>(dst.get(), (size_t)dStep * h, ValueKind::U8);
  cs.finish();
}

// ---------- planar Y + interleaved CbCr 4:1:1 -> packed C2 ----------
void p2_c2_411(const char *fname, const std::string &key, int w, int h,
               NppStatus (*f)(const Npp8u *, int, const Npp8u *, int, Npp8u *, int, NppiSize, NppStreamContext),
               uint64_t seed) {
  const int yStep = w + 5;
  const int cStep = std::max(w / 2, 1) + 6;
  const int dStep = 2 * w + 6;
  DeviceMemory<Npp8u> sy((size_t)yStep * h);
  sy.copyFromHost(make_uniform<Npp8u>((size_t)yStep * h, seed + 11));
  DeviceMemory<Npp8u> sc((size_t)cStep * h);
  sc.copyFromHost(make_uniform<Npp8u>((size_t)cStep * h, seed + 13));
  DeviceMemory<Npp8u> dst((size_t)dStep * h);
  dst.copyFromHost(make_uniform<Npp8u>((size_t)dStep * h, seed + 29));
  clear_cuda_error();
  NppStatus st = f(sy.get(), yStep, sc.get(), cStep, dst.get(), dStep, NppiSize{w, h}, current_ctx());
  Case cs("nppiC3YuvLayout/" + key, fname);
  cs.status(st);
  cs.device_output<Npp8u>(dst.get(), (size_t)dStep * h, ValueKind::U8);
  cs.finish();
}

// ---------- planar Y + interleaved CbCr -> planar P3 ----------
void p2_p3(const char *fname, const std::string &key, int w, int h, int outCw, int outCh,
           NppStatus (*f)(const Npp8u *, int, const Npp8u *, int, Npp8u **, int *, NppiSize, NppStreamContext),
           uint64_t seed) {
  const int yStep = w + 5;
  const int cStep = std::max(w / 2, 1) + 6;
  const int dyStep = w + 5;
  const int dcStep = std::max(outCw, 1) + 6;
  DeviceMemory<Npp8u> sy((size_t)yStep * h);
  sy.copyFromHost(make_uniform<Npp8u>((size_t)yStep * h, seed + 11));
  DeviceMemory<Npp8u> sc((size_t)cStep * h);
  sc.copyFromHost(make_uniform<Npp8u>((size_t)cStep * h, seed + 13));
  Planes dst = make_planes({(size_t)dyStep * h, (size_t)dcStep * h, (size_t)dcStep * h}, seed + 29);
  int steps[3] = {dyStep, dcStep, dcStep};
  clear_cuda_error();
  NppStatus st = f(sy.get(), yStep, sc.get(), cStep, dst.ptrs.data(), steps, NppiSize{w, h}, current_ctx());
  Case cs("nppiC3YuvLayout/" + key, fname);
  cs.status(st);
  cs.device_output<Npp8u>(dst.dev[0]->get(), (size_t)dyStep * h, ValueKind::U8);
  cs.device_output<Npp8u>(dst.dev[1]->get(), (size_t)dcStep * outCh, ValueKind::U8);
  cs.device_output<Npp8u>(dst.dev[2]->get(), (size_t)dcStep * outCh, ValueKind::U8);
  cs.finish();
}

// ---------- planar -> planar chroma resampling ----------
void p3_p3(const char *fname, const std::string &key, int w, int h, int srcCw, int srcCh, int dstCw, int dstCh,
           NppStatus (*f)(const Npp8u *const *, int *, Npp8u **, int *, NppiSize, NppStreamContext), uint64_t seed) {
  const int yStep = w + 5;
  const int scStep = std::max(srcCw, 1) + 6;
  const int dcStep = std::max(dstCw, 1) + 6;
  Planes src = make_planes({(size_t)yStep * h, (size_t)scStep * srcCh, (size_t)scStep * srcCh}, seed + 11);
  Planes dst = make_planes({(size_t)yStep * h, (size_t)dcStep * dstCh, (size_t)dcStep * dstCh}, seed + 29);
  int ssteps[3] = {yStep, scStep, scStep};
  int dsteps[3] = {yStep, dcStep, dcStep};
  clear_cuda_error();
  NppStatus st = f(sbuf(src), ssteps, dst.ptrs.data(), dsteps, NppiSize{w, h}, current_ctx());
  Case cs("nppiC3YuvLayout/" + key, fname);
  cs.status(st);
  cs.device_output<Npp8u>(dst.dev[0]->get(), (size_t)yStep * h, ValueKind::U8);
  cs.device_output<Npp8u>(dst.dev[1]->get(), (size_t)dcStep * dstCh, ValueKind::U8);
  cs.device_output<Npp8u>(dst.dev[2]->get(), (size_t)dcStep * dstCh, ValueKind::U8);
  cs.finish();
}

// ---------- planar -> planar Y + interleaved CbCr ----------
void p3_p2(const char *fname, const std::string &key, int w, int h, int srcCw, int srcCh, int dstCh,
           NppStatus (*f)(const Npp8u *const *, int *, Npp8u *, int, Npp8u *, int, NppiSize, NppStreamContext),
           uint64_t seed) {
  const int yStep = w + 5;
  const int scStep = std::max(srcCw, 1) + 6;
  const int dcStep = w + 8; // 4:1:1 -> 4:2:0 writes a full-width interleaved row
  Planes src = make_planes({(size_t)yStep * h, (size_t)scStep * srcCh, (size_t)scStep * srcCh}, seed + 11);
  DeviceMemory<Npp8u> dy((size_t)yStep * h);
  dy.copyFromHost(make_uniform<Npp8u>((size_t)yStep * h, seed + 29));
  DeviceMemory<Npp8u> dc((size_t)dcStep * h);
  dc.copyFromHost(make_uniform<Npp8u>((size_t)dcStep * h, seed + 31));
  int ssteps[3] = {yStep, scStep, scStep};
  clear_cuda_error();
  NppStatus st = f(sbuf(src), ssteps, dy.get(), yStep, dc.get(), dcStep, NppiSize{w, h}, current_ctx());
  Case cs("nppiC3YuvLayout/" + key, fname);
  cs.status(st);
  cs.device_output<Npp8u>(dy.get(), (size_t)yStep * h, ValueKind::U8);
  cs.device_output<Npp8u>(dc.get(), (size_t)dcStep * dstCh, ValueKind::U8);
  cs.finish();
}

// ---------- packed C2 reorder ----------
void c2_reorder(const char *fname, const std::string &key, int w, int h,
                NppStatus (*f)(const Npp8u *, int, Npp8u *, int, NppiSize, NppStreamContext), uint64_t seed) {
  const int sStep = 2 * w + 6;
  const int dStep = 2 * w + 8;
  DeviceMemory<Npp8u> src((size_t)sStep * h);
  src.copyFromHost(make_uniform<Npp8u>((size_t)sStep * h, seed + 11));
  DeviceMemory<Npp8u> dst((size_t)dStep * h);
  dst.copyFromHost(make_uniform<Npp8u>((size_t)dStep * h, seed + 29));
  clear_cuda_error();
  NppStatus st = f(src.get(), sStep, dst.get(), dStep, NppiSize{w, h}, current_ctx());
  Case cs("nppiC3YuvLayout/" + key, fname);
  cs.status(st);
  cs.device_output<Npp8u>(dst.get(), (size_t)dStep * h, ValueKind::U8);
  cs.finish();
}

} // namespace

TEST(NppiC3YuvLayoutDiff, YCbCr422) {
  c2_p3("nppiYCbCr422_8u_C2P3R", "ycbcr422/c2p3/17x13", 17, 13, 1, 1, nppiYCbCr422_8u_C2P3R_Ctx, 0xC3A01);
  c2_p3("nppiYCbCr422_8u_C2P3R", "ycbcr422/c2p3/8x6", 8, 6, 1, 1, nppiYCbCr422_8u_C2P3R_Ctx, 0xC3A02);
  p3_c2("nppiYCbCr422_8u_P3C2R", "ycbcr422/p3c2/17x13", 17, 13, 0, nppiYCbCr422_8u_P3C2R_Ctx, 0xC3A03);
  p3_c2("nppiYCbCr422_8u_P3C2R", "ycbcr422/p3c2/8x6", 8, 6, 0, nppiYCbCr422_8u_P3C2R_Ctx, 0xC3A04);
}

TEST(NppiC3YuvLayoutDiff, CbYCr422ToYCbCr422) {
  c2_reorder("nppiCbYCr422ToYCbCr422_8u_C2R", "cbycr2ycbcr/c2/17x13", 17, 13, nppiCbYCr422ToYCbCr422_8u_C2R_Ctx, 0xC3A05);
  c2_reorder("nppiCbYCr422ToYCbCr422_8u_C2R", "cbycr2ycbcr/c2/8x6", 8, 6, nppiCbYCr422ToYCbCr422_8u_C2R_Ctx, 0xC3A06);
  c2_p3("nppiCbYCr422ToYCbCr422_8u_C2P3R", "cbycr2ycbcr/c2p3/17x13", 17, 13, 1, 1,
        nppiCbYCr422ToYCbCr422_8u_C2P3R_Ctx, 0xC3A07);
  c2_p3("nppiCbYCr422ToYCbCr422_8u_C2P3R", "cbycr2ycbcr/c2p3/8x6", 8, 6, 1, 1,
        nppiCbYCr422ToYCbCr422_8u_C2P3R_Ctx, 0xC3A08);
}

TEST(NppiC3YuvLayoutDiff, CbYCr422To411And420) {
  c2_p3("nppiCbYCr422ToYCbCr411_8u_C2P3R", "cbycr2ycbcr411/c2p3/16x12", 16, 12, 2, 1,
        nppiCbYCr422ToYCbCr411_8u_C2P3R_Ctx, 0xC3A09);
  c2_p3("nppiCbYCr422ToYCbCr420_8u_C2P3R", "cbycr2ycbcr420/c2p3/16x12", 16, 12, 1, 2,
        nppiCbYCr422ToYCbCr420_8u_C2P3R_Ctx, 0xC3A0A);
  c2_p2("nppiCbYCr422ToYCbCr420_8u_C2P2R", "cbycr2ycbcr420/c2p2/16x12", 16, 12, 1, 2,
        nppiCbYCr422ToYCbCr420_8u_C2P2R_Ctx, 0xC3A0B);
  c2_p3("nppiCbYCr422ToYCrCb420_8u_C2P3R", "cbycr2ycrcb420/c2p3/16x12", 16, 12, 1, 2,
        nppiCbYCr422ToYCrCb420_8u_C2P3R_Ctx, 0xC3A0C);
}

TEST(NppiC3YuvLayoutDiff, YCbCr411) {
  p2_p3("nppiYCbCr411_8u_P2P3R", "ycbcr411/p2p3/16x12", 16, 12, 4, 12, nppiYCbCr411_8u_P2P3R_Ctx, 0xC3A0D);
  p3_p2("nppiYCbCr411_8u_P3P2R", "ycbcr411/p3p2/16x12", 16, 12, 4, 12, 12, nppiYCbCr411_8u_P3P2R_Ctx, 0xC3A0E);
}

TEST(NppiC3YuvLayoutDiff, YCbCr411To420) {
  p2_p3("nppiYCbCr411ToYCbCr420_8u_P2P3R", "ycbcr411to420/p2p3/16x12", 16, 12, 8, 6,
        nppiYCbCr411ToYCbCr420_8u_P2P3R_Ctx, 0xC3A0F);
  p3_p2("nppiYCbCr411ToYCbCr420_8u_P3P2R", "ycbcr411to420/p3p2/16x12", 16, 12, 4, 12, 6,
        nppiYCbCr411ToYCbCr420_8u_P3P2R_Ctx, 0xC3A10);
  p3_p3("nppiYCbCr411ToYCbCr420_8u_P3R", "ycbcr411to420/p3/16x12", 16, 12, 4, 12, 8, 6,
        nppiYCbCr411ToYCbCr420_8u_P3R_Ctx, 0xC3A11);
}

TEST(NppiC3YuvLayoutDiff, YCbCr411To422) {
  p2_c2_411("nppiYCbCr411ToYCbCr422_8u_P2C2R", "ycbcr411to422/p2c2/16x12", 16, 12,
            nppiYCbCr411ToYCbCr422_8u_P2C2R_Ctx, 0xC3A12);
  p2_p3("nppiYCbCr411ToYCbCr422_8u_P2P3R", "ycbcr411to422/p2p3/16x12", 16, 12, 8, 12,
        nppiYCbCr411ToYCbCr422_8u_P2P3R_Ctx, 0xC3A13);
  p3_c2("nppiYCbCr411ToYCbCr422_8u_P3C2R", "ycbcr411to422/p3c2/16x12", 16, 12, 1,
        nppiYCbCr411ToYCbCr422_8u_P3C2R_Ctx, 0xC3A14);
  p3_p3("nppiYCbCr411ToYCbCr422_8u_P3R", "ycbcr411to422/p3/16x12", 16, 12, 4, 12, 8, 12,
        nppiYCbCr411ToYCbCr422_8u_P3R_Ctx, 0xC3A15);
}

TEST(NppiC3YuvLayoutDiff, YCbCr411ToYCrCb) {
  p2_p3("nppiYCbCr411ToYCrCb420_8u_P2P3R", "ycbcr411toycrcb420/p2p3/16x12", 16, 12, 8, 6,
        nppiYCbCr411ToYCrCb420_8u_P2P3R_Ctx, 0xC3A16);
  p3_c2("nppiYCbCr411ToYCrCb422_8u_P3C2R", "ycbcr411toycrcb422/p3c2/16x12", 16, 12, 1,
        nppiYCbCr411ToYCrCb422_8u_P3C2R_Ctx, 0xC3A17);
  p3_p3("nppiYCbCr411ToYCrCb422_8u_P3R", "ycbcr411toycrcb422/p3/16x12", 16, 12, 4, 12, 8, 12,
        nppiYCbCr411ToYCrCb422_8u_P3R_Ctx, 0xC3A18);
}

TEST(NppiC3YuvLayoutDiff, YCbCr422To411) {
  c2_p2("nppiYCbCr422ToYCbCr411_8u_C2P2R", "ycbcr422to411/c2p2/16x12", 16, 12, 2, 1,
        nppiYCbCr422ToYCbCr411_8u_C2P2R_Ctx, 0xC3A19);
  c2_p3("nppiYCbCr422ToYCbCr411_8u_C2P3R", "ycbcr422to411/c2p3/16x12", 16, 12, 2, 1,
        nppiYCbCr422ToYCbCr411_8u_C2P3R_Ctx, 0xC3A1A);
  p3_p3("nppiYCbCr422ToYCbCr411_8u_P3R", "ycbcr422to411/p3/16x12", 16, 12, 8, 12, 4, 12,
        nppiYCbCr422ToYCbCr411_8u_P3R_Ctx, 0xC3A1B);
  p3_p2("nppiYCbCr422ToYCbCr411_8u_P3P2R", "ycbcr422to411/p3p2/16x12", 16, 12, 8, 12, 12,
        nppiYCbCr422ToYCbCr411_8u_P3P2R_Ctx, 0xC3A1C);
}

TEST(NppiC3YuvLayoutDiff, YCbCr422To420) {
  c2_p2("nppiYCbCr422ToYCbCr420_8u_C2P2R", "ycbcr422to420/c2p2/16x12", 16, 12, 1, 2,
        nppiYCbCr422ToYCbCr420_8u_C2P2R_Ctx, 0xC3A1D);
  c2_p3("nppiYCbCr422ToYCbCr420_8u_C2P3R", "ycbcr422to420/c2p3/17x13", 17, 13, 1, 2,
        nppiYCbCr422ToYCbCr420_8u_C2P3R_Ctx, 0xC3A1E);
  c2_p3("nppiYCbCr422ToYCbCr420_8u_C2P3R", "ycbcr422to420/c2p3/16x12", 16, 12, 1, 2,
        nppiYCbCr422ToYCbCr420_8u_C2P3R_Ctx, 0xC3A1F);
  p3_p3("nppiYCbCr422ToYCbCr420_8u_P3R", "ycbcr422to420/p3/17x13", 17, 13, 8, 13, 8, 6,
        nppiYCbCr422ToYCbCr420_8u_P3R_Ctx, 0xC3A20);
  p3_p2("nppiYCbCr422ToYCbCr420_8u_P3P2R", "ycbcr422to420/p3p2/16x12", 16, 12, 8, 12, 6,
        nppiYCbCr422ToYCbCr420_8u_P3P2R_Ctx, 0xC3A21);
}

TEST(NppiC3YuvLayoutDiff, YCbCr422ToYCrCb422And420) {
  c2_p3("nppiYCbCr422ToYCrCb420_8u_C2P3R", "ycbcr422toycrcb420/c2p3/16x12", 16, 12, 1, 2,
        nppiYCbCr422ToYCrCb420_8u_C2P3R_Ctx, 0xC3A22);
  c2_reorder("nppiYCbCr422ToYCrCb422_8u_C2R", "ycbcr422toycrcb422/c2/17x13", 17, 13,
             nppiYCbCr422ToYCrCb422_8u_C2R_Ctx, 0xC3A23);
  p3_c2("nppiYCbCr422ToYCrCb422_8u_P3C2R", "ycbcr422toycrcb422/p3c2/16x12", 16, 12, 0,
        nppiYCbCr422ToYCrCb422_8u_P3C2R_Ctx, 0xC3A24);
}

TEST(NppiC3YuvLayoutDiff, YCrCb420ToYCbCr422) {
  p3_c2("nppiYCrCb420ToYCbCr422_8u_P3C2R", "ycrcb420to422/p3c2/16x12", 16, 12, 2,
        nppiYCrCb420ToYCbCr422_8u_P3C2R_Ctx, 0xC3A25);
  p3_p3("nppiYCrCb420ToYCbCr422_8u_P3R", "ycrcb420to422/p3/16x12", 16, 12, 8, 6, 8, 12,
        nppiYCrCb420ToYCbCr422_8u_P3R_Ctx, 0xC3A26);
}

TEST(NppiC3YuvLayoutDiff, YCrCb422ToYCbCr) {
  c2_p3("nppiYCrCb422ToYCbCr411_8u_C2P3R", "ycrcb422to411/c2p3/16x12", 16, 12, 2, 1,
        nppiYCrCb422ToYCbCr411_8u_C2P3R_Ctx, 0xC3A27);
  c2_p3("nppiYCrCb422ToYCbCr420_8u_C2P3R", "ycrcb422to420/c2p3/16x12", 16, 12, 1, 2,
        nppiYCrCb422ToYCbCr420_8u_C2P3R_Ctx, 0xC3A28);
  c2_p3("nppiYCrCb422ToYCbCr422_8u_C2P3R", "ycrcb422to422/c2p3/17x13", 17, 13, 1, 1,
        nppiYCrCb422ToYCbCr422_8u_C2P3R_Ctx, 0xC3A29);
}

TEST(NppiC3YuvLayoutDiff, ErrorAndProbePaths) {
  const int w = 16, h = 12;
  const int yStep = w + 5, cStep = w / 2 + 6, dStep = 2 * w + 8;
  Planes p3 = make_planes({(size_t)(w + 5) * h, (size_t)cStep * h, (size_t)cStep * h}, 0xC3B00);
  DeviceMemory<Npp8u> c2((size_t)(2 * w + 8) * h);
  c2.copyFromHost(make_uniform<Npp8u>((size_t)(2 * w + 8) * h, 0xC3B01));
  DeviceMemory<Npp8u> packed((size_t)dStep * h);
  packed.copyFromHost(make_uniform<Npp8u>((size_t)dStep * h, 0xC3B02));
  DeviceMemory<Npp8u> yPlane((size_t)yStep * h);
  yPlane.copyFromHost(make_uniform<Npp8u>((size_t)yStep * h, 0xC3B03));
  DeviceMemory<Npp8u> cPlane((size_t)cStep * h);
  cPlane.copyFromHost(make_uniform<Npp8u>((size_t)cStep * h, 0xC3B04));
  int steps[3] = {yStep, cStep, cStep};
  NppStreamContext ctx = current_ctx();
  auto probe = [&](const std::string &tag, const char *fname, auto &&call) {
    clear_cuda_error();
    NppStatus st = call();
    Case c("nppiC3YuvLayout/err/" + tag, fname);
    c.status(st);
    c.finish();
  };
  // C2 -> P3: null source, ROI and step ordering, negative step.
  probe("c2p3_null_src", "nppiYCbCr422_8u_C2P3R",
        [&] { return nppiYCbCr422_8u_C2P3R_Ctx(nullptr, 2 * w + 8, p3.ptrs.data(), steps, NppiSize{w, h}, ctx); });
  probe("c2p3_neg_roi", "nppiYCbCr422_8u_C2P3R",
        [&] { return nppiYCbCr422_8u_C2P3R_Ctx(c2.get(), 2 * w + 8, p3.ptrs.data(), steps, NppiSize{-2, h}, ctx); });
  probe("c2p3_zero_roi", "nppiYCbCr422_8u_C2P3R",
        [&] { return nppiYCbCr422_8u_C2P3R_Ctx(c2.get(), 2 * w + 8, p3.ptrs.data(), steps, NppiSize{0, 0}, ctx); });
  probe("c2p3_small_src_step", "nppiYCbCr422_8u_C2P3R",
        [&] { return nppiYCbCr422_8u_C2P3R_Ctx(c2.get(), 2 * w - 1, p3.ptrs.data(), steps, NppiSize{w, h}, ctx); });
  probe("c2p3_neg_src_step", "nppiYCbCr422_8u_C2P3R",
        [&] { return nppiYCbCr422_8u_C2P3R_Ctx(c2.get(), -2 * w, p3.ptrs.data(), steps, NppiSize{w, h}, ctx); });
  // C2 -> P2: null luma destination is checked before the step.
  probe("c2p2_null_y", "nppiYCbCr422ToYCbCr420_8u_C2P2R",
        [&] { return nppiYCbCr422ToYCbCr420_8u_C2P2R_Ctx(c2.get(), 2 * w - 1, nullptr, yStep, cPlane.get(), cStep,
                                                        NppiSize{w, h}, ctx); });
  probe("c2p2_small_src_step", "nppiYCbCr422ToYCbCr420_8u_C2P2R",
        [&] { return nppiYCbCr422ToYCbCr420_8u_C2P2R_Ctx(c2.get(), 2 * w - 1, yPlane.get(), yStep, cPlane.get(), cStep,
                                                        NppiSize{w, h}, ctx); });
  probe("c2p2_small_y_step", "nppiYCbCr422ToYCbCr420_8u_C2P2R",
        [&] { return nppiYCbCr422ToYCbCr420_8u_C2P2R_Ctx(c2.get(), 2 * w + 8, yPlane.get(), w - 1, cPlane.get(), cStep,
                                                        NppiSize{w, h}, ctx); });
  // P3 -> C2: destination stride is validated before the source pointers.
  probe("p3c2_small_dst_step", "nppiYCbCr422_8u_P3C2R",
        [&] { return nppiYCbCr422_8u_P3C2R_Ctx(sbuf(p3), steps, packed.get(), 2 * w - 1, NppiSize{w, h}, ctx); });
  probe("p3c2_neg_roi", "nppiYCbCr422_8u_P3C2R",
        [&] { return nppiYCbCr422_8u_P3C2R_Ctx(sbuf(p3), steps, packed.get(), dStep, NppiSize{-2, h}, ctx); });
  // P3 -> P3: luma strides are validated.
  int badY[3] = {w - 1, cStep, cStep};
  probe("p3p3_small_src_y_step", "nppiYCbCr422ToYCbCr411_8u_P3R",
        [&] { return nppiYCbCr422ToYCbCr411_8u_P3R_Ctx(sbuf(p3), badY, p3.ptrs.data(), steps, NppiSize{w, h}, ctx); });
  // P3 -> P2
  probe("p3p2_small_dst_y_step", "nppiYCbCr411_8u_P3P2R",
        [&] { return nppiYCbCr411_8u_P3P2R_Ctx(sbuf(p3), steps, yPlane.get(), w - 1, cPlane.get(), cStep,
                                              NppiSize{w, h}, ctx); });
  // P2 -> P3 / P2 -> C2
  probe("p2p3_null_luma", "nppiYCbCr411_8u_P2P3R",
        [&] { return nppiYCbCr411_8u_P2P3R_Ctx(nullptr, yStep, cPlane.get(), cStep, p3.ptrs.data(), steps,
                                               NppiSize{w, h}, ctx); });
  probe("p2c2_small_src_y_step", "nppiYCbCr411ToYCbCr422_8u_P2C2R",
        [&] { return nppiYCbCr411ToYCbCr422_8u_P2C2R_Ctx(yPlane.get(), w - 1, cPlane.get(), cStep, packed.get(), dStep,
                                                         NppiSize{w, h}, ctx); });
  probe("p2c2_small_dst_step", "nppiYCbCr411ToYCbCr422_8u_P2C2R",
        [&] { return nppiYCbCr411ToYCbCr422_8u_P2C2R_Ctx(yPlane.get(), yStep, cPlane.get(), cStep, packed.get(),
                                                        2 * w - 1, NppiSize{w, h}, ctx); });
  // C2 reorder
  probe("c2r_small_src_step", "nppiYCbCr422ToYCrCb422_8u_C2R",
        [&] { return nppiYCbCr422ToYCrCb422_8u_C2R_Ctx(c2.get(), 2 * w - 1, packed.get(), dStep, NppiSize{w, h}, ctx); });
  probe("c2r_small_dst_step", "nppiYCbCr422ToYCrCb422_8u_C2R",
        [&] { return nppiYCbCr422ToYCrCb422_8u_C2R_Ctx(c2.get(), 2 * w + 8, packed.get(), 2 * w - 1, NppiSize{w, h},
                                                      ctx); });
}
