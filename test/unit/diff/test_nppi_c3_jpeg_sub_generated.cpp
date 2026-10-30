// Differential tests for c3 JPEG 4:2:0 / 4:2:2 YCbCr conversions:
//   nppiRGBToYCbCr420_JPEG / nppiBGRToYCbCr420_JPEG (C3P3R, P3R)
//   nppiRGBToYCbCr422_JPEG / nppiBGRToYCbCr422_JPEG (C3P3R, P3R)
//   nppiYCbCr420ToRGB_JPEG / nppiYCbCr420ToBGR_JPEG (P3C3R, P3R)
//   nppiYCbCr422ToRGB_JPEG / nppiYCbCr422ToBGR_JPEG (P3C3R, P3R)
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiC3JpegSubDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records ./diff_tests --gtest_filter='NppiC3JpegSubDiff.*'

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <memory>
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

void capture_planes(npp_diff::Case &c, const Planes &p, size_t bytes) {
  for (const auto &d : p.dev) {
    c.device_output<Npp8u>(d->get(), bytes, ValueKind::U8);
  }
}

const Npp8u **sbuf(const Planes &p) { return const_cast<const Npp8u **>(p.ptrs.data()); }

// ---- forward: RGB/BGR -> 420/422 ----
void fwd_packed(const char *fname, const std::string &key, int width, int height, int yf,
                NppStatus (*f)(const Npp8u *, int, Npp8u **, int *, NppiSize, NppStreamContext), uint64_t seed) {
  const int sStep = width * 3 + 11;
  const int dStepY = width + 9;
  const int dStepC = width / 2 + 5;
  const size_t se = (size_t)sStep * height;
  const size_t deY = (size_t)dStepY * height;
  const size_t deC = (size_t)dStepC * ((height + yf - 1) / yf);
  std::vector<Npp8u> hs = make_uniform<Npp8u>(se, seed + 11);
  DeviceMemory<Npp8u> src(se);
  src.copyFromHost(hs);
  Planes dst = make_planes({deY, deC, deC}, seed + 29);
  int steps[3] = {dStepY, dStepC, dStepC};
  clear_cuda_error();
  NppStatus st = f(src.get(), sStep, dst.ptrs.data(), steps, NppiSize{width, height}, current_ctx());
  Case c("nppiC3JpegSub/" + key, fname);
  c.input(hs.data(), hs.size());
  c.status(st);
  c.device_output<Npp8u>(dst.dev[0]->get(), deY, ValueKind::U8);
  c.device_output<Npp8u>(dst.dev[1]->get(), deC, ValueKind::U8);
  c.device_output<Npp8u>(dst.dev[2]->get(), deC, ValueKind::U8);
  c.finish();
}

void fwd_planar(const char *fname, const std::string &key, int width, int height, int yf,
                NppStatus (*f)(const Npp8u *const *, int, Npp8u **, int *, NppiSize, NppStreamContext),
                uint64_t seed) {
  const int sStep = width + 13;
  const int dStepY = width + 5;
  const int dStepC = width / 2 + 7;
  const size_t se = (size_t)sStep * height;
  const size_t deY = (size_t)dStepY * height;
  const size_t deC = (size_t)dStepC * ((height + yf - 1) / yf);
  Planes src = make_planes({se, se, se}, seed + 11);
  Planes dst = make_planes({deY, deC, deC}, seed + 29);
  int steps[3] = {dStepY, dStepC, dStepC};
  clear_cuda_error();
  NppStatus st = f(sbuf(src), sStep, dst.ptrs.data(), steps, NppiSize{width, height}, current_ctx());
  Case c("nppiC3JpegSub/" + key, fname);
  c.status(st);
  c.device_output<Npp8u>(dst.dev[0]->get(), deY, ValueKind::U8);
  c.device_output<Npp8u>(dst.dev[1]->get(), deC, ValueKind::U8);
  c.device_output<Npp8u>(dst.dev[2]->get(), deC, ValueKind::U8);
  c.finish();
}

// ---- inverse: 420/422 -> RGB/BGR ----
void inv_packed(const char *fname, const std::string &key, int width, int height, int yf,
                NppStatus (*f)(const Npp8u *const *, int *, Npp8u *, int, NppiSize, NppStreamContext),
                uint64_t seed) {
  const int sStepY = width + 3;
  const int sStepC = width / 2 + 5;
  const int dStep = width * 3 + 7;
  const size_t seY = (size_t)sStepY * height;
  const size_t seC = (size_t)sStepC * ((height + yf - 1) / yf);
  const size_t de = (size_t)dStep * height;
  Planes src = make_planes({seY, seC, seC}, seed + 11);
  std::vector<Npp8u> hd = make_uniform<Npp8u>(de, seed + 29);
  DeviceMemory<Npp8u> dst(de);
  dst.copyFromHost(hd);
  int steps[3] = {sStepY, sStepC, sStepC};
  clear_cuda_error();
  NppStatus st = f(sbuf(src), steps, dst.get(), dStep, NppiSize{width, height}, current_ctx());
  Case c("nppiC3JpegSub/" + key, fname);
  c.input(hd.data(), hd.size());
  c.status(st);
  c.device_output<Npp8u>(dst.get(), de, ValueKind::U8);
  c.finish();
}

void inv_planar(const char *fname, const std::string &key, int width, int height, int yf,
                NppStatus (*f)(const Npp8u *const *, int *, Npp8u **, int, NppiSize, NppStreamContext),
                uint64_t seed) {
  const int sStepY = width + 5;
  const int sStepC = width / 2 + 9;
  const int dStep = width + 17;
  const size_t seY = (size_t)sStepY * height;
  const size_t seC = (size_t)sStepC * ((height + yf - 1) / yf);
  const size_t de = (size_t)dStep * height;
  Planes src = make_planes({seY, seC, seC}, seed + 11);
  Planes dst = make_planes({de, de, de}, seed + 29);
  int steps[3] = {sStepY, sStepC, sStepC};
  clear_cuda_error();
  NppStatus st = f(sbuf(src), steps, dst.ptrs.data(), dStep, NppiSize{width, height}, current_ctx());
  Case c("nppiC3JpegSub/" + key, fname);
  c.status(st);
  capture_planes(c, dst, de);
  c.finish();
}

// ---- error / probe paths (P3R forward, packed dst inverse) ----
struct ErrBufs {
  Planes src, dst;
  int steps[3];
  ErrBufs()
      : src(make_planes({16 * 4, 16 * 4, 16 * 4}, 0xC0FFEED)), dst(make_planes({16 * 4, 8 * 4, 8 * 4}, 0xC0FFEEE)),
        steps{16, 8, 8} {}
};

void err_fwd(const std::string &tag, const Npp8u **s, int ss, Npp8u **d, int *dsteps, NppiSize roi) {
  clear_cuda_error();
  NppStatus st = nppiRGBToYCbCr420_JPEG_8u_P3R_Ctx(s, ss, d, dsteps, roi, current_ctx());
  Case c("nppiC3JpegSub/err/fwd/" + tag, "nppiRGBToYCbCr420_JPEG_8u_P3R");
  c.status(st);
  c.finish();
}

void err_inv(const std::string &tag, int *ssteps, const Npp8u **s, Npp8u *d, int ds, NppiSize roi) {
  clear_cuda_error();
  NppStatus st = nppiYCbCr420ToRGB_JPEG_8u_P3C3R_Ctx(s, ssteps, d, ds, roi, current_ctx());
  Case c("nppiC3JpegSub/err/inv/" + tag, "nppiYCbCr420ToRGB_JPEG_8u_P3C3R");
  c.status(st);
  c.finish();
}

} // namespace

TEST(NppiC3JpegSubDiff, RgbToYCbCr420Jpeg) {
  fwd_packed("nppiRGBToYCbCr420_JPEG_8u_C3P3R", "fwd420_c3/17x13", 17, 13, 2,
             nppiRGBToYCbCr420_JPEG_8u_C3P3R_Ctx, 0xC10FF01);
  fwd_packed("nppiRGBToYCbCr420_JPEG_8u_C3P3R", "fwd420_c3/16x12", 16, 12, 2,
             nppiRGBToYCbCr420_JPEG_8u_C3P3R_Ctx, 0xC10FF02);
  fwd_planar("nppiRGBToYCbCr420_JPEG_8u_P3R", "fwd420_p3/17x13", 17, 13, 2, nppiRGBToYCbCr420_JPEG_8u_P3R_Ctx,
             0xC10FF03);
  fwd_planar("nppiRGBToYCbCr420_JPEG_8u_P3R", "fwd420_p3/6x4", 6, 4, 2, nppiRGBToYCbCr420_JPEG_8u_P3R_Ctx, 0xC10FF04);
}

TEST(NppiC3JpegSubDiff, BgrToYCbCr420Jpeg) {
  fwd_packed("nppiBGRToYCbCr420_JPEG_8u_C3P3R", "bgr/fwd420_c3/17x13", 17, 13, 2,
             nppiBGRToYCbCr420_JPEG_8u_C3P3R_Ctx, 0xC10FF05);
  fwd_planar("nppiBGRToYCbCr420_JPEG_8u_P3R", "bgr/fwd420_p3/17x13", 17, 13, 2, nppiBGRToYCbCr420_JPEG_8u_P3R_Ctx,
             0xC10FF06);
}

TEST(NppiC3JpegSubDiff, RgbToYCbCr422Jpeg) {
  fwd_packed("nppiRGBToYCbCr422_JPEG_8u_C3P3R", "fwd422_c3/17x13", 17, 13, 1,
             nppiRGBToYCbCr422_JPEG_8u_C3P3R_Ctx, 0xC10FF07);
  fwd_packed("nppiRGBToYCbCr422_JPEG_8u_C3P3R", "fwd422_c3/16x13", 16, 13, 1,
             nppiRGBToYCbCr422_JPEG_8u_C3P3R_Ctx, 0xC10FF08);
  fwd_planar("nppiRGBToYCbCr422_JPEG_8u_P3R", "fwd422_p3/17x13", 17, 13, 1, nppiRGBToYCbCr422_JPEG_8u_P3R_Ctx,
             0xC10FF09);
  fwd_planar("nppiRGBToYCbCr422_JPEG_8u_P3R", "fwd422_p3/6x4", 6, 4, 1, nppiRGBToYCbCr422_JPEG_8u_P3R_Ctx, 0xC10FF0A);
}

TEST(NppiC3JpegSubDiff, BgrToYCbCr422Jpeg) {
  fwd_packed("nppiBGRToYCbCr422_JPEG_8u_C3P3R", "bgr/fwd422_c3/17x13", 17, 13, 1,
             nppiBGRToYCbCr422_JPEG_8u_C3P3R_Ctx, 0xC10FF0B);
  fwd_planar("nppiBGRToYCbCr422_JPEG_8u_P3R", "bgr/fwd422_p3/17x13", 17, 13, 1, nppiBGRToYCbCr422_JPEG_8u_P3R_Ctx,
             0xC10FF0C);
}

TEST(NppiC3JpegSubDiff, YCbCr420ToRgbJpeg) {
  inv_packed("nppiYCbCr420ToRGB_JPEG_8u_P3C3R", "inv420_c3/17x13", 17, 13, 2,
             nppiYCbCr420ToRGB_JPEG_8u_P3C3R_Ctx, 0xC10FF0D);
  inv_packed("nppiYCbCr420ToRGB_JPEG_8u_P3C3R", "inv420_c3/16x12", 16, 12, 2,
             nppiYCbCr420ToRGB_JPEG_8u_P3C3R_Ctx, 0xC10FF0E);
  inv_planar("nppiYCbCr420ToRGB_JPEG_8u_P3R", "inv420_p3/17x13", 17, 13, 2, nppiYCbCr420ToRGB_JPEG_8u_P3R_Ctx,
             0xC10FF0F);
  inv_planar("nppiYCbCr420ToRGB_JPEG_8u_P3R", "inv420_p3/6x4", 6, 4, 2, nppiYCbCr420ToRGB_JPEG_8u_P3R_Ctx, 0xC10FF10);
}

TEST(NppiC3JpegSubDiff, YCbCr420ToBgrJpeg) {
  inv_packed("nppiYCbCr420ToBGR_JPEG_8u_P3C3R", "bgr/inv420_c3/17x13", 17, 13, 2,
             nppiYCbCr420ToBGR_JPEG_8u_P3C3R_Ctx, 0xC10FF11);
  inv_planar("nppiYCbCr420ToBGR_JPEG_8u_P3R", "bgr/inv420_p3/17x13", 17, 13, 2,
             nppiYCbCr420ToBGR_JPEG_8u_P3R_Ctx, 0xC10FF12);
}

TEST(NppiC3JpegSubDiff, YCbCr422ToRgbJpeg) {
  inv_packed("nppiYCbCr422ToRGB_JPEG_8u_P3C3R", "inv422_c3/17x13", 17, 13, 1,
             nppiYCbCr422ToRGB_JPEG_8u_P3C3R_Ctx, 0xC10FF13);
  inv_packed("nppiYCbCr422ToRGB_JPEG_8u_P3C3R", "inv422_c3/16x13", 16, 13, 1,
             nppiYCbCr422ToRGB_JPEG_8u_P3C3R_Ctx, 0xC10FF14);
  inv_planar("nppiYCbCr422ToRGB_JPEG_8u_P3R", "inv422_p3/17x13", 17, 13, 1, nppiYCbCr422ToRGB_JPEG_8u_P3R_Ctx,
             0xC10FF15);
  inv_planar("nppiYCbCr422ToRGB_JPEG_8u_P3R", "inv422_p3/6x4", 6, 4, 1, nppiYCbCr422ToRGB_JPEG_8u_P3R_Ctx, 0xC10FF16);
}

TEST(NppiC3JpegSubDiff, YCbCr422ToBgrJpeg) {
  inv_packed("nppiYCbCr422ToBGR_JPEG_8u_P3C3R", "bgr/inv422_c3/17x13", 17, 13, 1,
             nppiYCbCr422ToBGR_JPEG_8u_P3C3R_Ctx, 0xC10FF17);
  inv_planar("nppiYCbCr422ToBGR_JPEG_8u_P3R", "bgr/inv422_p3/17x13", 17, 13, 1,
             nppiYCbCr422ToBGR_JPEG_8u_P3R_Ctx, 0xC10FF18);
}

TEST(NppiC3JpegSubDiff, ErrorAndProbePaths) {
  ErrBufs b;
  static Npp8u *nullplane[3] = {nullptr, nullptr, nullptr};
  const Npp8u **nullp = const_cast<const Npp8u **>(nullplane);
  err_fwd("null_src_ptr", nullp, 16, b.dst.ptrs.data(), b.steps, NppiSize{8, 4});
  err_fwd("zero_step", sbuf(b.src), 0, b.dst.ptrs.data(), b.steps, NppiSize{8, 4});
  err_fwd("neg_step", sbuf(b.src), -4, b.dst.ptrs.data(), b.steps, NppiSize{8, 4});
  err_fwd("neg_roi", sbuf(b.src), 16, b.dst.ptrs.data(), b.steps, NppiSize{-2, 4});
  err_fwd("empty_roi", sbuf(b.src), 16, b.dst.ptrs.data(), b.steps, NppiSize{0, 0});
  err_fwd("small_step", sbuf(b.src), 2, b.dst.ptrs.data(), b.steps, NppiSize{8, 4});
  err_fwd("zero_dst_step", sbuf(b.src), 16, b.dst.ptrs.data(), b.steps, NppiSize{8, 4});
  // dst Y step boundary (width 8) and chroma step boundary (width/2 = 4)
  {
    int steps[3] = {8, 4, 4};
    err_fwd("dst_step_eq_min", sbuf(b.src), 16, b.dst.ptrs.data(), steps, NppiSize{8, 4});
    int below[3] = {8, 3, 4};
    err_fwd("chroma_step_below_min", sbuf(b.src), 16, b.dst.ptrs.data(), below, NppiSize{8, 4});
    int belowy[3] = {7, 4, 4};
    err_fwd("dst_y_step_below_min", sbuf(b.src), 16, b.dst.ptrs.data(), belowy, NppiSize{8, 4});
  }
  // odd width + bad step: does the warning win or the step error?
  {
    int below[3] = {17, 3, 3};
    err_fwd("odd_width_and_bad_chroma_step", sbuf(b.src), 17, b.dst.ptrs.data(), below, NppiSize{17, 4});
  }
  // odd width/height on 422: parity probe (16x13 -> warning or not?)
  {
    Case c("nppiC3JpegSub/probe/422_even_w_odd_h", "nppiRGBToYCbCr422_JPEG_8u_P3R");
    clear_cuda_error();
    NppStatus st = nppiRGBToYCbCr422_JPEG_8u_P3R_Ctx(sbuf(b.src), 16, b.dst.ptrs.data(), b.steps, NppiSize{16, 13},
                                                     current_ctx());
    c.status(st);
    c.finish();
  }
  {
    Case c("nppiC3JpegSub/probe/420_zero_width", "nppiRGBToYCbCr420_JPEG_8u_P3R");
    clear_cuda_error();
    NppStatus st = nppiRGBToYCbCr420_JPEG_8u_P3R_Ctx(sbuf(b.src), 16, b.dst.ptrs.data(), b.steps, NppiSize{0, 4},
                                                     current_ctx());
    c.status(st);
    c.finish();
  }
  {
    Case c("nppiC3JpegSub/probe/420_zero_height", "nppiRGBToYCbCr420_JPEG_8u_P3R");
    clear_cuda_error();
    NppStatus st = nppiRGBToYCbCr420_JPEG_8u_P3R_Ctx(sbuf(b.src), 16, b.dst.ptrs.data(), b.steps, NppiSize{8, 0},
                                                     current_ctx());
    c.status(st);
    c.finish();
  }
  // inverse probes
  DeviceMemory<Npp8u> dstbuf(16 * 4);
  std::vector<Npp8u> hd = make_uniform<Npp8u>(16 * 4, 0xC10FF19);
  dstbuf.copyFromHost(hd);
  int ssteps[3] = {16, 8, 8};
  err_inv("null_src_ptr", ssteps, nullp, dstbuf.get(), 64, NppiSize{8, 4});
  err_inv("zero_step", ssteps, sbuf(b.src), dstbuf.get(), 0, NppiSize{8, 4});
  err_inv("neg_roi", ssteps, sbuf(b.src), dstbuf.get(), 64, NppiSize{-2, 4});
  {
    int below[3] = {16, 3, 8};
    err_inv("src_chroma_step_below_min", below, sbuf(b.src), dstbuf.get(), 64, NppiSize{8, 4});
  }
}

namespace {
void probe_status(const std::string &tag, const char *fname, NppStatus st) {
  Case c("nppiC3JpegSub/probe2/" + tag, fname);
  c.status(st);
  c.finish();
}
} // namespace
TEST(NppiC3JpegSubDiff, ProbeRound2) {
  // buffers sized for ROI {16,13}: planar src 3x(16*13), packed src 48*13,
  // dst planes {16*13, 8*8, 8*8}, packed dst 64*13.
  Planes src = make_planes({16 * 13, 16 * 13, 16 * 13}, 0xC10FF1B);
  DeviceMemory<Npp8u> packed_src_dev(48 * 13);
  std::vector<Npp8u> packed_src = make_uniform<Npp8u>(48 * 13, 0xC10FF1C);
  packed_src_dev.copyFromHost(packed_src);
  Planes dst = make_planes({16 * 13, 8 * 8, 8 * 8}, 0xC10FF1D);
  DeviceMemory<Npp8u> pdst(64 * 13);
  std::vector<Npp8u> hd = make_uniform<Npp8u>(64 * 13, 0xC10FF1A);
  pdst.copyFromHost(hd);
  int st16[3] = {16, 8, 8};
  const Npp8u *ps = packed_src_dev.get();
  NppStreamContext ctx = current_ctx();
  // parity of the trailing dimension, per variant
  probe_status("fwd420_c3_16x13", "nppiRGBToYCbCr420_JPEG_8u_C3P3R",
               nppiRGBToYCbCr420_JPEG_8u_C3P3R_Ctx(ps, 48, dst.ptrs.data(), st16, NppiSize{16, 13}, ctx));
  probe_status("fwd420_p3_16x13", "nppiRGBToYCbCr420_JPEG_8u_P3R",
               nppiRGBToYCbCr420_JPEG_8u_P3R_Ctx(sbuf(src), 16, dst.ptrs.data(), st16, NppiSize{16, 13}, ctx));
  probe_status("fwd422_p3_16x13", "nppiRGBToYCbCr422_JPEG_8u_P3R",
               nppiRGBToYCbCr422_JPEG_8u_P3R_Ctx(sbuf(src), 16, dst.ptrs.data(), st16, NppiSize{16, 13}, ctx));
  probe_status("inv420_c3_16x13", "nppiYCbCr420ToRGB_JPEG_8u_P3C3R",
               nppiYCbCr420ToRGB_JPEG_8u_P3C3R_Ctx(sbuf(src), st16, pdst.get(), 64, NppiSize{16, 13}, ctx));
  probe_status("inv420_p3_16x13", "nppiYCbCr420ToRGB_JPEG_8u_P3R",
               nppiYCbCr420ToRGB_JPEG_8u_P3R_Ctx(sbuf(src), st16, dst.ptrs.data(), 16, NppiSize{16, 13}, ctx));
  probe_status("inv422_p3_16x13", "nppiYCbCr422ToRGB_JPEG_8u_P3R",
               nppiYCbCr422ToRGB_JPEG_8u_P3R_Ctx(sbuf(src), st16, dst.ptrs.data(), 16, NppiSize{16, 13}, ctx));
  probe_status("inv422_bgrp3_16x13_wide", "nppiYCbCr422ToBGR_JPEG_8u_P3R",
               nppiYCbCr422ToBGR_JPEG_8u_P3R_Ctx(sbuf(src), st16, dst.ptrs.data(), 64, NppiSize{16, 13}, ctx));
  probe_status("inv420_bgrp3_16x13_narrow", "nppiYCbCr420ToBGR_JPEG_8u_P3R",
               nppiYCbCr420ToBGR_JPEG_8u_P3R_Ctx(sbuf(src), st16, dst.ptrs.data(), 16, NppiSize{16, 13}, ctx));
  // src-plane step validation on the inverse: Y plane below width
  int sy[3] = {7, 8, 8};
  probe_status("inv420_c3_srcy_small", "nppiYCbCr420ToRGB_JPEG_8u_P3C3R",
               nppiYCbCr420ToRGB_JPEG_8u_P3C3R_Ctx(sbuf(src), sy, pdst.get(), 64, NppiSize{8, 4}, ctx));
  // dst Y step below width for the packed-source forward
  int dy[3] = {7, 4, 4};
  probe_status("fwd420_c3_dsty_small", "nppiRGBToYCbCr420_JPEG_8u_C3P3R",
               nppiRGBToYCbCr420_JPEG_8u_C3P3R_Ctx(ps, 24, dst.ptrs.data(), dy, NppiSize{8, 4}, ctx));
  probe_status("fwd420_c3_dsty_eq", "nppiRGBToYCbCr420_JPEG_8u_C3P3R",
               nppiRGBToYCbCr420_JPEG_8u_C3P3R_Ctx(ps, 24, dst.ptrs.data(), st16, NppiSize{8, 4}, ctx));
  // packed src step boundary for C3P3R (min = 3*width?)
  probe_status("fwd420_c3_src_23", "nppiRGBToYCbCr420_JPEG_8u_C3P3R",
               nppiRGBToYCbCr420_JPEG_8u_C3P3R_Ctx(ps, 23, dst.ptrs.data(), st16, NppiSize{8, 4}, ctx));
  probe_status("fwd420_c3_src_24", "nppiRGBToYCbCr420_JPEG_8u_C3P3R",
               nppiRGBToYCbCr420_JPEG_8u_C3P3R_Ctx(ps, 24, dst.ptrs.data(), st16, NppiSize{8, 4}, ctx));
  // zero chroma steps (both directions)
  int cz[3] = {16, 0, 8};
  probe_status("fwd420_p3_chroma0", "nppiRGBToYCbCr420_JPEG_8u_P3R",
               nppiRGBToYCbCr420_JPEG_8u_P3R_Ctx(sbuf(src), 16, dst.ptrs.data(), cz, NppiSize{8, 4}, ctx));
  probe_status("inv420_c3_chroma0", "nppiYCbCr420ToRGB_JPEG_8u_P3C3R",
               nppiYCbCr420ToRGB_JPEG_8u_P3C3R_Ctx(sbuf(src), cz, pdst.get(), 64, NppiSize{8, 4}, ctx));
}
