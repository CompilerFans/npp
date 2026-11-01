// Differential tests for the 4:2:2 "video" (BT.601 studio) conversions:
//   nppiRGBToYCbCr422 / nppiBGRToYCbCr422 (C3C2R, C3P3R, AC4C2R, AC4P3R, P3C2R)
//   nppiRGBToCbYCr422 / nppiRGBToCbYCr422Gamma / nppiBGRToCbYCr422 (C3C2R, AC4C2R)
//   nppiYCbCr422ToRGB (C2C3R, C2P3R, P3C3R)
//   nppiYCbCr422ToBGR / nppiCbYCr422ToBGR / nppiCbYCr422ToRGB (C2C3R, C2C4R, P3C3R)
//   nppiYCbCr422ToCbYCr422 (C2R)
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiC3Conv422Diff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records ./diff_tests --gtest_filter='NppiC3Conv422Diff.*'

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

// ---------- forward: packed source (C3 / AC4) -> packed C2 ----------
void fwd_packed_c2(const char *fname, const std::string &key, int width, int height, int srcPp,
                   NppStatus (*f)(const Npp8u *, int, Npp8u *, int, NppiSize, NppStreamContext), uint64_t seed) {
  const int sStep = width * srcPp + 12;
  const int dStep = 2 * width + 10;
  const size_t se = (size_t)sStep * height;
  const size_t de = (size_t)dStep * height;
  std::vector<Npp8u> hs = make_uniform<Npp8u>(se, seed + 11);
  DeviceMemory<Npp8u> src(se);
  src.copyFromHost(hs);
  DeviceMemory<Npp8u> dst(de);
  std::vector<Npp8u> hd = make_uniform<Npp8u>(de, seed + 29);
  dst.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st = f(src.get(), sStep, dst.get(), dStep, NppiSize{width, height}, current_ctx());
  Case c("nppiC3Conv422/" + key, fname);
  c.status(st);
  c.device_output<Npp8u>(dst.get(), de, ValueKind::U8);
  c.finish();
}

// ---------- forward: packed source -> planar 3-plane ----------
void fwd_packed_p3(const char *fname, const std::string &key, int width, int height, int srcPp,
                   NppStatus (*f)(const Npp8u *, int, Npp8u **, int *, NppiSize, NppStreamContext),
                   uint64_t seed) {
  const int sStep = width * srcPp + 12;
  const int dStepY = width + 5;
  const int dStepC = width / 2 + 7;
  const size_t se = (size_t)sStep * height;
  const size_t deY = (size_t)dStepY * height;
  const size_t deC = (size_t)dStepC * height;
  std::vector<Npp8u> hs = make_uniform<Npp8u>(se, seed + 11);
  DeviceMemory<Npp8u> src(se);
  src.copyFromHost(hs);
  Planes dst = make_planes({deY, deC, deC}, seed + 29);
  int steps[3] = {dStepY, dStepC, dStepC};
  clear_cuda_error();
  NppStatus st = f(src.get(), sStep, dst.ptrs.data(), steps, NppiSize{width, height}, current_ctx());
  Case c("nppiC3Conv422/" + key, fname);
  c.status(st);
  c.device_output<Npp8u>(dst.dev[0]->get(), deY, ValueKind::U8);
  c.device_output<Npp8u>(dst.dev[1]->get(), deC, ValueKind::U8);
  c.device_output<Npp8u>(dst.dev[2]->get(), deC, ValueKind::U8);
  c.finish();
}

// ---------- forward: planar source -> packed C2 ----------
void fwd_p3_c2(const char *fname, const std::string &key, int width, int height,
               NppStatus (*f)(const Npp8u *const *, int, Npp8u *, int, NppiSize, NppStreamContext), uint64_t seed) {
  const int sStep = width + 14;
  const int dStep = 2 * width + 10;
  const size_t se = (size_t)sStep * height;
  const size_t de = (size_t)dStep * height;
  Planes src = make_planes({se, se, se}, seed + 11);
  DeviceMemory<Npp8u> dst(de);
  std::vector<Npp8u> hd = make_uniform<Npp8u>(de, seed + 29);
  dst.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st = f(sbuf(src), sStep, dst.get(), dStep, NppiSize{width, height}, current_ctx());
  Case c("nppiC3Conv422/" + key, fname);
  c.status(st);
  c.device_output<Npp8u>(dst.get(), de, ValueKind::U8);
  c.finish();
}

// ---------- inverse: packed C2 source -> packed C3 / C4 ----------
void inv_c2_packed(const char *fname, const std::string &key, int width, int height, int dstPp,
                   NppStatus (*f)(const Npp8u *, int, Npp8u *, int, NppiSize, NppStreamContext), uint64_t seed) {
  const int sStep = 2 * width + 6;
  const int dStep = width * dstPp + 7;
  const size_t se = (size_t)sStep * height;
  const size_t de = (size_t)dStep * height;
  std::vector<Npp8u> hs = make_uniform<Npp8u>(se, seed + 11);
  DeviceMemory<Npp8u> src(se);
  src.copyFromHost(hs);
  DeviceMemory<Npp8u> dst(de);
  std::vector<Npp8u> hd = make_uniform<Npp8u>(de, seed + 29);
  dst.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st = f(src.get(), sStep, dst.get(), dStep, NppiSize{width, height}, current_ctx());
  Case c("nppiC3Conv422/" + key, fname);
  c.status(st);
  c.device_output<Npp8u>(dst.get(), de, ValueKind::U8);
  c.finish();
}

// ---------- inverse: packed C2 source -> packed C4 with alpha constant ----------
void inv_c2_c4(const char *fname, const std::string &key, int width, int height, Npp8u aval,
               NppStatus (*f)(const Npp8u *, int, Npp8u *, int, NppiSize, Npp8u, NppStreamContext), uint64_t seed) {
  const int sStep = 2 * width + 6;
  const int dStep = width * 4 + 8;
  const size_t se = (size_t)sStep * height;
  const size_t de = (size_t)dStep * height;
  std::vector<Npp8u> hs = make_uniform<Npp8u>(se, seed + 11);
  DeviceMemory<Npp8u> src(se);
  src.copyFromHost(hs);
  DeviceMemory<Npp8u> dst(de);
  std::vector<Npp8u> hd = make_uniform<Npp8u>(de, seed + 29);
  dst.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st = f(src.get(), sStep, dst.get(), dStep, NppiSize{width, height}, aval, current_ctx());
  Case c("nppiC3Conv422/" + key, fname);
  c.status(st);
  c.device_output<Npp8u>(dst.get(), de, ValueKind::U8);
  c.finish();
}

// ---------- inverse: packed C2 source -> planar ----------
void inv_c2_p3(const char *fname, const std::string &key, int width, int height,
               NppStatus (*f)(const Npp8u *, int, Npp8u **, int, NppiSize, NppStreamContext), uint64_t seed) {
  const int sStep = 2 * width + 6;
  const int dStep = width + 17;
  const size_t se = (size_t)sStep * height;
  const size_t de = (size_t)dStep * height;
  std::vector<Npp8u> hs = make_uniform<Npp8u>(se, seed + 11);
  DeviceMemory<Npp8u> src(se);
  src.copyFromHost(hs);
  Planes dst = make_planes({de, de, de}, seed + 29);
  clear_cuda_error();
  NppStatus st = f(src.get(), sStep, dst.ptrs.data(), dStep, NppiSize{width, height}, current_ctx());
  Case c("nppiC3Conv422/" + key, fname);
  c.status(st);
  capture_planes(c, dst, de);
  c.finish();
}

// ---------- inverse: planar source (step array) -> packed C3 ----------
void inv_p3_packed(const char *fname, const std::string &key, int width, int height, int dstPp,
                   NppStatus (*f)(const Npp8u *const *, int *, Npp8u *, int, NppiSize, NppStreamContext),
                   uint64_t seed) {
  const int sStepY = width + 5;
  const int sStepC = width / 2 + 10;
  const int dStep = width * dstPp + 7;
  const size_t seY = (size_t)sStepY * height;
  const size_t seC = (size_t)sStepC * height;
  const size_t de = (size_t)dStep * height;
  Planes src = make_planes({seY, seC, seC}, seed + 11);
  DeviceMemory<Npp8u> dst(de);
  std::vector<Npp8u> hd = make_uniform<Npp8u>(de, seed + 29);
  dst.copyFromHost(hd);
  int steps[3] = {sStepY, sStepC, sStepC};
  clear_cuda_error();
  NppStatus st = f(sbuf(src), steps, dst.get(), dStep, NppiSize{width, height}, current_ctx());
  Case c("nppiC3Conv422/" + key, fname);
  c.status(st);
  c.device_output<Npp8u>(dst.get(), de, ValueKind::U8);
  c.finish();
}

} // namespace

TEST(NppiC3Conv422Diff, RgbToYCbCr422) {
  fwd_packed_c2("nppiRGBToYCbCr422_8u_C3C2R", "rgb2ycbcr/c3c2/17x13", 17, 13, 3,
                nppiRGBToYCbCr422_8u_C3C2R_Ctx, 0xC20FF01);
  fwd_packed_c2("nppiRGBToYCbCr422_8u_C3C2R", "rgb2ycbcr/c3c2/16x12", 16, 12, 3,
                nppiRGBToYCbCr422_8u_C3C2R_Ctx, 0xC20FF02);
  fwd_packed_p3("nppiRGBToYCbCr422_8u_C3P3R", "rgb2ycbcr/c3p3/17x13", 17, 13, 3,
                nppiRGBToYCbCr422_8u_C3P3R_Ctx, 0xC20FF03);
  fwd_packed_p3("nppiRGBToYCbCr422_8u_C3P3R", "rgb2ycbcr/c3p3/6x4", 6, 4, 3, nppiRGBToYCbCr422_8u_C3P3R_Ctx,
                0xC20FF04);
  fwd_p3_c2("nppiRGBToYCbCr422_8u_P3C2R", "rgb2ycbcr/p3c2/17x13", 17, 13, nppiRGBToYCbCr422_8u_P3C2R_Ctx, 0xC20FF05);
  fwd_p3_c2("nppiRGBToYCbCr422_8u_P3C2R", "rgb2ycbcr/p3c2/6x4", 6, 4, nppiRGBToYCbCr422_8u_P3C2R_Ctx, 0xC20FF06);
}

TEST(NppiC3Conv422Diff, BgrToYCbCr422) {
  fwd_packed_c2("nppiBGRToYCbCr422_8u_C3C2R", "bgr2ycbcr/c3c2/17x13", 17, 13, 3,
                nppiBGRToYCbCr422_8u_C3C2R_Ctx, 0xC20FF07);
  fwd_packed_c2("nppiBGRToYCbCr422_8u_AC4C2R", "bgr2ycbcr/ac4c2/17x13", 17, 13, 4,
                nppiBGRToYCbCr422_8u_AC4C2R_Ctx, 0xC20FF08);
  fwd_packed_p3("nppiBGRToYCbCr422_8u_C3P3R", "bgr2ycbcr/c3p3/17x13", 17, 13, 3,
                nppiBGRToYCbCr422_8u_C3P3R_Ctx, 0xC20FF09);
  fwd_packed_p3("nppiBGRToYCbCr422_8u_AC4P3R", "bgr2ycbcr/ac4p3/17x13", 17, 13, 4,
                nppiBGRToYCbCr422_8u_AC4P3R_Ctx, 0xC20FF0A);
}

TEST(NppiC3Conv422Diff, RgbBgrToCbYCr422) {
  fwd_packed_c2("nppiRGBToCbYCr422_8u_C3C2R", "rgb2cbycr/c3c2/17x13", 17, 13, 3,
                nppiRGBToCbYCr422_8u_C3C2R_Ctx, 0xC20FF0B);
  fwd_packed_c2("nppiRGBToCbYCr422_8u_C3C2R", "rgb2cbycr/c3c2/16x12", 16, 12, 3,
                nppiRGBToCbYCr422_8u_C3C2R_Ctx, 0xC20FF0C);
  {
    // Gamma variant: src step chosen as a multiple of 4 (see err/gamma_* probes)
    const int width = 17, height = 13;
    const int sStep = 68, dStep = 44;
    const size_t se = (size_t)sStep * height, de = (size_t)dStep * height;
    std::vector<Npp8u> hs = make_uniform<Npp8u>(se, 0xC20FF0D + 11);
    DeviceMemory<Npp8u> src(se);
    src.copyFromHost(hs);
    DeviceMemory<Npp8u> dst(de);
    std::vector<Npp8u> hd = make_uniform<Npp8u>(de, 0xC20FF0D + 29);
    dst.copyFromHost(hd);
    clear_cuda_error();
    NppStatus st = nppiRGBToCbYCr422Gamma_8u_C3C2R_Ctx(src.get(), sStep, dst.get(), dStep,
                                                       NppiSize{width, height}, current_ctx());
    Case c("nppiC3Conv422/rgbgamma2cbycr/c3c2/17x13", "nppiRGBToCbYCr422Gamma_8u_C3C2R");
    c.status(st);
    c.device_output<Npp8u>(dst.get(), de, ValueKind::U8);
    c.finish();
  }
  fwd_packed_c2("nppiBGRToCbYCr422_8u_AC4C2R", "bgr2cbycr/ac4c2/17x13", 17, 13, 4,
                nppiBGRToCbYCr422_8u_AC4C2R_Ctx, 0xC20FF0E);
}

TEST(NppiC3Conv422Diff, YCbCr422ToRgb) {
  inv_c2_packed("nppiYCbCr422ToRGB_8u_C2C3R", "ycbcr2rgb/c2c3/17x13", 17, 13, 3,
                nppiYCbCr422ToRGB_8u_C2C3R_Ctx, 0xC20FF0F);
  inv_c2_packed("nppiYCbCr422ToRGB_8u_C2C3R", "ycbcr2rgb/c2c3/16x12", 16, 12, 3,
                nppiYCbCr422ToRGB_8u_C2C3R_Ctx, 0xC20FF10);
  inv_c2_p3("nppiYCbCr422ToRGB_8u_C2P3R", "ycbcr2rgb/c2p3/17x13", 17, 13, nppiYCbCr422ToRGB_8u_C2P3R_Ctx, 0xC20FF11);
  inv_p3_packed("nppiYCbCr422ToRGB_8u_P3C3R", "ycbcr2rgb/p3c3/17x13", 17, 13, 3,
                nppiYCbCr422ToRGB_8u_P3C3R_Ctx, 0xC20FF12);
  inv_p3_packed("nppiYCbCr422ToRGB_8u_P3C3R", "ycbcr2rgb/p3c3/6x4", 6, 4, 3, nppiYCbCr422ToRGB_8u_P3C3R_Ctx, 0xC20FF13);
}

TEST(NppiC3Conv422Diff, YCbCr422ToBgr) {
  inv_c2_packed("nppiYCbCr422ToBGR_8u_C2C3R", "ycbcr2bgr/c2c3/17x13", 17, 13, 3,
                nppiYCbCr422ToBGR_8u_C2C3R_Ctx, 0xC20FF14);
  // NOTE: odd-width C2C4R output of NPP 13.4 is not reproducible from the src
  // rows (reads land in unrelated rows), so only even width is recorded here.
  inv_c2_c4("nppiYCbCr422ToBGR_8u_C2C4R", "ycbcr2bgr/c2c4/16x12", 16, 12, 212,
            nppiYCbCr422ToBGR_8u_C2C4R_Ctx, 0xC20FF15);
  inv_p3_packed("nppiYCbCr422ToBGR_8u_P3C3R", "ycbcr2bgr/p3c3/17x13", 17, 13, 3,
                nppiYCbCr422ToBGR_8u_P3C3R_Ctx, 0xC20FF16);
}

TEST(NppiC3Conv422Diff, CbYCr422ToRgbBgr) {
  inv_c2_packed("nppiCbYCr422ToRGB_8u_C2C3R", "cbycr2rgb/c2c3/17x13", 17, 13, 3,
                nppiCbYCr422ToRGB_8u_C2C3R_Ctx, 0xC20FF17);
  inv_p3_packed("nppiYCbCr422ToBGR_8u_P3C3R", "ycbcr2bgr/p3c3/6x4", 6, 4, 3,
                nppiYCbCr422ToBGR_8u_P3C3R_Ctx, 0xC20FF18);
  inv_c2_c4("nppiCbYCr422ToBGR_8u_C2C4R", "cbycr2bgr/c2c4/16x12", 16, 12, 137,
            nppiCbYCr422ToBGR_8u_C2C4R_Ctx, 0xC20FF19);
}

TEST(NppiC3Conv422Diff, YCbCr422ToCbYCr422) {
  fwd_packed_c2("nppiYCbCr422ToCbYCr422_8u_C2R", "reorder/c2/17x13", 17, 13, 2,
                nppiYCbCr422ToCbYCr422_8u_C2R_Ctx, 0xC20FF1A);
  fwd_packed_c2("nppiYCbCr422ToCbYCr422_8u_C2R", "reorder/c2/16x12", 16, 12, 2,
                nppiYCbCr422ToCbYCr422_8u_C2R_Ctx, 0xC20FF1B);
}

TEST(NppiC3Conv422Diff, ErrorAndProbePaths) {
  // buffers sized for the largest probe (ROI up to 17x13, packed rows 68,
  // C2 rows 40, planar rows 40)
  Planes ps = make_planes({68 * 13, 68 * 13, 68 * 13}, 0xC0FFEED);
  DeviceMemory<Npp8u> c2src(40 * 13);
  std::vector<Npp8u> hc2 = make_uniform<Npp8u>(40 * 13, 0xC0FFEEE);
  c2src.copyFromHost(hc2);
  DeviceMemory<Npp8u> dstbuf(68 * 13);
  std::vector<Npp8u> hd = make_uniform<Npp8u>(68 * 13, 0xC0FFEEF);
  dstbuf.copyFromHost(hd);
  Planes pd = make_planes({40 * 13, 40 * 13, 40 * 13}, 0xC0FFEF0);
  NppStreamContext ctx = current_ctx();
  auto probe = [&](const std::string &tag, const char *fname, auto &&call) {
    clear_cuda_error(); // drop any sticky CUDA error from earlier cases
    NppStatus st = call();
    Case c("nppiC3Conv422/err/" + tag, fname);
    c.status(st);
    c.finish();
  };
  // ---- forward C3C2R: null / ROI / min step / evenness / parity ----
  probe("fwd_null_src", "nppiRGBToYCbCr422_8u_C3C2R",
        [&] { return nppiRGBToYCbCr422_8u_C3C2R_Ctx(nullptr, 64, dstbuf.get(), 42, NppiSize{16, 12}, ctx); });
  probe("fwd_null_dst", "nppiRGBToYCbCr422_8u_C3C2R",
        [&] { return nppiRGBToYCbCr422_8u_C3C2R_Ctx(ps.ptrs[0], 64, nullptr, 42, NppiSize{16, 12}, ctx); });
  probe("fwd_neg_roi", "nppiRGBToYCbCr422_8u_C3C2R",
        [&] { return nppiRGBToYCbCr422_8u_C3C2R_Ctx(ps.ptrs[0], 64, dstbuf.get(), 42, NppiSize{-2, 12}, ctx); });
  probe("fwd_zero_roi", "nppiRGBToYCbCr422_8u_C3C2R",
        [&] { return nppiRGBToYCbCr422_8u_C3C2R_Ctx(ps.ptrs[0], 64, dstbuf.get(), 42, NppiSize{0, 0}, ctx); });
  probe("fwd_small_src_step", "nppiRGBToYCbCr422_8u_C3C2R",
        [&] { return nppiRGBToYCbCr422_8u_C3C2R_Ctx(ps.ptrs[0], 47, dstbuf.get(), 42, NppiSize{16, 12}, ctx); });
  probe("fwd_src_step_eq", "nppiRGBToYCbCr422_8u_C3C2R",
        [&] { return nppiRGBToYCbCr422_8u_C3C2R_Ctx(ps.ptrs[0], 48, dstbuf.get(), 42, NppiSize{16, 12}, ctx); });
  probe("fwd_zero_dst_step", "nppiRGBToYCbCr422_8u_C3C2R",
        [&] { return nppiRGBToYCbCr422_8u_C3C2R_Ctx(ps.ptrs[0], 64, dstbuf.get(), 0, NppiSize{16, 12}, ctx); });
  probe("fwd_dst_2w_eq", "nppiRGBToYCbCr422_8u_C3C2R",
        [&] { return nppiRGBToYCbCr422_8u_C3C2R_Ctx(ps.ptrs[0], 64, dstbuf.get(), 32, NppiSize{16, 12}, ctx); });
  probe("fwd_dst_2w_m1", "nppiRGBToYCbCr422_8u_C3C2R",
        [&] { return nppiRGBToYCbCr422_8u_C3C2R_Ctx(ps.ptrs[0], 64, dstbuf.get(), 31, NppiSize{16, 12}, ctx); });
  probe("fwd_dst_odd", "nppiRGBToYCbCr422_8u_C3C2R",
        [&] { return nppiRGBToYCbCr422_8u_C3C2R_Ctx(ps.ptrs[0], 64, dstbuf.get(), 33, NppiSize{16, 12}, ctx); });
  probe("fwd_src_odd", "nppiRGBToYCbCr422_8u_C3C2R",
        [&] { return nppiRGBToYCbCr422_8u_C3C2R_Ctx(ps.ptrs[0], 49, dstbuf.get(), 42, NppiSize{16, 12}, ctx); });
  probe("fwd_neg_step", "nppiRGBToYCbCr422_8u_C3C2R",
        [&] { return nppiRGBToYCbCr422_8u_C3C2R_Ctx(ps.ptrs[0], -4, dstbuf.get(), 42, NppiSize{16, 12}, ctx); });
  probe("fwd_w17", "nppiRGBToYCbCr422_8u_C3C2R",
        [&] { return nppiRGBToYCbCr422_8u_C3C2R_Ctx(ps.ptrs[0], 52, dstbuf.get(), 36, NppiSize{17, 12}, ctx); });
  probe("fwd_h13", "nppiRGBToYCbCr422_8u_C3C2R",
        [&] { return nppiRGBToYCbCr422_8u_C3C2R_Ctx(ps.ptrs[0], 64, dstbuf.get(), 42, NppiSize{16, 13}, ctx); });
  probe("fwd_w17_h13", "nppiRGBToYCbCr422_8u_C3C2R",
        [&] { return nppiRGBToYCbCr422_8u_C3C2R_Ctx(ps.ptrs[0], 52, dstbuf.get(), 36, NppiSize{17, 13}, ctx); });
  // ---- forward AC4C2R (BGR): src min / dst evenness ----
  probe("ac4_src_4w_m2", "nppiBGRToYCbCr422_8u_AC4C2R",
        [&] { return nppiBGRToYCbCr422_8u_AC4C2R_Ctx(ps.ptrs[0], 62, dstbuf.get(), 42, NppiSize{16, 12}, ctx); });
  probe("ac4_src_4w_eq", "nppiBGRToYCbCr422_8u_AC4C2R",
        [&] { return nppiBGRToYCbCr422_8u_AC4C2R_Ctx(ps.ptrs[0], 64, dstbuf.get(), 42, NppiSize{16, 12}, ctx); });
  probe("ac4_dst_odd", "nppiBGRToYCbCr422_8u_AC4C2R",
        [&] { return nppiBGRToYCbCr422_8u_AC4C2R_Ctx(ps.ptrs[0], 64, dstbuf.get(), 33, NppiSize{16, 12}, ctx); });
  // ---- forward C3P3R: planar dst step boundaries ----
  {
    int eq[3] = {16, 8, 8};
    int below[3] = {15, 8, 8};
    int cbelow[3] = {16, 7, 8};
    int odd[3] = {17, 9, 9};
    probe("fwdp3_dsty_eq", "nppiRGBToYCbCr422_8u_C3P3R",
          [&] { return nppiRGBToYCbCr422_8u_C3P3R_Ctx(ps.ptrs[0], 64, pd.ptrs.data(), eq, NppiSize{16, 12}, ctx); });
    probe("fwdp3_dsty_below", "nppiRGBToYCbCr422_8u_C3P3R",
          [&] { return nppiRGBToYCbCr422_8u_C3P3R_Ctx(ps.ptrs[0], 64, pd.ptrs.data(), below, NppiSize{16, 12}, ctx); });
    probe("fwdp3_dstcb_below", "nppiRGBToYCbCr422_8u_C3P3R",
          [&] { return nppiRGBToYCbCr422_8u_C3P3R_Ctx(ps.ptrs[0], 64, pd.ptrs.data(), cbelow, NppiSize{16, 12}, ctx); });
    probe("fwdp3_dst_odd", "nppiRGBToYCbCr422_8u_C3P3R",
          [&] { return nppiRGBToYCbCr422_8u_C3P3R_Ctx(ps.ptrs[0], 64, pd.ptrs.data(), odd, NppiSize{16, 12}, ctx); });
    probe("fwdp3_h13", "nppiRGBToYCbCr422_8u_C3P3R",
          [&] { return nppiRGBToYCbCr422_8u_C3P3R_Ctx(ps.ptrs[0], 64, pd.ptrs.data(), eq, NppiSize{16, 13}, ctx); });
  }
  // ---- forward P3C2R: planar src min / C2 dst min / evenness ----
  probe("p3c2_src_eq", "nppiRGBToYCbCr422_8u_P3C2R",
        [&] { return nppiRGBToYCbCr422_8u_P3C2R_Ctx(sbuf(pd), 16, dstbuf.get(), 42, NppiSize{16, 12}, ctx); });
  probe("p3c2_src_below", "nppiRGBToYCbCr422_8u_P3C2R",
        [&] { return nppiRGBToYCbCr422_8u_P3C2R_Ctx(sbuf(pd), 15, dstbuf.get(), 42, NppiSize{16, 12}, ctx); });
  probe("p3c2_src_odd", "nppiRGBToYCbCr422_8u_P3C2R",
        [&] { return nppiRGBToYCbCr422_8u_P3C2R_Ctx(sbuf(pd), 21, dstbuf.get(), 42, NppiSize{16, 12}, ctx); });
  probe("p3c2_dst_2w_eq", "nppiRGBToYCbCr422_8u_P3C2R",
        [&] { return nppiRGBToYCbCr422_8u_P3C2R_Ctx(sbuf(pd), 16, dstbuf.get(), 32, NppiSize{16, 12}, ctx); });
  probe("p3c2_dst_2w_m1", "nppiRGBToYCbCr422_8u_P3C2R",
        [&] { return nppiRGBToYCbCr422_8u_P3C2R_Ctx(sbuf(pd), 16, dstbuf.get(), 31, NppiSize{16, 12}, ctx); });
  probe("p3c2_w17", "nppiRGBToYCbCr422_8u_P3C2R",
        [&] { return nppiRGBToYCbCr422_8u_P3C2R_Ctx(sbuf(pd), 16, dstbuf.get(), 36, NppiSize{17, 12}, ctx); });
  // ---- inverse C2C3R: null / ROI / min / evenness / parity ----
  probe("inv_null_src", "nppiYCbCr422ToRGB_8u_C2C3R",
        [&] { return nppiYCbCr422ToRGB_8u_C2C3R_Ctx(nullptr, 34, dstbuf.get(), 64, NppiSize{16, 12}, ctx); });
  probe("inv_neg_roi", "nppiYCbCr422ToRGB_8u_C2C3R",
        [&] { return nppiYCbCr422ToRGB_8u_C2C3R_Ctx(c2src.get(), 34, dstbuf.get(), 64, NppiSize{16, -3}, ctx); });
  probe("inv_zero_roi", "nppiYCbCr422ToRGB_8u_C2C3R",
        [&] { return nppiYCbCr422ToRGB_8u_C2C3R_Ctx(c2src.get(), 34, dstbuf.get(), 64, NppiSize{0, 12}, ctx); });
  probe("inv_src_2w_m1", "nppiYCbCr422ToRGB_8u_C2C3R",
        [&] { return nppiYCbCr422ToRGB_8u_C2C3R_Ctx(c2src.get(), 31, dstbuf.get(), 64, NppiSize{16, 12}, ctx); });
  probe("inv_src_2w_eq", "nppiYCbCr422ToRGB_8u_C2C3R",
        [&] { return nppiYCbCr422ToRGB_8u_C2C3R_Ctx(c2src.get(), 32, dstbuf.get(), 64, NppiSize{16, 12}, ctx); });
  probe("inv_src_odd", "nppiYCbCr422ToRGB_8u_C2C3R",
        [&] { return nppiYCbCr422ToRGB_8u_C2C3R_Ctx(c2src.get(), 33, dstbuf.get(), 64, NppiSize{16, 12}, ctx); });
  probe("inv_dst_step_eq", "nppiYCbCr422ToRGB_8u_C2C3R",
        [&] { return nppiYCbCr422ToRGB_8u_C2C3R_Ctx(c2src.get(), 34, dstbuf.get(), 48, NppiSize{16, 12}, ctx); });
  probe("inv_dst_step_below", "nppiYCbCr422ToRGB_8u_C2C3R",
        [&] { return nppiYCbCr422ToRGB_8u_C2C3R_Ctx(c2src.get(), 34, dstbuf.get(), 47, NppiSize{16, 12}, ctx); });
  probe("inv_dst_odd", "nppiYCbCr422ToRGB_8u_C2C3R",
        [&] { return nppiYCbCr422ToRGB_8u_C2C3R_Ctx(c2src.get(), 34, dstbuf.get(), 49, NppiSize{16, 12}, ctx); });
  probe("inv_w17", "nppiYCbCr422ToRGB_8u_C2C3R",
        [&] { return nppiYCbCr422ToRGB_8u_C2C3R_Ctx(c2src.get(), 34, dstbuf.get(), 52, NppiSize{17, 12}, ctx); });
  probe("inv_h13", "nppiYCbCr422ToRGB_8u_C2C3R",
        [&] { return nppiYCbCr422ToRGB_8u_C2C3R_Ctx(c2src.get(), 34, dstbuf.get(), 64, NppiSize{16, 13}, ctx); });
  probe("inv_w17_h13", "nppiYCbCr422ToRGB_8u_C2C3R",
        [&] { return nppiYCbCr422ToRGB_8u_C2C3R_Ctx(c2src.get(), 34, dstbuf.get(), 52, NppiSize{17, 13}, ctx); });
  // ---- inverse C2C4R (BGR, nAval): dst min / parity ----
  probe("c2c4_dst_4w_m2", "nppiYCbCr422ToBGR_8u_C2C4R",
        [&] { return nppiYCbCr422ToBGR_8u_C2C4R_Ctx(c2src.get(), 34, dstbuf.get(), 62, NppiSize{16, 12}, 7, ctx); });
  probe("c2c4_dst_4w_eq", "nppiYCbCr422ToBGR_8u_C2C4R",
        [&] { return nppiYCbCr422ToBGR_8u_C2C4R_Ctx(c2src.get(), 34, dstbuf.get(), 64, NppiSize{16, 12}, 7, ctx); });
  probe("c2c4_w17", "nppiYCbCr422ToBGR_8u_C2C4R",
        [&] { return nppiYCbCr422ToBGR_8u_C2C4R_Ctx(c2src.get(), 34, dstbuf.get(), 68, NppiSize{17, 12}, 7, ctx); });
  // ---- inverse C2P3R (planar dst, single step): min / oddness ----
  probe("c2p3_dst_eq", "nppiYCbCr422ToRGB_8u_C2P3R",
        [&] { return nppiYCbCr422ToRGB_8u_C2P3R_Ctx(c2src.get(), 34, pd.ptrs.data(), 16, NppiSize{16, 12}, ctx); });
  probe("c2p3_dst_below", "nppiYCbCr422ToRGB_8u_C2P3R",
        [&] { return nppiYCbCr422ToRGB_8u_C2P3R_Ctx(c2src.get(), 34, pd.ptrs.data(), 15, NppiSize{16, 12}, ctx); });
  probe("c2p3_dst_odd", "nppiYCbCr422ToRGB_8u_C2P3R",
        [&] { return nppiYCbCr422ToRGB_8u_C2P3R_Ctx(c2src.get(), 34, pd.ptrs.data(), 17, NppiSize{16, 12}, ctx); });
  // ---- inverse P3C3R (planar src with step array) ----
  {
    int ssteps[3] = {32, 16, 16};
    int bad[3] = {15, 16, 16};
    int badc[3] = {32, 7, 16};
    int odd[3] = {33, 17, 17};
    probe("invp3_srcy_eq", "nppiYCbCr422ToRGB_8u_P3C3R",
          [&] { return nppiYCbCr422ToRGB_8u_P3C3R_Ctx(sbuf(pd), ssteps, dstbuf.get(), 64, NppiSize{16, 12}, ctx); });
    probe("invp3_srcy_below", "nppiYCbCr422ToRGB_8u_P3C3R",
          [&] { return nppiYCbCr422ToRGB_8u_P3C3R_Ctx(sbuf(pd), bad, dstbuf.get(), 64, NppiSize{16, 12}, ctx); });
    probe("invp3_srccb_below", "nppiYCbCr422ToRGB_8u_P3C3R",
          [&] { return nppiYCbCr422ToRGB_8u_P3C3R_Ctx(sbuf(pd), badc, dstbuf.get(), 64, NppiSize{16, 12}, ctx); });
    probe("invp3_src_odd", "nppiYCbCr422ToRGB_8u_P3C3R",
          [&] { return nppiYCbCr422ToRGB_8u_P3C3R_Ctx(sbuf(pd), odd, dstbuf.get(), 64, NppiSize{16, 12}, ctx); });
    probe("invp3_dst_odd", "nppiYCbCr422ToRGB_8u_P3C3R",
          [&] { return nppiYCbCr422ToRGB_8u_P3C3R_Ctx(sbuf(pd), ssteps, dstbuf.get(), 49, NppiSize{16, 12}, ctx); });
    probe("invp3_w17", "nppiYCbCr422ToRGB_8u_P3C3R",
          [&] { return nppiYCbCr422ToRGB_8u_P3C3R_Ctx(sbuf(pd), ssteps, dstbuf.get(), 52, NppiSize{17, 12}, ctx); });
  }
  // ---- reorder YCbCr422 -> CbYCr422 (C2R) ----
  probe("reord_src_2w_eq", "nppiYCbCr422ToCbYCr422_8u_C2R",
        [&] { return nppiYCbCr422ToCbYCr422_8u_C2R_Ctx(c2src.get(), 32, dstbuf.get(), 34, NppiSize{16, 12}, ctx); });
  probe("reord_src_2w_m1", "nppiYCbCr422ToCbYCr422_8u_C2R",
        [&] { return nppiYCbCr422ToCbYCr422_8u_C2R_Ctx(c2src.get(), 31, dstbuf.get(), 34, NppiSize{16, 12}, ctx); });
  probe("reord_src_odd", "nppiYCbCr422ToCbYCr422_8u_C2R",
        [&] { return nppiYCbCr422ToCbYCr422_8u_C2R_Ctx(c2src.get(), 33, dstbuf.get(), 34, NppiSize{16, 12}, ctx); });
  probe("reord_dst_odd", "nppiYCbCr422ToCbYCr422_8u_C2R",
        [&] { return nppiYCbCr422ToCbYCr422_8u_C2R_Ctx(c2src.get(), 32, dstbuf.get(), 33, NppiSize{16, 12}, ctx); });
  probe("reord_null", "nppiYCbCr422ToCbYCr422_8u_C2R",
        [&] { return nppiYCbCr422ToCbYCr422_8u_C2R_Ctx(nullptr, 32, dstbuf.get(), 34, NppiSize{16, 12}, ctx); });
  probe("reord_neg_roi", "nppiYCbCr422ToCbYCr422_8u_C2R",
        [&] { return nppiYCbCr422ToCbYCr422_8u_C2R_Ctx(c2src.get(), 32, dstbuf.get(), 34, NppiSize{16, -1}, ctx); });
  probe("reord_w17", "nppiYCbCr422ToCbYCr422_8u_C2R",
        [&] { return nppiYCbCr422ToCbYCr422_8u_C2R_Ctx(c2src.get(), 34, dstbuf.get(), 36, NppiSize{17, 12}, ctx); });
  // ---- Gamma variant src step constraint (C3 src) ----
  probe("gamma_w17_s66", "nppiRGBToCbYCr422Gamma_8u_C3C2R",
        [&] { return nppiRGBToCbYCr422Gamma_8u_C3C2R_Ctx(ps.ptrs[0], 66, dstbuf.get(), 36, NppiSize{17, 12}, ctx); });
  probe("gamma_w17_s68", "nppiRGBToCbYCr422Gamma_8u_C3C2R",
        [&] { return nppiRGBToCbYCr422Gamma_8u_C3C2R_Ctx(ps.ptrs[0], 68, dstbuf.get(), 36, NppiSize{17, 12}, ctx); });
  probe("gamma_w16_s62", "nppiRGBToCbYCr422Gamma_8u_C3C2R",
        [&] { return nppiRGBToCbYCr422Gamma_8u_C3C2R_Ctx(ps.ptrs[0], 62, dstbuf.get(), 42, NppiSize{16, 12}, ctx); });
  probe("gamma_w16_s63", "nppiRGBToCbYCr422Gamma_8u_C3C2R",
        [&] { return nppiRGBToCbYCr422Gamma_8u_C3C2R_Ctx(ps.ptrs[0], 63, dstbuf.get(), 42, NppiSize{16, 12}, ctx); });
  probe("gamma_w16_s66", "nppiRGBToCbYCr422Gamma_8u_C3C2R",
        [&] { return nppiRGBToCbYCr422Gamma_8u_C3C2R_Ctx(ps.ptrs[0], 66, dstbuf.get(), 42, NppiSize{16, 12}, ctx); });
  probe("gamma_w16_s60", "nppiRGBToCbYCr422Gamma_8u_C3C2R",
        [&] { return nppiRGBToCbYCr422Gamma_8u_C3C2R_Ctx(ps.ptrs[0], 60, dstbuf.get(), 42, NppiSize{16, 12}, ctx); });
  // ---- AC4 src evenness scope ----
  probe("ac4_src_66", "nppiBGRToYCbCr422_8u_AC4C2R",
        [&] { return nppiBGRToYCbCr422_8u_AC4C2R_Ctx(ps.ptrs[0], 66, dstbuf.get(), 42, NppiSize{16, 12}, ctx); });
  probe("ac4_src_69", "nppiBGRToYCbCr422_8u_AC4C2R",
        [&] { return nppiBGRToYCbCr422_8u_AC4C2R_Ctx(ps.ptrs[0], 69, dstbuf.get(), 42, NppiSize{16, 12}, ctx); });
  probe("ac4_src_63", "nppiBGRToYCbCr422_8u_AC4C2R",
        [&] { return nppiBGRToYCbCr422_8u_AC4C2R_Ctx(ps.ptrs[0], 63, dstbuf.get(), 42, NppiSize{16, 12}, ctx); });
  {
    int dsteps[3] = {16, 8, 8};
    probe("ac4p3_src_65", "nppiBGRToYCbCr422_8u_AC4P3R",
          [&] { return nppiBGRToYCbCr422_8u_AC4P3R_Ctx(ps.ptrs[0], 65, pd.ptrs.data(), dsteps, NppiSize{16, 12}, ctx); });
  }
  // ---- C2C4R dst evenness ----
  probe("c2c4_dst_65", "nppiYCbCr422ToBGR_8u_C2C4R",
        [&] { return nppiYCbCr422ToBGR_8u_C2C4R_Ctx(c2src.get(), 34, dstbuf.get(), 65, NppiSize{16, 12}, 7, ctx); });
  probe("c2c4_dst_66", "nppiYCbCr422ToBGR_8u_C2C4R",
        [&] { return nppiYCbCr422ToBGR_8u_C2C4R_Ctx(c2src.get(), 34, dstbuf.get(), 66, NppiSize{16, 12}, 7, ctx); });
  // ---- P3C3R inverse warning parity ----
  {
    int ssteps[3] = {32, 16, 16};
    probe("invp3_h13", "nppiYCbCr422ToRGB_8u_P3C3R",
          [&] { return nppiYCbCr422ToRGB_8u_P3C3R_Ctx(sbuf(pd), ssteps, dstbuf.get(), 64, NppiSize{16, 13}, ctx); });
    probe("invp3_w17_h13", "nppiYCbCr422ToRGB_8u_P3C3R",
          [&] { return nppiYCbCr422ToRGB_8u_P3C3R_Ctx(sbuf(pd), ssteps, dstbuf.get(), 64, NppiSize{17, 13}, ctx); });
  }
  // ---- remaining C2 evenness corners ----
  probe("p3c2_dst_odd", "nppiRGBToYCbCr422_8u_P3C2R",
        [&] { return nppiRGBToYCbCr422_8u_P3C2R_Ctx(sbuf(pd), 16, dstbuf.get(), 33, NppiSize{16, 12}, ctx); });
  probe("c2p3_src_odd", "nppiYCbCr422ToRGB_8u_C2P3R",
        [&] { return nppiYCbCr422ToRGB_8u_C2P3R_Ctx(c2src.get(), 33, pd.ptrs.data(), 16, NppiSize{16, 12}, ctx); });
  probe("c2p3_w17h12", "nppiYCbCr422ToRGB_8u_C2P3R",
        [&] { return nppiYCbCr422ToRGB_8u_C2P3R_Ctx(c2src.get(), 40, pd.ptrs.data(), 18, NppiSize{17, 12}, ctx); });
  probe("c2p3_w16h13", "nppiYCbCr422ToRGB_8u_C2P3R",
        [&] { return nppiYCbCr422ToRGB_8u_C2P3R_Ctx(c2src.get(), 40, pd.ptrs.data(), 16, NppiSize{16, 13}, ctx); });
  probe("p3c2_h13", "nppiRGBToYCbCr422_8u_P3C2R",
        [&] { return nppiRGBToYCbCr422_8u_P3C2R_Ctx(sbuf(pd), 16, dstbuf.get(), 42, NppiSize{16, 13}, ctx); });
  {
    int dsteps[3] = {16, 8, 8};
    probe("ac4p3_h13", "nppiBGRToYCbCr422_8u_AC4P3R",
          [&] { return nppiBGRToYCbCr422_8u_AC4P3R_Ctx(ps.ptrs[0], 64, pd.ptrs.data(), dsteps, NppiSize{16, 13}, ctx); });
    probe("ac4p3_w17h12", "nppiBGRToYCbCr422_8u_AC4P3R",
          [&] { return nppiBGRToYCbCr422_8u_AC4P3R_Ctx(ps.ptrs[0], 68, pd.ptrs.data(), dsteps, NppiSize{17, 12}, ctx); });
  }
  // ---- zero ROI vs invalid steps (validation order) ----
  probe("fwd_zero_roi_badstep", "nppiRGBToYCbCr422_8u_C3C2R",
        [&] { return nppiRGBToYCbCr422_8u_C3C2R_Ctx(ps.ptrs[0], 0, dstbuf.get(), 0, NppiSize{0, 0}, ctx); });
  probe("inv_zero_roi_badstep", "nppiYCbCr422ToRGB_8u_C2C3R",
        [&] { return nppiYCbCr422ToRGB_8u_C2C3R_Ctx(c2src.get(), 0, dstbuf.get(), 0, NppiSize{0, 12}, ctx); });
  // ---- Gamma variant: validation order before unconditional STEP_ERROR ----
  probe("gamma_null_src", "nppiRGBToCbYCr422Gamma_8u_C3C2R",
        [&] { return nppiRGBToCbYCr422Gamma_8u_C3C2R_Ctx(nullptr, 64, dstbuf.get(), 42, NppiSize{16, 12}, ctx); });
  probe("gamma_neg_roi", "nppiRGBToCbYCr422Gamma_8u_C3C2R",
        [&] { return nppiRGBToCbYCr422Gamma_8u_C3C2R_Ctx(ps.ptrs[0], 64, dstbuf.get(), 42, NppiSize{-2, 12}, ctx); });
  probe("gamma_zero_roi", "nppiRGBToCbYCr422Gamma_8u_C3C2R",
        [&] { return nppiRGBToCbYCr422Gamma_8u_C3C2R_Ctx(ps.ptrs[0], 64, dstbuf.get(), 42, NppiSize{0, 0}, ctx); });
  // ---- odd-width step-minimum boundaries (w vs even width) ----
  {
    int dsteps[3] = {16, 9, 9};
    probe("fwdp3_w17_dsty16", "nppiRGBToYCbCr422_8u_C3P3R",
          [&] { return nppiRGBToYCbCr422_8u_C3P3R_Ctx(ps.ptrs[0], 54, pd.ptrs.data(), dsteps, NppiSize{17, 12}, ctx); });
  }
  probe("p3c2_w17_src15", "nppiRGBToYCbCr422_8u_P3C2R",
        [&] { return nppiRGBToYCbCr422_8u_P3C2R_Ctx(sbuf(pd), 15, dstbuf.get(), 36, NppiSize{17, 12}, ctx); });
  {
    int s17[3] = {16, 9, 9};
    probe("invp3_w17_srcy16", "nppiYCbCr422ToRGB_8u_P3C3R",
          [&] { return nppiYCbCr422ToRGB_8u_P3C3R_Ctx(sbuf(pd), s17, dstbuf.get(), 54, NppiSize{17, 12}, ctx); });
  }
  probe("c2p3_w17_dst16", "nppiYCbCr422ToRGB_8u_C2P3R",
        [&] { return nppiYCbCr422ToRGB_8u_C2P3R_Ctx(c2src.get(), 34, pd.ptrs.data(), 16, NppiSize{17, 12}, ctx); });
  // ---- odd-width min-step boundaries: w vs even width (we = w & ~1) ----
  probe("fwd_w17_src49", "nppiRGBToYCbCr422_8u_C3C2R",
        [&] { return nppiRGBToYCbCr422_8u_C3C2R_Ctx(ps.ptrs[0], 49, dstbuf.get(), 36, NppiSize{17, 12}, ctx); });
  {
    int de17[3] = {16, 9, 9};
    probe("fwdp3_w17_src49", "nppiRGBToYCbCr422_8u_C3P3R",
          [&] { return nppiRGBToYCbCr422_8u_C3P3R_Ctx(ps.ptrs[0], 49, pd.ptrs.data(), de17, NppiSize{17, 12}, ctx); });
  }
  probe("ac4_w17_src65", "nppiBGRToYCbCr422_8u_AC4C2R",
        [&] { return nppiBGRToYCbCr422_8u_AC4C2R_Ctx(ps.ptrs[0], 65, dstbuf.get(), 36, NppiSize{17, 12}, ctx); });
  probe("inv_w17_src33", "nppiYCbCr422ToRGB_8u_C2C3R",
        [&] { return nppiYCbCr422ToRGB_8u_C2C3R_Ctx(c2src.get(), 33, dstbuf.get(), 52, NppiSize{17, 12}, ctx); });
  probe("inv_w17_dst49", "nppiYCbCr422ToRGB_8u_C2C3R",
        [&] { return nppiYCbCr422ToRGB_8u_C2C3R_Ctx(c2src.get(), 34, dstbuf.get(), 49, NppiSize{17, 12}, ctx); });
  probe("c2p3_w17_src33", "nppiYCbCr422ToRGB_8u_C2P3R",
        [&] { return nppiYCbCr422ToRGB_8u_C2P3R_Ctx(c2src.get(), 33, pd.ptrs.data(), 16, NppiSize{17, 12}, ctx); });
  probe("c2c4_w17_dst65", "nppiYCbCr422ToBGR_8u_C2C4R",
        [&] { return nppiYCbCr422ToBGR_8u_C2C4R_Ctx(c2src.get(), 34, dstbuf.get(), 65, NppiSize{17, 12}, 7, ctx); });
  probe("reord_w17_src33", "nppiYCbCr422ToCbYCr422_8u_C2R",
        [&] { return nppiYCbCr422ToCbYCr422_8u_C2R_Ctx(c2src.get(), 33, dstbuf.get(), 36, NppiSize{17, 12}, ctx); });
  probe("reord_w17_dst33", "nppiYCbCr422ToCbYCr422_8u_C2R",
        [&] { return nppiYCbCr422ToCbYCr422_8u_C2R_Ctx(c2src.get(), 34, dstbuf.get(), 33, NppiSize{17, 12}, ctx); });
}
