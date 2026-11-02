// Differential tests for the c3 colour-space conversion families
// (nppi_color_conversion.h):
//
//   nppiRGBToYCrCb420, nppiBGRToYCrCb420, nppiBGRToYCrCb420_709CSC,
//   nppiBGRToYCbCr420_709CSC, nppiBGRToYCbCr420_709HDTV,
//   nppiRGBToYCrCb422, nppiBGRToCbYCr422_709HDTV,
//   nppiYCbCr420ToBGR{,709CSC,709HDTV}, nppiYCrCb420ToRGB,
//   nppiYCbCrToBGR_709CSC, nppiCbYCr422ToBGR_709HDTV,
//   nppiRGBToYCC, nppiYCCToRGB,
//   nppiCMYKOrYCCKToRGB/BGR_JPEG, nppiYCCKToCMYK_JPEG_601
//
// Measured NVIDIA NPP 13.4 (RTX 5090) semantics:
//   * BT.601 video range: Y = 16 + .257R + .504G + .098B;
//     Cb = 128 - .148R - .291G + .439B; Cr = 128 + .439R - .368G - .071B.
//   * BT.709 video range (CSC): .183/.614/.062 (+16), chroma .101/.338/.439
//     and .439/.399/.040.
//   * BT.709 HDTV (full range): .213/.715/.072 (no offset), chroma
//     .117/.394/.511 and .511/.464/.047.
//   * Forward 4:2:0 chroma is the average (sum >> 2) of the per-pixel
//     truncated chroma over each 2x2 block; forward 4:2:2 C2 averages pairs.
//   * 4:2:0 forward conversions report NPP_DOUBLE_SIZE_WARNING for odd
//     extents; the 709CSC 4:2:0 inverse warns for odd heights only.
//   * Planar 709CSC source/destination is full resolution (4:4:4).
//   * JPEG CMYK uses an integer (255-C)*(255-K)/255 path with a measured
//     exact-division off-by-one table; see nppi_c3_cmyk_mul.h.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiC3Color*Diff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records ./diff_tests --gtest_filter='NppiC3Color*Diff.*'

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

const Npp8u **sbuf(const Planes &p) { return const_cast<const Npp8u **>(p.ptrs.data()); }

// --- packed RGB/BGR -> planar 4:2:0 ---
void fwd420(const char *fname, const std::string &key, int w, int h, int pp, int bgr,
            NppStatus (*f)(const Npp8u *, int, Npp8u **, int *, NppiSize, NppStreamContext), uint64_t seed) {
  const int sStep = pp * w + 6;
  const int yStep = w + 5;
  const int cStep = w / 2 + 6;
  DeviceMemory<Npp8u> src((size_t)sStep * h);
  src.copyFromHost(make_uniform<Npp8u>((size_t)sStep * h, seed + 11));
  Planes dst = make_planes({(size_t)yStep * h, (size_t)cStep * h, (size_t)cStep * h}, seed + 29);
  int steps[3] = {yStep, cStep, cStep};
  clear_cuda_error();
  NppStatus st = f(src.get(), sStep, dst.ptrs.data(), steps, NppiSize{w, h}, current_ctx());
  Case c("nppiC3Color/" + key, fname);
  c.status(st);
  c.device_output<Npp8u>(dst.dev[0]->get(), (size_t)yStep * h, ValueKind::U8);
  c.device_output<Npp8u>(dst.dev[1]->get(), (size_t)cStep * (h / 2), ValueKind::U8);
  c.device_output<Npp8u>(dst.dev[2]->get(), (size_t)cStep * (h / 2), ValueKind::U8);
  c.finish();
  (void)bgr;
}

// --- packed RGB/BGR -> packed 4:2:2 C2 (even widths only) ---
void fwdC2(const char *fname, const std::string &key, int w, int h, int pp,
           NppStatus (*f)(const Npp8u *, int, Npp8u *, int, NppiSize, NppStreamContext), uint64_t seed) {
  const int sStep = pp * w + 6;
  const int dStep = 2 * w + 8;
  DeviceMemory<Npp8u> src((size_t)sStep * h);
  src.copyFromHost(make_uniform<Npp8u>((size_t)sStep * h, seed + 11));
  DeviceMemory<Npp8u> dst((size_t)dStep * h);
  dst.copyFromHost(make_uniform<Npp8u>((size_t)dStep * h, seed + 29));
  clear_cuda_error();
  NppStatus st = f(src.get(), sStep, dst.get(), dStep, NppiSize{w, h}, current_ctx());
  Case c("nppiC3Color/" + key, fname);
  c.status(st);
  c.device_output<Npp8u>(dst.get(), (size_t)dStep * h, ValueKind::U8);
  c.finish();
}

// --- planar RGB -> packed 4:2:2 YCrCb (even widths only) ---
void fwdP3C2(const char *fname, const std::string &key, int w, int h,
             NppStatus (*f)(const Npp8u *const *, int, Npp8u *, int, NppiSize, NppStreamContext), uint64_t seed) {
  const int sStep = w + 5;
  const int dStep = 2 * w + 8;
  Planes src = make_planes({(size_t)sStep * h, (size_t)sStep * h, (size_t)sStep * h}, seed + 11);
  DeviceMemory<Npp8u> dst((size_t)dStep * h);
  dst.copyFromHost(make_uniform<Npp8u>((size_t)dStep * h, seed + 29));
  clear_cuda_error();
  NppStatus st = f(const_cast<const Npp8u **>(src.ptrs.data()), sStep, dst.get(), dStep, NppiSize{w, h}, current_ctx());
  Case c("nppiC3Color/" + key, fname);
  c.status(st);
  c.device_output<Npp8u>(dst.get(), (size_t)dStep * h, ValueKind::U8);
  c.finish();
}

// --- planar YUV 4:2:0 -> packed RGB/BGR ---
template <class FN>
void inv420(const char *fname, const std::string &key, int w, int h, int pp, Npp8u aval, FN f, uint64_t seed) {
  const int yStep = w + 5;
  const int cStep = w / 2 + 6;
  const int dStep = pp * w + 8;
  const int ch = h / 2 > 0 ? h / 2 : 1;
  Planes src = make_planes({(size_t)yStep * h, (size_t)cStep * ch, (size_t)cStep * ch}, seed + 11);
  DeviceMemory<Npp8u> dst((size_t)dStep * h);
  dst.copyFromHost(make_uniform<Npp8u>((size_t)dStep * h, seed + 29));
  int steps[3] = {yStep, cStep, cStep};
  clear_cuda_error();
  NppStatus st = f(sbuf(src), steps, dst.get(), dStep, NppiSize{w, h}, aval, current_ctx());
  Case c("nppiC3Color/" + key, fname);
  c.status(st);
  c.device_output<Npp8u>(dst.get(), (size_t)dStep * h, ValueKind::U8);
  c.finish();
}

// --- planar 4:4:4 YCbCr -> packed BGR ---
template <class FN>
void inv422(const char *fname, const std::string &key, int w, int h, int pp, Npp8u aval, FN f, uint64_t seed) {
  const int sStep = w + 5;
  const int dStep = pp * w + 8;
  Planes src = make_planes({(size_t)sStep * h, (size_t)sStep * h, (size_t)sStep * h}, seed + 11);
  DeviceMemory<Npp8u> dst((size_t)dStep * h);
  dst.copyFromHost(make_uniform<Npp8u>((size_t)dStep * h, seed + 29));
  clear_cuda_error();
  NppStatus st = f(sbuf(src), sStep, dst.get(), dStep, NppiSize{w, h}, aval, current_ctx());
  Case c("nppiC3Color/" + key, fname);
  c.status(st);
  c.device_output<Npp8u>(dst.get(), (size_t)dStep * h, ValueKind::U8);
  c.finish();
}

// --- packed CbYCr422 -> packed BGR (even widths only) ---
template <class FN>
void invC2(const char *fname, const std::string &key, int w, int h, int pp, Npp8u aval, FN f, uint64_t seed) {
  const int sStep = 2 * w + 8;
  const int dStep = pp * w + 8;
  DeviceMemory<Npp8u> src((size_t)sStep * h);
  src.copyFromHost(make_uniform<Npp8u>((size_t)sStep * h, seed + 11));
  DeviceMemory<Npp8u> dst((size_t)dStep * h);
  dst.copyFromHost(make_uniform<Npp8u>((size_t)dStep * h, seed + 29));
  clear_cuda_error();
  NppStatus st = f(src.get(), sStep, dst.get(), dStep, NppiSize{w, h}, aval, current_ctx());
  Case c("nppiC3Color/" + key, fname);
  c.status(st);
  c.device_output<Npp8u>(dst.get(), (size_t)dStep * h, ValueKind::U8);
  c.finish();
}

// --- packed 3/4-channel conversions (YCC, RGB<->YCC) ---
void packed(const char *fname, const std::string &key, int w, int h, int pp,
            NppStatus (*f)(const Npp8u *, int, Npp8u *, int, NppiSize, NppStreamContext), uint64_t seed) {
  const int sStep = pp * w + 6;
  const int dStep = pp * w + 8;
  DeviceMemory<Npp8u> src((size_t)sStep * h);
  src.copyFromHost(make_uniform<Npp8u>((size_t)sStep * h, seed + 11));
  DeviceMemory<Npp8u> dst((size_t)dStep * h);
  dst.copyFromHost(make_uniform<Npp8u>((size_t)dStep * h, seed + 29));
  clear_cuda_error();
  NppStatus st = f(src.get(), sStep, dst.get(), dStep, NppiSize{w, h}, current_ctx());
  Case c("nppiC3Color/" + key, fname);
  c.status(st);
  c.device_output<Npp8u>(dst.get(), (size_t)dStep * h, ValueKind::U8);
  c.finish();
}

// --- planar CMYK/YCCK (4 planes in, 3 planes out) ---
void cmykPlanar(const char *fname, const std::string &key, int w, int h,
                NppStatus (*f)(const Npp8u **, int, Npp8u **, int, NppiSize, NppStreamContext), uint64_t seed) {
  const int sStep = w + 5;
  const int dStep = w + 6;
  Planes src = make_planes({(size_t)sStep * h, (size_t)sStep * h, (size_t)sStep * h, (size_t)sStep * h}, seed + 11);
  Planes dst = make_planes({(size_t)dStep * h, (size_t)dStep * h, (size_t)dStep * h}, seed + 29);
  clear_cuda_error();
  NppStatus st = f(const_cast<const Npp8u **>(src.ptrs.data()), sStep, dst.ptrs.data(), dStep, NppiSize{w, h}, current_ctx());
  Case c("nppiC3Color/" + key, fname);
  c.status(st);
  for (int i = 0; i < 3; ++i) {
    c.device_output<Npp8u>(dst.dev[i]->get(), (size_t)dStep * h, ValueKind::U8);
  }
  c.finish();
}

// --- planar CMYK/YCCK (4 planes in, 4 planes out) ---
void ycckPlanar(const char *fname, const std::string &key, int w, int h,
                NppStatus (*f)(const Npp8u **, int, Npp8u **, int, NppiSize, NppStreamContext), uint64_t seed) {
  const int sStep = w + 5;
  const int dStep = w + 6;
  Planes src = make_planes({(size_t)sStep * h, (size_t)sStep * h, (size_t)sStep * h, (size_t)sStep * h}, seed + 11);
  Planes dst = make_planes({(size_t)dStep * h, (size_t)dStep * h, (size_t)dStep * h, (size_t)dStep * h}, seed + 29);
  clear_cuda_error();
  NppStatus st = f(const_cast<const Npp8u **>(src.ptrs.data()), sStep, dst.ptrs.data(), dStep, NppiSize{w, h}, current_ctx());
  Case c("nppiC3Color/" + key, fname);
  c.status(st);
  for (int i = 0; i < 4; ++i) {
    c.device_output<Npp8u>(dst.dev[i]->get(), (size_t)dStep * h, ValueKind::U8);
  }
  c.finish();
}

// --- packed CMYK/YCCK (4 planes in, packed 3-channel out) ---
void cmykPacked(const char *fname, const std::string &key, int w, int h, int pp,
                NppStatus (*f)(const Npp8u **, int, Npp8u *, int, NppiSize, NppStreamContext), uint64_t seed) {
  const int sStep = w + 5;
  const int dStep = pp * w + 8;
  Planes src = make_planes({(size_t)sStep * h, (size_t)sStep * h, (size_t)sStep * h, (size_t)sStep * h}, seed + 11);
  DeviceMemory<Npp8u> dst((size_t)dStep * h);
  dst.copyFromHost(make_uniform<Npp8u>((size_t)dStep * h, seed + 29));
  clear_cuda_error();
  NppStatus st = f(const_cast<const Npp8u **>(src.ptrs.data()), sStep, dst.get(), dStep, NppiSize{w, h}, current_ctx());
  Case c("nppiC3Color/" + key, fname);
  c.status(st);
  c.device_output<Npp8u>(dst.get(), (size_t)dStep * h, ValueKind::U8);
  c.finish();
}

} // namespace

TEST(NppiC3ColorFwdDiff, RgbTo420) {
  fwd420("nppiRGBToYCrCb420_8u_AC4P3R", "rgb2ycrcb420/16x12", 16, 12, 4, 0, nppiRGBToYCrCb420_8u_AC4P3R_Ctx, 0xC3C01);
  fwd420("nppiRGBToYCrCb420_8u_AC4P3R", "rgb2ycrcb420/17x13", 17, 13, 4, 0, nppiRGBToYCrCb420_8u_AC4P3R_Ctx, 0xC3C02);
  fwd420("nppiBGRToYCrCb420_8u_C3P3R", "bgr2ycrcb420/16x12", 16, 12, 3, 1, nppiBGRToYCrCb420_8u_C3P3R_Ctx, 0xC3C03);
  fwd420("nppiBGRToYCrCb420_8u_AC4P3R", "bgr2ycrcb420/ac4/16x13", 16, 13, 4, 1,
         nppiBGRToYCrCb420_8u_AC4P3R_Ctx, 0xC3C04);
}

TEST(NppiC3ColorFwdDiff, BgrTo709_420) {
  fwd420("nppiBGRToYCbCr420_709CSC_8u_C3P3R", "bgr2ycbcr420csc/16x12", 16, 12, 3, 1,
         nppiBGRToYCbCr420_709CSC_8u_C3P3R_Ctx, 0xC3C05);
  fwd420("nppiBGRToYCbCr420_709CSC_8u_AC4P3R", "bgr2ycbcr420csc/ac4/17x13", 17, 13, 4, 1,
         nppiBGRToYCbCr420_709CSC_8u_AC4P3R_Ctx, 0xC3C06);
  fwd420("nppiBGRToYCrCb420_709CSC_8u_C3P3R", "bgr2ycrcb420csc/16x12", 16, 12, 3, 1,
         nppiBGRToYCrCb420_709CSC_8u_C3P3R_Ctx, 0xC3C07);
  fwd420("nppiBGRToYCrCb420_709CSC_8u_AC4P3R", "bgr2ycrcb420csc/ac4/16x13", 16, 13, 4, 1,
         nppiBGRToYCrCb420_709CSC_8u_AC4P3R_Ctx, 0xC3C08);
  fwd420("nppiBGRToYCbCr420_709HDTV_8u_AC4P3R", "bgr2ycbcr420hdtv/ac4/16x12", 16, 12, 4, 1,
         nppiBGRToYCbCr420_709HDTV_8u_AC4P3R_Ctx, 0xC3C09);
  fwd420("nppiBGRToYCbCr420_709HDTV_8u_AC4P3R", "bgr2ycbcr420hdtv/ac4/17x13", 17, 13, 4, 1,
         nppiBGRToYCbCr420_709HDTV_8u_AC4P3R_Ctx, 0xC3C0A);
}

TEST(NppiC3ColorFwdDiff, RgbTo422) {
  fwdC2("nppiRGBToYCrCb422_8u_C3C2R", "rgb2ycrcb422/c3c2/16x12", 16, 12, 3,
        nppiRGBToYCrCb422_8u_C3C2R_Ctx, 0xC3C0B);
  fwdC2("nppiRGBToYCrCb422_8u_C3C2R", "rgb2ycrcb422/c3c2/8x6", 8, 6, 3,
        nppiRGBToYCrCb422_8u_C3C2R_Ctx, 0xC3C0C);
  fwdP3C2("nppiRGBToYCrCb422_8u_P3C2R", "rgb2ycrcb422/p3c2/16x12", 16, 12,
          nppiRGBToYCrCb422_8u_P3C2R_Ctx, 0xC3C0D);
  fwdP3C2("nppiRGBToYCrCb422_8u_P3C2R", "rgb2ycrcb422/p3c2/8x6", 8, 6,
          nppiRGBToYCrCb422_8u_P3C2R_Ctx, 0xC3C0E);
  fwdC2("nppiBGRToCbYCr422_709HDTV_8u_C3C2R", "bgr2cbycr422/16x12", 16, 12, 3,
        nppiBGRToCbYCr422_709HDTV_8u_C3C2R_Ctx, 0xC3C0F);
  fwdC2("nppiBGRToCbYCr422_709HDTV_8u_AC4C2R", "bgr2cbycr422/ac4/16x13", 16, 13, 4,
        nppiBGRToCbYCr422_709HDTV_8u_AC4C2R_Ctx, 0xC3C10);
}

TEST(NppiC3ColorInvDiff, ToBgr420) {
  inv420("nppiYCbCr420ToBGR_8u_P3C4R", "ycbcr420tobgr/16x12", 16, 12, 4, 9,
         [&](const Npp8u *const *p, int *ss, Npp8u *d, int ds, NppiSize r, Npp8u a, NppStreamContext c) { return nppiYCbCr420ToBGR_8u_P3C4R_Ctx(p, ss, d, ds, r, a, c); }, 0xC3D01);
  inv420("nppiYCbCr420ToBGR_8u_P3C4R", "ycbcr420tobgr/17x13", 17, 13, 4, 9,
         [&](const Npp8u *const *p, int *ss, Npp8u *d, int ds, NppiSize r, Npp8u a, NppStreamContext c) { return nppiYCbCr420ToBGR_8u_P3C4R_Ctx(p, ss, d, ds, r, a, c); }, 0xC3D02);
  inv420("nppiYCrCb420ToRGB_8u_P3C4R", "ycrcb420torgb/16x12", 16, 12, 4, 3,
         [&](const Npp8u *const *p, int *ss, Npp8u *d, int ds, NppiSize r, Npp8u a, NppStreamContext c) { return nppiYCrCb420ToRGB_8u_P3C4R_Ctx(p, ss, d, ds, r, a, c); }, 0xC3D03);
  inv420("nppiYCrCb420ToRGB_8u_P3C4R", "ycrcb420torgb/8x5", 8, 5, 4, 3,
         [&](const Npp8u *const *p, int *ss, Npp8u *d, int ds, NppiSize r, Npp8u a, NppStreamContext c) { return nppiYCrCb420ToRGB_8u_P3C4R_Ctx(p, ss, d, ds, r, a, c); }, 0xC3D04);
  inv420("nppiYCbCr420ToBGR_709HDTV_8u_P3C4R", "ycbcr420tobgrhdtv/16x12", 16, 12, 4, 7,
         [&](const Npp8u *const *p, int *ss, Npp8u *d, int ds, NppiSize r, Npp8u a, NppStreamContext c) { return nppiYCbCr420ToBGR_709HDTV_8u_P3C4R_Ctx(p, ss, d, ds, r, a, c); }, 0xC3D05);
  inv420("nppiYCbCr420ToBGR_709CSC_8u_P3C3R", "ycbcr420tobgrcsc/16x12", 16, 12, 3, 0,
         [&](const Npp8u *const *p, int *ss, Npp8u *d, int ds, NppiSize r, Npp8u, NppStreamContext c) { return nppiYCbCr420ToBGR_709CSC_8u_P3C3R_Ctx(p, ss, d, ds, r, c); }, 0xC3D06);
  inv420("nppiYCbCr420ToBGR_709CSC_8u_P3C3R", "ycbcr420tobgrcsc/16x13", 16, 13, 3, 0,
         [&](const Npp8u *const *p, int *ss, Npp8u *d, int ds, NppiSize r, Npp8u, NppStreamContext c) { return nppiYCbCr420ToBGR_709CSC_8u_P3C3R_Ctx(p, ss, d, ds, r, c); }, 0xC3D07);
}

TEST(NppiC3ColorInvDiff, PlanarAndC2) {
  inv422("nppiYCbCrToBGR_709CSC_8u_P3C3R", "ycbcr2bgrcsc/16x12", 16, 12, 3, 0,
         [&](const Npp8u *const *p, int ss, Npp8u *d, int ds, NppiSize r, Npp8u, NppStreamContext c) { return nppiYCbCrToBGR_709CSC_8u_P3C3R_Ctx(p, ss, d, ds, r, c); }, 0xC3D08);
  inv422("nppiYCbCrToBGR_709CSC_8u_P3C3R", "ycbcr2bgrcsc/17x13", 17, 13, 3, 0,
         [&](const Npp8u *const *p, int ss, Npp8u *d, int ds, NppiSize r, Npp8u, NppStreamContext c) { return nppiYCbCrToBGR_709CSC_8u_P3C3R_Ctx(p, ss, d, ds, r, c); }, 0xC3D09);
  invC2("nppiCbYCr422ToBGR_709HDTV_8u_C2C3R", "cbycr2bgr/16x12", 16, 12, 3, 0,
        [&](const Npp8u *sp, int ss, Npp8u *d, int ds, NppiSize r, Npp8u, NppStreamContext c) { return nppiCbYCr422ToBGR_709HDTV_8u_C2C3R_Ctx(sp, ss, d, ds, r, c); }, 0xC3D0A);
  invC2("nppiCbYCr422ToBGR_709HDTV_8u_C2C4R", "cbycr2bgr/c4/8x6", 8, 6, 4, 41,
        [&](const Npp8u *sp, int ss, Npp8u *d, int ds, NppiSize r, Npp8u a, NppStreamContext c) { return nppiCbYCr422ToBGR_709HDTV_8u_C2C4R_Ctx(sp, ss, d, ds, r, a, c); }, 0xC3D0B);
}

TEST(NppiC3ColorInvDiff, PhotoYcc) {
  packed("nppiRGBToYCC_8u_C3R", "rgb2ycc/16x12", 16, 12, 3, nppiRGBToYCC_8u_C3R_Ctx, 0xC3D0C);
  packed("nppiRGBToYCC_8u_C3R", "rgb2ycc/17x13", 17, 13, 3, nppiRGBToYCC_8u_C3R_Ctx, 0xC3D0D);
  packed("nppiRGBToYCC_8u_AC4R", "rgb2ycc/ac4/16x12", 16, 12, 4, nppiRGBToYCC_8u_AC4R_Ctx, 0xC3D0E);
  packed("nppiYCCToRGB_8u_C3R", "ycc2rgb/16x12", 16, 12, 3, nppiYCCToRGB_8u_C3R_Ctx, 0xC3D0F);
  packed("nppiYCCToRGB_8u_AC4R", "ycc2rgb/ac4/16x12", 16, 12, 4, nppiYCCToRGB_8u_AC4R_Ctx, 0xC3D10);
}

TEST(NppiC3ColorInvDiff, JpegCmyk) {
  cmykPacked("nppiCMYKOrYCCKToRGB_JPEG_8u_P4C3R", "cmyk2rgb/16x12", 16, 12, 3,
             nppiCMYKOrYCCKToRGB_JPEG_8u_P4C3R_Ctx, 0xC3D11);
  cmykPacked("nppiCMYKOrYCCKToBGR_JPEG_8u_P4C3R", "cmyk2bgr/16x12", 16, 12, 3,
             nppiCMYKOrYCCKToBGR_JPEG_8u_P4C3R_Ctx, 0xC3D12);
  cmykPlanar("nppiCMYKOrYCCKToRGB_JPEG_8u_P4P3R", "cmyk2rgb/p3/17x13", 17, 13,
             nppiCMYKOrYCCKToRGB_JPEG_8u_P4P3R_Ctx, 0xC3D13);
  cmykPlanar("nppiCMYKOrYCCKToBGR_JPEG_8u_P4P3R", "cmyk2bgr/p3/16x12", 16, 12,
             nppiCMYKOrYCCKToBGR_JPEG_8u_P4P3R_Ctx, 0xC3D14);
  ycckPlanar("nppiYCCKToCMYK_JPEG_601_8u_P4R", "ycck2cmyk/16x12", 16, 12,
             nppiYCCKToCMYK_JPEG_601_8u_P4R_Ctx, 0xC3D15);
}

TEST(NppiC3ColorDiff, ErrorAndProbePaths) {
  const int w = 16, h = 12;
  const int yStep = w + 5, cStep = w / 2 + 6;
  Planes p3 = make_planes({(size_t)yStep * h, (size_t)cStep * h, (size_t)cStep * h}, 0xC3E00);
  Planes p4 = make_planes({(size_t)yStep * h, (size_t)yStep * h, (size_t)yStep * h, (size_t)yStep * h}, 0xC3E03);
  const Npp8u *c4[4] = {p4.ptrs[0], p4.ptrs[1], p4.ptrs[2], p4.ptrs[3]};
  DeviceMemory<Npp8u> packed((size_t)(4 * w + 8) * h);
  packed.copyFromHost(make_uniform<Npp8u>((size_t)(4 * w + 8) * h, 0xC3E01));
  DeviceMemory<Npp8u> src((size_t)(4 * w + 8) * h);
  src.copyFromHost(make_uniform<Npp8u>((size_t)(4 * w + 8) * h, 0xC3E02));
  int steps[3] = {yStep, cStep, cStep};
  NppStreamContext ctx = current_ctx();
  auto probe = [&](const std::string &tag, const char *fname, auto &&call) {
    clear_cuda_error();
    NppStatus st = call();
    Case c("nppiC3Color/err/" + tag, fname);
    c.status(st);
    c.finish();
  };
  probe("fwd420_null_src", "nppiBGRToYCbCr420_709CSC_8u_C3P3R",
        [&] { return nppiBGRToYCbCr420_709CSC_8u_C3P3R_Ctx(nullptr, 3 * w, p3.ptrs.data(), steps, NppiSize{w, h}, ctx); });
  probe("fwd420_neg_roi", "nppiBGRToYCbCr420_709CSC_8u_C3P3R",
        [&] { return nppiBGRToYCbCr420_709CSC_8u_C3P3R_Ctx(src.get(), 3 * w, p3.ptrs.data(), steps, NppiSize{-2, h},
                                                          ctx); });
  probe("fwd420_zero_roi", "nppiBGRToYCbCr420_709CSC_8u_C3P3R",
        [&] { return nppiBGRToYCbCr420_709CSC_8u_C3P3R_Ctx(src.get(), 3 * w, p3.ptrs.data(), steps, NppiSize{0, 0},
                                                          ctx); });
  probe("fwd420_small_src_step", "nppiBGRToYCbCr420_709CSC_8u_C3P3R",
        [&] { return nppiBGRToYCbCr420_709CSC_8u_C3P3R_Ctx(src.get(), 3 * w - 1, p3.ptrs.data(), steps, NppiSize{w, h},
                                                          ctx); });
  probe("fwdC2_null_src", "nppiRGBToYCrCb422_8u_C3C2R",
        [&] { return nppiRGBToYCrCb422_8u_C3C2R_Ctx(nullptr, 3 * w, packed.get(), 2 * w + 8, NppiSize{w, h}, ctx); });
  probe("fwdC2_small_src_step", "nppiRGBToYCrCb422_8u_C3C2R",
        [&] { return nppiRGBToYCrCb422_8u_C3C2R_Ctx(src.get(), 3 * w - 1, packed.get(), 2 * w + 8, NppiSize{w, h}, ctx); });
  probe("fwdC2_small_dst_step", "nppiRGBToYCrCb422_8u_C3C2R",
        [&] { return nppiRGBToYCrCb422_8u_C3C2R_Ctx(src.get(), 3 * w, packed.get(), 2 * w - 1, NppiSize{w, h}, ctx); });
  probe("fwdP3C2_small_src_step", "nppiRGBToYCrCb422_8u_P3C2R",
        [&] { return nppiRGBToYCrCb422_8u_P3C2R_Ctx(sbuf(p3), w - 1, packed.get(), 2 * w + 8, NppiSize{w, h}, ctx); });
  probe("fwdP3C2_small_dst_step", "nppiRGBToYCrCb422_8u_P3C2R",
        [&] { return nppiRGBToYCrCb422_8u_P3C2R_Ctx(sbuf(p3), w, packed.get(), 2 * w - 1, NppiSize{w, h}, ctx); });
  probe("inv420_neg_roi", "nppiYCbCr420ToBGR_8u_P3C4R",
        [&] { return nppiYCbCr420ToBGR_8u_P3C4R_Ctx(sbuf(p3), steps, packed.get(), 4 * w + 8, NppiSize{-2, h}, 9, ctx); });
  probe("inv420_small_src_y_step", "nppiYCbCr420ToBGR_8u_P3C4R",
        [&] {
          int s[3] = {w - 1, cStep, cStep};
          return nppiYCbCr420ToBGR_8u_P3C4R_Ctx(sbuf(p3), s, packed.get(), 4 * w + 8, NppiSize{w, h}, 9, ctx);
        });
  probe("inv420_small_dst_step", "nppiYCbCr420ToBGR_8u_P3C4R",
        [&] { return nppiYCbCr420ToBGR_8u_P3C4R_Ctx(sbuf(p3), steps, packed.get(), 4 * w - 1, NppiSize{w, h}, 9, ctx); });
  probe("inv422_small_src_step", "nppiYCbCrToBGR_709CSC_8u_P3C3R",
        [&] { return nppiYCbCrToBGR_709CSC_8u_P3C3R_Ctx(sbuf(p3), yStep - 1, packed.get(), 3 * w + 8, NppiSize{w, h}, ctx); });
  probe("inv422_small_dst_step", "nppiYCbCrToBGR_709CSC_8u_P3C3R",
        [&] { return nppiYCbCrToBGR_709CSC_8u_P3C3R_Ctx(sbuf(p3), yStep, packed.get(), 3 * w - 1, NppiSize{w, h}, ctx); });
  probe("invC2_null_src", "nppiCbYCr422ToBGR_709HDTV_8u_C2C3R",
        [&] { return nppiCbYCr422ToBGR_709HDTV_8u_C2C3R_Ctx(nullptr, 2 * w + 8, packed.get(), 3 * w + 8, NppiSize{w, h},
                                                           ctx); });
  probe("invC2_small_src_step", "nppiCbYCr422ToBGR_709HDTV_8u_C2C3R",
        [&] { return nppiCbYCr422ToBGR_709HDTV_8u_C2C3R_Ctx(src.get(), 2 * w - 1, packed.get(), 3 * w + 8,
                                                           NppiSize{w, h}, ctx); });
  probe("invC2_small_dst_step", "nppiCbYCr422ToBGR_709HDTV_8u_C2C3R",
        [&] { return nppiCbYCr422ToBGR_709HDTV_8u_C2C3R_Ctx(src.get(), 2 * w + 8, packed.get(), 3 * w - 1,
                                                           NppiSize{w, h}, ctx); });
  probe("ycc_fwd_null_src", "nppiRGBToYCC_8u_C3R",
        [&] { return nppiRGBToYCC_8u_C3R_Ctx(nullptr, 3 * w, packed.get(), 3 * w, NppiSize{w, h}, ctx); });
  probe("ycc_fwd_small_src_step", "nppiRGBToYCC_8u_C3R",
        [&] { return nppiRGBToYCC_8u_C3R_Ctx(src.get(), 3 * w - 1, packed.get(), 3 * w, NppiSize{w, h}, ctx); });
  probe("ycc_inv_small_src_step", "nppiYCCToRGB_8u_C3R",
        [&] { return nppiYCCToRGB_8u_C3R_Ctx(src.get(), 3 * w - 1, packed.get(), 3 * w, NppiSize{w, h}, ctx); });
  probe("cmyk_small_src_step", "nppiCMYKOrYCCKToRGB_JPEG_8u_P4C3R",
        [&] { return nppiCMYKOrYCCKToRGB_JPEG_8u_P4C3R_Ctx(c4, w - 1, packed.get(), 3 * w + 8, NppiSize{w, h}, ctx); });
  probe("cmyk_small_dst_step", "nppiCMYKOrYCCKToRGB_JPEG_8u_P4C3R",
        [&] { return nppiCMYKOrYCCKToRGB_JPEG_8u_P4C3R_Ctx(c4, w, packed.get(), 3 * w - 1, NppiSize{w, h}, ctx); });
  probe("ycck_small_src_step", "nppiYCCKToCMYK_JPEG_601_8u_P4R",
        [&] { return nppiYCCKToCMYK_JPEG_601_8u_P4R_Ctx(c4, w - 1, p4.ptrs.data(), w, NppiSize{w, h}, ctx); });
}
