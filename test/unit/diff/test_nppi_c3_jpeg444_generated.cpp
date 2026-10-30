// Differential tests for c3 JPEG 4:4:4 YCbCr conversions:
//   nppiRGBToYCbCr444_JPEG / nppiBGRToYCbCr444_JPEG (C3P3R, P3R)
//   nppiYCbCr444ToRGB_JPEG / nppiYCbCr444ToBGR_JPEG (P3C3R, P3R)
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiC3Jpeg444Diff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records ./diff_tests --gtest_filter='NppiC3Jpeg444Diff.*'

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

Planes make_planes(int n, size_t bytes, uint64_t seed) {
  Planes p;
  for (int i = 0; i < n; ++i) {
    std::vector<Npp8u> host = make_uniform<Npp8u>(bytes, seed + 7 * i);
    p.dev.push_back(std::make_unique<DeviceMemory<Npp8u>>(bytes));
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

// Forward: packed RGB/BGR -> 3 planes.  Covers padded steps, a tiny ROI and
// the documented error paths.
void fwd_packed(const char *fname, const std::string &key, int width, int height, bool bgr,
                NppStatus (*f)(const Npp8u *, int, Npp8u **, int, NppiSize, NppStreamContext), uint64_t seed) {
  const int sStep = width * 3 + 11;
  const int dStep = width + 9;
  const size_t se = (size_t)sStep * height;
  const size_t de = (size_t)dStep * height;
  std::vector<Npp8u> hs = make_uniform<Npp8u>(se, seed + 11);
  DeviceMemory<Npp8u> src(se);
  src.copyFromHost(hs);
  Planes dst = make_planes(3, de, seed + 29);
  clear_cuda_error();
  NppStatus st = f(src.get(), sStep, dst.ptrs.data(), dStep, NppiSize{width, height}, current_ctx());
  Case c("nppiC3Jpeg444/" + key, fname);
  c.input(hs.data(), hs.size());
  c.status(st);
  capture_planes(c, dst, de);
  c.finish();
}

// Forward: planar RGB/BGR -> 3 planes.
void fwd_planar(const char *fname, const std::string &key, int width, int height,
                NppStatus (*f)(const Npp8u *const *, int, Npp8u **, int, NppiSize, NppStreamContext),
                uint64_t seed) {
  const int sStep = width + 13;
  const int dStep = width + 5;
  const size_t se = (size_t)sStep * height;
  const size_t de = (size_t)dStep * height;
  Planes src = make_planes(3, se, seed + 11);
  Planes dst = make_planes(3, de, seed + 29);
  clear_cuda_error();
  NppStatus st = f(const_cast<const Npp8u **>(src.ptrs.data()), sStep, dst.ptrs.data(), dStep,
                   NppiSize{width, height}, current_ctx());
  Case c("nppiC3Jpeg444/" + key, fname);
  c.status(st);
  capture_planes(c, dst, de);
  c.finish();
}

// Inverse: 3 planes -> packed RGB/BGR.
void inv_packed(const char *fname, const std::string &key, int width, int height,
                NppStatus (*f)(const Npp8u *const *, int, Npp8u *, int, NppiSize, NppStreamContext), uint64_t seed) {
  const int sStep = width + 7;
  const int dStep = width * 3 + 5;
  const size_t se = (size_t)sStep * height;
  const size_t de = (size_t)dStep * height;
  Planes src = make_planes(3, se, seed + 11);
  std::vector<Npp8u> hd = make_uniform<Npp8u>(de, seed + 29);
  DeviceMemory<Npp8u> dst(de);
  dst.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st = f(const_cast<const Npp8u **>(src.ptrs.data()), sStep, dst.get(), dStep, NppiSize{width, height},
                   current_ctx());
  Case c("nppiC3Jpeg444/" + key, fname);
  c.input(hd.data(), hd.size());
  c.status(st);
  c.device_output<Npp8u>(dst.get(), de, ValueKind::U8);
  c.finish();
}

// Inverse: 3 planes -> 3 planes.
void inv_planar(const char *fname, const std::string &key, int width, int height,
                NppStatus (*f)(const Npp8u *const *, int, Npp8u **, int, NppiSize, NppStreamContext),
                uint64_t seed) {
  const int sStep = width + 3;
  const int dStep = width + 17;
  const size_t se = (size_t)sStep * height;
  const size_t de = (size_t)dStep * height;
  Planes src = make_planes(3, se, seed + 11);
  Planes dst = make_planes(3, de, seed + 29);
  clear_cuda_error();
  NppStatus st = f(const_cast<const Npp8u **>(src.ptrs.data()), sStep, dst.ptrs.data(), dStep,
                   NppiSize{width, height}, current_ctx());
  Case c("nppiC3Jpeg444/" + key, fname);
  c.status(st);
  capture_planes(c, dst, de);
  c.finish();
}

void error_cases(const char *fname, const std::string &key,
                 NppStatus (*f)(const Npp8u *const *, int, Npp8u **, int, NppiSize, NppStreamContext),
                 uint64_t seed) {
  const int w = 8, h = 4, step = 16;
  Planes src = make_planes(3, (size_t)step * h, seed);
  Planes dst = make_planes(3, (size_t)step * h, seed + 1);
  auto run = [&](const std::string &tag, const Npp8u **s, int ss, Npp8u **d, int ds, NppiSize roi) {
    clear_cuda_error();
    NppStatus st = f(s, ss, d, ds, roi, current_ctx());
    Case c("nppiC3Jpeg444/" + key + "/" + tag, fname);
    c.status(st);
    c.finish();
  };
  Npp8u **sp = dst.ptrs.data();
  run("null_src", nullptr, step, dst.ptrs.data(), step, NppiSize{w, h});
  const Npp8u **nullsrc = nullptr;
  run("null_src_ptr", nullsrc, step, dst.ptrs.data(), step, NppiSize{w, h});
  run("null_dst", const_cast<const Npp8u **>(src.ptrs.data()), step, nullptr, step, NppiSize{w, h});
  run("zero_step", const_cast<const Npp8u **>(src.ptrs.data()), 0, dst.ptrs.data(), step, NppiSize{w, h});
  run("neg_step", const_cast<const Npp8u **>(src.ptrs.data()), -4, dst.ptrs.data(), step, NppiSize{w, h});
  run("neg_roi", const_cast<const Npp8u **>(src.ptrs.data()), step, dst.ptrs.data(), step, NppiSize{-2, 4});
  run("empty_roi", const_cast<const Npp8u **>(src.ptrs.data()), step, dst.ptrs.data(), step, NppiSize{0, 0});
  (void)sp;
}

} // namespace

TEST(NppiC3Jpeg444Diff, RgbToYCbCr444Jpeg) {
  fwd_packed("nppiRGBToYCbCr444_JPEG_8u_C3P3R", "fwd_c3/17x13", 17, 13, false, nppiRGBToYCbCr444_JPEG_8u_C3P3R_Ctx,
             0xC0FFEE1);
  fwd_packed("nppiRGBToYCbCr444_JPEG_8u_C3P3R", "fwd_c3/6x4", 6, 4, false, nppiRGBToYCbCr444_JPEG_8u_C3P3R_Ctx,
             0xC0FFEE2);
  fwd_packed("nppiRGBToYCbCr444_JPEG_8u_C3P3R", "fwd_c3/1x1", 1, 1, false, nppiRGBToYCbCr444_JPEG_8u_C3P3R_Ctx,
             0xC0FFEE3);
  fwd_planar("nppiRGBToYCbCr444_JPEG_8u_P3R", "fwd_p3/17x13", 17, 13, nppiRGBToYCbCr444_JPEG_8u_P3R_Ctx, 0xC0FFEE4);
  fwd_planar("nppiRGBToYCbCr444_JPEG_8u_P3R", "fwd_p3/6x4", 6, 4, nppiRGBToYCbCr444_JPEG_8u_P3R_Ctx, 0xC0FFEE5);
}

TEST(NppiC3Jpeg444Diff, BgrToYCbCr444Jpeg) {
  fwd_packed("nppiBGRToYCbCr444_JPEG_8u_C3P3R", "fwd_c3/17x13", 17, 13, true,
             nppiBGRToYCbCr444_JPEG_8u_C3P3R_Ctx, 0xC0FFEE6);
  fwd_planar("nppiBGRToYCbCr444_JPEG_8u_P3R", "fwd_p3/17x13", 17, 13, nppiBGRToYCbCr444_JPEG_8u_P3R_Ctx, 0xC0FFEE7);
}

TEST(NppiC3Jpeg444Diff, YCbCr444ToRgbJpeg) {
  inv_packed("nppiYCbCr444ToRGB_JPEG_8u_P3C3R", "inv_c3/17x13", 17, 13, nppiYCbCr444ToRGB_JPEG_8u_P3C3R_Ctx,
             0xC0FFEE8);
  inv_packed("nppiYCbCr444ToRGB_JPEG_8u_P3C3R", "inv_c3/6x4", 6, 4, nppiYCbCr444ToRGB_JPEG_8u_P3C3R_Ctx, 0xC0FFEE9);
  inv_planar("nppiYCbCr444ToRGB_JPEG_8u_P3R", "inv_p3/17x13", 17, 13, nppiYCbCr444ToRGB_JPEG_8u_P3R_Ctx, 0xC0FFEEA);
}

TEST(NppiC3Jpeg444Diff, YCbCr444ToBgrJpeg) {
  inv_packed("nppiYCbCr444ToBGR_JPEG_8u_P3C3R", "inv_c3/17x13", 17, 13, nppiYCbCr444ToBGR_JPEG_8u_P3C3R_Ctx,
             0xC0FFEEB);
  inv_planar("nppiYCbCr444ToBGR_JPEG_8u_P3R", "inv_p3/17x13", 17, 13, nppiYCbCr444ToBGR_JPEG_8u_P3R_Ctx, 0xC0FFEEC);
}

TEST(NppiC3Jpeg444Diff, ErrorPaths) {
  error_cases("nppiYCbCr444ToRGB_JPEG_8u_P3R", "err/inv", nppiYCbCr444ToRGB_JPEG_8u_P3R_Ctx, 0xC0FFEED);
  error_cases("nppiRGBToYCbCr444_JPEG_8u_P3R", "err/fwd", nppiRGBToYCbCr444_JPEG_8u_P3R_Ctx, 0xC0FFEEE);
}
