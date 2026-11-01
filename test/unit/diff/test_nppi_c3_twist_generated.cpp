// Differential tests for the c3 ColorTwist sampling-format conversions
// (nppi_color_conversion.h) and the plain nppiYCrCb422ToRGB family.
//
// Measured NVIDIA NPP 13.4 (RTX 5090) semantics:
//   dst[c] = aTwist[c][0]*src[0] + aTwist[c][1]*src[1] + aTwist[c][2]*src[2]
//            + aTwist[c][3], rounded to nearest and saturated.
//   * 4:2:0 chroma is the rounded average of the twisted per-pixel values
//     over each 2x2 block; luma is per pixel.
//   * 4:2:2 packed layouts are [U Y0 V Y1]; planar chroma is pair averaged.
//   * Even width and height are required for the 4:2:0 variants; the tests
//     stay on even extents (the reference's odd-extent warning paths
//     re-shape the ROI and are not reproducible with our geometry).
//   * The twist entry points only validate null sources and negative ROIs;
//     strides are not checked (measured), so no stride probes are recorded.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiC3Twist*Diff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records ./diff_tests --gtest_filter='NppiC3Twist*Diff.*'

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

void twist_matrix(Npp32f m[3][4], uint64_t seed) {
  // Deterministic, non-trivial matrix with a broad coefficient range.
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      uint64_t v = seed * 6364136223846793005ULL + 1442695040888963407ULL;
      seed = v;
      const double unit = static_cast<double>((v >> 33) & 0xffff) / 65535.0;
      m[i][j] = static_cast<Npp32f>(j == 3 ? unit * 300.0 - 40.0 : unit * 2.6 - 1.3);
    }
  }
}

// --- ColorTwist forward 4:2:0 ---
void fwd420(const char *fname, const std::string &key, int w, int h,
            NppStatus (*f)(const Npp8u *, int, Npp8u **, int *, NppiSize, const Npp32f[3][4], NppStreamContext),
            uint64_t seed) {
  const int sStep = 3 * w + 6;
  const int yStep = w + 4;
  const int cStep = w / 2 + 5;
  DeviceMemory<Npp8u> src((size_t)sStep * h);
  src.copyFromHost(make_uniform<Npp8u>((size_t)sStep * h, seed + 11));
  Planes dst = make_planes({(size_t)yStep * h, (size_t)cStep * h, (size_t)cStep * h}, seed + 29);
  Npp32f m[3][4];
  twist_matrix(m, seed + 3);
  int steps[3] = {yStep, cStep, cStep};
  clear_cuda_error();
  NppStatus st = f(src.get(), sStep, dst.ptrs.data(), steps, NppiSize{w, h}, m, current_ctx());
  Case c("nppiC3Twist/" + key, fname);
  c.status(st);
  c.device_output<Npp8u>(dst.dev[0]->get(), (size_t)yStep * h, ValueKind::U8);
  c.device_output<Npp8u>(dst.dev[1]->get(), (size_t)cStep * (h / 2), ValueKind::U8);
  c.device_output<Npp8u>(dst.dev[2]->get(), (size_t)cStep * (h / 2), ValueKind::U8);
  c.finish();
}

void fwd420P3(const char *fname, const std::string &key, int w, int h,
              NppStatus (*f)(const Npp8u *const *, int *, Npp8u **, int *, NppiSize, const Npp32f[3][4],
                             NppStreamContext),
              uint64_t seed) {
  const int sStep = w + 4;
  const int yStep = w + 4;
  const int cStep = w / 2 + 5;
  Planes src = make_planes({(size_t)sStep * h, (size_t)sStep * h, (size_t)sStep * h}, seed + 11);
  Planes dst = make_planes({(size_t)yStep * h, (size_t)cStep * h, (size_t)cStep * h}, seed + 29);
  Npp32f m[3][4];
  twist_matrix(m, seed + 3);
  int ssteps[3] = {sStep, sStep, sStep};
  int dsteps[3] = {yStep, cStep, cStep};
  clear_cuda_error();
  NppStatus st = f(sbuf(src), ssteps, dst.ptrs.data(), dsteps, NppiSize{w, h}, m, current_ctx());
  Case c("nppiC3Twist/" + key, fname);
  c.status(st);
  c.device_output<Npp8u>(dst.dev[0]->get(), (size_t)yStep * h, ValueKind::U8);
  c.device_output<Npp8u>(dst.dev[1]->get(), (size_t)cStep * (h / 2), ValueKind::U8);
  c.device_output<Npp8u>(dst.dev[2]->get(), (size_t)cStep * (h / 2), ValueKind::U8);
  c.finish();
}

// --- ColorTwist forward 4:2:2 (packed or planar destination) ---
void fwd422C2(const char *fname, const std::string &key, int w, int h,
              NppStatus (*f)(const Npp8u *, int, Npp8u *, int, NppiSize, const Npp32f[3][4], NppStreamContext),
              uint64_t seed) {
  const int sStep = 3 * w + 6;
  const int dStep = 2 * w + 8;
  DeviceMemory<Npp8u> src((size_t)sStep * h);
  src.copyFromHost(make_uniform<Npp8u>((size_t)sStep * h, seed + 11));
  DeviceMemory<Npp8u> dst((size_t)dStep * h);
  dst.copyFromHost(make_uniform<Npp8u>((size_t)dStep * h, seed + 29));
  Npp32f m[3][4];
  twist_matrix(m, seed + 3);
  clear_cuda_error();
  NppStatus st = f(src.get(), sStep, dst.get(), dStep, NppiSize{w, h}, m, current_ctx());
  Case c("nppiC3Twist/" + key, fname);
  c.status(st);
  c.device_output<Npp8u>(dst.get(), (size_t)dStep * h, ValueKind::U8);
  c.finish();
}

void fwd422P3(const char *fname, const std::string &key, int w, int h,
              NppStatus (*f)(const Npp8u *const *, int *, Npp8u **, int *, NppiSize, const Npp32f[3][4],
                             NppStreamContext),
              uint64_t seed) {
  const int sStep = w + 4;
  const int cStep = w / 2 + 5;
  Planes src = make_planes({(size_t)sStep * h, (size_t)sStep * h, (size_t)sStep * h}, seed + 11);
  Planes dst = make_planes({(size_t)sStep * h, (size_t)cStep * h, (size_t)cStep * h}, seed + 29);
  Npp32f m[3][4];
  twist_matrix(m, seed + 3);
  int ssteps[3] = {sStep, sStep, sStep};
  int dsteps[3] = {sStep, cStep, cStep};
  clear_cuda_error();
  NppStatus st = f(sbuf(src), ssteps, dst.ptrs.data(), dsteps, NppiSize{w, h}, m, current_ctx());
  Case c("nppiC3Twist/" + key, fname);
  c.status(st);
  c.device_output<Npp8u>(dst.dev[0]->get(), (size_t)sStep * h, ValueKind::U8);
  c.device_output<Npp8u>(dst.dev[1]->get(), (size_t)cStep * h, ValueKind::U8);
  c.device_output<Npp8u>(dst.dev[2]->get(), (size_t)cStep * h, ValueKind::U8);
  c.finish();
}

// packed RGB source -> planar 4:2:2
void fwd422C3P3(const char *fname, const std::string &key, int w, int h,
                NppStatus (*f)(const Npp8u *, int, Npp8u **, int *, NppiSize, const Npp32f[3][4], NppStreamContext),
                uint64_t seed) {
  const int sStep = 3 * w + 6;
  const int cStep = w / 2 + 5;
  DeviceMemory<Npp8u> src((size_t)sStep * h);
  src.copyFromHost(make_uniform<Npp8u>((size_t)sStep * h, seed + 11));
  Planes dst = make_planes({(size_t)(w + 4) * h, (size_t)cStep * h, (size_t)cStep * h}, seed + 29);
  Npp32f m[3][4];
  twist_matrix(m, seed + 3);
  int dsteps[3] = {w + 4, cStep, cStep};
  clear_cuda_error();
  NppStatus st = f(src.get(), sStep, dst.ptrs.data(), dsteps, NppiSize{w, h}, m, current_ctx());
  Case c("nppiC3Twist/" + key, fname);
  c.status(st);
  c.device_output<Npp8u>(dst.dev[0]->get(), (size_t)(w + 4) * h, ValueKind::U8);
  c.device_output<Npp8u>(dst.dev[1]->get(), (size_t)cStep * h, ValueKind::U8);
  c.device_output<Npp8u>(dst.dev[2]->get(), (size_t)cStep * h, ValueKind::U8);
  c.finish();
}

// --- ColorTwist inverse 4:2:0 ---
template <class FN>
void inv420(const char *fname, const std::string &key, int w, int h, int dstPp, Npp8u alpha, FN f, uint64_t seed) {
  const int yStep = w + 4;
  const int cStep = w / 2 + 5;
  const int ch = h / 2;
  Planes src = make_planes({(size_t)yStep * h, (size_t)cStep * ch, (size_t)cStep * ch}, seed + 11);
  DeviceMemory<Npp8u> dst((size_t)(dstPp * w + 8) * h);
  dst.copyFromHost(make_uniform<Npp8u>((size_t)(dstPp * w + 8) * h, seed + 29));
  Npp32f m[3][4];
  twist_matrix(m, seed + 3);
  int ssteps[3] = {yStep, cStep, cStep};
  clear_cuda_error();
  NppStatus st = f(sbuf(src), ssteps, dst.get(), dstPp * w + 8, NppiSize{w, h}, m, alpha, current_ctx());
  Case c("nppiC3Twist/" + key, fname);
  c.status(st);
  c.device_output<Npp8u>(dst.get(), (size_t)(dstPp * w + 8) * h, ValueKind::U8);
  c.finish();
}

void inv420P3(const char *fname, const std::string &key, int w, int h,
              NppStatus (*f)(const Npp8u *const *, int *, Npp8u **, int *, NppiSize, const Npp32f[3][4],
                             NppStreamContext),
              uint64_t seed) {
  const int yStep = w + 4;
  const int cStep = w / 2 + 5;
  const int ch = h / 2;
  Planes src = make_planes({(size_t)yStep * h, (size_t)cStep * ch, (size_t)cStep * ch}, seed + 11);
  Planes dst = make_planes({(size_t)yStep * h, (size_t)yStep * h, (size_t)yStep * h}, seed + 29);
  Npp32f m[3][4];
  twist_matrix(m, seed + 3);
  int ssteps[3] = {yStep, cStep, cStep};
  int dsteps[3] = {yStep, yStep, yStep};
  clear_cuda_error();
  NppStatus st = f(sbuf(src), ssteps, dst.ptrs.data(), dsteps, NppiSize{w, h}, m, current_ctx());
  Case c("nppiC3Twist/" + key, fname);
  c.status(st);
  c.device_output<Npp8u>(dst.dev[0]->get(), (size_t)yStep * h, ValueKind::U8);
  c.device_output<Npp8u>(dst.dev[1]->get(), (size_t)yStep * h, ValueKind::U8);
  c.device_output<Npp8u>(dst.dev[2]->get(), (size_t)yStep * h, ValueKind::U8);
  c.finish();
}

// --- ColorTwist inverse 4:2:2 ---
void inv422P3(const char *fname, const std::string &key, int w, int h,
              NppStatus (*f)(const Npp8u *const *, int *, Npp8u **, int *, NppiSize, const Npp32f[3][4],
                             NppStreamContext),
              uint64_t seed) {
  const int sStep = w + 4;
  const int cStep = w / 2 + 5;
  Planes src = make_planes({(size_t)sStep * h, (size_t)cStep * h, (size_t)cStep * h}, seed + 11);
  Planes dst = make_planes({(size_t)sStep * h, (size_t)sStep * h, (size_t)sStep * h}, seed + 29);
  Npp32f m[3][4];
  twist_matrix(m, seed + 3);
  int ssteps[3] = {sStep, cStep, cStep};
  int dsteps[3] = {sStep, sStep, sStep};
  clear_cuda_error();
  NppStatus st = f(sbuf(src), ssteps, dst.ptrs.data(), dsteps, NppiSize{w, h}, m, current_ctx());
  Case c("nppiC3Twist/" + key, fname);
  c.status(st);
  for (int i = 0; i < 3; ++i) {
    c.device_output<Npp8u>(dst.dev[i]->get(), (size_t)sStep * h, ValueKind::U8);
  }
  c.finish();
}

template <class FN>
void inv422C(const char *fname, const std::string &key, int w, int h, int dstPp, Npp8u alpha, FN f, uint64_t seed) {
  const int sStep = w + 4;
  const int cStep = w / 2 + 5;
  const int dStep = dstPp * w + 8;
  Planes src = make_planes({(size_t)sStep * h, (size_t)cStep * h, (size_t)cStep * h}, seed + 11);
  DeviceMemory<Npp8u> dst((size_t)dStep * h);
  dst.copyFromHost(make_uniform<Npp8u>((size_t)dStep * h, seed + 29));
  Npp32f m[3][4];
  twist_matrix(m, seed + 3);
  int ssteps[3] = {sStep, cStep, cStep};
  clear_cuda_error();
  NppStatus st = f(sbuf(src), ssteps, dst.get(), dStep, NppiSize{w, h}, m, alpha, current_ctx());
  Case c("nppiC3Twist/" + key, fname);
  c.status(st);
  c.device_output<Npp8u>(dst.get(), (size_t)dStep * h, ValueKind::U8);
  c.finish();
}

void inv422C2(const char *fname, const std::string &key, int w, int h,
              NppStatus (*f)(const Npp8u *, int, Npp8u *, int, NppiSize, const Npp32f[3][4], NppStreamContext),
              uint64_t seed) {
  const int sStep = 2 * w + 8;
  const int dStep = 3 * w + 8;
  DeviceMemory<Npp8u> src((size_t)sStep * h);
  src.copyFromHost(make_uniform<Npp8u>((size_t)sStep * h, seed + 11));
  DeviceMemory<Npp8u> dst((size_t)dStep * h);
  dst.copyFromHost(make_uniform<Npp8u>((size_t)dStep * h, seed + 29));
  Npp32f m[3][4];
  twist_matrix(m, seed + 3);
  clear_cuda_error();
  NppStatus st = f(src.get(), sStep, dst.get(), dStep, NppiSize{w, h}, m, current_ctx());
  Case c("nppiC3Twist/" + key, fname);
  c.status(st);
  c.device_output<Npp8u>(dst.get(), (size_t)dStep * h, ValueKind::U8);
  c.finish();
}

// --- plain YCrCb422 -> RGB ---
void ycrcbC2(const char *fname, const std::string &key, int w, int h,
             NppStatus (*f)(const Npp8u *, int, Npp8u *, int, NppiSize, NppStreamContext), uint64_t seed) {
  const int sStep = 2 * w + 8;
  const int dStep = 3 * w + 8;
  DeviceMemory<Npp8u> src((size_t)sStep * h);
  src.copyFromHost(make_uniform<Npp8u>((size_t)sStep * h, seed + 11));
  DeviceMemory<Npp8u> dst((size_t)dStep * h);
  dst.copyFromHost(make_uniform<Npp8u>((size_t)dStep * h, seed + 29));
  clear_cuda_error();
  NppStatus st = f(src.get(), sStep, dst.get(), dStep, NppiSize{w, h}, current_ctx());
  Case c("nppiC3Twist/" + key, fname);
  c.status(st);
  c.device_output<Npp8u>(dst.get(), (size_t)dStep * h, ValueKind::U8);
  c.finish();
}

void ycrcbP3(const char *fname, const std::string &key, int w, int h,
             NppStatus (*f)(const Npp8u *, int, Npp8u **, int, NppiSize, NppStreamContext), uint64_t seed) {
  const int sStep = 2 * w + 8;
  const int dStep = w + 5;
  DeviceMemory<Npp8u> src((size_t)sStep * h);
  src.copyFromHost(make_uniform<Npp8u>((size_t)sStep * h, seed + 11));
  Planes dst = make_planes({(size_t)dStep * h, (size_t)dStep * h, (size_t)dStep * h}, seed + 29);
  clear_cuda_error();
  NppStatus st = f(src.get(), sStep, dst.ptrs.data(), dStep, NppiSize{w, h}, current_ctx());
  Case c("nppiC3Twist/" + key, fname);
  c.status(st);
  for (int i = 0; i < 3; ++i) {
    c.device_output<Npp8u>(dst.dev[i]->get(), (size_t)dStep * h, ValueKind::U8);
  }
  c.finish();
}

} // namespace

TEST(NppiC3TwistFwdDiff, Yuv420) {
  fwd420("nppiRGBToYUV420_8u_ColorTwist32f_C3P3R", "fwd420c3/16x12", 16, 12,
         nppiRGBToYUV420_8u_ColorTwist32f_C3P3R_Ctx, 0xD3A01);
  fwd420("nppiRGBToYUV420_8u_ColorTwist32f_C3P3R", "fwd420c3/8x6", 8, 6,
         nppiRGBToYUV420_8u_ColorTwist32f_C3P3R_Ctx, 0xD3A02);
  fwd420P3("nppiRGBToYUV420_8u_ColorTwist32f_P3R", "fwd420p3/16x12", 16, 12,
           nppiRGBToYUV420_8u_ColorTwist32f_P3R_Ctx, 0xD3A03);
  fwd420P3("nppiRGBToYUV420_8u_ColorTwist32f_P3R", "fwd420p3/32x16", 32, 16,
           nppiRGBToYUV420_8u_ColorTwist32f_P3R_Ctx, 0xD3A04);
}

TEST(NppiC3TwistFwdDiff, Yuv422) {
  fwd422C2("nppiRGBToYUV422_8u_ColorTwist32f_C3C2R", "fwd422c2/16x12", 16, 12,
           nppiRGBToYUV422_8u_ColorTwist32f_C3C2R_Ctx, 0xD3A05);
  fwd422C2("nppiRGBToYUV422_8u_ColorTwist32f_C3C2R", "fwd422c2/8x6", 8, 6,
           nppiRGBToYUV422_8u_ColorTwist32f_C3C2R_Ctx, 0xD3A06);
  fwd422C3P3("nppiRGBToYUV422_8u_ColorTwist32f_C3P3R", "fwd422c3p3/16x12", 16, 12,
             nppiRGBToYUV422_8u_ColorTwist32f_C3P3R_Ctx, 0xD3A07);
  fwd422P3("nppiRGBToYUV422_8u_ColorTwist32f_P3R", "fwd422p3/16x12", 16, 12,
           nppiRGBToYUV422_8u_ColorTwist32f_P3R_Ctx, 0xD3A08);
}

TEST(NppiC3TwistInvDiff, Yuv420) {
  inv420P3("nppiYUV420ToRGB_8u_ColorTwist32f_P3R", "inv420p3/16x12", 16, 12,
           nppiYUV420ToRGB_8u_ColorTwist32f_P3R_Ctx, 0xD3A09);
  inv420("nppiYUV420ToRGB_8u_ColorTwist32f_P3C3R", "inv420c3/16x12", 16, 12, 3, 0,
         [&](const Npp8u *const *p, int *ss, Npp8u *d, int ds, NppiSize r, const Npp32f mm[3][4], Npp8u,
             NppStreamContext c) { return nppiYUV420ToRGB_8u_ColorTwist32f_P3C3R_Ctx(p, ss, d, ds, r, mm, c); },
         0xD3A0A);
  inv420("nppiYUV420ToRGB_8u_ColorTwist32f_P3C4R", "inv420c4/8x6", 8, 6, 4, 255,
         [&](const Npp8u *const *p, int *ss, Npp8u *d, int ds, NppiSize r, const Npp32f mm[3][4], Npp8u,
             NppStreamContext c) { return nppiYUV420ToRGB_8u_ColorTwist32f_P3C4R_Ctx(p, ss, d, ds, r, mm, c); },
         0xD3A0B);
  inv420("nppiYUV420ToRGB_8u_ColorTwist32f_P3AC4R", "inv420ac4/16x12", 16, 12, 4, 37,
         [&](const Npp8u *const *p, int *ss, Npp8u *d, int ds, NppiSize r, const Npp32f mm[3][4], Npp8u a,
             NppStreamContext c) { return nppiYUV420ToRGB_8u_ColorTwist32f_P3AC4R_Ctx(p, ss, d, ds, r, mm, a, c); },
         0xD3A0C);
}

TEST(NppiC3TwistInvDiff, Yuv422) {
  inv422P3("nppiYUV422ToRGB_8u_ColorTwist32f_P3R", "inv422p3/16x12", 16, 12,
           nppiYUV422ToRGB_8u_ColorTwist32f_P3R_Ctx, 0xD3A0D);
  inv422P3("nppiYUV422ToRGB_8u_ColorTwist32f_P3R", "inv422p3/8x6", 8, 6,
           nppiYUV422ToRGB_8u_ColorTwist32f_P3R_Ctx, 0xD3A0E);
  inv422C("nppiYUV422ToRGB_8u_ColorTwist32f_P3C3R", "inv422c3/16x12", 16, 12, 3, 0,
          [&](const Npp8u *const *p, int *ss, Npp8u *d, int ds, NppiSize r, const Npp32f mm[3][4], Npp8u,
              NppStreamContext c) { return nppiYUV422ToRGB_8u_ColorTwist32f_P3C3R_Ctx(p, ss, d, ds, r, mm, c); },
          0xD3A0F);
  inv422C("nppiYUV422ToRGB_8u_ColorTwist32f_P3AC4R", "inv422ac4/8x6", 8, 6, 4, 29,
          [&](const Npp8u *const *p, int *ss, Npp8u *d, int ds, NppiSize r, const Npp32f mm[3][4], Npp8u a,
              NppStreamContext c) { return nppiYUV422ToRGB_8u_ColorTwist32f_P3AC4R_Ctx(p, ss, d, ds, r, mm, a, c); },
          0xD3A10);
  inv422C2("nppiYUV422ToRGB_8u_ColorTwist32f_C2C3R", "inv422c2/16x12", 16, 12,
           nppiYUV422ToRGB_8u_ColorTwist32f_C2C3R_Ctx, 0xD3A11);
}

TEST(NppiC3TwistInvDiff, YCrCb422ToRgb) {
  ycrcbC2("nppiYCrCb422ToRGB_8u_C2C3R", "ycrcb2rgb/c2c3/16x12", 16, 12,
          nppiYCrCb422ToRGB_8u_C2C3R_Ctx, 0xD3A12);
  ycrcbC2("nppiYCrCb422ToRGB_8u_C2C3R", "ycrcb2rgb/c2c3/8x6", 8, 6,
          nppiYCrCb422ToRGB_8u_C2C3R_Ctx, 0xD3A13);
  ycrcbP3("nppiYCrCb422ToRGB_8u_C2P3R", "ycrcb2rgb/c2p3/16x12", 16, 12,
          nppiYCrCb422ToRGB_8u_C2P3R_Ctx, 0xD3A14);
}

TEST(NppiC3TwistDiff, ErrorAndProbePaths) {
  const int w = 16, h = 12;
  const int yStep = w + 4, cStep = w / 2 + 5;
  Planes p3 = make_planes({(size_t)yStep * h, (size_t)cStep * h, (size_t)cStep * h}, 0xD3E00);
  DeviceMemory<Npp8u> packed((size_t)(4 * w + 8) * h);
  packed.copyFromHost(make_uniform<Npp8u>((size_t)(4 * w + 8) * h, 0xD3E01));
  DeviceMemory<Npp8u> src((size_t)(4 * w + 8) * h);
  src.copyFromHost(make_uniform<Npp8u>((size_t)(4 * w + 8) * h, 0xD3E02));
  int steps[3] = {yStep, cStep, cStep};
  Npp32f m[3][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}};
  NppStreamContext ctx = current_ctx();
  auto probe = [&](const std::string &tag, const char *fname, auto &&call) {
    clear_cuda_error();
    NppStatus st = call();
    Case c("nppiC3Twist/err/" + tag, fname);
    c.status(st);
    c.finish();
  };
  probe("fwd420_null_src", "nppiRGBToYUV420_8u_ColorTwist32f_C3P3R",
        [&] { return nppiRGBToYUV420_8u_ColorTwist32f_C3P3R_Ctx(nullptr, 3 * w, p3.ptrs.data(), steps, NppiSize{w, h}, m, ctx); });
  probe("fwd420_neg_roi", "nppiRGBToYUV420_8u_ColorTwist32f_C3P3R",
        [&] { return nppiRGBToYUV420_8u_ColorTwist32f_C3P3R_Ctx(src.get(), 3 * w, p3.ptrs.data(), steps, NppiSize{-2, h}, m, ctx); });
  probe("fwd422_null_src", "nppiRGBToYUV422_8u_ColorTwist32f_C3C2R",
        [&] { return nppiRGBToYUV422_8u_ColorTwist32f_C3C2R_Ctx(nullptr, 3 * w, packed.get(), 2 * w + 8, NppiSize{w, h}, m, ctx); });
  probe("inv420_neg_roi", "nppiYUV420ToRGB_8u_ColorTwist32f_P3C3R",
        [&] { return nppiYUV420ToRGB_8u_ColorTwist32f_P3C3R_Ctx(sbuf(p3), steps, packed.get(), 3 * w + 8, NppiSize{-2, h}, m, ctx); });
  probe("inv422c2_null_src", "nppiYUV422ToRGB_8u_ColorTwist32f_C2C3R",
        [&] { return nppiYUV422ToRGB_8u_ColorTwist32f_C2C3R_Ctx(nullptr, 2 * w + 8, packed.get(), 3 * w + 8, NppiSize{w, h}, m, ctx); });
  probe("ycrcb_null_src", "nppiYCrCb422ToRGB_8u_C2C3R",
        [&] { return nppiYCrCb422ToRGB_8u_C2C3R_Ctx(nullptr, 2 * w + 8, packed.get(), 3 * w + 8, NppiSize{w, h}, ctx); });
  probe("ycrcb_small_src_step", "nppiYCrCb422ToRGB_8u_C2C3R",
        [&] { return nppiYCrCb422ToRGB_8u_C2C3R_Ctx(src.get(), 2 * w - 1, packed.get(), 3 * w + 8, NppiSize{w, h}, ctx); });
}
