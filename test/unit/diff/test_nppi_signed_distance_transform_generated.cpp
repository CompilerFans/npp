// Differential tests for the signed PBA distance transforms.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   tools/remote5090.sh record 'NppiSignedDistanceTransformDiff.*'
//
// The transform is not the Euclidean distance: the reference's
// generateAdditionalRelativeShiftOutput computes, for the displacement
// (dx,dy) to the nearest opposite-class site and C = 30 + cutoff,
//   t = sqrt((max(|dx|+sx, |dy|+sy) - C)^2 + min(...)^2)
// with +t inside and -t outside.  The 30 is a fixed band constant.  The
// three output flavours differ:
//   * signed 32f: voronoi = nearest outside site, indices = nearest
//     opposite site, relative = pixel - site (signed);
//   * abs 32f: sites are the nearest inside pixel, relative is the absolute
//     Manhattan distance, and the transform is positive;
//   * abs 64f without an antialiasing buffer: same as the signed transform
//     but the relative output is absolute.
// The 32f64f abs entry point requires an antialiasing buffer; its AA path is
// not reproduced (only its status contract is checked here).
//
// Test shapes are closed regions fully inside the ROI.  NPP itself faults on
// fragmented masks and on images without an outside class for the abs
// variants, so those are intentionally not recorded.

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

using namespace npp_diff;
using namespace npp_functional_test;

namespace {

NppStreamContext current_context() {
  NppStreamContext ctx{};
  NppStatus st = nppGetStreamContext(&ctx);
  EXPECT_EQ(st, NPP_NO_ERROR) << "nppGetStreamContext failed";
  return ctx;
}

constexpr float kSentinel = -1.0e30f;

enum class Shape { Rect, Circle, Blobs };

std::vector<Npp32f> makeMask(int width, int height, int pad, Shape shape) {
  const int stride = width + pad;
  std::vector<Npp32f> img((size_t)stride * height, 0.0f);
  if (shape == Shape::Rect) {
    for (int y = height / 3; y < 2 * height / 3; ++y)
      for (int x = width / 3; x < 2 * width / 3; ++x) img[(size_t)y * stride + x] = 1.0f;
  } else if (shape == Shape::Circle) {
    const int cx = width / 2, cy = height / 2, r = width / 6;
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x)
        if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r) img[(size_t)y * stride + x] = 1.0f;
  } else {
    const int blobs[][3] = {{width / 4, height / 4, width / 10},
                            {2 * width / 3, height / 3, width / 12},
                            {width / 2, 3 * height / 4, width / 8}};
    for (const auto &b : blobs)
      for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
          if ((x - b[0]) * (x - b[0]) + (y - b[1]) * (y - b[1]) <= b[2] * b[2])
            img[(size_t)y * stride + x] = 1.0f;
  }
  return img;
}

void status_case(const std::string &id, const char *fname, NppStatus st) {
  Case c(id, fname);
  c.status(st);
  c.finish();
}

// variant: 0 = signed 32f, 1 = abs 32f, 2 = abs 64f (no AA)
void profile_case(const std::string &id, const char *fname, int width, int height, int pad, Shape shape,
                  float cutoff, float sx, float sy, int variant) {
  const int stride = width + pad;
  std::vector<Npp32f> img = makeMask(width, height, pad, shape);
  DeviceMemory<Npp32f> src(img.size());
  src.copyFromHost(img);
  std::vector<Npp64f> img64(img.begin(), img.end());
  DeviceMemory<Npp64f> src64(img64.size());
  if (variant == 2) src64.copyFromHost(img64);

  size_t bufferSize = 0;
  if (variant == 0 || variant == 1) {
    EXPECT_EQ(nppiSignedDistanceTransformPBAGetBufferSize({width, height}, &bufferSize), NPP_NO_ERROR);
  } else {
    EXPECT_EQ(nppiSignedDistanceTransformPBAGet64fBufferSize({width, height}, &bufferSize), NPP_NO_ERROR);
  }
  DeviceMemory<Npp8u> buffer(bufferSize > 0 ? bufferSize : 1);

  const int vorStep = width * 4;
  const size_t vorCount = (size_t)vorStep * height / (size_t)sizeof(Npp16s);
  DeviceMemory<Npp16s> voronoi(vorCount), indices(vorCount);
  DeviceMemory<Npp16u> relative(vorCount);
  std::vector<Npp16s> vorInit(vorCount, (Npp16s)-12345);
  std::vector<Npp16u> relInit(vorCount, (Npp16u)54321);
  voronoi.copyFromHost(vorInit);
  indices.copyFromHost(vorInit);
  relative.copyFromHost(relInit);

  DeviceMemory<Npp32f> transform32((size_t)width * height);
  DeviceMemory<Npp64f> transform64((size_t)width * height);

  NppStreamContext ctx = current_context();
  NppStatus st;
  if (variant == 0) {
    st = nppiSignedDistanceTransformPBA_32f_C1R_Ctx(
        src.get(), stride * (int)sizeof(Npp32f), cutoff, sx, sy, voronoi.get(), vorStep, indices.get(), vorStep,
        (Npp16s *)relative.get(), vorStep, transform32.get(), width * (int)sizeof(Npp32f), NppiSize{width, height},
        buffer.get(), ctx);
  } else if (variant == 1) {
    st = nppiSignedDistanceTransformAbsPBA_32f_C1R_Ctx(
        src.get(), stride * (int)sizeof(Npp32f), cutoff, sx, sy, voronoi.get(), vorStep, indices.get(), vorStep,
        relative.get(), vorStep, transform32.get(), width * (int)sizeof(Npp32f), NppiSize{width, height},
        buffer.get(), ctx);
  } else {
    st = nppiSignedDistanceTransformAbsPBA_64f_C1R_Ctx(
        src64.get(), stride * (int)sizeof(Npp64f), cutoff, sx, sy, voronoi.get(), vorStep, indices.get(), vorStep,
        relative.get(), vorStep, transform64.get(), width * (int)sizeof(Npp64f), NppiSize{width, height},
        buffer.get(), nullptr, ctx);
  }
  clear_cuda_error();

  Case c(id, fname);
  c.status(st);
  c.input(img.data(), img.size() * sizeof(Npp32f));
  if (st == NPP_NO_ERROR) {
    std::vector<Npp16s> hostVoronoi(vorCount), hostIndices(vorCount);
    std::vector<Npp16u> hostRelative(vorCount);
    voronoi.copyToHost(hostVoronoi);
    indices.copyToHost(hostIndices);
    relative.copyToHost(hostRelative);
    c.host_output(hostVoronoi.data(), hostVoronoi.size(), ValueKind::I16);
    c.host_output(hostIndices.data(), hostIndices.size(), ValueKind::I16);
    c.host_output(hostRelative.data(), hostRelative.size(), ValueKind::U16);
    if (variant == 0 || variant == 1) {
      std::vector<Npp32f> hostTransform((size_t)width * height);
      transform32.copyToHost(hostTransform);
      c.host_output(hostTransform.data(), hostTransform.size(), ValueKind::F32, 1e-5, 1e-5);
    } else {
      std::vector<Npp64f> hostTransform((size_t)width * height);
      transform64.copyToHost(hostTransform);
      c.host_output(hostTransform.data(), hostTransform.size(), ValueKind::F64, 1e-9, 1e-9);
    }
  }
  c.finish();
}

}  // namespace

TEST(NppiSignedDistanceTransformDiff, Signed32f) {
  profile_case("s2/SignedDT/signed_rect_64_o0", "nppiSignedDistanceTransformPBA_32f_C1R_Ctx", 64, 64, 0,
               Shape::Rect, 0.0f, 0.0f, 0.0f, 0);
  profile_case("s2/SignedDT/signed_circle_64_pad3", "nppiSignedDistanceTransformPBA_32f_C1R_Ctx", 64, 64, 3,
               Shape::Circle, 0.0f, 0.0f, 0.0f, 0);
  profile_case("s2/SignedDT/signed_blobs_96x80", "nppiSignedDistanceTransformPBA_32f_C1R_Ctx", 96, 80, 0,
               Shape::Blobs, 0.0f, 0.0f, 0.0f, 0);
  profile_case("s2/SignedDT/signed_rect_64_cut05", "nppiSignedDistanceTransformPBA_32f_C1R_Ctx", 64, 64, 0,
               Shape::Rect, 0.5f, 0.0f, 0.0f, 0);
  profile_case("s2/SignedDT/signed_rect_64_shift", "nppiSignedDistanceTransformPBA_32f_C1R_Ctx", 64, 64, 0,
               Shape::Rect, 0.0f, 0.25f, -0.25f, 0);
  profile_case("s2/SignedDT/signed_rect_64_cut09_shift", "nppiSignedDistanceTransformPBA_32f_C1R_Ctx", 64, 64, 0,
               Shape::Rect, 0.9f, -0.5f, 0.5f, 0);
}

TEST(NppiSignedDistanceTransformDiff, Abs32f) {
  profile_case("s2/SignedDT/abs32_rect_64", "nppiSignedDistanceTransformAbsPBA_32f_C1R_Ctx", 64, 64, 0,
               Shape::Rect, 0.0f, 0.0f, 0.0f, 1);
  profile_case("s2/SignedDT/abs32_circle_64_pad2", "nppiSignedDistanceTransformAbsPBA_32f_C1R_Ctx", 64, 64, 2,
               Shape::Circle, 0.0f, 0.0f, 0.0f, 1);
  profile_case("s2/SignedDT/abs32_blobs_96x80", "nppiSignedDistanceTransformAbsPBA_32f_C1R_Ctx", 96, 80, 0,
               Shape::Blobs, 0.0f, 0.0f, 0.0f, 1);
  profile_case("s2/SignedDT/abs32_rect_64_shift", "nppiSignedDistanceTransformAbsPBA_32f_C1R_Ctx", 64, 64, 0,
               Shape::Rect, 0.25f, 0.25f, -0.25f, 1);
}

TEST(NppiSignedDistanceTransformDiff, Abs64fNoAA) {
  profile_case("s2/SignedDT/abs64_rect_64", "nppiSignedDistanceTransformAbsPBA_64f_C1R_Ctx", 64, 64, 0,
               Shape::Rect, 0.0f, 0.0f, 0.0f, 2);
  profile_case("s2/SignedDT/abs64_circle_64_pad2", "nppiSignedDistanceTransformAbsPBA_64f_C1R_Ctx", 64, 64, 2,
               Shape::Circle, 0.0f, 0.0f, 0.0f, 2);
  profile_case("s2/SignedDT/abs64_blobs_96x80", "nppiSignedDistanceTransformAbsPBA_64f_C1R_Ctx", 96, 80, 0,
               Shape::Blobs, 0.0f, 0.0f, 0.0f, 2);
  profile_case("s2/SignedDT/abs64_rect_64_cut05_shift", "nppiSignedDistanceTransformAbsPBA_64f_C1R_Ctx", 64, 64, 0,
               Shape::Rect, 0.5f, 0.25f, -0.25f, 2);
}

TEST(NppiSignedDistanceTransformDiff, Contract) {
  const int w = 64, h = 64;
  std::vector<Npp32f> img((size_t)w * h, 0.0f);
  for (int y = 20; y < 44; ++y)
    for (int x = 20; x < 44; ++x) img[(size_t)y * w + x] = 1.0f;
  DeviceMemory<Npp32f> src(img.size());
  src.copyFromHost(img);
  std::vector<Npp64f> img64(img.begin(), img.end());
  DeviceMemory<Npp64f> src64(img64.size());
  src64.copyFromHost(img64);
  size_t bufferSize = 0;
  nppiSignedDistanceTransformPBAGetBufferSize({w, h}, &bufferSize);
  DeviceMemory<Npp8u> buffer(bufferSize);
  size_t aaSize = 0;
  nppiSignedDistanceTransformPBAGetAntialiasingBufferSize({w, h}, &aaSize);
  DeviceMemory<Npp8u> aa(aaSize > 0 ? aaSize : 1);
  std::vector<Npp8u> aaZero(aaSize > 0 ? aaSize : 1, 0);
  aa.copyFromHost(aaZero);
  const int vorStep = w * 4;
  const size_t vorCount = (size_t)vorStep * h / sizeof(Npp16s);
  DeviceMemory<Npp16s> voronoi(vorCount), indices(vorCount);
  DeviceMemory<Npp16u> relative(vorCount);
  DeviceMemory<Npp32f> transform((size_t)w * h);
  DeviceMemory<Npp64f> transform64((size_t)w * h);
  NppStreamContext ctx = current_context();
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();

  const char *signedName = "nppiSignedDistanceTransformPBA_32f_C1R_Ctx";
  const char *abs32Name = "nppiSignedDistanceTransformAbsPBA_32f_C1R_Ctx";
  const char *abs64Name = "nppiSignedDistanceTransformAbsPBA_64f_C1R_Ctx";
  const char *abs32f64Name = "nppiSignedDistanceTransformAbsPBA_32f64f_C1R_Ctx";

  status_case("s2/SignedDT/signed_null_src", signedName,
              nppiSignedDistanceTransformPBA_32f_C1R_Ctx(nullptr, w * 4, 0, 0, 0, voronoi.get(), vorStep,
                                                         indices.get(), vorStep, nullptr, vorStep, transform.get(),
                                                         w * 4, {w, h}, buffer.get(), ctx));
  status_case("s2/SignedDT/signed_null_buffer", signedName,
              nppiSignedDistanceTransformPBA_32f_C1R_Ctx(src.get(), w * 4, 0, 0, 0, voronoi.get(), vorStep,
                                                         indices.get(), vorStep, nullptr, vorStep, transform.get(),
                                                         w * 4, {w, h}, nullptr, ctx));
  status_case("s2/SignedDT/signed_roi_32", signedName,
              nppiSignedDistanceTransformPBA_32f_C1R_Ctx(src.get(), w * 4, 0, 0, 0, voronoi.get(), vorStep,
                                                         indices.get(), vorStep, nullptr, vorStep, transform.get(),
                                                         w * 4, {32, 32}, buffer.get(), ctx));
  status_case("s2/SignedDT/signed_neg_roi", signedName,
              nppiSignedDistanceTransformPBA_32f_C1R_Ctx(src.get(), w * 4, 0, 0, 0, voronoi.get(), vorStep,
                                                         indices.get(), vorStep, nullptr, vorStep, transform.get(),
                                                         w * 4, {-1, h}, buffer.get(), ctx));
  status_case("s2/SignedDT/signed_short_voronoi_step", signedName,
              nppiSignedDistanceTransformPBA_32f_C1R_Ctx(src.get(), w * 4, 0, 0, 0, voronoi.get(), w * 2,
                                                         indices.get(), vorStep, nullptr, vorStep, transform.get(),
                                                         w * 4, {w, h}, buffer.get(), ctx));
  status_case("s2/SignedDT/signed_short_transform_step", signedName,
              nppiSignedDistanceTransformPBA_32f_C1R_Ctx(src.get(), w * 4, 0, 0, 0, voronoi.get(), vorStep,
                                                         indices.get(), vorStep, nullptr, vorStep, transform.get(),
                                                         w * 2, {w, h}, buffer.get(), ctx));
  status_case("s2/SignedDT/signed_one_output", signedName,
              nppiSignedDistanceTransformPBA_32f_C1R_Ctx(src.get(), w * 4, 0, 0, 0, nullptr, vorStep, nullptr,
                                                         vorStep, nullptr, vorStep, transform.get(), w * 4, {w, h},
                                                         buffer.get(), ctx));
  status_case("s2/SignedDT/signed_cut_neg", signedName,
              nppiSignedDistanceTransformPBA_32f_C1R_Ctx(src.get(), w * 4, -0.5f, 0, 0, voronoi.get(), vorStep,
                                                         indices.get(), vorStep, nullptr, vorStep, transform.get(),
                                                         w * 4, {w, h}, buffer.get(), ctx));
  status_case("s2/SignedDT/signed_cut_one", signedName,
              nppiSignedDistanceTransformPBA_32f_C1R_Ctx(src.get(), w * 4, 1.0f, 0, 0, voronoi.get(), vorStep,
                                                         indices.get(), vorStep, nullptr, vorStep, transform.get(),
                                                         w * 4, {w, h}, buffer.get(), ctx));
  status_case("s2/SignedDT/signed_cut_inf", signedName,
              nppiSignedDistanceTransformPBA_32f_C1R_Ctx(src.get(), w * 4, inf, 0, 0, voronoi.get(), vorStep,
                                                         indices.get(), vorStep, nullptr, vorStep, transform.get(),
                                                         w * 4, {w, h}, buffer.get(), ctx));
  status_case("s2/SignedDT/signed_cut_nan", signedName,
              nppiSignedDistanceTransformPBA_32f_C1R_Ctx(src.get(), w * 4, nan, 0, 0, voronoi.get(), vorStep,
                                                         indices.get(), vorStep, nullptr, vorStep, transform.get(),
                                                         w * 4, {w, h}, buffer.get(), ctx));

  status_case("s2/SignedDT/abs32_null_src", abs32Name,
              nppiSignedDistanceTransformAbsPBA_32f_C1R_Ctx(nullptr, w * 4, 0, 0, 0, voronoi.get(), vorStep,
                                                            indices.get(), vorStep, relative.get(), vorStep,
                                                            transform.get(), w * 4, {w, h}, buffer.get(), ctx));
  status_case("s2/SignedDT/abs32_roi_32", abs32Name,
              nppiSignedDistanceTransformAbsPBA_32f_C1R_Ctx(src.get(), w * 4, 0, 0, 0, voronoi.get(), vorStep,
                                                            indices.get(), vorStep, relative.get(), vorStep,
                                                            transform.get(), w * 4, {32, 32}, buffer.get(), ctx));
  status_case("s2/SignedDT/abs32_short_step", abs32Name,
              nppiSignedDistanceTransformAbsPBA_32f_C1R_Ctx(src.get(), w * 4, 0, 0, 0, voronoi.get(), w * 2,
                                                            indices.get(), vorStep, relative.get(), vorStep,
                                                            transform.get(), w * 4, {w, h}, buffer.get(), ctx));

  status_case("s2/SignedDT/abs64_null_src", abs64Name,
              nppiSignedDistanceTransformAbsPBA_64f_C1R_Ctx(nullptr, w * 8, 0, 0, 0, voronoi.get(), vorStep,
                                                            indices.get(), vorStep, relative.get(), vorStep,
                                                            nullptr, w * 8, {w, h}, buffer.get(), nullptr, ctx));
  status_case("s2/SignedDT/abs64_roi_32", abs64Name,
              nppiSignedDistanceTransformAbsPBA_64f_C1R_Ctx(src64.get(), w * 8, 0, 0, 0, voronoi.get(), vorStep,
                                                            indices.get(), vorStep, relative.get(), vorStep, nullptr,
                                                            w * 8, {32, 32}, buffer.get(), nullptr, ctx));
  status_case("s2/SignedDT/abs64_valid_aa", abs64Name,
              nppiSignedDistanceTransformAbsPBA_64f_C1R_Ctx(src64.get(), w * 8, 0, 0, 0, voronoi.get(), vorStep,
                                                            indices.get(), vorStep, relative.get(), vorStep,
                                                            transform64.get(), w * 8, {w, h}, buffer.get(),
                                                            aa.get(), ctx));

  // 32f64f abs: the antialiasing buffer is mandatory; the AA values are not
  // reproduced, so these cases are status-only.
  status_case("s2/SignedDT/abs32f64_null_aa", abs32f64Name,
              nppiSignedDistanceTransformAbsPBA_32f64f_C1R_Ctx(
                  src.get(), w * 4, 0, 0, 0, voronoi.get(), vorStep, indices.get(), vorStep, relative.get(), vorStep,
                  nullptr, w * 8, {w, h}, buffer.get(), nullptr, ctx));
  status_case("s2/SignedDT/abs32f64_roi_32", abs32f64Name,
              nppiSignedDistanceTransformAbsPBA_32f64f_C1R_Ctx(
                  src.get(), w * 4, 0, 0, 0, voronoi.get(), vorStep, indices.get(), vorStep, relative.get(), vorStep,
                  nullptr, w * 8, {32, 32}, buffer.get(), aa.get(), ctx));
  status_case("s2/SignedDT/abs32f64_valid_aa", abs32f64Name,
              nppiSignedDistanceTransformAbsPBA_32f64f_C1R_Ctx(
                  src.get(), w * 4, 0, 0, 0, voronoi.get(), vorStep, indices.get(), vorStep, relative.get(), vorStep,
                  nullptr, w * 8, {w, h}, buffer.get(), aa.get(), ctx));
  clear_cuda_error();
}
