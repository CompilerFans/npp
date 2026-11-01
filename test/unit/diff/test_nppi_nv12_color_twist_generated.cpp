// Differential tests for the NV12 <-> RGB ColorTwist functions of package c4:
//   nppiNV12ToRGB_16u_ColorTwist32f_P2C3R_Ctx
//   nppiRGBToNV12_{8u,16u}_ColorTwist32f_{C3P2R,P3P2R}_Ctx
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiNv12ColorTwistDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records ./diff_tests \
//     --gtest_filter='NppiNv12ColorTwistDiff.*'
//
// Semantics pinned here (NVIDIA NPP 13.4, 5090): the caller's 3x4 matrix is
// applied to (Y,U,V,1) or (R,G,B,1); results are rounded half-up and clamped
// to the element range.  NV12->RGB takes chroma from the top-left sample of
// each 2x2 block and writes RGB per pixel; RGB->NV12 writes Y per pixel and
// the 2x2-averaged (float average, rounded half-up) U/V pair into the
// interleaved UV plane.  Only the ROI is written; row padding is untouched.
// Error paths: null array/destination -> NPP_NULL_POINTER_ERROR, negative
// width/height -> NPP_SIZE_ERROR, odd width (NV12ToRGB) or odd width/height
// (RGBToNV12) -> NPP_DOUBLE_SIZE_WARNING with no write.  NV12ToRGB accepts an
// odd height and processes ceil(height/2)*2 rows (the extra row belongs to the
// last chroma group).

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <cstdint>
#include <memory>
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
  EXPECT_EQ(nppGetStreamContext(&ctx), NPP_NO_ERROR) << "nppGetStreamContext failed";
  return ctx;
}

ValueKind kind_for(int elemBytes) { return elemBytes == 1 ? ValueKind::U8 : ValueKind::U16; }

// Two deterministic matrix flavours: a real BT.601-ish conversion (negative
// coefficients and offsets, exercises clamping) and a fractional matrix that
// exercises rounding.
void matrix601(Npp32f m[3][4]) {
  const Npp32f src[3][4] = {{0.2568F, 0.5041F, 0.0979F, 16.0F},
                            {-0.1482F, -0.2910F, 0.4392F, 128.0F},
                            {0.4392F, -0.3678F, -0.0714F, 128.0F}};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      m[i][j] = src[i][j];
    }
  }
}

void matrixFrac(Npp32f m[3][4]) {
  const Npp32f src[3][4] = {{0.5F, -1.25F, 0.75F, 0.5F},
                            {1.5F, 0.25F, -0.5F, 12.25F},
                            {-0.75F, 0.5F, 1.25F, 3.5F}};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      m[i][j] = src[i][j];
    }
  }
}

void pickMatrix(int variant, Npp32f m[3][4]) {
  if (variant == 0) {
    matrix601(m);
  } else {
    matrixFrac(m);
  }
}

// 16-bit NV12 -> packed RGB.  dst is captured whole so padding coverage is
// part of the golden.
void nv12_to_rgb_case(const std::string &idBase, int width, int height, int padY, int padUV, int padDst,
                      int variant, uint64_t seed) {
  const int yStep = width * 2 + padY;
  const int uvStep = width * 2 + padUV;
  const int dstStep = width * 3 * 2 + padDst;
  // NVIDIA writes ceil(height/2)*2 output rows for an odd ROI height (the odd
  // row is part of the last 2x2 chroma group), so source and destination
  // buffers are sized for the processed height.
  const int procH = (height + 1) & ~1;
  const size_t yElems = (size_t)yStep / 2 * procH;
  const size_t uvElems = (size_t)uvStep / 2 * (procH / 2);
  const size_t dstElems = (size_t)dstStep / 2 * procH;
  std::vector<Npp16u> hy = make_uniform<Npp16u>(yElems, seed + 3);
  std::vector<Npp16u> huv = make_uniform<Npp16u>(uvElems, seed + 5);
  std::vector<Npp16u> hd = make_uniform<Npp16u>(dstElems, seed + 7);
  DeviceMemory<Npp16u> &y = arena<Npp16u>(yElems);
  DeviceMemory<Npp16u> &uv = arena<Npp16u>(uvElems);
  DeviceMemory<Npp16u> &d = arena<Npp16u>(dstElems);
  y.copyFromHost(hy);
  uv.copyFromHost(huv);
  d.copyFromHost(hd);
  const Npp16u *src[2] = {y.get(), uv.get()};
  int steps[2] = {yStep, uvStep};
  NppiSize o{width, height};
  Npp32f m[3][4];
  pickMatrix(variant, m);
  clear_cuda_error();
  NppStatus st = nppiNV12ToRGB_16u_ColorTwist32f_P2C3R_Ctx(src, steps, d.get(), dstStep, o, m, current_context());
  Case c("nppiNV12ToRGB/" + idBase, "nppiNV12ToRGB_16u_ColorTwist32f_P2C3R");
  c.input(hy.data(), hy.size() * sizeof(Npp16u));
  c.input(huv.data(), huv.size() * sizeof(Npp16u));
  c.input(hd.data(), hd.size() * sizeof(Npp16u));
  c.input(m, sizeof(m));
  c.status(st);
  c.device_output<Npp16u>(d.get(), dstElems, kind_for(2));
  c.finish();
}

// RGB -> NV12 (C3P2R packed or P3P2R planar), 8u or 16u.
void rgb_to_nv12_case(const std::string &idBase, bool planar, int elemBytes, int width, int height, int padSrc,
                      int padY, int padUV, int variant, uint64_t seed) {
  const int srcStep = width * 3 * elemBytes + padSrc;
  const int yStep = width * elemBytes + padY;
  const int uvStep = width * elemBytes + padUV;
  const size_t srcElems = (size_t)srcStep / elemBytes * height;
  const size_t planeElems = (size_t)width * height;
  const size_t yElems = (size_t)yStep / elemBytes * height;
  const size_t uvElems = (size_t)uvStep / elemBytes * (height / 2);
  Npp32f m[3][4];
  pickMatrix(variant, m);
  const std::string fname = std::string("nppiRGBToNV12_") + (elemBytes == 1 ? "8u" : "16u") +
                            (planar ? "_ColorTwist32f_P3P2R" : "_ColorTwist32f_C3P2R");
  // Include dtype/layout in the case id: the four variants write the same
  // nppiRGBToNV12/<id> paths otherwise.
  const std::string caseId = std::string("nppiRGBToNV12/") + (elemBytes == 1 ? "8u" : "16u") +
                             (planar ? "P3_" : "C3_") + idBase;
  const ValueKind kind = kind_for(elemBytes);

  if (elemBytes == 1) {
    std::vector<Npp8u> hs = make_uniform<Npp8u>(srcElems, seed + 11);
    std::vector<Npp8u> hp0 = make_uniform<Npp8u>(planeElems, seed + 13);
    std::vector<Npp8u> hp1 = make_uniform<Npp8u>(planeElems, seed + 17);
    std::vector<Npp8u> hp2 = make_uniform<Npp8u>(planeElems, seed + 19);
    std::vector<Npp8u> hy = make_uniform<Npp8u>(yElems, seed + 23);
    std::vector<Npp8u> huv = make_uniform<Npp8u>(uvElems, seed + 29);
    DeviceMemory<Npp8u> &s = arena<Npp8u>(srcElems);
    DeviceMemory<Npp8u> &p0 = arena<Npp8u>(planeElems);
    DeviceMemory<Npp8u> &p1 = arena<Npp8u>(planeElems);
    DeviceMemory<Npp8u> &p2 = arena<Npp8u>(planeElems);
    DeviceMemory<Npp8u> &dy = arena<Npp8u>(yElems);
    DeviceMemory<Npp8u> &duv = arena<Npp8u>(uvElems);
    s.copyFromHost(hs);
    p0.copyFromHost(hp0);
    p1.copyFromHost(hp1);
    p2.copyFromHost(hp2);
    dy.copyFromHost(hy);
    duv.copyFromHost(huv);
    Npp8u *dst[2] = {dy.get(), duv.get()};
    int dstSteps[2] = {yStep, uvStep};
    NppiSize o{width, height};
    clear_cuda_error();
    NppStatus st;
    if (planar) {
      const Npp8u *src[3] = {p0.get(), p1.get(), p2.get()};
      int srcSteps[3] = {width, width, width};
      st = nppiRGBToNV12_8u_ColorTwist32f_P3P2R_Ctx(src, srcSteps, dst, dstSteps, o, m, current_context());
    } else {
      st = nppiRGBToNV12_8u_ColorTwist32f_C3P2R_Ctx(s.get(), srcStep, dst, dstSteps, o, m, current_context());
    }
    Case c(caseId, fname);
    if (planar) {
      c.input(hp0.data(), hp0.size());
      c.input(hp1.data(), hp1.size());
      c.input(hp2.data(), hp2.size());
    } else {
      c.input(hs.data(), hs.size());
    }
    c.input(m, sizeof(m));
    c.status(st);
    c.device_output<Npp8u>(dy.get(), yElems, kind);
    c.device_output<Npp8u>(duv.get(), uvElems, kind);
    c.finish();
  } else {
    std::vector<Npp16u> hs = make_uniform<Npp16u>(srcElems, seed + 11);
    std::vector<Npp16u> hp0 = make_uniform<Npp16u>(planeElems, seed + 13);
    std::vector<Npp16u> hp1 = make_uniform<Npp16u>(planeElems, seed + 17);
    std::vector<Npp16u> hp2 = make_uniform<Npp16u>(planeElems, seed + 19);
    std::vector<Npp16u> hy = make_uniform<Npp16u>(yElems, seed + 23);
    std::vector<Npp16u> huv = make_uniform<Npp16u>(uvElems, seed + 29);
    DeviceMemory<Npp16u> &s = arena<Npp16u>(srcElems);
    DeviceMemory<Npp16u> &p0 = arena<Npp16u>(planeElems);
    DeviceMemory<Npp16u> &p1 = arena<Npp16u>(planeElems);
    DeviceMemory<Npp16u> &p2 = arena<Npp16u>(planeElems);
    DeviceMemory<Npp16u> &dy = arena<Npp16u>(yElems);
    DeviceMemory<Npp16u> &duv = arena<Npp16u>(uvElems);
    s.copyFromHost(hs);
    p0.copyFromHost(hp0);
    p1.copyFromHost(hp1);
    p2.copyFromHost(hp2);
    dy.copyFromHost(hy);
    duv.copyFromHost(huv);
    Npp16u *dst[2] = {dy.get(), duv.get()};
    int dstSteps[2] = {yStep, uvStep};
    NppiSize o{width, height};
    clear_cuda_error();
    NppStatus st;
    if (planar) {
      const Npp16u *src[3] = {p0.get(), p1.get(), p2.get()};
      int srcSteps[3] = {width * 2, width * 2, width * 2};
      st = nppiRGBToNV12_16u_ColorTwist32f_P3P2R_Ctx(src, srcSteps, dst, dstSteps, o, m, current_context());
    } else {
      st = nppiRGBToNV12_16u_ColorTwist32f_C3P2R_Ctx(s.get(), srcStep, dst, dstSteps, o, m, current_context());
    }
    Case c(caseId, fname);
    if (planar) {
      c.input(hp0.data(), hp0.size() * 2);
      c.input(hp1.data(), hp1.size() * 2);
      c.input(hp2.data(), hp2.size() * 2);
    } else {
      c.input(hs.data(), hs.size() * 2);
    }
    c.input(m, sizeof(m));
    c.status(st);
    c.device_output<Npp16u>(dy.get(), yElems, kind);
    c.device_output<Npp16u>(duv.get(), uvElems, kind);
    c.finish();
  }
}

// NV12ToRGB error/odd-size paths (all valid pointers except the null cases).
void nv12_to_rgb_error(const std::string &id, int nullIndex, NppiSize roi) {
  const int width = 8, height = 4;
  const int yStep = width * 2 + 4, uvStep = width * 2 + 4, dstStep = width * 3 * 2;
  const size_t yElems = (size_t)yStep / 2 * height;
  const size_t uvElems = (size_t)uvStep / 2 * (height / 2);
  const size_t dstElems = (size_t)dstStep / 2 * height;
  std::vector<Npp16u> hy = make_uniform<Npp16u>(yElems, 77);
  std::vector<Npp16u> huv = make_uniform<Npp16u>(uvElems, 78);
  std::vector<Npp16u> hd = make_uniform<Npp16u>(dstElems, 79);
  DeviceMemory<Npp16u> &y = arena<Npp16u>(yElems);
  DeviceMemory<Npp16u> &uv = arena<Npp16u>(uvElems);
  DeviceMemory<Npp16u> &d = arena<Npp16u>(dstElems);
  y.copyFromHost(hy);
  uv.copyFromHost(huv);
  d.copyFromHost(hd);
  const Npp16u *src[2] = {y.get(), uv.get()};
  int steps[2] = {yStep, uvStep};
  Npp32f m[3][4];
  matrixFrac(m);
  clear_cuda_error();
  NppStatus st;
  if (nullIndex == 1) {
    // A null plane inside a valid array is not validated by NVIDIA (it faults
    // the context); the array pointer itself is what returns NULL_POINTER.
    st = nppiNV12ToRGB_16u_ColorTwist32f_P2C3R_Ctx(nullptr, steps, d.get(), dstStep, roi, m, current_context());
  } else if (nullIndex == 2) {
    st = nppiNV12ToRGB_16u_ColorTwist32f_P2C3R_Ctx(src, steps, nullptr, dstStep, roi, m, current_context());
  } else {
    st = nppiNV12ToRGB_16u_ColorTwist32f_P2C3R_Ctx(src, steps, d.get(), dstStep, roi, m, current_context());
  }
  Case c("nppiNV12ToRGB/" + id, "nppiNV12ToRGB_16u_ColorTwist32f_P2C3R");
  c.status(st);
  c.device_output<Npp16u>(d.get(), dstElems, kind_for(2));
  c.finish();
}

}  // namespace

class NppiNv12ColorTwistDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

TEST_F(NppiNv12ColorTwistDiff, Nv12ToRgb16u_Roi) {
  nv12_to_rgb_case("bt601/32x16", 32, 16, 16, 8, 8, 0, 5101);
  nv12_to_rgb_case("frac/16x8", 16, 8, 0, 4, 16, 1, 5102);
  nv12_to_rgb_case("frac/2x2", 2, 2, 0, 0, 0, 1, 5103);
  nv12_to_rgb_case("bt601/8x3_oddh", 8, 3, 6, 2, 4, 0, 5104);
  nv12_to_rgb_case("frac/64x32", 64, 32, 16, 16, 16, 1, 5105);
}

TEST_F(NppiNv12ColorTwistDiff, RgbToNv12_8u_C3R) {
  rgb_to_nv12_case("bt601/32x16", false, 1, 32, 16, 8, 8, 8, 0, 6101);
  rgb_to_nv12_case("frac/16x8", false, 1, 16, 8, 0, 4, 4, 1, 6102);
  rgb_to_nv12_case("frac/2x2", false, 1, 2, 2, 0, 0, 0, 1, 6103);
  rgb_to_nv12_case("bt601/64x32", false, 1, 64, 32, 16, 16, 16, 0, 6104);
}

TEST_F(NppiNv12ColorTwistDiff, RgbToNv12_8u_P3R) {
  rgb_to_nv12_case("bt601/32x16", true, 1, 32, 16, 0, 8, 8, 0, 6201);
  rgb_to_nv12_case("frac/16x8", true, 1, 16, 8, 0, 0, 4, 1, 6202);
  rgb_to_nv12_case("frac/4x4", true, 1, 4, 4, 0, 0, 0, 1, 6203);
}

TEST_F(NppiNv12ColorTwistDiff, RgbToNv12_16u_C3R) {
  rgb_to_nv12_case("bt601/32x16", false, 2, 32, 16, 8, 8, 8, 0, 6301);
  rgb_to_nv12_case("frac/16x8", false, 2, 16, 8, 0, 4, 4, 1, 6302);
  rgb_to_nv12_case("frac/4x4", false, 2, 4, 4, 0, 0, 0, 1, 6303);
}

TEST_F(NppiNv12ColorTwistDiff, RgbToNv12_16u_P3R) {
  rgb_to_nv12_case("bt601/32x16", true, 2, 32, 16, 0, 8, 8, 0, 6401);
  rgb_to_nv12_case("frac/16x8", true, 2, 16, 8, 0, 0, 4, 1, 6402);
  rgb_to_nv12_case("frac/4x4", true, 2, 4, 4, 0, 0, 0, 1, 6403);
}

TEST_F(NppiNv12ColorTwistDiff, Nv12ToRgb16u_Errors) {
  nv12_to_rgb_error("null_src", 1, {8, 4});
  nv12_to_rgb_error("null_dst", 2, {8, 4});
  nv12_to_rgb_error("neg_w", 0, {-2, 4});
  nv12_to_rgb_error("neg_h", 0, {8, -2});
  nv12_to_rgb_error("odd_w", 0, {7, 4});
  nv12_to_rgb_error("odd_h", 0, {8, 3});
}

TEST_F(NppiNv12ColorTwistDiff, RgbToNv12_8u_Errors) {
  const int width = 8, height = 4;
  const int srcStep = width * 3, yStep = width, uvStep = width;
  const size_t srcElems = (size_t)srcStep * height;
  const size_t yElems = (size_t)yStep * height;
  const size_t uvElems = (size_t)uvStep * (height / 2);
  std::vector<Npp8u> hs = make_uniform<Npp8u>(srcElems, 88);
  std::vector<Npp8u> hy = make_uniform<Npp8u>(yElems, 89);
  std::vector<Npp8u> huv = make_uniform<Npp8u>(uvElems, 90);
  DeviceMemory<Npp8u> &s = arena<Npp8u>(srcElems);
  DeviceMemory<Npp8u> &dy = arena<Npp8u>(yElems);
  DeviceMemory<Npp8u> &duv = arena<Npp8u>(uvElems);
  s.copyFromHost(hs);
  dy.copyFromHost(hy);
  duv.copyFromHost(huv);
  Npp8u *dst[2] = {dy.get(), duv.get()};
  int dstSteps[2] = {yStep, uvStep};
  Npp32f m[3][4];
  matrixFrac(m);
  struct ErrCase {
    const char *id;
    const Npp8u *src;
    Npp8u **dst;
    NppiSize roi;
  } cases[] = {
      {"null_src", nullptr, dst, {8, 4}},
      {"null_dst", s.get(), nullptr, {8, 4}},
      {"neg_w", s.get(), dst, {-2, 4}},
      {"neg_h", s.get(), dst, {8, -2}},
      {"odd_w", s.get(), dst, {7, 4}},
      {"odd_h", s.get(), dst, {8, 3}},
  };
  for (const auto &e : cases) {
    clear_cuda_error();
    NppStatus st = nppiRGBToNV12_8u_ColorTwist32f_C3P2R_Ctx(e.src, srcStep, e.dst, dstSteps, e.roi, m,
                                                            current_context());
    Case c(std::string("nppiRGBToNV12/8u_") + e.id, "nppiRGBToNV12_8u_ColorTwist32f_C3P2R");
    c.status(st);
    c.device_output<Npp8u>(dy.get(), yElems, kind_for(1));
    c.device_output<Npp8u>(duv.get(), uvElems, kind_for(1));
    c.finish();
  }
}
