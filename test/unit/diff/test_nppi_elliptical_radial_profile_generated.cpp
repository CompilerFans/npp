// Differential tests for nppiEllipticalRadialProfile_32f_C1R_Ctx.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   tools/remote5090.sh record 'NppiEllipticalRadialProfileDiff.*'
//
// The executor reproduces the sm_120 SASS of
// executeComputeProfileEllipticalIfE...: ring k is the dense band
// k^2 <= t^2 < (k+1)^2 with t^2 = (dx*cos + dy*sin)^2
// + (dy*cos - dx*sin)^2 / ratio^2, scanned over a cropped window
// [floor(cx - k*ratio), ceil(cx + k*ratio)] x
// [floor(cy - k*ratio), +ceil(2*(k+1)*ratio)].  The crop is observable: the
// 33x33, ratio 2, 24-sample impulse map has 1065 of 1089 pixels assigned.
// The host contract order (null, negative ROI, ratio < 1, empty no-ops) and
// the plain `ratio < 1` comparison (NaN/+inf accepted, -inf rejected) were
// measured on the 5090.

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

enum class ImageKind { Pattern, Constant, Zeros };

void status_case(const std::string &id, const char *fname, NppStatus st) {
  Case c(id, fname);
  c.status(st);
  c.finish();
}

// Runs one elliptical profile case.  `pad` extra floats per row exercise the
// source step, the ROI is width x height and the centre may lie outside it.
void profile_case(const std::string &id, int width, int height, int pad, NppiPoint32f center, float ratio,
                  float orientation, int samples, ImageKind kind = ImageKind::Pattern) {
  const char *fname = "nppiEllipticalRadialProfile_32f_C1R_Ctx";
  const int stride = width + pad;
  std::vector<Npp32f> img((size_t)stride * height, 0.0f);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < stride; ++x) {
      if (kind == ImageKind::Pattern) {
        img[(size_t)y * stride + x] =
            (Npp32f)(((x * 11 + y * 7) % 23) - 11) * 0.5f + (Npp32f)(x & 3) * 0.25f;
      } else if (kind == ImageKind::Constant) {
        img[(size_t)y * stride + x] = 3.5f;
      }
    }
  }
  DeviceMemory<Npp32f> src(img.size());
  src.copyFromHost(img);

  const size_t n = samples > 0 ? (size_t)samples : 1;
  DeviceMemory<NppiProfileData> prof(n);
  std::vector<NppiProfileData> init(n, NppiProfileData{-12345, kSentinel, kSentinel});
  prof.copyFromHost(init);

  NppStreamContext ctx = current_context();
  NppStatus st = nppiEllipticalRadialProfile_32f_C1R_Ctx(src.get(), stride * (int)sizeof(Npp32f),
                                                         NppiSize{width, height}, center, ratio, orientation,
                                                         prof.get(), (Npp16u)samples, ctx);
  clear_cuda_error();

  Case c(id, fname);
  c.status(st);
  c.input(img.data(), img.size() * sizeof(Npp32f));
  if (st == NPP_NO_ERROR && samples > 0) {
    std::vector<NppiProfileData> host;
    prof.copyToHost(host);
    std::vector<int> counts((size_t)samples);
    std::vector<float> means((size_t)samples), sds((size_t)samples);
    for (int i = 0; i < samples; ++i) {
      counts[(size_t)i] = host[(size_t)i].nPixels;
      means[(size_t)i] = host[(size_t)i].nMeanIntensity;
      sds[(size_t)i] = host[(size_t)i].nStdDevIntensity;
    }
    c.host_output(counts.data(), counts.size(), ValueKind::I32);
    c.host_output(means.data(), means.size(), ValueKind::F32, 1e-5, 1e-5);
    c.host_output(sds.data(), sds.size(), ValueKind::F32, 1e-5, 1e-5);
  }
  c.finish();
}

}  // namespace

TEST(NppiEllipticalRadialProfileDiff, HalfPixelCenter) {
  profile_case("s2/Elliptical/ell_33x33_half_r2_o0_samples24", 33, 33, 0, NppiPoint32f{16.5f, 16.5f}, 2.0f, 0.0f,
               24);
}

TEST(NppiEllipticalRadialProfileDiff, Rotated) {
  profile_case("s2/Elliptical/ell_33x33_half_r2_o30deg_samples24", 33, 33, 0, NppiPoint32f{16.5f, 16.5f}, 2.0f,
               0.52359878f, 24);
  profile_case("s2/Elliptical/ell_33x33_half_r2_o45deg_samples24", 33, 33, 0, NppiPoint32f{16.5f, 16.5f}, 2.0f,
               0.78539816f, 24);
  profile_case("s2/Elliptical/ell_33x33_half_r2_o60deg_samples24", 33, 33, 0, NppiPoint32f{16.5f, 16.5f}, 2.0f,
               1.04719755f, 24);
}

TEST(NppiEllipticalRadialProfileDiff, AnisotropyAndIsotropy) {
  profile_case("s2/Elliptical/ell_33x33_half_r1.5_o0.7_samples24", 33, 33, 0, NppiPoint32f{16.5f, 16.5f}, 1.5f,
               0.7f, 24);
  profile_case("s2/Elliptical/ell_33x33_int_r1_o0.3_samples24", 33, 33, 0, NppiPoint32f{16.0f, 16.0f}, 1.0f, 0.3f,
               24);
}

TEST(NppiEllipticalRadialProfileDiff, PaddedStepAndRectRoi) {
  profile_case("s2/Elliptical/ell_40x25_pad3_r2_o0.9_samples24", 40, 25, 3, NppiPoint32f{20.5f, 12.5f}, 2.0f, 0.9f,
               24);
}

TEST(NppiEllipticalRadialProfileDiff, CenterOutside) {
  profile_case("s2/Elliptical/ell_33x33_center_out_r2_o0_samples8", 33, 33, 0, NppiPoint32f{-3.5f, -2.5f}, 2.0f,
               0.0f, 8);
  profile_case("s2/Elliptical/ell_33x33_center_near_r1.1_o1.2_samples8", 33, 33, 0, NppiPoint32f{31.75f, 0.25f},
               1.1f, 1.2f, 8);
}

TEST(NppiEllipticalRadialProfileDiff, ConstantAndZeroImages) {
  profile_case("s2/Elliptical/ell_21x15_const_r1.5_o0.4_samples12", 21, 15, 0, NppiPoint32f{10.5f, 7.5f}, 1.5f,
               0.4f, 12, ImageKind::Constant);
  profile_case("s2/Elliptical/ell_17x9_zeros_r2_o0_samples12", 17, 9, 0, NppiPoint32f{8.5f, 4.5f}, 2.0f, 0.0f, 12,
               ImageKind::Zeros);
}

TEST(NppiEllipticalRadialProfileDiff, ManySamples) {
  profile_case("s2/Elliptical/ell_33x33_half_r1.25_o2.0_samples48", 33, 33, 0, NppiPoint32f{16.5f, 16.5f}, 1.25f,
               2.0f, 48);
}

TEST(NppiEllipticalRadialProfileDiff, Contract) {
  const int w = 21, h = 13, stride = 21;
  std::vector<Npp32f> img((size_t)stride * h, 2.0f);
  DeviceMemory<Npp32f> src(img.size());
  src.copyFromHost(img);
  DeviceMemory<NppiProfileData> prof(8);
  std::vector<NppiProfileData> init(8, NppiProfileData{-12345, kSentinel, kSentinel});
  prof.copyFromHost(init);
  NppStreamContext ctx = current_context();
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const char *fname = "nppiEllipticalRadialProfile_32f_C1R_Ctx";

  status_case("s2/Elliptical/ell_null_src", fname,
              nppiEllipticalRadialProfile_32f_C1R_Ctx(nullptr, stride * 4, NppiSize{w, h}, NppiPoint32f{10.5f, 6.5f},
                                                      2.0f, 0.0f, prof.get(), 8, ctx));
  status_case("s2/Elliptical/ell_null_profile", fname,
              nppiEllipticalRadialProfile_32f_C1R_Ctx(src.get(), stride * 4, NppiSize{w, h},
                                                      NppiPoint32f{10.5f, 6.5f}, 2.0f, 0.0f, nullptr, 8, ctx));
  status_case("s2/Elliptical/ell_neg_width", fname,
              nppiEllipticalRadialProfile_32f_C1R_Ctx(src.get(), stride * 4, NppiSize{-1, h},
                                                      NppiPoint32f{10.5f, 6.5f}, 2.0f, 0.0f, prof.get(), 8, ctx));
  status_case("s2/Elliptical/ell_neg_height", fname,
              nppiEllipticalRadialProfile_32f_C1R_Ctx(src.get(), stride * 4, NppiSize{w, -3},
                                                      NppiPoint32f{10.5f, 6.5f}, 2.0f, 0.0f, prof.get(), 8, ctx));
  status_case("s2/Elliptical/ell_zero_width", fname,
              nppiEllipticalRadialProfile_32f_C1R_Ctx(src.get(), stride * 4, NppiSize{0, h},
                                                      NppiPoint32f{10.5f, 6.5f}, 2.0f, 0.0f, prof.get(), 8, ctx));
  status_case("s2/Elliptical/ell_zero_height", fname,
              nppiEllipticalRadialProfile_32f_C1R_Ctx(src.get(), stride * 4, NppiSize{w, 0},
                                                      NppiPoint32f{10.5f, 6.5f}, 2.0f, 0.0f, prof.get(), 8, ctx));
  status_case("s2/Elliptical/ell_zero_samples", fname,
              nppiEllipticalRadialProfile_32f_C1R_Ctx(src.get(), stride * 4, NppiSize{w, h},
                                                      NppiPoint32f{10.5f, 6.5f}, 2.0f, 0.0f, prof.get(), 0, ctx));
  // Host check order: the ratio test runs before the empty-ROI no-op.
  status_case("s2/Elliptical/ell_ratio_half_zero_width", fname,
              nppiEllipticalRadialProfile_32f_C1R_Ctx(src.get(), stride * 4, NppiSize{0, h},
                                                      NppiPoint32f{10.5f, 6.5f}, 0.5f, 0.0f, prof.get(), 8, ctx));
  status_case("s2/Elliptical/ell_ratio_0999", fname,
              nppiEllipticalRadialProfile_32f_C1R_Ctx(src.get(), stride * 4, NppiSize{w, h},
                                                      NppiPoint32f{10.5f, 6.5f}, 0.999f, 0.0f, prof.get(), 8, ctx));
  status_case("s2/Elliptical/ell_ratio_nan", fname,
              nppiEllipticalRadialProfile_32f_C1R_Ctx(src.get(), stride * 4, NppiSize{w, h},
                                                      NppiPoint32f{10.5f, 6.5f}, nan, 0.0f, prof.get(), 8, ctx));
  status_case("s2/Elliptical/ell_ratio_ninf", fname,
              nppiEllipticalRadialProfile_32f_C1R_Ctx(src.get(), stride * 4, NppiSize{w, h},
                                                      NppiPoint32f{10.5f, 6.5f}, -inf, 0.0f, prof.get(), 8, ctx));
  status_case("s2/Elliptical/ell_orientation_nan", fname,
              nppiEllipticalRadialProfile_32f_C1R_Ctx(src.get(), stride * 4, NppiSize{w, h},
                                                      NppiPoint32f{10.5f, 6.5f}, 2.0f, nan, prof.get(), 8, ctx));
  clear_cuda_error();
}
