// Differential tests for nppiCircularRadialProfile_32f_C1R_Ctx.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   tools/remote5090.sh record 'NppiRadialProfileDiff.*'
//
// Contracts pinned by the 5090 impulse-scan probe
// (tools/5090_probes_s2/radial_probe.cu, artifacts/5090/s2_probe_out/rad_map.txt):
//   * ring 0 holds every pixel whose centre is less than one pixel away from
//     oCenter (one pixel for an integer centre, four for a half-pixel centre);
//   * ring k >= 1 is the digitised circle of radius k: one pixel per ROI row on
//     each side, floor(cx - v) and ceil(cx + v) with v = sqrt(k^2 - (y-cy)^2);
//     radii beyond nProfileSamples - 1 are not sampled at all, so the profile is
//     sparse and rings outside the ROI stay empty;
//   * nPixels/mean/population-stddev over the sampled pixels, {0,0,0} if empty;
//   * no step validation, negative ROIs are NPP_SIZE_ERROR, empty ones are a
//     successful no-op.

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <cstdint>
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

void status_case(const std::string &id, const char *fname, NppStatus st) {
  Case c(id, fname);
  c.status(st);
  c.finish();
}

// Runs one profile case: `pad` extra floats per row exercise the source step,
// the ROI is width x height and the centre may lie outside the ROI.
void profile_case(const std::string &id, int width, int height, int pad, NppiPoint32f center, int samples) {
  const char *fname = "nppiCircularRadialProfile_32f_C1R_Ctx";
  const int stride = width + pad;
  std::vector<Npp32f> img((size_t)stride * height, 0.0f);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < stride; ++x) {
      img[(size_t)y * stride + x] = (Npp32f)(((x * 11 + y * 7) % 23) - 11) * 0.5f + (Npp32f)(x & 3) * 0.25f;
    }
  DeviceMemory<Npp32f> src(img.size());
  src.copyFromHost(img);

  const size_t n = samples > 0 ? (size_t)samples : 1;
  DeviceMemory<NppiProfileData> prof(n);
  std::vector<NppiProfileData> init(n, NppiProfileData{-12345, kSentinel, kSentinel});
  prof.copyFromHost(init);

  NppStreamContext ctx = current_context();
  NppStatus st = nppiCircularRadialProfile_32f_C1R_Ctx(src.get(), stride * (int)sizeof(Npp32f),
                                                       NppiSize{width, height}, center, prof.get(),
                                                       (Npp16u)samples, ctx);
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

TEST(NppiRadialProfileDiff, Circular_HalfPixels) {
  profile_case("s2/Radial/circ_33x33_half_samples24", 33, 33, 0, NppiPoint32f{16.5f, 16.5f}, 24);
}

TEST(NppiRadialProfileDiff, Circular_PaddedStep) {
  profile_case("s2/Radial/circ_37x29_pad5_samples19", 37, 29, 5, NppiPoint32f{18.0f, 14.0f}, 19);
}

TEST(NppiRadialProfileDiff, Circular_IntegerCenter) {
  profile_case("s2/Radial/circ_33x33_int_center_samples16", 33, 33, 2, NppiPoint32f{16.0f, 16.0f}, 16);
}

TEST(NppiRadialProfileDiff, Circular_OffCenter) {
  profile_case("s2/Radial/circ_33x33_off_center_samples20", 33, 33, 0, NppiPoint32f{10.5f, 20.5f}, 20);
}

TEST(NppiRadialProfileDiff, Circular_CenterOutside) {
  profile_case("s2/Radial/circ_33x33_center_out_samples10", 33, 33, 0, NppiPoint32f{-3.5f, -2.5f}, 10);
  profile_case("s2/Radial/circ_33x33_center_near_samples8", 33, 33, 0, NppiPoint32f{31.75f, 0.25f}, 8);
}

TEST(NppiRadialProfileDiff, Circular_ManySamples) {
  profile_case("s2/Radial/circ_33x33_samples48", 33, 33, 0, NppiPoint32f{16.5f, 16.5f}, 48);
  profile_case("s2/Radial/circ_17x9_samples12", 17, 9, 3, NppiPoint32f{8.5f, 4.5f}, 12);
}

TEST(NppiRadialProfileDiff, Circular_Contract) {
  const int w = 21, h = 13, stride = 21;
  std::vector<Npp32f> img((size_t)stride * h, 2.0f);
  DeviceMemory<Npp32f> src(img.size());
  src.copyFromHost(img);
  DeviceMemory<NppiProfileData> prof(8);
  std::vector<NppiProfileData> init(8, NppiProfileData{-12345, kSentinel, kSentinel});
  prof.copyFromHost(init);
  NppStreamContext ctx = current_context();

  status_case("s2/Radial/circ_null_src", "nppiCircularRadialProfile_32f_C1R_Ctx",
              nppiCircularRadialProfile_32f_C1R_Ctx(nullptr, stride * 4, NppiSize{w, h}, NppiPoint32f{10.5f, 6.5f},
                                                    prof.get(), 8, ctx));
  status_case("s2/Radial/circ_null_profile", "nppiCircularRadialProfile_32f_C1R_Ctx",
              nppiCircularRadialProfile_32f_C1R_Ctx(src.get(), stride * 4, NppiSize{w, h}, NppiPoint32f{10.5f, 6.5f},
                                                    nullptr, 8, ctx));
  status_case("s2/Radial/circ_neg_width", "nppiCircularRadialProfile_32f_C1R_Ctx",
              nppiCircularRadialProfile_32f_C1R_Ctx(src.get(), stride * 4, NppiSize{-1, h}, NppiPoint32f{10.5f, 6.5f},
                                                    prof.get(), 8, ctx));
  status_case("s2/Radial/circ_neg_height", "nppiCircularRadialProfile_32f_C1R_Ctx",
              nppiCircularRadialProfile_32f_C1R_Ctx(src.get(), stride * 4, NppiSize{w, -3}, NppiPoint32f{10.5f, 6.5f},
                                                    prof.get(), 8, ctx));
  status_case("s2/Radial/circ_zero_width", "nppiCircularRadialProfile_32f_C1R_Ctx",
              nppiCircularRadialProfile_32f_C1R_Ctx(src.get(), stride * 4, NppiSize{0, h}, NppiPoint32f{10.5f, 6.5f},
                                                    prof.get(), 8, ctx));
  status_case("s2/Radial/circ_zero_height", "nppiCircularRadialProfile_32f_C1R_Ctx",
              nppiCircularRadialProfile_32f_C1R_Ctx(src.get(), stride * 4, NppiSize{w, 0}, NppiPoint32f{10.5f, 6.5f},
                                                    prof.get(), 8, ctx));
  status_case("s2/Radial/circ_zero_samples", "nppiCircularRadialProfile_32f_C1R_Ctx",
              nppiCircularRadialProfile_32f_C1R_Ctx(src.get(), stride * 4, NppiSize{w, h}, NppiPoint32f{10.5f, 6.5f},
                                                    prof.get(), 0, ctx));
  clear_cuda_error();
}

TEST(NppiRadialProfileDiff, Circular_ConstantImage) {
  const int w = 21, h = 15, stride = w;
  std::vector<Npp32f> img((size_t)stride * h, 3.5f);
  DeviceMemory<Npp32f> src(img.size());
  src.copyFromHost(img);
  const int samples = 12;
  DeviceMemory<NppiProfileData> prof((size_t)samples);
  std::vector<NppiProfileData> init((size_t)samples, NppiProfileData{-12345, kSentinel, kSentinel});
  prof.copyFromHost(init);
  NppStreamContext ctx = current_context();
  NppStatus st = nppiCircularRadialProfile_32f_C1R_Ctx(src.get(), stride * 4, NppiSize{w, h},
                                                       NppiPoint32f{10.5f, 7.5f}, prof.get(), samples, ctx);
  clear_cuda_error();
  Case c("s2/Radial/circ_constant_image", "nppiCircularRadialProfile_32f_C1R_Ctx");
  c.status(st);
  c.input(img.data(), img.size() * sizeof(Npp32f));
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
  c.host_output(means.data(), means.size(), ValueKind::F32, 1e-6, 1e-6);
  c.host_output(sds.data(), sds.size(), ValueKind::F32, 1e-6, 1e-6);
  c.finish();
}
