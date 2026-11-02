// Sanity and status-contract tests for nppiEllipticalRadialProfile_32f_C1R_Ctx.
// The metric, per-ring scan window and host check order come from the NPP 13.4
// SASS and the 5090 probes; value conformance is covered by the differential
// test generated from the 5090 goldens.
#include <cmath>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <npp.h>

namespace {

constexpr int W = 33, H = 33;

std::vector<Npp32f> makeConstant(Npp32f value) { return std::vector<Npp32f>(W * H, value); }

NppStatus call(const std::vector<Npp32f> &image, float ratio, float orientation, Npp16u samples,
               std::vector<NppiProfileData> *profile) {
  Npp32f *dSrc = nullptr;
  NppiProfileData *dProfile = nullptr;
  cudaMalloc(&dSrc, image.size() * sizeof(Npp32f));
  cudaMemcpy(dSrc, image.data(), image.size() * sizeof(Npp32f), cudaMemcpyHostToDevice);
  cudaMalloc(&dProfile, samples * sizeof(NppiProfileData));
  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  NppStatus status = nppiEllipticalRadialProfile_32f_C1R_Ctx(dSrc, W * sizeof(Npp32f), {W, H}, {16.0f, 16.0f},
                                                             ratio, orientation, dProfile, samples, ctx);
  cudaDeviceSynchronize();
  profile->resize(samples);
  cudaMemcpy(profile->data(), dProfile, samples * sizeof(NppiProfileData), cudaMemcpyDeviceToHost);
  cudaFree(dSrc);
  cudaFree(dProfile);
  return status;
}

}  // namespace

TEST(NppiEllipticalRadialProfile, ConstantImage) {
  std::vector<NppiProfileData> profile;
  const NppStatus status = call(makeConstant(5.0f), 2.0f, 0.0f, 12, &profile);
  ASSERT_EQ(status, NPP_NO_ERROR);
  int total = 0;
  for (const NppiProfileData &ring : profile) {
    total += ring.nPixels;
    if (ring.nPixels > 0) {
      EXPECT_NEAR(ring.nMeanIntensity, 5.0f, 1e-5f);
      EXPECT_NEAR(ring.nStdDevIntensity, 0.0f, 1e-5f);
    }
  }
  EXPECT_GT(total, 0);
}

TEST(NppiEllipticalRadialProfile, StatusContract) {
  std::vector<NppiProfileData> profile;
  EXPECT_EQ(call(makeConstant(1.0f), 0.5f, 0.0f, 8, &profile), NPP_RANGE_ERROR);
  EXPECT_EQ(call(makeConstant(1.0f), 0.0f, 0.0f, 8, &profile), NPP_RANGE_ERROR);
  EXPECT_EQ(call(makeConstant(1.0f), 1.0f, 0.0f, 8, &profile), NPP_NO_ERROR);
  EXPECT_EQ(call(makeConstant(1.0f), 2.0f, 0.3f, 8, &profile), NPP_NO_ERROR);
  EXPECT_EQ(call(makeConstant(1.0f), std::nanf(""), 0.0f, 8, &profile), NPP_NO_ERROR);
  EXPECT_EQ(call(makeConstant(1.0f), 1.0f, std::nanf(""), 8, &profile), NPP_NO_ERROR);

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  NppiProfileData out{};
  EXPECT_EQ(nppiEllipticalRadialProfile_32f_C1R_Ctx(nullptr, W * 4, {W, H}, {16, 16}, 1.0f, 0.0f, &out, 8, ctx),
            NPP_NULL_POINTER_ERROR);

  // Check order measured on the 5090: null pointers, negative ROI, ratio < 1,
  // and only then the empty-ROI/empty-samples no-ops.  NaN and +inf pass the
  // plain `ratio < 1` comparison; -inf does not.
  Npp32f *dSrc = nullptr;
  NppiProfileData *dOut = nullptr;
  cudaMalloc(&dSrc, W * H * sizeof(Npp32f));
  cudaMemset(dSrc, 0, W * H * sizeof(Npp32f));
  cudaMalloc(&dOut, 8 * sizeof(NppiProfileData));
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  EXPECT_EQ(nppiEllipticalRadialProfile_32f_C1R_Ctx(dSrc, W * 4, {-1, H}, {16, 16}, 0.5f, 0.0f, dOut, 8, ctx),
            NPP_SIZE_ERROR);
  EXPECT_EQ(nppiEllipticalRadialProfile_32f_C1R_Ctx(dSrc, W * 4, {0, H}, {16, 16}, 0.5f, 0.0f, dOut, 8, ctx),
            NPP_RANGE_ERROR);
  EXPECT_EQ(nppiEllipticalRadialProfile_32f_C1R_Ctx(dSrc, W * 4, {W, H}, {16, 16}, 0.5f, 0.0f, dOut, 0, ctx),
            NPP_RANGE_ERROR);
  EXPECT_EQ(nppiEllipticalRadialProfile_32f_C1R_Ctx(dSrc, W * 4, {W, H}, {16, 16}, 0.999f, 0.0f, dOut, 8, ctx),
            NPP_RANGE_ERROR);
  EXPECT_EQ(nppiEllipticalRadialProfile_32f_C1R_Ctx(dSrc, W * 4, {W, H}, {16, 16}, inf, 0.0f, dOut, 8, ctx),
            NPP_NO_ERROR);
  EXPECT_EQ(nppiEllipticalRadialProfile_32f_C1R_Ctx(dSrc, W * 4, {W, H}, {16, 16}, -inf, 0.0f, dOut, 8, ctx),
            NPP_RANGE_ERROR);
  EXPECT_EQ(nppiEllipticalRadialProfile_32f_C1R_Ctx(dSrc, W * 4, {W, H}, {16, 16}, nan, 0.0f, dOut, 8, ctx),
            NPP_NO_ERROR);
  EXPECT_EQ(nppiEllipticalRadialProfile_32f_C1R_Ctx(dSrc, W * 4, {0, H}, {16, 16}, nan, 0.0f, dOut, 8, ctx),
            NPP_NO_ERROR);
  cudaFree(dSrc);
  cudaFree(dOut);
}

TEST(NppiEllipticalRadialProfile, AnisotropyChangesRings) {
  std::vector<NppiProfileData> isotropic, anisotropic;
  ASSERT_EQ(call(makeConstant(1.0f), 1.0f, 0.0f, 12, &isotropic), NPP_NO_ERROR);
  ASSERT_EQ(call(makeConstant(1.0f), 2.0f, 0.0f, 12, &anisotropic), NPP_NO_ERROR);
  bool differs = false;
  for (int i = 0; i < 12; ++i) {
    if (isotropic[i].nPixels != anisotropic[i].nPixels) differs = true;
  }
  EXPECT_TRUE(differs);
}
