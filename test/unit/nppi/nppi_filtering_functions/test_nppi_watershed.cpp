// Sanity tests for the best-effort nppiSegmentWatershed_16u implementation.
// The reference pSrcDst output is an internal state image (not reproduced);
// these tests pin this executor's documented topographic-watershed behaviour
// and the measured status contract.
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <npp.h>

#include <algorithm>
#include <vector>

namespace {

constexpr int W = 16, H = 16;

struct Result {
  NppStatus status;
  std::vector<Npp16u> image;
  std::vector<Npp32u> labels;
};

Result run(const std::vector<Npp16u> &src, NppiNorm norm, NppiWatershedSegmentBoundaryType boundary,
           bool wantLabels) {
  Result r;
  Npp16u *dSrc = nullptr;
  Npp32u *dLabels = nullptr;
  Npp8u *dBuffer = nullptr;
  size_t bufferSize = 0;
  EXPECT_EQ(nppiSegmentWatershedGetBufferSize_16u_C1R({W, H}, &bufferSize), NPP_NO_ERROR);
  cudaMalloc(&dSrc, src.size() * sizeof(Npp16u));
  cudaMalloc(&dLabels, src.size() * sizeof(Npp32u));
  cudaMalloc(&dBuffer, bufferSize);
  cudaMemcpy(dSrc, src.data(), src.size() * sizeof(Npp16u), cudaMemcpyHostToDevice);
  cudaMemset(dLabels, 0, src.size() * sizeof(Npp32u));
  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  r.status = nppiSegmentWatershed_16u_C1IR_Ctx(dSrc, W * sizeof(Npp16u), wantLabels ? dLabels : nullptr,
                                               W * sizeof(Npp32u), norm, boundary, {W, H}, dBuffer, ctx);
  cudaDeviceSynchronize();
  r.image.resize(src.size());
  r.labels.resize(src.size());
  cudaMemcpy(r.image.data(), dSrc, r.image.size() * sizeof(Npp16u), cudaMemcpyDeviceToHost);
  if (wantLabels) cudaMemcpy(r.labels.data(), dLabels, r.labels.size() * sizeof(Npp32u), cudaMemcpyDeviceToHost);
  cudaFree(dSrc);
  cudaFree(dLabels);
  cudaFree(dBuffer);
  return r;
}

std::vector<Npp16u> twoBasins() {
  std::vector<Npp16u> img((size_t)W * H);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const int a = (x - 4) * (x - 4) + (y - 8) * (y - 8);
      const int b = (x - 11) * (x - 11) + (y - 8) * (y - 8);
      img[(size_t)y * W + x] = (Npp16u)std::min(a, b);
    }
  }
  return img;
}

}  // namespace

TEST(NppiWatershed, TwoBasins) {
  const std::vector<Npp16u> img = twoBasins();
  Result r = run(img, nppiNormL1, NPP_WATERSHED_SEGMENT_BOUNDARIES_NONE, true);
  ASSERT_EQ(r.status, NPP_NO_ERROR);

  // NONE keeps the basin fill: every pixel takes the intensity of its seed
  // (the descending minimum).  Both minima of this image are 0, so the whole
  // output is 0 even though the source is a distance-like ramp.
  std::vector<Npp16u> want(img.size());
  for (size_t i = 0; i < img.size(); ++i) want[i] = img[r.labels[i]];
  EXPECT_EQ(r.image, want);

  // Two distinct seeds, and the valley pixels near each minimum share them.
  const Npp32u left = r.labels[(size_t)8 * W + 4];
  const Npp32u right = r.labels[(size_t)8 * W + 11];
  EXPECT_EQ(left, (Npp32u)(8 * W + 4));
  EXPECT_EQ(right, (Npp32u)(8 * W + 11));
  EXPECT_NE(left, right);
  EXPECT_EQ(r.labels[(size_t)8 * W + 3], left);
  EXPECT_EQ(r.labels[(size_t)8 * W + 12], right);

  // Some pixel between the basins must lie on a boundary (different labels).
  bool split = false;
  for (int x = 5; x < 11; ++x) {
    if (r.labels[(size_t)8 * W + x] == left && r.labels[(size_t)8 * W + x + 1] == right) split = true;
  }
  EXPECT_TRUE(split);
}

TEST(NppiWatershed, BoundaryPolicies) {
  const std::vector<Npp16u> img = twoBasins();
  Result none = run(img, nppiNormL1, NPP_WATERSHED_SEGMENT_BOUNDARIES_NONE, true);
  Result black = run(img, nppiNormL1, NPP_WATERSHED_SEGMENT_BOUNDARIES_BLACK, true);
  Result white = run(img, nppiNormL1, NPP_WATERSHED_SEGMENT_BOUNDARIES_WHITE, true);
  Result only = run(img, nppiNormL1, NPP_WATERSHED_SEGMENT_BOUNDARIES_ONLY, true);
  ASSERT_EQ(black.status, NPP_NO_ERROR);
  ASSERT_EQ(white.status, NPP_NO_ERROR);
  ASSERT_EQ(only.status, NPP_NO_ERROR);

  // ONLY paints 0 exactly on boundary pixels and 65535 everywhere else, so it
  // identifies the boundary set without ambiguity (a basin whose seed
  // intensity is 65535 would not be distinguishable from a WHITE boundary).
  int boundaries = 0;
  for (size_t i = 0; i < img.size(); ++i) {
    const bool boundary = only.image[i] == 0;
    const Npp16u fill = img[none.labels[i]];  // seed intensity of the basin
    if (boundary) {
      ++boundaries;
      EXPECT_EQ(black.image[i], 0);
      EXPECT_EQ(white.image[i], 65535);
    } else {
      EXPECT_EQ(only.image[i], 65535);
      EXPECT_EQ(black.image[i], fill);
      EXPECT_EQ(white.image[i], fill);
    }
  }
  EXPECT_GT(boundaries, 0);
}

TEST(NppiWatershed, Contract) {
  std::vector<Npp16u> img((size_t)W * H, 5);
  Npp16u *dSrc = nullptr;
  Npp32u *dLabels = nullptr;
  Npp8u *dBuffer = nullptr;
  size_t bufferSize = 0;
  nppiSegmentWatershedGetBufferSize_16u_C1R({W, H}, &bufferSize);
  cudaMalloc(&dSrc, img.size() * sizeof(Npp16u));
  cudaMalloc(&dLabels, img.size() * sizeof(Npp32u));
  cudaMalloc(&dBuffer, bufferSize);
  cudaMemcpy(dSrc, img.data(), img.size() * sizeof(Npp16u), cudaMemcpyHostToDevice);
  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);

  EXPECT_EQ(nppiSegmentWatershed_16u_C1IR_Ctx(nullptr, W * 2, dLabels, W * 4, nppiNormL1,
                                             NPP_WATERSHED_SEGMENT_BOUNDARIES_NONE, {W, H}, dBuffer, ctx),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiSegmentWatershed_16u_C1IR_Ctx(dSrc, W * 2, dLabels, W * 4, nppiNormL1,
                                             NPP_WATERSHED_SEGMENT_BOUNDARIES_NONE, {W, H}, nullptr, ctx),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiSegmentWatershed_16u_C1IR_Ctx(dSrc, W * 2, dLabels, W * 4, nppiNormL1,
                                             NPP_WATERSHED_SEGMENT_BOUNDARIES_NONE, {0, H}, dBuffer, ctx),
            NPP_SIZE_ERROR);
  EXPECT_EQ(nppiSegmentWatershed_16u_C1IR_Ctx(dSrc, W * 2, dLabels, W * 4, (NppiNorm)7,
                                             NPP_WATERSHED_SEGMENT_BOUNDARIES_NONE, {W, H}, dBuffer, ctx),
            NPP_BAD_ARGUMENT_ERROR);
  EXPECT_EQ(nppiSegmentWatershed_16u_C1IR_Ctx(dSrc, W * 2, dLabels, W * 4, nppiNormL1,
                                             (NppiWatershedSegmentBoundaryType)9, {W, H}, dBuffer, ctx),
            NPP_ERROR);
  // Marker labels are optional.
  EXPECT_EQ(nppiSegmentWatershed_16u_C1IR_Ctx(dSrc, W * 2, nullptr, 0, nppiNormL1,
                                             NPP_WATERSHED_SEGMENT_BOUNDARIES_NONE, {W, H}, dBuffer, ctx),
            NPP_NO_ERROR);
  cudaFree(dSrc);
  cudaFree(dLabels);
  cudaFree(dBuffer);
}
