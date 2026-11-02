// Sanity tests for the best-effort contour interpolation/geometry functions.
// NPP 13.4 exports the MarchingSquares symbols but no longer declares them (or
// their direction structs) in its public headers, and the legacy
// OutputGeometryLists symbol is gone entirely, so these paths have no
// differential oracle; the tests pin this executor's documented behaviour.
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <npp.h>

#include <vector>

namespace {

constexpr int W = 8, H = 8, N = W * H, kMaxLabels = 4;

struct Scene {
  std::vector<Npp8u> contours = std::vector<Npp8u>(N, 0);
  std::vector<NppiPoint32f> interp32 = std::vector<NppiPoint32f>(N, NppiPoint32f{-1.0f, -1.0f});
  std::vector<NppiPoint64f> interp64 = std::vector<NppiPoint64f>(N, NppiPoint64f{-1.0, -1.0});
  std::vector<NppiContourPixelDirectionInfo> direction = std::vector<NppiContourPixelDirectionInfo>(N);
  std::vector<NppiContourPixelGeometryInfo> geometryDev = std::vector<NppiContourPixelGeometryInfo>(N);
  std::vector<NppiContourPixelGeometryInfo> geometryHost = std::vector<NppiContourPixelGeometryInfo>(N);
  std::vector<NppiPoint32f> interpGeometry32 = std::vector<NppiPoint32f>(N);
  std::vector<NppiPoint64f> interpGeometry64 = std::vector<NppiPoint64f>(N);
  std::vector<Npp32u> found = std::vector<Npp32u>(kMaxLabels, 7);
  std::vector<Npp32u> offset = std::vector<Npp32u>(kMaxLabels, 7);
  std::vector<NppiContourBlockSegment> blockHost = std::vector<NppiContourBlockSegment>(kMaxLabels);
};

struct DeviceScene {
  Npp8u *contours = nullptr;
  NppiPoint32f *interp32 = nullptr;
  NppiPoint64f *interp64 = nullptr;
  NppiContourPixelDirectionInfo *direction = nullptr;
  NppiContourPixelGeometryInfo *geometry = nullptr;
  NppiPoint32f *interpGeometry32 = nullptr;
  NppiPoint64f *interpGeometry64 = nullptr;
  Npp32u *offset = nullptr;
  NppiContourBlockSegment *block = nullptr;

  DeviceScene() {
    cudaMalloc(&contours, N);
    cudaMalloc(&interp32, N * sizeof(NppiPoint32f));
    cudaMalloc(&interp64, N * sizeof(NppiPoint64f));
    cudaMalloc(&direction, N * sizeof(NppiContourPixelDirectionInfo));
    cudaMalloc(&geometry, N * sizeof(NppiContourPixelGeometryInfo));
    cudaMalloc(&interpGeometry32, N * sizeof(NppiPoint32f));
    cudaMalloc(&interpGeometry64, N * sizeof(NppiPoint64f));
    cudaMalloc(&offset, kMaxLabels * sizeof(Npp32u));
    cudaMalloc(&block, kMaxLabels * sizeof(NppiContourBlockSegment));
    cudaMemset(direction, 0, N * sizeof(NppiContourPixelDirectionInfo));
  }
  ~DeviceScene() {
    cudaFree(contours);
    cudaFree(interp32);
    cudaFree(interp64);
    cudaFree(direction);
    cudaFree(geometry);
    cudaFree(interpGeometry32);
    cudaFree(interpGeometry64);
    cudaFree(offset);
    cudaFree(block);
  }
};

constexpr int kDirStep = W * (int)sizeof(NppiContourPixelDirectionInfo);

}  // namespace

TEST(NppiContours, MarchingSquares32f) {
  Scene host;
  host.contours[(size_t)3 * W + 4] = 1;
  DeviceScene dev;
  cudaMemcpy(dev.contours, host.contours.data(), N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev.interp32, host.interp32.data(), N * sizeof(NppiPoint32f), cudaMemcpyHostToDevice);
  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  const NppStatus st = nppiContoursImageMarchingSquaresInterpolation_32f_C1R_Ctx(
      dev.contours, W, dev.interp32, W * (int)sizeof(NppiPoint32f), dev.direction, kDirStep, dev.geometry,
      host.geometryHost.data(), dev.interpGeometry32, host.found.data(), dev.offset, host.offset.data(), N,
      kMaxLabels, 1, 2, dev.block, host.blockHost.data(), {W, H}, ctx);
  ASSERT_EQ(st, NPP_NO_ERROR);
  cudaDeviceSynchronize();

  std::vector<NppiPoint32f> out(N);
  cudaMemcpy(out.data(), dev.interp32, out.size() * sizeof(NppiPoint32f), cudaMemcpyDeviceToHost);
  EXPECT_FLOAT_EQ(out[(size_t)3 * W + 4].x, 4.0f);
  EXPECT_FLOAT_EQ(out[(size_t)3 * W + 4].y, 3.0f);
  for (int i = 0; i < N; ++i) {
    if (i == 3 * W + 4) continue;
    EXPECT_FLOAT_EQ(out[(size_t)i].x, 0.0f);
    EXPECT_FLOAT_EQ(out[(size_t)i].y, 0.0f);
  }
  EXPECT_EQ(host.found[1], 0u);
  EXPECT_EQ(host.offset[2], 0u);
  EXPECT_EQ(host.blockHost[2].nMarkerLabelID, 2u);
}

TEST(NppiContours, MarchingSquares64f) {
  Scene host;
  host.contours[(size_t)1 * W + 2] = 1;
  DeviceScene dev;
  cudaMemcpy(dev.contours, host.contours.data(), N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev.interp64, host.interp64.data(), N * sizeof(NppiPoint64f), cudaMemcpyHostToDevice);
  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  const NppStatus st = nppiContoursImageMarchingSquaresInterpolation_64f_C1R_Ctx(
      dev.contours, W, dev.interp64, W * (int)sizeof(NppiPoint64f), dev.direction, kDirStep, dev.geometry,
      host.geometryHost.data(), dev.interpGeometry64, host.found.data(), dev.offset, host.offset.data(), N,
      kMaxLabels, 0, 1, dev.block, host.blockHost.data(), {W, H}, ctx);
  ASSERT_EQ(st, NPP_NO_ERROR);
  std::vector<NppiPoint64f> out(N);
  cudaMemcpy(out.data(), dev.interp64, out.size() * sizeof(NppiPoint64f), cudaMemcpyDeviceToHost);
  EXPECT_DOUBLE_EQ(out[(size_t)1 * W + 2].x, 2.0);
  EXPECT_DOUBLE_EQ(out[(size_t)1 * W + 2].y, 1.0);
}

TEST(NppiContours, Contract) {
  Scene host;
  DeviceScene dev;
  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  EXPECT_EQ(nppiContoursImageMarchingSquaresInterpolation_32f_C1R_Ctx(
                nullptr, W, dev.interp32, W * 8, dev.direction, kDirStep, dev.geometry, host.geometryHost.data(),
                dev.interpGeometry32, host.found.data(), dev.offset, host.offset.data(), N, kMaxLabels, 1, 2,
                dev.block, host.blockHost.data(), {W, H}, ctx),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiContoursImageMarchingSquaresInterpolation_32f_C1R_Ctx(
                dev.contours, W, dev.interp32, W * 8, dev.direction, kDirStep, dev.geometry,
                host.geometryHost.data(), dev.interpGeometry32, host.found.data(), dev.offset, host.offset.data(),
                N, kMaxLabels, 3, 2, dev.block, host.blockHost.data(), {W, H}, ctx),
            NPP_BAD_ARGUMENT_ERROR);
}

TEST(NppiContours, OutputGeometryLists) {
  const int labels = 2;
  const int wx = 4, hx = 4;
  std::vector<NppiCompressedMarkerLabelsInfo> marker(labels);
  marker[0].nContourPixelCount = 2;
  marker[0].nContourPixelsFound = 2;
  marker[1].nContourPixelCount = 3;
  marker[1].nContourPixelsFound = 3;
  std::vector<NppiContourPixelGeometryInfo> geometry(8);
  std::vector<Npp8u> image((size_t)wx * hx, 0);
  std::vector<Npp32u> counts(labels, 0), found(labels, 0), offset(labels, 0);
  const NppStatus st = nppiCompressedMarkerLabelsUFContoursOutputGeometryLists_C1R(
      geometry.data(), marker.data(), image.data(), wx, counts.data(), found.data(), offset.data(), labels, 0,
      labels - 1, 1, {wx, hx});
  ASSERT_EQ(st, NPP_NO_ERROR);
  EXPECT_EQ(offset[0], 0u);
  EXPECT_EQ(offset[1], 2u);
  EXPECT_EQ(counts[0], 2u);
  EXPECT_EQ(counts[1], 3u);
  EXPECT_EQ(found[1], 3u);
  EXPECT_EQ(geometry[0].oContourOrderedGeometryLocation.x, 0);
  EXPECT_EQ(geometry[0].oContourOrderedGeometryLocation.y, 0);
  EXPECT_EQ(geometry[3].oContourOrderedGeometryLocation.x, 3);
  EXPECT_EQ(geometry[3].oContourOrderedGeometryLocation.y, 0);
  EXPECT_EQ(geometry[4].oContourOrderedGeometryLocation.x, 0);
  EXPECT_EQ(geometry[4].oContourOrderedGeometryLocation.y, 1);
  EXPECT_EQ(geometry[2].nOrderIndex, 0);
  EXPECT_EQ(geometry[2].nFirstIndex, 2u);
  EXPECT_EQ(geometry[2].nLastIndex, 4u);
  EXPECT_EQ(geometry[4].nOrderIndex, 2);

  EXPECT_EQ(nppiCompressedMarkerLabelsUFContoursOutputGeometryLists_C1R(
                nullptr, marker.data(), image.data(), wx, counts.data(), found.data(), offset.data(), labels, 0,
                labels - 1, 1, {wx, hx}),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiCompressedMarkerLabelsUFContoursOutputGeometryLists_C1R(
                geometry.data(), marker.data(), image.data(), wx, counts.data(), found.data(), offset.data(), labels,
                1, 0, 1, {wx, hx}),
            NPP_BAD_ARGUMENT_ERROR);
}
