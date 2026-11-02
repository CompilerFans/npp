// Differential tests for the host-only Gauss pyramid destination-ROI helpers
// nppiGetFilterGaussPyramidLayer{Down,Up}BorderDstROI.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiPyramidRoiDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records ./diff_tests \
//     --gtest_filter='NppiPyramidRoiDiff.*'
//
// The grid pins the float32 formula (including the extent-1 clamp of the down
// helper and the w+1 lower clamp of both up outputs), all validation statuses
// and the untouched (-1, -1) outputs on error.
#include "npp_diff_harness.h"

#include <cstdio>
#include <string>

using namespace npp_diff;

namespace {

std::string rateTag(float rate) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.9g", rate);
  return buf;
}

void runDown(int width, int height, float rate) {
  NppiSize size{-7, -7};
  clear_cuda_error();
  NppStatus st = nppiGetFilterGaussPyramidLayerDownBorderDstROI(width, height, &size, rate);
  Case c(make_id({"pyramidROI", "down", std::to_string(width), std::to_string(height), rateTag(rate)}),
         "nppiGetFilterGaussPyramidLayerDownBorderDstROI");
  c.status(st);
  c.host_output<int>(&size.width, 2, ValueKind::I32);
  c.finish();
}

void runDownNull() {
  clear_cuda_error();
  NppStatus st = nppiGetFilterGaussPyramidLayerDownBorderDstROI(16, 16, nullptr, 2.0f);
  Case c(make_id({"pyramidROI", "down", "null"}), "nppiGetFilterGaussPyramidLayerDownBorderDstROI");
  c.status(st);
  c.finish();
}

void runUp(int width, int height, float rate) {
  NppiSize minSize{-7, -7};
  NppiSize maxSize{-7, -7};
  clear_cuda_error();
  NppStatus st = nppiGetFilterGaussPyramidLayerUpBorderDstROI(width, height, &minSize, &maxSize, rate);
  Case c(make_id({"pyramidROI", "up", std::to_string(width), std::to_string(height), rateTag(rate)}),
         "nppiGetFilterGaussPyramidLayerUpBorderDstROI");
  c.status(st);
  c.host_output<int>(&minSize.width, 2, ValueKind::I32);
  c.host_output<int>(&maxSize.width, 2, ValueKind::I32);
  c.finish();
}

void runUpNull(bool nullMin) {
  NppiSize minSize{-7, -7};
  NppiSize maxSize{-7, -7};
  clear_cuda_error();
  NppStatus st = nppiGetFilterGaussPyramidLayerUpBorderDstROI(
      16, 16, nullMin ? nullptr : &minSize, nullMin ? &maxSize : nullptr, 2.0f);
  Case c(make_id({"pyramidROI", "up", nullMin ? "null_min" : "null_max"}),
         "nppiGetFilterGaussPyramidLayerUpBorderDstROI");
  c.status(st);
  c.finish();
}

const int kSizes[][2] = {{1, 1},  {2, 2},  {3, 3},   {4, 4},   {5, 5},   {7, 7},   {8, 8},
                         {15, 15}, {16, 16}, {17, 17}, {31, 13}, {64, 40}, {100, 37},
                         {1, 40}, {40, 1}, {2, 3},   {3, 2}};
const float kRates[] = {1.0001f, 1.05f, 1.1f, 1.2f,  1.25f, 1.3333334f, 1.5f, 1.75f,
                        2.0f,    2.5f,  3.0f, 3.5f,  4.0f,  5.0f,       7.0f, 10.0f};

}  // namespace

TEST(NppiPyramidRoiDiff, Down) {
  for (const auto &size : kSizes) {
    for (float rate : kRates) {
      runDown(size[0], size[1], rate);
    }
  }
}

TEST(NppiPyramidRoiDiff, DownInvalid) {
  const int badSizes[][2] = {{0, 10}, {10, 0}, {-1, 10}, {10, -1}};
  const float badRates[] = {1.0f, 0.5f, 10.5f, 0.0f, -1.0f};
  for (const auto &size : badSizes) {
    runDown(size[0], size[1], 2.0f);
  }
  for (float rate : badRates) {
    runDown(16, 16, rate);
  }
  runDownNull();
}

TEST(NppiPyramidRoiDiff, Up) {
  for (const auto &size : kSizes) {
    for (float rate : kRates) {
      runUp(size[0], size[1], rate);
    }
  }
}

TEST(NppiPyramidRoiDiff, UpInvalid) {
  const int badSizes[][2] = {{0, 10}, {10, 0}, {-1, 10}, {10, -1}};
  const float badRates[] = {1.0f, 0.5f, 10.5f, 0.0f, -1.0f};
  for (const auto &size : badSizes) {
    runUp(size[0], size[1], 2.0f);
  }
  for (float rate : badRates) {
    runUp(16, 16, rate);
  }
  runUpNull(true);
  runUpNull(false);
}
