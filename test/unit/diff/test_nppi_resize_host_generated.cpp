// Differential tests for the host-only resize helpers (package g4):
//   nppiGetResizeRect, nppiGetResizeTiledSourceOffset and
//   nppiResizeAdvancedGetBufferHostSize_8u_C1R.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   tools/remote5090.sh record 'NppiResizeHostDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppiResizeHostDiff.*'
//
// Measured semantics (5090, NPP 13.4), see also the header comment of
// src/nppi/nppi_geometry_transforms/nppi_resize_host.cpp:
//   * nppiGetResizeRect validation order null -> src size -> factor ->
//     interpolation; result components round half away from zero
//     (x = round(src.x*fx + sx), width = round(src.width*fx)).
//     Accepted interpolation modes: NN, LINEAR, CUBIC, CUBIC2P_BSPLINE,
//     CUBIC2P_CATMULLROM, CUBIC2P_B05C03, SUPER, LANCZOS.  LANCZOS3_ADVANCED
//     and the NPPI_SMOOTH_EDGE flag are rejected.
//   * nppiGetResizeTiledSourceOffset: any rect width/height < 1 ->
//     NPP_RESIZE_NO_OPERATION_ERROR (checked before the null pointer);
//     source offsets are ignored; results truncate toward zero
//     (0.75 -> 0, -0.75 -> 0, 10.5 -> 10, -10.5 -> -10).
//   * nppiResizeAdvancedGetBufferHostSize_8u_C1R: buffer size is
//     4 * dstWidth * srcHeight bytes with 32-bit wrap-around and no size
//     validation; any interpolation mode other than LANCZOS3_ADVANCED
//     (including SMOOTH_EDGE|LANCZOS3_ADVANCED) returns
//     NPP_NOT_IMPLEMENTED_ERROR.
//   * NVIDIA 13.4 crashes on a NULL hpBufferSize (documented as
//     NPP_NULL_POINTER_ERROR); the path cannot be recorded and is not
//     exercised here.  mcMPP returns the documented error instead.

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <cstring>
#include <string>

using namespace npp_diff;
using namespace npp_functional_test;

namespace {

// Deterministic poison so that the recorded bytes of an error case (whose
// outputs are never compared) stay stable between record and verify runs.
constexpr int kPoison = -1182270313;

void resize_rect_case(const char *id, NppiRect src, double fx, double fy, double sx, double sy, int interpolation) {
  Case c(id, "nppiGetResizeRect");
  NppiRect rect{kPoison, kPoison, kPoison, kPoison};
  NppStatus st = nppiGetResizeRect(src, &rect, fx, fy, sx, sy, interpolation);
  c.status(st);
  c.host_output(&rect, 4, ValueKind::I32);
  c.finish();
}

void tiled_offset_case(const char *id, NppiRect src, NppiRect dst) {
  Case c(id, "nppiGetResizeTiledSourceOffset");
  NppiPoint point{kPoison, kPoison};
  NppStatus st = nppiGetResizeTiledSourceOffset(src, dst, &point);
  c.status(st);
  c.host_output(&point, 2, ValueKind::I32);
  c.finish();
}

void buffer_size_case(const char *id, NppiSize src, NppiSize dst, int interpolation) {
  Case c(id, "nppiResizeAdvancedGetBufferHostSize_8u_C1R");
  int size = kPoison;
  NppStatus st = nppiResizeAdvancedGetBufferHostSize_8u_C1R(src, dst, &size, interpolation);
  c.status(st);
  c.host_output(&size, 1, ValueKind::I32);
  c.finish();
}

}  // namespace

class NppiResizeHostDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

TEST_F(NppiResizeHostDiff, GetResizeRectValues) {
  resize_rect_case("resize-host/getrect/up2", {2, 3, 10, 7}, 2, 2, 0, 0, NPPI_INTER_NN);
  resize_rect_case("resize-host/getrect/down05", {2, 3, 10, 7}, 0.5, 0.5, 0, 0, NPPI_INTER_NN);
  resize_rect_case("resize-host/getrect/mixed", {2, 3, 10, 7}, 1.5, 2.25, 1.3, -2.7, NPPI_INTER_LINEAR);
  resize_rect_case("resize-host/getrect/fx-neq-fy", {2, 3, 10, 7}, 2, 3, 0, 0, NPPI_INTER_LINEAR);
  resize_rect_case("resize-host/getrect/half-away-negative", {-3, -3, 7, 7}, 1, 1, -0.5, 0.5, NPPI_INTER_NN);
  resize_rect_case("resize-host/getrect/half-away-positive", {-3, -3, 7, 7}, 1, 1, 0.5, -0.5, NPPI_INTER_NN);
  resize_rect_case("resize-host/getrect/size-halves", {-2, -4, 5, 9}, 0.5, 0.5, 0, 0, NPPI_INTER_NN);
  resize_rect_case("resize-host/getrect/offset-src", {5, 5, 3, 3}, 0.5, 0.5, 2.5, 1.5, NPPI_INTER_LANCZOS);
  resize_rect_case("resize-host/getrect/thirds", {2, 3, 10, 7}, 1.0 / 3.0, 2.0 / 3.0, 0, 0, NPPI_INTER_NN);
  resize_rect_case("resize-host/getrect/identity", {0, 0, 8, 8}, 1, 1, 0, 0, NPPI_INTER_NN);
  resize_rect_case("resize-host/getrect/quarter-shift", {1, 1, 3, 3}, 2.5, 2.5, -1.25, 2.25, NPPI_INTER_LINEAR);
  resize_rect_case("resize-host/getrect/tiny", {0, 0, 1, 1}, 2, 2, 0, 0, NPPI_INTER_NN);
  resize_rect_case("resize-host/getrect/interpolation-super", {0, 0, 8, 8}, 2, 2, 0, 0, NPPI_INTER_SUPER);
  resize_rect_case("resize-host/getrect/interpolation-lanczos", {0, 0, 8, 8}, 2, 2, 0, 0, NPPI_INTER_LANCZOS);
  resize_rect_case("resize-host/getrect/interpolation-cubic", {0, 0, 8, 8}, 2, 2, 0, 0, NPPI_INTER_CUBIC);
  resize_rect_case("resize-host/getrect/interpolation-bspline", {2, 3, 10, 7}, 2, 2, 0, 0, NPPI_INTER_CUBIC2P_BSPLINE);
  resize_rect_case("resize-host/getrect/interpolation-catmullrom", {2, 3, 10, 7}, 2, 2, 0, 0,
                   NPPI_INTER_CUBIC2P_CATMULLROM);
  resize_rect_case("resize-host/getrect/interpolation-b05c03", {2, 3, 10, 7}, 2, 2, 0, 0, NPPI_INTER_CUBIC2P_B05C03);
}

TEST_F(NppiResizeHostDiff, GetResizeRectErrors) {
  resize_rect_case("resize-host/getrect/error/interp-undefined", {2, 3, 10, 7}, 2, 2, 0, 0, NPPI_INTER_UNDEFINED);
  resize_rect_case("resize-host/getrect/error/interp-advanced", {2, 3, 10, 7}, 2, 2, 0, 0,
                   NPPI_INTER_LANCZOS3_ADVANCED);
  resize_rect_case("resize-host/getrect/error/interp-unknown", {2, 3, 10, 7}, 2, 2, 0, 0, 999);
  resize_rect_case("resize-host/getrect/error/interp-smooth-edge", {2, 3, 10, 7}, 2, 2, 0, 0, (int)NPPI_SMOOTH_EDGE);
  resize_rect_case("resize-host/getrect/error/interp-smooth-edge-nn", {2, 3, 10, 7}, 2, 2, 0, 0,
                   (int)NPPI_SMOOTH_EDGE | NPPI_INTER_NN);
  resize_rect_case("resize-host/getrect/error/factor-negative-x", {2, 3, 10, 7}, -1, 2, 0, 0, NPPI_INTER_NN);
  resize_rect_case("resize-host/getrect/error/factor-negative-y", {2, 3, 10, 7}, 2, -1, 0, 0, NPPI_INTER_NN);
  resize_rect_case("resize-host/getrect/error/factor-zero", {2, 3, 10, 7}, 0, 0, 0, 0, NPPI_INTER_NN);
  resize_rect_case("resize-host/getrect/error/size-width-zero", {0, 0, 0, 8}, 2, 2, 0, 0, NPPI_INTER_NN);
  resize_rect_case("resize-host/getrect/error/size-height-zero", {0, 0, 8, 0}, 2, 2, 0, 0, NPPI_INTER_NN);
  resize_rect_case("resize-host/getrect/error/size-width-negative", {0, 0, -8, 8}, 2, 2, 0, 0, NPPI_INTER_NN);
  // Validation order: factor error wins over the interpolation error ...
  resize_rect_case("resize-host/getrect/error/order-factor-over-interp", {2, 3, 10, 7}, -1, 2, 0, 0, 999);
  // ... the size error wins over both, and the null check wins over all.
  resize_rect_case("resize-host/getrect/error/order-size-over-interp", {0, 0, 0, 8}, 2, 2, 0, 0, 999);
  resize_rect_case("resize-host/getrect/error/order-size-over-factor", {0, 0, 0, 8}, -1, 2, 0, 0, NPPI_INTER_NN);
  {
    Case c("resize-host/getrect/error/null", "nppiGetResizeRect");
    NppStatus st = nppiGetResizeRect({2, 3, 10, 7}, nullptr, 2, 2, 0, 0, NPPI_INTER_NN);
    c.status(st);
    c.finish();
  }
  {
    Case c("resize-host/getrect/error/null-over-size", "nppiGetResizeRect");
    NppStatus st = nppiGetResizeRect({0, 0, 0, 8}, nullptr, 2, 2, 0, 0, NPPI_INTER_NN);
    c.status(st);
    c.finish();
  }
}

TEST_F(NppiResizeHostDiff, TiledSourceOffsetValues) {
  tiled_offset_case("resize-host/tiled/identity-scale", {0, 0, 8, 8}, {8, 8, 8, 8});
  tiled_offset_case("resize-host/tiled/negative-tile", {0, 0, 8, 8}, {-8, -8, 8, 8});
  tiled_offset_case("resize-host/tiled/half-size", {0, 0, 8, 8}, {4, 4, 4, 4});
  tiled_offset_case("resize-host/tiled/uneven", {2, 3, 10, 7}, {20, 21, 20, 14});
  tiled_offset_case("resize-host/tiled/negative-tie", {0, 0, 10, 10}, {-3, -7, 20, 4});
  tiled_offset_case("resize-host/tiled/non-tie", {0, 0, 3, 3}, {2, 0, 8, 8});
  tiled_offset_case("resize-host/tiled/non-tie-negative", {0, 0, 3, 3}, {-2, 0, 8, 8});
  tiled_offset_case("resize-host/tiled/fraction", {0, 0, 3, 3}, {1, 1, 8, 8});
  tiled_offset_case("resize-host/tiled/upscale", {0, 0, 16, 16}, {-4, -4, 8, 8});
  tiled_offset_case("resize-host/tiled/large", {0, 0, 10, 10}, {1000000, 0, 3, 3});
  tiled_offset_case("resize-host/tiled/src-offset-ignored", {99, 99, 8, 8}, {8, 8, 8, 8});
  tiled_offset_case("resize-host/tiled/identity", {2, 3, 10, 7}, {0, 0, 10, 7});
}

TEST_F(NppiResizeHostDiff, TiledSourceOffsetErrors) {
  tiled_offset_case("resize-host/tiled/error/src-width-zero", {0, 0, 0, 8}, {0, 0, 8, 8});
  tiled_offset_case("resize-host/tiled/error/src-height-zero", {0, 0, 8, 0}, {0, 0, 8, 8});
  tiled_offset_case("resize-host/tiled/error/dst-size-zero", {0, 0, 8, 8}, {0, 0, 0, 0});
  tiled_offset_case("resize-host/tiled/error/dst-height-negative", {0, 0, 8, 8}, {0, 0, 10, -4});
  tiled_offset_case("resize-host/tiled/error/dst-width-negative", {0, 0, 8, 8}, {0, 0, -8, 8});
  {
    Case c("resize-host/tiled/error/null", "nppiGetResizeTiledSourceOffset");
    NppStatus st = nppiGetResizeTiledSourceOffset({0, 0, 8, 8}, {8, 8, 8, 8}, nullptr);
    c.status(st);
    c.finish();
  }
  // The size check happens before the null check (measured).
  {
    Case c("resize-host/tiled/error/order-size-over-null", "nppiGetResizeTiledSourceOffset");
    NppStatus st = nppiGetResizeTiledSourceOffset({0, 0, 0, 8}, {0, 0, 8, 8}, nullptr);
    c.status(st);
    c.finish();
  }
}

TEST_F(NppiResizeHostDiff, AdvancedBufferSizeValues) {
  buffer_size_case("resize-host/advanced-size/8x8-to-16x16", {8, 8}, {16, 16}, NPPI_INTER_LANCZOS3_ADVANCED);
  buffer_size_case("resize-host/advanced-size/16x16-to-8x8", {16, 16}, {8, 8}, NPPI_INTER_LANCZOS3_ADVANCED);
  buffer_size_case("resize-host/advanced-size/same", {8, 8}, {8, 8}, NPPI_INTER_LANCZOS3_ADVANCED);
  buffer_size_case("resize-host/advanced-size/wide", {100, 50}, {200, 100}, NPPI_INTER_LANCZOS3_ADVANCED);
  buffer_size_case("resize-host/advanced-size/down", {8, 8}, {4, 4}, NPPI_INTER_LANCZOS3_ADVANCED);
  buffer_size_case("resize-host/advanced-size/odd", {3, 5}, {7, 3}, NPPI_INTER_LANCZOS3_ADVANCED);
  buffer_size_case("resize-host/advanced-size/src-height-zero", {4, 0}, {4, 4}, NPPI_INTER_LANCZOS3_ADVANCED);
  buffer_size_case("resize-host/advanced-size/dst-width-zero", {4, 4}, {0, 4}, NPPI_INTER_LANCZOS3_ADVANCED);
  buffer_size_case("resize-host/advanced-size/both-zero", {0, 0}, {8, 0}, NPPI_INTER_LANCZOS3_ADVANCED);
  buffer_size_case("resize-host/advanced-size/negative", {-3, 5}, {4, -2}, NPPI_INTER_LANCZOS3_ADVANCED);
  buffer_size_case("resize-host/advanced-size/wraparound", {65536, 65536}, {65536, 65536},
                   NPPI_INTER_LANCZOS3_ADVANCED);
}

TEST_F(NppiResizeHostDiff, AdvancedBufferSizeErrors) {
  buffer_size_case("resize-host/advanced-size/error/lanczos", {8, 8}, {16, 16}, NPPI_INTER_LANCZOS);
  buffer_size_case("resize-host/advanced-size/error/nn", {8, 8}, {16, 16}, NPPI_INTER_NN);
  buffer_size_case("resize-host/advanced-size/error/linear", {8, 8}, {16, 16}, NPPI_INTER_LINEAR);
  buffer_size_case("resize-host/advanced-size/error/unknown", {8, 8}, {16, 16}, 99);
  buffer_size_case("resize-host/advanced-size/error/smooth-edge", {8, 8}, {16, 16},
                   (int)NPPI_SMOOTH_EDGE | NPPI_INTER_LANCZOS3_ADVANCED);
}
