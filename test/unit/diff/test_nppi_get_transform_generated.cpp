// Differential tests for package g5: the host-only geometry transform
// helpers nppiGetAffineTransform / nppiGetAffineQuad / nppiGetAffineBound,
// nppiGetPerspectiveTransform / nppiGetPerspectiveQuad /
// nppiGetPerspectiveBound and nppiGetRotateQuad / nppiGetRotateBound.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiGetTransformDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records ./diff_tests \
//     --gtest_filter='NppiGetTransformDiff.*'
//
// All outputs are host double arrays captured with host_output(F64).  The
// success cases pin down the corner order of the ROI quadrilaterals, the
// affine "first three vertices" rule (including the
// NPP_AFFINE_QUAD_INCORRECT_WARNING threshold) and the rotation direction;
// the error cases pin down the status codes for degenerate quads and for
// empty/negative ROIs.  Null host pointers are never passed: NVIDIA does not
// validate them and the reference run would crash.

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <cmath>
#include <vector>

using namespace npp_diff;
using namespace npp_functional_test;

namespace {

// Tolerances for host double outputs: matrix solves only differ in the last
// few ulps between implementations, while convention/order errors change the
// values grossly.
constexpr double kRtol = 1e-9;
constexpr double kAtol = 1e-12;

const double kAffIdent[2][3] = {{1, 0, 0}, {0, 1, 0}};
const double kPerspIdent[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
const double kQuadUnit[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};

void run_get_affine_transform(const char *key, NppiRect roi, const double quad[4][2]) {
  double coeffs[2][3] = {{0, 0, 0}, {0, 0, 0}};
  clear_cuda_error();
  NppStatus st = nppiGetAffineTransform(roi, quad, coeffs);
  Case c(make_id({"nppiGetAffineTransform", key}), "nppiGetAffineTransform");
  c.input(quad, sizeof(double) * 8);
  c.status(st);
  c.host_output<double>(&coeffs[0][0], 6, ValueKind::F64, kRtol, kAtol);
  c.finish();
}

TEST(NppiGetTransformDiff, GetAffineTransform) {
  // Exactly consistent quads plus literal identity/translation cases.
  const double q0[4][2] = {{0, 0}, {16, 0}, {16, 8}, {0, 8}};
  run_get_affine_transform("identity", NppiRect{0, 0, 16, 8}, q0);
  const double q1[4][2] = {{10, 10}, {24, 10}, {24, 18}, {10, 18}};
  run_get_affine_transform("translate", NppiRect{2, 3, 16, 8}, q1);
  const double q2[4][2] = {{0, 0}, {20, 0}, {20, 12}, {0, 12}};
  run_get_affine_transform("scale", NppiRect{1, 1, 10, 6}, q2);
  // A generic affine (rotation + shear + translation) applied to the ROI
  // corners: quad[i] = A * corner_i.  The corner order shows up directly in
  // the recorded coefficients.
  {
    const double a[2][3] = {{1.2, 0.1, 5.0}, {-0.2, 0.9, -3.0}};
    const NppiRect roi{1, 2, 13, 9};
    const double xs[4] = {(double)roi.x, (double)(roi.x + roi.width), (double)(roi.x + roi.width),
                          (double)roi.x};
    const double ys[4] = {(double)roi.y, (double)roi.y, (double)(roi.y + roi.height),
                          (double)(roi.y + roi.height)};
    double quad[4][2];
    for (int i = 0; i < 4; ++i) {
      quad[i][0] = a[0][0] * xs[i] + a[0][1] * ys[i] + a[0][2];
      quad[i][1] = a[1][0] * xs[i] + a[1][1] * ys[i] + a[1][2];
    }
    run_get_affine_transform("generic", roi, quad);
    // Fourth vertex inconsistent by a wide margin: NVIDIA is expected to
    // return NPP_AFFINE_QUAD_INCORRECT_WARNING and still fill the
    // coefficients from the first three vertices.
    double bad[4][2];
    for (int i = 0; i < 4; ++i) {
      bad[i][0] = quad[i][0];
      bad[i][1] = quad[i][1];
    }
    bad[3][0] += 3.5;
    bad[3][1] -= 2.25;
    run_get_affine_transform("inconsistent4", roi, bad);
    // Fourth vertex off by a tiny epsilon: pins the warning threshold.
    double tiny[4][2];
    for (int i = 0; i < 4; ++i) {
      tiny[i][0] = quad[i][0];
      tiny[i][1] = quad[i][1];
    }
    tiny[3][0] += 1e-9;
    tiny[3][1] -= 1e-9;
    run_get_affine_transform("inconsistent4_eps", roi, tiny);
  }
  // Degenerate quads.
  const double q3[4][2] = {{4, 4}, {4, 4}, {10, 8}, {2, 9}};
  run_get_affine_transform("duplicate01", NppiRect{0, 0, 16, 8}, q3);
  const double q4[4][2] = {{0, 0}, {5, 5}, {10, 10}, {15, 15}};
  run_get_affine_transform("collinear", NppiRect{0, 0, 16, 8}, q4);
  // ROI contract.
  run_get_affine_transform("roi_w0", NppiRect{0, 0, 0, 8}, kQuadUnit);
  run_get_affine_transform("roi_h0", NppiRect{0, 0, 8, 0}, kQuadUnit);
  run_get_affine_transform("roi_neg", NppiRect{1, 1, -4, 8}, kQuadUnit);
  const double q5[4][2] = {{7, 9}, {8, 9}, {8, 10}, {7, 10}};
  run_get_affine_transform("roi_1x1", NppiRect{3, 4, 1, 1}, q5);
}

TEST(NppiGetTransformDiff, GetAffineQuad) {
  auto run = [](const char *key, NppiRect roi, const double a[2][3]) {
    double quad[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
    clear_cuda_error();
    NppStatus st = nppiGetAffineQuad(roi, quad, a);
    Case c(make_id({"nppiGetAffineQuad", key}), "nppiGetAffineQuad");
    c.input(a, sizeof(double) * 6);
    c.status(st);
    c.host_output<double>(&quad[0][0], 8, ValueKind::F64, kRtol, kAtol);
    c.finish();
  };
  run("identity", NppiRect{0, 0, 16, 8}, kAffIdent);
  run("offset", NppiRect{3, 5, 16, 8}, kAffIdent);
  const double rot30[2][3] = {{std::cos(30.0 * M_PI / 180.0), -std::sin(30.0 * M_PI / 180.0), 4.0},
                              {std::sin(30.0 * M_PI / 180.0), std::cos(30.0 * M_PI / 180.0), -2.0}};
  run("rot30", NppiRect{2, 1, 20, 10}, rot30);
  const double shear[2][3] = {{1, 0.5, 1.0}, {0.25, 1, 2.0}};
  run("shear", NppiRect{0, 0, 12, 12}, shear);
  const double mirror[2][3] = {{-1.5, 0, 20.0}, {0, 1.5, -3.0}};
  run("mirror", NppiRect{1, 1, 10, 6}, mirror);
  run("roi_w0", NppiRect{0, 0, 0, 8}, kAffIdent);
  run("roi_neg", NppiRect{0, 0, 8, -3}, kAffIdent);
}

TEST(NppiGetTransformDiff, GetAffineBound) {
  auto run = [](const char *key, NppiRect roi, const double a[2][3]) {
    double bound[2][2] = {{0, 0}, {0, 0}};
    clear_cuda_error();
    NppStatus st = nppiGetAffineBound(roi, bound, a);
    Case c(make_id({"nppiGetAffineBound", key}), "nppiGetAffineBound");
    c.input(a, sizeof(double) * 6);
    c.status(st);
    c.host_output<double>(&bound[0][0], 4, ValueKind::F64, kRtol, kAtol);
    c.finish();
  };
  run("identity", NppiRect{0, 0, 16, 8}, kAffIdent);
  const double offset[2][3] = {{1, 0, 10}, {0, 1, -4}};
  run("offset", NppiRect{3, 5, 16, 8}, offset);
  const double rot30[2][3] = {{std::cos(30.0 * M_PI / 180.0), -std::sin(30.0 * M_PI / 180.0), 0.0},
                              {std::sin(30.0 * M_PI / 180.0), std::cos(30.0 * M_PI / 180.0), 0.0}};
  run("rot30", NppiRect{0, 0, 20, 10}, rot30);
  const double rot200[2][3] = {{std::cos(200.0 * M_PI / 180.0), -std::sin(200.0 * M_PI / 180.0), 1.0},
                               {std::sin(200.0 * M_PI / 180.0), std::cos(200.0 * M_PI / 180.0), 1.0}};
  run("rot200", NppiRect{2, 3, 20, 10}, rot200);
  const double shear[2][3] = {{1, 0.5, 1.0}, {0.25, 1, 2.0}};
  run("shear", NppiRect{0, 0, 12, 12}, shear);
  const double zeros[2][3] = {{0, 0, 0}, {0, 0, 0}};
  run("singular", NppiRect{0, 0, 12, 12}, zeros);
  run("roi_w0", NppiRect{0, 0, 0, 8}, kAffIdent);
}

TEST(NppiGetTransformDiff, GetPerspectiveTransform) {
  auto run = [](const char *key, NppiRect roi, const double quad[4][2]) {
    double coeffs[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    clear_cuda_error();
    NppStatus st = nppiGetPerspectiveTransform(roi, quad, coeffs);
    Case c(make_id({"nppiGetPerspectiveTransform", key}), "nppiGetPerspectiveTransform");
    c.input(quad, sizeof(double) * 8);
    c.status(st);
    c.host_output<double>(&coeffs[0][0], 9, ValueKind::F64, kRtol, kAtol);
    c.finish();
  };
  // Identity-like.
  const double q0[4][2] = {{0, 0}, {16, 0}, {16, 8}, {0, 8}};
  run("identity", NppiRect{0, 0, 16, 8}, q0);
  // Genuinely perspective (non affine) destination quad.
  const double q1[4][2] = {{3, 5}, {25, 2}, {28, 21}, {6, 24}};
  run("perspective", NppiRect{1, 2, 13, 9}, q1);
  const double q2[4][2] = {{0, 0}, {20, 0}, {15, 10}, {5, 10}};
  run("trapezoid", NppiRect{0, 0, 20, 10}, q2);
  // Degenerate quads.
  const double q3[4][2] = {{4, 4}, {4, 4}, {10, 8}, {2, 9}};
  run("duplicate01", NppiRect{0, 0, 16, 8}, q3);
  const double q4[4][2] = {{0, 0}, {5, 5}, {10, 10}, {15, 15}};
  run("collinear", NppiRect{0, 0, 16, 8}, q4);
  // ROI contract.
  run("roi_w0", NppiRect{0, 0, 0, 8}, kQuadUnit);
  run("roi_h0", NppiRect{0, 0, 8, 0}, kQuadUnit);
  run("roi_neg", NppiRect{1, 1, -4, 8}, kQuadUnit);
  const double q5[4][2] = {{7, 9}, {8, 9}, {8, 10}, {7, 10}};
  run("roi_1x1", NppiRect{3, 4, 1, 1}, q5);
}

TEST(NppiGetTransformDiff, GetPerspectiveQuad) {
  auto run = [](const char *key, NppiRect roi, const double a[3][3]) {
    double quad[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
    clear_cuda_error();
    NppStatus st = nppiGetPerspectiveQuad(roi, quad, a);
    Case c(make_id({"nppiGetPerspectiveQuad", key}), "nppiGetPerspectiveQuad");
    c.input(a, sizeof(double) * 9);
    c.status(st);
    c.host_output<double>(&quad[0][0], 8, ValueKind::F64, kRtol, kAtol);
    c.finish();
  };
  run("identity", NppiRect{0, 0, 16, 8}, kPerspIdent);
  const double offset[3][3] = {{1, 0, 4}, {0, 1, -2}, {0, 0, 1}};
  run("offset", NppiRect{3, 5, 16, 8}, offset);
  const double persp[3][3] = {{1.5, 0.2, 3.0}, {-0.1, 0.8, -2.0}, {0.001, 0.0005, 1.0}};
  run("perspective", NppiRect{1, 1, 12, 8}, persp);
  const double persp2[3][3] = {{0.9, -0.05, 2.0}, {0.04, 1.1, 1.0}, {-0.0008, 0.0006, 1.0}};
  run("perspective2", NppiRect{0, 0, 20, 12}, persp2);
  const double singular[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 0}};
  run("singular", NppiRect{0, 0, 12, 12}, singular);
  run("roi_w0", NppiRect{0, 0, 0, 8}, kPerspIdent);
}

TEST(NppiGetTransformDiff, GetPerspectiveBound) {
  auto run = [](const char *key, NppiRect roi, const double a[3][3]) {
    double bound[2][2] = {{0, 0}, {0, 0}};
    clear_cuda_error();
    NppStatus st = nppiGetPerspectiveBound(roi, bound, a);
    Case c(make_id({"nppiGetPerspectiveBound", key}), "nppiGetPerspectiveBound");
    c.input(a, sizeof(double) * 9);
    c.status(st);
    c.host_output<double>(&bound[0][0], 4, ValueKind::F64, kRtol, kAtol);
    c.finish();
  };
  run("identity", NppiRect{0, 0, 16, 8}, kPerspIdent);
  const double offset[3][3] = {{1, 0, 4}, {0, 1, -2}, {0, 0, 1}};
  run("offset", NppiRect{3, 5, 16, 8}, offset);
  const double persp[3][3] = {{1.5, 0.2, 3.0}, {-0.1, 0.8, -2.0}, {0.001, 0.0005, 1.0}};
  run("perspective", NppiRect{1, 1, 12, 8}, persp);
  const double rot30[3][3] = {{std::cos(30.0 * M_PI / 180.0), -std::sin(30.0 * M_PI / 180.0), 0.0},
                              {std::sin(30.0 * M_PI / 180.0), std::cos(30.0 * M_PI / 180.0), 0.0},
                              {0.0, 0.0, 1.0}};
  run("rot30", NppiRect{0, 0, 20, 10}, rot30);
  const double singular[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 0}};
  run("singular", NppiRect{0, 0, 12, 12}, singular);
  run("roi_w0", NppiRect{0, 0, 0, 8}, kPerspIdent);
}

TEST(NppiGetTransformDiff, GetRotateQuad) {
  auto run = [](const char *key, NppiRect roi, double angle, double sx, double sy) {
    double quad[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
    clear_cuda_error();
    NppStatus st = nppiGetRotateQuad(roi, quad, angle, sx, sy);
    Case c(make_id({"nppiGetRotateQuad", key}), "nppiGetRotateQuad");
    c.status(st);
    c.host_output<double>(&quad[0][0], 8, ValueKind::F64, kRtol, kAtol);
    c.finish();
  };
  const NppiRect roi{5, 7, 20, 12};
  run("angle0", roi, 0.0, 0.0, 0.0);
  run("angle30", roi, 30.0, 0.0, 0.0);
  run("angle90", roi, 90.0, 0.0, 0.0);
  run("angle180", roi, 180.0, 0.0, 0.0);
  run("angle_neg45_5", roi, -45.5, 0.0, 0.0);
  run("angle359", roi, 359.0, 0.0, 0.0);
  run("angle30_shift", roi, 30.0, 10.0, -3.5);
  run("origin_angle30", NppiRect{0, 0, 20, 12}, 30.0, 0.0, 0.0);
  run("origin_angle200", NppiRect{0, 0, 20, 12}, 200.0, 5.0, 5.0);
  run("roi_w0", NppiRect{0, 0, 0, 8}, 30.0, 0.0, 0.0);
  run("roi_neg", NppiRect{0, 0, -8, 8}, 30.0, 0.0, 0.0);
}

TEST(NppiGetTransformDiff, GetRotateBound) {
  auto run = [](const char *key, NppiRect roi, double angle, double sx, double sy) {
    double bound[2][2] = {{0, 0}, {0, 0}};
    clear_cuda_error();
    NppStatus st = nppiGetRotateBound(roi, bound, angle, sx, sy);
    Case c(make_id({"nppiGetRotateBound", key}), "nppiGetRotateBound");
    c.status(st);
    c.host_output<double>(&bound[0][0], 4, ValueKind::F64, kRtol, kAtol);
    c.finish();
  };
  const NppiRect roi{5, 7, 20, 12};
  run("angle0", roi, 0.0, 0.0, 0.0);
  run("angle30", roi, 30.0, 0.0, 0.0);
  run("angle90", roi, 90.0, 0.0, 0.0);
  run("angle180", roi, 180.0, 0.0, 0.0);
  run("angle_neg45_5", roi, -45.5, 0.0, 0.0);
  run("angle30_shift", roi, 30.0, 10.0, -3.5);
  run("origin_angle30", NppiRect{0, 0, 20, 12}, 30.0, 0.0, 0.0);
  run("origin_angle200", NppiRect{0, 0, 20, 12}, 200.0, 5.0, 5.0);
  run("roi_w0", NppiRect{0, 0, 0, 8}, 30.0, 0.0, 0.0);
  run("roi_neg", NppiRect{0, 0, -8, 8}, 30.0, 0.0, 0.0);
}

// Round trips: Transform followed by Quad must reproduce the quad; the
// rotate helpers compose the same way.
TEST(NppiGetTransformDiff, RoundTrip) {
  const NppiRect roi{1, 2, 13, 9};
  const double quad[4][2] = {{3.0, 5.0}, {25.0, 2.0}, {28.0, 21.0}, {6.0, 24.0}};
  double coeffs[2][3] = {{0, 0, 0}, {0, 0, 0}};
  clear_cuda_error();
  NppStatus st1 = nppiGetAffineTransform(roi, quad, coeffs);
  Case c1(make_id({"nppiGetTransform", "roundtrip_affine"}), "nppiGetAffineTransform");
  c1.status(st1);
  c1.host_output<double>(&coeffs[0][0], 6, ValueKind::F64, kRtol, kAtol);
  c1.finish();

  double back[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
  clear_cuda_error();
  NppStatus st2 = nppiGetAffineQuad(roi, back, coeffs);
  Case c2(make_id({"nppiGetTransform", "roundtrip_affine_quad"}), "nppiGetAffineQuad");
  c2.status(st2);
  c2.host_output<double>(&back[0][0], 8, ValueKind::F64, kRtol, kAtol);
  c2.finish();

  double pcoeffs[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  clear_cuda_error();
  NppStatus st3 = nppiGetPerspectiveTransform(roi, quad, pcoeffs);
  Case c3(make_id({"nppiGetTransform", "roundtrip_persp"}), "nppiGetPerspectiveTransform");
  c3.status(st3);
  c3.host_output<double>(&pcoeffs[0][0], 9, ValueKind::F64, kRtol, kAtol);
  c3.finish();

  double pback[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
  clear_cuda_error();
  NppStatus st4 = nppiGetPerspectiveQuad(roi, pback, pcoeffs);
  Case c4(make_id({"nppiGetTransform", "roundtrip_persp_quad"}), "nppiGetPerspectiveQuad");
  c4.status(st4);
  c4.host_output<double>(&pback[0][0], 8, ValueKind::F64, kRtol, kAtol);
  c4.finish();

  double rq[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
  clear_cuda_error();
  NppStatus st5 = nppiGetRotateQuad(roi, rq, 33.0, 7.0, -2.0);
  Case c5(make_id({"nppiGetTransform", "roundtrip_rotate_quad"}), "nppiGetRotateQuad");
  c5.status(st5);
  c5.host_output<double>(&rq[0][0], 8, ValueKind::F64, kRtol, kAtol);
  c5.finish();

  double rb[2][2] = {{0, 0}, {0, 0}};
  clear_cuda_error();
  NppStatus st6 = nppiGetRotateBound(roi, rb, 33.0, 7.0, -2.0);
  Case c6(make_id({"nppiGetTransform", "roundtrip_rotate_bound"}), "nppiGetRotateBound");
  c6.status(st6);
  c6.host_output<double>(&rb[0][0], 4, ValueKind::F64, kRtol, kAtol);
  c6.finish();
}

}  // namespace
