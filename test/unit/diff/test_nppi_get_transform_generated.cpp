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

void run_perspective_transform(const char *key, NppiRect roi, const double quad[4][2]) {
  double coeffs[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  clear_cuda_error();
  NppStatus st = nppiGetPerspectiveTransform(roi, quad, coeffs);
  Case c(make_id({"nppiGetPerspectiveTransform", key}), "nppiGetPerspectiveTransform");
  c.input(quad, sizeof(double) * 8);
  c.status(st);
  c.host_output<double>(&coeffs[0][0], 9, ValueKind::F64, kRtol, kAtol);
  c.finish();
}

// quad[i] = a * corner_i for the full-extent ROI corners
// (x,y), (x+w,y), (x+w,y+h), (x,y+h).
void affine_quad_from_coeffs(NppiRect roi, const double a[2][3], double quad[4][2]) {
  const double xs[4] = {(double)roi.x, (double)(roi.x + roi.width), (double)(roi.x + roi.width),
                        (double)roi.x};
  const double ys[4] = {(double)roi.y, (double)roi.y, (double)(roi.y + roi.height),
                        (double)(roi.y + roi.height)};
  for (int i = 0; i < 4; ++i) {
    quad[i][0] = a[0][0] * xs[i] + a[0][1] * ys[i] + a[0][2];
    quad[i][1] = a[1][0] * xs[i] + a[1][1] * ys[i] + a[1][2];
  }
}

// The recorded goldens show that NVIDIA's affine solve is exact for
// axis-aligned quads but shifts aCoeffs[1][2] for sheared ones.  These probes
// vary the ROI position / size (and the two off-diagonal coefficients
// separately) so the exact formula can be read out of the recorded bytes.
TEST(NppiGetTransformDiff, GetAffineTransformShear) {
  const double a[2][3] = {{1.2, 0.1, 5.0}, {-0.2, 0.9, -3.0}};
  const NppiRect rois[] = {NppiRect{0, 2, 13, 9},  NppiRect{5, 2, 13, 9},  NppiRect{1, 0, 13, 9},
                           NppiRect{1, 7, 13, 9},  NppiRect{1, 2, 26, 9},  NppiRect{1, 2, 13, 18},
                           NppiRect{3, 4, 7, 5},   NppiRect{0, 0, 13, 9},  NppiRect{4, 6, 20, 11}};
  const char *keys[] = {"shear_x0", "shear_x5", "shear_y0", "shear_y7",
                        "shear_w26", "shear_h18", "shear_small", "shear_origin", "shear_generic2"};
  for (size_t k = 0; k < sizeof(rois) / sizeof(rois[0]); ++k) {
    double quad[4][2];
    affine_quad_from_coeffs(rois[k], a, quad);
    run_get_affine_transform(keys[k], rois[k], quad);
  }
  // Off-diagonal terms separated: only a01 != 0, then only a10 != 0.
  {
    const double ax[2][3] = {{1, 0.4, 2}, {0, 1, 1}};
    double quad[4][2];
    affine_quad_from_coeffs(NppiRect{2, 3, 15, 10}, ax, quad);
    run_get_affine_transform("shear_only01", NppiRect{2, 3, 15, 10}, quad);
  }
  {
    const double ay[2][3] = {{1, 0, 2}, {0.3, 1, 1}};
    double quad[4][2];
    affine_quad_from_coeffs(NppiRect{2, 3, 15, 10}, ay, quad);
    run_get_affine_transform("shear_only10", NppiRect{2, 3, 15, 10}, quad);
  }
  // 4th vertex consistent with the shifted affine (a12 - 3.3 instead of -3.0)
  // but vertices 0..2 consistent with the exact one: if NVIDIA checks the 4th
  // vertex against its own (shifted) result this must return NPP_NO_ERROR.
  {
    const NppiRect roi{1, 2, 13, 9};
    double quad[4][2];
    affine_quad_from_coeffs(roi, a, quad);
    quad[3][1] += -0.3;
    run_get_affine_transform("shifted_consistent4", roi, quad);
  }
  // ROI width/height of exactly 1 (degenerate under the w-1/h-1 corner
  // convention used by the Quad/Bound helpers).
  {
    double quad[4][2];
    affine_quad_from_coeffs(NppiRect{3, 4, 2, 2}, a, quad);
    run_get_affine_transform("roi_2x2", NppiRect{3, 4, 2, 2}, quad);
    affine_quad_from_coeffs(NppiRect{3, 4, 1, 8}, a, quad);
    run_get_affine_transform("roi_w1", NppiRect{3, 4, 1, 8}, quad);
    affine_quad_from_coeffs(NppiRect{3, 4, 8, 1}, a, quad);
    run_get_affine_transform("roi_h1", NppiRect{3, 4, 8, 1}, quad);
  }
}

// Pins the tolerance of the 4th-vertex consistency check: an exactly
// consistent translation quad with the 4th vertex perturbed by eps.
TEST(NppiGetTransformDiff, GetAffineTransformWarnThreshold) {
  const NppiRect roi{2, 3, 16, 8};
  const double t[2][3] = {{1, 0, 8}, {0, 1, 7}};
  double base[4][2];
  affine_quad_from_coeffs(roi, t, base);
  const double eps[] = {1e-3, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-12, 1e-15};
  const char *keys[] = {"eps1e-3", "eps1e-6", "eps1e-7", "eps1e-8",
                        "eps1e-9", "eps1e-10", "eps1e-12", "eps1e-15"};
  for (size_t k = 0; k < sizeof(eps) / sizeof(eps[0]); ++k) {
    double quad[4][2];
    for (int i = 0; i < 4; ++i) {
      quad[i][0] = base[i][0];
      quad[i][1] = base[i][1];
    }
    quad[3][0] += eps[k];
    quad[3][1] -= eps[k];
    run_get_affine_transform(keys[k], roi, quad);
  }
  {
    double quad[4][2];
    for (int i = 0; i < 4; ++i) {
      quad[i][0] = base[i][0];
      quad[i][1] = base[i][1];
    }
    quad[3][0] += 1e-9;
    run_get_affine_transform("eps_x_only", roi, quad);
  }
  {
    double quad[4][2];
    for (int i = 0; i < 4; ++i) {
      quad[i][0] = base[i][0];
      quad[i][1] = base[i][1];
    }
    quad[3][1] += 1e-9;
    run_get_affine_transform("eps_y_only", roi, quad);
  }
  {
    double quad[4][2];
    for (int i = 0; i < 4; ++i) {
      quad[i][0] = base[i][0];
      quad[i][1] = base[i][1];
    }
    quad[3][0] -= 1e-9;
    quad[3][1] += 1e-9;
    run_get_affine_transform("eps_neg", roi, quad);
  }
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
  // Genuinely perspective quad at a non-origin ROI: quad[i] = H * corner_i
  // with a non-degenerate projective row, so only an exact four-pair
  // homography solve reproduces it.
  {
    const double h[3][3] = {{1.1, 0.15, 2.0}, {-0.08, 0.95, 1.0}, {0.002, 0.001, 1.0}};
    const NppiRect roi{2, 3, 10, 7};
    const double xs[4] = {(double)roi.x, (double)(roi.x + roi.width), (double)(roi.x + roi.width),
                          (double)roi.x};
    const double ys[4] = {(double)roi.y, (double)roi.y, (double)(roi.y + roi.height),
                          (double)(roi.y + roi.height)};
    double quad[4][2];
    for (int i = 0; i < 4; ++i) {
      const double w = h[2][0] * xs[i] + h[2][1] * ys[i] + h[2][2];
      quad[i][0] = (h[0][0] * xs[i] + h[0][1] * ys[i] + h[0][2]) / w;
      quad[i][1] = (h[1][0] * xs[i] + h[1][1] * ys[i] + h[1][2]) / w;
    }
    run("persp_nonorigin", roi, quad);
  }
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

// Degenerate / threshold probes that pin down the remaining unknowns:
//  - GetAffineTransform: is the singular-matrix check an exact determinant
//    comparison or an epsilon test?  Does the ROI contract win over the
//    degenerate-quad path?  How do NaN quad entries propagate?
//  - GetPerspectiveTransform: how close to degenerate may a quad be before
//    NVIDIA's solver returns all-NaN coefficients?
//  - GetPerspectiveQuad: are the coefficients treated as scale invariant
//    (i.e. is the projection row really divided through)?
TEST(NppiGetTransformDiff, GetTransformDegenerateProbes) {
  const NppiRect roi{1, 2, 13, 9};
  const double kCollinear[4][2] = {{0, 0}, {5, 5}, {10, 10}, {3, 8}};

  // Determinant of the solved 2x2 linear part is exactly d: exact-zero
  // comparison returns the warning path, an epsilon test returns
  // NPP_COEFFICIENT_ERROR.
  const double dets[] = {1e-9, 1e-13, 1e-15};
  const char *det_keys[] = {"affdet_1e-9", "affdet_1e-13", "affdet_1e-15"};
  for (size_t k = 0; k < 3; ++k) {
    const double a[2][3] = {{1, 0.4, 2}, {0, dets[k], 1}};
    double quad[4][2];
    affine_quad_from_coeffs(roi, a, quad);
    run_get_affine_transform(det_keys[k], roi, quad);
  }
  // ROI contract (w < 2) with a degenerate quad: which status wins?
  run_get_affine_transform("aff_roi1x1_collinear", NppiRect{3, 4, 1, 1}, kCollinear);
  // NaN in the quad entries.
  {
    const double a[2][3] = {{1, 0.4, 2}, {0, 1, 1}};
    double quad[4][2];
    affine_quad_from_coeffs(roi, a, quad);
    quad[2][0] = std::nan("");
    run_get_affine_transform("aff_nan_quad", roi, quad);
  }

  // Perspective: exactly / nearly degenerate quads.  The base quad is the
  // affine image of the ROI corners; the variants break one pair.
  const double ap[2][3] = {{2, 0.3, 4}, {0.2, 1.5, -1}};
  double base[4][2];
  affine_quad_from_coeffs(roi, ap, base);
  {
    double q[4][2];
    for (int i = 0; i < 4; ++i) { q[i][0] = base[i][0]; q[i][1] = base[i][1]; }
    q[2][0] = base[3][0]; q[2][1] = base[3][1];
    run_perspective_transform("persp_dup23", roi, q);
  }
  const double nears[] = {1e-3, 1e-6, 1e-9, 1e-12};
  const char *near_keys[] = {"persp_near01_1e-3", "persp_near01_1e-6",
                             "persp_near01_1e-9", "persp_near01_1e-12"};
  for (size_t k = 0; k < 4; ++k) {
    double q[4][2];
    for (int i = 0; i < 4; ++i) { q[i][0] = base[i][0]; q[i][1] = base[i][1]; }
    q[1][0] = q[0][0] + nears[k];
    q[1][1] = q[0][1];
    run_perspective_transform(near_keys[k], roi, q);
  }
  run_perspective_transform("persp_collinear3", roi, kCollinear);
  const double col_eps[] = {1e-6, 1e-9, 1e-12};
  const char *col_keys[] = {"persp_nearcol3_1e-6", "persp_nearcol3_1e-9", "persp_nearcol3_1e-12"};
  for (size_t k = 0; k < 3; ++k) {
    double q[4][2] = {{0, 0}, {5, 5}, {10 + col_eps[k], 10}, {3, 8}};
    run_perspective_transform(col_keys[k], roi, q);
  }
  run_perspective_transform("persp_roi1x1_collinear", NppiRect{3, 4, 1, 1}, kCollinear);

  // GetPerspectiveQuad with an unnormalised (scaled) coefficient matrix.
  {
    const double h[3][3] = {{2, 0, 0}, {0, 2, 0}, {0, 0, 2}};
    double quad[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
    clear_cuda_error();
    NppStatus st = nppiGetPerspectiveQuad(NppiRect{2, 3, 10, 7}, quad, h);
    Case c(make_id({"nppiGetPerspectiveQuad", "unnormalized"}), "nppiGetPerspectiveQuad");
    c.input(h, sizeof(double) * 9);
    c.status(st);
    c.host_output<double>(&quad[0][0], 8, ValueKind::F64, kRtol, kAtol);
    c.finish();
  }
  // Rotation angle wrapping.
  {
    double quad[4][2];
    clear_cuda_error();
    NppStatus st = nppiGetRotateQuad(NppiRect{5, 7, 20, 12}, quad, 720.0, 1.0, 2.0);
    Case c(make_id({"nppiGetRotateQuad", "angle720"}), "nppiGetRotateQuad");
    c.status(st);
    c.host_output<double>(&quad[0][0], 8, ValueKind::F64, kRtol, kAtol);
    c.finish();
  }
}

}  // namespace
