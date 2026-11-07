// Host-only geometry-transform helpers for this module:
//   nppiGetAffineTransform      nppiGetAffineQuad      nppiGetAffineBound
//   nppiGetPerspectiveTransform nppiGetPerspectiveQuad nppiGetPerspectiveBound
//   nppiGetRotateQuad           nppiGetRotateBound
//
// These functions have no _Ctx twins (they never touch the GPU), so they are
// hand-written instead of gen_family.py generated.  The semantics below,
// including the exact arithmetic order, the degenerate-input behaviour and
// the NaN propagation rules, were recovered from the NVIDIA NPP 13.4
// implementations (libnppig.so.13) and validated  against
// reference outputs recorded in the reference implementation (test_nppi_get_transform_generated.cpp).
// Notable NVIDIA behaviours that the documentation does not spell out:
//   * The affine solve uses the first three quad vertices and writes the
//     coefficients BEFORE checking the determinant; a zero (but not NaN)
//     determinant returns NPP_COEFFICIENT_ERROR with the coefficients left
//     in place.  aCoeffs[1][2] is computed from aCoeffs[0][1] (NVIDIA's own
//     formula), and the fourth vertex is checked with a relative
//     DBL_EPSILON tolerance, returning NPP_AFFINE_QUAD_INCORRECT_WARNING.
//   * The perspective solve rejects s < 0, t < 0 and 0 >= s + t (all
//     comparisons NaN-tolerant, so NaN quads flow into NaN coefficients
//     with NPP_NO_ERROR) before writing anything.
//   * Quad/Bound helpers use the (x + w - 1, y + h - 1) corner convention
//     and reject empty ROIs (width or height <= 0) with NPP_SIZE_ERROR
//     without writing any output.
//   * Bounds are double[2][2] = {{xmin, ymin}, {xmax, ymax}} folded from
//     corner 0 with NaN-safe comparisons.

#include "npp.h"

#include <cmath>
#include <cfloat>

namespace {

// (x, y), (x + w - 1, y), (x + w - 1, y + h - 1), (x, y + h - 1)
inline void roiCorners(NppiRect roi, double xs[4], double ys[4]) {
  xs[0] = (double)roi.x;
  ys[0] = (double)roi.y;
  xs[1] = (double)(roi.x + roi.width - 1);
  ys[1] = (double)roi.y;
  xs[2] = (double)(roi.x + roi.width - 1);
  ys[2] = (double)(roi.y + roi.height - 1);
  xs[3] = (double)roi.x;
  ys[3] = (double)(roi.y + roi.height - 1);
}

// bound = {{xmin, ymin}, {xmax, ymax}}; NaN inputs never replace an entry
// (NVIDIA folds with comisd/jbe and comisd/ja, both unordered-keeping).
inline void boundFromQuad(const double quad[4][2], double bound[2][2]) {
  bound[0][0] = quad[0][0];
  bound[0][1] = quad[0][1];
  bound[1][0] = quad[0][0];
  bound[1][1] = quad[0][1];
  for (int i = 1; i < 4; ++i) {
    if (bound[0][0] > quad[i][0]) bound[0][0] = quad[i][0];
    if (bound[0][1] > quad[i][1]) bound[0][1] = quad[i][1];
    if (quad[i][0] > bound[1][0]) bound[1][0] = quad[i][0];
    if (quad[i][1] > bound[1][1]) bound[1][1] = quad[i][1];
  }
}

}  // namespace

NppStatus nppiGetAffineTransform(NppiRect oSrcROI, const double aQuad[4][2], double aCoeffs[2][3]) {
  if (oSrcROI.width <= 1 || oSrcROI.height <= 1) {
    return NPP_RECTANGLE_ERROR;
  }
  const double x = (double)oSrcROI.x;
  const double y = (double)oSrcROI.y;
  const double oneW = 1.0 / (double)oSrcROI.width;
  const double oneH = 1.0 / (double)oSrcROI.height;
  const double t00 = (aQuad[1][0] - aQuad[0][0]) * oneW;
  const double t01 = (aQuad[2][0] - aQuad[1][0]) * oneH;
  const double t10 = (aQuad[1][1] - aQuad[0][1]) * oneW;
  const double t11 = (aQuad[2][1] - aQuad[1][1]) * oneH;
  const double t02 = aQuad[0][0] - (t01 * y + t00 * x);
  const double t12 = aQuad[0][1] - (t11 * y + t01 * x);
  aCoeffs[0][0] = t00;
  aCoeffs[0][1] = t01;
  aCoeffs[0][2] = t02;
  aCoeffs[1][0] = t10;
  aCoeffs[1][1] = t11;
  aCoeffs[1][2] = t12;
  const double det = t00 * t11 - t01 * t10;
  if (det == 0.0) {  // NaN determinant falls through to the vertex check
    return NPP_COEFFICIENT_ERROR;
  }
  const double yh = (double)(oSrcROI.y + oSrcROI.height);
  const double px = t01 * yh + t00 * x + t02;
  const double py = t10 * x + t11 * yh + t12;
  const double apx = fabs(px);
  const double apy = fabs(py);
  const double tolx = (apx > 1.0) ? apx * DBL_EPSILON: DBL_EPSILON;
  const double toly = ((apy > 1.0) ? apy: 1.0) * DBL_EPSILON;
  if (fabs(px - aQuad[3][0]) > tolx || fabs(py - aQuad[3][1]) > toly) {
    return NPP_AFFINE_QUAD_INCORRECT_WARNING;
  }
  return NPP_NO_ERROR;
}

NppStatus nppiGetAffineQuad(NppiRect oSrcROI, double aQuad[4][2], const double aCoeffs[2][3]) {
  if (oSrcROI.width <= 0 || oSrcROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  double xs[4], ys[4];
  roiCorners(oSrcROI, xs, ys);
  for (int i = 0; i < 4; ++i) {
    aQuad[i][0] = aCoeffs[0][0] * xs[i] + aCoeffs[0][1] * ys[i] + aCoeffs[0][2];
    aQuad[i][1] = aCoeffs[1][0] * xs[i] + aCoeffs[1][1] * ys[i] + aCoeffs[1][2];
  }
  return NPP_NO_ERROR;
}

NppStatus nppiGetAffineBound(NppiRect oSrcROI, double aBound[2][2], const double aCoeffs[2][3]) {
  if (oSrcROI.width <= 0 || oSrcROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  double quad[4][2];
  double xs[4], ys[4];
  roiCorners(oSrcROI, xs, ys);
  for (int i = 0; i < 4; ++i) {
    quad[i][0] = aCoeffs[0][0] * xs[i] + aCoeffs[0][1] * ys[i] + aCoeffs[0][2];
    quad[i][1] = aCoeffs[1][0] * xs[i] + aCoeffs[1][1] * ys[i] + aCoeffs[1][2];
  }
  boundFromQuad(quad, aBound);
  return NPP_NO_ERROR;
}

NppStatus nppiGetPerspectiveTransform(NppiRect oSrcROI, const double quad[4][2], double aCoeffs[3][3]) {
  if (oSrcROI.width <= 1 || oSrcROI.height <= 1) {
    return NPP_RECTANGLE_ERROR;
  }
  // Affine basis relative to vertex 0: vertices 1 and 3 span the basis,
  // vertex 2 is the corrected far corner.
  const double a11 = quad[1][0] - quad[0][0];
  const double a12 = quad[3][0] - quad[0][0];
  const double a21 = quad[1][1] - quad[0][1];
  const double a22 = quad[3][1] - quad[0][1];
  const double b1 = quad[2][0] - quad[0][0];
  const double b2 = quad[2][1] - quad[0][1];
  const double det = a22 * a11 - a12 * a21;
  const double s = (a22 * b1 - a12 * b2) / det;
  if (s < 0.0) {
    return NPP_COEFFICIENT_ERROR;
  }
  const double t = (b1 * a21 - b2 * a11) / (a12 * a21 - a22 * a11);
  if (t < 0.0) {
    return NPP_COEFFICIENT_ERROR;
  }
  const double st = s + t;
  if (0.0 >= st) {
    return NPP_COEFFICIENT_ERROR;
  }
  const double oneW = 1.0 / (double)oSrcROI.width;
  const double oneH = 1.0 / (double)oSrcROI.height;
  const double X = (double)oSrcROI.x;
  const double Y = (double)oSrcROI.y;
  const double st1 = st - 1.0;
  const double A = (a11 * s + quad[0][0] * (1.0 - t)) * oneW;
  const double B = (a12 * t + quad[0][0] * (1.0 - s)) * oneH;
  const double C = quad[0][0] * st1 - (A * X + B * Y);
  const double D = (a21 * s + quad[0][1] * (1.0 - t)) * oneW;
  const double E = (a22 * t + quad[0][1] * (1.0 - s)) * oneH;
  const double F = quad[0][1] * st1 - (D * X + E * Y);
  const double G = (1.0 - t) * oneW;
  const double H = (1.0 - s) * oneH;
  const double I = (t + (s - (G * X + H * Y))) - 1.0;
  aCoeffs[0][0] = A / I;
  aCoeffs[0][1] = B / I;
  aCoeffs[0][2] = C / I;
  aCoeffs[1][0] = D / I;
  aCoeffs[1][1] = E / I;
  aCoeffs[1][2] = F / I;
  aCoeffs[2][0] = G / I;
  aCoeffs[2][1] = H / I;
  aCoeffs[2][2] = I / I;
  return NPP_NO_ERROR;
}

NppStatus nppiGetPerspectiveQuad(NppiRect oSrcROI, double quad[4][2], const double aCoeffs[3][3]) {
  if (oSrcROI.width <= 0 || oSrcROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  double xs[4], ys[4];
  roiCorners(oSrcROI, xs, ys);
  for (int i = 0; i < 4; ++i) {
    const double w = aCoeffs[2][0] * xs[i] + aCoeffs[2][1] * ys[i] + aCoeffs[2][2];
    quad[i][0] = (aCoeffs[0][0] * xs[i] + aCoeffs[0][1] * ys[i] + aCoeffs[0][2]) / w;
    quad[i][1] = (aCoeffs[1][0] * xs[i] + aCoeffs[1][1] * ys[i] + aCoeffs[1][2]) / w;
  }
  return NPP_NO_ERROR;
}

NppStatus nppiGetPerspectiveBound(NppiRect oSrcROI, double bound[2][2], const double aCoeffs[3][3]) {
  if (oSrcROI.width <= 0 || oSrcROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  double quad[4][2];
  double xs[4], ys[4];
  roiCorners(oSrcROI, xs, ys);
  for (int i = 0; i < 4; ++i) {
    const double w = aCoeffs[2][0] * xs[i] + aCoeffs[2][1] * ys[i] + aCoeffs[2][2];
    quad[i][0] = (aCoeffs[0][0] * xs[i] + aCoeffs[0][1] * ys[i] + aCoeffs[0][2]) / w;
    quad[i][1] = (aCoeffs[1][0] * xs[i] + aCoeffs[1][1] * ys[i] + aCoeffs[1][2]) / w;
  }
  boundFromQuad(quad, bound);
  return NPP_NO_ERROR;
}

NppStatus nppiGetRotateQuad(NppiRect oSrcROI, double aQuad[4][2], double nAngle, double nShiftX,
                            double nShiftY) {
  if (oSrcROI.width <= 0 || oSrcROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  const double rad = nAngle * 3.141592653589793 / 180.0;
  const double s = sin(rad);
  const double c = cos(rad);
  double xs[4], ys[4];
  roiCorners(oSrcROI, xs, ys);
  for (int i = 0; i < 4; ++i) {
    aQuad[i][0] = xs[i] * c + ys[i] * s + nShiftX;
    aQuad[i][1] = ys[i] * c - xs[i] * s + nShiftY;
  }
  return NPP_NO_ERROR;
}

NppStatus nppiGetRotateBound(NppiRect oSrcROI, double aBoundingBox[2][2], double nAngle, double nShiftX,
                             double nShiftY) {
  double quad[4][2];
  NppStatus status = nppiGetRotateQuad(oSrcROI, quad, nAngle, nShiftX, nShiftY);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  boundFromQuad(quad, aBoundingBox);
  return NPP_NO_ERROR;
}
