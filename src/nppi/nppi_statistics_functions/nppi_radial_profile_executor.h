// Radial profile executors (package s2).
//
// Pipeline decoded from NVIDIA NPP 13.4 with impulse-scan probes
// (tools/5090_probes_s2/radial_probe.cu, artifacts/5090/s2_probe_out/rad_map.txt):
//   * ring 0 collects every ROI pixel whose centre is less than one pixel away
//     from oCenter (one pixel for an integer centre, four for a half-pixel one);
//   * ring k (1 <= k < nProfileSamples) is the digitised circle of radius k:
//     for every ROI row y with |y - cy| <= k it collects
//     floor(cx - sqrt(k^2 - (y - cy)^2)) and ceil(cx + sqrt(k^2 - (y - cy)^2));
//     pixels that no ring lands on are simply not sampled, so the profile is a
//     sparse set of up to two pixels per row per ring;
//   * nPixels counts the sampled pixels, nMeanIntensity is their mean and
//     nStdDevIntensity the *population* standard deviation; rings without
//     samples are written as {0, 0, 0};
//   * the ring index is not normalised by the ROI size: radii below
//     nProfileSamples - 1 are covered and everything further out is ignored.
// This model reproduces all 3439 sampled pixels of the seven recorded probe
// configurations (fractional/integer/negative centres, square and 40x25 ROIs).
//
// The elliptical variant uses a different rule (dense band membership with a
// cropped per-ring scan window); it was decoded from the NPP 13.4 sm_120 SASS
// and is documented on runEllipticalRadialProfile below.
#ifndef NPPI_RADIAL_PROFILE_EXECUTOR_H
#define NPPI_RADIAL_PROFILE_EXECUTOR_H

#include "npp.h"

namespace nppi {
namespace statistics {

NppStatus runCircularRadialProfile(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, NppiPoint32f oCenter,
                                   NppiProfileData *pProfileData, Npp16u nProfileSamples, cudaStream_t stream);

// Elliptical variant: ring k is the dense band k^2 <= t^2 < (k+1)^2 with
// t^2 = (dx*cos + dy*sin)^2 + (dy*cos - dx*sin)^2 / ratio^2 (the NPP 13.4
// sm_120 kernel passes f0=cos, f1=-sin into that FMA form).  The reference
// crops each ring's scan window to the inner radius (|x-cx| <= k*ratio),
// which drops the central columns above the centre for some rings; the
// executor reproduces that crop.  ratio < 1 is NPP_RANGE_ERROR (NaN and +inf
// are accepted).
NppStatus runEllipticalRadialProfile(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI,
                                     NppiPoint32f oCenter, Npp32f nAstigmatismRatio,
                                     Npp32f nOrientationRadians, NppiProfileData *pProfileData,
                                     Npp16u nProfileSamples, cudaStream_t stream);

}  // namespace statistics
}  // namespace nppi

#endif  // NPPI_RADIAL_PROFILE_EXECUTOR_H
