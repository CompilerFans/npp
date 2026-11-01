// SSIM / MS-SSIM / WMSSSIM executor (package s2).
//
// Pipeline decoded from NVIDIA NPP 13.4 with graph capture + prefix replay
// (tools/5090_probes_s2/wgraph.cu and analyze_wgraph*.py):
//   * 11x11 Gaussian, sigma = 1.5, normalized, clamp-to-edge padding, applied
//     separably in float32 to a, b, a^2, b^2 and a*b;
//   * mu = E[x], var = E[x^2] - mu^2, cov = E[ab] - mu_a*mu_b (biased);
//   * l = (2 mu_a mu_b + C1)/(mu_a^2 + mu_b^2 + C1), C1 = 6.5025;
//     cs = (2 cov + C2)/(var_a + var_b + C2), C2 = 58.5225;
//   * SSIM = mean over all pixels of l*cs;
//   * WMSSSIM = product over 5 levels of the per-level mean of cs^w (levels
//     0..3) and (l*cs)^w (level 4), w = {0.0448, 0.2856, 0.3001, 0.2363,
//     0.1333}; levels are decimated with the 2x2 shift-clamp average
//     (rows/cols {max(0, 2i-1), 2i}, output ceil(n/2)).
// The measured SSIM values match this model to ~1e-7; WMSSSIM to ~1e-4 (the
// remaining gap is NPP float32 accumulation noise).
//
// The C3 entry points evaluate each colour channel independently.  Scratch is
// allocated internally (a conservative requirement is reported by the
// GetBufferHostSize helpers).
#ifndef NPPI_SSIM_EXECUTOR_H
#define NPPI_SSIM_EXECUTOR_H

#include "npp.h"

namespace nppi {
namespace statistics {

NppStatus ssimGetBufferHostSize(NppiSize oSizeROI, int channels, int *hpBufferSize);
NppStatus ssimBatchGetBufferHostSize(NppiSize oSizeROI, int channels, int *hpBufferSize);
NppStatus wmsssimGetBufferHostSize(NppiSize oSizeROI, int channels, int *hpBufferSize);
NppStatus wmsssimBatchGetBufferHostSize(NppiSize oSizeROI, int channels, int *hpBufferSize);
NppStatus msssimGetBufferHostSize(NppiSize oSizeROI, int *hpBufferSize);

NppStatus runSSIM(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, NppiSize oSizeROI,
                  int channels, Npp32f *pSSIM, Npp8u *pDeviceBuffer, cudaStream_t stream);
NppStatus runSSIMBatch(const NppiImageDescriptor *pSrc1, const NppiImageDescriptor *pSrc2, int nBatchSize,
                       NppiSize oSizeROI, int channels, Npp32f *pSSIM, bool advanced, cudaStream_t stream);
NppStatus runWMSSSIM(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, NppiSize oSizeROI,
                     int channels, Npp32f *pWMSSSIM, Npp8u *pDeviceBuffer, cudaStream_t stream);
NppStatus runWMSSSIMBatch(const NppiImageDescriptor *pSrc1, const NppiImageDescriptor *pSrc2, int nBatchSize,
                          NppiSize oSizeROI, int channels, Npp32f *pWMSSSIM, bool advanced, cudaStream_t stream);

}  // namespace statistics
}  // namespace nppi

#endif  // NPPI_SSIM_EXECUTOR_H
