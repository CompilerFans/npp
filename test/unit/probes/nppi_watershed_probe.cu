// Probe nppiSegmentWatershed_16u: input interpretation, label output,
// boundary types and connectivity.
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>
#include <nppi.h>
#include <nppi_filtering_functions.h>

extern "C" {
NppStatus nppGetStreamContext(NppStreamContext *c);
}

#define CK(x)                                                                                                      \
  do {                                                                                                             \
    cudaError_t e_ = (x);                                                                                          \
    if (e_ != cudaSuccess) {                                                                                       \
      printf("cuda err %s %d\n", cudaGetErrorString(e_), __LINE__);                                                \
      return;                                                                                                      \
    }                                                                                                              \
  } while (0)

namespace {
constexpr int W = 32, H = 32;

void run(const char *name, const std::vector<unsigned short> &src, NppiNorm norm,
         NppiWatershedSegmentBoundaryType boundary, bool labels) {
  size_t bufferSize = 0;
  NppStatus gs = nppiSegmentWatershedGetBufferSize_16u_C1R({W, H}, &bufferSize);
  printf("=== %s norm=%d boundary=%d getbuf=%d size=%zu\n", name, (int)norm, (int)boundary, (int)gs,
         bufferSize);
  if (gs != NPP_NO_ERROR) return;
  unsigned short *dSrc = nullptr;
  unsigned int *dLabels = nullptr;
  unsigned char *dBuffer = nullptr;
  CK(cudaMalloc(&dSrc, (size_t)W * H * 2));
  CK(cudaMalloc(&dLabels, (size_t)W * H * 4));
  CK(cudaMalloc(&dBuffer, bufferSize > 0 ? bufferSize : 1));
  CK(cudaMemcpy(dSrc, src.data(), (size_t)W * H * 2, cudaMemcpyHostToDevice));
  CK(cudaMemset(dLabels, 0xEE, (size_t)W * H * 4));
  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  NppStatus st = nppiSegmentWatershed_16u_C1IR_Ctx(dSrc, W * 2, labels ? dLabels : nullptr,
                                                   labels ? W * 4 : 0, norm, boundary, {W, H}, dBuffer, ctx);
  CK(cudaDeviceSynchronize());
  printf("status=%d\n", (int)st);
  std::vector<unsigned short> out((size_t)W * H);
  std::vector<unsigned int> lab((size_t)W * H);
  CK(cudaMemcpy(out.data(), dSrc, out.size() * 2, cudaMemcpyDeviceToHost));
  if (labels) CK(cudaMemcpy(lab.data(), dLabels, lab.size() * 4, cudaMemcpyDeviceToHost));
  for (int y = 0; y < H; ++y) {
    if (y % 4 != 0) continue;
    printf("y=%2d:", y);
    for (int x = 0; x < W; ++x) printf(" %5u", out[y * W + x]);
    printf("\n");
  }
  if (labels) {
    printf("labels y=0:");
    for (int x = 0; x < W; ++x) printf(" %u", lab[x]);
    printf("\nlabels y=16:");
    for (int x = 0; x < W; ++x) printf(" %u", lab[16 * W + x]);
    printf("\n");
  }
  cudaFree(dSrc);
  cudaFree(dLabels);
  cudaFree(dBuffer);
}
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  std::vector<unsigned short> vee((size_t)W * H), flat((size_t)W * H, 0), markers((size_t)W * H, 0),
      ramp((size_t)W * H);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const int d1 = (x - 8) * (x - 8) + (y - 16) * (y - 16);
      const int d2 = (x - 24) * (x - 24) + (y - 16) * (y - 16);
      vee[y * W + x] = (unsigned short)(d1 < d2 ? d1 : d2);
      ramp[y * W + x] = (unsigned short)(x * 1000);
    }
  }
  markers[16 * W + 8] = 1;
  markers[16 * W + 24] = 2;
  std::vector<unsigned short> flatmin((size_t)W * H);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      int d1 = (x - 8) * (x - 8) + (y - 16) * (y - 16);
      int d2 = (x - 24) * (x - 24) + (y - 16) * (y - 16);
      if (x >= 7 && x <= 9 && y >= 15 && y <= 17) d1 = 0;
      if (x >= 23 && x <= 25 && y >= 15 && y <= 17) d2 = 0;
      flatmin[y * W + x] = (unsigned short)(d1 < d2 ? d1 : d2);
    }
  }
  run("vee_none", vee, nppiNormL1, NPP_WATERSHED_SEGMENT_BOUNDARIES_NONE, true);
  run("vee_black", vee, nppiNormL1, NPP_WATERSHED_SEGMENT_BOUNDARIES_BLACK, true);
  run("vee_white", vee, nppiNormL1, NPP_WATERSHED_SEGMENT_BOUNDARIES_WHITE, false);
  run("vee_only", vee, nppiNormL1, NPP_WATERSHED_SEGMENT_BOUNDARIES_ONLY, false);
  run("vee_inf", vee, nppiNormInf, NPP_WATERSHED_SEGMENT_BOUNDARIES_BLACK, false);
  run("flat", flat, nppiNormL1, NPP_WATERSHED_SEGMENT_BOUNDARIES_NONE, true);
  run("markers", markers, nppiNormL1, NPP_WATERSHED_SEGMENT_BOUNDARIES_NONE, true);
  run("ramp", ramp, nppiNormL1, NPP_WATERSHED_SEGMENT_BOUNDARIES_NONE, true);
  run("flatmin_none", flatmin, nppiNormL1, NPP_WATERSHED_SEGMENT_BOUNDARIES_NONE, true);
  run("flatmin_black", flatmin, nppiNormL1, NPP_WATERSHED_SEGMENT_BOUNDARIES_BLACK, true);
  run("flatmin_white", flatmin, nppiNormL1, NPP_WATERSHED_SEGMENT_BOUNDARIES_WHITE, true);
  run("flatmin_contrast", flatmin, nppiNormL1, NPP_WATERSHED_SEGMENT_BOUNDARIES_CONTRAST, true);
  run("flatmin_only", flatmin, nppiNormL1, NPP_WATERSHED_SEGMENT_BOUNDARIES_ONLY, true);
  return 0;
}
