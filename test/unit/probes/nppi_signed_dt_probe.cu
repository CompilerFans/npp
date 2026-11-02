// Probe nppiSignedDistanceTransformPBA_32f: signed distance convention,
// coverage handling, cutoff and sub-pixel shifts.
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
constexpr int W = 64, H = 64;
constexpr int STEP16 = W * 2 * 2;

void run(const char *name, const std::vector<float> &src, float cutoff, float sx, float sy) {
  size_t bufferSize = 0;
  NppStatus gs = nppiSignedDistanceTransformPBAGetBufferSize({W, H}, &bufferSize);
  printf("=== %s cutoff=%g shift=(%g,%g) getbuf=%d buffer=%zu\n", name, cutoff, sx, sy, (int)gs, bufferSize);
  if (gs != NPP_NO_ERROR) return;
  float *dSrc = nullptr;
  Npp8u *dBuffer = nullptr;
  Npp16s *dVor = nullptr, *dInd = nullptr, *dRel = nullptr;
  Npp32f *dTr = nullptr;
  CK(cudaMalloc(&dSrc, (size_t)W * H * 4));
  CK(cudaMalloc(&dBuffer, bufferSize > 0 ? bufferSize : 1));
  CK(cudaMalloc(&dVor, (size_t)STEP16 * H));
  CK(cudaMalloc(&dInd, (size_t)STEP16 * H));
  CK(cudaMalloc(&dRel, (size_t)STEP16 * H));
  CK(cudaMalloc(&dTr, (size_t)W * 4 * H));
  CK(cudaMemcpy(dSrc, src.data(), (size_t)W * H * 4, cudaMemcpyHostToDevice));
  CK(cudaMemset(dVor, 0xEE, (size_t)STEP16 * H));
  CK(cudaMemset(dTr, 0xEE, (size_t)W * 4 * H));
  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  NppStatus st = nppiSignedDistanceTransformPBA_32f_C1R_Ctx(dSrc, W * 4, cutoff, sx, sy, dVor, STEP16, dInd,
                                                             STEP16, dRel, STEP16, dTr, W * 4, {W, H}, dBuffer,
                                                             ctx);
  CK(cudaDeviceSynchronize());
  printf("status=%d\n", (int)st);
  std::vector<Npp32f> tr((size_t)W * H);
  std::vector<Npp16s> vor((size_t)STEP16 / 2 * H), ind((size_t)STEP16 / 2 * H), rel((size_t)STEP16 / 2 * H);
  CK(cudaMemcpy(tr.data(), dTr, tr.size() * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(vor.data(), dVor, vor.size() * 2, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(ind.data(), dInd, ind.size() * 2, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(rel.data(), dRel, rel.size() * 2, cudaMemcpyDeviceToHost));
  const int pts[][2] = {{20, 32}, {19, 32}, {21, 32}, {32, 20}, {32, 19}, {32, 21},
                        {32, 32}, {10, 10}, {0, 0},   {43, 43}, {44, 44}, {19, 19}};
  for (const auto &p : pts) {
    const int i = p[1] * (STEP16 / 2) + p[0] * 2;
    printf("  (%2d,%2d) tr=%.9g vor=(%d,%d) ind=(%d,%d) rel=(%d,%d)\n", p[0], p[1], tr[p[1] * W + p[0]], vor[i],
           vor[i + 1], ind[i], ind[i + 1], rel[i], rel[i + 1]);
  }
  cudaFree(dSrc);
  cudaFree(dBuffer);
  cudaFree(dVor);
  cudaFree(dInd);
  cudaFree(dRel);
  cudaFree(dTr);
}

}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  std::vector<float> binary((size_t)W * H, 0.0f);
  for (int y = 20; y <= 43; ++y)
    for (int x = 20; x <= 43; ++x) binary[y * W + x] = 1.0f;
  std::vector<float> half = binary;
  for (int y = 20; y <= 43; ++y) {
    half[y * W + 19] = 0.5f;
    half[y * W + 44] = 0.5f;
  }
  for (int x = 20; x <= 43; ++x) {
    half[19 * W + x] = 0.5f;
    half[44 * W + x] = 0.5f;
  }
  run("binary", binary, 0.0f, 0.0f, 0.0f);
  run("binary_cutoff05", binary, 0.5f, 0.0f, 0.0f);
  run("half_edges", half, 0.0f, 0.0f, 0.0f);
  run("binary_shift", binary, 0.0f, 0.25f, -0.25f);
  return 0;
}
