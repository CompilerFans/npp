// f13: mcMPP label-markers + compress single-image smoke probe.
#include "npp.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

int main() {
  const int W = 14, H = 10;
  std::vector<Npp8u> src((size_t)W * H);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      src[y * W + x] = (Npp8u)((((x) / 3) * 2 + ((y) / 2) * 3) % 4 * 60);
  Npp8u *ds = nullptr;
  Npp32u *dl = nullptr;
  cudaMalloc(&ds, src.size());
  cudaMalloc(&dl, src.size() * 4);
  cudaMemset(dl, 0xAA, src.size() * 4);
  cudaMemcpy(ds, src.data(), src.size(), cudaMemcpyHostToDevice);
  NppStreamContext ctx{};
  ctx.hStream = 0;
  Npp8u *lbuf = nullptr;
  cudaMalloc(&lbuf, 1 << 20);
  NppStatus st = nppiLabelMarkersUF_8u32u_C1R_Ctx(ds, W, dl, W * 4, {W, H}, nppiNormL1, lbuf, ctx);
  cudaError_t e = cudaDeviceSynchronize();
  std::vector<Npp32u> labels((size_t)W * H);
  cudaMemcpy(labels.data(), dl, labels.size() * 4, cudaMemcpyDeviceToHost);
  printf("label st=%d cuda=%s labels:", (int)st, cudaGetErrorName(e));
  for (auto v : labels) printf(" %u", v);
  printf("\n");
  int newNumber = -1;
  Npp8u *cbuf = nullptr;
  cudaMalloc(&cbuf, 1 << 20);
  NppStatus cs = nppiCompressMarkerLabelsUF_32u_C1IR_Ctx(dl, W * 4, {W, H}, W * H, &newNumber, cbuf, ctx);
  e = cudaDeviceSynchronize();
  cudaMemcpy(labels.data(), dl, labels.size() * 4, cudaMemcpyDeviceToHost);
  printf("compress st=%d cuda=%s newNumber=%d out:", (int)cs, cudaGetErrorName(e), newNumber);
  for (auto v : labels) printf(" %u", v);
  printf("\n");
  return 0;
}
