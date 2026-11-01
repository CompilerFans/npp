// f13: compress semantics on real UF label output.
#include "npp.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

int main() {
  const int W = 12, H = 9;
  std::vector<Npp8u> src((size_t)W * H);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) src[y * W + x] = (Npp8u)(((x / 3) + (y / 2)) % 3 * 40);
  Npp8u *ds = nullptr;
  Npp32u *dl = nullptr;
  cudaMalloc(&ds, src.size());
  cudaMalloc(&dl, src.size() * 4);
  cudaMemcpy(ds, src.data(), src.size(), cudaMemcpyHostToDevice);
  NppStreamContext ctx{};
  ctx.hStream = 0;
  int lmbuf = 0;
  nppiLabelMarkersUFGetBufferSize_32u_C1R({W, H}, &lmbuf);
  Npp8u *lbuf = nullptr;
  cudaMalloc(&lbuf, lmbuf);
  NppStatus st = nppiLabelMarkersUF_8u32u_C1R_Ctx(ds, W, dl, W * 4, {W, H}, nppiNormL1, lbuf, ctx);
  std::vector<Npp32u> labels((size_t)W * H);
  cudaMemcpy(labels.data(), dl, labels.size() * 4, cudaMemcpyDeviceToHost);
  printf("labelmarkers st=%d labels:", (int)st);
  for (auto v : labels) printf(" %u", v);
  printf("\n");

  int bufSize = 0;
  nppiCompressMarkerLabelsGetBufferSize_32u_C1R(W * H, &bufSize);
  Npp8u *cbuf = nullptr;
  cudaMalloc(&cbuf, bufSize);
  int newNumber = -1;
  NppStatus cs = nppiCompressMarkerLabelsUF_32u_C1IR_Ctx(dl, W * 4, {W, H}, W * H, &newNumber, cbuf, ctx);
  std::vector<Npp32u> out(labels.size());
  cudaMemcpy(out.data(), dl, out.size() * 4, cudaMemcpyDeviceToHost);
  printf("compress st=%d newNumber=%d out:", (int)cs, newNumber);
  for (auto v : out) printf(" %u", v);
  printf("\n");
  return 0;
}
