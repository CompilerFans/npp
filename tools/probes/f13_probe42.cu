// f13: replay the NppiLabelMarkersDiff.Batch 8u32u/Inf case (Rng seed 0x41+1).
#include "npp.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

static uint64_t state;
static uint64_t next() {
  uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
static int integer(int lo, int hi) {
  uint64_t span = (uint64_t)hi - (uint64_t)lo;
  return (int)((uint64_t)lo + next() % (span + 1));
}

int main(int argc, char **argv) {
  const int W = 12, H = 9, nImg = 2;
  uint64_t seed = argc > 1 ? strtoull(argv[1], nullptr, 0) : 0x42;
  state = seed;
  std::vector<Npp8u> hs((size_t)W * H * nImg);
  for (auto &v : hs) v = (Npp8u)((integer(0, 3) % 4) * 40);
  Npp8u *ds = nullptr;
  Npp32u *db = nullptr;
  cudaMalloc(&ds, hs.size());
  cudaMalloc(&db, hs.size() * 4);
  cudaMemcpy(ds, hs.data(), hs.size(), cudaMemcpyHostToDevice);
  cudaMemset(db, 0, hs.size() * 4);
  Npp8u *lbuf = nullptr;
  cudaMalloc(&lbuf, 1 << 20);
  NppStreamContext ctx{};
  ctx.hStream = 0;
  NppiImageDescriptor simg[2] = {{(char *)ds, W, {W, H}},
                                 {(char *)ds + (size_t)W * H, W, {W, H}}};
  NppiImageDescriptor dimg[2] = {{(char *)db, W * 4, {W, H}},
                                 {(char *)db + (size_t)W * H * 4, W * 4, {W, H}}};
  NppStatus st = nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx(simg, dimg, 2, {W, H}, nppiNormInf, ctx);
  cudaError_t e = cudaDeviceSynchronize();
  std::vector<Npp32u> out(hs.size());
  cudaMemcpy(out.data(), db, out.size() * 4, cudaMemcpyDeviceToHost);
  // single-image reference for image 0 and image 1
  std::vector<Npp32u> single(hs.size(), 0);
  for (int i = 0; i < nImg; ++i) {
    Npp32u *d1 = nullptr;
    cudaMalloc(&d1, (size_t)W * H * 4);
    cudaMemset(d1, 0, (size_t)W * H * 4);
    NppStatus s1 = nppiLabelMarkersUF_8u32u_C1R_Ctx(ds + (size_t)i * W * H, W, d1, W * 4, {W, H},
                                                    nppiNormInf, lbuf, ctx);
    if (s1 != 0) printf("single %d st=%d\n", i, (int)s1);
    cudaMemcpy(single.data() + (size_t)i * W * H, d1, (size_t)W * H * 4, cudaMemcpyDeviceToHost);
    cudaFree(d1);
  }
  printf("single:");
  for (size_t i = 0; i < single.size(); ++i) printf(" %u", single[i]);
  printf("\n");
  printf("st=%d cuda=%s\nsrc:", (int)st, cudaGetErrorName(e));
  for (auto v : hs) printf(" %d", (int)v);
  printf("\nout:");
  for (size_t i = 0; i < out.size(); ++i) printf(" %u", out[i]);
  printf("\n");
  return 0;
}
