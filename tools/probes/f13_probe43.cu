// f13: minimal connectivity cases for label markers (NPP vs host BFS truth).
#include "npp.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>

static void run(const char *name, std::vector<Npp8u> src, int W, int H, NppiNorm norm) {
  Npp8u *ds = nullptr;
  Npp32u *dl = nullptr;
  cudaMalloc(&ds, src.size());
  cudaMalloc(&dl, src.size() * 4);
  cudaMemcpy(ds, src.data(), src.size(), cudaMemcpyHostToDevice);
  cudaMemset(dl, 0, src.size() * 4);
  Npp8u *buf = nullptr;
  cudaMalloc(&buf, 1 << 20);
  NppStreamContext ctx{};
  ctx.hStream = 0;
  NppStatus st = nppiLabelMarkersUF_8u32u_C1R_Ctx(ds, W, dl, W * 4, {W, H}, norm, buf, ctx);
  std::vector<Npp32u> out(src.size());
  cudaMemcpy(out.data(), dl, out.size() * 4, cudaMemcpyDeviceToHost);
  printf("%-18s st=%d out:", name, (int)st);
  for (auto v : out) printf(" %u", v);
  printf("  src:");
  for (auto v : src) printf(" %u", (unsigned)v);
  printf("\n");
  cudaFree(ds);
  cudaFree(dl);
  cudaFree(buf);
}

int main() {
  run("diag-2", {40, 0, 0, 40}, 2, 2, nppiNormInf);
  run("diag-2-L1", {40, 0, 0, 40}, 2, 2, nppiNormL1);
  run("diag-chain", {0, 40, 0, 0, 0, 40, 40, 0, 0}, 3, 3, nppiNormInf);
  run("bl-diag", {40, 0, 40, 40}, 2, 2, nppiNormInf);
  run("br-diag", {40, 40, 0, 40}, 2, 2, nppiNormInf);
  // exact failing 12x9 pattern
  std::vector<Npp8u> s(108);
  uint64_t state = 0x52;
  auto next = [&]() {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  };
  for (auto &v : s) v = (Npp8u)(((next() % 4) % 4) * 40);
  run("seed-0x52-inf", s, 12, 9, nppiNormInf);
  run("seed-0x52-l1", s, 12, 9, nppiNormL1);
  return 0;
}
