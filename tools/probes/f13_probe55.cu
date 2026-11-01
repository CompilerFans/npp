// f13: bilateral replay of NppiBilateralGaussDiff.Values (8u_C1R, seed 0xB1).
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>

static uint64_t st_;
static uint64_t next() {
  uint64_t z = (st_ += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
static uint32_t uni32() { return (uint32_t)next(); }

int main(int argc, char **argv) {
  int radius = argc > 1 ? atoi(argv[1]) : 2, step = argc > 2 ? atoi(argv[2]) : 1;
  uint64_t seed = argc > 3 ? strtoull(argv[3], nullptr, 0) : 0xB1ull;
  const int srcW = 20, srcH = 16, W = 16, H = 12;
  const int pad = argc > 4 ? atoi(argv[4]) : 0;
  const int dStep = W + pad;
  const size_t de = (size_t)H * dStep;
  const int sStep = srcW + pad;
  st_ = seed + 11;
  std::vector<Npp8u> hs((size_t)srcH * sStep);
  for (auto &v : hs) v = (Npp8u)uni32();
  st_ = seed + 29;
  std::vector<Npp8u> hd(de);
  for (auto &v : hd) v = (Npp8u)uni32();
  Npp8u *ds = nullptr, *dd = nullptr;
  cudaMalloc(&ds, hs.size());
  cudaMalloc(&dd, hd.size());
  cudaMemcpy(ds, hs.data(), hs.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dd, hd.data(), hd.size(), cudaMemcpyHostToDevice);
  NppStreamContext ctx{};
  ctx.hStream = 0;
  NppStatus st = nppiFilterBilateralGaussBorder_8u_C1R_Ctx(ds, sStep, {srcW, srcH}, {0, 0}, dd, dStep,
                                                            {W, H}, radius, step, 10.f, 2.f,
                                                            NPP_BORDER_REPLICATE, ctx);
  std::vector<Npp8u> out(de);
  cudaMemcpy(out.data(), dd, out.size(), cudaMemcpyDeviceToHost);
  printf("st=%d out:", (int)st);
  for (auto v : out) printf(" %u", (unsigned)v);
  printf("\n");
  return 0;
}
