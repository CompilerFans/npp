// f13: bilateral differential (build once against NPP, once against mcMPP).
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char **argv) {
  const int W = 16, H = 12, srcW = 20, srcH = 16, step = 1, radius = argc > 1 ? atoi(argv[1]) : 2;
  const float vs = 10.f, ps = 2.f;
  std::vector<Npp8u> hs((size_t)srcW * srcH), hd((size_t)W * H);
  unsigned s = 7;
  for (auto &v : hs) { s = s * 1103515245u + 12345u; v = (Npp8u)((s >> 16) & 0xFF); }
  s = 99;
  for (auto &v : hd) { s = s * 1103515245u + 12345u; v = (Npp8u)((s >> 16) & 0xFF); }
  Npp8u *ds = nullptr, *dd = nullptr;
  cudaMalloc(&ds, hs.size());
  cudaMalloc(&dd, hd.size());
  cudaMemcpy(ds, hs.data(), hs.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dd, hd.data(), hd.size(), cudaMemcpyHostToDevice);
  NppStreamContext ctx{};
  ctx.hStream = 0;
  NppStatus st = nppiFilterBilateralGaussBorder_8u_C1R_Ctx(
      ds, srcW, {srcW, srcH}, {0, 0}, dd, W, {W, H}, radius, step, vs, ps, NPP_BORDER_REPLICATE, ctx);
  std::vector<Npp8u> out((size_t)W * H);
  cudaMemcpy(out.data(), dd, out.size(), cudaMemcpyDeviceToHost);
  printf("st=%d out:", (int)st);
  for (auto v : out) printf(" %u", (unsigned)v);
  printf("\n");
  return 0;
}
