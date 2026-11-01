// f13: verbatim replay of bil_errors' border_const case.
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>

extern "C" NppStatus nppGetStreamContext(NppStreamContext *p);

static uint64_t st_;
static uint64_t next() {
  uint64_t z = (st_ += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

int main() {
  const int W = 16, H = 10;
  const int sStep = W;
  const int dRow = 6;
  const int dStep = W;
  const size_t se = (size_t)H * sStep;
  const size_t de = (size_t)8 * dStep;
  st_ = 0xB1 + 3;
  std::vector<Npp8u> hs(se), hd(de);
  for (auto &v : hs) v = (Npp8u)next();
  st_ = 0xB1 + 5;
  for (auto &v : hd) v = (Npp8u)next();
  Npp8u *ds = nullptr, *dd = nullptr;
  cudaMalloc(&ds, se);
  cudaMalloc(&dd, de);
  cudaMemcpy(ds, hs.data(), se, cudaMemcpyHostToDevice);
  cudaMemcpy(dd, hd.data(), de, cudaMemcpyHostToDevice);
  NppStreamContext ctx{};
  NppStatus cs = nppGetStreamContext(&ctx);
  printf("ctx status=%d dev=%d stream=%p sm=%u\n", (int)cs, ctx.nCudaDeviceId, (void*)ctx.hStream, ctx.nMultiProcessorCount);
  NppStatus st = nppiFilterBilateralGaussBorder_8u_C1R_Ctx(
      ds, sStep, {W, H}, {0, 0}, dd, dStep, {6, 6}, 1, 1, 10.f, 2.f, NPP_BORDER_CONSTANT, ctx);
  cudaError_t e = cudaDeviceSynchronize();
  printf("border_const st=%d cuda=%s\n", (int)st, cudaGetErrorName(e));
  printf("dRow=%d\n", dRow);
  return 0;
}
