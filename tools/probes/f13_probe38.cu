// f13: nppiCompressMarkerLabelsUF_32u_C1IR renumbering model.
#include "npp.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

static void one(const char *name, std::vector<Npp32u> in) {
  int n = (int)in.size();
  Npp32u *d = nullptr;
  cudaMalloc(&d, n * 4);
  cudaMemcpy(d, in.data(), n * 4, cudaMemcpyHostToDevice);
  int bufSize = 0;
  nppiCompressMarkerLabelsGetBufferSize_32u_C1R(n, &bufSize);
  Npp8u *buf = nullptr;
  cudaMalloc(&buf, bufSize);
  int newNumber = -1;
  NppStreamContext ctx{};
  ctx.hStream = 0;
  NppStatus st = nppiCompressMarkerLabelsUF_32u_C1IR_Ctx(d, n * 4, {n, 1}, n, &newNumber, buf, ctx);
  cudaError_t e = cudaDeviceSynchronize();
  std::vector<Npp32u> out(n);
  cudaMemcpy(out.data(), d, n * 4, cudaMemcpyDeviceToHost);
  printf("%-14s st=%d cuda=%s newNumber=%d in:", name, (int)st, cudaGetErrorName(e), newNumber);
  for (int v : in) printf(" %d", (int)v);
  printf(" -> out:");
  for (int v : out) printf(" %d", (int)v);
  printf("\n");
}

int main() {
  one("0,1,2,3", {0, 1, 2, 3});
  one("1,2,3,4", {1, 2, 3, 4});
  one("0,5,9,20", {0, 5, 9, 20});
  one("2,7,9,11", {2, 7, 9, 11});
  one("0,0,0,0", {0, 0, 0, 0});
  one("5,5,5,5", {5, 5, 5, 5});
  one("0,2,2,7", {0, 2, 2, 7});
  one("3,0,3,0", {3, 0, 3, 0});
  return 0;
}
