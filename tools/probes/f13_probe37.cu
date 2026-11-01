// f13: replay the compress-batch diff case against the mcMPP library and dump
// the values so they can be compared with the NPP golden (probe-side debug).
#include "npp.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

int main(int argc, char **argv) {
  const int W = 14, H = 10, nImg = 2;
  std::vector<Npp32u> host((size_t)W * H * nImg);
  const Npp32u palette[5] = {0, 5, 9, 20, 33};
  for (int img = 0; img < nImg; ++img)
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        host[(size_t)img * W * H + y * W + x] = palette[((x / 3) + (y / 2) + img) % 5];
  for (int img = 1; img < nImg; ++img)
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) host[(size_t)img * W * H + y * W + x] += (Npp32u)(img * 7);

  Npp32u *d = nullptr;
  cudaMalloc(&d, host.size() * 4);
  cudaMemcpy(d, host.data(), host.size() * 4, cudaMemcpyHostToDevice);
  const int bufBytes = 65556 + 8 * W * H;
  Npp8u *b0 = nullptr, *b1 = nullptr;
  cudaMalloc(&b0, bufBytes);
  cudaMalloc(&b1, bufBytes);
  NppiImageDescriptor him[2] = {{(char *)d, W * 4, {W, H}}, {(char *)d + (size_t)W * H * 4, W * 4, {W, H}}};
  NppiBufferDescriptor hbu[2] = {{(char *)b0, bufBytes}, {(char *)b1, bufBytes}};
  Npp32u hnm[2] = {0xAAAA5555u, 0xAAAA5555u};
  NppStreamContext ctx{};
  ctx.hStream = 0;
  int advanced = argc > 1 ? atoi(argv[1]) : 0;
  NppStatus st = advanced ? nppiCompressMarkerLabelsUFBatch_32u_C1IR_Advanced_Ctx(him, hbu, hnm, nImg, {W, H},
                                                                                   W * H, ctx)
                          : nppiCompressMarkerLabelsUFBatch_32u_C1IR_Ctx(him, hbu, hnm, nImg, {W, H}, W * H, ctx);
  cudaError_t e = cudaDeviceSynchronize();
  std::vector<Npp32u> out(host.size());
  cudaMemcpy(out.data(), d, out.size() * 4, cudaMemcpyDeviceToHost);
  printf("st=%d cuda=%s nm=%u,%u\n", (int)st, cudaGetErrorName(e), hnm[0], hnm[1]);
  printf("labels:");
  for (size_t i = 0; i < out.size(); ++i) printf(" %u", out[i]);
  printf("\n");
  return 0;
}
