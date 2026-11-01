// f13: replay the NppiFloodFillDiff picture through the mcMPP implementation
// (links against the built libmpp) and print the resulting grid.
#include <nppi.h>
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static NppStreamContext make_ctx() {
  NppStreamContext c;
  memset(&c, 0, sizeof c);
  c.hStream = 0;
  cudaGetDevice(&c.nCudaDeviceId);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, c.nCudaDeviceId);
  c.nMultiProcessorCount = props.multiProcessorCount;
  c.nMaxThreadsPerMultiProcessor = props.maxThreadsPerMultiProcessor;
  c.nMaxThreadsPerBlock = props.maxThreadsPerBlock;
  c.nSharedMemPerBlock = props.sharedMemPerBlock;
  c.nCudaDevAttrComputeCapabilityMajor = props.major;
  c.nCudaDevAttrComputeCapabilityMinor = props.minor;
  cudaStreamGetFlags(c.hStream, &c.nStreamFlags);
  return c;
}

int main() {
  const int W = 24, H = 18;
  const NppStreamContext C = make_ctx();
  std::vector<Npp8u> h((size_t)W * H);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) h[(size_t)y * W + x] = (Npp8u)(((x / 6 + y / 6) % 3) * 40);
  for (int i = 0; i < 8; ++i) h[(size_t)(2 + i) * W + (2 + i)] = (Npp8u)200;
  for (int x = 0; x < W; ++x) h[(size_t)12 * W + x] = (Npp8u)160;
  for (int y = 15; y < H; ++y)
    for (int x = 18; x < W; ++x) h[(size_t)y * W + x] = (Npp8u)200;
  printf("input:\n");
  for (int y = 0; y < H; ++y) {
    printf("  ");
    for (int x = 0; x < W; ++x) printf("%3d", (int)h[(size_t)y * W + x]);
    printf("\n");
  }
  Npp8u *D = nullptr, *buf = nullptr;
  cudaMalloc(&D, h.size());
  int bs = 0;
  NppStatus bst = nppiFloodFillGetBufferSize(NppiSize{W, H}, &bs);
  printf("GetBufferSize -> %d (st %d)\n", bs, (int)bst);
  cudaMalloc(&buf, bs > 0 ? bs : 4096);
  cudaMemcpy(D, h.data(), h.size(), cudaMemcpyHostToDevice);
  NppiConnectedRegion reg;
  memset(&reg, 0xEE, sizeof reg);
  NppStatus st = nppiFloodFill_8u_C1IR_Ctx(D, W, NppiPoint{3, 3}, 77, nppiNormL1, NppiSize{W, H},
                                           &reg, buf, C);
  cudaMemcpy(h.data(), D, h.size(), cudaMemcpyDeviceToHost);
  printf("out -> %d region box=(%d,%d,%d,%d) count=%u seed=(%u,%u,%u)\n", (int)st,
         reg.oBoundingBox.x, reg.oBoundingBox.y, reg.oBoundingBox.width, reg.oBoundingBox.height,
         reg.nConnectedPixelCount, reg.aSeedPixelValue[0], reg.aSeedPixelValue[1],
         reg.aSeedPixelValue[2]);
  for (int y = 0; y < H; ++y) {
    printf("  ");
    for (int x = 0; x < W; ++x) printf("%3d", (int)h[(size_t)y * W + x]);
    printf("\n");
  }
  cudaFree(D);
  cudaFree(buf);
  return 0;
}
