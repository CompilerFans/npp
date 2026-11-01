// f13: map the C3 seed-pixel read address of the FloodFill families.  The image
// encodes the pixel index in channel 0, so aSeedPixelValue[0] of the returned
// NppiConnectedRegion reveals which pixel NPP actually read.
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
  setvbuf(stdout, nullptr, _IONBF, 0);
  const NppStreamContext C = make_ctx();
  const int W = 8, H = 6;
  int bs = 0;
  nppiFloodFillGetBufferSize(NppiSize{W, H}, &bs);
  printf("buffer %d\n", bs);
  Npp8u *buf = nullptr;
  cudaMalloc(&buf, bs + 4096);

  // 8u C3: channel 0 = y*W+x, channel 1 = 100, channel 2 = 200.
  {
    std::vector<Npp8u> h((size_t)W * H * 3);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        size_t i = ((size_t)y * W + x) * 3;
        h[i] = (Npp8u)(y * W + x);
        h[i + 1] = 100;
        h[i + 2] = 200;
      }
    Npp8u *D = nullptr;
    cudaMalloc(&D, h.size());
    printf("8u C3 seedmap (read index -> seed coordinates):\n");
    for (int sy = 0; sy < H; ++sy) {
      for (int sx = 0; sx < W; ++sx) {
        cudaMemcpy(D, h.data(), h.size(), cudaMemcpyHostToDevice);
        NppiConnectedRegion reg;
        memset(&reg, 0xEE, sizeof reg);
        Npp8u nv[3] = {1, 2, 3};
        NppStatus st = nppiFloodFill_8u_C3IR_Ctx(D, 3 * W, NppiPoint{sx, sy}, nv, nppiNormL1,
                                                 NppiSize{W, H}, &reg, buf, C);
        unsigned r = reg.aSeedPixelValue[0];
        printf("  (%2d,%2d) off=%3d -> idx %2u = (%d,%d)%s", sx, sy, sy * 3 * W + sx, r,
               (int)(r % W), (int)(r / W), st == NPP_NO_ERROR ? "" : " ERR");
        if ((sx + 1) % 4 == 0) printf("\n");
      }
    }
    // 16u C3 on the same grid (index encoded in the first channel).
    std::vector<Npp16u> h16((size_t)W * H * 3);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        size_t i = ((size_t)y * W + x) * 3;
        h16[i] = (Npp16u)(y * W + x);
        h16[i + 1] = 400;
        h16[i + 2] = 500;
      }
    Npp16u *D16 = nullptr;
    cudaMalloc(&D16, h16.size() * 2);
    printf("16u C3 seedmap:\n");
    for (int sy = 0; sy < H; ++sy) {
      for (int sx = 0; sx < W; ++sx) {
        cudaMemcpy(D16, h16.data(), h16.size() * 2, cudaMemcpyHostToDevice);
        NppiConnectedRegion reg;
        memset(&reg, 0xEE, sizeof reg);
        Npp16u nv[3] = {1, 2, 3};
        NppStatus st = nppiFloodFill_16u_C3IR_Ctx(D16, 6 * W, NppiPoint{sx, sy}, nv, nppiNormL1,
                                                  NppiSize{W, H}, &reg, buf, C);
        unsigned r = reg.aSeedPixelValue[0];
        printf("  (%2d,%2d) off=%3d -> idx %2u = (%d,%d)%s", sx, sy, sy * 6 * W + 2 * sx, r,
               (int)(r % W), (int)(r / W), st == NPP_NO_ERROR ? "" : " ERR");
        if ((sx + 1) % 4 == 0) printf("\n");
      }
    }
    cudaFree(D);
    cudaFree(D16);
  }
  cudaFree(buf);
  return 0;
}
