// f13: fit the C3 seed-read address of the FloodFill families (8u/16u, W=8/12).
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

template <typename T>
void seedmap(const char *tag, int W, int H, const NppStreamContext &C, Npp8u *buf) {
  std::vector<T> h((size_t)W * H * 3);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      size_t i = ((size_t)y * W + x) * 3;
      h[i] = (T)(y * W + x);
      h[i + 1] = (T)1000;
      h[i + 2] = (T)2000;
    }
  const size_t pad = 256;
  T *D = nullptr;
  cudaMalloc(&D, h.size() * sizeof(T) + 2 * pad);
  T *base = D + pad / sizeof(T);
  const int step = (int)sizeof(T) * 3 * W;
  printf("%s (W=%d, step=%d, C^2*sizeof=%d):\n", tag, W, step,
         (int)(sizeof(T) * 9));
  for (int sy = 0; sy < H; ++sy) {
    for (int sx = 0; sx < W; ++sx) {
      cudaMemcpy(base, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
      NppiConnectedRegion reg;
      memset(&reg, 0xEE, sizeof reg);
      T nv[3] = {1, 2, 3};
      NppStatus st;
      if (sizeof(T) == 1) {
        Npp8u *p8 = (Npp8u *)base;
        Npp8u v8[3] = {1, 2, 3};
        st = nppiFloodFill_8u_C3IR_Ctx(p8, step, NppiPoint{sx, sy}, v8, nppiNormL1, NppiSize{W, H},
                                       &reg, buf, C);
      } else {
        Npp16u *p16 = (Npp16u *)base;
        Npp16u v16[3] = {1, 2, 3};
        st = nppiFloodFill_16u_C3IR_Ctx(p16, step, NppiPoint{sx, sy}, v16, nppiNormL1,
                                        NppiSize{W, H}, &reg, buf, C);
      }
      (void)nv;
      const long long off = (long long)sy * step + (long long)sx * 3 * sizeof(T);
      const long long fitted = (long long)sy * step + (long long)sx * 9 * sizeof(T);
      printf("  (%2d,%2d) correct=%3lld fit=%3lld -> idx %3u", sx, sy, off, fitted,
             reg.aSeedPixelValue[0]);
      if ((sx + 1) % 4 == 0) printf("\n");
      if (st != NPP_NO_ERROR) printf(" [st %d]", (int)st);
    }
  }
  cudaFree(D);
}

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const NppStreamContext C = make_ctx();
  Npp8u *buf = nullptr;
  cudaMalloc(&buf, 1 << 16);
  seedmap<Npp8u>("8u C3", 8, 4, C, buf);
  seedmap<Npp8u>("8u C3", 12, 4, C, buf);
  seedmap<Npp16u>("16u C3", 8, 4, C, buf);
  cudaFree(buf);
  return 0;
}
