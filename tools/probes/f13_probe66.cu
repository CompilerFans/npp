// f13: nppiDistanceTransformPBA_8u32f_C1R - basic semantics probe.
#include <nppi.h>
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>

static const char *sname(NppStatus s) {
  static char b[32];
  if (s == NPP_NO_ERROR) return "OK";
  if (s == NPP_NULL_POINTER_ERROR) return "NULL";
  if (s == NPP_SIZE_ERROR) return "SIZE";
  if (s == NPP_STEP_ERROR) return "STEP";
  if (s == NPP_NOT_SUPPORTED_MODE_ERROR) return "NOTSUP";
  if (s == NPP_BAD_ARGUMENT_ERROR) return "BADARG";
  if (s == NPP_OUT_OFF_RANGE_ERROR) return "OUTOFRANGE";
  if (s == NPP_CUDA_KERNEL_EXECUTION_ERROR) return "KERNEL";
  snprintf(b, sizeof b, "%d", (int)s);
  return b;
}

static NppStreamContext make_ctx() {
  NppStreamContext c;
  memset(&c, 0, sizeof c);
  c.hStream = 0;
  cudaGetDevice(&c.nCudaDeviceId);
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, c.nCudaDeviceId);
  c.nMultiProcessorCount = p.multiProcessorCount;
  c.nMaxThreadsPerMultiProcessor = p.maxThreadsPerMultiProcessor;
  c.nMaxThreadsPerBlock = p.maxThreadsPerBlock;
  c.nSharedMemPerBlock = p.sharedMemPerBlock;
  c.nCudaDevAttrComputeCapabilityMajor = p.major;
  c.nCudaDevAttrComputeCapabilityMinor = p.minor;
  cudaStreamGetFlags(c.hStream, &c.nStreamFlags);
  return c;
}

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const NppStreamContext C = make_ctx();
  const int W = 12, H = 10;
  size_t bs = 0;
  NppStatus st = nppiDistanceTransformPBAGetBufferSize(NppiSize{W, H}, &bs);
  printf("buffer size -> %s %zu\n", sname(st), bs);
  size_t abs = 0;
  st = nppiDistanceTransformPBAGetAntialiasingBufferSize(NppiSize{W, H}, &abs);
  printf("antialiasing buffer -> %s %zu\n", sname(st), abs);

  std::vector<Npp8u> hs((size_t)W * H, 255);
  hs[2 * W + 2] = 0;   // site at (2,2)
  hs[7 * W + 9] = 0;   // site at (9,7)
  hs[1 * W + 8] = 3;   // out-of-range value (not a site with min=max=0)

  Npp8u *D = nullptr, *buf = nullptr;
  Npp16s *vor = nullptr, *idx = nullptr, *rel = nullptr;
  Npp32f *tr = nullptr;
  cudaMalloc(&D, W * H);
  cudaMalloc(&buf, bs ? bs : 1024);
  cudaMalloc(&vor, W * H * 2);
  cudaMalloc(&idx, W * H * 2);
  cudaMalloc(&rel, W * H * 2);
  cudaMalloc(&tr, W * H * 4);
  cudaMemcpy(D, hs.data(), hs.size(), cudaMemcpyHostToDevice);

  st = nppiDistanceTransformPBA_8u32f_C1R_Ctx(D, W, 0, 0, vor, 2 * W, idx, 2 * W, rel, 2 * W, tr, 4 * W,
                                              NppiSize{W, H}, buf, C);
  printf("8u32f min=0 max=0 -> %s\n", sname(st));
  std::vector<Npp16s> hv((size_t)W * H), hi((size_t)W * H), hr((size_t)W * H);
  std::vector<Npp32f> ht((size_t)W * H);
  cudaMemcpy(hv.data(), vor, W * H * 2, cudaMemcpyDeviceToHost);
  cudaMemcpy(hi.data(), idx, W * H * 2, cudaMemcpyDeviceToHost);
  cudaMemcpy(hr.data(), rel, W * H * 2, cudaMemcpyDeviceToHost);
  cudaMemcpy(ht.data(), tr, W * H * 4, cudaMemcpyDeviceToHost);
  printf("transform:\n");
  for (int y = 0; y < H; ++y) {
    printf("  ");
    for (int x = 0; x < W; ++x) printf("%6.1f", ht[y * W + x]);
    printf("\n");
  }
  printf("voronoi (src value):\n");
  for (int y = 0; y < H; ++y) {
    printf("  ");
    for (int x = 0; x < W; ++x) printf("%3d", (int)hv[y * W + x]);
    printf("\n");
  }
  printf("indices:\n");
  for (int y = 0; y < H; ++y) {
    printf("  ");
    for (int x = 0; x < W; ++x) printf("%3d", (int)hi[y * W + x]);
    printf("\n");
  }
  printf("relative manhattan:\n");
  for (int y = 0; y < H; ++y) {
    printf("  ");
    for (int x = 0; x < W; ++x) printf("%3d", (int)hr[y * W + x]);
    printf("\n");
  }
  cudaFree(D); cudaFree(buf); cudaFree(vor); cudaFree(idx); cudaFree(rel); cudaFree(tr);
  return 0;
}
