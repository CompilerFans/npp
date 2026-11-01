// f13: nppiDistanceTransformPBA_8u32f with 2-short Voronoi/indices/manhattan
// outputs (per the header's step requirement of width*2*sizeof(Npp16s)).
#include <nppi.h>
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>

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
  const int W = 64, H = 64;
  std::vector<Npp8u> hs((size_t)W * H, 0);
  struct Site { int x, y, v; };
  std::vector<Site> sites = {{5, 5, 10}, {40, 20, 11}, {30, 55, 12}};
  for (auto &s : sites) hs[(size_t)s.y * W + s.x] = (Npp8u)s.v;
  size_t bs = 0;
  nppiDistanceTransformPBAGetBufferSize(NppiSize{W, H}, &bs);
  Npp8u *D, *buf;
  Npp16s *vor, *idx, *rel;
  Npp32f *tr;
  cudaMalloc(&D, W * H);
  cudaMalloc(&buf, bs);
  cudaMalloc(&vor, W * H * 4);
  cudaMalloc(&idx, W * H * 4);
  cudaMalloc(&rel, W * H * 4);
  cudaMalloc(&tr, W * H * 4);
  cudaMemcpy(D, hs.data(), hs.size(), cudaMemcpyHostToDevice);
  int st = (int)nppiDistanceTransformPBA_8u32f_C1R_Ctx(D, W, 10, 12, vor, 4 * W, idx, 4 * W, rel,
                                                       4 * W, tr, 4 * W, NppiSize{W, H}, buf, C);
  printf("call -> %d\n", st);
  std::vector<Npp16s> hv((size_t)W * H * 2), hi((size_t)W * H * 2), hr((size_t)W * H * 2);
  std::vector<Npp32f> ht((size_t)W * H);
  cudaMemcpy(hv.data(), vor, W * H * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(hi.data(), idx, W * H * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(hr.data(), rel, W * H * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(ht.data(), tr, W * H * 4, cudaMemcpyDeviceToHost);
  int px[][2] = {{0,0},{4,4},{5,5},{2,2},{20,10},{39,20},{40,20},{50,30},{30,40},{30,55},{63,63}};
  printf("  (x,y)     tr    vor0 vor1   idx0 idx1   rel0 rel1\n");
  for (auto &p : px) {
    int i = p[1] * W + p[0];
    printf(" (%2d,%2d) %7.2f %5d%5d %6d%5d %6d%5d\n", p[0], p[1], ht[i], (int)hv[2*i], (int)hv[2*i+1],
           (int)hi[2*i], (int)hi[2*i+1], (int)hr[2*i], (int)hr[2*i+1]);
  }
  printf("transform global max: ");
  float mx = 0; int mi = 0;
  for (int i = 0; i < W * H; ++i) if (ht[i] > mx) { mx = ht[i]; mi = i; }
  printf("%g at (%d,%d)\n", mx, mi % W, mi / W);
  cudaFree(D); cudaFree(buf); cudaFree(vor); cudaFree(idx); cudaFree(rel); cudaFree(tr);
  return 0;
}
