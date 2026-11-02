// f13: nppiDistanceTransformPBA_8u32f - compare NPP against a brute-force
// nearest-site model (squared Euclidean distance, linear site index, source
// value and manhattan distance outputs) on a 64x64 image.
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

int main(int argc, char **argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const NppStreamContext C = make_ctx();
  const int W = 64, H = 64;
  std::vector<Npp8u> hs((size_t)W * H, 0);
  // sites: value encodes the site id (10+id); everything else 0
  struct Site { int x, y, v; };
  std::vector<Site> sites = {{5, 5, 10}, {40, 20, 11}, {30, 55, 12}, {60, 60, 13}, {20, 30, 14}};
  for (auto &s : sites) hs[(size_t)s.y * W + s.x] = (Npp8u)s.v;
  if (argc > 1) {  // random-ish extra sites
    for (int i = 0; i < atoi(argv[1]); ++i) {
      int x = (i * 7919 + 3) % W, y = (i * 104729 + 11) % H;
      hs[(size_t)y * W + x] = 10;
      sites.push_back({x, y, 10});
    }
  }

  size_t bs = 0;
  NppStatus st = nppiDistanceTransformPBAGetBufferSize(NppiSize{W, H}, &bs);
  printf("buffer %zu (st %d)\n", bs, (int)st);
  Npp8u *D, *buf;
  Npp16s *vor, *idx, *rel;
  Npp32f *tr;
  cudaMalloc(&D, W * H);
  cudaMalloc(&buf, bs);
  cudaMalloc(&vor, W * H * 2);
  cudaMalloc(&idx, W * H * 2);
  cudaMalloc(&rel, W * H * 2);
  cudaMalloc(&tr, W * H * 4);
  cudaMemcpy(D, hs.data(), hs.size(), cudaMemcpyHostToDevice);
  st = nppiDistanceTransformPBA_8u32f_C1R_Ctx(D, W, 10, 14, vor, 2 * W, idx, 2 * W, rel, 2 * W, tr,
                                              4 * W, NppiSize{W, H}, buf, C);
  printf("call -> %d\n", (int)st);
  std::vector<Npp16s> hv((size_t)W * H), hi((size_t)W * H), hr((size_t)W * H);
  std::vector<Npp32f> ht((size_t)W * H);
  cudaMemcpy(hv.data(), vor, W * H * 2, cudaMemcpyDeviceToHost);
  cudaMemcpy(hi.data(), idx, W * H * 2, cudaMemcpyDeviceToHost);
  cudaMemcpy(hr.data(), rel, W * H * 2, cudaMemcpyDeviceToHost);
  cudaMemcpy(ht.data(), tr, W * H * 4, cudaMemcpyDeviceToHost);

  // brute force
  int badT = 0, badV = 0, badI = 0, badR = 0;
  int firstBad[4] = {-1, -1, -1, -1};
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      int best = -1;
      long long bestD = 1LL << 60;
      for (size_t s = 0; s < sites.size(); ++s) {
        long long dx = x - sites[s].x, dy = y - sites[s].y;
        long long d = dx * dx + dy * dy;
        if (d < bestD) { bestD = d; best = (int)s; }
      }
      const int i = y * W + x;
      if (best < 0) continue;
      if (ht[i] != (Npp32f)bestD) { if (!badT++) firstBad[0] = i; }
      if ((int)hv[i] != sites[best].v) { if (!badV++) firstBad[1] = i; }
      if ((int)hi[i] != sites[best].y * W + sites[best].x) { if (!badI++) firstBad[2] = i; }
      int man = abs(x - sites[best].x) + abs(y - sites[best].y);
      if ((int)hr[i] != man) { if (!badR++) firstBad[3] = i; }
    }
  printf("mismatches: transform %d voronoi %d indices %d relman %d\n", badT, badV, badI, badR);
  for (int k = 0; k < 4; ++k) {
    int i = firstBad[k];
    if (i < 0) continue;
    int x = i % W, y = i / W;
    printf("  first bad[%d] at (%d,%d): npp t=%g v=%d idx=%d rel=%d\n", k, x, y, ht[i], (int)hv[i],
           (int)hi[i], (int)hr[i]);
  }
  // exact distance: is the transform squared or linear?
  printf("sample row 5 x=0..12: ");
  for (int x = 0; x < 13; ++x) printf("%g ", ht[5 * W + x]);
  printf("\n");
  printf("sample row 6 x=0..12: ");
  for (int x = 0; x < 13; ++x) printf("%g ", ht[6 * W + x]);
  printf("\n");
  printf("idx row 5: ");
  for (int x = 0; x < 13; ++x) printf("%d ", (int)hi[5 * W + x]);
  printf("\nrel row 6: ");
  for (int x = 0; x < 13; ++x) printf("%d ", (int)hr[6 * W + x]);
  printf("\n");
  cudaFree(D); cudaFree(buf); cudaFree(vor); cudaFree(idx); cudaFree(rel); cudaFree(tr);
  return 0;
}
