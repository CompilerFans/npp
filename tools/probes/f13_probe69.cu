// f13: which step values does nppiDistanceTransformPBA_8u32f_C1R accept?
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
  hs[(size_t)5 * W + 5] = 10;
  hs[(size_t)40 * W + 20] = 11;
  size_t bs = 0;
  nppiDistanceTransformPBAGetBufferSize(NppiSize{W, H}, &bs);
  Npp8u *D, *buf;
  Npp16s *vor, *idx, *rel;
  Npp32f *tr;
  cudaMalloc(&D, W * H);
  cudaMalloc(&buf, bs);
  cudaMalloc(&vor, W * H * 2 * 4);
  cudaMalloc(&idx, W * H * 2 * 4);
  cudaMalloc(&rel, W * H * 2 * 4);
  cudaMalloc(&tr, W * H * 4 * 4);
  cudaMemcpy(D, hs.data(), hs.size(), cudaMemcpyHostToDevice);
  int srcSteps[] = {W, 2 * W, 4 * W, 256, 512};
  int steps16[] = {2 * W, 4 * W, 8 * W, 128, 256, 512};
  int steps32[] = {4 * W, 8 * W, 16 * W, 256, 512, 1024};
  auto call = [&](int ss, int s16v, int s16i, int s16r, int s32) {
    return (int)nppiDistanceTransformPBA_8u32f_C1R_Ctx(D, ss, 10, 11, vor, s16v, idx, s16i, rel,
                                                       s16r, tr, s32, NppiSize{W, H}, buf, C);
  };
  printf("baseline (64/128/128/128/256) -> %d\n", call(W, 2 * W, 2 * W, 2 * W, 4 * W));
  for (int s : srcSteps) printf("srcstep %4d -> %d\n", s, call(s, 2 * W, 2 * W, 2 * W, 4 * W));
  for (int s : steps16) printf("vorstep %4d -> %d\n", s, call(W, s, 2 * W, 2 * W, 4 * W));
  for (int s : steps16) printf("idxstep %4d -> %d\n", s, call(W, 2 * W, s, 2 * W, 4 * W));
  for (int s : steps16) printf("relstep %4d -> %d\n", s, call(W, 2 * W, 2 * W, s, 4 * W));
  for (int s : steps32) printf("trstep  %4d -> %d\n", s, call(W, 2 * W, 2 * W, 2 * W, s));
  // null pointers
  printf("null transform -> %d\n",
         (int)nppiDistanceTransformPBA_8u32f_C1R_Ctx(D, W, 10, 11, vor, 2 * W, idx, 2 * W, rel,
                                                     2 * W, nullptr, 4 * W, NppiSize{W, H}, buf, C));
  printf("null buffer -> %d\n",
         (int)nppiDistanceTransformPBA_8u32f_C1R_Ctx(D, W, 10, 11, vor, 2 * W, idx, 2 * W, rel,
                                                     2 * W, tr, 4 * W, NppiSize{W, H}, nullptr, C));
  printf("null src -> %d\n",
         (int)nppiDistanceTransformPBA_8u32f_C1R_Ctx(nullptr, W, 10, 11, vor, 2 * W, idx, 2 * W,
                                                     rel, 2 * W, tr, 4 * W, NppiSize{W, H}, buf, C));
  cudaFree(D); cudaFree(buf); cudaFree(vor); cudaFree(idx); cudaFree(rel); cudaFree(tr);
  return 0;
}
