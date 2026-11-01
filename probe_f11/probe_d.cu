// Probe D: one risky unsharp parameter case per invocation (run under timeout).
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define CK(x) do { cudaError_t e_=(x); if(e_){printf("CUDAERR %s\n", cudaGetErrorString(e_)); exit(1);} } while(0)

static NppStreamContext mkctx() {
  NppStreamContext c{};
  c.hStream = 0;
  int dev = 0; CK(cudaGetDevice(&dev)); c.nCudaDeviceId = dev;
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
  c.nMultiProcessorCount = prop.multiProcessorCount;
  c.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
  c.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
  c.nSharedMemPerBlock = prop.sharedMemPerBlock;
  return c;
}

int main(int argc, char **argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 2) { printf("usage: d <case>\n"); return 2; }
  std::string cs = argv[1];
  NppStreamContext c = mkctx();
  const int W = 16, H = 8;
  Npp8u *ds, *dd, *buf;
  CK(cudaMalloc(&ds, W * H * 32)); CK(cudaMalloc(&dd, W * H * 32)); CK(cudaMalloc(&buf, 1 << 20));
  CK(cudaMemset(ds, 100, W * H * 32));
  NppStatus st = NPP_ERROR;

  // ---- GetBufferSize degenerate cases
  if (cs.rfind("gbs:", 0) == 0) {
    float r = 1.0f, sg = 1.0f;
    sscanf(argv[2], "%f", &r); sscanf(argv[3], "%f", &sg);
    printf("BEFORE gbs %s r=%g s=%g\n", cs.c_str(), r, sg);
    int n = -1;
    if (cs == "gbs:8u_C1R") st = nppiFilterUnsharpGetBufferSize_8u_C1R(r, sg, &n);
    else if (cs == "gbs:32f_C1R") st = nppiFilterUnsharpGetBufferSize_32f_C1R(r, sg, &n);
    else if (cs == "gbs:8u_AC4R") st = nppiFilterUnsharpGetBufferSize_8u_AC4R(r, sg, &n);
    else if (cs == "gbs:16s_C3R") st = nppiFilterUnsharpGetBufferSize_16s_C3R(r, sg, &n);
    printf("AFTER gbs %s status=%d size=%d\n", cs.c_str(), (int)st, n);
    return 0;
  }
  // ---- UnsharpBorder parameter degenerates (8u C1)
  float r = 2.0f, sg = 1.0f, w = 1.2f, thr = 0.0f;
  NppiSize roi{W, H};
  if (cs == "radneg") r = -1.0f;
  else if (cs == "radzero") r = 0.0f;
  else if (cs == "signeg") sg = -1.0f;
  else if (cs == "sigzero") sg = 0.0f;
  else if (cs == "radbig") r = 100.0f;
  else if (cs == "sigbig") sg = 100.0f;
  else { printf("unknown case\n"); return 2; }
  printf("BEFORE unsharp %s r=%g s=%g\n", cs.c_str(), r, sg);
  st = nppiFilterUnsharpBorder_8u_C1R_Ctx(ds, W, NppiPoint{0, 0}, dd, W, roi, r, sg, w, thr,
                                          NPP_BORDER_REPLICATE, buf, c);
  printf("AFTER unsharp %s status=%d\n", cs.c_str(), (int)st);
  return 0;
}
