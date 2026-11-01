// Probe M: unsharp final edges:
//  1) threshold shape: random data, thr>0 (hard-zero vs shrink), thr<0
//  2) GBS edges: r>0 with sigma=0; sigma=NaN; the four floats around 1e-6
//  3) single tiny-sigma GBS call (argv sigma hex-bits) for hang-boundary mapping
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>

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
static uint32_t lcg_state = 12345;
static uint32_t lcg() { lcg_state = lcg_state * 1664525u + 1013904223u; return lcg_state; }

static void thr_case(const char *tag, int dt, int W, int H, float r, float sg, float w, float thr) {
  const int es = (dt == 0) ? 1 : 4;
  const size_t n = (size_t)W * H;
  void *hs = malloc(n * es), *hd = malloc(n * es);
  lcg_state = 555;
  for (size_t i = 0; i < n; ++i) {
    uint32_t x = lcg();
    if (dt == 0) ((Npp8u *)hs)[i] = (Npp8u)((x >> 16) & 0xFF);
    else ((Npp32f *)hs)[i] = (float)((int)((x >> 8) & 0xFFFF) - 32768) * 0.0002f; // +-6.5
  }
  memset(hd, 0, n * es);
  void *ds, *dd, *buf; CK(cudaMalloc(&ds, n * es)); CK(cudaMalloc(&dd, n * es)); CK(cudaMalloc(&buf, 1 << 20));
  CK(cudaMemcpy(ds, hs, n * es, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dd, hd, n * es, cudaMemcpyHostToDevice));
  int step = W * es;
  NppStatus st;
  if (dt == 0)
    st = nppiFilterUnsharpBorder_8u_C1R_Ctx((Npp8u *)ds, step, NppiPoint{0, 0}, (Npp8u *)dd, step, NppiSize{W, H}, r, sg, w, thr, NPP_BORDER_REPLICATE, (Npp8u *)buf, mkctx());
  else
    st = nppiFilterUnsharpBorder_32f_C1R_Ctx((Npp32f *)ds, step, NppiPoint{0, 0}, (Npp32f *)dd, step, NppiSize{W, H}, r, sg, w, thr, NPP_BORDER_REPLICATE, (Npp8u *)buf, mkctx());
  CK(cudaMemcpy(hd, dd, n * es, cudaMemcpyDeviceToHost));
  printf("THRX %s dt=%d W=%d H=%d r=%a s=%a w=%a thr=%a status=%d\n", tag, dt, W, H, r, sg, w, thr, (int)st);
  for (int y = 0; y < H; ++y) {
    printf("  S%d", y);
    for (int x = 0; x < W; ++x) {
      if (dt == 0) printf(" %d", ((Npp8u *)hs)[y * W + x]);
      else printf(" %a", ((Npp32f *)hs)[y * W + x]);
    }
    printf("\n");
  }
  for (int y = 0; y < H; ++y) {
    printf("  D%d", y);
    for (int x = 0; x < W; ++x) {
      if (dt == 0) printf(" %d", ((Npp8u *)hd)[y * W + x]);
      else printf(" %a", ((Npp32f *)hd)[y * W + x]);
    }
    printf("\n");
  }
  cudaFree(ds); cudaFree(dd); free(hs); free(hd);
}

int main(int argc, char **argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  cudaFree(0);
  int part = (argc > 1) ? atoi(argv[1]) : 1;
  if (part == 1) {
    printf("== part1 threshold shape (random, r=2 s=1 k=5)\n");
    thr_case("t32_pos", 3, 24, 12, 2.0f, 1.0f, 1.0f, 0.5f);
    thr_case("t32_pos2", 3, 24, 12, 2.0f, 1.0f, 1.7f, 0.25f);
    thr_case("t32_neg", 3, 24, 12, 2.0f, 1.0f, 1.0f, -0.5f);
    thr_case("t8_pos", 0, 24, 12, 2.0f, 1.0f, 1.0f, 3.0f);
    thr_case("t8_neg", 0, 24, 12, 2.0f, 1.0f, 1.0f, -3.0f);
  }
  if (part == 2) {
    printf("== part2 GBS edges\n");
    struct { float r, s; const char *t; } cs[] = {
      {0.5f, 0.0f, "r05_s0"}, {1.0f, 0.0f, "r1_s0"}, {2.0f, 0.0f, "r2_s0"},
      {1.0f, NAN, "r1_snan"},
    };
    for (int dt = 0; dt < 4; ++dt) {
      for (auto &c : cs) {
        int n = -1; NppStatus st;
        switch (dt) {
        case 0: st = nppiFilterUnsharpGetBufferSize_8u_C1R(c.r, c.s, &n); break;
        case 1: st = nppiFilterUnsharpGetBufferSize_16u_C1R(c.r, c.s, &n); break;
        case 2: st = nppiFilterUnsharpGetBufferSize_16s_C1R(c.r, c.s, &n); break;
        default: st = nppiFilterUnsharpGetBufferSize_32f_C1R(c.r, c.s, &n); break;
        }
        printf("GBSE dt=%d %s r=%a s=%a status=%d size=%d\n", dt, c.t, c.r, c.s, (int)st, n);
      }
    }
    uint32_t bits[] = {0x358637BD, 0x358637BE, 0x358637BF, 0x358637C0}; // ~1e-6..1e-6+3ulp
    for (int dt = 0; dt < 4; ++dt) {
      for (uint32_t b : bits) {
        float s; memcpy(&s, &b, 4);
        int n = -1; NppStatus st;
        switch (dt) {
        case 0: st = nppiFilterUnsharpGetBufferSize_8u_C1R(0.0f, s, &n); break;
        case 1: st = nppiFilterUnsharpGetBufferSize_16u_C1R(0.0f, s, &n); break;
        case 2: st = nppiFilterUnsharpGetBufferSize_16s_C1R(0.0f, s, &n); break;
        default: st = nppiFilterUnsharpGetBufferSize_32f_C1R(0.0f, s, &n); break;
        }
        printf("GBSE dt=%d six sig=%a status=%d size=%d\n", dt, s, (int)st, n);
      }
    }
  }
  if (part == 4) {
    printf("== part4 GBS auto with sigma<0\n");
    for (int dt = 0; dt < 4; ++dt) {
      for (float s : {-1.0f, -0.5f}) {
        int n = -1; NppStatus st;
        switch (dt) {
        case 0: st = nppiFilterUnsharpGetBufferSize_8u_C1R(0.0f, s, &n); break;
        case 1: st = nppiFilterUnsharpGetBufferSize_16u_C1R(0.0f, s, &n); break;
        case 2: st = nppiFilterUnsharpGetBufferSize_16s_C1R(0.0f, s, &n); break;
        default: st = nppiFilterUnsharpGetBufferSize_32f_C1R(0.0f, s, &n); break;
        }
        printf("GBSA dt=%d s=%a status=%d size=%d\n", dt, s, (int)st, n);
      }
    }
  }
  if (part == 3) {
    // single auto GBS call at sigma given as argv[2] decimal; for hang mapping
    if (argc > 2) {
      float s = (float)atof(argv[2]);
      int n = -1;
      NppStatus st = nppiFilterUnsharpGetBufferSize_8u_C1R(0.0f, s, &n);
      printf("TINYS sigma=%.10g status=%d size=%d\n", s, (int)st, n);
    }
  }
  return 0;
}
