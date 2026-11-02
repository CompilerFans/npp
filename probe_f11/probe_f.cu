// Probe F: unsharp kernel quantization per dtype, r=0 auto-radius tables,
// and the kernel-vs-ROI size constraint.
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

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

static Npp8u *g_buf = nullptr;

static void gbs_call(const char *var, float r, float s, int *need, NppStatus *st) {
  if (!strcmp(var,"8u_C1R")) *st = nppiFilterUnsharpGetBufferSize_8u_C1R(r,s,need);
  else if (!strcmp(var,"16u_C1R")) *st = nppiFilterUnsharpGetBufferSize_16u_C1R(r,s,need);
  else if (!strcmp(var,"16s_C1R")) *st = nppiFilterUnsharpGetBufferSize_16s_C1R(r,s,need);
  else if (!strcmp(var,"32f_C1R")) *st = nppiFilterUnsharpGetBufferSize_32f_C1R(r,s,need);
  else if (!strcmp(var,"16s_AC4R")) *st = nppiFilterUnsharpGetBufferSize_16s_AC4R(r,s,need);
  else if (!strcmp(var,"8u_C4R")) *st = nppiFilterUnsharpGetBufferSize_8u_C4R(r,s,need);
  else *st = NPP_ERROR;
}

template <class T>
static NppStatus ush_c1(const T *src, int sstep, NppiPoint off, T *dst, int dstep, NppiSize roi,
                        float rad, float sig, float w, float thr, NppiBorderType bord, Npp8u *buf) {
  if constexpr (std::is_same_v<T, Npp8u>) return nppiFilterUnsharpBorder_8u_C1R_Ctx(src,sstep,off,dst,dstep,roi,rad,sig,w,thr,bord,buf,mkctx());
  else if constexpr (std::is_same_v<T, Npp16u>) return nppiFilterUnsharpBorder_16u_C1R_Ctx(src,sstep,off,dst,dstep,roi,rad,sig,w,thr,bord,buf,mkctx());
  else if constexpr (std::is_same_v<T, Npp16s>) return nppiFilterUnsharpBorder_16s_C1R_Ctx(src,sstep,off,dst,dstep,roi,rad,sig,w,thr,bord,buf,mkctx());
  else return nppiFilterUnsharpBorder_32f_C1R_Ctx(src,sstep,off,dst,dstep,roi,rad,sig,w,thr,bord,buf,mkctx());
}

// dump kernel summary from g_buf (k*k floats)
static void ksum(const char *tag, int k) {
  int n = k * k;
  float *h = (float*)malloc(n * 4);
  CK(cudaMemcpy(h, g_buf, n * 4, cudaMemcpyDeviceToHost));
  double sum = 0; float mn = 1e30f, mx = -1e30f;
  for (int i = 0; i < n; ++i) { sum += h[i]; if (h[i] < mn) mn = h[i]; if (h[i] > mx) mx = h[i]; }
  printf("KSUM %s k=%d sum=%.9g min=%.9g max=%.9g\n", tag, k, sum, mn, mx);
  printf("KCTR %s center_row:", tag);
  for (int x = 0; x < k; ++x) printf(" %.9g", h[(k / 2) * k + x]);
  printf("\n");
  if (k >= 5) {
    printf("KCOR %s corner5:", tag);
    for (int y = 0; y < 5; ++y) for (int x = 0; x < 5; ++x) printf(" %.9g", h[y * k + x]);
    printf("\n");
  }
  // check tap grid: are all values multiples of a quantum?
  double q = 0;
  for (int i = 1; i < n; ++i) {
    double a = h[i], b = h[0];
    // try to detect quantum via gcd-ish approach on significant values
  }
  (void)q;
  free(h);
}

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  CK(cudaMalloc(&g_buf, 1 << 22));
  const int SW = 128, SH = 128; // source scratch
  Npp8u *s8; Npp16u *s16u; Npp16s *s16s; Npp32f *s32;
  CK(cudaMalloc(&s8, SW * SH)); CK(cudaMalloc(&s16u, SW * SH * 2));
  CK(cudaMalloc(&s16s, SW * SH * 2)); CK(cudaMalloc(&s32, SW * SH * 4));
  CK(cudaMemset(s8, 100, SW * SH)); CK(cudaMemset(s16u, 1000, SW * SH * 2));
  CK(cudaMemset(s16s, 100, SW * SH * 2)); CK(cudaMemset(s32, 0, SW * SH * 4));

  // ---------------- Part A: r=0 auto-radius table per dtype
  printf("== r0 table\n");
  {
    const char *vars[] = {"8u_C1R", "16u_C1R", "16s_C1R", "32f_C1R", "16s_AC4R", "8u_C4R"};
    float sigs[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.65f, 0.8f, 1.0f, 1.5f, 2.0f, 3.0f, 5.0f, 10.0f, 20.0f, 40.0f};
    for (auto v : vars)
      for (float s : sigs) {
        int need = -1; NppStatus st;
        gbs_call(v, 0.0f, s, &need, &st);
        printf("R0 %s s=%.6g status=%d size=%d\n", v, s, (int)st, need);
      }
  }

  // ---------------- Part B: kernel dumps per dtype (r>0 and r=0)
  printf("== kernels\n");
  {
    struct KD { const char *tag; const char *var; float r, s; };
    KD kds[] = {
      {"8u_r2_s1", "8u_C1R", 2.0f, 1.0f}, {"16u_r2_s1", "16u_C1R", 2.0f, 1.0f},
      {"16s_r2_s1", "16s_C1R", 2.0f, 1.0f}, {"32f_r2_s1", "32f_C1R", 2.0f, 1.0f},
      {"8u_r1_s1", "8u_C1R", 1.0f, 1.0f}, {"8u_r4_s1", "8u_C1R", 4.0f, 1.0f},
      {"8u_r2_s05", "8u_C1R", 2.0f, 0.5f}, {"8u_r2_s2", "8u_C1R", 2.0f, 2.0f},
      {"8u_r0_s1", "8u_C1R", 0.0f, 1.0f}, {"8u_r0_s2", "8u_C1R", 0.0f, 2.0f},
      {"8u_r0_s5", "8u_C1R", 0.0f, 5.0f}, {"16u_r0_s1", "16u_C1R", 0.0f, 1.0f},
      {"16s_r0_s1", "16s_C1R", 0.0f, 1.0f}, {"32f_r0_s1", "32f_C1R", 0.0f, 1.0f},
      {"32f_r0_s2", "32f_C1R", 0.0f, 2.0f}, {"32f_r0_s5", "32f_C1R", 0.0f, 5.0f},
    };
    for (auto &kd : kds) {
      int need = -1; NppStatus st;
      gbs_call(kd.var, kd.r, kd.s, &need, &st);
      if (st != 0 || need <= 0) { printf("KD %s gbs_status=%d need=%d\n", kd.tag, (int)st, need); continue; }
      int k = (int)sqrtf((float)(need / 4));
      if ((long long)k * k * 4 != need) { printf("KD %s need=%d not k^2*4\n", kd.tag, need); continue; }
      // image big enough for the kernel
      int W = k + 8, H = k + 8;
      if (W > SW) W = SW; if (H > SH) H = SH;
      CK(cudaMemset(g_buf, 0x5A, 1 << 22));
      NppStatus us = NPP_ERROR;
      if (!strcmp(kd.var, "8u_C1R")) us = ush_c1<Npp8u>(s8, SW, NppiPoint{0,0}, s8 + SW*SH/2, SW, NppiSize{W,H}, kd.r, kd.s, 1.0f, 0.0f, NPP_BORDER_REPLICATE, g_buf);
      else if (!strcmp(kd.var, "16u_C1R")) us = ush_c1<Npp16u>(s16u, SW*2, NppiPoint{0,0}, s16u + SW*SH/4, SW*2, NppiSize{W,H}, kd.r, kd.s, 1.0f, 0.0f, NPP_BORDER_REPLICATE, g_buf);
      else if (!strcmp(kd.var, "16s_C1R")) us = ush_c1<Npp16s>(s16s, SW*2, NppiPoint{0,0}, s16s + SW*SH/4, SW*2, NppiSize{W,H}, kd.r, kd.s, 1.0f, 0.0f, NPP_BORDER_REPLICATE, g_buf);
      else us = ush_c1<Npp32f>(s32, SW*4, NppiPoint{0,0}, s32 + SW*SH/8, SW*4, NppiSize{W,H}, kd.r, kd.s, 1.0f, 0.0f, NPP_BORDER_REPLICATE, g_buf);
      printf("KD %s k=%d status=%d\n", kd.tag, k, (int)us);
      if (us == 0) ksum(kd.tag, k);
    }
  }

  // ---------------- Part C: kernel-vs-ROI constraint (exact k control via r>0)
  printf("== roi constraint\n");
  {
    // r=8 -> k=17 ; r=10 -> k=21 ; r=1 -> k=3 ; r=0 -> k=9? no: use r>0
    struct RC { int r; int w, h; };
    RC rcs[] = {
      {8, 16, 16}, {8, 17, 17}, {8, 18, 18}, {8, 17, 16}, {8, 16, 17},
      {10, 20, 20}, {10, 21, 21}, {10, 21, 20}, {10, 20, 21},
      {1, 1, 1}, {1, 2, 2}, {1, 3, 3}, {1, 5, 1}, {1, 1, 5},
      {2, 4, 4}, {2, 5, 5}, {2, 6, 3}, {2, 3, 6},
      {0, 15, 15}, {0, 16, 16},  // r=0, s=1 -> 8u k=7
      {20, 41, 41}, {20, 40, 41}, {20, 41, 40},
    };
    for (auto &rc : rcs) {
      NppStatus st = nppiFilterUnsharpBorder_8u_C1R_Ctx(s8, SW, NppiPoint{0,0}, s8 + SW*SH/2, SW,
          NppiSize{rc.w, rc.h}, (float)rc.r, 1.0f, 1.0f, 0.0f, NPP_BORDER_REPLICATE, g_buf, mkctx());
      printf("ROI r=%d k=%d w=%d h=%d status=%d\n", rc.r, 2*rc.r+1, rc.w, rc.h, (int)st);
    }
  }
  return 0;
}
