// Probe P: unsharp threshold scale + AC4 alpha + GBS for the test matrix.
//
//  part1: threshold scan.  Flat background with one delta pixel; report the
//         destination value at the delta centre (sharpened or copied) for a
//         grid of (amplitude, threshold) per dtype.
//  part2: AC4/C4 alpha: destination pre-filled with a sentinel, report dst
//         alpha and whether the alpha channel was written.
//  part3: GBS sizes for the parameter matrix used by the diff tests.
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
static Npp8u *g_buf = nullptr;

// run one unsharp call for dtype dt (0=8u,1=16u,2=16s,3=32f), C1, flat bg + delta at (3,3)
static NppStatus run_flat(int dt, float bg, float amp, int W, int H, float w, float thr, void *hdOut) {
  const int es = (dt == 0) ? 1 : (dt == 3) ? 4 : 2;
  void *hs = malloc((size_t)W * H * es);
  for (int i = 0; i < W * H; ++i) {
    if (dt == 0) ((Npp8u *)hs)[i] = (Npp8u)bg;
    else if (dt == 1) ((Npp16u *)hs)[i] = (Npp16u)bg;
    else if (dt == 2) ((Npp16s *)hs)[i] = (Npp16s)bg;
    else ((Npp32f *)hs)[i] = bg;
  }
  if (dt == 0) ((Npp8u *)hs)[3 * W + 3] = (Npp8u)(bg + amp);
  else if (dt == 1) ((Npp16u *)hs)[3 * W + 3] = (Npp16u)(bg + amp);
  else if (dt == 2) ((Npp16s *)hs)[3 * W + 3] = (Npp16s)(bg + amp);
  else ((Npp32f *)hs)[3 * W + 3] = bg + amp;
  void *ds, *dd; CK(cudaMalloc(&ds, (size_t)W * H * es)); CK(cudaMalloc(&dd, (size_t)W * H * es));
  CK(cudaMemcpy(ds, hs, (size_t)W * H * es, cudaMemcpyHostToDevice));
  CK(cudaMemset(dd, 0, (size_t)W * H * es));
  NppStatus st;
  if (dt == 0) st = nppiFilterUnsharpBorder_8u_C1R_Ctx((Npp8u *)ds, W, NppiPoint{0, 0}, (Npp8u *)dd, W, NppiSize{W, H}, 2.0f, 1.0f, w, thr, NPP_BORDER_REPLICATE, g_buf, mkctx());
  else if (dt == 1) st = nppiFilterUnsharpBorder_16u_C1R_Ctx((Npp16u *)ds, W * 2, NppiPoint{0, 0}, (Npp16u *)dd, W * 2, NppiSize{W, H}, 2.0f, 1.0f, w, thr, NPP_BORDER_REPLICATE, g_buf, mkctx());
  else if (dt == 2) st = nppiFilterUnsharpBorder_16s_C1R_Ctx((Npp16s *)ds, W * 2, NppiPoint{0, 0}, (Npp16s *)dd, W * 2, NppiSize{W, H}, 2.0f, 1.0f, w, thr, NPP_BORDER_REPLICATE, g_buf, mkctx());
  else st = nppiFilterUnsharpBorder_32f_C1R_Ctx((Npp32f *)ds, W * 4, NppiPoint{0, 0}, (Npp32f *)dd, W * 4, NppiSize{W, H}, 2.0f, 1.0f, w, thr, NPP_BORDER_REPLICATE, g_buf, mkctx());
  CK(cudaMemcpy(hdOut, dd, (size_t)W * H * es, cudaMemcpyDeviceToHost));
  free(hs);
  cudaFree(ds);
  cudaFree(dd);
  return st;
}

int main(int argc, char **argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  cudaFree(0);
  CK(cudaMalloc(&g_buf, 1 << 22));
  int part = (argc > 1) ? atoi(argv[1]) : 0;
  const int W = 8, H = 6;

  if (part == 1) {
    printf("== part1 threshold scan (w=1, r=2 sigma=1, delta at (3,3))\n");
    struct C { int dt; float bg, amp; } cs[] = {
      {0, 100, 8}, {0, 100, 16}, {0, 100, 32}, {0, 100, 64}, {0, 100, 100}, {0, 100, 127}, {0, 100, 155},
      {1, 1000, 2000}, {1, 1000, 8000}, {1, 1000, 16384}, {1, 1000, 32767},
      {2, -1000, 2000}, {2, -1000, 8000}, {2, -1000, 16384}, {2, -1000, 32767},
      {3, 1.0f, 0.5f}, {3, 1.0f, 1.0f}, {3, 0.0f, 100.0f}, {3, 0.0f, 837.0f}, {3, 0.0f, 5000.0f},
    };
    float thrs[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f, 1.5f, 2.0f, 4.0f, 8.0f, 16.0f, 64.0f, 256.0f,
                    1e-6f, 1e-4f, 1e-2f, 0.1f};
    for (auto &c : cs) {
      for (float t : thrs) {
        const int es = (c.dt == 0) ? 1 : (c.dt == 3) ? 4 : 2;
        void *hd = malloc((size_t)W * H * es);
        NppStatus st = run_flat(c.dt, c.bg, c.amp, W, H, 1.0f, t, hd);
        printf("THR dt=%d bg=%g amp=%g thr=%a status=%d src=", c.dt, c.bg, c.amp, t, (int)st);
        if (c.dt == 0) printf("%d", ((Npp8u *)hd)[3 * W + 3]);
        else if (c.dt == 1) printf("%d", ((Npp16u *)hd)[3 * W + 3]);
        else if (c.dt == 2) printf("%d", ((Npp16s *)hd)[3 * W + 3]);
        else printf("%a", ((Npp32f *)hd)[3 * W + 3]);
        printf(" bg_out=");
        if (c.dt == 0) printf("%d", ((Npp8u *)hd)[0]);
        else if (c.dt == 1) printf("%d", ((Npp16u *)hd)[0]);
        else if (c.dt == 2) printf("%d", ((Npp16s *)hd)[0]);
        else printf("%a", ((Npp32f *)hd)[0]);
        printf("\n");
        free(hd);
      }
    }
  }
  if (part == 2) {
    printf("== part2 AC4/C4 alpha sentinel\n");
    const int W2 = 10, H2 = 8;
    for (int dt = 0; dt < 4; ++dt) {
      const int es = (dt == 0) ? 1 : (dt == 3) ? 4 : 2;
      const size_t n = (size_t)W2 * H2 * 4;
      void *hs = malloc(n * es), *hd = malloc(n * es);
      for (size_t i = 0; i < n; ++i) {
        if (dt == 0) ((Npp8u *)hs)[i] = (Npp8u)(i * 7 + 3);
        else if (dt == 1) ((Npp16u *)hs)[i] = (Npp16u)(i * 977 + 5);
        else if (dt == 2) ((Npp16s *)hs)[i] = (Npp16s)(i * 977 - 30000);
        else ((Npp32f *)hs)[i] = (float)((int)(i % 101) - 50) * 0.25f;
        if (dt == 0) ((Npp8u *)hd)[i] = 0xAA;
        else if (dt == 1) ((Npp16u *)hd)[i] = 0xAAAA;
        else if (dt == 2) ((Npp16s *)hd)[i] = (Npp16s)-12345;
        else ((Npp32f *)hd)[i] = -777.0f;
      }
      void *ds, *dd; CK(cudaMalloc(&ds, n * es)); CK(cudaMalloc(&dd, n * es));
      CK(cudaMemcpy(ds, hs, n * es, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dd, hd, n * es, cudaMemcpyDeviceToHost));
      const int sstep = W2 * 4 * es, dstep = W2 * 4 * es;
      NppStatus st;
      if (dt == 0) st = nppiFilterUnsharpBorder_8u_AC4R_Ctx((Npp8u *)ds, sstep, NppiPoint{0, 0}, (Npp8u *)dd, dstep, NppiSize{W2, H2}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf, mkctx());
      else if (dt == 1) st = nppiFilterUnsharpBorder_16u_AC4R_Ctx((Npp16u *)ds, sstep, NppiPoint{0, 0}, (Npp16u *)dd, dstep, NppiSize{W2, H2}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf, mkctx());
      else if (dt == 2) st = nppiFilterUnsharpBorder_16s_AC4R_Ctx((Npp16s *)ds, sstep, NppiPoint{0, 0}, (Npp16s *)dd, dstep, NppiSize{W2, H2}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf, mkctx());
      else st = nppiFilterUnsharpBorder_32f_AC4R_Ctx((Npp32f *)ds, sstep, NppiPoint{0, 0}, (Npp32f *)dd, dstep, NppiSize{W2, H2}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf, mkctx());
      CK(cudaMemcpy(hd, dd, n * es, cudaMemcpyDeviceToHost));
      printf("AC4 dt=%d status=%d alpha_first=", dt, (int)st);
      if (dt == 0) printf("%d rgb=%d", ((Npp8u *)hd)[3], ((Npp8u *)hd)[0]);
      else if (dt == 1) printf("%d rgb=%d", ((Npp16u *)hd)[3], ((Npp16u *)hd)[0]);
      else if (dt == 2) printf("%d rgb=%d", ((Npp16s *)hd)[3], ((Npp16s *)hd)[0]);
      else printf("%a rgb=%a", ((Npp32f *)hd)[3], ((Npp32f *)hd)[0]);
      printf("\n");
      free(hs); free(hd); cudaFree(ds); cudaFree(dd);
    }
  }
  if (part == 3) {
    printf("== part3 GBS matrix\n");
    struct C { float r, s; } cs[] = {
      {1.0f, 1.0f}, {2.0f, 1.0f}, {3.0f, 1.0f}, {2.0f, 0.7f}, {2.0f, 1.5f},
      {0.0f, 1.0f}, {0.0f, 0.5f}, {0.0f, 2.0f}, {0.5f, 1.0f}, {1.5f, 0.8f},
      {0.0f, 3.0f}, {4.0f, 2.0f}, {0.0f, 0.25f},
    };
    for (auto &c : cs) {
      int a = -1, b = -1, d = -1, e = -1, f = -1, g = -1;
      NppStatus sa = nppiFilterUnsharpGetBufferSize_8u_C1R(c.r, c.s, &a);
      NppStatus sb = nppiFilterUnsharpGetBufferSize_8u_AC4R(c.r, c.s, &b);
      NppStatus sd = nppiFilterUnsharpGetBufferSize_16u_C1R(c.r, c.s, &d);
      NppStatus se = nppiFilterUnsharpGetBufferSize_16s_C3R(c.r, c.s, &e);
      NppStatus sf = nppiFilterUnsharpGetBufferSize_32f_C4R(c.r, c.s, &f);
      NppStatus sg = nppiFilterUnsharpGetBufferSize_32f_AC4R(c.r, c.s, &g);
      printf("GBS r=%a s=%a 8u_c1=%d/%d 8u_ac4=%d/%d 16u_c1=%d/%d 16s_c3=%d/%d 32f_c4=%d/%d 32f_ac4=%d/%d\n",
             c.r, c.s, (int)sa, a, (int)sb, b, (int)sd, d, (int)se, e, (int)sf, f, (int)sg, g);
    }
  }
  cudaFree(g_buf);
  return 0;
}
