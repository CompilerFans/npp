// Probe O: unsharp border geometry (32f exact dumps) and threshold semantics.
//
//  part1: 32f C1, image 24x16 random, several (pSrc shift, oSrcOffset) combos,
//         ROI 10x8, r=2 sigma=1 w=1.2 thr=0.  Full S and D hex dumps so the
//         read geometry can be solved exactly (blur = s - (d-s)/w).
//  part2: threshold semantics with a strong delta so |src-blur| is large:
//         8u and 32f, several thresholds.
//  part4: 8u/16u/16s/32f constant-image emission (value 128/30000/-1000/1.5)
//         at r=2 sigma=1 for w in {1.2,1,-0.7,0} and threshold 0/0.5.
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

static Npp8u *g_buf = nullptr;

static NppStatus ush32(const Npp32f *s, int ss, NppiPoint off, Npp32f *d, int ds, NppiSize roi,
                       float r, float sg, float w, float thr, NppiBorderType b, Npp8u *buf) {
  return nppiFilterUnsharpBorder_32f_C1R_Ctx(s, ss, off, d, ds, roi, r, sg, w, thr, b, buf, mkctx());
}
static NppStatus ush8(const Npp8u *s, int ss, NppiPoint off, Npp8u *d, int ds, NppiSize roi,
                      float r, float sg, float w, float thr, NppiBorderType b, Npp8u *buf) {
  return nppiFilterUnsharpBorder_8u_C1R_Ctx(s, ss, off, d, ds, roi, r, sg, w, thr, b, buf, mkctx());
}

static void geo_case(const char *tag, int shiftX, int shiftY, int offX, int offY) {
  const int SW = 24, SH = 16, W = 10, H = 8;
  Npp32f hs[SW * SH], hd[W * H];
  lcg_state = 777;
  for (int i = 0; i < SW * SH; ++i)
    hs[i] = (float)((int)((lcg() >> 8) & 0xFFFF) - 32768) * 0.0004f;
  Npp32f *ds, *dd;
  CK(cudaMalloc(&ds, sizeof(hs)));
  CK(cudaMalloc(&dd, sizeof(hd)));
  CK(cudaMemcpy(ds, hs, sizeof(hs), cudaMemcpyHostToDevice));
  for (int i = 0; i < W * H; ++i) hd[i] = -777.0f;
  CK(cudaMemcpy(dd, hd, sizeof(hd), cudaMemcpyHostToDevice));
  NppStatus st = ush32(ds + shiftY * SW + shiftX, SW * 4, NppiPoint{offX, offY}, dd, W * 4,
                       NppiSize{W, H}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf);
  CK(cudaMemcpy(hd, dd, sizeof(hd), cudaMemcpyDeviceToHost));
  printf("GEO %s shift=(%d,%d) off=(%d,%d) status=%d\n", tag, shiftX, shiftY, offX, offY, (int)st);
  for (int y = 0; y < SH; ++y) {
    printf("  S%d", y);
    for (int x = 0; x < SW; ++x) printf(" %a", hs[y * SW + x]);
    printf("\n");
  }
  for (int y = 0; y < H; ++y) {
    printf("  D%d", y);
    for (int x = 0; x < W; ++x) printf(" %a", hd[y * W + x]);
    printf("\n");
  }
  cudaFree(ds);
  cudaFree(dd);
}

static void const_case(const char *tag, int dt, float w, float thr) {
  const int W = 8, H = 8;
  const int es = (dt == 0) ? 1 : (dt == 3) ? 4 : 2;
  void *hs = malloc((size_t)W * H * es), *hd = malloc((size_t)W * H * es);
  for (int i = 0; i < W * H; ++i) {
    if (dt == 0) ((Npp8u *)hs)[i] = 128;
    else if (dt == 1) ((Npp16u *)hs)[i] = 30000;
    else if (dt == 2) ((Npp16s *)hs)[i] = -1000;
    else ((Npp32f *)hs)[i] = 1.5f;
  }
  memset(hd, 0, (size_t)W * H * es);
  void *ds, *dd;
  CK(cudaMalloc(&ds, (size_t)W * H * es));
  CK(cudaMalloc(&dd, (size_t)W * H * es));
  CK(cudaMemcpy(ds, hs, (size_t)W * H * es, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dd, hd, (size_t)W * H * es, cudaMemcpyHostToDevice));
  NppStatus st;
  if (dt == 0) st = ush8((Npp8u *)ds, W, NppiPoint{0, 0}, (Npp8u *)dd, W, NppiSize{W, H}, 2.0f, 1.0f, w, thr, NPP_BORDER_REPLICATE, g_buf);
  else if (dt == 1) st = nppiFilterUnsharpBorder_16u_C1R_Ctx((Npp16u *)ds, W * 2, NppiPoint{0, 0}, (Npp16u *)dd, W * 2, NppiSize{W, H}, 2.0f, 1.0f, w, thr, NPP_BORDER_REPLICATE, g_buf, mkctx());
  else if (dt == 2) st = nppiFilterUnsharpBorder_16s_C1R_Ctx((Npp16s *)ds, W * 2, NppiPoint{0, 0}, (Npp16s *)dd, W * 2, NppiSize{W, H}, 2.0f, 1.0f, w, thr, NPP_BORDER_REPLICATE, g_buf, mkctx());
  else st = ush32((Npp32f *)ds, W * 4, NppiPoint{0, 0}, (Npp32f *)dd, W * 4, NppiSize{W, H}, 2.0f, 1.0f, w, thr, NPP_BORDER_REPLICATE, g_buf);
  CK(cudaMemcpy(hd, dd, (size_t)W * H * es, cudaMemcpyDeviceToHost));
  printf("CONST %s dt=%d w=%a thr=%a status=%d d00=", tag, dt, w, thr, (int)st);
  if (dt == 0) printf("%d\n", ((Npp8u *)hd)[0]);
  else if (dt == 1) printf("%d\n", ((Npp16u *)hd)[0]);
  else if (dt == 2) printf("%d\n", ((Npp16s *)hd)[0]);
  else printf("%a\n", ((Npp32f *)hd)[0]);
  free(hs);
  free(hd);
  cudaFree(ds);
  cudaFree(dd);
}

static void thr_case8(const char *tag, float thr) {
  const int W = 12, H = 6;
  Npp8u hs[W * H], hd[W * H];
  for (int i = 0; i < W * H; ++i) hs[i] = 100;
  hs[3 * W + 3] = 200;
  hs[3 * W + 4] = 0;
  Npp8u *ds, *dd;
  CK(cudaMalloc(&ds, sizeof(hs)));
  CK(cudaMalloc(&dd, sizeof(hd)));
  CK(cudaMemcpy(ds, hs, sizeof(hs), cudaMemcpyHostToDevice));
  for (int i = 0; i < W * H; ++i) hd[i] = 7;
  CK(cudaMemcpy(dd, hd, sizeof(hd), cudaMemcpyHostToDevice));
  NppStatus st = ush8(ds, W, NppiPoint{0, 0}, dd, W, NppiSize{W, H}, 2.0f, 1.0f, 1.0f, thr, NPP_BORDER_REPLICATE, g_buf);
  CK(cudaMemcpy(hd, dd, sizeof(hd), cudaMemcpyDeviceToHost));
  printf("THR8 %s thr=%a status=%d\n", tag, thr, (int)st);
  for (int y = 0; y < H; ++y) {
    printf("  D%d", y);
    for (int x = 0; x < W; ++x) printf(" %d", hd[y * W + x]);
    printf("\n");
  }
  cudaFree(ds);
  cudaFree(dd);
}

static void thr_case32(const char *tag, float thr) {
  const int W = 12, H = 6;
  Npp32f hs[W * H], hd[W * H];
  for (int i = 0; i < W * H; ++i) hs[i] = 1.0f;
  hs[3 * W + 3] = 2.0f;
  hs[3 * W + 4] = 0.0f;
  Npp32f *ds, *dd;
  CK(cudaMalloc(&ds, sizeof(hs)));
  CK(cudaMalloc(&dd, sizeof(hd)));
  CK(cudaMemcpy(ds, hs, sizeof(hs), cudaMemcpyHostToDevice));
  for (int i = 0; i < W * H; ++i) hd[i] = -7.0f;
  CK(cudaMemcpy(dd, hd, sizeof(hd), cudaMemcpyHostToDevice));
  NppStatus st = ush32(ds, W * 4, NppiPoint{0, 0}, dd, W * 4, NppiSize{W, H}, 2.0f, 1.0f, 1.0f, thr, NPP_BORDER_REPLICATE, g_buf);
  CK(cudaMemcpy(hd, dd, sizeof(hd), cudaMemcpyDeviceToHost));
  printf("THR32 %s thr=%a status=%d\n", tag, thr, (int)st);
  for (int y = 0; y < H; ++y) {
    printf("  D%d", y);
    for (int x = 0; x < W; ++x) printf(" %a", hd[y * W + x]);
    printf("\n");
  }
  cudaFree(ds);
  cudaFree(dd);
}

int main(int argc, char **argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  cudaFree(0);
  CK(cudaMalloc(&g_buf, 1 << 22));
  int part = (argc > 1) ? atoi(argv[1]) : 0;

  if (part == 1) {
    printf("== part1 geometry (32f)\n");
    geo_case("origin_off00", 0, 0, 0, 0);
    geo_case("origin_off32", 0, 0, 3, 2);
    geo_case("origin_off10", 0, 0, 1, 0);
    geo_case("shift32_off32", 3, 2, 3, 2);
  }
  if (part == 2) {
    printf("== part2 threshold\n");
    for (float t : {0.0f, 0.5f, 1.0f, 2.0f, 3.0f, 5.0f, 10.0f, 100.0f}) thr_case8("d8", t);
    for (float t : {0.0f, 1e-6f, 0.001f, 0.01f, 0.1f, 0.25f, 0.5f, 10.0f}) thr_case32("d32", t);
  }
  if (part == 4) {
    printf("== part4 constant emission\n");
    for (int dt = 0; dt < 4; ++dt) {
      for (float w : {1.2f, 1.0f, 0.0f, -0.7f}) {
        for (float t : {0.0f, 0.5f}) const_case("const", dt, w, t);
      }
    }
  }
  cudaFree(g_buf);
  return 0;
}
