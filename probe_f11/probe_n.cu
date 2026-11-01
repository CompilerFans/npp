// Probe N: unsharp border offset/addressing geometry via 32f impulse responses.
//  part 1: impulse sweep, 32f C1, w=1 thr=0, r=2 sigma=1, ROI 8x8
//          cfg A: pSrc=buffer origin, offset (2,1)   [probe-l part7 geometry]
//          cfg B: pSrc=buffer+(2,1), offset (2,1)    [contract-correct usage]
//          cfg C: pSrc=buffer origin, offset (0,0)   [baseline]
//          each impulse: buffer 32x20 zeros + 1.0 at P; dst 16x12 sentinel-filled
//  part 2: 8u impulse sweep with w=-1 (dst = trunc(blur)) for cfg A/B
//  part 3: dst write-region map (one impulse, full dst dump, sentinel)
//  part 4: 32f AC4 / 8u AC4 status re-test with printed args
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

static const int BW = 32, BH = 20;   // source buffer
static const int DW = 16, DH = 12;   // dst buffer
static const int RW = 8, RH = 8;     // ROI

int main(int argc, char **argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  cudaFree(0);
  CK(cudaMalloc(&g_buf, 1 << 22));
  int part = (argc > 1) ? atoi(argv[1]) : 1;
  const int sweep_w = (argc > 2) ? atoi(argv[2]) : 16;
  const int sweep_h = (argc > 3) ? atoi(argv[3]) : 12;

  if (part == 1 || part == 3) {
    float *hs = (float *)calloc((size_t)BW * BH, 4);
    float *hd = (float *)malloc((size_t)DW * DH * 4);
    float *ds, *dd;
    CK(cudaMalloc(&ds, (size_t)BW * BH * 4));
    CK(cudaMalloc(&dd, (size_t)DW * DH * 4));
    const float SENT = -12345.0f;
    struct Cfg { const char *name; int bx, by; int ox, oy; };
    Cfg cfgs[3] = {{"A", 0, 0, 2, 1}, {"B", 2, 1, 2, 1}, {"C", 0, 0, 0, 0}};
    for (int ci = 0; ci < 3; ++ci) {
      printf("== part%d cfg %s pSrc=(%d,%d) off=(%d,%d) ROI %dx%d\n", part, cfgs[ci].name,
             cfgs[ci].bx, cfgs[ci].by, cfgs[ci].ox, cfgs[ci].oy, RW, RH);
      for (int py = 0; py < sweep_h; ++py) {
        for (int px = 0; px < sweep_w; ++px) {
          memset(hs, 0, (size_t)BW * BH * 4);
          hs[py * BW + px] = 1.0f;
          CK(cudaMemcpy(ds, hs, (size_t)BW * BH * 4, cudaMemcpyHostToDevice));
          for (int i = 0; i < DW * DH; ++i) hd[i] = SENT;
          CK(cudaMemcpy(dd, hd, (size_t)DW * DH * 4, cudaMemcpyHostToDevice));
          Npp32f *psrc = ds + cfgs[ci].by * BW + cfgs[ci].bx;
          NppStatus st = nppiFilterUnsharpBorder_32f_C1R_Ctx(psrc, BW * 4, NppiPoint{cfgs[ci].ox, cfgs[ci].oy},
                                                             dd, DW * 4, NppiSize{RW, RH}, 2.0f, 1.0f, 1.0f, 0.0f,
                                                             NPP_BORDER_REPLICATE, g_buf, mkctx());
          if (st != 0) { printf("%s P(%d,%d) status=%d\n", cfgs[ci].name, px, py, (int)st); continue; }
          CK(cudaMemcpy(hd, dd, (size_t)DW * DH * 4, cudaMemcpyDeviceToHost));
          printf("%s P(%d,%d)", cfgs[ci].name, px, py);
          int nresp = 0;
          for (int y = 0; y < DH; ++y)
            for (int x = 0; x < DW; ++x)
              if (hd[y * DW + x] != SENT && hd[y * DW + x] != 0.0f) {
                printf(" [%d,%d]=%a", y, x, hd[y * DW + x]); ++nresp;
              }
          printf(" n=%d\n", nresp);
        }
      }
      if (part == 3) {
        // full dst dump for one center impulse
        memset(hs, 0, (size_t)BW * BH * 4);
        hs[4 * BW + 4] = 1.0f;
        CK(cudaMemcpy(ds, hs, (size_t)BW * BH * 4, cudaMemcpyHostToDevice));
        for (int i = 0; i < DW * DH; ++i) hd[i] = SENT;
        CK(cudaMemcpy(dd, hd, (size_t)DW * DH * 4, cudaMemcpyHostToDevice));
        Npp32f *psrc = ds + cfgs[ci].by * BW + cfgs[ci].bx;
        NppStatus st = nppiFilterUnsharpBorder_32f_C1R_Ctx(psrc, BW * 4, NppiPoint{cfgs[ci].ox, cfgs[ci].oy},
                                                           dd, DW * 4, NppiSize{RW, RH}, 2.0f, 1.0f, 1.0f, 0.0f,
                                                           NPP_BORDER_REPLICATE, g_buf, mkctx());
        printf("%s FULLDUMP status=%d\n", cfgs[ci].name, (int)st);
        CK(cudaMemcpy(hd, dd, (size_t)DW * DH * 4, cudaMemcpyDeviceToHost));
        for (int y = 0; y < DH; ++y) {
          printf("  D%d", y);
          for (int x = 0; x < DW; ++x) printf(" %a", hd[y * DW + x]);
          printf("\n");
        }
      }
    }
    cudaFree(ds); cudaFree(dd); free(hs); free(hd);
  }

  if (part == 2) {
    Npp8u *hs = (Npp8u *)calloc((size_t)BW * BH, 1);
    Npp8u *hd = (Npp8u *)malloc((size_t)DW * DH);
    Npp8u *ds, *dd;
    CK(cudaMalloc(&ds, (size_t)BW * BH));
    CK(cudaMalloc(&dd, (size_t)DW * DH));
    struct Cfg { const char *name; int bx, by; int ox, oy; };
    Cfg cfgs[2] = {{"A8", 0, 0, 2, 1}, {"B8", 2, 1, 2, 1}};
    for (int ci = 0; ci < 2; ++ci) {
      printf("== part2 cfg %s pSrc=(%d,%d) off=(%d,%d) w=-1\n", cfgs[ci].name, cfgs[ci].bx, cfgs[ci].by,
             cfgs[ci].ox, cfgs[ci].oy);
      for (int py = 0; py < sweep_h; ++py) {
        for (int px = 0; px < sweep_w; ++px) {
          memset(hs, 0, (size_t)BW * BH);
          hs[py * BW + px] = 255;
          CK(cudaMemcpy(ds, hs, (size_t)BW * BH, cudaMemcpyHostToDevice));
          memset(hd, 0xAA, (size_t)DW * DH);
          CK(cudaMemcpy(dd, hd, (size_t)DW * DH, cudaMemcpyHostToDevice));
          Npp8u *psrc = ds + cfgs[ci].by * BW + cfgs[ci].bx;
          NppStatus st = nppiFilterUnsharpBorder_8u_C1R_Ctx(psrc, BW, NppiPoint{cfgs[ci].ox, cfgs[ci].oy},
                                                             dd, DW, NppiSize{RW, RH}, 2.0f, 1.0f, -1.0f, 0.0f,
                                                             NPP_BORDER_REPLICATE, g_buf, mkctx());
          if (st != 0) { printf("%s P(%d,%d) status=%d\n", cfgs[ci].name, px, py, (int)st); continue; }
          CK(cudaMemcpy(hd, dd, (size_t)DW * DH, cudaMemcpyDeviceToHost));
          printf("%s P(%d,%d)", cfgs[ci].name, px, py);
          int nresp = 0;
          for (int y = 0; y < DH; ++y)
            for (int x = 0; x < DW; ++x)
              if (hd[y * DW + x] != 0xAA && hd[y * DW + x] != 0) {
                printf(" [%d,%d]=%d", y, x, hd[y * DW + x]); ++nresp;
              }
          printf(" n=%d\n", nresp);
        }
      }
    }
    cudaFree(ds); cudaFree(dd); free(hs); free(hd);
  }

  if (part == 4) {
    printf("== part4 AC4 status re-test\n");
    // 32f AC4: 16x12 image, r=2 sigma=1
    {
      const int W = 16, H = 12, C = 4;
      float *hs = (float *)calloc((size_t)W * H * C, 4);
      float *hd = (float *)malloc((size_t)W * H * C * 4);
      for (int i = 0; i < W * H * C; ++i) hs[i] = (float)(i % 97) * 0.25f;
      float *ds, *dd; CK(cudaMalloc(&ds, (size_t)W * H * C * 4)); CK(cudaMalloc(&dd, (size_t)W * H * C * 4));
      CK(cudaMemcpy(ds, hs, (size_t)W * H * C * 4, cudaMemcpyHostToDevice));
      NppStatus st = nppiFilterUnsharpBorder_32f_AC4R_Ctx(ds, W * C * 4, NppiPoint{0, 0}, dd, W * C * 4,
                                                          NppiSize{W, H}, 2.0f, 1.0f, 1.2f, 0.0f,
                                                          NPP_BORDER_REPLICATE, g_buf, mkctx());
      CK(cudaMemcpy(hd, dd, (size_t)W * H * C * 4, cudaMemcpyDeviceToHost));
      printf("AC4 32f status=%d alpha0=%a alpha1=%a rgb0=%a\n", (int)st, hd[3], hd[W * C + 3], hd[0]);
      st = nppiFilterUnsharpBorder_32f_C4R_Ctx(ds, W * C * 4, NppiPoint{0, 0}, dd, W * C * 4,
                                               NppiSize{W, H}, 2.0f, 1.0f, 1.2f, 0.0f,
                                               NPP_BORDER_REPLICATE, g_buf, mkctx());
      printf("C4  32f status=%d\n", (int)st);
      // C3 for comparison
      float *hs3 = (float *)calloc((size_t)W * H * 3, 4);
      float *ds3, *dd3; CK(cudaMalloc(&ds3, (size_t)W * H * 3 * 4)); CK(cudaMalloc(&dd3, (size_t)W * H * 3 * 4));
      CK(cudaMemcpy(ds3, hs3, (size_t)W * H * 3 * 4, cudaMemcpyHostToDevice));
      st = nppiFilterUnsharpBorder_32f_C3R_Ctx(ds3, W * 3 * 4, NppiPoint{0, 0}, dd3, W * 3 * 4,
                                               NppiSize{W, H}, 2.0f, 1.0f, 1.2f, 0.0f,
                                               NPP_BORDER_REPLICATE, g_buf, mkctx());
      printf("C3  32f status=%d\n", (int)st);
      cudaFree(ds); cudaFree(dd); cudaFree(ds3); cudaFree(dd3); free(hs); free(hd); free(hs3);
    }
    // 8u AC4
    {
      const int W = 16, H = 12, C = 4;
      Npp8u *hs = (Npp8u *)malloc((size_t)W * H * C);
      Npp8u *hd = (Npp8u *)malloc((size_t)W * H * C);
      for (int i = 0; i < W * H * C; ++i) hs[i] = (Npp8u)(i % 251);
      Npp8u *ds, *dd; CK(cudaMalloc(&ds, (size_t)W * H * C)); CK(cudaMalloc(&dd, (size_t)W * H * C));
      CK(cudaMemcpy(ds, hs, (size_t)W * H * C, cudaMemcpyHostToDevice));
      NppStatus st = nppiFilterUnsharpBorder_8u_AC4R_Ctx(ds, W * C, NppiPoint{0, 0}, dd, W * C,
                                                         NppiSize{W, H}, 2.0f, 1.0f, 1.2f, 0.0f,
                                                         NPP_BORDER_REPLICATE, g_buf, mkctx());
      CK(cudaMemcpy(hd, dd, (size_t)W * H * C, cudaMemcpyDeviceToHost));
      printf("AC4 8u status=%d alpha0=%d alpha1=%d rgb0=%d\n", (int)st, hd[3], hd[W * C + 3], hd[0]);
      cudaFree(ds); cudaFree(dd); free(hs); free(hd);
    }
  }
  return 0;
}
