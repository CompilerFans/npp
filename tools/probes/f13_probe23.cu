// f13 probe 23: UP pyramid residual questions:
//  (1) exact bits of eps-noise at r=3/4/10, (2) taps=4 chain, (3) taps status,
//  (4) UP DstROI helper sweep, (5) UP status matrix, (6) 8u/16u saturation+neg, (7) 16u .5 rounding.
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>

static const char *es(NppStatus s) { static char b[32]; snprintf(b, 32, "%d", (int)s); return b; }
static NppStreamContext ctx;

static void up32a(const char *tag, Npp32f *d, int W, int H, float r, int t, const Npp32f *k,
                  NppiBorderType bt, int x0, int nx, int row) {
  NppiSize mn{}, mx{};
  nppiGetFilterGaussPyramidLayerUpBorderDstROI(W, H, &mn, &mx, r);
  NppiSize dst = mx;
  Npp32f *o; cudaMalloc(&o, 1 << 22); cudaMemset(o, 0, 1 << 22);
  NppStatus st = nppiFilterGaussPyramidLayerUpBorder_32f_C1R_Ctx(
      d, W * 4, NppiSize{W, H}, NppiPoint{0, 0}, o, dst.width * 4, dst, r, t, k, bt, ctx);
  std::vector<Npp32f> v((size_t)dst.width * dst.height);
  cudaMemcpy(v.data(), o, v.size() * 4, cudaMemcpyDeviceToHost);
  printf("%s st=%s:", tag, es(st));
  for (int x = x0; x < x0 + nx && x < dst.width; ++x) printf(" %a", v[row * dst.width + x]);
  printf("\n");
  cudaFree(o);
}

int main(int argc, char **argv) {
  int part = argc > 1 ? atoi(argv[1]) : 0;
  ctx.hStream = 0;
  if (part == 1) {  // exact bits of eps-noise
    const int W = 200, H = 5;
    std::vector<Npp32f> h((size_t)W * H);
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) h[y * W + x] = (float)x;
    Npp32f *d; cudaMalloc(&d, W * H * 4); cudaMemcpy(d, h.data(), W * H * 4, cudaMemcpyHostToDevice);
    Npp32f e1[3]={0,1,0}, e2[3]={0,0,1}, e0[3]={1,0,0};
    Npp32f f1[5]={0,1,0,0,0}, f2[5]={0,0,1,0,0}, f3[5]={0,0,0,1,0}, f4[5]={0,0,0,0,1};
    float rs[] = {3.0f, 4.0f, 10.0f, 2.0f, 5.0f};
    for (float r : rs) {
      char t[64];
      snprintf(t,64,"r=%g e0",r); up32a(t,d,W,H,r,3,e0,NPP_BORDER_REPLICATE,9,4,2);
      snprintf(t,64,"r=%g e1",r); up32a(t,d,W,H,r,3,e1,NPP_BORDER_REPLICATE,9,4,2);
      snprintf(t,64,"r=%g e2",r); up32a(t,d,W,H,r,3,e2,NPP_BORDER_REPLICATE,9,4,2);
      snprintf(t,64,"r=%g e0b",r); up32a(t,d,W,H,r,3,e0,NPP_BORDER_REPLICATE,28,4,2);
      snprintf(t,64,"r=%g e1b",r); up32a(t,d,W,H,r,3,e1,NPP_BORDER_REPLICATE,28,4,2);
      snprintf(t,64,"r=%g e2b",r); up32a(t,d,W,H,r,3,e2,NPP_BORDER_REPLICATE,28,4,2);
      snprintf(t,64,"r=%g f3",r); up32a(t,d,W,H,r,5,f3,NPP_BORDER_REPLICATE,9,4,2);
      snprintf(t,64,"r=%g f4",r); up32a(t,d,W,H,r,5,f4,NPP_BORDER_REPLICATE,9,4,2);
      snprintf(t,64,"r=%g f4b",r); up32a(t,d,W,H,r,5,f4,NPP_BORDER_REPLICATE,48,4,2);
    }
    cudaFree(d);
  }
  if (part == 2) {  // taps=4 chain + taps status
    const int W = 129, H = 5;
    std::vector<Npp32f> h((size_t)W * H);
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) h[y * W + x] = (float)x;
    Npp32f *d; cudaMalloc(&d, W * H * 4); cudaMemcpy(d, h.data(), W * H * 4, cudaMemcpyHostToDevice);
    Npp32f t4[4][4] = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
    const char *tn[4] = {"T0","T1","T2","T3"};
    float rs[] = {1.5f, 1.25f, 2.5f, 2.0f, 1.1f};
    for (float r : rs)
      for (int i = 0; i < 4; ++i) { char t[64]; snprintf(t,64,"r=%g taps4 %s",r,tn[i]);
        up32a(t,d,W,H,r,4,t4[i],NPP_BORDER_REPLICATE,0,18,2); }
    // taps status
    for (int tp : {0, 1, 2, 3, 4, 5, 6, 7, 101, -3}) {
      Npp32f k[8] = {0.25f, 0.5f, 0.25f, 0, 0, 0, 0, 0};
      NppiSize mn{}, mx{}; nppiGetFilterGaussPyramidLayerUpBorderDstROI(16, 16, &mn, &mx, 1.5f);
      Npp32f so[256]; Npp32f *o; cudaMalloc(&o, 4096); cudaMemset(o, 0, 4096);
      Npp32f sd[256]; for (int i = 0; i < 256; ++i) sd[i] = 1;
      Npp32f *dd; cudaMalloc(&dd, 1024); cudaMemcpy(dd, sd, 1024, cudaMemcpyHostToDevice);
      NppStatus st = nppiFilterGaussPyramidLayerUpBorder_32f_C1R_Ctx(
          dd, 64, NppiSize{16, 16}, NppiPoint{0, 0}, o, mx.width * 4, mx, 1.5f, tp, k,
          NPP_BORDER_REPLICATE, ctx);
      printf("taps=%d st=%s\n", tp, es(st));
      cudaFree(o); cudaFree(dd);
    }
    cudaFree(d);
  }
  if (part == 3) {  // UP DstROI helper sweep + status matrix
    int Ws[] = {1, 2, 3, 5, 8, 9, 10, 17, 100};
    float rs[] = {1.0001f, 1.1f, 1.25f, 1.3f, 1.5f, 2.0f, 2.4f, 2.5f, 3.0f, 3.5f, 4.0f, 7.0f, 10.0f, 1.0f, 0.5f, 10.1f, -1.5f};
    for (int W : Ws) for (float r : rs) {
      NppiSize mn{}, mx{}; NppStatus st = nppiGetFilterGaussPyramidLayerUpBorderDstROI(W, 5, &mn, &mx, r);
      printf("helper W=%d r=%g st=%s min=%dx%d max=%dx%d\n", W, r, es(st), mn.width, mn.height, mx.width, mx.height);
    }
    // status matrix (32f C1R): rate / border / null / step
    Npp32f sd[256]; for (int i = 0; i < 256; ++i) sd[i] = 1;
    Npp32f *dd; cudaMalloc(&dd, 1024); cudaMemcpy(dd, sd, 1024, cudaMemcpyHostToDevice);
    Npp32f *o; cudaMalloc(&o, 1 << 16);
    Npp32f k[3] = {0.25f, 0.5f, 0.25f};
    struct { const char *n; float r; int taps; int border; int step; int nullp; NppiSize sz; } cs[] = {
      {"ok", 1.5f, 3, 0, 96, 0, {24, 24}},
      {"r=1.0", 1.0f, 3, 0, 96, 0, {24, 24}},
      {"r=0.9", 0.9f, 3, 0, 96, 0, {24, 24}},
      {"r=10.0", 10.0f, 3, 0, 96, 0, {24, 24}},
      {"r=10.1", 10.1f, 3, 0, 96, 0, {24, 24}},
      {"r=-1", -1.0f, 3, 0, 96, 0, {24, 24}},
      {"const-b", 1.5f, 3, 1, 96, 0, {24, 24}},
      {"mirror-b", 1.5f, 3, 2, 96, 0, {24, 24}},
      {"step=23", 1.5f, 3, 0, 23, 0, {24, 24}},
      {"step=24", 1.5f, 3, 0, 24, 0, {24, 24}},
      {"nullk", 1.5f, 3, 0, 96, 1, {24, 24}},
      {"nulldst", 1.5f, 3, 0, 96, 2, {24, 24}},
      {"nullsrc", 1.5f, 3, 0, 96, 3, {24, 24}},
      {"bigroi", 1.5f, 3, 0, 4096, 0, {100, 100}},
      {"roi0", 1.5f, 3, 0, 96, 0, {0, 0}},
      {"onepx", 1.5f, 3, 0, 96, 0, {1, 1}},
    };
    for (auto &c : cs) {
      cudaMemset(o, 0, 1 << 16);
      NppiBorderType bt = c.border == 1 ? NPP_BORDER_CONSTANT : c.border == 2 ? NPP_BORDER_MIRROR : NPP_BORDER_REPLICATE;
      NppStatus st = nppiFilterGaussPyramidLayerUpBorder_32f_C1R_Ctx(
          c.nullp == 3 ? nullptr : dd, 64, NppiSize{16, 16}, NppiPoint{0, 0},
          c.nullp == 2 ? nullptr : o, c.step, c.sz, c.r, c.taps, c.nullp == 1 ? nullptr : k, bt, ctx);
      printf("upstat %-9s st=%s\n", c.n, es(st));
    }
    // src size 0 / negative
    for (int sw : {0, -3}) { cudaMemset(o, 0, 1 << 16);
      NppStatus st = nppiFilterGaussPyramidLayerUpBorder_32f_C1R_Ctx(
          dd, 64, NppiSize{sw, 16}, NppiPoint{0, 0}, o, 96, NppiSize{24, 24}, 1.5f, 3, k,
          NPP_BORDER_REPLICATE, ctx);
      printf("upstat srcW=%d st=%s\n", sw, es(st)); }
    // negative offset
    for (int ox : {-1, -5}) { cudaMemset(o, 0, 1 << 16);
      NppStatus st = nppiFilterGaussPyramidLayerUpBorder_32f_C1R_Ctx(
          dd, 64, NppiSize{16, 16}, NppiPoint{ox, 0}, o, 96, NppiSize{24, 24}, 1.5f, 3, k,
          NPP_BORDER_REPLICATE, ctx);
      printf("upstat offx=%d st=%s\n", ox, es(st)); }
    cudaFree(o); cudaFree(dd);
  }
  if (part == 4) {  // 8u/16u saturation + negative + .5 rounding
    { // 8u: const 200, kernel {2,2,2} -> 1200 ; and negative
      Npp8u sd[256]; for (int i = 0; i < 256; ++i) sd[i] = 200;
      Npp8u *dd; cudaMalloc(&dd, 256); cudaMemcpy(dd, sd, 256, cudaMemcpyHostToDevice);
      Npp8u *o; cudaMalloc(&o, 4096); cudaMemset(o, 0, 4096);
      Npp32f kbig[3] = {2, 2, 2};
      NppStatus st = nppiFilterGaussPyramidLayerUpBorder_8u_C1R_Ctx(
          dd, 16, NppiSize{16, 16}, NppiPoint{0, 0}, o, 16, NppiSize{16, 16}, 2.0f, 3, kbig,
          NPP_BORDER_REPLICATE, ctx);
      Npp8u v[256]; cudaMemcpy(v, o, 256, cudaMemcpyDeviceToHost);
      printf("8u big k st=%s v[0..3]=%d %d %d %d\n", es(st), v[0], v[1], v[2], v[3]);
      cudaMemset(o, 0, 4096);
      Npp8u sd2[256]; for (int i = 0; i < 256; ++i) sd2[i] = 10;
      cudaMemcpy(dd, sd2, 256, cudaMemcpyHostToDevice);
      Npp32f kneg[3] = {-1, 3, -1};
      st = nppiFilterGaussPyramidLayerUpBorder_8u_C1R_Ctx(
          dd, 16, NppiSize{16, 16}, NppiPoint{0, 0}, o, 16, NppiSize{16, 16}, 2.0f, 3, kneg,
          NPP_BORDER_REPLICATE, ctx);
      cudaMemcpy(v, o, 256, cudaMemcpyDeviceToHost);
      printf("8u neg k st=%s v[0..3]=%d %d %d %d\n", es(st), v[0], v[1], v[2], v[3]);
      cudaFree(o); cudaFree(dd);
    }
    { // 16u: const 40000, kernel {2,2,2} -> 240000; negative
      Npp16u sd[256]; for (int i = 0; i < 256; ++i) sd[i] = 40000;
      Npp16u *dd; cudaMalloc(&dd, 512); cudaMemcpy(dd, sd, 512, cudaMemcpyHostToDevice);
      Npp16u *o; cudaMalloc(&o, 8192); cudaMemset(o, 0, 8192);
      Npp32f kbig[3] = {2, 2, 2};
      NppStatus st = nppiFilterGaussPyramidLayerUpBorder_16u_C1R_Ctx(
          dd, 32, NppiSize{16, 16}, NppiPoint{0, 0}, o, 32, NppiSize{16, 16}, 2.0f, 3, kbig,
          NPP_BORDER_REPLICATE, ctx);
      Npp16u v[256]; cudaMemcpy(v, o, 512, cudaMemcpyDeviceToHost);
      printf("16u big k st=%s v[0..3]=%d %d %d %d\n", es(st), v[0], v[1], v[2], v[3]);
      cudaMemset(o, 0, 8192);
      Npp16u sd2[256]; for (int i = 0; i < 256; ++i) sd2[i] = 10;
      cudaMemcpy(dd, sd2, 512, cudaMemcpyHostToDevice);
      Npp32f kneg[3] = {-1, 3, -1};
      st = nppiFilterGaussPyramidLayerUpBorder_16u_C1R_Ctx(
          dd, 32, NppiSize{16, 16}, NppiPoint{0, 0}, o, 32, NppiSize{16, 16}, 2.0f, 3, kneg,
          NPP_BORDER_REPLICATE, ctx);
      cudaMemcpy(v, o, 512, cudaMemcpyDeviceToHost);
      printf("16u neg k st=%s v[0..3]=%d %d %d %d\n", es(st), v[0], v[1], v[2], v[3]);
      cudaFree(o); cudaFree(dd);
    }
    { // 16u exact-.5 rounding: 7x ramp, r=1.5 e2 tap at .5 -> 3.5 -> trunc 3 / round 4
      const int W = 64, H = 4;
      std::vector<Npp16u> h((size_t)W * H);
      for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) h[y * W + x] = (Npp16u)(7 * x);
      Npp16u *dd; cudaMalloc(&dd, W * H * 2); cudaMemcpy(dd, h.data(), W * H * 2, cudaMemcpyHostToDevice);
      Npp16u *o; cudaMalloc(&o, 1 << 16); cudaMemset(o, 0, 1 << 16);
      Npp32f e2[3] = {0, 0, 1};
      NppiSize mn{}, mx{}; nppiGetFilterGaussPyramidLayerUpBorderDstROI(W, H, &mn, &mx, 1.5f);
      NppStatus st = nppiFilterGaussPyramidLayerUpBorder_16u_C1R_Ctx(
          dd, W * 2, NppiSize{W, H}, NppiPoint{0, 0}, o, mx.width * 2, mx, 1.5f, 3, e2,
          NPP_BORDER_REPLICATE, ctx);
      std::vector<Npp16u> v((size_t)mx.width * mx.height);
      cudaMemcpy(v.data(), o, v.size() * 2, cudaMemcpyDeviceToHost);
      printf("16u 7x e2 st=%s row2:", es(st));
      for (int x = 0; x < 16; ++x) printf(" %d", (int)v[2 * mx.width + x]);
      printf("\n");
      cudaFree(o); cudaFree(dd);
    }
    { // 8u/16u DOWN with k=1/3 (recheck the all-zero anomaly, with status)
      const int W = 32, H = 4;
      std::vector<Npp8u> h8((size_t)W * H);
      std::vector<Npp16u> h16((size_t)W * H);
      for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) { h8[y * W + x] = (Npp8u)(6 * x); h16[y * W + x] = (Npp16u)(6 * x); }
      Npp8u *d8; cudaMalloc(&d8, W * H); cudaMemcpy(d8, h8.data(), W * H, cudaMemcpyHostToDevice);
      Npp16u *d16; cudaMalloc(&d16, W * H * 2); cudaMemcpy(d16, h16.data(), W * H * 2, cudaMemcpyHostToDevice);
      Npp8u *o8; cudaMalloc(&o8, 4096); cudaMemset(o8, 0, 4096);
      Npp16u *o16; cudaMalloc(&o16, 4096); cudaMemset(o16, 0, 4096);
      Npp32f k3[3] = {1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f};
      NppStatus s8 = nppiFilterGaussPyramidLayerDownBorder_8u_C1R_Ctx(
          d8, W, NppiSize{W, H}, NppiPoint{0, 0}, o8, 16, NppiSize{16, 4}, 2.0f, 3, k3,
          NPP_BORDER_REPLICATE, ctx);
      NppStatus s16 = nppiFilterGaussPyramidLayerDownBorder_16u_C1R_Ctx(
          d16, W * 2, NppiSize{W, H}, NppiPoint{0, 0}, o16, 32, NppiSize{16, 4}, 2.0f, 3, k3,
          NPP_BORDER_REPLICATE, ctx);
      Npp8u v8[64]; cudaMemcpy(v8, o8, 64, cudaMemcpyDeviceToHost);
      Npp16u v16[64]; cudaMemcpy(v16, o16, 128, cudaMemcpyDeviceToHost);
      printf("down k1/3 8u st=%s row1:", es(s8));
      for (int x = 0; x < 8; ++x) printf(" %d", (int)v8[16 + x]);
      printf(" | 16u st=%s:", es(s16));
      for (int x = 0; x < 8; ++x) printf(" %d", (int)v16[16 + x]);
      printf("\n");
      cudaFree(o8); cudaFree(o16); cudaFree(d8); cudaFree(d16);
    }
  }
  return 0;
}
