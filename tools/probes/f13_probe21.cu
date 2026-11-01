// f13 probe 21: UP pyramid - T7/T4/T6 tap structure, new-rate model predictions,
// MIRROR border, 8u/16u rounding + fraction quantization, kernel quantization.
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

static const char *es(NppStatus s) { static char b[32]; snprintf(b, 32, "%d", (int)s); return b; }
static NppStreamContext ctx;

static void up32(const char *tag, Npp32f *d, int W, int H, float r, int t, const Npp32f *k,
                 NppiBorderType bt, int nprint, int row) {
  NppiSize mn{}, mx{};
  nppiGetFilterGaussPyramidLayerUpBorderDstROI(W, H, &mn, &mx, r);
  NppiSize dst = mx;
  Npp32f *o; cudaMalloc(&o, 1 << 22); cudaMemset(o, 0, 1 << 22);
  NppStatus st = nppiFilterGaussPyramidLayerUpBorder_32f_C1R_Ctx(
      d, W * 4, NppiSize{W, H}, NppiPoint{0, 0}, o, dst.width * 4, dst, r, t, k, bt, ctx);
  std::vector<Npp32f> v((size_t)dst.width * dst.height);
  cudaMemcpy(v.data(), o, v.size() * 4, cudaMemcpyDeviceToHost);
  printf("%s st=%s w=%d row%d:", tag, es(st), dst.width, row);
  for (int x = 0; x < nprint && x < dst.width; ++x) printf(" %.7g", v[row * dst.width + x]);
  printf("\n");
  cudaFree(o);
}

int main(int argc, char **argv) {
  int part = argc > 1 ? atoi(argv[1]) : 0;
  ctx.hStream = 0;
  const int W = 129, H = 5;
  std::vector<Npp32f> h((size_t)W * H);
  for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) h[y * W + x] = (float)x;
  Npp32f *d; cudaMalloc(&d, W * H * 4); cudaMemcpy(d, h.data(), W * H * 4, cudaMemcpyHostToDevice);

  if (part == 1) {  // T7 + T6 + T4 unit taps at core rates
    Npp32f g0[7]={1,0,0,0,0,0,0}, g3[7]={0,0,0,1,0,0,0}, g4[7]={0,0,0,0,1,0,0},
           g5[7]={0,0,0,0,0,1,0}, g6[7]={0,0,0,0,0,0,1};
    Npp32f s0[6]={1,0,0,0,0,0}, s2[6]={0,0,1,0,0,0}, s3[6]={0,0,0,1,0,0},
           s4[6]={0,0,0,0,1,0}, s5[6]={0,0,0,0,0,1};
    Npp32f q1[4]={0,1,0,0}, q2[4]={0,0,1,0}, q3[4]={0,0,0,1};
    float rs[] = {1.5f, 1.25f, 2.5f, 1.1f, 2.0f};
    for (float r : rs) {
      char t[64];
      snprintf(t,64,"r=%g g0",r); up32(t,d,W,H,r,7,g0,NPP_BORDER_REPLICATE,20,2);
      snprintf(t,64,"r=%g g3",r); up32(t,d,W,H,r,7,g3,NPP_BORDER_REPLICATE,20,2);
      snprintf(t,64,"r=%g g4",r); up32(t,d,W,H,r,7,g4,NPP_BORDER_REPLICATE,20,2);
      snprintf(t,64,"r=%g g5",r); up32(t,d,W,H,r,7,g5,NPP_BORDER_REPLICATE,20,2);
      snprintf(t,64,"r=%g g6",r); up32(t,d,W,H,r,7,g6,NPP_BORDER_REPLICATE,20,2);
    }
    for (float r : rs) {
      char t[64];
      snprintf(t,64,"r=%g s0",r); up32(t,d,W,H,r,6,s0,NPP_BORDER_REPLICATE,20,2);
      snprintf(t,64,"r=%g s2",r); up32(t,d,W,H,r,6,s2,NPP_BORDER_REPLICATE,20,2);
      snprintf(t,64,"r=%g s3",r); up32(t,d,W,H,r,6,s3,NPP_BORDER_REPLICATE,20,2);
      snprintf(t,64,"r=%g s4",r); up32(t,d,W,H,r,6,s4,NPP_BORDER_REPLICATE,20,2);
      snprintf(t,64,"r=%g s5",r); up32(t,d,W,H,r,6,s5,NPP_BORDER_REPLICATE,20,2);
    }
    float rq[] = {1.25f, 2.5f, 1.1f};
    for (float r : rq) {
      char t[64];
      snprintf(t,64,"r=%g q1",r); up32(t,d,W,H,r,4,q1,NPP_BORDER_REPLICATE,20,2);
      snprintf(t,64,"r=%g q2",r); up32(t,d,W,H,r,4,q2,NPP_BORDER_REPLICATE,20,2);
      snprintf(t,64,"r=%g q3",r); up32(t,d,W,H,r,4,q3,NPP_BORDER_REPLICATE,20,2);
    }
  }

  if (part == 2) {  // new rates: model predictions for e1(center) e2(R1) f4(R2)
    Npp32f e1[3]={0,1,0}, e2[3]={0,0,1}, f4[5]={0,0,0,0,1}, f3[5]={0,0,0,1,0};
    float rs[] = {1.2f, 1.75f, 1.05f, 1.3f, 1.37f, 3.5f, 10.0f, 3.0f, 4.0f, 1.6f, 1.4f, 2.25f};
    for (float r : rs) {
      char t[64];
      snprintf(t,64,"r=%g e1",r); up32(t,d,W,H,r,3,e1,NPP_BORDER_REPLICATE,46,2);
      snprintf(t,64,"r=%g e2",r); up32(t,d,W,H,r,3,e2,NPP_BORDER_REPLICATE,46,2);
      snprintf(t,64,"r=%g f3",r); up32(t,d,W,H,r,5,f3,NPP_BORDER_REPLICATE,46,2);
      snprintf(t,64,"r=%g f4",r); up32(t,d,W,H,r,5,f4,NPP_BORDER_REPLICATE,46,2);
    }
  }

  if (part == 3) {  // MIRROR vs REPLICATE, small image, 3/5-tap kernels, full row dump
    const int W2 = 9, H2 = 9;
    std::vector<Npp32f> h2((size_t)W2 * H2);
    for (int y = 0; y < H2; ++y) for (int x = 0; x < W2; ++x) h2[y * W2 + x] = (float)(10 * y + x);
    Npp32f *d2; cudaMalloc(&d2, W2 * H2 * 4); cudaMemcpy(d2, h2.data(), W2 * H2 * 4, cudaMemcpyHostToDevice);
    Npp32f id3[3]={0,1,0}, k125[3]={0.25f,0.5f,0.25f}, k5[5]={0.0625f,0.25f,0.375f,0.25f,0.0625f};
    for (int bt = 0; bt < 2; ++bt) {
      NppiBorderType b = bt ? NPP_BORDER_MIRROR : NPP_BORDER_REPLICATE;
      const char *bn = bt ? "MIRROR" : "REPL  ";
      char t[64];
      snprintf(t,64,"%s id3 r1.5",bn); up32(t,d2,W2,H2,1.5f,3,id3,b,14,0);
      snprintf(t,64,"%s k125 r1.5",bn); up32(t,d2,W2,H2,1.5f,3,k125,b,14,0);
      snprintf(t,64,"%s k5 r1.5",bn); up32(t,d2,W2,H2,1.5f,5,k5,b,14,0);
      snprintf(t,64,"%s k5 r2.5",bn); up32(t,d2,W2,H2,2.5f,5,k5,b,23,0);
    }
    // DOWN mirror check too
    for (int bt = 0; bt < 2; ++bt) {
      NppiBorderType b = bt ? NPP_BORDER_MIRROR : NPP_BORDER_REPLICATE;
      const char *bn = bt ? "MIRROR" : "REPL  ";
      NppiSize mn{}, mx{};
      nppiGetFilterGaussPyramidLayerDownBorderDstROI(W2, H2, &mx, 1.5f);
      NppiSize dst = mx;
      Npp32f *o; cudaMalloc(&o, 4096); cudaMemset(o, 0, 4096);
      Npp32f id3[3]={0,1,0};
      NppStatus st = nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx(
          d2, W2 * 4, NppiSize{W2, H2}, NppiPoint{0, 0}, o, dst.width * 4, dst, 1.5f, 3, id3, b, ctx);
      std::vector<Npp32f> v((size_t)dst.width * dst.height);
      cudaMemcpy(v.data(), o, v.size() * 4, cudaMemcpyDeviceToHost);
      printf("down %s id3 r1.5 st=%s dst=%dx%d row0:", bn, es(st), dst.width, dst.height);
      for (int x = 0; x < dst.width; ++x) printf(" %.7g", v[x]);
      printf("\n");
      cudaFree(o);
    }
  }

  if (part == 4) {  // 8u/16u rounding + fraction quantization: 6x ramp sources
    { // 16u UP r=1.5, unit taps
      const int W3 = 64, H3 = 4;
      std::vector<Npp16u> h16((size_t)W3 * H3);
      for (int y = 0; y < H3; ++y) for (int x = 0; x < W3; ++x) h16[y * W3 + x] = (Npp16u)(6 * x);
      Npp16u *d16; cudaMalloc(&d16, W3 * H3 * 2); cudaMemcpy(d16, h16.data(), W3 * H3 * 2, cudaMemcpyHostToDevice);
      NppiSize mn{}, mx{}; nppiGetFilterGaussPyramidLayerUpBorderDstROI(W3, H3, &mn, &mx, 1.5f);
      NppiSize dst = mx;
      Npp16u *o; cudaMalloc(&o, 1 << 16); 
      Npp32f ks[][5] = {{0,1,0},{0,0,1},{0.25f,0.5f,0.25f},{0.0625f,0.25f,0.375f,0.25f,0.0625f},{0,0,0,0,1}};
      const char *kn[] = {"e1","e2","k125","k5","f4"};
      int kt[] = {3,3,3,5,5};
      for (int i = 0; i < 5; ++i) {
        cudaMemset(o, 0, 1 << 16);
        NppStatus st = nppiFilterGaussPyramidLayerUpBorder_16u_C1R_Ctx(
            d16, W3 * 2, NppiSize{W3, H3}, NppiPoint{0, 0}, o, dst.width * 2, dst, 1.5f, kt[i], ks[i],
            NPP_BORDER_REPLICATE, ctx);
        std::vector<Npp16u> v((size_t)dst.width * dst.height);
        cudaMemcpy(v.data(), o, v.size() * 2, cudaMemcpyDeviceToHost);
        printf("16uUP r=1.5 %s st=%s row2:", kn[i], es(st));
        for (int x = 0; x < 24 && x < dst.width; ++x) printf(" %d", (int)v[2 * dst.width + x]);
        printf("\n");
      }
      cudaFree(o); cudaFree(d16);
    }
    { // 8u UP r=1.5 (src 6x ramp, W small to stay <256)
      const int W3 = 40, H3 = 4;
      std::vector<Npp8u> h8((size_t)W3 * H3);
      for (int y = 0; y < H3; ++y) for (int x = 0; x < W3; ++x) h8[y * W3 + x] = (Npp8u)(6 * x);
      Npp8u *d8; cudaMalloc(&d8, W3 * H3); cudaMemcpy(d8, h8.data(), W3 * H3, cudaMemcpyHostToDevice);
      NppiSize mn{}, mx{}; nppiGetFilterGaussPyramidLayerUpBorderDstROI(W3, H3, &mn, &mx, 1.5f);
      NppiSize dst = mx;
      Npp8u *o; cudaMalloc(&o, 1 << 16);
      Npp32f ks[][5] = {{0,1,0},{0,0,1},{0.25f,0.5f,0.25f},{0,0,0,0,1}};
      const char *kn[] = {"e1","e2","k125","f4"};
      int kt[] = {3,3,3,5};
      for (int i = 0; i < 4; ++i) {
        cudaMemset(o, 0, 1 << 16);
        NppStatus st = nppiFilterGaussPyramidLayerUpBorder_8u_C1R_Ctx(
            d8, W3, NppiSize{W3, H3}, NppiPoint{0, 0}, o, dst.width, dst, 1.5f, kt[i], ks[i],
            NPP_BORDER_REPLICATE, ctx);
        std::vector<Npp8u> v((size_t)dst.width * dst.height);
        cudaMemcpy(v.data(), o, v.size(), cudaMemcpyDeviceToHost);
        printf("8uUP r=1.5 %s st=%s row2:", kn[i], es(st));
        for (int x = 0; x < 24 && x < dst.width; ++x) printf(" %d", (int)v[2 * dst.width + x]);
        printf("\n");
      }
      cudaFree(o); cudaFree(d8);
    }
    { // DOWN 16u rounding sharp: isolated pixels 10 and 14, r=2, k125
      const int W4 = 16, H4 = 16;
      std::vector<Npp16u> h16((size_t)W4 * H4, 0);
      h16[7 * W4 + 3] = 10;   // col 3 row 7 -> single-axis responses 2.5 / 3.5 combos
      h16[3 * W4 + 11] = 14;
      Npp16u *d16; cudaMalloc(&d16, W4 * H4 * 2); cudaMemcpy(d16, h16.data(), W4 * H4 * 2, cudaMemcpyHostToDevice);
      Npp16u *o; cudaMalloc(&o, 4096); cudaMemset(o, 0, 4096);
      Npp32f k125[3] = {0.25f, 0.5f, 0.25f};
      NppStatus st = nppiFilterGaussPyramidLayerDownBorder_16u_C1R_Ctx(
          d16, W4 * 2, NppiSize{W4, H4}, NppiPoint{0, 0}, o, 16, NppiSize{8, 8}, 2.0f, 3, k125,
          NPP_BORDER_REPLICATE, ctx);
      std::vector<Npp16u> v(64); cudaMemcpy(v.data(), o, 128, cudaMemcpyDeviceToHost);
      printf("16uDOWN r=2 k125 st=%s rows1-5:\n", es(st));
      for (int y = 1; y < 6; ++y) { printf("  "); for (int x = 0; x < 8; ++x) printf(" %d", (int)v[y * 8 + x]); printf("\n"); }
      cudaFree(o); cudaFree(d16);
    }
    { // kernel quantization: non-dyadic 1/3 kernel, 8u + 32f, DOWN r=2, src 6x ramp
      const int W5 = 32, H5 = 4;
      std::vector<Npp8u> h8((size_t)W5 * H5);
      std::vector<Npp32f> h32((size_t)W5 * H5);
      for (int y = 0; y < H5; ++y) for (int x = 0; x < W5; ++x) { h8[y * W5 + x] = (Npp8u)(6 * x); h32[y * W5 + x] = 6.0f * x; }
      Npp8u *d8; cudaMalloc(&d8, W5 * H5); cudaMemcpy(d8, h8.data(), W5 * H5, cudaMemcpyHostToDevice);
      Npp32f *d32; cudaMalloc(&d32, W5 * H5 * 4); cudaMemcpy(d32, h32.data(), W5 * H5 * 4, cudaMemcpyHostToDevice);
      Npp32f k3[3] = {1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f};
      Npp8u *o8; cudaMalloc(&o8, 4096); cudaMemset(o8, 0, 4096);
      Npp32f *o32; cudaMalloc(&o32, 4096); cudaMemset(o32, 0, 4096);
      nppiFilterGaussPyramidLayerDownBorder_8u_C1R_Ctx(d8, W5, NppiSize{W5, H5}, NppiPoint{0, 0}, o8, 32, NppiSize{16, 4}, 2.0f, 3, k3, NPP_BORDER_REPLICATE, ctx);
      nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx(d32, W5 * 4, NppiSize{W5, H5}, NppiPoint{0, 0}, o32, 64, NppiSize{16, 4}, 2.0f, 3, k3, NPP_BORDER_REPLICATE, ctx);
      std::vector<Npp8u> v8(64); cudaMemcpy(v8.data(), o8, 64, cudaMemcpyDeviceToHost);
      std::vector<Npp32f> v32(64); cudaMemcpy(v32.data(), o32, 256, cudaMemcpyDeviceToHost);
      printf("k=1/3 8uDOWN row1:"); for (int x = 0; x < 16; ++x) printf(" %d", (int)v8[16 + x]); printf("\n");
      printf("k=1/3 32fDOWN row1:"); for (int x = 0; x < 16; ++x) printf(" %.8g", v32[16 + x]); printf("\n");
      cudaFree(o8); cudaFree(o32); cudaFree(d8); cudaFree(d32);
    }
  }
  return 0;
}
