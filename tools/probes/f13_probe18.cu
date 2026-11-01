// f13 probe 18: pyramid helper sweeps, UP sampling geometry, offset, rounding.
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

static const char *es(NppStatus s) { static char b[32]; snprintf(b, 32, "%d", (int)s); return b; }

int main(int argc, char **argv) {
  int part = argc > 1 ? atoi(argv[1]) : 0;
  NppStreamContext c{}; c.hStream = 0;

  if (part == 1) {  // DOWN helper big sweep -> file p18_down.txt
    FILE *f = fopen("/tmp/p18_down.txt", "w");
    float rates[] = {1.05f,1.1f,1.15f,1.2f,1.25f,1.3f,1.35f,1.4f,1.45f,1.5f,1.55f,1.6f,1.7f,1.8f,1.9f,1.95f,2.0f,2.5f,3.0f,4.0f,6.0f,9.0f,10.0f};
    for (float r : rates) {
      fprintf(f, "r %.6g:", r);
      for (int w = 2; w <= 48; ++w) {
        NppiSize d{};
        nppiGetFilterGaussPyramidLayerDownBorderDstROI(w, 17, &d, r);
        fprintf(f, " %d:%d", w, d.width);
      }
      fprintf(f, "\n");
    }
    // H independence + negative/zero cases + rate validation
    for (int h : {0, 1, 2, 5, 17, 100}) {
      NppiSize d{-1,-1};
      NppStatus st = nppiGetFilterGaussPyramidLayerDownBorderDstROI(10, h, &d, 1.5f);
      printf("H=%d st=%s d=%dx%d\n", h, es(st), d.width, d.height);
    }
    for (int w : {-3, 0, 1, 2}) {
      NppiSize d{};
      NppStatus st = nppiGetFilterGaussPyramidLayerDownBorderDstROI(w, 10, &d, 1.5f);
      printf("W=%d st=%s d=%dx%d\n", w, es(st), d.width, d.height);
    }
    for (float r : {0.5f, 1.0f, 1.0001f, 10.0f, 10.1f, 20.0f, -1.5f}) {
      NppiSize d{};
      NppStatus st = nppiGetFilterGaussPyramidLayerDownBorderDstROI(10, 10, &d, r);
      printf("rate=%g st=%s d=%dx%d\n", r, es(st), d.width, d.height);
    }
    NppStatus st = nppiGetFilterGaussPyramidLayerDownBorderDstROI(10, 10, nullptr, 1.5f);
    printf("null out st=%s\n", es(st));
    fclose(f);
    printf("part1 done\n");
  }

  if (part == 2) {  // UP helper sweep -> file p18_up.txt
    FILE *f = fopen("/tmp/p18_up.txt", "w");
    float rates[] = {1.05f,1.1f,1.2f,1.25f,1.3f,1.4f,1.5f,1.6f,1.75f,1.9f,2.0f,2.4f,2.5f,3.0f,3.5f,4.0f,5.5f,7.5f,9.9f,10.0f};
    for (float r : rates) {
      fprintf(f, "r %.6g:", r);
      for (int w = 2; w <= 40; ++w) {
        NppiSize mn{}, mx{};
        nppiGetFilterGaussPyramidLayerUpBorderDstROI(w, 17, &mn, &mx, r);
        fprintf(f, " %d:%d/%d", w, mn.width, mx.width);
      }
      fprintf(f, "\n");
    }
    for (int w : {-3, 0, 1, 2}) {
      NppiSize mn{}, mx{};
      NppStatus st = nppiGetFilterGaussPyramidLayerUpBorderDstROI(w, 10, &mn, &mx, 1.5f);
      printf("W=%d st=%s min=%dx%d max=%dx%d\n", w, es(st), mn.width, mn.height, mx.width, mx.height);
    }
    for (float r : {0.5f, 1.0f, 10.0f, 10.1f, -1.5f}) {
      NppiSize mn{}, mx{};
      NppStatus st = nppiGetFilterGaussPyramidLayerUpBorderDstROI(10, 10, &mn, &mx, r);
      printf("rate=%g st=%s min=%dx%d max=%dx%d\n", r, es(st), mn.width, mn.height, mx.width, mx.height);
    }
    NppStatus st = nppiGetFilterGaussPyramidLayerUpBorderDstROI(10, 10, nullptr, nullptr, 1.5f);
    printf("null out st=%s\n", es(st));
    fclose(f);
    printf("part2 done\n");
  }

  if (part == 3) {  // UP identity sampling map, src 33x33 v = x + 100y, dump row0 and col0
    const int W = 33, H = 33;
    std::vector<Npp32f> h((size_t)W * H);
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) h[y * W + x] = (float)(x + 100 * y);
    Npp32f *d; cudaMalloc(&d, W * H * 4); cudaMemcpy(d, h.data(), W * H * 4, cudaMemcpyHostToDevice);
    Npp32f kern[3] = {0.0f, 1.0f, 0.0f};
    Npp32f *o; cudaMalloc(&o, 1 << 20); cudaMemset(o, 0, 1 << 20);
    float rates[] = {1.05f,1.1f,1.2f,1.25f,1.3f,1.4f,1.5f,1.6f,1.75f,2.0f,2.5f,3.0f,3.5f,4.0f};
    for (float r : rates) {
      NppiSize mn{}, mx{};
      nppiGetFilterGaussPyramidLayerUpBorderDstROI(W, H, &mn, &mx, r);
      NppiSize dst = mx;
      cudaMemset(o, 0, 1 << 20);
      NppStatus st = nppiFilterGaussPyramidLayerUpBorder_32f_C1R_Ctx(
          d, W * 4, NppiSize{W, H}, NppiPoint{0, 0}, o, dst.width * 4, dst, r, 3, kern,
          NPP_BORDER_REPLICATE, c);
      std::vector<Npp32f> v((size_t)dst.width * dst.height);
      cudaMemcpy(v.data(), o, v.size() * 4, cudaMemcpyDeviceToHost);
      printf("r=%g st=%s dst=%dx%d\n  row0:", r, es(st), dst.width, dst.height);
      for (int x = 0; x < dst.width; ++x) printf(" %.6g", v[x]);
      printf("\n  col0:");
      for (int y = 0; y < dst.height; ++y) printf(" %.6g", v[y * dst.width]);
      printf("\n");
    }
  }

  if (part == 4) {  // UP non-identity kernel delta: k=[.25,.5,.25], delta at (8,8), src 33x33
    const int W = 33, H = 33;
    std::vector<Npp32f> h((size_t)W * H, 0.0f);
    h[8 * W + 8] = 1.0f;
    Npp32f *d; cudaMalloc(&d, W * H * 4); cudaMemcpy(d, h.data(), W * H * 4, cudaMemcpyHostToDevice);
    Npp32f kern[3] = {0.25f, 0.5f, 0.25f};
    Npp32f *o; cudaMalloc(&o, 1 << 20); cudaMemset(o, 0, 1 << 20);
    float rates[] = {1.5f, 1.25f, 2.5f, 3.0f, 4.0f};
    for (float r : rates) {
      NppiSize mn{}, mx{};
      nppiGetFilterGaussPyramidLayerUpBorderDstROI(W, H, &mn, &mx, r);
      NppiSize dst = mx;
      cudaMemset(o, 0, 1 << 20);
      NppStatus st = nppiFilterGaussPyramidLayerUpBorder_32f_C1R_Ctx(
          d, W * 4, NppiSize{W, H}, NppiPoint{0, 0}, o, dst.width * 4, dst, r, 3, kern,
          NPP_BORDER_REPLICATE, c);
      std::vector<Npp32f> v((size_t)dst.width * dst.height);
      cudaMemcpy(v.data(), o, v.size() * 4, cudaMemcpyDeviceToHost);
      printf("up k125 r=%g delta(8,8) st=%s dst=%dx%d:", r, es(st), dst.width, dst.height);
      int n = 0;
      for (size_t i = 0; i < v.size(); ++i)
        if (v[i] != 0.0f) { if (++n <= 30) printf(" (%d,%d)=%.6g", (int)(i % dst.width), (int)(i / dst.width), v[i]); }
      printf(" [%d nz]\n", n);
    }
  }

  if (part == 5) {  // UP integer rate with non-identity kernel -> duplication?
    const int W = 16, H = 16;
    std::vector<Npp32f> h((size_t)W * H, 0.0f);
    h[5 * W + 5] = 1.0f;
    Npp32f *d; cudaMalloc(&d, W * H * 4); cudaMemcpy(d, h.data(), W * H * 4, cudaMemcpyHostToDevice);
    Npp32f kern[3] = {1.0f / 3, 1.0f / 3, 1.0f / 3};
    Npp32f *o; cudaMalloc(&o, 1 << 18); cudaMemset(o, 0, 1 << 18);
    for (float r : {3.0f, 4.0f}) {
      NppiSize mn{}, mx{};
      nppiGetFilterGaussPyramidLayerUpBorderDstROI(W, H, &mn, &mx, r);
      NppiSize dst = mx;
      cudaMemset(o, 0, 1 << 18);
      NppStatus st = nppiFilterGaussPyramidLayerUpBorder_32f_C1R_Ctx(
          d, W * 4, NppiSize{W, H}, NppiPoint{0, 0}, o, dst.width * 4, dst, r, 3, kern,
          NPP_BORDER_REPLICATE, c);
      std::vector<Npp32f> v((size_t)dst.width * dst.height);
      cudaMemcpy(v.data(), o, v.size() * 4, cudaMemcpyDeviceToHost);
      printf("up k13 r=%g delta(5,5) st=%s dst=%dx%d:", r, es(st), dst.width, dst.height);
      for (size_t i = 0; i < v.size(); ++i)
        if (v[i] != 0.0f) printf(" (%d,%d)=%.6g", (int)(i % dst.width), (int)(i / dst.width), v[i]);
      printf("\n");
    }
    // 5-tap kernel with integer rate: still duplication?
    Npp32f kern5[5] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};
    NppiSize mn{}, mx{};
    nppiGetFilterGaussPyramidLayerUpBorderDstROI(W, H, &mn, &mx, 2.0f);
    NppiSize dst = mx;
    cudaMemset(o, 0, 1 << 18);
    NppStatus st = nppiFilterGaussPyramidLayerUpBorder_32f_C1R_Ctx(
        d, W * 4, NppiSize{W, H}, NppiPoint{0, 0}, o, dst.width * 4, dst, 2.0f, 5, kern5,
        NPP_BORDER_REPLICATE, c);
    std::vector<Npp32f> v((size_t)dst.width * dst.height);
    cudaMemcpy(v.data(), o, v.size() * 4, cudaMemcpyDeviceToHost);
    printf("up k5 r=2 delta(5,5) st=%s:", es(st));
    for (size_t i = 0; i < v.size(); ++i)
      if (v[i] != 0.0f) printf(" (%d,%d)=%.6g", (int)(i % dst.width), (int)(i / dst.width), v[i]);
    printf("\n");
  }

  if (part == 6) {  // DOWN offset semantics: src 16x16 v=x+100y, offset (3,4), taps 3 identity
    const int W = 16, H = 16;
    std::vector<Npp32f> h((size_t)W * H);
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) h[y * W + x] = (float)(x + 100 * y);
    Npp32f *d; cudaMalloc(&d, W * H * 4); cudaMemcpy(d, h.data(), W * H * 4, cudaMemcpyHostToDevice);
    Npp32f kern[3] = {0.0f, 1.0f, 0.0f};
    Npp32f *o; cudaMalloc(&o, 65536); cudaMemset(o, 0, 65536);
    // offset (3,4): pSrc = base + 4*step + 3; ROI 6x5 at that offset
    NppStatus st = nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx(
        d + 4 * W + 3, W * 4, NppiSize{W, H}, NppiPoint{3, 4}, o, 6 * 4, NppiSize{6, 5},
        1.5f, 3, kern, NPP_BORDER_REPLICATE, c);
    std::vector<Npp32f> v(6 * 5);
    cudaMemcpy(v.data(), o, 6 * 5 * 4, cudaMemcpyDeviceToHost);
    printf("down offset(3,4) roi 6x5 r=1.5 st=%s\n", es(st));
    for (int y = 0; y < 5; ++y) {
      printf("  ");
      for (int x = 0; x < 6; ++x) printf(" %7.3f", v[y * 6 + x]);
      printf("\n");
    }
    // same for UP
    NppiSize mn{}, mx{};
    nppiGetFilterGaussPyramidLayerUpBorderDstROI(6, 5, &mn, &mx, 1.5f);
    cudaMemset(o, 0, 65536);
    st = nppiFilterGaussPyramidLayerUpBorder_32f_C1R_Ctx(
        d + 4 * W + 3, W * 4, NppiSize{W, H}, NppiPoint{3, 4}, o, mx.width * 4, mx,
        1.5f, 3, kern, NPP_BORDER_REPLICATE, c);
    std::vector<Npp32f> u((size_t)mx.width * mx.height);
    cudaMemcpy(u.data(), o, u.size() * 4, cudaMemcpyDeviceToHost);
    printf("up offset(3,4) r=1.5 st=%s dst=%dx%d\n", es(st), mx.width, mx.height);
    for (int y = 0; y < 4; ++y) {
      printf("  ");
      for (int x = 0; x < 8; ++x) printf(" %8.3f", u[y * mx.width + x]);
      printf("\n");
    }
  }

  if (part == 7) {  // DOWN 8u/16u rounding: quarters
    const int W = 16, H = 16;
    // 8u: a(i)=(i==3), b(j)=(j==7): dst values in {0,.25,.5,.75}
    std::vector<Npp8u> h8((size_t)W * H, 0);
    std::vector<Npp16u> h16((size_t)W * H, 0);
    for (int j = 0; j < H; ++j) { h8[j * W + 3] = 1; h8[7 * W + j] += 1; }
    for (int j = 0; j < H; ++j) { h16[j * W + 3] = 3; h16[7 * W + j] += 3; }
    Npp8u *d8; cudaMalloc(&d8, W * H); cudaMemcpy(d8, h8.data(), W * H, cudaMemcpyHostToDevice);
    Npp16u *d16; cudaMalloc(&d16, W * H * 2); cudaMemcpy(d16, h16.data(), W * H * 2, cudaMemcpyHostToDevice);
    Npp32f kern[3] = {0.25f, 0.5f, 0.25f};
    Npp8u *o8; cudaMalloc(&o8, 4096); cudaMemset(o8, 0, 4096);
    Npp16u *o16; cudaMalloc(&o16, 8192); cudaMemset(o16, 0, 8192);
    nppiFilterGaussPyramidLayerDownBorder_8u_C1R_Ctx(
        d8, W, NppiSize{W, H}, NppiPoint{0, 0}, o8, 8, NppiSize{8, 8}, 2.0f, 3, kern,
        NPP_BORDER_REPLICATE, c);
    nppiFilterGaussPyramidLayerDownBorder_16u_C1R_Ctx(
        d16, W * 2, NppiSize{W, H}, NppiPoint{0, 0}, o16, 16, NppiSize{8, 8}, 2.0f, 3, kern,
        NPP_BORDER_REPLICATE, c);
    std::vector<Npp8u> v8(64); cudaMemcpy(v8.data(), o8, 64, cudaMemcpyDeviceToHost);
    std::vector<Npp16u> v16(64); cudaMemcpy(v16.data(), o16, 128, cudaMemcpyDeviceToHost);
    printf("8u rows 0-3:\n");
    for (int y = 0; y < 4; ++y) { printf("  "); for (int x = 0; x < 8; ++x) printf(" %d", (int)v8[y * 8 + x]); printf("\n"); }
    printf("16u rows 2-5:\n");
    for (int y = 2; y < 6; ++y) { printf("  "); for (int x = 0; x < 8; ++x) printf(" %d", (int)v16[y * 8 + x]); printf("\n"); }
  }
  return 0;
}
