// f13 probe 17: pyramid down DstROI sweep + even taps + 8u rounding + pyramid Up.
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const char *es(NppStatus s) { static char b[32]; snprintf(b, 32, "%d", (int)s); return b; }

int main(int argc, char **argv) {
  int part = argc > 1 ? atoi(argv[1]) : 0;

  if (part == 1) {  // DstROI down sweep: W x rate -> dstW
    float rates[] = {1.1f, 1.2f, 1.25f, 1.3f, 1.5f, 1.75f, 2.0f, 2.4f, 2.5f, 3.0f, 3.5f, 4.0f, 7.0f};
    for (float r : rates) {
      printf("rate %.6g:", r);
      for (int w = 4; w <= 24; ++w) {
        NppiSize d{};
        nppiGetFilterGaussPyramidLayerDownBorderDstROI(w, 9, &d, r);
        printf(" %d:%d", w, d.width);
      }
      printf("\n");
    }
  }

  if (part == 2) {  // even taps: 4-tap controlled kernel, delta at (2,2), rate 2
    const float k4[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    std::vector<Npp32f> h(14 * 14, 0.0f);
    h[2 * 14 + 2] = 1.0f;
    Npp32f *d; cudaMalloc(&d, 14 * 14 * 4); cudaMemcpy(d, h.data(), 14 * 14 * 4, cudaMemcpyHostToDevice);
    Npp32f *o; cudaMalloc(&o, 8192); cudaMemset(o, 0, 8192);
    NppiSize dst{}; nppiGetFilterGaussPyramidLayerDownBorderDstROI(14, 14, &dst, 2.0f);
    NppStreamContext c{}; c.hStream = 0;
    NppStatus st = nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx(
        d, 56, NppiSize{14, 14}, NppiPoint{0, 0}, o, dst.width * 4, dst, 2.0f, 4, k4,
        NPP_BORDER_REPLICATE, c);
    printf("k4 r2 delta(2,2) st=%s dst=%dx%d:", es(st), dst.width, dst.height);
    std::vector<Npp32f> v((size_t)dst.width * dst.height);
    cudaMemcpy(v.data(), o, v.size() * 4, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < v.size(); ++i)
      if (v[i] != 0.0f) printf(" (%d,%d)=%.6g", (int)(i % dst.width), (int)(i / dst.width), v[i]);
    printf("\n");
  }

  if (part == 3) {  // 8u/16u rounding: kernel that yields k+0.5 / k+0.25
    // src 8x8 constant 10, kernel [0.05, 0.9, 0.05]: interior = 10; use deltas:
    // make dst exactly 10.5: src0=10 src1=11 -> weights (0.05*10 + 0.9*10 + 0.05*11)
    const int W = 8, H = 8;
    std::vector<Npp8u> h8((size_t)W * H, 10);
    std::vector<Npp16u> h16((size_t)W * H, 1000);
    for (int y = 0; y < H; ++y) { h8[y * W + 1] = 11; h16[y * W + 1] = 1001; }
    Npp8u *d8; cudaMalloc(&d8, 4096); cudaMemcpy(d8, h8.data(), W * H, cudaMemcpyHostToDevice);
    Npp16u *d16; cudaMalloc(&d16, 8192); cudaMemcpy(d16, h16.data(), W * H * 2, cudaMemcpyHostToDevice);
    Npp32f kern[3] = {0.05f, 0.9f, 0.05f};
    Npp8u *o8; cudaMalloc(&o8, 4096);
    Npp16u *o16; cudaMalloc(&o16, 8192);
    NppiSize dst{}; nppiGetFilterGaussPyramidLayerDownBorderDstROI(W, H, &dst, 2.0f);
    NppStreamContext c{}; c.hStream = 0;
    // horizontal sum at interior dst: 0.05*10+0.9*10+0.05*11 = 10.05; vertical same
    nppiFilterGaussPyramidLayerDownBorder_8u_C1R_Ctx(
        d8, W, NppiSize{W, H}, NppiPoint{0, 0}, o8, dst.width, dst, 2.0f, 3, kern,
        NPP_BORDER_REPLICATE, c);
    nppiFilterGaussPyramidLayerDownBorder_16u_C1R_Ctx(
        d16, W * 2, NppiSize{W, H}, NppiPoint{0, 0}, o16, dst.width * 2, dst, 2.0f, 3, kern,
        NPP_BORDER_REPLICATE, c);
    std::vector<Npp8u> v8((size_t)dst.width * dst.height);
    std::vector<Npp16u> v16((size_t)dst.width * dst.height);
    cudaMemcpy(v8.data(), o8, v8.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(v16.data(), o16, v16.size() * 2, cudaMemcpyDeviceToHost);
    printf("8u row0:"); for (auto x : v8) printf(" %d", (int)x);
    printf("\n16u row0:"); for (auto x : v16) printf(" %d", (int)x);
    printf("\n");
    // half-way: kernel [0.25,0.5,0.25] on 10/11 stripes
    Npp32f kern2[3] = {0.25f, 0.5f, 0.25f};
    nppiFilterGaussPyramidLayerDownBorder_8u_C1R_Ctx(
        d8, W, NppiSize{W, H}, NppiPoint{0, 0}, o8, dst.width, dst, 2.0f, 3, kern2,
        NPP_BORDER_REPLICATE, c);
    cudaMemcpy(v8.data(), o8, v8.size(), cudaMemcpyDeviceToHost);
    printf("8u k225 row0:"); for (auto x : v8) printf(" %d", (int)x);
    printf("\n");
  }

  if (part == 4) {  // pyramid UP: identity kernel, rate 2, src v=x+100y
    const int W = 8, H = 8;
    std::vector<Npp32f> h((size_t)W * H);
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) h[y * W + x] = (float)(x + 100 * y);
    Npp32f *d; cudaMalloc(&d, 4096); cudaMemcpy(d, h.data(), W * H * 4, cudaMemcpyHostToDevice);
    Npp32f kern[3] = {0.0f, 1.0f, 0.0f};
    Npp32f *o; cudaMalloc(&o, 16384); cudaMemset(o, 0xEE, 16384);
    NppStreamContext c{}; c.hStream = 0;
    float rates[] = {2.0f, 1.5f, 3.0f, 1.25f};
    for (float r : rates) {
      NppiSize mn{}, mx{};
      NppStatus sth = nppiGetFilterGaussPyramidLayerUpBorderDstROI(W, H, &mn, &mx, r);
      printf("up r=%g helper st=%s min=%dx%d max=%dx%d\n", r, es(sth), mn.width, mn.height, mx.width, mx.height);
      NppiSize dst = mx;
      cudaMemset(o, 0xEE, 16384);
      NppStatus st = nppiFilterGaussPyramidLayerUpBorder_32f_C1R_Ctx(
          d, W * 4, NppiSize{W, H}, NppiPoint{0, 0}, o, dst.width * 4, dst, r, 3, kern,
          NPP_BORDER_REPLICATE, c);
      printf("  up st=%s dst=%dx%d:", es(st), dst.width, dst.height);
      if (st == 0) {
        std::vector<Npp32f> v((size_t)dst.width * dst.height);
        cudaMemcpy(v.data(), o, v.size() * 4, cudaMemcpyDeviceToHost);
        for (int y = 0; y < std::min(dst.height, 3); ++y) {
          printf(" |");
          for (int x = 0; x < std::min(dst.width, 8); ++x) printf(" %.5g", v[y * dst.width + x]);
        }
      }
      printf("\n");
    }
  }

  if (part == 5) {  // pyramid UP delta: 3-tap equal kernel rate 2, delta at (2,2), 8x8
    const int W = 8, H = 8;
    std::vector<Npp32f> h((size_t)W * H, 0.0f);
    h[2 * W + 2] = 1.0f;
    Npp32f *d; cudaMalloc(&d, 4096); cudaMemcpy(d, h.data(), W * H * 4, cudaMemcpyHostToDevice);
    Npp32f kern[3] = {1.0f / 3, 1.0f / 3, 1.0f / 3};
    Npp32f *o; cudaMalloc(&o, 16384); cudaMemset(o, 0, 16384);
    NppStreamContext c{}; c.hStream = 0;
    NppiSize mn{}, mx{}; nppiGetFilterGaussPyramidLayerUpBorderDstROI(W, H, &mn, &mx, 2.0f);
    NppiSize dst = mx;
    NppStatus st = nppiFilterGaussPyramidLayerUpBorder_32f_C1R_Ctx(
        d, W * 4, NppiSize{W, H}, NppiPoint{0, 0}, o, dst.width * 4, dst, 2.0f, 3, kern,
        NPP_BORDER_REPLICATE, c);
    printf("up k3 r2 delta(2,2) st=%s dst=%dx%d:", es(st), dst.width, dst.height);
    std::vector<Npp32f> v((size_t)dst.width * dst.height);
    cudaMemcpy(v.data(), o, v.size() * 4, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < v.size(); ++i)
      if (v[i] != 0.0f) printf(" (%d,%d)=%.6g", (int)(i % dst.width), (int)(i / dst.width), v[i]);
    printf("\n");
  }
  return 0;
}
