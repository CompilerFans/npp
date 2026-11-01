// f13 probe 19: UP kernel-dependence of sampling positions (ramp probes).
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

  if (part == 1) {  // UP ramp v=x, various symmetric kernels -> dst = pos(t) (if kernel sum=1)
    const int W = 65, H = 5;
    std::vector<Npp32f> h((size_t)W * H);
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) h[y * W + x] = (float)x;
    Npp32f *d; cudaMalloc(&d, W * H * 4); cudaMemcpy(d, h.data(), W * H * 4, cudaMemcpyHostToDevice);
    struct K { const char *n; int t; Npp32f k[5]; };
    K ks[] = {{"id3", 3, {0, 1, 0}}, {"k125", 3, {0.25f, 0.5f, 0.25f}},
              {"k5", 5, {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f}},
              {"k111", 3, {1.f, 1.f, 1.f}}};
    float rates[] = {1.25f, 1.5f, 2.0f, 2.5f};
    Npp32f *o; cudaMalloc(&o, 1 << 20);
    for (float r : rates) {
      NppiSize mn{}, mx{};
      nppiGetFilterGaussPyramidLayerUpBorderDstROI(W, H, &mn, &mx, r);
      NppiSize dst = mx;
      for (K &k : ks) {
        cudaMemset(o, 0, 1 << 20);
        NppStatus st = nppiFilterGaussPyramidLayerUpBorder_32f_C1R_Ctx(
            d, W * 4, NppiSize{W, H}, NppiPoint{0, 0}, o, dst.width * 4, dst, r, k.t, k.k,
            NPP_BORDER_REPLICATE, c);
        std::vector<Npp32f> v((size_t)dst.width * dst.height);
        cudaMemcpy(v.data(), o, v.size() * 4, cudaMemcpyDeviceToHost);
        printf("r=%g %s st=%s dst=%dx%d row2:", r, k.n, es(st), dst.width, dst.height);
        for (int x = 0; x < 30 && x < dst.width; ++x) printf(" %.6g", v[2 * dst.width + x]);
        printf("\n");
      }
    }
  }

  if (part == 2) {  // UP ramp with ASYMMETRIC kernels -> reveals tap spacing s (dst = pos + s*m)
    const int W = 65, H = 5;
    std::vector<Npp32f> h((size_t)W * H);
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) h[y * W + x] = (float)x;
    Npp32f *d; cudaMalloc(&d, W * H * 4); cudaMemcpy(d, h.data(), W * H * 4, cudaMemcpyHostToDevice);
    struct K { const char *n; int t; Npp32f k[5]; };
    K ks[] = {{"id3", 3, {0, 1, 0}}, {"ka3", 3, {0.6f, 0.3f, 0.1f}},  // m=-0.5
              {"ka3r", 3, {0.1f, 0.3f, 0.6f}},                          // m=+0.5
              {"kb3", 3, {0.5f, 0.5f, 0.0f}}, {"kb3r", 3, {0.0f, 0.5f, 0.5f}},
              {"ka4", 4, {0.4f, 0.3f, 0.2f, 0.1f}},                     // c=1, m=0
              {"ka4b", 4, {0.1f, 0.2f, 0.3f, 0.4f}}};                   // c=1, m=0
    float rates[] = {1.5f, 2.0f};
    Npp32f *o; cudaMalloc(&o, 1 << 20);
    for (float r : rates) {
      NppiSize mn{}, mx{};
      nppiGetFilterGaussPyramidLayerUpBorderDstROI(W, H, &mn, &mx, r);
      NppiSize dst = mx;
      for (K &k : ks) {
        cudaMemset(o, 0, 1 << 20);
        NppStatus st = nppiFilterGaussPyramidLayerUpBorder_32f_C1R_Ctx(
            d, W * 4, NppiSize{W, H}, NppiPoint{0, 0}, o, dst.width * 4, dst, r, k.t, k.k,
            NPP_BORDER_REPLICATE, c);
        std::vector<Npp32f> v((size_t)dst.width * dst.height);
        cudaMemcpy(v.data(), o, v.size() * 4, cudaMemcpyDeviceToHost);
        printf("r=%g %s st=%s row2:", r, k.n, es(st));
        for (int x = 0; x < 20 && x < dst.width; ++x) printf(" %.6g", v[2 * dst.width + x]);
        printf("\n");
      }
    }
  }

  if (part == 3) {  // UP helper W=1 / H=1 anomaly
    int ws[] = {1, 2, 3};
    float rates[] = {1.25f, 1.5f, 2.0f, 2.5f, 3.0f};
    for (int W : ws) for (int H : ws) for (float r : rates) {
      NppiSize mn{}, mx{};
      NppStatus st = nppiGetFilterGaussPyramidLayerUpBorderDstROI(W, H, &mn, &mx, r);
      printf("W=%d H=%d r=%g st=%s min=%dx%d max=%dx%d\n", W, H, r, es(st), mn.width, mn.height,
             mx.width, mx.height);
    }
  }

  if (part == 4) {  // UP k125 r=1.5 delta(8,8) full row 8 dump
    const int W = 33, H = 33;
    std::vector<Npp32f> h((size_t)W * H, 0.0f);
    h[8 * W + 8] = 1.0f;
    Npp32f *d; cudaMalloc(&d, W * H * 4); cudaMemcpy(d, h.data(), W * H * 4, cudaMemcpyHostToDevice);
    Npp32f kern[3] = {0.25f, 0.5f, 0.25f};
    Npp32f *o; cudaMalloc(&o, 1 << 20); cudaMemset(o, 0, 1 << 20);
    float rates[] = {1.5f, 1.25f};
    for (float r : rates) {
      NppiSize mn{}, mx{};
      nppiGetFilterGaussPyramidLayerUpBorderDstROI(W, H, &mn, &mx, r);
      NppiSize dst = mx;
      cudaMemset(o, 0, 1 << 20);
      nppiFilterGaussPyramidLayerUpBorder_32f_C1R_Ctx(d, W * 4, NppiSize{W, H}, NppiPoint{0, 0},
          o, dst.width * 4, dst, r, 3, kern, NPP_BORDER_REPLICATE, c);
      std::vector<Npp32f> v((size_t)dst.width * dst.height);
      cudaMemcpy(v.data(), o, v.size() * 4, cudaMemcpyDeviceToHost);
      printf("up k125 r=%g delta(8,8) row8 t=6..20:", r);
      for (int t = 6; t <= 20; ++t) printf(" %.6g", v[8 * dst.width + t]);
      printf("\n");
    }
  }

  if (part == 5) {  // DOWN ramp with asymmetric kernel: dst = rate*x + m (m= sum k[i]*(i-c))
    const int W = 65, H = 5;
    std::vector<Npp32f> h((size_t)W * H);
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) h[y * W + x] = (float)x;
    Npp32f *d; cudaMalloc(&d, W * H * 4); cudaMemcpy(d, h.data(), W * H * 4, cudaMemcpyHostToDevice);
    struct K { const char *n; int t; Npp32f k[5]; };
    K ks[] = {{"id3", 3, {0, 1, 0}}, {"ka3", 3, {0.6f, 0.3f, 0.1f}}, {"ka4", 4, {0.4f, 0.3f, 0.2f, 0.1f}}};
    Npp32f *o; cudaMalloc(&o, 1 << 20);
    for (float r : {1.5f}) {
      NppiSize dst; dst.width = (int)((W - 1) / r) + 1; dst.height = (int)((H - 1) / r) + 1;
      for (K &k : ks) {
        cudaMemset(o, 0, 1 << 20);
        NppStatus st = nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx(
            d, W * 4, NppiSize{W, H}, NppiPoint{0, 0}, o, dst.width * 4, dst, r, k.t, k.k,
            NPP_BORDER_REPLICATE, c);
        std::vector<Npp32f> v((size_t)dst.width * dst.height);
        cudaMemcpy(v.data(), o, v.size() * 4, cudaMemcpyDeviceToHost);
        printf("down r=%g %s st=%s dst=%dx%d row2:", r, k.n, es(st), dst.width, dst.height);
        for (int x = 0; x < 12; ++x) printf(" %.6g", v[2 * dst.width + x]);
        printf("\n");
      }
    }
  }
  return 0;
}
