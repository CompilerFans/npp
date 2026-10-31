// f13 probe 20: UP per-tap sampling functions via unit-vector kernels on a ramp.
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
  const int W = 65, H = 5;
  std::vector<Npp32f> h((size_t)W * H);
  for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) h[y * W + x] = (float)x;
  Npp32f *d; cudaMalloc(&d, W * H * 4); cudaMemcpy(d, h.data(), W * H * 4, cudaMemcpyHostToDevice);
  struct K { const char *n; int t; Npp32f k[6]; };
  K ks[] = {
    {"e0", 3, {1, 0, 0}}, {"e1", 3, {0, 1, 0}}, {"e2", 3, {0, 0, 1}},
    {"u0", 2, {1, 0}}, {"u1", 2, {0, 1}},
    {"f0", 5, {1, 0, 0, 0, 0}}, {"f1", 5, {0, 1, 0, 0, 0}}, {"f2", 5, {0, 0, 1, 0, 0}},
    {"f3", 5, {0, 0, 0, 1, 0}}, {"f4", 5, {0, 0, 0, 0, 1}},
    {"t1", 1, {1}},
    {"h01", 3, {0.5f, 0, 0.5f}}, {"h10", 3, {0.5f, 0.5f, 0}},
  };
  float rates[] = {1.5f, 1.25f, 2.0f, 2.5f, 1.1f};
  Npp32f *o; cudaMalloc(&o, 1 << 20);
  int which = part;  // part = index into rates
  float r = rates[which];
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
    for (int x = 0; x < 18 && x < dst.width; ++x) printf(" %.6g", v[2 * dst.width + x]);
    printf("\n");
  }
  return 0;
}
