// f13 probe 28: HarrisCornersBorder border behaviour on a small image.
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  const int W = 6, H = 6;
  std::vector<Npp8u> h((size_t)W * H);
  for (int i = 0; i < W * H; ++i) h[i] = (Npp8u)((i * 37 + 11) % 97);
  printf("src");
  for (int i = 0; i < W * H; ++i) printf(" %d", (int)h[i]);
  printf("\n");
  Npp8u *d;
  Npp32f *o;
  Npp8u *buf;
  cudaMalloc(&d, W * H);
  cudaMalloc(&o, W * H * 4);
  cudaMalloc(&buf, 1 << 16);
  cudaMemcpy(d, h.data(), W * H, cudaMemcpyHostToDevice);
  NppStreamContext c{};
  c.hStream = 0;
  struct C { const char *n; NppiDifferentialKernel f; NppiMaskSize m, a; float k, sc; };
  C cs[] = {{"sob3av3", NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_3_X_3, 0.04f, 1.f},
            {"sob3av5", NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_5_X_5, 0.04f, 1.f},
            {"sob5av3", NPP_FILTER_SOBEL, NPP_MASK_SIZE_5_X_5, NPP_MASK_SIZE_3_X_3, 0.04f, 1.f},
            {"sch3av3", NPP_FILTER_SCHARR, NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_3_X_3, 0.04f, 1.f}};
  for (auto &x : cs) {
    cudaMemset(o, 0, W * H * 4);
    NppStatus st = nppiFilterHarrisCornersBorder_8u32f_C1R_Ctx(d, W, {W, H}, {0, 0}, o, W * 4,
                                                               {W, H}, x.f, x.m, x.a, x.k, x.sc,
                                                               NPP_BORDER_REPLICATE, buf, c);
    std::vector<Npp32f> v(W * H);
    cudaMemcpy(v.data(), o, W * H * 4, cudaMemcpyDeviceToHost);
    printf("%s st=%d:", x.n, (int)st);
    for (int i = 0; i < W * H; ++i) printf(" %.9g", v[i]);
    printf("\n");
  }
  cudaFree(d);
  cudaFree(o);
  cudaFree(buf);
  return 0;
}
