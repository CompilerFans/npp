// f13: pyramid-up footprint of single-tap kernels on a delta source.
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

static void run(float r, const float *k, int taps, const char *name, int px, int py, int W, int H) {
  std::vector<Npp32f> h((size_t)W * H, 0.0f);
  h[py * W + px] = 1.0f;
  Npp32f *d, *o;
  cudaMalloc(&d, (size_t)W * H * 4);
  cudaMalloc(&o, 1 << 22);
  cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
  cudaMemset(o, 0, 1 << 22);
  NppiSize mn{}, mx{};
  nppiGetFilterGaussPyramidLayerUpBorderDstROI(W, H, &mn, &mx, r);
  NppStreamContext c{};
  c.hStream = 0;
  NppStatus st = nppiFilterGaussPyramidLayerUpBorder_32f_C1R_Ctx(
      d, W * 4, {W, H}, {0, 0}, o, mx.width * 4, mx, r, taps, k, NPP_BORDER_REPLICATE, c);
  std::vector<Npp32f> v((size_t)mx.width * mx.height);
  cudaMemcpy(v.data(), o, v.size() * 4, cudaMemcpyDeviceToHost);
  printf("r=%-5g %-8s st=%d dst=%dx%d:", r, name, (int)st, mx.width, mx.height);
  for (size_t i = 0; i < v.size(); ++i)
    if (v[i] != 0.0f) printf(" (%d,%d)=%.9g", (int)(i % mx.width), (int)(i / mx.width), v[i]);
  printf("\n");
  cudaFree(d);
  cudaFree(o);
}

int main() {
  const float k1[3] = {0.0f, 1.0f, 0.0f};
  const float kl[3] = {1.0f, 0.0f, 0.0f};
  const float kr[3] = {0.0f, 0.0f, 1.0f};
  const float kh[3] = {0.5f, 0.0f, 0.5f};
  const int W = 33, H = 33;
  for (float r : {1.25f, 1.5f, 2.0f, 2.5f, 3.0f}) run(r, k1, 3, "k010", 8, 8, W, H);
  const float k2[3] = {0.0f, 2.0f, 0.0f};
  const float kh2[3] = {0.0f, 0.5f, 0.0f};
  const float k13[3] = {0.33333334f, 0.33333334f, 0.33333334f};
  run(1.25f, k2, 3, "k020", 8, 8, W, H);
  run(1.25f, kh2, 3, "k00.5", 8, 8, W, H);
  run(1.25f, k13, 3, "k13", 8, 8, W, H);
  run(1.05f, k1, 3, "k010", 8, 8, W, H);
  run(1.1f, k1, 3, "k010", 8, 8, W, H);
  for (float r : {1.25f, 2.5f}) {
    run(r, kl, 3, "k100", 8, 8, W, H);
    run(r, kr, 3, "k001", 8, 8, W, H);
    run(r, kh, 3, "k101", 8, 8, W, H);
  }
  return 0;
}
