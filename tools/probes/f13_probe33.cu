// f13: Advanced vs non-Advanced details for CompressMarkerLabelsUFBatch_32u.
#include "npp.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

static const int W = 10, H = 7;

int main() {
  std::vector<Npp32u> h((size_t)W * H * 2);
  for (int img = 0; img < 2; ++img)
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        h[img * W * H + y * W + x] = (Npp32u)(((x / 3) + (y / 2)) % 3 + 1 + 3 * img);

  Npp32u *d = nullptr;
  cudaMalloc(&d, h.size() * 4);
  cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice);

  int gbs = 0;
  NppStatus gst = nppiCompressMarkerLabelsGetBufferSize_32u_C1R(W * H, &gbs);
  printf("GetBufferSize(%d px) st=%d -> %d\n", W * H, (int)gst, gbs);

  struct Case { const char *name; int batch; int perBuf; NppiSize roi; bool advanced; };
  const int sizes[] = {0, 100, 65556, 65556 + 16, gbs};
  NppStreamContext ctx{};
  ctx.hStream = 0;

  auto run = [&](const char *name, int batch, int perBuf, NppiSize roi, bool advanced) {
    cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
    Npp8u *b0 = nullptr, *b1 = nullptr;
    if (perBuf > 0) { cudaMalloc(&b0, perBuf); cudaMalloc(&b1, perBuf); }
    NppiImageDescriptor im[2] = {{(char *)d, W * 4, {W, H}},
                                 {(char *)d + (size_t)W * H * 4, W * 4, {W, H}}};
    NppiBufferDescriptor bu[2] = {{b0, perBuf}, {b1, perBuf}};
    Npp32u nm[2] = {0xEEEEEEEEu, 0xEEEEEEEEu};
    NppStatus st = advanced
        ? nppiCompressMarkerLabelsUFBatch_32u_C1IR_Advanced_Ctx(im, bu, nm, batch, roi, perBuf, ctx)
        : nppiCompressMarkerLabelsUFBatch_32u_C1IR_Ctx(im, bu, nm, batch, roi, perBuf, ctx);
    cudaError_t e = cudaDeviceSynchronize();
    printf("%-26s adv=%d batch=%d buf=%d roi=%dx%d st=%d cuda=%s nm=%u,%u\n", name,
           (int)advanced, batch, perBuf, roi.width, roi.height, (int)st, cudaGetErrorName(e),
           nm[0], nm[1]);
    if (b0) cudaFree(b0);
    if (b1) cudaFree(b1);
  };

  for (bool adv : {false, true}) {
    run("batch0", 0, gbs, {W, H}, adv);
    run("batch1", 1, gbs, {W, H}, adv);
    run("batch2", 2, gbs, {W, H}, adv);
    run("roi0", 2, gbs, {0, H}, adv);
    run("roi-neg", 2, gbs, {-1, H}, adv);
    run("roi-bigger", 2, gbs, {W + 3, H + 2}, adv);
    for (int s : sizes) { char n[64]; snprintf(n, sizeof n, "buf=%d", s); run(n, 2, s, {W, H}, adv); }
  }

  // Device-side batch list / buffer list.
  {
    NppiImageDescriptor him[2] = {{(char *)d, W * 4, {W, H}},
                                  {(char *)d + (size_t)W * H * 4, W * 4, {W, H}}};
    Npp8u *b0 = nullptr, *b1 = nullptr;
    cudaMalloc(&b0, gbs);
    cudaMalloc(&b1, gbs);
    NppiBufferDescriptor hbu[2] = {{b0, gbs}, {b1, gbs}};
    NppiImageDescriptor *dim = nullptr;
    NppiBufferDescriptor *dbu = nullptr;
    Npp32u *dnm = nullptr;
    cudaMalloc(&dim, sizeof him);
    cudaMalloc(&dbu, sizeof hbu);
    cudaMalloc(&dnm, 2 * sizeof(Npp32u));
    cudaMemcpy(dim, him, sizeof him, cudaMemcpyHostToDevice);
    cudaMemcpy(dbu, hbu, sizeof hbu, cudaMemcpyHostToDevice);
    cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
    NppStatus st = nppiCompressMarkerLabelsUFBatch_32u_C1IR_Ctx(dim, dbu, dnm, 2, NppiSize{W, H}, gbs, ctx);
    NppStatus st2 = nppiCompressMarkerLabelsUFBatch_32u_C1IR_Advanced_Ctx(dim, dbu, dnm, 2, NppiSize{W, H}, gbs, ctx);
    cudaError_t e = cudaDeviceSynchronize();
    Npp32u nm[2] = {0, 0};
    cudaMemcpy(nm, dnm, sizeof nm, cudaMemcpyDeviceToHost);
    printf("device lists plain=%d adv=%d cuda=%s nm=%u,%u\n", (int)st, (int)st2, cudaGetErrorName(e), nm[0], nm[1]);
  }
  return 0;
}
