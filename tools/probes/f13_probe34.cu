// f13: validation order/details for CompressMarkerLabelsUFBatch_32u.
#include "npp.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

static const int W = 10, H = 7;

int main() {
  std::vector<Npp32u> h((size_t)W * H * 2, 1);
  Npp32u *d = nullptr;
  cudaMalloc(&d, h.size() * 4);
  int gbs = 0;
  nppiCompressMarkerLabelsGetBufferSize_32u_C1R(W * H, &gbs);

  NppStreamContext ctx{};
  ctx.hStream = 0;

  auto run = [&](const char *name, bool adv, int batch, NppiSize roi, int bufSize, NppiSize img0,
                 NppiSize img1) {
    Npp8u *b0 = nullptr, *b1 = nullptr;
    if (bufSize > 0) { cudaMalloc(&b0, bufSize); cudaMalloc(&b1, bufSize); }
    NppiImageDescriptor im[2] = {{(char *)d, W * 4, img0},
                                 {(char *)d + (size_t)W * H * 4, W * 4, img1}};
    NppiBufferDescriptor bu[2] = {{b0, bufSize}, {b1, bufSize}};
    Npp32u nm[2] = {0, 0};
    cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
    NppStatus st = adv
        ? nppiCompressMarkerLabelsUFBatch_32u_C1IR_Advanced_Ctx(im, bu, nm, batch, roi, bufSize, ctx)
        : nppiCompressMarkerLabelsUFBatch_32u_C1IR_Ctx(im, bu, nm, batch, roi, bufSize, ctx);
    cudaError_t e = cudaDeviceSynchronize();
    printf("%-32s adv=%d st=%d cuda=%s nm=%u,%u\n", name, (int)adv, (int)st, cudaGetErrorName(e),
           nm[0], nm[1]);
    if (b0) cudaFree(b0);
    if (b1) cudaFree(b1);
  };

  for (bool adv : {false, true}) {
    run("batch0+roi-1", adv, 0, {-1, H}, gbs, {W, H}, {W, H});
    run("batch0+roi0", adv, 0, {0, H}, gbs, {W, H}, {W, H});
    run("roi-1+buf0", adv, 2, {-1, H}, 0, {W, H}, {W, H});
    run("roi0+buf0", adv, 2, {0, H}, 0, {W, H}, {W, H});
    run("roi0+batch1", adv, 1, {0, H}, gbs, {W, H}, {W, H});
    run("img0 roi 0", adv, 2, {W, H}, gbs, {0, H}, {W, H});
    run("img1 roi -1", adv, 2, {W, H}, gbs, {W, H}, {-1, H});
    run("img roi > max", adv, 2, {W - 2, H - 2}, gbs, {W, H}, {W, H});
    run("img roi < max", adv, 2, {W + 3, H + 3}, gbs, {W - 2, H - 2}, {W, H});
    run("img step 0", adv, 2, {W, H}, gbs, {W, H}, {W, H});  // handled separately below
    run("batch -1", adv, -1, {W, H}, gbs, {W, H}, {W, H});
    run("roi neg both", adv, 2, {-1, -1}, gbs, {W, H}, {W, H});
    run("buf 1", adv, 2, {W, H}, 1, {W, H}, {W, H});
  }

  // image step 0 / null data
  {
    Npp8u *b0 = nullptr; cudaMalloc(&b0, gbs);
    NppiImageDescriptor im[2] = {{(char *)d, 0, {W, H}}, {(char *)d + (size_t)W * H * 4, W * 4, {W, H}}};
    NppiBufferDescriptor bu[2] = {{b0, gbs}, {b0, gbs}};
    Npp32u nm[2] = {0, 0};
    NppStatus st = nppiCompressMarkerLabelsUFBatch_32u_C1IR_Ctx(im, bu, nm, 2, NppiSize{W, H}, gbs, ctx);
    printf("img0 step0 plain st=%d\n", (int)st);
    NppStatus st2 = nppiCompressMarkerLabelsUFBatch_32u_C1IR_Advanced_Ctx(im, bu, nm, 2, NppiSize{W, H}, gbs, ctx);
    printf("img0 step0 adv   st=%d\n", (int)st2);
    NppiImageDescriptor im2[2] = {{nullptr, W * 4, {W, H}}, {(char *)d + (size_t)W * H * 4, W * 4, {W, H}}};
    st = nppiCompressMarkerLabelsUFBatch_32u_C1IR_Ctx(im2, bu, nm, 2, NppiSize{W, H}, gbs, ctx);
    st2 = nppiCompressMarkerLabelsUFBatch_32u_C1IR_Advanced_Ctx(im2, bu, nm, 2, NppiSize{W, H}, gbs, ctx);
    printf("img0 null plain=%d adv=%d\n", (int)st, (int)st2);
    NppiBufferDescriptor bu2[2] = {{nullptr, gbs}, {b0, gbs}};
    NppiImageDescriptor im3[2] = {{(char *)d, W * 4, {W, H}}, {(char *)d + (size_t)W * H * 4, W * 4, {W, H}}};
    st = nppiCompressMarkerLabelsUFBatch_32u_C1IR_Ctx(im3, bu2, nm, 2, NppiSize{W, H}, gbs, ctx);
    st2 = nppiCompressMarkerLabelsUFBatch_32u_C1IR_Advanced_Ctx(im3, bu2, nm, 2, NppiSize{W, H}, gbs, ctx);
    printf("buf0 null plain=%d adv=%d\n", (int)st, (int)st2);
  }
  return 0;
}
