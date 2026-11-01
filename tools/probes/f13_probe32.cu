// f13: semantics of nppiCompressMarkerLabelsUFBatch_32u_C1IR (non-Advanced).
// Goal: confirm it forwards to the Advanced form with oSizeROI as the largest
// ROI and nPerImageBufferSize as the largest per-image buffer, and pin down the
// validation order/status codes.
#include "npp.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const int W = 10, H = 7;
static const int kFixed = 65556;  // measured fixed overhead of the UF buffer

static void fill_labels(std::vector<Npp32u> &v) {
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) v[y * W + x] = (Npp32u)(((x / 3) + (y / 2)) % 3 + 1);
}

int main() {
  std::vector<Npp32u> h;
  h.resize((size_t)W * H * 2);
  {
    std::vector<Npp32u> a((size_t)W * H);
    fill_labels(a);
    for (int i = 0; i < 2; ++i)
      for (size_t j = 0; j < a.size(); ++j) h[i * a.size() + j] = a[j] + (Npp32u)(3 * i);
  }
  Npp32u *d = nullptr;
  cudaMalloc(&d, h.size() * sizeof(Npp32u));
  cudaMemcpy(d, h.data(), h.size() * sizeof(Npp32u), cudaMemcpyHostToDevice);

  // per-image buffer size: fixed overhead + 2 * (number of labels)
  const int perImgBuf = kFixed + 8 * 2;
  Npp8u *buf[2] = {nullptr, nullptr};
  for (int i = 0; i < 2; ++i) cudaMalloc(&buf[i], perImgBuf);

  NppiImageDescriptor imgs[2] = {{(char *)d, W * 4, {W, H}},
                                 {(char *)d + (size_t)W * H * 4, W * 4, {W, H}}};
  NppiBufferDescriptor bufs[2] = {{buf[0], perImgBuf}, {buf[1], perImgBuf}};
  Npp32u newMax[2] = {0xFFFFFFFFu, 0xFFFFFFFFu};

  NppStreamContext ctx{};
  ctx.hStream = 0;

  // Reference: Advanced form with the same arguments.
  {
    NppiImageDescriptor aimgs[2] = {{(char *)d, W * 4, {W, H}},
                                    {(char *)d + (size_t)W * H * 4, W * 4, {W, H}}};
    Npp32u nm[2] = {0, 0};
    NppStatus st = nppiCompressMarkerLabelsUFBatch_32u_C1IR_Advanced_Ctx(
        aimgs, bufs, nm, 2, NppiSize{W, H}, perImgBuf, ctx);
    cudaDeviceSynchronize();
    std::vector<Npp32u> out(h.size());
    cudaMemcpy(out.data(), d, out.size() * 4, cudaMemcpyDeviceToHost);
    printf("advanced st=%d newMax=%u,%u out:", (int)st, nm[0], nm[1]);
    for (size_t i = 0; i < out.size(); ++i) printf(" %u", out[i]);
    printf("\n");
  }

  // Reset the label image for the non-Advanced call.
  cudaMemcpy(d, h.data(), h.size() * sizeof(Npp32u), cudaMemcpyHostToDevice);
  {
    NppStatus st = nppiCompressMarkerLabelsUFBatch_32u_C1IR_Ctx(imgs, bufs, newMax, 2,
                                                                NppiSize{W, H}, perImgBuf, ctx);
    cudaDeviceSynchronize();
    std::vector<Npp32u> out(h.size());
    cudaMemcpy(out.data(), d, out.size() * 4, cudaMemcpyDeviceToHost);
    printf("plain    st=%d newMax=%u,%u out:", (int)st, newMax[0], newMax[1]);
    for (size_t i = 0; i < out.size(); ++i) printf(" %u", out[i]);
    printf("\n");
  }

  // Validation matrix (fresh image each time).
  struct Case { const char *name; int batch; int perBuf; NppiSize roi; };
  Case cases[] = {
      {"batch=0", 0, perImgBuf, {W, H}},
      {"batch=1", 1, perImgBuf, {W, H}},
      {"roi=0", 2, perImgBuf, {0, H}},
      {"roi=-1", 2, perImgBuf, {-1, H}},
      {"perBuf=0", 2, 0, {W, H}},
      {"perBuf=small", 2, 100, {W, H}},
      {"perBuf=overhead", 2, kFixed, {W, H}},
      {"perBuf=overhead+2", 2, kFixed + 2, {W, H}},
  };
  for (auto &c : cases) {
    cudaMemcpy(d, h.data(), h.size() * sizeof(Npp32u), cudaMemcpyDeviceToHost);
    NppiImageDescriptor ci[2] = {{(char *)d, W * 4, {W, H}},
                                 {(char *)d + (size_t)W * H * 4, W * 4, {W, H}}};
    NppiBufferDescriptor cb[2] = {{buf[0], c.perBuf}, {buf[1], c.perBuf}};
    Npp32u nm[2] = {0, 0};
    NppStatus st = nppiCompressMarkerLabelsUFBatch_32u_C1IR_Ctx(ci, cb, nm, c.batch, c.roi,
                                                                c.perBuf, ctx);
    cudaDeviceSynchronize();
    printf("%-20s st=%d nm=%u,%u\n", c.name, (int)st, nm[0], nm[1]);
  }

  // Null pointers.
  {
    Npp32u nm[2] = {0, 0};
    NppStatus a = nppiCompressMarkerLabelsUFBatch_32u_C1IR_Ctx(nullptr, bufs, nm, 2, NppiSize{W, H},
                                                               perImgBuf, ctx);
    NppStatus b = nppiCompressMarkerLabelsUFBatch_32u_C1IR_Ctx(imgs, nullptr, nm, 2, NppiSize{W, H},
                                                               perImgBuf, ctx);
    NppStatus c = nppiCompressMarkerLabelsUFBatch_32u_C1IR_Ctx(imgs, bufs, nullptr, 2,
                                                               NppiSize{W, H}, perImgBuf, ctx);
    printf("nulls st=%d,%d,%d\n", (int)a, (int)b, (int)c);
  }
  return 0;
}
