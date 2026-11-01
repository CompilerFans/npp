// f13 probe 16: pyramid down impulse responses (which weights land where).
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const char *es(NppStatus s) { static char b[32]; snprintf(b, 32, "%d", (int)s); return b; }

// dump dst for a single-delta source
static void delta_dump(const char *tag, int W, int H, int px, int py, int taps,
                       const float *kern, float rate, NppiBorderType bt) {
  std::vector<Npp32f> h((size_t)W * H, 0.0f);
  h[(size_t)py * W + px] = 1.0f;
  Npp32f *d; cudaMalloc(&d, W * H * 4); cudaMemcpy(d, h.data(), W * H * 4, cudaMemcpyHostToDevice);
  Npp32f *o; cudaMalloc(&o, 8192); cudaMemset(o, 0, 8192);
  NppiSize dst{};
  nppiGetFilterGaussPyramidLayerDownBorderDstROI(W, H, &dst, rate);
  NppStreamContext c{}; c.hStream = 0;
  NppStatus st = nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx(
      d, W * 4, NppiSize{W, H}, NppiPoint{0, 0}, o, dst.width * 4, dst, rate, taps, kern, bt, c);
  printf("%-22s st=%-4s dst=%dx%d:", tag, es(st), dst.width, dst.height);
  if (st == 0) {
    std::vector<Npp32f> v((size_t)dst.width * dst.height);
    cudaMemcpy(v.data(), o, v.size() * 4, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < v.size(); ++i)
      if (v[i] != 0.0f) printf(" (%d,%d)=%.6g", (int)(i % dst.width), (int)(i / dst.width), v[i]);
  }
  printf("\n");
  cudaFree(d); cudaFree(o);
}

int main(int argc, char **argv) {
  int part = argc > 1 ? atoi(argv[1]) : 0;
  const float k3[3] = {1.0f / 3, 1.0f / 3, 1.0f / 3};
  const float k5[5] = {1.0f / 16, 4.0f / 16, 6.0f / 16, 4.0f / 16, 1.0f / 16};
  if (part == 1) {  // 3 taps, rate 2, replicate: delta sweep
    for (int py = 0; py <= 4; ++py)
      for (int px = 0; px <= 4; ++px) {
        char tag[32]; snprintf(tag, 32, "k3 r2 delta(%d,%d)", px, py);
        delta_dump(tag, 12, 12, px, py, 3, k3, 2.0f, NPP_BORDER_REPLICATE);
      }
  }
  if (part == 2) {  // 5 taps, rate 2, replicate: delta sweep
    for (int py = 0; py <= 6; py += 2)
      for (int px = 0; px <= 6; px += 2) {
        char tag[32]; snprintf(tag, 32, "k5 r2 delta(%d,%d)", px, py);
        delta_dump(tag, 14, 14, px, py, 5, k5, 2.0f, NPP_BORDER_REPLICATE);
      }
  }
  if (part == 3) {  // 3 taps, rate 1.5 (non-integer): delta sweep
    for (int py = 0; py <= 3; ++py)
      for (int px = 0; px <= 3; ++px) {
        char tag[32]; snprintf(tag, 32, "k3 r1.5 delta(%d,%d)", px, py);
        delta_dump(tag, 12, 12, px, py, 3, k3, 1.5f, NPP_BORDER_REPLICATE);
      }
  }
  if (part == 4) {  // mirror vs replicate, 3 taps rate 2, deltas near border
    for (int px = 0; px <= 3; ++px) {
      char tag[32];
      snprintf(tag, 32, "k3 r2 MIR delta(%d,2)", px);
      delta_dump(tag, 12, 12, px, 2, 3, k3, 2.0f, NPP_BORDER_MIRROR);
    }
  }
  if (part == 5) {  // 5 taps rate 1.5: sweep to find bilinear tap geometry
    for (int px = 0; px <= 6; ++px) {
      char tag[32]; snprintf(tag, 32, "k5 r1.5 delta(%d,0)", px);
      delta_dump(tag, 14, 14, px, 0, 5, k5, 1.5f, NPP_BORDER_REPLICATE);
    }
  }
  if (part == 6) {  // 1 tap? and rate 3
    const float k1[1] = {1.0f};
    delta_dump("k1 r3 delta(3,3)", 12, 12, 3, 3, 1, k1, 3.0f, NPP_BORDER_REPLICATE);
    delta_dump("k1 r3 delta(0,0)", 12, 12, 0, 0, 1, k1, 3.0f, NPP_BORDER_REPLICATE);
    delta_dump("k3 r3 delta(3,3)", 12, 12, 3, 3, 3, k3, 3.0f, NPP_BORDER_REPLICATE);
    delta_dump("k3 r3 delta(0,3)", 12, 12, 0, 3, 3, k3, 3.0f, NPP_BORDER_REPLICATE);
  }
  return 0;
}
