// Compares the mcMPP circular radial profile executor against NVIDIA NPP 13.4
// on the probe configurations (run on the RTX 5090).
#include <cuda_runtime.h>
#include <npp.h>
#include <nppi.h>
#include <cmath>
#include <cstdio>
#include <vector>

#include "../../src/nppi/nppi_statistics_functions/nppi_radial_profile_executor.cu"

extern "C" {
NppStatus nppGetStreamContext(NppStreamContext *c);
}
static NppStreamContext ctx() {
  NppStreamContext c{};
  nppGetStreamContext(&c);
  return c;
}

static int gFail = 0;

static void cmp(const char *tag, int W, int H, float cx, float cy, int samples, int kind) {
  const int stride = W + 3;
  std::vector<float> img((size_t)stride * H);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < stride; ++x) img[(size_t)y * stride + x] = (float)(((x * 11 + y * 7) % 23) - 11) * 0.5f + (float)(x & 3) * 0.25f;
  if (kind == 1) for (auto &v : img) v = 3.5f;
  if (kind == 2) for (auto &v : img) v = 0.0f;
  float *d = nullptr;
  cudaMalloc(&d, img.size() * sizeof(float));
  cudaMemcpy(d, img.data(), img.size() * sizeof(float), cudaMemcpyHostToDevice);
  NppiProfileData *ref = nullptr, *mine = nullptr;
  cudaMalloc(&ref, (size_t)samples * sizeof(NppiProfileData));
  cudaMalloc(&mine, (size_t)samples * sizeof(NppiProfileData));
  std::vector<NppiProfileData> zero((size_t)samples, NppiProfileData{-777, -7.f, -7.f});
  cudaMemcpy(ref, zero.data(), zero.size() * sizeof(NppiProfileData), cudaMemcpyHostToDevice);
  cudaMemcpy(mine, zero.data(), zero.size() * sizeof(NppiProfileData), cudaMemcpyHostToDevice);
  NppStreamContext c = ctx();
  NppStatus s1 = nppiCircularRadialProfile_32f_C1R_Ctx(d, stride * 4, NppiSize{W, H}, NppiPoint32f{cx, cy}, ref, (Npp16u)samples, c);
  NppStatus s2 = nppi::statistics::runCircularRadialProfile(d, stride * 4, NppiSize{W, H}, NppiPoint32f{cx, cy}, mine, (Npp16u)samples, 0);
  std::vector<NppiProfileData> r((size_t)samples), m((size_t)samples);
  cudaMemcpy(r.data(), ref, r.size() * sizeof(NppiProfileData), cudaMemcpyDeviceToHost);
  cudaMemcpy(m.data(), mine, m.size() * sizeof(NppiProfileData), cudaMemcpyDeviceToHost);
  int bad = 0;
  for (int i = 0; i < samples; ++i) {
    if (r[i].nPixels != m[i].nPixels || fabsf(r[i].nMeanIntensity - m[i].nMeanIntensity) > 2e-5f ||
        fabsf(r[i].nStdDevIntensity - m[i].nStdDevIntensity) > 2e-5f) {
      if (bad < 4)
        printf("   [%s] ring %d npp(n=%d, m=%.7g, sd=%.7g) mcpp(n=%d, m=%.7g, sd=%.7g)\n", tag, i, r[i].nPixels,
               r[i].nMeanIntensity, r[i].nStdDevIntensity, m[i].nPixels, m[i].nMeanIntensity, m[i].nStdDevIntensity);
      ++bad;
    }
  }
  printf("%-26s st=%d/%d rings=%d mismatched=%d\n", tag, (int)s1, (int)s2, samples, bad);
  if (bad || s1 != s2) ++gFail;
  cudaFree(d); cudaFree(ref); cudaFree(mine);
}

int main() {
  cmp("33x33 c=(16.5,16.5) S=24", 33, 33, 16.5f, 16.5f, 24, 0);
  cmp("33x33 c=(16.5,16.5) S=12", 33, 33, 16.5f, 16.5f, 12, 0);
  cmp("33x33 c=(16.5,16.5) S=48", 33, 33, 16.5f, 16.5f, 48, 0);
  cmp("33x33 c=(16,16) S=24", 33, 33, 16.0f, 16.0f, 24, 0);
  cmp("33x33 c=(10.5,20.5) S=16", 33, 33, 10.5f, 20.5f, 16, 0);
  cmp("40x25 c=(20.5,12.5) S=24", 40, 25, 20.5f, 12.5f, 24, 0);
  cmp("33x33 c=(-3.5,-2.5) S=8", 33, 33, -3.5f, -2.5f, 8, 0);
  cmp("33x33 c=(31.75,0.25) S=8", 33, 33, 31.75f, 0.25f, 8, 0);
  cmp("33x33 c=(16.3,16.7) S=24", 33, 33, 16.3f, 16.7f, 24, 0);
  cmp("21x15 c=(10.5,7.5) S=12 const", 21, 15, 10.5f, 7.5f, 12, 1);
  cmp("21x15 c=(10.5,7.5) S=12 zeros", 21, 15, 10.5f, 7.5f, 12, 2);
  cmp("17x9 c=(8.5,4.5) S=12", 17, 9, 8.5f, 4.5f, 12, 0);
  cmp("5x5 c=(2.5,2.5) S=6", 5, 5, 2.5f, 2.5f, 6, 0);
  cmp("1x1 c=(0.5,0.5) S=3", 1, 1, 0.5f, 0.5f, 3, 0);
  // randomised centres
  srand(7);
  for (int t = 0; t < 40; ++t) {
    int W = 3 + rand() % 60, H = 3 + rand() % 60;
    float cx = (float)(rand() % (W * 4)) * 0.25f - (float)(W * 0.5f);
    float cy = (float)(rand() % (H * 4)) * 0.25f - (float)(H * 0.5f);
    int S = 1 + rand() % 40;
    char tag[96];
    snprintf(tag, sizeof tag, "rnd%02d %dx%d c=(%.2f,%.2f) S=%d", t, W, H, cx, cy, S);
    cmp(tag, W, H, cx, cy, S, t % 3);
  }
  printf("FAILURES=%d\n", gFail);
  return 0;
}
