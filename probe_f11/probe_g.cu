// Probe G: identify exact host/device arithmetic of NPP unsharp Gaussian taps.
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

struct Case { int r; float s; const char *tag; };

static void dump(const char *name, const float *taps, int k, int r) {
  double sum = 0;
  for (int i = 0; i < k * k; ++i) sum += taps[i];
  printf("%s sum=%.9g ctr_row:", name, sum);
  for (int x = 0; x < k; ++x) printf(" %a", taps[r * k + x]);
  printf(" | corner:");
  for (int y = 0; y < (k < 3 ? k : 3); ++y) for (int x = 0; x < (k < 3 ? k : 3); ++x) printf(" %a", taps[y * k + x]);
  printf("\n");
}

__global__ void dev_taps(float *out, int r, float sigma, int mode) {
  int k = 2 * r + 1;
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= k * k) return;
  int y = i / k, x = i % k;
  float dx = (float)(x - r), dy = (float)(y - r);
  float s2 = 2.0f * sigma * sigma;
  if (mode == 0) {          // device expf, 2D, S2 accumulated on device (single pass impossible; use __shared__? just compute full sums per thread)
    float g = expf(-(dx * dx + dy * dy) / s2);
    float S = 0.f;
    for (int j = 0; j < k * k; ++j) {
      int jy = j / k, jx = j % k;
      float jdx = (float)(jx - r), jdy = (float)(jy - r);
      S += expf(-(jdx * jdx + jdy * jdy) / s2);
    }
    out[i] = g / S;
  } else if (mode == 1) {   // device expf, 1D normalized
    float vx = expf(-(dx * dx) / s2);
    float vy = expf(-(dy * dy) / s2);
    float S1 = 0.f;
    for (int j = 0; j < k; ++j) {
      float jd = (float)(j - r);
      S1 += expf(-(jd * jd) / s2);
    }
    out[i] = (vx / S1) * (vy / S1);
  } else {                  // device __expf 1D
    float vx = __expf(-(dx * dx) / s2);
    float vy = __expf(-(dy * dy) / s2);
    float S1 = 0.f;
    for (int j = 0; j < k; ++j) {
      float jd = (float)(j - r);
      S1 += __expf(-(jd * jd) / s2);
    }
    out[i] = (vx / S1) * (vy / S1);
  }
}

int main() {
  Case cases[] = {{2,1.0f,"r2_s1"}, {1,1.0f,"r1_s1"}, {4,1.0f,"r4_s1"}, {2,0.5f,"r2_s05"},
                  {2,2.0f,"r2_s2"}, {3,1.0f,"r3_s1"}, {12,5.0f,"r12_s5"}, {35,5.0f,"r35_s5"}};
  for (auto &c : cases) {
    int r = c.r, k = 2 * r + 1;
    float s2h = 2.0f * c.s * c.s;
    double s2d = 2.0 * c.s * c.s;
    printf("== %s r=%d s=%g\n", c.tag, r, c.s);
    // H1: host double 2D
    {
      float *t = new float[k * k];
      double S = 0;
      for (int y = 0; y < k; ++y) for (int x = 0; x < k; ++x) {
        double dx = x - r, dy = y - r;
        S += exp(-(dx * dx + dy * dy) / s2d);
      }
      for (int y = 0; y < k; ++y) for (int x = 0; x < k; ++x) {
        double dx = x - r, dy = y - r;
        t[y * k + x] = (float)(exp(-(dx * dx + dy * dy) / s2d) / S);
      }
      dump("H1_host_double_2D", t, k, r);
      delete[] t;
    }
    // H2: host double 1D, w then product
    {
      float *t = new float[k * k];
      double S = 0;
      double v[512];
      for (int x = 0; x < k; ++x) { double dx = x - r; v[x] = exp(-(dx * dx) / s2d); S += v[x]; }
      float w[512];
      for (int x = 0; x < k; ++x) w[x] = (float)(v[x] / S);
      for (int y = 0; y < k; ++y) for (int x = 0; x < k; ++x) t[y * k + x] = w[x] * w[y];
      dump("H2_host_double_1D", t, k, r);
      delete[] t;
    }
    // H3: host float expf 2D
    {
      float *t = new float[k * k];
      float S = 0.f;
      for (int y = 0; y < k; ++y) for (int x = 0; x < k; ++x) {
        float dx = (float)(x - r), dy = (float)(y - r);
        S += expf(-(dx * dx + dy * dy) / s2h);
      }
      for (int y = 0; y < k; ++y) for (int x = 0; x < k; ++x) {
        float dx = (float)(x - r), dy = (float)(y - r);
        t[y * k + x] = expf(-(dx * dx + dy * dy) / s2h) / S;
      }
      dump("H3_host_expf_2D", t, k, r);
      delete[] t;
    }
    // H4: host float expf 1D
    {
      float *t = new float[k * k];
      float S = 0.f;
      float v[512], w[512];
      for (int x = 0; x < k; ++x) { float dx = (float)(x - r); v[x] = expf(-(dx * dx) / s2h); S += v[x]; }
      for (int x = 0; x < k; ++x) w[x] = v[x] / S;
      for (int y = 0; y < k; ++y) for (int x = 0; x < k; ++x) t[y * k + x] = w[x] * w[y];
      dump("H4_host_expf_1D", t, k, r);
      delete[] t;
    }
    // H5: host float expf 1D, sum center-out (v0 + 2*v1 + ...)
    {
      float *t = new float[k * k];
      float S = 0.f;
      float v[512], w[512];
      for (int x = 0; x < k; ++x) { float dx = (float)(x - r); v[x] = expf(-(dx * dx) / s2h); }
      S = v[r];
      for (int d = 1; d <= r; ++d) S += 2.f * v[r + d];
      for (int x = 0; x < k; ++x) w[x] = v[x] / S;
      for (int y = 0; y < k; ++y) for (int x = 0; x < k; ++x) t[y * k + x] = w[x] * w[y];
      dump("H5_host_expf_1Dco", t, k, r);
      delete[] t;
    }
    // H6: host float expf 2D, sum center-out row order
    {
      float *t = new float[k * k];
      float g[512 * 512]; // only for small k
      if (k <= 71) {
        float S = 0.f;
        for (int y = 0; y < k; ++y) for (int x = 0; x < k; ++x) {
          float dx = (float)(x - r), dy = (float)(y - r);
          g[y * k + x] = expf(-(dx * dx + dy * dy) / s2h);
        }
        for (int j = 0; j < k * k; ++j) S += g[j];
        for (int j = 0; j < k * k; ++j) t[j] = g[j] / S;
        dump("H6_host_expf_2Dsum", t, k, r);
      }
      delete[] t;
    }
    // device variants
    if (k <= 71) {
      float *d_out, *h = new float[k * k];
      cudaMalloc(&d_out, k * k * 4);
      for (int mode = 0; mode < 3; ++mode) {
        int n = k * k;
        dev_taps<<<(n + 255) / 256, 256>>>(d_out, r, c.s, mode);
        cudaMemcpy(h, d_out, n * 4, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        dump(mode == 0 ? "D0_dev_expf_2D" : (mode == 1 ? "D1_dev_expf_1D" : "D2_dev__expf_1D"), h, k, r);
      }
      cudaFree(d_out);
      delete[] h;
    }
  }
  return 0;
}
