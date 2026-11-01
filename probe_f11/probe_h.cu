// Probe H: reciprocal-multiply tap variants.
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

int main() {
  Case cases[] = {{2,1.0f,"r2_s1"}, {1,1.0f,"r1_s1"}, {4,1.0f,"r4_s1"}, {2,0.5f,"r2_s05"},
                  {2,2.0f,"r2_s2"}, {3,1.0f,"r3_s1"}, {12,5.0f,"r12_s5"}, {35,5.0f,"r35_s5"}};
  for (auto &c : cases) {
    int r = c.r, k = 2 * r + 1;
    float s2h = 2.0f * c.s * c.s;
    double s2d = 2.0 * c.s * c.s;
    printf("== %s r=%d s=%g\n", c.tag, r, c.s);
    static float g[512 * 512];
    static float t[512 * 512];
    // M1: float expf 2D, S2 float row-major, tap = g * (1/S2)
    {
      float S = 0.f;
      for (int y = 0; y < k; ++y) for (int x = 0; x < k; ++x) {
        float dx = (float)(x - r), dy = (float)(y - r);
        g[y * k + x] = expf(-(dx * dx + dy * dy) / s2h);
        S += g[y * k + x];
      }
      float inv = 1.0f / S;
      for (int j = 0; j < k * k; ++j) t[j] = g[j] * inv;
      dump("M1_f32_2D_mulinv", t, k, r);
    }
    // M2: float expf 2D, S2 double, tap = g * (float)(1/S2d)
    {
      double S = 0;
      for (int y = 0; y < k; ++y) for (int x = 0; x < k; ++x) {
        float dx = (float)(x - r), dy = (float)(y - r);
        g[y * k + x] = expf(-(dx * dx + dy * dy) / s2h);
        S += g[y * k + x];
      }
      float inv = (float)(1.0 / S);
      for (int j = 0; j < k * k; ++j) t[j] = g[j] * inv;
      dump("M2_f32g_dS_mulinv", t, k, r);
    }
    // M3: float expf 1D, w = v * (1/S1), tap = w(x)*w(y)
    {
      float v[1024], w[1024];
      float S = 0.f;
      for (int x = 0; x < k; ++x) { float dx = (float)(x - r); v[x] = expf(-(dx * dx) / s2h); S += v[x]; }
      float inv = 1.0f / S;
      for (int x = 0; x < k; ++x) w[x] = v[x] * inv;
      for (int y = 0; y < k; ++y) for (int x = 0; x < k; ++x) t[y * k + x] = w[x] * w[y];
      dump("M3_f32_1D_mulinv", t, k, r);
    }
    // M4: float expf 1D, tap = (v(x)*v(y)) * (1/(S1*S1))
    {
      float v[1024];
      float S = 0.f;
      for (int x = 0; x < k; ++x) { float dx = (float)(x - r); v[x] = expf(-(dx * dx) / s2h); S += v[x]; }
      float inv2 = 1.0f / (S * S);
      for (int y = 0; y < k; ++y) for (int x = 0; x < k; ++x) t[y * k + x] = (v[x] * v[y]) * inv2;
      dump("M4_f32_1D_prodinv2", t, k, r);
    }
    // M5: double exp 2D, S2 double, tap = (float)(g * (1.0/S))
    {
      double S = 0;
      for (int y = 0; y < k; ++y) for (int x = 0; x < k; ++x) {
        double dx = x - r, dy = y - r;
        S += exp(-(dx * dx + dy * dy) / s2d);
      }
      double inv = 1.0 / S;
      for (int y = 0; y < k; ++y) for (int x = 0; x < k; ++x) {
        double dx = x - r, dy = y - r;
        t[y * k + x] = (float)(exp(-(dx * dx + dy * dy) / s2d) * inv);
      }
      dump("M5_dbl_2D_mulinv", t, k, r);
    }
    // M6: float expf 2D, S2 float REVERSE sum order, tap = g*(1/S)
    {
      float S = 0.f;
      for (int y = k - 1; y >= 0; --y) for (int x = k - 1; x >= 0; --x) {
        float dx = (float)(x - r), dy = (float)(y - r);
        g[y * k + x] = expf(-(dx * dx + dy * dy) / s2h);
      }
      for (int j = k * k - 1; j >= 0; --j) S += g[j];
      float inv = 1.0f / S;
      for (int j = 0; j < k * k; ++j) t[j] = g[j] * inv;
      dump("M6_f32_2D_revsum", t, k, r);
    }
    // M7: float expf 2D, separate exp args expf(-dx*dx/s2)*expf(-dy*dy/s2) as 2D product, S2 float, mul inv
    {
      float S = 0.f;
      for (int y = 0; y < k; ++y) for (int x = 0; x < k; ++x) {
        float dx = (float)(x - r), dy = (float)(y - r);
        g[y * k + x] = expf(-(dx * dx) / s2h) * expf(-(dy * dy) / s2h);
        S += g[y * k + x];
      }
      float inv = 1.0f / S;
      for (int j = 0; j < k * k; ++j) t[j] = g[j] * inv;
      dump("M7_f32_sep_2D_mulinv", t, k, r);
    }
    // M8: float expf 1D but exp arg -(d*d)/(s+s)/... no; try expf(-(d/s)*(d/s)/2)
    {
      float v[1024], w[1024];
      float S = 0.f;
      for (int x = 0; x < k; ++x) { float dx = (float)(x - r); v[x] = expf(-(dx / c.s) * (dx / c.s) / 2.0f); S += v[x]; }
      float inv = 1.0f / S;
      for (int x = 0; x < k; ++x) w[x] = v[x] * inv;
      for (int y = 0; y < k; ++y) for (int x = 0; x < k; ++x) t[y * k + x] = w[x] * w[y];
      dump("M8_f32_1D_divfirst", t, k, r);
    }
  }
  return 0;
}
