// Second-round NVIDIA NPP 13.4 semantics probe for package s2.
// Resolves: QualityIndex C3/AC4 statistics sharing, CountInRange AC4 arrays,
// histogram C3/AC4 binning + alpha, RectStdDev 32s32f/32s Sfs, EvenLevelsHost,
// radial profile ring membership via impulses, and SSIM/MSSSIM values.
#include <cuda_runtime.h>
#include <npp.h>
#include <nppi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
NppStatus nppGetStreamContext(NppStreamContext *pNppStreamContext);
}

static const char *sname(NppStatus s) {
  switch (s) {
    case NPP_SUCCESS: return "NPP_SUCCESS";
    case NPP_NULL_POINTER_ERROR: return "NPP_NULL_POINTER_ERROR";
    case NPP_SIZE_ERROR: return "NPP_SIZE_ERROR";
    case NPP_STEP_ERROR: return "NPP_STEP_ERROR";
    case NPP_RANGE_ERROR: return "NPP_RANGE_ERROR";
    case NPP_QUALITY_INDEX_ERROR: return "NPP_QUALITY_INDEX_ERROR";
    case NPP_RECTANGLE_ERROR: return "NPP_RECTANGLE_ERROR";
    case NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR: return "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR";
    case NPP_NO_OPERATION_WARNING: return "NPP_NO_OPERATION_WARNING";
    default: break;
  }
  static char buf[4][32];
  static int i = 0;
  i = (i + 1) & 3;
  snprintf(buf[i], 32, "STATUS_%d", (int)s);
  return buf[i];
}
#define PS(tag, s) printf("  %-16s status=%d (%s)\n", tag, (int)(s), sname(s))
#define CHK(tag) do { cudaError_t e = cudaDeviceSynchronize(); printf("  %-16s cuda=%s\n", tag, cudaGetErrorString(e)); (void)cudaGetLastError(); } while (0)

static Npp8u *gBuf64 = nullptr;
static NppStreamContext ctx() {
  NppStreamContext c{};
  nppGetStreamContext(&c);
  return c;
}

template <typename T> struct Dev {
  T *p = nullptr;
  size_t n = 0;
  explicit Dev(size_t count) : n(count) { cudaMalloc(&p, count * sizeof(T)); }
  ~Dev() { if (p) cudaFree(p); }
  void put(const std::vector<T> &h) { cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice); }
  std::vector<T> get() const {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
  }
  void fill(T v) { std::vector<T> h(n, v); put(h); }
};

static void probe_qidx3() {
  const int W = 8, H = 4;
  const int N = W * H;
  std::vector<Npp32f> a((size_t)N * 3), b((size_t)N * 3);
  for (int i = 0; i < N; ++i) {
    float v = (float)(i % 7) - 3.0f;
    a[i * 3 + 0] = v;
    b[i * 3 + 0] = 2.0f * v;
    a[i * 3 + 1] = v + 100.0f;
    b[i * 3 + 1] = 2.0f * v + 100.0f;
    a[i * 3 + 2] = 10.0f * v;
    b[i * 3 + 2] = 20.0f * v;
  }
  Dev<Npp32f> s1((size_t)N * 3), s2((size_t)N * 3), out(4);
  s1.put(a);
  s2.put(b);
  out.fill(-1e30f);
  NppStatus s = nppiQualityIndex_32f_C3R_Ctx(s1.p, W * 12, s2.p, W * 12, NppiSize{W, H}, out.p, gBuf64, ctx());
  PS("C3_varies", s);
  auto v = out.get();
  printf("  out=[%.9g %.9g %.9g %.9g]\n", v[0], v[1], v[2], v[3]);
  for (int k = 0; k < 3; ++k) {
    double ma = 0, mb = 0, va = 0, vb = 0, cov = 0;
    for (int i = 0; i < N; ++i) { ma += a[i * 3 + k]; mb += b[i * 3 + k]; }
    ma /= N; mb /= N;
    for (int i = 0; i < N; ++i) {
      va += (a[i * 3 + k] - ma) * (a[i * 3 + k] - ma);
      vb += (b[i * 3 + k] - mb) * (b[i * 3 + k] - mb);
      cov += (a[i * 3 + k] - ma) * (b[i * 3 + k] - mb);
    }
    va /= N - 1; vb /= N - 1; cov /= N - 1;
    double q = 4.0 * cov * ma * mb / ((ma * ma + mb * mb) * (va + vb));
    printf("  ch%d mu_a=%.6g mu_b=%.6g va=%.6g vb=%.6g cov=%.6g  Q_perchan=%.9g\n", k, ma, mb, va, vb, cov, q);
  }
  {
    std::vector<Npp32f> c(a);
    for (int i = 0; i < N; ++i) c[i * 3 + 2] = 5.0f;
    Dev<Npp32f> d((size_t)N * 3);
    d.put(c);
    out.fill(-1e30f);
    NppStatus st = nppiQualityIndex_32f_C3R_Ctx(d.p, W * 12, s2.p, W * 12, NppiSize{W, H}, out.p, gBuf64, ctx());
    PS("C3_ch2const", st);
    auto vv = out.get();
    printf("  out=[%.9g %.9g %.9g]\n", vv[0], vv[1], vv[2]);
  }
  {
    std::vector<Npp32f> pa((size_t)N * 4), pb((size_t)N * 4);
    for (int i = 0; i < N; ++i)
      for (int k = 0; k < 3; ++k) { pa[i * 4 + k] = a[i * 3 + k]; pb[i * 4 + k] = b[i * 3 + k]; }
    for (int i = 0; i < N; ++i) { pa[i * 4 + 3] = 7.0f; pb[i * 4 + 3] = 200.0f; }
    Dev<Npp32f> d1((size_t)N * 4), d2((size_t)N * 4);
    d1.put(pa);
    d2.put(pb);
    out.fill(-1e30f);
    NppStatus st = nppiQualityIndex_32f_AC4R_Ctx(d1.p, W * 16, d2.p, W * 16, NppiSize{W, H}, out.p, gBuf64, ctx());
    PS("AC4_varies", st);
    auto vv = out.get();
    printf("  out=[%.9g %.9g %.9g %.9g]\n", vv[0], vv[1], vv[2], vv[3]);
  }
}

static void probe_cir_ac4() {
  const int W = 8, H = 4;
  std::vector<Npp8u> t((size_t)W * H * 4);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      for (int c = 0; c < 4; ++c) t[((size_t)y * W + x) * 4 + c] = (Npp8u)((x * 7 + y * 13 + c * 29 + 1) & 0xff);
  for (size_t i = 3; i < t.size(); i += 4) t[i] = 128;
  Dev<Npp8u> src(t.size());
  src.put(t);
  Dev<int> cnt(4);
  cnt.fill(-12345);
  Npp8u lo[3] = {10, 0, 200}, hi[3] = {100, 10, 255};
  NppStatus s = nppiCountInRange_8u_AC4R_Ctx(src.p, W * 4, NppiSize{W, H}, cnt.p, lo, hi, gBuf64, ctx());
  PS("AC4", s);
  auto c = cnt.get();
  printf("  counts=[%d %d %d %d]\n", c[0], c[1], c[2], c[3]);
  int gt[3] = {0, 0, 0};
  for (int i = 0; i < W * H; ++i)
    for (int k = 0; k < 3; ++k) {
      Npp8u v = t[(size_t)i * 4 + k];
      if (v >= lo[k] && v <= hi[k]) gt[k]++;
    }
  printf("  truth=[%d %d %d]\n", gt[0], gt[1], gt[2]);
  std::vector<Npp32f> f(t.size());
  for (size_t i = 0; i < t.size(); ++i) f[i] = (float)t[i] * 0.5f - 20.0f;
  Dev<Npp32f> fsrc(f.size());
  fsrc.put(f);
  Dev<int> fcnt(4);
  fcnt.fill(-12345);
  Npp32f flo[3] = {-20.0f, 0.0f, 80.0f}, fhi[3] = {30.0f, 10.0f, 110.0f};
  s = nppiCountInRange_32f_AC4R_Ctx(fsrc.p, (int)(W * 4 * sizeof(Npp32f)), NppiSize{W, H}, fcnt.p, flo, fhi, gBuf64,
                                    ctx());
  PS("AC4_32f", s);
  auto fc = fcnt.get();
  printf("  counts=[%d %d %d]\n", fc[0], fc[1], fc[2]);
}

static void fmtbin(const char *tag, NppStatus s, const std::vector<Npp32s> &v) {
  printf("  %-16s st=%d bins=[", tag, (int)s);
  for (size_t i = 0; i < v.size(); ++i) printf("%d%s", v[i], i + 1 == v.size() ? "" : " ");
  printf("]\n");
}

static void probe_hist3() {
  const int W = 32, H = 16;
  std::vector<Npp8u> t((size_t)W * H * 3);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      for (int c = 0; c < 3; ++c) t[((size_t)y * W + x) * 3 + c] = (Npp8u)((x * 7 + y * 13 + c * 29 + 1) & 0xff);
  Dev<Npp8u> src(t.size());
  src.put(t);
  {
    Dev<Npp32s> h0(3), h1(3), h2(3);
    h0.fill(-1); h1.fill(-1); h2.fill(-1);
    Npp32s *hist[3] = {h0.p, h1.p, h2.p};
    int levels[3] = {3, 3, 3};
    Npp32s lo[3] = {64, 0, 100}, hi[3] = {192, 256, 256};
    NppStatus s = nppiHistogramEven_8u_C3R_Ctx(src.p, W * 3, NppiSize{W, H}, hist, levels, lo, hi, gBuf64, ctx());
    fmtbin("Even_C3_ch0", s, h0.get());
    std::vector<Npp32s> b1 = h1.get(), b2 = h2.get();
    printf("    ch1=[%d %d %d] ch2=[%d %d %d]\n", b1[0], b1[1], b1[2], b2[0], b2[1], b2[2]);
  }
  {
    std::vector<Npp8u> q((size_t)W * H * 4);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        for (int c = 0; c < 4; ++c) q[((size_t)y * W + x) * 4 + c] = (Npp8u)((x * 5 + y * 11 + c * 31 + 3) & 0xff);
    Dev<Npp8u> s4(q.size());
    s4.put(q);
    Dev<Npp32s> h0(4), h1(4), h2(4);
    h0.fill(-1); h1.fill(-1); h2.fill(-1);
    Npp32s *hist[3] = {h0.p, h1.p, h2.p};
    int levels[3] = {4, 4, 4};
    Npp32s lo[3] = {0, 0, 0}, hi[3] = {256, 256, 256};
    NppStatus s = nppiHistogramEven_8u_AC4R_Ctx(s4.p, W * 4, NppiSize{W, H}, hist, levels, lo, hi, gBuf64, ctx());
    fmtbin("Even_AC4_ch0", s, h0.get());
    std::vector<Npp32s> b1 = h1.get();
    printf("    ch1=[%d %d %d %d]\n", b1[0], b1[1], b1[2], b1[3]);
  }
  {
    Dev<Npp32s> h0(4), h1(4), h2(4);
    h0.fill(-1); h1.fill(-1); h2.fill(-1);
    Npp32s *hist[3] = {h0.p, h1.p, h2.p};
    std::vector<Npp32s> lv0 = {0, 64, 128, 192, 256};
    std::vector<Npp32s> lv1 = {0, 85, 170, 256};
    std::vector<Npp32s> lv2 = {0, 128, 129, 256};
    Dev<Npp32s> l0(5), l1(4), l2(4);
    l0.put(lv0); l1.put(lv1); l2.put(lv2);
    const Npp32s *levels[3] = {l0.p, l1.p, l2.p};
    int nl[3] = {5, 4, 4};
    NppStatus s = nppiHistogramRange_8u_C3R_Ctx(src.p, W * 3, NppiSize{W, H}, hist, levels, nl, gBuf64, ctx());
    fmtbin("Range_C3_ch0", s, h0.get());
    std::vector<Npp32s> b1 = h1.get(), b2 = h2.get();
    printf("    ch1=[%d %d %d] ch2=[%d %d %d]\n", b1[0], b1[1], b1[2], b2[0], b2[1], b2[2]);
    std::vector<Npp8u> t2 = t;
    t2[0] = 255;
    Dev<Npp8u> s2(t2.size());
    s2.put(t2);
    Dev<Npp32s> k0(4);
    k0.fill(-1);
    Npp32s *h[3] = {k0.p, h0.p, h0.p};
    NppStatus st = nppiHistogramRange_8u_C3R_Ctx(s2.p, W * 3, NppiSize{W, H}, h, levels, nl, gBuf64, ctx());
    fmtbin("Range_top255", st, k0.get());
  }
  {
    std::vector<Npp8u> t2((size_t)W * H);
    for (int i = 0; i < W * H; ++i) t2[i] = (Npp8u)(i % 256);
    Dev<Npp8u> src1(t2.size());
    src1.put(t2);
    Dev<Npp32s> h(3);
    h.fill(-1);
    NppStatus s = nppiHistogramEven_8u_C1R_Ctx(src1.p, W, NppiSize{W, H}, h.p, 3, 64, 192, gBuf64, ctx());
    fmtbin("Even_c1_top", s, h.get());
    int c0 = 0, c1 = 0, c2 = 0;
    for (auto v : t2) { if (v >= 64 && v < 128) c0++; else if (v >= 128 && v < 192) c1++; else if (v >= 192 && v < 256) c2++; }
    printf("    truth=[%d %d] plus192+=%d\n", c0, c1, c2);
  }
}

static void probe_rect2() {
  const int IW = 13, IH = 13;
  std::vector<double> img((size_t)12 * 12);
  for (int y = 0; y < 12; ++y)
    for (int x = 0; x < 12; ++x) img[(size_t)y * 12 + x] = (double)((x * 5 + y * 3) % 7);
  std::vector<Npp32s> Ii((size_t)IW * IH, 0), Si((size_t)IW * IH, 0);
  std::vector<double> Id((size_t)IW * IH, 0.0), Sd((size_t)IW * IH, 0.0);
  for (int y = 1; y < IH; ++y)
    for (int x = 1; x < IW; ++x) {
      double v = img[(size_t)(y - 1) * 12 + (x - 1)], q = v * v;
      Id[(size_t)y * IW + x] = v + Id[(size_t)(y - 1) * IW + x] + Id[(size_t)y * IW + x - 1] -
                               Id[(size_t)(y - 1) * IW + x - 1];
      Sd[(size_t)y * IW + x] = q + Sd[(size_t)(y - 1) * IW + x] + Sd[(size_t)y * IW + x - 1] -
                               Sd[(size_t)(y - 1) * IW + x - 1];
      Ii[(size_t)y * IW + x] = (Npp32s)Id[(size_t)y * IW + x];
      Si[(size_t)y * IW + x] = (Npp32s)Sd[(size_t)y * IW + x];
    }
  Dev<Npp32s> dI((size_t)IW * IH), dS((size_t)IW * IH);
  dI.put(Ii);
  dS.put(Si);
  Dev<Npp64f> sqd((size_t)IW * IH);
  std::vector<Npp64f> sq((size_t)IW * IH);
  for (size_t i = 0; i < sq.size(); ++i) sq[i] = Sd[i];
  sqd.put(sq);
  Dev<Npp32f> df((size_t)8 * 8);
  NppiRect rect{1, 1, 3, 3};
  NppStatus s = nppiRectStdDev_32s32f_C1R_Ctx(dI.p, IW * 4, sqd.p, IW * 8, df.p, 8 * 4, NppiSize{8, 8}, rect, ctx());
  PS("32s32f", s);
  auto v = df.get();
  printf("  dst row0: %g %g %g | row1: %g %g %g\n", v[0], v[1], v[2], v[8], v[9], v[10]);
  for (int scale : {0, 1, 2, -1, 8}) {
    Dev<Npp32s> di((size_t)8 * 8);
    di.fill(-12345);
    NppStatus st = nppiRectStdDev_32s_C1RSfs_Ctx(dI.p, IW * 4, dS.p, IW * 4, di.p, 8 * 4, NppiSize{8, 8}, rect, scale,
                                                 ctx());
    auto vi = di.get();
    printf("  32sSfs s=%d st=%d row0=[%d %d %d %d] row1=[%d %d %d %d]\n", scale, (int)st, vi[0], vi[1], vi[2], vi[3],
           vi[8], vi[9], vi[10], vi[11]);
  }
  printf("  ref double sqrt row0:");
  for (int x = 0; x < 4; ++x) {
    double m1 = 0, m2 = 0;
    for (int dy = 0; dy < 3; ++dy)
      for (int dx = 0; dx < 3; ++dx) {
        double q = img[(size_t)(dy + 1) * 12 + (x + 1 + dx)];
        m1 += q;
        m2 += q * q;
      }
    printf(" %.9g", sqrt(m2 / 9.0 - (m1 / 9.0) * (m1 / 9.0)));
  }
  printf("\n");
}

static void probe_levels2() {
  struct C { int n; Npp32s lo, hi; };
  C cs[] = {{3, 64, 192}, {5, 0, 256}, {4, -10, 10}, {2, 5, 5}, {3, 10, 5}, {1, 0, 255}, {257, 0, 256}, {6, 0, 10},
            {4, 0, 10}, {3, 0, 7}, {7, 0, 10}, {9, 0, 100}, {4, 5, -5}, {5, 0, 1}, {3, -1, 0}};
  for (auto &c : cs) {
    std::vector<Npp32s> lv(c.n > 0 ? c.n : 1, -777);
    NppStatus s = nppiEvenLevelsHost_32s(lv.data(), c.n, c.lo, c.hi);
    printf("  levels n=%d lo=%d hi=%d st=%d vals=[", c.n, c.lo, c.hi, (int)s);
    for (int i = 0; i < c.n && i < 12; ++i) printf("%d ", lv[i]);
    printf("]\n");
  }
}

static void probe_radial_imp() {
  const int W = 33, H = 33;
  std::vector<Npp32f> img((size_t)W * H, 0.0f);
  struct P { int x, y; };
  P ps[] = {{16,16},{17,17},{16,17},{15,16},{15,17},{14,16},{14,14},{16,14},{18,18},{15,15},{13,16},{17,14},
            {12,12},{20,20},{16,20},{16,12},{0,16},{32,16},{16,0},{16,32}};
  Dev<Npp32f> src((size_t)W * H);
  Dev<NppiProfileData> prof(40);
  for (auto &pp : ps) {
    std::vector<Npp32f> one((size_t)W * H, 0.0f);
    one[(size_t)pp.y * W + pp.x] = 1.0f;
    src.put(one);
    prof.fill(NppiProfileData{0, -1e30f, -1e30f});
    NppStatus s = nppiCircularRadialProfile_32f_C1R_Ctx(src.p, W * 4, NppiSize{W, H}, NppiPoint32f{16.5f, 16.5f},
                                                        prof.p, 24, ctx());
    auto v = prof.get();
    printf("  imp(%2d,%2d) st=%d:", pp.x, pp.y, (int)s);
    for (int i = 0; i < 24; ++i)
      if (v[i].nPixels) printf(" ring%d(n=%d,m=%g)", i, v[i].nPixels, v[i].nMeanIntensity);
    printf("\n");
  }
  {
    std::vector<Npp32f> zero((size_t)W * H, 0.0f);
    src.put(zero);
    prof.fill(NppiProfileData{0, -1e30f, -1e30f});
    nppiCircularRadialProfile_32f_C1R_Ctx(src.p, W * 4, NppiSize{W, H}, NppiPoint32f{16.5f, 16.5f}, prof.p, 24, ctx());
    auto v = prof.get();
    printf("  counts:");
    for (int i = 0; i < 24; ++i) printf(" %d", v[i].nPixels);
    printf("\n");
  }
  P es[] = {{16,16},{17,17},{16,17},{15,16},{18,16},{16,18},{20,16},{16,20},{22,16},{16,22},{18,18},{14,16},
            {12,16},{16,12},{24,16},{16,24}};
  for (auto &pp : es) {
    std::vector<Npp32f> one((size_t)W * H, 0.0f);
    one[(size_t)pp.y * W + pp.x] = 1.0f;
    src.put(one);
    prof.fill(NppiProfileData{0, -1e30f, -1e30f});
    NppStatus s = nppiEllipticalRadialProfile_32f_C1R_Ctx(src.p, W * 4, NppiSize{W, H}, NppiPoint32f{16.5f, 16.5f},
                                                          2.0f, 0.0f, prof.p, 24, ctx());
    auto v = prof.get();
    printf("  eimp(%2d,%2d) st=%d:", pp.x, pp.y, (int)s);
    for (int i = 0; i < 24; ++i)
      if (v[i].nPixels) printf(" ring%d(n=%d,m=%g)", i, v[i].nPixels, v[i].nMeanIntensity);
    printf("\n");
  }
  {
    std::vector<Npp32f> zero((size_t)W * H, 0.0f);
    src.put(zero);
    prof.fill(NppiProfileData{0, -1e30f, -1e30f});
    nppiEllipticalRadialProfile_32f_C1R_Ctx(src.p, W * 4, NppiSize{W, H}, NppiPoint32f{16.5f, 16.5f}, 2.0f, 0.0f,
                                            prof.p, 24, ctx());
    auto v = prof.get();
    printf("  ecounts:");
    for (int i = 0; i < 24; ++i) printf(" %d", v[i].nPixels);
    printf("\n");
  }
}


static void probe_radial_scan(int samples) {
  const int W = 33, H = 33;
  Dev<Npp32f> src((size_t)W * H);
  Dev<NppiProfileData> prof(300);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      std::vector<Npp32f> one((size_t)W * H, 0.0f);
      one[(size_t)y * W + x] = 1.0f;
      src.put(one);
      prof.fill(NppiProfileData{0, -1e30f, -1e30f});
      NppStatus s = nppiCircularRadialProfile_32f_C1R_Ctx(src.p, W * 4, NppiSize{W, H}, NppiPoint32f{16.5f, 16.5f},
                                                          prof.p, (Npp16u)samples, ctx());
      if (s != NPP_SUCCESS) { printf("%d %d st=%d\n", x, y, (int)s); continue; }
      auto v = prof.get();
      for (int i = 0; i < samples; ++i) {
        if (v[i].nPixels > 0 && v[i].nMeanIntensity > 0.0f) {
          double mult = (double)v[i].nMeanIntensity * (double)v[i].nPixels;
          printf("%d %d %d %.6f\n", x, y, i, mult);
        }
      }
    }
}


static void probe_hist_err() {
  const int W = 8, H = 4;
  std::vector<Npp8u> t((size_t)W * H * 3, 5);
  Dev<Npp8u> src(t.size());
  src.put(t);
  Dev<Npp32s> hh(256);
  hh.fill(0);
  Npp32s *hist[3] = {hh.p, hh.p, hh.p};
  int levels[3] = {4, 4, 4};
  Npp32s lo[3] = {0, 0, 0}, hi[3] = {256, 256, 256};
  Dev<Npp32s> lv(4);
  lv.fill(0);
  const Npp32s *lev[3] = {lv.p, lv.p, lv.p};
  PS("null_src", nppiHistogramEven_8u_C3R_Ctx(nullptr, W * 3, NppiSize{W, H}, hist, levels, lo, hi, gBuf64, ctx()));
  PS("null_hist", nppiHistogramEven_8u_C3R_Ctx(src.p, W * 3, NppiSize{W, H}, nullptr, levels, lo, hi, gBuf64, ctx()));
  PS("null_levels", nppiHistogramEven_8u_C3R_Ctx(src.p, W * 3, NppiSize{W, H}, hist, nullptr, lo, hi, gBuf64, ctx()));
  PS("zro_roi", nppiHistogramEven_8u_C3R_Ctx(src.p, W * 3, NppiSize{0, H}, hist, levels, lo, hi, gBuf64, ctx()));
  PS("neg_roi", nppiHistogramEven_8u_C3R_Ctx(src.p, W * 3, NppiSize{-1, H}, hist, levels, lo, hi, gBuf64, ctx()));
  PS("short_step", nppiHistogramEven_8u_C3R_Ctx(src.p, 3, NppiSize{W, H}, hist, levels, lo, hi, gBuf64, ctx()));
  int one[3] = {1, 4, 4};
  PS("one_level", nppiHistogramEven_8u_C3R_Ctx(src.p, W * 3, NppiSize{W, H}, hist, one, lo, hi, gBuf64, ctx()));
  Npp32s revlo[3] = {256, 0, 0}, revhi[3] = {0, 256, 256};
  PS("rev_range", nppiHistogramEven_8u_C3R_Ctx(src.p, W * 3, NppiSize{W, H}, hist, levels, revlo, revhi, gBuf64, ctx()));
  int nl[3] = {4, 4, 4};
  PS("rng_null_src", nppiHistogramRange_8u_C3R_Ctx(nullptr, W * 3, NppiSize{W, H}, hist, lev, nl, gBuf64, ctx()));
  PS("rng_null_lev", nppiHistogramRange_8u_C3R_Ctx(src.p, W * 3, NppiSize{W, H}, hist, nullptr, nl, gBuf64, ctx()));
  int n1[3] = {1, 4, 4};
  PS("rng_one_level", nppiHistogramRange_8u_C3R_Ctx(src.p, W * 3, NppiSize{W, H}, hist, lev, n1, gBuf64, ctx()));
  PS("rng_zro_roi", nppiHistogramRange_8u_C3R_Ctx(src.p, W * 3, NppiSize{0, H}, hist, lev, nl, gBuf64, ctx()));
  // AC4 one level / rev
  Dev<Npp8u> src4((size_t)W * H * 4);
  src4.fill(7);
  PS("ac4_null_src", nppiHistogramEven_8u_AC4R_Ctx(nullptr, W * 4, NppiSize{W, H}, hist, levels, lo, hi, gBuf64, ctx()));
  PS("ac4_rev", nppiHistogramEven_8u_AC4R_Ctx(src4.p, W * 4, NppiSize{W, H}, hist, levels, revlo, revhi, gBuf64, ctx()));
  CHK("hist_err");
}

static void probe_wmss_small() {
  NppStreamContext c = ctx();
  Dev<Npp32f> out(4);
  for (int sz : {2, 3, 4, 8, 10, 11, 12, 15, 16, 17}) {
    Dev<Npp8u> s1((size_t)sz * sz), s2((size_t)sz * sz);
    std::vector<Npp8u> a((size_t)sz * sz), b((size_t)sz * sz);
    for (int i = 0; i < sz * sz; ++i) { a[i] = (Npp8u)((i * 7 + 5) & 0xff); b[i] = (Npp8u)((i * 7 + 6) & 0xff); }
    s1.put(a); s2.put(b);
    out.fill(-1e30f);
    NppStatus s = nppiWMSSSIM_8u_C1R_Ctx(s1.p, sz, s2.p, sz, NppiSize{sz, sz}, out.p, gBuf64, c);
    printf("  WMSSSIM %d st=%d val=%.9g\n", sz, (int)s, out.get()[0]);
    out.fill(-1e30f);
    s = nppiMSSSIM_8u_C1R_Ctx(s1.p, sz, s2.p, sz, NppiSize{sz, sz}, out.p, gBuf64, c);
    printf("  MSSSIM  %d st=%d val=%.9g\n", sz, (int)s, out.get()[0]);
  }
  for (int sz : {176, 180, 184, 190, 199, 200}) {
    Dev<Npp8u> s1((size_t)sz * sz), s2((size_t)sz * sz);
    std::vector<Npp8u> a((size_t)sz * sz), b((size_t)sz * sz);
    for (int i = 0; i < sz * sz; ++i) { a[i] = (Npp8u)((i * 7 + 5) & 0xff); b[i] = (Npp8u)((i * 7 + 6) & 0xff); }
    s1.put(a); s2.put(b);
    out.fill(-1e30f);
    NppStatus s = nppiMSSSIM_8u_C1R_Ctx(s1.p, sz, s2.p, sz, NppiSize{sz, sz}, out.p, gBuf64, c);
    printf("  MSSSIM  %d st=%d val=%.9g\n", sz, (int)s, out.get()[0]);
  }
}


static void probe_hist16() {
  const int W=25,H=11;
  std::vector<Npp16u> t((size_t)W*H*3);
  unsigned s=12345;
  for (auto &v: t) { s=s*1103515245u+12345u; v=(Npp16u)(s&0xffff); }
  Dev<Npp16u> src(t.size()); src.put(t);
  int levels[3]={257,129,65};
  Npp32s lo[3]={0,0,1024}, hi[3]={65536,65536,64512};
  Dev<Npp32s> h0(257),h1(129),h2(65);
  h0.fill(-1);h1.fill(-1);h2.fill(-1);
  Npp32s *hist[3]={h0.p,h1.p,h2.p};
  NppStatus st=nppiHistogramEven_16u_C3R_Ctx(src.p,W*3*2,NppiSize{W,H},hist,levels,lo,hi,gBuf64,ctx());
  PS("even16_c3",st);
  CHK("after");
  printf("  totals=%d %d %d\n",[&]{int c=0;for(auto v:h0.get())if(v>0)c+=v;return c;}(),
         [&]{int c=0;for(auto v:h1.get())if(v>0)c+=v;return c;}(),
         [&]{int c=0;for(auto v:h2.get())if(v>0)c+=v;return c;}());
  // try per-channel with separate buffers aligned
  Dev<Npp32s> k0(257),k1(256),k2(256);
  k0.fill(0);k1.fill(0);k2.fill(0);
  Npp32s *h2p[3]={k0.p,k1.p,k2.p};
  int lv2[3]={257,129,65};
  st=nppiHistogramEven_16u_C3R_Ctx(src.p,W*3*2,NppiSize{W,H},h2p,lv2,lo,hi,gBuf64,ctx());
  PS("even16_c3_v2",st); CHK("after2");
}


static void probe_even16s() {
  const int W=32,H=16; // 512 pixels: values 2*i-512 -> -512..510
  for (int mode=0; mode<2; ++mode) {
    std::vector<Npp16s> t((size_t)W*H*4);
    for (int i=0;i<W*H;++i) { Npp16s v=(Npp16s)((i*211)%60001-30000); for(int c=0;c<4;++c) t[(size_t)i*4+c]=v; }
    Dev<Npp16s> src(t.size()); src.put(t);
    Dev<Npp32s> h0(16),h1(16),h2(16); h0.fill(-1);h1.fill(-1);h2.fill(-1);
    Npp32s *hist[3]={h0.p,h1.p,h2.p};
    int levels[3]={17,17,17};
    Npp32s lo[3]={mode? 0:-1000, mode?0:-1000, mode?0:-1000};
    Npp32s hi[3]={mode?2000:1000, mode?2000:1000, mode?2000:1000};
    NppStatus st=nppiHistogramEven_16s_AC4R_Ctx(src.p,W*4*2,NppiSize{W,H},hist,levels,lo,hi,gBuf64,ctx());
    PS(mode?"even16s_pos":"even16s_neg",st); CHK("e16s");
    auto a=h0.get();
    long tot=0; for(auto v:a) if(v>0) tot+=v;
    printf("  ch0 total=%ld bins=[",tot); for(int k=0;k<16;++k) printf("%d ",a[k]); printf("]\n");
    // truth for documented mapping
    long b[16]={0};
    for (int i=0;i<W*H;++i){ int v=(i*211)%60001-30000; if(v>=lo[0]&&v<hi[0]) b[(v-lo[0])*16/(hi[0]-lo[0])]++; }
    printf("  truth=["); for(int k=0;k<16;++k) printf("%ld ",b[k]); printf("]\n");
  }
}


static void probe_hist16b() {
  const int W=25,H=11;
  std::vector<Npp16u> t((size_t)W*H*3);
  unsigned s=987654321u;
  for (auto &v: t) { s=s*1103515245u+12345u; v=(Npp16u)(s&0xffff); }
  Dev<Npp16u> src(t.size()); src.put(t);
  int levels[3]={257,257,257};
  Npp32s lo[3]={0,1024,0}, hi[3]={65536,64512,65536};
  Dev<Npp32s> h0(320),h1(320),h2(320);
  h0.fill(-1);h1.fill(-1);h2.fill(-1);
  Npp32s *hist[3]={h0.p,h1.p,h2.p};
  NppStatus st=nppiHistogramEven_16u_C3R_Ctx(src.p,W*3*2,NppiSize{W,H},hist,levels,lo,hi,gBuf64,ctx());
  PS("even16u_c3",st); CHK("e16u");
  auto a=h0.get(), b=h1.get(), c=h2.get();
  auto sum=[&](std::vector<Npp32s>&v){long s=0;for(int i=0;i<256;++i) if(v[i]>0) s+=v[i]; return s;};
  printf("  sums=%ld %ld %ld\n",sum(a),sum(b),sum(c));
  // truth
  long ta=0,tb=0,tc=0;
  for (int i=0;i<W*H;++i) {
    int v0=t[(size_t)i*3+0], v1=t[(size_t)i*3+1], v2=t[(size_t)i*3+2];
    if (v0>=0&&v0<65536) ta++; if (v1>=1024&&v1<64512) tb++; if (v2>=0&&v2<65536) tc++;
  }
  printf("  truth=%ld %ld %ld\n",ta,tb,tc);
  bool same=true; for(int i=0;i<256;++i) if(a[i]!=c[i]) same=false;
  printf("  ch0==ch2? %d\n",(int)same);
}

static void probe_ssim2() {
  NppStreamContext c = ctx();
  Dev<Npp32f> out(4);
  for (int sz : {16, 64}) {
    Dev<Npp8u> s1((size_t)sz * sz), s2((size_t)sz * sz);
    s1.fill(100);
    s2.fill(110);
    out.fill(-1e30f);
    NppStatus s = nppiSSIM_8u_C1R_Ctx(s1.p, sz, s2.p, sz, NppiSize{sz, sz}, out.p, gBuf64, c);
    printf("  SSIM const %d st=%d val=%.9g\n", sz, (int)s, out.get()[0]);
    out.fill(-1e30f);
    s = nppiMSSSIM_8u_C1R_Ctx(s1.p, sz, s2.p, sz, NppiSize{sz, sz}, out.p, gBuf64, c);
    printf("  MSSSIM const %d st=%d val=%.9g\n", sz, (int)s, out.get()[0]);
    out.fill(-1e30f);
    s = nppiWMSSSIM_8u_C1R_Ctx(s1.p, sz, s2.p, sz, NppiSize{sz, sz}, out.p, gBuf64, c);
    printf("  WMSSSIM const %d st=%d val=%.9g\n", sz, (int)s, out.get()[0]);
  }
  for (int sz : {16, 32, 64, 128, 256}) {
    std::vector<Npp8u> a((size_t)sz * sz), b((size_t)sz * sz);
    for (int y = 0; y < sz; ++y)
      for (int x = 0; x < sz; ++x) {
        a[(size_t)y * sz + x] = (Npp8u)((x * 7 + y * 13 + 5) & 0xff);
        b[(size_t)y * sz + x] = (Npp8u)((x * 7 + y * 13 + 6) & 0xff);
      }
    Dev<Npp8u> s1(a.size()), s2(b.size());
    s1.put(a); s2.put(b);
    out.fill(-1e30f);
    NppStatus s = nppiSSIM_8u_C1R_Ctx(s1.p, sz, s2.p, sz, NppiSize{sz, sz}, out.p, gBuf64, c);
    printf("  SSIM tex %d st=%d val=%.9g\n", sz, (int)s, out.get()[0]);
    out.fill(-1e30f);
    s = nppiMSSSIM_8u_C1R_Ctx(s1.p, sz, s2.p, sz, NppiSize{sz, sz}, out.p, gBuf64, c);
    printf("  MSSSIM tex %d st=%d val=%.9g\n", sz, (int)s, out.get()[0]);
    out.fill(-1e30f);
    s = nppiWMSSSIM_8u_C1R_Ctx(s1.p, sz, s2.p, sz, NppiSize{sz, sz}, out.p, gBuf64, c);
    printf("  WMSSSIM tex %d st=%d val=%.9g\n", sz, (int)s, out.get()[0]);
    std::vector<Npp8u> a3((size_t)sz * sz * 3), b3((size_t)sz * sz * 3);
    for (size_t i = 0; i < a3.size(); ++i) { a3[i] = a[i / 3]; b3[i] = b[i / 3]; }
    Dev<Npp8u> q1(a3.size()), q2(b3.size());
    q1.put(a3); q2.put(b3);
    out.fill(-1e30f);
    s = nppiSSIM_8u_C3R_Ctx(q1.p, sz * 3, q2.p, sz * 3, NppiSize{sz, sz}, out.p, gBuf64, c);
    printf("  SSIM3 tex %d st=%d val=[%.9g %.9g %.9g]\n", sz, (int)s, out.get()[0], out.get()[1], out.get()[2]);
  }
  Dev<Npp8u> a(256), b(256);
  a.fill(1); b.fill(2);
  out.fill(-1e30f);
  PS("ssim_null", nppiSSIM_8u_C1R_Ctx(nullptr, 16, b.p, 16, NppiSize{16, 16}, out.p, gBuf64, c));
  PS("ssim_small", nppiSSIM_8u_C1R_Ctx(a.p, 16, b.p, 16, NppiSize{1, 1}, out.p, gBuf64, c));
  PS("ssim_10", nppiSSIM_8u_C1R_Ctx(a.p, 16, b.p, 16, NppiSize{10, 10}, out.p, gBuf64, c));
  PS("msssim_128", nppiMSSSIM_8u_C1R_Ctx(a.p, 16, b.p, 16, NppiSize{128, 128}, out.p, gBuf64, c));
  PS("wmsssim_1", nppiWMSSSIM_8u_C1R_Ctx(a.p, 16, b.p, 16, NppiSize{1, 1}, out.p, gBuf64, c));
}

int main(int argc, char **argv) {
  if (argc < 2) return 2;
  cudaMalloc(&gBuf64, 64 << 20);
  setbuf(stdout, nullptr);
  const char *p = argv[1];
  if (!strcmp(p, "qidx3")) probe_qidx3();
  else if (!strcmp(p, "cir_ac4")) probe_cir_ac4();
  else if (!strcmp(p, "hist3")) probe_hist3();
  else if (!strcmp(p, "rect2")) probe_rect2();
  else if (!strcmp(p, "levels")) probe_levels2();
  else if (!strcmp(p, "radial")) probe_radial_imp();
  else if (!strcmp(p, "ssim2")) probe_ssim2();
  else if (!strcmp(p, "hist_err")) probe_hist_err();
  else if (!strcmp(p, "hist16")) probe_hist16();
  else if (!strcmp(p, "even16s")) probe_even16s();
  else if (!strcmp(p, "hist16b")) probe_hist16b();
  else if (!strcmp(p, "wmss_small")) probe_wmss_small();
  else if (!strcmp(p, "radial_scan")) probe_radial_scan(24);
  else { fprintf(stderr, "unknown probe %s\n", p); return 2; }
  fflush(stdout);
  return 0;
}
