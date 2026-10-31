// Standalone NVIDIA NPP 13.4 semantics probe for package s2 (statistics).
// Runs each probe in a forked child so device/host faults do not kill the run.
#include <cuda_runtime.h>
#include <npp.h>
#include <nppi.h>
// NPP 13 removed nppGetStreamContext from the public headers (libnppc still
// exports it); declare it like the shared test framework shim does.
extern "C" {
NppStatus nppGetStreamContext(NppStreamContext *pNppStreamContext);
}
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const char *sname(NppStatus s) {
  switch (s) {
    case NPP_SUCCESS: return "NPP_SUCCESS";
    case NPP_NULL_POINTER_ERROR: return "NPP_NULL_POINTER_ERROR";
    case NPP_SIZE_ERROR: return "NPP_SIZE_ERROR";
    case NPP_STEP_ERROR: return "NPP_STEP_ERROR";
    case NPP_NOT_EVEN_STEP_ERROR: return "NPP_NOT_EVEN_STEP_ERROR";
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
#define PS(tag, s) printf("  %-14s status=%d (%s)\n", tag, (int)(s), sname(s))
#define CHK(tag) do { cudaError_t e = cudaDeviceSynchronize(); \
  printf("  %-14s cuda=%s\n", tag, cudaGetErrorString(e)); (void)cudaGetLastError(); } while (0)

static Npp8u *gBuf64 = nullptr;
static NppStreamContext ctx() { NppStreamContext c{}; nppGetStreamContext(&c); return c; }

template <typename T> struct Dev {
  T *p = nullptr; size_t n = 0;
  explicit Dev(size_t count) : n(count) { cudaMalloc(&p, count * sizeof(T)); }
  ~Dev() { if (p) cudaFree(p); }
  void put(const std::vector<T> &h) { cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice); }
  std::vector<T> get() const { std::vector<T> h(n); cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost); return h; }
  void fill(T v) { std::vector<T> h(n, v); put(h); }
};

// deterministic textures
static std::vector<Npp8u> tex8(int w, int h, int ch, unsigned seed) {
  std::vector<Npp8u> v((size_t)w * h * ch);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      for (int c = 0; c < ch; ++c)
        v[((size_t)y * w + x) * ch + c] = (Npp8u)((x * 7 + y * 13 + c * 29 + seed) & 0xff);
  return v;
}
static std::vector<Npp32f> texf(int w, int h, int ch, unsigned seed) {
  std::vector<Npp32f> v((size_t)w * h * ch);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      for (int c = 0; c < ch; ++c)
        v[((size_t)y * w + x) * ch + c] = (float)((int)((x * 7 + y * 13 + c * 29 + seed) & 0xff) - 128) / 64.0f;
  return v;
}

// ======================= CountInRange array conventions =====================
static void probe_cir(const char *mode) {
  const int W = 8, H = 4, CH = 3;
  Dev<Npp8u> src((size_t)W * H * CH);
  src.put(tex8(W, H, CH, 1));
  // per-channel bounds: ch0 [10,100], ch1 [0,10], ch2 [200,255]
  Npp8u loHost[3] = {10, 0, 200}, hiHost[3] = {100, 10, 255};
  Dev<Npp8u> loD(3), hiD(3);
  loD.put(std::vector<Npp8u>(loHost, loHost + 3));
  hiD.put(std::vector<Npp8u>(hiHost, hiHost + 3));
  Dev<int> countsD(3);
  countsD.fill(-12345);
  int countsHost[3] = {-12345, -12345, -12345};
  NppStreamContext c = ctx();
  NppStatus s;
  if (!strcmp(mode, "dev_all"))
    s = nppiCountInRange_8u_C3R_Ctx(src.p, W * CH, NppiSize{W, H}, countsD.p, loD.p, hiD.p, gBuf64, c);
  else if (!strcmp(mode, "devc_hostb"))
    s = nppiCountInRange_8u_C3R_Ctx(src.p, W * CH, NppiSize{W, H}, countsD.p, loHost, hiHost, gBuf64, c);
  else if (!strcmp(mode, "host_all"))
    s = nppiCountInRange_8u_C3R_Ctx(src.p, W * CH, NppiSize{W, H}, countsHost, loHost, hiHost, gBuf64, c);
  else { printf("unknown mode\n"); return; }
  PS("call", s);
  std::vector<int> dc = countsD.get();
  printf("  devCounts=[%d %d %d] hostCounts=[%d %d %d]\n", dc[0], dc[1], dc[2], countsHost[0], countsHost[1],
         countsHost[2]);
  CHK("after");
  // ground truth
  std::vector<Npp8u> t = tex8(W, H, CH, 1);
  int gt[3] = {0, 0, 0};
  for (size_t i = 0; i < t.size(); ++i)
    if (t[i] >= loHost[i % 3] && t[i] <= hiHost[i % 3]) gt[i % 3]++;
  printf("  groundTruth(inclusive)=[%d %d %d]\n", gt[0], gt[1], gt[2]);
  int gte[3] = {0, 0, 0};
  for (size_t i = 0; i < t.size(); ++i)
    if (t[i] > loHost[i % 3] && t[i] < hiHost[i % 3]) gte[i % 3]++;
  printf("  groundTruth(exclusive)=[%d %d %d]\n", gte[0], gte[1], gte[2]);
}

// CountInRange bound semantics on C1R
static void probe_cir_bounds() {
  // image values 1..4 in a 4x1
  std::vector<Npp8u> v = {1, 2, 3, 4};
  Dev<Npp8u> src(4); src.put(v);
  Dev<int> cnt(1);
  NppStreamContext c = ctx();
  struct { Npp8u lo, hi; const char *tag; } cases[] = {
      {2, 3, "b23"}, {3, 3, "b33"}, {3, 2, "rev32"}, {1, 4, "b14"}, {5, 9, "none"}, {0, 255, "all"}};
  for (auto &k : cases) {
    cnt.fill(-1);
    NppStatus s = nppiCountInRange_8u_C1R_Ctx(src.p, 4, NppiSize{4, 1}, cnt.p, k.lo, k.hi, gBuf64, c);
    printf("  %-6s lo=%d hi=%d ", k.tag, k.lo, k.hi); PS("st", s);
    printf("        count=%d\n", cnt.get()[0]);
  }
  // 32f version with negative bounds
  std::vector<Npp32f> fv = {-2.0f, -1.0f, 0.0f, 1.5f};
  Dev<Npp32f> fsrc(4); fsrc.put(fv);
  Dev<int> fcnt(1);
  struct { float lo, hi; const char *tag; } fc[] = {{-2.0f, 0.0f, "f[-2,0]"}, {0.0f, 1.5f, "f[0,1.5]"},
                                                    {1.5f, -2.0f, "frev"}, {0.25f, 0.25f, "fmid"}};
  for (auto &k : fc) {
    fcnt.fill(-1);
    NppStatus s = nppiCountInRange_32f_C1R_Ctx(fsrc.p, 16, NppiSize{4, 1}, fcnt.p, k.lo, k.hi, gBuf64, c);
    printf("  %-9s ", k.tag); PS("st", s);
    printf("        count=%d\n", fcnt.get()[0]);
  }
}

// ======================= QualityIndex arity & value =========================
static void probe_qidx() {
  const int W = 17, H = 13;
  NppStreamContext c = ctx();
  Dev<Npp32f> out(4);
  {  // C3R arity probe: sentinel prefill
    Dev<Npp8u> s1((size_t)W * H * 3), s2((size_t)W * H * 3);
    s1.put(tex8(W, H, 3, 1));
    s2.put(tex8(W, H, 3, 2));
    out.fill(-1e30f);
    NppStatus s = nppiQualityIndex_8u32f_C3R_Ctx(s1.p, W * 3, s2.p, W * 3, NppiSize{W, H}, out.p, gBuf64, c);
    PS("C3R", s);
    auto v = out.get();
    printf("  out=[%g %g %g %g]\n", v[0], v[1], v[2], v[3]);
    CHK("C3R");
  }
  {  // AC4R arity + alpha ignore
    Dev<Npp8u> s1((size_t)W * H * 4), s2((size_t)W * H * 4);
    std::vector<Npp8u> a = tex8(W, H, 3, 1), b = tex8(W, H, 3, 1);
    std::vector<Npp8u> pa((size_t)W * H * 4), pb((size_t)W * H * 4);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        for (int k = 0; k < 4; ++k) {
          pa[((size_t)y * W + x) * 4 + k] = (k == 3) ? 0 : a[((size_t)y * W + x) * 3 + k];
          pb[((size_t)y * W + x) * 4 + k] = (k == 3) ? 255 : b[((size_t)y * W + x) * 3 + k];
        }
    s1.put(pa); s2.put(pb);
    out.fill(-1e30f);
    NppStatus s = nppiQualityIndex_8u32f_AC4R_Ctx(s1.p, W * 4, s2.p, W * 4, NppiSize{W, H}, out.p, gBuf64, c);
    PS("AC4R", s);
    auto v = out.get();
    printf("  out=[%g %g %g %g] (identical RGB, alpha differs)\n", v[0], v[1], v[2], v[3]);
    CHK("AC4R");
  }
  {  // hand-computed 4x1 case
    Dev<Npp8u> s1(4), s2(4);
    s1.put(std::vector<Npp8u>{1, 2, 3, 4});
    s2.put(std::vector<Npp8u>{1, 2, 3, 5});
    out.fill(-1e30f);
    NppStatus s = nppiQualityIndex_8u32f_C1R_Ctx(s1.p, 4, s2.p, 4, NppiSize{4, 1}, out.p, gBuf64, c);
    PS("formula", s);
    printf("  out=[%g %g] (expected 1.1040..)\n", out.get()[0], out.get()[1]);
  }
  {  // constant images
    Dev<Npp8u> s1(16), s2(16);
    s1.fill(7); s2.put(tex8(4, 4, 1, 3));
    out.fill(-1e30f);
    NppStatus s = nppiQualityIndex_8u32f_C1R_Ctx(s1.p, 4, s2.p, 4, NppiSize{4, 4}, out.p, gBuf64, c);
    PS("const", s);
    printf("  out=[%g]\n", out.get()[0]);
  }
  {  // 1x1
    Dev<Npp8u> s1(1), s2(1);
    s1.fill(3); s2.fill(9);
    out.fill(-1e30f);
    NppStatus s = nppiQualityIndex_8u32f_C1R_Ctx(s1.p, 1, s2.p, 1, NppiSize{1, 1}, out.p, gBuf64, c);
    PS("1x1", s);
    printf("  out=[%g]\n", out.get()[0]);
  }
}

// ======================= MSE / PSNR values =================================
static void probe_mse_psnr() {
  NppStreamContext c = ctx();
  Dev<Npp32f> out(6);
  const int W = 16, H = 8;
  {  // identical images -> MSE 0, PSNR ?
    Dev<Npp8u> s((size_t)W * H);
    s.put(tex8(W, H, 1, 5));
    out.fill(-1e30f);
    NppStatus s1 = nppiMSE_8u_C1R_Ctx(s.p, W, s.p, W, NppiSize{W, H}, out.p, gBuf64, c);
    PS("MSE_identical", s1);
    auto v = out.get();
    printf("  mse=[%g %g]\n", v[0], v[1]);
    out.fill(-1e30f);
    NppStatus s2 = nppiPSNR_8u_C1R_Ctx(s.p, W, s.p, W, NppiSize{W, H}, out.p, gBuf64, c);
    PS("PSNR_identical", s2);
    v = out.get();
    printf("  psnr=[%g %g]\n", v[0], v[1]);
    CHK("identical");
  }
  {  // known diff: src2 = src1+2 -> MSE=4, PSNR=20*log10(255/2)=42.1246
    std::vector<Npp8u> a = tex8(W, H, 1, 5), b(a);
    for (auto &v : b) v = (Npp8u)(v + 2);
    Dev<Npp8u> s1((size_t)W * H), s2((size_t)W * H);
    s1.put(a); s2.put(b);
    out.fill(-1e30f);
    NppStatus s1s = nppiMSE_8u_C1R_Ctx(s1.p, W, s2.p, W, NppiSize{W, H}, out.p, gBuf64, c);
    PS("MSE_plus2", s1s);
    auto v = out.get();
    printf("  mse=[%g %g] (expect 4.0)\n", v[0], v[1]);
    out.fill(-1e30f);
    NppStatus s2s = nppiPSNR_8u_C1R_Ctx(s1.p, W, s2.p, W, NppiSize{W, H}, out.p, gBuf64, c);
    PS("PSNR_plus2", s2s);
    v = out.get();
    printf("  psnr=[%g %g] (20log10(255/2)=%g)\n", v[0], v[1], 20.0 * log10(255.0 / 2.0));
    CHK("plus2");
  }
  {  // C3R arity: per-channel values, 4th float stays sentinel?
    Dev<Npp8u> s1((size_t)W * H * 3), s2((size_t)W * H * 3);
    s1.put(tex8(W, H, 3, 5));
    std::vector<Npp8u> b = tex8(W, H, 3, 9);
    for (size_t i = 0; i < b.size(); ++i) b[i] = (Npp8u)(b[i] + (i % 3) * 4);
    s2.put(b);
    out.fill(-1e30f);
    NppStatus s = nppiMSE_8u_C3R_Ctx(s1.p, W * 3, s2.p, W * 3, NppiSize{W, H}, out.p, gBuf64, c);
    PS("MSE_C3R", s);
    auto v = out.get();
    printf("  mse=[%g %g %g %g]\n", v[0], v[1], v[2], v[3]);
    out.fill(-1e30f);
    s = nppiPSNR_8u_C3R_Ctx(s1.p, W * 3, s2.p, W * 3, NppiSize{W, H}, out.p, gBuf64, c);
    PS("PSNR_C3R", s);
    v = out.get();
    printf("  psnr=[%g %g %g %g]\n", v[0], v[1], v[2], v[3]);
    CHK("C3R");
  }
  {  // 32f MSE with non-integer values; is it mean of squared diff?
    std::vector<Npp32f> a = texf(W, H, 1, 5);
    Dev<Npp32f> s1((size_t)W * H), s2((size_t)W * H);
    s1.put(a);
    std::vector<Npp32f> b(a);
    for (auto &v : b) v += 1.0f;
    s2.put(b);
    out.fill(-1e30f);
    NppStatus s = nppiMSE_8u_C1R_Ctx(nullptr, 0, nullptr, 0, NppiSize{0, 0}, nullptr, nullptr, c);  // placeholder
    (void)s;
    // use the real 8u API above; 32f MSE does not exist in this package
    CHK("32f");
  }
}

// ======================= SSIM / MSSSIM / WMSSSIM ============================
static void probe_ssim() {
  NppStreamContext c = ctx();
  Dev<Npp32f> out(6);
  struct Sz { int w, h; };
  Sz sizes[] = {{16, 16}, {32, 32}, {64, 64}, {256, 256}, {8, 8}, {5, 5}, {4, 4}, {1, 1}, {2, 2}};
  for (auto &z : sizes) {
    Dev<Npp8u> s1((size_t)z.w * z.h), s2((size_t)z.w * z.h);
    s1.put(tex8(z.w, z.h, 1, 5));
    s2.put(tex8(z.w, z.h, 1, 6));
    out.fill(-1e30f);
    NppStatus s = nppiSSIM_8u_C1R_Ctx(s1.p, z.w, s2.p, z.w, NppiSize{z.w, z.h}, out.p, gBuf64, c);
    printf("  SSIM %dx%d ", z.w, z.h); PS("st", s);
    printf("        val=[%g %g]\n", out.get()[0], out.get()[1]);
    CHK("ssim");
    // identical
    out.fill(-1e30f);
    s = nppiSSIM_8u_C1R_Ctx(s1.p, z.w, s1.p, z.w, NppiSize{z.w, z.h}, out.p, gBuf64, c);
    printf("        identical val=[%g]\n", out.get()[0]);
    CHK("ssim_id");
  }
  {  // constant vs constant and constant vs different constant
    Dev<Npp8u> s1(256), s2(256);
    s1.fill(100); s2.fill(100);
    out.fill(-1e30f);
    NppStatus s = nppiSSIM_8u_C1R_Ctx(s1.p, 16, s2.p, 16, NppiSize{16, 16}, out.p, gBuf64, c);
    PS("SSIM_const_same", s);
    printf("  val=[%g]\n", out.get()[0]);
    s2.fill(110);
    out.fill(-1e30f);
    s = nppiSSIM_8u_C1R_Ctx(s1.p, 16, s2.p, 16, NppiSize{16, 16}, out.p, gBuf64, c);
    PS("SSIM_const_diff", s);
    printf("  val=[%g] (formula (2*a*b+C)/(a^2+b^2+C)=%.9g)\n", out.get()[0],
           (2.0 * 100 * 110 + 6.5025) / (100.0 * 100 + 110.0 * 110 + 58.5225));
  }
  {  // C3R arity
    Dev<Npp8u> s1((size_t)64 * 64 * 3), s2((size_t)64 * 64 * 3);
    s1.put(tex8(64, 64, 3, 5));
    s2.put(tex8(64, 64, 3, 6));
    out.fill(-1e30f);
    NppStatus s = nppiSSIM_8u_C3R_Ctx(s1.p, 64 * 3, s2.p, 64 * 3, NppiSize{64, 64}, out.p, gBuf64, c);
    PS("SSIM_C3R", s);
    auto v = out.get();
    printf("  val=[%g %g %g %g]\n", v[0], v[1], v[2], v[3]);
    CHK("c3r");
  }
}

static void probe_msssim() {
  NppStreamContext c = ctx();
  Dev<Npp32f> out(4);
  struct Sz { int w, h; };
  Sz sizes[] = {{256, 256}, {64, 64}, {32, 32}, {17, 17}, {16, 16}, {15, 15}, {8, 8}, {4, 4}, {1, 1}};
  for (auto &z : sizes) {
    Dev<Npp8u> s1((size_t)z.w * z.h), s2((size_t)z.w * z.h);
    s1.put(tex8(z.w, z.h, 1, 5));
    s2.put(tex8(z.w, z.h, 1, 6));
    out.fill(-1e30f);
    NppStatus s = nppiMSSSIM_8u_C1R_Ctx(s1.p, z.w, s2.p, z.w, NppiSize{z.w, z.h}, out.p, gBuf64, c);
    printf("  MSSSIM %dx%d ", z.w, z.h); PS("st", s);
    printf("        val=[%g]\n", out.get()[0]);
    CHK("msssim");
    out.fill(-1e30f);
    s = nppiWMSSSIM_8u_C1R_Ctx(s1.p, z.w, s2.p, z.w, NppiSize{z.w, z.h}, out.p, gBuf64, c);
    printf("  WMSSSIM %dx%d ", z.w, z.h); PS("st", s);
    printf("        val=[%g]\n", out.get()[0]);
    CHK("wmsssim");
    out.fill(-1e30f);
    s = nppiWMSSSIM_8u_C3R_Ctx(s1.p, z.w * 3, s2.p, z.w * 3, NppiSize{z.w, z.h}, out.p, gBuf64, c);
    printf("  WMSSSIMC3 %dx%d ", z.w, z.h); PS("st", s);
    auto v = out.get();
    printf("        val=[%g %g %g]\n", v[0], v[1], v[2]);
    CHK("wmsssimc3");
  }
  {  // identical images
    Dev<Npp8u> s1((size_t)256 * 256);
    s1.put(tex8(256, 256, 1, 5));
    out.fill(-1e30f);
    NppStatus s = nppiMSSSIM_8u_C1R_Ctx(s1.p, 256, s1.p, 256, NppiSize{256, 256}, out.p, gBuf64, c);
    PS("MSSSIM_identical", s);
    printf("  val=[%g]\n", out.get()[0]);
    out.fill(-1e30f);
    s = nppiWMSSSIM_8u_C1R_Ctx(s1.p, 256, s1.p, 256, NppiSize{256, 256}, out.p, gBuf64, c);
    PS("WMSSSIM_identical", s);
    printf("  val=[%g]\n", out.get()[0]);
  }
}

// ======================= Histogram C3R / AC4R ===============================
static void probe_hist() {
  NppStreamContext c = ctx();
  const int W = 32, H = 16;
  {  // Even 8u C3R: device pHist[3], host levels arrays
    Dev<Npp8u> src((size_t)W * H * 3);
    src.put(tex8(W, H, 3, 1));
    Dev<Npp32s> h0(256), h1(256), h2(256);
    h0.fill(-1); h1.fill(-1); h2.fill(-1);
    Npp32s *hist[3] = {h0.p, h1.p, h2.p};
    int levels[3] = {257, 257, 257};
    Npp32s lo[3] = {0, 0, 0}, hi[3] = {256, 256, 256};
    NppStatus s = nppiHistogramEven_8u_C3R_Ctx(src.p, W * 3, NppiSize{W, H}, hist, levels, lo, hi, gBuf64, c);
    PS("Even_8u_C3R", s);
    CHK("even_c3r");
    // print a few bins + total
    auto a = h0.get(), b = h1.get(), d = h2.get();
    long ta = 0, tb = 0, td = 0;
    for (int i = 0; i < 256; ++i) { ta += a[i]; tb += b[i]; td += d[i]; }
    printf("  totals=[%ld %ld %ld] (expect %d each) bins10..14 ch0=[%d %d %d %d %d]\n", ta, tb, td, W * H, a[10],
           a[11], a[12], a[13], a[14]);
    // narrow range: lo=64 hi=192 with 3 levels (2 bins) pins boundary semantics
    Dev<Npp32s> k0(2), k1(2), k2(2);
    k0.fill(-1); k1.fill(-1); k2.fill(-1);
    Npp32s *hist2[3] = {k0.p, k1.p, k2.p};
    int levels2[3] = {3, 3, 3};
    Npp32s lo2[3] = {64, 64, 64}, hi2[3] = {192, 192, 192};
    s = nppiHistogramEven_8u_C3R_Ctx(src.p, W * 3, NppiSize{W, H}, hist2, levels2, lo2, hi2, gBuf64, c);
    PS("Even_8u_C3R_3lvl", s);
    auto x0 = k0.get(), x1 = k1.get(), x2 = k2.get();
    printf("  bins=[%d %d] [%d %d] [%d %d] (level step=(192-64)/2=64: bin0=[64,128) bin1=[128,192])\n", x0[0], x0[1],
           x1[0], x1[1], x2[0], x2[1]);
    std::vector<Npp8u> t = tex8(W, H, 3, 1);
    long e0 = 0, e0in = 0, e1 = 0, e1in = 0;
    for (size_t i = 0; i < t.size(); i += 3) {
      if (t[i] >= 64 && t[i] < 128) e0++;
      if (t[i] == 128) e0in++;
      if (t[i] >= 128 && t[i] <= 192) e1++;
      if (t[i] == 192) e1in++;
    }
    printf("  truth ch0: [64,128)=%ld incl128=%ld [128,192]=%ld incl192=%ld\n", e0, e0in, e1, e1in);
  }
  {  // Even 8u AC4R: alpha ignored
    Dev<Npp8u> src((size_t)W * H * 4);
    std::vector<Npp8u> t = tex8(W, H, 4, 1);
    for (size_t i = 3; i < t.size(); i += 4) t[i] = 255;  // alpha constant
    src.put(t);
    Dev<Npp32s> h0(256), h1(256), h2(256);
    h0.fill(-1); h1.fill(-1); h2.fill(-1);
    Npp32s *hist[3] = {h0.p, h1.p, h2.p};
    int levels[3] = {257, 257, 257};
    Npp32s lo[3] = {0, 0, 0}, hi[3] = {256, 256, 256};
    NppStatus s = nppiHistogramEven_8u_AC4R_Ctx(src.p, W * 4, NppiSize{W, H}, hist, levels, lo, hi, gBuf64, c);
    PS("Even_8u_AC4R", s);
    CHK("even_ac4");
    auto a = h0.get(), b = h1.get(), d = h2.get();
    long ta = 0;
    for (int i = 0; i < 256; ++i) ta += a[i];
    printf("  ch0 total=%ld (expect %d)\n", ta, W * H);
  }
  {  // Range 8u C3R: device level arrays
    Dev<Npp8u> src((size_t)W * H * 3);
    src.put(tex8(W, H, 3, 1));
    Dev<Npp32s> h0(4), h1(4), h2(4);
    h0.fill(-1); h1.fill(-1); h2.fill(-1);
    Npp32s *hist[3] = {h0.p, h1.p, h2.p};
    std::vector<Npp32s> lv = {0, 64, 128, 192, 256};
    Dev<Npp32s> l0(5), l1(5), l2(5);
    l0.put(lv); l1.put(lv); l2.put(lv);
    const Npp32s *levels[3] = {l0.p, l1.p, l2.p};
    int nLevels[3] = {5, 5, 5};
    NppStatus s = nppiHistogramRange_8u_C3R_Ctx(src.p, W * 3, NppiSize{W, H}, hist, levels, nLevels, gBuf64, c);
    PS("Range_8u_C3R", s);
    CHK("range_c3r");
    auto x0 = h0.get();
    printf("  bins=[%d %d %d %d] (boundaries 0/64/128/192/256)\n", x0[0], x0[1], x0[2], x0[3]);
  }
  {  // Range 32f C3R
    Dev<Npp32f> src((size_t)W * H * 3);
    src.put(texf(W, H, 3, 1));
    Dev<Npp32s> h0(3), h1(3), h2(3);
    h0.fill(-1); h1.fill(-1); h2.fill(-1);
    Npp32s *hist[3] = {h0.p, h1.p, h2.p};
    std::vector<Npp32f> lv = {-2.0f, 0.0f, 2.0f};
    Dev<Npp32f> l0(3), l1(3), l2(3);
    l0.put(lv); l1.put(lv); l2.put(lv);
    const Npp32f *levels[3] = {l0.p, l1.p, l2.p};
    int nLevels[3] = {3, 3, 3};
    NppStatus s = nppiHistogramRange_32f_C3R_Ctx(src.p, W * 3 * 4, NppiSize{W, H}, hist, levels, nLevels, gBuf64, c);
    PS("Range_32f_C3R", s);
    CHK("range32f");
    auto x0 = h0.get();
    printf("  bins=[%d %d] (levels -2/0/2)\n", x0[0], x0[1]);
    // ground truth
    std::vector<Npp32f> t = texf(W, H, 3, 1);
    long a = 0, b = 0, out = 0;
    for (size_t i = 0; i < t.size(); i += 3) {
      if (t[i] >= -2.0f && t[i] < 0.0f) a++;
      else if (t[i] >= 0.0f && t[i] < 2.0f) b++;
      else out++;
    }
    printf("  truth=[%ld %ld] out_of_range=%ld\n", a, b, out);
  }
}

// ======================= Radial profiles ====================================
static void probe_radial() {
  NppStreamContext c = ctx();
  const int W = 33, H = 33;
  // radial pattern centered at (16,16): value = distance
  std::vector<Npp32f> img((size_t)W * H);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      float dx = x - 16.5f, dy = y - 16.5f;
      img[(size_t)y * W + x] = sqrtf(dx * dx + dy * dy);
    }
  Dev<Npp32f> src((size_t)W * H);
  src.put(img);
  Dev<NppiProfileData> prof(258);
  for (int samples : {8, 16, 258}) {
    prof.fill(NppiProfileData{0, -1e30f, -1e30f});
    NppStatus s = nppiCircularRadialProfile_32f_C1R_Ctx(src.p, W * 4, NppiSize{W, H},
                                                        NppiPoint32f{16.5f, 16.5f}, prof.p, (Npp16u)samples, c);
    printf("  radial samples=%d ", samples); PS("st", s);
    auto v = prof.get();
    for (int i = 0; i < (samples > 8 ? 8 : samples); ++i)
      printf("    [%d] nPix=%d mean=%g std=%g\n", i, v[i].nPixels, v[i].nMeanIntensity, v[i].nStdDevIntensity);
    CHK("radial");
  }
  {  // center offset from middle
    prof.fill(NppiProfileData{0, -1e30f, -1e30f});
    NppStatus s = nppiCircularRadialProfile_32f_C1R_Ctx(src.p, W * 4, NppiSize{W, H}, NppiPoint32f{2.0f, 3.0f},
                                                        prof.p, 16, c);
    PS("radial_offcenter", s);
    auto v = prof.get();
    for (int i = 0; i < 6; ++i)
      printf("    [%d] nPix=%d mean=%g std=%g\n", i, v[i].nPixels, v[i].nMeanIntensity, v[i].nStdDevIntensity);
    CHK("offcenter");
  }
  {  // elliptical with ratio 2.0, orientation 0
    prof.fill(NppiProfileData{0, -1e30f, -1e30f});
    NppStatus s = nppiEllipticalRadialProfile_32f_C1R_Ctx(src.p, W * 4, NppiSize{W, H}, NppiPoint32f{16.5f, 16.5f},
                                                           2.0f, 0.0f, prof.p, 16, c);
    PS("elliptical", s);
    auto v = prof.get();
    for (int i = 0; i < 6; ++i)
      printf("    [%d] nPix=%d mean=%g std=%g\n", i, v[i].nPixels, v[i].nMeanIntensity, v[i].nStdDevIntensity);
    CHK("elliptical");
  }
  {  // orientation 45deg
    prof.fill(NppiProfileData{0, -1e30f, -1e30f});
    NppStatus s = nppiEllipticalRadialProfile_32f_C1R_Ctx(src.p, W * 4, NppiSize{W, H}, NppiPoint32f{16.5f, 16.5f},
                                                           2.0f, 0.7853981633974483f, prof.p, 16, c);
    PS("elliptical_45", s);
    auto v = prof.get();
    for (int i = 0; i < 4; ++i)
      printf("    [%d] nPix=%d mean=%g std=%g\n", i, v[i].nPixels, v[i].nMeanIntensity, v[i].nStdDevIntensity);
    CHK("ell45");
  }
}

// ======================= RectStdDev variants ================================
static void probe_rect() {
  NppStreamContext c = ctx();
  // original 12x12 image; integral images are 13x13 (zero first row/col)
  const int OW = 12, OH = 12;
  const int IW = OW + 1, IH = OH + 1;
  std::vector<double> img((size_t)OW * OH);
  for (int y = 0; y < OH; ++y)
    for (int x = 0; x < OW; ++x) img[(size_t)y * OW + x] = (double)((x * 5 + y * 3) % 7);
  auto integ = [&](int w, int h) {
    std::vector<double> I((size_t)w * h, 0.0), S((size_t)w * h, 0.0);
    for (int y = 1; y < h; ++y)
      for (int x = 1; x < w; ++x) {
        double v = img[(size_t)(y - 1) * OW + (x - 1)];
        I[(size_t)y * w + x] = v + I[(size_t)(y - 1) * w + x] + I[(size_t)y * w + x - 1] - I[(size_t)(y - 1) * w + x - 1];
        double q = v * v;
        S[(size_t)y * w + x] = q + S[(size_t)(y - 1) * w + x] + S[(size_t)y * w + x - 1] - S[(size_t)(y - 1) * w + x - 1];
      }
    return std::make_pair(I, S);
  };
  auto IS = integ(IW, IH);
  std::vector<Npp32f> If((size_t)IW * IH), Sf((size_t)IW * IH);
  std::vector<Npp32s> Ii((size_t)IW * IH), Si((size_t)IW * IH);
  std::vector<Npp64f> Sd((size_t)IW * IH);
  for (size_t i = 0; i < (size_t)IW * IH; ++i) {
    If[i] = (Npp32f)IS.first[i];
    Sf[i] = (Npp32f)IS.second[i];
    Ii[i] = (Npp32s)IS.first[i];
    Si[i] = (Npp32s)IS.second[i];
    Sd[i] = IS.second[i];
  }
  Dev<Npp32f> dIf((size_t)IW * IH), dSf((size_t)IW * IH);
  Dev<Npp32s> dIi((size_t)IW * IH), dSi((size_t)IW * IH);
  Dev<Npp64f> dSd((size_t)IW * IH);
  dIf.put(If); dSf.put(Sf); dIi.put(Ii); dSi.put(Si); dSd.put(Sd);
  const int W = 8, H = 8;  // ROI
  NppiRect rect{1, 1, 3, 3};
  Dev<Npp32f> dstf((size_t)W * H);
  Dev<Npp32s> dsti((size_t)W * H);
  NppStatus s;
  {  // 32f variant with proper integral images
    dstf.fill(-1e30f);
    s = nppiRectStdDev_32f_C1R_Ctx(dIf.p, IW * 4, dSd.p, IW * 8, dstf.p, W * 4, NppiSize{W, H}, rect, c);
    PS("RectStdDev_32f", s);
    auto vf = dstf.get();
    printf("  dst row0: %g %g %g %g row1: %g %g %g\n", vf[0], vf[1], vf[2], vf[3], vf[W], vf[W + 1], vf[W + 2]);
  }
  {  // 32f variant with 32f squared integral (wrong dtype but steps right)
    dstf.fill(-1e30f);
    s = nppiRectStdDev_32f_C1R_Ctx(dIf.p, IW * 4, (const Npp64f *)dSf.p, IW * 8, dstf.p, W * 4, NppiSize{W, H}, rect, c);
    PS("RectStdDev_32f_32fSqr", s);
  }
  {  // 32s Sfs variant
    dsti.fill(-12345);
    s = nppiRectStdDev_32s_C1RSfs_Ctx(dIi.p, IW * 4, dSi.p, IW * 4, dsti.p, W * 4, NppiSize{W, H}, rect, 0, c);
    PS("RectStdDev_32sSfs0", s);
    auto vi = dsti.get();
    printf("  dst row0: %d %d %d %d row1: %d %d %d\n", vi[0], vi[1], vi[2], vi[3], vi[W], vi[W + 1], vi[W + 2]);
  }
  {  // 32s Sfs with scale 2
    dsti.fill(-12345);
    s = nppiRectStdDev_32s_C1RSfs_Ctx(dIi.p, IW * 4, dSi.p, IW * 4, dsti.p, W * 4, NppiSize{W, H}, rect, 2, c);
    PS("RectStdDev_32sSfs2", s);
    auto vi = dsti.get();
    printf("  dst row0: %d %d %d %d row1: %d %d %d\n", vi[0], vi[1], vi[2], vi[3], vi[W], vi[W + 1], vi[W + 2]);
  }
  // reference: window over ORIGINAL image at (x+rect.x, y+rect.y), size rect
  printf("  ref rows 0..1:");
  for (int py = 0; py < 2; ++py) {
    double m1 = 0, m2 = 0;
    int n = rect.width * rect.height;
    for (int dy = 0; dy < rect.height; ++dy)
      for (int dx = 0; dx < rect.width; ++dx) {
        double v = img[(size_t)(py + rect.y + dy) * OW + (rect.x + dx)];
        m1 += v;
        m2 += v * v;
      }
    double var = m2 / n - (m1 / n) * (m1 / n);
    if (var < 0) var = 0;
    printf(" [row%d sqrt=%.9g]", py, sqrt(var));
  }
  printf("\n");
  // step validation: too-small src step for 13-wide integral (needs 52)
  {
    dstf.fill(-1e30f);
    s = nppiRectStdDev_32f_C1R_Ctx(dIf.p, 32, dSd.p, IW * 8, dstf.p, W * 4, NppiSize{W, H}, rect, c);
    PS("32f_shortSrcStep", s);
    dstf.fill(-1e30f);
    s = nppiRectStdDev_32f_C1R_Ctx(dIf.p, IW * 4, dSd.p, 64, dstf.p, W * 4, NppiSize{W, H}, rect, c);
    PS("32f_shortSqrStep", s);
    dstf.fill(-1e30f);
    s = nppiRectStdDev_32f_C1R_Ctx(dIf.p, IW * 4, dSd.p, IW * 8, dstf.p, 16, NppiSize{W, H}, rect, c);
    PS("32f_shortDstStep", s);
    dstf.fill(-1e30f);
    s = nppiRectStdDev_32f_C1R_Ctx(dIf.p, 36, dSd.p, IW * 8, dstf.p, W * 4, NppiSize{W, H}, rect, c);
    PS("32f_oddSrcStep", s);
  }
  CHK("rect");
}

// ======================= batch metrics ======================================
static void probe_batch(int stage) {
  NppStreamContext c = ctx();
  const int N = 3;
  int w[N] = {16, 24, 32}, h[N] = {8, 12, 16};
  NppiImageDescriptor d1[N], d2[N];
  NppiBufferDescriptor bufs[N];
  Dev<Npp8u> pool(64 * 64 * 2 * N);
  // lay out images sequentially
  size_t off = 0;
  std::vector<Npp8u> blob(pool.n);
  for (int i = 0; i < N; ++i) {
    std::vector<Npp8u> a = tex8(w[i], h[i], 1, 10 + i), b = tex8(w[i], h[i], 1, 20 + i);
    memcpy(blob.data() + off, a.data(), a.size());
    d1[i].pData = pool.p + off; d1[i].nStep = w[i]; d1[i].oSize = NppiSize{w[i], h[i]};
    off += a.size();
    memcpy(blob.data() + off, b.data(), b.size());
    d2[i].pData = pool.p + off; d2[i].nStep = w[i]; d2[i].oSize = NppiSize{w[i], h[i]};
    off += b.size();
  }
  pool.put(blob);
  Dev<NppiImageDescriptor> dd1(N), dd2(N);
  dd1.put(std::vector<NppiImageDescriptor>(d1, d1 + N));
  dd2.put(std::vector<NppiImageDescriptor>(d2, d2 + N));
  Dev<NppiBufferDescriptor> dbufs(N);
  Dev<Npp32f> out(N * 8);
  Dev<Npp8u> scratch(64 << 20);
  for (int i = 0; i < N; ++i) {
    bufs[i].pData = scratch.p + (size_t)i * (21 << 20);
    bufs[i].nBufferSize = 21 << 20;
  }
  dbufs.put(std::vector<NppiBufferDescriptor>(bufs, bufs + N));
  if (stage <= 0) {  // MSEBatch with common ROI (16x8: min of all sizes)
    out.fill(-1e30f);
    NppStatus s = nppiMSEBatch_8u_C1R_Ctx(dd1.p, dd2.p, N, NppiSize{16, 8}, out.p, dbufs.p, c);
    PS("MSEBatch", s);
    auto v = out.get();
    printf("  out=[%g %g %g %g ...] (nBatchSize=3, ROI=16x8)\n", v[0], v[1], v[2], v[3]);
    CHK("msebatch");
    // single-image MSE for comparison on image 0 with ROI 16x8
    Dev<Npp32f> one(2);
    one.fill(-1e30f);
    NppStatus s1 = nppiMSE_8u_C1R_Ctx((Npp8u *)d1[0].pData, 16, (Npp8u *)d2[0].pData, 16, NppiSize{16, 8}, one.p, gBuf64, c);
    PS("MSE_single_img0", s1);
    printf("  single=[%g]\n", one.get()[0]);
  }
  if (stage <= 1) {  // MSEBatch Advanced (per-image sizes)
    out.fill(-1e30f);
    NppStatus s = nppiMSEBatch_8u_C1R_Advanced_Ctx(dd1.p, dd2.p, N, NppiSize{32, 16}, out.p, dbufs.p, c);
    PS("MSEBatchAdv", s);
    auto v = out.get();
    printf("  out=[%g %g %g %g ...]\n", v[0], v[1], v[2], v[3]);
    CHK("msebatchadv");
    // per-image single MSE with full sizes
    for (int i = 0; i < N; ++i) {
      Dev<Npp32f> one(1);
      one.fill(-1e30f);
      NppStatus s1 = nppiMSE_8u_C1R_Ctx((Npp8u *)d1[i].pData, w[i], (Npp8u *)d2[i].pData, w[i], NppiSize{w[i], h[i]},
                                        one.p, gBuf64, c);
      printf("  single full img%d=[%g] st=%d\n", i, one.get()[0], (int)s1);
    }
  }
  if (stage <= 1) {  // PSNRBatch
    out.fill(-1e30f);
    NppStatus s = nppiPSNRBatch_8u_C1R_Ctx(dd1.p, dd2.p, N, NppiSize{16, 8}, out.p, dbufs.p, c);
    PS("PSNRBatch", s);
    auto v = out.get();
    printf("  out=[%g %g %g %g ...]\n", v[0], v[1], v[2], v[3]);
    CHK("psnrbatch");
    out.fill(-1e30f);
    s = nppiPSNRBatch_8u_C1R_Advanced_Ctx(dd1.p, dd2.p, N, NppiSize{32, 16}, out.p, dbufs.p, c);
    PS("PSNRBatchAdv", s);
    v = out.get();
    printf("  out=[%g %g %g %g ...]\n", v[0], v[1], v[2], v[3]);
  }
  if (stage <= 2) {  // SSIMBatch
    out.fill(-1e30f);
    NppStatus s = nppiSSIMBatch_8u_C1R_Ctx(dd1.p, dd2.p, N, NppiSize{16, 8}, out.p, dbufs.p, c);
    PS("SSIMBatch", s);
    auto v = out.get();
    printf("  out=[%g %g %g %g ...]\n", v[0], v[1], v[2], v[3]);
    CHK("ssimbatch");
    Dev<Npp32f> one(1);
    one.fill(-1e30f);
    NppStatus s1 = nppiSSIM_8u_C1R_Ctx((Npp8u *)d1[0].pData, 16, (Npp8u *)d2[0].pData, 16, NppiSize{16, 8}, one.p, gBuf64, c);
    printf("  single img0=[%g] st=%d\n", one.get()[0], (int)s1);
  }
  if (stage <= 2) {  // WMSSSIMBatch (needs bigger ROIs?)
    out.fill(-1e30f);
    NppStatus s = nppiWMSSSIMBatch_8u_C1R_Ctx(dd1.p, dd2.p, N, NppiSize{16, 8}, out.p, dbufs.p, c);
    PS("WMSSSIMBatch", s);
    auto v = out.get();
    printf("  out=[%g %g %g %g ...]\n", v[0], v[1], v[2], v[3]);
    CHK("wmsssimbatch");
    out.fill(-1e30f);
    s = nppiWMSSSIMBatch_8u_C1R_Advanced_Ctx(dd1.p, dd2.p, N, NppiSize{32, 16}, out.p, dbufs.p, c);
    PS("WMSSSIMBatchAdv", s);
    v = out.get();
    printf("  out=[%g %g %g %g ...]\n", v[0], v[1], v[2], v[3]);
  }
  if (stage <= 3) {  // C3R batch
    // build 3-channel images
    const int W = 16, H = 8;
    Dev<Npp8u> pool3((size_t)W * H * 3 * 2 * N);
    std::vector<Npp8u> blob3(pool3.n);
    size_t off = 0;
    for (int i = 0; i < N; ++i) {
      std::vector<Npp8u> a = tex8(W, H, 3, 10 + i), b = tex8(W, H, 3, 20 + i);
      memcpy(blob3.data() + off, a.data(), a.size());
      d1[i].pData = pool3.p + off; d1[i].nStep = W * 3; d1[i].oSize = NppiSize{W, H};
      off += a.size();
      memcpy(blob3.data() + off, b.data(), b.size());
      d2[i].pData = pool3.p + off; d2[i].nStep = W * 3; d2[i].oSize = NppiSize{W, H};
      off += b.size();
    }
    pool3.put(blob3);
    dd1.put(std::vector<NppiImageDescriptor>(d1, d1 + N));
    dd2.put(std::vector<NppiImageDescriptor>(d2, d2 + N));
    out.fill(-1e30f);
    NppStatus s = nppiMSEBatch_8u_C3R_Ctx(dd1.p, dd2.p, N, NppiSize{W, H}, out.p, dbufs.p, c);
    PS("MSEBatch_C3R", s);
    auto v = out.get();
    printf("  out=[%g %g %g %g %g %g ...] (3 floats per image?)\n", v[0], v[1], v[2], v[3], v[4], v[5]);
    CHK("c3batch");
    out.fill(-1e30f);
    s = nppiSSIMBatch_8u_C3R_Ctx(dd1.p, dd2.p, N, NppiSize{W, H}, out.p, dbufs.p, c);
    PS("SSIMBatch_C3R", s);
    v = out.get();
    printf("  out=[%g %g %g %g %g %g ...]\n", v[0], v[1], v[2], v[3], v[4], v[5]);
  }
  if (stage <= 4) {  // batch zero size / null
    out.fill(-1e30f);
    NppStatus s = nppiMSEBatch_8u_C1R_Ctx(nullptr, nullptr, 0, NppiSize{16, 8}, out.p, dbufs.p, c);
    PS("MSEBatch_n0_null", s);
    s = nppiMSEBatch_8u_C1R_Ctx(dd1.p, dd2.p, 0, NppiSize{16, 8}, out.p, dbufs.p, c);
    PS("MSEBatch_n0", s);
    s = nppiMSEBatch_8u_C1R_Ctx(nullptr, dd2.p, N, NppiSize{16, 8}, out.p, dbufs.p, c);
    PS("MSEBatch_null1", s);
    s = nppiMSEBatch_8u_C1R_Ctx(dd1.p, dd2.p, N, NppiSize{-1, 8}, out.p, dbufs.p, c);
    PS("MSEBatch_negroi", s);
    s = nppiMSEBatch_8u_C1R_Ctx(dd1.p, dd2.p, N, NppiSize{0, 8}, out.p, dbufs.p, c);
    PS("MSEBatch_zroi_CRASHES", s);
    s = nppiMSEBatch_8u_C1R_Ctx(dd1.p, dd2.p, N, NppiSize{16, 8}, nullptr, dbufs.p, c);
    PS("MSEBatch_nullout", s);
    s = nppiMSEBatch_8u_C1R_Ctx(dd1.p, dd2.p, N, NppiSize{16, 8}, out.p, nullptr, c);
    PS("MSEBatch_nullbuf", s);
  }
}

// ======================= SSIMBatch layout diagnostics ========================
// The plain SSIMBatch gave ~1e-5 for images whose (step != ROI.w || oSize != ROI).
// Diagnose what NPP actually reads: fill pool/workspace with known patterns.
static void probe_ssimb2() {
  NppStreamContext c = ctx();
  const int N = 3;
  int w[N] = {16, 24, 32}, h[N] = {8, 12, 16};
  NppiImageDescriptor d1[N], d2[N];
  NppiBufferDescriptor bufs[N];
  Dev<Npp8u> pool(64 * 64 * 2 * N);
  Dev<Npp8u> scratch(64 << 20);
  for (int i = 0; i < N; ++i) {
    bufs[i].pData = scratch.p + (size_t)i * (21 << 20);
    bufs[i].nBufferSize = 21 << 20;
  }
  Dev<NppiBufferDescriptor> dbufs(N);
  dbufs.put(std::vector<NppiBufferDescriptor>(bufs, bufs + N));
  Dev<NppiImageDescriptor> dd1(N), dd2(N);
  Dev<Npp32f> out(N * 4);

  // V0: pool filled with position pattern (k&0xff), scratch zeroed
  // V1: pool 0xAB, scratch zeroed
  // V2: pool 0xAB, scratch 0xCD
  for (int variant = 0; variant < 3; ++variant) {
    std::vector<Npp8u> blob(pool.n);
    for (size_t k = 0; k < blob.size(); ++k) blob[k] = (variant == 0) ? (Npp8u)(k & 0xff) : (Npp8u)0xAB;
    size_t off = 0;
    for (int i = 0; i < N; ++i) {
      std::vector<Npp8u> a = tex8(w[i], h[i], 1, 10 + i), b = tex8(w[i], h[i], 1, 20 + i);
      memcpy(blob.data() + off, a.data(), a.size());
      d1[i].pData = pool.p + off; d1[i].nStep = w[i]; d1[i].oSize = NppiSize{w[i], h[i]};
      off += a.size();
      memcpy(blob.data() + off, b.data(), b.size());
      d2[i].pData = pool.p + off; d2[i].nStep = w[i]; d2[i].oSize = NppiSize{w[i], h[i]};
      off += b.size();
    }
    pool.put(blob);
    if (variant >= 1) cudaMemset(scratch.p, variant == 1 ? 0x00 : 0xCD, 64 << 20);
    else cudaMemset(scratch.p, 0x00, 64 << 20);
    dd1.put(std::vector<NppiImageDescriptor>(d1, d1 + N));
    dd2.put(std::vector<NppiImageDescriptor>(d2, d2 + N));
    out.fill(-1e30f);
    NppStatus s = nppiSSIMBatch_8u_C1R_Ctx(dd1.p, dd2.p, N, NppiSize{16, 8}, out.p, dbufs.p, c);
    printf("  V%d(st=%d) out=[%g %g %g]\n", variant, (int)s, out.get()[0], out.get()[1], out.get()[2]);
    CHK("ssimb2");
  }
  // V3: uniform 16x8 images, steps 16 (control)
  // V4: uniform 16x8 data, steps 16/24/32 (padded)
  // V5: all images 24x12 step 24, common ROI 16x8
  for (int variant = 3; variant <= 5; ++variant) {
    const int RW = 16, RH = 8;
    std::vector<Npp8u> blob(pool.n, 0xAB);
    size_t off = 0;
    for (int i = 0; i < N; ++i) {
      int iw, ih, istep;
      if (variant == 3) { iw = 16; ih = 8; istep = 16; }
      else if (variant == 4) { iw = 16; ih = 8; istep = 16 + 8 * i; }
      else { iw = 24; ih = 12; istep = 24; }
      std::vector<Npp8u> a = tex8(iw, ih, 1, 10 + i), b = tex8(iw, ih, 1, 20 + i);
      for (int y = 0; y < ih; ++y) memcpy(blob.data() + off + (size_t)y * istep, a.data() + (size_t)y * iw, iw);
      d1[i].pData = pool.p + off; d1[i].nStep = istep; d1[i].oSize = NppiSize{iw, ih};
      off += (size_t)ih * istep + 64;
      for (int y = 0; y < ih; ++y) memcpy(blob.data() + off + (size_t)y * istep, b.data() + (size_t)y * iw, iw);
      d2[i].pData = pool.p + off; d2[i].nStep = istep; d2[i].oSize = NppiSize{iw, ih};
      off += (size_t)ih * istep + 64;
    }
    pool.put(blob);
    cudaMemset(scratch.p, 0x00, 64 << 20);
    dd1.put(std::vector<NppiImageDescriptor>(d1, d1 + N));
    dd2.put(std::vector<NppiImageDescriptor>(d2, d2 + N));
    out.fill(-1e30f);
    NppStatus s = nppiSSIMBatch_8u_C1R_Ctx(dd1.p, dd2.p, N, NppiSize{RW, RH}, out.p, dbufs.p, c);
    printf("  V%d(st=%d) out=[%g %g %g]\n", variant, (int)s, out.get()[0], out.get()[1], out.get()[2]);
    CHK("ssimb2v");
  }
}

// ======================= WMSSSIM / MSSSIM internals =========================
static void probe_wmss2() {
  NppStreamContext c = ctx();
  Dev<Npp32f> out(8);
  // A. C3R with channel-independent data (same value in all 3 channels)
  {
    struct Sz { int w, h; };
    Sz sizes[] = {{64, 64}, {32, 32}, {16, 16}};
    for (auto &z : sizes) {
      std::vector<Npp8u> a, b;
      for (int y = 0; y < z.h; ++y)
        for (int x = 0; x < z.w; ++x)
          for (int cc = 0; cc < 3; ++cc) {
            a.push_back((Npp8u)((x * 7 + y * 13 + 5) & 0xff));
            b.push_back((Npp8u)((x * 7 + y * 13 + 6) & 0xff));
          }
      Dev<Npp8u> s1(a.size()), s2(b.size());
      s1.put(a); s2.put(b);
      out.fill(-1e30f);
      NppStatus s = nppiWMSSSIM_8u_C3R_Ctx(s1.p, z.w * 3, s2.p, z.w * 3, NppiSize{z.w, z.h}, out.p, gBuf64, c);
      auto v = out.get();
      printf("  C3R_samech %dx%d st=%d val=[%g %g %g]\n", z.w, z.h, (int)s, v[0], v[1], v[2]);
      CHK("wmss2a");
      // C1R on the same logical image
      std::vector<Npp8u> a1, b1;
      for (int y = 0; y < z.h; ++y)
        for (int x = 0; x < z.w; ++x) {
          a1.push_back((Npp8u)((x * 7 + y * 13 + 5) & 0xff));
          b1.push_back((Npp8u)((x * 7 + y * 13 + 6) & 0xff));
        }
      Dev<Npp8u> t1(a1.size()), t2(b1.size());
      t1.put(a1); t2.put(b1);
      out.fill(-1e30f);
      s = nppiWMSSSIM_8u_C1R_Ctx(t1.p, z.w, t2.p, z.w, NppiSize{z.w, z.h}, out.p, gBuf64, c);
      printf("    C1R ref val=[%g]\n", out.get()[0]);
    }
  }
  // B. const C3R
  {
    Dev<Npp8u> s1((size_t)64 * 64 * 3); Dev<Npp8u> s2((size_t)64 * 64 * 3);
    s1.fill(100); s2.fill(110);
    out.fill(-1e30f);
    NppStatus s = nppiWMSSSIM_8u_C3R_Ctx(s1.p, 64 * 3, s2.p, 64 * 3, NppiSize{64, 64}, out.p, gBuf64, c);
    auto v = out.get();
    printf("  C3R_const64 st=%d val=[%g %g %g]\n", (int)s, v[0], v[1], v[2]);
    Dev<Npp8u> u1((size_t)64 * 64); Dev<Npp8u> u2((size_t)64 * 64);
    u1.fill(100); u2.fill(110);
    out.fill(-1e30f);
    s = nppiWMSSSIM_8u_C1R_Ctx(u1.p, 64, u2.p, 64, NppiSize{64, 64}, out.p, gBuf64, c);
    printf("  C1R_const64 st=%d val=[%g]\n", (int)s, out.get()[0]);
  }
  // C. C1R more sizes/seeds for weight fitting
  {
    struct SZ { int w, h; int s1, s2; };
    SZ cases[] = {{18,18,5,6},{20,20,5,6},{24,24,5,6},{28,28,5,6},{40,40,5,6},{48,48,5,6},
                  {96,96,5,6},{128,128,5,6},{64,64,1,2},{128,128,1,2},{64,64,7,9},{64,32,5,6},
                  {32,64,5,6},{48,16,5,6},{64,64,5,5}};
    for (auto &k : cases) {
      Dev<Npp8u> s1((size_t)k.w * k.h), s2((size_t)k.w * k.h);
      s1.put(tex8(k.w, k.h, 1, k.s1));
      s2.put(tex8(k.w, k.h, 1, k.s2));
      out.fill(-1e30f);
      NppStatus s = nppiWMSSSIM_8u_C1R_Ctx(s1.p, k.w, s2.p, k.w, NppiSize{k.w, k.h}, out.p, gBuf64, c);
      printf("  C1R %dx%d s%d/%d st=%d val=[%g]\n", k.w, k.h, k.s1, k.s2, (int)s, out.get()[0]);
      CHK("wmss2c");
    }
    // const C1R
    int csz[] = {16, 32, 64, 256};
    for (int sz : csz) {
      Dev<Npp8u> s1((size_t)sz * sz); Dev<Npp8u> s2((size_t)sz * sz);
      s1.fill(100); s2.fill(110);
      out.fill(-1e30f);
      NppStatus s = nppiWMSSSIM_8u_C1R_Ctx(s1.p, sz, s2.p, sz, NppiSize{sz, sz}, out.p, gBuf64, c);
      printf("  C1R_const%d val=[%g] st=%d\n", sz, out.get()[0], (int)s);
      out.fill(-1e30f);
      s = nppiMSSSIM_8u_C1R_Ctx(s1.p, sz, s2.p, sz, NppiSize{sz, sz}, out.p, gBuf64, c);
      printf("  MSSSIM_const%d val=[%g] st=%d\n", sz, out.get()[0], (int)s);
    }
  }
  // D. MSSSIM size thresholds
  {
    struct SZ { int w, h; };
    SZ cases[] = {{128,128},{200,200},{255,255},{257,257},{512,512},{300,300},{256,128},{128,256}};
    for (auto &k : cases) {
      Dev<Npp8u> s1((size_t)k.w * k.h), s2((size_t)k.w * k.h);
      s1.put(tex8(k.w, k.h, 1, 5));
      s2.put(tex8(k.w, k.h, 1, 6));
      out.fill(-1e30f);
      NppStatus s = nppiMSSSIM_8u_C1R_Ctx(s1.p, k.w, s2.p, k.w, NppiSize{k.w, k.h}, out.p, gBuf64, c);
      printf("  MSSSIM %dx%d st=%d val=[%g]\n", k.w, k.h, (int)s, out.get()[0]);
      CHK("wmss2d");
      out.fill(-1e30f);
      s = nppiWMSSSIM_8u_C1R_Ctx(s1.p, k.w, s2.p, k.w, NppiSize{k.w, k.h}, out.p, gBuf64, c);
      printf("  WMSSSIM %dx%d st=%d val=[%g]\n", k.w, k.h, (int)s, out.get()[0]);
    }
  }
}

// ======================= WMSSSIM pyramid sampling ===========================
// Determine how NPP downsamples between scales: put a single outlier pixel in
// src2 at positions that decimation (stride-2 subsampling) would keep vs drop.
static void probe_wmss3() {
  NppStreamContext c = ctx();
  Dev<Npp32f> out(8);
  struct P { int x, y; };
  P posns[] = {{0,0},{1,1},{32,32},{33,33},{62,62},{63,63},{16,16},{2,2}};
  for (int sz : {64, 65}) {
    for (auto &pp : posns) {
      if (pp.x >= sz || pp.y >= sz) continue;
      std::vector<Npp8u> a, b;
      for (int y = 0; y < sz; ++y)
        for (int x = 0; x < sz; ++x) {
          Npp8u v = (Npp8u)(((x + y) & 0x3f) + 10);
          a.push_back(v);
          b.push_back(v);
        }
      b[(size_t)pp.y * sz + pp.x] = 200;  // single outlier (a=10 at (0,0))
      Dev<Npp8u> s1(a.size()), s2(b.size());
      s1.put(a); s2.put(b);
      out.fill(-1e30f);
      NppStatus s = nppiWMSSSIM_8u_C1R_Ctx(s1.p, sz, s2.p, sz, NppiSize{sz, sz}, out.p, gBuf64, c);
      printf("  WMSSSIM %d outlier(%d,%d) st=%d val=[%g]\n", sz, pp.x, pp.y, (int)s, out.get()[0]);
      CHK("wmss3");
    }
  }
  // SSIM at more sizes (scale-0 validation)
  {
    Dev<Npp32f> o2(4);
    for (int sz : {9, 10, 12, 17, 18, 20, 24, 28, 40, 48, 96, 128}) {
      Dev<Npp8u> s1((size_t)sz * sz), s2((size_t)sz * sz);
      s1.put(tex8(sz, sz, 1, 5));
      s2.put(tex8(sz, sz, 1, 6));
      o2.fill(-1e30f);
      NppStatus s = nppiSSIM_8u_C1R_Ctx(s1.p, sz, s2.p, sz, NppiSize{sz, sz}, o2.p, gBuf64, c);
      printf("  SSIM %dx%d st=%d val=[%g]\n", sz, sz, (int)s, o2.get()[0]);
      CHK("wmss3s");
    }
  }
  // same outlier trick for MSSSIM at 256
  {
    for (auto &pp : (P[]){{0,0},{1,1},{128,128},{254,254}}) {
      std::vector<Npp8u> a, b;
      for (int y = 0; y < 256; ++y)
        for (int x = 0; x < 256; ++x) {
          Npp8u v = (Npp8u)(((x + y) & 0x3f) + 10);
          a.push_back(v); b.push_back(v);
        }
      b[(size_t)pp.y * 256 + pp.x] = 200;
      Dev<Npp8u> s1(a.size()), s2(b.size());
      s1.put(a); s2.put(b);
      out.fill(-1e30f);
      NppStatus s = nppiMSSSIM_8u_C1R_Ctx(s1.p, 256, s2.p, 256, NppiSize{256, 256}, out.p, gBuf64, c);
      printf("  MSSSIM 256 outlier(%d,%d) st=%d val=[%g]\n", pp.x, pp.y, (int)s, out.get()[0]);
      out.fill(-1e30f);
      s = nppiWMSSSIM_8u_C1R_Ctx(s1.p, 256, s2.p, 256, NppiSize{256, 256}, out.p, gBuf64, c);
      printf("  WMSSSIM 256 outlier(%d,%d) st=%d val=[%g]\n", pp.x, pp.y, (int)s, out.get()[0]);
      CHK("wmss3m");
    }
  }
  // MSSSIM size threshold between 128 and 200
  {
    for (int sz : {129, 130, 132, 136, 144, 148, 152, 160, 164, 168, 176, 180, 184, 192, 196, 198, 199}) {
      Dev<Npp8u> s1((size_t)sz * sz), s2((size_t)sz * sz);
      s1.put(tex8(sz, sz, 1, 5));
      s2.put(tex8(sz, sz, 1, 6));
      out.fill(-1e30f);
      NppStatus s = nppiMSSSIM_8u_C1R_Ctx(s1.p, sz, s2.p, sz, NppiSize{sz, sz}, out.p, gBuf64, c);
      printf("  MSSSIM_thr %d st=%d val=[%g]\n", sz, (int)s, out.get()[0]);
      CHK("wmss3t");
    }
  }
}

// ======================= WMSSSIM level isolation =============================
// (a) single-pixel outlier scan along one axis reveals the level-2 block layout:
//     standard {2i,2i+1}, shifted {2i-1,2i} (corner exact), base-clamp (spike at
//     x=1 from overlapping blocks {0,1},{1,2});
// (b) last row/col modification reveals whether row N-1 survives decimation;
// (c) separable anti-symmetric patterns H_k cancel in every 2x2 block after k
//     decimations -> isolates per-level css contributions and the weights;
// (d) const pairs at various sizes pin the level-5 luminance term exactly.
static int hk1(int x, int k) {  // h_0(x)=alt; h_{k+1}(x)=h_k(x>0 ? (x-1)/2 : 0)
  if (k == 0) return (x & 1) ? -1 : 1;
  return hk1(x > 0 ? ((x - 1) >> 1) : 0, k - 1);
}
static void probe_wmss4() {
  NppStreamContext c = ctx();
  Dev<Npp32f> out(8);
  // A. axis scans of single-pixel outliers on the no-wrap ramp
  for (int n : {64, 65}) {
    for (int axis = 0; axis < 2; ++axis) {
      for (int p = 0; p < n; ++p) {
        std::vector<Npp8u> a, b;
        for (int y = 0; y < n; ++y)
          for (int x = 0; x < n; ++x) {
            Npp8u v = (Npp8u)(((x + y) & 0x3f) + 10);
            a.push_back(v);
            b.push_back(v);
          }
        int ox = axis ? 16 : p, oy = axis ? p : 16;
        b[(size_t)oy * n + ox] = 200;
        Dev<Npp8u> s1(a.size()), s2(b.size());
        s1.put(a); s2.put(b);
        out.fill(-1e30f);
        NppStatus s = nppiWMSSSIM_8u_C1R_Ctx(s1.p, n, s2.p, n, NppiSize{n, n}, out.p, gBuf64, c);
        printf("  SCAN n=%d ax=%d p=%d st=%d v=[%.6f]\n", n, axis, p, (int)s, out.get()[0]);
      }
      CHK("wmss4scan");
    }
  }
  // B. last row / column wholesale modification
  for (int n : {64, 65}) {
    for (int mode = 0; mode < 4; ++mode) {  // 0:row n-1  1:row n-2  2:col n-1  3:col n-2
      std::vector<Npp8u> a, b;
      for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x) {
          Npp8u v = (Npp8u)(((x + y) & 0x3f) + 10);
          a.push_back(v);
          b.push_back(v);
        }
      int line = (mode & 1) ? n - 2 : n - 1;
      for (int i = 0; i < n; ++i) {
        if (mode < 2) b[(size_t)line * n + i] = 200;
        else b[(size_t)i * n + line] = 200;
      }
      Dev<Npp8u> s1(a.size()), s2(b.size());
      s1.put(a); s2.put(b);
      out.fill(-1e30f);
      NppStatus s = nppiWMSSSIM_8u_C1R_Ctx(s1.p, n, s2.p, n, NppiSize{n, n}, out.p, gBuf64, c);
      printf("  LINE n=%d mode=%d st=%d v=[%.6f]\n", n, mode, (int)s, out.get()[0]);
      CHK("wmss4line");
    }
  }
  // C. H_k anti-symmetric separable patterns (A/B sign-flipped)
  {
    for (int k = 0; k < 4; ++k) {
      int n = 64;
      std::vector<Npp8u> a, b;
      for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x) {
          int h = hk1(x, k) * hk1(y, k);
          a.push_back((Npp8u)(20 + 8 * h));
          b.push_back((Npp8u)(20 - 8 * h));
        }
      Dev<Npp8u> s1(a.size()), s2(b.size());
      s1.put(a); s2.put(b);
      out.fill(-1e30f);
      NppStatus s = nppiWMSSSIM_8u_C1R_Ctx(s1.p, n, s2.p, n, NppiSize{n, n}, out.p, gBuf64, c);
      printf("  HK k=%d st=%d v=[%.6f]\n", k, (int)s, out.get()[0]);
      CHK("wmss4hk");
    }
  }
  // D. const pairs across sizes (exact level-5 luminance)
  {
    struct CP { int va, vb; };
    for (auto &cp : (CP[]){{100, 110}, {5, 6}}) {
      for (int n : {16, 17, 31, 32, 63, 64, 65, 176, 200}) {
        Dev<Npp8u> s1((size_t)n * n), s2((size_t)n * n);
        s1.fill((Npp8u)cp.va); s2.fill((Npp8u)cp.vb);
        out.fill(-1e30f);
        NppStatus s = nppiWMSSSIM_8u_C1R_Ctx(s1.p, n, s2.p, n, NppiSize{n, n}, out.p, gBuf64, c);
        printf("  CONST a=%d b=%d n=%d st=%d v=[%.7f]\n", cp.va, cp.vb, n, (int)s, out.get()[0]);
        CHK("wmss4const");
      }
    }
  }
  // E. 2x2 corner block probe: outlier at the four corner taps of block (0,0)
  {
    struct P2 { int x, y; };
    for (auto &pp : (P2[]){{0,0},{0,1},{1,0},{1,1},{0,2},{2,0},{0,3},{3,0}}) {
      int n = 64;
      std::vector<Npp8u> a, b;
      for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x) {
          Npp8u v = (Npp8u)(((x + y) & 0x3f) + 10);
          a.push_back(v);
          b.push_back(v);
        }
      b[(size_t)pp.y * n + pp.x] = 200;
      Dev<Npp8u> s1(a.size()), s2(b.size());
      s1.put(a); s2.put(b);
      out.fill(-1e30f);
      NppStatus s = nppiWMSSSIM_8u_C1R_Ctx(s1.p, n, s2.p, n, NppiSize{n, n}, out.p, gBuf64, c);
      printf("  CORNER (%d,%d) st=%d v=[%.6f]\n", pp.x, pp.y, (int)s, out.get()[0]);
      CHK("wmss4corner");
    }
  }
}

// ======================= buffer sizes =======================================
static void probe_bufsz() {
  NppStreamContext c = ctx();
  size_t n = 0;
  struct R { int w, h; };
  R rs[] = {{16, 16}, {64, 32}, {256, 256}, {1024, 768}, {1, 1}};
  for (auto &r : rs) {
    NppiSize z{r.w, r.h};
    printf("  ROI %dx%d:", r.w, r.h);
    if (nppiQualityIndexGetBufferHostSize_8u32f_C1R_Ctx(z, &n, c) == NPP_SUCCESS) printf(" qIdx8uC1=%d", n);
    if (nppiQualityIndexGetBufferHostSize_8u32f_C3R_Ctx(z, &n, c) == NPP_SUCCESS) printf(" qIdx8uC3=%d", n);
    if (nppiQualityIndexGetBufferHostSize_8u32f_AC4R_Ctx(z, &n, c) == NPP_SUCCESS) printf(" qIdx8uAC4=%d", n);
    if (nppiQualityIndexGetBufferHostSize_32f_C1R_Ctx(z, &n, c) == NPP_SUCCESS) printf(" qIdx32fC1=%d", n);
    if (nppiCountInRangeGetBufferHostSize_8u_C1R_Ctx(z, &n, c) == NPP_SUCCESS) printf(" cir8uC1=%d", n);
    if (nppiCountInRangeGetBufferHostSize_8u_C3R_Ctx(z, &n, c) == NPP_SUCCESS) printf(" cir8uC3=%d", n);
    if (nppiCountInRangeGetBufferHostSize_32f_AC4R_Ctx(z, &n, c) == NPP_SUCCESS) printf(" cir32fAC4=%d", n);
    if (nppiMSEGetBufferHostSize_8u_C1R_Ctx(z, &n, c) == NPP_SUCCESS) printf(" mse8uC1=%d", n);
    if (nppiMSEGetBufferHostSize_8u_C3R_Ctx(z, &n, c) == NPP_SUCCESS) printf(" mse8uC3=%d", n);
    if (nppiPSNRGetBufferHostSize_8u_C1R_Ctx(z, &n, c) == NPP_SUCCESS) printf(" psnr8uC1=%d", n);
    if (nppiSSIMGetBufferHostSize_8u_C1R_Ctx(z, &n, c) == NPP_SUCCESS) printf(" ssim8uC1=%d", n);
    if (nppiSSIMGetBufferHostSize_8u_C3R_Ctx(z, &n, c) == NPP_SUCCESS) printf(" ssim8uC3=%d", n);
    if (nppiMSSSIMGetBufferHostSize_8u_C1R_Ctx(z, &n, c) == NPP_SUCCESS) printf(" msssim8uC1=%d", n);
    if (nppiWMSSSIMGetBufferHostSize_8u_C1R_Ctx(z, &n, c) == NPP_SUCCESS) printf(" wmsssim8uC1=%d", n);
    if (nppiWMSSSIMGetBufferHostSize_8u_C3R_Ctx(z, &n, c) == NPP_SUCCESS) printf(" wmsssim8uC3=%d", n);
    if (nppiMSEBatchGetBufferHostSize_8u_C1R_Ctx(z, &n, c) == NPP_SUCCESS) printf(" mseB8uC1=%d", n);
    if (nppiSSIMBatchGetBufferHostSize_8u_C1R_Ctx(z, &n, c) == NPP_SUCCESS) printf(" ssimB8uC1=%d", n);
    if (nppiWMSSSIMBatchGetBufferHostSize_8u_C1R_Ctx(z, &n, c) == NPP_SUCCESS) printf(" wmsssimB8uC1=%d", n);
    if (nppiWMSSSIMBatchGetBufferHostSize_8u_C3R_Ctx(z, &n, c) == NPP_SUCCESS) printf(" wmsssimB8uC3=%d", n);
    printf("\n");
  }
}

// ======================= contract probes ====================================
static void probe_contract() {
  NppStreamContext c = ctx();
  Dev<Npp8u> s1(64), s2(64);
  Dev<Npp32f> o(4);
  Dev<int> cnt(4);
  Dev<Npp32s> h1(64);
  Dev<NppiProfileData> prof(16);
  printf("--- QualityIndex 8u C1R ---\n");
  PS("null_src1", nppiQualityIndex_8u32f_C1R_Ctx(nullptr, 4, s2.p, 4, NppiSize{2, 2}, o.p, gBuf64, c));
  PS("null_src2", nppiQualityIndex_8u32f_C1R_Ctx(s1.p, 4, nullptr, 4, NppiSize{2, 2}, o.p, gBuf64, c));
  PS("null_out", nppiQualityIndex_8u32f_C1R_Ctx(s1.p, 4, s2.p, 4, NppiSize{2, 2}, nullptr, gBuf64, c));
  PS("null_buf", nppiQualityIndex_8u32f_C1R_Ctx(s1.p, 4, s2.p, 4, NppiSize{2, 2}, o.p, nullptr, c));
  PS("zro_roi", nppiQualityIndex_8u32f_C1R_Ctx(s1.p, 4, s2.p, 4, NppiSize{0, 2}, o.p, gBuf64, c));
  PS("neg_roi", nppiQualityIndex_8u32f_C1R_Ctx(s1.p, 4, s2.p, 4, NppiSize{-1, 2}, o.p, gBuf64, c));
  PS("short_step", nppiQualityIndex_8u32f_C1R_Ctx(s1.p, 2, s2.p, 4, NppiSize{4, 2}, o.p, gBuf64, c));
  PS("short_step2", nppiQualityIndex_8u32f_C1R_Ctx(s1.p, 4, s2.p, 2, NppiSize{4, 2}, o.p, gBuf64, c));
  printf("--- CountInRange 8u C1R ---\n");
  PS("null_src", nppiCountInRange_8u_C1R_Ctx(nullptr, 4, NppiSize{2, 2}, cnt.p, 0, 255, gBuf64, c));
  PS("null_out", nppiCountInRange_8u_C1R_Ctx(s1.p, 4, NppiSize{2, 2}, nullptr, 0, 255, gBuf64, c));
  PS("null_buf", nppiCountInRange_8u_C1R_Ctx(s1.p, 4, NppiSize{2, 2}, cnt.p, 0, 255, nullptr, c));
  PS("zro_roi", nppiCountInRange_8u_C1R_Ctx(s1.p, 4, NppiSize{0, 2}, cnt.p, 0, 255, gBuf64, c));
  PS("neg_roi", nppiCountInRange_8u_C1R_Ctx(s1.p, 4, NppiSize{-1, 2}, cnt.p, 0, 255, gBuf64, c));
  PS("short_step", nppiCountInRange_8u_C1R_Ctx(s1.p, 2, NppiSize{4, 2}, cnt.p, 0, 255, gBuf64, c));
  PS("rev_range", nppiCountInRange_8u_C1R_Ctx(s1.p, 4, NppiSize{2, 2}, cnt.p, 200, 100, gBuf64, c));
  PS("eq_range", nppiCountInRange_8u_C1R_Ctx(s1.p, 4, NppiSize{2, 2}, cnt.p, 100, 100, gBuf64, c));
  printf("--- CountInRange 8u C3R (null arrays) ---\n");
  {
    Dev<Npp8u> lo(3), hi(3);
    PS("null_counts", nppiCountInRange_8u_C3R_Ctx(s1.p, 12, NppiSize{2, 2}, nullptr, lo.p, hi.p, gBuf64, c));
    PS("null_lo", nppiCountInRange_8u_C3R_Ctx(s1.p, 12, NppiSize{2, 2}, cnt.p, nullptr, hi.p, gBuf64, c));
    PS("null_hi", nppiCountInRange_8u_C3R_Ctx(s1.p, 12, NppiSize{2, 2}, cnt.p, lo.p, nullptr, gBuf64, c));
  }
  printf("--- MSE 8u C1R ---\n");
  PS("null_src1", nppiMSE_8u_C1R_Ctx(nullptr, 4, s2.p, 4, NppiSize{2, 2}, o.p, gBuf64, c));
  PS("null_out", nppiMSE_8u_C1R_Ctx(s1.p, 4, s2.p, 4, NppiSize{2, 2}, nullptr, gBuf64, c));
  PS("null_buf", nppiMSE_8u_C1R_Ctx(s1.p, 4, s2.p, 4, NppiSize{2, 2}, o.p, nullptr, c));
  PS("zro_roi", nppiMSE_8u_C1R_Ctx(s1.p, 4, s2.p, 4, NppiSize{0, 2}, o.p, gBuf64, c));
  PS("neg_roi", nppiMSE_8u_C1R_Ctx(s1.p, 4, s2.p, 4, NppiSize{-1, 2}, o.p, gBuf64, c));
  PS("short_step", nppiMSE_8u_C1R_Ctx(s1.p, 2, s2.p, 4, NppiSize{4, 2}, o.p, gBuf64, c));
  PS("noteven_step", nppiMSE_8u_C3R_Ctx(s1.p, 7, s2.p, 7, NppiSize{2, 2}, o.p, gBuf64, c));
  printf("--- PSNR 8u C1R ---\n");
  PS("null_src1", nppiPSNR_8u_C1R_Ctx(nullptr, 4, s2.p, 4, NppiSize{2, 2}, o.p, gBuf64, c));
  PS("zro_roi", nppiPSNR_8u_C1R_Ctx(s1.p, 4, s2.p, 4, NppiSize{0, 2}, o.p, gBuf64, c));
  PS("short_step", nppiPSNR_8u_C1R_Ctx(s1.p, 2, s2.p, 4, NppiSize{4, 2}, o.p, gBuf64, c));
  printf("--- SSIM 8u C1R ---\n");
  PS("null_src1", nppiSSIM_8u_C1R_Ctx(nullptr, 16, s2.p, 16, NppiSize{16, 16}, o.p, gBuf64, c));
  PS("null_out", nppiSSIM_8u_C1R_Ctx(s1.p, 16, s2.p, 16, NppiSize{16, 16}, nullptr, gBuf64, c));
  PS("null_buf", nppiSSIM_8u_C1R_Ctx(s1.p, 16, s2.p, 16, NppiSize{16, 16}, o.p, nullptr, c));
  PS("zro_roi", nppiSSIM_8u_C1R_Ctx(s1.p, 16, s2.p, 16, NppiSize{0, 16}, o.p, gBuf64, c));
  PS("neg_roi", nppiSSIM_8u_C1R_Ctx(s1.p, 16, s2.p, 16, NppiSize{-1, 16}, o.p, gBuf64, c));
  PS("short_step", nppiSSIM_8u_C1R_Ctx(s1.p, 4, s2.p, 16, NppiSize{16, 16}, o.p, gBuf64, c));
  PS("small_roi", nppiSSIM_8u_C1R_Ctx(s1.p, 16, s2.p, 16, NppiSize{1, 1}, o.p, gBuf64, c));
  PS("roi_3x3", nppiSSIM_8u_C1R_Ctx(s1.p, 16, s2.p, 16, NppiSize{3, 3}, o.p, gBuf64, c));
  printf("--- MSSSIM 8u C1R ---\n");
  PS("null_src1", nppiMSSSIM_8u_C1R_Ctx(nullptr, 16, s2.p, 16, NppiSize{256, 256}, o.p, gBuf64, c));
  PS("small_roi", nppiMSSSIM_8u_C1R_Ctx(s1.p, 16, s2.p, 16, NppiSize{16, 16}, o.p, gBuf64, c));
  PS("zro_roi", nppiMSSSIM_8u_C1R_Ctx(s1.p, 16, s2.p, 16, NppiSize{0, 16}, o.p, gBuf64, c));
  PS("null_buf", nppiMSSSIM_8u_C1R_Ctx(s1.p, 16, s2.p, 16, NppiSize{256, 256}, o.p, nullptr, c));
  printf("--- WMSSSIM 8u C1R ---\n");
  PS("null_src1", nppiWMSSSIM_8u_C1R_Ctx(nullptr, 16, s2.p, 16, NppiSize{256, 256}, o.p, gBuf64, c));
  PS("small_roi", nppiWMSSSIM_8u_C1R_Ctx(s1.p, 16, s2.p, 16, NppiSize{16, 16}, o.p, gBuf64, c));
  PS("zro_roi", nppiWMSSSIM_8u_C1R_Ctx(s1.p, 16, s2.p, 16, NppiSize{0, 16}, o.p, gBuf64, c));
  printf("--- RadialProfile 32f ---\n");
  Dev<Npp32f> f1(64);
  PS("null_src", nppiCircularRadialProfile_32f_C1R_Ctx(nullptr, 64, NppiSize{8, 8}, NppiPoint32f{4, 4}, prof.p, 8, c));
  PS("null_prof", nppiCircularRadialProfile_32f_C1R_Ctx(f1.p, 32, NppiSize{8, 8}, NppiPoint32f{4, 4}, nullptr, 8, c));
  PS("zro_roi", nppiCircularRadialProfile_32f_C1R_Ctx(f1.p, 32, NppiSize{0, 8}, NppiPoint32f{4, 4}, prof.p, 8, c));
  PS("neg_roi", nppiCircularRadialProfile_32f_C1R_Ctx(f1.p, 32, NppiSize{-1, 8}, NppiPoint32f{4, 4}, prof.p, 8, c));
  PS("zero_samples", nppiCircularRadialProfile_32f_C1R_Ctx(f1.p, 32, NppiSize{8, 8}, NppiPoint32f{4, 4}, prof.p, 0, c));
  PS("out_center", nppiCircularRadialProfile_32f_C1R_Ctx(f1.p, 32, NppiSize{8, 8}, NppiPoint32f{100, 100}, prof.p, 8, c));
  PS("short_step", nppiCircularRadialProfile_32f_C1R_Ctx(f1.p, 8, NppiSize{8, 8}, NppiPoint32f{4, 4}, prof.p, 8, c));
  PS("ell_null", nppiEllipticalRadialProfile_32f_C1R_Ctx(nullptr, 64, NppiSize{8, 8}, NppiPoint32f{4, 4}, 2.0f, 0.0f,
                                                          prof.p, 8, c));
  PS("ell_zero_ratio", nppiEllipticalRadialProfile_32f_C1R_Ctx(f1.p, 32, NppiSize{8, 8}, NppiPoint32f{4, 4}, 0.0f, 0.0f,
                                                               prof.p, 8, c));
  printf("--- RectStdDev 32f ---\n");
  Dev<Npp64f> sq(64);
  Dev<Npp32f> df(64);
  PS("null_src", nppiRectStdDev_32f_C1R_Ctx(nullptr, 32, sq.p, 64, df.p, 32, NppiSize{4, 4}, NppiRect{0, 0, 2, 2}, c));
  PS("null_sqr", nppiRectStdDev_32f_C1R_Ctx(f1.p, 32, nullptr, 64, df.p, 32, NppiSize{4, 4}, NppiRect{0, 0, 2, 2}, c));
  PS("null_dst", nppiRectStdDev_32f_C1R_Ctx(f1.p, 32, sq.p, 64, nullptr, 32, NppiSize{4, 4}, NppiRect{0, 0, 2, 2}, c));
  PS("zro_roi", nppiRectStdDev_32f_C1R_Ctx(f1.p, 32, sq.p, 64, df.p, 32, NppiSize{0, 4}, NppiRect{0, 0, 2, 2}, c));
  PS("neg_roi", nppiRectStdDev_32f_C1R_Ctx(f1.p, 32, sq.p, 64, df.p, 32, NppiSize{-1, 4}, NppiRect{0, 0, 2, 2}, c));
  PS("short_step", nppiRectStdDev_32f_C1R_Ctx(f1.p, 8, sq.p, 64, df.p, 32, NppiSize{4, 4}, NppiRect{0, 0, 2, 2}, c));
  PS("bad_rect", nppiRectStdDev_32f_C1R_Ctx(f1.p, 32, sq.p, 64, df.p, 32, NppiSize{4, 4}, NppiRect{0, 0, 0, 2}, c));
  PS("neg_rect", nppiRectStdDev_32f_C1R_Ctx(f1.p, 32, sq.p, 64, df.p, 32, NppiSize{4, 4}, NppiRect{-1, 0, 2, 2}, c));
  PS("rect_out", nppiRectStdDev_32f_C1R_Ctx(f1.p, 32, sq.p, 64, df.p, 32, NppiSize{4, 4}, NppiRect{2, 2, 4, 4}, c));
  printf("--- Histogram C3R contract ---\n");
  Dev<Npp32s> hh(256);
  Dev<Npp32s> ll(257);
  {
    Npp32s *hist[3] = {hh.p, hh.p, hh.p};
    int levels[3] = {257, 257, 257};
    Npp32s lo[3] = {0, 0, 0}, hi[3] = {256, 256, 256};
    PS("null_src", nppiHistogramEven_8u_C3R_Ctx(nullptr, 12, NppiSize{2, 2}, hist, levels, lo, hi, gBuf64, c));
    PS("null_hist", nppiHistogramEven_8u_C3R_Ctx(s1.p, 12, NppiSize{2, 2}, nullptr, levels, lo, hi, gBuf64, c));
    PS("null_hist1", nppiHistogramEven_8u_C3R_Ctx(s1.p, 12, NppiSize{2, 2}, hist, nullptr, lo, hi, gBuf64, c));
    PS("null_levels", nppiHistogramEven_8u_C3R_Ctx(s1.p, 12, NppiSize{2, 2}, hist, nullptr, lo, hi, gBuf64, c));
    PS("null_buf", nppiHistogramEven_8u_C3R_Ctx(s1.p, 12, NppiSize{2, 2}, hist, levels, lo, hi, nullptr, c));
    PS("zro_roi", nppiHistogramEven_8u_C3R_Ctx(s1.p, 12, NppiSize{0, 2}, hist, levels, lo, hi, gBuf64, c));
    PS("neg_roi", nppiHistogramEven_8u_C3R_Ctx(s1.p, 12, NppiSize{-1, 2}, hist, levels, lo, hi, gBuf64, c));
    PS("short_step", nppiHistogramEven_8u_C3R_Ctx(s1.p, 3, NppiSize{4, 2}, hist, levels, lo, hi, gBuf64, c));
    int badlv[3] = {1, 257, 257};
    PS("one_level", nppiHistogramEven_8u_C3R_Ctx(s1.p, 12, NppiSize{2, 2}, hist, badlv, lo, hi, gBuf64, c));
    int neglv[3] = {-1, 257, 257};
    PS("neg_levels", nppiHistogramEven_8u_C3R_Ctx(s1.p, 12, NppiSize{2, 2}, hist, neglv, lo, hi, gBuf64, c));
    Npp32s revlo[3] = {256, 0, 0}, revhi[3] = {0, 256, 256};
    PS("rev_range", nppiHistogramEven_8u_C3R_Ctx(s1.p, 12, NppiSize{2, 2}, hist, levels, revlo, revhi, gBuf64, c));
    const Npp32s *lv[3] = {ll.p, ll.p, ll.p};
    int nl[3] = {2, 2, 2};
    PS("range_null_src", nppiHistogramRange_8u_C3R_Ctx(nullptr, 12, NppiSize{2, 2}, hist, lv, nl, gBuf64, c));
    PS("range_null_levels", nppiHistogramRange_8u_C3R_Ctx(s1.p, 12, NppiSize{2, 2}, hist, nullptr, nl, gBuf64, c));
    PS("range_one_level", nppiHistogramRange_8u_C3R_Ctx(s1.p, 12, NppiSize{2, 2}, hist, lv, nl, gBuf64, c));
  }
}

// ======================= even levels host ===================================
static void probe_levels() {
  // nppiEvenLevelsHost_32s_Ctx is not exported by NPP 13.4; check the symbol.
  printf("  (no _Ctx symbol in NPP 13.4; host-only helper)\n");
}


// Focused AC4R QualityIndex probe: fully controlled per-channel values.
static void probe_qidx_ac4() {
  NppStreamContext c = ctx();
  const int W = 4, H = 2;
  const int NP = W * H;
  Npp8u base[NP] = {10, 20, 30, 40, 50, 60, 70, 80};
  std::vector<Npp8u> pa((size_t)NP * 4), pb((size_t)NP * 4);
  for (int i = 0; i < NP; ++i) {
    pa[(size_t)i * 4 + 0] = (Npp8u)(base[i]);
    pa[(size_t)i * 4 + 1] = (Npp8u)(base[i] + 5);
    pa[(size_t)i * 4 + 2] = (Npp8u)(base[i] + 7);
    pb[(size_t)i * 4 + 0] = (Npp8u)(base[i] + 2);
    pb[(size_t)i * 4 + 1] = (Npp8u)(base[i] + 3);
    pb[(size_t)i * 4 + 2] = (Npp8u)(base[i] + 1);
    pa[(size_t)i * 4 + 3] = 0;
    pb[(size_t)i * 4 + 3] = 255;
  }
  Dev<Npp8u> s1((size_t)NP * 4), s2((size_t)NP * 4);
  Dev<Npp32f> out(4);
  s1.put(pa); s2.put(pb);
  out.fill(-1e30f);
  NppStatus s = nppiQualityIndex_8u32f_AC4R_Ctx(s1.p, W * 4, s2.p, W * 4, NppiSize{W, H}, out.p, gBuf64, c);
  PS("AC4R_alphaDiff", s);
  auto v = out.get();
  printf("  out=[%.9g %.9g %.9g %.9g]\n", v[0], v[1], v[2], v[3]);
  CHK("ac4");
  // same alpha in both images
  for (int i = 0; i < NP; ++i) pb[(size_t)i * 4 + 3] = 0;
  s2.put(pb);
  out.fill(-1e30f);
  s = nppiQualityIndex_8u32f_AC4R_Ctx(s1.p, W * 4, s2.p, W * 4, NppiSize{W, H}, out.p, gBuf64, c);
  PS("AC4R_alphaSame", s);
  v = out.get();
  printf("  out=[%.9g %.9g %.9g %.9g]\n", v[0], v[1], v[2], v[3]);
  // C3R baseline with the same RGB data
  Dev<Npp8u> t1((size_t)NP * 3), t2((size_t)NP * 3);
  std::vector<Npp8u> qa((size_t)NP * 3), qb((size_t)NP * 3);
  for (int i = 0; i < NP; ++i) {
    qa[(size_t)i * 3 + 0] = pa[(size_t)i * 4 + 0];
    qa[(size_t)i * 3 + 1] = pa[(size_t)i * 4 + 1];
    qa[(size_t)i * 3 + 2] = pa[(size_t)i * 4 + 2];
    qb[(size_t)i * 3 + 0] = pb[(size_t)i * 4 + 0];
    qb[(size_t)i * 3 + 1] = pb[(size_t)i * 4 + 1];
    qb[(size_t)i * 3 + 2] = pb[(size_t)i * 4 + 2];
  }
  t1.put(qa); t2.put(qb);
  out.fill(-1e30f);
  s = nppiQualityIndex_8u32f_C3R_Ctx(t1.p, W * 3, t2.p, W * 3, NppiSize{W, H}, out.p, gBuf64, c);
  PS("C3R_same_rgb", s);
  v = out.get();
  printf("  out=[%.9g %.9g %.9g %.9g]\n", v[0], v[1], v[2], v[3]);
  CHK("c3r");
}

// QualityIndex follow-up: const/zero-mean/1x1 edge cases + 32f C3R ch0-shared
// mean/variance hypothesis (Q_k = 4*cov_k*mu0a*mu0b/((mu0a^2+mu0b^2)*(var0a+var0b))).
static void probe_qidx2() {
  NppStreamContext c = ctx();
  Dev<Npp32f> out(4);
  const int W = 5, H = 3;
  // 1. C1R: src2 constant (src1 varying)
  {
    Dev<Npp8u> s1((size_t)W * H), s2((size_t)W * H);
    s1.put(tex8(W, H, 1, 3)); s2.fill(7);
    out.fill(-1e30f);
    NppStatus s = nppiQualityIndex_8u32f_C1R_Ctx(s1.p, W, s2.p, W, NppiSize{W, H}, out.p, gBuf64, c);
    PS("C1R_src2const", s);
    printf("  out=[%g %g]\n", out.get()[0], out.get()[1]);
  }
  // 2. C1R: both constant and equal
  {
    Dev<Npp8u> s1(16), s2(16);
    s1.fill(7); s2.fill(7);
    out.fill(-1e30f);
    NppStatus s = nppiQualityIndex_8u32f_C1R_Ctx(s1.p, 4, s2.p, 4, NppiSize{4, 4}, out.p, gBuf64, c);
    PS("C1R_bothconst", s);
    printf("  out=[%g]\n", out.get()[0]);
  }
  // 3. C1R: zero means (values symmetric around 0), nonzero var
  {
    std::vector<Npp32f> m, n;
    for (int i = 0; i < W * H; ++i) { m.push_back((i & 1) ? 1.5f : -1.5f); n.push_back((i & 1) ? -0.5f : 0.5f); }
    Dev<Npp32f> s1((size_t)W * H), s2((size_t)W * H);
    s1.put(m); s2.put(n);
    out.fill(-1e30f);
    NppStatus s = nppiQualityIndex_32f_C1R_Ctx(s1.p, W * 4, s2.p, W * 4, NppiSize{W, H}, out.p, gBuf64, c);
    PS("C1R_zeromean", s);
    printf("  out=[%g %g]\n", out.get()[0], out.get()[1]);
  }
  // 4. C1R: 1x1 with zero pixels (mu=var=0 -> 0/0)
  {
    Dev<Npp32f> s1(1), s2(1);
    s1.fill(0.0f); s2.fill(0.0f);
    out.fill(-1e30f);
    NppStatus s = nppiQualityIndex_32f_C1R_Ctx(s1.p, 4, s2.p, 4, NppiSize{1, 1}, out.p, gBuf64, c);
    PS("C1R_1x1_zero", s);
    printf("  out=[%g]\n", out.get()[0]);
  }
  // 5. C3R: src1 channel0 constant, ch1/ch2 varying
  {
    std::vector<Npp8u> a = tex8(W, H, 3, 3);
    for (size_t i = 0; i < a.size(); i += 3) a[i] = 42;  // channel 0 constant
    Dev<Npp8u> s1((size_t)W * H * 3), s2((size_t)W * H * 3);
    s1.put(a); s2.put(tex8(W, H, 3, 4));
    out.fill(-1e30f);
    NppStatus s = nppiQualityIndex_8u32f_C3R_Ctx(s1.p, W * 3, s2.p, W * 3, NppiSize{W, H}, out.p, gBuf64, c);
    PS("C3R_src1ch0const", s);
    auto v = out.get();
    printf("  out=[%g %g %g %g]\n", v[0], v[1], v[2], v[3]);
  }
  // 6. C3R: src1 channel1 constant only (ch0 varying)
  {
    std::vector<Npp8u> a = tex8(W, H, 3, 3);
    for (size_t i = 0; i < a.size(); i += 3) a[i + 1] = 42;  // channel 1 constant
    Dev<Npp8u> s1((size_t)W * H * 3), s2((size_t)W * H * 3);
    s1.put(a); s2.put(tex8(W, H, 3, 4));
    out.fill(-1e30f);
    NppStatus s = nppiQualityIndex_8u32f_C3R_Ctx(s1.p, W * 3, s2.p, W * 3, NppiSize{W, H}, out.p, gBuf64, c);
    PS("C3R_src1ch1const", s);
    auto v = out.get();
    printf("  out=[%g %g %g %g]\n", v[0], v[1], v[2], v[3]);
  }
  // 7. C3R: src2 fully constant
  {
    Dev<Npp8u> s1((size_t)W * H * 3), s2((size_t)W * H * 3);
    s1.put(tex8(W, H, 3, 3)); s2.fill(9);
    out.fill(-1e30f);
    NppStatus s = nppiQualityIndex_8u32f_C3R_Ctx(s1.p, W * 3, s2.p, W * 3, NppiSize{W, H}, out.p, gBuf64, c);
    PS("C3R_src2const", s);
    auto v = out.get();
    printf("  out=[%g %g %g %g]\n", v[0], v[1], v[2], v[3]);
  }
  // 8. C3R 1x1
  {
    Dev<Npp32f> s1(3), s2(3);
    s1.put(std::vector<Npp32f>{1.f, 2.f, 3.f}); s2.put(std::vector<Npp32f>{4.f, 5.f, 6.f});
    out.fill(-1e30f);
    NppStatus s = nppiQualityIndex_32f_C3R_Ctx(s1.p, 12, s2.p, 12, NppiSize{1, 1}, out.p, gBuf64, c);
    PS("C3R_1x1", s);
    auto v = out.get();
    printf("  out=[%g %g %g %g]\n", v[0], v[1], v[2], v[3]);
  }
  // 9. 32f C3R value check vs ch0-shared hypothesis
  {
    std::vector<Npp32f> a = texf(W, H, 3, 11), b = texf(W, H, 3, 12);
    Dev<Npp32f> s1((size_t)W * H * 3), s2((size_t)W * H * 3);
    s1.put(a); s2.put(b);
    out.fill(-1e30f);
    NppStatus s = nppiQualityIndex_32f_C3R_Ctx(s1.p, W * 12, s2.p, W * 12, NppiSize{W, H}, out.p, gBuf64, c);
    PS("C3R_32f", s);
    auto v = out.get();
    printf("  out=[%g %g %g %g]\n", v[0], v[1], v[2], v[3]);
    // print per-channel stats so the hypothesis can be checked offline
    for (int k = 0; k < 3; ++k) {
      double ma = 0, mb = 0, va = 0, vb = 0, cov = 0;
      int n = W * H;
      std::vector<float> A, B;
      for (int i = 0; i < n; ++i) { A.push_back(a[(size_t)i * 3 + k]); B.push_back(b[(size_t)i * 3 + k]); }
      for (int i = 0; i < n; ++i) { ma += A[i]; mb += B[i]; }
      ma /= n; mb /= n;
      for (int i = 0; i < n; ++i) { va += (A[i] - ma) * (A[i] - ma); vb += (B[i] - mb) * (B[i] - mb); cov += (A[i] - ma) * (B[i] - mb); }
      va /= n - 1; vb /= n - 1; cov /= n - 1;
      printf("  ch%d: mu_a=%.9g mu_b=%.9g var_a=%.9g var_b=%.9g cov=%.9g\n", k, ma, mb, va, vb, cov);
    }
  }
  // 10. 16u C1R arithmetic (does it differ?)
  {
    const int n = W * H;
    std::vector<Npp16u> a, b;
    for (int i = 0; i < n; ++i) { a.push_back((Npp16u)(100 + i)); b.push_back((Npp16u)(100 + i * 2)); }
    Dev<Npp16u> s1(n), s2(n);
    s1.put(a); s2.put(b);
    out.fill(-1e30f);
    NppStatus s = nppiQualityIndex_16u32f_C1R_Ctx(s1.p, W * 2, s2.p, W * 2, NppiSize{W, H}, out.p, gBuf64, c);
    PS("C1R_16u", s);
    printf("  out=[%g]\n", out.get()[0]);
  }
  CHK("qidx2");
}

// ======================= dense GBHS dump (formula fitting) ==================
static void dump_gbsz(const char *which) {
  NppStreamContext c = ctx();
  struct Ent { const char *tag; NppStatus (*fn)(NppiSize, size_t *, NppStreamContext); };
  Ent ents[24];
  int n = 0;
  if (!strcmp(which, "qidx")) {
    ents[n++] = {"q8c1", (NppStatus (*)(NppiSize, size_t *, NppStreamContext))nppiQualityIndexGetBufferHostSize_8u32f_C1R_Ctx};
    ents[n++] = {"q16c1", (NppStatus (*)(NppiSize, size_t *, NppStreamContext))nppiQualityIndexGetBufferHostSize_16u32f_C1R_Ctx};
    ents[n++] = {"qfc1", (NppStatus (*)(NppiSize, size_t *, NppStreamContext))nppiQualityIndexGetBufferHostSize_32f_C1R_Ctx};
    ents[n++] = {"q8c3", (NppStatus (*)(NppiSize, size_t *, NppStreamContext))nppiQualityIndexGetBufferHostSize_8u32f_C3R_Ctx};
    ents[n++] = {"q8ac4", (NppStatus (*)(NppiSize, size_t *, NppStreamContext))nppiQualityIndexGetBufferHostSize_8u32f_AC4R_Ctx};
  } else if (!strcmp(which, "cir")) {
    ents[n++] = {"cir8c1", (NppStatus (*)(NppiSize, size_t *, NppStreamContext))nppiCountInRangeGetBufferHostSize_8u_C1R_Ctx};
    ents[n++] = {"cirfc1", (NppStatus (*)(NppiSize, size_t *, NppStreamContext))nppiCountInRangeGetBufferHostSize_32f_C1R_Ctx};
    ents[n++] = {"cir8c3", (NppStatus (*)(NppiSize, size_t *, NppStreamContext))nppiCountInRangeGetBufferHostSize_8u_C3R_Ctx};
    ents[n++] = {"cir8ac4", (NppStatus (*)(NppiSize, size_t *, NppStreamContext))nppiCountInRangeGetBufferHostSize_8u_AC4R_Ctx};
    ents[n++] = {"cirfc3", (NppStatus (*)(NppiSize, size_t *, NppStreamContext))nppiCountInRangeGetBufferHostSize_32f_C3R_Ctx};
    ents[n++] = {"cirfac4", (NppStatus (*)(NppiSize, size_t *, NppStreamContext))nppiCountInRangeGetBufferHostSize_32f_AC4R_Ctx};
  } else if (!strcmp(which, "metric")) {
    ents[n++] = {"mse8c1", (NppStatus (*)(NppiSize, size_t *, NppStreamContext))nppiMSEGetBufferHostSize_8u_C1R_Ctx};
    ents[n++] = {"mse8c3", (NppStatus (*)(NppiSize, size_t *, NppStreamContext))nppiMSEGetBufferHostSize_8u_C3R_Ctx};
    ents[n++] = {"psnr8c1", (NppStatus (*)(NppiSize, size_t *, NppStreamContext))nppiPSNRGetBufferHostSize_8u_C1R_Ctx};
    ents[n++] = {"ssim8c1", (NppStatus (*)(NppiSize, size_t *, NppStreamContext))nppiSSIMGetBufferHostSize_8u_C1R_Ctx};
    ents[n++] = {"ssim8c3", (NppStatus (*)(NppiSize, size_t *, NppStreamContext))nppiSSIMGetBufferHostSize_8u_C3R_Ctx};
    ents[n++] = {"msssim8c1", (NppStatus (*)(NppiSize, size_t *, NppStreamContext))nppiMSSSIMGetBufferHostSize_8u_C1R_Ctx};
    ents[n++] = {"wmsssim8c1", (NppStatus (*)(NppiSize, size_t *, NppStreamContext))nppiWMSSSIMGetBufferHostSize_8u_C1R_Ctx};
    ents[n++] = {"wmsssim8c3", (NppStatus (*)(NppiSize, size_t *, NppStreamContext))nppiWMSSSIMGetBufferHostSize_8u_C3R_Ctx};
  } else if (!strcmp(which, "batch")) {
    ents[n++] = {"mseB8c1", (NppStatus (*)(NppiSize, size_t *, NppStreamContext))nppiMSEBatchGetBufferHostSize_8u_C1R_Ctx};
    ents[n++] = {"mseB8c3", (NppStatus (*)(NppiSize, size_t *, NppStreamContext))nppiMSEBatchGetBufferHostSize_8u_C3R_Ctx};
    ents[n++] = {"psnrB8c1", (NppStatus (*)(NppiSize, size_t *, NppStreamContext))nppiPSNRBatchGetBufferHostSize_8u_C1R_Ctx};
    ents[n++] = {"ssimB8c1", (NppStatus (*)(NppiSize, size_t *, NppStreamContext))nppiSSIMBatchGetBufferHostSize_8u_C1R_Ctx};
    ents[n++] = {"ssimB8c3", (NppStatus (*)(NppiSize, size_t *, NppStreamContext))nppiSSIMBatchGetBufferHostSize_8u_C3R_Ctx};
    ents[n++] = {"wmsssimB8c1", (NppStatus (*)(NppiSize, size_t *, NppStreamContext))nppiWMSSSIMBatchGetBufferHostSize_8u_C1R_Ctx};
    ents[n++] = {"wmsssimB8c3", (NppStatus (*)(NppiSize, size_t *, NppStreamContext))nppiWMSSSIMBatchGetBufferHostSize_8u_C3R_Ctx};
  } else {
    printf("unknown family\n");
    return;
  }
  static const int sweep[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 20,
                              24, 28, 31, 32, 33, 40, 48, 60, 63, 64, 65, 80, 96, 112, 120, 127, 128, 129, 160,
                              176, 192, 224, 240, 255, 256, 257, 320, 384, 448, 511, 512, 513, 640, 768, 896, 1023,
                              1024, 1025, 1280, 1536, 2047, 2048, 2560, 3072, 4096, 6144, 8192};
  const int ns = (int)(sizeof(sweep) / sizeof(sweep[0]));
  struct P { int w, h; };
  static const P pairs[] = {{2, 3},  {3, 2},   {17, 13},  {13, 17},  {32, 32},  {48, 48},  {64, 64},  {100, 100},
                            {128, 128}, {176, 176}, {256, 256}, {320, 240}, {640, 480}, {1024, 768}, {1920, 1080},
                            {2048, 2048}, {4096, 4096}, {8192, 64}, {64, 8192}, {1, 4096}, {4096, 1}};
  const int np = (int)(sizeof(pairs) / sizeof(pairs[0]));
  for (int e = 0; e < n; ++e) {
    for (int i = 0; i < ns; ++i) {
      size_t buf = (size_t)-1;
      NppStatus s = ents[e].fn(NppiSize{sweep[i], 8}, &buf, c);
      printf("%s %d 8 %lld %d\n", ents[e].tag, sweep[i], (long long)buf, (int)s);
      buf = (size_t)-1;
      s = ents[e].fn(NppiSize{8, sweep[i]}, &buf, c);
      printf("%s 8 %d %lld %d\n", ents[e].tag, sweep[i], (long long)buf, (int)s);
      if (sweep[i] <= 18) {
        buf = (size_t)-1;
        s = ents[e].fn(NppiSize{sweep[i], 1}, &buf, c);
        printf("%s %d 1 %lld %d\n", ents[e].tag, sweep[i], (long long)buf, (int)s);
        buf = (size_t)-1;
        s = ents[e].fn(NppiSize{1, sweep[i]}, &buf, c);
        printf("%s 1 %d %lld %d\n", ents[e].tag, sweep[i], (long long)buf, (int)s);
      }
    }
    for (int i = 0; i < np; ++i) {
      size_t buf = (size_t)-1;
      NppStatus s = ents[e].fn(NppiSize{pairs[i].w, pairs[i].h}, &buf, c);
      printf("%s %d %d %lld %d\n", ents[e].tag, pairs[i].w, pairs[i].h, (long long)buf, (int)s);
    }
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <probe>\n", argv[0]);
    return 2;
  }
  cudaMalloc(&gBuf64, 64 << 20);
  cudaError_t ce = cudaGetLastError();
  if (ce != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(ce));
    return 3;
  }
  const char *p = argv[1];
  setbuf(stdout, nullptr);  // unbuffered: survive crashes when redirected
  if (!strcmp(p, "cir_dev_all")) probe_cir("dev_all");
  else if (!strcmp(p, "cir_devc_hostb")) probe_cir("devc_hostb");
  else if (!strcmp(p, "cir_host_all")) probe_cir("host_all");
  else if (!strcmp(p, "cir_bounds")) probe_cir_bounds();
  else if (!strcmp(p, "qidx")) probe_qidx();
  else if (!strcmp(p, "qidx_ac4")) probe_qidx_ac4();
  else if (!strcmp(p, "qidx2")) probe_qidx2();
  else if (!strcmp(p, "mse")) probe_mse_psnr();
  else if (!strcmp(p, "ssim")) probe_ssim();
  else if (!strcmp(p, "msssim")) probe_msssim();
  else if (!strcmp(p, "ssimb2")) probe_ssimb2();
  else if (!strcmp(p, "wmss2")) probe_wmss2();
  else if (!strcmp(p, "wmss3")) probe_wmss3();
  else if (!strcmp(p, "wmss4")) probe_wmss4();
  else if (!strcmp(p, "hist")) probe_hist();
  else if (!strcmp(p, "radial")) probe_radial();
  else if (!strcmp(p, "rect")) probe_rect();
  else if (!strncmp(p, "batch", 5)) probe_batch(p[5] ? atoi(p + 5) : 0);
  else if (!strncmp(p, "gbsz:", 5)) dump_gbsz(p + 5);
  else if (!strcmp(p, "bufsz")) probe_bufsz();
  else if (!strcmp(p, "contract")) probe_contract();
  else if (!strcmp(p, "levels")) probe_levels();
  else { fprintf(stderr, "unknown probe %s\n", p); return 2; }
  fflush(stdout);
  return 0;
}
