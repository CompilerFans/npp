// NPP 13.4 radial profile (circular/elliptical) semantics probe for package s2.
//  usage: radial_probe map|vals|errs
#include <cuda_runtime.h>
#include <npp.h>
#include <nppi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>

extern "C" {
NppStatus nppGetStreamContext(NppStreamContext *pNppStreamContext);
}

static NppStreamContext ctx() {
  NppStreamContext c{};
  nppGetStreamContext(&c);
  return c;
}

static const char *sname(NppStatus s) {
  switch (s) {
    case NPP_SUCCESS: return "OK";
    case NPP_NULL_POINTER_ERROR: return "NULLPTR";
    case NPP_SIZE_ERROR: return "SIZE";
    case NPP_STEP_ERROR: return "STEP";
    case NPP_RANGE_ERROR: return "RANGE";
    case NPP_BAD_ARGUMENT_ERROR: return "BADARG";
    default: break;
  }
  static char b[4][24];
  static int i = 0;
  i = (i + 1) & 3;
  snprintf(b[i], 24, "ST%d", (int)s);
  return b[i];
}

template <typename T> struct Dev {
  T *p = nullptr;
  size_t n = 0;
  explicit Dev(size_t c) : n(c) { cudaMalloc(&p, c * sizeof(T)); }
  ~Dev() { if (p) cudaFree(p); }
  void put(const std::vector<T> &h) { cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice); }
  std::vector<T> get() const {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
  }
};

static char b36(int v) { return v < 10 ? (char)('0' + v) : (char)('a' + v - 10); }

static void ringMap(const char *tag, int W, int H, float cx, float cy, int samples, bool elliptical, float ratio,
                    float orient) {
  Dev<Npp32f> src((size_t)W * H);
  Dev<NppiProfileData> prof((size_t)samples + 4);
  NppStreamContext c = ctx();
  std::vector<char> map((size_t)W * H, '.');
  std::vector<float> img((size_t)W * H, 0.0f);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      img[(size_t)y * W + x] = 1.0f;
      src.put(img);
      img[(size_t)y * W + x] = 0.0f;
      prof.put(std::vector<NppiProfileData>((size_t)samples + 4, NppiProfileData{0, -1e30f, -1e30f}));
      NppStatus s =
          elliptical ? nppiEllipticalRadialProfile_32f_C1R_Ctx(src.p, W * 4, NppiSize{W, H}, NppiPoint32f{cx, cy},
                                                                ratio, orient, prof.p, (Npp16u)samples, c)
                     : nppiCircularRadialProfile_32f_C1R_Ctx(src.p, W * 4, NppiSize{W, H}, NppiPoint32f{cx, cy},
                                                             prof.p, (Npp16u)samples, c);
      if (s != NPP_SUCCESS) {
        if (x == 0 && y == 0) printf("%s status=%d(%s)\n", tag, (int)s, sname(s));
        continue;
      }
      std::vector<NppiProfileData> v = prof.get();
      int hit = 0, idx = -1;
      for (int i = 0; i < samples; ++i) {
        if (v[i].nPixels > 0 && v[i].nMeanIntensity != 0.0f) {
          ++hit;
          idx = i;
        }
      }
      if (hit == 1) map[(size_t)y * W + x] = b36(idx);
      else if (hit > 1) map[(size_t)y * W + x] = '?';
    }
  }
  printf("%s W=%d H=%d c=(%g,%g) S=%d ell=%d ratio=%g orient=%g\n", tag, W, H, cx, cy, samples, (int)elliptical,
         ratio, orient);
  for (int y = 0; y < H; ++y) {
    printf("  ");
    for (int x = 0; x < W; ++x) putchar(map[(size_t)y * W + x]);
    printf("\n");
  }
  std::vector<int> cnt((size_t)samples, 0);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      char ch = map[(size_t)y * W + x];
      if (ch != '.' && ch != '?') cnt[ch <= '9' ? ch - '0' : ch - 'a' + 10]++;
    }
  printf("  counts:");
  for (int i = 0; i < samples; ++i) printf(" %d", cnt[i]);
  printf("\n");
}

static void mapSection() {
  ringMap("circ33_24", 33, 33, 16.5f, 16.5f, 24, false, 0, 0);
  ringMap("circ33_12", 33, 33, 16.5f, 16.5f, 12, false, 0, 0);
  ringMap("circ33_48", 33, 33, 16.5f, 16.5f, 48, false, 0, 0);
  ringMap("circ33int_24", 33, 33, 16.0f, 16.0f, 24, false, 0, 0);
  ringMap("circ33off_16", 33, 33, 10.5f, 20.5f, 16, false, 0, 0);
  ringMap("circ40x25_24", 40, 25, 20.5f, 12.5f, 24, false, 0, 0);
  ringMap("circ_out_8", 33, 33, -3.5f, -2.5f, 8, false, 0, 0);
  ringMap("ell_ratio2", 33, 33, 16.5f, 16.5f, 24, true, 2.0f, 0.0f);
  ringMap("ell_ratio05", 33, 33, 16.5f, 16.5f, 24, true, 0.5f, 0.0f);
  ringMap("ell_rot90", 33, 33, 16.5f, 16.5f, 24, true, 2.0f, 1.5707963267948966f);
  ringMap("ell_rot45", 33, 33, 16.5f, 16.5f, 24, true, 2.0f, 0.7853981633974483f);
}

static void valSection() {
  const int W = 33, H = 33, S = 24;
  Dev<Npp32f> src((size_t)W * H);
  Dev<NppiProfileData> prof(S);
  NppStreamContext c = ctx();
  std::vector<Npp32f> img((size_t)W * H);
  for (size_t i = 0; i < img.size(); ++i) img[i] = 3.0f;
  src.put(img);
  prof.put(std::vector<NppiProfileData>(S, NppiProfileData{0, -1e30f, -1e30f}));
  NppStatus s = nppiCircularRadialProfile_32f_C1R_Ctx(src.p, W * 4, NppiSize{W, H}, NppiPoint32f{16.5f, 16.5f},
                                                      prof.p, S, c);
  printf("uniform3 st=%d(%s)\n", (int)s, sname(s));
  std::vector<NppiProfileData> v = prof.get();
  for (int i = 0; i < S; ++i)
    printf("  r%02d n=%d mean=%.9g sd=%.9g\n", i, v[i].nPixels, v[i].nMeanIntensity, v[i].nStdDevIntensity);
  std::vector<Npp32f> img2((size_t)W * H, 0.0f);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) img2[(size_t)y * W + x] = ((x + y) & 1) ? 1.0f : 0.0f;
  src.put(img2);
  prof.put(std::vector<NppiProfileData>(S, NppiProfileData{0, -1e30f, -1e30f}));
  s = nppiCircularRadialProfile_32f_C1R_Ctx(src.p, W * 4, NppiSize{W, H}, NppiPoint32f{16.5f, 16.5f}, prof.p, S, c);
  printf("checker st=%d(%s)\n", (int)s, sname(s));
  v = prof.get();
  for (int i = 0; i < 6; ++i)
    printf("  r%02d n=%d mean=%.9g sd=%.9g\n", i, v[i].nPixels, v[i].nMeanIntensity, v[i].nStdDevIntensity);
  std::vector<Npp32f> img3((size_t)W * H);
  for (size_t i = 0; i < img3.size(); ++i) img3[i] = (Npp32f)((i * 37) % 17) - 8.0f;
  src.put(img3);
  prof.put(std::vector<NppiProfileData>(S, NppiProfileData{0, -1e30f, -1e30f}));
  s = nppiCircularRadialProfile_32f_C1R_Ctx(src.p, W * 4, NppiSize{W, H}, NppiPoint32f{-3.5f, -2.5f}, prof.p, S, c);
  printf("outside st=%d(%s)\n", (int)s, sname(s));
  v = prof.get();
  for (int i = 0; i < 8; ++i)
    printf("  r%02d n=%d mean=%.9g sd=%.9g\n", i, v[i].nPixels, v[i].nMeanIntensity, v[i].nStdDevIntensity);
}

static Npp32f *gSrc = nullptr;
static NppiProfileData *gProf = nullptr;
static NppStreamContext gCtx;

#define CASE(tag, call)                                     \
  static void fn_##tag() {                                  \
    NppStatus st_ = (call);                                 \
    printf("  %-16s st=%d(%s)\n", #tag, (int)st_, sname(st_)); \
    cudaError_t e = cudaDeviceSynchronize();                \
    printf("  %-16s cuda=%s\n", #tag, cudaGetErrorString(e)); \
    cudaGetLastError();                                     \
  }

CASE(null_src, nppiCircularRadialProfile_32f_C1R_Ctx(nullptr, 33 * 4, NppiSize{33, 33}, NppiPoint32f{16.5f, 16.5f}, gProf, 24, gCtx))
CASE(null_prof, nppiCircularRadialProfile_32f_C1R_Ctx(gSrc, 33 * 4, NppiSize{33, 33}, NppiPoint32f{16.5f, 16.5f}, nullptr, 24, gCtx))
CASE(zero_roi, nppiCircularRadialProfile_32f_C1R_Ctx(gSrc, 33 * 4, NppiSize{0, 33}, NppiPoint32f{16.5f, 16.5f}, gProf, 24, gCtx))
CASE(zero_w0h0, nppiCircularRadialProfile_32f_C1R_Ctx(gSrc, 33 * 4, NppiSize{0, 0}, NppiPoint32f{16.5f, 16.5f}, gProf, 24, gCtx))
CASE(neg_roi, nppiCircularRadialProfile_32f_C1R_Ctx(gSrc, 33 * 4, NppiSize{-1, 33}, NppiPoint32f{16.5f, 16.5f}, gProf, 24, gCtx))
CASE(neg_h, nppiCircularRadialProfile_32f_C1R_Ctx(gSrc, 33 * 4, NppiSize{33, -2}, NppiPoint32f{16.5f, 16.5f}, gProf, 24, gCtx))
CASE(step4, nppiCircularRadialProfile_32f_C1R_Ctx(gSrc, 4, NppiSize{33, 33}, NppiPoint32f{16.5f, 16.5f}, gProf, 24, gCtx))
CASE(step_odd, nppiCircularRadialProfile_32f_C1R_Ctx(gSrc, 33 * 4 - 5, NppiSize{33, 33}, NppiPoint32f{16.5f, 16.5f}, gProf, 24, gCtx))
CASE(step_zero, nppiCircularRadialProfile_32f_C1R_Ctx(gSrc, 0, NppiSize{33, 33}, NppiPoint32f{16.5f, 16.5f}, gProf, 24, gCtx))
CASE(step_neg, nppiCircularRadialProfile_32f_C1R_Ctx(gSrc, -132, NppiSize{33, 33}, NppiPoint32f{16.5f, 16.5f}, gProf, 24, gCtx))
CASE(samples0, nppiCircularRadialProfile_32f_C1R_Ctx(gSrc, 33 * 4, NppiSize{33, 33}, NppiPoint32f{16.5f, 16.5f}, gProf, 0, gCtx))
CASE(samples1, nppiCircularRadialProfile_32f_C1R_Ctx(gSrc, 33 * 4, NppiSize{33, 33}, NppiPoint32f{16.5f, 16.5f}, gProf, 1, gCtx))
CASE(samples2, nppiCircularRadialProfile_32f_C1R_Ctx(gSrc, 33 * 4, NppiSize{33, 33}, NppiPoint32f{16.5f, 16.5f}, gProf, 2, gCtx))
CASE(samples1000, nppiCircularRadialProfile_32f_C1R_Ctx(gSrc, 33 * 4, NppiSize{33, 33}, NppiPoint32f{16.5f, 16.5f}, gProf, 1000, gCtx))
CASE(samples4096, nppiCircularRadialProfile_32f_C1R_Ctx(gSrc, 33 * 4, NppiSize{33, 33}, NppiPoint32f{16.5f, 16.5f}, gProf, 4096, gCtx))
CASE(samples65535, nppiCircularRadialProfile_32f_C1R_Ctx(gSrc, 33 * 4, NppiSize{33, 33}, NppiPoint32f{16.5f, 16.5f}, gProf, 65535, gCtx))
CASE(center_far, nppiCircularRadialProfile_32f_C1R_Ctx(gSrc, 33 * 4, NppiSize{33, 33}, NppiPoint32f{1e6f, -1e6f}, gProf, 24, gCtx))
CASE(center_nan, nppiCircularRadialProfile_32f_C1R_Ctx(gSrc, 33 * 4, NppiSize{33, 33}, NppiPoint32f{NAN, NAN}, gProf, 24, gCtx))
CASE(center_out_ok, nppiCircularRadialProfile_32f_C1R_Ctx(gSrc, 33 * 4, NppiSize{33, 33}, NppiPoint32f{-3.5f, -2.5f}, gProf, 24, gCtx))
CASE(roi_off_neg, nppiCircularRadialProfile_32f_C1R_Ctx(gSrc, 33 * 4, NppiSize{-33, 33}, NppiPoint32f{-3.5f, -2.5f}, gProf, 24, gCtx))
CASE(ell_null_src, nppiEllipticalRadialProfile_32f_C1R_Ctx(nullptr, 33 * 4, NppiSize{33, 33}, NppiPoint32f{16.5f, 16.5f}, 2.f, 0.f, gProf, 24, gCtx))
CASE(ell_null_prof, nppiEllipticalRadialProfile_32f_C1R_Ctx(gSrc, 33 * 4, NppiSize{33, 33}, NppiPoint32f{16.5f, 16.5f}, 2.f, 0.f, nullptr, 24, gCtx))
CASE(ell_neg_roi, nppiEllipticalRadialProfile_32f_C1R_Ctx(gSrc, 33 * 4, NppiSize{-1, 33}, NppiPoint32f{16.5f, 16.5f}, 2.f, 0.f, gProf, 24, gCtx))
CASE(ell_zero_roi, nppiEllipticalRadialProfile_32f_C1R_Ctx(gSrc, 33 * 4, NppiSize{0, 33}, NppiPoint32f{16.5f, 16.5f}, 2.f, 0.f, gProf, 24, gCtx))
CASE(ell_step4, nppiEllipticalRadialProfile_32f_C1R_Ctx(gSrc, 4, NppiSize{33, 33}, NppiPoint32f{16.5f, 16.5f}, 2.f, 0.f, gProf, 24, gCtx))
CASE(ell_ratio0, nppiEllipticalRadialProfile_32f_C1R_Ctx(gSrc, 33 * 4, NppiSize{33, 33}, NppiPoint32f{16.5f, 16.5f}, 0.f, 0.f, gProf, 24, gCtx))
CASE(ell_ratio_neg, nppiEllipticalRadialProfile_32f_C1R_Ctx(gSrc, 33 * 4, NppiSize{33, 33}, NppiPoint32f{16.5f, 16.5f}, -2.f, 0.f, gProf, 24, gCtx))
CASE(ell_ratio_tiny, nppiEllipticalRadialProfile_32f_C1R_Ctx(gSrc, 33 * 4, NppiSize{33, 33}, NppiPoint32f{16.5f, 16.5f}, 1e-6f, 0.f, gProf, 24, gCtx))
CASE(ell_ratio_1, nppiEllipticalRadialProfile_32f_C1R_Ctx(gSrc, 33 * 4, NppiSize{33, 33}, NppiPoint32f{16.5f, 16.5f}, 1.f, 0.f, gProf, 24, gCtx))
CASE(ell_ratio_nan, nppiEllipticalRadialProfile_32f_C1R_Ctx(gSrc, 33 * 4, NppiSize{33, 33}, NppiPoint32f{16.5f, 16.5f}, NAN, 0.f, gProf, 24, gCtx))
CASE(ell_orient_nan, nppiEllipticalRadialProfile_32f_C1R_Ctx(gSrc, 33 * 4, NppiSize{33, 33}, NppiPoint32f{16.5f, 16.5f}, 2.f, NAN, gProf, 24, gCtx))
CASE(ell_samples0, nppiEllipticalRadialProfile_32f_C1R_Ctx(gSrc, 33 * 4, NppiSize{33, 33}, NppiPoint32f{16.5f, 16.5f}, 2.f, 0.f, gProf, 0, gCtx))
CASE(ell_samples1, nppiEllipticalRadialProfile_32f_C1R_Ctx(gSrc, 33 * 4, NppiSize{33, 33}, NppiPoint32f{16.5f, 16.5f}, 2.f, 0.f, gProf, 1, gCtx))

static void (*gCases[])() = {fn_null_src, fn_null_prof, fn_zero_roi, fn_zero_w0h0, fn_neg_roi, fn_neg_h, fn_step4, fn_step_odd, fn_step_zero, fn_step_neg, fn_samples0, fn_samples1, fn_samples2, fn_samples1000, fn_samples4096, fn_samples65535, fn_center_far, fn_center_nan, fn_center_out_ok, fn_roi_off_neg, fn_ell_null_src, fn_ell_null_prof, fn_ell_neg_roi, fn_ell_zero_roi, fn_ell_step4, fn_ell_ratio0, fn_ell_ratio_neg, fn_ell_ratio_tiny, fn_ell_ratio_1, fn_ell_ratio_nan, fn_ell_orient_nan, fn_ell_samples0, fn_ell_samples1};
static const char *gNames[] = {"null_src", "null_prof", "zero_roi", "zero_w0h0", "neg_roi", "neg_h", "step4", "step_odd", "step_zero", "step_neg", "samples0", "samples1", "samples2", "samples1000", "samples4096", "samples65535", "center_far", "center_nan", "center_out_ok", "roi_off_neg", "ell_null_src", "ell_null_prof", "ell_neg_roi", "ell_zero_roi", "ell_step4", "ell_ratio0", "ell_ratio_neg", "ell_ratio_tiny", "ell_ratio_1", "ell_ratio_nan", "ell_orient_nan", "ell_samples0", "ell_samples1"};
static const int gNumCases = 33;

static void initGlobals() {
  static Dev<Npp32f> *src = nullptr;
  static Dev<NppiProfileData> *prof = nullptr;
  if (!src) {
    src = new Dev<Npp32f>((size_t)33 * 33);
    prof = new Dev<NppiProfileData>(70000);
    src->put(std::vector<Npp32f>((size_t)33 * 33, 1.0f));
  }
  gSrc = src->p;
  gProf = prof->p;
  gCtx = ctx();
}

static void runOneCase(int i) {
  initGlobals();
  gCases[i]();
  cudaDeviceSynchronize();
  cudaGetLastError();
}

static void listCases() {
  for (int i = 0; i < gNumCases; ++i) printf("%d %s\n", i, gNames[i]);
}

int main(int argc, char **argv) {
  const char *what = argc > 1 ? argv[1] : "map";
  if (!strcmp(what, "map")) mapSection();
  else if (!strcmp(what, "vals")) valSection();
  else if (!strcmp(what, "list")) listCases();
  else if (!strcmp(what, "case")) {
    initGlobals();
    runOneCase(atoi(argv[2]));
  }
  return 0;
}
