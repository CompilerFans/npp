// 5090 probe for package f10 (nppiFilterGauss*/nppiFilterWienerBorder semantics).
// Not part of the build; compiled ad-hoc on the RTX 5090 against NVIDIA NPP 13.4.
#include <npp.h>
extern "C" {
cudaStream_t nppGetStream(void);
NppStatus nppGetStreamContext(NppStreamContext *pNppStreamContext);
}
#include <nppi_filtering_functions.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static NppStreamContext ctx() {
  NppStreamContext c{};
  nppGetStreamContext(&c);
  return c;
}

static void dump_f(const char *tag, const std::vector<float> &v, int w, int h, int y) {
  printf("%s row%d:", tag, y);
  for (int x = 0; x < w; ++x) printf(" %.9g", v[(size_t)y * w + x]);
  printf("\n");
}

static void colsum(const std::vector<float> &v, int w, int h, std::vector<float> &out) {
  out.assign(w, 0.f);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) out[x] += v[(size_t)y * w + x];
}

// ---------------- stage gauss32 : kernel coefficients, mask validation -------
static void gauss32() {
  const int W = 41, H = 41;
  std::vector<float> src((size_t)W * H, 0.f), dst((size_t)W * H, -1.f);
  src[(size_t)(H / 2) * W + W / 2] = 1.0f;
  float *d_src, *d_dst;
  cudaMalloc(&d_src, src.size() * 4);
  cudaMalloc(&d_dst, dst.size() * 4);
  cudaMemcpy(d_src, src.data(), src.size() * 4, cudaMemcpyHostToDevice);
  const NppiMaskSize masks[] = {NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_5_X_5, NPP_MASK_SIZE_7_X_7,
                               NPP_MASK_SIZE_9_X_9, NPP_MASK_SIZE_11_X_11, NPP_MASK_SIZE_13_X_13,
                               NPP_MASK_SIZE_15_X_15, NPP_MASK_SIZE_1_X_3, NPP_MASK_SIZE_3_X_1,
                               (NppiMaskSize)0, (NppiMaskSize)900, (NppiMaskSize)-1};
  for (NppiMaskSize ms : masks) {
    int m = (int)ms;
    memset(dst.data(), -1, dst.size() * 4);
    cudaMemcpy(d_dst, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice);
    NppStatus st = nppiFilterGauss_32f_C1R_Ctx(d_src, W * 4, d_dst, W * 4, NppiSize{W, H}, ms, ctx());
    cudaMemcpy(dst.data(), d_dst, dst.size() * 4, cudaMemcpyDeviceToHost);
    std::vector<float> cs;
    colsum(dst, W, H, cs);
    double total = 0;
    for (float f : cs) total += f;
    printf("mask=%d status=%d total=%.9g\n", m, (int)st, total);
    if (st == 0) {
      dump_f("  centerrow", dst, W, H, H / 2);
      printf("  colsum   :");
      for (int x = W / 2 - 9; x <= W / 2 + 9; ++x) printf(" %.9g", cs[x]);
      printf("\n");
    }
  }
  cudaFree(d_src); cudaFree(d_dst);
}

// ---------------- stage gauss8 : rounding + integer saturation ---------------
static void gauss8() {
  const int W = 9, H = 9;
  std::vector<unsigned char> src((size_t)W * H), dst((size_t)W * H, 0xAA);
  for (size_t i = 0; i < src.size(); ++i) src[i] = (unsigned char)((i * 37 + 11) & 0xFF);
  unsigned char *d_src, *d_dst;
  cudaMalloc(&d_src, src.size());
  cudaMalloc(&d_dst, dst.size());
  cudaMemcpy(d_src, src.data(), src.size(), cudaMemcpyHostToDevice);
  const NppiMaskSize masks[] = {NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_5_X_5, NPP_MASK_SIZE_7_X_7};
  for (NppiMaskSize m : masks) {
    memset(dst.data(), 0xAA, dst.size());
    cudaMemcpy(d_dst, dst.data(), dst.size(), cudaMemcpyHostToDevice);
    NppStatus st = nppiFilterGauss_8u_C1R_Ctx(d_src, W, d_dst, W, NppiSize{W, H}, m, ctx());
    cudaMemcpy(dst.data(), d_dst, dst.size(), cudaMemcpyDeviceToHost);
    printf("8u mask=%d status=%d\n  out:", m, (int)st);
    for (int y = 2; y <= 4; ++y) {
      for (int x = 0; x < W; ++x) printf(" %3d", dst[(size_t)y * W + x]);
      printf("\n     ");
    }
    printf("\n");
  }
  // 16u / 16s saturation behaviour with a bright delta
  std::vector<unsigned short> s16((size_t)W * H, 0), d16((size_t)W * H, 0xFFFF);
  std::vector<short> s16s((size_t)W * H, 0), d16s((size_t)W * H, -1);
  s16[(size_t)4 * W + 4] = 65535;
  s16s[(size_t)4 * W + 4] = 32767;
  unsigned short *ds, *dd; short *dss, *ddd;
  cudaMalloc(&ds, s16.size() * 2); cudaMalloc(&dd, d16.size() * 2);
  cudaMalloc(&dss, s16s.size() * 2); cudaMalloc(&ddd, d16s.size() * 2);
  cudaMemcpy(ds, s16.data(), s16.size() * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dss, s16s.data(), s16s.size() * 2, cudaMemcpyHostToDevice);
  NppStatus a = nppiFilterGauss_16u_C1R_Ctx(ds, W * 2, dd, W * 2, NppiSize{W, H}, NPP_MASK_SIZE_3_X_3, ctx());
  NppStatus b = nppiFilterGauss_16s_C1R_Ctx(dss, W * 2, ddd, W * 2, NppiSize{W, H}, NPP_MASK_SIZE_3_X_3, ctx());
  cudaMemcpy(d16.data(), dd, d16.size() * 2, cudaMemcpyDeviceToHost);
  cudaMemcpy(d16s.data(), ddd, d16s.size() * 2, cudaMemcpyDeviceToHost);
  printf("16u st=%d 16s st=%d\n16u:", (int)a, (int)b);
  for (int x = 0; x < W; ++x) printf(" %d", d16[(size_t)4 * W + x]);
  printf("\n16s:");
  for (int x = 0; x < W; ++x) printf(" %d", d16s[(size_t)4 * W + x]);
  printf("\n");
  cudaFree(ds); cudaFree(dd); cudaFree(dss); cudaFree(ddd);
  cudaFree(d_src); cudaFree(d_dst);
}

// ---------------- stage edge : plain margin reads ----------------------------
static void edge() {
  const int W = 5, H = 5, M = 2;           // margin for 5x5 mask
  const int BW = W + 2 * M, BH = H + 2 * M;
  for (int pass = 0; pass < 2; ++pass) {
    std::vector<unsigned char> buf((size_t)BW * BH);
    for (int y = 0; y < BH; ++y)
      for (int x = 0; x < BW; ++x)
        buf[(size_t)y * BW + x] = pass == 0 ? 9 : (unsigned char)(100 + y * BW + x);
    std::vector<unsigned char> dst((size_t)W * H, 0xAA);
    unsigned char *d_buf, *d_dst;
    cudaMalloc(&d_buf, buf.size());
    cudaMalloc(&d_dst, dst.size());
    cudaMemcpy(d_buf, buf.data(), buf.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, dst.data(), dst.size(), cudaMemcpyHostToDevice);
    unsigned char *p = d_buf + (size_t)M * BW + M;
    NppStatus st = nppiFilterGauss_8u_C1R_Ctx(p, BW, d_dst, W, NppiSize{W, H}, NPP_MASK_SIZE_5_X_5, ctx());
    cudaMemcpy(dst.data(), d_dst, dst.size(), cudaMemcpyDeviceToHost);
    printf("pass=%d status=%d dst:", pass, (int)st);
    for (int i = 0; i < 3; ++i) printf(" %d", dst[i]);
    printf(" | corner %d center %d\n", dst[0], dst[(size_t)2 * W + 2]);
    cudaFree(d_buf); cudaFree(d_dst);
  }
}

// ---------------- stage gborder : border types -------------------------------
static void gborder() {
  const int W = 5, H = 5;
  std::vector<unsigned char> src((size_t)W * H), dst((size_t)W * H, 0xAA);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) src[(size_t)y * W + x] = (unsigned char)(y * W + x);
  unsigned char *d_src, *d_dst;
  cudaMalloc(&d_src, src.size()); cudaMalloc(&d_dst, dst.size());
  cudaMemcpy(d_src, src.data(), src.size(), cudaMemcpyHostToDevice);
  const int bs[] = {0, 1, 2, 3, 4, 5, 100, -1};
  for (int b : bs) {
    memset(dst.data(), 0xAA, dst.size());
    cudaMemcpy(d_dst, dst.data(), dst.size(), cudaMemcpyHostToDevice);
    NppStatus st = nppiFilterGaussBorder_8u_C1R_Ctx(d_src, W, NppiSize{W, H}, NppiPoint{0, 0}, d_dst, W,
                                                   NppiSize{W, H}, NPP_MASK_SIZE_3_X_3, (NppiBorderType)b, ctx());
    cudaMemcpy(dst.data(), d_dst, dst.size(), cudaMemcpyDeviceToHost);
    printf("border=%d mask=3x3 status=%d dst[0..4]:", b, (int)st);
    for (int x = 0; x < W; ++x) printf(" %d", dst[x]);
    printf("\n");
  }
  const NppiMaskSize masks[] = {NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_5_X_5, NPP_MASK_SIZE_7_X_7,
                                NPP_MASK_SIZE_9_X_9, NPP_MASK_SIZE_11_X_11, NPP_MASK_SIZE_13_X_13,
                                NPP_MASK_SIZE_15_X_15, NPP_MASK_SIZE_1_X_3, (NppiMaskSize)0, (NppiMaskSize)900};
  for (NppiMaskSize m : masks) {
    memset(dst.data(), 0xAA, dst.size());
    cudaMemcpy(d_dst, dst.data(), dst.size(), cudaMemcpyHostToDevice);
    NppStatus st = nppiFilterGaussBorder_8u_C1R_Ctx(d_src, W, NppiSize{W, H}, NppiPoint{0, 0}, d_dst, W,
                                                   NppiSize{W, H}, m, NPP_BORDER_REPLICATE, ctx());
    cudaMemcpy(dst.data(), d_dst, dst.size(), cudaMemcpyDeviceToHost);
    printf("replicate mask=%d status=%d dst0=%d\n", (int)m, (int)st, dst[0]);
  }
  // mask 15x15 needs a bigger image
  cudaFree(d_src); cudaFree(d_dst);
}

// ---------------- stage adv : GaussAdvanced plain ----------------------------
static void adv() {
  const int W = 21, H = 21;
  std::vector<float> src((size_t)W * H, 0.f), dst((size_t)W * H, -1.f);
  src[(size_t)H / 2 * W + W / 2] = 1.0f;
  float *d_src, *d_dst;
  cudaMalloc(&d_src, src.size() * 4); cudaMalloc(&d_dst, dst.size() * 4);
  cudaMemcpy(d_src, src.data(), src.size() * 4, cudaMemcpyHostToDevice);
  std::vector<float> k3{0.25f, 0.5f, 0.25f};
  std::vector<float> k5{0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};
  std::vector<float> k1{1.0f};
  std::vector<float> k2{0.5f, 0.5f};
  std::vector<float> kun{1.f, 1.f, 1.f};
  std::vector<float> kneg{0.5f, -0.5f, 1.f};
  std::vector<float> kbig{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f};
  struct E { int taps; std::vector<float> *k; const char *tag; } cases[] = {
      {3, &k3, "k=0.25/0.5/0.25"},     {5, &k5, "k=binom5"},  {1, &k1, "k1"},
      {2, &k2, "even2"},               {3, &kun, "unnorm3"},  {3, &kneg, "neg"},
      {7, &kbig, "taps7_k7"},          {0, &k3, "taps0"},     {-1, &k3, "taps-1"},
      {33, &k3, "taps33_k3"},
  };
  for (auto &c : cases) {
    std::vector<float> kk(std::max(c.taps, 0), 0.f);
    for (size_t i = 0; i < kk.size() && i < c.k->size(); ++i) kk[i] = (*c.k)[i];
    float *d_k; cudaMalloc(&d_k, std::max(c.taps, 1) * 4);
    cudaMemcpy(d_k, kk.data(), kk.size() * 4, cudaMemcpyHostToDevice);
    memset(dst.data(), -1, dst.size() * 4);
    cudaMemcpy(d_dst, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice);
    NppStatus st = nppiFilterGaussAdvanced_32f_C1R_Ctx(d_src, W * 4, d_dst, W * 4, NppiSize{W, H}, c.taps, d_k, ctx());
    cudaMemcpy(dst.data(), d_dst, dst.size() * 4, cudaMemcpyDeviceToHost);
    double total = 0; for (float f : dst) total += f;
    printf("adv taps=%d %s status=%d total=%.9g\n", c.taps, c.tag, (int)st, total);
    if (st == 0) dump_f("  centerrow", dst, W, H, H / 2);
    cudaFree(d_k);
  }
  cudaFree(d_src); cudaFree(d_dst);
}

// ---------------- stage adv8 : GaussAdvanced 8u rounding/edge ----------------
static void adv8() {
  const int W = 7, H = 7, M = 2;
  const int BW = W + 2 * M, BH = H + 2 * M;
  std::vector<float> k{0.2f, 0.2f, 0.2f, 0.2f, 0.2f};
  float *d_k; cudaMalloc(&d_k, k.size() * 4); cudaMemcpy(d_k, k.data(), k.size() * 4, cudaMemcpyHostToDevice);
  for (int pass = 0; pass < 2; ++pass) {
    std::vector<unsigned char> buf((size_t)BW * BH);
    for (int y = 0; y < BH; ++y)
      for (int x = 0; x < BW; ++x) buf[(size_t)y * BW + x] = pass == 0 ? 10 : (unsigned char)(x * 13 + y * 7);
    std::vector<unsigned char> dst((size_t)W * H, 0xAA);
    unsigned char *d_buf, *d_dst;
    cudaMalloc(&d_buf, buf.size()); cudaMalloc(&d_dst, dst.size());
    cudaMemcpy(d_buf, buf.data(), buf.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, dst.data(), dst.size(), cudaMemcpyHostToDevice);
    unsigned char *p = d_buf + (size_t)M * BW + M;
    NppStatus st = nppiFilterGaussAdvanced_8u_C1R_Ctx(p, BW, d_dst, W, NppiSize{W, H}, 5, d_k, ctx());
    cudaMemcpy(dst.data(), d_dst, dst.size(), cudaMemcpyDeviceToHost);
    printf("adv8 pass=%d status=%d out[0..6]:", pass, (int)st);
    for (int x = 0; x < W; ++x) printf(" %d", dst[x]);
    printf("\n");
    cudaFree(d_buf); cudaFree(d_dst);
  }
  // taps larger than the image / kernel pointer null
  unsigned char *d_s, *d_d;
  cudaMalloc(&d_s, 64); cudaMalloc(&d_d, 64);
  NppStatus s1 = nppiFilterGaussAdvanced_8u_C1R_Ctx(d_s, 7, d_d, 7, NppiSize{7, 7}, 0, nullptr, ctx());
  NppStatus s2 = nppiFilterGaussAdvanced_8u_C1R_Ctx(nullptr, 7, d_d, 7, NppiSize{7, 7}, 3, d_k, ctx());
  NppStatus s3 = nppiFilterGaussAdvanced_8u_C1R_Ctx(d_s, 7, d_d, 7, NppiSize{7, 7}, 33, d_k, ctx());
  printf("adv8 null-kernel=%d null-src=%d taps33=%d\n", (int)s1, (int)s2, (int)s3);
  cudaFree(d_s); cudaFree(d_d); cudaFree(d_k);
}

// ---------------- stage advb : GaussAdvancedBorder ---------------------------
static void advb() {
  const int W = 5, H = 5;
  std::vector<float> k{0.25f, 0.5f, 0.25f};
  float *d_k; cudaMalloc(&d_k, 3 * 4); cudaMemcpy(d_k, k.data(), 3 * 4, cudaMemcpyHostToDevice);
  std::vector<unsigned char> src((size_t)W * H), dst((size_t)W * H, 0xAA);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) src[(size_t)y * W + x] = (unsigned char)(y * W + x);
  unsigned char *d_src, *d_dst;
  cudaMalloc(&d_src, src.size()); cudaMalloc(&d_dst, dst.size());
  cudaMemcpy(d_src, src.data(), src.size(), cudaMemcpyHostToDevice);
  for (int b = 0; b <= 4; ++b) {
    memset(dst.data(), 0xAA, dst.size());
    cudaMemcpy(d_dst, dst.data(), dst.size(), cudaMemcpyHostToDevice);
    NppStatus st = nppiFilterGaussAdvancedBorder_8u_C1R_Ctx(d_src, W, NppiSize{W, H}, NppiPoint{0, 0}, d_dst, W,
                                                            NppiSize{W, H}, 3, d_k, (NppiBorderType)b, ctx());
    cudaMemcpy(dst.data(), d_dst, dst.size(), cudaMemcpyDeviceToHost);
    printf("advb border=%d status=%d row0:", b, (int)st);
    for (int x = 0; x < W; ++x) printf(" %d", dst[x]);
    printf("\n");
  }
  // taps validation
  const int taps[] = {1, 2, 3, 5, 17, 33, 0, -1};
  for (int t : taps) {
    NppStatus st = nppiFilterGaussAdvancedBorder_8u_C1R_Ctx(d_src, W, NppiSize{W, H}, NppiPoint{0, 0}, d_dst, W,
                                                            NppiSize{W, H}, t, d_k, NPP_BORDER_REPLICATE, ctx());
    printf("advb taps=%d status=%d\n", t, (int)st);
  }
  NppStatus st = nppiFilterGaussAdvancedBorder_8u_C1R_Ctx(d_src, W, NppiSize{W, H}, NppiPoint{0, 0}, d_dst, W,
                                                          NppiSize{W, H}, 3, nullptr, NPP_BORDER_REPLICATE, ctx());
  printf("advb null-kernel status=%d\n", (int)st);
  cudaFree(d_src); cudaFree(d_dst); cudaFree(d_k);
}

// ---------------- stage wiener : WienerBorder --------------------------------
static void wiener() {
  const int W = 5, H = 5;
  std::vector<unsigned char> src((size_t)W * H), dst((size_t)W * H, 0xAA);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) src[(size_t)y * W + x] = (unsigned char)(y * W + x);
  unsigned char *d_src, *d_dst;
  cudaMalloc(&d_src, src.size()); cudaMalloc(&d_dst, dst.size());
  cudaMemcpy(d_src, src.data(), src.size(), cudaMemcpyHostToDevice);
  for (int b = 0; b <= 4; ++b) {
    float noise[4] = {1.f, 1.f, 1.f, 1.f};
    memset(dst.data(), 0xAA, dst.size());
    cudaMemcpy(d_dst, dst.data(), dst.size(), cudaMemcpyHostToDevice);
    NppStatus st = nppiFilterWienerBorder_8u_C1R_Ctx(d_src, W, NppiSize{W, H}, NppiPoint{0, 0}, d_dst, W,
                                                     NppiSize{W, H}, NppiSize{3, 3}, NppiPoint{1, 1}, noise,
                                                     (NppiBorderType)b, ctx());
    cudaMemcpy(dst.data(), d_dst, dst.size(), cudaMemcpyDeviceToHost);
    printf("wiener border=%d status=%d row0:", b, (int)st);
    for (int x = 0; x < W; ++x) printf(" %d", dst[x]);
    printf("\n");
  }
  struct MC { int mw, mh, ax, ay; float noise; const char *tag; } cases[] = {
      {3, 3, 1, 1, 0.f, "noise0"},     {3, 3, 1, 1, 1.f, "noise1"},
      {3, 3, 1, 1, 1000.f, "noise1e3"}, {3, 3, 1, 1, -1.f, "noise-1"},
      {3, 3, 0, 0, 1.f, "anchor00"},   {3, 3, 2, 2, 1.f, "anchor22"},
      {3, 3, 3, 3, 1.f, "anchor33"},   {3, 3, -1, 0, 1.f, "anchor-1"},
      {2, 2, 0, 0, 1.f, "mask2x2"},    {4, 4, 1, 1, 1.f, "mask4x4"},
      {1, 1, 0, 0, 1.f, "mask1x1"},    {5, 5, 2, 2, 1.f, "mask5x5"},
      {7, 7, 3, 3, 1.f, "mask7x7"},    {0, 0, 0, 0, 1.f, "mask0"},
  };
  for (auto &c : cases) {
    float noise[4] = {c.noise, c.noise, c.noise, c.noise};
    memset(dst.data(), 0xAA, dst.size());
    cudaMemcpy(d_dst, dst.data(), dst.size(), cudaMemcpyHostToDevice);
    NppStatus st = nppiFilterWienerBorder_8u_C1R_Ctx(d_src, W, NppiSize{W, H}, NppiPoint{0, 0}, d_dst, W,
                                                     NppiSize{W, H}, NppiSize{c.mw, c.mh}, NppiPoint{c.ax, c.ay},
                                                     noise, NPP_BORDER_REPLICATE, ctx());
    cudaMemcpy(dst.data(), d_dst, dst.size(), cudaMemcpyDeviceToHost);
    printf("wiener %s status=%d out:", c.tag, (int)st);
    for (int x = 0; x < W; ++x) printf(" %d", dst[x]);
    printf("\n");
  }
  // 32f version: exact math on a small image
  std::vector<float> fs((size_t)W * H), fd((size_t)W * H, -1.f);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) fs[(size_t)y * W + x] = (float)(y * W + x);
  float *d_fs, *d_fd;
  cudaMalloc(&d_fs, fs.size() * 4); cudaMalloc(&d_fd, fd.size() * 4);
  cudaMemcpy(d_fs, fs.data(), fs.size() * 4, cudaMemcpyHostToDevice);
  float n1[4] = {1.f, 1.f, 1.f}, n2[4] = {0.f, 0.f, 0.f}, n3[4] = {25.f, 25.f, 25.f};
  for (int c = 0; c < 3; ++c) {
    cudaMemcpy(d_fd, fd.data(), fd.size() * 4, cudaMemcpyHostToDevice);
    float *n = c == 0 ? n1 : (c == 1 ? n2 : n3);
    NppStatus st = nppiFilterWienerBorder_32f_C1R_Ctx(d_fs, W * 4, NppiSize{W, H}, NppiPoint{0, 0}, d_fd, W * 4,
                                                      NppiSize{W, H}, NppiSize{3, 3}, NppiPoint{1, 1}, n,
                                                      NPP_BORDER_REPLICATE, ctx());
    cudaMemcpy(fd.data(), d_fd, fd.size() * 4, cudaMemcpyDeviceToHost);
    printf("wiener32 n=%g status=%d row0:", *n, (int)st);
    for (int x = 0; x < W; ++x) printf(" %.9g", fd[x]);
    printf("\n");
  }
  cudaFree(d_fs); cudaFree(d_fd);
  cudaFree(d_src); cudaFree(d_dst);
}


// ---------------- stage edge2 : margin vs clamp discrimination ---------------
static unsigned char lcg(unsigned &st) { st = st * 1103515245u + 12345u; return (unsigned char)((st >> 16) & 0xFF); }

static void edge2() {
  const int W = 5, H = 5;
  struct E { NppiMaskSize mask; int r; const char *tag; } cases[] = {
      {NPP_MASK_SIZE_3_X_3, 1, "3x3"},
      {NPP_MASK_SIZE_5_X_5, 2, "5x5"},
      {NPP_MASK_SIZE_7_X_7, 3, "7x7"},
  };
  for (auto &c : cases) {
    const int taps = (c.mask == NPP_MASK_SIZE_3_X_3) ? 3 : (c.mask == NPP_MASK_SIZE_5_X_5 ? 5 : 7);
    const int BW = W + 2 * c.r, BH = H + 2 * c.r;
    std::vector<unsigned char> buf((size_t)BW * BH);
    unsigned st = 12345;
    for (auto &v : buf) v = lcg(st);
    std::vector<unsigned char> dst((size_t)W * H, 0xAA);
    unsigned char *d_buf, *d_dst;
    cudaMalloc(&d_buf, buf.size()); cudaMalloc(&d_dst, dst.size());
    cudaMemcpy(d_buf, buf.data(), buf.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, dst.data(), dst.size(), cudaMemcpyHostToDevice);
    unsigned char *p = d_buf + (size_t)c.r * BW + c.r;
    NppStatus stat = nppiFilterGauss_8u_C1R_Ctx(p, BW, d_dst, W, NppiSize{W, H}, c.mask, ctx());
    cudaMemcpy(dst.data(), d_dst, dst.size(), cudaMemcpyDeviceToHost);
    // host kernel from sigma = 0.6 + 0.4*(taps/2)
    double sigma = 0.6 + 0.4 * (taps / 2);
    std::vector<double> k(taps);
    double sum = 0;
    for (int i = 0; i < taps; ++i) {
      int d = i - c.r;
      k[i] = exp(-(double)d * d / (2.0 * sigma * sigma));
      sum += k[i];
    }
    for (auto &v : k) v /= sum;
    int matchMargin = 0, matchClamp = 0, total = 0;
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        double acc = 0, accC = 0;
        for (int j = 0; j < taps; ++j)
          for (int i = 0; i < taps; ++i) {
            int sx = x + i - c.r, sy = y + j - c.r;
            acc += k[i] * k[j] * buf[(size_t)(sy + c.r) * BW + (sx + c.r)];
            int cx = sx < 0 ? 0 : (sx >= W ? W - 1 : sx);
            int cy = sy < 0 ? 0 : (sy >= H ? H - 1 : sy);
            accC += k[i] * k[j] * buf[(size_t)(cy + c.r) * BW + (cx + c.r)];
          }
        unsigned char got = dst[(size_t)y * W + x];
        unsigned char em = (unsigned char)floor(acc + 0.5), ec = (unsigned char)floor(accC + 0.5);
        total++;
        if (got == em) matchMargin++;
        if (got == ec) matchClamp++;
      }
    printf("gauss8 %s(%d) status=%d margin=%d/%d clamp=%d/%d dst00=%d\n", c.tag, taps, (int)stat, matchMargin, total,
           matchClamp, total, dst[0]);
    cudaFree(d_buf); cudaFree(d_dst);
  }
}

// ---------------- stage wiener2 : wiener math fit --------------------------
static void wiener2() {
  const int W = 7, H = 7;
  struct Case { int mw, mh, ax, ay; float n; const char *tag; };
  // pseudo random 7x7 image
  std::vector<float> src((size_t)W * H);
  unsigned rng = 12345;
  for (auto &v : src) { rng = rng * 1103515245u + 12345u; v = (float)((rng >> 16) % 21); }
  float *d_src, *d_dst;
  cudaMalloc(&d_src, src.size() * 4); cudaMalloc(&d_dst, src.size() * 4);
  cudaMemcpy(d_src, src.data(), src.size() * 4, cudaMemcpyHostToDevice);
  printf("src:\n");
  for (int y = 0; y < H; ++y) { for (int x = 0; x < W; ++x) printf(" %.6g", src[(size_t)y*W+x]); printf("\n"); }
  const Case cases[] = {
      {3, 3, 1, 1, 0.f, "m33-a11-n0"},   {3, 3, 1, 1, 0.25f, "m33-a11-n025"},
      {3, 3, 1, 1, 0.5f, "m33-a11-n05"}, {3, 3, 1, 1, 1.f, "m33-a11-n1"},
      {3, 3, 1, 1, 2.f, "m33-a11-n2"},   {5, 5, 2, 2, 0.f, "m55-a22-n0"},
      {5, 5, 2, 2, 1.f, "m55-a22-n1"},   {1, 1, 0, 0, 0.f, "m11-n0"},
      {1, 1, 0, 0, 1.f, "m11-n1"},       {3, 1, 1, 0, 0.f, "m31-n0"},
      {3, 1, 1, 0, 1.f, "m31-n1"},       {1, 3, 0, 1, 0.f, "m13-n0"},
      {3, 3, 0, 0, 0.f, "m33-a00-n0"},   {3, 3, 2, 2, 0.f, "m33-a22-n0"},
      {3, 3, 0, 2, 0.f, "m33-a02-n0"},   {3, 3, 2, 0, 0.f, "m33-a20-n0"},
  };
  for (auto &c : cases) {
    std::vector<float> dst((size_t)W * H, -1.f);
    cudaMemcpy(d_dst, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice);
    float noise[4] = {c.n, c.n, c.n, c.n};
    NppStatus st = nppiFilterWienerBorder_32f_C1R_Ctx(d_src, W * 4, NppiSize{W, H}, NppiPoint{0, 0}, d_dst, W * 4,
                                                      NppiSize{W, H}, NppiSize{c.mw, c.mh}, NppiPoint{c.ax, c.ay},
                                                      noise, NPP_BORDER_REPLICATE, ctx());
    cudaMemcpy(dst.data(), d_dst, dst.size() * 4, cudaMemcpyDeviceToHost);
    printf("== %s status=%d\n", c.tag, (int)st);
    for (int y = 0; y < H; ++y) { for (int x = 0; x < W; ++x) printf(" %.8g", dst[(size_t)y*W+x]); printf("\n"); }
  }
  // constant image: variance == 0 paths
  std::vector<float> cs((size_t)W * H, 7.5f);
  cudaMemcpy(d_src, cs.data(), cs.size() * 4, cudaMemcpyHostToDevice);
  for (float n : {0.f, 0.5f, 1.f}) {
    std::vector<float> dst((size_t)W * H, -1.f);
    cudaMemcpy(d_dst, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice);
    float noise[4] = {n, n, n, n};
    NppStatus st = nppiFilterWienerBorder_32f_C1R_Ctx(d_src, W * 4, NppiSize{W, H}, NppiPoint{0, 0}, d_dst, W * 4,
                                                      NppiSize{W, H}, NppiSize{3, 3}, NppiPoint{1, 1}, noise,
                                                      NPP_BORDER_REPLICATE, ctx());
    cudaMemcpy(dst.data(), d_dst, dst.size() * 4, cudaMemcpyDeviceToHost);
    printf("== const n=%g status=%d center=%g corner=%g\n", n, (int)st, dst[3 * W + 3], dst[0]);
  }
  cudaFree(d_src); cudaFree(d_dst);
}

// ---------------- stage round : rounding rule via GaussAdvanced --------------
static void round_stage() {
  const int W = 4, H = 3, M = 2, R = 2;   // R = row of the ROI inside the buffer
  const int BW = W + 2 * M, BH = H + 2 * R;
  // kernel [0.5, 0, 0.5] -> result (left + right) / 2 exactly on the ROI row
  std::vector<float> k{0.5f, 0.f, 0.5f};
  float *d_k; cudaMalloc(&d_k, 3 * 4); cudaMemcpy(d_k, k.data(), 3 * 4, cudaMemcpyHostToDevice);
  std::vector<float> k4{0.25f, 0.25f, 0.25f, 0.25f};
  float *d_k4; cudaMalloc(&d_k4, 4 * 4); cudaMemcpy(d_k4, k4.data(), 4 * 4, cudaMemcpyHostToDevice);
  unsigned char *d_buf, *d_dst;
  cudaMalloc(&d_buf, (size_t)BW * BH); cudaMalloc(&d_dst, (size_t)W * H);
  {
    std::vector<unsigned char> buf((size_t)BW * BH);
    for (int y = 0; y < BH; ++y)
      for (int x = 0; x < BW; ++x) buf[(size_t)y * BW + x] = (unsigned char)(1 + x * x + 2 * x);
    cudaMemcpy(d_buf, buf.data(), buf.size(), cudaMemcpyHostToDevice);
    NppStatus st = nppiFilterGaussAdvanced_8u_C1R_Ctx(d_buf + (size_t)R * BW + M, BW, d_dst, W, NppiSize{W, H},
                                                      3, d_k, ctx());
    std::vector<unsigned char> o8((size_t)W * H, 0xEE);
    cudaMemcpy(o8.data(), d_dst, o8.size(), cudaMemcpyDeviceToHost);
    printf("adv8 round status=%d :", (int)st);
    for (int i = 0; i < W; ++i)
      printf(" %d(=%.1f)", o8[(size_t)(H / 2) * W + i],
             (buf[(size_t)R * BW + M + i - 1] + buf[(size_t)R * BW + M + i + 1]) / 2.0);
    printf("  dstrow0=%d\n", o8[0]);
  }
  {
    std::vector<short> buf((size_t)BW * BH);
    for (int y = 0; y < BH; ++y)
      for (int x = 0; x < BW; ++x) buf[(size_t)y * BW + x] = (short)(-1 - x * x - 2 * x);
    short *d_s, *d_sd;
    cudaMalloc(&d_s, buf.size() * 2); cudaMalloc(&d_sd, (size_t)W * H * 2);
    cudaMemcpy(d_s, buf.data(), buf.size() * 2, cudaMemcpyHostToDevice);
    NppStatus st = nppiFilterGaussAdvanced_16s_C1R_Ctx(d_s + (size_t)R * BW + M, BW * 2, d_sd, W * 2,
                                                       NppiSize{W, H}, 3, d_k, ctx());
    std::vector<short> o16((size_t)W * H, -9999);
    cudaMemcpy(o16.data(), d_sd, o16.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(d_s); cudaFree(d_sd);
    printf("adv16s round status=%d :", (int)st);
    for (int i = 0; i < W; ++i)
      printf(" %d(=%.1f)", o16[(size_t)(H / 2) * W + i],
             (buf[(size_t)R * BW + M + i - 1] + buf[(size_t)R * BW + M + i + 1]) / 2.0);
    printf("\n");
  }
  {
    std::vector<unsigned short> buf((size_t)BW * BH);
    for (int y = 0; y < BH; ++y)
      for (int x = 0; x < BW; ++x) buf[(size_t)y * BW + x] = (unsigned short)(1 + x * x + 2 * x);
    unsigned short *d_u, *d_ud;
    cudaMalloc(&d_u, buf.size() * 2); cudaMalloc(&d_ud, (size_t)W * H * 2);
    cudaMemcpy(d_u, buf.data(), buf.size() * 2, cudaMemcpyHostToDevice);
    // taps must be odd: use 5 taps [0.25,0.25,0,0.25,0.25] -> mean of +/-2
    std::vector<float> k5{0.25f, 0.25f, 0.f, 0.25f, 0.25f};
    float *d_k5; cudaMalloc(&d_k5, 5 * 4); cudaMemcpy(d_k5, k5.data(), 5 * 4, cudaMemcpyHostToDevice);
    NppStatus st = nppiFilterGaussAdvanced_16u_C1R_Ctx(d_u + (size_t)R * BW + M, BW * 2, d_ud, W * 2,
                                                       NppiSize{W, H}, 5, d_k5, ctx());
    std::vector<unsigned short> o16u((size_t)W * H, 9999);
    cudaMemcpy(o16u.data(), d_ud, o16u.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(d_u); cudaFree(d_ud);
    printf("adv16u taps5 status=%d :", (int)st);
    for (int i = 0; i < W; ++i)
      printf(" %d(=%.2f)", o16u[(size_t)(H / 2) * W + i],
             (buf[(size_t)R * BW + M + i - 2] + buf[(size_t)R * BW + M + i + 2]) / 2.0);
    printf("\n");
    cudaFree(d_k5);
  }
  {
    std::vector<float> buf((size_t)BW * BH);
    for (int y = 0; y < BH; ++y)
      for (int x = 0; x < BW; ++x) buf[(size_t)y * BW + x] = (float)(1 + x * x + 2 * x);
    float *d_f, *d_fd;
    cudaMalloc(&d_f, buf.size() * 4); cudaMalloc(&d_fd, (size_t)W * H * 4);
    cudaMemcpy(d_f, buf.data(), buf.size() * 4, cudaMemcpyHostToDevice);
    NppStatus st = nppiFilterGaussAdvanced_32f_C1R_Ctx(d_f + (size_t)R * BW + M, BW * 4, d_fd,
                                                       W * 4, NppiSize{W, H}, 3, d_k, ctx());
    std::vector<float> o32((size_t)W * H, -9999.f);
    cudaMemcpy(o32.data(), d_fd, o32.size() * 4, cudaMemcpyDeviceToHost);
    cudaFree(d_f); cudaFree(d_fd);
    printf("adv32 status=%d :", (int)st);
    for (int i = 0; i < W; ++i) printf(" %.9g", o32[(size_t)(H / 2) * W + i]);
    printf("\n");
  }
  // y orientation of the (asymmetric) kernel: delta at the ROI centre of a
  // 7x7 ROI with kernel k=[0.1..0.7] shows which k index maps to dy=-1.
  {
    const int W7 = 9, H7 = 9;
    std::vector<float> buf((size_t)W7 * H7, 0.f);
    buf[(size_t)(H7 / 2) * W7 + W7 / 2] = 1.f;
    float *d_b, *d_d;
    cudaMalloc(&d_b, buf.size() * 4); cudaMalloc(&d_d, buf.size() * 4);
    cudaMemcpy(d_b, buf.data(), buf.size() * 4, cudaMemcpyHostToDevice);
    std::vector<float> k7{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f};
    float *d_k7; cudaMalloc(&d_k7, 7 * 4); cudaMemcpy(d_k7, k7.data(), 7 * 4, cudaMemcpyHostToDevice);
    NppStatus st = nppiFilterGaussAdvanced_32f_C1R_Ctx(d_b, W7 * 4, d_d, W7 * 4, NppiSize{W7, H7}, 7, d_k7, ctx());
    std::vector<float> o((size_t)W7 * H7, -1.f);
    cudaMemcpy(o.data(), d_d, o.size() * 4, cudaMemcpyDeviceToHost);
    printf("adv32 k=[0.1..0.7] delta status=%d\n  col(x=4):", (int)st);
    for (int y = 1; y < 8; ++y) printf(" %.9g", o[(size_t)y * W7 + 4]);
    printf("\n  row(y=4):");
    for (int x = 1; x < 8; ++x) printf(" %.9g", o[(size_t)4 * W7 + x]);
    printf("\n");
    cudaFree(d_b); cudaFree(d_d); cudaFree(d_k7);
  }
  cudaFree(d_k); cudaFree(d_k4); cudaFree(d_buf); cudaFree(d_dst);
}


// ---------------- stage edge3 : detailed margin-vs-clamp dump ----------------
static void edge3() {
  const int W = 5, H = 5;
  struct E { NppiMaskSize mask; int taps; int r; const char *tag; } cases[] = {
      {NPP_MASK_SIZE_3_X_3, 3, 1, "3x3"}, {NPP_MASK_SIZE_5_X_5, 5, 2, "5x5"},
  };
  for (auto &c : cases) {
    const int BW = W + 2 * c.r, BH = H + 2 * c.r;
    std::vector<float> buf((size_t)BW * BH);
    unsigned st = 777;
    for (auto &v : buf) v = lcg(st) / 255.0f;
    std::vector<float> dst((size_t)W * H, -1.f);
    float *d_buf, *d_dst;
    cudaMalloc(&d_buf, buf.size() * 4); cudaMalloc(&d_dst, dst.size() * 4);
    cudaMemcpy(d_buf, buf.data(), buf.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice);
    float *p = d_buf + (size_t)c.r * BW + c.r;
    NppStatus stat = nppiFilterGauss_32f_C1R_Ctx(p, BW * 4, d_dst, W * 4, NppiSize{W, H}, c.mask, ctx());
    cudaMemcpy(dst.data(), d_dst, dst.size() * 4, cudaMemcpyDeviceToHost);
    double sigma = 0.6 + 0.4 * (c.taps / 2);
    std::vector<double> k(c.taps);
    double sum = 0;
    for (int i = 0; i < c.taps; ++i) {
      int d = i - c.r;
      k[i] = exp(-(double)d * d / (2.0 * sigma * sigma));
      sum += k[i];
    }
    for (auto &v : k) v /= sum;
    int mm = 0, mc = 0;
    printf("edge3 %s status=%d\n", c.tag, (int)stat);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        double acc = 0, accC = 0;
        for (int j = 0; j < c.taps; ++j)
          for (int i = 0; i < c.taps; ++i) {
            int sx = x + i - c.r, sy = y + j - c.r;
            acc += k[i] * k[j] * buf[(size_t)(sy + c.r) * BW + (sx + c.r)];
            int cx = sx < 0 ? 0 : (sx >= W ? W - 1 : sx);
            int cy = sy < 0 ? 0 : (sy >= H ? H - 1 : sy);
            accC += k[i] * k[j] * buf[(size_t)(cy + c.r) * BW + (cx + c.r)];
          }
        float got = dst[(size_t)y * W + x];
        if (fabs(got - acc) < 1e-6) mm++;
        if (fabs(got - accC) < 1e-6) mc++;
        if ((y * W + x) < 6 || (y * W + x) == 12)
          printf("  (%d,%d) got=%.9g margin=%.9g clamp=%.9g\n", x, y, got, acc, accC);
      }
    printf("  margin=%d clamp=%d (of %d)\n", mm, mc, W * H);
    cudaFree(d_buf); cudaFree(d_dst);
  }
}

// ---------------- stage wiener3 : isolate horizontal / vertical structure -------
static void wiener_run_1d(const float *h_src, int W, int H, int mw, int mh, int ax, int ay, float n,
                          const char *tag) {
  float *d_s, *d_d;
  cudaMalloc(&d_s, (size_t)W * H * 4); cudaMalloc(&d_d, (size_t)W * H * 4);
  cudaMemcpy(d_s, h_src, (size_t)W * H * 4, cudaMemcpyHostToDevice);
  std::vector<float> dst((size_t)W * H, -1.f);
  cudaMemcpy(d_d, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice);
  float noise[4] = {n, n, n, n};
  NppStatus st = nppiFilterWienerBorder_32f_C1R_Ctx(d_s, W * 4, NppiSize{W, H}, NppiPoint{0, 0}, d_d,
                                                    W * 4, NppiSize{W, H}, NppiSize{mw, mh},
                                                    NppiPoint{ax, ay}, noise, NPP_BORDER_REPLICATE, ctx());
  cudaMemcpy(dst.data(), d_d, dst.size() * 4, cudaMemcpyDeviceToHost);
  printf("== %s status=%d\n", tag, (int)st);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) printf(" %.9g", dst[(size_t)y * W + x]);
    printf("\n");
  }
  cudaFree(d_s); cudaFree(d_d);
}

static void wiener3() {
  const int V0[7] = {3, 17, 5, 9, 0, 12, 8};
  { // H=1 : single row, 7 columns
    std::vector<float> src(7);
    for (int x = 0; x < 7; ++x) src[x] = (float)V0[x];
    wiener_run_1d(src.data(), 7, 1, 3, 3, 1, 1, 1.f, "h1 m33 a11 n1");
    wiener_run_1d(src.data(), 7, 1, 3, 3, 1, 1, 0.f, "h1 m33 a11 n0");
    wiener_run_1d(src.data(), 7, 1, 5, 5, 2, 2, 1.f, "h1 m55 a22 n1");
    wiener_run_1d(src.data(), 7, 1, 3, 1, 1, 0, 1.f, "h1 m31 a10 n1");
    wiener_run_1d(src.data(), 7, 1, 3, 3, 0, 0, 1.f, "h1 m33 a00 n1");
    wiener_run_1d(src.data(), 7, 1, 3, 3, 2, 2, 1.f, "h1 m33 a22 n1");
    wiener_run_1d(src.data(), 7, 1, 3, 3, 1, 1, 2.f, "h1 m33 a11 n2");
  }
  { // W=1 : single column, 7 rows
    std::vector<float> src(7);
    for (int y = 0; y < 7; ++y) src[y] = (float)V0[y];
    wiener_run_1d(src.data(), 1, 7, 3, 3, 1, 1, 1.f, "w1 m33 a11 n1");
    wiener_run_1d(src.data(), 1, 7, 3, 3, 1, 1, 0.f, "w1 m33 a11 n0");
    wiener_run_1d(src.data(), 1, 7, 1, 3, 0, 1, 1.f, "w1 m13 a01 n1");
    wiener_run_1d(src.data(), 1, 7, 1, 3, 0, 1, 0.f, "w1 m13 a01 n0");
    wiener_run_1d(src.data(), 1, 7, 5, 5, 2, 2, 1.f, "w1 m55 a22 n1");
    wiener_run_1d(src.data(), 1, 7, 3, 3, 0, 0, 1.f, "w1 m33 a00 n1");
    wiener_run_1d(src.data(), 1, 7, 3, 3, 2, 2, 1.f, "w1 m33 a22 n1");
    wiener_run_1d(src.data(), 1, 7, 3, 3, 0, 2, 1.f, "w1 m33 a02 n1");
    wiener_run_1d(src.data(), 1, 7, 3, 3, 2, 0, 1.f, "w1 m33 a20 n1");
    wiener_run_1d(src.data(), 1, 7, 3, 1, 1, 0, 1.f, "w1 m31 a10 n1");
    wiener_run_1d(src.data(), 1, 7, 3, 3, 1, 1, 2.f, "w1 m33 a11 n2");
  }
  { // 7x7 random image, anchor variants with n=1 (verify convention on the n>0 path)
    const int W = 7, H = 7;
    std::vector<float> src((size_t)W * H);
    unsigned rng = 12345;
    for (auto &v : src) { rng = rng * 1103515245u + 12345u; v = (float)((rng >> 16) % 21); }
    float *d_s, *d_d;
    cudaMalloc(&d_s, src.size() * 4); cudaMalloc(&d_d, src.size() * 4);
    cudaMemcpy(d_s, src.data(), src.size() * 4, cudaMemcpyHostToDevice);
    struct AC { int ax, ay; const char *tag; } acs[] = {{0,0,"a00"},{1,1,"a11"},{2,2,"a22"},{0,2,"a02"},{2,0,"a20"}};
    for (auto &a : acs) {
      std::vector<float> dst((size_t)W * H, -1.f);
      cudaMemcpy(d_d, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice);
      float noise[4] = {1.f, 1.f, 1.f, 1.f};
      NppStatus st = nppiFilterWienerBorder_32f_C1R_Ctx(d_s, W * 4, NppiSize{W, H}, NppiPoint{0, 0}, d_d,
                                                        W * 4, NppiSize{W, H}, NppiSize{3, 3},
                                                        NppiPoint{a.ax, a.ay}, noise, NPP_BORDER_REPLICATE, ctx());
      cudaMemcpy(dst.data(), d_d, dst.size() * 4, cudaMemcpyDeviceToHost);
      printf("== r7 %s-n1 status=%d\n", a.tag, (int)st);
      for (int y = 0; y < H; ++y) { for (int x = 0; x < W; ++x) printf(" %.9g", dst[(size_t)y*W+x]); printf("\n"); }
    }
    // determinism: same case 3x, different dst preinit
    for (int rep = 0; rep < 3; ++rep) {
      std::vector<float> dst((size_t)W * H, rep == 2 ? 999.f : -1.f);
      cudaMemcpy(d_d, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice);
      float noise[4] = {1.f, 1.f, 1.f, 1.f};
      nppiFilterWienerBorder_32f_C1R_Ctx(d_s, W * 4, NppiSize{W, H}, NppiPoint{0, 0}, d_d, W * 4,
                                         NppiSize{W, H}, NppiSize{3, 3}, NppiPoint{1, 1}, noise,
                                         NPP_BORDER_REPLICATE, ctx());
      cudaMemcpy(dst.data(), d_d, dst.size() * 4, cudaMemcpyDeviceToHost);
      printf("== r7 det rep=%d :", rep);
      for (int i = 0; i < W; ++i) printf(" %.9g", dst[(size_t)3 * W + i]);
      printf("\n");
    }
    cudaFree(d_s); cudaFree(d_d);
  }
}

int main(int argc, char **argv) {
  const char *stage = argc > 1 ? argv[1] : "all";
  if (!strcmp(stage, "gauss32") || !strcmp(stage, "all")) { printf("== gauss32 ==\n"); gauss32(); }
  if (!strcmp(stage, "gauss8") || !strcmp(stage, "all")) { printf("== gauss8 ==\n"); gauss8(); }
  if (!strcmp(stage, "edge") || !strcmp(stage, "all")) { printf("== edge ==\n"); edge(); }
  if (!strcmp(stage, "gborder") || !strcmp(stage, "all")) { printf("== gborder ==\n"); gborder(); }
  if (!strcmp(stage, "adv") || !strcmp(stage, "all")) { printf("== adv ==\n"); adv(); }
  if (!strcmp(stage, "adv8") || !strcmp(stage, "all")) { printf("== adv8 ==\n"); adv8(); }
  if (!strcmp(stage, "advb") || !strcmp(stage, "all")) { printf("== advb ==\n"); advb(); }
  if (!strcmp(stage, "wiener") || !strcmp(stage, "all")) { printf("== wiener ==\n"); wiener(); }
  if (!strcmp(stage, "wiener2")) { printf("== wiener2 ==\n"); wiener2(); }
  if (!strcmp(stage, "wiener3")) { printf("== wiener3 ==\n"); wiener3(); }
  if (!strcmp(stage, "edge2") || !strcmp(stage, "all")) { printf("== edge2 ==\n"); edge2(); }
  if (!strcmp(stage, "round") || !strcmp(stage, "all")) { printf("== round ==\n"); round_stage(); }
  if (!strcmp(stage, "edge3") || !strcmp(stage, "all")) { printf("== edge3 ==\n"); edge3(); }
  printf("done\n");
  return 0;
}
