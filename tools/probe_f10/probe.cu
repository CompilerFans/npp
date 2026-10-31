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

// ---------------- stage wiener4 : formula disambiguation --------------------
static void w4_status(const char *tag, const float *h_src, int W, int H, int mw, int mh, int ax, int ay,
                      float n, NppiBorderType b) {
  float *d_s, *d_d;
  cudaMalloc(&d_s, (size_t)W * H * 4);
  cudaMalloc(&d_d, (size_t)W * H * 4);
  cudaMemcpy(d_s, h_src, (size_t)W * H * 4, cudaMemcpyHostToDevice);
  float noise[4] = {n, n, n, n};
  NppStatus st = nppiFilterWienerBorder_32f_C1R_Ctx(d_s, W * 4, NppiSize{W, H}, NppiPoint{0, 0}, d_d,
                                                    W * 4, NppiSize{W, H}, NppiSize{mw, mh},
                                                    NppiPoint{ax, ay}, noise, b, ctx());
  float o0 = -12345.f;
  cudaMemcpy(&o0, d_d, 4, cudaMemcpyDeviceToHost);
  printf("w4 %s status=%d out00=%.9g\n", tag, (int)st, o0);
  cudaFree(d_s);
  cudaFree(d_d);
}

static void wiener4() {
  const int W = 7, H = 7;
  std::vector<float> src((size_t)W * H);
  unsigned rng = 12345;
  for (auto &v : src) { rng = rng * 1103515245u + 12345u; v = (float)((rng >> 16) % 21); }
  std::vector<float> cst((size_t)W * H, 7.5f);
  // A. noise range sweep (random image has window variances down to ~3.5)
  for (float n : {0.f, 0.25f, 0.5f, 0.75f, 1.f, 1.25f, 1.5f, 1.75f, 2.f, 3.f, 4.f, 5.f, 100.f, 1000.f, -0.5f, -1.f})
    w4_status("rand m33 n-sweep", src.data(), W, H, 3, 3, 1, 1, n, NPP_BORDER_REPLICATE);
  for (float n : {0.f, 0.5f, 1.f, 2.f, 4.f, 1000.f, -1.f})
    w4_status("const m33 n-sweep", cst.data(), W, H, 3, 3, 1, 1, n, NPP_BORDER_REPLICATE);
  for (float n : {0.f, 0.5f, 1.f, 2.f, 4.f})
    w4_status("rand m11 n-sweep", src.data(), W, H, 1, 1, 0, 0, n, NPP_BORDER_REPLICATE);
  for (float n : {0.f, 0.5f, 1.f, 2.f, 4.f})
    w4_status("rand m55 n-sweep", src.data(), W, H, 5, 5, 2, 2, n, NPP_BORDER_REPLICATE);
  // B. 2D mask-shape sweep on the same random image (n=1 and n=0.5)
  struct MS { int mw, mh, ax, ay; } mss[] = {{3, 3, 1, 1}, {5, 5, 2, 2}, {3, 5, 1, 2}, {5, 3, 2, 1},
                                             {3, 7, 1, 3}, {7, 3, 3, 1}, {2, 2, 0, 0}, {4, 4, 1, 1},
                                             {6, 6, 2, 2}, {2, 4, 0, 1}};
  for (auto &m : mss) {
    char tag[64];
    snprintf(tag, sizeof tag, "r7 m%dx%d a%d%d n1", m.mw, m.mh, m.ax, m.ay);
    wiener_run_1d(src.data(), W, H, m.mw, m.mh, m.ax, m.ay, 1.f, tag);
    snprintf(tag, sizeof tag, "r7 m%dx%d a%d%d n05", m.mw, m.mh, m.ax, m.ay);
    wiener_run_1d(src.data(), W, H, m.mw, m.mh, m.ax, m.ay, 0.5f, tag);
  }
  // C. 1D anchor sweep both orientations (asymmetry check)
  {
    std::vector<float> col(7), row(7);
    const float V0[7] = {3, 17, 5, 9, 0, 12, 8};
    for (int i = 0; i < 7; ++i) { col[i] = V0[i]; row[i] = V0[i]; }
    for (int a = 0; a < 3; ++a) {
      char tag[64];
      snprintf(tag, sizeof tag, "w1 m13 a0%d n1", a);
      wiener_run_1d(col.data(), 1, 7, 1, 3, 0, a, 1.f, tag);
      snprintf(tag, sizeof tag, "h1 m31 a%d0 n1", a);
      wiener_run_1d(row.data(), 7, 1, 3, 1, a, 0, 1.f, tag);
    }
  }
  // D. structured 1D patterns (n = 1, 0.5, 0.25, 0)
  struct Pat { const float v[7]; const char *tag; } pats[] = {
      {{0, 1, 2, 3, 4, 5, 6}, "ramp"},       {{0, 0, 0, 1, 0, 0, 0}, "delta"},
      {{5, 5, 6, 5, 5, 5, 5}, "nearconst"},  {{5, 5, 5, 5, 5, 5, 5}, "const5"},
      {{1, 2, 3, 4, 5, 6, 7}, "ramp1"},      {{0, 0, 1, 0, 0, 0, 0}, "delta0"},
      {{10, 0, 10, 0, 10, 0, 10}, "alt"},    {{2, 2, 2, 8, 2, 2, 2}, "step"},
  };
  for (auto &p : pats) {
    std::vector<float> col(p.v, p.v + 7), row(p.v, p.v + 7);
    for (float n : {1.f, 0.5f, 0.f}) {
      char tag[64];
      snprintf(tag, sizeof tag, "w1 %s m13 a01 n%g", p.tag, n);
      wiener_run_1d(col.data(), 1, 7, 1, 3, 0, 1, n, tag);
      snprintf(tag, sizeof tag, "h1 %s m31 a10 n%g", p.tag, n);
      wiener_run_1d(row.data(), 7, 1, 3, 1, 1, 0, n, tag);
      snprintf(tag, sizeof tag, "h1w %s m33 a11 n%g", p.tag, n);
      wiener_run_1d(row.data(), 7, 1, 3, 3, 1, 1, n, tag);
    }
  }
  // E. border types (32f, replicate image identity row0/col0)
  {
    float *d_s, *d_d;
    cudaMalloc(&d_s, (size_t)W * H * 4);
    cudaMalloc(&d_d, (size_t)W * H * 4);
    cudaMemcpy(d_s, src.data(), (size_t)W * H * 4, cudaMemcpyHostToDevice);
    const int bs[] = {0, 1, 2, 3, 4, 5, 6, 100, -1};
    for (int b : bs) {
      std::vector<float> dst((size_t)W * H, -1.f);
      cudaMemcpy(d_d, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice);
      float noise[4] = {1.f, 1.f, 1.f, 1.f};
      NppStatus st = nppiFilterWienerBorder_32f_C1R_Ctx(d_s, W * 4, NppiSize{W, H}, NppiPoint{0, 0}, d_d,
                                                        W * 4, NppiSize{W, H}, NppiSize{3, 3}, NppiPoint{1, 1},
                                                        noise, (NppiBorderType)b, ctx());
      cudaMemcpy(dst.data(), d_d, dst.size() * 4, cudaMemcpyDeviceToHost);
      printf("w4 border=%d status=%d\n  row0:", b, (int)st);
      for (int x = 0; x < W; ++x) printf(" %.9g", dst[x]);
      printf("\n  col0:");
      for (int y = 0; y < H; ++y) printf(" %.9g", dst[(size_t)y * W]);
      printf("\n");
    }
    // anchor / mask validation on the same buffers
    struct AV { int mw, mh, ax, ay; const char *tag; } avs[] = {
        {3, 3, 1, 1, "ok"},      {3, 3, 3, 3, "a33"},   {3, 3, -1, 0, "a-1"},
        {3, 3, 0, 0, "a00"},     {3, 3, 2, 2, "a22"},   {3, 3, 5, 0, "a50"},
        {0, 0, 0, 0, "m00"},     {-3, -3, 0, 0, "m-3"}, {2, 2, 0, 0, "m22"},
        {4, 4, 1, 1, "m44"},     {8, 8, 3, 3, "m88"},   {17, 17, 8, 8, "m1717"},
        {1, 1, 0, 0, "m11"},     {3, 2, 1, 0, "m32"},   {2, 3, 0, 1, "m23"},
        {100, 100, 50, 50, "m100"}, {7, 7, 3, 3, "m77"},
    };
    for (auto &a : avs) {
      float noise[4] = {1.f, 1.f, 1.f, 1.f};
      NppStatus st = nppiFilterWienerBorder_32f_C1R_Ctx(d_s, W * 4, NppiSize{W, H}, NppiPoint{0, 0}, d_d,
                                                        W * 4, NppiSize{W, H}, NppiSize{a.mw, a.mh},
                                                        NppiPoint{a.ax, a.ay}, noise, NPP_BORDER_REPLICATE, ctx());
      printf("w4 anchor/mask %s status=%d\n", a.tag, (int)st);
    }
    cudaFree(d_s);
    cudaFree(d_d);
  }
  // F. 8u / 16s rounding + saturation
  {
    const int W8 = 9, H8 = 9;
    std::vector<unsigned char> s8((size_t)W8 * H8), d8((size_t)W8 * H8, 0xEE);
    for (int i = 0; i < W8 * H8; ++i) s8[i] = (unsigned char)((i * 37 + 11) & 0xFF);
    std::vector<short> s16((size_t)W8 * H8), d16((size_t)W8 * H8, -12345);
    for (int i = 0; i < W8 * H8; ++i) s16[i] = (short)(((i * 137 + 51) % 600) - 300);
    unsigned char *d_s8; short *d_s16;
    unsigned char *d_d8; short *d_d16;
    cudaMalloc(&d_s8, s8.size()); cudaMalloc(&d_d8, d8.size());
    cudaMalloc(&d_s16, s16.size() * 2); cudaMalloc(&d_d16, d16.size() * 2);
    cudaMemcpy(d_s8, s8.data(), s8.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s16, s16.data(), s16.size() * 2, cudaMemcpyHostToDevice);
    for (float n : {1.f, 0.5f, 0.25f}) {
      float noise[4] = {n, n, n, n};
      cudaMemcpy(d_d8, d8.data(), d8.size(), cudaMemcpyHostToDevice);
      NppStatus st = nppiFilterWienerBorder_8u_C1R_Ctx(d_s8, W8, NppiSize{W8, H8}, NppiPoint{0, 0}, d_d8, W8,
                                                       NppiSize{W8, H8}, NppiSize{3, 3}, NppiPoint{1, 1}, noise,
                                                       NPP_BORDER_REPLICATE, ctx());
      cudaMemcpy(d8.data(), d_d8, d8.size(), cudaMemcpyDeviceToHost);
      printf("w4 8u n=%g status=%d\n", n, (int)st);
      for (int y = 0; y < 3; ++y) {
        printf("  row%d:", y);
        for (int x = 0; x < W8; ++x) printf(" %3d", d8[(size_t)y * W8 + x]);
        printf("\n");
      }
      cudaMemcpy(d_d16, d16.data(), d16.size() * 2, cudaMemcpyHostToDevice);
      st = nppiFilterWienerBorder_16s_C1R_Ctx(d_s16, W8 * 2, NppiSize{W8, H8}, NppiPoint{0, 0}, d_d16, W8 * 2,
                                              NppiSize{W8, H8}, NppiSize{3, 3}, NppiPoint{1, 1}, noise,
                                              NPP_BORDER_REPLICATE, ctx());
      cudaMemcpy(d16.data(), d_d16, d16.size() * 2, cudaMemcpyDeviceToHost);
      printf("w4 16s n=%g status=%d\n", n, (int)st);
      for (int y = 0; y < 3; ++y) {
        printf("  row%d:", y);
        for (int x = 0; x < W8; ++x) printf(" %6d", d16[(size_t)y * W8 + x]);
        printf("\n");
      }
    }
    cudaFree(d_s8); cudaFree(d_d8); cudaFree(d_s16); cudaFree(d_d16);
  }
}


// ---------------- stage wiener5 : 2D impulse response, V>n regime -------------
static void w5(const char *tag, const std::vector<float> &src, int W, int H,
               int mw, int mh, int ax, int ay, float n) {
  float *d_s, *d_d;
  cudaMalloc(&d_s, (size_t)W * H * 4); cudaMalloc(&d_d, (size_t)W * H * 4);
  cudaMemcpy(d_s, src.data(), (size_t)W * H * 4, cudaMemcpyHostToDevice);
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

static void wiener5() {
  // A. 9x9 impulse (30 at center), m33/m55 : clean 2D impulse response, V >> 1
  {
    const int W = 9, H = 9;
    std::vector<float> src((size_t)W * H, 0.f);
    src[(size_t)4 * W + 4] = 30.f;
    for (float n : {1.f, 0.5f, 0.25f, 2.f})
      for (auto &m : {NppiSize{3, 3}, NppiSize{5, 5}}) {
        char tag[64];
        snprintf(tag, sizeof tag, "imp %dx%d n%g", m.width, m.height, n);
        w5(tag, src, W, H, m.width, m.height, m.width / 2, m.height / 2, n);
      }
  }
  // B. noise sweep across V=88.9 boundary at the impulse center (m33)
  {
    const int W = 9, H = 9;
    std::vector<float> src((size_t)W * H, 0.f);
    src[(size_t)4 * W + 4] = 30.f;
    float *d_s, *d_d;
    cudaMalloc(&d_s, src.size() * 4); cudaMalloc(&d_d, src.size() * 4);
    cudaMemcpy(d_s, src.data(), src.size() * 4, cudaMemcpyHostToDevice);
    for (float n : {0.f, 0.25f, 0.5f, 1.f, 2.f, 4.f, 8.f, 16.f, 32.f, 64.f, 80.f, 88.f, 88.9f, 90.f, 100.f, 200.f, 1000.f}) {
      std::vector<float> dst(src.size(), -1.f);
      cudaMemcpy(d_d, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice);
      float noise[4] = {n, n, n, n};
      NppStatus st = nppiFilterWienerBorder_32f_C1R_Ctx(d_s, W * 4, NppiSize{W, H}, NppiPoint{0, 0}, d_d,
                                                        W * 4, NppiSize{W, H}, NppiSize{3, 3},
                                                        NppiPoint{1, 1}, noise, NPP_BORDER_REPLICATE, ctx());
      cudaMemcpy(dst.data(), d_d, dst.size() * 4, cudaMemcpyDeviceToHost);
      printf("== imp n%g m33 status=%d\n", n, (int)st);
      for (int y = 3; y <= 5; ++y) {
        for (int x = 3; x <= 5; ++x) printf(" %.9g", dst[(size_t)y * W + x]);
        printf("\n");
      }
    }
    cudaFree(d_s); cudaFree(d_d);
  }
  // C. 1D impulses at V>n (delta 30): h1 m31 / h1w m33 / w1 m13
  {
    std::vector<float> row(7, 0.f), col(7, 0.f);
    row[3] = 30.f; col[3] = 30.f;
    for (float n : {1.f, 0.5f})
      for (auto &t : {std::pair<int, int>{3, 1}, std::pair<int, int>{1, 3}, std::pair<int, int>{3, 3}}) {
        char tag[64];
        snprintf(tag, sizeof tag, "h1 imp m%dx%d n%g", t.first, t.second, n);
        w5(tag, row, 7, 1, t.first, t.second, t.first / 2, 0, n);
        snprintf(tag, sizeof tag, "w1 imp m%dx%d n%g", t.first, t.second, n);
        w5(tag, col, 1, 7, t.first, t.second, 0, t.second / 2, n);
      }
  }
  // D. vertical bar 3x5 (cols 0,5,0) : does the vertical mask see neighbours?
  {
    const int W = 3, H = 5;
    std::vector<float> src((size_t)W * H, 0.f);
    for (int y = 0; y < H; ++y) src[(size_t)y * W + 1] = 5.f;
    for (float n : {1.f, 0.5f}) {
      char tag[64];
      snprintf(tag, sizeof tag, "bar m13 a01 n%g", n); w5(tag, src, W, H, 1, 3, 0, 1, n);
      snprintf(tag, sizeof tag, "bar m31 a10 n%g", n); w5(tag, src, W, H, 3, 1, 1, 0, n);
      snprintf(tag, sizeof tag, "bar m33 a11 n%g", n); w5(tag, src, W, H, 3, 3, 1, 1, n);
    }
  }
  // E. const-value sweep : gamma_v(c) (m13 W=1), gamma_2D(c) (h1w m33, 7x7 m33)
  for (float c : {0.f, 0.5f, 1.f, 2.f, 5.f, 10.f, 20.f, 50.f}) {
    char tag[64];
    std::vector<float> col(7, c);
    snprintf(tag, sizeof tag, "cv w1 m13 c%g n1", c); w5(tag, col, 1, 7, 1, 3, 0, 1, 1.f);
    snprintf(tag, sizeof tag, "cv h1w m33 c%g n1", c); w5(tag, col, 7, 1, 3, 3, 1, 1, 1.f);
    std::vector<float> sq((size_t)7 * 7, c);
    snprintf(tag, sizeof tag, "cv 7x7 m33 c%g n1", c); w5(tag, sq, 7, 7, 3, 3, 1, 1, 1.f);
    snprintf(tag, sizeof tag, "cv 7x7 m55 c%g n1", c); w5(tag, sq, 7, 7, 5, 5, 2, 2, 1.f);
  }
  // F. two-delta superposition : 30 at (1,1) and (3,3) on 5x5
  {
    const int W = 5, H = 5;
    std::vector<float> two((size_t)W * H, 0.f), one((size_t)W * H, 0.f);
    two[(size_t)1 * W + 1] = 30.f; two[(size_t)3 * W + 3] = 30.f;
    one[(size_t)1 * W + 1] = 30.f;
    w5("two m33 n1", two, W, H, 3, 3, 1, 1, 1.f);
    w5("one m33 n1", one, W, H, 3, 3, 1, 1, 1.f);
  }
}

// ---------------- stage valmat : validation matrices for all 5 families ------
static void valmat() {
  const int W = 9, H = 9;
  float *d_s, *d_d;
  cudaMalloc(&d_s, (size_t)W * H * 4);
  cudaMalloc(&d_d, (size_t)W * H * 4);
  cudaMemset(d_s, 0x3f, (size_t)W * H * 4);
  float k3[3] = {0.25f, 0.5f, 0.25f};
  float *d_k;
  cudaMalloc(&d_k, 3 * 4);
  cudaMemcpy(d_k, k3, 12, cudaMemcpyHostToDevice);
  float noise[4] = {0.5f, 0.5f, 0.5f, 0.5f};
  const NppiSize roi{5, 5};
  const NppiSize sz{9, 9};
  const NppiPoint off{2, 2};
  NppStreamContext c = ctx();
  auto P = [](const char *tag, NppStatus st) { printf("vm %s status=%d\n", tag, (int)st); };
  // ---- Gauss plain
  P("gauss ok", nppiFilterGauss_32f_C1R_Ctx(d_s, 36, d_d, 36, roi, NPP_MASK_SIZE_3_X_3, c));
  P("gauss nullsrc", nppiFilterGauss_32f_C1R_Ctx(nullptr, 36, d_d, 36, roi, NPP_MASK_SIZE_3_X_3, c));
  P("gauss nulldst", nppiFilterGauss_32f_C1R_Ctx(d_s, 36, nullptr, 36, roi, NPP_MASK_SIZE_3_X_3, c));
  P("gauss negroi", nppiFilterGauss_32f_C1R_Ctx(d_s, 36, d_d, 36, NppiSize{5, -1}, NPP_MASK_SIZE_3_X_3, c));
  P("gauss zeroroi", nppiFilterGauss_32f_C1R_Ctx(d_s, 36, d_d, 36, NppiSize{0, 0}, NPP_MASK_SIZE_3_X_3, c));
  P("gauss zroi_badmask", nppiFilterGauss_32f_C1R_Ctx(d_s, 36, d_d, 36, NppiSize{0, 0}, (NppiMaskSize)0, c));
  P("gauss badmask0", nppiFilterGauss_32f_C1R_Ctx(d_s, 36, d_d, 36, roi, (NppiMaskSize)0, c));
  P("gauss badmask900", nppiFilterGauss_32f_C1R_Ctx(d_s, 36, d_d, 36, roi, (NppiMaskSize)900, c));
  P("gauss badmask-1", nppiFilterGauss_32f_C1R_Ctx(d_s, 36, d_d, 36, roi, (NppiMaskSize)-1, c));
  P("gauss negroi_badmask", nppiFilterGauss_32f_C1R_Ctx(d_s, 36, d_d, 36, NppiSize{-1, 5}, (NppiMaskSize)0, c));
  P("gauss stepsmall", nppiFilterGauss_32f_C1R_Ctx(d_s, 36, d_d, 19, roi, NPP_MASK_SIZE_3_X_3, c));
  P("gauss stepodd", nppiFilterGauss_32f_C1R_Ctx(d_s, 36, d_d, 21, roi, NPP_MASK_SIZE_3_X_3, c));
  P("gauss zroi_badstep", nppiFilterGauss_32f_C1R_Ctx(d_s, 36, d_d, 3, NppiSize{0, 0}, NPP_MASK_SIZE_3_X_3, c));
  P("gauss srcstep0", nppiFilterGauss_32f_C1R_Ctx(d_s, 0, d_d, 36, roi, NPP_MASK_SIZE_3_X_3, c));
  P("gauss srcstepneg", nppiFilterGauss_32f_C1R_Ctx(d_s, -36, d_d, 36, roi, NPP_MASK_SIZE_3_X_3, c));
  P("gauss srcstepodd", nppiFilterGauss_32f_C1R_Ctx(d_s, 34, d_d, 36, roi, NPP_MASK_SIZE_3_X_3, c));
  // ---- GaussBorder
  P("gb ok", nppiFilterGaussBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 36, roi, NPP_MASK_SIZE_3_X_3,
                                              NPP_BORDER_REPLICATE, c));
  P("gb nullsrc", nppiFilterGaussBorder_32f_C1R_Ctx(nullptr, 36, sz, off, d_d, 36, roi, NPP_MASK_SIZE_3_X_3,
                                                   NPP_BORDER_REPLICATE, c));
  P("gb nulldst", nppiFilterGaussBorder_32f_C1R_Ctx(d_s, 36, sz, off, nullptr, 36, roi, NPP_MASK_SIZE_3_X_3,
                                                   NPP_BORDER_REPLICATE, c));
  P("gb negroi", nppiFilterGaussBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 36, NppiSize{5, -1},
                                                  NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, c));
  P("gb zeroroi", nppiFilterGaussBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 36, NppiSize{0, 0},
                                                   NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, c));
  P("gb srcsize0", nppiFilterGaussBorder_32f_C1R_Ctx(d_s, 36, NppiSize{0, 5}, off, d_d, 36, roi,
                                                     NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, c));
  P("gb srcsizeneg", nppiFilterGaussBorder_32f_C1R_Ctx(d_s, 36, NppiSize{-1, 5}, off, d_d, 36, roi,
                                                       NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, c));
  P("gb offneg", nppiFilterGaussBorder_32f_C1R_Ctx(d_s, 36, sz, NppiPoint{-1, 0}, d_d, 36, roi,
                                                   NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, c));
  P("gb offpast", nppiFilterGaussBorder_32f_C1R_Ctx(d_s, 36, sz, NppiPoint{5, 5}, d_d, 36, roi,
                                                    NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, c));
  P("gb offway", nppiFilterGaussBorder_32f_C1R_Ctx(d_s, 36, sz, NppiPoint{20, 20}, d_d, 36, roi,
                                                   NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, c));
  P("gb offpluroi", nppiFilterGaussBorder_32f_C1R_Ctx(d_s, 36, sz, NppiPoint{5, 5}, d_d, 36, NppiSize{5, 5},
                                                      NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, c));
  P("gb badborder", nppiFilterGaussBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 36, roi, NPP_MASK_SIZE_3_X_3,
                                                      NPP_BORDER_CONSTANT, c));
  P("gb border100", nppiFilterGaussBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 36, roi, NPP_MASK_SIZE_3_X_3,
                                                      (NppiBorderType)100, c));
  P("gb border-1", nppiFilterGaussBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 36, roi, NPP_MASK_SIZE_3_X_3,
                                                     (NppiBorderType)-1, c));
  P("gb badmask", nppiFilterGaussBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 36, roi, (NppiMaskSize)0,
                                                    NPP_BORDER_REPLICATE, c));
  P("gb mask15", nppiFilterGaussBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 36, roi, NPP_MASK_SIZE_15_X_15,
                                                   NPP_BORDER_REPLICATE, c));
  P("gb stepsmall", nppiFilterGaussBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 19, roi, NPP_MASK_SIZE_3_X_3,
                                                      NPP_BORDER_REPLICATE, c));
  P("gb stepodd", nppiFilterGaussBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 21, roi, NPP_MASK_SIZE_3_X_3,
                                                    NPP_BORDER_REPLICATE, c));
  P("gb zroi_offneg", nppiFilterGaussBorder_32f_C1R_Ctx(d_s, 36, sz, NppiPoint{0, -1}, d_d, 36, NppiSize{0, 0},
                                                        NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, c));
  P("gb zroi_badborder", nppiFilterGaussBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 36, NppiSize{0, 0},
                                                           NPP_MASK_SIZE_3_X_3, NPP_BORDER_CONSTANT, c));
  P("gb zroi_badstep", nppiFilterGaussBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 3, NppiSize{0, 0},
                                                         NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, c));
  // ---- GaussAdvanced plain
  P("adv ok", nppiFilterGaussAdvanced_32f_C1R_Ctx(d_s, 36, d_d, 36, roi, 3, d_k, c));
  P("adv nullsrc", nppiFilterGaussAdvanced_32f_C1R_Ctx(nullptr, 36, d_d, 36, roi, 3, d_k, c));
  P("adv nulldst", nppiFilterGaussAdvanced_32f_C1R_Ctx(d_s, 36, nullptr, 36, roi, 3, d_k, c));
  P("adv nullkernel", nppiFilterGaussAdvanced_32f_C1R_Ctx(d_s, 36, d_d, 36, roi, 3, nullptr, c));
  P("adv negroi", nppiFilterGaussAdvanced_32f_C1R_Ctx(d_s, 36, d_d, 36, NppiSize{5, -1}, 3, d_k, c));
  P("adv zeroroi", nppiFilterGaussAdvanced_32f_C1R_Ctx(d_s, 36, d_d, 36, NppiSize{0, 0}, 3, d_k, c));
  P("adv taps0", nppiFilterGaussAdvanced_32f_C1R_Ctx(d_s, 36, d_d, 36, roi, 0, d_k, c));
  P("adv taps-1", nppiFilterGaussAdvanced_32f_C1R_Ctx(d_s, 36, d_d, 36, roi, -1, d_k, c));
  P("adv taps1", nppiFilterGaussAdvanced_32f_C1R_Ctx(d_s, 36, d_d, 36, roi, 1, d_k, c));
  P("adv taps2", nppiFilterGaussAdvanced_32f_C1R_Ctx(d_s, 36, d_d, 36, roi, 2, d_k, c));
  P("adv taps33", nppiFilterGaussAdvanced_32f_C1R_Ctx(d_s, 36, d_d, 36, roi, 33, d_k, c));
  P("adv taps65", nppiFilterGaussAdvanced_32f_C1R_Ctx(d_s, 36, d_d, 36, roi, 65, d_k, c));
  P("adv stepsmall", nppiFilterGaussAdvanced_32f_C1R_Ctx(d_s, 36, d_d, 19, roi, 3, d_k, c));
  P("adv stepodd", nppiFilterGaussAdvanced_32f_C1R_Ctx(d_s, 36, d_d, 21, roi, 3, d_k, c));
  P("adv zroi_nullkernel", nppiFilterGaussAdvanced_32f_C1R_Ctx(d_s, 36, d_d, 36, NppiSize{0, 0}, 3, nullptr, c));
  P("adv zroi_taps0", nppiFilterGaussAdvanced_32f_C1R_Ctx(d_s, 36, d_d, 36, NppiSize{0, 0}, 0, d_k, c));
  // ---- GaussAdvancedBorder
  P("advb ok", nppiFilterGaussAdvancedBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 36, roi, 3, d_k,
                                                         NPP_BORDER_REPLICATE, c));
  P("advb nullsrc", nppiFilterGaussAdvancedBorder_32f_C1R_Ctx(nullptr, 36, sz, off, d_d, 36, roi, 3, d_k,
                                                              NPP_BORDER_REPLICATE, c));
  P("advb nulldst", nppiFilterGaussAdvancedBorder_32f_C1R_Ctx(d_s, 36, sz, off, nullptr, 36, roi, 3, d_k,
                                                              NPP_BORDER_REPLICATE, c));
  P("advb nullkernel", nppiFilterGaussAdvancedBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 36, roi, 3, nullptr,
                                                                 NPP_BORDER_REPLICATE, c));
  P("advb negroi", nppiFilterGaussAdvancedBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 36, NppiSize{5, -1}, 3, d_k,
                                                             NPP_BORDER_REPLICATE, c));
  P("advb zeroroi", nppiFilterGaussAdvancedBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 36, NppiSize{0, 0}, 3, d_k,
                                                              NPP_BORDER_REPLICATE, c));
  P("advb srcsize0", nppiFilterGaussAdvancedBorder_32f_C1R_Ctx(d_s, 36, NppiSize{0, 5}, off, d_d, 36, roi, 3, d_k,
                                                               NPP_BORDER_REPLICATE, c));
  P("advb offneg", nppiFilterGaussAdvancedBorder_32f_C1R_Ctx(d_s, 36, sz, NppiPoint{-1, 0}, d_d, 36, roi, 3, d_k,
                                                             NPP_BORDER_REPLICATE, c));
  P("advb offpluroi", nppiFilterGaussAdvancedBorder_32f_C1R_Ctx(d_s, 36, sz, NppiPoint{5, 5}, d_d, 36,
                                                                NppiSize{5, 5}, 3, d_k, NPP_BORDER_REPLICATE, c));
  P("advb badborder", nppiFilterGaussAdvancedBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 36, roi, 3, d_k,
                                                                NPP_BORDER_CONSTANT, c));
  P("advb border100", nppiFilterGaussAdvancedBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 36, roi, 3, d_k,
                                                                (NppiBorderType)100, c));
  P("advb taps0", nppiFilterGaussAdvancedBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 36, roi, 0, d_k,
                                                            NPP_BORDER_REPLICATE, c));
  P("advb taps-1", nppiFilterGaussAdvancedBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 36, roi, -1, d_k,
                                                             NPP_BORDER_REPLICATE, c));
  P("advb taps33", nppiFilterGaussAdvancedBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 36, roi, 33, d_k,
                                                             NPP_BORDER_REPLICATE, c));
  P("advb stepsmall", nppiFilterGaussAdvancedBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 19, roi, 3, d_k,
                                                                NPP_BORDER_REPLICATE, c));
  P("advb stepodd", nppiFilterGaussAdvancedBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 21, roi, 3, d_k,
                                                              NPP_BORDER_REPLICATE, c));
  // ---- WienerBorder (32f C1)
  P("wi ok", nppiFilterWienerBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 36, roi, NppiSize{3, 3},
                                                NppiPoint{1, 1}, noise, NPP_BORDER_REPLICATE, c));
  P("wi nullsrc", nppiFilterWienerBorder_32f_C1R_Ctx(nullptr, 36, sz, off, d_d, 36, roi, NppiSize{3, 3},
                                                     NppiPoint{1, 1}, noise, NPP_BORDER_REPLICATE, c));
  P("wi nulldst", nppiFilterWienerBorder_32f_C1R_Ctx(d_s, 36, sz, off, nullptr, 36, roi, NppiSize{3, 3},
                                                     NppiPoint{1, 1}, noise, NPP_BORDER_REPLICATE, c));
  P("wi negroi", nppiFilterWienerBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 36, NppiSize{5, -1}, NppiSize{3, 3},
                                                    NppiPoint{1, 1}, noise, NPP_BORDER_REPLICATE, c));
  P("wi zeroroi", nppiFilterWienerBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 36, NppiSize{0, 0}, NppiSize{3, 3},
                                                     NppiPoint{1, 1}, noise, NPP_BORDER_REPLICATE, c));
  P("wi srcsize0", nppiFilterWienerBorder_32f_C1R_Ctx(d_s, 36, NppiSize{0, 5}, off, d_d, 36, roi,
                                                      NppiSize{3, 3}, NppiPoint{1, 1}, noise,
                                                      NPP_BORDER_REPLICATE, c));
  P("wi offneg", nppiFilterWienerBorder_32f_C1R_Ctx(d_s, 36, sz, NppiPoint{-1, 0}, d_d, 36, roi, NppiSize{3, 3},
                                                    NppiPoint{1, 1}, noise, NPP_BORDER_REPLICATE, c));
  P("wi offpluroi", nppiFilterWienerBorder_32f_C1R_Ctx(d_s, 36, sz, NppiPoint{5, 5}, d_d, 36, NppiSize{5, 5},
                                                       NppiSize{3, 3}, NppiPoint{1, 1}, noise,
                                                       NPP_BORDER_REPLICATE, c));
  P("wi badborder", nppiFilterWienerBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 36, roi, NppiSize{3, 3},
                                                       NppiPoint{1, 1}, noise, NPP_BORDER_CONSTANT, c));
  P("wi border100", nppiFilterWienerBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 36, roi, NppiSize{3, 3},
                                                       NppiPoint{1, 1}, noise, (NppiBorderType)100, c));
  P("wi anchor33", nppiFilterWienerBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 36, roi, NppiSize{3, 3},
                                                      NppiPoint{3, 3}, noise, NPP_BORDER_REPLICATE, c));
  P("wi anchor-1", nppiFilterWienerBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 36, roi, NppiSize{3, 3},
                                                      NppiPoint{-1, 0}, noise, NPP_BORDER_REPLICATE, c));
  P("wi mask0", nppiFilterWienerBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 36, roi, NppiSize{0, 0},
                                                   NppiPoint{1, 1}, noise, NPP_BORDER_REPLICATE, c));
  P("wi mask-3", nppiFilterWienerBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 36, roi, NppiSize{-3, -3},
                                                    NppiPoint{1, 1}, noise, NPP_BORDER_REPLICATE, c));
  P("wi mask2x2", nppiFilterWienerBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 36, roi, NppiSize{2, 2},
                                                     NppiPoint{0, 0}, noise, NPP_BORDER_REPLICATE, c));
  P("wi stepsmall", nppiFilterWienerBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 19, roi, NppiSize{3, 3},
                                                       NppiPoint{1, 1}, noise, NPP_BORDER_REPLICATE, c));
  P("wi stepodd", nppiFilterWienerBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 21, roi, NppiSize{3, 3},
                                                     NppiPoint{1, 1}, noise, NPP_BORDER_REPLICATE, c));
  // 8u C4 step alignment (pixel-aligned steps)
  {
    unsigned char *s4, *d4;
    cudaMalloc(&s4, (size_t)W * H * 4);
    cudaMalloc(&d4, (size_t)W * H * 4);
    cudaMemset(s4, 0x5a, (size_t)W * H * 4);
    P("gauss8uc4 ok", nppiFilterGauss_8u_C4R_Ctx(s4, 36, d4, 36, roi, NPP_MASK_SIZE_3_X_3, c));
    P("gauss8uc4 step21", nppiFilterGauss_8u_C4R_Ctx(s4, 36, d4, 21, roi, NPP_MASK_SIZE_3_X_3, c));
    P("gauss8uc4 step22", nppiFilterGauss_8u_C4R_Ctx(s4, 36, d4, 22, roi, NPP_MASK_SIZE_3_X_3, c));
    P("gb8uc4 step22", nppiFilterGaussBorder_8u_C4R_Ctx(s4, 36, sz, off, d4, 22, roi, NPP_MASK_SIZE_3_X_3,
                                                        NPP_BORDER_REPLICATE, c));
    P("advb8uc4 step22", nppiFilterGaussAdvancedBorder_8u_C4R_Ctx(s4, 36, sz, off, d4, 22, roi, 3, d_k,
                                                                  NPP_BORDER_REPLICATE, c));
    P("wi8uc4 step22", nppiFilterWienerBorder_8u_C4R_Ctx(s4, 36, sz, off, d4, 22, roi, NppiSize{3, 3},
                                                         NppiPoint{1, 1}, noise, NPP_BORDER_REPLICATE, c));
    P("wi8uc4 step21", nppiFilterWienerBorder_8u_C4R_Ctx(s4, 36, sz, off, d4, 21, roi, NppiSize{3, 3},
                                                         NppiPoint{1, 1}, noise, NPP_BORDER_REPLICATE, c));
    // AC4 alpha-preserve check: prefill dst alpha, run, dump one pixel's alpha
    cudaMemset(d4, 0xEE, (size_t)W * H * 4);
    NppStatus st = nppiFilterGauss_8u_AC4R_Ctx(s4, 36, d4, 36, roi, NPP_MASK_SIZE_3_X_3, c);
    unsigned char px[4] = {0, 0, 0, 0};
    cudaMemcpy(px, d4 + 2 * 36 + 2 * 4, 4, cudaMemcpyDeviceToHost);
    printf("vm gauss8uac4 status=%d px=[%d %d %d %d]\n", (int)st, px[0], px[1], px[2], px[3]);
    cudaMemset(d4, 0xEE, (size_t)W * H * 4);
    st = nppiFilterWienerBorder_8u_AC4R_Ctx(s4, 36, sz, off, d4, 36, roi, NppiSize{3, 3}, NppiPoint{1, 1},
                                            noise, NPP_BORDER_REPLICATE, c);
    cudaMemcpy(px, d4 + 2 * 36 + 2 * 4, 4, cudaMemcpyDeviceToHost);
    printf("vm wiener8uac4 status=%d px=[%d %d %d %d]\n", (int)st, px[0], px[1], px[2], px[3]);
    cudaFree(s4);
    cudaFree(d4);
  }
  // NOTE: null aNoise SEGFAULTS host-side on NPP 13.4 (last so the rest of the
  // matrix completes before the crash); keep as the final call.
  P("wi nullnoise", nppiFilterWienerBorder_32f_C1R_Ctx(d_s, 36, sz, off, d_d, 36, roi, NppiSize{3, 3},
                                                       NppiPoint{1, 1}, nullptr, NPP_BORDER_REPLICATE, c));
  fflush(stdout);
  cudaFree(d_s);
  cudaFree(d_d);
  cudaFree(d_k);
}


// ---------------- stage gap1 : offset read geometry via identity kernel ------
static void sync_chk(const char *tag, NppStatus st) {
  cudaError_t e = cudaDeviceSynchronize();
  cudaError_t le = cudaGetLastError();
  printf("%s st=%d sync=%d lasterr=%d\n", tag, (int)st, (int)e, (int)le);
}

static void gap1() {
  const int W = 9, H = 9;
  std::vector<float> src((size_t)W * H);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) src[(size_t)y * W + x] = (float)(y * 10 + x);
  float *d_s, *d_d, *d_k;
  cudaMalloc(&d_s, src.size() * 4);
  cudaMalloc(&d_d, 81 * 4);
  cudaMalloc(&d_k, 3 * 4);
  cudaMemcpy(d_s, src.data(), src.size() * 4, cudaMemcpyHostToDevice);
  float idk[3] = {0.f, 1.f, 0.f};
  cudaMemcpy(d_k, idk, 12, cudaMemcpyHostToDevice);
  auto dump = [&](const char *tag, NppStatus st, NppiSize roi) {
    printf("gap1 %s st=%d\n", tag, (int)st);
    std::vector<float> out((size_t)roi.width * roi.height, -777.f);
    cudaMemcpy(out.data(), d_d, out.size() * 4, cudaMemcpyDeviceToHost);
    for (int y = 0; y < roi.height; ++y) {
      printf("  ");
      for (int x = 0; x < roi.width; ++x) printf(" %g", out[(size_t)y * roi.width + x]);
      printf("\n");
    }
  };
  auto runb = [&](const char *tag, NppiPoint off, NppiSize roi, NppiBorderType b) {
    std::vector<float> mark((size_t)roi.width * roi.height, -777.f);
    cudaMemcpy(d_d, mark.data(), mark.size() * 4, cudaMemcpyHostToDevice);
    NppStatus st = nppiFilterGaussAdvancedBorder_32f_C1R_Ctx(
        d_s, W * 4, NppiSize{W, H}, off, d_d, W * 4, roi, 3, d_k, b, ctx());
    cudaDeviceSynchronize();
    dump(tag, st, roi);
  };
  runb("advb-id off23 r55 rep", NppiPoint{2, 3}, NppiSize{5, 5}, NPP_BORDER_REPLICATE);
  runb("advb-id off23 r77 rep", NppiPoint{2, 3}, NppiSize{7, 7}, NPP_BORDER_REPLICATE);
  runb("advb-id off66 r55 rep", NppiPoint{6, 6}, NppiSize{5, 5}, NPP_BORDER_REPLICATE);
  runb("advb-id off00 r99 rep", NppiPoint{0, 0}, NppiSize{9, 9}, NPP_BORDER_REPLICATE);
  runb("advb-id off23 r55 mir", NppiPoint{2, 3}, NppiSize{5, 5}, (NppiBorderType)4);
  runb("advb-id off00 r99 mir", NppiPoint{0, 0}, NppiSize{9, 9}, (NppiBorderType)4);
  runb("advb-id off77 r55 mir", NppiPoint{7, 7}, NppiSize{5, 5}, (NppiBorderType)4);
  float n05[1] = {0.5f};
  float n5[5] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
  auto runw = [&](const char *tag, NppiPoint off, NppiSize roi) {
    std::vector<float> mark((size_t)roi.width * roi.height, -777.f);
    cudaMemcpy(d_d, mark.data(), mark.size() * 4, cudaMemcpyHostToDevice);
    NppStatus st = nppiFilterWienerBorder_32f_C1R_Ctx(
        d_s, W * 4, NppiSize{W, H}, off, d_d, W * 4, roi, NppiSize{1, 1}, NppiPoint{0, 0},
        n05, NPP_BORDER_REPLICATE, ctx());
    cudaDeviceSynchronize();
    dump(tag, st, roi);
  };
  runw("wi-1x1 off23 r55", NppiPoint{2, 3}, NppiSize{5, 5});
  runw("wi-1x1 off66 r55", NppiPoint{6, 6}, NppiSize{5, 5});
  // Wiener 1x3 a00 on delta image: reveals row mapping under offset
  {
    std::vector<float> imp((size_t)W * H, 0.f);
    imp[(size_t)4 * W + 4] = 30.f;
    cudaMemcpy(d_s, imp.data(), imp.size() * 4, cudaMemcpyHostToDevice);
    float noise[1] = {0.0f};
    std::vector<float> mark(25, -777.f);
    cudaMemcpy(d_d, mark.data(), 100, cudaMemcpyHostToDevice);
    NppStatus st = nppiFilterWienerBorder_32f_C1R_Ctx(
        d_s, W * 4, NppiSize{W, H}, NppiPoint{2, 3}, d_d, W * 4, NppiSize{5, 5},
        NppiSize{1, 3}, NppiPoint{0, 0}, noise, NPP_BORDER_REPLICATE, ctx());
    cudaDeviceSynchronize();
    dump("wi-1x3a00-delta off23 r55", st, NppiSize{5, 5});
    cudaMemcpy(d_s, src.data(), src.size() * 4, cudaMemcpyHostToDevice);
  }
  // GaussBorder 3x3 with offset (cross-check executor + fixed taps)
  auto rung = [&](const char *tag, NppiPoint off, NppiSize roi) {
    std::vector<float> mark((size_t)roi.width * roi.height, -777.f);
    cudaMemcpy(d_d, mark.data(), mark.size() * 4, cudaMemcpyHostToDevice);
    NppStatus st = nppiFilterGaussBorder_32f_C1R_Ctx(d_s, W * 4, NppiSize{W, H}, off, d_d,
                                                     W * 4, roi, NPP_MASK_SIZE_3_X_3,
                                                     NPP_BORDER_REPLICATE, ctx());
    cudaDeviceSynchronize();
    dump(tag, st, roi);
  };
  rung("gb-3x3 off23 r55", NppiPoint{2, 3}, NppiSize{5, 5});
  rung("gb-3x3 off00 r99", NppiPoint{0, 0}, NppiSize{9, 9});
  {
    std::vector<float> mark(25, -777.f);
    cudaMemcpy(d_d, mark.data(), 100, cudaMemcpyHostToDevice);
    NppStatus st = nppiFilterGaussBorder_32f_C1R_Ctx(d_s, W * 4, NppiSize{W, H}, NppiPoint{2, 3},
                                                     d_d, W * 4, NppiSize{5, 5},
                                                     NPP_MASK_SIZE_5_X_5, NPP_BORDER_REPLICATE, ctx());
    cudaDeviceSynchronize();
    dump("gb-5x5 off23 r55", st, NppiSize{5, 5});
  }
  (void)n5;
  cudaFree(d_s); cudaFree(d_d); cudaFree(d_k);
}

// ---------------- stage gap2 : AC4 alpha behavior per family ------------------
static void gap2() {
  const int W = 9, H = 9, MW = 4;  // margin for plain families (3-tap max)
  const NppiSize roi{5, 5};
  const NppiSize sz{W, H};
  const NppiPoint off{2, 3};
  NppStreamContext c = ctx();
  // 8u: 13x13 buffer, RGB=100/101/102, A=200; plain families get interior pointer
  {
    const int BW = W + 2 * MW, BH = H + 2 * MW;
    std::vector<unsigned char> s4((size_t)BW * BH * 4);
    for (int y = 0; y < BH; ++y)
      for (int x = 0; x < BW; ++x) {
        s4[((size_t)y * BW + x) * 4 + 0] = 100;
        s4[((size_t)y * BW + x) * 4 + 1] = 101;
        s4[((size_t)y * BW + x) * 4 + 2] = 102;
        s4[((size_t)y * BW + x) * 4 + 3] = 200;
      }
    unsigned char *d_buf, *d_dst;
    cudaMalloc(&d_buf, s4.size());
    cudaMalloc(&d_dst, (size_t)W * H * 4);
    cudaMemcpy(d_buf, s4.data(), s4.size(), cudaMemcpyHostToDevice);
    unsigned char *p = d_buf + ((size_t)MW * BW + MW) * 4;
    auto case8 = [&](const char *tag, NppStatus st) {
      cudaDeviceSynchronize();
      unsigned char px[8] = {0};
      cudaMemcpy(px, d_dst + ((size_t)2 * (W * 4) + 2 * 4), 4, cudaMemcpyDeviceToHost);
      cudaMemcpy(px + 4, d_dst + 4, 4, cudaMemcpyDeviceToHost);
      printf("gap2 %s st=%d p22=[%d %d %d %d] p00=[%d %d %d %d] lasterr=%d\n", tag, (int)st,
             px[0], px[1], px[2], px[3], px[4], px[5], px[6], px[7], (int)cudaGetLastError());
    };
    std::vector<unsigned char> mark((size_t)W * H * 4, 0x77);
    auto pre = [&]() { cudaMemcpy(d_dst, mark.data(), mark.size(), cudaMemcpyHostToDevice); };
    pre();
    case8("gauss8u-AC4", nppiFilterGauss_8u_AC4R_Ctx(p, BW * 4, d_dst, W * 4, roi,
                                                     NPP_MASK_SIZE_3_X_3, c));
    pre();
    case8("gauss8u-C4", nppiFilterGauss_8u_C4R_Ctx(p, BW * 4, d_dst, W * 4, roi,
                                                    NPP_MASK_SIZE_3_X_3, c));
    pre();
    float noise4[4] = {0.5f, 0.5f, 0.5f, 0.5f};
    case8("wiener8u-AC4",
          nppiFilterWienerBorder_8u_AC4R_Ctx(p, BW * 4, sz, NppiPoint{MW, MW}, d_dst, W * 4,
                                             roi, NppiSize{3, 3}, NppiPoint{1, 1}, noise4,
                                             NPP_BORDER_REPLICATE, c));
    pre();
    case8("wiener8u-C4",
          nppiFilterWienerBorder_8u_C4R_Ctx(p, BW * 4, sz, NppiPoint{MW, MW}, d_dst, W * 4,
                                            roi, NppiSize{3, 3}, NppiPoint{1, 1}, noise4,
                                            NPP_BORDER_REPLICATE, c));
    // 32f AC4 on the same geometry
    std::vector<float> sf((size_t)BW * BH * 4);
    for (size_t i = 0; i < sf.size(); i += 4) {
      sf[i] = 100.f; sf[i + 1] = 101.f; sf[i + 2] = 102.f; sf[i + 3] = 200.f;
    }
    float *d_fbuf, *d_fdst;
    cudaMalloc(&d_fbuf, sf.size() * 4);
    cudaMalloc(&d_fdst, (size_t)W * H * 4);
    cudaMemcpy(d_fbuf, sf.data(), sf.size() * 4, cudaMemcpyHostToDevice);
    float *pf = d_fbuf + ((size_t)MW * BW + MW) * 4;
    float kern[3] = {0.f, 1.f, 0.f};
    float *d_k; cudaMalloc(&d_k, 12); cudaMemcpy(d_k, kern, 12, cudaMemcpyHostToDevice);
    std::vector<float> fmark((size_t)W * H * 4, -12345.f);
    auto fpre = [&]() { cudaMemcpy(d_fdst, fmark.data(), fmark.size() * 4, cudaMemcpyHostToDevice); };
    auto casef = [&](const char *tag, NppStatus st) {
      cudaDeviceSynchronize();
      float px[8] = {0};
      cudaMemcpy(px, d_fdst + (size_t)2 * (W * 4) + 2 * 4, 8, cudaMemcpyDeviceToHost);
      cudaMemcpy(px + 4, d_fdst + 4, 8, cudaMemcpyDeviceToHost);
      printf("gap2 %s st=%d p22=[%g %g %g %g] p00=[%g %g %g %g] lasterr=%d\n", tag, (int)st,
             px[0], px[1], px[2], px[3], px[4], px[5], px[6], px[7], (int)cudaGetLastError());
    };
    fpre();
    casef("gauss32f-AC4", nppiFilterGauss_32f_AC4R_Ctx(pf, BW * 4, d_fdst, W * 4, roi,
                                                        NPP_MASK_SIZE_3_X_3, c));
    fpre();
    casef("gauss32f-C4", nppiFilterGauss_32f_C4R_Ctx(pf, BW * 4, d_fdst, W * 4, roi,
                                                      NPP_MASK_SIZE_3_X_3, c));
    fpre();
    casef("gb32f-AC4", nppiFilterGaussBorder_32f_AC4R_Ctx(pf, BW * 4, sz, NppiPoint{MW, MW},
                                                           d_fdst, W * 4, roi,
                                                           NPP_MASK_SIZE_3_X_3,
                                                           NPP_BORDER_REPLICATE, c));
    fpre();
    casef("adv32f-AC4", nppiFilterGaussAdvanced_32f_AC4R_Ctx(pf, BW * 4, d_fdst, W * 4, roi,
                                                              3, d_k, c));
    fpre();
    casef("advb32f-AC4",
          nppiFilterGaussAdvancedBorder_32f_AC4R_Ctx(pf, BW * 4, sz, NppiPoint{MW, MW},
                                                     d_fdst, W * 4, roi, 3, d_k,
                                                     NPP_BORDER_REPLICATE, c));
    fpre();
    float noise3[3] = {0.5f, 0.5f, 0.5f};
    casef("wiener32f-AC4",
          nppiFilterWienerBorder_32f_AC4R_Ctx(pf, BW * 4, sz, NppiPoint{MW, MW}, d_fdst,
                                              W * 4, roi, NppiSize{3, 3}, NppiPoint{1, 1},
                                              noise3, NPP_BORDER_REPLICATE, c));
    fpre();
    casef("wiener32f-C4",
          nppiFilterWienerBorder_32f_C4R_Ctx(pf, BW * 4, sz, NppiPoint{MW, MW}, d_fdst,
                                             W * 4, roi, NppiSize{3, 3}, NppiPoint{1, 1},
                                             noise4, NPP_BORDER_REPLICATE, c));
    // 16s AC4 (negative RGB + alpha)
    std::vector<short> s16((size_t)BW * BH * 4);
    for (size_t i = 0; i < s16.size(); i += 4) {
      s16[i] = -100; s16[i + 1] = -101; s16[i + 2] = -102; s16[i + 3] = -200;
    }
    short *d_s16, *d_d16;
    cudaMalloc(&d_s16, s16.size() * 2);
    cudaMalloc(&d_d16, (size_t)W * H * 4 * 2);
    cudaMemcpy(d_s16, s16.data(), s16.size() * 2, cudaMemcpyHostToDevice);
    short *p16 = d_s16 + ((size_t)MW * BW + MW) * 4;
    std::vector<short> m16((size_t)W * H * 4, -12345);
    cudaMemcpy(d_d16, m16.data(), m16.size() * 2, cudaMemcpyHostToDevice);
    NppStatus st = nppiFilterGauss_16s_AC4R_Ctx(p16, BW * 8, d_d16, W * 8, roi,
                                                NPP_MASK_SIZE_3_X_3, c);
    cudaDeviceSynchronize();
    short q[8] = {0};
    cudaMemcpy(q, d_d16 + (size_t)2 * (W * 4) + 2 * 4, 8, cudaMemcpyDeviceToHost);
    printf("gap2 gauss16s-AC4 st=%d p22=[%d %d %d %d] lasterr=%d\n", (int)st, q[0], q[1], q[2],
           q[3], (int)cudaGetLastError());
    cudaFree(d_s16); cudaFree(d_d16);
    cudaFree(d_fbuf); cudaFree(d_fdst); cudaFree(d_k);
    cudaFree(d_buf); cudaFree(d_dst);
  }
}

// ---------------- stage gap3 : Wiener 8u/16s integer semantics ---------------
static void gap3() {
  const int W = 9, H = 9;
  NppStreamContext c = ctx();
  // 8u dumps
  {
    unsigned char *d_s, *d_d;
    cudaMalloc(&d_s, (size_t)W * H);
    cudaMalloc(&d_d, (size_t)W * H);
    auto run = [&](const char *tag, const std::vector<unsigned char> &img, NppiSize mask,
                   NppiPoint anc, float n, int rows0, int rows1, bool allrows) {
      cudaMemcpy(d_s, img.data(), img.size(), cudaMemcpyHostToDevice);
      std::vector<unsigned char> mark((size_t)W * H, 0x77);
      cudaMemcpy(d_d, mark.data(), mark.size(), cudaMemcpyHostToDevice);
      float noise[1] = {n};
      NppStatus st = nppiFilterWienerBorder_8u_C1R_Ctx(
          d_s, W, NppiSize{W, H}, NppiPoint{0, 0}, d_d, W, NppiSize{W, H}, mask, anc, noise,
          NPP_BORDER_REPLICATE, c);
      cudaDeviceSynchronize();
      std::vector<unsigned char> out((size_t)W * H, 0x77);
      cudaMemcpy(out.data(), d_d, out.size(), cudaMemcpyDeviceToHost);
      printf("gap3 %s st=%d\n", tag, (int)st);
      int y0 = allrows ? 0 : rows0, y1 = allrows ? H - 1 : rows1;
      for (int y = y0; y <= y1; ++y) {
        printf("  r%d:", y);
        for (int x = 0; x < W; ++x) printf(" %d", out[(size_t)y * W + x]);
        printf("\n");
      }
    };
    std::vector<unsigned char> ramp((size_t)W * H), z((size_t)W * H), imp((size_t)W * H),
        c2((size_t)W * H, 2), c9((size_t)W * H, 9), c10((size_t)W * H, 10), c101((size_t)W * H, 101);
    for (size_t i = 0; i < ramp.size(); ++i) ramp[i] = (unsigned char)((i * 37 + 11) & 0xFF);
    imp[(size_t)4 * W + 4] = 30;
    std::vector<unsigned char> imp100((size_t)W * H); imp100[(size_t)4 * W + 4] = 100;
    std::vector<unsigned char> imp255((size_t)W * H); imp255[(size_t)4 * W + 4] = 255;
    run("w8u ramp m33 n1", ramp, NppiSize{3, 3}, NppiPoint{1, 1}, 1.f, 2, 6, true);
    run("w8u ramp m33 n0", ramp, NppiSize{3, 3}, NppiPoint{1, 1}, 0.f, 2, 6, true);
    run("w8u ramp m33 n05", ramp, NppiSize{3, 3}, NppiPoint{1, 1}, 0.5f, 4, 4, false);
    run("w8u ramp m33 n025", ramp, NppiSize{3, 3}, NppiPoint{1, 1}, 0.25f, 4, 4, false);
    run("w8u imp30 m33 n1", imp, NppiSize{3, 3}, NppiPoint{1, 1}, 1.f, 3, 5, true);
    run("w8u imp30 m33 n0", imp, NppiSize{3, 3}, NppiPoint{1, 1}, 0.f, 3, 5, true);
    run("w8u imp100 m33 n0", imp100, NppiSize{3, 3}, NppiPoint{1, 1}, 0.f, 3, 5, true);
    run("w8u imp255 m55 n0", imp255, NppiSize{5, 5}, NppiPoint{2, 2}, 0.f, 2, 6, true);
    run("w8u imp255 m55 n1", imp255, NppiSize{5, 5}, NppiPoint{2, 2}, 1.f, 2, 6, true);
    run("w8u c2 m33 n1", c2, NppiSize{3, 3}, NppiPoint{1, 1}, 1.f, 3, 5, true);
    run("w8u c2 m33 n0", c2, NppiSize{3, 3}, NppiPoint{1, 1}, 0.f, 3, 5, true);
    run("w8u c9 m33 n1", c9, NppiSize{3, 3}, NppiPoint{1, 1}, 1.f, 3, 5, true);
    run("w8u c10 m33 n0", c10, NppiSize{3, 3}, NppiPoint{1, 1}, 0.f, 3, 5, true);
    run("w8u c10 m33 n1", c10, NppiSize{3, 3}, NppiPoint{1, 1}, 1.f, 3, 5, true);
    run("w8u c101 m22a00 n1", c101, NppiSize{2, 2}, NppiPoint{0, 0}, 1.f, 3, 5, true);
    run("w8u c101 m22a00 n0", c101, NppiSize{2, 2}, NppiPoint{0, 0}, 0.f, 3, 5, true);
    run("w8u c101 m55 n0", c101, NppiSize{5, 5}, NppiPoint{2, 2}, 0.f, 4, 4, false);
    // n-dependence on a mixed image
    for (float n : {0.1f, 0.9f}) run("w8u ramp m33 nx", ramp, NppiSize{3, 3}, NppiPoint{1, 1}, n, 4, 4, false);
    // 2x1 / 1x2 masks on ramp
    run("w8u ramp m21a00 n0", ramp, NppiSize{2, 1}, NppiPoint{0, 0}, 0.f, 4, 4, false);
    run("w8u ramp m12a00 n0", ramp, NppiSize{1, 2}, NppiPoint{0, 0}, 0.f, 4, 4, false);
    // per-channel noise validation (32f C3)
    cudaFree(d_s); cudaFree(d_d);
  }
  {
    float *d_s, *d_d;
    cudaMalloc(&d_s, (size_t)W * H * 3 * 4);
    cudaMalloc(&d_d, (size_t)W * H * 3 * 4);
    cudaMemset(d_s, 0x5A, (size_t)W * H * 3 * 4);
    float bad1[3] = {0.5f, 1.5f, 0.5f};
    float bad2[3] = {-0.5f, 0.5f, 0.5f};
    float good[3] = {0.5f, 0.5f, 0.5f};
    printf("gap3 wi-c3 noise155 st=%d\n", (int)nppiFilterWienerBorder_32f_C3R_Ctx(
        d_s, W * 12, NppiSize{W, H}, NppiPoint{0, 0}, d_d, W * 12, NppiSize{W, H},
        NppiSize{3, 3}, NppiPoint{1, 1}, bad1, NPP_BORDER_REPLICATE, c));
    printf("gap3 wi-c3 noisem05 st=%d\n", (int)nppiFilterWienerBorder_32f_C3R_Ctx(
        d_s, W * 12, NppiSize{W, H}, NppiPoint{0, 0}, d_d, W * 12, NppiSize{W, H},
        NppiSize{3, 3}, NppiPoint{1, 1}, bad2, NPP_BORDER_REPLICATE, c));
    printf("gap3 wi-c3 noisegood st=%d\n", (int)nppiFilterWienerBorder_32f_C3R_Ctx(
        d_s, W * 12, NppiSize{W, H}, NppiPoint{0, 0}, d_d, W * 12, NppiSize{W, H},
        NppiSize{3, 3}, NppiPoint{1, 1}, good, NPP_BORDER_REPLICATE, c));
    printf("gap3 wi-c1 noise11 st=%d\n", (int)nppiFilterWienerBorder_32f_C1R_Ctx(
        d_s, W * 12, NppiSize{W, H}, NppiPoint{0, 0}, d_d, W * 12, NppiSize{W, H},
        NppiSize{3, 3}, NppiPoint{1, 1}, bad1, NPP_BORDER_REPLICATE, c));
    cudaFree(d_s); cudaFree(d_d);
  }
  // 16s dumps
  {
    short *d_s, *d_d;
    cudaMalloc(&d_s, (size_t)W * H * 2);
    cudaMalloc(&d_d, (size_t)W * H * 2);
    auto run = [&](const char *tag, const std::vector<short> &img, NppiSize mask,
                   NppiPoint anc, float n, int rows0, int rows1, bool allrows) {
      cudaMemcpy(d_s, img.data(), img.size() * 2, cudaMemcpyHostToDevice);
      std::vector<short> mark((size_t)W * H, -12345);
      cudaMemcpy(d_d, mark.data(), mark.size() * 2, cudaMemcpyHostToDevice);
      float noise[1] = {n};
      NppStatus st = nppiFilterWienerBorder_16s_C1R_Ctx(
          d_s, W * 2, NppiSize{W, H}, NppiPoint{0, 0}, d_d, W * 2, NppiSize{W, H}, mask, anc,
          noise, NPP_BORDER_REPLICATE, c);
      cudaDeviceSynchronize();
      std::vector<short> out((size_t)W * H, -12345);
      cudaMemcpy(out.data(), d_d, out.size() * 2, cudaMemcpyDeviceToHost);
      printf("gap3 %s st=%d\n", tag, (int)st);
      int y0 = allrows ? 0 : rows0, y1 = allrows ? H - 1 : rows1;
      for (int y = y0; y <= y1; ++y) {
        printf("  r%d:", y);
        for (int x = 0; x < W; ++x) printf(" %d", out[(size_t)y * W + x]);
        printf("\n");
      }
    };
    std::vector<short> ramp((size_t)W * H), imp((size_t)W * H), c9((size_t)W * H, 9),
        c10((size_t)W * H, 10), cm9((size_t)W * H, -9), cm10((size_t)W * H, -10);
    for (size_t i = 0; i < ramp.size(); ++i) ramp[i] = (short)(((i * 137 + 51) % 600) - 300);
    imp[(size_t)4 * W + 4] = 1000;
    run("w16s ramp m33 n1", ramp, NppiSize{3, 3}, NppiPoint{1, 1}, 1.f, 3, 5, true);
    run("w16s ramp m33 n0", ramp, NppiSize{3, 3}, NppiPoint{1, 1}, 0.f, 3, 5, true);
    run("w16s ramp m33 n05", ramp, NppiSize{3, 3}, NppiPoint{1, 1}, 0.5f, 4, 4, false);
    run("w16s imp1000 m33 n0", imp, NppiSize{3, 3}, NppiPoint{1, 1}, 0.f, 3, 5, true);
    run("w16s imp1000 m33 n1", imp, NppiSize{3, 3}, NppiPoint{1, 1}, 1.f, 3, 5, true);
    run("w16s c9 m33 n1", c9, NppiSize{3, 3}, NppiPoint{1, 1}, 1.f, 4, 4, false);
    run("w16s c10 m33 n0", c10, NppiSize{3, 3}, NppiPoint{1, 1}, 0.f, 4, 4, false);
    run("w16s c10 m33 n1", c10, NppiSize{3, 3}, NppiPoint{1, 1}, 1.f, 4, 4, false);
    run("w16s cm9 m33 n1", cm9, NppiSize{3, 3}, NppiPoint{1, 1}, 1.f, 4, 4, false);
    run("w16s cm10 m33 n0", cm10, NppiSize{3, 3}, NppiPoint{1, 1}, 0.f, 4, 4, false);
    run("w16s cm10 m33 n1", cm10, NppiSize{3, 3}, NppiPoint{1, 1}, 1.f, 4, 4, false);
    cudaFree(d_s); cudaFree(d_d);
  }
}

// ---------------- stage gap4 : GaussAdvanced anchors/even taps/arithmetic ----
static void gap4() {
  NppStreamContext c = ctx();
  // 32f anchor probes: 21x21 zero image, impulse at (10,10), plain Advanced
  {
    const int W = 21, H = 21;
    std::vector<float> src((size_t)W * H, 0.f), dst((size_t)W * H, -777.f);
    src[(size_t)10 * W + 10] = 1.f;
    float *d_s, *d_d, *d_k;
    cudaMalloc(&d_s, src.size() * 4);
    cudaMalloc(&d_d, dst.size() * 4);
    cudaMalloc(&d_k, 8 * 4);
    cudaMemcpy(d_s, src.data(), src.size() * 4, cudaMemcpyHostToDevice);
    struct K { int taps; float k[8]; const char *tag; } cases[] = {
        {3, {1.f, 0.f, 0.f}, "t3-100"},
        {3, {0.f, 0.f, 1.f}, "t3-001"},
        {4, {1.f, 0.f, 0.f, 0.f}, "t4-1000"},
        {4, {0.f, 0.f, 0.f, 1.f}, "t4-0001"},
        {4, {0.f, 1.f, 0.f, 0.f}, "t4-0100"},
        {6, {1.f, 0.f, 0.f, 0.f, 0.f, 0.f}, "t6-100000"},
        {6, {0.f, 0.f, 0.f, 0.f, 0.f, 1.f}, "t6-000001"},
    };
    for (auto &kc : cases) {
      cudaMemcpy(d_k, kc.k, kc.taps * 4, cudaMemcpyHostToDevice);
      std::vector<float> mark((size_t)W * H, -777.f);
      cudaMemcpy(d_d, mark.data(), mark.size() * 4, cudaMemcpyHostToDevice);
      NppStatus st = nppiFilterGaussAdvanced_32f_C1R_Ctx(d_s, W * 4, d_d, W * 4,
                                                         NppiSize{W, H}, kc.taps, d_k, c);
      cudaDeviceSynchronize();
      std::vector<float> out((size_t)W * H, -777.f);
      cudaMemcpy(out.data(), d_d, out.size() * 4, cudaMemcpyDeviceToHost);
      printf("gap4 adv-anchor %s st=%d", kc.tag, (int)st);
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
          if (out[(size_t)y * W + x] > 0.5f) printf(" hit(%d,%d)=%.6g", x, y, out[(size_t)y * W + x]);
      printf("\n");
    }
    // even/odd taps status checks (plain)
    for (int t : {2, 4, 6, -1, -3}) {
      NppStatus st = nppiFilterGaussAdvanced_32f_C1R_Ctx(d_s, W * 4, d_d, W * 4,
                                                         NppiSize{5, 5}, t, d_k, c);
      printf("gap4 adv taps=%d st=%d\n", t, (int)st);
    }
    // AdvancedBorder taps status + even-tap identity dump
    float idk[4] = {0.f, 1.f, 0.f, 0.f};
    cudaMemcpy(d_k, idk, 16, cudaMemcpyHostToDevice);
    for (int t : {4, 6, -3}) {
      NppStatus st = nppiFilterGaussAdvancedBorder_32f_C1R_Ctx(
          d_s, W * 4, NppiSize{W, H}, NppiPoint{0, 0}, d_d, W * 4, NppiSize{5, 5}, t, d_k,
          NPP_BORDER_REPLICATE, c);
      printf("gap4 advb taps=%d st=%d\n", t, (int)st);
    }
    float idk1[4] = {1.f, 0.f, 0.f, 0.f};
    cudaMemcpy(d_k, idk1, 16, cudaMemcpyHostToDevice);
    std::vector<float> imp2((size_t)W * H, 0.f);
    imp2[(size_t)2 * W + 2] = 1.f;
    cudaMemcpy(d_s, imp2.data(), imp2.size() * 4, cudaMemcpyHostToDevice);
    std::vector<float> mark((size_t)W * H, -777.f);
    cudaMemcpy(d_d, mark.data(), mark.size() * 4, cudaMemcpyHostToDevice);
    NppStatus st = nppiFilterGaussAdvancedBorder_32f_C1R_Ctx(
        d_s, W * 4, NppiSize{W, H}, NppiPoint{0, 0}, d_d, W * 4, NppiSize{5, 5}, 4, d_k,
        NPP_BORDER_REPLICATE, c);
    cudaDeviceSynchronize();
    std::vector<float> out((size_t)W * H, -777.f);
    cudaMemcpy(out.data(), d_d, out.size() * 4, cudaMemcpyDeviceToHost);
    printf("gap4 advb-id t4-1000 rep st=%d:\n", (int)st);
    for (int y = 0; y < 5; ++y) {
      printf("  ");
      for (int x = 0; x < 5; ++x) printf(" %g", out[(size_t)y * W + x]);
      printf("\n");
    }
    cudaMemcpy(d_d, mark.data(), mark.size() * 4, cudaMemcpyHostToDevice);
    st = nppiFilterGaussAdvancedBorder_32f_C1R_Ctx(
        d_s, W * 4, NppiSize{W, H}, NppiPoint{0, 0}, d_d, W * 4, NppiSize{5, 5}, 4, d_k,
        (NppiBorderType)4, c);
    cudaDeviceSynchronize();
    cudaMemcpy(out.data(), d_d, out.size() * 4, cudaMemcpyDeviceToHost);
    printf("gap4 advb-id t4-1000 mir st=%d:\n", (int)st);
    for (int y = 0; y < 5; ++y) {
      printf("  ");
      for (int x = 0; x < 5; ++x) printf(" %g", out[(size_t)y * W + x]);
      printf("\n");
    }
    cudaFree(d_s); cudaFree(d_d); cudaFree(d_k);
  }
  // 8u/16s arithmetic: interior rows of 9x9 with k3/k5 (2D-pre vs separable fit)
  {
    const int W = 9, H = 9;
    float k3[3] = {0.25f, 0.5f, 0.25f};
    float k5[5] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};
    float ku[3] = {1.f, 2.f, 1.f};
    float *d_k; cudaMalloc(&d_k, 5 * 4);
    unsigned char *d_s, *d_d;
    cudaMalloc(&d_s, (size_t)W * H);
    cudaMalloc(&d_d, (size_t)W * H);
    std::vector<unsigned char> img((size_t)W * H);
    for (size_t i = 0; i < img.size(); ++i) img[i] = (unsigned char)((i * 37 + 11) & 0xFF);
    cudaMemcpy(d_s, img.data(), img.size(), cudaMemcpyHostToDevice);
    auto run8 = [&](const char *tag, int taps, const float *k) {
      cudaMemcpy(d_k, k, taps * 4, cudaMemcpyHostToDevice);
      std::vector<unsigned char> mark((size_t)W * H, 0x77);
      cudaMemcpy(d_d, mark.data(), mark.size(), cudaMemcpyHostToDevice);
      NppStatus st = nppiFilterGaussAdvanced_8u_C1R_Ctx(d_s, W, d_d, W, NppiSize{W, H}, taps,
                                                        d_k, c);
      cudaDeviceSynchronize();
      std::vector<unsigned char> out((size_t)W * H, 0x77);
      cudaMemcpy(out.data(), d_d, out.size(), cudaMemcpyDeviceToHost);
      printf("gap4 adv8 %s st=%d r2/r4/r6:\n", tag, (int)st);
      for (int y : {2, 4, 6}) {
        printf("  r%d:", y);
        for (int x = 1; x < 8; ++x) printf(" %d", out[(size_t)y * W + x]);
        printf("\n");
      }
    };
    run8("k3", 3, k3);
    run8("k5", 5, k5);
    run8("ku123", 3, ku);
    // AdvancedBorder 8u (replicate) with k3: rows 0/4/8
    cudaMemcpy(d_k, k3, 12, cudaMemcpyHostToDevice);
    std::vector<unsigned char> mark((size_t)W * H, 0x77);
    cudaMemcpy(d_d, mark.data(), mark.size(), cudaMemcpyHostToDevice);
    NppStatus st = nppiFilterGaussAdvancedBorder_8u_C1R_Ctx(
        d_s, W, NppiSize{W, H}, NppiPoint{0, 0}, d_d, W, NppiSize{W, H}, 3, d_k,
        NPP_BORDER_REPLICATE, c);
    cudaDeviceSynchronize();
    std::vector<unsigned char> out((size_t)W * H, 0x77);
    cudaMemcpy(out.data(), d_d, out.size(), cudaMemcpyDeviceToHost);
    printf("gap4 advb8 k3 rep st=%d:\n", (int)st);
    for (int y : {0, 4, 8}) {
      printf("  r%d:", y);
      for (int x = 0; x < W; ++x) printf(" %d", out[(size_t)y * W + x]);
      printf("\n");
    }
    cudaFree(d_s); cudaFree(d_d); cudaFree(d_k);
    // 16s
    short *d_ss, *d_dd;
    cudaMalloc(&d_ss, (size_t)W * H * 2);
    cudaMalloc(&d_dd, (size_t)W * H * 2);
    std::vector<short> img16((size_t)W * H);
    for (size_t i = 0; i < img16.size(); ++i) img16[i] = (short)(((i * 137 + 51) % 600) - 300);
    cudaMemcpy(d_ss, img16.data(), img16.size() * 2, cudaMemcpyHostToDevice);
    float *d_kk; cudaMalloc(&d_kk, 5 * 4);
    cudaMemcpy(d_kk, k3, 12, cudaMemcpyHostToDevice);
    std::vector<short> mark16((size_t)W * H, -12345);
    cudaMemcpy(d_dd, mark16.data(), mark16.size() * 2, cudaMemcpyHostToDevice);
    st = nppiFilterGaussAdvanced_16s_C1R_Ctx(d_ss, W * 2, d_dd, W * 2, NppiSize{W, H}, 3, d_kk, c);
    cudaDeviceSynchronize();
    std::vector<short> out16((size_t)W * H, -12345);
    cudaMemcpy(out16.data(), d_dd, out16.size() * 2, cudaMemcpyDeviceToHost);
    printf("gap4 adv16s k3 st=%d:\n", (int)st);
    for (int y : {2, 4, 6}) {
      printf("  r%d:", y);
      for (int x = 1; x < 8; ++x) printf(" %d", out16[(size_t)y * W + x]);
      printf("\n");
    }
    cudaFree(d_ss); cudaFree(d_dd); cudaFree(d_kk);
  }
}

// ---------------- stage gap5 : step alignment matrix -------------------------
static void gap5() {
  NppStreamContext c = ctx();
  const int W = 9, H = 9;
  const NppiSize roi{5, 5};
  const NppiSize sz{W, H};
  unsigned char *s8, *d8;
  unsigned short *s16;
  short *s16s;
  float *sf;
  cudaMalloc(&s8, (size_t)W * H * 4);
  cudaMalloc(&d8, (size_t)W * H * 4);
  cudaMalloc(&s16, (size_t)W * H * 2);
  cudaMalloc(&s16s, (size_t)W * H * 2);
  cudaMalloc(&sf, (size_t)W * H * 4 * 4);
  cudaMemset(s8, 0x5A, (size_t)W * H * 4);
  cudaMemset(s16, 0x11, (size_t)W * H * 2);
  cudaMemset(s16s, 0x11, (size_t)W * H * 2);
  cudaMemset(sf, 0x3F, (size_t)W * H * 4 * 4);
  printf("gap5 gauss8uC1 dststep11 st=%d\n", (int)nppiFilterGauss_8u_C1R_Ctx(s8, W, d8, 11, roi, NPP_MASK_SIZE_3_X_3, c));
  printf("gap5 gauss8uC1 dststep10 st=%d\n", (int)nppiFilterGauss_8u_C1R_Ctx(s8, W, d8, 10, roi, NPP_MASK_SIZE_3_X_3, c));
  printf("gap5 gauss8uC1 dststep5 st=%d\n", (int)nppiFilterGauss_8u_C1R_Ctx(s8, W, d8, 5, roi, NPP_MASK_SIZE_3_X_3, c));
  printf("gap5 gauss8uC3 dststep16 st=%d\n", (int)nppiFilterGauss_8u_C3R_Ctx(s8, 27, d8, 16, roi, NPP_MASK_SIZE_3_X_3, c));
  printf("gap5 gauss8uC3 dststep17 st=%d\n", (int)nppiFilterGauss_8u_C3R_Ctx(s8, 27, d8, 17, roi, NPP_MASK_SIZE_3_X_3, c));
  printf("gap5 gauss16uC1 dststep11 st=%d\n", (int)nppiFilterGauss_16u_C1R_Ctx(s16, 18, d8, 11, roi, NPP_MASK_SIZE_3_X_3, c));
  printf("gap5 gauss16sC3 dststep31 st=%d\n", (int)nppiFilterGauss_16s_C3R_Ctx(s16s, 54, (short *)d8, 31, roi, NPP_MASK_SIZE_3_X_3, c));
  printf("gap5 gauss32fC3 dststep61 st=%d\n", (int)nppiFilterGauss_32f_C3R_Ctx(sf, 108, (float *)d8, 61, roi, NPP_MASK_SIZE_3_X_3, c));
  printf("gap5 gauss32fC3 dststep64 st=%d\n", (int)nppiFilterGauss_32f_C3R_Ctx(sf, 108, (float *)d8, 64, roi, NPP_MASK_SIZE_3_X_3, c));
  float noise[4] = {0.5f, 0.5f, 0.5f, 0.5f};
  printf("gap5 wi8uC1 dststep11 st=%d\n", (int)nppiFilterWienerBorder_8u_C1R_Ctx(s8, 9, sz, NppiPoint{0, 0}, d8, 11, roi, NppiSize{3, 3}, NppiPoint{1, 1}, noise, NPP_BORDER_REPLICATE, c));
  printf("gap5 gb32fC1 srcstep37 st=%d\n", (int)nppiFilterGaussBorder_32f_C1R_Ctx(sf, 37, sz, NppiPoint{2, 2}, (float *)d8, 20, roi, NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, c));
  printf("gap5 gb8uC1 srcstep10 st=%d\n", (int)nppiFilterGaussBorder_8u_C1R_Ctx(s8, 10, sz, NppiPoint{2, 2}, d8, 5, roi, NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, c));
  printf("gap5 wi32fC1 srcstep37 st=%d\n", (int)nppiFilterWienerBorder_32f_C1R_Ctx(sf, 37, sz, NppiPoint{2, 2}, (float *)d8, 20, roi, NppiSize{3, 3}, NppiPoint{1, 1}, noise, NPP_BORDER_REPLICATE, c));
  cudaFree(s8); cudaFree(d8); cudaFree(s16); cudaFree(s16s); cudaFree(sf);
}

// ---------------- stage gap6 : Gauss integer masks 5/9/15 --------------------
static void gap6() {
  const int W = 17, H = 17;
  NppStreamContext c = ctx();
  unsigned char *d_s, *d_d;
  cudaMalloc(&d_s, (size_t)W * H);
  cudaMalloc(&d_d, (size_t)W * H);
  std::vector<unsigned char> img((size_t)W * H);
  for (size_t i = 0; i < img.size(); ++i) img[i] = (unsigned char)((i * 37 + 11) & 0xFF);
  cudaMemcpy(d_s, img.data(), img.size(), cudaMemcpyHostToDevice);
  const NppiMaskSize ms[] = {NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_5_X_5, NPP_MASK_SIZE_9_X_9,
                             NPP_MASK_SIZE_15_X_15};
  for (NppiMaskSize m : ms) {
    std::vector<unsigned char> mark((size_t)W * H, 0x77);
    cudaMemcpy(d_d, mark.data(), mark.size(), cudaMemcpyHostToDevice);
    NppStatus st = nppiFilterGauss_8u_C1R_Ctx(d_s, W, d_d, W, NppiSize{W, H}, m, c);
    cudaDeviceSynchronize();
    std::vector<unsigned char> out((size_t)W * H, 0x77);
    cudaMemcpy(out.data(), d_d, out.size(), cudaMemcpyDeviceToHost);
    printf("gap6 g8u mask=%d st=%d r8:", (int)m, (int)st);
    for (int x = 0; x < W; ++x) printf(" %d", out[(size_t)8 * W + x]);
    printf("\n");
  }
  cudaFree(d_s); cudaFree(d_d);
  short *d_ss, *d_dd;
  cudaMalloc(&d_ss, (size_t)W * H * 2);
  cudaMalloc(&d_dd, (size_t)W * H * 2);
  std::vector<short> img16((size_t)W * H);
  for (size_t i = 0; i < img16.size(); ++i) img16[i] = (short)(((i * 137 + 51) % 1800) - 900);
  cudaMemcpy(d_ss, img16.data(), img16.size() * 2, cudaMemcpyHostToDevice);
  for (NppiMaskSize m : ms) {
    std::vector<short> mark((size_t)W * H, -12345);
    cudaMemcpy(d_dd, mark.data(), mark.size() * 2, cudaMemcpyHostToDevice);
    NppStatus st = nppiFilterGauss_16s_C1R_Ctx(d_ss, W * 2, d_dd, W * 2, NppiSize{W, H}, m, c);
    cudaDeviceSynchronize();
    std::vector<short> out((size_t)W * H, -12345);
    cudaMemcpy(out.data(), d_dd, out.size() * 2, cudaMemcpyDeviceToHost);
    printf("gap6 g16s mask=%d st=%d r8:", (int)m, (int)st);
    for (int x = 0; x < W; ++x) printf(" %d", out[(size_t)8 * W + x]);
    printf("\n");
  }
  cudaFree(d_ss); cudaFree(d_dd);
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
  if (!strcmp(stage, "wiener4")) { printf("== wiener4 ==\n"); wiener4(); }
  if (!strcmp(stage, "wiener5")) { printf("== wiener5 ==\n"); wiener5(); }
  if (!strcmp(stage, "valmat") || !strcmp(stage, "all")) { printf("== valmat ==\n"); valmat(); }
  if (!strcmp(stage, "gap1")) { printf("== gap1 ==\n"); gap1(); }
  if (!strcmp(stage, "gap2")) { printf("== gap2 ==\n"); gap2(); }
  if (!strcmp(stage, "gap3")) { printf("== gap3 ==\n"); gap3(); }
  if (!strcmp(stage, "gap4")) { printf("== gap4 ==\n"); gap4(); }
  if (!strcmp(stage, "gap5")) { printf("== gap5 ==\n"); gap5(); }
  if (!strcmp(stage, "gap6")) { printf("== gap6 ==\n"); gap6(); }
  printf("done\n");
  return 0;
}
