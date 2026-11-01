// f9d: fine-grained median GetBufferSize selection grid (host-only)
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <npp.h>

extern "C" NppStatus nppGetStreamContext(NppStreamContext *pNppStreamContext);
static NppStreamContext ctx;
static const char *sn(NppStatus s) {
  switch (s) {
    case NPP_NO_ERROR: return "OK";
    case NPP_SIZE_ERROR: return "SIZE";
    case NPP_BAD_ARGUMENT_ERROR: return "BADARG";
    default: return "?";
  }
}
#define BS(dt, lay) nppiFilterMedianGetBufferSize_##dt##_##lay##_Ctx
#define BSB(dt, lay) nppiFilterMedianBorderGetBufferSize_##dt##_##lay##_Ctx

template <class F>
static void val0(const char *nm, F f) {
  Npp32u s = 0xdeadbeef;
  NppStatus st = f(NppiSize{0, 0}, &s);
  printf("%-16s mask0x0: %u (%d %s)\n", nm, s, (int)st, sn(st));
  s = 0xdeadbeef; st = f(NppiSize{0, 3}, &s);
  printf("%-16s mask0x3: %u (%d %s)\n", nm, s, (int)st, sn(st));
  s = 0xdeadbeef; st = f(NppiSize{3, 0}, &s);
  printf("%-16s mask3x0: %u (%d %s)\n", nm, s, (int)st, sn(st));
}

static void grid8u() {
  printf("== grid 8u C1: '.'=0 '#'>0, w=1..40 cols, h=1..40 rows ==\n");
  for (int h = 1; h <= 40; ++h) {
    printf("h%2d ", h);
    for (int w = 1; w <= 40; ++w) {
      Npp32u s = 1;
      BS(8u, C1R)(NppiSize{16, 16}, NppiSize{w, h}, &s, ctx);
      putchar(s ? '#' : '.');
    }
    printf("\n");
  }
  printf("== grid 8u C3 ==\n");
  for (int h = 1; h <= 40; ++h) {
    printf("h%2d ", h);
    for (int w = 1; w <= 40; ++w) {
      Npp32u s = 1;
      BS(8u, C3R)(NppiSize{16, 16}, NppiSize{w, h}, &s, ctx);
      putchar(s ? '#' : '.');
    }
    printf("\n");
  }
  printf("== grid 8u C4 ==\n");
  for (int h = 1; h <= 40; ++h) {
    printf("h%2d ", h);
    for (int w = 1; w <= 40; ++w) {
      Npp32u s = 1;
      BS(8u, C4R)(NppiSize{16, 16}, NppiSize{w, h}, &s, ctx);
      putchar(s ? '#' : '.');
    }
    printf("\n");
  }
  printf("== grid 8u Border C1 (REPLICATE) ==\n");
  for (int h = 1; h <= 40; ++h) {
    printf("h%2d ", h);
    for (int w = 1; w <= 40; ++w) {
      Npp32u s = 1;
      BSB(8u, C1R)(NppiSize{16, 16}, NppiSize{w, h}, &s, NPP_BORDER_REPLICATE, ctx);
      putchar(s ? '#' : '.');
    }
    printf("\n");
  }
  fflush(stdout);
}

static void griddt(const char *tag, int which) {
  printf("== grid %s C1 ==\n", tag);
  for (int h = 1; h <= 40; ++h) {
    printf("h%2d ", h);
    for (int w = 1; w <= 40; ++w) {
      Npp32u s = 1;
      if (which == 0) BS(16u, C1R)(NppiSize{16, 16}, NppiSize{w, h}, &s, ctx);
      else if (which == 1) BS(16s, C1R)(NppiSize{16, 16}, NppiSize{w, h}, &s, ctx);
      else BS(32f, C1R)(NppiSize{16, 16}, NppiSize{w, h}, &s, ctx);
      putchar(s ? '#' : '.');
    }
    printf("\n");
  }
  fflush(stdout);
}

static void tall() {
  printf("== tall/wide specials 8u C1: mask, size ==\n");
  const int ms[][2] = {{1,126},{1,127},{1,128},{1,129},{1,130},{2,63},{2,64},{2,65},
    {3,42},{3,43},{4,31},{4,32},{5,25},{5,26},{6,21},{6,22},{7,18},{7,19},{8,15},{8,16},
    {9,14},{9,15},{10,12},{10,13},{11,12},{12,11},{13,11},{14,11},{15,10},{16,9},{17,9},
    {63,1},{64,1},{65,1},{127,2},{128,2},{129,2},{48,1},{96,1},{49,2},{50,2},{48,2}};
  for (auto &m : ms) {
    Npp32u s = 1;
    BS(8u, C1R)(NppiSize{16, 16}, NppiSize{m[0], m[1]}, &s, ctx);
    printf("mask %3dx%-3d: %u\n", m[0], m[1], s);
  }
  printf("== same for 16u C1 ==\n");
  for (auto &m : ms) {
    Npp32u s = 1;
    BS(16u, C1R)(NppiSize{16, 16}, NppiSize{m[0], m[1]}, &s, ctx);
    printf("mask %3dx%-3d: %u\n", m[0], m[1], s);
  }
  printf("== same for 16s C1 ==\n");
  for (auto &m : ms) {
    Npp32u s = 1;
    BS(16s, C1R)(NppiSize{16, 16}, NppiSize{m[0], m[1]}, &s, ctx);
    printf("mask %3dx%-3d: %u\n", m[0], m[1], s);
  }
  printf("== same for 32f C1 ==\n");
  for (auto &m : ms) {
    Npp32u s = 1;
    BS(32f, C1R)(NppiSize{16, 16}, NppiSize{m[0], m[1]}, &s, ctx);
    printf("mask %3dx%-3d: %u\n", m[0], m[1], s);
  }
  // verify formula 49152*es*area on nonzero cells
  printf("== formula check (8u C3/C4/AC4, 16u, 32f) ==\n");
  const int ms2[][2] = {{12,12},{14,14},{15,16},{16,16},{1,64},{2,33},{9,16},{1,129},{13,13},{11,11},{10,10},{5,13},{4,17},{6,11},{3,22}};
  for (auto &m : ms2) {
    Npp32u s3=1,s4=1,sa=1,s32=1,s16=1;
    BS(8u, C3R)(NppiSize{16,16}, NppiSize{m[0],m[1]}, &s3, ctx);
    BS(8u, C4R)(NppiSize{16,16}, NppiSize{m[0],m[1]}, &s4, ctx);
    BS(8u, AC4R)(NppiSize{16,16}, NppiSize{m[0],m[1]}, &sa, ctx);
    BS(16u, C1R)(NppiSize{16,16}, NppiSize{m[0],m[1]}, &s16, ctx);
    BS(32f, C1R)(NppiSize{16,16}, NppiSize{m[0],m[1]}, &s32, ctx);
    printf("mask %2dx%-2d area=%4d: C3=%-9u C4=%-9u AC4=%-9u 16u=%-9u 32f=%-9u (k49152*C1*es: C3? C4? 16u=%u 32f=%u)\n",
           m[0], m[1], m[0]*m[1], s3, s4, sa, s16, s32, 98304u*(unsigned)(m[0]*m[1]), 196608u*(unsigned)(m[0]*m[1]));
  }
  fflush(stdout);
}

int main(int argc, char **argv) {
  nppGetStreamContext(&ctx);
  printf("nppGetStreamContext -> %d\n", 0);
  const char *sec = argc > 1 ? argv[1] : "all";
  if (!strcmp(sec, "all") || !strcmp(sec, "grid8u")) grid8u();
  if (!strcmp(sec, "all") || !strcmp(sec, "grid16u")) griddt("16u", 0);
  if (!strcmp(sec, "all") || !strcmp(sec, "grid16s")) griddt("16s", 1);
  if (!strcmp(sec, "all") || !strcmp(sec, "grid32f")) griddt("32f", 2);
  if (!strcmp(sec, "all") || !strcmp(sec, "tall")) tall();
  if (!strcmp(sec, "all") || !strcmp(sec, "val0")) {
    val0("8u C1", [](NppiSize m, Npp32u *s) { return BS(8u, C1R)(NppiSize{16, 16}, m, s, ctx); });
    val0("32f C1", [](NppiSize m, Npp32u *s) { return BS(32f, C1R)(NppiSize{16, 16}, m, s, ctx); });
    val0("B8u C1", [](NppiSize m, Npp32u *s) { return BSB(8u, C1R)(NppiSize{16, 16}, m, s, NPP_BORDER_REPLICATE, ctx); });
  }
  printf("== probe_f9d done ==\n");
  return 0;
}
