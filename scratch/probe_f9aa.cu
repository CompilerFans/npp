// Advanced: validation precedence + offset rule (8u_C1R / 16u_C3R spot checks)
#include <cuda_runtime.h>
#include <npp.h>
#include "nppcore.h"
extern "C" { NppStatus nppGetStreamContext(NppStreamContext *); }
#include <cstdio>
static NppStreamContext g_ctx;
static const char *sn(NppStatus s) {
  switch (s) {
    case NPP_NO_ERROR: return "OK";
    case NPP_BAD_ARGUMENT_ERROR: return "BAD_ARG";
    case NPP_SIZE_ERROR: return "SIZE";
    case NPP_NULL_POINTER_ERROR: return "NULLPTR";
    case NPP_OUT_OFF_RANGE_ERROR: return "OUT_OFF";
    case NPP_STEP_ERROR: return "STEP";
    case NPP_NOT_EVEN_STEP_ERROR: return "NOTEVEN";
    case NPP_NOT_SUPPORTED_MODE_ERROR: return "NOTSUP";
    default: return "OTHER";
  }
}
struct A {
  const Npp8u *s = nullptr; int ss = 64; NppiSize sz{64, 64}; NppiPoint o{0, 0};
  Npp8u *d = nullptr; int ds = 64; NppiSize r{64, 64}; NppiSize m{3, 3}; NppiPoint an{1, 1};
  NppiBorderType b = NPP_BORDER_REPLICATE; Npp8u *p = nullptr;
};
static void run(const char *t, A a) {
  printf("%-30s -> %s\n", t,
         sn(nppiFilterBoxBorderAdvanced_8u_C1R_Ctx(a.s, a.ss, a.sz, a.o, a.d, a.ds, a.r, a.m, a.an, a.b, a.p, g_ctx)));
  fflush(stdout);
}
int main() {
  nppGetStreamContext(&g_ctx);
  Npp8u *s = nullptr, *d = nullptr, *p = nullptr;
  cudaMalloc(&s, 1 << 20); cudaMemset(s, 1, 1 << 20);
  cudaMalloc(&d, 1 << 20); cudaMemset(d, 0, 1 << 20);
  cudaMalloc(&p, 1 << 20); cudaMemset(p, 0, 1 << 20);
  A base; base.s = s; base.d = d; base.p = p;
  run("base", base);
  // offsets
  int offs[][3] = {{63, 0, 1}, {64, 0, 1}, {65, 0, 1}, {0, 63, 1}, {0, 64, 1}, {0, 65, 1},
                   {-1, 0, 1}, {0, -1, 1}, {-1000, 0, 1}, {0, -1000, 1}};
  for (auto &oo : offs) { A a = base; a.o = {oo[0], oo[1]}; char t[64]; snprintf(t, sizeof t, "offset %d,%d", oo[0], oo[1]); run(t, a); }
  // offsets with small src
  { A a = base; a.sz = {8, 8}; a.r = {8, 8}; a.ds = 8; a.ss = 8; a.o = {7, 7}; run("src8 offset 7,7", a); }
  { A a = base; a.sz = {8, 8}; a.r = {8, 8}; a.ds = 8; a.ss = 8; a.o = {8, 8}; run("src8 offset 8,8", a); }
  { A a = base; a.sz = {8, 8}; a.r = {8, 8}; a.ds = 8; a.ss = 8; a.o = {0, 0}; run("src8 offset 0,0", a); }
  { A a = base; a.sz = {0, 8}; run("srcsize 0x8", a); }
  { A a = base; a.sz = {8, 0}; run("srcsize 8x0", a); }
  { A a = base; a.sz = {-1, 8}; run("srcsize -1x8", a); }
  // precedence
  { A a = base; a.b = NPP_BORDER_MIRROR; a.s = nullptr; run("null src + mirror", a); }
  { A a = base; a.b = NPP_BORDER_MIRROR; a.r = {-1, 4}; run("neg roi + mirror", a); }
  { A a = base; a.r = {-1, 4}; a.ds = 0; run("neg roi + dstep0", a); }
  { A a = base; a.ds = 0; a.p = nullptr; run("dstep0 + null buf", a); }
  { A a = base; a.r = {0, 0}; a.ds = 0; run("empty roi + dstep0", a); }
  { A a = base; a.r = {0, 0}; a.b = NPP_BORDER_MIRROR; run("empty roi + mirror", a); }
  { A a = base; a.r = {0, 0}; a.p = nullptr; run("empty roi + null buf", a); }
  { A a = base; a.r = {0, 8}; run("roi 0x8", a); }
  { A a = base; a.r = {8, 0}; run("roi 8x0", a); }
  { A a = base; a.r = {8, 0}; a.ds = 8; run("roi 8x0 dstep8", a); }
  { A a = base; a.r = {0, 8}; a.ds = 0; run("roi 0x8 dstep0", a); }
  { A a = base; a.r = {0, 0}; a.ds = 0; a.b = NPP_BORDER_MIRROR; a.p = nullptr; run("empty+dstep0+mirror+nullbuf", a); }
  { A a = base; a.m = {-5, -5}; a.an = {-9, -9}; run("neg mask + bad anchor", a); }
  { A a = base; a.m = {100, 100}; a.an = {99, 99}; run("mask 100 anchor 99", a); }
  // dst step alignment for 8u (es=1): any value ok; negative
  { A a = base; a.ds = -1; run("dstep -1", a); }
  { A a = base; a.ds = -64; run("dstep -64", a); }
  { A a = base; a.ds = 63; run("dstep 63", a); }
  { A a = base; a.ds = 64; a.r = {64, 64}; run("dstep 64", a); }
  { A a = base; a.r = {56, 56}; a.o = {8, 8}; a.ds = 56; run("roi56 off8 dstep56", a); }
  { A a = base; a.r = {56, 56}; a.o = {0, 8}; a.ds = 56; run("roi56 off0,8 dstep56", a); }
  // second round: precedence details
  { A a = base; a.s = nullptr; a.r = {-1, 4}; run("nullsrc + negroi", a); }
  { A a = base; a.s = nullptr; a.d = nullptr; run("nullsrc + nulldst", a); }
  { A a = base; a.d = nullptr; a.p = nullptr; run("nulldst + nullbuf", a); }
  { A a = base; a.p = nullptr; a.r = {-1, 4}; run("nullbuf + negroi", a); }
  { A a = base; a.o = {99, 0}; a.b = NPP_BORDER_MIRROR; run("badoff + mirror", a); }
  { A a = base; a.o = {99, 0}; a.p = nullptr; run("badoff + nullbuf", a); }
  { A a = base; a.r = {-1, 4}; a.o = {99, 0}; run("negroi + badoff", a); }
  { A a = base; a.r = {-1, 4}; a.sz = {0, 0}; run("negroi + srcsize0", a); }
  { A a = base; a.sz = {0, 0}; a.p = nullptr; run("srcsize0 + nullbuf", a); }
  { A a = base; a.sz = {0, 0}; a.b = NPP_BORDER_MIRROR; run("srcsize0 + mirror", a); }
  { A a = base; a.r = {8, 0}; a.ds = 0; run("roi 8x0 dstep0", a); }
  { A a = base; a.r = {0, 8}; a.b = NPP_BORDER_MIRROR; run("roi 0x8 mirror", a); }
  { A a = base; a.r = {0, 0}; a.o = {99, 0}; run("empty roi + badoff", a); }
  { A a = base; a.r = {0, 0}; a.sz = {0, 0}; run("empty roi + srcsize0", a); }
  return 0;
}
