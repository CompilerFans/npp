// nppiFilterBoxBorderAdvanced validation matrix (8u_C1R + others), RTX 5090.
#include <cuda_runtime.h>
#include <npp.h>
#include "nppcore.h"
extern "C" { NppStatus nppGetStreamContext(NppStreamContext *); }
#include <cstdio>
#include <vector>

static NppStreamContext g_ctx;
static const char *sn(NppStatus s) {
  switch (s) {
    case NPP_NO_ERROR: return "OK";
    case NPP_BAD_ARGUMENT_ERROR: return "BAD_ARG";
    case NPP_SIZE_ERROR: return "SIZE";
    case NPP_NULL_POINTER_ERROR: return "NULLPTR";
    case NPP_OUT_OFF_RANGE_ERROR: return "OUT_OFF";
    case NPP_STEP_ERROR: return "STEP";
    case NPP_NOT_EVEN_STEP_ERROR: return "NOTEVEVEN";
    case NPP_MASK_SIZE_ERROR: return "MASK";
    case NPP_ANCHOR_ERROR: return "ANCHOR";
    case NPP_NOT_SUPPORTED_MODE_ERROR: return "NOTSUP";
    case NPP_CUDA_KERNEL_EXECUTION_ERROR: return "CUDAKERN";
    default: return "OTHER";
  }
}

struct Args {
  const Npp8u *src = nullptr; int sstep = 64; NppiSize ssize{64, 64}; NppiPoint soff{0, 0};
  Npp8u *dst = nullptr; int dstep = 64; NppiSize roi{64, 64}; NppiSize mask{3, 3};
  NppiPoint anchor{1, 1}; NppiBorderType border = NPP_BORDER_REPLICATE; Npp8u *buf = nullptr;
};

static void run(const char *tag, Args a) {
  NppStatus st = nppiFilterBoxBorderAdvanced_8u_C1R_Ctx(a.src, a.sstep, a.ssize, a.soff, a.dst,
      a.dstep, a.roi, a.mask, a.anchor, a.border, a.buf, g_ctx);
  printf("%-34s -> %s\n", tag, sn(st));
  fflush(stdout);
}

int main() {
  nppGetStreamContext(&g_ctx);
  Npp8u *sp = nullptr, *dp = nullptr, *bp = nullptr;
  cudaMalloc(&sp, 8192); cudaMemset(sp, 7, 8192);
  cudaMalloc(&dp, 8192); cudaMemset(dp, 0, 8192);
  cudaMalloc(&bp, 65536); cudaMemset(bp, 0, 65536);
  Args base; base.src = sp; base.dst = dp; base.buf = bp;
  run("baseline 64x64 m3", base);
  { Args a = base; a.src = nullptr; run("null pSrc", a); }
  { Args a = base; a.dst = nullptr; run("null pDst", a); }
  { Args a = base; a.buf = nullptr; run("null pBuffer", a); }
  { Args a = base; a.roi = {-1, 64}; run("neg roi w", a); }
  { Args a = base; a.roi = {64, -1}; run("neg roi h", a); }
  { Args a = base; a.roi = {0, 64}; run("zero roi w", a); }
  { Args a = base; a.roi = {0, 0}; run("zero roi", a); }
  { Args a = base; a.mask = {-1, 3}; run("neg mask w", a); }
  { Args a = base; a.mask = {3, -1}; run("neg mask h", a); }
  { Args a = base; a.mask = {0, 0}; run("zero mask", a); }
  { Args a = base; a.mask = {0, 3}; run("zero mask w", a); }
  { Args a = base; a.anchor = {3, 1}; run("anchor x == maskW", a); }
  { Args a = base; a.anchor = {-1, 1}; run("anchor x = -1", a); }
  { Args a = base; a.anchor = {2, 2}; run("anchor 2,2 for 3x3", a); }
  { Args a = base; a.dstep = 32; run("dst step 32 < 64", a); }
  { Args a = base; a.dstep = 63; run("dst step 63", a); }
  { Args a = base; a.dstep = 0; run("dst step 0", a); }
  { Args a = base; a.dstep = -64; run("dst step -64", a); }
  { Args a = base; a.sstep = 32; run("src step 32 < 64", a); }
  { Args a = base; a.sstep = 0; run("src step 0", a); }
  { Args a = base; a.sstep = 63; run("src step 63", a); }
  { Args a = base; a.border = NPP_BORDER_CONSTANT; run("border constant", a); }
  { Args a = base; a.border = NPP_BORDER_WRAP; run("border wrap", a); }
  { Args a = base; a.border = NPP_BORDER_MIRROR; run("border mirror", a); }
  { Args a = base; a.border = NPP_BORDER_NONE; run("border none", a); }
  { Args a = base; a.border = (NppiBorderType)5; run("border 5", a); }
  { Args a = base; a.border = (NppiBorderType)7; run("border 7", a); }
  { Args a = base; a.soff = {8, 8}; a.roi = {56, 56}; run("offset 8 roi 56", a); }
  { Args a = base; a.soff = {-4, -4}; a.roi = {56, 56}; run("offset -4 roi 56", a); }
  { Args a = base; a.roi = {128, 128}; run("roi > src", a); }
  { Args a = base; a.ssize = {0, 0}; run("src size 0", a); }
  { Args a = base; a.mask = {65, 3}; run("mask wider than src", a); }
  return 0;
}
