#include <npp.h>
#include <nppcore.h>
#include <cuda_runtime.h>
#include <cstdio>
extern "C" { cudaStream_t nppGetStream(void); NppStatus nppGetStreamContext(NppStreamContext *); }
static NppStreamContext g;
template <class T> struct Buf { T *p; int step; Buf(int w,int h){ step=w*sizeof(T); cudaMalloc(&p,(size_t)step*h);} ~Buf(){cudaFree(p);} };
static void oc(const char *tag, NppStatus s){ printf("%-30s %d\n", tag, (int)s); }
int main(){ nppGetStreamContext(&g);
  const int SW=13,SH=11,W=9,H=7; NppiSize sz{SW,SH}; NppiPoint off{2,3};
  int masks[][2]={{1,1},{3,3},{5,5},{3,5},{5,3},{7,7}};
  int ans[][2]={{0,0},{1,1},{2,2}};
  for (auto &m:masks) for (auto &a:ans) {
    Buf<Npp8u> s(SW,SH),d(W,H); char t[64];
    snprintf(t,64,"8uC1 4 mask%dx%d a%d,%d",m[0],m[1],a[0],a[1]);
    oc(t,nppiFilterBoxBorder_8u_C1R_Ctx(s.p,s.step,sz,off,d.p,d.step,NppiSize{W,H},NppiSize{m[0],m[1]},NppiPoint{a[0],a[1]},NPP_BORDER_MIRROR,g));
    snprintf(t,64,"8uC1 5 mask%dx%d a%d,%d",m[0],m[1],a[0],a[1]);
    oc(t,nppiFilterBoxBorder_8u_C1R_Ctx(s.p,s.step,sz,off,d.p,d.step,NppiSize{W,H},NppiSize{m[0],m[1]},NppiPoint{a[0],a[1]},(NppiBorderType)5,g));
  }
  // 3x3 mirror per layout
  NppiPoint an{1,1}; NppiSize m3{3,3};
  { Buf<Npp8u>s(SW,SH),d(W*4,H); oc("8uC3 4 3x3",nppiFilterBoxBorder_8u_C3R_Ctx(s.p,s.step,sz,off,d.p,d.step,NppiSize{W,H},m3,an,NPP_BORDER_MIRROR,g)); }
  { Buf<Npp8u>s(SW*4,SH),d(W*4,H); oc("8uC4 4 3x3",nppiFilterBoxBorder_8u_C4R_Ctx(s.p,s.step,sz,off,d.p,d.step,NppiSize{W,H},m3,an,NPP_BORDER_MIRROR,g)); }
  { Buf<Npp8u>s(SW*4,SH),d(W*4,H); oc("8uAC4 4 3x3",nppiFilterBoxBorder_8u_AC4R_Ctx(s.p,s.step,sz,off,d.p,d.step,NppiSize{W,H},m3,an,NPP_BORDER_MIRROR,g)); }
  { Buf<Npp16u>s(SW,SH),d(W,H); oc("16uC1 4 3x3",nppiFilterBoxBorder_16u_C1R_Ctx(s.p,s.step,sz,off,d.p,d.step,NppiSize{W,H},m3,an,NPP_BORDER_MIRROR,g)); }
  { Buf<Npp16u>s(SW*3,SH),d(W*3,H); oc("16uC3 4 3x3",nppiFilterBoxBorder_16u_C3R_Ctx(s.p,s.step,sz,off,d.p,d.step,NppiSize{W,H},m3,an,NPP_BORDER_MIRROR,g)); }
  { Buf<Npp16u>s(SW*4,SH),d(W*4,H); oc("16uC4 4 3x3",nppiFilterBoxBorder_16u_C4R_Ctx(s.p,s.step,sz,off,d.p,d.step,NppiSize{W,H},m3,an,NPP_BORDER_MIRROR,g)); }
  { Buf<Npp16u>s(SW*4,SH),d(W*4,H); oc("16uAC4 4 3x3",nppiFilterBoxBorder_16u_AC4R_Ctx(s.p,s.step,sz,off,d.p,d.step,NppiSize{W,H},m3,an,NPP_BORDER_MIRROR,g)); }
  { Buf<Npp16s>s(SW,SH),d(W,H); oc("16sC1 4 3x3",nppiFilterBoxBorder_16s_C1R_Ctx(s.p,s.step,sz,off,d.p,d.step,NppiSize{W,H},m3,an,NPP_BORDER_MIRROR,g)); }
  { Buf<Npp16s>s(SW*3,SH),d(W*3,H); oc("16sC3 4 3x3",nppiFilterBoxBorder_16s_C3R_Ctx(s.p,s.step,sz,off,d.p,d.step,NppiSize{W,H},m3,an,NPP_BORDER_MIRROR,g)); }
  { Buf<Npp16s>s(SW*4,SH),d(W*4,H); oc("16sC4 4 3x3",nppiFilterBoxBorder_16s_C4R_Ctx(s.p,s.step,sz,off,d.p,d.step,NppiSize{W,H},m3,an,NPP_BORDER_MIRROR,g)); }
  { Buf<Npp16s>s(SW*4,SH),d(W*4,H); oc("16sAC4 4 3x3",nppiFilterBoxBorder_16s_AC4R_Ctx(s.p,s.step,sz,off,d.p,d.step,NppiSize{W,H},m3,an,NPP_BORDER_MIRROR,g)); }
  { Buf<Npp32f>s(SW,SH),d(W,H); oc("32fC1 4 3x3",nppiFilterBoxBorder_32f_C1R_Ctx(s.p,s.step,sz,off,d.p,d.step,NppiSize{W,H},m3,an,NPP_BORDER_MIRROR,g)); }
  { Buf<Npp32f>s(SW*3,SH),d(W*3,H); oc("32fC3 4 3x3",nppiFilterBoxBorder_32f_C3R_Ctx(s.p,s.step,sz,off,d.p,d.step,NppiSize{W,H},m3,an,NPP_BORDER_MIRROR,g)); }
  { Buf<Npp32f>s(SW*4,SH),d(W*4,H); oc("32fC4 4 3x3",nppiFilterBoxBorder_32f_C4R_Ctx(s.p,s.step,sz,off,d.p,d.step,NppiSize{W,H},m3,an,NPP_BORDER_MIRROR,g)); }
  { Buf<Npp32f>s(SW*4,SH),d(W*4,H); oc("32fAC4 4 3x3",nppiFilterBoxBorder_32f_AC4R_Ctx(s.p,s.step,sz,off,d.p,d.step,NppiSize{W,H},m3,an,NPP_BORDER_MIRROR,g)); }
  { Buf<Npp32f>s(SW*4,SH),d(W*4,H); oc("32fAC4 repl 3x3",nppiFilterBoxBorder_32f_AC4R_Ctx(s.p,s.step,sz,off,d.p,d.step,NppiSize{W,H},m3,an,NPP_BORDER_REPLICATE,g)); }
  return 0; }
