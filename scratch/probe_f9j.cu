// f9j: zero-ROI status per dtype/layout for box / boxborder / min / max
#include <npp.h>
#include <nppcore.h>
#include <cuda_runtime.h>
#include <cstdio>
extern "C" { cudaStream_t nppGetStream(void); NppStatus nppGetStreamContext(NppStreamContext *); }
static NppStreamContext g;
template <class T> struct B { T*p; int step; B(int w,int h){step=w*sizeof(T);cudaMalloc(&p,(size_t)step*h);} ~B(){cudaFree(p);} };
#define CHK(tag, expr) printf("%-18s %d\n", tag, (int)(expr))
int main(){ nppGetStreamContext(&g); NppiSize m3{3,3}; NppiPoint a1{1,1};
  { B<Npp8u>s(8,8),d(8,8); CHK("box 8uC1 roi0", nppiFilterBox_8u_C1R_Ctx(s.p,s.step,d.p,d.step,NppiSize{0,7},m3,a1,g)); }
  { B<Npp8u>s(32,32),d(32,32); CHK("box 8uC4 roi0", nppiFilterBox_8u_C4R_Ctx(s.p,s.step,d.p,d.step,NppiSize{0,7},m3,a1,g)); }
  { B<Npp8u>s(32,32),d(32,32); CHK("box 8uAC4 roi0", nppiFilterBox_8u_AC4R_Ctx(s.p,s.step,d.p,d.step,NppiSize{0,7},m3,a1,g)); }
  { B<Npp16u>s(8,8),d(8,8); CHK("box 16uC1 roi0", nppiFilterBox_16u_C1R_Ctx(s.p,s.step,d.p,d.step,NppiSize{0,7},m3,a1,g)); }
  { B<Npp16u>s(32,32),d(32,32); CHK("box 16uC3 roi0", nppiFilterBox_16u_C3R_Ctx(s.p,s.step,d.p,d.step,NppiSize{0,7},m3,a1,g)); }
  { B<Npp16s>s(8,8),d(8,8); CHK("box 16sC1 roi0", nppiFilterBox_16s_C1R_Ctx(s.p,s.step,d.p,d.step,NppiSize{0,7},m3,a1,g)); }
  { B<Npp32f>s(8,8),d(8,8); CHK("box 32fC1 roi0", nppiFilterBox_32f_C1R_Ctx(s.p,s.step,d.p,d.step,NppiSize{0,7},m3,a1,g)); }
  { B<Npp64f>s(8,8),d(8,8); CHK("box 64fC1 roi0", nppiFilterBox_64f_C1R_Ctx(s.p,s.step,d.p,d.step,NppiSize{0,7},m3,a1,g)); }
  // h zero
  { B<Npp8u>s(8,8),d(8,8); CHK("box 8uC1 roi0h", nppiFilterBox_8u_C1R_Ctx(s.p,s.step,d.p,d.step,NppiSize{7,0},m3,a1,g)); }
  { B<Npp16u>s(8,8),d(8,8); CHK("box 16uC1 roi0h", nppiFilterBox_16u_C1R_Ctx(s.p,s.step,d.p,d.step,NppiSize{7,0},m3,a1,g)); }
  // border
  NppiSize src{13,11}; NppiPoint off{2,3};
  { B<Npp8u>s(13,11),d(8,8); CHK("boxb 8uC1 roi0", nppiFilterBoxBorder_8u_C1R_Ctx(s.p,s.step,src,off,d.p,d.step,NppiSize{0,7},m3,a1,NPP_BORDER_REPLICATE,g)); }
  { B<Npp8u>s(52,11),d(32,8); CHK("boxb 8uC4 roi0", nppiFilterBoxBorder_8u_C4R_Ctx(s.p,s.step,src,off,d.p,d.step,NppiSize{0,7},m3,a1,NPP_BORDER_REPLICATE,g)); }
  { B<Npp16u>s(13,11),d(8,8); CHK("boxb 16uC1 roi0", nppiFilterBoxBorder_16u_C1R_Ctx(s.p,s.step,src,off,d.p,d.step,NppiSize{0,7},m3,a1,NPP_BORDER_REPLICATE,g)); }
  { B<Npp32f>s(13,11),d(8,8); CHK("boxb 32fC1 roi0", nppiFilterBoxBorder_32f_C1R_Ctx(s.p,s.step,src,off,d.p,d.step,NppiSize{0,7},m3,a1,NPP_BORDER_REPLICATE,g)); }
  // min/max plain + border zero roi per dtype
  { B<Npp8u>s(8,8),d(8,8); CHK("max 8uC1 roi0", nppiFilterMax_8u_C1R_Ctx(s.p,s.step,d.p,d.step,NppiSize{0,7},m3,a1,g)); }
  { B<Npp8u>s(8,8),d(8,8); CHK("min 8uC1 roi0", nppiFilterMin_8u_C1R_Ctx(s.p,s.step,d.p,d.step,NppiSize{0,7},m3,a1,g)); }
  { B<Npp8u>s(13,11),d(8,8); CHK("maxb 8uC1 roi0", nppiFilterMaxBorder_8u_C1R_Ctx(s.p,s.step,src,off,d.p,d.step,NppiSize{0,7},m3,a1,NPP_BORDER_REPLICATE,g)); }
  { B<Npp8u>s(13,11),d(8,8); CHK("med 8uC4 roi0", nppiFilterMedian_8u_C4R_Ctx(s.p,s.step,d.p,d.step,NppiSize{0,7},m3,a1,nullptr,g)); }
  return 0; }
