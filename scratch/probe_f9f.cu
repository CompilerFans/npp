// f9f: targeted follow-ups found by scratch/verify_f9.cu
//  A) nppiFilterBoxBorder border/mask acceptance per layout
//  B) median even-count rounding for 2x2 (16s) and median-border mirror 1x2
#include <npp.h>
#include <nppcore.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

extern "C" { cudaStream_t nppGetStream(void); NppStatus nppGetStreamContext(NppStreamContext *); }
static NppStreamContext g;
static const char *sn(NppStatus s) {
  switch (s) {
  case NPP_NO_ERROR: return "OK";
  case NPP_MASK_SIZE_ERROR: return "MASK_SIZE";
  case NPP_NOT_SUPPORTED_MODE_ERROR: return "NOT_SUPPORTED";
  case NPP_OUT_OFF_RANGE_ERROR: return "OUT_OFF";
  default: return "other";
  }
}
template <class T> struct Buf { T *p; int step; Buf(int w,int h){ step=w*sizeof(T); cudaMalloc(&p,(size_t)step*h); cudaMemset(p,0xAB,(size_t)step*h);} ~Buf(){cudaFree(p);} };

static void boxb_accept() {
  const int SW=13, SH=11, W=9, H=7;
  int masks[][2]={{1,1},{3,1},{1,3},{3,3},{5,3},{3,5},{5,5},{7,7},{7,3},{9,5},{3,3}};
  int types[]={2,4,5};
  const char *tn[]={"rep2","mir4","v5"};
  for (int mi=0; mi<11; ++mi)
    for (int ti=0; ti<3; ++ti) {
      int mw=masks[mi][0], mh=masks[mi][1];
      // 8u C1
      Buf<Npp8u> s8(SW,SH), d8(W,H);
      NppiSize sz{SW,SH}; NppiPoint off{2,3}, an{1,1};
      NppStatus a=nppiFilterBoxBorder_8u_C1R_Ctx(s8.p,s8.step,sz,off,d8.p,d8.step,NppiSize{W,H},NppiSize{mw,mh},an,(NppiBorderType)types[ti],g);
      NppiPoint an2{mw/2,mh/2};
      NppStatus a2=nppiFilterBoxBorder_8u_C1R_Ctx(s8.p,s8.step,sz,off,d8.p,d8.step,NppiSize{W,H},NppiSize{mw,mh},an2,(NppiBorderType)types[ti],g);
      // 32f C1
      Buf<Npp32f> s32(SW,SH), d32(W,H);
      NppStatus b=nppiFilterBoxBorder_32f_C1R_Ctx(s32.p,s32.step,sz,off,d32.p,d32.step,NppiSize{W,H},NppiSize{mw,mh},an,(NppiBorderType)types[ti],g);
      NppStatus b2=nppiFilterBoxBorder_32f_C1R_Ctx(s32.p,s32.step,sz,off,d32.p,d32.step,NppiSize{W,H},NppiSize{mw,mh},an2,(NppiBorderType)types[ti],g);
      // 8u AC4 / 32f AC4, anchor centre
      Buf<Npp8u> sa4(SW*4,SH), da4(W*4,H);
      NppStatus c=nppiFilterBoxBorder_8u_AC4R_Ctx(sa4.p,sa4.step,sz,off,da4.p,da4.step,NppiSize{W,H},NppiSize{mw,mh},an2,(NppiBorderType)types[ti],g);
      Buf<Npp32f> sf4(SW*4,SH), df4(W*4,H);
      NppStatus e=nppiFilterBoxBorder_32f_AC4R_Ctx(sf4.p,sf4.step,sz,off,df4.p,df4.step,NppiSize{W,H},NppiSize{mw,mh},an2,(NppiBorderType)types[ti],g);
      printf("mask %dx%d %-4s an11: 8uC1=%s 32fC1=%s | anC: 8uC1=%s 32fC1=%s 8uAC4=%s 32fAC4=%s\n",
             mw,mh,tn[ti],sn(a),sn(b),sn(a2),sn(b2),sn(c),sn(e));
      if (c!=NPP_NO_ERROR||e!=NPP_NO_ERROR) printf("     raw AC4 status 8u=%d 32f=%d\n",(int)c,(int)e);
    }
}

static void median_rounding() {
  // 16s C1 plain 2x2 anchor(1,0), image 8x8 values 10*i+j-50
  {
    const int SW=12,SH=12,W=5,H=5;
    std::vector<Npp16s> h((size_t)SW*SH);
    for (int j=0;j<SH;++j) for(int i=0;i<SW;++i) h[(size_t)j*SW+i]=(Npp16s)(10*i+j-50);
    Buf<Npp16s> s(SW,SH); cudaMemcpy2D(s.p,s.step,h.data(),SW*2,SW*2,SH,cudaMemcpyHostToDevice);
    Buf<Npp16s> d(W,H);
    NppStatus st=nppiFilterMedian_16s_C1R_Ctx(s.p,s.step,d.p,d.step,NppiSize{W,H},NppiSize{2,2},NppiPoint{1,0},nullptr,g);
    std::vector<Npp16s> o((size_t)W*H); cudaMemcpy2D(o.data(),W*2,d.p,d.step,W*2,H,cudaMemcpyDeviceToHost);
    printf("median16s 2x2 a1,0: %d %s\n", (int)st, sn(st));
    for (int y=0;y<H;++y){ for(int x=0;x<W;++x){ int a=h[(size_t)y*SW+x-1], b=h[(size_t)y*SW+x], c=h[(size_t)(y+1)*SW+x-1], e=h[(size_t)(y+1)*SW+x];
        int v[4]={a,b,c,e}; std::sort(v,v+4); int t=(v[1]+v[2])/2; int fl=(v[1]+v[2])>>1;
        printf("  (%d,%d) got=%d win=%d,%d,%d,%d mid=%d,%d trunc=%d floor=%d\n",x,y,o[(size_t)y*W+x],v[0],v[1],v[2],v[3],v[1],v[2],t,fl);} }
  }
  // 16s median border mirror 1x2 vertical, src = i-j
  {
    const int SW=10,SH=10,W=6,H=1;
    std::vector<Npp16s> h((size_t)SW*SH);
    for (int j=0;j<SH;++j) for(int i=0;i<SW;++i) h[(size_t)j*SW+i]=(Npp16s)(i-j);
    Buf<Npp16s> s(SW,SH); cudaMemcpy2D(s.p,s.step,h.data(),SW*2,SW*2,SH,cudaMemcpyHostToDevice);
    Buf<Npp16s> d(W,H);
    NppStatus st=nppiFilterMedianBorder_16s_C1R_Ctx(s.p,s.step,NppiSize{SW,SH},NppiPoint{0,0},d.p,d.step,NppiSize{W,H},NppiSize{1,2},NppiPoint{0,0},nullptr,NPP_BORDER_MIRROR,g);
    std::vector<Npp16s> o((size_t)W*H); cudaMemcpy2D(o.data(),W*2,d.p,d.step,W*2,H,cudaMemcpyDeviceToHost);
    printf("medianborder16s mirror 1x2 a0,0: %d %s ->", (int)st, sn(st));
    for (int x=0;x<W;++x) printf(" %d", o[x]);
    printf("\n");
  }
}
int main(){ nppGetStreamContext(&g); boxb_accept(); median_rounding(); return 0; }
