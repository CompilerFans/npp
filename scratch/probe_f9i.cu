// f9i: even-mask median window geometry via a positive ramp source
#include <npp.h>
#include <nppcore.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <vector>
extern "C" { cudaStream_t nppGetStream(void); NppStatus nppGetStreamContext(NppStreamContext *); }
static NppStreamContext g;
template <class T> struct Buf { T*p; int step; Buf(int w,int h){step=w*sizeof(T);cudaMalloc(&p,(size_t)step*h);cudaMemset(p,0,(size_t)step*h);} ~Buf(){cudaFree(p);} };

static void run(int mw,int mh,int ax,int ay,bool border){
  const int SW=16,SH=16,W=9,H=7,M=6;
  const int hsW=SW+2*M, hsH=SH+2*M;
  std::vector<int> him((size_t)hsW*hsH,0);
  for(int j=0;j<SH;++j) for(int i=0;i<SW;++i) him[(size_t)(j+M)*hsW+i+M]=1000*i+j;
  Buf<Npp16s> s(hsW,hsH);
  for(int j=0;j<SH;++j){ std::vector<Npp16s> row(SW); for(int i=0;i<SW;++i) row[i]=(Npp16s)him[(size_t)(j+M)*hsW+i+M];
    cudaMemcpy((char*)s.p+(size_t)(j+M)*s.step+(size_t)M*2, row.data(), SW*2, cudaMemcpyHostToDevice);}
  Buf<Npp16s> d(12,H);
  NppStatus st;
  char t[40]; snprintf(t,40,"%dx%d a%d,%d %s",mw,mh,ax,ay,border?"border":"plain");
  if(!border) st=nppiFilterMedian_16s_C1R_Ctx((Npp16s*)((char*)s.p+(size_t)M*s.step+(size_t)M*2),s.step,d.p,d.step,NppiSize{W,H},NppiSize{mw,mh},NppiPoint{ax,ay},nullptr,g);
  else st=nppiFilterMedianBorder_16s_C1R_Ctx((Npp16s*)((char*)s.p+(size_t)M*s.step+(size_t)M*2),s.step,NppiSize{SW,SH},NppiPoint{0,0},d.p,d.step,NppiSize{W,H},NppiSize{mw,mh},NppiPoint{ax,ay},nullptr,NPP_BORDER_REPLICATE,g);
  std::vector<Npp16s> o((size_t)W*H); cudaMemcpy2D(o.data(),W*2,d.p,d.step,W*2,H,cudaMemcpyDeviceToHost);
  { std::vector<Npp16s> chk((size_t)SW*SH); cudaMemcpy2D(chk.data(),SW*2,(const char*)s.p+(size_t)M*s.step+(size_t)M*2,s.step,SW*2,SH,cudaMemcpyDeviceToHost);
    printf("  src row1:"); for(int i=0;i<SW;++i) printf(" %d",chk[(size_t)SW+i]); printf("\n");
    printf("  dst row1:"); for(int x=0;x<W;++x) printf(" %d",o[(size_t)W+x]); printf("\n"); }
  int best=9999,bdx=0,bdy=0; long long besttr=9999;
  for(int dx=-3;dx<=3;++dx) for(int dy=-3;dy<=3;++dy){
    int bad=0, badtr=0;
    for(int y=0;y<H;++y) for(int x=0;x<W;++x){
      std::vector<int> w;
      for(int j=0;j<mh;++j) for(int i=0;i<mw;++i){
        int sx=x+i-ax+dx, sy=y+j-ay+dy;
        if(border){sx=std::clamp(sx,0,SW-1);sy=std::clamp(sy,0,SH-1);}
        w.push_back(him[(size_t)(sy+M)*hsW+sx+M]);
      }
      std::sort(w.begin(),w.end()); int n=(int)w.size();
      long long tr;
      if(n%2) tr=w[n/2]; else tr=(w[(n-1)/2]+w[n/2])/2;
      if(o[(size_t)y*W+x]!=tr) ++bad;
      long long fl = (n%2)? w[n/2] : ((long long)w[(n-1)/2]+w[n/2])>>1;
      if(o[(size_t)y*W+x]!=fl) ++badtr;
    }
    if(bad<best){best=bad;bdx=dx;bdy=dy;besttr=badtr;}
  }
  printf("%-24s st=%d best offset dx=%d dy=%d bad=%d (floor bad=%lld)\n",t,(int)st,bdx,bdy,best,besttr);
  if(best>0){ printf("  dst:"); for(int k=0;k<W*H;++k) printf(" %d",o[k]); printf("\n"); }
}
int main(){ nppGetStreamContext(&g);
  run(1,2,0,1,false); run(2,1,1,0,false); run(2,2,1,1,false); run(3,3,1,1,false);
  run(1,2,0,1,true);  run(2,1,1,0,true);  run(2,2,1,1,true);  run(3,3,1,1,true);
  return 0; }
