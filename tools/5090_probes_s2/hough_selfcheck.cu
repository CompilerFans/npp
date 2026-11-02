// Compares the mcMPP Hough line transform with NVIDIA NPP 13.4.
//
// Build and run on the RTX 5090:
//   nvcc -O2 -std=c++17 -o /tmp/hough_selfcheck hough_selfcheck.cu \
//       -lnppif -lnppial -lnppc -lcudart && /tmp/hough_selfcheck
//
// The include macro renames the public entry points so the probe can call the
// mcMPP definitions and the NPP versions from libnppif side by side.  It
// compares sorted line lists (NPP's own multi-line order comes from unordered
// parallel writes).  Clean single/multi-line images at delta (1,1) match
// exactly; the threshold-1 single-pixel case differs in trig/bin rounding.
#include <cuda_runtime.h>
#include <npp.h>
#include <nppi.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
extern "C" { NppStatus nppGetStreamContext(NppStreamContext *c); }
#define nppiFilterHoughLine_8u32f_C1R_Ctx mine_hough
#define nppiFilterHoughLine_8u32f_C1R mine_hough_nc
#define nppiFilterHoughLineRegion_8u32f_C1R_Ctx mine_hough_r
#define nppiFilterHoughLineRegion_8u32f_C1R mine_hough_rnc
#include "../../src/nppi/nppi_filtering_functions/nppi_hough_line.cu"
#undef nppiFilterHoughLine_8u32f_C1R_Ctx
#undef nppiFilterHoughLine_8u32f_C1R
#undef nppiFilterHoughLineRegion_8u32f_C1R_Ctx
#undef nppiFilterHoughLineRegion_8u32f_C1R
static NppStreamContext ctx(){NppStreamContext c{};nppGetStreamContext(&c);return c;}
static int gFail=0;
static bool cmpLine(const NppPointPolar&a,const NppPointPolar&b){ if(a.rho!=b.rho)return a.rho<b.rho; return a.theta<b.theta; }
static void run(const char*tag,const std::vector<Npp8u>&img,int W,int H,int thr,int maxl,bool region=false,NppPointPolar lo={0,0},NppPointPolar hi={0,0}){
  int bs=0;nppiFilterHoughLineGetBufferSize({W,H},{1,1},maxl,&bs);
  Npp8u*sR=nullptr,*bR=nullptr,*sM=nullptr,*bM=nullptr;NppPointPolar*lR=nullptr,*lM=nullptr;int*cR=nullptr,*cM=nullptr;
  cudaMalloc(&sR,img.size());cudaMalloc(&bR,bs);cudaMalloc(&lR,(size_t)maxl*sizeof(NppPointPolar));cudaMalloc(&cR,4);
  cudaMalloc(&sM,img.size());cudaMalloc(&bM,bs);cudaMalloc(&lM,(size_t)maxl*sizeof(NppPointPolar));cudaMalloc(&cM,4);
  cudaMemcpy(sR,img.data(),img.size(),cudaMemcpyHostToDevice);cudaMemcpy(sM,img.data(),img.size(),cudaMemcpyHostToDevice);
  cudaMemset(bR,0,bs);cudaMemset(bM,0,bs);cudaMemset(cR,0,4);cudaMemset(cM,0,4);
  int sRst,sMst;
  if(region){
    NppPointPolar rb[2]={lo,hi};
    sRst=nppiFilterHoughLineRegion_8u32f_C1R_Ctx(sR,W,{W,H},{1,1},thr,lR,rb,maxl,cR,bR,ctx());
    sMst=mine_hough_r(sM,W,{W,H},{1,1},thr,lM,rb,maxl,cM,bM,ctx());
  } else {
    sRst=nppiFilterHoughLine_8u32f_C1R_Ctx(sR,W,{W,H},{1,1},thr,lR,maxl,cR,bR,ctx());
    sMst=mine_hough(sM,W,{W,H},{1,1},thr,lM,maxl,cM,bM,ctx());
  }
  cudaDeviceSynchronize();
  int nR=0,nM=0;cudaMemcpy(&nR,cR,4,cudaMemcpyDeviceToHost);cudaMemcpy(&nM,cM,4,cudaMemcpyDeviceToHost);
  std::vector<NppPointPolar> LR(std::max(1,maxl)),LM(std::max(1,maxl));
  cudaMemcpy(LR.data(),lR,LR.size()*sizeof(NppPointPolar),cudaMemcpyDeviceToHost);
  cudaMemcpy(LM.data(),lM,LM.size()*sizeof(NppPointPolar),cudaMemcpyDeviceToHost);
  LR.resize(std::min(nR,maxl));LM.resize(std::min(nM,maxl));
  std::sort(LR.begin(),LR.end(),cmpLine);std::sort(LM.begin(),LM.end(),cmpLine);
  int bad=0; for(size_t i=0;i<LR.size()&&i<LM.size();++i) if(fabsf(LR[i].rho-LM[i].rho)>1e-4f||fabsf(LR[i].theta-LM[i].theta)>1e-4f)++bad;
  if(LR.size()!=LM.size())bad+=100;
  printf("%-26s st=%d/%d count=%d/%d mismatch=%d",tag,sRst,sMst,nR,nM,bad);
  if(bad){printf(" | NPP:");for(auto&l:LR)printf(" (%.2f,%.2f)",l.rho,l.theta);printf(" | MINE:");for(auto&l:LM)printf(" (%.2f,%.2f)",l.rho,l.theta);}
  printf("\n");
  if(sRst!=sMst||bad)++gFail;
  cudaFree(sR);cudaFree(bR);cudaFree(lR);cudaFree(cR);cudaFree(sM);cudaFree(bM);cudaFree(lM);cudaFree(cM);
}
int main(){
  const int W=64,H=64;
  std::vector<Npp8u> h(W*H,0),v(W*H,0),d(W*H,0),two(W*H,0),rect(W*H,0),one(W*H,0);
  for(int x=0;x<W;++x)h[20*W+x]=255;
  for(int y=0;y<H;++y)v[y*W+20]=255;
  for(int i=0;i<W;++i)d[i*W+i]=255;
  for(int x=0;x<W;++x){two[20*W+x]=255;two[40*W+x]=255;}
  for(int y=10;y<=50;++y)for(int x=10;x<=50;++x)if(y==10||y==50||x==10||x==50)rect[y*W+x]=255;
  one[20*W+30]=255;
  run("horizontal",h,W,H,50,16);
  run("vertical",v,W,H,50,16);
  run("diagonal",d,W,H,50,16);
  run("single",one,W,H,1,16);
  run("two lines thr50",two,W,H,50,16);
  run("two lines thr60",two,W,H,60,16);
  run("rect thr50",rect,W,H,50,16);
  run("rect thr50 region",rect,W,H,50,16,true,{0,0},{64,91});
  run("horiz region out",h,W,H,50,16,true,{0,100},{64,200});
  printf("FAILURES=%d\n",gFail);
  return gFail?1:0;
}
