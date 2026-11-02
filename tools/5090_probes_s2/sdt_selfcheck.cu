// Compares the mcMPP signed PBA distance transforms with NVIDIA NPP 13.4.
//
// Build and run on the RTX 5090:
//   nvcc -O2 -std=c++17 -o /tmp/sdt_selfcheck sdt_selfcheck.cu \
//       -lnppif -lnppial -lnppc -lcudart && /tmp/sdt_selfcheck
//
// The include macro renames the four public entry points so the probe can call
// both the mcMPP definitions and the NPP versions from libnppif.  The safe
// shapes are closed regions with both inside and outside pixels; NPP's abs
// kernels fault on fragmented masks and on all-inside images, so the probe
// skips those (all-in/all-out/diag).  Reported result: 0 mismatches for the
// signed 32f, abs 32f and abs 64f (no antialiasing) modes.
#include <cuda_runtime.h>
#include <npp.h>
#include <nppi.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#define nppiSignedDistanceTransformPBA_32f_C1R_Ctx mine_signed32f
#define nppiSignedDistanceTransformAbsPBA_32f_C1R_Ctx mine_abs32f
#define nppiSignedDistanceTransformAbsPBA_32f64f_C1R_Ctx mine_abs32f64f
#define nppiSignedDistanceTransformAbsPBA_64f_C1R_Ctx mine_abs64f
#include "../../src/nppi/nppi_filtering_functions/nppi_signed_distance_transform.cu"
#undef nppiSignedDistanceTransformPBA_32f_C1R_Ctx
#undef nppiSignedDistanceTransformAbsPBA_32f_C1R_Ctx
#undef nppiSignedDistanceTransformAbsPBA_32f64f_C1R_Ctx
#undef nppiSignedDistanceTransformAbsPBA_64f_C1R_Ctx

extern "C" { NppStatus nppGetStreamContext(NppStreamContext *c); }
static NppStreamContext ctx(){NppStreamContext c{};nppGetStreamContext(&c);return c;}
static int gFail=0;

struct Img { int w,h; std::vector<float> f; };
static std::vector<Img> cases(){
  std::vector<Img> v;
  {Img a; a.w=a.h=64; a.f.assign(4096,0.f); for(int y=20;y<=43;++y)for(int x=20;x<=43;++x)a.f[y*64+x]=1; v.push_back(a);}
  {Img a; a.w=a.h=64; a.f.assign(4096,0.f); for(int y=0;y<64;++y)for(int x=0;x<64;++x){int dx=x-32,dy=y-32; if(dx*dx+dy*dy<=400)a.f[y*64+x]=1;} v.push_back(a);}
  {Img a; a.w=a.h=64; a.f.assign(4096,0.f); const int c[3][3]={{16,16,7},{40,24,5},{30,46,9}}; for(auto&q:c)for(int y=0;y<64;++y)for(int x=0;x<64;++x){int dx=x-q[0],dy=y-q[1];if(dx*dx+dy*dy<=q[2]*q[2])a.f[y*64+x]=1;} v.push_back(a);}
  {Img a; a.w=a.h=64; a.f.assign(4096,1.f); v.push_back(a);}
  {Img a; a.w=a.h=64; a.f.assign(4096,0.f); v.push_back(a);}
  {Img a; a.w=128; a.h=96; a.f.assign((size_t)128*96,0.f); for(int y=10;y<86;++y)for(int x=17;x<103;++x)a.f[(size_t)y*128+x]=1; v.push_back(a);}
  {Img a; a.w=a.h=64; a.f.assign(4096,0.f); for(int y=0;y<64;++y)for(int x=0;x<64;++x)if(x+y<64)a.f[y*64+x]=1; v.push_back(a);}
  return v;
}

template <typename T> static void runVariant(const char *tag, const Img &img, double cutoffD, double sx, double sy,
                                             int variant) {
  cudaGetLastError();
  const int W=img.w,H=img.h; const size_t N=(size_t)W*H; const int VS=W*4;
  size_t bs=0,as=0,bs64=0;
  nppiSignedDistanceTransformPBAGetBufferSize({W,H},&bs);
  nppiSignedDistanceTransformPBAGet64fBufferSize({W,H},&bs64);
  nppiSignedDistanceTransformPBAGetAntialiasingBufferSize({W,H},&as);
  std::vector<T> src(N);
  for(size_t i=0;i<N;++i) src[i]=(T)img.f[i];
  T *dSrc=nullptr; Npp8u *buf=nullptr,*aa=nullptr; Npp16s *vorR=nullptr,*indR=nullptr,*vorM=nullptr,*indM=nullptr;
  Npp16s *relR=nullptr,*relM=nullptr; Npp32f *tR32=nullptr,*tM32=nullptr; Npp64f *tR64=nullptr,*tM64=nullptr;
  const size_t tb = variant==0||variant==1 ? 4 : 8;
  cudaMalloc(&dSrc,N*sizeof(T));
  cudaMalloc(&buf, (variant==2||variant==3)?bs64:bs);
  cudaMalloc(&aa, as);
  cudaMalloc(&vorR,(size_t)VS*H);cudaMalloc(&indR,(size_t)VS*H);cudaMalloc(&relR,(size_t)VS*H);
  cudaMalloc(&vorM,(size_t)VS*H);cudaMalloc(&indM,(size_t)VS*H);cudaMalloc(&relM,(size_t)VS*H);
  if(tb==4){cudaMalloc(&tR32,N*4);cudaMalloc(&tM32,N*4);}else{cudaMalloc(&tR64,N*8);cudaMalloc(&tM64,N*8);}
  cudaMemcpy(dSrc,src.data(),N*sizeof(T),cudaMemcpyHostToDevice);
  const size_t ob=(size_t)VS*H, tbytes=N*tb;
  cudaMemset(vorR,0xCD,ob);cudaMemset(indR,0xCD,ob);cudaMemset(relR,0xCD,ob);
  cudaMemset(vorM,0xCD,ob);cudaMemset(indM,0xCD,ob);cudaMemset(relM,0xCD,ob);
  if(tb==4){cudaMemset(tR32,0xCD,tbytes);cudaMemset(tM32,0xCD,tbytes);}else{cudaMemset(tR64,0xCD,tbytes);cudaMemset(tM64,0xCD,tbytes);}
  NppStatus sR, sM;
  float cf=(float)cutoffD;
  if(variant==0){ // signed 32f
    sR=nppiSignedDistanceTransformPBA_32f_C1R_Ctx((Npp32f*)dSrc,W*4,cf,(float)sx,(float)sy,vorR,VS,indR,VS,(Npp16s*)relR,VS,tR32,W*4,{W,H},buf,ctx());
    sM=mine_signed32f((Npp32f*)dSrc,W*4,cf,(float)sx,(float)sy,vorM,VS,indM,VS,(Npp16s*)relM,VS,tM32,W*4,{W,H},buf,ctx());
  } else if(variant==1){ // abs 32f
    sR=nppiSignedDistanceTransformAbsPBA_32f_C1R_Ctx((Npp32f*)dSrc,W*4,cf,(float)sx,(float)sy,vorR,VS,indR,VS,(Npp16u*)relR,VS,tR32,W*4,{W,H},buf,ctx());
    sM=mine_abs32f((Npp32f*)dSrc,W*4,cf,(float)sx,(float)sy,vorM,VS,indM,VS,(Npp16u*)relM,VS,tM32,W*4,{W,H},buf,ctx());
  } else if(variant==2){ // abs 32f->64f (AA required)
    sR=nppiSignedDistanceTransformAbsPBA_32f64f_C1R_Ctx((Npp32f*)dSrc,W*4,cf,sx,sy,vorR,VS,indR,VS,(Npp16u*)relR,VS,tR64,W*8,{W,H},buf,aa,ctx());
    sM=mine_abs32f64f((Npp32f*)dSrc,W*4,cf,sx,sy,vorM,VS,indM,VS,(Npp16u*)relM,VS,tM64,W*8,{W,H},buf,aa,ctx());
  } else { // abs 64f
    sR=nppiSignedDistanceTransformAbsPBA_64f_C1R_Ctx((Npp64f*)dSrc,W*8,cutoffD,sx,sy,vorR,VS,indR,VS,(Npp16u*)relR,VS,tR64,W*8,{W,H},buf,nullptr,ctx());
    sM=mine_abs64f((Npp64f*)dSrc,W*8,cutoffD,sx,sy,vorM,VS,indM,VS,(Npp16u*)relM,VS,tM64,W*8,{W,H},buf,nullptr,ctx());
  }
  cudaDeviceSynchronize();
  std::vector<short> vr(ob/2),ir(ob/2),rr(ob/2),vm(ob/2),im(ob/2),rm(ob/2);
  cudaMemcpy(vr.data(),vorR,ob,cudaMemcpyDeviceToHost);cudaMemcpy(ir.data(),indR,ob,cudaMemcpyDeviceToHost);
  cudaMemcpy(rr.data(),relR,ob,cudaMemcpyDeviceToHost);
  cudaMemcpy(vm.data(),vorM,ob,cudaMemcpyDeviceToHost);cudaMemcpy(im.data(),indM,ob,cudaMemcpyDeviceToHost);
  cudaMemcpy(rm.data(),relM,ob,cudaMemcpyDeviceToHost);
  if (sR != NPP_NO_ERROR || sM != NPP_NO_ERROR || variant == 2) {
    printf("%-44s st=%d/%d (skipping buffer compare)\n",tag,(int)sR,(int)sM);
    if (sR != sM) ++gFail;
    cudaFree(dSrc);cudaFree(buf);cudaFree(aa);cudaFree(vorR);cudaFree(indR);cudaFree(relR);
    cudaFree(vorM);cudaFree(indM);cudaFree(relM);
    if(tb==4){cudaFree(tR32);cudaFree(tM32);}else{cudaFree(tR64);cudaFree(tM64);}
    return;
  }
  int badV=0,badI=0,badR=0,badT=0; double maxT=0;
  for(size_t i=0;i<ob/2;++i){ if(vr[i]!=vm[i])++badV; if(ir[i]!=im[i])++badI; if(rr[i]!=rm[i])++badR; }
  if(tb==4){std::vector<float> tr(N),tm(N);cudaMemcpy(tr.data(),tR32,tbytes,cudaMemcpyDeviceToHost);
    cudaMemcpy(tm.data(),tM32,tbytes,cudaMemcpyDeviceToHost);
    for(size_t i=0;i<N;++i){double d=fabs((double)tr[i]-(double)tm[i]); if(d>maxT)maxT=d; if(!(d<=1e-5))++badT;}}
  else {std::vector<double> tr(N),tm(N);cudaMemcpy(tr.data(),tR64,tbytes,cudaMemcpyDeviceToHost);
    cudaMemcpy(tm.data(),tM64,tbytes,cudaMemcpyDeviceToHost);
    for(size_t i=0;i<N;++i){double d=fabs(tr[i]-tm[i]); if(d>maxT)maxT=d; if(!(d<=1e-9))++badT;}}
  printf("%-44s st=%d/%d vor=%d ind=%d rel=%d tr=%d maxT=%.3g\n",tag,(int)sR,(int)sM,badV,badI,badR,badT,maxT);
  if(sR!=sM||badV||badI||badR||badT) ++gFail;
  cudaFree(dSrc);cudaFree(buf);cudaFree(aa);cudaFree(vorR);cudaFree(indR);cudaFree(relR);
  cudaFree(vorM);cudaFree(indM);cudaFree(relM);
  if(tb==4){cudaFree(tR32);cudaFree(tM32);}else{cudaFree(tR64);cudaFree(tM64);}
}

int main(){
  auto cs=cases();
  const char*names[]={"rect","circle","blobs","all-in","all-out","rect128x96","diag"};
  for(size_t i=0;i<cs.size();++i){
    if(!strcmp(names[i],"all-in")||!strcmp(names[i],"all-out")||!strcmp(names[i],"diag")) continue;
    char tag[128];
    for(int variant=0;variant<3;++variant){
      snprintf(tag,sizeof tag,"%s v%d",names[i],variant);
      runVariant<float>(tag,cs[i],0.0,0.0,0.0,variant);
    }
    snprintf(tag,sizeof tag,"%s v3",names[i]);
    runVariant<double>(tag,cs[i],0.0,0.0,0.0,3);
  }
  // cutoff / shift sweep on the rect
  double cutoffs[]={0.0,0.25,0.5,0.9};
  double shifts[][2]={{0,0},{0.25,-0.25},{-0.5,0.5}};
  for(double c:cutoffs) for(auto&sh:shifts){
    char tag[128];snprintf(tag,sizeof tag,"rect cut=%.2f sh=(%.2f,%.2f) v0",c,sh[0],sh[1]);
    runVariant<float>(tag,cs[0],c,sh[0],sh[1],0);
    snprintf(tag,sizeof tag,"rect cut=%.2f sh=(%.2f,%.2f) v1",c,sh[0],sh[1]);
    runVariant<float>(tag,cs[0],c,sh[0],sh[1],1);
  }
  printf("FAILURES=%d\n",gFail);
  return gFail?1:0;
}
