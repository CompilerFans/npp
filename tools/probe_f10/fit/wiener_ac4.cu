// wiener_ac4: AC4 Wiener vs per-channel / noise[0] hypotheses.
#include <npp.h>
extern "C" { cudaStream_t nppGetStream(void); NppStatus nppGetStreamContext(NppStreamContext*); }
#include <nppi_filtering_functions.h>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>
static NppStreamContext ctx(){NppStreamContext c{}; nppGetStreamContext(&c); return c;}
static unsigned st=777;
static float rnd(){st=st*1103515245u+12345u; return float((st>>16)%200);}
// per-channel model, noise array nz[4], scale per channel given by sc[4], AC4 semantics (3 channels)
static void model(const unsigned char*src,int W,int H,int nch,const float*nz,const float*sc,unsigned char*dst){
  const int mw=3,mh=3,ax=1,ay=1;
  for(int y=0;y<H;++y)for(int x=0;x<W;++x)for(int c=0;c<nch;++c){
    float sum=0,vsum=0; const int rows=mh-1;
    for(int r=0;r<rows;++r){int dy=ay-r;int x0=ax-(mw-1)+(r?1:0);
      for(int dx=x0;dx<=ax;++dx){int yy=y+dy;if(yy<0)yy=0;if(yy>H-1)yy=H-1;int xx=x+dx;if(xx<0)xx=0;if(xx>W-1)xx=W-1;
        sum+=src[((size_t)yy*W+xx)*4+c];}}
    float N=9.f,mu=sum/N;
    for(int r=0;r<rows;++r){int dy=ay-r;int x0=ax-(mw-1)+(r?1:0);
      for(int dx=x0;dx<=ax;++dx){int yy=y+dy;if(yy<0)yy=0;if(yy>H-1)yy=H-1;int xx=x+dx;if(xx<0)xx=0;if(xx>W-1)xx=W-1;
        float v=src[((size_t)yy*W+xx)*4+c]; vsum+=v*v-mu*mu;}}
    float V=vsum/N; float s=src[((size_t)y*W+x)*4+c];
    float ne=nz[c]*sc[c]; float o;
    if(nz[c]==0)o=V; else if(V>ne)o=mu+(V-ne)/V*(s-mu); else o=mu;
    if(o<0)o=0; if(o>255)o=255; dst[((size_t)y*W+x)*4+c]=(unsigned char)o;
  }
}
int main(){
  const int W=9,H=9; NppStreamContext c=ctx();
  unsigned char *s,*d; cudaMalloc(&s,(size_t)W*H*4); cudaMalloc(&d,(size_t)W*H*4);
  std::vector<unsigned char> im((size_t)W*H*4),o((size_t)W*H*4),m((size_t)W*H*4);
  for(size_t i=0;i<im.size();i+=4){im[i]=(unsigned char)rnd();im[i+1]=(unsigned char)rnd();im[i+2]=(unsigned char)rnd();im[i+3]=(unsigned char)rnd();}
  cudaMemcpy(s,im.data(),im.size(),cudaMemcpyHostToDevice);
  float nz[4]={0.02f,0.5f,0.1f,0.03f};
  std::vector<unsigned char> mk((size_t)W*H*4,0x77);
  cudaMemcpy(d,mk.data(),mk.size(),cudaMemcpyHostToDevice);
  NppStatus stt=nppiFilterWienerBorder_8u_AC4R_Ctx(s,W*4,NppiSize{W,H},NppiPoint{0,0},d,W*4,NppiSize{W,H},
    NppiSize{3,3},NppiPoint{1,1},nz,NPP_BORDER_REPLICATE,c);
  cudaDeviceSynchronize(); cudaMemcpy(o.data(),d,o.size(),cudaMemcpyDeviceToHost);
  struct H{const char*t;float n[4];float s[4];} hs[]={
    {"per-ch scale1",{nz[0],nz[1],nz[2],0},{1,1,1,1}},
    {"per-ch scale255",{nz[0],nz[1],nz[2],0},{65025,65025,65025,1}},
    {"n0 all scale1",{nz[0],nz[0],nz[0],0},{1,1,1,1}},
    {"n0 all scale255",{nz[0],nz[0],nz[0],0},{65025,65025,65025,1}},
    {"n3 all scale1",{nz[3],nz[3],nz[3],0},{1,1,1,1}},
    {"n3 all scale255",{nz[3],nz[3],nz[3],0},{65025,65025,65025,1}},
  };
  printf("ac4 st=%d\n",(int)stt);
  for(auto&hh:hs){
    model(im.data(),W,H,3,hh.n,hh.s,m.data());
    int bad=0;double mx=0;
    for(size_t i=0;i<o.size();++i){ if(i%4==3) continue; double e=fabs((double)o[i]-(double)m[i]); if(e>mx)mx=e; if(e>0)++bad;}
    printf("  %-18s bad=%d maxd=%.4g\n",hh.t,bad,mx);
    (void)hh;
  }
  // also C3 reference on the same RGB for comparison with AC4
  unsigned char *d3; cudaMalloc(&d3,(size_t)W*H*4);
  cudaMemcpy(d3,mk.data(),mk.size(),cudaMemcpyHostToDevice);
  stt=nppiFilterWienerBorder_8u_C3R_Ctx(s,W*4,NppiSize{W,H},NppiPoint{0,0},d3,W*4,NppiSize{W,H},
    NppiSize{3,3},NppiPoint{1,1},nz,NPP_BORDER_REPLICATE,c);
  cudaDeviceSynchronize();
  std::vector<unsigned char> o3((size_t)W*H*4,0x77);
  cudaMemcpy(o3.data(),d3,o3.size(),cudaMemcpyDeviceToHost);
  int bad=0;double mx=0;
  for(int y=0;y<H;++y)for(int x=0;x<W;++x)for(int ch=0;ch<3;++ch){
    int i=(y*W+x)*4+ch; double e=fabs((double)o[i]-(double)o3[i]); if(e>mx)mx=e; if(e>0)++bad;}
  printf("  ac4-vs-c3         bad=%d maxd=%.4g\n",bad,mx);
  return 0;
}
