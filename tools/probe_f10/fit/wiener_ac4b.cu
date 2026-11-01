// wiener_ac4b: AC4 window shape + model hypotheses.
#include <npp.h>
extern "C" { cudaStream_t nppGetStream(void); NppStatus nppGetStreamContext(NppStreamContext*); }
#include <nppi_filtering_functions.h>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>
static NppStreamContext ctx(){NppStreamContext c{}; nppGetStreamContext(&c); return c;}
static unsigned st=4242;
static float rnd(){st=st*1103515245u+12345u; return float((st>>16)%200);}
static void model_rect(const unsigned char*src,int W,int H,const float*nz,const float*sc,unsigned char*dst){
  const int mw=3,mh=3,ax=1,ay=1;
  for(int y=0;y<H;++y)for(int x=0;x<W;++x)for(int c=0;c<3;++c){
    float sum=0,vsum=0;
    for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){
      int yy=y+dy;if(yy<0)yy=0;if(yy>H-1)yy=H-1;int xx=x+dx;if(xx<0)xx=0;if(xx>W-1)xx=W-1;
      sum+=src[((size_t)yy*W+xx)*4+c];}
    float N=9.f,mu=sum/N;
    for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){
      int yy=y+dy;if(yy<0)yy=0;if(yy>H-1)yy=H-1;int xx=x+dx;if(xx<0)xx=0;if(xx>W-1)xx=W-1;
      float v=src[((size_t)yy*W+xx)*4+c]; vsum+=v*v-mu*mu;}
    float V=vsum/N; float s=src[((size_t)y*W+x)*4+c];
    float ne=nz[c]*sc[c]; float o;
    if(nz[c]==0)o=V; else if(V>ne)o=mu+(V-ne)/V*(s-mu); else o=mu;
    if(o<0)o=0; if(o>255)o=255; dst[((size_t)y*W+x)*4+c]=(unsigned char)o;
  }
}
int main(){
  const int W=17,H=17; NppStreamContext c=ctx();
  unsigned char *s,*d; cudaMalloc(&s,(size_t)W*H*4); cudaMalloc(&d,(size_t)W*H*4);
  // impulse support
  std::vector<unsigned char> im((size_t)W*H*4,0),o((size_t)W*H*4,0x77);
  const int P=8;
  for(size_t i=0;i<im.size();i+=4){im[i]=120;im[i+1]=120;im[i+2]=120;im[i+3]=50;}
  im[((size_t)P*W+P)*4+0]=200;
  float n0[4]={0,0,0,0};
  cudaMemcpy(s,im.data(),im.size(),cudaMemcpyHostToDevice);
  std::vector<unsigned char> mk((size_t)W*H*4,0x77);
  cudaMemcpy(d,mk.data(),mk.size(),cudaMemcpyHostToDevice);
  NppStatus stt=nppiFilterWienerBorder_8u_AC4R_Ctx(s,W*4,NppiSize{W,H},NppiPoint{0,0},d,W*4,NppiSize{W,H},
    NppiSize{3,3},NppiPoint{1,1},n0,NPP_BORDER_REPLICATE,c);
  cudaDeviceSynchronize();cudaMemcpy(o.data(),d,o.size(),cudaMemcpyDeviceToHost);
  printf("ac4 n0 imp st=%d supp offsets (ch0!=0):",(int)stt);
  for(int dy=-3;dy<=3;++dy)for(int dx=-3;dx<=3;++dx){int x=P+dx,y=P+dy;if(o[((size_t)y*W+x)*4]!=0)printf(" (%d,%d)=%d",dx,dy,o[((size_t)y*W+x)*4]);}
  printf("\n");
  // random data model comparison
  std::vector<unsigned char> r((size_t)W*H*4);
  for(size_t i=0;i<r.size();i+=4){r[i]=(unsigned char)rnd();r[i+1]=(unsigned char)rnd();r[i+2]=(unsigned char)rnd();r[i+3]=(unsigned char)rnd();}
  cudaMemcpy(s,r.data(),r.size(),cudaMemcpyHostToDevice);
  float nz[4]={0.02f,0.5f,0.1f,0.03f};
  cudaMemcpy(d,mk.data(),mk.size(),cudaMemcpyHostToDevice);
  stt=nppiFilterWienerBorder_8u_AC4R_Ctx(s,W*4,NppiSize{W,H},NppiPoint{0,0},d,W*4,NppiSize{W,H},
    NppiSize{3,3},NppiPoint{1,1},nz,NPP_BORDER_REPLICATE,c);
  cudaDeviceSynchronize();cudaMemcpy(o.data(),d,o.size(),cudaMemcpyDeviceToHost);
  struct HH{const char*t;float n[4];float sc[4];} hs[]={
    {"rect per-ch s1",{nz[0],nz[1],nz[2],0},{1,1,1,1}},
    {"rect per-ch s255",{nz[0],nz[1],nz[2],0},{65025,65025,65025,1}},
    {"rect n0 s1",{nz[0],nz[0],nz[0],0},{1,1,1,1}},
    {"rect n0 s255",{nz[0],nz[0],nz[0],0},{65025,65025,65025,1}},
  };
  for(auto&hh:hs){
    std::vector<unsigned char> m(r.size(),0x77);
    model_rect(r.data(),W,H,hh.n,hh.sc,m.data());
    int bad=0;double mx=0;
    for(size_t i=0;i<o.size();++i){if(i%4==3)continue;double e=fabs((double)o[i]-(double)m[i]);if(e>mx)mx=e;if(e>0)++bad;}
    printf("  %-18s bad=%d maxd=%.4g\n",hh.t,bad,mx);
  }
  return 0;
}
