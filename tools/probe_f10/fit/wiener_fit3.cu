// wiener_fit3: validate 16s / 16u / C3 models of nppiFilterWienerBorder.
#include <npp.h>
extern "C" { cudaStream_t nppGetStream(void); NppStatus nppGetStreamContext(NppStreamContext*); }
#include <nppi_filtering_functions.h>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <type_traits>
#include <vector>
static NppStreamContext ctx(){NppStreamContext c{}; nppGetStreamContext(&c); return c;}
struct Wc{int w,h,ax,ay;};
template <typename T>
static float dmax();
template <> float dmax<Npp8u>(){return 255.f;}
template <> float dmax<Npp16u>(){return 65535.f;}
template <> float dmax<Npp16s>(){return 32767.f;}
template <typename T>
static void model(const T*src,int W,int H,Wc m,const float*noise,int nch,T*dst){
  const int R=m.h>1?m.h-1:1;
  for(int y=0;y<H;++y)for(int x=0;x<W;++x)for(int ch=0;ch<nch;++ch){
    float sum=0,sumsq=0; int K=0;
    if(m.h==1){
      for(int dx=m.ax-(m.w-1);dx<=m.ax;++dx){
        int xx=x+dx; if(xx<0)xx=0; if(xx>W-1)xx=W-1;
        int yy=y+m.ay; if(yy<0)yy=0; if(yy>H-1)yy=H-1;
        sum+=float(src[((size_t)yy*W+xx)*nch+ch]);
      }
      int by=y+m.ay-1; if(by<0)by=0; if(by>H-1)by=H-1;
      for(int dx=m.ax-(m.w-1);dx<=m.ax;++dx){
        int xx=x+dx; if(xx<0)xx=0; if(xx>W-1)xx=W-1;
        float v=float(src[((size_t)by*W+xx)*nch+ch]); sumsq+=v*v; ++K;
      }
    } else {
      for(int r=0;r<R;++r){
        int dy=m.ay-r; int x0=(r==0)?m.ax-(m.w-1):m.ax-(m.w-1)+1;
        for(int dx=x0;dx<=m.ax;++dx){
          int yy=y+dy; if(yy<0)yy=0; if(yy>H-1)yy=H-1;
          int xx=x+dx; if(xx<0)xx=0; if(xx>W-1)xx=W-1;
          float v=float(src[((size_t)yy*W+xx)*nch+ch]); sum+=v; sumsq+=v*v; ++K;
        }
      }
    }
    float N=float(m.w*m.h); float rN=1.f/N; float mu=sum*rN;
    float vsum=0.f;
    if(m.h==1){
      int by=y+m.ay-1; if(by<0)by=0; if(by>H-1)by=H-1;
      for(int dx=m.ax-(m.w-1);dx<=m.ax;++dx){
        int xx=x+dx; if(xx<0)xx=0; if(xx>W-1)xx=W-1;
        float v=float(src[((size_t)by*W+xx)*nch+ch]); vsum += std::fma(v,v,-mu*mu);
      }
    } else {
      const int R2=m.h-1;
      for(int r=0;r<R2;++r){
        int dy=m.ay-r; int x0=(r==0)?m.ax-(m.w-1):m.ax-(m.w-1)+1;
        for(int dx=x0;dx<=m.ax;++dx){
          int yy=y+dy; if(yy<0)yy=0; if(yy>H-1)yy=H-1;
          int xx=x+dx; if(xx<0)xx=0; if(xx>W-1)xx=W-1;
          float v=float(src[((size_t)yy*W+xx)*nch+ch]); vsum += std::fma(v,v,-mu*mu);
        }
      }
    }
    float V=vsum*rN;
    float s=float(src[((size_t)y*W+x)*nch+ch]); float o; float ne=noise[ch]*dmax<T>()*dmax<T>();
    if(noise[ch]==0.f) o=V;
    else if(V>ne) o=mu+(V-ne)/V*(s-mu);
    else o=mu;
    float hi=dmax<T>(); float lo=std::is_same_v<T,Npp16s>?-dmax<T>()-1.f:0.f;
    if(o>hi)o=hi; if(o<lo)o=lo;
    dst[((size_t)y*W+x)*nch+ch]=(T)o;
  }
}
int main(){
  const int W=23,H=19; NppStreamContext c=ctx();
  struct IC{Wc m;float n;const char*t;} ics[]={
    {{3,3,1,1},1.f,"m33n1"},{{3,3,1,1},0.1f,"m33n01"},{{3,3,1,1},0.f,"m33n0"},
    {{5,5,2,2},0.5f,"m55n05"},{{4,4,1,1},0.25f,"m44n025"},{{3,1,1,0},1.f,"m31n1"},
    {{7,7,3,3},0.1f,"m77n01"},{{2,2,0,0},0.f,"m22n0"}};
  unsigned st=99;
  // ---- 16s C1 ----
  {
    short *s,*d; cudaMalloc(&s,(size_t)W*H*2); cudaMalloc(&d,(size_t)W*H*2);
    std::vector<short> im((size_t)W*H),o((size_t)W*H),m((size_t)W*H);
    for(size_t i=0;i<im.size();++i){st=st*1103515245u+12345u; im[i]=(short)((int)((st>>16)%65536)-32768);}
    cudaMemcpy(s,im.data(),im.size()*2,cudaMemcpyHostToDevice);
    for(auto&cc:ics){
      float nz[4]={cc.n,cc.n,cc.n,cc.n}; std::vector<short> mk((size_t)W*H,-12345);
      cudaMemcpy(d,mk.data(),mk.size()*2,cudaMemcpyHostToDevice);
      nppiFilterWienerBorder_16s_C1R_Ctx(s,W*2,NppiSize{W,H},NppiPoint{0,0},d,W*2,NppiSize{W,H},
        NppiSize{cc.m.w,cc.m.h},NppiPoint{cc.m.ax,cc.m.ay},nz,NPP_BORDER_REPLICATE,c);
      cudaDeviceSynchronize(); cudaMemcpy(o.data(),d,o.size()*2,cudaMemcpyDeviceToHost);
      model<short>(im.data(),W,H,cc.m,nz,1,m.data());
      int bad=0; double mx=0; for(size_t i=0;i<o.size();++i){double e=fabs((double)o[i]-(double)m[i]); if(e>mx)mx=e; if(e>0)++bad;}
      printf("16s %-9s bad=%d/%zu maxd=%.5g\n",cc.t,bad,o.size(),mx);
    }
    cudaFree(s); cudaFree(d);
  }
  // ---- 8u C3 with per-channel noise ----
  {
    const int W3=23;
    unsigned char *s,*d; cudaMalloc(&s,(size_t)W3*H*3); cudaMalloc(&d,(size_t)W3*H*3);
    std::vector<unsigned char> im((size_t)W3*H*3),o((size_t)W3*H*3),m((size_t)W3*H*3);
    for(size_t i=0;i<im.size();++i){st=st*1103515245u+12345u; im[i]=(unsigned char)((st>>16)&0xFF);}
    cudaMemcpy(s,im.data(),im.size(),cudaMemcpyHostToDevice);
    float nz[3]={0.5f,0.1f,1.f};
    std::vector<unsigned char> mk((size_t)W3*H*3,0x77);
    cudaMemcpy(d,mk.data(),mk.size(),cudaMemcpyHostToDevice);
    NppStatus stt=nppiFilterWienerBorder_8u_C3R_Ctx(s,W3*3,NppiSize{W3,H},NppiPoint{0,0},d,W3*3,NppiSize{W3,H},
      NppiSize{3,3},NppiPoint{1,1},nz,NPP_BORDER_REPLICATE,c);
    cudaDeviceSynchronize(); cudaMemcpy(o.data(),d,o.size(),cudaMemcpyDeviceToHost);
    model<unsigned char>(im.data(),W3,H,Wc{3,3,1,1},nz,3,m.data());
    int bad=0; double mx=0; for(size_t i=0;i<o.size();++i){double e=fabs((double)o[i]-(double)m[i]); if(e>mx)mx=e; if(e>0)++bad;}
    printf("8uC3 st=%d bad=%d/%zu maxd=%.5g\n",(int)stt,bad,o.size(),mx);
    cudaFree(s); cudaFree(d);
  }
  return 0;
}
