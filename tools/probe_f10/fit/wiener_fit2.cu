// wiener_fit2: pin down (a) the h==1 statistics window and (b) the integer
// noise scaling / emission of nppiFilterWienerBorder.
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

// window variant: rows from dy0, count rows, first row full w, others w-1
static void wmodel_f(const float*src,int W,int H,Wc m,float noise,int nch,float*dst,
                     int rowStartOff /*0: dy=ay.. ; 1: dy=ay+? */){
  const int R = m.h>1 ? m.h-1 : 1;
  for(int y=0;y<H;++y)for(int x=0;x<W;++x)for(int ch=0;ch<nch;++ch){
    float sum=0,sumsq=0; int K=0;
    if(m.h==1){
      for(int dx=m.ax-(m.w-1);dx<=m.ax;++dx){
        int xx=x+dx; if(xx<0)xx=0; if(xx>W-1)xx=W-1;
        int yy=y+m.ay+rowStartOff; if(yy<0)yy=0; if(yy>H-1)yy=H-1;
        sum+=src[((size_t)yy*W+xx)*nch+ch];
      }
      int by=y+m.ay-1+rowStartOff; if(by<0)by=0; if(by>H-1)by=H-1;
      for(int dx=m.ax-(m.w-1);dx<=m.ax;++dx){
        int xx=x+dx; if(xx<0)xx=0; if(xx>W-1)xx=W-1;
        float v=src[((size_t)by*W+xx)*nch+ch]; sumsq+=v*v; ++K;
      }
    } else {
      for(int r=0;r<R;++r){
        int dy=m.ay-r+rowStartOff;
        int x0=(r==0)?m.ax-(m.w-1):m.ax-(m.w-1)+1;
        for(int dx=x0;dx<=m.ax;++dx){
          int yy=y+dy; if(yy<0)yy=0; if(yy>H-1)yy=H-1;
          int xx=x+dx; if(xx<0)xx=0; if(xx>W-1)xx=W-1;
          float v=src[((size_t)yy*W+xx)*nch+ch]; sum+=v; sumsq+=v*v; ++K;
        }
      }
    }
    float N=float(m.w*m.h); float mu=sum/N; float V=(sumsq-mu*mu*float(K))/N;
    float s=src[((size_t)y*W+x)*nch+ch]; float o;
    if(noise==0.f) o=V;
    else if(V>noise) o=mu+(V-noise)/V*(s-mu);
    else o=mu;
    dst[((size_t)y*W+x)*nch+ch]=o;
  }
}

template <typename T>
static void wmodel_i(const T*src,int W,int H,Wc m,float noise,int nch,T*dst,float scale){
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
    float N=float(m.w*m.h); float mu=sum/N; float V=(sumsq-mu*mu*float(K))/N;
    float s=float(src[((size_t)y*W+x)*nch+ch]);
    float ne=noise*scale; float o;
    if(noise==0.f) o=V;
    else if(V>ne) o=mu+(V-ne)/V*(s-mu);
    else o=mu;
    float hi=float(std::is_same_v<T,Npp8u>?255:(std::is_same_v<T,Npp16u>?65535:32767));
    float lo=std::is_same_v<T,Npp16s>?-32768.f:0.f;
    if(o>hi)o=hi; if(o<lo)o=lo;
    dst[((size_t)y*W+x)*nch+ch]=(T)o;
  }
}

int main(){
  const int W=23,H=19; NppStreamContext c=ctx();
  float *s,*d; cudaMalloc(&s,(size_t)W*H*4); cudaMalloc(&d,(size_t)W*H*4);
  std::vector<float> img((size_t)W*H),out((size_t)W*H),mod((size_t)W*H);
  unsigned st=4242;
  for(size_t i=0;i<img.size();++i){st=st*1103515245u+12345u; img[i]=(float)((st>>16)%400)-200.f;}
  cudaMemcpy(s,img.data(),img.size()*4,cudaMemcpyHostToDevice);
  float noise[4]={1.f,1.f,1.f,1.f};
  struct C{Wc m;const char*t;} cs[]={{{3,1,1,0},"m31a10"},{{3,1,0,0},"m31a00"},{{3,1,2,0},"m31a20"},
                                     {{1,3,0,1},"m13a01"},{{1,3,0,0},"m13a00"},{{1,5,0,2},"m15a02"}};
  for(auto&cc:cs){
    std::vector<float> mark((size_t)W*H,-1.f);
    cudaMemcpy(d,mark.data(),mark.size()*4,cudaMemcpyHostToDevice);
    NppStatus stt=nppiFilterWienerBorder_32f_C1R_Ctx(s,W*4,NppiSize{W,H},NppiPoint{0,0},d,W*4,NppiSize{W,H},
      NppiSize{cc.m.w,cc.m.h},NppiPoint{cc.m.ax,cc.m.ay},noise,NPP_BORDER_REPLICATE,c);
    cudaDeviceSynchronize();
    cudaMemcpy(out.data(),d,out.size()*4,cudaMemcpyDeviceToHost);
    for(int off=-1;off<=1;++off){
      wmodel_f(img.data(),W,H,cc.m,1.f,1,mod.data(),off);
      double maxd=0; for(size_t i=0;i<out.size();++i)maxd=std::max(maxd,(double)fabs(out[i]-mod[i]));
      printf("32f %-8s st=%d off=%d maxd=%.5g\n",cc.t,(int)stt,off,maxd);
    }
  }
  // 8u random image models
  {
    unsigned char *s8,*d8; cudaMalloc(&s8,(size_t)W*H); cudaMalloc(&d8,(size_t)W*H);
    std::vector<unsigned char> im8((size_t)W*H),o8((size_t)W*H),m8((size_t)W*H);
    for(size_t i=0;i<im8.size();++i){st=st*1103515245u+12345u; im8[i]=(unsigned char)((st>>16)&0xFF);}
    cudaMemcpy(s8,im8.data(),im8.size(),cudaMemcpyHostToDevice);
    struct IC{Wc m;float n;const char*t;} ics[]={{{3,3,1,1},1.f,"m33n1"},{{3,3,1,1},0.5f,"m33n05"},
      {{3,3,1,1},0.1f,"m33n01"},{{3,3,1,1},0.f,"m33n0"},{{5,5,2,2},1.f,"m55n1"},
      {{3,3,0,0},0.1f,"m33a00n01"},{{4,4,1,1},0.25f,"m44n025"},{{2,2,0,0},0.f,"m22n0"},
      {{3,1,1,0},1.f,"m31a10n1"},{{3,1,1,0},0.1f,"m31a10n01"},{{3,1,0,0},0.f,"m31a00n0"},
      {{5,1,2,0},0.5f,"m51a20n05"},{{1,1,0,0},1.f,"m11n1"},{{2,1,0,0},0.f,"m21n0"}};
    for(auto&cc:ics){
      float nz[4]={cc.n,cc.n,cc.n,cc.n};
      std::vector<unsigned char> mark((size_t)W*H,0x77);
      cudaMemcpy(d8,mark.data(),mark.size(),cudaMemcpyHostToDevice);
      NppStatus stt=nppiFilterWienerBorder_8u_C1R_Ctx(s8,W,NppiSize{W,H},NppiPoint{0,0},d8,W,NppiSize{W,H},
        NppiSize{cc.m.w,cc.m.h},NppiPoint{cc.m.ax,cc.m.ay},nz,NPP_BORDER_REPLICATE,c);
      cudaDeviceSynchronize();
      cudaMemcpy(o8.data(),d8,o8.size(),cudaMemcpyDeviceToHost);
      wmodel_i<unsigned char>(im8.data(),W,H,cc.m,cc.n,1,m8.data(),255.f*255.f);
      int bad=0; double maxd=0;
      for(size_t i=0;i<o8.size();++i){double e=fabs((double)o8[i]-(double)m8[i]); if(e>maxd)maxd=e; if(e>0)++bad;}
      printf("8u %-9s st=%d bad=%d/%zu maxd=%.5g\n",cc.t,(int)stt,bad,o8.size(),maxd);
    }
    cudaFree(s8); cudaFree(d8);
  }
  // 16s noise-scale sweep on a constant image
  {
    const int w=9,h=9; short *s16,*d16; cudaMalloc(&s16,(size_t)w*h*2); cudaMalloc(&d16,(size_t)w*h*2);
    std::vector<short> im((size_t)w*h,30000),o((size_t)w*h);
    cudaMemcpy(s16,im.data(),im.size()*2,cudaMemcpyHostToDevice);
    printf("16s const30000 V~%.5g\n", 0.0);
    for(float n=0.2f;n<=0.5f;n+=0.05f){
      float nz[4]={n,n,n,n}; std::vector<short> mk((size_t)w*h,-12345);
      cudaMemcpy(d16,mk.data(),mk.size()*2,cudaMemcpyHostToDevice);
      nppiFilterWienerBorder_16s_C1R_Ctx(s16,w*2,NppiSize{w,h},NppiPoint{0,0},d16,w*2,NppiSize{w,h},
        NppiSize{3,3},NppiPoint{1,1},nz,NPP_BORDER_REPLICATE,c);
      cudaDeviceSynchronize();
      cudaMemcpy(o.data(),d16,o.size()*2,cudaMemcpyDeviceToHost);
      printf("  16s n=%.2f out=%d\n",n,(int)o[40]);
    }
    cudaFree(s16); cudaFree(d16);
  }
  return 0;
}
