// Shared plain-C interface for the f10 executor validation harness.
#pragma once
extern "C" {
int ours_canon(int status);
int npp_canon(int status);
int ours_wiener(const void *src, int sstep, int srcW, int srcH, int offX, int offY, void *dst, int dstep,
                int W, int H, int mw, int mh, int ax, int ay, const float *noise, int border, int dtype,
                int channels, int ac4);
int npp_wiener(const void *src, int sstep, int srcW, int srcH, int offX, int offY, void *dst, int dstep,
               int W, int H, int mw, int mh, int ax, int ay, const float *noise, int border, int dtype,
               int channels, int ac4);
int ours_gauss(const void *src, int sstep, void *dst, int dstep, int W, int H, int mask, int dtype,
               int channels, int ac4);
int npp_gauss(const void *src, int sstep, void *dst, int dstep, int W, int H, int mask, int dtype,
              int channels, int ac4);
int ours_gauss_border(const void *src, int sstep, int srcW, int srcH, int offX, int offY, void *dst,
                      int dstep, int W, int H, int mask, int border, int dtype, int channels, int ac4);
int npp_gauss_border(const void *src, int sstep, int srcW, int srcH, int offX, int offY, void *dst,
                     int dstep, int W, int H, int mask, int border, int dtype, int channels, int ac4);
int ours_adv(const void *src, int sstep, void *dst, int dstep, int W, int H, int taps, const float *kernel,
             int dtype, int channels, int ac4);
int npp_adv(const void *src, int sstep, void *dst, int dstep, int W, int H, int taps, const float *kernel,
            int dtype, int channels, int ac4);
int ours_adv_border(const void *src, int sstep, int srcW, int srcH, int offX, int offY, void *dst, int dstep,
                    int W, int H, int taps, const float *kernel, int border, int dtype, int channels, int ac4);
int npp_adv_border(const void *src, int sstep, int srcW, int srcH, int offX, int offY, void *dst, int dstep,
                   int W, int H, int taps, const float *kernel, int border, int dtype, int channels, int ac4);
}
