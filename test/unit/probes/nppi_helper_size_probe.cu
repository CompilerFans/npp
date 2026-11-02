// Probe the host-side NPP helper contracts that are still missing:
//   nppiGetFilterGaussPyramidLayer{Down,Up}BorderDstROI
//   nppiFilterBoxBorderAdvancedGetDeviceBufferSize
//   nppiFilterHarrisCornersBorderGetBufferSize
//   nppiFilterHoughLineGetBufferSize
//   nppiHistogramOfGradientsBorderGet{BufferSize,DescriptorsSize}
//   nppiSegmentWatershedGetBufferSize_16u_C1R
//   nppiSignedDistanceTransformPBAGetAntialiasingBufferSize
#include <cstdio>
#include <cstddef>

#include <nppi.h>
#include <nppi_filtering_functions.h>

static void down() {
  const int ws[] = {0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 101};
  const float rates[] = {0.5f, 1.0f, 1.0001f, 1.25f, 1.5f, 2.0f, 2.5f, 3.0f, 4.0f, 10.0f, 10.5f, 0.0f, -1.0f};
  for (int w : ws) {
    for (int h : ws) {
      for (float r : rates) {
        NppiSize d = {-1, -1};
        NppStatus st = nppiGetFilterGaussPyramidLayerDownBorderDstROI(w, h, &d, r);
        printf("DOWN %d %d %.9g %d %d %d\n", w, h, r, (int)st, d.width, d.height);
      }
    }
  }
}

static void up() {
  const int ws[] = {0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 101};
  const float rates[] = {0.5f, 1.0f, 1.0001f, 1.25f, 1.5f, 2.0f, 2.5f, 3.0f, 4.0f, 10.0f, 10.5f, 0.0f, -1.0f};
  for (int w : ws) {
    for (int h : ws) {
      for (float r : rates) {
        NppiSize mn = {-1, -1}, mx = {-1, -1};
        NppStatus st = nppiGetFilterGaussPyramidLayerUpBorderDstROI(w, h, &mn, &mx, r);
        printf("UP %d %d %.9g %d %d %d %d %d\n", w, h, r, (int)st, mn.width, mn.height, mx.width, mx.height);
      }
    }
  }
}

static void box() {
  const int ws[] = {0, 1, 2, 7, 8, 17, 32, 33, 64, 100};
  const int chs[] = {0, 1, 3, 4, 5};
  for (int w : ws) {
    for (int h : ws) {
      for (int c : chs) {
        int bs = -1;
        NppStatus st = nppiFilterBoxBorderAdvancedGetDeviceBufferSize({w, h}, c, &bs);
        printf("BOX %d %d %d %d %d\n", w, h, c, (int)st, bs);
      }
    }
  }
}

static void harris() {
  const int ws[] = {0, 1, 2, 3, 7, 8, 17, 32, 33, 64, 100, 1000};
  for (int w : ws) {
    for (int h : ws) {
      int bs = -1;
      NppStatus st = nppiFilterHarrisCornersBorderGetBufferSize({w, h}, &bs);
      printf("HARRIS %d %d %d %d\n", w, h, (int)st, bs);
    }
  }
}

static void hough() {
  const int ws[] = {0, 1, 2, 7, 8, 17, 32, 33, 64, 100};
  const NppPointPolar deltas[] = {{1.0f, 1.0f}, {0.5f, 0.5f}, {2.9f, 2.9f}, {0.0f, 1.0f}, {3.0f, 1.0f}, {1.0f, 3.0f}};
  const int counts[] = {0, 1, 10, 100};
  for (int w : ws) {
    for (int h : ws) {
      for (const NppPointPolar &d : deltas) {
        for (int n : counts) {
          int bs = -1;
          NppStatus st = nppiFilterHoughLineGetBufferSize({w, h}, d, n, &bs);
          printf("HOUGH %d %d %.9g %.9g %d %d %d\n", w, h, d.rho, d.theta, n, (int)st, bs);
        }
      }
    }
  }
}

static void hog() {
  struct Cfg {
    int cell, block, bins, w, h;
  };
  const Cfg cfgs[] = {{8, 16, 9, 64, 128}, {4, 8, 9, 32, 64}, {2, 4, 9, 16, 32},
                      {8, 16, 9, 128, 128}, {8, 16, 9, 64, 64}, {1, 2, 9, 16, 32},
                      {8, 16, 1, 64, 128}, {8, 24, 9, 64, 128}, {0, 16, 9, 64, 128}};
  for (const Cfg &c : cfgs) {
    NppiHOGConfig cfg{c.cell, c.block, c.bins, {c.w, c.h}};
    for (int nloc : {0, 1, 2, 128, 129}) {
      int ds = -1;
      NppStatus st = nppiHistogramOfGradientsBorderGetDescriptorsSize(cfg, nloc, &ds);
      printf("HOGDESC %d %d %d %d %d %d %d %d\n", c.cell, c.block, c.bins, c.w, c.h, nloc, (int)st, ds);
    }
    NppiPoint locs[2] = {{0, 0}, {10, 10}};
    for (int nloc : {0, 1, 2}) {
      NppiSize roi = {c.w, c.h};
      int bs = -1;
      NppStatus st = nppiHistogramOfGradientsBorderGetBufferSize(cfg, locs, nloc, roi, &bs);
      printf("HOGBUF %d %d %d %d %d %d %d\n", c.cell, c.block, c.bins, c.w, c.h, nloc, (int)st);
      printf("HOGBUFVAL %d\n", bs);
    }
  }
}

static void watershed() {
  const int ws[] = {0, 1, 2, 7, 8, 17, 32, 33, 64, 100, 1000};
  for (int w : ws) {
    for (int h : ws) {
      size_t bs = (size_t)-1;
      NppStatus st = nppiSegmentWatershedGetBufferSize_16u_C1R({w, h}, &bs);
      printf("WATERSHED %d %d %d %llu\n", w, h, (int)st, (unsigned long long)bs);
    }
  }
}

static void pbaa() {
  const int ws[] = {0, 1, 2, 7, 8, 17, 32, 33, 63, 64, 65, 100, 101, 256};
  for (int w : ws) {
    for (int h : ws) {
      size_t bs = (size_t)-1;
      NppStatus st = nppiSignedDistanceTransformPBAGetAntialiasingBufferSize({w, h}, &bs);
      printf("PBAAA %d %d %d %llu\n", w, h, (int)st, (unsigned long long)bs);
    }
  }
}

int main() {
  down();
  up();
  box();
  harris();
  hough();
  hog();
  watershed();
  pbaa();
  return 0;
}
