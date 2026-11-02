// Second probe for the missing NPP host-size helpers: invalid inputs, null
// pointers, and the Hough / Watershed / HOG formulas.
#include <cstdio>
#include <cstddef>

#include <nppi.h>
#include <nppi_filtering_functions.h>

static void box() {
  const int ws[] = {-1, 0, 1, 2, 7};
  const int chs[] = {0, 1, 2, 3, 4, 5, 6, 8};
  for (int w : ws) {
    for (int h : ws) {
      for (int c : chs) {
        int bs = -12345;
        NppStatus st = nppiFilterBoxBorderAdvancedGetDeviceBufferSize({w, h}, c, &bs);
        printf("BOX %d %d %d %d %d\n", w, h, c, (int)st, bs);
      }
    }
  }
}

static void harris() {
  const int ws[] = {-2, -1, 0, 1, 2, 3, 1000};
  for (int w : ws) {
    for (int h : ws) {
      int bs = -12345;
      NppStatus st = nppiFilterHarrisCornersBorderGetBufferSize({w, h}, &bs);
      printf("HARRIS %d %d %d %d\n", w, h, (int)st, bs);
    }
  }
}

static void watershed() {
  const int ws[] = {-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 31, 32, 33, 100};
  for (int w : ws) {
    for (int h : ws) {
      size_t bs = (size_t)-12345;
      NppStatus st = nppiSegmentWatershedGetBufferSize_16u_C1R({w, h}, &bs);
      printf("WATERSHED %d %d %d %llu\n", w, h, (int)st, (unsigned long long)bs);
    }
  }
}

static void pbaa() {
  const int ws[] = {-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 31, 32, 33, 100};
  for (int w : ws) {
    for (int h : ws) {
      size_t bs = (size_t)-12345;
      NppStatus st = nppiSignedDistanceTransformPBAGetAntialiasingBufferSize({w, h}, &bs);
      printf("PBAAA %d %d %d %llu\n", w, h, (int)st, (unsigned long long)bs);
    }
  }
}

static void hough() {
  const int ws[] = {0, 1, 2, 3, 4, 8, 16, 17, 32};
  const float rhos[] = {0.0f, 0.24f, 0.25f, 0.26f, 0.5f, 1.0f, 2.99f, 3.0f, 3.01f};
  const float thetas[] = {0.0f, 0.24f, 0.25f, 0.26f, 0.5f, 1.0f, 2.99f, 3.0f, 3.01f};
  for (int w : ws) {
    for (int h : ws) {
      for (float rho : rhos) {
        for (float theta : thetas) {
          int bs = -12345;
          NppStatus st = nppiFilterHoughLineGetBufferSize({w, h}, {rho, theta}, 1, &bs);
          printf("HOUGH %d %d %.9g %.9g 1 %d %d\n", w, h, rho, theta, (int)st, bs);
        }
      }
    }
  }
  const int counts[] = {0, 1, 2, 10, 1000};
  for (int n : counts) {
    int bs = -12345;
    NppStatus st = nppiFilterHoughLineGetBufferSize({8, 8}, {1.0f, 1.0f}, n, &bs);
    printf("HOUGHN %d %d %d\n", n, (int)st, bs);
  }
}

static void hogbuf() {
  struct Cfg {
    int cell, block, bins, w, h;
  };
  const Cfg cfgs[] = {
      {8, 16, 9, 64, 128}, {4, 8, 9, 32, 64}, {2, 4, 9, 16, 32}, {8, 16, 9, 128, 128},
      {8, 16, 9, 640, 640}, {8, 16, 9, 256, 256}, {8, 32, 9, 64, 128}, {4, 8, 12, 32, 64},
  };
  for (const Cfg &c : cfgs) {
    NppiHOGConfig cfg{c.cell, c.block, c.bins, {c.w, c.h}};
    const NppiPoint locs[] = {{0, 0}, {1, 1}, {100, 100}, {200, 200}, {300, 300}};
    for (int nloc = 0; nloc <= 5; ++nloc) {
      NppiSize roi = {c.w, c.h};
      int bs = -12345;
      NppStatus st = nppiHistogramOfGradientsBorderGetBufferSize(cfg, locs, nloc, roi, &bs);
      printf("HOGBUF %d %d %d %d %d %d %d %d\n", c.cell, c.block, c.bins, c.w, c.h, nloc, (int)st, bs);
    }
    int ds = -12345;
    NppStatus st = nppiHistogramOfGradientsBorderGetDescriptorsSize(cfg, 3, &ds);
    printf("HOGDESC3 %d %d %d %d %d %d %d\n", c.cell, c.block, c.bins, c.w, c.h, (int)st, ds);
  }
}

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  box();
  harris();
  watershed();
  pbaa();
  hough();
  hogbuf();
  return 0;
}
