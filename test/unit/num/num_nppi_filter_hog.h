// CPU-golden coverage for nppiHistogramOfGradientsBorder
// (nppi_filtering_functions.h).
//
// Measured NPP 13.4 semantics (RTX 5090):
//   * per-pixel gradients are the centered [-1,0,+1] differences, REPLICATE
//     clamped at the source image bounds; the magnitude is sqrt(gx^2+gy^2)
//     and the orientation atan2(gy,gx) in degrees, negative angles +180;
//   * each pixel's magnitude is split linearly between the two nearest
//     angular bins; a bin index equal to nBins (exactly 180 degrees) is
//     dropped;
//   * every 2x2-cell block is normalised by its largest bin value;
//   * descriptor layout is [blockY][cellY][blockX][cellX][bin];
//   * C3 sources use the per-pixel channel with the largest gradient
//     magnitude (first channel wins ties).
//   The reference's inline atan approximation differs from atanf by <= 1 ulp,
//   which is far below the float comparison tolerance for the axis-aligned
//   ramps used here.
//   NPP 13.4 requires a source step of at least 128 bytes for this family
//   (probe-measured; shorter steps return -16), so the driver uses a
//   256-byte pitch.
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace filthog {

inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

template <typename Src, int C, class Call>
void run_hog(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  const int cell = 8, block = 16, bins = 9, win = 32;
  const int W = 48, H = 48;
  const int cellsX = win / cell, cellsY = win / cell;
  const int blocksX = 2 * (win / block) - 1, blocksY = blocksX;
  const int cpb = block / cell;
  const int descCount = blocksX * blocksY * cpb * cpb * bins;
  const int nLoc = 2;
  const NppiPoint locs[2] = {{0, 0}, {16, 16}};
  struct Pat {
    int kind;  // 0 horizontal ramp, 1 vertical ramp
  };
  static const Pat pats[] = {{0}, {1}};
  bool failed = false;
  for (int pi = 0; pi < 2 && !failed; ++pi) {
    // NPP requires a source pitch of at least 128 bytes for this family.
    Buf<Src> *s = new Buf<Src>(W, H, C, 256, 0);
    if (s->step < 128) {
      delete s;
      GTEST_SKIP() << "[" << id << "] pitch too small";
      return;
    }
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        for (int c = 0; c < C; ++c) {
          const int v = pats[pi].kind == 0 ? x * 3 : y * 3;
          s->hptr()[(size_t)y * (s->step / (int)sizeof(Src)) + (size_t)x * C + c] =
              (Src)(v % (std::is_same_v<Src, Npp8u> ? 200 : 4000));
        }
    s->dev.copyFromHost(s->host);
    Buf<Npp32f> *d = new Buf<Npp32f>(descCount, nLoc, 1, 0, 0x5A);
    d->dev.copyFromHost(d->host);
    const std::vector<uint8_t> pre = d->host;
    int need = 0;
    nppiHistogramOfGradientsBorderGetBufferSize(
        NppiHOGConfig{cell, block, bins, NppiSize{win, win}}, locs, nLoc, NppiSize{W, H}, &need);
    npp_functional_test::DeviceMemory<Npp8u> scratch((size_t)(need > 0 ? need : 1));
    NppiHOGConfig cfg{cell, block, bins, NppiSize{win, win}};
    const Src *sp = reinterpret_cast<const Src *>(s->dptr());
    npp_diff::clear_cuda_error();
    const NppStatus st = call(sp, s->step, NppiSize{W, H}, NppiPoint{0, 0}, locs, nLoc,
                              reinterpret_cast<Npp32f *>(d->dptr()), NppiSize{W, H}, cfg,
                              scratch.get(), (int)NPP_BORDER_REPLICATE, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      delete s;
      delete d;
      continue;
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "[" << id << "] sync";
    ++cnt.executed;
    std::vector<uint8_t> want = pre;
    const int srow = s->step / (int)sizeof(Src);
    auto samp = [&](int ix, int iy, int c) -> float {
      ix = clampi(ix, 0, W - 1);
      iy = clampi(iy, 0, H - 1);
      return (float)as_double(s->hptr()[(size_t)iy * srow + (size_t)ix * C + c]);
    };
    for (int li = 0; li < nLoc; ++li) {
      std::vector<float> hist((size_t)cellsX * cellsY * bins, 0.0f);
      for (int cellY = 0; cellY < cellsY; ++cellY)
        for (int cellX = 0; cellX < cellsX; ++cellX) {
          float local[16] = {0};
          for (int y = cellY * cell; y < (cellY + 1) * cell; ++y)
            for (int x = cellX * cell; x < (cellX + 1) * cell; ++x) {
              const int sx = locs[li].x + x, sy = locs[li].y + y;
              float bestMag = -1.0f, bestAngle = 0.0f;
              for (int c = 0; c < C; ++c) {
                const float gx = samp(sx + 1, sy, c) - samp(sx - 1, sy, c);
                const float gy = samp(sx, sy + 1, c) - samp(sx, sy - 1, c);
                const float mag = std::sqrt(gx * gx + gy * gy);
                if (mag > bestMag) {
                  bestMag = mag;
                  float ang = (float)std::atan2((double)gy, (double)gx) * 57.2957763671875f;
                  if (ang < 0.0f) ang += 180.0f;
                  bestAngle = ang;
                }
              }
              if (bestMag <= 0.0f) continue;
              const float pos = bestAngle * ((float)bins / 180.0f);
              const int b0 = (int)std::floor(pos);
              if (b0 >= bins) continue;
              const float frac = pos - (float)b0;
              local[b0] += bestMag * (1.0f - frac);
              if (b0 + 1 < bins) local[b0 + 1] += bestMag * frac;
            }
          const int base = (cellY * cellsX + cellX) * bins;
          for (int b = 0; b < bins; ++b) hist[(size_t)base + b] = local[b];
        }
      std::vector<float> blockMax((size_t)blocksX * blocksY, 1.0f);
      for (int b = 0; b < blocksX * blocksY; ++b) {
        const int by = b / blocksX, bx = b % blocksX;
        float mx = 0.0f;
        for (int cy = 0; cy < cpb; ++cy)
          for (int cx = 0; cx < cpb; ++cx) {
            const int base = ((by + cy) * cellsX + (bx + cx)) * bins;
            for (int bin = 0; bin < bins; ++bin) mx = std::max(mx, hist[(size_t)base + bin]);
          }
        blockMax[(size_t)b] = mx > 0.0f ? mx : 1.0f;
      }
      for (int i = 0; i < descCount; ++i) {
        int rest = i;
        const int bin = rest % bins;
        rest /= bins;
        const int cx = rest % cpb;
        rest /= cpb;
        const int bx = rest % blocksX;
        rest /= blocksX;
        const int cy = rest % cpb;
        const int by = rest / cpb;
        const int cellIndex = (by + cy) * cellsX + (bx + cx);
        const float v = hist[(size_t)cellIndex * bins + (size_t)bin] /
                        blockMax[(size_t)by * blocksX + (size_t)bx];
        const size_t off = (size_t)li * (size_t)descCount * sizeof(Npp32f) + (size_t)i * sizeof(Npp32f);
        std::memcpy(want.data() + off, &v, sizeof(Npp32f));
      }
    }
    d->dev.copyToHost(d->host);
    const GenCase dc{descCount, nLoc, 0, 0, 0, kSmall, pats[pi].kind};
    const bool ok = rotate_check<Npp32f, 1, 1>(id, dc, NPPI_INTER_NN, pre, d->host, want,
                                               d->step, {}, cnt, first);
    delete s;
    delete d;
    if (!ok || cnt.mismatched >= 8) failed = true;
  }
  finish(id, cnt, first);
}

}  // namespace filthog
}  // namespace num
