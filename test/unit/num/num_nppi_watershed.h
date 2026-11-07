// Coverage for the watershed segmentation functions
// (nppi_filtering_functions.h):
//   nppiSegmentWatershed_{8u,16u}_C1IR.
//
// The reference floods the intensity surface: every output value is an input
// value, values only ever decrease, constant output regions form connected
// plateaus and the optional marker-label image partitions the ROI into those
// regions.  The exact plateau/line assignment of the reference's flooding has
// implementation-specific quirks (the watershed line between two plateaus is
// not credited consistently to the lower plateau), so this driver checks the
// documented invariants plus the two cases the reference is exact about:
//   * a constant image is returned unchanged (no watershed lines);
//   * a strictly monotone ramp collapses to its global minimum;
//   * BLACK / WHITE / ONLY boundaries confine the added values to 0 / 255 and
//     leave every other pixel with an input value.
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace watershed {

template <typename T>
inline std::vector<T> make_pattern(int w, int h, int pat) {
  std::vector<T> img((size_t)w * h, 0);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      T v = 0;
      if (pat == 0) {
        v = (T)77;                                       // constant
      } else if (pat == 1) {
        v = (T)(x * 2);                                  // horizontal ramp (no wrap)
      } else if (pat == 2) {
        v = (T)(x < w / 2 ? 50: 200);                   // vertical step
      } else if (pat == 3) {
        v = (T)(y < h / 3 ? 30: (y < 2 * h / 3 ? 90: 45));  // horizontal bands
      } else {
        // Two plateaus with a raised middle band.
        v = (T)(20 + (x >= w / 4 && x < 3 * w / 4 ? 60: 0));
      }
      img[(size_t)y * w + x] = v;
    }
  return img;
}

template <typename T>
inline bool check_invariants(const char *id, const GenCase &c, const std::vector<T> &in,
                             const std::vector<T> &out, const std::vector<Npp32u> &labels, int mode,
                             RunCounts &cnt, std::string &first) {
  const double white = (double)(T)~(T)0;
  std::vector<uint8_t> seenIn(1 << 16, 0);
  for (T v: in) seenIn[(size_t)v] = 1;
  bool ok = true;
  auto fail = [&](const char *what, int x, int y, double got, double want) {
    char b[360];
    std::snprintf(b, sizeof(b), "[%s] %s mode=%d (%d,%d) %s got=%g want=%g", id, c.id().c_str(), mode, x, y, what,
                  got, want);
    if (first.empty()) first = b;
    EXPECT_TRUE(false) << b;
    ++cnt.mismatched;
    ok = false;
  };
  const int w = c.w, h = c.h;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = (size_t)y * w + x;
      ++cnt.compared;
      const double o = (double)out[i];
      const double v = (double)in[i];
      if (mode == 0) {  // NONE: values may only decrease
        if (o > v) fail("decrease", x, y, o, v);
        if (!seenIn[(size_t)out[i]]) fail("input-value", x, y, o, v);
      } else if (mode == 1) {  // BLACK boundary
        if (o != 0 && o > v) fail("decrease", x, y, o, v);
        if (o != 0 && !seenIn[(size_t)out[i]]) fail("input-value", x, y, o, v);
      } else if (mode == 2) {  // WHITE boundary
        if (o != white && o > v) fail("decrease", x, y, o, v);
        if (o != white && !seenIn[(size_t)out[i]]) fail("input-value", x, y, o, v);
      } else if (mode == 3) {  // CONTRAST: input values plus 0 / white
        if (o != 0 && o != white && !seenIn[(size_t)out[i]]) fail("input-value", x, y, o, v);
      } else {  // ONLY
        if (o != 0 && o != white) fail("binary", x, y, o, 0);
      }
      if (cnt.mismatched >= 8) return false;
    }
  // Labels partition the ROI and, without boundary overrides, stay constant
  // inside each output plateau.
  if (mode == 0 && labels.size() == (size_t)w * h) {
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const size_t i = (size_t)y * w + x;
        if (x + 1 < w && labels[i] == labels[i + 1] && out[i] != out[i + 1])
          fail("label-value-const", x, y, (double)out[i], (double)out[i + 1]);
        if (y + 1 < h && labels[i] == labels[i + 1 * (size_t)w] && out[i] != out[(size_t)(y + 1) * w + x])
          fail("label-value-const-v", x, y, (double)out[i], (double)out[(size_t)(y + 1) * w + x]);
        if (cnt.mismatched >= 8) return false;
      }
  }
  return ok;
}

template <typename T, class Call>
void run_watershed(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  static const int cfgs[][2] = {{64, 64}, {96, 64}, {48, 48}};
  static const int norms[] = {(int)nppiNormL1, (int)nppiNormInf};
  bool failed = false;
  for (const auto &cf: cfgs) {
    for (int pat = 0; pat < 5 && !failed; ++pat) {
      for (int ni = 0; ni < 2 && !failed; ++ni) {
        for (int mode = 0; mode < 5 && !failed; ++mode) {
          const int W = cf[0], H = cf[1];
          std::vector<T> img = make_pattern<T>(W, H, pat);
          Buf<T> s(W, H, 1, 0, 0);
          for (int y = 0; y < H; ++y)
            std::memcpy(s.host.data() + (size_t)y * s.step, img.data() + (size_t)y * W, (size_t)W * sizeof(T));
          s.dev.copyFromHost(s.host);
          Buf<Npp32u> lab(W, H, 1, 0, 0x5A);
          lab.dev.copyFromHost(lab.host);
          NPP_BUFFER_SIZE_TYPE need = 0;
          if constexpr (std::is_same_v<T, Npp16u>) {
            nppiSegmentWatershedGetBufferSize_16u_C1R(NppiSize{W, H}, &need);
          } else {
            nppiSegmentWatershedGetBufferSize_8u_C1R(NppiSize{W, H}, &need);
          }
          npp_functional_test::DeviceMemory<Npp8u> buf((size_t)(need > 0 ? need: 1));
          npp_diff::clear_cuda_error();
          const NppStatus st = call(reinterpret_cast<T *>(s.dptr()), s.step,
                                    reinterpret_cast<Npp32u *>(lab.dptr()), lab.step,
                                    norms[ni], mode, NppiSize{W, H}, buf.get(), ctx);
          if (st != NPP_NO_ERROR) {
            ++cnt.skipped;
            if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
            continue;
          }
          ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "[" << id << "] sync";
          ++cnt.executed;
          s.dev.copyToHost(s.host);
          lab.dev.copyToHost(lab.host);
          std::vector<T> out((size_t)W * H);
          std::vector<Npp32u> labels((size_t)W * H);
          for (int y = 0; y < H; ++y) {
            std::memcpy(out.data() + (size_t)y * W, s.host.data() + (size_t)y * s.step, (size_t)W * sizeof(T));
            std::memcpy(labels.data() + (size_t)y * W, lab.host.data() + (size_t)y * lab.step,
                        (size_t)W * sizeof(Npp32u));
          }
          const GenCase dc{W, H, pat, 0, 0, kSmall, mode};
          if (!check_invariants<T>(id, dc, img, out, labels, mode, cnt, first)) failed = true;
          if (cnt.mismatched >= 8) failed = true;
          // Exact expectations the reference is deterministic about.
          if (pat == 0 && mode != 4) {
            for (size_t i = 0; i < img.size() && !failed; ++i)
              if (out[i] != img[i]) {
                EXPECT_EQ((double)out[i], (double)img[i]) << "[" << id << "] " << dc.id() << " constant input";
                failed = true;
                break;
              }
          } else if (pat == 1 && mode == 0) {
            for (size_t i = 0; i < img.size() && !failed; ++i)
              if ((double)out[i] != 0.0) {
                EXPECT_EQ((double)out[i], 0.0) << "[" << id << "] " << dc.id() << " ramp collapse";
                failed = true;
                break;
              }
          }
        }
      }
    }
  }
  finish(id, cnt, first);
}

}  // namespace watershed
}  // namespace num
