// CPU-golden coverage for the UF connected-region label generators
// (nppi_filtering_functions.h):
//   nppiLabelMarkersUF_{8u,16u,32u}32u_C1R and the plain/Advanced batch
//   entry points.
//
// Measured NPP 13.4 semantics (RTX 5090):
//   * every pixel belongs to the connected region of equal-valued pixels
//     (4-way for nppiNormL1, 8-way for nppiNormInf); there is no special
//     treatment of zero-valued pixels;
//   * the label stored for a region is the linear ROI index y*width + x of
//     the region's first pixel in raster order (so a region whose top-left
//     pixel is (0,0) is labelled 0, and labels are otherwise sparse);
//   * the index uses the ROI width even when the source or destination pitch
//     is larger; the batch flavours use each image's own ROI width;
//   * the plain batch applies its oSizeROI to every image, the Advanced batch
//     uses each descriptor's oSize.
#pragma once

#include "num_nppi_geom.h"

#include <memory>

namespace num {
namespace labelmarkers {

// Raster-first component labels for an image (see the file comment).
template <typename Src>
inline std::vector<Npp32u> golden_labels(const std::vector<Src> &img, int w, int h, bool eight) {
  std::vector<Npp32u> lab((size_t)w * h, 0);
  std::vector<uint8_t> seen((size_t)w * h, 0);
  std::vector<int> q;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const int s = y * w + x;
      if (seen[s]) continue;
      const Src v = img[s];
      const Npp32u label = (Npp32u)s;
      seen[s] = 1;
      lab[s] = label;
      q.clear();
      q.push_back(s);
      for (size_t qi = 0; qi < q.size(); ++qi) {
        const int cx = q[qi] % w, cy = q[qi] / w;
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            if (!eight && dx != 0 && dy != 0) continue;
            const int nx = cx + dx, ny = cy + dy;
            if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
            const int ns = ny * w + nx;
            if (seen[ns] || img[ns] != v) continue;
            seen[ns] = 1;
            lab[ns] = label;
            q.push_back(ns);
          }
      }
    }
  return lab;
}

// Source patterns.  Coordinates stay inside the smallest ROI used (48x48) and
// no region starts at (0,0), so label 0 is not produced.
template <typename Src>
inline std::vector<Src> make_pattern(int w, int h, int pat) {
  std::vector<Src> img((size_t)w * h, (Src)0);
  auto put = [&](int x0, int y0, int x1, int y1, Src v) {
    for (int y = y0; y < y1 && y < h; ++y)
      for (int x = x0; x < x1 && x < w; ++x) img[(size_t)y * w + x] = v;
  };
  if (pat == 0) {
    // Three separate blocks: two sharing a value, one different.
    put(8, 8, 12, 12, (Src)5);
    put(20, 8, 24, 12, (Src)5);
    put(w / 2, h / 2, w / 2 + 4, h / 2 + 4, (Src)9);
  } else if (pat == 1) {
    // Diagonal chain: one region for 8-way connectivity, three for 4-way.
    put(8, 8, 12, 12, (Src)5);
    put(12, 12, 16, 16, (Src)5);
    put(16, 16, 20, 20, (Src)5);
  } else if (pat == 2) {
    // Large block with an interior background hole.
    put(8, 8, w - 8, h - 8, (Src)5);
    put(w / 2 - 1, h / 2 - 1, w / 2 + 2, h / 2 + 2, (Src)0);
  } else {
    // L-shaped region and a separate rectangle of another value.
    put(8, 8, 12, h / 2, (Src)3);
    put(8, h / 2, w / 2, h / 2 + 4, (Src)3);
    put(w / 2 + 4, 8, w / 2 + 8, 12, (Src)7);
  }
  return img;
}

inline bool check_labels(const char *id, const GenCase &c, const std::vector<Npp32u> &got,
                         const std::vector<Npp32u> &want, int w, int h, RunCounts &cnt,
                         std::string &first) {
  bool ok = true;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = (size_t)y * w + x;
      ++cnt.compared;
      if (got[i] == want[i]) continue;
      char b[320];
      std::snprintf(b, sizeof(b), "[%s] %s (%d,%d) got=%u want=%u", id, c.id().c_str(), x, y, got[i], want[i]);
      if (first.empty()) first = b;
      EXPECT_TRUE(false) << b;
      ++cnt.mismatched;
      ok = false;
      if (cnt.mismatched >= 8) return false;
    }
  return ok;
}

inline std::vector<Npp32u> host_labels(const Buf<Npp32u> &d, int w, int h) {
  std::vector<Npp32u> out((size_t)w * h);
  for (int y = 0; y < h; ++y)
    std::memcpy(out.data() + (size_t)y * w,
                d.host.data() + (size_t)y * d.step, (size_t)w * sizeof(Npp32u));
  return out;
}

template <typename Src, class Call>
void run_label(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  static const int cfgs[][2] = {{64, 64}, {48, 48}, {96, 64}};
  static const int norms[] = {(int)nppiNormL1, (int)nppiNormInf};
  bool failed = false;
  for (const auto &cf : cfgs) {
    for (int pat = 0; pat < 4 && !failed; ++pat) {
      for (int ni = 0; ni < 2 && !failed; ++ni) {
        const int W = cf[0], H = cf[1];
        const bool eight = norms[ni] == (int)nppiNormInf;
        std::vector<Src> img = make_pattern<Src>(W, H, pat);
        Buf<Src> s(W, H, 1, 0, 0);
        for (int y = 0; y < H; ++y)
          std::memcpy(s.host.data() + (size_t)y * s.step, img.data() + (size_t)y * W, (size_t)W * sizeof(Src));
        s.dev.copyFromHost(s.host);
        Buf<Npp32u> d(W, H, 1, 0, 0x5A);
        d.dev.copyFromHost(d.host);
        int need = 0;
        nppiLabelMarkersUFGetBufferSize_32u_C1R(NppiSize{W, H}, &need);
        npp_functional_test::DeviceMemory<Npp8u> buf((size_t)(need > 0 ? need : 1));
        npp_diff::clear_cuda_error();
        const NppStatus st = call(reinterpret_cast<Src *>(s.dptr()), s.step, reinterpret_cast<Npp32u *>(d.dptr()),
                                  d.step, NppiSize{W, H}, norms[ni], buf.get(), ctx);
        if (st != NPP_NO_ERROR) {
          ++cnt.skipped;
          if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
          continue;
        }
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "[" << id << "] sync";
        ++cnt.executed;
        d.dev.copyToHost(d.host);
        if (d.step != W * (int)sizeof(Npp32u)) {
          ADD_FAILURE() << "[" << id << "] unexpected dst step " << d.step;
          failed = true;
          continue;
        }
        const std::vector<Npp32u> got = host_labels(d, W, H);
        const std::vector<Npp32u> want = golden_labels<Src>(img, W, H, eight);
        const GenCase dc{W, H, pat, 0, 0, kSmall, ni};
        if (!check_labels(id, dc, got, want, W, H, cnt, first)) failed = true;
        if (cnt.mismatched >= 8) failed = true;
      }
    }
  }
  finish(id, cnt, first);
}

// Batched label generation.  The plain flavour applies one shared ROI, the
// Advanced flavour uses each descriptor's oSize (different sizes per image).
template <typename Src, bool Advanced, class Call>
void run_label_batch(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  const int nb = 3;
  static const int pats[nb] = {0, 2, 1};
  static const int advSizes[nb][2] = {{64, 64}, {48, 48}, {96, 64}};
  static const int norms[] = {(int)nppiNormL1, (int)nppiNormInf};
  bool failed = false;
  for (int ni = 0; ni < 2 && !failed; ++ni) {
    const bool eight = norms[ni] == (int)nppiNormInf;
    int ws[nb], hs[nb];
    for (int i = 0; i < nb; ++i) {
      ws[i] = Advanced ? advSizes[i][0] : 64;
      hs[i] = Advanced ? advSizes[i][1] : 64;
    }
    std::vector<std::vector<Src>> imgs(nb);
    std::vector<std::unique_ptr<npp_functional_test::DeviceMemory<Src>>> src(nb);
    std::vector<std::unique_ptr<npp_functional_test::DeviceMemory<Npp32u>>> dst(nb);
    std::vector<NppiImageDescriptor> sdesc(nb), ddesc(nb);
    for (int i = 0; i < nb; ++i) {
      imgs[i] = make_pattern<Src>(ws[i], hs[i], pats[i]);
      src[i].reset(new npp_functional_test::DeviceMemory<Src>(imgs[i].size()));
      src[i]->copyFromHost(imgs[i]);
      std::vector<Npp32u> sentinel(imgs[i].size(), 0x5A5A5A5Au);
      dst[i].reset(new npp_functional_test::DeviceMemory<Npp32u>(imgs[i].size()));
      dst[i]->copyFromHost(sentinel);
      sdesc[i].pData = src[i]->get();
      sdesc[i].nStep = ws[i] * (int)sizeof(Src);
      sdesc[i].oSize = NppiSize{ws[i], hs[i]};
      ddesc[i].pData = dst[i]->get();
      ddesc[i].nStep = ws[i] * (int)sizeof(Npp32u);
      ddesc[i].oSize = NppiSize{ws[i], hs[i]};
    }
    npp_functional_test::DeviceMemory<NppiImageDescriptor> sdev(nb), ddev(nb);
    sdev.copyFromHost(sdesc);
    ddev.copyFromHost(ddesc);
    const NppiSize roi = Advanced ? NppiSize{96, 64} : NppiSize{64, 64};
    npp_diff::clear_cuda_error();
    const NppStatus st = call(sdev.get(), ddev.get(), nb, roi, norms[ni], ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "[" << id << "] sync";
    ++cnt.executed;
    for (int i = 0; i < nb; ++i) {
      const int W = ws[i], H = hs[i];
      std::vector<Npp32u> got(imgs[i].size());
      cudaMemcpy(got.data(), dst[i]->get(), got.size() * sizeof(Npp32u), cudaMemcpyDeviceToHost);
      const std::vector<Npp32u> want = golden_labels<Src>(imgs[i], W, H, eight);
      const GenCase dc{W, H, pats[i], 0, 0, kSmall, ni * 10 + i};
      if (!check_labels(id, dc, got, want, W, H, cnt, first)) failed = true;
    }
    if (cnt.mismatched >= 8) failed = true;
  }
  finish(id, cnt, first);
}

}  // namespace labelmarkers
}  // namespace num
