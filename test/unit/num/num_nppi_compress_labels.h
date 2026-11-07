// CPU-reference output coverage for the UF marker-label renumbering functions
// (nppi_filtering_functions.h):
//   nppiCompressMarkerLabelsUF_32u_C1IR and the plain/Advanced batch
//   entry points.
//
// Measured NPP 13.4 semantics (reference):
//   * the sparse label IDs are renumbered 1..N in ascending order of the old
//     label value (the label generators store the region's raster-first
//     index, so this is the order the regions were first encountered);
//   * label 0 is treated as "no label" and is left untouched;
//   * the single-image call returns pNewNumber = N + 1 in host memory, while
//     the batch calls return the maximum new label N per image in a device
//     list;
//   * the single-image call requires nStartingNumber = ROI width * ROI height
//     and nSrcDstStep = ROI width * sizeof(Npp32u); the batch flavours take
//     per-image buffer descriptors and, for Advanced, per-image ROIs.
#pragma once

#include "num_nppi_geom.h"

#include <algorithm>
#include <map>
#include <memory>

namespace num {
namespace compresslabels {

// Dense renumbering (see the file comment).  newMax receives N.
inline std::vector<Npp32u> golden_compress(const std::vector<Npp32u> &lab, int *newMax) {
  std::vector<Npp32u> uniq;
  for (Npp32u v: lab)
    if (v != 0) uniq.push_back(v);
  std::sort(uniq.begin(), uniq.end());
  uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
  std::map<Npp32u, Npp32u> remap;
  for (size_t i = 0; i < uniq.size(); ++i) remap[uniq[i]] = (Npp32u)(i + 1);
  std::vector<Npp32u> out(lab.size(), 0);
  for (size_t i = 0; i < lab.size(); ++i) out[i] = lab[i] ? remap[lab[i]]: 0;
  if (newMax) *newMax = (int)uniq.size();
  return out;
}

// Source patterns shared with the label-marker driver (plus one whose first
// region starts at (0,0), which keeps label 0 through the compression).
template <typename Src>
inline std::vector<Src> make_pattern(int w, int h, int pat) {
  if (pat == 4) {
    std::vector<Src> img((size_t)w * h, (Src)5);
    for (int y = h / 2 - 1; y < h / 2 + 2; ++y)
      for (int x = w / 2 - 1; x < w / 2 + 2; ++x) img[(size_t)y * w + x] = (Src)0;
    return img;
  }
  std::vector<Src> img((size_t)w * h, (Src)0);
  auto put = [&](int x0, int y0, int x1, int y1, Src v) {
    for (int y = y0; y < y1 && y < h; ++y)
      for (int x = x0; x < x1 && x < w; ++x) img[(size_t)y * w + x] = v;
  };
  if (pat == 0) {
    put(8, 8, 12, 12, (Src)5);
    put(20, 8, 24, 12, (Src)5);
    put(w / 2, h / 2, w / 2 + 4, h / 2 + 4, (Src)9);
  } else if (pat == 1) {
    put(8, 8, 12, 12, (Src)5);
    put(12, 12, 16, 16, (Src)5);
    put(16, 16, 20, 20, (Src)5);
  } else if (pat == 2) {
    put(8, 8, w - 8, h - 8, (Src)5);
    put(w / 2 - 1, h / 2 - 1, w / 2 + 2, h / 2 + 2, (Src)0);
  } else {
    put(8, 8, 12, h / 2, (Src)3);
    put(8, h / 2, w / 2, h / 2 + 4, (Src)3);
    put(w / 2 + 4, 8, w / 2 + 8, 12, (Src)7);
  }
  return img;
}

// Sparse labels as produced by nppiLabelMarkersUF for the given image.
inline std::vector<Npp32u> seed_labels(const std::vector<Npp8u> &img, int w, int h) {
  std::vector<Npp32u> lab((size_t)w * h, 0);
  std::vector<uint8_t> seen((size_t)w * h, 0);
  std::vector<int> q;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const int s = y * w + x;
      if (seen[s]) continue;
      const Npp8u v = img[s];
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
            if (dx != 0 && dy != 0) continue;  // 4-way, as used for the seeds
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

inline bool check_values(const char *id, const GenCase &c, const std::vector<Npp32u> &got,
                         const std::vector<Npp32u> &want, int w, int h, RunCounts &cnt, std::string &first) {
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

template <class Call>
void run_compress(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  static const int cfgs[][2] = {{64, 64}, {48, 48}, {96, 64}};
  bool failed = false;
  for (const auto &cf: cfgs) {
    for (int pat = 0; pat < 5 && !failed; ++pat) {
      const int W = cf[0], H = cf[1];
      const std::vector<Npp8u> img = make_pattern<Npp8u>(W, H, pat);
      const std::vector<Npp32u> seed = seed_labels(img, W, H);
      int newMax = 0;
      const std::vector<Npp32u> want = golden_compress(seed, &newMax);
      Buf<Npp32u> d(W, H, 1, 0, 0x5A);
      for (int y = 0; y < H; ++y)
        std::memcpy(d.host.data() + (size_t)y * d.step, seed.data() + (size_t)y * W,
                    (size_t)W * sizeof(Npp32u));
      d.dev.copyFromHost(d.host);
      int need = 0;
      nppiCompressMarkerLabelsGetBufferSize_32u_C1R(W * H, &need);
      npp_functional_test::DeviceMemory<Npp8u> buf((size_t)(need > 0 ? need: 1));
      int newNumber = 0x5A5A5A5A;
      npp_diff::clear_cuda_error();
      const NppStatus st = call(reinterpret_cast<Npp32u *>(d.dptr()), d.step, NppiSize{W, H}, W * H,
                                &newNumber, buf.get(), ctx);
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        continue;
      }
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "[" << id << "] sync";
      ++cnt.executed;
      d.dev.copyToHost(d.host);
      std::vector<Npp32u> got((size_t)W * H);
      for (int y = 0; y < H; ++y)
        std::memcpy(got.data() + (size_t)y * W, d.host.data() + (size_t)y * d.step,
                    (size_t)W * sizeof(Npp32u));
      const GenCase dc{W, H, pat, 0, 0, kSmall, 0};
      if (!check_values(id, dc, got, want, W, H, cnt, first)) failed = true;
      EXPECT_EQ(newNumber, newMax + 1) << "[" << id << "] " << dc.id() << " pNewNumber";
      if (newNumber != newMax + 1) failed = true;
      if (cnt.mismatched >= 8) failed = true;
    }
  }
  finish(id, cnt, first);
}

template <bool Advanced, class Call>
void run_compress_batch(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  const int nb = 3;
  static const int pats[nb] = {0, 2, 4};
  static const int advSizes[nb][2] = {{64, 64}, {48, 48}, {96, 64}};
  int ws[nb], hs[nb];
  for (int i = 0; i < nb; ++i) {
    ws[i] = Advanced ? advSizes[i][0]: 64;
    hs[i] = Advanced ? advSizes[i][1]: 64;
  }
  bool failed = false;
  std::vector<std::vector<Npp32u>> seeds(nb), wants(nb);
  std::vector<int> newMaxes(nb, 0);
  std::vector<std::unique_ptr<npp_functional_test::DeviceMemory<Npp32u>>> img(nb);
  std::vector<std::unique_ptr<npp_functional_test::DeviceMemory<Npp8u>>> bufs(nb);
  std::vector<NppiImageDescriptor> desc(nb);
  std::vector<NppiBufferDescriptor> bdesc(nb);
  int largest = 0;
  for (int i = 0; i < nb; ++i) {
    const int W = ws[i], H = hs[i];
    const std::vector<Npp8u> src = make_pattern<Npp8u>(W, H, pats[i]);
    seeds[i] = seed_labels(src, W, H);
    wants[i] = golden_compress(seeds[i], &newMaxes[i]);
    img[i].reset(new npp_functional_test::DeviceMemory<Npp32u>(seeds[i].size()));
    img[i]->copyFromHost(seeds[i]);
    int need = 0;
    nppiCompressMarkerLabelsGetBufferSize_32u_C1R(W * H, &need);
    if (need > largest) largest = need;
    bufs[i].reset(new npp_functional_test::DeviceMemory<Npp8u>((size_t)(need > 0 ? need: 1)));
    desc[i].pData = img[i]->get();
    desc[i].nStep = W * (int)sizeof(Npp32u);
    desc[i].oSize = NppiSize{W, H};
    bdesc[i].pData = bufs[i]->get();
    bdesc[i].nBufferSize = need;
  }
  npp_functional_test::DeviceMemory<NppiImageDescriptor> ddev(nb);
  ddev.copyFromHost(desc);
  npp_functional_test::DeviceMemory<NppiBufferDescriptor> bdev(nb);
  bdev.copyFromHost(bdesc);
  npp_functional_test::DeviceMemory<Npp32u> ndev(nb);
  std::vector<Npp32u> nsent(nb, 0x5A5A5A5Au);
  ndev.copyFromHost(nsent);
  const NppiSize roi = Advanced ? NppiSize{96, 64}: NppiSize{64, 64};
  npp_diff::clear_cuda_error();
  const NppStatus st = call(ddev.get(), bdev.get(), (unsigned int *)ndev.get(), nb, roi, largest, ctx);
  if (st != NPP_NO_ERROR) {
    ++cnt.skipped;
    if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
    finish(id, cnt, first);
    return;
  }
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "[" << id << "] sync";
  ++cnt.executed;
  std::vector<Npp32u> nhost(nb);
  cudaMemcpy(nhost.data(), ndev.get(), nb * sizeof(Npp32u), cudaMemcpyDeviceToHost);
  for (int i = 0; i < nb; ++i) {
    const int W = ws[i], H = hs[i];
    std::vector<Npp32u> got(seeds[i].size());
    cudaMemcpy(got.data(), img[i]->get(), got.size() * sizeof(Npp32u), cudaMemcpyDeviceToHost);
    const GenCase dc{W, H, pats[i], 0, 0, kSmall, 0};
    if (!check_values(id, dc, got, wants[i], W, H, cnt, first)) failed = true;
    EXPECT_EQ((int)nhost[i], newMaxes[i]) << "[" << id << "] " << dc.id() << " newMax";
    if ((int)nhost[i] != newMaxes[i]) failed = true;
  }
  if (cnt.mismatched >= 8) failed = true;
  (void)failed;
  finish(id, cnt, first);
}

}  // namespace compresslabels
}  // namespace num
