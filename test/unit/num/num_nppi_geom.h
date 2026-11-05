#pragma once

// CPU-golden drivers for the nppi geometry transform families.
//
// Measured NPP 13.4 mirror contract:
//  * NPP_HORIZONTAL_AXIS flips top-bottom, NPP_VERTICAL_AXIS flips
//    left-right and NPP_BOTH_AXIS flips both.
//  * AC4 layouts copy only the three colour channels; the destination alpha
//    keeps its pre-call value (in place, the alpha stays at each position and
//    is not swapped).
//  * The in-place entry points require an even ROI width and height.
//  * nppiMirrorBatch operates on a device array of NppiMirrorBatchCXR
//    ({pSrc, nSrcStep, pDst, nDstStep}); the in-place variant uses pDst and
//    ignores pSrc.  The batch size must be greater than one.

#include "num_suite.h"

namespace num {

struct MirrorAxes {
  bool flipX;
  bool flipY;
};

inline MirrorAxes mirror_axes(NppiAxis flip) {
  return {(flip == NPP_VERTICAL_AXIS || flip == NPP_BOTH_AXIS),
          (flip == NPP_HORIZONTAL_AXIS || flip == NPP_BOTH_AXIS)};
}

template <typename S, int C, int A>
inline bool mirror_check(const char *id, const GenCase &c, NppiAxis flip, const std::vector<uint8_t> &pre,
                         const std::vector<uint8_t> &got, const std::vector<uint8_t> &want, int step,
                         RunCounts &cnt, std::string &first) {
  for (int y = 0; y < c.h; ++y)
    for (int x = 0; x < c.w; ++x) {
      const size_t base = off_bytes<S>(c.oy + y, c.ox + x, step, C);
      for (int ch = 0; ch < C; ++ch) {
        const size_t off = base + (size_t)ch * sizeof(S);
        ++cnt.compared;
        bool ok;
        if (ch >= A) {
          ok = std::memcmp(got.data() + off, pre.data() + off, sizeof(S)) == 0;
        } else {
          ok = equal_elem<S>(got.data() + off, want.data() + off, 0.0, 0.0);
        }
        if (!ok) {
          char b[400];
          std::snprintf(b, sizeof(b), "[%s] %s flip=%d (%d,%d,ch=%d) got=%s want=%s", id, c.id().c_str(),
                        (int)flip, x, y, ch, dbg_elem<S>(got.data() + off).c_str(),
                        dbg_elem<S>(want.data() + off).c_str());
          if (first.empty()) first = b;
          EXPECT_TRUE(false) << b;
          ++cnt.mismatched;
          return false;
        }
      }
    }
  return true;
}

template <typename S, int C, int A, bool Inplace, class Call>
void run_mirror(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  const NppiAxis flips[] = {NPP_HORIZONTAL_AXIS, NPP_VERTICAL_AXIS, NPP_BOTH_AXIS};
  for (const GenCase &c : make_gen_cases(0)) {
    if constexpr (Inplace) {
      if ((c.w & 1) || (c.h & 1)) continue;  // the in-place entry points require even dimensions
    }
    Buf<S> s(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    Buf<S> d(c.w + c.ox, c.h + c.oy, C, c.pad, 0x5A);
    const uint64_t seed = case_seed(id, c.w, c.h, s.step, d.step, (int)c.regime, 0);
    fill_regime(s, C, seed, c.regime, 0, false, false, std::is_floating_point_v<S>);
    s.dev.copyFromHost(s.host);
    d.dev.copyFromHost(d.host);
    for (NppiAxis flip : flips) {
      const std::vector<uint8_t> pre = Inplace ? s.host : d.host;
      npp_diff::clear_cuda_error();
      NppStatus st;
      if constexpr (Inplace) {
        st = call(reinterpret_cast<S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, C)), s.step,
                  NppiSize{c.w, c.h}, flip, ctx);
      } else {
        st = call(reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, C)), s.step,
                  reinterpret_cast<S *>(d.dptr() + off_bytes<S>(c.oy, c.ox, d.step, C)), d.step,
                  NppiSize{c.w, c.h}, flip, ctx);
      }
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        continue;
      }
      const cudaError_t se = cudaDeviceSynchronize();
      ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
      ++cnt.executed;
      const MirrorAxes ax = mirror_axes(flip);
      const std::vector<uint8_t> &srcHost = Inplace ? pre : s.host;
      Buf<S> &out = Inplace ? s : d;
      out.dev.copyToHost(out.host);
      const int srcStep = Inplace ? s.step : s.step;
      std::vector<uint8_t> want = pre;  // alpha channels stay at their pre-call values
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x) {
          const int sx = ax.flipX ? (c.w - 1 - x) : x;
          const int sy = ax.flipY ? (c.h - 1 - y) : y;
          const size_t db = off_bytes<S>(c.oy + y, c.ox + x, out.step, C);
          const size_t sb = off_bytes<S>(c.oy + sy, c.ox + sx, srcStep, C);
          for (int ch = 0; ch < A; ++ch) {
            std::memcpy(want.data() + db + (size_t)ch * sizeof(S),
                        srcHost.data() + sb + (size_t)ch * sizeof(S), sizeof(S));
          }
        }
      if (!mirror_check<S, C, A>(id, c, flip, pre, out.host, want, out.step, cnt, first)) {
        if (cnt.mismatched >= 8) break;
      }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename S, int C, int A, bool Inplace, class Call>
void run_mirror_batch(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  const NppiAxis flips[] = {NPP_HORIZONTAL_AXIS, NPP_VERTICAL_AXIS, NPP_BOTH_AXIS};
  struct BatchCfg {
    int w, h, batch, pad;
  };
  static const BatchCfg cfgs[] = {{7, 5, 3, 0}, {33, 17, 2, 0}, {64, 48, 3, 16}, {1, 1, 2, 0}};
  for (const BatchCfg &cfg : cfgs) {
    if constexpr (Inplace) {
      if ((cfg.w & 1) || (cfg.h & 1)) continue;
    }
    const int B = cfg.batch;
    std::vector<Buf<S> *> bufs;
    std::vector<std::vector<uint8_t>> srcHost(B), preHost(B);
    const uint64_t seed = case_seed(id, cfg.w, cfg.h, cfg.pad, B, 0, 0);
    for (int i = 0; i < B; ++i) {
      Buf<S> *a = new Buf<S>(cfg.w, cfg.h, C, cfg.pad, 0);
      Buf<S> *b = new Buf<S>(cfg.w, cfg.h, C, cfg.pad, 0x5A);
      fill_regime(*a, C, seed ^ mix64((uint64_t)i), (i & 1) ? kWide : kSmall, i);
      a->dev.copyFromHost(a->host);
      b->dev.copyFromHost(b->host);
      bufs.push_back(a);
      bufs.push_back(b);
      srcHost[(size_t)i] = a->host;
      preHost[(size_t)i] = b->host;
    }
    for (NppiAxis flip : flips) {
      std::vector<NppiMirrorBatchCXR> h(B);
      for (int i = 0; i < B; ++i) {
        Buf<S> *a = bufs[(size_t)i * 2];
        Buf<S> *b = bufs[(size_t)i * 2 + 1];
        if constexpr (Inplace) {
          h[i] = {nullptr, 0, (void *)b->dptr(), b->step};
        } else {
          h[i] = {(const void *)a->dptr(), a->step, (void *)b->dptr(), b->step};
        }
      }
      Buf<NppiMirrorBatchCXR> dl(B, 1, 1, 0, 0);
      std::memcpy(dl.host.data(), h.data(), sizeof(NppiMirrorBatchCXR) * B);
      dl.dev.copyFromHost(dl.host);
      npp_diff::clear_cuda_error();
      const NppStatus st = call(NppiSize{cfg.w, cfg.h}, flip,
                                reinterpret_cast<NppiMirrorBatchCXR *>(dl.dptr()), B, ctx);
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        continue;
      }
      const cudaError_t se = cudaDeviceSynchronize();
      ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << cfg.w << "x" << cfg.h << " sync: "
                                 << cudaGetErrorString(se);
      ++cnt.executed;
      const MirrorAxes ax = mirror_axes(flip);
      bool failed = false;
      for (int i = 0; i < B && !failed; ++i) {
        Buf<S> *out = bufs[(size_t)i * 2 + 1];
        out->dev.copyToHost(out->host);
        const std::vector<uint8_t> &src = Inplace ? preHost[(size_t)i] : srcHost[(size_t)i];
        std::vector<uint8_t> want = preHost[(size_t)i];
        const int srcStep = Inplace ? out->step : bufs[(size_t)i * 2]->step;
        for (int y = 0; y < cfg.h; ++y)
          for (int x = 0; x < cfg.w; ++x) {
            const int sx = ax.flipX ? (cfg.w - 1 - x) : x;
            const int sy = ax.flipY ? (cfg.h - 1 - y) : y;
            const size_t db = off_bytes<S>(y, x, out->step, C);
            const size_t sb = off_bytes<S>(sy, sx, srcStep, C);
            for (int ch = 0; ch < A; ++ch) {
              std::memcpy(want.data() + db + (size_t)ch * sizeof(S),
                          src.data() + sb + (size_t)ch * sizeof(S), sizeof(S));
            }
          }
        GenCase gc{cfg.w, cfg.h, 0, 0, 0, kSmall, 0};
        if (!mirror_check<S, C, A>(id, gc, flip, preHost[(size_t)i], out->host, want, out->step, cnt, first)) {
          failed = true;
        }
        if (cnt.mismatched >= 8) break;
      }
      if (failed) break;
    }
    for (auto *p : bufs) delete p;
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

}  // namespace num
