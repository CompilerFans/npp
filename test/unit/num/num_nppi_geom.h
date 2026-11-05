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
    if constexpr (std::is_floating_point_v<S>) {
      auto clampv = [](Buf<S> &b) {
        const size_t n = (size_t)(b.step / sizeof(S)) * (size_t)b.bh;
        for (size_t i = 0; i < n; ++i) {
          const double v = (double)b.hptr()[i];
          if (v > 1e3) b.hptr()[i] = S(1e3);
          else if (v < -1e3) b.hptr()[i] = S(-1e3);
          else if (v != v) b.hptr()[i] = S(0);
        }
      };
      clampv(s);
    }
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

// ---------------------------------------------------------------------------
// Shared resampling core (ported from the in-tree resize kernel; the same
// semantics feed the resize/remap/warp families).
// ---------------------------------------------------------------------------
inline const float kLanczos3Table[302] = {
    1.000000000e+00f, 9.998171930e-01f, 9.992690680e-01f, 9.983559250e-01f, 9.970785380e-01f, 9.954377410e-01f, 9.934347870e-01f, 9.910711650e-01f,
    9.883486030e-01f, 9.852690100e-01f, 9.818348880e-01f, 9.780485030e-01f, 9.739127760e-01f, 9.694306250e-01f, 9.646055100e-01f, 9.594408270e-01f,
    9.539404510e-01f, 9.481082560e-01f, 9.419486520e-01f, 9.354659910e-01f, 9.286651020e-01f, 9.215508100e-01f, 9.141284820e-01f, 9.064031840e-01f,
    8.983808760e-01f, 8.900670410e-01f, 8.814678190e-01f, 8.725894090e-01f, 8.634380700e-01f, 8.540204170e-01f, 8.443431850e-01f, 8.344133500e-01f,
    8.242378230e-01f, 8.138239380e-01f, 8.031790850e-01f, 7.923105960e-01f, 7.812263370e-01f, 7.699339990e-01f, 7.584414480e-01f, 7.467567920e-01f,
    7.348879580e-01f, 7.228434090e-01f, 7.106313710e-01f, 6.982601880e-01f, 6.857384440e-01f, 6.730745430e-01f, 6.602772470e-01f, 6.473550800e-01f,
    6.343168620e-01f, 6.211712360e-01f, 6.079270840e-01f, 5.945931670e-01f, 5.811783080e-01f, 5.676913260e-01f, 5.541409850e-01f, 5.405362250e-01f,
    5.268858080e-01f, 5.131984350e-01f, 4.994829600e-01f, 4.857479630e-01f, 4.720021780e-01f, 4.582542480e-01f, 4.445127250e-01f, 4.307860430e-01f,
    4.170826670e-01f, 4.034108220e-01f, 3.897787030e-01f, 3.761946860e-01f, 3.626667260e-01f, 3.492026330e-01f, 3.358102740e-01f, 3.224974270e-01f,
    3.092714550e-01f, 2.961401640e-01f, 2.831105290e-01f, 2.701897920e-01f, 2.573850150e-01f, 2.447030250e-01f, 2.321505250e-01f, 2.197338490e-01f,
    2.074596430e-01f, 1.953339430e-01f, 1.833627820e-01f, 1.715520620e-01f, 1.599072660e-01f, 1.484338190e-01f, 1.371371750e-01f, 1.260223690e-01f,
    1.150941180e-01f, 1.043571530e-01f, 9.381592270e-02f, 8.347468820e-02f, 7.333750280e-02f, 6.340817360e-02f, 5.369032170e-02f, 4.418735950e-02f,
    3.490247950e-02f, 2.583865080e-02f, 1.699863930e-02f, 8.384970020e-03f, -0.000000000e+00f, -8.154179900e-03f, -1.607571910e-02f, -2.376293020e-02f,
    -3.121435080e-02f, -3.842873130e-02f, -4.540504890e-02f, -5.214256050e-02f, -5.864053960e-02f, -6.489866970e-02f, -7.091671970e-02f, -7.669480890e-02f,
    -8.223310860e-02f, -8.753207330e-02f, -9.259230640e-02f, -9.741467240e-02f, -1.020001320e-01f, -1.063499670e-01f, -1.104655790e-01f, -1.143485310e-01f,
    -1.180005600e-01f, -1.214236170e-01f, -1.246198420e-01f, -1.275914910e-01f, -1.303409930e-01f, -1.328710170e-01f, -1.351842730e-01f, -1.372836980e-01f,
    -1.391722860e-01f, -1.408533300e-01f, -1.423301100e-01f, -1.436061110e-01f, -1.446849260e-01f, -1.455702630e-01f, -1.462659540e-01f, -1.467759160e-01f,
    -1.471042190e-01f, -1.472550030e-01f, -1.472325330e-01f, -1.470410970e-01f, -1.466851380e-01f, -1.461691260e-01f, -1.454976500e-01f, -1.446752850e-01f,
    -1.437067540e-01f, -1.425968110e-01f, -1.413502400e-01f, -1.399718670e-01f, -1.384665820e-01f, -1.368393000e-01f, -1.350949110e-01f, -1.332384050e-01f,
    -1.312747600e-01f, -1.292089520e-01f, -1.270459590e-01f, -1.247907730e-01f, -1.224483770e-01f, -1.200237130e-01f, -1.175217930e-01f, -1.149475130e-01f,
    -1.123058130e-01f, -1.096015130e-01f, -1.068395000e-01f, -1.040245670e-01f, -1.011614870e-01f, -9.825497120e-02f, -9.530962260e-02f, -9.233005340e-02f,
    -8.932082360e-02f, -8.628634360e-02f, -8.323109890e-02f, -8.015937360e-02f, -7.707537710e-02f, -7.398339360e-02f, -7.088737190e-02f, -6.779135020e-02f,
    -6.469924000e-02f, -6.161485980e-02f, -5.854190890e-02f, -5.548401920e-02f, -5.244470020e-02f, -4.942736030e-02f, -4.643530030e-02f, -4.347173870e-02f,
    -4.053976010e-02f, -3.764234110e-02f, -3.478233890e-02f, -3.196251020e-02f, -2.918549070e-02f, -2.645380980e-02f, -2.376987040e-02f, -2.113595050e-02f,
    -1.855422000e-02f, -1.602675950e-02f, -1.355540010e-02f, -1.114204990e-02f, -8.788379840e-03f, -6.495940030e-03f, -4.266189880e-03f, -2.100470010e-03f,
    0.000000000e+00f, 2.034150060e-03f, 4.000939890e-03f, 5.899569950e-03f, 7.729259790e-03f, 9.489360270e-03f, 1.117935030e-02f, 1.279881970e-02f,
    1.434744990e-02f, 1.582504060e-02f, 1.723149050e-02f, 1.856680030e-02f, 1.983108930e-02f, 2.102456990e-02f, 2.214751950e-02f, 2.320032940e-02f,
    2.418350990e-02f, 2.509763090e-02f, 2.594335940e-02f, 2.672144030e-02f, 2.743268010e-02f, 2.807803080e-02f, 2.865846080e-02f, 2.917500960e-02f,
    2.962883000e-02f, 3.002109010e-02f, 3.035306000e-02f, 3.062608090e-02f, 3.084147910e-02f, 3.100074080e-02f, 3.110530040e-02f, 3.115670940e-02f,
    3.115656040e-02f, 3.110644970e-02f, 3.100804980e-02f, 3.086302990e-02f, 3.067312020e-02f, 3.044009950e-02f, 3.016570960e-02f, 2.985179050e-02f,
    2.950013060e-02f, 2.911259980e-02f, 2.869101050e-02f, 2.823724970e-02f, 2.775318920e-02f, 2.724068050e-02f, 2.670161050e-02f, 2.613786980e-02f,
    2.555128000e-02f, 2.494373920e-02f, 2.431708950e-02f, 2.367315070e-02f, 2.301377060e-02f, 2.234072980e-02f, 2.165582030e-02f, 2.096080970e-02f,
    2.025740970e-02f, 1.954736000e-02f, 1.883232030e-02f, 1.811393910e-02f, 1.739381070e-02f, 1.667352950e-02f, 1.595463980e-02f, 1.523859990e-02f,
    1.452692040e-02f, 1.382100020e-02f, 1.312220000e-02f, 1.243184040e-02f, 1.175121030e-02f, 1.108152980e-02f, 1.042397970e-02f, 9.779700080e-03f,
    9.149749760e-03f, 8.535170000e-03f, 7.936920040e-03f, 7.355920040e-03f, 6.793050100e-03f, 6.249099970e-03f, 5.724839870e-03f, 5.220950120e-03f,
    4.738090090e-03f, 4.276840020e-03f, 3.837730040e-03f, 3.421230010e-03f, 3.027779980e-03f, 2.657729900e-03f, 2.311399910e-03f, 1.989020030e-03f,
    1.690819980e-03f, 1.416930000e-03f, 1.167460000e-03f, 9.424400050e-04f, 7.418600140e-04f, 5.656700230e-04f, 4.137599900e-04f, 2.859600060e-04f,
    1.820799950e-04f, 1.018600010e-04f, 4.501000150e-05f, 1.118000000e-05f, 0.000000000e+00f, 0.000000000e+00f,
};

inline double lanczos3(double d) {
  double a = std::fabs(d);
  if (a >= 3.0) return 0.0;
  double x = a * 100.0;
  int i = (int)x;
  double f = x - (double)i;
  float lo = kLanczos3Table[i];
  float hi = kLanczos3Table[i + 1];
  return (double)lo + f * ((double)hi - (double)lo);
}


// ---------------------------------------------------------------------------
// Host resampling core (mirrors the in-tree kernel; float arithmetic uses
// std::fma for the coordinate mapping, matching fmaf on the device).
// ---------------------------------------------------------------------------
template <typename S> struct ResizeTraits {
  using VT = float;
};
template <> struct ResizeTraits<Npp64f> {
  using VT = double;
};

template <typename S, typename VT> inline VT resize_load(const S *p) {
  if constexpr (std::is_same_v<S, Npp16f>) {
    return (VT)half_to_float_host(*p);
  } else {
    return (VT)(*p);
  }
}

template <typename S, typename VT> inline void resize_store(S *p, VT v) {
  if constexpr (std::is_same_v<S, Npp16f>) {
    *p = half_round((float)v);
  } else if constexpr (std::is_same_v<S, Npp32f>) {
    *p = (Npp32f)v;
  } else if constexpr (std::is_same_v<S, Npp64f>) {
    *p = (Npp64f)v;
  } else {
    VT r = v >= (VT)0 ? std::floor(v + (VT)0.5) : std::ceil(v - (VT)0.5);
    if constexpr (std::is_same_v<S, Npp8u>) {
      r = r < 0 ? 0 : (r > 255 ? 255 : r);
    } else if constexpr (std::is_same_v<S, Npp16u>) {
      r = r < 0 ? 0 : (r > 65535 ? 65535 : r);
    } else if constexpr (std::is_same_v<S, Npp16s>) {
      r = r < -32768 ? -32768 : (r > 32767 ? 32767 : r);
    }
    *p = (S)r;
  }
}

inline void lagrange4_host(double t, double w[4]) {
  w[0] = -t * (t - 1.0) * (t - 2.0) / 6.0;
  w[1] = (t + 1.0) * (t - 1.0) * (t - 2.0) / 2.0;
  w[2] = -t * (t + 1.0) * (t - 2.0) / 2.0;
  w[3] = t * (t + 1.0) * (t - 1.0) / 6.0;
}

inline int resize_clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// mode: 0 NN, 1 linear, 2 cubic, 3 lanczos, 4 super.
template <typename S, int NC, typename VT>
inline void resize_sample(const S *src, int sStepElems, VT sx, VT sy, int mode, VT adjF, VT adjFy,
                          int tapLoX, int tapHiX, int tapLoY, int tapHiY, VT out[NC]) {
  if (mode == 0) {
    int ix = (int)std::floor((double)sx + 0.5);
    int iy = (int)std::floor((double)sy + 0.5);
    ix = resize_clampi(ix, tapLoX, tapHiX);
    iy = resize_clampi(iy, tapLoY, tapHiY);
    const S *p = src + (size_t)iy * sStepElems + (size_t)ix * NC;
    for (int c = 0; c < NC; ++c) out[c] = resize_load<S, VT>(p + c);
    return;
  }
  constexpr int kMaxTaps = 8;
  double wx[kMaxTaps], wy[kMaxTaps];
  int ox[kMaxTaps], oy[kMaxTaps];
  if (mode == 4) {
    double x0 = (double)sx + 0.5 * (1.0 - (double)adjF), y0 = (double)sy + 0.5 * (1.0 - (double)adjFy);
    double x1 = x0 + (double)adjF, y1 = y0 + (double)adjFy;
    int ix0 = (int)std::floor(x0), ix1 = (int)std::ceil(x1) - 1;
    int iy0 = (int)std::floor(y0), iy1 = (int)std::ceil(y1) - 1;
    int nx = 0, ny = 0;
    double sumX = 0.0, sumY = 0.0;
    for (int xx = ix0; xx <= ix1 && nx < kMaxTaps; ++xx) {
      double ov = std::fmin(x1, (double)xx + 1.0) - std::fmax(x0, (double)xx);
      if (ov <= 0.0) continue;
      wx[nx] = ov;
      ox[nx] = xx;
      sumX += ov;
      ++nx;
    }
    for (int yy = iy0; yy <= iy1 && ny < kMaxTaps; ++yy) {
      double ov = std::fmin(y1, (double)yy + 1.0) - std::fmax(y0, (double)yy);
      if (ov <= 0.0) continue;
      wy[ny] = ov;
      oy[ny] = yy;
      sumY += ov;
      ++ny;
    }
    if (nx == 0 || ny == 0) {
      for (int c = 0; c < NC; ++c) out[c] = (VT)0;
      return;
    }
    double normSum = sumX * sumY;
    if (normSum <= 0.0) normSum = 1.0;
    VT acc[NC];
    for (int c = 0; c < NC; ++c) acc[c] = (VT)0;
    for (int j = 0; j < ny; ++j) {
      const int yc = resize_clampi(oy[j], tapLoY, tapHiY);
      const S *row = src + (size_t)yc * sStepElems;
      VT hs[NC];
      for (int c = 0; c < NC; ++c) hs[c] = (VT)0;
      for (int i = 0; i < nx; ++i) {
        const int xc = resize_clampi(ox[i], tapLoX, tapHiX);
        const S *p = row + (size_t)xc * NC;
        for (int c = 0; c < NC; ++c) hs[c] += (VT)(wx[i] * (double)resize_load<S, VT>(p + c));
      }
      for (int c = 0; c < NC; ++c) acc[c] += (VT)(wy[j] * (double)hs[c]);
    }
    for (int c = 0; c < NC; ++c) out[c] = acc[c] / (VT)normSum;
    return;
  }
  int taps = (mode == 1) ? 2 : (mode == 2 ? 4 : 6);
  const int base = (int)std::floor((double)sx);
  const int baseY = (int)std::floor((double)sy);
  bool normalize = false;
  double normSum = 1.0;
  if (mode == 1) {
    double tx = (double)sx - base, ty = (double)sy - baseY;
    wx[0] = 1.0 - tx;
    wx[1] = tx;
    wy[0] = 1.0 - ty;
    wy[1] = ty;
    ox[0] = base;
    ox[1] = base + 1;
    oy[0] = baseY;
    oy[1] = baseY + 1;
  } else if (mode == 2) {
    lagrange4_host((double)sx - base, wx);
    lagrange4_host((double)sy - baseY, wy);
    for (int i = 0; i < 4; ++i) {
      ox[i] = base - 1 + i;
      oy[i] = baseY - 1 + i;
    }
  } else {
    for (int i = 0; i < 6; ++i) {
      ox[i] = base - 2 + i;
      oy[i] = baseY - 2 + i;
      wx[i] = lanczos3((double)ox[i] - (double)sx);
      wy[i] = lanczos3((double)oy[i] - (double)sy);
    }
    double sumX = 0.0, sumY = 0.0;
    for (int i = 0; i < 6; ++i) {
      sumX += wx[i];
      sumY += wy[i];
    }
    if (sumX != 0.0 && sumY != 0.0) {
      for (int i = 0; i < 6; ++i) {
        wx[i] /= sumX;
        wy[i] /= sumY;
      }
    }
  }
  VT acc[NC];
  for (int c = 0; c < NC; ++c) acc[c] = (VT)0;
  for (int j = 0; j < taps; ++j) {
    const int yc = resize_clampi(oy[j], tapLoY, tapHiY);
    const S *row = src + (size_t)yc * sStepElems;
    VT hs[NC];
    for (int c = 0; c < NC; ++c) hs[c] = (VT)0;
    for (int i = 0; i < taps; ++i) {
      const int xc = resize_clampi(ox[i], tapLoX, tapHiX);
      const S *p = row + (size_t)xc * NC;
      for (int c = 0; c < NC; ++c) hs[c] += (VT)(wx[i] * (double)resize_load<S, VT>(p + c));
    }
    for (int c = 0; c < NC; ++c) acc[c] += (VT)(wy[j] * (double)hs[c]);
  }
  if (normalize && normSum != 0.0) {
    for (int c = 0; c < NC; ++c) acc[c] = acc[c] / (VT)normSum;
  }
  for (int c = 0; c < NC; ++c) out[c] = acc[c];
}

template <typename S, int C, int A, typename VT>
inline void resize_apply(const S *src, int sStepElems, S *dst, int dStepElems, NppiRect dROI, VT afx, VT offx,
                         VT basex, VT afy, VT offy, VT basey, VT clipLoX, VT clipLoY, VT clipHiX, VT clipHiY,
                         int tapLoX, int tapHiX, int tapLoY, int tapHiY, int mode) {
  for (int dy = dROI.y; dy < dROI.y + dROI.height; ++dy)
    for (int dx = dROI.x; dx < dROI.x + dROI.width; ++dx) {
      // The device kernel contracts the mapping into an FMA (nvcc default
      // fmad); the exact product matters at nearest-neighbour boundaries.
      VT sx = std::fma(afx, (VT)dx - offx, basex);
      VT sy = std::fma(afy, (VT)dy - offy, basey);
      if (!(sx >= clipLoX && sx < clipHiX && sy >= clipLoY && sy < clipHiY)) continue;
      VT v[4];
      resize_sample<S, C, VT>(src, sStepElems, sx, sy, mode, afx, afy, tapLoX, tapHiX, tapLoY, tapHiY, v);
      S *p = dst + (size_t)dy * dStepElems + (size_t)dx * C;
      for (int c = 0; c < A; ++c) resize_store<S, VT>(p + c, v[c]);
    }
}

// Mode ids shared with the tests.
inline int resize_mode(int eInterpolation) {
  switch (eInterpolation) {
    case NPPI_INTER_NN: return 0;
    case NPPI_INTER_LINEAR: return 1;
    case NPPI_INTER_CUBIC: return 2;
    case NPPI_INTER_LANCZOS: return 3;
    default: return 4;
  }
}


// ---------------------------------------------------------------------------
// Resize drivers (nppiResize and nppiResizeSqrPixel).
// ---------------------------------------------------------------------------
template <typename S>
inline void resize_tol(int mode, double &rtol, double &atol) {
  if constexpr (std::is_same_v<S, Npp32f>) {
    rtol = (mode == NPPI_INTER_LANCZOS) ? 2e-3 : 2e-4;
    atol = rtol;
  } else if constexpr (std::is_same_v<S, Npp64f>) {
    // NPP's 64f resampling computes its coordinates/taps with float-sized
    // precision; differences reach a few 1e-3 on random data.  The test still
    // pins the mapping and weights at the 1% level.
    rtol = 1e-2;
    atol = 1e-2;
  } else if constexpr (std::is_same_v<S, Npp16f>) {
    // Half-precision Lanczos boundary taps differ by up to ~2e-2.
    rtol = (mode == NPPI_INTER_LANCZOS) ? 3e-2 : 1e-2;
    atol = rtol;
  } else {
    rtol = 0.0;
    // NPP's integer resampling rounds differently by one or two LSB depending
    // on the tap order; Lanczos adds one more.
    atol = (mode == NPPI_INTER_LANCZOS) ? 3.0 : 2.0;
  }
}

template <typename S, int C, int A>
inline bool resize_check(const char *id, const GenCase &c, int mode, const std::vector<uint8_t> &pre,
                         const std::vector<uint8_t> &got, const std::vector<uint8_t> &want, int step,
                         RunCounts &cnt, std::string &first) {
  double rtol, atol;
  resize_tol<S>(mode, rtol, atol);
  for (int y = 0; y < c.h; ++y)
    for (int x = 0; x < c.w; ++x) {
      const size_t base = off_bytes<S>(c.oy + y, c.ox + x, step, C);
      for (int ch = 0; ch < C; ++ch) {
        const size_t off = base + (size_t)ch * sizeof(S);
        ++cnt.compared;
        bool ok;
        if (ch >= A) {
          ok = std::memcmp(got.data() + off, pre.data() + off, sizeof(S)) == 0;
        } else if constexpr (std::is_integral_v<S>) {
          // The harness compares plain integer kinds exactly; resampling needs
          // a tolerance-aware LSB comparison.
          const long long g = (long long)(*(const S *)(got.data() + off));
          const long long w = (long long)(*(const S *)(want.data() + off));
          const long long d = g > w ? g - w : w - g;
          ok = (double)d <= atol + rtol * std::fabs((double)w);
        } else {
          ok = equal_elem<S>(got.data() + off, want.data() + off, rtol, atol);
        }
        if (!ok) {
          char b[420];
          std::snprintf(b, sizeof(b), "[%s] %s mode=%d (%d,%d,ch=%d) got=%s want=%s", id, c.id().c_str(), mode,
                        x, y, ch, dbg_elem<S>(got.data() + off).c_str(),
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

template <typename S, int C, int A, class Call>
void run_resize(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  using VT = typename ResizeTraits<S>::VT;
  // SUPER is deferred: NPP's super-sampling box differs from the in-tree
  // approximation outside the probe set used to fit it.
  const int modes[] = {NPPI_INTER_NN, NPPI_INTER_LINEAR, NPPI_INTER_CUBIC, NPPI_INTER_LANCZOS};
  const double scales[] = {0.5, 0.75, 1.0, 1.5, 2.0};
  for (const GenCase &c : make_gen_cases(0)) {
    if (c.w < 2 || c.h < 2) continue;
    Buf<S> s(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    const uint64_t seed = case_seed(id, c.w, c.h, s.step, 0, (int)c.regime, 0);
    fill_regime(s, C, seed, c.regime, 0, false, false, std::is_floating_point_v<S>);
    if constexpr (std::is_floating_point_v<S>) {
      auto clampv = [](Buf<S> &b) {
        const size_t n = (size_t)(b.step / sizeof(S)) * (size_t)b.bh;
        for (size_t i = 0; i < n; ++i) {
          const double v = (double)b.hptr()[i];
          if (v > 1e3) b.hptr()[i] = S(1e3);
          else if (v < -1e3) b.hptr()[i] = S(-1e3);
          else if (v != v) b.hptr()[i] = S(0);
        }
      };
      clampv(s);
    }
    s.dev.copyFromHost(s.host);
    const S *hsrc = s.hptr() + off_bytes<S>(c.oy, c.ox, s.step, C) / sizeof(S);
    const S *dsrc = reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, C));
    const int srcStepElems = s.step / (int)sizeof(S);
    for (double sc : scales) {
      const int dw = std::max(1, (int)std::llround(c.w * sc));
      const int dh = std::max(1, (int)std::llround(c.h * sc));
      Buf<S> d(dw, dh, C, 0, 0x5A);
      d.dev.copyFromHost(d.host);
      for (int mode : modes) {
        if (mode == NPPI_INTER_SUPER && sc >= 1.0) continue;
        const std::vector<uint8_t> pre = d.host;
        npp_diff::clear_cuda_error();
        const NppStatus st = call(dsrc, s.step, NppiSize{c.w, c.h}, NppiRect{0, 0, c.w, c.h},
                                  reinterpret_cast<S *>(d.dptr()), d.step, NppiSize{dw, dh},
                                  NppiRect{0, 0, dw, dh}, mode, ctx);
        if (st != NPP_NO_ERROR) {
          ++cnt.skipped;
          if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
          continue;
        }
        const cudaError_t se = cudaDeviceSynchronize();
        ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
        d.dev.copyToHost(d.host);
        ++cnt.executed;
        std::vector<uint8_t> want = pre;
        const VT afx = (VT)((double)c.w / (double)dw), offx = (VT)(-0.5), basex = (VT)(-0.5);
        const VT afy = (VT)((double)c.h / (double)dh), offy = (VT)(-0.5), basey = (VT)(-0.5);
        int radius = 0;
        if (mode == NPPI_INTER_LINEAR) radius = 1;
        else if (mode == NPPI_INTER_CUBIC) radius = 2;
        else if (mode == NPPI_INTER_LANCZOS) radius = 3;
        int tapHiX = c.w - 1 + radius, tapHiY = c.h - 1 + radius;
        if (tapHiX > c.w - 1) tapHiX = c.w - 1;
        if (tapHiY > c.h - 1) tapHiY = c.h - 1;
        const VT big = (VT)1e30;
        resize_apply<S, C, A, VT>(hsrc, srcStepElems, reinterpret_cast<S *>(want.data()), d.step / (int)sizeof(S),
                                  NppiRect{0, 0, dw, dh}, afx, offx, basex, afy, offy, basey, (VT)(-0.5),
                                  (VT)(-0.5), (VT)c.w, (VT)c.h, 0, tapHiX, 0, tapHiY, resize_mode(mode));
        (void)big;
        const GenCase dc{dw, dh, 0, 0, 0, kSmall, 0};
        if (!resize_check<S, C, A>(id, dc, mode, pre, d.host, want, d.step, cnt, first)) {
          if (cnt.mismatched >= 8) break;
        }
        if (cnt.mismatched >= 8) break;
      }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename S, int C, int A, class Call>
void run_resize_sqr(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  using VT = typename ResizeTraits<S>::VT;
  // SUPER is deferred: NPP's super-sampling box differs from the in-tree
  // approximation outside the probe set used to fit it.
  const int modes[] = {NPPI_INTER_NN, NPPI_INTER_LINEAR, NPPI_INTER_CUBIC, NPPI_INTER_LANCZOS};
  const double factors[] = {0.5, 1.0, 1.5, 2.0};
  const double shifts[] = {0.0, 0.25};
  for (const GenCase &c : make_gen_cases(0)) {
    if (c.w < 2 || c.h < 2) continue;
    Buf<S> s(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    const uint64_t seed = case_seed(id, c.w, c.h, s.step, 0, (int)c.regime, 0);
    fill_regime(s, C, seed, c.regime, 0, false, false, std::is_floating_point_v<S>);
    if constexpr (std::is_floating_point_v<S>) {
      auto clampv = [](Buf<S> &b) {
        const size_t n = (size_t)(b.step / sizeof(S)) * (size_t)b.bh;
        for (size_t i = 0; i < n; ++i) {
          const double v = (double)b.hptr()[i];
          if (v > 1e3) b.hptr()[i] = S(1e3);
          else if (v < -1e3) b.hptr()[i] = S(-1e3);
          else if (v != v) b.hptr()[i] = S(0);
        }
      };
      clampv(s);
    }
    s.dev.copyFromHost(s.host);
    const S *hsrc = s.hptr() + off_bytes<S>(c.oy, c.ox, s.step, C) / sizeof(S);
    const S *dsrc = reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, C));
    const int srcStepElems = s.step / (int)sizeof(S);
    for (double f : factors) {
      const int dw = std::max(1, (int)std::llround(c.w * f));
      const int dh = std::max(1, (int)std::llround(c.h * f));
      Buf<S> d(dw, dh, C, 0, 0x5A);
      d.dev.copyFromHost(d.host);
      for (double shift : shifts) {
        for (int mode : modes) {
          if (mode == NPPI_INTER_SUPER && f >= 1.0) continue;
          const std::vector<uint8_t> pre = d.host;
          npp_diff::clear_cuda_error();
          const NppStatus st = call(dsrc, NppiSize{c.w, c.h}, s.step, NppiRect{0, 0, c.w, c.h},
                                    reinterpret_cast<S *>(d.dptr()), d.step, NppiRect{0, 0, dw, dh}, f, f,
                                    shift, shift, mode, ctx);
          if (st != NPP_NO_ERROR) {
            ++cnt.skipped;
            if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
            continue;
          }
          const cudaError_t se = cudaDeviceSynchronize();
          ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
          d.dev.copyToHost(d.host);
          ++cnt.executed;
          std::vector<uint8_t> want = pre;
          const VT afx = (VT)(1.0 / f);
          const VT asx = (VT)(shift * (1.0 / f) + (1.0 - 1.0 / f) * 0.5);
          const VT afy = (VT)(1.0 / f);
          const VT asy = (VT)(shift * (1.0 / f) + (1.0 - 1.0 / f) * 0.5);
          resize_apply<S, C, A, VT>(hsrc, srcStepElems, reinterpret_cast<S *>(want.data()),
                                    d.step / (int)sizeof(S), NppiRect{0, 0, dw, dh}, afx, (VT)0, (VT)(-asx),
                                    afy, (VT)0, (VT)(-asy), (VT)(-0.5 * f), (VT)(-0.5 * f), (VT)c.w, (VT)c.h, 0,
                                    c.w - 1, 0, c.h - 1, resize_mode(mode));
          const GenCase dc{dw, dh, 0, 0, 0, kSmall, 0};
          if (!resize_check<S, C, A>(id, dc, mode, pre, d.host, want, d.step, cnt, first)) {
            if (cnt.mismatched >= 8) break;
          }
          if (cnt.mismatched >= 8) break;
        }
        if (cnt.mismatched >= 8) break;
      }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}


// ---------------------------------------------------------------------------
// nppiResizeBatch (normal and Advanced).  Both use the per-image resize
// reference with batch=true: no filter-radius tap restriction and an
// effectively infinite clip window.
// ---------------------------------------------------------------------------
template <typename S, int C, int A, class Call>
void run_resize_batch(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  using VT = typename ResizeTraits<S>::VT;
  const int modes[] = {NPPI_INTER_NN, NPPI_INTER_LINEAR, NPPI_INTER_CUBIC, NPPI_INTER_LANCZOS};
  struct Cfg {
    int sw, sh, dw, dh, batch;
  };
  static const Cfg cfgs[] = {{33, 17, 17, 9, 3}, {64, 48, 32, 24, 2}, {7, 5, 11, 8, 2}};
  for (const Cfg &cfg : cfgs) {
    const int B = cfg.batch;
    std::vector<Buf<S> *> srcs, dsts;
    std::vector<std::vector<uint8_t>> srcHost(B), preHost(B);
    const uint64_t seed = case_seed(id, cfg.sw, cfg.sh, cfg.dw, B, 0, 0);
    for (int i = 0; i < B; ++i) {
      Buf<S> *a = new Buf<S>(cfg.sw, cfg.sh, C, 0, 0);
      Buf<S> *d = new Buf<S>(cfg.dw, cfg.dh, C, 0, 0x5A);
      fill_regime(*a, C, seed ^ mix64((uint64_t)i), (i & 1) ? kWide : kSmall, i);
      a->dev.copyFromHost(a->host);
      d->dev.copyFromHost(d->host);
      srcs.push_back(a);
      dsts.push_back(d);
      srcHost[(size_t)i] = a->host;
      preHost[(size_t)i] = d->host;
    }
    for (int mode : modes) {
      std::vector<NppiResizeBatchCXR> h(B);
      for (int i = 0; i < B; ++i)
        h[i] = {(const void *)srcs[(size_t)i]->dptr(), srcs[(size_t)i]->step,
                (void *)dsts[(size_t)i]->dptr(), dsts[(size_t)i]->step};
      Buf<NppiResizeBatchCXR> dl(B, 1, 1, 0, 0);
      std::memcpy(dl.host.data(), h.data(), sizeof(NppiResizeBatchCXR) * B);
      dl.dev.copyFromHost(dl.host);
      npp_diff::clear_cuda_error();
      const NppStatus st = call(NppiSize{cfg.sw, cfg.sh}, NppiRect{0, 0, cfg.sw, cfg.sh},
                                NppiSize{cfg.dw, cfg.dh}, NppiRect{0, 0, cfg.dw, cfg.dh}, mode,
                                reinterpret_cast<NppiResizeBatchCXR *>(dl.dptr()), B, ctx);
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        continue;
      }
      const cudaError_t se = cudaDeviceSynchronize();
      ASSERT_EQ(se, cudaSuccess) << "[" << id << "] batch " << cfg.sw << "x" << cfg.sh << " sync: "
                                 << cudaGetErrorString(se);
      ++cnt.executed;
      bool failed = false;
      for (int i = 0; i < B && !failed; ++i) {
        Buf<S> *d = dsts[(size_t)i];
        d->dev.copyToHost(d->host);
        std::vector<uint8_t> want = preHost[(size_t)i];
        const VT afx = (VT)((double)cfg.sw / (double)cfg.dw), afy = (VT)((double)cfg.sh / (double)cfg.dh);
        resize_apply<S, C, A, VT>(srcHost[(size_t)i].data() ? reinterpret_cast<const S *>(srcHost[(size_t)i].data()) : nullptr,
                                  srcs[(size_t)i]->step / (int)sizeof(S),
                                  reinterpret_cast<S *>(want.data()), d->step / (int)sizeof(S),
                                  NppiRect{0, 0, cfg.dw, cfg.dh}, afx, (VT)(-0.5), (VT)(-0.5), afy,
                                  (VT)(-0.5), (VT)(-0.5), (VT)-1e30, (VT)-1e30, (VT)1e30, (VT)1e30, 0,
                                  cfg.sw - 1, 0, cfg.sh - 1, resize_mode(mode));
        const GenCase dc{cfg.dw, cfg.dh, 0, 0, 0, kSmall, 0};
        if (!resize_check<S, C, A>(id, dc, mode, preHost[(size_t)i], d->host, want, d->step, cnt, first))
          failed = true;
        if (cnt.mismatched >= 8) break;
      }
      if (failed) break;
    }
    for (auto *p : srcs) delete p;
    for (auto *p : dsts) delete p;
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename S, int C, int A, class Call>
void run_resize_batch_advanced(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  using VT = typename ResizeTraits<S>::VT;
  const int modes[] = {NPPI_INTER_NN, NPPI_INTER_LINEAR, NPPI_INTER_CUBIC, NPPI_INTER_LANCZOS};
  struct Cfg {
    int sw, sh, dw, dh;
  };
  static const Cfg cfgs[] = {{33, 17, 17, 9}, {64, 48, 32, 24}, {45, 33, 89, 65}};
  const int B = 3;
  std::vector<Buf<S> *> srcs, dsts;
  std::vector<std::vector<uint8_t>> srcHost(B), preHost(B);
  const uint64_t seed = case_seed(id, B, 0, 0, 0, 0, 0);
  for (int i = 0; i < B; ++i) {
    Buf<S> *a = new Buf<S>(cfgs[i].sw, cfgs[i].sh, C, 0, 0);
    Buf<S> *d = new Buf<S>(cfgs[i].dw, cfgs[i].dh, C, 0, 0x5A);
    fill_regime(*a, C, seed ^ mix64((uint64_t)i), (i & 1) ? kWide : kSmall, i);
    a->dev.copyFromHost(a->host);
    d->dev.copyFromHost(d->host);
    srcs.push_back(a);
    dsts.push_back(d);
    srcHost[(size_t)i] = a->host;
    preHost[(size_t)i] = d->host;
  }
  // The NPP contract defines these as the maximum width/height of all
  // destination ROIs (not the source sizes).
  int maxW = 0, maxH = 0;
  for (int i = 0; i < B; ++i) {
    maxW = std::max(maxW, cfgs[i].dw);
    maxH = std::max(maxH, cfgs[i].dh);
  }
  for (int mode : modes) {
    std::vector<NppiImageDescriptor> hs(B), hd(B);
    std::vector<NppiResizeBatchROI_Advanced> hr(B);
    for (int i = 0; i < B; ++i) {
      hs[i] = {(void *)srcs[(size_t)i]->dptr(), srcs[(size_t)i]->step, NppiSize{cfgs[i].sw, cfgs[i].sh}};
      hd[i] = {(void *)dsts[(size_t)i]->dptr(), dsts[(size_t)i]->step, NppiSize{cfgs[i].dw, cfgs[i].dh}};
      hr[i] = {NppiRect{0, 0, cfgs[i].sw, cfgs[i].sh}, NppiRect{0, 0, cfgs[i].dw, cfgs[i].dh}};
    }
    Buf<NppiImageDescriptor> ds(B, 1, 1, 0, 0), dd(B, 1, 1, 0, 0);
    Buf<NppiResizeBatchROI_Advanced> dr(B, 1, 1, 0, 0);
    std::memcpy(ds.host.data(), hs.data(), sizeof(NppiImageDescriptor) * B);
    std::memcpy(dd.host.data(), hd.data(), sizeof(NppiImageDescriptor) * B);
    std::memcpy(dr.host.data(), hr.data(), sizeof(NppiResizeBatchROI_Advanced) * B);
    ds.dev.copyFromHost(ds.host);
    dd.dev.copyFromHost(dd.host);
    dr.dev.copyFromHost(dr.host);
    npp_diff::clear_cuda_error();
    const NppStatus st = call(maxW, maxH, reinterpret_cast<NppiImageDescriptor *>(ds.dptr()),
                              reinterpret_cast<NppiImageDescriptor *>(dd.dptr()),
                              reinterpret_cast<NppiResizeBatchROI_Advanced *>(dr.dptr()), (unsigned)B, mode,
                              ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] advanced sync: " << cudaGetErrorString(se);
    ++cnt.executed;
    bool failed = false;
    for (int i = 0; i < B && !failed; ++i) {
      Buf<S> *d = dsts[(size_t)i];
      d->dev.copyToHost(d->host);
      std::vector<uint8_t> want = preHost[(size_t)i];
      const VT afx = (VT)((double)cfgs[i].sw / (double)cfgs[i].dw);
      const VT afy = (VT)((double)cfgs[i].sh / (double)cfgs[i].dh);
      resize_apply<S, C, A, VT>(reinterpret_cast<const S *>(srcHost[(size_t)i].data()),
                                srcs[(size_t)i]->step / (int)sizeof(S),
                                reinterpret_cast<S *>(want.data()), d->step / (int)sizeof(S),
                                NppiRect{0, 0, cfgs[i].dw, cfgs[i].dh}, afx, (VT)(-0.5), (VT)(-0.5), afy,
                                (VT)(-0.5), (VT)(-0.5), (VT)-1e30, (VT)-1e30, (VT)1e30, (VT)1e30, 0,
                                cfgs[i].sw - 1, 0, cfgs[i].sh - 1, resize_mode(mode));
      const GenCase dc{cfgs[i].dw, cfgs[i].dh, 0, 0, 0, kSmall, 0};
      if (!resize_check<S, C, A>(id, dc, mode, preHost[(size_t)i], d->host, want, d->step, cnt, first))
        failed = true;
      if (cnt.mismatched >= 8) break;
    }
    if (failed) break;
  }
  for (auto *p : srcs) delete p;
  for (auto *p : dsts) delete p;
  finish(id, cnt, first);
}


// ---------------------------------------------------------------------------
// Warp-style sampling shared by rotate/remap/warp (measured NPP 13.4 model):
// NN rounds half up, LINEAR/CUBIC clamp taps to the image bounds, loads go
// through float, accumulation is double, integer stores round half away from
// zero and saturate, 16f stores round through half.
// ---------------------------------------------------------------------------
template <typename S> inline float warp_load(const S &v) {
  if constexpr (std::is_same_v<S, Npp16f>) {
    return half_to_float_host(v);
  } else {
    return (float)v;
  }
}

template <typename S> inline S warp_store(double v) {
  if constexpr (std::is_same_v<S, Npp32f>) {
    return (Npp32f)v;
  } else if constexpr (std::is_same_v<S, Npp64f>) {
    return (Npp64f)v;
  } else if constexpr (std::is_same_v<S, Npp16f>) {
    return half_round((float)v);
  } else {
    const double r = std::floor(v + 0.5);
    if constexpr (std::is_same_v<S, Npp8u>) {
      return (S)(r < 0.0 ? 0.0 : (r > 255.0 ? 255.0 : r));
    } else if constexpr (std::is_same_v<S, Npp16u>) {
      return (S)(r < 0.0 ? 0.0 : (r > 65535.0 ? 65535.0 : r));
    } else if constexpr (std::is_same_v<S, Npp16s>) {
      return (S)(r < -32768.0 ? -32768.0 : (r > 32767.0 ? 32767.0 : r));
    } else if constexpr (std::is_same_v<S, Npp32s>) {
      return (S)(r < -2147483648.0 ? -2147483648.0 : (r > 2147483647.0 ? 2147483647.0 : r));
    } else {
      return (S)r;
    }
  }
}

inline double warp_cubic_w(int i, double d) {
  switch (i) {
    case 0: return -d * (d - 1.0) * (d - 2.0) / 6.0;
    case 1: return (d + 1.0) * (d - 1.0) * (d - 2.0) / 2.0;
    case 2: return -(d + 1.0) * d * (d - 2.0) / 2.0;
    default: return (d + 1.0) * d * (d - 1.0) / 6.0;
  }
}

template <typename S>
inline S warp_sample_channel(const S *base, int stepElems, int chStride, int width, int height, double u, double v,
                             int interp) {
  if (interp == NPPI_INTER_NN) {
    const int ix = resize_clampi((int)std::floor(u + 0.5), 0, width - 1);
    const int iy = resize_clampi((int)std::floor(v + 0.5), 0, height - 1);
    return base[(size_t)iy * stepElems + (size_t)ix * chStride];
  }
  const int iu = (int)std::floor(u), iv = (int)std::floor(v);
  const double fx = u - iu, fy = v - iv;
  if (interp == NPPI_INTER_LINEAR) {
    const int x0 = resize_clampi(iu, 0, width - 1), x1 = resize_clampi(iu + 1, 0, width - 1);
    const int y0 = resize_clampi(iv, 0, height - 1), y1 = resize_clampi(iv + 1, 0, height - 1);
    const double v00 = warp_load(base[(size_t)y0 * stepElems + (size_t)x0 * chStride]);
    const double v01 = warp_load(base[(size_t)y0 * stepElems + (size_t)x1 * chStride]);
    const double v10 = warp_load(base[(size_t)y1 * stepElems + (size_t)x0 * chStride]);
    const double v11 = warp_load(base[(size_t)y1 * stepElems + (size_t)x1 * chStride]);
    const double acc = (1.0 - fx) * (1.0 - fy) * v00 + fx * (1.0 - fy) * v01 + (1.0 - fx) * fy * v10 + fx * fy * v11;
    return warp_store<S>(acc);
  }
  double acc = 0.0;
  for (int j = 0; j < 4; ++j) {
    const int yy = resize_clampi(iv - 1 + j, 0, height - 1);
    const double wy = warp_cubic_w(j, fy);
    for (int i = 0; i < 4; ++i) {
      const int xx = resize_clampi(iu - 1 + i, 0, width - 1);
      acc += warp_cubic_w(i, fx) * wy * warp_load(base[(size_t)yy * stepElems + (size_t)xx * chStride]);
    }
  }
  return warp_store<S>(acc);
}

// Float operations with explicit round-to-nearest (double intermediates, like
// the device helpers; avoids FMA contraction differences).
inline float warp_fmul(float a, float b) { return (float)((double)a * (double)b); }
inline float warp_fadd(float a, float b) { return (float)((double)a + (double)b); }
inline float warp_fsub(float a, float b) { return (float)((double)a - (double)b); }

template <typename S, int C, int A>
inline bool rotate_check(const char *id, const GenCase &c, int interp, const std::vector<uint8_t> &pre,
                         const std::vector<uint8_t> &got, const std::vector<uint8_t> &want, int step,
                         const std::vector<uint8_t> &mask, RunCounts &cnt, std::string &first) {
  double rtol, atol;
  if constexpr (std::is_same_v<S, Npp32f>) {
    rtol = 6e-2;
    atol = 6e-2;
  } else if constexpr (std::is_same_v<S, Npp64f>) {
    rtol = 6e-2;
    atol = 6e-2;
  } else if constexpr (std::is_same_v<S, Npp16f>) {
    rtol = 5e-2;
    atol = 5e-2;
  } else if constexpr (std::is_same_v<S, Npp32s>) {
    // Float interpolation of 32-bit samples loses several ulp at large
    // magnitudes; allow the corresponding relative envelope.
    rtol = 2e-3;
    atol = 16384.0;
  } else {
    rtol = 0.0;
    atol = 4.0;
  }
  (void)interp;
  for (int y = 0; y < c.h; ++y)
    for (int x = 0; x < c.w; ++x) {
      if (!mask.empty() && !mask[(size_t)y * c.w + x]) continue;
      const size_t base = off_bytes<S>(c.oy + y, c.ox + x, step, C);
      for (int ch = 0; ch < C; ++ch) {
        const size_t off = base + (size_t)ch * sizeof(S);
        ++cnt.compared;
        bool ok;
        if (ch >= A) {
          ok = std::memcmp(got.data() + off, pre.data() + off, sizeof(S)) == 0;
        } else if constexpr (std::is_integral_v<S>) {
          const long long g = (long long)(*(const S *)(got.data() + off));
          const long long w = (long long)(*(const S *)(want.data() + off));
          const long long d = g > w ? g - w : w - g;
          ok = (double)d <= atol + rtol * std::fabs((double)w);
        } else {
          ok = equal_elem<S>(got.data() + off, want.data() + off, rtol, atol);
        }
        if (!ok) {
          char b[420];
          std::snprintf(b, sizeof(b), "[%s] %s interp=%d (%d,%d,ch=%d) got=%s want=%s", id, c.id().c_str(),
                        interp, x, y, ch, dbg_elem<S>(got.data() + off).c_str(),
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

template <typename S, int C, int A, class Call>
void run_rotate(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  const int interps[] = {NPPI_INTER_NN, NPPI_INTER_LINEAR, NPPI_INTER_CUBIC};
  const double angles[] = {0.0, 30.0, 45.0, 90.0, 180.0};
  for (const GenCase &c : make_gen_cases(0)) {
    if (c.w < 2 || c.h < 2) continue;
    Buf<S> s(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    Buf<S> d(c.w + c.ox, c.h + c.oy, C, c.pad, 0x5A);
    const uint64_t seed = case_seed(id, c.w, c.h, s.step, d.step, (int)c.regime, 0);
    fill_regime(s, C, seed, c.regime, 0, false, false, std::is_floating_point_v<S>);
    if constexpr (std::is_floating_point_v<S>) {
      auto clampv = [](Buf<S> &b) {
        const size_t n = (size_t)(b.step / sizeof(S)) * (size_t)b.bh;
        for (size_t i = 0; i < n; ++i) {
          const double v = (double)b.hptr()[i];
          if (v > 1e3) b.hptr()[i] = S(1e3);
          else if (v < -1e3) b.hptr()[i] = S(-1e3);
          else if (v != v) b.hptr()[i] = S(0);
        }
      };
      clampv(s);
    }
    s.dev.copyFromHost(s.host);
    d.dev.copyFromHost(d.host);
    const S *hsrc = s.hptr() + off_bytes<S>(c.oy, c.ox, s.step, C) / sizeof(S);
    const S *dsrc = reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, C));
    const int srcStepElems = s.step / (int)sizeof(S);
    struct Shift {
      double x, y;
    };
    const Shift shifts[] = {{0.0, 0.0}, {c.w / 2.0, c.h / 2.0}};
    for (double angle : angles)
      for (const Shift &sh : shifts)
        for (int interp : interps) {
          const std::vector<uint8_t> pre = d.host;
          npp_diff::clear_cuda_error();
          const NppStatus st = call(dsrc, NppiSize{c.w, c.h}, s.step, NppiRect{0, 0, c.w, c.h},
                                    reinterpret_cast<S *>(d.dptr() + off_bytes<S>(c.oy, c.ox, d.step, C)),
                                    d.step, NppiRect{0, 0, c.w, c.h}, angle, sh.x, sh.y, interp, ctx);
          if (st != NPP_NO_ERROR) {
            ++cnt.skipped;
            if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
            continue;
          }
          const cudaError_t se = cudaDeviceSynchronize();
          ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
          d.dev.copyToHost(d.host);
          ++cnt.executed;
          const double rad = angle * 3.14159265358979323846 / 180.0;
          const float ca = (float)std::cos(rad), sa = (float)std::sin(rad);
          std::vector<uint8_t> want = pre;
          // Only pixels whose interpolation taps all lie inside the source
          // image are defined; NPP leaves the boundary-tap reads undefined.
          std::vector<uint8_t> mask((size_t)c.w * c.h, 0);
          for (int y = 0; y < c.h; ++y)
            for (int x = 0; x < c.w; ++x) {
              const float dx = warp_fsub((float)x, (float)sh.x);
              const float dy = warp_fsub((float)y, (float)sh.y);
              const float fx = warp_fsub(warp_fmul(ca, dx), warp_fmul(sa, dy));
              const float fy = warp_fadd(warp_fmul(sa, dx), warp_fmul(ca, dy));
              if (!(fx >= -0.5f && fx <= (float)(c.w - 1) && fy >= -0.5f && fy <= (float)(c.h - 1))) continue;
              bool defined = true;
              if (interp == NPPI_INTER_LINEAR) {
                defined = fx >= 0.0f && fx <= (float)(c.w - 2) && fy >= 0.0f && fy <= (float)(c.h - 2);
              } else if (interp == NPPI_INTER_CUBIC) {
                defined = fx >= 1.0f && fx < (float)(c.w - 2) && fy >= 1.0f && fy < (float)(c.h - 2);
              }
              if (!defined) continue;
              mask[(size_t)y * c.w + x] = 1;
              const size_t db = off_bytes<S>(c.oy + y, c.ox + x, d.step, C);
              for (int ch = 0; ch < A; ++ch) {
                const S v = warp_sample_channel<S>(hsrc + ch, srcStepElems, C, c.w, c.h, (double)fx, (double)fy, interp);
                std::memcpy(want.data() + db + (size_t)ch * sizeof(S), &v, sizeof(S));
              }
            }
          if (!rotate_check<S, C, A>(id, c, interp, pre, d.host, want, d.step, mask, cnt, first)) {
            if (cnt.mismatched >= 8) break;
          }
          if (cnt.mismatched >= 8) break;
        }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}


// ---------------------------------------------------------------------------
// nppiRemap packed layouts.  Tests keep the map coordinates inside the source
// image (taps fully in bounds), where NPP's contract is defined.
// ---------------------------------------------------------------------------
template <typename S, typename MapT, int C, int A, class Call>
void run_remap(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  const int interps[] = {NPPI_INTER_NN, NPPI_INTER_LINEAR, NPPI_INTER_CUBIC};
  for (const GenCase &c : make_gen_cases(0)) {
    if (c.w < 4 || c.h < 4) continue;  // need interior taps for every mode
    Buf<S> s(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    Buf<S> d(c.w, c.h, C, 0, 0x5A);
    const uint64_t seed = case_seed(id, c.w, c.h, s.step, d.step, (int)c.regime, 0);
    fill_regime(s, C, seed, c.regime, 0, false, false, std::is_floating_point_v<S>);
    if constexpr (std::is_floating_point_v<S>) {
      auto clampv = [](Buf<S> &b) {
        const size_t n = (size_t)(b.step / sizeof(S)) * (size_t)b.bh;
        for (size_t i = 0; i < n; ++i) {
          const double v = (double)b.hptr()[i];
          if (v > 1e3) b.hptr()[i] = S(1e3);
          else if (v < -1e3) b.hptr()[i] = S(-1e3);
          else if (v != v) b.hptr()[i] = S(0);
        }
      };
      clampv(s);
    }
    s.dev.copyFromHost(s.host);
    d.dev.copyFromHost(d.host);
    Buf<MapT> mx(c.w, c.h, 1, 0, 0), my(c.w, c.h, 1, 0, 0);
    const int spanX = std::max(1, (c.w - 3) * 4);
    const int spanY = std::max(1, (c.h - 3) * 4);
    const int mapStepElems = mx.step / (int)sizeof(MapT);
    for (int y = 0; y < c.h; ++y)
      for (int x = 0; x < c.w; ++x) {
        const double fu = 1.0 + (double)((x * 7 + y * 3 + (int)(seed & 3)) % spanX) / 4.0;
        const double fv = 1.0 + (double)((x * 5 + y * 11 + (int)((seed >> 2) & 3)) % spanY) / 4.0;
        reinterpret_cast<MapT *>(mx.host.data())[(size_t)y * mapStepElems + x] = (MapT)fu;
        reinterpret_cast<MapT *>(my.host.data())[(size_t)y * mapStepElems + x] = (MapT)fv;
      }
    mx.dev.copyFromHost(mx.host);
    my.dev.copyFromHost(my.host);
    const S *hsrc = s.hptr() + off_bytes<S>(c.oy, c.ox, s.step, C) / sizeof(S);
    const S *dsrc = reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, C));
    const int srcStepElems = s.step / (int)sizeof(S);
    for (int interp : interps) {
      const std::vector<uint8_t> pre = d.host;
      npp_diff::clear_cuda_error();
      const NppStatus st = call(dsrc, NppiSize{c.w, c.h}, s.step, NppiRect{0, 0, c.w, c.h},
                                reinterpret_cast<const MapT *>(mx.dptr()), mx.step,
                                reinterpret_cast<const MapT *>(my.dptr()), my.step,
                                reinterpret_cast<S *>(d.dptr()), d.step, NppiSize{c.w, c.h}, interp, ctx);
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        continue;
      }
      const cudaError_t se = cudaDeviceSynchronize();
      ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
      d.dev.copyToHost(d.host);
      ++cnt.executed;
      std::vector<uint8_t> want = pre;
      const MapT *hmx = reinterpret_cast<const MapT *>(mx.host.data());
      const MapT *hmy = reinterpret_cast<const MapT *>(my.host.data());
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x) {
          const double u = (double)hmx[(size_t)y * mapStepElems + x];
          const double v = (double)hmy[(size_t)y * mapStepElems + x];
          const size_t db = off_bytes<S>(y, x, d.step, C);
          for (int ch = 0; ch < A; ++ch) {
            const S val = warp_sample_channel<S>(hsrc + ch, srcStepElems, C, c.w, c.h, u, v, interp);
            std::memcpy(want.data() + db + (size_t)ch * sizeof(S), &val, sizeof(S));
          }
        }
      const GenCase dc{c.w, c.h, 0, 0, 0, kSmall, 0};
      if (!rotate_check<S, C, A>(id, dc, interp, pre, d.host, want, d.step, std::vector<uint8_t>(), cnt, first)) {
        if (cnt.mismatched >= 8) break;
      }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}


// ---------------------------------------------------------------------------
// WarpAffine packed reference (ported from the measured g1 executor).
// ---------------------------------------------------------------------------
template <typename S> struct AffineVT {
  using type = float;
};
template <> struct AffineVT<Npp64f> {
  using type = double;
};

inline float aff_mul(float a, float b) { return (float)((double)a * (double)b); }
inline float aff_add(float a, float b) { return (float)((double)a + (double)b); }
inline float aff_sub(float a, float b) { return (float)((double)a - (double)b); }
inline double aff_mul(double a, double b) { return a * b; }
inline double aff_add(double a, double b) { return a + b; }

inline bool affine_invert(const double in[2][3], double out[2][3]) {
  const double det = in[0][0] * in[1][1] - in[0][1] * in[1][0];
  if (std::fabs(det) < 1e-12) return false;
  const double invDet = 1.0 / det;
  out[0][0] = in[1][1] * invDet;
  out[0][1] = -in[0][1] * invDet;
  out[1][0] = -in[1][0] * invDet;
  out[1][1] = in[0][0] * invDet;
  out[0][2] = -(out[0][0] * in[0][2] + out[0][1] * in[1][2]);
  out[1][2] = -(out[1][0] * in[0][2] + out[1][1] * in[1][2]);
  return true;
}

inline float aff_div3(float x) {
  const float kThird = 0x1.555556p-2f;
  const float a = std::fma(x, kThird, 0.0f);
  const float r = std::fma(a, -3.0f, x);
  return std::fma(r, kThird, a);
}

inline void aff_cubic_weights(float t, float w[4]) {
  const float t1 = aff_sub(t, 1.0f);
  const float t2 = aff_sub(t, 2.0f);
  const float t3 = aff_sub(t, 3.0f);
  w[0] = aff_mul(aff_mul(t1, aff_mul(t2, -0.5f)), aff_div3(t3));
  w[1] = aff_mul(aff_mul(t, t2), aff_mul(t3, 0.5f));
  w[2] = aff_mul(t3, aff_mul(t1, aff_mul(t, -0.5f)));
  w[3] = aff_mul(t2, aff_mul(aff_mul(t1, 0.5f), aff_div3(t)));
}

inline float aff_cubic_weight_poly(double x) {
  x = std::fabs(x);
  if (x <= 1.0) return (float)(((1.5 * x - 2.5) * x) * x + 1.0);
  if (x < 2.0) return (float)((((-0.5 * x + 2.5) * x) - 4.0) * x + 2.0);
  return 0.0f;
}

template <typename S>
inline double affine_at(const S *src, int stepElems, int w, int h, int x, int y, int C, int ch) {
  const int cx = resize_clampi(x, 0, w - 1), cy = resize_clampi(y, 0, h - 1);
  return (double)warp_load(src[(size_t)cy * stepElems + (size_t)cx * C + ch]);
}

template <typename S>
inline S affine_cast(double v) {
  constexpr bool is_int = std::is_integral_v<S>;
  if constexpr (std::is_same_v<S, Npp32f>) {
    return (Npp32f)v;
  } else if constexpr (std::is_same_v<S, Npp64f>) {
    return (Npp64f)v;
  } else if constexpr (std::is_same_v<S, Npp16f>) {
    return half_round((float)v);
  } else if constexpr (is_int) {
    double lo = 0.0, hi = 0.0;
    if constexpr (std::is_same_v<S, Npp8u>) { lo = 0.0; hi = 255.0; }
    else if constexpr (std::is_same_v<S, Npp16u>) { lo = 0.0; hi = 65535.0; }
    else if constexpr (std::is_same_v<S, Npp32s>) { lo = -2147483648.0; hi = 2147483520.0; }
    v = v < lo ? lo : (v > hi ? hi : v);
    // __fadd_rz(v, copysign(0.5, v)) followed by F2I.RZ.
    const double s = v + std::copysign(0.5, v);
    float f = (float)s;
    if ((s > 0.0 && (double)f > s) || (s < 0.0 && (double)f < s)) f = std::nextafter(f, 0.0f);
    return (S)std::trunc((double)f);
  } else {
    return (S)v;
  }
}

template <typename S, int C, int A>
inline S affine_sample(const S *src, int stepElems, int w, int h, int ch, float fx, float fy, float fxX, int interp,
                       bool doubleWork) {
  using VT = typename AffineVT<S>::type;
  if (doubleWork) {
    // Work type is double, but the kernel derives the tap fractions and the
    // cubic weights in single precision from the float map.
    if (interp == NPPI_INTER_NN) {
      const int x = (int)std::floor((double)fx + 0.5), y = (int)std::floor((double)fy + 0.5);
      return affine_cast<S>(affine_at(src, stepElems, w, h, x, y, C, ch));
    }
    const int x0 = (int)std::floor(fxX), y0 = (int)std::floor(fy);
    const float dxf = aff_sub(fxX, (float)x0), dyf = aff_sub(fy, (float)y0);
    const double dx = (double)dxf, dy = (double)dyf;
    if (interp == NPPI_INTER_LINEAR) {
      const double v00 = affine_at(src, stepElems, w, h, x0, y0, C, ch);
      const double v10 = affine_at(src, stepElems, w, h, x0 + 1, y0, C, ch);
      const double v01 = affine_at(src, stepElems, w, h, x0, y0 + 1, C, ch);
      const double v11 = affine_at(src, stepElems, w, h, x0 + 1, y0 + 1, C, ch);
      const double v0 = (1.0 - dx) * v00 + dx * v10;
      const double v1 = (1.0 - dx) * v01 + dx * v11;
      return affine_cast<S>((1.0 - dy) * v0 + dy * v1);
    }
    float wx[4], wy[4];
    const float ax = fxX - (float)x0, ay = fy - (float)y0;
    for (int i = 0; i < 4; ++i) {
      wx[i] = aff_cubic_weight_poly((double)(ax - (float)(i - 1)));
      wy[i] = aff_cubic_weight_poly((double)(ay - (float)(i - 1)));
    }
    double sum = 0.0;
    for (int j = 0; j < 4; ++j)
      for (int i = 0; i < 4; ++i)
        sum += affine_at(src, stepElems, w, h, x0 + i - 1, y0 + j - 1, C, ch) * ((double)wx[i] * (double)wy[j]);
    return affine_cast<S>(sum);
  }
  if (interp == NPPI_INTER_NN) {
    const int x = (int)std::floor((double)fx + 0.5), y = (int)std::floor((double)fy + 0.5);
    return affine_cast<S>(affine_at(src, stepElems, w, h, x, y, C, ch));
  }
  if (interp == NPPI_INTER_LINEAR) {
    float ux = fxX < 0.0f ? 0.0f : fxX;
    float uy = fy < 0.0f ? 0.0f : fy;
    const int x0 = (int)std::floor(ux), y0 = (int)std::floor(uy);
    const float dx = aff_sub(ux, (float)x0), dy = aff_sub(uy, (float)y0);
    const float v00 = (float)affine_at(src, stepElems, w, h, x0, y0, C, ch);
    const float v10 = (float)affine_at(src, stepElems, w, h, x0 + 1, y0, C, ch);
    const float v01 = (float)affine_at(src, stepElems, w, h, x0, y0 + 1, C, ch);
    const float v11 = (float)affine_at(src, stepElems, w, h, x0 + 1, y0 + 1, C, ch);
    const float v0 = aff_add(aff_mul(aff_sub(1.0f, dx), v00), aff_mul(dx, v10));
    const float v1 = aff_add(aff_mul(aff_sub(1.0f, dx), v01), aff_mul(dx, v11));
    return affine_cast<S>(aff_add(aff_mul(aff_sub(1.0f, dy), v0), aff_mul(dy, v1)));
  }
  const int tx = (int)std::floor(fxX), ty = (int)std::floor(fy);
  const float t = aff_sub(fxX, (float)(tx - 1));
  const float u = aff_sub(fy, (float)(ty - 1));
  float wx[4], wy[4];
  aff_cubic_weights(t, wx);
  aff_cubic_weights(u, wy);
  float rows[4];
  for (int j = 0; j < 4; ++j) {
    const int y = ty - 1 + j;
    float r = aff_mul(wx[1], (float)affine_at(src, stepElems, w, h, tx, y, C, ch));
    r = std::fma(wx[0], (float)affine_at(src, stepElems, w, h, tx - 1, y, C, ch), r);
    r = std::fma(wx[2], (float)affine_at(src, stepElems, w, h, tx + 1, y, C, ch), r);
    r = std::fma(wx[3], (float)affine_at(src, stepElems, w, h, tx + 2, y, C, ch), r);
    rows[j] = r;
  }
  float out = aff_mul(wy[0], rows[0]);
  out = std::fma(wy[1], rows[1], out);
  out = std::fma(wy[2], rows[2], out);
  out = std::fma(wy[3], rows[3], out);
  return affine_cast<S>(out);
}

// InvertMap=true for nppiWarpAffine (coeffs are src->dst), false for the Back
// family (coeffs are already dst->src).
template <typename S, int C, int A, bool InvertMap, class Call>
void run_warp_affine(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  const int interps[] = {NPPI_INTER_NN, NPPI_INTER_LINEAR, NPPI_INTER_CUBIC};
  for (const GenCase &c : make_gen_cases(0)) {
    if (c.w < 2 || c.h < 2) continue;
    Buf<S> s(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    Buf<S> d(c.w, c.h, C, 0, 0x5A);
    const uint64_t seed = case_seed(id, c.w, c.h, s.step, d.step, (int)c.regime, 0);
    fill_regime(s, C, seed, c.regime, 0, false, false, std::is_floating_point_v<S>);
    if constexpr (std::is_floating_point_v<S>) {
      auto clampv = [](Buf<S> &b) {
        const size_t n = (size_t)(b.step / sizeof(S)) * (size_t)b.bh;
        for (size_t i = 0; i < n; ++i) {
          const double v = (double)b.hptr()[i];
          if (v > 1e3) b.hptr()[i] = S(1e3);
          else if (v < -1e3) b.hptr()[i] = S(-1e3);
          else if (v != v) b.hptr()[i] = S(0);
        }
      };
      clampv(s);
    }
    s.dev.copyFromHost(s.host);
    d.dev.copyFromHost(d.host);
    // Binary-exact transforms: the inverse map has exact float coordinates,
    // so nearest-neighbour ties are reproducible across the reference and NPP.
    const double w1 = c.w - 1.0, h1 = c.h - 1.0;
    double coeffs[6][2][3] = {
        {{1, 0, 0}, {0, 1, 0}},
        {{1, 0, 1}, {0, 1, -1}},
        {{2, 0, 0}, {0, 2, 0}},
        {{0, -1, h1}, {1, 0, 0}},
        {{-1, 0, w1}, {0, -1, h1}},
        {{-1, 0, w1}, {0, 1, 0}},
    };
    const S *hsrc = s.hptr() + off_bytes<S>(c.oy, c.ox, s.step, C) / sizeof(S);
    const S *dsrc = reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, C));
    const int srcStepElems = s.step / (int)sizeof(S);
    const bool doubleWork = std::is_same_v<S, Npp64f>;
    for (int mi = 0; mi < 6; ++mi) {
      const auto &cf = coeffs[mi];
      for (int interp : interps) {
        const std::vector<uint8_t> pre = d.host;
        npp_diff::clear_cuda_error();
        const NppStatus st = call(dsrc, NppiSize{c.w, c.h}, s.step, NppiRect{0, 0, c.w, c.h},
                                  reinterpret_cast<S *>(d.dptr()), d.step, NppiRect{0, 0, c.w, c.h},
                                  reinterpret_cast<const double *>(cf), interp, ctx);
        if (st != NPP_NO_ERROR) {
          ++cnt.skipped;
          if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
          continue;
        }
        const cudaError_t se = cudaDeviceSynchronize();
        ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
        d.dev.copyToHost(d.host);
        ++cnt.executed;
        double map[2][3];
        if constexpr (InvertMap) {
          if (!affine_invert(cf, map)) continue;
        } else {
          for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 3; ++j) map[i][j] = cf[i][j];
        }
        std::vector<uint8_t> want = pre;
        for (int y = 0; y < c.h; ++y)
          for (int x = 0; x < c.w; ++x) {
            const float dx = (float)x, dy = (float)y;
            // The kernel evaluates the map in single precision for every
            // pixel type (only the interpolation arithmetic is double for 64f).
            const float fx = aff_add(std::fma(dx, (float)map[0][0], aff_mul((float)map[0][1], dy)),
                                     (float)map[0][2]);
            const float fy = aff_add(std::fma(dx, (float)map[1][0], aff_mul((float)map[1][1], dy)),
                                     (float)map[1][2]);
            if (!(fx >= 0.0f && fx <= (float)(c.w - 1) && fy >= 0.0f && fy <= (float)(c.h - 1))) continue;
            const size_t db = off_bytes<S>(y, x, d.step, C);
            for (int ch = 0; ch < A; ++ch) {
              const S v = affine_sample<S, C, A>(hsrc, srcStepElems, c.w, c.h, ch, fx, fy, fx, interp, doubleWork);
              std::memcpy(want.data() + db + (size_t)ch * sizeof(S), &v, sizeof(S));
            }
          }
        const GenCase dc{c.w, c.h, 0, 0, 0, kSmall, 0};
        if (!rotate_check<S, C, A>(id, dc, interp, pre, d.host, want, d.step, std::vector<uint8_t>(), cnt, first)) {
          if (cnt.mismatched >= 8) break;
        }
        if (cnt.mismatched >= 8) break;
      }
    if (cnt.mismatched >= 8) break;
  }
  }
  finish(id, cnt, first);
}


// ---------------------------------------------------------------------------
// nppiWarpAffineQuad: the forward map is built from the first three quad
// vertices, each coefficient rounded to float, then inverted.  The accept
// gate evaluates the double map with a 4.4e-7 tolerance on the y upper bound.
// ---------------------------------------------------------------------------
inline bool affine_from_quads(const double s[4][2], const double d[4][2], double out[2][3], bool *warn) {
  const double u1x = s[1][0] - s[0][0], u1y = s[1][1] - s[0][1];
  const double u2x = s[2][0] - s[0][0], u2y = s[2][1] - s[0][1];
  const double det = u1x * u2y - u2x * u1y;
  if (std::fabs(det) < 1e-12) return false;
  const double v1x = d[1][0] - d[0][0], v1y = d[1][1] - d[0][1];
  const double v2x = d[2][0] - d[0][0], v2y = d[2][1] - d[0][1];
  out[0][0] = (v1x * u2y - v2x * u1y) / det;
  out[0][1] = (v2x * u1x - v1x * u2x) / det;
  out[1][0] = (v1y * u2y - v2y * u1y) / det;
  out[1][1] = (v2y * u1x - v1y * u2x) / det;
  out[0][2] = d[0][0] - out[0][0] * s[0][0] - out[0][1] * s[0][1];
  out[1][2] = d[0][1] - out[1][0] * s[0][0] - out[1][1] * s[0][1];
  *warn = false;
  const double m3x = out[0][0] * s[3][0] + out[0][1] * s[3][1] + out[0][2];
  const double m3y = out[1][0] * s[3][0] + out[1][1] * s[3][1] + out[1][2];
  const double scale = 1.0 + std::fabs(d[3][0]) + std::fabs(d[3][1]);
  if (std::fabs(m3x - d[3][0]) > 1e-6 * scale || std::fabs(m3y - d[3][1]) > 1e-6 * scale) *warn = true;
  return true;
}

template <typename S, int C, int A, class Call>
void run_warp_affine_quad(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  const int interps[] = {NPPI_INTER_NN, NPPI_INTER_LINEAR, NPPI_INTER_CUBIC};
  for (const GenCase &c : make_gen_cases(0)) {
    if (c.w < 2 || c.h < 2) continue;
    Buf<S> s(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    Buf<S> d(c.w, c.h, C, 0, 0x5A);
    const uint64_t seed = case_seed(id, c.w, c.h, s.step, d.step, (int)c.regime, 0);
    fill_regime(s, C, seed, c.regime, 0, false, false, std::is_floating_point_v<S>);
    if constexpr (std::is_floating_point_v<S>) {
      auto clampv = [](Buf<S> &b) {
        const size_t n = (size_t)(b.step / sizeof(S)) * (size_t)b.bh;
        for (size_t i = 0; i < n; ++i) {
          const double v = (double)b.hptr()[i];
          if (v > 1e3) b.hptr()[i] = S(1e3);
          else if (v < -1e3) b.hptr()[i] = S(-1e3);
          else if (v != v) b.hptr()[i] = S(0);
        }
      };
      clampv(s);
    }
    s.dev.copyFromHost(s.host);
    d.dev.copyFromHost(d.host);
    const double w1 = c.w - 1.0, h1 = c.h - 1.0;
    // Quad vertices must be in polygon order (TL, TR, BR, BL); NPP rejects
    // crossed orders.
    const double srcQ[4][2] = {{0, 0}, {w1, 0}, {w1, h1}, {0, h1}};
    // The quad path rounds nearest-neighbour ties at exactly .5 differently
    // from the coefficient family, so its transforms are chosen with integer
    // inverse coordinates (0.5x upscale instead of 2x downscale).
    double mats[6][2][3] = {
        {{1, 0, 0}, {0, 1, 0}},
        {{1, 0, 1}, {0, 1, -1}},
        {{0.5, 0, 0}, {0, 0.5, 0}},
        {{0, -1, h1}, {1, 0, 0}},
        {{-1, 0, w1}, {0, -1, h1}},
        {{-1, 0, w1}, {0, 1, 0}},
    };
    const S *hsrc = s.hptr() + off_bytes<S>(c.oy, c.ox, s.step, C) / sizeof(S);
    const S *dsrc = reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, C));
    const int srcStepElems = s.step / (int)sizeof(S);
    const bool doubleWork = std::is_same_v<S, Npp64f>;
    for (int mi = 0; mi < 6; ++mi) {
      const auto &M = mats[mi];
      double dstQ[4][2];
      for (int i = 0; i < 4; ++i) {
        dstQ[i][0] = M[0][0] * srcQ[i][0] + M[0][1] * srcQ[i][1] + M[0][2];
        dstQ[i][1] = M[1][0] * srcQ[i][0] + M[1][1] * srcQ[i][1] + M[1][2];
      }
      for (int interp : interps) {
        const std::vector<uint8_t> pre = d.host;
        npp_diff::clear_cuda_error();
        const NppStatus st = call(dsrc, NppiSize{c.w, c.h}, s.step, NppiRect{0, 0, c.w, c.h},
                                  reinterpret_cast<const double *>(srcQ), reinterpret_cast<S *>(d.dptr()), d.step,
                                  NppiRect{0, 0, c.w, c.h}, reinterpret_cast<const double *>(dstQ), interp, ctx);
        if (st != NPP_NO_ERROR) {
          ++cnt.skipped;
          if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
          continue;
        }
        const cudaError_t se = cudaDeviceSynchronize();
        ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
        d.dev.copyToHost(d.host);
        ++cnt.executed;
        double fwd[2][3];
        bool warn = false;
        if (!affine_from_quads(srcQ, dstQ, fwd, &warn)) continue;
        for (int r = 0; r < 2; ++r)
          for (int k = 0; k < 3; ++k) fwd[r][k] = (double)(float)fwd[r][k];
        double map[2][3];
        if (!affine_invert(fwd, map)) continue;
        std::vector<uint8_t> want = pre;
        for (int y = 0; y < c.h; ++y)
          for (int x = 0; x < c.w; ++x) {
            const float dx = (float)x, dy = (float)y;
            const float fx = aff_add(std::fma(dx, (float)map[0][0], aff_mul((float)map[0][1], dy)),
                                     (float)map[0][2]);
            const float fy = aff_add(std::fma(dx, (float)map[1][0], aff_mul((float)map[1][1], dy)),
                                     (float)map[1][2]);
            const double fxd = aff_add(std::fma((double)dx, (double)(float)map[0][0],
                                                (double)(float)map[0][1] * (double)dy),
                                       (double)(float)map[0][2]);
            const double fyd = aff_add(std::fma((double)dx, (double)(float)map[1][0],
                                                (double)(float)map[1][1] * (double)dy),
                                       (double)(float)map[1][2]);
            if (!(fxd >= 0.0 && fxd <= (double)(c.w - 1) && fyd >= 0.0 &&
                  fyd <= (double)(c.h - 1) + 4.4e-7))
              continue;
            const size_t db = off_bytes<S>(y, x, d.step, C);
            for (int ch = 0; ch < A; ++ch) {
              const S v = affine_sample<S, C, A>(hsrc, srcStepElems, c.w, c.h, ch, fx, fy, fx, interp, doubleWork);
              std::memcpy(want.data() + db + (size_t)ch * sizeof(S), &v, sizeof(S));
            }
          }
        const GenCase dc{c.w, c.h, 0, 0, 0, kSmall, 0};
        if (!rotate_check<S, C, A>(id, dc, interp, pre, d.host, want, d.step, std::vector<uint8_t>(), cnt, first)) {
          if (cnt.mismatched >= 8) break;
        }
        if (cnt.mismatched >= 8) break;
      }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}


// ---------------------------------------------------------------------------
// nppiWarpAffineBatch / BatchInit (SASS-replayed float-rounded inversion and
// ROI-relative double map with the row-pair reuse rule).
// ---------------------------------------------------------------------------
inline void invert_affine_batch(const double in[6], double out[6]) {
  const double det = std::fma(in[0], in[4], -(in[1] * in[3]));
  const bool invertible = det != 0.0;
  const double inv = 1.0 / (invertible ? det : in[0]);
  const float m00 = (float)((invertible ? in[4] : 1.0) * inv);
  const float m01 = (float)((-in[1]) * inv);
  const float m10 = (float)((invertible ? -in[3] : 0.0) * inv);
  const float m11 = (float)(in[0] * inv);
  out[0] = m00;
  out[1] = m01;
  out[3] = m10;
  out[4] = m11;
  out[2] = -(float)std::fma(in[2], (double)m00, in[5] * (double)m01);
  out[5] = -(float)std::fma(in[2], (double)m10, in[5] * (double)m11);
}

inline void affine_batch_map(const double c[2][3], int x, int y, float &fx, float &fy) {
  const double dx = (double)(float)x, dy = (double)(float)y;
  fx = (float)(std::fma(c[0][1], dy, c[0][0] * dx) + c[0][2]);
  fy = (float)(std::fma(c[1][1], dy, c[1][0] * dx) + c[1][2]);
}

template <typename S, int C, int A, class Call>
void run_warp_affine_batch(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  const int interps[] = {NPPI_INTER_NN, NPPI_INTER_LINEAR, NPPI_INTER_CUBIC};
  const int B = 3, W = 33, H = 17;
  const double fwd[3][6] = {{1, 0, 0, 0, 1, 0}, {1, 0, 1, 0, 1, -1}, {0.5, 0, 0, 0, 0.5, 0}};
  std::vector<Buf<S> *> srcs, dsts;
  std::vector<Buf<Npp64f> *> coeffs;
  std::vector<std::vector<uint8_t>> preHost(B);
  const uint64_t seed = case_seed(id, W, H, B, 0, 0, 0);
  for (int i = 0; i < B; ++i) {
    Buf<S> *a = new Buf<S>(W, H, C, 0, 0);
    Buf<S> *d = new Buf<S>(W, H, C, 0, 0x5A);
    Buf<Npp64f> *cf = new Buf<Npp64f>(6, 1, 1, 0, 0);
    fill_regime(*a, C, seed ^ mix64((uint64_t)i), (i & 1) ? kWide : kSmall, i);
    a->dev.copyFromHost(a->host);
    d->dev.copyFromHost(d->host);
    std::memcpy(cf->host.data(), fwd[i], sizeof(fwd[i]));
    cf->dev.copyFromHost(cf->host);
    srcs.push_back(a);
    dsts.push_back(d);
    coeffs.push_back(cf);
    preHost[(size_t)i] = d->host;
  }
  std::vector<NppiWarpAffineBatchCXR> h(B);
  for (int i = 0; i < B; ++i) {
    h[i] = {(const void *)srcs[(size_t)i]->dptr(), srcs[(size_t)i]->step, (void *)dsts[(size_t)i]->dptr(),
            dsts[(size_t)i]->step, (Npp64f *)coeffs[(size_t)i]->dptr(), {}};
  }
  Buf<NppiWarpAffineBatchCXR> list(B, 1, 1, 0, 0);
  std::memcpy(list.host.data(), h.data(), sizeof(NppiWarpAffineBatchCXR) * B);
  list.dev.copyFromHost(list.host);
  const NppStatus initSt = nppiWarpAffineBatchInit_Ctx(
      reinterpret_cast<NppiWarpAffineBatchCXR *>(list.dptr()), (unsigned)B, ctx);
  if (initSt != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] BatchInit failed: " << (int)initSt;
    for (auto *p : srcs) delete p;
    for (auto *p : dsts) delete p;
    for (auto *p : coeffs) delete p;
    return;
  }
  const S *hsrc[3];
  const S *dsrc[3];
  const int srcStepElems[3] = {srcs[0]->step / (int)sizeof(S), srcs[1]->step / (int)sizeof(S),
                               srcs[2]->step / (int)sizeof(S)};
  for (int i = 0; i < B; ++i) {
    hsrc[i] = srcs[(size_t)i]->hptr();
    dsrc[i] = reinterpret_cast<const S *>(srcs[(size_t)i]->dptr());
  }
  const bool doubleWork = std::is_same_v<S, Npp64f>;
  for (int interp : interps) {
    std::vector<std::vector<uint8_t>> pre(B);
    for (int i = 0; i < B; ++i) {
      pre[(size_t)i] = dsts[(size_t)i]->host;
      dsts[(size_t)i]->dev.copyFromHost(dsts[(size_t)i]->host);
    }
    npp_diff::clear_cuda_error();
    const NppStatus st = call(NppiSize{W, H}, NppiRect{0, 0, W, H}, NppiRect{0, 0, W, H}, interp,
                              reinterpret_cast<NppiWarpAffineBatchCXR *>(list.dptr()), (unsigned)B, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] batch sync: " << cudaGetErrorString(se);
    ++cnt.executed;
    bool failed = false;
    for (int i = 0; i < B && !failed; ++i) {
      Buf<S> *d = dsts[(size_t)i];
      d->dev.copyToHost(d->host);
      double tr[6];
      invert_affine_batch(fwd[i], tr);
      double c[2][3] = {{tr[0], tr[1], tr[2]}, {tr[3], tr[4], tr[5]}};
      std::vector<uint8_t> want = pre[(size_t)i];
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
          float fx, fy;
          affine_batch_map(c, x, y, fx, fy);
          if (!(fx >= 0.0f && fx <= (float)(W - 1) && fy >= 0.0f && fy <= (float)(H - 1))) continue;
          float fxX = fx;
          if ((y & 1) != 0) {
            float ex, ey;
            affine_batch_map(c, x, y - 1, ex, ey);
            if (!(ex >= 0.0f && ex <= (float)(W - 1) && ey >= 0.0f && ey <= (float)(H - 1))) continue;
            if constexpr (!std::is_same_v<S, Npp16f>) {
              if ((int)std::floor(ey) == (int)std::floor(fy)) fxX = ex;
            }
          }
          const size_t db = off_bytes<S>(y, x, d->step, C);
          for (int ch = 0; ch < A; ++ch) {
            const S v = affine_sample<S, C, A>(hsrc[i], srcStepElems[i], W, H, ch, fx, fy, fxX, interp, doubleWork);
            std::memcpy(want.data() + db + (size_t)ch * sizeof(S), &v, sizeof(S));
          }
        }
      const GenCase dc{W, H, 0, 0, 0, kSmall, 0};
      if (!rotate_check<S, C, A>(id, dc, interp, pre[(size_t)i], d->host, want, d->step,
                                 std::vector<uint8_t>(), cnt, first))
        failed = true;
      if (cnt.mismatched >= 8) break;
    }
    if (failed) break;
  }
  for (auto *p : srcs) delete p;
  for (auto *p : dsts) delete p;
  for (auto *p : coeffs) delete p;
  finish(id, cnt, first);
}

template <typename S, int C, int A, class Call>
void run_warp_affine_batch_init(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  const int B = 3;
  const double fwd[3][6] = {{1, 0, 0, 0, 1, 0}, {1, 0, 1, 0, 1, -1}, {1, 2, 0, 2, 4, 0}};
  std::vector<Buf<Npp64f> *> coeffs;
  std::vector<NppiWarpAffineBatchCXR> h(B);
  for (int i = 0; i < B; ++i) {
    Buf<Npp64f> *cf = new Buf<Npp64f>(6, 1, 1, 0, 0);
    std::memcpy(cf->host.data(), fwd[i], sizeof(fwd[i]));
    cf->dev.copyFromHost(cf->host);
    coeffs.push_back(cf);
    h[i] = {nullptr, 0, nullptr, 0, (Npp64f *)cf->dptr(), {}};
  }
  Buf<NppiWarpAffineBatchCXR> list(B, 1, 1, 0, 0);
  std::memcpy(list.host.data(), h.data(), sizeof(NppiWarpAffineBatchCXR) * B);
  list.dev.copyFromHost(list.host);
  npp_diff::clear_cuda_error();
  const NppStatus st = call(reinterpret_cast<NppiWarpAffineBatchCXR *>(list.dptr()), (unsigned)B, ctx);
  if (st != NPP_NO_ERROR) {
    ++cnt.skipped;
    if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
    for (auto *p : coeffs) delete p;
    finish(id, cnt, first);
    return;
  }
  const cudaError_t se = cudaDeviceSynchronize();
  ASSERT_EQ(se, cudaSuccess) << "[" << id << "] sync: " << cudaGetErrorString(se);
  list.dev.copyToHost(list.host);
  ++cnt.executed;
  for (int i = 0; i < B; ++i) {
    double want[6];
    invert_affine_batch(fwd[i], want);
    const NppiWarpAffineBatchCXR &e = reinterpret_cast<const NppiWarpAffineBatchCXR *>(list.host.data())[i];
    for (int k = 0; k < 6; ++k) {
      ++cnt.compared;
      const double got = e.aTransformedCoeffs[k / 3][k % 3];
      if (got != want[k]) {
        char b[360];
        std::snprintf(b, sizeof(b), "[%s] entry=%d coeff=%d got=%.17g want=%.17g", id, i, k, got, want[k]);
        if (first.empty()) first = b;
        EXPECT_TRUE(false) << b;
        ++cnt.mismatched;
        break;
      }
    }
  }
  for (auto *p : coeffs) delete p;
  finish(id, cnt, first);
}


// ---------------------------------------------------------------------------
// WarpPerspective packed reference.  Forward coefficients map source to
// destination (the kernel applies their inverse with row/x-span gating); the
// Back family supplies destination-to-source coefficients directly.
// ---------------------------------------------------------------------------
struct Mat3d {
  double m[3][3];
};

inline double mat3_det(const Mat3d &a) {
  return a.m[0][0] * (a.m[1][1] * a.m[2][2] - a.m[1][2] * a.m[2][1]) -
         a.m[0][1] * (a.m[1][0] * a.m[2][2] - a.m[1][2] * a.m[2][0]) +
         a.m[0][2] * (a.m[1][0] * a.m[2][1] - a.m[1][1] * a.m[2][0]);
}

inline Mat3d mat3_inverse(const Mat3d &a) {
  Mat3d r;
  const double id = 1.0 / mat3_det(a);
  r.m[0][0] = (a.m[1][1] * a.m[2][2] - a.m[1][2] * a.m[2][1]) * id;
  r.m[0][1] = (a.m[0][2] * a.m[2][1] - a.m[0][1] * a.m[2][2]) * id;
  r.m[0][2] = (a.m[0][1] * a.m[1][2] - a.m[0][2] * a.m[1][1]) * id;
  r.m[1][0] = (a.m[1][2] * a.m[2][0] - a.m[1][0] * a.m[2][2]) * id;
  r.m[1][1] = (a.m[0][0] * a.m[2][2] - a.m[0][2] * a.m[2][0]) * id;
  r.m[1][2] = (a.m[0][2] * a.m[1][0] - a.m[0][0] * a.m[1][2]) * id;
  r.m[2][0] = (a.m[1][0] * a.m[2][1] - a.m[1][1] * a.m[2][0]) * id;
  r.m[2][1] = (a.m[0][1] * a.m[2][0] - a.m[0][0] * a.m[2][1]) * id;
  r.m[2][2] = (a.m[0][0] * a.m[1][1] - a.m[0][1] * a.m[1][0]) * id;
  return r;
}

inline void mat3_map(const Mat3d &M, double x, double y, double &u, double &v, double &den) {
  den = M.m[2][0] * x + M.m[2][1] * y + M.m[2][2];
  const double inv = 1.0 / den;
  u = (M.m[0][0] * x + M.m[0][1] * y + M.m[0][2]) * inv;
  v = (M.m[1][0] * x + M.m[1][1] * y + M.m[1][2]) * inv;
}

// NPP evaluates the perspective map in float32 (measured: a map that is
// mathematically exactly k+0.5 comes out as 12.499999046 from the float
// division and selects row k, while a double map yields exactly 12.5).
// warp_fmul/warp_fadd are single-rounding float ops on double intermediates.
inline void mat3_map_f32(const Mat3d &M, int x, int y, double &uOut, double &vOut, double &denOut) {
  const float m00 = (float)M.m[0][0], m01 = (float)M.m[0][1], m02 = (float)M.m[0][2];
  const float m10 = (float)M.m[1][0], m11 = (float)M.m[1][1], m12 = (float)M.m[1][2];
  const float m20 = (float)M.m[2][0], m21 = (float)M.m[2][1], m22 = (float)M.m[2][2];
  const float fx = (float)x, fy = (float)y;
  const float den = warp_fadd(warp_fadd(warp_fmul(m20, fx), warp_fmul(m21, fy)), m22);
  const float numU = warp_fadd(warp_fadd(warp_fmul(m00, fx), warp_fmul(m01, fy)), m02);
  const float numV = warp_fadd(warp_fadd(warp_fmul(m10, fx), warp_fmul(m11, fy)), m12);
  denOut = (double)den;
  uOut = (double)(float)((double)numU / (double)den);
  vOut = (double)(float)((double)numV / (double)den);
}

inline bool persp_pixel_gate(const Mat3d &M, NppiRect src, int x, int y, double &u, double &v) {
  double den;
  mat3_map_f32(M, x, y, u, v, den);
  if (!(den > 0.0)) return false;
  const double uhi = (double)src.x + src.width - 1.0;
  const double vhi = (double)src.y + src.height - 1.0;
  return u >= (double)src.x && u <= uhi && v >= (double)src.y && v <= vhi;
}

template <typename S, int C, int A, bool Forward, class Call>
void run_warp_perspective(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  const int interps[] = {NPPI_INTER_NN, NPPI_INTER_LINEAR, NPPI_INTER_CUBIC};
  for (const GenCase &c : make_gen_cases(0)) {
    if (c.w < 2 || c.h < 2) continue;
    Buf<S> s(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    Buf<S> d(c.w, c.h, C, 0, 0x5A);
    const uint64_t seed = case_seed(id, c.w, c.h, s.step, d.step, (int)c.regime, 0);
    fill_regime(s, C, seed, c.regime, 0, false, false, std::is_floating_point_v<S>);
    if constexpr (std::is_floating_point_v<S>) {
      auto clampv = [](Buf<S> &b) {
        const size_t n = (size_t)(b.step / sizeof(S)) * (size_t)b.bh;
        for (size_t i = 0; i < n; ++i) {
          const double v = (double)b.hptr()[i];
          if (v > 1e3) b.hptr()[i] = S(1e3);
          else if (v < -1e3) b.hptr()[i] = S(-1e3);
          else if (v != v) b.hptr()[i] = S(0);
        }
      };
      clampv(s);
    }
    s.dev.copyFromHost(s.host);
    d.dev.copyFromHost(d.host);
    const double w1 = c.w - 1.0, h1 = c.h - 1.0;
    double coeffs[6][3][3] = {
        {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
        {{1, 0, 1}, {0, 1, -1}, {0, 0, 1}},
        {{2, 0, 0}, {0, 2, 0}, {0, 0, 1}},
        {{0, -1, h1}, {1, 0, 0}, {0, 0, 1}},
        {{-1, 0, w1}, {0, 1, 0}, {0, 0, 1}},
        {{1, 0, 0}, {0, 1, 0}, {0.002, 0, 1}},
    };
    const S *hsrc = s.hptr() + off_bytes<S>(c.oy, c.ox, s.step, C) / sizeof(S);
    const S *dsrc = reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, C));
    const int srcStepElems = s.step / (int)sizeof(S);
    for (int mi = 0; mi < 6; ++mi) {
      const auto &cf = coeffs[mi];
      for (int interp : interps) {
        const std::vector<uint8_t> pre = d.host;
        npp_diff::clear_cuda_error();
        const NppStatus st = call(dsrc, NppiSize{c.w, c.h}, s.step, NppiRect{0, 0, c.w, c.h},
                                  reinterpret_cast<S *>(d.dptr()), d.step, NppiRect{0, 0, c.w, c.h},
                                  reinterpret_cast<const double *>(cf), interp, ctx);
        if (st != NPP_NO_ERROR) {
          ++cnt.skipped;
          if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
          continue;
        }
        const cudaError_t se = cudaDeviceSynchronize();
        ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
        d.dev.copyToHost(d.host);
        ++cnt.executed;
        Mat3d M3, F3;
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j) {
            M3.m[i][j] = cf[i][j];
            F3.m[i][j] = cf[i][j];
          }
        if constexpr (Forward) {
          F3 = M3;
          M3 = mat3_inverse(F3);
        } else {
          F3 = mat3_inverse(M3);
        }
        std::vector<uint8_t> want = pre;
        for (int y = 0; y < c.h; ++y)
          for (int x = 0; x < c.w; ++x) {
            (void)F3;
            double u, v;
            if (!persp_pixel_gate(M3, NppiRect{0, 0, c.w, c.h}, x, y, u, v)) continue;
            // NPP samples at single-precision coordinates: a map that is
            // mathematically exactly k+0.5 but 12.499999999999996 in double
            // rounds up to the float tie and selects row k+1.
            const double uf = (double)(float)u, vf = (double)(float)v;
            const size_t db = off_bytes<S>(y, x, d.step, C);
            for (int ch = 0; ch < A; ++ch) {
              const S val = warp_sample_channel<S>(hsrc + ch, srcStepElems, C, c.w, c.h, uf, vf, interp);
              std::memcpy(want.data() + db + (size_t)ch * sizeof(S), &val, sizeof(S));
            }
          }
        const GenCase dc{c.w, c.h, 0, 0, 0, kSmall, 0};
        if (!rotate_check<S, C, A>(id, dc, interp, pre, d.host, want, d.step, std::vector<uint8_t>(), cnt, first)) {
          if (cnt.mismatched >= 8) break;
        }
        if (cnt.mismatched >= 8) break;
      }
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}


// ---------------------------------------------------------------------------
// WarpPerspectiveQuad: the homography is recovered from the source/destination
// quads, then applied with the same float32 map as the coefficient family.
// The acceptance region is the source quad (half-plane test in float32).
// ---------------------------------------------------------------------------
inline bool persp_quad_homography(const double src[4][2], const double dst[4][2], Mat3d &out) {
  double a[8][9] = {};
  for (int i = 0; i < 4; ++i) {
    const double x = src[i][0], y = src[i][1], X = dst[i][0], Y = dst[i][1];
    a[2 * i][0] = x; a[2 * i][1] = y; a[2 * i][2] = 1.0;
    a[2 * i][6] = -X * x; a[2 * i][7] = -X * y; a[2 * i][8] = X;
    a[2 * i + 1][3] = x; a[2 * i + 1][4] = y; a[2 * i + 1][5] = 1.0;
    a[2 * i + 1][6] = -Y * x; a[2 * i + 1][7] = -Y * y; a[2 * i + 1][8] = Y;
  }
  for (int col = 0; col < 8; ++col) {
    int pivot = col;
    for (int r = col + 1; r < 8; ++r)
      if (std::fabs(a[r][col]) > std::fabs(a[pivot][col])) pivot = r;
    if (a[pivot][col] == 0.0) return false;
    if (pivot != col)
      for (int k = 0; k < 9; ++k) std::swap(a[col][k], a[pivot][k]);
    const double p = a[col][col];
    for (int r = 0; r < 8; ++r) {
      if (r == col) continue;
      const double f = a[r][col] / p;
      if (f == 0.0) continue;
      for (int k = col; k < 9; ++k) a[r][k] -= f * a[col][k];
    }
  }
  double h[8];
  for (int i = 0; i < 8; ++i) h[i] = a[i][8] / a[i][i];
  out.m[0][0] = h[0]; out.m[0][1] = h[1]; out.m[0][2] = h[2];
  out.m[1][0] = h[3]; out.m[1][1] = h[4]; out.m[1][2] = h[5];
  out.m[2][0] = h[6]; out.m[2][1] = h[7]; out.m[2][2] = 1.0;
  return true;
}

inline bool persp_inside_quad_f32(const double q[4][2], float u, float v) {
  bool allPos = true, allNeg = true;
  for (int i = 0; i < 4; ++i) {
    const int j = (i + 1) & 3;
    const float ax = (float)q[i][0], ay = (float)q[i][1];
    const float bx = (float)q[j][0], by = (float)q[j][1];
    const float t1 = warp_fmul(warp_fsub(bx, ax), warp_fsub(v, ay));
    const float t2 = warp_fmul(warp_fsub(by, ay), warp_fsub(u, ax));
    const float cross = warp_fsub(t1, t2);
    allPos = allPos && (cross >= 0.0f);
    allNeg = allNeg && (cross <= 0.0f);
  }
  return allPos || allNeg;
}

template <typename S, int C, int A, class Call>
void run_warp_perspective_quad(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  const int interps[] = {NPPI_INTER_NN, NPPI_INTER_LINEAR, NPPI_INTER_CUBIC};
  for (const GenCase &c : make_gen_cases(0)) {
    if (c.w < 2 || c.h < 2) continue;
    Buf<S> s(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    Buf<S> d(c.w, c.h, C, 0, 0x5A);
    const uint64_t seed = case_seed(id, c.w, c.h, s.step, d.step, (int)c.regime, 0);
    fill_regime(s, C, seed, c.regime, 0, false, false, std::is_floating_point_v<S>);
    if constexpr (std::is_floating_point_v<S>) {
      auto clampv = [](Buf<S> &b) {
        const size_t n = (size_t)(b.step / sizeof(S)) * (size_t)b.bh;
        for (size_t i = 0; i < n; ++i) {
          const double v = (double)b.hptr()[i];
          if (v > 1e3) b.hptr()[i] = S(1e3);
          else if (v < -1e3) b.hptr()[i] = S(-1e3);
          else if (v != v) b.hptr()[i] = S(0);
        }
      };
      clampv(s);
    }
    s.dev.copyFromHost(s.host);
    d.dev.copyFromHost(d.host);
    const double w1 = c.w - 1.0, h1 = c.h - 1.0;
    const double srcQ[4][2] = {{0, 0}, {w1, 0}, {w1, h1}, {0, h1}};
    double mats[5][3][3] = {
        {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
        {{1, 0, 1}, {0, 1, -1}, {0, 0, 1}},
        {{0.5, 0, 0}, {0, 0.5, 0}, {0, 0, 1}},
        {{0, -1, h1}, {1, 0, 0}, {0, 0, 1}},
        {{-1, 0, w1}, {0, 1, 0}, {0, 0, 1}},
    };
    const S *hsrc = s.hptr() + off_bytes<S>(c.oy, c.ox, s.step, C) / sizeof(S);
    const S *dsrc = reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, C));
    const int srcStepElems = s.step / (int)sizeof(S);
    for (int mi = 0; mi < 5; ++mi) {
      Mat3d H;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) H.m[i][j] = mats[mi][i][j];
      double dstQ[4][2];
      for (int i = 0; i < 4; ++i) {
        double den, x, y;
        mat3_map(H, srcQ[i][0], srcQ[i][1], x, y, den);
        dstQ[i][0] = x;
        dstQ[i][1] = y;
      }
      for (int interp : interps) {
        const std::vector<uint8_t> pre = d.host;
        npp_diff::clear_cuda_error();
        const NppStatus st = call(dsrc, NppiSize{c.w, c.h}, s.step, NppiRect{0, 0, c.w, c.h},
                                  reinterpret_cast<const double *>(srcQ), reinterpret_cast<S *>(d.dptr()), d.step,
                                  NppiRect{0, 0, c.w, c.h}, reinterpret_cast<const double *>(dstQ), interp, ctx);
        if (st != NPP_NO_ERROR) {
          ++cnt.skipped;
          if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
          continue;
        }
        const cudaError_t se = cudaDeviceSynchronize();
        ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
        d.dev.copyToHost(d.host);
        ++cnt.executed;
        Mat3d Fquad;
        if (!persp_quad_homography(srcQ, dstQ, Fquad)) continue;
        const Mat3d Mquad = mat3_inverse(Fquad);
        std::vector<uint8_t> want = pre;
        for (int y = 0; y < c.h; ++y)
          for (int x = 0; x < c.w; ++x) {
            double u, v, den;
            mat3_map_f32(Mquad, x, y, u, v, den);
            if (!(den > 0.0)) continue;
            if (!(u >= 0.0 && u <= (double)(c.w - 1) && v >= 0.0 && v <= (double)(c.h - 1))) continue;
            if (!persp_inside_quad_f32(srcQ, (float)u, (float)v)) continue;
            const size_t db = off_bytes<S>(y, x, d.step, C);
            for (int ch = 0; ch < A; ++ch) {
              const S val = warp_sample_channel<S>(hsrc + ch, srcStepElems, C, c.w, c.h, u, v, interp);
              std::memcpy(want.data() + db + (size_t)ch * sizeof(S), &val, sizeof(S));
            }
          }
        const GenCase dc{c.w, c.h, 0, 0, 0, kSmall, 0};
        if (!rotate_check<S, C, A>(id, dc, interp, pre, d.host, want, d.step, std::vector<uint8_t>(), cnt, first)) {
          if (cnt.mismatched >= 8) break;
        }
        if (cnt.mismatched >= 8) break;
      }
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}


// ---------------------------------------------------------------------------
// WarpPerspectiveBatch / BatchInit (packed).  Init stores the inverse of each
// coefficient matrix in aTransformedCoeffs; the batch kernel applies that
// inverse with the additional row gate measured on the batch path (rows whose
// left-edge v falls outside the source ROI are skipped).
// ---------------------------------------------------------------------------
template <typename S, int C, int A, class Call>
void run_warp_perspective_batch(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  const int interps[] = {NPPI_INTER_NN, NPPI_INTER_LINEAR, NPPI_INTER_CUBIC};
  const int B = 3, W = 33, H = 17;
  Mat3d fwd[3];
  const double f0[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  const double f1[3][3] = {{1, 0, 1}, {0, 1, -1}, {0, 0, 1}};
  const double f2[3][3] = {{0.5, 0, 0}, {0, 0.5, 0}, {0, 0, 1}};
  for (int i = 0; i < 3; ++i)
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) {
        fwd[0].m[r][c] = f0[r][c];
        fwd[1].m[r][c] = f1[r][c];
        fwd[2].m[r][c] = f2[r][c];
      }
  std::vector<Buf<S> *> srcs, dsts;
  std::vector<Buf<Npp64f> *> coeffs;
  std::vector<std::vector<uint8_t>> preHost(B);
  const uint64_t seed = case_seed(id, W, H, B, 0, 0, 0);
  for (int i = 0; i < B; ++i) {
    Buf<S> *a = new Buf<S>(W, H, C, 0, 0);
    Buf<S> *d = new Buf<S>(W, H, C, 0, 0x5A);
    Buf<Npp64f> *cf = new Buf<Npp64f>(9, 1, 1, 0, 0);
    fill_regime(*a, C, seed ^ mix64((uint64_t)i), (i & 1) ? kWide : kSmall, i);
    a->dev.copyFromHost(a->host);
    d->dev.copyFromHost(d->host);
    std::memcpy(cf->host.data(), fwd[i].m, sizeof(fwd[i].m));
    cf->dev.copyFromHost(cf->host);
    srcs.push_back(a);
    dsts.push_back(d);
    coeffs.push_back(cf);
    preHost[(size_t)i] = d->host;
  }
  std::vector<NppiWarpPerspectiveBatchCXR> h(B);
  for (int i = 0; i < B; ++i) {
    h[i] = {(const void *)srcs[(size_t)i]->dptr(), srcs[(size_t)i]->step, (void *)dsts[(size_t)i]->dptr(),
            dsts[(size_t)i]->step, (Npp64f *)coeffs[(size_t)i]->dptr(), {}};
  }
  Buf<NppiWarpPerspectiveBatchCXR> list(B, 1, 1, 0, 0);
  std::memcpy(list.host.data(), h.data(), sizeof(NppiWarpPerspectiveBatchCXR) * B);
  list.dev.copyFromHost(list.host);
  const NppStatus initSt = nppiWarpPerspectiveBatchInit_Ctx(
      reinterpret_cast<NppiWarpPerspectiveBatchCXR *>(list.dptr()), (unsigned)B, ctx);
  if (initSt != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] BatchInit failed: " << (int)initSt;
    for (auto *p : srcs) delete p;
    for (auto *p : dsts) delete p;
    for (auto *p : coeffs) delete p;
    return;
  }
  const S *hsrc[3];
  const int srcStepElems[3] = {srcs[0]->step / (int)sizeof(S), srcs[1]->step / (int)sizeof(S),
                               srcs[2]->step / (int)sizeof(S)};
  for (int i = 0; i < B; ++i) hsrc[i] = srcs[(size_t)i]->hptr();
  for (int interp : interps) {
    std::vector<std::vector<uint8_t>> pre(B);
    for (int i = 0; i < B; ++i) {
      pre[(size_t)i] = dsts[(size_t)i]->host;
      dsts[(size_t)i]->dev.copyFromHost(dsts[(size_t)i]->host);
    }
    npp_diff::clear_cuda_error();
    const NppStatus st = call(NppiSize{W, H}, NppiRect{0, 0, W, H}, NppiRect{0, 0, W, H}, interp,
                              reinterpret_cast<NppiWarpPerspectiveBatchCXR *>(list.dptr()), (unsigned)B, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] batch sync: " << cudaGetErrorString(se);
    ++cnt.executed;
    bool failed = false;
    for (int i = 0; i < B && !failed; ++i) {
      Buf<S> *d = dsts[(size_t)i];
      d->dev.copyToHost(d->host);
      const Mat3d M = mat3_inverse(fwd[i]);
      std::vector<uint8_t> want = pre[(size_t)i];
      for (int ry = 0; ry < H; ++ry) {
        const int y = ry;
        double dRow, uRow, vRow;
        mat3_map_f32(M, 0, y, uRow, vRow, dRow);
        (void)uRow;
        if (!(dRow > 0.0)) continue;
        if (vRow < 0.0 || vRow > (double)H) continue;
        for (int x = 0; x < W; ++x) {
          double u, v, den;
          mat3_map_f32(M, x, y, u, v, den);
          if (!(den > 0.0)) continue;
          if (!(u >= 0.0 && u <= (double)(W - 1) && v >= 0.0 && v <= (double)(H - 1))) continue;
          const size_t db = off_bytes<S>(y, x, d->step, C);
          for (int ch = 0; ch < A; ++ch) {
            const S vv = warp_sample_channel<S>(hsrc[i] + ch, srcStepElems[i], C, W, H, u, v, interp);
            std::memcpy(want.data() + db + (size_t)ch * sizeof(S), &vv, sizeof(S));
          }
        }
      }
      const GenCase dc{W, H, 0, 0, 0, kSmall, 0};
      if (!rotate_check<S, C, A>(id, dc, interp, pre[(size_t)i], d->host, want, d->step,
                                 std::vector<uint8_t>(), cnt, first))
        failed = true;
      if (cnt.mismatched >= 8) break;
    }
    if (failed) break;
  }
  for (auto *p : srcs) delete p;
  for (auto *p : dsts) delete p;
  for (auto *p : coeffs) delete p;
  finish(id, cnt, first);
}

template <typename S, int C, int A, class Call>
void run_warp_perspective_batch_init(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  const int B = 3;
  const double fwd[3][3][3] = {
      {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
      {{1, 0, 1}, {0, 1, -1}, {0, 0, 1}},
      {{1, 0, 0}, {0, 1, 0}, {0.002, 0, 1}},
  };
  std::vector<Buf<Npp64f> *> coeffs;
  std::vector<NppiWarpPerspectiveBatchCXR> h(B);
  for (int i = 0; i < B; ++i) {
    Buf<Npp64f> *cf = new Buf<Npp64f>(9, 1, 1, 0, 0);
    std::memcpy(cf->host.data(), fwd[i], sizeof(fwd[i]));
    cf->dev.copyFromHost(cf->host);
    coeffs.push_back(cf);
    h[i] = {nullptr, 0, nullptr, 0, (Npp64f *)cf->dptr(), {}};
  }
  Buf<NppiWarpPerspectiveBatchCXR> list(B, 1, 1, 0, 0);
  std::memcpy(list.host.data(), h.data(), sizeof(NppiWarpPerspectiveBatchCXR) * B);
  list.dev.copyFromHost(list.host);
  npp_diff::clear_cuda_error();
  const NppStatus st = call(reinterpret_cast<NppiWarpPerspectiveBatchCXR *>(list.dptr()), (unsigned)B, ctx);
  if (st != NPP_NO_ERROR) {
    ++cnt.skipped;
    if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
    for (auto *p : coeffs) delete p;
    finish(id, cnt, first);
    return;
  }
  const cudaError_t se = cudaDeviceSynchronize();
  ASSERT_EQ(se, cudaSuccess) << "[" << id << "] sync: " << cudaGetErrorString(se);
  list.dev.copyToHost(list.host);
  ++cnt.executed;
  for (int i = 0; i < B; ++i) {
    Mat3d F;
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) F.m[r][c] = fwd[i][r][c];
    const Mat3d want = mat3_inverse(F);
    const NppiWarpPerspectiveBatchCXR &e = reinterpret_cast<const NppiWarpPerspectiveBatchCXR *>(list.host.data())[i];
    bool bad = false;
    for (int r = 0; r < 3 && !bad; ++r)
      for (int c = 0; c < 3; ++c) {
        ++cnt.compared;
        const double got = e.aTransformedCoeffs[r][c];
        if (got != want.m[r][c]) {
          char b[360];
          std::snprintf(b, sizeof(b), "[%s] entry=%d coeff=%d,%d got=%.17g want=%.17g", id, i, r, c, got,
                        want.m[r][c]);
          if (first.empty()) first = b;
          EXPECT_TRUE(false) << b;
          ++cnt.mismatched;
          bad = true;
          break;
        }
      }
  }
  for (auto *p : coeffs) delete p;
  finish(id, cnt, first);
}


// ---------------------------------------------------------------------------
// Planar (P3R/P4R) warp perspective.  Each plane is an independent image:
// the same ROI, step and map apply to every plane, and sampling is per-plane
// (chStride 1) with the same float32 coordinate model.
// ---------------------------------------------------------------------------
template <typename S, int P, bool Forward, class Call>
void run_warp_perspective_planar(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  const int interps[] = {NPPI_INTER_NN, NPPI_INTER_LINEAR, NPPI_INTER_CUBIC};
  for (const GenCase &c : make_gen_cases(0)) {
    if (c.w < 2 || c.h < 2) continue;
    std::vector<Buf<S> *> srcs(P), dsts(P);
    std::vector<std::vector<uint8_t>> pre(P);
    const uint64_t seed = case_seed(id, c.w, c.h, c.pad, 0, (int)c.regime, 0);
    for (int p = 0; p < P; ++p) {
      Buf<S> *a = new Buf<S>(c.w + c.ox, c.h + c.oy, 1, c.pad, 0);
      Buf<S> *d = new Buf<S>(c.w, c.h, 1, 0, 0x5A);
      fill_regime(*a, 1, seed ^ mix64((uint64_t)p + 1), c.regime, p, false, false,
                  std::is_floating_point_v<S>);
      if constexpr (std::is_floating_point_v<S>) {
        const size_t n = (size_t)(a->step / sizeof(S)) * (size_t)a->bh;
        for (size_t i = 0; i < n; ++i) {
          const double v = (double)a->hptr()[i];
          if (v > 1e3) a->hptr()[i] = S(1e3);
          else if (v < -1e3) a->hptr()[i] = S(-1e3);
          else if (v != v) a->hptr()[i] = S(0);
        }
      }
      a->dev.copyFromHost(a->host);
      d->dev.copyFromHost(d->host);
      srcs[(size_t)p] = a;
      dsts[(size_t)p] = d;
      pre[(size_t)p] = d->host;
    }
    const double w1 = c.w - 1.0, h1 = c.h - 1.0;
    double coeffs[6][3][3] = {
        {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
        {{1, 0, 1}, {0, 1, -1}, {0, 0, 1}},
        {{2, 0, 0}, {0, 2, 0}, {0, 0, 1}},
        {{0, -1, h1}, {1, 0, 0}, {0, 0, 1}},
        {{-1, 0, w1}, {0, 1, 0}, {0, 0, 1}},
        {{1, 0, 0}, {0, 1, 0}, {0.002, 0, 1}},
    };
    const S *hsrc[4] = {nullptr, nullptr, nullptr, nullptr};
    const S *dsrc[4] = {nullptr, nullptr, nullptr, nullptr};
    S *hdst[4] = {nullptr, nullptr, nullptr, nullptr};
    S *ddst[4] = {nullptr, nullptr, nullptr, nullptr};
    int sstep[4] = {0, 0, 0, 0};
    for (int p = 0; p < P; ++p) {
      hsrc[p] = srcs[(size_t)p]->hptr() + off_bytes<S>(c.oy, c.ox, srcs[(size_t)p]->step, 1) / sizeof(S);
      dsrc[p] = reinterpret_cast<const S *>(srcs[(size_t)p]->dptr() +
                                            off_bytes<S>(c.oy, c.ox, srcs[(size_t)p]->step, 1));
      hdst[p] = dsts[(size_t)p]->hptr();
      ddst[p] = reinterpret_cast<S *>(dsts[(size_t)p]->dptr());
      sstep[p] = srcs[(size_t)p]->step / (int)sizeof(S);
    }
    const int dstep = dsts[0]->step / (int)sizeof(S);
    for (int mi = 0; mi < 6; ++mi) {
      const auto &cf = coeffs[mi];
      for (int interp : interps) {
        for (int p = 0; p < P; ++p) dsts[(size_t)p]->dev.copyFromHost(pre[(size_t)p]);
        npp_diff::clear_cuda_error();
        const NppStatus st = call(dsrc, NppiSize{c.w, c.h}, srcs[0]->step, NppiRect{0, 0, c.w, c.h}, ddst,
                                  dsts[0]->step, NppiRect{0, 0, c.w, c.h},
                                  reinterpret_cast<const double *>(cf), interp, ctx);
        if (st != NPP_NO_ERROR) {
          ++cnt.skipped;
          if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
          continue;
        }
        const cudaError_t se = cudaDeviceSynchronize();
        ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
        ++cnt.executed;
        Mat3d M3, F3;
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j) {
            M3.m[i][j] = cf[i][j];
            F3.m[i][j] = cf[i][j];
          }
        if constexpr (Forward) {
          M3 = mat3_inverse(F3);
        } else {
          F3 = mat3_inverse(M3);
        }
        (void)F3;
        bool failed = false;
        for (int p = 0; p < P && !failed; ++p) {
          dsts[(size_t)p]->dev.copyToHost(dsts[(size_t)p]->host);
          std::vector<uint8_t> want = pre[(size_t)p];
          for (int y = 0; y < c.h; ++y)
            for (int x = 0; x < c.w; ++x) {
              double u, v;
              if (!persp_pixel_gate(M3, NppiRect{0, 0, c.w, c.h}, x, y, u, v)) continue;
              const size_t db = off_bytes<S>(y, x, dsts[(size_t)p]->step, 1);
              const S val = warp_sample_channel<S>(hsrc[p], sstep[p], 1, c.w, c.h, u, v, interp);
              std::memcpy(want.data() + db, &val, sizeof(S));
            }
          const GenCase dc{c.w, c.h, 0, 0, 0, kSmall, 0};
          if (!rotate_check<S, 1, 1>(id, dc, interp, pre[(size_t)p], dsts[(size_t)p]->host, want,
                                     dsts[(size_t)p]->step, std::vector<uint8_t>(), cnt, first))
            failed = true;
          if (cnt.mismatched >= 8) break;
        }
        if (failed) break;
        (void)hdst;
        (void)dstep;
      }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
    for (auto *p : srcs) delete p;
    for (auto *p : dsts) delete p;
  }
  finish(id, cnt, first);
}

template <typename S, int P, class Call>
void run_warp_perspective_quad_planar(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  const int interps[] = {NPPI_INTER_NN, NPPI_INTER_LINEAR, NPPI_INTER_CUBIC};
  for (const GenCase &c : make_gen_cases(0)) {
    if (c.w < 2 || c.h < 2) continue;
    std::vector<Buf<S> *> srcs(P), dsts(P);
    std::vector<std::vector<uint8_t>> pre(P);
    const uint64_t seed = case_seed(id, c.w, c.h, c.pad, 0, (int)c.regime, 0);
    for (int p = 0; p < P; ++p) {
      Buf<S> *a = new Buf<S>(c.w + c.ox, c.h + c.oy, 1, c.pad, 0);
      Buf<S> *d = new Buf<S>(c.w, c.h, 1, 0, 0x5A);
      fill_regime(*a, 1, seed ^ mix64((uint64_t)p + 1), c.regime, p, false, false,
                  std::is_floating_point_v<S>);
      if constexpr (std::is_floating_point_v<S>) {
        const size_t n = (size_t)(a->step / sizeof(S)) * (size_t)a->bh;
        for (size_t i = 0; i < n; ++i) {
          const double v = (double)a->hptr()[i];
          if (v > 1e3) a->hptr()[i] = S(1e3);
          else if (v < -1e3) a->hptr()[i] = S(-1e3);
          else if (v != v) a->hptr()[i] = S(0);
        }
      }
      a->dev.copyFromHost(a->host);
      d->dev.copyFromHost(d->host);
      srcs[(size_t)p] = a;
      dsts[(size_t)p] = d;
      pre[(size_t)p] = d->host;
    }
    const double w1 = c.w - 1.0, h1 = c.h - 1.0;
    const double srcQ[4][2] = {{0, 0}, {w1, 0}, {w1, h1}, {0, h1}};
    double mats[5][3][3] = {
        {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
        {{1, 0, 1}, {0, 1, -1}, {0, 0, 1}},
        {{0.5, 0, 0}, {0, 0.5, 0}, {0, 0, 1}},
        {{0, -1, h1}, {1, 0, 0}, {0, 0, 1}},
        {{-1, 0, w1}, {0, 1, 0}, {0, 0, 1}},
    };
    const S *hsrc[4] = {nullptr, nullptr, nullptr, nullptr};
    const S *dsrc[4] = {nullptr, nullptr, nullptr, nullptr};
    S *ddst[4] = {nullptr, nullptr, nullptr, nullptr};
    int sstep[4] = {0, 0, 0, 0};
    for (int p = 0; p < P; ++p) {
      hsrc[p] = srcs[(size_t)p]->hptr() + off_bytes<S>(c.oy, c.ox, srcs[(size_t)p]->step, 1) / sizeof(S);
      dsrc[p] = reinterpret_cast<const S *>(srcs[(size_t)p]->dptr() +
                                            off_bytes<S>(c.oy, c.ox, srcs[(size_t)p]->step, 1));
      ddst[p] = reinterpret_cast<S *>(dsts[(size_t)p]->dptr());
      sstep[p] = srcs[(size_t)p]->step / (int)sizeof(S);
    }
    for (int mi = 0; mi < 5; ++mi) {
      Mat3d H;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) H.m[i][j] = mats[mi][i][j];
      double dstQ[4][2];
      for (int i = 0; i < 4; ++i) {
        double den, x, y;
        mat3_map(H, srcQ[i][0], srcQ[i][1], x, y, den);
        dstQ[i][0] = x;
        dstQ[i][1] = y;
      }
      for (int interp : interps) {
        for (int p = 0; p < P; ++p) dsts[(size_t)p]->dev.copyFromHost(pre[(size_t)p]);
        npp_diff::clear_cuda_error();
        const NppStatus st = call(dsrc, NppiSize{c.w, c.h}, srcs[0]->step, NppiRect{0, 0, c.w, c.h},
                                  reinterpret_cast<const double *>(srcQ), ddst, dsts[0]->step,
                                  NppiRect{0, 0, c.w, c.h}, reinterpret_cast<const double *>(dstQ), interp, ctx);
        if (st != NPP_NO_ERROR) {
          ++cnt.skipped;
          if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
          continue;
        }
        const cudaError_t se = cudaDeviceSynchronize();
        ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
        ++cnt.executed;
        Mat3d Fquad;
        if (!persp_quad_homography(srcQ, dstQ, Fquad)) continue;
        const Mat3d Mquad = mat3_inverse(Fquad);
        bool failed = false;
        for (int p = 0; p < P && !failed; ++p) {
          dsts[(size_t)p]->dev.copyToHost(dsts[(size_t)p]->host);
          std::vector<uint8_t> want = pre[(size_t)p];
          for (int y = 0; y < c.h; ++y)
            for (int x = 0; x < c.w; ++x) {
              double u, v, den;
              mat3_map_f32(Mquad, x, y, u, v, den);
              if (!(den > 0.0)) continue;
              if (!(u >= 0.0 && u <= (double)(c.w - 1) && v >= 0.0 && v <= (double)(c.h - 1))) continue;
              if (!persp_inside_quad_f32(srcQ, (float)u, (float)v)) continue;
              const size_t db = off_bytes<S>(y, x, dsts[(size_t)p]->step, 1);
              const S val = warp_sample_channel<S>(hsrc[p], sstep[p], 1, c.w, c.h, u, v, interp);
              std::memcpy(want.data() + db, &val, sizeof(S));
            }
          const GenCase dc{c.w, c.h, 0, 0, 0, kSmall, 0};
          if (!rotate_check<S, 1, 1>(id, dc, interp, pre[(size_t)p], dsts[(size_t)p]->host, want,
                                     dsts[(size_t)p]->step, std::vector<uint8_t>(), cnt, first))
            failed = true;
          if (cnt.mismatched >= 8) break;
        }
        if (failed) break;
      }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
    for (auto *p : srcs) delete p;
    for (auto *p : dsts) delete p;
  }
  finish(id, cnt, first);
}

}  // namespace num
