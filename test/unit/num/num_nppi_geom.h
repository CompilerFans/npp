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
          ok = (g > w ? g - w : w - g) <= (long long)atol;
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

}  // namespace num
