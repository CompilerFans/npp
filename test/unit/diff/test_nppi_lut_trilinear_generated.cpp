// Differential tests for package c1: nppiLUT_Trilinear (8u C4 / AC4 / AC4IR).
//
// pValues holds aLevels[2] (z / blue) planes, each aLevels[1] (y / green) rows
// of aLevels[0] (x / red) entries; every entry is one 32-bit word whose low 3
// bytes are the B, G, R output.  The cases below use entries whose bytes are
// independent (R=x*85, G=y*85, B=z*85) so the recorded golden reveals both the
// axis order and the pValues layout.  The 3-level axes case pins the indexing
// with distinct prime factors.

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <memory>
#include <vector>

// NOTE: in this family pLevels is a HOST pointer to an array of 3 HOST
// pointers (the level values are read on the host by NVIDIA NPP), while
// pValues is a DEVICE pointer.  Passing device pointers for pLevels crashes
// real NPP with a segfault.

using namespace npp_diff;
using namespace npp_functional_test;
namespace {

template <typename T> DeviceMemory<T> &dev_buf(const std::vector<T> &h) {
  static std::vector<std::unique_ptr<DeviceMemory<T>>> arena;
  arena.push_back(std::make_unique<DeviceMemory<T>>(h.size()));
  DeviceMemory<T> &d = *arena.back();
  if (!h.empty()) d.copyFromHost(h);
  return d;
}

NppStreamContext current_ctx() {
  NppStreamContext c{};
  NppStatus st = nppGetStreamContext(&c);
  EXPECT_EQ(st, NPP_NO_ERROR);
  return c;
}

int align_step(int step, int es) { return ((step + es - 1) / es) * es; }

// Build the packed pValues array: value(x,y,z) = R | (G << 8) | (B << 16)
std::vector<Npp32u> make_values(const int aLevels[3], unsigned r, unsigned g, unsigned b) {
  std::vector<Npp32u> v;
  for (int z = 0; z < aLevels[2]; ++z)
    for (int y = 0; y < aLevels[1]; ++y)
      for (int x = 0; x < aLevels[0]; ++x) v.push_back(r * x | (g * y) << 8 | (b * z) << 16);
  return v;
}

struct TriRunner {
  const char *fname;
  std::string key;
  int width, height, pad;
  std::vector<Npp8u> src;
  int aLevels[3];
  std::vector<Npp32u> values;
  std::vector<Npp8u> levels0, levels1, levels2;
  bool inPlace = false;
  bool useC4 = false;  // C4R (alpha mapped) vs AC4R (alpha untouched)

  void run() {
    const int px = 4;
    const int step = align_step(width * px + pad, px);
    const size_t n = (size_t)height * step;
    std::vector<Npp8u> hs(src);
    hs.resize(n);
    std::vector<Npp8u> hd = make_uniform<Npp8u>(n, 0xA5);
    DeviceMemory<Npp8u> &s = dev_buf(hs);
    DeviceMemory<Npp8u> &d = inPlace ? s : dev_buf(hd);
    DeviceMemory<Npp32u> &pv = dev_buf(values);
    Npp8u *pl[3] = {levels0.data(), levels1.data(), levels2.data()};
    int al[3] = {aLevels[0], aLevels[1], aLevels[2]};
    Case c("nppiLUT_trilinear/" + key, fname);
    c.input(hs.data(), hs.size());
    if (inPlace) {
      clear_cuda_error();
      c.status(nppiLUT_Trilinear_8u_AC4IR_Ctx(s.get(), step, NppiSize{width, height}, pv.get(), pl, al,
                                              current_ctx()));
    } else if (useC4) {
      clear_cuda_error();
      c.status(nppiLUT_Trilinear_8u_C4R_Ctx(s.get(), step, d.get(), step, NppiSize{width, height}, pv.get(), pl,
                                            al, current_ctx()));
    } else {
      clear_cuda_error();
      c.status(nppiLUT_Trilinear_8u_AC4R_Ctx(s.get(), step, d.get(), step, NppiSize{width, height}, pv.get(), pl,
                                             al, current_ctx()));
    }
    c.device_output<Npp8u>(d.get(), n, ValueKind::U8);
    c.finish();
  }
};

}  // namespace

class NppiLutTrilinearDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

// A full 2x2x2 grid over levels {0,255} and every 8-bit input ramp: the golden
// directly shows the blend weights and the RGB byte order.
TEST_F(NppiLutTrilinearDiff, Trilinear_Grid2) {
  std::vector<Npp8u> src;
  for (int v = 0; v < 256; ++v) {
    src.push_back((Npp8u)v);           // R
    src.push_back((Npp8u)v);           // G
    src.push_back((Npp8u)v);           // B
    src.push_back(0xC3);               // A (must be untouched)
  }
  TriRunner r;
  r.fname = "nppiLUT_Trilinear_8u_AC4R";
  r.key = "grid2";
  r.width = 256; r.height = 1; r.pad = 0;
  r.src = src;
  r.aLevels[0] = 2; r.aLevels[1] = 2; r.aLevels[2] = 2;
  r.values = make_values(r.aLevels, 85, 85, 85);
  r.levels0 = {0, 255};
  r.levels1 = {0, 255};
  r.levels2 = {0, 255};
  r.run();

  // corners + interior points with non-equal R/G/B
  TriRunner r2;
  r2.fname = "nppiLUT_Trilinear_8u_AC4R";
  r2.key = "grid2_mixed";
  r2.width = 32; r2.height = 1; r2.pad = 4;
  std::vector<Npp8u> mixed;
  const int pts[8][3] = {{0, 0, 0}, {255, 0, 0}, {0, 255, 0}, {0, 0, 255},
                         {255, 255, 255}, {128, 64, 192}, {13, 200, 77}, {250, 5, 128}};
  for (auto &p : pts) {
    mixed.push_back((Npp8u)p[0]);
    mixed.push_back((Npp8u)p[1]);
    mixed.push_back((Npp8u)p[2]);
    mixed.push_back(0x5A);
  }
  r2.src = mixed;
  r2.aLevels[0] = 2; r2.aLevels[1] = 2; r2.aLevels[2] = 2;
  r2.values = make_values(r2.aLevels, 85, 85, 85);
  r2.levels0 = {0, 255};
  r2.levels1 = {0, 255};
  r2.levels2 = {0, 255};
  r2.run();

  // in-place variant
  TriRunner r3;
  r3.fname = "nppiLUT_Trilinear_8u_AC4IR";
  r3.key = "grid2_ip";
  r3.width = 32; r3.height = 1; r3.pad = 4;
  r3.src = mixed;
  r3.aLevels[0] = 2; r3.aLevels[1] = 2; r3.aLevels[2] = 2;
  r3.values = make_values(r3.aLevels, 85, 85, 85);
  r3.levels0 = {0, 255};
  r3.levels1 = {0, 255};
  r3.levels2 = {0, 255};
  r3.inPlace = true;
  r3.run();
}

// 3 levels per axis: pins pValues indexing (x fastest, then y, then z) and the
// interval-search/interpolation between non-uniform levels.
TEST_F(NppiLutTrilinearDiff, Trilinear_Grid3) {
  std::vector<Npp8u> src;
  for (int z = 0; z < 3; ++z)
    for (int y = 0; y < 3; ++y)
      for (int x = 0; x < 3; ++x) {
        src.push_back((Npp8u)(x * 100));
        src.push_back((Npp8u)(y * 100));
        src.push_back((Npp8u)(z * 100));
        src.push_back(0x77);
        // midpoint between grid nodes
        src.push_back((Npp8u)(x * 100 + 50));
        src.push_back((Npp8u)(y * 100 + 50));
        src.push_back((Npp8u)(z * 100 + 50));
        src.push_back(0x78);
      }
  TriRunner r;
  r.fname = "nppiLUT_Trilinear_8u_AC4R";
  r.key = "grid3";
  r.width = 54; r.height = 1; r.pad = 8;
  r.src = src;
  r.aLevels[0] = 3; r.aLevels[1] = 3; r.aLevels[2] = 3;
  r.values = make_values(r.aLevels, 1, 1, 1);  // R=x, G=y, B=z
  r.levels0 = {0, 100, 200};
  r.levels1 = {0, 100, 200};
  r.levels2 = {0, 100, 200};
  r.run();
}

// C4R (alpha is also mapped through the 4th byte if present) and a random
// 2D ROI with padding.
TEST_F(NppiLutTrilinearDiff, Trilinear_C4R_Random) {
  auto rnd = make_uniform<Npp8u>(17 * 13 * 4, 900);
  const int aLevels[3] = {3, 3, 3};
  TriRunner r;
  r.fname = "nppiLUT_Trilinear_8u_C4R";
  r.key = "c4_random";
  r.useC4 = true;
  r.width = 17; r.height = 13; r.pad = 6;
  r.src = rnd;
  r.aLevels[0] = 3; r.aLevels[1] = 3; r.aLevels[2] = 3;
  r.values = make_values(r.aLevels, 23, 41, 59);
  r.levels0 = {0, 120, 250};
  r.levels1 = {10, 130, 240};
  r.levels2 = {20, 110, 220};
  r.run();
  // 4th byte of pValues entries must be ignored: set it to garbage.
  std::vector<Npp32u> v = make_values(r.aLevels, 23, 41, 59);
  for (auto &e : v) e |= 0xAB000000u;
  TriRunner r2 = r;
  r2.key = "c4_random_highbyte";
  r2.values = v;
  r2.run();
}

TEST_F(NppiLutTrilinearDiff, Trilinear_Contract) {
  auto run = [&](const std::string &key, NppiSize roi) {
    const int aLevels[3] = {2, 2, 2};
    std::vector<Npp8u> hs = make_uniform<Npp8u>(4096, 901);
    std::vector<Npp8u> hd = make_uniform<Npp8u>(4096, 902);
    DeviceMemory<Npp8u> &s = dev_buf(hs);
    DeviceMemory<Npp8u> &d = dev_buf(hd);
    std::vector<Npp32u> pv = make_values(aLevels, 85, 85, 85);
    DeviceMemory<Npp32u> &pvd = dev_buf(pv);
    std::vector<Npp8u> l0 = {0, 255}, l1 = {0, 255}, l2 = {0, 255};
    Npp8u *pl[3] = {l0.data(), l1.data(), l2.data()};
    int al[3] = {2, 2, 2};
    clear_cuda_error();
    NppStatus st = nppiLUT_Trilinear_8u_AC4R_Ctx(s.get(), 68, d.get(), 68, roi, pvd.get(), pl, al, current_ctx());
    Case c("nppiLUT_trilinear/contract/" + key, "nppiLUT_Trilinear_8u_AC4R");
    c.status(st);
    c.finish();
  };
  run("emptyRoi", NppiSize{0, 4});
  run("negRoi", NppiSize{-4, 4});
  run("zeroRoi", NppiSize{0, 0});
  run("h0", NppiSize{16, 0});
  // level count above 256 on one axis
  {
    std::vector<Npp8u> hs = make_uniform<Npp8u>(4096, 905);
    std::vector<Npp8u> hd = make_uniform<Npp8u>(4096, 906);
    DeviceMemory<Npp8u> &s = dev_buf(hs);
    DeviceMemory<Npp8u> &d = dev_buf(hd);
    int al[3] = {2, 257, 2};
    std::vector<Npp32u> pv = make_values(al, 85, 85, 85);
    DeviceMemory<Npp32u> &pvd = dev_buf(pv);
    std::vector<Npp8u> l0 = {0, 255}, l1(257), l2 = {0, 255};
    for (int i = 0; i < 257; ++i) l1[i] = static_cast<Npp8u>(i);
    Npp8u *pl[3] = {l0.data(), l1.data(), l2.data()};
    clear_cuda_error();
    NppStatus st = nppiLUT_Trilinear_8u_AC4R_Ctx(s.get(), 68, d.get(), 68, NppiSize{16, 4}, pvd.get(), pl, al,
                                                 current_ctx());
    Case c("nppiLUT_trilinear/contract/levels257", "nppiLUT_Trilinear_8u_AC4R");
    c.status(st);
    c.finish();
  }
  // axis with a single level
  {
    const int aLevels[3] = {2, 1, 2};
    std::vector<Npp8u> hs = make_uniform<Npp8u>(4096, 903);
    std::vector<Npp8u> hd = make_uniform<Npp8u>(4096, 904);
    DeviceMemory<Npp8u> &s = dev_buf(hs);
    DeviceMemory<Npp8u> &d = dev_buf(hd);
    std::vector<Npp32u> pv = make_values(aLevels, 85, 85, 85);
    DeviceMemory<Npp32u> &pvd = dev_buf(pv);
    std::vector<Npp8u> l0 = {0, 255}, l1 = {128}, l2 = {0, 255};
    Npp8u *pl[3] = {l0.data(), l1.data(), l2.data()};
    int al[3] = {2, 1, 2};
    clear_cuda_error();
    NppStatus st = nppiLUT_Trilinear_8u_AC4R_Ctx(s.get(), 68, d.get(), 68, NppiSize{16, 4}, pvd.get(), pl, al,
                                                 current_ctx());
    Case c("nppiLUT_trilinear/contract/axis1level", "nppiLUT_Trilinear_8u_AC4R");
    c.status(st);
    c.finish();
  }
}
