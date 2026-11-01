# g5 report — nppi_geometry_transforms (perspective / rotate)

Branch: `pkg/g5` (not pushed).  Baseline: NVIDIA NPP 13.4 on the RTX 5090.

## 1. Family / function inventory

`api_analysis/packages/g5.csv` lists 204 functions; all of them are now
`real` (`api_analysis/api_catalog.csv`, refreshed):

| family | functions | status before | now |
|---|---:|---|---|
| `nppiWarpPerspective` | 54 | 38 real, 16 P3/P4 missing | all real (regenerated wrappers) |
| `nppiWarpPerspectiveBack` | 48 | 32 real, 16 P3/P4 missing | all real |
| `nppiWarpPerspectiveQuad` | 48 | all missing | all real |
| `nppiWarpPerspectiveBatch` | 22 | all missing | all real |
| `nppiRotate` | 24 | 18 real (NN only, wrong semantics), 6 AC4 stubs | all real |
| `nppiGet{Affine,Perspective}*` / `nppiGetRotate*` | 8 | real (landed earlier) | unchanged |

108 functions were implemented in this round; the 8 host helpers came from the
earlier `nppi_get_transform.cpp` work.  `nppiWarpPerspectiveBatchInit` (not in
the package list, but required by the batch tests) is generated too.

## 2. What was built

* `src/nppi/nppi_geometry_transforms/nppi_warp_perspective_executor.{h,cu}` –
  packed (C1/C3/C4/AC4, 8u/16u/32s/32f/16f) and planar (P3/P4) kernels for
  forward, backward, quad and batch plus `BatchInit`, with the measured
  validation order and error codes.
* `src/nppi/nppi_geometry_transforms/nppi_warp_sampling.cuh` – NN / bilinear /
  four-point cubic Lagrange sampling, image-bound tap clamping, integer
  round-half-up with saturation, Npp16f through `__half`, AC4 alpha preserved.
* `src/nppi/nppi_geometry_transforms/nppi_rotate_executor.{h,cu}` – float32
  rotation mapping, measured acceptance box, no fill, AC4 alpha preserved and
  the NPP even-step rule.
* Six family specs under `mpp_scripts/families/` (`nppi_warp_perspective*`,
  `nppi_rotate`) and their `gen_family.py`-generated wrappers; the legacy
  hand-written `nppi_warp_perspective.{cpp,cu}` and `nppi_rotate.{cpp,cu}` are
  removed and the six rotate AC4 stubs are pruned from `nppi_not_support.cpp`.
* `docs/agent_packages/g5_notes.md` records the measured semantics (added
  addendum: forward-vs-batch row gate, cubic Lagrange weights, quad region and
  knife-edge behaviour, rotate rules, contract-code table).

## 3. 5090 measurement and verification

Recording (NVIDIA reference, `mcpp-5090` MCP, remote `~/mcMPP-g5`):

* `diff_tests --gtest_filter='NppiWarpPerspective*:NppiRotateDiff*'` in record
  mode: **62 tests, 62 passed, 0 crashes**; 453 goldens produced.
* `artifacts/5090/npp_records/{nppiWarpPerspective,nppiWarpPerspectiveBatch,nppiWarpPerspectiveQuad,nppiRotate}`
  refreshed from that recording (the batch/planar/rotate goldens are
  byte-identical to the previous ones apart from input hashes; the quad goldens
  changed because the probe geometry was shifted, see §4).

mcMPP-side verification: the worktree cannot build, so the executor, the
generated wrappers and the four diff suites were compiled on the 5090 with
`nvcc` against the repository headers (`API-unified`) and run against the
recorded goldens (`NPP_VERIFY_DIR`):

* **66/66 differential tests pass, three consecutive runs** (24 quad + 15 batch
  + 20 planar + 6 rotate, including the non-Ctx cases that only exist on the
  mcMPP side).
* Extra cross-check of the *packed* forward/backward entry points (not covered
  by a diff suite): C1R identity / scale2 / persp NN / LINEAR / CUBIC forward
  and identity / scale2 / persp NN / LINEAR backward reproduce the recorded
  planar goldens byte for byte.

## 4. Test and harness changes (no coverage removed)

1. Quad probe geometries shifted by `(0.31, 0.17)` (both quads together, so all
   mappings stay identical).  With integer corners the projected vertices land
   exactly on pixel centres and NVIDIA's own decision flips on the last bit of
   its coefficient arithmetic (two `kQuadPersp` vertices skipped, the
   `kQuadSame` vertices written).  Every edge decision is now ≥ 5e-3 px away
   from a pixel centre.
2. `NonCtx_Batch` now records the same outputs as the `_Ctx` case it mirrors
   (status plus the post-Init and post-call `aTransformedCoeffs`), which its
   golden id requires.
3. `test/unit/framework/npp_test_base.h`: `DeviceMemory(size_t)` left `ptr_`
   uninitialised before `allocate()` called `free()`; the stale pointer could
   name a live device buffer and produced a sticky "illegal memory access" in
   asynchronous runs.

## 5. Measured semantics (summary)

See `docs/agent_packages/g5_notes.md` for the full table; the essentials:

* forward (single image): x span of the source-ROI image at the row's left-edge
  `v`, then the per-pixel gate `den > 0`, `u,v` inside the source ROI rect
  (inclusive of `x+w-1`); batch additionally drops rows whose left-edge `v`
  leaves the ROI; backward is per-pixel only;
* quad: src->dst homography solved from the eight vertex equations, write
  region = image of the source quadrilateral;
* rotate: float32 mapping, write iff `roi-0.5 <= f <= roi+size-1`, no fill;
* interpolation: NN `floor(u+0.5)`, bilinear, four-point cubic Lagrange
  (`floor-1..floor+2`), taps clamped to the image; integers round half up and
  saturate, AC4 alpha untouched;
* contract codes per family are tabulated in the notes.

## 6. Not covered / residual risk

* Local (C500/MACA) build and the legacy unit tests could not be run here (the
  worktree cannot configure the cu-bridge toolchain).  The lead's C500
  verification is the first real mcMPP build.
* `Npp16f` LINEAR/CUBIC warps are implemented but only NN is covered by a
  golden (the batch 16f cases are NN-only); the packed non-planar
  forward/backward paths are covered by the cross-check described in §3 rather
  than by a recorded suite.
* The 32f AC4 rotate cases record `NPP_NOT_EVEN_STEP_ERROR` (the test's
  392-byte step is not 16-byte aligned), so they only pin the status.

## 7. Reproducing the 5090 verification

Recording (NVIDIA reference; `mcpp-5090` MCP, `remote_dir=~/mcMPP-g5`):

```bash
mcp__mcpp-5090__sync   repo_path=<worktree> remote_dir=/home/metax/mcMPP-g5
mcp__mcpp-5090__build  remote_dir=/home/metax/mcMPP-g5 jobs=4
mcp__mcpp-5090__record remote_dir=/home/metax/mcMPP-g5 target=diff_tests \
                       filter='NppiWarpPerspective*:NppiRotateDiff*' out_dir=npp_records_g5
```

mcMPP-side verification (no local build): the executor, the six generated
wrappers, a four-line `nppGetStreamContext` stub and the four diff suites are
compiled with `nvcc -arch=sm_89 -I API-unified -I src/include` and linked
against `libgtest`; the resulting binary is run with
`NPP_VERIFY_DIR=<record dir>` and passes 66/66 tests.  The stub, the harness
script and the packed cross-check driver live outside the repository
(`/home/metax/g5dev`, `/home/metax/g5probe`) and are only developer tooling.
