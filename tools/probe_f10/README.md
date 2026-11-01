# f10 probe: NPP 13.4 semantics for nppi_filtering_functions.h (Gauss / GaussAdvanced / Wiener)

`probe.cu` is a standalone program compiled and run on the RTX 5090 host
(`nvcc -O1 -std=c++17 -I/usr/local/cuda/include probe.cu -o probe -lnppif -lnppc -lcudart`).
It calls the real NVIDIA NPP 13.4 filtering functions with controlled inputs and
prints raw result grids / status codes, so observed behaviour (never the docs) is
the contract implemented in src/.

Stages (selected by argv[1]):
  gauss32 / gauss8  - nppiFilterGauss_32f_C1R / _8u_C1R over mask sizes 3..15
  edge / edge2 / edge3 - ROI/step/border edge behaviour for GaussBorder/AdvancedBorder
  gborder           - GaussBorder border/mask matrix (status codes)
  adv / adv8        - GaussAdvanced kernel taps, normalisation, reversal, truncation
  advb              - GaussAdvancedBorder: MIRROR (reflect-101) vs REPLICATE, taps
  round             - rounding of 8u/16s outputs on exact .5 inputs
  wiener            - WienerBorder anchor/window convention, noise range, mask matrix
  wiener2           - WienerBorder full 7x7 grids for mask/anchor/noise combos

`fit/` holds captured probe output and small python analysis scripts used to
reverse-engineer the Wiener mean/variance computation:
  wiener2.txt  - captured `wiener2` stage output (131 lines)
  fit_wiener.py, fit2.py, fit3.py - hypothesis search over window/stat models

Findings are summarised in docs/agent_packages/f10-notes.md as they are pinned down.

## f10 implementation findings (2026-11-01)

Additional probes used to finish the package (all run on the RTX 5090):

| file | purpose |
|---|---|
| `fit/gauss_arith.cu` | extracts the exact fixed Gauss 2D kernels, compares float/double/FMA accumulation and kernel-generation variants, and validates integer emission for masks 3..15 |
| `fit/valprobe.cu`, `fit/wiener_order.cu` | validation order / status matrices for all five families |
| `fit/wiener_shape.cu`, `fit/wiener_sets.cu` | recovers the Wiener statistics window by impulse-response support and by classifying the mean set A / variance set B |
| `fit/wiener_h1.cu` | shows the h==1 mask quirk (mean row ay, variance row ay-1) |
| `fit/wiener_fit2.cu`, `fit/wiener_fit3.cu` | validates the Wiener formula, integer noise scaling and emission for 8u/16s |
| `fit/wiener_c4.cu`, `fit/wiener_ac4.cu`, `fit/wiener_ac4b.cu` | C4 alpha-channel noise and the AC4 rectangle-window path |
| `fit/wval_*.{cu,cpp,h}` | compiles the real mcMPP executors with the repository headers next to NVIDIA NPP and diffs statuses + outputs over a wide matrix (468 positive cases + status matrix; reproducible with `nvcc`/`g++`, see the file headers) |

Pinned semantics:

- `nppiFilterGauss`: square masks 3..15 only; 1D Gaussian sigma = 0.6 + 0.4*(taps/2), 2D outer product, correlation origin (taps-1)/2, float32 FMA, integer truncation + saturation. Plain reads raw margins; Border clamps (REPLICATE only).
- `nppiFilterGaussAdvanced`: odd taps >= 3, user kernel outer product; empty-ROI tap quirks and the OUT_OFF_RANGE status for negative taps.
- `nppiFilterWienerBorder`: quirky window (h==1 mean/variance rows differ; otherwise h-1 rows, first row w columns, later rows w-1 columns); AC4 instead uses the plain w x h rectangle. mean/var are normalised by w*h (variance can be negative); aNoise==0 emits the variance; integer dtypes scale the noise by the dynamic range squared; integer output truncates and saturates.
- Error-code values differ between the repo headers and NVIDIA NPP 13.4 (`NPP_NOT_SUPPORTED_MODE_ERROR`, `NPP_STEP_ERROR`, ...); the differential harness compares status *names*, so never hard-code numeric codes in tests.
