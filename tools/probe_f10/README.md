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
