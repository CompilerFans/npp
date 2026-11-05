#!/usr/bin/env python3
"""Generate CPU-golden tests for the nppi geometry transforms.

The first batch covers nppiMirror and nppiMirrorBatch; the same generator is
extended as the resize/remap/warp families are pinned.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Optional, List

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "api_analysis" / "api_catalog.csv"
CTYPE = {"8u": "Npp8u", "16u": "Npp16u", "16s": "Npp16s", "32s": "Npp32s",
         "16f": "Npp16f", "32f": "Npp32f", "64f": "Npp64f"}
LAYOUT = {"C1": (1, 1), "C3": (3, 3), "C4": (4, 4), "AC4": (4, 3)}


def emit_mirror(row: dict) -> Optional[str]:
    fn = row["function_name"]
    name = re.sub(r"\W", "_", fn)
    m = re.search(r"_(AC4|C1|C3|C4)(I?)R_Ctx$", fn)
    if m is None:
        return None
    tok, inplace = m.group(1), m.group(2) == "I"
    if tok not in LAYOUT:
        return None
    C, A = LAYOUT[tok]
    S = CTYPE.get(row["dtype"])
    if S is None:
        return None
    if inplace:
        lam = (f"[]({S} *sd, int st, NppiSize r, NppiAxis f, NppStreamContext c) "
               f"{{ (void)st; (void)r; (void)f; (void)c; return {fn}(sd, st, r, f, c); }}")
        prefix = f"num::run_mirror<{S}, {C}, {A}, true>"
    else:
        lam = (f"[](const {S} *a, int sa, {S} *d, int ds, NppiSize r, NppiAxis f, "
               f"NppStreamContext c) "
               f"{{ (void)sa; (void)ds; (void)r; (void)f; (void)c; "
               f"return {fn}(a, sa, d, ds, r, f, c); }}")
        prefix = f"num::run_mirror<{S}, {C}, {A}, false>"
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")


def emit_mirror_batch(row: dict) -> Optional[str]:
    fn = row["function_name"]
    name = re.sub(r"\W", "_", fn)
    m = re.search(r"_(AC4|C1|C3|C4)(I?)R_Ctx$", fn)
    if m is None:
        return None
    tok, inplace = m.group(1), m.group(2) == "I"
    if tok not in LAYOUT:
        return None
    C, A = LAYOUT[tok]
    S = CTYPE.get(row["dtype"])
    if S is None:
        return None
    lam = (f"[](NppiSize r, NppiAxis f, NppiMirrorBatchCXR *b, int nb, NppStreamContext c) "
           f"{{ (void)r; (void)f; (void)b; (void)nb; (void)c; "
           f"return {fn}(r, f, b, nb, c); }}")
    prefix = f"num::run_mirror_batch<{S}, {C}, {A}, {str(inplace).lower()}>"
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")



def _layout(fn: str):
    m = re.search(r"_(AC4|C1|C3|C4)R_Ctx$", fn)
    if m is None:
        return None
    tok = m.group(1)
    return tok, LAYOUT[tok]


def emit_resize(row: dict) -> Optional[str]:
    fn = row["function_name"]
    lay = _layout(fn)
    S = CTYPE.get(row["dtype"])
    if lay is None or S is None:
        return None
    C, A = lay[1]
    name = re.sub(r"\W", "_", fn)
    lam = (f"[](const {S} *a, int sa, NppiSize ss, NppiRect sr, {S} *d, int ds, "
           f"NppiSize dsz, NppiRect dr, int m, NppStreamContext c) "
           f"{{ (void)sa; (void)ss; (void)sr; (void)ds; (void)dsz; (void)dr; (void)m; (void)c; "
           f"return {fn}(a, sa, ss, sr, d, ds, dsz, dr, m, c); }}")
    prefix = f"num::run_resize<{S}, {C}, {A}>"
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")


def emit_resize_sqr(row: dict) -> Optional[str]:
    fn = row["function_name"]
    lay = _layout(fn)
    S = CTYPE.get(row["dtype"])
    if lay is None or S is None:
        return None
    # NPP's 64f SqrPixel path computes with reduced (float-sized) precision;
    # the double CPU reference does not reproduce it (deferred).
    if row["dtype"] == "64f":
        return None
    C, A = lay[1]
    name = re.sub(r"\W", "_", fn)
    lam = (f"[](const {S} *a, NppiSize ss, int sa, NppiRect sr, {S} *d, int ds, "
           f"NppiRect dr, double fx, double fy, double shx, double shy, int m, "
           f"NppStreamContext c) {{ (void)sa; (void)ss; (void)sr; (void)ds; (void)dr; "
           f"(void)fx; (void)fy; (void)shx; (void)shy; (void)m; (void)c; "
           f"return {fn}(a, ss, sa, sr, d, ds, dr, fx, fy, shx, shy, m, c); }}")
    prefix = f"num::run_resize_sqr<{S}, {C}, {A}>"
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")



def emit_resize_batch(row: dict) -> Optional[str]:
    fn = row["function_name"]
    m = re.search(r"_(AC4|C1|C3|C4)R(_Advanced)?_Ctx$", fn)
    if m is None:
        return None
    tok, adv = m.group(1), m.group(2) is not None
    S = CTYPE.get(row["dtype"])
    if S is None:
        return None
    C, A = LAYOUT[tok]
    name = re.sub(r"\W", "_", fn)
    if adv:
        lam = (f"[](int mw, int mh, NppiImageDescriptor *s, NppiImageDescriptor *d, "
               f"NppiResizeBatchROI_Advanced *r, unsigned nb, int m, NppStreamContext c) "
               f"{{ (void)mw; (void)mh; (void)s; (void)d; (void)r; (void)nb; (void)m; (void)c; "
               f"return {fn}(mw, mh, s, d, r, nb, m, c); }}")
        prefix = f"num::run_resize_batch_advanced<{S}, {C}, {A}>"
    else:
        lam = (f"[](NppiSize ssz, NppiRect sr, NppiSize dsz, NppiRect dr, int m, "
               f"NppiResizeBatchCXR *b, unsigned nb, NppStreamContext c) "
               f"{{ (void)ssz; (void)sr; (void)dsz; (void)dr; (void)m; (void)b; (void)nb; (void)c; "
               f"return {fn}(ssz, sr, dsz, dr, m, b, nb, c); }}")
        prefix = f"num::run_resize_batch<{S}, {C}, {A}>"
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")



def emit_rotate(row: dict) -> Optional[str]:
    fn = row["function_name"]
    lay = _layout(fn)
    S = CTYPE.get(row["dtype"])
    if lay is None or S is None:
        return None
    C, A = lay[1]
    name = re.sub(r"\W", "_", fn)
    lam = (f"[](const {S} *a, NppiSize ss, int sa, NppiRect sr, {S} *d, int ds, "
           f"NppiRect dr, double ang, double shx, double shy, int m, NppStreamContext c) "
           f"{{ (void)ss; (void)sa; (void)sr; (void)ds; (void)dr; (void)ang; (void)shx; "
           f"(void)shy; (void)m; (void)c; return {fn}(a, ss, sa, sr, d, ds, dr, ang, shx, shy, m, c); }}")
    prefix = f"num::run_rotate<{S}, {C}, {A}>"
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")



def emit_remap(row: dict) -> Optional[str]:
    fn = row["function_name"]
    lay = _layout(fn)
    S = CTYPE.get(row["dtype"])
    if lay is None or S is None:
        return None
    # 64f remap maps are Npp64f and NPP's 64f sampling precision is deferred.
    if row["dtype"] == "64f":
        return None
    C, A = lay[1]
    name = re.sub(r"\W", "_", fn)
    lam = (f"[](const {S} *a, NppiSize ss, int sa, NppiRect sr, const Npp32f *mx, int mxs, "
           f"const Npp32f *my, int mys, {S} *d, int ds, NppiSize droi, int m, NppStreamContext c) "
           f"{{ (void)ss; (void)sa; (void)sr; (void)mx; (void)mxs; (void)my; (void)mys; "
           f"(void)ds; (void)droi; (void)m; (void)c; "
           f"return {fn}(a, ss, sa, sr, mx, mxs, my, mys, d, ds, droi, m, c); }}")
    prefix = f"num::run_remap<{S}, Npp32f, {C}, {A}>"
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")



def emit_warp_affine(row: dict) -> Optional[str]:
    fn = row["function_name"]
    lay = _layout(fn)
    S = CTYPE.get(row["dtype"])
    if lay is None or S is None:
        return None
    C, A = lay[1]
    name = re.sub(r"\W", "_", fn)
    lam = (f"[](const {S} *a, NppiSize ss, int sa, NppiRect sr, {S} *d, int ds, "
           f"NppiRect dr, const double *cf, int m, NppStreamContext c) "
           f"{{ (void)ss; (void)sa; (void)sr; (void)ds; (void)dr; (void)m; (void)c; "
           f"return {fn}(a, ss, sa, sr, d, ds, dr, (const double (*)[3])cf, m, c); }}")
    prefix = f"num::run_warp_affine<{S}, {C}, {A}, true>"
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")



def emit_warp_affine_back(row: dict) -> Optional[str]:
    fn = row["function_name"]
    lay = _layout(fn)
    S = CTYPE.get(row["dtype"])
    if lay is None or S is None:
        return None
    C, A = lay[1]
    name = re.sub(r"\W", "_", fn)
    lam = (f"[](const {S} *a, NppiSize ss, int sa, NppiRect sr, {S} *d, int ds, "
           f"NppiRect dr, const double *cf, int m, NppStreamContext c) "
           f"{{ (void)ss; (void)sa; (void)sr; (void)ds; (void)dr; (void)m; (void)c; "
           f"return {fn}(a, ss, sa, sr, d, ds, dr, (const double (*)[3])cf, m, c); }}")
    prefix = f"num::run_warp_affine<{S}, {C}, {A}, false>"
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")



def emit_warp_affine_quad(row: dict) -> Optional[str]:
    fn = row["function_name"]
    lay = _layout(fn)
    S = CTYPE.get(row["dtype"])
    if lay is None or S is None:
        return None
    C, A = lay[1]
    name = re.sub(r"\W", "_", fn)
    lam = (f"[](const {S} *a, NppiSize ss, int sa, NppiRect sr, const double *sq, {S} *d, "
           f"int ds, NppiRect dr, const double *dq, int m, NppStreamContext c) "
           f"{{ (void)ss; (void)sa; (void)sr; (void)ds; (void)dr; (void)m; (void)c; "
           f"return {fn}(a, ss, sa, sr, (const double (*)[2])sq, d, ds, dr, "
           f"(const double (*)[2])dq, m, c); }}")
    prefix = f"num::run_warp_affine_quad<{S}, {C}, {A}>"
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")



def emit_warp_affine_batch(row: dict) -> Optional[str]:
    fn = row["function_name"]
    lay = _layout(fn)
    S = CTYPE.get(row["dtype"])
    if lay is None or S is None:
        return None
    C, A = lay[1]
    name = re.sub(r"\W", "_", fn)
    lam = ("[](NppiSize ssz, NppiRect sr, NppiRect dr, int m, NppiWarpAffineBatchCXR *b, "
           "unsigned nb, NppStreamContext c) { (void)ssz; (void)sr; (void)dr; (void)m; "
           f"(void)b; (void)nb; (void)c; return {fn}(ssz, sr, dr, m, b, nb, c); }}")
    prefix = f"num::run_warp_affine_batch<{S}, {C}, {A}>"
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")


def emit_warp_affine_batch_init(row: dict) -> Optional[str]:
    fn = row["function_name"]
    name = re.sub(r"\W", "_", fn)
    lam = ("[](NppiWarpAffineBatchCXR *b, unsigned nb, NppStreamContext c) { "
           f"(void)b; (void)nb; (void)c; return {fn}(b, nb, c); }}")
    prefix = "num::run_warp_affine_batch_init<Npp8u, 1, 1>"
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")



def emit_warp_perspective(row: dict, forward: bool) -> Optional[str]:
    fn = row["function_name"]
    lay = _layout(fn)
    S = CTYPE.get(row["dtype"])
    if lay is None or S is None:
        return None
    C, A = lay[1]
    name = re.sub(r"\W", "_", fn)
    lam = (f"[](const {S} *a, NppiSize ss, int sa, NppiRect sr, {S} *d, int ds, "
           f"NppiRect dr, const double *cf, int m, NppStreamContext c) "
           f"{{ (void)ss; (void)sa; (void)sr; (void)ds; (void)dr; (void)m; (void)c; "
           f"return {fn}(a, ss, sa, sr, d, ds, dr, (const double (*)[3])cf, m, c); }}")
    prefix = f"num::run_warp_perspective<{S}, {C}, {A}, {str(forward).lower()}>"
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")


def emit_warp_perspective_quad(row: dict) -> Optional[str]:
    fn = row["function_name"]
    lay = _layout(fn)
    S = CTYPE.get(row["dtype"])
    if lay is None or S is None:
        return None
    C, A = lay[1]
    name = re.sub(r"\W", "_", fn)
    lam = (f"[](const {S} *a, NppiSize ss, int sa, NppiRect sr, const double *sq, {S} *d, int ds, "
           f"NppiRect dr, const double *dq, int m, NppStreamContext c) "
           f"{{ (void)ss; (void)sa; (void)sr; (void)ds; (void)dr; (void)m; (void)c; "
           f"return {fn}(a, ss, sa, sr, (const double (*)[2])sq, d, ds, dr, "
           f"(const double (*)[2])dq, m, c); }}")
    prefix = f"num::run_warp_perspective_quad<{S}, {C}, {A}>"
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")


def emit_warp_perspective_batch(row: dict) -> Optional[str]:
    fn = row["function_name"]
    lay = _layout(fn)
    S = CTYPE.get(row["dtype"])
    if lay is None or S is None:
        return None
    C, A = lay[1]
    name = re.sub(r"\W", "_", fn)
    lam = ("[](NppiSize ss, NppiRect sr, NppiRect dr, int m, NppiWarpPerspectiveBatchCXR *b, "
           "unsigned nb, NppStreamContext c) { (void)ss; (void)sr; (void)dr; (void)m; "
           f"(void)b; (void)nb; (void)c; return {fn}(ss, sr, dr, m, b, nb, c); }}")
    prefix = f"num::run_warp_perspective_batch<{S}, {C}, {A}>"
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")


def emit_warp_perspective_batch_init(row: dict) -> Optional[str]:
    fn = row["function_name"]
    name = re.sub(r"\W", "_", fn)
    lam = ("[](NppiWarpPerspectiveBatchCXR *b, unsigned nb, NppStreamContext c) { "
           f"(void)b; (void)nb; (void)c; return {fn}(b, nb, c); }}")
    prefix = "num::run_warp_perspective_batch_init<Npp8u, 1, 1>"
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")


def _planar_count(fn: str) -> Optional[int]:
    m = re.search(r"_P([34])R", fn)
    return int(m.group(1)) if m else None


def emit_warp_perspective_planar(row: dict, forward: bool) -> Optional[str]:
    fn = row["function_name"]
    P = _planar_count(fn)
    S = CTYPE.get(row["dtype"])
    if P is None or S is None:
        return None
    name = re.sub(r"\W", "_", fn)
    lam = (f"[](const {S} **sp, NppiSize ss, int sstep, NppiRect sr, {S} **dp, int dstep, "
           f"NppiRect dr, const double *cf, int m, NppStreamContext c) "
           f"{{ (void)ss; (void)sstep; (void)sr; (void)dstep; (void)dr; (void)m; (void)c; "
           f"return {fn}(sp, ss, sstep, sr, dp, dstep, dr, (const double (*)[3])cf, m, c); }}")
    prefix = f"num::run_warp_perspective_planar<{S}, {P}, {str(forward).lower()}>"
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")


def emit_warp_perspective_quad_planar(row: dict) -> Optional[str]:
    fn = row["function_name"]
    P = _planar_count(fn)
    S = CTYPE.get(row["dtype"])
    if P is None or S is None:
        return None
    name = re.sub(r"\W", "_", fn)
    lam = (f"[](const {S} **sp, NppiSize ss, int sstep, NppiRect sr, const double *sq, {S} **dp, "
           f"int dstep, NppiRect dr, const double *dq, int m, NppStreamContext c) "
           f"{{ (void)ss; (void)sstep; (void)sr; (void)dstep; (void)dr; (void)m; (void)c; "
           f"return {fn}(sp, ss, sstep, sr, (const double (*)[2])sq, dp, dstep, dr, "
           f"(const double (*)[2])dq, m, c); }}")
    prefix = f"num::run_warp_perspective_quad_planar<{S}, {P}>"
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")


def emit_warp_affine_planar(row: dict, invert: bool) -> Optional[str]:
    fn = row["function_name"]
    P = _planar_count(fn)
    S = CTYPE.get(row["dtype"])
    if P is None or S is None:
        return None
    name = re.sub(r"\W", "_", fn)
    lam = (f"[](const {S} **sp, NppiSize ss, int sstep, NppiRect sr, {S} **dp, int dstep, "
           f"NppiRect dr, const double *cf, int m, NppStreamContext c) "
           f"{{ (void)ss; (void)sstep; (void)sr; (void)dstep; (void)dr; (void)m; (void)c; "
           f"return {fn}(sp, ss, sstep, sr, dp, dstep, dr, (const double (*)[3])cf, m, c); }}")
    prefix = f"num::run_warp_affine_planar<{S}, {P}, {str(invert).lower()}>"
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")


def emit_warp_affine_quad_planar(row: dict) -> Optional[str]:
    fn = row["function_name"]
    P = _planar_count(fn)
    S = CTYPE.get(row["dtype"])
    if P is None or S is None:
        return None
    name = re.sub(r"\W", "_", fn)
    lam = (f"[](const {S} **sp, NppiSize ss, int sstep, NppiRect sr, const double *sq, {S} **dp, "
           f"int dstep, NppiRect dr, const double *dq, int m, NppStreamContext c) "
           f"{{ (void)ss; (void)sstep; (void)sr; (void)dstep; (void)dr; (void)m; (void)c; "
           f"return {fn}(sp, ss, sstep, sr, (const double (*)[2])sq, dp, dstep, dr, "
           f"(const double (*)[2])dq, m, c); }}")
    prefix = f"num::run_warp_affine_quad_planar<{S}, {P}>"
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")


def emit_resize_planar(row: dict) -> Optional[str]:
    fn = row["function_name"]
    P = _planar_count(fn)
    S = CTYPE.get(row["dtype"])
    if P is None or S is None:
        return None
    name = re.sub(r"\W", "_", fn)
    lam = (f"[](const {S} **sp, int sstep, NppiSize ss, NppiRect sr, {S} **dp, int dstep, "
           f"NppiSize ds, NppiRect dr, int m, NppStreamContext c) "
           f"{{ (void)sstep; (void)ss; (void)sr; (void)dstep; (void)ds; (void)dr; (void)m; (void)c; "
           f"return {fn}(sp, sstep, ss, sr, dp, dstep, ds, dr, m, c); }}")
    prefix = f"num::run_resize_planar<{S}, {P}>"
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")


def emit_resize_sqr_planar(row: dict) -> Optional[str]:
    fn = row["function_name"]
    P = _planar_count(fn)
    S = CTYPE.get(row["dtype"])
    if P is None or S is None or row["dtype"] == "64f":
        return None
    name = re.sub(r"\W", "_", fn)
    lam = (f"[](const {S} **sp, NppiSize ss, int sstep, NppiRect sr, {S} **dp, int dstep, NppiRect dr, "
           f"double fx, double fy, double sx, double sy, int m, NppStreamContext c) "
           f"{{ (void)ss; (void)sstep; (void)sr; (void)dstep; (void)dr; (void)m; (void)c; "
           f"return {fn}(sp, ss, sstep, sr, dp, dstep, dr, fx, fy, sx, sy, m, c); }}")
    prefix = f"num::run_resize_sqr_planar<{S}, {P}>"
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")


def emit_remap_planar(row: dict) -> Optional[str]:
    fn = row["function_name"]
    P = _planar_count(fn)
    S = CTYPE.get(row["dtype"])
    if P is None or S is None or row["dtype"] == "64f":
        return None
    name = re.sub(r"\W", "_", fn)
    lam = (f"[](const {S} **sp, NppiSize ss, int sstep, NppiRect sr, const float *mx, int mxs, "
           f"const float *my, int mys, {S} **dp, int dstep, NppiSize ds, int m, NppStreamContext c) "
           f"{{ (void)ss; (void)sstep; (void)sr; (void)mxs; (void)mys; (void)dstep; (void)ds; (void)m; (void)c; "
           f"return {fn}(sp, ss, sstep, sr, mx, mxs, my, mys, dp, dstep, ds, m, c); }}")
    prefix = f"num::run_remap_planar<{S}, {P}>"
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(open(CATALOG))
            if r["module"] == "nppi_geometry_transforms.h" and r["ctx"] == "1"]
    out: List[str] = []
    skipped: List[dict] = []
    for row in rows:
        fam = row["family"]
        text = None
        if fam == "nppiMirror":
            text = emit_mirror(row)
        elif fam == "nppiMirrorBatch":
            text = emit_mirror_batch(row)
        elif fam == "nppiResize":
            text = (emit_resize_planar(row) if _planar_count(row["function_name"])
                    else emit_resize(row))
        elif fam == "nppiResizeSqrPixel":
            text = (emit_resize_sqr_planar(row) if _planar_count(row["function_name"])
                    else emit_resize_sqr(row))
        elif fam == "nppiResizeBatch":
            text = emit_resize_batch(row)
        elif fam == "nppiRotate":
            text = emit_rotate(row)
        elif fam == "nppiRemap":
            text = (emit_remap_planar(row) if _planar_count(row["function_name"])
                    else emit_remap(row))
        elif fam == "nppiWarpAffine":
            text = (emit_warp_affine_planar(row, True) if _planar_count(row["function_name"])
                    else emit_warp_affine(row))
        elif fam == "nppiWarpAffineBack":
            text = (emit_warp_affine_planar(row, False) if _planar_count(row["function_name"])
                    else emit_warp_affine_back(row))
        elif fam == "nppiWarpAffineQuad":
            text = (emit_warp_affine_quad_planar(row) if _planar_count(row["function_name"])
                    else emit_warp_affine_quad(row))
        elif fam == "nppiWarpAffineBatch":
            text = emit_warp_affine_batch(row)
        elif fam == "nppiWarpAffineBatchInit":
            text = emit_warp_affine_batch_init(row)
        elif fam == "nppiWarpPerspective":
            text = (emit_warp_perspective_planar(row, True) if _planar_count(row["function_name"])
                    else emit_warp_perspective(row, True))
        elif fam == "nppiWarpPerspectiveBack":
            text = (emit_warp_perspective_planar(row, False) if _planar_count(row["function_name"])
                    else emit_warp_perspective(row, False))
        elif fam == "nppiWarpPerspectiveQuad":
            text = (emit_warp_perspective_quad_planar(row) if _planar_count(row["function_name"])
                    else emit_warp_perspective_quad(row))
        elif fam == "nppiWarpPerspectiveBatch":
            text = emit_warp_perspective_batch(row)
        elif fam == "nppiWarpPerspectiveBatchInit":
            text = emit_warp_perspective_batch_init(row)
        if text is None:
            skipped.append(dict(function=row["function_name"], family=fam,
                                reason="later geometry batch / unsupported"))
            continue
        out.append(text)
    header = ("// GENERATED by tools/gen_nppi_geom_tests.py -- DO NOT EDIT.\n"
              "// module: nppi_geometry_transforms.h (mirror batch)\n"
              "#include \"num_nppi_geom.h\"\n\n")
    dest = Path(args.out) if args.out else ROOT / "test" / "unit" / "num" / "test_gen_nppi_geometry.cpp"
    if args.emit:
        dest.write_text(header + "\n".join(out))
        report = ROOT / "test" / "unit" / "num" / "generated_report.json"
        prev = {}
        if report.exists():
            try:
                prev = json.loads(report.read_text())
            except json.JSONDecodeError:
                prev = {}
        prev["nppi_geometry_transforms.h"] = dict(emitted=len(out), skipped=skipped)
        report.write_text(json.dumps(prev, indent=1))
        print(f"emitted {len(out)} tests to {dest}; skipped {len(skipped)}")
    else:
        print(f"would emit {len(out)} tests; skipped {len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
