#!/usr/bin/env python3
"""
Compare two benchmark CSV files and generate:
  - a comparison CSV
  - an optional HTML report
  - an optional console table

Comparison metric:
  mpp_time_pct_of_nvidia = (mpp_avg_time_ms / nvidia_avg_time_ms) * 100

Interpretation:
  < 100%  => MPP is faster than NVIDIA for this scenario
  = 100%  => parity
  > 100%  => MPP is slower than NVIDIA
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Key:
    function_name: str
    size: str
    variant_tags: str
    data_type: str
    channels: str


@dataclass
class Row:
    function_name: str
    size: str
    variant_tags: str
    data_type: str
    channels: str
    avg_ms: float
    throughput_gbps: float
    success: bool
    error: str


def _clean(value: Optional[str]) -> str:
    return (value or "").strip()


def _to_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _to_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def extract_family(function_name: str) -> str:
    name = function_name
    for prefix in ("nppi", "npps", "npp"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name.split("_", 1)[0] or function_name


def parse_size_value(size: str) -> Tuple[Optional[float], str]:
    text = size.strip()
    if "x" in text:
        parts = text.lower().split("x")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return float(int(parts[0]) * int(parts[1])), "image_pixels"
    if text.endswith("K") and text[:-1].isdigit():
        return float(int(text[:-1]) * 1000), "signal_len"
    if text.endswith("M") and text[:-1].isdigit():
        return float(int(text[:-1]) * 1_000_000), "signal_len"
    if text.isdigit():
        return float(int(text)), "numeric"
    return None, "unknown"


def normalize_scenario_fields(raw: dict) -> Tuple[str, str]:
    size = _clean(raw.get("size"))
    variant_tags = _clean(raw.get("variant_tags"))
    if not variant_tags and "|" in size:
        size, variant_tags = [part.strip() for part in size.split("|", 1)]
    return size, variant_tags


def read_benchmark_csv(path: str) -> Dict[Key, Row]:
    success_acc: Dict[Key, Dict[str, float]] = {}
    first_failure: Dict[Key, Row] = {}

    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            function_name = _clean(raw.get("function_name"))
            size, variant_tags = normalize_scenario_fields(raw)
            data_type = _clean(raw.get("data_type"))
            channels = _clean(raw.get("channels"))
            success = _to_bool(_clean(raw.get("success")))
            avg_ms = _to_float(_clean(raw.get("avg_time_ms")))
            throughput = _to_float(_clean(raw.get("throughput_gbps")))
            error = _clean(raw.get("error"))

            key = Key(
                function_name=function_name,
                size=size,
                variant_tags=variant_tags,
                data_type=data_type,
                channels=channels,
            )
            if success:
                bucket = success_acc.setdefault(
                    key,
                    {
                        "avg_sum": 0.0,
                        "tp_sum": 0.0,
                        "count": 0.0,
                    },
                )
                bucket["avg_sum"] += avg_ms
                bucket["tp_sum"] += throughput
                bucket["count"] += 1.0
            elif key not in first_failure:
                first_failure[key] = Row(
                    function_name=function_name,
                    size=size,
                    variant_tags=variant_tags,
                    data_type=data_type,
                    channels=channels,
                    avg_ms=0.0,
                    throughput_gbps=0.0,
                    success=False,
                    error=error,
                )

    results: Dict[Key, Row] = {}
    for key, bucket in success_acc.items():
        count = max(1.0, bucket["count"])
        results[key] = Row(
            function_name=key.function_name,
            size=key.size,
            variant_tags=key.variant_tags,
            data_type=key.data_type,
            channels=key.channels,
            avg_ms=bucket["avg_sum"] / count,
            throughput_gbps=bucket["tp_sum"] / count,
            success=True,
            error="",
        )

    for key, row in first_failure.items():
        results.setdefault(key, row)

    return results


def compute_mpp_time_pct_of_nvidia(mpp_ms: float, nvidia_ms: float) -> Optional[float]:
    if mpp_ms > 0.0 and nvidia_ms > 0.0:
        return (mpp_ms / nvidia_ms) * 100.0
    return None


def gmean_percent(values: Iterable[float]) -> Optional[float]:
    ratios = [value / 100.0 for value in values if value is not None and value > 0.0]
    if not ratios:
        return None
    return math.exp(sum(math.log(v) for v in ratios) / len(ratios)) * 100.0


def build_family_rows(mpp: Dict[Key, Row], nvidia: Dict[Key, Row], keys: Iterable[Key]) -> List[dict]:
    grouped: Dict[str, List[float]] = {}
    for key in keys:
        mr = mpp.get(key)
        nr = nvidia.get(key)
        pct = compute_mpp_time_pct_of_nvidia(
            mr.avg_ms if (mr and mr.success) else 0.0,
            nr.avg_ms if (nr and nr.success) else 0.0,
        )
        if pct is None:
            continue
        grouped.setdefault(extract_family(key.function_name), []).append(pct)

    rows = []
    for family, values in sorted(grouped.items()):
        agg = gmean_percent(values)
        if agg is None:
            continue
        rows.append(
            {
                "family": family,
                "mpp_time_pct_of_nvidia": agg,
                "count": len(values),
            }
        )
    return rows


def write_compare_csv(
    out_path: str,
    mpp: Dict[Key, Row],
    nvidia: Dict[Key, Row],
    keys: Iterable[Key],
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "function_name",
                "size",
                "variant_tags",
                "data_type",
                "channels",
                "mpp_avg_time_ms",
                "nvidia_avg_time_ms",
                "mpp_time_pct_of_nvidia",
                "mpp_throughput_gbps",
                "nvidia_throughput_gbps",
                "mpp_success",
                "nvidia_success",
                "mpp_error",
                "nvidia_error",
            ]
        )

        for key in keys:
            mr = mpp.get(key)
            nr = nvidia.get(key)
            m_ms = mr.avg_ms if (mr and mr.success) else 0.0
            n_ms = nr.avg_ms if (nr and nr.success) else 0.0
            pct = compute_mpp_time_pct_of_nvidia(m_ms, n_ms)

            writer.writerow(
                [
                    key.function_name,
                    key.size,
                    key.variant_tags,
                    key.data_type,
                    key.channels,
                    f"{m_ms:.6f}" if m_ms > 0 else "",
                    f"{n_ms:.6f}" if n_ms > 0 else "",
                    f"{pct:.2f}" if pct is not None else "",
                    f"{mr.throughput_gbps:.6f}" if (mr and mr.success) else "",
                    f"{nr.throughput_gbps:.6f}" if (nr and nr.success) else "",
                    "true" if (mr and mr.success) else "false",
                    "true" if (nr and nr.success) else "false",
                    mr.error if mr else "Missing",
                    nr.error if nr else "Missing",
                ]
            )


def print_table(
    mpp_label: str,
    nvidia_label: str,
    mpp: Dict[Key, Row],
    nvidia: Dict[Key, Row],
    keys: List[Key],
    limit: int,
    sort_mode: str,
) -> None:
    reverse = sort_mode in ("pct_desc", "speedup_asc")

    def score(key: Key) -> float:
        mr = mpp.get(key)
        nr = nvidia.get(key)
        pct = compute_mpp_time_pct_of_nvidia(
            mr.avg_ms if (mr and mr.success) else 0.0,
            nr.avg_ms if (nr and nr.success) else 0.0,
        )
        if pct is None:
            return math.inf if not reverse else -math.inf
        return pct

    ordered = sorted(keys, key=score, reverse=reverse)
    print(
        f"{'Function':<40} {'Family':<18} {'Size':<14} {'Variant':<14} {'Type':<8} {'Ch':<4} "
        f"{mpp_label + '(ms)':<14} {nvidia_label + '(ms)':<16} {'MPP%OfNV':<10}"
    )
    print("-" * 148)
    shown = 0
    for key in ordered:
        if shown >= limit:
            break
        mr = mpp.get(key)
        nr = nvidia.get(key)
        m_ms = mr.avg_ms if (mr and mr.success) else 0.0
        n_ms = nr.avg_ms if (nr and nr.success) else 0.0
        pct = compute_mpp_time_pct_of_nvidia(m_ms, n_ms)
        pct_text = f"{pct:.2f}%" if pct is not None else "N/A"
        print(
            f"{key.function_name:<40} {extract_family(key.function_name):<18} {key.size:<14} "
            f"{(key.variant_tags or '-'): <14} {key.data_type:<8} {key.channels:<4} "
            f"{(f'{m_ms:.4f}' if m_ms > 0 else 'N/A'):<14} "
            f"{(f'{n_ms:.4f}' if n_ms > 0 else 'N/A'):<16} {pct_text:<10}"
        )
        shown += 1


def write_html_report(
    out_path: str,
    mpp_label: str,
    nvidia_label: str,
    mpp: Dict[Key, Row],
    nvidia: Dict[Key, Row],
    keys: List[Key],
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    table_rows: List[dict] = []
    for key in keys:
        mr = mpp.get(key)
        nr = nvidia.get(key)
        m_ms = mr.avg_ms if (mr and mr.success) else 0.0
        n_ms = nr.avg_ms if (nr and nr.success) else 0.0
        pct = compute_mpp_time_pct_of_nvidia(m_ms, n_ms)
        size_value, size_kind = parse_size_value(key.size)
        table_rows.append(
            {
                "function_name": key.function_name,
                "family": extract_family(key.function_name),
                "size": key.size,
                "variant_tags": key.variant_tags,
                "data_type": key.data_type,
                "channels": key.channels,
                "mpp_ms": m_ms if m_ms > 0 else None,
                "nvidia_ms": n_ms if n_ms > 0 else None,
                "mpp_tp": mr.throughput_gbps if (mr and mr.success and mr.throughput_gbps > 0) else None,
                "nvidia_tp": nr.throughput_gbps if (nr and nr.success and nr.throughput_gbps > 0) else None,
                "mpp_time_pct_of_nvidia": pct,
                "size_value": size_value,
                "size_kind": size_kind,
                "mpp_ok": bool(mr and mr.success),
                "nvidia_ok": bool(nr and nr.success),
                "mpp_err": mr.error if mr else "Missing",
                "nvidia_err": nr.error if nr else "Missing",
            }
        )

    family_rows = build_family_rows(mpp, nvidia, keys)

    doc = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Benchmark Comparison Report</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 16px; color: #111827; }
    .meta { color: #4b5563; margin-bottom: 12px; }
    .grid { display: grid; grid-template-columns: 1fr; gap: 12px; }
    @media (min-width: 1100px) { .grid { grid-template-columns: 1fr 1fr; } }
    .card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; background: #fff; }
    .pill { display: inline-block; padding: 1px 6px; border-radius: 10px; background: #f3f4f6; }
    .small { font-size: 12px; color: #6b7280; }
    input, select { padding: 6px 8px; }
    input { width: 420px; max-width: 100%; }
    table { border-collapse: collapse; width: 100%; font-size: 13px; }
    th, td { border-bottom: 1px solid #eee; padding: 6px 8px; text-align: left; vertical-align: top; }
    th { position: sticky; top: 0; background: #fafafa; cursor: pointer; }
    .ok { color: #166534; }
    .bad { color: #991b1b; }
    .na { color: #666; }
    canvas { width: 100%; height: 280px; border: 1px solid #f0f0f0; border-radius: 8px; }
  </style>
</head>
<body>
  <h2>Benchmark Comparison</h2>
  <div class="meta">
    <div>MPP: <span class="pill">__MPP_LABEL__</span> &nbsp; NVIDIA: <span class="pill">__NVIDIA_LABEL__</span></div>
    <div class="small">Primary metric: MPP time as % of NVIDIA time. Lower is better. 100% means parity.</div>
  </div>

  <div class="grid">
    <div class="card">
      <div><b>Top Catch-Up Families</b> <span class="small">(lowest MPP% of NVIDIA)</span></div>
      <canvas id="familyCatchup"></canvas>
    </div>
    <div class="card">
      <div><b>Top Behind Families</b> <span class="small">(highest MPP% of NVIDIA)</span></div>
      <canvas id="familyBehind"></canvas>
    </div>
    <div class="card" style="grid-column: 1 / -1;">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap;">
        <div><b>Scenario Curve</b> <span class="small">(per API over size)</span></div>
        <div>
          <select id="metricSel">
            <option value="ms">Avg Time (ms)</option>
            <option value="tp">Throughput (GB/s)</option>
          </select>
          <select id="seriesSel"></select>
        </div>
      </div>
      <canvas id="lineSeries"></canvas>
    </div>
  </div>

  <div style="margin: 10px 0;">
    <input id="q" placeholder="Filter by function name or family..." />
  </div>

  <table id="t">
    <thead>
      <tr>
        <th data-k="function_name">Function</th>
        <th data-k="family">Family</th>
        <th data-k="size">Size</th>
        <th data-k="variant_tags">Variant</th>
        <th data-k="data_type">Type</th>
        <th data-k="channels">Ch</th>
        <th data-k="mpp_ms">MPP (ms)</th>
        <th data-k="nvidia_ms">NVIDIA (ms)</th>
        <th data-k="mpp_time_pct_of_nvidia">MPP%OfNVIDIA</th>
        <th>Status</th>
      </tr>
    </thead>
    <tbody></tbody>
  </table>

  <script>
    const rows = __ROWS__;
    const familyRows = __FAMILY_ROWS__;
    const mppLabel = "__MPP_LABEL__";
    const nvidiaLabel = "__NVIDIA_LABEL__";
    let sortKey = "mpp_time_pct_of_nvidia";
    let sortDesc = false;

    function safe(v) {
      return String(v ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    function num(v) {
      return (v === null || v === undefined || v === "") ? NaN : Number(v);
    }

    function cmp(a, b) {
      const av = a[sortKey], bv = b[sortKey];
      const an = num(av), bn = num(bv);
      if (!Number.isNaN(an) && !Number.isNaN(bn)) return an - bn;
      return String(av ?? "").localeCompare(String(bv ?? ""));
    }

    function canvasSize(canvas) {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      return dpr;
    }

    function drawBars(canvas, items, titleMode) {
      const dpr = canvasSize(canvas);
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const padL = 14 * dpr, padR = 16 * dpr, padT = 20 * dpr, padB = 24 * dpr;
      const w = canvas.width - padL - padR;
      const h = canvas.height - padT - padB;
      if (items.length === 0) {
        ctx.fillStyle = "#666";
        ctx.font = `${12 * dpr}px sans-serif`;
        ctx.fillText("No comparable family data.", padL, padT + 20 * dpr);
        return;
      }
      const maxV = Math.max(...items.map(x => Number(x.mpp_time_pct_of_nvidia)));
      const barH = Math.max(10 * dpr, Math.floor(h / items.length) - 4 * dpr);
      ctx.font = `${11 * dpr}px sans-serif`;
      for (let i = 0; i < items.length; ++i) {
        const row = items[i];
        const value = Number(row.mpp_time_pct_of_nvidia);
        const y = padT + i * (barH + 4 * dpr);
        const bw = (value / maxV) * w;
        ctx.fillStyle = value <= 100.0 ? "#16a34a" : "#dc2626";
        ctx.fillRect(padL, y, bw, barH);
        ctx.fillStyle = "#111";
        ctx.fillText(`${row.family} (${row.count})`, padL, y - 2 * dpr);
        ctx.fillText(value.toFixed(1) + "%", padL + bw + 6 * dpr, y + barH - 2 * dpr);
      }
      ctx.fillStyle = "#666";
      ctx.font = `${10 * dpr}px sans-serif`;
      ctx.fillText(titleMode, padL, canvas.height - 8 * dpr);
    }

    function buildSeriesIndex() {
      const grouped = new Map();
      for (const row of rows) {
        const key = `${row.function_name}|${row.variant_tags}|${row.data_type}|${row.channels}`;
        if (!grouped.has(key)) grouped.set(key, []);
        grouped.get(key).push(row);
      }
      return { grouped, keys: Array.from(grouped.keys()).sort() };
    }

    function drawLineChart(canvas, points, metric) {
      const dpr = canvasSize(canvas);
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const padL = 44 * dpr, padR = 16 * dpr, padT = 16 * dpr, padB = 32 * dpr;
      const w = canvas.width - padL - padR;
      const h = canvas.height - padT - padB;

      if (!points || points.length === 0) {
        ctx.fillStyle = "#666";
        ctx.font = `${12 * dpr}px sans-serif`;
        ctx.fillText("No data for selected series.", padL, padT + 20 * dpr);
        return;
      }

      const sorted = points.slice();
      const hasNumericSize = sorted.every(p => typeof p.size_value === "number");
      if (hasNumericSize) sorted.sort((a, b) => a.size_value - b.size_value);

      const pickY = (point, which) => {
        if (metric === "tp") return which === "mpp" ? point.mpp_tp : point.nvidia_tp;
        return which === "mpp" ? point.mpp_ms : point.nvidia_ms;
      };

      const ys = [];
      for (const point of sorted) {
        const ym = pickY(point, "mpp");
        const yn = pickY(point, "nvidia");
        if (ym !== null) ys.push(ym);
        if (yn !== null) ys.push(yn);
      }
      if (ys.length === 0) {
        ctx.fillStyle = "#666";
        ctx.font = `${12 * dpr}px sans-serif`;
        ctx.fillText("No numeric values for selected metric.", padL, padT + 20 * dpr);
        return;
      }

      const yMin = 0.0;
      const yMax = Math.max(1e-9, Math.max(...ys));
      const xAt = index => padL + (sorted.length === 1 ? 0 : (index / (sorted.length - 1)) * w);
      const yAt = value => padT + (1 - (value - yMin) / (yMax - yMin)) * h;

      ctx.strokeStyle = "#ddd";
      ctx.lineWidth = 1 * dpr;
      ctx.beginPath();
      ctx.moveTo(padL, padT);
      ctx.lineTo(padL, padT + h);
      ctx.lineTo(padL + w, padT + h);
      ctx.stroke();

      ctx.fillStyle = "#666";
      ctx.font = `${10 * dpr}px sans-serif`;
      for (let tick = 0; tick <= 4; ++tick) {
        const value = yMin + (tick / 4) * (yMax - yMin);
        const y = yAt(value);
        ctx.strokeStyle = "#f0f0f0";
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(padL + w, y);
        ctx.stroke();
        ctx.fillText(value.toFixed(2), 6 * dpr, y + 3 * dpr);
      }

      function drawSeries(which, color) {
        const hasAny = sorted.some(point => pickY(point, which) !== null);
        if (!hasAny) return;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2 * dpr;
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < sorted.length; ++i) {
          const value = pickY(sorted[i], which);
          if (value === null) continue;
          const x = xAt(i);
          const y = yAt(value);
          if (!started) {
            ctx.moveTo(x, y);
            started = true;
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.stroke();
        ctx.fillStyle = color;
        for (let i = 0; i < sorted.length; ++i) {
          const value = pickY(sorted[i], which);
          if (value === null) continue;
          ctx.beginPath();
          ctx.arc(xAt(i), yAt(value), 3 * dpr, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      drawSeries("mpp", "#2563eb");
      drawSeries("nvidia", "#f97316");

      const step = Math.max(1, Math.floor(sorted.length / 6));
      for (let i = 0; i < sorted.length; i += step) {
        ctx.fillText(String(sorted[i].size), xAt(i) - 10 * dpr, padT + h + 14 * dpr);
      }

      ctx.fillStyle = "#2563eb";
      ctx.fillRect(padL, padT - 10 * dpr, 10 * dpr, 3 * dpr);
      ctx.fillStyle = "#111";
      ctx.fillText(mppLabel + (metric === "tp" ? " (GB/s)" : " (ms)"), padL + 14 * dpr, padT - 6 * dpr);
      ctx.fillStyle = "#f97316";
      ctx.fillRect(padL + 140 * dpr, padT - 10 * dpr, 10 * dpr, 3 * dpr);
      ctx.fillStyle = "#111";
      ctx.fillText(nvidiaLabel + (metric === "tp" ? " (GB/s)" : " (ms)"), padL + 154 * dpr, padT - 6 * dpr);
    }

    function renderTable() {
      const query = document.getElementById("q").value.trim().toLowerCase();
      const tbody = document.querySelector("#t tbody");
      tbody.innerHTML = "";
      let filtered = rows;
      if (query) {
        filtered = rows.filter(row =>
          row.function_name.toLowerCase().includes(query) || row.family.toLowerCase().includes(query)
        );
      }
      filtered = filtered.slice().sort((a, b) => {
        const result = cmp(a, b);
        return sortDesc ? -result : result;
      });
      for (const row of filtered) {
        const pct = row.mpp_time_pct_of_nvidia;
        const pctClass = pct === null ? "na" : (pct <= 100.0 ? "ok" : "bad");
        const status = (row.mpp_ok && row.nvidia_ok) ? "OK" : "Partial";
        const statusDetail = (!row.mpp_ok || !row.nvidia_ok) ? `mpp=${row.mpp_ok} nvidia=${row.nvidia_ok}` : "";
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${safe(row.function_name)}</td>
          <td>${safe(row.family)}</td>
          <td>${safe(row.size)}</td>
          <td>${safe(row.variant_tags || "-")}</td>
          <td>${safe(row.data_type)}</td>
          <td>${safe(row.channels)}</td>
          <td>${row.mpp_ms === null ? "N/A" : Number(row.mpp_ms).toFixed(4)}</td>
          <td>${row.nvidia_ms === null ? "N/A" : Number(row.nvidia_ms).toFixed(4)}</td>
          <td class="${pctClass}">${pct === null ? "N/A" : Number(pct).toFixed(2) + "%"}</td>
          <td>${safe(status)}${statusDetail ? `<div class="small">${safe(statusDetail)}</div>` : ""}</td>
        `;
        tbody.appendChild(tr);
      }
    }

    function initCharts() {
      const catchup = familyRows
        .slice()
        .sort((a, b) => Number(a.mpp_time_pct_of_nvidia) - Number(b.mpp_time_pct_of_nvidia))
        .slice(0, 12);
      const behind = familyRows
        .slice()
        .sort((a, b) => Number(b.mpp_time_pct_of_nvidia) - Number(a.mpp_time_pct_of_nvidia))
        .slice(0, 12);
      drawBars(document.getElementById("familyCatchup"), catchup, "Closer to or faster than NVIDIA");
      drawBars(document.getElementById("familyBehind"), behind, "Most behind NVIDIA");

      const index = buildSeriesIndex();
      const seriesSel = document.getElementById("seriesSel");
      const metricSel = document.getElementById("metricSel");
      seriesSel.innerHTML = "";
      for (const key of index.keys) {
        const [fn, dt, ch] = key.split("|");
        const option = document.createElement("option");
        option.value = key;
        option.textContent = `${fn}  ${dt}  C${ch}`;
        seriesSel.appendChild(option);
      }
      function rerenderSeries() {
        const key = seriesSel.value;
        drawLineChart(document.getElementById("lineSeries"), index.grouped.get(key) || [], metricSel.value);
      }
      seriesSel.onchange = rerenderSeries;
      metricSel.onchange = rerenderSeries;
      if (index.keys.length > 0) {
        seriesSel.value = index.keys[0];
        rerenderSeries();
      }
    }

    document.getElementById("q").addEventListener("input", renderTable);
    document.querySelectorAll("th[data-k]").forEach(th => {
      th.addEventListener("click", () => {
        const key = th.getAttribute("data-k");
        if (sortKey === key) sortDesc = !sortDesc;
        else {
          sortKey = key;
          sortDesc = key !== "mpp_time_pct_of_nvidia";
        }
        renderTable();
      });
    });

    window.addEventListener("resize", () => {
      initCharts();
      renderTable();
    });

    renderTable();
    initCharts();
  </script>
</body>
</html>
"""

    doc = (
        doc.replace("__MPP_LABEL__", html.escape(mpp_label, quote=True))
        .replace("__NVIDIA_LABEL__", html.escape(nvidia_label, quote=True))
        .replace("__ROWS__", json.dumps(table_rows, ensure_ascii=False))
        .replace("__FAMILY_ROWS__", json.dumps(family_rows, ensure_ascii=False))
    )

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(doc)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Compare two benchmark CSV files and generate reports.")
    parser.add_argument("mpp_csv", help="MPP benchmark CSV (e.g. out_mpp.csv)")
    parser.add_argument("nvidia_csv", help="NVIDIA benchmark CSV (e.g. out_nvidia.csv)")
    parser.add_argument("-o", "--output", default="", help="Output compare CSV path (default: <mpp_prefix>_compare.csv)")
    parser.add_argument("--html", default="", help="Optional HTML report output path")
    parser.add_argument("--labels", nargs=2, metavar=("MPP_LABEL", "NVIDIA_LABEL"), default=None, help="Set labels")
    parser.add_argument("--mpp-label", default="MPP", help="MPP label for table/report")
    parser.add_argument("--nvidia-label", default="NVIDIA NPP", help="NVIDIA label for table/report")
    parser.add_argument("--filter", default="", help="Only include functions whose name contains this substring")
    parser.add_argument("--print-table", action="store_true", help="Print a console comparison table")
    parser.add_argument("--limit", type=int, default=200, help="Max rows to print in console table")
    parser.add_argument(
        "--sort",
        choices=["pct_asc", "pct_desc", "speedup_asc", "speedup_desc"],
        default="pct_asc",
        help="Console sort order. pct_asc means closest to / faster than NVIDIA first.",
    )
    args = parser.parse_args(argv)

    if args.labels is not None:
        args.mpp_label, args.nvidia_label = args.labels

    mpp = read_benchmark_csv(args.mpp_csv)
    nvidia = read_benchmark_csv(args.nvidia_csv)
    if not mpp or not nvidia:
        print("No data to compare (one side is empty).", file=sys.stderr)
        return 2

    keys = sorted(
        set(mpp.keys()) | set(nvidia.keys()),
        key=lambda key: (key.function_name, key.size, key.variant_tags, key.data_type, key.channels),
    )
    if args.filter:
        keys = [key for key in keys if args.filter in key.function_name]

    out_csv = args.output
    if not out_csv:
        prefix = os.path.splitext(args.mpp_csv)[0]
        if prefix.endswith("_mpp"):
            prefix = prefix[:-4]
        out_csv = prefix + "_compare.csv"

    write_compare_csv(out_csv, mpp, nvidia, keys)
    print(f"Comparison CSV saved to: {out_csv}")

    if args.print_table:
        print("")
        print_table(args.mpp_label, args.nvidia_label, mpp, nvidia, keys, args.limit, args.sort)

    if args.html:
        write_html_report(args.html, args.mpp_label, args.nvidia_label, mpp, nvidia, keys)
        print(f"HTML report saved to: {args.html}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
