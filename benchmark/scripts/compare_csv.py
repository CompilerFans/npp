#!/usr/bin/env python3
"""
Compare two benchmark CSV files and generate:
  - a comparison CSV
  - an optional HTML report

Comparison metric:
  target_perf_pct_of_baseline = (baseline_avg_time_ms / target_avg_time_ms) * 100

Interpretation:
  > 100: target faster than baseline
  = 100: parity
  < 100: target slower than baseline
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
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


def slugify_label(value: str) -> str:
    text = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "label"


def metric_column_name(target_label: str, baseline_label: str) -> str:
    return f"{slugify_label(target_label)}_vs_{slugify_label(baseline_label)}"


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


def compute_perf_pct_of_baseline(target_ms: float, baseline_ms: float) -> Optional[float]:
    if target_ms > 0.0 and baseline_ms > 0.0:
        return (baseline_ms / target_ms) * 100.0
    return None


def gmean_percent(values: Iterable[float]) -> Optional[float]:
    ratios = [value / 100.0 for value in values if value is not None and value > 0.0]
    if not ratios:
        return None
    return math.exp(sum(math.log(v) for v in ratios) / len(ratios)) * 100.0


def build_family_rows(target: Dict[Key, Row], baseline: Dict[Key, Row], keys: Iterable[Key]) -> List[dict]:
    grouped_perf_pct: Dict[str, List[float]] = {}
    for key in keys:
        tr = target.get(key)
        br = baseline.get(key)
        perf_pct = compute_perf_pct_of_baseline(
            tr.avg_ms if (tr and tr.success) else 0.0,
            br.avg_ms if (br and br.success) else 0.0,
        )
        if perf_pct is None:
            continue
        family = extract_family(key.function_name)
        grouped_perf_pct.setdefault(family, []).append(perf_pct)

    rows = []
    for family, perf_values in sorted(grouped_perf_pct.items()):
        perf_agg = gmean_percent(perf_values)
        if perf_agg is None:
            continue
        rows.append(
            {
                "family": family,
                "perf_pct": perf_agg,
                "count": len(perf_values),
            }
        )
    return rows


def write_compare_csv(
    out_path: str,
    target: Dict[Key, Row],
    baseline: Dict[Key, Row],
    keys: Iterable[Key],
    metric_col_name: str,
    target_prefix: str,
    baseline_prefix: str,
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
                f"{target_prefix}_avg_time_ms",
                f"{baseline_prefix}_avg_time_ms",
                metric_col_name,
                f"{target_prefix}_throughput_gbps",
                f"{baseline_prefix}_throughput_gbps",
            ]
        )

        for key in keys:
            tr = target.get(key)
            br = baseline.get(key)
            t_ms = tr.avg_ms if (tr and tr.success) else 0.0
            b_ms = br.avg_ms if (br and br.success) else 0.0
            perf_pct = compute_perf_pct_of_baseline(t_ms, b_ms)

            writer.writerow(
                [
                    key.function_name,
                    key.size,
                    key.variant_tags,
                    key.data_type,
                    key.channels,
                    f"{t_ms:.6f}" if t_ms > 0 else "",
                    f"{b_ms:.6f}" if b_ms > 0 else "",
                    f"{perf_pct:.2f}" if perf_pct is not None else "",
                    f"{tr.throughput_gbps:.6f}" if (tr and tr.success) else "",
                    f"{br.throughput_gbps:.6f}" if (br and br.success) else "",
                ]
            )


def write_html_report(
    out_path: str,
    target_label: str,
    baseline_label: str,
    target: Dict[Key, Row],
    baseline: Dict[Key, Row],
    keys: List[Key],
    metric_col_name: str,
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    table_rows: List[dict] = []
    for key in keys:
        tr = target.get(key)
        br = baseline.get(key)
        t_ms = tr.avg_ms if (tr and tr.success) else 0.0
        b_ms = br.avg_ms if (br and br.success) else 0.0
        perf_pct = compute_perf_pct_of_baseline(t_ms, b_ms)
        table_rows.append(
            {
                "function_name": key.function_name,
                "family": extract_family(key.function_name),
                "size": key.size,
                "variant_tags": key.variant_tags,
                "data_type": key.data_type,
                "channels": key.channels,
                "target_ms": t_ms if t_ms > 0 else None,
                "baseline_ms": b_ms if b_ms > 0 else None,
                "target_tp": tr.throughput_gbps if (tr and tr.success and tr.throughput_gbps > 0) else None,
                "baseline_tp": br.throughput_gbps if (br and br.success and br.throughput_gbps > 0) else None,
                "perf_pct": perf_pct,
                "target_ok": bool(tr and tr.success),
                "baseline_ok": bool(br and br.success),
                "target_err": tr.error if tr else "Missing",
                "baseline_err": br.error if br else "Missing",
            }
        )

    family_rows = build_family_rows(target, baseline, keys)

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
    input { padding: 6px 8px; }
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
    <div>Target: <span class="pill">__TARGET_LABEL__</span> &nbsp; Baseline: <span class="pill">__BASELINE_LABEL__</span></div>
    <div class="small">Primary metric: __METRIC_COL__ = (baseline ms / target ms) * 100. Higher is better; &gt;100 means target is faster.</div>
  </div>

  <div class="grid">
    <div class="card">
      <div><b>Top Performing Families</b> <span class="small">(highest __METRIC_COL__)</span></div>
      <canvas id="familyCatchup"></canvas>
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
        <th data-k="target_ms">Target (ms)</th>
        <th data-k="baseline_ms">Baseline (ms)</th>
        <th data-k="perf_pct">__METRIC_COL__</th>
        <th>Status</th>
      </tr>
    </thead>
    <tbody></tbody>
  </table>

  <script>
    const rows = __ROWS__;
    const familyRows = __FAMILY_ROWS__;
    let sortKey = "perf_pct";
    let sortDesc = true;

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
        const maxV = Math.max(...items.map(x => Number(x.perf_pct)));
      const barH = Math.max(10 * dpr, Math.floor(h / items.length) - 4 * dpr);
      ctx.font = `${11 * dpr}px sans-serif`;
      for (let i = 0; i < items.length; ++i) {
        const row = items[i];
        const value = Number(row.perf_pct);
        const y = padT + i * (barH + 4 * dpr);
        const bw = (value / maxV) * w;
        ctx.fillStyle = value >= 100.0 ? "#16a34a" : "#dc2626";
        ctx.fillRect(padL, y, bw, barH);
        ctx.fillStyle = "#111";
        ctx.fillText(`${row.family} (${row.count})`, padL, y - 2 * dpr);
        ctx.fillText(value.toFixed(1) + "%", padL + bw + 6 * dpr, y + barH - 2 * dpr);
      }
      ctx.fillStyle = "#666";
      ctx.font = `${10 * dpr}px sans-serif`;
      ctx.fillText(titleMode, padL, canvas.height - 8 * dpr);
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
        const perfPct = row.perf_pct;
        const perfPctClass = perfPct === null ? "na" : (perfPct >= 100.0 ? "ok" : "bad");
        const status = (row.target_ok && row.baseline_ok) ? "OK" : "Partial";
        const statusDetail = (!row.target_ok || !row.baseline_ok)
          ? `target=${row.target_ok} baseline=${row.baseline_ok}` : "";
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${safe(row.function_name)}</td>
          <td>${safe(row.family)}</td>
          <td>${safe(row.size)}</td>
          <td>${safe(row.variant_tags || "-")}</td>
          <td>${safe(row.data_type)}</td>
          <td>${safe(row.channels)}</td>
          <td>${row.target_ms === null ? "N/A" : Number(row.target_ms).toFixed(4)}</td>
          <td>${row.baseline_ms === null ? "N/A" : Number(row.baseline_ms).toFixed(4)}</td>
          <td class="${perfPctClass}">${perfPct === null ? "N/A" : Number(perfPct).toFixed(2) + "%"}</td>
          <td>${safe(status)}${statusDetail ? `<div class="small">${safe(statusDetail)}</div>` : ""}</td>
        `;
        tbody.appendChild(tr);
      }
    }

    function renderCharts() {
      const catchup = familyRows
        .slice()
        .sort((a, b) => Number(b.perf_pct) - Number(a.perf_pct))
        .slice(0, 12);
      drawBars(document.getElementById("familyCatchup"), catchup, "Higher is better");
    }

    document.getElementById("q").addEventListener("input", renderTable);
    document.querySelectorAll("th[data-k]").forEach(th => {
      th.addEventListener("click", () => {
        const key = th.getAttribute("data-k");
        if (sortKey === key) sortDesc = !sortDesc;
        else {
          sortKey = key;
          sortDesc = key === "perf_pct";
        }
        renderTable();
      });
    });

    window.addEventListener("resize", () => {
      renderCharts();
      renderTable();
    });

    renderTable();
    renderCharts();
  </script>
</body>
</html>
"""

    doc = (
        doc.replace("__TARGET_LABEL__", html.escape(target_label, quote=True))
        .replace("__BASELINE_LABEL__", html.escape(baseline_label, quote=True))
        .replace("__METRIC_COL__", html.escape(metric_col_name, quote=True))
        .replace("__ROWS__", json.dumps(table_rows, ensure_ascii=False))
        .replace("__FAMILY_ROWS__", json.dumps(family_rows, ensure_ascii=False))
    )

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(doc)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Compare two benchmark CSV files and generate reports.")
    parser.add_argument("--csv-files", nargs=2, metavar=("TARGET_CSV", "BASELINE_CSV"), required=True,
                        help="Input CSVs: <target> <baseline>")
    parser.add_argument("-o", "--output", default="", help="Output compare CSV path (default: <target_prefix>_compare.csv)")
    parser.add_argument("--html", default="", help="Optional HTML report output path")
    parser.add_argument("--labels", nargs=2, metavar=("TARGET_LABEL", "BASELINE_LABEL"),
                        default=("target", "baseline"), help="Display labels for target and baseline")
    parser.add_argument("--baseline", default="", help="Baseline label name (must be one of --labels). Default: second label")
    parser.add_argument("--filter", default="", help="Only include functions whose name contains this substring")
    args = parser.parse_args(argv)

    csv_a, csv_b = args.csv_files
    label_a, label_b = args.labels

    baseline_label = args.baseline if args.baseline else label_b
    if baseline_label not in (label_a, label_b):
        parser.error(f"--baseline must be one of labels: {label_a}, {label_b}")

    if baseline_label == label_a:
        target_csv, baseline_csv = csv_b, csv_a
        target_label, baseline_label = label_b, label_a
    else:
        target_csv, baseline_csv = csv_a, csv_b
        target_label, baseline_label = label_a, label_b

    metric_col_name = metric_column_name(target_label, baseline_label)
    target_prefix = slugify_label(target_label)
    baseline_prefix = slugify_label(baseline_label)

    target = read_benchmark_csv(target_csv)
    baseline = read_benchmark_csv(baseline_csv)
    if not target or not baseline:
        print("No data to compare (one side is empty).", file=sys.stderr)
        return 2

    keys = sorted(
        set(target.keys()) | set(baseline.keys()),
        key=lambda key: (key.function_name, key.size, key.variant_tags, key.data_type, key.channels),
    )
    if args.filter:
        keys = [key for key in keys if args.filter in key.function_name]

    out_csv = args.output
    if not out_csv:
        prefix = os.path.splitext(target_csv)[0]
        if prefix.endswith("_mpp"):
            prefix = prefix[:-4]
        out_csv = prefix + "_compare.csv"

    write_compare_csv(out_csv, target, baseline, keys, metric_col_name, target_prefix, baseline_prefix)
    print(f"Comparison CSV saved to: {out_csv}")

    if args.html:
        write_html_report(args.html, target_label, baseline_label, target, baseline, keys, metric_col_name)
        print(f"HTML report saved to: {args.html}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
