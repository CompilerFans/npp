#!/usr/bin/env python3
"""
Progress report for performance work:
  1) MPP(after) vs MPP(before)
  2) MPP(after) vs NVIDIA(reference)

Duplicate rows for the same scenario are aggregated with mean(avg_time_ms)
across successful rows.
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
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class ScenarioKey:
    function_name: str
    size: str
    variant_tags: str
    data_type: str
    channels: str


@dataclass(frozen=True)
class ApiKey:
    function_name: str
    data_type: str
    channels: str


@dataclass
class ScenarioRow:
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


def normalize_scenario_fields(raw: dict) -> tuple[str, str]:
    size = _clean(raw.get("size"))
    variant_tags = _clean(raw.get("variant_tags"))
    if not variant_tags and "|" in size:
        size, variant_tags = [part.strip() for part in size.split("|", 1)]
    return size, variant_tags


def read_benchmark_csv(path: str) -> Dict[ScenarioKey, ScenarioRow]:
    success_acc: Dict[ScenarioKey, Dict[str, float]] = {}
    first_failure: Dict[ScenarioKey, ScenarioRow] = {}

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

            key = ScenarioKey(
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
                first_failure[key] = ScenarioRow(
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

    results: Dict[ScenarioKey, ScenarioRow] = {}
    for key, bucket in success_acc.items():
        count = max(1.0, bucket["count"])
        results[key] = ScenarioRow(
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


def safe_ratio(num: float, den: float) -> Optional[float]:
    if num > 0.0 and den > 0.0:
        return num / den
    return None


def gmean(values: Iterable[float]) -> Optional[float]:
    vals = [value for value in values if value is not None and value > 0.0]
    if not vals:
        return None
    return math.exp(sum(math.log(value) for value in vals) / len(vals))


def write_summary_csv(
    out_path: str,
    before_mpp: Dict[ScenarioKey, ScenarioRow],
    after_mpp: Dict[ScenarioKey, ScenarioRow],
    ref_nvidia: Dict[ScenarioKey, ScenarioRow],
    name_filter: str = "",
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    apis: Dict[ApiKey, List[ScenarioKey]] = {}
    all_keys = set(before_mpp.keys()) | set(after_mpp.keys()) | set(ref_nvidia.keys())
    for scenario in all_keys:
        api_key = ApiKey(scenario.function_name, scenario.data_type, scenario.channels)
        apis.setdefault(api_key, []).append(scenario)

    with open(out_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "function_name",
                "data_type",
                "channels",
                "scenarios_total",
                "scenarios_compared_before_after",
                "scenarios_compared_after_nvidia",
                "improve_factor_gmean_before_over_after",
                "improve_pct",
                "gap_factor_gmean_after_over_nvidia",
                "gap_pct_after_over_nvidia",
                "catchup_factor_gmean",
                "catchup_pct",
                "before_fail_count",
                "after_fail_count",
                "nvidia_fail_count",
            ]
        )

        for api_key in sorted(apis, key=lambda item: (item.function_name, item.data_type, item.channels)):
            if name_filter and name_filter not in api_key.function_name:
                continue

            before_after_ratios: List[float] = []
            after_nvidia_ratios: List[float] = []
            catchup_ratios: List[float] = []
            before_fail = 0
            after_fail = 0
            nvidia_fail = 0
            compared_before_after = 0
            compared_after_nvidia = 0

            for scenario in apis[api_key]:
                before = before_mpp.get(scenario)
                after = after_mpp.get(scenario)
                nvidia = ref_nvidia.get(scenario)

                if before and not before.success:
                    before_fail += 1
                if after and not after.success:
                    after_fail += 1
                if nvidia and not nvidia.success:
                    nvidia_fail += 1

                if before and before.success and after and after.success:
                    ratio = safe_ratio(before.avg_ms, after.avg_ms)
                    if ratio is not None:
                        before_after_ratios.append(ratio)
                        compared_before_after += 1

                if after and after.success and nvidia and nvidia.success:
                    ratio = safe_ratio(after.avg_ms, nvidia.avg_ms)
                    if ratio is not None:
                        after_nvidia_ratios.append(ratio)
                        compared_after_nvidia += 1

                if before and before.success and after and after.success and nvidia and nvidia.success:
                    gap_before = safe_ratio(before.avg_ms, nvidia.avg_ms)
                    gap_after = safe_ratio(after.avg_ms, nvidia.avg_ms)
                    if gap_before is not None and gap_after is not None and gap_after > 0.0:
                        catchup_ratios.append(gap_before / gap_after)

            improve = gmean(before_after_ratios)
            gap = gmean(after_nvidia_ratios)
            catchup = gmean(catchup_ratios)

            writer.writerow(
                [
                    api_key.function_name,
                    api_key.data_type,
                    api_key.channels,
                    len(apis[api_key]),
                    compared_before_after,
                    compared_after_nvidia,
                    f"{improve:.6f}" if improve is not None else "",
                    f"{((improve - 1.0) * 100.0):.2f}" if improve is not None else "",
                    f"{gap:.6f}" if gap is not None else "",
                    f"{(gap * 100.0):.2f}" if gap is not None else "",
                    f"{catchup:.6f}" if catchup is not None else "",
                    f"{((catchup - 1.0) * 100.0):.2f}" if catchup is not None else "",
                    before_fail,
                    after_fail,
                    nvidia_fail,
                ]
            )


def build_family_rows(summary_rows: List[dict]) -> List[dict]:
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for row in summary_rows:
        family = row["family"]
        bucket = grouped.setdefault(family, {"improve_pct": [], "gap_pct": [], "catchup_pct": []})
        if row["improve_pct"] is not None:
            bucket["improve_pct"].append(row["improve_pct"])
        if row["gap_pct"] is not None:
            bucket["gap_pct"].append(row["gap_pct"])
        if row["catchup_pct"] is not None:
            bucket["catchup_pct"].append(row["catchup_pct"])

    out = []
    for family, bucket in sorted(grouped.items()):
        improve = sum(bucket["improve_pct"]) / len(bucket["improve_pct"]) if bucket["improve_pct"] else None
        gap = sum(bucket["gap_pct"]) / len(bucket["gap_pct"]) if bucket["gap_pct"] else None
        catchup = sum(bucket["catchup_pct"]) / len(bucket["catchup_pct"]) if bucket["catchup_pct"] else None
        out.append(
            {
                "family": family,
                "improve_pct": improve,
                "gap_pct": gap,
                "catchup_pct": catchup,
                "count": max(len(bucket["improve_pct"]), len(bucket["gap_pct"]), len(bucket["catchup_pct"])),
            }
        )
    return out


def write_html_report(
    out_path: str,
    before_label: str,
    after_label: str,
    nvidia_label: str,
    summary_rows: List[dict],
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    family_rows = build_family_rows(summary_rows)

    doc = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Benchmark Progress Report</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 16px; color: #111827; }
    .meta { color: #4b5563; margin-bottom: 12px; }
    .grid { display: grid; grid-template-columns: 1fr; gap: 12px; }
    @media (min-width: 1100px) { .grid { grid-template-columns: 1fr 1fr; } }
    .card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; background: #fff; }
    .pill { display: inline-block; padding: 1px 6px; border-radius: 10px; background: #f3f4f6; }
    .small { font-size: 12px; color: #6b7280; }
    input { padding: 6px 8px; width: 420px; max-width: 100%; }
    table { border-collapse: collapse; width: 100%; font-size: 13px; }
    th, td { border-bottom: 1px solid #eee; padding: 6px 8px; text-align: left; vertical-align: top; }
    th { position: sticky; top: 0; background: #fafafa; cursor: pointer; }
    .ok { color: #166534; }
    .bad { color: #991b1b; }
    .na { color: #666; }
    canvas { width: 100%; height: 300px; border: 1px solid #f0f0f0; border-radius: 8px; }
  </style>
</head>
<body>
  <h2>Benchmark Progress</h2>
  <div class="meta">
    <div>MPP: <span class="pill">__BEFORE__</span> → <span class="pill">__AFTER__</span> &nbsp; Target: <span class="pill">__NVIDIA__</span></div>
    <div class="small">Improve% is relative to previous MPP. Gap% is current MPP time as % of NVIDIA time. Lower gap is better; 100% means parity.</div>
  </div>

  <div class="grid">
    <div class="card">
      <div><b>Top Improved Families</b> <span class="small">(highest Improve%)</span></div>
      <canvas id="familyImprove"></canvas>
    </div>
    <div class="card">
      <div><b>Top Catch-Up Families</b> <span class="small">(lowest Gap%)</span></div>
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
        <th data-k="data_type">Type</th>
        <th data-k="channels">Ch</th>
        <th data-k="improve_pct">Improve%</th>
        <th data-k="gap_pct">Gap% (After/NVIDIA)</th>
        <th data-k="catchup_pct">Catchup%</th>
        <th data-k="scenarios_compared_before_after">BA</th>
        <th data-k="scenarios_compared_after_nvidia">AN</th>
        <th data-k="after_fail_count">AfterFail</th>
      </tr>
    </thead>
    <tbody></tbody>
  </table>

  <script>
    const rows = __ROWS__;
    const familyRows = __FAMILY_ROWS__;
    let sortKey = "gap_pct";
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

    function drawBars(canvas, items, field, lowerIsBetter) {
      const dpr = canvasSize(canvas);
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const padL = 14 * dpr, padR = 16 * dpr, padT = 20 * dpr, padB = 24 * dpr;
      const w = canvas.width - padL - padR;
      const h = canvas.height - padT - padB;
      if (items.length === 0) {
        ctx.fillStyle = "#666";
        ctx.font = `${12 * dpr}px sans-serif`;
        ctx.fillText("No family data.", padL, padT + 20 * dpr);
        return;
      }
      const values = items.map(x => Number(x[field]));
      const maxV = Math.max(...values.map(v => Math.abs(v)));
      const barH = Math.max(10 * dpr, Math.floor(h / items.length) - 4 * dpr);
      ctx.font = `${11 * dpr}px sans-serif`;
      for (let i = 0; i < items.length; ++i) {
        const row = items[i];
        const value = Number(row[field]);
        const y = padT + i * (barH + 4 * dpr);
        const bw = (Math.abs(value) / Math.max(maxV, 1e-9)) * w;
        const good = lowerIsBetter ? value <= 100.0 : value >= 0.0;
        ctx.fillStyle = good ? "#16a34a" : "#dc2626";
        ctx.fillRect(padL, y, bw, barH);
        ctx.fillStyle = "#111";
        ctx.fillText(`${row.family} (${row.count})`, padL, y - 2 * dpr);
        ctx.fillText(value.toFixed(1) + "%", padL + bw + 6 * dpr, y + barH - 2 * dpr);
      }
    }

    function renderTable() {
      const q = document.getElementById("q").value.trim().toLowerCase();
      const tbody = document.querySelector("#t tbody");
      tbody.innerHTML = "";
      let filtered = rows;
      if (q) {
        filtered = rows.filter(row => row.function_name.toLowerCase().includes(q) || row.family.toLowerCase().includes(q));
      }
      filtered = filtered.slice().sort((a, b) => {
        const result = cmp(a, b);
        return sortDesc ? -result : result;
      });
      for (const row of filtered) {
        const improveClass = row.improve_pct === null ? "na" : (row.improve_pct >= 0.0 ? "ok" : "bad");
        const gapClass = row.gap_pct === null ? "na" : (row.gap_pct <= 100.0 ? "ok" : "bad");
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${safe(row.function_name)}</td>
          <td>${safe(row.family)}</td>
          <td>${safe(row.data_type)}</td>
          <td>${safe(row.channels)}</td>
          <td class="${improveClass}">${row.improve_pct === null ? "N/A" : row.improve_pct.toFixed(2)}</td>
          <td class="${gapClass}">${row.gap_pct === null ? "N/A" : row.gap_pct.toFixed(2)}</td>
          <td>${row.catchup_pct === null ? "N/A" : row.catchup_pct.toFixed(2)}</td>
          <td>${row.scenarios_compared_before_after}</td>
          <td>${row.scenarios_compared_after_nvidia}</td>
          <td>${row.after_fail_count}</td>
        `;
        tbody.appendChild(tr);
      }
    }

    function initCharts() {
      const topImprove = familyRows
        .filter(row => row.improve_pct !== null)
        .slice()
        .sort((a, b) => Number(b.improve_pct) - Number(a.improve_pct))
        .slice(0, 12);
      const topCatchup = familyRows
        .filter(row => row.gap_pct !== null)
        .slice()
        .sort((a, b) => Number(a.gap_pct) - Number(b.gap_pct))
        .slice(0, 12);
      drawBars(document.getElementById("familyImprove"), topImprove, "improve_pct", false);
      drawBars(document.getElementById("familyCatchup"), topCatchup, "gap_pct", true);
    }

    document.getElementById("q").addEventListener("input", renderTable);
    document.querySelectorAll("th[data-k]").forEach(th => {
      th.addEventListener("click", () => {
        const key = th.getAttribute("data-k");
        if (sortKey === key) sortDesc = !sortDesc;
        else {
          sortKey = key;
          sortDesc = key === "improve_pct" || key === "after_fail_count";
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
        doc.replace("__BEFORE__", html.escape(before_label, quote=True))
        .replace("__AFTER__", html.escape(after_label, quote=True))
        .replace("__NVIDIA__", html.escape(nvidia_label, quote=True))
        .replace("__ROWS__", json.dumps(summary_rows, ensure_ascii=False))
        .replace("__FAMILY_ROWS__", json.dumps(family_rows, ensure_ascii=False))
    )

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(doc)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Generate a progress report: MPP(before→after) + gap-to-NVIDIA.")
    parser.add_argument("mpp_before_csv", help="MPP before CSV (baseline)")
    parser.add_argument("mpp_after_csv", help="MPP after CSV (current)")
    parser.add_argument("nvidia_csv", help="NVIDIA reference CSV (target)")
    parser.add_argument("-o", "--output", default="", help="Output progress summary CSV path")
    parser.add_argument("--html", default="", help="Optional HTML report output path")
    parser.add_argument("--labels", nargs=3, metavar=("BEFORE", "AFTER", "NVIDIA"), default=None, help="Labels for report")
    parser.add_argument("--filter", default="", help="Only include functions whose name contains this substring")
    args = parser.parse_args(argv)

    before_label, after_label, nvidia_label = ("before", "after", "NVIDIA") if not args.labels else tuple(args.labels)

    before = read_benchmark_csv(args.mpp_before_csv)
    after = read_benchmark_csv(args.mpp_after_csv)
    nvidia = read_benchmark_csv(args.nvidia_csv)
    if not before or not after or not nvidia:
        print("One of the inputs is empty; cannot generate progress report.", file=sys.stderr)
        return 2

    out_csv = args.output or (os.path.splitext(args.mpp_after_csv)[0] + "_progress.csv")
    write_summary_csv(out_csv, before, after, nvidia, name_filter=args.filter)
    print(f"Progress summary CSV saved to: {out_csv}")

    summary_rows: List[dict] = []
    with open(out_csv, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            fn = row["function_name"]
            if args.filter and args.filter not in fn:
                continue

            def fnum(key: str) -> Optional[float]:
                value = _clean(row.get(key))
                if not value:
                    return None
                try:
                    return float(value)
                except Exception:
                    return None

            summary_rows.append(
                {
                    "function_name": fn,
                    "family": extract_family(fn),
                    "data_type": row["data_type"],
                    "channels": row["channels"],
                    "scenarios_total": int(row["scenarios_total"] or 0),
                    "scenarios_compared_before_after": int(row["scenarios_compared_before_after"] or 0),
                    "scenarios_compared_after_nvidia": int(row["scenarios_compared_after_nvidia"] or 0),
                    "improve_pct": fnum("improve_pct"),
                    "gap_pct": fnum("gap_pct_after_over_nvidia"),
                    "catchup_pct": fnum("catchup_pct"),
                    "after_fail_count": int(row["after_fail_count"] or 0),
                }
            )

    if args.html:
        write_html_report(args.html, before_label, after_label, nvidia_label, summary_rows)
        print(f"Progress HTML report saved to: {args.html}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
