#!/usr/bin/env python3
"""
Latency Summary Tool
Reads inference_log_detailed.csv and prints mean/p95/min/max for key latency metrics.
Works without external dependencies.
"""
import argparse
import csv
import os
from typing import List, Dict, Optional, Tuple

Metric = Dict[str, float]

def parse_float(v: str) -> Optional[float]:
    try:
        v = v.strip()
        if v == "" or v.lower() in {"na", "none", "null"}:
            return None
        return float(v)
    except Exception:
        return None

# Map possible header variants to canonical keys
HEADER_ALIASES = {
    "inference time (ms)": "inf_ms",
    "transport latency (ms)": "transport_ms",
    "server ack latency (ms)": "ack_ms",
    "total response time (ms)": "e2e_ms",
    # Optional extras if present
    "confidence": "conf",
}

def canonicalize_headers(headers: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for h in headers:
        key = h.strip().lower()
        if key in HEADER_ALIASES:
            mapping[h] = HEADER_ALIASES[key]
    return mapping

def collect_metrics(rows: List[Dict[str, str]], header_map: Dict[str, str]) -> Dict[str, List[float]]:
    data: Dict[str, List[float]] = {"inf_ms": [], "transport_ms": [], "ack_ms": [], "e2e_ms": []}
    for row in rows:
        for src_h, canon in header_map.items():
            if canon in data:
                val = parse_float(row.get(src_h, ""))
                if val is not None:
                    data[canon].append(val)
    return data

def stats(values: List[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    n = len(values)
    vals = sorted(values)
    mean = sum(vals) / n
    min_v = vals[0]
    max_v = vals[-1]
    # median
    if n % 2 == 1:
        p50 = vals[n // 2]
    else:
        p50 = (vals[n // 2 - 1] + vals[n // 2]) / 2.0
    # p95 (nearest-rank with interpolation)
    import math
    pos = 0.95 * (n - 1)
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        p95 = vals[int(pos)]
    else:
        w = pos - lower
        p95 = vals[lower] * (1 - w) + vals[upper] * w
    return {"count": n, "mean": mean, "p50": p50, "p95": p95, "min": min_v, "max": max_v}

def fmt_stats(name: str, st: Optional[Dict[str, float]]) -> str:
    if st is None:
        return f"- {name}: no data"
    return (f"- {name}: count={st['count']} | mean={st['mean']:.2f} ms | p50={st['p50']:.2f} ms | "
            f"p95={st['p95']:.2f} ms | min={st['min']:.2f} ms | max={st['max']:.2f} ms")

def main():
    ap = argparse.ArgumentParser(description="Summarize latencies from CSV")
    ap.add_argument("--csv", default="inference_log_detailed.csv", help="Path to CSV log")
    ap.add_argument("--out", default=None, help="Optional text file to write summary")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV not found: {args.csv}")
        return 1

    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        header_map = canonicalize_headers(headers)
        rows = list(reader)

    data = collect_metrics(rows, header_map)
    inf_stats = stats(data.get("inf_ms", []))
    transport_stats = stats(data.get("transport_ms", []))
    ack_stats = stats([v for v in data.get("ack_ms", []) if v > 0.0])  # exclude zeros (no ACK)
    e2e_stats = stats(data.get("e2e_ms", []))

    lines = [
        "Latency Summary",
        f"Source: {args.csv}",
        fmt_stats("Inference Time", inf_stats),
        fmt_stats("Transport Latency", transport_stats),
        fmt_stats("Server ACK Latency", ack_stats),
        fmt_stats("Total Response Time", e2e_stats),
    ]

    text = "\n".join(lines)
    print(text)

    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(text + "\n")
            print(f"[OK] Written summary to {args.out}")
        except Exception as e:
            print(f"[WARN] Failed to write output: {e}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
