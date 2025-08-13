from __future__ import annotations

import io
import json
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask import send_file
import base64

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.units import inch

# Reuse metrics computation from the existing analysis
from analyze_performance import compute_metrics, Metrics

# New: NeoLoad XML parsing
import xml.etree.ElementTree as ET

app = Flask(__name__)


def _json_error(status: int, message: str, details: Dict[str, Any] | None = None):
    payload = {"error": message}
    if details:
        payload["details"] = details
    response = jsonify(payload)
    response.status_code = status
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return response


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return response

# Serve the dashboard UI
import os
from flask import send_from_directory

@app.route("/", methods=["GET"])  # Root serves the dashboard
def index():
    resp = send_from_directory(os.path.dirname(__file__), "dashboard.html")
    try:
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
    except Exception:
        pass
    return resp

@app.route("/dashboard.html", methods=["GET"])  # Convenience path
def dashboard_html():
    resp = send_from_directory(os.path.dirname(__file__), "dashboard.html")
    try:
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
    except Exception:
        pass
    return resp

@app.route("/favicon.ico", methods=["GET"])  # Avoid 404 noise
def favicon():
    return ("", 204)

@app.route("/api/analyze", methods=["OPTIONS"])  # Preflight
def analyze_options():
    return ("", 204)


def detect_tool(file_bytes: bytes, filename: str) -> str:
    name_lower = (filename or "").lower()
    # Simple filename heuristic first
    if name_lower.endswith(".csv"):
        return "jmeter"
    if name_lower.endswith(".json"):
        return "k6"
    if name_lower.endswith(".xml"):
        return "neoload"

    # Try JSON content detection
    head = file_bytes[:4096].lstrip()
    try:
        sample = json.loads(head.decode("utf-8", errors="ignore"))
        if isinstance(sample, dict):
            # k6 summary JSON has top-level "metrics"
            if "metrics" in sample:
                return "k6"
            if str(sample.get("type", "")).lower() == "point":
                return "k6"
        if isinstance(sample, list) and sample:
            if any(str(getattr(x, "get", lambda *_: None)("type"), "").lower() == "point" for x in sample if isinstance(x, dict)):
                return "k6"
    except Exception:
        pass

    # Try CSV header detection
    try:
        head_text = head.decode("utf-8", errors="ignore")
        first_line = head_text.splitlines()[0] if head_text.splitlines() else ""
        if "label" in first_line and "," in first_line:
            return "jmeter"
    except Exception:
        pass

    # Try XML detection
    try:
        text = head.decode("utf-8", errors="ignore").strip()
        if text.startswith("<?xml") or text.startswith("<report") or "</report>" in text:
            return "neoload"
    except Exception:
        pass

    return "unknown"


def parse_jmeter_csv(file_bytes: bytes) -> pd.DataFrame:
    buffer = io.BytesIO(file_bytes)
    df = pd.read_csv(buffer, low_memory=False)

    required_cols = {"timestamp", "elapsed", "label", "responseCode"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df["elapsed"] = pd.to_numeric(df["elapsed"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["responseCode"] = df["responseCode"].astype(str)
    df = df.dropna(subset=["elapsed", "timestamp"]).copy()

    # Cast timestamp to int64 milliseconds if possible (handle scientific notation)
    df["timestamp"] = df["timestamp"].astype(np.int64)

    # Derived fields
    df["is_success"] = df["responseCode"].astype(str).str.startswith("2")
    df["end_timestamp"] = df["timestamp"] + df["elapsed"].astype(np.int64)
    return df


def parse_k6_json(file_bytes: bytes) -> pd.DataFrame:
    text = file_bytes.decode("utf-8", errors="ignore").strip()

    # First try: k6 summary metrics JSON (single object with "metrics")
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "metrics" in data:
            metrics_obj = data.get("metrics", {}) or {}
            state_obj = data.get("state", {}) or {}

            dur_vals = (metrics_obj.get("http_req_duration", {}) or {}).get("values", {}) or {}
            reqs_vals = (metrics_obj.get("http_reqs", {}) or {}).get("values", {}) or {}
            failed_vals = (metrics_obj.get("http_req_failed", {}) or {}).get("values", {}) or {}

            avg_latency_ms = float(dur_vals.get("avg") or 0.0)
            total_requests = int(reqs_vals.get("count") or 0)
            failed_count = failed_vals.get("count")
            if failed_count is None:
                rate = float(failed_vals.get("rate") or 0.0)
                failed_count = int(round(rate * total_requests)) if total_requests else 0
            else:
                failed_count = int(failed_count)
            success_count = max(0, total_requests - failed_count)

            # Determine test duration in milliseconds
            test_duration_ms = None
            if isinstance(state_obj.get("testRunDurationMs"), (int, float)):
                test_duration_ms = float(state_obj.get("testRunDurationMs"))
            else:
                # Fallback: derive from rate if available
                rate = reqs_vals.get("rate")
                if rate:
                    try:
                        test_duration_ms = float(total_requests) / float(rate) * 1000.0
                    except Exception:
                        test_duration_ms = None
            if not test_duration_ms or not math.isfinite(test_duration_ms) or test_duration_ms <= 0:
                # Safe fallback: spread by 10ms per request
                test_duration_ms = max(1.0, float(total_requests)) * 10.0

            # Build a normalized DataFrame
            n = int(total_requests) if total_requests else (success_count + failed_count)
            if n <= 0:
                # No requests — return empty frame with required columns
                return pd.DataFrame(columns=["timestamp", "elapsed", "label", "responseCode", "is_success", "end_timestamp"])  # type: ignore[list-item]

            # Evenly space timestamps across duration
            timestamps = np.linspace(0, max(test_duration_ms - max(avg_latency_ms, 0.0), 0.0), num=n)
            # Response codes: successes as 200, failures as 500
            codes: List[str] = ["200"] * int(success_count) + ["500"] * int(failed_count)
            if len(codes) < n:
                codes += ["200"] * (n - len(codes))
            elif len(codes) > n:
                codes = codes[:n]

            # Synthesize a distribution using available summary stats so quantiles are distinct
            def _as_float(val, default):
                try:
                    return float(val)
                except Exception:
                    return float(default)

            min_v = _as_float(dur_vals.get("min"), avg_latency_ms)
            med_v = _as_float(dur_vals.get("med"), avg_latency_ms)
            p90_v = _as_float(dur_vals.get("p(90)") or dur_vals.get("p90"), med_v)
            p95_v = _as_float(dur_vals.get("p(95)") or dur_vals.get("p95"), p90_v)
            max_v = _as_float(dur_vals.get("max"), max(med_v, p95_v))
            # Ensure non-decreasing order
            med_v = max(med_v, min_v)
            p90_v = max(p90_v, med_v)
            p95_v = max(p95_v, p90_v)
            max_v = max(max_v, p95_v)

            n50 = int(round(0.50 * n))
            n90 = int(round(0.40 * n))
            n95 = int(round(0.05 * n))
            n99 = max(0, n - (n50 + n90 + n95))
            chunks: List[np.ndarray] = []
            if n50 > 0:
                chunks.append(np.full(n50, med_v, dtype=float))
            if n90 > 0:
                if p90_v != med_v:
                    chunks.append(np.linspace(med_v, p90_v, num=n90, endpoint=False, dtype=float))
                else:
                    chunks.append(np.full(n90, p90_v, dtype=float))
            if n95 > 0:
                if p95_v != p90_v:
                    chunks.append(np.linspace(p90_v, p95_v, num=n95, endpoint=False, dtype=float))
                else:
                    chunks.append(np.full(n95, p95_v, dtype=float))
            if n99 > 0:
                if max_v != p95_v:
                    chunks.append(np.linspace(p95_v, max_v, num=n99, dtype=float))
                else:
                    chunks.append(np.full(n99, max_v, dtype=float))
            elapsed_values = np.concatenate(chunks) if chunks else np.full(n, avg_latency_ms, dtype=float)
            # Adjust size in case of rounding mismatch
            if len(elapsed_values) < n:
                pad = np.full(n - len(elapsed_values), avg_latency_ms, dtype=float)
                elapsed_values = np.concatenate([elapsed_values, pad])
            elif len(elapsed_values) > n:
                elapsed_values = elapsed_values[:n]

            df = pd.DataFrame(
                {
                    "timestamp": timestamps.astype(np.int64),
                    "elapsed": elapsed_values.astype(float),
                    "label": ["http_req"] * n,
                    "responseCode": codes,
                }
            )
            df["responseCode"] = df["responseCode"].astype(str)
            df["is_success"] = df["responseCode"].str.startswith("2") | df["responseCode"].str.startswith("3")
            df["end_timestamp"] = df["timestamp"] + df["elapsed"].astype(np.int64)
            return df
    except Exception:
        pass

    # Support JSON array or JSONL of k6 output points
    items: List[Dict[str, Any]] = []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            items = [x for x in data if isinstance(x, dict)]
        elif isinstance(data, dict):
            items = [data]
    except Exception:
        # Try JSON Lines
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    items.append(obj)
            except Exception:
                continue

    if not items:
        raise ValueError("No JSON objects found in uploaded file")

    rows: List[Dict[str, Any]] = []
    for obj in items:
        type_str = str(obj.get("type", "")).lower()
        if type_str != "point":
            continue

        metric = obj.get("metric") or obj.get("name") or "metric"
        data = obj.get("data", {}) or {}
        tags = obj.get("tags", {}) or {}

        # Duration value
        value = data.get("value")
        if value is None:
            value = obj.get("value")
        if value is None:
            continue

        try:
            elapsed_ms = float(value)
        except Exception:
            # Skip non-numeric values
            continue

        # Heuristic: if value looks like seconds (< 10) and there is a "unit"
        unit = data.get("unit") or obj.get("unit")
        if unit:
            if str(unit).lower() in {"s", "sec", "second", "seconds"}:
                elapsed_ms *= 1000.0
            elif str(unit).lower() in {"ms", "millisecond", "milliseconds"}:
                pass
            elif str(unit).lower() in {"us", "microsecond", "microseconds"}:
                elapsed_ms /= 1000.0
            elif str(unit).lower() in {"ns", "nanosecond", "nanoseconds"}:
                elapsed_ms /= 1_000_000.0
        else:
            # If the magnitude looks like seconds, convert to ms
            if elapsed_ms < 10.0:
                elapsed_ms *= 1000.0

        # Timestamp
        ts = obj.get("time") or data.get("time") or obj.get("ts") or obj.get("timestamp")
        ts_ms: float | None = None
        if ts is not None:
            try:
                ts_val = float(ts)
                # Heuristics: detect ns/us/s vs ms
                if ts_val > 1e15:  # ns
                    ts_ms = ts_val / 1_000_000.0
                elif ts_val > 1e12:  # us
                    ts_ms = ts_val / 1_000.0
                elif ts_val > 1e10:  # ms in scientific format or large epoch
                    ts_ms = ts_val
                elif ts_val > 1e9:  # seconds epoch
                    ts_ms = ts_val * 1000.0
                else:
                    ts_ms = ts_val  # assume already ms scale
            except Exception:
                pass

        # Fallback sequential timestamp if missing
        if ts_ms is None:
            ts_ms = float(len(rows))

        # Label and response code
        label = tags.get("name") or tags.get("url") or tags.get("path") or metric
        status = str(tags.get("status", ""))
        if not status:
            status = "200"
        is_success = status.startswith("2") or status.startswith("3")

        rows.append(
            {
                "timestamp": ts_ms,
                "elapsed": elapsed_ms,
                "label": label,
                "responseCode": status,
            }
        )

    if not rows:
        raise ValueError("No k6 'point' records with numeric values found")

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["elapsed"] = pd.to_numeric(df["elapsed"], errors="coerce")
    df = df.dropna(subset=["timestamp", "elapsed"]).copy()

    # Normalize
    df["timestamp"] = df["timestamp"].astype(np.int64)
    df["responseCode"] = df["responseCode"].astype(str)
    df["is_success"] = df["responseCode"].str.startswith("2") | df["responseCode"].str.startswith("3")
    df["end_timestamp"] = df["timestamp"] + df["elapsed"].astype(np.int64)
    return df


# New: NeoLoad XML parsing
def parse_neoload_xml(file_bytes: bytes) -> pd.DataFrame:
    try:
        root = ET.fromstring(file_bytes.decode("utf-8", errors="ignore"))
    except Exception:
        # try bytes directly
        root = ET.fromstring(file_bytes)

    # Find <test>
    test_el = root.find(".//test")
    start_ms = None
    end_ms = None
    duration_s = None
    if test_el is not None:
        if test_el.get("start"):
            try:
                start_ms = float(test_el.get("start"))
            except Exception:
                start_ms = None
        if test_el.get("end"):
            try:
                end_ms = float(test_el.get("end"))
            except Exception:
                end_ms = None
        if test_el.get("duration"):
            try:
                duration_s = float(test_el.get("duration"))
            except Exception:
                duration_s = None

    if start_ms is not None and end_ms is not None:
        test_duration_ms = max(0.0, end_ms - start_ms)
    elif duration_s is not None:
        test_duration_ms = max(0.0, duration_s * 1000.0)
        start_ms = 0.0
    else:
        test_duration_ms = 0.0
        start_ms = 0.0

    # Summary counts
    success_count = 0
    failure_count = 0
    summary_el = root.find(".//statistics/summary")
    if summary_el is not None:
        try:
            success_count = int(float(summary_el.get("successCount") or 0))
        except Exception:
            success_count = 0
        try:
            failure_count = int(float(summary_el.get("failureCount") or 0))
        except Exception:
            failure_count = 0

    total_requests = success_count + failure_count

    # Transactions
    tx_els = root.findall(".//statistics/transactions/transaction")
    transactions: List[Dict[str, Any]] = []
    for tx in tx_els:
        name = tx.get("name") or "transaction"
        try:
            avg = float(tx.get("avgResponseTime") or 0.0)
        except Exception:
            avg = 0.0
        try:
            fail_rate_pct = float(tx.get("failureRate") or 0.0)
        except Exception:
            fail_rate_pct = 0.0
        # p90 is extracted but not directly used in DF; kept for potential future use
        try:
            p90 = float(tx.get("p90") or 0.0)
        except Exception:
            p90 = 0.0
        transactions.append({"name": name, "avg": avg, "fail_rate_pct": fail_rate_pct, "p90": p90})

    if not transactions:
        # If no per-transaction data, synthesize from summary
        if total_requests <= 0:
            return pd.DataFrame(columns=["timestamp", "elapsed", "label", "responseCode", "is_success", "end_timestamp"])  # type: ignore[list-item]
        avg_overall = 0.0
        if summary_el is not None and summary_el.get("avgResponseTime"):
            try:
                avg_overall = float(summary_el.get("avgResponseTime"))
            except Exception:
                avg_overall = 0.0
        n = int(total_requests)
        timestamps = np.linspace(start_ms or 0.0, (start_ms or 0.0) + max(test_duration_ms - avg_overall, 0.0), num=n)
        codes: List[str] = ["200"] * int(success_count) + ["500"] * int(failure_count)
        if len(codes) < n:
            codes += ["200"] * (n - len(codes))
        elif len(codes) > n:
            codes = codes[:n]
        df = pd.DataFrame(
            {
                "timestamp": timestamps.astype(np.int64),
                "elapsed": np.full(n, float(avg_overall), dtype=float),
                "label": ["neoload"] * n,
                "responseCode": codes,
            }
        )
        df["responseCode"] = df["responseCode"].astype(str)
        df["is_success"] = df["responseCode"].str.startswith("2") | df["responseCode"].str.startswith("3")
        df["end_timestamp"] = df["timestamp"] + df["elapsed"].astype(np.int64)
        return df

    # Allocate counts per transaction
    num_tx = len(transactions)
    if total_requests <= 0:
        # If no total count, synthesize one request per transaction
        counts = [1] * num_tx
        total_requests = num_tx
        success_count = num_tx
        failure_count = 0
    else:
        base = total_requests // num_tx
        rem = total_requests % num_tx
        counts = [base + (1 if i < rem else 0) for i in range(num_tx)]

    rows: List[Dict[str, Any]] = []
    # Build per-row data according to counts and failure rate per transaction
    for idx, tx in enumerate(transactions):
        count_i = int(counts[idx])
        if count_i <= 0:
            continue
        fail_i = int(round(count_i * (tx["fail_rate_pct"] / 100.0)))
        succ_i = max(0, count_i - fail_i)
        # Success rows
        for _ in range(succ_i):
            rows.append({
                "label": tx["name"],
                "elapsed": float(tx["avg"]),
                "responseCode": "200",
            })
        # Failure rows
        for _ in range(fail_i):
            rows.append({
                "label": tx["name"],
                "elapsed": float(tx["avg"]),
                "responseCode": "500",
            })

    n = len(rows)
    if n == 0:
        return pd.DataFrame(columns=["timestamp", "elapsed", "label", "responseCode", "is_success", "end_timestamp"])  # type: ignore[list-item]

    # Evenly space timestamps across duration
    timestamps = np.linspace(start_ms or 0.0, (start_ms or 0.0) + max(test_duration_ms - min([r["elapsed"] for r in rows] or [0.0]), 0.0), num=n)
    for i in range(n):
        rows[i]["timestamp"] = float(timestamps[i])

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype(np.int64)
    df["elapsed"] = pd.to_numeric(df["elapsed"], errors="coerce")
    df["responseCode"] = df["responseCode"].astype(str)
    df["is_success"] = df["responseCode"].str.startswith("2") | df["responseCode"].str.startswith("3")
    df["end_timestamp"] = df["timestamp"] + df["elapsed"].astype(np.int64)
    return df


# Enriched chart data including new visualizations
def prepare_chart_data(df: pd.DataFrame) -> Dict[str, Any]:
    # Latency over time
    lot_df = df.sort_values("timestamp")[ ["timestamp", "elapsed" ] ].copy()
    latency_over_time = [
        {"timestamp": int(row.timestamp), "latency_ms": float(row.elapsed)}
        for row in lot_df.itertuples(index=False)
    ]

    # Top 10 slowest by label
    by_label = (
        df.groupby("label")["elapsed"].agg(["count", "mean", "median", "max"]).sort_values("mean", ascending=False)
    )
    top10 = by_label.head(10).reset_index()
    top10_slowest = [
        {
            "label": str(r.label),
            "count": int(r["count"]),
            "mean": float(r["mean"]),
            "median": float(r["median"]),
            "max": float(r["max"]),
        }
        for _, r in top10.iterrows()
    ]

    # Errors by endpoint (for pie chart)
    error_df = df.loc[~df["is_success"]]
    errors_by_label_series = error_df.groupby("label").size().sort_values(ascending=False)
    error_distribution = [
        {"label": str(idx), "error_count": int(val)} for idx, val in errors_by_label_series.items()
    ]

    # Latency histogram buckets
    all_lat = df["elapsed"].to_numpy(dtype=float)
    if all_lat.size > 0:
        bins = min(30, max(5, int(round(math.sqrt(all_lat.size)))))
        counts, bin_edges = np.histogram(all_lat, bins=bins)
        # Use midpoints for labeling
        labels = []
        for i in range(len(bin_edges) - 1):
            start = float(bin_edges[i])
            end = float(bin_edges[i + 1])
            labels.append(f"{start:.0f}-{end:.0f}")
        latency_histogram = [
            {"bucket": labels[i], "count": int(counts[i])} for i in range(len(counts))
        ]
    else:
        latency_histogram = []

    # Throughput over time (requests started per second)
    if len(df) > 0:
        sec_series = (df["timestamp"] // 1000).astype(np.int64)
        throughput = sec_series.value_counts().sort_index()
        throughput_over_time = [
            {"timestamp": int(sec * 1000), "rps": int(count)} for sec, count in throughput.items()
        ]
    else:
        throughput_over_time = []

    return {
        "latency_over_time": latency_over_time,
        "top10_slowest": top10_slowest,
        "error_distribution": error_distribution,
        "latency_histogram": latency_histogram,
        "throughput_over_time": throughput_over_time,
    }


def _compute_per_endpoint_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    # Error counts per endpoint
    error_counts = (
        df.loc[~df["is_success"]].groupby("label").size().sort_values(ascending=False).rename("error_count")
    )

    # Latency stats per endpoint
    def _quantile(g: pd.Series, q: float) -> float:
        try:
            return float(g.quantile(q))
        except Exception:
            return float("nan")

    stats = df.groupby("label")["elapsed"].agg(count="count", average="mean", p50="median")
    stats = stats.assign(
        p90=df.groupby("label")["elapsed"].quantile(0.90),
        p95=df.groupby("label")["elapsed"].quantile(0.95),
    ).reset_index()

    # Attach error counts
    stats["error_count"] = stats["label"].map(error_counts).fillna(0).astype(int)

    latency_by_label = [
        {
            "label": str(r["label"]),
            "count": int(r["count"]),
            "average": float(r["average"]),
            "p50": float(r["p50"]),
            "p90": float(r["p90"]),
            "p95": float(r["p95"]),
            "error_count": int(r["error_count"]),
        }
        for _, r in stats.iterrows()
    ]

    # Throughput over time is already part of charts, but include in analysis for completeness
    if len(df) > 0:
        sec_series = (df["timestamp"] // 1000).astype(np.int64)
        throughput = sec_series.value_counts().sort_index()
        throughput_over_time = [
            {"timestamp": int(sec * 1000), "rps": int(count)} for sec, count in throughput.items()
        ]
    else:
        throughput_over_time = []

    errors_by_label = [
        {"label": str(lbl), "error_count": int(cnt)} for lbl, cnt in error_counts.items()
    ]

    return {
        "errors_by_label": errors_by_label,
        "latency_by_label": latency_by_label,
        "throughput_over_time": throughput_over_time,
    }


def metrics_to_dict(m: Metrics) -> Dict[str, Any]:
    return {
        "average_latency_ms": float(m.average_latency_ms) if not math.isnan(m.average_latency_ms) else None,
        "median_latency_ms": float(m.median_latency_ms) if not math.isnan(m.median_latency_ms) else None,
        "p90_latency_ms": float(m.p90_latency_ms) if not math.isnan(m.p90_latency_ms) else None,
        "p95_latency_ms": float(m.p95_latency_ms) if not math.isnan(m.p95_latency_ms) else None,
        "throughput_rps": float(m.throughput_rps),
        "error_rate_pct": float(m.error_rate_pct),
        "total_requests": int(m.total_requests),
        "total_successes": int(m.total_successes),
        "total_errors": int(m.total_errors),
        "test_duration_seconds": float(m.test_duration_seconds),
    }


def analyze_file_bytes(file_bytes: bytes, filename: str) -> Tuple[str, Dict[str, Any]]:
    tool = detect_tool(file_bytes, filename)
    if tool == "jmeter":
        df = parse_jmeter_csv(file_bytes)
    elif tool == "k6":
        df = parse_k6_json(file_bytes)
    elif tool == "neoload":
        df = parse_neoload_xml(file_bytes)
    else:
        # Try CSV/JSON/XML as fallback
        last_err: Exception | None = None
        for fallback_tool, parser in (
            ("jmeter", parse_jmeter_csv),
            ("k6", parse_k6_json),
            ("neoload", parse_neoload_xml),
        ):
            try:
                df = parser(file_bytes)
                tool = fallback_tool
                break
            except Exception as e_any:
                last_err = e_any
                continue
        else:
            raise ValueError("Unsupported or unrecognized file format")

    metrics = compute_metrics(df)
    charts = prepare_chart_data(df)
    analysis = _compute_per_endpoint_analysis(df)

    # Build metrics dict and include overall throughput based on total requests
    metrics_dict = metrics_to_dict(metrics)
    duration = float(metrics_dict.get("test_duration_seconds") or 0.0)
    total_requests = int(metrics_dict.get("total_requests") or 0)
    if duration > 0:
        metrics_dict["overall_throughput_rps"] = float(total_requests) / duration
    else:
        metrics_dict["overall_throughput_rps"] = 0.0

    return tool, {
        "tool": tool,
        "metrics": metrics_dict,
        "charts": charts,
        "analysis": analysis,
    }


def _build_findings_text(metrics: Dict[str, Any], charts: Dict[str, Any], analysis: Dict[str, Any]) -> str:
    parts: List[str] = []
    # Executive summary line
    parts.append(
        (
            f"Test executed {int(metrics.get('total_requests', 0)):,} requests over "
            f"{float(metrics.get('test_duration_seconds', 0.0)):.2f}s with average latency "
            f"{float(metrics.get('average_latency_ms', 0.0)):.0f} ms and throughput "
            f"{float(metrics.get('throughput_rps', 0.0)):.2f} req/s. Error rate was "
            f"{float(metrics.get('error_rate_pct', 0.0)):.2f}%."
        )
    )

    # Bottleneck insight from top10_slowest
    top10 = charts.get("top10_slowest") or []
    if top10:
        worst = top10[0]
        parts.append(
            f"Bottleneck: '{worst['label']}' shows highest mean latency {worst['mean']:.0f} ms (50th percentile {worst['median']:.0f} ms, max {worst['max']:.0f} ms)."
        )
    else:
        parts.append("Bottleneck: No endpoints available for latency ranking.")

    # Error concentration
    err_by_label = analysis.get("errors_by_label") or []
    if err_by_label:
        top_err = err_by_label[0]
        parts.append(
            f"Errors concentrated on '{top_err['label']}' with {int(top_err['error_count'])} failures."
        )
    else:
        parts.append("No errors recorded during the test.")

    # Throughput pattern
    thr = charts.get("throughput_over_time") or []
    if thr:
        rps_values = [p.get("rps", 0) for p in thr]
        if rps_values:
            peak = max(rps_values)
            avg = sum(rps_values) / max(len(rps_values), 1)
            if peak > avg * 1.5 and peak - avg >= 5:
                parts.append("Throughput spiked significantly relative to the average; verify ramp-up and any autoscaling events.")
            else:
                parts.append("Throughput remained relatively steady with no severe spikes detected.")

    # Recommendations
    rec_label = top10[0]["label"] if top10 else None
    if rec_label:
        parts.append(
            f"Recommendation: Investigate server-side processing for '{rec_label}', including database queries and upstream calls. Add tracing around the slowest handlers."
        )
    else:
        parts.append(
            "Recommendation: Review test coverage and data collection; insufficient data for endpoint-level recommendations."
        )

    return "\n".join(parts)


def _decode_data_url(data_url: str) -> bytes:
    if not data_url:
        return b""
    if "," in data_url:
        b64 = data_url.split(",", 1)[1]
    else:
        b64 = data_url
    return base64.b64decode(b64)


def generate_pdf_report(payload: Dict[str, Any]) -> bytes:
    tool = payload.get("tool")
    metrics = payload.get("metrics") or {}
    charts = payload.get("charts") or {}
    analysis = payload.get("analysis") or {}
    chart_images = payload.get("chartImages") or {}

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title="Performance Test Report")
    styles = getSampleStyleSheet()
    story: List[Any] = []

    # Title
    title = f"Performance Test Analysis Report" + (f" — {tool.upper()}" if tool else "")
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 12))

    # Executive Summary (non-technical)
    story.append(Paragraph("Executive Summary", styles["Heading2"]))
    summary_text = _build_findings_text(metrics, charts, analysis)
    story.append(Spacer(1, 6))
    for line in summary_text.split("\n"):
        story.append(Paragraph(line, styles["BodyText"]))
        story.append(Spacer(1, 4))
    story.append(Spacer(1, 8))

    # Key Metrics table (overall)
    table_data = [
        ["Metric", "Value"],
        ["Total Requests", f"{int(metrics.get('total_requests', 0)):,}"],
        ["Successful Requests", f"{int(metrics.get('total_successes', 0)):,}"],
        ["Errors", f"{int(metrics.get('total_errors', 0)):,}"],
        ["Error Rate (%)", f"{float(metrics.get('error_rate_pct', 0.0)):.2f}"],
        ["Test Duration (s)", f"{float(metrics.get('test_duration_seconds', 0.0)):.2f}"],
        ["Throughput (req/s)", f"{float(metrics.get('throughput_rps', 0.0)):.2f}"],
        ["Average Latency (ms)", f"{float(metrics.get('average_latency_ms', 0.0)):.2f}"],
        ["50th percentile (ms)", f"{float(metrics.get('median_latency_ms', 0.0)):.2f}"],
        ["90th percentile (ms)", f"{float(metrics.get('p90_latency_ms', 0.0)):.2f}"],
        ["95th percentile (ms)", f"{float(metrics.get('p95_latency_ms', 0.0)):.2f}"],
    ]
    tbl = Table(table_data, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.black),
        ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("BACKGROUND", (0,1), (-1,-1), colors.whitesmoke),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 12))

    # Per-endpoint statistics table
    endpoint_rows = analysis.get("latency_by_label") or []
    if endpoint_rows:
        story.append(Paragraph("Per-Endpoint Statistics", styles["Heading2"]))
        endpoint_table = [[
            "Endpoint", "Count", "Errors", "Avg (ms)", "50th percentile", "90th percentile", "95th percentile"
        ]]
        for r in endpoint_rows:
            endpoint_table.append([
                str(r.get("label", "")),
                f"{int(r.get('count', 0)):,}",
                f"{int(r.get('error_count', 0)):,}",
                f"{float(r.get('average', 0.0)):.1f}",
                f"{float(r.get('p50', 0.0)):.1f}",
                f"{float(r.get('p90', 0.0)):.1f}",
                f"{float(r.get('p95', 0.0)):.1f}",
            ])
        endpoint_tbl = Table(endpoint_table, hAlign="LEFT")
        endpoint_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.black),
            ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
            ("ALIGN", (0,0), (-1,-1), "LEFT"),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ]))
        story.append(endpoint_tbl)
        story.append(Spacer(1, 12))

    # Visualizations
    story.append(Paragraph("Visualizations", styles["Heading2"]))
    story.append(Spacer(1, 8))

    def _add_chart(title: str, key: str):
        img_bytes = _decode_data_url(chart_images.get(key, ""))
        if img_bytes:
            buf_ = io.BytesIO(img_bytes)
            story.append(Paragraph(title, styles["Heading3"]))
            story.append(RLImage(buf_, width=6.5*inch, height=3.0*inch))
            story.append(Spacer(1, 10))

    _add_chart("Latency Over Time", "latency_over_time")
    _add_chart("Top 10 Slowest Requests", "top10_slowest")
    _add_chart("Error Distribution by Endpoint", "error_distribution")
    _add_chart("Latency Distribution Histogram", "latency_histogram")
    _add_chart("Throughput Over Time", "throughput_over_time")

    # AI-Powered Analysis and Recommendations
    story.append(Paragraph("AI-Powered Analysis and Recommendations", styles["Heading2"]))
    ai_text = []
    # Identify specific endpoints with both high latency and errors
    high_lat = sorted(endpoint_rows, key=lambda r: (-(r.get("average") or 0.0), -(r.get("error_count") or 0)))
    if high_lat:
        top = high_lat[0]
        ai_text.append(
            f"Investigate the server-side processing for '{top.get('label')}', which has high average latency "
            f"({float(top.get('average', 0.0)):.0f} ms) and {int(top.get('error_count', 0))} errors."
        )
    err_rows = analysis.get("errors_by_label") or []
    if err_rows:
        worst_err = err_rows[0]
        ai_text.append(
            f"Prioritize reliability fixes on '{worst_err.get('label')}' with {int(worst_err.get('error_count', 0))} errors (e.g., retry strategy, timeouts)."
        )
    thr_rows = charts.get("throughput_over_time") or []
    if thr_rows:
        rps_vals = [p.get("rps", 0) for p in thr_rows]
        if rps_vals:
            if max(rps_vals) > 0 and (max(rps_vals) - (sum(rps_vals)/len(rps_vals))) > 5:
                ai_text.append("Correlate throughput spikes with latency to detect saturation or autoscaling lag.")
            else:
                ai_text.append("Throughput appeared stable; focus on reducing 90th/95th percentile latencies.")
    if not ai_text:
        ai_text.append("No critical issues detected from the available data. Consider expanding test coverage.")

    for line in ai_text:
        story.append(Paragraph(f"- {line}", styles["BodyText"]))
        story.append(Spacer(1, 4))

    doc.build(story)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes


@app.route("/api/download-report", methods=["POST"])  # returns a PDF file
def download_report():
    # Expect JSON body with metrics, charts, analysis and base64 chartImages
    try:
        data = request.get_json(force=True, silent=False)
    except Exception as e:
        return _json_error(400, "Invalid JSON payload", {"reason": str(e)})

    if not isinstance(data, dict):
        return _json_error(400, "JSON body must be an object with 'metrics' and 'charts'")
    if not data.get("metrics"):
        return _json_error(400, "Missing required 'metrics' in request body")

    try:
        pdf = generate_pdf_report(data)
        return send_file(
            io.BytesIO(pdf),
            mimetype="application/pdf",
            as_attachment=True,
            download_name="performance_report.pdf",
        )
    except Exception as e:
        return _json_error(500, "Failed to generate PDF report", {"reason": str(e)})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return _json_error(400, "Missing file in multipart/form-data under field name 'file'")

    up = request.files["file"]
    filename = up.filename or "uploaded"

    try:
        content = up.read()
        if not content:
            return _json_error(400, "Uploaded file is empty")
    except Exception as e:
        return _json_error(400, "Unable to read uploaded file", {"reason": str(e)})

    try:
        tool, payload = analyze_file_bytes(content, filename)
    except ValueError as ve:
        return _json_error(422, str(ve))
    except Exception as e:
        return _json_error(500, "Failed to analyze file", {"reason": str(e)})

    response = jsonify(payload)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return response


if __name__ == "__main__":
    # Local dev server
    app.run(host="0.0.0.0", port=5000, debug=True) 