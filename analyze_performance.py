#!/usr/bin/env python3
"""
Analyze a JMeter-style results CSV and generate metrics, visualizations, and a Markdown report.

Inputs
- CSV with columns: timestamp, elapsed, label, responseCode, responseMessage, threadName, success, bytes, sentBytes, grpThreads, allThreads, URL

Outputs
- Console: Key performance metrics
- Files:
  - charts/latency_over_time.png
  - charts/latency_distribution.png
  - charts/top10_slowest_requests.png
  - charts/error_rate_by_endpoint.png
  - report.md (Markdown executive summary)

Usage
  python analyze_performance.py --input Results.csv --output report.md --charts-dir charts
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style="whitegrid")


@dataclass
class Metrics:
    average_latency_ms: float
    median_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    throughput_rps: float
    error_rate_pct: float
    total_requests: int
    total_successes: int
    total_errors: int
    test_duration_seconds: float


def read_results_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        # Fallback to common case-insensitive name on Windows
        alt = "results.csv" if os.path.basename(csv_path).lower() != "results.csv" else "Results.csv"
        if os.path.exists(alt):
            csv_path = alt
        else:
            raise FileNotFoundError(f"Could not find CSV file at '{csv_path}' or '{alt}'.")

    df = pd.read_csv(csv_path, low_memory=False)

    required_cols = {"timestamp", "elapsed", "label", "responseCode"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Normalize types
    df["elapsed"] = pd.to_numeric(df["elapsed"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["responseCode"] = df["responseCode"].astype(str)

    # Drop rows with missing core fields
    df = df.dropna(subset=["elapsed", "timestamp"]).copy()

    # Cast timestamp to int64 milliseconds if possible
    df["timestamp"] = df["timestamp"].astype(np.int64)

    # Add derived fields
    df["is_success"] = df["responseCode"] == "200"
    df["end_timestamp"] = df["timestamp"] + df["elapsed"].astype(np.int64)

    return df


def compute_metrics(df: pd.DataFrame) -> Metrics:
    total_requests = len(df)
    total_successes = int(df["is_success"].sum())
    total_errors = int((~df["is_success"]).sum())

    average_latency_ms = float(df["elapsed"].mean()) if total_requests else float("nan")
    median_latency_ms = float(df["elapsed"].median()) if total_requests else float("nan")
    p90_latency_ms = float(df["elapsed"].quantile(0.90)) if total_requests else float("nan")
    p95_latency_ms = float(df["elapsed"].quantile(0.95)) if total_requests else float("nan")

    # Test duration from first request start to last request complete (handles identical timestamps)
    if total_requests > 0:
        start_ms = int(df["timestamp"].min())
        end_ms = int(df["end_timestamp"].max())
        duration_seconds = max(0.0, (end_ms - start_ms) / 1000.0)
    else:
        duration_seconds = 0.0

    if duration_seconds <= 0.0:
        throughput_rps = 0.0
    else:
        throughput_rps = total_successes / duration_seconds

    error_rate_pct = (total_errors / total_requests * 100.0) if total_requests else 0.0

    return Metrics(
        average_latency_ms=average_latency_ms,
        median_latency_ms=median_latency_ms,
        p90_latency_ms=p90_latency_ms,
        p95_latency_ms=p95_latency_ms,
        throughput_rps=throughput_rps,
        error_rate_pct=error_rate_pct,
        total_requests=total_requests,
        total_successes=total_successes,
        total_errors=total_errors,
        test_duration_seconds=duration_seconds,
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_latency_over_time(df: pd.DataFrame, out_path: str) -> None:
    plot_df = df.sort_values("timestamp").copy()
    plot_df["datetime"] = pd.to_datetime(plot_df["timestamp"], unit="ms", utc=True)

    plt.figure(figsize=(12, 6))
    plt.plot(plot_df["datetime"], plot_df["elapsed"], marker="o", linestyle="-", linewidth=1, markersize=3)
    plt.title("Latency Over Time")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Latency (ms)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_latency_distribution(df: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=(10, 6))
    sns.histplot(df["elapsed"], bins=30, kde=True)
    plt.title("Latency Distribution")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_top10_slowest(df: pd.DataFrame, out_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    by_label = (
        df.groupby("label")["elapsed"]
        .agg(["count", "mean", "median", "max"])  # type: ignore[arg-type]
        .sort_values("mean", ascending=False)
    )
    top10 = by_label.head(10).reset_index()

    plt.figure(figsize=(12, 6))
    sns.barplot(data=top10, x="mean", y="label", orient="h", palette="viridis")
    plt.title("Top 10 Slowest Requests by Average Latency")
    plt.xlabel("Average Latency (ms)")
    plt.ylabel("Label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return by_label, top10


def plot_errors_by_endpoint(df: pd.DataFrame, out_path: str) -> pd.DataFrame:
    error_df = df.loc[~df["is_success"]].copy()
    by_label_errors = error_df.groupby("label").size().sort_values(ascending=False).rename("error_count").reset_index()

    plt.figure(figsize=(12, 6))
    if not by_label_errors.empty:
        sns.barplot(data=by_label_errors, x="error_count", y="label", orient="h", palette="rocket")
        plt.title("Errors by Endpoint (Count)")
        plt.xlabel("Error Count")
        plt.ylabel("Label")
    else:
        plt.text(0.5, 0.5, "No errors recorded", ha="center", va="center", fontsize=14)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return by_label_errors


def print_metrics(metrics: Metrics) -> None:
    print("\n=== Key Performance Metrics ===")
    print(f"Total Requests:          {metrics.total_requests}")
    print(f"Successful Requests:     {metrics.total_successes}")
    print(f"Errors:                  {metrics.total_errors}")
    print(f"Test Duration (s):       {metrics.test_duration_seconds:.2f}")
    print(f"Throughput (req/s):      {metrics.throughput_rps:.2f}")
    print(f"Average Latency (ms):    {metrics.average_latency_ms:.2f}")
    print(f"50th percentile Latency (ms): {metrics.median_latency_ms:.2f}")
    print(f"90th percentile Latency (ms): {metrics.p90_latency_ms:.2f}")
    print(f"95th percentile Latency (ms): {metrics.p95_latency_ms:.2f}")
    print(f"Error Rate (%):          {metrics.error_rate_pct:.2f}%\n")


def build_findings(metrics: Metrics, by_label_latency: pd.DataFrame, by_label_errors: pd.DataFrame) -> Tuple[str, Dict[str, str]]:
    insights: Dict[str, str] = {}

    # Major bottleneck by highest mean latency
    if not by_label_latency.empty:
        worst = by_label_latency.iloc[0]
        worst_label = by_label_latency.index[0]
        insights["bottleneck"] = (
            f"'{worst_label}' exhibits the highest mean latency at {worst['mean']:.0f} ms "
            f"(50th percentile {worst['median']:.0f} ms, max {worst['max']:.0f} ms)."
        )
    else:
        insights["bottleneck"] = "No endpoints to analyze for latency."

    # Stability assessment based on error rate
    if metrics.error_rate_pct > 5.0:
        stability = "High error rate observed (>5%). System stability is poor."
    elif metrics.error_rate_pct > 1.0:
        stability = "Moderate error rate (1-5%). System stability is marginal."
    elif metrics.error_rate_pct > 0.0:
        stability = "Low error rate (<1%). System appears stable with minor issues."
    else:
        stability = "No errors observed. System appears stable."
    insights["stability"] = stability

    # Error concentration by endpoint
    if not by_label_errors.empty:
        top_err = by_label_errors.iloc[0]
        insights["errors_by_endpoint"] = (
            f"Errors are concentrated on '{top_err['label']}' with {int(top_err['error_count'])} occurrences."
        )
    else:
        insights["errors_by_endpoint"] = "No error concentration detected (0 errors)."

    # Recommendation
    if not by_label_latency.empty:
        rec_label = by_label_latency.index[0]
        recommendation = (
            f"Investigate server-side processing and dependent services for '{rec_label}', "
            f"including database queries and upstream calls. Add tracing around the slowest handlers."
        )
    else:
        recommendation = "Review test coverage and data collection; insufficient data for endpoint-level recommendations."
    insights["recommendation"] = recommendation

    # Compose markdown bullets
    findings_md = (
        f"- **Bottleneck**: {insights['bottleneck']}\n"
        f"- **Stability**: {insights['stability']}\n"
        f"- **Errors**: {insights['errors_by_endpoint']}\n"
        f"- **Next Steps**: {insights['recommendation']}\n"
    )

    return findings_md, insights


def write_markdown_report(
    output_path: str,
    metrics: Metrics,
    charts_paths: Dict[str, str],
    findings_md: str,
) -> None:
    md = []
    md.append("**Performance Test Analysis Report**\n")
    md.append(
        "This report summarizes the performance test results, focusing on end-user latency, throughput, and reliability. "
        "It highlights key metrics, visual trends, and recommended next steps.\n"
    )

    md.append("\n### Key Performance Metrics\n")
    md.append(
        "- **Total Requests**: {total_requests}\n"
        "- **Successful Requests**: {total_successes}\n"
        "- **Errors**: {total_errors}\n"
        "- **Test Duration (s)**: {duration:.2f}\n"
        "- **Throughput (req/s)**: {throughput:.2f}\n"
        "- **Average Latency (ms)**: {avg:.2f}\n"
        "- **50th percentile (ms)**: {p50:.2f}\n"
        "- **90th percentile (ms)**: {p90:.2f}\n"
        "- **95th percentile (ms)**: {p95:.2f}\n"
        "- **Error Rate (%)**: {err:.2f}%\n".format(
            total_requests=metrics.total_requests,
            total_successes=metrics.total_successes,
            total_errors=metrics.total_errors,
            duration=metrics.test_duration_seconds,
            throughput=metrics.throughput_rps,
            avg=metrics.average_latency_ms,
            p50=metrics.median_latency_ms,
            p90=metrics.p90_latency_ms,
            p95=metrics.p95_latency_ms,
            err=metrics.error_rate_pct,
        )
    )

    md.append("\n### Visualizations\n")
    md.append(f"![Latency Over Time]({charts_paths['latency_over_time']})\n")
    md.append(f"![Latency Distribution]({charts_paths['latency_distribution']})\n")
    md.append(f"![Top 10 Slowest Requests]({charts_paths['top10_slowest']})\n")
    md.append(f"![Errors by Endpoint]({charts_paths['errors_by_endpoint']})\n")

    md.append("\n### Key Findings and Recommendations\n")
    md.append(findings_md)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze performance test CSV and generate a report.")
    parser.add_argument("--input", default="Results.csv", help="Path to the results CSV (default: Results.csv)")
    parser.add_argument("--output", default="report.md", help="Path to write the Markdown report (default: report.md)")
    parser.add_argument("--charts-dir", default="charts", help="Directory to save charts (default: charts)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = read_results_csv(args.input)
    metrics = compute_metrics(df)

    ensure_dir(args.charts_dir)
    charts = {
        "latency_over_time": os.path.join(args.charts_dir, "latency_over_time.png"),
        "latency_distribution": os.path.join(args.charts_dir, "latency_distribution.png"),
        "top10_slowest": os.path.join(args.charts_dir, "top10_slowest_requests.png"),
        "errors_by_endpoint": os.path.join(args.charts_dir, "error_rate_by_endpoint.png"),
    }

    # Generate charts
    plot_latency_over_time(df, charts["latency_over_time"])
    plot_latency_distribution(df, charts["latency_distribution"])
    by_label_latency, top10_latency = plot_top10_slowest(df, charts["top10_slowest"])
    by_label_errors = plot_errors_by_endpoint(df, charts["errors_by_endpoint"])

    # Console metrics
    print_metrics(metrics)

    # Findings and report
    findings_md, _ = build_findings(metrics, by_label_latency, by_label_errors)
    write_markdown_report(args.output, metrics, charts, findings_md)

    print(f"Charts saved in: {os.path.abspath(args.charts_dir)}")
    print(f"Report written to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main() 