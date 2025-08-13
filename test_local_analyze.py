from app import analyze_file_bytes

if __name__ == "__main__":
    with open("Results.csv", "rb") as f:
        tool, payload = analyze_file_bytes(f.read(), "Results.csv")
    print("tool:", tool)
    m = payload["metrics"]
    print("metrics:", {k: m[k] for k in [
        "total_requests","total_successes","total_errors",
        "error_rate_pct","throughput_rps","average_latency_ms","p90_latency_ms"
    ]})
    print("latency_over_time_count:", len(payload["charts"]["latency_over_time"]))
    print("top10_slowest_count:", len(payload["charts"]["top10_slowest"])) 