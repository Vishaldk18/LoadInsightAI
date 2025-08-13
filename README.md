### AI‑Powered Performance Test Result Analysis & Visualization

A lightweight tool to ingest performance test results (JMeter CSV, k6 JSON, NeoLoad XML), compute key KPIs (avg/p50/p90/p95, throughput, error rate), and render an interactive web dashboard with one‑click PDF export. Also includes a CLI to generate charts and a Markdown report.

### Features
- **Unified ingestion**: JMeter `.csv`, k6 `.json` (summary or stream/points), NeoLoad `.xml`
- **Core KPIs**: average, 50th/90th/95th percentile latency, throughput, error rate
- **Visuals**: latency over time, throughput over time, latency histogram, error distribution, top 10 slowest endpoints
- **Per‑endpoint breakdown**: avg/p50/p90/p95 and error counts
- **Export**: executive PDF report with embedded charts
- **Dark mode** dashboard UI

### Tech Stack
- **Backend**: Python (Flask, pandas, numpy)
- **Frontend**: HTML/CSS/JavaScript, Chart.js
- **Reporting**: reportlab (PDF)
- **CLI utilities**: matplotlib, seaborn (image charts)

### Supported Inputs
- **JMeter CSV**: requires columns: `timestamp`, `elapsed`, `label`, `responseCode`
- **k6 JSON**:
  - Summary JSON with `metrics.http_req_duration.values` (avg/med/p(90)/p(95)) and `metrics.http_reqs`/`metrics.http_req_failed`
  - Stream/points JSON (array or JSONL) containing objects with `type: "point"`
- **NeoLoad XML**: `<statistics>` summary with transactions (avg, p90, failure rate)

### Installation
- Python 3.x recommended (3.9+)
- From the project root:
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run the Dashboard
```bash
python app.py
```
- Open `http://localhost:5000`
- Use “Upload and Analyze” to select a results file (`.csv`, `.json`, or `.xml`)
- Optionally click “Download Report” to export a PDF

### API Endpoints
- `GET /` or `GET /dashboard.html`
  - Serves the dashboard UI
- `POST /api/analyze`
  - Multipart form with a single file under field name `file`
  - Response JSON:
```json
{
  "tool": "jmeter|k6|neoload",
  "metrics": {
    "total_requests": 0,
    "total_successes": 0,
    "total_errors": 0,
    "error_rate_pct": 0.0,
    "test_duration_seconds": 0.0,
    "throughput_rps": 0.0,
    "overall_throughput_rps": 0.0,
    "average_latency_ms": 0.0,
    "median_latency_ms": 0.0,
    "p90_latency_ms": 0.0,
    "p95_latency_ms": 0.0
  },
  "charts": {
    "latency_over_time": [{ "timestamp": 0, "latency_ms": 0.0 }],
    "top10_slowest": [{ "label": "", "count": 0, "mean": 0.0, "median": 0.0, "max": 0.0 }],
    "error_distribution": [{ "label": "", "error_count": 0 }],
    "latency_histogram": [{ "bucket": "0-10", "count": 0 }],
    "throughput_over_time": [{ "timestamp": 0, "rps": 0 }]
  },
  "analysis": {
    "latency_by_label": [{ "label": "", "count": 0, "average": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "error_count": 0 }],
    "errors_by_label": [{ "label": "", "error_count": 0 }],
    "throughput_over_time": [{ "timestamp": 0, "rps": 0 }]
  }
}
```
- `POST /api/download-report`
  - Accepts JSON with `tool`, `metrics`, `charts`, `analysis`, and `chartImages` (data URLs for current charts)
  - Returns a `application/pdf` attachment

### Metrics Definitions
- **average_latency_ms**: mean of per‑request `elapsed` (ms)
- **median_latency_ms (p50)**: 50th percentile of `elapsed` (ms)
- **p90_latency_ms / p95_latency_ms**: 90th/95th percentiles of `elapsed` (ms)
- **test_duration_seconds**: from first request start to last request completion
- **throughput_rps**: `total_successes / test_duration_seconds`
- **overall_throughput_rps**: `total_requests / test_duration_seconds`
- **error_rate_pct**: `(total_errors / total_requests) * 100`

### Charts (Dashboard)
- **Latency over Time**: time series of per‑sample latency
- **Top 10 Slowest Requests**: by mean latency per endpoint (label)
- **Error Distribution**: errors by endpoint
- **Latency Histogram**: distribution of response times
- **Throughput over Time**: requests per second

### Per‑Endpoint Analysis (API `analysis` payload)
- `latency_by_label`: `label`, `count`, `average`, `p50`, `p90`, `p95`, `error_count`
- `errors_by_label`: `label`, `error_count`

### CLI: Offline Analysis and Report
Generate charts and a Markdown summary from a JMeter‑style CSV:
```bash
python analyze_performance.py --input Results.csv --output report.md --charts-dir charts
```
Outputs:
- `charts/latency_over_time.png`
- `charts/latency_distribution.png`
- `charts/top10_slowest_requests.png`
- `charts/error_rate_by_endpoint.png`
- `report.md`

Quick preview of metrics in console is also printed by the CLI.

### Example Requests
- Analyze via cURL:
```bash
curl -F "file=@Results.csv" http://localhost:5000/api/analyze
```
- Download PDF via cURL:
```bash
curl -X POST http://localhost:5000/api/download-report \
  -H "Content-Type: application/json" \
  -d @payload.json --output performance_report.pdf
```

### Project Structure
- `app.py`: Flask API, file detection/parsing, metrics, charts, PDF report
- `dashboard.html`: standalone UI (uploads, charts, theme, PDF download)
- `analyze_performance.py`: CLI analysis, image charts, Markdown report
- `requirements.txt`: runtime dependencies
- `Results.csv`, `k6_result.json`, `results.xml`: sample inputs
- `test_local_analyze.py`: quick local analysis helper
- `ONE_PAGER.md`: executive one‑pager overview

### Troubleshooting
- 422 from `/api/analyze`: check input format and required fields (e.g., JMeter CSV must include `timestamp, elapsed, label, responseCode`)
- Empty charts: ensure file has valid rows and non‑NaN `elapsed`
- k6 inputs: tool supports both summary JSON and point streams; ensure `http_req_duration` is present (summary) or `type: "point"` entries (stream)
- NeoLoad inputs: ensure `<statistics>` with `<transaction>` entries or a `<summary>` is present
- CORS: API sets permissive CORS headers for local use; harden for production as needed

### Notes
- “Latency” in this project is full request response time (same as response time), in milliseconds.
- For Windows PowerShell, activate the venv with: `.venv\Scripts\Activate.ps1`

### License
Specify your project license here (e.g., MIT). 

Here’s the quickest, reliable path to host this Flask app on Azure App Service (Linux).

Prereqs
- Azure account and Azure CLI installed.
- Code in a git repo (local is fine).
- Add a production WSGI server.

1) Prep the app for production
- Add Gunicorn to `requirements.txt`:
```text
gunicorn>=20,<22
```
- Test locally with Gunicorn:
```bash
pip install -r requirements.txt
gunicorn --bind=0.0.0.0:8000 app:app
```

2) Provision Azure App Service (Linux) and deploy
- From your project folder:
```powershell
# Login
az login

# Set vars
$APP_NAME="your-unique-app-name"
$RG="perf-analyzer-rg"
$LOC="eastus"

# Create resource group and web app (creates plan automatically)
az group create --name $RG --location $LOC
az webapp up --name $APP_NAME --resource-group $RG --location $LOC --sku B1 --runtime "PYTHON:3.10"
```

3) Set the startup command
- Configure the app to run with Gunicorn:
```powershell
az webapp config set --resource-group $RG --name $APP_NAME --startup-file "gunicorn --bind=0.0.0.0 --timeout 600 app:app"
```

4) Redeploy (if needed)
- If you make changes later, re-run:
```powershell
az webapp up --name $APP_NAME --resource-group $RG --runtime "PYTHON:3.10"
```

5) Verify and tail logs
```powershell
az webapp browse --resource-group $RG --name $APP_NAME
az webapp log config --name $APP_NAME --resource-group $RG --application-logging filesystem --level information
az webapp log tail --name $APP_NAME --resource-group $RG
```

Alternative: GitHub Actions (CI/CD)
- Push code to GitHub.
- In Azure portal: your Web App → Deployment Center → GitHub → select repo/branch → Python 3.10 → Save.
- In Configuration, set Startup Command to:
```
gunicorn --bind=0.0.0.0 --timeout 600 app:app
```

Notes
- Keep `app.py` exposing `app = Flask(__name__)` (already done).
- Keep `requirements.txt` in repo root (already done).
- Don’t commit `.venv/`, `__pycache__/`, PDFs, or generated charts (use `.gitignore`).
- Default root `/` serves `dashboard.html`. APIs: `/api/analyze`, `/api/download-report`.

### Deploy to Azure via GitHub Actions

- Ensure your Azure Web App (Linux) exists and is set to Python 3.10 (or update the workflow `PYTHON_VERSION`).
- We added a `wsgi.py` so Azure (Oryx) can auto-start the app via Gunicorn as `wsgi:app`.
- Use the GitHub Actions workflow in `.github/workflows/azure-webapp.yml`:
  1. In Azure Portal, open your Web App → Get Publish Profile and download it.
  2. In your GitHub repo → Settings → Secrets and variables → Actions → New repository secret:
     - Name: `AZURE_WEBAPP_PUBLISH_PROFILE`
     - Value: full contents of the downloaded publish profile `.PublishSettings` file
  3. Edit the workflow `AZURE_WEBAPP_NAME` to match your app name.
  4. Push to `main`. The workflow will build and deploy automatically.

If you prefer Azure CLI instead, see the section above for `az webapp up` and setting the startup command.
``` 