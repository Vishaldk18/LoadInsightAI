### Project/Solution/Topic Name
AI‑Powered Performance Test Result Analysis & Visualization

Team Name: _______________________

Team Members: ____________________

### DESCRIPTION
The application provides a fast, consistent way to analyze performance test results and visualize insights. It ingests output from popular tools—JMeter (CSV), k6 (JSON), and NeoLoad (XML)—normalizes the data, computes key metrics (avg/median/90th Percentile/95th Percentile latency, throughput, error rate), and renders interactive charts in a web dashboard. It also generates an executive PDF report and lightweight AI recommendations to guide next steps.

### PROBLEM STATEMENT
Project teams often spend significant time stitching together results from different load‑testing tools and manually creating summaries and visuals. This leads to inconsistent metrics, slow feedback, and limited comparability across runs, making it harder to spot hotspots and prioritize fixes quickly.

### SOLUTION PROPOSED
- Unified ingestion of JMeter CSV, k6 JSON, and NeoLoad XML results
- Automatic calculation of core KPIs: average/median/p90/p95 latency, throughput, success/error counts, error rate
- Endpoint‑level breakdowns: top 10 slowest, error distribution by endpoint
- Time‑series views: latency over time and throughput over time
- One‑click executive PDF report generation (with embedded charts)
- AI‑style recommendations highlighting likely bottlenecks and priorities
- Modern web dashboard with dark mode and shareable outputs

### TECHNOLOGY STACK
- Framework: Flask (Python)
- Language: Python 3.x (backend), HTML/CSS/JavaScript (frontend)
- Data/Analysis: pandas, numpy
- Visualization: Chart.js (frontend), matplotlib + seaborn (backend charting utilities)
- Reporting: reportlab (PDF)
- Supported Inputs: JMeter CSV, k6 JSON (summary or points), NeoLoad XML

### BENEFITS & SAVING
- **Accurate, consistent metrics**: Standardized KPIs across tools and runs
- **Time & cost savings**: Eliminates manual collation and slide‑building
- **Actionable insights**: Endpoint‑level hotspots and percentile views drive targeted tuning
- **Shareable reporting**: One‑click PDF export for stakeholders and audit trails
- **Repeatable & scalable**: Works across projects, data formats, and test sizes
- **Faster feedback loops**: Quick uploads and instant visuals shorten performance triage

### QUICK START
- Install: `pip install -r requirements.txt`
- Run: `python app.py` then open `http://localhost:5000`
- Upload a results file (`.csv`, `.json`, or `.xml`) to view metrics and charts
- Export a PDF via "Download Report"

### KEY ENDPOINTS
- `GET /` or `GET /dashboard.html`: Serves the dashboard UI
- `POST /api/analyze`: Accepts an uploaded file under field name `file`; returns metrics, per‑endpoint breakdowns, and chart data as JSON
- `POST /api/download-report`: Accepts JSON payload of metrics and chart images; returns a generated PDF 