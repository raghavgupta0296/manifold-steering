from pathlib import Path


project_root = Path(__file__).resolve().parents[1]
MODEL_SLUG = "llama31_8b"
LAYER = 12
RUN_SLUG = f"weekdays_{MODEL_SLUG}_layer{LAYER}"
ARTIFACT_DIR = project_root / "artifacts" / f"{MODEL_SLUG}_layer{LAYER}"
RESULTS_DIR = project_root / "results"
REPORT_PATH = RESULTS_DIR / f"{RUN_SLUG}_report.html"

SOURCE_ARTIFACT_PATH = ARTIFACT_DIR / f"{RUN_SLUG}.pt"
ACTIVATION_PCA_PATH = ARTIFACT_DIR / f"{RUN_SLUG}_pca32.pt"
PROBABILITY_PCA_PATH = ARTIFACT_DIR / f"{RUN_SLUG}_probability_pca3.pt"
GEOMETRY_COMPARISON_PATH = ARTIFACT_DIR / f"{RUN_SLUG}_geometry_comparison.pt"
SPLINE_FITS_PATH = ARTIFACT_DIR / f"{RUN_SLUG}_spline_fits.pt"
PCA_FEATURE_STEERING_PATH = ARTIFACT_DIR / f"{RUN_SLUG}_pca_feature_steering.pt"


def ensure_report() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if REPORT_PATH.exists():
        return

    REPORT_PATH.write_text(
        """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Weekdays Manifold Steering Report</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f1e8;
      --panel: #fffaf2;
      --panel-soft: #f3eadf;
      --border: #ded1c1;
      --text: #25211c;
      --muted: #6f6258;
      --accent: #b35c44;
    }

    * {
      box-sizing: border-box;
    }

    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      max-width: 1480px;
      margin: 0 auto;
      padding: 32px 24px 48px;
      background:
        radial-gradient(circle at top left, rgba(179, 92, 68, 0.16), transparent 34rem),
        linear-gradient(180deg, #fffaf2 0%, #f7f1e8 48%, #efe3d3 100%);
      color: var(--text);
      line-height: 1.55;
    }
    h1 {
      margin: 0 0 28px;
      font-size: clamp(2rem, 4vw, 3.25rem);
      line-height: 1.05;
    }
    h2 {
      margin: 0 0 12px;
      color: #2f2822;
    }
    section {
      margin-bottom: 36px;
      padding: 24px;
      background: rgba(255, 250, 242, 0.9);
      border: 1px solid var(--border);
      border-radius: 10px;
      box-shadow: 0 18px 42px rgba(82, 55, 34, 0.12);
    }
    p {
      color: var(--muted);
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 24px;
    }
    .chart {
      height: 620px;
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      background: var(--panel-soft);
    }
    .wide {
      grid-column: 1 / -1;
    }
    code {
      color: var(--accent);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      background: rgba(255, 250, 242, 0.78);
      border-radius: 8px;
      overflow: hidden;
    }
    td, th {
      border: 1px solid var(--border);
      padding: 8px 10px;
      vertical-align: top;
    }
    th {
      background: #eadccb;
      color: #2f2822;
      text-align: left;
    }
    tr:nth-child(even) td {
      background: rgba(179, 92, 68, 0.05);
    }
    @media (max-width: 1200px) {
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <h1>Weekdays Manifold Steering Report</h1>
</body>
</html>
""",
        encoding="utf-8",
    )


def append_or_replace_section(section_id: str, html: str) -> None:
    ensure_report()
    start = f"<!-- BEGIN {section_id} -->"
    end = f"<!-- END {section_id} -->"
    block = f"{start}\n{html}\n{end}"

    text = REPORT_PATH.read_text(encoding="utf-8")
    if start in text and end in text:
        before = text.split(start)[0]
        after = text.split(end, 1)[1]
        text = before + block + after
    else:
        text = text.replace("</body>", f"{block}\n</body>")

    REPORT_PATH.write_text(text, encoding="utf-8")
