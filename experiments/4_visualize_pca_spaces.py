from pathlib import Path
import json

import torch

from report_html import REPORT_PATH, append_or_replace_section


project_root = Path(__file__).resolve().parents[1]

COLORS = {
    "Monday": "#ef4444",
    "Tuesday": "#f97316",
    "Wednesday": "#eab308",
    "Thursday": "#22c55e",
    "Friday": "#06b6d4",
    "Saturday": "#3b82f6",
    "Sunday": "#a855f7",
}


def build_centroids(features: torch.Tensor, examples: list[dict], days: list[str]) -> list[dict]:
    centroids = []
    for day in days:
        indices = [
            index
            for index, example in enumerate(examples)
            if example["answer"] == day
        ]
        centroid = features[indices].mean(dim=0)
        centroids.append(
            {
                "day": day,
                "pc1": centroid[0].item(),
                "pc2": centroid[1].item(),
                "pc3": centroid[2].item(),
            }
        )
    return centroids


def build_activation_data() -> tuple[dict, list[dict], list[dict]]:
    artifact_path = project_root / "artifacts" / "weekdays_llama31_8b_layer28_pca32.pt"
    artifact = torch.load(artifact_path, map_location="cpu")

    features = artifact["features"]
    examples = artifact["examples"]
    days = artifact["days"]
    centroids = build_centroids(features, examples, days)

    points = []
    for example, coords in zip(examples, features):
        points.append(
            {
                "answer": example["answer"],
                "entity": example["entity"],
                "number_word": example["number_word"],
                "number_int": example["number_int"],
                "raw_input": example["raw_input"],
                "predicted_answer": example["predicted_answer"],
                "correct": example["correct"],
                "pc1": coords[0].item(),
                "pc2": coords[1].item(),
                "pc3": coords[2].item(),
            }
        )

    metadata = {
        "model_name": artifact["model_name"],
        "layer": artifact["layer"],
        "shape": tuple(features.shape),
        "days": days,
    }
    return metadata, points, centroids


def build_probability_data() -> tuple[dict, list[dict], list[dict]]:
    artifact_path = project_root / "artifacts" / "weekdays_probability_pca3.pt"
    artifact = torch.load(artifact_path, map_location="cpu")

    features = artifact["features"]
    examples = artifact["examples"]
    days = artifact["days"]
    probabilities = artifact["probabilities"]
    centroids = [
        {
            "day": day,
            "pc1": coords[0].item(),
            "pc2": coords[1].item(),
            "pc3": coords[2].item(),
        }
        for day, coords in zip(days, artifact["centroids"])
    ]

    points = []
    for example, coords, probs in zip(examples, features, probabilities):
        points.append(
            {
                "answer": example["answer"],
                "entity": example["entity"],
                "number_word": example["number_word"],
                "number_int": example["number_int"],
                "raw_input": example["raw_input"],
                "predicted_answer": example["predicted_answer"],
                "correct": example["correct"],
                "pc1": coords[0].item(),
                "pc2": coords[1].item(),
                "pc3": coords[2].item(),
                "correct_probability": probs[days.index(example["answer"])].item(),
                "other_probability": probs[-1].item(),
            }
        )

    metadata = {
        "model_name": artifact["model_name"],
        "shape": tuple(features.shape),
        "days": days,
        "explained_variance": [
            round(value.item(), 4)
            for value in artifact["explained_variance_ratio"]
        ],
    }
    return metadata, points, centroids


def main() -> None:
    activation_metadata, activation_points, activation_centroids = build_activation_data()
    probability_metadata, probability_points, probability_centroids = build_probability_data()
    days = activation_metadata["days"]

    html = f"""
  <section id="experiment-4-pca-visualization">
    <h2>4. PCA Visualizations</h2>
  <p>
    Activation space: <code>{activation_metadata["model_name"]}</code>,
    layer <code>{activation_metadata["layer"]}</code>,
    features <code>{activation_metadata["shape"]}</code>.
    Probability space: PCA over <code>sqrt(probabilities)</code>,
    features <code>{probability_metadata["shape"]}</code>,
    explained variance <code>{probability_metadata["explained_variance"]}</code>.
  </p>

  <div class="grid">
    <div id="activation2d" class="chart"></div>
    <div id="activation3d" class="chart"></div>
    <div id="probability2d" class="chart"></div>
    <div id="probability3d" class="chart"></div>
  </div>

  <script>
    const days = {json.dumps(days)};
    const colors = {json.dumps(COLORS)};
    const activationPoints = {json.dumps(activation_points)};
    const activationCentroids = {json.dumps(activation_centroids)};
    const probabilityPoints = {json.dumps(probability_points)};
    const probabilityCentroids = {json.dumps(probability_centroids)};

    function hoverText(p) {{
      const rows = [
        `input: ${{p.raw_input.replace("\\n", " ")}}`,
        `answer: ${{p.answer}}`,
        `predicted: ${{p.predicted_answer}}`,
        `correct: ${{p.correct}}`,
      ];
      if ("correct_probability" in p) {{
        rows.push(`P(correct): ${{p.correct_probability.toFixed(4)}}`);
        rows.push(`P(other): ${{p.other_probability.toFixed(4)}}`);
      }} else {{
        rows.push(`entity: ${{p.entity}}`);
        rows.push(`number: ${{p.number_word}} (${{p.number_int}})`);
      }}
      return rows.join("<br>");
    }}

    function pointTraces2d(points) {{
      return days.map(day => {{
        const group = points.filter(p => p.answer === day);
        return {{
          type: "scatter",
          mode: "markers",
          name: day,
          x: group.map(p => p.pc1),
          y: group.map(p => p.pc2),
          text: group.map(hoverText),
          hoverinfo: "text",
          marker: {{
            size: 10,
            color: colors[day],
            line: {{ color: "#fffaf2", width: 1 }},
          }},
          showlegend: true,
        }};
      }});
    }}

    function pointTraces3d(points) {{
      return days.map(day => {{
        const group = points.filter(p => p.answer === day);
        return {{
          type: "scatter3d",
          mode: "markers",
          name: day,
          x: group.map(p => p.pc1),
          y: group.map(p => p.pc2),
          z: group.map(p => p.pc3),
          text: group.map(hoverText),
          hoverinfo: "text",
          marker: {{ size: 4, color: colors[day] }},
          showlegend: false,
        }};
      }});
    }}

    function centroidTrace2d(centroids) {{
      const loop = centroids.concat([centroids[0]]);
      return {{
        type: "scatter",
        mode: "lines+markers+text",
        name: "centroids",
        x: loop.map(c => c.pc1),
        y: loop.map(c => c.pc2),
        text: loop.map(c => c.day),
        textposition: "top center",
        line: {{ color: "#25211c", width: 3 }},
        marker: {{ size: 14, color: loop.map(c => colors[c.day]) }},
        hoverinfo: "text",
        showlegend: true,
      }};
    }}

    function centroidTrace3d(centroids) {{
      const loop = centroids.concat([centroids[0]]);
      return {{
        type: "scatter3d",
        mode: "lines+markers+text",
        name: "centroids",
        x: loop.map(c => c.pc1),
        y: loop.map(c => c.pc2),
        z: loop.map(c => c.pc3),
        text: loop.map(c => c.day),
        textposition: "top center",
        line: {{ color: "#25211c", width: 8 }},
        marker: {{ size: 7, color: loop.map(c => colors[c.day]) }},
        hoverinfo: "text",
        showlegend: false,
      }};
    }}

    function layout2d(title, prefix) {{
      return {{
        title,
        paper_bgcolor: "#fffaf2",
        plot_bgcolor: "#f3eadf",
        font: {{ color: "#25211c" }},
        xaxis: {{ title: `${{prefix}} PC1`, gridcolor: "#ded1c1" }},
        yaxis: {{ title: `${{prefix}} PC2`, gridcolor: "#ded1c1", scaleanchor: "x", scaleratio: 1 }},
        legend: {{ orientation: "h" }},
        margin: {{ t: 50, l: 55, r: 20, b: 55 }},
      }};
    }}

    function layout3d(title, prefix) {{
      return {{
        title,
        paper_bgcolor: "#fffaf2",
        plot_bgcolor: "#f3eadf",
        font: {{ color: "#25211c" }},
        scene: {{
          xaxis: {{ title: `${{prefix}} PC1`, gridcolor: "#ded1c1" }},
          yaxis: {{ title: `${{prefix}} PC2`, gridcolor: "#ded1c1" }},
          zaxis: {{ title: `${{prefix}} PC3`, gridcolor: "#ded1c1" }},
        }},
        margin: {{ t: 50, l: 0, r: 0, b: 0 }},
      }};
    }}

    Plotly.newPlot(
      "activation2d",
      [...pointTraces2d(activationPoints), centroidTrace2d(activationCentroids)],
      layout2d("Activation PCA: 2D", "Activation")
    );
    Plotly.newPlot(
      "activation3d",
      [...pointTraces3d(activationPoints), centroidTrace3d(activationCentroids)],
      layout3d("Activation PCA: 3D", "Activation")
    );
    Plotly.newPlot(
      "probability2d",
      [...pointTraces2d(probabilityPoints), centroidTrace2d(probabilityCentroids)],
      layout2d("Probability Hellinger PCA: 2D", "Hellinger")
    );
    Plotly.newPlot(
      "probability3d",
      [...pointTraces3d(probabilityPoints), centroidTrace3d(probabilityCentroids)],
      layout3d("Probability Hellinger PCA: 3D", "Hellinger")
    );
  </script>
  </section>
"""

    append_or_replace_section("experiment-4-pca-visualization", html)
    print(f"Updated report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
