from pathlib import Path
import json

import torch

from report_html import (
    ACTIVATION_PCA_PATH,
    GEOMETRY_COMPARISON_PATH,
    PROBABILITY_PCA_PATH,
    REPORT_PATH,
    append_or_replace_section,
)


project_root = Path(__file__).resolve().parents[1]


def build_centroids(
    features: torch.Tensor,
    examples: list[dict],
    days: list[str],
) -> tuple[torch.Tensor, list[int]]:
    rows = []
    counts = []
    for day in days:
        indices = [
            index
            for index, example in enumerate(examples)
            if example["answer"] == day
        ]
        rows.append(features[indices].mean(dim=0))
        counts.append(len(indices))

    return torch.stack(rows), counts


def upper_triangle_values(matrix: torch.Tensor) -> torch.Tensor:
    indices = torch.triu_indices(matrix.shape[0], matrix.shape[1], offset=1)
    return matrix[indices[0], indices[1]]


def correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    return torch.dot(x_centered, y_centered) / (
        torch.linalg.vector_norm(x_centered) * torch.linalg.vector_norm(y_centered)
    )


def print_neighbor_distances(
    title: str,
    days: list[str],
    distances: torch.Tensor,
) -> None:
    print(title)
    for index, day in enumerate(days):
        next_index = (index + 1) % len(days)
        next_day = days[next_index]
        print(f"{day:>9} -> {next_day:<9}: {distances[index, next_index].item():.3f}")
    print()


def print_distance_matrix(
    title: str,
    days: list[str],
    distances: torch.Tensor,
) -> None:
    print(title)
    print(" " * 10 + " ".join(f"{day[:3]:>8}" for day in days))
    for day, row in zip(days, distances):
        values = " ".join(f"{value.item():8.3f}" for value in row)
        print(f"{day[:3]:>9} {values}")
    print()


def append_geometry_html(
    days: list[str],
    activation_distances: torch.Tensor,
    probability_distances: torch.Tensor,
    distance_correlation: torch.Tensor,
) -> None:
    pair_points = []
    for i, left in enumerate(days):
        for j, right in enumerate(days):
            if j <= i:
                continue
            pair_points.append(
                {
                    "pair": f"{left} - {right}",
                    "activation_distance": activation_distances[i, j].item(),
                    "probability_distance": probability_distances[i, j].item(),
                }
            )

    section = f"""

  <section id="geometry-comparison">
    <h2>Geometry Comparison</h2>
    <p>
      Pairwise distance correlation:
      <code>{distance_correlation.item():.3f}</code>.
    </p>
    <div class="grid">
      <div id="activationDistances" class="chart"></div>
      <div id="probabilityDistances" class="chart"></div>
      <div id="distanceCorrelation" class="chart wide"></div>
    </div>
  </section>

  <script>
    const geometryDays = {json.dumps(days)};
    const activationDistances = {json.dumps(activation_distances.tolist())};
    const probabilityDistances = {json.dumps(probability_distances.tolist())};
    const pairPoints = {json.dumps(pair_points)};
    const pairwiseDistanceCorrelation = {distance_correlation.item()};

    function heatmapLayout(title) {{
      return {{
        title,
        paper_bgcolor: "#fffaf2",
        plot_bgcolor: "#f3eadf",
        font: {{ color: "#25211c" }},
        xaxis: {{ tickangle: -30 }},
        yaxis: {{ autorange: "reversed" }},
        margin: {{ t: 60, l: 85, r: 30, b: 85 }},
      }};
    }}

    function distanceHeatmap(z) {{
      return [{{
        type: "heatmap",
        z,
        x: geometryDays,
        y: geometryDays,
        colorscale: "Viridis",
        hovertemplate: "%{{y}} - %{{x}}<br>distance=%{{z:.3f}}<extra></extra>",
        colorbar: {{ title: "distance" }},
      }}];
    }}

    Plotly.newPlot(
      "activationDistances",
      distanceHeatmap(activationDistances),
      heatmapLayout("Activation Centroid Distances")
    );
    Plotly.newPlot(
      "probabilityDistances",
      distanceHeatmap(probabilityDistances),
      heatmapLayout("Probability Centroid Distances")
    );
    Plotly.newPlot(
      "distanceCorrelation",
      [{{
        type: "scatter",
        mode: "markers+text",
        name: "weekday pairs",
        x: pairPoints.map(p => p.activation_distance),
        y: pairPoints.map(p => p.probability_distance),
        text: pairPoints.map(p => p.pair),
        textposition: "top center",
        hovertemplate:
          "%{{text}}<br>activation=%{{x:.3f}}<br>probability=%{{y:.3f}}<extra></extra>",
        marker: {{
          size: 12,
          color: "#b35c44",
          line: {{ color: "#fffaf2", width: 1 }},
        }},
      }}],
      {{
        title: `Activation vs Probability Pairwise Distances (r=${{pairwiseDistanceCorrelation.toFixed(3)}})`,
        paper_bgcolor: "#fffaf2",
        plot_bgcolor: "#f3eadf",
        font: {{ color: "#25211c" }},
        xaxis: {{ title: "Activation centroid distance", gridcolor: "#ded1c1" }},
        yaxis: {{ title: "Probability centroid distance", gridcolor: "#ded1c1" }},
        margin: {{ t: 60, l: 75, r: 30, b: 70 }},
      }}
    );
  </script>
"""

    append_or_replace_section("experiment-5-geometry-comparison", section)


def main() -> None:
    activation_path = ACTIVATION_PCA_PATH
    probability_path = PROBABILITY_PCA_PATH
    output_path = GEOMETRY_COMPARISON_PATH

    activation_artifact = torch.load(activation_path, map_location="cpu")
    probability_artifact = torch.load(probability_path, map_location="cpu")

    days = activation_artifact["days"]
    examples = activation_artifact["examples"]

    activation_centroids, activation_counts = build_centroids(
        activation_artifact["features"],
        examples,
        days,
    )
    probability_centroids = probability_artifact["centroids"]
    probability_counts = activation_counts

    activation_distances = torch.cdist(activation_centroids, activation_centroids)
    probability_distances = torch.cdist(probability_centroids, probability_centroids)

    activation_distance_values = upper_triangle_values(activation_distances)
    probability_distance_values = upper_triangle_values(probability_distances)
    distance_correlation = correlation(
        activation_distance_values,
        probability_distance_values,
    )

    torch.save(
        {
            "days": days,
            "activation_source_path": str(activation_path),
            "probability_source_path": str(probability_path),
            "activation_centroids": activation_centroids,
            "probability_centroids": probability_centroids,
            "activation_counts": activation_counts,
            "probability_counts": probability_counts,
            "activation_pairwise_distances": activation_distances,
            "probability_pairwise_distances": probability_distances,
            "pairwise_distance_correlation": distance_correlation,
        },
        output_path,
    )

    print(f"Saved geometry comparison: {output_path}")
    print(f"Activation centroid shape: {tuple(activation_centroids.shape)}")
    print(f"Probability centroid shape: {tuple(probability_centroids.shape)}")
    print(f"Counts by day: {dict(zip(days, activation_counts))}")
    print(f"Pairwise distance correlation: {distance_correlation.item():.3f}")
    print()

    print_neighbor_distances("Activation neighbor distances:", days, activation_distances)
    print_neighbor_distances("Probability neighbor distances:", days, probability_distances)
    print_distance_matrix("Activation pairwise distances:", days, activation_distances)
    print_distance_matrix("Probability pairwise distances:", days, probability_distances)

    append_geometry_html(
        days=days,
        activation_distances=activation_distances,
        probability_distances=probability_distances,
        distance_correlation=distance_correlation,
    )
    print(f"Updated report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
