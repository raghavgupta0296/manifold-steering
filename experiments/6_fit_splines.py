from pathlib import Path
import json
import math

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


def bernoulli4_kernel(distance: torch.Tensor, period: float) -> torch.Tensor:
    x = (distance % period) / period
    x = torch.minimum(x, 1.0 - x)
    return x**4 - 2.0 * x**3 + x**2 - 1.0 / 30.0


def fit_periodic_spline(
    control_angles: torch.Tensor,
    values: torch.Tensor,
    period: float,
    smoothness: float = 0.0,
) -> dict[str, torch.Tensor | float]:
    diff = (control_angles[:, None] - control_angles[None, :]).abs()
    kernel = bernoulli4_kernel(diff, period)
    if smoothness > 0:
        kernel = kernel + smoothness * torch.eye(kernel.shape[0])

    ones = torch.ones(kernel.shape[0], 1)
    system = torch.zeros(kernel.shape[0] + 1, kernel.shape[0] + 1)
    system[:-1, :-1] = kernel
    system[:-1, -1:] = ones
    system[-1:, :-1] = ones.T

    rhs = torch.zeros(kernel.shape[0] + 1, values.shape[1])
    rhs[:-1] = values

    solution = torch.linalg.solve(system, rhs)
    return {
        "control_angles": control_angles,
        "values": values,
        "weights": solution[:-1],
        "bias": solution[-1],
        "period": period,
        "smoothness": smoothness,
    }


def evaluate_periodic_spline(
    spline: dict[str, torch.Tensor | float],
    query_angles: torch.Tensor,
) -> torch.Tensor:
    control_angles = spline["control_angles"]
    weights = spline["weights"]
    bias = spline["bias"]
    period = spline["period"]

    diff = (query_angles[:, None] - control_angles[None, :]).abs()
    kernel = bernoulli4_kernel(diff, period)
    return kernel @ weights + bias


def build_centroids(
    features: torch.Tensor,
    examples: list[dict],
    days: list[str],
) -> torch.Tensor:
    rows = []
    for day in days:
        indices = [
            index
            for index, example in enumerate(examples)
            if example["answer"] == day
        ]
        rows.append(features[indices].mean(dim=0))
    return torch.stack(rows)


def make_path_points(path: torch.Tensor, angles: torch.Tensor, days: list[str]) -> list[dict]:
    points = []
    for coords, angle in zip(path, angles):
        weekday_position = (angle.item() / (2.0 * math.pi)) * len(days)
        points.append(
            {
                "pc1": coords[0].item(),
                "pc2": coords[1].item(),
                "pc3": coords[2].item(),
                "angle": angle.item(),
                "weekday_position": weekday_position,
            }
        )
    return points


def make_centroid_points(centroids: torch.Tensor, days: list[str]) -> list[dict]:
    return [
        {
            "day": day,
            "pc1": centroid[0].item(),
            "pc2": centroid[1].item(),
            "pc3": centroid[2].item(),
        }
        for day, centroid in zip(days, centroids)
    ]


def main() -> None:
    activation_path = project_root / "artifacts" / "weekdays_llama31_8b_layer28_pca32.pt"
    probability_path = project_root / "artifacts" / "weekdays_probability_pca3.pt"
    output_path = project_root / "artifacts" / "weekdays_spline_fits.pt"

    activation_artifact = torch.load(activation_path, map_location="cpu")
    probability_artifact = torch.load(probability_path, map_location="cpu")

    days = activation_artifact["days"]
    examples = activation_artifact["examples"]
    period = 2.0 * math.pi
    control_angles = torch.linspace(0.0, period, len(days) + 1)[:-1]
    query_angles = torch.linspace(0.0, period, 141)

    activation_centroids = build_centroids(
        activation_artifact["features"],
        examples,
        days,
    )
    probability_centroids = probability_artifact["centroids"]

    activation_spline = fit_periodic_spline(
        control_angles=control_angles,
        values=activation_centroids,
        period=period,
    )
    probability_spline = fit_periodic_spline(
        control_angles=control_angles,
        values=probability_centroids,
        period=period,
    )

    activation_path_points = evaluate_periodic_spline(activation_spline, query_angles)
    probability_path_points = evaluate_periodic_spline(probability_spline, query_angles)

    activation_reconstruction = evaluate_periodic_spline(
        activation_spline,
        control_angles,
    )
    probability_reconstruction = evaluate_periodic_spline(
        probability_spline,
        control_angles,
    )
    activation_reconstruction_error = torch.linalg.vector_norm(
        activation_reconstruction - activation_centroids,
        dim=1,
    )
    probability_reconstruction_error = torch.linalg.vector_norm(
        probability_reconstruction - probability_centroids,
        dim=1,
    )

    torch.save(
        {
            "days": days,
            "control_angles": control_angles,
            "query_angles": query_angles,
            "activation_centroids": activation_centroids,
            "probability_centroids": probability_centroids,
            "activation_path": activation_path_points,
            "probability_path": probability_path_points,
            "activation_reconstruction_error": activation_reconstruction_error,
            "probability_reconstruction_error": probability_reconstruction_error,
            "activation_spline": activation_spline,
            "probability_spline": probability_spline,
        },
        output_path,
    )

    activation_plot_path = make_path_points(activation_path_points, query_angles, days)
    probability_plot_path = make_path_points(probability_path_points, query_angles, days)
    activation_plot_centroids = make_centroid_points(activation_centroids, days)
    probability_plot_centroids = make_centroid_points(probability_centroids, days)

    section = f"""
  <section id="experiment-6-spline-fits">
    <h2>6. Periodic Spline Fits</h2>
    <p>
      Fitted a periodic spline from weekday angle to activation PCA centroids and probability PCA centroids.
      Saved artifact: <code>{output_path}</code><br>
      Activation reconstruction error max: <code>{activation_reconstruction_error.max().item():.6f}</code><br>
      Probability reconstruction error max: <code>{probability_reconstruction_error.max().item():.6f}</code>
    </p>
    <div class="grid">
      <div id="activationSpline2d" class="chart"></div>
      <div id="activationSpline3d" class="chart"></div>
      <div id="probabilitySpline2d" class="chart"></div>
      <div id="probabilitySpline3d" class="chart"></div>
    </div>
  </section>

  <script>
    const splineDays = {json.dumps(days)};
    const splineColors = {json.dumps(COLORS)};
    const activationSplinePath = {json.dumps(activation_plot_path)};
    const probabilitySplinePath = {json.dumps(probability_plot_path)};
    const activationSplineCentroids = {json.dumps(activation_plot_centroids)};
    const probabilitySplineCentroids = {json.dumps(probability_plot_centroids)};

    function splinePathTrace2d(points, name) {{
      return {{
        type: "scatter",
        mode: "lines",
        name,
        x: points.map(p => p.pc1),
        y: points.map(p => p.pc2),
        line: {{ color: "#25211c", width: 4 }},
        hovertemplate:
          "angle=%{{customdata[0]:.3f}}<br>weekday position=%{{customdata[1]:.3f}}<extra></extra>",
        customdata: points.map(p => [p.angle, p.weekday_position]),
      }};
    }}

    function splinePathTrace3d(points, name) {{
      return {{
        type: "scatter3d",
        mode: "lines",
        name,
        x: points.map(p => p.pc1),
        y: points.map(p => p.pc2),
        z: points.map(p => p.pc3),
        line: {{ color: "#25211c", width: 8 }},
        hovertemplate:
          "angle=%{{customdata[0]:.3f}}<br>weekday position=%{{customdata[1]:.3f}}<extra></extra>",
        customdata: points.map(p => [p.angle, p.weekday_position]),
      }};
    }}

    function splineCentroidTrace2d(centroids) {{
      return {{
        type: "scatter",
        mode: "markers+text",
        name: "centroids",
        x: centroids.map(c => c.pc1),
        y: centroids.map(c => c.pc2),
        text: centroids.map(c => c.day),
        textposition: "top center",
        marker: {{
          size: 15,
          color: centroids.map(c => splineColors[c.day]),
          line: {{ color: "#fffaf2", width: 1 }},
        }},
        hoverinfo: "text",
      }};
    }}

    function splineCentroidTrace3d(centroids) {{
      return {{
        type: "scatter3d",
        mode: "markers+text",
        name: "centroids",
        x: centroids.map(c => c.pc1),
        y: centroids.map(c => c.pc2),
        z: centroids.map(c => c.pc3),
        text: centroids.map(c => c.day),
        textposition: "top center",
        marker: {{ size: 7, color: centroids.map(c => splineColors[c.day]) }},
        hoverinfo: "text",
      }};
    }}

    function splineLayout2d(title, prefix) {{
      return {{
        title,
        paper_bgcolor: "#fffaf2",
        plot_bgcolor: "#f3eadf",
        font: {{ color: "#25211c" }},
        xaxis: {{ title: `${{prefix}} PC1`, gridcolor: "#ded1c1" }},
        yaxis: {{ title: `${{prefix}} PC2`, gridcolor: "#ded1c1", scaleanchor: "x", scaleratio: 1 }},
        margin: {{ t: 50, l: 55, r: 20, b: 55 }},
      }};
    }}

    function splineLayout3d(title, prefix) {{
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
      "activationSpline2d",
      [splinePathTrace2d(activationSplinePath, "activation spline"), splineCentroidTrace2d(activationSplineCentroids)],
      splineLayout2d("Activation Periodic Spline: 2D", "Activation")
    );
    Plotly.newPlot(
      "activationSpline3d",
      [splinePathTrace3d(activationSplinePath, "activation spline"), splineCentroidTrace3d(activationSplineCentroids)],
      splineLayout3d("Activation Periodic Spline: 3D", "Activation")
    );
    Plotly.newPlot(
      "probabilitySpline2d",
      [splinePathTrace2d(probabilitySplinePath, "probability spline"), splineCentroidTrace2d(probabilitySplineCentroids)],
      splineLayout2d("Probability Periodic Spline: 2D", "Hellinger")
    );
    Plotly.newPlot(
      "probabilitySpline3d",
      [splinePathTrace3d(probabilitySplinePath, "probability spline"), splineCentroidTrace3d(probabilitySplineCentroids)],
      splineLayout3d("Probability Periodic Spline: 3D", "Hellinger")
    );
  </script>
"""
    append_or_replace_section("experiment-6-spline-fits", section)

    print(f"Saved spline artifact: {output_path}")
    print(
        "Activation reconstruction error:",
        [round(value.item(), 6) for value in activation_reconstruction_error],
    )
    print(
        "Probability reconstruction error:",
        [round(value.item(), 6) for value in probability_reconstruction_error],
    )
    print(f"Updated report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
