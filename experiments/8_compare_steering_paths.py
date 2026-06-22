from argparse import ArgumentParser
from pathlib import Path
import json
import math
import sys

import torch

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from data.weekdays import WeekdayData
from model.llama import LlamaModel
from report_html import REPORT_PATH, append_or_replace_section


COLORS = {
    "Monday": "#ef4444",
    "Tuesday": "#f97316",
    "Wednesday": "#eab308",
    "Thursday": "#22c55e",
    "Friday": "#06b6d4",
    "Saturday": "#3b82f6",
    "Sunday": "#a855f7",
    "other": "#9ca3af",
}


METHODS = {
    "baseline": "Baseline",
    "raw_linear": "Raw residual linear",
    "pca_linear": "PCA linear",
    "spline_manifold": "Spline manifold",
    "pca_additive": "PCA additive sweep",
}


def bernoulli4_kernel(distance: torch.Tensor, period: float) -> torch.Tensor:
    x = (distance % period) / period
    x = torch.minimum(x, 1.0 - x)
    return x**4 - 2.0 * x**3 + x**2 - 1.0 / 30.0


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


def build_forward_arc_indices(days: list[str], source_day: str, target_day: str) -> list[int]:
    source = days.index(source_day)
    target = days.index(target_day)
    indices = []
    index = source
    while True:
        indices.append(index)
        if index == target:
            return indices
        index = (index + 1) % len(days)


def build_linear_path(start: torch.Tensor, end: torch.Tensor, n_steps: int) -> torch.Tensor:
    alphas = torch.linspace(0.0, 1.0, n_steps)
    return start.unsqueeze(0) + alphas.unsqueeze(1) * (end - start).unsqueeze(0)


def build_spline_path(
    spline: dict[str, torch.Tensor | float],
    days: list[str],
    source_day: str,
    target_day: str,
    n_steps: int,
) -> torch.Tensor:
    period = 2.0 * math.pi
    source_angle = period * days.index(source_day) / len(days)
    target_angle = period * days.index(target_day) / len(days)
    if target_angle < source_angle:
        target_angle += period
    query_angles = torch.linspace(source_angle, target_angle, n_steps) % period
    return evaluate_periodic_spline(spline, query_angles)


def expand_path_point(point: torch.Tensor, n_prompts: int) -> torch.Tensor:
    return point.unsqueeze(0).expand(n_prompts, -1)


def collect_replaced_path_probabilities(
    model: LlamaModel,
    prompts: list[str],
    days: list[str],
    layer: int,
    token_position: int,
    output_prefix: str,
    path: torch.Tensor,
    mode: str,
    pca_mean: torch.Tensor | None = None,
    pca_rotation: torch.Tensor | None = None,
    batch_size: int = 8,
) -> torch.Tensor:
    rows = []
    for step, point in enumerate(path):
        print(f"  step {step + 1:>2}/{len(path)}", flush=True)
        replacements = expand_path_point(point, len(prompts))
        if mode == "raw":
            probabilities = model.gather_replaced_prediction_probabilities(
                prompts=prompts,
                answer_values=days,
                layer=layer,
                replacement_activations=replacements,
                token_position=token_position,
                output_prefix=output_prefix,
                full_vocab_softmax=True,
                include_other=True,
                batch_size=batch_size,
            )
        elif mode == "pca":
            if pca_mean is None or pca_rotation is None:
                raise ValueError("PCA replacement requires pca_mean and pca_rotation.")
            probabilities = model.gather_pca_feature_replaced_prediction_probabilities(
                prompts=prompts,
                answer_values=days,
                layer=layer,
                pca_mean=pca_mean,
                pca_rotation=pca_rotation,
                replacement_features=replacements,
                token_position=token_position,
                output_prefix=output_prefix,
                full_vocab_softmax=True,
                include_other=True,
                batch_size=batch_size,
            )
        else:
            raise ValueError(f"Unknown replacement mode: {mode}")
        rows.append(probabilities)
    return torch.stack(rows)


def collect_additive_probabilities(
    model: LlamaModel,
    prompts: list[str],
    examples: list[dict],
    days: list[str],
    pca_centroids: torch.Tensor,
    layer: int,
    token_position: int,
    output_prefix: str,
    pca_mean: torch.Tensor,
    pca_rotation: torch.Tensor,
    target_day: str,
    coefficients: torch.Tensor,
    batch_size: int = 8,
) -> torch.Tensor:
    target_centroid = pca_centroids[days.index(target_day)]
    source_centroids = torch.stack(
        [pca_centroids[days.index(example["answer"])] for example in examples]
    )
    steering_vectors = target_centroid.unsqueeze(0) - source_centroids

    rows = []
    for step, coefficient in enumerate(coefficients):
        print(
            f"  coefficient {coefficient.item():.3f} "
            f"({step + 1:>2}/{len(coefficients)})",
            flush=True,
        )
        probabilities = model.gather_pca_feature_steered_prediction_probabilities(
            prompts=prompts,
            answer_values=days,
            layer=layer,
            pca_mean=pca_mean,
            pca_rotation=pca_rotation,
            feature_steering_vectors=steering_vectors,
            coefficient=coefficient.item(),
            token_position=token_position,
            output_prefix=output_prefix,
            full_vocab_softmax=True,
            include_other=True,
            batch_size=batch_size,
        )
        rows.append(probabilities)
    return torch.stack(rows)


def compress_sequence(values: list[str]) -> list[str]:
    compressed = []
    for value in values:
        if not compressed or compressed[-1] != value:
            compressed.append(value)
    return compressed


def summarize_method(
    method: str,
    probabilities: torch.Tensor,
    days: list[str],
    target_day: str,
    arc_indices: list[int],
) -> dict:
    weekday_probabilities = probabilities[:, :, : len(days)]
    mean_probabilities = probabilities.mean(dim=1)
    final_weekday = weekday_probabilities[-1]
    target_index = days.index(target_day)

    final_predictions = final_weekday.argmax(dim=-1)
    final_argmax_rate = (final_predictions == target_index).float().mean().item()
    final_target_probability = final_weekday[:, target_index].mean().item()

    sqrt_probabilities = torch.sqrt(mean_probabilities.clamp(min=0.0))
    jumps = torch.linalg.vector_norm(
        sqrt_probabilities[1:] - sqrt_probabilities[:-1],
        dim=-1,
    )

    arc_mass = mean_probabilities[:, arc_indices].sum(dim=-1)
    off_arc_or_other_mass = 1.0 - arc_mass
    dominant_indices = mean_probabilities[:, : len(days)].argmax(dim=-1).tolist()
    dominant_sequence = compress_sequence([days[index] for index in dominant_indices])

    return {
        "method": method,
        "label": METHODS[method],
        "final_target_probability": final_target_probability,
        "final_target_argmax_rate": final_argmax_rate,
        "mean_hellinger_jump": jumps.mean().item() if len(jumps) else 0.0,
        "max_hellinger_jump": jumps.max().item() if len(jumps) else 0.0,
        "mean_arc_mass": arc_mass.mean().item(),
        "mean_off_arc_or_other_mass": off_arc_or_other_mass.mean().item(),
        "dominant_sequence": dominant_sequence,
    }


def make_plot_rows(
    path_probabilities: dict[str, torch.Tensor],
    labels: list[str],
    x_values: dict[str, list[float]],
) -> list[dict]:
    rows = []
    for method, probabilities in path_probabilities.items():
        mean_probabilities = probabilities.mean(dim=1)
        for step, probs in enumerate(mean_probabilities):
            for label, probability in zip(labels, probs):
                rows.append(
                    {
                        "method": method,
                        "method_label": METHODS[method],
                        "step": step,
                        "x": x_values[method][step],
                        "label": label,
                        "probability": probability.item(),
                    }
                )
    return rows


def path_points_for_plot(
    path: torch.Tensor,
    pca_mean: torch.Tensor,
    pca_rotation: torch.Tensor,
    raw: bool = False,
) -> list[dict]:
    coords = (path - pca_mean) @ pca_rotation if raw else path
    return [
        {
            "step": index,
            "pc1": point[0].item(),
            "pc2": point[1].item(),
            "pc3": point[2].item(),
        }
        for index, point in enumerate(coords)
    ]


def append_html(
    output_path: Path,
    summaries: list[dict],
    plot_rows: list[dict],
    pca_paths: dict[str, list[dict]],
    section_suffix: str,
) -> None:
    dom_suffix = section_suffix
    js_suffix = "".join(ch for ch in section_suffix if ch.isalnum())
    table_rows = "\n".join(
        f"""
        <tr>
          <td>{row["label"]}</td>
          <td>{row["final_target_probability"]:.4f}</td>
          <td>{row["final_target_argmax_rate"]:.2%}</td>
          <td>{row["mean_hellinger_jump"]:.4f}</td>
          <td>{row["max_hellinger_jump"]:.4f}</td>
          <td>{row["mean_off_arc_or_other_mass"]:.4f}</td>
          <td><code>{' -> '.join(row["dominant_sequence"])}</code></td>
        </tr>
        """
        for row in summaries
    )
    section_id = f"experiment-8-steering-path-comparison{section_suffix}"
    section = f"""
  <section id="{section_id}">
    <h2>8. Steering Path Comparison</h2>
    <p>
      Monday to Thursday, 50 path-style intervention points by default, averaged over fixed prompts.
      Paper-like variants are raw/PCA linear and spline manifold paths; PCA additive is our extra diagnostic.
      Saved artifact: <code>{output_path}</code>
    </p>
    <table>
      <thead>
        <tr>
          <th>Method</th>
          <th>Final P(Thursday)</th>
          <th>Final Thursday Argmax</th>
          <th>Mean Hellinger Jump</th>
          <th>Max Hellinger Jump</th>
          <th>Mean Off-Arc/Other Mass</th>
          <th>Dominant Sequence</th>
        </tr>
      </thead>
      <tbody>{table_rows}</tbody>
    </table>
    <div class="grid">
      <div id="pathProbabilities{dom_suffix}" class="chart wide"></div>
      <div id="pathGeometry{dom_suffix}" class="chart wide"></div>
    </div>
  </section>

  <script>
    const steeringPathRows{js_suffix} = {json.dumps(plot_rows)};
    const steeringPathMethods{js_suffix} = {json.dumps(METHODS)};
    const steeringPathColors{js_suffix} = {json.dumps(COLORS)};
    const steeringPcaPaths{js_suffix} = {json.dumps(pca_paths)};

    function pathProbabilityTrace{js_suffix}(method, label) {{
      const rows = steeringPathRows{js_suffix}.filter(
        r => r.method === method && r.label === label
      );
      return {{
        type: "scatter",
        mode: "lines",
        name: `${{steeringPathMethods{js_suffix}[method]}} / ${{label}}`,
        x: rows.map(r => r.x),
        y: rows.map(r => r.probability),
        line: {{
          color: steeringPathColors{js_suffix}[label],
          width: label === "other" ? 2 : 3,
          dash: method === "spline_manifold" ? "solid" :
                method === "pca_linear" ? "dash" :
                method === "raw_linear" ? "dot" :
                method === "pca_additive" ? "dashdot" : "longdash",
        }},
        visible:
          method === "spline_manifold" ||
          (method === "raw_linear" && (label === "Monday" || label === "Thursday"))
            ? true
            : "legendonly",
        hovertemplate:
          `${{steeringPathMethods{js_suffix}[method]}}<br>${{label}}<br>x=%{{x:.3f}}<br>p=%{{y:.4f}}<extra></extra>`,
      }};
    }}

    const pathProbabilityLabels{js_suffix} = Object.keys(steeringPathColors{js_suffix});
    const pathProbabilityMethods{js_suffix} = Object.keys(steeringPathMethods{js_suffix});
    const pathProbabilityTraces{js_suffix} = [];
    for (const method of pathProbabilityMethods{js_suffix}) {{
      for (const label of pathProbabilityLabels{js_suffix}) {{
        pathProbabilityTraces{js_suffix}.push(pathProbabilityTrace{js_suffix}(method, label));
      }}
    }}

    Plotly.newPlot(
      "pathProbabilities{dom_suffix}",
      pathProbabilityTraces{js_suffix},
      {{
        title: "Output Probabilities Along Steering Paths",
        paper_bgcolor: "#fffaf2",
        plot_bgcolor: "#f3eadf",
        font: {{ color: "#25211c" }},
        xaxis: {{ title: "path position / additive coefficient", gridcolor: "#ded1c1" }},
        yaxis: {{ title: "mean probability", gridcolor: "#ded1c1", range: [0, 1] }},
        legend: {{ orientation: "h" }},
        margin: {{ t: 55, l: 70, r: 30, b: 70 }},
      }}
    );

    const geometryTraces{js_suffix} = Object.entries(steeringPcaPaths{js_suffix}).map(([method, points]) => ({{
      type: "scatter3d",
      mode: "lines+markers",
      name: steeringPathMethods{js_suffix}[method],
      x: points.map(p => p.pc1),
      y: points.map(p => p.pc2),
      z: points.map(p => p.pc3),
      marker: {{ size: 3 }},
      line: {{ width: 5 }},
      hovertemplate:
        `${{steeringPathMethods{js_suffix}[method]}}<br>step=%{{customdata}}<extra></extra>`,
      customdata: points.map(p => p.step),
    }}));
    Plotly.newPlot(
      "pathGeometry{dom_suffix}",
      geometryTraces{js_suffix},
      {{
        title: "Path Geometry in Activation PCA Space",
        paper_bgcolor: "#fffaf2",
        plot_bgcolor: "#f3eadf",
        font: {{ color: "#25211c" }},
        scene: {{
          xaxis: {{ title: "PC1", gridcolor: "#ded1c1" }},
          yaxis: {{ title: "PC2", gridcolor: "#ded1c1" }},
          zaxis: {{ title: "PC3", gridcolor: "#ded1c1" }},
        }},
        margin: {{ t: 55, l: 0, r: 0, b: 0 }},
      }}
    );
  </script>
"""
    append_or_replace_section(section_id, section)


def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--n-prompts", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_day = "Monday"
    target_day = "Thursday"

    source_path = project_root / "artifacts" / "weekdays_llama31_8b_layer28.pt"
    pca_path = project_root / "artifacts" / "weekdays_llama31_8b_layer28_pca32.pt"
    spline_path = project_root / "artifacts" / "weekdays_spline_fits.pt"
    output_path = (
        project_root
        / "artifacts"
        / f"weekdays_steering_path_comparison{args.output_suffix}.pt"
    )

    artifact = torch.load(source_path, map_location="cpu")
    pca_artifact = torch.load(pca_path, map_location="cpu")
    spline_artifact = torch.load(spline_path, map_location="cpu")
    data = WeekdayData()
    model = LlamaModel(artifact["model_name"]).load()

    days = artifact["days"]
    labels = days + ["other"]
    layer = artifact["layer"]
    token_position = artifact["token_position"]
    prompt_indices = list(range(args.n_prompts))
    examples = [artifact["examples"][index] for index in prompt_indices]
    prompts = [example["raw_input"] for example in examples]

    raw_centroids = build_centroids(
        artifact["activations"].float(),
        artifact["examples"],
        days,
    )
    pca_centroids = build_centroids(
        pca_artifact["features"].float(),
        pca_artifact["examples"],
        days,
    )
    raw_path = build_linear_path(
        raw_centroids[days.index(source_day)],
        raw_centroids[days.index(target_day)],
        args.n_steps,
    )
    pca_linear_path = build_linear_path(
        pca_centroids[days.index(source_day)],
        pca_centroids[days.index(target_day)],
        args.n_steps,
    )
    spline_manifold_path = build_spline_path(
        spline_artifact["activation_spline"],
        days,
        source_day,
        target_day,
        args.n_steps,
    )
    additive_coefficients = torch.linspace(0.0, 1.5, args.n_steps)

    print(f"Model: {model.model_name}")
    print(f"Layer: {layer}, token position: {token_position}")
    print(f"Pair: {source_day} -> {target_day}")
    print(f"Prompts: {prompt_indices}")
    print(f"Steps: {args.n_steps}")
    print()

    baseline_once = model.gather_prediction_probabilities(
        prompts=prompts,
        answer_values=days,
        output_prefix=data.output_prefix,
        full_vocab_softmax=True,
        include_other=True,
        batch_size=args.batch_size,
    )
    path_probabilities = {
        "baseline": baseline_once.unsqueeze(0).expand(args.n_steps, -1, -1).clone()
    }

    print("Collecting raw residual linear path", flush=True)
    path_probabilities["raw_linear"] = collect_replaced_path_probabilities(
        model=model,
        prompts=prompts,
        days=days,
        layer=layer,
        token_position=token_position,
        output_prefix=data.output_prefix,
        path=raw_path,
        mode="raw",
        batch_size=args.batch_size,
    )

    print("Collecting PCA linear path", flush=True)
    path_probabilities["pca_linear"] = collect_replaced_path_probabilities(
        model=model,
        prompts=prompts,
        days=days,
        layer=layer,
        token_position=token_position,
        output_prefix=data.output_prefix,
        path=pca_linear_path,
        mode="pca",
        pca_mean=pca_artifact["mean"].float(),
        pca_rotation=pca_artifact["rotation"].float(),
        batch_size=args.batch_size,
    )

    print("Collecting spline manifold path", flush=True)
    path_probabilities["spline_manifold"] = collect_replaced_path_probabilities(
        model=model,
        prompts=prompts,
        days=days,
        layer=layer,
        token_position=token_position,
        output_prefix=data.output_prefix,
        path=spline_manifold_path,
        mode="pca",
        pca_mean=pca_artifact["mean"].float(),
        pca_rotation=pca_artifact["rotation"].float(),
        batch_size=args.batch_size,
    )

    print("Collecting PCA additive endpoint sweep", flush=True)
    path_probabilities["pca_additive"] = collect_additive_probabilities(
        model=model,
        prompts=prompts,
        examples=examples,
        days=days,
        pca_centroids=pca_centroids,
        layer=layer,
        token_position=token_position,
        output_prefix=data.output_prefix,
        pca_mean=pca_artifact["mean"].float(),
        pca_rotation=pca_artifact["rotation"].float(),
        target_day=target_day,
        coefficients=additive_coefficients,
        batch_size=args.batch_size,
    )

    arc_indices = build_forward_arc_indices(days, source_day, target_day)
    summaries = [
        summarize_method(method, probabilities, days, target_day, arc_indices)
        for method, probabilities in path_probabilities.items()
    ]
    x_values = {
        method: torch.linspace(0.0, 1.0, args.n_steps).tolist()
        for method in path_probabilities
    }
    x_values["pca_additive"] = additive_coefficients.tolist()
    plot_rows = make_plot_rows(path_probabilities, labels, x_values)
    pca_paths = {
        "raw_linear": path_points_for_plot(
            raw_path,
            pca_artifact["mean"].float(),
            pca_artifact["rotation"].float(),
            raw=True,
        ),
        "pca_linear": path_points_for_plot(
            pca_linear_path,
            pca_artifact["mean"].float(),
            pca_artifact["rotation"].float(),
        ),
        "spline_manifold": path_points_for_plot(
            spline_manifold_path,
            pca_artifact["mean"].float(),
            pca_artifact["rotation"].float(),
        ),
    }

    torch.save(
        {
            "source_path": str(source_path),
            "pca_path": str(pca_path),
            "spline_path": str(spline_path),
            "model_name": model.model_name,
            "layer": layer,
            "token_position": token_position,
            "source_day": source_day,
            "target_day": target_day,
            "n_steps": args.n_steps,
            "prompt_indices": prompt_indices,
            "days": days,
            "labels": labels,
            "method_labels": METHODS,
            "path_probabilities": path_probabilities,
            "mean_path_probabilities": {
                method: probabilities.mean(dim=1)
                for method, probabilities in path_probabilities.items()
            },
            "summaries": summaries,
            "raw_path": raw_path,
            "pca_linear_path": pca_linear_path,
            "spline_manifold_path": spline_manifold_path,
            "additive_coefficients": additive_coefficients,
        },
        output_path,
    )

    append_html(
        output_path=output_path,
        summaries=summaries,
        plot_rows=plot_rows,
        pca_paths=pca_paths,
        section_suffix=args.output_suffix.replace("_", "-"),
    )

    print()
    for row in summaries:
        print(
            f"{row['label']:<24} "
            f"final P({target_day})={row['final_target_probability']:.4f} "
            f"argmax={row['final_target_argmax_rate']:.2%} "
            f"mean jump={row['mean_hellinger_jump']:.4f} "
            f"off-arc/other={row['mean_off_arc_or_other_mass']:.4f} "
            f"sequence={' -> '.join(row['dominant_sequence'])}"
        )
    print()
    print(f"Saved artifact: {output_path}")
    print(f"Updated report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
