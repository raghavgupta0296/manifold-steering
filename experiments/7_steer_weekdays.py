from pathlib import Path
import json
import sys

import torch

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from data.weekdays import WeekdayData
from model.llama import LlamaModel
from report_html import REPORT_PATH, append_or_replace_section


def build_centroids(
    features: torch.Tensor,
    examples: list[dict],
    days: list[str],
) -> dict[str, torch.Tensor]:
    centroids = {}
    for day in days:
        indices = [
            index
            for index, example in enumerate(examples)
            if example["answer"] == day
        ]
        centroids[day] = features[indices].mean(dim=0)
    return centroids


def make_steering_vectors(
    examples: list[dict],
    target_day: str,
    centroids: dict[str, torch.Tensor],
) -> torch.Tensor:
    vectors = []
    target_centroid = centroids[target_day]
    for example in examples:
        source_centroid = centroids[example["answer"]]
        vectors.append(target_centroid - source_centroid)
    return torch.stack(vectors)


def summarize_run(
    days: list[str],
    examples: list[dict],
    baseline_probabilities: torch.Tensor,
    steered_probabilities: torch.Tensor,
    target_day: str,
    coefficient: float,
) -> dict:
    target_index = days.index(target_day)
    weekday_probabilities = steered_probabilities[:, : len(days)]
    predicted_indices = weekday_probabilities.argmax(dim=-1).tolist()
    predicted_answers = [days[index] for index in predicted_indices]
    off_target_indices = [
        index
        for index, example in enumerate(examples)
        if example["answer"] != target_day
    ]

    target_probabilities = steered_probabilities[:, target_index]
    baseline_target_probabilities = baseline_probabilities[:, target_index]
    off_target_predictions = [
        predicted_answers[index] == target_day
        for index in off_target_indices
    ]

    return {
        "target_day": target_day,
        "coefficient": coefficient,
        "mean_target_probability": target_probabilities.mean().item(),
        "mean_target_probability_lift": (
            target_probabilities - baseline_target_probabilities
        ).mean().item(),
        "off_target_argmax_rate": (
            sum(off_target_predictions) / len(off_target_predictions)
            if off_target_predictions
            else 0.0
        ),
    }


def build_prompt_rows(
    days: list[str],
    examples: list[dict],
    baseline_probabilities: torch.Tensor,
    steered_probabilities: torch.Tensor,
    target_day: str,
    coefficient: float,
) -> list[dict]:
    rows = []
    target_index = days.index(target_day)
    weekday_probabilities = steered_probabilities[:, : len(days)]
    predicted_indices = weekday_probabilities.argmax(dim=-1).tolist()

    for index, (example, predicted_index) in enumerate(zip(examples, predicted_indices)):
        source_index = days.index(example["answer"])
        rows.append(
            {
                "raw_input": example["raw_input"],
                "source_answer": example["answer"],
                "target_day": target_day,
                "coefficient": coefficient,
                "predicted_answer": days[predicted_index],
                "baseline_source_probability": baseline_probabilities[
                    index,
                    source_index,
                ].item(),
                "baseline_target_probability": baseline_probabilities[
                    index,
                    target_index,
                ].item(),
                "steered_source_probability": steered_probabilities[
                    index,
                    source_index,
                ].item(),
                "steered_target_probability": steered_probabilities[
                    index,
                    target_index,
                ].item(),
            }
        )
    return rows


def append_steering_html(output_path: Path, summaries: list[dict]) -> None:
    coefficient_one = [
        row for row in summaries if abs(row["coefficient"] - 1.0) < 1e-9
    ]
    table_rows = "\n".join(
        f"""
        <tr>
          <td>{row["target_day"]}</td>
          <td>{row["mean_target_probability"]:.4f}</td>
          <td>{row["mean_target_probability_lift"]:+.4f}</td>
          <td>{row["off_target_argmax_rate"]:.2%}</td>
        </tr>
        """
        for row in coefficient_one
    )

    section = f"""
  <section id="experiment-7-weekday-steering">
    <h2>7. Weekday PCA Feature Steering</h2>
    <p>
      For each target weekday, added
      <code>target PCA centroid - source answer PCA centroid</code>
      in the cached activation PCA feature space, reconstructed the residual stream,
      and measured next-token weekday probabilities.
      Saved artifact: <code>{output_path}</code>
    </p>
    <table>
      <thead>
        <tr>
          <th>Target</th>
          <th>Mean P(target)</th>
          <th>Mean P(target) lift</th>
          <th>Off-target argmax rate</th>
        </tr>
      </thead>
      <tbody>{table_rows}</tbody>
    </table>
    <div id="weekdaySteeringLift" class="chart wide"></div>
  </section>

  <script>
    const steeringSummaries = {json.dumps(summaries)};
    const steeringTargets = [...new Set(steeringSummaries.map(r => r.target_day))];
    const steeringTraces = steeringTargets.map(target => {{
      const rows = steeringSummaries.filter(r => r.target_day === target);
      return {{
        type: "scatter",
        mode: "lines+markers",
        name: target,
        x: rows.map(r => r.coefficient),
        y: rows.map(r => r.mean_target_probability_lift),
        hovertemplate:
          "target=" + target + "<br>coefficient=%{{x}}<br>lift=%{{y:.4f}}<extra></extra>",
      }};
    }});
    Plotly.newPlot(
      "weekdaySteeringLift",
      steeringTraces,
      {{
        title: "Target Probability Lift by Steering Strength",
        paper_bgcolor: "#fffaf2",
        plot_bgcolor: "#f3eadf",
        font: {{ color: "#25211c" }},
        xaxis: {{ title: "steering coefficient", gridcolor: "#ded1c1" }},
        yaxis: {{ title: "mean P(target) lift", gridcolor: "#ded1c1" }},
        margin: {{ t: 50, l: 70, r: 30, b: 60 }},
      }}
    );
  </script>
"""
    append_or_replace_section("experiment-7-weekday-steering", section)


def main() -> None:
    source_path = project_root / "artifacts" / "weekdays_llama31_8b_layer28.pt"
    pca_path = project_root / "artifacts" / "weekdays_llama31_8b_layer28_pca32.pt"
    output_path = project_root / "artifacts" / "weekdays_pca_feature_steering.pt"

    artifact = torch.load(source_path, map_location="cpu")
    pca_artifact = torch.load(pca_path, map_location="cpu")
    data = WeekdayData()
    model = LlamaModel(artifact["model_name"]).load()

    days = artifact["days"]
    layer = artifact["layer"]
    examples = artifact["examples"]
    prompts = [example["raw_input"] for example in examples]
    baseline_probabilities = artifact["probabilities"].float()
    centroids = build_centroids(
        features=pca_artifact["features"].float(),
        examples=examples,
        days=days,
    )

    coefficients = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    summaries = []
    prompt_rows = []

    print(f"Model: {model.model_name}")
    print(f"Layer: {layer}")
    print(f"PCA features: {tuple(pca_artifact['features'].shape)}")
    print(f"Examples: {len(examples)}")
    print(f"Coefficients: {coefficients}")
    print()

    for target_day in days:
        steering_vectors = make_steering_vectors(examples, target_day, centroids)
        for coefficient in coefficients:
            steered_probabilities = (
                model.gather_pca_feature_steered_prediction_probabilities(
                    prompts=prompts,
                    answer_values=days,
                    layer=layer,
                    pca_mean=pca_artifact["mean"].float(),
                    pca_rotation=pca_artifact["rotation"].float(),
                    feature_steering_vectors=steering_vectors,
                    coefficient=coefficient,
                    token_position=artifact["token_position"],
                    output_prefix=data.output_prefix,
                    full_vocab_softmax=True,
                    include_other=True,
                    batch_size=8,
                )
            )
            summary = summarize_run(
                days=days,
                examples=examples,
                baseline_probabilities=baseline_probabilities,
                steered_probabilities=steered_probabilities,
                target_day=target_day,
                coefficient=coefficient,
            )
            summaries.append(summary)
            prompt_rows.extend(
                build_prompt_rows(
                    days=days,
                    examples=examples,
                    baseline_probabilities=baseline_probabilities,
                    steered_probabilities=steered_probabilities,
                    target_day=target_day,
                    coefficient=coefficient,
                )
            )
            print(
                f"{target_day:>9} coef={coefficient:.1f} "
                f"mean P(target)={summary['mean_target_probability']:.4f} "
                f"lift={summary['mean_target_probability_lift']:+.4f} "
                f"off-target argmax={summary['off_target_argmax_rate']:.2%}"
            )

    torch.save(
        {
            "source_path": str(source_path),
            "pca_path": str(pca_path),
            "model_name": model.model_name,
            "layer": layer,
            "token_position": artifact["token_position"],
            "steering_space": "activation_pca32",
            "days": days,
            "coefficients": coefficients,
            "summaries": summaries,
            "prompt_rows": prompt_rows,
        },
        output_path,
    )
    append_steering_html(output_path, summaries)

    print()
    print(f"Saved steering artifact: {output_path}")
    print(f"Updated report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
