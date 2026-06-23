from pathlib import Path

import torch

from report_html import (
    ACTIVATION_PCA_PATH,
    PROBABILITY_PCA_PATH,
    REPORT_PATH,
    SOURCE_ARTIFACT_PATH,
    append_or_replace_section,
)


project_root = Path(__file__).resolve().parents[1]


def fit_pca(features: torch.Tensor, k: int) -> dict[str, torch.Tensor]:
    mean = features.mean(dim=0, keepdim=True)
    centered = features - mean

    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    rotation = vh[:k].T
    pca_features = centered @ rotation

    variances = singular_values.pow(2)
    explained_variance_ratio = variances / variances.sum()

    return {
        "mean": mean.squeeze(0),
        "rotation": rotation,
        "features": pca_features,
        "singular_values": singular_values[:k],
        "explained_variance_ratio": explained_variance_ratio[:k],
    }


def print_pca_summary(name: str, result: dict[str, torch.Tensor]) -> None:
    explained = result["explained_variance_ratio"]

    print(name)
    print(f"  Rotation shape: {tuple(result['rotation'].shape)}")
    print(f"  Feature shape: {tuple(result['features'].shape)}")
    print(
        "  Explained variance:",
        [round(value.item(), 4) for value in explained],
    )
    print(
        "  Cumulative explained variance:",
        [round(value.item(), 4) for value in explained.cumsum(dim=0)],
    )
    print(f"  Total explained variance: {explained.sum().item():.2%}")
    print()


def pca_html_summary(name: str, result: dict[str, torch.Tensor]) -> str:
    explained = result["explained_variance_ratio"]
    cumulative = explained.cumsum(dim=0)
    return f"""
      <h3>{name}</h3>
      <p>
        Rotation shape: <code>{tuple(result['rotation'].shape)}</code><br>
        Feature shape: <code>{tuple(result['features'].shape)}</code><br>
        Explained variance: <code>{[round(value.item(), 4) for value in explained]}</code><br>
        Cumulative explained variance: <code>{[round(value.item(), 4) for value in cumulative]}</code><br>
        Total explained variance: <code>{explained.sum().item():.2%}</code>
      </p>
    """


def main() -> None:
    input_path = SOURCE_ARTIFACT_PATH
    activation_output_path = ACTIVATION_PCA_PATH
    probability_output_path = PROBABILITY_PCA_PATH

    artifact = torch.load(input_path, map_location="cpu")

    activation_k = 32
    activations = artifact["activations"].float()
    activation_pca = fit_pca(activations, activation_k)
    torch.save(
        {
            "source_path": str(input_path),
            "model_name": artifact["model_name"],
            "layer": artifact["layer"],
            "token_position": artifact["token_position"],
            "k": activation_k,
            "mean": activation_pca["mean"],
            "rotation": activation_pca["rotation"],
            "features": activation_pca["features"],
            "singular_values": activation_pca["singular_values"],
            "explained_variance_ratio": activation_pca["explained_variance_ratio"],
            "examples": artifact["examples"],
            "days": artifact["days"],
        },
        activation_output_path,
    )

    probability_k = 3
    labels = artifact["days"] + ["other"]
    probabilities = artifact["probabilities"].float()
    hellinger = torch.sqrt(probabilities.clamp(min=0.0))
    probability_pca = fit_pca(hellinger, probability_k)

    probability_centroids = []
    hellinger_centroids = []
    for day in artifact["days"]:
        indices = [
            index
            for index, example in enumerate(artifact["examples"])
            if example["answer"] == day
        ]
        mean_probability = probabilities[indices].mean(dim=0)
        probability_centroids.append(mean_probability)
        hellinger_centroid = (
            torch.sqrt(mean_probability.clamp(min=0.0)) - probability_pca["mean"]
        ) @ probability_pca["rotation"]
        hellinger_centroids.append(hellinger_centroid)

    torch.save(
        {
            "source_path": str(input_path),
            "model_name": artifact["model_name"],
            "labels": labels,
            "days": artifact["days"],
            "probabilities": probabilities,
            "hellinger_features": hellinger,
            "mean": probability_pca["mean"],
            "rotation": probability_pca["rotation"],
            "features": probability_pca["features"],
            "probability_centroids": torch.stack(probability_centroids),
            "centroids": torch.stack(hellinger_centroids),
            "explained_variance_ratio": probability_pca["explained_variance_ratio"],
            "examples": artifact["examples"],
        },
        probability_output_path,
    )

    print(f"Loaded source artifact: {input_path}")
    print(f"Loaded activations: {tuple(activations.shape)}")
    print(f"Loaded probabilities: {tuple(probabilities.shape)}")
    print(f"Saved activation PCA artifact: {activation_output_path}")
    print(f"Saved probability PCA artifact: {probability_output_path}")
    print()
    print_pca_summary("Activation PCA", activation_pca)
    print_pca_summary("Probability Hellinger PCA", probability_pca)

    append_or_replace_section(
        "experiment-3-pca",
        f"""
  <section id="experiment-3-pca">
    <h2>3. PCA Spaces</h2>
    <p>
      Source artifact: <code>{input_path}</code><br>
      Activation PCA artifact: <code>{activation_output_path}</code><br>
      Probability PCA artifact: <code>{probability_output_path}</code>
    </p>
    {pca_html_summary("Activation PCA", activation_pca)}
    {pca_html_summary("Probability Hellinger PCA", probability_pca)}
  </section>
""",
    )
    print(f"Updated report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
