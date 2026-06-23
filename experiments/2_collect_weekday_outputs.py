from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from data.weekdays import WeekdayData
from model.llama import LlamaModel
from report_html import LAYER, REPORT_PATH, SOURCE_ARTIFACT_PATH, append_or_replace_section


def main() -> None:
    data = WeekdayData()
    model = LlamaModel(LlamaModel.paper_model_name).load()

    layer = LAYER
    artifact_path = SOURCE_ARTIFACT_PATH
    examples = data.as_list()
    prompts = [example.raw_input for example in examples]

    outputs = model.gather_activations_and_prediction_probabilities(
        prompts=prompts,
        layers=[layer],
        answer_values=data.days,
        token_position=-1,
        output_prefix=data.output_prefix,
        full_vocab_softmax=True,
        include_other=True,
        batch_size=1,
    )

    activations = outputs["activations"][layer]
    probabilities = outputs["probabilities"]

    weekday_probabilities = probabilities[:, : len(data.days)]
    predicted_indices = weekday_probabilities.argmax(dim=-1).tolist()
    predicted_answers = [data.days[index] for index in predicted_indices]

    correct = [
        predicted_answer == example.answer
        for predicted_answer, example in zip(predicted_answers, examples)
    ]
    mean_correct_probability = sum(
        probabilities[index, data.days.index(example.answer)].item()
        for index, example in enumerate(examples)
    ) / len(examples)

    print(f"Model: {model.model_name}")
    print(f"Layer: {layer}")
    print(f"Examples: {len(examples)}")
    print(f"Activation shape: {tuple(activations.shape)}")
    print(f"Probability shape: {tuple(probabilities.shape)}")
    print(f"Accuracy: {sum(correct)}/{len(correct)} = {sum(correct) / len(correct):.2%}")
    print(f"Mean correct probability: {mean_correct_probability:.2%}")

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(
        {
            "model_name": model.model_name,
            "layer": layer,
            "token_position": -1,
            "days": data.days,
            "output_prefix": data.output_prefix,
            "examples": [
                {
                    "entity": example.entity,
                    "number_word": example.number_word,
                    "number_int": example.number_int,
                    "answer": example.answer,
                    "raw_input": example.raw_input,
                    "raw_output": example.raw_output,
                    "predicted_answer": predicted_answer,
                    "correct": is_correct,
                }
                for example, predicted_answer, is_correct in zip(
                    examples,
                    predicted_answers,
                    correct,
                )
            ],
            "activations": activations,
            "probabilities": probabilities,
            "accuracy": sum(correct) / len(correct),
            "mean_correct_probability": mean_correct_probability,
        },
        artifact_path,
    )
    print(f"Saved artifact: {artifact_path}")

    append_or_replace_section(
        "experiment-2-collection",
        f"""
  <section id="experiment-2-collection">
    <h2>2. Cached Activations and Probabilities</h2>
    <p>
      Model: <code>{model.model_name}</code><br>
      Layer: <code>{layer}</code>, token position: <code>last token</code><br>
      Activation tensor: <code>{tuple(activations.shape)}</code><br>
      Probability tensor: <code>{tuple(probabilities.shape)}</code><br>
      Accuracy sanity check: <code>{sum(correct)}/{len(correct)} = {sum(correct) / len(correct):.2%}</code><br>
      Mean correct probability: <code>{mean_correct_probability:.2%}</code><br>
      Saved artifact: <code>{artifact_path}</code>
    </p>
  </section>
""",
    )
    print(f"Updated report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
